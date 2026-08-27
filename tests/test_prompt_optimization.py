from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from scripts import optimize_prompts as optimization_cli

from literature_multiverse.config import load_question_config
from literature_multiverse.extract import extraction_prompt_replacements
from literature_multiverse.lineage import (
    MissingArtifactError,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.prompt_optimization import (
    GEPAPromptAdapter,
    GEPAUnavailable,
    OptimizationContractError,
    OptimizationExample,
    OptimizationSplitManifest,
    SplitArtifact,
    compare_frozen_test_to_seed,
    create_split_bundle,
    evaluate_frozen_test,
    load_manifest_split,
    load_optimization_examples,
    load_split_manifest,
    optimize_prompts,
    provider_request_key,
)
from literature_multiverse.providers import FixtureProvider, ProviderBudgetExceeded
from literature_multiverse.verification import verification_output_schema

EXTRACTION_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "eligible": {"type": "boolean"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["increase", "no_effect", "decrease", "mixed", "unclear"],
                    },
                    "evidence_quote": {"type": ["string", "null"]},
                    "evidence_lines": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                    },
                },
                "required": ["direction", "evidence_quote", "evidence_lines"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["eligible", "findings"],
    "additionalProperties": False,
}


def extraction_example(
    example_id: str,
    *,
    paper_id: str | None = None,
    group_id: str | None = None,
) -> OptimizationExample:
    return OptimizationExample(
        example_id=example_id,
        paper_id=paper_id or f"paper-{example_id}",
        group_id=group_id or f"group-{example_id}",
        prompt_kind="extraction",
        replacements={"SOURCE_TEXT": f"source for {example_id}"},
        expected_output={
            "eligible": True,
            "findings": [
                {
                    "direction": "increase",
                    "evidence_quote": "Training increased VO2max.",
                    "evidence_lines": ["L1"],
                }
            ],
        },
        label_paths=["/eligible", "/findings/0/direction"],
        output_schema=EXTRACTION_SCHEMA,
        content_lines={
            "1": {"text": "Training increased VO2max.", "section": "Results"}
        },
    )


def verification_example(example_id: str) -> OptimizationExample:
    return OptimizationExample(
        example_id=example_id,
        paper_id=f"paper-{example_id}",
        group_id=f"group-{example_id}",
        prompt_kind="quote_verification",
        replacements={
            "DIRECTION_DEFINITIONS_JSON": json.dumps(
                {
                    "increase": "higher than comparator",
                    "no_effect": "no supported difference",
                    "decrease": "lower than comparator",
                },
                sort_keys=True,
            ),
            "FINDINGS_JSON": json.dumps(
                [
                    {
                        "finding_id": "f1",
                        "direction": "increase",
                        "evidence_quote": "Training increased VO2max.",
                    }
                ],
                sort_keys=True,
            ),
        },
        expected_output={
            "decisions": [
                {
                    "finding_id": "f1",
                    "model_status": "agree",
                    "rationale": "The quoted result reports an increase.",
                }
            ]
        },
        label_paths=["/decisions/0/finding_id", "/decisions/0/model_status"],
        output_schema=verification_output_schema(["f1"]),
    )


def _frozen_extraction_winner(
    *,
    directory: Path,
    manifest_path: Path,
    winner_text: str,
    seed_text: str,
) -> Path:
    directory.mkdir()
    prompt_path = directory / "frozen_extraction.md"
    atomic_write_text(prompt_path, winner_text)
    winner_path = directory / "frozen_winner.json"
    atomic_write_json(
        winner_path,
        {
            "frozen_prompt_bundle_version": "1",
            "manifest_sha256": sha256_file(manifest_path),
            "test_evaluated_at_freeze": False,
            "seed_prompt_sha256s": {
                "extraction": hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
            },
            "prompts": {
                "extraction": {
                    "path": prompt_path.name,
                    "sha256": hashlib.sha256(
                        winner_text.encode("utf-8")
                    ).hexdigest(),
                }
            },
        },
    )
    return winner_path


def test_fixture_provider_evaluation_is_deterministic_grounded_and_cached() -> None:
    example = extraction_example("ex-1")
    candidate = {
        "extraction_prompt": "Prompt version: `gepa-fixture-v1`\nPaper:\n[[SOURCE_TEXT]]\n"
    }
    expected_labels_before = deepcopy(example.expected_output)
    key = provider_request_key(example, candidate)
    provider = FixtureProvider(
        {("gepa-extraction", key): example.model_dump(mode="json")["expected_output"]}
    )
    adapter = GEPAPromptAdapter(provider, cost_cap_usd=0.01)

    first = adapter.evaluate([example], candidate, capture_traces=True)
    second = adapter.evaluate([example], candidate, capture_traces=False)

    assert first.scores == [1.0]
    assert first.objective_scores == [
        {
            "extraction_correctness": 1.0,
            "grounding_schema_validity": 1.0,
            "cost_efficiency": 1.0,
        }
    ]
    assert first.trajectories is not None
    assert first.trajectories[0]["grounding_results"][0]["grounding_status"] == "exact"
    assert first.num_metric_calls == 1
    assert second.scores == first.scores
    assert second.num_metric_calls == 0
    assert provider.calls == [("gepa-extraction", key)]
    assert example.expected_output == expected_labels_before


def test_adapter_renders_existing_extraction_template_and_attaches_paper(
    repo_root: Path,
) -> None:
    config = load_question_config(
        repo_root / "configs/questions/fixture-a.yaml", require_locked=True
    )
    example = extraction_example("existing-template").model_copy(
        update={"replacements": extraction_prompt_replacements(config)}
    )
    candidate = {
        "extraction_prompt": (repo_root / "prompts/extraction.md").read_text(encoding="utf-8")
    }
    key = provider_request_key(example, candidate)
    provider = FixtureProvider(
        {("gepa-extraction", key): example.model_dump(mode="json")["expected_output"]}
    )

    result = GEPAPromptAdapter(provider).evaluate(
        [example], candidate, capture_traces=True
    )

    assert result.scores == [1.0]
    assert result.trajectories is not None
    trajectory = result.trajectories[0]
    assert trajectory["prompt_version"] == "extraction-v3"
    assert trajectory["request_prompt_sha256"] != trajectory["rendered_prompt_sha256"]


def test_adapter_renders_existing_quote_verification_template(repo_root: Path) -> None:
    example = verification_example("verification-template")
    candidate = {
        "quote_verification_prompt": (
            repo_root / "prompts/quote_verification.md"
        ).read_text(encoding="utf-8")
    }
    key = provider_request_key(example, candidate)
    provider = FixtureProvider(
        {
            ("gepa-quote_verification", key): example.model_dump(mode="json")[
                "expected_output"
            ]
        }
    )

    result = GEPAPromptAdapter(provider).evaluate(
        [example], candidate, capture_traces=True
    )

    assert result.scores == [1.0]
    assert result.trajectories is not None
    trajectory = result.trajectories[0]
    assert trajectory["prompt_version"] == "quote-verification-v1"
    assert trajectory["request_prompt_sha256"] == trajectory["rendered_prompt_sha256"]


def test_invalid_candidate_prompt_scores_zero_without_provider_call() -> None:
    example = extraction_example("ex-2")
    provider = FixtureProvider({})
    adapter = GEPAPromptAdapter(provider)

    result = adapter.evaluate(
        [example],
        {"extraction_prompt": "Prompt version: `bad-v1`\n[[WRONG_TOKEN]]"},
        capture_traces=True,
    )

    assert result.scores == [0.0]
    assert result.objective_scores == [
        {
            "extraction_correctness": 0.0,
            "grounding_schema_validity": 0.0,
            "cost_efficiency": 0.0,
        }
    ]
    assert provider.calls == []
    assert result.trajectories is not None
    assert "prompt contract" in result.trajectories[0]["error"]


def test_reflective_dataset_contains_prespecified_labels_and_diagnostics() -> None:
    example = extraction_example("ex-3")
    candidate = {
        "extraction_prompt": "Prompt version: `gepa-fixture-v1`\n[[SOURCE_TEXT]]\n"
    }
    key = provider_request_key(example, candidate)
    wrong = deepcopy(example.expected_output)
    wrong["findings"][0]["direction"] = "decrease"
    provider = FixtureProvider({("gepa-extraction", key): wrong})
    adapter = GEPAPromptAdapter(provider)
    evaluated = adapter.evaluate([example], candidate, capture_traces=True)

    reflective = adapter.make_reflective_dataset(
        candidate, evaluated, ["extraction_prompt"]
    )

    assert list(reflective) == ["extraction_prompt"]
    feedback = json.loads(reflective["extraction_prompt"][0]["Feedback"])
    assert feedback["expected_labels"]["/findings/0/direction"] == "increase"
    assert feedback["predicted_labels"]["/findings/0/direction"] == "decrease"
    assert feedback["objective_scores"]["extraction_correctness"] == 0.5


def test_split_bundle_keeps_linked_papers_and_groups_disjoint(tmp_path: Path) -> None:
    examples = [
        extraction_example("a", paper_id="paper-shared", group_id="group-a"),
        extraction_example("b", paper_id="paper-shared", group_id="group-bridge"),
        extraction_example("c", paper_id="paper-c", group_id="group-bridge"),
        extraction_example("d"),
        extraction_example("e"),
        extraction_example("f"),
        extraction_example("g"),
    ]
    source = tmp_path / "examples.jsonl"
    atomic_write_jsonl(source, [example.model_dump(mode="json") for example in examples])

    manifest_path = create_split_bundle(source, tmp_path / "splits", seed=1729)
    manifest = load_split_manifest(manifest_path)

    split_pairs = (
        (manifest.train, manifest.dev),
        (manifest.train, manifest.test),
        (manifest.dev, manifest.test),
    )
    for left, right in split_pairs:
        assert set(left.paper_ids).isdisjoint(right.paper_ids)
        assert set(left.group_ids).isdisjoint(right.group_ids)
        assert set(left.example_ids).isdisjoint(right.example_ids)
    linked = {"a", "b", "c"}
    assert any(
        linked <= set(split.example_ids)
        for split in (manifest.train, manifest.dev, manifest.test)
    )


def test_optimizer_loader_never_opens_test_split(tmp_path: Path) -> None:
    examples = [extraction_example(character) for character in "abcdef"]
    source = tmp_path / "examples.jsonl"
    atomic_write_jsonl(source, [example.model_dump(mode="json") for example in examples])
    manifest_path = create_split_bundle(source, tmp_path / "splits", seed=7)
    manifest = load_split_manifest(manifest_path)
    test_path = manifest_path.parent / manifest.test.path
    test_path.rename(test_path.with_suffix(".withheld"))

    train, dev = load_optimization_examples(manifest_path)

    assert train
    assert dev
    with pytest.raises(MissingArtifactError):
        load_manifest_split(manifest_path, "test")


def test_single_arm_test_evaluation_checks_guards_before_opening_test(
    tmp_path: Path,
) -> None:
    examples = [extraction_example(character) for character in "abcdef"]
    source = tmp_path / "examples.jsonl"
    atomic_write_jsonl(source, [example.model_dump(mode="json") for example in examples])
    manifest_path = create_split_bundle(source, tmp_path / "splits", seed=7)
    manifest = load_split_manifest(manifest_path)
    test_path = manifest_path.parent / manifest.test.path
    test_path.rename(test_path.with_suffix(".withheld"))
    seed_text = "Prompt version: `seed-v1`\n[[SOURCE_TEXT]]\n"
    winner_path = _frozen_extraction_winner(
        directory=tmp_path / "winner",
        manifest_path=manifest_path,
        winner_text=seed_text,
        seed_text=seed_text,
    )
    provider = FixtureProvider({})

    occupied = tmp_path / "occupied.json"
    atomic_write_text(occupied, "already evaluated")
    with pytest.raises(OptimizationContractError, match="report already exists"):
        evaluate_frozen_test(
            manifest_path=manifest_path,
            winner_path=winner_path,
            provider=provider,
            output_path=occupied,
        )

    winner = json.loads(winner_path.read_text(encoding="utf-8"))
    winner["test_evaluated_at_freeze"] = True
    winner_path.write_text(json.dumps(winner), encoding="utf-8")
    with pytest.raises(OptimizationContractError, match="test labels were unopened"):
        evaluate_frozen_test(
            manifest_path=manifest_path,
            winner_path=winner_path,
            provider=provider,
            output_path=tmp_path / "new-report.json",
        )
    assert provider.calls == []


def test_split_loader_rejects_hash_drift(tmp_path: Path) -> None:
    examples = [extraction_example(character) for character in "abcdef"]
    source = tmp_path / "examples.jsonl"
    atomic_write_jsonl(source, [example.model_dump(mode="json") for example in examples])
    manifest_path = create_split_bundle(source, tmp_path / "splits", seed=11)
    manifest = load_split_manifest(manifest_path)
    train_path = manifest_path.parent / manifest.train.path
    train_path.write_text(train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(OptimizationContractError, match="hash mismatch"):
        load_manifest_split(manifest_path, "train")


def test_paired_heldout_report_scores_same_examples_and_reports_deltas(
    tmp_path: Path,
) -> None:
    examples = [extraction_example(f"paired-{index}") for index in range(12)]
    source = tmp_path / "examples.jsonl"
    atomic_write_jsonl(source, [example.model_dump(mode="json") for example in examples])
    manifest_path = create_split_bundle(source, tmp_path / "splits", seed=41)
    test_examples = load_manifest_split(manifest_path, "test")
    winner_text = "Prompt version: `winner-v1`\n[[SOURCE_TEXT]]\n"
    seed_text = "Prompt version: `seed-v1`\n[[SOURCE_TEXT]]\n"
    seed_path = tmp_path / "seed.md"
    atomic_write_text(seed_path, seed_text)
    winner_path = _frozen_extraction_winner(
        directory=tmp_path / "winner",
        manifest_path=manifest_path,
        winner_text=winner_text,
        seed_text=seed_text,
    )
    winner_candidate = {"extraction_prompt": winner_text}
    seed_candidate = {"extraction_prompt": seed_text}
    responses: dict[tuple[str, str], dict[str, object]] = {}
    expected_outcomes: list[str] = []
    for index, example in enumerate(test_examples):
        expected = deepcopy(example.expected_output)
        wrong = deepcopy(expected)
        wrong["findings"][0]["direction"] = "decrease"
        pattern = index % 3
        if pattern == 0:
            winner_response, seed_response, outcome = expected, wrong, "win"
        elif pattern == 1:
            winner_response, seed_response, outcome = expected, expected, "tie"
        else:
            winner_response, seed_response, outcome = wrong, expected, "loss"
        expected_outcomes.append(outcome)
        responses[
            (
                "gepa-extraction",
                provider_request_key(
                    example,
                    winner_candidate,
                    request_namespace="heldout-winner",
                ),
            )
        ] = winner_response
        responses[
            (
                "gepa-extraction",
                provider_request_key(
                    example,
                    seed_candidate,
                    request_namespace="heldout-seed",
                ),
            )
        ] = seed_response
    provider = FixtureProvider(responses)
    report_path = tmp_path / "paired-report.json"

    compare_frozen_test_to_seed(
        manifest_path=manifest_path,
        winner_path=winner_path,
        seed_templates={"extraction": seed_path},
        provider=provider,
        output_path=report_path,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["by_prompt_kind"]["extraction"]["per_example"]
    assert [row["example_id"] for row in rows] == [
        example.example_id for example in test_examples
    ]
    assert [row["winner_outcome"] for row in rows] == expected_outcomes
    assert all("objective_scores" in row["winner"] for row in rows)
    assert all("objective_scores" in row["seed"] for row in rows)
    assert all("output" in row["winner"] and "output" in row["seed"] for row in rows)
    observed_delta = sum(
        row["paired_delta_winner_minus_seed"]["scalar_score"] for row in rows
    ) / len(rows)
    assert report["overall"]["scalar_score"][
        "mean_paired_delta_winner_minus_seed"
    ] == pytest.approx(observed_delta)
    assert report["overall"]["winner_win_tie_loss"] == {
        outcome: expected_outcomes.count(outcome) for outcome in ("win", "tie", "loss")
    }
    assert report["manifest_sha256"] == sha256_file(manifest_path)
    assert report["winner_prompt_sha256s"]["extraction"] == hashlib.sha256(
        winner_text.encode("utf-8")
    ).hexdigest()
    assert report["seed_templates"]["extraction"]["sha256"] == hashlib.sha256(
        seed_text.encode("utf-8")
    ).hexdigest()
    assert len(provider.calls) == 2 * len(test_examples)
    assert len(set(provider.calls)) == len(provider.calls)


def test_paired_arms_have_distinct_request_keys_when_prompts_are_identical(
    tmp_path: Path,
) -> None:
    examples = [extraction_example(f"same-{index}") for index in range(9)]
    source = tmp_path / "examples.jsonl"
    atomic_write_jsonl(source, [example.model_dump(mode="json") for example in examples])
    manifest_path = create_split_bundle(source, tmp_path / "splits", seed=43)
    test_examples = load_manifest_split(manifest_path, "test")
    prompt_text = "Prompt version: `same-v1`\n[[SOURCE_TEXT]]\n"
    seed_path = tmp_path / "seed.md"
    atomic_write_text(seed_path, prompt_text)
    winner_path = _frozen_extraction_winner(
        directory=tmp_path / "winner",
        manifest_path=manifest_path,
        winner_text=prompt_text,
        seed_text=prompt_text,
    )
    candidate = {"extraction_prompt": prompt_text}
    responses = {
        ("gepa-extraction", key): example.expected_output
        for example in test_examples
        for key in (
            provider_request_key(
                example, candidate, request_namespace="heldout-winner"
            ),
            provider_request_key(example, candidate, request_namespace="heldout-seed"),
        )
    }
    provider = FixtureProvider(responses)

    output = tmp_path / "paired.json"
    compare_frozen_test_to_seed(
        manifest_path=manifest_path,
        winner_path=winner_path,
        seed_templates={"extraction": seed_path},
        provider=provider,
        output_path=output,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(provider.calls) == 2 * len(test_examples)
    assert len(set(provider.calls)) == len(provider.calls)
    assert report["overall"]["winner_win_tie_loss"] == {
        "win": 0,
        "tie": len(test_examples),
        "loss": 0,
    }
    request_keys = [key for _operation, key in provider.calls]
    assert any(key.startswith("heldout-winner-") for key in request_keys)
    assert any(key.startswith("heldout-seed-") for key in request_keys)


def test_paired_comparison_validates_frozen_seed_before_opening_test(
    tmp_path: Path,
) -> None:
    examples = [extraction_example(f"guard-{index}") for index in range(8)]
    source = tmp_path / "examples.jsonl"
    atomic_write_jsonl(source, [example.model_dump(mode="json") for example in examples])
    manifest_path = create_split_bundle(source, tmp_path / "splits", seed=47)
    manifest = load_split_manifest(manifest_path)
    test_path = manifest_path.parent / manifest.test.path
    test_path.rename(test_path.with_suffix(".withheld"))
    original_seed = "Prompt version: `seed-original-v1`\n[[SOURCE_TEXT]]\n"
    wrong_seed_path = tmp_path / "wrong-seed.md"
    atomic_write_text(
        wrong_seed_path,
        "Prompt version: `seed-wrong-v1`\n[[SOURCE_TEXT]]\n",
    )
    winner_path = _frozen_extraction_winner(
        directory=tmp_path / "winner",
        manifest_path=manifest_path,
        winner_text="Prompt version: `winner-v1`\n[[SOURCE_TEXT]]\n",
        seed_text=original_seed,
    )

    with pytest.raises(OptimizationContractError, match="seed template hash"):
        compare_frozen_test_to_seed(
            manifest_path=manifest_path,
            winner_path=winner_path,
            seed_templates={"extraction": wrong_seed_path},
            provider=FixtureProvider({}),
            output_path=tmp_path / "report.json",
        )


def test_missing_official_gepa_has_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import literature_multiverse.prompt_optimization as module

    def missing(_name: str):
        raise ImportError("not installed")

    monkeypatch.setattr(module.importlib, "import_module", missing)
    with pytest.raises(GEPAUnavailable, match="official GEPA is not installed"):
        module._load_official_gepa_optimize()


def test_official_reflection_lm_is_instantiated_once_and_exposes_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import literature_multiverse.prompt_optimization as module

    class FakeLM:
        constructions = 0

        def __init__(self, model: str, **kwargs: object) -> None:
            type(self).constructions += 1
            self.model = model
            self.kwargs = kwargs
            self.total_cost = 0.125
            self.total_tokens_in = 101
            self.total_tokens_out = 19

    class FakeTrackingLM:
        pass

    monkeypatch.setattr(
        module,
        "_load_official_gepa_lm_types",
        lambda: (FakeLM, FakeTrackingLM),
    )
    lm = module._prepare_official_reflection_lm(
        "anthropic/fixture",
        {"max_tokens": 1200, "num_retries": 0, "temperature": None},
    )

    assert FakeLM.constructions == 1
    assert lm.model == "anthropic/fixture"
    assert lm.kwargs == {
        "max_tokens": 1200,
        "num_retries": 0,
        "temperature": None,
    }
    usage = module._reflection_lm_usage(lm)
    assert usage == {
        "reflection_lm_usage_version": "1",
        "status": "available",
        "total_cost_usd": 0.125,
        "total_input_tokens": 101,
        "total_output_tokens": 19,
        "tracker": usage["tracker"],
    }
    assert usage["tracker"].endswith(".<locals>.FakeLM")


def test_tracked_callable_reflection_usage_is_windowed_per_prompt_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import literature_multiverse.prompt_optimization as module

    class UnusedLM:
        pass

    class UnusedTrackingLM:
        pass

    class SharedTrackedCallable:
        total_cost = 0.0
        total_tokens_in = 0
        total_tokens_out = 0

        def __call__(self, prompt: object) -> str:
            return str(prompt)

    monkeypatch.setattr(
        module,
        "_load_official_gepa_lm_types",
        lambda: (UnusedLM, UnusedTrackingLM),
    )
    shared = SharedTrackedCallable()
    first = module._prepare_official_reflection_lm(shared, None)
    shared.total_cost = 0.25
    shared.total_tokens_in = 100
    shared.total_tokens_out = 10
    first_usage = module._reflection_lm_usage(first)
    second = module._prepare_official_reflection_lm(shared, None)
    shared.total_cost = 0.75
    shared.total_tokens_in = 300
    shared.total_tokens_out = 30
    second_usage = module._reflection_lm_usage(second)

    aggregate = module._aggregate_reflection_lm_usage(
        {"extraction": first_usage, "quote_verification": second_usage}
    )
    assert first_usage["total_cost_usd"] == 0.25
    assert second_usage["total_cost_usd"] == 0.5
    assert aggregate["total_cost_usd"] == 0.75
    assert aggregate["total_input_tokens"] == 300
    assert aggregate["total_output_tokens"] == 30


def _write_two_kind_manifest(directory: Path) -> Path:
    split_examples = {
        split: [
            extraction_example(f"{split}-extraction"),
            verification_example(f"{split}-verification"),
        ]
        for split in ("train", "dev", "test")
    }
    artifacts: dict[str, SplitArtifact] = {}
    for split, examples in split_examples.items():
        path = directory / f"{split}.jsonl"
        atomic_write_jsonl(path, [example.model_dump(mode="json") for example in examples])
        artifacts[split] = SplitArtifact(
            path=path.name,
            sha256=sha256_file(path),
            rows=len(examples),
            example_ids=sorted(example.example_id for example in examples),
            paper_ids=sorted(example.paper_id for example in examples),
            group_ids=sorted(example.group_id for example in examples),
        )
    manifest = OptimizationSplitManifest(
        seed=0,
        train_fraction=0.34,
        dev_fraction=0.33,
        source_examples_sha256="f" * 64,
        train=artifacts["train"],
        dev=artifacts["dev"],
        test=artifacts["test"],
    )
    manifest_path = directory / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def test_optimize_cli_preflight_reserves_task_and_per_kind_reflection_budgets(
    tmp_path: Path,
) -> None:
    manifest_dir = tmp_path / "splits"
    manifest_dir.mkdir()
    manifest_path = _write_two_kind_manifest(manifest_dir)
    archive_root = tmp_path / "archives"
    atomic_write_json(
        archive_root / "old.provider.json",
        {"estimated_cost_usd": 2.0},
    )
    args = optimization_cli._parser().parse_args(
        [
            "optimize",
            "--manifest",
            str(manifest_path),
            "--run-dir",
            str(tmp_path / "run"),
            "--seed",
            "19",
            "--reflection-lm",
            "anthropic/reflection",
            "--max-budget-usd",
            "3",
            "--max-reflection-cost-usd-per-prompt",
            "1",
            "--reflection-batch-headroom-usd-per-prompt",
            "0.5",
        ]
    )

    preflight = optimization_cli._optimization_budget_preflight(
        args, archive_root=archive_root
    )

    assert preflight == {
        "planning_ceiling_usd": 50.0,
        "existing_archived_provider_ceiling_usd": 2.0,
        "task_rollout_ceiling_usd": 3.0,
        "active_prompt_kinds": 2,
        "reflection_stop_ceiling_usd": 2.0,
        "reflection_batch_headroom_usd": 1.0,
        "projected_combined_ceiling_usd": 8.0,
        "task_provider_global_limit_usd": 47.0,
    }

    args.max_budget_usd = 46.0
    with pytest.raises(ProviderBudgetExceeded, match="combined optimization preflight"):
        optimization_cli._optimization_budget_preflight(
            args, archive_root=archive_root
        )


def test_two_prompt_kinds_get_isolated_reflection_lms_and_aggregate_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import literature_multiverse.prompt_optimization as module

    manifest_dir = tmp_path / "splits"
    manifest_dir.mkdir()
    manifest_path = _write_two_kind_manifest(manifest_dir)
    extraction_template = tmp_path / "extraction.md"
    verification_template = tmp_path / "verification.md"
    atomic_write_text(
        extraction_template,
        "Prompt version: `two-kind-extraction-v1`\nExtract:\n[[SOURCE_TEXT]]\n",
    )
    atomic_write_text(
        verification_template,
        "Prompt version: `two-kind-verification-v1`\n"
        "Definitions: [[DIRECTION_DEFINITIONS_JSON]]\n"
        "Findings: [[FINDINGS_JSON]]\n",
    )

    class FakeLM:
        instances: ClassVar[list[FakeLM]] = []

        def __init__(self, model: str, **kwargs: object) -> None:
            self.model = model
            self.kwargs = kwargs
            self.total_cost = 0.0
            self.total_tokens_in = 0
            self.total_tokens_out = 0
            type(self).instances.append(self)

    class FakeTrackingLM:
        pass

    optimizer_calls: list[dict[str, object]] = []

    def fake_optimize(**kwargs: object) -> SimpleNamespace:
        optimizer_calls.append(kwargs)
        candidate = kwargs["seed_candidate"]
        assert isinstance(candidate, dict)
        lm = kwargs["reflection_lm"]
        assert isinstance(lm, FakeLM)
        assert lm.total_cost == 0.0
        assert kwargs["reflection_lm_kwargs"] is None
        component = next(iter(candidate))
        if component == "extraction_prompt":
            lm.total_cost = 0.25
            lm.total_tokens_in = 100
            lm.total_tokens_out = 10
        else:
            lm.total_cost = 0.5
            lm.total_tokens_in = 200
            lm.total_tokens_out = 20
        return SimpleNamespace(
            best_candidate=candidate,
            candidates=[candidate],
            parents=[[None]],
            val_aggregate_scores=[1.0],
            best_idx=0,
            val_aggregate_subscores=None,
            per_objective_best_candidates=None,
            objective_pareto_front=None,
            total_metric_calls=1,
            seed=kwargs["seed"],
        )

    monkeypatch.setattr(
        module, "_load_official_gepa_optimize", lambda: (fake_optimize, "fixture-gepa")
    )
    monkeypatch.setattr(
        module,
        "_load_official_gepa_lm_types",
        lambda: (FakeLM, FakeTrackingLM),
    )
    templates = {
        "extraction": extraction_template,
        "quote_verification": verification_template,
    }
    task_identity = {
        "provider": "fixture",
        "model": "task-model-a",
        "effort": "low",
        "max_tokens": 1200,
        "max_budget_usd": 3.0,
    }

    run = optimize_prompts(
        manifest_path=manifest_path,
        seed_templates=templates,
        provider=FixtureProvider({}),
        run_dir=tmp_path / "run-a",
        reflection_lm="anthropic/reflection-a",
        reflection_lm_kwargs={"max_tokens": 100, "num_retries": 0},
        task_provider_identity=task_identity,
        max_metric_calls_per_prompt=1,
        max_reflection_cost_usd_per_prompt=1.0,
        reflection_minibatch_size=1,
        seed=23,
    )
    trace = json.loads(run.trace_path.read_text(encoding="utf-8"))

    assert len(FakeLM.instances) == 2
    assert FakeLM.instances[0] is not FakeLM.instances[1]
    assert [call["max_reflection_cost"] for call in optimizer_calls] == [1.0, 1.0]
    assert trace["reflection_lm_usage"]["total_cost_usd"] == 0.75
    assert trace["reflection_lm_usage"]["total_input_tokens"] == 300
    assert trace["reflection_lm_usage"]["total_output_tokens"] == 30
    assert trace["reflection_lm_usage"]["by_prompt_kind"]["extraction"][
        "total_cost_usd"
    ] == 0.25
    assert trace["reflection_lm_usage"]["by_prompt_kind"]["quote_verification"][
        "total_cost_usd"
    ] == 0.5
    assert trace["reflection_lm_identity"] == {
        "kind": "model",
        "model": "anthropic/reflection-a",
    }
    assert trace["task_provider_identity_sha256"] == hash_canonical(
        trace["task_provider_identity"]
    )

    other_reflection = optimize_prompts(
        manifest_path=manifest_path,
        seed_templates=templates,
        provider=FixtureProvider({}),
        run_dir=tmp_path / "run-b",
        reflection_lm="anthropic/reflection-b",
        reflection_lm_kwargs={"max_tokens": 100, "num_retries": 0},
        task_provider_identity=task_identity,
        max_metric_calls_per_prompt=1,
        max_reflection_cost_usd_per_prompt=1.0,
        reflection_minibatch_size=1,
        seed=23,
    )
    changed_task_identity = {**task_identity, "model": "task-model-b"}
    other_task = optimize_prompts(
        manifest_path=manifest_path,
        seed_templates=templates,
        provider=FixtureProvider({}),
        run_dir=tmp_path / "run-c",
        reflection_lm="anthropic/reflection-a",
        reflection_lm_kwargs={"max_tokens": 100, "num_retries": 0},
        task_provider_identity=changed_task_identity,
        max_metric_calls_per_prompt=1,
        max_reflection_cost_usd_per_prompt=1.0,
        reflection_minibatch_size=1,
        seed=23,
    )
    run_ids = {
        json.loads(path.trace_path.read_text(encoding="utf-8"))["run_id"]
        for path in (run, other_reflection, other_task)
    }
    assert len(run_ids) == 3


def test_injected_optimizer_keeps_original_callable_identity(tmp_path: Path) -> None:
    import literature_multiverse.prompt_optimization as module

    manifest_dir = tmp_path / "splits"
    manifest_dir.mkdir()
    manifest_path = _write_two_kind_manifest(manifest_dir)
    extraction_template = tmp_path / "extraction.md"
    verification_template = tmp_path / "verification.md"
    atomic_write_text(
        extraction_template,
        "Prompt version: `injected-extraction-v1`\nExtract:\n[[SOURCE_TEXT]]\n",
    )
    atomic_write_text(
        verification_template,
        "Prompt version: `injected-verification-v1`\n"
        "Definitions: [[DIRECTION_DEFINITIONS_JSON]]\n"
        "Findings: [[FINDINGS_JSON]]\n",
    )
    reflection = lambda prompt: str(prompt)  # noqa: E731
    observed_lms: list[object] = []

    def injected_optimize(**kwargs: object) -> SimpleNamespace:
        observed_lms.append(kwargs["reflection_lm"])
        candidate = kwargs["seed_candidate"]
        return SimpleNamespace(
            best_candidate=candidate,
            candidates=[candidate],
            parents=[[None]],
            val_aggregate_scores=[1.0],
            best_idx=0,
            total_metric_calls=1,
            seed=kwargs["seed"],
        )

    run = optimize_prompts(
        manifest_path=manifest_path,
        seed_templates={
            "extraction": extraction_template,
            "quote_verification": verification_template,
        },
        provider=FixtureProvider({}),
        run_dir=tmp_path / "run",
        reflection_lm=reflection,
        max_metric_calls_per_prompt=1,
        max_reflection_cost_usd_per_prompt=1.0,
        reflection_minibatch_size=1,
        seed=23,
        optimize_fn=injected_optimize,
    )
    trace = json.loads(run.trace_path.read_text(encoding="utf-8"))

    assert observed_lms == [reflection, reflection]
    assert trace["reflection_lm_usage"]["status"] == "unavailable"
    assert trace["reflection_lm_identity"] == module._reflection_lm_identity(reflection)
    assert "0x" not in json.dumps(trace["reflection_lm_identity"])


def test_cli_wires_separate_task_and_reflection_cost_budgets() -> None:
    args = optimization_cli._parser().parse_args(
        [
            "optimize",
            "--manifest",
            "splits/manifest.json",
            "--run-dir",
            "artifacts/gepa-run",
            "--seed",
            "19",
            "--reflection-lm",
            "anthropic/claude-sonnet-5",
            "--cost-cap-usd",
            "0.03",
            "--max-reflection-cost-usd-per-prompt",
            "4.5",
            "--reflection-max-tokens",
            "1200",
            "--reflection-temperature",
            "0.4",
            "--live",
        ]
    )

    assert args.cost_cap_usd == 0.03
    assert args.max_reflection_cost_usd_per_prompt == 4.5
    assert args.reflection_batch_headroom_usd_per_prompt == 0.5
    assert args.reflection_max_tokens == 1200
    assert args.reflection_temperature == 0.4

    compare_args = optimization_cli._parser().parse_args(
        [
            "compare-test",
            "--manifest",
            "splits/manifest.json",
            "--winner",
            "artifacts/gepa-run/frozen_winner.json",
            "--seed-extraction-template",
            "prompts/original-extraction.md",
            "--output",
            "artifacts/gepa-paired-test.json",
            "--live",
        ]
    )
    assert compare_args.command == "compare-test"
    assert compare_args.seed_extraction_template == Path(
        "prompts/original-extraction.md"
    )
    assert compare_args.seed_verification_template is None


def test_official_gepa_014_adapter_contract_when_extra_is_installed(tmp_path: Path) -> None:
    gepa = pytest.importorskip("gepa")
    example = extraction_example("official-api")
    candidate = {
        "extraction_prompt": "Prompt version: `gepa-official-v1`\n[[SOURCE_TEXT]]\n"
    }
    key = provider_request_key(example, candidate)
    provider = FixtureProvider(
        {("gepa-extraction", key): example.model_dump(mode="json")["expected_output"]}
    )
    adapter = GEPAPromptAdapter(provider)

    result = gepa.optimize(
        seed_candidate=candidate,
        trainset=[example],
        valset=[example],
        adapter=adapter,
        reflection_lm=lambda _prompt: f"```\n{candidate['extraction_prompt']}\n```",
        max_metric_calls=1,
        max_reflection_cost=0.01,
        reflection_minibatch_size=1,
        seed=23,
        run_dir=str(tmp_path / "official-gepa"),
        use_merge=True,
        cache_evaluation=True,
        track_best_outputs=True,
        display_progress_bar=False,
    )

    assert result.best_candidate == candidate
    assert result.val_aggregate_scores[result.best_idx] == 1.0
    assert result.val_aggregate_subscores == [
        {
            "extraction_correctness": 1.0,
            "grounding_schema_validity": 1.0,
            "cost_efficiency": 1.0,
        }
    ]


def test_official_gepa_wrapper_freezes_trace_when_extra_is_installed(
    tmp_path: Path,
) -> None:
    pytest.importorskip("gepa")
    examples = [extraction_example(character) for character in "abcdef"]
    source = tmp_path / "examples.jsonl"
    atomic_write_jsonl(source, [example.model_dump(mode="json") for example in examples])
    manifest_path = create_split_bundle(source, tmp_path / "splits", seed=31)
    train, dev = load_optimization_examples(manifest_path)
    template_path = tmp_path / "seed.md"
    template_path.write_text(
        "Prompt version: `gepa-wrapper-v1`\n[[SOURCE_TEXT]]\n", encoding="utf-8"
    )
    candidate = {"extraction_prompt": template_path.read_text(encoding="utf-8")}
    responses = {
        ("gepa-extraction", provider_request_key(example, candidate)): example.model_dump(
            mode="json"
        )["expected_output"]
        for example in [*train, *dev]
    }

    run = optimize_prompts(
        manifest_path=manifest_path,
        seed_templates={"extraction": template_path},
        provider=FixtureProvider(responses),
        run_dir=tmp_path / "run",
        reflection_lm=lambda _prompt: f"```\n{candidate['extraction_prompt']}\n```",
        max_metric_calls_per_prompt=len(dev),
        max_reflection_cost_usd_per_prompt=0.01,
        reflection_minibatch_size=1,
        seed=37,
    )

    assert run.winner_path.is_file()
    winner = json.loads(run.winner_path.read_text(encoding="utf-8"))
    assert set(winner["seed_prompt_sha256s"]) == {"extraction"}
    trace = json.loads(run.trace_path.read_text(encoding="utf-8"))
    assert trace["gepa_version"] != "unknown"
    assert trace["test_split_opened"] is False
    assert trace["test_evaluated"] is False
    assert trace["component_traces"]["extraction"]["best_score"] == 1.0
    assert trace["reflection_lm_usage"]["status"] == "available"
    assert trace["reflection_lm_usage"]["total_cost_usd"] == 0.0
    assert isinstance(trace["reflection_lm_usage"]["total_input_tokens"], int)
    assert isinstance(trace["reflection_lm_usage"]["total_output_tokens"], int)
