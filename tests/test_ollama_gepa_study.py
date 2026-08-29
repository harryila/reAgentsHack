from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import literature_multiverse.ollama_gepa_study as study
from literature_multiverse.evidence_inference import EVIDENCE_INFERENCE_OUTPUT_SCHEMA
from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file
from literature_multiverse.local_ollama import (
    OllamaGenerationConfig,
    OllamaGenerationResult,
    OllamaIdentity,
)
from literature_multiverse.prompt_optimization import (
    OptimizationExample,
    OptimizationSplitManifest,
    SplitArtifact,
)

SEED_PROMPT = """# Fixture extraction

Prompt version: `fixture-seed-v1`

Outcome: [[OUTCOME]]
Intervention: [[INTERVENTION]]
Comparator: [[COMPARATOR]]

Return the required structured direction and an exact quote/line citation.
"""

BETTER_PROMPT = """# Fixture extraction with explicit-marker handling

Prompt version: `fixture-gepa-winner-v1`

Outcome: [[OUTCOME]]
Intervention: [[INTERVENTION]]
Comparator: [[COMPARATOR]]

Use the explicit direction marker when present. Return the required structured direction and an
exact quote with the exact source line ID. Never infer a direction from ordering alone.
"""

EXPECTED_STUDY_SOURCE_CLOSURE = {
    "pyproject.toml",
    "scripts/run_ollama_gepa_study.py",
    "src/literature_multiverse/__init__.py",
    "src/literature_multiverse/evidence_inference.py",
    "src/literature_multiverse/evidence_inference_diagnostic.py",
    "src/literature_multiverse/evidence_inference_ollama.py",
    "src/literature_multiverse/grounding.py",
    "src/literature_multiverse/lineage.py",
    "src/literature_multiverse/local_ollama.py",
    "src/literature_multiverse/models.py",
    "src/literature_multiverse/ollama_gepa_study.py",
    "src/literature_multiverse/paths.py",
    "src/literature_multiverse/prompt_optimization.py",
    "src/literature_multiverse/prompting.py",
    "src/literature_multiverse/providers.py",
    "uv.lock",
}


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def test_source_hashes_cover_complete_internal_dependency_closure() -> None:
    observed = study._source_code_hashes()
    assert set(observed) == EXPECTED_STUDY_SOURCE_CLOSURE
    for relative, expected in observed.items():
        assert sha256_file(Path(relative)) == expected


class FakeOllamaClient:
    def __init__(self, *, allow_generate: bool = True) -> None:
        self.allow_generate = allow_generate
        self.calls: list[dict[str, Any]] = []

    def inspect_identity(self, config: OllamaGenerationConfig) -> OllamaIdentity:
        return OllamaIdentity(
            ollama_version=config.expected_ollama_version,
            model=config.model,
            model_digest=config.model_digest,
            parameter_size="1.2B",
            quantization_level="Q8_0",
            model_format="gguf",
            model_family="llama",
        )

    def generate(
        self,
        *,
        prompt: str,
        output_schema: dict[str, Any],
        config: OllamaGenerationConfig,
    ) -> OllamaGenerationResult:
        if not self.allow_generate:
            raise AssertionError("receipt replay unexpectedly called the model")
        assert not _contains_key(output_schema, "pattern")
        self.calls.append({"prompt": prompt, "schema": deepcopy(output_schema)})
        if "new_component" in output_schema.get("properties", {}):
            response = {"new_component": BETTER_PROMPT}
        else:
            marker = "DIRECTION:"
            actual = prompt.split(marker, 1)[1].split()[0].strip(".,;`")
            predicted = actual if "explicit-marker handling" in prompt else "no_effect"
            line_enum = output_schema["properties"]["findings"]["items"]["properties"][
                "evidence_lines"
            ]["items"]["enum"]
            line_id = line_enum[0]
            source_text = f"DIRECTION:{actual} explicit fixture trial result."
            response = {
                "eligible": True,
                "findings": [
                    {
                        "direction": predicted,
                        "evidence_quote": source_text,
                        "evidence_lines": [line_id],
                    }
                ],
            }
        return OllamaGenerationResult(
            model=config.model,
            response_text=json.dumps(response, sort_keys=True),
            done=True,
            done_reason="stop",
            total_duration_ns=2_000_000,
            load_duration_ns=10,
            prompt_eval_count=50,
            prompt_eval_duration_ns=100,
            eval_count=40,
            eval_duration_ns=100,
        )


def _example(split: str, index: int) -> OptimizationExample:
    directions = ("increase", "decrease", "no_effect")
    direction = directions[index % len(directions)]
    paper_number = index // 2
    line_text = f"DIRECTION:{direction} explicit fixture trial result."
    return OptimizationExample(
        example_id=f"{split}-example-{index:03d}",
        paper_id=f"{split}-paper-{paper_number:03d}",
        group_id=f"{split}-group-{paper_number:03d}",
        prompt_kind="extraction",
        replacements={
            "OUTCOME": f"fixture outcome {index}",
            "INTERVENTION": "fixture intervention",
            "COMPARATOR": "fixture comparator",
        },
        expected_output={
            "eligible": True,
            "findings": [
                {
                    "direction": direction,
                    "evidence_quote": line_text,
                    "evidence_lines": ["L1"],
                }
            ],
        },
        label_paths=["/findings/0/direction"],
        output_schema=deepcopy(EVIDENCE_INFERENCE_OUTPUT_SCHEMA),
        content_lines={"L1": {"line_number": 1, "section": "Results", "text": line_text}},
        line_sections={"L1": "Results"},
        source_accessible=True,
    )


def _write_split(path: Path, examples: list[OptimizationExample]) -> SplitArtifact:
    path.write_text(
        "".join(
            json.dumps(example.model_dump(mode="json"), sort_keys=True) + "\n"
            for example in examples
        ),
        encoding="utf-8",
    )
    return SplitArtifact(
        path=path.name,
        sha256=sha256_file(path),
        rows=len(examples),
        example_ids=sorted(example.example_id for example in examples),
        paper_ids=sorted({example.paper_id for example in examples}),
        group_ids=sorted({example.group_id for example in examples}),
    )


def _bundle(tmp_path: Path) -> tuple[Path, Path, str]:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train = [_example("train", index) for index in range(30)]
    dev = [_example("dev", index) for index in range(30)]
    test = [_example("test", index) for index in range(30)]
    train_artifact = _write_split(data_dir / "train.jsonl", train)
    dev_artifact = _write_split(data_dir / "dev.jsonl", dev)
    test_path = data_dir / "test.jsonl"
    test_artifact = _write_split(test_path, test)
    test_text = test_path.read_text(encoding="utf-8")
    manifest = OptimizationSplitManifest(
        algorithm="official-paper-groups-v1",
        seed=7,
        train_fraction=0.7,
        dev_fraction=0.1,
        source_examples_sha256="0" * 64,
        train=train_artifact,
        dev=dev_artifact,
        test=test_artifact,
    )
    manifest_path = data_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    seed_path = tmp_path / "seed.md"
    seed_path.write_text(SEED_PROMPT, encoding="utf-8")
    config = {
        "study_version": study.STUDY_VERSION,
        "manifest_path": str(manifest_path),
        "seed_prompt_path": str(seed_path),
        "private_run_dir": str(tmp_path / "private"),
        "public_summary_path": str(tmp_path / "public" / "summary.json"),
        "generation": {
            "model": "llama3.2:1b",
            "model_digest": "a" * 64,
            "expected_ollama_version": "0.15.1",
            "seed": 17,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "num_ctx": 4096,
            "num_predict": 96,
            "keep_alive": "1m",
        },
        "optimization": {
            "subset_seed": 19,
            "train_target_rows": 24,
            "dev_target_rows": 24,
            "max_metric_calls": 100,
            "reflection_minibatch_size": 2,
            "candidate_selection_strategy": "pareto",
            "frontier_type": "hybrid",
            "batch_sampler": "epoch_shuffled",
            "module_selector": "round_robin",
            "acceptance_criterion": "improvement_or_equal",
            "skip_perfect_score": False,
            "use_merge": False,
            "cache_evaluation": False,
            "track_best_outputs": False,
            "gepa_seed": 23,
            "minimum_reflection_proposals": 3,
        },
        "metrics": {
            "direction_accuracy_weight": 0.55,
            "direction_distribution_fidelity_weight": 0.0,
            "formal_grounding_validity_weight": 0.2,
            "structured_output_validity_weight": 0.1,
            "generation_success_weight": 0.1,
            "token_efficiency_weight": 0.025,
            "latency_sla_success_weight": 0.025,
            "latency_sla_seconds": 10.0,
            "bootstrap_seed": 29,
            "bootstrap_replicates": 1000,
        },
    }
    config_path = tmp_path / "study.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return config_path, test_path, test_text


class FakeGEPAResult:
    def __init__(self, candidates: list[dict[str, str]], *, best_idx: int) -> None:
        seed_objectives = {objective: 0.5 for objective in study._OBJECTIVES}
        winner_objectives = {objective: 0.8 for objective in study._OBJECTIVES}
        self.payload = {
            "candidates": candidates,
            "parents": [[None], [0]],
            "val_aggregate_scores": [0.5, 0.8],
            "val_subscores": [{}, {}],
            "best_outputs_valset": None,
            "per_val_instance_best_candidates": {},
            "val_aggregate_subscores": [seed_objectives, winner_objectives],
            "per_objective_best_candidates": {},
            "objective_pareto_front": {},
            "discovery_eval_counts": [24, 76],
            "total_metric_calls": 100,
            "num_full_val_evals": 2,
            "run_dir": "private",
            "seed": 23,
            "best_idx": best_idx,
            "validation_schema_version": 2,
        }

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.payload)


def _fake_optimize(*, select_seed: bool = False):
    def optimize(**kwargs: Any) -> FakeGEPAResult:
        assert kwargs["task_lm"] is None
        assert kwargs["reflection_lm"] is None
        assert kwargs["cache_evaluation"] is False
        assert kwargs["frontier_type"] == "hybrid"
        adapter = kwargs["adapter"]
        seed = kwargs["seed_candidate"]
        evaluation = adapter.evaluate(kwargs["trainset"][:2], seed, capture_traces=True)
        reflective = adapter.make_reflective_dataset(seed, evaluation, [study.COMPONENT])
        proposal = seed
        for _ in range(3):
            proposal = adapter.propose_new_texts(seed, reflective, [study.COMPONENT])
        assert proposal[study.COMPONENT] == BETTER_PROMPT.strip()
        return FakeGEPAResult([seed, proposal], best_idx=0 if select_seed else 1)

    return optimize


def _patch_gepa_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(study, "_official_gepa_version", lambda: study.OFFICIAL_GEPA_VERSION)


def test_stage_separation_freeze_paired_test_and_public_redaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, test_path, test_text = _bundle(tmp_path)
    _patch_gepa_version(monkeypatch)
    client = FakeOllamaClient()

    # A physically absent test payload proves prepare/optimization do not open or hash it.
    test_path.unlink()
    plan_path = study.prepare_optimization_plan(config_path=config_path, client=client)
    winner_path = study.run_optimization(
        config_path=config_path,
        client=client,
        optimize_fn=_fake_optimize(),
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    winner = json.loads(winner_path.read_text(encoding="utf-8"))
    assert plan["test_payload_opened"] is False
    assert winner["test_payload_opened"] is False
    assert winner["reflection_proposals"] == 3
    assert winner["seed_retained"] is False

    test_path.write_text(test_text, encoding="utf-8")
    private_path, public_path = study.run_paired_test(
        config_path=config_path,
        client=client,
    )
    private = json.loads(private_path.read_text(encoding="utf-8"))
    public = study.validate_public_summary(json.loads(public_path.read_text(encoding="utf-8")))
    assert private["examples"] == 30
    assert private["articles"] == 15
    assert public["observed_improvement_rule_satisfied"] is True
    assert public["confirmatory_claim_allowed"] is False
    assert public["schemas"]["generation_schema_avoids_regex_for_ollama_0_15_1"] is True
    assert "direction_macro_recall" in public["metrics"]
    assert (
        public["optimizer"]["development_only_objectives"]["direction_distribution_fidelity"] == 0.8
    )
    serialized = json.dumps(public, sort_keys=True)
    assert "test-example" not in serialized
    assert "test-paper" not in serialized
    assert "DIRECTION:" not in serialized
    assert "parsed_output" not in serialized

    # Completed stages are idempotent: no test rescoring or summary retuning.
    call_count = len(client.calls)
    assert study.run_paired_test(config_path=config_path, client=client) == (
        private_path,
        public_path,
    )
    assert len(client.calls) == call_count


def test_receipts_are_tamper_evident_and_replay_without_physical_call(
    tmp_path: Path,
) -> None:
    example = _example("train", 0)
    config = OllamaGenerationConfig(
        model="llama3.2:1b",
        model_digest="a" * 64,
        expected_ollama_version="0.15.1",
        seed=1,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        num_ctx=4096,
        num_predict=96,
        keep_alive="1m",
    )
    metrics = study.MetricSettings(
        direction_accuracy_weight=0.55,
        direction_distribution_fidelity_weight=0.0,
        formal_grounding_validity_weight=0.2,
        structured_output_validity_weight=0.1,
        generation_success_weight=0.1,
        token_efficiency_weight=0.025,
        latency_sla_success_weight=0.025,
        latency_sla_seconds=10,
        bootstrap_seed=1,
        bootstrap_replicates=1000,
    )
    first_client = FakeOllamaClient()
    identity = first_client.inspect_identity(config)
    first = study.OllamaGEPAAdapter(
        client=first_client,
        identity=identity,
        config=config,
        metrics=metrics,
        receipt_root=tmp_path / "receipts",
        namespace="resume-test",
    )
    evaluated = first.evaluate([example], {study.COMPONENT: BETTER_PROMPT})
    assert evaluated.num_metric_calls == 1
    assert len(first_client.calls) == 1

    replay_client = FakeOllamaClient(allow_generate=False)
    replay = study.OllamaGEPAAdapter(
        client=replay_client,
        identity=identity,
        config=config,
        metrics=metrics,
        receipt_root=tmp_path / "receipts",
        namespace="resume-test",
    )
    replayed = replay.evaluate([example], {study.COMPONENT: BETTER_PROMPT})
    assert replayed.num_metric_calls == 1
    assert replay.get_adapter_state()["task_receipt_replays"] == 1

    receipt_path = next((tmp_path / "receipts").glob("task/*/*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["success"] = False
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(study.OllamaGEPAStudyError, match="self-hash mismatch"):
        study.validate_private_receipts(tmp_path / "receipts")


def test_scientific_objectives_require_valid_shape_and_exact_original_citation() -> None:
    assert study._direction_distribution_fidelity(
        ["increase", "decrease", "no_effect"],
        ["increase", "increase", "increase"],
    ) == pytest.approx(1 / 3)
    assert (
        study._direction_distribution_fidelity(
            ["increase", "decrease", "no_effect"],
            ["increase", "decrease", "no_effect"],
        )
        == 1.0
    )
    example = _example("train", 0)
    source = example.model_dump(mode="json")
    source["content_lines"]["L2"] = {
        "line_number": 2,
        "section": "Results",
        "text": "Unrelated result line.",
    }
    source["line_sections"]["L2"] = "Results"
    example = OptimizationExample.model_validate(source)
    result = OllamaGenerationResult(
        model="llama3.2:1b",
        response_text="{}",
        done=True,
        total_duration_ns=100,
        eval_count=10,
    )
    metrics = study.MetricSettings(
        direction_accuracy_weight=0.55,
        direction_distribution_fidelity_weight=0.0,
        formal_grounding_validity_weight=0.2,
        structured_output_validity_weight=0.1,
        generation_success_weight=0.1,
        token_efficiency_weight=0.025,
        latency_sla_success_weight=0.025,
        latency_sla_seconds=10,
        bootstrap_seed=1,
        bootstrap_replicates=1000,
    )
    wrong_citation = deepcopy(example.expected_output)
    wrong_citation["findings"][0]["evidence_lines"] = ["L2"]
    objectives, details = study._score_parsed_output(
        example=example,
        parsed=wrong_citation,
        result=result,
        metrics=metrics,
        num_predict=96,
    )
    assert objectives["direction_accuracy"] == 1.0
    assert objectives["formal_grounding_validity"] == 0.0
    assert details["grounding_result"]["relocated_from_line_numbers"] == [2]

    invalid_shape = {**deepcopy(example.expected_output), "unexpected": True}
    invalid_objectives, _ = study._score_parsed_output(
        example=example,
        parsed=invalid_shape,
        result=result,
        metrics=metrics,
        num_predict=96,
    )
    assert invalid_objectives["direction_accuracy"] == 0.0
    assert invalid_objectives["structured_output_validity"] == 0.0


def test_no_results_projection_is_a_zero_score_without_model_fallback(tmp_path: Path) -> None:
    source = _example("train", 0).model_dump(mode="json")
    source["content_lines"]["L1"]["section"] = "Methods"
    source["line_sections"]["L1"] = "Methods"
    example = OptimizationExample.model_validate(source)
    config = OllamaGenerationConfig(
        model="llama3.2:1b",
        model_digest="a" * 64,
        expected_ollama_version="0.15.1",
        seed=1,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        num_ctx=4096,
        num_predict=96,
        keep_alive="1m",
    )
    metrics = study.MetricSettings(
        direction_accuracy_weight=0.55,
        direction_distribution_fidelity_weight=0.0,
        formal_grounding_validity_weight=0.2,
        structured_output_validity_weight=0.1,
        generation_success_weight=0.1,
        token_efficiency_weight=0.025,
        latency_sla_success_weight=0.025,
        latency_sla_seconds=10,
        bootstrap_seed=1,
        bootstrap_replicates=1000,
    )
    client = FakeOllamaClient(allow_generate=False)
    adapter = study.OllamaGEPAAdapter(
        client=client,
        identity=client.inspect_identity(config),
        config=config,
        metrics=metrics,
        receipt_root=tmp_path / "receipts",
        namespace="no-results",
    )
    evaluated = adapter.evaluate([example], {study.COMPONENT: SEED_PROMPT})
    assert evaluated.num_metric_calls == 1
    assert evaluated.scores == [0.0]
    assert evaluated.outputs[0]["error_category"] == "no_results_passage"
    assert all(value == 0.0 for value in evaluated.objective_scores[0].values())


def test_plan_tamper_and_frozen_winner_tamper_fail_closed_before_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, test_path, _ = _bundle(tmp_path)
    _patch_gepa_version(monkeypatch)
    client = FakeOllamaClient()
    study.prepare_optimization_plan(config_path=config_path, client=client)
    paths = study.study_paths(config_path)
    plan = json.loads(paths.plan.read_text(encoding="utf-8"))
    plan["source_code_sha256s"]["src/literature_multiverse/grounding.py"] = "f" * 64
    plan_without_hash = deepcopy(plan)
    plan_without_hash.pop("plan_sha256")
    plan["plan_sha256"] = hash_canonical(plan_without_hash)
    atomic_write_json(paths.plan, plan, force=True)
    test_path.unlink()
    with pytest.raises(study.OllamaGEPAStudyError, match="implementation changed"):
        study.validate_optimization_plan(config_path=config_path, client=client)


def test_winner_prompt_tamper_fails_before_missing_test_is_accessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, test_path, _ = _bundle(tmp_path)
    _patch_gepa_version(monkeypatch)
    client = FakeOllamaClient()
    study.prepare_optimization_plan(config_path=config_path, client=client)
    study.run_optimization(
        config_path=config_path,
        client=client,
        optimize_fn=_fake_optimize(),
    )
    paths = study.study_paths(config_path)
    paths.winner_prompt.write_text(
        paths.winner_prompt.read_text(encoding="utf-8") + "\nTAMPERED\n",
        encoding="utf-8",
    )
    test_path.unlink()
    with pytest.raises(study.OllamaGEPAStudyError, match="lineage no longer matches"):
        study.run_paired_test(config_path=config_path, client=client)


def test_seed_retention_never_becomes_an_improvement_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _, _ = _bundle(tmp_path)
    _patch_gepa_version(monkeypatch)
    client = FakeOllamaClient()
    study.prepare_optimization_plan(config_path=config_path, client=client)
    study.run_optimization(
        config_path=config_path,
        client=client,
        optimize_fn=_fake_optimize(select_seed=True),
    )
    _, public_path = study.run_paired_test(config_path=config_path, client=client)
    public = json.loads(public_path.read_text(encoding="utf-8"))
    assert public["seed_retained"] is True
    assert public["observed_improvement_rule_satisfied"] is False
    assert public["status"] == "seed_retained_no_improvement_claim"


def test_official_gepa_014_accepts_local_adapter_contract(tmp_path: Path) -> None:
    gepa = pytest.importorskip("gepa")
    example = _example("train", 0)
    config = OllamaGenerationConfig(
        model="llama3.2:1b",
        model_digest="a" * 64,
        expected_ollama_version="0.15.1",
        seed=1,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        num_ctx=4096,
        num_predict=96,
        keep_alive="1m",
    )
    metrics = study.MetricSettings(
        direction_accuracy_weight=0.55,
        direction_distribution_fidelity_weight=0.0,
        formal_grounding_validity_weight=0.2,
        structured_output_validity_weight=0.1,
        generation_success_weight=0.1,
        token_efficiency_weight=0.025,
        latency_sla_success_weight=0.025,
        latency_sla_seconds=10,
        bootstrap_seed=1,
        bootstrap_replicates=1000,
    )
    client = FakeOllamaClient()
    adapter = study.OllamaGEPAAdapter(
        client=client,
        identity=client.inspect_identity(config),
        config=config,
        metrics=metrics,
        receipt_root=tmp_path / "receipts",
        namespace="official-contract",
    )
    result = gepa.optimize(
        seed_candidate={study.COMPONENT: SEED_PROMPT},
        trainset=[example],
        valset=[example],
        adapter=adapter,
        max_metric_calls=8,
        reflection_minibatch_size=1,
        skip_perfect_score=False,
        acceptance_criterion="improvement_or_equal",
        cache_evaluation=False,
        run_dir=str(tmp_path / "gepa"),
        display_progress_bar=False,
        seed=1,
    )
    assert result.total_metric_calls is not None and result.total_metric_calls >= 8
    assert result.num_candidates >= 1
    assert adapter.get_adapter_state()["reflection_proposals"] >= 1
    assert isinstance(SimpleNamespace(result=result).result.best_candidate, dict)
