from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from scripts import evaluate_evidence_inference_diagnostic as diagnostic_cli

import literature_multiverse.evidence_inference_diagnostic as diagnostic
from literature_multiverse.evidence_inference import convert_evidence_inference
from literature_multiverse.evidence_inference_diagnostic import (
    EvidenceInferenceDiagnosticError,
    article_clustered_interval,
    build_provider_free_diagnostic_bundle,
    build_public_diagnostic_summary,
    lexical_extraction_output,
    run_full_lexical_diagnostic,
    score_diagnostic_output,
    validate_diagnostic_report,
    validate_prediction_ledger,
    validate_public_diagnostic_summary,
)
from literature_multiverse.lineage import OutputExistsError, sha256_file
from literature_multiverse.prompt_optimization import (
    OptimizationExample,
    load_manifest_split,
    load_split_manifest,
)
from literature_multiverse.providers import sha256_json


@pytest.fixture
def evidence_fixture(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "evidence_inference_v2"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _provider_receipt(
    example: OptimizationExample,
    candidate: dict[str, str],
    *,
    model: str = "fixture-model",
) -> dict[str, Any]:
    candidate_sha256 = diagnostic._candidate_sha256(candidate)
    receipt: dict[str, Any] = {
        "operation": "gepa-extraction",
        "request_key": f"{example.example_id}-{candidate_sha256[:16]}",
        "provider": "fixture-provider",
        "model": model,
        "effort": "low",
        "max_tokens": 400,
        "system": None,
        "prompt": diagnostic._provider_prompt(example, candidate),
        "output_schema": example.output_schema,
        "output_schema_original_sha256": None,
        "output_schema_provider": None,
        "output_schema_provider_sha256": None,
        "output_schema_transform": None,
        "status": "complete",
        "failure": None,
        "response_text": json.dumps(example.expected_output, sort_keys=True),
        "parsed_json": example.expected_output,
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "estimated_cost_usd": 0.01,
        "cost_basis": "fixture_reported_usage",
    }
    request_fields = (
        "operation",
        "request_key",
        "provider",
        "model",
        "effort",
        "max_tokens",
        "system",
        "prompt",
        "output_schema",
        "output_schema_original_sha256",
        "output_schema_provider",
        "output_schema_provider_sha256",
        "output_schema_transform",
    )
    receipt["request_sha256"] = sha256_json(
        {field: receipt.get(field) for field in request_fields}
    )
    return receipt


def _build_bound_archive(
    manifest_path: Path,
    seed_prompt_path: Path,
    archive_root: Path,
    *,
    mutation_model: str = "fixture-model",
    duplicate: str | None = None,
    corrupt_trace_manifest: bool = False,
    corrupt_trace_ids: bool = False,
    corrupt_trace_candidates: bool = False,
    corrupt_opened_test_ids: bool = False,
) -> Path:
    manifest = load_split_manifest(manifest_path)
    train = load_manifest_split(manifest_path, "train")[0]
    test = load_manifest_split(manifest_path, "test")[0]
    seed_text = seed_prompt_path.read_text(encoding="utf-8")
    seed_candidate = {"extraction_prompt": seed_text}
    mutation_candidate = {
        "extraction_prompt": seed_text.rstrip()
        + "\n\nAdditional diagnostic instruction: preserve exact source bytes."
    }
    candidates = [seed_candidate, mutation_candidate]
    candidate_hashes = [diagnostic._candidate_sha256(value) for value in candidates]
    run_dir = archive_root / "fixture-run"
    extraction_dir = run_dir / "extraction"
    _write_json(extraction_dir / "candidates.json", candidates)
    (extraction_dir / "run_log.txt").write_text(
        "Iteration 1: Proposed new text for extraction_prompt: "
        + mutation_candidate["extraction_prompt"]
        + "\nIteration 1: New subsample score: 1.0\n",
        encoding="utf-8",
    )
    _write_json(
        extraction_dir / "run_log.json",
        [
            {
                "i": 0,
                "selected_program_candidate": 0,
                "subsample_ids": [0],
                "n_tasks": 1,
                "tasks": [{"parent_idx": 0, "subsample_ids": [0]}],
            }
        ],
    )
    manifest_sha256 = sha256_file(manifest_path)
    trace = {
        "optimization_trace_version": "1",
        "run_id": "fixture-run",
        "optimizer": "gepa.optimize",
        "gepa_version": "fixture",
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": (
            "0" * 64 if corrupt_trace_manifest else manifest_sha256
        ),
        "manifest_seed": manifest.seed,
        "optimizer_seed": 4,
        "optimization_splits": ["train", "dev"],
        "test_split_opened": False,
        "test_evaluated": False,
        "objective_weights": {},
        "cost_cap_usd": 0.02,
        "max_metric_calls_per_prompt": 2,
        "max_reflection_cost_usd_per_prompt": 0.5,
        "reflection_minibatch_size": 1,
        "reflection_lm_kwargs": {"max_tokens": 100, "num_retries": 0},
        "train_example_ids": (
            ["ei2-prompt-999999"]
            if corrupt_trace_ids
            else manifest.train.example_ids
        ),
        "dev_example_ids": manifest.dev.example_ids,
        "seed_prompt_sha256s": {
            "extraction": hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
        },
        "component_traces": {
            "extraction": {
                "candidates": candidates,
                "candidate_sha256s": (
                    ["f" * 64, candidate_hashes[1]]
                    if corrupt_trace_candidates
                    else candidate_hashes
                ),
                "val_aggregate_scores": [1.0, 1.0],
            }
        },
    }
    _write_json(run_dir / "optimization_trace.json", trace)
    attempts = run_dir / "provider_attempts"
    receipts = [
        (train, seed_candidate, "fixture-model"),
        (train, mutation_candidate, mutation_model),
        (test, seed_candidate, "fixture-model"),
    ]
    written: list[Path] = []
    for example, candidate, model in receipts:
        receipt = _provider_receipt(example, candidate, model=model)
        path = attempts / f"{receipt['request_key']}.provider.json"
        _write_json(path, receipt)
        written.append(path)
    if duplicate is not None:
        duplicate_payload = json.loads(written[0].read_text(encoding="utf-8"))
        if duplicate == "nonidentical":
            duplicate_payload["attempted_at"] = "different"
        _write_json(attempts / f"duplicate-{written[0].name}", duplicate_payload)
    _write_json(
        run_dir / "heldout-test.json",
        {
            "heldout_evaluation_version": "1",
            "manifest_sha256": manifest_sha256,
            "split": "test",
            "test_example_ids": [
                train.example_id if corrupt_opened_test_ids else test.example_id
            ],
            "winner_sha256": "fixture",
        },
    )
    return archive_root


def test_fixed_lexical_extractor_has_formal_provenance_not_entailment(
    evidence_fixture: Path, tmp_path: Path
) -> None:
    converted = convert_evidence_inference(evidence_fixture, tmp_path / "converted")
    example = load_manifest_split(converted.manifest_path, "test")[0]

    output, disposition = lexical_extraction_output(example)
    scored = score_diagnostic_output(
        example, output, baseline_disposition=disposition
    )

    assert output["findings"][0]["direction"] == "decrease"
    assert scored["exact_structured_output_validity"] == 1.0
    assert scored["task_shape_consistency"] == 1.0
    assert scored["direction_accuracy"] == 1.0
    assert scored["formal_quote_line_provenance_validity"] == 1.0
    assert scored["schema_direction_provenance_joint_validity"] == 1.0


def test_task_shape_contradiction_is_separate_from_json_schema_validity(
    evidence_fixture: Path, tmp_path: Path
) -> None:
    converted = convert_evidence_inference(evidence_fixture, tmp_path / "converted")
    example = load_manifest_split(converted.manifest_path, "test")[0]
    contradictory = {
        "eligible": False,
        "findings": deepcopy(example.expected_output["findings"]),
    }

    scored = score_diagnostic_output(example, contradictory)

    assert scored["exact_structured_output_validity"] == 1.0
    assert scored["task_shape_consistency"] == 0.0
    assert scored["schema_direction_provenance_joint_validity"] == 0.0
    assert scored["primary_failure"] == "task_shape_inconsistent"


def test_article_clustered_intervals_are_deterministic_and_cluster_by_paper() -> None:
    rows = [
        {"paper_id": "p1", "metric": 1.0},
        {"paper_id": "p1", "metric": 0.0},
        {"paper_id": "p2", "metric": 1.0},
    ]

    first = article_clustered_interval(rows, "metric", seed=7, replicates=200)
    second = article_clustered_interval(rows, "metric", seed=7, replicates=200)

    assert first == second
    assert first["estimate"] == pytest.approx(2 / 3)
    assert first["rows"] == 3
    assert first["articles"] == 2


def test_prediction_freeze_precedes_scoring_and_uses_no_provider(
    evidence_fixture: Path, tmp_path: Path
) -> None:
    converted = convert_evidence_inference(evidence_fixture, tmp_path / "converted")
    examples = load_manifest_split(converted.manifest_path, "test")

    result = run_full_lexical_diagnostic(examples, seed=3, replicates=100)

    assert result["provider_calls"] == 0
    assert result["training_labels_used"] is False
    assert result["prediction_stage_received_label_fields"] is False
    assert len(result["prediction_freeze_sha256"]) == 64


def test_bound_report_ledger_public_summary_and_cli(
    evidence_fixture: Path, repo_root: Path, tmp_path: Path
) -> None:
    converted = convert_evidence_inference(evidence_fixture, tmp_path / "converted")
    seed_prompt = repo_root / "prompts" / "evidence_inference_extraction.md"
    archive_root = _build_bound_archive(
        converted.manifest_path, seed_prompt, tmp_path / "gepa"
    )
    report, ledger = build_provider_free_diagnostic_bundle(
        manifest_path=converted.manifest_path,
        previously_opened_manifest_path=converted.manifest_path,
        seed_prompt_path=seed_prompt,
        archive_roots=[archive_root],
        seed=11,
        replicates=100,
    )
    public = build_public_diagnostic_summary(report, ledger)

    assert validate_diagnostic_report(report) == report
    assert validate_prediction_ledger(ledger) == ledger
    assert validate_public_diagnostic_summary(public) == public
    assert report["semantic_support_or_entailment_measured"] is False
    assert report["registered_previous_provider_test_attempt_rows"] == 1
    assert report["excluded_rows_on_provider_touched_articles"] == 1
    assert report["provider_call_unseen_paper_diagnostic_rows"] == 0
    assert report["prediction_ledger"]["ledger_sha256"] == ledger["ledger_sha256"]
    assert len(report["execution_fingerprint_sha256"]) == 64
    assert len(report["code_fingerprint"]["code_fingerprint_sha256"]) == 64
    assert len(report["input_fingerprint"]["input_fingerprint_sha256"]) == 64
    assert ledger["contains_evidence_quotes"] is False
    serialized_public = json.dumps(public, sort_keys=True)
    assert "direction_confusion" not in serialized_public
    assert "predicted_direction" not in serialized_public
    assert "ei2-prompt" not in serialized_public
    assert "PMC100" not in serialized_public

    tampered_ledger = deepcopy(ledger)
    # This fixture's provider-call-unseen subset is empty, so attack the all-row ledger.
    subset = tampered_ledger["all_opened_test_rows"]
    subset["rows"][0]["evidence_quote"] = "protected source text"
    subset_payload = {key: value for key, value in subset.items() if key != "ledger_sha256"}
    subset["ledger_sha256"] = diagnostic.hash_canonical(subset_payload)
    ledger_payload = {
        key: value for key, value in tampered_ledger.items() if key != "ledger_sha256"
    }
    tampered_ledger["ledger_sha256"] = diagnostic.hash_canonical(ledger_payload)
    with pytest.raises(
        EvidenceInferenceDiagnosticError,
        match="protected source or label fields",
    ):
        validate_prediction_ledger(tampered_ledger)

    tampered_public = deepcopy(public)
    public_payload = {
        key: value for key, value in tampered_public.items() if key != "public_summary_sha256"
    }
    public_payload["raw_predictions"] = [{"predicted_direction": "increase"}]
    tampered_public = {
        **public_payload,
        "public_summary_sha256": diagnostic.hash_canonical(public_payload),
    }
    with pytest.raises(
        EvidenceInferenceDiagnosticError,
        match="row-level identities or predictions",
    ):
        validate_public_diagnostic_summary(tampered_public)

    output = tmp_path / "report.json"
    ledger_output = tmp_path / "ledger.json"
    public_output = tmp_path / "public.json"
    args = [
        "--manifest",
        str(converted.manifest_path),
        "--previously-opened-manifest",
        str(converted.manifest_path),
        "--seed-prompt",
        str(seed_prompt),
        "--archive-root",
        str(archive_root),
        "--output",
        str(output),
        "--prediction-ledger-output",
        str(ledger_output),
        "--public-summary-output",
        str(public_output),
        "--bootstrap-replicates",
        "100",
    ]
    assert diagnostic_cli.main(args) == 0
    assert output.is_file() and ledger_output.is_file() and public_output.is_file()
    with pytest.raises(OutputExistsError):
        diagnostic_cli.main(args)


@pytest.mark.parametrize(
    ("archive_options", "message"),
    [
        ({"corrupt_trace_manifest": True}, "trace manifest hash mismatch"),
        ({"corrupt_trace_ids": True}, "trace train identities mismatch"),
        ({"corrupt_trace_candidates": True}, "trace candidate binding mismatch"),
        (
            {"corrupt_opened_test_ids": True},
            "opened test evaluation artifact does not bind its receipts",
        ),
        (
            {"mutation_model": "different-model"},
            "unequal provider execution contracts",
        ),
        (
            {"duplicate": "nonidentical"},
            "nonidentical duplicate candidate/example provider receipts",
        ),
    ],
)
def test_archive_contract_violations_fail_closed(
    evidence_fixture: Path,
    repo_root: Path,
    tmp_path: Path,
    archive_options: dict[str, Any],
    message: str,
) -> None:
    converted = convert_evidence_inference(evidence_fixture, tmp_path / "converted")
    seed_prompt = repo_root / "prompts" / "evidence_inference_extraction.md"
    archive_root = _build_bound_archive(
        converted.manifest_path,
        seed_prompt,
        tmp_path / "gepa",
        **archive_options,
    )

    with pytest.raises(EvidenceInferenceDiagnosticError, match=message):
        build_provider_free_diagnostic_bundle(
            manifest_path=converted.manifest_path,
            previously_opened_manifest_path=converted.manifest_path,
            seed_prompt_path=seed_prompt,
            archive_roots=[archive_root],
            replicates=100,
        )


def test_byte_identical_duplicate_receipt_is_deduplicated(
    evidence_fixture: Path, repo_root: Path, tmp_path: Path
) -> None:
    converted = convert_evidence_inference(evidence_fixture, tmp_path / "converted")
    seed_prompt = repo_root / "prompts" / "evidence_inference_extraction.md"
    archive_root = _build_bound_archive(
        converted.manifest_path,
        seed_prompt,
        tmp_path / "gepa",
        duplicate="identical",
    )

    report, _ = build_provider_free_diagnostic_bundle(
        manifest_path=converted.manifest_path,
        previously_opened_manifest_path=converted.manifest_path,
        seed_prompt_path=seed_prompt,
        archive_roots=[archive_root],
        replicates=100,
    )

    run = report["archived_gepa_response_replay"]["runs"][0]
    assert run["byte_identical_duplicate_receipts_deduplicated"] == 1
    assert run["nonidentical_duplicate_receipts_accepted"] == 0


def test_cache_defect_claim_requires_exact_evidence_binding(
    evidence_fixture: Path, repo_root: Path, tmp_path: Path
) -> None:
    converted = convert_evidence_inference(evidence_fixture, tmp_path / "converted")
    seed_prompt = repo_root / "prompts" / "evidence_inference_extraction.md"
    archive_root = _build_bound_archive(
        converted.manifest_path, seed_prompt, tmp_path / "gepa"
    )
    report, _ = build_provider_free_diagnostic_bundle(
        manifest_path=converted.manifest_path,
        previously_opened_manifest_path=converted.manifest_path,
        seed_prompt_path=seed_prompt,
        archive_roots=[archive_root],
        replicates=100,
    )
    payload = deepcopy({key: value for key, value in report.items() if key != "report_sha256"})
    payload["archived_gepa_response_replay"]["runs"][0][
        "cache_integrity_finding"
    ] = {
        "status": "fail_closed_trace_score_excluded",
        "evidence_binding_status": "verified_exact_archived_run",
        "evidence_binding": {"incorrect": True},
        "archived_trace_mutation_dev_score_usable": False,
        "archived_trace_score_citation_allowed": False,
        "expected_dev_rows": 5,
        "clean_common_dev_receipts": 3,
        "missing_mutation_dev_receipts": 2,
    }
    tampered = {**payload, "report_sha256": diagnostic.hash_canonical(payload)}

    with pytest.raises(
        EvidenceInferenceDiagnosticError,
        match="cache-collision finding did not fail closed",
    ):
        validate_diagnostic_report(tampered)
