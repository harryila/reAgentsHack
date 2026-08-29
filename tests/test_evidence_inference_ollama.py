from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import literature_multiverse.evidence_inference_ollama as ollama_diagnostic
from literature_multiverse.evidence_inference import convert_evidence_inference
from literature_multiverse.evidence_inference_ollama import (
    DEFAULT_GENERATION_CONFIG,
    EvidenceInferenceOllamaError,
    build_public_summary,
    canonical_json_file_sha256,
    prepare_input_bundle,
    run_prediction_stage,
    score_frozen_predictions,
    validate_input_bundle,
    validate_prediction_ledger,
    validate_private_report,
    validate_public_summary,
)
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.local_ollama import (
    OllamaGenerationConfig,
    OllamaGenerationResult,
    OllamaIdentity,
)
from literature_multiverse.prompt_optimization import load_manifest_split


@pytest.fixture
def evidence_fixture(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "evidence_inference_v2"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class _FakeOllamaClient:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = list(outputs)
        self.generate_calls = 0
        self.prompts: list[str] = []

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
        del output_schema, config
        self.prompts.append(prompt)
        output = self.outputs[self.generate_calls]
        self.generate_calls += 1
        return OllamaGenerationResult(
            model=DEFAULT_GENERATION_CONFIG.model,
            response_text=json.dumps(output, sort_keys=True),
            done=True,
            done_reason="stop",
            total_duration_ns=100,
            load_duration_ns=10,
            prompt_eval_count=20,
            prompt_eval_duration_ns=30,
            eval_count=10,
            eval_duration_ns=60,
        )


def _prepared_fixture(
    *,
    evidence_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    conversion = convert_evidence_inference(evidence_fixture, tmp_path / "converted")
    test_examples = load_manifest_split(conversion.manifest_path, "test")
    assert len(test_examples) == 1
    example = test_examples[0]
    lexical_output, disposition = ollama_diagnostic.lexical_diagnostic.lexical_extraction_output(
        ollama_diagnostic.lexical_diagnostic._label_stripped_input(example)
    )
    lexical_row = ollama_diagnostic.lexical_diagnostic._redacted_prediction_ledger_row(
        {
            "example_id": example.example_id,
            "paper_id": example.paper_id,
            "output": lexical_output,
            "disposition": disposition,
        }
    )
    subset = {
        "rows": [lexical_row],
        "ledger_sha256": "subset-bound-hash",
    }
    lexical_ledger = {
        "ledger_sha256": "lexical-ledger-bound-hash",
        "provider_call_unseen_paper_subset": subset,
    }
    provider_report = {
        "report_sha256": "provider-report-bound-hash",
        "prediction_ledger": {"ledger_sha256": lexical_ledger["ledger_sha256"]},
        "manifest_file_sha256": sha256_file(conversion.manifest_path),
        "provider_call_unseen_paper_diagnostic_rows": 1,
        "provider_call_unseen_paper_diagnostic_articles": 1,
    }
    provider_report_path = tmp_path / "provider-report.json"
    lexical_ledger_path = tmp_path / "lexical-ledger.json"
    _write_json(provider_report_path, provider_report)
    _write_json(lexical_ledger_path, lexical_ledger)
    monkeypatch.setattr(
        ollama_diagnostic,
        "validate_diagnostic_report",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        ollama_diagnostic,
        "validate_lexical_prediction_ledger",
        lambda value: dict(value),
    )
    bundle = prepare_input_bundle(
        manifest_path=conversion.manifest_path,
        provider_free_report_path=provider_report_path,
        lexical_prediction_ledger_path=lexical_ledger_path,
    )
    return conversion.manifest_path, bundle, provider_report, lexical_ledger


def test_prepare_projection_strips_labels_and_prediction_receipts_resume_and_tamper(
    evidence_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bundle, _, _ = _prepared_fixture(
        evidence_fixture=evidence_fixture,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    serialized = json.dumps(bundle["rows"], sort_keys=True)
    assert "expected_output" not in serialized
    assert "label_paths" not in serialized
    assert bundle["contains_gold_labels"] is False
    assert bundle["row_count"] == 1
    assert bundle["rows"][0]["passages"]

    example_id = bundle["rows"][0]["example_id"]
    model_output = {
        "eligible": True,
        "findings": [
            {
                "direction": "decrease",
                "evidence_quote": bundle["rows"][0]["passages"][0]["text"],
                "evidence_lines": [bundle["rows"][0]["passages"][0]["line_id"]],
            }
        ],
    }
    client = _FakeOllamaClient([model_output])
    receipts = tmp_path / "receipts"
    ledger_path = tmp_path / "prediction-ledger.json"
    first = run_prediction_stage(
        input_bundle=bundle,
        receipts_dir=receipts,
        prediction_ledger_path=ledger_path,
        client=client,
    )
    second = run_prediction_stage(
        input_bundle=bundle,
        receipts_dir=receipts,
        prediction_ledger_path=ledger_path,
        client=client,
    )
    assert client.generate_calls == 1
    assert first == second
    assert first["all_expected_predictions_frozen"] is True
    assert first["prediction_stage_received_label_fields"] is False
    assert example_id not in "\n".join(client.prompts)
    assert "expected_output" not in "\n".join(client.prompts)

    receipt_path = next(receipts.glob("*.json"))
    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["status"] = "execution_failure"
    _write_json(receipt_path, tampered)
    with pytest.raises(EvidenceInferenceOllamaError, match="receipt hash mismatch"):
        run_prediction_stage(
            input_bundle=bundle,
            receipts_dir=receipts,
            prediction_ledger_path=ledger_path,
            client=client,
        )


def test_input_bundle_rejects_nested_label_field(
    evidence_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bundle, _, _ = _prepared_fixture(
        evidence_fixture=evidence_fixture,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    tampered = deepcopy(bundle)
    tampered["rows"][0]["expected_output"] = {"direction": "decrease"}
    row_payload = deepcopy(tampered["rows"][0])
    row_payload.pop("input_row_sha256")
    tampered["rows"][0]["input_row_sha256"] = hash_canonical(row_payload)
    payload = deepcopy(tampered)
    payload.pop("input_bundle_sha256")
    tampered["input_bundle_sha256"] = hash_canonical(payload)

    with pytest.raises(EvidenceInferenceOllamaError, match="protected label field"):
        validate_input_bundle(tampered)


def test_retry_failures_replaces_only_validated_failed_receipt(
    evidence_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bundle, _, _ = _prepared_fixture(
        evidence_fixture=evidence_fixture,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    receipts = tmp_path / "receipts"
    ledger_path = tmp_path / "prediction-ledger.json"
    failed = run_prediction_stage(
        input_bundle=bundle,
        receipts_dir=receipts,
        prediction_ledger_path=ledger_path,
        client=_FakeOllamaClient([]),
    )
    assert failed["status_counts"] == {"execution_failure": 1}

    recovered_client = _FakeOllamaClient([{"eligible": False, "findings": []}])
    recovered = run_prediction_stage(
        input_bundle=bundle,
        receipts_dir=receipts,
        prediction_ledger_path=ledger_path,
        client=recovered_client,
        retry_failures=True,
    )
    assert recovered_client.generate_calls == 1
    assert recovered["status_counts"] == {"complete_schema_valid": 1}


def test_frozen_prediction_validation_happens_before_label_loader(
    evidence_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, bundle, provider_report, lexical_ledger = _prepared_fixture(
        evidence_fixture=evidence_fixture,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    output = {
        "eligible": False,
        "findings": [],
    }
    client = _FakeOllamaClient([output])
    receipts = tmp_path / "receipts"
    ledger_path = tmp_path / "prediction-ledger.json"
    ledger = run_prediction_stage(
        input_bundle=bundle,
        receipts_dir=receipts,
        prediction_ledger_path=ledger_path,
        client=client,
    )
    tampered = deepcopy(ledger)
    tampered["receipts"][0]["receipt_sha256"] = "f" * 64
    label_opened = False

    def trap_loader(*args: Any, **kwargs: Any) -> list[Any]:
        nonlocal label_opened
        del args, kwargs
        label_opened = True
        return []

    provider_path = tmp_path / "provider-report.json"
    lexical_path = tmp_path / "lexical-ledger.json"
    _write_json(provider_path, provider_report)
    _write_json(lexical_path, lexical_ledger)
    with pytest.raises(EvidenceInferenceOllamaError, match="ledger hash mismatch"):
        score_frozen_predictions(
            input_bundle=bundle,
            prediction_ledger=tampered,
            receipts_dir=receipts,
            manifest_path=manifest_path,
            provider_free_report_path=provider_path,
            lexical_prediction_ledger_path=lexical_path,
            replicates=100,
            label_loader=trap_loader,
        )
    assert label_opened is False


def test_scoring_public_redaction_and_paired_clustered_metrics(
    evidence_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, bundle, provider_report, lexical_ledger = _prepared_fixture(
        evidence_fixture=evidence_fixture,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    example = load_manifest_split(manifest_path, "test")[0]
    output, _ = ollama_diagnostic.lexical_diagnostic.lexical_extraction_output(
        ollama_diagnostic.lexical_diagnostic._label_stripped_input(example)
    )
    client = _FakeOllamaClient([output])
    receipts = tmp_path / "receipts"
    ledger = run_prediction_stage(
        input_bundle=bundle,
        receipts_dir=receipts,
        prediction_ledger_path=tmp_path / "prediction-ledger.json",
        client=client,
    )
    provider_path = tmp_path / "provider-report.json"
    lexical_path = tmp_path / "lexical-ledger.json"
    _write_json(provider_path, provider_report)
    _write_json(lexical_path, lexical_ledger)
    report = score_frozen_predictions(
        input_bundle=bundle,
        prediction_ledger=ledger,
        receipts_dir=receipts,
        manifest_path=manifest_path,
        provider_free_report_path=provider_path,
        lexical_prediction_ledger_path=lexical_path,
        replicates=100,
    )
    public = build_public_summary(report)

    assert validate_private_report(report) == report
    assert validate_public_summary(public) == public
    assert report["prediction_ledger_validated_before_label_file_open"] is True
    assert report["external_provider_calls"] == 0
    assert report["execution_inputs"]["input_bundle_file_sha256"] == (
        canonical_json_file_sha256(bundle)
    )
    assert report["execution_inputs"]["prediction_ledger_file_sha256"] == (
        canonical_json_file_sha256(ledger)
    )
    assert public["input_bundle_file_sha256"] == canonical_json_file_sha256(bundle)
    assert public["prediction_ledger_file_sha256"] == canonical_json_file_sha256(
        ledger
    )
    paired = public["paired_comparison"]["metrics"]["direction_accuracy"]
    assert paired["local_ollama_minus_fixed_lexical"]["estimate"] == 0.0
    assert sum(public["local_ollama"]["prediction_output_distribution"].values()) == 1
    assert sum(public["fixed_lexical"]["prediction_output_distribution"].values()) == 1
    serialized = json.dumps(public, sort_keys=True)
    assert "ei2-prompt" not in serialized
    assert "PMC" not in serialized
    assert "response_text" not in serialized

    tampered = deepcopy(public)
    tampered["raw_prediction"] = "PMC123 should never be public"
    payload = deepcopy(tampered)
    payload.pop("public_summary_sha256")
    tampered["public_summary_sha256"] = hash_canonical(payload)
    with pytest.raises(EvidenceInferenceOllamaError, match="protected field"):
        validate_public_summary(tampered)


def test_partial_ledger_cannot_be_scored() -> None:
    row_payload = {
        "example_id": "example-1",
        "paper_id": "paper-1",
        "query": {"INTERVENTION": "i", "COMPARATOR": "c", "OUTCOME": "o"},
        "source_accessible": True,
        "passages": [],
        "projection_sha256": hash_canonical([]),
    }
    row = {**row_payload, "input_row_sha256": hash_canonical(row_payload)}
    bundle_payload = {
        "input_bundle_version": ollama_diagnostic.INPUT_BUNDLE_VERSION,
        "status": "input_only_provider_call_unseen_non_pristine_diagnostic",
        "diagnostic_version": ollama_diagnostic.OLLAMA_DIAGNOSTIC_VERSION,
        "contains_gold_labels": False,
        "contains_expected_outputs": False,
        "contains_label_paths": False,
        "source_split_physically_colocates_inputs_and_labels": True,
        "prediction_stage_can_access_source_split": False,
        "test_labels_previously_opened": True,
        "test_split_pristine": False,
        "manifest_file_sha256": "a" * 64,
        "test_split_jsonl_sha256": "b" * 64,
        "provider_free_report_sha256": "c" * 64,
        "lexical_prediction_ledger_sha256": "d" * 64,
        "provider_call_unseen_subset_ledger_sha256": "e" * 64,
        "retrieval_config": deepcopy(ollama_diagnostic.DEFAULT_RETRIEVAL_CONFIG),
        "retrieval_config_sha256": hash_canonical(
            ollama_diagnostic.DEFAULT_RETRIEVAL_CONFIG
        ),
        "prompt_template_sha256": ollama_diagnostic.PROMPT_TEMPLATE_SHA256,
        "generation_schema_algorithm": (
            ollama_diagnostic.GENERATION_SCHEMA_ALGORITHM
        ),
        "generation_schema_sha256": ollama_diagnostic.GENERATION_SCHEMA_SHA256,
        "evaluation_schema_sha256": ollama_diagnostic.EVALUATION_SCHEMA_SHA256,
        "rows": [row],
        "row_count": 1,
        "article_count": 1,
    }
    bundle = {
        **bundle_payload,
        "input_bundle_sha256": hash_canonical(bundle_payload),
    }
    identity = _FakeOllamaClient([]).inspect_identity(DEFAULT_GENERATION_CONFIG)
    ledger = ollama_diagnostic._prediction_ledger_from_receipts(
        bundle=bundle,
        config=DEFAULT_GENERATION_CONFIG,
        identity=identity,
        receipt_rows=[],
    )
    with pytest.raises(EvidenceInferenceOllamaError, match="complete frozen"):
        validate_prediction_ledger(ledger, bundle=bundle, require_complete=True)


def test_ollama_generation_schema_uses_only_exposed_line_ids_and_no_regex() -> None:
    row = {
        "passages": [
            {"line_id": "L9", "text": "first"},
            {"line_id": "L3", "text": "second"},
            {"line_id": "L9", "text": "third"},
        ]
    }
    schema = ollama_diagnostic.generation_schema_for_row(row)
    line_items = schema["properties"]["findings"]["items"]["properties"][
        "evidence_lines"
    ]["items"]

    assert line_items["enum"] == ["L3", "L9"]
    assert "pattern" not in json.dumps(schema, sort_keys=True)
    assert schema["properties"]["findings"]["items"]["properties"][
        "evidence_lines"
    ]["maxItems"] == 1


def test_prediction_api_has_no_manifest_or_label_argument() -> None:
    parameters = set(inspect.signature(run_prediction_stage).parameters)

    assert "manifest_path" not in parameters
    assert "label_path" not in parameters
    assert "expected_output" not in parameters
