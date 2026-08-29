from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from literature_multiverse.config import load_question_config
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.local_ollama import (
    LocalOllamaError,
    OllamaGenerationConfig,
    OllamaGenerationResult,
    OllamaIdentity,
)
from literature_multiverse.native_extraction import (
    NativePublicationExtraction,
    native_extraction_prompt_replacements,
)
from literature_multiverse.native_grounding import TypedEvidenceGroundingPackage
from literature_multiverse.native_ollama_diagnostic import (
    DEFAULT_GENERATION_CONFIG,
    EXPECTED_CORPUS_CUTOFF,
    EXPECTED_QUESTION_ID,
    GENERATION_RECEIPT_VERSION,
    INPUT_BUNDLE_VERSION,
    NATIVE_OLLAMA_DIAGNOSTIC_VERSION,
    PREDICTION_LEDGER_VERSION,
    PRIVATE_REPORT_VERSION,
    PUBLIC_SUMMARY_VERSION,
    NativeOllamaDiagnosticError,
    _official_postvalidate,
    _projection_scope_issues,
    finalize_diagnostic,
    generation_schema_for_row,
    prepare_input_bundle,
    run_generation_schema_compatibility_preflight,
    run_prediction_stage,
    validate_current_diagnostic_context,
    validate_generation_receipt,
    validate_input_bundle,
    validate_prediction_ledger,
    validate_public_summary,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "benchmarks" / "native-antiox-ollama-v1.json"


class _FakeOllamaClient:
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = list(outputs)
        self.generate_calls = 0
        self.inspect_calls = 0
        self.prompts: list[str] = []
        self.schemas: list[dict[str, Any]] = []
        self.configs: list[OllamaGenerationConfig] = []

    def inspect_identity(self, config: OllamaGenerationConfig) -> OllamaIdentity:
        self.inspect_calls += 1
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
        self.prompts.append(prompt)
        self.schemas.append(output_schema)
        self.configs.append(config)
        output = self.outputs[self.generate_calls]
        self.generate_calls += 1
        response = output if isinstance(output, str) else json.dumps(output, sort_keys=True)
        return OllamaGenerationResult(
            model=config.model,
            response_text=response,
            done=True,
            done_reason="stop",
            total_duration_ns=100,
            load_duration_ns=10,
            prompt_eval_count=20,
            prompt_eval_duration_ns=30,
            eval_count=10,
            eval_duration_ns=60,
        )


class _PreflightOutageClient(_FakeOllamaClient):
    def __init__(self, outputs: list[Any], *, fail_on_inspect_call: int) -> None:
        super().__init__(outputs)
        self.fail_on_inspect_call = fail_on_inspect_call

    def inspect_identity(self, config: OllamaGenerationConfig) -> OllamaIdentity:
        if self.inspect_calls + 1 == self.fail_on_inspect_call:
            self.inspect_calls += 1
            raise LocalOllamaError("local Ollama request failed: ConnectionRefusedError")
        return super().inspect_identity(config)


class _RequestOutageClient(_FakeOllamaClient):
    def generate(
        self,
        *,
        prompt: str,
        output_schema: dict[str, Any],
        config: OllamaGenerationConfig,
    ) -> OllamaGenerationResult:
        self.generate_calls += 1
        raise LocalOllamaError("local Ollama request failed: ConnectionRefusedError")


@pytest.fixture(scope="module")
def native_bundle() -> dict[str, Any]:
    return prepare_input_bundle(config_path=CONFIG, repository_root=ROOT)


def _non_estimable_output() -> dict[str, Any]:
    return {
        "extraction_schema_version": "native-publication-extraction-v1",
        "status": "non_estimable",
        "studies": [],
        "non_estimability_reason": "numerical_result_absent",
        "non_estimability_detail": None,
        "warnings": [],
    }


def _estimable_output(row: dict[str, Any], *, exact_quote: bool) -> dict[str, Any]:
    passage = row["source_projection"][0]
    source_locator = row["source_record"]["source_document"]["source_locator"]
    quote = passage["text"] if exact_quote else "Invented result was 0.2 (SE 0.1)."
    return {
        "extraction_schema_version": "native-publication-extraction-v1",
        "status": "estimable",
        "studies": [
            {
                "key": "trial",
                "source_label": "reported trial",
                "design": "controlled trial",
                "registration_ids": [],
                "cohorts": [
                    {
                        "key": "cohort",
                        "source_labels": ["reported cohort"],
                        "registry_ids": [],
                        "dataset_ids": [],
                        "population_description": None,
                        "recruitment_period": None,
                        "total_sample_size": 20,
                        "arms": [
                            {
                                "key": "supplement",
                                "label": "vitamin supplement",
                                "role": "intervention",
                                "description": None,
                                "sample_size": 10,
                            },
                            {
                                "key": "control",
                                "label": "control",
                                "role": "control",
                                "description": None,
                                "sample_size": 10,
                            },
                        ],
                        "contrasts": [
                            {
                                "key": "primary",
                                "treatment_arm_key": "supplement",
                                "comparator_arm_key": "control",
                                "label": "supplement_vs_control",
                                "estimand": None,
                                "positive_direction_means": "higher aerobic adaptation",
                            }
                        ],
                        "findings": [
                            {
                                "key": "aerobic-result",
                                "contrast_key": "primary",
                                "outcome_name": "aerobic_capacity",
                                "timepoint": {
                                    "kind": "not_reported",
                                    "value": None,
                                    "lower": None,
                                    "upper": None,
                                    "unit": None,
                                    "anchor": None,
                                    "raw_label": None,
                                },
                                "analysis_population": None,
                                "effect": {
                                    "effect_format": "mean_difference",
                                    "availability": "available",
                                    "estimate": 0.2,
                                    "standard_error": 0.1,
                                    "variance": None,
                                    "ci_lower": None,
                                    "ci_upper": None,
                                    "ci_level": 0.95,
                                    "unit": "reported units",
                                    "treatment_mean": None,
                                    "treatment_sd": None,
                                    "treatment_n": None,
                                    "control_mean": None,
                                    "control_sd": None,
                                    "control_n": None,
                                    "treatment_events": None,
                                    "treatment_total": None,
                                    "control_events": None,
                                    "control_total": None,
                                    "reported_p_value": None,
                                    "reported_significance": "not_reported",
                                    "equivalence_conclusion": "not_tested",
                                    "equivalence_margin": None,
                                    "moderators": [],
                                    "extraction_method": "reported",
                                },
                                "evidence": {
                                    "source_locator": source_locator,
                                    "quote": quote,
                                    "section": passage["section"],
                                    "page": None,
                                    "char_start": None,
                                    "char_end": None,
                                    "line_ids": [passage["line_id"]],
                                },
                            }
                        ],
                    }
                ],
            }
        ],
        "non_estimability_reason": None,
        "non_estimability_detail": None,
        "warnings": [],
    }


def _contains_key(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(_contains_key(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, target) for item in value)
    return False


def _rehash(value: dict[str, Any], field: str) -> dict[str, Any]:
    payload = deepcopy(value)
    payload.pop(field, None)
    payload[field] = hash_canonical(payload)
    return payload


def test_prepare_is_exact_19_member_label_blind_bundle_and_schema_is_regex_free(
    native_bundle: dict[str, Any],
) -> None:
    validated = validate_input_bundle(native_bundle)

    assert validated["row_count"] == 19
    assert validated["source_manifest_records"] == 19
    assert validated["selection_scope"] == "legacy_eligible"
    assert validated["source_bridge_run_sha256"] == (
        "62d87d08b116d2af95da3e8646a293f9cdaf3507fd8b3e84319b2007367f26b3"
    )
    assert validated["corpus_cutoff"] == EXPECTED_CORPUS_CUTOFF
    assert validated["contains_legacy_findings"] is False
    assert validated["contains_legacy_directions"] is False
    assert validated["contains_anchor_expectations"] is False
    assert validated["contains_downstream_claim_payload"] is False
    assert validated["prompt_version"] == "native-extraction-v3"
    assert "outcomes.included_primary_endpoints" in validated["rendered_base_prompt"]
    assert "positive value" in validated["rendered_base_prompt"]
    assert "computed_from_reported_statistics" in validated["rendered_base_prompt"]
    assert {len(row["row_key"]) for row in validated["rows"]} == {64}
    assert all(
        not _contains_key(generation_schema_for_row(row), "pattern") for row in validated["rows"]
    )
    zero_projection_rows = [row for row in validated["rows"] if not row["source_projection"]]
    assert len(zero_projection_rows) == 2
    zero_schema = generation_schema_for_row(zero_projection_rows[0])
    line_enums: list[list[str]] = []

    def collect_line_enums(value: Any) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict) and isinstance(properties.get("line_ids"), dict):
                items = properties["line_ids"].get("items")
                if isinstance(items, dict) and isinstance(items.get("enum"), list):
                    line_enums.append(items["enum"])
            for item in value.values():
                collect_line_enums(item)
        elif isinstance(value, list):
            for item in value:
                collect_line_enums(item)

    collect_line_enums(zero_schema)
    assert line_enums
    assert set(map(tuple, line_enums)) == {("NO_EXPOSED_SOURCE_LINE",)}
    assert [branch["properties"]["status"]["enum"] for branch in zero_schema["oneOf"]] == [
        ["non_estimable"]
    ]
    Draft202012Validator.check_schema(zero_schema)
    Draft202012Validator(zero_schema).validate(_non_estimable_output())
    generation_schema = generation_schema_for_row(validated["rows"][0])
    Draft202012Validator.check_schema(generation_schema)
    assert generation_schema["oneOf"] == [
        {
            "title": "estimable publication evidence",
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["estimable"]},
                "studies": {"type": "array", "minItems": 1},
                "non_estimability_reason": {"type": "null"},
                "non_estimability_detail": {"type": "null"},
            },
            "required": ["status", "studies"],
        },
        {
            "title": "non-estimable publication evidence",
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["non_estimable"]},
                "studies": {"type": "array", "maxItems": 0},
                "non_estimability_reason": {"type": "string"},
            },
            "required": ["status", "non_estimability_reason"],
        },
    ]

    estimable_row = next(row for row in validated["rows"] if row["source_projection"])
    Draft202012Validator(generation_schema_for_row(estimable_row)).validate(
        _estimable_output(estimable_row, exact_quote=True)
    )
    impossible_zero_projection_estimable = _estimable_output(
        estimable_row,
        exact_quote=True,
    )
    impossible_zero_projection_estimable["studies"][0]["cohorts"][0]["findings"][0][
        "evidence"
    ]["source_locator"] = zero_projection_rows[0]["source_record"]["source_document"][
        "source_locator"
    ]
    impossible_zero_projection_estimable["studies"][0]["cohorts"][0]["findings"][0][
        "evidence"
    ]["line_ids"] = ["NO_EXPOSED_SOURCE_LINE"]
    assert list(
        Draft202012Validator(zero_schema).iter_errors(impossible_zero_projection_estimable)
    )

    tampered = deepcopy(validated)
    tampered["rows"][0]["expected_direction"] = "decrease"
    tampered["rows"][0] = _rehash(tampered["rows"][0], "input_row_sha256")
    tampered = _rehash(tampered, "input_bundle_sha256")
    with pytest.raises(NativeOllamaDiagnosticError, match="label_leak"):
        validate_input_bundle(tampered)

    unrecognized = deepcopy(validated)
    unrecognized["rows"][0]["gold_direction_alias"] = "decrease"
    unrecognized["rows"][0] = _rehash(unrecognized["rows"][0], "input_row_sha256")
    unrecognized = _rehash(unrecognized, "input_bundle_sha256")
    with pytest.raises(NativeOllamaDiagnosticError, match="row_invalid"):
        validate_input_bundle(unrecognized)


def test_closed_question_projection_matches_the_locked_question_yaml() -> None:
    diagnostic = json.loads(CONFIG.read_text(encoding="utf-8"))
    question = load_question_config(
        ROOT / diagnostic["question_config_path"],
        require_locked=True,
    )
    projected = json.loads(
        native_extraction_prompt_replacements(question)["QUESTION_SPEC_JSON"]
    )

    assert projected == diagnostic["question_spec"]
    assert hash_canonical(projected) == diagnostic["question_spec_sha256"]

    manifest = diagnostic["claim_manifest"]
    assert sorted(manifest["protocol"]["inclusion_criteria"]) == sorted(
        projected["eligibility"]["include"]
    )
    assert sorted(manifest["protocol"]["exclusion_criteria"]) == sorted(
        projected["eligibility"]["exclude"]
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("inclusion_criteria", ["different inclusion protocol"]),
        ("exclusion_criteria", ["different exclusion protocol"]),
    ],
)
def test_prepare_rejects_claim_protocol_drift_before_source_materialization(
    tmp_path: Path,
    field: str,
    replacement: list[str],
) -> None:
    diagnostic = json.loads(CONFIG.read_text(encoding="utf-8"))
    diagnostic["claim_manifest"]["protocol"][field] = replacement
    config_path = tmp_path / "drifted-config.json"
    config_path.write_text(json.dumps(diagnostic), encoding="utf-8")

    with pytest.raises(
        NativeOllamaDiagnosticError,
        match="native_diagnostic_config_invalid",
    ):
        prepare_input_bundle(config_path=config_path, repository_root=ROOT)


def test_current_execution_context_is_rechecked_before_any_generation(
    native_bundle: dict[str, Any],
) -> None:
    assert validate_current_diagnostic_context(native_bundle, repository_root=ROOT) == native_bundle

    tampered = deepcopy(native_bundle)
    execution = tampered["diagnostic_execution_identity"]
    script = next(
        item
        for item in execution["files"]
        if item["path"] == "scripts/run_native_ollama_diagnostic.py"
    )
    script["sha256"] = "f" * 64
    execution = _rehash(execution, "execution_sha256")
    tampered["diagnostic_execution_identity"] = execution
    tampered["diagnostic_execution_sha256"] = execution["execution_sha256"]
    tampered = _rehash(tampered, "input_bundle_sha256")

    with pytest.raises(NativeOllamaDiagnosticError, match="changed_after_prepare"):
        validate_current_diagnostic_context(tampered, repository_root=ROOT)


def test_projection_scope_accepts_multiple_projected_chunks_from_one_cited_line(
    native_bundle: dict[str, Any],
) -> None:
    row = next(
        row
        for row in native_bundle["rows"]
        if len({passage["line_id"] for passage in row["source_projection"]})
        < len(row["source_projection"])
    )
    counts: dict[str, int] = {}
    for passage in row["source_projection"]:
        counts[passage["line_id"]] = counts.get(passage["line_id"], 0) + 1
    passage = next(
        passage for passage in row["source_projection"] if counts[passage["line_id"]] > 1
    )
    payload = _estimable_output(row, exact_quote=True)
    evidence = payload["studies"][0]["cohorts"][0]["findings"][0]["evidence"]
    evidence["quote"] = passage["text"]
    evidence["section"] = passage["section"]
    evidence["line_ids"] = [passage["line_id"]]
    extraction = NativePublicationExtraction.model_validate(payload)

    assert _projection_scope_issues(extraction=extraction, row=row) == []


def test_official_postvalidation_rejects_json_type_coercion(
    native_bundle: dict[str, Any],
) -> None:
    row = next(row for row in native_bundle["rows"] if row["source_projection"])
    coercible = _estimable_output(row, exact_quote=True)
    coercible["studies"][0]["cohorts"][0]["findings"][0]["effect"]["estimate"] = "0.2"

    official, error = _official_postvalidate(coercible)

    assert official is None
    assert error is not None
    assert error.startswith("official_json_schema_validation_error:")


def test_expanded_schema_compatibility_preflight_is_source_and_label_free(
    native_bundle: dict[str, Any],
) -> None:
    client = _FakeOllamaClient([_non_estimable_output()])

    result = run_generation_schema_compatibility_preflight(client=client)

    assert result["status"] == "passed"
    assert result["contains_publication_content"] is False
    assert result["contains_scientific_claim"] is False
    assert result["contains_source_text"] is False
    assert result["contains_eligibility_or_answer_labels"] is False
    assert result["paper_prediction_receipt_written"] is False
    assert client.inspect_calls == 1
    assert client.generate_calls == 1
    assert client.configs[0].num_predict == 256
    assert all(branch.get("type") == "object" for branch in client.schemas[0]["oneOf"])
    Draft202012Validator.check_schema(client.schemas[0])

    serialized_request = json.dumps(
        {"prompt": client.prompts[0], "schema": client.schemas[0]},
        sort_keys=True,
    )
    for row in native_bundle["rows"]:
        source_document = row["source_record"]["source_document"]
        assert source_document["source_locator"] not in serialized_request
        for passage in row["source_projection"]:
            assert passage["line_id"] not in serialized_request
            assert passage["text"] not in serialized_request
    assert "expected_directions" not in serialized_request
    assert "anchor_papers" not in serialized_request
    assert "legacy_findings" not in serialized_request


def test_expanded_schema_compatibility_failure_precedes_scientific_rows() -> None:
    client = _RequestOutageClient([])

    with pytest.raises(
        NativeOllamaDiagnosticError,
        match="schema_compatibility_preflight_failed;no_scientific_row_request_was_made",
    ):
        run_generation_schema_compatibility_preflight(client=client)

    assert client.generate_calls == 1


def test_prediction_freezes_official_postvalidation_and_never_retries_terminal_rows(
    native_bundle: dict[str, Any],
    tmp_path: Path,
) -> None:
    rows_with_text = [row for row in native_bundle["rows"] if row["source_projection"]]
    exact_key = rows_with_text[0]["row_key"]
    mismatch_key = rows_with_text[1]["row_key"]
    outputs: list[Any] = []
    invalid_key: str | None = None
    for row in native_bundle["rows"]:
        if row["row_key"] == exact_key:
            outputs.append(_estimable_output(row, exact_quote=True))
        elif row["row_key"] == mismatch_key:
            outputs.append(_estimable_output(row, exact_quote=False))
        elif invalid_key is None:
            invalid_key = row["row_key"]
            outputs.append({"status": "estimable", "studies": []})
        else:
            outputs.append(_non_estimable_output())
    client = _FakeOllamaClient(outputs)
    receipts_dir = tmp_path / "receipts"
    ledger_path = tmp_path / "prediction-ledger.json"

    ledger = run_prediction_stage(
        input_bundle=native_bundle,
        receipts_dir=receipts_dir,
        prediction_ledger_path=ledger_path,
        client=client,
    )

    assert client.generate_calls == 19
    assert ledger["all_expected_receipts_frozen"] is True
    assert ledger["status_counts"] == {
        "official_schema_invalid": 1,
        "official_schema_valid": 18,
    }
    assert all(not _contains_key(schema, "pattern") for schema in client.schemas)
    rendered_prompts = "\n".join(client.prompts)
    assert "expected_directions" not in rendered_prompts
    assert "anchor_papers" not in rendered_prompts
    assert "legacy_findings" not in rendered_prompts

    no_retry = _FakeOllamaClient([])
    resumed = run_prediction_stage(
        input_bundle=native_bundle,
        receipts_dir=receipts_dir,
        prediction_ledger_path=ledger_path,
        client=no_retry,
    )
    assert no_retry.generate_calls == 0
    assert resumed == ledger

    invalid_summary = next(
        row for row in ledger["receipts"] if row["status"] == "official_schema_invalid"
    )
    receipt_path = receipts_dir / f"{invalid_summary['row_key'][:32]}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    row = next(row for row in native_bundle["rows"] if row["row_key"] == receipt["row_key"])
    identity = no_retry.inspect_identity(DEFAULT_GENERATION_CONFIG)
    forged = deepcopy(receipt)
    forged["status"] = "official_schema_valid"
    forged = _rehash(forged, "receipt_sha256")
    with pytest.raises(NativeOllamaDiagnosticError, match="status_mismatch"):
        validate_generation_receipt(
            forged,
            bundle=native_bundle,
            row=row,
            config=DEFAULT_GENERATION_CONFIG,
            identity=identity,
        )

    private, public = finalize_diagnostic(
        input_bundle=native_bundle,
        prediction_ledger=ledger,
        receipts_dir=receipts_dir,
        repository_root=ROOT,
        private_output_dir=tmp_path / "final",
        force=False,
    )
    package = TypedEvidenceGroundingPackage.model_validate_json(
        (tmp_path / "final" / "typed-evidence-grounding-package.json").read_text()
    )
    assert package.package_version == "typed-evidence-grounding-package-v4"
    assert package.corpus.corpus_version == "typed-evidence-corpus-v3"
    assert package.extraction_context_receipt is not None
    extraction_context = package.extraction_context_receipt.execution_context
    assert extraction_context.extraction_mode == "ollama_local"
    assert extraction_context.question_config.question_id == EXPECTED_QUESTION_ID
    assert len(extraction_context.rendered_prompts) == 19
    assert sum(
        receipt.call_count
        for receipt in extraction_context.provider_execution_receipts
    ) == 19
    assert (
        json.loads(
            (tmp_path / "final" / "native-extraction-context.json").read_text()
        )["context_sha256"]
        == extraction_context.context_sha256
    )
    assert package.corpus_cutoff == EXPECTED_CORPUS_CUTOFF
    assert package.source_manifest is not None
    assert len(package.source_manifest.records) == 19
    assert len(package.corpus.fragments) == 19
    assert private["extraction"]["official_estimable_attempts"] == 2
    assert private["extraction"]["downgraded_estimable_attempts"] == 1
    assert private["extraction"]["authorizing_receipts"] == 1
    assert private["extraction"]["downgrade_cause_counts"] == {
        "projection_scope:quote_not_exact_exposed_passage": 1,
        "projection_scope:section_not_exposed_quote_passage": 1,
    }
    assert private["certificate_status"] == "abstained"
    assert private["certificate_version"] == "literature-multiverse-verification-v5"
    assert private["certificate_run_id"].startswith("verify-")
    assert private["certificate_reasons"]
    assert len(private["complete_corpus_membership_sha256"]) == 64
    assert private["synthesis_mode"] == "evidence_graph_contract"
    assert private["synthesis_status"] == "insufficient"
    assert private["synthesis_reason"] == "timepoint_not_reported"
    assert public["certificate_reasons"] == private["certificate_reasons"]
    assert validate_public_summary(public) == public
    serialized = json.dumps(public, sort_keys=True)
    assert "PMC" not in serialized
    assert "source_locator" not in serialized
    assert "response_text" not in serialized
    assert str(ROOT) not in serialized

    leaked = deepcopy(public)
    leaked["generation"]["raw_model_answer"] = "not aggregate data"
    leaked = _rehash(leaked, "public_summary_sha256")
    with pytest.raises(NativeOllamaDiagnosticError, match="generation_counts_invalid"):
        validate_public_summary(leaked)

    malformed_reasons = deepcopy(public)
    malformed_reasons["certificate_reasons"] = [{}]
    malformed_reasons = _rehash(malformed_reasons, "public_summary_sha256")
    with pytest.raises(NativeOllamaDiagnosticError, match="certificate_reasons_invalid"):
        validate_public_summary(malformed_reasons)

    sensitive_reason = deepcopy(public)
    sensitive_reason["certificate_reasons"] = sorted(
        {*sensitive_reason["certificate_reasons"], "unreviewed DOI 10.1234/private"}
    )
    sensitive_reason = _rehash(sensitive_reason, "public_summary_sha256")
    with pytest.raises(NativeOllamaDiagnosticError, match="sensitive_value"):
        validate_public_summary(sensitive_reason)


def test_runtime_outage_stops_without_freezing_paper_failure_and_resume_is_exact(
    native_bundle: dict[str, Any],
    tmp_path: Path,
) -> None:
    receipts_dir = tmp_path / "receipts"
    ledger_path = tmp_path / "prediction-ledger.json"
    outage = _PreflightOutageClient(
        [_non_estimable_output() for _ in range(19)],
        # One run-level inspection, then two successful per-row preflights.
        fail_on_inspect_call=4,
    )

    with pytest.raises(
        NativeOllamaDiagnosticError,
        match="runtime_unavailable_before_request;no_row_receipt_was_frozen",
    ):
        run_prediction_stage(
            input_bundle=native_bundle,
            receipts_dir=receipts_dir,
            prediction_ledger_path=ledger_path,
            client=outage,
        )

    frozen_before_resume = {
        path.name: path.read_bytes() for path in sorted(receipts_dir.glob("*.json"))
    }
    assert outage.generate_calls == 2
    assert len(frozen_before_resume) == 2
    assert not ledger_path.exists()

    healthy = _FakeOllamaClient([_non_estimable_output() for _ in range(17)])
    ledger = run_prediction_stage(
        input_bundle=native_bundle,
        receipts_dir=receipts_dir,
        prediction_ledger_path=ledger_path,
        client=healthy,
    )

    assert healthy.generate_calls == 17
    assert ledger["all_expected_receipts_frozen"] is True
    assert ledger["status_counts"] == {"official_schema_valid": 19}
    assert all(
        (receipts_dir / name).read_bytes() == contents
        for name, contents in frozen_before_resume.items()
    )


def test_request_transport_failure_is_not_frozen_as_scientific_row_outcome(
    native_bundle: dict[str, Any],
    tmp_path: Path,
) -> None:
    receipts_dir = tmp_path / "receipts"
    ledger_path = tmp_path / "prediction-ledger.json"
    client = _RequestOutageClient([])

    with pytest.raises(
        NativeOllamaDiagnosticError,
        match="transport_failed_without_model_response;no_row_receipt_was_frozen",
    ):
        run_prediction_stage(
            input_bundle=native_bundle,
            receipts_dir=receipts_dir,
            prediction_ledger_path=ledger_path,
            client=client,
        )

    assert client.generate_calls == 1
    assert list(receipts_dir.glob("*.json")) == []
    assert not ledger_path.exists()


def test_prediction_rejects_unexpected_receipt_files_before_generation(
    native_bundle: dict[str, Any],
    tmp_path: Path,
) -> None:
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    (receipts_dir / "unrecognized.json").write_text("{}", encoding="utf-8")
    client = _FakeOllamaClient([_non_estimable_output()])

    with pytest.raises(NativeOllamaDiagnosticError, match="receipt_files_unexpected"):
        run_prediction_stage(
            input_bundle=native_bundle,
            receipts_dir=receipts_dir,
            prediction_ledger_path=tmp_path / "ledger.json",
            client=client,
            limit=1,
        )
    assert client.generate_calls == 0


def test_all_19_terminal_non_estimable_outputs_still_form_complete_v4_package(
    native_bundle: dict[str, Any],
    tmp_path: Path,
) -> None:
    receipts_dir = tmp_path / "receipts"
    ledger_path = tmp_path / "prediction-ledger.json"
    client = _FakeOllamaClient([_non_estimable_output() for _ in range(19)])
    ledger = run_prediction_stage(
        input_bundle=native_bundle,
        receipts_dir=receipts_dir,
        prediction_ledger_path=ledger_path,
        client=client,
    )
    validate_prediction_ledger(ledger, bundle=native_bundle, require_complete=True)

    private, public = finalize_diagnostic(
        input_bundle=native_bundle,
        prediction_ledger=ledger,
        receipts_dir=receipts_dir,
        repository_root=ROOT,
        private_output_dir=tmp_path / "final",
        force=False,
    )
    package = TypedEvidenceGroundingPackage.model_validate_json(
        (tmp_path / "final" / "typed-evidence-grounding-package.json").read_text()
    )

    assert len(package.corpus.fragments) == 19
    assert package.package_version == "typed-evidence-grounding-package-v4"
    assert package.extraction_context_receipt is not None
    assert len(package.grounding_receipts) == 19
    assert package.corpus.estimable_publication_ids == []
    assert len(package.corpus.non_estimable_publication_ids) == 19
    assert len(package.corpus.graph.publications) == 19
    assert package.corpus.graph.studies == []
    assert package.corpus.graph.outcome_estimates == []
    assert private["graph_counts"]["publications"] == 19
    assert private["graph_counts"]["outcome_estimates"] == 0
    assert private["certificate_status"] == "abstained"
    assert private["synthesis_status"] == "insufficient"
    assert private["synthesis_reason"] == "no_matching_estimates"
    assert public["population_count"] == 19
    assert validate_public_summary(public) == public


def test_native_ollama_cli_help_is_available() -> None:
    from scripts.run_native_ollama_diagnostic import build_parser

    parser = build_parser()
    help_text = parser.format_help()
    assert "prepare" in help_text
    assert "predict" in help_text
    assert "finalize" in help_text
    assert parser.parse_args(["predict"]).workspace == Path(
        "data/cache/native-antiox-ollama-v2-final-v1"
    )


def test_native_contract_versions_distinguish_changed_from_unchanged_schemas() -> None:
    assert NATIVE_OLLAMA_DIAGNOSTIC_VERSION.endswith("diagnostic-v2")
    assert PREDICTION_LEDGER_VERSION == "native-ollama-prediction-ledger-v2"
    assert PRIVATE_REPORT_VERSION == "native-ollama-private-report-v2"
    assert PUBLIC_SUMMARY_VERSION == "native-ollama-public-summary-v2"
    assert INPUT_BUNDLE_VERSION == "native-ollama-input-bundle-v1"
    assert GENERATION_RECEIPT_VERSION == "native-ollama-generation-receipt-v1"


def test_prepare_refuses_a_workspace_with_prediction_artifacts(tmp_path: Path) -> None:
    from scripts.run_native_ollama_diagnostic import (
        _assert_prepare_workspace_has_no_predictions,
    )

    clean = tmp_path / "clean"
    _assert_prepare_workspace_has_no_predictions(clean)
    receipts = clean / "generation-receipts"
    receipts.mkdir(parents=True)
    with pytest.raises(ValueError, match="choose a fresh ignored workspace"):
        _assert_prepare_workspace_has_no_predictions(clean)
