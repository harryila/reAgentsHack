"""Immutable, non-authorizing diagnosis of the contextual frontier v1 live run.

This audit never changes or retries the exact-once v1 requests.  It binds the
durable plan, terminal report, provider receipts, and post-call validations;
then it asks whether JSON/claim-order canonicalization alone can recover either
response.  It deliberately separates three questions that the v1 validator
conflated:

* provider/transport completion;
* exact source-grounding contract validity; and
* agreement with one code-owned single-contrast target.

The resulting artifact is a failure analysis, not extraction-accuracy evidence
and not a synthesis or release input.
"""

from __future__ import annotations

import json
import os
import stat
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, model_validator

from literature_multiverse.contextual_numeric_grounding_v3 import (
    _BINARY_REQUIRED_PATHS,
    project_contextual_grounded_outcome_v3,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file
from literature_multiverse.metasyn_contextual_frontier_runtime_v1 import (
    MetaSynContextualFrontierPlanV1,
    MetaSynContextualFrontierProviderReceiptV1,
    MetaSynContextualFrontierTerminalReportV1,
    MetaSynContextualFrontierValidationResultV1,
)
from literature_multiverse.models import SHA256_RE, ContractModel

AUDIT_VERSION = "metasyn-contextual-frontier-v1-failure-audit-v1"
REQUEST_AUDIT_VERSION = "metasyn-contextual-frontier-v1-request-failure-audit-v1"
DEFAULT_V1_WORKSPACE = Path("data/cache/metasyn/contextual-frontier-runtime-v1")
DEFAULT_OUTPUT_PATH = Path(
    "artifacts/diagnostics/metasyn-contextual-frontier-v1-failure-audit-v1.json"
)

_RUNTIME_MODULE_PATH = Path("src/literature_multiverse/metasyn_contextual_frontier_runtime_v1.py")
_GROUNDING_MODULE_PATH = Path("src/literature_multiverse/contextual_numeric_grounding_v3.py")
_REQUIRED_BINARY_PATHS = tuple(sorted(_BINARY_REQUIRED_PATHS))
_EXPECTED_REQUEST_KEYS = (
    "row17-candidate3-fable5-high",
    "row17-candidate2-fable5-high",
)
_SORT_ONLY_FAILURE_CODES = (
    "contextual_grounding_v3_binary_contract_mismatch",
    "contextual_grounding_v3_endpoint_marker_not_exact",
)
_RECOVERY_REQUIREMENTS = (
    "claims_must_be_keyed_by_exact_required_field_or_sorted_before_local_validation",
    "endpoint_result_and_endpoint_definition_roles_must_be_explicit",
    "expected_numeric_values_must_remain_hidden_from_provider",
    "fresh_v2_plan_pipeline_identity_and_exact_once_calls_required",
    "grounding_validity_and_hidden_accuracy_scoring_must_be_separate",
    "immutable_v1_remains_terminal_failed_and_must_not_be_retried",
    "source_unicode_must_be_exact_or_any_alignment_correction_must_be_recorded",
    "target_contrast_must_come_from_upstream_protocol_or_all_contrasts_must_be_enumerated",
    "validator_must_accept_frozen_provider_visible_passages_not_only_fixture_selected_passages",
)

Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]


class MetaSynContextualFrontierV1FailureAuditError(ValueError):
    """The immutable live archive or its failure diagnosis did not replay."""


class _FrozenExact(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        str_strip_whitespace=False,
    )


class BoundArtifactV1(_FrozenExact):
    relative_path: str
    byte_count: Annotated[int, Field(ge=1)]
    file_sha256: Sha256
    embedded_sha256: Sha256


class RequestFailureAuditV1(_FrozenExact):
    request_audit_version: Literal["metasyn-contextual-frontier-v1-request-failure-audit-v1"] = (
        REQUEST_AUDIT_VERSION
    )
    request_key: str
    witness_id: str
    request_sha256: Sha256
    provider_receipt_file_sha256: Sha256
    provider_receipt_sha256: Sha256
    provider_result_sha256: Sha256
    validation_file_sha256: Sha256
    validation_sha256: Sha256
    provider_outcome: Literal["completed"]
    response_model: Literal["claude-fable-5"]
    stop_reason: Literal["end_turn"]
    structured_json_completed: Literal[True] = True
    transport_attempt_count: Literal[1] = 1
    sdk_retry_count: Literal[0] = 0
    input_tokens: Annotated[int, Field(gt=0)]
    output_tokens: Annotated[int, Field(gt=0)]
    estimated_cost_usd_micros: Annotated[int, Field(gt=0)]
    charged_cost_ceiling_usd_micros: Annotated[int, Field(gt=0)]
    archived_validation_status: Literal["contextual_validation_failed_closed"]
    archived_failure_code: Literal["contextual_grounding_or_projection_failed_closed"]

    raw_claim_count: Annotated[int, Field(ge=1, le=32)]
    raw_claim_order_canonical: Literal[False] = False
    canonical_sort_applied_in_memory_only: Literal[True] = True
    canonical_sort_mutated_archive: Literal[False] = False
    sort_only_salvage_succeeded: Literal[False] = False
    sort_only_failure_code: Literal[
        "contextual_grounding_v3_binary_contract_mismatch",
        "contextual_grounding_v3_endpoint_marker_not_exact",
    ]
    required_binary_field_paths: tuple[str, ...]
    predicted_field_paths: tuple[str, ...]
    missing_required_field_paths: tuple[str, ...]
    extra_nonbinary_field_paths: tuple[str, ...]
    provider_schema_enforced_exact_binary_field_set: Literal[False] = False
    provider_schema_enforced_claim_order: Literal[False] = False

    provider_visible_passage_count: Annotated[int, Field(gt=0)]
    fixture_selected_passage_count: Annotated[int, Field(gt=0)]
    fixture_passage_surface_is_strict_subset: Literal[True] = True
    predicted_passage_ids: tuple[str, ...]
    predicted_provider_visible_but_fixture_excluded_passage_ids: tuple[str, ...]
    provider_visible_but_fixture_excluded_prediction: Literal[True] = True
    nonexact_support_quote_field_paths: tuple[str, ...]
    endpoint_quote_exact_in_provider_visible_passage: bool

    upstream_protocol_prespecified_exact_dose: Literal[False] = False
    source_exposes_multiple_active_doses: Literal[True] = True
    predicted_treatment_arm: str
    predicted_treatment_events: str
    predicted_treatment_total: str
    code_owned_target_treatment_arm: str
    code_owned_target_treatment_events: str
    code_owned_target_treatment_total: str
    predicted_arm_differs_from_code_owned_target: Literal[True] = True
    single_contrast_target_ambiguous_without_protocol_arm: Literal[True] = True
    prediction_vs_code_owned_target_mismatch_field_paths: tuple[str, ...]
    code_owned_target_is_not_provider_visible_accuracy_label: Literal[True] = True

    canonicalization_only_recovery_supported: Literal[False] = False
    value_or_provenance_rewrite_would_be_required: Literal[True] = True
    immutable_response_reclassified_as_success: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    request_audit_sha256: Sha256

    @model_validator(mode="after")
    def validate_request_audit(self) -> RequestFailureAuditV1:
        if self.required_binary_field_paths != _REQUIRED_BINARY_PATHS:
            raise ValueError("contextual_frontier_failure_audit_required_paths_drift")
        if self.sort_only_failure_code not in _SORT_ONLY_FAILURE_CODES:
            raise ValueError("contextual_frontier_failure_audit_sort_failure_unknown")
        if not self.predicted_provider_visible_but_fixture_excluded_passage_ids:
            raise ValueError("contextual_frontier_failure_audit_passage_mismatch_missing")
        if not self.prediction_vs_code_owned_target_mismatch_field_paths:
            raise ValueError("contextual_frontier_failure_audit_target_mismatch_missing")
        body = self.model_dump(mode="json", exclude={"request_audit_sha256"})
        if self.request_audit_sha256 != hash_canonical(body):
            raise ValueError("contextual_frontier_failure_audit_request_hash_mismatch")
        return self


class MetaSynContextualFrontierV1FailureAudit(_FrozenExact):
    audit_version: Literal["metasyn-contextual-frontier-v1-failure-audit-v1"] = AUDIT_VERSION
    status: Literal["audited_no_canonicalization_only_salvage"]
    source_workspace_relative_path: str
    bound_artifacts: tuple[BoundArtifactV1, ...]
    runtime_module_sha256: Sha256
    grounding_module_sha256: Sha256
    plan_sha256: Sha256
    runtime_pipeline_sha256: Sha256
    terminal_report_sha256: Sha256
    terminal_status: Literal["roster_exhausted_without_typed_graph"]
    provider_receipt_count: Literal[2] = 2
    structured_completed_response_count: Literal[2] = 2
    typed_graph_completed_response_count: Literal[0] = 0
    total_input_tokens: Annotated[int, Field(gt=0)]
    total_output_tokens: Annotated[int, Field(gt=0)]
    total_estimated_cost_usd_micros: Annotated[int, Field(gt=0)]
    total_charged_cost_ceiling_usd_micros: Annotated[int, Field(gt=0)]
    request_audits: tuple[RequestFailureAuditV1, RequestFailureAuditV1]
    request_audit_membership_sha256: Sha256

    immutable_v1_preserved: Literal[True] = True
    provider_or_transport_failure_explains_result: Literal[False] = False
    structured_json_failure_explains_result: Literal[False] = False
    claim_order_rejection_contributed_to_both_failures: Literal[True] = True
    canonical_sort_alone_salvages_any_response: Literal[False] = False
    fixture_selected_passage_surface_mismatch_contributed: Literal[True] = True
    protocol_arm_ambiguity_contributed: Literal[True] = True
    expected_numeric_values_exposed_to_provider: Literal[False] = False
    exposing_expected_numeric_values_would_invalidate_independent_extraction_claim: Literal[
        True
    ] = True
    exact_arm_may_be_provider_input_only_if_prespecified_upstream: Literal[True] = True
    otherwise_all_source_supported_contrasts_must_be_enumerated: Literal[True] = True
    structural_required_field_roster_may_be_provider_visible: Literal[True] = True
    grounding_validity_must_be_separate_from_hidden_target_scoring: Literal[True] = True
    safe_recovery_requirements: tuple[str, ...]

    source_scope: Literal["title_abstract_not_release_grade"]
    official_test_labels_opened: Literal[False] = False
    reference_accuracy_labels_opened: Literal[False] = False
    new_provider_calls_made: Literal[0] = 0
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    audit_sha256: Sha256

    @model_validator(mode="after")
    def validate_audit(self) -> MetaSynContextualFrontierV1FailureAudit:
        if tuple(item.request_key for item in self.request_audits) != _EXPECTED_REQUEST_KEYS:
            raise ValueError("contextual_frontier_failure_audit_request_order_mismatch")
        if self.safe_recovery_requirements != _RECOVERY_REQUIREMENTS:
            raise ValueError("contextual_frontier_failure_audit_recovery_contract_drift")
        if tuple(item.relative_path for item in self.bound_artifacts) != tuple(
            sorted(item.relative_path for item in self.bound_artifacts)
        ):
            raise ValueError("contextual_frontier_failure_audit_artifacts_not_canonical")
        if len({item.relative_path for item in self.bound_artifacts}) != len(self.bound_artifacts):
            raise ValueError("contextual_frontier_failure_audit_artifact_duplicate")
        if self.total_input_tokens != sum(item.input_tokens for item in self.request_audits):
            raise ValueError("contextual_frontier_failure_audit_input_tokens_mismatch")
        if self.total_output_tokens != sum(item.output_tokens for item in self.request_audits):
            raise ValueError("contextual_frontier_failure_audit_output_tokens_mismatch")
        if self.total_estimated_cost_usd_micros != sum(
            item.estimated_cost_usd_micros for item in self.request_audits
        ):
            raise ValueError("contextual_frontier_failure_audit_cost_mismatch")
        if self.total_charged_cost_ceiling_usd_micros != sum(
            item.charged_cost_ceiling_usd_micros for item in self.request_audits
        ):
            raise ValueError("contextual_frontier_failure_audit_ceiling_mismatch")
        expected_membership = hash_canonical(
            [item.request_audit_sha256 for item in self.request_audits]
        )
        if self.request_audit_membership_sha256 != expected_membership:
            raise ValueError("contextual_frontier_failure_audit_membership_hash_mismatch")
        body = self.model_dump(mode="json", exclude={"audit_sha256"})
        if self.audit_sha256 != hash_canonical(body):
            raise ValueError("contextual_frontier_failure_audit_hash_mismatch")
        return self


def _checked_json(path: Path) -> tuple[dict[str, Any], str, int]:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise MetaSynContextualFrontierV1FailureAuditError(
            "contextual_frontier_failure_audit_artifact_missing"
        ) from exc
    if not stat.S_ISREG(mode):
        raise MetaSynContextualFrontierV1FailureAuditError(
            "contextual_frontier_failure_audit_artifact_not_regular"
        )
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynContextualFrontierV1FailureAuditError(
            "contextual_frontier_failure_audit_artifact_not_json"
        ) from exc
    if not isinstance(value, dict):
        raise MetaSynContextualFrontierV1FailureAuditError(
            "contextual_frontier_failure_audit_artifact_not_object"
        )
    return value, sha256_file(path), len(raw)


def _usd_micros(value: str) -> int:
    micros = Decimal(value) * Decimal(1_000_000)
    integral = micros.to_integral_exact()
    if micros != integral or integral <= 0:
        raise MetaSynContextualFrontierV1FailureAuditError(
            "contextual_frontier_failure_audit_cost_not_exact_micros"
        )
    return int(integral)


def _exception_chain(exc: BaseException) -> tuple[str, ...]:
    output: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        output.append(str(current))
        current = current.__cause__
    return tuple(output)


def _sort_only_failure_code(
    *,
    item: Any,
    raw_outcome: dict[str, Any],
    runtime_pipeline_sha256: str,
    provider_execution_binding_sha256: str,
) -> str:
    sorted_outcome = deepcopy(raw_outcome)
    sorted_outcome["claims"] = sorted(
        sorted_outcome["claims"],
        key=lambda claim: (claim["field_path"], claim["passage_id"]),
    )
    try:
        project_contextual_grounded_outcome_v3(
            fixture_receipt=item.offline_witness,
            raw_outcome=sorted_outcome,
            runtime_pipeline_sha256=runtime_pipeline_sha256,
            provider_execution_binding_sha256=provider_execution_binding_sha256,
        )
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
            raise
        chain = _exception_chain(exc)
        matches = [code for code in _SORT_ONLY_FAILURE_CODES if any(code in x for x in chain)]
        if len(matches) != 1:
            raise MetaSynContextualFrontierV1FailureAuditError(
                "contextual_frontier_failure_audit_sort_replay_unclassified"
            ) from exc
        return matches[0]
    raise MetaSynContextualFrontierV1FailureAuditError(
        "contextual_frontier_failure_audit_unexpected_sort_only_salvage"
    )


def _embedded_binding(
    *, workspace: Path, relative_path: Path, embedded_sha256: str
) -> BoundArtifactV1:
    _, file_sha256, byte_count = _checked_json(workspace / relative_path)
    return BoundArtifactV1(
        relative_path=relative_path.as_posix(),
        byte_count=byte_count,
        file_sha256=file_sha256,
        embedded_sha256=embedded_sha256,
    )


def _request_audit(
    *,
    item: Any,
    receipt: MetaSynContextualFrontierProviderReceiptV1,
    receipt_file_sha256: str,
    validation: MetaSynContextualFrontierValidationResultV1,
    validation_file_sha256: str,
    runtime_pipeline_sha256: str,
) -> RequestFailureAuditV1:
    result = receipt.provider_result
    if (
        result.outcome != "completed"
        or result.parsed_json is None
        or result.response_model != "claude-fable-5"
        or result.stop_reason != "end_turn"
        or result.usage is None
        or result.estimated_cost_usd is None
        or result.charged_cost_upper_bound_usd is None
    ):
        raise MetaSynContextualFrontierV1FailureAuditError(
            "contextual_frontier_failure_audit_provider_result_not_completed"
        )
    raw = result.parsed_json
    claims = raw.get("claims")
    if not isinstance(claims, list) or not claims:
        raise MetaSynContextualFrontierV1FailureAuditError(
            "contextual_frontier_failure_audit_claims_missing"
        )
    keys = [(claim["field_path"], claim["passage_id"]) for claim in claims]
    if keys == sorted(set(keys)):
        raise MetaSynContextualFrontierV1FailureAuditError(
            "contextual_frontier_failure_audit_claims_unexpectedly_canonical"
        )
    predicted_paths = tuple(sorted(claim["field_path"] for claim in claims))
    predicted_path_set = set(predicted_paths)
    required = set(_REQUIRED_BINARY_PATHS)

    provider_passages = {
        passage.passage_id: passage
        for passage in item.offline_witness.provider_binding.context.passages
    }
    fixture_passage_ids = {passage.passage_id for passage in item.offline_witness.passages}
    predicted_passage_ids = tuple(sorted({claim["passage_id"] for claim in claims}))
    provider_only_predictions = tuple(sorted(set(predicted_passage_ids) - fixture_passage_ids))
    if any(passage_id not in provider_passages for passage_id in predicted_passage_ids):
        raise MetaSynContextualFrontierV1FailureAuditError(
            "contextual_frontier_failure_audit_prediction_not_provider_visible"
        )
    nonexact_quote_paths = tuple(
        sorted(
            claim["field_path"]
            for claim in claims
            if provider_passages[claim["passage_id"]].text.count(claim["support_quote"]) != 1
        )
    )
    endpoint_passage = provider_passages.get(raw["endpoint_passage_id"])
    endpoint_exact = (
        endpoint_passage is not None and endpoint_passage.text.count(raw["endpoint_quote"]) == 1
    )

    by_path = {claim["field_path"]: claim for claim in claims}
    target = item.offline_witness.semantic_target.expected_normalized_values
    mismatches = tuple(
        sorted(
            path
            for path in _REQUIRED_BINARY_PATHS
            if path not in by_path or by_path[path]["token"] != target[path]
        )
    )
    protocol_text = " ".join(
        (
            item.offline_witness.provider_binding.context.intervention_or_exposure,
            item.offline_witness.provider_binding.context.comparison,
        )
    )
    if "400" in protocol_text or "500" in protocol_text:
        raise MetaSynContextualFrontierV1FailureAuditError(
            "contextual_frontier_failure_audit_protocol_unexpectedly_dose_specific"
        )
    endpoint_text = endpoint_passage.text if endpoint_passage is not None else ""
    if "400-mg" not in endpoint_text or "500-mg" not in endpoint_text:
        raise MetaSynContextualFrontierV1FailureAuditError(
            "contextual_frontier_failure_audit_multiple_doses_not_visible"
        )

    sort_failure = _sort_only_failure_code(
        item=item,
        raw_outcome=raw,
        runtime_pipeline_sha256=runtime_pipeline_sha256,
        provider_execution_binding_sha256=validation.provider_execution_binding_sha256,
    )
    payload = {
        "request_audit_version": REQUEST_AUDIT_VERSION,
        "request_key": item.request.request_key,
        "witness_id": item.witness_id,
        "request_sha256": item.request_sha256,
        "provider_receipt_file_sha256": receipt_file_sha256,
        "provider_receipt_sha256": receipt.receipt_sha256,
        "provider_result_sha256": result.result_sha256,
        "validation_file_sha256": validation_file_sha256,
        "validation_sha256": validation.validation_sha256,
        "provider_outcome": result.outcome,
        "response_model": result.response_model,
        "stop_reason": result.stop_reason,
        "structured_json_completed": True,
        "transport_attempt_count": result.transport_attempt_count,
        "sdk_retry_count": result.sdk_retry_count,
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
        "estimated_cost_usd_micros": _usd_micros(result.estimated_cost_usd),
        "charged_cost_ceiling_usd_micros": _usd_micros(result.charged_cost_upper_bound_usd),
        "archived_validation_status": validation.status,
        "archived_failure_code": validation.failure_code,
        "raw_claim_count": len(claims),
        "raw_claim_order_canonical": False,
        "canonical_sort_applied_in_memory_only": True,
        "canonical_sort_mutated_archive": False,
        "sort_only_salvage_succeeded": False,
        "sort_only_failure_code": sort_failure,
        "required_binary_field_paths": _REQUIRED_BINARY_PATHS,
        "predicted_field_paths": predicted_paths,
        "missing_required_field_paths": tuple(sorted(required - predicted_path_set)),
        "extra_nonbinary_field_paths": tuple(sorted(predicted_path_set - required)),
        "provider_schema_enforced_exact_binary_field_set": False,
        "provider_schema_enforced_claim_order": False,
        "provider_visible_passage_count": len(provider_passages),
        "fixture_selected_passage_count": len(fixture_passage_ids),
        "fixture_passage_surface_is_strict_subset": True,
        "predicted_passage_ids": predicted_passage_ids,
        "predicted_provider_visible_but_fixture_excluded_passage_ids": (provider_only_predictions),
        "provider_visible_but_fixture_excluded_prediction": True,
        "nonexact_support_quote_field_paths": nonexact_quote_paths,
        "endpoint_quote_exact_in_provider_visible_passage": endpoint_exact,
        "upstream_protocol_prespecified_exact_dose": False,
        "source_exposes_multiple_active_doses": True,
        "predicted_treatment_arm": by_path["treatment_arm.label"]["token"],
        "predicted_treatment_events": by_path["effect.treatment_events"]["token"],
        "predicted_treatment_total": by_path["effect.treatment_total"]["token"],
        "code_owned_target_treatment_arm": target["treatment_arm.label"],
        "code_owned_target_treatment_events": target["effect.treatment_events"],
        "code_owned_target_treatment_total": target["effect.treatment_total"],
        "predicted_arm_differs_from_code_owned_target": True,
        "single_contrast_target_ambiguous_without_protocol_arm": True,
        "prediction_vs_code_owned_target_mismatch_field_paths": mismatches,
        "code_owned_target_is_not_provider_visible_accuracy_label": True,
        "canonicalization_only_recovery_supported": False,
        "value_or_provenance_rewrite_would_be_required": True,
        "immutable_response_reclassified_as_success": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_effectiveness_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return RequestFailureAuditV1.model_validate(
        {**payload, "request_audit_sha256": hash_canonical(payload)}
    )


def freeze_metasyn_contextual_frontier_v1_failure_audit(
    *, repository_root: Path, v1_workspace: Path = DEFAULT_V1_WORKSPACE
) -> MetaSynContextualFrontierV1FailureAudit:
    """Replay the immutable v1 archive and freeze a narrow failure diagnosis."""

    root = Path(repository_root).resolve(strict=True)
    workspace = v1_workspace if v1_workspace.is_absolute() else root / v1_workspace
    workspace = workspace.resolve(strict=True)
    try:
        workspace_relative = workspace.relative_to(root).as_posix()
    except ValueError as exc:
        raise MetaSynContextualFrontierV1FailureAuditError(
            "contextual_frontier_failure_audit_workspace_outside_repository"
        ) from exc

    plan_raw, plan_file_sha, plan_bytes = _checked_json(workspace / "00-prepared.json")
    terminal_raw, terminal_file_sha, terminal_bytes = _checked_json(workspace / "02-terminal.json")
    plan = MetaSynContextualFrontierPlanV1.model_validate(plan_raw)
    terminal = MetaSynContextualFrontierTerminalReportV1.model_validate(terminal_raw)
    if (
        terminal.plan_sha256 != plan.plan_sha256
        or terminal.runtime_pipeline_sha256 != plan.runtime_pipeline_sha256
        or terminal.status != "roster_exhausted_without_typed_graph"
        or tuple(terminal.attempted_request_keys) != _EXPECTED_REQUEST_KEYS
    ):
        raise MetaSynContextualFrontierV1FailureAuditError(
            "contextual_frontier_failure_audit_terminal_plan_mismatch"
        )

    bindings = [
        BoundArtifactV1(
            relative_path="00-prepared.json",
            byte_count=plan_bytes,
            file_sha256=plan_file_sha,
            embedded_sha256=plan.plan_sha256,
        ),
        BoundArtifactV1(
            relative_path="02-terminal.json",
            byte_count=terminal_bytes,
            file_sha256=terminal_file_sha,
            embedded_sha256=terminal.report_sha256,
        ),
    ]
    items = {item.request.request_key: item for item in plan.roster}
    request_audits: list[RequestFailureAuditV1] = []
    for request_key in _EXPECTED_REQUEST_KEYS:
        item = items[request_key]
        receipt_relative = Path("provider-receipts") / f"{request_key}.json"
        validation_relative = Path("validations") / f"{request_key}.json"
        receipt_raw, receipt_file_sha, receipt_bytes = _checked_json(workspace / receipt_relative)
        validation_raw, validation_file_sha, validation_bytes = _checked_json(
            workspace / validation_relative
        )
        receipt = MetaSynContextualFrontierProviderReceiptV1.model_validate(receipt_raw)
        validation = MetaSynContextualFrontierValidationResultV1.model_validate(validation_raw)
        if (
            receipt.request_key != request_key
            or validation.request_key != request_key
            or receipt.provider_result_sha256 != receipt.provider_result.result_sha256
            or validation.provider_receipt_sha256 != receipt.receipt_sha256
        ):
            raise MetaSynContextualFrontierV1FailureAuditError(
                "contextual_frontier_failure_audit_request_archive_mismatch"
            )
        bindings.extend(
            (
                BoundArtifactV1(
                    relative_path=receipt_relative.as_posix(),
                    byte_count=receipt_bytes,
                    file_sha256=receipt_file_sha,
                    embedded_sha256=receipt.receipt_sha256,
                ),
                BoundArtifactV1(
                    relative_path=validation_relative.as_posix(),
                    byte_count=validation_bytes,
                    file_sha256=validation_file_sha,
                    embedded_sha256=validation.validation_sha256,
                ),
            )
        )
        request_audits.append(
            _request_audit(
                item=item,
                receipt=receipt,
                receipt_file_sha256=receipt_file_sha,
                validation=validation,
                validation_file_sha256=validation_file_sha,
                runtime_pipeline_sha256=plan.runtime_pipeline_sha256,
            )
        )

    audits = tuple(request_audits)
    membership = hash_canonical([item.request_audit_sha256 for item in audits])
    payload = {
        "audit_version": AUDIT_VERSION,
        "status": "audited_no_canonicalization_only_salvage",
        "source_workspace_relative_path": workspace_relative,
        "bound_artifacts": tuple(sorted(bindings, key=lambda item: item.relative_path)),
        "runtime_module_sha256": sha256_file(root / _RUNTIME_MODULE_PATH),
        "grounding_module_sha256": sha256_file(root / _GROUNDING_MODULE_PATH),
        "plan_sha256": plan.plan_sha256,
        "runtime_pipeline_sha256": plan.runtime_pipeline_sha256,
        "terminal_report_sha256": terminal.report_sha256,
        "terminal_status": terminal.status,
        "provider_receipt_count": len(audits),
        "structured_completed_response_count": len(audits),
        "typed_graph_completed_response_count": 0,
        "total_input_tokens": sum(item.input_tokens for item in audits),
        "total_output_tokens": sum(item.output_tokens for item in audits),
        "total_estimated_cost_usd_micros": sum(item.estimated_cost_usd_micros for item in audits),
        "total_charged_cost_ceiling_usd_micros": sum(
            item.charged_cost_ceiling_usd_micros for item in audits
        ),
        "request_audits": audits,
        "request_audit_membership_sha256": membership,
        "immutable_v1_preserved": True,
        "provider_or_transport_failure_explains_result": False,
        "structured_json_failure_explains_result": False,
        "claim_order_rejection_contributed_to_both_failures": True,
        "canonical_sort_alone_salvages_any_response": False,
        "fixture_selected_passage_surface_mismatch_contributed": True,
        "protocol_arm_ambiguity_contributed": True,
        "expected_numeric_values_exposed_to_provider": False,
        "exposing_expected_numeric_values_would_invalidate_independent_extraction_claim": True,
        "exact_arm_may_be_provider_input_only_if_prespecified_upstream": True,
        "otherwise_all_source_supported_contrasts_must_be_enumerated": True,
        "structural_required_field_roster_may_be_provider_visible": True,
        "grounding_validity_must_be_separate_from_hidden_target_scoring": True,
        "safe_recovery_requirements": _RECOVERY_REQUIREMENTS,
        "source_scope": "title_abstract_not_release_grade",
        "official_test_labels_opened": False,
        "reference_accuracy_labels_opened": False,
        "new_provider_calls_made": 0,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_effectiveness_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierV1FailureAudit.model_validate(
        {**payload, "audit_sha256": hash_canonical(payload)}
    )


def validate_metasyn_contextual_frontier_v1_failure_audit(
    *,
    audit: MetaSynContextualFrontierV1FailureAudit | dict[str, Any],
    repository_root: Path,
    v1_workspace: Path = DEFAULT_V1_WORKSPACE,
    external_replay: bool = True,
) -> MetaSynContextualFrontierV1FailureAudit:
    try:
        canonical = MetaSynContextualFrontierV1FailureAudit.model_validate(audit)
    except ValueError as exc:
        raise MetaSynContextualFrontierV1FailureAuditError(
            "contextual_frontier_failure_audit_saved_artifact_invalid"
        ) from exc
    if external_replay:
        replayed = freeze_metasyn_contextual_frontier_v1_failure_audit(
            repository_root=repository_root,
            v1_workspace=v1_workspace,
        )
        if replayed != canonical:
            raise MetaSynContextualFrontierV1FailureAuditError(
                "contextual_frontier_failure_audit_external_replay_mismatch"
            )
    return canonical


def write_metasyn_contextual_frontier_v1_failure_audit(
    *, audit: MetaSynContextualFrontierV1FailureAudit, output_path: Path
) -> Path:
    path = Path(os.path.abspath(output_path))
    if path.exists() or path.is_symlink():
        saved_raw, _, _ = _checked_json(path)
        saved = MetaSynContextualFrontierV1FailureAudit.model_validate(saved_raw)
        if saved != audit:
            raise MetaSynContextualFrontierV1FailureAuditError(
                "contextual_frontier_failure_audit_output_replay_mismatch"
            )
        return path.resolve(strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, audit)
    return path.resolve(strict=True)


__all__ = [
    "AUDIT_VERSION",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_V1_WORKSPACE",
    "MetaSynContextualFrontierV1FailureAudit",
    "MetaSynContextualFrontierV1FailureAuditError",
    "RequestFailureAuditV1",
    "freeze_metasyn_contextual_frontier_v1_failure_audit",
    "validate_metasyn_contextual_frontier_v1_failure_audit",
    "write_metasyn_contextual_frontier_v1_failure_audit",
]
