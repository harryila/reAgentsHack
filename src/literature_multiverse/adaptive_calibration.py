"""Complete-question calibration for adaptive first-release verification.

This module fixes the optional-stopping defect in fixed-state selective calibration.
The statistical unit is one independent, complete question trajectory.  A trajectory
is generated once by a frozen, threshold-blind production scheduler and contains all
preselection states from prefix zero through a terminal budget/scheduler condition.
Every predeclared threshold and policy arm is replayed over that same path and yields
exactly one ``(accepted, error)`` Bernoulli pair for the question.

Reference verdicts live in a separate wrapper and are never members of a
policy-visible trajectory, scheduler state, feature row, or prospective candidate.
The simultaneous one-sided Clopper--Pearson bounds control binary reference loss
under exchangeable independent complete-question trajectories.  They do not control
scientific truth, distribution shift, retrieval failure outside the frozen protocol,
or any per-item conditional error probability.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field, JsonValue, TypeAdapter, field_validator, model_validator
from sklearn.linear_model import LogisticRegression

from literature_multiverse.calibration import clopper_pearson_upper
from literature_multiverse.independence_identity import (
    StrongIdentityKind,
    StrongIndependenceIdentityV1,
    canonicalize_authority_identity,
    freeze_strong_independence_identity,
    freeze_strong_independence_identity_from_component_tokens,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.production_policy import PRODUCTION_STOPPING_RULE

AdaptiveSplit = Literal["development", "calibration", "test"]
AdaptiveLabelSource = Literal["benchmark_annotation", "expert_adjudication", "simulation"]
OrdinaryTrajectoryTerminalReason = Literal[
    "budget_exhausted",
    "no_feasible_action",
    "all_items_resolved",
    "nonconfirmation_context_blocked",
]
TrajectoryTerminalReason = Literal[
    "budget_exhausted",
    "no_feasible_action",
    "all_items_resolved",
    "nonconfirmation_context_blocked",
    "full_nonconfirmation_release_gates_passed",
]
AdaptiveClaimDecision = Literal[
    "supported",
    "contradicted",
    "condition_dependent",
    "inconclusive",
    "not_evaluable",
]
ConditionConfirmationGateStatus = Literal[
    "not_applicable",
    "missing",
    "confirmed",
    "not_confirmed",
    "insufficient",
]
StrongIndependenceIdentifierKind = StrongIdentityKind
AdaptiveIndependenceIdentityV2 = StrongIndependenceIdentityV1

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_FORBIDDEN_POLICY_KEYS = {
    "gold_label",
    "ground_truth",
    "observed_error",
    "reference",
    "reference_verdict",
    "unsupported_claim",
}
_FORBIDDEN_TERMINAL_GATE_POLICY_KEYS = {
    "condition_confirmation_assessment",
    "condition_confirmation_outcome",
    "condition_confirmation_status",
    "gate_assessment_sha256",
    "scientific_gate_passed",
    "terminal_gate_result",
    "terminal_gate_status",
}
_FORBIDDEN_COLLECTION_SOURCE_OUTCOME_KEYS = {
    "adaptive_calibration_bundle_v2",
    "calibration_gate_result",
    "condition_confirmation_assessment",
    "condition_confirmation_gate",
    "gold_label",
    "ground_truth",
    "reference",
    "reference_sha256",
    "reference_verdict",
    "release_qualification_proof",
    "scientific_gate_passed",
    "terminal_gate_result",
}
_COST_TOLERANCE = 1e-9


class AdaptiveCalibrationError(ValueError):
    """A trajectory, calibration, or prospective-release contract was violated."""


def _validate_sha256(value: str, name: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"invalid_sha256:{name}")
    return value


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}_must_be_finite_nonnegative")
    return value


def _canonical_json_object(value: Mapping[str, Any], name: str) -> dict[str, JsonValue]:
    try:
        result = _JSON_OBJECT.validate_python(dict(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}_must_be_canonical_json_object") from exc
    return dict(sorted(result.items()))


def _reject_reference_leakage(
    value: Any,
    *,
    path: str = "policy_visible",
    allow_terminal_gate_outcomes: bool = False,
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold()
            if normalized in _FORBIDDEN_POLICY_KEYS or (
                normalized.startswith("reference_") and normalized != "reference_labels_unopened"
            ):
                raise AdaptiveCalibrationError(f"reference_label_leaked_into_policy:{path}.{key}")
            if (
                not allow_terminal_gate_outcomes
                and normalized in _FORBIDDEN_TERMINAL_GATE_POLICY_KEYS
            ):
                raise AdaptiveCalibrationError(
                    f"terminal_condition_outcome_leaked_into_policy:{path}.{key}"
                )
            _reject_reference_leakage(
                item,
                path=f"{path}.{key}",
                allow_terminal_gate_outcomes=allow_terminal_gate_outcomes,
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_reference_leakage(
                item,
                path=f"{path}[{index}]",
                allow_terminal_gate_outcomes=allow_terminal_gate_outcomes,
            )


def _reject_collection_source_outcome_leakage(
    value: Any,
    *,
    path: str = "collection_source_roster",
) -> None:
    """Reject held-out outcomes without confusing target reference orientation.

    ``reference_direction`` is part of the prespecified global target and is thus
    policy-visible. The broader policy-row scanner intentionally rejects every
    ``reference_*`` key, so it is too coarse for a full typed collection source.
    """

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold()
            if normalized in _FORBIDDEN_COLLECTION_SOURCE_OUTCOME_KEYS:
                raise AdaptiveCalibrationError(f"collection_source_outcome_leakage:{path}.{key}")
            _reject_collection_source_outcome_leakage(
                item,
                path=f"{path}.{key}",
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_collection_source_outcome_leakage(
                item,
                path=f"{path}[{index}]",
            )


class AdaptiveTargetSemanticsBindingV2(ContractModel):
    """Question-specific identity for the exact five-way global verdict target."""

    semantics_version: Literal["adaptive-global-target-semantics-v2"] = (
        "adaptive-global-target-semantics-v2"
    )
    question_id: Annotated[str, Field(min_length=1)]
    claim_spec_sha256: str
    global_condition_target_sha256: str
    decision_vocabulary: list[AdaptiveClaimDecision]
    condition_release_semantics: Literal[
        "global condition-dependent verdict requires independent terminal confirmation"
    ] = "global condition-dependent verdict requires independent terminal confirmation"
    target_semantics_sha256: str

    @field_validator(
        "claim_spec_sha256",
        "global_condition_target_sha256",
        "target_semantics_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("decision_vocabulary")
    @classmethod
    def validate_vocabulary(cls, value: list[AdaptiveClaimDecision]) -> list[AdaptiveClaimDecision]:
        expected: list[AdaptiveClaimDecision] = [
            "condition_dependent",
            "contradicted",
            "inconclusive",
            "not_evaluable",
            "supported",
        ]
        if value != expected:
            raise ValueError("adaptive_target_decision_vocabulary_mismatch")
        return value

    @model_validator(mode="after")
    def validate_semantics(self) -> AdaptiveTargetSemanticsBindingV2:
        payload = self.model_dump(mode="json", exclude={"target_semantics_sha256"})
        if hash_canonical(payload) != self.target_semantics_sha256:
            raise ValueError("adaptive_target_semantics_hash_mismatch")
        return self


def freeze_adaptive_target_semantics_v2(
    *,
    question_id: str,
    claim_spec_sha256: str,
    global_condition_target_sha256: str,
) -> AdaptiveTargetSemanticsBindingV2:
    """Bind a reference verdict to more than a bare classification string."""

    payload: dict[str, Any] = {
        "semantics_version": "adaptive-global-target-semantics-v2",
        "question_id": question_id,
        "claim_spec_sha256": claim_spec_sha256,
        "global_condition_target_sha256": global_condition_target_sha256,
        "decision_vocabulary": [
            "condition_dependent",
            "contradicted",
            "inconclusive",
            "not_evaluable",
            "supported",
        ],
        "condition_release_semantics": (
            "global condition-dependent verdict requires independent terminal confirmation"
        ),
    }
    return AdaptiveTargetSemanticsBindingV2.model_validate(
        {**payload, "target_semantics_sha256": hash_canonical(payload)}
    )


def hash_strong_independence_identifier_v2(
    *,
    kind: StrongIndependenceIdentifierKind,
    value: str,
) -> str:
    """Return a globally comparable digest without retaining the identifier value."""

    return canonicalize_authority_identity(kind=kind, value=value).token_sha256


def freeze_adaptive_independence_identity_v2(
    *,
    strong_components: Sequence[Mapping[StrongIndependenceIdentifierKind, Sequence[str]]],
    unverification_reasons: Sequence[str] = (),
) -> AdaptiveIndependenceIdentityV2:
    """Hash strong connected-component identities and discard all raw values."""

    canonical_components = [
        [
            canonicalize_authority_identity(kind=kind, value=value)
            for kind, values in component.items()
            for value in values
        ]
        for component in strong_components
    ]
    return freeze_strong_independence_identity(
        strong_components=canonical_components,
        reasons=unverification_reasons,
    )


def adaptive_independence_identity_from_condition_plan_v1(
    plan: Any,
) -> AdaptiveIndependenceIdentityV2:
    """Recompute content-silent question identity from a validated native plan.

    The supplied component digests are deliberately ignored. Every canonical token is
    reparsed and rehashed by the neutral identity module. A missing authority token or
    canonical alias conflict makes the whole complete-question identity unverified.
    """

    # Delayed import keeps the neutral identity module free of condition machinery.
    from literature_multiverse.condition_confirmation import ConditionConfirmationPlanV1

    if not isinstance(plan, ConditionConfirmationPlanV1):
        raise AdaptiveCalibrationError("adaptive_independence_condition_plan_contract_invalid")
    try:
        normalized = ConditionConfirmationPlanV1.model_validate(plan.model_dump(mode="json"))
    except ValueError as exc:
        raise AdaptiveCalibrationError(
            "adaptive_independence_condition_plan_integrity_changed"
        ) from exc
    reasons: list[str] = []
    component_token_sets: list[list[str]] = []
    target_estimate_ids = {
        estimate.estimate_id for estimate in normalized.roster.estimates if estimate.target_scope
    }
    target_assignments = [
        assignment
        for assignment in normalized.component_assignments
        if target_estimate_ids & set(assignment.estimate_ids)
    ]
    if not target_assignments:
        reasons.append("target_authority_identity_components_absent")
    for assignment in target_assignments:
        if assignment.authority_identity_conflict_sha256s:
            reasons.append("authority_identity_alias_conflict")
        if not assignment.split_identity_tokens:
            reasons.append("authority_identity_component_missing")
            continue
        component_token_sets.append(assignment.split_identity_tokens)
    if len(component_token_sets) != len(target_assignments):
        reasons.append("complete_question_authority_identity_incomplete")
    try:
        return freeze_strong_independence_identity_from_component_tokens(
            component_token_sets=component_token_sets,
            reasons=reasons,
        )
    except ValueError as exc:
        raise AdaptiveCalibrationError(
            "adaptive_independence_condition_plan_projection_failed"
        ) from exc


class ConditionOutcomeFirewallReceiptV1(ContractModel):
    """Outcome-free proof of the graph boundary seen by the online policy."""

    receipt_version: Literal["condition-outcome-firewall-receipt-v1"] = (
        "condition-outcome-firewall-receipt-v1"
    )
    plan_sha256: str
    materialization_receipt_sha256: str
    development_partition_sha256: str
    confirmation_partition_sha256: str
    online_graph_sha256: str
    pipeline_sha256: str
    synthesis_runner_sha256: str
    candidate_runner_sha256: str
    outcome_firewall_status: Literal["confirmation_partition_unopened_by_online_policy"] = (
        "confirmation_partition_unopened_by_online_policy"
    )
    firewall_receipt_sha256: str

    @field_validator(
        "plan_sha256",
        "materialization_receipt_sha256",
        "development_partition_sha256",
        "confirmation_partition_sha256",
        "online_graph_sha256",
        "pipeline_sha256",
        "synthesis_runner_sha256",
        "candidate_runner_sha256",
        "firewall_receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_receipt(self) -> ConditionOutcomeFirewallReceiptV1:
        payload = self.model_dump(mode="json", exclude={"firewall_receipt_sha256"})
        if hash_canonical(payload) != self.firewall_receipt_sha256:
            raise ValueError("condition_outcome_firewall_receipt_hash_mismatch")
        return self


def freeze_condition_outcome_firewall_receipt(
    *,
    plan_sha256: str,
    materialization_receipt_sha256: str,
    development_partition_sha256: str,
    confirmation_partition_sha256: str,
    online_graph_sha256: str,
    pipeline_sha256: str,
    synthesis_runner_sha256: str,
    candidate_runner_sha256: str,
) -> ConditionOutcomeFirewallReceiptV1:
    """Recompute the firewall receipt; callers cannot author its digest."""

    payload: dict[str, Any] = {
        "receipt_version": "condition-outcome-firewall-receipt-v1",
        "plan_sha256": plan_sha256,
        "materialization_receipt_sha256": materialization_receipt_sha256,
        "development_partition_sha256": development_partition_sha256,
        "confirmation_partition_sha256": confirmation_partition_sha256,
        "online_graph_sha256": online_graph_sha256,
        "pipeline_sha256": pipeline_sha256,
        "synthesis_runner_sha256": synthesis_runner_sha256,
        "candidate_runner_sha256": candidate_runner_sha256,
        "outcome_firewall_status": ("confirmation_partition_unopened_by_online_policy"),
    }
    return ConditionOutcomeFirewallReceiptV1.model_validate(
        {**payload, "firewall_receipt_sha256": hash_canonical(payload)}
    )


class ConditionCalibrationProjectionV1(ContractModel):
    """Outcome-free condition identity used by online state and later calibration.

    This projection deliberately has no model, assessment, gate status, prediction,
    effect, or confirmation-outcome field. A missing and a confirmed terminal gate
    therefore produce the same projection for the same frozen scientific state.
    """

    projection_version: Literal["condition-calibration-projection-v1"] = (
        "condition-calibration-projection-v1"
    )
    question_id: Annotated[str, Field(min_length=1)]
    target_semantics: AdaptiveTargetSemanticsBindingV2
    target_semantics_sha256: str
    independence_identity_sha256: str
    claim_spec_sha256: str
    question_config_sha256: str
    corpus_snapshot_sha256: str
    corpus_cutoff: Annotated[str, Field(min_length=1)]
    condition_target_sha256: str
    plan_sha256: str
    materialization_receipt_sha256: str
    full_graph_sha256: str
    development_graph_sha256: str
    confirmation_graph_sha256: str
    development_partition_sha256: str
    confirmation_partition_sha256: str
    confirmation_config_sha256: str
    pipeline_sha256: str
    online_graph_sha256: str
    synthesis_runner_sha256: str
    candidate_runner_sha256: str
    prespecified_moderator_names: Annotated[list[str], Field(min_length=1)]
    provisional_claim_decision: Literal["condition_dependent"] = "condition_dependent"
    outcome_firewall_status: Literal["confirmation_partition_unopened_by_online_policy"] = (
        "confirmation_partition_unopened_by_online_policy"
    )
    firewall_receipt: ConditionOutcomeFirewallReceiptV1
    firewall_receipt_sha256: str
    projection_sha256: str

    @field_validator(
        "target_semantics_sha256",
        "independence_identity_sha256",
        "claim_spec_sha256",
        "question_config_sha256",
        "corpus_snapshot_sha256",
        "condition_target_sha256",
        "plan_sha256",
        "materialization_receipt_sha256",
        "full_graph_sha256",
        "development_graph_sha256",
        "confirmation_graph_sha256",
        "development_partition_sha256",
        "confirmation_partition_sha256",
        "confirmation_config_sha256",
        "pipeline_sha256",
        "online_graph_sha256",
        "synthesis_runner_sha256",
        "candidate_runner_sha256",
        "firewall_receipt_sha256",
        "projection_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("prespecified_moderator_names")
    @classmethod
    def validate_moderators(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item.strip() for item in value):
            raise ValueError("condition_projection_moderators_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_projection(self) -> ConditionCalibrationProjectionV1:
        if (
            self.target_semantics_sha256 != self.target_semantics.target_semantics_sha256
            or self.question_id != self.target_semantics.question_id
            or self.claim_spec_sha256 != self.target_semantics.claim_spec_sha256
            or self.condition_target_sha256 != self.target_semantics.global_condition_target_sha256
        ):
            raise ValueError("condition_projection_target_semantics_mismatch")
        if self.online_graph_sha256 != self.development_graph_sha256:
            raise ValueError("condition_projection_online_graph_not_development_only")
        receipt = self.firewall_receipt
        expected_receipt_fields = {
            "plan_sha256": self.plan_sha256,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "development_partition_sha256": self.development_partition_sha256,
            "confirmation_partition_sha256": self.confirmation_partition_sha256,
            "online_graph_sha256": self.online_graph_sha256,
            "pipeline_sha256": self.pipeline_sha256,
            "synthesis_runner_sha256": self.synthesis_runner_sha256,
            "candidate_runner_sha256": self.candidate_runner_sha256,
            "outcome_firewall_status": self.outcome_firewall_status,
        }
        observed_receipt_fields = receipt.model_dump(
            mode="python",
            exclude={"receipt_version", "firewall_receipt_sha256"},
        )
        if (
            self.firewall_receipt_sha256 != receipt.firewall_receipt_sha256
            or observed_receipt_fields != expected_receipt_fields
        ):
            raise ValueError("condition_projection_firewall_receipt_mismatch")
        payload = self.model_dump(mode="json", exclude={"projection_sha256"})
        _reject_reference_leakage(payload)
        if hash_canonical(payload) != self.projection_sha256:
            raise ValueError("condition_calibration_projection_hash_mismatch")
        return self


def freeze_condition_calibration_projection(
    *,
    question_id: str,
    target_semantics: AdaptiveTargetSemanticsBindingV2,
    independence_identity: AdaptiveIndependenceIdentityV2,
    question_config_sha256: str,
    corpus_snapshot_sha256: str,
    corpus_cutoff: str,
    plan_sha256: str,
    materialization_receipt_sha256: str,
    full_graph_sha256: str,
    development_graph_sha256: str,
    confirmation_graph_sha256: str,
    development_partition_sha256: str,
    confirmation_partition_sha256: str,
    confirmation_config_sha256: str,
    pipeline_sha256: str,
    synthesis_runner_sha256: str,
    candidate_runner_sha256: str,
    prespecified_moderator_names: Sequence[str],
) -> ConditionCalibrationProjectionV1:
    """Freeze the development-only projection and compute its firewall receipt."""

    try:
        target_semantics = AdaptiveTargetSemanticsBindingV2.model_validate(
            target_semantics.model_dump(mode="json")
        )
        independence_identity = AdaptiveIndependenceIdentityV2.model_validate(
            independence_identity.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError("condition_projection_target_semantics_invalid") from exc
    firewall = freeze_condition_outcome_firewall_receipt(
        plan_sha256=plan_sha256,
        materialization_receipt_sha256=materialization_receipt_sha256,
        development_partition_sha256=development_partition_sha256,
        confirmation_partition_sha256=confirmation_partition_sha256,
        online_graph_sha256=development_graph_sha256,
        pipeline_sha256=pipeline_sha256,
        synthesis_runner_sha256=synthesis_runner_sha256,
        candidate_runner_sha256=candidate_runner_sha256,
    )
    payload: dict[str, Any] = {
        "projection_version": "condition-calibration-projection-v1",
        "question_id": question_id,
        "target_semantics": target_semantics,
        "target_semantics_sha256": target_semantics.target_semantics_sha256,
        "independence_identity_sha256": (independence_identity.independence_identity_sha256),
        "claim_spec_sha256": target_semantics.claim_spec_sha256,
        "question_config_sha256": question_config_sha256,
        "corpus_snapshot_sha256": corpus_snapshot_sha256,
        "corpus_cutoff": corpus_cutoff,
        "condition_target_sha256": (target_semantics.global_condition_target_sha256),
        "plan_sha256": plan_sha256,
        "materialization_receipt_sha256": materialization_receipt_sha256,
        "full_graph_sha256": full_graph_sha256,
        "development_graph_sha256": development_graph_sha256,
        "confirmation_graph_sha256": confirmation_graph_sha256,
        "development_partition_sha256": development_partition_sha256,
        "confirmation_partition_sha256": confirmation_partition_sha256,
        "confirmation_config_sha256": confirmation_config_sha256,
        "pipeline_sha256": pipeline_sha256,
        "online_graph_sha256": development_graph_sha256,
        "synthesis_runner_sha256": synthesis_runner_sha256,
        "candidate_runner_sha256": candidate_runner_sha256,
        "prespecified_moderator_names": sorted(set(prespecified_moderator_names)),
        "provisional_claim_decision": "condition_dependent",
        "outcome_firewall_status": ("confirmation_partition_unopened_by_online_policy"),
        "firewall_receipt": firewall,
        "firewall_receipt_sha256": firewall.firewall_receipt_sha256,
    }
    return ConditionCalibrationProjectionV1.model_validate(
        {**payload, "projection_sha256": hash_canonical(payload)}
    )


class ConditionConfirmationGateAssessmentV1(ContractModel):
    """Runtime scientific gate embedded before a certificate hash exists."""

    gate_version: Literal["condition-confirmation-release-gate-v1"] = (
        "condition-confirmation-release-gate-v1"
    )
    required: bool
    provisional_claim_decision: AdaptiveClaimDecision
    status: ConditionConfirmationGateStatus
    condition_projection_sha256: str | None = None
    target_sha256: str | None = None
    plan_sha256: str | None = None
    config_sha256: str | None = None
    model_sha256: str | None = None
    assessment_sha256: str | None = None
    scientific_gate_passed: bool
    reasons: list[str]
    interpretation: Literal[
        "held-out predictive association gate; not causal proof or scientific truth"
    ] = "held-out predictive association gate; not causal proof or scientific truth"
    gate_assessment_sha256: str

    @field_validator(
        "condition_projection_sha256",
        "target_sha256",
        "plan_sha256",
        "config_sha256",
        "model_sha256",
        "assessment_sha256",
        "gate_assessment_sha256",
    )
    @classmethod
    def validate_optional_hash(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _validate_sha256(value, info.field_name)

    @field_validator("required", "scientific_gate_passed", mode="before")
    @classmethod
    def validate_boolean(cls, value: object, info: Any) -> object:
        if not isinstance(value, bool):
            raise ValueError(f"condition_gate_{info.field_name}_must_be_boolean")
        return value

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not reason.strip() for reason in value):
            raise ValueError("condition_gate_reasons_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_gate(self) -> ConditionConfirmationGateAssessmentV1:
        identity_fields = (
            self.condition_projection_sha256,
            self.target_sha256,
            self.plan_sha256,
            self.config_sha256,
        )
        result_fields = (self.model_sha256, self.assessment_sha256)
        if not self.required:
            if (
                self.provisional_claim_decision == "condition_dependent"
                or self.status != "not_applicable"
                or not self.scientific_gate_passed
                or self.reasons
                or any(value is not None for value in (*identity_fields, *result_fields))
            ):
                raise ValueError("condition_gate_not_applicable_fields_mismatch")
        else:
            if (
                self.provisional_claim_decision != "condition_dependent"
                or self.status == "not_applicable"
                or any(value is None for value in identity_fields)
            ):
                raise ValueError("condition_gate_required_identity_incomplete")
            materialized = self.status in {
                "confirmed",
                "not_confirmed",
                "insufficient",
            }
            if materialized != all(value is not None for value in result_fields):
                raise ValueError("condition_gate_materialized_lineage_mismatch")
            expected_pass = self.status == "confirmed"
            if self.scientific_gate_passed != expected_pass:
                raise ValueError("condition_gate_pass_status_mismatch")
            if expected_pass == bool(self.reasons):
                raise ValueError("condition_gate_reason_status_mismatch")
        payload = self.model_dump(mode="json", exclude={"gate_assessment_sha256"})
        if hash_canonical(payload) != self.gate_assessment_sha256:
            raise ValueError("condition_gate_assessment_hash_mismatch")
        return self


def freeze_condition_confirmation_gate_assessment(
    *,
    provisional_claim_decision: AdaptiveClaimDecision,
    status: ConditionConfirmationGateStatus,
    reasons: Sequence[str] = (),
    condition_projection_sha256: str | None = None,
    target_sha256: str | None = None,
    plan_sha256: str | None = None,
    config_sha256: str | None = None,
    model_sha256: str | None = None,
    assessment_sha256: str | None = None,
) -> ConditionConfirmationGateAssessmentV1:
    """Freeze the non-self-referential scientific gate carried by release output."""

    required = provisional_claim_decision == "condition_dependent"
    payload: dict[str, Any] = {
        "gate_version": "condition-confirmation-release-gate-v1",
        "required": required,
        "provisional_claim_decision": provisional_claim_decision,
        "status": status,
        "condition_projection_sha256": condition_projection_sha256,
        "target_sha256": target_sha256,
        "plan_sha256": plan_sha256,
        "config_sha256": config_sha256,
        "model_sha256": model_sha256,
        "assessment_sha256": assessment_sha256,
        "scientific_gate_passed": (
            status == "confirmed" if required else status == "not_applicable"
        ),
        "reasons": sorted(set(reasons)),
        "interpretation": (
            "held-out predictive association gate; not causal proof or scientific truth"
        ),
    }
    return ConditionConfirmationGateAssessmentV1.model_validate(
        {**payload, "gate_assessment_sha256": hash_canonical(payload)}
    )


class AdaptivePolicyContext(ContractModel):
    """Exact deployed context for one prespecified scheduler/policy arm."""

    context_version: Literal["adaptive-policy-context-v1"] = "adaptive-policy-context-v1"
    policy_arm_id: Annotated[str, Field(min_length=1)]
    population_id: Annotated[str, Field(min_length=1)]
    pipeline_sha256: str
    allocation_policy: dict[str, JsonValue]
    budget_minutes: float
    cost_unit: Literal["person_minutes"] = "person_minutes"
    production_stopping_rule: Literal["stop_at_first_full_frozen_release_eligible_state"] = (
        PRODUCTION_STOPPING_RULE
    )
    release_config: dict[str, JsonValue]
    audit_config: dict[str, JsonValue]
    target_semantics: dict[str, JsonValue]
    corpus_protocol_context: dict[str, JsonValue]
    score_feature_names: list[str]
    policy_context_sha256: str

    @field_validator("pipeline_sha256", "policy_context_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("budget_minutes")
    @classmethod
    def validate_budget(cls, value: float) -> float:
        return _finite_nonnegative(value, "adaptive_policy_budget")

    @field_validator(
        "allocation_policy",
        "release_config",
        "audit_config",
        "target_semantics",
        "corpus_protocol_context",
    )
    @classmethod
    def validate_json_context(cls, value: dict[str, JsonValue], info: Any) -> dict[str, JsonValue]:
        result = _canonical_json_object(value, info.field_name)
        _reject_reference_leakage(result, path=info.field_name)
        return result

    @field_validator("score_feature_names")
    @classmethod
    def validate_feature_names(cls, value: list[str]) -> list[str]:
        if not value or value != sorted(set(value)) or any(not name for name in value):
            raise ValueError("adaptive_score_features_must_be_nonempty_sorted_unique")
        _reject_reference_leakage({name: 0 for name in value}, path="score_feature_names")
        return value

    @model_validator(mode="after")
    def validate_context(self) -> AdaptivePolicyContext:
        if self.production_stopping_rule != PRODUCTION_STOPPING_RULE:
            raise ValueError("adaptive_production_stopping_rule_mismatch")
        payload = self.model_dump(mode="json", exclude={"policy_context_sha256"})
        if hash_canonical(payload) != self.policy_context_sha256:
            raise ValueError("adaptive_policy_context_hash_mismatch")
        return self


def freeze_adaptive_policy_context(
    *,
    policy_arm_id: str,
    population_id: str,
    pipeline_sha256: str,
    allocation_policy: Mapping[str, Any],
    budget_minutes: float,
    release_config: Mapping[str, Any],
    audit_config: Mapping[str, Any],
    target_semantics: Mapping[str, Any],
    corpus_protocol_context: Mapping[str, Any],
    score_feature_names: Sequence[str],
) -> AdaptivePolicyContext:
    """Seal every deployed choice that may change an adaptive release trajectory."""

    payload: dict[str, Any] = {
        "context_version": "adaptive-policy-context-v1",
        "policy_arm_id": policy_arm_id,
        "population_id": population_id,
        "pipeline_sha256": pipeline_sha256,
        "allocation_policy": dict(allocation_policy),
        "budget_minutes": float(budget_minutes),
        "cost_unit": "person_minutes",
        "production_stopping_rule": PRODUCTION_STOPPING_RULE,
        "release_config": dict(release_config),
        "audit_config": dict(audit_config),
        "target_semantics": dict(target_semantics),
        "corpus_protocol_context": dict(corpus_protocol_context),
        "score_feature_names": sorted(score_feature_names),
    }
    return AdaptivePolicyContext.model_validate(
        {**payload, "policy_context_sha256": hash_canonical(payload)}
    )


class CompleteCorpusIdentity(ContractModel):
    """Complete publication membership for one question, including empty corpora."""

    identity_version: Literal["complete-corpus-membership-v1"] = "complete-corpus-membership-v1"
    corpus_id: Annotated[str, Field(min_length=1)]
    corpus_source_sha256: str
    corpus_cutoff: Annotated[str, Field(min_length=1)]
    membership_basis: Literal["source_manifest", "frozen_corpus_publications"]
    publication_ids: list[str]
    source_manifest_sha256: str | None = None
    membership_sha256: str

    @field_validator("corpus_source_sha256", "membership_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("source_manifest_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return None if value is None else _validate_sha256(value, "source_manifest_sha256")

    @field_validator("publication_ids")
    @classmethod
    def validate_publications(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item for item in value):
            raise ValueError("complete_corpus_publications_must_be_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_membership(self) -> CompleteCorpusIdentity:
        if (self.membership_basis == "source_manifest") != (
            self.source_manifest_sha256 is not None
        ):
            raise ValueError("complete_corpus_source_manifest_basis_mismatch")
        payload = self.model_dump(mode="json", exclude={"membership_sha256"})
        if hash_canonical(payload) != self.membership_sha256:
            raise ValueError("complete_corpus_membership_hash_mismatch")
        return self


def freeze_complete_corpus_identity(
    *,
    corpus_id: str,
    corpus_source_sha256: str,
    corpus_cutoff: str,
    publication_ids: Sequence[str],
    source_manifest_sha256: str | None = None,
) -> CompleteCorpusIdentity:
    basis = (
        "source_manifest" if source_manifest_sha256 is not None else "frozen_corpus_publications"
    )
    payload: dict[str, Any] = {
        "identity_version": "complete-corpus-membership-v1",
        "corpus_id": corpus_id,
        "corpus_source_sha256": corpus_source_sha256,
        "corpus_cutoff": corpus_cutoff,
        "membership_basis": basis,
        "publication_ids": sorted(set(publication_ids)),
        "source_manifest_sha256": source_manifest_sha256,
    }
    return CompleteCorpusIdentity.model_validate(
        {**payload, "membership_sha256": hash_canonical(payload)}
    )


class AdaptivePreselectionState(ContractModel):
    """One label-free no-active-action state on a threshold-blind scheduler path."""

    state_version: Literal["adaptive-preselection-state-v1"] = "adaptive-preselection-state-v1"
    prefix_index: Annotated[int, Field(ge=0)]
    audit_prefix_item_ids: list[str]
    audit_prefix_cost_minutes: float
    scheduler_state_sha256: str
    evidence_graph_sha256: str
    synthesis_sha256: str
    non_calibration_assessment_sha256: str
    non_calibration_gates_passed: bool
    non_calibration_blocking_reasons: list[str]
    claim_decision: Annotated[str, Field(min_length=1)]
    score_features: dict[str, float]
    scalar_risk_score: Annotated[float, Field(ge=0, le=1)] | None = None
    score_model_sha256: str | None = None
    state_sha256: str

    @field_validator(
        "scheduler_state_sha256",
        "evidence_graph_sha256",
        "synthesis_sha256",
        "non_calibration_assessment_sha256",
        "state_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("score_model_sha256")
    @classmethod
    def validate_optional_model_hash(cls, value: str | None) -> str | None:
        return None if value is None else _validate_sha256(value, "score_model_sha256")

    @field_validator("audit_prefix_cost_minutes")
    @classmethod
    def validate_cost(cls, value: float) -> float:
        return _finite_nonnegative(value, "adaptive_prefix_cost")

    @field_validator("audit_prefix_item_ids")
    @classmethod
    def validate_prefix(cls, value: list[str]) -> list[str]:
        if any(not item for item in value) or len(value) != len(set(value)):
            raise ValueError("adaptive_audit_prefix_items_must_be_ordered_unique")
        return value

    @field_validator("non_calibration_blocking_reasons")
    @classmethod
    def validate_reasons(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not reason for reason in value):
            raise ValueError("adaptive_noncalibration_reasons_must_be_sorted_unique")
        return value

    @field_validator("score_features")
    @classmethod
    def validate_features(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("adaptive_score_features_empty")
        if any(not name or not math.isfinite(number) for name, number in value.items()):
            raise ValueError("adaptive_score_features_must_be_named_finite")
        result = dict(sorted((name, float(number)) for name, number in value.items()))
        _reject_reference_leakage(result, path="score_features")
        return result

    @field_validator("non_calibration_gates_passed", mode="before")
    @classmethod
    def validate_strict_gate(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("adaptive_noncalibration_gate_must_be_boolean")
        return value

    @model_validator(mode="after")
    def validate_state(self) -> AdaptivePreselectionState:
        if self.prefix_index != len(self.audit_prefix_item_ids):
            raise ValueError("adaptive_prefix_index_identity_mismatch")
        if self.non_calibration_gates_passed == bool(self.non_calibration_blocking_reasons):
            raise ValueError("adaptive_noncalibration_gate_reason_mismatch")
        if (self.scalar_risk_score is None) != (self.score_model_sha256 is None):
            raise ValueError("adaptive_state_score_lineage_incomplete")
        payload = self.model_dump(mode="json", exclude={"state_sha256"})
        _reject_reference_leakage(payload)
        if hash_canonical(payload) != self.state_sha256:
            raise ValueError("adaptive_preselection_state_hash_mismatch")
        return self


def freeze_adaptive_preselection_state(
    *,
    prefix_index: int,
    audit_prefix_item_ids: Sequence[str],
    audit_prefix_cost_minutes: float,
    scheduler_state_sha256: str,
    evidence_graph_sha256: str,
    synthesis_sha256: str,
    non_calibration_assessment_sha256: str,
    non_calibration_gates_passed: bool,
    non_calibration_blocking_reasons: Sequence[str],
    claim_decision: str,
    score_features: Mapping[str, float],
    scalar_risk_score: float | None = None,
    score_model_sha256: str | None = None,
) -> AdaptivePreselectionState:
    payload: dict[str, Any] = {
        "state_version": "adaptive-preselection-state-v1",
        "prefix_index": prefix_index,
        "audit_prefix_item_ids": list(audit_prefix_item_ids),
        "audit_prefix_cost_minutes": float(audit_prefix_cost_minutes),
        "scheduler_state_sha256": scheduler_state_sha256,
        "evidence_graph_sha256": evidence_graph_sha256,
        "synthesis_sha256": synthesis_sha256,
        "non_calibration_assessment_sha256": non_calibration_assessment_sha256,
        "non_calibration_gates_passed": non_calibration_gates_passed,
        "non_calibration_blocking_reasons": sorted(set(non_calibration_blocking_reasons)),
        "claim_decision": claim_decision,
        "score_features": dict(score_features),
        "scalar_risk_score": scalar_risk_score,
        "score_model_sha256": score_model_sha256,
    }
    return AdaptivePreselectionState.model_validate(
        {**payload, "state_sha256": hash_canonical(payload)}
    )


class AdaptiveTerminalAuditCandidate(ContractModel):
    """Minimal complete terminal action snapshot needed to recompute feasibility."""

    item_id: Annotated[str, Field(min_length=1)]
    eligible: bool
    estimated_cost_minutes: float
    source_candidate_sha256: str

    @field_validator("eligible", mode="before")
    @classmethod
    def validate_strict_eligible(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("adaptive_terminal_candidate_eligible_must_be_boolean")
        return value

    @field_validator("estimated_cost_minutes")
    @classmethod
    def validate_cost(cls, value: float) -> float:
        return _finite_nonnegative(value, "adaptive_terminal_candidate_cost")

    @field_validator("source_candidate_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "source_candidate_sha256")


class AdaptiveTerminalSchedulerProof(ContractModel):
    """Terminal snapshot whose unresolved and feasible actions are recomputable."""

    proof_version: Literal["adaptive-terminal-scheduler-proof-v1"] = (
        "adaptive-terminal-scheduler-proof-v1"
    )
    terminal_reason: OrdinaryTrajectoryTerminalReason
    terminal_scheduler_state_sha256: str
    source_candidate_input_sha256: str
    candidates: list[AdaptiveTerminalAuditCandidate]
    resolved_item_ids: list[str]
    remaining_budget_minutes: float
    nonconfirmation_blocking_reasons: list[str] = Field(default_factory=list)
    active_action: Literal[False] = False
    proof_sha256: str

    @field_validator(
        "terminal_scheduler_state_sha256",
        "source_candidate_input_sha256",
        "proof_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("remaining_budget_minutes")
    @classmethod
    def validate_budget(cls, value: float) -> float:
        return _finite_nonnegative(value, "adaptive_terminal_remaining_budget")

    @field_validator("resolved_item_ids")
    @classmethod
    def validate_resolved_ids(cls, value: list[str]) -> list[str]:
        if any(not item for item in value) or len(value) != len(set(value)):
            raise ValueError("adaptive_terminal_resolved_items_invalid")
        return value

    @field_validator("nonconfirmation_blocking_reasons")
    @classmethod
    def validate_blocking_reasons(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not reason for reason in value):
            raise ValueError("adaptive_terminal_nonconfirmation_reasons_invalid")
        return value

    @model_validator(mode="after")
    def validate_terminal(self) -> AdaptiveTerminalSchedulerProof:
        candidate_ids = [candidate.item_id for candidate in self.candidates]
        if candidate_ids != sorted(set(candidate_ids)):
            raise ValueError("adaptive_terminal_candidates_not_sorted_unique")
        candidate_id_set = set(candidate_ids)
        resolved = set(self.resolved_item_ids)
        if not resolved <= candidate_id_set:
            raise ValueError("adaptive_terminal_resolved_candidate_missing")
        unresolved = candidate_id_set - resolved
        feasible = {
            candidate.item_id
            for candidate in self.candidates
            if candidate.item_id not in resolved
            and candidate.eligible
            and candidate.estimated_cost_minutes <= self.remaining_budget_minutes + _COST_TOLERANCE
        }
        if self.terminal_reason == "nonconfirmation_context_blocked":
            if not self.nonconfirmation_blocking_reasons:
                raise ValueError("adaptive_terminal_context_blockers_missing")
        elif self.nonconfirmation_blocking_reasons:
            raise ValueError("adaptive_terminal_context_blockers_reason_mismatch")
        elif self.terminal_reason == "all_items_resolved":
            if unresolved:
                raise ValueError("adaptive_terminal_all_resolved_has_unresolved_items")
        elif self.terminal_reason == "budget_exhausted":
            if (
                not unresolved
                or feasible
                or not math.isclose(
                    self.remaining_budget_minutes,
                    0.0,
                    rel_tol=0.0,
                    abs_tol=_COST_TOLERANCE,
                )
            ):
                raise ValueError("adaptive_terminal_budget_exhaustion_not_proven")
        elif not unresolved or feasible or self.remaining_budget_minutes <= _COST_TOLERANCE:
            raise ValueError("adaptive_terminal_no_feasible_action_not_proven")
        payload = self.model_dump(mode="json", exclude={"proof_sha256"})
        if hash_canonical(payload) != self.proof_sha256:
            raise ValueError("adaptive_terminal_scheduler_proof_hash_mismatch")
        return self


def freeze_adaptive_terminal_scheduler_proof(
    *,
    terminal_reason: OrdinaryTrajectoryTerminalReason,
    terminal_scheduler_state_sha256: str,
    source_candidate_input_sha256: str,
    candidates: Sequence[AdaptiveTerminalAuditCandidate],
    resolved_item_ids: Sequence[str],
    remaining_budget_minutes: float,
    nonconfirmation_blocking_reasons: Sequence[str] = (),
) -> AdaptiveTerminalSchedulerProof:
    try:
        normalized_candidates = [
            AdaptiveTerminalAuditCandidate.model_validate(candidate.model_dump(mode="json"))
            for candidate in candidates
        ]
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError("adaptive_terminal_candidate_integrity_changed") from exc
    payload: dict[str, Any] = {
        "proof_version": "adaptive-terminal-scheduler-proof-v1",
        "terminal_reason": terminal_reason,
        "terminal_scheduler_state_sha256": terminal_scheduler_state_sha256,
        "source_candidate_input_sha256": source_candidate_input_sha256,
        "candidates": sorted(normalized_candidates, key=lambda row: row.item_id),
        "resolved_item_ids": list(resolved_item_ids),
        "remaining_budget_minutes": float(remaining_budget_minutes),
        "nonconfirmation_blocking_reasons": sorted(set(nonconfirmation_blocking_reasons)),
        "active_action": False,
    }
    return AdaptiveTerminalSchedulerProof.model_validate(
        {**payload, "proof_sha256": hash_canonical(payload)}
    )


class ConditionGateInvocationProofV2(ContractModel):
    """Outcome-free proof that terminal confirmation may be invoked exactly once."""

    proof_version: Literal["condition-gate-invocation-proof-v2"] = (
        "condition-gate-invocation-proof-v2"
    )
    invocation_basis: Literal[
        "ordinary_scheduler_terminal",
        "first_nonconfirmation_eligible_state",
    ]
    terminal_reason: TrajectoryTerminalReason
    terminal_preselection_state: AdaptivePreselectionState
    terminal_preselection_state_sha256: str
    condition_projection: ConditionCalibrationProjectionV1
    condition_projection_sha256: str
    source_candidate_input_sha256: str
    available_actions: list[AdaptiveTerminalAuditCandidate]
    available_action_roster_sha256: str
    resolved_item_ids: list[str]
    remaining_budget_minutes: float
    unresolved_feasible_action_ids: list[str]
    ordinary_scheduler_proof: AdaptiveTerminalSchedulerProof | None = None
    active_action: Literal[False] = False
    confirmation_outcomes_unopened: Literal[True] = True
    reference_labels_unopened: Literal[True] = True
    proof_sha256: str

    @field_validator(
        "terminal_preselection_state_sha256",
        "condition_projection_sha256",
        "source_candidate_input_sha256",
        "available_action_roster_sha256",
        "proof_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("remaining_budget_minutes")
    @classmethod
    def validate_budget(cls, value: float) -> float:
        return _finite_nonnegative(value, "condition_invocation_remaining_budget")

    @field_validator("resolved_item_ids", "unresolved_feasible_action_ids")
    @classmethod
    def validate_item_ids(cls, value: list[str], info: Any) -> list[str]:
        if len(value) != len(set(value)) or any(not item for item in value):
            raise ValueError(f"condition_invocation_item_ids_invalid:{info.field_name}")
        if info.field_name == "unresolved_feasible_action_ids" and value != sorted(value):
            raise ValueError("condition_invocation_feasible_items_not_sorted")
        return value

    @model_validator(mode="after")
    def validate_invocation(self) -> ConditionGateInvocationProofV2:
        state = self.terminal_preselection_state
        projection = self.condition_projection
        if (
            self.terminal_preselection_state_sha256 != state.state_sha256
            or self.condition_projection_sha256 != projection.projection_sha256
            or state.claim_decision != "condition_dependent"
            or not state.non_calibration_gates_passed
            or state.non_calibration_blocking_reasons
            or state.evidence_graph_sha256 != projection.online_graph_sha256
        ):
            raise ValueError("condition_invocation_state_projection_mismatch")
        action_ids = [action.item_id for action in self.available_actions]
        if action_ids != sorted(set(action_ids)):
            raise ValueError("condition_invocation_actions_not_sorted_unique")
        if hash_canonical(self.available_actions) != self.available_action_roster_sha256:
            raise ValueError("condition_invocation_action_roster_hash_mismatch")
        resolved = set(self.resolved_item_ids)
        if resolved != set(state.audit_prefix_item_ids) or not resolved <= set(action_ids):
            raise ValueError("condition_invocation_resolved_prefix_mismatch")
        feasible = sorted(
            action.item_id
            for action in self.available_actions
            if action.item_id not in resolved
            and action.eligible
            and action.estimated_cost_minutes <= self.remaining_budget_minutes + _COST_TOLERANCE
        )
        if feasible != self.unresolved_feasible_action_ids:
            raise ValueError("condition_invocation_feasible_action_mismatch")
        if self.invocation_basis == "ordinary_scheduler_terminal":
            proof = self.ordinary_scheduler_proof
            if (
                proof is None
                or self.terminal_reason != proof.terminal_reason
                or proof.terminal_scheduler_state_sha256 != state.scheduler_state_sha256
                or proof.source_candidate_input_sha256 != self.source_candidate_input_sha256
                or proof.candidates != self.available_actions
                or proof.resolved_item_ids != self.resolved_item_ids
                or not math.isclose(
                    proof.remaining_budget_minutes,
                    self.remaining_budget_minutes,
                    rel_tol=0.0,
                    abs_tol=_COST_TOLERANCE,
                )
                or feasible
            ):
                raise ValueError("condition_invocation_ordinary_terminal_mismatch")
        elif (
            self.terminal_reason != "full_nonconfirmation_release_gates_passed"
            or self.ordinary_scheduler_proof is not None
            or not feasible
        ):
            raise ValueError("condition_invocation_early_stop_mismatch")
        payload = self.model_dump(mode="json", exclude={"proof_sha256"})
        _reject_reference_leakage(payload)
        if hash_canonical(payload) != self.proof_sha256:
            raise ValueError("condition_invocation_proof_hash_mismatch")
        return self

    @property
    def candidates(self) -> list[AdaptiveTerminalAuditCandidate]:
        return self.available_actions

    @property
    def terminal_scheduler_state_sha256(self) -> str:
        return self.terminal_preselection_state.scheduler_state_sha256


def freeze_condition_gate_invocation_proof_v2(
    *,
    terminal_preselection_state: AdaptivePreselectionState,
    condition_projection: ConditionCalibrationProjectionV1,
    source_candidate_input_sha256: str,
    available_actions: Sequence[AdaptiveTerminalAuditCandidate],
    remaining_budget_minutes: float,
    ordinary_scheduler_proof: AdaptiveTerminalSchedulerProof | None = None,
) -> ConditionGateInvocationProofV2:
    """Freeze the first outcome-free condition-gate invocation point."""

    try:
        state = AdaptivePreselectionState.model_validate(
            terminal_preselection_state.model_dump(mode="json")
        )
        projection = ConditionCalibrationProjectionV1.model_validate(
            condition_projection.model_dump(mode="json")
        )
        actions = sorted(
            (
                AdaptiveTerminalAuditCandidate.model_validate(action.model_dump(mode="json"))
                for action in available_actions
            ),
            key=lambda action: action.item_id,
        )
        ordinary = (
            None
            if ordinary_scheduler_proof is None
            else AdaptiveTerminalSchedulerProof.model_validate(
                ordinary_scheduler_proof.model_dump(mode="json")
            )
        )
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError("condition_invocation_input_integrity_changed") from exc
    resolved = list(state.audit_prefix_item_ids)
    feasible = sorted(
        action.item_id
        for action in actions
        if action.item_id not in set(resolved)
        and action.eligible
        and action.estimated_cost_minutes <= remaining_budget_minutes + _COST_TOLERANCE
    )
    payload: dict[str, Any] = {
        "proof_version": "condition-gate-invocation-proof-v2",
        "invocation_basis": (
            "ordinary_scheduler_terminal"
            if ordinary is not None
            else "first_nonconfirmation_eligible_state"
        ),
        "terminal_reason": (
            ordinary.terminal_reason
            if ordinary is not None
            else "full_nonconfirmation_release_gates_passed"
        ),
        "terminal_preselection_state": state,
        "terminal_preselection_state_sha256": state.state_sha256,
        "condition_projection": projection,
        "condition_projection_sha256": projection.projection_sha256,
        "source_candidate_input_sha256": source_candidate_input_sha256,
        "available_actions": actions,
        "available_action_roster_sha256": hash_canonical(actions),
        "resolved_item_ids": resolved,
        "remaining_budget_minutes": float(remaining_budget_minutes),
        "unresolved_feasible_action_ids": feasible,
        "ordinary_scheduler_proof": ordinary,
        "active_action": False,
        "confirmation_outcomes_unopened": True,
        "reference_labels_unopened": True,
    }
    return ConditionGateInvocationProofV2.model_validate(
        {**payload, "proof_sha256": hash_canonical(payload)}
    )


class AdaptivePolicyArmTrajectory(ContractModel):
    """A full, threshold-independent scheduler path for one policy arm."""

    arm_trajectory_version: Literal["adaptive-policy-arm-trajectory-v1"] = (
        "adaptive-policy-arm-trajectory-v1"
    )
    policy_arm_id: Annotated[str, Field(min_length=1)]
    policy_context_sha256: str
    generation_mode: Literal["threshold_blind_full_scheduler_trajectory"] = (
        "threshold_blind_full_scheduler_trajectory"
    )
    completeness_basis: Literal[
        "externally_attested_complete_scheduler_replay",
        "validated_v5_certificate_sequence",
    ] = "externally_attested_complete_scheduler_replay"
    source_certificate_sha256s: list[str] = Field(default_factory=list)
    terminal_decision_sha256: str | None = None
    states: list[AdaptivePreselectionState]
    terminal_reason: TrajectoryTerminalReason
    terminal_proof: AdaptiveTerminalSchedulerProof | ConditionGateInvocationProofV2
    arm_trajectory_sha256: str

    @field_validator("policy_context_sha256", "arm_trajectory_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("source_certificate_sha256s")
    @classmethod
    def validate_source_hashes(cls, value: list[str]) -> list[str]:
        for digest in value:
            _validate_sha256(digest, "source_certificate_sha256s")
        if len(value) != len(set(value)):
            raise ValueError("adaptive_source_certificate_hashes_not_unique")
        return value

    @field_validator("terminal_decision_sha256")
    @classmethod
    def validate_terminal_hash(cls, value: str | None) -> str | None:
        return None if value is None else _validate_sha256(value, "terminal_decision_sha256")

    @model_validator(mode="after")
    def validate_path(self) -> AdaptivePolicyArmTrajectory:
        if not self.states:
            raise ValueError("adaptive_arm_trajectory_states_empty")
        if self.states[0].prefix_index != 0:
            raise ValueError("adaptive_arm_trajectory_must_start_at_prefix_zero")
        if self.states[0].audit_prefix_item_ids or not math.isclose(
            self.states[0].audit_prefix_cost_minutes, 0.0, abs_tol=_COST_TOLERANCE
        ):
            raise ValueError("adaptive_prefix_zero_must_be_unaudited_and_zero_cost")
        scheduler_state_hashes = [state.scheduler_state_sha256 for state in self.states]
        if len(scheduler_state_hashes) != len(set(scheduler_state_hashes)):
            raise ValueError("adaptive_trajectory_scheduler_states_not_unique")
        if self.completeness_basis == "validated_v5_certificate_sequence":
            if (
                len(self.source_certificate_sha256s) != len(self.states)
                or self.terminal_decision_sha256 is None
            ):
                raise ValueError("adaptive_validated_trajectory_source_lineage_incomplete")
        elif self.source_certificate_sha256s or self.terminal_decision_sha256 is not None:
            raise ValueError("adaptive_attested_trajectory_forbids_certificate_lineage_claim")
        if (
            self.terminal_proof.terminal_reason != self.terminal_reason
            or self.terminal_proof.terminal_scheduler_state_sha256
            != self.states[-1].scheduler_state_sha256
            or self.terminal_proof.resolved_item_ids != self.states[-1].audit_prefix_item_ids
        ):
            raise ValueError("adaptive_trajectory_terminal_proof_state_mismatch")
        if isinstance(self.terminal_proof, AdaptiveTerminalSchedulerProof):
            terminal_state = self.states[-1]
            if self.terminal_reason == "nonconfirmation_context_blocked" and (
                terminal_state.non_calibration_gates_passed
                or self.terminal_proof.nonconfirmation_blocking_reasons
                != terminal_state.non_calibration_blocking_reasons
            ):
                raise ValueError("adaptive_terminal_context_blocker_state_mismatch")
        if isinstance(self.terminal_proof, ConditionGateInvocationProofV2):
            terminal = self.states[-1]
            unscored_terminal = freeze_adaptive_preselection_state(
                prefix_index=terminal.prefix_index,
                audit_prefix_item_ids=terminal.audit_prefix_item_ids,
                audit_prefix_cost_minutes=terminal.audit_prefix_cost_minutes,
                scheduler_state_sha256=terminal.scheduler_state_sha256,
                evidence_graph_sha256=terminal.evidence_graph_sha256,
                synthesis_sha256=terminal.synthesis_sha256,
                non_calibration_assessment_sha256=(terminal.non_calibration_assessment_sha256),
                non_calibration_gates_passed=terminal.non_calibration_gates_passed,
                non_calibration_blocking_reasons=(terminal.non_calibration_blocking_reasons),
                claim_decision=terminal.claim_decision,
                score_features=terminal.score_features,
            )
            if self.terminal_proof.terminal_preselection_state != unscored_terminal or any(
                state.claim_decision == "condition_dependent" and state.non_calibration_gates_passed
                for state in self.states[:-1]
            ):
                raise ValueError("condition_invocation_not_first_canonical_state")
        elif self.terminal_reason == "full_nonconfirmation_release_gates_passed":
            raise ValueError("condition_invocation_proof_required_for_early_stop")
        for previous, current in zip(self.states, self.states[1:], strict=False):
            if current.prefix_index != previous.prefix_index + 1:
                raise ValueError("adaptive_trajectory_prefix_indices_not_contiguous")
            if current.audit_prefix_item_ids[:-1] != previous.audit_prefix_item_ids:
                raise ValueError("adaptive_trajectory_audit_prefix_not_monotone")
            if current.audit_prefix_cost_minutes + _COST_TOLERANCE < (
                previous.audit_prefix_cost_minutes
            ):
                raise ValueError("adaptive_trajectory_cost_not_monotone")
        payload = self.model_dump(mode="json", exclude={"arm_trajectory_sha256"})
        if hash_canonical(payload) != self.arm_trajectory_sha256:
            raise ValueError("adaptive_arm_trajectory_hash_mismatch")
        return self


def freeze_adaptive_policy_arm_trajectory(
    *,
    policy_arm_id: str,
    policy_context_sha256: str,
    states: Sequence[AdaptivePreselectionState],
    terminal_reason: TrajectoryTerminalReason,
    terminal_candidates: Sequence[AdaptiveTerminalAuditCandidate],
    terminal_source_candidate_input_sha256: str,
    terminal_remaining_budget_minutes: float,
    terminal_nonconfirmation_blocking_reasons: Sequence[str] = (),
    source_certificate_sha256s: Sequence[str] = (),
    terminal_decision_sha256: str | None = None,
    terminal_condition_projection: ConditionCalibrationProjectionV1 | None = None,
    terminal_condition_invocation_proof: ConditionGateInvocationProofV2 | None = None,
) -> AdaptivePolicyArmTrajectory:
    try:
        normalized_states = [
            AdaptivePreselectionState.model_validate(state.model_dump(mode="json"))
            for state in states
        ]
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError("adaptive_preselection_state_integrity_changed") from exc
    if not normalized_states:
        raise AdaptiveCalibrationError("adaptive_arm_trajectory_states_empty")
    supplied_invocation = (
        None
        if terminal_condition_invocation_proof is None
        else ConditionGateInvocationProofV2.model_validate(
            terminal_condition_invocation_proof.model_dump(mode="json")
        )
    )
    ordinary_proof = (
        None
        if terminal_reason == "full_nonconfirmation_release_gates_passed"
        else freeze_adaptive_terminal_scheduler_proof(
            terminal_reason=terminal_reason,
            terminal_scheduler_state_sha256=normalized_states[-1].scheduler_state_sha256,
            source_candidate_input_sha256=terminal_source_candidate_input_sha256,
            candidates=terminal_candidates,
            resolved_item_ids=normalized_states[-1].audit_prefix_item_ids,
            remaining_budget_minutes=terminal_remaining_budget_minutes,
            nonconfirmation_blocking_reasons=(terminal_nonconfirmation_blocking_reasons),
        )
    )
    if terminal_condition_projection is None:
        if supplied_invocation is not None:
            raise AdaptiveCalibrationError("condition_invocation_proof_without_projection")
        if ordinary_proof is None:
            raise AdaptiveCalibrationError(
                "condition_invocation_projection_required_for_early_stop"
            )
        terminal_proof: AdaptiveTerminalSchedulerProof | ConditionGateInvocationProofV2 = (
            ordinary_proof
        )
    else:
        if supplied_invocation is None:
            terminal_proof = freeze_condition_gate_invocation_proof_v2(
                terminal_preselection_state=normalized_states[-1],
                condition_projection=terminal_condition_projection,
                source_candidate_input_sha256=terminal_source_candidate_input_sha256,
                available_actions=terminal_candidates,
                remaining_budget_minutes=terminal_remaining_budget_minutes,
                ordinary_scheduler_proof=ordinary_proof,
            )
        else:
            normalized_candidates = sorted(
                (
                    AdaptiveTerminalAuditCandidate.model_validate(candidate.model_dump(mode="json"))
                    for candidate in terminal_candidates
                ),
                key=lambda candidate: candidate.item_id,
            )
            if (
                supplied_invocation.condition_projection != terminal_condition_projection
                or supplied_invocation.terminal_reason != terminal_reason
                or supplied_invocation.source_candidate_input_sha256
                != terminal_source_candidate_input_sha256
                or supplied_invocation.available_actions != normalized_candidates
                or not math.isclose(
                    supplied_invocation.remaining_budget_minutes,
                    terminal_remaining_budget_minutes,
                    rel_tol=0.0,
                    abs_tol=_COST_TOLERANCE,
                )
            ):
                raise AdaptiveCalibrationError("condition_invocation_proof_factory_input_mismatch")
            terminal_proof = supplied_invocation
    payload: dict[str, Any] = {
        "arm_trajectory_version": "adaptive-policy-arm-trajectory-v1",
        "policy_arm_id": policy_arm_id,
        "policy_context_sha256": policy_context_sha256,
        "generation_mode": "threshold_blind_full_scheduler_trajectory",
        "completeness_basis": (
            "validated_v5_certificate_sequence"
            if source_certificate_sha256s or terminal_decision_sha256 is not None
            else "externally_attested_complete_scheduler_replay"
        ),
        "source_certificate_sha256s": list(source_certificate_sha256s),
        "terminal_decision_sha256": terminal_decision_sha256,
        "states": normalized_states,
        "terminal_reason": terminal_reason,
        "terminal_proof": terminal_proof,
    }
    return AdaptivePolicyArmTrajectory.model_validate(
        {**payload, "arm_trajectory_sha256": hash_canonical(payload)}
    )


class PolicyVisibleQuestionTrajectory(ContractModel):
    """Complete multi-arm question path with no reference verdict or loss label."""

    trajectory_version: Literal["policy-visible-question-trajectory-v1"] = (
        "policy-visible-question-trajectory-v1"
    )
    question_id: Annotated[str, Field(min_length=1)]
    split: AdaptiveSplit
    population_id: Annotated[str, Field(min_length=1)]
    domain: Annotated[str, Field(min_length=1)]
    corpus: CompleteCorpusIdentity
    arms: list[AdaptivePolicyArmTrajectory]
    trajectory_sha256: str

    @field_validator("trajectory_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "trajectory_sha256")

    @model_validator(mode="after")
    def validate_trajectory(self) -> PolicyVisibleQuestionTrajectory:
        arm_ids = [arm.policy_arm_id for arm in self.arms]
        if not arm_ids or arm_ids != sorted(set(arm_ids)):
            raise ValueError("adaptive_trajectory_arms_must_be_nonempty_sorted_unique")
        context_hashes = [arm.policy_context_sha256 for arm in self.arms]
        if len(context_hashes) != len(set(context_hashes)):
            raise ValueError("adaptive_trajectory_policy_contexts_duplicate")
        payload = self.model_dump(mode="json", exclude={"trajectory_sha256"})
        _reject_reference_leakage(payload)
        if hash_canonical(payload) != self.trajectory_sha256:
            raise ValueError("adaptive_question_trajectory_hash_mismatch")
        return self


def freeze_policy_visible_question_trajectory(
    *,
    question_id: str,
    split: AdaptiveSplit,
    population_id: str,
    domain: str,
    corpus: CompleteCorpusIdentity,
    arms: Sequence[AdaptivePolicyArmTrajectory],
) -> PolicyVisibleQuestionTrajectory:
    try:
        normalized_corpus = CompleteCorpusIdentity.model_validate(corpus.model_dump(mode="json"))
        normalized_arms = [
            AdaptivePolicyArmTrajectory.model_validate(arm.model_dump(mode="json")) for arm in arms
        ]
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError("adaptive_policy_visible_input_integrity_changed") from exc
    payload: dict[str, Any] = {
        "trajectory_version": "policy-visible-question-trajectory-v1",
        "question_id": question_id,
        "split": split,
        "population_id": population_id,
        "domain": domain,
        "corpus": normalized_corpus,
        "arms": sorted(normalized_arms, key=lambda arm: arm.policy_arm_id),
    }
    return PolicyVisibleQuestionTrajectory.model_validate(
        {**payload, "trajectory_sha256": hash_canonical(payload)}
    )


class QuestionReferenceVerdict(ContractModel):
    """Hidden adjudication sidecar, never passed to scheduler or release scoring."""

    reference_version: Literal["question-reference-verdict-v1"] = "question-reference-verdict-v1"
    question_id: Annotated[str, Field(min_length=1)]
    verdict: Annotated[str, Field(min_length=1)]
    label_source: AdaptiveLabelSource
    adjudication_protocol_sha256: str
    adjudication_artifact_sha256: str
    reference_sha256: str

    @field_validator(
        "adjudication_protocol_sha256",
        "adjudication_artifact_sha256",
        "reference_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_reference(self) -> QuestionReferenceVerdict:
        payload = self.model_dump(mode="json", exclude={"reference_sha256"})
        if hash_canonical(payload) != self.reference_sha256:
            raise ValueError("adaptive_reference_verdict_hash_mismatch")
        return self


def freeze_question_reference_verdict(
    *,
    question_id: str,
    verdict: str,
    label_source: AdaptiveLabelSource,
    adjudication_protocol_sha256: str,
    adjudication_artifact_sha256: str,
) -> QuestionReferenceVerdict:
    payload = {
        "reference_version": "question-reference-verdict-v1",
        "question_id": question_id,
        "verdict": verdict,
        "label_source": label_source,
        "adjudication_protocol_sha256": adjudication_protocol_sha256,
        "adjudication_artifact_sha256": adjudication_artifact_sha256,
    }
    return QuestionReferenceVerdict.model_validate(
        {**payload, "reference_sha256": hash_canonical(payload)}
    )


class LabeledQuestionTrajectory(ContractModel):
    """Calibration-only join between a label-free trajectory and hidden reference."""

    labeled_version: Literal["labeled-question-trajectory-v1"] = "labeled-question-trajectory-v1"
    visible: PolicyVisibleQuestionTrajectory
    reference: QuestionReferenceVerdict
    labeled_trajectory_sha256: str

    @field_validator("labeled_trajectory_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "labeled_trajectory_sha256")

    @model_validator(mode="after")
    def validate_join(self) -> LabeledQuestionTrajectory:
        if self.reference.question_id != self.visible.question_id:
            raise ValueError("adaptive_reference_question_identity_mismatch")
        _reject_reference_leakage(self.visible.model_dump(mode="json"))
        payload = self.model_dump(mode="json", exclude={"labeled_trajectory_sha256"})
        if hash_canonical(payload) != self.labeled_trajectory_sha256:
            raise ValueError("adaptive_labeled_trajectory_hash_mismatch")
        return self


def join_labeled_question_trajectory(
    *, visible: PolicyVisibleQuestionTrajectory, reference: QuestionReferenceVerdict
) -> LabeledQuestionTrajectory:
    try:
        normalized_visible = PolicyVisibleQuestionTrajectory.model_validate(
            visible.model_dump(mode="json")
        )
        normalized_reference = QuestionReferenceVerdict.model_validate(
            reference.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError("adaptive_labeled_join_input_integrity_changed") from exc
    payload = {
        "labeled_version": "labeled-question-trajectory-v1",
        "visible": normalized_visible,
        "reference": normalized_reference,
    }
    return LabeledQuestionTrajectory.model_validate(
        {**payload, "labeled_trajectory_sha256": hash_canonical(payload)}
    )


class AdaptiveTrajectoryScoreModel(ContractModel):
    """Development-only, question-weighted state risk model for one policy arm."""

    model_version: Literal["adaptive-trajectory-logistic-v1"] = "adaptive-trajectory-logistic-v1"
    policy_arm_id: str
    policy_context_sha256: str
    feature_names: list[str]
    means: list[float]
    scales: list[float]
    coefficients: list[float]
    intercept: float
    fitting_method: Literal["question_weighted_logistic", "smoothed_constant"]
    development_question_ids: list[str]
    seed: int
    score_model_sha256: str

    @field_validator("policy_context_sha256", "score_model_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_model(self) -> AdaptiveTrajectoryScoreModel:
        if not self.policy_arm_id:
            raise ValueError("adaptive_model_policy_arm_id_empty")
        width = len(self.feature_names)
        if not width or self.feature_names != sorted(set(self.feature_names)):
            raise ValueError("adaptive_model_feature_names_invalid")
        if not all(len(values) == width for values in (self.means, self.scales, self.coefficients)):
            raise ValueError("adaptive_model_dimensions_mismatch")
        if any(scale <= 0 or not math.isfinite(scale) for scale in self.scales):
            raise ValueError("adaptive_model_scales_invalid")
        if any(
            not math.isfinite(number)
            for number in (*self.means, *self.coefficients, self.intercept)
        ):
            raise ValueError("adaptive_model_values_nonfinite")
        if not self.development_question_ids or self.development_question_ids != sorted(
            set(self.development_question_ids)
        ):
            raise ValueError("adaptive_model_development_questions_invalid")
        payload = self.model_dump(mode="json", exclude={"score_model_sha256"})
        if hash_canonical(payload) != self.score_model_sha256:
            raise ValueError("adaptive_score_model_hash_mismatch")
        return self

    def score_features(self, features: Mapping[str, float]) -> float:
        if list(sorted(features)) != self.feature_names:
            raise AdaptiveCalibrationError("adaptive_score_feature_schema_mismatch")
        standardized = [
            (float(features[name]) - mean) / scale
            for name, mean, scale in zip(self.feature_names, self.means, self.scales, strict=True)
        ]
        logit = self.intercept + math.fsum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, standardized, strict=True)
        )
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)


class AdaptiveThresholdCandidate(ContractModel):
    candidate_version: Literal["adaptive-threshold-candidate-v1"] = (
        "adaptive-threshold-candidate-v1"
    )
    policy_arm_id: str
    policy_context_sha256: str
    score_model_sha256: str
    threshold: Annotated[float, Field(ge=0, le=1)]
    candidate_sha256: str

    @field_validator("policy_context_sha256", "score_model_sha256", "candidate_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_candidate(self) -> AdaptiveThresholdCandidate:
        if not self.policy_arm_id:
            raise ValueError("adaptive_threshold_candidate_policy_arm_id_empty")
        payload = self.model_dump(mode="json", exclude={"candidate_sha256"})
        if hash_canonical(payload) != self.candidate_sha256:
            raise ValueError("adaptive_threshold_candidate_hash_mismatch")
        return self


def _freeze_threshold_candidate(
    *,
    policy_arm_id: str,
    policy_context_sha256: str,
    score_model_sha256: str,
    threshold: float,
) -> AdaptiveThresholdCandidate:
    payload = {
        "candidate_version": "adaptive-threshold-candidate-v1",
        "policy_arm_id": policy_arm_id,
        "policy_context_sha256": policy_context_sha256,
        "score_model_sha256": score_model_sha256,
        "threshold": float(threshold),
    }
    return AdaptiveThresholdCandidate.model_validate(
        {**payload, "candidate_sha256": hash_canonical(payload)}
    )


class AdaptiveThresholdFamily(ContractModel):
    """Candidate family frozen from development before calibration labels open."""

    family_version: Literal["adaptive-threshold-family-v1"] = "adaptive-threshold-family-v1"
    definition_source: Literal["development_only"] = "development_only"
    development_question_ids: list[str]
    development_visible_input_sha256: str
    candidates: list[AdaptiveThresholdCandidate]
    family_sha256: str

    @field_validator("development_visible_input_sha256", "family_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_family(self) -> AdaptiveThresholdFamily:
        if not self.development_question_ids or self.development_question_ids != sorted(
            set(self.development_question_ids)
        ):
            raise ValueError("adaptive_family_development_questions_invalid")
        ordered = sorted(
            self.candidates,
            key=lambda row: (row.policy_arm_id, row.threshold, row.candidate_sha256),
        )
        if not ordered or self.candidates != ordered:
            raise ValueError("adaptive_threshold_candidates_not_sorted")
        if len({row.candidate_sha256 for row in self.candidates}) != len(self.candidates):
            raise ValueError("adaptive_threshold_candidates_duplicate")
        payload = self.model_dump(mode="json", exclude={"family_sha256"})
        if hash_canonical(payload) != self.family_sha256:
            raise ValueError("adaptive_threshold_family_hash_mismatch")
        return self


class AdaptiveCalibrationPlan(ContractModel):
    """Risk target and selection procedure sealed before calibration labels open."""

    plan_version: Literal["adaptive-calibration-plan-v1"] = "adaptive-calibration-plan-v1"
    alpha: Annotated[float, Field(gt=0, lt=1)]
    delta: Annotated[float, Field(gt=0, lt=1)]
    threshold_family_sha256: str
    reference_loss: Literal["exact_claim_decision_mismatch"] = "exact_claim_decision_mismatch"
    multiplicity_correction: Literal["bonferroni-clopper-pearson-across-arms-and-thresholds"] = (
        "bonferroni-clopper-pearson-across-arms-and-thresholds"
    )
    candidate_selection_rule: Literal[
        "max-accepted-then-min-upper-risk-then-max-threshold-then-max-arm-id"
    ] = "max-accepted-then-min-upper-risk-then-max-threshold-then-max-arm-id"
    plan_sha256: str

    @field_validator("threshold_family_sha256", "plan_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_plan(self) -> AdaptiveCalibrationPlan:
        payload = self.model_dump(mode="json", exclude={"plan_sha256"})
        if hash_canonical(payload) != self.plan_sha256:
            raise ValueError("adaptive_calibration_plan_hash_mismatch")
        return self


def _freeze_adaptive_calibration_plan(
    *,
    alpha: float,
    delta: float,
    threshold_family_sha256: str,
) -> AdaptiveCalibrationPlan:
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise AdaptiveCalibrationError("adaptive_alpha_delta_must_be_between_zero_and_one")
    payload: dict[str, Any] = {
        "plan_version": "adaptive-calibration-plan-v1",
        "alpha": float(alpha),
        "delta": float(delta),
        "threshold_family_sha256": threshold_family_sha256,
        "reference_loss": "exact_claim_decision_mismatch",
        "multiplicity_correction": ("bonferroni-clopper-pearson-across-arms-and-thresholds"),
        "candidate_selection_rule": (
            "max-accepted-then-min-upper-risk-then-max-threshold-then-max-arm-id"
        ),
    }
    return AdaptiveCalibrationPlan.model_validate(
        {**payload, "plan_sha256": hash_canonical(payload)}
    )


class AdaptiveCalibrationRoster(ContractModel):
    """Exact complete label-free calibration trajectories frozen before labels open."""

    roster_version: Literal["adaptive-calibration-roster-v1"] = "adaptive-calibration-roster-v1"
    freeze_state: Literal["calibration_labels_unopened"] = "calibration_labels_unopened"
    visible_trajectories: list[PolicyVisibleQuestionTrajectory]
    roster_sha256: str

    @field_validator("roster_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "roster_sha256")

    @model_validator(mode="after")
    def validate_roster(self) -> AdaptiveCalibrationRoster:
        question_ids = [row.question_id for row in self.visible_trajectories]
        if not question_ids or question_ids != sorted(set(question_ids)):
            raise ValueError("adaptive_calibration_roster_questions_invalid")
        if any(row.split != "calibration" for row in self.visible_trajectories):
            raise ValueError("adaptive_calibration_roster_split_mismatch")
        if any(
            state.scalar_risk_score is not None
            for row in self.visible_trajectories
            for arm in row.arms
            for state in arm.states
        ):
            raise ValueError("adaptive_calibration_roster_must_be_unscored")
        payload = self.model_dump(mode="json", exclude={"roster_sha256"})
        _reject_reference_leakage(payload)
        if hash_canonical(payload) != self.roster_sha256:
            raise ValueError("adaptive_calibration_roster_hash_mismatch")
        return self


def _freeze_adaptive_calibration_roster(
    visible_trajectories: Sequence[PolicyVisibleQuestionTrajectory],
) -> AdaptiveCalibrationRoster:
    payload: dict[str, Any] = {
        "roster_version": "adaptive-calibration-roster-v1",
        "freeze_state": "calibration_labels_unopened",
        "visible_trajectories": list(visible_trajectories),
    }
    return AdaptiveCalibrationRoster.model_validate(
        {**payload, "roster_sha256": hash_canonical(payload)}
    )


class AdaptiveSplitIdentity(ContractModel):
    split: Literal["development", "calibration"]
    question_ids: list[str]
    complete_publication_ids: list[str]
    domains: list[str]
    labeled_trajectory_sha256s: list[str]

    @model_validator(mode="after")
    def validate_identity(self) -> AdaptiveSplitIdentity:
        for name in (
            "question_ids",
            "complete_publication_ids",
            "domains",
            "labeled_trajectory_sha256s",
        ):
            values = getattr(self, name)
            if values != sorted(set(values)) or (name != "complete_publication_ids" and not values):
                raise ValueError(f"adaptive_split_identity_invalid:{name}")
        for digest in self.labeled_trajectory_sha256s:
            _validate_sha256(digest, "labeled_trajectory_sha256s")
        return self


class AdaptiveDevelopmentFreeze(ContractModel):
    """Development-only score models and threshold family frozen pre-calibration."""

    freeze_version: Literal["adaptive-development-freeze-v1"] = "adaptive-development-freeze-v1"
    freeze_state: Literal["calibration_labels_unopened"] = "calibration_labels_unopened"
    population_id: str
    policy_contexts: list[AdaptivePolicyContext]
    score_models: list[AdaptiveTrajectoryScoreModel]
    threshold_family: AdaptiveThresholdFamily
    calibration_plan: AdaptiveCalibrationPlan
    calibration_roster: AdaptiveCalibrationRoster
    development: AdaptiveSplitIdentity
    scored_development_trajectories: list[LabeledQuestionTrajectory]
    development_input_sha256: str
    development_freeze_sha256: str

    @field_validator("development_input_sha256", "development_freeze_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_freeze(self) -> AdaptiveDevelopmentFreeze:
        context_ids = [context.policy_arm_id for context in self.policy_contexts]
        model_ids = [model.policy_arm_id for model in self.score_models]
        if not context_ids or context_ids != sorted(set(context_ids)) or model_ids != context_ids:
            raise ValueError("adaptive_development_context_model_identity_mismatch")
        if any(context.population_id != self.population_id for context in self.policy_contexts):
            raise ValueError("adaptive_development_population_mismatch")
        _validate_labeled_trajectory_family(
            self.scored_development_trajectories,
            contexts=self.policy_contexts,
            required_split="development",
            require_scored=True,
            score_models=self.score_models,
        )
        expected_identity = _split_identity(
            self.scored_development_trajectories, split="development"
        )
        if self.development != expected_identity:
            raise ValueError("adaptive_development_identity_mismatch")
        if self.threshold_family.development_question_ids != self.development.question_ids:
            raise ValueError("adaptive_threshold_family_question_identity_mismatch")
        if self.calibration_plan.threshold_family_sha256 != self.threshold_family.family_sha256:
            raise ValueError("adaptive_calibration_plan_threshold_family_mismatch")
        _validate_policy_visible_trajectory_family(
            self.calibration_roster.visible_trajectories,
            contexts=self.policy_contexts,
            required_split="calibration",
            require_scored=False,
        )
        _validate_cross_split_independence(
            self.scored_development_trajectories,
            self.calibration_roster.visible_trajectories,
        )
        context_by_arm = {context.policy_arm_id: context for context in self.policy_contexts}
        model_by_arm = {model.policy_arm_id: model for model in self.score_models}
        for arm_id in context_ids:
            context = context_by_arm[arm_id]
            model = model_by_arm[arm_id]
            if model.policy_context_sha256 != context.policy_context_sha256:
                raise ValueError("adaptive_score_model_policy_context_mismatch")
            if model.feature_names != context.score_feature_names:
                raise ValueError("adaptive_score_model_feature_context_mismatch")
            if model.development_question_ids != self.development.question_ids:
                raise ValueError("adaptive_score_model_development_identity_mismatch")
        candidate_arms = {row.policy_arm_id for row in self.threshold_family.candidates}
        if candidate_arms != set(context_ids):
            raise ValueError("adaptive_threshold_candidate_arm_coverage_mismatch")
        for candidate in self.threshold_family.candidates:
            context = context_by_arm[candidate.policy_arm_id]
            model = model_by_arm[candidate.policy_arm_id]
            if candidate.policy_context_sha256 != context.policy_context_sha256:
                raise ValueError("adaptive_threshold_candidate_context_mismatch")
            if candidate.score_model_sha256 != model.score_model_sha256:
                raise ValueError("adaptive_threshold_candidate_model_mismatch")
        visible_hash = hash_canonical([row.visible for row in self.scored_development_trajectories])
        if self.threshold_family.development_visible_input_sha256 != visible_hash:
            raise ValueError("adaptive_threshold_family_visible_input_mismatch")
        expected_input_hash = hash_canonical(
            [
                _unscore_labeled_question_trajectory(row)
                for row in self.scored_development_trajectories
            ]
        )
        if self.development_input_sha256 != expected_input_hash:
            raise ValueError("adaptive_development_input_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"development_freeze_sha256"})
        if hash_canonical(payload) != self.development_freeze_sha256:
            raise ValueError("adaptive_development_freeze_hash_mismatch")
        return self


class AdaptiveQuestionOutcome(ContractModel):
    """Exactly one replay result for one candidate and one complete question."""

    question_id: str
    candidate_sha256: str
    accepted: bool
    error: bool
    first_release_prefix_index: int | None = Field(default=None, ge=0)
    release_state_sha256: str | None = None
    scalar_risk_score: Annotated[float, Field(ge=0, le=1)] | None = None
    reference_sha256: str

    @field_validator("candidate_sha256", "reference_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("release_state_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return None if value is None else _validate_sha256(value, "release_state_sha256")

    @field_validator("accepted", "error", mode="before")
    @classmethod
    def validate_strict_boolean(cls, value: object, info: Any) -> object:
        if not isinstance(value, bool):
            raise ValueError(f"adaptive_outcome_{info.field_name}_must_be_boolean")
        return value

    @model_validator(mode="after")
    def validate_outcome(self) -> AdaptiveQuestionOutcome:
        if self.error and not self.accepted:
            raise ValueError("adaptive_abstention_cannot_count_as_error")
        release_fields = (
            self.first_release_prefix_index,
            self.release_state_sha256,
            self.scalar_risk_score,
        )
        if self.accepted != all(value is not None for value in release_fields):
            raise ValueError("adaptive_outcome_release_fields_mismatch")
        return self


class AdaptiveCandidateCalibration(ContractModel):
    candidate: AdaptiveThresholdCandidate
    total_questions: Annotated[int, Field(ge=1)]
    outcomes: list[AdaptiveQuestionOutcome]
    accepted: Annotated[int, Field(ge=0)]
    errors: Annotated[int, Field(ge=0)]
    empirical_risk: float | None
    simultaneous_upper_risk: float | None
    passed: bool

    @model_validator(mode="after")
    def validate_counts(self) -> AdaptiveCandidateCalibration:
        if len(self.outcomes) != self.total_questions:
            raise ValueError("adaptive_candidate_outcome_denominator_mismatch")
        question_ids = [row.question_id for row in self.outcomes]
        if question_ids != sorted(set(question_ids)):
            raise ValueError("adaptive_candidate_outcomes_not_question_unique")
        if any(row.candidate_sha256 != self.candidate.candidate_sha256 for row in self.outcomes):
            raise ValueError("adaptive_candidate_outcome_identity_mismatch")
        if self.accepted != sum(row.accepted for row in self.outcomes):
            raise ValueError("adaptive_candidate_accepted_count_mismatch")
        if self.errors != sum(row.error for row in self.outcomes):
            raise ValueError("adaptive_candidate_error_count_mismatch")
        if self.errors > self.accepted:
            raise ValueError("adaptive_candidate_errors_exceed_accepted")
        expected_empirical = self.errors / self.accepted if self.accepted else None
        if self.empirical_risk != expected_empirical:
            raise ValueError("adaptive_candidate_empirical_risk_mismatch")
        return self


class AdaptiveCalibrationBundle(ContractModel):
    """Frozen simultaneous first-release guarantee over complete trajectories."""

    bundle_version: Literal["adaptive-question-trajectory-freeze-v1"] = (
        "adaptive-question-trajectory-freeze-v1"
    )
    freeze_state: Literal["test_labels_unopened"] = "test_labels_unopened"
    guarantee_scope: Literal[
        "exact decision-mismatch risk against the frozen adjudication protocol under "
        "exchangeable independent complete-question trajectories; not scientific truth "
        "or domain-shift robustness"
    ] = (
        "exact decision-mismatch risk against the frozen adjudication protocol under "
        "exchangeable independent complete-question trajectories; not scientific truth "
        "or domain-shift robustness"
    )
    population_id: str
    label_source: AdaptiveLabelSource
    adjudication_protocol_sha256: str
    alpha: Annotated[float, Field(gt=0, lt=1)]
    delta: Annotated[float, Field(gt=0, lt=1)]
    correction: Literal["bonferroni-clopper-pearson-across-arms-and-thresholds"] = (
        "bonferroni-clopper-pearson-across-arms-and-thresholds"
    )
    candidate_selection_rule: Literal[
        "max-accepted-then-min-upper-risk-then-max-threshold-then-max-arm-id"
    ] = "max-accepted-then-min-upper-risk-then-max-threshold-then-max-arm-id"
    development_freeze: AdaptiveDevelopmentFreeze
    development_freeze_sha256: str
    calibration: AdaptiveSplitIdentity
    scored_calibration_trajectories: list[LabeledQuestionTrajectory]
    calibration_input_sha256: str
    candidates: list[AdaptiveCandidateCalibration]
    selected_candidate_sha256: str | None
    status: Literal["calibrated", "abstain_all"]
    bundle_sha256: str

    @field_validator(
        "adjudication_protocol_sha256",
        "development_freeze_sha256",
        "calibration_input_sha256",
        "bundle_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("selected_candidate_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return None if value is None else _validate_sha256(value, "selected_candidate_sha256")

    @model_validator(mode="after")
    def validate_bundle(self) -> AdaptiveCalibrationBundle:
        if any(
            arm.terminal_reason == "full_nonconfirmation_release_gates_passed"
            for trajectory in (
                *self.development_freeze.scored_development_trajectories,
                *self.scored_calibration_trajectories,
            )
            for arm in trajectory.visible.arms
        ):
            raise ValueError("adaptive_v1_forbids_confirmation_aware_early_stop")
        if self.development_freeze.development_freeze_sha256 != (self.development_freeze_sha256):
            raise ValueError("adaptive_bundle_development_freeze_hash_mismatch")
        if self.development_freeze.population_id != self.population_id:
            raise ValueError("adaptive_bundle_population_mismatch")
        plan = self.development_freeze.calibration_plan
        if self.alpha != plan.alpha or self.delta != plan.delta:
            raise ValueError("adaptive_bundle_calibration_plan_risk_target_mismatch")
        if self.correction != plan.multiplicity_correction:
            raise ValueError("adaptive_bundle_calibration_plan_correction_mismatch")
        if self.candidate_selection_rule != plan.candidate_selection_rule:
            raise ValueError("adaptive_bundle_candidate_selection_rule_mismatch")
        _validate_labeled_trajectory_family(
            self.scored_calibration_trajectories,
            contexts=self.development_freeze.policy_contexts,
            required_split="calibration",
            require_scored=True,
            score_models=self.development_freeze.score_models,
        )
        unscored_calibration = [
            _unscore_labeled_question_trajectory(row)
            for row in self.scored_calibration_trajectories
        ]
        if [row.visible for row in unscored_calibration] != (
            self.development_freeze.calibration_roster.visible_trajectories
        ):
            raise ValueError("adaptive_calibration_visible_roster_mismatch")
        expected_identity = _split_identity(
            self.scored_calibration_trajectories, split="calibration"
        )
        if self.calibration != expected_identity:
            raise ValueError("adaptive_calibration_identity_mismatch")
        _validate_cross_split_independence(
            self.development_freeze.scored_development_trajectories,
            self.scored_calibration_trajectories,
        )
        if {row.reference.label_source for row in self.scored_calibration_trajectories} != {
            self.label_source
        }:
            raise ValueError("adaptive_calibration_label_source_mismatch")
        if {
            row.reference.adjudication_protocol_sha256
            for row in self.scored_calibration_trajectories
        } != {self.adjudication_protocol_sha256}:
            raise ValueError("adaptive_calibration_adjudication_protocol_mismatch")
        expected_calibration_input_hash = hash_canonical(unscored_calibration)
        if self.calibration_input_sha256 != expected_calibration_input_hash:
            raise ValueError("adaptive_calibration_input_hash_mismatch")
        family = self.development_freeze.threshold_family.candidates
        if [row.candidate for row in self.candidates] != family:
            raise ValueError("adaptive_calibration_candidate_family_mismatch")
        simultaneous_delta = self.delta / len(self.candidates)
        for row in self.candidates:
            expected_outcomes = _replay_candidate(
                self.scored_calibration_trajectories, row.candidate
            )
            if row.outcomes != expected_outcomes:
                raise ValueError("adaptive_calibration_replay_outcome_mismatch")
            expected_upper = (
                clopper_pearson_upper(row.errors, row.accepted, delta=simultaneous_delta)
                if row.accepted
                else None
            )
            if expected_upper is None:
                if row.simultaneous_upper_risk is not None:
                    raise ValueError("adaptive_calibration_upper_risk_mismatch")
            elif row.simultaneous_upper_risk is None or not math.isclose(
                row.simultaneous_upper_risk,
                expected_upper,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError("adaptive_calibration_upper_risk_mismatch")
            if row.passed != (expected_upper is not None and expected_upper <= self.alpha):
                raise ValueError("adaptive_calibration_pass_status_mismatch")
        passing = [row for row in self.candidates if row.passed]
        selected = _select_calibrated_candidate(passing)
        selected_hash = None if selected is None else selected.candidate.candidate_sha256
        if self.selected_candidate_sha256 != selected_hash:
            raise ValueError("adaptive_calibration_selected_candidate_mismatch")
        if (self.status == "calibrated") != (selected is not None):
            raise ValueError("adaptive_calibration_status_mismatch")
        payload = self.model_dump(mode="json", exclude={"bundle_sha256"})
        if hash_canonical(payload) != self.bundle_sha256:
            raise ValueError("adaptive_calibration_bundle_hash_mismatch")
        return self

    @property
    def selected(self) -> AdaptiveCandidateCalibration | None:
        if self.selected_candidate_sha256 is None:
            return None
        return next(
            row
            for row in self.candidates
            if row.candidate.candidate_sha256 == self.selected_candidate_sha256
        )


def _split_identity(
    rows: Sequence[LabeledQuestionTrajectory],
    *,
    split: Literal["development", "calibration"],
) -> AdaptiveSplitIdentity:
    return AdaptiveSplitIdentity(
        split=split,
        question_ids=sorted(row.visible.question_id for row in rows),
        complete_publication_ids=sorted(
            {
                publication_id
                for row in rows
                for publication_id in row.visible.corpus.publication_ids
            }
        ),
        domains=sorted({row.visible.domain for row in rows}),
        labeled_trajectory_sha256s=sorted(row.labeled_trajectory_sha256 for row in rows),
    )


def _validate_policy_visible_trajectory_family(
    rows: Sequence[PolicyVisibleQuestionTrajectory],
    *,
    contexts: Sequence[AdaptivePolicyContext],
    required_split: AdaptiveSplit,
    require_scored: bool,
    score_models: Sequence[AdaptiveTrajectoryScoreModel] = (),
) -> None:
    if not rows:
        raise AdaptiveCalibrationError(f"adaptive_{required_split}_trajectories_empty")
    if not contexts:
        raise AdaptiveCalibrationError("adaptive_policy_contexts_empty")
    question_ids = [row.question_id for row in rows]
    if question_ids != sorted(set(question_ids)):
        raise AdaptiveCalibrationError(f"adaptive_{required_split}_questions_must_be_sorted_unique")
    expected_arms = [context.policy_arm_id for context in contexts]
    context_by_arm = {context.policy_arm_id: context for context in contexts}
    model_by_arm = {model.policy_arm_id: model for model in score_models}
    populations = {row.population_id for row in rows}
    if populations != {contexts[0].population_id}:
        raise AdaptiveCalibrationError("adaptive_trajectory_population_mismatch")
    publication_owner: dict[str, str] = {}
    corpus_owner: dict[str, str] = {}
    source_manifest_owner: dict[str, str] = {}
    for visible in rows:
        if visible.split != required_split:
            raise AdaptiveCalibrationError("adaptive_trajectory_split_mismatch")
        if [arm.policy_arm_id for arm in visible.arms] != expected_arms:
            raise AdaptiveCalibrationError("adaptive_trajectory_policy_arm_family_mismatch")
        corpus_owner_question = corpus_owner.setdefault(
            visible.corpus.corpus_id, visible.question_id
        )
        if corpus_owner_question != visible.question_id:
            raise AdaptiveCalibrationError(
                f"complete_corpus_id_shared_between_questions:{visible.corpus.corpus_id}"
            )
        if visible.corpus.source_manifest_sha256 is not None:
            manifest_owner_question = source_manifest_owner.setdefault(
                visible.corpus.source_manifest_sha256, visible.question_id
            )
            if manifest_owner_question != visible.question_id:
                raise AdaptiveCalibrationError(
                    "complete_corpus_source_manifest_shared_between_questions:"
                    f"{visible.corpus.source_manifest_sha256}"
                )
        for arm in visible.arms:
            context = context_by_arm[arm.policy_arm_id]
            if arm.policy_context_sha256 != context.policy_context_sha256:
                raise AdaptiveCalibrationError("adaptive_trajectory_policy_context_mismatch")
            expected_cutoff = context.corpus_protocol_context.get("corpus_cutoff")
            if isinstance(expected_cutoff, str) and expected_cutoff != visible.corpus.corpus_cutoff:
                raise AdaptiveCalibrationError("adaptive_trajectory_corpus_cutoff_context_mismatch")
            for state in arm.states:
                if list(state.score_features) != context.score_feature_names:
                    raise AdaptiveCalibrationError("adaptive_trajectory_feature_schema_mismatch")
                if state.audit_prefix_cost_minutes > context.budget_minutes + _COST_TOLERANCE:
                    raise AdaptiveCalibrationError("adaptive_trajectory_prefix_exceeds_budget")
                if require_scored:
                    model = model_by_arm.get(arm.policy_arm_id)
                    if model is None or state.score_model_sha256 != model.score_model_sha256:
                        raise AdaptiveCalibrationError("adaptive_trajectory_score_model_mismatch")
                    expected_score = model.score_features(state.score_features)
                    if state.scalar_risk_score is None or not math.isclose(
                        state.scalar_risk_score,
                        expected_score,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    ):
                        raise AdaptiveCalibrationError("adaptive_trajectory_score_mismatch")
                elif state.scalar_risk_score is not None:
                    raise AdaptiveCalibrationError("adaptive_development_input_must_be_unscored")
            if arm.terminal_reason == "budget_exhausted" and not math.isclose(
                arm.states[-1].audit_prefix_cost_minutes,
                context.budget_minutes,
                rel_tol=0.0,
                abs_tol=_COST_TOLERANCE,
            ):
                raise AdaptiveCalibrationError("adaptive_budget_exhausted_terminal_cost_mismatch")
            expected_remaining = max(
                0.0,
                context.budget_minutes - arm.states[-1].audit_prefix_cost_minutes,
            )
            if not math.isclose(
                arm.terminal_proof.remaining_budget_minutes,
                expected_remaining,
                rel_tol=0.0,
                abs_tol=_COST_TOLERANCE,
            ):
                raise AdaptiveCalibrationError("adaptive_terminal_proof_budget_ledger_mismatch")
        for publication_id in visible.corpus.publication_ids:
            owner = publication_owner.setdefault(publication_id, visible.question_id)
            if owner != visible.question_id:
                raise AdaptiveCalibrationError(
                    f"complete_corpus_publication_shared_between_questions:{publication_id}"
                )


def _validate_labeled_trajectory_family(
    rows: Sequence[LabeledQuestionTrajectory],
    *,
    contexts: Sequence[AdaptivePolicyContext],
    required_split: AdaptiveSplit,
    require_scored: bool,
    score_models: Sequence[AdaptiveTrajectoryScoreModel] = (),
) -> None:
    if not rows:
        raise AdaptiveCalibrationError(f"adaptive_{required_split}_trajectories_empty")
    question_ids = [row.visible.question_id for row in rows]
    if question_ids != sorted(set(question_ids)):
        raise AdaptiveCalibrationError(f"adaptive_{required_split}_questions_must_be_sorted_unique")
    if any(row.reference.question_id != row.visible.question_id for row in rows):
        raise AdaptiveCalibrationError("adaptive_reference_question_identity_mismatch")
    _validate_policy_visible_trajectory_family(
        [row.visible for row in rows],
        contexts=contexts,
        required_split=required_split,
        require_scored=require_scored,
        score_models=score_models,
    )


def _validate_cross_split_independence(
    development: Sequence[LabeledQuestionTrajectory],
    calibration: Sequence[LabeledQuestionTrajectory | PolicyVisibleQuestionTrajectory],
) -> None:
    calibration_visible = [
        row.visible if isinstance(row, LabeledQuestionTrajectory) else row for row in calibration
    ]
    development_questions = {row.visible.question_id for row in development}
    calibration_questions = {row.question_id for row in calibration_visible}
    overlap = sorted(development_questions & calibration_questions)
    if overlap:
        raise AdaptiveCalibrationError(f"adaptive_question_cross_split_overlap:{overlap}")
    development_publications = {
        publication_id
        for row in development
        for publication_id in row.visible.corpus.publication_ids
    }
    calibration_publications = {
        publication_id
        for row in calibration_visible
        for publication_id in row.corpus.publication_ids
    }
    publication_overlap = sorted(development_publications & calibration_publications)
    if publication_overlap:
        raise AdaptiveCalibrationError(
            f"complete_corpus_publication_cross_split_overlap:{publication_overlap}"
        )
    development_corpus_ids = {row.visible.corpus.corpus_id for row in development}
    calibration_corpus_ids = {row.corpus.corpus_id for row in calibration_visible}
    corpus_id_overlap = sorted(development_corpus_ids & calibration_corpus_ids)
    if corpus_id_overlap:
        raise AdaptiveCalibrationError(
            f"complete_corpus_id_cross_split_overlap:{corpus_id_overlap}"
        )
    development_manifests = {
        row.visible.corpus.source_manifest_sha256
        for row in development
        if row.visible.corpus.source_manifest_sha256 is not None
    }
    calibration_manifests = {
        row.corpus.source_manifest_sha256
        for row in calibration_visible
        if row.corpus.source_manifest_sha256 is not None
    }
    manifest_overlap = sorted(development_manifests & calibration_manifests)
    if manifest_overlap:
        raise AdaptiveCalibrationError(
            f"complete_corpus_source_manifest_cross_split_overlap:{manifest_overlap}"
        )


def _fit_arm_model(
    rows: Sequence[LabeledQuestionTrajectory],
    *,
    context: AdaptivePolicyContext,
    seed: int,
) -> AdaptiveTrajectoryScoreModel:
    matrix: list[list[float]] = []
    labels: list[int] = []
    weights: list[float] = []
    for row in rows:
        arm = next(arm for arm in row.visible.arms if arm.policy_arm_id == context.policy_arm_id)
        question_weight = 1.0 / len(arm.states)
        for state in arm.states:
            matrix.append([state.score_features[name] for name in context.score_feature_names])
            labels.append(int(state.claim_decision != row.reference.verdict))
            weights.append(question_weight)
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(labels, dtype=int)
    sample_weight = np.asarray(weights, dtype=float)
    means = np.average(x, axis=0, weights=sample_weight)
    variance = np.average((x - means) ** 2, axis=0, weights=sample_weight)
    scales = np.sqrt(variance)
    scales[scales == 0] = 1.0
    standardized = (x - means) / scales
    if len(set(labels)) == 2:
        estimator = LogisticRegression(random_state=seed, max_iter=2000)
        estimator.fit(standardized, y, sample_weight=sample_weight)
        coefficients = estimator.coef_[0].tolist()
        intercept = float(estimator.intercept_[0])
        fitting_method = "question_weighted_logistic"
    else:
        # Development-only Laplace smoothing keeps an all-correct/all-error pilot
        # serializable without pretending it provides a useful discriminative fit.
        errors = math.fsum(weight * label for weight, label in zip(weights, labels, strict=True))
        total = math.fsum(weights)
        probability = (errors + 0.5) / (total + 1.0)
        coefficients = [0.0] * len(context.score_feature_names)
        intercept = math.log(probability / (1.0 - probability))
        fitting_method = "smoothed_constant"
    payload: dict[str, Any] = {
        "model_version": "adaptive-trajectory-logistic-v1",
        "policy_arm_id": context.policy_arm_id,
        "policy_context_sha256": context.policy_context_sha256,
        "feature_names": context.score_feature_names,
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coefficients": coefficients,
        "intercept": intercept,
        "fitting_method": fitting_method,
        "development_question_ids": sorted(row.visible.question_id for row in rows),
        "seed": seed,
    }
    return AdaptiveTrajectoryScoreModel.model_validate(
        {**payload, "score_model_sha256": hash_canonical(payload)}
    )


def _normalize_labeled_trajectory(
    row: LabeledQuestionTrajectory,
) -> LabeledQuestionTrajectory:
    if not isinstance(row, LabeledQuestionTrajectory):
        raise AdaptiveCalibrationError("adaptive_labeled_trajectory_contract_invalid")
    try:
        return LabeledQuestionTrajectory.model_validate(row.model_dump(mode="json"))
    except ValueError as exc:
        raise AdaptiveCalibrationError("adaptive_labeled_trajectory_integrity_changed") from exc


def _normalize_policy_visible_trajectory(
    row: PolicyVisibleQuestionTrajectory,
) -> PolicyVisibleQuestionTrajectory:
    if not isinstance(row, PolicyVisibleQuestionTrajectory):
        raise AdaptiveCalibrationError("adaptive_policy_visible_trajectory_contract_invalid")
    try:
        return PolicyVisibleQuestionTrajectory.model_validate(row.model_dump(mode="json"))
    except ValueError as exc:
        raise AdaptiveCalibrationError(
            "adaptive_policy_visible_trajectory_integrity_changed"
        ) from exc


def _normalize_policy_context(context: AdaptivePolicyContext) -> AdaptivePolicyContext:
    if not isinstance(context, AdaptivePolicyContext):
        raise AdaptiveCalibrationError("adaptive_policy_context_contract_invalid")
    try:
        return AdaptivePolicyContext.model_validate(context.model_dump(mode="json"))
    except ValueError as exc:
        raise AdaptiveCalibrationError("adaptive_policy_context_integrity_changed") from exc


def _normalize_score_model(
    model: AdaptiveTrajectoryScoreModel,
) -> AdaptiveTrajectoryScoreModel:
    if not isinstance(model, AdaptiveTrajectoryScoreModel):
        raise AdaptiveCalibrationError("adaptive_score_model_contract_invalid")
    try:
        return AdaptiveTrajectoryScoreModel.model_validate(model.model_dump(mode="json"))
    except ValueError as exc:
        raise AdaptiveCalibrationError("adaptive_score_model_integrity_changed") from exc


def score_labeled_question_trajectory(
    row: LabeledQuestionTrajectory,
    *,
    score_models: Sequence[AdaptiveTrajectoryScoreModel],
) -> LabeledQuestionTrajectory:
    """Return a score-bound copy; reference remains outside every visible state."""

    row = _normalize_labeled_trajectory(row)
    normalized_models = [_normalize_score_model(model) for model in score_models]
    if len({model.policy_arm_id for model in normalized_models}) != len(normalized_models):
        raise AdaptiveCalibrationError("adaptive_score_model_arm_duplicate")
    model_by_arm = {model.policy_arm_id: model for model in normalized_models}
    scored_arms: list[AdaptivePolicyArmTrajectory] = []
    for arm in row.visible.arms:
        try:
            model = model_by_arm[arm.policy_arm_id]
        except KeyError as exc:
            raise AdaptiveCalibrationError("adaptive_score_model_arm_missing") from exc
        scored_states = [
            freeze_adaptive_preselection_state(
                prefix_index=state.prefix_index,
                audit_prefix_item_ids=state.audit_prefix_item_ids,
                audit_prefix_cost_minutes=state.audit_prefix_cost_minutes,
                scheduler_state_sha256=state.scheduler_state_sha256,
                evidence_graph_sha256=state.evidence_graph_sha256,
                synthesis_sha256=state.synthesis_sha256,
                non_calibration_assessment_sha256=state.non_calibration_assessment_sha256,
                non_calibration_gates_passed=state.non_calibration_gates_passed,
                non_calibration_blocking_reasons=state.non_calibration_blocking_reasons,
                claim_decision=state.claim_decision,
                score_features=state.score_features,
                scalar_risk_score=model.score_features(state.score_features),
                score_model_sha256=model.score_model_sha256,
            )
            for state in arm.states
        ]
        scored_arms.append(
            freeze_adaptive_policy_arm_trajectory(
                policy_arm_id=arm.policy_arm_id,
                policy_context_sha256=arm.policy_context_sha256,
                states=scored_states,
                terminal_reason=arm.terminal_reason,
                terminal_candidates=arm.terminal_proof.candidates,
                terminal_source_candidate_input_sha256=(
                    arm.terminal_proof.source_candidate_input_sha256
                ),
                terminal_remaining_budget_minutes=(arm.terminal_proof.remaining_budget_minutes),
                source_certificate_sha256s=arm.source_certificate_sha256s,
                terminal_decision_sha256=arm.terminal_decision_sha256,
                terminal_condition_projection=(
                    arm.terminal_proof.condition_projection
                    if isinstance(
                        arm.terminal_proof,
                        ConditionGateInvocationProofV2,
                    )
                    else None
                ),
                terminal_condition_invocation_proof=(
                    arm.terminal_proof
                    if isinstance(
                        arm.terminal_proof,
                        ConditionGateInvocationProofV2,
                    )
                    else None
                ),
            )
        )
    visible = freeze_policy_visible_question_trajectory(
        question_id=row.visible.question_id,
        split=row.visible.split,
        population_id=row.visible.population_id,
        domain=row.visible.domain,
        corpus=row.visible.corpus,
        arms=scored_arms,
    )
    return join_labeled_question_trajectory(visible=visible, reference=row.reference)


def _unscore_labeled_question_trajectory(
    row: LabeledQuestionTrajectory,
) -> LabeledQuestionTrajectory:
    """Reconstruct the exact canonical pre-model trajectory for input-hash checks."""

    unscored_arms = []
    for arm in row.visible.arms:
        states = [
            freeze_adaptive_preselection_state(
                prefix_index=state.prefix_index,
                audit_prefix_item_ids=state.audit_prefix_item_ids,
                audit_prefix_cost_minutes=state.audit_prefix_cost_minutes,
                scheduler_state_sha256=state.scheduler_state_sha256,
                evidence_graph_sha256=state.evidence_graph_sha256,
                synthesis_sha256=state.synthesis_sha256,
                non_calibration_assessment_sha256=(state.non_calibration_assessment_sha256),
                non_calibration_gates_passed=state.non_calibration_gates_passed,
                non_calibration_blocking_reasons=(state.non_calibration_blocking_reasons),
                claim_decision=state.claim_decision,
                score_features=state.score_features,
            )
            for state in arm.states
        ]
        unscored_arms.append(
            freeze_adaptive_policy_arm_trajectory(
                policy_arm_id=arm.policy_arm_id,
                policy_context_sha256=arm.policy_context_sha256,
                states=states,
                terminal_reason=arm.terminal_reason,
                terminal_candidates=arm.terminal_proof.candidates,
                terminal_source_candidate_input_sha256=(
                    arm.terminal_proof.source_candidate_input_sha256
                ),
                terminal_remaining_budget_minutes=(arm.terminal_proof.remaining_budget_minutes),
                source_certificate_sha256s=arm.source_certificate_sha256s,
                terminal_decision_sha256=arm.terminal_decision_sha256,
                terminal_condition_projection=(
                    arm.terminal_proof.condition_projection
                    if isinstance(
                        arm.terminal_proof,
                        ConditionGateInvocationProofV2,
                    )
                    else None
                ),
                terminal_condition_invocation_proof=(
                    arm.terminal_proof
                    if isinstance(
                        arm.terminal_proof,
                        ConditionGateInvocationProofV2,
                    )
                    else None
                ),
            )
        )
    visible = freeze_policy_visible_question_trajectory(
        question_id=row.visible.question_id,
        split=row.visible.split,
        population_id=row.visible.population_id,
        domain=row.visible.domain,
        corpus=row.visible.corpus,
        arms=unscored_arms,
    )
    return join_labeled_question_trajectory(visible=visible, reference=row.reference)


def fit_adaptive_development(
    trajectories: Sequence[LabeledQuestionTrajectory],
    *,
    policy_contexts: Sequence[AdaptivePolicyContext],
    calibration_visible_trajectories: Sequence[PolicyVisibleQuestionTrajectory],
    alpha: float,
    delta: float,
    candidate_thresholds: Mapping[str, Sequence[float]] | None = None,
    seed: int = 20260827,
) -> AdaptiveDevelopmentFreeze:
    """Fit models and preregister thresholds/risk targets before calibration opens."""

    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise AdaptiveCalibrationError("adaptive_alpha_delta_must_be_between_zero_and_one")
    rows = sorted(
        (_normalize_labeled_trajectory(row) for row in trajectories),
        key=lambda row: row.visible.question_id,
    )
    contexts = sorted(
        (_normalize_policy_context(context) for context in policy_contexts),
        key=lambda context: context.policy_arm_id,
    )
    calibration_visible = sorted(
        (_normalize_policy_visible_trajectory(row) for row in calibration_visible_trajectories),
        key=lambda row: row.question_id,
    )
    if not contexts:
        raise AdaptiveCalibrationError("adaptive_policy_contexts_empty")
    _validate_labeled_trajectory_family(
        rows,
        contexts=contexts,
        required_split="development",
        require_scored=False,
    )
    _validate_policy_visible_trajectory_family(
        calibration_visible,
        contexts=contexts,
        required_split="calibration",
        require_scored=False,
    )
    _validate_cross_split_independence(rows, calibration_visible)
    models = [_fit_arm_model(rows, context=context, seed=seed) for context in contexts]
    scored = [score_labeled_question_trajectory(row, score_models=models) for row in rows]
    provided = None if candidate_thresholds is None else dict(candidate_thresholds)
    if provided is not None and set(provided) != {context.policy_arm_id for context in contexts}:
        raise AdaptiveCalibrationError("adaptive_threshold_arm_family_mismatch")
    candidates: list[AdaptiveThresholdCandidate] = []
    for context, model in zip(contexts, models, strict=True):
        if provided is None:
            thresholds = sorted(
                {
                    float(state.scalar_risk_score)
                    for row in scored
                    for arm in row.visible.arms
                    if arm.policy_arm_id == context.policy_arm_id
                    for state in arm.states
                    if state.scalar_risk_score is not None
                }
            )
        else:
            thresholds = sorted(set(float(value) for value in provided[context.policy_arm_id]))
        if not thresholds or any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in thresholds
        ):
            raise AdaptiveCalibrationError("adaptive_candidate_thresholds_invalid")
        candidates.extend(
            _freeze_threshold_candidate(
                policy_arm_id=context.policy_arm_id,
                policy_context_sha256=context.policy_context_sha256,
                score_model_sha256=model.score_model_sha256,
                threshold=threshold,
            )
            for threshold in thresholds
        )
    candidates.sort(key=lambda row: (row.policy_arm_id, row.threshold, row.candidate_sha256))
    visible_hash = hash_canonical([row.visible for row in scored])
    family_payload: dict[str, Any] = {
        "family_version": "adaptive-threshold-family-v1",
        "definition_source": "development_only",
        "development_question_ids": sorted(row.visible.question_id for row in rows),
        "development_visible_input_sha256": visible_hash,
        "candidates": candidates,
    }
    family = AdaptiveThresholdFamily.model_validate(
        {**family_payload, "family_sha256": hash_canonical(family_payload)}
    )
    calibration_plan = _freeze_adaptive_calibration_plan(
        alpha=alpha,
        delta=delta,
        threshold_family_sha256=family.family_sha256,
    )
    calibration_roster = _freeze_adaptive_calibration_roster(calibration_visible)
    identity = _split_identity(scored, split="development")
    payload: dict[str, Any] = {
        "freeze_version": "adaptive-development-freeze-v1",
        "freeze_state": "calibration_labels_unopened",
        "population_id": contexts[0].population_id,
        "policy_contexts": contexts,
        "score_models": models,
        "threshold_family": family,
        "calibration_plan": calibration_plan,
        "calibration_roster": calibration_roster,
        "development": identity,
        "scored_development_trajectories": scored,
        "development_input_sha256": hash_canonical(rows),
    }
    return AdaptiveDevelopmentFreeze.model_validate(
        {**payload, "development_freeze_sha256": hash_canonical(payload)}
    )


def _replay_candidate(
    trajectories: Sequence[LabeledQuestionTrajectory],
    candidate: AdaptiveThresholdCandidate,
) -> list[AdaptiveQuestionOutcome]:
    outcomes: list[AdaptiveQuestionOutcome] = []
    for row in sorted(trajectories, key=lambda item: item.visible.question_id):
        arm = next(arm for arm in row.visible.arms if arm.policy_arm_id == candidate.policy_arm_id)
        release_state = next(
            (
                state
                for state in arm.states
                if state.non_calibration_gates_passed
                and state.scalar_risk_score is not None
                and state.scalar_risk_score <= candidate.threshold
            ),
            None,
        )
        accepted = release_state is not None
        outcomes.append(
            AdaptiveQuestionOutcome(
                question_id=row.visible.question_id,
                candidate_sha256=candidate.candidate_sha256,
                accepted=accepted,
                error=(
                    accepted
                    and release_state is not None
                    and release_state.claim_decision != row.reference.verdict
                ),
                first_release_prefix_index=(
                    None if release_state is None else release_state.prefix_index
                ),
                release_state_sha256=(
                    None if release_state is None else release_state.state_sha256
                ),
                scalar_risk_score=(
                    None if release_state is None else release_state.scalar_risk_score
                ),
                reference_sha256=row.reference.reference_sha256,
            )
        )
    return outcomes


def _select_calibrated_candidate(
    passing: Sequence[AdaptiveCandidateCalibration],
) -> AdaptiveCandidateCalibration | None:
    if not passing:
        return None
    return max(
        passing,
        key=lambda row: (
            row.accepted,
            -float(row.simultaneous_upper_risk or 1.0),
            row.candidate.threshold,
            row.candidate.policy_arm_id,
        ),
    )


def calibrate_adaptive_first_release(
    development_freeze: AdaptiveDevelopmentFreeze,
    calibration_trajectories: Sequence[LabeledQuestionTrajectory],
) -> AdaptiveCalibrationBundle:
    """Calibrate one outcome per question across all frozen arms/thresholds."""
    try:
        development = AdaptiveDevelopmentFreeze.model_validate(
            development_freeze.model_dump(mode="json")
        )
    except ValueError as exc:
        raise AdaptiveCalibrationError("adaptive_development_freeze_integrity_changed") from exc
    raw = sorted(
        (_normalize_labeled_trajectory(row) for row in calibration_trajectories),
        key=lambda row: row.visible.question_id,
    )
    if any(
        arm.terminal_reason == "full_nonconfirmation_release_gates_passed"
        for trajectory in (
            *development.scored_development_trajectories,
            *raw,
        )
        for arm in trajectory.visible.arms
    ):
        raise AdaptiveCalibrationError("adaptive_v1_forbids_confirmation_aware_early_stop")
    if [row.visible for row in raw] != development.calibration_roster.visible_trajectories:
        raise AdaptiveCalibrationError("adaptive_calibration_visible_roster_mismatch")
    _validate_labeled_trajectory_family(
        raw,
        contexts=development.policy_contexts,
        required_split="calibration",
        require_scored=False,
    )
    _validate_cross_split_independence(development.scored_development_trajectories, raw)
    label_sources = {row.reference.label_source for row in raw}
    if len(label_sources) != 1:
        raise AdaptiveCalibrationError("adaptive_calibration_label_source_changed")
    adjudication_protocols = {row.reference.adjudication_protocol_sha256 for row in raw}
    if len(adjudication_protocols) != 1:
        raise AdaptiveCalibrationError("adaptive_calibration_adjudication_protocol_changed")
    scored = [
        score_labeled_question_trajectory(row, score_models=development.score_models) for row in raw
    ]
    candidate_family = development.threshold_family.candidates
    plan = development.calibration_plan
    simultaneous_delta = plan.delta / len(candidate_family)
    calibrated: list[AdaptiveCandidateCalibration] = []
    for candidate in candidate_family:
        outcomes = _replay_candidate(scored, candidate)
        accepted = sum(row.accepted for row in outcomes)
        errors = sum(row.error for row in outcomes)
        upper = (
            clopper_pearson_upper(errors, accepted, delta=simultaneous_delta) if accepted else None
        )
        calibrated.append(
            AdaptiveCandidateCalibration(
                candidate=candidate,
                total_questions=len(scored),
                outcomes=outcomes,
                accepted=accepted,
                errors=errors,
                empirical_risk=errors / accepted if accepted else None,
                simultaneous_upper_risk=upper,
                passed=upper is not None and upper <= plan.alpha,
            )
        )
    selected = _select_calibrated_candidate([row for row in calibrated if row.passed])
    identity = _split_identity(scored, split="calibration")
    payload: dict[str, Any] = {
        "bundle_version": "adaptive-question-trajectory-freeze-v1",
        "freeze_state": "test_labels_unopened",
        "guarantee_scope": (
            "exact decision-mismatch risk against the frozen adjudication protocol under "
            "exchangeable independent complete-question trajectories; not scientific truth "
            "or domain-shift robustness"
        ),
        "population_id": development.population_id,
        "label_source": next(iter(label_sources)),
        "adjudication_protocol_sha256": next(iter(adjudication_protocols)),
        "alpha": plan.alpha,
        "delta": plan.delta,
        "correction": plan.multiplicity_correction,
        "candidate_selection_rule": plan.candidate_selection_rule,
        "development_freeze": development,
        "development_freeze_sha256": development.development_freeze_sha256,
        "calibration": identity,
        "scored_calibration_trajectories": scored,
        "calibration_input_sha256": hash_canonical(raw),
        "candidates": calibrated,
        "selected_candidate_sha256": (
            None if selected is None else selected.candidate.candidate_sha256
        ),
        "status": "abstain_all" if selected is None else "calibrated",
    }
    return AdaptiveCalibrationBundle.model_validate(
        {**payload, "bundle_sha256": hash_canonical(payload)}
    )


def validate_adaptive_calibration_bundle_integrity(
    bundle: AdaptiveCalibrationBundle,
) -> AdaptiveCalibrationBundle:
    if not isinstance(bundle, AdaptiveCalibrationBundle):
        raise AdaptiveCalibrationError("adaptive_calibration_bundle_contract_invalid")
    try:
        return AdaptiveCalibrationBundle.model_validate(bundle.model_dump(mode="json"))
    except ValueError as exc:
        raise AdaptiveCalibrationError("adaptive_calibration_bundle_integrity_changed") from exc


class ProspectiveAdaptiveReleaseCandidate(ContractModel):
    """Unlabelled observed prefix supplied on every repeated production assessment."""

    candidate_version: Literal["prospective-adaptive-release-candidate-v1"] = (
        "prospective-adaptive-release-candidate-v1"
    )
    question_id: Annotated[str, Field(min_length=1)]
    population_id: Annotated[str, Field(min_length=1)]
    domain: Annotated[str, Field(min_length=1)]
    policy_arm_id: Annotated[str, Field(min_length=1)]
    policy_context_sha256: str
    corpus: CompleteCorpusIdentity
    observed_states: list[AdaptivePreselectionState]
    candidate_sha256: str

    @field_validator("policy_context_sha256", "candidate_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_candidate(self) -> ProspectiveAdaptiveReleaseCandidate:
        if not self.observed_states:
            raise ValueError("prospective_adaptive_observed_states_empty")
        if any(state.scalar_risk_score is not None for state in self.observed_states):
            raise ValueError("prospective_adaptive_candidate_must_be_unscored")
        # Reuse the same prefix invariants without claiming the online prefix is terminal.
        first = self.observed_states[0]
        if (
            first.prefix_index != 0
            or first.audit_prefix_item_ids
            or not math.isclose(first.audit_prefix_cost_minutes, 0.0, abs_tol=_COST_TOLERANCE)
        ):
            raise ValueError("prospective_adaptive_candidate_must_start_at_prefix_zero")
        for previous, current in zip(self.observed_states, self.observed_states[1:], strict=False):
            if current.prefix_index != previous.prefix_index + 1:
                raise ValueError("prospective_adaptive_prefix_indices_not_contiguous")
            if current.audit_prefix_item_ids[:-1] != previous.audit_prefix_item_ids:
                raise ValueError("prospective_adaptive_prefix_not_monotone")
            if current.audit_prefix_cost_minutes + _COST_TOLERANCE < (
                previous.audit_prefix_cost_minutes
            ):
                raise ValueError("prospective_adaptive_cost_not_monotone")
        payload = self.model_dump(mode="json", exclude={"candidate_sha256"})
        _reject_reference_leakage(payload)
        if hash_canonical(payload) != self.candidate_sha256:
            raise ValueError("prospective_adaptive_candidate_hash_mismatch")
        return self


def freeze_prospective_adaptive_candidate(
    *,
    question_id: str,
    population_id: str,
    domain: str,
    policy_arm_id: str,
    policy_context_sha256: str,
    corpus: CompleteCorpusIdentity,
    observed_states: Sequence[AdaptivePreselectionState],
) -> ProspectiveAdaptiveReleaseCandidate:
    try:
        normalized_corpus = CompleteCorpusIdentity.model_validate(corpus.model_dump(mode="json"))
        normalized_states = [
            AdaptivePreselectionState.model_validate(state.model_dump(mode="json"))
            for state in observed_states
        ]
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError("prospective_adaptive_input_integrity_changed") from exc
    payload: dict[str, Any] = {
        "candidate_version": "prospective-adaptive-release-candidate-v1",
        "question_id": question_id,
        "population_id": population_id,
        "domain": domain,
        "policy_arm_id": policy_arm_id,
        "policy_context_sha256": policy_context_sha256,
        "corpus": normalized_corpus,
        "observed_states": normalized_states,
    }
    return ProspectiveAdaptiveReleaseCandidate.model_validate(
        {**payload, "candidate_sha256": hash_canonical(payload)}
    )


class AdaptiveProspectiveAssessment(ContractModel):
    assessment_version: Literal["adaptive-first-release-assessment-v1"] = (
        "adaptive-first-release-assessment-v1"
    )
    question_id: Annotated[str, Field(min_length=1)]
    candidate_sha256: str
    frozen_bundle_sha256: str
    policy_context_sha256: str
    threshold_candidate_sha256: str | None
    scalar_risk_score: Annotated[float, Field(ge=0, le=1)] | None
    threshold: Annotated[float, Field(ge=0, le=1)] | None
    prefix_index: Annotated[int, Field(ge=0)]
    status: Literal["released", "abstained"]
    reason: Literal[
        "first_full_release_under_frozen_trajectory_policy",
        "noncalibration_gate_blocked",
        "risk_above_frozen_threshold",
        "policy_abstain_all",
        "simulation_calibration_not_valid_for_scientific_release",
    ]
    guarantee_scope: Literal[
        "exact decision-mismatch risk against the frozen adjudication protocol under "
        "exchangeable independent complete-question trajectories; not scientific truth "
        "or domain-shift robustness"
    ] = (
        "exact decision-mismatch risk against the frozen adjudication protocol under "
        "exchangeable independent complete-question trajectories; not scientific truth "
        "or domain-shift robustness"
    )
    assessment_sha256: str

    @field_validator(
        "candidate_sha256",
        "frozen_bundle_sha256",
        "policy_context_sha256",
        "assessment_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("threshold_candidate_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return None if value is None else _validate_sha256(value, "threshold_candidate_sha256")

    @model_validator(mode="after")
    def validate_assessment(self) -> AdaptiveProspectiveAssessment:
        threshold_fields = (
            self.threshold_candidate_sha256,
            self.scalar_risk_score,
            self.threshold,
        )
        has_threshold_lineage = all(value is not None for value in threshold_fields)
        has_no_threshold_lineage = all(value is None for value in threshold_fields)
        if not (has_threshold_lineage or has_no_threshold_lineage):
            raise ValueError("adaptive_assessment_threshold_lineage_incomplete")
        if self.reason == "first_full_release_under_frozen_trajectory_policy":
            if self.status != "released" or not has_threshold_lineage:
                raise ValueError("adaptive_release_assessment_inconsistent")
            assert self.scalar_risk_score is not None and self.threshold is not None
            if self.scalar_risk_score > self.threshold:
                raise ValueError("adaptive_release_score_exceeds_threshold")
        else:
            if self.status != "abstained":
                raise ValueError("adaptive_nonrelease_assessment_status_mismatch")
            if self.reason == "policy_abstain_all" and not has_no_threshold_lineage:
                raise ValueError("adaptive_abstain_all_forbids_threshold_lineage")
            if (
                self.reason
                in {
                    "noncalibration_gate_blocked",
                    "risk_above_frozen_threshold",
                }
                and not has_threshold_lineage
            ):
                raise ValueError("adaptive_thresholded_abstention_lineage_missing")
            if self.reason == "risk_above_frozen_threshold":
                assert self.scalar_risk_score is not None and self.threshold is not None
                if self.scalar_risk_score <= self.threshold:
                    raise ValueError("adaptive_risk_abstention_not_above_threshold")
        payload = self.model_dump(mode="json", exclude={"assessment_sha256"})
        if hash_canonical(payload) != self.assessment_sha256:
            raise ValueError("adaptive_prospective_assessment_hash_mismatch")
        return self


def assess_adaptive_release_candidate(
    candidate: ProspectiveAdaptiveReleaseCandidate,
    bundle: AdaptiveCalibrationBundle,
) -> AdaptiveProspectiveAssessment:
    """Apply the frozen first-release policy to a complete observed prefix.

    Repeated calls are valid because every call includes and replays the whole prefix.
    Continuing after an earlier qualifying prefix is rejected rather than silently
    turning optional stopping into a new, uncalibrated policy.
    """

    bundle = validate_adaptive_calibration_bundle_integrity(bundle)
    if not isinstance(candidate, ProspectiveAdaptiveReleaseCandidate):
        raise AdaptiveCalibrationError("prospective_adaptive_candidate_contract_invalid")
    try:
        candidate = ProspectiveAdaptiveReleaseCandidate.model_validate(
            candidate.model_dump(mode="json")
        )
    except ValueError as exc:
        raise AdaptiveCalibrationError("prospective_adaptive_candidate_integrity_changed") from exc
    if candidate.population_id != bundle.population_id:
        raise AdaptiveCalibrationError("prospective_adaptive_population_mismatch")
    frozen_questions = set(bundle.development_freeze.development.question_ids) | set(
        bundle.calibration.question_ids
    )
    if candidate.question_id in frozen_questions:
        raise AdaptiveCalibrationError("prospective_adaptive_question_overlap")
    frozen_publications = set(bundle.development_freeze.development.complete_publication_ids) | set(
        bundle.calibration.complete_publication_ids
    )
    overlap = sorted(frozen_publications & set(candidate.corpus.publication_ids))
    if overlap:
        raise AdaptiveCalibrationError(f"prospective_complete_corpus_publication_overlap:{overlap}")
    frozen_corpus_ids = {
        row.visible.corpus.corpus_id
        for row in (
            *bundle.development_freeze.scored_development_trajectories,
            *bundle.scored_calibration_trajectories,
        )
    }
    if candidate.corpus.corpus_id in frozen_corpus_ids:
        raise AdaptiveCalibrationError("prospective_complete_corpus_id_overlap")
    if candidate.corpus.source_manifest_sha256 is not None:
        frozen_source_manifests = {
            row.visible.corpus.source_manifest_sha256
            for row in (
                *bundle.development_freeze.scored_development_trajectories,
                *bundle.scored_calibration_trajectories,
            )
            if row.visible.corpus.source_manifest_sha256 is not None
        }
        if candidate.corpus.source_manifest_sha256 in frozen_source_manifests:
            raise AdaptiveCalibrationError("prospective_complete_corpus_source_manifest_overlap")
    if candidate.domain not in bundle.calibration.domains:
        raise AdaptiveCalibrationError("prospective_adaptive_domain_shift")
    context = next(
        (
            row
            for row in bundle.development_freeze.policy_contexts
            if row.policy_arm_id == candidate.policy_arm_id
        ),
        None,
    )
    if context is None or context.policy_context_sha256 != candidate.policy_context_sha256:
        raise AdaptiveCalibrationError("prospective_adaptive_policy_context_mismatch")
    expected_cutoff = context.corpus_protocol_context.get("corpus_cutoff")
    if isinstance(expected_cutoff, str) and expected_cutoff != candidate.corpus.corpus_cutoff:
        raise AdaptiveCalibrationError("prospective_adaptive_corpus_cutoff_context_mismatch")
    if any(
        list(state.score_features) != context.score_feature_names
        for state in candidate.observed_states
    ):
        raise AdaptiveCalibrationError("prospective_adaptive_feature_schema_mismatch")
    if (
        candidate.observed_states[-1].audit_prefix_cost_minutes
        > context.budget_minutes + _COST_TOLERANCE
    ):
        raise AdaptiveCalibrationError("prospective_adaptive_budget_mismatch")
    selected = bundle.selected
    prefix_index = candidate.observed_states[-1].prefix_index
    if selected is None:
        reason = (
            "simulation_calibration_not_valid_for_scientific_release"
            if bundle.label_source == "simulation"
            else "policy_abstain_all"
        )
        payload: dict[str, Any] = {
            "assessment_version": "adaptive-first-release-assessment-v1",
            "question_id": candidate.question_id,
            "candidate_sha256": candidate.candidate_sha256,
            "frozen_bundle_sha256": bundle.bundle_sha256,
            "policy_context_sha256": candidate.policy_context_sha256,
            "threshold_candidate_sha256": None,
            "scalar_risk_score": None,
            "threshold": None,
            "prefix_index": prefix_index,
            "status": "abstained",
            "reason": reason,
            "guarantee_scope": bundle.guarantee_scope,
        }
        return AdaptiveProspectiveAssessment.model_validate(
            {**payload, "assessment_sha256": hash_canonical(payload)}
        )
    threshold_candidate = selected.candidate
    if (
        candidate.policy_arm_id != threshold_candidate.policy_arm_id
        or candidate.policy_context_sha256 != threshold_candidate.policy_context_sha256
    ):
        raise AdaptiveCalibrationError("prospective_adaptive_policy_context_mismatch")
    model = next(
        row
        for row in bundle.development_freeze.score_models
        if row.policy_arm_id == candidate.policy_arm_id
    )
    scores = [model.score_features(state.score_features) for state in candidate.observed_states]
    qualifying_indices = [
        index
        for index, (state, score) in enumerate(zip(candidate.observed_states, scores, strict=True))
        if state.non_calibration_gates_passed and score <= threshold_candidate.threshold
    ]
    if qualifying_indices and qualifying_indices[0] != len(candidate.observed_states) - 1:
        raise AdaptiveCalibrationError("prospective_candidate_continued_after_first_release")
    current = candidate.observed_states[-1]
    score = scores[-1]
    if bundle.label_source == "simulation":
        status = "abstained"
        reason = "simulation_calibration_not_valid_for_scientific_release"
    elif qualifying_indices:
        status = "released"
        reason = "first_full_release_under_frozen_trajectory_policy"
    elif not current.non_calibration_gates_passed:
        status = "abstained"
        reason = "noncalibration_gate_blocked"
    else:
        status = "abstained"
        reason = "risk_above_frozen_threshold"
    payload = {
        "assessment_version": "adaptive-first-release-assessment-v1",
        "question_id": candidate.question_id,
        "candidate_sha256": candidate.candidate_sha256,
        "frozen_bundle_sha256": bundle.bundle_sha256,
        "policy_context_sha256": candidate.policy_context_sha256,
        "threshold_candidate_sha256": threshold_candidate.candidate_sha256,
        "scalar_risk_score": score,
        "threshold": threshold_candidate.threshold,
        "prefix_index": prefix_index,
        "status": status,
        "reason": reason,
        "guarantee_scope": bundle.guarantee_scope,
    }
    return AdaptiveProspectiveAssessment.model_validate(
        {**payload, "assessment_sha256": hash_canonical(payload)}
    )


def noncalibration_assessment_sha256(
    *,
    question_id: str,
    target: Any,
    pipeline_sha256: str,
    evidence_graph_sha256: str,
    synthesis_sha256: str,
    config_sha256: str,
    complete_matching_paper_ids: Sequence[str],
    evidence: Any,
    audit: Any,
    risk_features: Mapping[str, float],
    condition_calibration_projection: ConditionCalibrationProjectionV1 | None = None,
) -> str:
    """Hash the exact release inputs before the calibration gate, without labels."""

    projection = (
        None
        if condition_calibration_projection is None
        else ConditionCalibrationProjectionV1.model_validate(
            condition_calibration_projection.model_dump(mode="json")
        )
    )
    return hash_canonical(
        {
            "noncalibration_assessment_version": (
                "claim-release-noncalibration-v2"
                if projection is not None
                else "claim-release-noncalibration-v1"
            ),
            "question_id": question_id,
            "target": target,
            "pipeline_sha256": pipeline_sha256,
            "evidence_graph_sha256": evidence_graph_sha256,
            "synthesis_sha256": synthesis_sha256,
            "config_sha256": config_sha256,
            "complete_matching_paper_ids": sorted(set(complete_matching_paper_ids)),
            "evidence": evidence if projection is None else None,
            "condition_calibration_projection": projection,
            "audit": audit,
            "risk_features": dict(risk_features),
        }
    )


def _normalize_verification_certificate_v5(certificate: Any) -> Any:
    """Reparse a v5 certificate so mutable nested payloads cannot bypass hashes."""

    # Delayed import avoids the module cycle: certificate imports this module's
    # contract classes, while projection is only invoked after both modules load.
    from literature_multiverse.certificate import VerificationCertificate

    if not isinstance(certificate, VerificationCertificate):
        raise AdaptiveCalibrationError("adaptive_projection_certificate_contract_invalid")
    try:
        parsed = VerificationCertificate.model_validate(certificate.model_dump(mode="json"))
    except ValueError as exc:
        raise AdaptiveCalibrationError("adaptive_projection_certificate_integrity_changed") from exc
    if parsed.certificate_version != "literature-multiverse-verification-v5":
        raise AdaptiveCalibrationError("adaptive_projection_requires_certificate_v5")
    return parsed


def complete_corpus_identity_from_certificate_v5(
    certificate: Any,
) -> CompleteCorpusIdentity:
    """Project exact complete membership from a validated certificate-v5 object.

    Callers must pass a validated ``VerificationCertificate`` rather than an
    unparsed dictionary. The object is reparsed to defeat nested in-memory mutation.
    """

    certificate = _normalize_verification_certificate_v5(certificate)
    corpus = getattr(certificate, "corpus", None)
    source_graph = getattr(certificate, "source_evidence_graph", None)
    if not isinstance(corpus, Mapping) or source_graph is None:
        raise AdaptiveCalibrationError("adaptive_projection_certificate_contract_invalid")
    metadata = corpus.get("metadata")
    if not isinstance(metadata, Mapping):
        raise AdaptiveCalibrationError("adaptive_projection_corpus_metadata_missing")
    manifest = metadata.get("native_source_manifest")
    source_manifest_sha256 = metadata.get("source_manifest_sha256")
    if manifest is not None:
        if not isinstance(manifest, Mapping) or not isinstance(source_manifest_sha256, str):
            raise AdaptiveCalibrationError("adaptive_projection_source_manifest_invalid")
        records = manifest.get("records")
        if not isinstance(records, list):
            raise AdaptiveCalibrationError("adaptive_projection_source_manifest_records_missing")
        try:
            publication_ids = sorted(
                str(record["publication"]["publication_id"]) for record in records
            )
        except (KeyError, TypeError) as exc:
            raise AdaptiveCalibrationError(
                "adaptive_projection_source_manifest_publication_invalid"
            ) from exc
        graph_publication_ids = sorted(
            publication.publication_id for publication in source_graph.publications
        )
        if publication_ids != graph_publication_ids:
            raise AdaptiveCalibrationError(
                "adaptive_projection_complete_publication_membership_mismatch"
            )
    else:
        publication_ids = sorted(
            publication.publication_id for publication in source_graph.publications
        )
        source_manifest_sha256 = None
    cutoff = corpus.get("declared_corpus_cutoff")
    corpus_id = corpus.get("corpus_id")
    source_sha256 = corpus.get("source_sha256")
    if not all(isinstance(value, str) and value for value in (cutoff, corpus_id, source_sha256)):
        raise AdaptiveCalibrationError("adaptive_projection_corpus_identity_incomplete")
    projected = freeze_complete_corpus_identity(
        corpus_id=corpus_id,
        corpus_source_sha256=source_sha256,
        corpus_cutoff=cutoff,
        publication_ids=publication_ids,
        source_manifest_sha256=source_manifest_sha256,
    )
    if certificate.complete_corpus_identity != projected:
        raise AdaptiveCalibrationError(
            "adaptive_projection_embedded_complete_corpus_identity_mismatch"
        )
    return projected


def preselection_state_from_certificate_v5(
    certificate: Any,
) -> AdaptivePreselectionState:
    """Project one label-free preselection state from ProductionStopDecision."""

    certificate = _normalize_verification_certificate_v5(certificate)
    decision = getattr(certificate, "production_stop_decision", None)
    if decision is None or decision.stopping_rule != PRODUCTION_STOPPING_RULE:
        raise AdaptiveCalibrationError("adaptive_projection_production_stop_missing")
    return freeze_preselection_state_from_production_components(
        sequential_state=decision.evaluated_state,
        release_assessment=decision.release_assessment,
        blocking_adapter_reasons=decision.blocking_adapter_reasons,
    )


def freeze_preselection_state_from_production_components(
    *,
    sequential_state: Any,
    release_assessment: Any,
    blocking_adapter_reasons: Sequence[str],
) -> AdaptivePreselectionState:
    """Project one adaptive state from already-validated production components.

    This pure projection is shared by the online verifier, certificate validation,
    and certificate-sequence replay.  It intentionally consumes no reference label
    and removes only the calibration gate from the release reason ledger.
    """

    state = sequential_state
    if state is None:
        raise AdaptiveCalibrationError("adaptive_projection_requires_sequential_state")
    if state.session.active_action is not None:
        raise AdaptiveCalibrationError("adaptive_projection_state_is_not_preselection")
    assessment = release_assessment
    projection = getattr(assessment, "condition_calibration_projection", None)
    terminal_gate_deferred = getattr(assessment, "terminal_gate_deferred", False)
    confirmation_gate = getattr(assessment, "condition_confirmation_gate", None)
    if projection is not None:
        try:
            projection = ConditionCalibrationProjectionV1.model_validate(
                projection.model_dump(mode="json")
            )
        except (AttributeError, ValueError) as exc:
            raise AdaptiveCalibrationError(
                "adaptive_projection_condition_projection_invalid"
            ) from exc
        if not terminal_gate_deferred:
            raise AdaptiveCalibrationError(
                "adaptive_projection_condition_terminal_gate_not_deferred"
            )
        if confirmation_gate is None:
            raise AdaptiveCalibrationError(
                "adaptive_projection_condition_confirmation_gate_missing"
            )
        try:
            confirmation_gate = ConditionConfirmationGateAssessmentV1.model_validate(
                confirmation_gate.model_dump(mode="json")
            )
        except (AttributeError, ValueError) as exc:
            raise AdaptiveCalibrationError(
                "adaptive_projection_condition_confirmation_gate_invalid"
            ) from exc
        if (
            not confirmation_gate.required
            or confirmation_gate.status != "missing"
            or confirmation_gate.provisional_claim_decision != projection.provisional_claim_decision
            or confirmation_gate.condition_projection_sha256 != projection.projection_sha256
            or confirmation_gate.target_sha256 != projection.condition_target_sha256
            or confirmation_gate.plan_sha256 != projection.plan_sha256
            or confirmation_gate.config_sha256 != projection.confirmation_config_sha256
        ):
            raise AdaptiveCalibrationError(
                "adaptive_projection_condition_deferred_gate_contract_mismatch"
            )
    elif terminal_gate_deferred:
        raise AdaptiveCalibrationError(
            "adaptive_projection_terminal_gate_deferred_without_projection"
        )
    typed_terminal_blockers = {
        "condition_confirmation_required",
        "condition_dependent_confirmation_aware_calibration_required",
    }
    claim_reasons = []
    for reason in assessment.reasons:
        if reason.startswith("calibration:"):
            continue
        if terminal_gate_deferred and reason in typed_terminal_blockers:
            continue
        claim_reasons.append(reason)
    blockers = sorted(set(claim_reasons) | set(blocking_adapter_reasons))
    evidence = assessment.evidence
    if projection is not None:
        claim_decision = projection.provisional_claim_decision
    else:
        claim_decision_value = getattr(evidence, "classification", None)
        if claim_decision_value is None:
            claim_decision_value = getattr(evidence, "state", None)
        claim_decision = getattr(claim_decision_value, "value", claim_decision_value)
    if not isinstance(claim_decision, str) or not claim_decision:
        raise AdaptiveCalibrationError("adaptive_projection_claim_decision_missing")
    config_hash = assessment.config_sha256
    assessment_hash = noncalibration_assessment_sha256(
        question_id=assessment.question_id,
        target=assessment.target,
        pipeline_sha256=assessment.pipeline_sha256,
        evidence_graph_sha256=assessment.evidence_graph_sha256,
        synthesis_sha256=assessment.synthesis_sha256,
        config_sha256=config_hash,
        complete_matching_paper_ids=assessment.paper_ids,
        evidence=assessment.evidence,
        audit=assessment.audit,
        risk_features=assessment.risk_features,
        condition_calibration_projection=projection,
    )
    return freeze_adaptive_preselection_state(
        prefix_index=len(state.session.resolved_item_ids),
        audit_prefix_item_ids=state.session.resolved_item_ids,
        audit_prefix_cost_minutes=state.session.historical_realized_cost,
        scheduler_state_sha256=state.state_sha256,
        evidence_graph_sha256=assessment.evidence_graph_sha256,
        synthesis_sha256=assessment.synthesis_sha256,
        non_calibration_assessment_sha256=assessment_hash,
        non_calibration_gates_passed=not blockers,
        non_calibration_blocking_reasons=blockers,
        claim_decision=claim_decision,
        score_features=assessment.risk_features,
    )


def policy_visible_trajectory_from_certificate_v5_sequence(
    certificates: Sequence[Any],
    *,
    split: AdaptiveSplit,
    policy_context: AdaptivePolicyContext,
    terminal_reason: TrajectoryTerminalReason,
) -> PolicyVisibleQuestionTrajectory:
    """Build and verify one complete threshold-blind path from v5 verifier runs."""

    if not certificates:
        raise AdaptiveCalibrationError("adaptive_projection_certificate_sequence_empty")
    normalized_certificates = [
        _normalize_verification_certificate_v5(certificate) for certificate in certificates
    ]
    projected = [
        (preselection_state_from_certificate_v5(certificate), certificate)
        for certificate in normalized_certificates
    ]
    projected.sort(key=lambda pair: pair[0].prefix_index)
    states = [pair[0] for pair in projected]
    first_certificate = projected[0][1]
    manifest = first_certificate.claim_manifest
    if not isinstance(manifest, Mapping):
        raise AdaptiveCalibrationError("adaptive_projection_claim_manifest_invalid")
    question_id = manifest.get("question_id")
    population_id = manifest.get("population_id")
    domain = manifest.get("domain")
    if not all(isinstance(value, str) and value for value in (question_id, population_id, domain)):
        raise AdaptiveCalibrationError("adaptive_projection_claim_identity_incomplete")
    if population_id != policy_context.population_id:
        raise AdaptiveCalibrationError("adaptive_projection_population_context_mismatch")
    for state, certificate in projected:
        decision = certificate.production_stop_decision
        assessment = decision.release_assessment
        if assessment.pipeline_sha256 != policy_context.pipeline_sha256:
            raise AdaptiveCalibrationError("adaptive_projection_pipeline_context_mismatch")
        if (
            state.audit_prefix_cost_minutes > policy_context.budget_minutes + _COST_TOLERANCE
            or not math.isclose(
                decision.evaluated_state.session.budget,
                policy_context.budget_minutes,
                rel_tol=0.0,
                abs_tol=_COST_TOLERANCE,
            )
            or decision.evaluated_state.session.cost_unit != policy_context.cost_unit
        ):
            raise AdaptiveCalibrationError("adaptive_projection_budget_context_mismatch")
        embedded_context = certificate.adaptive_policy_context
        if embedded_context is not None and embedded_context != policy_context:
            raise AdaptiveCalibrationError("adaptive_projection_embedded_policy_context_mismatch")
    corpus = complete_corpus_identity_from_certificate_v5(first_certificate)
    for _, certificate in projected[1:]:
        if certificate.claim_manifest != first_certificate.claim_manifest:
            raise AdaptiveCalibrationError("adaptive_projection_manifest_changed_across_prefixes")
        if complete_corpus_identity_from_certificate_v5(certificate) != corpus:
            raise AdaptiveCalibrationError("adaptive_projection_corpus_changed_across_prefixes")
    expected_cutoff = policy_context.corpus_protocol_context.get("corpus_cutoff")
    if isinstance(expected_cutoff, str) and expected_cutoff != corpus.corpus_cutoff:
        raise AdaptiveCalibrationError("adaptive_projection_corpus_cutoff_context_mismatch")

    # Every nonterminal preselection state must deterministically select the exact
    # item resolved at the next prefix, and the next state must extend that selected
    # transition by exactly one correction. This rules out cherry-picked prefixes.
    for (_previous_state, previous_certificate), (
        current_state,
        current_certificate,
    ) in pairwise(projected):
        decision = previous_certificate.production_stop_decision
        result = decision.selection_result
        if decision.outcome != "selected_next_action" or result is None:
            raise AdaptiveCalibrationError(
                "adaptive_projection_nonterminal_prefix_did_not_select_next_action"
            )
        selected_item_id = result.action.item_id
        if current_state.audit_prefix_item_ids[-1] != selected_item_id:
            raise AdaptiveCalibrationError("adaptive_projection_selected_action_prefix_mismatch")
        previous_evaluated = decision.evaluated_state
        current_evaluated = current_certificate.production_stop_decision.evaluated_state
        assert previous_evaluated is not None and current_evaluated is not None
        if tuple(current_evaluated.session.resolved_item_ids) != (
            *tuple(previous_evaluated.session.resolved_item_ids),
            selected_item_id,
        ):
            raise AdaptiveCalibrationError("adaptive_projection_resolution_prefix_mismatch")
        selection_transitions = list(result.state.transitions)
        current_transitions = list(current_evaluated.transitions)
        if (
            len(current_transitions) != len(selection_transitions) + 1
            or current_transitions[:-1] != selection_transitions
            or current_transitions[-1].transition_kind != "correction"
            or current_transitions[-1].action != result.action
        ):
            raise AdaptiveCalibrationError(
                "adaptive_projection_selection_correction_chain_mismatch"
            )

    final_certificate = projected[-1][1]
    final_decision = final_certificate.production_stop_decision
    final_state = final_decision.evaluated_state
    assert final_state is not None
    candidate_ids = {candidate.item_id for candidate in final_state.candidates}
    resolved_ids = set(final_state.session.resolved_item_ids)
    unresolved_ids = candidate_ids - resolved_ids
    selectable_ids = {
        candidate.item_id
        for candidate in final_state.candidates
        if candidate.eligible
        and candidate.item_id not in final_state.session.selected_item_ids
        and candidate.estimated_cost <= final_state.session.remaining_budget + _COST_TOLERANCE
    }
    if terminal_reason == "all_items_resolved":
        terminal_valid = not unresolved_ids and final_decision.outcome in {
            "stopped_released",
            "no_feasible_action",
        }
    elif terminal_reason == "budget_exhausted":
        terminal_valid = (
            bool(unresolved_ids)
            and not selectable_ids
            and final_decision.outcome == "no_feasible_action"
            and math.isclose(
                final_state.session.remaining_budget,
                0.0,
                rel_tol=0.0,
                abs_tol=_COST_TOLERANCE,
            )
        )
    else:
        terminal_valid = (
            bool(unresolved_ids)
            and not selectable_ids
            and final_decision.outcome == "no_feasible_action"
        )
    if not terminal_valid:
        raise AdaptiveCalibrationError("adaptive_projection_terminal_reason_not_proven")
    arm = freeze_adaptive_policy_arm_trajectory(
        policy_arm_id=policy_context.policy_arm_id,
        policy_context_sha256=policy_context.policy_context_sha256,
        states=states,
        terminal_reason=terminal_reason,
        terminal_candidates=[
            AdaptiveTerminalAuditCandidate(
                item_id=candidate.item_id,
                eligible=candidate.eligible,
                estimated_cost_minutes=candidate.estimated_cost,
                source_candidate_sha256=candidate.candidate_sha256,
            )
            for candidate in final_state.candidates
        ],
        terminal_source_candidate_input_sha256=final_state.candidate_input_sha256,
        terminal_remaining_budget_minutes=final_state.session.remaining_budget,
        source_certificate_sha256s=[certificate.certificate_sha256 for _, certificate in projected],
        terminal_decision_sha256=final_decision.decision_sha256,
    )
    return freeze_policy_visible_question_trajectory(
        question_id=question_id,
        split=split,
        population_id=population_id,
        domain=domain,
        corpus=corpus,
        arms=[arm],
    )


class ConfirmationAwareArmTrajectoryV2(ContractModel):
    """Outcome-free, pre-bundle wrapper preserving every v1 prefix state hash."""

    wrapper_version: Literal["confirmation-aware-arm-trajectory-v2"] = (
        "confirmation-aware-arm-trajectory-v2"
    )
    base_arm: AdaptivePolicyArmTrajectory
    terminal_condition_required: bool
    terminal_condition_projection: ConditionCalibrationProjectionV1 | None = None
    terminal_condition_projection_sha256: str | None = None
    condition_gate_invocation_proof: ConditionGateInvocationProofV2 | None = None
    condition_gate_invocation_proof_sha256: str | None = None
    wrapper_sha256: str

    @field_validator(
        "terminal_condition_projection_sha256",
        "condition_gate_invocation_proof_sha256",
        "wrapper_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _validate_sha256(value, info.field_name)

    @field_validator("terminal_condition_required", mode="before")
    @classmethod
    def validate_required(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("confirmation_v2_terminal_required_must_be_boolean")
        return value

    @model_validator(mode="after")
    def validate_wrapper(self) -> ConfirmationAwareArmTrajectoryV2:
        terminal_state = self.base_arm.states[-1]
        expected_projection = terminal_state.claim_decision == "condition_dependent"
        expected_required = expected_projection and terminal_state.non_calibration_gates_passed
        if self.terminal_condition_required != expected_required:
            raise ValueError("confirmation_v2_terminal_requirement_mismatch")
        projection_fields = (
            self.terminal_condition_projection,
            self.terminal_condition_projection_sha256,
        )
        invocation_fields = (
            self.condition_gate_invocation_proof,
            self.condition_gate_invocation_proof_sha256,
        )
        if any(value is not None for value in projection_fields) != all(
            value is not None for value in projection_fields
        ):
            raise ValueError("confirmation_v2_terminal_projection_lineage_incomplete")
        if any(value is not None for value in invocation_fields) != all(
            value is not None for value in invocation_fields
        ):
            raise ValueError("confirmation_v2_terminal_invocation_lineage_incomplete")
        if expected_projection != all(value is not None for value in projection_fields):
            raise ValueError("confirmation_v2_terminal_projection_presence_mismatch")
        if expected_required != all(value is not None for value in invocation_fields):
            raise ValueError("confirmation_v2_terminal_invocation_presence_mismatch")
        if self.terminal_condition_projection is not None:
            projection = self.terminal_condition_projection
            invocation = self.condition_gate_invocation_proof
            if (
                self.terminal_condition_projection_sha256 != projection.projection_sha256
                or projection.online_graph_sha256 != terminal_state.evidence_graph_sha256
                or projection.pipeline_sha256 == ""
            ):
                raise ValueError("confirmation_v2_terminal_projection_state_mismatch")
            if invocation is not None and (
                self.condition_gate_invocation_proof_sha256 != invocation.proof_sha256
                or invocation.condition_projection != projection
                or self.base_arm.terminal_proof != invocation
            ):
                raise ValueError("confirmation_v2_terminal_invocation_state_mismatch")
        payload = self.model_dump(mode="json", exclude={"wrapper_sha256"})
        _reject_reference_leakage(payload)
        if hash_canonical(payload) != self.wrapper_sha256:
            raise ValueError("confirmation_v2_arm_wrapper_hash_mismatch")
        return self


def freeze_confirmation_aware_arm_trajectory(
    *,
    base_arm: AdaptivePolicyArmTrajectory,
    terminal_condition_projection: ConditionCalibrationProjectionV1 | None = None,
) -> ConfirmationAwareArmTrajectoryV2:
    """Wrap a complete pre-bundle path without adding circular certificate lineage."""

    try:
        base_arm = AdaptivePolicyArmTrajectory.model_validate(base_arm.model_dump(mode="json"))
        invocation = (
            base_arm.terminal_proof
            if isinstance(base_arm.terminal_proof, ConditionGateInvocationProofV2)
            else None
        )
        projection = (
            None
            if terminal_condition_projection is None
            else ConditionCalibrationProjectionV1.model_validate(
                terminal_condition_projection.model_dump(mode="json")
            )
        )
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError("confirmation_v2_arm_input_integrity_changed") from exc
    payload: dict[str, Any] = {
        "wrapper_version": "confirmation-aware-arm-trajectory-v2",
        "base_arm": base_arm,
        "terminal_condition_required": (
            base_arm.states[-1].claim_decision == "condition_dependent"
            and base_arm.states[-1].non_calibration_gates_passed
        ),
        "terminal_condition_projection": projection,
        "terminal_condition_projection_sha256": (
            None if projection is None else projection.projection_sha256
        ),
        "condition_gate_invocation_proof": invocation,
        "condition_gate_invocation_proof_sha256": (
            None if invocation is None else invocation.proof_sha256
        ),
    }
    return ConfirmationAwareArmTrajectoryV2.model_validate(
        {**payload, "wrapper_sha256": hash_canonical(payload)}
    )


class PolicyVisibleQuestionTrajectoryV2(ContractModel):
    """Outcome-free v2 trajectory with a target-bound terminal gate slot."""

    trajectory_version: Literal["policy-visible-question-trajectory-v2"] = (
        "policy-visible-question-trajectory-v2"
    )
    base_visible: PolicyVisibleQuestionTrajectory
    target_semantics: AdaptiveTargetSemanticsBindingV2
    target_semantics_sha256: str
    independence_identity: AdaptiveIndependenceIdentityV2
    independence_identity_sha256: str
    arms: list[ConfirmationAwareArmTrajectoryV2]
    trajectory_sha256: str

    @field_validator(
        "target_semantics_sha256",
        "independence_identity_sha256",
        "trajectory_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_trajectory(self) -> PolicyVisibleQuestionTrajectoryV2:
        if (
            self.target_semantics_sha256 != self.target_semantics.target_semantics_sha256
            or self.base_visible.question_id != self.target_semantics.question_id
            or self.independence_identity_sha256
            != self.independence_identity.independence_identity_sha256
        ):
            raise ValueError("confirmation_v2_question_target_semantics_mismatch")
        expected_arms = [arm.policy_arm_id for arm in self.base_visible.arms]
        observed_arms = [arm.base_arm.policy_arm_id for arm in self.arms]
        if observed_arms != expected_arms or [arm.base_arm for arm in self.arms] != (
            self.base_visible.arms
        ):
            raise ValueError("confirmation_v2_base_arm_family_mismatch")
        for arm in self.arms:
            projection = arm.terminal_condition_projection
            if projection is None:
                continue
            if (
                projection.question_id != self.base_visible.question_id
                or projection.target_semantics_sha256 != self.target_semantics_sha256
                or projection.claim_spec_sha256 != self.target_semantics.claim_spec_sha256
                or projection.condition_target_sha256
                != self.target_semantics.global_condition_target_sha256
                or projection.independence_identity_sha256 != self.independence_identity_sha256
                or projection.corpus_snapshot_sha256 != self.base_visible.corpus.membership_sha256
                or projection.corpus_cutoff != self.base_visible.corpus.corpus_cutoff
            ):
                raise ValueError("confirmation_v2_projection_question_context_mismatch")
        payload = self.model_dump(mode="json", exclude={"trajectory_sha256"})
        _reject_reference_leakage(payload)
        if hash_canonical(payload) != self.trajectory_sha256:
            raise ValueError("confirmation_v2_visible_trajectory_hash_mismatch")
        return self


def freeze_policy_visible_question_trajectory_v2(
    *,
    base_visible: PolicyVisibleQuestionTrajectory,
    target_semantics: AdaptiveTargetSemanticsBindingV2,
    independence_identity: AdaptiveIndependenceIdentityV2,
    arms: Sequence[ConfirmationAwareArmTrajectoryV2],
) -> PolicyVisibleQuestionTrajectoryV2:
    try:
        base_visible = PolicyVisibleQuestionTrajectory.model_validate(
            base_visible.model_dump(mode="json")
        )
        target_semantics = AdaptiveTargetSemanticsBindingV2.model_validate(
            target_semantics.model_dump(mode="json")
        )
        independence_identity = AdaptiveIndependenceIdentityV2.model_validate(
            independence_identity.model_dump(mode="json")
        )
        normalized_arms = [
            ConfirmationAwareArmTrajectoryV2.model_validate(arm.model_dump(mode="json"))
            for arm in arms
        ]
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError("confirmation_v2_visible_input_integrity_changed") from exc
    normalized_arms.sort(key=lambda row: row.base_arm.policy_arm_id)
    payload: dict[str, Any] = {
        "trajectory_version": "policy-visible-question-trajectory-v2",
        "base_visible": base_visible,
        "target_semantics": target_semantics,
        "target_semantics_sha256": target_semantics.target_semantics_sha256,
        "independence_identity": independence_identity,
        "independence_identity_sha256": (independence_identity.independence_identity_sha256),
        "arms": normalized_arms,
    }
    return PolicyVisibleQuestionTrajectoryV2.model_validate(
        {**payload, "trajectory_sha256": hash_canonical(payload)}
    )


class ConditionCalibrationGateResultV1(ContractModel):
    """Calibration-only gate derived from a replayed, non-releasable collection source."""

    result_version: Literal["condition-calibration-gate-result-v1"] = (
        "condition-calibration-gate-result-v1"
    )
    question_id: Annotated[str, Field(min_length=1)]
    policy_arm_id: Annotated[str, Field(min_length=1)]
    terminal_state_sha256: str
    condition_gate_invocation_proof_sha256: str
    condition_projection: ConditionCalibrationProjectionV1
    condition_projection_sha256: str
    gate_assessment: ConditionConfirmationGateAssessmentV1
    gate_assessment_sha256: str
    collection_source_sha256: str
    collection_source_decision_sha256: str
    status: Literal["missing", "confirmed", "not_confirmed", "insufficient"]
    result_sha256: str

    @field_validator(
        "terminal_state_sha256",
        "condition_gate_invocation_proof_sha256",
        "condition_projection_sha256",
        "gate_assessment_sha256",
        "collection_source_sha256",
        "collection_source_decision_sha256",
        "result_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_result(self) -> ConditionCalibrationGateResultV1:
        projection = self.condition_projection
        gate = self.gate_assessment
        if (
            self.condition_projection_sha256 != projection.projection_sha256
            or self.gate_assessment_sha256 != gate.gate_assessment_sha256
            or self.question_id != projection.question_id
            or not gate.required
            or gate.provisional_claim_decision != "condition_dependent"
            or gate.status != self.status
            or gate.condition_projection_sha256 != projection.projection_sha256
            or gate.target_sha256 != projection.condition_target_sha256
            or gate.plan_sha256 != projection.plan_sha256
            or gate.config_sha256 != projection.confirmation_config_sha256
        ):
            raise ValueError("confirmation_v2_calibration_gate_projection_mismatch")
        payload = self.model_dump(mode="json", exclude={"result_sha256"})
        if hash_canonical(payload) != self.result_sha256:
            raise ValueError("confirmation_v2_calibration_gate_result_hash_mismatch")
        return self

    @property
    def passed(self) -> bool:
        return self.status == "confirmed" and self.gate_assessment.scientific_gate_passed


def freeze_condition_calibration_gate_result_v1(
    *,
    question_id: str,
    policy_arm_id: str,
    condition_gate_invocation_proof: ConditionGateInvocationProofV2,
    gate_assessment: ConditionConfirmationGateAssessmentV1,
    collection_source_sha256: str,
    collection_source_decision_sha256: str,
) -> ConditionCalibrationGateResultV1:
    """Derive calibration gate lineage; only a full receipt may enter calibration."""

    try:
        invocation = ConditionGateInvocationProofV2.model_validate(
            condition_gate_invocation_proof.model_dump(mode="json")
        )
        projection = invocation.condition_projection
        gate = ConditionConfirmationGateAssessmentV1.model_validate(
            gate_assessment.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError(
            "confirmation_v2_calibration_gate_input_integrity_changed"
        ) from exc
    if gate.status == "not_applicable":
        raise AdaptiveCalibrationError("confirmation_v2_calibration_gate_not_required")
    payload: dict[str, Any] = {
        "result_version": "condition-calibration-gate-result-v1",
        "question_id": question_id,
        "policy_arm_id": policy_arm_id,
        "terminal_state_sha256": invocation.terminal_preselection_state.state_sha256,
        "condition_gate_invocation_proof_sha256": invocation.proof_sha256,
        "condition_projection": projection,
        "condition_projection_sha256": projection.projection_sha256,
        "gate_assessment": gate,
        "gate_assessment_sha256": gate.gate_assessment_sha256,
        "collection_source_sha256": collection_source_sha256,
        "collection_source_decision_sha256": collection_source_decision_sha256,
        "status": gate.status,
    }
    return ConditionCalibrationGateResultV1.model_validate(
        {**payload, "result_sha256": hash_canonical(payload)}
    )


class ConditionTerminalGateResultV2(ContractModel):
    """Production-only terminal gate derived by the immutable v6-to-v7 finalizer."""

    result_version: Literal["condition-terminal-gate-result-v2"] = (
        "condition-terminal-gate-result-v2"
    )
    question_id: Annotated[str, Field(min_length=1)]
    policy_arm_id: Annotated[str, Field(min_length=1)]
    terminal_state_sha256: str
    condition_gate_invocation_proof_sha256: str
    condition_projection: ConditionCalibrationProjectionV1
    condition_projection_sha256: str
    gate_assessment: ConditionConfirmationGateAssessmentV1
    gate_assessment_sha256: str
    source_v6_certificate_sha256: str
    source_v6_decision_sha256: str
    status: Literal["missing", "confirmed", "not_confirmed", "insufficient"]
    result_sha256: str

    @field_validator(
        "terminal_state_sha256",
        "condition_gate_invocation_proof_sha256",
        "condition_projection_sha256",
        "gate_assessment_sha256",
        "source_v6_certificate_sha256",
        "source_v6_decision_sha256",
        "result_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_result(self) -> ConditionTerminalGateResultV2:
        projection = self.condition_projection
        gate = self.gate_assessment
        if (
            self.condition_projection_sha256 != projection.projection_sha256
            or self.gate_assessment_sha256 != gate.gate_assessment_sha256
            or self.question_id != projection.question_id
            or not gate.required
            or gate.provisional_claim_decision != "condition_dependent"
            or gate.status != self.status
            or gate.condition_projection_sha256 != projection.projection_sha256
            or gate.target_sha256 != projection.condition_target_sha256
            or gate.plan_sha256 != projection.plan_sha256
            or gate.config_sha256 != projection.confirmation_config_sha256
        ):
            raise ValueError("confirmation_v2_terminal_gate_projection_mismatch")
        payload = self.model_dump(mode="json", exclude={"result_sha256"})
        if hash_canonical(payload) != self.result_sha256:
            raise ValueError("confirmation_v2_terminal_gate_result_hash_mismatch")
        return self

    @property
    def passed(self) -> bool:
        return self.status == "confirmed" and self.gate_assessment.scientific_gate_passed


def freeze_condition_terminal_gate_result_v2(
    *,
    question_id: str,
    policy_arm_id: str,
    condition_gate_invocation_proof: ConditionGateInvocationProofV2,
    gate_assessment: ConditionConfirmationGateAssessmentV1,
    source_v6_certificate_sha256: str,
    source_v6_decision_sha256: str,
) -> ConditionTerminalGateResultV2:
    """Bind a validated v6 runtime gate after the certificate hash exists."""

    try:
        invocation = ConditionGateInvocationProofV2.model_validate(
            condition_gate_invocation_proof.model_dump(mode="json")
        )
        projection = invocation.condition_projection
        gate = ConditionConfirmationGateAssessmentV1.model_validate(
            gate_assessment.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError(
            "confirmation_v2_terminal_gate_input_integrity_changed"
        ) from exc
    if gate.status == "not_applicable":
        raise AdaptiveCalibrationError("confirmation_v2_terminal_gate_not_required")
    payload: dict[str, Any] = {
        "result_version": "condition-terminal-gate-result-v2",
        "question_id": question_id,
        "policy_arm_id": policy_arm_id,
        "terminal_state_sha256": (invocation.terminal_preselection_state.state_sha256),
        "condition_gate_invocation_proof_sha256": (invocation.proof_sha256),
        "condition_projection": projection,
        "condition_projection_sha256": projection.projection_sha256,
        "gate_assessment": gate,
        "gate_assessment_sha256": gate.gate_assessment_sha256,
        "source_v6_certificate_sha256": source_v6_certificate_sha256,
        "source_v6_decision_sha256": source_v6_decision_sha256,
        "status": gate.status,
    }
    return ConditionTerminalGateResultV2.model_validate(
        {**payload, "result_sha256": hash_canonical(payload)}
    )


def condition_terminal_gate_result_from_certificate_v6(
    source_certificate: Any,
) -> ConditionTerminalGateResultV2:
    """Project only the explicit missing-gate slot from an outcome-free v6.

    The delayed import avoids a module cycle: certificate validation depends on the
    neutral adaptive contracts, while this bridge is only invoked after a v6 has
    already acquired its immutable certificate hash. A scientifically materialized
    gate result must instead be built by the v6-to-v7 finalizer after it independently
    validates the held-out assessment; this helper can never create a confirmed result.
    """

    from literature_multiverse.certificate import ConditionVerificationCertificateV6

    if not isinstance(source_certificate, ConditionVerificationCertificateV6):
        raise AdaptiveCalibrationError("confirmation_v2_source_certificate_contract_invalid")
    try:
        source = ConditionVerificationCertificateV6.model_validate(
            source_certificate.model_dump(mode="json")
        )
    except ValueError as exc:
        raise AdaptiveCalibrationError(
            "confirmation_v2_source_certificate_integrity_changed"
        ) from exc
    invocation = source.condition_gate_invocation_proof
    if invocation is None:
        raise AdaptiveCalibrationError("confirmation_v2_source_certificate_invocation_missing")
    if (
        source.status != "abstained"
        or source.production_stop_decision.outcome != "condition_gate_ready"
        or source.condition_confirmation_assessment is not None
        or source.condition_confirmation_gate.status != "missing"
        or source.condition_calibration_projection != invocation.condition_projection
        or source.production_stop_decision.condition_gate_invocation_proof != invocation
        or source.adaptive_release_candidate_v2.condition_gate_invocation_proof != invocation
        or source.adaptive_release_candidate_v2.terminal_gate_result is not None
    ):
        raise AdaptiveCalibrationError(
            "confirmation_v2_source_certificate_terminal_lineage_mismatch"
        )
    return freeze_condition_terminal_gate_result_v2(
        question_id=source.release_assessment.question_id,
        policy_arm_id=source.adaptive_policy_context.policy_arm_id,
        condition_gate_invocation_proof=invocation,
        gate_assessment=source.condition_confirmation_gate,
        source_v6_certificate_sha256=source.certificate_sha256,
        source_v6_decision_sha256=source.release_assessment.decision_sha256,
    )


class GateCompleteQuestionTrajectoryV2(ContractModel):
    """Externally replayed calibration receipts frozen before references open."""

    gate_join_version: Literal["gate-complete-question-trajectory-v2"] = (
        "gate-complete-question-trajectory-v2"
    )
    freeze_state: Literal["reference_labels_unopened"] = "reference_labels_unopened"
    visible: PolicyVisibleQuestionTrajectoryV2
    collection_source_roster_sha256: str | None = None
    collection_source_membership_sha256: str | None = None
    collection_source_anchors: list[ConditionCalibrationCollectionSourceAnchorV1] = Field(
        default_factory=list
    )
    calibration_assessment_receipts: list[dict[str, JsonValue]] = Field(default_factory=list)
    calibration_gate_results: list[ConditionCalibrationGateResultV1] = Field(default_factory=list)
    gate_join_sha256: str

    @field_validator(
        "collection_source_roster_sha256",
        "collection_source_membership_sha256",
        "gate_join_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_gate_join(self) -> GateCompleteQuestionTrajectoryV2:
        receipts = [
            _externally_replay_calibration_assessment_receipt(receipt)
            for receipt in self.calibration_assessment_receipts
        ]
        receipt_ids = [receipt.policy_arm_id for receipt in receipts]
        result_ids = [row.policy_arm_id for row in self.calibration_gate_results]
        required = [
            arm.base_arm.policy_arm_id
            for arm in self.visible.arms
            if arm.terminal_condition_required
        ]
        if receipt_ids != required or result_ids != required:
            raise ValueError("confirmation_v2_receipt_arm_coverage_mismatch")
        anchored = bool(required)
        if anchored != (
            self.collection_source_roster_sha256 is not None
            and self.collection_source_membership_sha256 is not None
            and bool(self.collection_source_anchors)
        ):
            raise ValueError("confirmation_v2_receipt_source_binding_incomplete")
        arm_by_id = {arm.base_arm.policy_arm_id: arm for arm in self.visible.arms}
        anchor_by_id = {anchor.policy_arm_id: anchor for anchor in self.collection_source_anchors}
        if list(anchor_by_id) != required:
            raise ValueError("confirmation_v2_receipt_source_anchor_coverage_mismatch")
        result_by_id = {result.policy_arm_id: result for result in self.calibration_gate_results}
        for receipt in receipts:
            result = result_by_id[receipt.policy_arm_id]
            arm = arm_by_id[result.policy_arm_id]
            projection = arm.terminal_condition_projection
            assert projection is not None
            anchor = anchor_by_id[result.policy_arm_id]
            expected_anchor = _collection_source_anchor(receipt.collection_source)
            if (
                result.question_id != self.visible.base_visible.question_id
                or receipt.question_id != self.visible.base_visible.question_id
                or receipt.policy_visible_question_trajectory != self.visible
                or receipt.calibration_gate_result != result
                or receipt.source_roster_sha256 != self.collection_source_roster_sha256
                or receipt.source_membership_sha256 != self.collection_source_membership_sha256
                or receipt.source_anchor != anchor
                or result.condition_projection != projection
                or result.terminal_state_sha256 != arm.base_arm.states[-1].state_sha256
                or result.condition_gate_invocation_proof_sha256
                != arm.condition_gate_invocation_proof_sha256
                or anchor != expected_anchor
            ):
                raise ValueError("confirmation_v2_receipt_terminal_lineage_mismatch")
        payload = self.model_dump(mode="json", exclude={"gate_join_sha256"})
        _reject_reference_leakage(payload, allow_terminal_gate_outcomes=True)
        if hash_canonical(payload) != self.gate_join_sha256:
            raise ValueError("confirmation_v2_gate_join_hash_mismatch")
        return self


def _externally_replay_calibration_assessment_receipt(receipt: Any) -> Any:
    try:
        from literature_multiverse.verifier import (
            validate_condition_calibration_assessment_receipt_external_replay,
        )

        return validate_condition_calibration_assessment_receipt_external_replay(receipt)
    except (ImportError, AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError(
            f"confirmation_v2_calibration_receipt_external_replay_failed:{exc}"
        ) from exc


def join_condition_calibration_assessment_receipts(
    *,
    visible: PolicyVisibleQuestionTrajectoryV2,
    calibration_roster: AdaptiveCalibrationRosterV2,
    calibration_assessment_receipts: Sequence[Any],
) -> GateCompleteQuestionTrajectoryV2:
    try:
        visible = PolicyVisibleQuestionTrajectoryV2.model_validate(visible.model_dump(mode="json"))
        roster = AdaptiveCalibrationRosterV2.model_validate(
            calibration_roster.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError("confirmation_v2_receipt_join_input_changed") from exc
    if visible not in roster.visible_trajectories:
        raise AdaptiveCalibrationError("confirmation_v2_receipt_visible_not_in_frozen_roster")
    receipts = sorted(
        (
            _externally_replay_calibration_assessment_receipt(receipt)
            for receipt in calibration_assessment_receipts
        ),
        key=lambda receipt: receipt.policy_arm_id,
    )
    required = [
        arm.base_arm.policy_arm_id for arm in visible.arms if arm.terminal_condition_required
    ]
    receipt_pairs = [(receipt.question_id, receipt.policy_arm_id) for receipt in receipts]
    expected_pairs = [(visible.base_visible.question_id, arm_id) for arm_id in required]
    if receipt_pairs != expected_pairs:
        raise AdaptiveCalibrationError("confirmation_v2_receipt_arm_coverage_mismatch")
    anchors = [
        anchor
        for anchor in roster.collection_source_anchors
        if anchor.question_id == visible.base_visible.question_id
        and anchor.policy_arm_id in required
    ]
    if required and roster.collection_source_status != ("externally_replayed_before_assessment"):
        raise AdaptiveCalibrationError("confirmation_v2_receipt_requires_preoutcome_source_roster")
    payload: dict[str, Any] = {
        "gate_join_version": "gate-complete-question-trajectory-v2",
        "freeze_state": "reference_labels_unopened",
        "visible": visible,
        "collection_source_roster_sha256": (
            roster.collection_source_roster_sha256 if required else None
        ),
        "collection_source_membership_sha256": (
            roster.collection_source_membership_sha256 if required else None
        ),
        "collection_source_anchors": anchors,
        "calibration_assessment_receipts": [
            receipt.model_dump(mode="json") for receipt in receipts
        ],
        "calibration_gate_results": [receipt.calibration_gate_result for receipt in receipts],
    }
    return GateCompleteQuestionTrajectoryV2.model_validate(
        {**payload, "gate_join_sha256": hash_canonical(payload)}
    )


def join_terminal_condition_gates(**_: Any) -> GateCompleteQuestionTrajectoryV2:
    """Reject the pre-v2 bare-result join; full replayed receipts are mandatory."""

    raise AdaptiveCalibrationError("confirmation_v2_bare_terminal_gate_results_forbidden")


class QuestionReferenceVerdictV2(ContractModel):
    """Hidden five-way verdict bound to the exact global-condition target."""

    reference_version: Literal["question-reference-verdict-v2"] = "question-reference-verdict-v2"
    question_id: Annotated[str, Field(min_length=1)]
    verdict: AdaptiveClaimDecision
    target_semantics_sha256: str
    claim_spec_sha256: str
    global_condition_target_sha256: str
    label_source: AdaptiveLabelSource
    adjudicator_count: Annotated[int, Field(ge=1)]
    adjudication_protocol_sha256: str
    adjudication_artifact_sha256: str
    reference_sha256: str

    @field_validator(
        "target_semantics_sha256",
        "claim_spec_sha256",
        "global_condition_target_sha256",
        "adjudication_protocol_sha256",
        "adjudication_artifact_sha256",
        "reference_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_reference(self) -> QuestionReferenceVerdictV2:
        if self.label_source == "expert_adjudication" and self.adjudicator_count < 2:
            raise ValueError("confirmation_v2_expert_reference_requires_two_adjudicators")
        payload = self.model_dump(mode="json", exclude={"reference_sha256"})
        if hash_canonical(payload) != self.reference_sha256:
            raise ValueError("confirmation_v2_reference_hash_mismatch")
        return self


def freeze_question_reference_verdict_v2(
    *,
    question_id: str,
    verdict: AdaptiveClaimDecision,
    target_semantics: AdaptiveTargetSemanticsBindingV2,
    label_source: AdaptiveLabelSource,
    adjudicator_count: int,
    adjudication_protocol_sha256: str,
    adjudication_artifact_sha256: str,
) -> QuestionReferenceVerdictV2:
    payload: dict[str, Any] = {
        "reference_version": "question-reference-verdict-v2",
        "question_id": question_id,
        "verdict": verdict,
        "target_semantics_sha256": target_semantics.target_semantics_sha256,
        "claim_spec_sha256": target_semantics.claim_spec_sha256,
        "global_condition_target_sha256": (target_semantics.global_condition_target_sha256),
        "label_source": label_source,
        "adjudicator_count": adjudicator_count,
        "adjudication_protocol_sha256": adjudication_protocol_sha256,
        "adjudication_artifact_sha256": adjudication_artifact_sha256,
    }
    return QuestionReferenceVerdictV2.model_validate(
        {**payload, "reference_sha256": hash_canonical(payload)}
    )


class LabeledQuestionTrajectoryV2(ContractModel):
    """Development-only reference join; terminal outcomes remain unnecessary."""

    labeled_version: Literal["labeled-question-trajectory-v2"] = "labeled-question-trajectory-v2"
    visible: PolicyVisibleQuestionTrajectoryV2
    reference: QuestionReferenceVerdictV2
    labeled_trajectory_sha256: str

    @field_validator("labeled_trajectory_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "labeled_trajectory_sha256")

    @model_validator(mode="after")
    def validate_join(self) -> LabeledQuestionTrajectoryV2:
        semantics = self.visible.target_semantics
        if (
            self.reference.question_id != self.visible.base_visible.question_id
            or self.reference.target_semantics_sha256 != semantics.target_semantics_sha256
            or self.reference.claim_spec_sha256 != semantics.claim_spec_sha256
            or self.reference.global_condition_target_sha256
            != semantics.global_condition_target_sha256
        ):
            raise ValueError("confirmation_v2_reference_target_identity_mismatch")
        _reject_reference_leakage(self.visible.model_dump(mode="json"))
        payload = self.model_dump(mode="json", exclude={"labeled_trajectory_sha256"})
        if hash_canonical(payload) != self.labeled_trajectory_sha256:
            raise ValueError("confirmation_v2_labeled_trajectory_hash_mismatch")
        return self


def join_labeled_question_trajectory_v2(
    *,
    visible: PolicyVisibleQuestionTrajectoryV2,
    reference: QuestionReferenceVerdictV2,
) -> LabeledQuestionTrajectoryV2:
    try:
        visible = PolicyVisibleQuestionTrajectoryV2.model_validate(visible.model_dump(mode="json"))
        reference = QuestionReferenceVerdictV2.model_validate(reference.model_dump(mode="json"))
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError("confirmation_v2_development_join_input_changed") from exc
    payload: dict[str, Any] = {
        "labeled_version": "labeled-question-trajectory-v2",
        "visible": visible,
        "reference": reference,
    }
    return LabeledQuestionTrajectoryV2.model_validate(
        {**payload, "labeled_trajectory_sha256": hash_canonical(payload)}
    )


class GateCompleteLabeledQuestionTrajectoryV2(ContractModel):
    """Calibration-only join after every terminal gate outcome is frozen."""

    labeled_version: Literal["gate-complete-labeled-question-trajectory-v2"] = (
        "gate-complete-labeled-question-trajectory-v2"
    )
    gate_complete: GateCompleteQuestionTrajectoryV2
    reference: QuestionReferenceVerdictV2
    labeled_trajectory_sha256: str

    @field_validator("labeled_trajectory_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "labeled_trajectory_sha256")

    @model_validator(mode="after")
    def validate_join(self) -> GateCompleteLabeledQuestionTrajectoryV2:
        visible = self.gate_complete.visible
        semantics = visible.target_semantics
        if (
            self.reference.question_id != visible.base_visible.question_id
            or self.reference.target_semantics_sha256 != semantics.target_semantics_sha256
            or self.reference.claim_spec_sha256 != semantics.claim_spec_sha256
            or self.reference.global_condition_target_sha256
            != semantics.global_condition_target_sha256
        ):
            raise ValueError("confirmation_v2_reference_target_identity_mismatch")
        payload = self.model_dump(mode="json", exclude={"labeled_trajectory_sha256"})
        if hash_canonical(payload) != self.labeled_trajectory_sha256:
            raise ValueError("confirmation_v2_gate_labeled_hash_mismatch")
        return self


def join_gate_complete_labeled_question_trajectory_v2(
    *,
    gate_complete: GateCompleteQuestionTrajectoryV2,
    reference: QuestionReferenceVerdictV2,
) -> GateCompleteLabeledQuestionTrajectoryV2:
    try:
        gate_complete = GateCompleteQuestionTrajectoryV2.model_validate(
            gate_complete.model_dump(mode="json")
        )
        reference = QuestionReferenceVerdictV2.model_validate(reference.model_dump(mode="json"))
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError("confirmation_v2_calibration_join_input_changed") from exc
    payload: dict[str, Any] = {
        "labeled_version": "gate-complete-labeled-question-trajectory-v2",
        "gate_complete": gate_complete,
        "reference": reference,
    }
    return GateCompleteLabeledQuestionTrajectoryV2.model_validate(
        {**payload, "labeled_trajectory_sha256": hash_canonical(payload)}
    )


def _validate_confirmation_v2_independence(
    rows: Sequence[PolicyVisibleQuestionTrajectoryV2],
    *,
    split_name: str,
) -> bool:
    token_owner: dict[str, str] = {}
    component_owner: dict[str, str] = {}
    for row in rows:
        question_id = row.base_visible.question_id
        identity = row.independence_identity
        for label, values, owner in (
            (
                "token",
                identity.strong_identity_token_sha256s,
                token_owner,
            ),
            (
                "component",
                identity.strong_component_sha256s,
                component_owner,
            ),
        ):
            for digest in values:
                previous = owner.setdefault(digest, question_id)
                if previous != question_id:
                    raise AdaptiveCalibrationError(
                        f"confirmation_v2_{split_name}_{label}_overlap:{digest}"
                    )
    return all(row.independence_identity.verification_status == "verified" for row in rows)


def _validate_confirmation_v2_cross_split_independence(
    development: Sequence[PolicyVisibleQuestionTrajectoryV2],
    calibration: Sequence[PolicyVisibleQuestionTrajectoryV2],
) -> None:
    for label, attribute in (
        ("token", "strong_identity_token_sha256s"),
        ("component", "strong_component_sha256s"),
    ):
        development_values = {
            digest
            for row in development
            for digest in getattr(row.independence_identity, attribute)
        }
        calibration_values = {
            digest
            for row in calibration
            for digest in getattr(row.independence_identity, attribute)
        }
        overlap = sorted(development_values & calibration_values)
        if overlap:
            raise AdaptiveCalibrationError(f"confirmation_v2_cross_split_{label}_overlap:{overlap}")


class ConditionCalibrationCollectionSourceAnchorV1(ContractModel):
    """Content-silent membership record for one pre-outcome collection source."""

    anchor_version: Literal["condition-calibration-source-anchor-v1"] = (
        "condition-calibration-source-anchor-v1"
    )
    question_id: Annotated[str, Field(min_length=1)]
    policy_arm_id: Annotated[str, Field(min_length=1)]
    policy_context_sha256: str
    visible_trajectory_sha256: str
    collection_source_sha256: str
    collection_source_decision_sha256: str
    anchor_sha256: str

    @field_validator(
        "policy_context_sha256",
        "visible_trajectory_sha256",
        "collection_source_sha256",
        "collection_source_decision_sha256",
        "anchor_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_anchor(self) -> ConditionCalibrationCollectionSourceAnchorV1:
        payload = self.model_dump(mode="json", exclude={"anchor_sha256"})
        if hash_canonical(payload) != self.anchor_sha256:
            raise ValueError("confirmation_v2_collection_source_anchor_hash_mismatch")
        return self


def _collection_source_anchor(source: Any) -> ConditionCalibrationCollectionSourceAnchorV1:
    visible = source.policy_visible_question_trajectory
    if visible is None:
        raise AdaptiveCalibrationError(
            "confirmation_v2_collection_source_complete_trajectory_required"
        )
    payload: dict[str, Any] = {
        "anchor_version": "condition-calibration-source-anchor-v1",
        "question_id": source.question_id,
        "policy_arm_id": source.policy_arm_id,
        "policy_context_sha256": source.adaptive_policy_context.policy_context_sha256,
        "visible_trajectory_sha256": visible.trajectory_sha256,
        "collection_source_sha256": source.collection_source_sha256,
        "collection_source_decision_sha256": (source.collection_source_decision_sha256),
    }
    return ConditionCalibrationCollectionSourceAnchorV1.model_validate(
        {**payload, "anchor_sha256": hash_canonical(payload)}
    )


def _externally_replay_collection_source(source: Any) -> Any:
    """Delayed public replay keeps this dependency-neutral module cycle-free."""

    try:
        from literature_multiverse.certificate import (
            ConditionCalibrationCollectionSourceV1,
        )
        from literature_multiverse.verifier import (
            validate_condition_calibration_collection_source_external_replay,
        )

        canonical = ConditionCalibrationCollectionSourceV1.model_validate(
            source.model_dump(mode="json") if hasattr(source, "model_dump") else source
        )
        return validate_condition_calibration_collection_source_external_replay(canonical)
    except (ImportError, AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError(
            f"confirmation_v2_collection_source_external_replay_failed:{exc}"
        ) from exc


def _structurally_canonicalize_collection_source(source: Any) -> Any:
    """Canonicalize a source without duplicating the roster replay boundary."""

    try:
        from literature_multiverse.certificate import (
            ConditionCalibrationCollectionSourceV1,
        )

        return ConditionCalibrationCollectionSourceV1.model_validate(
            source.model_dump(mode="json") if hasattr(source, "model_dump") else source
        )
    except (ImportError, AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError(
            f"confirmation_v2_collection_source_invalid:{exc}"
        ) from exc


class ConditionCalibrationCollectionSourceRosterV1(ContractModel):
    """Full outcome-free sources externally replayed and anchored before assessment."""

    roster_version: Literal["condition-calibration-source-roster-v1"] = (
        "condition-calibration-source-roster-v1"
    )
    freeze_state: Literal["condition_assessments_and_reference_labels_unopened"] = (
        "condition_assessments_and_reference_labels_unopened"
    )
    collection_sources: list[dict[str, JsonValue]]
    source_anchors: list[ConditionCalibrationCollectionSourceAnchorV1]
    source_membership_sha256: str
    source_roster_sha256: str

    @field_validator("source_membership_sha256", "source_roster_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_roster(self) -> ConditionCalibrationCollectionSourceRosterV1:
        if not self.collection_sources:
            raise ValueError("confirmation_v2_collection_source_roster_empty")
        replayed = [
            _externally_replay_collection_source(source) for source in self.collection_sources
        ]
        pairs = [(source.question_id, source.policy_arm_id) for source in replayed]
        if pairs != sorted(set(pairs)):
            raise ValueError("confirmation_v2_collection_source_pairs_must_be_sorted_unique")
        if any(source.collection_split != "calibration" for source in replayed):
            raise ValueError("confirmation_v2_collection_source_split_mismatch")
        expected_anchors = [_collection_source_anchor(source) for source in replayed]
        if self.source_anchors != expected_anchors:
            raise ValueError("confirmation_v2_collection_source_anchor_mismatch")
        visible_by_question: dict[str, PolicyVisibleQuestionTrajectoryV2] = {}
        for source in replayed:
            visible = source.policy_visible_question_trajectory
            assert visible is not None
            previous = visible_by_question.setdefault(source.question_id, visible)
            if previous != visible:
                raise ValueError("confirmation_v2_collection_source_question_trajectory_changed")
        expected_pairs = sorted(
            (
                visible.base_visible.question_id,
                arm.base_arm.policy_arm_id,
            )
            for visible in visible_by_question.values()
            for arm in visible.arms
        )
        if pairs != expected_pairs:
            raise ValueError("confirmation_v2_collection_source_arm_coverage_mismatch")
        if hash_canonical(self.source_anchors) != self.source_membership_sha256:
            raise ValueError("confirmation_v2_collection_source_membership_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"source_roster_sha256"})
        _reject_collection_source_outcome_leakage(payload)
        if hash_canonical(payload) != self.source_roster_sha256:
            raise ValueError("confirmation_v2_collection_source_roster_hash_mismatch")
        return self


def freeze_condition_calibration_collection_source_roster_v1(
    sources: Sequence[Any],
) -> ConditionCalibrationCollectionSourceRosterV1:
    """Canonicalize then replay once while sealing pre-assessment membership."""

    canonical = sorted(
        (_structurally_canonicalize_collection_source(source) for source in sources),
        key=lambda source: (source.question_id, source.policy_arm_id),
    )
    if not canonical:
        raise AdaptiveCalibrationError("confirmation_v2_collection_source_roster_empty")
    source_payloads = [source.model_dump(mode="json") for source in canonical]
    anchors = [_collection_source_anchor(source) for source in canonical]
    payload: dict[str, Any] = {
        "roster_version": "condition-calibration-source-roster-v1",
        "freeze_state": "condition_assessments_and_reference_labels_unopened",
        "collection_sources": source_payloads,
        "source_anchors": anchors,
        "source_membership_sha256": hash_canonical(anchors),
    }
    try:
        return ConditionCalibrationCollectionSourceRosterV1.model_validate(
            {**payload, "source_roster_sha256": hash_canonical(payload)}
        )
    except ValueError as exc:
        raise AdaptiveCalibrationError(str(exc)) from exc


class AdaptiveCalibrationRosterV2(ContractModel):
    """Question/gate slots frozen before confirmation outcomes or references open."""

    roster_version: Literal["adaptive-calibration-roster-v2"] = "adaptive-calibration-roster-v2"
    freeze_state: Literal["terminal_confirmation_outcomes_and_reference_labels_unopened"] = (
        "terminal_confirmation_outcomes_and_reference_labels_unopened"
    )
    visible_trajectories: list[PolicyVisibleQuestionTrajectoryV2]
    collection_source_roster_sha256: str | None = None
    collection_source_membership_sha256: str | None = None
    collection_source_anchors: list[ConditionCalibrationCollectionSourceAnchorV1] = Field(
        default_factory=list
    )
    collection_source_status: Literal[
        "externally_replayed_before_assessment",
        "unanchored_simulation_only",
    ]
    independence_verified: bool
    roster_sha256: str

    @field_validator("independence_verified", mode="before")
    @classmethod
    def validate_boolean(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("confirmation_v2_roster_independence_must_be_boolean")
        return value

    @field_validator(
        "collection_source_roster_sha256",
        "collection_source_membership_sha256",
        "roster_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_roster(self) -> AdaptiveCalibrationRosterV2:
        question_ids = [row.base_visible.question_id for row in self.visible_trajectories]
        if not question_ids or question_ids != sorted(set(question_ids)):
            raise ValueError("confirmation_v2_roster_questions_invalid")
        if any(row.base_visible.split != "calibration" for row in self.visible_trajectories):
            raise ValueError("confirmation_v2_roster_split_mismatch")
        if any(
            state.scalar_risk_score is not None
            for row in self.visible_trajectories
            for arm in row.arms
            for state in arm.base_arm.states
        ):
            raise ValueError("confirmation_v2_roster_must_be_unscored")
        expected_verified = _validate_confirmation_v2_independence(
            self.visible_trajectories,
            split_name="calibration",
        )
        if self.independence_verified != expected_verified:
            raise ValueError("confirmation_v2_roster_independence_status_mismatch")
        anchored = self.collection_source_status == ("externally_replayed_before_assessment")
        if anchored != (
            self.collection_source_roster_sha256 is not None
            and self.collection_source_membership_sha256 is not None
            and bool(self.collection_source_anchors)
        ):
            raise ValueError("confirmation_v2_collection_source_binding_incomplete")
        if anchored:
            anchor_pairs = [
                (anchor.question_id, anchor.policy_arm_id)
                for anchor in self.collection_source_anchors
            ]
            expected_pairs = sorted(
                (row.base_visible.question_id, arm.base_arm.policy_arm_id)
                for row in self.visible_trajectories
                for arm in row.arms
            )
            if anchor_pairs != expected_pairs:
                raise ValueError("confirmation_v2_collection_source_anchor_coverage_mismatch")
            visible_sha_by_question = {
                row.base_visible.question_id: row.trajectory_sha256
                for row in self.visible_trajectories
            }
            if any(
                anchor.visible_trajectory_sha256 != visible_sha_by_question[anchor.question_id]
                for anchor in self.collection_source_anchors
            ):
                raise ValueError("confirmation_v2_collection_source_visible_binding_mismatch")
            if (
                hash_canonical(self.collection_source_anchors)
                != self.collection_source_membership_sha256
            ):
                raise ValueError("confirmation_v2_collection_source_membership_hash_mismatch")
        elif self.collection_source_anchors:
            raise ValueError("confirmation_v2_unanchored_roster_has_source_anchors")
        payload = self.model_dump(mode="json", exclude={"roster_sha256"})
        _reject_reference_leakage(payload)
        if hash_canonical(payload) != self.roster_sha256:
            raise ValueError("confirmation_v2_roster_hash_mismatch")
        return self


def _freeze_adaptive_calibration_roster_v2(
    rows: Sequence[PolicyVisibleQuestionTrajectoryV2],
    *,
    collection_source_roster: ConditionCalibrationCollectionSourceRosterV1 | None,
) -> AdaptiveCalibrationRosterV2:
    normalized = sorted(rows, key=lambda row: row.base_visible.question_id)
    verified = _validate_confirmation_v2_independence(
        normalized,
        split_name="calibration",
    )
    payload: dict[str, Any] = {
        "roster_version": "adaptive-calibration-roster-v2",
        "freeze_state": ("terminal_confirmation_outcomes_and_reference_labels_unopened"),
        "visible_trajectories": normalized,
        "collection_source_roster_sha256": (
            None
            if collection_source_roster is None
            else collection_source_roster.source_roster_sha256
        ),
        "collection_source_membership_sha256": (
            None
            if collection_source_roster is None
            else collection_source_roster.source_membership_sha256
        ),
        "collection_source_anchors": (
            [] if collection_source_roster is None else collection_source_roster.source_anchors
        ),
        "collection_source_status": (
            "unanchored_simulation_only"
            if collection_source_roster is None
            else "externally_replayed_before_assessment"
        ),
        "independence_verified": verified,
    }
    return AdaptiveCalibrationRosterV2.model_validate(
        {**payload, "roster_sha256": hash_canonical(payload)}
    )


class AdaptiveCalibrationPlanV2(ContractModel):
    """Joint threshold/terminal-gate rule sealed before calibration opens."""

    plan_version: Literal["adaptive-calibration-plan-v2"] = "adaptive-calibration-plan-v2"
    alpha: Annotated[float, Field(gt=0, lt=1)]
    delta: Annotated[float, Field(gt=0, lt=1)]
    threshold_family_sha256: str
    calibration_roster_sha256: str
    condition_release_domains: list[str]
    reference_loss: Literal["exact_released_decision_mismatch"] = "exact_released_decision_mismatch"
    joint_release_rule: Literal[
        "first threshold-qualified base release; condition-dependent qualifies only at "
        "the first outcome-free terminal invocation proof with confirmed held-out gate"
    ] = (
        "first threshold-qualified base release; condition-dependent qualifies only at "
        "the first outcome-free terminal invocation proof with confirmed held-out gate"
    )
    multiplicity_correction: Literal[
        "bonferroni-clopper-pearson-across-candidates-overall-and-condition-domain-strata"
    ] = "bonferroni-clopper-pearson-across-candidates-overall-and-condition-domain-strata"
    candidate_selection_rule: Literal[
        "max-accepted-then-min-upper-risk-then-max-threshold-then-max-arm-id"
    ] = "max-accepted-then-min-upper-risk-then-max-threshold-then-max-arm-id"
    plan_sha256: str

    @field_validator(
        "threshold_family_sha256",
        "calibration_roster_sha256",
        "plan_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("condition_release_domains")
    @classmethod
    def validate_condition_release_domains(cls, value: list[str]) -> list[str]:
        if not value or value != sorted(set(value)) or any(not item for item in value):
            raise ValueError("confirmation_v2_condition_release_domains_invalid")
        return value

    @model_validator(mode="after")
    def validate_plan(self) -> AdaptiveCalibrationPlanV2:
        payload = self.model_dump(mode="json", exclude={"plan_sha256"})
        if hash_canonical(payload) != self.plan_sha256:
            raise ValueError("confirmation_v2_calibration_plan_hash_mismatch")
        return self


def _freeze_adaptive_calibration_plan_v2(
    *,
    alpha: float,
    delta: float,
    threshold_family_sha256: str,
    calibration_roster_sha256: str,
    condition_release_domains: Sequence[str],
) -> AdaptiveCalibrationPlanV2:
    payload: dict[str, Any] = {
        "plan_version": "adaptive-calibration-plan-v2",
        "alpha": float(alpha),
        "delta": float(delta),
        "threshold_family_sha256": threshold_family_sha256,
        "calibration_roster_sha256": calibration_roster_sha256,
        "condition_release_domains": sorted(set(condition_release_domains)),
        "reference_loss": "exact_released_decision_mismatch",
        "joint_release_rule": (
            "first threshold-qualified base release; condition-dependent qualifies only at "
            "the first outcome-free terminal invocation proof with confirmed held-out gate"
        ),
        "multiplicity_correction": (
            "bonferroni-clopper-pearson-across-candidates-overall-and-condition-domain-strata"
        ),
        "candidate_selection_rule": (
            "max-accepted-then-min-upper-risk-then-max-threshold-then-max-arm-id"
        ),
    }
    return AdaptiveCalibrationPlanV2.model_validate(
        {**payload, "plan_sha256": hash_canonical(payload)}
    )


def _v1_reference_from_v2(reference: QuestionReferenceVerdictV2) -> QuestionReferenceVerdict:
    return freeze_question_reference_verdict(
        question_id=reference.question_id,
        verdict=reference.verdict,
        label_source=reference.label_source,
        adjudication_protocol_sha256=reference.adjudication_protocol_sha256,
        adjudication_artifact_sha256=reference.adjudication_artifact_sha256,
    )


def _v1_labeled_from_v2(row: LabeledQuestionTrajectoryV2) -> LabeledQuestionTrajectory:
    return join_labeled_question_trajectory(
        visible=row.visible.base_visible,
        reference=_v1_reference_from_v2(row.reference),
    )


class AdaptiveDevelopmentFreezeV2(ContractModel):
    """Version-separated model/family plus outcome-free calibration gate roster."""

    freeze_version: Literal["adaptive-development-freeze-v2"] = "adaptive-development-freeze-v2"
    freeze_state: Literal["terminal_confirmation_outcomes_and_calibration_labels_unopened"] = (
        "terminal_confirmation_outcomes_and_calibration_labels_unopened"
    )
    base_freeze: AdaptiveDevelopmentFreeze
    development_trajectories: list[LabeledQuestionTrajectoryV2]
    calibration_roster: AdaptiveCalibrationRosterV2
    calibration_plan: AdaptiveCalibrationPlanV2
    independence_verified: bool
    development_freeze_sha256: str

    @field_validator("independence_verified", mode="before")
    @classmethod
    def validate_boolean(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("confirmation_v2_freeze_independence_must_be_boolean")
        return value

    @field_validator("development_freeze_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "development_freeze_sha256")

    @model_validator(mode="after")
    def validate_freeze(self) -> AdaptiveDevelopmentFreezeV2:
        question_ids = [
            row.visible.base_visible.question_id for row in self.development_trajectories
        ]
        if not question_ids or question_ids != sorted(set(question_ids)):
            raise ValueError("confirmation_v2_development_questions_invalid")
        if any(
            row.visible.base_visible.split != "development" for row in self.development_trajectories
        ):
            raise ValueError("confirmation_v2_development_split_mismatch")
        development_visible = [row.visible for row in self.development_trajectories]
        development_verified = _validate_confirmation_v2_independence(
            development_visible,
            split_name="development",
        )
        _validate_confirmation_v2_cross_split_independence(
            development_visible,
            self.calibration_roster.visible_trajectories,
        )
        expected_verified = development_verified and self.calibration_roster.independence_verified
        if self.independence_verified != expected_verified:
            raise ValueError("confirmation_v2_freeze_independence_status_mismatch")
        expected_v1_development = [
            _v1_labeled_from_v2(row) for row in self.development_trajectories
        ]
        observed_v1_development = [
            _unscore_labeled_question_trajectory(row)
            for row in self.base_freeze.scored_development_trajectories
        ]
        if observed_v1_development != expected_v1_development:
            raise ValueError("confirmation_v2_base_development_projection_mismatch")
        if self.base_freeze.calibration_roster.visible_trajectories != [
            row.base_visible for row in self.calibration_roster.visible_trajectories
        ]:
            raise ValueError("confirmation_v2_base_calibration_roster_mismatch")
        if (
            self.calibration_plan.threshold_family_sha256
            != self.base_freeze.threshold_family.family_sha256
            or self.calibration_plan.calibration_roster_sha256
            != self.calibration_roster.roster_sha256
            or self.calibration_plan.condition_release_domains
            != sorted(
                {row.base_visible.domain for row in self.calibration_roster.visible_trajectories}
            )
            or self.calibration_plan.alpha != self.base_freeze.calibration_plan.alpha
            or self.calibration_plan.delta != self.base_freeze.calibration_plan.delta
        ):
            raise ValueError("confirmation_v2_plan_base_freeze_mismatch")
        context_by_arm = {
            context.policy_arm_id: context for context in self.base_freeze.policy_contexts
        }
        for row in (
            *development_visible,
            *self.calibration_roster.visible_trajectories,
        ):
            for arm in row.arms:
                projection = arm.terminal_condition_projection
                if projection is not None and (
                    projection.pipeline_sha256
                    != context_by_arm[arm.base_arm.policy_arm_id].pipeline_sha256
                ):
                    raise ValueError("confirmation_v2_projection_pipeline_context_mismatch")
        payload = self.model_dump(mode="json", exclude={"development_freeze_sha256"})
        if hash_canonical(payload) != self.development_freeze_sha256:
            raise ValueError("confirmation_v2_development_freeze_hash_mismatch")
        return self


def fit_adaptive_development_v2(
    trajectories: Sequence[LabeledQuestionTrajectoryV2],
    *,
    policy_contexts: Sequence[AdaptivePolicyContext],
    calibration_visible_trajectories: Sequence[PolicyVisibleQuestionTrajectoryV2],
    calibration_collection_source_roster: (
        ConditionCalibrationCollectionSourceRosterV1 | None
    ) = None,
    alpha: float,
    delta: float,
    candidate_thresholds: Mapping[str, Sequence[float]] | None = None,
    seed: int = 20260827,
) -> AdaptiveDevelopmentFreezeV2:
    """Freeze v2 candidates before either calibration gate outcomes or labels open."""

    development = sorted(
        (
            LabeledQuestionTrajectoryV2.model_validate(row.model_dump(mode="json"))
            for row in trajectories
        ),
        key=lambda row: row.visible.base_visible.question_id,
    )
    calibration_visible = sorted(
        (
            PolicyVisibleQuestionTrajectoryV2.model_validate(row.model_dump(mode="json"))
            for row in calibration_visible_trajectories
        ),
        key=lambda row: row.base_visible.question_id,
    )
    source_roster = None
    if calibration_collection_source_roster is not None:
        try:
            source_roster = ConditionCalibrationCollectionSourceRosterV1.model_validate(
                calibration_collection_source_roster.model_dump(mode="json")
            )
        except (AttributeError, ValueError) as exc:
            raise AdaptiveCalibrationError(
                "confirmation_v2_collection_source_roster_integrity_changed"
            ) from exc
        source_visible_by_question: dict[str, PolicyVisibleQuestionTrajectoryV2] = {}
        for source_payload in source_roster.collection_sources:
            # The roster reparse immediately above is the mandatory full external
            # replay boundary; only structural access is needed for this equality
            # projection.
            source = _structurally_canonicalize_collection_source(source_payload)
            visible = source.policy_visible_question_trajectory
            assert visible is not None
            source_visible_by_question.setdefault(source.question_id, visible)
        if list(source_visible_by_question.values()) != calibration_visible:
            raise AdaptiveCalibrationError(
                "confirmation_v2_collection_source_visible_roster_mismatch"
            )
    development_visible = [row.visible for row in development]
    development_verified = _validate_confirmation_v2_independence(
        development_visible,
        split_name="development",
    )
    _validate_confirmation_v2_cross_split_independence(
        development_visible,
        calibration_visible,
    )
    roster = _freeze_adaptive_calibration_roster_v2(
        calibration_visible,
        collection_source_roster=source_roster,
    )
    base_freeze = fit_adaptive_development(
        [_v1_labeled_from_v2(row) for row in development],
        policy_contexts=policy_contexts,
        calibration_visible_trajectories=[row.base_visible for row in calibration_visible],
        alpha=alpha,
        delta=delta,
        candidate_thresholds=candidate_thresholds,
        seed=seed,
    )
    plan = _freeze_adaptive_calibration_plan_v2(
        alpha=alpha,
        delta=delta,
        threshold_family_sha256=base_freeze.threshold_family.family_sha256,
        calibration_roster_sha256=roster.roster_sha256,
        condition_release_domains=sorted({row.base_visible.domain for row in calibration_visible}),
    )
    payload: dict[str, Any] = {
        "freeze_version": "adaptive-development-freeze-v2",
        "freeze_state": ("terminal_confirmation_outcomes_and_calibration_labels_unopened"),
        "base_freeze": base_freeze,
        "development_trajectories": development,
        "calibration_roster": roster,
        "calibration_plan": plan,
        "independence_verified": (development_verified and roster.independence_verified),
    }
    return AdaptiveDevelopmentFreezeV2.model_validate(
        {**payload, "development_freeze_sha256": hash_canonical(payload)}
    )


class AdaptiveSplitIdentityV2(ContractModel):
    split: Literal["development", "calibration"]
    question_ids: list[str]
    domains: list[str]
    strong_identity_token_sha256s: list[str]
    strong_component_sha256s: list[str]
    labeled_trajectory_sha256s: list[str]
    independence_verified: bool

    @field_validator(
        "question_ids",
        "domains",
        "strong_identity_token_sha256s",
        "strong_component_sha256s",
        "labeled_trajectory_sha256s",
    )
    @classmethod
    def validate_sorted(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)) or (
            info.field_name in {"question_ids", "domains", "labeled_trajectory_sha256s"}
            and not value
        ):
            raise ValueError(f"confirmation_v2_split_identity_invalid:{info.field_name}")
        if info.field_name.endswith("sha256s"):
            for digest in value:
                _validate_sha256(digest, info.field_name)
        return value

    @field_validator("independence_verified", mode="before")
    @classmethod
    def validate_boolean(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("confirmation_v2_split_independence_must_be_boolean")
        return value


class GateCompleteCalibrationRosterV2(ContractModel):
    """Exact terminal-gate roster sealed before calibration references open."""

    roster_version: Literal["gate-complete-calibration-roster-v2"] = (
        "gate-complete-calibration-roster-v2"
    )
    freeze_state: Literal["reference_labels_unopened"] = "reference_labels_unopened"
    development_freeze_sha256: str
    calibration_roster_sha256: str
    collection_source_roster_sha256: str | None = None
    collection_source_membership_sha256: str | None = None
    trajectories: list[GateCompleteQuestionTrajectoryV2]
    gate_roster_sha256: str

    @field_validator(
        "development_freeze_sha256",
        "calibration_roster_sha256",
        "collection_source_roster_sha256",
        "collection_source_membership_sha256",
        "gate_roster_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_roster(self) -> GateCompleteCalibrationRosterV2:
        question_ids = [row.visible.base_visible.question_id for row in self.trajectories]
        if not question_ids or question_ids != sorted(set(question_ids)):
            raise ValueError("confirmation_v2_gate_roster_questions_invalid")
        if any(row.visible.base_visible.split != "calibration" for row in self.trajectories):
            raise ValueError("confirmation_v2_gate_roster_split_mismatch")
        for row in self.trajectories:
            required = any(arm.terminal_condition_required for arm in row.visible.arms)
            if required and (
                row.collection_source_roster_sha256 != self.collection_source_roster_sha256
                or row.collection_source_membership_sha256
                != self.collection_source_membership_sha256
            ):
                raise ValueError("confirmation_v2_gate_roster_source_lineage_mismatch")
        has_condition_receipts = any(
            row.calibration_assessment_receipts for row in self.trajectories
        )
        if has_condition_receipts != (
            self.collection_source_roster_sha256 is not None
            and self.collection_source_membership_sha256 is not None
        ):
            raise ValueError("confirmation_v2_gate_roster_source_binding_incomplete")
        payload = self.model_dump(mode="json", exclude={"gate_roster_sha256"})
        _reject_reference_leakage(payload, allow_terminal_gate_outcomes=True)
        if hash_canonical(payload) != self.gate_roster_sha256:
            raise ValueError("confirmation_v2_gate_roster_hash_mismatch")
        return self


def freeze_gate_complete_calibration_roster_v2(
    *,
    development_freeze: AdaptiveDevelopmentFreezeV2,
    trajectories: Sequence[GateCompleteQuestionTrajectoryV2],
) -> GateCompleteCalibrationRosterV2:
    """Seal every calibration terminal outcome without accepting a reference label."""

    try:
        freeze = AdaptiveDevelopmentFreezeV2.model_validate(
            development_freeze.model_dump(mode="json")
        )
        rows = sorted(
            (
                GateCompleteQuestionTrajectoryV2.model_validate(row.model_dump(mode="json"))
                for row in trajectories
            ),
            key=lambda row: row.visible.base_visible.question_id,
        )
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError("confirmation_v2_gate_roster_input_changed") from exc
    if [row.visible for row in rows] != freeze.calibration_roster.visible_trajectories:
        raise AdaptiveCalibrationError("confirmation_v2_gate_roster_visible_roster_mismatch")
    if any(
        any(arm.terminal_condition_required for arm in row.visible.arms)
        and (
            row.collection_source_roster_sha256
            != freeze.calibration_roster.collection_source_roster_sha256
            or row.collection_source_membership_sha256
            != freeze.calibration_roster.collection_source_membership_sha256
        )
        for row in rows
    ):
        raise AdaptiveCalibrationError("confirmation_v2_gate_roster_source_freeze_mismatch")
    payload: dict[str, Any] = {
        "roster_version": "gate-complete-calibration-roster-v2",
        "freeze_state": "reference_labels_unopened",
        "development_freeze_sha256": freeze.development_freeze_sha256,
        "calibration_roster_sha256": freeze.calibration_roster.roster_sha256,
        "collection_source_roster_sha256": (
            freeze.calibration_roster.collection_source_roster_sha256
            if any(row.calibration_assessment_receipts for row in rows)
            else None
        ),
        "collection_source_membership_sha256": (
            freeze.calibration_roster.collection_source_membership_sha256
            if any(row.calibration_assessment_receipts for row in rows)
            else None
        ),
        "trajectories": rows,
    }
    return GateCompleteCalibrationRosterV2.model_validate(
        {**payload, "gate_roster_sha256": hash_canonical(payload)}
    )


def _split_identity_v2(
    rows: Sequence[LabeledQuestionTrajectoryV2 | GateCompleteLabeledQuestionTrajectoryV2],
    *,
    split: Literal["development", "calibration"],
) -> AdaptiveSplitIdentityV2:
    visible_rows = [
        row.visible if isinstance(row, LabeledQuestionTrajectoryV2) else row.gate_complete.visible
        for row in rows
    ]
    return AdaptiveSplitIdentityV2(
        split=split,
        question_ids=sorted(row.base_visible.question_id for row in visible_rows),
        domains=sorted({row.base_visible.domain for row in visible_rows}),
        strong_identity_token_sha256s=sorted(
            {
                digest
                for row in visible_rows
                for digest in row.independence_identity.strong_identity_token_sha256s
            }
        ),
        strong_component_sha256s=sorted(
            {
                digest
                for row in visible_rows
                for digest in row.independence_identity.strong_component_sha256s
            }
        ),
        labeled_trajectory_sha256s=sorted(row.labeled_trajectory_sha256 for row in rows),
        independence_verified=all(
            row.independence_identity.verification_status == "verified" for row in visible_rows
        ),
    )


class AdaptiveQuestionOutcomeV2(ContractModel):
    """One joint-rule outcome for one candidate and complete question."""

    outcome_version: Literal["adaptive-question-outcome-v2"] = "adaptive-question-outcome-v2"
    question_id: Annotated[str, Field(min_length=1)]
    candidate_sha256: str
    accepted: bool
    error: bool
    released_claim_decision: AdaptiveClaimDecision | None = None
    first_release_prefix_index: int | None = Field(default=None, ge=0)
    release_state_sha256: str | None = None
    scalar_risk_score: Annotated[float, Field(ge=0, le=1)] | None = None
    calibration_gate_result_sha256: str | None = None
    reference_sha256: str
    target_semantics_sha256: str
    outcome_sha256: str

    @field_validator(
        "candidate_sha256",
        "release_state_sha256",
        "calibration_gate_result_sha256",
        "reference_sha256",
        "target_semantics_sha256",
        "outcome_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _validate_sha256(value, info.field_name)

    @field_validator("accepted", "error", mode="before")
    @classmethod
    def validate_boolean(cls, value: object, info: Any) -> object:
        if not isinstance(value, bool):
            raise ValueError(f"confirmation_v2_outcome_{info.field_name}_not_boolean")
        return value

    @model_validator(mode="after")
    def validate_outcome(self) -> AdaptiveQuestionOutcomeV2:
        if self.error and not self.accepted:
            raise ValueError("confirmation_v2_abstention_cannot_be_error")
        release_fields = (
            self.released_claim_decision,
            self.first_release_prefix_index,
            self.release_state_sha256,
            self.scalar_risk_score,
        )
        if self.accepted != all(value is not None for value in release_fields):
            raise ValueError("confirmation_v2_outcome_release_fields_mismatch")
        if self.accepted and (self.released_claim_decision == "condition_dependent") != (
            self.calibration_gate_result_sha256 is not None
        ):
            raise ValueError("confirmation_v2_condition_release_gate_lineage_mismatch")
        payload = self.model_dump(mode="json", exclude={"outcome_sha256"})
        if hash_canonical(payload) != self.outcome_sha256:
            raise ValueError("confirmation_v2_question_outcome_hash_mismatch")
        return self


def _replay_candidate_v2(
    trajectories: Sequence[GateCompleteLabeledQuestionTrajectoryV2],
    candidate: AdaptiveThresholdCandidate,
    *,
    score_models: Sequence[AdaptiveTrajectoryScoreModel],
) -> list[AdaptiveQuestionOutcomeV2]:
    model_by_arm = {model.policy_arm_id: model for model in score_models}
    outcomes: list[AdaptiveQuestionOutcomeV2] = []
    for row in sorted(
        trajectories,
        key=lambda item: item.gate_complete.visible.base_visible.question_id,
    ):
        visible = row.gate_complete.visible
        arm = next(
            arm for arm in visible.arms if arm.base_arm.policy_arm_id == candidate.policy_arm_id
        )
        gate_by_arm = {
            result.policy_arm_id: result for result in row.gate_complete.calibration_gate_results
        }
        gate = gate_by_arm.get(candidate.policy_arm_id)
        model = model_by_arm[candidate.policy_arm_id]
        release_state: AdaptivePreselectionState | None = None
        release_score: float | None = None
        release_gate: ConditionCalibrationGateResultV1 | None = None
        for index, state in enumerate(arm.base_arm.states):
            score = model.score_features(state.score_features)
            if not state.non_calibration_gates_passed or score > candidate.threshold:
                continue
            if state.claim_decision == "condition_dependent":
                if index != len(arm.base_arm.states) - 1 or gate is None or not gate.passed:
                    continue
                release_gate = gate
            elif state.claim_decision not in {
                "supported",
                "contradicted",
                "inconclusive",
                "not_evaluable",
            }:
                raise AdaptiveCalibrationError("confirmation_v2_state_claim_decision_invalid")
            release_state = state
            release_score = score
            break
        accepted = release_state is not None
        released_decision = None if release_state is None else release_state.claim_decision
        error = bool(accepted and released_decision != row.reference.verdict)
        payload: dict[str, Any] = {
            "outcome_version": "adaptive-question-outcome-v2",
            "question_id": visible.base_visible.question_id,
            "candidate_sha256": candidate.candidate_sha256,
            "accepted": accepted,
            "error": error,
            "released_claim_decision": released_decision,
            "first_release_prefix_index": (
                None if release_state is None else release_state.prefix_index
            ),
            "release_state_sha256": (None if release_state is None else release_state.state_sha256),
            "scalar_risk_score": release_score,
            "calibration_gate_result_sha256": (
                None if release_gate is None else release_gate.result_sha256
            ),
            "reference_sha256": row.reference.reference_sha256,
            "target_semantics_sha256": row.reference.target_semantics_sha256,
        }
        outcomes.append(
            AdaptiveQuestionOutcomeV2.model_validate(
                {**payload, "outcome_sha256": hash_canonical(payload)}
            )
        )
    return outcomes


class AdaptiveConditionDomainCalibrationV2(ContractModel):
    """Candidate-specific risk bound for confirmed condition releases in one domain."""

    candidate_sha256: str
    policy_arm_id: Annotated[str, Field(min_length=1)]
    domain: Annotated[str, Field(min_length=1)]
    total_questions: Annotated[int, Field(ge=1)]
    confirmed_condition_releases: Annotated[int, Field(ge=0)]
    errors: Annotated[int, Field(ge=0)]
    empirical_risk: Annotated[float, Field(ge=0, le=1)] | None
    simultaneous_upper_risk: Annotated[float, Field(ge=0, le=1)] | None
    passed: bool

    @field_validator("candidate_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "candidate_sha256")

    @field_validator("passed", mode="before")
    @classmethod
    def validate_boolean(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("confirmation_v2_condition_domain_passed_not_boolean")
        return value

    @model_validator(mode="after")
    def validate_counts(self) -> AdaptiveConditionDomainCalibrationV2:
        if self.confirmed_condition_releases > self.total_questions:
            raise ValueError("confirmation_v2_condition_domain_release_count_invalid")
        if self.errors > self.confirmed_condition_releases:
            raise ValueError("confirmation_v2_condition_domain_error_count_invalid")
        expected_empirical = (
            self.errors / self.confirmed_condition_releases
            if self.confirmed_condition_releases
            else None
        )
        if self.empirical_risk != expected_empirical:
            raise ValueError("confirmation_v2_condition_domain_empirical_risk_mismatch")
        if self.confirmed_condition_releases == 0 and (
            self.simultaneous_upper_risk is not None or self.passed
        ):
            raise ValueError("confirmation_v2_condition_domain_zero_support_must_fail")
        return self


def _condition_domain_calibrations_v2(
    trajectories: Sequence[GateCompleteLabeledQuestionTrajectoryV2],
    outcomes: Sequence[AdaptiveQuestionOutcomeV2],
    candidate: AdaptiveThresholdCandidate,
    *,
    domains: Sequence[str],
    simultaneous_delta: float,
    alpha: float,
) -> list[AdaptiveConditionDomainCalibrationV2]:
    domain_by_question = {
        row.gate_complete.visible.base_visible.question_id: (
            row.gate_complete.visible.base_visible.domain
        )
        for row in trajectories
    }
    if set(domain_by_question) != {row.question_id for row in outcomes}:
        raise AdaptiveCalibrationError("confirmation_v2_condition_domain_roster_mismatch")
    rows: list[AdaptiveConditionDomainCalibrationV2] = []
    for domain in domains:
        domain_outcomes = [row for row in outcomes if domain_by_question[row.question_id] == domain]
        confirmed = [
            row
            for row in domain_outcomes
            if row.accepted and row.released_claim_decision == "condition_dependent"
        ]
        errors = sum(row.error for row in confirmed)
        upper = (
            clopper_pearson_upper(
                errors,
                len(confirmed),
                delta=simultaneous_delta,
            )
            if confirmed
            else None
        )
        rows.append(
            AdaptiveConditionDomainCalibrationV2(
                candidate_sha256=candidate.candidate_sha256,
                policy_arm_id=candidate.policy_arm_id,
                domain=domain,
                total_questions=len(domain_outcomes),
                confirmed_condition_releases=len(confirmed),
                errors=errors,
                empirical_risk=errors / len(confirmed) if confirmed else None,
                simultaneous_upper_risk=upper,
                passed=upper is not None and upper <= alpha,
            )
        )
    return rows


class AdaptiveCandidateCalibrationV2(ContractModel):
    candidate: AdaptiveThresholdCandidate
    total_questions: Annotated[int, Field(ge=1)]
    outcomes: list[AdaptiveQuestionOutcomeV2]
    accepted: Annotated[int, Field(ge=0)]
    errors: Annotated[int, Field(ge=0)]
    empirical_risk: float | None
    simultaneous_upper_risk: float | None
    condition_domain_calibrations: list[AdaptiveConditionDomainCalibrationV2]
    passed: bool

    @model_validator(mode="after")
    def validate_counts(self) -> AdaptiveCandidateCalibrationV2:
        if len(self.outcomes) != self.total_questions:
            raise ValueError("confirmation_v2_candidate_denominator_mismatch")
        question_ids = [row.question_id for row in self.outcomes]
        if question_ids != sorted(set(question_ids)):
            raise ValueError("confirmation_v2_candidate_questions_not_unique")
        if any(row.candidate_sha256 != self.candidate.candidate_sha256 for row in self.outcomes):
            raise ValueError("confirmation_v2_candidate_outcome_identity_mismatch")
        if self.accepted != sum(row.accepted for row in self.outcomes):
            raise ValueError("confirmation_v2_candidate_accepted_count_mismatch")
        if self.errors != sum(row.error for row in self.outcomes):
            raise ValueError("confirmation_v2_candidate_error_count_mismatch")
        expected_empirical = self.errors / self.accepted if self.accepted else None
        if self.empirical_risk != expected_empirical:
            raise ValueError("confirmation_v2_candidate_empirical_risk_mismatch")
        domains = [row.domain for row in self.condition_domain_calibrations]
        if not domains or domains != sorted(set(domains)):
            raise ValueError("confirmation_v2_candidate_condition_domains_invalid")
        if any(
            row.candidate_sha256 != self.candidate.candidate_sha256
            or row.policy_arm_id != self.candidate.policy_arm_id
            for row in self.condition_domain_calibrations
        ):
            raise ValueError("confirmation_v2_candidate_condition_domain_identity_mismatch")
        return self


class AdaptiveCalibrationBundleV2(ContractModel):
    """Simultaneous exact-decision bound for the joint terminal-gate policy."""

    bundle_version: Literal["adaptive-question-trajectory-freeze-v2"] = (
        "adaptive-question-trajectory-freeze-v2"
    )
    freeze_state: Literal["test_labels_unopened"] = "test_labels_unopened"
    guarantee_scope: Literal[
        "exact released-decision mismatch overall and for confirmed condition-dependent "
        "releases within every frozen deployment domain for the frozen joint threshold "
        "and terminal confirmation policy under exchangeable independent complete "
        "questions; not scientific truth, causal proof, or domain-shift robustness"
    ] = (
        "exact released-decision mismatch overall and for confirmed condition-dependent "
        "releases within every frozen deployment domain for the frozen joint threshold "
        "and terminal confirmation policy under exchangeable independent complete "
        "questions; not scientific truth, causal proof, or domain-shift robustness"
    )
    population_id: Annotated[str, Field(min_length=1)]
    label_source: AdaptiveLabelSource
    adjudication_protocol_sha256: str
    alpha: Annotated[float, Field(gt=0, lt=1)]
    delta: Annotated[float, Field(gt=0, lt=1)]
    development_freeze: AdaptiveDevelopmentFreezeV2
    development_freeze_sha256: str
    gate_complete_roster: GateCompleteCalibrationRosterV2
    gate_complete_roster_sha256: str
    calibration: AdaptiveSplitIdentityV2
    calibration_trajectories: list[GateCompleteLabeledQuestionTrajectoryV2]
    calibration_input_sha256: str
    candidates: list[AdaptiveCandidateCalibrationV2]
    selected_candidate_sha256: str | None
    independence_verified: bool
    real_release_eligible: bool
    status: Literal["calibrated", "abstain_all"]
    bundle_sha256: str

    @field_validator(
        "adjudication_protocol_sha256",
        "development_freeze_sha256",
        "gate_complete_roster_sha256",
        "calibration_input_sha256",
        "selected_candidate_sha256",
        "bundle_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _validate_sha256(value, info.field_name)

    @field_validator("independence_verified", "real_release_eligible", mode="before")
    @classmethod
    def validate_boolean(cls, value: object, info: Any) -> object:
        if not isinstance(value, bool):
            raise ValueError(f"confirmation_v2_bundle_{info.field_name}_not_boolean")
        return value

    @model_validator(mode="after")
    def validate_bundle(self) -> AdaptiveCalibrationBundleV2:
        freeze = self.development_freeze
        if self.development_freeze_sha256 != freeze.development_freeze_sha256:
            raise ValueError("confirmation_v2_bundle_development_hash_mismatch")
        base = freeze.base_freeze
        if (
            self.population_id != base.population_id
            or self.alpha != freeze.calibration_plan.alpha
            or self.delta != freeze.calibration_plan.delta
        ):
            raise ValueError("confirmation_v2_bundle_plan_context_mismatch")
        gate_roster = self.gate_complete_roster
        if (
            self.gate_complete_roster_sha256 != gate_roster.gate_roster_sha256
            or gate_roster.development_freeze_sha256 != freeze.development_freeze_sha256
            or gate_roster.calibration_roster_sha256 != freeze.calibration_roster.roster_sha256
            or [row.gate_complete for row in self.calibration_trajectories]
            != gate_roster.trajectories
        ):
            raise ValueError("confirmation_v2_bundle_gate_roster_mismatch")
        visible = [row.gate_complete.visible for row in self.calibration_trajectories]
        if visible != freeze.calibration_roster.visible_trajectories:
            raise ValueError("confirmation_v2_bundle_visible_roster_mismatch")
        expected_identity = _split_identity_v2(
            self.calibration_trajectories,
            split="calibration",
        )
        if self.calibration != expected_identity:
            raise ValueError("confirmation_v2_bundle_split_identity_mismatch")
        if self.independence_verified != (
            freeze.independence_verified and expected_identity.independence_verified
        ):
            raise ValueError("confirmation_v2_bundle_independence_status_mismatch")
        if {row.reference.label_source for row in self.calibration_trajectories} != {
            self.label_source
        }:
            raise ValueError("confirmation_v2_bundle_label_source_mismatch")
        if {
            row.reference.adjudication_protocol_sha256 for row in self.calibration_trajectories
        } != {self.adjudication_protocol_sha256}:
            raise ValueError("confirmation_v2_bundle_adjudication_protocol_mismatch")
        if self.calibration_input_sha256 != hash_canonical(self.calibration_trajectories):
            raise ValueError("confirmation_v2_bundle_input_hash_mismatch")
        family = base.threshold_family.candidates
        if [row.candidate for row in self.candidates] != family:
            raise ValueError("confirmation_v2_bundle_candidate_family_mismatch")
        domains = self.calibration.domains
        if freeze.calibration_plan.condition_release_domains != domains:
            raise ValueError("confirmation_v2_bundle_condition_domain_plan_mismatch")
        simultaneous_delta = self.delta / (len(self.candidates) * (1 + len(domains)))
        for row in self.candidates:
            expected_outcomes = _replay_candidate_v2(
                self.calibration_trajectories,
                row.candidate,
                score_models=base.score_models,
            )
            if row.outcomes != expected_outcomes:
                raise ValueError("confirmation_v2_bundle_replay_outcome_mismatch")
            expected_upper = (
                clopper_pearson_upper(
                    row.errors,
                    row.accepted,
                    delta=simultaneous_delta,
                )
                if row.accepted
                else None
            )
            if expected_upper is None:
                if row.simultaneous_upper_risk is not None:
                    raise ValueError("confirmation_v2_bundle_upper_risk_mismatch")
            elif row.simultaneous_upper_risk is None or not math.isclose(
                row.simultaneous_upper_risk,
                expected_upper,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError("confirmation_v2_bundle_upper_risk_mismatch")
            expected_condition_domains = _condition_domain_calibrations_v2(
                self.calibration_trajectories,
                expected_outcomes,
                row.candidate,
                domains=domains,
                simultaneous_delta=simultaneous_delta,
                alpha=self.alpha,
            )
            if row.condition_domain_calibrations != expected_condition_domains:
                raise ValueError("confirmation_v2_bundle_condition_domain_replay_mismatch")
            expected_passed = (
                expected_upper is not None
                and expected_upper <= self.alpha
                and all(item.passed for item in expected_condition_domains)
            )
            if row.passed != expected_passed:
                raise ValueError("confirmation_v2_bundle_candidate_pass_mismatch")
        passing = [row for row in self.candidates if row.passed]
        selected = (
            None if not self.independence_verified else _select_calibrated_candidate_v2(passing)
        )
        selected_sha = None if selected is None else selected.candidate.candidate_sha256
        if self.selected_candidate_sha256 != selected_sha:
            raise ValueError("confirmation_v2_bundle_selected_candidate_mismatch")
        if (self.status == "calibrated") != (selected is not None):
            raise ValueError("confirmation_v2_bundle_status_mismatch")
        expected_real_eligible = (
            selected is not None
            and self.independence_verified
            and self.label_source != "simulation"
            and freeze.calibration_roster.collection_source_status
            == "externally_replayed_before_assessment"
            and self.gate_complete_roster.collection_source_roster_sha256
            == freeze.calibration_roster.collection_source_roster_sha256
            and self.gate_complete_roster.collection_source_membership_sha256
            == freeze.calibration_roster.collection_source_membership_sha256
        )
        if self.real_release_eligible != expected_real_eligible:
            raise ValueError("confirmation_v2_bundle_real_release_status_mismatch")
        payload = self.model_dump(mode="json", exclude={"bundle_sha256"})
        if hash_canonical(payload) != self.bundle_sha256:
            raise ValueError("confirmation_v2_bundle_hash_mismatch")
        return self

    @property
    def selected(self) -> AdaptiveCandidateCalibrationV2 | None:
        if self.selected_candidate_sha256 is None:
            return None
        return next(
            row
            for row in self.candidates
            if row.candidate.candidate_sha256 == self.selected_candidate_sha256
        )


def _select_calibrated_candidate_v2(
    passing: Sequence[AdaptiveCandidateCalibrationV2],
) -> AdaptiveCandidateCalibrationV2 | None:
    if not passing:
        return None
    return max(
        passing,
        key=lambda row: (
            row.accepted,
            -float(row.simultaneous_upper_risk or 1.0),
            row.candidate.threshold,
            row.candidate.policy_arm_id,
        ),
    )


def calibrate_confirmation_aware_first_release(
    development_freeze: AdaptiveDevelopmentFreezeV2,
    gate_complete_roster: GateCompleteCalibrationRosterV2,
    calibration_references: Sequence[QuestionReferenceVerdictV2],
) -> AdaptiveCalibrationBundleV2:
    """Calibrate the exact joint rule; a v1 winner is never reused or filtered."""

    try:
        freeze = AdaptiveDevelopmentFreezeV2.model_validate(
            development_freeze.model_dump(mode="json")
        )
        gate_roster = GateCompleteCalibrationRosterV2.model_validate(
            gate_complete_roster.model_dump(mode="json")
        )
        references = sorted(
            (
                QuestionReferenceVerdictV2.model_validate(row.model_dump(mode="json"))
                for row in calibration_references
            ),
            key=lambda row: row.question_id,
        )
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError("confirmation_v2_calibration_input_changed") from exc
    if (
        gate_roster.development_freeze_sha256 != freeze.development_freeze_sha256
        or gate_roster.calibration_roster_sha256 != freeze.calibration_roster.roster_sha256
        or gate_roster.collection_source_roster_sha256
        != freeze.calibration_roster.collection_source_roster_sha256
        or gate_roster.collection_source_membership_sha256
        != freeze.calibration_roster.collection_source_membership_sha256
        or [row.visible for row in gate_roster.trajectories]
        != freeze.calibration_roster.visible_trajectories
    ):
        raise AdaptiveCalibrationError("confirmation_v2_gate_roster_freeze_mismatch")
    expected_ids = [row.visible.base_visible.question_id for row in gate_roster.trajectories]
    reference_ids = [row.question_id for row in references]
    if reference_ids != expected_ids:
        raise AdaptiveCalibrationError("confirmation_v2_reference_roster_mismatch")
    reference_by_question = {row.question_id: row for row in references}
    rows = [
        join_gate_complete_labeled_question_trajectory_v2(
            gate_complete=row,
            reference=reference_by_question[row.visible.base_visible.question_id],
        )
        for row in gate_roster.trajectories
    ]
    label_sources = {row.reference.label_source for row in rows}
    protocols = {row.reference.adjudication_protocol_sha256 for row in rows}
    if len(label_sources) != 1:
        raise AdaptiveCalibrationError("confirmation_v2_label_source_changed")
    if len(protocols) != 1:
        raise AdaptiveCalibrationError("confirmation_v2_adjudication_protocol_changed")
    candidates: list[AdaptiveCandidateCalibrationV2] = []
    family = freeze.base_freeze.threshold_family.candidates
    domains = freeze.calibration_plan.condition_release_domains
    simultaneous_delta = freeze.calibration_plan.delta / (len(family) * (1 + len(domains)))
    for candidate in family:
        outcomes = _replay_candidate_v2(
            rows,
            candidate,
            score_models=freeze.base_freeze.score_models,
        )
        accepted = sum(row.accepted for row in outcomes)
        errors = sum(row.error for row in outcomes)
        upper = (
            clopper_pearson_upper(errors, accepted, delta=simultaneous_delta) if accepted else None
        )
        condition_domains = _condition_domain_calibrations_v2(
            rows,
            outcomes,
            candidate,
            domains=domains,
            simultaneous_delta=simultaneous_delta,
            alpha=freeze.calibration_plan.alpha,
        )
        candidates.append(
            AdaptiveCandidateCalibrationV2(
                candidate=candidate,
                total_questions=len(rows),
                outcomes=outcomes,
                accepted=accepted,
                errors=errors,
                empirical_risk=errors / accepted if accepted else None,
                simultaneous_upper_risk=upper,
                condition_domain_calibrations=condition_domains,
                passed=(
                    upper is not None
                    and upper <= freeze.calibration_plan.alpha
                    and all(row.passed for row in condition_domains)
                ),
            )
        )
    independence_verified = freeze.independence_verified and all(
        row.gate_complete.visible.independence_identity.verification_status == "verified"
        for row in rows
    )
    selected = (
        _select_calibrated_candidate_v2([row for row in candidates if row.passed])
        if independence_verified
        else None
    )
    label_source = next(iter(label_sources))
    payload: dict[str, Any] = {
        "bundle_version": "adaptive-question-trajectory-freeze-v2",
        "freeze_state": "test_labels_unopened",
        "guarantee_scope": (
            "exact released-decision mismatch overall and for confirmed condition-dependent "
            "releases within every frozen deployment domain for the frozen joint threshold "
            "and terminal confirmation policy under exchangeable independent complete "
            "questions; not scientific truth, causal proof, or domain-shift robustness"
        ),
        "population_id": freeze.base_freeze.population_id,
        "label_source": label_source,
        "adjudication_protocol_sha256": next(iter(protocols)),
        "alpha": freeze.calibration_plan.alpha,
        "delta": freeze.calibration_plan.delta,
        "development_freeze": freeze,
        "development_freeze_sha256": freeze.development_freeze_sha256,
        "gate_complete_roster": gate_roster,
        "gate_complete_roster_sha256": gate_roster.gate_roster_sha256,
        "calibration": _split_identity_v2(rows, split="calibration"),
        "calibration_trajectories": rows,
        "calibration_input_sha256": hash_canonical(rows),
        "candidates": candidates,
        "selected_candidate_sha256": (
            None if selected is None else selected.candidate.candidate_sha256
        ),
        "independence_verified": independence_verified,
        "real_release_eligible": (
            selected is not None
            and label_source != "simulation"
            and freeze.calibration_roster.collection_source_status
            == "externally_replayed_before_assessment"
        ),
        "status": "calibrated" if selected is not None else "abstain_all",
    }
    return AdaptiveCalibrationBundleV2.model_validate(
        {**payload, "bundle_sha256": hash_canonical(payload)}
    )


def validate_adaptive_calibration_bundle_v2_integrity(
    bundle: AdaptiveCalibrationBundleV2,
) -> AdaptiveCalibrationBundleV2:
    if not isinstance(bundle, AdaptiveCalibrationBundleV2):
        raise AdaptiveCalibrationError("confirmation_v2_bundle_contract_invalid")
    try:
        return AdaptiveCalibrationBundleV2.model_validate(bundle.model_dump(mode="json"))
    except ValueError as exc:
        raise AdaptiveCalibrationError("confirmation_v2_bundle_integrity_changed") from exc


class ConfirmationAwareReleaseQualificationProofV2(ContractModel):
    """Post-bundle proof that one terminal condition state passes the selected rule."""

    proof_version: Literal["confirmation-aware-release-qualification-proof-v2"] = (
        "confirmation-aware-release-qualification-proof-v2"
    )
    question_id: Annotated[str, Field(min_length=1)]
    policy_arm_id: Annotated[str, Field(min_length=1)]
    condition_gate_invocation_proof: ConditionGateInvocationProofV2
    condition_gate_invocation_proof_sha256: str
    frozen_bundle_sha256: str
    policy_context_sha256: str
    threshold_candidate: AdaptiveThresholdCandidate
    threshold_candidate_sha256: str
    score_model_sha256: str
    terminal_preselection_state_sha256: str
    scalar_risk_score: Annotated[float, Field(ge=0, le=1)]
    threshold: Annotated[float, Field(ge=0, le=1)]
    qualification_passed: Literal[True] = True
    proof_sha256: str

    @field_validator(
        "condition_gate_invocation_proof_sha256",
        "frozen_bundle_sha256",
        "policy_context_sha256",
        "threshold_candidate_sha256",
        "score_model_sha256",
        "terminal_preselection_state_sha256",
        "proof_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_qualification(self) -> ConfirmationAwareReleaseQualificationProofV2:
        invocation = self.condition_gate_invocation_proof
        candidate = self.threshold_candidate
        state = invocation.terminal_preselection_state
        if (
            self.condition_gate_invocation_proof_sha256 != invocation.proof_sha256
            or self.policy_arm_id != candidate.policy_arm_id
            or self.policy_context_sha256 != candidate.policy_context_sha256
            or self.threshold_candidate_sha256 != candidate.candidate_sha256
            or self.score_model_sha256 != candidate.score_model_sha256
            or self.terminal_preselection_state_sha256 != state.state_sha256
            or self.threshold != candidate.threshold
            or self.scalar_risk_score > self.threshold
        ):
            raise ValueError("confirmation_v2_release_qualification_mismatch")
        payload = self.model_dump(mode="json", exclude={"proof_sha256"})
        _reject_reference_leakage(payload)
        if hash_canonical(payload) != self.proof_sha256:
            raise ValueError("confirmation_v2_release_qualification_hash_mismatch")
        return self


def freeze_confirmation_aware_release_qualification_proof_v2(
    *,
    question_id: str,
    policy_arm_id: str,
    condition_gate_invocation_proof: ConditionGateInvocationProofV2,
    bundle: AdaptiveCalibrationBundleV2,
) -> ConfirmationAwareReleaseQualificationProofV2:
    """Score an invocation state only after the complete v2 bundle is frozen."""

    bundle = validate_adaptive_calibration_bundle_v2_integrity(bundle)
    try:
        invocation = ConditionGateInvocationProofV2.model_validate(
            condition_gate_invocation_proof.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError(
            "confirmation_v2_release_qualification_input_changed"
        ) from exc
    selected = bundle.selected
    if selected is None:
        raise AdaptiveCalibrationError("confirmation_v2_release_qualification_bundle_abstains_all")
    candidate = selected.candidate
    if candidate.policy_arm_id != policy_arm_id:
        raise AdaptiveCalibrationError("confirmation_v2_release_qualification_policy_arm_mismatch")
    model = next(
        row
        for row in bundle.development_freeze.base_freeze.score_models
        if row.policy_arm_id == policy_arm_id
    )
    score = model.score_features(invocation.terminal_preselection_state.score_features)
    if score > candidate.threshold:
        raise AdaptiveCalibrationError("confirmation_v2_release_qualification_risk_above_threshold")
    payload: dict[str, Any] = {
        "proof_version": "confirmation-aware-release-qualification-proof-v2",
        "question_id": question_id,
        "policy_arm_id": policy_arm_id,
        "condition_gate_invocation_proof": invocation,
        "condition_gate_invocation_proof_sha256": invocation.proof_sha256,
        "frozen_bundle_sha256": bundle.bundle_sha256,
        "policy_context_sha256": candidate.policy_context_sha256,
        "threshold_candidate": candidate,
        "threshold_candidate_sha256": candidate.candidate_sha256,
        "score_model_sha256": model.score_model_sha256,
        "terminal_preselection_state_sha256": (invocation.terminal_preselection_state.state_sha256),
        "scalar_risk_score": score,
        "threshold": candidate.threshold,
        "qualification_passed": True,
    }
    return ConfirmationAwareReleaseQualificationProofV2.model_validate(
        {**payload, "proof_sha256": hash_canonical(payload)}
    )


class ProspectiveAdaptiveReleaseCandidateV2(ContractModel):
    """Whole online prefix plus an optional post-selection terminal gate sidecar."""

    candidate_version: Literal["prospective-adaptive-release-candidate-v2"] = (
        "prospective-adaptive-release-candidate-v2"
    )
    base_candidate: ProspectiveAdaptiveReleaseCandidate
    target_semantics: AdaptiveTargetSemanticsBindingV2
    target_semantics_sha256: str
    independence_identity: AdaptiveIndependenceIdentityV2
    independence_identity_sha256: str
    condition_projection: ConditionCalibrationProjectionV1 | None = None
    condition_gate_invocation_proof: ConditionGateInvocationProofV2 | None = None
    release_qualification_proof: ConfirmationAwareReleaseQualificationProofV2 | None = None
    terminal_gate_result: ConditionTerminalGateResultV2 | None = None
    candidate_sha256: str

    @field_validator(
        "target_semantics_sha256",
        "independence_identity_sha256",
        "candidate_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_candidate(self) -> ProspectiveAdaptiveReleaseCandidateV2:
        base = self.base_candidate
        current = base.observed_states[-1]
        if (
            self.target_semantics_sha256 != self.target_semantics.target_semantics_sha256
            or self.independence_identity_sha256
            != self.independence_identity.independence_identity_sha256
            or base.question_id != self.target_semantics.question_id
        ):
            raise ValueError("confirmation_v2_prospective_identity_mismatch")
        requires_condition = current.claim_decision == "condition_dependent"
        if requires_condition != (self.condition_projection is not None):
            raise ValueError("confirmation_v2_prospective_projection_presence_mismatch")
        if self.condition_projection is not None:
            projection = self.condition_projection
            if (
                projection.question_id != base.question_id
                or projection.target_semantics_sha256 != self.target_semantics_sha256
                or projection.independence_identity_sha256 != self.independence_identity_sha256
                or projection.corpus_snapshot_sha256 != base.corpus.membership_sha256
                or projection.corpus_cutoff != base.corpus.corpus_cutoff
                or projection.online_graph_sha256 != current.evidence_graph_sha256
            ):
                raise ValueError("confirmation_v2_prospective_projection_mismatch")
        has_invocation = self.condition_gate_invocation_proof is not None
        has_qualification = self.release_qualification_proof is not None
        has_gate = self.terminal_gate_result is not None
        if (has_gate or has_qualification) and not has_invocation:
            raise ValueError("confirmation_v2_terminal_artifact_requires_invocation")
        if has_invocation:
            invocation = self.condition_gate_invocation_proof
            projection = self.condition_projection
            assert invocation is not None and projection is not None
            if (
                not requires_condition
                or invocation.terminal_preselection_state != current
                or invocation.condition_projection != projection
            ):
                raise ValueError("confirmation_v2_prospective_invocation_mismatch")
        if has_qualification:
            qualification = self.release_qualification_proof
            invocation = self.condition_gate_invocation_proof
            assert qualification is not None and invocation is not None
            if (
                qualification.question_id != base.question_id
                or qualification.policy_arm_id != base.policy_arm_id
                or qualification.condition_gate_invocation_proof != invocation
                or qualification.terminal_preselection_state_sha256 != current.state_sha256
            ):
                raise ValueError("confirmation_v2_prospective_qualification_mismatch")
        if has_gate:
            gate = self.terminal_gate_result
            projection = self.condition_projection
            invocation = self.condition_gate_invocation_proof
            assert gate is not None and projection is not None and invocation is not None
            if (
                gate.question_id != base.question_id
                or gate.policy_arm_id != base.policy_arm_id
                or gate.terminal_state_sha256 != current.state_sha256
                or gate.condition_gate_invocation_proof_sha256 != invocation.proof_sha256
                or gate.condition_projection != projection
            ):
                raise ValueError("confirmation_v2_prospective_gate_lineage_mismatch")
        payload = self.model_dump(mode="json", exclude={"candidate_sha256"})
        _reject_reference_leakage(payload, allow_terminal_gate_outcomes=True)
        if hash_canonical(payload) != self.candidate_sha256:
            raise ValueError("confirmation_v2_prospective_candidate_hash_mismatch")
        return self


def freeze_prospective_adaptive_candidate_v2(
    *,
    base_candidate: ProspectiveAdaptiveReleaseCandidate,
    target_semantics: AdaptiveTargetSemanticsBindingV2,
    independence_identity: AdaptiveIndependenceIdentityV2,
    condition_projection: ConditionCalibrationProjectionV1 | None = None,
    condition_gate_invocation_proof: ConditionGateInvocationProofV2 | None = None,
    release_qualification_proof: (ConfirmationAwareReleaseQualificationProofV2 | None) = None,
    terminal_gate_result: ConditionTerminalGateResultV2 | None = None,
) -> ProspectiveAdaptiveReleaseCandidateV2:
    try:
        base = ProspectiveAdaptiveReleaseCandidate.model_validate(
            base_candidate.model_dump(mode="json")
        )
        semantics = AdaptiveTargetSemanticsBindingV2.model_validate(
            target_semantics.model_dump(mode="json")
        )
        independence = AdaptiveIndependenceIdentityV2.model_validate(
            independence_identity.model_dump(mode="json")
        )
        projection = (
            None
            if condition_projection is None
            else ConditionCalibrationProjectionV1.model_validate(
                condition_projection.model_dump(mode="json")
            )
        )
        invocation = (
            None
            if condition_gate_invocation_proof is None
            else ConditionGateInvocationProofV2.model_validate(
                condition_gate_invocation_proof.model_dump(mode="json")
            )
        )
        qualification = (
            None
            if release_qualification_proof is None
            else ConfirmationAwareReleaseQualificationProofV2.model_validate(
                release_qualification_proof.model_dump(mode="json")
            )
        )
        gate = (
            None
            if terminal_gate_result is None
            else ConditionTerminalGateResultV2.model_validate(
                terminal_gate_result.model_dump(mode="json")
            )
        )
    except (AttributeError, ValueError) as exc:
        raise AdaptiveCalibrationError("confirmation_v2_prospective_input_changed") from exc
    payload: dict[str, Any] = {
        "candidate_version": "prospective-adaptive-release-candidate-v2",
        "base_candidate": base,
        "target_semantics": semantics,
        "target_semantics_sha256": semantics.target_semantics_sha256,
        "independence_identity": independence,
        "independence_identity_sha256": independence.independence_identity_sha256,
        "condition_projection": projection,
        "condition_gate_invocation_proof": invocation,
        "release_qualification_proof": qualification,
        "terminal_gate_result": gate,
    }
    return ProspectiveAdaptiveReleaseCandidateV2.model_validate(
        {**payload, "candidate_sha256": hash_canonical(payload)}
    )


ConfirmationAwareAssessmentReasonV2 = Literal[
    "first_full_release_under_confirmation_aware_frozen_policy",
    "noncalibration_gate_blocked",
    "risk_above_frozen_threshold",
    "policy_abstain_all",
    "simulation_calibration_not_valid_for_scientific_release",
    "strong_independence_unverified",
    "confirmation_aware_bundle_not_real_release_eligible",
    "terminal_condition_confirmation_pending",
    "terminal_condition_release_qualification_missing",
    "terminal_condition_confirmation_missing",
    "terminal_condition_confirmation_not_confirmed",
    "terminal_condition_confirmation_insufficient",
]


class AdaptiveProspectiveAssessmentV2(ContractModel):
    assessment_version: Literal["adaptive-first-release-assessment-v2"] = (
        "adaptive-first-release-assessment-v2"
    )
    question_id: Annotated[str, Field(min_length=1)]
    candidate_sha256: str
    frozen_bundle_sha256: str
    policy_context_sha256: str
    threshold_candidate_sha256: str | None
    scalar_risk_score: Annotated[float, Field(ge=0, le=1)] | None
    threshold: Annotated[float, Field(ge=0, le=1)] | None
    prefix_index: Annotated[int, Field(ge=0)]
    released_claim_decision: AdaptiveClaimDecision | None = None
    terminal_gate_result_sha256: str | None = None
    status: Literal["released", "abstained"]
    reason: ConfirmationAwareAssessmentReasonV2
    guarantee_scope: Literal[
        "exact released-decision mismatch overall and for confirmed condition-dependent "
        "releases within every frozen deployment domain for the frozen joint threshold "
        "and terminal confirmation policy under exchangeable independent complete "
        "questions; not scientific truth, causal proof, or domain-shift robustness"
    ] = (
        "exact released-decision mismatch overall and for confirmed condition-dependent "
        "releases within every frozen deployment domain for the frozen joint threshold "
        "and terminal confirmation policy under exchangeable independent complete "
        "questions; not scientific truth, causal proof, or domain-shift robustness"
    )
    assessment_sha256: str

    @field_validator(
        "candidate_sha256",
        "frozen_bundle_sha256",
        "policy_context_sha256",
        "threshold_candidate_sha256",
        "terminal_gate_result_sha256",
        "assessment_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_assessment(self) -> AdaptiveProspectiveAssessmentV2:
        threshold_fields = (
            self.threshold_candidate_sha256,
            self.scalar_risk_score,
            self.threshold,
        )
        threshold_lineage = all(value is not None for value in threshold_fields)
        no_threshold_lineage = all(value is None for value in threshold_fields)
        if not (threshold_lineage or no_threshold_lineage):
            raise ValueError("confirmation_v2_assessment_threshold_lineage_incomplete")
        released = self.status == "released"
        if released:
            if (
                self.reason != "first_full_release_under_confirmation_aware_frozen_policy"
                or not threshold_lineage
                or self.released_claim_decision is None
            ):
                raise ValueError("confirmation_v2_release_assessment_inconsistent")
            assert self.scalar_risk_score is not None and self.threshold is not None
            if self.scalar_risk_score > self.threshold:
                raise ValueError("confirmation_v2_release_score_above_threshold")
            if (self.released_claim_decision == "condition_dependent") != (
                self.terminal_gate_result_sha256 is not None
            ):
                raise ValueError("confirmation_v2_release_terminal_gate_mismatch")
        elif (
            self.released_claim_decision is not None or self.terminal_gate_result_sha256 is not None
        ):
            raise ValueError("confirmation_v2_abstention_has_release_lineage")
        payload = self.model_dump(mode="json", exclude={"assessment_sha256"})
        if hash_canonical(payload) != self.assessment_sha256:
            raise ValueError("confirmation_v2_prospective_assessment_hash_mismatch")
        return self


def _v2_frozen_visible_rows(
    bundle: AdaptiveCalibrationBundleV2,
) -> list[PolicyVisibleQuestionTrajectoryV2]:
    return [
        *[row.visible for row in bundle.development_freeze.development_trajectories],
        *bundle.development_freeze.calibration_roster.visible_trajectories,
    ]


def _assert_v2_prospective_independence(
    candidate: ProspectiveAdaptiveReleaseCandidateV2,
    bundle: AdaptiveCalibrationBundleV2,
) -> None:
    base = candidate.base_candidate
    frozen_rows = _v2_frozen_visible_rows(bundle)
    frozen_questions = {row.base_visible.question_id for row in frozen_rows}
    if base.question_id in frozen_questions:
        raise AdaptiveCalibrationError("confirmation_v2_prospective_question_overlap")
    frozen_tokens = {
        digest
        for row in frozen_rows
        for digest in row.independence_identity.strong_identity_token_sha256s
    }
    frozen_components = {
        digest
        for row in frozen_rows
        for digest in row.independence_identity.strong_component_sha256s
    }
    token_overlap = sorted(
        frozen_tokens & set(candidate.independence_identity.strong_identity_token_sha256s)
    )
    component_overlap = sorted(
        frozen_components & set(candidate.independence_identity.strong_component_sha256s)
    )
    if token_overlap:
        raise AdaptiveCalibrationError(
            f"confirmation_v2_prospective_strong_token_overlap:{token_overlap}"
        )
    if component_overlap:
        raise AdaptiveCalibrationError(
            f"confirmation_v2_prospective_strong_component_overlap:{component_overlap}"
        )


def assess_confirmation_aware_adaptive_release_candidate(
    candidate: ProspectiveAdaptiveReleaseCandidateV2,
    bundle: AdaptiveCalibrationBundleV2,
) -> AdaptiveProspectiveAssessmentV2:
    """Apply the v2 joint first-release rule to one complete observed prefix."""

    bundle = validate_adaptive_calibration_bundle_v2_integrity(bundle)
    if not isinstance(candidate, ProspectiveAdaptiveReleaseCandidateV2):
        raise AdaptiveCalibrationError("confirmation_v2_prospective_contract_invalid")
    try:
        candidate = ProspectiveAdaptiveReleaseCandidateV2.model_validate(
            candidate.model_dump(mode="json")
        )
    except ValueError as exc:
        raise AdaptiveCalibrationError("confirmation_v2_prospective_integrity_changed") from exc
    base = candidate.base_candidate
    _assert_v2_prospective_independence(candidate, bundle)
    if base.population_id != bundle.population_id:
        raise AdaptiveCalibrationError("confirmation_v2_prospective_population_mismatch")
    if base.domain not in bundle.calibration.domains:
        raise AdaptiveCalibrationError("confirmation_v2_prospective_domain_shift")
    context = next(
        (
            row
            for row in bundle.development_freeze.base_freeze.policy_contexts
            if row.policy_arm_id == base.policy_arm_id
        ),
        None,
    )
    if context is None or context.policy_context_sha256 != base.policy_context_sha256:
        raise AdaptiveCalibrationError("confirmation_v2_prospective_policy_context_mismatch")
    if candidate.condition_projection is not None and (
        candidate.condition_projection.pipeline_sha256 != context.pipeline_sha256
    ):
        raise AdaptiveCalibrationError("confirmation_v2_prospective_pipeline_mismatch")
    invocation = candidate.condition_gate_invocation_proof
    if invocation is not None and not math.isclose(
        invocation.remaining_budget_minutes,
        max(0.0, context.budget_minutes - base.observed_states[-1].audit_prefix_cost_minutes),
        rel_tol=0.0,
        abs_tol=_COST_TOLERANCE,
    ):
        raise AdaptiveCalibrationError("confirmation_v2_prospective_invocation_budget_mismatch")
    if any(
        list(state.score_features) != context.score_feature_names for state in base.observed_states
    ):
        raise AdaptiveCalibrationError("confirmation_v2_prospective_feature_schema_mismatch")
    selected = bundle.selected
    current = base.observed_states[-1]
    prefix_index = current.prefix_index
    threshold_candidate = None if selected is None else selected.candidate
    model = (
        None
        if threshold_candidate is None
        else next(
            row
            for row in bundle.development_freeze.base_freeze.score_models
            if row.policy_arm_id == base.policy_arm_id
        )
    )
    if threshold_candidate is not None and (
        threshold_candidate.policy_arm_id != base.policy_arm_id
        or threshold_candidate.policy_context_sha256 != base.policy_context_sha256
    ):
        raise AdaptiveCalibrationError("confirmation_v2_prospective_policy_context_mismatch")
    scores = (
        []
        if model is None
        else [model.score_features(state.score_features) for state in base.observed_states]
    )
    if threshold_candidate is not None:
        for state, score in zip(
            base.observed_states[:-1],
            scores[:-1],
            strict=True,
        ):
            if (
                state.claim_decision != "condition_dependent"
                and state.non_calibration_gates_passed
                and score <= threshold_candidate.threshold
            ):
                raise AdaptiveCalibrationError(
                    "confirmation_v2_prospective_continued_after_first_release"
                )

    gate_result = candidate.terminal_gate_result
    if selected is None:
        status = "abstained"
        reason: ConfirmationAwareAssessmentReasonV2 = "policy_abstain_all"
        score = None
        threshold = None
        threshold_sha = None
    else:
        assert model is not None and threshold_candidate is not None
        score = scores[-1]
        threshold = threshold_candidate.threshold
        threshold_sha = threshold_candidate.candidate_sha256
        if candidate.independence_identity.verification_status != "verified":
            status = "abstained"
            reason = "strong_independence_unverified"
        elif bundle.label_source == "simulation":
            status = "abstained"
            reason = "simulation_calibration_not_valid_for_scientific_release"
        elif not bundle.real_release_eligible:
            status = "abstained"
            reason = "confirmation_aware_bundle_not_real_release_eligible"
        elif not current.non_calibration_gates_passed:
            status = "abstained"
            reason = "noncalibration_gate_blocked"
        elif score > threshold:
            status = "abstained"
            reason = "risk_above_frozen_threshold"
        elif current.claim_decision == "condition_dependent":
            qualification = candidate.release_qualification_proof
            if invocation is None:
                status = "abstained"
                reason = "terminal_condition_confirmation_pending"
            elif qualification is None:
                status = "abstained"
                reason = "terminal_condition_release_qualification_missing"
            elif (
                qualification.frozen_bundle_sha256 != bundle.bundle_sha256
                or qualification.threshold_candidate != threshold_candidate
                or qualification.scalar_risk_score != score
            ):
                raise AdaptiveCalibrationError(
                    "confirmation_v2_prospective_qualification_bundle_mismatch"
                )
            elif gate_result is None or gate_result.status == "missing":
                status = "abstained"
                reason = "terminal_condition_confirmation_missing"
            elif gate_result.status == "not_confirmed":
                status = "abstained"
                reason = "terminal_condition_confirmation_not_confirmed"
            elif gate_result.status == "insufficient":
                status = "abstained"
                reason = "terminal_condition_confirmation_insufficient"
            else:
                status = "released"
                reason = "first_full_release_under_confirmation_aware_frozen_policy"
        else:
            status = "released"
            reason = "first_full_release_under_confirmation_aware_frozen_policy"
    released = status == "released"
    payload: dict[str, Any] = {
        "assessment_version": "adaptive-first-release-assessment-v2",
        "question_id": base.question_id,
        "candidate_sha256": candidate.candidate_sha256,
        "frozen_bundle_sha256": bundle.bundle_sha256,
        "policy_context_sha256": base.policy_context_sha256,
        "threshold_candidate_sha256": threshold_sha,
        "scalar_risk_score": score,
        "threshold": threshold,
        "prefix_index": prefix_index,
        "released_claim_decision": current.claim_decision if released else None,
        "terminal_gate_result_sha256": (
            gate_result.result_sha256
            if released
            and current.claim_decision == "condition_dependent"
            and gate_result is not None
            else None
        ),
        "status": status,
        "reason": reason,
        "guarantee_scope": bundle.guarantee_scope,
    }
    return AdaptiveProspectiveAssessmentV2.model_validate(
        {**payload, "assessment_sha256": hash_canonical(payload)}
    )


__all__ = [
    "AdaptiveCalibrationBundle",
    "AdaptiveCalibrationBundleV2",
    "AdaptiveCalibrationError",
    "AdaptiveCalibrationPlan",
    "AdaptiveCalibrationPlanV2",
    "AdaptiveCalibrationRoster",
    "AdaptiveCalibrationRosterV2",
    "AdaptiveCandidateCalibration",
    "AdaptiveCandidateCalibrationV2",
    "AdaptiveConditionDomainCalibrationV2",
    "AdaptiveDevelopmentFreeze",
    "AdaptiveDevelopmentFreezeV2",
    "AdaptiveIndependenceIdentityV2",
    "AdaptivePolicyArmTrajectory",
    "AdaptivePolicyContext",
    "AdaptivePreselectionState",
    "AdaptiveProspectiveAssessment",
    "AdaptiveProspectiveAssessmentV2",
    "AdaptiveQuestionOutcome",
    "AdaptiveQuestionOutcomeV2",
    "AdaptiveSplitIdentity",
    "AdaptiveSplitIdentityV2",
    "AdaptiveTargetSemanticsBindingV2",
    "AdaptiveTerminalAuditCandidate",
    "AdaptiveTerminalSchedulerProof",
    "AdaptiveThresholdCandidate",
    "AdaptiveThresholdFamily",
    "AdaptiveTrajectoryScoreModel",
    "CompleteCorpusIdentity",
    "ConditionCalibrationCollectionSourceAnchorV1",
    "ConditionCalibrationCollectionSourceRosterV1",
    "ConditionCalibrationGateResultV1",
    "ConditionCalibrationProjectionV1",
    "ConditionConfirmationGateAssessmentV1",
    "ConditionGateInvocationProofV2",
    "ConditionOutcomeFirewallReceiptV1",
    "ConditionTerminalGateResultV2",
    "ConfirmationAwareArmTrajectoryV2",
    "ConfirmationAwareReleaseQualificationProofV2",
    "GateCompleteCalibrationRosterV2",
    "GateCompleteLabeledQuestionTrajectoryV2",
    "GateCompleteQuestionTrajectoryV2",
    "LabeledQuestionTrajectory",
    "LabeledQuestionTrajectoryV2",
    "PolicyVisibleQuestionTrajectory",
    "PolicyVisibleQuestionTrajectoryV2",
    "ProspectiveAdaptiveReleaseCandidate",
    "ProspectiveAdaptiveReleaseCandidateV2",
    "QuestionReferenceVerdict",
    "QuestionReferenceVerdictV2",
    "adaptive_independence_identity_from_condition_plan_v1",
    "assess_adaptive_release_candidate",
    "assess_confirmation_aware_adaptive_release_candidate",
    "calibrate_adaptive_first_release",
    "calibrate_confirmation_aware_first_release",
    "complete_corpus_identity_from_certificate_v5",
    "condition_terminal_gate_result_from_certificate_v6",
    "fit_adaptive_development",
    "fit_adaptive_development_v2",
    "freeze_adaptive_independence_identity_v2",
    "freeze_adaptive_policy_arm_trajectory",
    "freeze_adaptive_policy_context",
    "freeze_adaptive_preselection_state",
    "freeze_adaptive_target_semantics_v2",
    "freeze_adaptive_terminal_scheduler_proof",
    "freeze_complete_corpus_identity",
    "freeze_condition_calibration_collection_source_roster_v1",
    "freeze_condition_calibration_gate_result_v1",
    "freeze_condition_calibration_projection",
    "freeze_condition_confirmation_gate_assessment",
    "freeze_condition_gate_invocation_proof_v2",
    "freeze_condition_outcome_firewall_receipt",
    "freeze_condition_terminal_gate_result_v2",
    "freeze_confirmation_aware_arm_trajectory",
    "freeze_confirmation_aware_release_qualification_proof_v2",
    "freeze_gate_complete_calibration_roster_v2",
    "freeze_policy_visible_question_trajectory",
    "freeze_policy_visible_question_trajectory_v2",
    "freeze_preselection_state_from_production_components",
    "freeze_prospective_adaptive_candidate",
    "freeze_prospective_adaptive_candidate_v2",
    "freeze_question_reference_verdict",
    "freeze_question_reference_verdict_v2",
    "hash_strong_independence_identifier_v2",
    "join_condition_calibration_assessment_receipts",
    "join_gate_complete_labeled_question_trajectory_v2",
    "join_labeled_question_trajectory",
    "join_labeled_question_trajectory_v2",
    "join_terminal_condition_gates",
    "noncalibration_assessment_sha256",
    "policy_visible_trajectory_from_certificate_v5_sequence",
    "preselection_state_from_certificate_v5",
    "score_labeled_question_trajectory",
    "validate_adaptive_calibration_bundle_integrity",
    "validate_adaptive_calibration_bundle_v2_integrity",
]
