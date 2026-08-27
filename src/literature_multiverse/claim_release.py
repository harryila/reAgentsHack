"""Fail-closed prospective release boundary for directional scientific claims.

The orchestrator joins four independently auditable contracts: the typed evidence
graph, conservative synthesis, budgeted human verification, and a frozen selective-
risk policy.  It accepts no correctness label or audit oracle.  A ``released`` result
means only that every declared gate permitted the *targeted directional statement*
under the frozen pipeline and population; it is not a guarantee of scientific truth.

Only ``increase`` and ``decrease`` are valid targets.  Non-significance, a legacy
``no_effect`` label, and a confidence interval containing the null can therefore never
be promoted to equivalence or to a directional release.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.budgeted_verification import (
    AllocationPolicy,
    AuditCandidate,
    ClaimModel,
    ReleaseGuardConfig,
    ReleaseGuardStatus,
    assess_prospective_release_guard,
    select_under_budget,
)
from literature_multiverse.calibration import (
    CalibrationContractError,
    FrozenCalibrationBundle,
    ProspectiveReleaseAssessment,
    ReleaseCandidate,
    assess_release_candidate,
    validate_frozen_calibration_bundle_integrity,
)
from literature_multiverse.evidence_graph import (
    EvidenceGraph,
    graph_risk_features,
    select_effect_evidence,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.meta_analysis import synthesize_evidence_graph
from literature_multiverse.models import SHA256_RE, ContractModel


class ClaimReleaseContractError(ValueError):
    """Inputs cannot be joined without weakening a release-gate contract."""


class TargetDirection(StrEnum):
    """The only directional claims this boundary is permitted to assess."""

    INCREASE = "increase"
    DECREASE = "decrease"


class EvidenceClassification(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    CONDITION_DEPENDENT = "condition_dependent"
    INCONCLUSIVE = "inconclusive"
    NOT_EVALUABLE = "not_evaluable"


class ClaimReleaseStatus(StrEnum):
    RELEASED = "released"
    ABSTAINED = "abstained"


class AuditResolutionProvenance(StrEnum):
    """Allowed external sources for a completed audit-resolution receipt."""

    BLINDED_HUMAN = "blinded_human"
    BENCHMARK_ADJUDICATION = "benchmark_adjudication"


class ClaimTarget(ContractModel):
    """A prespecified, oriented claim; equivalence/no-effect is intentionally absent."""

    direction: TargetDirection
    outcome_name: Annotated[str, Field(min_length=1)]
    contrast_id: Annotated[str, Field(min_length=1)] | None = None


class ClaimReleaseConfig(ContractModel):
    """Frozen synthesis behavior used before observing any correctness label."""

    config_version: Literal["claim-release-v1"] = "claim-release-v1"
    require_explicit_timepoint: bool = True
    require_prediction_interval_stability: bool = True
    confidence_level: Annotated[float, Field(gt=0, lt=1)] = 0.95
    assumed_within_paper_correlation: Annotated[float, Field(ge=0, le=1)] = 1.0
    audit_allocation_policy: AllocationPolicy = (
        AllocationPolicy.EXPECTED_CLAIM_LOSS_PER_COST
    )
    audit_seed: int = 0
    prespecified_condition_moderators: list[str] = Field(default_factory=list)
    condition_familywise_alpha: Annotated[float, Field(gt=0, lt=1)] = 0.05
    condition_min_papers_per_level: Annotated[int, Field(ge=2)] = 2

    @field_validator("prespecified_condition_moderators")
    @classmethod
    def validate_condition_moderators(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("condition_moderator_name_empty")
        if value != sorted(set(value)):
            raise ValueError("condition_moderators_must_be_sorted_unique")
        return value


class SynthesisEvidenceAssessment(ContractModel):
    """Conservative interpretation of one synthesis output for one target direction."""

    target_direction: TargetDirection
    classification: EvidenceClassification
    mode: str
    reason: str
    n_papers: Annotated[int, Field(ge=0)]
    estimate: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    prediction_interval_lower: float | None = None
    prediction_interval_upper: float | None = None
    directional_increase_fraction: Annotated[float, Field(ge=0, le=1)] | None = None
    directional_ci_lower: Annotated[float, Field(ge=0, le=1)] | None = None
    directional_ci_upper: Annotated[float, Field(ge=0, le=1)] | None = None
    condition_moderators: list[str] = Field(default_factory=list)
    condition_interpretation: str | None = None

    @model_validator(mode="after")
    def validate_intervals(self) -> SynthesisEvidenceAssessment:
        for lower_name, upper_name in (
            ("ci_lower", "ci_upper"),
            ("prediction_interval_lower", "prediction_interval_upper"),
            ("directional_ci_lower", "directional_ci_upper"),
        ):
            lower = getattr(self, lower_name)
            upper = getattr(self, upper_name)
            if (lower is None) != (upper is None):
                raise ValueError(f"claim_release_interval_incomplete:{lower_name}")
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"claim_release_interval_not_ordered:{lower_name}")
        numeric = (
            self.estimate,
            self.ci_lower,
            self.ci_upper,
            self.prediction_interval_lower,
            self.prediction_interval_upper,
            self.directional_increase_fraction,
            self.directional_ci_lower,
            self.directional_ci_upper,
        )
        if any(value is not None and not math.isfinite(value) for value in numeric):
            raise ValueError("claim_release_evidence_value_nonfinite")
        if self.condition_moderators != sorted(set(self.condition_moderators)):
            raise ValueError("claim_release_condition_moderators_not_sorted_unique")
        if self.classification is EvidenceClassification.CONDITION_DEPENDENT:
            if not self.condition_moderators or self.condition_interpretation is None:
                raise ValueError("condition_dependent_requires_moderator_evidence")
        elif self.condition_moderators or self.condition_interpretation is not None:
            raise ValueError("condition_metadata_requires_condition_dependent_classification")
        return self


class AuditResolutionReceipt(ContractModel):
    """Hash-bound declaration that one estimate was externally adjudicated.

    The receipt deliberately contains no correctness label or corrected value.  Its
    artifact hashes make the external adjudication and correction lineage auditable,
    while the current hashes prevent a receipt for one graph/synthesis snapshot from
    being replayed against another.  A valid receipt is still a declaration: hashes and
    provenance metadata do not prove that an adjudicator was competent.
    """

    receipt_version: Literal["audit-resolution-v1"] = "audit-resolution-v1"
    item_id: Annotated[str, Field(min_length=1)]
    provenance: AuditResolutionProvenance
    adjudicator_count: Annotated[int, Field(ge=1)]
    completed_at: datetime
    adjudication_protocol_sha256: str
    adjudication_artifact_sha256: str
    audited_evidence_item_sha256: str
    audited_graph_sha256: str
    audited_synthesis_sha256: str
    current_evidence_item_sha256: str
    current_graph_sha256: str
    current_synthesis_sha256: str
    current_candidate_input_sha256: str
    correction_lineage_sha256: str | None = None
    completion_status: Literal["complete"] = "complete"
    competence_semantics: Literal[
        "auditable provenance declaration; not proof of adjudicator competence"
    ] = "auditable provenance declaration; not proof of adjudicator competence"
    receipt_sha256: str

    @field_validator(
        "adjudication_protocol_sha256",
        "adjudication_artifact_sha256",
        "audited_evidence_item_sha256",
        "audited_graph_sha256",
        "audited_synthesis_sha256",
        "current_evidence_item_sha256",
        "current_graph_sha256",
        "current_synthesis_sha256",
        "current_candidate_input_sha256",
        "correction_lineage_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("invalid_audit_resolution_sha256")
        return value

    @field_validator("completed_at")
    @classmethod
    def validate_completed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("audit_resolution_completed_at_requires_timezone")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> AuditResolutionReceipt:
        if (
            self.provenance is AuditResolutionProvenance.BLINDED_HUMAN
            and self.adjudicator_count < 2
        ):
            raise ValueError("blinded_human_resolution_requires_two_adjudicators")
        snapshot_changed = any(
            audited != current
            for audited, current in (
                (self.audited_evidence_item_sha256, self.current_evidence_item_sha256),
                (self.audited_graph_sha256, self.current_graph_sha256),
                (self.audited_synthesis_sha256, self.current_synthesis_sha256),
            )
        )
        if snapshot_changed != (self.correction_lineage_sha256 is not None):
            raise ValueError("audit_resolution_correction_lineage_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("audit_resolution_receipt_hash_mismatch")
        return self


def freeze_audit_resolution_receipt(
    *,
    item_id: str,
    provenance: AuditResolutionProvenance,
    adjudicator_count: int,
    completed_at: datetime,
    adjudication_protocol_sha256: str,
    adjudication_artifact_sha256: str,
    audited_evidence_item_sha256: str,
    audited_graph_sha256: str,
    audited_synthesis_sha256: str,
    current_evidence_item_sha256: str,
    current_graph_sha256: str,
    current_synthesis_sha256: str,
    current_candidate_input_sha256: str,
    correction_lineage_sha256: str | None = None,
) -> AuditResolutionReceipt:
    """Freeze a self-hashed receipt from externally verified adjudication hashes."""

    completed_at_json = completed_at.isoformat()
    if completed_at_json.endswith("+00:00"):
        completed_at_json = f"{completed_at_json[:-6]}Z"
    payload: dict[str, Any] = {
        "receipt_version": "audit-resolution-v1",
        "item_id": item_id,
        "provenance": provenance,
        "adjudicator_count": adjudicator_count,
        "completed_at": completed_at_json,
        "adjudication_protocol_sha256": adjudication_protocol_sha256,
        "adjudication_artifact_sha256": adjudication_artifact_sha256,
        "audited_evidence_item_sha256": audited_evidence_item_sha256,
        "audited_graph_sha256": audited_graph_sha256,
        "audited_synthesis_sha256": audited_synthesis_sha256,
        "current_evidence_item_sha256": current_evidence_item_sha256,
        "current_graph_sha256": current_graph_sha256,
        "current_synthesis_sha256": current_synthesis_sha256,
        "current_candidate_input_sha256": current_candidate_input_sha256,
        "correction_lineage_sha256": correction_lineage_sha256,
        "completion_status": "complete",
        "competence_semantics": (
            "auditable provenance declaration; not proof of adjudicator competence"
        ),
    }
    return AuditResolutionReceipt.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


class AuditPrioritySummary(ContractModel):
    """Policy-visible ranking row; it contains no audit outcome or gold label."""

    item_id: str
    rank: Annotated[int, Field(ge=1)]
    selected_for_audit: bool
    resolved_before_release: bool
    probability_basis: str
    probability_influence: Annotated[float, Field(ge=0, le=1)]
    conclusion_flip: bool
    expected_claim_loss_reduction: Annotated[float, Field(ge=0)]
    expected_claim_loss_reduction_per_cost: Annotated[float, Field(ge=0)]
    decision_score_source: str | None = None


class AuditGateAssessment(ContractModel):
    """Budget, adjudication, and bounded residual-risk state kept distinct."""

    status: Literal["eligible", "blocked", "not_applicable"]
    reasons: list[str]
    expected_item_ids: list[str]
    candidate_item_ids: list[str]
    selected_item_ids: list[str]
    resolved_item_ids: list[str]
    unresolved_item_ids: list[str]
    budget: Annotated[float, Field(ge=0)]
    spent: Annotated[float, Field(ge=0)]
    cost_unit: str | None
    unresolved_expected_claim_loss: Annotated[float, Field(ge=0)]
    unresolved_conclusion_flip_item_ids: list[str]
    unresolved_high_influence_item_ids: list[str]
    unresolved_noncalibrated_item_ids: list[str]
    unresolved_without_probability_bound_item_ids: list[str]
    residual_decision_risk_upper_bound: Annotated[float, Field(ge=0, le=1)] | None
    residual_risk_bound_assumptions: list[str]
    ranking: list[AuditPrioritySummary]
    resolution_receipts: list[AuditResolutionReceipt]
    candidate_input_sha256: str
    resolution_ledger_sha256: str | None
    selection_sha256: str | None
    guard_sha256: str | None

    @field_validator(
        "candidate_input_sha256",
        "resolution_ledger_sha256",
        "selection_sha256",
        "guard_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("invalid_audit_gate_sha256")
        return value

    @model_validator(mode="after")
    def validate_identities(self) -> AuditGateAssessment:
        for name in (
            "expected_item_ids",
            "candidate_item_ids",
            "selected_item_ids",
            "resolved_item_ids",
            "unresolved_item_ids",
            "unresolved_conclusion_flip_item_ids",
            "unresolved_high_influence_item_ids",
            "unresolved_noncalibrated_item_ids",
            "unresolved_without_probability_bound_item_ids",
        ):
            values = getattr(self, name)
            if values != sorted(set(values)):
                raise ValueError(f"claim_release_audit_ids_not_sorted_unique:{name}")
        if self.expected_item_ids != self.candidate_item_ids:
            raise ValueError("claim_release_audit_identity_coverage_mismatch")
        if set(self.selected_item_ids) - set(self.candidate_item_ids):
            raise ValueError("claim_release_selected_audit_identity_unknown")
        if set(self.resolved_item_ids) - set(self.candidate_item_ids):
            raise ValueError("claim_release_resolved_audit_identity_unknown")
        receipt_ids = [receipt.item_id for receipt in self.resolution_receipts]
        if receipt_ids != sorted(set(receipt_ids)):
            raise ValueError("claim_release_resolution_receipt_ids_not_sorted_unique")
        if receipt_ids != self.resolved_item_ids:
            raise ValueError("claim_release_resolution_receipt_identity_mismatch")
        if set(self.unresolved_item_ids) != (
            set(self.candidate_item_ids) - set(self.resolved_item_ids)
        ):
            raise ValueError("claim_release_unresolved_audit_identity_mismatch")
        if self.status == "not_applicable" and self.candidate_item_ids:
            raise ValueError("claim_release_nonempty_audit_cannot_be_not_applicable")
        expected_ledger_hash = (
            hash_canonical(self.resolution_receipts) if self.resolution_receipts else None
        )
        if self.resolution_ledger_sha256 != expected_ledger_hash:
            raise ValueError("claim_release_resolution_ledger_hash_mismatch")
        return self


class CalibrationGateAssessment(ContractModel):
    """Deployment-side score result from an already frozen label-risk policy."""

    status: Literal["released", "abstained", "not_run"]
    reason: str
    frozen_bundle_sha256: str | None = None
    release_candidate_sha256: str | None = None
    prospective_assessment_sha256: str | None = None
    scalar_risk_score: Annotated[float, Field(ge=0, le=1)] | None = None
    threshold: Annotated[float, Field(ge=0, le=1)] | None = None
    label_source: Literal[
        "benchmark_annotation", "expert_adjudication", "simulation"
    ] | None = None
    guarantee_scope: Literal[
        "frozen label-risk policy under the declared population and pipeline; not scientific truth"
    ] = "frozen label-risk policy under the declared population and pipeline; not scientific truth"

    @field_validator(
        "frozen_bundle_sha256",
        "release_candidate_sha256",
        "prospective_assessment_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("invalid_claim_calibration_sha256")
        return value


class ClaimReleaseAssessment(ContractModel):
    """Hash-bound final decision; every gate must pass for ``released``."""

    assessment_version: Literal["prospective-claim-release-v1"] = (
        "prospective-claim-release-v1"
    )
    question_id: str
    target: ClaimTarget
    pipeline_sha256: str
    evidence_graph_sha256: str
    synthesis_sha256: str
    config_sha256: str
    paper_ids: list[str]
    evidence: SynthesisEvidenceAssessment
    audit: AuditGateAssessment
    risk_feature_schema_version: Literal["claim-release-risk-v1"] = (
        "claim-release-risk-v1"
    )
    risk_features: dict[str, float]
    risk_features_sha256: str
    calibration: CalibrationGateAssessment
    status: ClaimReleaseStatus
    reasons: list[str]
    decision_sha256: str
    release_semantics: Literal[
        "all declared prospective gates passed; not a guarantee of scientific truth"
    ] = "all declared prospective gates passed; not a guarantee of scientific truth"

    @field_validator(
        "pipeline_sha256",
        "evidence_graph_sha256",
        "synthesis_sha256",
        "config_sha256",
        "risk_features_sha256",
        "decision_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("invalid_claim_release_sha256")
        return value

    @field_validator("paper_ids")
    @classmethod
    def validate_paper_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("claim_release_paper_ids_not_sorted_unique")
        return value

    @field_validator("risk_features")
    @classmethod
    def validate_risk_features(cls, value: dict[str, float]) -> dict[str, float]:
        if list(value) != list(CLAIM_RELEASE_RISK_FEATURE_NAMES):
            raise ValueError("claim_release_risk_feature_schema_mismatch")
        if any(not math.isfinite(number) for number in value.values()):
            raise ValueError("claim_release_risk_feature_nonfinite")
        return value

    @model_validator(mode="after")
    def validate_release(self) -> ClaimReleaseAssessment:
        gates_pass = (
            self.evidence.classification is EvidenceClassification.SUPPORTED
            and self.audit.status == "eligible"
            and self.calibration.status == "released"
        )
        if (self.status is ClaimReleaseStatus.RELEASED) != gates_pass:
            raise ValueError("claim_release_status_gate_mismatch")
        if self.status is ClaimReleaseStatus.RELEASED and self.reasons:
            raise ValueError("released_claim_cannot_have_blocking_reasons")
        if self.status is ClaimReleaseStatus.ABSTAINED and not self.reasons:
            raise ValueError("abstained_claim_requires_reason")
        if self.reasons != list(dict.fromkeys(self.reasons)):
            raise ValueError("claim_release_reasons_not_unique")
        if hash_canonical(self.risk_features) != self.risk_features_sha256:
            raise ValueError("claim_release_risk_feature_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"decision_sha256"})
        if hash_canonical(payload) != self.decision_sha256:
            raise ValueError("claim_release_decision_hash_mismatch")
        return self


_GRAPH_FEATURE_NAMES = (
    "fraction_high_or_critical_risk_of_bias",
    "fraction_missing_source_quote",
    "fraction_non_estimable",
    "fraction_risk_of_bias_not_assessed",
    "fraction_timepoint_not_reported",
    "fraction_unresolved_cohort_identity",
    "n_cohorts",
    "n_estimates",
    "n_publications",
)

CLAIM_RELEASE_RISK_FEATURE_NAMES = tuple(
    sorted(
        {
            *(f"graph_{name}" for name in _GRAPH_FEATURE_NAMES),
            "synthesis_quantitative_available",
            "synthesis_directional_available",
            "synthesis_n_papers",
            "synthesis_ci_width",
            "synthesis_ci_crosses_null",
            "synthesis_prediction_interval_available",
            "synthesis_prediction_interval_width",
            "synthesis_prediction_interval_crosses_null",
            "synthesis_prediction_interval_target_stable",
            "synthesis_directional_ci_width",
            "synthesis_directional_ci_crosses_half",
            "synthesis_i_squared",
            "audit_candidate_count",
            "audit_selected_fraction",
            "audit_resolved_fraction",
            "audit_unresolved_fraction",
            "audit_unresolved_expected_claim_loss",
            "audit_max_unresolved_influence",
            "audit_unresolved_flip_fraction",
            "audit_unresolved_noncalibrated_fraction",
            "audit_unresolved_without_probability_bound_fraction",
            "audit_residual_decision_risk_upper_bound_available",
            "audit_residual_decision_risk_upper_bound",
        }
    )
)


def _target_side(lower: float, upper: float, direction: TargetDirection) -> bool:
    return lower > 0 if direction is TargetDirection.INCREASE else upper < 0


def _opposite_side(lower: float, upper: float, direction: TargetDirection) -> bool:
    return upper < 0 if direction is TargetDirection.INCREASE else lower > 0


def _synthesis_reason(synthesis: Mapping[str, Any]) -> str:
    graph_contract = synthesis.get("evidence_graph")
    if isinstance(graph_contract, Mapping) and graph_contract.get("selection_reason"):
        return str(graph_contract["selection_reason"])
    directional = synthesis.get("directional_fallback")
    if isinstance(directional, Mapping) and directional.get("reason"):
        return str(directional["reason"])
    quantitative = synthesis.get("quantitative")
    if isinstance(quantitative, Mapping) and quantitative.get("reason"):
        return str(quantitative["reason"])
    return "synthesis_insufficient"


def _classify_synthesis(
    synthesis: Mapping[str, Any],
    *,
    target: ClaimTarget,
    config: ClaimReleaseConfig,
) -> SynthesisEvidenceAssessment:
    mode = str(synthesis.get("mode", "unknown"))
    if synthesis.get("status") != "ok":
        return SynthesisEvidenceAssessment(
            target_direction=target.direction,
            classification=EvidenceClassification.NOT_EVALUABLE,
            mode=mode,
            reason=_synthesis_reason(synthesis),
            n_papers=0,
        )

    condition = synthesis.get("condition_analysis")
    qualifying_moderators: list[str] = []
    condition_interpretation: str | None = None
    if isinstance(condition, Mapping) and condition.get("status") == "condition_dependent":
        raw_qualifying = condition.get("qualifying_moderators")
        if not isinstance(raw_qualifying, list):
            raise ClaimReleaseContractError("condition_analysis_qualifiers_invalid")
        qualifying_moderators = sorted(
            str(row["moderator"])
            for row in raw_qualifying
            if isinstance(row, Mapping) and row.get("moderator")
        )
        if not qualifying_moderators:
            raise ClaimReleaseContractError("condition_analysis_qualifiers_empty")
        condition_interpretation = str(
            condition.get("interpretation", "predictive_association_not_causal")
        )

    if mode == "random_effects_meta_analysis":
        quantitative = synthesis.get("quantitative")
        if not isinstance(quantitative, Mapping) or quantitative.get("status") != "ok":
            raise ClaimReleaseContractError("quantitative_synthesis_payload_invalid")
        estimate = float(quantitative["estimate"])
        lower = float(quantitative["ci_lower"])
        upper = float(quantitative["ci_upper"])
        n_papers = int(quantitative["n_papers"])
        prediction = quantitative.get("prediction_interval")
        prediction_lower: float | None = None
        prediction_upper: float | None = None
        prediction_ok = isinstance(prediction, Mapping) and prediction.get("status") == "ok"
        if prediction_ok:
            prediction_lower = float(prediction["lower"])
            prediction_upper = float(prediction["upper"])

        if qualifying_moderators:
            classification = EvidenceClassification.CONDITION_DEPENDENT
            reason = "prespecified_qualitative_condition_dependence_detected"
        elif _opposite_side(lower, upper, target.direction):
            classification = EvidenceClassification.CONTRADICTED
            reason = "confidence_interval_supports_opposite_direction"
        elif not _target_side(lower, upper, target.direction):
            classification = EvidenceClassification.INCONCLUSIVE
            reason = "confidence_interval_includes_null"
        elif config.require_prediction_interval_stability and not prediction_ok:
            classification = EvidenceClassification.INCONCLUSIVE
            reason = "prediction_interval_required_but_unavailable"
        elif (
            config.require_prediction_interval_stability
            and prediction_lower is not None
            and prediction_upper is not None
            and not _target_side(prediction_lower, prediction_upper, target.direction)
        ):
            classification = EvidenceClassification.INCONCLUSIVE
            reason = "prediction_interval_not_stable_in_target_direction"
        else:
            classification = EvidenceClassification.SUPPORTED
            reason = "confidence_interval_and_required_prediction_interval_support_target"

        return SynthesisEvidenceAssessment(
            target_direction=target.direction,
            classification=classification,
            mode=mode,
            reason=reason,
            n_papers=n_papers,
            estimate=estimate,
            ci_lower=lower,
            ci_upper=upper,
            prediction_interval_lower=prediction_lower,
            prediction_interval_upper=prediction_upper,
            condition_moderators=qualifying_moderators,
            condition_interpretation=condition_interpretation,
        )

    if mode == "directional_sign_synthesis":
        directional = synthesis.get("directional_fallback")
        if not isinstance(directional, Mapping) or directional.get("status") != "ok":
            raise ClaimReleaseContractError("directional_synthesis_payload_invalid")
        interval = directional.get("increase_fraction_exact_ci_95")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ClaimReleaseContractError("directional_synthesis_interval_invalid")
        lower, upper = (float(interval[0]), float(interval[1]))
        fraction = float(directional["increase_fraction_among_nonzero"])
        n_papers = int(directional["n_papers"])
        target_supported = (
            lower > 0.5 if target.direction is TargetDirection.INCREASE else upper < 0.5
        )
        target_contradicted = (
            upper < 0.5 if target.direction is TargetDirection.INCREASE else lower > 0.5
        )
        if qualifying_moderators:
            classification = EvidenceClassification.CONDITION_DEPENDENT
            reason = "prespecified_qualitative_condition_dependence_detected"
        elif target_contradicted:
            classification = EvidenceClassification.CONTRADICTED
            reason = "exact_sign_interval_supports_opposite_direction"
        elif not target_supported:
            classification = EvidenceClassification.INCONCLUSIVE
            reason = "exact_sign_interval_includes_equal_direction_frequency"
        elif config.require_prediction_interval_stability:
            classification = EvidenceClassification.INCONCLUSIVE
            reason = "directional_fallback_cannot_satisfy_prediction_interval_requirement"
        else:
            classification = EvidenceClassification.SUPPORTED
            reason = "exact_sign_interval_supports_target_direction"
        return SynthesisEvidenceAssessment(
            target_direction=target.direction,
            classification=classification,
            mode=mode,
            reason=reason,
            n_papers=n_papers,
            directional_increase_fraction=fraction,
            directional_ci_lower=lower,
            directional_ci_upper=upper,
            condition_moderators=qualifying_moderators,
            condition_interpretation=condition_interpretation,
        )

    raise ClaimReleaseContractError(f"unknown_successful_synthesis_mode:{mode}")


def _audit_candidate_payload(candidate: AuditCandidate) -> dict[str, Any]:
    return asdict(candidate)


def evidence_item_sha256s(graph: EvidenceGraph) -> dict[str, str]:
    """Hash each estimate together with the exact source spans it cites."""

    span_by_id = {span.span_id: span for span in graph.evidence_spans}
    return {
        estimate.estimate_id: hash_canonical(
            {
                "outcome_estimate": estimate,
                "evidence_spans": [
                    span_by_id[span_id] for span_id in sorted(estimate.evidence_span_ids)
                ],
            }
        )
        for estimate in graph.outcome_estimates
    }


def _claim_model_payload(claim_model: ClaimModel) -> dict[str, Any]:
    return asdict(claim_model)


def _validate_exact_audit_coverage(
    *, expected_item_ids: Sequence[str], candidates: Sequence[AuditCandidate]
) -> list[str]:
    if any(not isinstance(candidate, AuditCandidate) for candidate in candidates):
        raise ClaimReleaseContractError("audit_input_must_contain_policy_visible_candidates_only")
    candidate_ids = [candidate.item_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ClaimReleaseContractError("audit_candidate_id_duplicate")
    expected = sorted(expected_item_ids)
    observed = sorted(candidate_ids)
    if expected != observed:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ClaimReleaseContractError(
            f"audit_identity_coverage_mismatch:missing={missing}:extra={extra}"
        )
    return observed


def _audit_gate(
    *,
    expected_item_ids: Sequence[str],
    candidates: Sequence[AuditCandidate],
    claim_model: ClaimModel,
    resolution_receipts: Sequence[AuditResolutionReceipt],
    evidence_item_sha256s: Mapping[str, str],
    evidence_graph_sha256: str,
    synthesis_sha256: str,
    budget: float,
    config: ClaimReleaseConfig,
    guard_config: ReleaseGuardConfig,
    identity_context: Mapping[str, Any],
) -> AuditGateAssessment:
    candidate_ids = _validate_exact_audit_coverage(
        expected_item_ids=expected_item_ids,
        candidates=candidates,
    )
    if any(not isinstance(receipt, AuditResolutionReceipt) for receipt in resolution_receipts):
        raise ClaimReleaseContractError("audit_resolution_receipt_contract_invalid")
    validated_receipts: list[AuditResolutionReceipt] = []
    for receipt in resolution_receipts:
        try:
            validated_receipts.append(
                AuditResolutionReceipt.model_validate(receipt.model_dump(mode="json"))
            )
        except ValueError as exc:
            raise ClaimReleaseContractError(
                f"audit_resolution_receipt_integrity_changed:{receipt.item_id}"
            ) from exc
    validated_receipts.sort(key=lambda receipt: receipt.item_id)
    resolved = [receipt.item_id for receipt in validated_receipts]
    if len(resolved) != len(set(resolved)):
        raise ClaimReleaseContractError("audit_resolution_receipt_id_duplicate")
    unknown_resolved = sorted(set(resolved) - set(candidate_ids))
    if unknown_resolved:
        raise ClaimReleaseContractError(
            f"audit_resolution_receipt_identity_unknown:{unknown_resolved}"
        )
    ordered_candidates = sorted(candidates, key=lambda candidate: candidate.item_id)
    candidate_payload = [
        _audit_candidate_payload(candidate) for candidate in ordered_candidates
    ]
    candidate_hash = hash_canonical(
        {
            "identity_context": identity_context,
            "claim_model": _claim_model_payload(claim_model),
            "candidates": candidate_payload,
        }
    )
    for receipt in validated_receipts:
        expected_evidence_hash = evidence_item_sha256s[receipt.item_id]
        if receipt.current_evidence_item_sha256 != expected_evidence_hash:
            raise ClaimReleaseContractError(
                f"audit_resolution_current_evidence_mismatch:{receipt.item_id}"
            )
        if receipt.current_graph_sha256 != evidence_graph_sha256:
            raise ClaimReleaseContractError(
                f"audit_resolution_current_graph_mismatch:{receipt.item_id}"
            )
        if receipt.current_synthesis_sha256 != synthesis_sha256:
            raise ClaimReleaseContractError(
                f"audit_resolution_current_synthesis_mismatch:{receipt.item_id}"
            )
        if receipt.current_candidate_input_sha256 != candidate_hash:
            raise ClaimReleaseContractError(
                f"audit_resolution_current_candidate_mismatch:{receipt.item_id}"
            )
    resolution_hash = (
        hash_canonical(validated_receipts) if validated_receipts else None
    )
    if not candidates:
        return AuditGateAssessment(
            status="not_applicable",
            reasons=["no_matching_audit_items"],
            expected_item_ids=[],
            candidate_item_ids=[],
            selected_item_ids=[],
            resolved_item_ids=[],
            unresolved_item_ids=[],
            budget=budget,
            spent=0,
            cost_unit=None,
            unresolved_expected_claim_loss=0,
            unresolved_conclusion_flip_item_ids=[],
            unresolved_high_influence_item_ids=[],
            unresolved_noncalibrated_item_ids=[],
            unresolved_without_probability_bound_item_ids=[],
            residual_decision_risk_upper_bound=0,
            residual_risk_bound_assumptions=[
                "no_unresolved_evidence_items",
            ],
            ranking=[],
            resolution_receipts=[],
            candidate_input_sha256=candidate_hash,
            resolution_ledger_sha256=resolution_hash,
            selection_sha256=None,
            guard_sha256=None,
        )

    selection = select_under_budget(
        ordered_candidates,
        claim_model,
        config.audit_allocation_policy,
        budget=budget,
        seed=config.audit_seed,
    )
    guard = assess_prospective_release_guard(
        ordered_candidates,
        claim_model,
        resolved_item_ids=resolved,
        config=guard_config,
    )
    selected = sorted(selection.selected_item_ids)
    unresolved = sorted(guard.unresolved_item_ids)
    ranking = [
        AuditPrioritySummary(
            item_id=row.item_id,
            rank=row.rank,
            selected_for_audit=row.item_id in selected,
            resolved_before_release=row.item_id in resolved,
            probability_basis=row.probability_basis.value,
            probability_influence=row.probability_influence,
            conclusion_flip=row.conclusion_flip,
            expected_claim_loss_reduction=row.expected_claim_loss_reduction,
            expected_claim_loss_reduction_per_cost=(
                row.expected_claim_loss_reduction_per_cost
            ),
            decision_score_source=row.decision_score_source,
        )
        for row in selection.ranking
    ]
    selection_payload = {
        "policy": selection.policy.value,
        "budget": selection.budget,
        "spent": selection.spent,
        "cost_unit": selection.cost_unit,
        "selected_item_ids": list(selection.selected_item_ids),
        "ranking": [row.model_dump(mode="json") for row in ranking],
    }
    guard_payload = {
        "status": guard.status.value,
        "reasons": list(guard.reasons),
        "resolved_item_ids": list(guard.resolved_item_ids),
        "unresolved_item_ids": list(guard.unresolved_item_ids),
        "unresolved_conclusion_flip_item_ids": list(
            guard.unresolved_conclusion_flip_item_ids
        ),
        "unresolved_high_influence_item_ids": list(
            guard.unresolved_high_influence_item_ids
        ),
        "unresolved_noncalibrated_item_ids": list(
            guard.unresolved_noncalibrated_item_ids
        ),
        "unresolved_without_probability_bound_item_ids": list(
            guard.unresolved_without_probability_bound_item_ids
        ),
        "unresolved_expected_claim_loss": guard.unresolved_expected_claim_loss,
        "residual_decision_risk_upper_bound": (
            guard.residual_decision_risk_upper_bound
        ),
        "residual_risk_bound_assumptions": list(
            guard.residual_risk_bound_assumptions
        ),
        "config": asdict(guard.config),
    }
    joined_reasons = list(guard.reasons)
    return AuditGateAssessment(
        status=(
            "eligible"
            if guard.status is ReleaseGuardStatus.ELIGIBLE_FOR_DOWNSTREAM_GATES
            else "blocked"
        ),
        reasons=joined_reasons,
        expected_item_ids=sorted(expected_item_ids),
        candidate_item_ids=candidate_ids,
        selected_item_ids=selected,
        resolved_item_ids=resolved,
        unresolved_item_ids=unresolved,
        budget=selection.budget,
        spent=selection.spent,
        cost_unit=selection.cost_unit,
        unresolved_expected_claim_loss=guard.unresolved_expected_claim_loss,
        unresolved_conclusion_flip_item_ids=sorted(
            guard.unresolved_conclusion_flip_item_ids
        ),
        unresolved_high_influence_item_ids=sorted(
            guard.unresolved_high_influence_item_ids
        ),
        unresolved_noncalibrated_item_ids=sorted(
            guard.unresolved_noncalibrated_item_ids
        ),
        unresolved_without_probability_bound_item_ids=sorted(
            guard.unresolved_without_probability_bound_item_ids
        ),
        residual_decision_risk_upper_bound=(
            guard.residual_decision_risk_upper_bound
        ),
        residual_risk_bound_assumptions=list(
            guard.residual_risk_bound_assumptions
        ),
        ranking=ranking,
        resolution_receipts=validated_receipts,
        candidate_input_sha256=candidate_hash,
        resolution_ledger_sha256=resolution_hash,
        selection_sha256=hash_canonical(selection_payload),
        guard_sha256=hash_canonical(guard_payload),
    )


def _risk_features(
    *,
    graph_features: Mapping[str, float | int],
    synthesis: Mapping[str, Any],
    evidence: SynthesisEvidenceAssessment,
    audit: AuditGateAssessment,
) -> dict[str, float]:
    values = {name: 0.0 for name in CLAIM_RELEASE_RISK_FEATURE_NAMES}
    for name in _GRAPH_FEATURE_NAMES:
        values[f"graph_{name}"] = float(graph_features[name])

    quantitative = synthesis.get("quantitative")
    if isinstance(quantitative, Mapping) and quantitative.get("status") == "ok":
        lower = float(quantitative["ci_lower"])
        upper = float(quantitative["ci_upper"])
        values["synthesis_quantitative_available"] = 1.0
        values["synthesis_n_papers"] = float(quantitative["n_papers"])
        values["synthesis_ci_width"] = upper - lower
        values["synthesis_ci_crosses_null"] = float(lower <= 0 <= upper)
        heterogeneity = quantitative.get("heterogeneity")
        if isinstance(heterogeneity, Mapping):
            values["synthesis_i_squared"] = float(heterogeneity["i_squared"])
        prediction = quantitative.get("prediction_interval")
        if isinstance(prediction, Mapping) and prediction.get("status") == "ok":
            prediction_lower = float(prediction["lower"])
            prediction_upper = float(prediction["upper"])
            values["synthesis_prediction_interval_available"] = 1.0
            values["synthesis_prediction_interval_width"] = (
                prediction_upper - prediction_lower
            )
            values["synthesis_prediction_interval_crosses_null"] = float(
                prediction_lower <= 0 <= prediction_upper
            )
            values["synthesis_prediction_interval_target_stable"] = float(
                _target_side(
                    prediction_lower,
                    prediction_upper,
                    evidence.target_direction,
                )
            )

    directional = synthesis.get("directional_fallback")
    if isinstance(directional, Mapping) and directional.get("status") == "ok":
        interval = directional["increase_fraction_exact_ci_95"]
        lower = float(interval[0])
        upper = float(interval[1])
        values["synthesis_directional_available"] = 1.0
        values["synthesis_n_papers"] = float(directional["n_papers"])
        values["synthesis_directional_ci_width"] = upper - lower
        values["synthesis_directional_ci_crosses_half"] = float(lower <= 0.5 <= upper)

    count = len(audit.candidate_item_ids)
    selected_count = len(audit.selected_item_ids)
    resolved_count = len(audit.resolved_item_ids)
    unresolved_count = len(audit.unresolved_item_ids)
    values["audit_candidate_count"] = float(count)
    if count:
        values["audit_selected_fraction"] = selected_count / count
        values["audit_resolved_fraction"] = resolved_count / count
        values["audit_unresolved_fraction"] = unresolved_count / count
        values["audit_unresolved_flip_fraction"] = (
            len(audit.unresolved_conclusion_flip_item_ids) / count
        )
        values["audit_unresolved_noncalibrated_fraction"] = (
            len(audit.unresolved_noncalibrated_item_ids) / count
        )
        values["audit_unresolved_without_probability_bound_fraction"] = (
            len(audit.unresolved_without_probability_bound_item_ids) / count
        )
    values["audit_unresolved_expected_claim_loss"] = (
        audit.unresolved_expected_claim_loss
    )
    if audit.residual_decision_risk_upper_bound is not None:
        values["audit_residual_decision_risk_upper_bound_available"] = 1.0
        values["audit_residual_decision_risk_upper_bound"] = (
            audit.residual_decision_risk_upper_bound
        )
    unresolved_rows = [
        row for row in audit.ranking if row.item_id in set(audit.unresolved_item_ids)
    ]
    values["audit_max_unresolved_influence"] = max(
        (row.probability_influence for row in unresolved_rows),
        default=0.0,
    )
    return {name: float(values[name]) for name in CLAIM_RELEASE_RISK_FEATURE_NAMES}


def _calibration_gate(
    *,
    question_id: str,
    population_id: str,
    domain: str,
    pipeline_sha256: str,
    paper_ids: Sequence[str],
    features: Mapping[str, float],
    bundle: FrozenCalibrationBundle | None,
) -> CalibrationGateAssessment:
    if bundle is None:
        return CalibrationGateAssessment(
            status="not_run",
            reason="frozen_calibration_bundle_absent",
        )
    try:
        bundle = validate_frozen_calibration_bundle_integrity(bundle)
    except CalibrationContractError as exc:
        raise ClaimReleaseContractError(f"calibration_contract_failed:{exc}") from exc
    missing = sorted(set(bundle.feature_names) - set(features))
    extra = sorted(set(features) - set(bundle.feature_names))
    if missing or extra or bundle.feature_names != list(features):
        raise ClaimReleaseContractError(
            f"calibration_feature_schema_mismatch:missing={missing}:extra={extra}"
        )
    if not paper_ids:
        return CalibrationGateAssessment(
            status="not_run",
            reason="no_matching_papers_for_release_candidate",
            frozen_bundle_sha256=bundle.bundle_sha256,
            label_source=bundle.label_source,
        )
    if bundle.label_source == "simulation":
        return CalibrationGateAssessment(
            status="abstained",
            reason="simulation_calibration_not_valid_for_scientific_release",
            frozen_bundle_sha256=bundle.bundle_sha256,
            label_source=bundle.label_source,
        )
    candidate = ReleaseCandidate(
        question_id=question_id,
        population_id=population_id,
        domain=domain,
        pipeline_sha256=pipeline_sha256,
        paper_ids=list(paper_ids),
        features=dict(features),
    )
    try:
        assessment: ProspectiveReleaseAssessment = assess_release_candidate(
            candidate,
            bundle,
        )
    except CalibrationContractError as exc:
        raise ClaimReleaseContractError(f"calibration_contract_failed:{exc}") from exc
    return CalibrationGateAssessment(
        status=assessment.status,
        reason=assessment.reason,
        frozen_bundle_sha256=bundle.bundle_sha256,
        release_candidate_sha256=assessment.candidate_sha256,
        prospective_assessment_sha256=hash_canonical(assessment),
        scalar_risk_score=assessment.scalar_risk_score,
        threshold=assessment.threshold,
        label_source=bundle.label_source,
    )


def assess_claim_release(
    *,
    graph: EvidenceGraph,
    question_id: str,
    population_id: str,
    domain: str,
    pipeline_sha256: str,
    target: ClaimTarget,
    audit_candidates: Sequence[AuditCandidate],
    claim_model: ClaimModel,
    audit_resolution_receipts: Sequence[AuditResolutionReceipt],
    audit_budget: float,
    frozen_calibration_bundle: FrozenCalibrationBundle | None,
    config: ClaimReleaseConfig | None = None,
    audit_guard_config: ReleaseGuardConfig | None = None,
) -> ClaimReleaseAssessment:
    """Assess a target claim without accepting oracle or correctness labels.

    Audit candidates must cover the matching graph outcome-estimate identities exactly.
    Resolved items require external :class:`AuditResolutionReceipt` objects bound to the
    current evidence, graph, synthesis, and candidate snapshot.  Unresolved items are
    permitted only when their declared marginal error-probability upper bounds sum to no
    more than the frozen residual-risk tolerance.  Budget selection alone never resolves
    an item, and receipts remain auditable declarations rather than proof of competence.
    """

    if not question_id.strip() or not population_id.strip() or not domain.strip():
        raise ClaimReleaseContractError("claim_identity_fields_must_be_nonempty")
    if not SHA256_RE.fullmatch(pipeline_sha256):
        raise ClaimReleaseContractError("claim_pipeline_sha256_invalid")
    if not math.isfinite(audit_budget) or audit_budget < 0:
        raise ClaimReleaseContractError("claim_audit_budget_invalid")
    if not isinstance(claim_model, ClaimModel):
        raise ClaimReleaseContractError("claim_model_contract_invalid")
    config = config or ClaimReleaseConfig()
    audit_guard_config = audit_guard_config or ReleaseGuardConfig(
        block_counterfactual_conclusion_flips=False
    )
    if not audit_guard_config.require_calibrated_error_probabilities:
        raise ClaimReleaseContractError("release_guard_must_block_noncalibrated_unresolved_items")
    if not audit_guard_config.require_error_probability_upper_bounds:
        raise ClaimReleaseContractError(
            "release_guard_must_require_probability_upper_bounds_for_unresolved_items"
        )

    matching_estimates = [
        estimate
        for estimate in graph.outcome_estimates
        if estimate.outcome_name == target.outcome_name
        and (target.contrast_id is None or estimate.contrast_id == target.contrast_id)
    ]
    expected_item_ids = sorted(estimate.estimate_id for estimate in matching_estimates)
    paper_ids = sorted({estimate.effect.paper_id for estimate in matching_estimates})
    candidate_ids = _validate_exact_audit_coverage(
        expected_item_ids=expected_item_ids,
        candidates=audit_candidates,
    )

    graph_hash = hash_canonical(graph)
    evidence_item_hashes = evidence_item_sha256s(graph)
    graph_feature_model = graph_risk_features(
        graph,
        outcome_name=target.outcome_name,
        contrast_id=target.contrast_id,
    )
    graph_feature_values = graph_feature_model.as_calibration_features()
    selection = select_effect_evidence(
        graph,
        outcome_name=target.outcome_name,
        contrast_id=target.contrast_id,
        require_explicit_timepoint=config.require_explicit_timepoint,
    )
    if selection.status == "ready":
        if sorted(selection.estimate_ids) != expected_item_ids:
            raise ClaimReleaseContractError("graph_selection_identity_mismatch")
        selected_papers = sorted({record.paper_id for record in selection.records})
        if selected_papers != paper_ids:
            raise ClaimReleaseContractError("graph_selection_paper_identity_mismatch")

    synthesis = synthesize_evidence_graph(
        graph,
        outcome_name=target.outcome_name,
        contrast_id=target.contrast_id,
        require_explicit_timepoint=config.require_explicit_timepoint,
        confidence_level=config.confidence_level,
        assumed_within_paper_correlation=config.assumed_within_paper_correlation,
        prespecified_moderators=config.prespecified_condition_moderators,
        condition_familywise_alpha=config.condition_familywise_alpha,
        condition_min_papers_per_level=config.condition_min_papers_per_level,
    )
    synthesis_hash = hash_canonical(synthesis)
    evidence = _classify_synthesis(synthesis, target=target, config=config)
    direct_baseline_decisions = {
        candidate.baseline_decision
        for candidate in audit_candidates
        if candidate.baseline_decision is not None
    }
    if len(direct_baseline_decisions) > 1:
        raise ClaimReleaseContractError("audit_candidate_baseline_decisions_mismatch")
    baseline_claim_probability = claim_model.probability(
        [
            candidate.baseline_contribution
            for candidate in sorted(audit_candidates, key=lambda item: item.item_id)
        ]
    )
    baseline_claim_conclusion = (
        next(iter(direct_baseline_decisions))
        if direct_baseline_decisions
        else claim_model.conclusion(baseline_claim_probability)
    )
    if (
        evidence.classification is EvidenceClassification.SUPPORTED
        and not baseline_claim_conclusion
    ):
        raise ClaimReleaseContractError(
            "claim_model_baseline_conclusion_inconsistent_with_supported_target"
        )

    identity_context = {
        "pipeline_sha256": pipeline_sha256,
        "evidence_graph_sha256": graph_hash,
        "synthesis_sha256": synthesis_hash,
        "target": target.model_dump(mode="json"),
        "expected_item_ids": expected_item_ids,
        "paper_ids": paper_ids,
    }
    audit = _audit_gate(
        expected_item_ids=candidate_ids,
        candidates=audit_candidates,
        claim_model=claim_model,
        resolution_receipts=audit_resolution_receipts,
        evidence_item_sha256s=evidence_item_hashes,
        evidence_graph_sha256=graph_hash,
        synthesis_sha256=synthesis_hash,
        budget=audit_budget,
        config=config,
        guard_config=audit_guard_config,
        identity_context=identity_context,
    )
    features = _risk_features(
        graph_features=graph_feature_values,
        synthesis=synthesis,
        evidence=evidence,
        audit=audit,
    )
    feature_hash = hash_canonical(features)
    calibration = _calibration_gate(
        question_id=question_id,
        population_id=population_id,
        domain=domain,
        pipeline_sha256=pipeline_sha256,
        paper_ids=paper_ids,
        features=features,
        bundle=frozen_calibration_bundle,
    )

    reasons: list[str] = []
    if evidence.classification is not EvidenceClassification.SUPPORTED:
        reasons.append(f"evidence:{evidence.classification.value}:{evidence.reason}")
    if audit.status != "eligible":
        reasons.extend(f"audit:{reason}" for reason in audit.reasons)
    if calibration.status != "released":
        reasons.append(f"calibration:{calibration.reason}")
    status = ClaimReleaseStatus.RELEASED if not reasons else ClaimReleaseStatus.ABSTAINED
    payload: dict[str, Any] = {
        "assessment_version": "prospective-claim-release-v1",
        "question_id": question_id,
        "target": target,
        "pipeline_sha256": pipeline_sha256,
        "evidence_graph_sha256": graph_hash,
        "synthesis_sha256": synthesis_hash,
        "config_sha256": hash_canonical(
            {
                "claim_release": config,
                "audit_guard": asdict(audit_guard_config),
            }
        ),
        "paper_ids": paper_ids,
        "evidence": evidence,
        "audit": audit,
        "risk_feature_schema_version": "claim-release-risk-v1",
        "risk_features": features,
        "risk_features_sha256": feature_hash,
        "calibration": calibration,
        "status": status,
        "reasons": reasons,
        "release_semantics": (
            "all declared prospective gates passed; not a guarantee of scientific truth"
        ),
    }
    return ClaimReleaseAssessment.model_validate(
        {**payload, "decision_sha256": hash_canonical(payload)}
    )


__all__ = [
    "CLAIM_RELEASE_RISK_FEATURE_NAMES",
    "AuditGateAssessment",
    "AuditResolutionProvenance",
    "AuditResolutionReceipt",
    "CalibrationGateAssessment",
    "ClaimReleaseAssessment",
    "ClaimReleaseConfig",
    "ClaimReleaseContractError",
    "ClaimReleaseStatus",
    "ClaimTarget",
    "EvidenceClassification",
    "SynthesisEvidenceAssessment",
    "TargetDirection",
    "assess_claim_release",
    "evidence_item_sha256s",
    "freeze_audit_resolution_receipt",
]
