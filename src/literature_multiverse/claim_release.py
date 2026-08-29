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

from literature_multiverse.adaptive_calibration import (
    AdaptiveCalibrationBundle,
    AdaptiveCalibrationError,
    ConditionCalibrationProjectionV1,
    ConditionConfirmationGateAssessmentV1,
    ProspectiveAdaptiveReleaseCandidate,
    assess_adaptive_release_candidate,
    noncalibration_assessment_sha256,
    validate_adaptive_calibration_bundle_integrity,
)
from literature_multiverse.audit_session import AuditResolutionReceiptV2
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
    validate_frozen_calibration_bundle_for_deployment,
    validate_frozen_calibration_bundle_integrity,
)
from literature_multiverse.claim_semantics import (
    ClaimSpecificationStatus,
    ClaimTargetV2,
    GlobalConditionDependenceTargetV1,
    QualifiedClaimVerdict,
    QualifiedClaimVerdictState,
    freeze_qualified_claim_verdict,
)
from literature_multiverse.evidence_graph import (
    EvidenceGraph,
    graph_risk_features,
    select_effect_evidence,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.meta_analysis import synthesize_evidence_graph
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.sequential_verification import (
    SequentialVerificationContractError,
    SequentialVerificationState,
    adaptive_preselection_history_from_state,
    resume_sequential_verification_state,
)


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

    config_version: Literal["claim-release-v2"] = "claim-release-v2"
    require_explicit_timepoint: bool = True
    require_prediction_interval_stability: bool = True
    confidence_level: Annotated[float, Field(gt=0, lt=1)] = 0.95
    assumed_within_cohort_correlation: Annotated[float, Field(ge=0, le=1)] = 1.0
    audit_allocation_policy: AllocationPolicy = AllocationPolicy.EXPECTED_CLAIM_LOSS_PER_COST
    audit_seed: int = 0
    prespecified_condition_moderators: list[str] = Field(default_factory=list)
    condition_familywise_alpha: Annotated[float, Field(gt=0, lt=1)] = 0.05
    condition_min_cohorts_per_level: Annotated[int, Field(ge=2)] = 2
    condition_confirmation_min_brier_improvement: Annotated[
        float, Field(ge=0)
    ] = 0.0

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
    exploratory_condition_moderators: list[str] = Field(default_factory=list)
    exploratory_condition_interpretation: str | None = None

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
        for name in ("condition_moderators", "exploratory_condition_moderators"):
            values = getattr(self, name)
            if values != sorted(set(values)):
                raise ValueError(
                    f"claim_release_condition_moderators_not_sorted_unique:{name}"
                )
        if self.classification is EvidenceClassification.CONDITION_DEPENDENT:
            if not self.condition_moderators or self.condition_interpretation is None:
                raise ValueError("condition_dependent_requires_moderator_evidence")
        elif self.condition_moderators or self.condition_interpretation is not None:
            raise ValueError("condition_metadata_requires_condition_dependent_classification")
        if bool(self.exploratory_condition_moderators) != (
            self.exploratory_condition_interpretation is not None
        ):
            raise ValueError("exploratory_condition_metadata_incomplete")
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
    """Budget, adjudication, influence, and item-cell triage state kept distinct."""

    status: Literal["eligible", "blocked", "not_applicable"]
    reasons: list[str]
    expected_item_ids: list[str]
    candidate_item_ids: list[str]
    selected_item_ids: list[str]
    resolved_item_ids: list[str]
    historical_selected_item_ids: list[str] = Field(default_factory=list)
    historical_resolved_item_ids: list[str] = Field(default_factory=list)
    unresolved_item_ids: list[str]
    budget: Annotated[float, Field(ge=0)]
    spent: Annotated[float, Field(ge=0)]
    cost_basis: Literal["estimated_selection", "realized_session"] = "estimated_selection"
    cost_unit: str | None
    unresolved_expected_claim_loss: Annotated[float, Field(ge=0)]
    unresolved_conclusion_flip_item_ids: list[str]
    unresolved_high_influence_item_ids: list[str]
    unresolved_noncalibrated_item_ids: list[str]
    unresolved_without_cell_rate_ucl_item_ids: list[str]
    unresolved_item_cell_ucl_sum: Annotated[float, Field(ge=0, le=1)] | None
    item_ucl_interpretation_limits: list[str]
    ranking: list[AuditPrioritySummary]
    resolution_receipts: list[AuditResolutionReceipt]
    resolution_receipts_v2: list[AuditResolutionReceiptV2] = Field(default_factory=list)
    sequential_state_sha256: str | None = None
    candidate_input_sha256: str
    resolution_ledger_sha256: str | None
    selection_sha256: str | None
    guard_sha256: str | None

    @field_validator(
        "candidate_input_sha256",
        "resolution_ledger_sha256",
        "selection_sha256",
        "guard_sha256",
        "sequential_state_sha256",
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
            "historical_selected_item_ids",
            "historical_resolved_item_ids",
            "unresolved_item_ids",
            "unresolved_conclusion_flip_item_ids",
            "unresolved_high_influence_item_ids",
            "unresolved_noncalibrated_item_ids",
            "unresolved_without_cell_rate_ucl_item_ids",
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
        receipt_ids_v1 = [receipt.item_id for receipt in self.resolution_receipts]
        receipt_ids_v2 = [receipt.item_id for receipt in self.resolution_receipts_v2]
        if receipt_ids_v1 != sorted(set(receipt_ids_v1)):
            raise ValueError("claim_release_resolution_receipt_ids_not_sorted_unique")
        if len(receipt_ids_v2) != len(set(receipt_ids_v2)):
            raise ValueError("claim_release_resolution_receipt_v2_ids_not_unique")
        if self.cost_basis == "estimated_selection":
            if self.resolution_receipts_v2 or self.sequential_state_sha256 is not None:
                raise ValueError("estimated_audit_gate_forbids_sequential_artifacts")
            if receipt_ids_v1 != self.resolved_item_ids:
                raise ValueError("claim_release_resolution_receipt_identity_mismatch")
            if self.historical_selected_item_ids != self.selected_item_ids:
                raise ValueError("estimated_audit_selected_history_mismatch")
            if self.historical_resolved_item_ids != self.resolved_item_ids:
                raise ValueError("estimated_audit_resolved_history_mismatch")
        else:
            if self.resolution_receipts or self.sequential_state_sha256 is None:
                raise ValueError("realized_audit_gate_requires_only_sequential_artifacts")
            # The v2 receipt ledger is chronological lineage, whereas the public
            # identity summary is canonical-sorted.  Compare membership exactly
            # without destroying the receipt order used by the ledger hash.
            if sorted(receipt_ids_v2) != self.historical_resolved_item_ids:
                raise ValueError("sequential_resolution_receipt_identity_mismatch")
            if not set(self.selected_item_ids) <= set(self.historical_selected_item_ids):
                raise ValueError("sequential_selected_history_mismatch")
            if not set(self.resolved_item_ids) <= set(self.historical_resolved_item_ids):
                raise ValueError("sequential_resolved_history_mismatch")
        if set(self.unresolved_item_ids) != (
            set(self.candidate_item_ids) - set(self.resolved_item_ids)
        ):
            raise ValueError("claim_release_unresolved_audit_identity_mismatch")
        if self.status == "not_applicable" and self.candidate_item_ids:
            raise ValueError("claim_release_nonempty_audit_cannot_be_not_applicable")
        receipt_ledger: list[AuditResolutionReceipt] | list[AuditResolutionReceiptV2]
        receipt_ledger = (
            self.resolution_receipts
            if self.cost_basis == "estimated_selection"
            else self.resolution_receipts_v2
        )
        expected_ledger_hash = hash_canonical(receipt_ledger) if receipt_ledger else None
        if self.resolution_ledger_sha256 != expected_ledger_hash:
            raise ValueError("claim_release_resolution_ledger_hash_mismatch")
        return self

class CalibrationGateAssessment(ContractModel):
    """Deployment-side score result from an already frozen label-risk policy."""

    status: Literal["released", "abstained", "not_run"]
    reason: str
    calibration_contract: Literal[
        "none",
        "fixed-single-decision-v2",
        "adaptive-first-release-trajectory-v1",
    ] = "none"
    frozen_bundle_sha256: str | None = None
    release_candidate_sha256: str | None = None
    prospective_assessment_sha256: str | None = None
    policy_context_sha256: str | None = None
    scalar_risk_score: Annotated[float, Field(ge=0, le=1)] | None = None
    threshold: Annotated[float, Field(ge=0, le=1)] | None = None
    label_source: Literal["benchmark_annotation", "expert_adjudication", "simulation"] | None = None
    guarantee_scope: Literal[
        "no calibration guarantee",
        "fixed single-decision label-risk policy; not valid after adaptive repeated looks",
        "exact decision-mismatch risk against the frozen adjudication protocol under "
        "exchangeable independent complete-question trajectories; not scientific truth "
        "or domain-shift robustness",
    ] = "no calibration guarantee"

    @field_validator(
        "frozen_bundle_sha256",
        "release_candidate_sha256",
        "prospective_assessment_sha256",
        "policy_context_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("invalid_claim_calibration_sha256")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> CalibrationGateAssessment:
        if self.calibration_contract == "none":
            if self.status == "released" or self.frozen_bundle_sha256 is not None:
                raise ValueError("uncalibrated_gate_cannot_release")
            if self.guarantee_scope != "no calibration guarantee":
                raise ValueError("uncalibrated_gate_guarantee_scope_mismatch")
        elif self.calibration_contract == "fixed-single-decision-v2":
            if self.frozen_bundle_sha256 is None or self.policy_context_sha256 is not None:
                raise ValueError("fixed_calibration_lineage_incomplete")
            if self.guarantee_scope != (
                "fixed single-decision label-risk policy; not valid after adaptive "
                "repeated looks"
            ):
                raise ValueError("fixed_calibration_guarantee_scope_mismatch")
        else:
            if (
                self.frozen_bundle_sha256 is None
                or self.release_candidate_sha256 is None
                or self.prospective_assessment_sha256 is None
                or self.policy_context_sha256 is None
            ):
                raise ValueError("adaptive_calibration_lineage_incomplete")
            if self.guarantee_scope != (
                "exact decision-mismatch risk against the frozen adjudication protocol "
                "under exchangeable independent complete-question trajectories; not "
                "scientific truth or domain-shift robustness"
            ):
                raise ValueError("adaptive_calibration_guarantee_scope_mismatch")
        return self


class ClaimReleaseAssessment(ContractModel):
    """Hash-bound final decision; every gate must pass for ``released``."""

    assessment_version: Literal["prospective-claim-release-v2"] = "prospective-claim-release-v2"
    question_id: str
    target: ClaimTarget
    pipeline_sha256: str
    evidence_graph_sha256: str
    synthesis_sha256: str
    config_sha256: str
    paper_ids: list[str]
    evidence: SynthesisEvidenceAssessment
    audit: AuditGateAssessment
    risk_feature_schema_version: Literal["claim-release-risk-v1"] = "claim-release-risk-v1"
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


class ConditionClaimReleaseAssessmentV1(ContractModel):
    """Development-only source decision for a global condition claim.

    This artifact is intentionally release-ineligible.  It carries the exact
    outcome-free projection used by the online scheduler and a typed scientific
    confirmation gate, but a final release is possible only in the separate v7
    certificate after confirmation-aware complete-question calibration replays the
    terminal join.  Same-corpus moderator analysis remains exploratory evidence.
    """

    assessment_version: Literal["condition-claim-release-source-v1"] = (
        "condition-claim-release-source-v1"
    )
    question_id: Annotated[str, Field(min_length=1)]
    target: GlobalConditionDependenceTargetV1
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
    condition_calibration_projection: ConditionCalibrationProjectionV1
    condition_confirmation_gate: ConditionConfirmationGateAssessmentV1
    terminal_gate_deferred: bool
    status: Literal[ClaimReleaseStatus.ABSTAINED] = ClaimReleaseStatus.ABSTAINED
    reasons: list[str]
    decision_sha256: str
    release_semantics: Literal[
        "source decision only; condition-dependent release requires held-out "
        "confirmation and confirmation-aware complete-question calibration in v7"
    ] = (
        "source decision only; condition-dependent release requires held-out "
        "confirmation and confirmation-aware complete-question calibration in v7"
    )

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
            raise ValueError("invalid_condition_claim_release_sha256")
        return value

    @field_validator("paper_ids", "reasons")
    @classmethod
    def validate_sorted_unique_strings(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)) or any(not item.strip() for item in value):
            raise ValueError(
                f"condition_claim_release_{info.field_name}_not_sorted_unique"
            )
        return value

    @field_validator("risk_features")
    @classmethod
    def validate_risk_features(cls, value: dict[str, float]) -> dict[str, float]:
        if list(value) != list(CLAIM_RELEASE_RISK_FEATURE_NAMES):
            raise ValueError("condition_claim_release_risk_feature_schema_mismatch")
        if any(not math.isfinite(number) for number in value.values()):
            raise ValueError("condition_claim_release_risk_feature_nonfinite")
        return value

    @field_validator("terminal_gate_deferred", mode="before")
    @classmethod
    def validate_strict_deferred(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("condition_claim_release_terminal_deferred_must_be_boolean")
        return value

    @model_validator(mode="after")
    def validate_source_decision(self) -> ConditionClaimReleaseAssessmentV1:
        projection = self.condition_calibration_projection
        gate = self.condition_confirmation_gate
        if (
            projection.question_id != self.question_id
            or projection.condition_target_sha256 != self.target.target_sha256
            or projection.pipeline_sha256 != self.pipeline_sha256
            or projection.online_graph_sha256 != self.evidence_graph_sha256
            or projection.prespecified_moderator_names != self.target.moderator_names
        ):
            raise ValueError("condition_claim_release_projection_context_mismatch")
        if (
            not gate.required
            or gate.provisional_claim_decision != "condition_dependent"
            or gate.status == "not_applicable"
            or gate.condition_projection_sha256 != projection.projection_sha256
            or gate.target_sha256 != projection.condition_target_sha256
            or gate.plan_sha256 != projection.plan_sha256
            or gate.config_sha256 != projection.confirmation_config_sha256
        ):
            raise ValueError("condition_claim_release_confirmation_gate_mismatch")
        expected_deferred = gate.status == "missing"
        if self.terminal_gate_deferred is not expected_deferred:
            raise ValueError("condition_claim_release_terminal_deferred_status_mismatch")
        required_reason = {
            "missing": "condition_confirmation_required",
            "confirmed": "condition_dependent_confirmation_aware_calibration_required",
            "not_confirmed": "condition_confirmation_not_confirmed",
            "insufficient": "condition_confirmation_insufficient",
        }[gate.status]
        if required_reason not in self.reasons:
            raise ValueError("condition_claim_release_required_blocker_missing")
        if gate.status == "missing" and (
            "condition_dependent_confirmation_aware_calibration_required"
            not in self.reasons
        ):
            raise ValueError(
                "condition_claim_release_confirmation_calibration_blocker_missing"
            )
        if self.evidence.classification is EvidenceClassification.CONDITION_DEPENDENT:
            raise ValueError("same_corpus_condition_evidence_cannot_be_final")
        if self.evidence.target_direction.value != self.target.reference_direction.value:
            raise ValueError("condition_claim_release_reference_direction_mismatch")
        if self.calibration.status == "released":
            raise ValueError("condition_source_decision_cannot_use_release_calibration")
        if hash_canonical(self.risk_features) != self.risk_features_sha256:
            raise ValueError("condition_claim_release_risk_feature_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"decision_sha256"})
        if hash_canonical(payload) != self.decision_sha256:
            raise ValueError("condition_claim_release_decision_hash_mismatch")
        return self


class QualifiedClaimEvidenceAssessment(ContractModel):
    """Hash-bound v2 synthesis gate; it does not bypass audit or calibration."""

    assessment_version: Literal["qualified-claim-evidence-v2"] = "qualified-claim-evidence-v2"
    target: ClaimTargetV2
    evidence_graph_sha256: str
    synthesis: dict[str, Any]
    synthesis_sha256: str
    verdict: QualifiedClaimVerdict
    assessment_sha256: str
    release_semantics: Literal[
        "evidence gate only; claim release still requires audit and calibration"
    ] = "evidence gate only; claim release still requires audit and calibration"

    @field_validator("evidence_graph_sha256", "synthesis_sha256", "assessment_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("qualified_claim_assessment_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_assessment(self) -> QualifiedClaimEvidenceAssessment:
        if hash_canonical(self.synthesis) != self.synthesis_sha256:
            raise ValueError("qualified_claim_synthesis_hash_mismatch")
        qualified = self.synthesis.get("qualified_claim")
        if (
            not isinstance(qualified, Mapping)
            or qualified.get("target_sha256") != self.target.claim_sha256
        ):
            raise ValueError("qualified_claim_assessment_target_mismatch")
        if self.verdict.target_sha256 != self.target.claim_sha256:
            raise ValueError("qualified_claim_verdict_target_mismatch")
        if self.verdict.synthesis_sha256 != self.synthesis_sha256:
            raise ValueError("qualified_claim_verdict_synthesis_mismatch")
        payload = self.model_dump(mode="json", exclude={"assessment_sha256"})
        if hash_canonical(payload) != self.assessment_sha256:
            raise ValueError("qualified_claim_assessment_hash_mismatch")
        return self


class QualifiedClaimReleaseAssessment(ContractModel):
    """Final v2 release decision for an exactly qualified, magnitude-aware claim.

    The evidence verdict is computed only on estimates matching the frozen condition
    predicates.  Human-audit and complete-question calibration remain independent,
    mandatory gates; a successful synthesis alone can never release the claim.
    """

    assessment_version: Literal["prospective-qualified-claim-release-v2"] = (
        "prospective-qualified-claim-release-v2"
    )
    question_id: Annotated[str, Field(min_length=1)]
    target: ClaimTargetV2
    pipeline_sha256: str
    evidence_graph_sha256: str
    synthesis_sha256: str
    config_sha256: str
    paper_ids: list[str]
    evidence: QualifiedClaimVerdict
    audit: AuditGateAssessment
    risk_feature_schema_version: Literal["claim-release-risk-v1"] = "claim-release-risk-v1"
    risk_features: dict[str, float]
    risk_features_sha256: str
    calibration: CalibrationGateAssessment
    status: ClaimReleaseStatus
    reasons: list[str]
    decision_sha256: str
    release_semantics: Literal[
        "prespecified qualified claim passed evidence, audit, and calibration gates; "
        "not scientific truth"
    ] = (
        "prespecified qualified claim passed evidence, audit, and calibration gates; "
        "not scientific truth"
    )

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
            raise ValueError("invalid_qualified_claim_release_sha256")
        return value

    @field_validator("paper_ids")
    @classmethod
    def validate_paper_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("qualified_claim_release_paper_ids_not_sorted_unique")
        return value

    @field_validator("risk_features")
    @classmethod
    def validate_risk_features(cls, value: dict[str, float]) -> dict[str, float]:
        if list(value) != list(CLAIM_RELEASE_RISK_FEATURE_NAMES):
            raise ValueError("qualified_claim_release_risk_feature_schema_mismatch")
        if any(not math.isfinite(number) for number in value.values()):
            raise ValueError("qualified_claim_release_risk_feature_nonfinite")
        return value

    @model_validator(mode="after")
    def validate_release(self) -> QualifiedClaimReleaseAssessment:
        gates_pass = (
            self.evidence.state is QualifiedClaimVerdictState.PRESPECIFIED_SUPPORTED
            and self.audit.status == "eligible"
            and self.calibration.status == "released"
        )
        if (self.status is ClaimReleaseStatus.RELEASED) != gates_pass:
            raise ValueError("qualified_claim_release_status_gate_mismatch")
        if self.target.claim_sha256 != self.evidence.target_sha256:
            raise ValueError("qualified_claim_release_target_verdict_mismatch")
        if self.synthesis_sha256 != self.evidence.synthesis_sha256:
            raise ValueError("qualified_claim_release_synthesis_verdict_mismatch")
        if self.status is ClaimReleaseStatus.RELEASED and self.reasons:
            raise ValueError("released_qualified_claim_cannot_have_blocking_reasons")
        if self.status is ClaimReleaseStatus.ABSTAINED and not self.reasons:
            raise ValueError("abstained_qualified_claim_requires_reason")
        if self.reasons != list(dict.fromkeys(self.reasons)):
            raise ValueError("qualified_claim_release_reasons_not_unique")
        if hash_canonical(self.risk_features) != self.risk_features_sha256:
            raise ValueError("qualified_claim_release_risk_feature_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"decision_sha256"})
        if hash_canonical(payload) != self.decision_sha256:
            raise ValueError("qualified_claim_release_decision_hash_mismatch")
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
            "audit_unresolved_without_cell_rate_ucl_fraction",
            "audit_item_cell_rate_ucl_sum_available",
            "audit_item_cell_rate_ucl_sum",
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
    exploratory_moderators: list[str] = []
    exploratory_interpretation: str | None = None
    if (
        isinstance(condition, Mapping)
        and condition.get("status") == "exploratory_qualitative_condition_signal"
    ):
        raw_qualifying = condition.get("qualifying_moderators")
        if not isinstance(raw_qualifying, list):
            raise ClaimReleaseContractError("condition_analysis_qualifiers_invalid")
        exploratory_moderators = sorted(
            str(row["moderator"])
            for row in raw_qualifying
            if isinstance(row, Mapping) and row.get("moderator")
        )
        if not exploratory_moderators:
            raise ClaimReleaseContractError("condition_analysis_qualifiers_empty")
        exploratory_interpretation = str(
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

        if _opposite_side(lower, upper, target.direction):
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
            exploratory_condition_moderators=exploratory_moderators,
            exploratory_condition_interpretation=exploratory_interpretation,
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
        if target_contradicted:
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
            exploratory_condition_moderators=exploratory_moderators,
            exploratory_condition_interpretation=exploratory_interpretation,
        )

    raise ClaimReleaseContractError(f"unknown_successful_synthesis_mode:{mode}")


def _qualified_selection_ids(selection: Mapping[str, Any], name: str) -> list[str]:
    raw = selection.get(name)
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ClaimReleaseContractError(f"qualified_claim_selection_ids_invalid:{name}")
    if raw != sorted(set(raw)):
        raise ClaimReleaseContractError(f"qualified_claim_selection_ids_not_sorted_unique:{name}")
    return raw


def classify_qualified_synthesis_evidence(
    synthesis: Mapping[str, Any],
    *,
    target: ClaimTargetV2,
    require_prediction_interval_stability: bool = True,
) -> QualifiedClaimVerdict:
    """Apply exact target conditions and a meaningful-effect boundary to synthesis.

    A discovered target is always a hypothesis-only result on its discovery corpus,
    irrespective of the numerical estimate. Prespecified targets require compatible
    magnitude synthesis on exactly the threshold's harmonized scale; sign-only evidence
    cannot establish a minimum meaningful magnitude.
    """

    qualified = synthesis.get("qualified_claim")
    if not isinstance(qualified, Mapping):
        raise ClaimReleaseContractError("qualified_claim_selection_missing")
    if qualified.get("target_sha256") != target.claim_sha256:
        raise ClaimReleaseContractError("qualified_claim_selection_target_mismatch")
    candidates = _qualified_selection_ids(qualified, "candidate_estimate_ids")
    matched = _qualified_selection_ids(qualified, "matched_estimate_ids")
    condition_excluded = _qualified_selection_ids(qualified, "condition_excluded_estimate_ids")
    missing = _qualified_selection_ids(qualified, "missing_condition_estimate_ids")
    type_mismatch = _qualified_selection_ids(qualified, "type_mismatch_estimate_ids")
    partitions = (matched, condition_excluded, missing, type_mismatch)
    flattened = [item for partition in partitions for item in partition]
    if len(flattened) != len(set(flattened)) or sorted(flattened) != candidates:
        raise ClaimReleaseContractError("qualified_claim_selection_partition_mismatch")
    synthesis_sha = hash_canonical(synthesis)
    mode = str(synthesis.get("mode", "unknown"))

    quantitative = synthesis.get("quantitative")
    quantitative_ok = isinstance(quantitative, Mapping) and quantitative.get("status") == "ok"
    estimate = float(quantitative["estimate"]) if quantitative_ok else None
    lower = float(quantitative["ci_lower"]) if quantitative_ok else None
    upper = float(quantitative["ci_upper"]) if quantitative_ok else None
    prediction = quantitative.get("prediction_interval") if quantitative_ok else None
    prediction_ok = isinstance(prediction, Mapping) and prediction.get("status") == "ok"
    prediction_lower = float(prediction["lower"]) if prediction_ok else None
    prediction_upper = float(prediction["upper"]) if prediction_ok else None

    if target.specification_status is ClaimSpecificationStatus.DISCOVERED_HYPOTHESIS:
        return freeze_qualified_claim_verdict(
            target=target,
            synthesis_sha256=synthesis_sha,
            state=QualifiedClaimVerdictState.DISCOVERED_HYPOTHESIS_ONLY,
            reason=(
                "discovered_condition_is_not_a_prespecified_confirmation_target_and_"
                "requires_independent_confirmation"
            ),
            mode=mode,
            matched_estimate_ids=matched,
            condition_excluded_estimate_ids=condition_excluded,
            missing_condition_estimate_ids=missing,
            type_mismatch_estimate_ids=type_mismatch,
            estimate=estimate,
            ci_lower=lower,
            ci_upper=upper,
            prediction_interval_lower=prediction_lower,
            prediction_interval_upper=prediction_upper,
        )

    common: dict[str, Any] = {
        "target": target,
        "synthesis_sha256": synthesis_sha,
        "mode": mode,
        "matched_estimate_ids": matched,
        "condition_excluded_estimate_ids": condition_excluded,
        "missing_condition_estimate_ids": missing,
        "type_mismatch_estimate_ids": type_mismatch,
        "estimate": estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "prediction_interval_lower": prediction_lower,
        "prediction_interval_upper": prediction_upper,
    }
    if synthesis.get("status") != "ok":
        return freeze_qualified_claim_verdict(
            **common,
            state=QualifiedClaimVerdictState.PRESPECIFIED_NOT_EVALUABLE,
            reason=_synthesis_reason(synthesis),
        )
    if mode != "random_effects_meta_analysis" or not quantitative_ok:
        return freeze_qualified_claim_verdict(
            **common,
            state=QualifiedClaimVerdictState.PRESPECIFIED_NOT_EVALUABLE,
            reason="meaningful_effect_threshold_requires_compatible_quantitative_synthesis",
        )

    threshold = target.meaningful_effect_threshold
    if quantitative.get("measure") != threshold.measure.value:
        return freeze_qualified_claim_verdict(
            **common,
            state=QualifiedClaimVerdictState.PRESPECIFIED_NOT_EVALUABLE,
            reason="meaningful_effect_threshold_measure_mismatch",
        )
    if quantitative.get("unit") != threshold.unit:
        return freeze_qualified_claim_verdict(
            **common,
            state=QualifiedClaimVerdictState.PRESPECIFIED_NOT_EVALUABLE,
            reason="meaningful_effect_threshold_unit_mismatch",
        )

    assert lower is not None and upper is not None
    delta = threshold.minimum_magnitude
    if target.direction.value == "increase":
        target_ci_margin = lower - delta
        opposite_ci_margin = -upper - delta
        prediction_margin = prediction_lower - delta if prediction_lower is not None else None
    else:
        target_ci_margin = -upper - delta
        opposite_ci_margin = lower - delta
        prediction_margin = -prediction_upper - delta if prediction_upper is not None else None
    decision_margin = (
        min(target_ci_margin, prediction_margin)
        if require_prediction_interval_stability and prediction_margin is not None
        else target_ci_margin
    )

    if opposite_ci_margin > 0:
        state = QualifiedClaimVerdictState.PRESPECIFIED_CONTRADICTED
        reason = "confidence_interval_supports_meaningful_opposite_effect"
    elif target_ci_margin <= 0:
        state = QualifiedClaimVerdictState.PRESPECIFIED_INCONCLUSIVE
        reason = "confidence_interval_does_not_exclude_submeaningful_effects"
    elif require_prediction_interval_stability and not prediction_ok:
        state = QualifiedClaimVerdictState.PRESPECIFIED_INCONCLUSIVE
        reason = "prediction_interval_required_but_unavailable"
    elif (
        require_prediction_interval_stability
        and prediction_margin is not None
        and prediction_margin <= 0
    ):
        state = QualifiedClaimVerdictState.PRESPECIFIED_INCONCLUSIVE
        reason = "prediction_interval_not_stable_beyond_meaningful_effect_threshold"
    else:
        state = QualifiedClaimVerdictState.PRESPECIFIED_SUPPORTED
        reason = "confidence_and_required_prediction_intervals_exceed_meaningful_threshold"
    return freeze_qualified_claim_verdict(
        **common,
        state=state,
        reason=reason,
        decision_margin=decision_margin,
    )


def assess_qualified_claim_evidence(
    *,
    graph: EvidenceGraph,
    target: ClaimTargetV2,
    config: ClaimReleaseConfig | None = None,
) -> QualifiedClaimEvidenceAssessment:
    """Run the v2 condition/magnitude evidence gate without implying final release."""

    config = config or ClaimReleaseConfig()
    synthesis = synthesize_evidence_graph(
        graph,
        outcome_name=target.outcome_name,
        contrast_id=target.contrast_id,
        require_explicit_timepoint=config.require_explicit_timepoint,
        confidence_level=config.confidence_level,
        assumed_within_cohort_correlation=config.assumed_within_cohort_correlation,
        prespecified_moderators=config.prespecified_condition_moderators,
        condition_familywise_alpha=config.condition_familywise_alpha,
        condition_min_cohorts_per_level=config.condition_min_cohorts_per_level,
        qualified_target=target,
    )
    synthesis_sha = hash_canonical(synthesis)
    verdict = classify_qualified_synthesis_evidence(
        synthesis,
        target=target,
        require_prediction_interval_stability=(config.require_prediction_interval_stability),
    )
    payload: dict[str, Any] = {
        "assessment_version": "qualified-claim-evidence-v2",
        "target": target,
        "evidence_graph_sha256": hash_canonical(graph),
        "synthesis": synthesis,
        "synthesis_sha256": synthesis_sha,
        "verdict": verdict,
        "release_semantics": (
            "evidence gate only; claim release still requires audit and calibration"
        ),
    }
    return QualifiedClaimEvidenceAssessment.model_validate(
        {**payload, "assessment_sha256": hash_canonical(payload)}
    )


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
    sequential_state: SequentialVerificationState | None = None,
) -> AuditGateAssessment:
    candidate_ids = _validate_exact_audit_coverage(
        expected_item_ids=expected_item_ids,
        candidates=candidates,
    )
    validated_receipts: list[AuditResolutionReceipt] = []
    validated_receipts_v2: list[AuditResolutionReceiptV2] = []
    sequential: SequentialVerificationState | None = None
    if sequential_state is None:
        if any(not isinstance(receipt, AuditResolutionReceipt) for receipt in resolution_receipts):
            raise ClaimReleaseContractError("audit_resolution_receipt_contract_invalid")
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
        historical_selected: list[str] | None = None
        historical_resolved = list(resolved)
        spent_override: float | None = None
        cost_basis: Literal["estimated_selection", "realized_session"] = "estimated_selection"
    else:
        if resolution_receipts:
            raise ClaimReleaseContractError(
                "sequential_audit_state_conflicts_with_v1_resolution_receipts"
            )
        try:
            sequential = resume_sequential_verification_state(sequential_state)
        except ValueError as exc:
            raise ClaimReleaseContractError("sequential_audit_state_integrity_changed") from exc
        session = sequential.session
        if (
            sequential.graph_sha256 != evidence_graph_sha256
            or sequential.synthesis_sha256 != synthesis_sha256
            or session.pipeline_sha256 != identity_context.get("pipeline_sha256")
            or not math.isclose(session.budget, budget, rel_tol=1e-12, abs_tol=1e-9)
        ):
            raise ClaimReleaseContractError("sequential_audit_state_release_context_mismatch")
        state_candidates = {row.item_id: row for row in sequential.candidates}
        if set(state_candidates) != set(candidate_ids):
            raise ClaimReleaseContractError("sequential_audit_current_candidate_identity_mismatch")
        by_id = {candidate.item_id: candidate for candidate in candidates}
        for item_id, state_candidate in state_candidates.items():
            if state_candidate.scientific_candidate_sha256 != hash_canonical(
                asdict(by_id[item_id])
            ):
                raise ClaimReleaseContractError(
                    f"sequential_audit_scientific_candidate_mismatch:{item_id}"
                )
        validated_receipts_v2 = [step.receipt for step in session.steps]
        historical_selected = sorted(session.selected_item_ids)
        historical_resolved = sorted(session.resolved_item_ids)
        resolved = sorted(set(historical_resolved) & set(candidate_ids))
        spent_override = session.current_realized_cost
        cost_basis = "realized_session"
    if len(resolved) != len(set(resolved)):
        raise ClaimReleaseContractError("audit_resolution_receipt_id_duplicate")
    unknown_resolved = sorted(set(resolved) - set(candidate_ids))
    if unknown_resolved:
        raise ClaimReleaseContractError(
            f"audit_resolution_receipt_identity_unknown:{unknown_resolved}"
        )
    ordered_candidates = sorted(candidates, key=lambda candidate: candidate.item_id)
    candidate_payload = [_audit_candidate_payload(candidate) for candidate in ordered_candidates]
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
    receipt_ledger = validated_receipts if sequential is None else validated_receipts_v2
    resolution_hash = hash_canonical(receipt_ledger) if receipt_ledger else None
    if not candidates:
        return AuditGateAssessment(
            status="not_applicable",
            reasons=["no_matching_audit_items"],
            expected_item_ids=[],
            candidate_item_ids=[],
            selected_item_ids=[],
            resolved_item_ids=[],
            historical_selected_item_ids=historical_selected or [],
            historical_resolved_item_ids=historical_resolved,
            unresolved_item_ids=[],
            budget=budget,
            spent=spent_override or 0,
            cost_basis=cost_basis,
            cost_unit=(sequential.session.cost_unit if sequential is not None else None),
            unresolved_expected_claim_loss=0,
            unresolved_conclusion_flip_item_ids=[],
            unresolved_high_influence_item_ids=[],
            unresolved_noncalibrated_item_ids=[],
            unresolved_without_cell_rate_ucl_item_ids=[],
            unresolved_item_cell_ucl_sum=0,
            item_ucl_interpretation_limits=[
                "no_unresolved_evidence_items",
            ],
            ranking=[],
            resolution_receipts=[],
            resolution_receipts_v2=validated_receipts_v2,
            sequential_state_sha256=(sequential.state_sha256 if sequential is not None else None),
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
    selected = (
        sorted(selection.selected_item_ids)
        if sequential is None
        else sorted(set(historical_selected or []) & set(candidate_ids))
    )
    if historical_selected is None:
        historical_selected = list(selected)
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
            expected_claim_loss_reduction_per_cost=(row.expected_claim_loss_reduction_per_cost),
            decision_score_source=row.decision_score_source,
        )
        for row in selection.ranking
    ]
    selection_payload = {
        "policy": selection.policy.value,
        "budget": selection.budget,
        "spent": selection.spent if spent_override is None else spent_override,
        "cost_basis": cost_basis,
        "cost_unit": selection.cost_unit,
        "selected_item_ids": selected,
        "historical_selected_item_ids": historical_selected,
        "ranking": [row.model_dump(mode="json") for row in ranking],
    }
    guard_payload = {
        "status": guard.status.value,
        "reasons": list(guard.reasons),
        "resolved_item_ids": list(guard.resolved_item_ids),
        "unresolved_item_ids": list(guard.unresolved_item_ids),
        "unresolved_conclusion_flip_item_ids": list(guard.unresolved_conclusion_flip_item_ids),
        "unresolved_high_influence_item_ids": list(guard.unresolved_high_influence_item_ids),
        "unresolved_noncalibrated_item_ids": list(guard.unresolved_noncalibrated_item_ids),
        "unresolved_without_cell_rate_ucl_item_ids": list(
            guard.unresolved_without_cell_rate_ucl_item_ids
        ),
        "unresolved_expected_claim_loss": guard.unresolved_expected_claim_loss,
        "unresolved_item_cell_ucl_sum": guard.unresolved_item_cell_ucl_sum,
        "item_ucl_interpretation_limits": list(guard.item_ucl_interpretation_limits),
        "config": asdict(guard.config),
    }
    joined_reasons = list(guard.reasons)
    if sequential is not None and sequential.session.active_action is not None:
        joined_reasons.append("active_audit_action_unresolved")
    return AuditGateAssessment(
        status=(
            "eligible"
            if guard.status is ReleaseGuardStatus.ELIGIBLE_FOR_DOWNSTREAM_GATES
            and not joined_reasons
            else "blocked"
        ),
        reasons=joined_reasons,
        expected_item_ids=sorted(expected_item_ids),
        candidate_item_ids=candidate_ids,
        selected_item_ids=selected,
        resolved_item_ids=resolved,
        historical_selected_item_ids=historical_selected,
        historical_resolved_item_ids=historical_resolved,
        unresolved_item_ids=unresolved,
        budget=selection.budget,
        spent=selection.spent if spent_override is None else spent_override,
        cost_basis=cost_basis,
        cost_unit=selection.cost_unit,
        unresolved_expected_claim_loss=guard.unresolved_expected_claim_loss,
        unresolved_conclusion_flip_item_ids=sorted(guard.unresolved_conclusion_flip_item_ids),
        unresolved_high_influence_item_ids=sorted(guard.unresolved_high_influence_item_ids),
        unresolved_noncalibrated_item_ids=sorted(guard.unresolved_noncalibrated_item_ids),
        unresolved_without_cell_rate_ucl_item_ids=sorted(
            guard.unresolved_without_cell_rate_ucl_item_ids
        ),
        unresolved_item_cell_ucl_sum=guard.unresolved_item_cell_ucl_sum,
        item_ucl_interpretation_limits=list(guard.item_ucl_interpretation_limits),
        ranking=ranking,
        resolution_receipts=validated_receipts,
        resolution_receipts_v2=validated_receipts_v2,
        sequential_state_sha256=(sequential.state_sha256 if sequential is not None else None),
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
            values["synthesis_prediction_interval_width"] = prediction_upper - prediction_lower
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
        values["audit_unresolved_without_cell_rate_ucl_fraction"] = (
            len(audit.unresolved_without_cell_rate_ucl_item_ids) / count
        )
    values["audit_unresolved_expected_claim_loss"] = audit.unresolved_expected_claim_loss
    if audit.unresolved_item_cell_ucl_sum is not None:
        values["audit_item_cell_rate_ucl_sum_available"] = 1.0
        values["audit_item_cell_rate_ucl_sum"] = audit.unresolved_item_cell_ucl_sum
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
    adaptive_bundle: AdaptiveCalibrationBundle | None,
    adaptive_candidate: ProspectiveAdaptiveReleaseCandidate | None,
    sequential_state: SequentialVerificationState | None,
    noncalibration_assessment_sha256: str,
    noncalibration_gates_passed: bool,
    noncalibration_blocking_reasons: Sequence[str],
    claim_decision: str,
) -> CalibrationGateAssessment:
    if bundle is not None and adaptive_bundle is not None:
        raise ClaimReleaseContractError("multiple_claim_calibration_bundles_supplied")
    if bundle is None and adaptive_bundle is None:
        return CalibrationGateAssessment(
            status="not_run",
            reason="frozen_calibration_bundle_absent",
        )
    if adaptive_bundle is not None:
        if adaptive_candidate is None:
            raise ClaimReleaseContractError("adaptive_release_candidate_required")
        try:
            adaptive_bundle = validate_adaptive_calibration_bundle_integrity(
                adaptive_bundle
            )
        except AdaptiveCalibrationError as exc:
            raise ClaimReleaseContractError(
                f"adaptive_calibration_contract_failed:{exc}"
            ) from exc
        if sequential_state is None:
            raise ClaimReleaseContractError(
                "adaptive_calibration_requires_sequential_production_state"
            )
        try:
            (
                ledger_history,
                ledger_policy_context_sha256,
                ledger_bundle_sha256,
            ) = adaptive_preselection_history_from_state(sequential_state)
        except SequentialVerificationContractError as exc:
            raise ClaimReleaseContractError(
                "adaptive_calibration_sequential_history_invalid"
            ) from exc
        selection_transitions_exist = any(
            transition.transition_kind == "selection"
            for transition in sequential_state.transitions
        )
        if (
            ledger_policy_context_sha256 is None
            or ledger_bundle_sha256 is None
        ):
            raise ClaimReleaseContractError(
                "adaptive_calibration_cannot_activate_after_state_genesis"
            )
        if selection_transitions_exist and not ledger_history:
            raise ClaimReleaseContractError(
                "adaptive_calibration_cannot_activate_after_nonadaptive_selection"
            )
        expected_historical_states = (
            adaptive_candidate.observed_states
            if sequential_state.session.active_action is not None
            else adaptive_candidate.observed_states[:-1]
        )
        history_checks = {
            "question": adaptive_candidate.question_id == question_id,
            "population": adaptive_candidate.population_id == population_id,
            "domain": adaptive_candidate.domain == domain,
            "observed_prefix": tuple(expected_historical_states) == ledger_history,
            "policy_context": ledger_policy_context_sha256
            == adaptive_candidate.policy_context_sha256,
            "calibration_bundle": ledger_bundle_sha256
            == adaptive_bundle.bundle_sha256,
        }
        failed_history_checks = sorted(
            name for name, passed in history_checks.items() if not passed
        )
        if failed_history_checks:
            raise ClaimReleaseContractError(
                "adaptive_release_candidate_history_mismatch:"
                f"{failed_history_checks}"
            )
        try:
            adaptive_assessment = assess_adaptive_release_candidate(
                adaptive_candidate,
                adaptive_bundle,
            )
        except AdaptiveCalibrationError as exc:
            raise ClaimReleaseContractError(
                f"adaptive_calibration_contract_failed:{exc}"
            ) from exc
        if sequential_state.session.active_action is not None:
            if adaptive_assessment.status == "released":
                raise ClaimReleaseContractError(
                    "adaptive_active_action_selected_after_qualifying_release"
                )
            active_payload = {
                "adaptive_active_action_block_version": "1",
                "bundle_sha256": adaptive_bundle.bundle_sha256,
                "candidate_sha256": adaptive_candidate.candidate_sha256,
                "sequential_state_sha256": sequential_state.state_sha256,
            }
            return CalibrationGateAssessment(
                status="abstained",
                reason="active_audit_action_unresolved_before_calibration",
                calibration_contract="adaptive-first-release-trajectory-v1",
                frozen_bundle_sha256=adaptive_bundle.bundle_sha256,
                release_candidate_sha256=adaptive_candidate.candidate_sha256,
                prospective_assessment_sha256=hash_canonical(active_payload),
                policy_context_sha256=adaptive_candidate.policy_context_sha256,
                label_source=adaptive_bundle.label_source,
                guarantee_scope=adaptive_bundle.guarantee_scope,
            )
        current = adaptive_candidate.observed_states[-1]
        exact_checks = {
            "question": adaptive_candidate.question_id == question_id,
            "population": adaptive_candidate.population_id == population_id,
            "domain": adaptive_candidate.domain == domain,
            "scheduler_state": current.scheduler_state_sha256
            == sequential_state.state_sha256,
            "audit_prefix": current.audit_prefix_item_ids
            == list(sequential_state.session.resolved_item_ids),
            "audit_cost": math.isclose(
                current.audit_prefix_cost_minutes,
                sequential_state.session.historical_realized_cost,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ),
            "graph": current.evidence_graph_sha256
            == hash_canonical(sequential_state.graph),
            "synthesis": current.synthesis_sha256
            == hash_canonical(sequential_state.synthesis),
            "assessment": current.non_calibration_assessment_sha256
            == noncalibration_assessment_sha256,
            "gates": current.non_calibration_gates_passed
            == noncalibration_gates_passed,
            "blockers": current.non_calibration_blocking_reasons
            == sorted(set(noncalibration_blocking_reasons)),
            "claim_decision": current.claim_decision == claim_decision,
            "features": current.score_features == dict(features),
        }
        failed = sorted(name for name, passed in exact_checks.items() if not passed)
        if failed:
            raise ClaimReleaseContractError(
                f"adaptive_release_candidate_current_state_mismatch:{failed}"
            )
        return CalibrationGateAssessment(
            status=adaptive_assessment.status,
            reason=adaptive_assessment.reason,
            calibration_contract="adaptive-first-release-trajectory-v1",
            frozen_bundle_sha256=adaptive_bundle.bundle_sha256,
            release_candidate_sha256=adaptive_candidate.candidate_sha256,
            prospective_assessment_sha256=adaptive_assessment.assessment_sha256,
            policy_context_sha256=adaptive_candidate.policy_context_sha256,
            scalar_risk_score=adaptive_assessment.scalar_risk_score,
            threshold=adaptive_assessment.threshold,
            label_source=adaptive_bundle.label_source,
            guarantee_scope=adaptive_bundle.guarantee_scope,
        )

    assert bundle is not None
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
    fixed_kwargs = {
        "calibration_contract": "fixed-single-decision-v2",
        "frozen_bundle_sha256": bundle.bundle_sha256,
        "label_source": bundle.label_source,
        "guarantee_scope": (
            "fixed single-decision label-risk policy; not valid after adaptive "
            "repeated looks"
        ),
    }
    if sequential_state is not None:
        try:
            validate_frozen_calibration_bundle_for_deployment(
                bundle,
                deployment_mode="adaptive_first_release_trajectory",
            )
        except CalibrationContractError as exc:
            fixed_state_reason = str(exc)
        else:  # pragma: no cover - defensive invariant for future bundle versions
            raise ClaimReleaseContractError(
                "fixed_state_bundle_unexpectedly_authorized_adaptive_trajectory"
            )
        return CalibrationGateAssessment(
            status="abstained",
            reason=fixed_state_reason,
            **fixed_kwargs,
        )
    if not paper_ids:
        return CalibrationGateAssessment(
            status="not_run",
            reason="no_matching_papers_for_release_candidate",
            **fixed_kwargs,
        )
    if bundle.label_source == "simulation":
        return CalibrationGateAssessment(
            status="abstained",
            reason="simulation_calibration_not_valid_for_scientific_release",
            **fixed_kwargs,
        )
    if domain not in bundle.calibration.domains:
        return CalibrationGateAssessment(
            status="abstained",
            reason="domain_shift_outside_frozen_calibration_support",
            **fixed_kwargs,
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
        calibration_contract="fixed-single-decision-v2",
        frozen_bundle_sha256=bundle.bundle_sha256,
        release_candidate_sha256=assessment.candidate_sha256,
        prospective_assessment_sha256=hash_canonical(assessment),
        scalar_risk_score=assessment.scalar_risk_score,
        threshold=assessment.threshold,
        label_source=bundle.label_source,
        guarantee_scope=(
            "fixed single-decision label-risk policy; not valid after adaptive "
            "repeated looks"
        ),
    )


def _assess_claim_release_impl(
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
    adaptive_calibration_bundle: AdaptiveCalibrationBundle | None = None,
    adaptive_release_candidate: ProspectiveAdaptiveReleaseCandidate | None = None,
    external_noncalibration_blocking_reasons: Sequence[str] = (),
    config: ClaimReleaseConfig | None = None,
    audit_guard_config: ReleaseGuardConfig | None = None,
    sequential_audit_state: SequentialVerificationState | None = None,
) -> ClaimReleaseAssessment:
    """Assess a target claim without accepting oracle or correctness labels.

    Audit candidates must cover the matching graph outcome-estimate identities exactly.
    Resolved items require external :class:`AuditResolutionReceipt` objects bound to the
    current evidence, graph, synthesis, and candidate snapshot. Group-average item-cell
    error-rate UCLs are scheduling/blocking features, never individual probabilities or
    residual claim-risk bounds; they cannot supersede influence/flip gates or authorize
    release. Budget selection alone never resolves an item, and receipts remain auditable
    declarations rather than proof of competence. Stateful release additionally requires
    a calibrated complete-question first-release trajectory.
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
    if not audit_guard_config.require_calibrated_item_scores:
        raise ClaimReleaseContractError(
            "release_guard_must_block_noncalibrated_unresolved_items"
        )
    if not audit_guard_config.require_item_cell_rate_ucls:
        raise ClaimReleaseContractError(
            "release_guard_must_require_cell_rate_ucls_for_unresolved_items"
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
        assumed_within_cohort_correlation=config.assumed_within_cohort_correlation,
        prespecified_moderators=config.prespecified_condition_moderators,
        condition_familywise_alpha=config.condition_familywise_alpha,
        condition_min_cohorts_per_level=config.condition_min_cohorts_per_level,
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
        sequential_state=sequential_audit_state,
    )
    features = _risk_features(
        graph_features=graph_feature_values,
        synthesis=synthesis,
        evidence=evidence,
        audit=audit,
    )
    feature_hash = hash_canonical(features)
    config_hash = hash_canonical(
        {
            "claim_release": config,
            "audit_guard": asdict(audit_guard_config),
        }
    )
    noncalibration_claim_reasons: list[str] = []
    if evidence.classification is not EvidenceClassification.SUPPORTED:
        noncalibration_claim_reasons.append(
            f"evidence:{evidence.classification.value}:{evidence.reason}"
        )
    if audit.status != "eligible":
        noncalibration_claim_reasons.extend(f"audit:{reason}" for reason in audit.reasons)
    all_noncalibration_blockers = sorted(
        set(noncalibration_claim_reasons)
        | set(external_noncalibration_blocking_reasons)
    )
    noncalibration_hash = noncalibration_assessment_sha256(
        question_id=question_id,
        target=target,
        pipeline_sha256=pipeline_sha256,
        evidence_graph_sha256=graph_hash,
        synthesis_sha256=synthesis_hash,
        config_sha256=config_hash,
        complete_matching_paper_ids=paper_ids,
        evidence=evidence,
        audit=audit,
        risk_features=features,
    )
    calibration = _calibration_gate(
        question_id=question_id,
        population_id=population_id,
        domain=domain,
        pipeline_sha256=pipeline_sha256,
        paper_ids=paper_ids,
        features=features,
        bundle=frozen_calibration_bundle,
        adaptive_bundle=adaptive_calibration_bundle,
        adaptive_candidate=adaptive_release_candidate,
        sequential_state=sequential_audit_state,
        noncalibration_assessment_sha256=noncalibration_hash,
        noncalibration_gates_passed=not all_noncalibration_blockers,
        noncalibration_blocking_reasons=all_noncalibration_blockers,
        claim_decision=evidence.classification.value,
    )

    reasons = list(noncalibration_claim_reasons)
    if calibration.status != "released":
        reasons.append(f"calibration:{calibration.reason}")
    status = ClaimReleaseStatus.RELEASED if not reasons else ClaimReleaseStatus.ABSTAINED
    payload: dict[str, Any] = {
        "assessment_version": "prospective-claim-release-v2",
        "question_id": question_id,
        "target": target,
        "pipeline_sha256": pipeline_sha256,
        "evidence_graph_sha256": graph_hash,
        "synthesis_sha256": synthesis_hash,
        "config_sha256": config_hash,
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
    adaptive_calibration_bundle: AdaptiveCalibrationBundle | None = None,
    adaptive_release_candidate: ProspectiveAdaptiveReleaseCandidate | None = None,
    external_noncalibration_blocking_reasons: Sequence[str] = (),
    config: ClaimReleaseConfig | None = None,
    audit_guard_config: ReleaseGuardConfig | None = None,
    sequential_audit_state: SequentialVerificationState | None = None,
) -> ClaimReleaseAssessment:
    """Assess a fixed decision; adaptive release is verifier-owned.

    A detached low-level call does not contain the manifest, pipeline verification,
    calibrated item-risk receipt, and predecessor scientific states needed to replay
    every adaptive checkpoint.  It therefore cannot safely authorize an adaptive
    first-release trajectory.  The unified verifier performs that replay and calls
    the private implementation only after the complete history has passed.
    """

    if (
        adaptive_calibration_bundle is not None
        or adaptive_release_candidate is not None
    ):
        raise ClaimReleaseContractError(
            "adaptive_release_requires_unified_verifier_history_replay"
        )
    return _assess_claim_release_impl(
        graph=graph,
        question_id=question_id,
        population_id=population_id,
        domain=domain,
        pipeline_sha256=pipeline_sha256,
        target=target,
        audit_candidates=audit_candidates,
        claim_model=claim_model,
        audit_resolution_receipts=audit_resolution_receipts,
        audit_budget=audit_budget,
        frozen_calibration_bundle=frozen_calibration_bundle,
        external_noncalibration_blocking_reasons=(
            external_noncalibration_blocking_reasons
        ),
        config=config,
        audit_guard_config=audit_guard_config,
        sequential_audit_state=sequential_audit_state,
    )


def assess_global_condition_claim_release_source(
    *,
    graph: EvidenceGraph,
    question_id: str,
    population_id: str,
    domain: str,
    pipeline_sha256: str,
    target: GlobalConditionDependenceTargetV1,
    condition_calibration_projection: ConditionCalibrationProjectionV1,
    condition_confirmation_gate: ConditionConfirmationGateAssessmentV1,
    audit_candidates: Sequence[AuditCandidate],
    claim_model: ClaimModel,
    audit_resolution_receipts: Sequence[AuditResolutionReceipt],
    audit_budget: float,
    condition_noncalibration_reasons: Sequence[str] = (),
    external_noncalibration_blocking_reasons: Sequence[str] = (),
    config: ClaimReleaseConfig | None = None,
    audit_guard_config: ReleaseGuardConfig | None = None,
    sequential_audit_state: SequentialVerificationState | None = None,
) -> ConditionClaimReleaseAssessmentV1:
    """Freeze the immutable, always-abstained source decision for manifest v3.

    The ordinary synthesis/audit calculations are delegated to the established v2
    assessor.  Their directional evidence label is retained only as exploratory
    development-corpus metadata.  The provisional five-way decision comes solely
    from the prospective global target projection; neither same-corpus moderator
    selection nor a v1 calibration bundle can authorize release.
    """

    try:
        projection = ConditionCalibrationProjectionV1.model_validate(
            condition_calibration_projection.model_dump(mode="json")
        )
        gate = ConditionConfirmationGateAssessmentV1.model_validate(
            condition_confirmation_gate.model_dump(mode="json")
        )
        target = GlobalConditionDependenceTargetV1.model_validate(
            target.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise ClaimReleaseContractError(
            "condition_claim_release_input_integrity_changed"
        ) from exc
    if (
        question_id != projection.question_id
        or projection.pipeline_sha256 != pipeline_sha256
        or projection.online_graph_sha256 != hash_canonical(graph)
        or projection.condition_target_sha256 != target.target_sha256
        or projection.prespecified_moderator_names != target.moderator_names
    ):
        raise ClaimReleaseContractError(
            "condition_claim_release_projection_context_mismatch"
        )
    if (
        not gate.required
        or gate.provisional_claim_decision != "condition_dependent"
        or gate.condition_projection_sha256 != projection.projection_sha256
        or gate.target_sha256 != target.target_sha256
        or gate.plan_sha256 != projection.plan_sha256
        or gate.config_sha256 != projection.confirmation_config_sha256
    ):
        raise ClaimReleaseContractError(
            "condition_claim_release_confirmation_gate_mismatch"
        )
    invalid_reasons = sorted(
        reason
        for reason in condition_noncalibration_reasons
        if not reason.strip() or reason.startswith("calibration:")
    )
    if invalid_reasons:
        raise ClaimReleaseContractError(
            f"condition_claim_release_noncalibration_reason_invalid:{invalid_reasons}"
        )

    directional_target = ClaimTarget(
        direction=target.reference_direction,
        outcome_name=target.outcome_name,
        contrast_id=target.contrast_id,
    )
    base = _assess_claim_release_impl(
        graph=graph,
        question_id=question_id,
        population_id=population_id,
        domain=domain,
        pipeline_sha256=pipeline_sha256,
        target=directional_target,
        audit_candidates=audit_candidates,
        claim_model=claim_model,
        audit_resolution_receipts=audit_resolution_receipts,
        audit_budget=audit_budget,
        frozen_calibration_bundle=None,
        adaptive_calibration_bundle=None,
        adaptive_release_candidate=None,
        external_noncalibration_blocking_reasons=(
            external_noncalibration_blocking_reasons
        ),
        config=config,
        audit_guard_config=audit_guard_config,
        sequential_audit_state=sequential_audit_state,
    )
    reasons = list(condition_noncalibration_reasons)
    if base.audit.status != "eligible":
        reasons.extend(f"audit:{reason}" for reason in base.audit.reasons)
    if gate.status == "missing":
        reasons.extend(
            [
                "condition_confirmation_required",
                "condition_dependent_confirmation_aware_calibration_required",
            ]
        )
    elif gate.status == "confirmed":
        reasons.append(
            "condition_dependent_confirmation_aware_calibration_required"
        )
    elif gate.status == "not_confirmed":
        reasons.append("condition_confirmation_not_confirmed")
    elif gate.status == "insufficient":
        reasons.append("condition_confirmation_insufficient")
    else:
        raise ClaimReleaseContractError(
            "condition_claim_release_confirmation_gate_status_invalid"
        )
    payload: dict[str, Any] = {
        "assessment_version": "condition-claim-release-source-v1",
        "question_id": question_id,
        "target": target,
        "pipeline_sha256": pipeline_sha256,
        "evidence_graph_sha256": base.evidence_graph_sha256,
        "synthesis_sha256": base.synthesis_sha256,
        "config_sha256": base.config_sha256,
        "paper_ids": base.paper_ids,
        "evidence": base.evidence,
        "audit": base.audit,
        "risk_feature_schema_version": "claim-release-risk-v1",
        "risk_features": base.risk_features,
        "risk_features_sha256": base.risk_features_sha256,
        "calibration": base.calibration,
        "condition_calibration_projection": projection,
        "condition_confirmation_gate": gate,
        "terminal_gate_deferred": gate.status == "missing",
        "status": ClaimReleaseStatus.ABSTAINED,
        "reasons": sorted(set(reasons)),
        "release_semantics": (
            "source decision only; condition-dependent release requires held-out "
            "confirmation and confirmation-aware complete-question calibration in v7"
        ),
    }
    return ConditionClaimReleaseAssessmentV1.model_validate(
        {**payload, "decision_sha256": hash_canonical(payload)}
    )


def _assess_claim_release_after_verifier_history_replay(
    **kwargs: Any,
) -> ClaimReleaseAssessment:
    """Verifier-internal adaptive assessment after whole-history replay."""

    return _assess_claim_release_impl(**kwargs)


def _assess_qualified_claim_release_impl(
    *,
    graph: EvidenceGraph,
    question_id: str,
    population_id: str,
    domain: str,
    pipeline_sha256: str,
    target: ClaimTargetV2,
    audit_candidates: Sequence[AuditCandidate],
    claim_model: ClaimModel,
    audit_resolution_receipts: Sequence[AuditResolutionReceipt],
    audit_budget: float,
    frozen_calibration_bundle: FrozenCalibrationBundle | None,
    adaptive_calibration_bundle: AdaptiveCalibrationBundle | None = None,
    adaptive_release_candidate: ProspectiveAdaptiveReleaseCandidate | None = None,
    external_noncalibration_blocking_reasons: Sequence[str] = (),
    config: ClaimReleaseConfig | None = None,
    audit_guard_config: ReleaseGuardConfig | None = None,
    sequential_audit_state: SequentialVerificationState | None = None,
) -> QualifiedClaimReleaseAssessment:
    """Join exact qualified synthesis to the same audit and calibration gates as v1.

    Only condition-matched estimates are audit candidates and calibration inputs.  The
    full graph remains hash-bound in the decision so excluded, missing, or type-invalid
    evidence is preserved rather than disappearing from provenance.
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
    if not audit_guard_config.require_calibrated_item_scores:
        raise ClaimReleaseContractError(
            "release_guard_must_block_noncalibrated_unresolved_items"
        )
    if not audit_guard_config.require_item_cell_rate_ucls:
        raise ClaimReleaseContractError(
            "release_guard_must_require_cell_rate_ucls_for_unresolved_items"
        )

    evidence_assessment = assess_qualified_claim_evidence(
        graph=graph,
        target=target,
        config=config,
    )
    synthesis = evidence_assessment.synthesis
    synthesis_hash = evidence_assessment.synthesis_sha256
    evidence = evidence_assessment.verdict
    expected_item_ids = list(evidence.matched_estimate_ids)
    estimates_by_id = {estimate.estimate_id: estimate for estimate in graph.outcome_estimates}
    if any(item_id not in estimates_by_id for item_id in expected_item_ids):
        raise ClaimReleaseContractError("qualified_claim_matched_estimate_unknown")
    paper_ids = sorted({estimates_by_id[item_id].effect.paper_id for item_id in expected_item_ids})
    candidate_ids = _validate_exact_audit_coverage(
        expected_item_ids=expected_item_ids,
        candidates=audit_candidates,
    )

    graph_hash = hash_canonical(graph)
    evidence_hashes = evidence_item_sha256s(graph)
    direct_baseline_decisions = {
        candidate.baseline_decision
        for candidate in audit_candidates
        if candidate.baseline_decision is not None
    }
    if len(direct_baseline_decisions) > 1:
        raise ClaimReleaseContractError("audit_candidate_baseline_decisions_mismatch")
    baseline_probability = claim_model.probability(
        [
            candidate.baseline_contribution
            for candidate in sorted(audit_candidates, key=lambda item: item.item_id)
        ]
    )
    baseline_conclusion = (
        next(iter(direct_baseline_decisions))
        if direct_baseline_decisions
        else claim_model.conclusion(baseline_probability)
    )
    if evidence.synthesis_gate_passed and not baseline_conclusion:
        raise ClaimReleaseContractError(
            "claim_model_baseline_conclusion_inconsistent_with_supported_target"
        )

    identity_context = {
        "pipeline_sha256": pipeline_sha256,
        "evidence_graph_sha256": graph_hash,
        "synthesis_sha256": synthesis_hash,
        "qualified_target_sha256": target.claim_sha256,
        "expected_item_ids": expected_item_ids,
        "paper_ids": paper_ids,
    }
    audit = _audit_gate(
        expected_item_ids=candidate_ids,
        candidates=audit_candidates,
        claim_model=claim_model,
        resolution_receipts=audit_resolution_receipts,
        evidence_item_sha256s=evidence_hashes,
        evidence_graph_sha256=graph_hash,
        synthesis_sha256=synthesis_hash,
        budget=audit_budget,
        config=config,
        guard_config=audit_guard_config,
        identity_context=identity_context,
        sequential_state=sequential_audit_state,
    )

    matched = set(expected_item_ids)
    active_graph = graph.model_copy(
        update={
            "outcome_estimates": [
                estimate for estimate in graph.outcome_estimates if estimate.estimate_id in matched
            ]
        }
    )
    graph_features = graph_risk_features(
        active_graph,
        outcome_name=target.outcome_name,
        contrast_id=target.contrast_id,
    ).as_calibration_features()
    classification_by_state = {
        QualifiedClaimVerdictState.PRESPECIFIED_SUPPORTED: EvidenceClassification.SUPPORTED,
        QualifiedClaimVerdictState.PRESPECIFIED_CONTRADICTED: EvidenceClassification.CONTRADICTED,
        QualifiedClaimVerdictState.PRESPECIFIED_INCONCLUSIVE: EvidenceClassification.INCONCLUSIVE,
        QualifiedClaimVerdictState.PRESPECIFIED_NOT_EVALUABLE: EvidenceClassification.NOT_EVALUABLE,
        QualifiedClaimVerdictState.DISCOVERED_HYPOTHESIS_ONLY: EvidenceClassification.INCONCLUSIVE,
    }
    feature_evidence = SynthesisEvidenceAssessment(
        target_direction=TargetDirection(target.direction.value),
        classification=classification_by_state[evidence.state],
        mode=evidence.mode,
        reason=evidence.reason,
        n_papers=len(paper_ids),
    )
    features = _risk_features(
        graph_features=graph_features,
        synthesis=synthesis,
        evidence=feature_evidence,
        audit=audit,
    )
    config_hash = hash_canonical(
        {
            "claim_release": config,
            "audit_guard": asdict(audit_guard_config),
            "qualified_target_sha256": target.claim_sha256,
        }
    )
    noncalibration_claim_reasons: list[str] = []
    if not evidence.synthesis_gate_passed:
        noncalibration_claim_reasons.append(
            f"evidence:{evidence.state.value}:{evidence.reason}"
        )
    if audit.status != "eligible":
        noncalibration_claim_reasons.extend(f"audit:{reason}" for reason in audit.reasons)
    all_noncalibration_blockers = sorted(
        set(noncalibration_claim_reasons)
        | set(external_noncalibration_blocking_reasons)
    )
    noncalibration_hash = noncalibration_assessment_sha256(
        question_id=question_id,
        target=target,
        pipeline_sha256=pipeline_sha256,
        evidence_graph_sha256=graph_hash,
        synthesis_sha256=synthesis_hash,
        config_sha256=config_hash,
        complete_matching_paper_ids=paper_ids,
        evidence=evidence,
        audit=audit,
        risk_features=features,
    )
    calibration = _calibration_gate(
        question_id=question_id,
        population_id=population_id,
        domain=domain,
        pipeline_sha256=pipeline_sha256,
        paper_ids=paper_ids,
        features=features,
        bundle=frozen_calibration_bundle,
        adaptive_bundle=adaptive_calibration_bundle,
        adaptive_candidate=adaptive_release_candidate,
        sequential_state=sequential_audit_state,
        noncalibration_assessment_sha256=noncalibration_hash,
        noncalibration_gates_passed=not all_noncalibration_blockers,
        noncalibration_blocking_reasons=all_noncalibration_blockers,
        claim_decision=evidence.state.value,
    )

    reasons = list(noncalibration_claim_reasons)
    if calibration.status != "released":
        reasons.append(f"calibration:{calibration.reason}")
    status = ClaimReleaseStatus.RELEASED if not reasons else ClaimReleaseStatus.ABSTAINED
    payload: dict[str, Any] = {
        "assessment_version": "prospective-qualified-claim-release-v2",
        "question_id": question_id,
        "target": target,
        "pipeline_sha256": pipeline_sha256,
        "evidence_graph_sha256": graph_hash,
        "synthesis_sha256": synthesis_hash,
        "config_sha256": config_hash,
        "paper_ids": paper_ids,
        "evidence": evidence,
        "audit": audit,
        "risk_feature_schema_version": "claim-release-risk-v1",
        "risk_features": features,
        "risk_features_sha256": hash_canonical(features),
        "calibration": calibration,
        "status": status,
        "reasons": reasons,
        "release_semantics": (
            "prespecified qualified claim passed evidence, audit, and calibration gates; "
            "not scientific truth"
        ),
    }
    return QualifiedClaimReleaseAssessment.model_validate(
        {**payload, "decision_sha256": hash_canonical(payload)}
    )


def assess_qualified_claim_release(
    *,
    graph: EvidenceGraph,
    question_id: str,
    population_id: str,
    domain: str,
    pipeline_sha256: str,
    target: ClaimTargetV2,
    audit_candidates: Sequence[AuditCandidate],
    claim_model: ClaimModel,
    audit_resolution_receipts: Sequence[AuditResolutionReceipt],
    audit_budget: float,
    frozen_calibration_bundle: FrozenCalibrationBundle | None,
    adaptive_calibration_bundle: AdaptiveCalibrationBundle | None = None,
    adaptive_release_candidate: ProspectiveAdaptiveReleaseCandidate | None = None,
    external_noncalibration_blocking_reasons: Sequence[str] = (),
    config: ClaimReleaseConfig | None = None,
    audit_guard_config: ReleaseGuardConfig | None = None,
    sequential_audit_state: SequentialVerificationState | None = None,
) -> QualifiedClaimReleaseAssessment:
    """Assess a fixed qualified decision; adaptive release is verifier-owned."""

    if (
        adaptive_calibration_bundle is not None
        or adaptive_release_candidate is not None
    ):
        raise ClaimReleaseContractError(
            "adaptive_release_requires_unified_verifier_history_replay"
        )
    return _assess_qualified_claim_release_impl(
        graph=graph,
        question_id=question_id,
        population_id=population_id,
        domain=domain,
        pipeline_sha256=pipeline_sha256,
        target=target,
        audit_candidates=audit_candidates,
        claim_model=claim_model,
        audit_resolution_receipts=audit_resolution_receipts,
        audit_budget=audit_budget,
        frozen_calibration_bundle=frozen_calibration_bundle,
        external_noncalibration_blocking_reasons=(
            external_noncalibration_blocking_reasons
        ),
        config=config,
        audit_guard_config=audit_guard_config,
        sequential_audit_state=sequential_audit_state,
    )


def _assess_qualified_claim_release_after_verifier_history_replay(
    **kwargs: Any,
) -> QualifiedClaimReleaseAssessment:
    """Verifier-internal qualified assessment after whole-history replay."""

    return _assess_qualified_claim_release_impl(**kwargs)


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
    "ConditionClaimReleaseAssessmentV1",
    "EvidenceClassification",
    "QualifiedClaimEvidenceAssessment",
    "QualifiedClaimReleaseAssessment",
    "SynthesisEvidenceAssessment",
    "TargetDirection",
    "assess_claim_release",
    "assess_global_condition_claim_release_source",
    "assess_qualified_claim_evidence",
    "assess_qualified_claim_release",
    "classify_qualified_synthesis_evidence",
    "evidence_item_sha256s",
    "freeze_audit_resolution_receipt",
]
