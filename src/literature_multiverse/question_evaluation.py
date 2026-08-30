"""Question-level replay evaluation for budgeted scientific-claim verification.

The benchmark unit is one complete, independent claim question. Policy ranking sees
only the frozen inputs attached to the current replay state. Expert verdicts and audit
dispositions are used only after selection for evaluation. Every selected audit prefix
that completes inside the deadline must have an externally generated pipeline replay
state. A deadline-truncated active action deliberately keeps the preceding scientific
state and blocks release; this module never fabricates a post-audit conclusion with an
additive approximation.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import platform
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from literature_multiverse.certificate import (
    ConditionVerificationCertificateV6,
    FinalConditionVerificationCertificateV7,
    VerificationCertificate,
)
from literature_multiverse.lineage import (
    atomic_write_jsonl,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    compute_pipeline_fingerprint,
)

_COST_TOLERANCE = 1e-9


class QuestionEvaluationContractError(ValueError):
    """The benchmark cannot support the requested leakage-free replay."""


class BenchmarkEvidenceKind(StrEnum):
    REAL_EXPERT_ADJUDICATED = "real_expert_adjudicated"
    SIMULATION = "simulation"
    DIAGNOSTIC = "diagnostic"


class BenchmarkSplit(StrEnum):
    TEST = "test"
    HELD_OUT_DOMAIN = "held_out_domain"
    PROSPECTIVE = "prospective"


class ReferenceVerdictSource(StrEnum):
    EXPERT_ADJUDICATION = "expert_adjudication"
    PLANTED_SIMULATION = "planted_simulation"
    DIAGNOSTIC_PROXY = "diagnostic_proxy"


class ReferenceClaimVerdictValue(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    CONDITION_DEPENDENT = "condition_dependent"
    INCONCLUSIVE = "inconclusive"
    NOT_EVALUABLE = "not_evaluable"


class AuditDisposition(StrEnum):
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"
    UNUSABLE = "unusable"


class AuditCostBasis(StrEnum):
    REALIZED_HUMAN_MINUTES = "realized_human_minutes"
    SIMULATED_MINUTES = "simulated_minutes"
    DIAGNOSTIC_MINUTES = "diagnostic_minutes"


class AuditCostAccounting(StrEnum):
    """What one duration includes; real efficiency always uses total labor."""

    TOTAL_PERSON_MINUTES = "total_person_minutes_across_all_reviewers_and_final_adjudication"
    SIMULATED_TOTAL_COST_MINUTES = "simulated_total_cost_minutes"
    DIAGNOSTIC_TOTAL_COST_MINUTES = "diagnostic_total_cost_minutes"


class ReplaySource(StrEnum):
    FROZEN_PIPELINE_RERUN = "frozen_pipeline_rerun"
    LEGACY_DECLARED_PIPELINE_RERUN = "legacy_declared_pipeline_rerun"
    PLANTED_SIMULATION = "planted_simulation"
    DIAGNOSTIC_APPROXIMATION = "diagnostic_approximation"


class ReplayReleaseStatus(StrEnum):
    RELEASED = "released"
    ABSTAINED = "abstained"


class ReplayStopReason(StrEnum):
    """Why a retrospective policy replay stopped selecting audit actions."""

    NO_AUDIT_POLICY = "no_audit_policy"
    ALL_ITEMS_RESOLVED = "all_items_resolved"
    FIXED_COUNT_REACHED = "fixed_count_reached"
    NO_ELIGIBLE_CANDIDATE_FITS_ESTIMATED_BUDGET = "no_eligible_candidate_fits_estimated_budget"
    BUDGET_EXHAUSTED_WITHOUT_ACTIVE_ACTION = "budget_exhausted_without_active_action"
    BUDGET_EXHAUSTED_WITH_ACTIVE_ACTION = "budget_exhausted_with_active_action"
    FIRST_FROZEN_RELEASE_ELIGIBLE_STATE = "first_frozen_release_eligible_state"


class ReplayStoppingRule(StrEnum):
    """Prespecified decision for whether auditing continues after safe release."""

    PRODUCTION_STOP_ON_RELEASE = "production_stop_on_first_frozen_release"
    ALLOCATE_TO_CAP_EXPERIMENTAL = "allocate_to_cap_experimental"


def _stopping_rule_contract(rule: ReplayStoppingRule) -> tuple[bool, str]:
    if rule is ReplayStoppingRule.PRODUCTION_STOP_ON_RELEASE:
        return (
            True,
            "Stop before selecting any further action when the current frozen verifier state "
            "is release-eligible; no future audit outcome, realized cost, or reference verdict "
            "is opened. This matches the public verifier's release stopping boundary.",
        )
    return (
        False,
        "Experimental controlled allocate-to-cap arm: continue selecting until the hard "
        "budget/fixed-count boundary even after an intermediate frozen state could release. "
        "This does not represent the public verifier's production stopping behavior.",
    )


class ReplayPolicy(StrEnum):
    RANDOM = "random"
    RISK_ONLY = "risk_only"
    DISAGREEMENT_ONLY = "disagreement_only"
    INFLUENCE_ONLY = "influence_only"
    RISK_X_INFLUENCE = "risk_x_influence"
    RISK_PER_COST = "risk_per_cost"
    DISAGREEMENT_PER_COST = "disagreement_per_cost"
    INFLUENCE_PER_COST = "influence_per_cost"
    RISK_X_INFLUENCE_PER_COST = "risk_x_influence_per_cost"
    FIXED_COUNT = "fixed_count"
    NO_AUDIT = "no_audit"
    AUDIT_ALL_UPPER_BOUND = "audit_all_upper_bound"


BUDGETED_REPLAY_POLICIES: tuple[ReplayPolicy, ...] = (
    ReplayPolicy.RANDOM,
    ReplayPolicy.RISK_ONLY,
    ReplayPolicy.DISAGREEMENT_ONLY,
    ReplayPolicy.INFLUENCE_ONLY,
    ReplayPolicy.RISK_X_INFLUENCE,
    ReplayPolicy.RISK_PER_COST,
    ReplayPolicy.DISAGREEMENT_PER_COST,
    ReplayPolicy.INFLUENCE_PER_COST,
    ReplayPolicy.RISK_X_INFLUENCE_PER_COST,
    ReplayPolicy.FIXED_COUNT,
    ReplayPolicy.NO_AUDIT,
)


def _validate_sha256(value: str, label: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"question_evaluation_sha256_invalid:{label}")
    return value


def _sorted_unique(values: list[str], label: str) -> list[str]:
    if values != sorted(set(values)):
        raise ValueError(f"question_evaluation_values_not_sorted_unique:{label}")
    return values


class PolicyInputProvenance(ContractModel):
    """Auditable declaration that policy scores were fit outside evaluation units."""

    artifact_sha256: str
    fit_question_ids: list[str] = Field(default_factory=list)
    fit_claim_ids: list[str] = Field(default_factory=list)
    fit_paper_ids: list[str] = Field(default_factory=list)
    score_pipeline_frozen_before_benchmark_labels: Literal[True] = True
    observes_reference_verdict: Literal[False] = False
    observes_future_audit_outcomes: Literal[False] = False
    provenance_semantics: Literal[
        "hash-bound declaration; not proof that the scoring pipeline avoided leakage"
    ] = "hash-bound declaration; not proof that the scoring pipeline avoided leakage"

    @field_validator("artifact_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "policy_input_provenance")

    @field_validator("fit_question_ids", "fit_claim_ids", "fit_paper_ids")
    @classmethod
    def validate_fit_ids(cls, value: list[str], info: Any) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError(f"policy_fit_identity_empty:{info.field_name}")
        return _sorted_unique(value, info.field_name)


class ReplayPolicyInput(ContractModel):
    """Only fields visible to an allocator at one replay state."""

    item_id: Annotated[str, Field(min_length=1)]
    canonical_order: Annotated[int, Field(ge=1)]
    risk_score: Annotated[float, Field(ge=0, le=1)]
    risk_basis: Literal["calibrated_cell_rate_ucl", "calibrated_score", "heuristic", "simulation"]
    disagreement_score: Annotated[float, Field(ge=0, le=1)]
    influence_score: Annotated[float, Field(ge=0, le=1)]
    estimated_minutes: Annotated[float, Field(gt=0)]
    eligible: bool = True
    ineligibility_reasons: list[str] = Field(default_factory=list)
    score_state_sha256: str

    @field_validator("risk_score", "disagreement_score", "influence_score", "estimated_minutes")
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("replay_policy_input_nonfinite")
        return value

    @field_validator("score_state_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "score_state")

    @field_validator("eligible", mode="before")
    @classmethod
    def validate_strict_eligible(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("replay_policy_input_eligible_must_be_boolean")
        return value

    @field_validator("ineligibility_reasons")
    @classmethod
    def validate_ineligibility_reasons(cls, value: list[str]) -> list[str]:
        if any(not reason.strip() for reason in value):
            raise ValueError("replay_policy_input_ineligibility_reason_empty")
        return _sorted_unique(value, "replay_policy_input_ineligibility_reasons")

    @model_validator(mode="after")
    def validate_eligibility(self) -> ReplayPolicyInput:
        if self.eligible == bool(self.ineligibility_reasons):
            raise ValueError("replay_policy_input_eligibility_reason_mismatch")
        return self


class ReferenceClaimVerdict(ContractModel):
    """Question-level reference outcome, never exposed to policy ranking."""

    verdict_version: Literal["question-reference-verdict-v1"] = "question-reference-verdict-v1"
    question_id: Annotated[str, Field(min_length=1)]
    claim_id: Annotated[str, Field(min_length=1)]
    verdict: ReferenceClaimVerdictValue
    source: ReferenceVerdictSource
    adjudicator_count: Annotated[int, Field(ge=1)]
    protocol_sha256: str
    artifact_sha256: str
    verdict_sha256: str

    @property
    def claim_supported(self) -> bool:
        return self.verdict is ReferenceClaimVerdictValue.SUPPORTED

    @field_validator("protocol_sha256", "artifact_sha256", "verdict_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_verdict(self) -> ReferenceClaimVerdict:
        if self.source is ReferenceVerdictSource.EXPERT_ADJUDICATION and self.adjudicator_count < 2:
            raise ValueError("expert_reference_verdict_requires_two_adjudicators")
        payload = self.model_dump(mode="json", exclude={"verdict_sha256"})
        if hash_canonical(payload) != self.verdict_sha256:
            raise ValueError("reference_verdict_hash_mismatch")
        return self


def freeze_reference_claim_verdict(
    *,
    question_id: str,
    claim_id: str,
    verdict: ReferenceClaimVerdictValue,
    source: ReferenceVerdictSource,
    adjudicator_count: int,
    protocol_sha256: str,
    artifact_sha256: str,
) -> ReferenceClaimVerdict:
    payload: dict[str, Any] = {
        "verdict_version": "question-reference-verdict-v1",
        "question_id": question_id,
        "claim_id": claim_id,
        "verdict": verdict,
        "source": source,
        "adjudicator_count": adjudicator_count,
        "protocol_sha256": protocol_sha256,
        "artifact_sha256": artifact_sha256,
    }
    return ReferenceClaimVerdict.model_validate(
        {**payload, "verdict_sha256": hash_canonical(payload)}
    )


class QuestionAuditEvent(ContractModel):
    """Completed item adjudication with its observed review cost."""

    event_version: Literal["question-audit-event-v2"] = "question-audit-event-v2"
    item_id: Annotated[str, Field(min_length=1)]
    disposition: AuditDisposition
    completed_at: datetime
    realized_minutes: Annotated[float, Field(gt=0)]
    cost_basis: AuditCostBasis
    cost_accounting: AuditCostAccounting
    adjudicator_count: Annotated[int, Field(ge=1)]
    protocol_sha256: str
    artifact_sha256: str
    correction_sha256: str | None = None
    event_sha256: str

    @field_validator("completed_at")
    @classmethod
    def validate_completed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("question_audit_event_timezone_required")
        return value

    @field_validator("realized_minutes")
    @classmethod
    def validate_minutes(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("question_audit_realized_minutes_nonfinite")
        return value

    @field_validator("protocol_sha256", "artifact_sha256", "correction_sha256", "event_sha256")
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        return _validate_sha256(value, info.field_name) if value is not None else None

    @model_validator(mode="after")
    def validate_event(self) -> QuestionAuditEvent:
        expected_accounting = {
            AuditCostBasis.REALIZED_HUMAN_MINUTES: (AuditCostAccounting.TOTAL_PERSON_MINUTES),
            AuditCostBasis.SIMULATED_MINUTES: (AuditCostAccounting.SIMULATED_TOTAL_COST_MINUTES),
            AuditCostBasis.DIAGNOSTIC_MINUTES: (AuditCostAccounting.DIAGNOSTIC_TOTAL_COST_MINUTES),
        }[self.cost_basis]
        if self.cost_accounting is not expected_accounting:
            raise ValueError("question_audit_cost_accounting_basis_mismatch")
        corrected = self.disposition is AuditDisposition.CORRECTED
        if corrected != (self.correction_sha256 is not None):
            raise ValueError("question_audit_correction_hash_mismatch")
        if self.cost_basis is AuditCostBasis.REALIZED_HUMAN_MINUTES and self.adjudicator_count < 2:
            raise ValueError("real_human_audit_requires_two_adjudicators")
        payload = self.model_dump(mode="json", exclude={"event_sha256"})
        if hash_canonical(payload) != self.event_sha256:
            raise ValueError("question_audit_event_hash_mismatch")
        return self


def freeze_question_audit_event(
    *,
    item_id: str,
    disposition: AuditDisposition,
    completed_at: datetime,
    realized_minutes: float,
    cost_basis: AuditCostBasis,
    adjudicator_count: int,
    protocol_sha256: str,
    artifact_sha256: str,
    correction_sha256: str | None = None,
) -> QuestionAuditEvent:
    completed = completed_at.isoformat()
    if completed.endswith("+00:00"):
        completed = f"{completed[:-6]}Z"
    payload: dict[str, Any] = {
        "event_version": "question-audit-event-v2",
        "item_id": item_id,
        "disposition": disposition,
        "completed_at": completed,
        "realized_minutes": realized_minutes,
        "cost_basis": cost_basis,
        "cost_accounting": {
            AuditCostBasis.REALIZED_HUMAN_MINUTES: (AuditCostAccounting.TOTAL_PERSON_MINUTES),
            AuditCostBasis.SIMULATED_MINUTES: (AuditCostAccounting.SIMULATED_TOTAL_COST_MINUTES),
            AuditCostBasis.DIAGNOSTIC_MINUTES: (AuditCostAccounting.DIAGNOSTIC_TOTAL_COST_MINUTES),
        }[AuditCostBasis(cost_basis)],
        "adjudicator_count": adjudicator_count,
        "protocol_sha256": protocol_sha256,
        "artifact_sha256": artifact_sha256,
        "correction_sha256": correction_sha256,
    }
    return QuestionAuditEvent.model_validate({**payload, "event_sha256": hash_canonical(payload)})


def _production_claim_classification(certificate: VerificationCertificate) -> str:
    """Project either release-assessment family onto the benchmark verdict vocabulary."""

    evidence = certificate.production_stop_decision.release_assessment.evidence
    classification = getattr(evidence, "classification", None)
    if classification is not None:
        return str(classification.value)
    state = str(evidence.state.value)
    return {
        "prespecified_supported": "supported",
        "prespecified_contradicted": "contradicted",
        "prespecified_inconclusive": "inconclusive",
        "prespecified_not_evaluable": "not_evaluable",
        "discovered_hypothesis_only": "inconclusive",
    }[state]


def _production_release_reasons(certificate: VerificationCertificate) -> list[str]:
    decision = certificate.production_stop_decision
    return sorted(set(decision.release_assessment.reasons) | set(decision.blocking_adapter_reasons))


def _production_policy_inputs(
    certificate: VerificationCertificate | ConditionVerificationCertificateV6,
) -> list[ReplayPolicyInput]:
    """Derive every policy-visible field from one validated preselection state."""

    decision = certificate.production_stop_decision
    state = decision.evaluated_state
    if state is None:
        raise QuestionEvaluationContractError("production_replay_requires_stateful_certificate")
    if state.session.cost_unit != "person_minutes":
        raise QuestionEvaluationContractError("production_replay_requires_person_minutes_cost_unit")
    scientific_by_id = {str(row.get("item_id")): row for row in certificate.audit_candidates}
    current_by_id = {row.item_id: row for row in state.candidates}
    ranking_by_id = {row.item_id: row for row in decision.release_assessment.audit.ranking}
    candidate_ids = decision.release_assessment.audit.candidate_item_ids
    expected_ids = set(candidate_ids)
    if (
        set(scientific_by_id) != expected_ids
        or set(current_by_id) != expected_ids
        or set(ranking_by_id) != expected_ids
    ):
        raise QuestionEvaluationContractError("production_replay_candidate_identity_mismatch")
    order_by_id = {item_id: index for index, item_id in enumerate(sorted(candidate_ids), start=1)}
    risk_basis_by_value = {
        "calibrated_cell_rate_ucl": "calibrated_cell_rate_ucl",
        "calibrated": "calibrated_score",
        "heuristic": "heuristic",
        "planted_simulation": "simulation",
    }
    inputs: list[ReplayPolicyInput] = []
    for item_id in decision.release_assessment.audit.unresolved_item_ids:
        scientific = scientific_by_id[item_id]
        current = current_by_id[item_id]
        ranking = ranking_by_id[item_id]
        probability_basis = scientific.get("probability_basis")
        if probability_basis not in risk_basis_by_value:
            raise QuestionEvaluationContractError(
                f"production_replay_probability_basis_invalid:{item_id}"
            )
        if scientific.get("cost_unit") != "person_minutes":
            raise QuestionEvaluationContractError(
                f"production_replay_candidate_cost_unit_invalid:{item_id}"
            )
        inputs.append(
            ReplayPolicyInput(
                item_id=item_id,
                canonical_order=order_by_id[item_id],
                risk_score=scientific.get("error_probability"),
                risk_basis=risk_basis_by_value[probability_basis],
                disagreement_score=scientific.get("disagreement_score"),
                influence_score=ranking.probability_influence,
                estimated_minutes=scientific.get("verification_cost"),
                eligible=current.eligible,
                ineligibility_reasons=current.ineligibility_reasons,
                score_state_sha256=state.synthesis_sha256,
            )
        )
    return sorted(inputs, key=lambda row: row.item_id)


class ProductionReplayBinding(ContractModel):
    """Self-contained projection of a validated v5 production certificate.

    The complete certificate is embedded deliberately.  A copied status or digest alone
    cannot prove which preselection state production evaluated, whether an action was
    already active, or whether an adapter blocker prevented release.
    """

    binding_version: Literal["production-certificate-replay-v2"] = (
        "production-certificate-replay-v2"
    )
    certificate: VerificationCertificate
    certificate_sha256: str
    production_stop_decision_sha256: str
    evaluated_sequential_state_sha256: str
    evaluated_session_sha256: str
    evaluated_transition_ledger_sha256: str
    evaluated_audit_prefix: list[str]
    evaluated_selected_item_ids: list[str]
    evaluated_active_action_item_id: str | None
    blocking_adapter_reasons: list[str]
    full_release_eligible: bool
    production_outcome: str
    release_decision_sha256: str
    release_status: ReplayReleaseStatus
    claim_classification: Literal[
        "supported",
        "contradicted",
        "condition_dependent",
        "inconclusive",
        "not_evaluable",
    ]
    release_reasons: list[str]
    source_evidence_graph_sha256: str
    current_evidence_graph_sha256: str
    current_synthesis_sha256: str
    pipeline_sha256: str
    pipeline_verification_sha256: str
    source_current_graph_lineage: dict[str, str]
    source_current_graph_lineage_sha256: str
    binding_sha256: str

    @field_validator(
        "certificate_sha256",
        "production_stop_decision_sha256",
        "evaluated_sequential_state_sha256",
        "evaluated_session_sha256",
        "evaluated_transition_ledger_sha256",
        "release_decision_sha256",
        "source_evidence_graph_sha256",
        "current_evidence_graph_sha256",
        "current_synthesis_sha256",
        "pipeline_sha256",
        "pipeline_verification_sha256",
        "source_current_graph_lineage_sha256",
        "binding_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("full_release_eligible", mode="before")
    @classmethod
    def validate_strict_full_release(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("production_replay_full_release_must_be_boolean")
        return value

    @model_validator(mode="after")
    def validate_binding(self) -> ProductionReplayBinding:
        # Additive subclasses carry their own exact certificate-version validator.
        # Returning here prevents a composed v8 binding from being coerced through
        # the legacy v5 projection while retaining the mature common field surface.
        if self.binding_version != "production-certificate-replay-v2":
            return self
        certificate = self.certificate
        if certificate.certificate_version != "literature-multiverse-verification-v5":
            raise ValueError("production_replay_requires_certificate_v5")
        decision = certificate.production_stop_decision
        state = decision.evaluated_state
        if state is None:
            raise ValueError("production_replay_requires_stateful_stop_decision")
        active_item_id = (
            state.session.active_action.item_id if state.session.active_action is not None else None
        )
        expected_release_status = (
            ReplayReleaseStatus.RELEASED
            if decision.full_release_eligible
            else ReplayReleaseStatus.ABSTAINED
        )
        expected_lineage = certificate.current_state_hashes
        expected = {
            "certificate_sha256": certificate.certificate_sha256,
            "production_stop_decision_sha256": decision.decision_sha256,
            "evaluated_sequential_state_sha256": state.state_sha256,
            "evaluated_session_sha256": state.session.session_sha256,
            "evaluated_transition_ledger_sha256": hash_canonical(state.transitions),
            "evaluated_audit_prefix": list(state.session.resolved_item_ids),
            "evaluated_selected_item_ids": list(state.session.selected_item_ids),
            "evaluated_active_action_item_id": active_item_id,
            "blocking_adapter_reasons": decision.blocking_adapter_reasons,
            "full_release_eligible": decision.full_release_eligible,
            "production_outcome": decision.outcome,
            "release_decision_sha256": decision.release_assessment.decision_sha256,
            "release_status": expected_release_status,
            "claim_classification": _production_claim_classification(certificate),
            "release_reasons": _production_release_reasons(certificate),
            "source_evidence_graph_sha256": certificate.source_evidence_graph_sha256,
            "current_evidence_graph_sha256": certificate.evidence_graph_sha256,
            "current_synthesis_sha256": certificate.synthesis_sha256,
            "pipeline_sha256": decision.release_assessment.pipeline_sha256,
            "pipeline_verification_sha256": (certificate.pipeline_verification.verification_sha256),
            "source_current_graph_lineage": expected_lineage,
            "source_current_graph_lineage_sha256": hash_canonical(expected_lineage),
        }
        observed = self.model_dump(
            mode="python",
            exclude={"binding_version", "certificate", "binding_sha256"},
        )
        for field_name, expected_value in expected.items():
            if observed[field_name] != expected_value:
                raise ValueError(f"production_replay_binding_mismatch:{field_name}")
        payload = self.model_dump(mode="json", exclude={"binding_sha256"})
        if hash_canonical(payload) != self.binding_sha256:
            raise ValueError("production_replay_binding_hash_mismatch")
        return self


def freeze_production_replay_binding(
    certificate: VerificationCertificate,
) -> ProductionReplayBinding:
    """Freeze the complete production decision needed by retrospective stopping."""

    certificate = VerificationCertificate.model_validate(certificate.model_dump(mode="json"))
    decision = certificate.production_stop_decision
    state = decision.evaluated_state
    if state is None:
        raise QuestionEvaluationContractError("production_replay_requires_stateful_v5_certificate")
    active_item_id = (
        state.session.active_action.item_id if state.session.active_action is not None else None
    )
    release_status = (
        ReplayReleaseStatus.RELEASED
        if decision.full_release_eligible
        else ReplayReleaseStatus.ABSTAINED
    )
    lineage = certificate.current_state_hashes
    payload: dict[str, Any] = {
        "binding_version": "production-certificate-replay-v2",
        "certificate": certificate,
        "certificate_sha256": certificate.certificate_sha256,
        "production_stop_decision_sha256": decision.decision_sha256,
        "evaluated_sequential_state_sha256": state.state_sha256,
        "evaluated_session_sha256": state.session.session_sha256,
        "evaluated_transition_ledger_sha256": hash_canonical(state.transitions),
        "evaluated_audit_prefix": list(state.session.resolved_item_ids),
        "evaluated_selected_item_ids": list(state.session.selected_item_ids),
        "evaluated_active_action_item_id": active_item_id,
        "blocking_adapter_reasons": decision.blocking_adapter_reasons,
        "full_release_eligible": decision.full_release_eligible,
        "production_outcome": decision.outcome,
        "release_decision_sha256": decision.release_assessment.decision_sha256,
        "release_status": release_status,
        "claim_classification": _production_claim_classification(certificate),
        "release_reasons": _production_release_reasons(certificate),
        "source_evidence_graph_sha256": certificate.source_evidence_graph_sha256,
        "current_evidence_graph_sha256": certificate.evidence_graph_sha256,
        "current_synthesis_sha256": certificate.synthesis_sha256,
        "pipeline_sha256": decision.release_assessment.pipeline_sha256,
        "pipeline_verification_sha256": (certificate.pipeline_verification.verification_sha256),
        "source_current_graph_lineage": lineage,
        "source_current_graph_lineage_sha256": hash_canonical(lineage),
    }
    return ProductionReplayBinding.model_validate(
        {**payload, "binding_sha256": hash_canonical(payload)}
    )


class ConditionProductionReplayBindingV7(ContractModel):
    """Self-contained exact-decision projection of a final condition v7 certificate."""

    binding_version: Literal["condition-production-certificate-replay-v7"] = (
        "condition-production-certificate-replay-v7"
    )
    certificate: FinalConditionVerificationCertificateV7
    certificate_sha256: str
    source_v6_certificate_sha256: str
    terminal_gate_result_sha256: str
    production_stop_decision_sha256: str
    evaluated_sequential_state_sha256: str
    evaluated_session_sha256: str
    evaluated_transition_ledger_sha256: str
    evaluated_audit_prefix: list[str]
    evaluated_selected_item_ids: list[str]
    evaluated_active_action_item_id: str | None
    policy_inputs: list[ReplayPolicyInput]
    policy_inputs_sha256: str
    blocking_adapter_reasons: list[str]
    full_release_eligible: bool
    production_outcome: Literal["final_condition_v7_join"] = "final_condition_v7_join"
    release_decision_sha256: str
    release_status: ReplayReleaseStatus
    claim_classification: Literal["condition_dependent"] = "condition_dependent"
    release_reasons: list[str]
    source_evidence_graph_sha256: str
    current_evidence_graph_sha256: str
    current_synthesis_sha256: str
    pipeline_sha256: str
    pipeline_verification_sha256: str
    source_current_graph_lineage: dict[str, str]
    source_current_graph_lineage_sha256: str
    binding_sha256: str

    @field_validator(
        "certificate_sha256",
        "source_v6_certificate_sha256",
        "terminal_gate_result_sha256",
        "production_stop_decision_sha256",
        "evaluated_sequential_state_sha256",
        "evaluated_session_sha256",
        "evaluated_transition_ledger_sha256",
        "policy_inputs_sha256",
        "release_decision_sha256",
        "source_evidence_graph_sha256",
        "current_evidence_graph_sha256",
        "current_synthesis_sha256",
        "pipeline_sha256",
        "pipeline_verification_sha256",
        "source_current_graph_lineage_sha256",
        "binding_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("full_release_eligible", mode="before")
    @classmethod
    def validate_release_boolean(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("condition_v7_replay_release_must_be_boolean")
        return value

    @model_validator(mode="after")
    def validate_binding(self) -> ConditionProductionReplayBindingV7:
        # The receipt-bound v9 subclass validates its exact terminal join below;
        # it must never be reinterpreted as the legacy v7 binding.
        if self.binding_version != "condition-production-certificate-replay-v7":
            return self
        certificate = self.certificate
        source = certificate.source_certificate_v6
        decision = source.production_stop_decision
        state = decision.evaluated_state
        active_item_id = (
            state.session.active_action.item_id if state.session.active_action is not None else None
        )
        expected_release_status = (
            ReplayReleaseStatus.RELEASED
            if certificate.status == "released"
            else ReplayReleaseStatus.ABSTAINED
        )
        expected_policy_inputs = _production_policy_inputs(source)
        lineage = {
            "development_evidence_graph": source.development_evidence_graph_sha256,
            "source_evidence_graph": source.source_evidence_graph_sha256,
            "source_v6_certificate": source.certificate_sha256,
            "terminal_gate_result": certificate.terminal_gate_result_sha256,
        }
        expected = {
            "certificate_sha256": certificate.certificate_sha256,
            "source_v6_certificate_sha256": source.certificate_sha256,
            "terminal_gate_result_sha256": certificate.terminal_gate_result_sha256,
            "production_stop_decision_sha256": decision.decision_sha256,
            "evaluated_sequential_state_sha256": state.state_sha256,
            "evaluated_session_sha256": state.session.session_sha256,
            "evaluated_transition_ledger_sha256": hash_canonical(state.transitions),
            "evaluated_audit_prefix": list(state.session.resolved_item_ids),
            "evaluated_selected_item_ids": list(state.session.selected_item_ids),
            "evaluated_active_action_item_id": active_item_id,
            # ``observed`` is a recursive ``model_dump(mode="python")`` below,
            # so nested contract models are dictionaries at this comparison
            # boundary.  Compare like with like; model instances are not equal
            # to their dumped mappings even when every field is identical.
            "policy_inputs": [row.model_dump(mode="python") for row in expected_policy_inputs],
            "policy_inputs_sha256": hash_canonical(expected_policy_inputs),
            "blocking_adapter_reasons": decision.blocking_adapter_reasons,
            "full_release_eligible": certificate.status == "released",
            "production_outcome": "final_condition_v7_join",
            "release_decision_sha256": certificate.release_assessment.decision_sha256,
            "release_status": expected_release_status,
            "claim_classification": "condition_dependent",
            "release_reasons": certificate.reasons,
            "source_evidence_graph_sha256": source.source_evidence_graph_sha256,
            "current_evidence_graph_sha256": state.graph_sha256,
            "current_synthesis_sha256": state.synthesis_sha256,
            "pipeline_sha256": source.release_assessment.pipeline_sha256,
            "pipeline_verification_sha256": (source.pipeline_verification.verification_sha256),
            "source_current_graph_lineage": lineage,
            "source_current_graph_lineage_sha256": hash_canonical(lineage),
        }
        observed = self.model_dump(
            mode="python",
            exclude={"binding_version", "certificate", "binding_sha256"},
        )
        for field_name, expected_value in expected.items():
            if observed[field_name] != expected_value:
                raise ValueError(f"condition_v7_production_replay_binding_mismatch:{field_name}")
        if certificate.status == "released" and active_item_id is not None:
            raise ValueError("condition_v7_release_has_active_audit_action")
        payload = self.model_dump(mode="json", exclude={"binding_sha256"})
        if hash_canonical(payload) != self.binding_sha256:
            raise ValueError("condition_v7_production_replay_binding_hash_mismatch")
        return self


def freeze_condition_production_replay_binding_v7(
    certificate: FinalConditionVerificationCertificateV7,
) -> ConditionProductionReplayBindingV7:
    """Freeze a final v7 certificate into an exact five-way replay binding."""

    try:
        certificate = FinalConditionVerificationCertificateV7.model_validate(
            certificate.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise QuestionEvaluationContractError(
            "condition_v7_production_certificate_integrity_changed"
        ) from exc
    source = certificate.source_certificate_v6
    decision = source.production_stop_decision
    state = decision.evaluated_state
    active_item_id = (
        state.session.active_action.item_id if state.session.active_action is not None else None
    )
    policy_inputs = _production_policy_inputs(source)
    lineage = {
        "development_evidence_graph": source.development_evidence_graph_sha256,
        "source_evidence_graph": source.source_evidence_graph_sha256,
        "source_v6_certificate": source.certificate_sha256,
        "terminal_gate_result": certificate.terminal_gate_result_sha256,
    }
    payload: dict[str, Any] = {
        "binding_version": "condition-production-certificate-replay-v7",
        "certificate": certificate,
        "certificate_sha256": certificate.certificate_sha256,
        "source_v6_certificate_sha256": source.certificate_sha256,
        "terminal_gate_result_sha256": certificate.terminal_gate_result_sha256,
        "production_stop_decision_sha256": decision.decision_sha256,
        "evaluated_sequential_state_sha256": state.state_sha256,
        "evaluated_session_sha256": state.session.session_sha256,
        "evaluated_transition_ledger_sha256": hash_canonical(state.transitions),
        "evaluated_audit_prefix": list(state.session.resolved_item_ids),
        "evaluated_selected_item_ids": list(state.session.selected_item_ids),
        "evaluated_active_action_item_id": active_item_id,
        "policy_inputs": policy_inputs,
        "policy_inputs_sha256": hash_canonical(policy_inputs),
        "blocking_adapter_reasons": decision.blocking_adapter_reasons,
        "full_release_eligible": certificate.status == "released",
        "production_outcome": "final_condition_v7_join",
        "release_decision_sha256": certificate.release_assessment.decision_sha256,
        "release_status": (
            ReplayReleaseStatus.RELEASED
            if certificate.status == "released"
            else ReplayReleaseStatus.ABSTAINED
        ),
        "claim_classification": "condition_dependent",
        "release_reasons": certificate.reasons,
        "source_evidence_graph_sha256": source.source_evidence_graph_sha256,
        "current_evidence_graph_sha256": state.graph_sha256,
        "current_synthesis_sha256": state.synthesis_sha256,
        "pipeline_sha256": source.release_assessment.pipeline_sha256,
        "pipeline_verification_sha256": (source.pipeline_verification.verification_sha256),
        "source_current_graph_lineage": lineage,
        "source_current_graph_lineage_sha256": hash_canonical(lineage),
    }
    return ConditionProductionReplayBindingV7.model_validate(
        {**payload, "binding_sha256": hash_canonical(payload)}
    )


class QuestionReplayState(ContractModel):
    """Actual frozen pipeline result after one exact ordered audit prefix."""

    replay_version: Literal["question-replay-state-v5"] = "question-replay-state-v5"
    question_id: Annotated[str, Field(min_length=1)]
    pipeline_sha256: str
    audit_sequence: list[str]
    policy_inputs: list[ReplayPolicyInput]
    release_status: ReplayReleaseStatus
    claim_classification: Literal[
        "supported",
        "contradicted",
        "condition_dependent",
        "inconclusive",
        "not_evaluable",
    ]
    release_reasons: list[str]
    graph_sha256: str
    synthesis_sha256: str
    release_assessment_sha256: str
    replay_source: ReplaySource
    production_binding: ProductionReplayBinding | ConditionProductionReplayBindingV7 | None = None
    replay_sha256: str

    @field_validator(
        "pipeline_sha256",
        "graph_sha256",
        "synthesis_sha256",
        "release_assessment_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("replay_sha256")
    @classmethod
    def validate_replay_hash(cls, value: str) -> str:
        return _validate_sha256(value, "replay")

    @model_validator(mode="after")
    def validate_state(self) -> QuestionReplayState:
        if len(self.audit_sequence) != len(set(self.audit_sequence)):
            raise ValueError("replay_audit_sequence_duplicate")
        input_ids = [row.item_id for row in self.policy_inputs]
        if input_ids != sorted(set(input_ids)):
            raise ValueError("replay_policy_inputs_must_be_sorted_unique")
        orders = [row.canonical_order for row in self.policy_inputs]
        if len(orders) != len(set(orders)):
            raise ValueError("replay_policy_canonical_orders_duplicate")
        if any(row.score_state_sha256 != self.synthesis_sha256 for row in self.policy_inputs):
            raise ValueError("replay_policy_input_state_hash_mismatch")
        released = self.release_status is ReplayReleaseStatus.RELEASED
        if released and self.release_reasons:
            raise ValueError("released_replay_state_cannot_have_reasons")
        if not released and not self.release_reasons:
            raise ValueError("abstained_replay_state_requires_reason")
        if self.release_reasons != list(dict.fromkeys(self.release_reasons)):
            raise ValueError("replay_release_reasons_duplicate")
        if self.replay_source is ReplaySource.FROZEN_PIPELINE_RERUN:
            binding = self.production_binding
            if binding is None:
                raise ValueError("production_replay_binding_required")
            if isinstance(binding, ConditionProductionReplayBindingV7):
                expected_question_id = (
                    binding.certificate.source_certificate_v6.release_assessment.question_id
                )
                expected_policy_inputs = binding.policy_inputs
            else:
                expected_question_id = (
                    binding.certificate.production_stop_decision.release_assessment.question_id
                )
                expected_policy_inputs = _production_policy_inputs(binding.certificate)
            if (
                self.question_id != expected_question_id
                or self.pipeline_sha256 != binding.pipeline_sha256
                or self.audit_sequence != binding.evaluated_audit_prefix
                or self.policy_inputs != expected_policy_inputs
                or self.release_status is not binding.release_status
                or self.claim_classification != binding.claim_classification
                or self.release_reasons != binding.release_reasons
                or self.graph_sha256 != binding.current_evidence_graph_sha256
                or self.synthesis_sha256 != binding.current_synthesis_sha256
                or self.release_assessment_sha256 != binding.release_decision_sha256
            ):
                raise ValueError("production_replay_state_projection_mismatch")
        elif self.production_binding is not None:
            raise ValueError("nonproduction_replay_forbids_production_binding")
        payload = self.model_dump(mode="json", exclude={"replay_sha256"})
        if hash_canonical(payload) != self.replay_sha256:
            raise ValueError("question_replay_state_hash_mismatch")
        return self


def freeze_question_replay_state(
    *,
    question_id: str,
    pipeline_sha256: str,
    audit_sequence: Sequence[str],
    policy_inputs: Sequence[ReplayPolicyInput | Mapping[str, Any]],
    release_status: ReplayReleaseStatus,
    claim_classification: str,
    release_reasons: Sequence[str],
    graph_sha256: str,
    synthesis_sha256: str,
    release_assessment_sha256: str,
    replay_source: ReplaySource,
) -> QuestionReplayState:
    if ReplaySource(replay_source) is ReplaySource.FROZEN_PIPELINE_RERUN:
        raise QuestionEvaluationContractError("frozen_pipeline_rerun_requires_certificate_factory")
    inputs = [
        row if isinstance(row, ReplayPolicyInput) else ReplayPolicyInput.model_validate(row)
        for row in policy_inputs
    ]
    inputs.sort(key=lambda row: row.item_id)
    payload: dict[str, Any] = {
        "replay_version": "question-replay-state-v5",
        "question_id": question_id,
        "pipeline_sha256": pipeline_sha256,
        "audit_sequence": list(audit_sequence),
        "policy_inputs": inputs,
        "release_status": release_status,
        "claim_classification": claim_classification,
        "release_reasons": list(release_reasons),
        "graph_sha256": graph_sha256,
        "synthesis_sha256": synthesis_sha256,
        "release_assessment_sha256": release_assessment_sha256,
        "replay_source": replay_source,
        "production_binding": None,
    }
    return QuestionReplayState.model_validate({**payload, "replay_sha256": hash_canonical(payload)})


def freeze_question_replay_state_from_certificate(
    certificate: VerificationCertificate | FinalConditionVerificationCertificateV7,
) -> QuestionReplayState:
    """Mechanically project one validated v5 or final condition-v7 decision."""

    if isinstance(certificate, FinalConditionVerificationCertificateV7):
        binding_v7 = freeze_condition_production_replay_binding_v7(certificate)
        source = binding_v7.certificate.source_certificate_v6
        payload_v7: dict[str, Any] = {
            "replay_version": "question-replay-state-v5",
            "question_id": source.release_assessment.question_id,
            "pipeline_sha256": binding_v7.pipeline_sha256,
            "audit_sequence": binding_v7.evaluated_audit_prefix,
            "policy_inputs": binding_v7.policy_inputs,
            "release_status": binding_v7.release_status,
            "claim_classification": binding_v7.claim_classification,
            "release_reasons": binding_v7.release_reasons,
            "graph_sha256": binding_v7.current_evidence_graph_sha256,
            "synthesis_sha256": binding_v7.current_synthesis_sha256,
            "release_assessment_sha256": binding_v7.release_decision_sha256,
            "replay_source": ReplaySource.FROZEN_PIPELINE_RERUN,
            "production_binding": binding_v7,
        }
        return QuestionReplayState.model_validate(
            {
                **payload_v7,
                "replay_sha256": hash_canonical(payload_v7),
            }
        )

    binding = freeze_production_replay_binding(certificate)
    payload: dict[str, Any] = {
        "replay_version": "question-replay-state-v5",
        "question_id": binding.certificate.claim_manifest["question_id"],
        "pipeline_sha256": binding.pipeline_sha256,
        "audit_sequence": binding.evaluated_audit_prefix,
        "policy_inputs": _production_policy_inputs(binding.certificate),
        "release_status": binding.release_status,
        "claim_classification": binding.claim_classification,
        "release_reasons": binding.release_reasons,
        "graph_sha256": binding.current_evidence_graph_sha256,
        "synthesis_sha256": binding.current_synthesis_sha256,
        "release_assessment_sha256": binding.release_decision_sha256,
        "replay_source": ReplaySource.FROZEN_PIPELINE_RERUN,
        "production_binding": binding,
    }
    return QuestionReplayState.model_validate({**payload, "replay_sha256": hash_canonical(payload)})


class ClaimQuestionBenchmarkRecord(ContractModel):
    """One immutable, independently adjudicated claim-question replay unit."""

    record_version: Literal["claim-question-benchmark-v3"] = "claim-question-benchmark-v3"
    question_id: Annotated[str, Field(min_length=1)]
    claim_id: Annotated[str, Field(min_length=1)]
    domain: Annotated[str, Field(min_length=1)]
    population_id: Annotated[str, Field(min_length=1)]
    split: BenchmarkSplit
    evidence_kind: BenchmarkEvidenceKind
    pipeline_sha256: str
    corpus_sha256: str
    paper_ids: Annotated[list[str], Field(min_length=1)]
    cohort_ids: Annotated[list[str], Field(min_length=1)]
    policy_input_provenance: PolicyInputProvenance
    reference_verdict: ReferenceClaimVerdict
    audit_events: Annotated[list[QuestionAuditEvent], Field(min_length=1)]
    replay_states: Annotated[list[QuestionReplayState], Field(min_length=2)]
    record_sha256: str

    @field_validator("pipeline_sha256", "corpus_sha256", "record_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("paper_ids", "cohort_ids")
    @classmethod
    def validate_ids(cls, value: list[str], info: Any) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError(f"benchmark_identity_empty:{info.field_name}")
        return _sorted_unique(value, info.field_name)

    @model_validator(mode="after")
    def validate_record(self) -> ClaimQuestionBenchmarkRecord:
        if (
            self.reference_verdict.question_id != self.question_id
            or self.reference_verdict.claim_id != self.claim_id
        ):
            raise ValueError("benchmark_reference_identity_mismatch")
        expected_reference_source = {
            BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED: (
                ReferenceVerdictSource.EXPERT_ADJUDICATION
            ),
            BenchmarkEvidenceKind.SIMULATION: ReferenceVerdictSource.PLANTED_SIMULATION,
            BenchmarkEvidenceKind.DIAGNOSTIC: ReferenceVerdictSource.DIAGNOSTIC_PROXY,
        }[self.evidence_kind]
        if self.reference_verdict.source is not expected_reference_source:
            raise ValueError("benchmark_reference_source_kind_mismatch")
        expected_cost_basis = {
            BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED: (AuditCostBasis.REALIZED_HUMAN_MINUTES),
            BenchmarkEvidenceKind.SIMULATION: AuditCostBasis.SIMULATED_MINUTES,
            BenchmarkEvidenceKind.DIAGNOSTIC: AuditCostBasis.DIAGNOSTIC_MINUTES,
        }[self.evidence_kind]
        if any(event.cost_basis is not expected_cost_basis for event in self.audit_events):
            raise ValueError("benchmark_audit_cost_kind_mismatch")
        replay_sources = {state.replay_source for state in self.replay_states}
        allowed_replay_sources = {
            BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED: {
                ReplaySource.FROZEN_PIPELINE_RERUN,
                ReplaySource.LEGACY_DECLARED_PIPELINE_RERUN,
            },
            BenchmarkEvidenceKind.SIMULATION: {ReplaySource.PLANTED_SIMULATION},
            BenchmarkEvidenceKind.DIAGNOSTIC: {ReplaySource.DIAGNOSTIC_APPROXIMATION},
        }[self.evidence_kind]
        if len(replay_sources) != 1 or not replay_sources <= allowed_replay_sources:
            raise ValueError("benchmark_replay_source_kind_mismatch")

        event_ids = [event.item_id for event in self.audit_events]
        if event_ids != sorted(set(event_ids)):
            raise ValueError("benchmark_audit_events_must_be_sorted_unique")
        state_sequences = [tuple(state.audit_sequence) for state in self.replay_states]
        if len(state_sequences) != len(set(state_sequences)):
            raise ValueError("benchmark_replay_sequence_duplicate")
        expected_state_order = sorted(state_sequences, key=lambda row: (len(row), row))
        if state_sequences != expected_state_order:
            raise ValueError("benchmark_replay_states_not_canonically_sorted")
        state_by_sequence = {tuple(state.audit_sequence): state for state in self.replay_states}
        if () not in state_by_sequence:
            raise ValueError("benchmark_baseline_replay_missing")
        baseline = state_by_sequence[()]
        baseline_ids = [row.item_id for row in baseline.policy_inputs]
        if baseline_ids != event_ids:
            raise ValueError("benchmark_baseline_policy_audit_identity_mismatch")
        order_by_id = {row.item_id: row.canonical_order for row in baseline.policy_inputs}
        if sorted(order_by_id.values()) != list(range(1, len(event_ids) + 1)):
            raise ValueError("benchmark_canonical_order_must_be_contiguous")
        canonical_sequence = tuple(sorted(event_ids, key=order_by_id.__getitem__))
        if canonical_sequence not in state_by_sequence:
            raise ValueError("benchmark_full_audit_replay_missing")
        if state_by_sequence[canonical_sequence].policy_inputs:
            raise ValueError("benchmark_full_audit_state_has_pending_inputs")
        for state in self.replay_states:
            if state.question_id != self.question_id:
                raise ValueError("benchmark_replay_question_identity_mismatch")
            if state.pipeline_sha256 != self.pipeline_sha256:
                raise ValueError("benchmark_replay_pipeline_identity_mismatch")
            sequence = set(state.audit_sequence)
            if sequence - set(event_ids):
                raise ValueError("benchmark_replay_sequence_item_unknown")
            remaining = sorted(set(event_ids) - sequence)
            if [row.item_id for row in state.policy_inputs] != remaining:
                raise ValueError("benchmark_replay_policy_identity_mismatch")
            for row in state.policy_inputs:
                if row.canonical_order != order_by_id[row.item_id]:
                    raise ValueError("benchmark_canonical_order_changed_across_replay")
        if replay_sources == {ReplaySource.FROZEN_PIPELINE_RERUN}:
            bindings = [
                state.production_binding
                for state in self.replay_states
                if state.production_binding is not None
            ]
            if len(bindings) != len(self.replay_states):
                raise ValueError("benchmark_production_binding_missing")
            if len({binding.binding_version for binding in bindings}) != 1:
                raise ValueError("benchmark_production_binding_version_changed")
            certificates = [
                (
                    binding.certificate.source_certificate_v6
                    if isinstance(binding, ConditionProductionReplayBindingV7)
                    else binding.certificate
                )
                for binding in bindings
            ]
            if any(
                certificate.claim_manifest.get("question_id") != self.question_id
                or certificate.corpus_sha256 != self.corpus_sha256
                or sorted(
                    publication.paper_id
                    for publication in certificate.source_evidence_graph.publications
                )
                != self.paper_ids
                or sorted(cohort.cohort_id for cohort in certificate.source_evidence_graph.cohorts)
                != self.cohort_ids
                for certificate in certificates
            ):
                raise ValueError("benchmark_production_certificate_identity_mismatch")
            if (
                len(
                    {
                        (
                            certificate.claim_manifest_sha256,
                            certificate.corpus_sha256,
                            certificate.source_evidence_graph_sha256,
                        )
                        for certificate in certificates
                    }
                )
                != 1
            ):
                raise ValueError("benchmark_production_certificate_series_mismatch")
            condition_bindings = [
                binding
                for binding in bindings
                if isinstance(binding, ConditionProductionReplayBindingV7)
            ]
            if (
                condition_bindings
                and len(
                    {
                        (
                            binding.certificate.source_certificate_v6.adaptive_policy_context.policy_context_sha256,
                            binding.certificate.source_certificate_v6.adaptive_calibration_bundle_v2.bundle_sha256,
                            binding.certificate.source_certificate_v6.condition_target_semantics.target_semantics_sha256,
                        )
                        for binding in condition_bindings
                    }
                )
                != 1
            ):
                raise ValueError("benchmark_condition_v7_policy_series_mismatch")
        if self.question_id in self.policy_input_provenance.fit_question_ids:
            raise ValueError("benchmark_question_leaks_into_policy_fit")
        if self.claim_id in self.policy_input_provenance.fit_claim_ids:
            raise ValueError("benchmark_claim_leaks_into_policy_fit")
        if set(self.paper_ids) & set(self.policy_input_provenance.fit_paper_ids):
            raise ValueError("benchmark_paper_leaks_into_policy_fit")
        payload = self.model_dump(mode="json", exclude={"record_sha256"})
        if hash_canonical(payload) != self.record_sha256:
            raise ValueError("claim_question_benchmark_record_hash_mismatch")
        return self


def freeze_claim_question_benchmark_record(
    *,
    question_id: str,
    claim_id: str,
    domain: str,
    population_id: str,
    split: BenchmarkSplit,
    evidence_kind: BenchmarkEvidenceKind,
    pipeline_sha256: str,
    corpus_sha256: str,
    paper_ids: Sequence[str],
    cohort_ids: Sequence[str],
    policy_input_provenance: PolicyInputProvenance,
    reference_verdict: ReferenceClaimVerdict,
    audit_events: Sequence[QuestionAuditEvent],
    replay_states: Sequence[QuestionReplayState],
) -> ClaimQuestionBenchmarkRecord:
    events = sorted(audit_events, key=lambda row: row.item_id)
    states = sorted(
        replay_states,
        key=lambda row: (len(row.audit_sequence), tuple(row.audit_sequence)),
    )
    payload: dict[str, Any] = {
        "record_version": "claim-question-benchmark-v3",
        "question_id": question_id,
        "claim_id": claim_id,
        "domain": domain,
        "population_id": population_id,
        "split": split,
        "evidence_kind": evidence_kind,
        "pipeline_sha256": pipeline_sha256,
        "corpus_sha256": corpus_sha256,
        "paper_ids": sorted(set(paper_ids)),
        "cohort_ids": sorted(set(cohort_ids)),
        "policy_input_provenance": policy_input_provenance,
        "reference_verdict": reference_verdict,
        "audit_events": events,
        "replay_states": states,
    }
    return ClaimQuestionBenchmarkRecord.model_validate(
        {**payload, "record_sha256": hash_canonical(payload)}
    )


class LoadedQuestionBenchmark(ContractModel):
    benchmark_file_sha256: str
    record_set_sha256: str
    pipeline_sha256: str
    evidence_kind: BenchmarkEvidenceKind
    split: BenchmarkSplit
    records: Annotated[list[ClaimQuestionBenchmarkRecord], Field(min_length=2)]

    @field_validator("benchmark_file_sha256", "record_set_sha256", "pipeline_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_loaded_benchmark(self) -> LoadedQuestionBenchmark:
        if self.records != sorted(self.records, key=lambda row: row.question_id):
            raise ValueError("loaded_benchmark_questions_not_canonically_sorted")
        if self.record_set_sha256 != hash_canonical(
            [record.record_sha256 for record in self.records]
        ):
            raise ValueError("loaded_benchmark_record_set_hash_mismatch")
        if {record.pipeline_sha256 for record in self.records} != {self.pipeline_sha256}:
            raise ValueError("loaded_benchmark_pipeline_identity_mismatch")
        if {record.evidence_kind for record in self.records} != {self.evidence_kind}:
            raise ValueError("loaded_benchmark_evidence_kind_mismatch")
        if {record.split for record in self.records} != {self.split}:
            raise ValueError("loaded_benchmark_split_mismatch")
        return self


def validate_question_independence(
    records: Sequence[ClaimQuestionBenchmarkRecord],
) -> BenchmarkEvidenceKind:
    """Reject duplicated question units, cross-question corpus overlap, and fit leakage."""

    if len(records) < 2:
        raise QuestionEvaluationContractError("benchmark_requires_two_independent_questions")
    pipeline_sha256s = {record.pipeline_sha256 for record in records}
    if len(pipeline_sha256s) != 1:
        raise QuestionEvaluationContractError("benchmark_pipeline_identity_mixed")
    splits = {record.split for record in records}
    if len(splits) != 1:
        raise QuestionEvaluationContractError("benchmark_splits_mixed")
    kinds = {record.evidence_kind for record in records}
    if len(kinds) != 1:
        raise QuestionEvaluationContractError("benchmark_evidence_kinds_mixed")
    for label, values in (
        ("question", [record.question_id for record in records]),
        ("claim", [record.claim_id for record in records]),
        ("corpus", [record.corpus_sha256 for record in records]),
        (
            "reference_artifact",
            [record.reference_verdict.artifact_sha256 for record in records],
        ),
    ):
        if len(values) != len(set(values)):
            raise QuestionEvaluationContractError(f"benchmark_{label}_identity_overlap")

    def reject_shared(label: str, rows: Sequence[tuple[str, Sequence[str]]]) -> None:
        owner: dict[str, str] = {}
        for question_id, identities in rows:
            for identity in identities:
                if identity in owner:
                    raise QuestionEvaluationContractError(
                        f"benchmark_{label}_overlap:{identity}:{owner[identity]}:{question_id}"
                    )
                owner[identity] = question_id

    reject_shared("paper", [(record.question_id, record.paper_ids) for record in records])
    reject_shared("cohort", [(record.question_id, record.cohort_ids) for record in records])
    reject_shared(
        "audit_artifact",
        [
            (
                record.question_id,
                [event.artifact_sha256 for event in record.audit_events],
            )
            for record in records
        ],
    )
    evaluation_questions = {record.question_id for record in records}
    evaluation_claims = {record.claim_id for record in records}
    evaluation_papers = {paper for record in records for paper in record.paper_ids}
    for record in records:
        provenance = record.policy_input_provenance
        if evaluation_questions & set(provenance.fit_question_ids):
            raise QuestionEvaluationContractError("benchmark_fit_evaluation_question_overlap")
        if evaluation_claims & set(provenance.fit_claim_ids):
            raise QuestionEvaluationContractError("benchmark_fit_evaluation_claim_overlap")
        if evaluation_papers & set(provenance.fit_paper_ids):
            raise QuestionEvaluationContractError("benchmark_fit_evaluation_paper_overlap")
    return next(iter(kinds))


def write_question_benchmark_jsonl(
    path: Path,
    records: Sequence[ClaimQuestionBenchmarkRecord],
    *,
    force: bool = False,
) -> LoadedQuestionBenchmark:
    """Validate independence, then write canonical immutable JSONL."""

    validate_question_independence(records)
    ordered = sorted(records, key=lambda row: row.question_id)
    atomic_write_jsonl(path, ordered, force=force)
    return load_question_benchmark(path)


def load_question_benchmark(path: Path) -> LoadedQuestionBenchmark:
    """Load and validate a self-hashed JSONL benchmark without accepting blank rows."""

    if not path.is_file():
        raise QuestionEvaluationContractError(f"benchmark_file_missing:{path.as_posix()}")
    records: list[ClaimQuestionBenchmarkRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise QuestionEvaluationContractError(f"benchmark_blank_jsonl_row:{line_number}")
            try:
                payload = json.loads(line)
                records.append(ClaimQuestionBenchmarkRecord.model_validate(payload))
            except (json.JSONDecodeError, ValueError) as exc:
                raise QuestionEvaluationContractError(
                    f"benchmark_jsonl_row_invalid:{line_number}:{exc}"
                ) from exc
    kind = validate_question_independence(records)
    ordered = sorted(records, key=lambda row: row.question_id)
    if records != ordered:
        raise QuestionEvaluationContractError("benchmark_questions_not_canonically_sorted")
    return LoadedQuestionBenchmark(
        benchmark_file_sha256=sha256_file(path),
        record_set_sha256=hash_canonical([record.record_sha256 for record in records]),
        pipeline_sha256=records[0].pipeline_sha256,
        evidence_kind=kind,
        split=records[0].split,
        records=records,
    )


_QUESTION_EVALUATION_DEPENDENCY_ENTRYPOINTS = (
    "scripts/build_question_replay_state.py",
    "scripts/evaluate_question_benchmark.py",
    "src/literature_multiverse/question_evaluation.py",
)


def _resolve_question_evaluation_local_import(
    *,
    repository_root: Path,
    current_path: str,
    module: str,
    level: int,
) -> str | None:
    current = Path(current_path).with_suffix("")
    if level:
        package_parts = list(current.parts[:-1])
        if level > len(package_parts):
            return None
        module_parts = package_parts[: len(package_parts) - (level - 1)]
        if module:
            module_parts.extend(module.split("."))
        candidates = [
            Path(*module_parts).with_suffix(".py"),
            Path(*module_parts) / "__init__.py",
        ]
    elif module == "literature_multiverse":
        candidates = [Path("src/literature_multiverse/__init__.py")]
    elif module.startswith("literature_multiverse."):
        relative = Path("src", *module.split("."))
        candidates = [relative.with_suffix(".py"), relative / "__init__.py"]
    elif module.startswith("scripts."):
        relative = Path(*module.split("."))
        candidates = [relative.with_suffix(".py")]
    else:
        return None
    for candidate in candidates:
        if (repository_root / candidate).is_file():
            return candidate.as_posix()
    return None


def _question_evaluation_python_dependency_closure(
    repository_root: Path,
) -> list[str]:
    """Mechanically bind every in-repository Python dependency of evaluation."""

    pending = list(_QUESTION_EVALUATION_DEPENDENCY_ENTRYPOINTS)
    observed = {"src/literature_multiverse/__init__.py"}
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        source_path = repository_root / relative
        if not source_path.is_file():
            raise QuestionEvaluationContractError(
                f"question_evaluation_dependency_missing:{relative}"
            )
        observed.add(relative)
        try:
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"),
                filename=relative,
            )
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise QuestionEvaluationContractError(
                f"question_evaluation_dependency_unreadable:{relative}"
            ) from exc
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom):
                modules.append((node.module or "", node.level))
            elif isinstance(node, ast.Import):
                modules.extend((alias.name, 0) for alias in node.names)
            for module, level in modules:
                dependency = _resolve_question_evaluation_local_import(
                    repository_root=repository_root,
                    current_path=relative,
                    module=module,
                    level=level,
                )
                if dependency is not None and dependency not in observed:
                    pending.append(dependency)
    return sorted(observed)


def compute_question_evaluation_pipeline_fingerprint(
    *, root: Path | None = None
) -> PipelineFingerprint:
    """Freeze the exact implementation that computes the reported policy result."""

    repository_root = root or Path(__file__).resolve().parents[2]
    python_closure = _question_evaluation_python_dependency_closure(repository_root)
    component = PipelineComponentSpec(
        component_id="question-benchmark-evaluation",
        component_version="8",
        file_paths=sorted(
            {
                "pyproject.toml",
                "uv.lock",
                *python_closure,
            }
        ),
        settings={
            "default_stopping_rule": (ReplayStoppingRule.PRODUCTION_STOP_ON_RELEASE.value),
            "dependency_closure_entrypoints": [
                "scripts/build_question_replay_state.py",
                "scripts/evaluate_question_benchmark.py",
                "src/literature_multiverse/question_evaluation.py",
            ],
            "in_repository_dependency_closure_bound": True,
            "installed_dependency_versions": {
                name: distribution_version(name) for name in ("numpy", "pydantic")
            },
            "platform_machine": platform.machine(),
            "platform_system": platform.system(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
    )
    return compute_pipeline_fingerprint(root=repository_root, components=[component])


class QuestionBenchmarkEvaluation(ContractModel):
    evaluation_version: Literal["question-policy-replay-evaluation-v7"] = (
        "question-policy-replay-evaluation-v7"
    )
    benchmark_file_sha256: str
    record_set_sha256: str
    pipeline_sha256: str
    evaluation_pipeline_fingerprint: PipelineFingerprint
    evidence_kind: BenchmarkEvidenceKind
    split: BenchmarkSplit
    scientific_claim_eligible: bool
    claim_scope: str
    causal_interpretation: str
    replay_assumptions: list[str]
    budgets_minutes: list[float]
    fixed_count: int
    random_seed: int
    bootstrap_draws: int
    bootstrap_seed: int
    stopping_rule: ReplayStoppingRule
    production_policy_match: bool
    stopping_rule_semantics: str
    primary_policy: ReplayPolicy
    policy_results: list[dict[str, Any]]
    paired_policy_comparisons: list[dict[str, Any]]
    audit_all_upper_bound: dict[str, Any]
    metric_definitions: dict[str, str]
    evaluation_sha256: str

    @field_validator(
        "benchmark_file_sha256",
        "record_set_sha256",
        "pipeline_sha256",
        "evaluation_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_evaluation(self) -> QuestionBenchmarkEvaluation:
        components = self.evaluation_pipeline_fingerprint.components
        if (
            len(components) != 1
            or components[0].component_id != "question-benchmark-evaluation"
            or components[0].component_version != "8"
        ):
            raise ValueError("question_evaluation_pipeline_contract_mismatch")
        expected_match, expected_semantics = _stopping_rule_contract(self.stopping_rule)
        if (
            self.production_policy_match != expected_match
            or self.stopping_rule_semantics != expected_semantics
        ):
            raise ValueError("question_evaluation_stopping_rule_contract_mismatch")
        for result in self.policy_results:
            if (
                result.get("stopping_rule") != self.stopping_rule.value
                or result.get("production_policy_match") != expected_match
                or result.get("pipeline_sha256") != self.pipeline_sha256
            ):
                raise ValueError("question_evaluation_policy_stopping_rule_mismatch")
            outcomes = result.get("question_outcomes")
            if not isinstance(outcomes, list) or any(
                not isinstance(row, dict)
                or row.get("stopping_rule") != self.stopping_rule.value
                or row.get("pipeline_sha256") != self.pipeline_sha256
                for row in outcomes
            ):
                raise ValueError("question_evaluation_outcome_stopping_rule_mismatch")
        if any(
            row.get("pipeline_sha256") != self.pipeline_sha256
            for row in self.paired_policy_comparisons
        ):
            raise ValueError("question_evaluation_comparison_pipeline_mismatch")
        if (
            self.audit_all_upper_bound.get("stopping_rule") != "exhaustive_upper_bound"
            or self.audit_all_upper_bound.get("production_policy_match") is not False
            or self.audit_all_upper_bound.get("pipeline_sha256") != self.pipeline_sha256
        ):
            raise ValueError("question_evaluation_upper_bound_stopping_rule_invalid")
        payload = self.model_dump(mode="json", exclude={"evaluation_sha256"})
        if hash_canonical(payload) != self.evaluation_sha256:
            raise ValueError("question_benchmark_evaluation_hash_mismatch")
        return self


def _random_priority(*, seed: int, item_id: str) -> float:
    """Mirror the production allocator's stable per-item seeded random priority."""

    digest = hashlib.sha256(f"budgeted-verification-v1\0{seed}\0{item_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _priority(
    row: ReplayPolicyInput,
    policy: ReplayPolicy,
    *,
    seed: int,
) -> float:
    if policy is ReplayPolicy.RANDOM:
        return _random_priority(seed=seed, item_id=row.item_id)
    if policy is ReplayPolicy.RISK_ONLY:
        return row.risk_score
    if policy is ReplayPolicy.DISAGREEMENT_ONLY:
        return row.disagreement_score
    if policy is ReplayPolicy.INFLUENCE_ONLY:
        return row.influence_score
    if policy is ReplayPolicy.RISK_X_INFLUENCE:
        return row.risk_score * row.influence_score
    if policy is ReplayPolicy.RISK_PER_COST:
        return row.risk_score / row.estimated_minutes
    if policy is ReplayPolicy.DISAGREEMENT_PER_COST:
        return row.disagreement_score / row.estimated_minutes
    if policy is ReplayPolicy.INFLUENCE_PER_COST:
        return row.influence_score / row.estimated_minutes
    if policy is ReplayPolicy.RISK_X_INFLUENCE_PER_COST:
        return row.risk_score * row.influence_score / row.estimated_minutes
    if policy is ReplayPolicy.FIXED_COUNT:
        return -float(row.canonical_order)
    raise QuestionEvaluationContractError(f"budgeted_policy_not_rankable:{policy.value}")


def _replay_question(
    record: ClaimQuestionBenchmarkRecord,
    *,
    policy: ReplayPolicy,
    budget_minutes: float,
    fixed_count: int,
    random_seed: int,
    stopping_rule: ReplayStoppingRule,
) -> dict[str, Any]:
    """Replay one policy with the production scheduler's two-stage cost protocol.

    Estimated minutes are policy-visible and determine selection feasibility.  The
    selected event's realized duration is revealed only after selection.  If it would
    finish beyond the hard deadline, the remaining minutes are charged to an active,
    incomplete action; its adjudication is not applied and release is forced to abstain.
    """

    state_by_sequence = {tuple(state.audit_sequence): state for state in record.replay_states}
    event_by_id = {event.item_id: event for event in record.audit_events}
    resolved: list[str] = []
    selected: list[str] = []
    historical_spent = 0.0
    active_item_id: str | None = None
    active_truncated_minutes = 0.0
    stop_reason: ReplayStopReason
    if policy is ReplayPolicy.NO_AUDIT:
        state = state_by_sequence[()]
        stop_reason = ReplayStopReason.NO_AUDIT_POLICY
    else:
        while True:
            state = state_by_sequence.get(tuple(resolved))
            if state is None:
                raise QuestionEvaluationContractError(
                    "benchmark_replay_state_missing:"
                    f"question={record.question_id}:sequence={resolved}"
                )
            # Production stopping consumes the complete certificate decision, never a
            # benchmark author's manually declared release string.  The binding includes
            # adapter blockers and the evaluated state's active-action gate.
            if stopping_rule is ReplayStoppingRule.PRODUCTION_STOP_ON_RELEASE:
                binding = state.production_binding
                if state.replay_source is not ReplaySource.FROZEN_PIPELINE_RERUN or binding is None:
                    raise QuestionEvaluationContractError(
                        "production_stopping_requires_certificate_bound_replay_state:"
                        f"question={record.question_id}:sequence={resolved}"
                    )
                if binding.evaluated_active_action_item_id is not None:
                    raise QuestionEvaluationContractError(
                        "production_replay_prefix_has_active_action:"
                        f"question={record.question_id}:sequence={resolved}"
                    )
                if binding.full_release_eligible:
                    stop_reason = ReplayStopReason.FIRST_FROZEN_RELEASE_ELIGIBLE_STATE
                    break
            if not state.policy_inputs:
                stop_reason = ReplayStopReason.ALL_ITEMS_RESOLVED
                break
            if policy is ReplayPolicy.FIXED_COUNT and len(selected) >= fixed_count:
                stop_reason = ReplayStopReason.FIXED_COUNT_REACHED
                break
            remaining_budget = max(0.0, budget_minutes - historical_spent)
            if remaining_budget == 0.0:
                stop_reason = ReplayStopReason.BUDGET_EXHAUSTED_WITHOUT_ACTIVE_ACTION
                break
            ranked = sorted(
                state.policy_inputs,
                key=lambda row: (
                    -_priority(
                        row,
                        policy,
                        seed=random_seed,
                    ),
                    row.item_id,
                ),
            )
            fitting = [
                row
                for row in ranked
                if row.eligible and row.estimated_minutes <= remaining_budget + _COST_TOLERANCE
            ]
            if not fitting:
                stop_reason = ReplayStopReason.NO_ELIGIBLE_CANDIDATE_FITS_ESTIMATED_BUDGET
                break
            next_item = fitting[0]
            selected.append(next_item.item_id)

            # This is the first point at which retrospective realized cost is opened.
            # Neither event disposition nor any reference verdict participates in ranking.
            realized = event_by_id[next_item.item_id].realized_minutes
            if realized > remaining_budget + _COST_TOLERANCE:
                active_item_id = next_item.item_id
                active_truncated_minutes = remaining_budget
                stop_reason = ReplayStopReason.BUDGET_EXHAUSTED_WITH_ACTIVE_ACTION
                break
            resolved.append(next_item.item_id)
            historical_spent = math.fsum((historical_spent, realized))
        state = state_by_sequence.get(tuple(resolved))
        if state is None:
            raise QuestionEvaluationContractError(
                f"benchmark_replay_state_missing:question={record.question_id}:sequence={resolved}"
            )
    current_spent = math.fsum((historical_spent, active_truncated_minutes))
    active_action = active_item_id is not None
    released = state.release_status is ReplayReleaseStatus.RELEASED and not active_action
    exact_decision_match = state.claim_classification == record.reference_verdict.verdict.value
    release_reasons = list(state.release_reasons)
    if active_action:
        release_reasons.append("budget_exhausted_active_audit_action_unresolved")
    payload = {
        "question_id": record.question_id,
        "pipeline_sha256": record.pipeline_sha256,
        "domain": record.domain,
        "selected_item_ids": selected,
        "attempted_item_ids": selected,
        "resolved_item_ids": resolved,
        "incomplete_item_ids": [] if active_item_id is None else [active_item_id],
        "active_action_item_id": active_item_id,
        "budget_exhausted_with_active_action": active_action,
        "historical_realized_minutes": historical_spent,
        "active_truncated_realized_minutes": active_truncated_minutes,
        "realized_minutes": current_spent,
        "budget_minutes": budget_minutes,
        "stopping_rule": stopping_rule.value,
        "stop_reason": stop_reason.value,
        "release_status": (
            ReplayReleaseStatus.RELEASED.value if released else ReplayReleaseStatus.ABSTAINED.value
        ),
        "claim_classification": state.claim_classification,
        "release_reasons": list(dict.fromkeys(release_reasons)),
        "reference_verdict": record.reference_verdict.verdict.value,
        "released_claim_error": released and not exact_decision_match,
        "correct_release": released and exact_decision_match,
        "appropriate_abstention": not released and not exact_decision_match,
        "missed_correct_decision_abstention": (not released and exact_decision_match),
        # Retained as a compatibility alias; its semantics are exact five-way
        # decision match as of evaluation-pipeline-v7.
        "missed_supported_abstention": not released and exact_decision_match,
        "scientific_state_replay_sha256": state.replay_sha256,
    }
    return {**payload, "retrospective_outcome_sha256": hash_canonical(payload)}


def _audit_all_question(record: ClaimQuestionBenchmarkRecord) -> dict[str, Any]:
    baseline = next(state for state in record.replay_states if not state.audit_sequence)
    canonical_order = {row.item_id: row.canonical_order for row in baseline.policy_inputs}
    sequence = tuple(sorted(canonical_order, key=canonical_order.__getitem__))
    state_by_sequence = {tuple(state.audit_sequence): state for state in record.replay_states}
    state = state_by_sequence[sequence]
    event_by_id = {event.item_id: event for event in record.audit_events}
    spent = math.fsum(event_by_id[item_id].realized_minutes for item_id in sequence)
    released = state.release_status is ReplayReleaseStatus.RELEASED
    exact_decision_match = state.claim_classification == record.reference_verdict.verdict.value
    payload = {
        "question_id": record.question_id,
        "pipeline_sha256": record.pipeline_sha256,
        "domain": record.domain,
        "selected_item_ids": list(sequence),
        "attempted_item_ids": list(sequence),
        "resolved_item_ids": list(sequence),
        "incomplete_item_ids": [],
        "active_action_item_id": None,
        "budget_exhausted_with_active_action": False,
        "historical_realized_minutes": spent,
        "active_truncated_realized_minutes": 0.0,
        "realized_minutes": spent,
        "budget_minutes": None,
        "stop_reason": ReplayStopReason.ALL_ITEMS_RESOLVED.value,
        "release_status": state.release_status.value,
        "claim_classification": state.claim_classification,
        "release_reasons": list(state.release_reasons),
        "reference_verdict": record.reference_verdict.verdict.value,
        "released_claim_error": released and not exact_decision_match,
        "correct_release": released and exact_decision_match,
        "appropriate_abstention": not released and not exact_decision_match,
        "missed_correct_decision_abstention": (not released and exact_decision_match),
        "missed_supported_abstention": not released and exact_decision_match,
        "scientific_state_replay_sha256": state.replay_sha256,
    }
    return {**payload, "retrospective_outcome_sha256": hash_canonical(payload)}


def _aggregate_metrics(
    outcomes: Sequence[Mapping[str, Any]],
    *,
    human_minutes: bool,
) -> dict[str, Any]:
    n_questions = len(outcomes)
    released = sum(row["release_status"] == "released" for row in outcomes)
    errors = sum(bool(row["released_claim_error"]) for row in outcomes)
    correct = sum(bool(row["correct_release"]) for row in outcomes)
    appropriate_abstentions = sum(bool(row["appropriate_abstention"]) for row in outcomes)
    missed_abstentions = sum(bool(row["missed_correct_decision_abstention"]) for row in outcomes)
    spent = math.fsum(float(row["realized_minutes"]) for row in outcomes)
    historical_spent = math.fsum(float(row["historical_realized_minutes"]) for row in outcomes)
    active_truncated = math.fsum(
        float(row["active_truncated_realized_minutes"]) for row in outcomes
    )
    completed_actions = sum(len(row["resolved_item_ids"]) for row in outcomes)
    attempted_actions = sum(len(row["selected_item_ids"]) for row in outcomes)
    incomplete_actions = sum(bool(row["budget_exhausted_with_active_action"]) for row in outcomes)
    efficiency = correct * 60.0 / spent if spent > 0 else None
    return {
        "n_questions": n_questions,
        "total_realized_minutes": spent,
        "historical_completed_realized_minutes": historical_spent,
        "active_truncated_realized_minutes": active_truncated,
        "completed_audit_actions": completed_actions,
        "attempted_audit_actions": attempted_actions,
        "incomplete_audit_actions": incomplete_actions,
        "questions_with_budget_exhausted_active_action": incomplete_actions,
        "released_claims": released,
        "abstained_claims": n_questions - released,
        "release_coverage": released / n_questions,
        "released_claim_errors": errors,
        "released_claim_error": errors / released if released else None,
        "correct_releases": correct,
        "correct_releases_per_human_hour": efficiency if human_minutes else None,
        "correct_releases_per_cost_hour": efficiency,
        "appropriate_abstentions": appropriate_abstentions,
        "missed_correct_decision_abstentions": missed_abstentions,
        "missed_supported_abstentions": missed_abstentions,
        "abstention_utility": (appropriate_abstentions - missed_abstentions) / n_questions,
    }


def _domain_metrics(
    outcomes: Sequence[Mapping[str, Any]],
    *,
    human_minutes: bool,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in outcomes:
        grouped[str(row["domain"])].append(row)
    return {
        domain: _aggregate_metrics(rows, human_minutes=human_minutes)
        for domain, rows in sorted(grouped.items())
    }


def _worst_domain_metrics(
    domains: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    def choose(key: str, *, maximum: bool) -> dict[str, Any] | None:
        candidates = [
            (domain, values[key]) for domain, values in domains.items() if values[key] is not None
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda row: ((-row[1]) if maximum else row[1], row[0]))
        domain, value = candidates[0]
        return {"domain": domain, "value": value}

    return {
        "release_coverage": choose("release_coverage", maximum=False),
        "released_claim_error": choose("released_claim_error", maximum=True),
        "correct_releases_per_human_hour": choose("correct_releases_per_human_hour", maximum=False),
        "correct_releases_per_cost_hour": choose("correct_releases_per_cost_hour", maximum=False),
        "abstention_utility": choose("abstention_utility", maximum=False),
        "domains_without_releases": sorted(
            domain for domain, values in domains.items() if values["released_claim_error"] is None
        ),
    }


def _bootstrap_seed(base_seed: int, *, policy: str, budget_minutes: float | None) -> int:
    digest = hashlib.sha256(
        f"question-clustered-bootstrap-v1\0{base_seed}\0{policy}\0{budget_minutes}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _bootstrap_intervals(
    outcomes: Sequence[Mapping[str, Any]],
    *,
    human_minutes: bool,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in outcomes:
        grouped[str(row["domain"])].append(row)
    rng = np.random.default_rng(seed)
    metric_names = (
        "release_coverage",
        "released_claim_error",
        "correct_releases_per_human_hour",
        "correct_releases_per_cost_hour",
        "abstention_utility",
        "worst_domain_release_coverage",
        "worst_domain_released_claim_error",
        "worst_domain_correct_releases_per_human_hour",
        "worst_domain_abstention_utility",
    )
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(draws):
        sample: list[Mapping[str, Any]] = []
        for domain in sorted(grouped):
            rows = grouped[domain]
            indices = rng.integers(0, len(rows), size=len(rows))
            sample.extend(rows[int(index)] for index in indices)
        aggregate = _aggregate_metrics(sample, human_minutes=human_minutes)
        domains = _domain_metrics(sample, human_minutes=human_minutes)
        worst = _worst_domain_metrics(domains)
        draw_values: dict[str, float | None] = {
            "release_coverage": aggregate["release_coverage"],
            "released_claim_error": aggregate["released_claim_error"],
            "correct_releases_per_human_hour": aggregate["correct_releases_per_human_hour"],
            "correct_releases_per_cost_hour": aggregate["correct_releases_per_cost_hour"],
            "abstention_utility": aggregate["abstention_utility"],
            "worst_domain_release_coverage": (
                worst["release_coverage"]["value"]
                if worst["release_coverage"] is not None
                else None
            ),
            "worst_domain_released_claim_error": (
                worst["released_claim_error"]["value"]
                if worst["released_claim_error"] is not None
                else None
            ),
            "worst_domain_correct_releases_per_human_hour": (
                worst["correct_releases_per_human_hour"]["value"]
                if worst["correct_releases_per_human_hour"] is not None
                else None
            ),
            "worst_domain_abstention_utility": (
                worst["abstention_utility"]["value"]
                if worst["abstention_utility"] is not None
                else None
            ),
        }
        for name, value in draw_values.items():
            if value is not None and math.isfinite(value):
                values[name].append(float(value))
    output: dict[str, Any] = {}
    for name in metric_names:
        valid = values[name]
        if valid:
            lower, upper = np.quantile(valid, [0.025, 0.975], method="linear")
            interval = [float(lower), float(upper)]
            reason = None
        else:
            interval = None
            reason = "metric_undefined_in_all_bootstrap_draws"
        output[name] = {
            "confidence_level": 0.95,
            "interval": interval,
            "valid_draws": len(valid),
            "requested_draws": draws,
            "undefined_draws": draws - len(valid),
            "reason": reason,
        }
    return output


def _policy_result(
    *,
    policy: ReplayPolicy,
    budget_minutes: float | None,
    outcomes: list[dict[str, Any]],
    human_minutes: bool,
    fixed_count: int,
    bootstrap_draws: int,
    bootstrap_seed: int,
    upper_bound: bool = False,
    stopping_rule: ReplayStoppingRule | None = None,
) -> dict[str, Any]:
    pipeline_sha256s = {str(row["pipeline_sha256"]) for row in outcomes}
    if len(pipeline_sha256s) != 1:
        raise QuestionEvaluationContractError("policy_outcome_pipeline_identity_mixed")
    metrics = _aggregate_metrics(outcomes, human_minutes=human_minutes)
    domains = _domain_metrics(outcomes, human_minutes=human_minutes)
    payload: dict[str, Any] = {
        "policy": policy.value,
        "pipeline_sha256": next(iter(pipeline_sha256s)),
        "budget_minutes_per_question": budget_minutes,
        "upper_bound": upper_bound,
        "stopping_rule": (
            stopping_rule.value if stopping_rule is not None else "exhaustive_upper_bound"
        ),
        "production_policy_match": (stopping_rule is ReplayStoppingRule.PRODUCTION_STOP_ON_RELEASE),
        "fixed_count": fixed_count if policy is ReplayPolicy.FIXED_COUNT else None,
        "selection_protocol": (
            (
                "production_stop_at_first_frozen_release_eligible_state_before_opening_"
                "future_policy_inputs_or_audit_outcomes; "
                if stopping_rule is ReplayStoppingRule.PRODUCTION_STOP_ON_RELEASE
                else (
                    "experimental_allocate_to_cap_without_intermediate_release_early_stop; "
                    if stopping_rule is ReplayStoppingRule.ALLOCATE_TO_CAP_EXPERIMENTAL
                    else "exhaustive_upper_bound; "
                )
            )
            + "adaptive_rerank_after_each_completed_logged_correction; priorities_use_only_"
            "current_state_inputs; estimated_minutes_determine_selection_feasibility_and_"
            "rank_cost_normalized_policies; scheduler_selects_the_highest_priority_eligible_"
            "item_whose_estimate_fits_remaining_budget; realized_minutes_are_opened_only_"
            "after_selection; completion_beyond_the_deadline_becomes_a_truncated_active_"
            "unresolved_action_that_is_not_applied_and_forces_abstention"
        ),
        "metrics": metrics,
        "domain_metrics": domains,
        "worst_domain_metrics": _worst_domain_metrics(domains),
        "bootstrap": {
            "method": "question_clustered_stratified_percentile_bootstrap",
            "cluster_unit": "complete_claim_question",
            "strata": "domain",
            "seed": bootstrap_seed,
            "bit_generator": "PCG64",
            "quantile_method": "linear",
            "intervals": _bootstrap_intervals(
                outcomes,
                human_minutes=human_minutes,
                draws=bootstrap_draws,
                seed=bootstrap_seed,
            ),
        },
        "question_outcomes": outcomes,
    }
    return {**payload, "result_sha256": hash_canonical(payload)}


_PAIRED_METRICS = (
    "release_coverage",
    "released_claim_error",
    "correct_releases_per_human_hour",
    "correct_releases_per_cost_hour",
    "abstention_utility",
)


def _paired_metric_deltas(
    primary: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    human_minutes: bool,
) -> dict[str, float | None]:
    primary_metrics = _aggregate_metrics(primary, human_minutes=human_minutes)
    baseline_metrics = _aggregate_metrics(baseline, human_minutes=human_minutes)
    output: dict[str, float | None] = {}
    for metric in _PAIRED_METRICS:
        left = primary_metrics[metric]
        right = baseline_metrics[metric]
        output[metric] = None if left is None or right is None else float(left - right)
    output["correct_releases_per_question"] = float(
        primary_metrics["correct_releases"] - baseline_metrics["correct_releases"]
    ) / len(primary)
    output["released_claim_errors_per_question"] = float(
        primary_metrics["released_claim_errors"] - baseline_metrics["released_claim_errors"]
    ) / len(primary)
    return output


def _paired_bootstrap_intervals(
    primary: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    human_minutes: bool,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if [row["question_id"] for row in primary] != [row["question_id"] for row in baseline]:
        raise QuestionEvaluationContractError("paired_policy_outcome_question_order_mismatch")
    domains: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(primary):
        if row["domain"] != baseline[index]["domain"]:
            raise QuestionEvaluationContractError("paired_policy_outcome_domain_mismatch")
        domains[str(row["domain"])].append(index)
    metric_names = (
        *_PAIRED_METRICS,
        "correct_releases_per_question",
        "released_claim_errors_per_question",
    )
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        indices: list[int] = []
        for domain in sorted(domains):
            domain_indices = domains[domain]
            sampled = rng.integers(0, len(domain_indices), size=len(domain_indices))
            indices.extend(domain_indices[int(index)] for index in sampled)
        primary_sample = [primary[index] for index in indices]
        baseline_sample = [baseline[index] for index in indices]
        deltas = _paired_metric_deltas(
            primary_sample,
            baseline_sample,
            human_minutes=human_minutes,
        )
        for name, value in deltas.items():
            if value is not None and math.isfinite(value):
                values[name].append(value)
    intervals: dict[str, Any] = {}
    for name in metric_names:
        valid = values[name]
        if valid:
            lower, upper = np.quantile(valid, [0.025, 0.975], method="linear")
            interval = [float(lower), float(upper)]
            probability_positive = sum(value > 0 for value in valid) / len(valid)
        else:
            interval = None
            probability_positive = None
        intervals[name] = {
            "confidence_level": 0.95,
            "interval": interval,
            "valid_draws": len(valid),
            "requested_draws": draws,
            "undefined_draws": draws - len(valid),
            "bootstrap_probability_primary_delta_gt_zero": probability_positive,
        }
    return intervals


def _paired_policy_comparison(
    *,
    primary_result: Mapping[str, Any],
    baseline_result: Mapping[str, Any],
    human_minutes: bool,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    primary_outcomes = primary_result["question_outcomes"]
    baseline_outcomes = baseline_result["question_outcomes"]
    if not isinstance(primary_outcomes, list) or not isinstance(baseline_outcomes, list):
        raise QuestionEvaluationContractError("paired_policy_outcomes_missing")
    if primary_result.get("pipeline_sha256") != baseline_result.get("pipeline_sha256"):
        raise QuestionEvaluationContractError("paired_policy_pipeline_identity_mismatch")
    point = _paired_metric_deltas(
        primary_outcomes,
        baseline_outcomes,
        human_minutes=human_minutes,
    )
    payload = {
        "comparison_version": "paired-policy-comparison-v1",
        "pipeline_sha256": primary_result["pipeline_sha256"],
        "primary_policy": primary_result["policy"],
        "baseline_policy": baseline_result["policy"],
        "budget_minutes_per_question": primary_result["budget_minutes_per_question"],
        "question_count": len(primary_outcomes),
        "delta_definition": "primary_minus_baseline_on_identical_complete_questions",
        "point_deltas": point,
        "bootstrap": {
            "method": "paired_question_clustered_stratified_percentile_bootstrap",
            "cluster_unit": "complete_claim_question",
            "strata": "domain",
            "seed": bootstrap_seed,
            "bit_generator": "PCG64",
            "quantile_method": "linear",
            "intervals": _paired_bootstrap_intervals(
                primary_outcomes,
                baseline_outcomes,
                human_minutes=human_minutes,
                draws=bootstrap_draws,
                seed=bootstrap_seed,
            ),
        },
    }
    return {**payload, "comparison_sha256": hash_canonical(payload)}


def evaluate_question_benchmark(
    benchmark: LoadedQuestionBenchmark,
    *,
    budgets_minutes: Sequence[float],
    fixed_count: int = 5,
    random_seed: int = 0,
    bootstrap_draws: int = 2000,
    bootstrap_seed: int = 0,
    allow_non_real: bool = False,
    policies: Sequence[ReplayPolicy] = BUDGETED_REPLAY_POLICIES,
    primary_policy: ReplayPolicy | None = None,
    stopping_rule: ReplayStoppingRule = ReplayStoppingRule.PRODUCTION_STOP_ON_RELEASE,
    evaluation_root: Path | None = None,
) -> QuestionBenchmarkEvaluation:
    """Replay policies at matched realized-minute caps over independent questions."""

    validate_question_independence(benchmark.records)
    production_bound = all(
        state.replay_source is ReplaySource.FROZEN_PIPELINE_RERUN
        and state.production_binding is not None
        for record in benchmark.records
        for state in record.replay_states
    )
    stopping_rule = ReplayStoppingRule(stopping_rule)
    if stopping_rule is ReplayStoppingRule.PRODUCTION_STOP_ON_RELEASE and not production_bound:
        raise QuestionEvaluationContractError(
            "production_stopping_requires_all_states_certificate_bound"
        )
    if benchmark.evidence_kind is not BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED:
        if not allow_non_real:
            raise QuestionEvaluationContractError(
                "non_real_benchmark_requires_explicit_allow_non_real"
            )
        scientific_claim_eligible = False
        claim_scope = (
            f"{benchmark.evidence_kind.value}_diagnostic_only; cannot support a real-world "
            "human-efficiency or released-claim-error claim"
        )
        causal_interpretation = (
            "mechanical retrospective policy replay only; no causal or real human-efficiency "
            "interpretation"
        )
    elif production_bound:
        scientific_claim_eligible = True
        claim_scope = (
            "retrospective expert-adjudicated replay under the frozen benchmark and "
            "declared non-overlap; not prospective scientific truth"
        )
        causal_interpretation = (
            "retrospective off-policy comparison under declared item-outcome and item-cost "
            "order invariance; not a randomized prospective policy comparison"
        )
    else:
        scientific_claim_eligible = False
        claim_scope = (
            "legacy_declared_pipeline_replay_only; real audit labels and costs do not "
            "establish production scientific-state lineage"
        )
        causal_interpretation = (
            "mechanical retrospective replay only; production stopping and released-claim "
            "error claims require certificate-bound v5 or final condition-v7 verifier states"
        )
    production_policy_match, stopping_rule_semantics = _stopping_rule_contract(stopping_rule)
    replay_assumptions = [
        (
            "item-level order invariance: an item's audit disposition and declared duration "
            "do not change with the previously completed audit prefix"
        ),
        (
            "each frozen replay state has the benchmark-kind-appropriate declared source for "
            "its exact completed audit prefix; only real expert inputs require a frozen "
            "pipeline rerun"
        ),
        (
            "policy inputs were frozen without future audit outcomes or the reference verdict; "
            "the provenance contract is auditable metadata, not causal proof"
        ),
        (
            "each budget is a hard per-question deadline and any selected action still active "
            "at that deadline blocks release"
        ),
        stopping_rule_semantics,
    ]
    budgets = sorted(float(value) for value in budgets_minutes)
    if not budgets:
        raise QuestionEvaluationContractError("question_evaluation_budgets_empty")
    if len(budgets) != len(set(budgets)):
        raise QuestionEvaluationContractError("question_evaluation_budgets_duplicate")
    if any(not math.isfinite(value) or value < 0 for value in budgets):
        raise QuestionEvaluationContractError("question_evaluation_budget_invalid")
    if fixed_count < 1:
        raise QuestionEvaluationContractError("question_evaluation_fixed_count_invalid")
    if bootstrap_draws < 100:
        raise QuestionEvaluationContractError(
            "question_evaluation_bootstrap_requires_at_least_100_draws"
        )
    parsed_policies = tuple(ReplayPolicy(policy) for policy in policies)
    if not parsed_policies or len(parsed_policies) != len(set(parsed_policies)):
        raise QuestionEvaluationContractError("question_evaluation_policies_invalid")
    if ReplayPolicy.AUDIT_ALL_UPPER_BOUND in parsed_policies:
        raise QuestionEvaluationContractError(
            "audit_all_is_reported_separately_from_matched_budget_policies"
        )
    primary_policy = (
        ReplayPolicy.RISK_X_INFLUENCE_PER_COST
        if primary_policy is None and ReplayPolicy.RISK_X_INFLUENCE_PER_COST in parsed_policies
        else ReplayPolicy(primary_policy or parsed_policies[0])
    )
    if primary_policy not in parsed_policies:
        raise QuestionEvaluationContractError("question_evaluation_primary_policy_not_evaluated")
    human_minutes = benchmark.evidence_kind is BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED
    results: list[dict[str, Any]] = []
    for policy in parsed_policies:
        for budget in budgets:
            outcomes = [
                _replay_question(
                    record,
                    policy=policy,
                    budget_minutes=budget,
                    fixed_count=fixed_count,
                    random_seed=random_seed,
                    stopping_rule=stopping_rule,
                )
                for record in benchmark.records
            ]
            row_seed = _bootstrap_seed(
                bootstrap_seed,
                policy=policy.value,
                budget_minutes=budget,
            )
            results.append(
                _policy_result(
                    policy=policy,
                    budget_minutes=budget,
                    outcomes=outcomes,
                    human_minutes=human_minutes,
                    fixed_count=fixed_count,
                    bootstrap_draws=bootstrap_draws,
                    bootstrap_seed=row_seed,
                    stopping_rule=stopping_rule,
                )
            )
    result_by_key = {
        (ReplayPolicy(row["policy"]), float(row["budget_minutes_per_question"])): row
        for row in results
    }
    paired_comparisons: list[dict[str, Any]] = []
    for budget in budgets:
        primary_result = result_by_key[(primary_policy, budget)]
        for baseline_policy in parsed_policies:
            if baseline_policy is primary_policy:
                continue
            baseline_result = result_by_key[(baseline_policy, budget)]
            comparison_seed = _bootstrap_seed(
                bootstrap_seed,
                policy=f"{primary_policy.value}-minus-{baseline_policy.value}",
                budget_minutes=budget,
            )
            paired_comparisons.append(
                _paired_policy_comparison(
                    primary_result=primary_result,
                    baseline_result=baseline_result,
                    human_minutes=human_minutes,
                    bootstrap_draws=bootstrap_draws,
                    bootstrap_seed=comparison_seed,
                )
            )
    upper_outcomes = [_audit_all_question(record) for record in benchmark.records]
    upper_seed = _bootstrap_seed(
        bootstrap_seed,
        policy=ReplayPolicy.AUDIT_ALL_UPPER_BOUND.value,
        budget_minutes=None,
    )
    upper = _policy_result(
        policy=ReplayPolicy.AUDIT_ALL_UPPER_BOUND,
        budget_minutes=None,
        outcomes=upper_outcomes,
        human_minutes=human_minutes,
        fixed_count=fixed_count,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=upper_seed,
        upper_bound=True,
        stopping_rule=None,
    )
    metric_definitions = {
        "release_coverage": "released questions / complete independent questions",
        "released_claim_error": (
            "released five-way decisions that differ exactly from the question-level "
            "reference verdict / released decisions; undefined when nothing is released"
        ),
        "correct_releases_per_human_hour": (
            "exact reference-matching releases * 60 / realized human minutes; available only "
            "for real expert-adjudicated inputs with positive review time"
        ),
        "correct_releases_per_cost_hour": (
            "exact reference-matching releases * 60 / declared cost minutes; simulation and "
            "diagnostic values are not human-efficiency claims"
        ),
        "abstention_utility": (
            "(abstentions when the current five-way decision mismatches the reference - "
            "abstentions when it matches) / complete independent questions"
        ),
        "realized_cost_accounting": (
            "completed adjudication minutes plus time actually spent on any selected action "
            "still incomplete at the hard deadline; estimated minutes are never counted as "
            "realized spending; real-human inputs are total person-minutes summed across all "
            "reviewers and final adjudication, not parallel wall-clock elapsed time"
        ),
        "budget_exhausted_active_action": (
            "a production-style selected action whose estimate fit the remaining budget but "
            "whose realized completion time crossed the deadline; its adjudication is not "
            "applied and the claim must abstain"
        ),
        "worst_domain_metrics": (
            "minimum coverage/efficiency/abstention utility and maximum released error "
            "over declared domains; domains with no releases are listed explicitly"
        ),
    }
    payload: dict[str, Any] = {
        "evaluation_version": "question-policy-replay-evaluation-v7",
        "benchmark_file_sha256": benchmark.benchmark_file_sha256,
        "record_set_sha256": benchmark.record_set_sha256,
        "pipeline_sha256": benchmark.pipeline_sha256,
        "evaluation_pipeline_fingerprint": (
            compute_question_evaluation_pipeline_fingerprint(root=evaluation_root)
        ),
        "evidence_kind": benchmark.evidence_kind,
        "split": benchmark.split,
        "scientific_claim_eligible": scientific_claim_eligible,
        "claim_scope": claim_scope,
        "causal_interpretation": causal_interpretation,
        "replay_assumptions": replay_assumptions,
        "budgets_minutes": budgets,
        "fixed_count": fixed_count,
        "random_seed": random_seed,
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": bootstrap_seed,
        "stopping_rule": stopping_rule,
        "production_policy_match": production_policy_match,
        "stopping_rule_semantics": stopping_rule_semantics,
        "primary_policy": primary_policy,
        "policy_results": results,
        "paired_policy_comparisons": paired_comparisons,
        "audit_all_upper_bound": upper,
        "metric_definitions": metric_definitions,
    }
    return QuestionBenchmarkEvaluation.model_validate(
        {**payload, "evaluation_sha256": hash_canonical(payload)}
    )


__all__ = [
    "BUDGETED_REPLAY_POLICIES",
    "AuditCostAccounting",
    "AuditCostBasis",
    "AuditDisposition",
    "BenchmarkEvidenceKind",
    "BenchmarkSplit",
    "ClaimQuestionBenchmarkRecord",
    "ConditionProductionReplayBindingV7",
    "LoadedQuestionBenchmark",
    "PolicyInputProvenance",
    "ProductionReplayBinding",
    "QuestionAuditEvent",
    "QuestionBenchmarkEvaluation",
    "QuestionEvaluationContractError",
    "QuestionReplayState",
    "ReferenceClaimVerdict",
    "ReferenceClaimVerdictValue",
    "ReferenceVerdictSource",
    "ReplayPolicy",
    "ReplayPolicyInput",
    "ReplayReleaseStatus",
    "ReplaySource",
    "ReplayStopReason",
    "ReplayStoppingRule",
    "compute_question_evaluation_pipeline_fingerprint",
    "evaluate_question_benchmark",
    "freeze_claim_question_benchmark_record",
    "freeze_condition_production_replay_binding_v7",
    "freeze_production_replay_binding",
    "freeze_question_audit_event",
    "freeze_question_replay_state",
    "freeze_question_replay_state_from_certificate",
    "freeze_reference_claim_verdict",
    "load_question_benchmark",
    "validate_question_independence",
    "write_question_benchmark_jsonl",
]
