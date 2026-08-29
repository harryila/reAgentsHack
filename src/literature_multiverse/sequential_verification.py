"""Verifier-specific orchestration for hash-bound sequential evidence auditing.

The generic :mod:`audit_session` ledger deliberately knows nothing about evidence
graphs or statistical synthesis.  This adapter binds that ledger to exact
``EvidenceGraph``/synthesis/candidate artifacts and makes one complete transition:

``select -> external adjudication -> selected-item correction -> actual reruns -> receipt``.

Scientific work remains injectable through typed callbacks, avoiding an import cycle
with the unified verifier.  Estimated candidate cost is used only for feasibility;
only the external adjudication artifact's measured realized cost enters the ledger.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Any, Literal, Protocol

from pydantic import Field, JsonValue, TypeAdapter, field_validator, model_validator

from literature_multiverse.adaptive_calibration import AdaptivePreselectionState
from literature_multiverse.audit_session import (
    AdjudicationArtifact,
    AuditActionPacket,
    AuditResolutionReceiptV2,
    AuditSessionContractError,
    CorrectionDisposition,
    EvidenceGraphCorrection,
    VerificationSession,
    VerificationSessionStatus,
    checkpoint_active_cost,
    create_verification_session,
    freeze_adjudication_artifact,
    freeze_evidence_graph_correction,
    resolve_audit_action,
    resume_verification_session,
    select_audit_action,
)
from literature_multiverse.budgeted_verification import (
    AllocationPolicy,
    AuditCandidate,
    ClaimModel,
    rank_candidates,
)
from literature_multiverse.evidence_graph import EvidenceGraph
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_COST_TOLERANCE = 1e-9


class SequentialVerificationContractError(ValueError):
    """A verifier-specific transition violated evidence or accounting lineage."""


class StaleSequentialVerificationStateError(SequentialVerificationContractError):
    """A caller supplied hashes for a superseded verifier state."""


def _sha256(value: str, field_name: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"invalid_sha256:{field_name}")
    return value


def _optional_sha256(value: str | None, field_name: str) -> str | None:
    if value is not None:
        _sha256(value, field_name)
    return value


def _finite_cost(value: float, field_name: str, *, positive: bool = False) -> float:
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{field_name}_must_be_finite_{qualifier}")
    return value


class CurrentAuditCandidate(ContractModel):
    """One current counterfactual action with deterministic scheduler inputs."""

    candidate_version: Literal["sequential-verifier-candidate-v1"] = (
        "sequential-verifier-candidate-v1"
    )
    item_id: Annotated[str, Field(min_length=1)]
    priority: float
    estimated_cost: float
    cost_unit: Annotated[str, Field(min_length=1)]
    eligible: bool
    ineligibility_reasons: list[str]
    scientific_candidate_sha256: str
    counterfactual_synthesis_sha256: str
    risk_bound_sha256: str | None = None
    candidate_sha256: str

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("sequential_candidate_priority_nonfinite")
        return value

    @field_validator("estimated_cost")
    @classmethod
    def validate_cost(cls, value: float) -> float:
        return _finite_cost(value, "sequential_candidate_estimated_cost", positive=True)

    @field_validator(
        "scientific_candidate_sha256",
        "counterfactual_synthesis_sha256",
        "candidate_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("risk_bound_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return _optional_sha256(value, "risk_bound_sha256")

    @field_validator("eligible", mode="before")
    @classmethod
    def validate_strict_boolean(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("sequential_candidate_eligible_must_be_boolean")
        return value

    @field_validator("ineligibility_reasons")
    @classmethod
    def validate_reasons(cls, value: list[str]) -> list[str]:
        if any(not reason for reason in value) or value != sorted(set(value)):
            raise ValueError("candidate_ineligibility_reasons_must_be_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_candidate(self) -> CurrentAuditCandidate:
        if self.eligible == bool(self.ineligibility_reasons):
            raise ValueError("candidate_eligibility_reason_mismatch")
        payload = self.model_dump(mode="json", exclude={"candidate_sha256"})
        if hash_canonical(payload) != self.candidate_sha256:
            raise ValueError("sequential_candidate_hash_mismatch")
        return self


def freeze_current_audit_candidate(
    *,
    item_id: str,
    priority: float,
    estimated_cost: float,
    cost_unit: str,
    scientific_candidate_sha256: str,
    counterfactual_synthesis_sha256: str,
    eligible: bool = True,
    ineligibility_reasons: Sequence[str] = (),
    risk_bound_sha256: str | None = None,
) -> CurrentAuditCandidate:
    """Seal the policy-visible inputs for one current audit action."""

    payload = {
        "candidate_version": "sequential-verifier-candidate-v1",
        "item_id": item_id,
        "priority": priority,
        "estimated_cost": estimated_cost,
        "cost_unit": cost_unit,
        "eligible": eligible,
        "ineligibility_reasons": sorted(set(ineligibility_reasons)),
        "scientific_candidate_sha256": scientific_candidate_sha256,
        "counterfactual_synthesis_sha256": counterfactual_synthesis_sha256,
        "risk_bound_sha256": risk_bound_sha256,
    }
    return CurrentAuditCandidate.model_validate(
        {**payload, "candidate_sha256": hash_canonical(payload)}
    )


def current_candidates_from_audit_candidates(
    candidates: Sequence[AuditCandidate],
    claim_model: ClaimModel,
    *,
    policy: AllocationPolicy,
    counterfactual_synthesis_sha256s: Mapping[str, str],
    risk_bound_sha256s: Mapping[str, str] | None = None,
    ineligibility_reasons: Mapping[str, Sequence[str]] | None = None,
    seed: int = 0,
) -> tuple[CurrentAuditCandidate, ...]:
    """Adapt existing graph-counterfactual candidates into sealed scheduler inputs."""

    candidate_ids = {candidate.item_id for candidate in candidates}
    if set(counterfactual_synthesis_sha256s) != candidate_ids:
        raise SequentialVerificationContractError(
            "counterfactual_synthesis_hash_identity_mismatch"
        )
    risk_hashes = risk_bound_sha256s or {}
    reasons = ineligibility_reasons or {}
    if not set(risk_hashes) <= candidate_ids or not set(reasons) <= candidate_ids:
        raise SequentialVerificationContractError("candidate_metadata_identity_unknown")
    if not candidates:
        return ()
    by_id = {candidate.item_id: candidate for candidate in candidates}
    ranking = rank_candidates(candidates, claim_model, policy, seed=seed)
    frozen = []
    for record in ranking:
        candidate = by_id[record.item_id]
        item_reasons = tuple(reasons.get(record.item_id, ()))
        frozen.append(
            freeze_current_audit_candidate(
                item_id=record.item_id,
                priority=record.priority,
                estimated_cost=candidate.verification_cost,
                cost_unit=candidate.cost_unit,
                scientific_candidate_sha256=hash_canonical(asdict(candidate)),
                counterfactual_synthesis_sha256=counterfactual_synthesis_sha256s[
                    record.item_id
                ],
                eligible=not item_reasons,
                ineligibility_reasons=item_reasons,
                risk_bound_sha256=risk_hashes.get(record.item_id),
            )
        )
    return tuple(frozen)


class VerifierCorrectionProvenance(ContractModel):
    """Explicit selected-estimate correction and rerun lineage."""

    provenance_version: Literal["verifier-correction-provenance-v1"] = (
        "verifier-correction-provenance-v1"
    )
    session_id: Annotated[str, Field(min_length=1)]
    step: Annotated[int, Field(ge=1)]
    item_id: Annotated[str, Field(min_length=1)]
    action_packet_sha256: str
    selected_candidate_sha256: str
    adjudication_artifact_sha256: str
    adjudication_payload_sha256: str
    disposition: CorrectionDisposition
    provenance: Annotated[str, Field(min_length=1)]
    correction_protocol_sha256: str
    external_correction_payload_sha256: str
    selected_estimate_before_sha256: str
    selected_estimate_after_sha256: str
    pre_graph_sha256: str
    post_graph_sha256: str
    synthesis_runner_sha256: str
    candidate_runner_sha256: str
    post_synthesis_sha256: str
    post_candidate_input_sha256: str
    provenance_sha256: str

    @field_validator(
        "action_packet_sha256",
        "selected_candidate_sha256",
        "adjudication_artifact_sha256",
        "adjudication_payload_sha256",
        "correction_protocol_sha256",
        "external_correction_payload_sha256",
        "selected_estimate_before_sha256",
        "selected_estimate_after_sha256",
        "pre_graph_sha256",
        "post_graph_sha256",
        "synthesis_runner_sha256",
        "candidate_runner_sha256",
        "post_synthesis_sha256",
        "post_candidate_input_sha256",
        "provenance_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_provenance(self) -> VerifierCorrectionProvenance:
        same_estimate = (
            self.selected_estimate_before_sha256 == self.selected_estimate_after_sha256
        )
        if self.disposition is CorrectionDisposition.NO_CHANGE and not same_estimate:
            raise ValueError("no_change_selected_estimate_hash_changed")
        if self.disposition is CorrectionDisposition.CORRECTED and same_estimate:
            raise ValueError("corrected_selected_estimate_hash_unchanged")
        payload = self.model_dump(mode="json", exclude={"provenance_sha256"})
        if hash_canonical(payload) != self.provenance_sha256:
            raise ValueError("verifier_correction_provenance_hash_mismatch")
        return self


class SequentialStateTransition(ContractModel):
    """One replayable mutation of a sequential verifier state.

    Selection and cost checkpoints are included alongside scientific corrections so
    the session's opaque state-history hashes never have to be trusted.  A correction
    carries the complete post-correction scientific state and every adjudication,
    provenance, correction, and receipt artifact needed to replay the transition.
    """

    transition_version: Literal["sequential-state-transition-v1"] = (
        "sequential-state-transition-v1"
    )
    transition_kind: Literal["selection", "active_cost_checkpoint", "correction"]
    previous_state_sha256: str
    previous_transition_sha256: str | None = None
    candidate: CurrentAuditCandidate | None = None
    action: AuditActionPacket | None = None
    adaptive_preselection_state: AdaptivePreselectionState | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    adaptive_policy_context_sha256: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    adaptive_calibration_bundle_sha256: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    active_realized_cost: float | None = None
    adjudication: AdjudicationArtifact | None = None
    correction_provenance: VerifierCorrectionProvenance | None = None
    correction: EvidenceGraphCorrection | None = None
    receipt: AuditResolutionReceiptV2 | None = None
    post_graph: EvidenceGraph | None = None
    post_graph_sha256: str | None = None
    post_synthesis: dict[str, JsonValue] | None = None
    post_synthesis_sha256: str | None = None
    post_candidates: list[CurrentAuditCandidate] | None = None
    post_candidate_input_sha256: str | None = None
    transition_sha256: str

    @field_validator("previous_state_sha256", "transition_sha256")
    @classmethod
    def validate_required_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator(
        "previous_transition_sha256",
        "adaptive_policy_context_sha256",
        "adaptive_calibration_bundle_sha256",
        "post_graph_sha256",
        "post_synthesis_sha256",
        "post_candidate_input_sha256",
    )
    @classmethod
    def validate_optional_hash(cls, value: str | None, info: Any) -> str | None:
        return _optional_sha256(value, info.field_name)

    @field_validator("active_realized_cost")
    @classmethod
    def validate_optional_cost(cls, value: float | None) -> float | None:
        if value is not None:
            return _finite_cost(value, "sequential_transition_active_realized_cost")
        return value

    @model_validator(mode="after")
    def validate_transition(self) -> SequentialStateTransition:
        adaptive_fields = (
            self.adaptive_preselection_state,
            self.adaptive_policy_context_sha256,
            self.adaptive_calibration_bundle_sha256,
        )
        if any(value is not None for value in adaptive_fields) and not all(
            value is not None for value in adaptive_fields
        ):
            raise ValueError("adaptive_selection_checkpoint_fields_incomplete")
        if self.adaptive_preselection_state is not None and (
            self.adaptive_preselection_state.scalar_risk_score is not None
            or self.adaptive_preselection_state.score_model_sha256 is not None
        ):
            raise ValueError("adaptive_selection_checkpoint_must_be_unscored")
        selection_fields = (self.candidate, self.action)
        correction_fields = (
            self.adjudication,
            self.correction_provenance,
            self.correction,
            self.receipt,
            self.post_graph,
            self.post_graph_sha256,
            self.post_synthesis,
            self.post_synthesis_sha256,
            self.post_candidates,
            self.post_candidate_input_sha256,
        )
        if self.transition_kind == "selection":
            if any(value is None for value in selection_fields):
                raise ValueError("selection_transition_requires_candidate_and_action")
            if self.active_realized_cost is not None or any(
                value is not None for value in correction_fields
            ):
                raise ValueError("selection_transition_has_forbidden_fields")
        elif self.transition_kind == "active_cost_checkpoint":
            if self.action is None or self.active_realized_cost is None:
                raise ValueError("checkpoint_transition_requires_action_and_cost")
            if (
                any(value is not None for value in adaptive_fields)
                or self.candidate is not None
                or any(value is not None for value in correction_fields)
            ):
                raise ValueError("checkpoint_transition_has_forbidden_fields")
        else:
            if (
                any(value is not None for value in adaptive_fields)
                or self.active_realized_cost is not None
                or any(value is None for value in (*selection_fields, *correction_fields))
            ):
                raise ValueError("correction_transition_fields_incomplete")
            assert self.post_graph is not None
            assert self.post_graph_sha256 is not None
            assert self.post_synthesis is not None
            assert self.post_synthesis_sha256 is not None
            assert self.post_candidates is not None
            assert self.post_candidate_input_sha256 is not None
            if hash_canonical(self.post_graph) != self.post_graph_sha256:
                raise ValueError("correction_transition_post_graph_hash_mismatch")
            if hash_canonical(self.post_synthesis) != self.post_synthesis_sha256:
                raise ValueError("correction_transition_post_synthesis_hash_mismatch")
            expected_candidates = sorted(
                self.post_candidates,
                key=lambda candidate: (-candidate.priority, candidate.item_id),
            )
            if self.post_candidates != expected_candidates:
                raise ValueError("correction_transition_candidates_not_sorted")
            if hash_canonical(self.post_candidates) != self.post_candidate_input_sha256:
                raise ValueError("correction_transition_post_candidates_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"transition_sha256"})
        if hash_canonical(payload) != self.transition_sha256:
            raise ValueError("sequential_state_transition_hash_mismatch")
        return self


class SequentialVerificationState(ContractModel):
    """Exact scientific artifacts paired with one generic session snapshot."""

    state_version: Literal["sequential-verification-state-v2"] = (
        "sequential-verification-state-v2"
    )
    adaptive_policy_context_sha256: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    adaptive_calibration_bundle_sha256: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    session: VerificationSession
    initial_graph: EvidenceGraph
    initial_synthesis: dict[str, JsonValue]
    initial_candidates: list[CurrentAuditCandidate]
    transitions: list[SequentialStateTransition]
    graph: EvidenceGraph
    graph_sha256: str
    synthesis: dict[str, JsonValue]
    synthesis_sha256: str
    candidates: list[CurrentAuditCandidate]
    candidate_input_sha256: str
    state_sha256: str

    @field_validator(
        "graph_sha256",
        "synthesis_sha256",
        "candidate_input_sha256",
        "state_sha256",
        "adaptive_policy_context_sha256",
        "adaptive_calibration_bundle_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_state(self) -> SequentialVerificationState:
        if (self.adaptive_policy_context_sha256 is None) != (
            self.adaptive_calibration_bundle_sha256 is None
        ):
            raise ValueError("sequential_state_adaptive_commitment_incomplete")
        if hash_canonical(self.initial_graph) != self.session.initial_graph_sha256:
            raise ValueError("sequential_state_initial_graph_hash_mismatch")
        if hash_canonical(self.initial_synthesis) != self.session.initial_synthesis_sha256:
            raise ValueError("sequential_state_initial_synthesis_hash_mismatch")
        if (
            hash_canonical(self.initial_candidates)
            != self.session.initial_candidate_input_sha256
        ):
            raise ValueError("sequential_state_initial_candidate_hash_mismatch")
        if hash_canonical(self.graph) != self.graph_sha256:
            raise ValueError("sequential_state_graph_hash_mismatch")
        if hash_canonical(self.synthesis) != self.synthesis_sha256:
            raise ValueError("sequential_state_synthesis_hash_mismatch")
        expected_candidates = sorted(
            self.candidates, key=lambda candidate: (-candidate.priority, candidate.item_id)
        )
        if self.candidates != expected_candidates:
            raise ValueError("sequential_state_candidates_not_deterministically_sorted")
        candidate_ids = [candidate.item_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("sequential_state_candidate_ids_duplicate")
        if hash_canonical(self.candidates) != self.candidate_input_sha256:
            raise ValueError("sequential_state_candidate_input_hash_mismatch")
        estimate_ids = {estimate.estimate_id for estimate in self.graph.outcome_estimates}
        if not set(candidate_ids) <= estimate_ids:
            raise ValueError("sequential_state_candidate_not_in_evidence_graph")
        if any(candidate.cost_unit != self.session.cost_unit for candidate in self.candidates):
            raise ValueError("sequential_state_candidate_cost_unit_mismatch")
        if (
            self.session.current_graph_sha256 != self.graph_sha256
            or self.session.current_synthesis_sha256 != self.synthesis_sha256
            or self.session.current_candidate_input_sha256 != self.candidate_input_sha256
        ):
            raise ValueError("sequential_state_session_artifact_hash_mismatch")
        if self.session.active_action is not None and (
            self.session.active_action.item_id not in candidate_ids
        ):
            raise ValueError("sequential_state_active_candidate_missing")
        payload = self.model_dump(mode="json", exclude={"state_sha256"})
        if hash_canonical(payload) != self.state_sha256:
            raise ValueError("sequential_verification_state_hash_mismatch")
        return _validate_sequential_transition_chain(self)


def _normalize_graph(graph: EvidenceGraph) -> EvidenceGraph:
    try:
        return EvidenceGraph.model_validate(graph.model_dump(mode="json"))
    except (AttributeError, ValueError) as exc:
        raise SequentialVerificationContractError("evidence_graph_contract_invalid") from exc


def _normalize_synthesis(synthesis: Mapping[str, Any]) -> dict[str, JsonValue]:
    try:
        return _JSON_OBJECT.validate_python(dict(synthesis))
    except (TypeError, ValueError) as exc:
        raise SequentialVerificationContractError("synthesis_not_canonical_json") from exc


def _normalize_candidates(
    candidates: Sequence[CurrentAuditCandidate],
) -> list[CurrentAuditCandidate]:
    try:
        frozen = [
            CurrentAuditCandidate.model_validate(candidate.model_dump(mode="json"))
            for candidate in candidates
        ]
    except (AttributeError, ValueError) as exc:
        raise SequentialVerificationContractError("current_candidate_contract_invalid") from exc
    if len({candidate.item_id for candidate in frozen}) != len(frozen):
        raise SequentialVerificationContractError("current_candidate_id_duplicate")
    return sorted(frozen, key=lambda candidate: (-candidate.priority, candidate.item_id))


def _state_payload(
    *,
    session: VerificationSession,
    initial_graph: EvidenceGraph,
    initial_synthesis: Mapping[str, Any],
    initial_candidates: Sequence[CurrentAuditCandidate],
    transitions: Sequence[SequentialStateTransition],
    graph: EvidenceGraph,
    synthesis: Mapping[str, Any],
    candidates: Sequence[CurrentAuditCandidate],
    adaptive_policy_context_sha256: str | None,
    adaptive_calibration_bundle_sha256: str | None,
) -> dict[str, Any]:
    payload = {
        "state_version": "sequential-verification-state-v2",
        "session": session,
        "initial_graph": initial_graph,
        "initial_synthesis": dict(initial_synthesis),
        "initial_candidates": list(initial_candidates),
        "transitions": list(transitions),
        "graph": graph,
        "graph_sha256": hash_canonical(graph),
        "synthesis": dict(synthesis),
        "synthesis_sha256": hash_canonical(synthesis),
        "candidates": list(candidates),
        "candidate_input_sha256": hash_canonical(candidates),
    }
    if adaptive_policy_context_sha256 is not None:
        if adaptive_calibration_bundle_sha256 is None:
            raise SequentialVerificationContractError(
                "sequential_state_adaptive_commitment_incomplete"
            )
        payload.update(
            {
                "adaptive_policy_context_sha256": (
                    adaptive_policy_context_sha256
                ),
                "adaptive_calibration_bundle_sha256": (
                    adaptive_calibration_bundle_sha256
                ),
            }
        )
    elif adaptive_calibration_bundle_sha256 is not None:
        raise SequentialVerificationContractError(
            "sequential_state_adaptive_commitment_incomplete"
        )
    return payload


def _freeze_transition(payload: Mapping[str, Any]) -> SequentialStateTransition:
    normalized = {
        "transition_version": "sequential-state-transition-v1",
        "previous_transition_sha256": None,
        "candidate": None,
        "action": None,
        "active_realized_cost": None,
        "adjudication": None,
        "correction_provenance": None,
        "correction": None,
        "receipt": None,
        "post_graph": None,
        "post_graph_sha256": None,
        "post_synthesis": None,
        "post_synthesis_sha256": None,
        "post_candidates": None,
        "post_candidate_input_sha256": None,
        **payload,
    }
    return SequentialStateTransition.model_validate(
        {
            **normalized,
            "transition_sha256": hash_canonical(normalized),
        }
    )


def _validate_adaptive_checkpoint_against_preselection(
    *,
    checkpoint: AdaptivePreselectionState,
    state_sha256: str,
    session: VerificationSession,
    graph_sha256: str,
    synthesis_sha256: str,
) -> None:
    """Bind one unscored adaptive checkpoint to the exact preselection snapshot."""

    if checkpoint.scalar_risk_score is not None or checkpoint.score_model_sha256 is not None:
        raise ValueError("adaptive_selection_checkpoint_must_be_unscored")
    if checkpoint.scheduler_state_sha256 != state_sha256:
        raise ValueError("adaptive_selection_scheduler_state_hash_mismatch")
    if checkpoint.audit_prefix_item_ids != list(session.resolved_item_ids):
        raise ValueError("adaptive_selection_resolved_prefix_identity_mismatch")
    if not math.isclose(
        checkpoint.audit_prefix_cost_minutes,
        session.historical_realized_cost,
        rel_tol=1e-12,
        abs_tol=_COST_TOLERANCE,
    ):
        raise ValueError("adaptive_selection_resolved_prefix_cost_mismatch")
    if checkpoint.evidence_graph_sha256 != graph_sha256:
        raise ValueError("adaptive_selection_evidence_graph_hash_mismatch")
    if checkpoint.synthesis_sha256 != synthesis_sha256:
        raise ValueError("adaptive_selection_synthesis_hash_mismatch")


def _scheduler_payload(
    *,
    session: VerificationSession,
    state_sha256: str,
    candidate_input_sha256: str,
    selected: CurrentAuditCandidate,
    selection_rank: int,
    adaptive_preselection_state: AdaptivePreselectionState | None = None,
    adaptive_policy_context_sha256: str | None = None,
    adaptive_calibration_bundle_sha256: str | None = None,
) -> dict[str, JsonValue]:
    """Build the exact action-scheduler identity, including adaptive lineage when used."""

    payload: dict[str, JsonValue] = {
        "scheduler_version": "sequential-verifier-highest-eligible-v1",
        "selection_state_sha256": state_sha256,
        "session_sha256": session.session_sha256,
        "policy_sha256": session.policy_sha256,
        "candidate_input_sha256": candidate_input_sha256,
        "selected_candidate_sha256": selected.candidate_sha256,
        "selected_item_id": selected.item_id,
        "selection_rank": selection_rank,
        "selection_priority": selected.priority,
        "remaining_budget": session.remaining_budget,
    }
    if adaptive_preselection_state is not None:
        if (
            adaptive_policy_context_sha256 is None
            or adaptive_calibration_bundle_sha256 is None
        ):
            raise ValueError("adaptive_selection_checkpoint_fields_incomplete")
        payload.update(
            {
                "adaptive_scheduler_binding_version": (
                    "sequential-adaptive-preselection-binding-v1"
                ),
                "pipeline_sha256": session.pipeline_sha256,
                "adaptive_preselection_state_sha256": (
                    adaptive_preselection_state.state_sha256
                ),
                "adaptive_policy_context_sha256": adaptive_policy_context_sha256,
                "adaptive_calibration_bundle_sha256": (
                    adaptive_calibration_bundle_sha256
                ),
            }
        )
    return payload


def _replay_selection_transition(
    *,
    session: VerificationSession,
    state_sha256: str,
    graph_sha256: str,
    synthesis_sha256: str,
    candidates: Sequence[CurrentAuditCandidate],
    transition: SequentialStateTransition,
) -> VerificationSession:
    if transition.candidate is None or transition.action is None:
        raise ValueError("selection_transition_fields_missing")
    if session.active_action is not None:
        raise ValueError("selection_transition_predecessor_has_active_action")
    selectable = [
        candidate
        for candidate in candidates
        if candidate.eligible
        and candidate.item_id not in session.selected_item_ids
        and candidate.estimated_cost <= session.remaining_budget + _COST_TOLERANCE
    ]
    if not selectable or transition.candidate != selectable[0]:
        raise ValueError("selection_transition_not_deterministic_next_candidate")
    selected = selectable[0]
    selection_rank = list(candidates).index(selected) + 1
    checkpoint = transition.adaptive_preselection_state
    if checkpoint is not None:
        _validate_adaptive_checkpoint_against_preselection(
            checkpoint=checkpoint,
            state_sha256=state_sha256,
            session=session,
            graph_sha256=graph_sha256,
            synthesis_sha256=synthesis_sha256,
        )
    scheduler_payload = _scheduler_payload(
        session=session,
        state_sha256=state_sha256,
        candidate_input_sha256=hash_canonical(candidates),
        selected=selected,
        selection_rank=selection_rank,
        adaptive_preselection_state=checkpoint,
        adaptive_policy_context_sha256=transition.adaptive_policy_context_sha256,
        adaptive_calibration_bundle_sha256=(
            transition.adaptive_calibration_bundle_sha256
        ),
    )
    selected_session, action = select_audit_action(
        session,
        expected_state_sha256=session.session_sha256,
        item_id=selected.item_id,
        scheduler_artifact_sha256=hash_canonical(scheduler_payload),
        estimated_cost=selected.estimated_cost,
        selected_at=transition.action.selected_at,
        selection_rank=selection_rank,
        selection_score=selected.priority,
    )
    if action != transition.action:
        raise ValueError("selection_transition_action_replay_mismatch")
    return selected_session


def _replay_correction_transition(
    *,
    session: VerificationSession,
    graph: EvidenceGraph,
    synthesis: Mapping[str, Any],
    candidates: Sequence[CurrentAuditCandidate],
    transition: SequentialStateTransition,
) -> tuple[
    VerificationSession,
    EvidenceGraph,
    dict[str, JsonValue],
    list[CurrentAuditCandidate],
]:
    action = session.active_action
    if action is None:
        raise ValueError("correction_transition_requires_active_action")
    if (
        transition.action is None
        or transition.candidate is None
        or transition.adjudication is None
        or transition.correction_provenance is None
        or transition.correction is None
        or transition.receipt is None
        or transition.post_graph is None
        or transition.post_synthesis is None
        or transition.post_candidates is None
    ):
        raise ValueError("correction_transition_fields_missing")
    if transition.action != action:
        raise ValueError("correction_transition_action_replay_mismatch")
    selected = next(
        (candidate for candidate in candidates if candidate.item_id == action.item_id),
        None,
    )
    if selected is None or transition.candidate != selected:
        raise ValueError("correction_transition_candidate_replay_mismatch")
    selected_before, selected_after = _validate_selected_estimate_correction(
        before=graph,
        after=transition.post_graph,
        selected_item_id=action.item_id,
        disposition=transition.correction.disposition,
    )
    post_graph_sha256 = hash_canonical(transition.post_graph)
    post_synthesis_sha256 = hash_canonical(transition.post_synthesis)
    post_candidate_sha256 = hash_canonical(transition.post_candidates)
    provenance = transition.correction_provenance
    if (
        provenance.session_id,
        provenance.step,
        provenance.item_id,
        provenance.action_packet_sha256,
        provenance.selected_candidate_sha256,
        provenance.adjudication_artifact_sha256,
        provenance.adjudication_payload_sha256,
        provenance.disposition,
        provenance.selected_estimate_before_sha256,
        provenance.selected_estimate_after_sha256,
        provenance.pre_graph_sha256,
        provenance.post_graph_sha256,
        provenance.post_synthesis_sha256,
        provenance.post_candidate_input_sha256,
    ) != (
        session.session_id,
        action.step,
        action.item_id,
        action.packet_sha256,
        selected.candidate_sha256,
        transition.adjudication.artifact_sha256,
        transition.adjudication.payload_sha256,
        transition.correction.disposition,
        selected_before,
        selected_after,
        hash_canonical(graph),
        post_graph_sha256,
        post_synthesis_sha256,
        post_candidate_sha256,
    ):
        raise ValueError("correction_transition_provenance_replay_mismatch")
    expected_correction = freeze_evidence_graph_correction(
        action,
        transition.adjudication,
        disposition=transition.correction.disposition,
        post_graph_sha256=post_graph_sha256,
        post_synthesis_sha256=post_synthesis_sha256,
        post_candidate_input_sha256=post_candidate_sha256,
        correction_artifact_sha256=provenance.provenance_sha256,
    )
    if expected_correction != transition.correction:
        raise ValueError("correction_transition_correction_replay_mismatch")
    resolved_session, receipt = resolve_audit_action(
        session,
        expected_state_sha256=session.session_sha256,
        adjudication=transition.adjudication,
        correction=expected_correction,
    )
    if receipt != transition.receipt:
        raise ValueError("correction_transition_receipt_replay_mismatch")
    return (
        resolved_session,
        transition.post_graph,
        dict(transition.post_synthesis),
        list(transition.post_candidates),
    )


def _validate_sequential_transition_chain(
    state: SequentialVerificationState,
) -> SequentialVerificationState:
    """Reconstruct every session mutation and scientific correction from genesis."""

    initial_candidates = _normalize_candidates(state.initial_candidates)
    if initial_candidates != state.initial_candidates:
        raise ValueError("sequential_state_initial_candidates_not_sorted")
    session = create_verification_session(
        session_id=state.session.session_id,
        created_at=state.session.created_at,
        pipeline_sha256=state.session.pipeline_sha256,
        policy_sha256=state.session.policy_sha256,
        budget=state.session.budget,
        cost_unit=state.session.cost_unit,
        graph_sha256=hash_canonical(state.initial_graph),
        synthesis_sha256=hash_canonical(state.initial_synthesis),
        candidate_input_sha256=hash_canonical(initial_candidates),
    )
    graph = state.initial_graph
    synthesis: dict[str, JsonValue] = dict(state.initial_synthesis)
    candidates = initial_candidates
    prefix: list[SequentialStateTransition] = []
    previous_transition_sha256: str | None = None
    adaptive_selection_mode = state.adaptive_policy_context_sha256 is not None
    adaptive_policy_context_sha256 = state.adaptive_policy_context_sha256
    adaptive_calibration_bundle_sha256 = (
        state.adaptive_calibration_bundle_sha256
    )
    for transition in state.transitions:
        pre_payload = _state_payload(
            session=session,
            initial_graph=state.initial_graph,
            initial_synthesis=state.initial_synthesis,
            initial_candidates=state.initial_candidates,
            transitions=prefix,
            graph=graph,
            synthesis=synthesis,
            candidates=candidates,
            adaptive_policy_context_sha256=(
                state.adaptive_policy_context_sha256
            ),
            adaptive_calibration_bundle_sha256=(
                state.adaptive_calibration_bundle_sha256
            ),
        )
        pre_state_sha256 = hash_canonical(pre_payload)
        if transition.previous_state_sha256 != pre_state_sha256:
            raise ValueError("sequential_transition_predecessor_state_mismatch")
        if transition.previous_transition_sha256 != previous_transition_sha256:
            raise ValueError("sequential_transition_predecessor_chain_mismatch")
        if transition.transition_kind == "selection":
            has_adaptive_checkpoint = transition.adaptive_preselection_state is not None
            if has_adaptive_checkpoint != adaptive_selection_mode:
                if has_adaptive_checkpoint:
                    raise ValueError(
                        "adaptive_selection_cannot_activate_after_state_genesis"
                    )
                raise ValueError("adaptive_state_selection_checkpoint_missing")
            if has_adaptive_checkpoint and (
                transition.adaptive_policy_context_sha256
                != adaptive_policy_context_sha256
            ):
                raise ValueError("adaptive_selection_policy_context_changed")
            if has_adaptive_checkpoint and (
                transition.adaptive_calibration_bundle_sha256
                != adaptive_calibration_bundle_sha256
            ):
                raise ValueError("adaptive_selection_calibration_bundle_changed")
            session = _replay_selection_transition(
                session=session,
                state_sha256=pre_state_sha256,
                graph_sha256=hash_canonical(graph),
                synthesis_sha256=hash_canonical(synthesis),
                candidates=candidates,
                transition=transition,
            )
        elif transition.transition_kind == "active_cost_checkpoint":
            if transition.action != session.active_action:
                raise ValueError("checkpoint_transition_action_replay_mismatch")
            assert transition.active_realized_cost is not None
            session = checkpoint_active_cost(
                session,
                expected_state_sha256=session.session_sha256,
                action_packet_sha256=transition.action.packet_sha256,
                active_realized_cost=transition.active_realized_cost,
            )
        else:
            session, graph, synthesis, candidates = _replay_correction_transition(
                session=session,
                graph=graph,
                synthesis=synthesis,
                candidates=candidates,
                transition=transition,
            )
        prefix.append(transition)
        previous_transition_sha256 = transition.transition_sha256
    if session != state.session:
        raise ValueError("sequential_transition_final_session_mismatch")
    if (
        graph != state.graph
        or synthesis != state.synthesis
        or candidates != state.candidates
    ):
        raise ValueError("sequential_transition_final_scientific_state_mismatch")
    return state


def _freeze_state(
    *,
    session: VerificationSession,
    graph: EvidenceGraph,
    synthesis: Mapping[str, Any],
    candidates: Sequence[CurrentAuditCandidate],
    adaptive_policy_context_sha256: str | None = None,
    adaptive_calibration_bundle_sha256: str | None = None,
    predecessor: SequentialVerificationState | None = None,
    transition: SequentialStateTransition | None = None,
) -> SequentialVerificationState:
    frozen_graph = _normalize_graph(graph)
    frozen_synthesis = _normalize_synthesis(synthesis)
    frozen_candidates = _normalize_candidates(candidates)
    if (predecessor is None) != (transition is None):
        raise SequentialVerificationContractError(
            "state_transition_requires_predecessor_and_transition"
        )
    if predecessor is None:
        initial_graph = frozen_graph
        initial_synthesis = frozen_synthesis
        initial_candidates = frozen_candidates
        transitions: list[SequentialStateTransition] = []
    else:
        if (
            adaptive_policy_context_sha256 is not None
            or adaptive_calibration_bundle_sha256 is not None
        ):
            raise SequentialVerificationContractError(
                "state_transition_cannot_replace_adaptive_commitment"
            )
        adaptive_policy_context_sha256 = (
            predecessor.adaptive_policy_context_sha256
        )
        adaptive_calibration_bundle_sha256 = (
            predecessor.adaptive_calibration_bundle_sha256
        )
        initial_graph = predecessor.initial_graph
        initial_synthesis = predecessor.initial_synthesis
        initial_candidates = predecessor.initial_candidates
        assert transition is not None
        transitions = [*predecessor.transitions, transition]
    payload = _state_payload(
        session=resume_verification_session(session),
        initial_graph=initial_graph,
        initial_synthesis=initial_synthesis,
        initial_candidates=initial_candidates,
        transitions=transitions,
        graph=frozen_graph,
        synthesis=frozen_synthesis,
        candidates=frozen_candidates,
        adaptive_policy_context_sha256=adaptive_policy_context_sha256,
        adaptive_calibration_bundle_sha256=adaptive_calibration_bundle_sha256,
    )
    return SequentialVerificationState.model_validate(
        {**payload, "state_sha256": hash_canonical(payload)}
    )


def create_sequential_verification_state(
    *,
    session_id: str,
    created_at: datetime,
    pipeline_sha256: str,
    policy_sha256: str,
    budget: float,
    cost_unit: str,
    graph: EvidenceGraph,
    synthesis: Mapping[str, Any],
    candidates: Sequence[CurrentAuditCandidate],
    adaptive_policy_context_sha256: str | None = None,
    adaptive_calibration_bundle_sha256: str | None = None,
) -> SequentialVerificationState:
    """Compute scientific artifact hashes and create their bound audit session."""

    frozen_graph = _normalize_graph(graph)
    frozen_synthesis = _normalize_synthesis(synthesis)
    frozen_candidates = _normalize_candidates(candidates)
    session = create_verification_session(
        session_id=session_id,
        created_at=created_at,
        pipeline_sha256=pipeline_sha256,
        policy_sha256=policy_sha256,
        budget=budget,
        cost_unit=cost_unit,
        graph_sha256=hash_canonical(frozen_graph),
        synthesis_sha256=hash_canonical(frozen_synthesis),
        candidate_input_sha256=hash_canonical(frozen_candidates),
    )
    return _freeze_state(
        session=session,
        graph=frozen_graph,
        synthesis=frozen_synthesis,
        candidates=frozen_candidates,
        adaptive_policy_context_sha256=adaptive_policy_context_sha256,
        adaptive_calibration_bundle_sha256=adaptive_calibration_bundle_sha256,
    )


def resume_sequential_verification_state(
    state: SequentialVerificationState | Mapping[str, Any],
) -> SequentialVerificationState:
    """Revalidate the session chain and all nested scientific artifact hashes."""

    raw = state.model_dump(mode="json") if isinstance(state, SequentialVerificationState) else state
    try:
        return SequentialVerificationState.model_validate(raw)
    except ValueError as exc:
        raise SequentialVerificationContractError(
            "sequential_verification_state_integrity_changed"
        ) from exc


def adaptive_preselection_history_from_state(
    state: SequentialVerificationState | Mapping[str, Any],
) -> tuple[tuple[AdaptivePreselectionState, ...], str | None, str | None]:
    """Return the append-only adaptive checkpoints and their frozen identities.

    The state is fully replayed before anything is returned.  A non-adaptive state
    returns ``((), None, None)``.  The full policy context and calibration bundle
    remain verifier-owned; this ledger freezes and exposes their exact hashes so the
    verifier can bind them to the validated objects and pipeline identity.
    """

    current = resume_sequential_verification_state(state)
    selections = [
        transition
        for transition in current.transitions
        if transition.transition_kind == "selection"
    ]
    checkpoints = tuple(
        transition.adaptive_preselection_state
        for transition in selections
        if transition.adaptive_preselection_state is not None
    )
    if not checkpoints:
        return (
            (),
            current.adaptive_policy_context_sha256,
            current.adaptive_calibration_bundle_sha256,
        )
    first = selections[0]
    if (
        first.adaptive_policy_context_sha256 is None
        or first.adaptive_calibration_bundle_sha256 is None
    ):
        raise SequentialVerificationContractError(
            "adaptive_selection_checkpoint_fields_incomplete"
        )
    if (
        first.adaptive_policy_context_sha256
        != current.adaptive_policy_context_sha256
        or first.adaptive_calibration_bundle_sha256
        != current.adaptive_calibration_bundle_sha256
    ):
        raise SequentialVerificationContractError(
            "adaptive_selection_state_commitment_mismatch"
        )
    return (
        checkpoints,
        first.adaptive_policy_context_sha256,
        first.adaptive_calibration_bundle_sha256,
    )


def selection_predecessor_states_from_state(
    state: SequentialVerificationState | Mapping[str, Any],
) -> tuple[SequentialVerificationState, ...]:
    """Reconstruct the exact no-active state immediately before every selection.

    The returned snapshots are derived from the replayed transition chain rather than
    accepted as detached history. They let the unified verifier recompute the complete
    non-calibration assessment behind every adaptive checkpoint, including its feature
    row and gate ledger.
    """

    current = resume_sequential_verification_state(state)
    initial_candidates = _normalize_candidates(current.initial_candidates)
    session = create_verification_session(
        session_id=current.session.session_id,
        created_at=current.session.created_at,
        pipeline_sha256=current.session.pipeline_sha256,
        policy_sha256=current.session.policy_sha256,
        budget=current.session.budget,
        cost_unit=current.session.cost_unit,
        graph_sha256=hash_canonical(current.initial_graph),
        synthesis_sha256=hash_canonical(current.initial_synthesis),
        candidate_input_sha256=hash_canonical(initial_candidates),
    )
    graph = current.initial_graph
    synthesis: dict[str, JsonValue] = dict(current.initial_synthesis)
    candidates = initial_candidates
    prefix: list[SequentialStateTransition] = []
    snapshots: list[SequentialVerificationState] = []
    for transition in current.transitions:
        pre_payload = _state_payload(
            session=session,
            initial_graph=current.initial_graph,
            initial_synthesis=current.initial_synthesis,
            initial_candidates=current.initial_candidates,
            transitions=prefix,
            graph=graph,
            synthesis=synthesis,
            candidates=candidates,
            adaptive_policy_context_sha256=(
                current.adaptive_policy_context_sha256
            ),
            adaptive_calibration_bundle_sha256=(
                current.adaptive_calibration_bundle_sha256
            ),
        )
        pre_state = SequentialVerificationState.model_validate(
            {**pre_payload, "state_sha256": hash_canonical(pre_payload)}
        )
        if transition.transition_kind == "selection":
            snapshots.append(pre_state)
            session = _replay_selection_transition(
                session=session,
                state_sha256=pre_state.state_sha256,
                graph_sha256=pre_state.graph_sha256,
                synthesis_sha256=pre_state.synthesis_sha256,
                candidates=candidates,
                transition=transition,
            )
        elif transition.transition_kind == "active_cost_checkpoint":
            if transition.action != session.active_action:
                raise SequentialVerificationContractError(
                    "checkpoint_transition_action_replay_mismatch"
                )
            assert transition.active_realized_cost is not None
            session = checkpoint_active_cost(
                session,
                expected_state_sha256=session.session_sha256,
                action_packet_sha256=transition.action.packet_sha256,
                active_realized_cost=transition.active_realized_cost,
            )
        else:
            session, graph, synthesis, candidates = _replay_correction_transition(
                session=session,
                graph=graph,
                synthesis=synthesis,
                candidates=candidates,
                transition=transition,
            )
        prefix.append(transition)
    return tuple(snapshots)


class SequentialStateExpectation(ContractModel):
    """Self-hashed optimistic-lock token covering every mutable verifier artifact."""

    expectation_version: Literal["sequential-state-expectation-v1"] = (
        "sequential-state-expectation-v1"
    )
    state_sha256: str
    session_sha256: str
    graph_sha256: str
    synthesis_sha256: str
    candidate_input_sha256: str
    active_action_packet_sha256: str | None
    expectation_sha256: str

    @field_validator(
        "state_sha256",
        "session_sha256",
        "graph_sha256",
        "synthesis_sha256",
        "candidate_input_sha256",
        "expectation_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("active_action_packet_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return _optional_sha256(value, "active_action_packet_sha256")

    @model_validator(mode="after")
    def validate_expectation(self) -> SequentialStateExpectation:
        payload = self.model_dump(mode="json", exclude={"expectation_sha256"})
        if hash_canonical(payload) != self.expectation_sha256:
            raise ValueError("sequential_state_expectation_hash_mismatch")
        return self


def freeze_state_expectation(
    state: SequentialVerificationState,
) -> SequentialStateExpectation:
    """Create the exact optimistic-lock token required by the next transition."""

    current = resume_sequential_verification_state(state)
    payload = {
        "expectation_version": "sequential-state-expectation-v1",
        "state_sha256": current.state_sha256,
        "session_sha256": current.session.session_sha256,
        "graph_sha256": current.graph_sha256,
        "synthesis_sha256": current.synthesis_sha256,
        "candidate_input_sha256": current.candidate_input_sha256,
        "active_action_packet_sha256": (
            None
            if current.session.active_action is None
            else current.session.active_action.packet_sha256
        ),
    }
    return SequentialStateExpectation.model_validate(
        {**payload, "expectation_sha256": hash_canonical(payload)}
    )


def _require_expected_state(
    state: SequentialVerificationState,
    expected: SequentialStateExpectation,
) -> SequentialVerificationState:
    current = resume_sequential_verification_state(state)
    try:
        token = SequentialStateExpectation.model_validate(expected.model_dump(mode="json"))
    except (AttributeError, ValueError) as exc:
        raise StaleSequentialVerificationStateError(
            "sequential_state_expectation_invalid"
        ) from exc
    observed = {
        "state": current.state_sha256,
        "session": current.session.session_sha256,
        "graph": current.graph_sha256,
        "synthesis": current.synthesis_sha256,
        "candidate_input": current.candidate_input_sha256,
        "active_action_packet": (
            None
            if current.session.active_action is None
            else current.session.active_action.packet_sha256
        ),
    }
    expected_values = {
        "state": token.state_sha256,
        "session": token.session_sha256,
        "graph": token.graph_sha256,
        "synthesis": token.synthesis_sha256,
        "candidate_input": token.candidate_input_sha256,
        "active_action_packet": token.active_action_packet_sha256,
    }
    for name in (
        "state",
        "session",
        "graph",
        "synthesis",
        "candidate_input",
        "active_action_packet",
    ):
        if observed[name] != expected_values[name]:
            raise StaleSequentialVerificationStateError(
                f"stale_sequential_{name}_hash"
            )
    return current


class SequentialSelectionResult(ContractModel):
    result_version: Literal["sequential-selection-result-v1"] = (
        "sequential-selection-result-v1"
    )
    previous_state_sha256: str
    state: SequentialVerificationState
    candidate: CurrentAuditCandidate
    action: AuditActionPacket
    result_sha256: str

    @field_validator("previous_state_sha256", "result_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_result(self) -> SequentialSelectionResult:
        if not self.state.transitions:
            raise ValueError("selection_result_transition_missing")
        transition = self.state.transitions[-1]
        if (
            transition.transition_kind != "selection"
            or transition.previous_state_sha256 != self.previous_state_sha256
            or transition.candidate != self.candidate
            or transition.action != self.action
        ):
            raise ValueError("selection_result_transition_mismatch")
        if self.state.session.active_action != self.action:
            raise ValueError("selection_result_action_not_active")
        if self.action.item_id != self.candidate.item_id:
            raise ValueError("selection_result_candidate_action_mismatch")
        if self.action.selection_state_sha256 != self.state.session.previous_session_sha256:
            raise ValueError("selection_result_prior_session_mismatch")
        payload = self.model_dump(mode="json", exclude={"result_sha256"})
        if hash_canonical(payload) != self.result_sha256:
            raise ValueError("sequential_selection_result_hash_mismatch")
        return self


def select_next_audit_candidate(
    state: SequentialVerificationState,
    *,
    expected: SequentialStateExpectation,
    selected_at: datetime,
    adaptive_preselection_state: AdaptivePreselectionState | None = None,
    adaptive_policy_context_sha256: str | None = None,
    adaptive_calibration_bundle_sha256: str | None = None,
) -> SequentialSelectionResult:
    """Select the highest-priority unresolved eligible candidate that currently fits."""

    current = _require_expected_state(state, expected)
    adaptive_fields = (
        adaptive_preselection_state,
        adaptive_policy_context_sha256,
        adaptive_calibration_bundle_sha256,
    )
    if any(value is not None for value in adaptive_fields) and not all(
        value is not None for value in adaptive_fields
    ):
        raise SequentialVerificationContractError(
            "adaptive_selection_checkpoint_fields_incomplete"
        )
    frozen_adaptive_state: AdaptivePreselectionState | None = None
    if adaptive_preselection_state is not None:
        try:
            frozen_adaptive_state = AdaptivePreselectionState.model_validate(
                adaptive_preselection_state.model_dump(mode="json")
            )
            if (
                adaptive_policy_context_sha256 is None
                or adaptive_calibration_bundle_sha256 is None
            ):
                raise ValueError("adaptive_selection_checkpoint_fields_incomplete")
            _sha256(
                adaptive_policy_context_sha256,
                "adaptive_policy_context_sha256",
            )
            _sha256(
                adaptive_calibration_bundle_sha256,
                "adaptive_calibration_bundle_sha256",
            )
            _validate_adaptive_checkpoint_against_preselection(
                checkpoint=frozen_adaptive_state,
                state_sha256=current.state_sha256,
                session=current.session,
                graph_sha256=current.graph_sha256,
                synthesis_sha256=current.synthesis_sha256,
            )
        except (AttributeError, ValueError) as exc:
            raise SequentialVerificationContractError(str(exc)) from exc
    prior_selections = [
        transition
        for transition in current.transitions
        if transition.transition_kind == "selection"
    ]
    state_is_adaptive = current.adaptive_policy_context_sha256 is not None
    current_selection_is_adaptive = frozen_adaptive_state is not None
    if state_is_adaptive != current_selection_is_adaptive:
        if current_selection_is_adaptive:
            raise SequentialVerificationContractError(
                "adaptive_selection_cannot_activate_after_state_genesis"
            )
        raise SequentialVerificationContractError(
            "adaptive_selection_history_checkpoint_removed"
            if prior_selections
            else "adaptive_state_selection_checkpoint_missing"
        )
    if state_is_adaptive and (
        adaptive_policy_context_sha256
        != current.adaptive_policy_context_sha256
        or adaptive_calibration_bundle_sha256
        != current.adaptive_calibration_bundle_sha256
    ):
        if (
            prior_selections
            and adaptive_policy_context_sha256
            != prior_selections[0].adaptive_policy_context_sha256
        ):
            raise SequentialVerificationContractError(
                "adaptive_selection_policy_context_changed"
            )
        if (
            prior_selections
            and adaptive_calibration_bundle_sha256
            != prior_selections[0].adaptive_calibration_bundle_sha256
        ):
            raise SequentialVerificationContractError(
                "adaptive_selection_calibration_bundle_changed"
            )
        raise SequentialVerificationContractError(
            "adaptive_selection_state_commitment_mismatch"
        )
    if prior_selections:
        prior_is_adaptive = (
            prior_selections[0].adaptive_preselection_state is not None
        )
        current_is_adaptive = frozen_adaptive_state is not None
        if current_is_adaptive != prior_is_adaptive:
            if current_is_adaptive:
                raise SequentialVerificationContractError(
                    "adaptive_selection_history_cannot_activate_midstream"
                )
            raise SequentialVerificationContractError(
                "adaptive_selection_history_checkpoint_removed"
            )
        if current_is_adaptive:
            if (
                adaptive_policy_context_sha256
                != prior_selections[0].adaptive_policy_context_sha256
            ):
                raise SequentialVerificationContractError(
                    "adaptive_selection_policy_context_changed"
                )
            if (
                adaptive_calibration_bundle_sha256
                != prior_selections[0].adaptive_calibration_bundle_sha256
            ):
                raise SequentialVerificationContractError(
                    "adaptive_selection_calibration_bundle_changed"
                )
    if current.session.status is VerificationSessionStatus.FINALIZED:
        raise SequentialVerificationContractError("finalized_session_is_terminal")
    if current.session.active_action is not None:
        raise SequentialVerificationContractError("session_already_has_active_action")
    if current.session.budget == 0 or current.session.remaining_budget == 0:
        raise SequentialVerificationContractError("zero_budget_selection_forbidden")
    selectable = [
        candidate
        for candidate in current.candidates
        if candidate.eligible
        and candidate.item_id not in current.session.selected_item_ids
        and candidate.estimated_cost
        <= current.session.remaining_budget + _COST_TOLERANCE
    ]
    if not selectable:
        raise SequentialVerificationContractError("no_eligible_candidate_fits_remaining_budget")
    selected = selectable[0]
    selection_rank = current.candidates.index(selected) + 1
    scheduler_payload = _scheduler_payload(
        session=current.session,
        state_sha256=current.state_sha256,
        candidate_input_sha256=current.candidate_input_sha256,
        selected=selected,
        selection_rank=selection_rank,
        adaptive_preselection_state=frozen_adaptive_state,
        adaptive_policy_context_sha256=adaptive_policy_context_sha256,
        adaptive_calibration_bundle_sha256=adaptive_calibration_bundle_sha256,
    )
    scheduler_sha256 = hash_canonical(scheduler_payload)
    try:
        selected_session, action = select_audit_action(
            current.session,
            expected_state_sha256=current.session.session_sha256,
            item_id=selected.item_id,
            scheduler_artifact_sha256=scheduler_sha256,
            estimated_cost=selected.estimated_cost,
            selected_at=selected_at,
            selection_rank=selection_rank,
            selection_score=selected.priority,
        )
    except AuditSessionContractError as exc:
        raise SequentialVerificationContractError(str(exc)) from exc
    transition_payload: dict[str, Any] = {
        "transition_kind": "selection",
        "previous_state_sha256": current.state_sha256,
        "previous_transition_sha256": (
            current.transitions[-1].transition_sha256
            if current.transitions
            else None
        ),
        "candidate": selected,
        "action": action,
    }
    if frozen_adaptive_state is not None:
        transition_payload.update(
            {
                "adaptive_preselection_state": frozen_adaptive_state,
                "adaptive_policy_context_sha256": adaptive_policy_context_sha256,
                "adaptive_calibration_bundle_sha256": (
                    adaptive_calibration_bundle_sha256
                ),
            }
        )
    transition = _freeze_transition(transition_payload)
    updated = _freeze_state(
        session=selected_session,
        graph=current.graph,
        synthesis=current.synthesis,
        candidates=current.candidates,
        predecessor=current,
        transition=transition,
    )
    payload = {
        "result_version": "sequential-selection-result-v1",
        "previous_state_sha256": current.state_sha256,
        "state": updated,
        "candidate": selected,
        "action": action,
    }
    return SequentialSelectionResult.model_validate(
        {**payload, "result_sha256": hash_canonical(payload)}
    )


class SequentialActiveCostCheckpointResult(ContractModel):
    """Hash-bound partial-time checkpoint for one unresolved selected action."""

    result_version: Literal["sequential-active-cost-checkpoint-v1"] = (
        "sequential-active-cost-checkpoint-v1"
    )
    previous_state_sha256: str
    state: SequentialVerificationState
    action_packet_sha256: str
    active_realized_cost: float
    result_sha256: str

    @field_validator(
        "previous_state_sha256", "action_packet_sha256", "result_sha256"
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("active_realized_cost")
    @classmethod
    def validate_active_cost(cls, value: float) -> float:
        return _finite_cost(value, "sequential_active_realized_cost")

    @model_validator(mode="after")
    def validate_result(self) -> SequentialActiveCostCheckpointResult:
        if not self.state.transitions:
            raise ValueError("active_cost_checkpoint_transition_missing")
        transition = self.state.transitions[-1]
        if (
            transition.transition_kind != "active_cost_checkpoint"
            or transition.previous_state_sha256 != self.previous_state_sha256
            or transition.action is None
            or transition.action.packet_sha256 != self.action_packet_sha256
            or transition.active_realized_cost != self.active_realized_cost
        ):
            raise ValueError("active_cost_checkpoint_transition_mismatch")
        action = self.state.session.active_action
        if action is None:
            raise ValueError("active_cost_checkpoint_action_missing")
        if action.packet_sha256 != self.action_packet_sha256:
            raise ValueError("active_cost_checkpoint_action_hash_mismatch")
        if not math.isclose(
            self.state.session.active_realized_cost,
            self.active_realized_cost,
            rel_tol=1e-12,
            abs_tol=_COST_TOLERANCE,
        ):
            raise ValueError("active_cost_checkpoint_value_mismatch")
        payload = self.model_dump(mode="json", exclude={"result_sha256"})
        if hash_canonical(payload) != self.result_sha256:
            raise ValueError("sequential_active_cost_checkpoint_hash_mismatch")
        return self


def checkpoint_selected_audit_cost(
    state: SequentialVerificationState,
    *,
    expected: SequentialStateExpectation,
    active_realized_cost: float,
) -> SequentialActiveCostCheckpointResult:
    """Charge measured partial time while leaving the selected action unresolved.

    This is the production representation of an action that is still underway when a
    hard budget deadline is reached.  It never applies an adjudication or mutates the
    evidence graph, and the active action continues to block release.
    """

    current = _require_expected_state(state, expected)
    action = current.session.active_action
    if action is None:
        raise SequentialVerificationContractError(
            "active_cost_checkpoint_requires_selected_action"
        )
    _finite_cost(active_realized_cost, "sequential_active_realized_cost")
    try:
        checkpointed_session = checkpoint_active_cost(
            current.session,
            expected_state_sha256=current.session.session_sha256,
            action_packet_sha256=action.packet_sha256,
            active_realized_cost=active_realized_cost,
        )
    except AuditSessionContractError as exc:
        raise SequentialVerificationContractError(str(exc)) from exc
    transition = _freeze_transition(
        {
            "transition_kind": "active_cost_checkpoint",
            "previous_state_sha256": current.state_sha256,
            "previous_transition_sha256": (
                current.transitions[-1].transition_sha256
                if current.transitions
                else None
            ),
            "action": action,
            "active_realized_cost": active_realized_cost,
        }
    )
    updated = _freeze_state(
        session=checkpointed_session,
        graph=current.graph,
        synthesis=current.synthesis,
        candidates=current.candidates,
        predecessor=current,
        transition=transition,
    )
    payload = {
        "result_version": "sequential-active-cost-checkpoint-v1",
        "previous_state_sha256": current.state_sha256,
        "state": updated,
        "action_packet_sha256": action.packet_sha256,
        "active_realized_cost": active_realized_cost,
    }
    return SequentialActiveCostCheckpointResult.model_validate(
        {**payload, "result_sha256": hash_canonical(payload)}
    )


def freeze_selected_adjudication(
    state: SequentialVerificationState,
    *,
    expected: SequentialStateExpectation,
    provenance: str,
    adjudicator_count: int,
    protocol_sha256: str,
    payload_sha256: str,
    completed_at: datetime,
    realized_cost: float,
) -> AdjudicationArtifact:
    """Freeze externally measured adjudication cost against the selected action."""

    current = _require_expected_state(state, expected)
    action = current.session.active_action
    if action is None:
        raise SequentialVerificationContractError("adjudication_requires_selected_action")
    _finite_cost(realized_cost, "adjudication_realized_cost", positive=True)
    if (
        current.session.historical_realized_cost + realized_cost
        > current.session.budget + _COST_TOLERANCE
    ):
        raise SequentialVerificationContractError(
            "adjudication_realized_cost_exceeds_budget"
        )
    try:
        return freeze_adjudication_artifact(
            action,
            provenance=provenance,
            adjudicator_count=adjudicator_count,
            protocol_sha256=protocol_sha256,
            payload_sha256=payload_sha256,
            completed_at=completed_at,
            realized_cost=realized_cost,
        )
    except (AuditSessionContractError, ValueError) as exc:
        raise SequentialVerificationContractError(str(exc)) from exc


class SynthesisRerunCallback(Protocol):
    """Rerun the actual frozen statistical synthesis on a corrected graph."""

    def __call__(self, graph: EvidenceGraph) -> Mapping[str, Any]: ...


class CounterfactualCandidateRerunCallback(Protocol):
    """Rebuild and reprioritize actual graph counterfactuals after synthesis."""

    def __call__(
        self,
        graph: EvidenceGraph,
        synthesis: Mapping[str, Any],
        session: VerificationSession,
    ) -> Sequence[CurrentAuditCandidate]: ...


class SequentialResolutionResult(ContractModel):
    result_version: Literal["sequential-resolution-result-v1"] = (
        "sequential-resolution-result-v1"
    )
    previous_state_sha256: str
    state: SequentialVerificationState
    adjudication: AdjudicationArtifact
    correction_provenance: VerifierCorrectionProvenance
    correction: EvidenceGraphCorrection
    receipt: AuditResolutionReceiptV2
    result_sha256: str

    @field_validator("previous_state_sha256", "result_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_result(self) -> SequentialResolutionResult:
        if not self.state.transitions:
            raise ValueError("resolution_transition_missing")
        transition = self.state.transitions[-1]
        if (
            transition.transition_kind != "correction"
            or transition.previous_state_sha256 != self.previous_state_sha256
            or transition.adjudication != self.adjudication
            or transition.correction_provenance != self.correction_provenance
            or transition.correction != self.correction
            or transition.receipt != self.receipt
        ):
            raise ValueError("resolution_transition_mismatch")
        if self.correction.correction_artifact_sha256 != (
            self.correction_provenance.provenance_sha256
        ):
            raise ValueError("resolution_correction_provenance_mismatch")
        if self.receipt.correction_sha256 != self.correction.correction_sha256:
            raise ValueError("resolution_receipt_correction_mismatch")
        if self.receipt.adjudication_artifact_sha256 != self.adjudication.artifact_sha256:
            raise ValueError("resolution_receipt_adjudication_mismatch")
        if not self.state.session.steps or self.state.session.steps[-1].receipt != self.receipt:
            raise ValueError("resolution_receipt_not_in_updated_session")
        payload = self.model_dump(mode="json", exclude={"result_sha256"})
        if hash_canonical(payload) != self.result_sha256:
            raise ValueError("sequential_resolution_result_hash_mismatch")
        return self


def _estimate_hashes(graph: EvidenceGraph) -> dict[str, str]:
    return {
        estimate.estimate_id: hash_canonical(estimate)
        for estimate in graph.outcome_estimates
    }


def _validate_selected_estimate_correction(
    *,
    before: EvidenceGraph,
    after: EvidenceGraph,
    selected_item_id: str,
    disposition: CorrectionDisposition,
) -> tuple[str, str]:
    before_estimates = _estimate_hashes(before)
    after_estimates = _estimate_hashes(after)
    if selected_item_id not in before_estimates:
        raise SequentialVerificationContractError("selected_item_not_estimate")
    before_ids = list(before_estimates)
    after_ids = list(after_estimates)
    removed_selected_ids = [item_id for item_id in before_ids if item_id != selected_item_id]
    permitted_orders = [before_ids]
    if disposition is CorrectionDisposition.CORRECTED:
        permitted_orders.append(removed_selected_ids)
    if after_ids not in permitted_orders:
        raise SequentialVerificationContractError("correction_estimate_identity_or_order_changed")
    before_payload = before.model_dump(mode="json")
    after_payload = after.model_dump(mode="json")
    before_payload.pop("outcome_estimates")
    after_payload.pop("outcome_estimates")
    if before_payload != after_payload:
        raise SequentialVerificationContractError(
            "correction_changed_unselected_graph_component"
        )
    changed = sorted(
        item_id
        for item_id in before_estimates
        if item_id not in after_estimates
        or before_estimates[item_id] != after_estimates[item_id]
    )
    expected = [] if disposition is CorrectionDisposition.NO_CHANGE else [selected_item_id]
    if changed != expected:
        raise SequentialVerificationContractError(
            f"correction_changed_unselected_or_wrong_estimate:{changed}"
        )
    after_sha256 = after_estimates.get(
        selected_item_id,
        hash_canonical(
            {
                "disposition": "removed_selected_estimate",
                "estimate_id": selected_item_id,
            }
        ),
    )
    return before_estimates[selected_item_id], after_sha256


def _rerun_scientific_state(
    *,
    graph: EvidenceGraph,
    session: VerificationSession,
    rerun_synthesis: SynthesisRerunCallback,
    rerun_candidates: CounterfactualCandidateRerunCallback,
) -> tuple[dict[str, JsonValue], list[CurrentAuditCandidate]]:
    graph_sha256 = hash_canonical(graph)
    synthesis_graph = _normalize_graph(graph)
    synthesis = _normalize_synthesis(rerun_synthesis(synthesis_graph))
    if hash_canonical(synthesis_graph) != graph_sha256:
        raise SequentialVerificationContractError("synthesis_callback_mutated_graph")
    candidate_graph = _normalize_graph(graph)
    candidate_synthesis = deepcopy(synthesis)
    candidates = _normalize_candidates(
        rerun_candidates(candidate_graph, candidate_synthesis, session)
    )
    if hash_canonical(candidate_graph) != graph_sha256:
        raise SequentialVerificationContractError("candidate_callback_mutated_graph")
    if hash_canonical(candidate_synthesis) != hash_canonical(synthesis):
        raise SequentialVerificationContractError("candidate_callback_mutated_synthesis")
    return synthesis, candidates


def resolve_selected_audit_candidate(
    state: SequentialVerificationState,
    *,
    expected: SequentialStateExpectation,
    adjudication: AdjudicationArtifact,
    disposition: CorrectionDisposition,
    corrected_graph: EvidenceGraph | None,
    correction_provenance: str,
    correction_protocol_sha256: str,
    external_correction_payload_sha256: str,
    synthesis_runner_sha256: str,
    candidate_runner_sha256: str,
    rerun_synthesis: SynthesisRerunCallback,
    rerun_candidates: CounterfactualCandidateRerunCallback,
) -> SequentialResolutionResult:
    """Apply only the selected correction, rerun science, and charge realized cost."""

    current = _require_expected_state(state, expected)
    if not isinstance(disposition, CorrectionDisposition):
        raise SequentialVerificationContractError("correction_disposition_enum_required")
    action = current.session.active_action
    if action is None:
        raise SequentialVerificationContractError("resolution_requires_selected_action")
    try:
        artifact = AdjudicationArtifact.model_validate(adjudication.model_dump(mode="json"))
    except (AttributeError, ValueError) as exc:
        raise SequentialVerificationContractError("adjudication_artifact_invalid") from exc
    if (
        artifact.session_id,
        artifact.step,
        artifact.item_id,
        artifact.action_packet_sha256,
    ) != (
        current.session.session_id,
        action.step,
        action.item_id,
        action.packet_sha256,
    ):
        raise SequentialVerificationContractError("unselected_adjudication_rejected")
    if artifact.cost_unit != current.session.cost_unit:
        raise SequentialVerificationContractError("adjudication_cost_unit_mismatch")
    if artifact.completed_at < action.selected_at:
        raise SequentialVerificationContractError("adjudication_completed_before_selection")
    if artifact.realized_cost + _COST_TOLERANCE < current.session.active_realized_cost:
        raise SequentialVerificationContractError(
            "adjudication_cost_below_active_checkpoint"
        )
    if (
        current.session.historical_realized_cost + artifact.realized_cost
        > current.session.budget + _COST_TOLERANCE
    ):
        raise SequentialVerificationContractError(
            "adjudication_realized_cost_exceeds_budget"
        )
    if disposition is CorrectionDisposition.NO_CHANGE:
        if corrected_graph is not None:
            raise SequentialVerificationContractError(
                "explicit_no_change_forbids_corrected_graph"
            )
        post_graph = _normalize_graph(current.graph)
    else:
        if corrected_graph is None:
            raise SequentialVerificationContractError(
                "corrected_disposition_requires_corrected_graph"
            )
        post_graph = _normalize_graph(corrected_graph)
    selected_before, selected_after = _validate_selected_estimate_correction(
        before=current.graph,
        after=post_graph,
        selected_item_id=action.item_id,
        disposition=disposition,
    )
    post_synthesis, post_candidates = _rerun_scientific_state(
        graph=post_graph,
        session=current.session,
        rerun_synthesis=rerun_synthesis,
        rerun_candidates=rerun_candidates,
    )
    post_graph_sha256 = hash_canonical(post_graph)
    post_synthesis_sha256 = hash_canonical(post_synthesis)
    post_candidate_input_sha256 = hash_canonical(post_candidates)
    if disposition is CorrectionDisposition.NO_CHANGE and (
        post_graph_sha256,
        post_synthesis_sha256,
        post_candidate_input_sha256,
    ) != (
        current.graph_sha256,
        current.synthesis_sha256,
        current.candidate_input_sha256,
    ):
        raise SequentialVerificationContractError("no_change_rerun_state_changed")
    selected_candidate = next(
        candidate for candidate in current.candidates if candidate.item_id == action.item_id
    )
    provenance_payload = {
        "provenance_version": "verifier-correction-provenance-v1",
        "session_id": current.session.session_id,
        "step": action.step,
        "item_id": action.item_id,
        "action_packet_sha256": action.packet_sha256,
        "selected_candidate_sha256": selected_candidate.candidate_sha256,
        "adjudication_artifact_sha256": artifact.artifact_sha256,
        "adjudication_payload_sha256": artifact.payload_sha256,
        "disposition": disposition,
        "provenance": correction_provenance,
        "correction_protocol_sha256": correction_protocol_sha256,
        "external_correction_payload_sha256": external_correction_payload_sha256,
        "selected_estimate_before_sha256": selected_before,
        "selected_estimate_after_sha256": selected_after,
        "pre_graph_sha256": current.graph_sha256,
        "post_graph_sha256": post_graph_sha256,
        "synthesis_runner_sha256": synthesis_runner_sha256,
        "candidate_runner_sha256": candidate_runner_sha256,
        "post_synthesis_sha256": post_synthesis_sha256,
        "post_candidate_input_sha256": post_candidate_input_sha256,
    }
    frozen_provenance = VerifierCorrectionProvenance.model_validate(
        {
            **provenance_payload,
            "provenance_sha256": hash_canonical(provenance_payload),
        }
    )
    correction = freeze_evidence_graph_correction(
        action,
        artifact,
        disposition=disposition,
        post_graph_sha256=post_graph_sha256,
        post_synthesis_sha256=post_synthesis_sha256,
        post_candidate_input_sha256=post_candidate_input_sha256,
        correction_artifact_sha256=frozen_provenance.provenance_sha256,
    )
    try:
        resolved_session, receipt = resolve_audit_action(
            current.session,
            expected_state_sha256=current.session.session_sha256,
            adjudication=artifact,
            correction=correction,
        )
    except AuditSessionContractError as exc:
        raise SequentialVerificationContractError(str(exc)) from exc
    transition = _freeze_transition(
        {
            "transition_kind": "correction",
            "previous_state_sha256": current.state_sha256,
            "previous_transition_sha256": (
                current.transitions[-1].transition_sha256
                if current.transitions
                else None
            ),
            "candidate": selected_candidate,
            "action": action,
            "adjudication": artifact,
            "correction_provenance": frozen_provenance,
            "correction": correction,
            "receipt": receipt,
            "post_graph": post_graph,
            "post_graph_sha256": post_graph_sha256,
            "post_synthesis": post_synthesis,
            "post_synthesis_sha256": post_synthesis_sha256,
            "post_candidates": post_candidates,
            "post_candidate_input_sha256": post_candidate_input_sha256,
        }
    )
    updated = _freeze_state(
        session=resolved_session,
        graph=post_graph,
        synthesis=post_synthesis,
        candidates=post_candidates,
        predecessor=current,
        transition=transition,
    )
    result_payload = {
        "result_version": "sequential-resolution-result-v1",
        "previous_state_sha256": current.state_sha256,
        "state": updated,
        "adjudication": artifact,
        "correction_provenance": frozen_provenance,
        "correction": correction,
        "receipt": receipt,
    }
    return SequentialResolutionResult.model_validate(
        {**result_payload, "result_sha256": hash_canonical(result_payload)}
    )


__all__ = [
    "CounterfactualCandidateRerunCallback",
    "CurrentAuditCandidate",
    "SequentialActiveCostCheckpointResult",
    "SequentialResolutionResult",
    "SequentialSelectionResult",
    "SequentialStateExpectation",
    "SequentialStateTransition",
    "SequentialVerificationContractError",
    "SequentialVerificationState",
    "StaleSequentialVerificationStateError",
    "SynthesisRerunCallback",
    "VerifierCorrectionProvenance",
    "adaptive_preselection_history_from_state",
    "checkpoint_selected_audit_cost",
    "create_sequential_verification_state",
    "current_candidates_from_audit_candidates",
    "freeze_current_audit_candidate",
    "freeze_selected_adjudication",
    "freeze_state_expectation",
    "resolve_selected_audit_candidate",
    "resume_sequential_verification_state",
    "select_next_audit_candidate",
    "selection_predecessor_states_from_state",
]
