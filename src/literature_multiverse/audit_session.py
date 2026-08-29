"""Hash-bound, resumable sessions for sequential human verification.

This module is deliberately an orchestration substrate, not a scientific verifier.
It knows only opaque graph, synthesis, candidate-input, policy, and pipeline hashes.
Callers own candidate ranking, human adjudication, graph correction, synthesis, and
release assessment.

The contracts make four distinctions that are easy to blur in an in-memory loop:

* selecting an action does not spend its estimated cost;
* partial work is active realized cost, completed receipts are historical cost, and
  current realized cost is their sum;
* an item can be resolved only through the action that selected it; and
* every mutation names the exact session hash it expects, so stale workers fail closed.

Session snapshots are self-hashed and form an append-only hash chain.  The transition
functions are pure and deterministic for identical inputs, which permits a serialized
snapshot to be validated and resumed without replaying external callbacks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Protocol

from pydantic import Field, ValidationInfo, field_validator, model_validator

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel

_COST_ABS_TOLERANCE = 1e-9
_SELF_HASH_CONTEXT_KEY = "literature_multiverse_audit_session_internal"
_SELF_HASH_CONTEXT_SENTINEL = object()


class AuditSessionContractError(ValueError):
    """A requested session transition would weaken the audit contract."""


class StaleAuditStateError(AuditSessionContractError):
    """A worker attempted a transition from a superseded session snapshot."""


class VerificationSessionStatus(StrEnum):
    """Lifecycle states for an immutable verification-session snapshot."""

    ACTIVE = "active"
    FINALIZED = "finalized"


class CorrectionDisposition(StrEnum):
    """Whether adjudication changed the evidence state."""

    NO_CHANGE = "no_change"
    CORRECTED = "corrected"


def _validate_sha256(value: str | None, field_name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if value is None or not SHA256_RE.fullmatch(value):
        raise ValueError(f"invalid_sha256:{field_name}")
    return value


def _validate_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"timezone_required:{field_name}")
    return value


def _validate_cost(value: float, field_name: str, *, positive: bool = False) -> float:
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{field_name}_must_be_finite_{qualifier}")
    return value


def _costs_equal(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=1e-12,
        abs_tol=_COST_ABS_TOLERANCE,
    )


def _skip_self_hash(info: ValidationInfo) -> bool:
    return (info.context or {}).get(_SELF_HASH_CONTEXT_KEY) is (
        _SELF_HASH_CONTEXT_SENTINEL
    )


class AuditActionPacket(ContractModel):
    """One scheduler decision bound to the exact state it observed.

    ``estimated_cost`` is used only to decide whether the item plausibly fits.  It is
    never added to session spending.  Only a completed adjudication's
    ``realized_cost`` can enter the historical ledger.
    """

    packet_version: Literal["audit-action-v2"] = "audit-action-v2"
    session_id: Annotated[str, Field(min_length=1)]
    step: Annotated[int, Field(ge=1)]
    item_id: Annotated[str, Field(min_length=1)]
    pipeline_sha256: str
    policy_sha256: str
    candidate_input_sha256: str
    graph_sha256: str
    synthesis_sha256: str
    selection_state_sha256: str
    scheduler_artifact_sha256: str
    estimated_cost: float
    cost_unit: Annotated[str, Field(min_length=1)]
    selection_rank: Annotated[int, Field(ge=1)] | None = None
    selection_score: float | None = None
    selected_at: datetime
    packet_sha256: str

    @field_validator(
        "pipeline_sha256",
        "policy_sha256",
        "candidate_input_sha256",
        "graph_sha256",
        "synthesis_sha256",
        "selection_state_sha256",
        "scheduler_artifact_sha256",
        "packet_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        validated = _validate_sha256(value, info.field_name)
        assert validated is not None
        return validated

    @field_validator("estimated_cost")
    @classmethod
    def validate_estimated_cost(cls, value: float) -> float:
        return _validate_cost(value, "audit_action_estimated_cost", positive=True)

    @field_validator("selection_score")
    @classmethod
    def validate_selection_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("audit_action_selection_score_nonfinite")
        return value

    @field_validator("selected_at")
    @classmethod
    def validate_selected_at(cls, value: datetime) -> datetime:
        return _validate_aware(value, "selected_at")

    @model_validator(mode="after")
    def validate_packet(self, info: ValidationInfo) -> AuditActionPacket:
        if (self.selection_rank is None) != (self.selection_score is None):
            raise ValueError("audit_action_rank_and_score_require_each_other")
        payload = self.model_dump(mode="json", exclude={"packet_sha256"})
        if not _skip_self_hash(info) and hash_canonical(payload) != self.packet_sha256:
            raise ValueError("audit_action_packet_hash_mismatch")
        return self


class AdjudicationArtifact(ContractModel):
    """Externally produced adjudication metadata and its realized cost.

    The payload remains external and is represented by ``payload_sha256``.  This
    contract intentionally does not assert that the adjudication is correct or that
    an adjudicator is competent.
    """

    artifact_version: Literal["adjudication-artifact-v2"] = "adjudication-artifact-v2"
    session_id: Annotated[str, Field(min_length=1)]
    step: Annotated[int, Field(ge=1)]
    item_id: Annotated[str, Field(min_length=1)]
    action_packet_sha256: str
    provenance: Annotated[str, Field(min_length=1)]
    adjudicator_count: Annotated[int, Field(ge=1)]
    protocol_sha256: str
    payload_sha256: str
    completed_at: datetime
    realized_cost: float
    cost_unit: Annotated[str, Field(min_length=1)]
    artifact_sha256: str

    @field_validator(
        "action_packet_sha256",
        "protocol_sha256",
        "payload_sha256",
        "artifact_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        validated = _validate_sha256(value, info.field_name)
        assert validated is not None
        return validated

    @field_validator("completed_at")
    @classmethod
    def validate_completed_at(cls, value: datetime) -> datetime:
        return _validate_aware(value, "completed_at")

    @field_validator("realized_cost")
    @classmethod
    def validate_realized_cost(cls, value: float) -> float:
        return _validate_cost(value, "adjudication_realized_cost", positive=True)

    @model_validator(mode="after")
    def validate_artifact(self, info: ValidationInfo) -> AdjudicationArtifact:
        payload = self.model_dump(mode="json", exclude={"artifact_sha256"})
        if not _skip_self_hash(info) and hash_canonical(payload) != self.artifact_sha256:
            raise ValueError("adjudication_artifact_hash_mismatch")
        return self


class EvidenceGraphCorrection(ContractModel):
    """Opaque before/after state produced by applying one adjudication."""

    correction_version: Literal["evidence-graph-correction-v2"] = (
        "evidence-graph-correction-v2"
    )
    session_id: Annotated[str, Field(min_length=1)]
    step: Annotated[int, Field(ge=1)]
    item_id: Annotated[str, Field(min_length=1)]
    action_packet_sha256: str
    adjudication_artifact_sha256: str
    disposition: CorrectionDisposition
    pre_graph_sha256: str
    pre_synthesis_sha256: str
    pre_candidate_input_sha256: str
    post_graph_sha256: str
    post_synthesis_sha256: str
    post_candidate_input_sha256: str
    correction_artifact_sha256: str
    correction_sha256: str

    @field_validator(
        "action_packet_sha256",
        "adjudication_artifact_sha256",
        "pre_graph_sha256",
        "pre_synthesis_sha256",
        "pre_candidate_input_sha256",
        "post_graph_sha256",
        "post_synthesis_sha256",
        "post_candidate_input_sha256",
        "correction_artifact_sha256",
        "correction_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        validated = _validate_sha256(value, info.field_name)
        assert validated is not None
        return validated

    @model_validator(mode="after")
    def validate_correction(self, info: ValidationInfo) -> EvidenceGraphCorrection:
        before = (
            self.pre_graph_sha256,
            self.pre_synthesis_sha256,
            self.pre_candidate_input_sha256,
        )
        after = (
            self.post_graph_sha256,
            self.post_synthesis_sha256,
            self.post_candidate_input_sha256,
        )
        if self.disposition is CorrectionDisposition.NO_CHANGE and before != after:
            raise ValueError("no_change_correction_must_preserve_state")
        if self.disposition is CorrectionDisposition.CORRECTED and before == after:
            raise ValueError("corrected_disposition_requires_state_change")
        payload = self.model_dump(mode="json", exclude={"correction_sha256"})
        if not _skip_self_hash(info) and hash_canonical(payload) != self.correction_sha256:
            raise ValueError("evidence_graph_correction_hash_mismatch")
        return self


class AuditResolutionReceiptV2(ContractModel):
    """Completed resolution bound to selection, state, correction, and actual cost."""

    receipt_version: Literal["audit-resolution-v2"] = "audit-resolution-v2"
    session_id: Annotated[str, Field(min_length=1)]
    step: Annotated[int, Field(ge=1)]
    item_id: Annotated[str, Field(min_length=1)]
    pipeline_sha256: str
    policy_sha256: str
    action_packet_sha256: str
    selection_state_sha256: str
    state_before_resolution_sha256: str
    candidate_input_sha256: str
    pre_graph_sha256: str
    pre_synthesis_sha256: str
    adjudication_protocol_sha256: str
    adjudication_artifact_sha256: str
    correction_sha256: str
    post_graph_sha256: str
    post_synthesis_sha256: str
    post_candidate_input_sha256: str
    realized_cost: float
    active_realized_cost_before_resolution: float
    historical_realized_cost_before_resolution: float
    cumulative_realized_cost_after_resolution: float
    budget: float
    remaining_budget_after_resolution: float
    cost_unit: Annotated[str, Field(min_length=1)]
    receipt_sha256: str

    @field_validator(
        "pipeline_sha256",
        "policy_sha256",
        "action_packet_sha256",
        "selection_state_sha256",
        "state_before_resolution_sha256",
        "candidate_input_sha256",
        "pre_graph_sha256",
        "pre_synthesis_sha256",
        "adjudication_protocol_sha256",
        "adjudication_artifact_sha256",
        "correction_sha256",
        "post_graph_sha256",
        "post_synthesis_sha256",
        "post_candidate_input_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        validated = _validate_sha256(value, info.field_name)
        assert validated is not None
        return validated

    @field_validator(
        "realized_cost",
        "active_realized_cost_before_resolution",
        "historical_realized_cost_before_resolution",
        "cumulative_realized_cost_after_resolution",
        "budget",
        "remaining_budget_after_resolution",
    )
    @classmethod
    def validate_costs(cls, value: float, info: Any) -> float:
        return _validate_cost(value, f"audit_receipt_{info.field_name}")

    @model_validator(mode="after")
    def validate_receipt(self, info: ValidationInfo) -> AuditResolutionReceiptV2:
        if self.realized_cost + _COST_ABS_TOLERANCE < (
            self.active_realized_cost_before_resolution
        ):
            raise ValueError("receipt_realized_cost_below_active_checkpoint")
        expected_cumulative = math.fsum(
            (self.historical_realized_cost_before_resolution, self.realized_cost)
        )
        if not _costs_equal(
            self.cumulative_realized_cost_after_resolution,
            expected_cumulative,
        ):
            raise ValueError("receipt_cumulative_realized_cost_mismatch")
        expected_remaining = self.budget - self.cumulative_realized_cost_after_resolution
        if expected_remaining < -_COST_ABS_TOLERANCE:
            raise ValueError("receipt_realized_cost_exceeds_budget")
        if not _costs_equal(
            self.remaining_budget_after_resolution,
            max(0.0, expected_remaining),
        ):
            raise ValueError("receipt_remaining_budget_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if not _skip_self_hash(info) and hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("audit_resolution_receipt_v2_hash_mismatch")
        return self


class SequentialAuditStepV2(ContractModel):
    """One complete select/adjudicate/correct/receipt transaction."""

    step_version: Literal["sequential-audit-step-v2"] = "sequential-audit-step-v2"
    session_id: Annotated[str, Field(min_length=1)]
    step: Annotated[int, Field(ge=1)]
    item_id: Annotated[str, Field(min_length=1)]
    action: AuditActionPacket
    adjudication: AdjudicationArtifact
    correction: EvidenceGraphCorrection
    receipt: AuditResolutionReceiptV2
    previous_step_sha256: str | None = None
    step_sha256: str

    @field_validator("previous_step_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return _validate_sha256(value, "previous_step_sha256", optional=True)

    @field_validator("step_sha256")
    @classmethod
    def validate_step_hash(cls, value: str) -> str:
        validated = _validate_sha256(value, "step_sha256")
        assert validated is not None
        return validated

    @model_validator(mode="after")
    def validate_step_links(self, info: ValidationInfo) -> SequentialAuditStepV2:
        identities = {
            (self.session_id, self.step, self.item_id),
            (self.action.session_id, self.action.step, self.action.item_id),
            (
                self.adjudication.session_id,
                self.adjudication.step,
                self.adjudication.item_id,
            ),
            (
                self.correction.session_id,
                self.correction.step,
                self.correction.item_id,
            ),
            (self.receipt.session_id, self.receipt.step, self.receipt.item_id),
        }
        if len(identities) != 1:
            raise ValueError("sequential_audit_step_identity_mismatch")
        if self.adjudication.action_packet_sha256 != self.action.packet_sha256:
            raise ValueError("adjudication_not_bound_to_selected_action")
        if self.correction.action_packet_sha256 != self.action.packet_sha256:
            raise ValueError("correction_not_bound_to_selected_action")
        if (
            self.correction.adjudication_artifact_sha256
            != self.adjudication.artifact_sha256
        ):
            raise ValueError("correction_not_bound_to_adjudication")
        if self.receipt.action_packet_sha256 != self.action.packet_sha256:
            raise ValueError("receipt_not_bound_to_selected_action")
        if self.receipt.adjudication_artifact_sha256 != self.adjudication.artifact_sha256:
            raise ValueError("receipt_not_bound_to_adjudication")
        if self.receipt.correction_sha256 != self.correction.correction_sha256:
            raise ValueError("receipt_not_bound_to_correction")
        if self.receipt.adjudication_protocol_sha256 != self.adjudication.protocol_sha256:
            raise ValueError("receipt_protocol_hash_mismatch")
        if not _costs_equal(self.receipt.realized_cost, self.adjudication.realized_cost):
            raise ValueError("receipt_adjudication_realized_cost_mismatch")
        if self.receipt.cost_unit != self.adjudication.cost_unit:
            raise ValueError("receipt_adjudication_cost_unit_mismatch")
        if self.adjudication.cost_unit != self.action.cost_unit:
            raise ValueError("adjudication_action_cost_unit_mismatch")
        if self.adjudication.completed_at < self.action.selected_at:
            raise ValueError("adjudication_completed_before_selection")
        if self.receipt.pipeline_sha256 != self.action.pipeline_sha256:
            raise ValueError("receipt_action_pipeline_hash_mismatch")
        if self.receipt.policy_sha256 != self.action.policy_sha256:
            raise ValueError("receipt_action_policy_hash_mismatch")
        if self.receipt.selection_state_sha256 != self.action.selection_state_sha256:
            raise ValueError("receipt_action_selection_state_hash_mismatch")
        if self.receipt.candidate_input_sha256 != self.action.candidate_input_sha256:
            raise ValueError("receipt_action_candidate_input_hash_mismatch")
        if (
            self.receipt.pre_graph_sha256,
            self.receipt.pre_synthesis_sha256,
        ) != (self.action.graph_sha256, self.action.synthesis_sha256):
            raise ValueError("receipt_action_pre_state_hash_mismatch")
        if (
            self.correction.pre_graph_sha256,
            self.correction.pre_synthesis_sha256,
            self.correction.pre_candidate_input_sha256,
        ) != (
            self.action.graph_sha256,
            self.action.synthesis_sha256,
            self.action.candidate_input_sha256,
        ):
            raise ValueError("correction_action_pre_state_hash_mismatch")
        if (
            self.receipt.post_graph_sha256,
            self.receipt.post_synthesis_sha256,
            self.receipt.post_candidate_input_sha256,
        ) != (
            self.correction.post_graph_sha256,
            self.correction.post_synthesis_sha256,
            self.correction.post_candidate_input_sha256,
        ):
            raise ValueError("receipt_correction_post_state_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"step_sha256"})
        if not _skip_self_hash(info) and hash_canonical(payload) != self.step_sha256:
            raise ValueError("sequential_audit_step_hash_mismatch")
        return self


class VerificationSession(ContractModel):
    """Immutable snapshot of a resumable sequential audit.

    ``historical_realized_cost`` is the sum of completed receipt costs.
    ``active_realized_cost`` is checkpointed work on the selected unresolved action.
    ``current_realized_cost`` is the sum of those two values.  Estimated action cost
    never appears in any of these ledgers.
    """

    session_version: Literal["verification-session-v2"] = "verification-session-v2"
    session_id: Annotated[str, Field(min_length=1)]
    created_at: datetime
    status: VerificationSessionStatus
    pipeline_sha256: str
    policy_sha256: str
    budget: float
    cost_unit: Annotated[str, Field(min_length=1)]
    initial_graph_sha256: str
    initial_synthesis_sha256: str
    initial_candidate_input_sha256: str
    current_graph_sha256: str
    current_synthesis_sha256: str
    current_candidate_input_sha256: str
    selected_item_ids: tuple[str, ...]
    resolved_item_ids: tuple[str, ...]
    steps: tuple[SequentialAuditStepV2, ...]
    active_action: AuditActionPacket | None
    historical_realized_cost: float
    active_realized_cost: float
    current_realized_cost: float
    remaining_budget: float
    transition_index: Annotated[int, Field(ge=0)]
    state_history_sha256s: tuple[str, ...]
    previous_session_sha256: str | None
    finalized_from_state_sha256: str | None
    final_assessment_state_sha256: str | None
    final_assessment_sha256: str | None
    final_reason: str | None
    finalized_at: datetime | None
    session_sha256: str

    @field_validator(
        "pipeline_sha256",
        "policy_sha256",
        "initial_graph_sha256",
        "initial_synthesis_sha256",
        "initial_candidate_input_sha256",
        "current_graph_sha256",
        "current_synthesis_sha256",
        "current_candidate_input_sha256",
        "session_sha256",
    )
    @classmethod
    def validate_required_hashes(cls, value: str, info: Any) -> str:
        validated = _validate_sha256(value, info.field_name)
        assert validated is not None
        return validated

    @field_validator(
        "previous_session_sha256",
        "finalized_from_state_sha256",
        "final_assessment_state_sha256",
        "final_assessment_sha256",
    )
    @classmethod
    def validate_optional_hashes(cls, value: str | None, info: Any) -> str | None:
        return _validate_sha256(value, info.field_name, optional=True)

    @field_validator("state_history_sha256s")
    @classmethod
    def validate_history_hashes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for state_hash in value:
            _validate_sha256(state_hash, "state_history_sha256s")
        if len(value) != len(set(value)):
            raise ValueError("session_state_history_hashes_not_unique")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _validate_aware(value, "created_at")

    @field_validator("finalized_at")
    @classmethod
    def validate_finalized_at(cls, value: datetime | None) -> datetime | None:
        return _validate_aware(value, "finalized_at") if value is not None else None

    @field_validator(
        "budget",
        "historical_realized_cost",
        "active_realized_cost",
        "current_realized_cost",
        "remaining_budget",
    )
    @classmethod
    def validate_costs(cls, value: float, info: Any) -> float:
        return _validate_cost(value, f"verification_session_{info.field_name}")

    @model_validator(mode="after")
    def validate_session(self, info: ValidationInfo) -> VerificationSession:
        if self.transition_index != len(self.state_history_sha256s):
            raise ValueError("session_transition_history_length_mismatch")
        expected_previous = (
            self.state_history_sha256s[-1] if self.state_history_sha256s else None
        )
        if self.previous_session_sha256 != expected_previous:
            raise ValueError("session_previous_state_hash_mismatch")

        step_numbers = tuple(step.step for step in self.steps)
        if step_numbers != tuple(range(1, len(self.steps) + 1)):
            raise ValueError("session_steps_not_contiguous")
        resolved_from_steps = tuple(step.item_id for step in self.steps)
        if len(resolved_from_steps) != len(set(resolved_from_steps)):
            raise ValueError("session_item_resolved_more_than_once")
        if self.resolved_item_ids != resolved_from_steps:
            raise ValueError("session_resolved_items_do_not_match_receipts")

        expected_selected = resolved_from_steps
        if self.active_action is not None:
            expected_selected = (*expected_selected, self.active_action.item_id)
        if self.selected_item_ids != expected_selected:
            raise ValueError("session_resolved_item_was_not_selected")
        if len(self.selected_item_ids) != len(set(self.selected_item_ids)):
            raise ValueError("session_item_selected_more_than_once")

        previous_step_hash: str | None = None
        previous_completed_at = self.created_at
        historical_cost_before_step = 0.0
        previous_resolution_history_index = -1
        for step in self.steps:
            if step.session_id != self.session_id:
                raise ValueError("session_step_session_id_mismatch")
            if step.action.pipeline_sha256 != self.pipeline_sha256:
                raise ValueError("session_step_pipeline_hash_mismatch")
            if step.action.policy_sha256 != self.policy_sha256:
                raise ValueError("session_step_policy_hash_mismatch")
            if step.action.cost_unit != self.cost_unit:
                raise ValueError("session_step_cost_unit_mismatch")
            if step.previous_step_sha256 != previous_step_hash:
                raise ValueError("session_step_hash_chain_mismatch")
            if step.action.selection_state_sha256 not in self.state_history_sha256s:
                raise ValueError("session_selection_state_missing_from_history")
            if step.receipt.state_before_resolution_sha256 not in (
                self.state_history_sha256s
            ):
                raise ValueError("session_resolution_state_missing_from_history")
            selection_history_index = self.state_history_sha256s.index(
                step.action.selection_state_sha256
            )
            resolution_history_index = self.state_history_sha256s.index(
                step.receipt.state_before_resolution_sha256
            )
            if selection_history_index <= previous_resolution_history_index:
                raise ValueError("session_selection_state_history_not_monotone")
            if resolution_history_index <= selection_history_index:
                raise ValueError("session_resolution_precedes_selection")
            if step.action.selected_at < previous_completed_at:
                raise ValueError("session_sequential_action_times_not_monotone")
            if not _costs_equal(
                step.receipt.historical_realized_cost_before_resolution,
                historical_cost_before_step,
            ):
                raise ValueError("session_receipt_historical_cost_chain_mismatch")
            if not _costs_equal(step.receipt.budget, self.budget):
                raise ValueError("session_receipt_budget_mismatch")
            if step.receipt.cost_unit != self.cost_unit:
                raise ValueError("session_receipt_cost_unit_mismatch")
            previous_step_hash = step.step_sha256
            previous_completed_at = step.adjudication.completed_at
            historical_cost_before_step = (
                step.receipt.cumulative_realized_cost_after_resolution
            )
            previous_resolution_history_index = resolution_history_index

        expected_graph = self.initial_graph_sha256
        expected_synthesis = self.initial_synthesis_sha256
        expected_candidates = self.initial_candidate_input_sha256
        if self.steps:
            last_correction = self.steps[-1].correction
            expected_graph = last_correction.post_graph_sha256
            expected_synthesis = last_correction.post_synthesis_sha256
            expected_candidates = last_correction.post_candidate_input_sha256
        if (
            self.current_graph_sha256,
            self.current_synthesis_sha256,
            self.current_candidate_input_sha256,
        ) != (expected_graph, expected_synthesis, expected_candidates):
            raise ValueError("session_current_evidence_state_mismatch")

        if self.active_action is not None:
            action = self.active_action
            if self.status is not VerificationSessionStatus.ACTIVE:
                raise ValueError("finalized_session_cannot_have_active_action")
            if action.session_id != self.session_id:
                raise ValueError("active_action_session_id_mismatch")
            if action.step != len(self.steps) + 1:
                raise ValueError("active_action_step_mismatch")
            if action.pipeline_sha256 != self.pipeline_sha256:
                raise ValueError("active_action_pipeline_hash_mismatch")
            if action.policy_sha256 != self.policy_sha256:
                raise ValueError("active_action_policy_hash_mismatch")
            if action.cost_unit != self.cost_unit:
                raise ValueError("active_action_cost_unit_mismatch")
            if action.selection_state_sha256 not in self.state_history_sha256s:
                raise ValueError("active_action_selection_state_missing_from_history")
            selection_history_index = self.state_history_sha256s.index(
                action.selection_state_sha256
            )
            if selection_history_index <= previous_resolution_history_index:
                raise ValueError("active_action_selection_state_history_not_monotone")
            if action.selected_at < previous_completed_at:
                raise ValueError("active_action_selected_before_prior_resolution")
            if (
                action.graph_sha256,
                action.synthesis_sha256,
                action.candidate_input_sha256,
            ) != (
                self.current_graph_sha256,
                self.current_synthesis_sha256,
                self.current_candidate_input_sha256,
            ):
                raise ValueError("active_action_is_stale_for_current_evidence_state")
        elif not _costs_equal(self.active_realized_cost, 0.0):
            raise ValueError("session_active_cost_without_active_action")

        receipt_cost = math.fsum(step.receipt.realized_cost for step in self.steps)
        if not _costs_equal(self.historical_realized_cost, receipt_cost):
            raise ValueError("session_historical_cost_does_not_match_receipts")
        expected_current_cost = math.fsum(
            (self.historical_realized_cost, self.active_realized_cost)
        )
        if not _costs_equal(self.current_realized_cost, expected_current_cost):
            raise ValueError("session_current_realized_cost_mismatch")
        expected_remaining = self.budget - self.current_realized_cost
        if expected_remaining < -_COST_ABS_TOLERANCE:
            raise ValueError("session_realized_cost_exceeds_budget")
        if not _costs_equal(self.remaining_budget, max(0.0, expected_remaining)):
            raise ValueError("session_remaining_budget_mismatch")
        if self.budget == 0 and (self.selected_item_ids or self.current_realized_cost != 0):
            raise ValueError("zero_budget_session_cannot_select_or_spend")

        final_fields = (
            self.finalized_from_state_sha256,
            self.final_assessment_state_sha256,
            self.final_assessment_sha256,
            self.final_reason,
            self.finalized_at,
        )
        if self.status is VerificationSessionStatus.ACTIVE:
            if any(value is not None for value in final_fields):
                raise ValueError("active_session_cannot_have_finalization_metadata")
        else:
            if any(value is None for value in final_fields):
                raise ValueError("finalized_session_requires_complete_metadata")
            if self.active_action is not None:
                raise ValueError("finalized_session_cannot_have_active_action")
            if self.finalized_from_state_sha256 != self.previous_session_sha256:
                raise ValueError("finalization_parent_state_hash_mismatch")
            if self.final_assessment_state_sha256 != self.finalized_from_state_sha256:
                raise ValueError("final_assessment_state_hash_mismatch")
            assert self.finalized_at is not None
            if self.finalized_at < self.created_at:
                raise ValueError("session_finalized_before_creation")
            if self.final_reason is None or not self.final_reason.strip():
                raise ValueError("session_final_reason_empty")

        payload = self.model_dump(mode="json", exclude={"session_sha256"})
        if not _skip_self_hash(info) and hash_canonical(payload) != self.session_sha256:
            raise ValueError("verification_session_hash_mismatch")
        return self


class EvidenceGraphCorrectionCallback(Protocol):
    """Apply adjudication and return opaque refreshed state hashes."""

    def __call__(
        self,
        session: VerificationSession,
        action: AuditActionPacket,
        adjudication: AdjudicationArtifact,
    ) -> EvidenceGraphCorrection: ...


class FinalAssessmentCallback(Protocol):
    """Run a downstream guard/release assessment and return its artifact hash."""

    def __call__(self, session: VerificationSession) -> str: ...


def _freeze_model[ModelT: ContractModel](
    model_type: type[ModelT],
    payload: Mapping[str, Any],
    *,
    hash_field: str,
) -> ModelT:
    draft = model_type.model_validate(
        {**payload, hash_field: "0" * 64},
        context={_SELF_HASH_CONTEXT_KEY: _SELF_HASH_CONTEXT_SENTINEL},
    )
    canonical_payload = draft.model_dump(mode="json", exclude={hash_field})
    return model_type.model_validate(
        {
            **canonical_payload,
            hash_field: hash_canonical(canonical_payload),
        }
    )


def _transition_session(
    session: VerificationSession,
    **updates: Any,
) -> VerificationSession:
    payload = session.model_dump(mode="json", exclude={"session_sha256"})
    payload.update(updates)
    payload["transition_index"] = session.transition_index + 1
    payload["state_history_sha256s"] = [
        *session.state_history_sha256s,
        session.session_sha256,
    ]
    payload["previous_session_sha256"] = session.session_sha256
    return _freeze_model(
        VerificationSession,
        payload,
        hash_field="session_sha256",
    )


def create_verification_session(
    *,
    session_id: str,
    created_at: datetime,
    pipeline_sha256: str,
    policy_sha256: str,
    budget: float,
    cost_unit: str,
    graph_sha256: str,
    synthesis_sha256: str,
    candidate_input_sha256: str,
) -> VerificationSession:
    """Create a deterministic initial snapshot; a zero budget is valid and inert."""

    payload = {
        "session_version": "verification-session-v2",
        "session_id": session_id,
        "created_at": created_at,
        "status": VerificationSessionStatus.ACTIVE,
        "pipeline_sha256": pipeline_sha256,
        "policy_sha256": policy_sha256,
        "budget": budget,
        "cost_unit": cost_unit,
        "initial_graph_sha256": graph_sha256,
        "initial_synthesis_sha256": synthesis_sha256,
        "initial_candidate_input_sha256": candidate_input_sha256,
        "current_graph_sha256": graph_sha256,
        "current_synthesis_sha256": synthesis_sha256,
        "current_candidate_input_sha256": candidate_input_sha256,
        "selected_item_ids": (),
        "resolved_item_ids": (),
        "steps": (),
        "active_action": None,
        "historical_realized_cost": 0.0,
        "active_realized_cost": 0.0,
        "current_realized_cost": 0.0,
        "remaining_budget": budget,
        "transition_index": 0,
        "state_history_sha256s": (),
        "previous_session_sha256": None,
        "finalized_from_state_sha256": None,
        "final_assessment_state_sha256": None,
        "final_assessment_sha256": None,
        "final_reason": None,
        "finalized_at": None,
    }
    return _freeze_model(
        VerificationSession,
        payload,
        hash_field="session_sha256",
    )


def resume_verification_session(
    snapshot: VerificationSession | Mapping[str, Any],
    *,
    expected_state_sha256: str | None = None,
) -> VerificationSession:
    """Revalidate every nested self-hash and the session chain before reuse."""

    raw = (
        snapshot.model_dump(mode="json")
        if isinstance(snapshot, VerificationSession)
        else snapshot
    )
    session = VerificationSession.model_validate(raw)
    if expected_state_sha256 is not None and session.session_sha256 != expected_state_sha256:
        raise StaleAuditStateError(
            "stale_audit_session_state:"
            f"expected={expected_state_sha256}:actual={session.session_sha256}"
        )
    return session


def select_audit_action(
    session: VerificationSession,
    *,
    expected_state_sha256: str,
    item_id: str,
    scheduler_artifact_sha256: str,
    estimated_cost: float,
    selected_at: datetime,
    selection_rank: int | None = None,
    selection_score: float | None = None,
) -> tuple[VerificationSession, AuditActionPacket]:
    """Select one unresolved item without charging its estimated cost."""

    current = resume_verification_session(
        session,
        expected_state_sha256=expected_state_sha256,
    )
    if current.status is VerificationSessionStatus.FINALIZED:
        raise AuditSessionContractError("finalized_session_is_terminal")
    if current.active_action is not None:
        raise AuditSessionContractError("session_already_has_active_action")
    if current.budget == 0:
        raise AuditSessionContractError("zero_budget_session_cannot_select")
    if item_id in current.selected_item_ids:
        raise AuditSessionContractError(f"audit_item_already_selected:{item_id}")
    _validate_cost(estimated_cost, "audit_action_estimated_cost", positive=True)
    if estimated_cost > current.remaining_budget + _COST_ABS_TOLERANCE:
        raise AuditSessionContractError("estimated_action_cost_exceeds_remaining_budget")
    if selected_at < current.created_at:
        raise AuditSessionContractError("audit_action_selected_before_session_creation")

    action_payload = {
        "packet_version": "audit-action-v2",
        "session_id": current.session_id,
        "step": len(current.steps) + 1,
        "item_id": item_id,
        "pipeline_sha256": current.pipeline_sha256,
        "policy_sha256": current.policy_sha256,
        "candidate_input_sha256": current.current_candidate_input_sha256,
        "graph_sha256": current.current_graph_sha256,
        "synthesis_sha256": current.current_synthesis_sha256,
        "selection_state_sha256": current.session_sha256,
        "scheduler_artifact_sha256": scheduler_artifact_sha256,
        "estimated_cost": estimated_cost,
        "cost_unit": current.cost_unit,
        "selection_rank": selection_rank,
        "selection_score": selection_score,
        "selected_at": selected_at,
    }
    action = _freeze_model(
        AuditActionPacket,
        action_payload,
        hash_field="packet_sha256",
    )
    selected = _transition_session(
        current,
        selected_item_ids=(*current.selected_item_ids, action.item_id),
        active_action=action.model_dump(mode="json"),
    )
    return selected, action


def checkpoint_active_cost(
    session: VerificationSession,
    *,
    expected_state_sha256: str,
    action_packet_sha256: str,
    active_realized_cost: float,
) -> VerificationSession:
    """Checkpoint monotone partial realized cost for the active action."""

    current = resume_verification_session(
        session,
        expected_state_sha256=expected_state_sha256,
    )
    if current.status is VerificationSessionStatus.FINALIZED:
        raise AuditSessionContractError("finalized_session_is_terminal")
    action = current.active_action
    if action is None:
        raise AuditSessionContractError("active_cost_requires_selected_action")
    if action.packet_sha256 != action_packet_sha256:
        raise StaleAuditStateError("active_action_packet_hash_mismatch")
    _validate_cost(active_realized_cost, "active_realized_cost")
    if active_realized_cost + _COST_ABS_TOLERANCE < current.active_realized_cost:
        raise AuditSessionContractError("active_realized_cost_cannot_decrease")
    if _costs_equal(active_realized_cost, current.active_realized_cost):
        return current
    current_cost = math.fsum((current.historical_realized_cost, active_realized_cost))
    if current_cost > current.budget + _COST_ABS_TOLERANCE:
        raise AuditSessionContractError("active_realized_cost_exceeds_budget")
    return _transition_session(
        current,
        active_realized_cost=active_realized_cost,
        current_realized_cost=current_cost,
        remaining_budget=max(0.0, current.budget - current_cost),
    )


def freeze_adjudication_artifact(
    action: AuditActionPacket,
    *,
    provenance: str,
    adjudicator_count: int,
    protocol_sha256: str,
    payload_sha256: str,
    completed_at: datetime,
    realized_cost: float,
) -> AdjudicationArtifact:
    """Freeze external adjudication metadata against one selected action."""

    if completed_at < action.selected_at:
        raise AuditSessionContractError("adjudication_completed_before_selection")
    payload = {
        "artifact_version": "adjudication-artifact-v2",
        "session_id": action.session_id,
        "step": action.step,
        "item_id": action.item_id,
        "action_packet_sha256": action.packet_sha256,
        "provenance": provenance,
        "adjudicator_count": adjudicator_count,
        "protocol_sha256": protocol_sha256,
        "payload_sha256": payload_sha256,
        "completed_at": completed_at,
        "realized_cost": realized_cost,
        "cost_unit": action.cost_unit,
    }
    return _freeze_model(
        AdjudicationArtifact,
        payload,
        hash_field="artifact_sha256",
    )


def freeze_evidence_graph_correction(
    action: AuditActionPacket,
    adjudication: AdjudicationArtifact,
    *,
    disposition: CorrectionDisposition,
    post_graph_sha256: str,
    post_synthesis_sha256: str,
    post_candidate_input_sha256: str,
    correction_artifact_sha256: str,
) -> EvidenceGraphCorrection:
    """Freeze refreshed state produced by applying an adjudication."""

    payload = {
        "correction_version": "evidence-graph-correction-v2",
        "session_id": action.session_id,
        "step": action.step,
        "item_id": action.item_id,
        "action_packet_sha256": action.packet_sha256,
        "adjudication_artifact_sha256": adjudication.artifact_sha256,
        "disposition": disposition,
        "pre_graph_sha256": action.graph_sha256,
        "pre_synthesis_sha256": action.synthesis_sha256,
        "pre_candidate_input_sha256": action.candidate_input_sha256,
        "post_graph_sha256": post_graph_sha256,
        "post_synthesis_sha256": post_synthesis_sha256,
        "post_candidate_input_sha256": post_candidate_input_sha256,
        "correction_artifact_sha256": correction_artifact_sha256,
    }
    return _freeze_model(
        EvidenceGraphCorrection,
        payload,
        hash_field="correction_sha256",
    )


def _freeze_resolution_receipt(
    session: VerificationSession,
    action: AuditActionPacket,
    adjudication: AdjudicationArtifact,
    correction: EvidenceGraphCorrection,
) -> AuditResolutionReceiptV2:
    cumulative = math.fsum(
        (session.historical_realized_cost, adjudication.realized_cost)
    )
    payload = {
        "receipt_version": "audit-resolution-v2",
        "session_id": session.session_id,
        "step": action.step,
        "item_id": action.item_id,
        "pipeline_sha256": session.pipeline_sha256,
        "policy_sha256": session.policy_sha256,
        "action_packet_sha256": action.packet_sha256,
        "selection_state_sha256": action.selection_state_sha256,
        "state_before_resolution_sha256": session.session_sha256,
        "candidate_input_sha256": action.candidate_input_sha256,
        "pre_graph_sha256": action.graph_sha256,
        "pre_synthesis_sha256": action.synthesis_sha256,
        "adjudication_protocol_sha256": adjudication.protocol_sha256,
        "adjudication_artifact_sha256": adjudication.artifact_sha256,
        "correction_sha256": correction.correction_sha256,
        "post_graph_sha256": correction.post_graph_sha256,
        "post_synthesis_sha256": correction.post_synthesis_sha256,
        "post_candidate_input_sha256": correction.post_candidate_input_sha256,
        "realized_cost": adjudication.realized_cost,
        "active_realized_cost_before_resolution": session.active_realized_cost,
        "historical_realized_cost_before_resolution": session.historical_realized_cost,
        "cumulative_realized_cost_after_resolution": cumulative,
        "budget": session.budget,
        "remaining_budget_after_resolution": max(0.0, session.budget - cumulative),
        "cost_unit": session.cost_unit,
    }
    return _freeze_model(
        AuditResolutionReceiptV2,
        payload,
        hash_field="receipt_sha256",
    )


def _freeze_sequential_step(
    session: VerificationSession,
    action: AuditActionPacket,
    adjudication: AdjudicationArtifact,
    correction: EvidenceGraphCorrection,
    receipt: AuditResolutionReceiptV2,
) -> SequentialAuditStepV2:
    payload = {
        "step_version": "sequential-audit-step-v2",
        "session_id": session.session_id,
        "step": action.step,
        "item_id": action.item_id,
        "action": action,
        "adjudication": adjudication,
        "correction": correction,
        "receipt": receipt,
        "previous_step_sha256": session.steps[-1].step_sha256 if session.steps else None,
    }
    return _freeze_model(
        SequentialAuditStepV2,
        payload,
        hash_field="step_sha256",
    )


def resolve_audit_action(
    session: VerificationSession,
    *,
    expected_state_sha256: str,
    adjudication: AdjudicationArtifact,
    correction: EvidenceGraphCorrection | None = None,
    apply_correction: EvidenceGraphCorrectionCallback | None = None,
) -> tuple[VerificationSession, AuditResolutionReceiptV2]:
    """Resolve the selected action and advance only from its exact current state.

    Supply either a prebuilt ``correction`` or an ``apply_correction`` callback.  The
    callback boundary keeps graph mutation and synthesis outside this generic module.
    """

    current = resume_verification_session(
        session,
        expected_state_sha256=expected_state_sha256,
    )
    if current.status is VerificationSessionStatus.FINALIZED:
        raise AuditSessionContractError("finalized_session_is_terminal")
    action = current.active_action
    if action is None:
        raise AuditSessionContractError("resolution_requires_selected_action")
    if (correction is None) == (apply_correction is None):
        raise AuditSessionContractError(
            "resolution_requires_exactly_one_correction_or_callback"
        )
    adjudication = AdjudicationArtifact.model_validate(adjudication.model_dump(mode="json"))
    if (
        adjudication.session_id,
        adjudication.step,
        adjudication.item_id,
        adjudication.action_packet_sha256,
    ) != (
        current.session_id,
        action.step,
        action.item_id,
        action.packet_sha256,
    ):
        raise AuditSessionContractError("adjudication_does_not_match_active_action")
    if adjudication.cost_unit != current.cost_unit:
        raise AuditSessionContractError("adjudication_cost_unit_mismatch")
    if adjudication.completed_at < action.selected_at:
        raise AuditSessionContractError("adjudication_completed_before_selection")
    if adjudication.realized_cost + _COST_ABS_TOLERANCE < current.active_realized_cost:
        raise AuditSessionContractError("adjudication_cost_below_active_checkpoint")
    if (
        current.historical_realized_cost + adjudication.realized_cost
        > current.budget + _COST_ABS_TOLERANCE
    ):
        raise AuditSessionContractError("adjudication_realized_cost_exceeds_budget")

    if apply_correction is not None:
        resolved_correction = apply_correction(current, action, adjudication)
    else:
        assert correction is not None
        resolved_correction = correction
    resolved_correction = EvidenceGraphCorrection.model_validate(
        resolved_correction.model_dump(mode="json")
    )
    if (
        resolved_correction.session_id,
        resolved_correction.step,
        resolved_correction.item_id,
        resolved_correction.action_packet_sha256,
        resolved_correction.adjudication_artifact_sha256,
    ) != (
        current.session_id,
        action.step,
        action.item_id,
        action.packet_sha256,
        adjudication.artifact_sha256,
    ):
        raise AuditSessionContractError("correction_does_not_match_active_action")
    if (
        resolved_correction.pre_graph_sha256,
        resolved_correction.pre_synthesis_sha256,
        resolved_correction.pre_candidate_input_sha256,
    ) != (
        current.current_graph_sha256,
        current.current_synthesis_sha256,
        current.current_candidate_input_sha256,
    ):
        raise StaleAuditStateError("correction_pre_state_is_stale")

    receipt = _freeze_resolution_receipt(
        current,
        action,
        adjudication,
        resolved_correction,
    )
    step = _freeze_sequential_step(
        current,
        action,
        adjudication,
        resolved_correction,
        receipt,
    )
    historical_cost = receipt.cumulative_realized_cost_after_resolution
    advanced = _transition_session(
        current,
        current_graph_sha256=resolved_correction.post_graph_sha256,
        current_synthesis_sha256=resolved_correction.post_synthesis_sha256,
        current_candidate_input_sha256=resolved_correction.post_candidate_input_sha256,
        resolved_item_ids=(*current.resolved_item_ids, action.item_id),
        steps=(*current.steps, step.model_dump(mode="json")),
        active_action=None,
        historical_realized_cost=historical_cost,
        active_realized_cost=0.0,
        current_realized_cost=historical_cost,
        remaining_budget=max(0.0, current.budget - historical_cost),
    )
    return advanced, receipt


def finalize_verification_session(
    session: VerificationSession,
    *,
    expected_state_sha256: str,
    final_assessment_sha256: str,
    reason: str,
    finalized_at: datetime,
) -> VerificationSession:
    """Finalize an idle session; repeating the same finalization is idempotent."""

    current = resume_verification_session(
        session,
        expected_state_sha256=expected_state_sha256,
    )
    validated_assessment_hash = _validate_sha256(
        final_assessment_sha256,
        "final_assessment_sha256",
    )
    assert validated_assessment_hash is not None
    _validate_aware(finalized_at, "finalized_at")
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise AuditSessionContractError("session_final_reason_empty")
    if current.status is VerificationSessionStatus.FINALIZED:
        if (
            current.final_assessment_sha256 == validated_assessment_hash
            and current.final_reason == normalized_reason
            and current.finalized_at == finalized_at
        ):
            return current
        raise AuditSessionContractError("finalized_session_is_terminal")
    if current.active_action is not None:
        raise AuditSessionContractError("cannot_finalize_with_active_action")

    return _transition_session(
        current,
        status=VerificationSessionStatus.FINALIZED,
        finalized_from_state_sha256=current.session_sha256,
        final_assessment_state_sha256=current.session_sha256,
        final_assessment_sha256=validated_assessment_hash,
        final_reason=normalized_reason,
        finalized_at=finalized_at,
    )


def finalize_with_callback(
    session: VerificationSession,
    *,
    expected_state_sha256: str,
    assess_final_state: FinalAssessmentCallback,
    reason: str,
    finalized_at: datetime,
) -> VerificationSession:
    """Run an external final assessment once, then bind its hash to this state."""

    current = resume_verification_session(
        session,
        expected_state_sha256=expected_state_sha256,
    )
    if current.status is VerificationSessionStatus.FINALIZED:
        raise AuditSessionContractError(
            "finalize_callback_not_replayed_for_finalized_session"
        )
    assessment_sha256 = assess_final_state(current)
    return finalize_verification_session(
        current,
        expected_state_sha256=current.session_sha256,
        final_assessment_sha256=assessment_sha256,
        reason=reason,
        finalized_at=finalized_at,
    )


__all__ = [
    "AdjudicationArtifact",
    "AuditActionPacket",
    "AuditResolutionReceiptV2",
    "AuditSessionContractError",
    "CorrectionDisposition",
    "EvidenceGraphCorrection",
    "EvidenceGraphCorrectionCallback",
    "FinalAssessmentCallback",
    "SequentialAuditStepV2",
    "StaleAuditStateError",
    "VerificationSession",
    "VerificationSessionStatus",
    "checkpoint_active_cost",
    "create_verification_session",
    "finalize_verification_session",
    "finalize_with_callback",
    "freeze_adjudication_artifact",
    "freeze_evidence_graph_correction",
    "resolve_audit_action",
    "resume_verification_session",
    "select_audit_action",
]
