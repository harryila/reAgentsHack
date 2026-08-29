from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from literature_multiverse.audit_session import (
    AuditSessionContractError,
    CorrectionDisposition,
    StaleAuditStateError,
    VerificationSession,
    VerificationSessionStatus,
    checkpoint_active_cost,
    create_verification_session,
    finalize_verification_session,
    finalize_with_callback,
    freeze_adjudication_artifact,
    freeze_evidence_graph_correction,
    resolve_audit_action,
    resume_verification_session,
    select_audit_action,
)
from literature_multiverse.lineage import hash_canonical

_CREATED_AT = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def _hash(label: str) -> str:
    return hash_canonical({"fixture": label})


def _session(*, session_id: str = "session-1", budget: float = 10.0):
    return create_verification_session(
        session_id=session_id,
        created_at=_CREATED_AT,
        pipeline_sha256=_hash("pipeline"),
        policy_sha256=_hash("policy"),
        budget=budget,
        cost_unit="reviewer_minutes",
        graph_sha256=_hash("graph-0"),
        synthesis_sha256=_hash("synthesis-0"),
        candidate_input_sha256=_hash("candidates-0"),
    )


def _select(session, *, item_id: str = "estimate-1", estimated_cost: float = 4.0):
    return select_audit_action(
        session,
        expected_state_sha256=session.session_sha256,
        item_id=item_id,
        scheduler_artifact_sha256=_hash(f"scheduler-{item_id}"),
        estimated_cost=estimated_cost,
        selection_rank=1,
        selection_score=0.75,
        selected_at=_CREATED_AT + timedelta(minutes=1 + 5 * len(session.steps)),
    )


def _adjudication(action, *, realized_cost: float = 2.25):
    return freeze_adjudication_artifact(
        action,
        provenance="blinded_human_adjudication",
        adjudicator_count=2,
        protocol_sha256=_hash("protocol"),
        payload_sha256=_hash(f"adjudication-{action.item_id}"),
        completed_at=_CREATED_AT + timedelta(minutes=5),
        realized_cost=realized_cost,
    )


def _correction(action, adjudication, *, suffix: str = "1"):
    return freeze_evidence_graph_correction(
        action,
        adjudication,
        disposition=CorrectionDisposition.CORRECTED,
        post_graph_sha256=_hash(f"graph-{suffix}"),
        post_synthesis_sha256=_hash(f"synthesis-{suffix}"),
        post_candidate_input_sha256=_hash(f"candidates-{suffix}"),
        correction_artifact_sha256=_hash(f"correction-{suffix}"),
    )


def test_create_and_select_are_deterministic_and_estimates_do_not_spend() -> None:
    initial = _session()

    selected, action = _select(initial)
    repeated, repeated_action = _select(_session())

    assert repeated_action == action
    assert repeated == selected
    assert selected.active_action == action
    assert selected.selected_item_ids == ("estimate-1",)
    assert selected.resolved_item_ids == ()
    assert selected.historical_realized_cost == 0.0
    assert selected.active_realized_cost == 0.0
    assert selected.current_realized_cost == 0.0
    assert selected.remaining_budget == 10.0
    assert selected.transition_index == 1
    assert selected.previous_session_sha256 == initial.session_sha256
    assert initial.active_action is None


def test_zero_budget_is_valid_but_cannot_select_and_can_finalize() -> None:
    session = _session(budget=0.0)

    with pytest.raises(AuditSessionContractError, match="zero_budget_session_cannot_select"):
        _select(session, estimated_cost=1.0)

    finalized = finalize_verification_session(
        session,
        expected_state_sha256=session.session_sha256,
        final_assessment_sha256=_hash("zero-budget-assessment"),
        reason="budget_exhausted",
        finalized_at=_CREATED_AT + timedelta(minutes=1),
    )

    assert finalized.status is VerificationSessionStatus.FINALIZED
    assert finalized.current_realized_cost == 0.0
    assert finalized.selected_item_ids == ()


def test_active_cost_is_monotone_hash_bound_and_included_in_current_cost() -> None:
    selected, action = _select(_session())
    checkpoint = checkpoint_active_cost(
        selected,
        expected_state_sha256=selected.session_sha256,
        action_packet_sha256=action.packet_sha256,
        active_realized_cost=1.5,
    )

    assert checkpoint.historical_realized_cost == 0.0
    assert checkpoint.active_realized_cost == 1.5
    assert checkpoint.current_realized_cost == 1.5
    assert checkpoint.remaining_budget == 8.5
    assert checkpoint_active_cost(
        checkpoint,
        expected_state_sha256=checkpoint.session_sha256,
        action_packet_sha256=action.packet_sha256,
        active_realized_cost=1.5,
    ) == checkpoint

    with pytest.raises(StaleAuditStateError, match="stale_audit_session_state"):
        checkpoint_active_cost(
            checkpoint,
            expected_state_sha256=selected.session_sha256,
            action_packet_sha256=action.packet_sha256,
            active_realized_cost=2.0,
        )
    with pytest.raises(
        AuditSessionContractError, match="active_realized_cost_cannot_decrease"
    ):
        checkpoint_active_cost(
            checkpoint,
            expected_state_sha256=checkpoint.session_sha256,
            action_packet_sha256=action.packet_sha256,
            active_realized_cost=1.0,
        )
    with pytest.raises(
        AuditSessionContractError, match="active_realized_cost_exceeds_budget"
    ):
        checkpoint_active_cost(
            checkpoint,
            expected_state_sha256=checkpoint.session_sha256,
            action_packet_sha256=action.packet_sha256,
            active_realized_cost=11.0,
        )


def test_resolution_charges_realized_cost_and_refreshes_all_state_hashes() -> None:
    selected, action = _select(_session())
    checkpoint = checkpoint_active_cost(
        selected,
        expected_state_sha256=selected.session_sha256,
        action_packet_sha256=action.packet_sha256,
        active_realized_cost=1.5,
    )
    adjudication = _adjudication(action, realized_cost=2.25)
    correction = _correction(action, adjudication)
    callback_calls = []

    def apply_correction(session, selected_action, artifact):
        callback_calls.append((session.session_sha256, selected_action, artifact))
        return correction

    resolved, receipt = resolve_audit_action(
        checkpoint,
        expected_state_sha256=checkpoint.session_sha256,
        adjudication=adjudication,
        apply_correction=apply_correction,
    )

    assert callback_calls == [(checkpoint.session_sha256, action, adjudication)]
    assert resolved.selected_item_ids == (action.item_id,)
    assert resolved.resolved_item_ids == (action.item_id,)
    assert resolved.active_action is None
    assert resolved.historical_realized_cost == 2.25
    assert resolved.active_realized_cost == 0.0
    assert resolved.current_realized_cost == 2.25
    assert resolved.remaining_budget == 7.75
    assert receipt.realized_cost == 2.25
    assert receipt.realized_cost != action.estimated_cost
    assert receipt.active_realized_cost_before_resolution == 1.5
    assert receipt.state_before_resolution_sha256 == checkpoint.session_sha256
    assert resolved.current_graph_sha256 == correction.post_graph_sha256
    assert resolved.current_synthesis_sha256 == correction.post_synthesis_sha256
    assert (
        resolved.current_candidate_input_sha256
        == correction.post_candidate_input_sha256
    )
    assert resume_verification_session(resolved.model_dump(mode="json")) == resolved

    selected_again, next_action = _select(resolved, item_id="estimate-2")
    assert next_action.candidate_input_sha256 == correction.post_candidate_input_sha256
    assert next_action.graph_sha256 == correction.post_graph_sha256
    assert selected_again.resolved_item_ids == (action.item_id,)


def test_resolution_requires_selection_and_matching_action() -> None:
    selected, action = _select(_session())
    adjudication = _adjudication(action)
    correction = _correction(action, adjudication)

    with pytest.raises(
        AuditSessionContractError, match="resolution_requires_selected_action"
    ):
        resolve_audit_action(
            _session(),
            expected_state_sha256=_session().session_sha256,
            adjudication=adjudication,
            correction=correction,
        )

    other_selected, _ = _select(_session(session_id="session-2"), item_id="estimate-2")
    with pytest.raises(
        AuditSessionContractError, match="adjudication_does_not_match_active_action"
    ):
        resolve_audit_action(
            other_selected,
            expected_state_sha256=other_selected.session_sha256,
            adjudication=adjudication,
            correction=correction,
        )

    assert selected.resolved_item_ids == ()


def test_resolution_rejects_cost_below_checkpoint_or_above_budget() -> None:
    selected, action = _select(_session())
    checkpoint = checkpoint_active_cost(
        selected,
        expected_state_sha256=selected.session_sha256,
        action_packet_sha256=action.packet_sha256,
        active_realized_cost=3.0,
    )
    too_low = _adjudication(action, realized_cost=2.0)
    too_low_correction = _correction(action, too_low)

    with pytest.raises(
        AuditSessionContractError, match="adjudication_cost_below_active_checkpoint"
    ):
        resolve_audit_action(
            checkpoint,
            expected_state_sha256=checkpoint.session_sha256,
            adjudication=too_low,
            correction=too_low_correction,
        )

    over_budget = _adjudication(action, realized_cost=11.0)
    over_budget_correction = _correction(action, over_budget)
    with pytest.raises(
        AuditSessionContractError, match="adjudication_realized_cost_exceeds_budget"
    ):
        resolve_audit_action(
            checkpoint,
            expected_state_sha256=checkpoint.session_sha256,
            adjudication=over_budget,
            correction=over_budget_correction,
        )


def test_stale_correction_state_is_rejected() -> None:
    selected, action = _select(_session())
    adjudication = _adjudication(action)
    correction = _correction(action, adjudication)
    payload = correction.model_dump(mode="json", exclude={"correction_sha256"})
    payload["pre_graph_sha256"] = _hash("unrelated-old-graph")
    stale = type(correction).model_validate(
        {**payload, "correction_sha256": hash_canonical(payload)}
    )

    with pytest.raises(StaleAuditStateError, match="correction_pre_state_is_stale"):
        resolve_audit_action(
            selected,
            expected_state_sha256=selected.session_sha256,
            adjudication=adjudication,
            correction=stale,
        )


def test_resume_rejects_nested_or_session_hash_tampering() -> None:
    selected, _ = _select(_session())
    tampered = selected.model_dump(mode="json")
    tampered["active_action"]["scheduler_artifact_sha256"] = _hash("replacement")

    with pytest.raises(ValueError, match="audit_action_packet_hash_mismatch"):
        resume_verification_session(tampered)

    tampered = selected.model_dump(mode="json")
    tampered["remaining_budget"] = 9.0
    with pytest.raises(ValueError, match="session_remaining_budget_mismatch"):
        resume_verification_session(tampered)


def test_selected_before_resolved_invariant_survives_rehashed_forgery() -> None:
    session = _session()
    payload = session.model_dump(mode="json", exclude={"session_sha256"})
    payload["selected_item_ids"] = ["unselected-item"]
    payload["resolved_item_ids"] = ["unselected-item"]

    with pytest.raises(ValueError, match="session_resolved_items_do_not_match_receipts"):
        VerificationSession.model_validate(
            {**payload, "session_sha256": hash_canonical(payload)}
        )


def test_finalize_is_terminal_hash_bound_and_idempotent() -> None:
    session = _session()
    assessment_hash = _hash("final-release-assessment")
    finalized_at = _CREATED_AT + timedelta(minutes=10)
    finalized = finalize_verification_session(
        session,
        expected_state_sha256=session.session_sha256,
        final_assessment_sha256=assessment_hash,
        reason="residual_guard_satisfied",
        finalized_at=finalized_at,
    )

    assert finalized.finalized_from_state_sha256 == session.session_sha256
    assert finalized.final_assessment_state_sha256 == session.session_sha256
    assert finalized.final_assessment_sha256 == assessment_hash
    assert finalize_verification_session(
        finalized,
        expected_state_sha256=finalized.session_sha256,
        final_assessment_sha256=assessment_hash,
        reason="residual_guard_satisfied",
        finalized_at=finalized_at,
    ) == finalized

    with pytest.raises(AuditSessionContractError, match="finalized_session_is_terminal"):
        _select(finalized)
    with pytest.raises(AuditSessionContractError, match="finalized_session_is_terminal"):
        checkpoint_active_cost(
            finalized,
            expected_state_sha256=finalized.session_sha256,
            action_packet_sha256=_hash("not-an-action"),
            active_realized_cost=1.0,
        )


def test_finalize_callback_is_bound_to_exact_idle_state_and_not_replayed() -> None:
    session = _session()
    calls = []

    def assess(current):
        calls.append(current.session_sha256)
        return _hash(f"assessment-{current.session_sha256}")

    finalized = finalize_with_callback(
        session,
        expected_state_sha256=session.session_sha256,
        assess_final_state=assess,
        reason="operator_stop",
        finalized_at=_CREATED_AT + timedelta(minutes=2),
    )

    assert calls == [session.session_sha256]
    assert finalized.final_assessment_state_sha256 == session.session_sha256
    with pytest.raises(
        AuditSessionContractError,
        match="finalize_callback_not_replayed_for_finalized_session",
    ):
        finalize_with_callback(
            finalized,
            expected_state_sha256=finalized.session_sha256,
            assess_final_state=assess,
            reason="operator_stop",
            finalized_at=_CREATED_AT + timedelta(minutes=2),
        )
    assert calls == [session.session_sha256]


def test_cannot_finalize_while_an_action_is_selected() -> None:
    selected, _ = _select(_session())

    with pytest.raises(
        AuditSessionContractError, match="cannot_finalize_with_active_action"
    ):
        finalize_verification_session(
            selected,
            expected_state_sha256=selected.session_sha256,
            final_assessment_sha256=_hash("premature-assessment"),
            reason="operator_stop",
            finalized_at=_CREATED_AT + timedelta(minutes=2),
        )
