from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from literature_multiverse.adaptive_calibration import (
    AdaptivePreselectionState,
    freeze_adaptive_preselection_state,
)
from literature_multiverse.audit_session import CorrectionDisposition
from literature_multiverse.budgeted_verification import (
    AllocationPolicy,
    AuditCandidate,
    ClaimModel,
    ProbabilityBasis,
    ScenarioKind,
)
from literature_multiverse.effects import EffectEvidence
from literature_multiverse.evidence_graph import (
    CohortIdentity,
    EvidenceGraph,
    GraphAdapterContext,
    OutcomeTimepoint,
    PublicationIdentity,
    adapt_effect_evidence,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.sequential_verification import (
    SequentialStateExpectation,
    SequentialVerificationContractError,
    SequentialVerificationState,
    StaleSequentialVerificationStateError,
    adaptive_preselection_history_from_state,
    checkpoint_selected_audit_cost,
    create_sequential_verification_state,
    current_candidates_from_audit_candidates,
    freeze_current_audit_candidate,
    freeze_selected_adjudication,
    freeze_state_expectation,
    resolve_selected_audit_candidate,
    resume_sequential_verification_state,
    select_next_audit_candidate,
)

_CREATED_AT = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def _hash(label: str) -> str:
    return hash_canonical({"fixture": label})


def _single_graph(suffix: str, estimate: float) -> EvidenceGraph:
    context = GraphAdapterContext(
        publication=PublicationIdentity(
            publication_id=f"publication-{suffix}",
            paper_id=f"paper-{suffix}",
            doc_id=f"document-{suffix}",
            doi=f"10.1000/{suffix}",
        ),
        study_id=f"study-{suffix}",
        cohort_identity=CohortIdentity(
            cohort_id=f"cohort-{suffix}",
            basis="reviewer_reconciled",
            rationale="Two reviewers reconciled the cohort identity.",
        ),
        treatment_arm_id=f"arm-{suffix}-treatment",
        comparator_arm_id=f"arm-{suffix}-control",
        contrast_id=f"contrast-{suffix}",
        contrast_label="intervention_vs_control",
        positive_direction_means="higher outcome value under intervention",
        treatment_label="intervention",
        comparator_label="control",
        timepoint=OutcomeTimepoint(kind="exact", value=4, unit="week"),
    )
    evidence = EffectEvidence(
        paper_id=f"paper-{suffix}",
        finding_id=f"finding-{suffix}",
        outcome="performance",
        contrast="intervention_vs_control",
        effect_format="hedges_g",
        estimate=estimate,
        standard_error=0.1,
        provenance={
            "source_locator": f"paper-{suffix}.pdf#page=4",
            "source_quote": f"The estimate was {estimate}.",
        },
    )
    return adapt_effect_evidence(evidence, context=context).graph


def _graph() -> EvidenceGraph:
    graphs = (_single_graph("a", 0.2), _single_graph("b", 0.4))
    return EvidenceGraph(
        publications=[item for graph in graphs for item in graph.publications],
        studies=[item for graph in graphs for item in graph.studies],
        cohorts=[item for graph in graphs for item in graph.cohorts],
        arms=[item for graph in graphs for item in graph.arms],
        contrasts=[item for graph in graphs for item in graph.contrasts],
        outcome_estimates=[item for graph in graphs for item in graph.outcome_estimates],
        evidence_spans=[item for graph in graphs for item in graph.evidence_spans],
    )


def _synthesis(graph: EvidenceGraph):
    return {
        "status": "ok",
        "estimate_values": {
            estimate.estimate_id: estimate.effect.estimate
            for estimate in graph.outcome_estimates
        },
    }


def _candidates(graph: EvidenceGraph, *_):
    frozen = []
    for estimate in graph.outcome_estimates:
        suffix = estimate.estimate_id.rsplit("-", maxsplit=1)[-1]
        frozen.append(
            freeze_current_audit_candidate(
                item_id=estimate.estimate_id,
                priority=0.9 if suffix == "a" else 0.8,
                estimated_cost=4.0 if suffix == "a" else 2.0,
                cost_unit="reviewer_minutes",
                scientific_candidate_sha256=hash_canonical(estimate),
                counterfactual_synthesis_sha256=hash_canonical(
                    {"leave_one_out": estimate.estimate_id, "graph": hash_canonical(graph)}
                ),
            )
        )
    return frozen


def _state(*, budget: float = 5.0, adaptive: bool = False):
    graph = _graph()
    return create_sequential_verification_state(
        session_id="session-1",
        created_at=_CREATED_AT,
        pipeline_sha256=_hash("pipeline"),
        policy_sha256=_hash("policy"),
        budget=budget,
        cost_unit="reviewer_minutes",
        graph=graph,
        synthesis=_synthesis(graph),
        candidates=_candidates(graph),
        adaptive_policy_context_sha256=(
            _hash("adaptive-context") if adaptive else None
        ),
        adaptive_calibration_bundle_sha256=(
            _hash("adaptive-bundle") if adaptive else None
        ),
    )


def _select(state):
    return select_next_audit_candidate(
        state,
        expected=freeze_state_expectation(state),
        selected_at=_CREATED_AT + timedelta(minutes=1),
    )


def _adaptive_checkpoint(state) -> AdaptivePreselectionState:
    return freeze_adaptive_preselection_state(
        prefix_index=len(state.session.resolved_item_ids),
        audit_prefix_item_ids=state.session.resolved_item_ids,
        audit_prefix_cost_minutes=state.session.historical_realized_cost,
        scheduler_state_sha256=state.state_sha256,
        evidence_graph_sha256=state.graph_sha256,
        synthesis_sha256=state.synthesis_sha256,
        non_calibration_assessment_sha256=hash_canonical(
            {
                "assessment": "label-free",
                "scheduler_state_sha256": state.state_sha256,
            }
        ),
        non_calibration_gates_passed=False,
        non_calibration_blocking_reasons=["scientific_release_gate_blocked"],
        claim_decision="supported",
        score_features={"evidence_count": float(len(state.graph.outcome_estimates))},
    )


def _select_adaptive(
    state,
    *,
    selected_at: datetime,
    context_sha256: str | None = None,
    bundle_sha256: str | None = None,
):
    return select_next_audit_candidate(
        state,
        expected=freeze_state_expectation(state),
        selected_at=selected_at,
        adaptive_preselection_state=_adaptive_checkpoint(state),
        adaptive_policy_context_sha256=context_sha256 or _hash("adaptive-context"),
        adaptive_calibration_bundle_sha256=bundle_sha256 or _hash("adaptive-bundle"),
    )


def _adjudication(selected, *, realized_cost: float = 1.5):
    return freeze_selected_adjudication(
        selected.state,
        expected=freeze_state_expectation(selected.state),
        provenance="blinded_human_adjudication",
        adjudicator_count=2,
        protocol_sha256=_hash("adjudication-protocol"),
        payload_sha256=_hash("adjudication-payload"),
        completed_at=_CREATED_AT + timedelta(minutes=3),
        realized_cost=realized_cost,
    )


def _correct_estimate(graph: EvidenceGraph, item_id: str, value: float) -> EvidenceGraph:
    payload = graph.model_dump(mode="json")
    selected = next(
        estimate for estimate in payload["outcome_estimates"] if estimate["estimate_id"] == item_id
    )
    selected["effect"]["estimate"] = value
    return EvidenceGraph.model_validate(payload)


def _remove_estimate(graph: EvidenceGraph, item_id: str) -> EvidenceGraph:
    payload = graph.model_dump(mode="json")
    payload["outcome_estimates"] = [
        estimate
        for estimate in payload["outcome_estimates"]
        if estimate["estimate_id"] != item_id
    ]
    return EvidenceGraph.model_validate(payload)


def _resolve(selected, adjudication, *, disposition, corrected_graph):
    return resolve_selected_audit_candidate(
        selected.state,
        expected=freeze_state_expectation(selected.state),
        adjudication=adjudication,
        disposition=disposition,
        corrected_graph=corrected_graph,
        correction_provenance="selected_estimate_reconciled_against_source",
        correction_protocol_sha256=_hash("correction-protocol"),
        external_correction_payload_sha256=_hash("correction-payload"),
        synthesis_runner_sha256=_hash("synthesis-runner"),
        candidate_runner_sha256=_hash("candidate-runner"),
        rerun_synthesis=_synthesis,
        rerun_candidates=_candidates,
    )


def test_selection_is_deterministic_and_estimated_cost_does_not_spend() -> None:
    state = _state()

    first = _select(state)
    repeated = _select(_state())

    assert first == repeated
    assert first.candidate.item_id.endswith("-a")
    assert first.action.selection_rank == 1
    assert first.action.estimated_cost == 4.0
    assert first.state.session.current_realized_cost == 0.0
    assert first.state.session.remaining_budget == 5.0
    assert adaptive_preselection_history_from_state(first.state) == ((), None, None)
    transition_payload = first.state.transitions[-1].model_dump(mode="json")
    assert "adaptive_preselection_state" not in transition_payload
    assert "adaptive_policy_context_sha256" not in transition_payload
    assert "adaptive_calibration_bundle_sha256" not in transition_payload


def test_budget_deadline_checkpoints_an_active_action_without_applying_adjudication() -> None:
    selected = _select(_state(budget=5.0))

    checkpoint = checkpoint_selected_audit_cost(
        selected.state,
        expected=freeze_state_expectation(selected.state),
        active_realized_cost=5.0,
    )

    assert checkpoint.state.graph_sha256 == selected.state.graph_sha256
    assert checkpoint.state.synthesis_sha256 == selected.state.synthesis_sha256
    assert checkpoint.state.candidate_input_sha256 == selected.state.candidate_input_sha256
    assert checkpoint.state.session.active_action == selected.action
    assert checkpoint.state.session.selected_item_ids == (selected.action.item_id,)
    assert checkpoint.state.session.resolved_item_ids == ()
    assert checkpoint.state.session.historical_realized_cost == 0.0
    assert checkpoint.state.session.active_realized_cost == 5.0
    assert checkpoint.state.session.current_realized_cost == 5.0
    assert checkpoint.state.session.remaining_budget == 0.0
    assert checkpoint.result_sha256 == hash_canonical(
        checkpoint.model_dump(mode="json", exclude={"result_sha256"})
    )

    with pytest.raises(
        SequentialVerificationContractError,
        match="active_realized_cost_cannot_decrease",
    ):
        checkpoint_selected_audit_cost(
            checkpoint.state,
            expected=freeze_state_expectation(checkpoint.state),
            active_realized_cost=4.9,
        )


def test_existing_counterfactual_candidates_have_a_direct_deterministic_adapter() -> None:
    candidates = [
        AuditCandidate(
            item_id="estimate-a",
            baseline_contribution=0.2,
            counterfactual_contribution=0.0,
            error_probability=0.3,
            probability_basis=ProbabilityBasis.HEURISTIC,
            probability_source="test-score-not-release-proof",
            verification_cost=2.0,
            cost_unit="reviewer_minutes",
            disagreement_score=0.1,
            scenario_kind=ScenarioKind.LEAVE_ONE_OUT,
            scenario_source="actual_synthesis_rerun",
        )
    ]

    frozen = current_candidates_from_audit_candidates(
        candidates,
        ClaimModel(intercept=0.0),
        policy=AllocationPolicy.RISK_X_INFLUENCE,
        counterfactual_synthesis_sha256s={"estimate-a": _hash("counterfactual-a")},
    )

    assert len(frozen) == 1
    assert frozen[0].item_id == "estimate-a"
    assert frozen[0].estimated_cost == 2.0
    assert frozen[0].scientific_candidate_sha256 == hash_canonical(
        {
            "item_id": "estimate-a",
            "baseline_contribution": 0.2,
            "counterfactual_contribution": 0.0,
            "error_probability": 0.3,
            "probability_basis": "heuristic",
            "probability_source": "test-score-not-release-proof",
            "verification_cost": 2.0,
            "cost_unit": "reviewer_minutes",
            "disagreement_score": 0.1,
            "scenario_kind": "leave_one_out",
            "scenario_source": "actual_synthesis_rerun",
            "baseline_decision_score": None,
            "counterfactual_decision_score": None,
            "decision_score_source": None,
            "baseline_decision": None,
            "counterfactual_decision": None,
        }
    )


def test_zero_budget_cannot_select() -> None:
    state = _state(budget=0.0)

    with pytest.raises(SequentialVerificationContractError, match="zero_budget"):
        _select(state)


def test_completed_adjudication_cannot_claim_zero_realized_review_cost() -> None:
    selected = _select(_state())

    with pytest.raises(
        ValueError,
        match="adjudication_realized_cost_must_be_finite_positive",
    ):
        _adjudication(selected, realized_cost=0.0)


def test_corrected_selected_estimate_reruns_and_charges_only_realized_cost() -> None:
    selected = _select(_state())
    adjudication = _adjudication(selected, realized_cost=1.5)
    corrected = _correct_estimate(
        selected.state.graph, selected.candidate.item_id, value=0.7
    )

    result = _resolve(
        selected,
        adjudication,
        disposition=CorrectionDisposition.CORRECTED,
        corrected_graph=corrected,
    )

    assert result.receipt.realized_cost == 1.5
    assert result.receipt.realized_cost != selected.candidate.estimated_cost
    assert result.state.session.historical_realized_cost == 1.5
    assert result.state.session.remaining_budget == 3.5
    assert result.state.graph_sha256 == hash_canonical(corrected)
    assert result.correction.correction_artifact_sha256 == (
        result.correction_provenance.provenance_sha256
    )
    assert result.correction_provenance.external_correction_payload_sha256 == _hash(
        "correction-payload"
    )
    resumed = resume_sequential_verification_state(result.state.model_dump(mode="json"))
    assert resumed == result.state

    next_selection = select_next_audit_candidate(
        result.state,
        expected=freeze_state_expectation(result.state),
        selected_at=_CREATED_AT + timedelta(minutes=4),
    )
    assert next_selection.candidate.item_id.endswith("-b")


def test_explicit_no_change_reruns_identically_and_still_records_receipt() -> None:
    selected = _select(_state())
    adjudication = _adjudication(selected, realized_cost=0.75)

    result = _resolve(
        selected,
        adjudication,
        disposition=CorrectionDisposition.NO_CHANGE,
        corrected_graph=None,
    )

    assert result.correction.disposition is CorrectionDisposition.NO_CHANGE
    assert result.correction.pre_graph_sha256 == result.correction.post_graph_sha256
    assert result.correction.pre_synthesis_sha256 == result.correction.post_synthesis_sha256
    assert result.receipt.realized_cost == 0.75


def test_selected_invalid_estimate_may_be_removed_without_touching_other_items() -> None:
    selected = _select(_state())
    adjudication = _adjudication(selected)
    corrected = _remove_estimate(selected.state.graph, selected.candidate.item_id)

    result = _resolve(
        selected,
        adjudication,
        disposition=CorrectionDisposition.CORRECTED,
        corrected_graph=corrected,
    )

    assert selected.candidate.item_id not in {
        estimate.estimate_id for estimate in result.state.graph.outcome_estimates
    }
    assert len(result.state.candidates) == 1


def test_correction_of_unselected_estimate_is_rejected_before_callbacks() -> None:
    selected = _select(_state())
    adjudication = _adjudication(selected)
    unselected_id = next(
        estimate.estimate_id
        for estimate in selected.state.graph.outcome_estimates
        if estimate.estimate_id != selected.candidate.item_id
    )
    wrong_graph = _correct_estimate(selected.state.graph, unselected_id, value=0.9)
    calls = []

    def should_not_run(_):
        calls.append("synthesis")
        return {}

    with pytest.raises(
        SequentialVerificationContractError,
        match="unselected_or_wrong_estimate",
    ):
        resolve_selected_audit_candidate(
            selected.state,
            expected=freeze_state_expectation(selected.state),
            adjudication=adjudication,
            disposition=CorrectionDisposition.CORRECTED,
            corrected_graph=wrong_graph,
            correction_provenance="wrong item",
            correction_protocol_sha256=_hash("correction-protocol"),
            external_correction_payload_sha256=_hash("wrong-correction"),
            synthesis_runner_sha256=_hash("synthesis-runner"),
            candidate_runner_sha256=_hash("candidate-runner"),
            rerun_synthesis=should_not_run,
            rerun_candidates=_candidates,
        )
    assert calls == []


def _changed_expectation(expectation, field: str) -> SequentialStateExpectation:
    payload = expectation.model_dump(mode="json", exclude={"expectation_sha256"})
    payload[field] = _hash(f"stale-{field}")
    return SequentialStateExpectation.model_validate(
        {**payload, "expectation_sha256": hash_canonical(payload)}
    )


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("session_sha256", "stale_sequential_session_hash"),
        ("graph_sha256", "stale_sequential_graph_hash"),
        ("candidate_input_sha256", "stale_sequential_candidate_input_hash"),
    ],
)
def test_selection_rejects_stale_session_graph_and_candidate_hashes(
    field: str, error: str
) -> None:
    state = _state()
    expectation = _changed_expectation(freeze_state_expectation(state), field)

    with pytest.raises(StaleSequentialVerificationStateError, match=error):
        select_next_audit_candidate(
            state,
            expected=expectation,
            selected_at=_CREATED_AT + timedelta(minutes=1),
        )


def test_realized_cost_over_budget_is_rejected() -> None:
    selected = _select(_state())

    with pytest.raises(
        SequentialVerificationContractError,
        match="realized_cost_exceeds_budget",
    ):
        _adjudication(selected, realized_cost=5.01)


def test_no_change_rejects_nondeterministic_candidate_rerun() -> None:
    selected = _select(_state())
    adjudication = _adjudication(selected)

    def changed_candidates(graph, *_):
        candidates = _candidates(graph)
        first = candidates[0]
        candidates[0] = freeze_current_audit_candidate(
            item_id=first.item_id,
            priority=0.1,
            estimated_cost=first.estimated_cost,
            cost_unit=first.cost_unit,
            scientific_candidate_sha256=first.scientific_candidate_sha256,
            counterfactual_synthesis_sha256=first.counterfactual_synthesis_sha256,
        )
        return candidates

    with pytest.raises(
        SequentialVerificationContractError,
        match="no_change_rerun_state_changed",
    ):
        resolve_selected_audit_candidate(
            selected.state,
            expected=freeze_state_expectation(selected.state),
            adjudication=adjudication,
            disposition=CorrectionDisposition.NO_CHANGE,
            corrected_graph=None,
            correction_provenance="no change",
            correction_protocol_sha256=_hash("correction-protocol"),
            external_correction_payload_sha256=_hash("no-change"),
            synthesis_runner_sha256=_hash("synthesis-runner"),
            candidate_runner_sha256=_hash("candidate-runner"),
            rerun_synthesis=_synthesis,
            rerun_candidates=changed_candidates,
        )


def _resolved_first_selection(*, adaptive: bool):
    selected = (
        _select_adaptive(
            _state(adaptive=True),
            selected_at=_CREATED_AT + timedelta(minutes=1),
        )
        if adaptive
        else _select(_state())
    )
    adjudication = _adjudication(selected, realized_cost=1.5)
    return _resolve(
        selected,
        adjudication,
        disposition=CorrectionDisposition.NO_CHANGE,
        corrected_graph=None,
    )


def _two_step_adaptive_selection():
    first_resolution = _resolved_first_selection(adaptive=True)
    return _select_adaptive(
        first_resolution.state,
        selected_at=_CREATED_AT + timedelta(minutes=4),
    )


def test_adaptive_preselection_history_is_append_only_across_two_steps() -> None:
    second = _two_step_adaptive_selection()

    checkpoints, context_sha256, bundle_sha256 = (
        adaptive_preselection_history_from_state(second.state)
    )

    assert [checkpoint.prefix_index for checkpoint in checkpoints] == [0, 1]
    assert checkpoints[0].audit_prefix_item_ids == []
    assert checkpoints[0].audit_prefix_cost_minutes == 0.0
    assert checkpoints[1].audit_prefix_item_ids == list(
        second.state.session.resolved_item_ids
    )
    assert checkpoints[1].audit_prefix_cost_minutes == 1.5
    assert context_sha256 == _hash("adaptive-context")
    assert bundle_sha256 == _hash("adaptive-bundle")
    assert second.state.transitions[-1].adaptive_preselection_state == checkpoints[-1]


def test_adaptive_preselection_history_cannot_activate_after_nonadaptive_selection() -> None:
    resolved = _resolved_first_selection(adaptive=False)

    with pytest.raises(
        SequentialVerificationContractError,
        match="adaptive_selection_cannot_activate_after_state_genesis",
    ):
        _select_adaptive(
            resolved.state,
            selected_at=_CREATED_AT + timedelta(minutes=4),
        )


def test_adaptive_preselection_checkpoint_cannot_be_removed_after_activation() -> None:
    resolved = _resolved_first_selection(adaptive=True)

    with pytest.raises(
        SequentialVerificationContractError,
        match="adaptive_selection_history_checkpoint_removed",
    ):
        select_next_audit_candidate(
            resolved.state,
            expected=freeze_state_expectation(resolved.state),
            selected_at=_CREATED_AT + timedelta(minutes=4),
        )


@pytest.mark.parametrize(
    ("context_sha256", "bundle_sha256", "error"),
    [
        (
            _hash("changed-context"),
            _hash("adaptive-bundle"),
            "adaptive_selection_policy_context_changed",
        ),
        (
            _hash("adaptive-context"),
            _hash("changed-bundle"),
            "adaptive_selection_calibration_bundle_changed",
        ),
    ],
)
def test_adaptive_context_and_calibration_bundle_cannot_switch_between_steps(
    context_sha256: str,
    bundle_sha256: str,
    error: str,
) -> None:
    resolved = _resolved_first_selection(adaptive=True)

    with pytest.raises(SequentialVerificationContractError, match=error):
        _select_adaptive(
            resolved.state,
            selected_at=_CREATED_AT + timedelta(minutes=4),
            context_sha256=context_sha256,
            bundle_sha256=bundle_sha256,
        )


def _forge_last_adaptive_checkpoint(
    state: SequentialVerificationState,
    *,
    field: str,
    value: object,
) -> dict[str, object]:
    payload = state.model_dump(mode="json", exclude={"state_sha256"})
    transition = next(
        transition
        for transition in reversed(payload["transitions"])
        if transition["transition_kind"] == "selection"
    )
    checkpoint = transition["adaptive_preselection_state"]
    checkpoint[field] = value
    checkpoint_payload = {
        key: item for key, item in checkpoint.items() if key != "state_sha256"
    }
    checkpoint["state_sha256"] = hash_canonical(checkpoint_payload)
    transition_payload = {
        key: item for key, item in transition.items() if key != "transition_sha256"
    }
    transition["transition_sha256"] = hash_canonical(transition_payload)
    return {**payload, "state_sha256": hash_canonical(payload)}


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        (
            "scheduler_state_sha256",
            _hash("forged-scheduler-state"),
            "adaptive_selection_scheduler_state_hash_mismatch",
        ),
        (
            "audit_prefix_item_ids",
            ["forged-resolved-item"],
            "adaptive_selection_resolved_prefix_identity_mismatch",
        ),
        (
            "audit_prefix_cost_minutes",
            2.5,
            "adaptive_selection_resolved_prefix_cost_mismatch",
        ),
        (
            "evidence_graph_sha256",
            _hash("forged-graph"),
            "adaptive_selection_evidence_graph_hash_mismatch",
        ),
        (
            "synthesis_sha256",
            _hash("forged-synthesis"),
            "adaptive_selection_synthesis_hash_mismatch",
        ),
    ],
)
def test_replay_rejects_rehashed_forged_adaptive_preselection_details(
    field: str,
    value: object,
    error: str,
) -> None:
    selected = _two_step_adaptive_selection()
    forged = _forge_last_adaptive_checkpoint(
        selected.state,
        field=field,
        value=value,
    )

    with pytest.raises(ValueError, match=error):
        SequentialVerificationState.model_validate(forged)
