from __future__ import annotations

import pytest

from literature_multiverse.budgeted_verification import (
    AllocationPolicy,
    AuditCandidate,
    AuditOracle,
    ClaimModel,
    ProbabilityBasis,
    ReleaseGuardConfig,
    ReleaseGuardStatus,
    ScenarioKind,
    assess_prospective_release_guard,
    evaluate_fixed_budgets,
    rank_candidates,
    select_under_budget,
)


def _candidate(
    item_id: str,
    *,
    baseline: float,
    counterfactual: float,
    risk: float,
    cost: float,
    disagreement: float = 0.0,
) -> AuditCandidate:
    return AuditCandidate(
        item_id=item_id,
        baseline_contribution=baseline,
        counterfactual_contribution=counterfactual,
        error_probability=risk,
        probability_basis=ProbabilityBasis.HEURISTIC,
        probability_source="unit-test-heuristic",
        verification_cost=cost,
        cost_unit="minutes",
        disagreement_score=disagreement,
        scenario_kind=ScenarioKind.CANDIDATE_CORRECTION,
        scenario_source="unit-test-counterfactual",
    )


def test_claim_model_is_numerically_stable() -> None:
    model = ClaimModel(intercept=0.0)

    assert model.probability([1000.0]) == 1.0
    assert model.probability([-1000.0]) == 0.0


def test_leave_one_out_requires_zero_contribution() -> None:
    with pytest.raises(ValueError, match="leave_one_out_contribution_must_be_zero"):
        AuditCandidate(
            item_id="x",
            baseline_contribution=1.0,
            counterfactual_contribution=0.2,
            error_probability=0.5,
            probability_basis=ProbabilityBasis.HEURISTIC,
            probability_source="test",
            verification_cost=1.0,
            cost_unit="minutes",
            disagreement_score=0.2,
            scenario_kind=ScenarioKind.LEAVE_ONE_OUT,
            scenario_source="test",
        )


def test_expected_loss_policy_uses_risk_influence_and_cost() -> None:
    candidates = [
        _candidate("large", baseline=1.2, counterfactual=0.0, risk=0.8, cost=2.0),
        _candidate("cheap", baseline=0.3, counterfactual=0.0, risk=0.7, cost=0.2),
        _candidate("risky", baseline=0.1, counterfactual=0.0, risk=0.95, cost=1.0),
    ]
    model = ClaimModel(intercept=-0.5)

    ranking = rank_candidates(
        candidates,
        model,
        AllocationPolicy.EXPECTED_CLAIM_LOSS_PER_COST,
    )

    assert ranking[0].item_id == "cheap"
    for row in ranking:
        assert row.expected_claim_loss_reduction == pytest.approx(
            row.error_probability * row.probability_influence
        )
        assert row.expected_claim_loss_reduction_per_cost == pytest.approx(
            row.expected_claim_loss_reduction / row.verification_cost
        )


def test_policy_orderings_are_distinct_and_auditable() -> None:
    candidates = [
        _candidate(
            "high-risk",
            baseline=0.1,
            counterfactual=0.0,
            risk=0.95,
            cost=1.0,
            disagreement=0.1,
        ),
        _candidate(
            "high-disagreement",
            baseline=0.2,
            counterfactual=0.0,
            risk=0.2,
            cost=1.0,
            disagreement=0.99,
        ),
        _candidate(
            "high-influence",
            baseline=1.0,
            counterfactual=0.0,
            risk=0.3,
            cost=1.0,
            disagreement=0.2,
        ),
    ]
    model = ClaimModel(intercept=-0.4)

    assert rank_candidates(candidates, model, AllocationPolicy.RISK_ONLY)[0].item_id == (
        "high-risk"
    )
    assert rank_candidates(
        candidates, model, AllocationPolicy.DISAGREEMENT
    )[0].item_id == "high-disagreement"
    assert rank_candidates(
        candidates, model, AllocationPolicy.INFLUENCE_ONLY
    )[0].item_id == "high-influence"


def test_component_ablation_priorities_match_declared_formulae() -> None:
    candidates = [
        _candidate("a", baseline=1.0, counterfactual=0.0, risk=0.8, cost=2.0),
        _candidate("b", baseline=0.2, counterfactual=0.0, risk=0.3, cost=0.5),
    ]
    model = ClaimModel(intercept=-0.4)
    rows = {
        policy: {
            row.item_id: row
            for row in rank_candidates(candidates, model, policy)
        }
        for policy in (
            AllocationPolicy.COST_ONLY,
            AllocationPolicy.RISK_X_INFLUENCE,
            AllocationPolicy.RISK_PER_COST,
            AllocationPolicy.INFLUENCE_PER_COST,
        )
    }

    for item_id, candidate in zip(("a", "b"), candidates, strict=True):
        risk_influence = rows[AllocationPolicy.RISK_X_INFLUENCE][item_id]
        risk_cost = rows[AllocationPolicy.RISK_PER_COST][item_id]
        influence_cost = rows[AllocationPolicy.INFLUENCE_PER_COST][item_id]
        cost_only = rows[AllocationPolicy.COST_ONLY][item_id]
        assert risk_influence.priority == pytest.approx(
            risk_influence.error_probability * risk_influence.probability_influence
        )
        assert risk_cost.priority == pytest.approx(
            candidate.error_probability / candidate.verification_cost
        )
        assert influence_cost.priority == pytest.approx(
            influence_cost.probability_influence / candidate.verification_cost
        )
        assert cost_only.priority == pytest.approx(1.0 / candidate.verification_cost)


def test_random_ranking_is_seeded_and_independent_of_input_order() -> None:
    candidates = [
        _candidate(str(index), baseline=0.1, counterfactual=0.0, risk=0.2, cost=1.0)
        for index in range(8)
    ]
    model = ClaimModel(intercept=0.0)

    first = rank_candidates(candidates, model, AllocationPolicy.RANDOM, seed=17)
    reordered = rank_candidates(
        list(reversed(candidates)), model, AllocationPolicy.RANDOM, seed=17
    )
    different = rank_candidates(candidates, model, AllocationPolicy.RANDOM, seed=18)

    assert [row.item_id for row in first] == [row.item_id for row in reordered]
    assert [row.item_id for row in first] != [row.item_id for row in different]


def test_selection_skips_nonfitting_item_and_honors_budget() -> None:
    candidates = [
        _candidate("expensive", baseline=1.0, counterfactual=0.0, risk=1.0, cost=3.0),
        _candidate("fits", baseline=0.2, counterfactual=0.0, risk=0.5, cost=1.0),
    ]
    model = ClaimModel(intercept=0.0)

    selection = select_under_budget(
        candidates,
        model,
        AllocationPolicy.INFLUENCE_ONLY,
        budget=1.5,
    )

    assert selection.selected_item_ids == ("fits",)
    assert selection.spent == 1.0


def test_evaluation_applies_only_selected_oracle_confirmed_corrections() -> None:
    candidates = [
        _candidate("error", baseline=1.0, counterfactual=0.0, risk=0.9, cost=1.0),
        _candidate("correct", baseline=0.6, counterfactual=0.0, risk=0.8, cost=1.0),
    ]
    oracles = [
        AuditOracle("error", True, 0.0, "adjudication"),
        AuditOracle("correct", False, 99.0, "adjudication"),
    ]
    model = ClaimModel(intercept=-0.8)

    rows = evaluate_fixed_budgets(
        candidates,
        oracles,
        model,
        budgets=[0.0, 1.0],
        policies=[AllocationPolicy.RISK_ONLY],
    )

    assert rows[0]["audited_claim_probability"] == rows[0]["baseline_claim_probability"]
    assert rows[1]["selected_item_ids"] == ["error"]
    assert rows[1]["errors_found"] == 1
    assert rows[1]["audited_claim_probability"] == rows[1]["oracle_claim_probability"]
    assert rows[1]["claim_loss_recovery_fraction"] == pytest.approx(1.0)


def test_evaluation_rejects_oracle_identity_mismatch() -> None:
    candidate = _candidate(
        "candidate", baseline=0.1, counterfactual=0.0, risk=0.2, cost=1.0
    )
    oracle = AuditOracle("different", False, 0.0, "adjudication")

    with pytest.raises(ValueError, match="audit_oracle_identity_mismatch"):
        evaluate_fixed_budgets(
            [candidate],
            [oracle],
            ClaimModel(intercept=0.0),
            budgets=[1.0],
        )


def test_mixed_cost_units_are_rejected() -> None:
    first = _candidate("first", baseline=0.1, counterfactual=0.0, risk=0.2, cost=1.0)
    second = AuditCandidate(
        item_id="second",
        baseline_contribution=0.1,
        counterfactual_contribution=0.0,
        error_probability=0.2,
        probability_basis=ProbabilityBasis.HEURISTIC,
        probability_source="test",
        verification_cost=1.0,
        cost_unit="dollars",
        disagreement_score=0.2,
        scenario_kind=ScenarioKind.CANDIDATE_CORRECTION,
        scenario_source="test",
    )

    with pytest.raises(ValueError, match="audit_candidate_cost_units_mixed"):
        rank_candidates([first, second], ClaimModel(0.0), AllocationPolicy.RISK_ONLY)


def test_release_guard_blocks_unresolved_high_influence_heuristic() -> None:
    candidate = _candidate(
        "material", baseline=1.0, counterfactual=0.0, risk=0.8, cost=1.0
    )

    decision = assess_prospective_release_guard(
        [candidate],
        ClaimModel(intercept=-0.4),
        resolved_item_ids=[],
    )

    assert decision.status is ReleaseGuardStatus.BLOCKED
    assert decision.unresolved_conclusion_flip_item_ids == ("material",)
    assert decision.unresolved_high_influence_item_ids == ("material",)
    assert decision.unresolved_noncalibrated_item_ids == ("material",)
    assert "unresolved_counterfactual_can_flip_conclusion" in decision.reasons
    assert "unresolved_error_probabilities_not_calibrated" in decision.reasons


def test_release_guard_treats_selected_but_unadjudicated_as_unresolved() -> None:
    candidate = _candidate(
        "assigned", baseline=1.0, counterfactual=0.0, risk=0.8, cost=1.0
    )
    selection = select_under_budget(
        [candidate],
        ClaimModel(intercept=-0.4),
        AllocationPolicy.EXPECTED_CLAIM_LOSS_PER_COST,
        budget=1.0,
    )

    decision = assess_prospective_release_guard(
        [candidate],
        ClaimModel(intercept=-0.4),
        resolved_item_ids=[],
    )

    assert selection.selected_item_ids == ("assigned",)
    assert decision.unresolved_item_ids == ("assigned",)
    assert decision.status is ReleaseGuardStatus.BLOCKED


def test_release_guard_all_resolved_is_only_eligible_for_downstream_gates() -> None:
    candidate = _candidate(
        "resolved", baseline=0.1, counterfactual=0.0, risk=0.2, cost=1.0
    )

    decision = assess_prospective_release_guard(
        [candidate],
        ClaimModel(intercept=0.0),
        resolved_item_ids=["resolved"],
        config=ReleaseGuardConfig(),
    )

    assert decision.status is ReleaseGuardStatus.ELIGIBLE_FOR_DOWNSTREAM_GATES
    assert decision.reasons == ()
    assert decision.unresolved_item_ids == ()
