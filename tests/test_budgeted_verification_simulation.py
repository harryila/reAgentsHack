from __future__ import annotations

import pytest

from literature_multiverse.budgeted_verification import AllocationPolicy
from literature_multiverse.budgeted_verification_simulation import (
    PAIRED_BINARY_BOOTSTRAP_DRAWS,
    simulate_audit_corpus,
    simulate_budgeted_verification_replicate,
    summarize_budgeted_verification_simulations,
)


def test_planted_corpus_is_deterministic_and_separates_policy_from_oracle() -> None:
    first = simulate_audit_corpus(seed=31, item_count=20)
    second = simulate_audit_corpus(seed=31, item_count=20)

    assert first == second
    assert all(
        candidate.item_id == oracle.item_id
        for candidate, oracle in zip(first.candidates, first.oracles, strict=True)
    )
    assert any(oracle.is_error for oracle in first.oracles)
    assert any(not oracle.is_error for oracle in first.oracles)


def test_replicate_contains_complete_policy_budget_grid() -> None:
    budgets = (3.0, 7.0)
    replicate = simulate_budgeted_verification_replicate(
        seed=71, item_count=24, budgets=budgets
    )

    evaluations = replicate["evaluations"]
    assert len(evaluations) == len(AllocationPolicy) * len(budgets)
    assert {
        (row["policy"], row["budget"]) for row in evaluations
    } == {(policy.value, budget) for policy in AllocationPolicy for budget in budgets}


def test_expected_loss_per_cost_wins_planted_low_budget_method_check() -> None:
    budgets = (5.0, 10.0)
    replicates = [
        simulate_budgeted_verification_replicate(
            seed=1000 + index, item_count=60, budgets=budgets
        )
        for index in range(40)
    ]
    summary = summarize_budgeted_verification_simulations(replicates, budgets=budgets)
    policies = summary["policies"]

    for budget in budgets:
        key = str(budget)
        proposed = policies["risk_x_influence_per_cost"]["budgets"][key][
            "mean_claim_loss_reduction"
        ]
        comparators = [
            policies[policy.value]["budgets"][key]["mean_claim_loss_reduction"]
            for policy in AllocationPolicy
            if policy is not AllocationPolicy.EXPECTED_CLAIM_LOSS_PER_COST
        ]
        assert proposed > max(comparators)


def test_summary_records_deterministic_aggregate_uncertainty() -> None:
    budgets = (5.0,)
    replicates = [
        simulate_budgeted_verification_replicate(
            seed=2000 + index, item_count=36, budgets=budgets
        )
        for index in range(12)
    ]

    first = summarize_budgeted_verification_simulations(replicates, budgets=budgets)
    second = summarize_budgeted_verification_simulations(replicates, budgets=budgets)

    assert first == second
    assert first["budgeted_verification_simulation_summary_version"] == "3"
    assert first["uncertainty"]["sampling_unit"] == (
        "independent_planted_corpus_replicate"
    )
    proposed = first["policies"]["risk_x_influence_per_cost"]["budgets"]["5.0"]
    assert proposed["replicates"] == 12
    assert proposed["claim_loss_recovery_fraction_nonmissing_replicates"] == 12
    assert proposed["claim_repaired_count"] == round(
        proposed["claim_repair_rate"] * proposed["replicates"]
    )
    for metric in (
        "mean_spent",
        "mean_selected",
        "mean_errors_found",
        "mean_claim_loss_reduction",
        "mean_claim_loss_recovery_fraction",
    ):
        lower, upper = proposed[f"{metric}_ci_95"]
        assert lower <= proposed[metric] <= upper
    repair_lower, repair_upper = proposed["claim_repair_rate_ci_95"]
    assert repair_lower <= proposed["claim_repair_rate"] <= repair_upper


def test_budget_five_contrasts_are_paired_and_reconcile() -> None:
    budgets = (5.0,)
    replicates = [
        simulate_budgeted_verification_replicate(
            seed=3000 + index, item_count=40, budgets=budgets
        )
        for index in range(20)
    ]
    summary = summarize_budgeted_verification_simulations(replicates, budgets=budgets)
    contrasts = summary["paired_contrasts"]["5.0"]
    assert contrasts["proposed_policy"] == "risk_x_influence_per_cost"

    for comparator in (
        "risk_x_influence",
        "risk_per_cost",
        "influence_per_cost",
        "risk_only",
        "random",
    ):
        contrast = contrasts["comparators"][comparator]
        recovery = contrast["claim_loss_recovery_fraction"]
        repair = contrast["claim_repair_rate"]
        assert contrast["paired_replicates"] == len(replicates)
        assert recovery["paired_nonmissing_replicates"] == len(replicates)
        assert recovery["proposed_minus_comparator_mean_difference"] == pytest.approx(
            recovery["proposed_mean"] - recovery["comparator_mean"]
        )
        assert repair["proposed_minus_comparator_rate_difference"] == pytest.approx(
            repair["proposed_rate"] - repair["comparator_rate"]
        )
        counts = repair["paired_outcome_counts"]
        assert sum(counts.values()) == len(replicates)
        assert repair["proposed_minus_comparator_rate_difference"] == pytest.approx(
            (
                counts["proposed_only_repaired"]
                - counts["comparator_only_repaired"]
            )
            / len(replicates)
        )
        assert repair["interval_method"] == (
            "paired_nonparametric_percentile_bootstrap"
        )
        assert repair["bootstrap_draws"] == PAIRED_BINARY_BOOTSTRAP_DRAWS
