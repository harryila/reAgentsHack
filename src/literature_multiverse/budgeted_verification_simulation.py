"""Planted method check for budgeted human-verification allocation.

The simulation creates candidate correction scenarios, planted item errors, and a
known fully corrected corpus-level claim.  It evaluates allocation mechanics only;
it provides no estimate of real extraction error probabilities or human-audit cost.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

from literature_multiverse.budgeted_verification import (
    AllocationPolicy,
    AuditCandidate,
    AuditOracle,
    ClaimModel,
    ProbabilityBasis,
    ScenarioKind,
    evaluate_fixed_budgets,
)

UNCERTAINTY_CONFIDENCE_LEVEL = 0.95
PAIRED_CONTRAST_BUDGET = 5.0
PAIRED_BINARY_BOOTSTRAP_DRAWS = 20_000
PAIRED_BINARY_BOOTSTRAP_BASE_SEED = 20260827
_NORMAL_975 = 1.959963984540054


@dataclass(frozen=True, slots=True)
class SimulatedAuditCorpus:
    candidates: tuple[AuditCandidate, ...]
    oracles: tuple[AuditOracle, ...]
    claim_model: ClaimModel


def _student_t_mean_interval(values: Sequence[float]) -> list[float | None]:
    """Two-sided interval for a replicate-level arithmetic mean.

    The interval is intentionally unbounded: clipping a t interval for a bounded
    metric changes its coverage without making that change visible.  With one
    replicate, variance is unidentified and the interval is recorded as unavailable.
    """

    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size == 0 or not np.all(np.isfinite(sample)):
        raise ValueError("budgeted_simulation_interval_sample_invalid")
    if sample.size < 2:
        return [None, None]
    mean = float(np.mean(sample))
    standard_error = float(np.std(sample, ddof=1) / math.sqrt(sample.size))
    if standard_error == 0.0:
        return [mean, mean]
    critical = float(
        student_t.ppf(
            0.5 + UNCERTAINTY_CONFIDENCE_LEVEL / 2.0,
            df=sample.size - 1,
        )
    )
    half_width = critical * standard_error
    return [mean - half_width, mean + half_width]


def _wilson_interval(successes: int, total: int) -> list[float | None]:
    """Two-sided Wilson score interval for one binary replicate-level rate."""

    if total < 0 or successes < 0 or successes > total:
        raise ValueError("budgeted_simulation_wilson_counts_invalid")
    if total == 0:
        return [None, None]
    proportion = successes / total
    denominator = 1.0 + _NORMAL_975**2 / total
    center = (proportion + _NORMAL_975**2 / (2.0 * total)) / denominator
    half_width = (
        _NORMAL_975
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + _NORMAL_975**2 / (4.0 * total**2)
        )
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _contrast_bootstrap_seed(*, comparator: str, budget: float) -> int:
    """Derive an order-independent PCG64 seed from the frozen base seed."""

    identity = (
        "budgeted-verification-paired-bootstrap-v1\0"
        f"{PAIRED_BINARY_BOOTSTRAP_BASE_SEED}\0{budget:.17g}\0{comparator}"
    )
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")


def _paired_binary_bootstrap_interval(
    differences: Sequence[float],
    *,
    seed: int,
) -> list[float]:
    """Fixed-seed percentile interval for a paired binary-rate difference."""

    sample = np.asarray(differences, dtype=float)
    if sample.ndim != 1 or sample.size == 0 or not np.all(np.isfinite(sample)):
        raise ValueError("budgeted_simulation_paired_bootstrap_sample_invalid")
    if not np.all(np.isin(sample, (-1.0, 0.0, 1.0))):
        raise ValueError("budgeted_simulation_paired_binary_difference_invalid")
    generator = np.random.Generator(np.random.PCG64(seed))
    bootstrap_means = np.empty(PAIRED_BINARY_BOOTSTRAP_DRAWS, dtype=float)
    batch_size = 1_000
    for start in range(0, PAIRED_BINARY_BOOTSTRAP_DRAWS, batch_size):
        stop = min(start + batch_size, PAIRED_BINARY_BOOTSTRAP_DRAWS)
        indices = generator.integers(
            0,
            sample.size,
            size=(stop - start, sample.size),
        )
        bootstrap_means[start:stop] = np.mean(sample[indices], axis=1)
    alpha = 1.0 - UNCERTAINTY_CONFIDENCE_LEVEL
    interval = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return [float(interval[0]), float(interval[1])]


def _paired_contrast_summary(
    proposed_rows: Sequence[Mapping[str, object]],
    comparator_rows: Sequence[Mapping[str, object]],
    *,
    comparator: str,
    budget: float,
) -> dict[str, object]:
    if len(proposed_rows) != len(comparator_rows) or not proposed_rows:
        raise ValueError("budgeted_simulation_paired_contrast_rows_invalid")

    paired_recovery = [
        (
            float(proposed["claim_loss_recovery_fraction"]),
            float(baseline["claim_loss_recovery_fraction"]),
        )
        for proposed, baseline in zip(proposed_rows, comparator_rows, strict=True)
        if proposed["claim_loss_recovery_fraction"] is not None
        and baseline["claim_loss_recovery_fraction"] is not None
    ]
    if not paired_recovery:
        raise ValueError("budgeted_simulation_paired_recovery_empty")
    recovery_differences = [proposed - baseline for proposed, baseline in paired_recovery]

    proposed_repairs = np.asarray(
        [bool(row["claim_repaired"]) for row in proposed_rows], dtype=int
    )
    comparator_repairs = np.asarray(
        [bool(row["claim_repaired"]) for row in comparator_rows], dtype=int
    )
    repair_differences = proposed_repairs - comparator_repairs
    proposed_only = int(np.sum((proposed_repairs == 1) & (comparator_repairs == 0)))
    comparator_only = int(np.sum((proposed_repairs == 0) & (comparator_repairs == 1)))
    both_repaired = int(np.sum((proposed_repairs == 1) & (comparator_repairs == 1)))
    neither_repaired = int(np.sum((proposed_repairs == 0) & (comparator_repairs == 0)))
    bootstrap_seed = _contrast_bootstrap_seed(comparator=comparator, budget=budget)

    return {
        "comparator_policy": comparator,
        "budget": float(budget),
        "paired_replicates": len(proposed_rows),
        "claim_loss_recovery_fraction": {
            "paired_nonmissing_replicates": len(paired_recovery),
            "proposed_mean": float(np.mean([row[0] for row in paired_recovery])),
            "comparator_mean": float(np.mean([row[1] for row in paired_recovery])),
            "proposed_minus_comparator_mean_difference": float(
                np.mean(recovery_differences)
            ),
            "confidence_interval_95": _student_t_mean_interval(recovery_differences),
            "interval_method": "two_sided_paired_student_t",
        },
        "claim_repair_rate": {
            "paired_replicates": len(repair_differences),
            "proposed_repaired_count": int(np.sum(proposed_repairs)),
            "comparator_repaired_count": int(np.sum(comparator_repairs)),
            "proposed_rate": float(np.mean(proposed_repairs)),
            "comparator_rate": float(np.mean(comparator_repairs)),
            "proposed_minus_comparator_rate_difference": float(
                np.mean(repair_differences)
            ),
            "confidence_interval_95": _paired_binary_bootstrap_interval(
                repair_differences,
                seed=bootstrap_seed,
            ),
            "interval_method": "paired_nonparametric_percentile_bootstrap",
            "bootstrap_draws": PAIRED_BINARY_BOOTSTRAP_DRAWS,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_bit_generator": "PCG64",
            "bootstrap_quantile_method": "linear",
            "paired_outcome_counts": {
                "proposed_only_repaired": proposed_only,
                "comparator_only_repaired": comparator_only,
                "both_repaired": both_repaired,
                "neither_repaired": neither_repaired,
            },
        },
    }


def simulate_audit_corpus(*, seed: int, item_count: int = 60) -> SimulatedAuditCorpus:
    """Generate one monotone planted correction problem.

    Each hypothetical correction reduces a positive distortion.  It is applied by
    the retrospective evaluator only when the planted oracle says the item is
    actually erroneous.  The policy sees the planted generative error probability,
    not the realized error label.
    """

    if item_count < 8:
        raise ValueError("budgeted_simulation_requires_eight_items")
    rng = np.random.default_rng(seed)
    latent_error_probabilities = np.clip(rng.beta(1.5, 4.5, item_count), 0.02, 0.85)
    realized_errors = rng.random(item_count) < latent_error_probabilities
    distortion_sizes = np.clip(rng.lognormal(-2.0, 0.65, item_count), 0.025, 0.65)
    verification_costs = np.clip(rng.lognormal(0.25, 0.45, item_count), 0.5, 4.0)
    disagreement_scores = np.clip(
        0.70 * latent_error_probabilities + rng.normal(0.10, 0.13, item_count),
        0.0,
        1.0,
    )
    true_contributions = rng.normal(0.0, 0.018, item_count)
    baseline_contributions = true_contributions + realized_errors * distortion_sizes
    # Center the observed claim just above the release threshold.  Removing enough
    # planted positive distortions can therefore repair the false-positive decision,
    # while the magnitude of each candidate's influence remains heterogeneous.
    target_baseline_score = float(rng.normal(0.25, 0.05))
    intercept = target_baseline_score - float(np.sum(baseline_contributions))
    claim_model = ClaimModel(
        claim_id=f"planted-claim-{seed}",
        intercept=intercept,
        temperature=1.0,
        decision_threshold=0.5,
    )
    candidates: list[AuditCandidate] = []
    oracles: list[AuditOracle] = []
    for index in range(item_count):
        item_id = f"sim-{seed}-item-{index:04d}"
        baseline = float(baseline_contributions[index])
        candidate_correction = baseline - float(distortion_sizes[index])
        candidates.append(
            AuditCandidate(
                item_id=item_id,
                baseline_contribution=baseline,
                counterfactual_contribution=candidate_correction,
                error_probability=float(latent_error_probabilities[index]),
                probability_basis=ProbabilityBasis.PLANTED_SIMULATION,
                probability_source="known_planted_generative_probability",
                verification_cost=float(verification_costs[index]),
                cost_unit="simulated_human_minutes",
                disagreement_score=float(disagreement_scores[index]),
                scenario_kind=ScenarioKind.CANDIDATE_CORRECTION,
                scenario_source="planted_candidate_correction_scenario",
            )
        )
        oracles.append(
            AuditOracle(
                item_id=item_id,
                is_error=bool(realized_errors[index]),
                corrected_contribution=float(true_contributions[index]),
                label_source="planted_simulation_oracle",
            )
        )
    return SimulatedAuditCorpus(tuple(candidates), tuple(oracles), claim_model)


def simulate_budgeted_verification_replicate(
    *,
    seed: int,
    item_count: int = 60,
    budgets: Sequence[float] = (5.0, 10.0, 20.0, 40.0),
) -> dict[str, object]:
    """Run the full component-ablation policy family on one planted corpus."""

    corpus = simulate_audit_corpus(seed=seed, item_count=item_count)
    detailed_evaluations = evaluate_fixed_budgets(
        corpus.candidates,
        corpus.oracles,
        corpus.claim_model,
        budgets=budgets,
        policies=tuple(AllocationPolicy),
        seed=seed,
    )
    # The public evaluator returns the complete audit trail.  The frozen paper
    # artifact retains only the fields needed to reproduce its aggregate curves;
    # candidates and detailed selections are deterministic from the recorded seed.
    artifact_fields = (
        "policy",
        "budget",
        "spent",
        "selected",
        "errors_found",
        "claim_loss_reduction",
        "claim_loss_recovery_fraction",
        "audited_conclusion_correct",
        "claim_repaired",
    )
    evaluations = [
        {key: row[key] for key in artifact_fields}
        for row in detailed_evaluations
    ]
    first = detailed_evaluations[0]
    return {
        "seed": seed,
        "item_count": item_count,
        "planted_errors": first["total_errors"],
        "baseline_claim_probability": first["baseline_claim_probability"],
        "oracle_claim_probability": first["oracle_claim_probability"],
        "baseline_conclusion": first["baseline_conclusion"],
        "oracle_conclusion": first["oracle_conclusion"],
        "evaluations": evaluations,
    }


def summarize_budgeted_verification_simulations(
    replicates: Sequence[Mapping[str, object]],
    *,
    budgets: Sequence[float],
) -> dict[str, object]:
    """Aggregate fixed-budget outcomes without making real-world claims."""

    if not replicates:
        raise ValueError("budgeted_simulation_replicates_empty")
    expected_keys = {
        (policy.value, float(budget))
        for policy in AllocationPolicy
        for budget in budgets
    }
    grouped: dict[tuple[str, float], list[Mapping[str, object]]] = {
        key: [] for key in expected_keys
    }
    for replicate in replicates:
        evaluations = replicate.get("evaluations")
        if not isinstance(evaluations, list):
            raise ValueError("budgeted_simulation_evaluations_invalid")
        seen: set[tuple[str, float]] = set()
        for row in evaluations:
            if not isinstance(row, Mapping):
                raise ValueError("budgeted_simulation_evaluation_row_invalid")
            key = (str(row["policy"]), float(row["budget"]))
            if key not in expected_keys or key in seen:
                raise ValueError("budgeted_simulation_policy_budget_grid_invalid")
            grouped[key].append(row)
            seen.add(key)
        if seen != expected_keys:
            raise ValueError("budgeted_simulation_policy_budget_grid_incomplete")

    policy_summaries: dict[str, dict[str, object]] = {}
    for policy in AllocationPolicy:
        budget_summaries: dict[str, object] = {}
        for budget in budgets:
            rows = grouped[(policy.value, float(budget))]
            spent = [float(row["spent"]) for row in rows]
            selected = [float(row["selected"]) for row in rows]
            errors_found = [float(row["errors_found"]) for row in rows]
            loss_reduction = [float(row["claim_loss_reduction"]) for row in rows]
            recovery = [
                float(row["claim_loss_recovery_fraction"])
                for row in rows
                if row["claim_loss_recovery_fraction"] is not None
            ]
            audited_correct = [bool(row["audited_conclusion_correct"]) for row in rows]
            repaired = [bool(row["claim_repaired"]) for row in rows]
            audited_correct_count = sum(audited_correct)
            repaired_count = sum(repaired)
            budget_summaries[str(float(budget))] = {
                "replicates": len(rows),
                "mean_spent": float(np.mean(spent)),
                "mean_spent_ci_95": _student_t_mean_interval(spent),
                "mean_selected": float(np.mean(selected)),
                "mean_selected_ci_95": _student_t_mean_interval(selected),
                "mean_errors_found": float(np.mean(errors_found)),
                "mean_errors_found_ci_95": _student_t_mean_interval(errors_found),
                "mean_claim_loss_reduction": float(np.mean(loss_reduction)),
                "mean_claim_loss_reduction_ci_95": _student_t_mean_interval(
                    loss_reduction
                ),
                "claim_loss_recovery_fraction_nonmissing_replicates": len(recovery),
                "mean_claim_loss_recovery_fraction": (
                    float(np.mean(recovery)) if recovery else None
                ),
                "mean_claim_loss_recovery_fraction_ci_95": (
                    _student_t_mean_interval(recovery) if recovery else [None, None]
                ),
                "audited_conclusion_correct_count": audited_correct_count,
                "audited_conclusion_accuracy": float(np.mean(audited_correct)),
                "audited_conclusion_accuracy_ci_95": _wilson_interval(
                    audited_correct_count, len(rows)
                ),
                "claim_repaired_count": repaired_count,
                "claim_repair_rate": float(np.mean(repaired)),
                "claim_repair_rate_ci_95": _wilson_interval(repaired_count, len(rows)),
            }
        policy_summaries[policy.value] = {"budgets": budget_summaries}

    paired_contrasts: dict[str, object] = {}
    if float(PAIRED_CONTRAST_BUDGET) in {float(budget) for budget in budgets}:
        proposed_policy = AllocationPolicy.EXPECTED_CLAIM_LOSS_PER_COST.value
        proposed_rows = grouped[(proposed_policy, PAIRED_CONTRAST_BUDGET)]
        paired_contrasts[str(PAIRED_CONTRAST_BUDGET)] = {
            "proposed_policy": proposed_policy,
            "comparators": {
                comparator.value: _paired_contrast_summary(
                    proposed_rows,
                    grouped[(comparator.value, PAIRED_CONTRAST_BUDGET)],
                    comparator=comparator.value,
                    budget=PAIRED_CONTRAST_BUDGET,
                )
                for comparator in (
                    AllocationPolicy.RISK_X_INFLUENCE,
                    AllocationPolicy.RISK_PER_COST,
                    AllocationPolicy.INFLUENCE_PER_COST,
                    AllocationPolicy.RISK_ONLY,
                    AllocationPolicy.RANDOM,
                )
            },
        }

    return {
        "budgeted_verification_simulation_summary_version": "3",
        "replicates": len(replicates),
        "budgets": [float(budget) for budget in budgets],
        "policies": policy_summaries,
        "uncertainty": {
            "confidence_level": UNCERTAINTY_CONFIDENCE_LEVEL,
            "sampling_unit": "independent_planted_corpus_replicate",
            "continuous_aggregate_mean_interval": {
                "method": "two_sided_student_t",
                "variance": "sample_variance_ddof_1",
                "unbounded_interval": True,
            },
            "binary_aggregate_rate_interval": {
                "method": "two_sided_wilson_score",
                "z": _NORMAL_975,
            },
            "paired_continuous_contrast_interval": {
                "method": "two_sided_paired_student_t",
                "pairing": "same_planted_corpus_replicate",
            },
            "paired_binary_contrast_interval": {
                "method": "paired_nonparametric_percentile_bootstrap",
                "pairing": "same_planted_corpus_replicate",
                "draws": PAIRED_BINARY_BOOTSTRAP_DRAWS,
                "base_seed": PAIRED_BINARY_BOOTSTRAP_BASE_SEED,
                "bit_generator": "PCG64",
                "quantile_method": "linear",
            },
        },
        "paired_contrasts": paired_contrasts,
        "interpretation": (
            "planted monotone correction simulation validates ranking and fixed-budget "
            "evaluation mechanics only; error probabilities, correction scenarios, "
            "costs, and audit labels are simulated and do not establish real-world "
            "verification performance"
        ),
    }


__all__ = [
    "PAIRED_BINARY_BOOTSTRAP_BASE_SEED",
    "PAIRED_BINARY_BOOTSTRAP_DRAWS",
    "PAIRED_CONTRAST_BUDGET",
    "UNCERTAINTY_CONFIDENCE_LEVEL",
    "SimulatedAuditCorpus",
    "simulate_audit_corpus",
    "simulate_budgeted_verification_replicate",
    "summarize_budgeted_verification_simulations",
]
