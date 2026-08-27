"""Planted simulations for selective scientific-claim release policies.

These simulations validate calibration mechanics and compare operating points.
They are intentionally tagged as simulation and cannot establish real-world risk.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from literature_multiverse.calibration import (
    RiskExample,
    calibrate_release_policy,
    evaluate_release_policy,
    fit_logistic_risk_model,
    score_examples,
)
from literature_multiverse.lineage import hash_canonical

DEFAULT_CANDIDATE_THRESHOLDS: tuple[float, ...] = (
    0.01,
    0.02,
    0.03,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
    0.30,
)


@dataclass(frozen=True, slots=True)
class SimulatedQuestion:
    example: RiskExample
    true_loss_probability: float
    paper_count: int


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def simulate_questions(
    *,
    seed: int,
    development_count: int,
    calibration_count: int,
    test_count: int,
) -> list[SimulatedQuestion]:
    """Generate independent question-corpora with known unsupported-claim risk."""

    if min(development_count, calibration_count, test_count) < 4:
        raise ValueError("each_simulation_split_requires_at_least_four_questions")
    rng = np.random.default_rng(seed)
    config = {
        "simulation_version": "risk-features-v2",
        "seed": seed,
        "development_count": development_count,
        "calibration_count": calibration_count,
        "test_count": test_count,
    }
    pipeline_sha256 = hash_canonical(config)
    rows: list[SimulatedQuestion] = []
    global_index = 0
    for split, count in (
        ("development", development_count),
        ("calibration", calibration_count),
        ("test", test_count),
    ):
        for split_index in range(count):
            paper_count = int(2 + rng.poisson(9))
            extraction_error = float(rng.beta(1.4, 8.0))
            ungrounded_fraction = float(
                np.clip(0.55 * extraction_error + rng.beta(1.2, 14.0), 0.0, 1.0)
            )
            verifier_disagreement = float(
                np.clip(0.45 * extraction_error + rng.beta(1.6, 7.0), 0.0, 1.0)
            )
            retrieval_gap = float(rng.beta(1.5, 6.5))
            bootstrap_instability = float(rng.beta(1.7, 5.0))
            heterogeneity = float(rng.beta(2.0, 2.2))
            moderator_instability = float(
                np.clip(
                    0.45 * bootstrap_instability
                    + 0.25 * heterogeneity
                    + rng.normal(0.0, 0.08),
                    0.0,
                    1.0,
                )
            )
            logit = (
                -5.3
                + 3.7 * extraction_error
                + 3.2 * ungrounded_fraction
                + 2.0 * verifier_disagreement
                + 1.7 * retrieval_gap
                + 2.2 * bootstrap_instability
                + 1.4 * moderator_instability
                + 0.8 * heterogeneity
                + 1.1 / math.sqrt(paper_count)
            )
            true_probability = _sigmoid(logit)
            unsupported = bool(rng.random() < true_probability)
            question_id = f"sim-{seed}-{split}-{split_index:05d}"
            paper_ids = [
                f"sim-paper-{seed}-{global_index:06d}-{paper_index:03d}"
                for paper_index in range(paper_count)
            ]
            example = RiskExample(
                question_id=question_id,
                split=split,
                population_id="planted-risk-simulation-v1",
                domain=("medicine", "psychology", "ecology")[global_index % 3],
                pipeline_sha256=pipeline_sha256,
                paper_ids=paper_ids,
                features={
                    "bootstrap_instability": bootstrap_instability,
                    "extraction_error": extraction_error,
                    "heterogeneity": heterogeneity,
                    "inverse_sqrt_papers": 1.0 / math.sqrt(paper_count),
                    "moderator_instability": moderator_instability,
                    "retrieval_gap": retrieval_gap,
                    "ungrounded_fraction": ungrounded_fraction,
                    "verifier_disagreement": verifier_disagreement,
                },
                unsupported_claim=unsupported,
                label_source="simulation",
            )
            rows.append(
                SimulatedQuestion(
                    example=example,
                    true_loss_probability=true_probability,
                    paper_count=paper_count,
                )
            )
            global_index += 1
    return rows


def _policy_summary(
    rows: Sequence[SimulatedQuestion], accepted: set[str]
) -> dict[str, float | int | None]:
    test = [row for row in rows if row.example.split == "test"]
    selected = [row for row in test if row.example.question_id in accepted]
    errors = sum(row.example.unsupported_claim for row in selected)
    return {
        "accepted": len(selected),
        "errors": errors,
        "coverage": len(selected) / len(test),
        "empirical_risk": errors / len(selected) if selected else None,
        "true_selective_risk": (
            sum(row.true_loss_probability for row in selected) / len(selected)
            if selected
            else None
        ),
    }


def simulate_replicate(
    *,
    seed: int,
    alpha: float = 0.10,
    delta: float = 0.05,
    development_count: int = 400,
    calibration_count: int = 2000,
    test_count: int = 2000,
    candidate_thresholds: Sequence[float] = DEFAULT_CANDIDATE_THRESHOLDS,
) -> dict[str, object]:
    """Run one complete development/calibration/test simulation replicate."""

    rows = simulate_questions(
        seed=seed,
        development_count=development_count,
        calibration_count=calibration_count,
        test_count=test_count,
    )
    examples = [row.example for row in rows]
    model = fit_logistic_risk_model(examples, seed=seed)
    policy = calibrate_release_policy(
        examples,
        model,
        alpha=alpha,
        delta=delta,
        candidate_thresholds=candidate_thresholds,
    )
    evaluation = evaluate_release_policy(examples, model, policy)
    test_examples = [example for example in examples if example.split == "test"]
    test_scores = score_examples(test_examples, model)

    calibrated_ids = {
        row.example.question_id
        for row in test_scores
        if policy.threshold is not None and row.score <= policy.threshold
    }
    uncalibrated_ids = {
        row.example.question_id for row in test_scores if row.score <= alpha
    }
    fixed_count_ids = {
        row.example.question_id
        for row in rows
        if row.example.split == "test" and row.paper_count >= 5
    }
    stability_ids = {
        row.example.question_id
        for row in rows
        if row.example.split == "test"
        and row.example.features["bootstrap_instability"] <= 0.20
    }
    return {
        "seed": seed,
        "alpha": alpha,
        "delta": delta,
        "calibration_status": policy.status,
        "calibrated_threshold": policy.threshold,
        "calibration_upper_risk": (
            None
            if policy.selected is None
            else policy.selected.simultaneous_upper_risk
        ),
        "test_interval_95": evaluation.risk_interval_95,
        "policies": {
            "calibrated": _policy_summary(rows, calibrated_ids),
            "uncalibrated_score_at_alpha": _policy_summary(rows, uncalibrated_ids),
            "fixed_at_least_five_papers": _policy_summary(rows, fixed_count_ids),
            "bootstrap_instability_only": _policy_summary(rows, stability_ids),
        },
    }


def summarize_replicates(
    replicates: Sequence[Mapping[str, object]], *, alpha: float
) -> dict[str, object]:
    """Aggregate repeated simulations into paper-ready policy comparisons."""

    if not replicates:
        raise ValueError("replicates_empty")
    policy_names = sorted(
        {
            name
            for replicate in replicates
            for name in (replicate.get("policies") or {})  # type: ignore[union-attr]
        }
    )
    policies: dict[str, dict[str, float | int | None]] = {}
    for name in policy_names:
        summaries = [replicate["policies"][name] for replicate in replicates]  # type: ignore[index]
        coverages = [float(summary["coverage"]) for summary in summaries]
        observed_risks = [
            float(summary["empirical_risk"])
            for summary in summaries
            if summary["empirical_risk"] is not None
        ]
        true_risks = [
            float(summary["true_selective_risk"])
            for summary in summaries
            if summary["true_selective_risk"] is not None
        ]
        policies[name] = {
            "mean_coverage": float(np.mean(coverages)),
            "sd_coverage": float(np.std(coverages)),
            "mean_empirical_risk": (
                float(np.mean(observed_risks)) if observed_risks else None
            ),
            "mean_true_selective_risk": (
                float(np.mean(true_risks)) if true_risks else None
            ),
            "true_risk_violation_count": sum(value > alpha for value in true_risks),
            "nonempty_replicates": len(true_risks),
        }
    return {
        "simulation_summary_version": "1",
        "replicate_count": len(replicates),
        "alpha": alpha,
        "interpretation": (
            "planted simulation validates policy mechanics only; it does not calibrate "
            "risk on real scientific questions"
        ),
        "policies": policies,
    }


__all__ = [
    "DEFAULT_CANDIDATE_THRESHOLDS",
    "SimulatedQuestion",
    "simulate_questions",
    "simulate_replicate",
    "summarize_replicates",
]
