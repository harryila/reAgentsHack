"""Planted evaluation of effect-size meta-regression against significance voting.

The simulation intentionally gives moderator levels different study precision.  A
vote-counting method that turns ``p < .05`` into a direction can therefore invent a
moderator even when the underlying effect is identical.  The experiment is a method
check, not evidence about real review performance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist

import numpy as np
from scipy.stats import fisher_exact

from literature_multiverse.effects import (
    EffectEvidence,
    EffectFormat,
    EffectProvenance,
    HarmonizedEffect,
    ReportedSignificance,
    harmonize_effect,
)
from literature_multiverse.meta_analysis import categorical_meta_regression


def _simulate_effects(
    *,
    rng: np.random.Generator,
    prefix: str,
    papers_per_level: int,
    moderator_effect: float,
    overall_effect: float,
    tau: float,
) -> list[HarmonizedEffect]:
    effects: list[HarmonizedEffect] = []
    for level in ("high_precision", "low_precision"):
        for index in range(papers_per_level):
            if level == "high_precision":
                standard_error = float(rng.uniform(0.08, 0.14))
                planted_mean = overall_effect
            else:
                standard_error = float(rng.uniform(0.25, 0.38))
                planted_mean = overall_effect + moderator_effect
            latent = float(rng.normal(planted_mean, tau))
            estimate = float(rng.normal(latent, standard_error))
            z_value = estimate / standard_error
            significance = (
                ReportedSignificance.SIGNIFICANT
                if abs(z_value) >= NormalDist().inv_cdf(0.975)
                else ReportedSignificance.NOT_SIGNIFICANT
            )
            paper_id = f"{prefix}-{level}-{index:04d}"
            result = harmonize_effect(
                EffectEvidence(
                    paper_id=paper_id,
                    finding_id=f"finding-{paper_id}",
                    outcome="planted-continuous-outcome",
                    contrast="treatment-minus-control",
                    effect_format=EffectFormat.MEAN_DIFFERENCE,
                    estimate=estimate,
                    standard_error=standard_error,
                    unit="standardized-units",
                    reported_significance=significance,
                    moderators={"precision_group": level},
                    provenance=EffectProvenance(
                        source_locator=f"simulation://{paper_id}",
                        extraction_method="computed_from_reported_statistics",
                    ),
                )
            )
            assert result.effect is not None
            effects.append(result.effect)
    return effects


def _significance_vote_p_value(effects: Sequence[HarmonizedEffect]) -> float:
    table: list[list[int]] = []
    for level in ("high_precision", "low_precision"):
        rows = [effect for effect in effects if effect.moderators["precision_group"] == level]
        positive_significant = sum(
            effect.reported_significance is ReportedSignificance.SIGNIFICANT
            and effect.estimate > 0
            for effect in rows
        )
        table.append([positive_significant, len(rows) - positive_significant])
    return float(fisher_exact(table, alternative="two-sided").pvalue)


def _smoothed_vote_probabilities(
    effects: Sequence[HarmonizedEffect],
) -> dict[str, float]:
    probabilities: dict[str, float] = {}
    for level in ("high_precision", "low_precision"):
        rows = [effect for effect in effects if effect.moderators["precision_group"] == level]
        positives = sum(
            effect.reported_significance is ReportedSignificance.SIGNIFICANT
            and effect.estimate > 0
            for effect in rows
        )
        probabilities[level] = (positives + 1) / (len(rows) + 2)
    return probabilities


def _meta_positive_probabilities(
    report: Mapping[str, object],
) -> dict[str, tuple[float, float, float]]:
    heterogeneity = report["heterogeneity"]
    assert isinstance(heterogeneity, Mapping)
    tau_squared = float(heterogeneity["residual_tau_squared"])
    estimates = report["level_estimates"]
    assert isinstance(estimates, list)
    probabilities: dict[str, tuple[float, float, float]] = {}
    for raw in estimates:
        assert isinstance(raw, Mapping)
        level = str(raw["level"])
        mean = float(raw["estimate"])
        mean_variance = float(raw["standard_error"]) ** 2
        probabilities[level] = (mean, tau_squared, mean_variance)
    return probabilities


def _heldout_brier(
    test_effects: Sequence[HarmonizedEffect],
    *,
    vote_probabilities: Mapping[str, float],
    meta_parameters: Mapping[str, tuple[float, float, float]],
) -> tuple[float, float]:
    vote_losses: list[float] = []
    meta_losses: list[float] = []
    for effect in test_effects:
        level = str(effect.moderators["precision_group"])
        target = float(effect.estimate > 0)
        vote_probability = float(vote_probabilities[level])
        mean, tau_squared, mean_variance = meta_parameters[level]
        predictive_sd = math.sqrt(tau_squared + mean_variance + effect.variance)
        meta_probability = NormalDist(mu=mean, sigma=predictive_sd).cdf(0.0)
        meta_probability = 1.0 - meta_probability
        vote_losses.append((vote_probability - target) ** 2)
        meta_losses.append((meta_probability - target) ** 2)
    return float(np.mean(meta_losses)), float(np.mean(vote_losses))


def simulate_meta_replicate(
    *,
    seed: int,
    moderator_effect: float,
    papers_per_level: int = 30,
    heldout_papers_per_level: int = 30,
    overall_effect: float = 0.25,
    tau: float = 0.12,
    alpha: float = 0.05,
) -> dict[str, float | int | bool]:
    """Run one planted train/test replicate with paper-independent effects."""

    if min(papers_per_level, heldout_papers_per_level) < 4:
        raise ValueError("meta_simulation_requires_four_papers_per_level")
    if tau < 0 or not 0 < alpha < 1:
        raise ValueError("meta_simulation_parameters_invalid")
    rng = np.random.default_rng(seed)
    train = _simulate_effects(
        rng=rng,
        prefix=f"sim-{seed}-train",
        papers_per_level=papers_per_level,
        moderator_effect=moderator_effect,
        overall_effect=overall_effect,
        tau=tau,
    )
    test = _simulate_effects(
        rng=rng,
        prefix=f"sim-{seed}-test",
        papers_per_level=heldout_papers_per_level,
        moderator_effect=moderator_effect,
        overall_effect=overall_effect,
        tau=tau,
    )
    report = categorical_meta_regression(
        train,
        "precision_group",
        reference_level="high_precision",
        min_papers_per_level=4,
    )
    if report["status"] != "ok":
        raise RuntimeError(f"planted_meta_regression_failed:{report['reason']}")
    omnibus = report["omnibus"]
    assert isinstance(omnibus, Mapping)
    meta_p_value = float(omnibus["p_value"])
    vote_p_value = _significance_vote_p_value(train)
    meta_brier, vote_brier = _heldout_brier(
        test,
        vote_probabilities=_smoothed_vote_probabilities(train),
        meta_parameters=_meta_positive_probabilities(report),
    )
    return {
        "seed": seed,
        "moderator_effect": moderator_effect,
        "meta_regression_p_value": meta_p_value,
        "significance_vote_p_value": vote_p_value,
        "meta_regression_detected": meta_p_value < alpha,
        "significance_vote_detected": vote_p_value < alpha,
        "meta_regression_heldout_brier": meta_brier,
        "significance_vote_heldout_brier": vote_brier,
    }


def summarize_meta_simulations(
    null_replicates: Sequence[Mapping[str, object]],
    moderator_replicates: Sequence[Mapping[str, object]],
    *,
    alpha: float,
) -> dict[str, object]:
    """Summarize false discovery, power, and held-out predictive scoring."""

    if not null_replicates or not moderator_replicates:
        raise ValueError("meta_simulation_replicates_empty")

    def scenario(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
        return {
            "replicates": len(rows),
            "meta_detection_rate": float(
                np.mean([bool(row["meta_regression_detected"]) for row in rows])
            ),
            "significance_vote_detection_rate": float(
                np.mean([bool(row["significance_vote_detected"]) for row in rows])
            ),
            "meta_mean_heldout_brier": float(
                np.mean([float(row["meta_regression_heldout_brier"]) for row in rows])
            ),
            "significance_vote_mean_heldout_brier": float(
                np.mean([float(row["significance_vote_heldout_brier"]) for row in rows])
            ),
        }

    return {
        "meta_simulation_summary_version": "1",
        "alpha": alpha,
        "interpretation": (
            "planted precision-confounded simulation validates method behavior only; "
            "it is not evidence of real-review accuracy"
        ),
        "null_moderator": scenario(null_replicates),
        "planted_moderator": scenario(moderator_replicates),
    }


__all__ = ["simulate_meta_replicate", "summarize_meta_simulations"]
