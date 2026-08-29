from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import t

from literature_multiverse.effects import (
    EffectEvidence,
    HarmonizationResult,
    PointDirection,
    harmonize_effect,
)
from literature_multiverse.meta_analysis import (
    aggregate_one_effect_per_paper,
    categorical_meta_regression,
    cohort_categorical_meta_regression,
    directional_synthesis,
    prespecified_cohort_condition_analysis,
    prespecified_condition_analysis,
    random_effects_meta_analysis,
    synthesize_with_directional_fallback,
)


def _result(
    paper_id: str,
    estimate: float | None,
    standard_error: float | None,
    *,
    finding_suffix: str = "1",
    outcome: str = "performance",
    effect_format: str = "hedges_g",
    unit: str | None = None,
    significance: str = "not_reported",
    moderator: str | int | float | bool | None = None,
) -> HarmonizationResult:
    evidence = EffectEvidence(
        paper_id=paper_id,
        finding_id=f"{paper_id}-{finding_suffix}",
        outcome=outcome,
        contrast="intervention_vs_control",
        effect_format=effect_format,
        unit=unit,
        estimate=estimate,
        standard_error=standard_error,
        reported_significance=significance,
        moderators={"dose": moderator},
        provenance={"source_locator": f"{paper_id}.pdf#table=2"},
    )
    return harmonize_effect(evidence)


def _effects(results: list[HarmonizationResult]):
    return [result.effect for result in results if result.effect is not None]


def test_random_effects_simulation_recovers_planted_mean() -> None:
    rng = np.random.default_rng(20260826)
    true_mean = 0.4
    effects = []
    for index in range(40):
        standard_error = float(rng.uniform(0.08, 0.18))
        observed = float(rng.normal(true_mean, np.sqrt(0.2**2 + standard_error**2)))
        result = _result(f"p{index:02}", observed, standard_error)
        assert result.effect is not None
        effects.append(result.effect)

    synthesis = random_effects_meta_analysis(effects)

    assert synthesis["status"] == "ok"
    assert synthesis["n_papers"] == 40
    assert synthesis["estimate"] == pytest.approx(true_mean, abs=0.12)
    assert synthesis["ci_lower"] < true_mean < synthesis["ci_upper"]
    assert synthesis["heterogeneity"]["tau_squared"] > 0
    assert synthesis["prediction_interval"]["lower"] < synthesis["estimate"]
    assert synthesis["prediction_interval"]["upper"] > synthesis["estimate"]


def test_random_effects_matches_independent_scalar_pm_mkh_reference() -> None:
    estimates = np.asarray([0.12, 0.34, 0.55, -0.08, 0.41], dtype=float)
    variances = np.asarray([0.04, 0.0225, 0.0625, 0.0324, 0.01], dtype=float)
    effects = _effects(
        [
            _result(f"reference-{index}", estimate, math.sqrt(variance))
            for index, (estimate, variance) in enumerate(zip(estimates, variances, strict=True))
        ]
    )

    # Independent scalar reference: solve sum(w_i(y_i-mu_w)^2)=k-1 with
    # scipy's bracketed root finder, without calling implementation helpers.
    def weighted_mean_and_q(tau_squared: float) -> tuple[float, float, float]:
        weights = 1.0 / (variances + tau_squared)
        weight_sum = float(np.sum(weights))
        mean = float(np.sum(weights * estimates) / weight_sum)
        q_value = float(np.sum(weights * (estimates - mean) ** 2))
        return mean, q_value, weight_sum

    degrees_freedom = len(estimates) - 1
    tau_squared = brentq(
        lambda value: weighted_mean_and_q(value)[1] - degrees_freedom,
        0.0,
        10.0,
    )
    mean, residual_q, weight_sum = weighted_mean_and_q(tau_squared)
    scale = max(1.0, residual_q / degrees_freedom)
    standard_error = math.sqrt(scale / weight_sum)
    critical = float(t.ppf(0.975, degrees_freedom))
    prediction_standard_error = math.sqrt(tau_squared + standard_error**2)
    prediction_critical = float(t.ppf(0.975, len(estimates) - 2))

    synthesis = random_effects_meta_analysis(effects)

    assert synthesis["heterogeneity"]["tau_squared"] == pytest.approx(tau_squared)
    assert synthesis["estimate"] == pytest.approx(mean)
    assert synthesis["standard_error"] == pytest.approx(standard_error)
    assert synthesis["ci_lower"] == pytest.approx(mean - critical * standard_error)
    assert synthesis["ci_upper"] == pytest.approx(mean + critical * standard_error)
    assert synthesis["prediction_interval"]["lower"] == pytest.approx(
        mean - prediction_critical * prediction_standard_error
    )
    assert synthesis["prediction_interval"]["upper"] == pytest.approx(
        mean + prediction_critical * prediction_standard_error
    )


def test_one_effect_per_paper_aggregation_prevents_pseudoreplication() -> None:
    results = [
        _result("paper-a", 0.2, 0.1, finding_suffix="1"),
        _result("paper-a", 0.2, 0.1, finding_suffix="2"),
        _result("paper-b", 0.4, 0.1),
    ]
    aggregation = aggregate_one_effect_per_paper(_effects(results))

    assert aggregation.status == "ok"
    assert aggregation.n_effects_input == 3
    assert aggregation.n_papers == 2
    paper_a = next(effect for effect in aggregation.effects if effect.paper_id == "paper-a")
    assert paper_a.estimate == pytest.approx(0.2)
    # rho=1 means duplicating an effect does not halve its variance.
    assert paper_a.variance == pytest.approx(0.1**2)
    assert paper_a.source_finding_ids == ["paper-a-1", "paper-a-2"]


def test_incompatible_effect_scales_return_explicit_insufficiency() -> None:
    results = [
        _result("paper-a", 0.2, 0.1),
        _result(
            "paper-b",
            2.0,
            0.2,
            outcome="performance",
            effect_format="mean_difference",
            unit="watts",
        ),
    ]
    synthesis = random_effects_meta_analysis(_effects(results))

    assert synthesis["status"] == "insufficient"
    assert synthesis["reason"] == "incompatible_effect_scales"


def test_single_paper_returns_insufficient_instead_of_fake_heterogeneity() -> None:
    result = _result("paper-a", 0.2, 0.1)
    synthesis = random_effects_meta_analysis(_effects([result]))

    assert synthesis["status"] == "insufficient"
    assert synthesis["reason"] == "fewer_than_two_papers"


def test_categorical_meta_regression_recovers_predictive_level_difference() -> None:
    rng = np.random.default_rng(19)
    results: list[HarmonizationResult] = []
    for index in range(36):
        level = "high" if index >= 18 else "low"
        planted = 0.65 if level == "high" else 0.05
        result = _result(
            f"paper-{index:02}",
            float(rng.normal(planted, 0.12)),
            0.12,
            moderator=level,
        )
        results.append(result)

    regression = categorical_meta_regression(_effects(results), "dose", reference_level="low")

    assert regression["status"] == "ok"
    assert regression["n_papers"] == 36
    assert regression["interpretation"] == "predictive_association_not_causal"
    coefficient = regression["coefficients"][0]
    assert coefficient["level"] == "high"
    assert coefficient["estimate_difference"] == pytest.approx(0.6, abs=0.12)
    assert coefficient["ci_lower"] > 0
    assert regression["omnibus"]["p_value"] < 0.01


def test_meta_regression_rejects_moderator_that_varies_within_paper() -> None:
    results = [
        _result("paper-a", 0.1, 0.1, finding_suffix="1", moderator="low"),
        _result("paper-a", 0.2, 0.1, finding_suffix="2", moderator="high"),
        _result("paper-b", 0.3, 0.1, moderator="high"),
        _result("paper-c", 0.0, 0.1, moderator="low"),
    ]
    regression = categorical_meta_regression(_effects(results), "dose")

    assert regression["status"] == "insufficient"
    assert regression["reason"] == "moderator_varies_within_paper"
    assert regression["conflicted_paper_ids"] == ["paper-a"]


def test_directional_fallback_uses_sign_not_reported_significance() -> None:
    results = [
        _result("paper-a", 0.3, None, significance="not_significant"),
        _result("paper-b", -0.2, None, significance="not_significant"),
        _result("paper-c", 0.0, None, significance="not_significant"),
        _result("paper-d", None, None, significance="inconclusive"),
        _result("paper-e", 0.4, None, finding_suffix="1"),
        _result("paper-e", -0.1, None, finding_suffix="2"),
    ]

    synthesis = directional_synthesis(results)

    assert synthesis["status"] == "ok"
    assert synthesis["paper_direction_counts"] == {
        "increase": 1,
        "decrease": 1,
        "exact_zero": 1,
        "mixed": 1,
        "unavailable": 1,
    }
    assert synthesis["paper_directions"]["paper-a"] == PointDirection.INCREASE.value
    assert synthesis["paper_directions"]["paper-b"] == PointDirection.DECREASE.value
    assert "no_effect" not in synthesis["paper_direction_counts"]


def test_end_to_end_wrapper_exposes_why_quantitative_synthesis_fell_back() -> None:
    results = [
        _result("paper-a", 0.3, None),
        _result("paper-b", 0.2, None),
    ]
    synthesis = synthesize_with_directional_fallback(results)

    assert synthesis["status"] == "ok"
    assert synthesis["mode"] == "directional_sign_synthesis"
    assert synthesis["quantitative"]["reason"] == "no_estimable_effects"
    assert synthesis["directional_fallback"]["paper_direction_counts"]["increase"] == 2


def test_prespecified_condition_analysis_requires_adjusted_direction_reversal() -> None:
    effects = _effects(
        [
            _result(
                f"paper-{level}-{index}",
                0.8 if level == "high" else -0.8,
                0.08,
                moderator=level,
            )
            for level in ("low", "high")
            for index in range(6)
        ]
    )

    analysis = prespecified_condition_analysis(effects, ["dose"])

    assert analysis["status"] == "exploratory_qualitative_condition_signal"
    assert analysis["interpretation"] == "predictive_association_not_causal"
    qualifier = analysis["qualifying_moderators"][0]
    assert qualifier["moderator"] == "dose"
    assert qualifier["positive_levels"] == ["high"]
    assert qualifier["negative_levels"] == ["low"]


def test_same_direction_moderator_difference_is_not_condition_dependent() -> None:
    effects = _effects(
        [
            _result(
                f"paper-{level}-{index}",
                0.8 if level == "high" else 0.2,
                0.08,
                moderator=level,
            )
            for level in ("low", "high")
            for index in range(6)
        ]
    )

    analysis = prespecified_condition_analysis(effects, ["dose"])

    assert analysis["status"] == "no_qualitative_condition_dependence_detected"
    assert analysis["qualifying_moderators"] == []


def test_cohort_meta_regression_counts_shared_cohort_once_across_reports() -> None:
    effects = _effects(
        [
            _result("paper-a", -0.5, 0.1, moderator="low"),
            _result("paper-b", -0.5, 0.1, moderator="low"),
            _result("paper-c", -0.4, 0.1, moderator="low"),
            _result("paper-d", 0.4, 0.1, moderator="high"),
            _result("paper-e", 0.5, 0.1, moderator="high"),
        ]
    )
    regression = cohort_categorical_meta_regression(
        effects,
        ["cohort-shared", "cohort-shared", "cohort-low", "cohort-high-1", "cohort-high-2"],
        "dose",
    )

    assert regression["status"] == "ok"
    assert regression["analysis_unit"] == "explicit_cohort"
    assert regression["n_effects_input"] == 5
    assert regression["n_publications"] == 5
    assert regression["n_cohorts"] == 4
    assert regression["support_by_level"] == {"high": 2, "low": 2}
    assert regression["provenance"]["cohort-shared"]["paper_ids"] == [
        "paper-a",
        "paper-b",
    ]
    shared = next(
        effect for effect in regression["cohort_effects"] if effect["cohort_id"] == "cohort-shared"
    )
    # Perfect unknown within-cohort correlation prevents the second report from
    # manufacturing extra precision.
    assert shared["variance"] == pytest.approx(0.1**2)


def test_cohort_meta_regression_allows_multiple_cohorts_from_one_paper() -> None:
    effects = _effects(
        [
            _result(
                "paper-a",
                estimate,
                0.1,
                finding_suffix=str(index),
                moderator=level,
            )
            for index, (level, estimate) in enumerate(
                (("low", -0.5), ("low", -0.4), ("high", 0.4), ("high", 0.5)),
                start=1,
            )
        ]
    )
    regression = cohort_categorical_meta_regression(
        effects,
        ["cohort-1", "cohort-2", "cohort-3", "cohort-4"],
        "dose",
    )

    assert regression["status"] == "ok"
    assert regression["n_publications"] == 1
    assert regression["n_cohorts"] == 4
    assert set(regression["provenance"]) == {
        "cohort-1",
        "cohort-2",
        "cohort-3",
        "cohort-4",
    }


def test_cohort_meta_regression_fails_closed_on_within_cohort_moderator_conflict() -> None:
    effects = _effects(
        [
            _result("paper-a", -0.5, 0.1, moderator="low"),
            _result("paper-b", -0.4, 0.1, moderator="high"),
            _result("paper-c", -0.3, 0.1, moderator="low"),
            _result("paper-d", 0.3, 0.1, moderator="high"),
            _result("paper-e", 0.4, 0.1, moderator="high"),
        ]
    )
    regression = cohort_categorical_meta_regression(
        effects,
        ["cohort-conflict", "cohort-conflict", "cohort-low", "cohort-high-1", "cohort-high-2"],
        "dose",
    )

    assert regression["status"] == "insufficient"
    assert regression["reason"] == "moderator_varies_within_cohort"
    assert regression["conflicted_cohort_ids"] == ["cohort-conflict"]
    condition = prespecified_cohort_condition_analysis(
        effects,
        ["cohort-conflict", "cohort-conflict", "cohort-low", "cohort-high-1", "cohort-high-2"],
        ["dose"],
    )
    assert condition["status"] == "insufficient"
    assert condition["reason"] == "prespecified_cohort_moderator_family_incomplete"


def test_cohort_moderators_preserve_bool_int_and_string_level_identity() -> None:
    conflicted = _effects(
        [
            _result("paper-a", -0.5, 0.1, moderator=True),
            _result("paper-b", -0.4, 0.1, moderator=1),
            _result("paper-c", 0.4, 0.1, moderator="high"),
            _result("paper-d", 0.5, 0.1, moderator="high"),
        ]
    )
    conflict = cohort_categorical_meta_regression(
        conflicted,
        ["cohort-shared", "cohort-shared", "cohort-high-1", "cohort-high-2"],
        "dose",
    )
    assert conflict["status"] == "insufficient"
    assert conflict["reason"] == "moderator_varies_within_cohort"

    distinct_levels = _effects(
        [
            _result("paper-int-1", -0.5, 0.1, moderator=1),
            _result("paper-int-2", -0.4, 0.1, moderator=1),
            _result("paper-str-1", 0.4, 0.1, moderator="1"),
            _result("paper-str-2", 0.5, 0.1, moderator="1"),
        ]
    )
    regression = cohort_categorical_meta_regression(
        distinct_levels,
        ["cohort-int-1", "cohort-int-2", "cohort-str-1", "cohort-str-2"],
        "dose",
    )
    assert regression["status"] == "ok"
    assert regression["support_by_level"] == {"1": 2, "int:1": 2}


def test_cohort_meta_regression_fails_closed_on_missing_moderator() -> None:
    effects = _effects(
        [
            _result("paper-a", -0.5, 0.1, moderator=None),
            _result("paper-b", -0.4, 0.1, moderator="low"),
            _result("paper-c", 0.4, 0.1, moderator="high"),
            _result("paper-d", 0.5, 0.1, moderator="high"),
        ]
    )
    regression = cohort_categorical_meta_regression(
        effects,
        ["cohort-missing", "cohort-low", "cohort-high-1", "cohort-high-2"],
        "dose",
    )

    assert regression["status"] == "insufficient"
    assert regression["reason"] == "moderator_missing_for_cohort"
    assert regression["missing_moderator_cohort_ids"] == ["cohort-missing"]


def test_cohort_condition_rule_requires_multiplicity_adjusted_opposite_intervals() -> None:
    cohort_ids = [f"cohort-{index}" for index in range(6)]
    borderline = _effects(
        [
            _result(
                f"paper-{index}",
                -0.2 if index < 3 else 0.2,
                0.1,
                moderator="low" if index < 3 else "high",
            )
            for index in range(6)
        ]
    )
    borderline_analysis = prespecified_cohort_condition_analysis(
        borderline,
        cohort_ids,
        ["dose"],
    )

    assert borderline_analysis["multiplicity_test_count"] == 3
    assert borderline_analysis["adjusted_alpha_per_test"] == pytest.approx(0.05 / 3)
    assert borderline_analysis["status"] == "no_qualitative_condition_dependence_detected"
    borderline_regression = borderline_analysis["analyses"][0]["regression"]
    # The adjusted omnibus detects a difference, but the simultaneous level
    # intervals do not establish opposite directions.
    assert (
        borderline_regression["omnibus"]["p_value"]
        <= borderline_analysis["adjusted_alpha_per_test"]
    )
    assert borderline_analysis["analyses"][0]["positive_levels"] == []
    assert borderline_analysis["analyses"][0]["negative_levels"] == []

    strong = _effects(
        [
            _result(
                f"paper-{index}",
                -0.8 if index < 3 else 0.8,
                0.1,
                moderator="low" if index < 3 else "high",
            )
            for index in range(6)
        ]
    )
    strong_analysis = prespecified_cohort_condition_analysis(
        strong,
        cohort_ids,
        ["dose"],
    )

    assert strong_analysis["status"] == "exploratory_qualitative_condition_signal"
    assert strong_analysis["qualifying_moderators"] == [
        {
            "moderator": "dose",
            "positive_levels": ["high"],
            "negative_levels": ["low"],
            "omnibus_p_value": pytest.approx(
                strong_analysis["analyses"][0]["regression"]["omnibus"]["p_value"]
            ),
        }
    ]
