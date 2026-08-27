from __future__ import annotations

import numpy as np
import pytest

from literature_multiverse.effects import (
    EffectEvidence,
    HarmonizationResult,
    PointDirection,
    harmonize_effect,
)
from literature_multiverse.meta_analysis import (
    aggregate_one_effect_per_paper,
    categorical_meta_regression,
    directional_synthesis,
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
    moderator: str | None = None,
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

    regression = categorical_meta_regression(
        _effects(results), "dose", reference_level="low"
    )

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
