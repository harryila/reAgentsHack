from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from literature_multiverse.effects import (
    EffectAvailability,
    EffectEvidence,
    EffectFormat,
    EffectProvenance,
    EquivalenceConclusion,
    HarmonizedMeasure,
    PointDirection,
    ReportedSignificance,
    harmonize_effect,
)


def _evidence(**overrides: object) -> EffectEvidence:
    payload: dict[str, object] = {
        "paper_id": "paper-1",
        "finding_id": "finding-1",
        "outcome": "systolic_blood_pressure",
        "contrast": "intervention_vs_control",
        "effect_format": "mean_difference",
        "unit": "mmHg",
        "estimate": 1.5,
        "standard_error": 0.5,
        "reported_significance": "not_significant",
        "provenance": {
            "source_locator": "paper-1.pdf#page=4",
            "source_quote": "The adjusted difference was 1.5 mmHg.",
        },
    }
    payload.update(overrides)
    return EffectEvidence.model_validate(payload)


def test_non_significant_positive_estimate_is_not_recoded_as_zero() -> None:
    result = harmonize_effect(_evidence())

    assert result.status == "estimable"
    assert result.point_direction is PointDirection.INCREASE
    assert result.reported_significance is ReportedSignificance.NOT_SIGNIFICANT
    assert result.equivalence_conclusion is EquivalenceConclusion.NOT_TESTED
    assert result.effect is not None
    assert result.effect.estimate == 1.5


def test_equivalence_conclusion_is_independent_of_point_direction() -> None:
    result = harmonize_effect(
        _evidence(
            estimate=-0.2,
            equivalence_conclusion="equivalent",
            equivalence_margin=1.0,
        )
    )

    assert result.point_direction is PointDirection.DECREASE
    assert result.equivalence_conclusion is EquivalenceConclusion.EQUIVALENT


def test_continuous_group_summaries_are_harmonized_to_hedges_g() -> None:
    result = harmonize_effect(
        _evidence(
            effect_format="hedges_g",
            unit=None,
            estimate=None,
            standard_error=None,
            treatment_mean=12.0,
            treatment_sd=2.0,
            treatment_n=20,
            control_mean=10.0,
            control_sd=2.0,
            control_n=20,
        )
    )

    assert result.status == "estimable"
    assert result.effect is not None
    assert result.effect.measure is HarmonizedMeasure.STANDARDIZED_MEAN_DIFFERENCE
    assert result.effect.estimate == pytest.approx(0.9801, abs=0.002)
    assert result.effect.variance > 0
    assert result.effect.derivation == "hedges_g_from_group_summaries"


def test_binary_counts_use_a_disclosed_continuity_correction() -> None:
    result = harmonize_effect(
        _evidence(
            outcome="event",
            effect_format="odds_ratio",
            unit=None,
            estimate=None,
            standard_error=None,
            treatment_events=0,
            treatment_total=20,
            control_events=5,
            control_total=20,
        )
    )

    assert result.status == "estimable"
    assert result.effect is not None
    assert result.effect.measure is HarmonizedMeasure.LOG_ODDS_RATIO
    assert result.effect.estimate < 0
    assert result.effect.continuity_correction == 0.5
    assert result.effect.derivation == "log_odds_ratio_from_2x2_counts"


def test_binary_and_continuous_group_statistics_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="choose_continuous_or_binary_group_statistics"):
        _evidence(
            effect_format="odds_ratio",
            unit=None,
            estimate=None,
            standard_error=None,
            treatment_mean=12.0,
            treatment_sd=2.0,
            treatment_n=20,
            control_mean=10.0,
            control_sd=2.0,
            control_n=20,
            treatment_events=3,
            treatment_total=20,
            control_events=5,
            control_total=20,
        )


def test_ratio_confidence_interval_is_converted_on_log_scale() -> None:
    result = harmonize_effect(
        _evidence(
            outcome="event",
            effect_format="odds_ratio",
            unit=None,
            estimate=2.0,
            standard_error=None,
            ci_lower=1.0,
            ci_upper=4.0,
        )
    )

    assert result.effect is not None
    expected_se = math.log(4.0) / (2 * 1.959963984540054)
    assert result.effect.estimate == pytest.approx(math.log(2.0))
    assert result.effect.standard_error == pytest.approx(expected_se)
    assert result.effect.derivation.startswith("log_se_from_reported_")


def test_point_estimate_must_lie_inside_reported_confidence_interval() -> None:
    with pytest.raises(ValidationError, match="effect_estimate_outside_confidence_interval"):
        _evidence(standard_error=None, ci_lower=2.0, ci_upper=3.0)


def test_redundant_direct_uncertainty_sources_are_rejected() -> None:
    with pytest.raises(ValidationError, match="direct_uncertainty_sources_mutually_exclusive"):
        _evidence(variance=0.25)


def test_nonfinite_moderator_values_are_rejected() -> None:
    with pytest.raises(ValidationError, match="moderator_values_must_be_finite"):
        _evidence(moderators={"dose": math.nan})


def test_point_estimate_without_uncertainty_is_explicit_directional_only() -> None:
    result = harmonize_effect(_evidence(standard_error=None, reported_significance="significant"))

    assert result.status == "insufficient"
    assert result.reason == "sampling_uncertainty_not_reported"
    assert result.point_direction is PointDirection.INCREASE
    assert result.reported_significance is ReportedSignificance.SIGNIFICANT
    assert result.effect is None


@pytest.mark.parametrize(
    ("availability", "expected_reason"),
    [
        (EffectAvailability.MISSING, "effect_not_reported"),
        (EffectAvailability.INCONCLUSIVE, "effect_report_inconclusive"),
    ],
)
def test_missing_and_inconclusive_are_distinct_insufficiency_states(
    availability: EffectAvailability, expected_reason: str
) -> None:
    result = harmonize_effect(
        _evidence(
            availability=availability,
            estimate=None,
            standard_error=None,
        )
    )

    assert result.status == "insufficient"
    assert result.reason == expected_reason
    assert result.point_direction is PointDirection.NOT_AVAILABLE


def test_mean_difference_requires_a_unit_to_prevent_invalid_pooling() -> None:
    with pytest.raises(ValidationError, match="mean_difference_requires_unit"):
        _evidence(unit=None)


def test_event_counts_must_be_coherent() -> None:
    with pytest.raises(ValidationError, match="treatment_events_exceed_total"):
        _evidence(
            effect_format=EffectFormat.LOG_RISK_RATIO,
            unit=None,
            estimate=None,
            standard_error=None,
            treatment_events=11,
            treatment_total=10,
            control_events=2,
            control_total=10,
        )


def test_provenance_is_required() -> None:
    with pytest.raises(ValidationError, match="provenance"):
        EffectEvidence(
            paper_id="p",
            finding_id="f",
            outcome="outcome",
            contrast="intervention_vs_control",
            effect_format="hedges_g",
            estimate=0.2,
            standard_error=0.1,
        )

    provenance = EffectProvenance(source_locator="table 2")
    assert provenance.extraction_method == "reported"
