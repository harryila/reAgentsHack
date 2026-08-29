"""Typed effect evidence and auditable harmonization onto meta-analytic scales.

The contracts in this module deliberately keep four concepts separate:

* the sign of a reported point estimate;
* the authors' reported significance conclusion;
* the result of an equivalence test; and
* whether a magnitude and sampling variance can be synthesized.

In particular, ``not_significant`` is never converted to an exact-zero effect.
"""

from __future__ import annotations

import math
from enum import StrEnum
from statistics import NormalDist
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.models import ContractModel


class EffectContractError(ValueError):
    """Effect evidence is internally inconsistent or cannot be interpreted safely."""


class EffectAvailability(StrEnum):
    """Whether the source supplies a point estimate that may be interpreted."""

    AVAILABLE = "available"
    MISSING = "missing"
    INCONCLUSIVE = "inconclusive"


class EffectFormat(StrEnum):
    """Supported source formats; ratio formats are harmonized onto log scales."""

    UNSPECIFIED = "unspecified"
    MEAN_DIFFERENCE = "mean_difference"
    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    ODDS_RATIO = "odds_ratio"
    LOG_ODDS_RATIO = "log_odds_ratio"
    RISK_RATIO = "risk_ratio"
    LOG_RISK_RATIO = "log_risk_ratio"


class HarmonizedMeasure(StrEnum):
    """Scales that can be pooled without silently mixing unlike estimands."""

    MEAN_DIFFERENCE = "mean_difference"
    STANDARDIZED_MEAN_DIFFERENCE = "standardized_mean_difference"
    LOG_ODDS_RATIO = "log_odds_ratio"
    LOG_RISK_RATIO = "log_risk_ratio"


class PointDirection(StrEnum):
    """Sign of a point estimate, independent of its uncertainty or significance."""

    INCREASE = "increase"
    DECREASE = "decrease"
    EXACT_ZERO = "exact_zero"
    NOT_AVAILABLE = "not_available"


class ReportedSignificance(StrEnum):
    """What the source reports about a conventional difference test."""

    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    NOT_REPORTED = "not_reported"
    INCONCLUSIVE = "inconclusive"


class EquivalenceConclusion(StrEnum):
    """What a prespecified equivalence analysis concludes, if one was reported."""

    EQUIVALENT = "equivalent"
    NOT_EQUIVALENT = "not_equivalent"
    INCONCLUSIVE = "inconclusive"
    NOT_TESTED = "not_tested"


class EffectProvenance(ContractModel):
    """Minimum source trail needed to audit a numerical effect."""

    source_locator: Annotated[str, Field(min_length=1)]
    source_quote: str | None = None
    extraction_method: Literal["reported", "computed_from_reported_statistics"] = "reported"


class EffectEvidence(ContractModel):
    """One paper-level or within-paper effect as represented in its source.

    ``estimate``, ``standard_error``, ``variance``, and confidence limits use the
    source ``effect_format`` scale.  Thus an odds-ratio confidence interval is also
    on the odds-ratio scale, while a log-odds-ratio interval is on the log scale.
    Group summaries or 2x2 counts can be supplied instead of a direct estimate.
    """

    paper_id: Annotated[str, Field(min_length=1)]
    finding_id: Annotated[str, Field(min_length=1)]
    outcome: Annotated[str, Field(min_length=1)]
    contrast: Annotated[str, Field(min_length=1)]
    effect_format: EffectFormat = EffectFormat.UNSPECIFIED
    availability: EffectAvailability = EffectAvailability.AVAILABLE

    estimate: float | None = None
    standard_error: Annotated[float, Field(gt=0)] | None = None
    variance: Annotated[float, Field(gt=0)] | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    ci_level: Annotated[float, Field(gt=0, lt=1)] = 0.95
    unit: str | None = None

    treatment_mean: float | None = None
    treatment_sd: Annotated[float, Field(gt=0)] | None = None
    treatment_n: Annotated[int, Field(ge=2)] | None = None
    control_mean: float | None = None
    control_sd: Annotated[float, Field(gt=0)] | None = None
    control_n: Annotated[int, Field(ge=2)] | None = None

    treatment_events: Annotated[int, Field(ge=0)] | None = None
    treatment_total: Annotated[int, Field(ge=1)] | None = None
    control_events: Annotated[int, Field(ge=0)] | None = None
    control_total: Annotated[int, Field(ge=1)] | None = None

    reported_p_value: Annotated[float, Field(ge=0, le=1)] | None = None
    reported_significance: ReportedSignificance = ReportedSignificance.NOT_REPORTED
    equivalence_conclusion: EquivalenceConclusion = EquivalenceConclusion.NOT_TESTED
    equivalence_margin: Annotated[float, Field(gt=0)] | None = None
    moderators: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    provenance: EffectProvenance

    @field_validator(
        "estimate",
        "ci_lower",
        "ci_upper",
        "treatment_mean",
        "control_mean",
    )
    @classmethod
    def finite_optional(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("effect_numeric_value_must_be_finite")
        return value

    @field_validator(
        "standard_error",
        "variance",
        "treatment_sd",
        "control_sd",
        "equivalence_margin",
    )
    @classmethod
    def finite_positive_optional(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("effect_uncertainty_value_must_be_finite")
        return value

    @field_validator("unit")
    @classmethod
    def normalize_unit(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("effect_unit_must_be_nonempty")
        return normalized

    @field_validator("moderators")
    @classmethod
    def validate_moderators(
        cls, value: dict[str, str | int | float | bool | None]
    ) -> dict[str, str | int | float | bool | None]:
        if any(not name.strip() for name in value):
            raise ValueError("moderator_names_must_be_nonempty")
        if any(isinstance(item, float) and not math.isfinite(item) for item in value.values()):
            raise ValueError("moderator_values_must_be_finite")
        return value

    @model_validator(mode="after")
    def validate_evidence(self) -> EffectEvidence:
        ci_supplied = self.ci_lower is not None or self.ci_upper is not None
        if ci_supplied and (self.ci_lower is None or self.ci_upper is None):
            raise ValueError("effect_confidence_interval_requires_both_limits")
        if (
            self.ci_lower is not None
            and self.ci_upper is not None
            and self.ci_lower >= self.ci_upper
        ):
            raise ValueError("effect_confidence_interval_not_ordered")
        if (
            self.estimate is not None
            and self.ci_lower is not None
            and self.ci_upper is not None
            and not self.ci_lower <= self.estimate <= self.ci_upper
        ):
            raise ValueError("effect_estimate_outside_confidence_interval")
        uncertainty_sources = sum(
            (
                self.standard_error is not None,
                self.variance is not None,
                ci_supplied,
            )
        )
        if uncertainty_sources > 1:
            raise ValueError("direct_uncertainty_sources_mutually_exclusive")

        continuous_descriptives = (
            self.treatment_mean,
            self.treatment_sd,
            self.control_mean,
            self.control_sd,
        )
        continuous = (*continuous_descriptives, self.treatment_n, self.control_n)
        # Direct Cohen's d may legitimately provide only the two group sizes, which
        # are enough for its small-sample bias correction.  Any descriptive group
        # statistic, however, requires the complete six-field summary.
        if any(value is not None for value in continuous_descriptives) and any(
            value is None for value in continuous
        ):
            raise ValueError("continuous_group_statistics_incomplete")
        if (self.treatment_n is None) != (self.control_n is None):
            raise ValueError("continuous_group_sizes_require_both_groups")
        direct_statistics = (
            self.estimate,
            self.standard_error,
            self.variance,
            self.ci_lower,
            self.ci_upper,
        )
        if any(value is not None for value in continuous_descriptives) and any(
            value is not None for value in direct_statistics
        ):
            raise ValueError("choose_direct_effect_or_continuous_group_statistics")

        binary = (
            self.treatment_events,
            self.treatment_total,
            self.control_events,
            self.control_total,
        )
        if any(value is not None for value in binary) and any(value is None for value in binary):
            raise ValueError("binary_group_statistics_incomplete")
        if any(value is not None for value in continuous) and any(
            value is not None for value in binary
        ):
            raise ValueError("choose_continuous_or_binary_group_statistics")
        if any(value is not None for value in binary) and any(
            value is not None for value in direct_statistics
        ):
            raise ValueError("choose_direct_effect_or_binary_group_statistics")
        if self.treatment_events is not None:
            assert self.treatment_total is not None
            assert self.control_events is not None
            assert self.control_total is not None
            if self.treatment_events > self.treatment_total:
                raise ValueError("treatment_events_exceed_total")
            if self.control_events > self.control_total:
                raise ValueError("control_events_exceed_total")

        if self.availability is not EffectAvailability.AVAILABLE and any(
            value is not None
            for value in (
                self.estimate,
                self.standard_error,
                self.variance,
                self.ci_lower,
                self.ci_upper,
                self.treatment_mean,
                self.treatment_events,
            )
        ):
            raise ValueError("unavailable_effect_cannot_supply_point_statistics")

        ratio_formats = {EffectFormat.ODDS_RATIO, EffectFormat.RISK_RATIO}
        if self.effect_format in ratio_formats:
            for value in (self.estimate, self.ci_lower, self.ci_upper):
                if value is not None and value <= 0:
                    raise ValueError("ratio_effect_values_must_be_positive")
        if self.effect_format is EffectFormat.MEAN_DIFFERENCE and self.availability is (
            EffectAvailability.AVAILABLE
        ):
            if self.unit is None:
                raise ValueError("mean_difference_requires_unit")
        elif self.effect_format is not EffectFormat.MEAN_DIFFERENCE and self.unit is not None:
            raise ValueError("only_mean_difference_accepts_unit")
        if (
            self.equivalence_conclusion
            in {
                EquivalenceConclusion.EQUIVALENT,
                EquivalenceConclusion.NOT_EQUIVALENT,
            }
            and self.equivalence_margin is None
        ):
            raise ValueError("equivalence_conclusion_requires_prespecified_margin")
        return self


class HarmonizedEffect(ContractModel):
    """An estimable effect and sampling variance on a canonical analysis scale."""

    paper_id: str
    finding_id: str
    outcome: str
    contrast: str
    measure: HarmonizedMeasure
    unit: str | None
    estimate: float
    variance: Annotated[float, Field(gt=0)]
    standard_error: Annotated[float, Field(gt=0)]
    point_direction: PointDirection
    reported_p_value: float | None
    reported_significance: ReportedSignificance
    equivalence_conclusion: EquivalenceConclusion
    equivalence_margin: float | None
    moderators: dict[str, str | int | float | bool | None]
    provenance: EffectProvenance
    derivation: Annotated[str, Field(min_length=1)]
    continuity_correction: Annotated[float, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def validate_harmonized_effect(self) -> HarmonizedEffect:
        if not all(math.isfinite(value) for value in (self.estimate, self.variance)):
            raise ValueError("harmonized_effect_values_must_be_finite")
        if not math.isclose(self.standard_error**2, self.variance, rel_tol=1e-10):
            raise ValueError("harmonized_standard_error_variance_mismatch")
        if self.point_direction is not _direction(self.estimate):
            raise ValueError("harmonized_point_direction_mismatch")
        if self.measure is HarmonizedMeasure.MEAN_DIFFERENCE:
            if self.unit is None:
                raise ValueError("harmonized_mean_difference_requires_unit")
        elif self.unit is not None:
            raise ValueError("harmonized_non_mean_difference_cannot_have_unit")
        return self


class HarmonizationResult(ContractModel):
    """Effect conversion result; insufficiency is data, not an exception."""

    status: Literal["estimable", "insufficient"]
    paper_id: str
    finding_id: str
    outcome: str
    contrast: str
    availability: EffectAvailability
    input_effect_format: EffectFormat
    point_direction: PointDirection
    reported_significance: ReportedSignificance
    equivalence_conclusion: EquivalenceConclusion
    reason: str | None
    effect: HarmonizedEffect | None
    provenance: EffectProvenance

    @model_validator(mode="after")
    def validate_result(self) -> HarmonizationResult:
        if self.status == "estimable" and (self.effect is None or self.reason is not None):
            raise ValueError("estimable_harmonization_requires_effect_only")
        if self.status == "insufficient" and (self.effect is not None or self.reason is None):
            raise ValueError("insufficient_harmonization_requires_reason_only")
        if self.effect is not None and (
            self.effect.paper_id != self.paper_id
            or self.effect.finding_id != self.finding_id
            or self.effect.outcome != self.outcome
            or self.effect.contrast != self.contrast
        ):
            raise ValueError("harmonization_effect_identity_mismatch")
        return self


def _direction(estimate: float | None) -> PointDirection:
    if estimate is None:
        return PointDirection.NOT_AVAILABLE
    if estimate > 0:
        return PointDirection.INCREASE
    if estimate < 0:
        return PointDirection.DECREASE
    return PointDirection.EXACT_ZERO


def _normal_se_from_interval(lower: float, upper: float, level: float) -> float:
    critical = NormalDist().inv_cdf(0.5 + level / 2)
    return (upper - lower) / (2 * critical)


def _hedges_correction(degrees_freedom: int) -> float:
    if degrees_freedom <= 1:
        raise EffectContractError("hedges_g_requires_at_least_four_total_participants")
    # Exact gamma-ratio correction, evaluated on the log scale for stability.
    return math.exp(
        math.lgamma(degrees_freedom / 2)
        - 0.5 * math.log(degrees_freedom / 2)
        - math.lgamma((degrees_freedom - 1) / 2)
    )


def _direct_uncertainty(evidence: EffectEvidence) -> tuple[float, str] | None:
    """Return variance on the source scale and its auditable derivation label."""

    if evidence.variance is not None:
        return evidence.variance, "reported_variance"
    if evidence.standard_error is not None:
        return evidence.standard_error**2, "reported_standard_error"
    if evidence.ci_lower is not None and evidence.ci_upper is not None:
        se = _normal_se_from_interval(evidence.ci_lower, evidence.ci_upper, evidence.ci_level)
        return se**2, f"normal_se_from_reported_{evidence.ci_level:g}_ci"
    return None


def _insufficient(
    evidence: EffectEvidence, *, reason: str, point_estimate: float | None = None
) -> HarmonizationResult:
    return HarmonizationResult(
        status="insufficient",
        paper_id=evidence.paper_id,
        finding_id=evidence.finding_id,
        outcome=evidence.outcome,
        contrast=evidence.contrast,
        availability=evidence.availability,
        input_effect_format=evidence.effect_format,
        point_direction=_direction(point_estimate),
        reported_significance=evidence.reported_significance,
        equivalence_conclusion=evidence.equivalence_conclusion,
        reason=reason,
        effect=None,
        provenance=evidence.provenance,
    )


def _estimable(
    evidence: EffectEvidence,
    *,
    measure: HarmonizedMeasure,
    estimate: float,
    variance: float,
    derivation: str,
    continuity_correction: float | None = None,
) -> HarmonizationResult:
    if not math.isfinite(estimate) or not math.isfinite(variance) or variance <= 0:
        raise EffectContractError("harmonized_effect_requires_finite_positive_variance")
    effect = HarmonizedEffect(
        paper_id=evidence.paper_id,
        finding_id=evidence.finding_id,
        outcome=evidence.outcome,
        contrast=evidence.contrast,
        measure=measure,
        unit=evidence.unit,
        estimate=estimate,
        variance=variance,
        standard_error=math.sqrt(variance),
        point_direction=_direction(estimate),
        reported_p_value=evidence.reported_p_value,
        reported_significance=evidence.reported_significance,
        equivalence_conclusion=evidence.equivalence_conclusion,
        equivalence_margin=evidence.equivalence_margin,
        moderators=evidence.moderators,
        provenance=evidence.provenance,
        derivation=derivation,
        continuity_correction=continuity_correction,
    )
    return HarmonizationResult(
        status="estimable",
        paper_id=evidence.paper_id,
        finding_id=evidence.finding_id,
        outcome=evidence.outcome,
        contrast=evidence.contrast,
        availability=evidence.availability,
        input_effect_format=evidence.effect_format,
        point_direction=effect.point_direction,
        reported_significance=evidence.reported_significance,
        equivalence_conclusion=evidence.equivalence_conclusion,
        reason=None,
        effect=effect,
        provenance=evidence.provenance,
    )


def _continuous_from_groups(evidence: EffectEvidence) -> tuple[float, float, str] | None:
    if evidence.treatment_mean is None:
        return None
    assert evidence.treatment_sd is not None
    assert evidence.treatment_n is not None
    assert evidence.control_mean is not None
    assert evidence.control_sd is not None
    assert evidence.control_n is not None
    difference = evidence.treatment_mean - evidence.control_mean
    if evidence.effect_format is EffectFormat.MEAN_DIFFERENCE:
        variance = (
            evidence.treatment_sd**2 / evidence.treatment_n
            + evidence.control_sd**2 / evidence.control_n
        )
        return difference, variance, "mean_difference_from_group_summaries"
    if evidence.effect_format not in {EffectFormat.COHENS_D, EffectFormat.HEDGES_G}:
        raise EffectContractError("continuous_summaries_require_continuous_effect_format")
    degrees_freedom = evidence.treatment_n + evidence.control_n - 2
    pooled_variance = (
        (evidence.treatment_n - 1) * evidence.treatment_sd**2
        + (evidence.control_n - 1) * evidence.control_sd**2
    ) / degrees_freedom
    if pooled_variance <= 0:
        raise EffectContractError("standardized_effect_requires_positive_pooled_variance")
    cohens_d = difference / math.sqrt(pooled_variance)
    correction = _hedges_correction(degrees_freedom)
    hedges_g = correction * cohens_d
    variance_d = (evidence.treatment_n + evidence.control_n) / (
        evidence.treatment_n * evidence.control_n
    ) + cohens_d**2 / (2 * degrees_freedom)
    return hedges_g, correction**2 * variance_d, "hedges_g_from_group_summaries"


def _binary_from_counts(
    evidence: EffectEvidence,
) -> tuple[float, float, HarmonizedMeasure, str, float | None] | None:
    if evidence.treatment_events is None:
        return None
    assert evidence.treatment_total is not None
    assert evidence.control_events is not None
    assert evidence.control_total is not None
    a = float(evidence.treatment_events)
    b = float(evidence.treatment_total - evidence.treatment_events)
    c = float(evidence.control_events)
    d = float(evidence.control_total - evidence.control_events)
    is_odds_ratio = evidence.effect_format in {
        EffectFormat.ODDS_RATIO,
        EffectFormat.LOG_ODDS_RATIO,
    }
    is_risk_ratio = evidence.effect_format in {
        EffectFormat.RISK_RATIO,
        EffectFormat.LOG_RISK_RATIO,
    }
    # Odds require all four cells to be positive. Risks require only the two event
    # counts to be positive; correcting a zero *non-event* cell changes an otherwise
    # estimable risk ratio (for example, 10/10 versus 5/10).
    correction = (
        0.5
        if (is_odds_ratio and min(a, b, c, d) == 0) or (is_risk_ratio and min(a, c) == 0)
        else 0.0
    )
    if correction:
        a, b, c, d = (cell + correction for cell in (a, b, c, d))
    if is_odds_ratio:
        estimate = math.log((a * d) / (b * c))
        variance = 1 / a + 1 / b + 1 / c + 1 / d
        return (
            estimate,
            variance,
            HarmonizedMeasure.LOG_ODDS_RATIO,
            "log_odds_ratio_from_2x2_counts",
            correction or None,
        )
    if is_risk_ratio:
        treatment_total = a + b
        control_total = c + d
        estimate = math.log((a / treatment_total) / (c / control_total))
        variance = 1 / a - 1 / treatment_total + 1 / c - 1 / control_total
        return (
            estimate,
            variance,
            HarmonizedMeasure.LOG_RISK_RATIO,
            "log_risk_ratio_from_2x2_counts",
            correction or None,
        )
    raise EffectContractError("binary_counts_require_binary_effect_format")


def _harmonize_direct(evidence: EffectEvidence) -> HarmonizationResult:
    assert evidence.estimate is not None
    uncertainty = _direct_uncertainty(evidence)
    point_estimate = evidence.estimate
    if uncertainty is None:
        if evidence.effect_format in {EffectFormat.ODDS_RATIO, EffectFormat.RISK_RATIO}:
            point_estimate = math.log(evidence.estimate)
        return _insufficient(
            evidence,
            reason="sampling_uncertainty_not_reported",
            point_estimate=point_estimate,
        )
    variance, uncertainty_derivation = uncertainty
    match evidence.effect_format:
        case EffectFormat.MEAN_DIFFERENCE:
            return _estimable(
                evidence,
                measure=HarmonizedMeasure.MEAN_DIFFERENCE,
                estimate=evidence.estimate,
                variance=variance,
                derivation=f"direct_mean_difference:{uncertainty_derivation}",
            )
        case EffectFormat.COHENS_D:
            if evidence.treatment_n is None or evidence.control_n is None:
                return _insufficient(
                    evidence,
                    reason="cohens_d_requires_group_sizes_for_bias_correction",
                    point_estimate=evidence.estimate,
                )
            correction = _hedges_correction(evidence.treatment_n + evidence.control_n - 2)
            return _estimable(
                evidence,
                measure=HarmonizedMeasure.STANDARDIZED_MEAN_DIFFERENCE,
                estimate=correction * evidence.estimate,
                variance=correction**2 * variance,
                derivation=f"cohens_d_to_hedges_g:{uncertainty_derivation}",
            )
        case EffectFormat.HEDGES_G:
            return _estimable(
                evidence,
                measure=HarmonizedMeasure.STANDARDIZED_MEAN_DIFFERENCE,
                estimate=evidence.estimate,
                variance=variance,
                derivation=f"direct_hedges_g:{uncertainty_derivation}",
            )
        case EffectFormat.ODDS_RATIO | EffectFormat.RISK_RATIO:
            measure = (
                HarmonizedMeasure.LOG_ODDS_RATIO
                if evidence.effect_format is EffectFormat.ODDS_RATIO
                else HarmonizedMeasure.LOG_RISK_RATIO
            )
            if evidence.ci_lower is not None and evidence.ci_upper is not None:
                critical = NormalDist().inv_cdf(0.5 + evidence.ci_level / 2)
                log_standard_error = (math.log(evidence.ci_upper) - math.log(evidence.ci_lower)) / (
                    2 * critical
                )
                log_variance = log_standard_error**2
                log_derivation = f"log_se_from_reported_{evidence.ci_level:g}_ratio_ci"
            else:
                # Delta-method conversion when SE/variance is reported on the ratio scale.
                log_variance = variance / evidence.estimate**2
                log_derivation = f"ratio_to_log_scale_delta_method:{uncertainty_derivation}"
            return _estimable(
                evidence,
                measure=measure,
                estimate=math.log(evidence.estimate),
                variance=log_variance,
                derivation=log_derivation,
            )
        case EffectFormat.LOG_ODDS_RATIO | EffectFormat.LOG_RISK_RATIO:
            measure = (
                HarmonizedMeasure.LOG_ODDS_RATIO
                if evidence.effect_format is EffectFormat.LOG_ODDS_RATIO
                else HarmonizedMeasure.LOG_RISK_RATIO
            )
            return _estimable(
                evidence,
                measure=measure,
                estimate=evidence.estimate,
                variance=variance,
                derivation=f"direct_log_ratio:{uncertainty_derivation}",
            )
        case EffectFormat.UNSPECIFIED:
            return _insufficient(
                evidence,
                reason="effect_format_unspecified",
                point_estimate=evidence.estimate,
            )


def harmonize_effect(evidence: EffectEvidence) -> HarmonizationResult:
    """Convert one evidence record to a canonical estimate/variance when possible.

    Expected absence of information returns ``status='insufficient'``. Impossible
    values and incompatible source formats raise :class:`EffectContractError`.
    """

    if evidence.availability is EffectAvailability.MISSING:
        return _insufficient(evidence, reason="effect_not_reported")
    if evidence.availability is EffectAvailability.INCONCLUSIVE:
        return _insufficient(evidence, reason="effect_report_inconclusive")

    if evidence.treatment_events is not None:
        assert evidence.treatment_total is not None
        assert evidence.control_events is not None
        assert evidence.control_total is not None
        if evidence.treatment_events == 0 and evidence.control_events == 0:
            return _insufficient(
                evidence,
                reason="binary_double_zero_event_study_noninformative",
            )
        if (
            evidence.treatment_events == evidence.treatment_total
            and evidence.control_events == evidence.control_total
        ):
            return _insufficient(
                evidence,
                reason="binary_double_all_event_study_noninformative",
            )

    binary = _binary_from_counts(evidence)
    if binary is not None:
        estimate, variance, measure, derivation, correction = binary
        return _estimable(
            evidence,
            measure=measure,
            estimate=estimate,
            variance=variance,
            derivation=derivation,
            continuity_correction=correction,
        )

    continuous = _continuous_from_groups(evidence)
    if continuous is not None:
        estimate, variance, derivation = continuous
        measure = (
            HarmonizedMeasure.MEAN_DIFFERENCE
            if evidence.effect_format is EffectFormat.MEAN_DIFFERENCE
            else HarmonizedMeasure.STANDARDIZED_MEAN_DIFFERENCE
        )
        return _estimable(
            evidence,
            measure=measure,
            estimate=estimate,
            variance=variance,
            derivation=derivation,
        )

    if evidence.estimate is not None:
        return _harmonize_direct(evidence)
    return _insufficient(evidence, reason="point_estimate_not_reported")


def harmonize_effects(evidence: list[EffectEvidence]) -> list[HarmonizationResult]:
    """Harmonize records independently while preserving their input order."""

    return [harmonize_effect(item) for item in evidence]


def effect_evidence_json_schema() -> dict[str, Any]:
    """Return the closed JSON Schema that harvest/extraction adapters can target."""

    schema = EffectEvidence.model_json_schema(mode="validation")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "urn:literature-multiverse:effect-evidence:v1"
    schema["title"] = "Literature Multiverse numerical effect evidence"
    return schema
