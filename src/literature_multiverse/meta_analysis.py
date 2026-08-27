"""Paper-clustered quantitative synthesis and sign-only fallback.

All inferential functions first reduce correlated within-paper records to one effect per
paper.  The default assumes perfect unknown within-paper correlation, so duplicate
outcomes never create artificial precision.  Categorical meta-regression is explicitly
predictive: its output describes corpus associations and contains no causal estimand.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import binomtest, chi2, f, t

from literature_multiverse.effects import (
    EquivalenceConclusion,
    HarmonizationResult,
    HarmonizedEffect,
    HarmonizedMeasure,
    PointDirection,
    ReportedSignificance,
    harmonize_effects,
)
from literature_multiverse.evidence_graph import (
    EvidenceGraph,
    graph_risk_features,
    select_effect_evidence,
)
from literature_multiverse.models import ContractModel


class MetaAnalysisContractError(ValueError):
    """Synthesis inputs violate an explicit statistical contract."""


class PaperEffect(ContractModel):
    """Exactly one conservative effect contribution from one source paper."""

    paper_id: str
    outcome: str
    contrast: str
    measure: HarmonizedMeasure
    unit: str | None
    estimate: float
    variance: Annotated[float, Field(gt=0)]
    point_direction: PointDirection
    source_finding_ids: list[str]
    source_locators: list[str]
    moderators: dict[str, str | int | float | bool | None]
    moderator_conflicts: list[str]
    reported_significance: list[ReportedSignificance]
    equivalence_conclusions: list[EquivalenceConclusion]
    aggregation_method: str
    assumed_within_paper_correlation: Annotated[float, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def validate_paper_effect(self) -> PaperEffect:
        if not math.isfinite(self.estimate) or not math.isfinite(self.variance):
            raise ValueError("paper_effect_values_must_be_finite")
        if self.point_direction is not _point_direction(self.estimate):
            raise ValueError("paper_effect_direction_mismatch")
        if self.source_finding_ids != sorted(set(self.source_finding_ids)):
            raise ValueError("paper_effect_finding_ids_must_be_sorted_unique")
        if self.source_locators != sorted(set(self.source_locators)):
            raise ValueError("paper_effect_source_locators_must_be_sorted_unique")
        return self


class PaperAggregationResult(ContractModel):
    """Typed output from the required one-effect-per-paper reduction."""

    status: Literal["ok", "insufficient"]
    reason: str | None
    n_effects_input: Annotated[int, Field(ge=0)]
    n_papers: Annotated[int, Field(ge=0)]
    effects: list[PaperEffect]

    @model_validator(mode="after")
    def validate_result(self) -> PaperAggregationResult:
        if self.status == "ok" and (self.reason is not None or not self.effects):
            raise ValueError("successful_paper_aggregation_requires_effects_only")
        if self.status == "insufficient" and (self.reason is None or self.effects):
            raise ValueError("insufficient_paper_aggregation_requires_reason_only")
        if self.n_papers != len(self.effects):
            raise ValueError("paper_aggregation_count_mismatch")
        return self


def _point_direction(estimate: float) -> PointDirection:
    if estimate > 0:
        return PointDirection.INCREASE
    if estimate < 0:
        return PointDirection.DECREASE
    return PointDirection.EXACT_ZERO


def _insufficient_aggregation(
    effects: Sequence[HarmonizedEffect], reason: str
) -> PaperAggregationResult:
    return PaperAggregationResult(
        status="insufficient",
        reason=reason,
        n_effects_input=len(effects),
        n_papers=0,
        effects=[],
    )


def aggregate_one_effect_per_paper(
    effects: Sequence[HarmonizedEffect],
    *,
    assumed_within_paper_correlation: float = 1.0,
) -> PaperAggregationResult:
    """Reduce multiple compatible rows to one equally weighted effect per paper.

    For ``m`` effects with variances ``v_i``, the variance of their arithmetic mean
    includes covariance ``rho * sqrt(v_i v_j)``.  The default ``rho=1`` is conservative
    when the within-paper covariance is unavailable and ensures duplicates add no
    precision.  Callers must justify any smaller prespecified value.
    """

    if not 0 <= assumed_within_paper_correlation <= 1:
        raise MetaAnalysisContractError("within_paper_correlation_must_be_in_unit_interval")
    if not effects:
        return _insufficient_aggregation(effects, "no_estimable_effects")
    finding_ids = [effect.finding_id for effect in effects]
    if len(finding_ids) != len(set(finding_ids)):
        raise MetaAnalysisContractError("duplicate_finding_id")
    signatures = {
        (effect.outcome, effect.contrast, effect.measure, effect.unit) for effect in effects
    }
    if len(signatures) != 1:
        outcomes = {effect.outcome for effect in effects}
        contrasts = {effect.contrast for effect in effects}
        measures = {(effect.measure, effect.unit) for effect in effects}
        if len(outcomes) > 1:
            reason = "incompatible_outcomes"
        elif len(contrasts) > 1:
            reason = "incompatible_contrasts"
        elif len(measures) > 1:
            reason = "incompatible_effect_scales"
        else:  # pragma: no cover - exhaustive guard for future signature fields
            reason = "incompatible_effects"
        return _insufficient_aggregation(effects, reason)

    grouped: dict[str, list[HarmonizedEffect]] = defaultdict(list)
    for effect in effects:
        if not effect.paper_id:
            raise MetaAnalysisContractError("effect_requires_paper_id")
        grouped[effect.paper_id].append(effect)

    paper_effects: list[PaperEffect] = []
    for paper_id in sorted(grouped):
        rows = grouped[paper_id]
        count = len(rows)
        estimate = sum(row.estimate for row in rows) / count
        variances = [row.variance for row in rows]
        covariance_sum = sum(variances)
        covariance_sum += 2 * assumed_within_paper_correlation * sum(
            math.sqrt(variances[left] * variances[right])
            for left in range(count)
            for right in range(left + 1, count)
        )
        variance = covariance_sum / count**2

        moderator_names = sorted({name for row in rows for name in row.moderators})
        moderators: dict[str, str | int | float | bool | None] = {}
        conflicts: list[str] = []
        for name in moderator_names:
            observed = {
                row.moderators.get(name)
                for row in rows
                if row.moderators.get(name) is not None
            }
            if len(observed) > 1:
                conflicts.append(name)
                moderators[name] = None
            elif observed:
                moderators[name] = next(iter(observed))
            else:
                moderators[name] = None

        paper_effects.append(
            PaperEffect(
                paper_id=paper_id,
                outcome=rows[0].outcome,
                contrast=rows[0].contrast,
                measure=rows[0].measure,
                unit=rows[0].unit,
                estimate=estimate,
                variance=variance,
                point_direction=_point_direction(estimate),
                source_finding_ids=sorted({row.finding_id for row in rows}),
                source_locators=sorted({row.provenance.source_locator for row in rows}),
                moderators=moderators,
                moderator_conflicts=conflicts,
                reported_significance=sorted(
                    {row.reported_significance for row in rows}, key=str
                ),
                equivalence_conclusions=sorted(
                    {row.equivalence_conclusion for row in rows}, key=str
                ),
                aggregation_method="equal_mean_with_prespecified_common_correlation",
                assumed_within_paper_correlation=assumed_within_paper_correlation,
            )
        )
    return PaperAggregationResult(
        status="ok",
        reason=None,
        n_effects_input=len(effects),
        n_papers=len(paper_effects),
        effects=paper_effects,
    )


def _weighted_fit(
    y: np.ndarray, variances: np.ndarray, design: np.ndarray, tau_squared: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    weights = 1 / (variances + tau_squared)
    information = design.T @ (weights[:, None] * design)
    if np.linalg.matrix_rank(information) < design.shape[1]:
        raise MetaAnalysisContractError("meta_regression_design_is_rank_deficient")
    covariance = np.linalg.inv(information)
    beta = covariance @ (design.T @ (weights * y))
    residuals = y - design @ beta
    q_value = float(np.sum(weights * residuals**2))
    return beta, covariance, weights, q_value


def _paule_mandel_tau_squared(
    y: np.ndarray, variances: np.ndarray, design: np.ndarray
) -> float:
    """Generalized Paule-Mandel tau²: solve weighted residual Q(tau²)=df."""

    degrees_freedom = len(y) - design.shape[1]
    if degrees_freedom <= 0:
        raise MetaAnalysisContractError("tau_squared_requires_positive_residual_df")
    _, _, _, q_zero = _weighted_fit(y, variances, design, 0.0)
    if q_zero <= degrees_freedom:
        return 0.0
    high = max(float(np.var(y, ddof=1)), float(np.median(variances)), 1e-8)
    for _ in range(80):
        _, _, _, q_high = _weighted_fit(y, variances, design, high)
        if q_high <= degrees_freedom:
            break
        high *= 2
    else:
        raise MetaAnalysisContractError("paule_mandel_tau_squared_failed_to_bracket")
    low = 0.0
    for _ in range(100):
        middle = (low + high) / 2
        _, _, _, q_middle = _weighted_fit(y, variances, design, middle)
        if q_middle > degrees_freedom:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _ratio_scale(
    measure: HarmonizedMeasure, estimate: float, lower: float, upper: float
) -> dict[str, float] | None:
    if measure not in {HarmonizedMeasure.LOG_ODDS_RATIO, HarmonizedMeasure.LOG_RISK_RATIO}:
        return None
    return {
        "estimate": math.exp(estimate),
        "ci_lower": math.exp(lower),
        "ci_upper": math.exp(upper),
    }


def _input_provenance(effects: Sequence[HarmonizedEffect]) -> dict[str, list[dict[str, str]]]:
    by_paper: dict[str, list[dict[str, str]]] = defaultdict(list)
    for effect in effects:
        by_paper[effect.paper_id].append(
            {
                "finding_id": effect.finding_id,
                "source_locator": effect.provenance.source_locator,
            }
        )
    return {
        paper_id: sorted(rows, key=lambda row: (row["finding_id"], row["source_locator"]))
        for paper_id, rows in sorted(by_paper.items())
    }


def random_effects_meta_analysis(
    effects: Sequence[HarmonizedEffect],
    *,
    confidence_level: float = 0.95,
    assumed_within_paper_correlation: float = 1.0,
) -> dict[str, Any]:
    """Pool one compatible outcome using PM tau² and modified Knapp-Hartung CI.

    Expected insufficiency is returned with ``status='insufficient'`` and a stable
    reason code.  The output includes a prediction interval, heterogeneity measures,
    and source-finding provenance for every paper contribution.
    """

    if not 0 < confidence_level < 1:
        raise MetaAnalysisContractError("confidence_level_must_be_between_zero_and_one")
    aggregation = aggregate_one_effect_per_paper(
        effects,
        assumed_within_paper_correlation=assumed_within_paper_correlation,
    )
    base: dict[str, Any] = {
        "method": {
            "model": "random_effects",
            "tau_squared": "generalized_paule_mandel",
            "uncertainty": "modified_knapp_hartung",
            "cluster_handling": "one_conservative_aggregate_per_paper",
            "confidence_level": confidence_level,
            "assumed_within_paper_correlation": assumed_within_paper_correlation,
        },
        "n_effects_input": len(effects),
        "n_papers": aggregation.n_papers,
        "input_provenance": _input_provenance(effects),
    }
    if aggregation.status == "insufficient":
        return {**base, "status": "insufficient", "reason": aggregation.reason}
    if aggregation.n_papers < 2:
        return {**base, "status": "insufficient", "reason": "fewer_than_two_papers"}

    paper_effects = aggregation.effects
    y = np.asarray([effect.estimate for effect in paper_effects], dtype=float)
    variances = np.asarray([effect.variance for effect in paper_effects], dtype=float)
    design = np.ones((len(y), 1), dtype=float)
    tau_squared = _paule_mandel_tau_squared(y, variances, design)
    beta, covariance, _, residual_q = _weighted_fit(y, variances, design, tau_squared)
    degrees_freedom = len(y) - 1
    # The max(1, q/df) modification prevents spuriously narrower intervals when q<df.
    scale = max(1.0, residual_q / degrees_freedom)
    standard_error = math.sqrt(float(covariance[0, 0]) * scale)
    critical = float(t.ppf(0.5 + confidence_level / 2, degrees_freedom))
    estimate = float(beta[0])
    lower = estimate - critical * standard_error
    upper = estimate + critical * standard_error
    measure = paper_effects[0].measure
    if len(y) >= 3:
        prediction_standard_error = math.sqrt(tau_squared + standard_error**2)
        prediction_critical = float(t.ppf(0.5 + confidence_level / 2, len(y) - 2))
        prediction_lower = estimate - prediction_critical * prediction_standard_error
        prediction_upper = estimate + prediction_critical * prediction_standard_error
        prediction_interval: dict[str, Any] = {
            "status": "ok",
            "lower": prediction_lower,
            "upper": prediction_upper,
            "degrees_freedom": len(y) - 2,
            "ratio_scale": _ratio_scale(
                measure, estimate, prediction_lower, prediction_upper
            ),
        }
    else:
        prediction_interval = {
            "status": "insufficient",
            "reason": "fewer_than_three_papers",
            "lower": None,
            "upper": None,
            "degrees_freedom": 0,
            "ratio_scale": None,
        }

    fixed_beta, _, _, fixed_q = _weighted_fit(y, variances, design, 0.0)
    del fixed_beta
    i_squared = max(0.0, (fixed_q - degrees_freedom) / fixed_q) if fixed_q > 0 else 0.0
    return {
        **base,
        "status": "ok",
        "reason": None,
        "outcome": paper_effects[0].outcome,
        "contrast": paper_effects[0].contrast,
        "measure": measure.value,
        "unit": paper_effects[0].unit,
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_lower": lower,
        "ci_upper": upper,
        "point_direction": _point_direction(estimate).value,
        "two_sided_p_value": float(2 * t.sf(abs(estimate / standard_error), degrees_freedom)),
        "ratio_scale": _ratio_scale(measure, estimate, lower, upper),
        "prediction_interval": prediction_interval,
        "heterogeneity": {
            "tau_squared": tau_squared,
            "cochran_q": fixed_q,
            "q_degrees_freedom": degrees_freedom,
            "q_p_value": float(chi2.sf(fixed_q, degrees_freedom)),
            "i_squared": i_squared,
            "i_squared_percent": 100 * i_squared,
        },
        "paper_effects": [effect.model_dump(mode="json") for effect in paper_effects],
        "provenance": {
            effect.paper_id: {
                "finding_ids": effect.source_finding_ids,
                "source_locators": effect.source_locators,
            }
            for effect in paper_effects
        },
    }


def categorical_meta_regression(
    effects: Sequence[HarmonizedEffect],
    moderator_name: str,
    *,
    reference_level: str | None = None,
    confidence_level: float = 0.95,
    min_papers_per_level: int = 2,
    assumed_within_paper_correlation: float = 1.0,
) -> dict[str, Any]:
    """Fit a paper-grouped categorical random-effects meta-regression.

    Coefficients are predictive associations with the recorded moderator, not causal
    effects.  A moderator that varies among effects from one paper is rejected rather
    than duplicated across a nominally grouped design.
    """

    if not moderator_name.strip():
        raise MetaAnalysisContractError("moderator_name_must_be_nonempty")
    if min_papers_per_level < 2:
        raise MetaAnalysisContractError("min_papers_per_level_must_be_at_least_two")
    if not 0 < confidence_level < 1:
        raise MetaAnalysisContractError("confidence_level_must_be_between_zero_and_one")
    aggregation = aggregate_one_effect_per_paper(
        effects,
        assumed_within_paper_correlation=assumed_within_paper_correlation,
    )
    base: dict[str, Any] = {
        "moderator": moderator_name,
        "interpretation": "predictive_association_not_causal",
        "method": {
            "model": "categorical_random_effects_meta_regression",
            "tau_squared": "generalized_paule_mandel",
            "uncertainty": "modified_knapp_hartung",
            "cluster_handling": "one_conservative_aggregate_per_paper",
            "confidence_level": confidence_level,
            "assumed_within_paper_correlation": assumed_within_paper_correlation,
        },
        "n_effects_input": len(effects),
        "n_papers_before_moderator_filter": aggregation.n_papers,
        "input_provenance": _input_provenance(effects),
    }
    if aggregation.status == "insufficient":
        return {**base, "status": "insufficient", "reason": aggregation.reason}
    conflicted = [
        effect.paper_id
        for effect in aggregation.effects
        if moderator_name in effect.moderator_conflicts
    ]
    if conflicted:
        return {
            **base,
            "status": "insufficient",
            "reason": "moderator_varies_within_paper",
            "conflicted_paper_ids": conflicted,
        }
    included = [
        effect
        for effect in aggregation.effects
        if effect.moderators.get(moderator_name) is not None
        and str(effect.moderators[moderator_name]).strip()
    ]
    included_paper_ids = {effect.paper_id for effect in included}
    missing = sorted(
        effect.paper_id
        for effect in aggregation.effects
        if effect.paper_id not in included_paper_ids
    )
    level_by_paper = {
        effect.paper_id: str(effect.moderators[moderator_name]) for effect in included
    }
    support = Counter(level_by_paper.values())
    common_base = {
        **base,
        "n_papers": len(included),
        "excluded_missing_moderator_paper_ids": missing,
        "support_by_level": dict(sorted(support.items())),
    }
    if len(support) < 2:
        return {
            **common_base,
            "status": "insufficient",
            "reason": "fewer_than_two_moderator_levels",
        }
    sparse = sorted(level for level, count in support.items() if count < min_papers_per_level)
    if sparse:
        return {
            **common_base,
            "status": "insufficient",
            "reason": "level_has_too_few_papers",
            "sparse_levels": sparse,
        }
    levels = sorted(support)
    reference = reference_level or levels[0]
    if reference not in support:
        return {
            **common_base,
            "status": "insufficient",
            "reason": "reference_level_not_observed",
            "requested_reference_level": reference,
        }
    comparison_levels = [level for level in levels if level != reference]
    design = np.asarray(
        [
            [1.0, *(float(level_by_paper[effect.paper_id] == level) for level in comparison_levels)]
            for effect in included
        ],
        dtype=float,
    )
    residual_degrees_freedom = len(included) - design.shape[1]
    if residual_degrees_freedom < 2:
        return {
            **common_base,
            "status": "insufficient",
            "reason": "fewer_than_two_residual_degrees_of_freedom",
        }

    y = np.asarray([effect.estimate for effect in included], dtype=float)
    variances = np.asarray([effect.variance for effect in included], dtype=float)
    tau_squared = _paule_mandel_tau_squared(y, variances, design)
    beta, model_covariance, _, residual_q = _weighted_fit(y, variances, design, tau_squared)
    scale = max(1.0, residual_q / residual_degrees_freedom)
    covariance = model_covariance * scale
    critical = float(t.ppf(0.5 + confidence_level / 2, residual_degrees_freedom))

    coefficients: list[dict[str, Any]] = []
    for index, level in enumerate(comparison_levels, start=1):
        standard_error = math.sqrt(float(covariance[index, index]))
        estimate = float(beta[index])
        coefficients.append(
            {
                "level": level,
                "reference_level": reference,
                "estimate_difference": estimate,
                "standard_error": standard_error,
                "ci_lower": estimate - critical * standard_error,
                "ci_upper": estimate + critical * standard_error,
                "two_sided_p_value": float(
                    2 * t.sf(abs(estimate / standard_error), residual_degrees_freedom)
                ),
            }
        )

    level_estimates: list[dict[str, Any]] = []
    for level in levels:
        contrast = np.asarray(
            [1.0, *(float(level == item) for item in comparison_levels)], dtype=float
        )
        estimate = float(contrast @ beta)
        standard_error = math.sqrt(float(contrast @ covariance @ contrast))
        level_estimates.append(
            {
                "level": level,
                "estimate": estimate,
                "standard_error": standard_error,
                "ci_lower": estimate - critical * standard_error,
                "ci_upper": estimate + critical * standard_error,
            }
        )

    moderator_beta = beta[1:]
    moderator_covariance = covariance[1:, 1:]
    omnibus_wald = float(moderator_beta @ np.linalg.inv(moderator_covariance) @ moderator_beta)
    numerator_df = len(comparison_levels)
    omnibus_f = omnibus_wald / numerator_df
    return {
        **common_base,
        "status": "ok",
        "reason": None,
        "outcome": included[0].outcome,
        "contrast": included[0].contrast,
        "measure": included[0].measure.value,
        "unit": included[0].unit,
        "reference_level": reference,
        "coefficients": coefficients,
        "level_estimates": level_estimates,
        "omnibus": {
            "f_statistic": omnibus_f,
            "numerator_degrees_freedom": numerator_df,
            "denominator_degrees_freedom": residual_degrees_freedom,
            "p_value": float(f.sf(omnibus_f, numerator_df, residual_degrees_freedom)),
        },
        "heterogeneity": {
            "residual_tau_squared": tau_squared,
            "residual_q": residual_q,
            "residual_degrees_freedom": residual_degrees_freedom,
        },
        "paper_effects": [effect.model_dump(mode="json") for effect in included],
        "provenance": {
            effect.paper_id: {
                "finding_ids": effect.source_finding_ids,
                "source_locators": effect.source_locators,
                "moderator_level": level_by_paper[effect.paper_id],
            }
            for effect in included
        },
    }


def directional_synthesis(results: Sequence[HarmonizationResult]) -> dict[str, Any]:
    """Paper-balanced sign synthesis when magnitudes cannot be combined.

    The function uses only point-estimate signs.  Reported non-significance and
    equivalence conclusions are retained in the source records but never turned into
    an exact zero.  Conflicting signs within a paper become ``mixed``.
    """

    by_paper: dict[str, list[HarmonizationResult]] = defaultdict(list)
    for result in results:
        by_paper[result.paper_id].append(result)
    paper_directions: dict[str, str] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for paper_id in sorted(by_paper):
        rows = by_paper[paper_id]
        observed = {
            row.point_direction
            for row in rows
            if row.point_direction is not PointDirection.NOT_AVAILABLE
        }
        if not observed:
            direction = "unavailable"
        elif len(observed) == 1:
            direction = next(iter(observed)).value
        else:
            direction = "mixed"
        paper_directions[paper_id] = direction
        provenance[paper_id] = {
            "finding_ids": sorted({row.finding_id for row in rows}),
            "source_locators": sorted({row.provenance.source_locator for row in rows}),
            "reported_significance": sorted(
                {row.reported_significance.value for row in rows}
            ),
            "equivalence_conclusions": sorted(
                {row.equivalence_conclusion.value for row in rows}
            ),
        }
    categories = ("increase", "decrease", "exact_zero", "mixed", "unavailable")
    counts = Counter(paper_directions.values())
    count_output = {category: counts[category] for category in categories}
    nonzero = counts["increase"] + counts["decrease"]
    targets = {(result.outcome, result.contrast) for result in results}
    base: dict[str, Any] = {
        "method": "paper_level_point_estimate_signs_only",
        "limitations": [
            "effect_magnitudes_and_precision_are_not_combined",
            "reported_non_significance_is_not_evidence_of_zero_effect",
            "exact_zero_requires_a_literal_zero_point_estimate",
        ],
        "n_effects_input": len(results),
        "n_papers": len(by_paper),
        "paper_direction_counts": count_output,
        "paper_directions": paper_directions,
        "provenance": provenance,
    }
    if len(targets) > 1:
        return {
            **base,
            "status": "insufficient",
            "reason": "incompatible_directional_targets",
            "observed_targets": [
                {"outcome": outcome, "contrast": contrast}
                for outcome, contrast in sorted(targets)
            ],
        }
    if targets:
        outcome, contrast = next(iter(targets))
        base.update({"outcome": outcome, "contrast": contrast})
    if nonzero < 2:
        return {
            **base,
            "status": "insufficient",
            "reason": "fewer_than_two_nonzero_paper_signs",
        }
    sign_test = binomtest(counts["increase"], nonzero, p=0.5, alternative="two-sided")
    interval = sign_test.proportion_ci(confidence_level=0.95, method="exact")
    return {
        **base,
        "status": "ok",
        "reason": None,
        "increase_fraction_among_nonzero": counts["increase"] / nonzero,
        "increase_fraction_exact_ci_95": [float(interval.low), float(interval.high)],
        "two_sided_exact_sign_test_p_value": float(sign_test.pvalue),
    }


def synthesize_with_directional_fallback(
    results: Sequence[HarmonizationResult],
    *,
    confidence_level: float = 0.95,
    assumed_within_paper_correlation: float = 1.0,
) -> dict[str, Any]:
    """Attempt magnitude synthesis, then expose an explicit sign-only fallback."""

    estimable = [result.effect for result in results if result.effect is not None]
    quantitative = random_effects_meta_analysis(
        estimable,
        confidence_level=confidence_level,
        assumed_within_paper_correlation=assumed_within_paper_correlation,
    )
    if quantitative["status"] == "ok":
        return {
            "status": "ok",
            "mode": "random_effects_meta_analysis",
            "quantitative": quantitative,
            "directional_fallback": None,
        }
    directional = directional_synthesis(results)
    return {
        "status": directional["status"],
        "mode": "directional_sign_synthesis" if directional["status"] == "ok" else "insufficient",
        "quantitative": quantitative,
        "directional_fallback": directional,
    }


def synthesize_evidence_graph(
    graph: EvidenceGraph,
    *,
    outcome_name: str | None = None,
    contrast_id: str | None = None,
    require_explicit_timepoint: bool = True,
    confidence_level: float = 0.95,
    assumed_within_paper_correlation: float = 1.0,
) -> dict[str, Any]:
    """Conservatively bridge the typed graph to the current synthesis engine.

    The existing estimator clusters by publication, whereas scientific independence is
    represented by cohort identity in :class:`EvidenceGraph`.  The bridge proceeds only
    when the selected graph has a one-to-one cohort/publication mapping.  Multi-report or
    multi-cohort publications return an explicit insufficiency result until a cohort-aware
    hierarchical estimator is implemented.
    """

    risk_features = graph_risk_features(
        graph,
        outcome_name=outcome_name,
        contrast_id=contrast_id,
    ).model_dump(mode="json")
    selection = select_effect_evidence(
        graph,
        outcome_name=outcome_name,
        contrast_id=contrast_id,
        require_explicit_timepoint=require_explicit_timepoint,
    )
    graph_contract = {
        "graph_schema_version": graph.graph_schema_version,
        "selection_status": selection.status,
        "selection_reason": selection.reason,
        "selected_estimate_ids": selection.estimate_ids,
        "warnings": selection.warnings,
        "risk_features": risk_features,
        "risk_feature_interpretation": (
            "prospective_label_free_inputs_not_a_calibrated_error_probability"
        ),
    }
    if selection.status == "insufficient":
        return {
            "status": "insufficient",
            "mode": "evidence_graph_contract",
            "reason": selection.reason,
            "quantitative": None,
            "directional_fallback": None,
            "evidence_graph": graph_contract,
        }
    results = harmonize_effects(selection.records)
    synthesis = synthesize_with_directional_fallback(
        results,
        confidence_level=confidence_level,
        assumed_within_paper_correlation=assumed_within_paper_correlation,
    )
    return {**synthesis, "evidence_graph": graph_contract}
