"""Paper-clustered quantitative synthesis and sign-only fallback.

All inferential functions first reduce correlated within-paper records to one effect per
paper.  The default assumes perfect unknown within-paper correlation, so duplicate
outcomes never create artificial precision.  Categorical meta-regression is explicitly
predictive: its output describes corpus associations and contains no causal estimand.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import binomtest, chi2, f, t

from literature_multiverse.budgeted_verification import (
    AuditCandidate,
    ClaimModel,
    ProbabilityBasis,
    ScenarioKind,
)
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


class SynthesisDecisionSnapshot(ContractModel):
    """A bounded decision-support score plus the explicit release margin.

    ``decision_score`` is a directional evidence score derived from the frequentist
    synthesis (or sign fraction); it is not a posterior probability that the claim is
    scientifically true.  ``supported`` follows the confidence/prediction-interval
    rule exactly, so audit influence does not substitute a probabilistic interpretation
    for that prespecified decision boundary.
    """

    target_direction: Literal["increase", "decrease"]
    classification: Literal[
        "supported",
        "contradicted",
        "condition_dependent",
        "inconclusive",
        "not_evaluable",
    ]
    supported: bool
    decision_score: Annotated[float, Field(ge=0, le=1)]
    decision_margin: float | None = None
    mode: str
    reason: str

    @model_validator(mode="after")
    def validate_snapshot(self) -> SynthesisDecisionSnapshot:
        if self.decision_margin is not None and not math.isfinite(self.decision_margin):
            raise ValueError("synthesis_decision_margin_nonfinite")
        if self.supported != (self.classification == "supported"):
            raise ValueError("synthesis_decision_supported_mismatch")
        return self


@dataclass(frozen=True, slots=True)
class GraphCounterfactualAuditPlan:
    """Actual baseline and leave-one-estimate-out synthesis reruns for audit policy."""

    claim_model: ClaimModel
    candidates: tuple[AuditCandidate, ...]
    baseline_synthesis: dict[str, Any]
    baseline_decision: SynthesisDecisionSnapshot
    counterfactual_syntheses: Mapping[str, dict[str, Any]]
    counterfactual_decisions: Mapping[str, SynthesisDecisionSnapshot]


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


class CohortEffect(ContractModel):
    """One conservative contribution from one explicit independent cohort."""

    cohort_id: str
    paper_ids: list[str]
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
    assumed_within_cohort_correlation: Annotated[float, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def validate_cohort_effect(self) -> CohortEffect:
        if not math.isfinite(self.estimate) or not math.isfinite(self.variance):
            raise ValueError("cohort_effect_values_must_be_finite")
        if self.point_direction is not _point_direction(self.estimate):
            raise ValueError("cohort_effect_direction_mismatch")
        for name in ("paper_ids", "source_finding_ids", "source_locators"):
            values = getattr(self, name)
            if values != sorted(set(values)):
                raise ValueError(f"cohort_effect_{name}_must_be_sorted_unique")
        return self


class CohortAggregationResult(ContractModel):
    """Typed output from the required one-effect-per-cohort reduction."""

    status: Literal["ok", "insufficient"]
    reason: str | None
    n_effects_input: Annotated[int, Field(ge=0)]
    n_publications: Annotated[int, Field(ge=0)]
    n_cohorts: Annotated[int, Field(ge=0)]
    effects: list[CohortEffect]

    @model_validator(mode="after")
    def validate_result(self) -> CohortAggregationResult:
        if self.status == "ok" and (self.reason is not None or not self.effects):
            raise ValueError("successful_cohort_aggregation_requires_effects_only")
        if self.status == "insufficient" and (self.reason is None or self.effects):
            raise ValueError("insufficient_cohort_aggregation_requires_reason_only")
        if self.n_cohorts != len(self.effects):
            raise ValueError("cohort_aggregation_count_mismatch")
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


def aggregate_one_effect_per_cohort(
    effects: Sequence[HarmonizedEffect],
    cohort_ids: Sequence[str],
    *,
    assumed_within_cohort_correlation: float = 1.0,
) -> CohortAggregationResult:
    """Reduce compatible reports to one conservative contribution per cohort.

    Equal averaging avoids choosing a preferred publication post hoc.  Its variance
    includes every pairwise covariance under the prespecified common correlation;
    ``rho=1`` is the fail-safe default when cross-report covariance is unavailable.
    """

    if len(effects) != len(cohort_ids):
        raise MetaAnalysisContractError("cohort_id_effect_count_mismatch")
    if not 0 <= assumed_within_cohort_correlation <= 1:
        raise MetaAnalysisContractError("within_cohort_correlation_must_be_in_unit_interval")
    if not effects:
        return CohortAggregationResult(
            status="insufficient",
            reason="no_estimable_effects",
            n_effects_input=0,
            n_publications=0,
            n_cohorts=0,
            effects=[],
        )
    if any(not cohort_id.strip() for cohort_id in cohort_ids):
        raise MetaAnalysisContractError("cohort_id_empty")
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
        else:  # pragma: no cover
            reason = "incompatible_effects"
        return CohortAggregationResult(
            status="insufficient",
            reason=reason,
            n_effects_input=len(effects),
            n_publications=len({effect.paper_id for effect in effects}),
            n_cohorts=0,
            effects=[],
        )

    grouped: dict[str, list[HarmonizedEffect]] = defaultdict(list)
    for cohort_id, effect in zip(cohort_ids, effects, strict=True):
        grouped[cohort_id].append(effect)
    cohort_effects: list[CohortEffect] = []
    for cohort_id in sorted(grouped):
        rows = grouped[cohort_id]
        count = len(rows)
        estimate = math.fsum(row.estimate for row in rows) / count
        variances = [row.variance for row in rows]
        covariance_sum = math.fsum(variances) + (
            2
            * assumed_within_cohort_correlation
            * math.fsum(
                math.sqrt(variances[left] * variances[right])
                for left in range(count)
                for right in range(left + 1, count)
            )
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
        cohort_effects.append(
            CohortEffect(
                cohort_id=cohort_id,
                paper_ids=sorted({row.paper_id for row in rows}),
                outcome=rows[0].outcome,
                contrast=rows[0].contrast,
                measure=rows[0].measure,
                unit=rows[0].unit,
                estimate=estimate,
                variance=variance,
                point_direction=_point_direction(estimate),
                source_finding_ids=sorted({row.finding_id for row in rows}),
                source_locators=sorted(
                    {row.provenance.source_locator for row in rows}
                ),
                moderators=moderators,
                moderator_conflicts=conflicts,
                reported_significance=sorted(
                    {row.reported_significance for row in rows}, key=str
                ),
                equivalence_conclusions=sorted(
                    {row.equivalence_conclusion for row in rows}, key=str
                ),
                aggregation_method=(
                    "equal_mean_with_prespecified_common_within_cohort_correlation"
                ),
                assumed_within_cohort_correlation=assumed_within_cohort_correlation,
            )
        )
    return CohortAggregationResult(
        status="ok",
        reason=None,
        n_effects_input=len(effects),
        n_publications=len({effect.paper_id for effect in effects}),
        n_cohorts=len(cohort_effects),
        effects=cohort_effects,
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


def cohort_random_effects_meta_analysis(
    effects: Sequence[HarmonizedEffect],
    cohort_ids: Sequence[str],
    *,
    confidence_level: float = 0.95,
    assumed_within_cohort_correlation: float = 1.0,
) -> dict[str, Any]:
    """Pool one compatible outcome with one contribution per explicit cohort."""

    if not 0 < confidence_level < 1:
        raise MetaAnalysisContractError("confidence_level_must_be_between_zero_and_one")
    aggregation = aggregate_one_effect_per_cohort(
        effects,
        cohort_ids,
        assumed_within_cohort_correlation=assumed_within_cohort_correlation,
    )
    base: dict[str, Any] = {
        "method": {
            "model": "random_effects",
            "tau_squared": "generalized_paule_mandel",
            "uncertainty": "modified_knapp_hartung",
            "cluster_handling": "one_conservative_aggregate_per_explicit_cohort",
            "confidence_level": confidence_level,
            "assumed_within_cohort_correlation": assumed_within_cohort_correlation,
        },
        "n_effects_input": len(effects),
        "n_papers": aggregation.n_publications,
        "n_publications": aggregation.n_publications,
        "n_cohorts": aggregation.n_cohorts,
        "input_provenance": _input_provenance(effects),
    }
    if aggregation.status == "insufficient":
        return {**base, "status": "insufficient", "reason": aggregation.reason}
    if aggregation.n_cohorts < 2:
        return {**base, "status": "insufficient", "reason": "fewer_than_two_cohorts"}

    cohort_effects = aggregation.effects
    y = np.asarray([effect.estimate for effect in cohort_effects], dtype=float)
    variances = np.asarray([effect.variance for effect in cohort_effects], dtype=float)
    design = np.ones((len(y), 1), dtype=float)
    tau_squared = _paule_mandel_tau_squared(y, variances, design)
    beta, covariance, _, residual_q = _weighted_fit(y, variances, design, tau_squared)
    degrees_freedom = len(y) - 1
    scale = max(1.0, residual_q / degrees_freedom)
    standard_error = math.sqrt(float(covariance[0, 0]) * scale)
    critical = float(t.ppf(0.5 + confidence_level / 2, degrees_freedom))
    estimate = float(beta[0])
    lower = estimate - critical * standard_error
    upper = estimate + critical * standard_error
    measure = cohort_effects[0].measure
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
            "reason": "fewer_than_three_cohorts",
            "lower": None,
            "upper": None,
            "degrees_freedom": 0,
            "ratio_scale": None,
        }
    _, _, _, fixed_q = _weighted_fit(y, variances, design, 0.0)
    i_squared = max(0.0, (fixed_q - degrees_freedom) / fixed_q) if fixed_q > 0 else 0.0
    return {
        **base,
        "status": "ok",
        "reason": None,
        "outcome": cohort_effects[0].outcome,
        "contrast": cohort_effects[0].contrast,
        "measure": measure.value,
        "unit": cohort_effects[0].unit,
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_lower": lower,
        "ci_upper": upper,
        "point_direction": _point_direction(estimate).value,
        "two_sided_p_value": float(
            2 * t.sf(abs(estimate / standard_error), degrees_freedom)
        ),
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
        "cohort_effects": [effect.model_dump(mode="json") for effect in cohort_effects],
        "provenance": {
            effect.cohort_id: {
                "paper_ids": effect.paper_ids,
                "finding_ids": effect.source_finding_ids,
                "source_locators": effect.source_locators,
            }
            for effect in cohort_effects
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


def prespecified_condition_analysis(
    effects: Sequence[HarmonizedEffect],
    moderator_names: Sequence[str],
    *,
    familywise_alpha: float = 0.05,
    min_papers_per_level: int = 2,
    assumed_within_paper_correlation: float = 1.0,
) -> dict[str, Any]:
    """Detect prespecified qualitative effect modification conservatively.

    A moderator qualifies only when its Bonferroni-adjusted omnibus test passes and
    at least one adjusted level interval lies strictly above zero while another lies
    strictly below zero.  The adjustment covers every prespecified omnibus test and
    every observed moderator-level interval.  This is a predictive corpus association,
    not a causal subgroup effect, and same-direction magnitude variation is deliberately
    not called ``condition_dependent``.
    """

    if not moderator_names:
        return {
            "status": "not_requested",
            "reason": "no_prespecified_moderators",
            "interpretation": "predictive_association_not_causal",
            "analyses": [],
            "qualifying_moderators": [],
        }
    if not 0 < familywise_alpha < 1:
        raise MetaAnalysisContractError("condition_familywise_alpha_invalid")
    normalized = [name.strip() for name in moderator_names]
    if any(not name for name in normalized):
        raise MetaAnalysisContractError("condition_moderator_name_empty")
    if len(normalized) != len(set(normalized)):
        raise MetaAnalysisContractError("condition_moderator_names_duplicate")

    observed_levels = {
        name: {
            str(effect.moderators[name])
            for effect in effects
            if effect.moderators.get(name) is not None
            and str(effect.moderators[name]).strip()
        }
        for name in normalized
    }
    multiplicity_tests = len(normalized) + sum(
        max(1, len(levels)) for levels in observed_levels.values()
    )
    adjusted_alpha = familywise_alpha / multiplicity_tests
    confidence_level = 1.0 - adjusted_alpha
    analyses: list[dict[str, Any]] = []
    qualifying: list[dict[str, Any]] = []
    for name in normalized:
        regression = categorical_meta_regression(
            effects,
            name,
            confidence_level=confidence_level,
            min_papers_per_level=min_papers_per_level,
            assumed_within_paper_correlation=assumed_within_paper_correlation,
        )
        row: dict[str, Any] = {
            "moderator": name,
            "adjusted_alpha": adjusted_alpha,
            "adjusted_confidence_level": confidence_level,
            "regression": regression,
            "qualifies": False,
            "positive_levels": [],
            "negative_levels": [],
        }
        if regression.get("status") == "ok":
            level_estimates = regression["level_estimates"]
            positive_levels = sorted(
                str(level["level"])
                for level in level_estimates
                if float(level["ci_lower"]) > 0
            )
            negative_levels = sorted(
                str(level["level"])
                for level in level_estimates
                if float(level["ci_upper"]) < 0
            )
            omnibus = regression["omnibus"]
            qualifies = (
                float(omnibus["p_value"]) <= adjusted_alpha
                and bool(positive_levels)
                and bool(negative_levels)
            )
            row.update(
                {
                    "qualifies": qualifies,
                    "positive_levels": positive_levels,
                    "negative_levels": negative_levels,
                }
            )
            if qualifies:
                qualifying.append(
                    {
                        "moderator": name,
                        "positive_levels": positive_levels,
                        "negative_levels": negative_levels,
                        "omnibus_p_value": float(omnibus["p_value"]),
                    }
                )
        analyses.append(row)

    return {
        "status": (
            "condition_dependent"
            if qualifying
            else "no_qualitative_condition_dependence_detected"
        ),
        "reason": (
            "prespecified_moderator_has_adjusted_opposite_direction_level_intervals"
            if qualifying
            else "no_prespecified_moderator_met_adjusted_qualitative_interaction_rule"
        ),
        "interpretation": "predictive_association_not_causal",
        "multiplicity_control": "bonferroni_familywise",
        "familywise_alpha": familywise_alpha,
        "multiplicity_test_count": multiplicity_tests,
        "adjusted_alpha_per_test": adjusted_alpha,
        "analyses": analyses,
        "qualifying_moderators": qualifying,
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


def cohort_directional_synthesis(
    results: Sequence[HarmonizationResult], cohort_ids: Sequence[str]
) -> dict[str, Any]:
    """Cohort-balanced sign synthesis when compatible magnitudes are unavailable."""

    if len(results) != len(cohort_ids):
        raise MetaAnalysisContractError("cohort_id_result_count_mismatch")
    by_cohort: dict[str, list[HarmonizationResult]] = defaultdict(list)
    for cohort_id, result in zip(cohort_ids, results, strict=True):
        if not cohort_id.strip():
            raise MetaAnalysisContractError("cohort_id_empty")
        by_cohort[cohort_id].append(result)
    cohort_directions: dict[str, str] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for cohort_id in sorted(by_cohort):
        rows = by_cohort[cohort_id]
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
        cohort_directions[cohort_id] = direction
        provenance[cohort_id] = {
            "paper_ids": sorted({row.paper_id for row in rows}),
            "finding_ids": sorted({row.finding_id for row in rows}),
            "source_locators": sorted({row.provenance.source_locator for row in rows}),
        }
    categories = ("increase", "decrease", "exact_zero", "mixed", "unavailable")
    counts = Counter(cohort_directions.values())
    count_output = {category: counts[category] for category in categories}
    nonzero = counts["increase"] + counts["decrease"]
    targets = {(result.outcome, result.contrast) for result in results}
    base: dict[str, Any] = {
        "method": "cohort_level_point_estimate_signs_only",
        "limitations": [
            "effect_magnitudes_and_precision_are_not_combined",
            "reported_non_significance_is_not_evidence_of_zero_effect",
            "exact_zero_requires_a_literal_zero_point_estimate",
        ],
        "n_effects_input": len(results),
        "n_papers": len({result.paper_id for result in results}),
        "n_publications": len({result.paper_id for result in results}),
        "n_cohorts": len(by_cohort),
        "cohort_direction_counts": count_output,
        "cohort_directions": cohort_directions,
        "provenance": provenance,
    }
    if len(targets) > 1:
        return {**base, "status": "insufficient", "reason": "incompatible_directional_targets"}
    if targets:
        outcome, contrast = next(iter(targets))
        base.update({"outcome": outcome, "contrast": contrast})
    if nonzero < 2:
        return {
            **base,
            "status": "insufficient",
            "reason": "fewer_than_two_nonzero_cohort_signs",
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


def synthesize_with_cohort_directional_fallback(
    results: Sequence[HarmonizationResult],
    cohort_ids: Sequence[str],
    *,
    confidence_level: float = 0.95,
    assumed_within_cohort_correlation: float = 1.0,
) -> dict[str, Any]:
    """Run cohort-unit magnitude synthesis, then a cohort-balanced sign fallback."""

    estimable_pairs = [
        (result.effect, cohort_id)
        for result, cohort_id in zip(results, cohort_ids, strict=True)
        if result.effect is not None
    ]
    estimable = [effect for effect, _ in estimable_pairs]
    estimable_cohorts = [cohort_id for _, cohort_id in estimable_pairs]
    quantitative = cohort_random_effects_meta_analysis(
        estimable,
        estimable_cohorts,
        confidence_level=confidence_level,
        assumed_within_cohort_correlation=assumed_within_cohort_correlation,
    )
    if quantitative["status"] == "ok":
        return {
            "status": "ok",
            "mode": "random_effects_meta_analysis",
            "quantitative": quantitative,
            "directional_fallback": None,
        }
    directional = cohort_directional_synthesis(results, cohort_ids)
    return {
        "status": directional["status"],
        "mode": "directional_sign_synthesis" if directional["status"] == "ok" else "insufficient",
        "quantitative": quantitative,
        "directional_fallback": directional,
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
    assumed_within_cohort_correlation: float = 1.0,
    excluded_estimate_ids: Sequence[str] = (),
    prespecified_moderators: Sequence[str] = (),
    condition_familywise_alpha: float = 0.05,
    condition_min_papers_per_level: int = 2,
) -> dict[str, Any]:
    """Bridge the typed graph to conservative explicit-cohort-unit synthesis."""

    if len(excluded_estimate_ids) != len(set(excluded_estimate_ids)):
        raise MetaAnalysisContractError("excluded_estimate_ids_duplicate")
    known_estimate_ids = {estimate.estimate_id for estimate in graph.outcome_estimates}
    unknown_exclusions = sorted(set(excluded_estimate_ids) - known_estimate_ids)
    if unknown_exclusions:
        raise MetaAnalysisContractError(
            f"excluded_estimate_ids_unknown:{unknown_exclusions}"
        )
    excluded = set(excluded_estimate_ids)
    active_graph = graph.model_copy(
        update={
            "outcome_estimates": [
                estimate
                for estimate in graph.outcome_estimates
                if estimate.estimate_id not in excluded
            ]
        }
    )
    risk_features = graph_risk_features(
        active_graph,
        outcome_name=outcome_name,
        contrast_id=contrast_id,
    ).model_dump(mode="json")
    selection = select_effect_evidence(
        active_graph,
        outcome_name=outcome_name,
        contrast_id=contrast_id,
        require_explicit_timepoint=require_explicit_timepoint,
    )
    graph_contract = {
        "graph_schema_version": graph.graph_schema_version,
        "selection_status": selection.status,
        "selection_reason": selection.reason,
        "selected_estimate_ids": selection.estimate_ids,
        "selected_cohort_ids": selection.cohort_ids,
        "selected_cohort_count": len(set(selection.cohort_ids)),
        "excluded_estimate_ids": sorted(excluded),
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
            "condition_analysis": None,
            "evidence_graph": graph_contract,
        }
    results = harmonize_effects(selection.records)
    synthesis = synthesize_with_cohort_directional_fallback(
        results,
        selection.cohort_ids,
        confidence_level=confidence_level,
        assumed_within_cohort_correlation=assumed_within_cohort_correlation,
    )
    estimable = [result.effect for result in results if result.effect is not None]
    cohort_to_papers: dict[str, set[str]] = defaultdict(set)
    paper_to_cohorts: dict[str, set[str]] = defaultdict(set)
    for cohort_id, record in zip(selection.cohort_ids, selection.records, strict=True):
        cohort_to_papers[cohort_id].add(record.paper_id)
        paper_to_cohorts[record.paper_id].add(cohort_id)
    condition_mapping_is_one_to_one = all(
        len(values) == 1 for values in (*cohort_to_papers.values(), *paper_to_cohorts.values())
    )
    if not prespecified_moderators:
        condition_analysis = None
    elif condition_mapping_is_one_to_one:
        condition_analysis = prespecified_condition_analysis(
            estimable,
            prespecified_moderators,
            familywise_alpha=condition_familywise_alpha,
            min_papers_per_level=condition_min_papers_per_level,
            assumed_within_paper_correlation=assumed_within_paper_correlation,
        )
    else:
        condition_analysis = {
            "status": "insufficient",
            "reason": "cohort_aware_condition_analysis_not_implemented_for_nonunique_mapping",
            "interpretation": "predictive_association_not_causal",
            "analyses": [],
            "qualifying_moderators": [],
        }
    return {
        **synthesis,
        "condition_analysis": condition_analysis,
        "evidence_graph": graph_contract,
    }


def synthesis_decision_snapshot(
    synthesis: Mapping[str, Any],
    *,
    target_direction: Literal["increase", "decrease"],
    require_prediction_interval_stability: bool = True,
) -> SynthesisDecisionSnapshot:
    """Map a synthesis to the exact directional decision and a bounded audit score."""

    mode = str(synthesis.get("mode", "unknown"))
    if synthesis.get("status") != "ok":
        reason = str(synthesis.get("reason") or "synthesis_insufficient")
        graph_contract = synthesis.get("evidence_graph")
        if isinstance(graph_contract, Mapping) and graph_contract.get("selection_reason"):
            reason = str(graph_contract["selection_reason"])
        return SynthesisDecisionSnapshot(
            target_direction=target_direction,
            classification="not_evaluable",
            supported=False,
            decision_score=0.5,
            mode=mode,
            reason=reason,
        )

    condition = synthesis.get("condition_analysis")
    condition_dependent = (
        isinstance(condition, Mapping)
        and condition.get("status") == "condition_dependent"
    )
    if mode == "random_effects_meta_analysis":
        quantitative = synthesis.get("quantitative")
        if not isinstance(quantitative, Mapping) or quantitative.get("status") != "ok":
            raise MetaAnalysisContractError("quantitative_synthesis_payload_invalid")
        estimate = float(quantitative["estimate"])
        lower = float(quantitative["ci_lower"])
        upper = float(quantitative["ci_upper"])
        two_sided_p = float(quantitative["two_sided_p_value"])
        aligned = estimate > 0 if target_direction == "increase" else estimate < 0
        score = 0.5 if estimate == 0 else (1 - two_sided_p / 2 if aligned else two_sided_p / 2)
        ci_margin = lower if target_direction == "increase" else -upper
        prediction = quantitative.get("prediction_interval")
        prediction_ok = isinstance(prediction, Mapping) and prediction.get("status") == "ok"
        prediction_margin: float | None = None
        if prediction_ok:
            prediction_margin = (
                float(prediction["lower"])
                if target_direction == "increase"
                else -float(prediction["upper"])
            )
        margin = (
            min(ci_margin, prediction_margin)
            if require_prediction_interval_stability and prediction_margin is not None
            else ci_margin
        )
        opposite = upper < 0 if target_direction == "increase" else lower > 0
        if condition_dependent:
            classification = "condition_dependent"
            reason = "prespecified_qualitative_condition_dependence_detected"
        elif opposite:
            classification = "contradicted"
            reason = "confidence_interval_supports_opposite_direction"
        elif ci_margin <= 0:
            classification = "inconclusive"
            reason = "confidence_interval_includes_null"
        elif require_prediction_interval_stability and not prediction_ok:
            classification = "inconclusive"
            reason = "prediction_interval_required_but_unavailable"
        elif require_prediction_interval_stability and prediction_margin is not None and (
            prediction_margin <= 0
        ):
            classification = "inconclusive"
            reason = "prediction_interval_not_stable_in_target_direction"
        else:
            classification = "supported"
            reason = "confidence_interval_and_required_prediction_interval_support_target"
        return SynthesisDecisionSnapshot(
            target_direction=target_direction,
            classification=classification,
            supported=classification == "supported",
            decision_score=max(0.0, min(1.0, score)),
            decision_margin=margin,
            mode=mode,
            reason=reason,
        )

    if mode == "directional_sign_synthesis":
        directional = synthesis.get("directional_fallback")
        if not isinstance(directional, Mapping) or directional.get("status") != "ok":
            raise MetaAnalysisContractError("directional_synthesis_payload_invalid")
        interval = directional["increase_fraction_exact_ci_95"]
        lower, upper = float(interval[0]), float(interval[1])
        increase_fraction = float(directional["increase_fraction_among_nonzero"])
        score = increase_fraction if target_direction == "increase" else 1 - increase_fraction
        margin = lower - 0.5 if target_direction == "increase" else 0.5 - upper
        opposite = upper < 0.5 if target_direction == "increase" else lower > 0.5
        if condition_dependent:
            classification = "condition_dependent"
            reason = "prespecified_qualitative_condition_dependence_detected"
        elif opposite:
            classification = "contradicted"
            reason = "exact_sign_interval_supports_opposite_direction"
        elif margin <= 0:
            classification = "inconclusive"
            reason = "exact_sign_interval_includes_equal_direction_frequency"
        elif require_prediction_interval_stability:
            classification = "inconclusive"
            reason = "directional_fallback_cannot_satisfy_prediction_interval_requirement"
        else:
            classification = "supported"
            reason = "exact_sign_interval_supports_target_direction"
        return SynthesisDecisionSnapshot(
            target_direction=target_direction,
            classification=classification,
            supported=classification == "supported",
            decision_score=score,
            decision_margin=margin,
            mode=mode,
            reason=reason,
        )
    raise MetaAnalysisContractError(f"unknown_successful_synthesis_mode:{mode}")


def _require_exact_item_mapping(
    values: Mapping[str, object], item_ids: set[str], label: str
) -> None:
    observed = set(values)
    if observed != item_ids:
        missing = sorted(item_ids - observed)
        extra = sorted(observed - item_ids)
        raise MetaAnalysisContractError(
            f"counterfactual_{label}_identity_mismatch:missing={missing}:extra={extra}"
        )


def build_graph_counterfactual_audit_plan(
    graph: EvidenceGraph,
    *,
    outcome_name: str,
    target_direction: Literal["increase", "decrease"],
    error_probabilities: Mapping[str, float],
    verification_costs: Mapping[str, float],
    probability_basis: ProbabilityBasis | Mapping[str, ProbabilityBasis],
    probability_source: str,
    cost_unit: str = "minutes",
    disagreement_scores: Mapping[str, float] | None = None,
    contrast_id: str | None = None,
    require_explicit_timepoint: bool = True,
    require_prediction_interval_stability: bool = True,
    confidence_level: float = 0.95,
    assumed_within_paper_correlation: float = 1.0,
    excluded_estimate_ids: Sequence[str] = (),
    prespecified_moderators: Sequence[str] = (),
    condition_familywise_alpha: float = 0.05,
    condition_min_papers_per_level: int = 2,
    claim_id: str = "graph-derived-directional-claim",
) -> GraphCounterfactualAuditPlan:
    """Create audit candidates from actual leave-one-out synthesis reruns.

    The direct scores carried by each candidate override the legacy additive bridge.
    Leave-one-out is a sensitivity scenario, not an oracle correction and not a claim
    probability; release-risk guarantees must be supplied separately through calibrated
    marginal error-probability upper bounds.
    """

    excluded = set(excluded_estimate_ids)
    matching_ids = {
        estimate.estimate_id
        for estimate in graph.outcome_estimates
        if estimate.estimate_id not in excluded
        and estimate.outcome_name == outcome_name
        and (contrast_id is None or estimate.contrast_id == contrast_id)
    }
    if not matching_ids:
        raise MetaAnalysisContractError("counterfactual_plan_has_no_matching_estimates")
    _require_exact_item_mapping(error_probabilities, matching_ids, "error_probability")
    _require_exact_item_mapping(verification_costs, matching_ids, "verification_cost")
    if disagreement_scores is not None:
        _require_exact_item_mapping(disagreement_scores, matching_ids, "disagreement")
    if isinstance(probability_basis, Mapping):
        _require_exact_item_mapping(probability_basis, matching_ids, "probability_basis")

    synthesis_kwargs: dict[str, Any] = {
        "outcome_name": outcome_name,
        "contrast_id": contrast_id,
        "require_explicit_timepoint": require_explicit_timepoint,
        "confidence_level": confidence_level,
        "assumed_within_paper_correlation": assumed_within_paper_correlation,
        "prespecified_moderators": prespecified_moderators,
        "condition_familywise_alpha": condition_familywise_alpha,
        "condition_min_papers_per_level": condition_min_papers_per_level,
    }
    baseline_synthesis = synthesize_evidence_graph(
        graph,
        excluded_estimate_ids=sorted(excluded),
        **synthesis_kwargs,
    )
    baseline_decision = synthesis_decision_snapshot(
        baseline_synthesis,
        target_direction=target_direction,
        require_prediction_interval_stability=require_prediction_interval_stability,
    )
    counterfactual_syntheses: dict[str, dict[str, Any]] = {}
    counterfactual_decisions: dict[str, SynthesisDecisionSnapshot] = {}
    candidates: list[AuditCandidate] = []
    for item_id in sorted(matching_ids):
        counterfactual = synthesize_evidence_graph(
            graph,
            excluded_estimate_ids=sorted({*excluded, item_id}),
            **synthesis_kwargs,
        )
        decision = synthesis_decision_snapshot(
            counterfactual,
            target_direction=target_direction,
            require_prediction_interval_stability=require_prediction_interval_stability,
        )
        counterfactual_syntheses[item_id] = counterfactual
        counterfactual_decisions[item_id] = decision
        basis = (
            probability_basis[item_id]
            if isinstance(probability_basis, Mapping)
            else probability_basis
        )
        candidates.append(
            AuditCandidate(
                item_id=item_id,
                baseline_contribution=0.0,
                counterfactual_contribution=0.0,
                error_probability=float(error_probabilities[item_id]),
                probability_basis=basis,
                probability_source=probability_source,
                verification_cost=float(verification_costs[item_id]),
                cost_unit=cost_unit,
                disagreement_score=(
                    float(disagreement_scores[item_id])
                    if disagreement_scores is not None
                    else 0.0
                ),
                scenario_kind=ScenarioKind.LEAVE_ONE_OUT,
                scenario_source=(
                    "actual_evidence_graph_leave_one_out_synthesis_rerun;"
                    "sensitivity_scenario_not_oracle_correction"
                ),
                baseline_decision_score=baseline_decision.decision_score,
                counterfactual_decision_score=decision.decision_score,
                decision_score_source=(
                    "frequentist_directional_evidence_score_not_truth_probability"
                ),
                baseline_decision=baseline_decision.supported,
                counterfactual_decision=decision.supported,
            )
        )
    return GraphCounterfactualAuditPlan(
        claim_model=ClaimModel(
            intercept=0.0,
            decision_threshold=confidence_level,
            claim_id=claim_id,
        ),
        candidates=tuple(candidates),
        baseline_synthesis=baseline_synthesis,
        baseline_decision=baseline_decision,
        counterfactual_syntheses=counterfactual_syntheses,
        counterfactual_decisions=counterfactual_decisions,
    )
