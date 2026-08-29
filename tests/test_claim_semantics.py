from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
from pydantic import ValidationError

from literature_multiverse.claim_release import (
    ClaimReleaseConfig,
    assess_qualified_claim_evidence,
)
from literature_multiverse.claim_semantics import (
    ConditionMatchStatus,
    ConditionPredicate,
    MeaningfulEffectThreshold,
    QualifiedClaimVerdictState,
    freeze_claim_target_v2,
    freeze_global_condition_dependence_target,
    freeze_qualified_claim_amendment,
    freeze_qualified_claim_verdict,
)
from literature_multiverse.effects import EffectEvidence
from literature_multiverse.evidence_graph import (
    ArmRole,
    CohortIdentity,
    EvidenceGraph,
    GraphAdapterContext,
    OutcomeTimepoint,
    PublicationIdentity,
    adapt_effect_evidence,
)


def test_global_condition_target_is_distinct_typed_and_tamper_evident() -> None:
    target = freeze_global_condition_dependence_target(
        claim_id="global-dose-polarity",
        reference_direction="increase",
        outcome_name="performance",
        contrast_label="intervention_vs_control",
        estimand="between-group standardized mean difference at four weeks",
        positive_direction_means="higher performance under intervention",
        treatment_role=ArmRole.INTERVENTION,
        comparator_role=ArmRole.CONTROL,
        measure="standardized_mean_difference",
        moderator_names=["species", "dose"],
    )

    assert target.contrast_id is None
    assert target.moderator_names == ["dose", "species"]
    assert target.target_semantics.endswith("not a causal interaction")
    tampered = target.model_dump(mode="json")
    tampered["positive_direction_means"] = "lower performance under intervention"
    with pytest.raises(ValidationError, match="global_condition_target_hash_mismatch"):
        type(target).model_validate(tampered)


def test_global_condition_target_rejects_incompatible_scale_and_units() -> None:
    with pytest.raises(ValidationError, match="mean_difference_requires_unit"):
        freeze_global_condition_dependence_target(
            claim_id="global-pressure-polarity",
            reference_direction="decrease",
            outcome_name="blood pressure",
            contrast_label="treatment_vs_control",
            estimand="between-group mean difference at endpoint",
            positive_direction_means="higher blood pressure under treatment",
            treatment_role=ArmRole.INTERVENTION,
            comparator_role=ArmRole.CONTROL,
            measure="mean_difference",
            moderator_names=["dose"],
        )


def _context(suffix: str) -> GraphAdapterContext:
    return GraphAdapterContext(
        publication=PublicationIdentity(
            publication_id=f"publication-{suffix}",
            paper_id=f"paper-{suffix}",
            doc_id=f"document-{suffix}",
        ),
        study_id=f"study-{suffix}",
        cohort_identity=CohortIdentity(
            cohort_id=f"cohort-{suffix}",
            basis="reviewer_reconciled",
            rationale="Registry and recruitment details reconciled.",
        ),
        treatment_arm_id=f"arm-{suffix}-treatment",
        comparator_arm_id=f"arm-{suffix}-control",
        contrast_id=f"contrast-{suffix}",
        contrast_label="intervention_vs_control",
        positive_direction_means="higher outcome under intervention",
        treatment_label="intervention",
        comparator_label="control",
        timepoint=OutcomeTimepoint(kind="exact", value=4, unit="week"),
    )


def _effect(
    suffix: str,
    *,
    estimate: float,
    moderator: object = "high",
    standard_error: float | None = 0.05,
) -> EffectEvidence:
    if moderator is _MISSING:
        moderators = {}
    elif isinstance(moderator, Mapping):
        moderators = dict(moderator)
    else:
        moderators = {"dose": moderator}
    return EffectEvidence(
        paper_id=f"paper-{suffix}",
        finding_id=f"finding-{suffix}",
        outcome="performance",
        contrast="intervention_vs_control",
        effect_format="hedges_g",
        estimate=estimate,
        standard_error=standard_error,
        moderators=moderators,
        provenance={
            "source_locator": f"paper-{suffix}.pdf#table=2",
            "source_quote": f"The standardized estimate was {estimate}.",
        },
    )


def _graph(
    rows: Sequence[tuple[str, float, object]],
    *,
    standard_error: float | None = 0.05,
) -> EvidenceGraph:
    parts = [
        adapt_effect_evidence(
            _effect(
                suffix,
                estimate=estimate,
                moderator=moderator,
                standard_error=standard_error,
            ),
            context=_context(suffix),
        ).graph
        for suffix, estimate, moderator in rows
    ]
    return EvidenceGraph(
        publications=[item for part in parts for item in part.publications],
        studies=[item for part in parts for item in part.studies],
        cohorts=[item for part in parts for item in part.cohorts],
        arms=[item for part in parts for item in part.arms],
        contrasts=[item for part in parts for item in part.contrasts],
        outcome_estimates=[item for part in parts for item in part.outcome_estimates],
        evidence_spans=[item for part in parts for item in part.evidence_spans],
    )


def _target(
    *,
    value: str = "high",
    minimum_magnitude: float = 0.2,
    specification_status: str = "prespecified",
    parent_claim_sha256: str | None = None,
    discovery_source_sha256: str | None = None,
):
    return freeze_claim_target_v2(
        claim_id=f"performance-increase-{value}",
        direction="increase",
        outcome_name="performance",
        conditions=[ConditionPredicate(moderator="dose", operator="equals", value=value)],
        meaningful_effect_threshold=MeaningfulEffectThreshold(
            minimum_magnitude=minimum_magnitude,
            measure="standardized_mean_difference",
        ),
        specification_status=specification_status,
        parent_claim_sha256=parent_claim_sha256,
        discovery_source_sha256=discovery_source_sha256,
    )


_MISSING = object()


def test_condition_predicates_are_typed_and_do_not_coerce_values() -> None:
    equals_integer = ConditionPredicate(moderator="dose", operator="equals", value=1)
    numeric_range = ConditionPredicate(
        moderator="age",
        operator="between",
        lower=18,
        upper=65,
        include_upper=False,
    )

    assert equals_integer.match({"dose": 1}) is ConditionMatchStatus.MATCHED
    assert equals_integer.match({"dose": 1.0}) is ConditionMatchStatus.NOT_MATCHED
    assert equals_integer.match({}) is ConditionMatchStatus.MISSING
    assert numeric_range.match({"age": 64}) is ConditionMatchStatus.MATCHED
    assert numeric_range.match({"age": "64"}) is ConditionMatchStatus.TYPE_MISMATCH


def test_numeric_ranges_do_not_lose_integer_precision_above_float_exactness() -> None:
    lower = 2**53
    predicate = ConditionPredicate(
        moderator="count",
        operator="between",
        lower=lower,
        upper=lower + 2,
        include_lower=False,
        include_upper=False,
    )

    assert predicate.match({"count": lower}) is ConditionMatchStatus.NOT_MATCHED
    assert predicate.match({"count": lower + 1}) is ConditionMatchStatus.MATCHED
    assert predicate.match({"count": lower + 2}) is ConditionMatchStatus.NOT_MATCHED


def test_claim_and_condition_names_and_units_are_canonically_trimmed() -> None:
    condition = ConditionPredicate(moderator=" dose ", operator="equals", value=" high ")
    threshold = MeaningfulEffectThreshold(
        minimum_magnitude=1.0,
        measure="mean_difference",
        unit=" mmHg ",
    )
    target = freeze_claim_target_v2(
        claim_id=" claim ",
        direction="increase",
        outcome_name=" pressure ",
        contrast_id=" treatment-v-control ",
        conditions=[condition],
        meaningful_effect_threshold=threshold,
    )

    assert condition.moderator == "dose"
    assert condition.value == "high"
    assert threshold.unit == "mmHg"
    assert target.claim_id == "claim"
    assert target.outcome_name == "pressure"
    assert target.contrast_id == "treatment-v-control"


def test_claim_target_hash_is_canonical_and_tamper_evident() -> None:
    threshold = MeaningfulEffectThreshold(
        minimum_magnitude=0.2,
        measure="standardized_mean_difference",
    )
    first = freeze_claim_target_v2(
        claim_id="qualified-target",
        direction="increase",
        outcome_name="performance",
        conditions=[
            ConditionPredicate(moderator="species", operator="equals", value="mouse"),
            ConditionPredicate(moderator="dose", operator="in", values=["high", "medium"]),
        ],
        meaningful_effect_threshold=threshold,
    )
    second = freeze_claim_target_v2(
        claim_id="qualified-target",
        direction="increase",
        outcome_name="performance",
        conditions=list(reversed(first.conditions)),
        meaningful_effect_threshold=threshold,
    )

    assert first.claim_sha256 == second.claim_sha256
    assert [row.moderator for row in first.conditions] == ["dose", "species"]
    tampered = first.model_dump(mode="json")
    tampered["outcome_name"] = "different outcome"
    with pytest.raises(ValidationError, match="qualified_claim_hash_mismatch"):
        type(first).model_validate(tampered)


def test_amendment_is_deterministic_and_can_only_be_a_hypothesis() -> None:
    parent = _target()
    source_synthesis_sha256 = "b" * 64
    first = freeze_qualified_claim_amendment(
        parent_target=parent,
        source_synthesis_sha256=source_synthesis_sha256,
        amended_claim_id="qualified-target-discovery",
        discovered_conditions=[
            ConditionPredicate(moderator="species", operator="equals", value="mouse"),
            ConditionPredicate(moderator="age", operator="between", lower=18, upper=65),
        ],
    )
    second = freeze_qualified_claim_amendment(
        parent_target=parent,
        source_synthesis_sha256=source_synthesis_sha256,
        amended_claim_id="qualified-target-discovery",
        discovered_conditions=list(
            reversed(
                [
                    condition
                    for condition in first.proposed_target.conditions
                    if condition.moderator != "dose"
                ]
            )
        ),
    )

    assert first.amendment_sha256 == second.amendment_sha256
    assert first.eligible_for_source_corpus_release is False
    assert first.proposed_target.specification_status == "discovered_hypothesis"
    with pytest.raises(ValidationError, match="qualified_verdict_discovery_state_mismatch"):
        freeze_qualified_claim_verdict(
            target=first.proposed_target,
            synthesis_sha256=source_synthesis_sha256,
            state=QualifiedClaimVerdictState.PRESPECIFIED_SUPPORTED,
            reason="must not be accepted",
            mode="random_effects_meta_analysis",
        )


def test_prespecified_condition_filters_before_synthesis_and_supports_threshold() -> None:
    graph = _graph(
        [
            *((f"high-{index}", 0.8, "high") for index in range(4)),
            *((f"low-{index}", -0.8, "low") for index in range(4)),
        ]
    )
    assessment = assess_qualified_claim_evidence(graph=graph, target=_target())

    assert assessment.verdict.state == "prespecified_supported"
    assert assessment.verdict.synthesis_gate_passed is True
    selection = assessment.synthesis["qualified_claim"]
    assert len(selection["matched_estimate_ids"]) == 4
    assert len(selection["condition_excluded_estimate_ids"]) == 4
    assert assessment.verdict.ci_lower > 0.2
    assert assessment.verdict.prediction_interval_lower > 0.2


def test_prespecified_condition_can_be_meaningfully_contradicted() -> None:
    graph = _graph(
        [
            *((f"high-{index}", 0.8, "high") for index in range(4)),
            *((f"low-{index}", -0.8, "low") for index in range(4)),
        ]
    )
    assessment = assess_qualified_claim_evidence(
        graph=graph,
        target=_target(value="low"),
    )

    assert assessment.verdict.state == "prespecified_contradicted"
    assert assessment.verdict.reason == ("confidence_interval_supports_meaningful_opposite_effect")


def test_nonzero_meaningful_threshold_prevents_null_only_support() -> None:
    graph = _graph([(f"high-{index}", 0.8, "high") for index in range(4)])
    assessment = assess_qualified_claim_evidence(
        graph=graph,
        target=_target(minimum_magnitude=0.9),
    )

    assert assessment.verdict.state == "prespecified_inconclusive"
    assert assessment.verdict.reason == (
        "confidence_interval_does_not_exclude_submeaningful_effects"
    )


def test_missing_condition_value_fails_the_entire_qualified_synthesis_closed() -> None:
    graph = _graph(
        [
            *((f"high-{index}", 0.8, "high") for index in range(4)),
            ("unknown", 0.8, _MISSING),
        ]
    )
    assessment = assess_qualified_claim_evidence(graph=graph, target=_target())

    assert assessment.synthesis["status"] == "insufficient"
    assert assessment.synthesis["reason"] == "condition_moderator_missing"
    assert assessment.verdict.state == "prespecified_not_evaluable"
    assert assessment.verdict.missing_condition_estimate_ids == ["estimate-finding-unknown"]


def test_condition_type_mismatch_fails_closed_in_graph_selection() -> None:
    graph = _graph([(f"row-{index}", 0.8, "1.5") for index in range(4)])
    target = freeze_claim_target_v2(
        claim_id="numeric-dose-target",
        direction="increase",
        outcome_name="performance",
        conditions=[
            ConditionPredicate(
                moderator="dose",
                operator="between",
                lower=1,
                upper=2,
            )
        ],
        meaningful_effect_threshold=MeaningfulEffectThreshold(
            minimum_magnitude=0.2,
            measure="standardized_mean_difference",
        ),
    )
    assessment = assess_qualified_claim_evidence(graph=graph, target=target)

    assert assessment.synthesis["reason"] == "condition_value_type_mismatch"
    assert assessment.verdict.state == "prespecified_not_evaluable"
    assert len(assessment.verdict.type_mismatch_estimate_ids) == 4


def test_sign_only_evidence_cannot_satisfy_a_magnitude_threshold() -> None:
    graph = _graph(
        [(f"high-{index}", 0.8, "high") for index in range(6)],
        standard_error=None,
    )
    assessment = assess_qualified_claim_evidence(
        graph=graph,
        target=_target(),
        config=ClaimReleaseConfig(require_prediction_interval_stability=False),
    )

    assert assessment.synthesis["mode"] == "directional_sign_synthesis"
    assert assessment.verdict.state == "prespecified_not_evaluable"
    assert assessment.verdict.reason == (
        "meaningful_effect_threshold_requires_compatible_quantitative_synthesis"
    )


def test_discovered_target_remains_hypothesis_only_despite_strong_effect() -> None:
    parent = _target(value="all")
    amendment = freeze_qualified_claim_amendment(
        parent_target=parent,
        source_synthesis_sha256="c" * 64,
        amended_claim_id="discovered-high-dose-effect",
        discovered_conditions=[
            ConditionPredicate(moderator="species", operator="equals", value="mouse")
        ],
    )
    rows = [(f"row-{index}", 1.2, {"dose": "all", "species": "mouse"}) for index in range(4)]
    graph = _graph(rows)
    assessment = assess_qualified_claim_evidence(
        graph=graph,
        target=amendment.proposed_target,
    )

    assert assessment.verdict.state == "discovered_hypothesis_only"
    assert assessment.verdict.synthesis_gate_passed is False
    assert "independent_confirmation" in assessment.verdict.reason
    assert assessment.release_semantics.startswith("evidence gate only")
