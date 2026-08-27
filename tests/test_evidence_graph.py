from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from literature_multiverse.effects import (
    EffectEvidence,
    PointDirection,
    ReportedSignificance,
    harmonize_effect,
)
from literature_multiverse.evidence_graph import (
    CohortIdentity,
    EvidenceGraph,
    GraphAdapterContext,
    OutcomeTimepoint,
    PublicationIdentity,
    adapt_effect_evidence,
    adapt_finding_row,
    evidence_graph_json_schema,
    graph_risk_features,
    select_effect_evidence,
)
from literature_multiverse.meta_analysis import synthesize_evidence_graph
from literature_multiverse.models import FindingRow, make_finding_id


def _context(
    suffix: str,
    *,
    paper_id: str | None = None,
    cohort_id: str | None = None,
    cohort_basis: str = "reviewer_reconciled",
) -> GraphAdapterContext:
    cohort = cohort_id or f"cohort-{suffix}"
    return GraphAdapterContext(
        publication=PublicationIdentity(
            publication_id=f"publication-{suffix}",
            paper_id=paper_id or f"paper-{suffix}",
            doc_id=f"document-{suffix}",
            doi=f"10.1000/{suffix}",
        ),
        study_id=f"study-{suffix}",
        cohort_identity=CohortIdentity(
            cohort_id=cohort,
            basis=cohort_basis,
            rationale="Two reviewers reconciled registry and recruitment details.",
        ),
        treatment_arm_id=f"arm-{suffix}-treatment",
        comparator_arm_id=f"arm-{suffix}-comparator",
        contrast_id=f"contrast-{suffix}",
        contrast_label="intervention_vs_control",
        positive_direction_means="higher outcome value under intervention",
        treatment_label="intervention",
        comparator_label="control",
        timepoint=OutcomeTimepoint(kind="exact", value=4, unit="week", anchor="baseline"),
    )


def _evidence(suffix: str, *, estimate: float = 0.2) -> EffectEvidence:
    return EffectEvidence(
        paper_id=f"paper-{suffix}",
        finding_id=f"finding-{suffix}",
        outcome="performance",
        contrast="intervention_vs_control",
        effect_format="hedges_g",
        estimate=estimate,
        standard_error=0.1,
        reported_significance="not_significant",
        provenance={
            "source_locator": f"paper-{suffix}.pdf#page=4",
            "source_quote": f"The standardized estimate was {estimate}.",
        },
    )


def _merge_graphs(*graphs: EvidenceGraph) -> EvidenceGraph:
    return EvidenceGraph(
        publications=[item for graph in graphs for item in graph.publications],
        studies=[item for graph in graphs for item in graph.studies],
        cohorts=[item for graph in graphs for item in graph.cohorts],
        arms=[item for graph in graphs for item in graph.arms],
        contrasts=[item for graph in graphs for item in graph.contrasts],
        outcome_estimates=[item for graph in graphs for item in graph.outcome_estimates],
        evidence_spans=[item for graph in graphs for item in graph.evidence_spans],
    )


def test_typed_effect_adapter_builds_full_referential_graph() -> None:
    evidence = _evidence("a")
    result = adapt_effect_evidence(evidence, context=_context("a"))

    assert result.status == "ready"
    assert result.graph.studies[0].primary_publication_id == "publication-a"
    assert result.graph.cohorts[0].cohort_id == "cohort-a"
    assert result.graph.contrasts[0].treatment_arm_id == "arm-a-treatment"
    estimate = result.graph.outcome_estimates[0]
    assert estimate.effect is evidence
    assert estimate.timepoint.value == 4
    assert estimate.evidence_span_ids == [result.graph.evidence_spans[0].span_id]


def test_non_significant_positive_effect_survives_graph_adapter_without_zero_recoding() -> None:
    result = adapt_effect_evidence(_evidence("a"), context=_context("a"))
    converted = harmonize_effect(result.graph.outcome_estimates[0].effect)

    assert converted.point_direction is PointDirection.INCREASE
    assert converted.reported_significance is ReportedSignificance.NOT_SIGNIFICANT
    assert converted.effect is not None
    assert converted.effect.estimate == 0.2


def test_legacy_no_effect_is_retained_only_as_ambiguous_source_label(
    finding_payload: dict[str, object],
) -> None:
    payload = deepcopy(finding_payload)
    payload.update(
        effect_direction="no_effect",
        effect_size_raw="reported as no detectable difference",
        significant=False,
        p_value=0.41,
    )
    payload["finding_id"] = make_finding_id(
        paper_id=str(payload["paper_id"]),
        map_result_id=str(payload["map_result_id"]),
        array_position=int(payload["array_position"]),
        outcome_name=str(payload["outcome_name"]),
        timepoint_raw=str(payload["timepoint_raw"]),
        dose_raw=str(payload["dose_raw"]),
        effect_direction="no_effect",
    )
    finding = FindingRow.model_validate(payload)
    result = adapt_finding_row(
        finding,
        context=_context("legacy", paper_id=finding.paper_id),
    )

    estimate = result.graph.outcome_estimates[0]
    harmonized = harmonize_effect(estimate.effect)
    assert result.status == "requires_review"
    assert estimate.legacy_reported_direction == "no_effect"
    assert estimate.legacy_effect_size_raw == "reported as no detectable difference"
    assert estimate.effect.estimate is None
    assert estimate.effect.reported_significance is ReportedSignificance.NOT_SIGNIFICANT
    assert result.graph.evidence_spans[0].section == "Results"
    assert result.graph.evidence_spans[0].line_ids == ["L10"]
    assert harmonized.status == "insufficient"
    assert harmonized.point_direction is PointDirection.NOT_AVAILABLE
    assert {issue.code for issue in result.issues} >= {
        "legacy_effect_not_quantitatively_interpretable",
        "legacy_no_effect_is_ambiguous",
    }


def test_legacy_free_text_timepoint_is_not_guessed_as_a_numeric_duration(
    finding_payload: dict[str, object],
) -> None:
    finding = FindingRow.model_validate(finding_payload)
    context = _context("legacy", paper_id=finding.paper_id).model_copy(update={"timepoint": None})
    result = adapt_finding_row(finding, context=context)

    timepoint = result.graph.outcome_estimates[0].timepoint
    assert timepoint.kind == "reported_text"
    assert timepoint.raw_label == "post"
    assert timepoint.value is None
    assert timepoint.unit is None


def test_graph_rejects_contrast_arm_from_another_cohort() -> None:
    first = adapt_effect_evidence(_evidence("a"), context=_context("a")).graph
    second = adapt_effect_evidence(_evidence("b"), context=_context("b")).graph
    payload = _merge_graphs(first, second).model_dump(mode="json")
    payload["contrasts"][0]["treatment_arm_id"] = "arm-b-treatment"

    with pytest.raises(ValidationError, match="different_cohort"):
        EvidenceGraph.model_validate(payload)


def test_graph_rejects_duplicate_publication_identity() -> None:
    first = adapt_effect_evidence(_evidence("a"), context=_context("a")).graph
    second = adapt_effect_evidence(_evidence("b"), context=_context("b")).graph
    payload = _merge_graphs(first, second).model_dump(mode="json")
    payload["publications"][1]["doi"] = payload["publications"][0]["doi"]

    with pytest.raises(ValidationError, match="publication_doi_values_not_unique"):
        EvidenceGraph.model_validate(payload)


def test_graph_rejects_dangling_risk_of_bias_evidence_span() -> None:
    graph = adapt_effect_evidence(_evidence("a"), context=_context("a")).graph
    payload = graph.model_dump(mode="json")
    payload["outcome_estimates"][0]["risk_of_bias"] = {
        "tool": "RoB 2",
        "overall": "some_concerns",
        "assessor": "reviewer-1",
        "domains": [
            {
                "domain_id": "randomization",
                "judgement": "some_concerns",
                "rationale": "Allocation concealment was unclear.",
                "evidence_span_ids": ["missing-span"],
            }
        ],
    }

    with pytest.raises(ValidationError, match="missing_risk_of_bias_evidence_span"):
        EvidenceGraph.model_validate(payload)


def test_graph_rejects_orphan_evidence_span_publication() -> None:
    graph = adapt_effect_evidence(_evidence("a"), context=_context("a")).graph
    payload = graph.model_dump(mode="json")
    payload["evidence_spans"].append(
        {
            "span_id": "orphan-span",
            "publication_id": "missing-publication",
            "source_locator": "missing.pdf#page=1",
            "quote": "A source passage from a publication absent from the graph.",
        }
    )

    with pytest.raises(ValidationError, match="missing_evidence_span_publication"):
        EvidenceGraph.model_validate(payload)


def test_evidence_span_rejects_locator_only_without_exact_representation() -> None:
    graph = adapt_effect_evidence(_evidence("a"), context=_context("a")).graph
    payload = graph.model_dump(mode="json")
    payload["evidence_spans"][0].update(
        quote=None,
        char_start=None,
        char_end=None,
        line_ids=[],
    )

    with pytest.raises(ValidationError, match="requires_quote_offsets_or_line_ids"):
        EvidenceGraph.model_validate(payload)


def test_evidence_span_accepts_exact_offsets_without_quote() -> None:
    graph = adapt_effect_evidence(_evidence("a"), context=_context("a")).graph
    payload = graph.model_dump(mode="json")
    payload["evidence_spans"][0].update(
        quote=None,
        char_start=10,
        char_end=25,
        line_ids=[],
    )

    validated = EvidenceGraph.model_validate(payload)

    assert validated.evidence_spans[0].quote is None
    assert validated.evidence_spans[0].char_start == 10
    assert graph_risk_features(validated).fraction_missing_source_quote == 1


def test_estimate_spans_must_match_effect_paper_publication_exactly() -> None:
    graph = adapt_effect_evidence(_evidence("a"), context=_context("a")).graph
    payload = graph.model_dump(mode="json")
    payload["publications"].append(
        {
            "publication_id": "publication-b",
            "paper_id": "paper-b",
        }
    )
    payload["studies"][0]["publication_ids"].append("publication-b")
    payload["studies"][0]["publication_ids"].sort()
    payload["evidence_spans"][0]["publication_id"] = "publication-b"

    with pytest.raises(ValidationError, match="mismatch_effect_paper"):
        EvidenceGraph.model_validate(payload)


def test_selection_refuses_unresolved_placeholder_cohort_identity() -> None:
    context = _context("a", cohort_basis="legacy_placeholder")
    result = adapt_effect_evidence(_evidence("a"), context=context)
    selection = select_effect_evidence(result.graph)

    assert result.status == "requires_review"
    assert selection.status == "insufficient"
    assert selection.reason == "unresolved_cohort_identity"


def test_synthesis_aggregates_one_cohort_reported_by_multiple_publications() -> None:
    first = adapt_effect_evidence(_evidence("a"), context=_context("a")).graph
    second = adapt_effect_evidence(_evidence("b"), context=_context("b")).graph
    payload = first.model_dump(mode="json")
    payload["publications"].extend(second.model_dump(mode="json")["publications"])
    payload["studies"][0]["publication_ids"].append("publication-b")
    payload["studies"][0]["publication_ids"].sort()
    second_span = second.evidence_spans[0].model_copy(
        update={"span_id": "span-second-publication"}
    )
    payload["evidence_spans"].append(second_span.model_dump(mode="json"))
    second_estimate = second.outcome_estimates[0].model_copy(
        update={
            "estimate_id": "estimate-second-publication",
            "contrast_id": first.contrasts[0].contrast_id,
            "evidence_span_ids": [second_span.span_id],
        }
    )
    payload["outcome_estimates"].append(second_estimate.model_dump(mode="json"))
    graph = EvidenceGraph.model_validate(payload)

    independent = adapt_effect_evidence(
        _evidence("c", estimate=0.3), context=_context("c")
    ).graph
    graph = _merge_graphs(graph, independent)

    selection = select_effect_evidence(graph)
    synthesis = synthesize_evidence_graph(graph)

    assert selection.status == "ready"
    assert selection.cohort_ids.count("cohort-a") == 2
    assert "shared_cohort_across_publications_aggregated_as_one_unit" in selection.warnings
    assert synthesis["quantitative"]["n_publications"] == 3
    assert synthesis["quantitative"]["n_cohorts"] == 2
    shared = next(
        row
        for row in synthesis["quantitative"]["cohort_effects"]
        if row["cohort_id"] == "cohort-a"
    )
    assert shared["paper_ids"] == ["paper-a", "paper-b"]
    assert shared["variance"] == pytest.approx(0.1**2)


def test_synthesis_allows_one_publication_reporting_multiple_cohorts() -> None:
    first = adapt_effect_evidence(_evidence("a"), context=_context("a")).graph
    second_evidence = _evidence("b").model_copy(update={"paper_id": "paper-a"})
    second = adapt_effect_evidence(
        second_evidence,
        context=_context("b", paper_id="paper-a"),
    ).graph
    payload = first.model_dump(mode="json")
    second_payload = second.model_dump(mode="json")
    second_payload["studies"][0]["publication_ids"] = ["publication-a"]
    second_payload["studies"][0]["primary_publication_id"] = "publication-a"
    second_payload["evidence_spans"][0]["publication_id"] = "publication-a"
    for key in ("studies", "cohorts", "arms", "contrasts", "outcome_estimates", "evidence_spans"):
        payload[key].extend(second_payload[key])
    graph = EvidenceGraph.model_validate(payload)

    selection = select_effect_evidence(graph)
    synthesis = synthesize_evidence_graph(graph)

    assert selection.status == "ready"
    assert set(selection.cohort_ids) == {"cohort-a", "cohort-b"}
    assert (
        "multiple_explicit_cohorts_in_publication_treated_as_distinct_units"
        in selection.warnings
    )
    assert synthesis["quantitative"]["n_publications"] == 1
    assert synthesis["quantitative"]["n_cohorts"] == 2
    assert {row["cohort_id"] for row in synthesis["quantitative"]["cohort_effects"]} == {
        "cohort-a",
        "cohort-b",
    }


def test_selection_refuses_incompatible_timepoints() -> None:
    first = adapt_effect_evidence(_evidence("a"), context=_context("a")).graph
    second_context = _context("b").model_copy(
        update={
            "timepoint": OutcomeTimepoint(
                kind="exact", value=8, unit="week", anchor="baseline"
            )
        }
    )
    second = adapt_effect_evidence(_evidence("b"), context=second_context).graph

    selection = select_effect_evidence(_merge_graphs(first, second))
    assert selection.status == "insufficient"
    assert selection.reason == "incompatible_timepoints"


def test_selection_refuses_inconsistent_contrast_orientation_signatures() -> None:
    first = adapt_effect_evidence(_evidence("a"), context=_context("a")).graph
    reversed_context = _context("b").model_copy(
        update={"positive_direction_means": "higher outcome value under comparator"}
    )
    second = adapt_effect_evidence(_evidence("b"), context=reversed_context).graph

    selection = select_effect_evidence(_merge_graphs(first, second))

    assert selection.status == "insufficient"
    assert selection.reason == "incompatible_contrast_orientations"


def test_graph_synthesis_runs_only_for_safe_cohort_publication_mapping() -> None:
    first = adapt_effect_evidence(_evidence("a", estimate=0.2), context=_context("a")).graph
    second = adapt_effect_evidence(_evidence("b", estimate=0.4), context=_context("b")).graph
    graph = _merge_graphs(first, second)

    synthesis = synthesize_evidence_graph(graph, outcome_name="performance")

    assert synthesis["status"] == "ok"
    assert synthesis["mode"] == "random_effects_meta_analysis"
    assert synthesis["quantitative"]["n_papers"] == 2
    assert synthesis["evidence_graph"]["selection_status"] == "ready"
    assert synthesis["evidence_graph"]["risk_feature_interpretation"] == (
        "prospective_label_free_inputs_not_a_calibrated_error_probability"
    )


def test_risk_features_are_label_free_and_expose_missing_assessments() -> None:
    graph = adapt_effect_evidence(_evidence("a"), context=_context("a")).graph
    features = graph_risk_features(graph)

    assert features.feature_schema_version == "1"
    assert features.n_estimates == 1
    assert features.n_publications == 1
    assert features.n_cohorts == 1
    assert features.fraction_non_estimable == 0
    assert features.fraction_missing_source_quote == 0
    assert features.fraction_timepoint_not_reported == 0
    assert features.fraction_risk_of_bias_not_assessed == 1
    assert "correct" not in features.model_dump()
    calibration_features = features.as_calibration_features()
    assert list(calibration_features) == sorted(calibration_features)
    assert calibration_features["n_estimates"] == 1.0
    assert "feature_schema_version" not in calibration_features


def test_risk_features_preserve_high_risk_judgement_with_span_provenance() -> None:
    graph = adapt_effect_evidence(_evidence("a"), context=_context("a")).graph
    payload = graph.model_dump(mode="json")
    span_id = payload["evidence_spans"][0]["span_id"]
    payload["outcome_estimates"][0]["risk_of_bias"] = {
        "tool": "RoB 2",
        "overall": "high",
        "assessor": "reviewer-1",
        "domains": [
            {
                "domain_id": "missing_outcome_data",
                "judgement": "high",
                "rationale": "Attrition differed by arm.",
                "evidence_span_ids": [span_id],
            }
        ],
    }
    validated = EvidenceGraph.model_validate(payload)

    features = graph_risk_features(validated)
    assert features.fraction_risk_of_bias_not_assessed == 0
    assert features.fraction_high_or_critical_risk_of_bias == 1


def test_evidence_graph_schema_is_closed_and_versioned() -> None:
    schema = evidence_graph_json_schema()

    assert schema["$id"] == "urn:literature-multiverse:evidence-graph:v1"
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
