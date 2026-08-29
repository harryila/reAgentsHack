from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from scripts.build_typed_evidence_corpus import main as build_typed_corpus_main

from literature_multiverse.effects import EffectEvidence
from literature_multiverse.evidence_graph import (
    CohortIdentity,
    EvidenceGraph,
    GraphAdapterContext,
    OutcomeTimepoint,
    PublicationIdentity,
    adapt_effect_evidence,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.meta_analysis import synthesize_evidence_graph
from literature_multiverse.typed_extraction import (
    FragmentStatus,
    NonEstimabilityReason,
    PublicationEvidenceFragment,
    SourceDocumentArtifact,
    TypedExtractionContractError,
    assemble_typed_evidence_corpus,
    freeze_publication_evidence_fragment,
    publication_fragment_json_schema,
)
from literature_multiverse.verifier import (
    ClaimManifest,
    LegacyAdapterConfig,
    ScientificClaim,
    VerificationContractError,
    VerificationProtocol,
    load_corpus,
    run_verification,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
GROUNDING_HASH = "c" * 64


def _graph(suffix: str, *, estimate: float = 0.2):
    context = GraphAdapterContext(
        publication=PublicationIdentity(
            publication_id=f"publication-{suffix}",
            paper_id=f"paper-{suffix}",
            doc_id=f"document-{suffix}",
            doi=f"10.1000/{suffix}",
        ),
        study_id=f"study-{suffix}",
        cohort_identity=CohortIdentity(
            cohort_id=f"cohort-{suffix}",
            basis="reviewer_reconciled",
            rationale="Two reviewers reconciled the cohort identity.",
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
    evidence = EffectEvidence(
        paper_id=f"paper-{suffix}",
        finding_id=f"finding-{suffix}",
        outcome="performance",
        contrast="intervention_vs_control",
        effect_format="hedges_g",
        estimate=estimate,
        standard_error=0.1,
        provenance={
            "source_locator": f"paper-{suffix}.pdf#page=4",
            "source_quote": f"The standardized estimate was {estimate}.",
        },
    )
    return adapt_effect_evidence(evidence, context=context).graph


def _source(suffix: str, *, sha256: str = HASH_A) -> SourceDocumentArtifact:
    return SourceDocumentArtifact(
        artifact_path=f"data/raw/documents/{suffix}.pdf",
        sha256=sha256,
        media_type="application/pdf",
        source_locator=f"archive:{suffix}",
    )


def _estimable(
    suffix: str,
    *,
    pipeline: str = HASH_B,
    grounding_receipt_sha256: str = GROUNDING_HASH,
):
    return freeze_publication_evidence_fragment(
        question_id="typed-question",
        publication_id=f"publication-{suffix}",
        paper_id=f"paper-{suffix}",
        publication=_graph(suffix).publications[0],
        pipeline_fingerprint_sha256=pipeline,
        source_document=_source(suffix),
        grounding_receipt_sha256=grounding_receipt_sha256,
        status=FragmentStatus.ESTIMABLE,
        graph=_graph(suffix),
    )


def test_native_fragments_merge_into_a_hash_bound_evidence_corpus() -> None:
    corpus = assemble_typed_evidence_corpus([_estimable("b"), _estimable("a")])

    assert [item.publication_id for item in corpus.fragments] == [
        "publication-a",
        "publication-b",
    ]
    assert corpus.graph is not None
    assert len(corpus.graph.outcome_estimates) == 2
    assert corpus.estimable_publication_ids == ["publication-a", "publication-b"]
    assert corpus.non_estimable_publication_ids == []
    assert len(corpus.corpus_sha256) == 64


def test_non_estimable_publication_remains_in_complete_corpus_accounting() -> None:
    missing = freeze_publication_evidence_fragment(
        question_id="typed-question",
        publication_id="publication-missing",
        paper_id="paper-missing",
        publication=PublicationIdentity(
            publication_id="publication-missing", paper_id="paper-missing"
        ),
        pipeline_fingerprint_sha256=HASH_B,
        source_document=_source("missing"),
        grounding_receipt_sha256=None,
        status=FragmentStatus.NON_ESTIMABLE,
        non_estimability_reason=NonEstimabilityReason.UNCERTAINTY_ABSENT,
        non_estimability_detail="The paper reports a point estimate without uncertainty.",
    )

    corpus = assemble_typed_evidence_corpus([_estimable("a"), missing])

    assert corpus.non_estimable_publication_ids == ["publication-missing"]
    assert corpus.issues[0].code == "non_estimable:uncertainty_absent"
    assert corpus.graph is not None
    assert len(corpus.graph.publications) == 2


def test_fully_rehashed_corpus_cannot_drop_fragment_derived_blocking_issue() -> None:
    missing = freeze_publication_evidence_fragment(
        question_id="typed-question",
        publication_id="publication-missing",
        paper_id="paper-missing",
        publication=PublicationIdentity(
            publication_id="publication-missing", paper_id="paper-missing"
        ),
        pipeline_fingerprint_sha256=HASH_B,
        source_document=_source("missing"),
        grounding_receipt_sha256=None,
        status=FragmentStatus.NON_ESTIMABLE,
        non_estimability_reason=NonEstimabilityReason.UNCERTAINTY_ABSENT,
    )
    corpus = assemble_typed_evidence_corpus([_estimable("a"), missing])
    payload = corpus.model_dump(mode="json", exclude={"corpus_sha256"})
    payload["issues"] = []

    with pytest.raises(
        ValidationError,
        match="typed_evidence_issues_fragment_projection_mismatch",
    ):
        type(corpus).model_validate(
            {**payload, "corpus_sha256": hash_canonical(payload)}
        )


def test_all_non_estimable_fragments_preserve_publications_in_empty_effect_graph() -> None:
    fragment = freeze_publication_evidence_fragment(
        question_id="typed-question",
        publication_id="publication-missing",
        paper_id="paper-missing",
        publication=PublicationIdentity(
            publication_id="publication-missing", paper_id="paper-missing"
        ),
        pipeline_fingerprint_sha256=HASH_B,
        source_document=_source("missing"),
        grounding_receipt_sha256=None,
        status=FragmentStatus.NON_ESTIMABLE,
        non_estimability_reason=NonEstimabilityReason.NO_TARGET_OUTCOME,
    )

    corpus = assemble_typed_evidence_corpus([fragment])

    assert [item.publication_id for item in corpus.graph.publications] == [
        "publication-missing"
    ]
    assert corpus.graph.studies == []
    assert corpus.graph.cohorts == []
    assert corpus.graph.arms == []
    assert corpus.graph.contrasts == []
    assert corpus.graph.outcome_estimates == []
    assert corpus.graph.evidence_spans == []
    assert corpus.estimable_publication_ids == []


def test_fully_rehashed_corpus_cannot_inject_graph_only_nodes() -> None:
    fragment = freeze_publication_evidence_fragment(
        question_id="typed-question",
        publication_id="publication-missing",
        paper_id="paper-missing",
        publication=PublicationIdentity(
            publication_id="publication-missing", paper_id="paper-missing"
        ),
        pipeline_fingerprint_sha256=HASH_B,
        source_document=_source("missing"),
        grounding_receipt_sha256=None,
        status=FragmentStatus.NON_ESTIMABLE,
        non_estimability_reason=NonEstimabilityReason.NO_TARGET_OUTCOME,
    )
    corpus = assemble_typed_evidence_corpus([fragment])
    payload = corpus.model_dump(mode="json", exclude={"corpus_sha256"})
    payload["graph"]["publications"][0]["title"] = "Injected graph-only title"

    with pytest.raises(ValidationError, match="graph_fragment_projection_mismatch"):
        type(corpus).model_validate(
            {**payload, "corpus_sha256": hash_canonical(payload)}
        )


def test_all_non_estimable_typed_corpus_loads_as_not_evaluable(
    tmp_path: Path,
) -> None:
    fragment = freeze_publication_evidence_fragment(
        question_id="typed-question",
        publication_id="publication-missing",
        paper_id="paper-missing",
        publication=PublicationIdentity(
            publication_id="publication-missing", paper_id="paper-missing"
        ),
        pipeline_fingerprint_sha256=HASH_B,
        source_document=_source("missing"),
        grounding_receipt_sha256=None,
        status=FragmentStatus.NON_ESTIMABLE,
        non_estimability_reason=NonEstimabilityReason.NO_TARGET_OUTCOME,
    )
    corpus = assemble_typed_evidence_corpus([fragment])
    path = tmp_path / "typed_evidence_corpus.json"
    path.write_text(json.dumps(corpus.model_dump(mode="json")))

    loaded = load_corpus(path, legacy_settings=LegacyAdapterConfig())
    synthesis = synthesize_evidence_graph(
        loaded.graph,
        outcome_name="score",
    )

    assert loaded.source_format == "typed_evidence_corpus_json"
    assert loaded.provenance_assurance.status == "unverified_source_provenance"
    assert synthesis["status"] == "insufficient"
    assert synthesis["evidence_graph"]["selection_reason"] == "no_matching_estimates"

    certificate = run_verification(
        manifest=ClaimManifest(
            question_id="typed-question",
            population_id="typed-population",
            domain="biomedicine",
            claim=ScientificClaim(
                statement="The intervention increases score.",
                direction="increase",
                outcome_name="score",
            ),
            protocol=VerificationProtocol(corpus_cutoff="closed-corpus-v1"),
        ),
        corpus=loaded,
        budget_minutes=10,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )
    assert certificate.status == "abstained"
    assert certificate.release_assessment.evidence.classification == "not_evaluable"
    assert certificate.release_assessment.evidence.reason == "no_matching_estimates"
    assert certificate.counterfactual_reruns == []
    assert certificate.release_assessment.audit.status == "not_applicable"
    assert "adapter:non_estimable:no_target_outcome" in certificate.reasons
    assert "adapter:unverified_source_provenance" in certificate.reasons


def test_fragment_hash_rejects_nested_graph_mutation() -> None:
    fragment = _estimable("a")
    payload = fragment.model_dump(mode="json")
    assert payload["graph"] is not None
    payload["graph"]["outcome_estimates"][0]["effect"]["estimate"] = 99

    with pytest.raises(ValidationError, match="fragment_hash_mismatch"):
        PublicationEvidenceFragment.model_validate(payload)


def test_estimable_fragment_requires_hash_bound_grounding_receipt() -> None:
    with pytest.raises(ValidationError, match="requires_grounding_receipt"):
        freeze_publication_evidence_fragment(
            question_id="typed-question",
            publication_id="publication-a",
            paper_id="paper-a",
            publication=_graph("a").publications[0],
            pipeline_fingerprint_sha256=HASH_B,
            source_document=_source("a"),
            grounding_receipt_sha256=None,
            status=FragmentStatus.ESTIMABLE,
            graph=_graph("a"),
        )


def test_fragment_rejects_mismatched_publication_identity() -> None:
    fragment = _estimable("a")
    payload = fragment.model_dump(mode="json")
    payload["publication_id"] = "publication-other"
    payload_without_hash = deepcopy(payload)
    payload_without_hash.pop("fragment_sha256")
    payload["fragment_sha256"] = hash_canonical(payload_without_hash)

    with pytest.raises(ValidationError, match="publication_id_mismatch"):
        PublicationEvidenceFragment.model_validate(payload)


def test_assembly_fails_closed_on_conflicting_global_node_identity() -> None:
    first = _estimable("a")
    second_graph = _graph("b")
    second_payload = second_graph.model_dump(mode="json")
    second_payload["studies"][0]["study_id"] = "study-a"
    second_payload["cohorts"][0]["study_id"] = "study-a"
    second = freeze_publication_evidence_fragment(
        question_id="typed-question",
        publication_id="publication-b",
        paper_id="paper-b",
        publication=second_graph.publications[0],
        pipeline_fingerprint_sha256=HASH_B,
        source_document=_source("b"),
        grounding_receipt_sha256=GROUNDING_HASH,
        status=FragmentStatus.ESTIMABLE,
        graph=EvidenceGraph.model_validate(second_payload),
    )

    with pytest.raises(TypedExtractionContractError, match="identity_collision"):
        assemble_typed_evidence_corpus([first, second])


def test_publication_fragment_schema_is_versioned() -> None:
    schema = publication_fragment_json_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith(":v3")


def test_unified_verifier_rejects_estimable_typed_corpus_without_receipts(
    tmp_path: Path,
) -> None:
    corpus = assemble_typed_evidence_corpus([_estimable("a")])
    path = tmp_path / "typed_evidence_corpus.json"
    path.write_text(json.dumps(corpus.model_dump(mode="json")))

    with pytest.raises(
        VerificationContractError,
        match="requires_grounding_package",
    ):
        load_corpus(path, legacy_settings=LegacyAdapterConfig())


def test_typed_corpus_cli_rejects_estimable_fragments_without_receipts(
    tmp_path: Path,
) -> None:
    fragment_path = tmp_path / "fragments.jsonl"
    fragment_path.write_text(
        json.dumps(_estimable("a").model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"

    with pytest.raises(
        ValueError,
        match="estimable_typed_corpus_requires_grounding_receipts",
    ):
        build_typed_corpus_main(
            [
                "--fragments",
                str(fragment_path),
                "--output-dir",
                str(output),
            ]
        )


def test_typed_corpus_cli_keeps_unlinked_non_estimable_fragments(
    tmp_path: Path,
) -> None:
    fragment = freeze_publication_evidence_fragment(
        question_id="typed-question",
        publication_id="publication-missing",
        paper_id="paper-missing",
        publication=PublicationIdentity(
            publication_id="publication-missing", paper_id="paper-missing"
        ),
        pipeline_fingerprint_sha256=HASH_B,
        source_document=_source("missing"),
        grounding_receipt_sha256=None,
        status=FragmentStatus.NON_ESTIMABLE,
        non_estimability_reason=NonEstimabilityReason.NO_TARGET_OUTCOME,
    )
    fragment_path = tmp_path / "fragments.jsonl"
    fragment_path.write_text(
        json.dumps(fragment.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"

    assert (
        build_typed_corpus_main(
            [
                "--fragments",
                str(fragment_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    assert (output / "typed_evidence_corpus.json").is_file()
    assert (output / "typed_evidence_grounding_package.json").is_file()
    graph = json.loads((output / "evidence_graph.json").read_text())
    assert [row["publication_id"] for row in graph["publications"]] == [
        "publication-missing"
    ]
    assert graph["outcome_estimates"] == []
    assert (output / "publication_evidence_fragment.schema.json").is_file()
