from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError
from scripts.authorize_synthesis_unit import main as authorization_cli_main
from tests.test_cohort_reconciliation import (
    _cohort_payload,
    _fragment,
    _two_publication_corpus,
)

from literature_multiverse.cohort_reconciliation import reconcile_native_cohorts
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.native_grounding import resolve_native_source_document
from literature_multiverse.synthesis_unit_authorization import (
    SynthesisAuthorizationError,
    authorize_synthesis_unit,
    freeze_source_identity_assertion,
    freeze_source_identity_citation,
)
from literature_multiverse.typed_extraction import (
    SourceDocumentArtifact,
    assemble_typed_evidence_corpus,
    freeze_publication_evidence_fragment,
)


def _materialize_sources(corpus, root: Path, *, identifier_text_suffix: str = ""):
    fragments = []
    for fragment in corpus.fragments:
        assert fragment.graph is not None
        cohort = fragment.graph.cohorts[0]
        identifier = (cohort.identity.registry_ids or cohort.identity.dataset_ids)[0]
        doc_id = fragment.publication.doc_id
        path = root / f"source-{fragment.publication_id}.json"
        path.write_text(
            json.dumps(
                {
                    doc_id: {
                        "L1": {
                            "section": "Methods",
                            "text": (
                                "Registry identifier "
                                f"{identifier}{identifier_text_suffix} identifies this "
                                "participant cohort."
                            ),
                        },
                        "L2": {
                            "section": "Methods",
                            "text": "The source separately describes participant enrollment.",
                        },
                    }
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        source = SourceDocumentArtifact(
            artifact_path=path.name,
            sha256=sha256_file(path),
            media_type="application/json",
            source_locator=f"json:{path.name}#/{doc_id}",
        )
        fragments.append(
            freeze_publication_evidence_fragment(
                question_id=fragment.question_id,
                publication_id=fragment.publication_id,
                paper_id=fragment.paper_id,
                publication=fragment.publication,
                pipeline_fingerprint_sha256=fragment.pipeline_fingerprint_sha256,
                source_document=source,
                grounding_receipt_sha256=fragment.grounding_receipt_sha256,
                status=fragment.status,
                graph=fragment.graph,
                extractor_warnings=fragment.extractor_warnings,
            )
        )
    return assemble_typed_evidence_corpus(fragments)


def _citation(corpus, publication_id: str, identifier: str, root: Path):
    fragment = next(row for row in corpus.fragments if row.publication_id == publication_id)
    assert fragment.graph is not None
    source = resolve_native_source_document(
        repository_root=root, source_document=fragment.source_document
    )
    lines = [source.lines[0]]
    quote = lines[0].text
    return freeze_source_identity_citation(
        publication_id=publication_id,
        original_cohort_id=fragment.graph.cohorts[0].cohort_id,
        source_document_sha256=fragment.source_document.sha256,
        grounding_receipt_sha256=fragment.grounding_receipt_sha256,
        source_locator=source.source_locator,
        quote=quote,
        line_ids=[lines[0].line_id],
        cited_identifier=identifier,
        source_payload_sha256=source.source_payload_sha256,
        cited_lines_sha256=hash_canonical(lines),
        cited_text_sha256=hashlib.sha256(quote.encode("utf-8")).hexdigest(),
    )


def _assertion(relationship: str, cohort_ids: list[str], citations):
    citations = sorted(citations, key=lambda row: row.citation_sha256)
    return freeze_source_identity_assertion(
        relationship=relationship,
        cohort_ids=sorted(cohort_ids),
        rationale="Frozen source review established this identity relationship.",
        citations=citations,
        reviewer_identity_sha256="c" * 64,
        review_protocol_sha256="d" * 64,
    )


def _rehash_citation(citation, **updates):
    payload = citation.model_dump(mode="json", exclude={"citation_sha256"})
    payload.update(updates)
    return type(citation).model_validate({**payload, "citation_sha256": hash_canonical(payload)})


def _rehash_assertion(assertion, **updates):
    payload = assertion.model_dump(mode="json", exclude={"assertion_sha256"})
    payload.update(updates)
    return type(assertion).model_validate({**payload, "assertion_sha256": hash_canonical(payload)})


def test_merged_single_cohort_requires_source_assertion_then_authorizes(tmp_path) -> None:
    corpus = _materialize_sources(
        _two_publication_corpus(first_registry=["NCT 00000001"], second_registry=["nct 00000001"]),
        tmp_path,
    )
    reconciliation = reconcile_native_cohorts(corpus=corpus)
    group = next(row for row in reconciliation.cohort_groups if len(row.member_ids) == 2)
    estimate_ids = sorted(
        row.estimate_id for row in reconciliation.reconciled_graph.outcome_estimates
    )

    with pytest.raises(
        SynthesisAuthorizationError,
        match="merged_cohort_lacks_source_identity_assertion",
    ):
        authorize_synthesis_unit(
            corpus=corpus,
            reconciliation=reconciliation,
            estimate_ids=estimate_ids,
            assertions=[],
            repository_root=tmp_path,
        )

    same = _assertion(
        "same_cohort",
        group.member_ids,
        [
            _citation(corpus, "publication-1", "NCT 00000001", tmp_path),
            _citation(corpus, "publication-2", "nct 00000001", tmp_path),
        ],
    )
    receipt = authorize_synthesis_unit(
        corpus=corpus,
        reconciliation=reconciliation,
        estimate_ids=estimate_ids,
        assertions=[same],
        repository_root=tmp_path,
    )
    assert receipt.authorizes_synthesis_input is True
    assert receipt.authorization_basis == ("single_cohort_cross_paper_independence_irrelevant")
    assert receipt.unresolved_overlap_pairs == []


def test_two_cohorts_abstain_until_exact_pairwise_independence_is_supported(tmp_path) -> None:
    corpus = _materialize_sources(
        _two_publication_corpus(first_registry=["NCT-ONE"], second_registry=["NCT-TWO"]),
        tmp_path,
    )
    reconciliation = reconcile_native_cohorts(corpus=corpus)
    estimate_ids = sorted(
        row.estimate_id for row in reconciliation.reconciled_graph.outcome_estimates
    )
    cohort_ids = sorted(row.canonical_id for row in reconciliation.cohort_groups)

    abstained = authorize_synthesis_unit(
        corpus=corpus,
        reconciliation=reconciliation,
        estimate_ids=estimate_ids,
        assertions=[],
        repository_root=tmp_path,
    )
    assert abstained.authorizes_synthesis_input is False
    assert abstained.unresolved_overlap_pairs == [cohort_ids]

    independent = _assertion(
        "independent_cohorts",
        cohort_ids,
        [
            _citation(corpus, "publication-1", "NCT-ONE", tmp_path),
            _citation(corpus, "publication-2", "NCT-TWO", tmp_path),
        ],
    )
    authorized = authorize_synthesis_unit(
        corpus=corpus,
        reconciliation=reconciliation,
        estimate_ids=estimate_ids,
        assertions=[independent],
        repository_root=tmp_path,
    )
    assert authorized.authorizes_synthesis_input is True
    assert authorized.authorization_basis == "all_pairwise_independence_source_adjudicated"


def test_publication_separation_and_labels_alone_never_authorize_independence(tmp_path) -> None:
    corpus = _materialize_sources(
        _two_publication_corpus(first_registry=["NCT-ONE"], second_registry=["NCT-TWO"]),
        tmp_path,
    )
    reconciliation = reconcile_native_cohorts(corpus=corpus)
    receipt = authorize_synthesis_unit(
        corpus=corpus,
        reconciliation=reconciliation,
        estimate_ids=sorted(
            row.estimate_id for row in reconciliation.reconciled_graph.outcome_estimates
        ),
        assertions=[],
        repository_root=tmp_path,
    )
    assert receipt.authorizes_synthesis_input is False


def test_citation_must_match_immutable_source_and_grounding_lineage(tmp_path) -> None:
    corpus = _materialize_sources(
        _two_publication_corpus(first_registry=["NCT-ONE"], second_registry=["NCT-TWO"]),
        tmp_path,
    )
    reconciliation = reconcile_native_cohorts(corpus=corpus)
    cohort_ids = sorted(row.canonical_id for row in reconciliation.cohort_groups)
    bad = _citation(corpus, "publication-1", "NCT-ONE", tmp_path).model_dump(mode="json")
    bad["source_document_sha256"] = "f" * 64
    bad.pop("citation_sha256")
    bad["citation_sha256"] = hash_canonical(bad)
    assertion = _assertion(
        "independent_cohorts",
        cohort_ids,
        [
            type(_citation(corpus, "publication-1", "NCT-ONE", tmp_path)).model_validate(bad),
            _citation(corpus, "publication-2", "NCT-TWO", tmp_path),
        ],
    )
    with pytest.raises(
        SynthesisAuthorizationError,
        match="assertion_citation_source_lineage_mismatch",
    ):
        authorize_synthesis_unit(
            corpus=corpus,
            reconciliation=reconciliation,
            estimate_ids=sorted(
                row.estimate_id for row in reconciliation.reconciled_graph.outcome_estimates
            ),
            assertions=[assertion],
            repository_root=tmp_path,
        )


def test_assertion_hash_tampering_is_rejected(tmp_path) -> None:
    corpus = _materialize_sources(
        _two_publication_corpus(first_registry=["NCT-ONE"], second_registry=["NCT-TWO"]),
        tmp_path,
    )
    assertion = _assertion(
        "independent_cohorts",
        ["cohort-a", "cohort-b"],
        [
            _citation(corpus, "publication-1", "NCT-ONE", tmp_path),
            _citation(corpus, "publication-2", "NCT-TWO", tmp_path),
        ],
    )
    payload = deepcopy(assertion.model_dump(mode="json"))
    payload["rationale"] = "Changed after review."
    with pytest.raises(ValidationError, match="assertion_hash_mismatch"):
        type(assertion).model_validate(payload)


def test_model_label_disguised_as_identifier_is_rejected(tmp_path) -> None:
    corpus = _materialize_sources(
        _two_publication_corpus(first_registry=["NCT-ONE"], second_registry=["NCT-TWO"]),
        tmp_path,
    )
    reconciliation = reconcile_native_cohorts(corpus=corpus)
    cohort_ids = sorted(row.canonical_id for row in reconciliation.cohort_groups)
    assertion = _assertion(
        "independent_cohorts",
        cohort_ids,
        [
            _citation(corpus, "publication-1", "invented-model-label", tmp_path),
            _citation(corpus, "publication-2", "NCT-TWO", tmp_path),
        ],
    )
    with pytest.raises(
        SynthesisAuthorizationError,
        match="citation_identifier_not_in_source_cohort",
    ):
        authorize_synthesis_unit(
            corpus=corpus,
            reconciliation=reconciliation,
            estimate_ids=sorted(
                row.estimate_id for row in reconciliation.reconciled_graph.outcome_estimates
            ),
            assertions=[assertion],
            repository_root=tmp_path,
        )


def test_cli_writes_hash_bound_abstention(tmp_path, capsys) -> None:
    corpus = _materialize_sources(
        _two_publication_corpus(first_registry=["NCT-ONE"], second_registry=["NCT-TWO"]),
        tmp_path,
    )
    reconciliation = reconcile_native_cohorts(corpus=corpus)
    corpus_path = tmp_path / "corpus.json"
    reconciliation_path = tmp_path / "reconciliation.json"
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "authorization.json"
    corpus_path.write_text(corpus.model_dump_json(), encoding="utf-8")
    reconciliation_path.write_text(reconciliation.model_dump_json(), encoding="utf-8")
    request_path.write_text(
        json.dumps(
            {
                "request_version": "source-backed-synthesis-request-v1",
                "estimate_ids": sorted(
                    row.estimate_id for row in reconciliation.reconciled_graph.outcome_estimates
                ),
                "assertions": [],
            }
        ),
        encoding="utf-8",
    )
    assert (
        authorization_cli_main(
            [
                "--corpus",
                str(corpus_path),
                "--reconciliation",
                str(reconciliation_path),
                "--repository-root",
                str(tmp_path),
                "--request",
                str(request_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert printed["authorizes_synthesis_input"] is False
    assert written["receipt_sha256"] == printed["receipt_sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"source_locator": "json:source-publication-1.json#/doc-1#escape"}, "locator_not_exact"),
        ({"line_ids": ["L999"]}, "line_unknown"),
        ({"quote": "An invented passage."}, "quote_not_exact"),
    ],
)
def test_citation_coordinates_are_verified_against_source_bytes(
    tmp_path, mutation, message
) -> None:
    corpus = _materialize_sources(
        _two_publication_corpus(first_registry=["NCT-ONE"], second_registry=["NCT-TWO"]),
        tmp_path,
    )
    reconciliation = reconcile_native_cohorts(corpus=corpus)
    cohort_ids = sorted(row.canonical_id for row in reconciliation.cohort_groups)
    first = _rehash_citation(_citation(corpus, "publication-1", "NCT-ONE", tmp_path), **mutation)
    assertion = _assertion(
        "independent_cohorts",
        cohort_ids,
        [first, _citation(corpus, "publication-2", "NCT-TWO", tmp_path)],
    )
    with pytest.raises(SynthesisAuthorizationError, match=message):
        authorize_synthesis_unit(
            corpus=corpus,
            reconciliation=reconciliation,
            estimate_ids=sorted(
                row.estimate_id for row in reconciliation.reconciled_graph.outcome_estimates
            ),
            assertions=[assertion],
            repository_root=tmp_path,
        )


def test_identifier_must_occur_inside_exact_cited_lines(tmp_path) -> None:
    corpus = _materialize_sources(
        _two_publication_corpus(first_registry=["NCT-ONE"], second_registry=["NCT-TWO"]),
        tmp_path,
    )
    reconciliation = reconcile_native_cohorts(corpus=corpus)
    source = resolve_native_source_document(
        repository_root=tmp_path,
        source_document=corpus.fragments[0].source_document,
    )
    line = source.lines[1]
    first = _rehash_citation(
        _citation(corpus, "publication-1", "NCT-ONE", tmp_path),
        line_ids=[line.line_id],
        quote=line.text,
        cited_lines_sha256=hash_canonical([line]),
        cited_text_sha256=hashlib.sha256(line.text.encode("utf-8")).hexdigest(),
    )
    cohort_ids = sorted(row.canonical_id for row in reconciliation.cohort_groups)
    assertion = _assertion(
        "independent_cohorts",
        cohort_ids,
        [first, _citation(corpus, "publication-2", "NCT-TWO", tmp_path)],
    )
    with pytest.raises(SynthesisAuthorizationError, match="identifier_outside_cited_span"):
        authorize_synthesis_unit(
            corpus=corpus,
            reconciliation=reconciliation,
            estimate_ids=sorted(
                row.estimate_id for row in reconciliation.reconciled_graph.outcome_estimates
            ),
            assertions=[assertion],
            repository_root=tmp_path,
        )


def test_source_byte_mutation_invalidates_authorization(tmp_path) -> None:
    corpus = _materialize_sources(
        _two_publication_corpus(first_registry=["NCT-ONE"], second_registry=["NCT-TWO"]),
        tmp_path,
    )
    reconciliation = reconcile_native_cohorts(corpus=corpus)
    path = tmp_path / corpus.fragments[0].source_document.artifact_path
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source_artifact_hash_mismatch"):
        authorize_synthesis_unit(
            corpus=corpus,
            reconciliation=reconciliation,
            estimate_ids=sorted(
                row.estimate_id for row in reconciliation.reconciled_graph.outcome_estimates
            ),
            assertions=[],
            repository_root=tmp_path,
        )


def test_unused_and_conflicting_assertions_are_rejected(tmp_path) -> None:
    corpus = _materialize_sources(
        _two_publication_corpus(first_registry=["NCT-ONE"], second_registry=["NCT-TWO"]),
        tmp_path,
    )
    reconciliation = reconcile_native_cohorts(corpus=corpus)
    cohort_ids = sorted(row.canonical_id for row in reconciliation.cohort_groups)
    citations = [
        _citation(corpus, "publication-1", "NCT-ONE", tmp_path),
        _citation(corpus, "publication-2", "NCT-TWO", tmp_path),
    ]
    independent = _assertion("independent_cohorts", cohort_ids, citations)
    one_estimate = [reconciliation.reconciled_graph.outcome_estimates[0].estimate_id]
    with pytest.raises(SynthesisAuthorizationError, match="unused_independence_assertion"):
        authorize_synthesis_unit(
            corpus=corpus,
            reconciliation=reconciliation,
            estimate_ids=one_estimate,
            assertions=[independent],
            repository_root=tmp_path,
        )
    with pytest.raises(SynthesisAuthorizationError, match="duplicate_source_identity"):
        authorize_synthesis_unit(
            corpus=corpus,
            reconciliation=reconciliation,
            estimate_ids=sorted(
                row.estimate_id for row in reconciliation.reconciled_graph.outcome_estimates
            ),
            assertions=[independent, independent],
            repository_root=tmp_path,
        )
    conflicting = _rehash_assertion(independent, relationship="same_cohort")
    with pytest.raises(SynthesisAuthorizationError, match=r"conflicting.*relationships"):
        authorize_synthesis_unit(
            corpus=corpus,
            reconciliation=reconciliation,
            estimate_ids=sorted(
                row.estimate_id for row in reconciliation.reconciled_graph.outcome_estimates
            ),
            assertions=[independent, conflicting],
            repository_root=tmp_path,
        )


def test_cli_positive_receipt_replays_identically(tmp_path, capsys) -> None:
    corpus = _materialize_sources(
        _two_publication_corpus(first_registry=["NCT-ONE"], second_registry=["NCT-TWO"]),
        tmp_path,
    )
    reconciliation = reconcile_native_cohorts(corpus=corpus)
    assertion = _assertion(
        "independent_cohorts",
        sorted(row.canonical_id for row in reconciliation.cohort_groups),
        [
            _citation(corpus, "publication-1", "NCT-ONE", tmp_path),
            _citation(corpus, "publication-2", "NCT-TWO", tmp_path),
        ],
    )
    corpus_path = tmp_path / "replay-corpus.json"
    reconciliation_path = tmp_path / "replay-reconciliation.json"
    request_path = tmp_path / "replay-request.json"
    corpus_path.write_text(corpus.model_dump_json(), encoding="utf-8")
    reconciliation_path.write_text(reconciliation.model_dump_json(), encoding="utf-8")
    request_path.write_text(
        json.dumps(
            {
                "request_version": "source-backed-synthesis-request-v1",
                "estimate_ids": sorted(
                    row.estimate_id for row in reconciliation.reconciled_graph.outcome_estimates
                ),
                "assertions": [assertion.model_dump(mode="json")],
            }
        ),
        encoding="utf-8",
    )
    hashes = []
    for name in ("first.json", "second.json"):
        assert (
            authorization_cli_main(
                [
                    "--corpus",
                    str(corpus_path),
                    "--reconciliation",
                    str(reconciliation_path),
                    "--repository-root",
                    str(tmp_path),
                    "--request",
                    str(request_path),
                    "--output",
                    str(tmp_path / name),
                ]
            )
            == 0
        )
        hashes.append(json.loads(capsys.readouterr().out)["receipt_sha256"])
    assert hashes[0] == hashes[1]
    assert json.loads((tmp_path / "first.json").read_text())["authorizes_synthesis_input"]


def test_same_publication_two_cohorts_require_citation_for_each_cohort(tmp_path) -> None:
    corpus = _materialize_sources(
        assemble_typed_evidence_corpus(
            [
                _fragment(
                    1,
                    cohorts=[
                        _cohort_payload(
                            "cohort-a", registry_ids=["NCT-A"], dataset_ids=[], estimate=0.1
                        ),
                        _cohort_payload(
                            "cohort-b", registry_ids=["NCT-B"], dataset_ids=[], estimate=0.2
                        ),
                    ],
                )
            ]
        ),
        tmp_path,
    )
    reconciliation = reconcile_native_cohorts(corpus=corpus)
    cohort_ids = sorted(row.canonical_id for row in reconciliation.cohort_groups)
    first = _citation(corpus, "publication-1", "NCT-A", tmp_path)
    repeated = _rehash_citation(
        first,
        quote="Registry identifier NCT-A",
    )
    assertion = _assertion("independent_cohorts", cohort_ids, [first, repeated])
    with pytest.raises(SynthesisAuthorizationError, match="cover_exact_original_cohorts"):
        authorize_synthesis_unit(
            corpus=corpus,
            reconciliation=reconciliation,
            estimate_ids=sorted(
                row.estimate_id for row in reconciliation.reconciled_graph.outcome_estimates
            ),
            assertions=[assertion],
            repository_root=tmp_path,
        )


def test_identifier_prefix_collision_does_not_count_as_span_support(tmp_path) -> None:
    corpus = _materialize_sources(
        _two_publication_corpus(first_registry=["NCT123"], second_registry=["NCT999"]),
        tmp_path,
        identifier_text_suffix="4",
    )
    reconciliation = reconcile_native_cohorts(corpus=corpus)
    assertion = _assertion(
        "independent_cohorts",
        sorted(row.canonical_id for row in reconciliation.cohort_groups),
        [
            _citation(corpus, "publication-1", "NCT123", tmp_path),
            _citation(corpus, "publication-2", "NCT999", tmp_path),
        ],
    )
    with pytest.raises(SynthesisAuthorizationError, match="identifier_outside_cited_span"):
        authorize_synthesis_unit(
            corpus=corpus,
            reconciliation=reconciliation,
            estimate_ids=sorted(
                row.estimate_id for row in reconciliation.reconciled_graph.outcome_estimates
            ),
            assertions=[assertion],
            repository_root=tmp_path,
        )
