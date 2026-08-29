from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from scripts.reconcile_native_cohorts import main as reconcile_cli_main

from literature_multiverse.cohort_reconciliation import (
    NativeCohortReconciliationError,
    NativeCohortReconciliationReceipt,
    ReviewerIdentityGroup,
    freeze_reviewer_cohort_reconciliation_artifact,
    reconcile_native_cohorts,
    reverify_native_cohort_reconciliation,
)
from literature_multiverse.evidence_graph import PublicationIdentity
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.meta_analysis import synthesize_evidence_graph
from literature_multiverse.native_extraction import (
    NativePublicationExtraction,
    freeze_native_publication_extraction,
)
from literature_multiverse.native_grounding import (
    freeze_typed_evidence_grounding_package,
    verify_native_publication_grounding,
)
from literature_multiverse.typed_extraction import (
    SourceDocumentArtifact,
    assemble_typed_evidence_corpus,
    freeze_publication_evidence_fragment,
)
from literature_multiverse.verifier import LegacyAdapterConfig, load_corpus

PIPELINE_HASH = "a" * 64
GROUNDING_HASH = "b" * 64


def _cohort_payload(
    key: str,
    *,
    registry_ids: list[str],
    dataset_ids: list[str],
    estimate: float,
) -> dict[str, object]:
    return {
        "key": key,
        "source_labels": [f"Reported cohort {key}"],
        "registry_ids": registry_ids,
        "dataset_ids": dataset_ids,
        "total_sample_size": 100,
        "arms": [
            {"key": "tx", "label": "Treatment", "role": "intervention"},
            {"key": "control", "label": "Control", "role": "control"},
        ],
        "contrasts": [
            {
                "key": "primary",
                "treatment_arm_key": "tx",
                "comparator_arm_key": "control",
                "label": "treatment_vs_control",
                "positive_direction_means": "higher values favor treatment",
            }
        ],
        "findings": [
            {
                "key": "finding",
                "contrast_key": "primary",
                "outcome_name": "outcome",
                "timepoint": {"kind": "exact", "value": 4, "unit": "week"},
                "effect": {
                    "effect_format": "hedges_g",
                    "estimate": estimate,
                    "standard_error": 0.1,
                },
                "evidence": {
                    "source_locator": f"source:{key}",
                    "quote": f"Effect for {key} was {estimate}.",
                    "line_ids": ["L1"],
                },
            }
        ],
    }


def _fragment(
    publication_number: int,
    *,
    cohorts: list[dict[str, object]],
    study_registration_ids: list[str] | None = None,
):
    publication_id = f"publication-{publication_number}"
    extraction = NativePublicationExtraction.model_validate(
        {
            "status": "estimable",
            "studies": [
                {
                    "key": "study",
                    "source_label": f"Study report {publication_number}",
                    "registration_ids": study_registration_ids or [],
                    "cohorts": cohorts,
                }
            ],
        }
    )
    publication = PublicationIdentity(
        publication_id=publication_id,
        paper_id=f"paper-{publication_number}",
        doc_id=f"doc-{publication_number}",
    )
    source = SourceDocumentArtifact(
        artifact_path=f"data/source-{publication_number}.json",
        sha256=f"{publication_number:x}" * 64,
        media_type="application/json",
        source_locator=f"json:data/source-{publication_number}.json#/doc-{publication_number}",
    )
    return freeze_native_publication_extraction(
        payload=extraction,
        question_id="reconciliation-question",
        publication=publication,
        pipeline_fingerprint_sha256=PIPELINE_HASH,
        source_document=source,
        grounding_receipt_sha256=GROUNDING_HASH,
    )


def _two_publication_corpus(
    *,
    first_registry: list[str],
    second_registry: list[str],
    first_dataset: list[str] | None = None,
    second_dataset: list[str] | None = None,
):
    return assemble_typed_evidence_corpus(
        [
            _fragment(
                1,
                cohorts=[
                    _cohort_payload(
                        "cohort-a",
                        registry_ids=first_registry,
                        dataset_ids=first_dataset or [],
                        estimate=0.2,
                    )
                ],
                study_registration_ids=first_registry,
            ),
            _fragment(
                2,
                cohorts=[
                    _cohort_payload(
                        "cohort-b",
                        registry_ids=second_registry,
                        dataset_ids=second_dataset or [],
                        estimate=0.3,
                    )
                ],
                study_registration_ids=second_registry,
            ),
        ]
    )


def test_exact_normalized_registry_id_merges_cross_publication_cohort_once() -> None:
    corpus = _two_publication_corpus(
        first_registry=["NCT 00000001"],
        second_registry=["nct 00000001"],
    )

    receipt = reconcile_native_cohorts(corpus=corpus)

    assert receipt.status == "strong_identifier_reconciled_limited"
    assert receipt.cross_publication_identity_assurance_complete is False
    assert receipt.merged_study_groups == 1
    assert receipt.merged_cohort_groups == 1
    assert receipt.reconciled_graph is not None
    assert len(receipt.reconciled_graph.studies) == 1
    assert len(receipt.reconciled_graph.cohorts) == 1
    assert len(receipt.reconciled_graph.outcome_estimates) == 2
    assert len(receipt.reconciled_graph.evidence_spans) == 2
    cohort_id = receipt.reconciled_graph.cohorts[0].cohort_id
    assert {arm.cohort_id for arm in receipt.reconciled_graph.arms} == {cohort_id}
    assert {contrast.cohort_id for contrast in receipt.reconciled_graph.contrasts} == {cohort_id}
    assert {
        estimate.effect.paper_id for estimate in receipt.reconciled_graph.outcome_estimates
    } == {"paper-1", "paper-2"}
    synthesis = synthesize_evidence_graph(
        receipt.reconciled_graph,
        outcome_name="outcome",
    )
    assert synthesis["evidence_graph"]["selected_cohort_count"] == 1
    assert synthesis["quantitative"]["n_cohorts"] == 1
    assert synthesis["quantitative"]["reason"] == "fewer_than_two_cohorts"


def test_distinct_cohorts_in_one_publication_remain_distinct() -> None:
    corpus = assemble_typed_evidence_corpus(
        [
            _fragment(
                1,
                cohorts=[
                    _cohort_payload(
                        "cohort-a",
                        registry_ids=["NCT-SAME-STUDY"],
                        dataset_ids=[],
                        estimate=0.2,
                    ),
                    _cohort_payload(
                        "cohort-b",
                        registry_ids=["NCT-SAME-STUDY"],
                        dataset_ids=[],
                        estimate=-0.1,
                    ),
                ],
                study_registration_ids=["MASTER-STUDY"],
            )
        ]
    )

    receipt = reconcile_native_cohorts(corpus=corpus)

    assert receipt.status == "single_publication_complete"
    assert receipt.merged_cohort_groups == 0
    assert receipt.reconciled_graph is not None
    assert len(receipt.reconciled_graph.cohorts) == 2
    assert len({contrast.cohort_id for contrast in receipt.reconciled_graph.contrasts}) == 2


def test_conflicting_registry_ids_linked_by_dataset_fail_closed() -> None:
    corpus = _two_publication_corpus(
        first_registry=["NCT-ONE"],
        second_registry=["NCT-TWO"],
        first_dataset=["DATASET-SHARED"],
        second_dataset=["dataset-shared"],
    )

    receipt = reconcile_native_cohorts(corpus=corpus)

    assert receipt.status == "requires_reviewer"
    assert receipt.merged_cohort_groups == 0
    assert "conflicting_registry_identifiers" in {issue.code for issue in receipt.issues}
    assert receipt.reconciled_graph is not None
    assert len(receipt.reconciled_graph.cohorts) == 2


def test_many_to_many_identifier_overlap_fails_closed() -> None:
    first = _fragment(
        1,
        cohorts=[
            _cohort_payload(
                "cohort-a",
                registry_ids=["NCT-SHARED"],
                dataset_ids=[],
                estimate=0.1,
            ),
            _cohort_payload(
                "cohort-b",
                registry_ids=["NCT-SHARED"],
                dataset_ids=[],
                estimate=0.2,
            ),
        ],
        study_registration_ids=["NCT-SHARED"],
    )
    second = _fragment(
        2,
        cohorts=[
            _cohort_payload(
                "cohort-c",
                registry_ids=["nct-shared"],
                dataset_ids=[],
                estimate=0.3,
            )
        ],
        study_registration_ids=["nct-shared"],
    )
    corpus = assemble_typed_evidence_corpus([first, second])

    receipt = reconcile_native_cohorts(corpus=corpus)

    assert receipt.status == "requires_reviewer"
    assert receipt.merged_cohort_groups == 0
    assert "ambiguous_many_to_many_identity" in {issue.code for issue in receipt.issues}
    assert receipt.reconciled_graph is not None
    assert len(receipt.reconciled_graph.cohorts) == 3


def test_complete_reviewer_artifact_can_reconcile_without_strong_ids() -> None:
    corpus = _two_publication_corpus(first_registry=[], second_registry=[])
    assert corpus.graph is not None
    studies = sorted(node.study_id for node in corpus.graph.studies)
    cohorts = sorted(node.cohort_id for node in corpus.graph.cohorts)
    artifact = freeze_reviewer_cohort_reconciliation_artifact(
        corpus=corpus,
        reviewer_identity_sha256="c" * 64,
        review_protocol_sha256="d" * 64,
        completed_at=datetime(2026, 8, 27, tzinfo=UTC),
        study_groups=[
            ReviewerIdentityGroup(
                member_ids=studies,
                rationale="The reviewer confirmed both publications report one study.",
            )
        ],
        cohort_groups=[
            ReviewerIdentityGroup(
                member_ids=cohorts,
                rationale="The reviewer confirmed both estimates use the same participants.",
            )
        ],
    )

    receipt = reconcile_native_cohorts(corpus=corpus, reviewer_artifact=artifact)

    assert receipt.status == "reviewer_complete"
    assert receipt.cross_publication_identity_assurance_complete is True
    assert receipt.merged_cohort_groups == 1
    assert receipt.reconciled_graph is not None
    assert len(receipt.reconciled_graph.cohorts) == 1


def test_fully_rehashed_reconciliation_tampering_is_rejected_by_replay() -> None:
    corpus = _two_publication_corpus(
        first_registry=["NCT-SHARED"],
        second_registry=["nct-shared"],
    )
    receipt = reconcile_native_cohorts(corpus=corpus)
    tampered = receipt.model_dump(mode="json", exclude={"receipt_sha256"})
    tampered_graph = deepcopy(tampered["reconciled_graph"])
    tampered_graph["outcome_estimates"][0]["effect"]["estimate"] = 9.9
    tampered["reconciled_graph"] = tampered_graph
    tampered["reconciled_graph_sha256"] = hash_canonical(tampered_graph)
    tampered_receipt = NativeCohortReconciliationReceipt.model_validate(
        {**tampered, "receipt_sha256": hash_canonical(tampered)}
    )

    with pytest.raises(
        NativeCohortReconciliationError,
        match="native_cohort_reconciliation_replay_mismatch",
    ):
        reverify_native_cohort_reconciliation(
            corpus=corpus,
            receipt=tampered_receipt,
        )


def test_reconciliation_cli_writes_worksheet_then_freezes_reviewer_receipt(
    tmp_path: Path,
) -> None:
    corpus = _two_publication_corpus(first_registry=[], second_registry=[])
    corpus_path = tmp_path / "typed-evidence-corpus.json"
    corpus_path.write_text(
        json.dumps(corpus.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    discovery_dir = tmp_path / "discovery"
    assert (
        reconcile_cli_main(
            [
                "--corpus",
                str(corpus_path),
                "--output-dir",
                str(discovery_dir),
            ]
        )
        == 0
    )
    discovery = json.loads((discovery_dir / "cohort_reconciliation_receipt.json").read_text())
    worksheet = json.loads((discovery_dir / "reviewer_partition_template.json").read_text())
    assert discovery["status"] == "strong_identifier_reconciled_limited"
    assert worksheet["all_studies_and_cohorts_reviewed"] is False

    assert corpus.graph is not None
    partition = {
        "partition_version": "reviewer-cohort-partition-input-v1",
        "input_corpus_sha256": corpus.corpus_sha256,
        "input_graph_sha256": hash_canonical(corpus.graph),
        "reviewer_identity_sha256": "c" * 64,
        "review_protocol_sha256": "d" * 64,
        "completed_at": "2026-08-27T12:00:00Z",
        "all_studies_and_cohorts_reviewed": True,
        "study_groups": [
            {
                "member_ids": sorted(node.study_id for node in corpus.graph.studies),
                "rationale": "The reviewer verified one study across both reports.",
            }
        ],
        "cohort_groups": [
            {
                "member_ids": sorted(node.cohort_id for node in corpus.graph.cohorts),
                "rationale": "The reviewer verified a shared participant cohort.",
            }
        ],
    }
    partition_path = tmp_path / "completed-partition.json"
    partition_path.write_text(json.dumps(partition), encoding="utf-8")
    reviewed_dir = tmp_path / "reviewed"
    assert (
        reconcile_cli_main(
            [
                "--corpus",
                str(corpus_path),
                "--reviewer-partition",
                str(partition_path),
                "--output-dir",
                str(reviewed_dir),
            ]
        )
        == 0
    )
    reviewed = json.loads((reviewed_dir / "cohort_reconciliation_receipt.json").read_text())
    assert reviewed["status"] == "reviewer_complete"
    assert reviewed["cross_publication_identity_assurance_complete"] is True
    assert (reviewed_dir / "reviewer_cohort_reconciliation.json").is_file()


def test_public_loader_uses_reconciled_graph_and_blocks_incomplete_identity_review(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "archive" / "sources.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        json.dumps(
            {
                "doc-1": {
                    "L1": {
                        "section": "Results",
                        "text": "Effect for cohort-a was 0.2.",
                    }
                },
                "doc-2": {
                    "L1": {
                        "section": "Results",
                        "text": "Effect for cohort-b was 0.3.",
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    fragments = []
    receipts = []
    for number, key, estimate in ((1, "cohort-a", 0.2), (2, "cohort-b", 0.3)):
        locator = f"json:archive/sources.json#/doc-{number}"
        cohort = _cohort_payload(
            key,
            registry_ids=["NCT-SHARED"],
            dataset_ids=[],
            estimate=estimate,
        )
        cohort["findings"][0]["evidence"]["source_locator"] = locator
        extraction = NativePublicationExtraction.model_validate(
            {
                "status": "estimable",
                "studies": [
                    {
                        "key": "study",
                        "source_label": f"Report {number}",
                        "registration_ids": ["NCT-SHARED"],
                        "cohorts": [cohort],
                    }
                ],
            }
        )
        publication = PublicationIdentity(
            publication_id=f"publication-{number}",
            paper_id=f"paper-{number}",
            doc_id=f"doc-{number}",
        )
        source = SourceDocumentArtifact(
            artifact_path="archive/sources.json",
            sha256=sha256_file(source_path),
            media_type="application/json",
            source_locator=locator,
        )
        receipt = verify_native_publication_grounding(
            repository_root=tmp_path,
            source_document=source,
            extraction=extraction,
        )
        assert receipt.authorizes_estimable_fragment is True
        receipts.append(receipt)
        fragments.append(
            freeze_native_publication_extraction(
                payload=extraction,
                question_id="reconciliation-question",
                publication=publication,
                pipeline_fingerprint_sha256=PIPELINE_HASH,
                source_document=source,
                grounding_receipt_sha256=receipt.receipt_sha256,
            )
        )
    corpus = assemble_typed_evidence_corpus(fragments)
    package = freeze_typed_evidence_grounding_package(
        corpus=corpus,
        grounding_receipts=receipts,
    )
    package_path = tmp_path / "typed_evidence_grounding_package.json"
    package_path.write_text(
        json.dumps(package.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )

    loaded = load_corpus(
        package_path,
        legacy_settings=LegacyAdapterConfig(),
        repository_root=tmp_path,
    )

    assert len(loaded.graph.cohorts) == 1
    assert loaded.metadata["cohort_reconciliation_status"] == (
        "strong_identifier_reconciled_limited"
    )
    assert "cross_publication_cohort_reconciliation_incomplete" in {
        issue.code for issue in loaded.adapter_issues
    }


def test_no_estimable_effects_preserve_publication_graph_and_need_no_identity_merge() -> None:
    publication = PublicationIdentity(
        publication_id="publication-empty",
        paper_id="paper-empty",
        doc_id="doc-empty",
    )
    fragment = freeze_publication_evidence_fragment(
        question_id="reconciliation-question",
        publication_id=publication.publication_id,
        paper_id=publication.paper_id,
        publication=publication,
        pipeline_fingerprint_sha256=PIPELINE_HASH,
        source_document=SourceDocumentArtifact(
            artifact_path="data/unavailable.json",
            sha256="e" * 64,
            media_type="application/json",
            source_locator="json:data/unavailable.json#/doc-empty",
        ),
        grounding_receipt_sha256=None,
        status="non_estimable",
        non_estimability_reason="source_document_incomplete",
    )
    corpus = assemble_typed_evidence_corpus([fragment])

    reconciliation = reconcile_native_cohorts(corpus=corpus)
    package = freeze_typed_evidence_grounding_package(
        corpus=corpus,
        grounding_receipts=[],
    )

    assert corpus.graph.publications == [publication]
    assert corpus.graph.outcome_estimates == []
    assert reconciliation.status == "no_estimable_graph"
    assert reconciliation.cross_publication_identity_assurance_complete is True
    assert reconciliation.input_graph_sha256 == hash_canonical(corpus.graph)
    assert reconciliation.reconciled_graph == corpus.graph
    assert package.cohort_reconciliation == reconciliation
