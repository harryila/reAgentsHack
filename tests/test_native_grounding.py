from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from literature_multiverse.evidence_graph import PublicationIdentity
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.native_extraction import (
    NativePublicationExtraction,
    NativeSourceManifest,
    NativeSourceRecord,
    freeze_native_publication_extraction,
)
from literature_multiverse.native_grounding import (
    NativeGroundingError,
    NativeGroundingReceipt,
    TypedEvidenceGroundingPackage,
    freeze_grounding_checked_publication_fragment,
    freeze_typed_evidence_grounding_package,
    resolve_native_source_document,
    reverify_typed_evidence_grounding_package,
    validate_typed_corpus_grounding,
    verify_native_publication_grounding,
)
from literature_multiverse.typed_extraction import (
    FragmentStatus,
    SourceDocumentArtifact,
    TypedEvidenceCorpus,
    assemble_typed_evidence_corpus,
    freeze_publication_evidence_fragment,
)

PIPELINE_HASH = "d" * 64


def _payload(
    *,
    source_locator: str,
    quote: str = "The mean difference was 2.5 (SE 0.5).",
    line_ids: list[str] | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
    section: str | None = "Results",
) -> dict[str, object]:
    return {
        "extraction_schema_version": "native-publication-extraction-v1",
        "status": "estimable",
        "studies": [
            {
                "key": "trial",
                "source_label": "Trial",
                "design": "parallel trial",
                "registration_ids": [],
                "cohorts": [
                    {
                        "key": "cohort",
                        "source_labels": ["reported cohort"],
                        "registry_ids": [],
                        "dataset_ids": [],
                        "population_description": None,
                        "recruitment_period": None,
                        "total_sample_size": 20,
                        "arms": [
                            {
                                "key": "treatment",
                                "label": "Treatment",
                                "role": "intervention",
                                "description": None,
                                "sample_size": 10,
                            },
                            {
                                "key": "control",
                                "label": "Control",
                                "role": "control",
                                "description": None,
                                "sample_size": 10,
                            },
                        ],
                        "contrasts": [
                            {
                                "key": "primary",
                                "treatment_arm_key": "treatment",
                                "comparator_arm_key": "control",
                                "label": "treatment_vs_control",
                                "estimand": None,
                                "positive_direction_means": "higher under treatment",
                            }
                        ],
                        "findings": [
                            {
                                "key": "result",
                                "contrast_key": "primary",
                                "outcome_name": "score",
                                "timepoint": {
                                    "kind": "exact",
                                    "value": 4,
                                    "unit": "week",
                                },
                                "analysis_population": None,
                                "effect": {
                                    "effect_format": "mean_difference",
                                    "availability": "available",
                                    "estimate": 2.5,
                                    "standard_error": 0.5,
                                    "unit": "points",
                                    "reported_significance": "not_reported",
                                    "equivalence_conclusion": "not_tested",
                                    "moderators": [],
                                    "extraction_method": "reported",
                                },
                                "evidence": {
                                    "source_locator": source_locator,
                                    "quote": quote,
                                    "section": section,
                                    "page": None,
                                    "char_start": char_start,
                                    "char_end": char_end,
                                    "line_ids": ["L20"] if line_ids is None else line_ids,
                                },
                            }
                        ],
                    }
                ],
            }
        ],
        "non_estimability_reason": None,
        "non_estimability_detail": None,
        "warnings": [],
    }


def _write_antiox_source(root: Path) -> tuple[Path, str]:
    path = root / "archive" / "source_lines.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "PMC/1": {
                    "L1": {"section": "Abstract", "text": "An abstract result."},
                    "L20": {
                        "section": "Results",
                        "text": "The mean difference was 2.5 (SE 0.5).",
                    },
                    "L21": {
                        "section": "Results",
                        "text": "A secondary exact result was 9.0 (SE 1.0).",
                    },
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path, "json:archive/source_lines.json#/PMC~11"


def _source(path: Path, locator: str) -> SourceDocumentArtifact:
    return SourceDocumentArtifact(
        artifact_path=path.relative_to(path.parents[1]).as_posix(),
        sha256=sha256_file(path),
        media_type="application/json",
        source_locator=locator,
    )


def _publication() -> PublicationIdentity:
    return PublicationIdentity(
        publication_id="publication-grounded",
        paper_id="paper-grounded",
        doc_id="PMC/1",
    )


def test_antiox_exact_lines_authorize_estimable_fragment(tmp_path: Path) -> None:
    path, locator = _write_antiox_source(tmp_path)
    source = _source(path, locator)
    extraction = NativePublicationExtraction.model_validate(_payload(source_locator=locator))

    resolved = resolve_native_source_document(
        repository_root=tmp_path,
        source_document=source,
    )
    receipt = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=extraction,
    )

    assert resolved.source_kind == "antiox_json_lines"
    assert [line.line_id for line in resolved.lines] == ["L1", "L20", "L21"]
    assert resolved.source_payload_sha256 == receipt.source_payload_sha256
    assert receipt.source_verified is True
    assert receipt.all_findings_exact is True
    assert receipt.authorizes_estimable_fragment is True
    assert receipt.finding_results[0].status == "exact"
    assert receipt.finding_results[0].resolved_line_numbers == [20]
    assert receipt.issues == []


def test_resolved_source_preserves_whitespace_crlf_and_non_ascii_offsets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "archive" / "source_lines.json"
    path.parent.mkdir(parents=True)
    source_payload = {
        "L1": {
            "section": "Methods",
            "text": "\t  \\usepackage{amsmath}\r\n",
        },
        "L2": {
            "section": "Results",
            "text": "  Mean difference was 2.5 ± 0.5 μg.  ",
        },
    }
    path.write_text(
        json.dumps({"PMC/1": source_payload}, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    locator = "json:archive/source_lines.json#/PMC~11"
    source = _source(path, locator)

    resolved = resolve_native_source_document(
        repository_root=tmp_path,
        source_document=source,
    )

    assert [line.text for line in resolved.lines] == [
        "\t  \\usepackage{amsmath}\r\n",
        "  Mean difference was 2.5 ± 0.5 μg.  ",
    ]
    assert resolved.source_text == "\n".join(line.text for line in resolved.lines)
    assert resolved.source_payload_sha256 == hash_canonical(source_payload)
    encoded_source = resolved.source_text.encode("utf-8")
    for line in resolved.lines:
        assert resolved.source_text[line.char_start : line.char_end] == line.text
        encoded_line = encoded_source[line.utf8_byte_start : line.utf8_byte_end]
        assert encoded_line.decode("utf-8") == line.text
    assert resolved.lines[1].char_end != resolved.lines[1].utf8_byte_end


def test_quote_mismatch_and_relocation_both_fail_closed(tmp_path: Path) -> None:
    path, locator = _write_antiox_source(tmp_path)
    source = _source(path, locator)
    mismatch = NativePublicationExtraction.model_validate(
        _payload(source_locator=locator, quote="The mean difference was 8.5 (SE 0.5).")
    )
    relocated = NativePublicationExtraction.model_validate(
        _payload(
            source_locator=locator,
            quote="A secondary exact result was 9.0 (SE 1.0).",
            line_ids=["L20"],
        )
    )

    mismatch_receipt = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=mismatch,
    )
    relocated_receipt = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=relocated,
    )

    assert mismatch_receipt.finding_results[0].status == "mismatch"
    assert mismatch_receipt.authorizes_estimable_fragment is False
    assert relocated_receipt.finding_results[0].status == "mismatch"
    assert "evidence_line_relocation_forbidden" in relocated_receipt.finding_results[0].issues


def test_native_exact_grounding_rejects_whitespace_normalization_and_ellipsis(
    tmp_path: Path,
) -> None:
    path, locator = _write_antiox_source(tmp_path)
    source = _source(path, locator)
    normalized_only = NativePublicationExtraction.model_validate(
        _payload(
            source_locator=locator,
            quote="The  mean difference was 2.5 (SE 0.5).",
        )
    )
    ellipsis_splice = NativePublicationExtraction.model_validate(
        _payload(
            source_locator=locator,
            quote="The mean difference ... 2.5 (SE 0.5).",
        )
    )

    for extraction in (normalized_only, ellipsis_splice):
        receipt = verify_native_publication_grounding(
            repository_root=tmp_path,
            source_document=source,
            extraction=extraction,
        )
        result = receipt.finding_results[0]
        assert result.status == "mismatch"
        assert "evidence_quote_not_exact_source_substring" in result.issues
        assert receipt.authorizes_estimable_fragment is False


def test_offsets_are_checked_against_canonical_resolved_text(tmp_path: Path) -> None:
    path, locator = _write_antiox_source(tmp_path)
    source = _source(path, locator)
    resolved = resolve_native_source_document(repository_root=tmp_path, source_document=source)
    result_line = next(line for line in resolved.lines if line.line_id == "L20")
    exact = NativePublicationExtraction.model_validate(
        _payload(
            source_locator=locator,
            line_ids=[],
            char_start=result_line.char_start,
            char_end=result_line.char_end,
        )
    )
    wrong = NativePublicationExtraction.model_validate(
        _payload(source_locator=locator, line_ids=[], char_start=0, char_end=10)
    )

    exact_receipt = verify_native_publication_grounding(
        repository_root=tmp_path, source_document=source, extraction=exact
    )
    wrong_receipt = verify_native_publication_grounding(
        repository_root=tmp_path, source_document=source, extraction=wrong
    )

    assert exact_receipt.finding_results[0].offset_check == "exact"
    assert exact_receipt.authorizes_estimable_fragment is True
    assert wrong_receipt.finding_results[0].offset_check == "mismatch"
    assert wrong_receipt.authorizes_estimable_fragment is False


def test_missing_coordinates_and_wrong_finding_locator_are_not_exact(tmp_path: Path) -> None:
    path, locator = _write_antiox_source(tmp_path)
    source = _source(path, locator)
    missing_coordinates = NativePublicationExtraction.model_validate(
        _payload(source_locator=locator, line_ids=[])
    )
    wrong_locator = NativePublicationExtraction.model_validate(
        _payload(source_locator="json:archive/source_lines.json#/OTHER")
    )

    missing_receipt = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=missing_coordinates,
    )
    wrong_receipt = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=wrong_locator,
    )

    assert missing_receipt.finding_results[0].status == "unverifiable"
    assert "evidence_coordinates_missing" in missing_receipt.finding_results[0].issues
    assert wrong_receipt.finding_results[0].status == "mismatch"
    assert "evidence_source_locator_mismatch" in wrong_receipt.finding_results[0].issues


def test_hash_drift_and_symlink_never_relocate_or_authorize(tmp_path: Path) -> None:
    path, locator = _write_antiox_source(tmp_path)
    source = _source(path, locator)
    extraction = NativePublicationExtraction.model_validate(_payload(source_locator=locator))
    path.write_text("{}", encoding="utf-8")

    drift = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=extraction,
    )

    assert drift.source_verified is False
    assert drift.observed_source_sha256 is not None
    assert drift.issues == ["source_artifact_hash_mismatch"]
    assert drift.finding_results == []
    assert drift.authorizes_estimable_fragment is False

    real = tmp_path / "real.json"
    real.write_text("{}", encoding="utf-8")
    link = tmp_path / "linked.json"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("symlinks unavailable")
    linked_source = SourceDocumentArtifact(
        artifact_path="linked.json",
        sha256=sha256_file(real),
        media_type="application/json",
        source_locator="json:linked.json#/PMC1",
    )
    linked = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=linked_source,
        extraction=extraction,
    )
    assert linked.issues == ["source_artifact_symlink_forbidden"]
    assert linked.authorizes_estimable_fragment is False


def _write_metasyn_source(root: Path) -> tuple[Path, str]:
    path = root / "corpus" / "shard.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "ID": 7,
                    "title": "Trial title",
                    "abstract": "The abstract mentions no numeric finding.",
                    "sections": [
                        {
                            "heading": "Results",
                            "text": "The mean difference was 2.5 (SE 0.5).",
                        }
                    ],
                },
                {
                    "ID": 8,
                    "title": "Other title",
                    "abstract": None,
                    "sections": [],
                },
            ]
        ),
        path,
        row_group_size=1,
    )
    locator = "parquet:corpus/shard.parquet#row_group=0&row_in_group=0&index_base=0&ID=7"
    return path, locator


def test_metasyn_physical_row_and_ID_resolve_conservatively(tmp_path: Path) -> None:
    path, locator = _write_metasyn_source(tmp_path)
    source = SourceDocumentArtifact(
        artifact_path="corpus/shard.parquet",
        sha256=sha256_file(path),
        media_type="application/vnd.apache.parquet",
        source_locator=locator,
    )
    extraction = NativePublicationExtraction.model_validate(
        _payload(source_locator=locator, line_ids=["L3"])
    )

    resolved = resolve_native_source_document(repository_root=tmp_path, source_document=source)
    receipt = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=extraction,
    )

    assert resolved.source_kind == "metasyn_parquet_row"
    assert [(line.line_id, line.section) for line in resolved.lines] == [
        ("L1", "Title"),
        ("L2", "Abstract"),
        ("L3", "Results"),
    ]
    assert receipt.authorizes_estimable_fragment is True

    wrong_id = source.model_copy(
        update={
            "source_locator": locator.replace("ID=7", "ID=8"),
        }
    )
    failed = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=wrong_id,
        extraction=extraction,
    )
    assert failed.issues == ["parquet_source_row_ID_mismatch"]
    assert failed.authorizes_estimable_fragment is False


def test_parquet_duplicate_keys_and_abstract_evidence_fail_closed(tmp_path: Path) -> None:
    path, locator = _write_metasyn_source(tmp_path)
    duplicate_locator = locator + "&ID=7"
    duplicate = SourceDocumentArtifact(
        artifact_path="corpus/shard.parquet",
        sha256=sha256_file(path),
        media_type="application/vnd.apache.parquet",
        source_locator=duplicate_locator,
    )
    extraction = NativePublicationExtraction.model_validate(
        _payload(source_locator=duplicate_locator, line_ids=["L3"])
    )
    failed = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=duplicate,
        extraction=extraction,
    )
    assert failed.issues == ["parquet_source_locator_keys_invalid"]

    source = SourceDocumentArtifact(
        artifact_path="corpus/shard.parquet",
        sha256=sha256_file(path),
        media_type="application/vnd.apache.parquet",
        source_locator=locator,
    )
    abstract = NativePublicationExtraction.model_validate(
        _payload(
            source_locator=locator,
            quote="The abstract mentions no numeric finding.",
            line_ids=["L2"],
            section="Abstract",
        )
    )
    abstract_receipt = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=abstract,
    )
    assert abstract_receipt.finding_results[0].status == "mismatch"
    assert "evidence_section_forbidden_or_unknown" in abstract_receipt.finding_results[0].issues
    assert abstract_receipt.authorizes_estimable_fragment is False


def test_non_estimable_extraction_and_receipt_tampering_do_not_authorize(tmp_path: Path) -> None:
    path, locator = _write_antiox_source(tmp_path)
    source = _source(path, locator)
    extraction = NativePublicationExtraction(
        status="non_estimable",
        studies=[],
        non_estimability_reason="numerical_result_absent",
    )

    receipt = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=extraction,
    )

    assert receipt.source_verified is True
    assert receipt.finding_results == []
    assert receipt.all_findings_exact is False
    assert receipt.authorizes_estimable_fragment is False
    tampered = receipt.model_dump(mode="json")
    tampered["authorizes_estimable_fragment"] = True
    with pytest.raises(ValidationError, match="native_grounding_authorization_mismatch"):
        NativeGroundingReceipt.model_validate(tampered)


def test_source_document_path_contract_forbids_parent_escape() -> None:
    payload = {
        "artifact_path": "../outside.json",
        "sha256": "a" * 64,
        "media_type": "application/json",
        "source_locator": "json:../outside.json#/doc",
    }
    with pytest.raises(ValidationError, match="source_document_artifact_path_must_be_relative"):
        SourceDocumentArtifact.model_validate(payload)


def test_finding_result_hash_is_bound(tmp_path: Path) -> None:
    path, locator = _write_antiox_source(tmp_path)
    source = _source(path, locator)
    extraction_payload = deepcopy(_payload(source_locator=locator))
    extraction = NativePublicationExtraction.model_validate(extraction_payload)
    receipt = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=extraction,
    )
    tampered = receipt.model_dump(mode="json")
    tampered["finding_results"][0]["issues"] = ["invented"]
    with pytest.raises(ValidationError, match="native_finding_grounding_hash_mismatch"):
        NativeGroundingReceipt.model_validate(tampered)


def test_corpus_join_requires_one_actual_authorizing_receipt_per_estimable_fragment(
    tmp_path: Path,
) -> None:
    path, locator = _write_antiox_source(tmp_path)
    source = _source(path, locator)
    extraction = NativePublicationExtraction.model_validate(_payload(source_locator=locator))
    receipt = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=extraction,
    )
    fragment = freeze_native_publication_extraction(
        payload=extraction,
        question_id="grounding-question",
        publication=_publication(),
        pipeline_fingerprint_sha256=PIPELINE_HASH,
        source_document=source,
        grounding_receipt_sha256=receipt.receipt_sha256,
    )
    corpus = assemble_typed_evidence_corpus([fragment])

    validation = validate_typed_corpus_grounding(
        corpus=corpus,
        grounding_receipts=[receipt],
    )
    package = freeze_typed_evidence_grounding_package(
        corpus=corpus,
        grounding_receipts=[receipt],
    )

    assert validation.estimable_authorized_receipts == 1
    assert validation.expected_non_estimable_extraction_receipts == 0
    assert validation.failed_estimable_grounding_receipts == 0
    assert package.grounding_validation.validation_sha256 == validation.validation_sha256
    assert package.package_version == "typed-evidence-grounding-package-v2"
    assert package.cohort_reconciliation is not None
    replay = reverify_typed_evidence_grounding_package(
        package=package,
        repository_root=tmp_path,
    )
    assert (
        replay.cohort_reconciliation_receipt_sha256 == package.cohort_reconciliation.receipt_sha256
    )
    legacy_payload = package.model_dump(mode="json", exclude={"package_sha256"})
    legacy_payload["package_version"] = "typed-evidence-grounding-package-v1"
    legacy_payload.pop("cohort_reconciliation")
    legacy_payload.pop("source_manifest")
    legacy_payload.pop("source_manifest_sha256")
    legacy_payload.pop("corpus_cutoff")
    legacy_package = TypedEvidenceGroundingPackage.model_validate(
        {**legacy_payload, "package_sha256": hash_canonical(legacy_payload)}
    )
    assert legacy_package.cohort_reconciliation is None
    with pytest.raises(NativeGroundingError, match="receipt_not_found"):
        validate_typed_corpus_grounding(corpus=corpus, grounding_receipts=[])
    with pytest.raises(NativeGroundingError, match="receipts_duplicate"):
        validate_typed_corpus_grounding(
            corpus=corpus,
            grounding_receipts=[receipt, receipt],
        )


def test_package_rejects_fully_rehashed_reconciliation_graph_tampering(
    tmp_path: Path,
) -> None:
    path, locator = _write_antiox_source(tmp_path)
    source = _source(path, locator)
    extraction = NativePublicationExtraction.model_validate(_payload(source_locator=locator))
    receipt = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=extraction,
    )
    fragment = freeze_native_publication_extraction(
        payload=extraction,
        question_id="grounding-question",
        publication=_publication(),
        pipeline_fingerprint_sha256=PIPELINE_HASH,
        source_document=source,
        grounding_receipt_sha256=receipt.receipt_sha256,
    )
    corpus = assemble_typed_evidence_corpus([fragment])
    package = freeze_typed_evidence_grounding_package(
        corpus=corpus,
        grounding_receipts=[receipt],
    )
    payload = package.model_dump(mode="json", exclude={"package_sha256"})
    reconciliation = payload["cohort_reconciliation"]
    reconciliation.pop("receipt_sha256")
    reconciliation["reconciled_graph"]["outcome_estimates"][0]["effect"]["estimate"] = 9.9
    reconciliation["reconciled_graph_sha256"] = hash_canonical(reconciliation["reconciled_graph"])
    reconciliation["receipt_sha256"] = hash_canonical(reconciliation)

    with pytest.raises(
        ValidationError,
        match="typed_evidence_grounding_package_reconciliation_invalid",
    ):
        TypedEvidenceGroundingPackage.model_validate(
            {**payload, "package_sha256": hash_canonical(payload)}
        )


def test_v3_package_binds_complete_source_manifest_membership_and_cutoff(
    tmp_path: Path,
) -> None:
    path, locator = _write_antiox_source(tmp_path)
    source = _source(path, locator)
    publication = _publication()
    extraction = NativePublicationExtraction.model_validate(_payload(source_locator=locator))
    receipt = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=extraction,
    )
    fragment = freeze_native_publication_extraction(
        payload=extraction,
        question_id="grounding-question",
        publication=publication,
        pipeline_fingerprint_sha256=PIPELINE_HASH,
        source_document=source,
        grounding_receipt_sha256=receipt.receipt_sha256,
    )
    corpus = assemble_typed_evidence_corpus([fragment])
    manifest = NativeSourceManifest(
        question_id="grounding-question",
        records=[
            NativeSourceRecord(
                doc_id="PMC/1",
                publication=publication,
                source_document=source,
            )
        ],
    )

    package = freeze_typed_evidence_grounding_package(
        corpus=corpus,
        grounding_receipts=[receipt],
        source_manifest=manifest,
        corpus_cutoff="closed-corpus-2026-08-27",
    )

    assert package.package_version == "typed-evidence-grounding-package-v3"
    assert package.source_manifest_sha256 == hash_canonical(manifest)
    assert package.corpus_cutoff == "closed-corpus-2026-08-27"
    replay = reverify_typed_evidence_grounding_package(
        package=package,
        repository_root=tmp_path,
    )
    assert replay.source_manifest_sha256 == package.source_manifest_sha256
    assert replay.source_manifest_records == 1

    omitted_corpus = assemble_typed_evidence_corpus(
        [
            freeze_publication_evidence_fragment(
                question_id="grounding-question",
                publication_id="publication-omitted",
                paper_id="paper-omitted",
                publication=PublicationIdentity(
                    publication_id="publication-omitted",
                    paper_id="paper-omitted",
                ),
                pipeline_fingerprint_sha256=PIPELINE_HASH,
                source_document=source,
                grounding_receipt_sha256=None,
                status=FragmentStatus.NON_ESTIMABLE,
                non_estimability_reason="no_target_outcome",
            )
        ]
    )
    with pytest.raises(
        ValidationError,
        match="source_membership_mismatch",
    ):
        freeze_typed_evidence_grounding_package(
            corpus=omitted_corpus,
            grounding_receipts=[],
            source_manifest=manifest,
            corpus_cutoff="closed-corpus-2026-08-27",
        )


def test_corpus_rejects_fully_rehashed_graph_not_projected_by_fragments(
    tmp_path: Path,
) -> None:
    path, locator = _write_antiox_source(tmp_path)
    source = _source(path, locator)
    extraction = NativePublicationExtraction.model_validate(_payload(source_locator=locator))
    receipt = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=extraction,
    )
    fragment = freeze_native_publication_extraction(
        payload=extraction,
        question_id="grounding-question",
        publication=_publication(),
        pipeline_fingerprint_sha256=PIPELINE_HASH,
        source_document=source,
        grounding_receipt_sha256=receipt.receipt_sha256,
    )
    corpus = assemble_typed_evidence_corpus([fragment])
    tampered_payload = corpus.model_dump(mode="json", exclude={"corpus_sha256"})
    tampered_payload["graph"]["outcome_estimates"][0]["effect"]["reported_p_value"] = 0.99
    with pytest.raises(
        ValidationError,
        match="typed_evidence_graph_fragment_projection_mismatch",
    ):
        TypedEvidenceCorpus.model_validate(
            {**tampered_payload, "corpus_sha256": hash_canonical(tampered_payload)}
        )


def test_corpus_join_separates_expected_non_estimable_from_failed_grounding(
    tmp_path: Path,
) -> None:
    path, locator = _write_antiox_source(tmp_path)
    source = _source(path, locator)
    expected_extraction = NativePublicationExtraction(
        status="non_estimable",
        studies=[],
        non_estimability_reason="numerical_result_absent",
    )
    expected_receipt = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=expected_extraction,
    )
    expected_fragment = freeze_native_publication_extraction(
        payload=expected_extraction,
        question_id="grounding-question",
        publication=_publication(),
        pipeline_fingerprint_sha256=PIPELINE_HASH,
        source_document=source,
        grounding_receipt_sha256=expected_receipt.receipt_sha256,
    )
    expected_corpus = assemble_typed_evidence_corpus([expected_fragment])
    expected_validation = validate_typed_corpus_grounding(
        corpus=expected_corpus,
        grounding_receipts=[expected_receipt],
    )
    assert expected_validation.expected_non_estimable_extraction_receipts == 1
    assert expected_validation.failed_estimable_grounding_receipts == 0

    failed_extraction = NativePublicationExtraction.model_validate(
        _payload(source_locator=locator, quote="Unsupported numerical quote.")
    )
    failed_receipt = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=failed_extraction,
    )
    failed_fragment = freeze_grounding_checked_publication_fragment(
        extraction=failed_extraction,
        grounding_receipt=failed_receipt,
        question_id="grounding-question",
        publication=_publication(),
        pipeline_fingerprint_sha256=PIPELINE_HASH,
        source_document=source,
    )
    failed_corpus = assemble_typed_evidence_corpus([failed_fragment])
    failed_validation = validate_typed_corpus_grounding(
        corpus=failed_corpus,
        grounding_receipts=[failed_receipt],
    )
    assert failed_validation.expected_non_estimable_extraction_receipts == 0
    assert failed_validation.failed_estimable_grounding_receipts == 1


def test_receipt_linked_non_estimable_fragment_is_a_deterministic_projection(
    tmp_path: Path,
) -> None:
    path, locator = _write_antiox_source(tmp_path)
    source = _source(path, locator)
    extraction = NativePublicationExtraction(
        status="non_estimable",
        studies=[],
        non_estimability_reason="numerical_result_absent",
    )
    receipt = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=extraction,
    )
    fragment = freeze_native_publication_extraction(
        payload=extraction,
        question_id="grounding-question",
        publication=_publication(),
        pipeline_fingerprint_sha256=PIPELINE_HASH,
        source_document=source,
        grounding_receipt_sha256=receipt.receipt_sha256,
    )
    tampered_payload = fragment.model_dump(mode="json", exclude={"fragment_sha256"})
    tampered_payload["non_estimability_reason"] = "no_target_outcome"
    tampered = type(fragment).model_validate(
        {**tampered_payload, "fragment_sha256": hash_canonical(tampered_payload)}
    )
    tampered_corpus = assemble_typed_evidence_corpus([tampered])

    with pytest.raises(
        NativeGroundingError,
        match="receipt_linked_fragment_projection_mismatch",
    ):
        freeze_typed_evidence_grounding_package(
            corpus=tampered_corpus,
            grounding_receipts=[receipt],
        )


def test_corpus_join_rejects_unreferenced_receipt(tmp_path: Path) -> None:
    path, locator = _write_antiox_source(tmp_path)
    source = _source(path, locator)
    extraction = NativePublicationExtraction.model_validate(_payload(source_locator=locator))
    receipt = verify_native_publication_grounding(
        repository_root=tmp_path,
        source_document=source,
        extraction=extraction,
    )
    fragment = freeze_publication_evidence_fragment(
        question_id="grounding-question",
        publication_id="publication-unlinked",
        paper_id="paper-unlinked",
        publication=PublicationIdentity(
            publication_id="publication-unlinked",
            paper_id="paper-unlinked",
        ),
        pipeline_fingerprint_sha256=PIPELINE_HASH,
        source_document=source,
        grounding_receipt_sha256=None,
        status=FragmentStatus.NON_ESTIMABLE,
        non_estimability_reason="no_target_outcome",
    )
    corpus = assemble_typed_evidence_corpus([fragment])

    with pytest.raises(NativeGroundingError, match="receipts_unreferenced"):
        validate_typed_corpus_grounding(
            corpus=corpus,
            grounding_receipts=[receipt],
        )
