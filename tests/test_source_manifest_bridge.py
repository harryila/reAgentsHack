from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError
from scripts.build_native_source_manifest import main as bridge_main

from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.native_extraction import NativeSourceManifest
from literature_multiverse.source_manifest_bridge import (
    DiagnosticSourceLedger,
    SourceContentScope,
    SourceManifestBridgeError,
    build_antiox_native_source_bridge,
    build_metasyn_native_source_bridge,
)


def _write_antiox_fixture(root: Path) -> tuple[Path, Path]:
    papers = root / "archive" / "papers.parquet"
    papers.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "paper_id": "doc:PMC1",
                    "doc_id": "PMC1",
                    "doi": "10.1000/one",
                    "pmid": "1001",
                    "title": "Included archived trial",
                    "pub_year": 2020,
                    "screen_status": "included",
                    "map_status": "success",
                    "eligible": True,
                },
                {
                    "paper_id": "doc:PMC2",
                    "doc_id": "PMC2",
                    "doi": None,
                    "pmid": None,
                    "title": "Included paper with unavailable source",
                    "pub_year": 2021,
                    "screen_status": "included",
                    "map_status": "success",
                    "eligible": False,
                },
                {
                    "paper_id": "doc:PMC3",
                    "doc_id": "PMC3",
                    "doi": "not-a-doi",
                    "pmid": "not-a-pmid",
                    "title": "Screen-excluded archived paper",
                    "pub_year": 2022,
                    "screen_status": "excluded",
                    "map_status": "not_mapped",
                    "eligible": False,
                },
            ]
        ),
        papers,
    )
    source_lines = root / "archive" / "source_lines.json"
    source_lines.write_text(
        json.dumps(
            {
                "PMC1": {
                    "L1": {"section": "Abstract", "text": "Archived title."},
                    "L20": {
                        "section": "Results",
                        "text": "A numerical result appears in the archived source.",
                    },
                },
                "PMC3": {"L1": {"section": "Abstract", "text": "Excluded title."}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return papers, source_lines


def _metasyn_row(
    corpus_id: int,
    *,
    title: str | None,
    abstract: str | None,
    sections: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "ID": corpus_id,
        "pmid": str(10000 + corpus_id),
        "title": title,
        "abstract": abstract,
        "authors": ["Archived Author"],
        "journal": "Archived Journal",
        "doi": f"10.1000/{corpus_id}",
        "year": "2020",
        "sections": sections,
        "pmc_id": None,
        "fulltext_source": "fixture_archive",
    }


def _write_metasyn_fixture(
    root: Path,
    *,
    duplicate_id: bool = False,
) -> tuple[Path, list[Path]]:
    corpus = root / "metasyn" / "corpus"
    corpus.mkdir(parents=True)
    shard_rows = [
        [
            _metasyn_row(3, title="Title three", abstract="Abstract three", sections=[]),
            _metasyn_row(
                7,
                title="Title seven",
                abstract="Abstract seven",
                sections=[{"heading": "Results", "text": "Full text result."}],
            ),
        ],
        [
            _metasyn_row(
                7 if duplicate_id else 11,
                title="Title eleven",
                abstract=None,
                sections=[],
            ),
            _metasyn_row(19, title=None, abstract=None, sections=[]),
        ],
    ]
    shards: list[Path] = []
    shard_contracts = []
    for index, rows in enumerate(shard_rows):
        path = corpus / f"train-{index:05d}-of-00002.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        shards.append(path)
        shard_contracts.append({"path": path.name, "rows": len(rows), "sha256": sha256_file(path)})
    license_path = root / "metasyn" / "LICENSE"
    license_path.write_text("Local fixture terms.\n", encoding="utf-8")
    manifest_path = root / "metasyn-corpus.json"
    manifest_path.write_text(
        json.dumps(
            {
                "corpus_manifest_version": "1",
                "dataset": "MetaSyn corpus",
                "source_repository": "THUIR/MetaSyn",
                "source_revision": "a" * 40,
                "local_root": "metasyn/corpus",
                "license_notice": {
                    "path": "metasyn/LICENSE",
                    "sha256": sha256_file(license_path),
                    "status": "local_evaluation_only_third_party_terms_apply",
                },
                "shards": shard_contracts,
                "total_rows": sum(len(rows) for rows in shard_rows),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path, shards


def test_antiox_bridge_binds_real_artifacts_and_keeps_scope_explicit(
    tmp_path: Path,
) -> None:
    papers, source_lines = _write_antiox_fixture(tmp_path)

    manifest, ledger = build_antiox_native_source_bridge(
        question_id="antiox-diagnostic",
        papers_path=papers,
        source_lines_path=source_lines,
        repository_root=tmp_path,
        expected_papers_sha256=sha256_file(papers),
        expected_source_lines_sha256=sha256_file(source_lines),
    )

    assert [record.doc_id for record in manifest.records] == ["PMC1"]
    source = manifest.records[0].source_document
    assert source.artifact_path == "archive/source_lines.json"
    assert source.sha256 == sha256_file(source_lines)
    assert source.source_locator == "json:archive/source_lines.json#/PMC1"
    assert ledger.source_records == 3
    assert ledger.native_manifest_records == 1
    assert ledger.source_available_records == 2
    assert ledger.source_absent_records == 1
    assert ledger.manifest_excluded_records == 2
    assert ledger.access_state.labels_previously_opened is True
    assert ledger.access_state.pristine_final_holdout_eligible is False
    assert ledger.access_state.scientific_role == "diagnostic_only"
    assert ledger.selection_scope == "successful_screened_in"
    by_doc = {record.doc_id: record for record in ledger.records}
    assert by_doc["PMC1"].content_scope is SourceContentScope.NUMBERED_SOURCE_LINES
    assert by_doc["PMC1"].estimability_status == "not_assessed_source_only"
    assert by_doc["PMC1"].extraction_attempted is False
    assert by_doc["PMC2"].manifest_exclusion_reason == "source_document_absent_or_empty"
    assert by_doc["PMC3"].manifest_exclusion_reason == ("outside_successful_screened_in_scope")
    assert by_doc["PMC3"].warnings == ["invalid_doi_omitted", "invalid_pmid_omitted"]
    assert ledger.native_source_manifest_sha256 == hash_canonical(manifest)


def test_antiox_bridge_all_source_scope_still_never_claims_estimability(
    tmp_path: Path,
) -> None:
    papers, source_lines = _write_antiox_fixture(tmp_path)

    manifest, ledger = build_antiox_native_source_bridge(
        question_id="antiox-diagnostic",
        papers_path=papers,
        source_lines_path=source_lines,
        repository_root=tmp_path,
        scope="all_source_available",
    )

    assert [record.doc_id for record in manifest.records] == ["PMC1", "PMC3"]
    assert all(record.extraction_attempted is False for record in ledger.records)
    assert {record.estimability_status for record in ledger.records} == {"not_assessed_source_only"}


def test_antiox_bridge_legacy_eligible_scope_is_explicit_and_diagnostic(
    tmp_path: Path,
) -> None:
    papers, source_lines = _write_antiox_fixture(tmp_path)

    manifest, ledger = build_antiox_native_source_bridge(
        question_id="antiox-diagnostic",
        papers_path=papers,
        source_lines_path=source_lines,
        repository_root=tmp_path,
        scope="legacy_eligible",
    )

    assert [record.doc_id for record in manifest.records] == ["PMC1"]
    by_doc = {record.doc_id: record for record in ledger.records}
    assert by_doc["PMC2"].manifest_exclusion_reason == (
        "outside_legacy_eligible_scope"
    )
    assert ledger.access_state.scientific_role == "diagnostic_only"
    assert ledger.selection_scope == "legacy_eligible"


def test_antiox_bridge_rejects_hash_drift_and_unknown_source_identity(
    tmp_path: Path,
) -> None:
    papers, source_lines = _write_antiox_fixture(tmp_path)

    with pytest.raises(SourceManifestBridgeError, match="source_artifact_hash_mismatch"):
        build_antiox_native_source_bridge(
            question_id="antiox-diagnostic",
            papers_path=papers,
            source_lines_path=source_lines,
            repository_root=tmp_path,
            expected_source_lines_sha256="0" * 64,
        )

    payload = json.loads(source_lines.read_text(encoding="utf-8"))
    payload["UNKNOWN"] = {"L1": {"section": "Results", "text": "Unjoinable source."}}
    source_lines.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        SourceManifestBridgeError,
        match="antiox_source_doc_ids_missing_from_papers",
    ):
        build_antiox_native_source_bridge(
            question_id="antiox-diagnostic",
            papers_path=papers,
            source_lines_path=source_lines,
            repository_root=tmp_path,
        )


def test_metasyn_bridge_hash_binds_selected_physical_rows_and_revision(
    tmp_path: Path,
) -> None:
    corpus_manifest, shards = _write_metasyn_fixture(tmp_path)

    manifest, ledger = build_metasyn_native_source_bridge(
        question_id="metasyn-diagnostic",
        corpus_manifest_path=corpus_manifest,
        repository_root=tmp_path,
        corpus_ids={7, 11},
    )

    by_doc = {record.doc_id: record for record in manifest.records}
    assert set(by_doc) == {"metasyn-corpus:7", "metasyn-corpus:11"}
    seven = by_doc["metasyn-corpus:7"].source_document
    assert seven.sha256 == sha256_file(shards[0])
    assert seven.source_locator.endswith("row_group=0&row_in_group=1&index_base=0&ID=7")
    eleven = by_doc["metasyn-corpus:11"].source_document
    assert eleven.sha256 == sha256_file(shards[1])
    assert eleven.source_locator.endswith("row_group=0&row_in_group=0&index_base=0&ID=11")
    ledger_by_doc = {record.doc_id: record for record in ledger.records}
    assert ledger_by_doc["metasyn-corpus:7"].content_scope is (
        SourceContentScope.FULL_TEXT_SECTIONS
    )
    assert ledger_by_doc["metasyn-corpus:11"].content_scope is (SourceContentScope.TITLE_ONLY)
    assert ledger.source_revision == "a" * 40
    assert ledger.selection_scope == "explicit_corpus_id_subset"
    assert ledger.dataset_version == f"THUIR/MetaSyn@{'a' * 40}"
    assert len([item for item in ledger.artifacts if item.role == "source_shard"]) == 2
    assert all(
        record.access_state.scientific_role == "diagnostic_only" for record in ledger.records
    )


def test_metasyn_missing_selection_duplicate_ids_and_shard_drift_fail_closed(
    tmp_path: Path,
) -> None:
    corpus_manifest, shards = _write_metasyn_fixture(tmp_path)

    with pytest.raises(SourceManifestBridgeError, match="metasyn_requested_corpus_ids_missing"):
        build_metasyn_native_source_bridge(
            question_id="metasyn-diagnostic",
            corpus_manifest_path=corpus_manifest,
            repository_root=tmp_path,
            corpus_ids={999},
        )

    shards[0].write_bytes(shards[0].read_bytes() + b"drift")
    with pytest.raises(SourceManifestBridgeError, match="corpus_shard_hash_mismatch"):
        build_metasyn_native_source_bridge(
            question_id="metasyn-diagnostic",
            corpus_manifest_path=corpus_manifest,
            repository_root=tmp_path,
            corpus_ids={7},
        )

    duplicate_root = tmp_path / "duplicate"
    duplicate_manifest, _ = _write_metasyn_fixture(duplicate_root, duplicate_id=True)
    with pytest.raises(SourceManifestBridgeError, match="metasyn_corpus_id_duplicate:7"):
        build_metasyn_native_source_bridge(
            question_id="metasyn-diagnostic",
            corpus_manifest_path=duplicate_manifest,
            repository_root=duplicate_root,
            corpus_ids={7},
        )


def test_diagnostic_ledger_self_hash_rejects_relabeling(tmp_path: Path) -> None:
    papers, source_lines = _write_antiox_fixture(tmp_path)
    _, ledger = build_antiox_native_source_bridge(
        question_id="antiox-diagnostic",
        papers_path=papers,
        source_lines_path=source_lines,
        repository_root=tmp_path,
    )
    tampered = ledger.model_dump(mode="json")
    tampered["license_status"] = "public_domain"

    with pytest.raises(ValidationError, match="diagnostic_source_ledger_hash_mismatch"):
        DiagnosticSourceLedger.model_validate(tampered)


def test_cli_writes_downstream_manifest_diagnostic_ledger_and_run_receipt(
    tmp_path: Path,
) -> None:
    papers, source_lines = _write_antiox_fixture(tmp_path)

    assert (
        bridge_main(
            [
                "--repository-root",
                str(tmp_path),
                "--question-id",
                "antiox-diagnostic",
                "--output-dir",
                "outputs",
                "--public-run-output",
                "public/source-run.json",
                "antiox",
                "--papers",
                papers.relative_to(tmp_path).as_posix(),
                "--source-lines",
                source_lines.relative_to(tmp_path).as_posix(),
            ]
        )
        == 0
    )

    output = tmp_path / "outputs"
    manifest = NativeSourceManifest.model_validate_json(
        (output / "native_source_manifest.json").read_text(encoding="utf-8")
    )
    ledger = DiagnosticSourceLedger.model_validate_json(
        (output / "diagnostic_source_ledger.json").read_text(encoding="utf-8")
    )
    run = json.loads((output / "source_manifest_bridge_run.json").read_text(encoding="utf-8"))
    public_run = json.loads((tmp_path / "public/source-run.json").read_text(encoding="utf-8"))
    assert len(manifest.records) == 1
    assert ledger.native_source_manifest_sha256 == hash_canonical(manifest)
    assert run["diagnostic_only"] is True
    assert run["selection_scope"] == "successful_screened_in"
    assert run["labels_previously_opened"] is True
    assert run["pristine_final_holdout_eligible"] is False
    assert run["run_sha256"] == hash_canonical(
        {key: value for key, value in run.items() if key != "run_sha256"}
    )
    assert public_run == run
    assert "records" not in public_run or isinstance(public_run["records"], int)


def test_metasyn_cli_requires_explicit_subset_or_all_rows(tmp_path: Path) -> None:
    corpus_manifest, _ = _write_metasyn_fixture(tmp_path)

    with pytest.raises(ValueError, match="MetaSyn requires"):
        bridge_main(
            [
                "--repository-root",
                str(tmp_path),
                "--question-id",
                "metasyn-diagnostic",
                "--output-dir",
                "outputs",
                "metasyn",
                "--corpus-manifest",
                corpus_manifest.relative_to(tmp_path).as_posix(),
            ]
        )
