from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import literature_multiverse.cli as cli_module
from literature_multiverse.acquisition import (
    AcquiredCorpusReplay,
    AcquisitionContractError,
    AcquisitionReplayReceiptV1,
    FrozenNativeExtractionRecordV1,
    ProtocolScreenDecisionV1,
    freeze_acquisition_manifest,
    freeze_native_extraction_ledger,
    freeze_protocol_screening_receipt,
    replay_frozen_acquisition,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file
from literature_multiverse.native_extraction import NativePublicationExtraction
from literature_multiverse.native_grounding import (
    NativeGroundingError,
    resolve_native_source_document,
)
from literature_multiverse.typed_extraction import SourceDocumentArtifact
from literature_multiverse.verifier import ClaimManifest, build_offline_fixture


def _extraction(*, source_locator: str) -> NativePublicationExtraction:
    return NativePublicationExtraction.model_validate(
        {
            "status": "estimable",
            "studies": [
                {
                    "key": "trial",
                    "source_label": "Trial",
                    "design": "randomized trial",
                    "registration_ids": [],
                    "cohorts": [
                        {
                            "key": "cohort",
                            "source_labels": ["reported cohort"],
                            "registry_ids": [],
                            "dataset_ids": [],
                            "total_sample_size": 100,
                            "arms": [
                                {
                                    "key": "treatment",
                                    "label": "Intervention",
                                    "role": "intervention",
                                    "sample_size": 50,
                                },
                                {
                                    "key": "control",
                                    "label": "Control",
                                    "role": "control",
                                    "sample_size": 50,
                                },
                            ],
                            "contrasts": [
                                {
                                    "key": "primary",
                                    "treatment_arm_key": "treatment",
                                    "comparator_arm_key": "control",
                                    "label": "intervention_vs_control",
                                    "positive_direction_means": "higher performance",
                                }
                            ],
                            "findings": [
                                {
                                    "key": "primary-result",
                                    "contrast_key": "primary",
                                    "outcome_name": "performance",
                                    "timepoint": {
                                        "kind": "exact",
                                        "value": 4,
                                        "unit": "week",
                                    },
                                    "effect": {
                                        "effect_format": "hedges_g",
                                        "estimate": 0.35,
                                        "standard_error": 0.1,
                                        "reported_significance": "significant",
                                    },
                                    "evidence": {
                                        "source_locator": source_locator,
                                        "quote": "The effect estimate was 0.35 (SE 0.10).",
                                        "section": "Results",
                                        "line_ids": ["L1"],
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )


def _claim() -> ClaimManifest:
    return ClaimManifest.model_validate(
        {
            "question_id": "acquisition-fixture",
            "population_id": "fixture-population",
            "domain": "fixture-domain",
            "claim": {
                "statement": "The intervention increases performance.",
                "direction": "increase",
                "outcome_name": "performance",
            },
            "protocol": {
                "corpus_cutoff": "frozen-fixture-v1",
                "inclusion_criteria": ["Randomized intervention studies"],
                "exclusion_criteria": ["Review articles"],
            },
        }
    )


def _inputs(root: Path) -> tuple[Path, Path, str]:
    source = root / "source" / "article.xml"
    source.parent.mkdir(parents=True)
    source.write_text(
        "<article><body><sec><title>Results</title>"
        "<p>The effect estimate was 0.35 (SE 0.10).</p>"
        "</sec></body></article>",
        encoding="utf-8",
    )
    source_sha256 = sha256_file(source)
    corpus = root / "source" / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "DOC-1",
                        "source": "frozen-fixture",
                        "title": "A randomized intervention trial",
                        "doi": "10.5555/acquisition.1",
                        "pub_year": 2024,
                        "article_type": "research-article",
                        "publication_status": "peer_reviewed",
                        "full_text_path": "article.xml",
                        "full_text_sha256": source_sha256,
                        "full_text_media_type": "application/xml",
                    },
                    {
                        "document_id": "DOC-2",
                        "source": "frozen-fixture",
                        "title": "A review of intervention trials",
                        "doi": "10.5555/acquisition.2",
                        "pub_year": 2025,
                        "article_type": "review-article",
                        "publication_status": "peer_reviewed",
                    },
                ],
                "search_results": {"intervention performance": ["DOC-1", "DOC-2"]},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    ledger = freeze_native_extraction_ledger(
        question_id="acquisition-fixture",
        records=[
            FrozenNativeExtractionRecordV1(
                doc_id="DOC-1",
                extraction=_extraction(source_locator=f"harvest-sha256:{source_sha256}"),
            )
        ],
    )
    ledger_path = root / "source" / "native-extractions.json"
    atomic_write_json(ledger_path, ledger)
    return corpus, ledger_path, source_sha256


def _manifest(root: Path, corpus: Path, ledger: Path) -> object:
    return freeze_acquisition_manifest(
        {
            "question_id": "acquisition-fixture",
            "corpus_cutoff": "frozen-fixture-v1",
            "frozen_corpus": {
                "path": corpus.relative_to(root).as_posix(),
                "sha256": sha256_file(corpus),
            },
            "queries": [{"family": "primary", "query": "intervention performance"}],
            "per_query_limit": 10,
            "retrieved_at": datetime(2026, 8, 29, 12, tzinfo=UTC),
            "allowed_article_types": ["research-article"],
            "expected_retrieved_doc_ids": ["DOC-1", "DOC-2"],
            "expected_included_paper_ids": ["doi:10.5555/acquisition.1"],
            "expected_excluded_paper_ids": ["doi:10.5555/acquisition.2"],
            "native_input": {
                "mode": "frozen_extraction_ledger",
                "artifact": {
                    "path": ledger.relative_to(root).as_posix(),
                    "sha256": sha256_file(ledger),
                },
            },
        }
    )


def test_harvest_archive_xml_locator_is_exact_and_hash_bound(tmp_path: Path) -> None:
    source = tmp_path / "article.xml"
    source.write_text(
        "<article><body><sec><title>Results</title>"
        "<p>The effect estimate was 0.35 (SE 0.10).</p>"
        "</sec></body></article>",
        encoding="utf-8",
    )
    digest = sha256_file(source)
    resolved = resolve_native_source_document(
        repository_root=tmp_path,
        source_document=SourceDocumentArtifact(
            artifact_path="article.xml",
            sha256=digest,
            media_type="application/xml",
            source_locator=f"harvest-sha256:{digest}",
        ),
    )

    assert resolved.source_kind == "harvest_archive_text"
    assert resolved.lines[0].section == "Results"
    assert resolved.lines[0].text == "The effect estimate was 0.35 (SE 0.10)."


def test_harvest_archive_xml_forbids_doctype_and_entities(tmp_path: Path) -> None:
    source = tmp_path / "article.xml"
    source.write_text(
        '<!DOCTYPE article [<!ENTITY unsafe "expanded">]>'
        "<article><body><p>&unsafe;</p></body></article>",
        encoding="utf-8",
    )
    digest = sha256_file(source)
    with pytest.raises(NativeGroundingError, match="harvest_xml_doctype_or_entity_forbidden"):
        resolve_native_source_document(
            repository_root=tmp_path,
            source_document=SourceDocumentArtifact(
                artifact_path="article.xml",
                sha256=digest,
                media_type="application/xml",
                source_locator=f"harvest-sha256:{digest}",
            ),
        )


def test_frozen_acquisition_replays_to_standard_native_corpus(tmp_path: Path) -> None:
    corpus_path, ledger_path, _ = _inputs(tmp_path)
    manifest = _manifest(tmp_path, corpus_path, ledger_path)
    output = tmp_path / "output"
    replay = replay_frozen_acquisition(
        manifest=manifest,
        claim_manifest=_claim(),
        repository_root=tmp_path,
        pipeline_sha256="d" * 64,
        output_dir=output,
    )

    assert replay.corpus.source_format == "typed_evidence_grounding_package_json"
    assert len(replay.corpus.graph.outcome_estimates) == 1
    assert replay.receipt.retrieved_doc_ids == ["DOC-1", "DOC-2"]
    assert replay.receipt.included_paper_ids == ["doi:10.5555/acquisition.1"]
    assert replay.receipt.excluded_paper_ids == ["doi:10.5555/acquisition.2"]
    assert replay.receipt.counts["native_fragments"] == 1
    assert "protocol_eligibility_screening_unverified" in {
        issue.code for issue in replay.corpus.adapter_issues
    }
    assert replay.corpus.provenance_release_eligible() is False
    assert replay.package_path.is_file()

    second_output = tmp_path / "replay-output"
    second = replay_frozen_acquisition(
        manifest=manifest,
        claim_manifest=_claim(),
        repository_root=tmp_path,
        pipeline_sha256="d" * 64,
        output_dir=second_output,
    )
    assert second.receipt.occurrence_membership_sha256 != (
        replay.receipt.occurrence_membership_sha256
    )
    assert second.receipt.screening_membership_sha256 == (
        replay.receipt.screening_membership_sha256
    )


def test_self_asserted_two_reviewer_screening_cannot_remove_screening_blocker(
    tmp_path: Path,
) -> None:
    corpus_path, ledger_path, _ = _inputs(tmp_path)
    claim = _claim()
    screening = freeze_protocol_screening_receipt(
        question_id=claim.question_id,
        protocol_sha256=hash_canonical(claim.protocol),
        corpus_cutoff=claim.protocol.corpus_cutoff,
        provenance="blinded_human",
        adjudicator_count=2,
        decisions=[
            ProtocolScreenDecisionV1(
                paper_id="doi:10.5555/acquisition.1",
                doc_id="DOC-1",
                status="included",
                reason="Meets the frozen population/intervention/outcome protocol.",
            ),
            ProtocolScreenDecisionV1(
                paper_id="doi:10.5555/acquisition.2",
                doc_id="DOC-2",
                status="excluded",
                reason="Review article, not a primary study.",
            ),
        ],
    )
    screening_path = tmp_path / "source" / "protocol-screening.json"
    atomic_write_json(screening_path, screening)
    manifest_payload = _manifest(tmp_path, corpus_path, ledger_path).model_dump(
        mode="json", exclude={"manifest_sha256"}
    )
    manifest_payload["protocol_screening"] = {
        "path": screening_path.relative_to(tmp_path).as_posix(),
        "sha256": sha256_file(screening_path),
    }
    manifest = freeze_acquisition_manifest(manifest_payload)

    replay = replay_frozen_acquisition(
        manifest=manifest,
        claim_manifest=claim,
        repository_root=tmp_path,
        pipeline_sha256="d" * 64,
        output_dir=tmp_path / "output",
    )

    assert replay.receipt.screening_authority == "blinded_human"
    assert replay.receipt.protocol_screening_receipt_sha256 == screening.receipt_sha256
    assert "missing_verified_screening_adjudication_package" in {
        issue.code for issue in replay.corpus.adapter_issues
    }
    assert replay.corpus.provenance_release_eligible() is False


def test_frozen_acquisition_rejects_query_truncation_before_extraction(
    tmp_path: Path,
) -> None:
    corpus_path, ledger_path, _ = _inputs(tmp_path)
    manifest_payload = _manifest(tmp_path, corpus_path, ledger_path).model_dump(
        mode="json", exclude={"manifest_sha256"}
    )
    manifest_payload["per_query_limit"] = 1
    manifest = freeze_acquisition_manifest(manifest_payload)

    with pytest.raises(AcquisitionContractError, match="acquisition_query_would_truncate"):
        replay_frozen_acquisition(
            manifest=manifest,
            claim_manifest=_claim(),
            repository_root=tmp_path,
            pipeline_sha256="d" * 64,
            output_dir=tmp_path / "output",
        )


def test_frozen_acquisition_rejects_missing_native_member(tmp_path: Path) -> None:
    corpus_path, ledger_path, _ = _inputs(tmp_path)
    empty_payload = {
        "ledger_version": "frozen-native-extraction-ledger-v1",
        "question_id": "acquisition-fixture",
        "records": [],
    }
    # The ledger contract itself fails closed before a graph can be assembled.
    empty_payload["ledger_sha256"] = hashlib.sha256(b"unused").hexdigest()
    ledger_path.write_text(json.dumps(empty_payload), encoding="utf-8")
    manifest_payload = _manifest(tmp_path, corpus_path, ledger_path).model_dump(
        mode="json", exclude={"manifest_sha256"}
    )
    manifest_payload["native_input"]["artifact"]["sha256"] = sha256_file(ledger_path)
    manifest = freeze_acquisition_manifest(manifest_payload)

    with pytest.raises(ValueError, match="List should have at least 1 item"):
        replay_frozen_acquisition(
            manifest=manifest,
            claim_manifest=_claim(),
            repository_root=tmp_path,
            pipeline_sha256="d" * 64,
            output_dir=tmp_path / "output",
        )


def test_public_verify_cli_routes_acquisition_into_standard_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    claim, corpus = build_offline_fixture()
    claim_path = tmp_path / "claim.json"
    claim_path.write_text(claim.model_dump_json(indent=2), encoding="utf-8")
    acquisition_path = tmp_path / "acquisition.json"
    acquisition_path.write_text("{}", encoding="utf-8")
    receipt_payload = {
        "receipt_version": "acquisition-replay-receipt-v1",
        "acquisition_manifest_sha256": "a" * 64,
        "claim_protocol_sha256": "b" * 64,
        "frozen_corpus_sha256": "c" * 64,
        "occurrence_membership_sha256": "d" * 64,
        "screening_membership_sha256": "e" * 64,
        "native_source_manifest_sha256": "f" * 64,
        "typed_grounding_package_sha256": "1" * 64,
        "native_mode": "frozen_extraction_ledger",
        "protocol_screening_receipt_sha256": None,
        "screening_authority": "deterministic_article_type_only",
        "archive_entries": [],
        "retrieved_doc_ids": ["fixture-doc"],
        "included_paper_ids": ["fixture-paper"],
        "excluded_paper_ids": [],
        "counts": {"retrieved_documents": 1},
        "limitations": ["offline_fixture"],
    }
    receipt = AcquisitionReplayReceiptV1.model_validate(
        {**receipt_payload, "receipt_sha256": hash_canonical(receipt_payload)}
    )

    monkeypatch.setattr(cli_module, "load_acquisition_manifest", lambda path: object())

    def replay(**kwargs: object) -> AcquiredCorpusReplay:
        assert kwargs["claim_manifest"] == claim
        return AcquiredCorpusReplay(
            corpus=corpus,
            receipt=receipt,
            package_path=Path("unused-fixture-package.json"),
        )

    monkeypatch.setattr(cli_module, "replay_frozen_acquisition", replay)
    output = tmp_path / "certificate"
    assert (
        cli_module.main(
            [
                "verify",
                "--claim",
                str(claim_path),
                "--acquisition-manifest",
                str(acquisition_path),
                "--budget-minutes",
                "30",
                "--analysis-only-uncalibrated-audit",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["acquisition_replay_receipt_sha256"] == receipt.receipt_sha256
    assert (output / "acquisition-replay-receipt.json").is_file()
    certificate = json.loads((output / "verification-certificate.json").read_text())
    assert certificate["corpus"]["source_format"] == "embedded_synthetic_fixture"
