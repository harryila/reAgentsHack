from __future__ import annotations

import json
from pathlib import Path

import pytest

from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.local_corpus_audit import (
    LOCAL_CORPUS_AUDIT_VERSION,
    LocalCorpusAuditError,
    _source_code_hashes,
    _verified_evidence_inference_evaluation,
    _verified_metasyn_article_payloads,
    _verified_private_packet_hashes,
    validate_local_corpus_audit,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_metasyn_arbitrary_metadata_json_is_not_an_article_payload(tmp_path: Path) -> None:
    _write_json(tmp_path / "manifest.json", {"articles": ["not an article corpus"]})
    _write_json(tmp_path / "nested" / "metadata.json", {"corpus_id": "p1"})

    assert _verified_metasyn_article_payloads(tmp_path, {"p1"}) == (0, False)


def test_metasyn_explicit_hash_bound_article_corpus_must_be_complete(tmp_path: Path) -> None:
    corpus = tmp_path / "matched-article-corpus"
    article = corpus / "articles" / "p1.txt"
    article.parent.mkdir(parents=True)
    article.write_text("full article", encoding="utf-8")
    _write_json(
        corpus / "manifest.json",
        {
            "matched_article_corpus_version": "1",
            "articles": [
                {"corpus_id": "p1", "path": "articles/p1.txt", "sha256": sha256_file(article)}
            ],
        },
    )

    assert _verified_metasyn_article_payloads(tmp_path, {"p1", "p2"}) == (1, False)
    assert _verified_metasyn_article_payloads(tmp_path, {"p1"}) == (1, True)


def test_evidence_inference_evaluation_is_absent_without_artifact(tmp_path: Path) -> None:
    assert _verified_evidence_inference_evaluation(None) == {
        "available": False,
        "status": "absent",
    }
    assert _verified_evidence_inference_evaluation(tmp_path / "missing.json") == {
        "available": False,
        "status": "absent",
    }


def test_evidence_inference_evaluation_verifies_fail_closed_receipt_audit(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.json"
    payload = {
        "public_summary_version": "evidence-inference-diagnostic-public-summary-v1",
        "status": "metadata_only_diagnostic_non_pristine",
        "full_report_sha256": "a" * 64,
        "cache_integrity_findings": [
            {
                "status": "fail_closed_trace_score_excluded",
                "archived_trace_score_citation_allowed": False,
                "expected_dev_rows": 12,
                "clean_common_dev_receipts": 10,
                "missing_mutation_dev_receipts": 2,
            }
        ],
    }
    _write_json(summary, {**payload, "public_summary_sha256": hash_canonical(payload)})
    result = _verified_evidence_inference_evaluation(summary)
    assert result["available"] is True
    assert result["archived_trace_score_citation_allowed"] is False
    tampered = json.loads(summary.read_text(encoding="utf-8"))
    tampered["cache_integrity_findings"][0]["clean_common_dev_receipts"] = 12
    _write_json(summary, tampered)
    with pytest.raises(LocalCorpusAuditError, match="evaluation_contract_invalid"):
        _verified_evidence_inference_evaluation(summary)


def _packet_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    manifest = tmp_path / "packet" / "manifest.json"
    files: dict[str, object] = {}
    for name in (
        "identity_key",
        "review_packet",
        "reviewer_a_decisions",
        "reviewer_b_decisions",
    ):
        path = manifest.parent / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        files[name] = {"path": path.name, "rows": 1, "sha256": sha256_file(path)}
    return manifest, {"local_private_files": files}


def test_private_packet_recomputes_all_file_hashes(tmp_path: Path) -> None:
    manifest, packet = _packet_fixture(tmp_path)
    verified = _verified_private_packet_hashes(manifest, packet)
    assert set(verified) == set(packet["local_private_files"])
    (manifest.parent / "review_packet.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(LocalCorpusAuditError, match="hash_mismatch"):
        _verified_private_packet_hashes(manifest, packet)


def test_private_packet_rejects_missing_out_of_root_and_symlink_files(tmp_path: Path) -> None:
    manifest, packet = _packet_fixture(tmp_path)
    metadata = packet["local_private_files"]
    assert isinstance(metadata, dict)

    missing_packet = json.loads(json.dumps(packet))
    missing_packet["local_private_files"]["review_packet"]["path"] = "missing.jsonl"
    with pytest.raises(LocalCorpusAuditError, match="private_file_missing"):
        _verified_private_packet_hashes(manifest, missing_packet)

    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    outside_packet = json.loads(json.dumps(packet))
    outside_packet["local_private_files"]["review_packet"] = {
        "path": "../outside.jsonl",
        "sha256": sha256_file(outside),
    }
    with pytest.raises(LocalCorpusAuditError, match="path_unsafe"):
        _verified_private_packet_hashes(manifest, outside_packet)

    target = manifest.parent / "review_packet.jsonl"
    link = manifest.parent / "linked.jsonl"
    link.symlink_to(target.name)
    symlink_packet = json.loads(json.dumps(packet))
    symlink_packet["local_private_files"]["review_packet"] = {
        "path": link.name,
        "sha256": sha256_file(target),
    }
    with pytest.raises(LocalCorpusAuditError, match="path_unsafe"):
        _verified_private_packet_hashes(manifest, symlink_packet)


def test_local_corpus_audit_full_payload_and_source_lineage_fail_closed() -> None:
    payload = {
        "local_corpus_audit_version": LOCAL_CORPUS_AUDIT_VERSION,
        "contains_article_text": False,
        "contains_question_text": False,
        "contains_titles": False,
        "contains_per_question_labels": False,
        "network_or_api_calls": 0,
        "source_code_sha256s": _source_code_hashes(),
    }
    report = {**payload, "audit_payload_sha256": hash_canonical(payload)}
    assert validate_local_corpus_audit(report, require_current_sources=True) == report

    tampered = json.loads(json.dumps(report))
    tampered["network_or_api_calls"] = 1
    with pytest.raises(LocalCorpusAuditError, match="integrity_invalid"):
        validate_local_corpus_audit(tampered)
