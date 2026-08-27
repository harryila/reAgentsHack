"""Metadata-only audit of cached corpora and the strongest runnable local control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from literature_multiverse.closed_corpus import (
    ClosedCorpusGoldQuestion,
    ClosedCorpusPrediction,
    evaluate_closed_corpus,
)
from literature_multiverse.lineage import atomic_write_json, sha256_file
from literature_multiverse.metasyn_benchmark import (
    load_metasyn_labels,
    load_metasyn_manifest,
    load_metasyn_predictions,
)
from literature_multiverse.records import read_parquet_records


class LocalCorpusAuditError(ValueError):
    """Local cache metadata or a frozen control cannot be reconciled."""


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocalCorpusAuditError(f"invalid_json:{path}") from exc
    if not isinstance(value, dict):
        raise LocalCorpusAuditError(f"json_root_must_be_object:{path}")
    return value


def _overlap_counts(sets: dict[str, set[str]]) -> dict[str, int]:
    names = sorted(sets)
    return {
        f"{left}__{right}": len(sets[left] & sets[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }


def _antiox_inventory(
    papers_path: Path, findings_path: Path, source_lines_path: Path
) -> dict[str, Any]:
    papers = read_parquet_records(papers_path)
    findings = read_parquet_records(findings_path)
    source_lines = _json_object(source_lines_path)
    successful_included = [
        row
        for row in papers
        if row.get("screen_status") == "included" and row.get("map_status") == "success"
    ]
    pipeline_eligible = [row for row in successful_included if row.get("eligible") is True]
    finding_paper_ids = {str(row.get("paper_id")) for row in findings}
    eligible_zero = [
        row for row in pipeline_eligible if str(row.get("paper_id")) not in finding_paper_ids
    ]
    source_doc_ids = set(source_lines)
    successful_doc_ids = {str(row.get("doc_id")) for row in successful_included}
    return {
        "scope": "single_training_question_not_held_out",
        "papers": len(papers),
        "successfully_mapped_screened_in_papers": len(successful_included),
        "pipeline_labeled_eligible_papers": len(pipeline_eligible),
        "accepted_findings": len(findings),
        "papers_with_accepted_findings": len(finding_paper_ids),
        "pipeline_eligible_papers_with_zero_accepted_findings": len(eligible_zero),
        "source_documents": len(source_doc_ids),
        "successful_papers_missing_source_document": len(successful_doc_ids - source_doc_ids),
        "external_gold_included_set_available": False,
        "human_gold_conclusions_available": False,
        "retrieval_recall_identifiable": False,
        "input_hashes": {
            "papers_parquet": sha256_file(papers_path),
            "findings_parquet": sha256_file(findings_path),
            "source_lines": sha256_file(source_lines_path),
        },
    }


def _verified_evidence_inference_evaluation(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False, "status": "absent"}
    summary = _json_object(path)
    pilot = summary.get("successful_pilot")
    if (
        summary.get("gepa_pilot_summary_version") != "1"
        or summary.get("benchmark") != "Evidence Inference 2.0"
        or not isinstance(pilot, dict)
        or pilot.get("status") != "valid_frozen_heldout_evaluation"
    ):
        raise LocalCorpusAuditError("evidence_inference_evaluation_contract_invalid")
    try:
        heldout = pilot["artifacts"]["heldout_test_report"]
        heldout_path = Path(heldout["path"])
        expected_hash = heldout["sha256"]
    except (KeyError, TypeError) as exc:
        raise LocalCorpusAuditError("evidence_inference_evaluation_contract_invalid") from exc
    if (
        heldout_path.is_absolute()
        or not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or not heldout_path.is_file()
        or sha256_file(heldout_path) != expected_hash
    ):
        raise LocalCorpusAuditError("evidence_inference_heldout_artifact_invalid")
    return {
        "available": True,
        "status": "verified_frozen_heldout_evaluation",
        "summary_sha256": sha256_file(path),
        "heldout_report_sha256": expected_hash,
    }


def _evidence_inference_inventory(
    dataset_root: Path,
    manifest_path: Path,
    evaluation_summary_path: Path | None,
) -> dict[str, Any]:
    manifest = _json_object(manifest_path)
    split_names = ("train", "dev", "test")
    try:
        paper_sets = {
            split: {str(item) for item in manifest[split]["paper_ids"]} for split in split_names
        }
        split_rows = {split: int(manifest[split]["rows"]) for split in split_names}
    except (KeyError, TypeError, ValueError) as exc:
        raise LocalCorpusAuditError("evidence_inference_manifest_shape_invalid") from exc
    text_files = sorted((dataset_root / "txt_files").glob("*.txt"))
    return {
        "scope": "single_paper_extraction_questions_not_systematic_reviews",
        "local_full_text_files": len(text_files),
        "converted_rows": split_rows,
        "converted_distinct_papers": {split: len(paper_sets[split]) for split in split_names},
        "paper_overlap_across_splits": _overlap_counts(paper_sets),
        "review_level_included_corpus_labels_available": False,
        "retrieval_recall_identifiable": False,
        "cached_extraction_evaluation": _verified_evidence_inference_evaluation(
            evaluation_summary_path
        ),
        "input_hashes": {
            "split_manifest": sha256_file(manifest_path),
        },
    }


def _verified_metasyn_article_payloads(
    cache_dir: Path, gold_corpus_ids: set[str]
) -> tuple[int, bool]:
    """Verify the explicit ``matched-article-corpus/manifest.json`` convention."""

    corpus_root = cache_dir / "matched-article-corpus"
    article_manifest_path = corpus_root / "manifest.json"
    if not article_manifest_path.exists():
        return 0, False
    if article_manifest_path.is_symlink() or corpus_root.is_symlink():
        raise LocalCorpusAuditError("metasyn_article_manifest_symlink_forbidden")
    article_manifest = _json_object(article_manifest_path)
    articles = article_manifest.get("articles")
    if article_manifest.get("matched_article_corpus_version") != "1" or not isinstance(
        articles, list
    ):
        raise LocalCorpusAuditError("metasyn_article_manifest_contract_invalid")
    resolved_root = corpus_root.resolve(strict=True)
    seen_ids: set[str] = set()
    for record in articles:
        if not isinstance(record, dict):
            raise LocalCorpusAuditError("metasyn_article_manifest_contract_invalid")
        corpus_id = record.get("corpus_id")
        relative = record.get("path")
        expected_hash = record.get("sha256")
        if (
            not isinstance(corpus_id, (str, int))
            or not isinstance(relative, str)
            or not relative
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
        ):
            raise LocalCorpusAuditError("metasyn_article_manifest_contract_invalid")
        corpus_id = str(corpus_id)
        relative_path = Path(relative)
        candidate = corpus_root / relative_path
        if relative_path.is_absolute() or candidate.is_symlink():
            raise LocalCorpusAuditError("metasyn_article_payload_path_unsafe")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise LocalCorpusAuditError("metasyn_article_payload_missing") from exc
        if resolved_root not in resolved.parents or not resolved.is_file():
            raise LocalCorpusAuditError("metasyn_article_payload_path_unsafe")
        if resolved.suffix.casefold() not in {".txt", ".xml", ".pdf", ".html"}:
            raise LocalCorpusAuditError("metasyn_article_payload_type_invalid")
        if corpus_id in seen_ids or corpus_id not in gold_corpus_ids:
            raise LocalCorpusAuditError("metasyn_article_corpus_id_invalid")
        if sha256_file(resolved) != expected_hash:
            raise LocalCorpusAuditError("metasyn_article_payload_hash_mismatch")
        seen_ids.add(corpus_id)
    return len(seen_ids), seen_ids == gold_corpus_ids


def _metasyn_inventory(manifest_path: Path, cache_dir: Path) -> tuple[dict[str, Any], list[Any]]:
    manifest = load_metasyn_manifest(manifest_path)
    labels = load_metasyn_labels(manifest_path, manifest)
    paper_sets = {
        split: {
            str(paper_id)
            for label in labels
            if label.split == split
            for paper_id in label.gold_matched_corpus_ids
        }
        for split in ("development", "calibration", "test")
    }
    all_gold_corpus_ids = set().union(*paper_sets.values())
    article_payload_count, article_corpus_complete = _verified_metasyn_article_payloads(
        cache_dir, all_gold_corpus_ids
    )
    return (
        {
            "scope": "review_level_questions_and_matched_identifiers",
            "reviews": {
                "development": manifest.development.rows,
                "calibration": manifest.calibration.rows,
                "test": manifest.test.rows,
                "quarantined_official_train": len(manifest.quarantined_official_train),
            },
            "distinct_gold_matched_papers": {
                split: len(paper_sets[split]) for split in sorted(paper_sets)
            },
            "gold_paper_overlap_across_splits": _overlap_counts(paper_sets),
            "article_payload_convention": "matched-article-corpus/manifest.json@1",
            "local_article_payload_files": article_payload_count,
            "matched_article_corpus_available": article_payload_count > 0,
            "matched_article_corpus_complete": article_corpus_complete,
            "real_retrieval_runnable": article_corpus_complete,
            "real_extraction_runnable": article_corpus_complete,
            "input_hashes": {
                "manifest": sha256_file(manifest_path),
                "source_train": manifest.source_train.sha256,
                "source_test": manifest.source_test.sha256,
                "private_evaluator_labels": manifest.evaluator_labels.sha256,
            },
        },
        labels,
    )


def _cached_metasyn_control(
    *,
    labels: list[Any],
    predictions_path: Path,
    existing_evaluation_path: Path,
) -> dict[str, Any]:
    test_labels = sorted(
        (label for label in labels if label.split == "test"),
        key=lambda row: row.question_id,
    )
    metasyn_predictions = load_metasyn_predictions(predictions_path)
    prediction_by_review = {row.review_id: row for row in metasyn_predictions}
    gold = [
        ClosedCorpusGoldQuestion(
            question_id=label.question_id,
            split="test",
            gold_paper_ids=sorted(
                f"metasyn-corpus:{paper_id}" for paper_id in label.gold_matched_corpus_ids
            ),
            gold_conclusion=label.gold_direction,
        )
        for label in test_labels
    ]
    system_predictions = []
    for label in test_labels:
        prediction = prediction_by_review.get(label.review_id)
        if prediction is None:
            continue
        retrieved = (
            None
            if prediction.retrieved_corpus_ids is None
            else sorted(f"metasyn-corpus:{item}" for item in prediction.retrieved_corpus_ids)
        )
        system_predictions.append(
            ClosedCorpusPrediction(
                question_id=label.question_id,
                arm="system",
                retrieval_source="not_run" if retrieved is None else "system",
                extraction_source="not_run",
                retrieved_paper_ids=retrieved,
                extracted_paper_ids=None,
                predicted_conclusion=prediction.predicted_direction,
                abstained=prediction.predicted_direction == "Abstain",
            )
        )
    evaluation = evaluate_closed_corpus(gold=gold, predictions=system_predictions)
    existing = _json_object(existing_evaluation_path)
    system = evaluation["arms"]["system"]
    if system["conclusion"]["correct"] != existing["direction"]["correct"]:
        raise LocalCorpusAuditError("cached_metasyn_direction_reconciliation_failed")
    if (
        system["retrieval"]["micro_recall_missing_as_zero"]
        != existing["retrieval"]["micro_recall_missing_as_zero"]
    ):
        raise LocalCorpusAuditError("cached_metasyn_retrieval_reconciliation_failed")
    return {
        "name": "cached_metasyn_fixed_positive_question_only_control",
        "status": "complete",
        "scientific_scope": (
            "Trivial review-question direction control; it emitted no retrieval IDs, "
            "performed no extraction, and is not an end-to-end literature system."
        ),
        "end_to_end_claim_supported": False,
        "evaluation": evaluation,
        "input_hashes": {
            "predictions": sha256_file(predictions_path),
            "existing_metasyn_evaluation": sha256_file(existing_evaluation_path),
        },
    }


def _verified_private_packet_hashes(manifest_path: Path, packet: dict[str, Any]) -> dict[str, str]:
    expected_names = {
        "identity_key",
        "review_packet",
        "reviewer_a_decisions",
        "reviewer_b_decisions",
    }
    private_files = packet.get("local_private_files")
    if not isinstance(private_files, dict) or set(private_files) != expected_names:
        raise LocalCorpusAuditError("antiox_private_file_set_invalid")
    packet_dir = manifest_path.parent
    if manifest_path.is_symlink() or packet_dir.is_symlink():
        raise LocalCorpusAuditError("antiox_private_packet_symlink_forbidden")
    try:
        resolved_packet_dir = packet_dir.resolve(strict=True)
    except OSError as exc:
        raise LocalCorpusAuditError("antiox_private_packet_directory_missing") from exc
    verified: dict[str, str] = {}
    for name, metadata in sorted(private_files.items()):
        if not isinstance(metadata, dict):
            raise LocalCorpusAuditError("antiox_private_file_metadata_invalid")
        relative = metadata.get("path")
        expected_hash = metadata.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
        ):
            raise LocalCorpusAuditError("antiox_private_file_metadata_invalid")
        relative_path = Path(relative)
        candidate = packet_dir / relative_path
        if relative_path.is_absolute() or candidate.is_symlink():
            raise LocalCorpusAuditError("antiox_private_file_path_unsafe")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise LocalCorpusAuditError("antiox_private_file_missing") from exc
        if resolved_packet_dir not in resolved.parents or not resolved.is_file():
            raise LocalCorpusAuditError("antiox_private_file_path_unsafe")
        actual_hash = sha256_file(resolved)
        if actual_hash != expected_hash:
            raise LocalCorpusAuditError("antiox_private_file_hash_mismatch")
        verified[name] = actual_hash
    return verified


def build_local_corpus_audit(
    *,
    metasyn_manifest_path: Path,
    metasyn_cache_dir: Path,
    metasyn_predictions_path: Path,
    metasyn_evaluation_path: Path,
    evidence_inference_root: Path,
    evidence_inference_manifest_path: Path,
    antiox_papers_path: Path,
    antiox_findings_path: Path,
    antiox_source_lines_path: Path,
    antiox_packet_manifest_path: Path,
    evidence_inference_evaluation_summary_path: Path | None = None,
) -> dict[str, Any]:
    """Return a deterministic, metadata-only feasibility and leakage audit."""

    metasyn, labels = _metasyn_inventory(metasyn_manifest_path, metasyn_cache_dir)
    evidence_inference = _evidence_inference_inventory(
        evidence_inference_root,
        evidence_inference_manifest_path,
        evidence_inference_evaluation_summary_path,
    )
    antiox = _antiox_inventory(antiox_papers_path, antiox_findings_path, antiox_source_lines_path)
    baseline = _cached_metasyn_control(
        labels=labels,
        predictions_path=metasyn_predictions_path,
        existing_evaluation_path=metasyn_evaluation_path,
    )
    packet = _json_object(antiox_packet_manifest_path)
    if (
        packet.get("human_review_packet_manifest_version") != "1"
        or packet.get("sample_size") != 60
        or packet.get("manifest_contains_article_text") is not False
        or packet.get("review_packet_contains_article_text") is not True
        or packet.get("contains_model_confidence") is not False
        or packet.get("all_eligible_zero_finding_papers_included") is not True
    ):
        raise LocalCorpusAuditError("antiox_human_packet_contract_invalid")
    verified_private_hashes = _verified_private_packet_hashes(antiox_packet_manifest_path, packet)
    return {
        "local_corpus_audit_version": "2",
        "contains_article_text": False,
        "contains_question_text": False,
        "contains_titles": False,
        "contains_per_question_labels": False,
        "network_or_api_calls": 0,
        "corpora": {
            "metasyn": metasyn,
            "evidence_inference_2": evidence_inference,
            "antiox_training": antiox,
        },
        "cached_local_baseline": baseline,
        "human_review_packet": {
            "status": "prepared_not_adjudicated",
            "question_id": packet["question_id"],
            "sample_size": packet["sample_size"],
            "reviewers": packet["reviewers"],
            "selected_strata": packet["selected_strata"],
            "all_eligible_zero_finding_papers_included": packet[
                "all_eligible_zero_finding_papers_included"
            ],
            "all_pipeline_eligible_papers_included": packet[
                "all_pipeline_eligible_papers_included"
            ],
            "metadata_summary_contains_article_text": False,
            "private_review_packet_contains_article_text": True,
            "contains_model_confidence": False,
            "private_manifest_sha256": sha256_file(antiox_packet_manifest_path),
            "private_file_hashes": verified_private_hashes,
        },
        "external_blockers": [
            {
                "code": "metasyn_matched_article_corpus_absent",
                "blocks": [
                    "real_retrieval",
                    "real_extraction",
                    "oracle_corpus_system_extraction",
                    "oracle_extraction_system_synthesis",
                ],
            },
            {
                "code": "antiox_external_gold_included_set_absent",
                "blocks": ["retrieval_recall", "scientific_accuracy"],
            },
            {
                "code": "human_adjudication_not_completed",
                "blocks": ["human_extraction_accuracy", "real_question_level_calibration"],
            },
        ],
        "claim_boundary": (
            "The local cache supports a leakage audit, a 60-paper blinded review packet, "
            "a verified cached Evidence Inference extraction pilot when present, and a "
            "trivial MetaSyn "
            "question-only control. It does not support a real closed-corpus end-to-end "
            "accuracy or retrieval-recall claim."
        ),
    }


def write_local_corpus_audit(path: Path, report: dict[str, Any], *, force: bool = False) -> None:
    atomic_write_json(path, report, force=force)


__all__ = [
    "LocalCorpusAuditError",
    "build_local_corpus_audit",
    "write_local_corpus_audit",
]
