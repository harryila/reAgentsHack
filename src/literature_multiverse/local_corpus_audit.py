"""Metadata-only audit of cached corpora and the strongest runnable local control."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from literature_multiverse.closed_corpus import (
    ClosedCorpusGoldQuestion,
    ClosedCorpusPrediction,
    evaluate_closed_corpus,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file
from literature_multiverse.metasyn_benchmark import (
    load_metasyn_labels,
    load_metasyn_manifest,
    load_metasyn_predictions,
)
from literature_multiverse.metasyn_retrieval import (
    MetaSynCorpusError,
    inspect_corpus_coverage,
    verify_corpus_manifest,
)
from literature_multiverse.records import read_parquet_records


class LocalCorpusAuditError(ValueError):
    """Local cache metadata or a frozen control cannot be reconciled."""


LOCAL_CORPUS_AUDIT_VERSION = "3"
_SOURCE_CODE_PATHS = (
    "pyproject.toml",
    "scripts/audit_local_corpora.py",
    "src/literature_multiverse/__init__.py",
    "src/literature_multiverse/calibration.py",
    "src/literature_multiverse/closed_corpus.py",
    "src/literature_multiverse/lineage.py",
    "src/literature_multiverse/local_corpus_audit.py",
    "src/literature_multiverse/metasyn_benchmark.py",
    "src/literature_multiverse/metasyn_retrieval.py",
    "src/literature_multiverse/models.py",
    "src/literature_multiverse/paths.py",
    "src/literature_multiverse/records.py",
    "uv.lock",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _source_code_hashes() -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[2]
    missing = [
        relative
        for relative in _SOURCE_CODE_PATHS
        if not (repository_root / relative).is_file()
    ]
    if missing:
        raise LocalCorpusAuditError(f"local_corpus_audit_source_missing:{missing}")
    return {
        relative: sha256_file(repository_root / relative)
        for relative in _SOURCE_CODE_PATHS
    }


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
    observed_hash = summary.get("public_summary_sha256")
    unhashed = {
        key: value for key, value in summary.items() if key != "public_summary_sha256"
    }
    findings = summary.get("cache_integrity_findings")
    if (
        summary.get("public_summary_version")
        != "evidence-inference-diagnostic-public-summary-v1"
        or summary.get("status") != "metadata_only_diagnostic_non_pristine"
        or not isinstance(observed_hash, str)
        or observed_hash != hash_canonical(unhashed)
        or not isinstance(findings, list)
        or len(findings) != 1
    ):
        raise LocalCorpusAuditError("evidence_inference_evaluation_contract_invalid")
    finding = findings[0]
    if (
        not isinstance(finding, dict)
        or finding.get("status") != "fail_closed_trace_score_excluded"
        or finding.get("archived_trace_score_citation_allowed") is not False
        or finding.get("expected_dev_rows") != 12
        or finding.get("clean_common_dev_receipts") != 10
        or finding.get("missing_mutation_dev_receipts") != 2
    ):
        raise LocalCorpusAuditError("evidence_inference_evaluation_contract_invalid")
    return {
        "available": True,
        "status": "verified_nonpristine_receipt_audit_trace_score_excluded",
        "summary_sha256": sha256_file(path),
        "public_summary_payload_sha256": observed_hash,
        "full_private_report_sha256": summary.get("full_report_sha256"),
        "archived_trace_score_citation_allowed": False,
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


def _metasyn_inventory(
    manifest_path: Path,
    cache_dir: Path,
    *,
    corpus_manifest_path: Path | None = None,
    repository_root: Path | None = None,
) -> tuple[dict[str, Any], list[Any]]:
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
    official_corpus: dict[str, Any] | None = None
    if corpus_manifest_path is not None:
        if repository_root is None:
            raise LocalCorpusAuditError("metasyn_corpus_repository_root_required")
        try:
            corpus_manifest, shard_paths = verify_corpus_manifest(
                corpus_manifest_path, repository_root=repository_root
            )
            coverage = inspect_corpus_coverage(
                shard_paths, required_corpus_ids=all_gold_corpus_ids
            )
        except MetaSynCorpusError as exc:
            raise LocalCorpusAuditError(f"metasyn_official_corpus_invalid:{exc}") from exc
        article_payload_count = coverage["rows"]
        article_corpus_complete = bool(coverage["required_gold_ids_complete"])
        official_corpus = {
            "source_repository": corpus_manifest.source_repository,
            "source_revision": corpus_manifest.source_revision,
            "manifest_sha256": sha256_file(corpus_manifest_path),
            "shard_sha256s": {
                shard.path: shard.sha256 for shard in corpus_manifest.shards
            },
            "coverage": coverage,
            "license_status": corpus_manifest.license_notice.status,
        }
    else:
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
            "article_payload_convention": (
                "revision_pinned_official_parquet_shards@1"
                if official_corpus is not None
                else "matched-article-corpus/manifest.json@1"
            ),
            "local_article_payload_files": article_payload_count,
            "matched_article_corpus_available": article_payload_count > 0,
            "matched_article_corpus_complete": article_corpus_complete,
            "real_retrieval_runnable": article_corpus_complete,
            # Payload availability is necessary but not sufficient: the production
            # extractor still emits legacy findings rather than typed numerical graph
            # records for this benchmark.
            "real_extraction_runnable": False,
            "official_corpus": official_corpus,
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
    metasyn_corpus_manifest_path: Path | None = None,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Return a deterministic, metadata-only feasibility and leakage audit."""

    metasyn, labels = _metasyn_inventory(
        metasyn_manifest_path,
        metasyn_cache_dir,
        corpus_manifest_path=metasyn_corpus_manifest_path,
        repository_root=repository_root,
    )
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
        packet.get("human_review_packet_manifest_version") not in {"1", "2"}
        or packet.get("sample_size") != 60
        or packet.get("manifest_contains_article_text") is not False
        or packet.get("review_packet_contains_article_text") is not True
        or packet.get("contains_model_confidence") is not False
        or packet.get("all_eligible_zero_finding_papers_included") is not True
    ):
        raise LocalCorpusAuditError("antiox_human_packet_contract_invalid")
    verified_private_hashes = _verified_private_packet_hashes(antiox_packet_manifest_path, packet)
    payload = {
        "local_corpus_audit_version": LOCAL_CORPUS_AUDIT_VERSION,
        "contains_article_text": False,
        "contains_question_text": False,
        "contains_titles": False,
        "contains_per_question_labels": False,
        "network_or_api_calls": 0,
        "source_code_sha256s": _source_code_hashes(),
        "hash_security_boundary": (
            "unkeyed reproducibility and tamper-evidence hashes; not signatures, "
            "authorship proof, freshness proof, or rollback protection"
        ),
        "corpora": {
            "metasyn": metasyn,
            "evidence_inference_2": evidence_inference,
            "antiox_training": antiox,
        },
        "cached_local_baseline": baseline,
        "human_review_packet": {
            "status": "prepared_not_adjudicated",
            "packet_manifest_version": packet["human_review_packet_manifest_version"],
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
            *(
                []
                if metasyn["matched_article_corpus_complete"]
                else [
                    {
                        "code": "metasyn_matched_article_corpus_absent",
                        "blocks": [
                            "real_retrieval",
                            "real_extraction",
                            "oracle_corpus_system_extraction",
                            "oracle_extraction_system_synthesis",
                        ],
                    }
                ]
            ),
            {
                "code": "metasyn_typed_extractor_not_connected",
                "blocks": [
                    "real_extraction",
                    "oracle_corpus_system_extraction",
                    "oracle_extraction_system_synthesis",
                    "closed_corpus_end_to_end_accuracy",
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
            "The local cache supports a leakage audit, a revision-pinned MetaSyn corpus, "
            "a real lexical retrieval baseline, a 60-paper blinded review packet, "
            "a receipt-audited non-pristine Evidence Inference diagnostic, and a "
            "trivial MetaSyn question-only control. MetaSyn Recall@k is identifiable only "
            "against its released matched-paper subset; the cache does not support an "
            "exhaustive-eligibility recall claim or closed-corpus end-to-end synthesis "
            "accuracy claim."
        ),
    }
    return {**payload, "audit_payload_sha256": hash_canonical(payload)}


def validate_local_corpus_audit(
    report: dict[str, Any], *, require_current_sources: bool = False
) -> dict[str, Any]:
    """Validate the metadata-only scope, full payload hash, and source inventory."""

    snapshot = deepcopy(report)
    observed = snapshot.pop("audit_payload_sha256", None)
    sources = snapshot.get("source_code_sha256s")
    if (
        snapshot.get("local_corpus_audit_version") != LOCAL_CORPUS_AUDIT_VERSION
        or snapshot.get("contains_article_text") is not False
        or snapshot.get("contains_question_text") is not False
        or snapshot.get("contains_titles") is not False
        or snapshot.get("contains_per_question_labels") is not False
        or snapshot.get("network_or_api_calls") != 0
        or not isinstance(observed, str)
        or observed != hash_canonical(snapshot)
        or not isinstance(sources, dict)
        or set(sources) != set(_SOURCE_CODE_PATHS)
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in sources.values()
        )
    ):
        raise LocalCorpusAuditError("local_corpus_audit_integrity_invalid")
    if require_current_sources and sources != _source_code_hashes():
        raise LocalCorpusAuditError("local_corpus_audit_source_lineage_stale")
    return report


def write_local_corpus_audit(path: Path, report: dict[str, Any], *, force: bool = False) -> None:
    validated = validate_local_corpus_audit(report, require_current_sources=True)
    atomic_write_json(path, validated, force=force)


__all__ = [
    "LocalCorpusAuditError",
    "build_local_corpus_audit",
    "validate_local_corpus_audit",
    "write_local_corpus_audit",
]
