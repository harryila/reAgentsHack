#!/usr/bin/env python3
"""Inspect or freeze cross-publication study/cohort reconciliation for a typed corpus."""

from __future__ import annotations

import argparse
import json
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from literature_multiverse.cohort_reconciliation import (
    ReviewerIdentityGroup,
    freeze_reviewer_cohort_reconciliation_artifact,
    reconcile_native_cohorts,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical
from literature_multiverse.typed_extraction import TypedEvidenceCorpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--reviewer-partition",
        type=Path,
        help=(
            "Completed partition JSON. Omit to run strong-ID candidate discovery and "
            "write a reviewer worksheet template."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"reconciliation_json_unreadable:{path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"reconciliation_json_must_be_object:{path}")
    return payload


def _completed_at(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("reviewer_partition_completed_at_must_be_string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reviewer_partition_completed_at_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("reviewer_partition_completed_at_requires_timezone")
    return parsed


def _groups(value: object, *, field: str) -> list[ReviewerIdentityGroup]:
    if not isinstance(value, list):
        raise ValueError(f"reviewer_partition_{field}_must_be_list")
    return [ReviewerIdentityGroup.model_validate(item) for item in value]


def _normalize_identifier(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().casefold().split())


def _reviewer_artifact(*, corpus: TypedEvidenceCorpus, partition: dict[str, Any]):
    allowed = {
        "partition_version",
        "input_corpus_sha256",
        "input_graph_sha256",
        "reviewer_identity_sha256",
        "review_protocol_sha256",
        "completed_at",
        "all_studies_and_cohorts_reviewed",
        "study_groups",
        "cohort_groups",
    }
    extras = sorted(set(partition) - allowed)
    missing = sorted(allowed - set(partition))
    if extras or missing:
        raise ValueError(f"reviewer_partition_keys_mismatch:missing={missing}:extra={extras}")
    if partition["partition_version"] != "reviewer-cohort-partition-input-v1":
        raise ValueError("reviewer_partition_version_unsupported")
    if partition["input_corpus_sha256"] != corpus.corpus_sha256:
        raise ValueError("reviewer_partition_corpus_hash_mismatch")
    if partition["input_graph_sha256"] != hash_canonical(corpus.graph):
        raise ValueError("reviewer_partition_graph_hash_mismatch")
    if partition["all_studies_and_cohorts_reviewed"] is not True:
        raise ValueError("reviewer_partition_requires_complete_attestation")
    return freeze_reviewer_cohort_reconciliation_artifact(
        corpus=corpus,
        reviewer_identity_sha256=str(partition["reviewer_identity_sha256"]),
        review_protocol_sha256=str(partition["review_protocol_sha256"]),
        completed_at=_completed_at(partition["completed_at"]),
        study_groups=_groups(partition["study_groups"], field="study_groups"),
        cohort_groups=_groups(partition["cohort_groups"], field="cohort_groups"),
    )


def _worksheet(corpus: TypedEvidenceCorpus) -> dict[str, Any]:
    graph = corpus.graph
    study_publications = {study.study_id: study.publication_ids for study in graph.studies}
    cohort_study = {cohort.cohort_id: cohort.study_id for cohort in graph.cohorts}
    return {
        "partition_version": "reviewer-cohort-partition-input-v1",
        "input_corpus_sha256": corpus.corpus_sha256,
        "input_graph_sha256": hash_canonical(graph),
        "reviewer_identity_sha256": "REPLACE_WITH_SHA256_OF_PSEUDONYMOUS_REVIEWER_ID",
        "review_protocol_sha256": "REPLACE_WITH_SHA256_OF_FROZEN_REVIEW_PROTOCOL",
        "completed_at": "REPLACE_WITH_TIMEZONE_AWARE_ISO8601",
        "all_studies_and_cohorts_reviewed": False,
        "study_inventory": [
            {
                "study_id": study.study_id,
                "publication_ids": study.publication_ids,
                "normalized_registration_ids": sorted(
                    {_normalize_identifier(identifier) for identifier in study.registration_ids}
                ),
            }
            for study in sorted(graph.studies, key=lambda item: item.study_id)
        ],
        "cohort_inventory": [
            {
                "cohort_id": cohort.cohort_id,
                "study_id": cohort.study_id,
                "publication_ids": study_publications[cohort.study_id],
                "normalized_registry_ids": sorted(
                    {
                        _normalize_identifier(identifier)
                        for identifier in cohort.identity.registry_ids
                    }
                ),
                "normalized_dataset_ids": sorted(
                    {
                        _normalize_identifier(identifier)
                        for identifier in cohort.identity.dataset_ids
                    }
                ),
            }
            for cohort in sorted(graph.cohorts, key=lambda item: item.cohort_id)
        ],
        "study_groups": [
            {
                "member_ids": [study.study_id],
                "rationale": "REPLACE_WITH_REVIEWER_RATIONALE",
            }
            for study in sorted(graph.studies, key=lambda item: item.study_id)
        ],
        "cohort_groups": [
            {
                "member_ids": [cohort.cohort_id],
                "rationale": "REPLACE_WITH_REVIEWER_RATIONALE",
            }
            for cohort in sorted(graph.cohorts, key=lambda item: item.cohort_id)
        ],
        "instructions": (
            "Partition every listed ID exactly once. Merge only identities the reviewer "
            "has verified; never merge two IDs from one publication. Remove inventory and "
            "instructions fields before submitting the completed partition."
        ),
        "cohort_to_study": cohort_study,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    corpus = TypedEvidenceCorpus.model_validate(_json_object(args.corpus))
    reviewer_artifact = None
    if args.reviewer_partition is not None:
        reviewer_artifact = _reviewer_artifact(
            corpus=corpus,
            partition=_json_object(args.reviewer_partition),
        )
    receipt = reconcile_native_cohorts(
        corpus=corpus,
        reviewer_artifact=reviewer_artifact,
    )
    output_dir: Path = args.output_dir
    atomic_write_json(
        output_dir / "cohort_reconciliation_receipt.json",
        receipt,
        force=args.force,
    )
    assert receipt.reconciled_graph is not None
    atomic_write_json(
        output_dir / "reconciled_evidence_graph.json",
        receipt.reconciled_graph,
        force=args.force,
    )
    if reviewer_artifact is None:
        atomic_write_json(
            output_dir / "reviewer_partition_template.json",
            _worksheet(corpus),
            force=args.force,
        )
    else:
        atomic_write_json(
            output_dir / "reviewer_cohort_reconciliation.json",
            reviewer_artifact,
            force=args.force,
        )
    print(
        json.dumps(
            {
                "status": receipt.status.value,
                "receipt_sha256": receipt.receipt_sha256,
                "cross_publication_identity_assurance_complete": (
                    receipt.cross_publication_identity_assurance_complete
                ),
                "candidate_components": len(receipt.candidates),
                "issues": len(receipt.issues),
                "merged_study_groups": receipt.merged_study_groups,
                "merged_cohort_groups": receipt.merged_cohort_groups,
                "output": (output_dir / "cohort_reconciliation_receipt.json").as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
