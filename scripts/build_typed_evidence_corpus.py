#!/usr/bin/env python3
"""Assemble hash-bound native publication fragments into a typed evidence corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.cohort_reconciliation import (
    ReviewerCohortReconciliationArtifact,
)
from literature_multiverse.lineage import atomic_write_json
from literature_multiverse.native_extraction import NativeSourceManifest
from literature_multiverse.native_grounding import (
    NativeExtractionExecutionContext,
    NativeGroundingReceipt,
    freeze_typed_evidence_grounding_package,
)
from literature_multiverse.typed_extraction import (
    PublicationEvidenceFragment,
    assemble_typed_evidence_corpus,
    publication_fragment_json_schema,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fragments",
        type=Path,
        nargs="+",
        required=True,
        help="JSON/JSONL publication fragment artifact(s)",
    )
    parser.add_argument(
        "--grounding-receipts",
        type=Path,
        nargs="+",
        help="JSON/JSONL native grounding receipt artifact(s)",
    )
    parser.add_argument(
        "--reviewer-reconciliation",
        type=Path,
        help=(
            "Optional complete reviewer-cohort-reconciliation-v1 JSON artifact; "
            "without it, only exact strong-ID candidates are reconciled and "
            "cross-publication assurance remains incomplete"
        ),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help=(
            "Complete native-source-manifest-v1. Supplying this together with "
            "--corpus-cutoff creates a membership-bound analysis package."
        ),
    )
    parser.add_argument(
        "--corpus-cutoff",
        help="Exact frozen corpus-cutoff identifier; requires --source-manifest",
    )
    parser.add_argument(
        "--extraction-context",
        type=Path,
        help=(
            "Exact native-extraction-execution-context-v1 bound by every v3 fragment. "
            "Required with v3 fragments to create the release-capable v4 package."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def _records(path: Path) -> list[Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"typed_fragment_file_unreadable:{path}") from exc
    if path.suffix.casefold() == ".jsonl":
        records: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"typed_fragment_jsonl_invalid:{path}:line={line_number}") from exc
        return records
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"typed_fragment_json_invalid:{path}") from exc
    if isinstance(payload, dict) and "fragments" in payload:
        payload = payload["fragments"]
    if isinstance(payload, dict):
        return [payload]
    if not isinstance(payload, list):
        raise ValueError(f"typed_fragment_json_requires_object_or_array:{path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    fragments = [
        PublicationEvidenceFragment.model_validate(record)
        for path in args.fragments
        for record in _records(path)
    ]
    corpus = assemble_typed_evidence_corpus(fragments)
    receipt_records = [
        record for path in args.grounding_receipts or [] for record in _records(path)
    ]
    grounding_receipts = [
        NativeGroundingReceipt.model_validate(record) for record in receipt_records
    ]
    if corpus.estimable_publication_ids and not grounding_receipts:
        raise ValueError("estimable_typed_corpus_requires_grounding_receipts")
    if (args.source_manifest is None) != (args.corpus_cutoff is None):
        raise ValueError("source_manifest_and_corpus_cutoff_must_be_supplied_together")
    source_manifest = None
    if args.source_manifest is not None:
        source_records = _records(args.source_manifest)
        if len(source_records) != 1:
            raise ValueError("source_manifest_requires_one_json_object")
        source_manifest = NativeSourceManifest.model_validate(source_records[0])
    extraction_context = None
    if args.extraction_context is not None:
        context_records = _records(args.extraction_context)
        if len(context_records) != 1:
            raise ValueError("extraction_context_requires_one_json_object")
        extraction_context = NativeExtractionExecutionContext.model_validate(
            context_records[0]
        )
    if corpus.corpus_version == "typed-evidence-corpus-v3" and extraction_context is None:
        raise ValueError("typed_evidence_corpus_v3_requires_extraction_context")
    if corpus.corpus_version == "typed-evidence-corpus-v2" and extraction_context is not None:
        raise ValueError("typed_evidence_corpus_v2_forbids_extraction_context")
    reviewer_reconciliation = None
    if args.reviewer_reconciliation is not None:
        reviewer_records = _records(args.reviewer_reconciliation)
        if len(reviewer_records) != 1:
            raise ValueError("reviewer_reconciliation_requires_one_json_object")
        reviewer_reconciliation = ReviewerCohortReconciliationArtifact.model_validate(
            reviewer_records[0]
        )
    package = freeze_typed_evidence_grounding_package(
        corpus=corpus,
        grounding_receipts=grounding_receipts,
        reviewer_reconciliation=reviewer_reconciliation,
        source_manifest=source_manifest,
        corpus_cutoff=args.corpus_cutoff,
        extraction_context=extraction_context,
    )
    output_dir: Path = args.output_dir
    atomic_write_json(
        output_dir / "publication_evidence_fragment.schema.json",
        publication_fragment_json_schema(),
        force=args.force,
    )
    atomic_write_json(output_dir / "typed_evidence_corpus.json", corpus, force=args.force)
    atomic_write_json(
        output_dir / "typed_evidence_grounding_package.json",
        package,
        force=args.force,
    )
    atomic_write_json(output_dir / "evidence_graph.json", corpus.graph, force=args.force)
    reconciliation = package.cohort_reconciliation
    assert reconciliation is not None
    assert reconciliation.reconciled_graph is not None
    atomic_write_json(
        output_dir / "reconciled_evidence_graph.json",
        reconciliation.reconciled_graph,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "corpus_sha256": corpus.corpus_sha256,
                "grounding_package_sha256": package.package_sha256,
                "grounding_receipts": len(package.grounding_receipts),
                "package_version": package.package_version,
                "source_manifest_sha256": package.source_manifest_sha256,
                "corpus_cutoff": package.corpus_cutoff,
                "cohort_reconciliation_status": reconciliation.status.value,
                "cohort_reconciliation_receipt_sha256": reconciliation.receipt_sha256,
                "cross_publication_identity_assurance_complete": (
                    reconciliation.cross_publication_identity_assurance_complete
                ),
                "merged_study_groups": reconciliation.merged_study_groups,
                "merged_cohort_groups": reconciliation.merged_cohort_groups,
                "estimable_publications": len(corpus.estimable_publication_ids),
                "non_estimable_publications": len(corpus.non_estimable_publication_ids),
                "output": (output_dir / "typed_evidence_grounding_package.json").as_posix(),
                "question_id": corpus.question_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
