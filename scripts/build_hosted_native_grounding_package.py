#!/usr/bin/env python3
"""Validate a future exact-once hosted run and optionally build native package v4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.cohort_reconciliation import (
    ReviewerCohortReconciliationArtifact,
)
from literature_multiverse.hosted_native_extraction_contract import (
    HostedNativeExtractionRunV1,
)
from literature_multiverse.hosted_native_grounding_bridge import (
    build_hosted_native_grounding_package_v1,
    validate_hosted_native_extraction_run_v1,
)
from literature_multiverse.lineage import atomic_write_json, atomic_write_jsonl


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate",
        help="Externally replay a hosted run without writing a package.",
    )
    validate.add_argument("--repository-root", type=Path, default=Path("."))
    validate.add_argument("--run", type=Path, required=True)

    build = subparsers.add_parser(
        "build",
        help="Externally replay the run, build package v4, and replay the package.",
    )
    build.add_argument("--repository-root", type=Path, default=Path("."))
    build.add_argument("--run", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--reviewer-reconciliation", type=Path)
    build.add_argument("--force", action="store_true")
    return parser


def _object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("hosted_native_cli_input_file_invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("hosted_native_cli_input_json_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("hosted_native_cli_input_not_object")
    return value


def _status(run: HostedNativeExtractionRunV1) -> dict[str, Any]:
    return {
        "run_version": run.run_version,
        "status": run.status,
        "run_sha256": run.run_sha256,
        "pipeline_fingerprint_sha256": run.pipeline_fingerprint_sha256,
        "provider_identity_sha256": run.provider_identity_sha256,
        "source_manifest_records": run.source_manifest_records,
        "terminal_call_count": len(run.calls),
        "completed_extraction_count": run.completed_extraction_count,
        "failed_or_ambiguous_count": run.failed_or_ambiguous_count,
        "diagnostic_or_fixture": run.diagnostic_or_fixture,
        "official_test_labels_opened": run.official_test_labels_opened,
        "v4_source_provenance_bridge_eligible": (run.v4_source_provenance_bridge_eligible),
        "claim_release_authority": run.claim_release_authority,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    raw = _object(args.run)
    if args.command == "validate":
        run, _ = validate_hosted_native_extraction_run_v1(
            run=raw,
            repository_root=args.repository_root,
        )
        print(json.dumps(_status(run), ensure_ascii=False, sort_keys=True))
        return 0

    reviewer = (
        ReviewerCohortReconciliationArtifact.model_validate(_object(args.reviewer_reconciliation))
        if args.reviewer_reconciliation is not None
        else None
    )
    result = build_hosted_native_grounding_package_v1(
        run=raw,
        repository_root=args.repository_root,
        reviewer_reconciliation=reviewer,
    )
    output: Path = args.output_dir
    atomic_write_json(
        output / "hosted_native_extraction_run.json",
        result.run,
        force=args.force,
    )
    atomic_write_json(
        output / "pipeline_verification.json",
        result.pipeline_verification,
        force=args.force,
    )
    atomic_write_json(
        output / "native_extraction_context.json",
        result.extraction_context,
        force=args.force,
    )
    atomic_write_jsonl(
        output / "publication_fragments.jsonl",
        result.fragments,
        force=args.force,
    )
    atomic_write_jsonl(
        output / "grounding_receipts.jsonl",
        result.grounding_receipts,
        force=args.force,
    )
    atomic_write_json(
        output / "typed_evidence_corpus.json",
        result.corpus,
        force=args.force,
    )
    atomic_write_json(
        output / "evidence_graph.json",
        result.corpus.graph,
        force=args.force,
    )
    reconciliation = result.package.cohort_reconciliation
    assert reconciliation is not None
    assert reconciliation.reconciled_graph is not None
    atomic_write_json(
        output / "reconciled_evidence_graph.json",
        reconciliation.reconciled_graph,
        force=args.force,
    )
    atomic_write_json(
        output / "typed_evidence_grounding_package.json",
        result.package,
        force=args.force,
    )
    atomic_write_json(
        output / "hosted_native_grounding_bridge_receipt.json",
        result.receipt,
        force=args.force,
    )
    print(
        json.dumps(
            {
                **_status(result.run),
                "grounding_package_version": result.package.package_version,
                "grounding_package_sha256": result.package.package_sha256,
                "bridge_receipt_sha256": result.receipt.receipt_sha256,
                "v4_source_provenance_input_eligible": (
                    result.receipt.v4_source_provenance_input_eligible
                ),
                "remaining_release_gates_external": (
                    result.receipt.remaining_release_gates_external
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
