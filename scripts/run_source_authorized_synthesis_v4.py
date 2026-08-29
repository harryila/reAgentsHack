#!/usr/bin/env python3
"""Replay V2 source authorizations and run the V4 synthesis-yield join."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.cohort_reconciliation import NativeCohortReconciliationReceipt
from literature_multiverse.lineage import atomic_write_json
from literature_multiverse.native_grounding import TypedEvidenceGroundingPackage
from literature_multiverse.source_authorized_synthesis_v4 import run_source_authorized_synthesis_v4
from literature_multiverse.synthesis_unit_authorization import SynthesisUnitAuthorizationReceipt
from literature_multiverse.typed_extraction import TypedEvidenceCorpus


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"source_authorized_v4_json_not_object:{path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--grounding-package", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, action="append", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    request = _object(args.request)
    if set(request) != {"request_version", "requested_estimate_ids", "authorized_receipt_sha256s"}:
        raise ValueError("source_authorized_v4_request_keys_mismatch")
    if request["request_version"] != "source-authorized-synthesis-v4-request-v1":
        raise ValueError("source_authorized_v4_request_version_unsupported")
    report = run_source_authorized_synthesis_v4(
        corpus=TypedEvidenceCorpus.model_validate(_object(args.corpus)),
        grounding_package=TypedEvidenceGroundingPackage.model_validate(
            _object(args.grounding_package)
        ),
        reconciliation=NativeCohortReconciliationReceipt.model_validate(
            _object(args.reconciliation)
        ),
        receipts=[
            SynthesisUnitAuthorizationReceipt.model_validate(_object(path))
            for path in args.authorization
        ],
        requested_estimate_ids=request["requested_estimate_ids"],
        authorized_receipt_sha256s=request["authorized_receipt_sha256s"],
        repository_root=args.repository_root,
        pipeline_root=args.pipeline_root,
    )
    atomic_write_json(args.output, report, force=args.force)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "report_sha256": report.report_sha256,
                "synthesized_units": sum(row.status == "synthesized" for row in report.units),
                "abstained_units": sum(row.status != "synthesized" for row in report.units),
                "verifier_replay_status": report.verifier_replay_status,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
