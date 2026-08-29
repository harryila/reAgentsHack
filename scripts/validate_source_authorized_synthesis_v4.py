#!/usr/bin/env python3
"""Replay-validate one source-authorized synthesis V4 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.cohort_reconciliation import NativeCohortReconciliationReceipt
from literature_multiverse.native_grounding import TypedEvidenceGroundingPackage
from literature_multiverse.source_authorized_synthesis_v4 import (
    SourceAuthorizedSynthesisReportV4,
    reverify_source_authorized_synthesis_v4_report,
)
from literature_multiverse.synthesis_unit_authorization import SynthesisUnitAuthorizationReceipt
from literature_multiverse.typed_extraction import TypedEvidenceCorpus


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"source_authorized_v4_json_not_object:{path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "corpus", "grounding-package", "reconciliation", "repository-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, action="append", required=True)
    parser.add_argument("--pipeline-root", type=Path)
    args = parser.parse_args(argv)
    report = reverify_source_authorized_synthesis_v4_report(
        report=SourceAuthorizedSynthesisReportV4.model_validate(_object(args.report)),
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
        repository_root=args.repository_root,
        pipeline_root=args.pipeline_root,
    )
    print(json.dumps({"status": "valid", "report_sha256": report.report_sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
