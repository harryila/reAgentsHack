#!/usr/bin/env python3
"""Replay-validate one source-authorized synthesis v3 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_source_authorized_synthesis_v3 import _object

from literature_multiverse.cohort_reconciliation import NativeCohortReconciliationReceipt
from literature_multiverse.native_grounding import TypedEvidenceGroundingPackage
from literature_multiverse.source_authorized_synthesis_v3 import (
    SourceAuthorizedSynthesisReportV3,
    reverify_source_authorized_synthesis_v3_report,
)
from literature_multiverse.synthesis_unit_authorization import (
    SynthesisUnitAuthorizationReceiptV1,
)
from literature_multiverse.typed_extraction import TypedEvidenceCorpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--grounding-package", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, action="append", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path)
    args = parser.parse_args(argv)
    validated = reverify_source_authorized_synthesis_v3_report(
        report=SourceAuthorizedSynthesisReportV3.model_validate(_object(args.report)),
        corpus=TypedEvidenceCorpus.model_validate(_object(args.corpus)),
        grounding_package=TypedEvidenceGroundingPackage.model_validate(
            _object(args.grounding_package)
        ),
        reconciliation=NativeCohortReconciliationReceipt.model_validate(
            _object(args.reconciliation)
        ),
        receipts=[
            SynthesisUnitAuthorizationReceiptV1.model_validate(_object(path))
            for path in args.authorization
        ],
        repository_root=args.repository_root,
        pipeline_root=args.pipeline_root,
    )
    print(json.dumps({"status": "valid", "report_sha256": validated.report_sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
