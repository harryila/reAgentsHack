#!/usr/bin/env python3
"""Freeze a conservative source-backed synthesis-unit authorization receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.cohort_reconciliation import (
    NativeCohortReconciliationReceipt,
)
from literature_multiverse.lineage import atomic_write_json
from literature_multiverse.synthesis_unit_authorization import (
    SourceIdentityAssertion,
    authorize_synthesis_unit,
)
from literature_multiverse.typed_extraction import TypedEvidenceCorpus


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"synthesis_authorization_json_not_object:{path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        required=True,
        help="Root against which every repository-relative source artifact is rehashed.",
    )
    parser.add_argument(
        "--request",
        type=Path,
        required=True,
        help="JSON object containing sorted estimate_ids and source assertions.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = _object(args.request)
    if set(request) != {"request_version", "estimate_ids", "assertions"}:
        raise ValueError("synthesis_authorization_request_keys_mismatch")
    if request["request_version"] != "source-backed-synthesis-request-v1":
        raise ValueError("synthesis_authorization_request_version_unsupported")
    receipt = authorize_synthesis_unit(
        corpus=TypedEvidenceCorpus.model_validate(_object(args.corpus)),
        reconciliation=NativeCohortReconciliationReceipt.model_validate(
            _object(args.reconciliation)
        ),
        estimate_ids=request["estimate_ids"],
        assertions=[SourceIdentityAssertion.model_validate(row) for row in request["assertions"]],
        repository_root=args.repository_root,
    )
    atomic_write_json(args.output, receipt, force=args.force)
    print(
        json.dumps(
            {
                "authorizes_synthesis_input": receipt.authorizes_synthesis_input,
                "authorization_basis": receipt.authorization_basis,
                "output": args.output.as_posix(),
                "receipt_sha256": receipt.receipt_sha256,
                "unresolved_overlap_pairs": receipt.unresolved_overlap_pairs,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
