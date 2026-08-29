#!/usr/bin/env python3
"""Validate two blinded reviews, isolate conflicts, and emit a public receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.human_review import evaluate_human_review_packet
from literature_multiverse.lineage import atomic_write_json, atomic_write_jsonl


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--reviewer-a",
        type=Path,
        help="completed copy of the immutable reviewer-A template",
    )
    parser.add_argument(
        "--reviewer-b",
        type=Path,
        help="completed copy of the immutable reviewer-B template",
    )
    parser.add_argument("--adjudicator", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--conflicts-output",
        type=Path,
        help="private blank third-adjudicator ledger; required when disagreements exist",
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (args.reviewer_a is None) != (args.reviewer_b is None):
        raise ValueError("--reviewer-a and --reviewer-b must be supplied together")
    summary, conflicts = evaluate_human_review_packet(
        manifest_path=args.manifest,
        reviewer_a_path=args.reviewer_a,
        reviewer_b_path=args.reviewer_b,
        adjudicator_path=args.adjudicator,
    )
    if conflicts:
        if args.conflicts_output is None:
            if summary["status"] == "awaiting_adjudication":
                raise ValueError("--conflicts-output is required for unresolved disagreements")
        else:
            atomic_write_jsonl(args.conflicts_output, conflicts, force=args.force)
    elif args.conflicts_output is not None:
        atomic_write_jsonl(args.conflicts_output, [], force=args.force)
    atomic_write_json(args.output, summary, force=args.force)
    if args.require_complete and summary["status"] != "complete":
        raise ValueError(f"human_review_not_complete:{summary['status']}")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "sample_size": summary["sample_size"],
                "conflicting_items": summary.get("conflicting_items"),
                "invalid_or_incomplete_rows": summary.get("invalid_or_incomplete_rows"),
                "output": args.output.as_posix(),
                "evaluation_sha256": summary["evaluation_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
