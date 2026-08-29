#!/usr/bin/env python3
"""Print a private, label-blind projection-v2 diagnostic as canonical JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.lineage import canonical_json_bytes
from literature_multiverse.metasyn_projection_v2 import (
    DEFAULT_FAILURE_STRATIFIED_ROWS,
    load_and_diagnose_execution_bundle_projection_v2,
)


def _parse_rows(value: str) -> tuple[int, ...]:
    try:
        rows = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rows_must_be_comma_separated_integers") from exc
    if not rows or list(rows) != sorted(set(rows)) or any(item < 0 for item in rows):
        raise argparse.ArgumentTypeError("rows_must_be_nonnegative_sorted_unique")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate and passage-anchor frozen MetaSyn v1 source projections. "
            "The report has no accuracy, synthesis, or release authority."
        )
    )
    parser.add_argument(
        "--execution-bundle",
        type=Path,
        required=True,
        help="Private frozen hosted execution-bundle.json",
    )
    parser.add_argument(
        "--rows",
        type=_parse_rows,
        default=DEFAULT_FAILURE_STRATIFIED_ROWS,
        help="Sorted comma-separated row ordinals (default: 9,10,18,27,29)",
    )
    args = parser.parse_args()
    report = load_and_diagnose_execution_bundle_projection_v2(
        args.execution_bundle,
        row_ordinals=args.rows,
    )
    # stdout only: the diagnostic command does not publish or persist an artifact.
    print(
        json.dumps(
            json.loads(canonical_json_bytes(report).decode("utf-8")),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
