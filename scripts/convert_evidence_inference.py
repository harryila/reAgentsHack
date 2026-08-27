#!/usr/bin/env python3
"""Convert official Evidence Inference 2.0 data to GEPA benchmark JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.evidence_inference import (
    convert_evidence_inference,
    write_evidence_inference_metadata_summary,
)
from literature_multiverse.paths import PATHS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PATHS.data_dir / "cache" / "evidence-inference-2.0",
        help="extracted official v2.0 archive root",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-officially-flagged",
        action="store_true",
        help="retain prompts the upstream README marks incorrect/questionable/malformed",
    )
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        default=None,
        help="deterministic smoke-test cap within each official split",
    )
    parser.add_argument(
        "--metadata-summary",
        type=Path,
        default=None,
        help="optional trackable metadata-only summary (identifiers, hashes, counts; no text)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = convert_evidence_inference(
        args.dataset_root,
        args.output_dir,
        include_flagged=args.include_officially_flagged,
        max_examples_per_split=args.max_examples_per_split,
    )
    metadata_summary = None
    if args.metadata_summary is not None:
        metadata_summary = write_evidence_inference_metadata_summary(
            result.manifest_path,
            result.report_path,
            args.metadata_summary,
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": result.manifest_path.as_posix(),
                "report": result.report_path.as_posix(),
                "metadata_summary": (
                    metadata_summary.as_posix() if metadata_summary is not None else None
                ),
                "rows": result.rows,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
