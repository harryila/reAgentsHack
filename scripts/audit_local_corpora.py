#!/usr/bin/env python3
"""Freeze a metadata-only audit of local closed-corpus feasibility and leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.local_corpus_audit import (
    build_local_corpus_audit,
    write_local_corpus_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/paper/closed-corpus-local-audit.json"),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_local_corpus_audit(
        metasyn_manifest_path=Path("artifacts/paper/metasyn-benchmark/manifest.json"),
        metasyn_cache_dir=Path("data/cache/metasyn"),
        metasyn_predictions_path=Path(
            "artifacts/paper/metasyn-fixed-positive-test/predictions.jsonl"
        ),
        metasyn_evaluation_path=Path("artifacts/paper/metasyn-fixed-positive-test/evaluation.json"),
        evidence_inference_root=Path("data/cache/evidence-inference-2.0"),
        evidence_inference_manifest_path=Path("data/cache/evidence-inference-gepa/manifest.json"),
        evidence_inference_evaluation_summary_path=Path(
            "artifacts/paper/evidence-inference-gepa-pilot-summary.json"
        ),
        antiox_papers_path=Path("data/processed/antiox-training/papers.parquet"),
        antiox_findings_path=Path("data/processed/antiox-training/findings.parquet"),
        antiox_source_lines_path=Path("data/raw/map/antiox-training/source_lines.json"),
        antiox_packet_manifest_path=Path("data/cache/human-audit/antiox-training-60/manifest.json"),
    )
    write_local_corpus_audit(args.output, report, force=args.force)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "baseline": report["cached_local_baseline"]["name"],
                "baseline_status": report["cached_local_baseline"]["status"],
                "blockers": [row["code"] for row in report["external_blockers"]],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
