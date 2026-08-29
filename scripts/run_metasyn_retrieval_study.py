#!/usr/bin/env python3
"""Run the staged, provider-free MetaSyn retrieval selection study."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from literature_multiverse.lineage import sha256_file
from literature_multiverse.metasyn_retrieval_study import (
    BOOTSTRAP_REPLICATES,
    run_retrieval_study,
)
from literature_multiverse.paths import PATHS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        default=Path("artifacts/paper/metasyn-benchmark/manifest.json"),
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=Path("configs/benchmarks/metasyn-corpus-c8fa07d.json"),
    )
    parser.add_argument(
        "--review-cache-dir",
        type=Path,
        default=Path("data/cache/metasyn"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/cache/metasyn/retrieval-study-v1"),
        help="Ignored local directory for identifier-bearing frozen predictions.",
    )
    parser.add_argument(
        "--public-summary",
        type=Path,
        default=Path("artifacts/diagnostics/metasyn-retrieval-study-v1.json"),
        help="Tracked aggregate-only, self-hashed result.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Deterministically recompute the fixed study; no configuration is tunable.",
    )
    return parser


def _rooted(path: Path, *, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = PATHS.root
    start = time.perf_counter()
    summary_path = _rooted(args.public_summary, root=root)
    summary = run_retrieval_study(
        benchmark_manifest_path=_rooted(args.benchmark_manifest, root=root),
        corpus_manifest_path=_rooted(args.corpus_manifest, root=root),
        repository_root=root,
        review_cache_dir=_rooted(args.review_cache_dir, root=root),
        work_dir=_rooted(args.work_dir, root=root),
        public_summary_path=summary_path,
        force=args.force,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    peak_rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mib = peak_rss_raw / (1024 * 1024) if sys.platform == "darwin" else peak_rss_raw / 1024
    print(
        json.dumps(
            {
                "status": summary["status"],
                "selected_candidate": summary["selection_protocol"]["selected_candidate"],
                "summary": summary_path.relative_to(root).as_posix(),
                "summary_sha256": sha256_file(summary_path),
                # Execution time is intentionally console-only and excluded from the
                # deterministic scientific artifact.
                "wall_seconds_console_only": round(time.perf_counter() - start, 3),
                "peak_rss_mib_console_only": round(peak_rss_mib, 3),
                "peak_rss_raw_console_only": peak_rss_raw,
                "peak_rss_raw_unit": "bytes" if sys.platform == "darwin" else "kibibytes",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
