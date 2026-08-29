#!/usr/bin/env python3
"""Run the staged, provider-free MetaSyn screening reranking study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.lineage import sha256_file
from literature_multiverse.metasyn_screening_study import (
    BOOTSTRAP_REPLICATES,
    evaluate_frozen_winner,
    fit_and_freeze_winner,
    prepare_label_blind_features,
    run_screening_study,
    validate_public_summary,
)
from literature_multiverse.paths import PATHS


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
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
        "--review-cache-dir", type=Path, default=Path("data/cache/metasyn")
    )
    parser.add_argument(
        "--retrieval-work-dir",
        type=Path,
        default=Path("data/cache/metasyn/retrieval-study-v1"),
        help="Ignored, identifier-bearing output of the frozen retrieval study.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data/cache/metasyn/screening-study-v1"),
        help="Ignored directory for pair features, labels, rankings, and receipts.",
    )
    parser.add_argument(
        "--public-summary",
        type=Path,
        default=Path("artifacts/diagnostics/metasyn-screening-study-v1.json"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("prepare", "fit", "evaluate", "run", "validate"):
        stage_parser = subparsers.add_parser(stage)
        _add_common_paths(stage_parser)
        if stage in {"prepare", "fit", "evaluate", "run"}:
            stage_parser.add_argument("--force", action="store_true")
        if stage in {"evaluate", "run"}:
            stage_parser.add_argument(
                "--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES
            )
    return parser


def _rooted(path: Path, *, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    root = PATHS.root
    return {
        "benchmark_manifest_path": _rooted(args.benchmark_manifest, root=root),
        "corpus_manifest_path": _rooted(args.corpus_manifest, root=root),
        "repository_root": root,
        "review_cache_dir": _rooted(args.review_cache_dir, root=root),
        "retrieval_work_dir": _rooted(args.retrieval_work_dir, root=root),
        "work_dir": _rooted(args.work_dir, root=root),
        "public_summary_path": _rooted(args.public_summary, root=root),
    }


def _print_result(*, stage: str, payload: dict[str, Any], path: Path | None = None) -> None:
    result: dict[str, Any] = {
        "stage": stage,
        "status": payload.get("status", "complete"),
    }
    if path is not None:
        result["artifact"] = path.relative_to(PATHS.root).as_posix()
        result["artifact_sha256"] = sha256_file(path)
    selected = payload.get("selected_candidate") or payload.get("protocol", {}).get(
        "selected_candidate"
    )
    if selected is not None:
        result["selected_candidate"] = selected
    print(json.dumps(result, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = _paths(args)
    work_dir = paths["work_dir"]
    public_summary = paths["public_summary_path"]
    if args.stage == "prepare":
        payload = prepare_label_blind_features(
            benchmark_manifest_path=paths["benchmark_manifest_path"],
            corpus_manifest_path=paths["corpus_manifest_path"],
            repository_root=paths["repository_root"],
            retrieval_work_dir=paths["retrieval_work_dir"],
            work_dir=work_dir,
            force=args.force,
        )
        _print_result(stage=args.stage, payload=payload, path=work_dir / "prepare_receipt.json")
        return 0
    if args.stage == "fit":
        payload = fit_and_freeze_winner(
            benchmark_manifest_path=paths["benchmark_manifest_path"],
            review_cache_dir=paths["review_cache_dir"],
            work_dir=work_dir,
            force=args.force,
        )
        _print_result(stage=args.stage, payload=payload, path=work_dir / "fit_receipt.json")
        return 0
    if args.stage == "evaluate":
        payload = evaluate_frozen_winner(
            benchmark_manifest_path=paths["benchmark_manifest_path"],
            review_cache_dir=paths["review_cache_dir"],
            repository_root=paths["repository_root"],
            work_dir=work_dir,
            public_summary_path=public_summary,
            force=args.force,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        _print_result(stage=args.stage, payload=payload, path=public_summary)
        return 0
    if args.stage == "run":
        payload = run_screening_study(
            benchmark_manifest_path=paths["benchmark_manifest_path"],
            corpus_manifest_path=paths["corpus_manifest_path"],
            repository_root=paths["repository_root"],
            review_cache_dir=paths["review_cache_dir"],
            retrieval_work_dir=paths["retrieval_work_dir"],
            work_dir=work_dir,
            public_summary_path=public_summary,
            force=args.force,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        _print_result(stage=args.stage, payload=payload, path=public_summary)
        return 0
    payload = validate_public_summary(
        repository_root=paths["repository_root"],
        work_dir=work_dir,
        public_summary_path=public_summary,
    )
    _print_result(stage=args.stage, payload=payload, path=public_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
