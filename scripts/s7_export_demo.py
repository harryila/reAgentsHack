#!/usr/bin/env python3
"""Stage, verify, and atomically promote one frozen demo release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.config import load_config_for_question
from literature_multiverse.export import ExportError, ReleaseSource, export_demo
from literature_multiverse.paths import REPO_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--candidate", choices=("v1", "scaled"), default="v1")
    parser.add_argument("--fallback", choices=("frozen-v1",))
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--extracted-dir", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--allow-dirty-demo", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.candidate == "scaled" and args.fallback != "frozen-v1":
        raise SystemExit("--candidate scaled requires --fallback frozen-v1")
    if args.candidate == "v1" and args.fallback is not None:
        raise SystemExit("--fallback is valid only for --candidate scaled")
    config = load_config_for_question(args.question, root=REPO_ROOT, require_locked=True)
    source = ReleaseSource.from_repository(
        REPO_ROOT,
        args.question,
        corpus_role=args.candidate,
        processed_dir=args.processed_dir,
        analysis_dir=args.analysis_dir,
        extracted_dir=args.extracted_dir,
    )
    try:
        manifest = export_demo(
            source,
            config,
            destination=args.destination,
            explicit_fixture=args.fixture,
            allow_dirty_demo=args.allow_dirty_demo,
            force=args.force,
        )
    except ExportError as exc:
        raise SystemExit(f"s7 export failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "complete",
                "question_id": manifest["question_id"],
                "variant": manifest["narrative_variant"],
                "disposition": manifest["release_selection"]["disposition"],
                "created_at": manifest["created_at"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

