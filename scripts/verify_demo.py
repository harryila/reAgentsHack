#!/usr/bin/env python3
"""Verify a frozen demo bundle using local bytes only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.config import load_config_for_question
from literature_multiverse.export import ExportError, verify_demo_bundle
from literature_multiverse.paths import PATHS, REPO_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--demo-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--allow-dirty-demo", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.offline:
        raise SystemExit("verification requires the explicit --offline flag")
    config = load_config_for_question(args.question, root=REPO_ROOT, require_locked=True)
    demo_dir = args.demo_dir or PATHS.demo_dir(args.question)
    try:
        manifest = verify_demo_bundle(
            demo_dir,
            config,
            explicit_fixture=args.fixture,
            allow_dirty_demo=args.allow_dirty_demo,
        )
    except ExportError as exc:
        raise SystemExit(f"demo verification failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": "valid",
                "question_id": manifest["question_id"],
                "variant": manifest["narrative_variant"],
                "disposition": manifest["release_selection"]["disposition"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

