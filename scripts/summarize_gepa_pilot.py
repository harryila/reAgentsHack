#!/usr/bin/env python3
"""Write a reproducible metadata-only summary of an archived GEPA pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.gepa_pilot_summary import (
    write_gepa_pilot_metadata_summary,
)
from literature_multiverse.paths import PATHS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed-prompt", type=Path, required=True)
    parser.add_argument("--failed-run-summary", type=Path, required=True)
    parser.add_argument("--failed-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = write_gepa_pilot_metadata_summary(
        run_dir=args.run_dir,
        manifest_path=args.manifest,
        seed_prompt_path=args.seed_prompt,
        failed_summary_path=args.failed_run_summary,
        failed_run_dir=args.failed_run_dir,
        output_path=args.output,
        repository_root=PATHS.root,
        force=args.force,
    )
    print(json.dumps({"output": output.as_posix(), "status": "complete"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
