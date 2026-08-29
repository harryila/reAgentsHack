#!/usr/bin/env python3
"""Validate every registered headline result using public checkout files only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.public_artifacts import validate_public_result_registry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = validate_public_result_registry(repository_root=args.repository_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
