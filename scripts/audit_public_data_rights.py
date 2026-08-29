#!/usr/bin/env python3
"""Audit tracked research-data rights declarations without printing corpus content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.lineage import OutputExistsError, atomic_write_json
from literature_multiverse.public_data_rights import (
    PublicDataRightsAuditError,
    audit_public_data_rights,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Policy JSON (default: configs/public-data-rights-v1.json under the repository)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the aggregate, self-hashed JSON report",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing --output report atomically",
    )
    parser.add_argument(
        "--require-release-ready",
        action="store_true",
        help="Exit 2 when any declared collection still lacks established redistribution rights",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve()
    policy = args.policy
    if policy is not None and not policy.is_absolute():
        policy = root / policy
    try:
        report = audit_public_data_rights(
            repository_root=root,
            policy_path=policy,
        )
    except PublicDataRightsAuditError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        return 2
    if args.output is not None:
        try:
            atomic_write_json(args.output, report, force=args.force)
        except OutputExistsError as exc:
            print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
            return 2
    print(json.dumps(report, sort_keys=True))
    if not report["policy_complete"]:
        return 2
    if args.require_release_ready and not report["release_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
