#!/usr/bin/env python3
"""Build or validate the fail-closed cross-artifact evidence-boundary ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.evidence_boundary_ledger_v1 import (
    EvidenceBoundaryLedgerV1,
    build_evidence_boundary_ledger,
    validate_evidence_boundary_ledger,
)
from literature_multiverse.lineage import atomic_write_json, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help=(
            "replay all registered aggregate/mechanics inputs and atomically write a "
            "new ledger; an existing destination is never overwritten"
        ),
    )
    build.add_argument("--repository-root", type=Path, default=Path("."))
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--v3-pre-call-blocker-plan",
        type=Path,
        help=(
            "optional repository-relative externally replayable v3 plan containing "
            "a zero-provider-call pre-call blocker"
        ),
    )

    validate = subparsers.add_parser(
        "validate", help="validate a ledger and optionally replay current registered inputs"
    )
    validate.add_argument("--repository-root", type=Path, default=Path("."))
    validate.add_argument("--ledger", type=Path, required=True)
    validate.add_argument(
        "--v3-pre-call-blocker-plan",
        type=Path,
        help="same optional plan path used when the ledger was built",
    )
    validate.add_argument(
        "--replay-current",
        action="store_true",
        help="rebuild from registered inputs and require an exact ledger match",
    )
    return parser


def _load_ledger(path: Path) -> EvidenceBoundaryLedgerV1:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"evidence_boundary_ledger_unreadable:{path.as_posix()}") from exc
    if not isinstance(value, dict):
        raise ValueError("evidence_boundary_ledger_not_object")
    return validate_evidence_boundary_ledger(value)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve(strict=True)
    if args.command == "build":
        ledger = build_evidence_boundary_ledger(
            repository_root=root,
            v3_pre_call_blocker_plan=args.v3_pre_call_blocker_plan,
        )
        atomic_write_json(args.output, ledger, force=False)
        print(
            json.dumps(
                {
                    "status": ledger.status,
                    "ledger": args.output.as_posix(),
                    "ledger_file_sha256": sha256_file(args.output),
                    "ledger_sha256": ledger.ledger_sha256,
                    "adaptive_policy_effectiveness_authority": (
                        ledger.decision_boundary.adaptive_policy_effectiveness_authority
                    ),
                    "claim_release_authority": (ledger.decision_boundary.claim_release_authority),
                },
                sort_keys=True,
            )
        )
        return 0

    ledger = _load_ledger(args.ledger)
    replayed = False
    if args.replay_current:
        current = build_evidence_boundary_ledger(
            repository_root=root,
            v3_pre_call_blocker_plan=args.v3_pre_call_blocker_plan,
        )
        if current != ledger:
            raise ValueError("evidence_boundary_ledger_current_replay_mismatch")
        replayed = True
    print(
        json.dumps(
            {
                "status": "valid",
                "ledger": args.ledger.as_posix(),
                "ledger_file_sha256": sha256_file(args.ledger),
                "ledger_sha256": ledger.ledger_sha256,
                "current_registered_inputs_replayed": replayed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
