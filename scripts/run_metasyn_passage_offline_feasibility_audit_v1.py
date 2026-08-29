#!/usr/bin/env python3
"""Freeze or externally validate the label-blind packet feasibility audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.metasyn_passage_offline_feasibility_audit_v1 import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V2_WORKSPACE,
    MetaSynPassageOfflineFeasibilityAuditV1,
    freeze_metasyn_passage_offline_feasibility_audit_v1,
    validate_metasyn_passage_offline_feasibility_audit_v1,
    write_metasyn_passage_offline_feasibility_audit_v1,
)


def _rooted(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "validate"))
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--v2-workspace", type=Path, default=DEFAULT_V2_WORKSPACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.repository_root.resolve(strict=True)
    v2_workspace = _rooted(args.v2_workspace, root)
    output = _rooted(args.output, root)
    if args.command == "freeze":
        audit = freeze_metasyn_passage_offline_feasibility_audit_v1(
            repository_root=root, v2_workspace=v2_workspace
        )
        write_metasyn_passage_offline_feasibility_audit_v1(audit=audit, output_path=output)
    else:
        audit = MetaSynPassageOfflineFeasibilityAuditV1.model_validate(
            json.loads(output.read_text(encoding="utf-8"))
        )
        audit = validate_metasyn_passage_offline_feasibility_audit_v1(
            audit=audit,
            repository_root=root,
            v2_workspace=v2_workspace,
            external_replay=True,
        )
    print(
        json.dumps(
            {
                "status": audit.status,
                "audit_sha256": audit.audit_sha256,
                "pipeline_sha256": audit.pipeline_sha256,
                "audited_unattempted_candidate_count": (audit.audited_unattempted_candidate_count),
                "reachable_candidate_count": audit.reachable_candidate_count,
                "provider_calls_made": audit.provider_calls_made,
                "claim_release_authority": audit.claim_release_authority,
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
