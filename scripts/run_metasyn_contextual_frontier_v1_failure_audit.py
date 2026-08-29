#!/usr/bin/env python3
"""Freeze or externally replay the immutable contextual-frontier v1 failure audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.metasyn_contextual_frontier_v1_failure_audit import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V1_WORKSPACE,
    MetaSynContextualFrontierV1FailureAudit,
    freeze_metasyn_contextual_frontier_v1_failure_audit,
    validate_metasyn_contextual_frontier_v1_failure_audit,
    write_metasyn_contextual_frontier_v1_failure_audit,
)


def _rooted(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "validate"))
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--v1-workspace", type=Path, default=DEFAULT_V1_WORKSPACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.repository_root.resolve(strict=True)
    workspace = _rooted(args.v1_workspace, root)
    output = _rooted(args.output, root)
    if args.command == "freeze":
        audit = freeze_metasyn_contextual_frontier_v1_failure_audit(
            repository_root=root,
            v1_workspace=workspace,
        )
        write_metasyn_contextual_frontier_v1_failure_audit(
            audit=audit,
            output_path=output,
        )
    else:
        audit = MetaSynContextualFrontierV1FailureAudit.model_validate(
            json.loads(output.read_text(encoding="utf-8"))
        )
        audit = validate_metasyn_contextual_frontier_v1_failure_audit(
            audit=audit,
            repository_root=root,
            v1_workspace=workspace,
            external_replay=True,
        )
    print(
        json.dumps(
            {
                "status": audit.status,
                "audit_sha256": audit.audit_sha256,
                "terminal_status": audit.terminal_status,
                "structured_completed_response_count": (audit.structured_completed_response_count),
                "typed_graph_completed_response_count": (
                    audit.typed_graph_completed_response_count
                ),
                "total_estimated_cost_usd_micros": (audit.total_estimated_cost_usd_micros),
                "canonical_sort_alone_salvages_any_response": (
                    audit.canonical_sort_alone_salvages_any_response
                ),
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
