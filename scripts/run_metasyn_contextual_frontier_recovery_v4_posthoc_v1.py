#!/usr/bin/env python3
"""Freeze the offline source-span-only recovery-v4 canonicalization artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.metasyn_contextual_frontier_recovery_v4_posthoc_v1 import (
    DEFAULT_IMMUTABLE_WORKSPACE,
    DEFAULT_WORKSPACE,
    freeze_metasyn_contextual_frontier_recovery_v4_posthoc_artifact_v1,
    write_metasyn_contextual_frontier_recovery_v4_posthoc_artifact_v1,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--immutable-workspace", type=Path, default=None)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    immutable = args.immutable_workspace
    if immutable is None:
        immutable = args.repository_root / DEFAULT_IMMUTABLE_WORKSPACE
    workspace = args.workspace
    if workspace is None:
        workspace = args.repository_root / DEFAULT_WORKSPACE
    if args.validate_only:
        artifact = freeze_metasyn_contextual_frontier_recovery_v4_posthoc_artifact_v1(
            repository_root=args.repository_root,
            immutable_workspace=immutable,
        )
    else:
        artifact = write_metasyn_contextual_frontier_recovery_v4_posthoc_artifact_v1(
            repository_root=args.repository_root,
            immutable_workspace=immutable,
            workspace=workspace,
        )
    print(
        json.dumps(
            {
                "artifact_sha256": artifact.artifact_sha256,
                "canonicalized_response_sha256": artifact.canonicalized_response_sha256,
                "canonicalizer_provider_calls_made": 0,
                "evaluation_sha256": artifact.evaluation_sha256,
                "recovery_label": artifact.recovery_label,
                "status": artifact.status,
                "upstream_v4_provider_attempt_count": 1,
                "upstream_v4_provider_response_completed": True,
                "claim_release_authority": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
