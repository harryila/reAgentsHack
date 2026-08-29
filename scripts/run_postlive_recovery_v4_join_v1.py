#!/usr/bin/env python3
"""Build, replay, or inspect the non-authorizing recovery-v4 posthoc join."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from literature_multiverse.lineage import atomic_write_json
from literature_multiverse.postlive_recovery_v4_join_v1 import (
    PostLiveRecoveryV4JoinArtifactV1,
    build_postlive_recovery_v4_join_from_artifact_v1,
    validate_postlive_recovery_v4_join_artifact_v1,
)


def _datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("generated_at_must_be_iso8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("generated_at_requires_timezone")
    return parsed


def _source_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--posthoc-artifact",
        type=Path,
        required=True,
        help="Frozen recovery-v4 posthoc artifact; it is opened read-only.",
    )
    parser.add_argument(
        "--immutable-v4-workspace",
        type=Path,
        help="Override the immutable recovery-v4 workspace used for external replay.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join the repaired recovery-v4 native projection to graph, synthesis, "
            "condition, and audit mechanics without scientific or release authority."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="Build and write the additive join artifact.")
    _source_argument(build)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--generated-at", type=_datetime, required=True)
    build.add_argument("--target-direction", choices=("increase", "decrease"), required=True)
    build.add_argument(
        "--prespecified-moderator",
        action="append",
        default=[],
        help="Repeat in sorted unique order; omission does not infer a condition claim.",
    )
    build.add_argument("--force", action="store_true")

    validate = commands.add_parser(
        "validate", help="Rebuild from the posthoc source and require exact equality."
    )
    _source_argument(validate)
    validate.add_argument("--input", type=Path, required=True)

    status = commands.add_parser("status", help="Contract-check and summarize one join.")
    status.add_argument("--input", type=Path, required=True)
    return parser


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("postlive_recovery_v4_join_cli_input_not_object")
    return value


def _status(artifact: PostLiveRecoveryV4JoinArtifactV1) -> dict[str, Any]:
    return {
        "join_version": artifact.join_version,
        "status": artifact.status,
        "artifact_sha256": artifact.artifact_sha256,
        "source_posthoc_artifact_sha256": artifact.source_posthoc_artifact_sha256,
        "source_v4_terminal_status": artifact.source_v4_terminal_status,
        "source_v4_runtime_workspace_success": artifact.source_v4_runtime_workspace_success,
        "canonicalizer_provider_calls_made": artifact.canonicalizer_provider_calls_made,
        "upstream_v4_provider_attempt_count": artifact.upstream_v4_provider_attempt_count,
        "upstream_v4_provider_response_completed": (
            artifact.upstream_v4_provider_response_completed
        ),
        "source_posthoc_external_replay_performed": (
            artifact.source_posthoc_external_replay_performed
        ),
        "source_posthoc_external_replay_sha256": (
            artifact.source_posthoc_external_replay_sha256
        ),
        "canonicalization_pipeline_sha256": artifact.canonicalization_pipeline_sha256,
        "source_repair_change_count": len(artifact.source_repair_changes),
        "evidence_graph_sha256": artifact.evidence_graph_sha256,
        "synthesis_sha256": artifact.synthesis_sha256,
        "condition_status": artifact.condition_mechanics.status,
        "audit_status": artifact.audit_mechanics.status,
        "audit_candidate_count": len(artifact.audit_mechanics.audit_candidates),
        "post_hoc_source_span_repair": artifact.post_hoc_source_span_repair,
        "extraction_accuracy_authority": artifact.extraction_accuracy_authority,
        "scientific_synthesis_authority": artifact.scientific_synthesis_authority,
        "claim_release_authority": artifact.claim_release_authority,
        "release_authorizing": artifact.release_authorizing,
        "blockers": artifact.blockers,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build":
        artifact = build_postlive_recovery_v4_join_from_artifact_v1(
            repository_root=args.repository_root,
            posthoc_artifact_path=args.posthoc_artifact,
            generated_at=args.generated_at,
            target_direction=args.target_direction,
            prespecified_moderators=args.prespecified_moderator,
            immutable_v4_workspace=args.immutable_v4_workspace,
        )
        if args.output.resolve() == args.posthoc_artifact.resolve():
            raise ValueError("postlive_recovery_v4_join_refuses_source_overwrite")
        atomic_write_json(args.output, artifact, force=args.force)
    elif args.command == "validate":
        artifact = validate_postlive_recovery_v4_join_artifact_v1(
            artifact=_load(args.input),
            repository_root=args.repository_root,
            posthoc_artifact_path=args.posthoc_artifact,
            immutable_v4_workspace=args.immutable_v4_workspace,
        )
    else:
        artifact = PostLiveRecoveryV4JoinArtifactV1.model_validate(_load(args.input))
    print(json.dumps(_status(artifact), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
