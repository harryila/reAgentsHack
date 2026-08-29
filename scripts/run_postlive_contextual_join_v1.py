#!/usr/bin/env python3
"""Build, externally replay, or inspect a non-authorizing post-live certificate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from literature_multiverse.lineage import atomic_write_json
from literature_multiverse.postlive_contextual_join_v1 import (
    PostLiveContextualCertificateV1,
    build_postlive_contextual_certificate_from_workspace_v1,
    validate_postlive_contextual_certificate_v1,
)


def _datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("generated_at_must_be_iso8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("generated_at_requires_timezone")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join a fully replayed contextual frontier terminal result to evidence-graph, "
            "synthesis, leave-one-out audit, and non-authorizing certificate mechanics."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="Replay a runtime workspace and write a join.")
    build.add_argument("--repository-root", type=Path, default=Path("."))
    build.add_argument("--runtime-workspace", type=Path, required=True)
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
        "validate", help="Externally replay runtime and require exact certificate equality."
    )
    validate.add_argument("--repository-root", type=Path, default=Path("."))
    validate.add_argument("--runtime-workspace", type=Path, required=True)
    validate.add_argument("--input", type=Path, required=True)

    status = commands.add_parser(
        "status", help="Contract-check and summarize without replaying the runtime workspace."
    )
    status.add_argument("--input", type=Path, required=True)
    return parser


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("postlive_contextual_join_cli_input_not_object")
    return value


def _status(certificate: PostLiveContextualCertificateV1) -> dict[str, Any]:
    audit = certificate.audit_mechanics
    condition = certificate.condition_mechanics
    return {
        "certificate_version": certificate.certificate_version,
        "status": certificate.status,
        "certificate_sha256": certificate.certificate_sha256,
        "terminal_report_sha256": certificate.terminal_report_sha256,
        "runtime_workspace_validation_sha256": (certificate.runtime_workspace_validation_sha256),
        "evidence_graph_sha256": certificate.evidence_graph_sha256,
        "synthesis_sha256": certificate.synthesis_sha256,
        "condition_status": condition.status,
        "condition_analysis_executed": condition.analysis_executed,
        "audit_status": audit.status,
        "audit_candidate_count": len(audit.audit_candidates),
        "audit_action_selected": audit.audit_action_selected,
        "human_adjudication_count": audit.human_adjudication_count,
        "item_error_calibration_performed": (audit.item_error_calibration_performed),
        "human_cost_measurement_performed": audit.human_cost_measurement_performed,
        "release_authorizing": certificate.release_authorizing,
        "claim_release_authority": certificate.claim_release_authority,
        "blockers": certificate.blockers,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build":
        certificate = build_postlive_contextual_certificate_from_workspace_v1(
            repository_root=args.repository_root,
            runtime_workspace=args.runtime_workspace,
            generated_at=args.generated_at,
            target_direction=args.target_direction,
            prespecified_moderators=args.prespecified_moderator,
            external_replay=True,
        )
        atomic_write_json(args.output, certificate, force=args.force)
    elif args.command == "validate":
        certificate = validate_postlive_contextual_certificate_v1(
            certificate=_load(args.input),
            repository_root=args.repository_root,
            runtime_workspace=args.runtime_workspace,
            external_replay=True,
        )
    else:
        certificate = PostLiveContextualCertificateV1.model_validate(_load(args.input))
    print(json.dumps(_status(certificate), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
