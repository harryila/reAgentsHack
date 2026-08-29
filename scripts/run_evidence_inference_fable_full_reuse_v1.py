#!/usr/bin/env python3
"""Prepare, execute, and externally replay the frozen full exact-wire overlay."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from literature_multiverse.evidence_inference_fable_full_reuse_v1 import (
    REUSE_DIRECTORY,
    REUSE_PLAN_FILE,
    EvidenceInferenceFableFullReuseError,
    EvidenceInferenceFableFullReusePlanV1,
    EvidenceInferenceFableReuseSourceV1,
    execute_evidence_inference_fable_full_reuse_v1,
    freeze_evidence_inference_fable_full_reuse_plan_v1,
    prepare_evidence_inference_fable_full_reuse_v1,
    require_evidence_inference_fable_full_reuse_scoring_v1,
    validate_evidence_inference_fable_full_reuse_v1,
)
from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    AnthropicFablePairedClientV1,
    EvidenceInferenceFableBudgetAuthorizationV1,
    EvidenceInferenceFablePairedRuntimeError,
    EvidenceInferenceFablePreparedRuntimeV1,
)
from literature_multiverse.evidence_inference_fable_retrospective_v1 import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_FULL_PLAN_PATH,
    DEFAULT_PILOT_PLAN_PATH,
    DEFAULT_RECOVERY_PILOT_PLAN_PATH,
    EvidenceInferenceFableRetrospectiveError,
    EvidenceInferenceFableRetrospectivePlanV1,
    validate_evidence_inference_fable_retrospective_plan_v1,
)
from literature_multiverse.providers import ProviderError, load_live_environment

DEFAULT_FULL_WORKSPACE = Path(
    "data/cache/evidence-inference-fable-retrospective-full-live-v1"
)
DEFAULT_SOURCE_WORKSPACE = Path(
    "data/cache/evidence-inference-fable-retrospective-pilot-live-v1"
)
DEFAULT_RECOVERY_WORKSPACE = Path(
    "data/cache/evidence-inference-fable-retrospective-pilot-recovery-v2-live"
)


class EvidenceInferenceFableFullReuseHarnessError(ValueError):
    """A CLI identity, path, environment, or execution guard failed."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "validate", "scoring-guard"):
        command = subcommands.add_parser(name)
        command.add_argument("--repository-root", type=Path, default=Path("."))
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
        command.add_argument("--full-plan", type=Path, default=DEFAULT_FULL_PLAN_PATH)
        command.add_argument(
            "--source-plan", type=Path, default=DEFAULT_PILOT_PLAN_PATH
        )
        command.add_argument(
            "--recovery-plan", type=Path, default=DEFAULT_RECOVERY_PILOT_PLAN_PATH
        )
        command.add_argument(
            "--workspace", type=Path, default=DEFAULT_FULL_WORKSPACE
        )
        command.add_argument(
            "--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE
        )
        command.add_argument(
            "--recovery-workspace", type=Path, default=DEFAULT_RECOVERY_WORKSPACE
        )
        command.add_argument("--expected-full-plan-sha256", required=True)
        command.add_argument("--expected-authorization-sha256", required=True)
        if name != "prepare":
            command.add_argument("--expected-reuse-plan-sha256", required=True)
        if name == "run":
            command.add_argument("--live", action="store_true")
            command.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def _rooted(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _safe_existing(root: Path, path: Path, *, directory: bool) -> Path:
    lexical = _rooted(root, path)
    if lexical.is_symlink():
        raise EvidenceInferenceFableFullReuseHarnessError(
            "fable_reuse_harness_symlink_forbidden"
        )
    resolved = lexical.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EvidenceInferenceFableFullReuseHarnessError(
            "fable_reuse_harness_path_escape"
        ) from exc
    if directory != resolved.is_dir():
        raise EvidenceInferenceFableFullReuseHarnessError(
            "fable_reuse_harness_path_kind_invalid"
        )
    return resolved


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceInferenceFableFullReuseHarnessError(
            "fable_reuse_harness_artifact_unsafe"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceInferenceFableFullReuseHarnessError(
            "fable_reuse_harness_artifact_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise EvidenceInferenceFableFullReuseHarnessError(
            "fable_reuse_harness_artifact_not_object"
        )
    return value


def _validated_plan(
    *, root: Path, path: Path, config: Path
) -> EvidenceInferenceFableRetrospectivePlanV1:
    source = _safe_existing(root, path, directory=False)
    serialized = EvidenceInferenceFableRetrospectivePlanV1.model_validate(_read(source))
    return validate_evidence_inference_fable_retrospective_plan_v1(
        repository_root=root,
        plan=serialized,
        config_path=config,
    )


def _context(args: argparse.Namespace) -> tuple[
    Path,
    EvidenceInferenceFableRetrospectivePlanV1,
    EvidenceInferenceFablePreparedRuntimeV1,
    EvidenceInferenceFableBudgetAuthorizationV1,
    list[EvidenceInferenceFableReuseSourceV1],
    EvidenceInferenceFableFullReusePlanV1,
]:
    root = args.repository_root.resolve(strict=True)
    config_source = _safe_existing(root, args.config, directory=False)
    config = config_source.relative_to(root)
    workspace = _safe_existing(root, args.workspace, directory=True)
    full_plan = _validated_plan(root=root, path=args.full_plan, config=config)
    source_plan = _validated_plan(root=root, path=args.source_plan, config=config)
    recovery_plan = _validated_plan(root=root, path=args.recovery_plan, config=config)
    prepared = EvidenceInferenceFablePreparedRuntimeV1.model_validate(
        _read(workspace / "00-prepared.json")
    )
    authorization = EvidenceInferenceFableBudgetAuthorizationV1.model_validate(
        _read(workspace / "01-authorization.json")
    )
    if (
        args.expected_full_plan_sha256 != full_plan.plan_sha256
        or args.expected_authorization_sha256 != authorization.authorization_sha256
        or prepared.retrospective_plan_sha256 != full_plan.plan_sha256
        or authorization.prepared_sha256 != prepared.prepared_sha256
    ):
        raise EvidenceInferenceFableFullReuseHarnessError(
            "fable_reuse_harness_full_identity_anchor_mismatch"
        )
    sources = [
        EvidenceInferenceFableReuseSourceV1(
            "poisoned_pilot_v1",
            source_plan,
            _safe_existing(root, args.source_workspace, directory=True),
        ),
        EvidenceInferenceFableReuseSourceV1(
            "recovery_pilot_v2",
            recovery_plan,
            _safe_existing(root, args.recovery_workspace, directory=True),
        ),
    ]
    expected_reuse = freeze_evidence_inference_fable_full_reuse_plan_v1(
        full_plan=full_plan,
        full_prepared=prepared,
        full_authorization=authorization,
        sources=sources,
    )
    if args.command != "prepare":
        archived = EvidenceInferenceFableFullReusePlanV1.model_validate(
            _read(workspace / REUSE_DIRECTORY / REUSE_PLAN_FILE)
        )
        if (
            args.expected_reuse_plan_sha256 != expected_reuse.plan_sha256
            or archived != expected_reuse
        ):
            raise EvidenceInferenceFableFullReuseHarnessError(
                "fable_reuse_harness_adoption_identity_anchor_mismatch"
            )
    return workspace, full_plan, prepared, authorization, sources, expected_reuse


def _summary(value: Any) -> dict[str, Any]:
    keep = {
        key: field
        for key, field in value.model_dump(mode="json").items()
        if key
        in {
            "plan_sha256",
            "full_plan_sha256",
            "full_authorization_sha256",
            "adopted_terminal_receipt_count",
            "inherited_ambiguous_failure_count",
            "maximum_new_provider_attempt_count",
            "terminal_sha256",
            "target_runtime_terminal_sha256",
            "target_runtime_status",
            "realized_adopted_terminal_receipt_count",
            "realized_inherited_ambiguous_failure_count",
            "new_provider_attempt_count",
            "target_accounted_spend_usd_micros",
            "new_provider_accounted_spend_usd_micros",
            "full_population_score_permitted",
        }
    }
    keep["artifact"] = type(value).__name__
    return keep


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace, full_plan, _prepared, authorization, sources, adoption_plan = _context(
        args
    )
    if args.command == "prepare":
        prepare_evidence_inference_fable_full_reuse_v1(
            workspace=workspace, adoption_plan=adoption_plan
        )
        value: Any = adoption_plan
    elif args.command == "run":
        if not args.live:
            raise EvidenceInferenceFableFullReuseHarnessError(
                "fable_reuse_live_flag_required"
            )
        env_file = _safe_existing(
            args.repository_root.resolve(strict=True), args.env_file, directory=False
        )
        load_live_environment(env_file, live_enabled=True)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise EvidenceInferenceFableFullReuseHarnessError(
                "fable_reuse_anthropic_api_key_missing"
            )
        if os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get(
            "ANTHROPIC_CUSTOM_HEADERS"
        ):
            raise EvidenceInferenceFableFullReuseHarnessError(
                "fable_reuse_custom_anthropic_transport_forbidden"
            )
        if authorization.authorization_sha256 != args.expected_authorization_sha256:
            raise EvidenceInferenceFableFullReuseHarnessError(
                "fable_reuse_authorization_changed_before_provider_boundary"
            )
        value = execute_evidence_inference_fable_full_reuse_v1(
            workspace=workspace,
            full_plan=full_plan,
            sources=sources,
            delegate=AnthropicFablePairedClientV1.from_anthropic_sdk(),
        )
    elif args.command == "validate":
        value = validate_evidence_inference_fable_full_reuse_v1(
            workspace=workspace, full_plan=full_plan, sources=sources
        )
    else:
        value = require_evidence_inference_fable_full_reuse_scoring_v1(
            workspace=workspace, full_plan=full_plan, sources=sources
        )
    print(json.dumps(_summary(value), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        EvidenceInferenceFableFullReuseError,
        EvidenceInferenceFableFullReuseHarnessError,
        EvidenceInferenceFablePairedRuntimeError,
        EvidenceInferenceFableRetrospectiveError,
        ProviderError,
        ValueError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
