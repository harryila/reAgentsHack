#!/usr/bin/env python3
"""Prepare, run, and externally replay the frozen full priority-union overlay."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from literature_multiverse.evidence_inference_fable_full_reuse_v1 import (
    EvidenceInferenceFableReuseSourceV1,
)
from literature_multiverse.evidence_inference_fable_full_union_reuse_v2 import (
    UNION_DIRECTORY,
    UNION_PLAN_FILE,
    EvidenceInferenceFableFullUnionPlanV2,
    EvidenceInferenceFableFullUnionReuseError,
    EvidenceInferenceFableFullUnionScoringLineageV2,
    EvidenceInferenceFableUnionSourceV2,
    derive_evidence_inference_fable_full_union_failure_burden_v2,
    execute_evidence_inference_fable_full_union_v2,
    freeze_evidence_inference_fable_full_union_plan_v2,
    materialize_evidence_inference_fable_full_union_public_evaluation_v2,
    prepare_evidence_inference_fable_full_union_v2,
    project_evidence_inference_fable_full_union_public_evaluation_v2,
    require_evidence_inference_fable_full_union_scoring_v2,
    validate_evidence_inference_fable_full_union_paths_v2,
    validate_evidence_inference_fable_full_union_v2,
)
from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    AnthropicFablePairedClientV1,
    EvidenceInferenceFableBudgetAuthorizationV2,
    EvidenceInferenceFablePairedRuntimeError,
    EvidenceInferenceFablePreparedRuntimeV1,
    parse_evidence_inference_fable_budget_authorization_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_scoring_v1 import (
    PublicPairedSummaryV1,
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

DEFAULT_TARGET_WORKSPACE = Path("data/cache/evidence-inference-fable-retrospective-full-live-v4")
DEFAULT_FULL_V2_SOURCE_WORKSPACE = Path(
    "data/cache/evidence-inference-fable-retrospective-full-live-v2"
)
DEFAULT_PILOT_SOURCE_WORKSPACE = Path(
    "data/cache/evidence-inference-fable-retrospective-pilot-live-v1"
)
DEFAULT_RECOVERY_SOURCE_WORKSPACE = Path(
    "data/cache/evidence-inference-fable-retrospective-pilot-recovery-v2-live"
)
FIXED_TARGET_BUDGET_USD_MICROS = 99_000_000
DEFAULT_PUBLIC_SUMMARY = Path(
    "artifacts/diagnostics/evidence-inference/fable-retrospective-full-summary-v1.json"
)
DEFAULT_PUBLIC_EVALUATION = Path(
    "artifacts/diagnostics/evidence-inference/fable-retrospective-full-union-evaluation-v2.json"
)
DEFAULT_PRIVATE_UNION_LINEAGE = Path("private/full-union-scoring-lineage-v2.json")


class EvidenceInferenceFableFullUnionHarnessError(ValueError):
    """A CLI identity, path, environment, or execution guard failed."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "validate", "scoring-guard", "project-public"):
        command = subcommands.add_parser(name)
        command.add_argument("--repository-root", type=Path, default=Path("."))
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
        command.add_argument("--full-plan", type=Path, default=DEFAULT_FULL_PLAN_PATH)
        command.add_argument("--pilot-plan", type=Path, default=DEFAULT_PILOT_PLAN_PATH)
        command.add_argument("--recovery-plan", type=Path, default=DEFAULT_RECOVERY_PILOT_PLAN_PATH)
        command.add_argument("--workspace", type=Path, default=DEFAULT_TARGET_WORKSPACE)
        command.add_argument(
            "--full-v2-source-workspace",
            type=Path,
            default=DEFAULT_FULL_V2_SOURCE_WORKSPACE,
        )
        command.add_argument(
            "--pilot-source-workspace",
            type=Path,
            default=DEFAULT_PILOT_SOURCE_WORKSPACE,
        )
        command.add_argument(
            "--recovery-source-workspace",
            type=Path,
            default=DEFAULT_RECOVERY_SOURCE_WORKSPACE,
        )
        command.add_argument("--expected-full-plan-sha256", required=True)
        command.add_argument("--expected-authorization-sha256", required=True)
        if name != "prepare":
            command.add_argument("--expected-union-plan-sha256", required=True)
        if name == "run":
            command.add_argument("--live", action="store_true")
            command.add_argument("--env-file", type=Path, default=Path(".env"))
        if name == "project-public":
            command.add_argument("--public-summary", type=Path, default=DEFAULT_PUBLIC_SUMMARY)
            command.add_argument("--union-scoring-lineage", type=Path)
            command.add_argument("--output", type=Path, default=DEFAULT_PUBLIC_EVALUATION)
    return parser


def _rooted(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _safe_existing(root: Path, path: Path, *, directory: bool) -> Path:
    lexical = _rooted(root, path)
    if lexical.is_symlink():
        raise EvidenceInferenceFableFullUnionHarnessError("fable_union_harness_symlink_forbidden")
    resolved = lexical.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EvidenceInferenceFableFullUnionHarnessError(
            "fable_union_harness_path_escape"
        ) from exc
    if directory != resolved.is_dir():
        raise EvidenceInferenceFableFullUnionHarnessError("fable_union_harness_path_kind_invalid")
    return resolved


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceInferenceFableFullUnionHarnessError("fable_union_harness_artifact_unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceInferenceFableFullUnionHarnessError(
            "fable_union_harness_artifact_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise EvidenceInferenceFableFullUnionHarnessError("fable_union_harness_artifact_not_object")
    return value


def _safe_private_lineage(*, root: Path, workspace: Path, requested: Path | None) -> Path:
    candidate = (
        workspace / DEFAULT_PRIVATE_UNION_LINEAGE if requested is None else _rooted(root, requested)
    )
    source = _safe_existing(root, candidate, directory=False)
    private_root = (workspace / "private").resolve(strict=True)
    try:
        source.relative_to(private_root)
    except ValueError as exc:
        raise EvidenceInferenceFableFullUnionHarnessError(
            "fable_union_harness_lineage_outside_private_namespace"
        ) from exc
    return source


def _safe_public_output(*, root: Path, requested: Path) -> Path:
    lexical = _rooted(root, requested)
    if lexical.exists() or lexical.is_symlink() or lexical.parent.is_symlink():
        raise EvidenceInferenceFableFullUnionHarnessError(
            "fable_union_harness_public_output_not_fresh"
        )
    parent = lexical.parent.resolve(strict=True)
    public_root = (root / "artifacts" / "diagnostics" / "evidence-inference").resolve(strict=True)
    try:
        parent.relative_to(public_root)
    except ValueError as exc:
        raise EvidenceInferenceFableFullUnionHarnessError(
            "fable_union_harness_public_output_path_escape"
        ) from exc
    return parent / lexical.name


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


def _context(
    args: argparse.Namespace,
) -> tuple[
    Path,
    EvidenceInferenceFableRetrospectivePlanV1,
    EvidenceInferenceFablePreparedRuntimeV1,
    EvidenceInferenceFableBudgetAuthorizationV2,
    list[EvidenceInferenceFableUnionSourceV2],
    EvidenceInferenceFableFullUnionPlanV2,
]:
    root = args.repository_root.resolve(strict=True)
    config_source = _safe_existing(root, args.config, directory=False)
    config = config_source.relative_to(root)
    workspace = _safe_existing(root, args.workspace, directory=True)
    full_plan = _validated_plan(root=root, path=args.full_plan, config=config)
    pilot_plan = _validated_plan(root=root, path=args.pilot_plan, config=config)
    recovery_plan = _validated_plan(root=root, path=args.recovery_plan, config=config)
    prepared = EvidenceInferenceFablePreparedRuntimeV1.model_validate(
        _read(workspace / "00-prepared.json")
    )
    authorization = parse_evidence_inference_fable_budget_authorization_v1(
        _read(workspace / "01-authorization.json")
    )
    if not isinstance(authorization, EvidenceInferenceFableBudgetAuthorizationV2):
        raise EvidenceInferenceFableFullUnionHarnessError(
            "fable_union_harness_headroom_authorization_v2_required"
        )
    if (
        args.expected_full_plan_sha256 != full_plan.plan_sha256
        or args.expected_authorization_sha256 != authorization.authorization_sha256
        or prepared.retrospective_plan_sha256 != full_plan.plan_sha256
        or authorization.prepared_sha256 != prepared.prepared_sha256
        or authorization.configured_total_budget_usd_micros != FIXED_TARGET_BUDGET_USD_MICROS
    ):
        raise EvidenceInferenceFableFullUnionHarnessError(
            "fable_union_harness_target_identity_anchor_mismatch"
        )

    pilot_source = EvidenceInferenceFableReuseSourceV1(
        "poisoned_pilot_v1",
        pilot_plan,
        _safe_existing(root, args.pilot_source_workspace, directory=True),
    )
    recovery_source = EvidenceInferenceFableReuseSourceV1(
        "recovery_pilot_v2",
        recovery_plan,
        _safe_existing(root, args.recovery_source_workspace, directory=True),
    )
    sources = [
        EvidenceInferenceFableUnionSourceV2(
            "poisoned_full_v2",
            full_plan,
            _safe_existing(root, args.full_v2_source_workspace, directory=True),
            (pilot_source, recovery_source),
        ),
        EvidenceInferenceFableUnionSourceV2(
            "poisoned_pilot_v1", pilot_plan, pilot_source.workspace
        ),
        EvidenceInferenceFableUnionSourceV2(
            "recovery_pilot_v2", recovery_plan, recovery_source.workspace
        ),
    ]
    validate_evidence_inference_fable_full_union_paths_v2(workspace=workspace, sources=sources)
    expected_union = freeze_evidence_inference_fable_full_union_plan_v2(
        full_plan=full_plan,
        full_prepared=prepared,
        full_authorization=authorization,
        sources=sources,
    )
    if args.command != "prepare":
        archived = EvidenceInferenceFableFullUnionPlanV2.model_validate(
            _read(workspace / UNION_DIRECTORY / UNION_PLAN_FILE)
        )
        if (
            args.expected_union_plan_sha256 != expected_union.plan_sha256
            or archived != expected_union
        ):
            raise EvidenceInferenceFableFullUnionHarnessError(
                "fable_union_harness_union_identity_anchor_mismatch"
            )
    return workspace, full_plan, prepared, authorization, sources, expected_union


def _summary(value: Any) -> dict[str, Any]:
    allowed = {
        "plan_sha256",
        "full_plan_sha256",
        "full_authorization_sha256",
        "adopted_terminal_receipt_count",
        "inherited_ambiguous_failure_count",
        "maximum_new_provider_attempt_count",
        "shadowed_lower_priority_candidate_count",
        "terminal_sha256",
        "target_runtime_terminal_sha256",
        "target_runtime_status",
        "realized_adopted_terminal_receipt_count",
        "realized_inherited_ambiguous_failure_count",
        "new_provider_attempt_count",
        "target_accounted_spend_usd_micros",
        "new_provider_accounted_spend_usd_micros",
        "full_population_score_permitted",
        "evaluation_sha256",
        "public_summary_sha256",
        "union_scoring_lineage_sha256",
        "inherited_failure_request_count_by_arm",
        "inherited_failure_locked_question_count_by_arm",
        "inherited_failure_locked_question_count",
        "adopted_target_accounted_spend_usd_micros",
        "failure_burden_sha256",
        "target_incident_count",
        "target_incident_locked_question_count",
        "target_incident_request_count_by_arm",
        "target_incident_locked_question_count_by_arm",
        "new_runtime_incident_request_count",
        "new_runtime_incident_locked_question_count",
        "new_runtime_incident_request_count_by_arm",
        "new_runtime_incident_locked_question_count_by_arm",
        "all_forced_zero_request_count",
        "all_forced_zero_locked_question_count",
        "all_forced_zero_request_count_by_arm",
        "all_forced_zero_locked_question_count_by_arm",
    }
    summary = {key: field for key, field in value.model_dump(mode="json").items() if key in allowed}
    summary["artifact"] = type(value).__name__
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Deliberately precede path, environment, SDK, and provider access.
    if args.command == "run" and not args.live:
        raise EvidenceInferenceFableFullUnionHarnessError("fable_union_live_flag_required")
    root = args.repository_root.resolve(strict=True)
    workspace, full_plan, _prepared, authorization, sources, union_plan = _context(args)
    if args.command == "prepare":
        prepare_evidence_inference_fable_full_union_v2(workspace=workspace, union_plan=union_plan)
        value: Any = union_plan
    elif args.command == "run":
        env_file = _safe_existing(
            args.repository_root.resolve(strict=True), args.env_file, directory=False
        )
        load_live_environment(env_file, live_enabled=True)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise EvidenceInferenceFableFullUnionHarnessError(
                "fable_union_anthropic_api_key_missing"
            )
        if os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("ANTHROPIC_CUSTOM_HEADERS"):
            raise EvidenceInferenceFableFullUnionHarnessError(
                "fable_union_custom_anthropic_transport_forbidden"
            )
        if authorization.authorization_sha256 != args.expected_authorization_sha256:
            raise EvidenceInferenceFableFullUnionHarnessError(
                "fable_union_authorization_changed_before_provider_boundary"
            )
        value = execute_evidence_inference_fable_full_union_v2(
            workspace=workspace,
            full_plan=full_plan,
            sources=sources,
            delegate=AnthropicFablePairedClientV1.from_anthropic_sdk(),
        )
    elif args.command == "validate":
        value = validate_evidence_inference_fable_full_union_v2(
            workspace=workspace, full_plan=full_plan, sources=sources
        )
    elif args.command == "scoring-guard":
        value = require_evidence_inference_fable_full_union_scoring_v2(
            workspace=workspace, full_plan=full_plan, sources=sources
        )
    else:
        terminal = validate_evidence_inference_fable_full_union_v2(
            workspace=workspace, full_plan=full_plan, sources=sources
        )
        public = PublicPairedSummaryV1.model_validate(
            _read(_safe_existing(root, args.public_summary, directory=False))
        )
        lineage = EvidenceInferenceFableFullUnionScoringLineageV2.model_validate(
            _read(
                _safe_private_lineage(
                    root=root,
                    workspace=workspace,
                    requested=args.union_scoring_lineage,
                )
            )
        )
        failure_burden = derive_evidence_inference_fable_full_union_failure_burden_v2(
            workspace=workspace,
            full_plan=full_plan,
            union_plan=union_plan,
            union_terminal=terminal,
        )
        value = project_evidence_inference_fable_full_union_public_evaluation_v2(
            full_plan=full_plan,
            union_plan=union_plan,
            union_terminal=terminal,
            public_summary=public,
            union_scoring_lineage=lineage,
            failure_burden=failure_burden,
        )
        materialize_evidence_inference_fable_full_union_public_evaluation_v2(
            evaluation=value,
            output_path=_safe_public_output(root=root, requested=args.output),
        )
    print(json.dumps(_summary(value), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        EvidenceInferenceFableFullUnionHarnessError,
        EvidenceInferenceFableFullUnionReuseError,
        EvidenceInferenceFablePairedRuntimeError,
        EvidenceInferenceFableRetrospectiveError,
        ProviderError,
        ValueError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
