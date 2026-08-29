#!/usr/bin/env python3
"""Prepare, execute, validate, and score the frozen Fable paired evaluation."""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from literature_multiverse.evidence_inference_fable_full_reuse_v1 import (
    REUSE_DIRECTORY,
    EvidenceInferenceFableFullReuseError,
    EvidenceInferenceFableReuseSourceV1,
    require_evidence_inference_fable_full_reuse_scoring_v1,
)
from literature_multiverse.evidence_inference_fable_full_union_reuse_v2 import (
    UNION_DIRECTORY,
    EvidenceInferenceFableFullUnionReuseError,
    EvidenceInferenceFableFullUnionScoringLineageV2,
    EvidenceInferenceFableFullUnionTerminalV2,
    EvidenceInferenceFableUnionSourceV2,
    freeze_evidence_inference_fable_full_union_scoring_lineage_v2,
    require_evidence_inference_fable_full_union_scoring_v2,
)
from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    AnthropicFablePairedClientV1,
    EvidenceInferenceFableBudgetAuthorizationArtifactV1,
    EvidenceInferenceFablePairedRuntimeError,
    EvidenceInferenceFablePreparedRuntimeV1,
    authorize_evidence_inference_fable_workspace_v1,
    execute_evidence_inference_fable_paired_v1,
    freeze_evidence_inference_fable_budget_authorization_v1,
    freeze_evidence_inference_fable_budget_authorization_v2,
    largest_certified_pair_liability_usd_micros_v1,
    parse_evidence_inference_fable_budget_authorization_v1,
    prepare_evidence_inference_fable_workspace_v1,
    reconstruct_evidence_inference_fable_prepared_runtime_v1,
    validate_evidence_inference_fable_workspace_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_inference_v1 import (
    EXPECTED_PILOT_PLAN_SHA256,
    EXPECTED_RECOVERY_PILOT_PLAN_SHA256,
    EvidenceInferenceFableInferenceError,
    require_full_preflight_gate_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_scoring_v1 import (
    EvidenceInferenceFableScoringError,
    PrivatePairedReportV1,
    PrivateReferenceLabelBundleV1,
    PublicPairedSummaryV1,
    ScoringCompletionCertificateV1,
    freeze_private_reference_label_bundle_v1,
    project_public_paired_summary_v1,
    repository_results_source_loader_v1,
    score_private_paired_report_v1,
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
from literature_multiverse.evidence_inference_fable_token_count_v1 import (
    AnthropicFableTokenCounterV1,
    EvidenceInferenceFableCountAuthorizationV1,
    EvidenceInferenceFableTokenCountError,
    execute_evidence_inference_fable_token_count_v1,
    freeze_evidence_inference_fable_count_authorization_v1,
    validate_evidence_inference_fable_token_count_v1,
)
from literature_multiverse.lineage import atomic_write_json, canonical_json_bytes
from literature_multiverse.prompt_optimization import load_manifest_split
from literature_multiverse.providers import ProviderError, load_live_environment

PILOT_PUBLIC_SUMMARY_PATH = Path(
    "artifacts/diagnostics/evidence-inference/fable-retrospective-pilot30-summary-v1.json"
)
FULL_PUBLIC_SUMMARY_PATH = Path(
    "artifacts/diagnostics/evidence-inference/fable-retrospective-full-summary-v1.json"
)
RECOVERY_PILOT_PUBLIC_SUMMARY_PATH = Path(
    "artifacts/diagnostics/evidence-inference/"
    "fable-retrospective-pilot30-recovery-v2-summary-v1.json"
)
FORBIDDEN_PUBLIC_NAMESPACES = (
    Path(".git"),
    Path(".github"),
    Path(".agents"),
    Path(".codex"),
    Path("paper"),
    Path("Formatting_Instructions_For_NeurIPS_2026 (2)"),
    Path("docs/paper"),
    Path("artifacts/paper"),
    Path("artifacts/submission"),
)
DEFAULT_POISONED_PILOT_WORKSPACE = Path(
    "data/cache/evidence-inference-fable-retrospective-pilot-live-v1"
)
DEFAULT_RECOVERY_PILOT_WORKSPACE = Path(
    "data/cache/evidence-inference-fable-retrospective-pilot-recovery-v2-live"
)
DEFAULT_POISONED_FULL_V2_WORKSPACE = Path(
    "data/cache/evidence-inference-fable-retrospective-full-live-v2"
)


class EvidenceInferenceFableHarnessError(ValueError):
    """A CLI path, identity, authorization, or private/public boundary failed."""


def _mode_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--mode",
        choices=(
            "pilot30_paired",
            "pilot30_recovery_v2_paired",
            "full_paired",
        ),
        required=True,
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    prepare = subcommands.add_parser("prepare", help="freeze a fresh label-blind workspace")
    _mode_arguments(prepare)

    authorize = subcommands.add_parser(
        "authorize", help="durably authorize an exact cumulative micro-dollar budget"
    )
    _mode_arguments(authorize)
    authorize.add_argument("--budget-usd-micros", type=int, required=True)
    authorize.add_argument("--token-count-workspace", type=Path)
    authorize.add_argument(
        "--input-token-headroom-per-request",
        type=int,
        choices=(0, 1024),
        default=0,
        help=(
            "use 1024 only for the explicit, hash-bound V2 certified liability "
            "authorization"
        ),
    )
    authorize.add_argument("--pilot-workspace", type=Path)
    authorize.add_argument("--pilot-certificate", type=Path)

    count_tokens = subcommands.add_parser(
        "count-tokens",
        help="certify exact per-request input-token liabilities with zero retries",
    )
    _mode_arguments(count_tokens)
    count_tokens.add_argument("--token-count-workspace", type=Path, required=True)
    count_tokens.add_argument("--live", action="store_true")
    count_tokens.add_argument("--env-file", type=Path, default=Path(".env"))
    count_tokens.add_argument("--expected-plan-sha256", required=True)

    validate_counts = subcommands.add_parser(
        "validate-counts", help="externally replay a terminal token-count workspace"
    )
    _mode_arguments(validate_counts)
    validate_counts.add_argument("--token-count-workspace", type=Path, required=True)
    validate_counts.add_argument("--expected-plan-sha256", required=True)

    run = subcommands.add_parser("run", help="execute or replay the exact-once roster")
    _mode_arguments(run)
    run.add_argument("--live", action="store_true")
    run.add_argument("--env-file", type=Path, default=Path(".env"))
    run.add_argument("--expected-plan-sha256", required=True)
    run.add_argument("--expected-authorization-sha256", required=True)
    run.add_argument("--pilot-workspace", type=Path)
    run.add_argument("--pilot-certificate", type=Path)

    validate = subcommands.add_parser("validate", help="externally replay a terminal workspace")
    _mode_arguments(validate)

    score = subcommands.add_parser(
        "score", help="open private labels only after complete terminal replay"
    )
    _mode_arguments(score)
    score.add_argument("--private-report", type=Path)
    score.add_argument("--public-summary", type=Path)
    score.add_argument("--completion-certificate", type=Path)
    score.add_argument("--union-scoring-lineage", type=Path)
    return parser


def _rooted(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _default_plan_path(mode: str) -> Path:
    return {
        "pilot30_paired": DEFAULT_PILOT_PLAN_PATH,
        "pilot30_recovery_v2_paired": DEFAULT_RECOVERY_PILOT_PLAN_PATH,
        "full_paired": DEFAULT_FULL_PLAN_PATH,
    }[mode]


def _default_public_path(mode: str) -> Path:
    return {
        "pilot30_paired": PILOT_PUBLIC_SUMMARY_PATH,
        "pilot30_recovery_v2_paired": RECOVERY_PILOT_PUBLIC_SUMMARY_PATH,
        "full_paired": FULL_PUBLIC_SUMMARY_PATH,
    }[mode]


def _read_object(path: Path, *, code: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceInferenceFableHarnessError(f"{code}_unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceInferenceFableHarnessError(f"{code}_invalid") from exc
    if not isinstance(value, dict):
        raise EvidenceInferenceFableHarnessError(f"{code}_not_object")
    return value


def _frozen_plan(
    *, root: Path, mode: str, config_path: Path, plan_path: Path | None
) -> tuple[EvidenceInferenceFableRetrospectivePlanV1, Any]:
    selected = plan_path or _default_plan_path(mode)
    if selected.is_absolute() or ".." in selected.parts:
        raise EvidenceInferenceFableHarnessError("fable_harness_plan_path_escape")
    lexical_path = root / selected
    if any(
        (root.joinpath(*selected.parts[:index])).is_symlink()
        for index in range(1, len(selected.parts) + 1)
    ):
        raise EvidenceInferenceFableHarnessError("fable_harness_plan_path_symlink")
    serialized_path = lexical_path.resolve(strict=True)
    try:
        serialized_path.relative_to(root)
    except ValueError as exc:
        raise EvidenceInferenceFableHarnessError("fable_harness_plan_path_escape") from exc
    serialized = validate_evidence_inference_fable_retrospective_plan_v1(
        repository_root=root,
        config_path=config_path,
        plan=_read_object(serialized_path, code="fable_harness_plan"),
    )
    reconstructed, prepared = reconstruct_evidence_inference_fable_prepared_runtime_v1(
        repository_root=root,
        mode=mode,  # type: ignore[arg-type]
        config_path=config_path,
    )
    if reconstructed != serialized:
        raise EvidenceInferenceFableHarnessError(
            "fable_harness_serialized_reconstructed_plan_mismatch"
        )
    return serialized, prepared


def _workspace(args: argparse.Namespace, root: Path) -> Path:
    lexical = _rooted(args.workspace, root)
    if lexical.is_symlink():
        raise EvidenceInferenceFableHarnessError("fable_harness_workspace_symlink")
    return lexical.resolve(strict=args.command != "prepare")


def _token_count_workspace(args: argparse.Namespace, root: Path) -> Path:
    lexical = _rooted(args.token_count_workspace, root)
    if lexical.is_symlink() or (lexical.exists() and not lexical.is_dir()):
        raise EvidenceInferenceFableHarnessError(
            "fable_harness_token_count_workspace_unsafe"
        )
    return lexical.resolve(strict=args.command != "count-tokens")


def _load_prepared(workspace: Path) -> EvidenceInferenceFablePreparedRuntimeV1:
    return EvidenceInferenceFablePreparedRuntimeV1.model_validate(
        _read_object(workspace / "00-prepared.json", code="fable_harness_prepared")
    )


def _load_authorization(
    workspace: Path,
) -> EvidenceInferenceFableBudgetAuthorizationArtifactV1:
    return parse_evidence_inference_fable_budget_authorization_v1(
        _read_object(workspace / "01-authorization.json", code="fable_harness_authorization")
    )


def _require_full_gate(
    *,
    args: argparse.Namespace,
    root: Path,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
) -> Any | None:
    if args.mode != "full_paired":
        return None
    if args.pilot_workspace is None or args.pilot_certificate is None:
        raise EvidenceInferenceFableHarnessError(
            "fable_full_authorization_requires_pilot_workspace_and_certificate"
        )
    certificate_source = _rooted(args.pilot_certificate, root)
    if certificate_source.is_symlink():
        raise EvidenceInferenceFableHarnessError(
            "fable_harness_pilot_certificate_symlink"
        )
    certificate = ScoringCompletionCertificateV1.model_validate(
        _read_object(
            certificate_source.resolve(strict=True),
            code="fable_harness_pilot_certificate",
        )
    )
    if certificate.plan_sha256 == EXPECTED_RECOVERY_PILOT_PLAN_SHA256:
        pilot_mode = "pilot30_recovery_v2_paired"
        pilot_path = DEFAULT_RECOVERY_PILOT_PLAN_PATH
    elif certificate.plan_sha256 == EXPECTED_PILOT_PLAN_SHA256:
        pilot_mode = "pilot30_paired"
        pilot_path = DEFAULT_PILOT_PLAN_PATH
    else:
        raise EvidenceInferenceFableHarnessError(
            "fable_harness_pilot_certificate_plan_unknown"
        )
    pilot_plan, _ = _frozen_plan(
        root=root,
        mode=pilot_mode,
        config_path=args.config,
        plan_path=pilot_path,
    )
    pilot_workspace_source = _rooted(args.pilot_workspace, root)
    if pilot_workspace_source.is_symlink():
        raise EvidenceInferenceFableHarnessError(
            "fable_harness_pilot_workspace_symlink"
        )
    return require_full_preflight_gate_v1(
        pilot_plan=pilot_plan,
        full_plan=full_plan,
        pilot_runtime_workspace=pilot_workspace_source.resolve(strict=True),
        scoring_certificate=certificate,
    )


def _private_label_loader(
    *,
    root: Path,
    config_path: Path,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
) -> Callable[[], PrivateReferenceLabelBundleV1]:
    """Return a closure; constructing it does not open a label-bearing split."""

    config_source = _rooted(config_path, root).resolve(strict=True)
    config = _read_object(config_source, code="fable_harness_config")
    manifest_value = (
        config.get("pilot_manifest_path")
        if plan.mode == "pilot30_paired"
        else config.get("full_manifest_path")
    )
    if not isinstance(manifest_value, str):
        raise EvidenceInferenceFableHarnessError("fable_harness_manifest_path_missing")
    manifest = _rooted(Path(manifest_value), root).resolve(strict=True)
    try:
        manifest.relative_to(root)
    except ValueError as exc:
        raise EvidenceInferenceFableHarnessError("fable_harness_manifest_path_escape") from exc

    def load() -> PrivateReferenceLabelBundleV1:
        examples = load_manifest_split(manifest, "test")
        expected: dict[str, Any] = {}
        planned_ids = {
            example_id
            for request in plan.roster
            for example_id in request.example_ids
        }
        for example in examples:
            if example.example_id not in planned_ids:
                continue
            findings = example.expected_output.get("findings")
            if (
                not isinstance(findings, list)
                or len(findings) != 1
                or not isinstance(findings[0], Mapping)
                or findings[0].get("direction")
                not in {"increase", "no_effect", "decrease"}
            ):
                raise EvidenceInferenceFableScoringError(
                    "fable_private_reference_direction_invalid"
                )
            expected[example.example_id] = findings[0]["direction"]
        return freeze_private_reference_label_bundle_v1(
            plan=plan,
            expected_directions=expected,
        )

    return load


def _require_full_reuse_scoring_guard(
    *,
    args: argparse.Namespace,
    root: Path,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    workspace: Path,
) -> EvidenceInferenceFableFullUnionTerminalV2 | None:
    """Validate exact-wire provenance before any full-test labels may open."""

    if args.mode != "full_paired":
        return None
    source_plan, _ = _frozen_plan(
        root=root,
        mode="pilot30_paired",
        config_path=args.config,
        plan_path=DEFAULT_PILOT_PLAN_PATH,
    )
    recovery_plan, _ = _frozen_plan(
        root=root,
        mode="pilot30_recovery_v2_paired",
        config_path=args.config,
        plan_path=DEFAULT_RECOVERY_PILOT_PLAN_PATH,
    )
    sources = [
        EvidenceInferenceFableReuseSourceV1(
            "poisoned_pilot_v1",
            source_plan,
            _rooted(DEFAULT_POISONED_PILOT_WORKSPACE, root).resolve(strict=True),
        ),
        EvidenceInferenceFableReuseSourceV1(
            "recovery_pilot_v2",
            recovery_plan,
            _rooted(DEFAULT_RECOVERY_PILOT_WORKSPACE, root).resolve(strict=True),
        ),
    ]
    if (workspace / UNION_DIRECTORY).exists():
        full_v2_plan, _ = _frozen_plan(
            root=root,
            mode="full_paired",
            config_path=args.config,
            plan_path=DEFAULT_FULL_PLAN_PATH,
        )
        full_v2_source = EvidenceInferenceFableUnionSourceV2(
            "poisoned_full_v2",
            full_v2_plan,
            _rooted(DEFAULT_POISONED_FULL_V2_WORKSPACE, root).resolve(strict=True),
            nested_reuse_sources=tuple(sources),
        )
        return require_evidence_inference_fable_full_union_scoring_v2(
            workspace=workspace,
            full_plan=full_plan,
            sources=[
                full_v2_source,
                EvidenceInferenceFableUnionSourceV2(
                    "poisoned_pilot_v1", source_plan, sources[0].workspace
                ),
                EvidenceInferenceFableUnionSourceV2(
                    "recovery_pilot_v2", recovery_plan, sources[1].workspace
                ),
            ],
        )
        return
    require_evidence_inference_fable_full_reuse_scoring_v1(
        workspace=workspace,
        full_plan=full_plan,
        sources=sources,
    )
    return None


def _safe_private_target(path: Path, workspace: Path) -> Path:
    candidate = path if path.is_absolute() else workspace / path
    resolved_parent = candidate.parent.resolve(strict=True)
    private_root = (workspace / "private").resolve(strict=True)
    try:
        resolved_parent.relative_to(private_root)
    except ValueError as exc:
        raise EvidenceInferenceFableHarnessError(
            "fable_private_artifact_must_remain_in_runtime_private_namespace"
        ) from exc
    return resolved_parent / candidate.name


def _safe_public_target(path: Path, root: Path) -> Path:
    candidate = _rooted(path, root)
    resolved_parent = candidate.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(root)
    except ValueError as exc:
        raise EvidenceInferenceFableHarnessError(
            "fable_public_summary_must_remain_in_repository"
        ) from exc
    resolved = resolved_parent / candidate.name
    forbidden = tuple((root / item).resolve() for item in FORBIDDEN_PUBLIC_NAMESPACES)
    if any(resolved == item or item in resolved.parents for item in forbidden):
        raise EvidenceInferenceFableHarnessError(
            "fable_sensitive_namespace_output_forbidden"
        )
    return resolved


def _materialize_score_bundle(
    *,
    private_report: PrivatePairedReportV1,
    public_summary: PublicPairedSummaryV1,
    completion_certificate: ScoringCompletionCertificateV1,
    private_path: Path,
    public_path: Path,
    certificate_path: Path,
    union_lineage: EvidenceInferenceFableFullUnionScoringLineageV2 | None = None,
    union_lineage_path: Path | None = None,
) -> None:
    """Publish private artifacts first and the public completion marker last."""

    private = PrivatePairedReportV1.model_validate(private_report.model_dump(mode="json"))
    public = PublicPairedSummaryV1.model_validate(public_summary.model_dump(mode="json"))
    certificate = ScoringCompletionCertificateV1.model_validate(
        completion_certificate.model_dump(mode="json")
    )
    if (
        public.private_report_sha256 != private.private_report_sha256
        or public.completion_certificate_sha256 != certificate.certificate_sha256
        or private.completion_certificate != certificate
        or public.plan_sha256 != private.plan_sha256
        or public.runtime_terminal_sha256 != private.runtime_terminal_sha256
        or (union_lineage is None) != (union_lineage_path is None)
    ):
        raise EvidenceInferenceFableHarnessError(
            "fable_score_bundle_private_public_binding_mismatch"
        )
    payloads: list[tuple[Path, Any]] = [
        (private_path, private),
        (certificate_path, certificate),
    ]
    if union_lineage is not None and union_lineage_path is not None:
        lineage = EvidenceInferenceFableFullUnionScoringLineageV2.model_validate(
            union_lineage.model_dump(mode="json")
        )
        if (
            lineage.target_runtime_terminal_sha256
            != private.runtime_terminal_sha256
            or lineage.completion_certificate_sha256
            != certificate.certificate_sha256
            or lineage.private_report_sha256 != private.private_report_sha256
            or lineage.public_summary_sha256 != public.public_summary_sha256
        ):
            raise EvidenceInferenceFableHarnessError(
                "fable_union_score_lineage_bundle_binding_mismatch"
            )
        payloads.append((union_lineage_path, lineage))
    # The unchanged public aggregate remains the completion marker and is last.
    payloads.append((public_path, public))
    _atomic_publish_score_payloads(payloads=payloads)


def _atomic_publish_score_payloads(*, payloads: Sequence[tuple[Path, Any]]) -> None:
    """Write a prevalidated bundle with rollback; the public payload must be last."""

    if len(payloads) not in {3, 4}:
        raise EvidenceInferenceFableHarnessError("fable_score_bundle_arity_invalid")
    payloads = tuple(payloads)
    targets = tuple(path for path, _ in payloads)
    if len({path.resolve() for path in targets}) != len(targets) or any(
        path.exists() or path.is_symlink() for path in targets
    ):
        raise EvidenceInferenceFableHarnessError(
            "fable_score_bundle_targets_not_fresh_or_distinct"
        )
    try:
        for target, payload in payloads:
            # The final/public payload is the terminal marker: it cannot precede
            # private row data, its certificate, or union lineage when present.
            atomic_write_json(target, payload)
    except BaseException:
        for target, payload in reversed(payloads):
            expected = canonical_json_bytes(payload) + b"\n"
            try:
                if not target.exists() and not target.is_symlink():
                    continue
                if (
                    target.is_symlink()
                    or not target.is_file()
                    or target.read_bytes() != expected
                ):
                    break
                target.unlink()
            except OSError:
                # Stop cleanup at the first artifact we cannot prove we own/remove.
                # Earlier entries are its dependencies and must remain available.
                break
        raise


def _summary(value: Any) -> dict[str, Any]:
    if isinstance(value, EvidenceInferenceFableRetrospectivePlanV1):
        return {
            "artifact": type(value).__name__,
            "mode": value.mode,
            "plan_sha256": value.plan_sha256,
            "requests": value.request_count,
            "provider_calls_made": value.provider_calls_made,
        }
    payload = value.model_dump(mode="json")
    keep = {
        key: payload[key]
        for key in (
            "status",
            "plan_sha256",
            "prepared_sha256",
            "authorization_sha256",
            "liability_basis",
            "terminal_sha256",
            "completed_request_count",
            "completed_pair_count",
            "cumulative_reported_spend_usd_micros",
            "certified_total_liability_usd_micros",
            "full_population_score_permitted",
            "private_report_sha256",
            "public_summary_sha256",
            "certificate_sha256",
            "claim_release_authority",
        )
        if key in payload
    }
    keep["artifact"] = type(value).__name__
    return keep


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve(strict=True)
    plan, prepared = _frozen_plan(
        root=root,
        mode=args.mode,
        config_path=args.config,
        plan_path=args.plan,
    )
    workspace = _workspace(args, root)

    if args.command == "prepare":
        prepare_evidence_inference_fable_workspace_v1(
            workspace=workspace,
            prepared=prepared,
        )
        value: Any = prepared
    elif args.command == "authorize":
        if args.budget_usd_micros <= 0:
            raise EvidenceInferenceFableHarnessError("fable_budget_must_be_positive")
        _require_full_gate(args=args, root=root, full_plan=plan)
        certified_count_terminal = None
        if args.token_count_workspace is not None:
            workspace_prepared = _load_prepared(workspace)
            if workspace_prepared != prepared:
                raise EvidenceInferenceFableHarnessError(
                    "fable_count_authorization_prepared_identity_mismatch"
                )
            certified = validate_evidence_inference_fable_token_count_v1(
                workspace=_token_count_workspace(args, root),
                prepared=prepared,
            )
            if certified.status != "completed_certified":
                raise EvidenceInferenceFableHarnessError(
                    "fable_count_authorization_requires_completed_certificate"
                )
            largest_pair_liability = largest_certified_pair_liability_usd_micros_v1(
                prepared=prepared,
                certified_request_liabilities_usd_micros=(
                    certified.certified_request_liabilities_usd_micros
                ),
            )
            if args.budget_usd_micros < largest_pair_liability:
                raise EvidenceInferenceFableHarnessError(
                    "fable_budget_below_certified_largest_pair_liability"
                )
            certified_count_terminal = certified.model_dump(mode="json")
        if args.input_token_headroom_per_request:
            if certified_count_terminal is None:
                raise EvidenceInferenceFableHarnessError(
                    "fable_headroom_authorization_requires_certified_counts"
                )
            value = freeze_evidence_inference_fable_budget_authorization_v2(
                prepared=prepared,
                configured_total_budget_usd_micros=args.budget_usd_micros,
                certified_count_terminal=certified_count_terminal,
            )
        else:
            value = freeze_evidence_inference_fable_budget_authorization_v1(
                prepared=prepared,
                configured_total_budget_usd_micros=args.budget_usd_micros,
                certified_count_terminal=certified_count_terminal,
            )
        authorize_evidence_inference_fable_workspace_v1(
            workspace=workspace,
            authorization=value,
        )
    elif args.command == "count-tokens":
        if not args.live:
            raise EvidenceInferenceFableHarnessError("fable_token_count_live_flag_required")
        workspace_prepared = _load_prepared(workspace)
        if args.expected_plan_sha256 != plan.plan_sha256 or workspace_prepared != prepared:
            raise EvidenceInferenceFableHarnessError(
                "fable_token_count_live_identity_anchor_mismatch"
            )
        count_workspace = _token_count_workspace(args, root)
        count_authorization = freeze_evidence_inference_fable_count_authorization_v1(
            prepared
        )
        count_authorization_path = count_workspace / "authorization.json"
        if count_authorization_path.exists():
            archived_count_authorization = (
                EvidenceInferenceFableCountAuthorizationV1.model_validate(
                    _read_object(
                        count_authorization_path,
                        code="fable_harness_token_count_authorization",
                    )
                )
            )
            if archived_count_authorization != count_authorization:
                raise EvidenceInferenceFableHarnessError(
                    "fable_token_count_live_authorization_anchor_mismatch"
                )
        if (count_workspace / "terminal.json").exists():
            value = validate_evidence_inference_fable_token_count_v1(
                workspace=count_workspace,
                prepared=prepared,
            )
        else:
            env_path = _rooted(args.env_file, root)
            load_live_environment(env_path, live_enabled=True)
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise EvidenceInferenceFableHarnessError(
                    "fable_anthropic_api_key_missing"
                )
            if os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get(
                "ANTHROPIC_CUSTOM_HEADERS"
            ):
                raise EvidenceInferenceFableHarnessError(
                    "fable_custom_anthropic_transport_forbidden"
                )
            value = execute_evidence_inference_fable_token_count_v1(
                workspace=count_workspace,
                prepared=prepared,
                authorization=count_authorization,
                counter=AnthropicFableTokenCounterV1.from_anthropic_sdk(),
            )
    elif args.command == "validate-counts":
        workspace_prepared = _load_prepared(workspace)
        if args.expected_plan_sha256 != plan.plan_sha256 or workspace_prepared != prepared:
            raise EvidenceInferenceFableHarnessError(
                "fable_token_count_validation_identity_anchor_mismatch"
            )
        value = validate_evidence_inference_fable_token_count_v1(
            workspace=_token_count_workspace(args, root),
            prepared=prepared,
        )
    elif args.command == "run":
        if not args.live:
            raise EvidenceInferenceFableHarnessError("fable_live_flag_required")
        workspace_prepared = _load_prepared(workspace)
        authorization = _load_authorization(workspace)
        if (
            args.expected_plan_sha256 != plan.plan_sha256
            or args.expected_authorization_sha256 != authorization.authorization_sha256
            or workspace_prepared != prepared
            or authorization.prepared_sha256 != prepared.prepared_sha256
        ):
            raise EvidenceInferenceFableHarnessError("fable_live_identity_anchor_mismatch")
        _require_full_gate(args=args, root=root, full_plan=plan)
        if args.mode == "full_paired" and any(
            (workspace / directory).exists()
            for directory in (REUSE_DIRECTORY, UNION_DIRECTORY)
        ):
            raise EvidenceInferenceFableHarnessError(
                "fable_full_reuse_workspace_requires_reuse_aware_cli"
            )
        env_path = _rooted(args.env_file, root)
        load_live_environment(env_path, live_enabled=True)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise EvidenceInferenceFableHarnessError("fable_anthropic_api_key_missing")
        if os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("ANTHROPIC_CUSTOM_HEADERS"):
            raise EvidenceInferenceFableHarnessError(
                "fable_custom_anthropic_transport_forbidden"
            )
        value = execute_evidence_inference_fable_paired_v1(
            workspace=workspace,
            plan=plan,
            client=AnthropicFablePairedClientV1.from_anthropic_sdk(),
        )
    elif args.command == "validate":
        value = validate_evidence_inference_fable_workspace_v1(
            workspace=workspace,
            plan=plan,
        )
    else:
        union_terminal = _require_full_reuse_scoring_guard(
            args=args,
            root=root,
            full_plan=plan,
            workspace=workspace,
        )
        private_parent = workspace / "private"
        if private_parent.is_symlink() or (
            private_parent.exists() and not private_parent.is_dir()
        ):
            raise EvidenceInferenceFableHarnessError(
                "fable_runtime_private_namespace_unsafe"
            )
        private_parent.mkdir(mode=0o700, exist_ok=True)
        if (
            private_parent.is_symlink()
            or not private_parent.is_dir()
            or stat.S_IMODE(private_parent.stat().st_mode) != 0o700
        ):
            raise EvidenceInferenceFableHarnessError(
                "fable_runtime_private_namespace_unsafe"
            )
        private_path = _safe_private_target(
            args.private_report or Path("private/scored-report-v1.json"),
            workspace,
        )
        public_path = _safe_public_target(
            args.public_summary or _default_public_path(args.mode),
            root,
        )
        certificate_path = _safe_private_target(
            args.completion_certificate or Path("private/completion-certificate-v1.json"),
            workspace,
        )
        union_lineage_path = (
            _safe_private_target(
                args.union_scoring_lineage
                or Path("private/full-union-scoring-lineage-v2.json"),
                workspace,
            )
            if union_terminal is not None
            else None
        )
        if (
            certificate_path.exists()
            or certificate_path in (private_path, public_path)
            or (
                union_lineage_path is not None
                and (
                    union_lineage_path.exists()
                    or union_lineage_path
                    in (private_path, public_path, certificate_path)
                )
            )
        ):
            raise EvidenceInferenceFableHarnessError(
                "fable_completion_certificate_target_not_fresh_or_distinct"
            )
        report = score_private_paired_report_v1(
            plan=plan,
            runtime_workspace=workspace,
            source_loader=repository_results_source_loader_v1(
                repository_root=root,
                config_path=args.config,
            ),
            private_label_loader=_private_label_loader(
                root=root,
                config_path=args.config,
                plan=plan,
            ),
        )
        public = project_public_paired_summary_v1(report)
        union_lineage = (
            freeze_evidence_inference_fable_full_union_scoring_lineage_v2(
                union_terminal=union_terminal,
                completion_certificate_sha256=(
                    report.completion_certificate.certificate_sha256
                ),
                private_report_sha256=report.private_report_sha256,
                public_summary_sha256=public.public_summary_sha256,
            )
            if union_terminal is not None
            else None
        )
        _materialize_score_bundle(
            private_report=report,
            public_summary=public,
            completion_certificate=report.completion_certificate,
            private_path=private_path,
            public_path=public_path,
            certificate_path=certificate_path,
            union_lineage=union_lineage,
            union_lineage_path=union_lineage_path,
        )
        value = public

    print(json.dumps(_summary(value), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        EvidenceInferenceFableHarnessError,
        EvidenceInferenceFableFullReuseError,
        EvidenceInferenceFableFullUnionReuseError,
        EvidenceInferenceFableInferenceError,
        EvidenceInferenceFablePairedRuntimeError,
        EvidenceInferenceFableRetrospectiveError,
        EvidenceInferenceFableScoringError,
        EvidenceInferenceFableTokenCountError,
        ProviderError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
