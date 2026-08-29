#!/usr/bin/env python3
"""Prepare and, only with --live, execute the at-most-once hosted numeric pilot."""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

from literature_multiverse.hosted_native_numeric_pilot_v1 import (
    DEFAULT_CANARY_WORKSPACE,
    DEFAULT_CONFIG_PATH,
    DEFAULT_WORKSPACE,
    AnthropicHostedNativeNumericGenerationClientV1,
    AnthropicHostedNativeNumericTokenCounterV1,
    HostedNativeNumericPilotError,
    authorize_hosted_native_numeric_pilot_v1,
    count_hosted_native_numeric_pilot_tokens_v1,
    execute_hosted_native_numeric_pilot_v1,
    freeze_hosted_native_numeric_pilot_plan_v1,
    load_hosted_native_numeric_pilot_plan_v1,
    load_hosted_native_numeric_terminal_v1,
    preflight_hosted_native_numeric_count_v1,
    preflight_hosted_native_numeric_execution_v1,
    prepare_hosted_native_numeric_pilot_v1,
    reserve_hosted_native_numeric_pilot_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-record, yield-only, at-most-once Fable native extraction pilot."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "prepare"):
        item = sub.add_parser(name)
        item.add_argument("--repository-root", type=Path, default=Path("."))
        item.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
        item.add_argument("--canary-workspace", type=Path, default=DEFAULT_CANARY_WORKSPACE)
        item.add_argument("--expected-canary-terminal-sha256", required=True)
        if name == "prepare":
            item.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    reserve = sub.add_parser("reserve")
    reserve.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    reserve.add_argument("--expected-plan-sha256", required=True)
    count = sub.add_parser("count")
    count.add_argument("--repository-root", type=Path, default=Path("."))
    count.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    count.add_argument("--expected-plan-sha256", required=True)
    count.add_argument("--expected-reservation-sha256", required=True)
    count.add_argument("--env-file", type=Path, default=Path(".env"))
    count.add_argument("--live", action="store_true")
    authorize = sub.add_parser("authorize")
    authorize.add_argument("--repository-root", type=Path, default=Path("."))
    authorize.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    authorize.add_argument("--expected-plan-sha256", required=True)
    authorize.add_argument("--expected-reservation-sha256", required=True)
    authorize.add_argument("--expected-count-certificate-sha256", required=True)
    authorize.add_argument("--phase-budget-usd-micros", type=int, required=True)
    execute = sub.add_parser("execute")
    execute.add_argument("--repository-root", type=Path, default=Path("."))
    execute.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    execute.add_argument("--expected-plan-sha256", required=True)
    execute.add_argument("--expected-generation-authorization-sha256", required=True)
    execute.add_argument("--env-file", type=Path, default=Path(".env"))
    execute.add_argument("--live", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("--repository-root", type=Path, default=Path("."))
    status.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    return parser


def _safe_live_environment_file(*, repository_root: Path, env_file: Path) -> Path:
    root = repository_root.resolve(strict=True)
    candidate = env_file if env_file.is_absolute() else root / env_file
    if candidate.is_symlink():
        raise HostedNativeNumericPilotError("hosted_numeric_live_env_symlink_forbidden")
    try:
        resolved = candidate.resolve(strict=True)
        metadata = candidate.stat(follow_symlinks=False)
    except (FileNotFoundError, OSError) as exc:
        raise HostedNativeNumericPilotError("hosted_numeric_live_env_missing_or_unsafe") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HostedNativeNumericPilotError("hosted_numeric_live_env_outside_repository") from exc
    if (
        resolved != candidate.absolute()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise HostedNativeNumericPilotError("hosted_numeric_live_env_not_regular_file")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise HostedNativeNumericPilotError("hosted_numeric_live_env_mode_must_be_0600")
    return resolved


def _parse_explicit_anthropic_api_key(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise HostedNativeNumericPilotError("hosted_numeric_live_env_invalid") from exc
    values: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        name, separator, value = line.partition("=")
        if separator and name.strip() == "ANTHROPIC_API_KEY":
            candidate = value.strip()
            if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {'"', "'"}:
                candidate = candidate[1:-1]
            values.append(candidate)
    if len(values) != 1 or not values[0]:
        raise HostedNativeNumericPilotError("hosted_numeric_anthropic_api_key_missing_or_duplicate")
    return values[0]


def _load_anthropic_api_key(*, repository_root: Path, env_file: Path) -> str:
    if os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("ANTHROPIC_CUSTOM_HEADERS"):
        raise HostedNativeNumericPilotError(
            "hosted_numeric_transport_environment_override_forbidden"
        )
    return _parse_explicit_anthropic_api_key(
        _safe_live_environment_file(
            repository_root=repository_root,
            env_file=env_file,
        )
    )


def _default_token_counter_factory(api_key: str) -> Any:
    return AnthropicHostedNativeNumericTokenCounterV1(api_key=api_key)


def _default_generation_client_factory(api_key: str) -> Any:
    return AnthropicHostedNativeNumericGenerationClientV1(api_key=api_key)


class _ForbiddenTokenCounter:
    def count_tokens(self, count_request: dict[str, object]) -> int:
        del count_request
        raise AssertionError("hosted_numeric_replay_attempted_token_count_transport")


class _ForbiddenGenerationClient:
    def generate(self, wire_request: dict[str, object]) -> Any:
        del wire_request
        raise AssertionError("hosted_numeric_replay_attempted_generation_transport")


def _summary(value: Any) -> dict[str, Any]:
    if hasattr(value, "plan_sha256") and hasattr(value, "surfaces"):
        return {
            "status": value.status,
            "plan_sha256": value.plan_sha256,
            "config_sha256": value.config_sha256,
            "question_config_sha256": value.question_config_sha256,
            "source_manifest_sha256": value.source_manifest_sha256,
            "pipeline_fingerprint_sha256": value.pipeline_fingerprint_sha256,
            "provider_identity_sha256": value.provider_identity.identity_sha256,
            "canary_success_binding_sha256": value.canary_success_binding_sha256,
            "canary_terminal_sha256": value.canary_terminal_sha256,
            "canary_terminal_artifact_sha256": value.canary_terminal_artifact_sha256,
            "request_keys": [item.intent.request_key for item in value.surfaces],
            "generation_schema_sha256s": {
                item.intent.request_key: item.generation_schema.schema_sha256
                for item in value.surfaces
            },
            "wire_optional_parameter_counts": {
                item.intent.request_key: item.compiled_schema.wire_optional_parameter_count
                for item in value.surfaces
            },
            "wire_union_parameter_counts": {
                item.intent.request_key: item.compiled_schema.wire_union_parameter_count
                for item in value.surfaces
            },
            "transport_mode": value.transport_mode,
            "provider_grammar_enabled": value.provider_grammar_enabled,
            "delivered_schema_structural_metrics": {
                item.intent.request_key: item.schema_structural_metrics.model_dump(mode="json")
                for item in value.surfaces
            },
            "offline_known_liability_usd_micros": sum(
                item.offline_known_request_liability_usd_micros for item in value.surfaces
            ),
            "maximum_new_liability_usd_micros": (value.maximum_new_liability_usd_micros),
            "provider_calls_made": False,
            "labels_opened": False,
            "claim_release_authority": False,
        }
    if hasattr(value, "certificate_sha256"):
        return {
            "certificate_sha256": value.certificate_sha256,
            "plan_sha256": value.plan_sha256,
            "counted_input_tokens": {
                item.request_key: item.counted_input_tokens for item in value.receipts
            },
            "certified_request_liability_usd_micros": {
                item.request_key: item.certified_request_liability_usd_micros
                for item in value.receipts
            },
            "certified_total_liability_usd_micros": (value.certified_total_liability_usd_micros),
            "generation_calls_made": 0,
        }
    if hasattr(value, "authorization_sha256"):
        return {
            "authorization_sha256": value.authorization_sha256,
            "plan_sha256": value.plan_sha256,
            "certified_maximum_liability_usd_micros": (
                value.certified_maximum_liability_usd_micros
            ),
            "full_new_liability_reservation_usd_micros": (
                value.full_new_liability_reservation_usd_micros
            ),
            "maximum_provider_attempts": value.maximum_provider_attempts,
        }
    if hasattr(value, "reservation_sha256"):
        return {
            "reservation_sha256": value.reservation_sha256,
            "plan_sha256": value.plan_sha256,
            "reconciled_project_liability_usd_micros": (
                value.reconciled_project_liability_usd_micros
            ),
            "new_liability_reserved_usd_micros": (value.new_liability_reserved_usd_micros),
            "companion_canary_reserved_usd_micros": (value.companion_canary_reserved_usd_micros),
            "combined_v4_phase_reserved_usd_micros": (value.combined_v4_phase_reserved_usd_micros),
            "project_liability_after_reservation_usd_micros": (
                value.project_liability_after_reservation_usd_micros
            ),
        }
    if hasattr(value, "run") and hasattr(value, "terminal"):
        return {
            "status": value.terminal.status,
            "terminal_sha256": value.terminal.terminal_sha256,
            "hosted_run_sha256": value.run.run_sha256,
            "completed_native_extractions": (value.terminal.completed_native_extractions),
            "failed_or_ambiguous_extractions": (value.terminal.failed_or_ambiguous_extractions),
            "release_grade_native_numeric_yield": (
                value.terminal.release_grade_native_numeric_yield
            ),
            "observed_generation_cost_usd_micros": (
                value.terminal.observed_generation_cost_usd_micros
            ),
            "generation_liability_accounted_usd_micros": (
                value.terminal.generation_liability_accounted_usd_micros
            ),
            "combined_v4_certified_liability_usd_micros": (
                value.terminal.combined_v4_certified_liability_usd_micros
            ),
            "combined_v4_liability_accounted_usd_micros": (
                value.terminal.combined_v4_liability_accounted_usd_micros
            ),
            "provider_usage_missing_count": value.terminal.provider_usage_missing_count,
            "request_budget_breach_count": value.terminal.request_budget_breach_count,
            "scientific_budget_breach_detected": (value.terminal.scientific_budget_breach_detected),
            "canary_success_binding_sha256": (value.terminal.canary_success_binding_sha256),
            "canary_terminal_sha256": value.terminal.canary_terminal_sha256,
            "canary_terminal_artifact_sha256": (value.terminal.canary_terminal_artifact_sha256),
            "claim_release_authority": False,
        }
    return {
        "status": value.status,
        "terminal_sha256": value.terminal_sha256,
        "hosted_run_sha256": value.hosted_run_sha256,
        "release_grade_native_numeric_yield": value.release_grade_native_numeric_yield,
        "observed_generation_cost_usd_micros": value.observed_generation_cost_usd_micros,
        "generation_liability_accounted_usd_micros": (
            value.generation_liability_accounted_usd_micros
        ),
        "combined_v4_certified_liability_usd_micros": (
            value.combined_v4_certified_liability_usd_micros
        ),
        "combined_v4_liability_accounted_usd_micros": (
            value.combined_v4_liability_accounted_usd_micros
        ),
        "provider_usage_missing_count": value.provider_usage_missing_count,
        "request_budget_breach_count": value.request_budget_breach_count,
        "scientific_budget_breach_detected": value.scientific_budget_breach_detected,
        "canary_success_binding_sha256": value.canary_success_binding_sha256,
        "canary_terminal_sha256": value.canary_terminal_sha256,
        "canary_terminal_artifact_sha256": value.canary_terminal_artifact_sha256,
        "claim_release_authority": False,
    }


def main(
    argv: list[str] | None = None,
    *,
    token_counter_factory: Callable[[str], Any] | None = None,
    generation_client_factory: Callable[[str], Any] | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        value = freeze_hosted_native_numeric_pilot_plan_v1(
            repository_root=args.repository_root,
            expected_canary_terminal_sha256=args.expected_canary_terminal_sha256,
            canary_workspace=args.canary_workspace,
            config_path=args.config,
        )
    elif args.command == "prepare":
        value = prepare_hosted_native_numeric_pilot_v1(
            repository_root=args.repository_root,
            expected_canary_terminal_sha256=args.expected_canary_terminal_sha256,
            canary_workspace=args.canary_workspace,
            workspace=args.workspace,
            config_path=args.config,
        )
    elif args.command == "reserve":
        value = reserve_hosted_native_numeric_pilot_v1(
            workspace=args.workspace,
            expected_plan_sha256=args.expected_plan_sha256,
        )
    elif args.command == "count":
        if not args.live:
            parser.error("count requires --live; no network call was made")
        transport_required = preflight_hosted_native_numeric_count_v1(
            repository_root=args.repository_root,
            workspace=args.workspace,
            expected_plan_sha256=args.expected_plan_sha256,
            expected_reservation_sha256=args.expected_reservation_sha256,
        )
        if transport_required:
            api_key = _load_anthropic_api_key(
                repository_root=args.repository_root,
                env_file=args.env_file,
            )
            factory = token_counter_factory or _default_token_counter_factory
            counter = factory(api_key)
        else:
            counter = _ForbiddenTokenCounter()
        value = count_hosted_native_numeric_pilot_tokens_v1(
            repository_root=args.repository_root,
            workspace=args.workspace,
            expected_plan_sha256=args.expected_plan_sha256,
            expected_reservation_sha256=args.expected_reservation_sha256,
            counter=counter,
        )
    elif args.command == "authorize":
        value = authorize_hosted_native_numeric_pilot_v1(
            repository_root=args.repository_root,
            workspace=args.workspace,
            expected_plan_sha256=args.expected_plan_sha256,
            expected_reservation_sha256=args.expected_reservation_sha256,
            expected_count_certificate_sha256=(args.expected_count_certificate_sha256),
            phase_budget_usd_micros=args.phase_budget_usd_micros,
        )
    elif args.command == "execute":
        if not args.live:
            parser.error("execute requires --live; no provider call was made")
        transport_required = preflight_hosted_native_numeric_execution_v1(
            repository_root=args.repository_root,
            workspace=args.workspace,
            expected_plan_sha256=args.expected_plan_sha256,
            expected_generation_authorization_sha256=(
                args.expected_generation_authorization_sha256
            ),
        )
        if transport_required:
            api_key = _load_anthropic_api_key(
                repository_root=args.repository_root,
                env_file=args.env_file,
            )
            factory = generation_client_factory or _default_generation_client_factory
            client = factory(api_key)
        else:
            client = _ForbiddenGenerationClient()
        value = execute_hosted_native_numeric_pilot_v1(
            repository_root=args.repository_root,
            workspace=args.workspace,
            expected_plan_sha256=args.expected_plan_sha256,
            expected_generation_authorization_sha256=(
                args.expected_generation_authorization_sha256
            ),
            client=client,
        )
    else:
        try:
            value = load_hosted_native_numeric_terminal_v1(
                repository_root=args.repository_root,
                workspace=args.workspace,
            )
        except HostedNativeNumericPilotError as exc:
            if str(exc) != "hosted_numeric_terminal_report_required":
                raise
            value = load_hosted_native_numeric_pilot_plan_v1(workspace=args.workspace)
    print(json.dumps(_summary(value), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
