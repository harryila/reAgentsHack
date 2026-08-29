#!/usr/bin/env python3
"""Prepare and execute one source-free prompt-JSON canary, at most once."""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from literature_multiverse.hosted_native_numeric_canary_v4 import (
    DEFAULT_WORKSPACE,
    AnthropicFablePromptJsonCanaryClientV4,
    HostedNativeNumericCanarySuccessV4,
    HostedNativeNumericCanaryV4Error,
    execute_hosted_native_numeric_canary_v4,
    freeze_hosted_native_numeric_canary_plan_v4,
    load_hosted_native_numeric_canary_status_v4,
    load_successful_hosted_native_numeric_canary_v4,
    preflight_hosted_native_numeric_canary_execution_v4,
    prepare_hosted_native_numeric_canary_v4,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Source-free, prompt-JSON Fable transport canary for numeric pilot v4."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--execution-id")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    prepare.add_argument("--execution-id")
    execute = commands.add_parser("execute")
    execute.add_argument("--repository-root", type=Path, default=Path("."))
    execute.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    execute.add_argument("--expected-plan-sha256", required=True)
    execute.add_argument("--expected-authorization-sha256", required=True)
    execute.add_argument("--env-file", type=Path, default=Path(".env"))
    execute.add_argument("--live", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    return parser


def _safe_env_file(*, repository_root: Path, env_file: Path) -> Path:
    root = repository_root.resolve(strict=True)
    candidate = env_file if env_file.is_absolute() else root / env_file
    if candidate.is_symlink():
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_live_env_symlink_forbidden"
        )
    try:
        resolved = candidate.resolve(strict=True)
        metadata = candidate.stat(follow_symlinks=False)
    except OSError as exc:
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_live_env_missing_or_unsafe"
        ) from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_live_env_outside_repository"
        ) from exc
    if (
        resolved != candidate.absolute()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_live_env_missing_or_unsafe"
        )
    return resolved


def _parse_explicit_api_key(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise HostedNativeNumericCanaryV4Error("hosted_numeric_canary_v4_live_env_invalid") from exc
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
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_anthropic_api_key_missing_or_duplicate"
        )
    return values[0]


def _load_api_key(*, repository_root: Path, env_file: Path) -> str:
    if os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("ANTHROPIC_CUSTOM_HEADERS"):
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_transport_environment_override_forbidden"
        )
    return _parse_explicit_api_key(
        _safe_env_file(repository_root=repository_root, env_file=env_file)
    )


class _ForbiddenReplayClient:
    def generate(self, wire_request: Mapping[str, Any]) -> Any:
        del wire_request
        raise AssertionError("hosted_numeric_canary_v4_replay_attempted_transport")


def _default_client_factory(api_key: str) -> Any:
    return AnthropicFablePromptJsonCanaryClientV4(api_key=api_key)


def _plan_summary(value: Any) -> dict[str, Any]:
    return {
        "status": value.status,
        "execution_id": value.execution_id,
        "plan_sha256": value.plan_sha256,
        "request_key": value.request_key,
        "request_sha256": value.request_sha256,
        "wire_request_sha256": value.wire_request_sha256,
        "compiled_schema_sha256": value.compiled_schema.compiled_schema_sha256,
        "wire_schema_sha256": value.compiled_schema.wire_schema_sha256,
        "delivered_schema_sha256": value.delivered_schema_sha256,
        "transport_mode": value.provider_config.transport_mode,
        "structured_grammar_enforced_by_provider": False,
        "output_format_present_in_call": False,
        "max_output_tokens": value.provider_config.max_output_tokens,
        "certified_request_liability_usd_micros": (value.certified_request_liability_usd_micros),
        "source_bearing": False,
        "provider_calls_made": False,
        "scientific_authority": False,
    }


def _terminal_summary(value: Any, *, workspace: Path) -> dict[str, Any]:
    summary = {
        "status": value.status,
        "execution_id": value.execution_id,
        "terminal_sha256": value.terminal_sha256,
        "request_sha256": value.request_sha256,
        "certified_request_liability_usd_micros": (value.certified_request_liability_usd_micros),
        "charged_cost_upper_bound_usd_micros": (value.charged_cost_upper_bound_usd_micros),
        "source_bearing": False,
        "scientific_authority": False,
    }
    if isinstance(value, HostedNativeNumericCanarySuccessV4):
        binding = load_successful_hosted_native_numeric_canary_v4(
            workspace=workspace,
            expected_terminal_sha256=value.terminal_sha256,
        )
        summary.update(
            {
                "provider": value.provider,
                "response_model": value.response_model,
                "transport_mode": value.transport_mode,
                "provider_result_sha256": value.provider_result_sha256,
                "terminal_artifact_sha256": binding.terminal_artifact_sha256,
                "binding_sha256": binding.binding_sha256,
                "observed_cost_usd_micros": value.observed_cost_usd_micros,
                "fixture_exact": True,
            }
        )
    else:
        summary["failure_code"] = value.failure_code
        summary["observed_cost_usd_micros"] = value.observed_cost_usd_micros
    return summary


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[[str], Any] | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        value = freeze_hosted_native_numeric_canary_plan_v4(execution_id=args.execution_id)
        summary = _plan_summary(value)
    elif args.command == "prepare":
        value, authorization = prepare_hosted_native_numeric_canary_v4(
            workspace=args.workspace,
            execution_id=args.execution_id,
        )
        summary = {
            **_plan_summary(value),
            "authorization_sha256": authorization.authorization_sha256,
            "authorization_durable_before_intent": True,
        }
    elif args.command == "execute":
        if not args.live:
            parser.error("execute requires --live; no provider call was made")
        transport_required = preflight_hosted_native_numeric_canary_execution_v4(
            workspace=args.workspace,
            expected_plan_sha256=args.expected_plan_sha256,
            expected_authorization_sha256=args.expected_authorization_sha256,
        )
        if transport_required:
            api_key = _load_api_key(
                repository_root=args.repository_root,
                env_file=args.env_file,
            )
            factory = client_factory or _default_client_factory
            client = factory(api_key)
            api_key = ""
        else:
            client = _ForbiddenReplayClient()
        value = execute_hosted_native_numeric_canary_v4(
            workspace=args.workspace,
            expected_plan_sha256=args.expected_plan_sha256,
            expected_authorization_sha256=args.expected_authorization_sha256,
            client=client,
        )
        summary = _terminal_summary(value, workspace=args.workspace)
    else:
        summary = load_hosted_native_numeric_canary_status_v4(workspace=args.workspace)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
