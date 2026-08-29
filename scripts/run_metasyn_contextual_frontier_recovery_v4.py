#!/usr/bin/env python3
"""Run the one-shot completed-only recovery-v4 lifecycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.metasyn_contextual_frontier_recovery_v4 import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_WORKSPACE,
    authorize_metasyn_contextual_frontier_recovery_v4,
    default_metasyn_contextual_frontier_recovery_client_v4,
    execute_metasyn_contextual_frontier_recovery_v4,
    freeze_metasyn_contextual_frontier_recovery_plan_v4,
    load_metasyn_contextual_frontier_recovery_plan_v4,
    prepare_metasyn_contextual_frontier_recovery_v4,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "prepare"):
        item = sub.add_parser(command)
        item.add_argument("--repository-root", type=Path, default=Path("."))
        item.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
        if command == "prepare":
            item.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    authorize = sub.add_parser("authorize")
    authorize.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    authorize.add_argument("--phase-budget-usd-micros", type=int, required=True)
    execute = sub.add_parser("execute")
    execute.add_argument("--repository-root", type=Path, default=Path("."))
    execute.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    return parser


def _summary(value: Any) -> dict[str, Any]:
    if hasattr(value, "terminal_sha256"):
        return {
            "status": value.status,
            "terminal_sha256": value.terminal_sha256,
            "provider_attempt_count_upper_bound": 1,
            "claim_release_authority": False,
        }
    if hasattr(value, "authorization_sha256"):
        return {
            "authorization_sha256": value.authorization_sha256,
            "maximum_provider_attempts": 1,
            "cost_ceiling_usd_micros": value.request_cost_ceiling_usd_micros,
            "claim_release_authority": False,
        }
    return {
        "status": value.status,
        "plan_sha256": value.plan_sha256,
        "request_sha256": value.request_sha256,
        "prompt_sha256": value.request.prompt_sha256,
        "wire_schema_sha256": value.wire_schema_sha256,
        "wire_schema_utf8_bytes": value.wire_schema_utf8_bytes,
        "wire_schema_property_slots": value.wire_schema_property_slots,
        "wire_schema_enum_values": value.wire_schema_enum_values,
        "wire_schema_union_keywords": value.wire_schema_union_keywords,
        "maximum_provider_calls": 1,
        "provider_calls_made": 0,
        "hard_cost_liability_usd_micros": value.hard_cost_liability_usd_micros,
        "claim_release_authority": False,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.command == "plan":
        value = freeze_metasyn_contextual_frontier_recovery_plan_v4(
            repository_root=args.repository_root, config_path=args.config
        )
    elif args.command == "prepare":
        value = prepare_metasyn_contextual_frontier_recovery_v4(
            repository_root=args.repository_root,
            workspace=args.workspace,
            config_path=args.config,
        )
    elif args.command == "authorize":
        value = authorize_metasyn_contextual_frontier_recovery_v4(
            workspace=args.workspace,
            phase_budget_usd_micros=args.phase_budget_usd_micros,
        )
    else:
        plan = load_metasyn_contextual_frontier_recovery_plan_v4(workspace=args.workspace)
        value = execute_metasyn_contextual_frontier_recovery_v4(
            repository_root=args.repository_root,
            workspace=args.workspace,
            client=default_metasyn_contextual_frontier_recovery_client_v4(plan),
        )
    print(json.dumps(_summary(value), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
