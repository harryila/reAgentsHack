#!/usr/bin/env python3
"""Prepare, authorize, execute, and replay the bounded Fable 5 smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.metasyn_contextual_frontier_runtime_v1 import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_WORKSPACE,
    MetaSynContextualFrontierClientV1,
    authorize_metasyn_contextual_frontier_runtime_v1,
    execute_metasyn_contextual_frontier_runtime_v1,
    freeze_metasyn_contextual_frontier_plan_v1,
    load_metasyn_contextual_frontier_plan_v1,
    prepare_metasyn_contextual_frontier_runtime_v1,
    validate_metasyn_contextual_frontier_runtime_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded exact-once Fable 5 contextual-grounding smoke."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "prepare"):
        item = sub.add_parser(name)
        item.add_argument("--repository-root", type=Path, default=Path("."))
        item.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
        if name == "prepare":
            item.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    authorize = sub.add_parser("authorize")
    authorize.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    authorize.add_argument("--phase-budget-usd-micros", type=int, required=True)
    execute = sub.add_parser("execute")
    execute.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    validate = sub.add_parser("validate")
    validate.add_argument("--repository-root", type=Path, default=Path("."))
    validate.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    validate.add_argument("--contract-only", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    return parser


def _summary(value: Any) -> dict[str, Any]:
    if hasattr(value, "plan"):
        plan = value.plan
        terminal = value.terminal_report
        return {
            "status": value.status,
            "workspace_validation_sha256": value.workspace_validation_sha256,
            "plan_sha256": plan.plan_sha256,
            "runtime_pipeline_sha256": plan.runtime_pipeline_sha256,
            "model": plan.provider_identity.model,
            "effort": plan.provider_identity.effort,
            "transport_mode": plan.provider_identity.transport_mode,
            "intent_count": value.intent_count,
            "provider_receipt_count": value.provider_receipt_count,
            "terminal_status": terminal.status if terminal else None,
            "successful_request_key": terminal.successful_request_key if terminal else None,
            "claim_release_authority": False,
        }
    if hasattr(value, "roster"):
        return {
            "status": value.status,
            "plan_sha256": value.plan_sha256,
            "runtime_pipeline_sha256": value.runtime_pipeline_sha256,
            "model": value.provider_identity.model,
            "effort": value.provider_identity.effort,
            "transport_mode": value.provider_identity.transport_mode,
            "request_keys": [item.request.request_key for item in value.roster],
            "total_cost_ceiling_usd_micros": value.total_cost_ceiling_usd_micros,
            "diagnostic_known_input_token_ceiling_total": (
                value.diagnostic_known_input_token_ceiling_total
            ),
            "diagnostic_known_surface_cost_usd_micros_total": (
                value.diagnostic_known_surface_cost_usd_micros_total
            ),
            "provider_calls_made": value.provider_calls_made,
            "claim_release_authority": False,
        }
    if hasattr(value, "authorized_calls"):
        return {
            "authorization_sha256": value.authorization_sha256,
            "authorized_call_count": value.authorized_call_count,
            "maximum_cost_liability_usd_micros": (value.maximum_cost_liability_usd_micros),
            "provider_calls_made_before_authorization": 0,
        }
    return {
        "status": value.status,
        "report_sha256": value.report_sha256,
        "attempted_request_keys": value.attempted_request_keys,
        "unattempted_request_keys": value.unattempted_request_keys,
        "successful_request_key": value.successful_request_key,
        "claim_release_authority": False,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.command == "plan":
        value = freeze_metasyn_contextual_frontier_plan_v1(
            repository_root=args.repository_root, config_path=args.config
        )
    elif args.command == "prepare":
        value = prepare_metasyn_contextual_frontier_runtime_v1(
            repository_root=args.repository_root,
            workspace=args.workspace,
            config_path=args.config,
        )
    elif args.command == "authorize":
        value = authorize_metasyn_contextual_frontier_runtime_v1(
            workspace=args.workspace,
            phase_budget_usd_micros=args.phase_budget_usd_micros,
        )
    elif args.command == "execute":
        plan = load_metasyn_contextual_frontier_plan_v1(workspace=args.workspace)
        client = MetaSynContextualFrontierClientV1(plan.config)
        value = execute_metasyn_contextual_frontier_runtime_v1(
            workspace=args.workspace, client=client
        )
    elif args.command == "validate":
        value = validate_metasyn_contextual_frontier_runtime_v1(
            repository_root=args.repository_root,
            workspace=args.workspace,
            external_replay=not args.contract_only,
        )
    else:
        value = validate_metasyn_contextual_frontier_runtime_v1(
            repository_root=Path("."),
            workspace=args.workspace,
            external_replay=False,
        )
    print(json.dumps(_summary(value), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
