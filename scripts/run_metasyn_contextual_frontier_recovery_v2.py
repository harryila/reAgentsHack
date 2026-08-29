#!/usr/bin/env python3
"""Prepare, authorize, execute, and replay the one-shot recovery-v2 smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.metasyn_contextual_frontier_recovery_lifecycle_v2 import (
    DEFAULT_WORKSPACE,
    MetaSynContextualFrontierRecoveryClientV2,
    authorize_metasyn_contextual_frontier_recovery_lifecycle_v2,
    execute_metasyn_contextual_frontier_recovery_lifecycle_v2,
    load_metasyn_contextual_frontier_recovery_plan_v2,
    prepare_metasyn_contextual_frontier_recovery_lifecycle_v2,
    status_metasyn_contextual_frontier_recovery_lifecycle_v2,
    validate_metasyn_contextual_frontier_recovery_lifecycle_v2,
)
from literature_multiverse.metasyn_contextual_frontier_recovery_v2 import (
    DEFAULT_CONFIG_PATH,
    freeze_metasyn_contextual_frontier_recovery_plan_v2,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot, exact-once Fable 5 recovery of the prespecified row-17 "
            "500-mg/placebo grounding smoke."
        )
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
    if hasattr(value, "workspace_validation_sha256"):
        state = value.status
        return {
            "status": state.status,
            "workspace_validation_sha256": value.workspace_validation_sha256,
            "plan_sha256": state.plan_sha256,
            "lifecycle_pipeline_sha256": state.lifecycle_pipeline_sha256,
            "provider_result_count": state.provider_result_count,
            "provider_receipt_count": state.provider_receipt_count,
            "terminal_report_sha256": state.terminal_report_sha256,
            "external_plan_and_lifecycle_source_replayed": (
                value.external_plan_and_lifecycle_source_replayed
            ),
            "claim_release_authority": False,
        }
    if hasattr(value, "request") and hasattr(value, "plan_sha256"):
        return {
            "status": value.status,
            "plan_sha256": value.plan_sha256,
            "runtime_pipeline_sha256": value.runtime_pipeline_sha256,
            "model": value.config.model,
            "effort": value.config.effort,
            "request_key": value.request.request_key,
            "request_sha256": value.request_sha256,
            "target_spec_sha256": value.target_spec_sha256,
            "hard_cost_liability_usd_micros": value.hard_cost_liability_usd_micros,
            "provider_calls_made": value.provider_calls_made,
            "maximum_provider_calls": value.maximum_provider_calls,
            "claim_release_authority": False,
        }
    if hasattr(value, "maximum_provider_attempts"):
        return {
            "authorization_sha256": value.authorization_sha256,
            "plan_sha256": value.plan_sha256,
            "lifecycle_pipeline_sha256": value.lifecycle_pipeline_sha256,
            "maximum_provider_attempts": value.maximum_provider_attempts,
            "maximum_cost_liability_usd_micros": value.maximum_cost_liability_usd_micros,
            "provider_calls_made_before_authorization": 0,
            "claim_release_authority": False,
        }
    if hasattr(value, "status_sha256"):
        return {
            "status": value.status,
            "status_sha256": value.status_sha256,
            "plan_sha256": value.plan_sha256,
            "lifecycle_pipeline_sha256": value.lifecycle_pipeline_sha256,
            "intent_count": value.intent_count,
            "provider_result_count": value.provider_result_count,
            "provider_receipt_count": value.provider_receipt_count,
            "terminal_report_sha256": value.terminal_report_sha256,
            "claim_release_authority": False,
        }
    return {
        "status": value.status,
        "terminal_report_sha256": value.report_sha256,
        "plan_sha256": value.plan_sha256,
        "lifecycle_pipeline_sha256": value.lifecycle_pipeline_sha256,
        "attempted_request_keys": value.attempted_request_keys,
        "provider_attempt_count_upper_bound": value.provider_attempt_count_upper_bound,
        "provider_receipt_count": value.provider_receipt_count,
        "fresh_native_typed_graph_observed": value.fresh_native_typed_graph_observed,
        "claim_release_authority": False,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.command == "plan":
        value = freeze_metasyn_contextual_frontier_recovery_plan_v2(
            repository_root=args.repository_root,
            config_path=args.config,
        )
    elif args.command == "prepare":
        value = prepare_metasyn_contextual_frontier_recovery_lifecycle_v2(
            repository_root=args.repository_root,
            workspace=args.workspace,
            config_path=args.config,
        )
    elif args.command == "authorize":
        value = authorize_metasyn_contextual_frontier_recovery_lifecycle_v2(
            workspace=args.workspace,
            phase_budget_usd_micros=args.phase_budget_usd_micros,
        )
    elif args.command == "execute":
        plan = load_metasyn_contextual_frontier_recovery_plan_v2(workspace=args.workspace)
        client = MetaSynContextualFrontierRecoveryClientV2(plan.transport_profile_config)
        value = execute_metasyn_contextual_frontier_recovery_lifecycle_v2(
            workspace=args.workspace,
            client=client,
        )
    elif args.command == "validate":
        value = validate_metasyn_contextual_frontier_recovery_lifecycle_v2(
            repository_root=args.repository_root,
            workspace=args.workspace,
            external_replay=not args.contract_only,
        )
    else:
        value = status_metasyn_contextual_frontier_recovery_lifecycle_v2(
            workspace=args.workspace,
        )
    print(json.dumps(_summary(value), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
