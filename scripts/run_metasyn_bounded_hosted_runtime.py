#!/usr/bin/env python3
"""Run the immutable hosted Anthropic MetaSyn bounded-yield diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from literature_multiverse.anthropic_bounded_generation import AnthropicBoundedClient
from literature_multiverse.metasyn_bounded_hosted_runtime import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_EXECUTION_WORKSPACE,
    DEFAULT_PILOT_WORKSPACE,
    MAX_PROMPT_JSON_PROVIDER_CALLS,
    MAX_STRUCTURED_PROVIDER_CALLS,
    MetaSynHostedRuntimeError,
    finalize_metasyn_hosted_runtime,
    load_current_metasyn_hosted_execution_bundle,
    prepare_metasyn_hosted_runtime,
    run_metasyn_hosted_full_roster,
    run_metasyn_hosted_preflight,
    run_metasyn_hosted_smoke,
    validate_finalized_metasyn_hosted_runtime,
    write_metasyn_hosted_execution_bundle,
)
from literature_multiverse.providers import load_live_environment


def _rooted(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _add_workspace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--workspace", type=Path, default=DEFAULT_EXECUTION_WORKSPACE)


def _add_anchor(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-execution-bundle-sha256", required=True)


def _add_live(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--live",
        action="store_true",
        help="explicitly authorize provider calls for this command",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="optional mode-0600 environment file, loaded only under --live",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help=(
            "externally replay the private adapter and freeze code/config/provider/"
            "schema/rate identity without reading credentials or making calls"
        ),
    )
    _add_workspace(prepare)
    prepare.add_argument("--pilot-workspace", type=Path, default=DEFAULT_PILOT_WORKSPACE)
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)

    validate_bundle = subparsers.add_parser(
        "validate-bundle",
        help="externally replay the current code, config, adapter, provider identity, and roster",
    )
    _add_workspace(validate_bundle)

    preflight = subparsers.add_parser(
        "preflight",
        help=(
            "freeze the 296-call/$20 authorization, materialize the exact eight-request "
            "source-free mixed-transport roster, and make zero to eight new provider "
            "calls depending on immutable resume state"
        ),
    )
    _add_workspace(preflight)
    _add_anchor(preflight)
    _add_live(preflight)

    smoke = subparsers.add_parser(
        "smoke",
        help=(
            "run the one prespecified source row; gate only transport, schema, grounding, "
            "and terminal status"
        ),
    )
    _add_workspace(smoke)
    _add_anchor(smoke)
    _add_live(smoke)

    full = subparsers.add_parser(
        "full-roster",
        help="resume the fixed sequential 32-row roster without repeating any intent",
    )
    _add_workspace(full)
    _add_anchor(full)
    _add_live(full)

    finalize = subparsers.add_parser(
        "finalize",
        help="freeze the private hosted ledger and yield-only report; no public artifact",
    )
    _add_workspace(finalize)
    _add_anchor(finalize)

    validate_final = subparsers.add_parser(
        "validate-final",
        help="externally replay every current request, receipt, row, aggregate, and hash",
    )
    _add_workspace(validate_final)
    _add_anchor(validate_final)
    return parser


def _live_client(args: argparse.Namespace, *, root: Path) -> AnthropicBoundedClient:
    if not args.live:
        raise MetaSynHostedRuntimeError("metasyn_hosted_live_flag_required")
    load_live_environment(_rooted(args.env_file, root), live_enabled=True)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise MetaSynHostedRuntimeError("metasyn_hosted_anthropic_api_key_missing_pre_call")
    _, bundle = load_current_metasyn_hosted_execution_bundle(
        workspace=_rooted(args.workspace, root),
        repository_root=root,
        external_replay=True,
    )
    return AnthropicBoundedClient(bundle.anthropic_config)


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    bundle = prepare_metasyn_hosted_runtime(
        repository_root=root,
        pilot_workspace=_rooted(args.pilot_workspace, root),
        config_path=_rooted(args.config, root),
    )
    output = write_metasyn_hosted_execution_bundle(
        execution_bundle=bundle,
        workspace=_rooted(args.workspace, root),
        repository_root=root,
    )
    return {
        "stage": "prepare",
        "status": bundle.status,
        "execution_bundle_path": output.relative_to(root).as_posix(),
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
        "provider_identity_sha256": bundle.provider_identity_sha256,
        "provider_pricing_table_sha256": bundle.anthropic_config.pricing_table_sha256,
        "publication_count": bundle.publication_count,
        "maximum_theoretical_provider_calls": bundle.maximum_theoretical_provider_calls,
        "maximum_structured_json_schema_calls": MAX_STRUCTURED_PROVIDER_CALLS,
        "maximum_prompt_json_schema_calls": MAX_PROMPT_JSON_PROVIDER_CALLS,
        "configured_cost_ceiling_usd_micros": (
            bundle.maximum_authorized_cost_usd_micros
        ),
        "operator_authorized_source_transmission": True,
        "provider_calls_made": False,
    }


def _validate_bundle(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    _, bundle = load_current_metasyn_hosted_execution_bundle(
        workspace=_rooted(args.workspace, root),
        repository_root=root,
        external_replay=True,
    )
    return {
        "stage": "validate-bundle",
        "status": "valid_current_external_replay",
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
        "publication_count": bundle.publication_count,
        "maximum_theoretical_provider_calls": bundle.maximum_theoretical_provider_calls,
        "maximum_structured_json_schema_calls": MAX_STRUCTURED_PROVIDER_CALLS,
        "maximum_prompt_json_schema_calls": MAX_PROMPT_JSON_PROVIDER_CALLS,
        "configured_cost_ceiling_usd_micros": (
            bundle.maximum_authorized_cost_usd_micros
        ),
        "provider_calls_made": False,
    }


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    workspace = _rooted(args.workspace, root)
    request_keys = {f"preflight-{index:02d}" for index in range(8)}

    def snapshot() -> tuple[set[str], dict[str, str]]:
        receipts_directory = workspace / "call-receipts"
        incidents_directory = workspace / "call-incidents"
        receipts = {
            path.stem
            for path in receipts_directory.glob("preflight-*.json")
            if path.stem in request_keys and path.is_file() and not path.is_symlink()
        }
        incidents: dict[str, str] = {}
        for path in incidents_directory.glob("preflight-*.json"):
            if path.stem not in request_keys or not path.is_file() or path.is_symlink():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            kind = payload.get("incident_kind") if isinstance(payload, dict) else None
            if isinstance(kind, str):
                incidents[path.stem] = kind
        return receipts, incidents

    before_receipts, before_incidents = snapshot()
    receipt = run_metasyn_hosted_preflight(
        workspace=workspace,
        repository_root=root,
        client=_live_client(args, root=root),
        expected_execution_bundle_sha256=args.expected_execution_bundle_sha256,
    )
    after_receipts, after_incidents = snapshot()
    new_receipts = after_receipts - before_receipts
    new_incidents = set(after_incidents) - set(before_incidents)
    new_provider_call_incidents = {
        key
        for key in new_incidents
        if after_incidents[key] == "provider_call_raised_after_durable_intent"
    }
    new_orphan_incidents = {
        key
        for key in new_incidents
        if after_incidents[key] == "orphan_intent_observed_on_resume"
    }
    return {
        "stage": "preflight",
        "status": receipt.status,
        "preflight_sha256": receipt.preflight_sha256,
        "passed_call_count": receipt.passed_call_count,
        "observed_provider_calls": receipt.observed_provider_calls,
        "exact_request_roster_size": 8,
        "new_provider_call_attempts_this_invocation": (
            len(new_receipts) + len(new_provider_call_incidents)
        ),
        "reused_terminal_outcomes_this_invocation": len(
            before_receipts | set(before_incidents)
        ),
        "new_terminal_incidents_this_invocation": len(new_incidents),
        "new_orphan_incidents_this_invocation": len(new_orphan_incidents),
        "possible_ambiguous_provider_calls": (
            receipt.possible_ambiguous_provider_calls
        ),
        "structured_json_schema_calls": receipt.structured_json_schema_calls,
        "prompt_json_schema_calls": receipt.prompt_json_schema_calls,
        "maximum_structured_json_schema_calls": MAX_STRUCTURED_PROVIDER_CALLS,
        "maximum_prompt_json_schema_calls": MAX_PROMPT_JSON_PROVIDER_CALLS,
        "transport_mode_policy": (
            "inventory-structured-json-schema-packet-prompt-json-schema-v1"
        ),
        "cost_authorization_sha256": receipt.cost_authorization_sha256,
        "observed_request_ceiling_usd_micros": (
            receipt.observed_request_ceiling_usd_micros
        ),
        "possible_ambiguous_charge_ceiling_usd_micros": (
            receipt.possible_ambiguous_charge_ceiling_usd_micros
        ),
        "durable_intent_liability_usd_micros": (
            receipt.durable_intent_liability_usd_micros
        ),
        "cost_authorization_ceiling_usd_micros": (
            receipt.cost_authorization_ceiling_usd_micros
        ),
        "configured_cost_ceiling_usd_micros": (
            receipt.configured_cost_ceiling_usd_micros
        ),
        "durable_intent_roster_sha256": receipt.durable_intent_roster_sha256,
        "source_bearing_provider_calls": 0,
        "application_retries": 0,
        "sdk_retries": 0,
    }


def _smoke(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    receipt = run_metasyn_hosted_smoke(
        workspace=_rooted(args.workspace, root),
        repository_root=root,
        client=_live_client(args, root=root),
        expected_execution_bundle_sha256=args.expected_execution_bundle_sha256,
    )
    return {
        "stage": "smoke",
        "status": receipt.status,
        "smoke_sha256": receipt.smoke_sha256,
        "row_status": receipt.row_status,
        "gate_dimensions": receipt.gate_dimensions,
        "scientific_correctness_evaluated": False,
    }


def _full(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    ledger = run_metasyn_hosted_full_roster(
        workspace=_rooted(args.workspace, root),
        repository_root=root,
        client=_live_client(args, root=root),
        expected_execution_bundle_sha256=args.expected_execution_bundle_sha256,
    )
    return {
        "stage": "full-roster",
        "status": ledger.status,
        "hosted_ledger_sha256": ledger.ledger_sha256,
        "publication_count": ledger.publication_count,
        "observed_source_provider_calls": ledger.observed_source_provider_calls,
        "possible_ambiguous_source_provider_calls": (
            ledger.possible_ambiguous_source_provider_calls
        ),
        "total_provider_call_attempts_or_possible_attempts": (
            ledger.total_provider_call_attempts_or_possible_attempts
        ),
        "structured_json_schema_calls": ledger.structured_json_schema_calls,
        "prompt_json_schema_calls": ledger.prompt_json_schema_calls,
        "maximum_structured_json_schema_calls": (
            ledger.maximum_structured_json_schema_calls
        ),
        "maximum_prompt_json_schema_calls": ledger.maximum_prompt_json_schema_calls,
        "transport_mode_policy": ledger.transport_mode_policy,
        "cost_authorization_sha256": ledger.cost_authorization_sha256,
        "observed_request_ceiling_usd_micros": (
            ledger.observed_request_ceiling_usd_micros
        ),
        "possible_ambiguous_charge_ceiling_usd_micros": (
            ledger.possible_ambiguous_charge_ceiling_usd_micros
        ),
        "durable_intent_count": ledger.durable_intent_count,
        "durable_intent_liability_usd_micros": (
            ledger.durable_intent_liability_usd_micros
        ),
        "cost_authorization_ceiling_usd_micros": (
            ledger.cost_authorization_ceiling_usd_micros
        ),
        "configured_cost_ceiling_usd_micros": (
            ledger.configured_cost_ceiling_usd_micros
        ),
        "durable_intent_roster_sha256": ledger.durable_intent_roster_sha256,
    }


def _finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    report = finalize_metasyn_hosted_runtime(
        workspace=_rooted(args.workspace, root),
        repository_root=root,
        expected_execution_bundle_sha256=args.expected_execution_bundle_sha256,
    )
    return {
        "stage": "finalize",
        "status": report.status,
        "private_report_sha256": report.report_sha256,
        "provider_neutral_yield_report_sha256": (report.provider_neutral_yield_report_sha256),
        "typed_publication_output_count": report.typed_publication_output_count,
        "typed_finding_count": report.typed_finding_count,
        "structured_json_schema_calls": report.structured_json_schema_calls,
        "prompt_json_schema_calls": report.prompt_json_schema_calls,
        "maximum_structured_json_schema_calls": (
            report.maximum_structured_json_schema_calls
        ),
        "maximum_prompt_json_schema_calls": report.maximum_prompt_json_schema_calls,
        "transport_mode_policy": report.transport_mode_policy,
        "cost_authorization_sha256": report.cost_authorization_sha256,
        "observed_request_ceiling_usd_micros": (
            report.observed_request_ceiling_usd_micros
        ),
        "possible_ambiguous_charge_ceiling_usd_micros": (
            report.possible_ambiguous_charge_ceiling_usd_micros
        ),
        "durable_intent_count": report.durable_intent_count,
        "durable_intent_liability_usd_micros": (
            report.durable_intent_liability_usd_micros
        ),
        "cost_authorization_ceiling_usd_micros": (
            report.cost_authorization_ceiling_usd_micros
        ),
        "configured_cost_ceiling_usd_micros": (
            report.configured_cost_ceiling_usd_micros
        ),
        "durable_intent_roster_sha256": report.durable_intent_roster_sha256,
        "public_artifact_materialized": False,
        "claim_release_authority": False,
    }


def _validate_final(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    report = validate_finalized_metasyn_hosted_runtime(
        workspace=_rooted(args.workspace, root),
        repository_root=root,
        expected_execution_bundle_sha256=args.expected_execution_bundle_sha256,
    )
    return {
        "stage": "validate-final",
        "status": "valid_full_private_external_replay",
        "private_report_sha256": report.report_sha256,
        "publication_count": report.publication_count,
        "structured_json_schema_calls": report.structured_json_schema_calls,
        "prompt_json_schema_calls": report.prompt_json_schema_calls,
        "maximum_structured_json_schema_calls": (
            report.maximum_structured_json_schema_calls
        ),
        "maximum_prompt_json_schema_calls": report.maximum_prompt_json_schema_calls,
        "transport_mode_policy": report.transport_mode_policy,
        "cost_authorization_sha256": report.cost_authorization_sha256,
        "observed_request_ceiling_usd_micros": (
            report.observed_request_ceiling_usd_micros
        ),
        "possible_ambiguous_charge_ceiling_usd_micros": (
            report.possible_ambiguous_charge_ceiling_usd_micros
        ),
        "durable_intent_count": report.durable_intent_count,
        "durable_intent_liability_usd_micros": (
            report.durable_intent_liability_usd_micros
        ),
        "cost_authorization_ceiling_usd_micros": (
            report.cost_authorization_ceiling_usd_micros
        ),
        "configured_cost_ceiling_usd_micros": (
            report.configured_cost_ceiling_usd_micros
        ),
        "durable_intent_roster_sha256": report.durable_intent_roster_sha256,
        "public_artifact_materialized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = _prepare(args)
    elif args.command == "validate-bundle":
        result = _validate_bundle(args)
    elif args.command == "preflight":
        result = _preflight(args)
    elif args.command == "smoke":
        result = _smoke(args)
    elif args.command == "full-roster":
        result = _full(args)
    elif args.command == "finalize":
        result = _finalize(args)
    else:
        result = _validate_final(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
