#!/usr/bin/env python3
"""Run the fresh exact-once passage-hosted MetaSyn yield diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from literature_multiverse.anthropic_bounded_generation import AnthropicBoundedClient
from literature_multiverse.metasyn_passage_hosted_bundle_v2 import (
    MetaSynPassageHostedExecutionBundleV2,
)
from literature_multiverse.metasyn_passage_hosted_runtime_v2 import (
    DEFAULT_EXECUTION_WORKSPACE,
    MetaSynPassageHostedRuntimeV2Error,
    authorize_metasyn_passage_hosted_runtime_v2,
    finalize_metasyn_passage_hosted_runtime_v2,
    freeze_metasyn_passage_packet_roster_v2,
    metasyn_passage_hosted_runtime_status_v2,
    prepare_metasyn_passage_hosted_runtime_v2,
    run_metasyn_passage_inventory_roster_v2,
    run_metasyn_passage_inventory_smoke_v2,
    run_metasyn_passage_packet_roster_v2,
    run_metasyn_passage_packet_smoke_v2,
    run_metasyn_passage_source_free_preflight_v2,
    validate_finalized_metasyn_passage_hosted_runtime_v2,
)
from literature_multiverse.providers import load_live_environment


def _rooted(value: Path, root: Path) -> Path:
    return value if value.is_absolute() else root / value


def _common(parser: argparse.ArgumentParser, *, anchor: bool = True) -> None:
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--workspace", type=Path, default=DEFAULT_EXECUTION_WORKSPACE)
    if anchor:
        parser.add_argument("--expected-execution-bundle-sha256", required=True)


def _live(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--live",
        action="store_true",
        help="explicitly authorize the already-frozen provider calls for this command",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    _common(prepare, anchor=False)
    authorize = sub.add_parser("authorize")
    _common(authorize)
    preflight = sub.add_parser("preflight")
    _common(preflight)
    _live(preflight)
    inventory_smoke = sub.add_parser("inventory-smoke")
    _common(inventory_smoke)
    _live(inventory_smoke)
    inventory = sub.add_parser("inventory-roster")
    _common(inventory)
    _live(inventory)
    packet_roster = sub.add_parser("freeze-packet-roster")
    _common(packet_roster)
    packet_smoke = sub.add_parser("packet-smoke")
    _common(packet_smoke)
    _live(packet_smoke)
    packet = sub.add_parser("packet-roster")
    _common(packet)
    _live(packet)
    finalize = sub.add_parser("finalize")
    _common(finalize)
    validate = sub.add_parser("validate-final")
    _common(validate)
    status = sub.add_parser("status")
    _common(status)
    return parser


def _client(args: argparse.Namespace, *, root: Path, workspace: Path) -> AnthropicBoundedClient:
    if not args.live:
        raise MetaSynPassageHostedRuntimeV2Error("metasyn_passage_runtime_v2_live_flag_required")
    load_live_environment(_rooted(args.env_file, root), live_enabled=True)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_anthropic_api_key_missing"
        )
    bundle_path = workspace / "execution-bundle.json"
    bundle = MetaSynPassageHostedExecutionBundleV2.model_validate(
        json.loads(bundle_path.read_text(encoding="utf-8"))
    )
    return AnthropicBoundedClient(bundle.anthropic_config)


def _summary(value: Any) -> dict[str, Any]:
    payload = value.model_dump(mode="json")
    allowed = {
        key: item
        for key, item in payload.items()
        if key
        in {
            "status",
            "execution_bundle_sha256",
            "bundle_pipeline_sha256",
            "authorization_sha256",
            "receipt_sha256",
            "ledger_sha256",
            "roster_sha256",
            "smoke_sha256",
            "report_sha256",
            "validation_sha256",
            "request_count",
            "typed_effect_count",
            "packet_call_count",
            "total_provider_attempts_or_possible_attempts",
            "reported_estimated_cost_usd_micros",
            "conservative_attempt_liability_usd_micros",
            "claim_release_authority",
        }
    }
    allowed["artifact_type"] = type(value).__name__
    return allowed


def main() -> int:
    args = build_parser().parse_args()
    root = args.repository_root.resolve(strict=True)
    workspace = _rooted(args.workspace, root)
    kwargs = {
        "repository_root": root,
        "workspace": workspace,
    }
    if args.command == "prepare":
        value: Any = prepare_metasyn_passage_hosted_runtime_v2(**kwargs)
    else:
        anchored = {
            **kwargs,
            "expected_execution_bundle_sha256": args.expected_execution_bundle_sha256,
        }
        if args.command == "authorize":
            value = authorize_metasyn_passage_hosted_runtime_v2(**anchored)
        elif args.command == "preflight":
            value = run_metasyn_passage_source_free_preflight_v2(
                **anchored, client=_client(args, root=root, workspace=workspace)
            )
        elif args.command == "inventory-smoke":
            value = run_metasyn_passage_inventory_smoke_v2(
                **anchored, client=_client(args, root=root, workspace=workspace)
            )
        elif args.command == "inventory-roster":
            value = run_metasyn_passage_inventory_roster_v2(
                **anchored, client=_client(args, root=root, workspace=workspace)
            )
        elif args.command == "freeze-packet-roster":
            value = freeze_metasyn_passage_packet_roster_v2(**anchored)
        elif args.command == "packet-smoke":
            value = run_metasyn_passage_packet_smoke_v2(
                **anchored, client=_client(args, root=root, workspace=workspace)
            )
        elif args.command == "packet-roster":
            value = run_metasyn_passage_packet_roster_v2(
                **anchored, client=_client(args, root=root, workspace=workspace)
            )
        elif args.command == "finalize":
            value = finalize_metasyn_passage_hosted_runtime_v2(**anchored)
        elif args.command == "validate-final":
            value = validate_finalized_metasyn_passage_hosted_runtime_v2(**anchored)
        else:
            value = metasyn_passage_hosted_runtime_status_v2(**anchored)
    output = value if isinstance(value, dict) else _summary(value)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
