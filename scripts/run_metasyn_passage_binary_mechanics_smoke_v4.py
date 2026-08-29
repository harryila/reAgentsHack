#!/usr/bin/env python3
"""Run the additive exact-once binary packet mechanics smoke v4."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from literature_multiverse.anthropic_bounded_generation import AnthropicBoundedClient
from literature_multiverse.metasyn_passage_binary_mechanics_smoke_v4 import (
    DEFAULT_BINARY_SMOKE_WORKSPACE,
    DEFAULT_V2_WORKSPACE,
    MetaSynPassageBinaryMechanicsSmokeV4Error,
    authorize_metasyn_passage_binary_mechanics_smoke_v4,
    finalize_metasyn_passage_binary_mechanics_smoke_v4,
    metasyn_passage_binary_mechanics_smoke_status_v4,
    prepare_metasyn_passage_binary_mechanics_smoke_v4,
    run_metasyn_passage_binary_mechanics_smoke_v4,
    validate_finalized_metasyn_passage_binary_mechanics_smoke_v4,
)
from literature_multiverse.metasyn_passage_hosted_bundle_v2 import (
    MetaSynPassageHostedExecutionBundleV2,
)
from literature_multiverse.providers import load_live_environment


def _rooted(value: Path, root: Path) -> Path:
    return value if value.is_absolute() else root / value


def _common(parser: argparse.ArgumentParser, *, anchor: bool = True) -> None:
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--workspace", type=Path, default=DEFAULT_BINARY_SMOKE_WORKSPACE)
    parser.add_argument("--v2-workspace", type=Path, default=DEFAULT_V2_WORKSPACE)
    if anchor:
        parser.add_argument("--expected-plan-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    _common(prepare, anchor=False)
    authorize = subparsers.add_parser("authorize")
    _common(authorize)
    smoke = subparsers.add_parser("smoke")
    _common(smoke)
    smoke.add_argument(
        "--live",
        action="store_true",
        help="explicitly enable the already-frozen at-most-two provider calls",
    )
    smoke.add_argument("--env-file", type=Path, default=Path(".env"))
    finalize = subparsers.add_parser("finalize")
    _common(finalize)
    validate = subparsers.add_parser("validate")
    _common(validate)
    status = subparsers.add_parser("status")
    _common(status)
    return parser


def _client(args: argparse.Namespace, *, root: Path, v2_workspace: Path) -> AnthropicBoundedClient:
    if not args.live:
        raise MetaSynPassageBinaryMechanicsSmokeV4Error("binary_mechanics_v4_live_flag_required")
    load_live_environment(_rooted(args.env_file, root), live_enabled=True)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_anthropic_api_key_missing"
        )
    bundle = MetaSynPassageHostedExecutionBundleV2.model_validate(
        json.loads((v2_workspace / "execution-bundle.json").read_text(encoding="utf-8"))
    )
    return AnthropicBoundedClient(bundle.anthropic_config)


def _summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    payload = value.model_dump(mode="json")
    allowed = {
        key: item
        for key, item in payload.items()
        if key
        in {
            "status",
            "plan_sha256",
            "pipeline_sha256",
            "authorization_receipt_sha256",
            "smoke_sha256",
            "report_sha256",
            "validation_sha256",
            "request_count",
            "authorized_call_count",
            "attempted_call_count",
            "typed_effect_count",
            "conservative_cost_ceiling_usd_micros",
            "claim_release_authority",
        }
    }
    allowed["artifact_type"] = type(value).__name__
    return allowed


def main() -> int:
    args = build_parser().parse_args()
    root = args.repository_root.resolve(strict=True)
    workspace = _rooted(args.workspace, root)
    v2_workspace = _rooted(args.v2_workspace, root)
    common = {
        "repository_root": root,
        "workspace": workspace,
        "v2_workspace": v2_workspace,
    }
    if args.command == "prepare":
        value: Any = prepare_metasyn_passage_binary_mechanics_smoke_v4(**common)
    else:
        anchored = {**common, "expected_plan_sha256": args.expected_plan_sha256}
        if args.command == "authorize":
            value = authorize_metasyn_passage_binary_mechanics_smoke_v4(**anchored)
        elif args.command == "smoke":
            value = run_metasyn_passage_binary_mechanics_smoke_v4(
                **anchored,
                client=_client(args, root=root, v2_workspace=v2_workspace),
            )
        elif args.command == "finalize":
            value = finalize_metasyn_passage_binary_mechanics_smoke_v4(**anchored)
        elif args.command == "validate":
            value = validate_finalized_metasyn_passage_binary_mechanics_smoke_v4(**anchored)
        else:
            value = metasyn_passage_binary_mechanics_smoke_status_v4(**anchored)
    print(json.dumps(_summary(value), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
