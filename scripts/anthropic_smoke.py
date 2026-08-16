#!/usr/bin/env python3
"""Run one tiny, budgeted Sonnet transport/structured-output smoke check."""

from __future__ import annotations

import argparse
import json

from literature_multiverse.paths import PATHS
from literature_multiverse.providers import AnthropicProvider, load_live_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.live:
        raise SystemExit("Anthropic smoke requires explicit --live")
    load_live_environment(PATHS.root / ".env", live_enabled=True)
    provider = AnthropicProvider(
        model="claude-sonnet-5",
        effort="low",
        max_tokens=64,
        archive_dir=PATHS.data_dir / "raw" / "providers" / "smoke",
        max_budget_usd=0.25,
        live_enabled=True,
        global_budget_dir=PATHS.data_dir / "raw" / "providers",
        global_max_budget_usd=50.0,
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"status": {"type": "string", "enum": ["ok"]}},
        "required": ["status"],
    }
    result = provider.generate(
        operation="transport_schema_smoke",
        request_key="anthropic-transport-schema-v1",
        prompt="Return the structured status value ok.",
        system="This is a transport and JSON Schema wiring test. Return only the schema output.",
        output_schema=schema,
    )
    if result.parsed_json != {"status": "ok"}:
        raise SystemExit("Anthropic smoke returned an unexpected structured result")
    print(
        json.dumps(
            {
                "status": "ok",
                "model": result.model,
                "estimated_cost_usd": result.estimated_cost_usd,
                "archive": PATHS.repository_relative(result.archive_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
