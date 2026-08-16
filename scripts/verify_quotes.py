#!/usr/bin/env python3
"""Independently verify every exact-grounded quote-to-direction claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.config import (
    authorize_stage,
    config_sha256,
    load_config_for_question,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical
from literature_multiverse.models import VerificationRecord
from literature_multiverse.paths import PATHS
from literature_multiverse.prompting import render_prompt_file
from literature_multiverse.providers import AnthropicProvider, load_live_environment
from literature_multiverse.records import read_parquet_records
from literature_multiverse.verification import (
    VerificationContractError,
    fixture_verification,
    requested_grounded_rows,
    verify_with_provider,
)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationContractError(f"invalid_json:{path}") from exc
    if not isinstance(value, dict):
        raise VerificationContractError(f"json_root_must_be_object:{path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--scope", choices=("grounded-v1", "full"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--max-budget-usd", type=float, default=15.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config_for_question(args.question, require_locked=True)
    if not args.fixture and not args.live:
        raise VerificationContractError("production verification requires explicit --live")
    authorize_stage(
        config,
        "s4",
        explicit_fixture=args.fixture,
        live_provider=args.live,
    )

    processed = PATHS.processed_dir(args.question)
    findings_path = processed / "findings.parquet"
    output_path = processed / "verification.json"
    findings = read_parquet_records(findings_path)
    request_rows = requested_grounded_rows(findings)
    requested_ids = [str(row["finding_id"]) for row in request_rows]

    if output_path.exists():
        if not args.resume:
            raise VerificationContractError(f"verification_already_exists:{output_path}")
        existing = VerificationRecord.model_validate(_read_object(output_path))
        if existing.requested_finding_ids != requested_ids:
            raise VerificationContractError("resume_verification_request_set_changed")
        print(
            json.dumps(
                {
                    "question_id": args.question,
                    "status": "reused",
                    "requested": len(requested_ids),
                    "output": PATHS.repository_relative(output_path),
                },
                sort_keys=True,
            )
        )
        return 0

    definitions = {
        "increase": config.target_relation.increase_definition,
        "no_effect": config.target_relation.no_effect_definition,
        "decrease": config.target_relation.decrease_definition,
        "comparator": config.target_relation.comparator,
        "outcome": config.target_relation.outcome,
    }
    rendered = render_prompt_file(
        PATHS.prompts_dir / "quote_verification.md",
        {
            "DIRECTION_DEFINITIONS_JSON": json.dumps(
                definitions,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            "FINDINGS_JSON": json.dumps(
                request_rows,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        },
    )
    request_key = hash_canonical(
        {
            "question_id": args.question,
            "scope": args.scope,
            "config_sha256": config_sha256(config),
            "finding_ids": requested_ids,
            "prompt_sha256": rendered.sha256,
        }
    )

    if args.fixture:
        overrides_path = processed / "fixture_verification_overrides.json"
        overrides = _read_object(overrides_path) if overrides_path.exists() else None
        record = fixture_verification(
            findings=findings,
            rendered_prompt=rendered,
            status_overrides=overrides,
        )
    else:
        load_live_environment(PATHS.root / ".env", live_enabled=args.live)
        provider = AnthropicProvider(
            model=args.model,
            effort=args.effort,
            max_tokens=args.max_tokens,
            archive_dir=PATHS.data_dir / "raw" / "providers" / args.question / "verification",
            max_budget_usd=args.max_budget_usd,
            live_enabled=args.live,
            global_budget_dir=PATHS.data_dir / "raw" / "providers",
            global_max_budget_usd=50.0,
        )
        record = verify_with_provider(
            findings=findings,
            rendered_prompt=rendered,
            provider=provider,
            request_key=request_key,
        )
    atomic_write_json(output_path, record, force=False)
    print(
        json.dumps(
            {
                "question_id": args.question,
                "status": "complete",
                "requested": len(requested_ids),
                "provider": record.provider,
                "model": record.model,
                "output": PATHS.repository_relative(output_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
