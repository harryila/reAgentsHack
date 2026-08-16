#!/usr/bin/env python3
"""Generate the one-shot comparison baseline for an exact analyzed cohort."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from literature_multiverse.baseline import (
    BaselineContractError,
    create_baseline,
    select_primary_rows,
    write_baseline_once,
)
from literature_multiverse.config import authorize_stage, load_config_for_question
from literature_multiverse.paths import PATHS
from literature_multiverse.providers import AnthropicProvider, load_live_environment


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaselineContractError(f"invalid_json:{path}") from exc
    if not isinstance(value, dict):
        raise BaselineContractError(f"json_root_must_be_object:{path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--max-budget-usd", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config_for_question(args.question, require_locked=True)
    if not config.demo:
        raise BaselineContractError("locked config is missing demo settings")
    if not args.fixture and not args.live:
        raise BaselineContractError("production baseline requires explicit --live")
    authorize_stage(
        config,
        "s7",
        explicit_fixture=args.fixture,
        live_provider=args.live,
    )

    processed = PATHS.processed_dir(args.question)
    analysis = PATHS.analysis_dir(args.question)
    findings_path = processed / "findings.parquet"
    verification_path = processed / "verification.json"
    m4_gate_path = analysis / "m4_gate.json"
    s5_run_path = analysis / "run.json"
    output_path = analysis / "baseline.json"

    findings = pd.read_parquet(findings_path).to_dict(orient="records")
    verification = _read_object(verification_path)
    m4_gate = _read_object(m4_gate_path)
    cohort_hash = m4_gate.get("cohort_hash")
    if not isinstance(cohort_hash, str):
        raise BaselineContractError("m4_gate_missing_cohort_hash")
    primary_family = config.outcomes.primary_family
    assert primary_family is not None
    primary_rows = select_primary_rows(
        findings,
        verification,
        primary_family=primary_family,
    )

    if args.fixture:
        run = _read_object(s5_run_path)
        completed_at = run.get("completed_at")
        if not isinstance(completed_at, str):
            raise BaselineContractError("fixture_s5_run_missing_completed_at")
        attempted_at = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        provider = None
    else:
        load_live_environment(PATHS.root / ".env", live_enabled=args.live)
        attempted_at = datetime.now(UTC)
        provider = AnthropicProvider(
            model=args.model,
            effort=args.effort,
            max_tokens=args.max_tokens,
            archive_dir=PATHS.data_dir / "raw" / "providers" / args.question / "baseline",
            max_budget_usd=args.max_budget_usd,
            live_enabled=args.live,
            global_budget_dir=PATHS.data_dir / "raw" / "providers",
            global_max_budget_usd=50.0,
        )
    artifact = create_baseline(
        cohort_hash=cohort_hash,
        research_question=config.research_question,
        primary_rows=primary_rows,
        prompt_path=PATHS.prompts_dir / "baseline_consensus.md",
        attempted_at=attempted_at,
        fixture_mode=args.fixture,
        provider=provider,
    )
    write_baseline_once(output_path, artifact)
    print(
        json.dumps(
            {
                "question_id": args.question,
                "status": artifact.status,
                "source": artifact.source,
                "output": PATHS.repository_relative(output_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
