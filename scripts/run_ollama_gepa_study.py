#!/usr/bin/env python3
"""Run the staged local-Ollama official-GEPA Evidence Inference diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.local_ollama import LocalOllamaClient
from literature_multiverse.ollama_gepa_study import (
    load_study_config,
    prepare_optimization_plan,
    run_optimization,
    run_paired_test,
    study_paths,
    validate_frozen_winner,
    validate_optimization_plan,
    validate_private_receipts,
    validate_public_summary,
)

DEFAULT_CONFIG = Path("configs/benchmarks/evidence-inference-ollama-gepa-v1.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-explicit local-Ollama GEPA study. Prepare and optimize touch only "
            "train/dev; the separate test command validates a frozen winner first."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "prepare", help="freeze train/dev selections and exact local runtime identity"
    )
    subparsers.add_parser("optimize", help="run or resume official GEPA and freeze its winner")
    subparsers.add_parser("test", help="run the one-shot paired test after winner validation")
    subparsers.add_parser(
        "audit", help="validate the current stage and every private model-call receipt"
    )
    subparsers.add_parser("status", help="show which staged artifacts currently exist")
    return parser


def _client(args: argparse.Namespace) -> LocalOllamaClient:
    return LocalOllamaClient(args.base_url, timeout_seconds=args.timeout_seconds)


def main() -> int:
    args = _parser().parse_args()
    config = load_study_config(args.config)
    paths = study_paths(args.config, config)
    if args.command == "status":
        print(
            json.dumps(
                {
                    "plan_exists": paths.plan.is_file(),
                    "gepa_checkpoint_exists": (paths.gepa_run_dir / "gepa_state.bin").is_file(),
                    "winner_exists": paths.winner.is_file(),
                    "paired_report_exists": paths.paired_report.is_file(),
                    "public_summary_exists": paths.public_summary.is_file(),
                    "private_receipts": validate_private_receipts(paths.receipts),
                },
                sort_keys=True,
            )
        )
        return 0
    client = _client(args)
    if args.command == "prepare":
        print(prepare_optimization_plan(config_path=args.config, client=client))
        return 0
    if args.command == "optimize":
        print(run_optimization(config_path=args.config, client=client))
        return 0
    if args.command == "test":
        private, public = run_paired_test(config_path=args.config, client=client)
        print(json.dumps({"private_report": str(private), "public_summary": str(public)}))
        return 0
    if paths.winner.is_file():
        validate_frozen_winner(config_path=args.config, client=client)
    elif paths.plan.is_file():
        validate_optimization_plan(config_path=args.config, client=client)
    receipt_counts = validate_private_receipts(paths.receipts)
    if paths.public_summary.is_file():
        validate_public_summary(json.loads(paths.public_summary.read_text(encoding="utf-8")))
    print(json.dumps({"status": "valid", "private_receipts": receipt_counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
