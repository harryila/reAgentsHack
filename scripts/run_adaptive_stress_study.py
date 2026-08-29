#!/usr/bin/env python3
"""Run the frozen misspecified adaptive-verification stress study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.adaptive_stress_study import (
    DEFAULT_SCENARIOS,
    build_adaptive_stress_study_artifact,
    freeze_stress_study_config,
)
from literature_multiverse.lineage import atomic_write_json, sha256_file

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_FILES = (
    "pyproject.toml",
    "scripts/run_adaptive_stress_study.py",
    "src/literature_multiverse/__init__.py",
    "src/literature_multiverse/adaptive_stress_study.py",
    "src/literature_multiverse/budgeted_verification.py",
    "src/literature_multiverse/lineage.py",
    "src/literature_multiverse/models.py",
    "src/literature_multiverse/paths.py",
    "uv.lock",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/diagnostics/adaptive-stress-study-v1.json"),
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--questions-per-scenario", type=int, default=160)
    parser.add_argument("--items-per-question", type=int, default=24)
    parser.add_argument(
        "--budgets-minutes", type=float, nargs="+", default=[15.0, 30.0, 60.0]
    )
    parser.add_argument(
        "--release-risk-thresholds",
        type=float,
        nargs="+",
        default=[0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 1.00],
    )
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--fixed-count", type=int, default=5)
    parser.add_argument("--release-monte-carlo-draws", type=int, default=192)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    parser.add_argument("--primary-budget-minutes", type=float, default=30.0)
    parser.add_argument("--primary-release-risk-threshold", type=float, default=0.10)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_hashes = {
        relative: sha256_file(_ROOT / relative) for relative in _SOURCE_FILES
    }
    # This exact object is sealed before the first synthetic question is generated.
    config = freeze_stress_study_config(
        seed=args.seed,
        questions_per_scenario=args.questions_per_scenario,
        items_per_question=args.items_per_question,
        budgets_minutes=args.budgets_minutes,
        release_risk_thresholds=args.release_risk_thresholds,
        scenarios=args.scenarios,
        fixed_count=args.fixed_count,
        release_monte_carlo_draws=args.release_monte_carlo_draws,
        bootstrap_draws=args.bootstrap_draws,
        primary_budget_minutes=args.primary_budget_minutes,
        primary_release_risk_threshold=args.primary_release_risk_threshold,
        source_files_sha256=source_hashes,
    )
    artifact = build_adaptive_stress_study_artifact(config)
    output = args.output if args.output.is_absolute() else _ROOT / args.output
    atomic_write_json(output, artifact, force=args.force)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact["artifact_sha256"],
                "config_sha256": config["config_sha256"],
                "independent_questions": artifact["summary"][
                    "independent_questions"
                ],
                "output": output.as_posix(),
                "simulation_only": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
