#!/usr/bin/env python3
"""Compare nine human-verification allocation policies in planted simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.budgeted_verification import AllocationPolicy
from literature_multiverse.budgeted_verification_simulation import (
    PAIRED_BINARY_BOOTSTRAP_BASE_SEED,
    PAIRED_BINARY_BOOTSTRAP_DRAWS,
    PAIRED_CONTRAST_BUDGET,
    UNCERTAINTY_CONFIDENCE_LEVEL,
    simulate_budgeted_verification_replicate,
    summarize_budgeted_verification_simulations,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_FILES = (
    "scripts/simulate_budgeted_verification.py",
    "src/literature_multiverse/budgeted_verification_simulation.py",
    "src/literature_multiverse/budgeted_verification.py",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--item-count", type=int, default=60)
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0, 40.0],
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.replicates < 1:
        raise ValueError("replicates_must_be_positive")
    if args.item_count < 8:
        raise ValueError("item_count_must_be_at_least_eight")
    if len(set(args.budgets)) != len(args.budgets):
        raise ValueError("budgets_must_be_unique")
    if any(budget < 0 for budget in args.budgets):
        raise ValueError("budgets_must_be_nonnegative")
    config = {
        "simulation_version": "budgeted-verification-v3",
        "replicates": args.replicates,
        "seed": args.seed,
        "item_count": args.item_count,
        "budgets": args.budgets,
        "policy_family": [policy.value for policy in AllocationPolicy],
        "generator_hyperparameters": {
            "latent_error_probability_beta": [1.5, 4.5],
            "latent_error_probability_clip": [0.02, 0.85],
            "distortion_lognormal": [-2.0, 0.65],
            "distortion_clip": [0.025, 0.65],
            "verification_cost_lognormal": [0.25, 0.45],
            "verification_cost_clip": [0.5, 4.0],
            "disagreement": {
                "risk_coefficient": 0.70,
                "normal_location": 0.10,
                "normal_scale": 0.13,
                "clip": [0.0, 1.0],
            },
            "true_contribution_normal": [0.0, 0.018],
            "target_baseline_score_normal": [0.25, 0.05],
            "claim_temperature": 1.0,
            "claim_decision_threshold": 0.5,
            "correction_direction": "monotone_positive_distortion_removal",
        },
        "source_files_sha256": {
            relative: sha256_file(_ROOT / relative) for relative in _SOURCE_FILES
        },
        "uncertainty": {
            "confidence_level": UNCERTAINTY_CONFIDENCE_LEVEL,
            "paired_contrast_budget": PAIRED_CONTRAST_BUDGET,
            "paired_binary_bootstrap_draws": PAIRED_BINARY_BOOTSTRAP_DRAWS,
            "paired_binary_bootstrap_base_seed": PAIRED_BINARY_BOOTSTRAP_BASE_SEED,
        },
    }
    replicates = [
        simulate_budgeted_verification_replicate(
            seed=args.seed + index,
            item_count=args.item_count,
            budgets=args.budgets,
        )
        for index in range(args.replicates)
    ]
    summary = summarize_budgeted_verification_simulations(
        replicates,
        budgets=args.budgets,
    )
    artifact = {
        "budgeted_verification_simulation_study_version": "4",
        "evidence_scope": {
            "artifact_kind": "planted_simulation",
            "real_world_evidence": False,
            "error_probability_basis": "known_planted_generative_probability",
            "verification_cost_basis": "simulated_human_minutes",
            "human_adjudication_performed": False,
        },
        "run_config": config,
        "run_config_sha256": hash_canonical(config),
        "summary": summary,
        "replicates": replicates,
    }
    atomic_write_json(args.output, artifact, force=args.force)
    print(json.dumps({"output": args.output.as_posix(), "summary": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
