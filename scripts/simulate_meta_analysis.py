#!/usr/bin/env python3
"""Compare meta-regression with significance-derived vote counting in simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file
from literature_multiverse.meta_simulation import (
    simulate_meta_replicate,
    summarize_meta_simulations,
)

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_FILES = (
    "scripts/simulate_meta_analysis.py",
    "src/literature_multiverse/__init__.py",
    "src/literature_multiverse/meta_simulation.py",
    "src/literature_multiverse/meta_analysis.py",
    "src/literature_multiverse/budgeted_verification.py",
    "src/literature_multiverse/claim_semantics.py",
    "src/literature_multiverse/effects.py",
    "src/literature_multiverse/evidence_graph.py",
    "src/literature_multiverse/lineage.py",
    "src/literature_multiverse/models.py",
    "src/literature_multiverse/paths.py",
    "pyproject.toml",
    "uv.lock",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--moderator-effect", type=float, default=0.35)
    parser.add_argument("--papers-per-level", type=int, default=30)
    parser.add_argument("--heldout-papers-per-level", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.replicates < 1:
        raise ValueError("replicates_must_be_positive")
    config = {
        "replicates": args.replicates,
        "seed": args.seed,
        "alpha": args.alpha,
        "moderator_effect": args.moderator_effect,
        "papers_per_level": args.papers_per_level,
        "heldout_papers_per_level": args.heldout_papers_per_level,
        "generator_hyperparameters": {
            "overall_effect": 0.25,
            "between_paper_standard_deviation": 0.12,
            "high_precision_standard_error_uniform": [0.08, 0.14],
            "low_precision_standard_error_uniform": [0.25, 0.38],
            "moderator_levels": ["high_precision", "low_precision"],
            "reported_significance_two_sided_alpha": 0.05,
            "vote_probability_smoothing": "add_one_beta_1_1",
            "meta_regression_min_papers_per_level": 4,
            "effect_measure": "mean_difference_standardized_units",
        },
        "source_files_sha256": {
            relative: sha256_file(_ROOT / relative) for relative in _SOURCE_FILES
        },
    }
    null_rows = [
        simulate_meta_replicate(
            seed=args.seed + index,
            moderator_effect=0.0,
            papers_per_level=args.papers_per_level,
            heldout_papers_per_level=args.heldout_papers_per_level,
            alpha=args.alpha,
        )
        for index in range(args.replicates)
    ]
    moderator_rows = [
        simulate_meta_replicate(
            seed=args.seed + args.replicates + index,
            moderator_effect=args.moderator_effect,
            papers_per_level=args.papers_per_level,
            heldout_papers_per_level=args.heldout_papers_per_level,
            alpha=args.alpha,
        )
        for index in range(args.replicates)
    ]
    summary = summarize_meta_simulations(null_rows, moderator_rows, alpha=args.alpha)
    artifact = {
        "meta_simulation_study_version": "3",
        "run_config": config,
        "run_config_sha256": hash_canonical(config),
        "summary": summary,
        "null_replicates": null_rows,
        "moderator_replicates": moderator_rows,
    }
    artifact["artifact_payload_sha256"] = hash_canonical(artifact)
    atomic_write_json(args.output, artifact, force=args.force)
    print(json.dumps({"output": args.output.as_posix(), "summary": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
