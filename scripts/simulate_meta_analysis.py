#!/usr/bin/env python3
"""Compare meta-regression with significance-derived vote counting in simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.lineage import atomic_write_json, hash_canonical
from literature_multiverse.meta_simulation import (
    simulate_meta_replicate,
    summarize_meta_simulations,
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
        "meta_simulation_study_version": "1",
        "run_config": config,
        "run_config_sha256": hash_canonical(config),
        "summary": summary,
        "null_replicates": null_rows,
        "moderator_replicates": moderator_rows,
    }
    atomic_write_json(args.output, artifact, force=args.force)
    print(json.dumps({"output": args.output.as_posix(), "summary": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

