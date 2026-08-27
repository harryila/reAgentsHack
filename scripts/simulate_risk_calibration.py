#!/usr/bin/env python3
"""Run planted repeated simulations of scientific-claim release calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.calibration_simulation import (
    simulate_replicate,
    summarize_replicates,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--development-count", type=int, default=400)
    parser.add_argument("--calibration-count", type=int, default=2000)
    parser.add_argument("--test-count", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.replicates < 1:
        raise ValueError("replicates_must_be_positive")
    run_config = {
        "replicates": args.replicates,
        "seed": args.seed,
        "alpha": args.alpha,
        "delta": args.delta,
        "development_count": args.development_count,
        "calibration_count": args.calibration_count,
        "test_count": args.test_count,
    }
    replicates = [
        simulate_replicate(
            seed=args.seed + index,
            alpha=args.alpha,
            delta=args.delta,
            development_count=args.development_count,
            calibration_count=args.calibration_count,
            test_count=args.test_count,
        )
        for index in range(args.replicates)
    ]
    artifact = {
        "simulation_study_version": "1",
        "run_config": run_config,
        "run_config_sha256": hash_canonical(run_config),
        "summary": summarize_replicates(replicates, alpha=args.alpha),
        "replicates": replicates,
    }
    atomic_write_json(args.output, artifact, force=args.force)
    print(
        json.dumps(
            {
                "replicates": args.replicates,
                "output": args.output.as_posix(),
                "summary": artifact["summary"]["policies"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
