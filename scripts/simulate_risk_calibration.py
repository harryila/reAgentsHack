#!/usr/bin/env python3
"""Run planted repeated simulations of scientific-claim release calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.calibration_simulation import (
    DEFAULT_CANDIDATE_THRESHOLDS,
    simulate_replicate,
    summarize_replicates,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_FILES = (
    "scripts/simulate_risk_calibration.py",
    "src/literature_multiverse/__init__.py",
    "src/literature_multiverse/calibration_simulation.py",
    "src/literature_multiverse/calibration.py",
    "src/literature_multiverse/lineage.py",
    "src/literature_multiverse/models.py",
    "src/literature_multiverse/paths.py",
    "pyproject.toml",
    "uv.lock",
)


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
    parser.add_argument(
        "--candidate-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_CANDIDATE_THRESHOLDS),
    )
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
        "candidate_thresholds": args.candidate_thresholds,
        "generator_hyperparameters": {
            "simulation_version": "risk-features-v2",
            "paper_count": {"offset": 2, "poisson_rate": 9},
            "extraction_error_beta": [1.4, 8.0],
            "ungrounded_residual_beta": [1.2, 14.0],
            "verifier_disagreement_residual_beta": [1.6, 7.0],
            "retrieval_gap_beta": [1.5, 6.5],
            "bootstrap_instability_beta": [1.7, 5.0],
            "heterogeneity_beta": [2.0, 2.2],
            "moderator_noise_normal": [0.0, 0.08],
            "loss_logit": {
                "intercept": -5.3,
                "extraction_error": 3.7,
                "ungrounded_fraction": 3.2,
                "verifier_disagreement": 2.0,
                "retrieval_gap": 1.7,
                "bootstrap_instability": 2.2,
                "moderator_instability": 1.4,
                "heterogeneity": 0.8,
                "inverse_sqrt_papers": 1.1,
            },
            "fixed_paper_gate": 5,
            "bootstrap_instability_gate": 0.20,
            "domains": ["medicine", "psychology", "ecology"],
        },
        "source_files_sha256": {
            relative: sha256_file(_ROOT / relative) for relative in _SOURCE_FILES
        },
    }
    replicates = [
        simulate_replicate(
            seed=args.seed + index,
            alpha=args.alpha,
            delta=args.delta,
            development_count=args.development_count,
            calibration_count=args.calibration_count,
            test_count=args.test_count,
            candidate_thresholds=args.candidate_thresholds,
        )
        for index in range(args.replicates)
    ]
    artifact = {
        "simulation_study_version": "3",
        "run_config": run_config,
        "run_config_sha256": hash_canonical(run_config),
        "summary": summarize_replicates(replicates, alpha=args.alpha),
        "replicates": replicates,
    }
    artifact["artifact_payload_sha256"] = hash_canonical(artifact)
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
