#!/usr/bin/env python3
"""Replay all registered audit policies over an immutable question JSONL benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.lineage import atomic_write_json
from literature_multiverse.question_evaluation import (
    ReplayPolicy,
    ReplayStoppingRule,
    evaluate_question_benchmark,
    load_question_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--budget-minutes",
        type=float,
        action="append",
        required=True,
        help="Per-question realized-minute cap; repeat for a cost curve.",
    )
    parser.add_argument("--fixed-count", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--primary-policy",
        choices=[
            policy.value
            for policy in ReplayPolicy
            if policy is not ReplayPolicy.AUDIT_ALL_UPPER_BOUND
        ],
        default=ReplayPolicy.RISK_X_INFLUENCE_PER_COST.value,
        help="Prespecified arm used for paired within-question comparisons.",
    )
    parser.add_argument(
        "--stopping-rule",
        choices=[rule.value for rule in ReplayStoppingRule],
        default=ReplayStoppingRule.PRODUCTION_STOP_ON_RELEASE.value,
        help=(
            "Prespecified stopping boundary. The default matches production by stopping "
            "at the first certificate-bound release-eligible state; every replay state "
            "must come from a validated v4 production certificate. Allocate-to-cap is "
            "experimental and is required for legacy/simulation/diagnostic rows."
        ),
    )
    parser.add_argument(
        "--allow-non-real",
        action="store_true",
        help=(
            "Permit simulation/diagnostic inputs. The output remains explicitly "
            "ineligible for a real scientific or human-efficiency claim; also select "
            "--stopping-rule allocate_to_cap_experimental."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark = load_question_benchmark(args.benchmark)
    result = evaluate_question_benchmark(
        benchmark,
        budgets_minutes=args.budget_minutes,
        fixed_count=args.fixed_count,
        random_seed=args.random_seed,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
        allow_non_real=args.allow_non_real,
        primary_policy=ReplayPolicy(args.primary_policy),
        stopping_rule=ReplayStoppingRule(args.stopping_rule),
    )
    atomic_write_json(args.output, result, force=args.force)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "evaluation_sha256": result.evaluation_sha256,
                "evidence_kind": result.evidence_kind.value,
                "scientific_claim_eligible": result.scientific_claim_eligible,
                "production_policy_match": result.production_policy_match,
                "question_count": len(benchmark.records),
                "policy_budget_rows": len(result.policy_results),
                "paired_policy_comparisons": len(result.paired_policy_comparisons),
                "pipeline_sha256": result.pipeline_sha256,
                "primary_policy": result.primary_policy.value,
                "stopping_rule": result.stopping_rule.value,
                "evaluation_pipeline_sha256": (
                    result.evaluation_pipeline_fingerprint.pipeline_sha256
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
