#!/usr/bin/env python3
"""Create leakage-safe splits, optimize prompts with official GEPA, or test a frozen winner."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from literature_multiverse.paths import PATHS
from literature_multiverse.prompt_optimization import (
    OptimizationContractError,
    compare_frozen_test_to_seed,
    create_split_bundle,
    evaluate_frozen_test,
    load_optimization_examples,
    load_split_manifest,
    optimize_prompts,
)
from literature_multiverse.providers import (
    AnthropicProvider,
    ProviderBudget,
    ProviderBudgetExceeded,
    load_live_environment,
)

_COMBINED_PLANNING_CEILING_USD = 50.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("split", help="write immutable disjoint split files")
    split.add_argument("--examples", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--seed", type=int, required=True)
    split.add_argument("--train-fraction", type=float, default=0.6)
    split.add_argument("--dev-fraction", type=float, default=0.2)

    validate = commands.add_parser(
        "validate", help="validate manifest metadata and open train/dev only"
    )
    validate.add_argument("--manifest", type=Path, required=True)

    optimize = commands.add_parser("optimize", help="run GEPA and freeze the dev winner")
    optimize.add_argument("--manifest", type=Path, required=True)
    optimize.add_argument("--run-dir", type=Path, required=True)
    optimize.add_argument(
        "--extraction-template", type=Path, default=PATHS.prompts_dir / "extraction.md"
    )
    optimize.add_argument(
        "--verification-template",
        type=Path,
        default=PATHS.prompts_dir / "quote_verification.md",
    )
    optimize.add_argument("--seed", type=int, required=True)
    optimize.add_argument("--reflection-lm", required=True)
    optimize.add_argument("--reflection-max-tokens", type=int, default=2000)
    optimize.add_argument("--reflection-temperature", type=float)
    optimize.add_argument("--max-metric-calls-per-prompt", type=int, default=100)
    optimize.add_argument("--max-reflection-cost-usd-per-prompt", type=float, default=10.0)
    optimize.add_argument(
        "--reflection-batch-headroom-usd-per-prompt",
        type=float,
        default=0.5,
        help=(
            "conservative extra reservation for the reflection call that may finish "
            "after each per-prompt GEPA cost stopper boundary"
        ),
    )
    optimize.add_argument("--reflection-minibatch-size", type=int, default=3)
    optimize.add_argument("--cost-cap-usd", type=float, default=0.02)
    _add_live_provider_args(optimize)

    test = commands.add_parser(
        "test",
        help="open the test split held out from optimization and evaluate one frozen winner",
    )
    test.add_argument("--manifest", type=Path, required=True)
    test.add_argument("--winner", type=Path, required=True)
    test.add_argument("--output", type=Path, required=True)
    test.add_argument("--cost-cap-usd", type=float, default=0.02)
    _add_live_provider_args(test)

    compare = commands.add_parser(
        "compare-test",
        help="paired held-out-from-optimization test comparison of winner versus seed",
    )
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--winner", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument(
        "--seed-extraction-template",
        type=Path,
        help="Explicit original extraction seed; no default is inferred.",
    )
    compare.add_argument(
        "--seed-verification-template",
        type=Path,
        help="Explicit original verification seed; no default is inferred.",
    )
    compare.add_argument("--cost-cap-usd", type=float, default=0.02)
    _add_live_provider_args(compare)
    return parser


def _add_live_provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--max-budget-usd", type=float, default=25.0)


def _live_provider(
    args: argparse.Namespace,
    archive_dir: Path,
    *,
    global_max_budget_usd: float = _COMBINED_PLANNING_CEILING_USD,
) -> AnthropicProvider:
    if not args.live:
        raise OptimizationContractError("GEPA provider calls require explicit --live")
    load_live_environment(PATHS.root / ".env", live_enabled=True)
    return AnthropicProvider(
        model=args.model,
        effort=args.effort,
        max_tokens=args.max_tokens,
        archive_dir=archive_dir,
        max_budget_usd=args.max_budget_usd,
        live_enabled=True,
        # GEPA run directories intentionally live in the gitignored data/cache tree
        # because provider receipts contain benchmark text.  Scan the repository root
        # so the task-provider ledger includes those receipts as well as ordinary
        # providers. Reflection uses LiteLLM and is reserved separately by preflight.
        global_budget_dir=PATHS.root,
        global_max_budget_usd=global_max_budget_usd,
    )


def _optimization_budget_preflight(
    args: argparse.Namespace,
    *,
    archive_root: Path = PATHS.root,
) -> dict[str, float | int]:
    if (
        not math.isfinite(args.reflection_batch_headroom_usd_per_prompt)
        or args.reflection_batch_headroom_usd_per_prompt < 0
    ):
        raise ValueError("reflection batch headroom must be finite and nonnegative")
    if not math.isfinite(args.max_budget_usd) or args.max_budget_usd <= 0:
        raise ValueError("task-provider max budget must be positive and finite")
    if (
        not math.isfinite(args.max_reflection_cost_usd_per_prompt)
        or args.max_reflection_cost_usd_per_prompt <= 0
    ):
        raise ValueError("reflection max cost must be positive and finite")

    train, dev = load_optimization_examples(args.manifest)
    active_prompt_kinds = len({example.prompt_kind for example in [*train, *dev]})
    if active_prompt_kinds < 1:
        raise OptimizationContractError("optimization benchmark has no active prompt kinds")
    existing_archived = ProviderBudget(
        archive_root, _COMBINED_PLANNING_CEILING_USD
    ).spent_usd()
    reflection_stop_ceiling = (
        active_prompt_kinds * args.max_reflection_cost_usd_per_prompt
    )
    reflection_batch_headroom = (
        active_prompt_kinds * args.reflection_batch_headroom_usd_per_prompt
    )
    reflection_reservation = reflection_stop_ceiling + reflection_batch_headroom
    projected = math.fsum(
        (existing_archived, args.max_budget_usd, reflection_reservation)
    )
    if projected > _COMBINED_PLANNING_CEILING_USD + 1e-12:
        raise ProviderBudgetExceeded(
            "combined optimization preflight would exceed the $50.00 planning ceiling: "
            f"archived=${existing_archived:.6f}, task_rollout=${args.max_budget_usd:.6f}, "
            f"reflection_plus_headroom=${reflection_reservation:.6f}, "
            f"projected=${projected:.6f}"
        )
    task_provider_global_limit = (
        _COMBINED_PLANNING_CEILING_USD - reflection_reservation
    )
    return {
        "planning_ceiling_usd": _COMBINED_PLANNING_CEILING_USD,
        "existing_archived_provider_ceiling_usd": existing_archived,
        "task_rollout_ceiling_usd": float(args.max_budget_usd),
        "active_prompt_kinds": active_prompt_kinds,
        "reflection_stop_ceiling_usd": reflection_stop_ceiling,
        "reflection_batch_headroom_usd": reflection_batch_headroom,
        "projected_combined_ceiling_usd": projected,
        "task_provider_global_limit_usd": task_provider_global_limit,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "split":
        manifest_path = create_split_bundle(
            args.examples,
            args.output_dir,
            seed=args.seed,
            train_fraction=args.train_fraction,
            dev_fraction=args.dev_fraction,
        )
        manifest = load_split_manifest(manifest_path)
        print(
            json.dumps(
                {
                    "manifest": manifest_path.as_posix(),
                    "rows": {
                        "train": manifest.train.rows,
                        "dev": manifest.dev.rows,
                        "test": manifest.test.rows,
                    },
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "validate":
        manifest = load_split_manifest(args.manifest)
        train, dev = load_optimization_examples(args.manifest)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "train_rows": len(train),
                    "dev_rows": len(dev),
                    "test_rows_from_manifest_only": manifest.test.rows,
                    "test_split_opened": False,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "optimize":
        budget_preflight = _optimization_budget_preflight(args)
        provider = _live_provider(
            args,
            args.run_dir / "provider_attempts",
            global_max_budget_usd=float(
                budget_preflight["task_provider_global_limit_usd"]
            ),
        )
        run = optimize_prompts(
            manifest_path=args.manifest,
            seed_templates={
                "extraction": args.extraction_template,
                "quote_verification": args.verification_template,
            },
            provider=provider,
            run_dir=args.run_dir,
            reflection_lm=args.reflection_lm,
            reflection_lm_kwargs={
                "max_tokens": args.reflection_max_tokens,
                "num_retries": 0,
                "temperature": args.reflection_temperature,
            },
            task_provider_identity={
                "provider": "anthropic",
                "model": args.model,
                "effort": args.effort,
                "max_tokens": args.max_tokens,
                "max_budget_usd": args.max_budget_usd,
            },
            combined_budget_preflight=budget_preflight,
            max_metric_calls_per_prompt=args.max_metric_calls_per_prompt,
            max_reflection_cost_usd_per_prompt=args.max_reflection_cost_usd_per_prompt,
            reflection_minibatch_size=args.reflection_minibatch_size,
            seed=args.seed,
            cost_cap_usd=args.cost_cap_usd,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "winner": run.winner_path.as_posix(),
                    "winner_sha256": run.winner_sha256,
                    "trace": run.trace_path.as_posix(),
                    "test_evaluated": False,
                    "budget_preflight": budget_preflight,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "test":
        provider = _live_provider(args, args.output.parent / "test_provider_attempts")
        report_path = evaluate_frozen_test(
            manifest_path=args.manifest,
            winner_path=args.winner,
            provider=provider,
            output_path=args.output,
            cost_cap_usd=args.cost_cap_usd,
        )
        print(json.dumps({"status": "complete", "report": report_path.as_posix()}))
        return 0

    if args.command == "compare-test":
        seed_templates = {
            prompt_kind: path
            for prompt_kind, path in (
                ("extraction", args.seed_extraction_template),
                ("quote_verification", args.seed_verification_template),
            )
            if path is not None
        }
        if not seed_templates:
            raise OptimizationContractError(
                "compare-test requires at least one explicit --seed-*-template"
            )
        provider = _live_provider(
            args, args.output.parent / "paired_test_provider_attempts"
        )
        report_path = compare_frozen_test_to_seed(
            manifest_path=args.manifest,
            winner_path=args.winner,
            seed_templates=seed_templates,
            provider=provider,
            output_path=args.output,
            cost_cap_usd=args.cost_cap_usd,
        )
        print(json.dumps({"status": "complete", "report": report_path.as_posix()}))
        return 0

    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
