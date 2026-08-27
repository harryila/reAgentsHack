#!/usr/bin/env python3
"""Prepare or evaluate the offline MetaSyn review-level benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.lineage import atomic_write_json, atomic_write_jsonl
from literature_multiverse.metasyn_benchmark import (
    build_metasyn_risk_examples,
    evaluate_metasyn_predictions,
    freeze_fixed_direction_baseline,
    load_metasyn_predictions,
    prepare_metasyn_benchmark,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Create leakage-safe model inputs and a private evaluator file."
    )
    prepare.add_argument(
        "--train-parquet",
        type=Path,
        default=Path("data/cache/metasyn/reviews-train.parquet"),
    )
    prepare.add_argument(
        "--test-parquet",
        type=Path,
        default=Path("data/cache/metasyn/reviews-test.parquet"),
    )
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=20260826)
    prepare.add_argument("--calibration-fraction", type=float, default=0.5)
    prepare.add_argument("--force", action="store_true")

    freeze = subparsers.add_parser(
        "freeze-fixed-direction",
        help="Freeze a label-blind constant-direction control for one named split.",
    )
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument(
        "--split",
        choices=("development", "calibration", "test"),
        required=True,
    )
    freeze.add_argument(
        "--direction",
        choices=("Positive", "Negative", "Mixed"),
        required=True,
    )
    freeze.add_argument("--selection-note", required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)

    evaluate = subparsers.add_parser(
        "evaluate", help="Score frozen predictions with the private evaluator labels."
    )
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--split",
        choices=("development", "calibration", "test", "all"),
        default="test",
        help="Split to score; defaults to the test set held out from model optimization.",
    )
    evaluate.add_argument(
        "--risk-examples-output",
        type=Path,
        help="Optional RiskExample JSONL; requires --pipeline-sha256.",
    )
    evaluate.add_argument("--pipeline-sha256")
    evaluate.add_argument("--population-id", default="metasyn-review-direction-v1")
    evaluate.add_argument("--force", action="store_true")
    return parser


def _prepare(args: argparse.Namespace) -> int:
    manifest = prepare_metasyn_benchmark(
        train_parquet=args.train_parquet,
        test_parquet=args.test_parquet,
        output_dir=args.output_dir,
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "development": manifest.development.rows,
                "calibration": manifest.calibration.rows,
                "test": manifest.test.rows,
                "quarantined_official_train": len(manifest.quarantined_official_train),
                "manifest": (args.output_dir / "manifest.json").as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    if (args.risk_examples_output is None) != (args.pipeline_sha256 is None):
        raise ValueError("--risk-examples-output and --pipeline-sha256 must be supplied together")
    predictions = load_metasyn_predictions(args.predictions)
    evaluation = evaluate_metasyn_predictions(
        manifest_path=args.manifest,
        predictions=predictions,
        evaluation_split=args.split,
    )
    atomic_write_json(args.output, evaluation, force=args.force)
    risk_count = None
    if args.risk_examples_output is not None:
        examples = build_metasyn_risk_examples(
            manifest_path=args.manifest,
            predictions=predictions,
            pipeline_sha256=args.pipeline_sha256,
            population_id=args.population_id,
            label_source="benchmark_annotation",
        )
        atomic_write_jsonl(args.risk_examples_output, examples, force=args.force)
        risk_count = len(examples)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "direction": evaluation["direction"],
                "retrieval": evaluation["retrieval"],
                "risk_examples": risk_count,
            },
            sort_keys=True,
        )
    )
    return 0


def _freeze_fixed_direction(args: argparse.Namespace) -> int:
    receipt = freeze_fixed_direction_baseline(
        manifest_path=args.manifest,
        split=args.split,
        direction=args.direction,
        selection_note=args.selection_note,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "predictions": (args.output_dir / receipt.predictions_path).as_posix(),
                "receipt": (args.output_dir / "freeze_receipt.json").as_posix(),
                "rows": receipt.rows,
                "class": receipt.predicted_class,
                "labels_opened": receipt.labels_opened,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "freeze-fixed-direction":
        return _freeze_fixed_direction(args)
    return _evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
