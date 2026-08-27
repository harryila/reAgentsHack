#!/usr/bin/env python3
"""Freeze a question-risk gate, then evaluate held-out test labels separately."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from literature_multiverse.calibration import (
    CalibrationContractError,
    FrozenCalibrationBundle,
    RiskExample,
    calibrate_release_policy,
    calibration_artifact,
    evaluate_frozen_calibration_bundle,
    evaluate_release_policy,
    fit_logistic_risk_model,
    freeze_calibration_bundle,
)
from literature_multiverse.lineage import atomic_write_json


def _add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--candidate-threshold",
        type=float,
        action="append",
        default=None,
        help=(
            "Predeclared risk-score threshold; repeat for a fixed family. "
            "By default every distinct development score is tested with correction."
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser(
        "freeze",
        help="Fit and calibrate from a JSONL that physically excludes test rows.",
    )
    freeze.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Development+calibration RiskExample JSONL (test rows forbidden).",
    )
    freeze.add_argument("--output", type=Path, required=True, help="Frozen bundle JSON.")
    _add_policy_arguments(freeze)
    freeze.add_argument("--force", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate-test",
        help="Open a test-only JSONL after loading and verifying a frozen bundle.",
    )
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument(
        "--input", type=Path, required=True, help="Held-out test-only RiskExample JSONL."
    )
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--expected-freeze-sha256",
        help="Optional externally recorded bundle hash; mismatch fails before test input opens.",
    )
    evaluate.add_argument("--force", action="store_true")

    diagnostic = subparsers.add_parser(
        "diagnostic-one-shot",
        help="Simulation-only compatibility path that loads all labels together.",
    )
    diagnostic.add_argument("--input", type=Path, required=True)
    diagnostic.add_argument("--output", type=Path, required=True)
    _add_policy_arguments(diagnostic)
    diagnostic.add_argument("--force", action="store_true")
    return parser


def _read_jsonl(path: Path) -> list[RiskExample]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CalibrationContractError(f"risk_input_unreadable:{path}") from exc
    payloads: list[Any] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payloads.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise CalibrationContractError(f"risk_input_invalid_json:line={line_number}") from exc
    try:
        return TypeAdapter(list[RiskExample]).validate_python(payloads)
    except ValidationError as exc:
        raise CalibrationContractError("risk_input_contract_invalid") from exc


def _read_bundle(path: Path) -> FrozenCalibrationBundle:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CalibrationContractError(f"frozen_bundle_unreadable:{path}") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationContractError("frozen_bundle_invalid_json") from exc
    try:
        return FrozenCalibrationBundle.model_validate(payload)
    except ValidationError as exc:
        raise CalibrationContractError("frozen_bundle_contract_invalid") from exc


def _freeze(args: argparse.Namespace) -> dict[str, Any]:
    examples = _read_jsonl(args.input)
    bundle = freeze_calibration_bundle(
        examples,
        alpha=args.alpha,
        delta=args.delta,
        seed=args.seed,
        candidate_thresholds=args.candidate_threshold,
    )
    atomic_write_json(args.output, bundle, force=args.force)
    return {
        "stage": "frozen_before_test_access",
        "status": bundle.policy.status,
        "threshold": bundle.policy.threshold,
        "bundle_sha256": bundle.bundle_sha256,
        "output": args.output.as_posix(),
    }


def _evaluate_test(args: argparse.Namespace) -> dict[str, Any]:
    # Deliberate order: validate and hash the freeze bundle before opening test labels.
    bundle = _read_bundle(args.bundle)
    if (
        args.expected_freeze_sha256 is not None
        and args.expected_freeze_sha256 != bundle.bundle_sha256
    ):
        raise CalibrationContractError("expected_freeze_sha256_mismatch")
    examples = _read_jsonl(args.input)
    artifact = evaluate_frozen_calibration_bundle(examples, bundle)
    atomic_write_json(args.output, artifact, force=args.force)
    evaluation = artifact["test_evaluation"]
    return {
        "stage": "held_out_test_after_freeze",
        "bundle_sha256": bundle.bundle_sha256,
        "test_coverage": evaluation["coverage"],
        "test_empirical_risk": evaluation["empirical_risk"],
        "output": args.output.as_posix(),
    }


def _diagnostic_one_shot(args: argparse.Namespace) -> dict[str, Any]:
    examples = _read_jsonl(args.input)
    if {row.label_source for row in examples} != {"simulation"}:
        raise CalibrationContractError("diagnostic_one_shot_requires_simulation_labels")
    model = fit_logistic_risk_model(examples, seed=args.seed)
    policy = calibrate_release_policy(
        examples,
        model,
        alpha=args.alpha,
        delta=args.delta,
        candidate_thresholds=args.candidate_threshold,
    )
    evaluation = evaluate_release_policy(examples, model, policy)
    artifact = calibration_artifact(
        examples=examples,
        model=model,
        policy=policy,
        evaluation=evaluation,
    )
    artifact["evaluation_stage"] = "simulation_only_one_shot_diagnostic"
    atomic_write_json(args.output, artifact, force=args.force)
    return {
        "stage": "simulation_only_one_shot_diagnostic",
        "status": policy.status,
        "threshold": policy.threshold,
        "test_coverage": evaluation.coverage,
        "test_empirical_risk": evaluation.empirical_risk,
        "output": args.output.as_posix(),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        summary = _freeze(args)
    elif args.command == "evaluate-test":
        summary = _evaluate_test(args)
    else:
        summary = _diagnostic_one_shot(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
