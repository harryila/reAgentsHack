#!/usr/bin/env python3
"""Run the staged, label-firewalled EvidenceBench grounding diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.evidencebench_diagnostic import (
    EvidenceBenchFrozenPlanV1,
    EvidenceBenchMaterializationReceiptV1,
    EvidenceBenchPredictionReceiptV1,
    EvidenceBenchPredictionRowV1,
    EvidenceBenchPrivateGoldV1,
    EvidenceBenchPublicSummaryV1,
    EvidenceBenchVisibleQuestionV1,
    audit_evidencebench_run,
    materialize_evidencebench_test,
    predict_evidencebench_test,
    prepare_evidencebench_plan,
    score_evidencebench_test,
    validate_evidencebench_prediction_freeze,
    write_evidencebench_materialization,
    write_evidencebench_plan,
    write_evidencebench_predictions,
)
from literature_multiverse.lineage import OutputExistsError, atomic_write_json


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _plan(path: Path) -> EvidenceBenchFrozenPlanV1:
    return EvidenceBenchFrozenPlanV1.model_validate(_json(path))


def _materialization_receipt(
    path: Path,
) -> EvidenceBenchMaterializationReceiptV1:
    return EvidenceBenchMaterializationReceiptV1.model_validate(_json(path))


def _prediction_receipt(path: Path) -> EvidenceBenchPredictionReceiptV1:
    return EvidenceBenchPredictionReceiptV1.model_validate(_json(path))


def _visible(path: Path) -> list[EvidenceBenchVisibleQuestionV1]:
    payload = _json(path)
    if not isinstance(payload, list):
        raise ValueError("evidencebench_visible_projection_not_list")
    return [EvidenceBenchVisibleQuestionV1.model_validate(row) for row in payload]


def _gold(path: Path) -> list[EvidenceBenchPrivateGoldV1]:
    payload = _json(path)
    if not isinstance(payload, list):
        raise ValueError("evidencebench_private_gold_not_list")
    return [EvidenceBenchPrivateGoldV1.model_validate(row) for row in payload]


def _predictions(path: Path) -> list[EvidenceBenchPredictionRowV1]:
    payload = _json(path)
    if not isinstance(payload, list):
        raise ValueError("evidencebench_predictions_not_list")
    return [EvidenceBenchPredictionRowV1.model_validate(row) for row in payload]


def _require_fresh_outputs(*paths: Path) -> None:
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("evidencebench_output_paths_alias")
    existing = [
        path.as_posix() for path in paths if path.exists() or path.is_symlink()
    ]
    if existing:
        raise OutputExistsError(",".join(sorted(existing)))


def _score_stage(
    args: argparse.Namespace,
    *,
    plan: EvidenceBenchFrozenPlanV1,
    materialization_receipt: EvidenceBenchMaterializationReceiptV1,
) -> EvidenceBenchPublicSummaryV1:
    """Validate the prediction freeze before invoking the private-gold reader."""

    prediction_receipt = _prediction_receipt(args.prediction_receipt)
    predictions = _predictions(args.predictions)
    validate_evidencebench_prediction_freeze(
        plan=plan,
        materialization_receipt=materialization_receipt,
        predictions=predictions,
        prediction_receipt=prediction_receipt,
    )
    return score_evidencebench_test(
        plan=plan,
        gold=_gold(args.gold),
        materialization_receipt=materialization_receipt,
        predictions=predictions,
        prediction_receipt=prediction_receipt,
    )


def _add_common_materialized_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="select on development data without accepting a test path"
    )
    prepare.add_argument("--development", type=Path, required=True)
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--bootstrap-seed", type=int, default=20260828)
    prepare.add_argument("--bootstrap-replicates", type=int, default=5000)

    materialize = subparsers.add_parser(
        "materialize", help="split pinned test bytes after the plan is frozen"
    )
    materialize.add_argument("--plan", type=Path, required=True)
    materialize.add_argument("--test", type=Path, required=True)
    materialize.add_argument("--visible", type=Path, required=True)
    materialize.add_argument("--gold", type=Path, required=True)
    materialize.add_argument("--receipt", type=Path, required=True)

    predict = subparsers.add_parser(
        "predict", help="rank sentences from the label-free projection only"
    )
    _add_common_materialized_inputs(predict)
    predict.add_argument("--visible", type=Path, required=True)
    predict.add_argument("--predictions", type=Path, required=True)
    predict.add_argument("--receipt", type=Path, required=True)

    score = subparsers.add_parser(
        "score", help="join frozen predictions to private gold and aggregate"
    )
    _add_common_materialized_inputs(score)
    score.add_argument("--gold", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--prediction-receipt", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)

    audit = subparsers.add_parser(
        "audit", help="replay every stage and compare exact frozen artifacts"
    )
    audit.add_argument("--development", type=Path, required=True)
    audit.add_argument("--test", type=Path, required=True)
    audit.add_argument("--plan", type=Path, required=True)
    audit.add_argument("--visible", type=Path, required=True)
    audit.add_argument("--gold", type=Path, required=True)
    audit.add_argument("--materialization-receipt", type=Path, required=True)
    audit.add_argument("--predictions", type=Path, required=True)
    audit.add_argument("--prediction-receipt", type=Path, required=True)
    audit.add_argument("--summary", type=Path, required=True)
    audit.add_argument("--receipt", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.stage == "prepare":
        _require_fresh_outputs(args.plan)
        plan = prepare_evidencebench_plan(
            development_path=args.development,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        write_evidencebench_plan(args.plan, plan)
        print(json.dumps(plan.model_dump(mode="json"), sort_keys=True))
        return

    plan = _plan(args.plan)
    if args.stage == "materialize":
        _require_fresh_outputs(args.visible, args.gold, args.receipt)
        visible, gold, receipt = materialize_evidencebench_test(
            plan=plan,
            raw_test_path=args.test,
        )
        write_evidencebench_materialization(
            visible_path=args.visible,
            gold_path=args.gold,
            receipt_path=args.receipt,
            visible=visible,
            gold=gold,
            receipt=receipt,
        )
        print(json.dumps(receipt.model_dump(mode="json"), sort_keys=True))
        return

    materialization_receipt = _materialization_receipt(
        args.materialization_receipt
    )
    if args.stage == "predict":
        _require_fresh_outputs(args.predictions, args.receipt)
        predictions, receipt = predict_evidencebench_test(
            plan=plan,
            visible=_visible(args.visible),
            materialization_receipt=materialization_receipt,
        )
        write_evidencebench_predictions(
            predictions_path=args.predictions,
            receipt_path=args.receipt,
            predictions=predictions,
            receipt=receipt,
        )
        print(json.dumps(receipt.model_dump(mode="json"), sort_keys=True))
        return

    if args.stage == "score":
        _require_fresh_outputs(args.output)
        summary = _score_stage(
            args,
            plan=plan,
            materialization_receipt=materialization_receipt,
        )
        atomic_write_json(args.output, summary.model_dump(mode="json"))
        print(json.dumps(summary.model_dump(mode="json"), sort_keys=True))
        return

    _require_fresh_outputs(args.receipt)
    frozen_visible = _visible(args.visible)
    frozen_predictions = _predictions(args.predictions)
    frozen_prediction_receipt = _prediction_receipt(args.prediction_receipt)
    validate_evidencebench_prediction_freeze(
        plan=plan,
        materialization_receipt=materialization_receipt,
        predictions=frozen_predictions,
        prediction_receipt=frozen_prediction_receipt,
    )
    frozen_gold = _gold(args.gold)
    frozen_summary = EvidenceBenchPublicSummaryV1.model_validate(_json(args.summary))
    audit_receipt = audit_evidencebench_run(
        development_path=args.development,
        raw_test_path=args.test,
        plan=plan,
        visible=frozen_visible,
        gold=frozen_gold,
        materialization_receipt=materialization_receipt,
        predictions=frozen_predictions,
        prediction_receipt=frozen_prediction_receipt,
        summary=frozen_summary,
    )
    atomic_write_json(args.receipt, audit_receipt.model_dump(mode="json"))
    print(json.dumps(audit_receipt.model_dump(mode="json"), sort_keys=True))


if __name__ == "__main__":
    main()
