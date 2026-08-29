#!/usr/bin/env python3
"""Freeze, build, or externally replay decisive equal-cost/error frontiers v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from literature_multiverse.adaptive_calibration import AdaptiveCalibrationBundleV2
from literature_multiverse.decisive_claim_evaluation_v1 import (
    DecisiveClaimEvaluationResultV1,
)
from literature_multiverse.decisive_claim_evaluation_v2 import (
    DecisiveClaimEvaluationFrontiersV2,
    DecisiveFrontierConfigV2,
    build_decisive_claim_evaluation_frontiers_v2,
    freeze_decisive_frontier_config_v2,
    validate_decisive_claim_evaluation_frontiers_v2,
)
from literature_multiverse.lineage import atomic_write_json


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"decisive_evaluation_v2_json_not_object:{path}")
    return value


def _model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    return model.model_validate(_object(path))


def _calibrations(paths: list[Path]) -> list[AdaptiveCalibrationBundleV2]:
    return [_model(path, AdaptiveCalibrationBundleV2) for path in paths]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser(
        "freeze-config",
        help="freeze common realized-cost cutoffs and the released-error ceiling before scoring",
    )
    config.add_argument(
        "--realized-minutes-per-question-cutoff",
        type=float,
        action="append",
        required=True,
    )
    config.add_argument("--released-error-ceiling", type=float, required=True)
    config.add_argument("--bootstrap-draws", type=int, default=2000)
    config.add_argument("--bootstrap-seed", type=int, default=20260831)
    config.add_argument("--minimum-complete-evaluation-questions", type=int, default=20)
    config.add_argument("--minimum-questions-per-domain", type=int, default=2)
    config.add_argument("--output", type=Path, required=True)

    build = commands.add_parser(
        "build",
        help="compile an already-scored real decisive v1 result without opening label files",
    )
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--source-result", type=Path, required=True)
    build.add_argument("--repository-root", type=Path, required=True)
    build.add_argument("--calibration", type=Path, action="append", default=[])
    build.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser(
        "validate",
        help="externally reconstruct a saved v2 frontier from exact public inputs",
    )
    validate.add_argument("--artifact", type=Path, required=True)
    validate.add_argument("--source-result", type=Path, required=True)
    validate.add_argument("--repository-root", type=Path, required=True)
    validate.add_argument("--calibration", type=Path, action="append", default=[])
    return parser


def _summary(value: DecisiveClaimEvaluationFrontiersV2, *, output: Path | None) -> dict[str, Any]:
    return {
        "artifact_type": type(value).__name__,
        "result_sha256": value.result_sha256,
        "evaluator_component_sha256": value.evaluator_component_sha256,
        "source_result_sha256": value.source_anchor.source_result_sha256,
        "compiled_policy_points": len(value.compiled_policy_points),
        "realized_cost_frontier_rows": len(value.realized_cost_frontier),
        "fixed_error_frontier_rows": len(value.fixed_error_frontier),
        "realized_cost_frontier_claim_authority": (
            value.realized_cost_frontier_claim_authority
        ),
        "fixed_error_frontier_claim_authority": value.fixed_error_frontier_claim_authority,
        "scientific_claim_eligible": value.scientific_claim_eligible,
        "claim_release_authority": False,
        "authority_blockers": value.authority_blockers,
        "output": None if output is None else output.as_posix(),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-config":
        value = freeze_decisive_frontier_config_v2(
            common_realized_person_minutes_per_question_cutoffs=(
                args.realized_minutes_per_question_cutoff
            ),
            released_error_ceiling=args.released_error_ceiling,
            bootstrap_draws=args.bootstrap_draws,
            bootstrap_seed=args.bootstrap_seed,
            minimum_complete_evaluation_questions=(
                args.minimum_complete_evaluation_questions
            ),
            minimum_questions_per_domain=args.minimum_questions_per_domain,
        )
        atomic_write_json(args.output, value, force=False)
        print(
            json.dumps(
                {
                    "artifact_type": type(value).__name__,
                    "config_sha256": value.config_sha256,
                    "claim_release_authority": False,
                    "output": args.output.as_posix(),
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "build":
        value = build_decisive_claim_evaluation_frontiers_v2(
            source_result=_model(args.source_result, DecisiveClaimEvaluationResultV1),
            config=_model(args.config, DecisiveFrontierConfigV2),
            repository_root=args.repository_root,
            calibration_bundles=_calibrations(args.calibration),
        )
        atomic_write_json(args.output, value, force=False)
        print(json.dumps(_summary(value, output=args.output), sort_keys=True))
        return 0

    value = _model(args.artifact, DecisiveClaimEvaluationFrontiersV2)
    validate_decisive_claim_evaluation_frontiers_v2(
        result=value,
        source_result=_model(args.source_result, DecisiveClaimEvaluationResultV1),
        repository_root=args.repository_root,
        calibration_bundles=_calibrations(args.calibration),
    )
    print(json.dumps(_summary(value, output=None), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
