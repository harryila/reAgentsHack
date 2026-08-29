#!/usr/bin/env python3
"""Prepare, materialize, calibrate, or replay MetaSyn item-risk artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.lineage import atomic_write_json
from literature_multiverse.metasyn_grounded_analysis_v2 import MetaSynGroundedAnalysisV2
from literature_multiverse.metasyn_item_risk_calibration_v1 import (
    ArtifactBackedItemRiskAssignmentV1,
    MetaSynItemRiskCalibrationRunV1,
    MetaSynItemRiskPreparationV1,
    MetaSynTerminalRiskFeatureSetV1,
    TerminalRiskFeatureRowV1,
    assign_artifact_backed_item_risk_v1,
    calibrate_metasyn_item_risk_v1,
    materialize_metasyn_terminal_risk_features_v1,
    prepare_metasyn_item_risk_calibration_v1,
    validate_artifact_backed_item_risk_assignment_v1,
    validate_metasyn_item_risk_calibration_run_v1,
    validate_metasyn_item_risk_preparation_v1,
    validate_metasyn_terminal_risk_features_v1,
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metasyn_item_risk_v1_json_not_object:{path}")
    return value


def _analysis(path: Path) -> MetaSynGroundedAnalysisV2:
    return MetaSynGroundedAnalysisV2.model_validate(_object(path))


def _summary(value: object, *, output: Path | None = None) -> dict[str, Any]:
    common: dict[str, Any] = {
        "accuracy_claim_authority": False,
        "claim_release_authority": False,
        "evaluation_labels_opened": False,
    }
    if output is not None:
        common["output"] = output.as_posix()
    if isinstance(value, MetaSynItemRiskPreparationV1):
        return {
            **common,
            "artifact_kind": "preparation",
            "artifact_sha256": value.preparation_sha256,
            "status": value.status,
            "eligible_item_count": value.eligible_item_count,
            "calibration_question_count": value.split.calibration_question_count,
            "evaluation_question_count": value.split.evaluation_question_count,
            "blockers": value.preparation_blockers,
        }
    if isinstance(value, MetaSynTerminalRiskFeatureSetV1):
        return {
            **common,
            "artifact_kind": "feature_set",
            "artifact_sha256": value.feature_set_sha256,
            "feature_row_count": value.feature_row_count,
            "shift_status": value.shift_assessment.status,
            "scores_computed_not_supplied": True,
        }
    if isinstance(value, MetaSynItemRiskCalibrationRunV1):
        return {
            **common,
            "artifact_kind": "calibration_run",
            "artifact_sha256": value.calibration_run_sha256,
            "status": value.status,
            "labels_opened": value.labels_opened,
            "bound_input_question_count": value.bound_input_question_count,
            "scheduling_authority": value.scheduling_authority,
            "blockers": value.blockers,
        }
    if isinstance(value, ArtifactBackedItemRiskAssignmentV1):
        return {
            **common,
            "artifact_kind": "risk_assignment",
            "artifact_sha256": value.assignment_sha256,
            "item_id": value.item_id,
            "domain": value.domain,
            "bin_id": value.bin_id,
            "conservative_group_upper_error_rate": (value.conservative_group_upper_error_rate),
            "scheduling_authority": True,
        }
    raise TypeError("metasyn_item_risk_v1_summary_type_unsupported")


def _write_and_print(value: object, *, output: Path, force: bool) -> None:
    atomic_write_json(output, value, force=force)
    print(json.dumps(_summary(value, output=output), sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="replay source and freeze label-blind roster/split/pipeline"
    )
    prepare.add_argument("--analysis", type=Path, required=True)
    prepare.add_argument("--repository-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--force", action="store_true")

    materialize = subparsers.add_parser(
        "materialize", help="recompute terminal-derived features and scheduling scores"
    )
    materialize.add_argument("--analysis", type=Path, required=True)
    materialize.add_argument("--preparation", type=Path, required=True)
    materialize.add_argument("--repository-root", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--force", action="store_true")

    calibrate = subparsers.add_parser(
        "calibrate", help="open only the calibration sidecar after complete preflight"
    )
    calibrate.add_argument("--analysis", type=Path, required=True)
    calibrate.add_argument("--preparation", type=Path, required=True)
    calibrate.add_argument("--features", type=Path, required=True)
    calibrate.add_argument("--repository-root", type=Path, required=True)
    calibrate.add_argument("--adjudication-sidecar-dir", type=Path)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--force", action="store_true")

    assign = subparsers.add_parser(
        "assign", help="replay one computed feature row and assign its conservative group UCL"
    )
    assign.add_argument("--analysis", type=Path, required=True)
    assign.add_argument("--preparation", type=Path, required=True)
    assign.add_argument("--features", type=Path, required=True)
    assign.add_argument("--calibration-run", type=Path, required=True)
    assign.add_argument("--feature-row", type=Path, required=True)
    assign.add_argument("--repository-root", type=Path, required=True)
    assign.add_argument("--adjudication-sidecar-dir", type=Path, required=True)
    assign.add_argument("--output", type=Path, required=True)
    assign.add_argument("--force", action="store_true")

    validate = subparsers.add_parser("validate", help="externally replay one saved artifact")
    validate.add_argument(
        "--kind",
        choices=("preparation", "features", "calibration", "assignment"),
        required=True,
    )
    validate.add_argument("--analysis", type=Path, required=True)
    validate.add_argument("--artifact", type=Path, required=True)
    validate.add_argument("--preparation", type=Path)
    validate.add_argument("--features", type=Path)
    validate.add_argument("--calibration-run", type=Path)
    validate.add_argument("--repository-root", type=Path, required=True)
    validate.add_argument("--adjudication-sidecar-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    analysis = _analysis(args.analysis)
    if args.command == "prepare":
        value = prepare_metasyn_item_risk_calibration_v1(
            analysis=analysis,
            repository_root=args.repository_root,
        )
        _write_and_print(value, output=args.output, force=args.force)
        return 0

    if args.command == "materialize":
        preparation = MetaSynItemRiskPreparationV1.model_validate(_object(args.preparation))
        value = materialize_metasyn_terminal_risk_features_v1(
            preparation=preparation,
            analysis=analysis,
            repository_root=args.repository_root,
        )
        _write_and_print(value, output=args.output, force=args.force)
        return 0

    if args.command == "calibrate":
        preparation = MetaSynItemRiskPreparationV1.model_validate(_object(args.preparation))
        features = MetaSynTerminalRiskFeatureSetV1.model_validate(_object(args.features))
        value = calibrate_metasyn_item_risk_v1(
            preparation=preparation,
            feature_set=features,
            analysis=analysis,
            repository_root=args.repository_root,
            adjudication_sidecar_directory=args.adjudication_sidecar_dir,
        )
        _write_and_print(value, output=args.output, force=args.force)
        return 0

    if args.command == "assign":
        preparation = MetaSynItemRiskPreparationV1.model_validate(_object(args.preparation))
        features = MetaSynTerminalRiskFeatureSetV1.model_validate(_object(args.features))
        calibration_run = MetaSynItemRiskCalibrationRunV1.model_validate(
            _object(args.calibration_run)
        )
        feature_row = TerminalRiskFeatureRowV1.model_validate(_object(args.feature_row))
        value = assign_artifact_backed_item_risk_v1(
            feature_row=feature_row,
            preparation=preparation,
            feature_set=features,
            calibration_run=calibration_run,
            analysis=analysis,
            repository_root=args.repository_root,
            adjudication_sidecar_directory=args.adjudication_sidecar_dir,
        )
        _write_and_print(value, output=args.output, force=args.force)
        return 0

    if args.kind == "preparation":
        value = validate_metasyn_item_risk_preparation_v1(
            preparation=MetaSynItemRiskPreparationV1.model_validate(_object(args.artifact)),
            analysis=analysis,
            repository_root=args.repository_root,
        )
    elif args.kind == "features":
        value = validate_metasyn_terminal_risk_features_v1(
            feature_set=MetaSynTerminalRiskFeatureSetV1.model_validate(_object(args.artifact)),
            analysis=analysis,
            repository_root=args.repository_root,
        )
    elif args.kind == "calibration":
        if args.preparation is None or args.features is None:
            raise ValueError(
                "metasyn_item_risk_v1_calibration_validation_requires_preparation_and_features"
            )
        value = validate_metasyn_item_risk_calibration_run_v1(
            calibration_run=MetaSynItemRiskCalibrationRunV1.model_validate(_object(args.artifact)),
            preparation=MetaSynItemRiskPreparationV1.model_validate(_object(args.preparation)),
            feature_set=MetaSynTerminalRiskFeatureSetV1.model_validate(_object(args.features)),
            analysis=analysis,
            repository_root=args.repository_root,
            adjudication_sidecar_directory=args.adjudication_sidecar_dir,
        )
    else:
        if (
            args.preparation is None
            or args.features is None
            or args.calibration_run is None
            or args.adjudication_sidecar_dir is None
        ):
            raise ValueError("metasyn_item_risk_v1_assignment_validation_requires_full_lineage")
        value = validate_artifact_backed_item_risk_assignment_v1(
            assignment=ArtifactBackedItemRiskAssignmentV1.model_validate(_object(args.artifact)),
            preparation=MetaSynItemRiskPreparationV1.model_validate(_object(args.preparation)),
            feature_set=MetaSynTerminalRiskFeatureSetV1.model_validate(_object(args.features)),
            calibration_run=MetaSynItemRiskCalibrationRunV1.model_validate(
                _object(args.calibration_run)
            ),
            analysis=analysis,
            repository_root=args.repository_root,
            adjudication_sidecar_directory=args.adjudication_sidecar_dir,
        )
    print(json.dumps({**_summary(value), "validated": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
