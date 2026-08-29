#!/usr/bin/env python3
"""Prepare, freeze, score, replay, or exercise decisive claim evaluation v1."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from literature_multiverse.decisive_claim_evaluation_v1 import (
    DecisiveClaimEvaluationResultV1,
    DecisiveEvaluationConfigV1,
    DecisiveEvaluationReadinessV1,
    DecisiveMechanicsFixtureReceiptV1,
    DecisivePolicyFreezeV1,
    DecisiveSplitManifestV1,
    EvaluationLabelManifestV1,
    FitStageReceiptV1,
    TrajectoryBundleV1,
    assess_decisive_evaluation_readiness_v1,
    build_decisive_mechanics_fixture_v1,
    freeze_decisive_policy_trajectories_v1,
    replay_decisive_policy_freeze_v1,
    score_decisive_claim_evaluation_v1,
    validate_decisive_claim_evaluation_result_v1,
)
from literature_multiverse.lineage import atomic_write_json


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"decisive_evaluation_v1_json_not_object:{path}")
    return value


def _model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    return model.model_validate(_object(path))


def _optional_model[ModelT: BaseModel](path: Path | None, model: type[ModelT]) -> ModelT | None:
    if path is None or not path.is_file():
        return None
    return _model(path, model)


def _time(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("decisive_evaluation_v1_timestamp_requires_timezone")
    return parsed


def _write(value: BaseModel, *, output: Path) -> None:
    atomic_write_json(output, value, force=False)


def _summary(value: BaseModel, output: Path | None = None) -> dict[str, Any]:
    common: dict[str, Any] = {
        "artifact_type": type(value).__name__,
        "claim_release_authority": False,
        "output": output.as_posix() if output is not None else None,
    }
    if isinstance(value, DecisiveEvaluationReadinessV1):
        return {
            **common,
            "artifact_sha256": value.readiness_sha256,
            "status": value.status,
            "blockers": value.blockers,
            "evaluation_label_contents_opened": False,
            "custody_portable": False,
            "real_scored_run_candidate": value.real_scored_run_candidate,
            "compiler_lineage_external_replay_verified": (
                value.compilation_replay_proof is not None
            ),
        }
    if isinstance(value, DecisivePolicyFreezeV1):
        return {
            **common,
            "artifact_sha256": value.freeze_sha256,
            "portable_semantic_artifact": True,
            "evaluation_reference_labels_opened": False,
            "policy_population_count": len(value.policy_populations),
            "evaluation_question_count": len(value.evaluation_question_ids),
            "compiler_lineage_external_replay_verified": (
                value.compilation_replay_proof is not None
            ),
        }
    if isinstance(value, DecisiveClaimEvaluationResultV1):
        return {
            **common,
            "artifact_sha256": value.result_sha256,
            "scientific_claim_eligible": value.scientific_claim_eligible,
            "empirical_scope": value.empirical_scope,
            "evaluation_question_count": len(value.opened_labels),
            "policy_population_count": len(value.scored_policy_populations),
            "paired_comparison_count": len(value.paired_policy_comparisons),
        }
    if isinstance(value, DecisiveMechanicsFixtureReceiptV1):
        return {
            **common,
            "artifact_sha256": value.fixture_receipt_sha256,
            "real_empirical_evidence": False,
        }
    raise TypeError("decisive_evaluation_v1_summary_type_unsupported")


def _shared_required(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--config", type=Path, required=True)
    subparser.add_argument("--repository-root", type=Path, required=True)


def _semantic_inputs(subparser: argparse.ArgumentParser, *, required: bool) -> None:
    subparser.add_argument("--split-manifest", type=Path, required=required)
    subparser.add_argument("--development-receipt", type=Path, required=required)
    subparser.add_argument("--calibration-receipt", type=Path, required=required)
    subparser.add_argument("--trajectory-bundle", type=Path, required=required)
    subparser.add_argument("--label-manifest", type=Path, required=required)
    subparser.add_argument("--label-root", type=Path, required=required)


def _compilation_inputs(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--trajectory-compilation-result",
        type=Path,
        help="Exact compiler result/receipt file; required for real trajectories",
    )
    subparser.add_argument(
        "--trajectory-compilation-source-roster",
        type=Path,
        help="Exact source-roster file used by the trajectory compiler",
    )
    subparser.add_argument(
        "--trajectory-compilation-source-root",
        type=Path,
        help="Root containing the roster's relative workspace and adjudication sources",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    readiness = commands.add_parser(
        "readiness",
        help="lstat sealed evaluation labels and emit a local nonportable custody receipt",
    )
    _shared_required(readiness)
    _semantic_inputs(readiness, required=False)
    _compilation_inputs(readiness)
    readiness.add_argument("--assessed-at", required=True)
    readiness.add_argument("--output", type=Path, required=True)

    freeze = commands.add_parser(
        "freeze",
        help="freeze portable label-blind policy trajectories before evaluation scoring",
    )
    _shared_required(freeze)
    _semantic_inputs(freeze, required=True)
    _compilation_inputs(freeze)
    freeze.add_argument("--custody", type=Path, required=True)
    freeze.add_argument("--frozen-at", required=True)
    freeze.add_argument("--output", type=Path, required=True)

    score = commands.add_parser(
        "score", help="replay the freeze, then open exact reference labels and score"
    )
    score.add_argument("--freeze", type=Path, required=True)
    score.add_argument("--custody", type=Path, required=True)
    score.add_argument("--repository-root", type=Path, required=True)
    score.add_argument("--label-root", type=Path, required=True)
    score.add_argument("--scored-at", required=True)
    score.add_argument("--output", type=Path, required=True)
    _compilation_inputs(score)

    validate = commands.add_parser("validate", help="externally replay a saved artifact")
    validate.add_argument("--kind", choices=("custody", "freeze", "result"), required=True)
    validate.add_argument("--artifact", type=Path, required=True)
    validate.add_argument("--repository-root", type=Path, required=True)
    validate.add_argument("--label-root", type=Path, required=True)
    validate.add_argument("--custody", type=Path)
    validate.add_argument("--config", type=Path)
    validate.add_argument("--split-manifest", type=Path)
    validate.add_argument("--development-receipt", type=Path)
    validate.add_argument("--calibration-receipt", type=Path)
    validate.add_argument("--trajectory-bundle", type=Path)
    validate.add_argument("--label-manifest", type=Path)
    _compilation_inputs(validate)

    fixture = commands.add_parser(
        "fixture", help="materialize a planted mechanics-only non-empirical evaluation"
    )
    _shared_required(fixture)
    fixture.add_argument("--output-root", type=Path, required=True)
    return parser


def _require_path(value: Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f"decisive_evaluation_v1_validation_argument_missing:{label}")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "readiness":
        config = _model(args.config, DecisiveEvaluationConfigV1)
        value = assess_decisive_evaluation_readiness_v1(
            config=config,
            repository_root=args.repository_root,
            assessed_at=_time(args.assessed_at),
            split_manifest=_optional_model(args.split_manifest, DecisiveSplitManifestV1),
            development_receipt=_optional_model(args.development_receipt, FitStageReceiptV1),
            calibration_receipt=_optional_model(args.calibration_receipt, FitStageReceiptV1),
            trajectory_bundle=_optional_model(args.trajectory_bundle, TrajectoryBundleV1),
            trajectory_compilation_result_path=args.trajectory_compilation_result,
            trajectory_compilation_source_roster_path=(args.trajectory_compilation_source_roster),
            trajectory_compilation_source_root=(args.trajectory_compilation_source_root),
            label_manifest=_optional_model(args.label_manifest, EvaluationLabelManifestV1),
            label_root=(
                args.label_root
                if args.label_root is not None and args.label_root.is_dir()
                else None
            ),
        )
        _write(value, output=args.output)
        print(json.dumps(_summary(value, args.output), sort_keys=True))
        return 0

    if args.command == "freeze":
        config = _model(args.config, DecisiveEvaluationConfigV1)
        value = freeze_decisive_policy_trajectories_v1(
            config=config,
            readiness=_model(args.custody, DecisiveEvaluationReadinessV1),
            split_manifest=_model(args.split_manifest, DecisiveSplitManifestV1),
            development_receipt=_model(args.development_receipt, FitStageReceiptV1),
            calibration_receipt=_model(args.calibration_receipt, FitStageReceiptV1),
            trajectory_bundle=_model(args.trajectory_bundle, TrajectoryBundleV1),
            trajectory_compilation_result_path=args.trajectory_compilation_result,
            trajectory_compilation_source_roster_path=(args.trajectory_compilation_source_roster),
            trajectory_compilation_source_root=(args.trajectory_compilation_source_root),
            label_manifest=_model(args.label_manifest, EvaluationLabelManifestV1),
            label_root=args.label_root,
            repository_root=args.repository_root,
            frozen_at=_time(args.frozen_at),
        )
        _write(value, output=args.output)
        print(json.dumps(_summary(value, args.output), sort_keys=True))
        return 0

    if args.command == "score":
        value = score_decisive_claim_evaluation_v1(
            frozen=_model(args.freeze, DecisivePolicyFreezeV1),
            custody=_model(args.custody, DecisiveEvaluationReadinessV1),
            repository_root=args.repository_root,
            label_root=args.label_root,
            scored_at=_time(args.scored_at),
            trajectory_compilation_result_path=args.trajectory_compilation_result,
            trajectory_compilation_source_roster_path=(args.trajectory_compilation_source_roster),
            trajectory_compilation_source_root=(args.trajectory_compilation_source_root),
        )
        _write(value, output=args.output)
        print(json.dumps(_summary(value, args.output), sort_keys=True))
        return 0

    if args.command == "fixture":
        value = build_decisive_mechanics_fixture_v1(
            output_root=args.output_root,
            repository_root=args.repository_root,
            config=_model(args.config, DecisiveEvaluationConfigV1),
        )
        print(
            json.dumps(
                _summary(value, args.output_root / "fixture-receipt.json"),
                sort_keys=True,
            )
        )
        return 0

    if args.kind == "custody":
        custody = _model(args.artifact, DecisiveEvaluationReadinessV1)
        replayed = assess_decisive_evaluation_readiness_v1(
            config=_model(_require_path(args.config, "config"), DecisiveEvaluationConfigV1),
            repository_root=args.repository_root,
            assessed_at=custody.assessed_at,
            split_manifest=_model(
                _require_path(args.split_manifest, "split_manifest"),
                DecisiveSplitManifestV1,
            ),
            development_receipt=_model(
                _require_path(args.development_receipt, "development_receipt"),
                FitStageReceiptV1,
            ),
            calibration_receipt=_model(
                _require_path(args.calibration_receipt, "calibration_receipt"),
                FitStageReceiptV1,
            ),
            trajectory_bundle=_model(
                _require_path(args.trajectory_bundle, "trajectory_bundle"),
                TrajectoryBundleV1,
            ),
            trajectory_compilation_result_path=args.trajectory_compilation_result,
            trajectory_compilation_source_roster_path=(args.trajectory_compilation_source_roster),
            trajectory_compilation_source_root=(args.trajectory_compilation_source_root),
            label_manifest=_model(
                _require_path(args.label_manifest, "label_manifest"),
                EvaluationLabelManifestV1,
            ),
            label_root=args.label_root,
        )
        if replayed != custody:
            raise ValueError("decisive_evaluation_v1_custody_external_replay_mismatch")
        value: BaseModel = custody
    elif args.kind == "freeze":
        value = replay_decisive_policy_freeze_v1(
            frozen=_model(args.artifact, DecisivePolicyFreezeV1),
            custody=_model(
                _require_path(args.custody, "custody"),
                DecisiveEvaluationReadinessV1,
            ),
            repository_root=args.repository_root,
            label_root=args.label_root,
            trajectory_compilation_result_path=args.trajectory_compilation_result,
            trajectory_compilation_source_roster_path=(args.trajectory_compilation_source_roster),
            trajectory_compilation_source_root=(args.trajectory_compilation_source_root),
        )
    else:
        value = validate_decisive_claim_evaluation_result_v1(
            result=_model(args.artifact, DecisiveClaimEvaluationResultV1),
            custody=_model(
                _require_path(args.custody, "custody"),
                DecisiveEvaluationReadinessV1,
            ),
            repository_root=args.repository_root,
            label_root=args.label_root,
            trajectory_compilation_result_path=args.trajectory_compilation_result,
            trajectory_compilation_source_roster_path=(args.trajectory_compilation_source_roster),
            trajectory_compilation_source_root=(args.trajectory_compilation_source_root),
        )
    print(json.dumps(_summary(value), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
