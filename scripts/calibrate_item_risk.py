#!/usr/bin/env python3
"""Freeze, calibrate, shift-check, and score artifact-backed item-risk bounds.

The four commands intentionally consume separate files.  In particular, ``calibrate``
recomputes the expected pipeline fingerprint before it opens either adjudicated unit
file, and it opens development units before calibration units.  Outputs are immutable,
self-hashed receipts; this CLI has no overwrite flag.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from literature_multiverse.item_risk_artifacts import (
    ExternalShiftDetectorReceipt,
    FixedRiskBinsReceipt,
    ItemRiskArtifactError,
    ItemRiskCalibrationRunReceipt,
    ItemRiskScoringRunReceipt,
    ProspectiveItemRiskInput,
    RiskBinDefinitionArtifact,
    ShiftAssessmentRunReceipt,
)
from literature_multiverse.item_risk_calibration import (
    ItemRiskCalibrationError,
    ItemRiskCalibrationUnit,
    calibrate_item_risk_bounds,
    make_fixed_risk_bin_family,
    score_item_risk_bound,
    seal_item_risk_candidate,
    seal_shift_assessment,
)
from literature_multiverse.lineage import (
    OutputExistsError,
    atomic_write_json,
    hash_canonical,
    sha256_bytes,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineFingerprint,
    PipelineFingerprintError,
    require_pipeline_fingerprint_match,
)

_PROBABILITY_KEYS = frozenset(
    {
        "error_probability",
        "manifest_probability",
        "probability",
        "probability_basis",
        "probability_source",
        "upper_error_probability",
    }
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser(
        "freeze-bins", help="Seal fixed bins from a prespecified/development-only JSON."
    )
    freeze.add_argument("--definition", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    calibrate = commands.add_parser(
        "calibrate",
        help="Calibrate from physically separate development and calibration JSONL files.",
    )
    calibrate.add_argument("--expected-pipeline", type=Path, required=True)
    calibrate.add_argument("--pipeline-root", type=Path, required=True)
    calibrate.add_argument("--fixed-bins", type=Path, required=True)
    calibrate.add_argument("--development-units", type=Path, required=True)
    calibrate.add_argument("--calibration-units", type=Path, required=True)
    calibrate.add_argument("--familywise-delta", type=float, required=True)
    calibrate.add_argument("--sampling-protocol-sha256", required=True)
    calibrate.add_argument("--error-event-definition", required=True)
    calibrate.add_argument("--shift-detector-id", required=True)
    calibrate.add_argument("--shift-detector-sha256", required=True)
    calibrate.add_argument(
        "--supported-domain",
        action="append",
        default=None,
        help="Repeat to restrict deployment to calibrated domains; default is all.",
    )
    calibrate.add_argument("--output", type=Path, required=True)

    shift = commands.add_parser(
        "assess-shift",
        help="Bind an external detector receipt and its exact artifact to a bundle.",
    )
    shift.add_argument("--calibration-run", type=Path, required=True)
    shift.add_argument("--detector-receipt", type=Path, required=True)
    shift.add_argument("--detector-artifact", type=Path, required=True)
    shift.add_argument("--output", type=Path, required=True)

    score = commands.add_parser(
        "score",
        help=(
            "Map prospective JSONL scheduling scores to proof-carrying group-average "
            "cell-rate UCLs."
        ),
    )
    score.add_argument("--calibration-run", type=Path, required=True)
    score.add_argument("--expected-pipeline", type=Path, required=True)
    score.add_argument("--pipeline-root", type=Path, required=True)
    score.add_argument("--shift-assessment", type=Path, required=True)
    score.add_argument("--candidates", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser


def _preflight_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise OutputExistsError(path.as_posix())


def _read_regular_bytes(path: Path, *, purpose: str) -> bytes:
    if path.is_symlink():
        raise ItemRiskArtifactError(f"{purpose}_symlink_forbidden:{path}")
    try:
        if not path.is_file():
            raise ItemRiskArtifactError(f"{purpose}_not_regular_file:{path}")
        return path.read_bytes()
    except OSError as exc:
        raise ItemRiskArtifactError(f"{purpose}_unreadable:{path}") from exc


def _json_payload(raw: bytes, *, purpose: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ItemRiskArtifactError(f"{purpose}_invalid_json") from exc


def _read_json_model[ModelT: BaseModel](
    path: Path,
    model: type[ModelT],
    *,
    purpose: str,
    access_order: list[str],
    event: str,
) -> tuple[ModelT, str]:
    raw = _read_regular_bytes(path, purpose=purpose)
    access_order.append(event)
    try:
        parsed = model.model_validate(_json_payload(raw, purpose=purpose))
    except ValidationError as exc:
        raise ItemRiskArtifactError(f"{purpose}_contract_invalid") from exc
    return parsed, sha256_bytes(raw)


def _read_jsonl_payloads(
    path: Path, *, purpose: str, access_order: list[str], event: str
) -> tuple[list[Any], str]:
    raw = _read_regular_bytes(path, purpose=purpose)
    access_order.append(event)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ItemRiskArtifactError(f"{purpose}_invalid_utf8") from exc
    payloads: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payloads.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ItemRiskArtifactError(
                f"{purpose}_invalid_json:line={line_number}"
            ) from exc
    if not payloads:
        raise ItemRiskArtifactError(f"{purpose}_empty")
    return payloads, sha256_bytes(raw)


def _validate_rows[ModelT: BaseModel](
    payloads: list[Any], model: type[ModelT], *, purpose: str
) -> list[ModelT]:
    try:
        return TypeAdapter(list[model]).validate_python(payloads)
    except ValidationError as exc:
        raise ItemRiskArtifactError(f"{purpose}_contract_invalid") from exc


def _forbid_manifest_probabilities(value: Any, *, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            if normalized in _PROBABILITY_KEYS or normalized.endswith("_probability"):
                raise ItemRiskArtifactError(
                    f"prospective_manifest_probability_forbidden:{location}.{key}"
                )
            _forbid_manifest_probabilities(nested, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _forbid_manifest_probabilities(nested, location=f"{location}[{index}]")


def _assert_physically_separate(left: Path, right: Path) -> None:
    if left.is_symlink() or right.is_symlink():
        raise ItemRiskArtifactError("calibration_unit_file_symlink_forbidden")
    try:
        left_resolved = left.resolve(strict=True)
        right_resolved = right.resolve(strict=True)
        left_stat = os.stat(left_resolved)
        right_stat = os.stat(right_resolved)
    except OSError as exc:
        raise ItemRiskArtifactError("calibration_unit_file_unavailable") from exc
    if (
        left_resolved == right_resolved
        or (left_stat.st_dev, left_stat.st_ino) == (right_stat.st_dev, right_stat.st_ino)
    ):
        raise ItemRiskArtifactError(
            "development_and_calibration_units_must_be_physically_separate"
        )


def _seal_receipt[ModelT: BaseModel](
    model: type[ModelT], payload: dict[str, Any]
) -> ModelT:
    return model.model_validate({**payload, "receipt_sha256": hash_canonical(payload)})


def _freeze_bins(args: argparse.Namespace) -> FixedRiskBinsReceipt:
    _preflight_output(args.output)
    access_order: list[str] = []
    definition, definition_file_sha256 = _read_json_model(
        args.definition,
        RiskBinDefinitionArtifact,
        purpose="risk_bin_definition",
        access_order=access_order,
        event="definition_opened",
    )
    try:
        family = make_fixed_risk_bin_family(
            edges=definition.edges,
            score_name=definition.score_name,
            score_model_sha256=definition.score_model_sha256,
            definition_source=definition.definition_source,
            definition_artifact_sha256=definition_file_sha256,
        )
    except (ItemRiskCalibrationError, ValidationError) as exc:
        raise ItemRiskArtifactError("risk_bin_definition_invalid") from exc
    access_order.append("fixed_bins_sealed")
    payload = {
        "receipt_version": "fixed-item-risk-bins-receipt-v1",
        "definition_file_sha256": definition_file_sha256,
        "definition": definition,
        "bin_family": family,
        "access_order": access_order,
    }
    receipt = _seal_receipt(FixedRiskBinsReceipt, payload)
    atomic_write_json(args.output, receipt)
    return receipt


def _calibrate(args: argparse.Namespace) -> ItemRiskCalibrationRunReceipt:
    _preflight_output(args.output)
    access_order: list[str] = []

    expected, expected_file_sha256 = _read_json_model(
        args.expected_pipeline,
        PipelineFingerprint,
        purpose="expected_pipeline_fingerprint",
        access_order=access_order,
        event="expected_pipeline_fingerprint_opened",
    )
    try:
        verification = require_pipeline_fingerprint_match(
            expected=expected, root=args.pipeline_root
        )
    except PipelineFingerprintError as exc:
        raise ItemRiskArtifactError("pipeline_fingerprint_not_matched") from exc
    access_order.append("pipeline_fingerprint_recomputed_and_matched")

    fixed_bins, fixed_bins_file_sha256 = _read_json_model(
        args.fixed_bins,
        FixedRiskBinsReceipt,
        purpose="fixed_risk_bins_receipt",
        access_order=access_order,
        event="fixed_bins_receipt_opened",
    )
    _assert_physically_separate(args.development_units, args.calibration_units)

    development_payloads, development_file_sha256 = _read_jsonl_payloads(
        args.development_units,
        purpose="development_units",
        access_order=access_order,
        event="development_units_opened",
    )
    development = _validate_rows(
        development_payloads, ItemRiskCalibrationUnit, purpose="development_units"
    )
    if any(row.split != "development" for row in development):
        raise ItemRiskArtifactError("development_units_contain_non_development_row")

    calibration_payloads, calibration_file_sha256 = _read_jsonl_payloads(
        args.calibration_units,
        purpose="calibration_units",
        access_order=access_order,
        event="calibration_units_opened",
    )
    calibration = _validate_rows(
        calibration_payloads, ItemRiskCalibrationUnit, purpose="calibration_units"
    )
    if any(row.split != "calibration" for row in calibration):
        raise ItemRiskArtifactError("calibration_units_contain_non_calibration_row")

    try:
        bundle = calibrate_item_risk_bounds(
            [*development, *calibration],
            pipeline_verification=verification,
            bin_family=fixed_bins.bin_family,
            familywise_delta=args.familywise_delta,
            sampling_protocol_sha256=args.sampling_protocol_sha256,
            error_event_definition=args.error_event_definition,
            shift_detector_id=args.shift_detector_id,
            shift_detector_sha256=args.shift_detector_sha256,
            supported_deployment_domains=args.supported_domain,
        )
    except (ItemRiskCalibrationError, ValidationError) as exc:
        raise ItemRiskArtifactError("item_risk_calibration_failed") from exc
    access_order.append("calibration_bundle_sealed")
    payload = {
        "receipt_version": "item-risk-calibration-run-v2",
        "expected_pipeline_file_sha256": expected_file_sha256,
        "fixed_bins_file_sha256": fixed_bins_file_sha256,
        "fixed_bins_receipt_sha256": fixed_bins.receipt_sha256,
        "development_units_file_sha256": development_file_sha256,
        "calibration_units_file_sha256": calibration_file_sha256,
        "development_unit_count": len(development),
        "calibration_unit_count": len(calibration),
        "pipeline_verification": verification,
        "bundle": bundle,
        "access_order": access_order,
    }
    receipt = _seal_receipt(ItemRiskCalibrationRunReceipt, payload)
    atomic_write_json(args.output, receipt)
    return receipt


def _assess_shift(args: argparse.Namespace) -> ShiftAssessmentRunReceipt:
    _preflight_output(args.output)
    access_order: list[str] = []
    calibration_run, calibration_run_file_sha256 = _read_json_model(
        args.calibration_run,
        ItemRiskCalibrationRunReceipt,
        purpose="item_risk_calibration_run",
        access_order=access_order,
        event="calibration_run_receipt_opened",
    )
    detector_receipt, detector_receipt_file_sha256 = _read_json_model(
        args.detector_receipt,
        ExternalShiftDetectorReceipt,
        purpose="external_shift_detector_receipt",
        access_order=access_order,
        event="external_detector_receipt_opened",
    )
    detector_artifact = _read_regular_bytes(
        args.detector_artifact, purpose="external_shift_detector_artifact"
    )
    detector_artifact_file_sha256 = sha256_bytes(detector_artifact)
    access_order.append("external_detector_artifact_opened_and_verified")
    bundle = calibration_run.bundle
    if (
        detector_receipt.calibration_bundle_sha256 != bundle.bundle_sha256
        or detector_receipt.detector_id != bundle.shift_detector_id
        or detector_receipt.detector_sha256 != bundle.shift_detector_sha256
    ):
        raise ItemRiskArtifactError("external_shift_detector_bundle_mismatch")
    if detector_artifact_file_sha256 != detector_receipt.detector_artifact_sha256:
        raise ItemRiskArtifactError("external_shift_detector_artifact_hash_mismatch")
    assessment = seal_shift_assessment(
        bundle=bundle,
        candidate_population_id=detector_receipt.candidate_population_id,
        candidate_domain=detector_receipt.candidate_domain,
        status=detector_receipt.status,
        assessment_input_sha256=detector_receipt.candidate_input_file_sha256,
        assessment_artifact_sha256=detector_artifact_file_sha256,
    )
    access_order.append("shift_assessment_sealed")
    payload = {
        "receipt_version": "item-risk-shift-run-v1",
        "calibration_run_file_sha256": calibration_run_file_sha256,
        "calibration_run_receipt_sha256": calibration_run.receipt_sha256,
        "detector_receipt_file_sha256": detector_receipt_file_sha256,
        "detector_receipt_sha256": detector_receipt.receipt_sha256,
        "detector_artifact_file_sha256": detector_artifact_file_sha256,
        "assessment": assessment,
        "access_order": access_order,
    }
    receipt = _seal_receipt(ShiftAssessmentRunReceipt, payload)
    atomic_write_json(args.output, receipt)
    return receipt


def _score(args: argparse.Namespace) -> ItemRiskScoringRunReceipt:
    _preflight_output(args.output)
    access_order: list[str] = []
    calibration_run, calibration_run_file_sha256 = _read_json_model(
        args.calibration_run,
        ItemRiskCalibrationRunReceipt,
        purpose="item_risk_calibration_run",
        access_order=access_order,
        event="calibration_run_receipt_opened",
    )
    expected, expected_file_sha256 = _read_json_model(
        args.expected_pipeline,
        PipelineFingerprint,
        purpose="expected_pipeline_fingerprint",
        access_order=access_order,
        event="expected_pipeline_fingerprint_opened",
    )
    if expected.pipeline_sha256 != calibration_run.bundle.pipeline_sha256:
        raise ItemRiskArtifactError("scoring_pipeline_does_not_match_calibration_bundle")
    try:
        verification = require_pipeline_fingerprint_match(
            expected=expected, root=args.pipeline_root
        )
    except PipelineFingerprintError as exc:
        raise ItemRiskArtifactError("pipeline_fingerprint_not_matched") from exc
    if (
        verification.verification_sha256
        != calibration_run.bundle.pipeline_verification_sha256
    ):
        raise ItemRiskArtifactError("scoring_pipeline_verification_mismatch")
    access_order.append("pipeline_fingerprint_recomputed_and_matched")
    shift_run, shift_run_file_sha256 = _read_json_model(
        args.shift_assessment,
        ShiftAssessmentRunReceipt,
        purpose="item_risk_shift_assessment",
        access_order=access_order,
        event="shift_assessment_receipt_opened",
    )
    if (
        shift_run.calibration_run_file_sha256 != calibration_run_file_sha256
        or shift_run.calibration_run_receipt_sha256 != calibration_run.receipt_sha256
        or shift_run.assessment.calibration_bundle_sha256
        != calibration_run.bundle.bundle_sha256
    ):
        raise ItemRiskArtifactError("shift_assessment_calibration_run_mismatch")

    candidate_payloads, candidate_input_file_sha256 = _read_jsonl_payloads(
        args.candidates,
        purpose="prospective_item_risk_candidates",
        access_order=access_order,
        event="prospective_candidates_opened",
    )
    _forbid_manifest_probabilities(candidate_payloads)
    candidate_inputs = _validate_rows(
        candidate_payloads,
        ProspectiveItemRiskInput,
        purpose="prospective_item_risk_candidates",
    )
    assessment = shift_run.assessment
    if assessment.assessment_input_sha256 != candidate_input_file_sha256:
        raise ItemRiskArtifactError("shift_assessment_candidate_input_hash_mismatch")
    item_ids = [row.item_id for row in candidate_inputs]
    if len(item_ids) != len(set(item_ids)):
        raise ItemRiskArtifactError("prospective_item_ids_must_be_unique")
    if any(
        row.population_id != assessment.candidate_population_id
        or row.domain != assessment.candidate_domain
        for row in candidate_inputs
    ):
        raise ItemRiskArtifactError("prospective_candidate_shift_scope_mismatch")
    candidate_inputs = sorted(candidate_inputs, key=lambda row: row.item_id)

    candidates = [
        seal_item_risk_candidate(
            item_id=row.item_id,
            question_id=row.question_id,
            paper_id=row.paper_id,
            population_id=row.population_id,
            domain=row.domain,
            pipeline_sha256=row.pipeline_sha256,
            score_model_sha256=row.score_model_sha256,
            score_input_sha256=row.score_input_sha256,
            risk_score=row.risk_score,
            shift_assessment=assessment,
        )
        for row in candidate_inputs
    ]
    bounds = [
        score_item_risk_bound(
            candidate=candidate,
            bundle=calibration_run.bundle,
            pipeline_verification=verification,
        )
        for candidate in candidates
    ]
    access_order.append("risk_bounds_scored")
    payload = {
        "receipt_version": "item-risk-scoring-run-v2",
        "calibration_run_file_sha256": calibration_run_file_sha256,
        "calibration_run_receipt_sha256": calibration_run.receipt_sha256,
        "calibration_bundle_sha256": calibration_run.bundle.bundle_sha256,
        "calibration_bundle": calibration_run.bundle,
        "expected_pipeline_file_sha256": expected_file_sha256,
        "shift_run_file_sha256": shift_run_file_sha256,
        "shift_run_receipt_sha256": shift_run.receipt_sha256,
        "candidate_input_file_sha256": candidate_input_file_sha256,
        "candidate_count": len(candidates),
        "pipeline_verification": verification,
        "candidates": candidates,
        "candidate_sha256s": [candidate.candidate_sha256 for candidate in candidates],
        "bounds": bounds,
        "access_order": access_order,
    }
    receipt = _seal_receipt(ItemRiskScoringRunReceipt, payload)
    atomic_write_json(args.output, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-bins":
        receipt = _freeze_bins(args)
        summary: dict[str, Any] = {
            "command": args.command,
            "receipt_sha256": receipt.receipt_sha256,
            "bin_family_sha256": receipt.bin_family.family_sha256,
            "bins": len(receipt.bin_family.bins),
        }
    elif args.command == "calibrate":
        receipt = _calibrate(args)
        summary = {
            "command": args.command,
            "receipt_sha256": receipt.receipt_sha256,
            "bundle_sha256": receipt.bundle.bundle_sha256,
            "development_units": receipt.development_unit_count,
            "calibration_units": receipt.calibration_unit_count,
        }
    elif args.command == "assess-shift":
        receipt = _assess_shift(args)
        summary = {
            "command": args.command,
            "receipt_sha256": receipt.receipt_sha256,
            "assessment_sha256": receipt.assessment.assessment_sha256,
            "status": receipt.assessment.status,
        }
    elif args.command == "score":
        receipt = _score(args)
        status_counts = Counter(bound.status for bound in receipt.bounds)
        summary = {
            "command": args.command,
            "receipt_sha256": receipt.receipt_sha256,
            "candidate_count": receipt.candidate_count,
            "status_counts": dict(sorted(status_counts.items())),
        }
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(f"unhandled_command:{args.command}")
    summary["output"] = args.output.as_posix()
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
