#!/usr/bin/env python3
"""Build frozen adaptive first-release calibration bundles with staged access.

The v1 commands use separate development-freeze and calibration-label stages. The
confirmation-aware v2 commands first freeze outcome-free collection sources, then
freeze development and the visible calibration roster, join externally replayed
held-out assessment receipts, and only then open reference labels. No command
accepts prospective test trajectories, and bare terminal-gate results are rejected.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from literature_multiverse.adaptive_calibration import (
    AdaptiveCalibrationError,
    AdaptiveDevelopmentFreeze,
    AdaptiveDevelopmentFreezeV2,
    AdaptivePolicyContext,
    ConditionCalibrationCollectionSourceRosterV1,
    GateCompleteCalibrationRosterV2,
    LabeledQuestionTrajectory,
    LabeledQuestionTrajectoryV2,
    PolicyVisibleQuestionTrajectory,
    PolicyVisibleQuestionTrajectoryV2,
    QuestionReferenceVerdict,
    QuestionReferenceVerdictV2,
    calibrate_adaptive_first_release,
    calibrate_confirmation_aware_first_release,
    fit_adaptive_development,
    fit_adaptive_development_v2,
    freeze_condition_calibration_collection_source_roster_v1,
    freeze_gate_complete_calibration_roster_v2,
    join_condition_calibration_assessment_receipts,
    join_labeled_question_trajectory,
)
from literature_multiverse.certificate import (
    ConditionCalibrationAssessmentReceiptV1,
    ConditionCalibrationCollectionSourceV1,
)
from literature_multiverse.lineage import (
    OutputExistsError,
    atomic_write_json,
    hash_canonical,
    sha256_bytes,
    sha256_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser(
        "freeze-development",
        help=("Fit on development labels and seal an exact label-free calibration roster."),
    )
    freeze.add_argument(
        "--development-trajectories",
        type=Path,
        required=True,
        help="Development-only LabeledQuestionTrajectory JSONL.",
    )
    freeze.add_argument(
        "--policy-contexts",
        type=Path,
        required=True,
        help="JSON array of frozen AdaptivePolicyContext objects.",
    )
    freeze.add_argument(
        "--calibration-visible-trajectories",
        type=Path,
        required=True,
        help="Calibration-only PolicyVisibleQuestionTrajectory JSONL; labels forbidden.",
    )
    freeze.add_argument("--alpha", type=float, default=0.10)
    freeze.add_argument("--delta", type=float, default=0.05)
    freeze.add_argument("--seed", type=int, default=20260827)
    freeze.add_argument(
        "--candidate-threshold",
        action="append",
        default=None,
        metavar="ARM_ID=THRESHOLD",
        help=(
            "Predeclare a threshold for one policy arm; repeat as needed. If used, "
            "every arm must appear. The default derives candidates from development only."
        ),
    )
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace an existing regular output file.",
    )

    calibrate = commands.add_parser(
        "build-calibration-bundle",
        help="Open the calibration reference sidecar only after verifying the freeze.",
    )
    calibrate.add_argument("--development-freeze", type=Path, required=True)
    calibrate.add_argument(
        "--expected-development-freeze-sha256",
        required=True,
        help="Self-hash printed by the earlier freeze-development stage.",
    )
    calibrate.add_argument(
        "--calibration-labels",
        type=Path,
        required=True,
        help="Calibration-only QuestionReferenceVerdict JSONL; exact roster required.",
    )
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace an existing regular output file.",
    )

    freeze_v2 = commands.add_parser(
        "freeze-development-v2",
        help=(
            "Freeze confirmation-aware development models and the outcome-free calibration roster."
        ),
    )
    freeze_v2.add_argument(
        "--development-trajectories",
        type=Path,
        required=True,
        help="Development-only LabeledQuestionTrajectoryV2 JSONL.",
    )
    freeze_v2.add_argument("--policy-contexts", type=Path, required=True)
    freeze_v2.add_argument(
        "--calibration-visible-trajectories",
        type=Path,
        required=True,
        help=(
            "Calibration-only PolicyVisibleQuestionTrajectoryV2 JSONL; terminal "
            "confirmation outcomes and references forbidden."
        ),
    )
    freeze_v2.add_argument(
        "--calibration-source-roster",
        type=Path,
        help=(
            "Externally replayed ConditionCalibrationCollectionSourceRosterV1. "
            "Omitting it creates an explicitly simulation-only, release-ineligible freeze."
        ),
    )
    freeze_v2.add_argument("--expected-calibration-source-roster-sha256")
    freeze_v2.add_argument("--expected-calibration-source-membership-sha256")
    freeze_v2.add_argument("--alpha", type=float, default=0.10)
    freeze_v2.add_argument("--delta", type=float, default=0.05)
    freeze_v2.add_argument("--seed", type=int, default=20260827)
    freeze_v2.add_argument(
        "--candidate-threshold",
        action="append",
        default=None,
        metavar="ARM_ID=THRESHOLD",
    )
    freeze_v2.add_argument("--output", type=Path, required=True)
    freeze_v2.add_argument("--force", action="store_true")

    gates_v2 = commands.add_parser(
        "freeze-terminal-gates-v2",
        help=(
            "Join every post-scheduler terminal confirmation result to the exact "
            "calibration roster before references open."
        ),
    )
    gates_v2.add_argument("--development-freeze", type=Path, required=True)
    gates_v2.add_argument("--expected-development-freeze-sha256", required=True)
    gates_v2.add_argument(
        "--calibration-assessment-receipts",
        type=Path,
        required=True,
        help=(
            "Full ConditionCalibrationAssessmentReceiptV1 JSONL; bare calibration "
            "or production gate results are forbidden."
        ),
    )
    gates_v2.add_argument("--expected-source-roster-sha256")
    gates_v2.add_argument("--expected-source-membership-sha256")
    gates_v2.add_argument("--output", type=Path, required=True)
    gates_v2.add_argument("--force", action="store_true")

    calibrate_v2 = commands.add_parser(
        "build-calibration-bundle-v2",
        help=(
            "Open v2 reference labels only after externally matching both the "
            "development freeze and complete terminal-gate roster."
        ),
    )
    calibrate_v2.add_argument("--development-freeze", type=Path, required=True)
    calibrate_v2.add_argument("--expected-development-freeze-sha256", required=True)
    calibrate_v2.add_argument("--gate-complete-roster", type=Path, required=True)
    calibrate_v2.add_argument("--expected-gate-complete-roster-sha256", required=True)
    calibrate_v2.add_argument(
        "--calibration-labels",
        type=Path,
        required=True,
        help="QuestionReferenceVerdictV2 JSONL; exact frozen roster required.",
    )
    calibrate_v2.add_argument("--output", type=Path, required=True)
    calibrate_v2.add_argument("--force", action="store_true")

    sources_v2 = commands.add_parser(
        "freeze-collection-sources-v2",
        help=(
            "Externally replay and freeze all outcome-free calibration collection "
            "sources before any held-out condition assessment opens."
        ),
    )
    sources_v2.add_argument("--collection-sources", type=Path, required=True)
    sources_v2.add_argument("--output", type=Path, required=True)
    sources_v2.add_argument("--force", action="store_true")
    return parser


def _preflight_output(path: Path, *, force: bool) -> None:
    if path.is_symlink():
        raise AdaptiveCalibrationError(f"adaptive_output_symlink_forbidden:{path}")
    if path.exists() and not force:
        raise OutputExistsError(path.as_posix())


def _reject_output_alias(output: Path, inputs: list[Path]) -> None:
    output_identity = output.resolve(strict=False)
    for input_path in inputs:
        if output_identity == input_path.resolve(strict=False):
            raise AdaptiveCalibrationError(f"adaptive_output_must_not_alias_input:{input_path}")


def _read_regular_bytes(path: Path, *, purpose: str) -> tuple[bytes, str]:
    if path.is_symlink():
        raise AdaptiveCalibrationError(f"{purpose}_symlink_forbidden:{path}")
    try:
        if not path.is_file():
            raise AdaptiveCalibrationError(f"{purpose}_not_regular_file:{path}")
        raw = path.read_bytes()
    except OSError as exc:
        raise AdaptiveCalibrationError(f"{purpose}_unreadable:{path}") from exc
    return raw, sha256_bytes(raw)


def _json_value(raw: bytes, *, purpose: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdaptiveCalibrationError(f"{purpose}_invalid_json") from exc


def _jsonl_values(
    raw: bytes,
    *,
    purpose: str,
    allow_empty: bool = False,
) -> list[Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AdaptiveCalibrationError(f"{purpose}_invalid_utf8") from exc
    values: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AdaptiveCalibrationError(f"{purpose}_invalid_json:line={line_number}") from exc
    if not values and not allow_empty:
        raise AdaptiveCalibrationError(f"{purpose}_empty")
    return values


def _read_json_model[ModelT: BaseModel](
    path: Path,
    model: type[ModelT],
    *,
    purpose: str,
) -> tuple[ModelT, str]:
    raw, file_sha256 = _read_regular_bytes(path, purpose=purpose)
    try:
        value = model.model_validate(_json_value(raw, purpose=purpose))
    except ValidationError as exc:
        raise AdaptiveCalibrationError(f"{purpose}_contract_invalid") from exc
    return value, file_sha256


def _read_json_array[ModelT: BaseModel](
    path: Path,
    model: type[ModelT],
    *,
    purpose: str,
) -> tuple[list[ModelT], str]:
    raw, file_sha256 = _read_regular_bytes(path, purpose=purpose)
    try:
        value = TypeAdapter(list[model]).validate_python(_json_value(raw, purpose=purpose))
    except ValidationError as exc:
        raise AdaptiveCalibrationError(f"{purpose}_contract_invalid") from exc
    if not value:
        raise AdaptiveCalibrationError(f"{purpose}_empty")
    return value, file_sha256


def _read_jsonl_models[ModelT: BaseModel](
    path: Path,
    model: type[ModelT],
    *,
    purpose: str,
    allow_empty: bool = False,
) -> tuple[list[ModelT], str]:
    raw, file_sha256 = _read_regular_bytes(path, purpose=purpose)
    try:
        value = TypeAdapter(list[model]).validate_python(
            _jsonl_values(raw, purpose=purpose, allow_empty=allow_empty)
        )
    except ValidationError as exc:
        raise AdaptiveCalibrationError(f"{purpose}_contract_invalid") from exc
    return value, file_sha256


def _candidate_thresholds(values: list[str] | None) -> dict[str, list[float]] | None:
    if values is None:
        return None
    parsed: defaultdict[str, list[float]] = defaultdict(list)
    for raw in values:
        arm_id, separator, threshold_text = raw.partition("=")
        if not separator or not arm_id.strip() or not threshold_text.strip():
            raise AdaptiveCalibrationError(f"adaptive_candidate_threshold_argument_invalid:{raw}")
        try:
            threshold = float(threshold_text)
        except ValueError as exc:
            raise AdaptiveCalibrationError(
                f"adaptive_candidate_threshold_argument_invalid:{raw}"
            ) from exc
        parsed[arm_id.strip()].append(threshold)
    return dict(parsed)


def _receipt(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "receipt_sha256": hash_canonical(payload)}


def _freeze_development(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = [
        args.development_trajectories,
        args.policy_contexts,
        args.calibration_visible_trajectories,
    ]
    _reject_output_alias(args.output, input_paths)
    _preflight_output(args.output, force=args.force)
    access_order: list[str] = []

    contexts, context_file_sha256 = _read_json_array(
        args.policy_contexts,
        AdaptivePolicyContext,
        purpose="adaptive_policy_contexts",
    )
    access_order.append("policy_contexts_opened_and_validated")
    development, development_file_sha256 = _read_jsonl_models(
        args.development_trajectories,
        LabeledQuestionTrajectory,
        purpose="adaptive_development_trajectories",
    )
    access_order.append("development_labels_opened_and_validated")
    calibration_visible, calibration_visible_file_sha256 = _read_jsonl_models(
        args.calibration_visible_trajectories,
        PolicyVisibleQuestionTrajectory,
        purpose="adaptive_calibration_visible_trajectories",
    )
    access_order.append("label_free_calibration_roster_opened_and_validated")
    freeze = fit_adaptive_development(
        development,
        policy_contexts=contexts,
        calibration_visible_trajectories=calibration_visible,
        alpha=args.alpha,
        delta=args.delta,
        candidate_thresholds=_candidate_thresholds(args.candidate_threshold),
        seed=args.seed,
    )
    access_order.append("development_freeze_computed_before_calibration_label_access")
    atomic_write_json(args.output, freeze, force=args.force)
    access_order.append("development_freeze_atomically_written")
    return _receipt(
        {
            "receipt_version": "adaptive-calibration-cli-receipt-v1",
            "stage": "development_frozen_before_calibration_labels_opened",
            "status": "frozen",
            "access_order": access_order,
            "calibration_labels_opened": False,
            "test_rows_accepted": False,
            "development_freeze_sha256": freeze.development_freeze_sha256,
            "development_freeze_file_sha256": sha256_file(args.output),
            "development_trajectories_file_sha256": development_file_sha256,
            "policy_contexts_file_sha256": context_file_sha256,
            "calibration_visible_trajectories_file_sha256": (calibration_visible_file_sha256),
            "development_question_count": len(freeze.development.question_ids),
            "calibration_roster_question_count": len(
                freeze.calibration_roster.visible_trajectories
            ),
            "candidate_count": len(freeze.threshold_family.candidates),
            "output": args.output.as_posix(),
        }
    )


def _freeze_development_v2(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = [
        args.development_trajectories,
        args.policy_contexts,
        args.calibration_visible_trajectories,
    ]
    source_binding_args = (
        args.calibration_source_roster,
        args.expected_calibration_source_roster_sha256,
        args.expected_calibration_source_membership_sha256,
    )
    if any(value is not None for value in source_binding_args) and not all(
        value is not None for value in source_binding_args
    ):
        raise AdaptiveCalibrationError("adaptive_v2_calibration_source_roster_binding_incomplete")
    if args.calibration_source_roster is not None:
        input_paths.append(args.calibration_source_roster)
    _reject_output_alias(args.output, input_paths)
    _preflight_output(args.output, force=args.force)
    access_order: list[str] = []

    contexts, context_file_sha256 = _read_json_array(
        args.policy_contexts,
        AdaptivePolicyContext,
        purpose="adaptive_v2_policy_contexts",
    )
    access_order.append("policy_contexts_opened_and_validated")
    source_roster = None
    source_roster_file_sha256 = None
    if args.calibration_source_roster is not None:
        source_roster, source_roster_file_sha256 = _read_json_model(
            args.calibration_source_roster,
            ConditionCalibrationCollectionSourceRosterV1,
            purpose="adaptive_v2_calibration_collection_source_roster",
        )
        access_order.append("outcome_free_collection_source_roster_opened_and_externally_replayed")
        if (
            args.expected_calibration_source_roster_sha256 != source_roster.source_roster_sha256
            or args.expected_calibration_source_membership_sha256
            != source_roster.source_membership_sha256
        ):
            raise AdaptiveCalibrationError("expected_v2_calibration_source_roster_binding_mismatch")
        access_order.append("external_collection_source_roster_and_membership_hashes_matched")
    development, development_file_sha256 = _read_jsonl_models(
        args.development_trajectories,
        LabeledQuestionTrajectoryV2,
        purpose="adaptive_v2_development_trajectories",
    )
    access_order.append("development_labels_opened_and_validated")
    calibration_visible, calibration_visible_file_sha256 = _read_jsonl_models(
        args.calibration_visible_trajectories,
        PolicyVisibleQuestionTrajectoryV2,
        purpose="adaptive_v2_calibration_visible_trajectories",
    )
    access_order.append("calibration_roster_opened_with_terminal_outcomes_and_references_unopened")
    freeze = fit_adaptive_development_v2(
        development,
        policy_contexts=contexts,
        calibration_visible_trajectories=calibration_visible,
        calibration_collection_source_roster=source_roster,
        alpha=args.alpha,
        delta=args.delta,
        candidate_thresholds=_candidate_thresholds(args.candidate_threshold),
        seed=args.seed,
    )
    access_order.append(
        "development_freeze_computed_before_terminal_outcomes_or_calibration_labels"
    )
    atomic_write_json(args.output, freeze, force=args.force)
    access_order.append("development_freeze_atomically_written")
    return _receipt(
        {
            "receipt_version": "adaptive-calibration-cli-receipt-v2",
            "stage": "confirmation_aware_development_frozen",
            "status": "frozen",
            "access_order": access_order,
            "calibration_assessment_receipts_opened": False,
            "calibration_labels_opened": False,
            "test_rows_accepted": False,
            "development_freeze_sha256": freeze.development_freeze_sha256,
            "development_freeze_file_sha256": sha256_file(args.output),
            "development_trajectories_file_sha256": development_file_sha256,
            "policy_contexts_file_sha256": context_file_sha256,
            "calibration_visible_trajectories_file_sha256": (calibration_visible_file_sha256),
            "collection_source_status": (freeze.calibration_roster.collection_source_status),
            "collection_source_roster_sha256": (
                freeze.calibration_roster.collection_source_roster_sha256
            ),
            "collection_source_membership_sha256": (
                freeze.calibration_roster.collection_source_membership_sha256
            ),
            "collection_source_roster_file_sha256": source_roster_file_sha256,
            "development_question_count": len(freeze.development_trajectories),
            "calibration_roster_question_count": len(
                freeze.calibration_roster.visible_trajectories
            ),
            "candidate_count": len(freeze.base_freeze.threshold_family.candidates),
            "independence_verified": freeze.independence_verified,
            "output": args.output.as_posix(),
        }
    )


def _freeze_collection_sources_v2(args: argparse.Namespace) -> dict[str, Any]:
    _reject_output_alias(args.output, [args.collection_sources])
    _preflight_output(args.output, force=args.force)
    sources, source_file_sha256 = _read_jsonl_models(
        args.collection_sources,
        ConditionCalibrationCollectionSourceV1,
        purpose="adaptive_v2_outcome_free_collection_sources",
    )
    roster = freeze_condition_calibration_collection_source_roster_v1(sources)
    atomic_write_json(args.output, roster, force=args.force)
    return _receipt(
        {
            "receipt_version": "adaptive-calibration-cli-receipt-v2",
            "stage": "outcome_free_collection_source_roster_frozen",
            "status": "frozen",
            "access_order": [
                "outcome_free_collection_sources_opened_and_externally_replayed",
                "source_membership_frozen_before_condition_assessment_access",
                "collection_source_roster_atomically_written",
            ],
            "condition_assessments_opened": False,
            "calibration_labels_opened": False,
            "collection_sources_file_sha256": source_file_sha256,
            "source_roster_sha256": roster.source_roster_sha256,
            "source_membership_sha256": roster.source_membership_sha256,
            "source_roster_file_sha256": sha256_file(args.output),
            "source_count": len(roster.collection_sources),
            "question_count": len({row.question_id for row in roster.source_anchors}),
            "output": args.output.as_posix(),
        }
    )


def _freeze_terminal_gates_v2(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = [args.development_freeze, args.calibration_assessment_receipts]
    _reject_output_alias(args.output, input_paths)
    _preflight_output(args.output, force=args.force)
    access_order: list[str] = []

    freeze, freeze_file_sha256 = _read_json_model(
        args.development_freeze,
        AdaptiveDevelopmentFreezeV2,
        purpose="adaptive_v2_development_freeze",
    )
    access_order.append("development_freeze_opened_and_integrity_validated")
    if args.expected_development_freeze_sha256 != freeze.development_freeze_sha256:
        raise AdaptiveCalibrationError("expected_v2_development_freeze_sha256_mismatch")
    access_order.append("external_development_freeze_sha256_matched")
    expected_pairs = sorted(
        (
            visible.base_visible.question_id,
            arm.base_arm.policy_arm_id,
        )
        for visible in freeze.calibration_roster.visible_trajectories
        for arm in visible.arms
        if arm.terminal_condition_required
    )
    supplied_source_binding = (
        args.expected_source_roster_sha256,
        args.expected_source_membership_sha256,
    )
    if expected_pairs:
        if (
            freeze.calibration_roster.collection_source_status
            != "externally_replayed_before_assessment"
            or args.expected_source_roster_sha256
            != freeze.calibration_roster.collection_source_roster_sha256
            or args.expected_source_membership_sha256
            != freeze.calibration_roster.collection_source_membership_sha256
        ):
            raise AdaptiveCalibrationError("expected_v2_collection_source_roster_binding_mismatch")
        access_order.append("external_preoutcome_source_roster_and_membership_hashes_matched")
    elif any(value is not None for value in supplied_source_binding):
        if not all(value is not None for value in supplied_source_binding) or (
            args.expected_source_roster_sha256
            != freeze.calibration_roster.collection_source_roster_sha256
            or args.expected_source_membership_sha256
            != freeze.calibration_roster.collection_source_membership_sha256
        ):
            raise AdaptiveCalibrationError("expected_v2_collection_source_roster_binding_mismatch")
        access_order.append("optional_preoutcome_source_roster_and_membership_hashes_matched")

    # This is the first operation in this process that may open held-out
    # confirmation outcomes. Receipt validation externally reruns its exact source.
    receipts, receipt_file_sha256 = _read_jsonl_models(
        args.calibration_assessment_receipts,
        ConditionCalibrationAssessmentReceiptV1,
        purpose="adaptive_v2_calibration_assessment_receipts",
        allow_empty=not expected_pairs,
    )
    access_order.append(
        "calibration_assessment_receipts_opened_after_source_and_development_freezes_matched"
    )
    observed_pairs = [(row.question_id, row.policy_arm_id) for row in receipts]
    if observed_pairs != sorted(set(observed_pairs)):
        raise AdaptiveCalibrationError(
            "adaptive_v2_calibration_assessment_receipts_not_unique_sorted"
        )
    if observed_pairs != expected_pairs:
        raise AdaptiveCalibrationError("adaptive_v2_calibration_assessment_receipt_roster_mismatch")
    receipt_by_pair = {(row.question_id, row.policy_arm_id): row for row in receipts}
    gate_complete = [
        join_condition_calibration_assessment_receipts(
            visible=visible,
            calibration_roster=freeze.calibration_roster,
            calibration_assessment_receipts=[
                receipt_by_pair[(visible.base_visible.question_id, arm.base_arm.policy_arm_id)]
                for arm in visible.arms
                if arm.terminal_condition_required
            ],
        )
        for visible in freeze.calibration_roster.visible_trajectories
    ]
    roster = freeze_gate_complete_calibration_roster_v2(
        development_freeze=freeze,
        trajectories=gate_complete,
    )
    access_order.append("complete_terminal_gate_roster_frozen_before_reference_access")
    atomic_write_json(args.output, roster, force=args.force)
    access_order.append("gate_complete_roster_atomically_written")
    return _receipt(
        {
            "receipt_version": "adaptive-calibration-cli-receipt-v2",
            "stage": "terminal_gate_roster_frozen_before_calibration_labels",
            "status": "frozen",
            "access_order": access_order,
            "calibration_assessment_receipts_opened": True,
            "calibration_labels_opened": False,
            "test_labels_opened": False,
            "development_freeze_sha256": freeze.development_freeze_sha256,
            "development_freeze_file_sha256": freeze_file_sha256,
            "calibration_assessment_receipts_file_sha256": receipt_file_sha256,
            "collection_source_roster_sha256": (roster.collection_source_roster_sha256),
            "collection_source_membership_sha256": (roster.collection_source_membership_sha256),
            "gate_complete_roster_sha256": roster.gate_roster_sha256,
            "gate_complete_roster_file_sha256": sha256_file(args.output),
            "calibration_question_count": len(roster.trajectories),
            "calibration_assessment_receipt_count": len(receipts),
            "output": args.output.as_posix(),
        }
    )


def _read_question_references(path: Path) -> tuple[list[QuestionReferenceVerdict], str]:
    return _read_jsonl_models(
        path,
        QuestionReferenceVerdict,
        purpose="adaptive_calibration_labels",
    )


def _join_exact_calibration_roster(
    freeze: AdaptiveDevelopmentFreeze,
    references: list[QuestionReferenceVerdict],
) -> list[LabeledQuestionTrajectory]:
    reference_ids = [row.question_id for row in references]
    if reference_ids != sorted(set(reference_ids)):
        raise AdaptiveCalibrationError("adaptive_calibration_label_questions_must_be_sorted_unique")
    expected_ids = [row.question_id for row in freeze.calibration_roster.visible_trajectories]
    if reference_ids != expected_ids:
        raise AdaptiveCalibrationError("adaptive_calibration_label_roster_mismatch")
    by_question = {row.question_id: row for row in references}
    return [
        join_labeled_question_trajectory(
            visible=visible,
            reference=by_question[visible.question_id],
        )
        for visible in freeze.calibration_roster.visible_trajectories
    ]


def _build_calibration_bundle(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = [args.development_freeze, args.calibration_labels]
    _reject_output_alias(args.output, input_paths)
    _preflight_output(args.output, force=args.force)
    access_order: list[str] = []

    freeze, freeze_file_sha256 = _read_json_model(
        args.development_freeze,
        AdaptiveDevelopmentFreeze,
        purpose="adaptive_development_freeze",
    )
    access_order.append("development_freeze_opened_and_integrity_validated")
    if args.expected_development_freeze_sha256 != freeze.development_freeze_sha256:
        raise AdaptiveCalibrationError("expected_development_freeze_sha256_mismatch")
    access_order.append("external_development_freeze_sha256_matched")

    # This is the first operation that opens the calibration-label file.
    references, reference_file_sha256 = _read_question_references(args.calibration_labels)
    access_order.append("calibration_labels_opened_after_freeze_match")
    calibration = _join_exact_calibration_roster(freeze, references)
    access_order.append("calibration_labels_joined_to_exact_frozen_roster")
    bundle = calibrate_adaptive_first_release(freeze, calibration)
    access_order.append("calibration_bundle_computed_with_test_labels_unopened")
    atomic_write_json(args.output, bundle, force=args.force)
    access_order.append("calibration_bundle_atomically_written")
    return _receipt(
        {
            "receipt_version": "adaptive-calibration-cli-receipt-v1",
            "stage": "calibration_bundle_created_after_development_freeze",
            "status": bundle.status,
            "access_order": access_order,
            "calibration_labels_opened": True,
            "test_labels_opened": False,
            "development_freeze_sha256": freeze.development_freeze_sha256,
            "development_freeze_file_sha256": freeze_file_sha256,
            "calibration_labels_file_sha256": reference_file_sha256,
            "bundle_sha256": bundle.bundle_sha256,
            "bundle_file_sha256": sha256_file(args.output),
            "selected_candidate_sha256": bundle.selected_candidate_sha256,
            "calibration_question_count": len(bundle.calibration.question_ids),
            "candidate_count": len(bundle.candidates),
            "output": args.output.as_posix(),
        }
    )


def _read_question_references_v2(
    path: Path,
) -> tuple[list[QuestionReferenceVerdictV2], str]:
    return _read_jsonl_models(
        path,
        QuestionReferenceVerdictV2,
        purpose="adaptive_v2_calibration_labels",
    )


def _build_calibration_bundle_v2(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = [
        args.development_freeze,
        args.gate_complete_roster,
        args.calibration_labels,
    ]
    _reject_output_alias(args.output, input_paths)
    _preflight_output(args.output, force=args.force)
    access_order: list[str] = []

    freeze, freeze_file_sha256 = _read_json_model(
        args.development_freeze,
        AdaptiveDevelopmentFreezeV2,
        purpose="adaptive_v2_development_freeze",
    )
    access_order.append("development_freeze_opened_and_integrity_validated")
    if args.expected_development_freeze_sha256 != freeze.development_freeze_sha256:
        raise AdaptiveCalibrationError("expected_v2_development_freeze_sha256_mismatch")
    access_order.append("external_development_freeze_sha256_matched")

    gate_roster, gate_roster_file_sha256 = _read_json_model(
        args.gate_complete_roster,
        GateCompleteCalibrationRosterV2,
        purpose="adaptive_v2_gate_complete_roster",
    )
    access_order.append("gate_complete_roster_opened_and_integrity_validated")
    if args.expected_gate_complete_roster_sha256 != gate_roster.gate_roster_sha256:
        raise AdaptiveCalibrationError("expected_v2_gate_complete_roster_sha256_mismatch")
    if (
        gate_roster.development_freeze_sha256 != freeze.development_freeze_sha256
        or gate_roster.calibration_roster_sha256 != freeze.calibration_roster.roster_sha256
    ):
        raise AdaptiveCalibrationError("adaptive_v2_gate_roster_freeze_mismatch")
    access_order.append("external_gate_complete_roster_sha256_and_lineage_matched")

    # This is the first operation that opens calibration references.
    references, reference_file_sha256 = _read_question_references_v2(args.calibration_labels)
    access_order.append("calibration_labels_opened_after_both_external_freeze_matches")
    bundle = calibrate_confirmation_aware_first_release(
        freeze,
        gate_roster,
        references,
    )
    access_order.append("joint_calibration_bundle_computed_with_test_labels_unopened")
    atomic_write_json(args.output, bundle, force=args.force)
    access_order.append("calibration_bundle_atomically_written")
    return _receipt(
        {
            "receipt_version": "adaptive-calibration-cli-receipt-v2",
            "stage": "confirmation_aware_calibration_bundle_frozen",
            "status": bundle.status,
            "access_order": access_order,
            "calibration_assessment_receipts_opened": True,
            "calibration_labels_opened": True,
            "test_labels_opened": False,
            "development_freeze_sha256": freeze.development_freeze_sha256,
            "development_freeze_file_sha256": freeze_file_sha256,
            "gate_complete_roster_sha256": gate_roster.gate_roster_sha256,
            "gate_complete_roster_file_sha256": gate_roster_file_sha256,
            "calibration_labels_file_sha256": reference_file_sha256,
            "bundle_sha256": bundle.bundle_sha256,
            "bundle_file_sha256": sha256_file(args.output),
            "selected_candidate_sha256": bundle.selected_candidate_sha256,
            "real_release_eligible": bundle.real_release_eligible,
            "calibration_question_count": len(bundle.calibration.question_ids),
            "candidate_count": len(bundle.candidates),
            "condition_release_domains": bundle.calibration.domains,
            "simultaneous_test_count": (
                len(bundle.candidates) * (1 + len(bundle.calibration.domains))
            ),
            "candidates_with_complete_condition_domain_support": sum(
                all(
                    stratum.confirmed_condition_releases > 0
                    for stratum in candidate.condition_domain_calibrations
                )
                for candidate in bundle.candidates
            ),
            "output": args.output.as_posix(),
        }
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-development":
        receipt = _freeze_development(args)
    elif args.command == "build-calibration-bundle":
        receipt = _build_calibration_bundle(args)
    elif args.command == "freeze-development-v2":
        receipt = _freeze_development_v2(args)
    elif args.command == "freeze-collection-sources-v2":
        receipt = _freeze_collection_sources_v2(args)
    elif args.command == "freeze-terminal-gates-v2":
        receipt = _freeze_terminal_gates_v2(args)
    elif args.command == "build-calibration-bundle-v2":
        receipt = _build_calibration_bundle_v2(args)
    else:  # pragma: no cover - argparse enforces the closed command set
        raise AssertionError(f"unhandled_command:{args.command}")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
