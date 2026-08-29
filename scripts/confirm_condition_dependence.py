#!/usr/bin/env python3
"""Materialize, prepare, fit, confirm, and replay held-out condition dependence.

``materialize`` is an explicit independent-custodian stage that opens the full graph
after target freeze and emits separate private partitions plus a content-silent receipt.
``prepare`` accepts only the outcome-blind identity/predictor roster and that receipt.
``fit`` accepts only the exact development graph. The analysis-side ``confirm`` command
first opens the full graph after externally supplied plan, model, and pipeline identities
match. Confirmation outputs are exclusive and an exact rerun is idempotent; they are
never overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from literature_multiverse.condition_confirmation import (
    ConditionConfirmationAssessmentV1,
    ConditionConfirmationConfigV1,
    ConditionConfirmationError,
    ConditionConfirmationFrozenModelV1,
    ConditionConfirmationMaterializationReceiptV1,
    ConditionConfirmationPlanV1,
    ConditionConfirmationTargetV1,
    LabelFreeGraphRosterV1,
    confirm_condition_dependence,
    fit_condition_confirmation_model,
    materialize_condition_confirmation_inputs,
    prepare_condition_confirmation_plan,
    validate_condition_confirmation_assessment,
    validate_condition_confirmation_materialization,
)
from literature_multiverse.evidence_graph import EvidenceGraph
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

    materialize = commands.add_parser(
        "materialize",
        help=(
            "Custodian-only: project a content-silent roster and exact private graph "
            "partitions from the frozen full graph."
        ),
    )
    _add_materialization_inputs(materialize)
    materialize.add_argument("--roster-output", type=Path, required=True)
    materialize.add_argument("--development-graph-output", type=Path, required=True)
    materialize.add_argument("--confirmation-graph-output", type=Path, required=True)
    materialize.add_argument("--receipt-output", type=Path, required=True)

    validate_materialization = commands.add_parser(
        "validate-materialization",
        help="Exact-replay every custodian projection and partition artifact.",
    )
    _add_materialization_inputs(validate_materialization)
    validate_materialization.add_argument("--roster", type=Path, required=True)
    validate_materialization.add_argument(
        "--development-graph", type=Path, required=True
    )
    validate_materialization.add_argument(
        "--confirmation-graph", type=Path, required=True
    )
    validate_materialization.add_argument("--receipt", type=Path, required=True)
    validate_materialization.add_argument(
        "--expected-receipt-sha256", required=True
    )

    prepare = commands.add_parser(
        "prepare",
        help="Freeze a component split without opening any effect or outcome value.",
    )
    prepare.add_argument("--target", type=Path, required=True)
    prepare.add_argument("--expected-target-sha256", required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--expected-config-sha256", required=True)
    prepare.add_argument("--roster", type=Path, required=True)
    prepare.add_argument("--expected-roster-sha256", required=True)
    prepare.add_argument("--materialization-receipt", type=Path, required=True)
    prepare.add_argument(
        "--expected-materialization-receipt-sha256", required=True
    )
    prepare.add_argument("--pipeline-sha256", required=True)
    prepare.add_argument(
        "--external-freeze-anchor",
        required=True,
        help="Externally recorded immutable commit, timestamp receipt, or registry anchor.",
    )
    prepare.add_argument("--output", type=Path, required=True)

    fit = commands.add_parser(
        "fit",
        help="Fit and freeze a model from the development partition only.",
    )
    fit.add_argument("--plan", type=Path, required=True)
    fit.add_argument("--expected-plan-sha256", required=True)
    fit.add_argument("--current-pipeline-sha256", required=True)
    fit.add_argument("--development-graph", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)

    confirm = commands.add_parser(
        "confirm",
        help="Run the one-shot held-out assessment after all frozen identities match.",
    )
    _add_confirmation_inputs(confirm)
    confirm.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser(
        "validate",
        help="Recompute the split, development model, and all confirmation metrics.",
    )
    _add_confirmation_inputs(validate)
    validate.add_argument("--assessment", type=Path, required=True)
    return parser


def _add_materialization_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--expected-target-sha256", required=True)
    parser.add_argument("--full-graph", type=Path, required=True)
    parser.add_argument("--expected-full-graph-sha256", required=True)


def _add_confirmation_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--current-pipeline-sha256", required=True)
    parser.add_argument("--full-graph", type=Path, required=True)


def _reject_output_alias(output: Path, inputs: list[Path]) -> None:
    output_identity = output.resolve(strict=False)
    for input_path in inputs:
        if output_identity == input_path.resolve(strict=False):
            raise ConditionConfirmationError(
                f"condition_confirmation_output_must_not_alias_input:{input_path}"
            )


def _reject_output_set_aliases(*, outputs: list[Path], inputs: list[Path]) -> None:
    resolved = [path.resolve(strict=False) for path in outputs]
    if len(resolved) != len(set(resolved)):
        raise ConditionConfirmationError(
            "condition_confirmation_materialization_outputs_must_be_distinct"
        )
    for output in outputs:
        _reject_output_alias(output, inputs)


def _preflight_new_output(path: Path) -> None:
    if path.is_symlink():
        raise ConditionConfirmationError(
            f"condition_confirmation_output_symlink_forbidden:{path}"
        )
    if path.exists():
        raise OutputExistsError(path.as_posix())


def _preflight_new_outputs(paths: list[Path]) -> None:
    for path in paths:
        _preflight_new_output(path)


def _read_regular_bytes(path: Path, *, purpose: str) -> tuple[bytes, str]:
    if path.is_symlink():
        raise ConditionConfirmationError(
            f"condition_confirmation_{purpose}_symlink_forbidden:{path}"
        )
    try:
        if not path.is_file():
            raise ConditionConfirmationError(
                f"condition_confirmation_{purpose}_not_regular_file:{path}"
            )
        raw = path.read_bytes()
    except OSError as exc:
        raise ConditionConfirmationError(
            f"condition_confirmation_{purpose}_unreadable:{path}"
        ) from exc
    return raw, sha256_bytes(raw)


def _read_json_model[ModelT: BaseModel](
    path: Path,
    model: type[ModelT],
    *,
    purpose: str,
) -> tuple[ModelT, str]:
    raw, file_sha256 = _read_regular_bytes(path, purpose=purpose)
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConditionConfirmationError(
            f"condition_confirmation_{purpose}_invalid_json"
        ) from exc
    try:
        value = model.model_validate(decoded)
    except ValidationError as exc:
        raise ConditionConfirmationError(
            f"condition_confirmation_{purpose}_contract_invalid"
        ) from exc
    return value, file_sha256


def _require_identity(*, expected: str, observed: str, purpose: str) -> None:
    if expected != observed:
        raise ConditionConfirmationError(
            f"condition_confirmation_expected_{purpose}_sha256_mismatch"
        )


def _receipt(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "receipt_sha256": hash_canonical(payload)}


def _read_materialization_source(
    args: argparse.Namespace,
) -> tuple[ConditionConfirmationTargetV1, EvidenceGraph, dict[str, str], list[str]]:
    access_order: list[str] = []
    target, target_file_sha256 = _read_json_model(
        args.target,
        ConditionConfirmationTargetV1,
        purpose="target",
    )
    _require_identity(
        expected=args.expected_target_sha256,
        observed=target.target_sha256,
        purpose="target",
    )
    access_order.append("frozen_target_opened_and_external_identity_matched")
    full_graph, full_graph_file_sha256 = _read_json_model(
        args.full_graph,
        EvidenceGraph,
        purpose="full_graph",
    )
    full_graph_sha256 = hash_canonical(full_graph)
    _require_identity(
        expected=args.expected_full_graph_sha256,
        observed=full_graph_sha256,
        purpose="full_graph",
    )
    access_order.append(
        "full_graph_outcomes_opened_by_independent_custodian_after_target_match"
    )
    return (
        target,
        full_graph,
        {
            "target_file_sha256": target_file_sha256,
            "full_graph_file_sha256": full_graph_file_sha256,
            "full_graph_sha256": full_graph_sha256,
        },
        access_order,
    )


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    inputs = [args.target, args.full_graph]
    outputs = [
        args.roster_output,
        args.development_graph_output,
        args.confirmation_graph_output,
        args.receipt_output,
    ]
    _reject_output_set_aliases(outputs=outputs, inputs=inputs)
    _preflight_new_outputs(outputs)
    target, full_graph, file_hashes, access_order = _read_materialization_source(args)
    roster, development_graph, confirmation_graph, receipt = (
        materialize_condition_confirmation_inputs(
            full_graph=full_graph,
            target=target,
        )
    )
    access_order.append(
        "strict_content_silent_roster_and_deterministic_partitions_derived"
    )
    atomic_write_json(args.roster_output, roster)
    atomic_write_json(args.development_graph_output, development_graph)
    atomic_write_json(args.confirmation_graph_output, confirmation_graph)
    atomic_write_json(args.receipt_output, receipt)
    access_order.append(
        "roster_private_partitions_and_commit_receipt_atomically_written_exclusive"
    )
    return _receipt(
        {
            "receipt_version": "condition-confirmation-cli-receipt-v1",
            "stage": "custodian_materialization_completed",
            "access_order": access_order,
            "full_graph_outcomes_opened_by_custodian": True,
            "effect_outcome_uncertainty_values_embedded_in_materialization_receipt": False,
            "target_sha256": target.target_sha256,
            "materialization_receipt_sha256": receipt.receipt_sha256,
            "roster_sha256": roster.roster_sha256,
            "development_graph_sha256": receipt.development_graph_sha256,
            "confirmation_graph_sha256": receipt.confirmation_graph_sha256,
            "roster_file_sha256": sha256_file(args.roster_output),
            "development_graph_file_sha256": sha256_file(
                args.development_graph_output
            ),
            "confirmation_graph_file_sha256": sha256_file(
                args.confirmation_graph_output
            ),
            "materialization_receipt_file_sha256": sha256_file(
                args.receipt_output
            ),
            **file_hashes,
        }
    )


def _validate_materialization(args: argparse.Namespace) -> dict[str, Any]:
    target, full_graph, file_hashes, access_order = _read_materialization_source(args)
    roster, roster_file_sha256 = _read_json_model(
        args.roster,
        LabelFreeGraphRosterV1,
        purpose="label_free_roster",
    )
    development_graph, development_file_sha256 = _read_json_model(
        args.development_graph,
        EvidenceGraph,
        purpose="development_graph",
    )
    confirmation_graph, confirmation_file_sha256 = _read_json_model(
        args.confirmation_graph,
        EvidenceGraph,
        purpose="confirmation_graph",
    )
    receipt, receipt_file_sha256 = _read_json_model(
        args.receipt,
        ConditionConfirmationMaterializationReceiptV1,
        purpose="materialization_receipt",
    )
    _require_identity(
        expected=args.expected_receipt_sha256,
        observed=receipt.receipt_sha256,
        purpose="materialization_receipt",
    )
    access_order.append("all_materialization_outputs_opened_and_identity_matched")
    validated = validate_condition_confirmation_materialization(
        full_graph=full_graph,
        target=target,
        roster=roster,
        development_graph=development_graph,
        confirmation_graph=confirmation_graph,
        receipt=receipt,
    )
    access_order.append("custodian_projection_and_partitions_exactly_recomputed")
    return _receipt(
        {
            "receipt_version": "condition-confirmation-cli-receipt-v1",
            "stage": "custodian_materialization_validated",
            "access_order": access_order,
            "full_graph_outcomes_opened_by_validator": True,
            "materialization_receipt_sha256": validated.receipt_sha256,
            "roster_sha256": roster.roster_sha256,
            "roster_file_sha256": roster_file_sha256,
            "development_graph_file_sha256": development_file_sha256,
            "confirmation_graph_file_sha256": confirmation_file_sha256,
            "materialization_receipt_file_sha256": receipt_file_sha256,
            **file_hashes,
        }
    )


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    inputs = [args.target, args.config, args.roster, args.materialization_receipt]
    _reject_output_alias(args.output, inputs)
    _preflight_new_output(args.output)
    access_order: list[str] = []
    target, target_file_sha256 = _read_json_model(
        args.target,
        ConditionConfirmationTargetV1,
        purpose="target",
    )
    _require_identity(
        expected=args.expected_target_sha256,
        observed=target.target_sha256,
        purpose="target",
    )
    access_order.append("self_hashed_target_opened_and_external_identity_matched")
    config, config_file_sha256 = _read_json_model(
        args.config,
        ConditionConfirmationConfigV1,
        purpose="config",
    )
    _require_identity(
        expected=args.expected_config_sha256,
        observed=config.config_sha256,
        purpose="config",
    )
    access_order.append("self_hashed_config_opened_and_external_identity_matched")
    roster, roster_file_sha256 = _read_json_model(
        args.roster,
        LabelFreeGraphRosterV1,
        purpose="label_free_roster",
    )
    _require_identity(
        expected=args.expected_roster_sha256,
        observed=roster.roster_sha256,
        purpose="roster",
    )
    access_order.append("strict_label_free_roster_opened_and_external_identity_matched")
    materialization_receipt, materialization_receipt_file_sha256 = _read_json_model(
        args.materialization_receipt,
        ConditionConfirmationMaterializationReceiptV1,
        purpose="materialization_receipt",
    )
    _require_identity(
        expected=args.expected_materialization_receipt_sha256,
        observed=materialization_receipt.receipt_sha256,
        purpose="materialization_receipt",
    )
    access_order.append(
        "content_silent_custodian_receipt_opened_and_external_identity_matched"
    )
    plan = prepare_condition_confirmation_plan(
        target=target,
        config=config,
        roster=roster,
        materialization_receipt=materialization_receipt,
        pipeline_sha256=args.pipeline_sha256,
        external_freeze_anchor=args.external_freeze_anchor,
    )
    access_order.append("deterministic_component_split_frozen_without_outcome_access")
    atomic_write_json(args.output, plan)
    access_order.append("plan_atomically_written_exclusive")
    return _receipt(
        {
            "receipt_version": "condition-confirmation-cli-receipt-v1",
            "stage": "confirmation_plan_prepared",
            "status": plan.status,
            "access_order": access_order,
            "effect_or_outcome_values_opened": False,
            "confirmation_outcomes_opened": False,
            "plan_sha256": plan.plan_sha256,
            "claim_spec_sha256": plan.claim_spec_sha256,
            "question_config_sha256": plan.question_config_sha256,
            "corpus_snapshot_sha256": plan.corpus_snapshot_sha256,
            "corpus_cutoff": plan.corpus_cutoff,
            "claim_contrast_id": plan.claim_contrast_id,
            "plan_file_sha256": sha256_file(args.output),
            "target_file_sha256": target_file_sha256,
            "config_file_sha256": config_file_sha256,
            "roster_file_sha256": roster_file_sha256,
            "materialization_receipt_sha256": (
                materialization_receipt.receipt_sha256
            ),
            "materialization_receipt_file_sha256": (
                materialization_receipt_file_sha256
            ),
            "development_component_count": len(plan.development_partition.component_ids),
            "confirmation_component_count": len(
                plan.confirmation_partition.component_ids
            ),
            "external_freeze_anchor": plan.external_freeze_anchor,
            "output": args.output.as_posix(),
        }
    )


def _read_and_preflight_plan(
    args: argparse.Namespace,
) -> tuple[ConditionConfirmationPlanV1, str, list[str]]:
    access_order: list[str] = []
    plan, plan_file_sha256 = _read_json_model(
        args.plan,
        ConditionConfirmationPlanV1,
        purpose="plan",
    )
    _require_identity(
        expected=args.expected_plan_sha256,
        observed=plan.plan_sha256,
        purpose="plan",
    )
    access_order.append("plan_opened_and_external_identity_matched")
    _require_identity(
        expected=args.current_pipeline_sha256,
        observed=plan.pipeline_sha256,
        purpose="pipeline",
    )
    access_order.append("current_pipeline_identity_matched_before_outcome_access")
    return plan, plan_file_sha256, access_order


def _fit(args: argparse.Namespace) -> dict[str, Any]:
    inputs = [args.plan, args.development_graph]
    _reject_output_alias(args.output, inputs)
    _preflight_new_output(args.output)
    plan, plan_file_sha256, access_order = _read_and_preflight_plan(args)
    development_graph, development_file_sha256 = _read_json_model(
        args.development_graph,
        EvidenceGraph,
        purpose="development_graph",
    )
    access_order.append("development_graph_opened_after_plan_and_pipeline_match")
    model = fit_condition_confirmation_model(
        plan,
        development_graph,
        current_pipeline_sha256=args.current_pipeline_sha256,
    )
    access_order.append("development_only_predictive_model_frozen")
    atomic_write_json(args.output, model)
    access_order.append("frozen_model_atomically_written_exclusive")
    return _receipt(
        {
            "receipt_version": "condition-confirmation-cli-receipt-v1",
            "stage": "development_model_fitted",
            "status": model.status,
            "access_order": access_order,
            "confirmation_outcomes_opened": False,
            "plan_sha256": plan.plan_sha256,
            "claim_spec_sha256": plan.claim_spec_sha256,
            "question_config_sha256": plan.question_config_sha256,
            "corpus_snapshot_sha256": plan.corpus_snapshot_sha256,
            "corpus_cutoff": plan.corpus_cutoff,
            "claim_contrast_id": plan.claim_contrast_id,
            "plan_file_sha256": plan_file_sha256,
            "development_graph_file_sha256": development_file_sha256,
            "model_sha256": model.model_sha256,
            "model_file_sha256": sha256_file(args.output),
            "selected_moderator": model.selected_moderator,
            "output": args.output.as_posix(),
        }
    )


def _read_and_preflight_model(
    args: argparse.Namespace,
    plan: ConditionConfirmationPlanV1,
) -> tuple[ConditionConfirmationFrozenModelV1, str]:
    model, model_file_sha256 = _read_json_model(
        args.model,
        ConditionConfirmationFrozenModelV1,
        purpose="frozen_model",
    )
    _require_identity(
        expected=args.expected_model_sha256,
        observed=model.model_sha256,
        purpose="model",
    )
    if model.plan_sha256 != plan.plan_sha256 or model.plan != plan:
        raise ConditionConfirmationError(
            "condition_confirmation_model_plan_identity_mismatch"
        )
    if model.status != "fitted":
        raise ConditionConfirmationError(
            "condition_confirmation_model_insufficient_confirmation_forbidden"
        )
    return model, model_file_sha256


def _frozen_confirmation_context(
    args: argparse.Namespace,
) -> tuple[
    ConditionConfirmationPlanV1,
    ConditionConfirmationFrozenModelV1,
    dict[str, str],
    list[str],
]:
    plan, plan_file_sha256, access_order = _read_and_preflight_plan(args)
    model, model_file_sha256 = _read_and_preflight_model(args, plan)
    access_order.append("frozen_model_opened_and_external_identity_matched")
    access_order.append("plan_model_pipeline_cross_binding_validated")
    return (
        plan,
        model,
        {
            "plan_file_sha256": plan_file_sha256,
            "model_file_sha256": model_file_sha256,
        },
        access_order,
    )


def _open_full_graph(
    args: argparse.Namespace,
    *,
    access_order: list[str],
) -> tuple[EvidenceGraph, str]:
    full_graph, full_graph_file_sha256 = _read_json_model(
        args.full_graph,
        EvidenceGraph,
        purpose="full_graph",
    )
    access_order.append("full_graph_opened_for_first_heldout_outcome_access")
    return full_graph, full_graph_file_sha256


def _confirm(args: argparse.Namespace) -> dict[str, Any]:
    inputs = [args.plan, args.model, args.full_graph]
    _reject_output_alias(args.output, inputs)
    if args.output.is_symlink():
        raise ConditionConfirmationError(
            f"condition_confirmation_output_symlink_forbidden:{args.output}"
        )
    plan, model, file_hashes, access_order = _frozen_confirmation_context(args)
    existing: ConditionConfirmationAssessmentV1 | None = None
    if args.output.exists():
        existing, _ = _read_json_model(
            args.output,
            ConditionConfirmationAssessmentV1,
            purpose="existing_assessment",
        )
        access_order.append("existing_assessment_opened_after_frozen_context_validation")
        if (
            existing.plan_sha256 != plan.plan_sha256
            or existing.model_sha256 != model.model_sha256
            or existing.pipeline_sha256 != plan.pipeline_sha256
        ):
            raise ConditionConfirmationError(
                "condition_confirmation_existing_assessment_context_mismatch"
            )
    full_graph, full_graph_file_sha256 = _open_full_graph(
        args,
        access_order=access_order,
    )
    file_hashes["full_graph_file_sha256"] = full_graph_file_sha256
    assessment = confirm_condition_dependence(
        plan=plan,
        model=model,
        full_graph=full_graph,
        current_pipeline_sha256=args.current_pipeline_sha256,
    )
    access_order.append("heldout_metrics_recomputed_from_exact_frozen_inputs")
    if existing is None:
        atomic_write_json(args.output, assessment)
        access_order.append("assessment_atomically_written_exclusive")
        write_disposition = "created"
    elif existing == assessment:
        access_order.append("existing_assessment_exactly_replayed_without_rewrite")
        write_disposition = "idempotent_existing_match"
    else:
        raise ConditionConfirmationError(
            "condition_confirmation_existing_assessment_recomputation_mismatch"
        )
    return _receipt(
        {
            "receipt_version": "condition-confirmation-cli-receipt-v1",
            "stage": "heldout_confirmation_assessed",
            "status": assessment.status,
            "access_order": access_order,
            "confirmation_outcomes_opened": True,
            "write_disposition": write_disposition,
            "plan_sha256": plan.plan_sha256,
            "model_sha256": model.model_sha256,
            "claim_spec_sha256": plan.claim_spec_sha256,
            "question_config_sha256": plan.question_config_sha256,
            "corpus_snapshot_sha256": plan.corpus_snapshot_sha256,
            "corpus_cutoff": plan.corpus_cutoff,
            "claim_contrast_id": plan.claim_contrast_id,
            "assessment_sha256": assessment.assessment_sha256,
            "assessment_file_sha256": sha256_file(args.output),
            "prediction_count": len(assessment.predictions),
            "confirmation_component_count": len(
                plan.confirmation_partition.component_ids
            ),
            **file_hashes,
            "output": args.output.as_posix(),
        }
    )


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    plan, model, file_hashes, access_order = _frozen_confirmation_context(args)
    full_graph, full_graph_file_sha256 = _open_full_graph(
        args,
        access_order=access_order,
    )
    file_hashes["full_graph_file_sha256"] = full_graph_file_sha256
    assessment, assessment_file_sha256 = _read_json_model(
        args.assessment,
        ConditionConfirmationAssessmentV1,
        purpose="assessment",
    )
    access_order.append("assessment_opened_after_frozen_context_validation")
    validated = validate_condition_confirmation_assessment(
        plan=plan,
        model=model,
        full_graph=full_graph,
        assessment=assessment,
        current_pipeline_sha256=args.current_pipeline_sha256,
    )
    access_order.append("split_model_predictions_and_metrics_exactly_recomputed")
    return _receipt(
        {
            "receipt_version": "condition-confirmation-cli-receipt-v1",
            "stage": "heldout_confirmation_validated",
            "status": validated.status,
            "access_order": access_order,
            "confirmation_outcomes_opened": True,
            "plan_sha256": plan.plan_sha256,
            "model_sha256": model.model_sha256,
            "claim_spec_sha256": plan.claim_spec_sha256,
            "question_config_sha256": plan.question_config_sha256,
            "corpus_snapshot_sha256": plan.corpus_snapshot_sha256,
            "corpus_cutoff": plan.corpus_cutoff,
            "claim_contrast_id": plan.claim_contrast_id,
            "assessment_sha256": validated.assessment_sha256,
            "assessment_file_sha256": assessment_file_sha256,
            **file_hashes,
        }
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize":
        receipt = _materialize(args)
    elif args.command == "validate-materialization":
        receipt = _validate_materialization(args)
    elif args.command == "prepare":
        receipt = _prepare(args)
    elif args.command == "fit":
        receipt = _fit(args)
    elif args.command == "confirm":
        receipt = _confirm(args)
    elif args.command == "validate":
        receipt = _validate(args)
    else:  # pragma: no cover - argparse closes this set
        raise AssertionError(f"condition_confirmation_unknown_command:{args.command}")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
