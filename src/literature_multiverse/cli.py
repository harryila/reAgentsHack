"""Supported command-line interface for Literature Multiverse."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from literature_multiverse.acquisition import (
    load_acquisition_manifest,
    replay_frozen_acquisition,
)
from literature_multiverse.adaptive_calibration import (
    AdaptiveCalibrationBundle,
    AdaptiveCalibrationBundleV2,
    AdaptivePolicyContext,
    ConditionCalibrationCollectionSourceRosterV1,
    PolicyVisibleQuestionTrajectoryV2,
)
from literature_multiverse.audit_session import CorrectionDisposition
from literature_multiverse.calibration import FrozenCalibrationBundle
from literature_multiverse.certificate import (
    ConditionCalibrationAssessmentReceiptV1,
    ConditionCalibrationCollectionSourceV1,
    ConditionVerificationCertificateV6,
    FinalConditionVerificationCertificateV7,
    freeze_condition_calibration_assessment_receipt_v1,
    match_validated_condition_calibration_collection_source_membership_v1,
    write_certificate_artifacts,
)
from literature_multiverse.claim_release import AuditResolutionReceipt
from literature_multiverse.condition_confirmation import (
    ConditionConfirmationAssessmentV1,
    ConditionConfirmationFrozenModelV1,
    ConditionConfirmationPlanV1,
)
from literature_multiverse.evidence_graph import EvidenceGraph
from literature_multiverse.item_risk_artifacts import ItemRiskScoringRunReceipt
from literature_multiverse.item_risk_calibration import (
    ItemRiskCalibrationBundle,
    ItemRiskCandidate,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file
from literature_multiverse.pipeline_fingerprint import (
    PipelineFingerprint,
    PipelineFingerprintVerification,
    require_pipeline_fingerprint_match,
)
from literature_multiverse.sequential_verification import (
    SequentialVerificationState,
    checkpoint_selected_audit_cost,
    freeze_selected_adjudication,
    freeze_state_expectation,
    resolve_selected_audit_candidate,
    select_next_audit_candidate,
)
from literature_multiverse.verifier import (
    build_offline_fixture,
    compute_candidate_runner_sha256,
    compute_synthesis_runner_sha256,
    compute_verification_policy_sha256,
    compute_verifier_pipeline_fingerprint,
    finalize_condition_verification,
    load_claim_manifest,
    load_corpus,
    prepare_verification_scientific_state,
    run_condition_calibration_collection,
    run_verification,
    sequential_candidates_from_prepared_state,
    validate_condition_calibration_collection_source_external_replay,
)


def _json_value(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}_json_unreadable:{path}") from exc


def _calibration_bundle(path: Path | None) -> FrozenCalibrationBundle | None:
    if path is None:
        return None
    payload = _json_value(path, label="calibration_bundle")
    if isinstance(payload, dict) and "frozen_calibration_bundle" in payload:
        payload = payload["frozen_calibration_bundle"]
    return FrozenCalibrationBundle.model_validate(payload)


def _adaptive_calibration_bundle(
    path: Path | None,
) -> AdaptiveCalibrationBundle | None:
    if path is None:
        return None
    payload = _json_value(path, label="adaptive_calibration_bundle")
    if isinstance(payload, dict) and "adaptive_calibration_bundle" in payload:
        payload = payload["adaptive_calibration_bundle"]
    return AdaptiveCalibrationBundle.model_validate(payload)


def _condition_adaptive_calibration_bundle(
    path: Path | None,
) -> AdaptiveCalibrationBundleV2 | None:
    if path is None:
        return None
    payload = _json_value(path, label="condition_adaptive_calibration_bundle")
    if isinstance(payload, dict) and "adaptive_calibration_bundle_v2" in payload:
        payload = payload["adaptive_calibration_bundle_v2"]
    return AdaptiveCalibrationBundleV2.model_validate(payload)


def _condition_plan(path: Path | None) -> ConditionConfirmationPlanV1 | None:
    if path is None:
        return None
    payload = _json_value(path, label="condition_plan")
    if isinstance(payload, dict) and "condition_plan" in payload:
        payload = payload["condition_plan"]
    return ConditionConfirmationPlanV1.model_validate(payload)


def _condition_development_graph(path: Path | None) -> EvidenceGraph | None:
    if path is None:
        return None
    payload = _json_value(path, label="condition_development_graph")
    if isinstance(payload, dict) and "development_evidence_graph" in payload:
        payload = payload["development_evidence_graph"]
    return EvidenceGraph.model_validate(payload)


def _condition_frozen_model(
    path: Path | None,
) -> ConditionConfirmationFrozenModelV1 | None:
    if path is None:
        return None
    payload = _json_value(path, label="condition_frozen_model")
    if isinstance(payload, dict) and "condition_frozen_model" in payload:
        payload = payload["condition_frozen_model"]
    return ConditionConfirmationFrozenModelV1.model_validate(payload)


def _condition_confirmation_assessment(
    path: Path,
) -> ConditionConfirmationAssessmentV1:
    """Open the terminal outcome only after the online policy proves invocation."""

    payload = _json_value(path, label="condition_confirmation_assessment")
    if isinstance(payload, dict) and "condition_confirmation_assessment" in payload:
        payload = payload["condition_confirmation_assessment"]
    return ConditionConfirmationAssessmentV1.model_validate(payload)


def _adaptive_policy_context(path: Path) -> AdaptivePolicyContext:
    payload = _json_value(path, label="adaptive_policy_context")
    if isinstance(payload, dict) and "adaptive_policy_context" in payload:
        payload = payload["adaptive_policy_context"]
    return AdaptivePolicyContext.model_validate(payload)


def _condition_visible_trajectory(
    path: Path | None,
) -> PolicyVisibleQuestionTrajectoryV2 | None:
    if path is None:
        return None
    payload = _json_value(path, label="condition_visible_trajectory")
    if isinstance(payload, dict) and "policy_visible_question_trajectory" in payload:
        payload = payload["policy_visible_question_trajectory"]
    return PolicyVisibleQuestionTrajectoryV2.model_validate(payload)


def _condition_collection_source(path: Path) -> ConditionCalibrationCollectionSourceV1:
    payload = _json_value(path, label="condition_collection_source")
    if isinstance(payload, dict) and "collection_source" in payload:
        payload = payload["collection_source"]
    source = ConditionCalibrationCollectionSourceV1.model_validate(payload)
    return validate_condition_calibration_collection_source_external_replay(source)


def _condition_collection_source_roster(
    path: Path,
) -> ConditionCalibrationCollectionSourceRosterV1:
    payload = _json_value(path, label="condition_collection_source_roster")
    if isinstance(payload, dict) and "collection_source_roster" in payload:
        payload = payload["collection_source_roster"]
    return ConditionCalibrationCollectionSourceRosterV1.model_validate(payload)


def _audit_receipts(path: Path | None) -> list[AuditResolutionReceipt]:
    if path is None:
        return []
    payload = _json_value(path, label="audit_receipts")
    if isinstance(payload, dict) and "receipts" in payload:
        payload = payload["receipts"]
    if not isinstance(payload, list):
        raise ValueError("audit_receipts_json_must_be_list_or_receipts_object")
    return [AuditResolutionReceipt.model_validate(item) for item in payload]


def _pipeline_fingerprint(path: Path | None) -> PipelineFingerprint | None:
    if path is None:
        return None
    payload = _json_value(path, label="pipeline_fingerprint")
    if isinstance(payload, dict) and "pipeline_fingerprint" in payload:
        payload = payload["pipeline_fingerprint"]
    return PipelineFingerprint.model_validate(payload)


def _item_risk_bundle(path: Path | None) -> ItemRiskCalibrationBundle | None:
    if path is None:
        return None
    payload = _json_value(path, label="item_risk_calibration")
    if isinstance(payload, dict) and "item_risk_calibration_bundle" in payload:
        payload = payload["item_risk_calibration_bundle"]
    return ItemRiskCalibrationBundle.model_validate(payload)


def _item_risk_candidates(path: Path | None) -> list[ItemRiskCandidate] | None:
    if path is None:
        return None
    payload = _json_value(path, label="item_risk_candidates")
    if isinstance(payload, dict) and "candidates" in payload:
        payload = payload["candidates"]
    if not isinstance(payload, list):
        raise ValueError("item_risk_candidates_json_must_be_list_or_candidates_object")
    return [ItemRiskCandidate.model_validate(item) for item in payload]


def _item_risk_scoring_receipt(
    path: Path | None,
) -> ItemRiskScoringRunReceipt | None:
    if path is None:
        return None
    payload = _json_value(path, label="item_risk_scoring_receipt")
    if isinstance(payload, dict) and "item_risk_scoring_receipt" in payload:
        payload = payload["item_risk_scoring_receipt"]
    return ItemRiskScoringRunReceipt.model_validate(payload)


def _sequential_audit_state(path: Path | None) -> SequentialVerificationState | None:
    if path is None:
        return None
    payload = _json_value(path, label="audit_state")
    if isinstance(payload, dict) and "sequential_audit_state" in payload:
        payload = payload["sequential_audit_state"]
    return SequentialVerificationState.model_validate(payload)


def _add_verify_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "verify",
        help="Verify one AI-generated claim against a frozen literature corpus.",
        description=(
            "Build the evidence graph, synthesize effects, prioritize graph-derived "
            "verification actions, assess release gates, and write JSON/HTML certificates."
        ),
    )
    parser.add_argument("--claim", type=Path, help="Claim manifest YAML or JSON")
    parser.add_argument(
        "--corpus",
        type=Path,
        help=(
            "Evidence graph JSON, verifier corpus bundle, legacy findings JSONL/parquet, "
            "or a directory containing evidence_graph.json/findings.parquet"
        ),
    )
    parser.add_argument(
        "--acquisition-manifest",
        type=Path,
        help=(
            "Self-hashed frozen acquisition manifest. Replays its exact local search, "
            "screening, and native extraction/package inputs before invoking the same "
            "verifier used by --corpus. Mutually exclusive with --corpus."
        ),
    )
    parser.add_argument(
        "--budget-minutes",
        type=float,
        required=True,
        help=("Maximum prospective person-minutes summed across reviewers and final adjudication"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help=(
            "Legacy fixed single-decision calibration JSON; analysis/receipt mode "
            "only and never valid for sequential adaptive release"
        ),
    )
    parser.add_argument(
        "--adaptive-calibration",
        type=Path,
        help=(
            "Frozen complete-question first-release trajectory calibration bundle; "
            "the verifier derives the prospective prefix from its audit ledger"
        ),
    )
    parser.add_argument(
        "--condition-adaptive-calibration",
        type=Path,
        help=(
            "Frozen confirmation-aware complete-question calibration v2 bundle; "
            "mandatory for manifest-v3 global condition verification"
        ),
    )
    parser.add_argument(
        "--condition-plan",
        type=Path,
        help=(
            "Outcome-firewalled condition-confirmation plan with its exact custodian "
            "materialization receipt"
        ),
    )
    parser.add_argument(
        "--condition-development-graph",
        type=Path,
        help=(
            "Frozen development partition used by online synthesis and audit policy; "
            "confirmation outcomes remain unopened"
        ),
    )
    parser.add_argument(
        "--condition-model",
        type=Path,
        help="Development-only frozen condition-confirmation model",
    )
    parser.add_argument(
        "--condition-assessment",
        type=Path,
        help=(
            "Terminal held-out assessment. The file is deliberately not opened until "
            "the outcome-free scheduler proves the exact condition-gate invocation state."
        ),
    )
    parser.add_argument(
        "--receipts",
        type=Path,
        help=(
            "Legacy static completed audit receipts; analysis-only and never a "
            "release-capable sequential trajectory"
        ),
    )
    parser.add_argument(
        "--analysis-only-uncalibrated-audit",
        action="store_true",
        help=(
            "Explicitly allow sequential action selection without adaptive "
            "calibration. The resulting state is permanently analysis-only and "
            "cannot later be upgraded to a release trajectory."
        ),
    )
    parser.add_argument(
        "--pipeline-fingerprint",
        type=Path,
        help=(
            "Optional expected computed-pipeline artifact; every file is rehashed "
            "before verification"
        ),
    )
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        help="Repository root used to recompute the pipeline (default: installed project root)",
    )
    parser.add_argument(
        "--item-risk-scoring-receipt",
        type=Path,
        help=(
            "Self-contained v2 item-risk scoring receipt with calibration bundle, "
            "prospective candidates, recomputable cell-rate bounds, and pipeline proof"
        ),
    )
    parser.add_argument(
        "--item-risk-calibration",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--item-risk-candidates",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--audit-state",
        type=Path,
        help=(
            "Resumable sequential audit state; its original graph must match --corpus "
            "and its corrected current graph, synthesis, candidates, pipeline, budget, "
            "and realized-cost transition chain are independently replayed"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory (default: artifacts/verification/<run-id>)",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Run the embedded provider-free integration fixture instead of input files",
    )
    parser.add_argument("--force", action="store_true", help="Replace certificate files")
    return parser


def _add_fingerprint_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "fingerprint",
        help="Compute and freeze the exact supported verifier code/prompt identity.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def _add_condition_collect_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "condition-collect",
        help=("Run one always-abstained pre-bundle condition-calibration policy arm."),
        description=(
            "Freeze an outcome-free collection source before confirmation outcomes, "
            "reference labels, or confirmation-aware calibration are opened."
        ),
    )
    parser.add_argument("--claim", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--budget-minutes", type=float, required=True)
    parser.add_argument(
        "--split",
        choices=["development", "calibration"],
        required=True,
    )
    parser.add_argument("--policy-context", type=Path, required=True)
    parser.add_argument("--condition-plan", type=Path, required=True)
    parser.add_argument("--condition-development-graph", type=Path, required=True)
    parser.add_argument("--condition-model", type=Path, required=True)
    parser.add_argument(
        "--policy-visible-trajectory",
        type=Path,
        help=(
            "Optional exact full multi-arm v2 trajectory. Omit only for a truly "
            "single-arm frozen calibration design."
        ),
    )
    parser.add_argument("--audit-state", type=Path)
    parser.add_argument("--pipeline-fingerprint", type=Path)
    parser.add_argument("--pipeline-root", type=Path)
    parser.add_argument("--item-risk-scoring-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def _add_condition_finalize_calibration_parser(
    subparsers: Any,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "condition-finalize-calibration",
        help=("Join a gate-ready collection source to one held-out assessment receipt."),
        description=(
            "The pre-outcome source roster and its externally recorded hashes are "
            "validated first. The selected source must be an exact roster member and "
            "gate-ready before the held-out assessment path is opened."
        ),
    )
    parser.add_argument("--source-roster", type=Path, required=True)
    parser.add_argument("--expected-source-roster-sha256", required=True)
    parser.add_argument("--expected-source-membership-sha256", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--condition-assessment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def _add_audit_select_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "audit-select",
        help="Select the next action from a resumable audit state without spending cost.",
    )
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-at", help="Timezone-aware ISO-8601 timestamp")
    parser.add_argument("--pipeline-fingerprint", type=Path)
    parser.add_argument("--pipeline-root", type=Path)
    parser.add_argument(
        "--analysis-only-uncalibrated-audit",
        action="store_true",
        help=(
            "Required for standalone selection from an uncalibrated state; the "
            "result is permanently analysis-only. Adaptive production selection "
            "must run through lm verify with its frozen calibration bundle."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def _add_audit_checkpoint_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "audit-checkpoint",
        help=(
            "Charge cumulative measured time to the active action without resolving or applying it."
        ),
        description=(
            "Checkpoint partial work on the selected audit action. The value is the "
            "cumulative measured time for that action, not a delta. The action remains "
            "active and blocks release; no adjudication or correction is accepted."
        ),
    )
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument(
        "--active-realized-minutes",
        type=float,
        required=True,
        help=(
            "Cumulative measured person-minutes spent across all reviewers on the "
            "current active action"
        ),
    )
    parser.add_argument("--pipeline-fingerprint", type=Path)
    parser.add_argument("--pipeline-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def _add_audit_resolve_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "audit-resolve",
        help=(
            "Bind a real adjudication to the selected action, apply its correction, "
            "rerun synthesis/counterfactuals, and charge measured minutes."
        ),
    )
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--claim", type=Path, required=True)
    parser.add_argument(
        "--disposition",
        choices=[item.value for item in CorrectionDisposition],
        required=True,
    )
    parser.add_argument(
        "--corrected-corpus",
        type=Path,
        help=(
            "Corrected graph containing only the selected-estimate adjudication; "
            "required for corrected and forbidden for no_change"
        ),
    )
    parser.add_argument("--adjudication-protocol", type=Path, required=True)
    parser.add_argument("--adjudication-payload", type=Path, required=True)
    parser.add_argument("--correction-protocol", type=Path, required=True)
    parser.add_argument("--correction-payload", type=Path, required=True)
    parser.add_argument(
        "--provenance",
        choices=["blinded_human", "benchmark_adjudication"],
        required=True,
    )
    parser.add_argument("--adjudicator-count", type=int, required=True)
    parser.add_argument(
        "--realized-minutes",
        type=float,
        required=True,
        help=("Total measured person-minutes across every reviewer and final adjudication"),
    )
    parser.add_argument("--completed-at", help="Timezone-aware ISO-8601 timestamp")
    parser.add_argument("--pipeline-fingerprint", type=Path)
    parser.add_argument("--pipeline-root", type=Path)
    parser.add_argument("--item-risk-scoring-receipt", type=Path)
    parser.add_argument("--item-risk-calibration", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--item-risk-candidates", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lm", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_verify_parser(subparsers)
    _add_condition_collect_parser(subparsers)
    _add_condition_finalize_calibration_parser(subparsers)
    _add_fingerprint_parser(subparsers)
    _add_audit_select_parser(subparsers)
    _add_audit_checkpoint_parser(subparsers)
    _add_audit_resolve_parser(subparsers)
    return parser


def _condition_collect(args: argparse.Namespace) -> int:
    repository_root = args.pipeline_root or Path(__file__).resolve().parents[2]
    manifest = load_claim_manifest(args.claim)
    corpus = load_corpus(
        args.corpus,
        legacy_settings=manifest.legacy_adapter,
        repository_root=repository_root,
    )
    plan = _condition_plan(args.condition_plan)
    development = _condition_development_graph(args.condition_development_graph)
    model = _condition_frozen_model(args.condition_model)
    assert plan is not None and development is not None and model is not None
    source = run_condition_calibration_collection(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=args.budget_minutes,
        collection_split=args.split,
        adaptive_policy_context=_adaptive_policy_context(args.policy_context),
        condition_plan=plan,
        condition_development_graph=development,
        condition_frozen_model=model,
        policy_visible_question_trajectory=_condition_visible_trajectory(
            args.policy_visible_trajectory
        ),
        expected_pipeline_fingerprint=_pipeline_fingerprint(args.pipeline_fingerprint),
        pipeline_root=args.pipeline_root,
        item_risk_scoring_receipt=_item_risk_scoring_receipt(args.item_risk_scoring_receipt),
        sequential_audit_state=_sequential_audit_state(args.audit_state),
        generated_at=datetime.now(UTC),
    )
    outputs: dict[Path, Any] = {
        args.output_dir / "condition-calibration-collection-source.json": source,
        args.output_dir / "sequential-audit-state.json": (source.sequential_audit_state),
    }
    if source.policy_visible_question_trajectory is not None:
        outputs[args.output_dir / "policy-visible-question-trajectory-v2.json"] = (
            source.policy_visible_question_trajectory
        )
    _write_json_outputs(outputs, force=args.force)
    print(
        json.dumps(
            {
                "collection_decision_sha256": (source.collection_decision.decision_sha256),
                "collection_source_sha256": source.collection_source_sha256,
                "gate_ready": (source.collection_decision.outcome == "condition_gate_ready"),
                "outcome": source.collection_decision.outcome,
                "policy_arm_id": source.policy_arm_id,
                "question_id": source.question_id,
                "selected_audit_item_id": (
                    None
                    if source.sequential_audit_state.session.active_action is None
                    else source.sequential_audit_state.session.active_action.item_id
                ),
                "source_path": (
                    args.output_dir / "condition-calibration-collection-source.json"
                ).as_posix(),
                "state_sha256": source.sequential_audit_state.state_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def _condition_finalize_calibration(args: argparse.Namespace) -> int:
    if args.output.exists() and not args.force:
        raise FileExistsError(f"condition_calibration_receipt_output_exists:{args.output}")
    # Validate the already-frozen, outcome-free roster and its external anchor
    # before even opening the source selected from it, much less the held-out
    # assessment.  The final atomic write retains its own race-safe preflight.
    roster = _condition_collection_source_roster(args.source_roster)
    if (
        roster.source_roster_sha256 != args.expected_source_roster_sha256
        or roster.source_membership_sha256 != args.expected_source_membership_sha256
    ):
        raise ValueError("condition_collection_source_roster_external_anchor_mismatch")
    source = _condition_collection_source(args.source)
    match_validated_condition_calibration_collection_source_membership_v1(
        collection_source_roster=roster,
        collection_source=source,
        expected_source_roster_sha256=args.expected_source_roster_sha256,
        expected_source_membership_sha256=args.expected_source_membership_sha256,
    )
    # Do not touch the outcome-bearing path until the immutable collection source
    # itself has replayed and proven the exact first gate-invocation state.
    if (
        source.collection_decision.outcome != "condition_gate_ready"
        or source.collection_decision.condition_gate_invocation_proof is None
        or source.policy_visible_question_trajectory is None
    ):
        raise ValueError("condition_collection_source_not_gate_ready")
    assessment = _condition_confirmation_assessment(args.condition_assessment)
    receipt: ConditionCalibrationAssessmentReceiptV1 = (
        freeze_condition_calibration_assessment_receipt_v1(
            collection_source_roster=roster,
            collection_source=source,
            condition_confirmation_assessment=assessment,
        )
    )
    atomic_write_json(args.output, receipt, force=args.force)
    print(
        json.dumps(
            {
                "calibration_gate_result_sha256": (receipt.calibration_gate_result.result_sha256),
                "collection_source_sha256": receipt.collection_source_sha256,
                "output": args.output.as_posix(),
                "question_id": receipt.question_id,
                "receipt_sha256": receipt.receipt_sha256,
                "status": receipt.calibration_gate_result.status,
            },
            sort_keys=True,
        )
    )
    return 0


def _verify(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    repository_root = args.pipeline_root or Path(__file__).resolve().parents[2]
    expected_fingerprint = _pipeline_fingerprint(args.pipeline_fingerprint)
    acquisition_replay = None
    if args.fixture:
        if (
            args.claim is not None
            or args.corpus is not None
            or args.acquisition_manifest is not None
        ):
            parser.error(
                "--fixture cannot be combined with --claim, --corpus, or --acquisition-manifest"
            )
        manifest, corpus = build_offline_fixture()
    else:
        if args.claim is None:
            parser.error("verify requires --claim unless --fixture is used")
        if (args.corpus is None) == (args.acquisition_manifest is None):
            parser.error(
                "verify requires exactly one of --corpus or --acquisition-manifest "
                "unless --fixture is used"
            )
        manifest = load_claim_manifest(args.claim)
        if args.acquisition_manifest is None:
            corpus = load_corpus(
                args.corpus,
                legacy_settings=manifest.legacy_adapter,
                repository_root=repository_root,
            )
        else:
            if args.output_dir is None:
                parser.error("--acquisition-manifest requires explicit --output-dir")
            if not args.force:
                existing = sorted(
                    path.as_posix()
                    for path in (
                        args.output_dir / "acquisition-replay-receipt.json",
                        args.output_dir / "verification-certificate.json",
                        args.output_dir / "verification-certificate.html",
                    )
                    if path.exists()
                )
                if existing:
                    raise FileExistsError(f"verification_acquisition_outputs_exist:{existing}")
            frozen_pipeline = expected_fingerprint or compute_verifier_pipeline_fingerprint(
                root=repository_root
            )
            pipeline_verification = require_pipeline_fingerprint_match(
                expected=frozen_pipeline,
                root=repository_root,
            )
            assert pipeline_verification.computed_pipeline_sha256 is not None
            acquisition_replay = replay_frozen_acquisition(
                manifest=load_acquisition_manifest(args.acquisition_manifest),
                claim_manifest=manifest,
                repository_root=repository_root,
                pipeline_sha256=pipeline_verification.computed_pipeline_sha256,
                output_dir=args.output_dir,
                force=args.force,
            )
            corpus = acquisition_replay.corpus

    condition_bundle = _condition_adaptive_calibration_bundle(args.condition_adaptive_calibration)
    condition_plan = _condition_plan(args.condition_plan)
    condition_development_graph = _condition_development_graph(args.condition_development_graph)
    condition_model = _condition_frozen_model(args.condition_model)
    if args.condition_assessment is not None and manifest.claim_manifest_version != "3":
        raise ValueError("condition_assessment_requires_manifest_v3")
    fixed_calibration = _calibration_bundle(args.calibration)
    adaptive_calibration = _adaptive_calibration_bundle(args.adaptive_calibration)
    audit_receipts = _audit_receipts(args.receipts)
    item_risk_receipt = _item_risk_scoring_receipt(args.item_risk_scoring_receipt)
    item_risk_bundle = _item_risk_bundle(args.item_risk_calibration)
    item_risk_candidates = _item_risk_candidates(args.item_risk_candidates)
    sequential_state_input = _sequential_audit_state(args.audit_state)
    generated_at = datetime.now(UTC)
    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=args.budget_minutes,
        frozen_calibration_bundle=fixed_calibration,
        adaptive_calibration_bundle=adaptive_calibration,
        adaptive_calibration_bundle_v2=condition_bundle,
        condition_plan=condition_plan,
        condition_development_graph=condition_development_graph,
        condition_frozen_model=condition_model,
        audit_resolution_receipts=audit_receipts,
        expected_pipeline_fingerprint=expected_fingerprint,
        pipeline_root=args.pipeline_root,
        item_risk_scoring_receipt=item_risk_receipt,
        item_risk_calibration_bundle=item_risk_bundle,
        item_risk_candidates=item_risk_candidates,
        sequential_audit_state=sequential_state_input,
        allow_uncalibrated_sequential_analysis=(args.analysis_only_uncalibrated_audit),
        generated_at=generated_at,
    )
    # The terminal assessment is an outcome-bearing artifact.  Its path is carried
    # through argument parsing but its bytes are not touched until an immutable,
    # outcome-free v6 run proves the exact invocation state.  This ordering is part
    # of the held-out firewall, not merely an I/O optimization.
    if args.condition_assessment is not None:
        gate_ready = (
            isinstance(certificate, ConditionVerificationCertificateV6)
            and certificate.production_stop_decision.outcome == "condition_gate_ready"
            and certificate.condition_gate_invocation_proof is not None
        )
        if gate_ready:
            if args.output_dir is not None and not args.force:
                prospective_paths = [
                    args.output_dir / "verification-certificate.json",
                    args.output_dir / "verification-certificate.html",
                ]
                if certificate.sequential_audit_state is not None:
                    prospective_paths.append(args.output_dir / "sequential-audit-state.json")
                existing = [path.as_posix() for path in prospective_paths if path.exists()]
                if existing:
                    raise FileExistsError(f"verification_certificate_outputs_exist:{existing}")
            assessment = _condition_confirmation_assessment(args.condition_assessment)
            certificate = finalize_condition_verification(
                source_certificate=certificate,
                condition_confirmation_assessment=assessment,
                generated_at=generated_at,
            )
            if not isinstance(certificate, FinalConditionVerificationCertificateV7):
                raise ValueError("condition_terminal_join_did_not_produce_v7")

    source_certificate = (
        certificate.source_certificate_v6
        if isinstance(certificate, FinalConditionVerificationCertificateV7)
        else certificate
    )
    sequential_state = source_certificate.sequential_audit_state
    output_dir = args.output_dir or (Path("artifacts") / "verification" / certificate.run_id)
    audit_state_path = output_dir / "sequential-audit-state.json"
    if sequential_state is not None and audit_state_path.exists() and not args.force:
        raise FileExistsError(
            f"verification_certificate_outputs_exist:{[audit_state_path.as_posix()]}"
        )
    artifacts = write_certificate_artifacts(certificate, output_dir, force=args.force)
    acquisition_receipt_path = output_dir / "acquisition-replay-receipt.json"
    if acquisition_replay is not None:
        atomic_write_json(
            acquisition_receipt_path,
            acquisition_replay.receipt,
            force=args.force,
        )
    if sequential_state is not None:
        atomic_write_json(
            audit_state_path,
            sequential_state,
            force=args.force,
        )
    adaptive_assessment = getattr(
        certificate,
        "adaptive_prospective_assessment_v2",
        getattr(source_certificate, "adaptive_prospective_assessment", None),
    )
    print(
        json.dumps(
            {
                "adaptive_assessment_sha256": (
                    None if adaptive_assessment is None else adaptive_assessment.assessment_sha256
                ),
                "acquisition_replay_receipt_path": (
                    acquisition_receipt_path.as_posix() if acquisition_replay is not None else None
                ),
                "acquisition_replay_receipt_sha256": (
                    acquisition_replay.receipt.receipt_sha256
                    if acquisition_replay is not None
                    else None
                ),
                "certificate_sha256": certificate.certificate_sha256,
                "certificate_version": certificate.certificate_version,
                "complete_corpus_membership_sha256": (
                    source_certificate.complete_corpus_identity.membership_sha256
                ),
                "decision_sha256": certificate.release_assessment.decision_sha256,
                "html_path": artifacts.html_path,
                "html_sha256": artifacts.html_sha256,
                "json_path": artifacts.json_path,
                "json_sha256": artifacts.json_sha256,
                "item_risk_scoring_receipt_sha256": (
                    None
                    if source_certificate.item_risk_scoring_receipt is None
                    else source_certificate.item_risk_scoring_receipt.receipt_sha256
                ),
                "question_id": manifest.question_id,
                "reasons": certificate.reasons,
                "run_id": certificate.run_id,
                "selected_audit_item_id": (
                    sequential_state.session.active_action.item_id
                    if sequential_state is not None
                    and sequential_state.session.active_action is not None
                    else None
                ),
                "sequential_audit_state_path": (
                    audit_state_path.as_posix() if sequential_state is not None else None
                ),
                "sequential_audit_state_sha256": (
                    sequential_state.state_sha256 if sequential_state is not None else None
                ),
                "status": certificate.status,
            },
            sort_keys=True,
        )
    )
    return 0


def _fingerprint(args: argparse.Namespace) -> int:
    fingerprint = compute_verifier_pipeline_fingerprint(root=args.pipeline_root)
    atomic_write_json(args.output, fingerprint, force=args.force)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "pipeline_sha256": fingerprint.pipeline_sha256,
                "status": "frozen",
            },
            sort_keys=True,
        )
    )
    return 0


def _aware_timestamp(raw: str | None) -> datetime:
    if raw is None:
        return datetime.now(UTC)
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"timestamp_invalid:{raw}") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp_requires_timezone")
    return value


def _write_json_outputs(
    outputs: dict[Path, object],
    *,
    force: bool,
) -> None:
    if not force:
        existing = sorted(path.as_posix() for path in outputs if path.exists())
        if existing:
            raise FileExistsError(f"audit_outputs_exist:{existing}")
    for path, payload in outputs.items():
        atomic_write_json(path, payload, force=force)


def _require_current_pipeline_for_audit_state(
    args: argparse.Namespace,
    state: SequentialVerificationState,
) -> PipelineFingerprintVerification:
    """Rehash the supported pipeline and reject a state from different code."""

    repository_root = args.pipeline_root or Path(__file__).resolve().parents[2]
    expected = _pipeline_fingerprint(args.pipeline_fingerprint)
    if expected is None:
        expected = compute_verifier_pipeline_fingerprint(root=repository_root)
    verification = require_pipeline_fingerprint_match(
        expected=expected,
        root=repository_root,
    )
    if verification.computed_pipeline_sha256 != state.session.pipeline_sha256:
        raise ValueError("audit_state_pipeline_does_not_match_current_pipeline")
    return verification


def _audit_select(args: argparse.Namespace) -> int:
    state = _sequential_audit_state(args.state)
    assert state is not None
    _require_current_pipeline_for_audit_state(args, state)
    if state.adaptive_policy_context_sha256 is not None:
        raise ValueError("adaptive_audit_selection_requires_verify_with_calibration_bundle")
    if not args.analysis_only_uncalibrated_audit:
        raise ValueError("uncalibrated_audit_selection_requires_analysis_only_opt_in")
    result = select_next_audit_candidate(
        state,
        expected=freeze_state_expectation(state),
        selected_at=_aware_timestamp(args.selected_at),
    )
    outputs = {
        args.output_dir / "sequential-audit-state.json": result.state,
        args.output_dir / "audit-selection-result.json": result,
        args.output_dir / "audit-action.json": result.action,
    }
    _write_json_outputs(outputs, force=args.force)
    print(
        json.dumps(
            {
                "action_packet_sha256": result.action.packet_sha256,
                "item_id": result.action.item_id,
                "remaining_budget": result.state.session.remaining_budget,
                "state_path": (args.output_dir / "sequential-audit-state.json").as_posix(),
                "state_sha256": result.state.state_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def _audit_checkpoint(args: argparse.Namespace) -> int:
    state = _sequential_audit_state(args.state)
    assert state is not None
    _require_current_pipeline_for_audit_state(args, state)
    result = checkpoint_selected_audit_cost(
        state,
        expected=freeze_state_expectation(state),
        active_realized_cost=args.active_realized_minutes,
    )
    action = result.state.session.active_action
    assert action is not None
    outputs = {
        args.output_dir / "sequential-audit-state.json": result.state,
        args.output_dir / "audit-active-cost-checkpoint.json": result,
        args.output_dir / "audit-action.json": action,
    }
    _write_json_outputs(outputs, force=args.force)
    print(
        json.dumps(
            {
                "action_packet_sha256": action.packet_sha256,
                "active_realized_minutes": result.state.session.active_realized_cost,
                "cumulative_realized_minutes": result.state.session.current_realized_cost,
                "historical_realized_minutes": (result.state.session.historical_realized_cost),
                "item_id": action.item_id,
                "release_blocked_by_active_action": True,
                "remaining_budget": result.state.session.remaining_budget,
                "state_path": (args.output_dir / "sequential-audit-state.json").as_posix(),
                "state_sha256": result.state.state_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def _audit_resolve(args: argparse.Namespace) -> int:
    state = _sequential_audit_state(args.state)
    assert state is not None
    active_action = state.session.active_action
    if active_action is None:
        raise ValueError("audit_resolution_requires_active_action")
    manifest = load_claim_manifest(args.claim)
    disposition = CorrectionDisposition(args.disposition)
    if disposition is CorrectionDisposition.CORRECTED and args.corrected_corpus is None:
        raise ValueError("corrected_disposition_requires_corrected_corpus")
    if disposition is CorrectionDisposition.NO_CHANGE and args.corrected_corpus is not None:
        raise ValueError("no_change_disposition_forbids_corrected_corpus")
    if args.provenance == "blinded_human" and args.adjudicator_count < 2:
        raise ValueError("blinded_human_resolution_requires_two_adjudicators")

    repository_root = args.pipeline_root or Path(__file__).resolve().parents[2]
    pipeline_verification = _require_current_pipeline_for_audit_state(args, state)
    if state.session.policy_sha256 != compute_verification_policy_sha256(manifest):
        raise ValueError("audit_state_claim_manifest_context_mismatch")

    if disposition is CorrectionDisposition.NO_CHANGE:
        post_graph = state.graph
        corrected_graph = None
    else:
        corrected = load_corpus(
            args.corrected_corpus,
            legacy_settings=manifest.legacy_adapter,
            repository_root=repository_root,
        )
        post_graph = corrected.graph
        corrected_graph = corrected.graph
    item_receipt = _item_risk_scoring_receipt(args.item_risk_scoring_receipt)
    if args.item_risk_calibration is not None or args.item_risk_candidates is not None:
        raise ValueError("detached_item_risk_inputs_forbidden_use_scoring_receipt_v2")
    if item_receipt is not None and item_receipt.pipeline_verification != pipeline_verification:
        raise ValueError("item_risk_scoring_receipt_pipeline_mismatch")
    item_bundle = None if item_receipt is None else item_receipt.calibration_bundle
    item_candidates = None if item_receipt is None else list(item_receipt.candidates)
    if any(candidate.risk_bound_sha256 is not None for candidate in state.candidates) and (
        item_receipt is None
    ):
        raise ValueError("artifact_backed_audit_state_requires_refreshed_scoring_receipt")
    prepared = prepare_verification_scientific_state(
        manifest=manifest,
        graph=post_graph,
        pipeline_verification=pipeline_verification,
        item_risk_calibration_bundle=item_bundle,
        item_risk_candidates=item_candidates,
        resolved_item_ids_for_risk_projection={
            *state.session.resolved_item_ids,
            active_action.item_id,
        },
    )
    refreshed_candidates = sequential_candidates_from_prepared_state(
        manifest=manifest,
        prepared=prepared,
    )

    expected_state = freeze_state_expectation(state)
    adjudication = freeze_selected_adjudication(
        state,
        expected=expected_state,
        provenance=args.provenance,
        adjudicator_count=args.adjudicator_count,
        protocol_sha256=sha256_file(args.adjudication_protocol),
        payload_sha256=sha256_file(args.adjudication_payload),
        completed_at=_aware_timestamp(args.completed_at),
        realized_cost=args.realized_minutes,
    )
    post_graph_sha256 = hash_canonical(post_graph)

    def rerun_synthesis(graph: Any) -> dict[str, Any]:
        if hash_canonical(graph) != post_graph_sha256:
            raise ValueError("audit_synthesis_callback_graph_mismatch")
        return prepared.synthesis

    def rerun_candidates(graph: Any, synthesis: Any, session: Any) -> tuple[Any, ...]:
        if (
            hash_canonical(graph) != post_graph_sha256
            or hash_canonical(synthesis) != hash_canonical(prepared.synthesis)
            or session.session_id != state.session.session_id
        ):
            raise ValueError("audit_candidate_callback_state_mismatch")
        return refreshed_candidates

    synthesis_runner_sha256 = compute_synthesis_runner_sha256(
        manifest=manifest,
        pipeline_sha256=state.session.pipeline_sha256,
    )
    candidate_runner_sha256 = compute_candidate_runner_sha256(
        manifest=manifest,
        pipeline_sha256=state.session.pipeline_sha256,
    )
    result = resolve_selected_audit_candidate(
        state,
        expected=expected_state,
        adjudication=adjudication,
        disposition=disposition,
        corrected_graph=corrected_graph,
        correction_provenance=args.provenance,
        correction_protocol_sha256=sha256_file(args.correction_protocol),
        external_correction_payload_sha256=sha256_file(args.correction_payload),
        synthesis_runner_sha256=synthesis_runner_sha256,
        candidate_runner_sha256=candidate_runner_sha256,
        rerun_synthesis=rerun_synthesis,
        rerun_candidates=rerun_candidates,
    )
    outputs = {
        args.output_dir / "sequential-audit-state.json": result.state,
        args.output_dir / "audit-resolution-result.json": result,
        args.output_dir / "audit-resolution-receipt.json": result.receipt,
        args.output_dir / "audit-correction-provenance.json": (result.correction_provenance),
        args.output_dir / "post-evidence-graph.json": result.state.graph,
        args.output_dir / "post-synthesis.json": result.state.synthesis,
        args.output_dir / "post-item-risk-bounds.json": list(prepared.item_risk_bounds),
    }
    _write_json_outputs(outputs, force=args.force)
    print(
        json.dumps(
            {
                "cumulative_realized_minutes": (result.state.session.current_realized_cost),
                "item_id": result.receipt.item_id,
                "receipt_sha256": result.receipt.receipt_sha256,
                "remaining_budget": result.state.session.remaining_budget,
                "state_path": (args.output_dir / "sequential-audit-state.json").as_posix(),
                "state_sha256": result.state.state_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "verify":
        return _verify(args, parser)
    if args.command == "condition-collect":
        return _condition_collect(args)
    if args.command == "condition-finalize-calibration":
        return _condition_finalize_calibration(args)
    if args.command == "fingerprint":
        return _fingerprint(args)
    if args.command == "audit-select":
        return _audit_select(args)
    if args.command == "audit-checkpoint":
        return _audit_checkpoint(args)
    if args.command == "audit-resolve":
        return _audit_resolve(args)
    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover - argparse exits


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
