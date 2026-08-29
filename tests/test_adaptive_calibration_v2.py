from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from pydantic import ValidationError

import literature_multiverse.adaptive_calibration as adaptive_calibration_module
from literature_multiverse.adaptive_calibration import (
    AdaptiveCalibrationBundleV2,
    AdaptiveCalibrationError,
    AdaptiveCalibrationPlanV2,
    AdaptiveCalibrationRosterV2,
    AdaptiveDevelopmentFreezeV2,
    AdaptivePolicyContext,
    AdaptivePreselectionState,
    AdaptiveTerminalAuditCandidate,
    ConditionCalibrationCollectionSourceAnchorV1,
    ConditionCalibrationGateResultV1,
    ConditionCalibrationProjectionV1,
    ConditionGateInvocationProofV2,
    GateCompleteCalibrationRosterV2,
    PolicyVisibleQuestionTrajectoryV2,
    ProspectiveAdaptiveReleaseCandidateV2,
    QuestionReferenceVerdictV2,
    assess_confirmation_aware_adaptive_release_candidate,
    calibrate_confirmation_aware_first_release,
    fit_adaptive_development_v2,
    freeze_adaptive_independence_identity_v2,
    freeze_adaptive_policy_arm_trajectory,
    freeze_adaptive_policy_context,
    freeze_adaptive_preselection_state,
    freeze_adaptive_target_semantics_v2,
    freeze_complete_corpus_identity,
    freeze_condition_calibration_gate_result_v1,
    freeze_condition_calibration_projection,
    freeze_condition_confirmation_gate_assessment,
    freeze_condition_terminal_gate_result_v2,
    freeze_confirmation_aware_arm_trajectory,
    freeze_confirmation_aware_release_qualification_proof_v2,
    freeze_gate_complete_calibration_roster_v2,
    freeze_policy_visible_question_trajectory,
    freeze_policy_visible_question_trajectory_v2,
    freeze_preselection_state_from_production_components,
    freeze_prospective_adaptive_candidate,
    freeze_prospective_adaptive_candidate_v2,
    freeze_question_reference_verdict_v2,
    join_condition_calibration_assessment_receipts,
    join_labeled_question_trajectory_v2,
    join_terminal_condition_gates,
    validate_adaptive_calibration_bundle_v2_integrity,
)
from literature_multiverse.independence_identity import (
    IndependenceIdentityError,
    canonicalize_authority_identity,
)
from literature_multiverse.lineage import hash_canonical


@pytest.fixture(autouse=True)
def _replay_explicit_synthetic_receipt_fixtures(monkeypatch):
    """Keep statistical unit tests light without weakening the public receipt API.

    Only dictionaries carrying this private test marker are projected. Every real
    object still reaches the production external-replay function, and dedicated
    red-team tests below prove unmarked/bare inputs fail closed.
    """

    original = adaptive_calibration_module._externally_replay_calibration_assessment_receipt

    def replay(receipt: Any) -> Any:
        if not isinstance(receipt, dict) or receipt.get("_synthetic_receipt_fixture") is not True:
            return original(receipt)
        visible = PolicyVisibleQuestionTrajectoryV2.model_validate(receipt["visible"])
        result = ConditionCalibrationGateResultV1.model_validate(receipt["calibration_gate_result"])
        source_payload = receipt["collection_source"]
        source = SimpleNamespace(
            question_id=receipt["question_id"],
            policy_arm_id=receipt["policy_arm_id"],
            adaptive_policy_context=SimpleNamespace(
                policy_context_sha256=receipt["policy_context_sha256"]
            ),
            policy_visible_question_trajectory=visible,
            collection_source_sha256=source_payload["collection_source_sha256"],
            collection_source_decision_sha256=source_payload["collection_source_decision_sha256"],
        )
        replayed = SimpleNamespace(
            question_id=receipt["question_id"],
            policy_arm_id=receipt["policy_arm_id"],
            policy_visible_question_trajectory=visible,
            calibration_gate_result=result,
            collection_source=source,
            source_anchor=adaptive_calibration_module._collection_source_anchor(source),
            source_roster_sha256=receipt["source_roster_sha256"],
            source_membership_sha256=receipt["source_membership_sha256"],
        )
        replayed.model_dump = lambda *, mode: receipt
        return replayed

    monkeypatch.setattr(
        adaptive_calibration_module,
        "_externally_replay_calibration_assessment_receipt",
        replay,
    )


def _digest(value: int | str) -> str:
    if isinstance(value, int):
        return f"{value:x}".rjust(64, "0")
    return hash_canonical({"fixture": value})


def _context() -> AdaptivePolicyContext:
    return freeze_adaptive_policy_context(
        policy_arm_id="adaptive",
        population_id="complete-biomedical-questions-v2",
        pipeline_sha256="a" * 64,
        allocation_policy={"name": "adaptive-voi", "seed": 31},
        budget_minutes=10.0,
        release_config={"version": "release-v2", "alpha": 0.99},
        audit_config={"version": "audit-v2", "unit": "person_minutes"},
        target_semantics={"loss": "exact_released_decision_mismatch"},
        corpus_protocol_context={"cutoff_rule": "before-2026-01-01"},
        score_feature_names=["risk"],
    )


def _identity(index: int, *, registry_index: int | None = None, verified: bool = True):
    reasons = () if verified else ("authority_identity_ambiguous",)
    registry = index if registry_index is None else registry_index
    return freeze_adaptive_independence_identity_v2(
        strong_components=[
            {
                "doi": [f"10.9000/question-{index}"],
                "pmid": [str(100_000 + index)],
                "registry_id": [f"clinicaltrials.gov:NCT{registry:08d}"],
                "dataset_id": [f"zenodo.org:dataset-{index}"],
            }
        ],
        unverification_reasons=reasons,
    )


def _state(
    index: int,
    *,
    prefix: int,
    decision: str,
    passed: bool,
    evidence_graph_sha256: str,
    risk: float = 0.05,
) -> AdaptivePreselectionState:
    return freeze_adaptive_preselection_state(
        prefix_index=prefix,
        audit_prefix_item_ids=[f"q{index}-item-{item}" for item in range(prefix)],
        audit_prefix_cost_minutes=float(prefix),
        scheduler_state_sha256=_digest(f"scheduler-{index}-{prefix}"),
        evidence_graph_sha256=evidence_graph_sha256,
        synthesis_sha256=_digest(f"synthesis-{index}-{prefix}"),
        non_calibration_assessment_sha256=_digest(f"assessment-{index}-{prefix}"),
        non_calibration_gates_passed=passed,
        non_calibration_blocking_reasons=[] if passed else ["evidence_not_ready"],
        claim_decision=decision,
        score_features={"risk": risk},
    )


def _projection(
    *,
    index: int,
    corpus_membership_sha256: str,
    corpus_cutoff: str,
    identity,
    semantics,
) -> ConditionCalibrationProjectionV1:
    development_graph_sha256 = _digest(f"development-graph-{index}")
    return freeze_condition_calibration_projection(
        question_id=f"question-{index:03d}",
        target_semantics=semantics,
        independence_identity=identity,
        question_config_sha256=_digest(f"question-config-{index}"),
        corpus_snapshot_sha256=corpus_membership_sha256,
        corpus_cutoff=corpus_cutoff,
        plan_sha256=_digest(f"condition-plan-{index}"),
        materialization_receipt_sha256=_digest(f"materialization-{index}"),
        full_graph_sha256=_digest(f"full-graph-{index}"),
        development_graph_sha256=development_graph_sha256,
        confirmation_graph_sha256=_digest(f"confirmation-graph-{index}"),
        development_partition_sha256=_digest(f"development-partition-{index}"),
        confirmation_partition_sha256=_digest(f"confirmation-partition-{index}"),
        confirmation_config_sha256=_digest(f"confirmation-config-{index}"),
        pipeline_sha256="a" * 64,
        synthesis_runner_sha256=_digest("synthesis-runner-v2"),
        candidate_runner_sha256=_digest("candidate-runner-v2"),
        prespecified_moderator_names=["dose"],
    )


def _visible(
    index: int,
    *,
    split: Literal["development", "calibration", "test"],
    context: AdaptivePolicyContext,
    decision: str = "condition_dependent",
    passed: bool = True,
    domain: str = "medicine",
    registry_index: int | None = None,
    verified_identity: bool = True,
) -> PolicyVisibleQuestionTrajectoryV2:
    question_id = f"question-{index:03d}"
    identity = _identity(
        index,
        registry_index=registry_index,
        verified=verified_identity,
    )
    semantics = freeze_adaptive_target_semantics_v2(
        question_id=question_id,
        claim_spec_sha256=_digest(f"claim-{index}"),
        global_condition_target_sha256=_digest(f"global-condition-{index}"),
    )
    corpus = freeze_complete_corpus_identity(
        corpus_id=f"corpus-{index:03d}",
        corpus_source_sha256=_digest(f"corpus-source-{index}"),
        corpus_cutoff="2025-12-31",
        publication_ids=[f"publication-{index:03d}"],
        source_manifest_sha256=_digest(f"manifest-{index}"),
    )
    projection = (
        _projection(
            index=index,
            corpus_membership_sha256=corpus.membership_sha256,
            corpus_cutoff=corpus.corpus_cutoff,
            identity=identity,
            semantics=semantics,
        )
        if decision == "condition_dependent"
        else None
    )
    evidence_graph_sha256 = (
        projection.online_graph_sha256
        if projection is not None
        else _digest(f"development-graph-{index}")
    )
    state = _state(
        index,
        prefix=0,
        decision=decision,
        passed=passed,
        evidence_graph_sha256=evidence_graph_sha256,
    )
    if projection is None:
        candidates: list[AdaptiveTerminalAuditCandidate] = []
        terminal_reason = "all_items_resolved"
    else:
        candidates = [
            AdaptiveTerminalAuditCandidate(
                item_id=f"q{index}-pending",
                eligible=True,
                estimated_cost_minutes=1.0,
                source_candidate_sha256=_digest(f"candidate-{index}"),
            )
        ]
        terminal_reason = (
            "full_nonconfirmation_release_gates_passed"
            if passed
            else "nonconfirmation_context_blocked"
        )
    base_arm = freeze_adaptive_policy_arm_trajectory(
        policy_arm_id=context.policy_arm_id,
        policy_context_sha256=context.policy_context_sha256,
        states=[state],
        terminal_reason=terminal_reason,  # type: ignore[arg-type]
        terminal_candidates=candidates,
        terminal_source_candidate_input_sha256=_digest(f"candidate-input-{index}"),
        terminal_remaining_budget_minutes=10.0,
        terminal_nonconfirmation_blocking_reasons=(
            [] if passed else state.non_calibration_blocking_reasons
        ),
        terminal_condition_projection=projection if passed else None,
    )
    base_visible = freeze_policy_visible_question_trajectory(
        question_id=question_id,
        split=split,
        population_id=context.population_id,
        domain=domain,
        corpus=corpus,
        arms=[base_arm],
    )
    wrapped = freeze_confirmation_aware_arm_trajectory(
        base_arm=base_arm,
        terminal_condition_projection=projection,
    )
    return freeze_policy_visible_question_trajectory_v2(
        base_visible=base_visible,
        target_semantics=semantics,
        independence_identity=identity,
        arms=[wrapped],
    )


def _reference(
    visible: PolicyVisibleQuestionTrajectoryV2,
    *,
    verdict: str | None = None,
    label_source: Literal[
        "benchmark_annotation", "expert_adjudication", "simulation"
    ] = "expert_adjudication",
) -> QuestionReferenceVerdictV2:
    return freeze_question_reference_verdict_v2(
        question_id=visible.base_visible.question_id,
        verdict=verdict or visible.arms[0].base_arm.states[-1].claim_decision,  # type: ignore[arg-type]
        target_semantics=visible.target_semantics,
        label_source=label_source,
        adjudicator_count=2,
        adjudication_protocol_sha256="d" * 64,
        adjudication_artifact_sha256=_digest(f"adjudication-{visible.base_visible.question_id}"),
    )


def _gate_result(
    visible: PolicyVisibleQuestionTrajectoryV2,
    *,
    status: Literal["missing", "confirmed", "not_confirmed", "insufficient"] = ("confirmed"),
):
    arm = visible.arms[0]
    projection = arm.terminal_condition_projection
    invocation = arm.condition_gate_invocation_proof
    assert projection is not None and invocation is not None
    reasons = [] if status == "confirmed" else [f"condition_confirmation_{status}"]
    materialized = status != "missing"
    gate = freeze_condition_confirmation_gate_assessment(
        provisional_claim_decision="condition_dependent",
        status=status,
        reasons=reasons,
        condition_projection_sha256=projection.projection_sha256,
        target_sha256=projection.condition_target_sha256,
        plan_sha256=projection.plan_sha256,
        config_sha256=projection.confirmation_config_sha256,
        model_sha256=_digest(f"condition-model-{visible.base_visible.question_id}")
        if materialized
        else None,
        assessment_sha256=_digest(
            f"condition-assessment-{visible.base_visible.question_id}-{status}"
        )
        if materialized
        else None,
    )
    return freeze_condition_terminal_gate_result_v2(
        question_id=visible.base_visible.question_id,
        policy_arm_id=arm.base_arm.policy_arm_id,
        condition_gate_invocation_proof=invocation,
        gate_assessment=gate,
        source_v6_certificate_sha256=_digest(f"source-v6-{visible.base_visible.question_id}"),
        source_v6_decision_sha256=_digest(f"source-v6-decision-{visible.base_visible.question_id}"),
    )


def _anchor_freeze_for_synthetic_receipts(
    freeze: AdaptiveDevelopmentFreezeV2,
    visible_rows: list[PolicyVisibleQuestionTrajectoryV2],
) -> AdaptiveDevelopmentFreezeV2:
    source_roster_sha256 = _digest("synthetic-source-roster")
    anchors: list[ConditionCalibrationCollectionSourceAnchorV1] = []
    for visible in visible_rows:
        for arm in visible.arms:
            question_id = visible.base_visible.question_id
            anchor_payload = {
                "anchor_version": "condition-calibration-source-anchor-v1",
                "question_id": question_id,
                "policy_arm_id": arm.base_arm.policy_arm_id,
                "policy_context_sha256": arm.base_arm.policy_context_sha256,
                "visible_trajectory_sha256": visible.trajectory_sha256,
                "collection_source_sha256": _digest(f"collection-source-{question_id}"),
                "collection_source_decision_sha256": _digest(f"collection-decision-{question_id}"),
            }
            anchors.append(
                ConditionCalibrationCollectionSourceAnchorV1.model_validate(
                    {
                        **anchor_payload,
                        "anchor_sha256": hash_canonical(anchor_payload),
                    }
                )
            )
    anchors.sort(key=lambda row: (row.question_id, row.policy_arm_id))
    roster_payload = freeze.calibration_roster.model_dump(mode="json", exclude={"roster_sha256"})
    roster_payload.update(
        {
            "collection_source_roster_sha256": source_roster_sha256,
            "collection_source_membership_sha256": hash_canonical(anchors),
            "collection_source_anchors": anchors,
            "collection_source_status": "externally_replayed_before_assessment",
        }
    )
    roster = AdaptiveCalibrationRosterV2.model_validate(
        {**roster_payload, "roster_sha256": hash_canonical(roster_payload)}
    )
    plan_payload = freeze.calibration_plan.model_dump(mode="json", exclude={"plan_sha256"})
    plan_payload["calibration_roster_sha256"] = roster.roster_sha256
    plan = AdaptiveCalibrationPlanV2.model_validate(
        {**plan_payload, "plan_sha256": hash_canonical(plan_payload)}
    )
    freeze_payload = freeze.model_dump(mode="json", exclude={"development_freeze_sha256"})
    freeze_payload["calibration_roster"] = roster
    freeze_payload["calibration_plan"] = plan
    return AdaptiveDevelopmentFreezeV2.model_validate(
        {
            **freeze_payload,
            "development_freeze_sha256": hash_canonical(freeze_payload),
        }
    )


def _synthetic_calibration_receipt(
    visible: PolicyVisibleQuestionTrajectoryV2,
    *,
    roster: AdaptiveCalibrationRosterV2,
    status: Literal["confirmed", "not_confirmed", "insufficient"],
) -> dict[str, Any]:
    arm = visible.arms[0]
    production_result = _gate_result(visible, status=status)
    anchor = next(
        row
        for row in roster.collection_source_anchors
        if row.question_id == visible.base_visible.question_id
        and row.policy_arm_id == arm.base_arm.policy_arm_id
    )
    invocation = arm.condition_gate_invocation_proof
    assert invocation is not None
    result = freeze_condition_calibration_gate_result_v1(
        question_id=visible.base_visible.question_id,
        policy_arm_id=arm.base_arm.policy_arm_id,
        condition_gate_invocation_proof=invocation,
        gate_assessment=production_result.gate_assessment,
        collection_source_sha256=anchor.collection_source_sha256,
        collection_source_decision_sha256=(anchor.collection_source_decision_sha256),
    )
    assert roster.collection_source_roster_sha256 is not None
    assert roster.collection_source_membership_sha256 is not None
    return {
        "_synthetic_receipt_fixture": True,
        "question_id": visible.base_visible.question_id,
        "policy_arm_id": arm.base_arm.policy_arm_id,
        "policy_context_sha256": arm.base_arm.policy_context_sha256,
        "visible": visible.model_dump(mode="json"),
        "collection_source": {
            "collection_source_sha256": anchor.collection_source_sha256,
            "collection_source_decision_sha256": (anchor.collection_source_decision_sha256),
        },
        "source_roster_sha256": roster.collection_source_roster_sha256,
        "source_membership_sha256": roster.collection_source_membership_sha256,
        "calibration_gate_result": result.model_dump(mode="json"),
    }


def _freeze_and_bundle(
    *,
    calibration_gate_status: Literal["confirmed", "not_confirmed", "insufficient"] = "confirmed",
    label_source: Literal[
        "benchmark_annotation", "expert_adjudication", "simulation"
    ] = "expert_adjudication",
    calibration_reference: str | None = None,
    decision: str = "condition_dependent",
    verified_identity: bool = True,
) -> tuple[
    AdaptivePolicyContext,
    list[PolicyVisibleQuestionTrajectoryV2],
    AdaptiveDevelopmentFreezeV2,
    GateCompleteCalibrationRosterV2,
    AdaptiveCalibrationBundleV2,
]:
    context = _context()
    development_visible = [
        _visible(
            1,
            split="development",
            context=context,
            decision=decision,
            verified_identity=verified_identity,
        )
    ]
    calibration_visible = [
        _visible(
            10,
            split="calibration",
            context=context,
            decision=decision,
            verified_identity=verified_identity,
        ),
        _visible(
            11,
            split="calibration",
            context=context,
            decision=decision,
            verified_identity=verified_identity,
        ),
    ]
    development = [
        join_labeled_question_trajectory_v2(
            visible=row,
            reference=_reference(row, label_source=label_source),
        )
        for row in development_visible
    ]
    freeze = fit_adaptive_development_v2(
        development,
        policy_contexts=[context],
        calibration_visible_trajectories=calibration_visible,
        alpha=0.99,
        delta=0.5,
        candidate_thresholds={"adaptive": [1.0]},
        seed=19,
    )
    if decision == "condition_dependent":
        freeze = _anchor_freeze_for_synthetic_receipts(
            freeze,
            calibration_visible,
        )
    gate_complete = [
        join_condition_calibration_assessment_receipts(
            visible=row,
            calibration_roster=freeze.calibration_roster,
            calibration_assessment_receipts=(
                [
                    _synthetic_calibration_receipt(
                        row,
                        roster=freeze.calibration_roster,
                        status=calibration_gate_status,
                    )
                ]
                if decision == "condition_dependent"
                else []
            ),
        )
        for row in calibration_visible
    ]
    gate_roster = freeze_gate_complete_calibration_roster_v2(
        development_freeze=freeze,
        trajectories=gate_complete,
    )
    references = [
        _reference(
            row,
            verdict=calibration_reference or decision,
            label_source=label_source,
        )
        for row in calibration_visible
    ]
    bundle = calibrate_confirmation_aware_first_release(
        freeze,
        gate_roster,
        references,
    )
    return context, calibration_visible, freeze, gate_roster, bundle


def test_v2_calibrates_exact_five_way_joint_release_and_replays_tamper_evidently() -> None:
    _, calibration, freeze, gate_roster, bundle = _freeze_and_bundle()

    assert freeze.freeze_state == ("terminal_confirmation_outcomes_and_calibration_labels_unopened")
    assert gate_roster.freeze_state == "reference_labels_unopened"
    assert bundle.status == "calibrated"
    assert bundle.real_release_eligible is True
    assert bundle.selected is not None
    outcomes = bundle.selected.outcomes
    assert len(outcomes) == len(calibration)
    assert all(row.accepted and not row.error for row in outcomes)
    assert {row.released_claim_decision for row in outcomes} == {"condition_dependent"}
    assert all(row.calibration_gate_result_sha256 is not None for row in outcomes)
    condition_domains = bundle.selected.condition_domain_calibrations
    assert len(condition_domains) == 1
    assert condition_domains[0].domain == "medicine"
    assert condition_domains[0].confirmed_condition_releases == 2
    assert condition_domains[0].errors == 0
    assert condition_domains[0].passed is True

    tampered = bundle.model_dump(mode="json")
    tampered["candidates"][0]["outcomes"][0]["error"] = True
    tampered_outcome = tampered["candidates"][0]["outcomes"][0]
    tampered_outcome["outcome_sha256"] = hash_canonical(
        {key: value for key, value in tampered_outcome.items() if key != "outcome_sha256"}
    )
    tampered["candidates"][0]["errors"] = 1
    tampered["candidates"][0]["empirical_risk"] = 0.5
    tampered["bundle_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "bundle_sha256"}
    )
    with pytest.raises(ValidationError, match="confirmation_v2_bundle_replay_outcome_mismatch"):
        AdaptiveCalibrationBundleV2.model_validate(tampered)


def test_v2_requires_confirmed_condition_release_support_in_every_frozen_domain() -> None:
    _, _, freeze, _, bundle = _freeze_and_bundle(decision="supported")

    assert freeze.calibration_plan.condition_release_domains == ["medicine"]
    candidate = bundle.candidates[0]
    assert candidate.accepted == 2
    assert candidate.errors == 0
    assert candidate.simultaneous_upper_risk is not None
    assert candidate.simultaneous_upper_risk <= bundle.alpha
    assert candidate.condition_domain_calibrations[0].confirmed_condition_releases == 0
    assert candidate.condition_domain_calibrations[0].simultaneous_upper_risk is None
    assert candidate.condition_domain_calibrations[0].passed is False
    assert candidate.passed is False
    assert bundle.selected is None
    assert bundle.status == "abstain_all"
    assert bundle.real_release_eligible is False


def test_v2_condition_domain_support_is_semantically_replayed_after_rehash() -> None:
    _, _, _, _, bundle = _freeze_and_bundle()
    tampered = bundle.model_dump(mode="json")
    stratum = tampered["candidates"][0]["condition_domain_calibrations"][0]
    stratum["confirmed_condition_releases"] = 1
    stratum["simultaneous_upper_risk"] = 0.75
    tampered["bundle_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "bundle_sha256"}
    )
    with pytest.raises(
        ValidationError,
        match="confirmation_v2_bundle_condition_domain_replay_mismatch",
    ):
        AdaptiveCalibrationBundleV2.model_validate(tampered)


def test_v2_candidate_fails_when_one_frozen_domain_has_no_condition_release_support() -> None:
    context = _context()
    development_visible = _visible(50, split="development", context=context)
    medicine = _visible(51, split="calibration", context=context, domain="medicine")
    oncology = _visible(
        52,
        split="calibration",
        context=context,
        domain="oncology",
        decision="supported",
    )
    freeze = fit_adaptive_development_v2(
        [
            join_labeled_question_trajectory_v2(
                visible=development_visible,
                reference=_reference(development_visible),
            )
        ],
        policy_contexts=[context],
        calibration_visible_trajectories=[medicine, oncology],
        alpha=0.99,
        delta=0.5,
        candidate_thresholds={"adaptive": [1.0]},
    )
    freeze = _anchor_freeze_for_synthetic_receipts(
        freeze,
        [medicine, oncology],
    )
    gate_roster = freeze_gate_complete_calibration_roster_v2(
        development_freeze=freeze,
        trajectories=[
            join_condition_calibration_assessment_receipts(
                visible=medicine,
                calibration_roster=freeze.calibration_roster,
                calibration_assessment_receipts=[
                    _synthetic_calibration_receipt(
                        medicine,
                        roster=freeze.calibration_roster,
                        status="confirmed",
                    )
                ],
            ),
            join_condition_calibration_assessment_receipts(
                visible=oncology,
                calibration_roster=freeze.calibration_roster,
                calibration_assessment_receipts=[],
            ),
        ],
    )
    bundle = calibrate_confirmation_aware_first_release(
        freeze,
        gate_roster,
        [_reference(medicine), _reference(oncology)],
    )

    assert freeze.calibration_plan.condition_release_domains == ["medicine", "oncology"]
    by_domain = {row.domain: row for row in bundle.candidates[0].condition_domain_calibrations}
    assert by_domain["medicine"].confirmed_condition_releases == 1
    assert by_domain["medicine"].passed is True
    assert by_domain["oncology"].confirmed_condition_releases == 0
    assert by_domain["oncology"].passed is False
    assert bundle.status == "abstain_all"


def test_terminal_gate_outcomes_are_post_policy_and_never_policy_features() -> None:
    _, calibration, _, _, _ = _freeze_and_bundle()
    visible = calibration[0]
    serialized_visible = visible.model_dump_json()
    assert "scientific_gate_passed" not in serialized_visible
    assert "condition_confirmation_status" not in serialized_visible

    with pytest.raises(
        ValidationError,
        match="terminal_condition_outcome_leaked_into_policy",
    ):
        freeze_adaptive_preselection_state(
            prefix_index=0,
            audit_prefix_item_ids=[],
            audit_prefix_cost_minutes=0.0,
            scheduler_state_sha256="1" * 64,
            evidence_graph_sha256="2" * 64,
            synthesis_sha256="3" * 64,
            non_calibration_assessment_sha256="4" * 64,
            non_calibration_gates_passed=False,
            non_calibration_blocking_reasons=["blocked"],
            claim_decision="condition_dependent",
            score_features={"terminal_gate_status": 1.0},
        )


def test_invocation_is_first_outcome_free_nonconfirmation_state_and_binds_actions() -> None:
    context = _context()
    visible = _visible(20, split="calibration", context=context)
    arm = visible.arms[0]
    proof = arm.condition_gate_invocation_proof
    assert isinstance(proof, ConditionGateInvocationProofV2)
    assert proof.invocation_basis == "first_nonconfirmation_eligible_state"
    assert proof.unresolved_feasible_action_ids == ["q20-pending"]
    assert proof.confirmation_outcomes_unopened is True
    assert proof.reference_labels_unopened is True

    tampered = proof.model_dump(mode="json")
    tampered["available_actions"][0]["estimated_cost_minutes"] = 11.0
    tampered["proof_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "proof_sha256"}
    )
    with pytest.raises(ValidationError, match="condition_invocation_action_roster_hash_mismatch"):
        ConditionGateInvocationProofV2.model_validate(tampered)

    state = arm.base_arm.states[-1]
    earlier = freeze_adaptive_preselection_state(
        prefix_index=0,
        audit_prefix_item_ids=[],
        audit_prefix_cost_minutes=0,
        scheduler_state_sha256=_digest("earlier-state"),
        evidence_graph_sha256=state.evidence_graph_sha256,
        synthesis_sha256=_digest("earlier-synthesis"),
        non_calibration_assessment_sha256=_digest("earlier-assessment"),
        non_calibration_gates_passed=True,
        non_calibration_blocking_reasons=[],
        claim_decision="condition_dependent",
        score_features=state.score_features,
    )
    later = freeze_adaptive_preselection_state(
        prefix_index=1,
        audit_prefix_item_ids=["q20-item-0"],
        audit_prefix_cost_minutes=1,
        scheduler_state_sha256=_digest("later-state"),
        evidence_graph_sha256=state.evidence_graph_sha256,
        synthesis_sha256=_digest("later-synthesis"),
        non_calibration_assessment_sha256=_digest("later-assessment"),
        non_calibration_gates_passed=True,
        non_calibration_blocking_reasons=[],
        claim_decision="condition_dependent",
        score_features=state.score_features,
    )
    with pytest.raises(ValidationError, match="condition_invocation_not_first_canonical_state"):
        freeze_adaptive_policy_arm_trajectory(
            policy_arm_id=context.policy_arm_id,
            policy_context_sha256=context.policy_context_sha256,
            states=[earlier, later],
            terminal_reason="full_nonconfirmation_release_gates_passed",
            terminal_candidates=[
                AdaptiveTerminalAuditCandidate(
                    item_id="q20-item-0",
                    eligible=True,
                    estimated_cost_minutes=1.0,
                    source_candidate_sha256=_digest("resolved-candidate"),
                ),
                AdaptiveTerminalAuditCandidate(
                    item_id="q20-pending",
                    eligible=True,
                    estimated_cost_minutes=1.0,
                    source_candidate_sha256=_digest("pending-candidate"),
                ),
            ],
            terminal_source_candidate_input_sha256=_digest("input-late"),
            terminal_remaining_budget_minutes=9.0,
            terminal_condition_projection=arm.terminal_condition_projection,
        )


def test_condition_gate_can_follow_an_ordinary_scheduler_terminal_proof() -> None:
    context = _context()
    visible = _visible(21, split="calibration", context=context)
    arm = visible.arms[0]
    projection = arm.terminal_condition_projection
    assert projection is not None
    ordinary = freeze_adaptive_policy_arm_trajectory(
        policy_arm_id=context.policy_arm_id,
        policy_context_sha256=context.policy_context_sha256,
        states=arm.base_arm.states,
        terminal_reason="all_items_resolved",
        terminal_candidates=[],
        terminal_source_candidate_input_sha256=_digest("ordinary-empty-roster"),
        terminal_remaining_budget_minutes=10.0,
        terminal_condition_projection=projection,
    )
    invocation = ordinary.terminal_proof
    assert isinstance(invocation, ConditionGateInvocationProofV2)
    assert invocation.invocation_basis == "ordinary_scheduler_terminal"
    assert invocation.ordinary_scheduler_proof is not None
    assert invocation.unresolved_feasible_action_ids == []


def test_context_blocked_terminal_binds_blockers_without_claiming_actions_infeasible() -> None:
    context = _context()
    visible = _visible(
        22,
        split="calibration",
        context=context,
        passed=False,
    )
    arm = visible.arms[0]
    proof = arm.base_arm.terminal_proof
    assert arm.base_arm.terminal_reason == "nonconfirmation_context_blocked"
    assert arm.terminal_condition_projection is not None
    assert arm.terminal_condition_required is False
    assert arm.condition_gate_invocation_proof is None
    assert proof.nonconfirmation_blocking_reasons == ["evidence_not_ready"]
    assert any(
        candidate.eligible and candidate.estimated_cost_minutes <= proof.remaining_budget_minutes
        for candidate in proof.candidates
    )

    tampered = arm.base_arm.model_dump(mode="json")
    tampered["terminal_proof"]["nonconfirmation_blocking_reasons"] = ["forged-blocker"]
    tampered["terminal_proof"]["proof_sha256"] = hash_canonical(
        {key: value for key, value in tampered["terminal_proof"].items() if key != "proof_sha256"}
    )
    tampered["arm_trajectory_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "arm_trajectory_sha256"}
    )
    with pytest.raises(
        ValidationError,
        match="adaptive_terminal_context_blocker_state_mismatch",
    ):
        type(arm.base_arm).model_validate(tampered)


@pytest.mark.parametrize("status", ["not_confirmed", "insufficient"])
def test_nonpassing_terminal_gate_abstains_in_joint_calibration(status: str) -> None:
    _, _, _, _, bundle = _freeze_and_bundle(
        calibration_gate_status=status,  # type: ignore[arg-type]
    )
    assert bundle.status == "abstain_all"
    assert bundle.selected is None
    assert all(not row.accepted for row in bundle.candidates[0].outcomes)


def test_missing_calibration_receipt_fails_before_reference_labels_open() -> None:
    context = _context()
    development_visible = _visible(60, split="development", context=context)
    calibration_visible = _visible(61, split="calibration", context=context)
    freeze = fit_adaptive_development_v2(
        [
            join_labeled_question_trajectory_v2(
                visible=development_visible,
                reference=_reference(development_visible),
            )
        ],
        policy_contexts=[context],
        calibration_visible_trajectories=[calibration_visible],
        alpha=0.99,
        delta=0.5,
        candidate_thresholds={"adaptive": [1.0]},
    )
    freeze = _anchor_freeze_for_synthetic_receipts(freeze, [calibration_visible])
    with pytest.raises(
        (AdaptiveCalibrationError, ValidationError),
        match="confirmation_v2_receipt_arm_coverage_mismatch",
    ):
        join_condition_calibration_assessment_receipts(
            visible=calibration_visible,
            calibration_roster=freeze.calibration_roster,
            calibration_assessment_receipts=[],
        )


def test_bare_or_invented_source_hash_gate_cannot_enter_receipt_roster() -> None:
    context = _context()
    development_visible = _visible(62, split="development", context=context)
    calibration_visible = _visible(63, split="calibration", context=context)
    freeze = fit_adaptive_development_v2(
        [
            join_labeled_question_trajectory_v2(
                visible=development_visible,
                reference=_reference(development_visible),
            )
        ],
        policy_contexts=[context],
        calibration_visible_trajectories=[calibration_visible],
        alpha=0.99,
        delta=0.5,
        candidate_thresholds={"adaptive": [1.0]},
    )
    freeze = _anchor_freeze_for_synthetic_receipts(freeze, [calibration_visible])
    bare = _gate_result(calibration_visible)
    with pytest.raises(
        AdaptiveCalibrationError,
        match="confirmation_v2_bare_terminal_gate_results_forbidden",
    ):
        join_terminal_condition_gates(
            visible=calibration_visible,
            terminal_gate_results=[bare],
        )
    with pytest.raises(
        AdaptiveCalibrationError,
        match="confirmation_v2_calibration_receipt_external_replay_failed",
    ):
        join_condition_calibration_assessment_receipts(
            visible=calibration_visible,
            calibration_roster=freeze.calibration_roster,
            calibration_assessment_receipts=[bare.model_dump(mode="json")],
        )


def test_collection_source_leakage_scan_distinguishes_target_orientation_from_outcome() -> None:
    adaptive_calibration_module._reject_collection_source_outcome_leakage(
        {
            "claim_manifest": {
                "global_condition_target": {"reference_direction": "higher"},
            }
        }
    )
    with pytest.raises(
        AdaptiveCalibrationError,
        match=r"collection_source_outcome_leakage:.*reference_verdict",
    ):
        adaptive_calibration_module._reject_collection_source_outcome_leakage(
            {"held_out": {"reference_verdict": "condition_dependent"}}
        )


def test_exact_decision_mismatch_counts_condition_reference_not_binary_support() -> None:
    _, _, _, _, bundle = _freeze_and_bundle(calibration_reference="supported")
    outcomes = bundle.candidates[0].outcomes
    assert all(row.accepted and row.error for row in outcomes)
    assert bundle.status == "abstain_all"


def test_strong_identity_is_authority_namespaced_content_silent_and_fail_closed() -> None:
    doi = canonicalize_authority_identity(kind="doi", value="10.9000/shared")
    pmid = canonicalize_authority_identity(kind="pmid", value="123")
    assert doi.token_sha256 != pmid.token_sha256
    identity = _identity(30)
    serialized = identity.model_dump_json()
    assert "10.9000/question-30" not in serialized
    assert "NCT00000030" not in serialized

    with pytest.raises(
        IndependenceIdentityError,
        match="authority_identity_namespace_unrecognized",
    ):
        freeze_adaptive_independence_identity_v2(
            strong_components=[{"registry_id": ["raw-local-id"]}]
        )

    _, _, _, _, simulated = _freeze_and_bundle(label_source="simulation")
    assert simulated.status == "calibrated"
    assert simulated.real_release_eligible is False

    _, _, unverified_freeze, _, unverified_bundle = _freeze_and_bundle(verified_identity=False)
    assert unverified_freeze.independence_verified is False
    assert unverified_bundle.independence_verified is False
    assert unverified_bundle.status == "abstain_all"
    assert unverified_bundle.selected is None
    assert unverified_bundle.real_release_eligible is False


def test_deferred_projector_uses_outcome_free_projection_and_exact_typed_blockers() -> None:
    context = _context()
    visible = _visible(35, split="calibration", context=context)
    projection = visible.arms[0].terminal_condition_projection
    assert projection is not None
    gate = freeze_condition_confirmation_gate_assessment(
        provisional_claim_decision="condition_dependent",
        status="missing",
        reasons=["condition_confirmation_pending"],
        condition_projection_sha256=projection.projection_sha256,
        target_sha256=projection.condition_target_sha256,
        plan_sha256=projection.plan_sha256,
        config_sha256=projection.confirmation_config_sha256,
    )
    sequential_state = SimpleNamespace(
        state_sha256=_digest("sequential-state"),
        session=SimpleNamespace(
            active_action=None,
            resolved_item_ids=[],
            historical_realized_cost=0.0,
        ),
    )
    assessment = SimpleNamespace(
        condition_calibration_projection=projection,
        terminal_gate_deferred=True,
        condition_confirmation_gate=gate,
        reasons=[
            "calibration:adaptive_confirmation_v2_required",
            "condition_confirmation_required",
            "condition_dependent_confirmation_aware_calibration_required",
        ],
        evidence={"exploratory_qualitative_condition_signal": True},
        config_sha256=_digest("release-config"),
        question_id=visible.base_visible.question_id,
        target={"claim": "content-free-fixture"},
        pipeline_sha256=context.pipeline_sha256,
        evidence_graph_sha256=projection.online_graph_sha256,
        synthesis_sha256=_digest("projected-synthesis"),
        paper_ids=["publication-035"],
        audit={"residual_risk_bound": 0.01},
        risk_features={"risk": 0.05},
    )
    projected = freeze_preselection_state_from_production_components(
        sequential_state=sequential_state,
        release_assessment=assessment,
        blocking_adapter_reasons=[],
    )
    assert projected.claim_decision == "condition_dependent"
    assert projected.non_calibration_gates_passed is True
    assert projected.non_calibration_blocking_reasons == []

    changed_gate = freeze_condition_confirmation_gate_assessment(
        provisional_claim_decision="condition_dependent",
        status="missing",
        reasons=["condition_confirmation_pending"],
        condition_projection_sha256=projection.projection_sha256,
        target_sha256=projection.condition_target_sha256,
        plan_sha256=projection.plan_sha256,
        config_sha256="f" * 64,
    )
    assessment.condition_confirmation_gate = changed_gate
    with pytest.raises(
        AdaptiveCalibrationError,
        match="adaptive_projection_condition_deferred_gate_contract_mismatch",
    ):
        freeze_preselection_state_from_production_components(
            sequential_state=sequential_state,
            release_assessment=assessment,
            blocking_adapter_reasons=[],
        )


def test_confirmation_outcomes_and_references_cannot_rewrite_invocation_proof() -> None:
    context = _context()
    visible = _visible(36, split="calibration", context=context)
    invocation = visible.arms[0].condition_gate_invocation_proof
    assert invocation is not None

    results = [
        _gate_result(visible, status=status)
        for status in ("missing", "confirmed", "not_confirmed", "insufficient")
    ]
    assert {result.condition_gate_invocation_proof_sha256 for result in results} == {
        invocation.proof_sha256
    }
    references = [
        _reference(visible, verdict=verdict)
        for verdict in (
            "supported",
            "contradicted",
            "condition_dependent",
            "inconclusive",
            "not_evaluable",
        )
    ]
    assert len({row.reference_sha256 for row in references}) == 5
    assert visible.arms[0].condition_gate_invocation_proof == invocation


def test_shared_multireport_cohort_identity_is_rejected_within_and_across_splits() -> None:
    context = _context()
    first = _visible(40, split="development", context=context, registry_index=999)
    second = _visible(41, split="development", context=context, registry_index=999)
    calibration = _visible(42, split="calibration", context=context)
    development = [
        join_labeled_question_trajectory_v2(visible=row, reference=_reference(row))
        for row in (first, second)
    ]
    with pytest.raises(AdaptiveCalibrationError, match="confirmation_v2_development_token_overlap"):
        fit_adaptive_development_v2(
            development,
            policy_contexts=[context],
            calibration_visible_trajectories=[calibration],
            alpha=0.99,
            delta=0.5,
            candidate_thresholds={"adaptive": [1.0]},
        )

    calibration_overlap = _visible(
        43,
        split="calibration",
        context=context,
        registry_index=999,
    )
    with pytest.raises(AdaptiveCalibrationError, match="confirmation_v2_cross_split_token_overlap"):
        fit_adaptive_development_v2(
            [
                join_labeled_question_trajectory_v2(
                    visible=first,
                    reference=_reference(first),
                )
            ],
            policy_contexts=[context],
            calibration_visible_trajectories=[calibration_overlap],
            alpha=0.99,
            delta=0.5,
            candidate_thresholds={"adaptive": [1.0]},
        )


def test_prospective_condition_release_requires_bundle_qualification_and_confirmed_gate() -> None:
    context, _, _, _, bundle = _freeze_and_bundle()
    visible = _visible(90, split="test", context=context)
    arm = visible.arms[0]
    invocation = arm.condition_gate_invocation_proof
    projection = arm.terminal_condition_projection
    assert invocation is not None and projection is not None
    base = freeze_prospective_adaptive_candidate(
        question_id=visible.base_visible.question_id,
        population_id=visible.base_visible.population_id,
        domain=visible.base_visible.domain,
        policy_arm_id=context.policy_arm_id,
        policy_context_sha256=context.policy_context_sha256,
        corpus=visible.base_visible.corpus,
        observed_states=arm.base_arm.states,
    )
    pending = freeze_prospective_adaptive_candidate_v2(
        base_candidate=base,
        target_semantics=visible.target_semantics,
        independence_identity=visible.independence_identity,
        condition_projection=projection,
        condition_gate_invocation_proof=invocation,
    )
    pending_assessment = assess_confirmation_aware_adaptive_release_candidate(
        pending,
        bundle,
    )
    assert pending_assessment.status == "abstained"
    assert pending_assessment.reason == "terminal_condition_release_qualification_missing"

    qualification = freeze_confirmation_aware_release_qualification_proof_v2(
        question_id=visible.base_visible.question_id,
        policy_arm_id=context.policy_arm_id,
        condition_gate_invocation_proof=invocation,
        bundle=bundle,
    )
    gate = _gate_result(visible, status="confirmed")
    candidate = freeze_prospective_adaptive_candidate_v2(
        base_candidate=base,
        target_semantics=visible.target_semantics,
        independence_identity=visible.independence_identity,
        condition_projection=projection,
        condition_gate_invocation_proof=invocation,
        release_qualification_proof=qualification,
        terminal_gate_result=gate,
    )
    assessment = assess_confirmation_aware_adaptive_release_candidate(candidate, bundle)
    assert assessment.status == "released"
    assert assessment.released_claim_decision == "condition_dependent"
    assert assessment.terminal_gate_result_sha256 == gate.result_sha256


def test_prospective_strong_overlap_and_coherently_rehashed_candidate_tamper_fail() -> None:
    context, calibration, _, _, bundle = _freeze_and_bundle()
    overlapping = _visible(91, split="test", context=context)
    payload = overlapping.model_dump(mode="json")
    payload["independence_identity"] = calibration[0].independence_identity.model_dump(mode="json")
    payload["independence_identity_sha256"] = calibration[0].independence_identity_sha256
    # The projection is deliberately omitted by using a non-condition candidate, so
    # the overlap test is isolated from projection identity binding.
    noncondition = _visible(
        92,
        split="test",
        context=context,
        decision="supported",
    )
    base = freeze_prospective_adaptive_candidate(
        question_id=noncondition.base_visible.question_id,
        population_id=noncondition.base_visible.population_id,
        domain=noncondition.base_visible.domain,
        policy_arm_id=context.policy_arm_id,
        policy_context_sha256=context.policy_context_sha256,
        corpus=noncondition.base_visible.corpus,
        observed_states=noncondition.arms[0].base_arm.states,
    )
    candidate = freeze_prospective_adaptive_candidate_v2(
        base_candidate=base,
        target_semantics=noncondition.target_semantics,
        independence_identity=calibration[0].independence_identity,
    )
    with pytest.raises(
        AdaptiveCalibrationError,
        match="confirmation_v2_prospective_strong_token_overlap",
    ):
        assess_confirmation_aware_adaptive_release_candidate(candidate, bundle)

    candidate_payload = candidate.model_dump(mode="json")
    candidate_payload["independence_identity_sha256"] = "f" * 64
    candidate_payload["candidate_sha256"] = hash_canonical(
        {key: value for key, value in candidate_payload.items() if key != "candidate_sha256"}
    )
    with pytest.raises(ValidationError, match="confirmation_v2_prospective_identity_mismatch"):
        ProspectiveAdaptiveReleaseCandidateV2.model_validate(candidate_payload)


def test_bundle_integrity_reparse_detects_nested_mutation() -> None:
    _, _, _, _, bundle = _freeze_and_bundle()
    mutated = deepcopy(bundle)
    object.__setattr__(mutated.candidates[0].outcomes[0], "error", True)
    with pytest.raises(AdaptiveCalibrationError, match="confirmation_v2_bundle_integrity_changed"):
        validate_adaptive_calibration_bundle_v2_integrity(mutated)
