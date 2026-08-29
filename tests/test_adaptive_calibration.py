from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from literature_multiverse.adaptive_calibration import (
    AdaptiveCalibrationBundle,
    AdaptiveCalibrationError,
    AdaptiveDevelopmentFreeze,
    AdaptivePolicyContext,
    AdaptivePreselectionState,
    AdaptiveTerminalAuditCandidate,
    LabeledQuestionTrajectory,
    assess_adaptive_release_candidate,
    calibrate_adaptive_first_release,
    fit_adaptive_development,
    freeze_adaptive_policy_arm_trajectory,
    freeze_adaptive_policy_context,
    freeze_adaptive_preselection_state,
    freeze_complete_corpus_identity,
    freeze_policy_visible_question_trajectory,
    freeze_prospective_adaptive_candidate,
    freeze_question_reference_verdict,
    join_labeled_question_trajectory,
)
from literature_multiverse.lineage import hash_canonical


def _contexts() -> list[AdaptivePolicyContext]:
    return [
        freeze_adaptive_policy_context(
            policy_arm_id=arm,
            population_id="biomed-claims-v1",
            pipeline_sha256="a" * 64,
            allocation_policy={"name": arm, "seed": 7},
            budget_minutes=10,
            release_config={"version": "release-v1", "alpha": 0.05},
            audit_config={
                "guard": "influence_and_expected_loss",
                "item_ucl_role": "scheduling_and_blocking_only",
            },
            target_semantics={
                "decision_states": ["supported", "contradicted", "inconclusive"],
                "loss": "claim_decision_differs_from_reference_verdict",
            },
            corpus_protocol_context={"cutoff_rule": "before-2026-01-01"},
            score_feature_names=["influence", "item_cell_rate_ucl"],
        )
        for arm in ("adaptive", "random")
    ]


def _state(
    *,
    question_index: int,
    arm_index: int,
    prefix: int,
    passed: bool,
    decision: str,
    feature: float,
) -> AdaptivePreselectionState:
    reasons = [] if passed else ["evidence:not_yet_release_eligible"]
    return freeze_adaptive_preselection_state(
        prefix_index=prefix,
        audit_prefix_item_ids=[f"q{question_index}-a{arm_index}-item-{i}" for i in range(prefix)],
        audit_prefix_cost_minutes=float(prefix * 3),
        scheduler_state_sha256=f"{question_index * 10 + arm_index * 3 + prefix + 1:x}".rjust(
            64, "0"
        ),
        evidence_graph_sha256=f"{question_index * 10 + prefix + 20:x}".rjust(64, "0"),
        synthesis_sha256=f"{question_index * 10 + prefix + 40:x}".rjust(64, "0"),
        non_calibration_assessment_sha256=(
            f"{question_index * 10 + arm_index * 3 + prefix + 60:x}".rjust(64, "0")
        ),
        non_calibration_gates_passed=passed,
        non_calibration_blocking_reasons=reasons,
        claim_decision=decision,
        score_features={"influence": feature, "item_cell_rate_ucl": 0.2},
    )


def _trajectory(
    *,
    question_index: int,
    split: str,
    contexts: list[AdaptivePolicyContext],
    reference: str,
    release_at_one: bool = True,
    empty_corpus: bool = False,
    source_manifest_extra: str | None = None,
) -> LabeledQuestionTrajectory:
    arms = []
    for arm_index, context in enumerate(contexts):
        decision = "supported" if question_index % 2 == 0 else "contradicted"
        states = [
            _state(
                question_index=question_index,
                arm_index=arm_index,
                prefix=0,
                passed=False,
                decision=decision,
                feature=0.8 if reference != decision else 0.2,
            ),
            _state(
                question_index=question_index,
                arm_index=arm_index,
                prefix=1,
                passed=release_at_one,
                decision=decision,
                feature=0.8 if reference != decision else 0.1,
            ),
        ]
        arms.append(
            freeze_adaptive_policy_arm_trajectory(
                policy_arm_id=context.policy_arm_id,
                policy_context_sha256=context.policy_context_sha256,
                states=states,
                terminal_reason="all_items_resolved",
                terminal_candidates=[
                    AdaptiveTerminalAuditCandidate(
                        item_id=item_id,
                        eligible=True,
                        estimated_cost_minutes=3.0,
                        source_candidate_sha256=hash_canonical({"item_id": item_id}),
                    )
                    for item_id in states[-1].audit_prefix_item_ids
                ],
                terminal_source_candidate_input_sha256=hash_canonical(
                    states[-1].audit_prefix_item_ids
                ),
                terminal_remaining_budget_minutes=(
                    context.budget_minutes - states[-1].audit_prefix_cost_minutes
                ),
            )
        )
    publications = [] if empty_corpus else [f"publication-{question_index}"]
    if source_manifest_extra is not None:
        publications.append(source_manifest_extra)
    corpus = freeze_complete_corpus_identity(
        corpus_id=f"corpus-{question_index}",
        corpus_source_sha256=f"{question_index + 100:x}".rjust(64, "0"),
        corpus_cutoff="2025-12-31",
        publication_ids=publications,
        source_manifest_sha256=f"{question_index + 200:x}".rjust(64, "0"),
    )
    visible = freeze_policy_visible_question_trajectory(
        question_id=f"question-{question_index}",
        split=split,  # type: ignore[arg-type]
        population_id="biomed-claims-v1",
        domain="medicine",
        corpus=corpus,
        arms=arms,
    )
    reference_row = freeze_question_reference_verdict(
        question_id=f"question-{question_index}",
        verdict=reference,
        label_source="expert_adjudication",
        adjudication_protocol_sha256="d" * 64,
        adjudication_artifact_sha256=f"{question_index + 400:x}".rjust(64, "0"),
    )
    return join_labeled_question_trajectory(visible=visible, reference=reference_row)


def _bundle() -> tuple[
    list[AdaptivePolicyContext], list[LabeledQuestionTrajectory], AdaptiveCalibrationBundle
]:
    contexts = _contexts()
    development = [
        _trajectory(
            question_index=index,
            split="development",
            contexts=contexts,
            reference=("supported" if index % 3 else "contradicted"),
        )
        for index in range(1, 7)
    ]
    calibration = [
        _trajectory(
            question_index=index,
            split="calibration",
            contexts=contexts,
            reference=("supported" if index % 2 == 0 else "contradicted"),
            release_at_one=index != 10,
            empty_corpus=index == 10,
        )
        for index in range(7, 12)
    ]
    freeze = fit_adaptive_development(
        development,
        policy_contexts=contexts,
        calibration_visible_trajectories=[row.visible for row in calibration],
        alpha=0.99,
        delta=0.5,
        candidate_thresholds={"adaptive": [1.0], "random": [1.0]},
        seed=9,
    )
    bundle = calibrate_adaptive_first_release(
        freeze,
        calibration,
    )
    return contexts, calibration, bundle


def _unscored_prefix(
    row: LabeledQuestionTrajectory,
    *,
    arm_id: str,
    count: int,
) -> list[AdaptivePreselectionState]:
    arm = next(arm for arm in row.visible.arms if arm.policy_arm_id == arm_id)
    return list(arm.states[:count])


def test_development_freezes_thresholds_before_calibration_and_replays_one_outcome() -> None:
    _, calibration, bundle = _bundle()

    assert bundle.development_freeze.freeze_state == "calibration_labels_unopened"
    assert bundle.development_freeze.threshold_family.definition_source == "development_only"
    assert bundle.status == "calibrated"
    assert bundle.selected is not None
    assert len(bundle.candidates) == 2
    for candidate in bundle.candidates:
        assert candidate.total_questions == len(calibration)
        assert len(candidate.outcomes) == len(calibration)
        assert {row.question_id for row in candidate.outcomes} == {
            row.visible.question_id for row in calibration
        }
        # The empty/forced-abstention question remains in the denominator.
        empty = next(row for row in candidate.outcomes if row.question_id == "question-10")
        assert empty.accepted is False
        assert empty.error is False


def test_reference_sidecar_cannot_leak_into_policy_visible_features() -> None:
    with pytest.raises(ValidationError, match="reference_label_leaked_into_policy"):
        freeze_adaptive_preselection_state(
            prefix_index=0,
            audit_prefix_item_ids=[],
            audit_prefix_cost_minutes=0,
            scheduler_state_sha256="1" * 64,
            evidence_graph_sha256="2" * 64,
            synthesis_sha256="3" * 64,
            non_calibration_assessment_sha256="4" * 64,
            non_calibration_gates_passed=False,
            non_calibration_blocking_reasons=["blocked"],
            claim_decision="supported",
            score_features={"reference_verdict": 1.0},
        )


def test_reference_sidecar_is_bound_to_exact_question() -> None:
    contexts = _contexts()
    row = _trajectory(
        question_index=21,
        split="development",
        contexts=contexts,
        reference="supported",
    )
    wrong_reference = freeze_question_reference_verdict(
        question_id="different-question",
        verdict="supported",
        label_source="expert_adjudication",
        adjudication_protocol_sha256="d" * 64,
        adjudication_artifact_sha256="e" * 64,
    )
    with pytest.raises(
        ValidationError,
        match="adaptive_reference_question_identity_mismatch",
    ):
        join_labeled_question_trajectory(
            visible=row.visible,
            reference=wrong_reference,
        )


def test_alpha_delta_are_frozen_before_calibration_and_bundle_cannot_override() -> None:
    _, _, bundle = _bundle()
    plan = bundle.development_freeze.calibration_plan
    assert (plan.alpha, plan.delta) == (0.99, 0.5)
    assert bundle.alpha == plan.alpha
    assert bundle.delta == plan.delta

    payload = bundle.model_dump(mode="json")
    payload["alpha"] = 0.98
    payload["bundle_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "bundle_sha256"}
    )
    with pytest.raises(
        ValidationError,
        match="adaptive_bundle_calibration_plan_risk_target_mismatch",
    ):
        AdaptiveCalibrationBundle.model_validate(payload)


def test_calibration_requires_one_frozen_adjudication_protocol() -> None:
    contexts = _contexts()
    development = [
        _trajectory(
            question_index=index,
            split="development",
            contexts=contexts,
            reference="supported",
        )
        for index in range(40, 44)
    ]
    freeze = fit_adaptive_development(
        development,
        policy_contexts=contexts,
        calibration_visible_trajectories=[
            _trajectory(
                question_index=index,
                split="calibration",
                contexts=contexts,
                reference="supported",
            ).visible
            for index in range(44, 46)
        ],
        alpha=0.99,
        delta=0.5,
        candidate_thresholds={"adaptive": [1.0], "random": [1.0]},
    )
    calibration = [
        _trajectory(
            question_index=index,
            split="calibration",
            contexts=contexts,
            reference="supported",
        )
        for index in range(44, 46)
    ]
    changed = calibration[1].model_dump(mode="json")
    reference = changed["reference"]
    reference["adjudication_protocol_sha256"] = "c" * 64
    reference["reference_sha256"] = hash_canonical(
        {key: value for key, value in reference.items() if key != "reference_sha256"}
    )
    changed["labeled_trajectory_sha256"] = hash_canonical(
        {key: value for key, value in changed.items() if key != "labeled_trajectory_sha256"}
    )
    calibration[1] = LabeledQuestionTrajectory.model_validate(changed)
    with pytest.raises(
        AdaptiveCalibrationError,
        match="adaptive_calibration_adjudication_protocol_changed",
    ):
        calibrate_adaptive_first_release(freeze, calibration)


def test_calibration_roster_rejects_omission_and_valid_substitution() -> None:
    contexts, calibration, bundle = _bundle()
    freeze = bundle.development_freeze
    with pytest.raises(
        AdaptiveCalibrationError,
        match="adaptive_calibration_visible_roster_mismatch",
    ):
        calibrate_adaptive_first_release(freeze, calibration[:-1])

    substitute = _trajectory(
        question_index=7,
        split="calibration",
        contexts=contexts,
        reference=calibration[0].reference.verdict,
        release_at_one=False,
    )
    substituted = [substitute, *calibration[1:]]
    with pytest.raises(
        AdaptiveCalibrationError,
        match="adaptive_calibration_visible_roster_mismatch",
    ):
        calibrate_adaptive_first_release(freeze, substituted)


def test_calibration_roster_input_order_is_canonical_and_tamper_evident() -> None:
    _, calibration, bundle = _bundle()
    reordered = calibrate_adaptive_first_release(
        bundle.development_freeze,
        list(reversed(calibration)),
    )
    assert reordered.bundle_sha256 == bundle.bundle_sha256

    mutated_freeze = deepcopy(bundle.development_freeze)
    mutated_freeze.calibration_roster.visible_trajectories[0].arms[0].states[0].score_features[
        "influence"
    ] = 0.999
    with pytest.raises(
        AdaptiveCalibrationError,
        match="adaptive_development_freeze_integrity_changed",
    ):
        calibrate_adaptive_first_release(mutated_freeze, calibration)


def test_threshold_dependent_trajectory_fields_are_closed_and_prefix_is_monotone() -> None:
    contexts = _contexts()
    row = _trajectory(
        question_index=20,
        split="development",
        contexts=contexts,
        reference="supported",
    )
    payload = row.visible.arms[0].model_dump(mode="json")
    payload["threshold"] = 0.4
    with pytest.raises(ValidationError, match="extra_forbidden"):
        type(row.visible.arms[0]).model_validate(payload)

    states = list(row.visible.arms[0].states)
    third = freeze_adaptive_preselection_state(
        prefix_index=2,
        audit_prefix_item_ids=["different-first", "different-second"],
        audit_prefix_cost_minutes=6,
        scheduler_state_sha256="9" * 64,
        evidence_graph_sha256="8" * 64,
        synthesis_sha256="7" * 64,
        non_calibration_assessment_sha256="6" * 64,
        non_calibration_gates_passed=True,
        non_calibration_blocking_reasons=[],
        claim_decision="supported",
        score_features={"influence": 0.1, "item_cell_rate_ucl": 0.2},
    )
    states.append(third)
    with pytest.raises(ValidationError, match="audit_prefix_not_monotone"):
        freeze_adaptive_policy_arm_trajectory(
            policy_arm_id=contexts[0].policy_arm_id,
            policy_context_sha256=contexts[0].policy_context_sha256,
            states=states,
            terminal_reason="all_items_resolved",
            terminal_candidates=[
                AdaptiveTerminalAuditCandidate(
                    item_id=item_id,
                    eligible=True,
                    estimated_cost_minutes=3.0,
                    source_candidate_sha256=hash_canonical({"item_id": item_id}),
                )
                for item_id in states[-1].audit_prefix_item_ids
            ],
            terminal_source_candidate_input_sha256=hash_canonical(states[-1].audit_prefix_item_ids),
            terminal_remaining_budget_minutes=4.0,
        )


def test_early_truncation_cannot_claim_terminal_with_unresolved_feasible_action() -> None:
    contexts = _contexts()
    state = _state(
        question_index=60,
        arm_index=0,
        prefix=0,
        passed=False,
        decision="supported",
        feature=0.2,
    )
    pending = AdaptiveTerminalAuditCandidate(
        item_id="pending-item",
        eligible=True,
        estimated_cost_minutes=3.0,
        source_candidate_sha256="a" * 64,
    )
    with pytest.raises(
        ValidationError,
        match="adaptive_terminal_all_resolved_has_unresolved_items",
    ):
        freeze_adaptive_policy_arm_trajectory(
            policy_arm_id=contexts[0].policy_arm_id,
            policy_context_sha256=contexts[0].policy_context_sha256,
            states=[state],
            terminal_reason="all_items_resolved",
            terminal_candidates=[pending],
            terminal_source_candidate_input_sha256="b" * 64,
            terminal_remaining_budget_minutes=10.0,
        )


def test_threshold_family_candidate_model_context_lineage_is_recomputed() -> None:
    _, _, bundle = _bundle()
    payload = bundle.development_freeze.model_dump(mode="json")
    candidate = payload["threshold_family"]["candidates"][0]
    candidate["score_model_sha256"] = "f" * 64
    candidate["candidate_sha256"] = hash_canonical(
        {key: value for key, value in candidate.items() if key != "candidate_sha256"}
    )
    family = payload["threshold_family"]
    family["family_sha256"] = hash_canonical(
        {key: value for key, value in family.items() if key != "family_sha256"}
    )
    plan = payload["calibration_plan"]
    plan["threshold_family_sha256"] = family["family_sha256"]
    plan["plan_sha256"] = hash_canonical(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    payload["development_freeze_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "development_freeze_sha256"}
    )
    with pytest.raises(
        ValidationError,
        match="adaptive_threshold_candidate_model_mismatch",
    ):
        AdaptiveDevelopmentFreeze.model_validate(payload)


def test_calibration_rejects_rehashed_outcome_and_input_tampering() -> None:
    _, _, bundle = _bundle()
    payload = bundle.model_dump(mode="json")
    outcome = next(row for row in payload["candidates"][0]["outcomes"] if row["accepted"])
    outcome["error"] = not outcome["error"]
    candidate_row = payload["candidates"][0]
    candidate_row["errors"] += 1
    candidate_row["empirical_risk"] = candidate_row["errors"] / candidate_row["accepted"]
    payload["bundle_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "bundle_sha256"}
    )
    with pytest.raises(ValidationError, match="adaptive_calibration_replay_outcome_mismatch"):
        AdaptiveCalibrationBundle.model_validate(payload)

    payload = bundle.model_dump(mode="json")
    payload["calibration_input_sha256"] = "f" * 64
    payload["bundle_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "bundle_sha256"}
    )
    with pytest.raises(ValidationError, match="adaptive_calibration_input_hash_mismatch"):
        AdaptiveCalibrationBundle.model_validate(payload)


def test_prospective_whole_prefix_allows_repeated_looks_only_until_first_release() -> None:
    contexts, _, bundle = _bundle()
    selected = bundle.selected
    assert selected is not None
    context = next(row for row in contexts if row.policy_arm_id == selected.candidate.policy_arm_id)
    prospective = _trajectory(
        question_index=30,
        split="test",
        contexts=contexts,
        reference="supported",
    )

    prefix_zero = freeze_prospective_adaptive_candidate(
        question_id=prospective.visible.question_id,
        population_id=prospective.visible.population_id,
        domain=prospective.visible.domain,
        policy_arm_id=context.policy_arm_id,
        policy_context_sha256=context.policy_context_sha256,
        corpus=prospective.visible.corpus,
        observed_states=_unscored_prefix(prospective, arm_id=context.policy_arm_id, count=1),
    )
    first = assess_adaptive_release_candidate(prefix_zero, bundle)
    assert first.status == "abstained"
    assert first.reason == "noncalibration_gate_blocked"

    prefix_one = freeze_prospective_adaptive_candidate(
        question_id=prospective.visible.question_id,
        population_id=prospective.visible.population_id,
        domain=prospective.visible.domain,
        policy_arm_id=context.policy_arm_id,
        policy_context_sha256=context.policy_context_sha256,
        corpus=prospective.visible.corpus,
        observed_states=_unscored_prefix(prospective, arm_id=context.policy_arm_id, count=2),
    )
    second = assess_adaptive_release_candidate(prefix_one, bundle)
    assert second.status == "released"
    assert second.prefix_index == 1

    states = _unscored_prefix(prospective, arm_id=context.policy_arm_id, count=2)
    qualifying_zero = freeze_adaptive_preselection_state(
        prefix_index=0,
        audit_prefix_item_ids=[],
        audit_prefix_cost_minutes=0,
        scheduler_state_sha256=states[0].scheduler_state_sha256,
        evidence_graph_sha256=states[0].evidence_graph_sha256,
        synthesis_sha256=states[0].synthesis_sha256,
        non_calibration_assessment_sha256=states[0].non_calibration_assessment_sha256,
        non_calibration_gates_passed=True,
        non_calibration_blocking_reasons=[],
        claim_decision=states[0].claim_decision,
        score_features=states[0].score_features,
    )
    continued = freeze_prospective_adaptive_candidate(
        question_id=prospective.visible.question_id,
        population_id=prospective.visible.population_id,
        domain=prospective.visible.domain,
        policy_arm_id=context.policy_arm_id,
        policy_context_sha256=context.policy_context_sha256,
        corpus=prospective.visible.corpus,
        observed_states=[qualifying_zero, states[1]],
    )
    with pytest.raises(
        AdaptiveCalibrationError,
        match="continued_after_first_release",
    ):
        assess_adaptive_release_candidate(continued, bundle)


def test_prospective_overlap_uses_complete_manifest_not_matching_estimates() -> None:
    contexts, calibration, bundle = _bundle()
    selected = bundle.selected
    assert selected is not None
    context = next(row for row in contexts if row.policy_arm_id == selected.candidate.policy_arm_id)
    hidden_excluded_publication = calibration[0].visible.corpus.publication_ids[0]
    prospective = _trajectory(
        question_index=31,
        split="test",
        contexts=contexts,
        reference="supported",
        source_manifest_extra=hidden_excluded_publication,
    )
    candidate = freeze_prospective_adaptive_candidate(
        question_id=prospective.visible.question_id,
        population_id=prospective.visible.population_id,
        domain=prospective.visible.domain,
        policy_arm_id=context.policy_arm_id,
        policy_context_sha256=context.policy_context_sha256,
        corpus=prospective.visible.corpus,
        observed_states=_unscored_prefix(prospective, arm_id=context.policy_arm_id, count=1),
    )
    with pytest.raises(
        AdaptiveCalibrationError,
        match="prospective_complete_corpus_publication_overlap",
    ):
        assess_adaptive_release_candidate(candidate, bundle)


def test_policy_context_hash_binds_budget_stop_rule_and_exact_configs() -> None:
    context = _contexts()[0]
    for field, value in (
        ("budget_minutes", 11.0),
        ("allocation_policy", {"name": "different"}),
        ("release_config", {"version": "changed"}),
        ("audit_config", {"guard": "changed"}),
        ("target_semantics", {"loss": "changed"}),
        ("corpus_protocol_context", {"cutoff_rule": "changed"}),
    ):
        payload = context.model_dump(mode="json")
        payload[field] = value
        with pytest.raises(ValidationError, match="adaptive_policy_context_hash_mismatch"):
            AdaptivePolicyContext.model_validate(payload)


def test_nested_mutation_is_detected_by_bundle_integrity_reparse() -> None:
    _, _, bundle = _bundle()
    mutated = deepcopy(bundle)
    mutated.scored_calibration_trajectories[0].visible.arms[0].states[0].score_features[
        "influence"
    ] = 0.999
    with pytest.raises(AdaptiveCalibrationError, match="integrity_changed"):
        assess_adaptive_release_candidate(
            freeze_prospective_adaptive_candidate(
                question_id="prospective-new",
                population_id=bundle.population_id,
                domain="medicine",
                policy_arm_id=bundle.selected.candidate.policy_arm_id,  # type: ignore[union-attr]
                policy_context_sha256=(
                    bundle.selected.candidate.policy_context_sha256  # type: ignore[union-attr]
                ),
                corpus=freeze_complete_corpus_identity(
                    corpus_id="new",
                    corpus_source_sha256="f" * 64,
                    corpus_cutoff="2025-12-31",
                    publication_ids=["new-publication"],
                    source_manifest_sha256="e" * 64,
                ),
                observed_states=[
                    freeze_adaptive_preselection_state(
                        prefix_index=0,
                        audit_prefix_item_ids=[],
                        audit_prefix_cost_minutes=0,
                        scheduler_state_sha256="1" * 64,
                        evidence_graph_sha256="2" * 64,
                        synthesis_sha256="3" * 64,
                        non_calibration_assessment_sha256="4" * 64,
                        non_calibration_gates_passed=False,
                        non_calibration_blocking_reasons=["blocked"],
                        claim_decision="supported",
                        score_features={"influence": 0.1, "item_cell_rate_ucl": 0.2},
                    )
                ],
            ),
            mutated,
        )


def test_nested_mutation_is_detected_in_prospective_candidate() -> None:
    contexts, _, bundle = _bundle()
    selected = bundle.selected
    assert selected is not None
    context = next(row for row in contexts if row.policy_arm_id == selected.candidate.policy_arm_id)
    prospective = _trajectory(
        question_index=50,
        split="test",
        contexts=contexts,
        reference="supported",
    )
    candidate = freeze_prospective_adaptive_candidate(
        question_id=prospective.visible.question_id,
        population_id=prospective.visible.population_id,
        domain=prospective.visible.domain,
        policy_arm_id=context.policy_arm_id,
        policy_context_sha256=context.policy_context_sha256,
        corpus=prospective.visible.corpus,
        observed_states=_unscored_prefix(
            prospective,
            arm_id=context.policy_arm_id,
            count=1,
        ),
    )
    candidate.observed_states[0].score_features["influence"] = 0.999
    with pytest.raises(
        AdaptiveCalibrationError,
        match="prospective_adaptive_candidate_integrity_changed",
    ):
        assess_adaptive_release_candidate(candidate, bundle)
