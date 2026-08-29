from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from literature_multiverse.decisive_claim_evaluation_v1 import (
    PRIMARY_POLICY_ARM_ID,
    DecisiveClaimEvaluationResultV1,
    build_decisive_mechanics_fixture_v1,
    freeze_decisive_evaluation_config_v1,
)
from literature_multiverse.decisive_claim_evaluation_v2 import (
    POINT_VERSION,
    CompiledPolicyPointV2,
    DecisiveClaimEvaluationFrontiersV2,
    DecisiveClaimEvaluationV2Error,
    DecisiveFrontierConfigV2,
    DecisiveFrontierSourceAnchorV2,
    _cost_comparisons,
    _cost_rows,
    _domain_metrics,
    _fixed_error_comparisons,
    _fixed_error_rows,
    _fixed_error_selections,
    _marginal_bootstrap,
    _metrics,
    _paired_bootstrap,
    _paired_deltas,
    _worst_domain,
    build_decisive_claim_evaluation_frontiers_v2,
    freeze_decisive_frontier_config_v2,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.question_evaluation import ReferenceClaimVerdictValue

ROOT = Path(__file__).resolve().parents[1]


def _outcome(
    *,
    question_index: int,
    arm: str,
    budget: float,
    minutes: float,
    released: bool,
    correct: bool,
) -> Any:
    from literature_multiverse.decisive_claim_evaluation_v1 import ScoredQuestionOutcomeV1

    question_id = f"question-{question_index:02d}"
    reference = ReferenceClaimVerdictValue.SUPPORTED.value
    predicted = reference if correct else ReferenceClaimVerdictValue.CONTRADICTED.value
    payload: dict[str, Any] = {
        "question_id": question_id,
        "claim_id": f"claim-{question_index:02d}",
        "domain": "domain-a" if question_index < 10 else "domain-b",
        "policy_arm_id": arm,
        "budget_minutes": float(budget),
        "policy_question_freeze_sha256": hash_canonical(
            {"question": question_id, "arm": arm, "budget": budget}
        ),
        "reference_verdict_sha256": hash_canonical({"reference": question_id}),
        "reference_verdict": reference,
        "predicted_classification": predicted,
        "predicted_condition_set_artifact_sha256": None,
        "reference_condition_set_artifact_sha256": None,
        "released": released,
        "abstained": not released,
        "classification_exact_match": correct,
        "condition_set_exact_match": None,
        "decision_exact_match": correct,
        "released_claim_error": released and not correct,
        "correct_release": released and correct,
        "appropriate_abstention": not released and not correct,
        "missed_correct_decision_abstention": not released and correct,
        "realized_minutes": float(minutes),
        "selected_actions": int(minutes > 0),
        "resolved_actions": int(minutes > 0),
        "active_action_at_deadline": False,
    }
    return ScoredQuestionOutcomeV1.model_validate(
        {**payload, "outcome_sha256": hash_canonical(payload)}
    )


def _outcomes(*, arm: str, budget: float, minutes: float, released: int, errors: int) -> list[Any]:
    return [
        _outcome(
            question_index=index,
            arm=arm,
            budget=budget,
            minutes=minutes,
            released=index < released,
            correct=not (released - errors <= index < released),
        )
        for index in range(20)
    ]


def _point(
    *,
    arm: str,
    budget: float,
    minutes: float,
    released: int,
    errors: int,
    authority: bool = False,
) -> tuple[CompiledPolicyPointV2, list[Any]]:
    outcomes = _outcomes(
        arm=arm,
        budget=budget,
        minutes=minutes,
        released=released,
        errors=errors,
    )
    domains = _domain_metrics(outcomes)
    payload: dict[str, Any] = {
        "point_version": POINT_VERSION,
        "policy_arm_id": arm,
        "nominal_budget_minutes_per_question": float(budget),
        "source_scored_population_sha256": hash_canonical({"scored": arm, "budget": budget}),
        "source_frozen_population_sha256": hash_canonical({"frozen": arm, "budget": budget}),
        "question_ids": [row.question_id for row in outcomes],
        "metrics": _metrics(outcomes),
        "domain_metrics": domains,
        "worst_domain_metrics": _worst_domain(domains),
        "question_clustered_uncertainty": _marginal_bootstrap(outcomes, draws=100, seed=17),
        "calibration_anchor_sha256": "a" * 64 if authority else None,
        "typed_calibration_anchor_present": authority,
        "released_error_claim_authority": authority,
    }
    point = CompiledPolicyPointV2.model_validate(
        {**payload, "point_sha256": hash_canonical(payload)}
    )
    return point, outcomes


def _frontier_inputs() -> tuple[Any, list[CompiledPolicyPointV2], dict[str, list[Any]]]:
    config = freeze_decisive_frontier_config_v2(
        common_realized_person_minutes_per_question_cutoffs=(3.0, 5.0, 8.0, 10.0),
        released_error_ceiling=0.2,
        bootstrap_draws=100,
    )
    specifications = (
        (PRIMARY_POLICY_ARM_ID, 5.0, 4.0, 10, 1),
        (PRIMARY_POLICY_ARM_ID, 10.0, 8.0, 15, 4),
        ("random_static", 5.0, 3.0, 8, 1),
        ("random_static", 10.0, 9.0, 14, 3),
    )
    points: list[CompiledPolicyPointV2] = []
    outcomes: dict[str, list[Any]] = {}
    for arm, budget, minutes, released, errors in specifications:
        point, rows = _point(
            arm=arm,
            budget=budget,
            minutes=minutes,
            released=released,
            errors=errors,
        )
        points.append(point)
        outcomes[point.point_sha256] = rows
    points.sort(key=lambda row: (row.policy_arm_id, row.nominal_budget_minutes_per_question))
    return config, points, outcomes


def test_config_is_prespecified_self_hashed_and_strict() -> None:
    config = freeze_decisive_frontier_config_v2(
        common_realized_person_minutes_per_question_cutoffs=(5, 10),
        released_error_ceiling=0.1,
        bootstrap_draws=100,
    )
    assert config.common_realized_person_minutes_per_question_cutoffs == [5.0, 10.0]
    payload = config.model_dump(mode="json")
    payload["released_error_ceiling"] = 0.2
    with pytest.raises(ValidationError, match="self_hash_mismatch"):
        type(config).model_validate(payload)


def test_config_rejects_negative_bootstrap_seed() -> None:
    with pytest.raises(ValidationError):
        freeze_decisive_frontier_config_v2(
            common_realized_person_minutes_per_question_cutoffs=(5, 10),
            released_error_ceiling=0.1,
            bootstrap_draws=100,
            bootstrap_seed=-1,
        )


def test_frontier_result_cannot_assert_authority_without_supporting_points() -> None:
    config = freeze_decisive_frontier_config_v2(
        common_realized_person_minutes_per_question_cutoffs=(5.0,),
        released_error_ceiling=0.1,
        bootstrap_draws=100,
    )
    anchor_payload: dict[str, Any] = {
        "anchor_version": "decisive-frontier-source-anchor-v2",
        "source_result_version": "decisive-claim-evaluation-result-v1",
        "source_result_sha256": "1" * 64,
        "source_policy_freeze_sha256": "2" * 64,
        "source_component_sha256": "3" * 64,
        "split_manifest_sha256": "4" * 64,
        "trajectory_bundle_sha256": "5" * 64,
        "opened_label_membership_sha256": "6" * 64,
        "scored_population_membership_sha256": "7" * 64,
        "pipeline_sha256": "8" * 64,
        "question_population_ids": ["population"],
        "question_population_membership_sha256": "9" * 64,
        "adjudication_protocol_sha256": "a" * 64,
        "evidence_kind": "real_expert_adjudicated",
        "evaluation_question_ids": ["question"],
        "domains": ["domain"],
        "complete_question_inputs_structurally_replayed_from_v1": True,
        "identical_question_population_across_policies": True,
        "identical_pipeline_across_policies": True,
        "simulation_or_fixture_input": False,
    }
    anchor = DecisiveFrontierSourceAnchorV2.model_validate(
        {**anchor_payload, "anchor_sha256": hash_canonical(anchor_payload)}
    )
    result_payload: dict[str, Any] = {
        "result_version": "decisive-claim-evaluation-frontiers-v2",
        "evaluator_component_sha256": "b" * 64,
        "config": config,
        "source_anchor": anchor,
        "calibration_anchors": [],
        "compiled_policy_points": [],
        "realized_cost_frontier": [],
        "realized_cost_paired_comparisons": [],
        "fixed_error_frontier": [],
        "fixed_error_policy_selections": [],
        "fixed_error_paired_comparisons": [],
        "metric_definitions": {},
        "input_labels_or_private_files_opened_by_v2": False,
        "policy_trajectories_rerun_by_v2": False,
        "simulation_or_fixture_inputs_accepted": False,
        "equal_nominal_deadline_misreported_as_equal_realized_cost": False,
        "realized_cost_frontier_claim_authority": True,
        "fixed_error_frontier_claim_authority": True,
        "scientific_claim_eligible": True,
        "claim_release_authority": False,
        "causal_or_prospective_authority": False,
        "small_sample_or_finite_sample_authority": False,
        "authority_blockers": [],
    }
    with pytest.raises(ValidationError, match="points_not_canonical"):
        DecisiveClaimEvaluationFrontiersV2.model_validate(
            {**result_payload, "result_sha256": hash_canonical(result_payload)}
        )


def test_frontier_result_replays_supporting_rosters_before_accepting_authority() -> None:
    config, points, outcomes = _frontier_inputs()
    anchor_payload: dict[str, Any] = {
        "anchor_version": "decisive-frontier-source-anchor-v2",
        "source_result_version": "decisive-claim-evaluation-result-v1",
        "source_result_sha256": "1" * 64,
        "source_policy_freeze_sha256": "2" * 64,
        "source_component_sha256": "3" * 64,
        "split_manifest_sha256": "4" * 64,
        "trajectory_bundle_sha256": "5" * 64,
        "opened_label_membership_sha256": "6" * 64,
        "scored_population_membership_sha256": "7" * 64,
        "pipeline_sha256": "8" * 64,
        "question_population_ids": ["population"],
        "question_population_membership_sha256": "9" * 64,
        "adjudication_protocol_sha256": "a" * 64,
        "evidence_kind": "real_expert_adjudicated",
        "evaluation_question_ids": points[0].question_ids,
        "domains": ["domain-a", "domain-b"],
        "complete_question_inputs_structurally_replayed_from_v1": True,
        "identical_question_population_across_policies": True,
        "identical_pipeline_across_policies": True,
        "simulation_or_fixture_input": False,
    }
    anchor = DecisiveFrontierSourceAnchorV2.model_validate(
        {**anchor_payload, "anchor_sha256": hash_canonical(anchor_payload)}
    )
    cost_rows = _cost_rows(points=points, config=config)
    cost_comparisons = _cost_comparisons(
        rows=cost_rows,
        points=points,
        outcomes_by_sha=outcomes,
        config=config,
    )
    fixed_rows = _fixed_error_rows(points=points, config=config)
    selections = _fixed_error_selections(
        points=points,
        rows=fixed_rows,
        config=config,
    )
    fixed_comparisons = _fixed_error_comparisons(
        selections=selections,
        points=points,
        outcomes_by_sha=outcomes,
        config=config,
    )
    result_payload: dict[str, Any] = {
        "result_version": "decisive-claim-evaluation-frontiers-v2",
        "evaluator_component_sha256": "b" * 64,
        "config": config,
        "source_anchor": anchor,
        "calibration_anchors": [],
        "compiled_policy_points": points,
        "realized_cost_frontier": cost_rows,
        "realized_cost_paired_comparisons": cost_comparisons,
        "fixed_error_frontier": fixed_rows,
        "fixed_error_policy_selections": selections,
        "fixed_error_paired_comparisons": fixed_comparisons,
        "metric_definitions": {},
        "input_labels_or_private_files_opened_by_v2": False,
        "policy_trajectories_rerun_by_v2": False,
        "simulation_or_fixture_inputs_accepted": False,
        "equal_nominal_deadline_misreported_as_equal_realized_cost": False,
        "realized_cost_frontier_claim_authority": False,
        "fixed_error_frontier_claim_authority": False,
        "scientific_claim_eligible": False,
        "claim_release_authority": False,
        "causal_or_prospective_authority": False,
        "small_sample_or_finite_sample_authority": False,
        "authority_blockers": sorted(
            [
                "common_realized_cost_frontier_comparison_roster_incomplete",
                "common_realized_cost_frontier_contains_uncalibrated_comparisons",
                "fixed_error_frontier_contains_descriptive_or_uncalibrated_comparisons",
                "real_typed_complete_question_calibration_bundles_missing",
            ]
        ),
    }
    result = DecisiveClaimEvaluationFrontiersV2.model_validate(
        {**result_payload, "result_sha256": hash_canonical(result_payload)}
    )
    forged = result.model_dump(mode="json", exclude={"result_sha256"})
    forged["realized_cost_frontier_claim_authority"] = True
    with pytest.raises(ValidationError, match="frontier_authority_mismatch"):
        DecisiveClaimEvaluationFrontiersV2.model_validate(
            {**forged, "result_sha256": hash_canonical(forged)}
        )


def test_freeze_config_cli_writes_replayable_contract(tmp_path: Path) -> None:
    output = tmp_path / "frontier-config.json"
    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/run_decisive_claim_evaluation_v2.py"),
            "freeze-config",
            "--realized-minutes-per-question-cutoff",
            "5",
            "--realized-minutes-per-question-cutoff",
            "10",
            "--released-error-ceiling",
            "0.1",
            "--bootstrap-draws",
            "100",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    config = DecisiveFrontierConfigV2.model_validate(json.loads(output.read_text(encoding="utf-8")))
    assert summary["config_sha256"] == config.config_sha256
    assert summary["claim_release_authority"] is False


def test_common_realized_cost_rows_select_without_using_outcomes() -> None:
    config, points, outcomes = _frontier_inputs()
    rows = _cost_rows(points=points, config=config)
    by_key = {
        (row.policy_arm_id, row.common_realized_person_minutes_per_question_cutoff): row
        for row in rows
    }
    assert by_key[(PRIMARY_POLICY_ARM_ID, 3.0)].point_available is False
    assert by_key[(PRIMARY_POLICY_ARM_ID, 5.0)].realized_person_minutes_per_question == 4.0
    assert by_key[(PRIMARY_POLICY_ARM_ID, 8.0)].selected_nominal_budget_minutes_per_question == 10.0
    assert by_key[("random_static", 8.0)].selected_nominal_budget_minutes_per_question == 5.0
    assert by_key[("random_static", 10.0)].realized_person_minutes_per_question == 9.0
    assert all(row.common_ceiling_comparison_not_exact_spend_equality for row in rows)

    comparisons = _cost_comparisons(
        rows=rows,
        points=points,
        outcomes_by_sha=outcomes,
        config=config,
    )
    assert comparisons
    at_five = next(row for row in comparisons if row.common_cutoff_or_error_ceiling == 5.0)
    assert at_five.exact_realized_spend_match is False
    assert at_five.released_error_claim_authority is False
    assert {row.metric for row in at_five.paired_question_clustered_uncertainty.intervals} == set(
        at_five.primary_minus_baseline_point_deltas
    )


def test_compiled_point_authority_cannot_be_added_without_calibration_anchor() -> None:
    point, _ = _point(
        arm=PRIMARY_POLICY_ARM_ID,
        budget=5.0,
        minutes=4.0,
        released=10,
        errors=1,
    )
    payload = point.model_dump(mode="json")
    payload["released_error_claim_authority"] = True
    payload["point_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "point_sha256"}
    )
    with pytest.raises(ValidationError, match="authority_without_calibration"):
        CompiledPolicyPointV2.model_validate(payload)


def test_fixed_error_frontier_is_descriptive_without_typed_calibration() -> None:
    config, points, outcomes = _frontier_inputs()
    rows = _fixed_error_rows(points=points, config=config)
    by_key = {(row.policy_arm_id, row.nominal_budget_minutes_per_question): row for row in rows}
    assert by_key[(PRIMARY_POLICY_ARM_ID, 5.0)].observed_ceiling_status == "meets"
    assert by_key[(PRIMARY_POLICY_ARM_ID, 10.0)].observed_ceiling_status == "exceeds"
    assert by_key[(PRIMARY_POLICY_ARM_ID, 5.0)].typed_calibration_ceiling_eligible is False

    selections = _fixed_error_selections(points=points, rows=rows, config=config)
    primary = next(row for row in selections if row.policy_arm_id == PRIMARY_POLICY_ARM_ID)
    assert primary.selection_basis == "observed_evaluation_error_descriptive"
    assert primary.release_coverage == 0.5
    assert primary.released_error_claim_authority is False

    comparisons = _fixed_error_comparisons(
        selections=selections,
        points=points,
        outcomes_by_sha=outcomes,
        config=config,
    )
    assert len(comparisons) == 1
    assert comparisons[0].released_error_claim_authority is False


def test_fixed_error_authority_does_not_fall_back_after_preselected_budget_fails() -> None:
    config = freeze_decisive_frontier_config_v2(
        common_realized_person_minutes_per_question_cutoffs=(10.0,),
        released_error_ceiling=0.2,
        bootstrap_draws=100,
    )
    low, _ = _point(
        arm=PRIMARY_POLICY_ARM_ID,
        budget=5,
        minutes=4,
        released=10,
        errors=1,
        authority=True,
    )
    high, _ = _point(
        arm=PRIMARY_POLICY_ARM_ID,
        budget=10,
        minutes=8,
        released=15,
        errors=4,
        authority=True,
    )
    points = sorted(
        [low, high],
        key=lambda row: (row.policy_arm_id, row.nominal_budget_minutes_per_question),
    )
    rows = _fixed_error_rows(points=points, config=config)
    selections = _fixed_error_selections(points=points, rows=rows, config=config)
    assert len(selections) == 1
    selection = selections[0]
    assert selection.selected_point_sha256 == low.point_sha256
    assert selection.selection_basis == "observed_evaluation_error_descriptive"
    assert selection.released_error_claim_authority is False


def test_exact_calibrated_points_propagate_comparison_authority() -> None:
    config = freeze_decisive_frontier_config_v2(
        common_realized_person_minutes_per_question_cutoffs=(5.0,),
        released_error_ceiling=0.2,
        bootstrap_draws=100,
    )
    primary, primary_outcomes = _point(
        arm=PRIMARY_POLICY_ARM_ID,
        budget=5.0,
        minutes=4.0,
        released=10,
        errors=1,
        authority=True,
    )
    baseline, baseline_outcomes = _point(
        arm="random_static",
        budget=5.0,
        minutes=3.0,
        released=8,
        errors=1,
        authority=True,
    )
    points = sorted(
        [primary, baseline],
        key=lambda row: (row.policy_arm_id, row.nominal_budget_minutes_per_question),
    )
    outcomes = {
        primary.point_sha256: primary_outcomes,
        baseline.point_sha256: baseline_outcomes,
    }
    cost_rows = _cost_rows(points=points, config=config)
    cost_comparisons = _cost_comparisons(
        rows=cost_rows,
        points=points,
        outcomes_by_sha=outcomes,
        config=config,
    )
    assert len(cost_comparisons) == 1
    assert cost_comparisons[0].released_error_claim_authority is True

    fixed_rows = _fixed_error_rows(points=points, config=config)
    selections = _fixed_error_selections(points=points, rows=fixed_rows, config=config)
    assert all(row.selection_basis == "typed_calibration_selected_point" for row in selections)
    fixed_comparisons = _fixed_error_comparisons(
        selections=selections,
        points=points,
        outcomes_by_sha=outcomes,
        config=config,
    )
    assert len(fixed_comparisons) == 1
    assert fixed_comparisons[0].released_error_claim_authority is True


def test_paired_question_cluster_bootstrap_is_deterministic_and_reports_worst_domain() -> None:
    primary = _outcomes(
        arm=PRIMARY_POLICY_ARM_ID,
        budget=5,
        minutes=4,
        released=10,
        errors=1,
    )
    baseline = _outcomes(
        arm="random_static",
        budget=5,
        minutes=3,
        released=8,
        errors=1,
    )
    first = _paired_bootstrap(primary, baseline, draws=100, seed=123)
    second = _paired_bootstrap(primary, baseline, draws=100, seed=123)
    assert first == second
    names = {row.metric for row in first.intervals}
    assert "worst_domain_release_coverage_delta" in names
    assert "worst_domain_released_claim_error_delta" in names
    point = _paired_deltas(primary, baseline)
    assert point["release_coverage_delta"] == pytest.approx(0.1)


def test_public_builder_rejects_planted_fixture_instead_of_upgrading_it(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fixture"
    build_decisive_mechanics_fixture_v1(
        output_root=output,
        repository_root=ROOT,
        config=freeze_decisive_evaluation_config_v1(
            budgets_minutes_per_question=(6.0,), bootstrap_draws=100
        ),
    )
    raw = json.loads((output / "evaluation-result.json").read_text(encoding="utf-8"))
    source = DecisiveClaimEvaluationResultV1.model_validate(raw)
    config = freeze_decisive_frontier_config_v2(
        common_realized_person_minutes_per_question_cutoffs=(6.0,),
        released_error_ceiling=0.1,
        bootstrap_draws=100,
    )
    with pytest.raises(
        DecisiveClaimEvaluationV2Error,
        match="simulation_or_fixture_input_rejected",
    ):
        build_decisive_claim_evaluation_frontiers_v2(
            source_result=source,
            config=config,
            repository_root=ROOT,
        )
