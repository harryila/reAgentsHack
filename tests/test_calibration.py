from __future__ import annotations

import math

import pytest

from literature_multiverse.calibration import (
    CalibratedReleasePolicy,
    CalibrationContractError,
    FrozenCalibrationBundle,
    ReleaseCandidate,
    RiskExample,
    assess_release_candidate,
    calibrate_release_policy,
    calibration_artifact,
    clopper_pearson_interval,
    clopper_pearson_upper,
    evaluate_frozen_calibration_bundle,
    evaluate_release_policy,
    fit_logistic_risk_model,
    freeze_calibration_bundle,
    score_examples,
    validate_split_integrity,
)
from literature_multiverse.calibration_simulation import (
    simulate_questions,
    simulate_replicate,
    summarize_replicates,
)


def _examples() -> list[RiskExample]:
    rows: list[RiskExample] = []
    specifications = {
        "development": (12, 12),
        "calibration": (25, 5),
        "test": (15, 5),
    }
    counter = 0
    for split, (supported, unsupported) in specifications.items():
        for index in range(supported + unsupported):
            is_unsupported = index >= supported
            uncertainty = 0.08 + (index % 5) * 0.01
            if is_unsupported:
                uncertainty = 0.82 + (index % 5) * 0.02
            rows.append(
                RiskExample(
                    question_id=f"{split}-q-{index:03d}",
                    split=split,
                    population_id="sim-v1",
                    domain="domain-a" if counter % 2 == 0 else "domain-b",
                    pipeline_sha256="a" * 64,
                    paper_ids=[f"paper-{counter:04d}"],
                    features={
                        "bootstrap_instability": uncertainty,
                        "ungrounded_fraction": uncertainty / 2.0,
                    },
                    unsupported_claim=is_unsupported,
                    label_source="simulation",
                )
            )
            counter += 1
    return rows


def _calibrated_bundle(rows: list[RiskExample]) -> FrozenCalibrationBundle:
    development_calibration = [row for row in rows if row.split != "test"]
    model = fit_logistic_risk_model(development_calibration, seed=7)
    calibration_scores = score_examples(
        [row for row in development_calibration if row.split == "calibration"], model
    )
    supported_scores = [
        row.score for row in calibration_scores if not row.example.unsupported_claim
    ]
    unsupported_scores = [
        row.score for row in calibration_scores if row.example.unsupported_claim
    ]
    threshold = (max(supported_scores) + min(unsupported_scores)) / 2.0
    return freeze_calibration_bundle(
        development_calibration,
        alpha=0.20,
        delta=0.05,
        seed=7,
        candidate_thresholds=[threshold],
    )


def _release_candidate(row: RiskExample) -> ReleaseCandidate:
    return ReleaseCandidate(
        question_id=row.question_id,
        population_id=row.population_id,
        domain=row.domain,
        pipeline_sha256=row.pipeline_sha256,
        paper_ids=row.paper_ids,
        features=row.features,
    )


def test_clopper_pearson_known_zero_error_bound() -> None:
    upper = clopper_pearson_upper(0, 59, delta=0.05)
    assert upper < 0.05
    assert upper == pytest.approx(1.0 - 0.05 ** (1.0 / 59.0))

    lower, descriptive_upper = clopper_pearson_interval(0, 59)
    assert lower == 0.0
    assert 0.0 < descriptive_upper < 0.07


def test_split_integrity_rejects_paper_leakage() -> None:
    rows = _examples()
    calibration_index = next(index for index, row in enumerate(rows) if row.split == "calibration")
    rows[calibration_index] = rows[calibration_index].model_copy(
        update={"paper_ids": rows[0].paper_ids}
    )
    with pytest.raises(CalibrationContractError, match="paper_crosses_risk_split"):
        validate_split_integrity(rows)


def test_split_integrity_rejects_shared_paper_between_questions_in_one_split() -> None:
    rows = _examples()
    rows[1] = rows[1].model_copy(update={"paper_ids": rows[0].paper_ids})

    with pytest.raises(CalibrationContractError, match="paper_shared_between_question_units"):
        validate_split_integrity(rows, require_disjoint_question_papers=True)


def test_split_integrity_rejects_simulation_to_real_calibration() -> None:
    rows = _examples()
    rows[-1] = rows[-1].model_copy(update={"population_id": "real-v1"})
    with pytest.raises(CalibrationContractError, match="population_changed"):
        validate_split_integrity(rows)


def test_frozen_benchmark_annotations_are_valid_real_label_sources() -> None:
    row = _examples()[0].model_copy(update={"label_source": "benchmark_annotation"})

    assert RiskExample.model_validate(row.model_dump()).label_source == "benchmark_annotation"


def test_fit_calibrate_and_evaluate_question_level_policy() -> None:
    rows = _examples()
    model = fit_logistic_risk_model(rows, seed=7)
    calibration_scores = score_examples([row for row in rows if row.split == "calibration"], model)
    supported_scores = [
        row.score for row in calibration_scores if not row.example.unsupported_claim
    ]
    unsupported_scores = [row.score for row in calibration_scores if row.example.unsupported_claim]
    threshold = (max(supported_scores) + min(unsupported_scores)) / 2.0
    assert max(supported_scores) < min(unsupported_scores)

    policy = calibrate_release_policy(
        rows,
        model,
        alpha=0.20,
        delta=0.05,
        candidate_thresholds=[threshold],
    )
    assert policy.status == "calibrated"
    assert policy.selected is not None
    assert policy.selected.accepted == 25
    assert policy.selected.errors == 0
    assert policy.selected.simultaneous_upper_risk is not None
    assert policy.selected.simultaneous_upper_risk < 0.20

    evaluation = evaluate_release_policy(rows, model, policy)
    assert evaluation.total == 20
    assert evaluation.accepted == 15
    assert evaluation.errors == 0
    assert evaluation.coverage == pytest.approx(0.75)
    assert evaluation.empirical_risk == 0.0
    assert set(evaluation.by_domain) == {"domain-a", "domain-b"}

    artifact = calibration_artifact(
        examples=rows,
        model=model,
        policy=policy,
        evaluation=evaluation,
    )
    assert artifact["label_source"] == "simulation"
    assert artifact["policy"]["status"] == "calibrated"
    assert artifact["test_risk_coverage_curve"][-1]["accepted"] == 20


def test_default_candidate_family_is_learned_from_development_not_calibration() -> None:
    rows = _examples()
    model = fit_logistic_risk_model(rows, seed=7)
    development_scores = sorted(
        {
            row.score
            for row in score_examples(
                [example for example in rows if example.split == "development"], model
            )
        }
    )

    policy = calibrate_release_policy(rows, model, alpha=0.20, delta=0.05)

    assert [candidate.threshold for candidate in policy.candidates] == development_scores


def test_calibration_rejects_score_model_from_different_development_questions() -> None:
    rows = _examples()
    model = fit_logistic_risk_model(rows, seed=7).model_copy(
        update={"development_question_ids": ["different-development-question"]}
    )

    with pytest.raises(CalibrationContractError, match="development_identity_mismatch"):
        calibrate_release_policy(
            rows,
            model,
            alpha=0.20,
            delta=0.05,
            candidate_thresholds=[0.5],
        )


def test_serialized_policy_rejects_inconsistent_selected_candidate() -> None:
    rows = _examples()
    model = fit_logistic_risk_model(rows, seed=7)
    policy = calibrate_release_policy(
        rows,
        model,
        alpha=0.20,
        delta=0.05,
        candidate_thresholds=[0.5],
    )
    payload = policy.model_dump(mode="json")
    payload["status"] = "abstain_all"
    payload["threshold"] = None
    payload["selected"] = None

    with pytest.raises(ValueError, match="selected_candidate_mismatch"):
        CalibratedReleasePolicy.model_validate(payload)


def test_staged_freeze_excludes_test_and_binds_later_evaluation() -> None:
    rows = _examples()
    development_calibration = [row for row in rows if row.split != "test"]
    test = [row for row in rows if row.split == "test"]
    model = fit_logistic_risk_model(development_calibration, seed=7)
    calibration_scores = score_examples(
        [row for row in development_calibration if row.split == "calibration"], model
    )
    supported_scores = [
        row.score for row in calibration_scores if not row.example.unsupported_claim
    ]
    unsupported_scores = [row.score for row in calibration_scores if row.example.unsupported_claim]
    threshold = (max(supported_scores) + min(unsupported_scores)) / 2.0

    bundle = freeze_calibration_bundle(
        development_calibration,
        alpha=0.20,
        delta=0.05,
        seed=7,
        candidate_thresholds=[threshold],
    )
    assert bundle.freeze_state == "test_labels_unopened"
    assert bundle.policy.status == "calibrated"
    assert not set(bundle.development.question_ids) & {row.question_id for row in test}
    assert not set(bundle.calibration.paper_ids) & {
        paper_id for row in test for paper_id in row.paper_ids
    }

    artifact = evaluate_frozen_calibration_bundle(test, bundle)
    assert artifact["evaluation_stage"] == "held_out_test_after_freeze"
    assert artifact["frozen_bundle_sha256"] == bundle.bundle_sha256
    assert artifact["test_evaluation"]["total"] == 20
    assert artifact["test_evaluation"]["accepted"] == 15


def test_prospective_assessment_releases_or_abstains_without_labels() -> None:
    rows = _examples()
    bundle = _calibrated_bundle(rows)
    supported = next(
        row for row in rows if row.split == "test" and not row.unsupported_claim
    )
    unsupported = next(row for row in rows if row.split == "test" and row.unsupported_claim)

    released = assess_release_candidate(_release_candidate(supported), bundle)
    abstained = assess_release_candidate(_release_candidate(unsupported), bundle)

    assert "unsupported_claim" not in ReleaseCandidate.model_fields
    assert released.status == "released"
    assert released.reason == "risk_within_frozen_policy"
    assert released.scalar_risk_score <= float(released.threshold or 0.0)
    assert abstained.status == "abstained"
    assert abstained.reason == "risk_above_threshold"
    assert abstained.scalar_risk_score > float(abstained.threshold or 1.0)
    assert released.frozen_bundle_sha256 == bundle.bundle_sha256


def test_prospective_assessment_rejects_drift_and_frozen_overlap() -> None:
    rows = _examples()
    bundle = _calibrated_bundle(rows)
    candidate = _release_candidate(next(row for row in rows if row.split == "test"))
    changes = (
        ({"pipeline_sha256": "b" * 64}, "pipeline_mismatch"),
        ({"population_id": "shifted-population"}, "population_mismatch"),
        ({"features": {"different_feature": 0.1}}, "feature_schema_mismatch"),
        ({"question_id": bundle.development.question_ids[0]}, "question_overlap"),
        ({"paper_ids": [bundle.calibration.paper_ids[0]]}, "paper_overlap"),
    )
    for update, match in changes:
        changed = candidate.model_copy(update=update)
        with pytest.raises(CalibrationContractError, match=match):
            assess_release_candidate(changed, bundle)


def test_prospective_assessment_honors_abstain_all_policy() -> None:
    rows = _examples()
    bundle = freeze_calibration_bundle(
        [row for row in rows if row.split != "test"],
        alpha=0.05,
        delta=0.05,
        candidate_thresholds=[1.0],
    )
    candidate = _release_candidate(next(row for row in rows if row.split == "test"))

    assessment = assess_release_candidate(candidate, bundle)

    assert bundle.policy.status == "abstain_all"
    assert assessment.status == "abstained"
    assert assessment.reason == "policy_abstain_all"
    assert assessment.threshold is None


def test_staged_freeze_rejects_any_test_row() -> None:
    with pytest.raises(CalibrationContractError, match="exclude_test"):
        freeze_calibration_bundle(_examples(), alpha=0.20, delta=0.05)


def test_one_shot_artifact_rejects_non_simulation_labels() -> None:
    rows = [row.model_copy(update={"label_source": "benchmark_annotation"}) for row in _examples()]
    model = fit_logistic_risk_model(rows)
    policy = calibrate_release_policy(
        rows, model, alpha=0.20, delta=0.05, candidate_thresholds=[0.5]
    )
    evaluation = evaluate_release_policy(rows, model, policy)
    with pytest.raises(CalibrationContractError, match="requires_simulation_labels"):
        calibration_artifact(
            examples=rows,
            model=model,
            policy=policy,
            evaluation=evaluation,
        )


def test_staged_test_rejects_frozen_question_and_paper_overlap() -> None:
    rows = _examples()
    bundle = freeze_calibration_bundle(
        [row for row in rows if row.split != "test"],
        alpha=0.20,
        delta=0.05,
        candidate_thresholds=[0.5],
    )
    test = [row for row in rows if row.split == "test"]
    question_overlap = test.copy()
    question_overlap[0] = question_overlap[0].model_copy(
        update={"question_id": bundle.development.question_ids[0]}
    )
    with pytest.raises(CalibrationContractError, match="frozen_test_question_overlap"):
        evaluate_frozen_calibration_bundle(question_overlap, bundle)

    paper_overlap = test.copy()
    paper_overlap[0] = paper_overlap[0].model_copy(
        update={"paper_ids": [bundle.calibration.paper_ids[0]]}
    )
    with pytest.raises(CalibrationContractError, match="frozen_test_paper_overlap"):
        evaluate_frozen_calibration_bundle(paper_overlap, bundle)


def test_staged_test_rejects_schema_pipeline_population_and_bundle_tampering() -> None:
    rows = _examples()
    bundle = freeze_calibration_bundle(
        [row for row in rows if row.split != "test"],
        alpha=0.20,
        delta=0.05,
        candidate_thresholds=[0.5],
    )
    test = [row for row in rows if row.split == "test"]
    changes = (
        ({"pipeline_sha256": "b" * 64}, "pipeline_mismatch"),
        ({"population_id": "different-population"}, "population_mismatch"),
        ({"features": {"new_feature": 0.1}}, "feature_schema_mismatch"),
    )
    for update, match in changes:
        changed = [row.model_copy(update=update) for row in test]
        with pytest.raises(CalibrationContractError, match=match):
            evaluate_frozen_calibration_bundle(changed, bundle)

    payload = bundle.model_dump(mode="json")
    payload["policy"]["alpha"] = 0.19
    with pytest.raises(ValueError, match="hash_mismatch"):
        FrozenCalibrationBundle.model_validate(payload)


def test_prospective_assessment_revalidates_nested_bundle_hashes_after_mutation() -> None:
    rows = _examples()
    bundle = _calibrated_bundle(rows)
    unsupported = next(row for row in rows if row.split == "test" and row.unsupported_claim)
    candidate = _release_candidate(unsupported)
    assert assess_release_candidate(candidate, bundle).status == "abstained"

    # In-place list mutation bypasses Pydantic assignment hooks.  The scoring boundary
    # must nevertheless revalidate the stored model/policy/bundle hashes.
    bundle.score_model.coefficients[:] = [
        -1_000_000.0 for _ in bundle.score_model.coefficients
    ]

    with pytest.raises(CalibrationContractError, match="integrity_changed"):
        assess_release_candidate(candidate, bundle)


def test_policy_abstains_all_when_no_threshold_is_certified() -> None:
    rows = _examples()
    model = fit_logistic_risk_model(rows)
    policy = calibrate_release_policy(
        rows,
        model,
        alpha=0.05,
        delta=0.05,
        candidate_thresholds=[1.0],
    )
    assert policy.status == "abstain_all"
    assert policy.threshold is None
    evaluation = evaluate_release_policy(rows, model, policy)
    assert evaluation.accepted == 0
    assert evaluation.empirical_risk is None


def test_model_requires_exact_feature_schema() -> None:
    rows = _examples()
    model = fit_logistic_risk_model(rows)
    with pytest.raises(CalibrationContractError, match="risk_feature_set_mismatch"):
        model.score_features({"bootstrap_instability": 0.2})
    score = model.score_features(rows[0].features)
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0


def test_planted_simulation_is_deterministic_and_population_scoped() -> None:
    first = simulate_questions(
        seed=11,
        development_count=10,
        calibration_count=10,
        test_count=10,
    )
    second = simulate_questions(
        seed=11,
        development_count=10,
        calibration_count=10,
        test_count=10,
    )
    assert first == second
    assert {row.example.label_source for row in first} == {"simulation"}
    assert {row.example.population_id for row in first} == {"planted-risk-simulation-v1"}
    validate_split_integrity([row.example for row in first])


def test_repeated_simulation_reports_nontransferable_policy_comparison() -> None:
    replicate = simulate_replicate(
        seed=17,
        development_count=80,
        calibration_count=160,
        test_count=240,
        alpha=0.20,
        candidate_thresholds=(0.03, 0.05, 0.10, 0.20),
    )
    assert set(replicate["policies"]) == {
        "bootstrap_instability_only",
        "calibrated",
        "fixed_at_least_five_papers",
        "uncalibrated_score_at_alpha",
    }
    summary = summarize_replicates([replicate], alpha=0.20)
    assert summary["replicate_count"] == 1
    assert "does not calibrate risk on real" in summary["interpretation"]
