from __future__ import annotations

import copy
from pathlib import Path

import pytest
from scripts.run_adaptive_stress_study import _SOURCE_FILES

from literature_multiverse.adaptive_stress_study import (
    DEFAULT_SCENARIOS,
    FIXED_BUDGET_POLICIES,
    AdaptiveStressStudyError,
    StressOracleItem,
    StressVisibleItem,
    build_adaptive_stress_study_artifact,
    freeze_stress_study_config,
    generate_stress_question_receipts,
    summarize_stress_question_receipts,
    validate_adaptive_stress_study_artifact,
    validate_stress_question_receipt,
    validate_stress_study_config,
)
from literature_multiverse.lineage import sha256_file

_EXPECTED_RUNNER_SOURCES = {
    "pyproject.toml",
    "scripts/run_adaptive_stress_study.py",
    "src/literature_multiverse/__init__.py",
    "src/literature_multiverse/adaptive_stress_study.py",
    "src/literature_multiverse/budgeted_verification.py",
    "src/literature_multiverse/lineage.py",
    "src/literature_multiverse/models.py",
    "src/literature_multiverse/paths.py",
    "uv.lock",
}


def _config(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "questions_per_scenario": 1,
        "items_per_question": 12,
        "budgets_minutes": (10.0, 20.0),
        "release_risk_thresholds": (0.05, 0.10, 1.0),
        "primary_budget_minutes": 10.0,
        "primary_release_risk_threshold": 0.10,
        "release_monte_carlo_draws": 32,
        "bootstrap_draws": 100,
    }
    values.update(overrides)
    return freeze_stress_study_config(**values)  # type: ignore[arg-type]


def test_public_runner_seals_complete_internal_source_closure() -> None:
    assert set(_SOURCE_FILES) == _EXPECTED_RUNNER_SOURCES
    assert all(sha256_file(Path(relative)) for relative in _SOURCE_FILES)


def test_frozen_config_is_deterministic_and_detects_tampering() -> None:
    first = _config()
    second = _config()
    assert first == second
    validate_stress_study_config(first)

    tampered = copy.deepcopy(first)
    tampered["fixed_count"] = 6
    with pytest.raises(AdaptiveStressStudyError, match="config_hash_mismatch"):
        validate_stress_study_config(tampered)


def test_policy_visible_and_hidden_oracle_contracts_have_no_shared_outcome_fields() -> None:
    visible_fields = set(StressVisibleItem.__dataclass_fields__)
    oracle_fields = set(StressOracleItem.__dataclass_fields__)
    assert visible_fields & {
        "true_contribution",
        "is_extraction_error",
        "reviewed_contribution",
        "reviewer_is_correct",
    } == set()
    assert oracle_fields & {"risk_score", "disagreement_score"} == set()


def test_receipts_cover_every_policy_budget_and_respect_matched_budgets() -> None:
    config = _config(scenarios=("correlated_shared_cohort",))
    receipt = generate_stress_question_receipts(config)[0]
    validate_stress_question_receipt(
        receipt, expected_config_sha256=str(config["config_sha256"])
    )
    evaluations = receipt["evaluations"]
    assert len(evaluations) == len(FIXED_BUDGET_POLICIES) * 2 + 2
    assert {row["policy"] for row in evaluations} == {
        *FIXED_BUDGET_POLICIES,
        "no_audit",
        "audit_all",
    }
    for row in evaluations:
        if row["budget_role"] == "matched_fixed_person_minutes":
            assert row["spent_person_minutes"] <= row["nominal_budget_minutes"] + 1e-9
        if row["policy"] == "no_audit":
            assert row["spent_person_minutes"] == 0.0
            assert row["selected_item_ids"] == []


def test_harsh_generator_contains_every_requested_misspecification() -> None:
    config = _config(
        questions_per_scenario=20,
        scenarios=("combined_domain_shift",),
    )
    receipts = generate_stress_question_receipts(config)
    totals = {
        key: sum(receipt["generator_diagnostics"][key] for receipt in receipts)
        for key in (
            "extraction_errors",
            "shared_component_errors",
            "missing_full_text",
            "reviewer_mistakes_among_accessible",
        )
    }
    assert all(value > 0 for value in totals.values())
    assert config["scenario_profiles"]["combined_domain_shift"][
        "risk_logit_slope"
    ] < 0
    assert all(
        abs(receipt["generator_diagnostics"]["interaction_strength"]) > 0
        for receipt in receipts
    )


def test_sequential_reallocation_really_changes_some_shared_cohort_paths() -> None:
    config = _config(
        questions_per_scenario=12,
        scenarios=("correlated_shared_cohort",),
    )
    receipts = generate_stress_question_receipts(config)
    changed = False
    for receipt in receipts:
        rows = {
            row["arm_id"]: row
            for row in receipt["evaluations"]
            if row["nominal_budget_minutes"] == 10.0
        }
        static = rows["adaptive_static__minutes-10"]["selected_item_ids"]
        sequential = rows["adaptive_sequential__minutes-10"]["selected_item_ids"]
        changed |= static != sequential
    assert changed


def test_summary_reconciles_release_error_coverage_and_question_unit() -> None:
    config = _config()
    receipts = generate_stress_question_receipts(config)
    summary = summarize_stress_question_receipts(receipts, config)
    assert summary["independent_questions"] == len(DEFAULT_SCENARIOS)
    assert summary["uncertainty"]["sampling_unit"] == (
        "independent_complete_simulated_question"
    )
    arm = summary["arms"]["adaptive_sequential__minutes-10"]
    final_point = arm["scopes"]["overall"]["curve"][-1]
    assert final_point["coverage"] == 1.0
    assert final_point["released_claims"] == final_point["questions"]
    assert final_point["released_claim_errors"] == round(
        final_point["released_claim_error_rate"] * final_point["released_claims"]
    )
    assert arm["budget_role"] == "matched_fixed_person_minutes"
    assert summary["arms"]["audit_all"]["budget_role"] == (
        "unmatched_exhaustive_review_reference"
    )


def test_aggregate_artifact_and_receipt_hashes_are_deterministic_and_fail_closed() -> None:
    config = _config(scenarios=("iid_control", "combined_domain_shift"))
    first = build_adaptive_stress_study_artifact(config)
    second = build_adaptive_stress_study_artifact(config)
    assert first == second
    validate_adaptive_stress_study_artifact(first)
    assert first["evidence_scope"]["simulation_only"] is True
    assert first["evidence_scope"]["real_world_evidence"] is False

    tampered_artifact = copy.deepcopy(first)
    tampered_artifact["summary"]["independent_questions"] += 1
    with pytest.raises(AdaptiveStressStudyError, match="artifact_hash_mismatch"):
        validate_adaptive_stress_study_artifact(tampered_artifact)

    receipt = generate_stress_question_receipts(config)[0]
    tampered_receipt = copy.deepcopy(receipt)
    tampered_receipt["evaluations"][0]["claim_decision_error"] = not tampered_receipt[
        "evaluations"
    ][0]["claim_decision_error"]
    with pytest.raises(AdaptiveStressStudyError, match="question_receipt_hash_mismatch"):
        validate_stress_question_receipt(tampered_receipt)
