from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import literature_multiverse.evidence_inference_fable_retrospective_v1 as retrospective
from literature_multiverse.evidence_inference_fable_retrospective_v1 import (
    EvidenceInferenceFableRetrospectiveError,
)

ROOT = Path(__file__).resolve().parents[1]


def test_pilot_plan_is_label_safe_article_batched_and_budget_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Path] = []
    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes

    def guarded_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path.suffix == ".jsonl" or path.name == "annotations_merged.csv":
            raise AssertionError(f"reference payload opened: {path}")
        opened.append(path)
        return original_read_text(path, *args, **kwargs)

    def guarded_bytes(path: Path, *args: Any, **kwargs: Any) -> bytes:
        if path.suffix == ".jsonl" or path.name == "annotations_merged.csv":
            raise AssertionError(f"reference payload opened: {path}")
        opened.append(path)
        return original_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    plan = retrospective.freeze_evidence_inference_fable_retrospective_plan_v1(
        repository_root=ROOT,
        mode="pilot30_paired",
    )

    assert (plan.unique_examples, plan.unique_articles, plan.request_count) == (30, 7, 14)
    assert sum(item.question_count for item in plan.roster) == 60
    assert plan.pilot_preflight_required_before_full_authorization is False
    assert plan.provider_calls_made == 0
    assert plan.labels_opened_by_planner is False
    assert plan.pilot_population_is_subset_of_full_test is True
    assert plan.pilot_is_mechanics_only_no_inferential_authority is True
    assert plan.total_full_context_hard_liability_usd_micros > (
        plan.total_diagnostic_known_surface_cost_usd_micros
    )
    assert not any(path.suffix == ".jsonl" for path in opened)
    assert not any(path.name == "annotations_merged.csv" for path in opened)


def test_full_paired_is_exactly_382_article_arm_calls_for_524_questions() -> None:
    plan = retrospective.freeze_evidence_inference_fable_retrospective_plan_v1(
        repository_root=ROOT,
        mode="full_paired",
    )

    assert (plan.unique_examples, plan.unique_articles) == (524, 191)
    assert (plan.arm_count, plan.request_count, plan.question_evaluations) == (
        2,
        382,
        1048,
    )
    assert min(item.question_count for item in plan.roster) == 1
    assert max(item.question_count for item in plan.roster) == 15
    assert plan.pilot_preflight_required_before_full_authorization is True
    assert plan.comparison_interpretation == (
        "exploratory_cross_model_transfer_on_historically_opened_test"
    )
    assert plan.exploratory_cross_model_transfer_comparison_permitted is True
    assert plan.confirmatory_gepa_improvement_claim_permitted is False
    assert plan.cross_model_and_batched_interface_transfer_only is True
    assert plan.scaled_optimizer_input_policy == (
        "single_question_fixed_results_passage_projection_v1"
    )
    assert plan.eligibility_metric_claim_authority is False


def test_scaled_seed_and_winner_are_distinct_fair_paired_arms() -> None:
    plan = retrospective.freeze_evidence_inference_fable_retrospective_plan_v1(
        repository_root=ROOT,
        mode="full_paired",
    )

    assert plan.scaled_winner_prompt_sha256 != plan.recorded_seed_prompt_sha256
    assert plan.scaled_winner_prompt_sha256 == plan.scaled_winner_embedded_prompt_sha256
    assert plan.recorded_seed_prompt_sha256 == (
        plan.seed_prompt_extracted_from_scaled_trace_sha256
    )
    assert plan.scaled_winner_candidate_count == 7
    pairs = [plan.roster[index : index + 2] for index in range(0, 382, 2)]
    assert all({first.arm, second.arm} == {"seed", "winner"} for first, second in pairs)
    assert all(first.article_id == second.article_id for first, second in pairs)
    assert abs(
        sum(first.arm == "seed" for first, _ in pairs)
        - sum(first.arm == "winner" for first, _ in pairs)
    ) <= 1
    with pytest.raises(
        EvidenceInferenceFableRetrospectiveError,
        match="historically_opened_test",
    ):
        retrospective.require_confirmatory_gepa_improvement_claim_v1(plan)


def test_output_cap_scales_by_article_question_count() -> None:
    plan = retrospective.freeze_evidence_inference_fable_retrospective_plan_v1(
        repository_root=ROOT,
        mode="full_paired",
    )
    for item in plan.roster:
        assert item.max_output_tokens == min(32000, 8192 + 1024 * item.question_count)
        assert item.cost.diagnostic_known_input_token_ceiling == (
            item.model_facing_utf8_bytes + 2048
        )
        assert item.cost.full_context_hard_liability_usd_micros == (
            1_000_000 * 10 + item.max_output_tokens * 50
        )
