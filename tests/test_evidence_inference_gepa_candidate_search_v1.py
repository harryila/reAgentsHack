from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from scripts import prepare_evidence_inference_gepa_candidate_search_v1 as cli
from tests.private_cache_support import require_private_cache

import literature_multiverse.evidence_inference_gepa_candidate_search_v1 as gepa_plan
from literature_multiverse.evidence_inference_gepa_candidate_search_v1 import (
    EvidenceInferenceGEPACandidateSearchPlanV1,
    freeze_gepa_candidate_search_development_decision_v1,
)
from literature_multiverse.lineage import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
FROZEN_AT = datetime(2026, 8, 29, tzinfo=UTC)

pytestmark = pytest.mark.private_cache

_PRIVATE_CANDIDATE_SEARCH_INPUTS = (
    "data/cache/evidence-inference-gepa/manifest.json",
    "data/cache/gepa/evidence-inference-first-pass-v2",
    "data/cache/evidence-inference-ollama-gepa-v1-final-v3",
)


@pytest.fixture(scope="module")
def frozen_plan_with_split_calls() -> tuple[
    EvidenceInferenceGEPACandidateSearchPlanV1, tuple[str, ...]
]:
    require_private_cache(*_PRIVATE_CANDIDATE_SEARCH_INPUTS)
    calls: list[str] = []
    original = gepa_plan.load_manifest_split

    def guarded_loader(path: Path, split_name: str):
        calls.append(split_name)
        if split_name not in {"train", "dev"}:
            raise AssertionError("candidate-search planner crossed the frozen test boundary")
        return original(path, split_name)

    with patch.object(gepa_plan, "load_manifest_split", side_effect=guarded_loader):
        plan = gepa_plan.freeze_evidence_inference_gepa_candidate_search_plan_v1(
            repository_root=ROOT,
            frozen_at=FROZEN_AT,
        )
    return plan, tuple(calls)


@pytest.fixture(scope="module")
def frozen_plan(
    frozen_plan_with_split_calls: tuple[
        EvidenceInferenceGEPACandidateSearchPlanV1, tuple[str, ...]
    ],
) -> EvidenceInferenceGEPACandidateSearchPlanV1:
    return frozen_plan_with_split_calls[0]


def test_freeze_reads_train_and_dev_only_and_grants_no_authority(
    frozen_plan_with_split_calls: tuple[
        EvidenceInferenceGEPACandidateSearchPlanV1, tuple[str, ...]
    ],
) -> None:
    plan, calls = frozen_plan_with_split_calls
    assert calls == ("train", "dev")
    assert plan.provider_calls_made == 0
    assert plan.reflection_calls_made == 0
    assert plan.task_calls_made == 0
    assert not plan.credentials_read
    assert not plan.network_accessed
    assert not plan.test_payload_opened
    assert not plan.test_payload_hashed
    assert not plan.test_labels_opened
    assert not plan.test_labels_scored
    assert not plan.test_example_ids_materialized_in_plan
    assert not plan.improvement_authority
    assert not plan.generalization_authority
    assert not plan.scientific_effectiveness_authority
    assert not plan.claim_release_authority


def test_multiple_genuinely_distinct_candidates_exist_before_evaluation(
    frozen_plan: EvidenceInferenceGEPACandidateSearchPlanV1,
) -> None:
    cheap, scaled = frozen_plan.tiers
    assert [cheap.tier, scaled.tier] == ["cheap_pilot", "scaled"]
    assert [len(cheap.pre_evaluation_candidates), len(scaled.pre_evaluation_candidates)] == [
        4,
        7,
    ]
    for tier in frozen_plan.tiers:
        candidates = tier.pre_evaluation_candidates
        assert candidates[0].role == "handwritten_seed"
        assert all(item.role == "code_owned_diverse_start" for item in candidates[1:])
        assert all(item.created_before_any_task_evaluation for item in candidates)
        assert all(not item.provider_generated for item in candidates)
        assert len({item.prompt_sha256 for item in candidates}) == len(candidates)
        assert len({item.normalized_prompt_sha256 for item in candidates}) == len(candidates)
        assert len({item.mutation_axis for item in candidates}) == len(candidates)
    assert {
        item.prompt_sha256 for item in cheap.pre_evaluation_candidates
    }.issubset({item.prompt_sha256 for item in scaled.pre_evaluation_candidates})


def test_article_disjoint_train_search_and_confirmation_memberships(
    frozen_plan: EvidenceInferenceGEPACandidateSearchPlanV1,
) -> None:
    assert frozen_plan.official_train_dev_article_overlap == 0
    assert frozen_plan.official_train_dev_group_overlap == 0
    for tier in frozen_plan.tiers:
        train = set(tier.train_membership.paper_ids)
        search = set(tier.dev_search_membership.paper_ids)
        confirmation = set(tier.dev_confirmation_membership.paper_ids)
        assert train.isdisjoint(search)
        assert train.isdisjoint(confirmation)
        assert search.isdisjoint(confirmation)
        for membership in (
            tier.train_membership,
            tier.dev_search_membership,
            tier.dev_confirmation_membership,
        ):
            assert membership.representatives == len(membership.example_ids)
            assert membership.representatives == len(membership.paper_ids)
            assert membership.representatives == len(membership.group_ids)


def test_exact_call_and_cost_ceilings_are_frozen(
    frozen_plan: EvidenceInferenceGEPACandidateSearchPlanV1,
) -> None:
    cheap = frozen_plan.tiers[0].call_budget
    assert (
        cheap.initial_train_task_call_ceiling,
        cheap.dev_search_task_call_ceiling,
        cheap.confirmation_task_call_ceiling,
    ) == (16, 30, 12)
    assert cheap.task_provider_call_ceiling == 58
    assert cheap.reflection_call_ceiling == 2
    assert cheap.total_provider_call_ceiling == 60
    assert cheap.total_hard_cost_liability_usd_micros == 13_373_440

    scaled = frozen_plan.tiers[1].call_budget
    assert (
        scaled.initial_train_task_call_ceiling,
        scaled.dev_search_task_call_ceiling,
        scaled.confirmation_task_call_ceiling,
    ) == (112, 320, 64)
    assert scaled.task_provider_call_ceiling == 496
    assert scaled.reflection_call_ceiling == 6
    assert scaled.total_provider_call_ceiling == 502
    assert scaled.total_hard_cost_liability_usd_micros == 109_363_200
    assert cheap.task_call_cost_ceiling_usd_micros == 215_040
    assert cheap.reflection_call_cost_ceiling_usd_micros == 450_560
    assert scaled.task_call_cost_ceiling_usd_micros == 215_040
    assert scaled.reflection_call_cost_ceiling_usd_micros == 450_560


def test_objectives_are_separate_and_selection_has_no_weighted_scalar(
    frozen_plan: EvidenceInferenceGEPACandidateSearchPlanV1,
) -> None:
    assert [item["objective"] for item in frozen_plan.objectives] == [
        "extraction_correctness",
        "formal_grounding_validity",
        "structured_output_validity",
        "provider_usage_and_cost",
    ]
    assert [item["hard_gate"] for item in frozen_plan.objectives] == [
        False,
        True,
        True,
        False,
    ]
    assert "noninferior_to_seed" in frozen_plan.selection_rule
    assert "weighted" not in frozen_plan.selection_rule
    assert all("weight" not in item for item in frozen_plan.objectives)


def test_prior_diagnosis_preserves_obsolete_and_authoritative_results(
    frozen_plan: EvidenceInferenceGEPACandidateSearchPlanV1,
) -> None:
    prior = frozen_plan.prior_diagnosis
    assert prior.obsolete_first_pass_candidate_count == 2
    assert prior.obsolete_first_pass_distinct_mutation_count == 1
    assert prior.obsolete_first_pass_seed_retained
    assert (
        prior.obsolete_first_pass_winner_prompt_sha256
        == prior.obsolete_first_pass_seed_prompt_sha256
    )
    assert prior.authoritative_scaled_candidate_count == 7
    assert prior.authoritative_scaled_reflection_proposals == 8
    assert not prior.authoritative_scaled_seed_retained
    assert (
        prior.authoritative_scaled_winner_prompt_sha256
        != prior.authoritative_scaled_seed_prompt_sha256
    )
    assert prior.authoritative_scaled_status == "no_improvement_claim"
    assert not prior.authoritative_scaled_observed_improvement_rule_satisfied
    assert (
        prior.new_search_rationale
        == "frontier_fable_high_transfer_with_separate_structured_grounding_objectives"
    )


def test_seed_winner_fails_closed_as_negative_result(
    frozen_plan: EvidenceInferenceGEPACandidateSearchPlanV1,
) -> None:
    decision = freeze_gepa_candidate_search_development_decision_v1(
        plan=frozen_plan,
        tier="scaled",
        winner_prompt_sha256=frozen_plan.seed_prompt_sha256,
    )
    assert decision.status == "seed_retained_negative_result"
    assert decision.winner_is_seed
    assert not decision.equal_budget_future_evaluation_required
    assert not decision.future_evaluation_performed
    assert not decision.improvement_authority
    assert not decision.generalization_authority


def test_nonseed_development_winner_still_has_no_improvement_authority(
    frozen_plan: EvidenceInferenceGEPACandidateSearchPlanV1,
) -> None:
    nonseed_sha = frozen_plan.tiers[1].pre_evaluation_candidates[1].prompt_sha256
    decision = freeze_gepa_candidate_search_development_decision_v1(
        plan=frozen_plan,
        tier="scaled",
        winner_prompt_sha256=nonseed_sha,
    )
    assert decision.status == "nonseed_development_candidate_frozen_no_improvement_claim"
    assert not decision.winner_is_seed
    assert decision.equal_budget_future_evaluation_required
    assert not decision.improvement_authority


def test_plan_tampering_fails_self_hash_validation(
    frozen_plan: EvidenceInferenceGEPACandidateSearchPlanV1,
) -> None:
    payload = deepcopy(frozen_plan.model_dump(mode="json"))
    payload["tiers"][0]["call_budget"]["total_provider_call_ceiling"] += 1
    with pytest.raises(ValidationError):
        EvidenceInferenceGEPACandidateSearchPlanV1.model_validate(payload)


def test_cli_status_is_contract_only_and_reports_boundaries(
    frozen_plan: EvidenceInferenceGEPACandidateSearchPlanV1,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = tmp_path / "plan.json"
    atomic_write_json(plan_path, frozen_plan)
    monkeypatch.setattr(cli, "_safe_plan_path", lambda **_kwargs: plan_path)
    monkeypatch.setattr(
        cli,
        "validate_evidence_inference_gepa_candidate_search_plan_v1",
        lambda **_kwargs: pytest.fail("status must not external-replay split payloads"),
    )
    assert cli.main(["status", "--repository-root", str(ROOT)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["validation"] == "serialized_contract_and_self_hash_only"
    assert summary["provider"] == {
        "effort": "high",
        "model": "claude-fable-5",
        "provider_calls_made": 0,
    }
    assert summary["test_boundary"] == {
        "labels_opened": False,
        "labels_scored": False,
        "payload_hashed": False,
        "payload_opened": False,
    }
    assert not any(summary["authorities"].values())


def test_cli_validate_external_replays_exact_plan(
    frozen_plan: EvidenceInferenceGEPACandidateSearchPlanV1,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_private_cache(*_PRIVATE_CANDIDATE_SEARCH_INPUTS)
    plan_path = tmp_path / "plan.json"
    atomic_write_json(plan_path, frozen_plan)
    monkeypatch.setattr(cli, "_safe_plan_path", lambda **_kwargs: plan_path)
    assert cli.main(["validate", "--repository-root", str(ROOT)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["validation"] == "external_train_dev_replay_exact"
    assert summary["plan_sha256"] == frozen_plan.plan_sha256
    assert [item["total_provider_call_ceiling"] for item in summary["tiers"]] == [
        60,
        502,
    ]
