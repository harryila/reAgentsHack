from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from scripts.run_evidence_inference_fable_retrospective_v1 import main as harness_main

import literature_multiverse.evidence_inference_fable_retrospective_inference_v1 as inference
import literature_multiverse.evidence_inference_fable_retrospective_v1 as retrospective
from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    reconstruct_evidence_inference_fable_prepared_runtime_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_inference_v1 import (
    EXPECTED_FULL_PLAN_SHA256,
    EXPECTED_PILOT_PLAN_SHA256,
    EXPECTED_RECOVERY_PILOT_PLAN_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
ATTEMPTED_ARTICLES = {
    "PMC1871574",
    "PMC2361948",
    "PMC2858204",
    "PMC2956883",
    "PMC5062234",
    "PMC5602855",
}
ATTEMPTED_REQUESTS = {
    "ei-fable-retro-v1-pilot30-test-seed-pmc2361948",
    "ei-fable-retro-v1-pilot30-test-seed-pmc2858204",
    "ei-fable-retro-v1-pilot30-test-seed-pmc2956883",
    "ei-fable-retro-v1-pilot30-test-seed-pmc5062234",
    "ei-fable-retro-v1-pilot30-test-seed-pmc5602855",
    "ei-fable-retro-v1-pilot30-test-winner-pmc1871574",
    "ei-fable-retro-v1-pilot30-test-winner-pmc2361948",
    "ei-fable-retro-v1-pilot30-test-winner-pmc2858204",
    "ei-fable-retro-v1-pilot30-test-winner-pmc2956883",
    "ei-fable-retro-v1-pilot30-test-winner-pmc5062234",
    "ei-fable-retro-v1-pilot30-test-winner-pmc5602855",
}


def test_recovery_v2_is_label_blind_exact_30_and_article_disjoint(
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
        mode="pilot30_recovery_v2_paired",
    )

    articles = {item.article_id for item in plan.roster}
    request_keys = {item.request_key for item in plan.roster}
    assert plan.plan_sha256 == EXPECTED_RECOVERY_PILOT_PLAN_SHA256
    assert (plan.unique_examples, plan.unique_articles, plan.request_count) == (30, 7, 14)
    assert sum(item.question_count for item in plan.roster) == 60
    assert not articles & ATTEMPTED_ARTICLES
    assert not request_keys & ATTEMPTED_REQUESTS
    assert all(key.startswith("ei-fable-retro-v2-") for key in request_keys)
    assert plan.pilot_is_mechanics_only_no_inferential_authority is True
    assert plan.confirmatory_claim_authority is False
    assert plan.calibration_authority is False
    assert plan.claim_release_authority is False
    assert not any(path.suffix == ".jsonl" for path in opened)
    assert not any(path.name == "annotations_merged.csv" for path in opened)


def test_recovery_v2_reconstructs_exact_runtime_surface_roster() -> None:
    plan, prepared = reconstruct_evidence_inference_fable_prepared_runtime_v1(
        repository_root=ROOT,
        mode="pilot30_recovery_v2_paired",
    )

    assert plan.plan_sha256 == EXPECTED_RECOVERY_PILOT_PLAN_SHA256
    assert prepared.retrospective_plan_sha256 == plan.plan_sha256
    assert prepared.request_roster_sha256 == plan.request_roster_sha256
    assert [item.request_key for item in prepared.surfaces] == [
        item.request_key for item in plan.roster
    ]
    inference._require_frozen_plan(plan)


def test_recovery_derivation_does_not_change_predeclared_source_plans() -> None:
    pilot = retrospective.freeze_evidence_inference_fable_retrospective_plan_v1(
        repository_root=ROOT,
        mode="pilot30_paired",
    )
    full = retrospective.freeze_evidence_inference_fable_retrospective_plan_v1(
        repository_root=ROOT,
        mode="full_paired",
    )

    assert pilot.plan_sha256 == EXPECTED_PILOT_PLAN_SHA256
    assert full.plan_sha256 == EXPECTED_FULL_PLAN_SHA256


def test_recovery_v2_mode_is_accepted_by_internal_full_preflight_mechanics() -> None:
    recovery = retrospective.freeze_evidence_inference_fable_retrospective_plan_v1(
        repository_root=ROOT,
        mode="pilot30_recovery_v2_paired",
    )
    full = retrospective.freeze_evidence_inference_fable_retrospective_plan_v1(
        repository_root=ROOT,
        mode="full_paired",
    )
    binding = inference._freeze_scoring_completion_binding_v1(
        plan=recovery,
        runtime_terminal_sha256="1" * 64,
        scoring_completion_certificate_sha256="2" * 64,
        private_scored_rows_sha256="3" * 64,
        scoring_artifact_sha256="4" * 64,
        terminal_receipt_count=recovery.request_count,
    )
    summary = inference._pilot_terminal_summary_from_validated_binding_v1(
        pilot_plan=recovery,
        full_plan=full,
        scoring_binding=binding,
    )
    decision = inference._evaluate_full_preflight_gate_from_summary_v1(
        pilot_plan=recovery,
        full_plan=full,
        pilot_terminal=summary,
    )

    assert decision.full_preflight_prerequisite_satisfied is True
    assert decision.provider_execution_or_spend_authority is False
    assert decision.scientific_claim_authority is False


def test_recovery_v2_bootstrap_accepts_recovery_population() -> None:
    recovery = retrospective.freeze_evidence_inference_fable_retrospective_plan_v1(
        repository_root=ROOT,
        mode="pilot30_recovery_v2_paired",
    )
    binding = inference._freeze_scoring_completion_binding_v1(
        plan=recovery,
        runtime_terminal_sha256="1" * 64,
        scoring_completion_certificate_sha256="2" * 64,
        private_scored_rows_sha256="3" * 64,
        scoring_artifact_sha256="4" * 64,
        terminal_receipt_count=recovery.request_count,
    )
    membership: dict[str, list[str]] = {}
    for request in recovery.roster:
        membership.setdefault(request.article_id, request.example_ids)
    clusters = [
        inference.freeze_article_cluster_paired_scores_v1(
            article_id=article_id,
            example_ids=example_ids,
            metric_success_counts={
                metric: (len(example_ids), len(example_ids))
                for metric in inference.METRICS
            },
        )
        for article_id, example_ids in sorted(membership.items())
    ]

    result = inference.bootstrap_paired_article_clusters_v1(
        plan=recovery,
        scoring_binding=binding,
        clusters=clusters,
    )

    assert result.population == "pilot30_recovery_v2_test"
    assert result.mode == "pilot30_recovery_v2_paired"
    assert result.question_count == 30
    assert result.pilot_mechanics_only_no_inferential_authority is True


def test_recovery_v2_harness_prepares_default_frozen_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "recovery-v2-runtime"
    assert (
        harness_main(
            [
                "prepare",
                "--repository-root",
                str(ROOT),
                "--mode",
                "pilot30_recovery_v2_paired",
                "--workspace",
                str(workspace),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "EvidenceInferenceFablePreparedRuntimeV1" in output
    prepared = (workspace / "00-prepared.json").read_text(encoding="utf-8")
    assert EXPECTED_RECOVERY_PILOT_PLAN_SHA256 in prepared
