from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    EvidenceInferenceFableProviderResultV1,
    authorize_evidence_inference_fable_workspace_v1,
    execute_evidence_inference_fable_paired_v1,
    freeze_evidence_inference_fable_budget_authorization_v1,
    freeze_evidence_inference_fable_call_surface_v1,
    freeze_evidence_inference_fable_prepared_runtime_v1,
    prepare_evidence_inference_fable_workspace_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_inference_v1 import (
    EXPECTED_FULL_PLAN_SHA256,
    EXPECTED_PILOT_PLAN_SHA256,
    ArticleClusterPairedScoresV1,
    EvidenceInferenceFableInferenceError,
    ScoringCompletionBindingV1,
    bootstrap_paired_article_clusters_v1,
    derive_scoring_completion_binding_from_workspace_v1,
    evaluate_full_preflight_gate_v1,
    freeze_article_cluster_paired_scores_v1,
    require_full_preflight_gate_v1,
    validate_paired_article_cluster_bootstrap_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_scoring_v1 import (
    ScoringCompletionCertificateV1,
)
from literature_multiverse.evidence_inference_fable_retrospective_v1 import (
    EvidenceInferenceFableRetrospectivePlanV1,
    freeze_evidence_inference_fable_retrospective_plan_v1,
)
from literature_multiverse.lineage import hash_canonical

ROOT = Path(__file__).resolve().parents[1]
FAKE_SCORED_ROWS_SHA = "2" * 64
FAKE_SCORING_ARTIFACT_SHA = "3" * 64
FAKE_RECEIPT_MEMBERSHIP_SHA = "4" * 64


class _OfflineFakeClient:
    """In-process transport double with no provider or network path."""

    def generate(self, surface: Any) -> EvidenceInferenceFableProviderResultV1:
        payload = {
            "result_version": "evidence-inference-fable-provider-result-v1",
            "request_key": surface.request_key,
            "surface_sha256": surface.surface_sha256,
            "transport_attempt_count": 1,
            "sdk_retry_count": 0,
            "outcome": "completed",
            "response_id": f"offline-{surface.request_key}",
            "response_model": "claude-fable-5",
            "parsed_json": {"offline_fixture": True},
            "input_tokens": 1,
            "output_tokens": 1,
            "reported_cost_usd_micros": 60,
            "charged_cost_usd_micros": 60,
            "cost_basis": "reported_usage",
            "response_text_sha256": None,
            "failure_code": None,
        }
        return EvidenceInferenceFableProviderResultV1.model_validate(
            {**payload, "result_sha256": hash_canonical(payload)}
        )


@pytest.fixture(scope="module")
def plans_and_surfaces() -> tuple[
    EvidenceInferenceFableRetrospectivePlanV1,
    EvidenceInferenceFableRetrospectivePlanV1,
    list[dict[str, Any]],
]:
    surfaces: list[dict[str, Any]] = []
    pilot = freeze_evidence_inference_fable_retrospective_plan_v1(
        repository_root=ROOT,
        mode="pilot30_paired",
        _model_surface_sink=surfaces,
    )
    full = freeze_evidence_inference_fable_retrospective_plan_v1(
        repository_root=ROOT,
        mode="full_paired",
    )
    return pilot, full, surfaces


@pytest.fixture(scope="module")
def completed_workspace_and_certificate(
    tmp_path_factory: pytest.TempPathFactory,
    plans_and_surfaces: tuple[
        EvidenceInferenceFableRetrospectivePlanV1,
        EvidenceInferenceFableRetrospectivePlanV1,
        list[dict[str, Any]],
    ],
) -> tuple[Path, ScoringCompletionCertificateV1]:
    pilot, _, model_surfaces = plans_and_surfaces
    surfaces = [
        freeze_evidence_inference_fable_call_surface_v1(
            roster_item=request,
            system=surface["system"],
            prompt=surface["prompt"],
            wire_schema=surface["wire_schema"],
        )
        for request, surface in zip(pilot.roster, model_surfaces, strict=True)
    ]
    prepared = freeze_evidence_inference_fable_prepared_runtime_v1(
        plan=pilot,
        surfaces=surfaces,
    )
    authorization = freeze_evidence_inference_fable_budget_authorization_v1(
        prepared=prepared,
        configured_total_budget_usd_micros=(
            pilot.total_full_context_hard_liability_usd_micros
        ),
    )
    workspace = tmp_path_factory.mktemp("fable-pilot-runtime") / "workspace"
    prepare_evidence_inference_fable_workspace_v1(
        workspace=workspace,
        prepared=prepared,
    )
    authorize_evidence_inference_fable_workspace_v1(
        workspace=workspace,
        authorization=authorization,
    )
    terminal = execute_evidence_inference_fable_paired_v1(
        workspace=workspace,
        plan=pilot,
        client=_OfflineFakeClient(),
    )
    assert terminal.status == "completed"
    certificate_payload = {
        "certificate_version": (
            "evidence-inference-fable-scoring-completion-certificate-v1"
        ),
        "scoring_version": "evidence-inference-fable-retrospective-scoring-v1",
        "status": "complete_private_scored_rows",
        "plan_sha256": pilot.plan_sha256,
        "runtime_terminal_sha256": terminal.terminal_sha256,
        "private_scored_rows_sha256": FAKE_SCORED_ROWS_SHA,
        "scoring_artifact_sha256": FAKE_SCORING_ARTIFACT_SHA,
        "receipt_membership_sha256": FAKE_RECEIPT_MEMBERSHIP_SHA,
        "planned_request_count": pilot.request_count,
        "terminal_receipt_count": pilot.request_count,
        "labels_loaded_only_after_complete_terminal_roster_validation": True,
        "all_terminal_receipt_lineage_validated": True,
        "invalid_batch_intention_to_evaluate": True,
        "eligible_false_is_unconditional_grounding_failure": True,
        "empty_finding_is_unconditional_grounding_failure": True,
        "exact_grounding_is_mechanical_not_entailment": True,
        "provider_execution_or_spend_authority": False,
        "confirmatory_gepa_improvement_authority": False,
        "scientific_claim_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    certificate = ScoringCompletionCertificateV1.model_validate(
        {
            **certificate_payload,
            "certificate_sha256": hash_canonical(certificate_payload),
        }
    )
    return workspace, certificate


def _clusters(
    plan: EvidenceInferenceFableRetrospectivePlanV1,
) -> list[ArticleClusterPairedScoresV1]:
    membership: dict[str, list[str]] = {}
    for request in plan.roster:
        membership.setdefault(request.article_id, request.example_ids)
        assert membership[request.article_id] == request.example_ids
    result: list[ArticleClusterPairedScoresV1] = []
    for index, article_id in enumerate(sorted(membership)):
        example_ids = membership[article_id]
        count = len(example_ids)
        result.append(
            freeze_article_cluster_paired_scores_v1(
                article_id=article_id,
                example_ids=example_ids,
                metric_success_counts={
                    "direction_accuracy": (
                        count // 2,
                        min(count, count // 2 + index % 2),
                    ),
                    "structured_output_reliability": (
                        max(0, count - index % 2),
                        count,
                    ),
                    "exact_grounding_reliability": (
                        count // 3,
                        min(count, count // 3 + 1),
                    ),
                },
            )
        )
    return result


def test_frozen_plan_hashes_are_exact(
    plans_and_surfaces: tuple[
        EvidenceInferenceFableRetrospectivePlanV1,
        EvidenceInferenceFableRetrospectivePlanV1,
        list[dict[str, Any]],
    ],
) -> None:
    pilot, full, _ = plans_and_surfaces
    assert pilot.plan_sha256 == EXPECTED_PILOT_PLAN_SHA256
    assert full.plan_sha256 == EXPECTED_FULL_PLAN_SHA256


def test_gate_pass_requires_external_runtime_replay_and_scorer_certificate(
    plans_and_surfaces: tuple[
        EvidenceInferenceFableRetrospectivePlanV1,
        EvidenceInferenceFableRetrospectivePlanV1,
        list[dict[str, Any]],
    ],
    completed_workspace_and_certificate: tuple[Path, ScoringCompletionCertificateV1],
) -> None:
    pilot, full, _ = plans_and_surfaces
    workspace, certificate = completed_workspace_and_certificate
    decision = require_full_preflight_gate_v1(
        pilot_plan=pilot,
        full_plan=full,
        pilot_runtime_workspace=workspace,
        scoring_certificate=certificate,
    )
    assert decision.status == "full_preflight_prerequisite_satisfied"
    assert decision.blockers == []
    assert decision.separate_budget_and_provider_execution_authorization_still_required
    assert decision.provider_execution_or_spend_authority is False
    assert decision.request_intent_creation_authority is False
    assert decision.pilot_inferential_authority is False
    assert decision.confirmatory_improvement_authority is False
    assert decision.claim_release_authority is False


def test_forged_scorer_terminal_binding_fails_closed(
    plans_and_surfaces: tuple[
        EvidenceInferenceFableRetrospectivePlanV1,
        EvidenceInferenceFableRetrospectivePlanV1,
        list[dict[str, Any]],
    ],
    completed_workspace_and_certificate: tuple[Path, ScoringCompletionCertificateV1],
) -> None:
    pilot, full, _ = plans_and_surfaces
    workspace, certificate = completed_workspace_and_certificate
    tampered_payload = certificate.model_dump(mode="json", exclude={"certificate_sha256"})
    tampered_payload["runtime_terminal_sha256"] = "9" * 64
    tampered = ScoringCompletionCertificateV1.model_validate(
        {**tampered_payload, "certificate_sha256": hash_canonical(tampered_payload)}
    )
    with pytest.raises(
        EvidenceInferenceFableInferenceError,
        match="runtime_or_scorer_completion_invalid",
    ):
        evaluate_full_preflight_gate_v1(
            pilot_plan=pilot,
            full_plan=full,
            pilot_runtime_workspace=workspace,
            scoring_certificate=tampered,
        )


def test_bootstrap_is_deterministic_replayable_and_exploratory_only(
    plans_and_surfaces: tuple[
        EvidenceInferenceFableRetrospectivePlanV1,
        EvidenceInferenceFableRetrospectivePlanV1,
        list[dict[str, Any]],
    ],
    completed_workspace_and_certificate: tuple[Path, ScoringCompletionCertificateV1],
) -> None:
    pilot, _, _ = plans_and_surfaces
    workspace, certificate = completed_workspace_and_certificate
    clusters = _clusters(pilot)
    binding = derive_scoring_completion_binding_from_workspace_v1(
        plan=pilot,
        runtime_workspace=workspace,
        scoring_certificate=certificate,
    )
    first = bootstrap_paired_article_clusters_v1(
        plan=pilot,
        scoring_binding=binding,
        clusters=clusters,
    )
    replayed = validate_paired_article_cluster_bootstrap_v1(
        plan=pilot,
        scoring_binding=binding,
        clusters=clusters,
        result=first,
    )
    assert replayed == first
    assert (first.replicates, first.seed) == (20_000, 20_260_829)
    assert (first.article_cluster_count, first.question_count) == (7, 30)
    assert first.runtime_terminal_sha256 == binding.runtime_terminal_sha256
    assert first.scoring_completion_certificate_sha256 == certificate.certificate_sha256
    assert first.pilot_mechanics_only_no_inferential_authority is True
    assert first.exploratory_interval_reporting_permitted is False
    assert first.all_reference_labels_historically_opened is True
    assert first.confirmatory_gepa_improvement_authority is False
    assert first.pristine_holdout_authority is False
    assert first.scientific_claim_authority is False
    assert first.claim_release_authority is False
    assert first.provider_calls_made_by_bootstrap == 0
    assert first.benchmark_rows_or_labels_opened_by_bootstrap is False
    assert all(
        item.percentile_95_lower <= item.winner_minus_seed_difference
        <= item.percentile_95_upper
        for item in first.estimates
    )


def test_bootstrap_rejects_incomplete_or_mismatched_article_aggregates(
    plans_and_surfaces: tuple[
        EvidenceInferenceFableRetrospectivePlanV1,
        EvidenceInferenceFableRetrospectivePlanV1,
        list[dict[str, Any]],
    ],
    completed_workspace_and_certificate: tuple[Path, ScoringCompletionCertificateV1],
) -> None:
    pilot, _, _ = plans_and_surfaces
    workspace, certificate = completed_workspace_and_certificate
    binding: ScoringCompletionBindingV1 = (
        derive_scoring_completion_binding_from_workspace_v1(
            plan=pilot,
            runtime_workspace=workspace,
            scoring_certificate=certificate,
        )
    )
    clusters = _clusters(pilot)
    with pytest.raises(
        EvidenceInferenceFableInferenceError,
        match="cluster_population_incomplete",
    ):
        bootstrap_paired_article_clusters_v1(
            plan=pilot,
            scoring_binding=binding,
            clusters=clusters[:-1],
        )
    first = clusters[0]
    mismatched = freeze_article_cluster_paired_scores_v1(
        article_id=first.article_id,
        example_ids=first.example_ids[:-1],
        metric_success_counts={metric.metric: (0, 0) for metric in first.metrics},
    )
    with pytest.raises(
        EvidenceInferenceFableInferenceError,
        match="cluster_example_membership_mismatch",
    ):
        bootstrap_paired_article_clusters_v1(
            plan=pilot,
            scoring_binding=binding,
            clusters=[mismatched, *clusters[1:]],
        )
