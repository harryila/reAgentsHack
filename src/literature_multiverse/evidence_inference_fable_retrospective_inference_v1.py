"""Fail-closed pilot gate and deterministic inference for Fable retrospective v1.

This module is deliberately downstream of provider execution and private scoring. It
does not open benchmark rows, labels, provider receipts, credentials, or a network
connection. Callers supply a hash-bound terminal summary and article-aggregated paired
success counts. The only reportable inference is exploratory cross-model transfer on a
historically opened test population.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, StrictInt, model_validator

from literature_multiverse.evidence_inference_fable_retrospective_v1 import (
    EvidenceInferenceFableRetrospectivePlanV1,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel

INFERENCE_VERSION = "evidence-inference-fable-retrospective-inference-v1"
PILOT_TERMINAL_VERSION = "evidence-inference-fable-pilot-terminal-summary-v1"
GATE_VERSION = "evidence-inference-fable-full-preflight-gate-v1"
SCORING_BINDING_VERSION = "evidence-inference-fable-scoring-completion-binding-v1"
ARTICLE_SCORE_VERSION = "evidence-inference-fable-article-paired-scores-v1"
BOOTSTRAP_VERSION = "evidence-inference-fable-paired-article-bootstrap-v1"

EXPECTED_PILOT_PLAN_SHA256 = (
    "0e9637290f065e45d5e0013f0d612def76972fee76e2f6e878f44d565cb90655"
)
EXPECTED_RECOVERY_PILOT_PLAN_SHA256 = (
    "142bc85caf0edc951998a73b9f72428e4fd3637e929123d31383d53ce1a9454d"
)
EXPECTED_FULL_PLAN_SHA256 = (
    "75d94201849e815561165d06467b63c03662828285d8aa6c3fd39933a4bf5864"
)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20_260_829
RATE_QUANTUM = Decimal("0.000000001")

MetricName = Literal[
    "direction_accuracy",
    "structured_output_reliability",
    "exact_grounding_reliability",
]
METRICS: tuple[MetricName, ...] = (
    "direction_accuracy",
    "structured_output_reliability",
    "exact_grounding_reliability",
)

Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]
Count = Annotated[StrictInt, Field(ge=0)]
PositiveCount = Annotated[StrictInt, Field(ge=1)]
Rate = Annotated[Decimal, Field(ge=Decimal("0"), le=Decimal("1"), decimal_places=9)]
SignedRate = Annotated[
    Decimal,
    Field(ge=Decimal("-1"), le=Decimal("1"), decimal_places=9),
]


class EvidenceInferenceFableInferenceError(ValueError):
    """An inference, lineage, or preflight invariant failed closed."""


class _Frozen(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=False,
    )


def _self_hash(model: _Frozen, field: str, code: str) -> None:
    if getattr(model, field) != hash_canonical(
        model.model_dump(mode="json", exclude={field})
    ):
        raise ValueError(code)


def _expected_plan_sha(plan: EvidenceInferenceFableRetrospectivePlanV1) -> str:
    return {
        "pilot30_paired": EXPECTED_PILOT_PLAN_SHA256,
        "pilot30_recovery_v2_paired": EXPECTED_RECOVERY_PILOT_PLAN_SHA256,
        "full_paired": EXPECTED_FULL_PLAN_SHA256,
    }[plan.mode]


def _require_frozen_plan(plan: EvidenceInferenceFableRetrospectivePlanV1) -> None:
    if plan.plan_sha256 != _expected_plan_sha(plan):
        raise EvidenceInferenceFableInferenceError(
            "evidence_inference_fable_frozen_plan_sha_mismatch"
        )
    if (
        plan.paired_bootstrap_replicates != BOOTSTRAP_REPLICATES
        or plan.paired_bootstrap_seed != BOOTSTRAP_SEED
        or tuple(plan.paired_primary_metrics) != METRICS
        or plan.prior_exposure.all_reference_labels_historically_opened is not True
        or plan.confirmatory_gepa_improvement_claim_permitted is not False
        or plan.confirmatory_claim_authority is not False
        or plan.claim_release_authority is not False
    ):
        raise EvidenceInferenceFableInferenceError(
            "evidence_inference_fable_plan_claim_or_inference_boundary_drift"
        )


class PilotTerminalSummaryV1(_Frozen):
    terminal_version: Literal[
        "evidence-inference-fable-pilot-terminal-summary-v1"
    ] = PILOT_TERMINAL_VERSION
    pilot_plan_sha256: Sha256
    full_plan_sha256_bound_by_gate: Sha256
    source_terminal_artifact_sha256: Sha256
    source_scoring_artifact_sha256: Sha256
    scoring_completion_certificate_sha256: Sha256
    runtime_terminal_external_replay_validated: Literal[True] = True
    scoring_completion_certificate_validated: Literal[True] = True
    full_plan_matches_predeclared_frozen_sha: Literal[True] = True
    terminal_status: Literal["complete_passed", "complete_failed", "incomplete"]
    planned_request_count: Literal[14] = 14
    terminal_receipt_count: Annotated[StrictInt, Field(ge=0, le=14)]
    scored_request_count: Annotated[StrictInt, Field(ge=0, le=14)]
    unresolved_intent_count: Count
    orphan_attempt_count: Count
    duplicate_attempt_count: Count
    retry_attempt_count: Count
    all_request_and_receipt_lineage_valid: bool
    label_access_started_only_after_all_requests_terminal: bool
    full_prompt_schema_thresholds_and_roster_unchanged_by_pilot: bool
    pilot_is_mechanics_only: Literal[True] = True
    exploratory_or_confirmatory_effect_claim_from_pilot_permitted: Literal[False] = False
    summary_sha256: Sha256

    @model_validator(mode="after")
    def validate_summary(self) -> PilotTerminalSummaryV1:
        if self.terminal_status == "complete_passed" and (
            self.terminal_receipt_count != self.planned_request_count
            or self.scored_request_count != self.planned_request_count
            or self.unresolved_intent_count != 0
            or self.orphan_attempt_count != 0
            or self.duplicate_attempt_count != 0
            or self.retry_attempt_count != 0
            or not self.all_request_and_receipt_lineage_valid
            or not self.label_access_started_only_after_all_requests_terminal
            or not self.full_prompt_schema_thresholds_and_roster_unchanged_by_pilot
        ):
            raise ValueError("evidence_inference_fable_pilot_pass_summary_inconsistent")
        _self_hash(
            self,
            "summary_sha256",
            "evidence_inference_fable_pilot_summary_hash_mismatch",
        )
        return self


def _pilot_terminal_summary_from_validated_binding_v1(
    *,
    pilot_plan: EvidenceInferenceFableRetrospectivePlanV1,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    scoring_binding: ScoringCompletionBindingV1,
) -> PilotTerminalSummaryV1:
    """Normalize artifacts only after external runtime and scorer validation."""

    _require_frozen_plan(pilot_plan)
    _require_frozen_plan(full_plan)
    if pilot_plan.mode not in {
        "pilot30_paired",
        "pilot30_recovery_v2_paired",
    } or full_plan.mode != "full_paired":
        raise EvidenceInferenceFableInferenceError(
            "evidence_inference_fable_pilot_summary_plan_modes_invalid"
        )
    payload = {
        "terminal_version": PILOT_TERMINAL_VERSION,
        "pilot_plan_sha256": pilot_plan.plan_sha256,
        "full_plan_sha256_bound_by_gate": full_plan.plan_sha256,
        "source_terminal_artifact_sha256": scoring_binding.runtime_terminal_sha256,
        "source_scoring_artifact_sha256": scoring_binding.private_scored_rows_sha256,
        "scoring_completion_certificate_sha256": (
            scoring_binding.scoring_completion_certificate_sha256
        ),
        "runtime_terminal_external_replay_validated": True,
        "scoring_completion_certificate_validated": True,
        "full_plan_matches_predeclared_frozen_sha": True,
        "terminal_status": "complete_passed",
        "planned_request_count": pilot_plan.request_count,
        "terminal_receipt_count": scoring_binding.terminal_receipt_count,
        "scored_request_count": scoring_binding.terminal_receipt_count,
        "unresolved_intent_count": 0,
        "orphan_attempt_count": 0,
        "duplicate_attempt_count": 0,
        "retry_attempt_count": 0,
        "all_request_and_receipt_lineage_valid": True,
        "label_access_started_only_after_all_requests_terminal": True,
        "full_prompt_schema_thresholds_and_roster_unchanged_by_pilot": True,
        "pilot_is_mechanics_only": True,
        "exploratory_or_confirmatory_effect_claim_from_pilot_permitted": False,
    }
    return PilotTerminalSummaryV1.model_validate(
        {**payload, "summary_sha256": hash_canonical(payload)}
    )


class FullPreflightGateDecisionV1(_Frozen):
    gate_version: Literal[
        "evidence-inference-fable-full-preflight-gate-v1"
    ] = GATE_VERSION
    pilot_plan_sha256: Sha256
    full_plan_sha256: Sha256
    pilot_terminal_summary_sha256: Sha256
    source_terminal_artifact_sha256: Sha256
    status: Literal[
        "full_preflight_prerequisite_satisfied",
        "full_preflight_blocked",
    ]
    full_preflight_prerequisite_satisfied: bool
    blockers: list[str]
    separate_budget_and_provider_execution_authorization_still_required: Literal[
        True
    ] = True
    provider_execution_or_spend_authority: Literal[False] = False
    request_intent_creation_authority: Literal[False] = False
    pilot_inferential_authority: Literal[False] = False
    confirmatory_improvement_authority: Literal[False] = False
    scientific_claim_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    provider_calls_made_by_gate: Literal[0] = 0
    labels_opened_by_gate: Literal[False] = False
    gate_sha256: Sha256

    @model_validator(mode="after")
    def validate_gate(self) -> FullPreflightGateDecisionV1:
        if self.blockers != sorted(set(self.blockers)):
            raise ValueError("evidence_inference_fable_gate_blockers_not_canonical")
        passed = not self.blockers
        if (
            self.full_preflight_prerequisite_satisfied != passed
            or (self.status == "full_preflight_prerequisite_satisfied") != passed
        ):
            raise ValueError("evidence_inference_fable_gate_status_mismatch")
        _self_hash(
            self,
            "gate_sha256",
            "evidence_inference_fable_gate_hash_mismatch",
        )
        return self


def _evaluate_full_preflight_gate_from_summary_v1(
    *,
    pilot_plan: EvidenceInferenceFableRetrospectivePlanV1,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    pilot_terminal: PilotTerminalSummaryV1,
) -> FullPreflightGateDecisionV1:
    """Evaluate the pilot prerequisite without granting spend or execution authority."""

    _require_frozen_plan(pilot_plan)
    _require_frozen_plan(full_plan)
    blockers: list[str] = []
    if pilot_plan.mode not in {
        "pilot30_paired",
        "pilot30_recovery_v2_paired",
    }:
        blockers.append("pilot_plan_mode_invalid")
    if full_plan.mode != "full_paired":
        blockers.append("full_plan_mode_invalid")
    if not full_plan.pilot_preflight_required_before_full_authorization:
        blockers.append("full_plan_does_not_require_pilot_preflight")
    if pilot_terminal.pilot_plan_sha256 != pilot_plan.plan_sha256:
        blockers.append("pilot_terminal_plan_sha_mismatch")
    if (
        pilot_terminal.full_plan_sha256_bound_by_gate != full_plan.plan_sha256
    ):
        blockers.append("predeclared_full_plan_sha_mismatch")
    if pilot_terminal.terminal_status != "complete_passed":
        blockers.append("pilot_terminal_not_complete_passed")
    if pilot_terminal.terminal_receipt_count != pilot_plan.request_count:
        blockers.append("pilot_terminal_receipt_count_incomplete")
    if pilot_terminal.scored_request_count != pilot_plan.request_count:
        blockers.append("pilot_scored_request_count_incomplete")
    if pilot_terminal.unresolved_intent_count:
        blockers.append("pilot_unresolved_intents_present")
    if pilot_terminal.orphan_attempt_count:
        blockers.append("pilot_orphan_attempts_present")
    if pilot_terminal.duplicate_attempt_count:
        blockers.append("pilot_duplicate_attempts_present")
    if pilot_terminal.retry_attempt_count:
        blockers.append("pilot_retry_attempts_present")
    if not pilot_terminal.all_request_and_receipt_lineage_valid:
        blockers.append("pilot_request_or_receipt_lineage_invalid")
    if not pilot_terminal.label_access_started_only_after_all_requests_terminal:
        blockers.append("pilot_labels_opened_before_terminal_completion")
    if not pilot_terminal.full_prompt_schema_thresholds_and_roster_unchanged_by_pilot:
        blockers.append("pilot_changed_frozen_full_protocol")
    canonical_blockers = sorted(set(blockers))
    payload = {
        "gate_version": GATE_VERSION,
        "pilot_plan_sha256": pilot_plan.plan_sha256,
        "full_plan_sha256": full_plan.plan_sha256,
        "pilot_terminal_summary_sha256": pilot_terminal.summary_sha256,
        "source_terminal_artifact_sha256": (
            pilot_terminal.source_terminal_artifact_sha256
        ),
        "status": (
            "full_preflight_prerequisite_satisfied"
            if not canonical_blockers
            else "full_preflight_blocked"
        ),
        "full_preflight_prerequisite_satisfied": not canonical_blockers,
        "blockers": canonical_blockers,
        "separate_budget_and_provider_execution_authorization_still_required": True,
        "provider_execution_or_spend_authority": False,
        "request_intent_creation_authority": False,
        "pilot_inferential_authority": False,
        "confirmatory_improvement_authority": False,
        "scientific_claim_authority": False,
        "claim_release_authority": False,
        "provider_calls_made_by_gate": 0,
        "labels_opened_by_gate": False,
    }
    return FullPreflightGateDecisionV1.model_validate(
        {**payload, "gate_sha256": hash_canonical(payload)}
    )


def evaluate_full_preflight_gate_v1(
    *,
    pilot_plan: EvidenceInferenceFableRetrospectivePlanV1,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    pilot_runtime_workspace: Path,
    scoring_certificate: Any,
) -> FullPreflightGateDecisionV1:
    """Replay actual runtime state and scorer certificate before passing the gate."""

    binding = derive_scoring_completion_binding_from_workspace_v1(
        plan=pilot_plan,
        runtime_workspace=pilot_runtime_workspace,
        scoring_certificate=scoring_certificate,
    )
    summary = _pilot_terminal_summary_from_validated_binding_v1(
        pilot_plan=pilot_plan,
        full_plan=full_plan,
        scoring_binding=binding,
    )
    return _evaluate_full_preflight_gate_from_summary_v1(
        pilot_plan=pilot_plan,
        full_plan=full_plan,
        pilot_terminal=summary,
    )


def require_full_preflight_gate_v1(
    *,
    pilot_plan: EvidenceInferenceFableRetrospectivePlanV1,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    pilot_runtime_workspace: Path,
    scoring_certificate: Any,
) -> FullPreflightGateDecisionV1:
    """Return a passed prerequisite or fail closed with canonical blockers."""

    decision = evaluate_full_preflight_gate_v1(
        pilot_plan=pilot_plan,
        full_plan=full_plan,
        pilot_runtime_workspace=pilot_runtime_workspace,
        scoring_certificate=scoring_certificate,
    )
    if not decision.full_preflight_prerequisite_satisfied:
        raise EvidenceInferenceFableInferenceError(
            "evidence_inference_fable_full_preflight_blocked:"
            + ",".join(decision.blockers)
        )
    return decision


class ScoringCompletionBindingV1(_Frozen):
    binding_version: Literal[
        "evidence-inference-fable-scoring-completion-binding-v1"
    ] = SCORING_BINDING_VERSION
    plan_sha256: Sha256
    runtime_terminal_sha256: Sha256
    scoring_completion_certificate_sha256: Sha256
    private_scored_rows_sha256: Sha256
    scoring_artifact_sha256: Sha256
    planned_request_count: PositiveCount
    terminal_receipt_count: PositiveCount
    all_planned_requests_terminal: Literal[True] = True
    labels_opened_only_after_all_receipts_terminal: Literal[True] = True
    invalid_batch_counts_every_locked_question_incorrect: Literal[True] = True
    eligible_false_counts_unconditional_grounding_failure: Literal[True] = True
    scored_rows_aggregated_by_article_before_bootstrap: Literal[True] = True
    binding_sha256: Sha256

    @model_validator(mode="after")
    def validate_binding(self) -> ScoringCompletionBindingV1:
        if self.terminal_receipt_count != self.planned_request_count:
            raise ValueError("evidence_inference_fable_scoring_not_fully_terminal")
        _self_hash(
            self,
            "binding_sha256",
            "evidence_inference_fable_scoring_binding_hash_mismatch",
        )
        return self


def _freeze_scoring_completion_binding_v1(
    *,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    runtime_terminal_sha256: str,
    scoring_completion_certificate_sha256: str,
    private_scored_rows_sha256: str,
    scoring_artifact_sha256: str,
    terminal_receipt_count: int,
) -> ScoringCompletionBindingV1:
    """Bind a complete private scorer artifact before aggregate inference."""

    _require_frozen_plan(plan)
    payload = {
        "binding_version": SCORING_BINDING_VERSION,
        "plan_sha256": plan.plan_sha256,
        "runtime_terminal_sha256": runtime_terminal_sha256,
        "scoring_completion_certificate_sha256": (
            scoring_completion_certificate_sha256
        ),
        "private_scored_rows_sha256": private_scored_rows_sha256,
        "scoring_artifact_sha256": scoring_artifact_sha256,
        "planned_request_count": plan.request_count,
        "terminal_receipt_count": terminal_receipt_count,
        "all_planned_requests_terminal": True,
        "labels_opened_only_after_all_receipts_terminal": True,
        "invalid_batch_counts_every_locked_question_incorrect": True,
        "eligible_false_counts_unconditional_grounding_failure": True,
        "scored_rows_aggregated_by_article_before_bootstrap": True,
    }
    return ScoringCompletionBindingV1.model_validate(
        {**payload, "binding_sha256": hash_canonical(payload)}
    )


def derive_scoring_completion_binding_from_workspace_v1(
    *,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    runtime_workspace: Path,
    scoring_certificate: Any,
) -> ScoringCompletionBindingV1:
    """Externally replay runtime state and validate its label-free scorer certificate."""

    _require_frozen_plan(plan)
    from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
        validate_evidence_inference_fable_workspace_v1,
    )
    from literature_multiverse.evidence_inference_fable_retrospective_scoring_v1 import (
        ScoringCompletionCertificateV1,
    )

    terminal = validate_evidence_inference_fable_workspace_v1(
        workspace=runtime_workspace,
        plan=plan,
    )
    certificate = ScoringCompletionCertificateV1.model_validate(
        scoring_certificate.model_dump(mode="json")
        if isinstance(scoring_certificate, ScoringCompletionCertificateV1)
        else scoring_certificate
    )
    if (
        terminal.status != "completed"
        or terminal.completed_request_count != plan.request_count
        or not terminal.full_population_score_permitted
        or certificate.status != "complete_private_scored_rows"
        or certificate.plan_sha256 != plan.plan_sha256
        or certificate.runtime_terminal_sha256 != terminal.terminal_sha256
        or certificate.planned_request_count != plan.request_count
        or certificate.terminal_receipt_count != plan.request_count
        or not certificate.labels_loaded_only_after_complete_terminal_roster_validation
        or not certificate.all_terminal_receipt_lineage_validated
        or not certificate.invalid_batch_intention_to_evaluate
        or not certificate.eligible_false_is_unconditional_grounding_failure
        or not certificate.empty_finding_is_unconditional_grounding_failure
        or not certificate.exact_grounding_is_mechanical_not_entailment
        or certificate.provider_execution_or_spend_authority
        or certificate.confirmatory_gepa_improvement_authority
        or certificate.scientific_claim_authority
        or certificate.calibration_authority
        or certificate.claim_release_authority
    ):
        raise EvidenceInferenceFableInferenceError(
            "evidence_inference_fable_runtime_or_scorer_completion_invalid"
        )
    return _freeze_scoring_completion_binding_v1(
        plan=plan,
        runtime_terminal_sha256=terminal.terminal_sha256,
        scoring_completion_certificate_sha256=certificate.certificate_sha256,
        private_scored_rows_sha256=certificate.private_scored_rows_sha256,
        scoring_artifact_sha256=certificate.scoring_artifact_sha256,
        terminal_receipt_count=certificate.terminal_receipt_count,
    )


class PairedMetricCountsV1(_Frozen):
    metric: MetricName
    denominator: PositiveCount
    seed_success_count: Count
    winner_success_count: Count

    @model_validator(mode="after")
    def validate_counts(self) -> PairedMetricCountsV1:
        if (
            self.seed_success_count > self.denominator
            or self.winner_success_count > self.denominator
        ):
            raise ValueError("evidence_inference_fable_success_count_exceeds_denominator")
        return self


class ArticleClusterPairedScoresV1(_Frozen):
    score_version: Literal[
        "evidence-inference-fable-article-paired-scores-v1"
    ] = ARTICLE_SCORE_VERSION
    article_id: Annotated[str, Field(pattern=r"^PMC[1-9][0-9]*$")]
    example_ids: list[Annotated[str, Field(pattern=r"^ei2-prompt-[1-9][0-9]*$")]]
    question_count: PositiveCount
    metrics: Annotated[list[PairedMetricCountsV1], Field(min_length=3, max_length=3)]
    article_score_sha256: Sha256

    @model_validator(mode="after")
    def validate_scores(self) -> ArticleClusterPairedScoresV1:
        if (
            self.example_ids != sorted(set(self.example_ids))
            or self.question_count != len(self.example_ids)
            or [item.metric for item in self.metrics] != list(METRICS)
            or any(item.denominator != self.question_count for item in self.metrics)
        ):
            raise ValueError("evidence_inference_fable_article_score_contract_mismatch")
        _self_hash(
            self,
            "article_score_sha256",
            "evidence_inference_fable_article_score_hash_mismatch",
        )
        return self


def freeze_article_cluster_paired_scores_v1(
    *,
    article_id: str,
    example_ids: Sequence[str],
    metric_success_counts: Mapping[MetricName, tuple[int, int]],
) -> ArticleClusterPairedScoresV1:
    """Freeze one label-free article aggregate (seed successes, winner successes)."""

    ordered_ids = sorted(example_ids)
    if set(metric_success_counts) != set(METRICS):
        raise EvidenceInferenceFableInferenceError(
            "evidence_inference_fable_article_metric_membership_mismatch"
        )
    metrics = [
        PairedMetricCountsV1(
            metric=metric,
            denominator=len(ordered_ids),
            seed_success_count=metric_success_counts[metric][0],
            winner_success_count=metric_success_counts[metric][1],
        )
        for metric in METRICS
    ]
    payload = {
        "score_version": ARTICLE_SCORE_VERSION,
        "article_id": article_id,
        "example_ids": ordered_ids,
        "question_count": len(ordered_ids),
        "metrics": metrics,
    }
    return ArticleClusterPairedScoresV1.model_validate(
        {**payload, "article_score_sha256": hash_canonical(payload)}
    )


class MetricBootstrapEstimateV1(_Frozen):
    metric: MetricName
    seed_success_count: Count
    winner_success_count: Count
    denominator: PositiveCount
    seed_rate: Rate
    winner_rate: Rate
    winner_minus_seed_difference: SignedRate
    percentile_95_lower: SignedRate
    percentile_95_upper: SignedRate

    @model_validator(mode="after")
    def validate_estimate(self) -> MetricBootstrapEstimateV1:
        if (
            self.seed_success_count > self.denominator
            or self.winner_success_count > self.denominator
            or self.seed_rate
            != _rate(Fraction(self.seed_success_count, self.denominator))
            or self.winner_rate
            != _rate(Fraction(self.winner_success_count, self.denominator))
            or self.percentile_95_lower > self.percentile_95_upper
            or self.winner_minus_seed_difference
            != _rate(
                Fraction(
                    self.winner_success_count - self.seed_success_count,
                    self.denominator,
                )
            )
        ):
            raise ValueError("evidence_inference_fable_bootstrap_estimate_mismatch")
        return self


class PairedArticleClusterBootstrapV1(_Frozen):
    bootstrap_version: Literal[
        "evidence-inference-fable-paired-article-bootstrap-v1"
    ] = BOOTSTRAP_VERSION
    inference_pipeline_version: Literal[
        "evidence-inference-fable-retrospective-inference-v1"
    ] = INFERENCE_VERSION
    plan_sha256: Sha256
    scoring_completion_binding_sha256: Sha256
    runtime_terminal_sha256: Sha256
    scoring_completion_certificate_sha256: Sha256
    private_scored_rows_sha256: Sha256
    scoring_artifact_sha256: Sha256
    mode: Literal[
        "pilot30_paired",
        "pilot30_recovery_v2_paired",
        "full_paired",
    ]
    population: Literal[
        "pilot30_test",
        "pilot30_recovery_v2_test",
        "full_test",
    ]
    article_cluster_count: PositiveCount
    question_count: PositiveCount
    cluster_score_membership_sha256: Sha256
    replicates: Literal[20000] = BOOTSTRAP_REPLICATES
    seed: Literal[20260829] = BOOTSTRAP_SEED
    sampling_algorithm: Literal[
        "sha256_counter_modulo_n_resample_n_articles_with_replacement-v1"
    ] = "sha256_counter_modulo_n_resample_n_articles_with_replacement-v1"
    interval_method: Literal[
        "paired_article_cluster_percentile_95_nearest_rank_no_interpolation"
    ] = "paired_article_cluster_percentile_95_nearest_rank_no_interpolation"
    lower_order_index_zero_based: Literal[499] = 499
    upper_order_index_zero_based: Literal[19499] = 19499
    primary_metrics: Annotated[list[MetricName], Field(min_length=3, max_length=3)]
    estimates: Annotated[list[MetricBootstrapEstimateV1], Field(min_length=3, max_length=3)]
    all_reference_labels_historically_opened: Literal[True] = True
    interpretation: Literal[
        "exploratory_cross_model_transfer_on_historically_opened_test"
    ] = "exploratory_cross_model_transfer_on_historically_opened_test"
    exploratory_interval_reporting_permitted: bool
    pilot_mechanics_only_no_inferential_authority: bool
    confirmatory_gepa_improvement_authority: Literal[False] = False
    pristine_holdout_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    scientific_claim_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    provider_calls_made_by_bootstrap: Literal[0] = 0
    credentials_opened_by_bootstrap: Literal[False] = False
    network_opened_by_bootstrap: Literal[False] = False
    benchmark_rows_or_labels_opened_by_bootstrap: Literal[False] = False
    bootstrap_sha256: Sha256

    @model_validator(mode="after")
    def validate_bootstrap(self) -> PairedArticleClusterBootstrapV1:
        pilot = self.mode in {
            "pilot30_paired",
            "pilot30_recovery_v2_paired",
        }
        if (
            self.primary_metrics != list(METRICS)
            or [item.metric for item in self.estimates] != list(METRICS)
            or self.pilot_mechanics_only_no_inferential_authority != pilot
            or self.exploratory_interval_reporting_permitted != (not pilot)
        ):
            raise ValueError("evidence_inference_fable_bootstrap_boundary_mismatch")
        _self_hash(
            self,
            "bootstrap_sha256",
            "evidence_inference_fable_bootstrap_hash_mismatch",
        )
        return self


def _rate(value: Fraction) -> Decimal:
    return (Decimal(value.numerator) / Decimal(value.denominator)).quantize(
        RATE_QUANTUM,
        rounding=ROUND_HALF_EVEN,
    )


def _sample_index(*, plan_sha256: str, replicate: int, draw: int, size: int) -> int:
    material = (
        f"{BOOTSTRAP_VERSION}:{plan_sha256}:{BOOTSTRAP_SEED}:{replicate}:{draw}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest(), "big") % size


def _expected_article_membership(
    plan: EvidenceInferenceFableRetrospectivePlanV1,
) -> dict[str, tuple[str, ...]]:
    expected: dict[str, tuple[str, ...]] = {}
    for offset in range(0, len(plan.roster), 2):
        first, second = plan.roster[offset : offset + 2]
        if first.article_id != second.article_id or first.example_ids != second.example_ids:
            raise EvidenceInferenceFableInferenceError(
                "evidence_inference_fable_plan_pair_membership_drift"
            )
        expected[first.article_id] = tuple(first.example_ids)
    if len(expected) != plan.unique_articles:
        raise EvidenceInferenceFableInferenceError(
            "evidence_inference_fable_plan_article_count_drift"
        )
    return expected


def bootstrap_paired_article_clusters_v1(
    *,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    scoring_binding: ScoringCompletionBindingV1,
    clusters: Sequence[ArticleClusterPairedScoresV1 | Mapping[str, Any]],
) -> PairedArticleClusterBootstrapV1:
    """Compute fixed-seed paired cluster intervals from private aggregate counts."""

    _require_frozen_plan(plan)
    if (
        scoring_binding.plan_sha256 != plan.plan_sha256
        or scoring_binding.planned_request_count != plan.request_count
    ):
        raise EvidenceInferenceFableInferenceError(
            "evidence_inference_fable_scoring_binding_plan_mismatch"
        )
    parsed = [
        item
        if isinstance(item, ArticleClusterPairedScoresV1)
        else ArticleClusterPairedScoresV1.model_validate(item)
        for item in clusters
    ]
    if [item.article_id for item in parsed] != sorted(
        {item.article_id for item in parsed}
    ):
        raise EvidenceInferenceFableInferenceError(
            "evidence_inference_fable_cluster_scores_not_canonical"
        )
    expected = _expected_article_membership(plan)
    if len(parsed) != plan.unique_articles or set(expected) != {
        item.article_id for item in parsed
    }:
        raise EvidenceInferenceFableInferenceError(
            "evidence_inference_fable_cluster_population_incomplete"
        )
    for item in parsed:
        if tuple(item.example_ids) != expected[item.article_id]:
            raise EvidenceInferenceFableInferenceError(
                "evidence_inference_fable_cluster_example_membership_mismatch"
            )
    if sum(item.question_count for item in parsed) != plan.unique_examples:
        raise EvidenceInferenceFableInferenceError(
            "evidence_inference_fable_cluster_question_count_mismatch"
        )

    by_article_metric = [
        {metric.metric: metric for metric in article.metrics} for article in parsed
    ]
    replicate_differences: dict[MetricName, list[Fraction]] = {
        metric: [] for metric in METRICS
    }
    size = len(parsed)
    for replicate in range(BOOTSTRAP_REPLICATES):
        denominator = 0
        seed_success = {metric: 0 for metric in METRICS}
        winner_success = {metric: 0 for metric in METRICS}
        for draw in range(size):
            index = _sample_index(
                plan_sha256=plan.plan_sha256,
                replicate=replicate,
                draw=draw,
                size=size,
            )
            denominator += parsed[index].question_count
            for metric in METRICS:
                counts = by_article_metric[index][metric]
                seed_success[metric] += counts.seed_success_count
                winner_success[metric] += counts.winner_success_count
        for metric in METRICS:
            replicate_differences[metric].append(
                Fraction(winner_success[metric] - seed_success[metric], denominator)
            )

    estimates: list[MetricBootstrapEstimateV1] = []
    total_denominator = sum(item.question_count for item in parsed)
    for metric in METRICS:
        seed_total = sum(
            by_article_metric[index][metric].seed_success_count
            for index in range(size)
        )
        winner_total = sum(
            by_article_metric[index][metric].winner_success_count
            for index in range(size)
        )
        ordered = sorted(replicate_differences[metric])
        estimates.append(
            MetricBootstrapEstimateV1(
                metric=metric,
                seed_success_count=seed_total,
                winner_success_count=winner_total,
                denominator=total_denominator,
                seed_rate=_rate(Fraction(seed_total, total_denominator)),
                winner_rate=_rate(Fraction(winner_total, total_denominator)),
                winner_minus_seed_difference=_rate(
                    Fraction(winner_total - seed_total, total_denominator)
                ),
                percentile_95_lower=_rate(ordered[499]),
                percentile_95_upper=_rate(ordered[19_499]),
            )
        )
    payload = {
        "bootstrap_version": BOOTSTRAP_VERSION,
        "inference_pipeline_version": INFERENCE_VERSION,
        "plan_sha256": plan.plan_sha256,
        "scoring_completion_binding_sha256": scoring_binding.binding_sha256,
        "runtime_terminal_sha256": scoring_binding.runtime_terminal_sha256,
        "scoring_completion_certificate_sha256": (
            scoring_binding.scoring_completion_certificate_sha256
        ),
        "private_scored_rows_sha256": scoring_binding.private_scored_rows_sha256,
        "scoring_artifact_sha256": scoring_binding.scoring_artifact_sha256,
        "mode": plan.mode,
        "population": plan.population,
        "article_cluster_count": len(parsed),
        "question_count": total_denominator,
        "cluster_score_membership_sha256": hash_canonical(
            [item.article_score_sha256 for item in parsed]
        ),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "sampling_algorithm": (
            "sha256_counter_modulo_n_resample_n_articles_with_replacement-v1"
        ),
        "interval_method": (
            "paired_article_cluster_percentile_95_nearest_rank_no_interpolation"
        ),
        "lower_order_index_zero_based": 499,
        "upper_order_index_zero_based": 19499,
        "primary_metrics": list(METRICS),
        "estimates": estimates,
        "all_reference_labels_historically_opened": True,
        "interpretation": (
            "exploratory_cross_model_transfer_on_historically_opened_test"
        ),
        "exploratory_interval_reporting_permitted": plan.mode == "full_paired",
        "pilot_mechanics_only_no_inferential_authority": plan.mode
        in {"pilot30_paired", "pilot30_recovery_v2_paired"},
        "confirmatory_gepa_improvement_authority": False,
        "pristine_holdout_authority": False,
        "calibration_authority": False,
        "scientific_claim_authority": False,
        "claim_release_authority": False,
        "provider_calls_made_by_bootstrap": 0,
        "credentials_opened_by_bootstrap": False,
        "network_opened_by_bootstrap": False,
        "benchmark_rows_or_labels_opened_by_bootstrap": False,
    }
    return PairedArticleClusterBootstrapV1.model_validate(
        {**payload, "bootstrap_sha256": hash_canonical(payload)}
    )


def validate_paired_article_cluster_bootstrap_v1(
    *,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    scoring_binding: ScoringCompletionBindingV1,
    clusters: Sequence[ArticleClusterPairedScoresV1 | Mapping[str, Any]],
    result: PairedArticleClusterBootstrapV1 | Mapping[str, Any],
) -> PairedArticleClusterBootstrapV1:
    """Externally recompute all 20,000 replicates and require byte-semantic equality."""

    observed = (
        result
        if isinstance(result, PairedArticleClusterBootstrapV1)
        else PairedArticleClusterBootstrapV1.model_validate(result)
    )
    expected = bootstrap_paired_article_clusters_v1(
        plan=plan,
        scoring_binding=scoring_binding,
        clusters=clusters,
    )
    if observed != expected:
        raise EvidenceInferenceFableInferenceError(
            "evidence_inference_fable_bootstrap_external_replay_mismatch"
        )
    return observed


__all__ = [
    "EXPECTED_FULL_PLAN_SHA256",
    "EXPECTED_PILOT_PLAN_SHA256",
    "EXPECTED_RECOVERY_PILOT_PLAN_SHA256",
    "ArticleClusterPairedScoresV1",
    "EvidenceInferenceFableInferenceError",
    "FullPreflightGateDecisionV1",
    "MetricBootstrapEstimateV1",
    "PairedArticleClusterBootstrapV1",
    "PairedMetricCountsV1",
    "PilotTerminalSummaryV1",
    "ScoringCompletionBindingV1",
    "bootstrap_paired_article_clusters_v1",
    "derive_scoring_completion_binding_from_workspace_v1",
    "evaluate_full_preflight_gate_v1",
    "freeze_article_cluster_paired_scores_v1",
    "require_full_preflight_gate_v1",
    "validate_paired_article_cluster_bootstrap_v1",
]
