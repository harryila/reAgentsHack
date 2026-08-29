"""Private post-terminal scoring for the retrospective Evidence Inference Fable run.

Provider receipts are validated as a complete, exactly-once roster before the supplied
label loader is called.  Any unusable provider response or invalid article batch assigns
zero to every locked question in that request.  An empty finding or ``eligible=false``
is an unconditional grounding failure on this all-eligible benchmark population.

The private report retains row identities, reference directions, parsed items, and exact
grounding diagnostics.  Its public projection contains only closed aggregate fields and
cryptographic bindings to the private report and frozen plan.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.validators import validator_for
from pydantic import ConfigDict, Field, StrictInt, model_validator

from literature_multiverse.evidence_inference import _content_lines, _source_lines
from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    EvidenceInferenceFableIntentV1,
    EvidenceInferenceFablePreparedRuntimeV1,
    EvidenceInferenceFableReceiptV1,
    EvidenceInferenceFableTerminalV1,
    parse_evidence_inference_fable_budget_authorization_v1,
    validate_evidence_inference_fable_workspace_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_inference_v1 import (
    PairedArticleClusterBootstrapV1,
    bootstrap_paired_article_clusters_v1,
    derive_scoring_completion_binding_from_workspace_v1,
    freeze_article_cluster_paired_scores_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_v1 import (
    DEFAULT_CONFIG_PATH,
    ArticleBatchRequestV1,
    EvidenceInferenceFableRetrospectivePlanV1,
    _output_schema,
)
from literature_multiverse.grounding import GroundingContractError, ground_evidence
from literature_multiverse.lineage import atomic_write_json, hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel

SCORING_VERSION = "evidence-inference-fable-retrospective-scoring-v1"
RECEIPT_VERSION = "evidence-inference-fable-terminal-scoring-receipt-v1"
LABEL_BUNDLE_VERSION = "evidence-inference-fable-private-reference-labels-v1"
PRIVATE_SCORED_ROWS_VERSION = "evidence-inference-fable-private-scored-rows-v1"
COMPLETION_CERTIFICATE_VERSION = (
    "evidence-inference-fable-scoring-completion-certificate-v1"
)
PRIVATE_REPORT_VERSION = "evidence-inference-fable-private-paired-report-v1"
PUBLIC_SUMMARY_VERSION = "evidence-inference-fable-public-paired-summary-v1"

PRIMARY_METRICS = (
    "direction_accuracy",
    "structured_output_reliability",
    "exact_grounding_reliability",
)
ARMS = ("seed", "winner")

Direction = Literal["increase", "no_effect", "decrease"]
Arm = Literal["seed", "winner"]
MetricName = Literal[
    "direction_accuracy",
    "structured_output_reliability",
    "exact_grounding_reliability",
]
ProviderOutcome = Literal[
    "provider_response",
    "provider_response_unusable",
    "transport_failed_or_ambiguous",
]
CostBasis = Literal[
    "reported_usage",
    "full_context_hard_liability_unknown_usage",
    "certified_provider_token_liability_unknown_usage",
    "certified_provider_token_plus_headroom_liability_unknown_usage",
]
PrimaryFailure = Literal[
    "success",
    "provider_attempt_failure",
    "invalid_article_batch",
    "eligible_false",
    "missing_finding",
    "direction_incorrect",
    "exact_grounding_invalid",
]


class EvidenceInferenceFableScoringError(ValueError):
    """A terminal, label, grounding, scoring, or report boundary failed closed."""


class _Frozen(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]
Count = Annotated[StrictInt, Field(ge=0)]
PositiveCount = Annotated[StrictInt, Field(ge=1)]
Binary = Literal[0, 1]


def _self_hash(model: _Frozen, field: str, code: str) -> None:
    if getattr(model, field) != hash_canonical(
        model.model_dump(mode="json", exclude={field})
    ):
        raise ValueError(code)


class ProviderUsageV1(_Frozen):
    input_tokens: Count
    output_tokens: Count
    cache_creation_input_tokens: Literal[0] = 0
    cache_read_input_tokens: Literal[0] = 0


class TerminalScoringReceiptV1(_Frozen):
    receipt_version: Literal["evidence-inference-fable-terminal-scoring-receipt-v1"] = (
        RECEIPT_VERSION
    )
    scoring_version: Literal["evidence-inference-fable-retrospective-scoring-v1"] = (
        SCORING_VERSION
    )
    plan_sha256: Sha256
    runtime_terminal_sha256: Sha256
    runtime_intent_sha256: Sha256
    runtime_receipt_sha256: Sha256
    runtime_provider_result_sha256: Sha256
    request_sha256: Sha256
    request_key: str
    article_id: str
    arm: Arm
    provider_outcome: ProviderOutcome
    transport_attempt_count: Literal[1] = 1
    sdk_retry_count: Literal[0] = 0
    application_retry_count: Literal[0] = 0
    terminal_before_reference_label_access: Literal[True] = True
    parsed_batch: dict[str, Any] | None
    usage: ProviderUsageV1 | None
    accounted_cost_basis: CostBasis
    accounted_cost_usd_micros: Count
    provider_result_sha256: Sha256
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> TerminalScoringReceiptV1:
        if self.provider_outcome == "provider_response":
            if (
                self.parsed_batch is None
                or self.usage is None
                or self.accounted_cost_basis != "reported_usage"
            ):
                raise ValueError("ei_fable_terminal_provider_response_invalid")
        elif self.provider_outcome == "provider_response_unusable":
            if self.parsed_batch is not None:
                raise ValueError("ei_fable_terminal_unusable_response_has_batch")
        elif self.parsed_batch is not None or self.usage is not None:
            raise ValueError("ei_fable_terminal_transport_failure_has_response")
        if self.usage is not None:
            expected_cost = self.usage.input_tokens * 10 + self.usage.output_tokens * 50
            if (
                self.accounted_cost_basis != "reported_usage"
                or self.accounted_cost_usd_micros != expected_cost
            ):
                raise ValueError("ei_fable_terminal_reported_cost_mismatch")
        elif self.accounted_cost_basis not in {
            "full_context_hard_liability_unknown_usage",
            "certified_provider_token_liability_unknown_usage",
            "certified_provider_token_plus_headroom_liability_unknown_usage",
        }:
            raise ValueError("ei_fable_terminal_unknown_usage_cost_basis_invalid")
        expected_result = hash_canonical(
            {
                "request_sha256": self.request_sha256,
                "provider_outcome": self.provider_outcome,
                "parsed_batch": self.parsed_batch,
                "usage": (
                    self.usage.model_dump(mode="json") if self.usage is not None else None
                ),
                "accounted_cost_basis": self.accounted_cost_basis,
                "accounted_cost_usd_micros": self.accounted_cost_usd_micros,
            }
        )
        if self.provider_result_sha256 != expected_result:
            raise ValueError("ei_fable_terminal_provider_result_hash_mismatch")
        _self_hash(self, "receipt_sha256", "ei_fable_terminal_receipt_hash_mismatch")
        return self


class PrivateReferenceLabelV1(_Frozen):
    example_id: str
    article_id: str
    expected_direction: Direction


class PrivateReferenceLabelBundleV1(_Frozen):
    label_bundle_version: Literal["evidence-inference-fable-private-reference-labels-v1"] = (
        LABEL_BUNDLE_VERSION
    )
    plan_sha256: Sha256
    population: Literal[
        "pilot30_test",
        "pilot30_recovery_v2_test",
        "full_test",
    ]
    examples: PositiveCount
    articles: PositiveCount
    labels: list[PrivateReferenceLabelV1]
    label_membership_sha256: Sha256
    label_bundle_sha256: Sha256

    @model_validator(mode="after")
    def validate_bundle(self) -> PrivateReferenceLabelBundleV1:
        if (
            self.labels
            != sorted(self.labels, key=lambda item: item.example_id)
            or len({item.example_id for item in self.labels}) != len(self.labels)
            or self.examples != len(self.labels)
            or self.articles != len({item.article_id for item in self.labels})
            or self.label_membership_sha256
            != hash_canonical([item.model_dump(mode="json") for item in self.labels])
        ):
            raise ValueError("ei_fable_private_label_bundle_alias_mismatch")
        _self_hash(self, "label_bundle_sha256", "ei_fable_private_label_bundle_hash_mismatch")
        return self


class ArmRowScoreV1(_Frozen):
    provider_outcome: ProviderOutcome
    batch_structured_output_valid: bool
    predicted_eligible: bool | None
    predicted_direction: Direction | None
    structured_output_reliability: Binary
    direction_accuracy: Binary
    exact_grounding_reliability: Binary
    direction_and_grounding_joint: Binary
    conditional_grounding_evaluated: Binary
    conditional_exact_grounding: Binary
    grounding_status: Literal["exact", "missing", "mismatch", "unverifiable"] | None
    primary_failure: PrimaryFailure
    batch_validation_error: str | None
    parsed_item: dict[str, Any] | None
    grounding_detail: dict[str, Any] | None

    @model_validator(mode="after")
    def validate_score(self) -> ArmRowScoreV1:
        if (
            self.direction_and_grounding_joint
            != self.direction_accuracy * self.exact_grounding_reliability
            or self.conditional_exact_grounding > self.conditional_grounding_evaluated
            or self.exact_grounding_reliability > self.conditional_grounding_evaluated
        ):
            raise ValueError("ei_fable_arm_row_score_alias_mismatch")
        if not self.batch_structured_output_valid and any(
            (
                self.structured_output_reliability,
                self.direction_accuracy,
                self.exact_grounding_reliability,
                self.conditional_grounding_evaluated,
            )
        ):
            raise ValueError("ei_fable_invalid_batch_received_credit")
        if self.predicted_eligible is False and (
            self.direction_accuracy
            or self.exact_grounding_reliability
            or self.conditional_grounding_evaluated
        ):
            raise ValueError("ei_fable_ineligible_prediction_received_credit")
        return self


class PrivatePairedRowV1(_Frozen):
    example_id: str
    article_id: str
    expected_direction: Direction
    seed: ArmRowScoreV1
    winner: ArmRowScoreV1
    row_sha256: Sha256

    @model_validator(mode="after")
    def validate_row(self) -> PrivatePairedRowV1:
        _self_hash(self, "row_sha256", "ei_fable_private_row_hash_mismatch")
        return self


class ArmAggregateV1(_Frozen):
    requests: PositiveCount
    question_evaluations: PositiveCount
    valid_batch_requests: Count
    provider_outcome_counts: dict[ProviderOutcome, Count]
    primary_failure_counts: dict[PrimaryFailure, Count]
    metric_success_counts: dict[MetricName, Count]
    metric_rates: dict[MetricName, Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]]
    conditional_grounding_denominator: Count
    conditional_exact_grounding_numerator: Count
    conditional_exact_grounding_rate: (
        Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None
    )
    usage_reported_requests: Count
    usage_missing_requests: Count
    input_tokens: Count
    output_tokens: Count
    accounted_cost_usd_micros: Count

    @model_validator(mode="after")
    def validate_aggregate(self) -> ArmAggregateV1:
        if (
            sum(self.provider_outcome_counts.values()) != self.requests
            or sum(self.primary_failure_counts.values()) != self.question_evaluations
            or set(self.metric_success_counts) != set(PRIMARY_METRICS)
            or set(self.metric_rates) != set(PRIMARY_METRICS)
            or any(
                self.metric_rates[metric]
                != self.metric_success_counts[metric] / self.question_evaluations
                for metric in PRIMARY_METRICS
            )
            or self.usage_reported_requests + self.usage_missing_requests != self.requests
            or self.conditional_exact_grounding_numerator
            > self.conditional_grounding_denominator
        ):
            raise ValueError("ei_fable_arm_aggregate_alias_mismatch")
        expected_conditional = (
            self.conditional_exact_grounding_numerator
            / self.conditional_grounding_denominator
            if self.conditional_grounding_denominator
            else None
        )
        if self.conditional_exact_grounding_rate != expected_conditional:
            raise ValueError("ei_fable_conditional_grounding_rate_mismatch")
        return self


class PrivateScoredRowsV1(_Frozen):
    scored_rows_version: Literal["evidence-inference-fable-private-scored-rows-v1"] = (
        PRIVATE_SCORED_ROWS_VERSION
    )
    scoring_version: Literal["evidence-inference-fable-retrospective-scoring-v1"] = (
        SCORING_VERSION
    )
    status: Literal["complete_private_scored_rows"] = (
        "complete_private_scored_rows"
    )
    plan_sha256: Sha256
    runtime_terminal_sha256: Sha256
    population: Literal[
        "pilot30_test",
        "pilot30_recovery_v2_test",
        "full_test",
    ]
    label_bundle_sha256: Sha256
    receipt_membership_sha256: Sha256
    scoring_artifact_sha256: Sha256
    examples: PositiveCount
    articles: PositiveCount
    requests: PositiveCount
    rows: list[PrivatePairedRowV1]
    arms: dict[Arm, ArmAggregateV1]
    labels_loaded_only_after_complete_terminal_roster_validation: Literal[True] = True
    invalid_batch_intention_to_evaluate: Literal[True] = True
    eligible_false_is_unconditional_grounding_failure: Literal[True] = True
    empty_finding_is_unconditional_grounding_failure: Literal[True] = True
    exact_grounding_is_mechanical_not_entailment: Literal[True] = True
    all_reference_labels_historically_opened: Literal[True] = True
    exploratory_cross_model_transfer_only: Literal[True] = True
    confirmatory_gepa_improvement_claim_permitted: Literal[False] = False
    gepa_optimization_improvement_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    generalization_authority: Literal[False] = False
    eligibility_metric_claim_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    private_scored_rows_sha256: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> PrivateScoredRowsV1:
        if (
            self.rows != sorted(self.rows, key=lambda item: item.example_id)
            or self.examples != len(self.rows)
            or self.articles != len({item.article_id for item in self.rows})
            or set(self.arms) != set(ARMS)
            or any(self.arms[arm].question_evaluations != self.examples for arm in ARMS)
            or self.requests % 2
            or any(self.arms[arm].requests != self.requests // 2 for arm in ARMS)
            or any(
                self.arms[arm].metric_success_counts[metric]
                != sum(int(getattr(getattr(row, arm), metric)) for row in self.rows)
                for arm in ARMS
                for metric in PRIMARY_METRICS
            )
        ):
            raise ValueError("ei_fable_private_scored_rows_alias_mismatch")
        _self_hash(
            self,
            "private_scored_rows_sha256",
            "ei_fable_private_scored_rows_hash_mismatch",
        )
        return self


class ScoringCompletionCertificateV1(_Frozen):
    certificate_version: Literal[
        "evidence-inference-fable-scoring-completion-certificate-v1"
    ] = COMPLETION_CERTIFICATE_VERSION
    scoring_version: Literal["evidence-inference-fable-retrospective-scoring-v1"] = (
        SCORING_VERSION
    )
    status: Literal["complete_private_scored_rows"] = "complete_private_scored_rows"
    plan_sha256: Sha256
    runtime_terminal_sha256: Sha256
    private_scored_rows_sha256: Sha256
    scoring_artifact_sha256: Sha256
    receipt_membership_sha256: Sha256
    planned_request_count: PositiveCount
    terminal_receipt_count: PositiveCount
    labels_loaded_only_after_complete_terminal_roster_validation: Literal[True] = True
    all_terminal_receipt_lineage_validated: Literal[True] = True
    invalid_batch_intention_to_evaluate: Literal[True] = True
    eligible_false_is_unconditional_grounding_failure: Literal[True] = True
    empty_finding_is_unconditional_grounding_failure: Literal[True] = True
    exact_grounding_is_mechanical_not_entailment: Literal[True] = True
    provider_execution_or_spend_authority: Literal[False] = False
    confirmatory_gepa_improvement_authority: Literal[False] = False
    scientific_claim_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    certificate_sha256: Sha256

    @model_validator(mode="after")
    def validate_certificate(self) -> ScoringCompletionCertificateV1:
        if self.terminal_receipt_count != self.planned_request_count:
            raise ValueError("ei_fable_scoring_certificate_terminal_roster_incomplete")
        _self_hash(
            self,
            "certificate_sha256",
            "ei_fable_scoring_certificate_hash_mismatch",
        )
        return self


class PrivatePairedReportV1(_Frozen):
    private_report_version: Literal["evidence-inference-fable-private-paired-report-v1"] = (
        PRIVATE_REPORT_VERSION
    )
    scoring_version: Literal["evidence-inference-fable-retrospective-scoring-v1"] = (
        SCORING_VERSION
    )
    status: Literal["complete_exploratory_retrospective_paired_score"] = (
        "complete_exploratory_retrospective_paired_score"
    )
    plan_sha256: Sha256
    runtime_terminal_sha256: Sha256
    population: Literal[
        "pilot30_test",
        "pilot30_recovery_v2_test",
        "full_test",
    ]
    scored_rows: PrivateScoredRowsV1
    completion_certificate: ScoringCompletionCertificateV1
    paired_article_cluster_bootstrap: PairedArticleClusterBootstrapV1
    private_report_sha256: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> PrivatePairedReportV1:
        estimates = {
            item.metric: item
            for item in self.paired_article_cluster_bootstrap.estimates
        }
        if (
            self.scored_rows.plan_sha256 != self.plan_sha256
            or self.scored_rows.runtime_terminal_sha256
            != self.runtime_terminal_sha256
            or self.scored_rows.population != self.population
            or self.completion_certificate.plan_sha256 != self.plan_sha256
            or self.completion_certificate.runtime_terminal_sha256
            != self.runtime_terminal_sha256
            or self.completion_certificate.private_scored_rows_sha256
            != self.scored_rows.private_scored_rows_sha256
            or self.completion_certificate.scoring_artifact_sha256
            != self.scored_rows.scoring_artifact_sha256
            or self.paired_article_cluster_bootstrap.plan_sha256 != self.plan_sha256
            or self.paired_article_cluster_bootstrap.runtime_terminal_sha256
            != self.runtime_terminal_sha256
            or self.paired_article_cluster_bootstrap.scoring_completion_certificate_sha256
            != self.completion_certificate.certificate_sha256
            or self.paired_article_cluster_bootstrap.private_scored_rows_sha256
            != self.scored_rows.private_scored_rows_sha256
            or self.paired_article_cluster_bootstrap.scoring_artifact_sha256
            != self.scored_rows.scoring_artifact_sha256
            or self.paired_article_cluster_bootstrap.population != self.population
            or self.paired_article_cluster_bootstrap.question_count
            != self.scored_rows.examples
            or self.paired_article_cluster_bootstrap.article_cluster_count
            != self.scored_rows.articles
            or any(
                estimates[metric].seed_success_count
                != self.scored_rows.arms["seed"].metric_success_counts[metric]
                or estimates[metric].winner_success_count
                != self.scored_rows.arms["winner"].metric_success_counts[metric]
                or estimates[metric].denominator != self.scored_rows.examples
                for metric in PRIMARY_METRICS
            )
        ):
            raise ValueError("ei_fable_private_report_binding_mismatch")
        _self_hash(self, "private_report_sha256", "ei_fable_private_report_hash_mismatch")
        return self


class PublicPairedSummaryV1(_Frozen):
    public_summary_version: Literal["evidence-inference-fable-public-paired-summary-v1"] = (
        PUBLIC_SUMMARY_VERSION
    )
    scoring_version: Literal["evidence-inference-fable-retrospective-scoring-v1"] = (
        SCORING_VERSION
    )
    status: Literal["aggregate_only_exploratory_retrospective_paired_score"] = (
        "aggregate_only_exploratory_retrospective_paired_score"
    )
    private_report_sha256: Sha256
    completion_certificate_sha256: Sha256
    plan_sha256: Sha256
    runtime_terminal_sha256: Sha256
    population: Literal[
        "pilot30_test",
        "pilot30_recovery_v2_test",
        "full_test",
    ]
    examples: PositiveCount
    articles: PositiveCount
    requests: PositiveCount
    arms: dict[Arm, ArmAggregateV1]
    paired_article_cluster_bootstrap: PairedArticleClusterBootstrapV1
    contains_article_or_question_text: Literal[False] = False
    contains_article_or_example_identifiers: Literal[False] = False
    contains_reference_or_per_example_labels: Literal[False] = False
    contains_raw_or_per_example_predictions: Literal[False] = False
    contains_evidence_quotes_or_line_references: Literal[False] = False
    contains_absolute_paths: Literal[False] = False
    all_reference_labels_historically_opened: Literal[True] = True
    exploratory_cross_model_transfer_only: Literal[True] = True
    confirmatory_gepa_improvement_claim_permitted: Literal[False] = False
    gepa_optimization_improvement_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    generalization_authority: Literal[False] = False
    eligibility_metric_claim_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    required_caveats: list[
        Literal[
            "historically_opened_test_not_pristine_or_confirmatory",
            "cross_model_and_article_batched_interface_transfer_only",
            "formal_exact_grounding_is_not_semantic_entailment",
            "all_retained_examples_are_eligibility_positive",
        ]
    ]
    public_summary_sha256: Sha256

    @model_validator(mode="after")
    def validate_summary(self) -> PublicPairedSummaryV1:
        estimates = {
            item.metric: item
            for item in self.paired_article_cluster_bootstrap.estimates
        }
        if (
            set(self.arms) != set(ARMS)
            or self.paired_article_cluster_bootstrap.plan_sha256 != self.plan_sha256
            or self.paired_article_cluster_bootstrap.runtime_terminal_sha256
            != self.runtime_terminal_sha256
            or self.paired_article_cluster_bootstrap.scoring_completion_certificate_sha256
            != self.completion_certificate_sha256
            or self.paired_article_cluster_bootstrap.question_count != self.examples
            or self.paired_article_cluster_bootstrap.article_cluster_count
            != self.articles
            or any(
                estimates[metric].seed_success_count
                != self.arms["seed"].metric_success_counts[metric]
                or estimates[metric].winner_success_count
                != self.arms["winner"].metric_success_counts[metric]
                or estimates[metric].denominator != self.examples
                for metric in PRIMARY_METRICS
            )
            or self.required_caveats
            != [
                "historically_opened_test_not_pristine_or_confirmatory",
                "cross_model_and_article_batched_interface_transfer_only",
                "formal_exact_grounding_is_not_semantic_entailment",
                "all_retained_examples_are_eligibility_positive",
            ]
        ):
            raise ValueError("ei_fable_public_summary_alias_mismatch")
        _self_hash(self, "public_summary_sha256", "ei_fable_public_summary_hash_mismatch")
        return self


class ResultsSourceLoader(Protocol):
    def __call__(self, article_id: str) -> Mapping[str, Any]: ...


def _canonical_plan(
    plan: EvidenceInferenceFableRetrospectivePlanV1,
) -> EvidenceInferenceFableRetrospectivePlanV1:
    return EvidenceInferenceFableRetrospectivePlanV1.model_validate(
        plan.model_dump(mode="json")
    )


def _request_by_sha(
    plan: EvidenceInferenceFableRetrospectivePlanV1,
) -> dict[str, ArticleBatchRequestV1]:
    return {item.request_sha256: item for item in plan.roster}


def _adapt_runtime_scoring_receipt_v1(
    *,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    runtime_terminal: EvidenceInferenceFableTerminalV1,
    runtime_intent: EvidenceInferenceFableIntentV1,
    runtime_receipt: EvidenceInferenceFableReceiptV1,
    authorized_request_liability_usd_micros: int,
    unknown_usage_cost_basis: Literal[
        "full_context_hard_liability_unknown_usage",
        "certified_provider_token_liability_unknown_usage",
        "certified_provider_token_plus_headroom_liability_unknown_usage",
    ],
) -> TerminalScoringReceiptV1:
    """Adapt one validated runtime receipt; arbitrary scoring receipts are forbidden."""

    canonical = _canonical_plan(plan)
    terminal = EvidenceInferenceFableTerminalV1.model_validate(runtime_terminal)
    intent = EvidenceInferenceFableIntentV1.model_validate(runtime_intent)
    receipt = EvidenceInferenceFableReceiptV1.model_validate(runtime_receipt)
    by_key = {item.request_key: item for item in canonical.roster}
    request = by_key.get(receipt.request_key)
    result = receipt.provider_result
    if (
        terminal.status != "completed"
        or not terminal.full_population_score_permitted
        or terminal.completed_request_count != canonical.request_count
        or request is None
        or intent.request_key != request.request_key
        or intent.surface.article_request_sha256 != request.request_sha256
        or receipt.intent_sha256 != intent.intent_sha256
        or result.request_key != request.request_key
        or result.surface_sha256 != intent.surface.surface_sha256
        or receipt.locked_question_count != request.question_count
        or receipt.locked_questions_scored_incorrect
        != (request.question_count if result.outcome == "failed" else 0)
    ):
        raise EvidenceInferenceFableScoringError(
            "runtime_receipt_terminal_or_request_binding_mismatch"
        )
    if result.cost_basis == "reported_usage":
        if (
            result.input_tokens is None
            or result.output_tokens is None
            or result.reported_cost_usd_micros is None
        ):
            raise EvidenceInferenceFableScoringError("runtime_receipt_usage_missing")
        expected_cost = result.input_tokens * 10 + result.output_tokens * 50
        if (
            result.reported_cost_usd_micros != expected_cost
            or result.charged_cost_usd_micros != expected_cost
        ):
            raise EvidenceInferenceFableScoringError(
                "runtime_receipt_reported_cost_mismatch"
            )
        canonical_usage: ProviderUsageV1 | None = ProviderUsageV1(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        cost_basis: CostBasis = "reported_usage"
    else:
        if (
            result.input_tokens is not None
            or result.output_tokens is not None
            or result.reported_cost_usd_micros is not None
            or result.charged_cost_usd_micros
            != authorized_request_liability_usd_micros
        ):
            raise EvidenceInferenceFableScoringError(
                "runtime_receipt_unknown_usage_liability_mismatch"
            )
        canonical_usage = None
        cost_basis = unknown_usage_cost_basis
    if result.outcome == "completed":
        provider_outcome: ProviderOutcome = "provider_response"
    elif result.failure_code in {
        "provider_call_raised_after_durable_intent",
        "provider_result_invalid_after_return",
    }:
        provider_outcome = "transport_failed_or_ambiguous"
    else:
        provider_outcome = "provider_response_unusable"
    parsed = deepcopy(result.parsed_json)
    cost = result.charged_cost_usd_micros
    result_payload = {
        "request_sha256": request.request_sha256,
        "provider_outcome": provider_outcome,
        "parsed_batch": parsed,
        "usage": (
            canonical_usage.model_dump(mode="json") if canonical_usage is not None else None
        ),
        "accounted_cost_basis": cost_basis,
        "accounted_cost_usd_micros": cost,
    }
    payload = {
        "receipt_version": RECEIPT_VERSION,
        "scoring_version": SCORING_VERSION,
        "plan_sha256": canonical.plan_sha256,
        "runtime_terminal_sha256": terminal.terminal_sha256,
        "runtime_intent_sha256": intent.intent_sha256,
        "runtime_receipt_sha256": receipt.receipt_sha256,
        "runtime_provider_result_sha256": result.result_sha256,
        "request_sha256": request.request_sha256,
        "request_key": request.request_key,
        "article_id": request.article_id,
        "arm": request.arm,
        "provider_outcome": provider_outcome,
        "transport_attempt_count": 1,
        "sdk_retry_count": 0,
        "application_retry_count": 0,
        "terminal_before_reference_label_access": True,
        "parsed_batch": parsed,
        "usage": canonical_usage,
        "accounted_cost_basis": cost_basis,
        "accounted_cost_usd_micros": cost,
        "provider_result_sha256": hash_canonical(result_payload),
    }
    return TerminalScoringReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def _read_runtime_json(path: Path, workspace: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceInferenceFableScoringError("runtime_scoring_artifact_unsafe")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(workspace)
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidenceInferenceFableScoringError(
            "runtime_scoring_artifact_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise EvidenceInferenceFableScoringError("runtime_scoring_artifact_not_object")
    return value


def replay_terminal_scoring_receipts_v1(
    *,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    runtime_workspace: Path,
) -> tuple[EvidenceInferenceFableTerminalV1, tuple[TerminalScoringReceiptV1, ...]]:
    """Externally replay the completed runtime and adapt its exact receipt roster."""

    canonical = _canonical_plan(plan)
    workspace = runtime_workspace.resolve(strict=True)
    terminal_before = validate_evidence_inference_fable_workspace_v1(
        workspace=workspace,
        plan=canonical,
    )
    if (
        terminal_before.status != "completed"
        or not terminal_before.full_population_score_permitted
        or terminal_before.completed_request_count != canonical.request_count
    ):
        raise EvidenceInferenceFableScoringError(
            "completed_runtime_required_for_full_population_score"
        )
    prepared = EvidenceInferenceFablePreparedRuntimeV1.model_validate(
        _read_runtime_json(workspace / "00-prepared.json", workspace)
    )
    authorization = parse_evidence_inference_fable_budget_authorization_v1(
        _read_runtime_json(workspace / "01-authorization.json", workspace)
    )
    if (
        prepared.retrospective_plan_sha256 != canonical.plan_sha256
        or prepared.request_roster_sha256 != canonical.request_roster_sha256
        or prepared.prepared_sha256 != terminal_before.prepared_sha256
    ):
        raise EvidenceInferenceFableScoringError("runtime_prepared_plan_binding_mismatch")
    surfaces = {item.request_key: item for item in prepared.surfaces}
    if authorization.authorization_sha256 != terminal_before.authorization_sha256:
        raise EvidenceInferenceFableScoringError(
            "runtime_authorization_terminal_binding_mismatch"
        )

    def authorized_liability(request: ArticleBatchRequestV1) -> int:
        if authorization.liability_basis != "full_context_fallback":
            try:
                return authorization.certified_request_liabilities_usd_micros[
                    request.request_key
                ]
            except KeyError as exc:
                raise EvidenceInferenceFableScoringError(
                    "runtime_certified_liability_missing"
                ) from exc
        return request.cost.full_context_hard_liability_usd_micros

    adapted: list[TerminalScoringReceiptV1] = []
    cumulative_charged_cost = 0
    for index, request in enumerate(canonical.roster):
        surface = surfaces.get(request.request_key)
        if (
            surface is None
            or surface.article_request_sha256 != request.request_sha256
        ):
            raise EvidenceInferenceFableScoringError(
                "runtime_surface_plan_binding_mismatch"
            )
        intent = EvidenceInferenceFableIntentV1.model_validate(
            _read_runtime_json(
                workspace / "intents" / f"{request.request_key}.json",
                workspace,
            )
        )
        receipt = EvidenceInferenceFableReceiptV1.model_validate(
            _read_runtime_json(
                workspace / "receipts" / f"{request.request_key}.json",
                workspace,
            )
        )
        pair_start = index - (index % 2)
        expected_pair_liability = sum(
            authorized_liability(item)
            for item in canonical.roster[pair_start : pair_start + 2]
        )
        if (
            intent.prepared_sha256 != prepared.prepared_sha256
            or intent.authorization_sha256 != terminal_before.authorization_sha256
            or intent.pair_index != index // 2
            or intent.surface != surface
            or intent.cumulative_reported_spend_before_pair_usd_micros
            != cumulative_charged_cost
            or intent.pair_hard_liability_usd_micros != expected_pair_liability
        ):
            raise EvidenceInferenceFableScoringError(
                "runtime_intent_external_replay_binding_mismatch"
            )
        adapted.append(
            _adapt_runtime_scoring_receipt_v1(
                plan=canonical,
                runtime_terminal=terminal_before,
                runtime_intent=intent,
                runtime_receipt=receipt,
                authorized_request_liability_usd_micros=authorized_liability(
                    request
                ),
                unknown_usage_cost_basis=(
                    "full_context_hard_liability_unknown_usage"
                    if authorization.liability_basis == "full_context_fallback"
                    else (
                        "certified_provider_token_liability_unknown_usage"
                        if authorization.liability_basis
                        == "certified_provider_token_count"
                        else (
                            "certified_provider_token_plus_headroom_liability_unknown_usage"
                        )
                    )
                ),
            )
        )
        cumulative_charged_cost += receipt.provider_result.charged_cost_usd_micros
    terminal_after = validate_evidence_inference_fable_workspace_v1(
        workspace=workspace,
        plan=canonical,
    )
    if terminal_after != terminal_before:
        raise EvidenceInferenceFableScoringError("runtime_changed_during_scoring_replay")
    if cumulative_charged_cost != terminal_after.cumulative_reported_spend_usd_micros:
        raise EvidenceInferenceFableScoringError("runtime_scoring_spend_replay_mismatch")
    return terminal_after, tuple(adapted)


def freeze_private_reference_label_bundle_v1(
    *,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    expected_directions: Mapping[str, Direction],
) -> PrivateReferenceLabelBundleV1:
    """Freeze private directions supplied only after terminal validation."""

    canonical = _canonical_plan(plan)
    article_by_example: dict[str, str] = {}
    for request in canonical.roster:
        for example_id in request.example_ids:
            previous = article_by_example.setdefault(example_id, request.article_id)
            if previous != request.article_id:
                raise EvidenceInferenceFableScoringError("plan_example_article_alias_drift")
    if set(expected_directions) != set(article_by_example):
        raise EvidenceInferenceFableScoringError("private_label_population_mismatch")
    labels = [
        PrivateReferenceLabelV1(
            example_id=example_id,
            article_id=article_by_example[example_id],
            expected_direction=expected_directions[example_id],
        )
        for example_id in sorted(article_by_example)
    ]
    membership = hash_canonical([item.model_dump(mode="json") for item in labels])
    payload = {
        "label_bundle_version": LABEL_BUNDLE_VERSION,
        "plan_sha256": canonical.plan_sha256,
        "population": canonical.population,
        "examples": canonical.unique_examples,
        "articles": canonical.unique_articles,
        "labels": labels,
        "label_membership_sha256": membership,
    }
    return PrivateReferenceLabelBundleV1.model_validate(
        {**payload, "label_bundle_sha256": hash_canonical(payload)}
    )


def repository_results_source_loader_v1(
    *,
    repository_root: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> ResultsSourceLoader:
    """Return a source-only loader; this helper never opens benchmark JSONL or labels."""

    root = Path(os.path.abspath(repository_root)).resolve(strict=True)
    config_source = (root / config_path).resolve(strict=True)
    try:
        config_source.relative_to(root)
        config = json.loads(config_source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidenceInferenceFableScoringError("scoring_source_config_invalid") from exc
    article_root = config.get("article_text_root")
    if not isinstance(article_root, str):
        raise EvidenceInferenceFableScoringError("scoring_article_root_missing")
    source_root = (root / article_root).resolve(strict=True)
    try:
        source_root.relative_to(root)
    except ValueError as exc:
        raise EvidenceInferenceFableScoringError("scoring_article_root_unsafe") from exc

    def load(article_id: str) -> Mapping[str, Any]:
        if not article_id.startswith("PMC") or not article_id[3:].isdigit():
            raise EvidenceInferenceFableScoringError("scoring_article_id_invalid")
        path = source_root / f"{article_id}.txt"
        if path.is_symlink():
            raise EvidenceInferenceFableScoringError("scoring_article_symlink_forbidden")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(source_root)
            source = resolved.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise EvidenceInferenceFableScoringError("scoring_article_source_invalid") from exc
        lines = _content_lines(_source_lines(source))
        if not lines:
            raise EvidenceInferenceFableScoringError("scoring_article_results_empty")
        return lines

    return load


def _validate_terminal_roster(
    *,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    runtime_terminal: EvidenceInferenceFableTerminalV1,
    receipts: Sequence[TerminalScoringReceiptV1 | Mapping[str, Any]],
) -> dict[str, TerminalScoringReceiptV1]:
    terminal = EvidenceInferenceFableTerminalV1.model_validate(runtime_terminal)
    if (
        terminal.status != "completed"
        or not terminal.full_population_score_permitted
        or terminal.completed_request_count != plan.request_count
    ):
        raise EvidenceInferenceFableScoringError(
            "completed_runtime_required_for_private_scoring"
        )
    canonical_receipts = [
        item
        if isinstance(item, TerminalScoringReceiptV1)
        else TerminalScoringReceiptV1.model_validate(item)
        for item in receipts
    ]
    by_request: dict[str, TerminalScoringReceiptV1] = {}
    planned = _request_by_sha(plan)
    for receipt in canonical_receipts:
        if receipt.request_sha256 in by_request:
            raise EvidenceInferenceFableScoringError("duplicate_terminal_receipt")
        request = planned.get(receipt.request_sha256)
        if request is None or (
            receipt.plan_sha256 != plan.plan_sha256
            or receipt.runtime_terminal_sha256 != terminal.terminal_sha256
            or receipt.request_key != request.request_key
            or receipt.article_id != request.article_id
            or receipt.arm != request.arm
        ):
            raise EvidenceInferenceFableScoringError("terminal_receipt_plan_binding_mismatch")
        if receipt.usage is None:
            full_context_bound = request.cost.full_context_hard_liability_usd_micros
            if (
                receipt.accounted_cost_basis
                == "full_context_hard_liability_unknown_usage"
                and receipt.accounted_cost_usd_micros != full_context_bound
            ) or (
                receipt.accounted_cost_basis
                in {
                    "certified_provider_token_liability_unknown_usage",
                    "certified_provider_token_plus_headroom_liability_unknown_usage",
                }
                and not 1 <= receipt.accounted_cost_usd_micros <= full_context_bound
            ):
                raise EvidenceInferenceFableScoringError(
                    "terminal_unknown_cost_not_hard_bounded"
                )
        by_request[receipt.request_sha256] = receipt
    if set(by_request) != set(planned) or len(by_request) != plan.request_count:
        raise EvidenceInferenceFableScoringError(
            "full_terminal_roster_required_before_private_label_access"
        )
    return by_request


def _validate_sources(
    *,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    source_loader: ResultsSourceLoader,
) -> dict[str, Mapping[str, Any]]:
    expected: dict[str, str] = {}
    for request in plan.roster:
        previous = expected.setdefault(
            request.article_id, request.results_source_lines_sha256
        )
        if previous != request.results_source_lines_sha256:
            raise EvidenceInferenceFableScoringError("plan_article_source_hash_drift")
    loaded: dict[str, Mapping[str, Any]] = {}
    for article_id, expected_sha in sorted(expected.items()):
        lines = deepcopy(dict(source_loader(article_id)))
        if hash_canonical(lines) != expected_sha:
            raise EvidenceInferenceFableScoringError("scoring_article_source_hash_mismatch")
        loaded[article_id] = lines
    return loaded


def _batch_error(request: ArticleBatchRequestV1, receipt: TerminalScoringReceiptV1) -> str | None:
    if receipt.provider_outcome != "provider_response":
        return "provider_attempt_failure"
    schema = _output_schema(request.example_ids)
    if hash_canonical(schema) != request.full_acceptance_schema_sha256:
        raise EvidenceInferenceFableScoringError("scoring_schema_plan_hash_mismatch")
    try:
        validator = validator_for(schema)
        validator.check_schema(schema)
        validator(schema).validate(receipt.parsed_batch)
    except (SchemaError, ValidationError):
        return "invalid_article_batch"
    return None


def _failed_arm_score(
    *, receipt: TerminalScoringReceiptV1, error: str
) -> ArmRowScoreV1:
    failure: PrimaryFailure = (
        "provider_attempt_failure"
        if error == "provider_attempt_failure"
        else "invalid_article_batch"
    )
    return ArmRowScoreV1(
        provider_outcome=receipt.provider_outcome,
        batch_structured_output_valid=False,
        predicted_eligible=None,
        predicted_direction=None,
        structured_output_reliability=0,
        direction_accuracy=0,
        exact_grounding_reliability=0,
        direction_and_grounding_joint=0,
        conditional_grounding_evaluated=0,
        conditional_exact_grounding=0,
        grounding_status=None,
        primary_failure=failure,
        batch_validation_error=error,
        parsed_item=None,
        grounding_detail=None,
    )


def _score_valid_item(
    *,
    receipt: TerminalScoringReceiptV1,
    item: Mapping[str, Any],
    expected: Direction,
    source_lines: Mapping[str, Any],
) -> ArmRowScoreV1:
    eligible = item["eligible"]
    findings = item["findings"]
    finding = findings[0] if eligible and len(findings) == 1 else None
    direction = finding["direction"] if isinstance(finding, Mapping) else None
    grounding_detail: dict[str, Any] | None = None
    grounding_status: str | None = None
    exact = False
    if isinstance(finding, Mapping):
        try:
            grounding_detail = ground_evidence(
                finding.get("evidence_quote"),
                finding.get("evidence_lines"),
                source_lines,
                source_accessible=True,
            )
            raw_grounding_status = grounding_detail.get("grounding_status")
            grounding_status = (
                raw_grounding_status
                if raw_grounding_status
                in {"exact", "missing", "mismatch", "unverifiable"}
                else "unverifiable"
            )
            exact = bool(
                grounding_status == "exact"
                and grounding_detail.get("section_flagged") is False
            )
        except (GroundingContractError, TypeError, ValueError):
            grounding_detail = {"grounding_status": "unverifiable"}
            grounding_status = "unverifiable"
    correct = bool(direction == expected)
    if not eligible:
        failure: PrimaryFailure = "eligible_false"
    elif finding is None:
        failure = "missing_finding"
    elif not correct:
        failure = "direction_incorrect"
    elif not exact:
        failure = "exact_grounding_invalid"
    else:
        failure = "success"
    return ArmRowScoreV1(
        provider_outcome=receipt.provider_outcome,
        batch_structured_output_valid=True,
        predicted_eligible=eligible,
        predicted_direction=direction,
        structured_output_reliability=1,
        direction_accuracy=int(correct),
        exact_grounding_reliability=int(exact),
        direction_and_grounding_joint=int(correct and exact),
        conditional_grounding_evaluated=int(finding is not None),
        conditional_exact_grounding=int(exact),
        grounding_status=grounding_status,
        primary_failure=failure,
        batch_validation_error=None,
        parsed_item=deepcopy(dict(item)),
        grounding_detail=grounding_detail,
    )


def _aggregate_arm(
    *,
    arm: Arm,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    receipts: Mapping[str, TerminalScoringReceiptV1],
    rows: Sequence[PrivatePairedRowV1],
    valid_batch_requests: Mapping[str, bool],
) -> ArmAggregateV1:
    arm_receipts = [
        receipts[item.request_sha256] for item in plan.roster if item.arm == arm
    ]
    scores = [getattr(row, arm) for row in rows]
    successes = {
        metric: sum(int(getattr(score, metric)) for score in scores)
        for metric in PRIMARY_METRICS
    }
    conditional_denominator = sum(score.conditional_grounding_evaluated for score in scores)
    conditional_numerator = sum(score.conditional_exact_grounding for score in scores)
    reported = [receipt for receipt in arm_receipts if receipt.usage is not None]
    return ArmAggregateV1(
        requests=len(arm_receipts),
        question_evaluations=len(scores),
        valid_batch_requests=sum(
            valid_batch_requests[receipt.request_sha256] for receipt in arm_receipts
        ),
        provider_outcome_counts=dict(
            sorted(Counter(receipt.provider_outcome for receipt in arm_receipts).items())
        ),
        primary_failure_counts=dict(
            sorted(Counter(score.primary_failure for score in scores).items())
        ),
        metric_success_counts=successes,
        metric_rates={metric: successes[metric] / len(scores) for metric in PRIMARY_METRICS},
        conditional_grounding_denominator=conditional_denominator,
        conditional_exact_grounding_numerator=conditional_numerator,
        conditional_exact_grounding_rate=(
            conditional_numerator / conditional_denominator
            if conditional_denominator
            else None
        ),
        usage_reported_requests=len(reported),
        usage_missing_requests=len(arm_receipts) - len(reported),
        input_tokens=sum(receipt.usage.input_tokens for receipt in reported if receipt.usage),
        output_tokens=sum(receipt.usage.output_tokens for receipt in reported if receipt.usage),
        accounted_cost_usd_micros=sum(
            receipt.accounted_cost_usd_micros for receipt in arm_receipts
        ),
    )


def _bootstrap_placeholder(
    *,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    runtime_workspace: Path,
    completion_certificate: ScoringCompletionCertificateV1,
    rows: Sequence[PrivatePairedRowV1],
) -> PairedArticleClusterBootstrapV1:
    """Dispatch to the separately frozen deterministic inference boundary."""
    by_article: dict[str, list[PrivatePairedRowV1]] = defaultdict(list)
    for row in rows:
        by_article[row.article_id].append(row)
    clusters = []
    for article_id, article_rows in sorted(by_article.items()):
        metric_success_counts = {
            metric: (
                sum(int(getattr(item.seed, metric)) for item in article_rows),
                sum(int(getattr(item.winner, metric)) for item in article_rows),
            )
            for metric in PRIMARY_METRICS
        }
        clusters.append(
            freeze_article_cluster_paired_scores_v1(
                article_id=article_id,
                example_ids=sorted(item.example_id for item in article_rows),
                metric_success_counts=metric_success_counts,
            )
        )
    binding = derive_scoring_completion_binding_from_workspace_v1(
        plan=plan,
        runtime_workspace=runtime_workspace,
        scoring_certificate=completion_certificate,
    )
    result = bootstrap_paired_article_clusters_v1(
        plan=plan,
        scoring_binding=binding,
        clusters=clusters,
    )
    return result


def _score_private_scored_rows_from_replay_v1(
    *,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    runtime_terminal: EvidenceInferenceFableTerminalV1,
    terminal_receipts: Sequence[TerminalScoringReceiptV1 | Mapping[str, Any]],
    source_loader: ResultsSourceLoader,
    private_label_loader: Callable[[], PrivateReferenceLabelBundleV1],
) -> PrivateScoredRowsV1:
    """Score only after the entire terminal roster validates; labels are loaded last."""

    canonical_plan = _canonical_plan(plan)
    receipts = _validate_terminal_roster(
        plan=canonical_plan,
        runtime_terminal=runtime_terminal,
        receipts=terminal_receipts,
    )
    sources = _validate_sources(plan=canonical_plan, source_loader=source_loader)
    # This is intentionally the first label-bearing access in the scorer.
    label_bundle = PrivateReferenceLabelBundleV1.model_validate(
        private_label_loader().model_dump(mode="json")
    )
    labels = {item.example_id: item for item in label_bundle.labels}
    planned_examples = {
        example_id for request in canonical_plan.roster for example_id in request.example_ids
    }
    if (
        label_bundle.plan_sha256 != canonical_plan.plan_sha256
        or label_bundle.population != canonical_plan.population
        or set(labels) != planned_examples
        or label_bundle.examples != canonical_plan.unique_examples
        or label_bundle.articles != canonical_plan.unique_articles
    ):
        raise EvidenceInferenceFableScoringError("private_label_bundle_plan_mismatch")

    score_by_arm_example: dict[tuple[Arm, str], ArmRowScoreV1] = {}
    valid_batch_requests: dict[str, bool] = {}
    for request in canonical_plan.roster:
        receipt = receipts[request.request_sha256]
        error = _batch_error(request, receipt)
        valid_batch_requests[request.request_sha256] = error is None
        if error is not None:
            for example_id in request.example_ids:
                score_by_arm_example[(request.arm, example_id)] = _failed_arm_score(
                    receipt=receipt, error=error
                )
            continue
        assert receipt.parsed_batch is not None  # validated provider-response contract
        results = receipt.parsed_batch["results"]
        for example_id in request.example_ids:
            label = labels[example_id]
            score_by_arm_example[(request.arm, example_id)] = _score_valid_item(
                receipt=receipt,
                item=results[example_id],
                expected=label.expected_direction,
                source_lines=sources[request.article_id],
            )

    rows: list[PrivatePairedRowV1] = []
    for example_id, label in sorted(labels.items()):
        row_payload = {
            "example_id": example_id,
            "article_id": label.article_id,
            "expected_direction": label.expected_direction,
            "seed": score_by_arm_example[("seed", example_id)],
            "winner": score_by_arm_example[("winner", example_id)],
        }
        rows.append(
            PrivatePairedRowV1.model_validate(
                {**row_payload, "row_sha256": hash_canonical(row_payload)}
            )
        )
    receipt_hashes = sorted(receipt.receipt_sha256 for receipt in receipts.values())
    scoring_artifact_payload = {
        "plan_sha256": canonical_plan.plan_sha256,
        "label_bundle_sha256": label_bundle.label_bundle_sha256,
        "receipt_hashes": receipt_hashes,
        "row_hashes": [row.row_sha256 for row in rows],
    }
    scoring_artifact_sha = hash_canonical(scoring_artifact_payload)
    arms = {
        arm: _aggregate_arm(
            arm=arm,  # type: ignore[arg-type]
            plan=canonical_plan,
            receipts=receipts,
            rows=rows,
            valid_batch_requests=valid_batch_requests,
        )
        for arm in ARMS
    }
    payload = {
        "scored_rows_version": PRIVATE_SCORED_ROWS_VERSION,
        "scoring_version": SCORING_VERSION,
        "status": "complete_private_scored_rows",
        "plan_sha256": canonical_plan.plan_sha256,
        "runtime_terminal_sha256": runtime_terminal.terminal_sha256,
        "population": canonical_plan.population,
        "label_bundle_sha256": label_bundle.label_bundle_sha256,
        "receipt_membership_sha256": hash_canonical(receipt_hashes),
        "scoring_artifact_sha256": scoring_artifact_sha,
        "examples": canonical_plan.unique_examples,
        "articles": canonical_plan.unique_articles,
        "requests": canonical_plan.request_count,
        "rows": rows,
        "arms": arms,
        "labels_loaded_only_after_complete_terminal_roster_validation": True,
        "invalid_batch_intention_to_evaluate": True,
        "eligible_false_is_unconditional_grounding_failure": True,
        "empty_finding_is_unconditional_grounding_failure": True,
        "exact_grounding_is_mechanical_not_entailment": True,
        "all_reference_labels_historically_opened": True,
        "exploratory_cross_model_transfer_only": True,
        "confirmatory_gepa_improvement_claim_permitted": False,
        "gepa_optimization_improvement_authority": False,
        "scientific_effectiveness_authority": False,
        "generalization_authority": False,
        "eligibility_metric_claim_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return PrivateScoredRowsV1.model_validate(
        {**payload, "private_scored_rows_sha256": hash_canonical(payload)}
    )


def freeze_scoring_completion_certificate_v1(
    *,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    runtime_terminal: EvidenceInferenceFableTerminalV1,
    scored_rows: PrivateScoredRowsV1,
) -> ScoringCompletionCertificateV1:
    """Bind replayed runtime completion to the frozen private scored-row artifact."""

    canonical = _canonical_plan(plan)
    terminal = EvidenceInferenceFableTerminalV1.model_validate(runtime_terminal)
    scored = PrivateScoredRowsV1.model_validate(scored_rows.model_dump(mode="json"))
    if (
        terminal.status != "completed"
        or not terminal.full_population_score_permitted
        or terminal.completed_request_count != canonical.request_count
        or scored.plan_sha256 != canonical.plan_sha256
        or scored.runtime_terminal_sha256 != terminal.terminal_sha256
        or scored.requests != canonical.request_count
    ):
        raise EvidenceInferenceFableScoringError(
            "scoring_completion_certificate_source_mismatch"
        )
    payload = {
        "certificate_version": COMPLETION_CERTIFICATE_VERSION,
        "scoring_version": SCORING_VERSION,
        "status": "complete_private_scored_rows",
        "plan_sha256": canonical.plan_sha256,
        "runtime_terminal_sha256": terminal.terminal_sha256,
        "private_scored_rows_sha256": scored.private_scored_rows_sha256,
        "scoring_artifact_sha256": scored.scoring_artifact_sha256,
        "receipt_membership_sha256": scored.receipt_membership_sha256,
        "planned_request_count": canonical.request_count,
        "terminal_receipt_count": scored.requests,
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
    return ScoringCompletionCertificateV1.model_validate(
        {**payload, "certificate_sha256": hash_canonical(payload)}
    )


def freeze_private_paired_report_v1(
    *,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    runtime_workspace: Path,
    runtime_terminal: EvidenceInferenceFableTerminalV1,
    scored_rows: PrivateScoredRowsV1,
    completion_certificate: ScoringCompletionCertificateV1,
) -> PrivatePairedReportV1:
    """Add deterministic paired inference to completed private row scoring."""

    canonical = _canonical_plan(plan)
    terminal = EvidenceInferenceFableTerminalV1.model_validate(runtime_terminal)
    scored = PrivateScoredRowsV1.model_validate(scored_rows.model_dump(mode="json"))
    certificate = ScoringCompletionCertificateV1.model_validate(
        completion_certificate.model_dump(mode="json")
    )
    bootstrap = _bootstrap_placeholder(
        plan=canonical,
        runtime_workspace=runtime_workspace,
        completion_certificate=certificate,
        rows=scored.rows,
    )
    payload = {
        "private_report_version": PRIVATE_REPORT_VERSION,
        "scoring_version": SCORING_VERSION,
        "status": "complete_exploratory_retrospective_paired_score",
        "plan_sha256": canonical.plan_sha256,
        "runtime_terminal_sha256": terminal.terminal_sha256,
        "population": canonical.population,
        "scored_rows": scored,
        "completion_certificate": certificate,
        "paired_article_cluster_bootstrap": bootstrap,
    }
    return PrivatePairedReportV1.model_validate(
        {**payload, "private_report_sha256": hash_canonical(payload)}
    )


def score_private_paired_report_v1(
    *,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    runtime_workspace: Path,
    source_loader: ResultsSourceLoader,
    private_label_loader: Callable[[], PrivateReferenceLabelBundleV1],
) -> PrivatePairedReportV1:
    """Replay one completed workspace and produce the bound private paired report."""

    runtime_terminal, receipts = replay_terminal_scoring_receipts_v1(
        plan=plan,
        runtime_workspace=runtime_workspace,
    )
    scored = _score_private_scored_rows_from_replay_v1(
        plan=plan,
        runtime_terminal=runtime_terminal,
        terminal_receipts=receipts,
        source_loader=source_loader,
        private_label_loader=private_label_loader,
    )
    certificate = freeze_scoring_completion_certificate_v1(
        plan=plan,
        runtime_terminal=runtime_terminal,
        scored_rows=scored,
    )
    return freeze_private_paired_report_v1(
        plan=plan,
        runtime_workspace=runtime_workspace,
        runtime_terminal=runtime_terminal,
        scored_rows=scored,
        completion_certificate=certificate,
    )


def project_public_paired_summary_v1(
    report: PrivatePairedReportV1,
) -> PublicPairedSummaryV1:
    """Project an identifier-, label-, prediction-, and quote-free public aggregate."""

    private = PrivatePairedReportV1.model_validate(report.model_dump(mode="json"))
    bootstrap = private.paired_article_cluster_bootstrap
    payload = {
        "public_summary_version": PUBLIC_SUMMARY_VERSION,
        "scoring_version": SCORING_VERSION,
        "status": "aggregate_only_exploratory_retrospective_paired_score",
        "private_report_sha256": private.private_report_sha256,
        "completion_certificate_sha256": (
            private.completion_certificate.certificate_sha256
        ),
        "plan_sha256": private.plan_sha256,
        "runtime_terminal_sha256": private.runtime_terminal_sha256,
        "population": private.population,
        "examples": private.scored_rows.examples,
        "articles": private.scored_rows.articles,
        "requests": private.scored_rows.requests,
        "arms": private.scored_rows.arms,
        "paired_article_cluster_bootstrap": bootstrap,
        "contains_article_or_question_text": False,
        "contains_article_or_example_identifiers": False,
        "contains_reference_or_per_example_labels": False,
        "contains_raw_or_per_example_predictions": False,
        "contains_evidence_quotes_or_line_references": False,
        "contains_absolute_paths": False,
        "all_reference_labels_historically_opened": True,
        "exploratory_cross_model_transfer_only": True,
        "confirmatory_gepa_improvement_claim_permitted": False,
        "gepa_optimization_improvement_authority": False,
        "scientific_effectiveness_authority": False,
        "generalization_authority": False,
        "eligibility_metric_claim_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
        "required_caveats": [
            "historically_opened_test_not_pristine_or_confirmatory",
            "cross_model_and_article_batched_interface_transfer_only",
            "formal_exact_grounding_is_not_semantic_entailment",
            "all_retained_examples_are_eligibility_positive",
        ],
    }
    return PublicPairedSummaryV1.model_validate(
        {**payload, "public_summary_sha256": hash_canonical(payload)}
    )


def materialize_private_and_public_reports_v1(
    *,
    private_report: PrivatePairedReportV1,
    public_summary: PublicPairedSummaryV1,
    private_path: Path,
    public_path: Path,
) -> tuple[Path, Path]:
    """Persist fresh private/public artifacts without overwriting either boundary."""

    private = PrivatePairedReportV1.model_validate(private_report.model_dump(mode="json"))
    public = PublicPairedSummaryV1.model_validate(public_summary.model_dump(mode="json"))
    if public.private_report_sha256 != private.private_report_sha256:
        raise EvidenceInferenceFableScoringError("public_private_report_binding_mismatch")
    if (
        private_path.exists()
        or public_path.exists()
        or private_path.resolve() == public_path.resolve()
    ):
        raise EvidenceInferenceFableScoringError("scoring_report_target_not_fresh")
    atomic_write_json(private_path, private.model_dump(mode="json"))
    atomic_write_json(public_path, public.model_dump(mode="json"))
    return private_path, public_path


__all__ = [
    "EvidenceInferenceFableScoringError",
    "PrivatePairedReportV1",
    "PrivateReferenceLabelBundleV1",
    "PrivateScoredRowsV1",
    "PublicPairedSummaryV1",
    "ScoringCompletionCertificateV1",
    "freeze_private_reference_label_bundle_v1",
    "materialize_private_and_public_reports_v1",
    "project_public_paired_summary_v1",
    "repository_results_source_loader_v1",
    "score_private_paired_report_v1",
]
