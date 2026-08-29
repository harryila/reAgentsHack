"""Label-safe offline plan for a retrospective Evidence Inference Fable benchmark.

The planner reads only public manifests, model-visible question fields, article source
text, and frozen optimizer metadata.  It never opens benchmark JSONL rows, annotations,
reference answers, credentials, or a provider connection.  One provider call owns every
qualified question for one article and one arm.  The paired arms are the exact seed and
scaled GEPA-winner prompts frozen before test access.  Because the official test labels
were historically opened, the comparison is exploratory cross-model transfer rather than
a pristine or confirmatory GEPA-improvement experiment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, StrictInt, model_validator

from literature_multiverse.anthropic_bounded_generation import (
    ANTHROPIC_SDK_VERSION,
    compile_anthropic_bounded_schema,
)
from literature_multiverse.evidence_inference import _content_lines, _source_lines
from literature_multiverse.evidence_inference_gepa_scaled_readiness_v1 import (
    freeze_evidence_inference_gepa_scaled_readiness_v1,
)
from literature_multiverse.lineage import (
    atomic_write_json,
    canonical_json_bytes,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.models import SHA256_RE, ContractModel

CONFIG_VERSION = "evidence-inference-fable-retrospective-config-v1"
PLAN_VERSION = "evidence-inference-fable-retrospective-plan-v1"
REQUEST_VERSION = "evidence-inference-fable-article-request-v1"
DEFAULT_CONFIG_PATH = Path("configs/benchmarks/evidence-inference-fable-retrospective-v1.json")
DEFAULT_PILOT_PLAN_PATH = Path(
    "artifacts/diagnostics/evidence-inference/fable-retrospective-pilot30-plan-v1.json"
)
DEFAULT_RECOVERY_PILOT_PLAN_PATH = Path(
    "artifacts/diagnostics/evidence-inference/"
    "fable-retrospective-pilot30-recovery-v2-plan-v1.json"
)
DEFAULT_FULL_PLAN_PATH = Path(
    "artifacts/diagnostics/evidence-inference/fable-retrospective-full-plan-v1.json"
)
DEFAULT_RECOVERY_EXCLUSION_LEDGER_PATH = Path(
    "artifacts/diagnostics/evidence-inference/"
    "fable-retrospective-pilot-recovery-v2-exclusions.json"
)
DEFAULT_RECOVERY_EXECUTION_POLICY_PATH = Path(
    "artifacts/diagnostics/evidence-inference/"
    "fable-retrospective-pilot-recovery-v2-execution-policy.json"
)

FROZEN_PILOT_PLAN_SHA256 = (
    "0e9637290f065e45d5e0013f0d612def76972fee76e2f6e878f44d565cb90655"
)
FROZEN_FULL_PLAN_SHA256 = (
    "75d94201849e815561165d06467b63c03662828285d8aa6c3fd39933a4bf5864"
)
RECOVERY_V2_ARTICLE_COUNT = 7
RECOVERY_V2_QUESTION_COUNT = 30

MODEL = "claude-fable-5"
EFFORT = "high"
SERVICE_TIER = "standard_only"
INPUT_RATE = 10
OUTPUT_RATE = 50
MAXIMUM_INPUT_TOKENS = 1_000_000
FIXED_FRAMING_TOKENS = 2_048
BASE_OUTPUT_TOKENS = 8_192
OUTPUT_TOKENS_PER_QUESTION = 1_024
MAXIMUM_OUTPUT_TOKENS = 32_000

SYSTEM_PROMPT = (
    "Apply the supplied frozen single-question Evidence Inference policy independently "
    "to every locked question. The policy controls scientific interpretation and exact "
    "source grounding. Only its single-question output-container wording is superseded "
    "by the supplied article-batch JSON schema. For each results key, substitute that "
    "question's outcome, intervention, and comparator for the policy placeholders. Return "
    "one schema-conforming object and no prose. Do not use knowledge outside the supplied "
    "BODY.RESULTS lines."
)

ExecutionMode = Literal[
    "pilot30_paired",
    "pilot30_recovery_v2_paired",
    "full_paired",
]
Population = Literal["pilot30_test", "pilot30_recovery_v2_test", "full_test"]
Arm = Literal["seed", "winner"]


class EvidenceInferenceFableRetrospectiveError(ValueError):
    """An offline source, request, cost, or claim-boundary invariant failed."""


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


def _self_hash(model: _Frozen, field: str, code: str) -> None:
    expected = hash_canonical(model.model_dump(mode="json", exclude={field}))
    if getattr(model, field) != expected:
        raise ValueError(code)


class EvidenceInferenceFableRetrospectiveConfigV1(_Frozen):
    config_version: Literal["evidence-inference-fable-retrospective-config-v1"]
    model: Literal["claude-fable-5"]
    effort: Literal["high"]
    service_tier: Literal["standard_only"]
    transport_mode: Literal["structured_json_schema"]
    anthropic_sdk_version: Literal["0.120.2"]
    input_rate_usd_per_million_tokens: Literal[10]
    output_rate_usd_per_million_tokens: Literal[50]
    maximum_input_tokens: Literal[1000000]
    fixed_framing_tokens: Literal[2048]
    base_output_tokens_per_article: Literal[8192]
    additional_output_tokens_per_question: Literal[1024]
    maximum_output_tokens_per_article: Literal[32000]
    sdk_retries_per_request: Literal[0]
    application_retries_per_request: Literal[0]
    orphan_or_ambiguous_attempt_retry_permitted: Literal[False]
    prompt_caching_permitted: Literal[False]
    full_manifest_path: str
    full_conversion_report_path: str
    pilot_manifest_path: str
    pilot_conversion_report_path: str
    low_budget_manifest_path: str
    low_budget_conversion_report_path: str
    question_table_path: str
    article_text_root: str
    scaled_winner_bundle_path: str
    scaled_winner_prompt_path: str
    scaled_gepa_result_path: str
    scaled_optimization_plan_path: str
    expected_full_test_examples: Literal[524]
    expected_full_test_articles: Literal[191]
    expected_pilot_test_examples: Literal[30]
    expected_pilot_test_articles: Literal[7]
    expected_low_budget_test_examples: Literal[12]
    expected_low_budget_test_articles: Literal[12]
    historical_provider_test_rows: Literal[12]
    historical_provider_test_articles: Literal[12]
    provider_call_unseen_but_label_opened_rows: Literal[482]
    provider_call_unseen_but_label_opened_articles: Literal[179]
    historical_sonnet_low_model_facing_bytes: Literal[133034]
    historical_sonnet_low_input_tokens: Literal[61406]
    historical_completed_output_tokens_mean: Literal[147]
    historical_completed_calls: Literal[9]
    historical_max_token_failures: Literal[3]
    pilot_must_complete_before_full_authorization: Literal[True]
    budget_authority_basis: Literal["full_context_hard_liability_per_next_request"]
    budget_exhaustion_semantics: Literal[
        "clean_terminal_before_next_intent_full_population_score_forbidden"
    ]
    labels_may_be_opened_only_after_all_provider_receipts_are_terminal: Literal[True]
    scaled_gepa_candidate_count: Literal[7]
    scaled_gepa_seed_retained: Literal[False]
    paired_article_cluster_bootstrap_replicates: Literal[20000]
    paired_article_cluster_bootstrap_seed: Literal[20260829]
    exploratory_cross_model_transfer_comparison_permitted: Literal[True]
    confirmatory_gepa_improvement_claim_permitted: Literal[False]
    confirmatory_claim_permitted: Literal[False]
    claim_release_authority: Literal[False]


class PriorExposureV1(_Frozen):
    all_reference_labels_historically_opened: Literal[True] = True
    confirmatory_claim_permitted: Literal[False] = False
    full_test_examples: Literal[524] = 524
    full_test_articles: Literal[191] = 191
    historical_provider_test_rows: Literal[12] = 12
    historical_provider_test_articles: Literal[12] = 12
    provider_call_unseen_but_label_opened_rows: Literal[482] = 482
    provider_call_unseen_but_label_opened_articles: Literal[179] = 179
    complete_official_test_previously_scored: Literal[True] = True
    pristine_holdout_claim_permitted: Literal[False] = False


class AttemptedRequestExclusionV2(_Frozen):
    request_key: Annotated[
        str,
        Field(pattern=r"^ei-fable-retro-v1-pilot30-test-(?:seed|winner)-pmc[1-9][0-9]*$"),
    ]
    article_id: Annotated[str, Field(pattern=r"^PMC[1-9][0-9]*$")]
    intent_file_sha256: Sha256


class RecoveryPilotExclusionLedgerV2(_Frozen):
    ledger_version: Literal["evidence-inference-fable-pilot-recovery-exclusions-v2"]
    source_pilot_plan_sha256: Sha256
    source_prepared_sha256: Sha256
    source_authorization_sha256: Sha256
    source_terminal_file: str
    source_terminal_file_sha256: Sha256
    source_terminal_sha256: Sha256
    source_terminal_status: Literal["terminal_ambiguous_attempt_poison"]
    source_intent_directory: str
    source_incident_file: str
    source_incident_file_sha256: Sha256
    attempted_request_count: PositiveCount
    attempted_article_count: PositiveCount
    attempted_requests: list[AttemptedRequestExclusionV2]
    attempted_article_ids: list[Annotated[str, Field(pattern=r"^PMC[1-9][0-9]*$")]]
    retry_of_any_attempted_request_permitted: Literal[False]
    article_level_exclusion_required: Literal[True]
    exclusion_ledger_sha256: Sha256

    @model_validator(mode="after")
    def validate_ledger(self) -> RecoveryPilotExclusionLedgerV2:
        request_keys = [item.request_key for item in self.attempted_requests]
        article_ids = sorted({item.article_id for item in self.attempted_requests})
        if (
            self.source_pilot_plan_sha256 != FROZEN_PILOT_PLAN_SHA256
            or request_keys != sorted(set(request_keys))
            or self.attempted_request_count != len(request_keys)
            or self.attempted_article_ids != article_ids
            or self.attempted_article_count != len(article_ids)
        ):
            raise ValueError("evidence_inference_fable_recovery_exclusion_alias_mismatch")
        _self_hash(
            self,
            "exclusion_ledger_sha256",
            "evidence_inference_fable_recovery_exclusion_hash_mismatch",
        )
        return self


class RecoveryPilotExecutionPolicyV2(_Frozen):
    policy_version: Literal[
        "evidence-inference-fable-pilot-recovery-execution-policy-v2"
    ]
    mode: Literal["pilot30_recovery_v2_paired"]
    transport_attempts_per_request: Literal[1]
    sdk_retries_per_request: Literal[0]
    application_retries_per_request: Literal[0]
    provider_exception_semantics: Literal["cost_bearing_terminal_failed_request"]
    failed_request_locked_question_score: Literal[
        "all_incorrect_intention_to_evaluate"
    ]
    remaining_never_attempted_requests_continue: Literal[True]
    retry_of_failed_or_ambiguous_attempt_permitted: Literal[False]
    mechanics_only_no_inferential_authority: Literal[True]
    confirmatory_claim_authority: Literal[False]
    claim_release_authority: Literal[False]
    policy_sha256: Sha256

    @model_validator(mode="after")
    def validate_policy(self) -> RecoveryPilotExecutionPolicyV2:
        _self_hash(
            self,
            "policy_sha256",
            "evidence_inference_fable_recovery_execution_policy_hash_mismatch",
        )
        return self


class ArticleRequestCostV1(_Frozen):
    cost_version: Literal["evidence-inference-fable-article-cost-v1"] = (
        "evidence-inference-fable-article-cost-v1"
    )
    model_facing_utf8_bytes: PositiveCount
    fixed_framing_tokens: Literal[2048] = FIXED_FRAMING_TOKENS
    diagnostic_known_input_token_ceiling: PositiveCount
    maximum_input_tokens: Literal[1000000] = MAXIMUM_INPUT_TOKENS
    max_output_tokens: Annotated[StrictInt, Field(ge=1, le=32000)]
    input_rate_usd_per_million_tokens: Literal[10] = INPUT_RATE
    output_rate_usd_per_million_tokens: Literal[50] = OUTPUT_RATE
    diagnostic_known_surface_cost_usd_micros: PositiveCount
    full_context_hard_liability_usd_micros: PositiveCount
    cost_sha256: Sha256

    @model_validator(mode="after")
    def validate_cost(self) -> ArticleRequestCostV1:
        known_input = self.model_facing_utf8_bytes + FIXED_FRAMING_TOKENS
        diagnostic = known_input * INPUT_RATE + self.max_output_tokens * OUTPUT_RATE
        hard = MAXIMUM_INPUT_TOKENS * INPUT_RATE + self.max_output_tokens * OUTPUT_RATE
        if (
            self.diagnostic_known_input_token_ceiling != known_input
            or known_input > MAXIMUM_INPUT_TOKENS
            or self.diagnostic_known_surface_cost_usd_micros != diagnostic
            or self.full_context_hard_liability_usd_micros != hard
        ):
            raise ValueError("evidence_inference_fable_cost_alias_mismatch")
        _self_hash(self, "cost_sha256", "evidence_inference_fable_cost_hash_mismatch")
        return self


class ArticleBatchRequestV1(_Frozen):
    request_version: Literal["evidence-inference-fable-article-request-v1"] = REQUEST_VERSION
    execution_index: Count
    population: Population
    arm: Arm
    request_key: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{2,127}$")]
    article_id: Annotated[str, Field(pattern=r"^PMC[1-9][0-9]*$")]
    question_count: PositiveCount
    example_ids: list[Annotated[str, Field(pattern=r"^ei2-prompt-[1-9][0-9]*$")]]
    source_text_sha256: Sha256
    results_source_lines_sha256: Sha256
    question_payload_sha256: Sha256
    arm_policy_sha256: Sha256
    system_sha256: Sha256
    prompt_sha256: Sha256
    full_acceptance_schema_sha256: Sha256
    wire_schema_sha256: Sha256
    compiled_schema_sha256: Sha256
    model_facing_surface_sha256: Sha256
    model_facing_utf8_bytes: PositiveCount
    max_output_tokens: Annotated[StrictInt, Field(ge=1, le=32000)]
    cost: ArticleRequestCostV1
    request_sha256: Sha256

    @model_validator(mode="after")
    def validate_request(self) -> ArticleBatchRequestV1:
        if (
            self.example_ids != sorted(set(self.example_ids))
            or self.question_count != len(self.example_ids)
            or self.model_facing_utf8_bytes != self.cost.model_facing_utf8_bytes
            or self.max_output_tokens != self.cost.max_output_tokens
        ):
            raise ValueError("evidence_inference_fable_request_alias_mismatch")
        _self_hash(self, "request_sha256", "evidence_inference_fable_request_hash_mismatch")
        return self


class EvidenceInferenceFableRetrospectivePlanV1(_Frozen):
    plan_version: Literal["evidence-inference-fable-retrospective-plan-v1"] = PLAN_VERSION
    status: Literal["offline_prepared_zero_provider_calls"] = "offline_prepared_zero_provider_calls"
    mode: ExecutionMode
    population: Population
    config_sha256: Sha256
    input_artifact_sha256s: dict[str, Sha256]
    scaled_winner_bundle_sha256: Sha256
    scaled_winner_prompt_sha256: Sha256
    scaled_winner_embedded_prompt_sha256: Sha256
    scaled_winner_candidate_count: Literal[7] = 7
    recorded_seed_prompt_sha256: Sha256
    seed_prompt_extracted_from_scaled_trace_sha256: Sha256
    prompt_hash_relationship: Literal["distinct"] = "distinct"
    comparison_interpretation: Literal[
        "exploratory_cross_model_transfer_on_historically_opened_test"
    ] = "exploratory_cross_model_transfer_on_historically_opened_test"
    exploratory_cross_model_transfer_comparison_permitted: Literal[True] = True
    confirmatory_gepa_improvement_claim_permitted: Literal[False] = False
    prior_exposure: PriorExposureV1
    prior_exposure_readiness_receipt_sha256: Sha256
    unique_examples: PositiveCount
    unique_articles: PositiveCount
    arm_count: Literal[2] = 2
    question_evaluations: PositiveCount
    request_count: PositiveCount
    roster: list[ArticleBatchRequestV1]
    request_roster_sha256: Sha256
    total_model_facing_utf8_bytes: PositiveCount
    total_diagnostic_known_input_token_ceiling: PositiveCount
    total_max_output_tokens: PositiveCount
    total_diagnostic_known_surface_cost_usd_micros: PositiveCount
    total_full_context_hard_liability_usd_micros: PositiveCount
    historical_lower_heuristic_input_tokens: PositiveCount
    historical_lower_heuristic_output_tokens: PositiveCount
    historical_lower_heuristic_cost_usd_micros: PositiveCount
    historical_heuristic_is_not_budget_authority: Literal[True] = True
    primary_inference_unit: Literal["article_cluster"] = "article_cluster"
    paired_interval_method: Literal["paired_article_cluster_bootstrap"] = (
        "paired_article_cluster_bootstrap"
    )
    paired_bootstrap_replicates: Literal[20000] = 20000
    paired_bootstrap_seed: Literal[20260829] = 20260829
    paired_primary_metrics: list[
        Literal[
            "direction_accuracy",
            "structured_output_reliability",
            "exact_grounding_reliability",
        ]
    ]
    arm_order_balancing: Literal[
        "label_blind_article_hash_order_with_strict_alternating_first_arm"
    ] = "label_blind_article_hash_order_with_strict_alternating_first_arm"
    model_input_policy: Literal["all_BODY_RESULTS_lines_once_per_article"] = (
        "all_BODY_RESULTS_lines_once_per_article"
    )
    scaled_optimizer_input_policy: Literal[
        "single_question_fixed_results_passage_projection_v1"
    ] = "single_question_fixed_results_passage_projection_v1"
    cross_model_and_batched_interface_transfer_only: Literal[True] = True
    qualified_population_all_expected_eligible: Literal[True] = True
    eligibility_negative_examples: Literal[0] = 0
    eligibility_metric_claim_authority: Literal[False] = False
    invalid_batch_counts_each_locked_question_incorrect: Literal[True] = True
    eligible_false_counts_unconditional_grounded_answer_failure: Literal[True] = True
    conditional_grounding_reported_separately: Literal[True] = True
    pilot_population_is_subset_of_full_test: Literal[True] = True
    pilot_is_mechanics_only_no_inferential_authority: Literal[True] = True
    pilot_preflight_required_before_full_authorization: bool
    full_plan_must_be_frozen_before_pilot_execution: Literal[True] = True
    pilot_results_may_not_change_prompt_schema_thresholds_or_full_roster: Literal[True] = True
    full_population_score_requires_every_planned_request_terminal: Literal[True] = True
    budget_exhaustion_before_intent_is_clean_terminal: Literal[True] = True
    budget_exhaustion_after_intent_is_ambiguous_and_never_retried: Literal[True] = True
    labels_opened_by_planner: Literal[False] = False
    benchmark_jsonl_opened_by_planner: Literal[False] = False
    annotations_opened_by_planner: Literal[False] = False
    provider_calls_made: Literal[0] = 0
    credentials_opened: Literal[False] = False
    network_opened: Literal[False] = False
    plan_contains_article_or_question_text: Literal[False] = False
    retrospective_accuracy_claim_authority_before_execution: Literal[False] = False
    confirmatory_claim_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    plan_sha256: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> EvidenceInferenceFableRetrospectivePlanV1:
        if (
            self.scaled_winner_prompt_sha256 != self.scaled_winner_embedded_prompt_sha256
            or self.recorded_seed_prompt_sha256
            != self.seed_prompt_extracted_from_scaled_trace_sha256
            or self.scaled_winner_prompt_sha256 == self.recorded_seed_prompt_sha256
            or self.question_evaluations != self.unique_examples * 2
            or self.request_count != self.unique_articles * 2
            or len(self.roster) != self.request_count
            or [item.execution_index for item in self.roster] != list(range(self.request_count))
            or len({item.request_key for item in self.roster}) != self.request_count
            or self.request_roster_sha256
            != hash_canonical([item.request_sha256 for item in self.roster])
            or self.total_model_facing_utf8_bytes
            != sum(item.model_facing_utf8_bytes for item in self.roster)
            or self.total_diagnostic_known_input_token_ceiling
            != sum(item.cost.diagnostic_known_input_token_ceiling for item in self.roster)
            or self.total_max_output_tokens != sum(item.max_output_tokens for item in self.roster)
            or self.total_diagnostic_known_surface_cost_usd_micros
            != sum(item.cost.diagnostic_known_surface_cost_usd_micros for item in self.roster)
            or self.total_full_context_hard_liability_usd_micros
            != sum(item.cost.full_context_hard_liability_usd_micros for item in self.roster)
            or self.pilot_preflight_required_before_full_authorization
            != (self.mode == "full_paired")
            or self.paired_primary_metrics
            != [
                "direction_accuracy",
                "structured_output_reliability",
                "exact_grounding_reliability",
            ]
        ):
            raise ValueError("evidence_inference_fable_plan_alias_mismatch")
        seed_first = 0
        winner_first = 0
        for offset in range(0, len(self.roster), 2):
            first, second = self.roster[offset : offset + 2]
            if (
                first.article_id != second.article_id
                or {first.arm, second.arm} != {"seed", "winner"}
                or first.example_ids != second.example_ids
                or first.source_text_sha256 != second.source_text_sha256
                or first.results_source_lines_sha256 != second.results_source_lines_sha256
                or first.question_payload_sha256 != second.question_payload_sha256
                or first.full_acceptance_schema_sha256 != second.full_acceptance_schema_sha256
                or first.wire_schema_sha256 != second.wire_schema_sha256
                or first.max_output_tokens != second.max_output_tokens
                or first.prompt_sha256 == second.prompt_sha256
            ):
                raise ValueError("evidence_inference_fable_paired_arm_drift")
            seed_first += int(first.arm == "seed")
            winner_first += int(first.arm == "winner")
        if abs(seed_first - winner_first) > 1:
            raise ValueError("evidence_inference_fable_arm_order_unbalanced")
        _self_hash(self, "plan_sha256", "evidence_inference_fable_plan_hash_mismatch")
        return self


def _safe_file(root: Path, relative: str | Path) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise EvidenceInferenceFableRetrospectiveError("unsafe_repository_path")
    lexical = root / value
    if lexical.is_symlink():
        raise EvidenceInferenceFableRetrospectiveError("symlinked_repository_file_forbidden")
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise EvidenceInferenceFableRetrospectiveError("repository_file_missing") from exc
    if not resolved.is_file():
        raise EvidenceInferenceFableRetrospectiveError("repository_path_not_file")
    return resolved


def _load_json(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceInferenceFableRetrospectiveError(code) from exc
    if not isinstance(value, dict):
        raise EvidenceInferenceFableRetrospectiveError(code)
    return value


def _output_schema(example_ids: Sequence[str]) -> dict[str, Any]:
    finding = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["increase", "no_effect", "decrease"],
            },
            "evidence_quote": {"type": "string", "minLength": 1},
            "evidence_lines": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "string",
                    "pattern": r"^L[1-9][0-9]*(?:-L[1-9][0-9]*)?$",
                },
            },
        },
        "required": ["direction", "evidence_quote", "evidence_lines"],
        "additionalProperties": False,
    }
    item = {
        "type": "object",
        "properties": {
            "eligible": {"type": "boolean"},
            "findings": {
                "type": "array",
                "minItems": 0,
                "maxItems": 1,
                "items": finding,
            },
        },
        "required": ["eligible", "findings"],
        "additionalProperties": False,
    }
    ids = sorted(example_ids)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "results": {
                "type": "object",
                "properties": {example_id: deepcopy(item) for example_id in ids},
                "required": ids,
                "additionalProperties": False,
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def _question_rows(*, question_table: Path, expected_ids: set[str]) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    try:
        handle = question_table.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise EvidenceInferenceFableRetrospectiveError("question_table_unreadable") from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {"PromptID", "PMCID", "Outcome", "Intervention", "Comparator"}
        if not required.issubset(reader.fieldnames or ()):
            raise EvidenceInferenceFableRetrospectiveError("question_table_headers_invalid")
        for row in reader:
            example_id = f"ei2-prompt-{row['PromptID']}"
            if example_id not in expected_ids:
                continue
            if example_id in selected:
                raise EvidenceInferenceFableRetrospectiveError("duplicate_question_id")
            selected[example_id] = {
                "example_id": example_id,
                "paper_id": f"PMC{row['PMCID']}",
                "outcome": row["Outcome"].strip(),
                "intervention": row["Intervention"].strip(),
                "comparator": row["Comparator"].strip(),
            }
    if set(selected) != expected_ids or any(
        not all(value for value in row.values()) for row in selected.values()
    ):
        raise EvidenceInferenceFableRetrospectiveError("question_population_incomplete")
    return selected


def _render_prompt(
    *, frozen_policy: str, article_id: str, questions: Sequence[Mapping[str, str]], lines: Any
) -> str:
    payload = {
        "article_id": article_id,
        "locked_questions": [dict(item) for item in questions],
        "body_results_source_lines": lines,
    }
    return (
        "<frozen_single_question_policy>\n"
        f"{frozen_policy.rstrip()}\n"
        "</frozen_single_question_policy>\n\n"
        "<article_batch_input>\n"
        f"{canonical_json_bytes(payload).decode('utf-8')}\n"
        "</article_batch_input>\n"
    )


def _cost(*, model_facing_bytes: int, max_output_tokens: int) -> ArticleRequestCostV1:
    known_input = model_facing_bytes + FIXED_FRAMING_TOKENS
    payload = {
        "cost_version": "evidence-inference-fable-article-cost-v1",
        "model_facing_utf8_bytes": model_facing_bytes,
        "fixed_framing_tokens": FIXED_FRAMING_TOKENS,
        "diagnostic_known_input_token_ceiling": known_input,
        "maximum_input_tokens": MAXIMUM_INPUT_TOKENS,
        "max_output_tokens": max_output_tokens,
        "input_rate_usd_per_million_tokens": INPUT_RATE,
        "output_rate_usd_per_million_tokens": OUTPUT_RATE,
        "diagnostic_known_surface_cost_usd_micros": (
            known_input * INPUT_RATE + max_output_tokens * OUTPUT_RATE
        ),
        "full_context_hard_liability_usd_micros": (
            MAXIMUM_INPUT_TOKENS * INPUT_RATE + max_output_tokens * OUTPUT_RATE
        ),
    }
    return ArticleRequestCostV1.model_validate({**payload, "cost_sha256": hash_canonical(payload)})


def _population_contract(
    *, config: EvidenceInferenceFableRetrospectiveConfigV1, mode: ExecutionMode
) -> tuple[Population, str, str, int, int]:
    if mode == "pilot30_paired":
        return (
            "pilot30_test",
            config.pilot_manifest_path,
            config.pilot_conversion_report_path,
            config.expected_pilot_test_examples,
            config.expected_pilot_test_articles,
        )
    return (
        "full_test",
        config.full_manifest_path,
        config.full_conversion_report_path,
        config.expected_full_test_examples,
        config.expected_full_test_articles,
    )


def _request_order(population: Population, article_ids: Sequence[str]) -> list[str]:
    return sorted(
        article_ids,
        key=lambda article_id: hashlib.sha256(
            f"{PLAN_VERSION}:{population}:{article_id}".encode("ascii")
        ).hexdigest(),
    )


def _recovery_exclusion_ledger_v2(
    *,
    root: Path,
    pilot_plan: EvidenceInferenceFableRetrospectivePlanV1,
) -> tuple[RecoveryPilotExclusionLedgerV2, Path]:
    """Replay the label-free durable-intent ledger that defines v2 exclusions."""

    ledger_path = _safe_file(root, DEFAULT_RECOVERY_EXCLUSION_LEDGER_PATH)
    try:
        ledger = RecoveryPilotExclusionLedgerV2.model_validate(
            _load_json(ledger_path, code="recovery_exclusion_ledger_invalid")
        )
    except ValueError as exc:
        raise EvidenceInferenceFableRetrospectiveError(
            "recovery_exclusion_ledger_contract_invalid"
        ) from exc
    if pilot_plan.plan_sha256 != ledger.source_pilot_plan_sha256:
        raise EvidenceInferenceFableRetrospectiveError(
            "recovery_exclusion_pilot_plan_mismatch"
        )

    terminal_path = _safe_file(root, ledger.source_terminal_file)
    terminal = _load_json(terminal_path, code="recovery_source_terminal_invalid")
    terminal_payload = {key: value for key, value in terminal.items() if key != "terminal_sha256"}
    if (
        sha256_file(terminal_path) != ledger.source_terminal_file_sha256
        or terminal.get("terminal_sha256") != ledger.source_terminal_sha256
        or terminal.get("terminal_sha256") != hash_canonical(terminal_payload)
        or terminal.get("status") != ledger.source_terminal_status
        or terminal.get("prepared_sha256") != ledger.source_prepared_sha256
        or terminal.get("authorization_sha256") != ledger.source_authorization_sha256
        or terminal.get("full_population_score_permitted") is not False
        or terminal.get("next_pair_index") != 5
        or terminal.get("completed_request_count") != 10
    ):
        raise EvidenceInferenceFableRetrospectiveError(
            "recovery_source_terminal_lineage_invalid"
        )

    incident_path = _safe_file(root, ledger.source_incident_file)
    incident = _load_json(incident_path, code="recovery_source_incident_invalid")
    incident_payload = {key: value for key, value in incident.items() if key != "incident_sha256"}
    if (
        sha256_file(incident_path) != ledger.source_incident_file_sha256
        or incident.get("incident_sha256") != hash_canonical(incident_payload)
        or incident.get("status") != "terminal_ambiguous_attempt_poison"
        or incident.get("retry_permitted") is not False
        or incident.get("request_key")
        != "ei-fable-retro-v1-pilot30-test-winner-pmc1871574"
    ):
        raise EvidenceInferenceFableRetrospectiveError(
            "recovery_source_incident_lineage_invalid"
        )

    intent_relative = Path(ledger.source_intent_directory)
    if intent_relative.is_absolute() or ".." in intent_relative.parts:
        raise EvidenceInferenceFableRetrospectiveError(
            "recovery_source_intent_directory_unsafe"
        )
    intent_directory = root / intent_relative
    if intent_directory.is_symlink() or not intent_directory.is_dir():
        raise EvidenceInferenceFableRetrospectiveError(
            "recovery_source_intent_directory_invalid"
        )
    observed_names = sorted(
        item.name
        for item in intent_directory.iterdir()
        if item.is_file() and not item.is_symlink() and item.suffix == ".json"
    )
    expected_names = sorted(f"{item.request_key}.json" for item in ledger.attempted_requests)
    if observed_names != expected_names:
        raise EvidenceInferenceFableRetrospectiveError(
            "recovery_attempted_intent_roster_drift"
        )

    pilot_requests = {item.request_key: item for item in pilot_plan.roster}
    for exclusion in ledger.attempted_requests:
        original = pilot_requests.get(exclusion.request_key)
        intent_path = _safe_file(root, intent_relative / f"{exclusion.request_key}.json")
        intent = _load_json(intent_path, code="recovery_source_intent_invalid")
        intent_payload = {key: value for key, value in intent.items() if key != "intent_sha256"}
        surface = intent.get("surface")
        if (
            original is None
            or original.article_id != exclusion.article_id
            or sha256_file(intent_path) != exclusion.intent_file_sha256
            or intent.get("intent_sha256") != hash_canonical(intent_payload)
            or intent.get("request_key") != exclusion.request_key
            or intent.get("permitted_provider_attempts") != 1
            or intent.get("application_retries_permitted") != 0
            or intent.get("sdk_retries_permitted") != 0
            or intent.get("orphan_or_ambiguous_attempt_is_terminal") is not True
            or not isinstance(surface, Mapping)
            or surface.get("request_key") != exclusion.request_key
            or surface.get("article_request_sha256") != original.request_sha256
        ):
            raise EvidenceInferenceFableRetrospectiveError(
                "recovery_source_intent_lineage_invalid"
            )
    if incident.get("request_key") not in {
        item.request_key for item in ledger.attempted_requests
    }:
        raise EvidenceInferenceFableRetrospectiveError(
            "recovery_incident_not_in_attempted_roster"
        )
    return ledger, ledger_path


def _select_recovery_articles_v2(
    *,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    excluded_article_ids: set[str],
) -> tuple[str, ...]:
    """Choose the earliest full-plan 7-subset totaling 30 locked questions."""

    available: list[tuple[str, int]] = []
    for offset in range(0, len(full_plan.roster), 2):
        first, second = full_plan.roster[offset : offset + 2]
        if first.article_id != second.article_id or first.question_count != second.question_count:
            raise EvidenceInferenceFableRetrospectiveError(
                "recovery_full_plan_pair_drift"
            )
        if first.article_id not in excluded_article_ids:
            available.append((first.article_id, first.question_count))

    states: dict[tuple[int, int], tuple[int, ...]] = {(0, 0): ()}
    for index, (_, question_count) in enumerate(available):
        updated = dict(states)
        for (selected_count, total_questions), selection in states.items():
            if selected_count >= RECOVERY_V2_ARTICLE_COUNT:
                continue
            key = (selected_count + 1, total_questions + question_count)
            candidate = (*selection, index)
            incumbent = updated.get(key)
            if incumbent is None or candidate < incumbent:
                updated[key] = candidate
        states = updated
    selection = states.get((RECOVERY_V2_ARTICLE_COUNT, RECOVERY_V2_QUESTION_COUNT))
    if selection is None:
        raise EvidenceInferenceFableRetrospectiveError(
            "recovery_exact_7_article_30_question_subset_unavailable"
        )
    return tuple(available[index][0] for index in selection)


def _freeze_evidence_inference_fable_recovery_plan_v2(
    *,
    root: Path,
    config_path: Path,
    model_surface_sink: list[dict[str, Any]] | None,
) -> EvidenceInferenceFableRetrospectivePlanV1:
    """Derive a fresh-key, article-disjoint mechanics pilot from the frozen full plan."""

    full_surfaces: list[dict[str, Any]] = []
    full_plan = freeze_evidence_inference_fable_retrospective_plan_v1(
        repository_root=root,
        mode="full_paired",
        config_path=config_path,
        _model_surface_sink=full_surfaces,
    )
    pilot_plan = freeze_evidence_inference_fable_retrospective_plan_v1(
        repository_root=root,
        mode="pilot30_paired",
        config_path=config_path,
    )
    frozen_full_path = _safe_file(root, DEFAULT_FULL_PLAN_PATH)
    frozen_pilot_path = _safe_file(root, DEFAULT_PILOT_PLAN_PATH)
    try:
        serialized_full = EvidenceInferenceFableRetrospectivePlanV1.model_validate(
            _load_json(frozen_full_path, code="recovery_frozen_full_plan_invalid")
        )
        serialized_pilot = EvidenceInferenceFableRetrospectivePlanV1.model_validate(
            _load_json(frozen_pilot_path, code="recovery_frozen_pilot_plan_invalid")
        )
    except ValueError as exc:
        raise EvidenceInferenceFableRetrospectiveError(
            "recovery_source_plan_contract_invalid"
        ) from exc
    if (
        full_plan != serialized_full
        or pilot_plan != serialized_pilot
        or full_plan.plan_sha256 != FROZEN_FULL_PLAN_SHA256
        or pilot_plan.plan_sha256 != FROZEN_PILOT_PLAN_SHA256
    ):
        raise EvidenceInferenceFableRetrospectiveError(
            "recovery_predeclared_source_plan_mismatch"
        )
    ledger, ledger_path = _recovery_exclusion_ledger_v2(
        root=root,
        pilot_plan=pilot_plan,
    )
    execution_policy_path = _safe_file(root, DEFAULT_RECOVERY_EXECUTION_POLICY_PATH)
    try:
        execution_policy = RecoveryPilotExecutionPolicyV2.model_validate(
            _load_json(
                execution_policy_path,
                code="recovery_execution_policy_invalid",
            )
        )
    except ValueError as exc:
        raise EvidenceInferenceFableRetrospectiveError(
            "recovery_execution_policy_contract_invalid"
        ) from exc
    excluded_articles = set(ledger.attempted_article_ids)
    selected_articles = _select_recovery_articles_v2(
        full_plan=full_plan,
        excluded_article_ids=excluded_articles,
    )
    if set(selected_articles) & excluded_articles:
        raise EvidenceInferenceFableRetrospectiveError(
            "recovery_selected_attempted_article_forbidden"
        )

    by_article_arm: dict[tuple[str, Arm], tuple[ArticleBatchRequestV1, dict[str, Any]]] = {}
    for request, surface in zip(full_plan.roster, full_surfaces, strict=True):
        by_article_arm[(request.article_id, request.arm)] = (request, surface)
    ordered_articles = _request_order("pilot30_recovery_v2_test", selected_articles)
    roster: list[ArticleBatchRequestV1] = []
    selected_surfaces: list[dict[str, Any]] = []
    for article_offset, article_id in enumerate(ordered_articles):
        arm_order: list[Arm] = ["seed", "winner"] if article_offset % 2 == 0 else [
            "winner",
            "seed",
        ]
        for arm in arm_order:
            original, surface = by_article_arm[(article_id, arm)]
            base = original.model_dump(
                mode="json",
                exclude={
                    "request_version",
                    "execution_index",
                    "population",
                    "arm",
                    "request_key",
                    "request_sha256",
                },
            )
            request_payload = {
                "request_version": REQUEST_VERSION,
                "execution_index": len(roster),
                "population": "pilot30_recovery_v2_test",
                "arm": arm,
                "request_key": (
                    "ei-fable-retro-v2-pilot30-recovery-test-"
                    f"{arm}-{article_id.lower()}"
                ),
                **base,
            }
            roster.append(
                ArticleBatchRequestV1.model_validate(
                    {**request_payload, "request_sha256": hash_canonical(request_payload)}
                )
            )
            selected_surfaces.append(deepcopy(surface))

    if (
        len(roster) != RECOVERY_V2_ARTICLE_COUNT * 2
        or sum(item.question_count for item in roster) != RECOVERY_V2_QUESTION_COUNT * 2
        or {item.article_id for item in roster} != set(selected_articles)
        or {item.request_key for item in roster}
        & {item.request_key for item in ledger.attempted_requests}
    ):
        raise EvidenceInferenceFableRetrospectiveError(
            "recovery_roster_disjointness_or_count_invalid"
        )

    config_source = _safe_file(root, config_path)
    config = EvidenceInferenceFableRetrospectiveConfigV1.model_validate(
        _load_json(config_source, code="config_invalid")
    )
    total_surface = sum(item.model_facing_utf8_bytes for item in roster)
    heuristic_input = math.ceil(
        total_surface
        * config.historical_sonnet_low_input_tokens
        / config.historical_sonnet_low_model_facing_bytes
    )
    heuristic_output = (
        RECOVERY_V2_QUESTION_COUNT
        * 2
        * config.historical_completed_output_tokens_mean
    )
    payload = full_plan.model_dump(mode="json", exclude={"plan_sha256"})
    payload.update(
        {
            "mode": "pilot30_recovery_v2_paired",
            "population": "pilot30_recovery_v2_test",
            "input_artifact_sha256s": {
                **full_plan.input_artifact_sha256s,
                "predeclared_full_plan_file": sha256_file(frozen_full_path),
                "predeclared_full_plan_semantic": full_plan.plan_sha256,
                "predeclared_pilot_v1_plan_file": sha256_file(frozen_pilot_path),
                "recovery_v2_exclusion_ledger": sha256_file(ledger_path),
                "recovery_v2_exclusion_ledger_semantic": ledger.exclusion_ledger_sha256,
                "recovery_v2_execution_policy": sha256_file(execution_policy_path),
                "recovery_v2_execution_policy_semantic": execution_policy.policy_sha256,
                "recovery_v2_source_terminal_file": ledger.source_terminal_file_sha256,
                "recovery_v2_source_terminal_semantic": ledger.source_terminal_sha256,
            },
            "unique_examples": RECOVERY_V2_QUESTION_COUNT,
            "unique_articles": RECOVERY_V2_ARTICLE_COUNT,
            "question_evaluations": RECOVERY_V2_QUESTION_COUNT * 2,
            "request_count": RECOVERY_V2_ARTICLE_COUNT * 2,
            "roster": roster,
            "request_roster_sha256": hash_canonical(
                [item.request_sha256 for item in roster]
            ),
            "total_model_facing_utf8_bytes": total_surface,
            "total_diagnostic_known_input_token_ceiling": sum(
                item.cost.diagnostic_known_input_token_ceiling for item in roster
            ),
            "total_max_output_tokens": sum(item.max_output_tokens for item in roster),
            "total_diagnostic_known_surface_cost_usd_micros": sum(
                item.cost.diagnostic_known_surface_cost_usd_micros for item in roster
            ),
            "total_full_context_hard_liability_usd_micros": sum(
                item.cost.full_context_hard_liability_usd_micros for item in roster
            ),
            "historical_lower_heuristic_input_tokens": heuristic_input,
            "historical_lower_heuristic_output_tokens": heuristic_output,
            "historical_lower_heuristic_cost_usd_micros": (
                heuristic_input * INPUT_RATE + heuristic_output * OUTPUT_RATE
            ),
            "pilot_preflight_required_before_full_authorization": False,
        }
    )
    plan = EvidenceInferenceFableRetrospectivePlanV1.model_validate(
        {**payload, "plan_sha256": hash_canonical(payload)}
    )
    if model_surface_sink is not None:
        model_surface_sink.extend(selected_surfaces)
    return plan


def freeze_evidence_inference_fable_retrospective_plan_v1(
    *,
    repository_root: Path,
    mode: ExecutionMode,
    config_path: Path = DEFAULT_CONFIG_PATH,
    _model_surface_sink: list[dict[str, Any]] | None = None,
) -> EvidenceInferenceFableRetrospectivePlanV1:
    """Freeze one label-safe request roster without provider or benchmark-row access."""

    try:
        root = Path(os.path.abspath(repository_root)).resolve(strict=True)
    except OSError as exc:
        raise EvidenceInferenceFableRetrospectiveError("repository_root_missing") from exc
    if mode == "pilot30_recovery_v2_paired":
        return _freeze_evidence_inference_fable_recovery_plan_v2(
            root=root,
            config_path=config_path,
            model_surface_sink=_model_surface_sink,
        )
    config_source = _safe_file(root, config_path)
    config = EvidenceInferenceFableRetrospectiveConfigV1.model_validate(
        _load_json(config_source, code="config_invalid")
    )
    if config.anthropic_sdk_version != ANTHROPIC_SDK_VERSION:
        raise EvidenceInferenceFableRetrospectiveError("anthropic_sdk_contract_drift")
    readiness = freeze_evidence_inference_gepa_scaled_readiness_v1(repository_root=root)
    prior_facts = readiness.prior_study
    if (
        prior_facts.all_labels_historically_opened is not True
        or prior_facts.confirmatory_claim_allowed is not False
        or prior_facts.accepted_candidate_count != config.scaled_gepa_candidate_count
        or prior_facts.actual_metric_calls != 864
        or prior_facts.paired_test_examples != config.expected_full_test_examples
        or prior_facts.paired_test_articles != config.expected_full_test_articles
        or prior_facts.provider_touched_test_rows != config.historical_provider_test_rows
        or prior_facts.provider_touched_test_articles != config.historical_provider_test_articles
        or prior_facts.provider_call_unseen_but_label_opened_rows
        != config.provider_call_unseen_but_label_opened_rows
        or prior_facts.provider_call_unseen_but_label_opened_articles
        != config.provider_call_unseen_but_label_opened_articles
    ):
        raise EvidenceInferenceFableRetrospectiveError("prior_exposure_ledger_drift")

    population, manifest_rel, report_rel, expected_examples, expected_articles = (
        _population_contract(config=config, mode=mode)
    )
    manifest_path = _safe_file(root, manifest_rel)
    report_path = _safe_file(root, report_rel)
    low_manifest_path = _safe_file(root, config.low_budget_manifest_path)
    low_report_path = _safe_file(root, config.low_budget_conversion_report_path)
    full_manifest_path = _safe_file(root, config.full_manifest_path)
    full_report_path = _safe_file(root, config.full_conversion_report_path)
    question_table = _safe_file(root, config.question_table_path)
    winner_bundle_path = _safe_file(root, config.scaled_winner_bundle_path)
    winner_prompt_path = _safe_file(root, config.scaled_winner_prompt_path)
    gepa_result_path = _safe_file(root, config.scaled_gepa_result_path)
    optimization_plan_path = _safe_file(root, config.scaled_optimization_plan_path)

    manifest = _load_json(manifest_path, code="manifest_invalid")
    report = _load_json(report_path, code="conversion_report_invalid")
    low_manifest = _load_json(low_manifest_path, code="low_budget_manifest_invalid")
    low_report = _load_json(low_report_path, code="low_budget_report_invalid")
    full_manifest = (
        manifest
        if manifest_path == full_manifest_path
        else _load_json(full_manifest_path, code="full_manifest_invalid")
    )
    full_report = (
        report
        if report_path == full_report_path
        else _load_json(full_report_path, code="full_conversion_report_invalid")
    )
    winner = _load_json(winner_bundle_path, code="frozen_winner_invalid")
    gepa_result = _load_json(gepa_result_path, code="scaled_gepa_result_invalid")
    optimization_plan = _load_json(optimization_plan_path, code="scaled_optimization_plan_invalid")

    for candidate_manifest, candidate_report in (
        (manifest_path, report),
        (low_manifest_path, low_report),
        (full_manifest_path, full_report),
    ):
        if candidate_report.get("manifest_sha256") != sha256_file(candidate_manifest):
            raise EvidenceInferenceFableRetrospectiveError("manifest_report_hash_mismatch")
        boundary = candidate_report.get("model_input_boundary")
        if not isinstance(boundary, Mapping) or (
            boundary.get("annotation_fields_rendered_to_model") is not False
            or boundary.get("labels_location") != "expected_output and label_paths only"
            or boundary.get("replacements_source") != "prompts_merged.csv only"
            or boundary.get("source_lines_source")
            != "all BODY.RESULTS lines from the article; never a gold window"
        ):
            raise EvidenceInferenceFableRetrospectiveError("model_input_boundary_drift")
    winner_prompt = winner_prompt_path.read_text(encoding="utf-8")
    winner_prompt_hash = sha256_file(winner_prompt_path)
    embedded_winner_prompt = winner.get("winner_prompt")
    official_result = gepa_result.get("official_gepa_result")
    candidates = official_result.get("candidates") if isinstance(official_result, Mapping) else None
    if not isinstance(candidates, list) or len(candidates) != config.scaled_gepa_candidate_count:
        raise EvidenceInferenceFableRetrospectiveError("scaled_gepa_candidates_invalid")
    first_candidate = candidates[0]
    if not isinstance(first_candidate, Mapping):
        raise EvidenceInferenceFableRetrospectiveError("scaled_gepa_seed_candidate_invalid")
    seed_prompt = first_candidate.get("extraction_prompt")
    if not isinstance(seed_prompt, str) or not seed_prompt:
        raise EvidenceInferenceFableRetrospectiveError("scaled_gepa_seed_prompt_missing")
    seed_prompt_hash = hashlib.sha256(seed_prompt.encode("utf-8")).hexdigest()
    embedded_winner_hash = (
        hashlib.sha256(embedded_winner_prompt.encode("utf-8")).hexdigest()
        if isinstance(embedded_winner_prompt, str)
        else None
    )
    recorded_seed = winner.get("seed_prompt_sha256")
    if (
        recorded_seed != seed_prompt_hash
        or optimization_plan.get("seed_prompt_sha256") != seed_prompt_hash
        or winner.get("winner_prompt_sha256") != winner_prompt_hash
        or winner.get("winner_prompt_file_sha256") != winner_prompt_hash
        or embedded_winner_hash != winner_prompt_hash
        or embedded_winner_prompt != winner_prompt
        or winner.get("manifest_file_sha256") != sha256_file(full_manifest_path)
        or winner.get("trace_file_sha256") != sha256_file(gepa_result_path)
        or winner.get("plan_file_sha256") != sha256_file(optimization_plan_path)
        or winner.get("candidate_count") != config.scaled_gepa_candidate_count
        or winner.get("seed_retained") is not config.scaled_gepa_seed_retained
        or winner.get("winner_index") != official_result.get("best_idx")
        or winner.get("stage") != "winner_frozen_before_test_payload_access"
        or winner.get("test_payload_opened") is not False
        or winner.get("test_labels_scored") is not False
    ):
        raise EvidenceInferenceFableRetrospectiveError("scaled_prompt_lineage_mismatch")
    if seed_prompt_hash == winner_prompt_hash:
        raise EvidenceInferenceFableRetrospectiveError(
            "scaled_seed_winner_prompt_hash_equality_forbidden"
        )
    low_test = low_manifest.get("test")
    if not isinstance(low_test, Mapping) or (
        low_test.get("rows") != config.expected_low_budget_test_examples
        or len(low_test.get("paper_ids", [])) != config.expected_low_budget_test_articles
    ):
        raise EvidenceInferenceFableRetrospectiveError("low_budget_population_count_mismatch")

    test = manifest.get("test")
    if not isinstance(test, Mapping):
        raise EvidenceInferenceFableRetrospectiveError("test_manifest_missing")
    example_ids = test.get("example_ids")
    paper_ids = test.get("paper_ids")
    if (
        not isinstance(example_ids, list)
        or not isinstance(paper_ids, list)
        or len(example_ids) != expected_examples
        or len(set(example_ids)) != expected_examples
        or len(paper_ids) != expected_articles
        or len(set(paper_ids)) != expected_articles
        or test.get("rows") != expected_examples
    ):
        raise EvidenceInferenceFableRetrospectiveError("test_population_count_mismatch")
    full_test = full_manifest.get("test")
    if not isinstance(full_test, Mapping) or (
        not set(example_ids).issubset(set(full_test.get("example_ids", [])))
        or not set(paper_ids).issubset(set(full_test.get("paper_ids", [])))
    ):
        raise EvidenceInferenceFableRetrospectiveError("pilot_full_population_relation_drift")
    if report.get("input_hashes", {}).get("prompts_merged.csv") != sha256_file(question_table):
        raise EvidenceInferenceFableRetrospectiveError("question_table_hash_mismatch")

    questions_by_id = _question_rows(question_table=question_table, expected_ids=set(example_ids))
    questions_by_article: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in questions_by_id.values():
        questions_by_article[row["paper_id"]].append(row)
    if set(questions_by_article) != set(paper_ids):
        raise EvidenceInferenceFableRetrospectiveError("question_article_mapping_mismatch")

    policies = {"seed": seed_prompt, "winner": winner_prompt}
    policy_hashes = {"seed": seed_prompt_hash, "winner": winner_prompt_hash}
    base_requests: dict[str, dict[Arm, dict[str, Any]]] = {}
    model_surfaces: dict[tuple[str, Arm], dict[str, Any]] = {}
    article_root = Path(config.article_text_root)
    for article_id in paper_ids:
        source_path = _safe_file(root, article_root / f"{article_id}.txt")
        source_bytes = source_path.read_bytes()
        try:
            source_text = source_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EvidenceInferenceFableRetrospectiveError("article_utf8_invalid") from exc
        lines = _content_lines(_source_lines(source_text))
        if not lines:
            raise EvidenceInferenceFableRetrospectiveError("article_results_lines_empty")
        rows = sorted(questions_by_article[article_id], key=lambda item: item["example_id"])
        ids = [row["example_id"] for row in rows]
        schema = _output_schema(ids)
        schema_sha = hash_canonical(schema)
        compiled = compile_anthropic_bounded_schema(
            original_schema=schema,
            full_acceptance_schema_sha256=schema_sha,
        )
        max_output = min(
            MAXIMUM_OUTPUT_TOKENS,
            BASE_OUTPUT_TOKENS + OUTPUT_TOKENS_PER_QUESTION * len(ids),
        )
        per_arm: dict[Arm, dict[str, Any]] = {}
        for arm in ("seed", "winner"):
            prompt = _render_prompt(
                frozen_policy=policies[arm],
                article_id=article_id,
                questions=rows,
                lines=lines,
            )
            model_surface = {
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "wire_schema": compiled.wire_schema,
            }
            model_facing_bytes = (
                len(SYSTEM_PROMPT.encode("utf-8"))
                + len(prompt.encode("utf-8"))
                + len(canonical_json_bytes(compiled.wire_schema))
            )
            per_arm[arm] = {
                "article_id": article_id,
                "question_count": len(ids),
                "example_ids": ids,
                "source_text_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "results_source_lines_sha256": hash_canonical(lines),
                "question_payload_sha256": hash_canonical(rows),
                "arm_policy_sha256": policy_hashes[arm],
                "system_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "full_acceptance_schema_sha256": schema_sha,
                "wire_schema_sha256": compiled.wire_schema_sha256,
                "compiled_schema_sha256": compiled.compiled_schema_sha256,
                "model_facing_surface_sha256": hash_canonical(model_surface),
                "model_facing_utf8_bytes": model_facing_bytes,
                "max_output_tokens": max_output,
                "cost": _cost(
                    model_facing_bytes=model_facing_bytes,
                    max_output_tokens=max_output,
                ),
            }
            model_surfaces[(article_id, arm)] = {
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "wire_schema": deepcopy(compiled.wire_schema),
            }
        base_requests[article_id] = per_arm

    roster: list[ArticleBatchRequestV1] = []
    ordered_articles = _request_order(population, list(base_requests))
    for article_offset, article_id in enumerate(ordered_articles):
        arm_order: list[Arm] = ["seed", "winner"] if article_offset % 2 == 0 else ["winner", "seed"]
        for arm in arm_order:
            execution_index = len(roster)
            base = base_requests[article_id][arm]
            request_payload = {
                "request_version": REQUEST_VERSION,
                "execution_index": execution_index,
                "population": population,
                "arm": arm,
                "request_key": (
                    f"ei-fable-retro-v1-{population.replace('_', '-')}-{arm}-{article_id.lower()}"
                ),
                **base,
            }
            roster.append(
                ArticleBatchRequestV1.model_validate(
                    {
                        **request_payload,
                        "request_sha256": hash_canonical(request_payload),
                    }
                )
            )
            if _model_surface_sink is not None:
                _model_surface_sink.append(deepcopy(model_surfaces[(article_id, arm)]))

    total_surface = sum(item.model_facing_utf8_bytes for item in roster)
    heuristic_input = math.ceil(
        total_surface
        * config.historical_sonnet_low_input_tokens
        / config.historical_sonnet_low_model_facing_bytes
    )
    heuristic_output = expected_examples * 2 * config.historical_completed_output_tokens_mean
    heuristic_cost = heuristic_input * INPUT_RATE + heuristic_output * OUTPUT_RATE
    input_hashes = {
        "config_file": sha256_file(config_source),
        "population_manifest": sha256_file(manifest_path),
        "population_conversion_report": sha256_file(report_path),
        "low_budget_manifest": sha256_file(low_manifest_path),
        "low_budget_conversion_report": sha256_file(low_report_path),
        "question_table": sha256_file(question_table),
        "scaled_winner_bundle": sha256_file(winner_bundle_path),
        "scaled_winner_prompt": winner_prompt_hash,
        "scaled_gepa_result": sha256_file(gepa_result_path),
        "scaled_optimization_plan": sha256_file(optimization_plan_path),
    }
    payload = {
        "plan_version": PLAN_VERSION,
        "status": "offline_prepared_zero_provider_calls",
        "mode": mode,
        "population": population,
        "config_sha256": hash_canonical(config.model_dump(mode="json")),
        "input_artifact_sha256s": input_hashes,
        "scaled_winner_bundle_sha256": sha256_file(winner_bundle_path),
        "scaled_winner_prompt_sha256": winner_prompt_hash,
        "scaled_winner_embedded_prompt_sha256": embedded_winner_hash,
        "scaled_winner_candidate_count": config.scaled_gepa_candidate_count,
        "recorded_seed_prompt_sha256": recorded_seed,
        "seed_prompt_extracted_from_scaled_trace_sha256": seed_prompt_hash,
        "prompt_hash_relationship": "distinct",
        "comparison_interpretation": (
            "exploratory_cross_model_transfer_on_historically_opened_test"
        ),
        "exploratory_cross_model_transfer_comparison_permitted": True,
        "confirmatory_gepa_improvement_claim_permitted": False,
        "prior_exposure": PriorExposureV1(),
        "prior_exposure_readiness_receipt_sha256": readiness.receipt_sha256,
        "unique_examples": expected_examples,
        "unique_articles": expected_articles,
        "arm_count": 2,
        "question_evaluations": expected_examples * 2,
        "request_count": expected_articles * 2,
        "roster": roster,
        "request_roster_sha256": hash_canonical([item.request_sha256 for item in roster]),
        "total_model_facing_utf8_bytes": total_surface,
        "total_diagnostic_known_input_token_ceiling": sum(
            item.cost.diagnostic_known_input_token_ceiling for item in roster
        ),
        "total_max_output_tokens": sum(item.max_output_tokens for item in roster),
        "total_diagnostic_known_surface_cost_usd_micros": sum(
            item.cost.diagnostic_known_surface_cost_usd_micros for item in roster
        ),
        "total_full_context_hard_liability_usd_micros": sum(
            item.cost.full_context_hard_liability_usd_micros for item in roster
        ),
        "historical_lower_heuristic_input_tokens": heuristic_input,
        "historical_lower_heuristic_output_tokens": heuristic_output,
        "historical_lower_heuristic_cost_usd_micros": heuristic_cost,
        "historical_heuristic_is_not_budget_authority": True,
        "primary_inference_unit": "article_cluster",
        "paired_interval_method": "paired_article_cluster_bootstrap",
        "paired_bootstrap_replicates": config.paired_article_cluster_bootstrap_replicates,
        "paired_bootstrap_seed": config.paired_article_cluster_bootstrap_seed,
        "paired_primary_metrics": [
            "direction_accuracy",
            "structured_output_reliability",
            "exact_grounding_reliability",
        ],
        "arm_order_balancing": ("label_blind_article_hash_order_with_strict_alternating_first_arm"),
        "model_input_policy": "all_BODY_RESULTS_lines_once_per_article",
        "scaled_optimizer_input_policy": ("single_question_fixed_results_passage_projection_v1"),
        "cross_model_and_batched_interface_transfer_only": True,
        "qualified_population_all_expected_eligible": True,
        "eligibility_negative_examples": 0,
        "eligibility_metric_claim_authority": False,
        "invalid_batch_counts_each_locked_question_incorrect": True,
        "eligible_false_counts_unconditional_grounded_answer_failure": True,
        "conditional_grounding_reported_separately": True,
        "pilot_population_is_subset_of_full_test": True,
        "pilot_is_mechanics_only_no_inferential_authority": True,
        "pilot_preflight_required_before_full_authorization": mode == "full_paired",
        "full_plan_must_be_frozen_before_pilot_execution": True,
        "pilot_results_may_not_change_prompt_schema_thresholds_or_full_roster": True,
        "full_population_score_requires_every_planned_request_terminal": True,
        "budget_exhaustion_before_intent_is_clean_terminal": True,
        "budget_exhaustion_after_intent_is_ambiguous_and_never_retried": True,
        "labels_opened_by_planner": False,
        "benchmark_jsonl_opened_by_planner": False,
        "annotations_opened_by_planner": False,
        "provider_calls_made": 0,
        "credentials_opened": False,
        "network_opened": False,
        "plan_contains_article_or_question_text": False,
        "retrospective_accuracy_claim_authority_before_execution": False,
        "confirmatory_claim_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return EvidenceInferenceFableRetrospectivePlanV1.model_validate(
        {**payload, "plan_sha256": hash_canonical(payload)}
    )


def require_confirmatory_gepa_improvement_claim_v1(
    plan: EvidenceInferenceFableRetrospectivePlanV1,
) -> None:
    """Fail closed because historical label exposure prevents a confirmatory claim."""

    if plan.prior_exposure.all_reference_labels_historically_opened:
        raise EvidenceInferenceFableRetrospectiveError(
            "confirmatory_gepa_improvement_forbidden_historically_opened_test"
        )
    raise EvidenceInferenceFableRetrospectiveError(
        "confirmatory_gepa_improvement_not_supported_by_retrospective_v1"
    )


def validate_evidence_inference_fable_retrospective_plan_v1(
    *,
    repository_root: Path,
    plan: EvidenceInferenceFableRetrospectivePlanV1 | Mapping[str, Any],
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> EvidenceInferenceFableRetrospectivePlanV1:
    """Externally replay a serialized plan from the current label-safe inputs."""

    try:
        observed = (
            plan
            if isinstance(plan, EvidenceInferenceFableRetrospectivePlanV1)
            else EvidenceInferenceFableRetrospectivePlanV1.model_validate(plan)
        )
    except ValueError as exc:
        raise EvidenceInferenceFableRetrospectiveError(
            "evidence_inference_fable_plan_contract_or_hash_invalid"
        ) from exc
    expected = freeze_evidence_inference_fable_retrospective_plan_v1(
        repository_root=repository_root,
        mode=observed.mode,
        config_path=config_path,
    )
    if observed != expected:
        raise EvidenceInferenceFableRetrospectiveError(
            "evidence_inference_fable_plan_external_replay_mismatch"
        )
    return observed


def write_evidence_inference_fable_retrospective_plan_v1(
    *,
    repository_root: Path,
    mode: ExecutionMode,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_path: Path | None = None,
    force: bool = False,
) -> EvidenceInferenceFableRetrospectivePlanV1:
    """Freeze and atomically persist one text-free, zero-provider-call plan."""

    root = repository_root.resolve(strict=True)
    default_path = {
        "pilot30_paired": DEFAULT_PILOT_PLAN_PATH,
        "pilot30_recovery_v2_paired": DEFAULT_RECOVERY_PILOT_PLAN_PATH,
        "full_paired": DEFAULT_FULL_PLAN_PATH,
    }[mode]
    selected = output_path or default_path
    if selected.is_absolute() or ".." in selected.parts:
        raise EvidenceInferenceFableRetrospectiveError(
            "evidence_inference_fable_plan_output_path_escape"
        )
    target = root / selected
    if target.is_symlink() or target.parent.is_symlink():
        raise EvidenceInferenceFableRetrospectiveError(
            "evidence_inference_fable_plan_output_symlink_forbidden"
        )
    plan = freeze_evidence_inference_fable_retrospective_plan_v1(
        repository_root=root,
        mode=mode,
        config_path=config_path,
    )
    atomic_write_json(target, plan, force=force)
    return plan


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_FULL_PLAN_PATH",
    "DEFAULT_PILOT_PLAN_PATH",
    "DEFAULT_RECOVERY_EXCLUSION_LEDGER_PATH",
    "DEFAULT_RECOVERY_EXECUTION_POLICY_PATH",
    "DEFAULT_RECOVERY_PILOT_PLAN_PATH",
    "FROZEN_FULL_PLAN_SHA256",
    "FROZEN_PILOT_PLAN_SHA256",
    "EvidenceInferenceFableRetrospectiveConfigV1",
    "EvidenceInferenceFableRetrospectiveError",
    "EvidenceInferenceFableRetrospectivePlanV1",
    "RecoveryPilotExclusionLedgerV2",
    "RecoveryPilotExecutionPolicyV2",
    "freeze_evidence_inference_fable_retrospective_plan_v1",
    "require_confirmatory_gepa_improvement_claim_v1",
    "validate_evidence_inference_fable_retrospective_plan_v1",
    "write_evidence_inference_fable_retrospective_plan_v1",
]
