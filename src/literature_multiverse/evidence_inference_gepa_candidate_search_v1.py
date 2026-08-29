"""Offline freeze for a bounded, multi-start Evidence Inference GEPA search.

The plan is intentionally execution-free.  It opens only the official train and
development split payloads, materializes article-disjoint representative memberships,
and freezes genuinely distinct code-owned prompt candidates before any model
evaluation.  It never opens or hashes the test JSONL, constructs a provider client,
reads credentials, or grants an improvement claim.

Two historical results are kept distinct: the obsolete first-pass pilot retained the
seed after one mutation, while the authoritative scaled local-model study produced a
distinct seven-candidate winner but no held-out improvement.  A new Fable-high search
is therefore a frontier-model transfer and structured-grounding experiment, not a
repair of a purportedly single-candidate repository history.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.prompt_optimization import (
    OptimizationExample,
    OptimizationSplitManifest,
    load_manifest_split,
    load_split_manifest,
)

PLAN_VERSION = "evidence-inference-gepa-candidate-search-plan-v1"
CONFIG_VERSION = "evidence-inference-gepa-candidate-search-config-v1"
CANDIDATE_VERSION = "evidence-inference-gepa-pre-evaluation-candidate-v1"
MEMBERSHIP_VERSION = "evidence-inference-gepa-representative-membership-v1"
DECISION_VERSION = "evidence-inference-gepa-development-decision-v1"
DEFAULT_CONFIG_PATH = Path(
    "configs/benchmarks/evidence-inference-gepa-candidate-search-v1.json"
)
SOURCE_PATH = Path(
    "src/literature_multiverse/evidence_inference_gepa_candidate_search_v1.py"
)

FABLE_MODEL = "claude-fable-5"
FABLE_EFFORT = "high"
FABLE_INPUT_RATE = Decimal("10")
FABLE_OUTPUT_RATE = Decimal("50")
USD_MICROS = Decimal("1000000")

OBJECTIVE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "objective": "extraction_correctness",
        "direction": "maximize",
        "definition": "exact_match_of_reported_intervention_vs_comparator_direction",
        "uses_reference_labels": True,
        "hard_gate": False,
    },
    {
        "objective": "formal_grounding_validity",
        "direction": "maximize",
        "definition": "exact_quote_containment_and_declared_results_line_membership",
        "uses_reference_labels": False,
        "hard_gate": True,
    },
    {
        "objective": "structured_output_validity",
        "direction": "maximize",
        "definition": "strict_json_schema_validation_without_repair",
        "uses_reference_labels": False,
        "hard_gate": True,
    },
    {
        "objective": "provider_usage_and_cost",
        "direction": "minimize",
        "definition": "reported_input_and_output_tokens_priced_at_frozen_rates",
        "uses_reference_labels": False,
        "hard_gate": False,
    },
)


class GEPACandidateSearchPlanError(ValueError):
    """An offline search-plan input or lineage contract failed closed."""


class _Frozen(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]


def _self_hash(model: ContractModel, field: str, code: str) -> None:
    if getattr(model, field) != hash_canonical(
        model.model_dump(mode="json", exclude={field})
    ):
        raise ValueError(code)


def _aware(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"gepa_candidate_search_timezone_required:{field}")
    return value


def _datetime_json(value: datetime) -> str:
    rendered = _aware(value, "datetime_json").isoformat()
    return f"{rendered[:-6]}Z" if rendered.endswith("+00:00") else rendered


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized_prompt(value: str) -> str:
    return " ".join(value.casefold().split())


def _usd_micros(value: Decimal) -> int:
    micros = value * USD_MICROS
    if micros != micros.to_integral_value():
        raise GEPACandidateSearchPlanError("gepa_candidate_search_cost_not_exact_micros")
    return int(micros)


def _call_cost_micros(
    *, input_tokens: int, output_tokens: int, input_rate: Decimal, output_rate: Decimal
) -> int:
    cost = (
        Decimal(input_tokens) * input_rate / Decimal("1000000")
        + Decimal(output_tokens) * output_rate / Decimal("1000000")
    )
    return _usd_micros(cost)


def _relative_path(value: Path, field: str) -> Path:
    if value.is_absolute() or ".." in value.parts or value.as_posix() != str(value):
        raise ValueError(f"gepa_candidate_search_relative_path_required:{field}")
    return value


def _repository_file(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise GEPACandidateSearchPlanError("gepa_candidate_search_path_escape")
    candidate = root / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise GEPACandidateSearchPlanError(
            f"gepa_candidate_search_file_invalid:{relative.as_posix()}"
        )
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise GEPACandidateSearchPlanError("gepa_candidate_search_path_escape") from exc
    return resolved


def _repository_directory(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise GEPACandidateSearchPlanError("gepa_candidate_search_path_escape")
    candidate = root / relative
    if candidate.is_symlink() or not candidate.is_dir():
        raise GEPACandidateSearchPlanError(
            f"gepa_candidate_search_directory_invalid:{relative.as_posix()}"
        )
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise GEPACandidateSearchPlanError("gepa_candidate_search_path_escape") from exc
    return resolved


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GEPACandidateSearchPlanError(f"gepa_candidate_search_{label}_unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GEPACandidateSearchPlanError(
            f"gepa_candidate_search_{label}_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise GEPACandidateSearchPlanError(
            f"gepa_candidate_search_{label}_not_object"
        )
    return value


class FableProviderPlanConfigV1(_Frozen):
    provider: Literal["anthropic_first_party_api"]
    model: Literal["claude-fable-5"]
    effort: Literal["high"]
    service_tier: Literal["standard_only"]
    transport: Literal["structured_json_schema"]
    input_rate_usd_per_million_tokens: Decimal
    output_rate_usd_per_million_tokens: Decimal
    task_input_token_ceiling: Annotated[int, Field(ge=1)]
    task_output_token_ceiling: Annotated[int, Field(ge=1)]
    reflection_input_token_ceiling: Annotated[int, Field(ge=1)]
    reflection_output_token_ceiling: Annotated[int, Field(ge=1)]
    sdk_retries_per_request: Literal[0]
    application_retries_per_request: Literal[0]
    fallback_requests_permitted: Literal[0]

    @model_validator(mode="after")
    def validate_provider(self) -> FableProviderPlanConfigV1:
        if (
            self.input_rate_usd_per_million_tokens != FABLE_INPUT_RATE
            or self.output_rate_usd_per_million_tokens != FABLE_OUTPUT_RATE
            or self.task_input_token_ceiling != 16384
            or self.task_output_token_ceiling != 1024
            or self.reflection_input_token_ceiling != 24576
            or self.reflection_output_token_ceiling != 4096
        ):
            raise ValueError("gepa_candidate_search_provider_ceiling_drift")
        return self


class MutationTemplateConfigV1(_Frozen):
    candidate_id: Annotated[str, Field(pattern=r"^candidate-[a-z0-9-]+$")]
    mutation_axis: Annotated[str, Field(pattern=r"^[a-z0-9_]+$")]
    instruction: Annotated[str, Field(min_length=80, max_length=1200)]


class SearchTierConfigV1(_Frozen):
    tier: Literal["cheap_pilot", "scaled"]
    initial_mutation_count: Annotated[int, Field(ge=3)]
    train_representative_articles: Annotated[int, Field(ge=1)]
    reflection_call_ceiling: Annotated[int, Field(ge=1)]
    accepted_reflection_candidates_required: Annotated[int, Field(ge=1)]
    dev_search_representative_articles: Annotated[int, Field(ge=1)]
    dev_confirmation_representative_articles: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def validate_tier(self) -> SearchTierConfigV1:
        expected = {
            "cheap_pilot": (3, 4, 2, 1, 6, 6),
            "scaled": (6, 16, 6, 3, 32, 32),
        }[self.tier]
        observed = (
            self.initial_mutation_count,
            self.train_representative_articles,
            self.reflection_call_ceiling,
            self.accepted_reflection_candidates_required,
            self.dev_search_representative_articles,
            self.dev_confirmation_representative_articles,
        )
        if observed != expected:
            raise ValueError("gepa_candidate_search_tier_budget_drift")
        if self.accepted_reflection_candidates_required > self.reflection_call_ceiling:
            raise ValueError("gepa_candidate_search_reflection_acceptance_impossible")
        return self


class GEPACandidateSearchConfigV1(_Frozen):
    config_version: Literal["evidence-inference-gepa-candidate-search-config-v1"]
    manifest_path: Path
    seed_prompt_path: Path
    legacy_pilot_directory: Path
    prior_scaled_directory: Path
    prior_scaled_public_summary_path: Path
    output_plan_path: Path
    subset_seed: Literal[20260829]
    provider: FableProviderPlanConfigV1
    mutation_templates: Annotated[list[MutationTemplateConfigV1], Field(min_length=6)]
    tiers: Annotated[list[SearchTierConfigV1], Field(min_length=2, max_length=2)]

    @field_validator(
        "manifest_path",
        "seed_prompt_path",
        "legacy_pilot_directory",
        "prior_scaled_directory",
        "prior_scaled_public_summary_path",
        "output_plan_path",
    )
    @classmethod
    def validate_path(cls, value: Path, info: Any) -> Path:
        return _relative_path(value, info.field_name)

    @model_validator(mode="after")
    def validate_config(self) -> GEPACandidateSearchConfigV1:
        if [item.tier for item in self.tiers] != ["cheap_pilot", "scaled"]:
            raise ValueError("gepa_candidate_search_tier_order_invalid")
        candidate_ids = [item.candidate_id for item in self.mutation_templates]
        axes = [item.mutation_axis for item in self.mutation_templates]
        instructions = [item.instruction for item in self.mutation_templates]
        if (
            len(candidate_ids) != len(set(candidate_ids))
            or len(axes) != len(set(axes))
            or len(instructions) != len(set(instructions))
        ):
            raise ValueError("gepa_candidate_search_mutation_templates_not_canonical")
        if max(item.initial_mutation_count for item in self.tiers) > len(
            self.mutation_templates
        ):
            raise ValueError("gepa_candidate_search_insufficient_mutation_templates")
        return self


class PreEvaluationCandidateV1(_Frozen):
    candidate_version: Literal[
        "evidence-inference-gepa-pre-evaluation-candidate-v1"
    ] = CANDIDATE_VERSION
    candidate_id: str
    role: Literal["handwritten_seed", "code_owned_diverse_start"]
    mutation_axis: str
    prompt_text: str
    prompt_sha256: Sha256
    normalized_prompt_sha256: Sha256
    seed_prompt_sha256: Sha256
    created_before_any_task_evaluation: Literal[True] = True
    provider_generated: Literal[False] = False
    development_labels_consulted: Literal[False] = False
    test_payload_opened: Literal[False] = False
    candidate_sha256: Sha256

    @model_validator(mode="after")
    def validate_candidate(self) -> PreEvaluationCandidateV1:
        if (
            self.prompt_sha256 != _sha256_text(self.prompt_text)
            or self.normalized_prompt_sha256
            != _sha256_text(_normalized_prompt(self.prompt_text))
        ):
            raise ValueError("gepa_candidate_search_candidate_prompt_hash_mismatch")
        placeholders = (
            self.prompt_text.count("[[OUTCOME]]"),
            self.prompt_text.count("[[INTERVENTION]]"),
            self.prompt_text.count("[[COMPARATOR]]"),
        )
        if placeholders != (1, 1, 1):
            raise ValueError("gepa_candidate_search_candidate_placeholder_contract")
        if self.role == "handwritten_seed":
            if self.prompt_sha256 != self.seed_prompt_sha256 or self.mutation_axis != "seed":
                raise ValueError("gepa_candidate_search_seed_candidate_mismatch")
        elif self.prompt_sha256 == self.seed_prompt_sha256 or self.mutation_axis == "seed":
            raise ValueError("gepa_candidate_search_mutation_not_distinct_from_seed")
        _self_hash(
            self,
            "candidate_sha256",
            "gepa_candidate_search_candidate_self_hash_mismatch",
        )
        return self


class RepresentativeMembershipV1(_Frozen):
    membership_version: Literal[
        "evidence-inference-gepa-representative-membership-v1"
    ] = MEMBERSHIP_VERSION
    tier: Literal["cheap_pilot", "scaled"]
    phase: Literal["train_trajectories", "dev_search", "dev_confirmation"]
    official_split: Literal["train", "dev"]
    selection_algorithm: Literal[
        "namespaced_sha256_ranked_articles_then_one_namespaced_sha256_example-v1"
    ] = "namespaced_sha256_ranked_articles_then_one_namespaced_sha256_example-v1"
    example_ids: list[str]
    paper_ids: list[str]
    group_ids: list[str]
    representatives: Annotated[int, Field(ge=1)]
    selected_payload_sha256: Sha256
    label_free_identity_sha256: Sha256
    membership_sha256: Sha256

    @model_validator(mode="after")
    def validate_membership(self) -> RepresentativeMembershipV1:
        if (
            self.example_ids != sorted(set(self.example_ids))
            or self.paper_ids != sorted(set(self.paper_ids))
            or self.group_ids != sorted(set(self.group_ids))
            or not (
                len(self.example_ids)
                == len(self.paper_ids)
                == len(self.group_ids)
                == self.representatives
            )
        ):
            raise ValueError("gepa_candidate_search_membership_not_one_per_article")
        _self_hash(
            self,
            "membership_sha256",
            "gepa_candidate_search_membership_self_hash_mismatch",
        )
        return self


class TierCallBudgetV1(_Frozen):
    tier: Literal["cheap_pilot", "scaled"]
    initial_candidate_count: Annotated[int, Field(ge=4)]
    initial_distinct_nonseed_candidate_count: Annotated[int, Field(ge=3)]
    initial_train_task_call_ceiling: Annotated[int, Field(ge=1)]
    reflection_call_ceiling: Annotated[int, Field(ge=1)]
    accepted_distinct_reflection_candidates_required: Annotated[int, Field(ge=1)]
    maximum_dev_candidate_count: Annotated[int, Field(ge=5)]
    dev_search_task_call_ceiling: Annotated[int, Field(ge=1)]
    seed_nonseed_confirmation_arms: Literal[2] = 2
    calls_per_confirmation_arm: Annotated[int, Field(ge=1)]
    confirmation_task_call_ceiling: Annotated[int, Field(ge=2)]
    task_provider_call_ceiling: Annotated[int, Field(ge=1)]
    total_provider_call_ceiling: Annotated[int, Field(ge=1)]
    task_call_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    reflection_call_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    task_cost_liability_usd_micros: Annotated[int, Field(ge=1)]
    reflection_cost_liability_usd_micros: Annotated[int, Field(ge=1)]
    total_hard_cost_liability_usd_micros: Annotated[int, Field(ge=1)]
    application_retries: Literal[0] = 0
    sdk_retries: Literal[0] = 0
    fallback_calls: Literal[0] = 0
    failures_and_duplicates_consume_ceiling: Literal[True] = True
    no_early_stopping_within_a_candidate_comparison: Literal[True] = True
    budget_sha256: Sha256

    @model_validator(mode="after")
    def validate_budget(self) -> TierCallBudgetV1:
        if (
            self.initial_distinct_nonseed_candidate_count
            != self.initial_candidate_count - 1
            or self.maximum_dev_candidate_count
            != self.initial_candidate_count
            + self.accepted_distinct_reflection_candidates_required
            or self.confirmation_task_call_ceiling
            != self.seed_nonseed_confirmation_arms * self.calls_per_confirmation_arm
            or self.task_provider_call_ceiling
            != self.initial_train_task_call_ceiling
            + self.dev_search_task_call_ceiling
            + self.confirmation_task_call_ceiling
            or self.total_provider_call_ceiling
            != self.task_provider_call_ceiling + self.reflection_call_ceiling
            or self.task_cost_liability_usd_micros
            != self.task_provider_call_ceiling
            * self.task_call_cost_ceiling_usd_micros
            or self.reflection_cost_liability_usd_micros
            != self.reflection_call_ceiling
            * self.reflection_call_cost_ceiling_usd_micros
            or self.total_hard_cost_liability_usd_micros
            != self.task_cost_liability_usd_micros
            + self.reflection_cost_liability_usd_micros
        ):
            raise ValueError("gepa_candidate_search_budget_arithmetic_mismatch")
        _self_hash(
            self,
            "budget_sha256",
            "gepa_candidate_search_budget_self_hash_mismatch",
        )
        return self


class SearchTierPlanV1(_Frozen):
    tier: Literal["cheap_pilot", "scaled"]
    pre_evaluation_candidates: Annotated[list[PreEvaluationCandidateV1], Field(min_length=4)]
    train_membership: RepresentativeMembershipV1
    dev_search_membership: RepresentativeMembershipV1
    dev_confirmation_membership: RepresentativeMembershipV1
    call_budget: TierCallBudgetV1
    reflection_stage_occurs_only_after_initial_train_trajectories: Literal[True] = True
    reflected_candidate_schema_requires_full_prompt_and_mutation_rationale: Literal[True] = True
    reflected_candidates_must_pass_placeholder_and_normalized_hash_uniqueness: Literal[
        True
    ] = True
    reflected_candidate_gate_before_dev_search: Literal[
        "exact_required_distinct_count_or_fail_closed_without_dev_evaluation"
    ] = "exact_required_distinct_count_or_fail_closed_without_dev_evaluation"
    all_dev_candidates_receive_identical_search_membership_and_one_attempt_each: Literal[
        True
    ] = True
    seed_and_selected_nonseed_receive_identical_confirmation_membership: Literal[
        True
    ] = True
    candidate_search_mechanics_only: Literal[True] = True
    improvement_authority: Literal[False] = False
    test_evaluation_authority: Literal[False] = False
    tier_sha256: Sha256

    @model_validator(mode="after")
    def validate_tier(self) -> SearchTierPlanV1:
        candidates = self.pre_evaluation_candidates
        ids = [item.candidate_id for item in candidates]
        hashes = [item.prompt_sha256 for item in candidates]
        normalized = [item.normalized_prompt_sha256 for item in candidates]
        axes = [item.mutation_axis for item in candidates]
        if (
            candidates[0].role != "handwritten_seed"
            or any(item.role != "code_owned_diverse_start" for item in candidates[1:])
            or len(ids) != len(set(ids))
            or len(hashes) != len(set(hashes))
            or len(normalized) != len(set(normalized))
            or len(axes) != len(set(axes))
            or len(candidates) != self.call_budget.initial_candidate_count
        ):
            raise ValueError("gepa_candidate_search_pre_evaluation_diversity_missing")
        memberships = (
            self.train_membership,
            self.dev_search_membership,
            self.dev_confirmation_membership,
        )
        if any(item.tier != self.tier for item in memberships):
            raise ValueError("gepa_candidate_search_tier_membership_alias_mismatch")
        if (
            set(self.train_membership.paper_ids)
            & (
                set(self.dev_search_membership.paper_ids)
                | set(self.dev_confirmation_membership.paper_ids)
            )
            or set(self.dev_search_membership.paper_ids)
            & set(self.dev_confirmation_membership.paper_ids)
        ):
            raise ValueError("gepa_candidate_search_article_overlap")
        if (
            self.call_budget.initial_train_task_call_ceiling
            != len(candidates) * self.train_membership.representatives
            or self.call_budget.dev_search_task_call_ceiling
            != self.call_budget.maximum_dev_candidate_count
            * self.dev_search_membership.representatives
            or self.call_budget.calls_per_confirmation_arm
            != self.dev_confirmation_membership.representatives
        ):
            raise ValueError("gepa_candidate_search_budget_membership_mismatch")
        _self_hash(
            self,
            "tier_sha256",
            "gepa_candidate_search_tier_self_hash_mismatch",
        )
        return self


class PriorGEPADiagnosisV1(_Frozen):
    obsolete_first_pass_trace_file_sha256: Sha256
    obsolete_first_pass_winner_file_sha256: Sha256
    obsolete_first_pass_seed_prompt_sha256: Sha256
    obsolete_first_pass_winner_prompt_sha256: Sha256
    obsolete_first_pass_candidate_count: Literal[2] = 2
    obsolete_first_pass_distinct_mutation_count: Literal[1] = 1
    obsolete_first_pass_metric_calls: Literal[46] = 46
    obsolete_first_pass_declared_metric_cap: Literal[40] = 40
    obsolete_first_pass_seed_dev_score: float
    obsolete_first_pass_mutation_dev_score: float
    obsolete_first_pass_seed_retained: Literal[True] = True
    obsolete_first_pass_test_payload_opened_during_optimization: Literal[False] = False
    authoritative_scaled_plan_file_sha256: Sha256
    authoritative_scaled_trace_file_sha256: Sha256
    authoritative_scaled_winner_file_sha256: Sha256
    authoritative_scaled_public_summary_file_sha256: Sha256
    authoritative_scaled_seed_prompt_sha256: Sha256
    authoritative_scaled_winner_prompt_sha256: Sha256
    authoritative_scaled_candidate_count: Literal[7] = 7
    authoritative_scaled_reflection_proposals: Literal[8] = 8
    authoritative_scaled_seed_retained: Literal[False] = False
    authoritative_scaled_observed_improvement_rule_satisfied: Literal[False] = False
    authoritative_scaled_status: Literal["no_improvement_claim"] = "no_improvement_claim"
    authoritative_scaled_test_payload_opened_before_winner_freeze: Literal[False] = False
    authoritative_scaled_paired_report_file_sha256_declared_not_recomputed: Sha256
    authoritative_scaled_paired_report_opened_by_this_planner: Literal[False] = False
    new_search_rationale: Literal[
        "frontier_fable_high_transfer_with_separate_structured_grounding_objectives"
    ] = "frontier_fable_high_transfer_with_separate_structured_grounding_objectives"
    diagnosis_sha256: Sha256

    @model_validator(mode="after")
    def validate_diagnosis(self) -> PriorGEPADiagnosisV1:
        if (
            self.obsolete_first_pass_seed_prompt_sha256
            != self.obsolete_first_pass_winner_prompt_sha256
            or self.authoritative_scaled_seed_prompt_sha256
            == self.authoritative_scaled_winner_prompt_sha256
        ):
            raise ValueError("gepa_candidate_search_prior_result_distinction_lost")
        _self_hash(
            self,
            "diagnosis_sha256",
            "gepa_candidate_search_diagnosis_self_hash_mismatch",
        )
        return self


class EvidenceInferenceGEPACandidateSearchPlanV1(_Frozen):
    plan_version: Literal["evidence-inference-gepa-candidate-search-plan-v1"] = (
        PLAN_VERSION
    )
    frozen_at: datetime
    status: Literal["offline_multi_start_plan_frozen_zero_provider_calls"] = (
        "offline_multi_start_plan_frozen_zero_provider_calls"
    )
    config_file_sha256: Sha256
    planner_source_sha256: Sha256
    manifest_file_sha256: Sha256
    manifest_source_examples_sha256: Sha256
    train_split_sha256: Sha256
    dev_split_sha256: Sha256
    test_split_declared_sha256_not_recomputed: Sha256
    seed_prompt_sha256: Sha256
    provider: FableProviderPlanConfigV1
    provider_identity_sha256: Sha256
    objectives: Annotated[list[dict[str, Any]], Field(min_length=4, max_length=4)]
    objective_membership_sha256: Sha256
    optimizer_contract: Literal[
        "gepa_style_reflective_multi_start_search_requires_separate_runtime_adapter"
    ] = "gepa_style_reflective_multi_start_search_requires_separate_runtime_adapter"
    selection_rule: Literal[
        "schema_and_grounding_noninferior_to_seed_then_maximize_extraction_correctness_then_minimize_cost_then_candidate_sha"
    ] = (
        "schema_and_grounding_noninferior_to_seed_then_maximize_extraction_correctness_then_minimize_cost_then_candidate_sha"
    )
    prior_diagnosis: PriorGEPADiagnosisV1
    tiers: Annotated[list[SearchTierPlanV1], Field(min_length=2, max_length=2)]
    official_train_dev_article_overlap: Literal[0] = 0
    official_train_dev_group_overlap: Literal[0] = 0
    multiple_distinct_candidates_frozen_before_first_evaluation: Literal[True] = True
    task_and_reflection_model_identical_fable_high: Literal[True] = True
    provider_calls_made: Literal[0] = 0
    reflection_calls_made: Literal[0] = 0
    task_calls_made: Literal[0] = 0
    credentials_read: Literal[False] = False
    network_accessed: Literal[False] = False
    test_payload_opened: Literal[False] = False
    test_payload_hashed: Literal[False] = False
    test_labels_opened: Literal[False] = False
    test_labels_scored: Literal[False] = False
    test_example_ids_materialized_in_plan: Literal[False] = False
    winner_seed_equality_rule: Literal[
        "if_winner_prompt_sha_equals_seed_report_seed_retained_negative_result_and_stop"
    ] = "if_winner_prompt_sha_equals_seed_report_seed_retained_negative_result_and_stop"
    future_paired_evaluation_rule: Literal[
        "only_after_nonseed_winner_freeze_use_same_examples_one_attempt_per_arm_equal_calls_and_balanced_order"
    ] = (
        "only_after_nonseed_winner_freeze_use_same_examples_one_attempt_per_arm_"
        "equal_calls_and_balanced_order"
    )
    historical_labels_opened_elsewhere_pristine_claim: Literal[False] = False
    improvement_authority: Literal[False] = False
    reliability_authority: Literal[False] = False
    generalization_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    plan_sha256: Sha256

    @field_validator("frozen_at")
    @classmethod
    def validate_frozen_at(cls, value: datetime) -> datetime:
        return _aware(value, "frozen_at")

    @model_validator(mode="after")
    def validate_plan(self) -> EvidenceInferenceGEPACandidateSearchPlanV1:
        if self.objectives != list(OBJECTIVE_SPECS) or self.objective_membership_sha256 != (
            hash_canonical(self.objectives)
        ):
            raise ValueError("gepa_candidate_search_objective_contract_mismatch")
        if [item.tier for item in self.tiers] != ["cheap_pilot", "scaled"]:
            raise ValueError("gepa_candidate_search_plan_tier_order_invalid")
        provider_identity = {
            "provider": self.provider.provider,
            "model": self.provider.model,
            "effort": self.provider.effort,
            "service_tier": self.provider.service_tier,
            "transport": self.provider.transport,
            "input_rate_usd_per_million_tokens": str(
                self.provider.input_rate_usd_per_million_tokens
            ),
            "output_rate_usd_per_million_tokens": str(
                self.provider.output_rate_usd_per_million_tokens
            ),
            "task_input_token_ceiling": self.provider.task_input_token_ceiling,
            "task_output_token_ceiling": self.provider.task_output_token_ceiling,
            "reflection_input_token_ceiling": self.provider.reflection_input_token_ceiling,
            "reflection_output_token_ceiling": (
                self.provider.reflection_output_token_ceiling
            ),
            "sdk_retries_per_request": self.provider.sdk_retries_per_request,
            "application_retries_per_request": (
                self.provider.application_retries_per_request
            ),
            "fallback_requests_permitted": (
                self.provider.fallback_requests_permitted
            ),
            "pricing_source": "repository_frozen_provider_rate_table_verified_2026_08_29",
        }
        if self.provider_identity_sha256 != hash_canonical(provider_identity):
            raise ValueError("gepa_candidate_search_provider_identity_mismatch")
        if any(
            item.pre_evaluation_candidates[0].prompt_sha256 != self.seed_prompt_sha256
            for item in self.tiers
        ):
            raise ValueError("gepa_candidate_search_seed_alias_mismatch")
        _self_hash(
            self,
            "plan_sha256",
            "gepa_candidate_search_plan_self_hash_mismatch",
        )
        return self


class GEPACandidateSearchDevelopmentDecisionV1(_Frozen):
    decision_version: Literal["evidence-inference-gepa-development-decision-v1"] = (
        DECISION_VERSION
    )
    plan_sha256: Sha256
    tier_sha256: Sha256
    seed_prompt_sha256: Sha256
    winner_prompt_sha256: Sha256
    status: Literal[
        "seed_retained_negative_result",
        "nonseed_development_candidate_frozen_no_improvement_claim",
    ]
    winner_is_seed: bool
    equal_budget_future_evaluation_required: bool
    future_evaluation_performed: Literal[False] = False
    improvement_authority: Literal[False] = False
    generalization_authority: Literal[False] = False
    decision_sha256: Sha256

    @model_validator(mode="after")
    def validate_decision(self) -> GEPACandidateSearchDevelopmentDecisionV1:
        if self.winner_is_seed:
            if (
                self.winner_prompt_sha256 != self.seed_prompt_sha256
                or self.status != "seed_retained_negative_result"
                or self.equal_budget_future_evaluation_required
            ):
                raise ValueError("gepa_candidate_search_seed_negative_result_mismatch")
        elif (
            self.winner_prompt_sha256 == self.seed_prompt_sha256
            or self.status != "nonseed_development_candidate_frozen_no_improvement_claim"
            or not self.equal_budget_future_evaluation_required
        ):
            raise ValueError("gepa_candidate_search_nonseed_decision_mismatch")
        _self_hash(
            self,
            "decision_sha256",
            "gepa_candidate_search_decision_self_hash_mismatch",
        )
        return self


def load_gepa_candidate_search_config_v1(
    *, repository_root: Path, config_path: Path = DEFAULT_CONFIG_PATH
) -> GEPACandidateSearchConfigV1:
    root = repository_root.resolve(strict=True)
    source = _repository_file(root, config_path)
    return GEPACandidateSearchConfigV1.model_validate(
        _read_object(source, "config")
    )


def _candidate(
    *,
    candidate_id: str,
    role: Literal["handwritten_seed", "code_owned_diverse_start"],
    mutation_axis: str,
    prompt_text: str,
    seed_prompt_sha256: str,
) -> PreEvaluationCandidateV1:
    payload = {
        "candidate_version": CANDIDATE_VERSION,
        "candidate_id": candidate_id,
        "role": role,
        "mutation_axis": mutation_axis,
        "prompt_text": prompt_text,
        "prompt_sha256": _sha256_text(prompt_text),
        "normalized_prompt_sha256": _sha256_text(_normalized_prompt(prompt_text)),
        "seed_prompt_sha256": seed_prompt_sha256,
        "created_before_any_task_evaluation": True,
        "provider_generated": False,
        "development_labels_consulted": False,
        "test_payload_opened": False,
    }
    return PreEvaluationCandidateV1.model_validate(
        {**payload, "candidate_sha256": hash_canonical(payload)}
    )


def _initial_candidates(
    *,
    seed_text: str,
    templates: Sequence[MutationTemplateConfigV1],
    count: int,
) -> list[PreEvaluationCandidateV1]:
    seed_sha = _sha256_text(seed_text)
    candidates = [
        _candidate(
            candidate_id="candidate-seed",
            role="handwritten_seed",
            mutation_axis="seed",
            prompt_text=seed_text,
            seed_prompt_sha256=seed_sha,
        )
    ]
    for template in templates[:count]:
        prompt = (
            seed_text.rstrip()
            + "\n\n## Candidate-specific decision rule\n\n"
            + template.instruction
            + "\n"
        )
        candidates.append(
            _candidate(
                candidate_id=template.candidate_id,
                role="code_owned_diverse_start",
                mutation_axis=template.mutation_axis,
                prompt_text=prompt,
                seed_prompt_sha256=seed_sha,
            )
        )
    return candidates


def _ranked_representatives(
    examples: Sequence[OptimizationExample],
    *,
    namespace: str,
    count: int,
    excluded_papers: set[str] | None = None,
) -> list[OptimizationExample]:
    excluded = excluded_papers or set()
    by_paper: dict[str, list[OptimizationExample]] = defaultdict(list)
    for example in examples:
        if example.paper_id not in excluded:
            by_paper[example.paper_id].append(example)
    ranked_papers = sorted(
        by_paper,
        key=lambda paper_id: (
            _sha256_text(f"{namespace}:paper:{paper_id}"),
            paper_id,
        ),
    )
    if len(ranked_papers) < count:
        raise GEPACandidateSearchPlanError(
            f"gepa_candidate_search_insufficient_articles:{namespace}"
        )
    representatives: list[OptimizationExample] = []
    for paper_id in ranked_papers[:count]:
        representatives.append(
            min(
                by_paper[paper_id],
                key=lambda item: (
                    _sha256_text(f"{namespace}:example:{item.example_id}"),
                    item.example_id,
                ),
            )
        )
    return representatives


def _membership(
    *,
    tier: Literal["cheap_pilot", "scaled"],
    phase: Literal["train_trajectories", "dev_search", "dev_confirmation"],
    official_split: Literal["train", "dev"],
    examples: Sequence[OptimizationExample],
) -> RepresentativeMembershipV1:
    ordered = sorted(examples, key=lambda item: item.example_id)
    identities = [
        {
            "example_id": item.example_id,
            "paper_id": item.paper_id,
            "group_id": item.group_id,
        }
        for item in ordered
    ]
    payload = {
        "membership_version": MEMBERSHIP_VERSION,
        "tier": tier,
        "phase": phase,
        "official_split": official_split,
        "selection_algorithm": (
            "namespaced_sha256_ranked_articles_then_one_namespaced_sha256_example-v1"
        ),
        "example_ids": sorted(item.example_id for item in ordered),
        "paper_ids": sorted(item.paper_id for item in ordered),
        "group_ids": sorted(item.group_id for item in ordered),
        "representatives": len(ordered),
        "selected_payload_sha256": hash_canonical(
            [item.model_dump(mode="json") for item in ordered]
        ),
        "label_free_identity_sha256": hash_canonical(identities),
    }
    return RepresentativeMembershipV1.model_validate(
        {**payload, "membership_sha256": hash_canonical(payload)}
    )


def _budget(
    *,
    tier: SearchTierConfigV1,
    initial_candidate_count: int,
    task_call_cost_micros: int,
    reflection_call_cost_micros: int,
) -> TierCallBudgetV1:
    initial_train = initial_candidate_count * tier.train_representative_articles
    maximum_dev_candidates = (
        initial_candidate_count + tier.accepted_reflection_candidates_required
    )
    dev_search = maximum_dev_candidates * tier.dev_search_representative_articles
    confirmation = 2 * tier.dev_confirmation_representative_articles
    task_calls = initial_train + dev_search + confirmation
    payload = {
        "tier": tier.tier,
        "initial_candidate_count": initial_candidate_count,
        "initial_distinct_nonseed_candidate_count": initial_candidate_count - 1,
        "initial_train_task_call_ceiling": initial_train,
        "reflection_call_ceiling": tier.reflection_call_ceiling,
        "accepted_distinct_reflection_candidates_required": (
            tier.accepted_reflection_candidates_required
        ),
        "maximum_dev_candidate_count": maximum_dev_candidates,
        "dev_search_task_call_ceiling": dev_search,
        "seed_nonseed_confirmation_arms": 2,
        "calls_per_confirmation_arm": tier.dev_confirmation_representative_articles,
        "confirmation_task_call_ceiling": confirmation,
        "task_provider_call_ceiling": task_calls,
        "total_provider_call_ceiling": task_calls + tier.reflection_call_ceiling,
        "task_call_cost_ceiling_usd_micros": task_call_cost_micros,
        "reflection_call_cost_ceiling_usd_micros": reflection_call_cost_micros,
        "task_cost_liability_usd_micros": task_calls * task_call_cost_micros,
        "reflection_cost_liability_usd_micros": (
            tier.reflection_call_ceiling * reflection_call_cost_micros
        ),
        "total_hard_cost_liability_usd_micros": (
            task_calls * task_call_cost_micros
            + tier.reflection_call_ceiling * reflection_call_cost_micros
        ),
        "application_retries": 0,
        "sdk_retries": 0,
        "fallback_calls": 0,
        "failures_and_duplicates_consume_ceiling": True,
        "no_early_stopping_within_a_candidate_comparison": True,
    }
    return TierCallBudgetV1.model_validate(
        {**payload, "budget_sha256": hash_canonical(payload)}
    )


def _prior_diagnosis(
    *,
    root: Path,
    config: GEPACandidateSearchConfigV1,
    seed_sha256: str,
) -> PriorGEPADiagnosisV1:
    legacy = _repository_directory(root, config.legacy_pilot_directory)
    legacy_trace_path = _repository_file(
        root, config.legacy_pilot_directory / "optimization_trace.json"
    )
    legacy_winner_path = _repository_file(
        root, config.legacy_pilot_directory / "frozen_winner.json"
    )
    legacy_trace = _read_object(legacy_trace_path, "legacy_trace")
    legacy_winner = _read_object(legacy_winner_path, "legacy_winner")
    component = legacy_trace.get("component_traces", {}).get("extraction")
    if not isinstance(component, Mapping):
        raise GEPACandidateSearchPlanError("gepa_candidate_search_legacy_trace_invalid")
    legacy_candidates = component.get("candidates")
    legacy_hashes = component.get("candidate_sha256s")
    legacy_scores = component.get("val_aggregate_scores")
    frozen_prompt = _repository_file(
        root, config.legacy_pilot_directory / "frozen_extraction.md"
    )
    if (
        legacy_trace.get("optimization_splits") != ["train", "dev"]
        or legacy_trace.get("test_split_opened") is not False
        or legacy_trace.get("test_evaluated") is not False
        or legacy_winner.get("test_evaluated_at_freeze") is not False
        or legacy_trace.get("seed_prompt_sha256s", {}).get("extraction") != seed_sha256
        or legacy_trace.get("winning_prompt_sha256s", {}).get("extraction")
        != seed_sha256
        or legacy_winner.get("seed_prompt_sha256s", {}).get("extraction") != seed_sha256
        or sha256_file(frozen_prompt) != seed_sha256
        or not isinstance(legacy_candidates, list)
        or not isinstance(legacy_hashes, list)
        or not isinstance(legacy_scores, list)
        or not (
            len(legacy_candidates) == len(legacy_hashes) == len(legacy_scores)
        )
    ):
        raise GEPACandidateSearchPlanError("gepa_candidate_search_legacy_lineage_mismatch")
    if (
        len(legacy_candidates) != 2
        or len(set(legacy_hashes)) != 2
        or component.get("best_idx") != 0
        or component.get("total_metric_calls") != 46
        or legacy_trace.get("max_metric_calls_per_prompt") != 40
    ):
        raise GEPACandidateSearchPlanError("gepa_candidate_search_legacy_diagnosis_drift")
    if any(
        not isinstance(candidate, Mapping)
        or hash_canonical(dict(candidate)) != legacy_hashes[index]
        for index, candidate in enumerate(legacy_candidates)
    ):
        raise GEPACandidateSearchPlanError(
            "gepa_candidate_search_legacy_candidate_hash_mismatch"
        )
    del legacy

    scaled = _repository_directory(root, config.prior_scaled_directory)
    scaled_plan_path = _repository_file(
        root, config.prior_scaled_directory / "optimization-plan.json"
    )
    scaled_trace_path = _repository_file(
        root, config.prior_scaled_directory / "gepa-result.json"
    )
    scaled_winner_path = _repository_file(
        root, config.prior_scaled_directory / "frozen-winner.json"
    )
    public_path = _repository_file(root, config.prior_scaled_public_summary_path)
    scaled_trace = _read_object(scaled_trace_path, "scaled_trace")
    scaled_winner = _read_object(scaled_winner_path, "scaled_winner")
    public = _read_object(public_path, "scaled_public_summary")
    official = scaled_trace.get("official_gepa_result")
    public_optimizer = public.get("optimizer")
    public_lineage = public.get("lineage")
    if (
        not isinstance(official, Mapping)
        or not isinstance(public_optimizer, Mapping)
        or not isinstance(public_lineage, Mapping)
        or scaled_winner.get("seed_prompt_sha256") != seed_sha256
        or scaled_winner.get("winner_prompt_sha256") == seed_sha256
        or scaled_winner.get("candidate_count") != 7
        or scaled_winner.get("reflection_proposals") != 8
        or scaled_winner.get("seed_retained") is not False
        or scaled_winner.get("test_payload_opened") is not False
        or scaled_winner.get("test_labels_scored") is not False
        or official.get("num_full_val_evals") != 7
        or public.get("seed_prompt_sha256") != seed_sha256
        or public.get("winner_prompt_sha256")
        != scaled_winner.get("winner_prompt_sha256")
        or public.get("seed_retained") is not False
        or public.get("observed_improvement_rule_satisfied") is not False
        or public.get("status") != "no_improvement_claim"
        or public_optimizer.get("accepted_candidate_count") != 7
        or scaled_winner.get("plan_file_sha256") != sha256_file(scaled_plan_path)
        or scaled_winner.get("trace_file_sha256") != sha256_file(scaled_trace_path)
        or public_lineage.get("plan_file_sha256") != sha256_file(scaled_plan_path)
        or public_lineage.get("winner_bundle_file_sha256")
        != sha256_file(scaled_winner_path)
    ):
        raise GEPACandidateSearchPlanError("gepa_candidate_search_scaled_lineage_mismatch")
    paired_declared = public_lineage.get("private_paired_report_file_sha256")
    if not isinstance(paired_declared, str) or SHA256_RE.fullmatch(paired_declared) is None:
        raise GEPACandidateSearchPlanError(
            "gepa_candidate_search_scaled_paired_hash_missing"
        )
    del scaled

    payload = {
        "obsolete_first_pass_trace_file_sha256": sha256_file(legacy_trace_path),
        "obsolete_first_pass_winner_file_sha256": sha256_file(legacy_winner_path),
        "obsolete_first_pass_seed_prompt_sha256": seed_sha256,
        "obsolete_first_pass_winner_prompt_sha256": seed_sha256,
        "obsolete_first_pass_candidate_count": 2,
        "obsolete_first_pass_distinct_mutation_count": 1,
        "obsolete_first_pass_metric_calls": 46,
        "obsolete_first_pass_declared_metric_cap": 40,
        "obsolete_first_pass_seed_dev_score": float(legacy_scores[0]),
        "obsolete_first_pass_mutation_dev_score": float(legacy_scores[1]),
        "obsolete_first_pass_seed_retained": True,
        "obsolete_first_pass_test_payload_opened_during_optimization": False,
        "authoritative_scaled_plan_file_sha256": sha256_file(scaled_plan_path),
        "authoritative_scaled_trace_file_sha256": sha256_file(scaled_trace_path),
        "authoritative_scaled_winner_file_sha256": sha256_file(scaled_winner_path),
        "authoritative_scaled_public_summary_file_sha256": sha256_file(public_path),
        "authoritative_scaled_seed_prompt_sha256": seed_sha256,
        "authoritative_scaled_winner_prompt_sha256": str(
            scaled_winner["winner_prompt_sha256"]
        ),
        "authoritative_scaled_candidate_count": 7,
        "authoritative_scaled_reflection_proposals": 8,
        "authoritative_scaled_seed_retained": False,
        "authoritative_scaled_observed_improvement_rule_satisfied": False,
        "authoritative_scaled_status": "no_improvement_claim",
        "authoritative_scaled_test_payload_opened_before_winner_freeze": False,
        "authoritative_scaled_paired_report_file_sha256_declared_not_recomputed": (
            paired_declared
        ),
        "authoritative_scaled_paired_report_opened_by_this_planner": False,
        "new_search_rationale": (
            "frontier_fable_high_transfer_with_separate_structured_grounding_objectives"
        ),
    }
    return PriorGEPADiagnosisV1.model_validate(
        {**payload, "diagnosis_sha256": hash_canonical(payload)}
    )


def _tier_plan(
    *,
    config: GEPACandidateSearchConfigV1,
    tier: SearchTierConfigV1,
    seed_text: str,
    train: Sequence[OptimizationExample],
    dev: Sequence[OptimizationExample],
    task_call_cost_micros: int,
    reflection_call_cost_micros: int,
) -> SearchTierPlanV1:
    candidates = _initial_candidates(
        seed_text=seed_text,
        templates=config.mutation_templates,
        count=tier.initial_mutation_count,
    )
    namespace = f"{config.subset_seed}:{tier.tier}"
    selected_train = _ranked_representatives(
        train,
        namespace=f"{namespace}:train",
        count=tier.train_representative_articles,
    )
    selected_dev_search = _ranked_representatives(
        dev,
        namespace=f"{namespace}:dev",
        count=tier.dev_search_representative_articles,
    )
    selected_search_papers = {item.paper_id for item in selected_dev_search}
    selected_dev_confirmation = _ranked_representatives(
        dev,
        namespace=f"{namespace}:dev",
        count=tier.dev_confirmation_representative_articles,
        excluded_papers=selected_search_papers,
    )
    train_membership = _membership(
        tier=tier.tier,
        phase="train_trajectories",
        official_split="train",
        examples=selected_train,
    )
    dev_search_membership = _membership(
        tier=tier.tier,
        phase="dev_search",
        official_split="dev",
        examples=selected_dev_search,
    )
    dev_confirmation_membership = _membership(
        tier=tier.tier,
        phase="dev_confirmation",
        official_split="dev",
        examples=selected_dev_confirmation,
    )
    budget = _budget(
        tier=tier,
        initial_candidate_count=len(candidates),
        task_call_cost_micros=task_call_cost_micros,
        reflection_call_cost_micros=reflection_call_cost_micros,
    )
    payload = {
        "tier": tier.tier,
        "pre_evaluation_candidates": candidates,
        "train_membership": train_membership,
        "dev_search_membership": dev_search_membership,
        "dev_confirmation_membership": dev_confirmation_membership,
        "call_budget": budget,
        "reflection_stage_occurs_only_after_initial_train_trajectories": True,
        "reflected_candidate_schema_requires_full_prompt_and_mutation_rationale": True,
        "reflected_candidates_must_pass_placeholder_and_normalized_hash_uniqueness": True,
        "reflected_candidate_gate_before_dev_search": (
            "exact_required_distinct_count_or_fail_closed_without_dev_evaluation"
        ),
        "all_dev_candidates_receive_identical_search_membership_and_one_attempt_each": True,
        "seed_and_selected_nonseed_receive_identical_confirmation_membership": True,
        "candidate_search_mechanics_only": True,
        "improvement_authority": False,
        "test_evaluation_authority": False,
    }
    return SearchTierPlanV1.model_validate(
        {**payload, "tier_sha256": hash_canonical(payload)}
    )


def freeze_evidence_inference_gepa_candidate_search_plan_v1(
    *,
    repository_root: Path,
    frozen_at: datetime,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> EvidenceInferenceGEPACandidateSearchPlanV1:
    """Freeze the zero-call, train/dev-only Fable search plan."""

    root = repository_root.resolve(strict=True)
    created = _aware(frozen_at, "frozen_at")
    config_file = _repository_file(root, config_path)
    config = load_gepa_candidate_search_config_v1(
        repository_root=root, config_path=config_path
    )
    manifest_file = _repository_file(root, config.manifest_path)
    seed_file = _repository_file(root, config.seed_prompt_path)
    source_file = _repository_file(root, SOURCE_PATH)
    manifest: OptimizationSplitManifest = load_split_manifest(manifest_file)

    # These are the only split payload access calls.  The test JSONL is not opened,
    # hashed, inspected, or passed to another function.
    train = load_manifest_split(manifest_file, "train")
    dev = load_manifest_split(manifest_file, "dev")
    official_train_dev_paper_overlap = len(
        set(manifest.train.paper_ids) & set(manifest.dev.paper_ids)
    )
    official_train_dev_group_overlap = len(
        set(manifest.train.group_ids) & set(manifest.dev.group_ids)
    )
    if official_train_dev_paper_overlap or official_train_dev_group_overlap:
        raise GEPACandidateSearchPlanError(
            "gepa_candidate_search_official_train_dev_overlap"
        )
    try:
        seed_text = seed_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise GEPACandidateSearchPlanError(
            "gepa_candidate_search_seed_prompt_invalid"
        ) from exc
    seed_sha = _sha256_text(seed_text)
    provider = config.provider
    provider_identity = {
        "provider": provider.provider,
        "model": provider.model,
        "effort": provider.effort,
        "service_tier": provider.service_tier,
        "transport": provider.transport,
        "input_rate_usd_per_million_tokens": str(
            provider.input_rate_usd_per_million_tokens
        ),
        "output_rate_usd_per_million_tokens": str(
            provider.output_rate_usd_per_million_tokens
        ),
        "task_input_token_ceiling": provider.task_input_token_ceiling,
        "task_output_token_ceiling": provider.task_output_token_ceiling,
        "reflection_input_token_ceiling": provider.reflection_input_token_ceiling,
        "reflection_output_token_ceiling": provider.reflection_output_token_ceiling,
        "sdk_retries_per_request": provider.sdk_retries_per_request,
        "application_retries_per_request": provider.application_retries_per_request,
        "fallback_requests_permitted": provider.fallback_requests_permitted,
        "pricing_source": "repository_frozen_provider_rate_table_verified_2026_08_29",
    }
    task_cost = _call_cost_micros(
        input_tokens=provider.task_input_token_ceiling,
        output_tokens=provider.task_output_token_ceiling,
        input_rate=provider.input_rate_usd_per_million_tokens,
        output_rate=provider.output_rate_usd_per_million_tokens,
    )
    reflection_cost = _call_cost_micros(
        input_tokens=provider.reflection_input_token_ceiling,
        output_tokens=provider.reflection_output_token_ceiling,
        input_rate=provider.input_rate_usd_per_million_tokens,
        output_rate=provider.output_rate_usd_per_million_tokens,
    )
    tiers = [
        _tier_plan(
            config=config,
            tier=tier,
            seed_text=seed_text,
            train=train,
            dev=dev,
            task_call_cost_micros=task_cost,
            reflection_call_cost_micros=reflection_cost,
        )
        for tier in config.tiers
    ]
    prior = _prior_diagnosis(root=root, config=config, seed_sha256=seed_sha)
    payload = {
        "plan_version": PLAN_VERSION,
        "frozen_at": _datetime_json(created),
        "status": "offline_multi_start_plan_frozen_zero_provider_calls",
        "config_file_sha256": sha256_file(config_file),
        "planner_source_sha256": sha256_file(source_file),
        "manifest_file_sha256": sha256_file(manifest_file),
        "manifest_source_examples_sha256": manifest.source_examples_sha256,
        "train_split_sha256": manifest.train.sha256,
        "dev_split_sha256": manifest.dev.sha256,
        "test_split_declared_sha256_not_recomputed": manifest.test.sha256,
        "seed_prompt_sha256": seed_sha,
        "provider": provider,
        "provider_identity_sha256": hash_canonical(provider_identity),
        "objectives": list(OBJECTIVE_SPECS),
        "objective_membership_sha256": hash_canonical(list(OBJECTIVE_SPECS)),
        "optimizer_contract": (
            "gepa_style_reflective_multi_start_search_requires_separate_runtime_adapter"
        ),
        "selection_rule": (
            "schema_and_grounding_noninferior_to_seed_then_maximize_extraction_correctness_"
            "then_minimize_cost_then_candidate_sha"
        ),
        "prior_diagnosis": prior,
        "tiers": tiers,
        "official_train_dev_article_overlap": 0,
        "official_train_dev_group_overlap": 0,
        "multiple_distinct_candidates_frozen_before_first_evaluation": True,
        "task_and_reflection_model_identical_fable_high": True,
        "provider_calls_made": 0,
        "reflection_calls_made": 0,
        "task_calls_made": 0,
        "credentials_read": False,
        "network_accessed": False,
        "test_payload_opened": False,
        "test_payload_hashed": False,
        "test_labels_opened": False,
        "test_labels_scored": False,
        "test_example_ids_materialized_in_plan": False,
        "winner_seed_equality_rule": (
            "if_winner_prompt_sha_equals_seed_report_seed_retained_negative_result_and_stop"
        ),
        "future_paired_evaluation_rule": (
            "only_after_nonseed_winner_freeze_use_same_examples_one_attempt_per_arm_equal_"
            "calls_and_balanced_order"
        ),
        "historical_labels_opened_elsewhere_pristine_claim": False,
        "improvement_authority": False,
        "reliability_authority": False,
        "generalization_authority": False,
        "scientific_effectiveness_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return EvidenceInferenceGEPACandidateSearchPlanV1.model_validate(
        {**payload, "plan_sha256": hash_canonical(payload)}
    )


def validate_evidence_inference_gepa_candidate_search_plan_v1(
    *,
    repository_root: Path,
    plan: EvidenceInferenceGEPACandidateSearchPlanV1 | Mapping[str, Any],
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> EvidenceInferenceGEPACandidateSearchPlanV1:
    """Rebuild from train/dev inputs and require exact plan equality."""

    raw = (
        plan.model_dump(mode="json")
        if isinstance(plan, EvidenceInferenceGEPACandidateSearchPlanV1)
        else plan
    )
    try:
        observed = EvidenceInferenceGEPACandidateSearchPlanV1.model_validate(raw)
    except ValueError as exc:
        raise GEPACandidateSearchPlanError(
            "gepa_candidate_search_plan_contract_or_hash_invalid"
        ) from exc
    expected = freeze_evidence_inference_gepa_candidate_search_plan_v1(
        repository_root=repository_root,
        config_path=config_path,
        frozen_at=observed.frozen_at,
    )
    if observed != expected:
        raise GEPACandidateSearchPlanError(
            "gepa_candidate_search_plan_external_replay_mismatch"
        )
    return observed


def freeze_gepa_candidate_search_development_decision_v1(
    *,
    plan: EvidenceInferenceGEPACandidateSearchPlanV1,
    tier: Literal["cheap_pilot", "scaled"],
    winner_prompt_sha256: str,
) -> GEPACandidateSearchDevelopmentDecisionV1:
    """Apply the fail-closed seed-retention rule without evaluating any example."""

    selected = next((item for item in plan.tiers if item.tier == tier), None)
    if selected is None:
        raise GEPACandidateSearchPlanError("gepa_candidate_search_tier_unknown")
    if SHA256_RE.fullmatch(winner_prompt_sha256) is None:
        raise GEPACandidateSearchPlanError("gepa_candidate_search_winner_sha_invalid")
    # A nonseed SHA may name a reflected candidate. The runtime must separately bind
    # that SHA to its frozen candidate ledger before using this decision artifact.
    winner_is_seed = winner_prompt_sha256 == plan.seed_prompt_sha256
    payload = {
        "decision_version": DECISION_VERSION,
        "plan_sha256": plan.plan_sha256,
        "tier_sha256": selected.tier_sha256,
        "seed_prompt_sha256": plan.seed_prompt_sha256,
        "winner_prompt_sha256": winner_prompt_sha256,
        "status": (
            "seed_retained_negative_result"
            if winner_is_seed
            else "nonseed_development_candidate_frozen_no_improvement_claim"
        ),
        "winner_is_seed": winner_is_seed,
        "equal_budget_future_evaluation_required": not winner_is_seed,
        "future_evaluation_performed": False,
        "improvement_authority": False,
        "generalization_authority": False,
    }
    return GEPACandidateSearchDevelopmentDecisionV1.model_validate(
        {**payload, "decision_sha256": hash_canonical(payload)}
    )


def write_evidence_inference_gepa_candidate_search_plan_v1(
    *,
    repository_root: Path,
    frozen_at: datetime,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_path: Path | None = None,
    force: bool = False,
) -> EvidenceInferenceGEPACandidateSearchPlanV1:
    """Freeze and atomically write the plan; no provider boundary is reachable."""

    root = repository_root.resolve(strict=True)
    config = load_gepa_candidate_search_config_v1(
        repository_root=root, config_path=config_path
    )
    relative_output = output_path or config.output_plan_path
    if relative_output.is_absolute() or ".." in relative_output.parts:
        raise GEPACandidateSearchPlanError("gepa_candidate_search_output_path_escape")
    target = root / relative_output
    plan = freeze_evidence_inference_gepa_candidate_search_plan_v1(
        repository_root=root,
        config_path=config_path,
        frozen_at=frozen_at,
    )
    atomic_write_json(target, plan, force=force)
    return plan


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "EvidenceInferenceGEPACandidateSearchPlanV1",
    "GEPACandidateSearchDevelopmentDecisionV1",
    "GEPACandidateSearchPlanError",
    "PreEvaluationCandidateV1",
    "RepresentativeMembershipV1",
    "SearchTierPlanV1",
    "TierCallBudgetV1",
    "freeze_evidence_inference_gepa_candidate_search_plan_v1",
    "freeze_gepa_candidate_search_development_decision_v1",
    "load_gepa_candidate_search_config_v1",
    "validate_evidence_inference_gepa_candidate_search_plan_v1",
    "write_evidence_inference_gepa_candidate_search_plan_v1",
]
