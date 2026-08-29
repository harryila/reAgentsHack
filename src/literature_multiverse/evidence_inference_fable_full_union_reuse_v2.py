"""Priority-union exact-wire overlay for the third full Fable execution.

The poisoned full-v2 workspace is immutable and has first priority.  The original
pilot and recovery pilot may contribute only exact-wire artifacts not already
represented by full-v2.  Ambiguous attempts are inherited as failed requests and
are never retried.  All unmatched requests are delegated to the ordinary paired
runtime, which retains the strict cumulative whole-pair budget gate.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import ConfigDict, Field, StrictInt, model_validator

from literature_multiverse.evidence_inference_fable_full_reuse_v1 import (
    REUSE_DIRECTORY as NESTED_REUSE_DIRECTORY,
)
from literature_multiverse.evidence_inference_fable_full_reuse_v1 import (
    EvidenceInferenceFableFullReuseTerminalV1,
    EvidenceInferenceFableReuseRecordV1,
    EvidenceInferenceFableReuseSourceV1,
    _artifact_map,
    _incident_map,
    _load_source,
    _read_object,
    validate_evidence_inference_fable_full_reuse_v1,
)
from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    INCIDENT_SANITIZATION_POLICY,
    EvidenceInferenceFableBudgetAuthorizationV1,
    EvidenceInferenceFableBudgetAuthorizationV2,
    EvidenceInferenceFableCallSurfaceV1,
    EvidenceInferenceFableClientProtocol,
    EvidenceInferenceFableIncidentArtifactV1,
    EvidenceInferenceFableIncidentV1,
    EvidenceInferenceFableIncidentV2,
    EvidenceInferenceFableIntentV1,
    EvidenceInferenceFablePreparedRuntimeV1,
    EvidenceInferenceFableProviderResultV1,
    EvidenceInferenceFableReceiptV1,
    EvidenceInferenceFableTerminalV1,
    execute_evidence_inference_fable_paired_v1,
    parse_evidence_inference_fable_budget_authorization_v1,
    validate_evidence_inference_fable_workspace_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_scoring_v1 import (
    PublicPairedSummaryV1,
)
from literature_multiverse.evidence_inference_fable_retrospective_v1 import (
    EvidenceInferenceFableRetrospectivePlanV1,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel

Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]
Count = Annotated[StrictInt, Field(ge=0)]
PositiveCount = Annotated[StrictInt, Field(ge=1)]
Micros = Annotated[StrictInt, Field(ge=0)]

UNION_DIRECTORY = "full-reuse-v2"
UNION_PLAN_FILE = "00-union-plan.json"
UNION_TERMINAL_FILE = "02-union-terminal.json"
EXPECTED_FULL_REQUESTS = 382
EXPECTED_RECEIPTS = 22
EXPECTED_AMBIGUITIES = 2
EXPECTED_NEW_CALLS = 358
EXPECTED_SHADOWED_CANDIDATES = 8
EXPECTED_TRANSITIVE_NESTED_RECORDS = 8
EXPECTED_INHERITED_FAILURE_REQUESTS_BY_ARM = {"seed": 0, "winner": 2}
EXPECTED_INHERITED_FAILURE_QUESTIONS_BY_ARM = {"seed": 0, "winner": 16}

UnionSourceSlot = Literal[
    "poisoned_full_v2",
    "poisoned_pilot_v1",
    "recovery_pilot_v2",
]
AdoptionKind = Literal["terminal_receipt", "inherited_ambiguous_failure"]


class EvidenceInferenceFableFullUnionReuseError(ValueError):
    """The priority-union overlay failed closed."""


class _Frozen(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False)


def _self_hash(model: _Frozen, field: str, code: str) -> None:
    if getattr(model, field) != hash_canonical(model.model_dump(mode="json", exclude={field})):
        raise ValueError(code)


def validate_evidence_inference_fable_full_union_paths_v2(
    *, workspace: Path, sources: list[EvidenceInferenceFableUnionSourceV2]
) -> None:
    target = workspace.resolve(strict=True)
    for source in sources:
        source_root = source.workspace.resolve(strict=True)
        if (
            target == source_root
            or target.is_relative_to(source_root)
            or source_root.is_relative_to(target)
        ):
            raise EvidenceInferenceFableFullUnionReuseError(
                "fable_union_target_source_path_overlap"
            )


@contextmanager
def _union_lock(workspace: Path) -> Any:
    descriptor = os.open(
        workspace / ".full-reuse-v2.lock",
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "a+b", closefd=True) as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
    finally:
        pass


@dataclass(frozen=True)
class EvidenceInferenceFableUnionSourceV2:
    """Caller-owned source paths; paths never enter serialized artifacts."""

    slot: UnionSourceSlot
    plan: EvidenceInferenceFableRetrospectivePlanV1
    workspace: Path
    nested_reuse_sources: tuple[EvidenceInferenceFableReuseSourceV1, ...] = ()


class EvidenceInferenceFableUnionSourceBindingV2(_Frozen):
    slot: UnionSourceSlot
    priority: Annotated[StrictInt, Field(ge=0, le=2)]
    plan_sha256: Sha256
    prepared_sha256: Sha256
    authorization_sha256: Sha256
    terminal_sha256: Sha256
    terminal_status: Literal["completed", "terminal_ambiguous_attempt_poison"]
    nested_reuse_terminal_sha256: Sha256 | None
    source_paths_serialized: Literal[False] = False
    source_workspace_mutation_permitted: Literal[False] = False


class EvidenceInferenceFableUnionEntryV2(_Frozen):
    entry_version: Literal["evidence-inference-fable-full-union-entry-v2"] = (
        "evidence-inference-fable-full-union-entry-v2"
    )
    adoption_kind: AdoptionKind
    target_execution_index: Count
    target_request_key: str
    target_surface_sha256: Sha256
    wire_call_sha256: Sha256
    locked_question_count: PositiveCount
    source_slot: UnionSourceSlot
    source_priority: Annotated[StrictInt, Field(ge=0, le=2)]
    source_plan_sha256: Sha256
    source_prepared_sha256: Sha256
    source_authorization_sha256: Sha256
    source_terminal_sha256: Sha256
    source_nested_reuse_terminal_sha256: Sha256 | None
    source_nested_reuse_record_sha256: Sha256 | None
    source_intent_sha256: Sha256
    source_request_key: str
    source_surface_sha256: Sha256
    source_receipt_sha256: Sha256 | None
    source_provider_result_sha256: Sha256 | None
    source_incident_sha256: Sha256 | None
    source_incident_kind: (
        Literal[
            "provider_call_raised_after_durable_intent",
            "provider_result_invalid_after_return",
        ]
        | None
    )
    source_charged_cost_usd_micros: PositiveCount
    source_retry_permitted: Literal[False] = False
    target_provider_attempts_permitted_for_entry: Literal[0] = 0
    entry_sha256: Sha256

    @model_validator(mode="after")
    def validate_entry(self) -> EvidenceInferenceFableUnionEntryV2:
        receipt_shape = (
            self.source_receipt_sha256 is not None
            and self.source_provider_result_sha256 is not None
            and self.source_incident_sha256 is None
            and self.source_incident_kind is None
        )
        incident_shape = (
            self.source_receipt_sha256 is None
            and self.source_provider_result_sha256 is None
            and self.source_incident_sha256 is not None
            and self.source_incident_kind is not None
        )
        if (self.adoption_kind == "terminal_receipt") != receipt_shape or (
            self.adoption_kind == "inherited_ambiguous_failure"
        ) != incident_shape:
            raise ValueError("fable_union_entry_shape_invalid")
        if (
            Path(self.target_request_key).name != self.target_request_key
            or Path(self.source_request_key).name != self.source_request_key
        ):
            raise ValueError("fable_union_request_key_unsafe")
        _self_hash(self, "entry_sha256", "fable_union_entry_hash_mismatch")
        return self


class EvidenceInferenceFableFullUnionPlanV2(_Frozen):
    plan_version: Literal["evidence-inference-fable-full-union-plan-v2"] = (
        "evidence-inference-fable-full-union-plan-v2"
    )
    full_plan_sha256: Sha256
    full_prepared_sha256: Sha256
    full_authorization_sha256: Sha256
    configured_total_budget_usd_micros: Literal[99_000_000] = 99_000_000
    full_request_count: Literal[382] = EXPECTED_FULL_REQUESTS
    source_priority: list[UnionSourceSlot]
    source_bindings: list[EvidenceInferenceFableUnionSourceBindingV2]
    entries: list[EvidenceInferenceFableUnionEntryV2]
    adopted_terminal_receipt_count: Literal[22] = EXPECTED_RECEIPTS
    inherited_ambiguous_failure_count: Literal[2] = EXPECTED_AMBIGUITIES
    maximum_new_provider_attempt_count: Literal[358] = EXPECTED_NEW_CALLS
    shadowed_lower_priority_candidate_count: Literal[8] = EXPECTED_SHADOWED_CANDIDATES
    transitively_reused_nested_record_count: Literal[8] = EXPECTED_TRANSITIVE_NESTED_RECORDS
    exact_wire_hash_and_deep_call_equality_required: Literal[True] = True
    source_workspaces_immutable: Literal[True] = True
    inherited_ambiguity_retry_permitted: Literal[False] = False
    labels_opened: Literal[False] = False
    provider_calls_made_while_planning: Literal[0] = 0
    scientific_claim_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    plan_sha256: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> EvidenceInferenceFableFullUnionPlanV2:
        expected_priority = [
            "poisoned_full_v2",
            "poisoned_pilot_v1",
            "recovery_pilot_v2",
        ]
        if (
            self.source_priority != expected_priority
            or [item.slot for item in self.source_bindings] != expected_priority
            or [item.priority for item in self.source_bindings] != [0, 1, 2]
            or [entry.target_execution_index for entry in self.entries]
            != sorted(entry.target_execution_index for entry in self.entries)
            or len({entry.target_request_key for entry in self.entries}) != len(self.entries)
            or len({entry.wire_call_sha256 for entry in self.entries}) != len(self.entries)
            or sum(entry.adoption_kind == "terminal_receipt" for entry in self.entries)
            != EXPECTED_RECEIPTS
            or sum(entry.adoption_kind == "inherited_ambiguous_failure" for entry in self.entries)
            != EXPECTED_AMBIGUITIES
            or sum(entry.source_nested_reuse_record_sha256 is not None for entry in self.entries)
            != EXPECTED_TRANSITIVE_NESTED_RECORDS
            or len(self.entries) + EXPECTED_NEW_CALLS != EXPECTED_FULL_REQUESTS
        ):
            raise ValueError("fable_union_plan_roster_invalid")
        _self_hash(self, "plan_sha256", "fable_union_plan_hash_mismatch")
        return self


class EvidenceInferenceFableUnionRecordV2(_Frozen):
    record_version: Literal["evidence-inference-fable-full-union-record-v2"] = (
        "evidence-inference-fable-full-union-record-v2"
    )
    union_plan_sha256: Sha256
    entry_sha256: Sha256
    adoption_kind: AdoptionKind
    source_slot: UnionSourceSlot
    source_terminal_sha256: Sha256
    source_nested_reuse_terminal_sha256: Sha256 | None
    source_nested_reuse_record_sha256: Sha256 | None
    source_intent_sha256: Sha256
    source_receipt_sha256: Sha256 | None
    source_provider_result_sha256: Sha256 | None
    source_incident_sha256: Sha256 | None
    target_authorization_sha256: Sha256
    target_request_key: str
    target_surface_sha256: Sha256
    wire_call_sha256: Sha256
    target_intent_sha256: Sha256
    target_provider_result_sha256: Sha256
    expected_target_receipt_sha256: Sha256
    expected_target_incident_sha256: Sha256 | None
    target_provider_attempt_count: Literal[0] = 0
    source_attempt_retry_permitted: Literal[False] = False
    locked_questions_scored_incorrect: Count
    charged_cost_usd_micros: PositiveCount
    record_sha256: Sha256

    @model_validator(mode="after")
    def validate_record(self) -> EvidenceInferenceFableUnionRecordV2:
        if (self.adoption_kind == "terminal_receipt") != (
            self.source_receipt_sha256 is not None
            and self.source_provider_result_sha256 is not None
            and self.source_incident_sha256 is None
            and self.expected_target_incident_sha256 is None
        ) or (self.adoption_kind == "inherited_ambiguous_failure") != (
            self.source_receipt_sha256 is None
            and self.source_provider_result_sha256 is None
            and self.source_incident_sha256 is not None
            and self.expected_target_incident_sha256 is not None
        ):
            raise ValueError("fable_union_record_shape_invalid")
        _self_hash(self, "record_sha256", "fable_union_record_hash_mismatch")
        return self


class EvidenceInferenceFableFullUnionTerminalV2(_Frozen):
    terminal_version: Literal["evidence-inference-fable-full-union-terminal-v2"] = (
        "evidence-inference-fable-full-union-terminal-v2"
    )
    union_plan_sha256: Sha256
    target_runtime_terminal_sha256: Sha256
    target_runtime_status: Literal[
        "completed",
        "clean_budget_exhaustion_before_next_pair",
        "terminal_ambiguous_attempt_poison",
    ]
    target_completed_request_count: Count
    realized_adopted_terminal_receipt_count: Count
    realized_inherited_ambiguous_failure_count: Count
    new_provider_attempt_count: Count
    maximum_new_provider_attempt_count: Literal[358] = EXPECTED_NEW_CALLS
    target_accounted_spend_usd_micros: Micros
    adopted_target_accounted_spend_usd_micros: Micros
    new_provider_accounted_spend_usd_micros: Micros
    source_terminal_artifact_lineage_count: Count
    inherited_ambiguous_attempts_retried: Literal[0] = 0
    target_provider_attempts_for_adopted_entries: Literal[0] = 0
    full_population_score_permitted: bool
    scoring_requires_this_union_terminal: Literal[True] = True
    scientific_claim_authority: Literal[False] = False
    confirmatory_gepa_improvement_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    terminal_sha256: Sha256

    @model_validator(mode="after")
    def validate_terminal(self) -> EvidenceInferenceFableFullUnionTerminalV2:
        complete = (
            self.target_runtime_status == "completed"
            and self.realized_adopted_terminal_receipt_count == EXPECTED_RECEIPTS
            and self.realized_inherited_ambiguous_failure_count == EXPECTED_AMBIGUITIES
            and self.new_provider_attempt_count == EXPECTED_NEW_CALLS
        )
        if (
            self.new_provider_attempt_count > EXPECTED_NEW_CALLS
            or self.source_terminal_artifact_lineage_count
            != self.realized_adopted_terminal_receipt_count
            + self.realized_inherited_ambiguous_failure_count
            or self.target_accounted_spend_usd_micros
            != self.adopted_target_accounted_spend_usd_micros
            + self.new_provider_accounted_spend_usd_micros
            or self.full_population_score_permitted != complete
        ):
            raise ValueError("fable_union_terminal_counts_invalid")
        _self_hash(self, "terminal_sha256", "fable_union_terminal_hash_mismatch")
        return self


class EvidenceInferenceFableFullUnionScoringLineageV2(_Frozen):
    """Post-score sidecar binding the unchanged scoring artifacts to union replay."""

    lineage_version: Literal["evidence-inference-fable-full-union-scoring-lineage-v2"] = (
        "evidence-inference-fable-full-union-scoring-lineage-v2"
    )
    status: Literal["complete_union_scoring_lineage"] = "complete_union_scoring_lineage"
    union_plan_sha256: Sha256
    union_terminal_sha256: Sha256
    target_runtime_terminal_sha256: Sha256
    target_runtime_status: Literal["completed"] = "completed"
    completion_certificate_sha256: Sha256
    private_report_sha256: Sha256
    public_summary_sha256: Sha256
    adopted_terminal_receipt_count: Literal[22] = EXPECTED_RECEIPTS
    inherited_ambiguous_failure_count: Literal[2] = EXPECTED_AMBIGUITIES
    new_provider_attempt_count: Literal[358] = EXPECTED_NEW_CALLS
    target_provider_attempts_for_adopted_entries: Literal[0] = 0
    inherited_ambiguous_attempts_retried: Literal[0] = 0
    contains_article_or_example_identifiers: Literal[False] = False
    contains_reference_or_per_example_labels: Literal[False] = False
    contains_raw_or_per_example_predictions: Literal[False] = False
    scientific_claim_authority: Literal[False] = False
    confirmatory_gepa_improvement_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    lineage_sha256: Sha256

    @model_validator(mode="after")
    def validate_lineage(
        self,
    ) -> EvidenceInferenceFableFullUnionScoringLineageV2:
        _self_hash(self, "lineage_sha256", "fable_union_scoring_lineage_hash_mismatch")
        return self


class EvidenceInferenceFableArmFailureBurdenV2(_Frozen):
    request_count: Count
    locked_question_count: Count


class EvidenceInferenceFableFullUnionFailureBurdenV2(_Frozen):
    """Identifier-free aggregate of target receipts forced to zero."""

    burden_version: Literal["evidence-inference-fable-full-union-failure-burden-v2"] = (
        "evidence-inference-fable-full-union-failure-burden-v2"
    )
    full_plan_sha256: Sha256
    union_plan_sha256: Sha256
    union_terminal_sha256: Sha256
    target_runtime_terminal_sha256: Sha256
    target_receipt_count: Literal[382] = EXPECTED_FULL_REQUESTS
    target_receipt_membership_sha256: Sha256
    target_incident_count: Count
    target_incident_membership_sha256: Sha256
    inherited_ambiguity_by_arm: dict[
        Literal["seed", "winner"], EvidenceInferenceFableArmFailureBurdenV2
    ]
    new_runtime_incident_by_arm: dict[
        Literal["seed", "winner"], EvidenceInferenceFableArmFailureBurdenV2
    ]
    all_target_incident_by_arm: dict[
        Literal["seed", "winner"], EvidenceInferenceFableArmFailureBurdenV2
    ]
    failed_receipt_without_target_incident_by_arm: dict[
        Literal["seed", "winner"], EvidenceInferenceFableArmFailureBurdenV2
    ]
    all_forced_zero_by_arm: dict[
        Literal["seed", "winner"], EvidenceInferenceFableArmFailureBurdenV2
    ]
    inherited_ambiguity_request_count: Literal[2] = EXPECTED_AMBIGUITIES
    inherited_ambiguity_locked_question_count: Literal[16] = 16
    new_runtime_incident_request_count: Count
    new_runtime_incident_locked_question_count: Count
    all_target_incident_request_count: Count
    all_target_incident_locked_question_count: Count
    failed_receipt_without_target_incident_request_count: Count
    failed_receipt_without_target_incident_locked_question_count: Count
    all_forced_zero_request_count: Count
    all_forced_zero_locked_question_count: Count
    failed_receipt_scoring_policy: Literal["all_locked_questions_incorrect_no_retry"] = (
        "all_locked_questions_incorrect_no_retry"
    )
    contains_request_or_example_identifiers: Literal[False] = False
    contains_provider_messages_or_source_paths: Literal[False] = False
    burden_sha256: Sha256

    @model_validator(mode="after")
    def validate_burden(self) -> EvidenceInferenceFableFullUnionFailureBurdenV2:
        expected_arms = {"seed", "winner"}

        def totals(
            values: Mapping[str, EvidenceInferenceFableArmFailureBurdenV2],
        ) -> tuple[int, int]:
            return (
                sum(item.request_count for item in values.values()),
                sum(item.locked_question_count for item in values.values()),
            )

        partitions = (
            self.inherited_ambiguity_by_arm,
            self.new_runtime_incident_by_arm,
            self.all_target_incident_by_arm,
            self.failed_receipt_without_target_incident_by_arm,
            self.all_forced_zero_by_arm,
        )
        if any(set(values) != expected_arms for values in partitions):
            raise ValueError("fable_union_failure_burden_arm_membership_invalid")
        inherited = totals(self.inherited_ambiguity_by_arm)
        new_incidents = totals(self.new_runtime_incident_by_arm)
        all_incidents = totals(self.all_target_incident_by_arm)
        nonincident = totals(self.failed_receipt_without_target_incident_by_arm)
        forced = totals(self.all_forced_zero_by_arm)
        if (
            self.inherited_ambiguity_by_arm["seed"].request_count != 0
            or self.inherited_ambiguity_by_arm["seed"].locked_question_count != 0
            or self.inherited_ambiguity_by_arm["winner"].request_count != 2
            or self.inherited_ambiguity_by_arm["winner"].locked_question_count != 16
            or inherited
            != (
                self.inherited_ambiguity_request_count,
                self.inherited_ambiguity_locked_question_count,
            )
            or new_incidents
            != (
                self.new_runtime_incident_request_count,
                self.new_runtime_incident_locked_question_count,
            )
            or all_incidents
            != (
                self.all_target_incident_request_count,
                self.all_target_incident_locked_question_count,
            )
            or nonincident
            != (
                self.failed_receipt_without_target_incident_request_count,
                self.failed_receipt_without_target_incident_locked_question_count,
            )
            or forced
            != (
                self.all_forced_zero_request_count,
                self.all_forced_zero_locked_question_count,
            )
            or self.target_incident_count != self.all_target_incident_request_count
        ):
            raise ValueError("fable_union_failure_burden_alias_mismatch")
        for arm in expected_arms:
            inherited_arm = self.inherited_ambiguity_by_arm[arm]
            new_arm = self.new_runtime_incident_by_arm[arm]
            incident_arm = self.all_target_incident_by_arm[arm]
            nonincident_arm = self.failed_receipt_without_target_incident_by_arm[arm]
            forced_arm = self.all_forced_zero_by_arm[arm]
            if (
                incident_arm.request_count != inherited_arm.request_count + new_arm.request_count
                or incident_arm.locked_question_count
                != inherited_arm.locked_question_count + new_arm.locked_question_count
                or forced_arm.request_count
                != incident_arm.request_count + nonincident_arm.request_count
                or forced_arm.locked_question_count
                != incident_arm.locked_question_count + nonincident_arm.locked_question_count
            ):
                raise ValueError("fable_union_failure_burden_partition_mismatch")
        _self_hash(
            self,
            "burden_sha256",
            "fable_union_failure_burden_hash_mismatch",
        )
        return self


class EvidenceInferenceFableFullUnionPublicEvaluationV2(_Frozen):
    """Aggregate-only public projection of completed union scoring provenance."""

    evaluation_version: Literal["evidence-inference-fable-full-union-public-evaluation-v2"] = (
        "evidence-inference-fable-full-union-public-evaluation-v2"
    )
    status: Literal["aggregate_only_completed_union_evaluation"] = (
        "aggregate_only_completed_union_evaluation"
    )
    full_plan_sha256: Sha256
    union_plan_sha256: Sha256
    union_terminal_sha256: Sha256
    target_runtime_terminal_sha256: Sha256
    public_summary_sha256: Sha256
    union_scoring_lineage_sha256: Sha256
    failure_burden_sha256: Sha256
    completion_certificate_sha256: Sha256
    private_report_sha256: Sha256
    population: Literal["full_test"] = "full_test"
    examples: Literal[524] = 524
    articles: Literal[191] = 191
    requests: Literal[382] = EXPECTED_FULL_REQUESTS
    adopted_terminal_receipt_count: Literal[22] = EXPECTED_RECEIPTS
    inherited_ambiguous_failure_count: Literal[2] = EXPECTED_AMBIGUITIES
    new_provider_attempt_count: Literal[358] = EXPECTED_NEW_CALLS
    inherited_failure_request_count_by_arm: dict[Literal["seed", "winner"], Count]
    inherited_failure_locked_question_count_by_arm: dict[Literal["seed", "winner"], Count]
    inherited_failure_locked_question_count: Literal[16] = 16
    inherited_failure_scoring_policy: Literal["all_locked_questions_incorrect_no_retry"] = (
        "all_locked_questions_incorrect_no_retry"
    )
    realized_failure_burden: EvidenceInferenceFableFullUnionFailureBurdenV2
    target_incident_count: Count
    target_incident_locked_question_count: Count
    target_incident_request_count_by_arm: dict[Literal["seed", "winner"], Count]
    target_incident_locked_question_count_by_arm: dict[Literal["seed", "winner"], Count]
    new_runtime_incident_request_count: Count
    new_runtime_incident_locked_question_count: Count
    new_runtime_incident_request_count_by_arm: dict[Literal["seed", "winner"], Count]
    new_runtime_incident_locked_question_count_by_arm: dict[Literal["seed", "winner"], Count]
    all_forced_zero_request_count: Count
    all_forced_zero_locked_question_count: Count
    all_forced_zero_request_count_by_arm: dict[Literal["seed", "winner"], Count]
    all_forced_zero_locked_question_count_by_arm: dict[Literal["seed", "winner"], Count]
    target_accounted_spend_usd_micros: Micros
    adopted_target_accounted_spend_usd_micros: Micros
    new_provider_accounted_spend_usd_micros: Micros
    accounted_spend_is_not_provider_invoice: Literal[True] = True
    union_of_multiple_exact_wire_runs: Literal[True] = True
    public_sidecar_written_after_completed_score: Literal[True] = True
    contains_article_or_example_identifiers: Literal[False] = False
    contains_reference_or_per_example_labels: Literal[False] = False
    contains_raw_or_per_example_predictions: Literal[False] = False
    contains_article_or_question_text: Literal[False] = False
    contains_evidence_quotes_or_line_references: Literal[False] = False
    contains_source_paths: Literal[False] = False
    exploratory_retrospective_benchmark_reporting_permitted: Literal[True] = True
    confirmatory_gepa_improvement_authority: Literal[False] = False
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
            "union_of_multiple_exact_wire_runs",
            "inherited_ambiguities_are_winner_only_intention_to_evaluate_failures",
            "all_realized_failed_receipts_are_intention_to_evaluate_zeroes",
        ]
    ]
    evaluation_sha256: Sha256

    @model_validator(mode="after")
    def validate_evaluation(
        self,
    ) -> EvidenceInferenceFableFullUnionPublicEvaluationV2:
        expected_caveats = [
            "historically_opened_test_not_pristine_or_confirmatory",
            "cross_model_and_article_batched_interface_transfer_only",
            "formal_exact_grounding_is_not_semantic_entailment",
            "all_retained_examples_are_eligibility_positive",
            "union_of_multiple_exact_wire_runs",
            "inherited_ambiguities_are_winner_only_intention_to_evaluate_failures",
            "all_realized_failed_receipts_are_intention_to_evaluate_zeroes",
        ]
        if (
            self.inherited_failure_request_count_by_arm
            != EXPECTED_INHERITED_FAILURE_REQUESTS_BY_ARM
            or self.inherited_failure_locked_question_count_by_arm
            != EXPECTED_INHERITED_FAILURE_QUESTIONS_BY_ARM
            or self.inherited_failure_locked_question_count
            != sum(self.inherited_failure_locked_question_count_by_arm.values())
            or self.failure_burden_sha256 != self.realized_failure_burden.burden_sha256
            or self.target_incident_count != self.realized_failure_burden.target_incident_count
            or self.target_incident_locked_question_count
            != self.realized_failure_burden.all_target_incident_locked_question_count
            or self.new_runtime_incident_request_count
            != self.realized_failure_burden.new_runtime_incident_request_count
            or self.new_runtime_incident_locked_question_count
            != self.realized_failure_burden.new_runtime_incident_locked_question_count
            or self.all_forced_zero_request_count
            != self.realized_failure_burden.all_forced_zero_request_count
            or self.all_forced_zero_locked_question_count
            != self.realized_failure_burden.all_forced_zero_locked_question_count
            or self.target_accounted_spend_usd_micros
            != self.adopted_target_accounted_spend_usd_micros
            + self.new_provider_accounted_spend_usd_micros
            or self.required_caveats != expected_caveats
        ):
            raise ValueError("fable_union_public_evaluation_alias_mismatch")
        expected_aliases = {
            "target_incident_request_count_by_arm": {
                arm: value.request_count
                for arm, value in self.realized_failure_burden.all_target_incident_by_arm.items()
            },
            "target_incident_locked_question_count_by_arm": {
                arm: value.locked_question_count
                for arm, value in self.realized_failure_burden.all_target_incident_by_arm.items()
            },
            "new_runtime_incident_request_count_by_arm": {
                arm: value.request_count
                for arm, value in self.realized_failure_burden.new_runtime_incident_by_arm.items()
            },
            "new_runtime_incident_locked_question_count_by_arm": {
                arm: value.locked_question_count
                for arm, value in self.realized_failure_burden.new_runtime_incident_by_arm.items()
            },
            "all_forced_zero_request_count_by_arm": {
                arm: value.request_count
                for arm, value in self.realized_failure_burden.all_forced_zero_by_arm.items()
            },
            "all_forced_zero_locked_question_count_by_arm": {
                arm: value.locked_question_count
                for arm, value in self.realized_failure_burden.all_forced_zero_by_arm.items()
            },
        }
        if any(getattr(self, field) != value for field, value in expected_aliases.items()):
            raise ValueError("fable_union_public_evaluation_burden_alias_mismatch")
        _self_hash(
            self,
            "evaluation_sha256",
            "fable_union_public_evaluation_hash_mismatch",
        )
        return self


def freeze_evidence_inference_fable_full_union_scoring_lineage_v2(
    *,
    union_terminal: EvidenceInferenceFableFullUnionTerminalV2,
    completion_certificate_sha256: str,
    private_report_sha256: str,
    public_summary_sha256: str,
) -> EvidenceInferenceFableFullUnionScoringLineageV2:
    if (
        union_terminal.target_runtime_status != "completed"
        or not union_terminal.full_population_score_permitted
        or union_terminal.realized_adopted_terminal_receipt_count != EXPECTED_RECEIPTS
        or union_terminal.realized_inherited_ambiguous_failure_count != EXPECTED_AMBIGUITIES
        or union_terminal.new_provider_attempt_count != EXPECTED_NEW_CALLS
    ):
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_complete_terminal_required_for_scoring_lineage"
        )
    payload = {
        "lineage_version": ("evidence-inference-fable-full-union-scoring-lineage-v2"),
        "status": "complete_union_scoring_lineage",
        "union_plan_sha256": union_terminal.union_plan_sha256,
        "union_terminal_sha256": union_terminal.terminal_sha256,
        "target_runtime_terminal_sha256": (union_terminal.target_runtime_terminal_sha256),
        "target_runtime_status": "completed",
        "completion_certificate_sha256": completion_certificate_sha256,
        "private_report_sha256": private_report_sha256,
        "public_summary_sha256": public_summary_sha256,
        "adopted_terminal_receipt_count": EXPECTED_RECEIPTS,
        "inherited_ambiguous_failure_count": EXPECTED_AMBIGUITIES,
        "new_provider_attempt_count": EXPECTED_NEW_CALLS,
        "target_provider_attempts_for_adopted_entries": 0,
        "inherited_ambiguous_attempts_retried": 0,
        "contains_article_or_example_identifiers": False,
        "contains_reference_or_per_example_labels": False,
        "contains_raw_or_per_example_predictions": False,
        "scientific_claim_authority": False,
        "confirmatory_gepa_improvement_authority": False,
        "claim_release_authority": False,
    }
    return EvidenceInferenceFableFullUnionScoringLineageV2.model_validate(
        {**payload, "lineage_sha256": hash_canonical(payload)}
    )


def freeze_evidence_inference_fable_full_union_failure_burden_v2(
    *,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    union_plan: EvidenceInferenceFableFullUnionPlanV2,
    union_terminal: EvidenceInferenceFableFullUnionTerminalV2,
    target_terminal: EvidenceInferenceFableTerminalV1,
    receipts: Mapping[str, EvidenceInferenceFableReceiptV1 | Mapping[str, Any]],
    incidents: Mapping[str, EvidenceInferenceFableIncidentArtifactV1],
) -> EvidenceInferenceFableFullUnionFailureBurdenV2:
    """Aggregate validated target failures without serializing request identities."""

    canonical_full = EvidenceInferenceFableRetrospectivePlanV1.model_validate(
        full_plan.model_dump(mode="json")
    )
    canonical_union = EvidenceInferenceFableFullUnionPlanV2.model_validate(
        union_plan.model_dump(mode="json")
    )
    terminal = EvidenceInferenceFableFullUnionTerminalV2.model_validate(
        union_terminal.model_dump(mode="json")
    )
    runtime = EvidenceInferenceFableTerminalV1.model_validate(
        target_terminal.model_dump(mode="json")
    )
    planned = {request.request_key: request for request in canonical_full.roster}
    canonical_receipts = {
        key: (
            value
            if isinstance(value, EvidenceInferenceFableReceiptV1)
            else EvidenceInferenceFableReceiptV1.model_validate(value)
        )
        for key, value in receipts.items()
    }
    canonical_incidents = dict(incidents)
    if (
        canonical_full.mode != "full_paired"
        or canonical_union.full_plan_sha256 != canonical_full.plan_sha256
        or terminal.union_plan_sha256 != canonical_union.plan_sha256
        or terminal.target_runtime_terminal_sha256 != runtime.terminal_sha256
        or terminal.target_runtime_status != "completed"
        or runtime.status != "completed"
        or not terminal.full_population_score_permitted
        or not runtime.full_population_score_permitted
        or runtime.completed_request_count != EXPECTED_FULL_REQUESTS
        or set(canonical_receipts) != set(planned)
        or len(canonical_receipts) != EXPECTED_FULL_REQUESTS
        or not set(canonical_incidents).issubset(planned)
    ):
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_failure_burden_source_binding_invalid"
        )
    for key, receipt in canonical_receipts.items():
        request = planned[key]
        if (
            receipt.request_key != key
            or receipt.locked_question_count != request.question_count
            or receipt.locked_questions_scored_incorrect
            != (request.question_count if receipt.provider_result.outcome == "failed" else 0)
        ):
            raise EvidenceInferenceFableFullUnionReuseError(
                "fable_union_failure_burden_receipt_binding_invalid"
            )
    for key, incident in canonical_incidents.items():
        receipt = canonical_receipts[key]
        if (
            not isinstance(incident, EvidenceInferenceFableIncidentV2)
            or incident.request_key != key
            or incident.intent_sha256 != receipt.intent_sha256
            or incident.derived_provider_result_sha256 != receipt.provider_result.result_sha256
            or receipt.provider_result.outcome != "failed"
        ):
            raise EvidenceInferenceFableFullUnionReuseError(
                "fable_union_failure_burden_incident_binding_invalid"
            )

    inherited_keys = {
        entry.target_request_key
        for entry in canonical_union.entries
        if entry.adoption_kind == "inherited_ambiguous_failure"
    }
    union_entry_keys = {entry.target_request_key for entry in canonical_union.entries}
    incident_keys = set(canonical_incidents)
    failed_keys = {
        key
        for key, receipt in canonical_receipts.items()
        if receipt.provider_result.outcome == "failed"
    }
    new_incident_keys = incident_keys - inherited_keys
    if (
        len(inherited_keys) != EXPECTED_AMBIGUITIES
        or not inherited_keys.issubset(incident_keys)
        or not incident_keys.issubset(failed_keys)
        or new_incident_keys & union_entry_keys
    ):
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_failure_burden_partition_invalid"
        )
    nonincident_failed_keys = failed_keys - incident_keys

    def aggregate(keys: set[str]) -> dict[str, EvidenceInferenceFableArmFailureBurdenV2]:
        result: dict[str, EvidenceInferenceFableArmFailureBurdenV2] = {}
        for arm in ("seed", "winner"):
            arm_requests = [planned[key] for key in keys if planned[key].arm == arm]
            result[arm] = EvidenceInferenceFableArmFailureBurdenV2(
                request_count=len(arm_requests),
                locked_question_count=sum(item.question_count for item in arm_requests),
            )
        return result

    inherited = aggregate(inherited_keys)
    new_incidents = aggregate(new_incident_keys)
    all_incidents = aggregate(incident_keys)
    nonincident = aggregate(nonincident_failed_keys)
    forced = aggregate(failed_keys)

    def totals(
        values: Mapping[str, EvidenceInferenceFableArmFailureBurdenV2],
    ) -> tuple[int, int]:
        return (
            sum(item.request_count for item in values.values()),
            sum(item.locked_question_count for item in values.values()),
        )

    new_incident_totals = totals(new_incidents)
    all_incident_totals = totals(all_incidents)
    nonincident_totals = totals(nonincident)
    forced_totals = totals(forced)
    payload = {
        "burden_version": "evidence-inference-fable-full-union-failure-burden-v2",
        "full_plan_sha256": canonical_full.plan_sha256,
        "union_plan_sha256": canonical_union.plan_sha256,
        "union_terminal_sha256": terminal.terminal_sha256,
        "target_runtime_terminal_sha256": runtime.terminal_sha256,
        "target_receipt_count": EXPECTED_FULL_REQUESTS,
        "target_receipt_membership_sha256": hash_canonical(
            sorted(receipt.receipt_sha256 for receipt in canonical_receipts.values())
        ),
        "target_incident_count": len(canonical_incidents),
        "target_incident_membership_sha256": hash_canonical(
            sorted(incident.incident_sha256 for incident in canonical_incidents.values())
        ),
        "inherited_ambiguity_by_arm": inherited,
        "new_runtime_incident_by_arm": new_incidents,
        "all_target_incident_by_arm": all_incidents,
        "failed_receipt_without_target_incident_by_arm": nonincident,
        "all_forced_zero_by_arm": forced,
        "inherited_ambiguity_request_count": EXPECTED_AMBIGUITIES,
        "inherited_ambiguity_locked_question_count": 16,
        "new_runtime_incident_request_count": new_incident_totals[0],
        "new_runtime_incident_locked_question_count": new_incident_totals[1],
        "all_target_incident_request_count": all_incident_totals[0],
        "all_target_incident_locked_question_count": all_incident_totals[1],
        "failed_receipt_without_target_incident_request_count": nonincident_totals[0],
        "failed_receipt_without_target_incident_locked_question_count": (nonincident_totals[1]),
        "all_forced_zero_request_count": forced_totals[0],
        "all_forced_zero_locked_question_count": forced_totals[1],
        "failed_receipt_scoring_policy": "all_locked_questions_incorrect_no_retry",
        "contains_request_or_example_identifiers": False,
        "contains_provider_messages_or_source_paths": False,
    }
    return EvidenceInferenceFableFullUnionFailureBurdenV2.model_validate(
        {**payload, "burden_sha256": hash_canonical(payload)}
    )


def derive_evidence_inference_fable_full_union_failure_burden_v2(
    *,
    workspace: Path,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    union_plan: EvidenceInferenceFableFullUnionPlanV2,
    union_terminal: EvidenceInferenceFableFullUnionTerminalV2,
) -> EvidenceInferenceFableFullUnionFailureBurdenV2:
    """Externally replay a completed target, then project its failure aggregates."""

    terminal_before = validate_evidence_inference_fable_workspace_v1(
        workspace=workspace, plan=full_plan
    )
    receipts = _artifact_map(workspace / "receipts", EvidenceInferenceFableReceiptV1)
    incidents = _incident_map(workspace / "incidents")
    result = freeze_evidence_inference_fable_full_union_failure_burden_v2(
        full_plan=full_plan,
        union_plan=union_plan,
        union_terminal=union_terminal,
        target_terminal=terminal_before,
        receipts=receipts,
        incidents=incidents,
    )
    terminal_after = validate_evidence_inference_fable_workspace_v1(
        workspace=workspace, plan=full_plan
    )
    if terminal_after != terminal_before:
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_failure_burden_runtime_changed_during_projection"
        )
    return result


def project_evidence_inference_fable_full_union_public_evaluation_v2(
    *,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    union_plan: EvidenceInferenceFableFullUnionPlanV2,
    union_terminal: EvidenceInferenceFableFullUnionTerminalV2,
    public_summary: PublicPairedSummaryV1,
    union_scoring_lineage: EvidenceInferenceFableFullUnionScoringLineageV2,
    failure_burden: EvidenceInferenceFableFullUnionFailureBurdenV2,
) -> EvidenceInferenceFableFullUnionPublicEvaluationV2:
    """Project only safe aggregate provenance after completed union scoring.

    The caller must first externally replay the union terminal.  Re-validating every
    supplied frozen model here makes the projection fail closed if a serialized input
    was altered between that replay and this post-score step.
    """

    canonical_full = EvidenceInferenceFableRetrospectivePlanV1.model_validate(
        full_plan.model_dump(mode="json")
    )
    canonical_union = EvidenceInferenceFableFullUnionPlanV2.model_validate(
        union_plan.model_dump(mode="json")
    )
    terminal = EvidenceInferenceFableFullUnionTerminalV2.model_validate(
        union_terminal.model_dump(mode="json")
    )
    public = PublicPairedSummaryV1.model_validate(public_summary.model_dump(mode="json"))
    lineage = EvidenceInferenceFableFullUnionScoringLineageV2.model_validate(
        union_scoring_lineage.model_dump(mode="json")
    )
    burden = EvidenceInferenceFableFullUnionFailureBurdenV2.model_validate(
        failure_burden.model_dump(mode="json")
    )
    if (
        canonical_full.mode != "full_paired"
        or canonical_full.population != "full_test"
        or canonical_full.unique_examples != 524
        or canonical_full.unique_articles != 191
        or canonical_full.request_count != EXPECTED_FULL_REQUESTS
        or canonical_union.full_plan_sha256 != canonical_full.plan_sha256
        or terminal.union_plan_sha256 != canonical_union.plan_sha256
        or terminal.target_runtime_status != "completed"
        or terminal.target_completed_request_count != EXPECTED_FULL_REQUESTS
        or not terminal.full_population_score_permitted
        or lineage.union_plan_sha256 != canonical_union.plan_sha256
        or lineage.union_terminal_sha256 != terminal.terminal_sha256
        or lineage.target_runtime_terminal_sha256 != terminal.target_runtime_terminal_sha256
        or public.status != "aggregate_only_exploratory_retrospective_paired_score"
        or public.population != "full_test"
        or public.plan_sha256 != canonical_full.plan_sha256
        or public.runtime_terminal_sha256 != terminal.target_runtime_terminal_sha256
        or public.public_summary_sha256 != lineage.public_summary_sha256
        or public.completion_certificate_sha256 != lineage.completion_certificate_sha256
        or public.private_report_sha256 != lineage.private_report_sha256
        or burden.full_plan_sha256 != canonical_full.plan_sha256
        or burden.union_plan_sha256 != canonical_union.plan_sha256
        or burden.union_terminal_sha256 != terminal.terminal_sha256
        or burden.target_runtime_terminal_sha256 != terminal.target_runtime_terminal_sha256
        or public.examples != 524
        or public.articles != 191
        or public.requests != EXPECTED_FULL_REQUESTS
    ):
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_public_evaluation_source_binding_invalid"
        )

    request_counts = {"seed": 0, "winner": 0}
    question_counts = {"seed": 0, "winner": 0}
    for entry in canonical_union.entries:
        if entry.adoption_kind != "inherited_ambiguous_failure":
            continue
        try:
            request = canonical_full.roster[entry.target_execution_index]
        except IndexError as exc:
            raise EvidenceInferenceFableFullUnionReuseError(
                "fable_union_public_evaluation_entry_index_invalid"
            ) from exc
        if (
            request.request_key != entry.target_request_key
            or request.question_count != entry.locked_question_count
        ):
            raise EvidenceInferenceFableFullUnionReuseError(
                "fable_union_public_evaluation_failure_binding_invalid"
            )
        request_counts[request.arm] += 1
        question_counts[request.arm] += request.question_count
    if (
        request_counts != EXPECTED_INHERITED_FAILURE_REQUESTS_BY_ARM
        or question_counts != EXPECTED_INHERITED_FAILURE_QUESTIONS_BY_ARM
    ):
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_public_evaluation_failure_aggregate_drift"
        )

    payload = {
        "evaluation_version": ("evidence-inference-fable-full-union-public-evaluation-v2"),
        "status": "aggregate_only_completed_union_evaluation",
        "full_plan_sha256": canonical_full.plan_sha256,
        "union_plan_sha256": canonical_union.plan_sha256,
        "union_terminal_sha256": terminal.terminal_sha256,
        "target_runtime_terminal_sha256": terminal.target_runtime_terminal_sha256,
        "public_summary_sha256": public.public_summary_sha256,
        "union_scoring_lineage_sha256": lineage.lineage_sha256,
        "failure_burden_sha256": burden.burden_sha256,
        "completion_certificate_sha256": lineage.completion_certificate_sha256,
        "private_report_sha256": lineage.private_report_sha256,
        "population": "full_test",
        "examples": 524,
        "articles": 191,
        "requests": EXPECTED_FULL_REQUESTS,
        "adopted_terminal_receipt_count": EXPECTED_RECEIPTS,
        "inherited_ambiguous_failure_count": EXPECTED_AMBIGUITIES,
        "new_provider_attempt_count": EXPECTED_NEW_CALLS,
        "inherited_failure_request_count_by_arm": request_counts,
        "inherited_failure_locked_question_count_by_arm": question_counts,
        "inherited_failure_locked_question_count": sum(question_counts.values()),
        "inherited_failure_scoring_policy": ("all_locked_questions_incorrect_no_retry"),
        "realized_failure_burden": burden,
        "target_incident_count": burden.target_incident_count,
        "target_incident_locked_question_count": (burden.all_target_incident_locked_question_count),
        "target_incident_request_count_by_arm": {
            arm: value.request_count for arm, value in burden.all_target_incident_by_arm.items()
        },
        "target_incident_locked_question_count_by_arm": {
            arm: value.locked_question_count
            for arm, value in burden.all_target_incident_by_arm.items()
        },
        "new_runtime_incident_request_count": burden.new_runtime_incident_request_count,
        "new_runtime_incident_locked_question_count": (
            burden.new_runtime_incident_locked_question_count
        ),
        "new_runtime_incident_request_count_by_arm": {
            arm: value.request_count for arm, value in burden.new_runtime_incident_by_arm.items()
        },
        "new_runtime_incident_locked_question_count_by_arm": {
            arm: value.locked_question_count
            for arm, value in burden.new_runtime_incident_by_arm.items()
        },
        "all_forced_zero_request_count": burden.all_forced_zero_request_count,
        "all_forced_zero_locked_question_count": (burden.all_forced_zero_locked_question_count),
        "all_forced_zero_request_count_by_arm": {
            arm: value.request_count for arm, value in burden.all_forced_zero_by_arm.items()
        },
        "all_forced_zero_locked_question_count_by_arm": {
            arm: value.locked_question_count for arm, value in burden.all_forced_zero_by_arm.items()
        },
        "target_accounted_spend_usd_micros": (terminal.target_accounted_spend_usd_micros),
        "adopted_target_accounted_spend_usd_micros": (
            terminal.adopted_target_accounted_spend_usd_micros
        ),
        "new_provider_accounted_spend_usd_micros": (
            terminal.new_provider_accounted_spend_usd_micros
        ),
        "accounted_spend_is_not_provider_invoice": True,
        "union_of_multiple_exact_wire_runs": True,
        "public_sidecar_written_after_completed_score": True,
        "contains_article_or_example_identifiers": False,
        "contains_reference_or_per_example_labels": False,
        "contains_raw_or_per_example_predictions": False,
        "contains_article_or_question_text": False,
        "contains_evidence_quotes_or_line_references": False,
        "contains_source_paths": False,
        "exploratory_retrospective_benchmark_reporting_permitted": True,
        "confirmatory_gepa_improvement_authority": False,
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
            "union_of_multiple_exact_wire_runs",
            "inherited_ambiguities_are_winner_only_intention_to_evaluate_failures",
            "all_realized_failed_receipts_are_intention_to_evaluate_zeroes",
        ],
    }
    return EvidenceInferenceFableFullUnionPublicEvaluationV2.model_validate(
        {**payload, "evaluation_sha256": hash_canonical(payload)}
    )


def materialize_evidence_inference_fable_full_union_public_evaluation_v2(
    *,
    evaluation: EvidenceInferenceFableFullUnionPublicEvaluationV2,
    output_path: Path,
) -> Path:
    """Write a fresh public completion sidecar after every dependency validates."""

    canonical = EvidenceInferenceFableFullUnionPublicEvaluationV2.model_validate(
        evaluation.model_dump(mode="json")
    )
    if (
        output_path.exists()
        or output_path.is_symlink()
        or output_path.parent.is_symlink()
        or not output_path.parent.is_dir()
    ):
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_public_evaluation_target_not_fresh_or_safe"
        )
    atomic_write_json(output_path, canonical)
    return output_path


@dataclass(frozen=True)
class _UnionState:
    source: EvidenceInferenceFableUnionSourceV2
    prepared: EvidenceInferenceFablePreparedRuntimeV1
    authorization: EvidenceInferenceFableBudgetAuthorizationV1
    terminal: EvidenceInferenceFableTerminalV1
    intents: Mapping[str, EvidenceInferenceFableIntentV1]
    receipts: Mapping[str, EvidenceInferenceFableReceiptV1]
    incidents: Mapping[str, EvidenceInferenceFableIncidentArtifactV1]
    nested_terminal: EvidenceInferenceFableFullReuseTerminalV1 | None
    nested_records: Mapping[str, EvidenceInferenceFableReuseRecordV1]


def _load_union_source(source: EvidenceInferenceFableUnionSourceV2) -> _UnionState:
    # The V1 source dataclass has no runtime validator; casting lets the mature
    # standard replay loader validate the workspace without serializing its path.
    legacy_spec = EvidenceInferenceFableReuseSourceV1(
        cast(Any, source.slot), source.plan, source.workspace
    )
    state = _load_source(legacy_spec)
    nested_terminal = None
    nested_records: Mapping[str, EvidenceInferenceFableReuseRecordV1] = {}
    if source.slot == "poisoned_full_v2":
        if len(source.nested_reuse_sources) != 2:
            raise EvidenceInferenceFableFullUnionReuseError(
                "fable_union_full_v2_nested_sources_missing"
            )
        nested_terminal = validate_evidence_inference_fable_full_reuse_v1(
            workspace=source.workspace,
            full_plan=source.plan,
            sources=list(source.nested_reuse_sources),
        )
        nested_records = _artifact_map(
            source.workspace / NESTED_REUSE_DIRECTORY / "records",
            EvidenceInferenceFableReuseRecordV1,
        )
    elif source.nested_reuse_sources:
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_nested_sources_only_allowed_for_full_v2"
        )
    return _UnionState(
        source=source,
        prepared=state.prepared,
        authorization=state.authorization,
        terminal=state.terminal,
        intents=state.intents,
        receipts=state.receipts,
        incidents=state.incidents,
        nested_terminal=nested_terminal,
        nested_records=nested_records,
    )


def _wire_identity(surface: EvidenceInferenceFableCallSurfaceV1) -> tuple[Any, ...]:
    return (
        surface.model,
        surface.effort,
        surface.service_tier,
        surface.max_output_tokens,
        surface.system,
        surface.prompt,
        surface.wire_schema,
    )


def _entry(
    *,
    target_index: int,
    target_surface: EvidenceInferenceFableCallSurfaceV1,
    priority: int,
    state: _UnionState,
    intent: EvidenceInferenceFableIntentV1,
    receipt: EvidenceInferenceFableReceiptV1 | None,
    incident: EvidenceInferenceFableIncidentV1 | None,
) -> EvidenceInferenceFableUnionEntryV2:
    if (
        (receipt is None) == (incident is None)
        or target_surface.wire_call_sha256 != intent.surface.wire_call_sha256
        or _wire_identity(target_surface) != _wire_identity(intent.surface)
        or target_surface.locked_question_count != intent.surface.locked_question_count
    ):
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_exact_wire_deep_identity_mismatch"
        )
    nested_record = state.nested_records.get(intent.request_key)
    result = None if receipt is None else receipt.provider_result
    payload = {
        "entry_version": "evidence-inference-fable-full-union-entry-v2",
        "adoption_kind": (
            "terminal_receipt" if receipt is not None else "inherited_ambiguous_failure"
        ),
        "target_execution_index": target_index,
        "target_request_key": target_surface.request_key,
        "target_surface_sha256": target_surface.surface_sha256,
        "wire_call_sha256": target_surface.wire_call_sha256,
        "locked_question_count": target_surface.locked_question_count,
        "source_slot": state.source.slot,
        "source_priority": priority,
        "source_plan_sha256": state.source.plan.plan_sha256,
        "source_prepared_sha256": state.prepared.prepared_sha256,
        "source_authorization_sha256": state.authorization.authorization_sha256,
        "source_terminal_sha256": state.terminal.terminal_sha256,
        "source_nested_reuse_terminal_sha256": (
            None if state.nested_terminal is None else state.nested_terminal.terminal_sha256
        ),
        "source_nested_reuse_record_sha256": (
            None if nested_record is None else nested_record.record_sha256
        ),
        "source_intent_sha256": intent.intent_sha256,
        "source_request_key": intent.request_key,
        "source_surface_sha256": intent.surface.surface_sha256,
        "source_receipt_sha256": None if receipt is None else receipt.receipt_sha256,
        "source_provider_result_sha256": (None if result is None else result.result_sha256),
        "source_incident_sha256": None if incident is None else incident.incident_sha256,
        "source_incident_kind": None if incident is None else incident.kind,
        "source_charged_cost_usd_micros": (
            result.charged_cost_usd_micros
            if result is not None
            else incident.charged_cost_usd_micros
        ),
        "source_retry_permitted": False,
        "target_provider_attempts_permitted_for_entry": 0,
    }
    return EvidenceInferenceFableUnionEntryV2.model_validate(
        {**payload, "entry_sha256": hash_canonical(payload)}
    )


def freeze_evidence_inference_fable_full_union_plan_v2(
    *,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    full_prepared: EvidenceInferenceFablePreparedRuntimeV1,
    full_authorization: EvidenceInferenceFableBudgetAuthorizationV2,
    sources: list[EvidenceInferenceFableUnionSourceV2],
) -> EvidenceInferenceFableFullUnionPlanV2:
    """Freeze the verified priority union; no labels or providers are opened."""

    expected_slots = [
        "poisoned_full_v2",
        "poisoned_pilot_v1",
        "recovery_pilot_v2",
    ]
    if (
        full_plan.mode != "full_paired"
        or full_prepared.retrospective_plan_sha256 != full_plan.plan_sha256
        or full_authorization.prepared_sha256 != full_prepared.prepared_sha256
        or full_authorization.configured_total_budget_usd_micros != 99_000_000
        or len(full_prepared.surfaces) != EXPECTED_FULL_REQUESTS
        or [source.slot for source in sources] != expected_slots
    ):
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_target_or_source_binding_invalid"
        )
    states = [_load_union_source(source) for source in sources]
    target_by_wire: dict[str, tuple[int, EvidenceInferenceFableCallSurfaceV1]] = {}
    for index, surface in enumerate(full_prepared.surfaces):
        if surface.wire_call_sha256 in target_by_wire:
            raise EvidenceInferenceFableFullUnionReuseError("fable_union_duplicate_target_wire")
        target_by_wire[surface.wire_call_sha256] = (index, surface)
    chosen: dict[
        str,
        tuple[
            int,
            _UnionState,
            EvidenceInferenceFableIntentV1,
            EvidenceInferenceFableReceiptV1 | None,
            EvidenceInferenceFableIncidentV1 | None,
        ],
    ] = {}
    shadowed = 0
    for priority, state in enumerate(states):
        for request_key, intent in state.intents.items():
            if intent.surface.wire_call_sha256 not in target_by_wire:
                continue
            receipt = state.receipts.get(request_key)
            artifact = state.incidents.get(request_key)
            incident = artifact if isinstance(artifact, EvidenceInferenceFableIncidentV1) else None
            if receipt is None and incident is None:
                continue
            wire = intent.surface.wire_call_sha256
            if wire in chosen:
                shadowed += 1
                continue
            chosen[wire] = (priority, state, intent, receipt, incident)
    entries = []
    for wire, (priority, state, intent, receipt, incident) in chosen.items():
        target_index, target_surface = target_by_wire[wire]
        entries.append(
            _entry(
                target_index=target_index,
                target_surface=target_surface,
                priority=priority,
                state=state,
                intent=intent,
                receipt=receipt,
                incident=incident,
            )
        )
    entries.sort(key=lambda item: item.target_execution_index)
    bindings = [
        EvidenceInferenceFableUnionSourceBindingV2(
            slot=state.source.slot,
            priority=priority,
            plan_sha256=state.source.plan.plan_sha256,
            prepared_sha256=state.prepared.prepared_sha256,
            authorization_sha256=state.authorization.authorization_sha256,
            terminal_sha256=state.terminal.terminal_sha256,
            terminal_status=state.terminal.status,
            nested_reuse_terminal_sha256=(
                None if state.nested_terminal is None else state.nested_terminal.terminal_sha256
            ),
            source_paths_serialized=False,
            source_workspace_mutation_permitted=False,
        )
        for priority, state in enumerate(states)
    ]
    payload = {
        "plan_version": "evidence-inference-fable-full-union-plan-v2",
        "full_plan_sha256": full_plan.plan_sha256,
        "full_prepared_sha256": full_prepared.prepared_sha256,
        "full_authorization_sha256": full_authorization.authorization_sha256,
        "configured_total_budget_usd_micros": 99_000_000,
        "full_request_count": EXPECTED_FULL_REQUESTS,
        "source_priority": expected_slots,
        "source_bindings": bindings,
        "entries": entries,
        "adopted_terminal_receipt_count": EXPECTED_RECEIPTS,
        "inherited_ambiguous_failure_count": EXPECTED_AMBIGUITIES,
        "maximum_new_provider_attempt_count": EXPECTED_NEW_CALLS,
        "shadowed_lower_priority_candidate_count": shadowed,
        "transitively_reused_nested_record_count": sum(
            entry.source_nested_reuse_record_sha256 is not None for entry in entries
        ),
        "exact_wire_hash_and_deep_call_equality_required": True,
        "source_workspaces_immutable": True,
        "inherited_ambiguity_retry_permitted": False,
        "labels_opened": False,
        "provider_calls_made_while_planning": 0,
        "scientific_claim_authority": False,
        "claim_release_authority": False,
    }
    try:
        return EvidenceInferenceFableFullUnionPlanV2.model_validate(
            {**payload, "plan_sha256": hash_canonical(payload)}
        )
    except ValueError as exc:
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_expected_22_receipts_2_ambiguities_358_new"
        ) from exc


def prepare_evidence_inference_fable_full_union_v2(
    *, workspace: Path, union_plan: EvidenceInferenceFableFullUnionPlanV2
) -> None:
    if workspace.is_symlink() or not workspace.is_dir():
        raise EvidenceInferenceFableFullUnionReuseError("fable_union_workspace_unsafe")
    prepared = EvidenceInferenceFablePreparedRuntimeV1.model_validate(
        _read_object(workspace / "00-prepared.json")
    )
    parsed_authorization = parse_evidence_inference_fable_budget_authorization_v1(
        _read_object(workspace / "01-authorization.json")
    )
    if not isinstance(parsed_authorization, EvidenceInferenceFableBudgetAuthorizationV2):
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_target_requires_headroom_authorization_v2"
        )
    authorization = parsed_authorization
    if (
        prepared.prepared_sha256 != union_plan.full_prepared_sha256
        or authorization.authorization_sha256 != union_plan.full_authorization_sha256
        or authorization.configured_total_budget_usd_micros != 99_000_000
        or (workspace / "02-terminal.json").exists()
        or any((workspace / name).exists() for name in ("intents", "receipts", "incidents"))
    ):
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_target_not_fresh_or_binding_mismatch"
        )
    root = workspace / UNION_DIRECTORY
    path = root / UNION_PLAN_FILE
    if root.exists():
        if root.is_symlink() or not root.is_dir() or not path.is_file():
            raise EvidenceInferenceFableFullUnionReuseError("fable_union_directory_replay_unsafe")
        if EvidenceInferenceFableFullUnionPlanV2.model_validate(_read_object(path)) != union_plan:
            raise EvidenceInferenceFableFullUnionReuseError("fable_union_plan_replay_mismatch")
        return
    root.mkdir(mode=0o700)
    (root / "records").mkdir(mode=0o700)
    atomic_write_json(path, union_plan)


def _target_liability(
    *,
    authorization: EvidenceInferenceFableBudgetAuthorizationV2,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    entry: EvidenceInferenceFableUnionEntryV2,
) -> int:
    try:
        liability = authorization.certified_request_liabilities_usd_micros[entry.target_request_key]
    except KeyError as exc:
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_target_liability_missing"
        ) from exc
    if (
        liability
        > full_plan.roster[entry.target_execution_index].cost.full_context_hard_liability_usd_micros
    ):
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_target_liability_exceeds_surface_bound"
        )
    return liability


@dataclass(frozen=True)
class _Derived:
    result: EvidenceInferenceFableProviderResultV1
    receipt: EvidenceInferenceFableReceiptV1
    incident: EvidenceInferenceFableIncidentV2 | None
    record: EvidenceInferenceFableUnionRecordV2


def _derive(
    *,
    entry: EvidenceInferenceFableUnionEntryV2,
    union_plan: EvidenceInferenceFableFullUnionPlanV2,
    target_intent: EvidenceInferenceFableIntentV1,
    target_surface: EvidenceInferenceFableCallSurfaceV1,
    target_authorization: EvidenceInferenceFableBudgetAuthorizationV2,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    states: Mapping[UnionSourceSlot, _UnionState],
) -> _Derived:
    state = states[entry.source_slot]
    source_intent = state.intents.get(entry.source_request_key)
    if (
        source_intent is None
        or source_intent.intent_sha256 != entry.source_intent_sha256
        or target_intent.request_key != entry.target_request_key
        or target_intent.surface != target_surface
        or target_surface.surface_sha256 != entry.target_surface_sha256
        or target_surface.wire_call_sha256 != entry.wire_call_sha256
        or _wire_identity(target_surface) != _wire_identity(source_intent.surface)
        or target_intent.authorization_sha256 != target_authorization.authorization_sha256
    ):
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_target_or_source_intent_mismatch"
        )
    target_incident = None
    if entry.adoption_kind == "terminal_receipt":
        source_receipt = state.receipts.get(entry.source_request_key)
        if (
            source_receipt is None
            or source_receipt.receipt_sha256 != entry.source_receipt_sha256
            or source_receipt.provider_result.result_sha256 != entry.source_provider_result_sha256
        ):
            raise EvidenceInferenceFableFullUnionReuseError("fable_union_source_receipt_mismatch")
        source_payload = source_receipt.provider_result.model_dump(
            mode="json", exclude={"request_key", "surface_sha256", "result_sha256"}
        )
        result_payload = {
            "request_key": target_surface.request_key,
            "surface_sha256": target_surface.surface_sha256,
            **source_payload,
        }
        result = EvidenceInferenceFableProviderResultV1.model_validate(
            {**result_payload, "result_sha256": hash_canonical(result_payload)}
        )
    else:
        source_incident = state.incidents.get(entry.source_request_key)
        liability = _target_liability(
            authorization=target_authorization,
            full_plan=full_plan,
            entry=entry,
        )
        if (
            not isinstance(source_incident, EvidenceInferenceFableIncidentV1)
            or source_incident.incident_sha256 != entry.source_incident_sha256
            or source_incident.kind != entry.source_incident_kind
            or source_incident.retry_permitted
            or source_incident.charged_cost_usd_micros != entry.source_charged_cost_usd_micros
        ):
            raise EvidenceInferenceFableFullUnionReuseError(
                "fable_union_source_ambiguity_or_liability_mismatch"
            )
        failure_code = cast(
            Literal[
                "provider_call_raised_after_durable_intent",
                "provider_result_invalid_after_return",
            ],
            source_incident.kind,
        )
        result_payload = {
            "result_version": "evidence-inference-fable-provider-result-v1",
            "request_key": target_surface.request_key,
            "surface_sha256": target_surface.surface_sha256,
            "transport_attempt_count": 1,
            "sdk_retry_count": 0,
            "outcome": "failed",
            "response_id": None,
            "response_model": None,
            "parsed_json": None,
            "input_tokens": None,
            "output_tokens": None,
            "reported_cost_usd_micros": None,
            "charged_cost_usd_micros": liability,
            "cost_basis": "unknown_usage_hard_liability",
            "response_text_sha256": None,
            "failure_code": failure_code,
        }
        result = EvidenceInferenceFableProviderResultV1.model_validate(
            {**result_payload, "result_sha256": hash_canonical(result_payload)}
        )
        incident_payload = {
            "incident_version": "evidence-inference-fable-incident-v2",
            "status": "failed_request_archived_continue",
            "kind": failure_code,
            "intent_sha256": target_intent.intent_sha256,
            "request_key": target_surface.request_key,
            "charged_cost_usd_micros": liability,
            "cost_basis": "unknown_usage_hard_liability",
            "retry_permitted": False,
            "sanitization_policy": INCIDENT_SANITIZATION_POLICY,
            "exception_type": "InheritedSourceAmbiguousAttempt",
            "http_status_code": None,
            "provider_request_id": None,
            "message_redacted": ("Inherited exact-wire ambiguity; provider call was not retried."),
            "message_was_truncated": False,
            "derived_provider_result_sha256": result.result_sha256,
        }
        target_incident = EvidenceInferenceFableIncidentV2.model_validate(
            {**incident_payload, "incident_sha256": hash_canonical(incident_payload)}
        )
    receipt_payload = {
        "receipt_version": "evidence-inference-fable-receipt-v1",
        "intent_sha256": target_intent.intent_sha256,
        "request_key": target_surface.request_key,
        "provider_result": result,
        "locked_question_count": target_surface.locked_question_count,
        "locked_questions_scored_incorrect": (
            target_surface.locked_question_count if result.outcome == "failed" else 0
        ),
    }
    receipt = EvidenceInferenceFableReceiptV1.model_validate(
        {**receipt_payload, "receipt_sha256": hash_canonical(receipt_payload)}
    )
    record_payload = {
        "record_version": "evidence-inference-fable-full-union-record-v2",
        "union_plan_sha256": union_plan.plan_sha256,
        "entry_sha256": entry.entry_sha256,
        "adoption_kind": entry.adoption_kind,
        "source_slot": entry.source_slot,
        "source_terminal_sha256": entry.source_terminal_sha256,
        "source_nested_reuse_terminal_sha256": (entry.source_nested_reuse_terminal_sha256),
        "source_nested_reuse_record_sha256": entry.source_nested_reuse_record_sha256,
        "source_intent_sha256": entry.source_intent_sha256,
        "source_receipt_sha256": entry.source_receipt_sha256,
        "source_provider_result_sha256": entry.source_provider_result_sha256,
        "source_incident_sha256": entry.source_incident_sha256,
        "target_authorization_sha256": target_authorization.authorization_sha256,
        "target_request_key": entry.target_request_key,
        "target_surface_sha256": entry.target_surface_sha256,
        "wire_call_sha256": entry.wire_call_sha256,
        "target_intent_sha256": target_intent.intent_sha256,
        "target_provider_result_sha256": result.result_sha256,
        "expected_target_receipt_sha256": receipt.receipt_sha256,
        "expected_target_incident_sha256": (
            None if target_incident is None else target_incident.incident_sha256
        ),
        "target_provider_attempt_count": 0,
        "source_attempt_retry_permitted": False,
        "locked_questions_scored_incorrect": receipt.locked_questions_scored_incorrect,
        "charged_cost_usd_micros": result.charged_cost_usd_micros,
    }
    record = EvidenceInferenceFableUnionRecordV2.model_validate(
        {**record_payload, "record_sha256": hash_canonical(record_payload)}
    )
    return _Derived(result=result, receipt=receipt, incident=target_incident, record=record)


def _write_record(workspace: Path, record: EvidenceInferenceFableUnionRecordV2) -> None:
    path = workspace / UNION_DIRECTORY / "records" / f"{record.target_request_key}.json"
    if path.exists():
        if EvidenceInferenceFableUnionRecordV2.model_validate(_read_object(path)) != record:
            raise EvidenceInferenceFableFullUnionReuseError("fable_union_record_replay_mismatch")
    else:
        atomic_write_json(path, record)


class _UnionClient:
    def __init__(
        self,
        *,
        workspace: Path,
        full_plan: EvidenceInferenceFableRetrospectivePlanV1,
        authorization: EvidenceInferenceFableBudgetAuthorizationV2,
        union_plan: EvidenceInferenceFableFullUnionPlanV2,
        states: Mapping[UnionSourceSlot, _UnionState],
        delegate: EvidenceInferenceFableClientProtocol,
    ) -> None:
        self.workspace = workspace
        self.full_plan = full_plan
        self.authorization = authorization
        self.union_plan = union_plan
        self.states = states
        self.delegate = delegate
        self.entries = {entry.target_request_key: entry for entry in union_plan.entries}

    def generate(
        self, surface: EvidenceInferenceFableCallSurfaceV1
    ) -> EvidenceInferenceFableProviderResultV1:
        entry = self.entries.get(surface.request_key)
        if entry is None:
            return self.delegate.generate(surface)
        intent = EvidenceInferenceFableIntentV1.model_validate(
            _read_object(self.workspace / "intents" / f"{surface.request_key}.json")
        )
        derived = _derive(
            entry=entry,
            union_plan=self.union_plan,
            target_intent=intent,
            target_surface=surface,
            target_authorization=self.authorization,
            full_plan=self.full_plan,
            states=self.states,
        )
        _write_record(self.workspace, derived.record)
        if derived.incident is not None:
            path = self.workspace / "incidents" / f"{surface.request_key}.json"
            if path.exists():
                if (
                    EvidenceInferenceFableIncidentV2.model_validate(_read_object(path))
                    != derived.incident
                ):
                    raise EvidenceInferenceFableFullUnionReuseError(
                        "fable_union_target_incident_replay_mismatch"
                    )
            else:
                atomic_write_json(path, derived.incident)
        return derived.result


def _context(
    *,
    workspace: Path,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    sources: list[EvidenceInferenceFableUnionSourceV2],
) -> tuple[
    EvidenceInferenceFablePreparedRuntimeV1,
    EvidenceInferenceFableBudgetAuthorizationV2,
    EvidenceInferenceFableFullUnionPlanV2,
    dict[UnionSourceSlot, _UnionState],
]:
    prepared = EvidenceInferenceFablePreparedRuntimeV1.model_validate(
        _read_object(workspace / "00-prepared.json")
    )
    parsed_authorization = parse_evidence_inference_fable_budget_authorization_v1(
        _read_object(workspace / "01-authorization.json")
    )
    if not isinstance(parsed_authorization, EvidenceInferenceFableBudgetAuthorizationV2):
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_target_requires_headroom_authorization_v2"
        )
    authorization = parsed_authorization
    archived = EvidenceInferenceFableFullUnionPlanV2.model_validate(
        _read_object(workspace / UNION_DIRECTORY / UNION_PLAN_FILE)
    )
    expected = freeze_evidence_inference_fable_full_union_plan_v2(
        full_plan=full_plan,
        full_prepared=prepared,
        full_authorization=authorization,
        sources=sources,
    )
    if archived != expected:
        raise EvidenceInferenceFableFullUnionReuseError("fable_union_plan_external_replay_mismatch")
    states = {source.slot: _load_union_source(source) for source in sources}
    return prepared, authorization, archived, states


def _recover(
    *,
    workspace: Path,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    prepared: EvidenceInferenceFablePreparedRuntimeV1,
    authorization: EvidenceInferenceFableBudgetAuthorizationV2,
    union_plan: EvidenceInferenceFableFullUnionPlanV2,
    states: Mapping[UnionSourceSlot, _UnionState],
) -> None:
    surfaces = {surface.request_key: surface for surface in prepared.surfaces}
    for entry in union_plan.entries:
        intent_path = workspace / "intents" / f"{entry.target_request_key}.json"
        receipt_path = workspace / "receipts" / f"{entry.target_request_key}.json"
        incident_path = workspace / "incidents" / f"{entry.target_request_key}.json"
        record_path = workspace / UNION_DIRECTORY / "records" / f"{entry.target_request_key}.json"
        if not any(
            path.exists() for path in (intent_path, receipt_path, incident_path, record_path)
        ):
            continue
        if not intent_path.exists():
            raise EvidenceInferenceFableFullUnionReuseError(
                "fable_union_artifact_without_target_intent"
            )
        intent = EvidenceInferenceFableIntentV1.model_validate(_read_object(intent_path))
        derived = _derive(
            entry=entry,
            union_plan=union_plan,
            target_intent=intent,
            target_surface=surfaces[entry.target_request_key],
            target_authorization=authorization,
            full_plan=full_plan,
            states=states,
        )
        if not record_path.exists() and (receipt_path.exists() or incident_path.exists()):
            raise EvidenceInferenceFableFullUnionReuseError(
                "fable_union_result_without_prior_record"
            )
        _write_record(workspace, derived.record)
        if derived.incident is not None:
            if incident_path.exists():
                if (
                    EvidenceInferenceFableIncidentV2.model_validate(_read_object(incident_path))
                    != derived.incident
                ):
                    raise EvidenceInferenceFableFullUnionReuseError(
                        "fable_union_recovered_incident_mismatch"
                    )
            else:
                atomic_write_json(incident_path, derived.incident)
        elif incident_path.exists():
            raise EvidenceInferenceFableFullUnionReuseError(
                "fable_union_receipt_adoption_has_incident"
            )
        if receipt_path.exists():
            if (
                EvidenceInferenceFableReceiptV1.model_validate(_read_object(receipt_path))
                != derived.receipt
            ):
                raise EvidenceInferenceFableFullUnionReuseError(
                    "fable_union_recovered_receipt_mismatch"
                )
        else:
            atomic_write_json(receipt_path, derived.receipt)


def _validate_records(
    *,
    workspace: Path,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    prepared: EvidenceInferenceFablePreparedRuntimeV1,
    authorization: EvidenceInferenceFableBudgetAuthorizationV2,
    union_plan: EvidenceInferenceFableFullUnionPlanV2,
    states: Mapping[UnionSourceSlot, _UnionState],
    target_terminal: EvidenceInferenceFableTerminalV1,
) -> None:
    records = _artifact_map(
        workspace / UNION_DIRECTORY / "records", EvidenceInferenceFableUnionRecordV2
    )
    expected = {
        entry.target_request_key: entry
        for entry in union_plan.entries
        if entry.target_execution_index < target_terminal.completed_request_count
    }
    if set(records) != set(expected):
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_realized_record_roster_mismatch"
        )
    surfaces = {surface.request_key: surface for surface in prepared.surfaces}
    for key, entry in expected.items():
        intent = EvidenceInferenceFableIntentV1.model_validate(
            _read_object(workspace / "intents" / f"{key}.json")
        )
        derived = _derive(
            entry=entry,
            union_plan=union_plan,
            target_intent=intent,
            target_surface=surfaces[key],
            target_authorization=authorization,
            full_plan=full_plan,
            states=states,
        )
        receipt = EvidenceInferenceFableReceiptV1.model_validate(
            _read_object(workspace / "receipts" / f"{key}.json")
        )
        if records[key] != derived.record or receipt != derived.receipt:
            raise EvidenceInferenceFableFullUnionReuseError(
                "fable_union_realized_artifact_mismatch"
            )
        incident_path = workspace / "incidents" / f"{key}.json"
        if derived.incident is None:
            if incident_path.exists():
                raise EvidenceInferenceFableFullUnionReuseError(
                    "fable_union_unexpected_target_incident"
                )
        elif (
            EvidenceInferenceFableIncidentV2.model_validate(_read_object(incident_path))
            != derived.incident
        ):
            raise EvidenceInferenceFableFullUnionReuseError(
                "fable_union_inherited_incident_mismatch"
            )


def _terminal(
    *,
    workspace: Path,
    union_plan: EvidenceInferenceFableFullUnionPlanV2,
    target_terminal: EvidenceInferenceFableTerminalV1,
) -> EvidenceInferenceFableFullUnionTerminalV2:
    records = _artifact_map(
        workspace / UNION_DIRECTORY / "records", EvidenceInferenceFableUnionRecordV2
    )
    intents = _artifact_map(workspace / "intents", EvidenceInferenceFableIntentV1)
    receipts = _artifact_map(workspace / "receipts", EvidenceInferenceFableReceiptV1)
    adopted_cost = 0
    receipt_count = 0
    ambiguity_count = 0
    for key, record in records.items():
        receipt = receipts.get(key)
        if receipt is None or receipt.receipt_sha256 != record.expected_target_receipt_sha256:
            raise EvidenceInferenceFableFullUnionReuseError(
                "fable_union_terminal_record_receipt_mismatch"
            )
        adopted_cost += receipt.provider_result.charged_cost_usd_micros
        receipt_count += record.adoption_kind == "terminal_receipt"
        ambiguity_count += record.adoption_kind == "inherited_ambiguous_failure"
    payload = {
        "terminal_version": "evidence-inference-fable-full-union-terminal-v2",
        "union_plan_sha256": union_plan.plan_sha256,
        "target_runtime_terminal_sha256": target_terminal.terminal_sha256,
        "target_runtime_status": target_terminal.status,
        "target_completed_request_count": target_terminal.completed_request_count,
        "realized_adopted_terminal_receipt_count": receipt_count,
        "realized_inherited_ambiguous_failure_count": ambiguity_count,
        "new_provider_attempt_count": len(intents) - len(records),
        "maximum_new_provider_attempt_count": EXPECTED_NEW_CALLS,
        "target_accounted_spend_usd_micros": (target_terminal.cumulative_reported_spend_usd_micros),
        "adopted_target_accounted_spend_usd_micros": adopted_cost,
        "new_provider_accounted_spend_usd_micros": (
            target_terminal.cumulative_reported_spend_usd_micros - adopted_cost
        ),
        "source_terminal_artifact_lineage_count": len(records),
        "inherited_ambiguous_attempts_retried": 0,
        "target_provider_attempts_for_adopted_entries": 0,
        "full_population_score_permitted": (
            target_terminal.status == "completed"
            and receipt_count == EXPECTED_RECEIPTS
            and ambiguity_count == EXPECTED_AMBIGUITIES
            and len(intents) - len(records) == EXPECTED_NEW_CALLS
        ),
        "scoring_requires_this_union_terminal": True,
        "scientific_claim_authority": False,
        "confirmatory_gepa_improvement_authority": False,
        "claim_release_authority": False,
    }
    if payload["new_provider_accounted_spend_usd_micros"] < 0:
        raise EvidenceInferenceFableFullUnionReuseError("fable_union_adopted_spend_exceeds_total")
    return EvidenceInferenceFableFullUnionTerminalV2.model_validate(
        {**payload, "terminal_sha256": hash_canonical(payload)}
    )


def execute_evidence_inference_fable_full_union_v2(
    *,
    workspace: Path,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    sources: list[EvidenceInferenceFableUnionSourceV2],
    delegate: EvidenceInferenceFableClientProtocol,
) -> EvidenceInferenceFableFullUnionTerminalV2:
    validate_evidence_inference_fable_full_union_paths_v2(workspace=workspace, sources=sources)
    with _union_lock(workspace):
        prepared, authorization, union_plan, states = _context(
            workspace=workspace, full_plan=full_plan, sources=sources
        )
        _recover(
            workspace=workspace,
            full_plan=full_plan,
            prepared=prepared,
            authorization=authorization,
            union_plan=union_plan,
            states=states,
        )
        target_terminal = execute_evidence_inference_fable_paired_v1(
            workspace=workspace,
            plan=full_plan,
            client=_UnionClient(
                workspace=workspace,
                full_plan=full_plan,
                authorization=authorization,
                union_plan=union_plan,
                states=states,
                delegate=delegate,
            ),
        )
        _validate_records(
            workspace=workspace,
            full_plan=full_plan,
            prepared=prepared,
            authorization=authorization,
            union_plan=union_plan,
            states=states,
            target_terminal=target_terminal,
        )
        observed = _terminal(
            workspace=workspace,
            union_plan=union_plan,
            target_terminal=target_terminal,
        )
        path = workspace / UNION_DIRECTORY / UNION_TERMINAL_FILE
        if path.exists():
            if (
                EvidenceInferenceFableFullUnionTerminalV2.model_validate(_read_object(path))
                != observed
            ):
                raise EvidenceInferenceFableFullUnionReuseError(
                    "fable_union_terminal_replay_mismatch"
                )
        else:
            atomic_write_json(path, observed)
        return observed


def validate_evidence_inference_fable_full_union_v2(
    *,
    workspace: Path,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    sources: list[EvidenceInferenceFableUnionSourceV2],
) -> EvidenceInferenceFableFullUnionTerminalV2:
    validate_evidence_inference_fable_full_union_paths_v2(workspace=workspace, sources=sources)
    with _union_lock(workspace):
        prepared, authorization, union_plan, states = _context(
            workspace=workspace, full_plan=full_plan, sources=sources
        )
        target_terminal = validate_evidence_inference_fable_workspace_v1(
            workspace=workspace, plan=full_plan
        )
        _validate_records(
            workspace=workspace,
            full_plan=full_plan,
            prepared=prepared,
            authorization=authorization,
            union_plan=union_plan,
            states=states,
            target_terminal=target_terminal,
        )
        expected = _terminal(
            workspace=workspace,
            union_plan=union_plan,
            target_terminal=target_terminal,
        )
        archived = EvidenceInferenceFableFullUnionTerminalV2.model_validate(
            _read_object(workspace / UNION_DIRECTORY / UNION_TERMINAL_FILE)
        )
        if archived != expected:
            raise EvidenceInferenceFableFullUnionReuseError(
                "fable_union_terminal_external_replay_mismatch"
            )
        return archived


def require_evidence_inference_fable_full_union_scoring_v2(
    *,
    workspace: Path,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    sources: list[EvidenceInferenceFableUnionSourceV2],
) -> EvidenceInferenceFableFullUnionTerminalV2:
    terminal = validate_evidence_inference_fable_full_union_v2(
        workspace=workspace, full_plan=full_plan, sources=sources
    )
    if not terminal.full_population_score_permitted:
        raise EvidenceInferenceFableFullUnionReuseError(
            "fable_union_scoring_prerequisite_not_satisfied"
        )
    return terminal
