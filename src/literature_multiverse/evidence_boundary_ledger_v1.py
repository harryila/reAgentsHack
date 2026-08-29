"""Fail-closed cross-artifact evidence and scientific-authority ledger.

The repository contains several useful but intentionally different kinds of evidence:
real retrospective benchmark metrics, label-blind execution-yield diagnostics,
misspecified simulation stress tests, and executable evaluation contracts that have no
adjudicated empirical artifact yet.  This module replays those boundaries into one
strict ledger.  It never interprets simulation or successful mechanics as evidence of
real adaptive-policy effectiveness, calibration, or claim-release safety.

Most rows open only aggregate public artifacts and self-hashed mechanics reports.  One
separately marked row externally replays the authorized passage-runtime execution
bundle, which contains public-source passage payloads.  Evaluator labels, reference
verdicts, and human adjudication records remain outside this ledger's input surface.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.adaptive_stress_study import (
    build_adaptive_stress_study_artifact,
    validate_adaptive_stress_study_artifact,
)
from literature_multiverse.hosted_exact_once import freeze_hosted_exact_once_intent
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.metasyn_passage_hosted_bundle_v2 import (
    MetaSynPassageHostedExecutionBundleV2,
    validate_metasyn_passage_hosted_execution_bundle_v2,
)
from literature_multiverse.metasyn_passage_hosted_runtime_v2 import (
    InventoryLedgerV2,
    PacketCallResultV2,
    PacketRosterV2,
    PacketSmokeReceiptV2,
    metasyn_passage_hosted_runtime_status_v2,
    validate_metasyn_passage_inventory_ledger_v2,
    validate_metasyn_passage_packet_result_v2,
    validate_metasyn_passage_packet_roster_v2,
)
from literature_multiverse.metasyn_synthesis_yield import (
    MetaSynSynthesisYieldPublicSummaryV1,
    MetaSynSynthesisYieldReportV1,
    validate_metasyn_synthesis_yield_public_summary,
)
from literature_multiverse.metasyn_synthesis_yield_v2 import (
    MetaSynSynthesisYieldPublicSummaryV2,
    MetaSynSynthesisYieldReportV2,
    validate_metasyn_synthesis_yield_v2_public_summary,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.pipeline_fingerprint import require_pipeline_fingerprint_match
from literature_multiverse.public_artifacts import (
    PublicArtifactValidationError,
    _validate_local_suite,
)
from literature_multiverse.question_evaluation import (
    compute_question_evaluation_pipeline_fingerprint,
)

LEDGER_VERSION = "evidence-boundary-ledger-v1"
RECORD_VERSION = "evidence-boundary-record-v1"
NEXT_REQUIRED_AUTHORITY_GATE = (
    "two complete questions establish only structural replay. Item-risk calibration "
    "eligibility requires at least five calibration-side complete questions (at least "
    "four entering the conservative UCL) plus at least four question/paper-disjoint "
    "unopened evaluation questions. The main adaptive-effectiveness claim additionally "
    "requires a substantially larger, prespecified independently expert-adjudicated test "
    "population with realized total person-minutes, exact certificate-bound replay "
    "states, frozen policy inputs, and no question/claim/paper overlap; none of these "
    "minima by itself authorizes calibrated release"
)
PROHIBITED_INFERENCES = sorted(
    {
        "A contract-valid evaluator is not an empirical policy evaluation.",
        ("A label-blind completed synthesis would prove mechanics only, not synthesis accuracy."),
        "A procedural split cannot restore pristine status after labels were opened.",
        (
            "Simulation policy contrasts cannot establish real human-efficiency "
            "or released-claim error."
        ),
        "Typed-effect yield cannot establish extraction accuracy without reference labels.",
        "Zero or positive mechanics yield cannot authorize claim release.",
    }
)

_REQUIRED_RECORD_IDS = {
    "adaptive_stress_simulation_v1",
    "local_benchmark_suite_v1",
    "metasyn_passage_runtime_v2_failed_smoke",
    "metasyn_synthesis_yield_v1",
    "metasyn_synthesis_yield_v2",
    "question_policy_evaluation_contract_v7",
}
_OPTIONAL_RECORD_ID = "metasyn_passage_rescue_v3_pre_call_blocker"


class EvidenceBoundaryLedgerError(ValueError):
    """A required input or derived authority boundary was invalid."""


class EvidenceClass(StrEnum):
    REAL_RETROSPECTIVE_NONPRISTINE = "real_retrospective_nonpristine"
    REAL_LABEL_BLIND_MECHANICS = "real_label_blind_mechanics"
    REAL_INCOMPLETE_SOURCE_EXECUTION = "real_incomplete_source_execution"
    REAL_SOURCE_PREFLIGHT_BLOCKED = "real_source_preflight_blocked"
    SIMULATED = "simulated"
    CONTRACT_ONLY = "contract_only"


class ValidationDepth(StrEnum):
    FULL_SYNTHETIC_RECOMPUTATION = "full_synthetic_recomputation"
    AGGREGATE_CROSS_ARTIFACT_REPLAY = "aggregate_cross_artifact_replay"
    AGGREGATE_SELF_HASH_AND_CURRENT_SOURCE_LINEAGE = (
        "aggregate_self_hash_and_current_source_lineage"
    )
    PRIVATE_TO_PUBLIC_EXACT_REPLAY = "private_to_public_exact_replay"
    INCOMPLETE_EXACT_ONCE_WORKSPACE_REPLAY = "incomplete_exact_once_workspace_replay"
    CONTRACT_FINGERPRINT_ONLY = "contract_fingerprint_only"


class SourcePayloadState(StrEnum):
    SYNTHETIC_NOT_APPLICABLE = "synthetic_not_applicable"
    OPENED_UPSTREAM_AGGREGATE_ONLY_HERE = "opened_upstream_aggregate_only_here"
    OPENED_UPSTREAM_PRIVATE_MECHANICS_ONLY_HERE = "opened_upstream_private_mechanics_only_here"
    UNOPENED_CONTRACT_ONLY = "unopened_contract_only"


class LabelState(StrEnum):
    SYNTHETIC_ORACLE_ONLY = "synthetic_oracle_only"
    BENCHMARK_LABELS_PREVIOUSLY_OPENED_AGGREGATES_ONLY = (
        "benchmark_labels_previously_opened_aggregates_only"
    )
    REFERENCE_FIELDS_EXPLICITLY_UNOPENED = "reference_fields_explicitly_unopened"
    NO_LABELS_CONTRACT_ONLY = "no_labels_contract_only"


class TypedEffectStatus(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    ZERO_RUNTIME_TYPED_EFFECTS = "zero_runtime_typed_effects"
    RUNTIME_TYPED_EFFECTS_WITHOUT_RELEASE_GRADE_ESTIMATES = (
        "runtime_typed_effects_without_release_grade_estimates"
    )
    RELEASE_GRADE_ESTIMATES_PRESENT = "release_grade_estimates_present"


class SynthesisMechanicsStatus(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    GRAPH_BUILT_WITH_ZERO_ESTIMABLE_EFFECTS = "graph_built_with_zero_estimable_effects"
    ESTIMABLE_GRAPH_WITHOUT_COMPLETED_SYNTHESIS = "estimable_graph_without_completed_synthesis"
    COMPLETED_MECHANICS_ONLY = "completed_mechanics_only"


class AuthorityKind(StrEnum):
    NONE_CONTRACT_ONLY = "none_contract_only"
    SIMULATION_BEHAVIOR_ONLY = "simulation_behavior_only"
    RETROSPECTIVE_MATCHED_SUBSET_METRIC_ONLY = "retrospective_matched_subset_metric_only"
    REAL_EXECUTION_YIELD_ONLY = "real_execution_yield_only"
    REAL_OFFLINE_PREFLIGHT_BLOCKER_ONLY = "real_offline_preflight_blocker_only"


class AuthorizedEmpiricalScope(StrEnum):
    NONE = "none"
    REAL_OFFLINE_PREFLIGHT_BLOCKER_ONLY = "real_offline_preflight_blocker_only"
    RETROSPECTIVE_MATCHED_SUBSET_METRICS_ONLY = (
        "retrospective_nonpristine_matched_subset_metrics_only"
    )
    REAL_EXECUTION_YIELD_MECHANICS_ONLY = "real_label_blind_execution_yield_mechanics_only"


_AUTHORITY_SCOPE_BY_KIND = {
    AuthorityKind.NONE_CONTRACT_ONLY: AuthorizedEmpiricalScope.NONE,
    AuthorityKind.SIMULATION_BEHAVIOR_ONLY: AuthorizedEmpiricalScope.NONE,
    AuthorityKind.RETROSPECTIVE_MATCHED_SUBSET_METRIC_ONLY: (
        AuthorizedEmpiricalScope.RETROSPECTIVE_MATCHED_SUBSET_METRICS_ONLY
    ),
    AuthorityKind.REAL_EXECUTION_YIELD_ONLY: (
        AuthorizedEmpiricalScope.REAL_EXECUTION_YIELD_MECHANICS_ONLY
    ),
    AuthorityKind.REAL_OFFLINE_PREFLIGHT_BLOCKER_ONLY: (
        AuthorizedEmpiricalScope.REAL_OFFLINE_PREFLIGHT_BLOCKER_ONLY
    ),
}

_AUTHORITY_KIND_BY_EVIDENCE_CLASS = {
    EvidenceClass.SIMULATED: AuthorityKind.SIMULATION_BEHAVIOR_ONLY,
    EvidenceClass.CONTRACT_ONLY: AuthorityKind.NONE_CONTRACT_ONLY,
    EvidenceClass.REAL_RETROSPECTIVE_NONPRISTINE: (
        AuthorityKind.RETROSPECTIVE_MATCHED_SUBSET_METRIC_ONLY
    ),
    EvidenceClass.REAL_LABEL_BLIND_MECHANICS: AuthorityKind.REAL_EXECUTION_YIELD_ONLY,
    EvidenceClass.REAL_INCOMPLETE_SOURCE_EXECUTION: AuthorityKind.REAL_EXECUTION_YIELD_ONLY,
    EvidenceClass.REAL_SOURCE_PREFLIGHT_BLOCKED: (
        AuthorityKind.REAL_OFFLINE_PREFLIGHT_BLOCKER_ONLY
    ),
}

_RUNTIME_STATE_BY_EVIDENCE_CLASS = {
    EvidenceClass.SIMULATED: "completed_artifact",
    EvidenceClass.CONTRACT_ONLY: "contract_only_not_executed",
    EvidenceClass.REAL_RETROSPECTIVE_NONPRISTINE: "completed_artifact",
    EvidenceClass.REAL_LABEL_BLIND_MECHANICS: "finalized_mechanics_report",
    EvidenceClass.REAL_INCOMPLETE_SOURCE_EXECUTION: (
        "packet_roster_frozen_failed_smoke_not_finalized"
    ),
    EvidenceClass.REAL_SOURCE_PREFLIGHT_BLOCKED: (
        "pre_call_blocked_zero_provider_calls_not_execution"
    ),
}

_VALIDATION_DEPTH_BY_RECORD_ID = {
    "adaptive_stress_simulation_v1": ValidationDepth.FULL_SYNTHETIC_RECOMPUTATION,
    "local_benchmark_suite_v1": ValidationDepth.AGGREGATE_CROSS_ARTIFACT_REPLAY,
    "metasyn_retrieval_study_v1": (ValidationDepth.AGGREGATE_SELF_HASH_AND_CURRENT_SOURCE_LINEAGE),
    "metasyn_passage_runtime_v2_failed_smoke": (
        ValidationDepth.INCOMPLETE_EXACT_ONCE_WORKSPACE_REPLAY
    ),
    "metasyn_passage_rescue_v3_pre_call_blocker": (ValidationDepth.AGGREGATE_CROSS_ARTIFACT_REPLAY),
    "metasyn_synthesis_yield_v1": ValidationDepth.PRIVATE_TO_PUBLIC_EXACT_REPLAY,
    "metasyn_synthesis_yield_v2": ValidationDepth.PRIVATE_TO_PUBLIC_EXACT_REPLAY,
    "question_policy_evaluation_contract_v7": ValidationDepth.CONTRACT_FINGERPRINT_ONLY,
}

_SOURCE_STATE_BY_EVIDENCE_CLASS = {
    EvidenceClass.SIMULATED: SourcePayloadState.SYNTHETIC_NOT_APPLICABLE,
    EvidenceClass.CONTRACT_ONLY: SourcePayloadState.UNOPENED_CONTRACT_ONLY,
    EvidenceClass.REAL_RETROSPECTIVE_NONPRISTINE: (
        SourcePayloadState.OPENED_UPSTREAM_AGGREGATE_ONLY_HERE
    ),
    EvidenceClass.REAL_LABEL_BLIND_MECHANICS: (
        SourcePayloadState.OPENED_UPSTREAM_PRIVATE_MECHANICS_ONLY_HERE
    ),
    EvidenceClass.REAL_INCOMPLETE_SOURCE_EXECUTION: (
        SourcePayloadState.OPENED_UPSTREAM_PRIVATE_MECHANICS_ONLY_HERE
    ),
    EvidenceClass.REAL_SOURCE_PREFLIGHT_BLOCKED: (
        SourcePayloadState.OPENED_UPSTREAM_PRIVATE_MECHANICS_ONLY_HERE
    ),
}

_LABEL_STATE_BY_EVIDENCE_CLASS = {
    EvidenceClass.SIMULATED: LabelState.SYNTHETIC_ORACLE_ONLY,
    EvidenceClass.CONTRACT_ONLY: LabelState.NO_LABELS_CONTRACT_ONLY,
    EvidenceClass.REAL_RETROSPECTIVE_NONPRISTINE: (
        LabelState.BENCHMARK_LABELS_PREVIOUSLY_OPENED_AGGREGATES_ONLY
    ),
    EvidenceClass.REAL_LABEL_BLIND_MECHANICS: (LabelState.REFERENCE_FIELDS_EXPLICITLY_UNOPENED),
    EvidenceClass.REAL_INCOMPLETE_SOURCE_EXECUTION: (
        LabelState.REFERENCE_FIELDS_EXPLICITLY_UNOPENED
    ),
    EvidenceClass.REAL_SOURCE_PREFLIGHT_BLOCKED: (LabelState.REFERENCE_FIELDS_EXPLICITLY_UNOPENED),
}

_SCOPE_PRIORITY = {
    AuthorizedEmpiricalScope.NONE: 0,
    AuthorizedEmpiricalScope.REAL_OFFLINE_PREFLIGHT_BLOCKER_ONLY: 1,
    AuthorizedEmpiricalScope.RETROSPECTIVE_MATCHED_SUBSET_METRICS_ONLY: 2,
    AuthorizedEmpiricalScope.REAL_EXECUTION_YIELD_MECHANICS_ONLY: 3,
}


class RegisteredArtifact(ContractModel):
    path: str
    file_sha256: str
    semantic_hash_field: str
    semantic_sha256: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value.startswith("./"):
            raise ValueError("evidence_ledger_artifact_path_not_repository_relative")
        return value

    @field_validator("file_sha256", "semantic_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"evidence_ledger_hash_invalid:{info.field_name}")
        return value


class ValidationReceipt(ContractModel):
    depth: ValidationDepth
    validator_names: Annotated[list[str], Field(min_length=1)]
    exact_replay_match: Literal[True] = True
    self_hash_validated: bool
    current_source_lineage_validated: bool
    raw_empirical_payload_recomputed: Literal[False] = False
    raw_evaluator_or_human_labels_opened: Literal[False] = False

    @field_validator("validator_names")
    @classmethod
    def validate_names(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item for item in value):
            raise ValueError("evidence_ledger_validator_names_not_sorted_unique")
        return value


class SourceAccessBoundary(ContractModel):
    source_payload_state: SourcePayloadState
    raw_source_payload_opened_by_ledger: bool = False
    aggregate_or_mechanics_artifact_contains_raw_article_text: bool = False
    reference_fields_opened_by_ledger: Literal[False] = False


class RuntimeCompletionBoundary(ContractModel):
    state: Literal[
        "completed_artifact",
        "finalized_mechanics_report",
        "packet_roster_frozen_failed_smoke_not_finalized",
        "pre_call_blocked_zero_provider_calls_not_execution",
        "contract_only_not_executed",
    ]
    workspace_finalized: bool | None
    terminal_provider_call_count: Annotated[int, Field(ge=0)]
    remaining_provider_calls_permitted: bool | None
    terminal_roster_complete: bool

    @model_validator(mode="after")
    def validate_state(self) -> RuntimeCompletionBoundary:
        if self.state == "completed_artifact":
            if (
                self.workspace_finalized is not None
                or self.terminal_provider_call_count != 0
                or self.remaining_provider_calls_permitted is not None
                or not self.terminal_roster_complete
            ):
                raise ValueError("evidence_ledger_completed_artifact_state_invalid")
        elif self.state == "packet_roster_frozen_failed_smoke_not_finalized":
            if (
                self.workspace_finalized is not False
                or self.remaining_provider_calls_permitted is not False
                or self.terminal_provider_call_count <= 0
                or self.terminal_roster_complete
            ):
                raise ValueError("evidence_ledger_incomplete_runtime_state_invalid")
        elif self.state == "finalized_mechanics_report":
            if (
                self.workspace_finalized is not True
                or self.terminal_provider_call_count != 0
                or self.remaining_provider_calls_permitted is not False
                or not self.terminal_roster_complete
            ):
                raise ValueError("evidence_ledger_final_runtime_state_invalid")
        elif self.state == "pre_call_blocked_zero_provider_calls_not_execution":
            if (
                self.workspace_finalized is not None
                or self.remaining_provider_calls_permitted is not False
                or self.terminal_provider_call_count != 0
                or self.terminal_roster_complete
            ):
                raise ValueError("evidence_ledger_pre_call_blocker_state_invalid")
        elif self.state == "contract_only_not_executed" and (
            self.workspace_finalized is not None
            or self.remaining_provider_calls_permitted is not None
            or self.terminal_provider_call_count != 0
            or self.terminal_roster_complete
        ):
            raise ValueError("evidence_ledger_contract_runtime_state_invalid")
        return self


class LabelAccessBoundary(ContractModel):
    label_state: LabelState
    human_expert_adjudication_present: Literal[False] = False
    human_expert_adjudication_opened_by_ledger: Literal[False] = False
    complete_human_adjudicated_question_count: Literal[0] = 0


class IndependenceBoundary(ContractModel):
    observation_unit: str
    observed_unit_count: Annotated[int, Field(ge=0)]
    counts_by_partition: dict[str, Annotated[int, Field(ge=0)]]
    uncertainty_resampling_unit: str
    complete_independent_claim_question_count: Literal[0] = 0
    repeated_units_across_artifacts: bool

    @field_validator("counts_by_partition")
    @classmethod
    def validate_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if value != dict(sorted(value.items())):
            raise ValueError("evidence_ledger_partition_counts_not_sorted")
        return value

    @model_validator(mode="after")
    def validate_total(self) -> IndependenceBoundary:
        if self.observed_unit_count != sum(self.counts_by_partition.values()):
            raise ValueError("evidence_ledger_partition_count_total_mismatch")
        return self


class TypedEffectYieldBoundary(ContractModel):
    status: TypedEffectStatus
    publications_evaluated: int | None
    runtime_contract_typed_publications: int | None
    release_grade_estimable_publications: int | None
    graph_estimates: int | None

    @model_validator(mode="after")
    def validate_counts(self) -> TypedEffectYieldBoundary:
        values = (
            self.publications_evaluated,
            self.runtime_contract_typed_publications,
            self.release_grade_estimable_publications,
            self.graph_estimates,
        )
        if self.status is TypedEffectStatus.NOT_APPLICABLE:
            if any(value is not None for value in values):
                raise ValueError("evidence_ledger_not_applicable_typed_counts_present")
            return self
        if any(value is None or value < 0 for value in values):
            raise ValueError("evidence_ledger_typed_counts_missing_or_negative")
        typed = int(self.runtime_contract_typed_publications or 0)
        release_grade = int(self.release_grade_estimable_publications or 0)
        publications = int(self.publications_evaluated or 0)
        if not 0 <= release_grade <= typed <= publications:
            raise ValueError("evidence_ledger_typed_count_order_invalid")
        expected = (
            TypedEffectStatus.ZERO_RUNTIME_TYPED_EFFECTS
            if typed == 0
            else (
                TypedEffectStatus.RELEASE_GRADE_ESTIMATES_PRESENT
                if release_grade > 0
                else TypedEffectStatus.RUNTIME_TYPED_EFFECTS_WITHOUT_RELEASE_GRADE_ESTIMATES
            )
        )
        if self.status is not expected:
            raise ValueError("evidence_ledger_typed_status_count_mismatch")
        return self


class SynthesisMechanicsBoundary(ContractModel):
    status: SynthesisMechanicsStatus
    graph_construction_completed_questions: int | None
    questions_with_estimable_graph: int | None
    synthesis_attempted_groups: int | None
    synthesis_completed_groups: int | None
    questions_with_completed_synthesis: int | None

    @model_validator(mode="after")
    def validate_counts(self) -> SynthesisMechanicsBoundary:
        values = (
            self.graph_construction_completed_questions,
            self.questions_with_estimable_graph,
            self.synthesis_attempted_groups,
            self.synthesis_completed_groups,
            self.questions_with_completed_synthesis,
        )
        if self.status is SynthesisMechanicsStatus.NOT_APPLICABLE:
            if any(value is not None for value in values):
                raise ValueError("evidence_ledger_not_applicable_synthesis_counts_present")
            return self
        if any(value is None or value < 0 for value in values):
            raise ValueError("evidence_ledger_synthesis_counts_missing_or_negative")
        estimable = int(self.questions_with_estimable_graph or 0)
        attempted = int(self.synthesis_attempted_groups or 0)
        completed = int(self.synthesis_completed_groups or 0)
        completed_questions = int(self.questions_with_completed_synthesis or 0)
        if completed > attempted or completed_questions > completed:
            raise ValueError("evidence_ledger_synthesis_count_order_invalid")
        expected = (
            SynthesisMechanicsStatus.GRAPH_BUILT_WITH_ZERO_ESTIMABLE_EFFECTS
            if estimable == 0
            else (
                SynthesisMechanicsStatus.COMPLETED_MECHANICS_ONLY
                if completed > 0
                else SynthesisMechanicsStatus.ESTIMABLE_GRAPH_WITHOUT_COMPLETED_SYNTHESIS
            )
        )
        if self.status is not expected:
            raise ValueError("evidence_ledger_synthesis_status_count_mismatch")
        return self


class AuthorityBoundary(ContractModel):
    authority_kind: AuthorityKind
    authorized_empirical_scope: AuthorizedEmpiricalScope
    real_world_effectiveness_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    scientific_synthesis_accuracy_authority: Literal[False] = False
    release_risk_calibration_eligible: Literal[False] = False
    adaptive_policy_effectiveness_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_scope(self) -> AuthorityBoundary:
        if self.authorized_empirical_scope is not _AUTHORITY_SCOPE_BY_KIND[self.authority_kind]:
            raise ValueError("evidence_ledger_authority_scope_kind_mismatch")
        return self


class EvidenceBoundaryRecord(ContractModel):
    record_version: Literal["evidence-boundary-record-v1"] = RECORD_VERSION
    record_id: str
    evidence_class: EvidenceClass
    scientific_role: str
    registered_artifacts: Annotated[list[RegisteredArtifact], Field(min_length=1)]
    validation: ValidationReceipt
    runtime_completion: RuntimeCompletionBoundary
    source_access: SourceAccessBoundary
    label_access: LabelAccessBoundary
    independence: IndependenceBoundary
    typed_effect_yield: TypedEffectYieldBoundary
    synthesis_mechanics: SynthesisMechanicsBoundary
    authority: AuthorityBoundary
    limitations: Annotated[list[str], Field(min_length=1)]
    record_sha256: str

    @field_validator("registered_artifacts")
    @classmethod
    def validate_artifact_order(cls, value: list[RegisteredArtifact]) -> list[RegisteredArtifact]:
        paths = [item.path for item in value]
        if paths != sorted(set(paths)):
            raise ValueError("evidence_ledger_artifacts_not_sorted_unique")
        return value

    @field_validator("limitations")
    @classmethod
    def validate_limitations(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item for item in value):
            raise ValueError("evidence_ledger_limitations_not_sorted_unique")
        return value

    @field_validator("record_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("evidence_ledger_record_hash_invalid")
        return value

    @model_validator(mode="after")
    def validate_boundary(self) -> EvidenceBoundaryRecord:
        if (
            self.authority.authority_kind
            is not _AUTHORITY_KIND_BY_EVIDENCE_CLASS[self.evidence_class]
        ):
            raise ValueError("evidence_ledger_authority_kind_class_mismatch")
        if self.runtime_completion.state != _RUNTIME_STATE_BY_EVIDENCE_CLASS[self.evidence_class]:
            raise ValueError("evidence_ledger_runtime_state_class_mismatch")
        if self.record_id not in _VALIDATION_DEPTH_BY_RECORD_ID:
            raise ValueError("evidence_ledger_record_id_unknown")
        if self.validation.depth is not _VALIDATION_DEPTH_BY_RECORD_ID[self.record_id]:
            raise ValueError("evidence_ledger_validation_depth_class_mismatch")
        expected_self_hash = self.evidence_class is not EvidenceClass.CONTRACT_ONLY
        if (
            self.validation.self_hash_validated is not expected_self_hash
            or not self.validation.current_source_lineage_validated
        ):
            raise ValueError("evidence_ledger_validation_receipt_class_mismatch")
        if (
            self.source_access.source_payload_state
            is not _SOURCE_STATE_BY_EVIDENCE_CLASS[self.evidence_class]
        ):
            raise ValueError("evidence_ledger_source_state_class_mismatch")
        if self.label_access.label_state is not _LABEL_STATE_BY_EVIDENCE_CLASS[self.evidence_class]:
            raise ValueError("evidence_ledger_label_state_class_mismatch")
        expected_raw_source_access = self.evidence_class in {
            EvidenceClass.REAL_INCOMPLETE_SOURCE_EXECUTION,
            EvidenceClass.REAL_SOURCE_PREFLIGHT_BLOCKED,
        }
        if (
            self.source_access.raw_source_payload_opened_by_ledger is not expected_raw_source_access
            or self.source_access.aggregate_or_mechanics_artifact_contains_raw_article_text
            is not expected_raw_source_access
        ):
            raise ValueError("evidence_ledger_raw_source_access_class_mismatch")
        expected_typed_not_applicable = self.evidence_class in {
            EvidenceClass.SIMULATED,
            EvidenceClass.CONTRACT_ONLY,
            EvidenceClass.REAL_RETROSPECTIVE_NONPRISTINE,
            EvidenceClass.REAL_SOURCE_PREFLIGHT_BLOCKED,
        }
        if (self.typed_effect_yield.status is TypedEffectStatus.NOT_APPLICABLE) is not (
            expected_typed_not_applicable
        ):
            raise ValueError("evidence_ledger_typed_status_class_mismatch")
        expected_synthesis_not_applicable = (
            self.evidence_class is not EvidenceClass.REAL_LABEL_BLIND_MECHANICS
        )
        if (
            self.synthesis_mechanics.status is SynthesisMechanicsStatus.NOT_APPLICABLE
        ) is not expected_synthesis_not_applicable:
            raise ValueError("evidence_ledger_synthesis_status_class_mismatch")
        if self.evidence_class is EvidenceClass.SIMULATED:
            if (
                self.source_access.source_payload_state
                is not SourcePayloadState.SYNTHETIC_NOT_APPLICABLE
                or self.label_access.label_state is not LabelState.SYNTHETIC_ORACLE_ONLY
                or self.authority.authority_kind is not AuthorityKind.SIMULATION_BEHAVIOR_ONLY
            ):
                raise ValueError("evidence_ledger_simulation_boundary_invalid")
        elif self.evidence_class is EvidenceClass.CONTRACT_ONLY:
            if (
                self.validation.depth is not ValidationDepth.CONTRACT_FINGERPRINT_ONLY
                or self.source_access.source_payload_state
                is not SourcePayloadState.UNOPENED_CONTRACT_ONLY
                or self.label_access.label_state is not LabelState.NO_LABELS_CONTRACT_ONLY
                or self.authority.authority_kind is not AuthorityKind.NONE_CONTRACT_ONLY
            ):
                raise ValueError("evidence_ledger_contract_only_boundary_invalid")
        elif self.evidence_class in {
            EvidenceClass.REAL_LABEL_BLIND_MECHANICS,
            EvidenceClass.REAL_INCOMPLETE_SOURCE_EXECUTION,
            EvidenceClass.REAL_SOURCE_PREFLIGHT_BLOCKED,
        }:
            if (
                self.label_access.label_state is not LabelState.REFERENCE_FIELDS_EXPLICITLY_UNOPENED
                or self.authority.authority_kind
                not in {
                    AuthorityKind.REAL_EXECUTION_YIELD_ONLY,
                    AuthorityKind.REAL_OFFLINE_PREFLIGHT_BLOCKER_ONLY,
                }
            ):
                raise ValueError("evidence_ledger_mechanics_boundary_invalid")
            if self.evidence_class is EvidenceClass.REAL_INCOMPLETE_SOURCE_EXECUTION and (
                self.runtime_completion.state != "packet_roster_frozen_failed_smoke_not_finalized"
            ):
                raise ValueError("evidence_ledger_incomplete_execution_promoted")
            if self.evidence_class is EvidenceClass.REAL_SOURCE_PREFLIGHT_BLOCKED and (
                self.runtime_completion.state
                != "pre_call_blocked_zero_provider_calls_not_execution"
                or self.authority.authority_kind
                is not AuthorityKind.REAL_OFFLINE_PREFLIGHT_BLOCKER_ONLY
                or self.typed_effect_yield.status is not TypedEffectStatus.NOT_APPLICABLE
            ):
                raise ValueError("evidence_ledger_pre_call_blocker_promoted_to_execution")
        elif (
            self.label_access.label_state
            is not LabelState.BENCHMARK_LABELS_PREVIOUSLY_OPENED_AGGREGATES_ONLY
            or self.authority.authority_kind
            is not AuthorityKind.RETROSPECTIVE_MATCHED_SUBSET_METRIC_ONLY
        ):
            raise ValueError("evidence_ledger_retrospective_boundary_invalid")
        payload = self.model_dump(mode="json", exclude={"record_sha256"})
        if hash_canonical(payload) != self.record_sha256:
            raise ValueError("evidence_ledger_record_hash_mismatch")
        return self


def _strongest_authorized_scope(
    records: list[EvidenceBoundaryRecord],
) -> AuthorizedEmpiricalScope:
    return max(
        (record.authority.authorized_empirical_scope for record in records),
        key=_SCOPE_PRIORITY.__getitem__,
        default=AuthorizedEmpiricalScope.NONE,
    )


class LedgerDecisionBoundary(ContractModel):
    real_data_records: Annotated[int, Field(ge=0)]
    simulated_records: Annotated[int, Field(ge=0)]
    contract_only_records: Annotated[int, Field(ge=0)]
    raw_source_payloads_opened_by_ledger: Annotated[int, Field(ge=0)]
    upstream_benchmark_labels_previously_opened: bool
    human_expert_adjudication_present: Literal[False] = False
    complete_independent_human_adjudicated_questions: Literal[0] = 0
    total_runtime_contract_typed_publications: Annotated[int, Field(ge=0)]
    total_release_grade_estimable_publications: Annotated[int, Field(ge=0)]
    any_completed_synthesis_mechanics: bool
    release_risk_calibration_eligible: Literal[False] = False
    adaptive_policy_effectiveness_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    strongest_authorized_real_empirical_scope: AuthorizedEmpiricalScope
    next_required_authority_gate: str

    @model_validator(mode="after")
    def validate_next_gate(self) -> LedgerDecisionBoundary:
        if self.next_required_authority_gate != NEXT_REQUIRED_AUTHORITY_GATE:
            raise ValueError("evidence_ledger_next_authority_gate_mismatch")
        return self


class EvidenceBoundaryLedgerV1(ContractModel):
    ledger_version: Literal["evidence-boundary-ledger-v1"] = LEDGER_VERSION
    status: Literal["validated_fail_closed_evidence_boundary"] = (
        "validated_fail_closed_evidence_boundary"
    )
    registered_input_set_sha256: str
    ledger_implementation_sha256: str
    ledger_implementation_file_sha256s: dict[str, str]
    question_evaluation_pipeline_sha256: str
    records: Annotated[list[EvidenceBoundaryRecord], Field(min_length=6, max_length=7)]
    decision_boundary: LedgerDecisionBoundary
    prohibited_inferences: Annotated[list[str], Field(min_length=5)]
    ledger_sha256: str

    @field_validator(
        "registered_input_set_sha256",
        "ledger_implementation_sha256",
        "question_evaluation_pipeline_sha256",
        "ledger_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"evidence_ledger_hash_invalid:{info.field_name}")
        return value

    @field_validator("ledger_implementation_file_sha256s")
    @classmethod
    def validate_implementation_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        if value != dict(sorted(value.items())) or any(
            not SHA256_RE.fullmatch(item) for item in value.values()
        ):
            raise ValueError("evidence_ledger_implementation_hashes_invalid")
        return value

    @field_validator("records")
    @classmethod
    def validate_record_order(
        cls, value: list[EvidenceBoundaryRecord]
    ) -> list[EvidenceBoundaryRecord]:
        ids = [item.record_id for item in value]
        if ids != sorted(set(ids)):
            raise ValueError("evidence_ledger_records_not_sorted_unique")
        record_set = frozenset(ids)
        if record_set not in {
            frozenset(_REQUIRED_RECORD_IDS),
            frozenset(_REQUIRED_RECORD_IDS | {_OPTIONAL_RECORD_ID}),
        }:
            raise ValueError("evidence_ledger_record_roster_mismatch")
        return value

    @field_validator("prohibited_inferences")
    @classmethod
    def validate_inferences(cls, value: list[str]) -> list[str]:
        if value != PROHIBITED_INFERENCES:
            raise ValueError("evidence_ledger_prohibited_inferences_mismatch")
        return value

    @model_validator(mode="after")
    def validate_ledger(self) -> EvidenceBoundaryLedgerV1:
        counts = Counter(item.evidence_class for item in self.records)
        decision = self.decision_boundary
        if (
            decision.real_data_records
            != counts[EvidenceClass.REAL_RETROSPECTIVE_NONPRISTINE]
            + counts[EvidenceClass.REAL_LABEL_BLIND_MECHANICS]
            + counts[EvidenceClass.REAL_INCOMPLETE_SOURCE_EXECUTION]
            + counts[EvidenceClass.REAL_SOURCE_PREFLIGHT_BLOCKED]
            or decision.simulated_records != counts[EvidenceClass.SIMULATED]
            or decision.contract_only_records != counts[EvidenceClass.CONTRACT_ONLY]
        ):
            raise ValueError("evidence_ledger_class_counts_mismatch")
        typed = sum(
            item.typed_effect_yield.runtime_contract_typed_publications or 0
            for item in self.records
        )
        release_grade = sum(
            item.typed_effect_yield.release_grade_estimable_publications or 0
            for item in self.records
        )
        if (
            decision.total_runtime_contract_typed_publications != typed
            or decision.total_release_grade_estimable_publications != release_grade
            or decision.any_completed_synthesis_mechanics
            != any(
                item.synthesis_mechanics.status is SynthesisMechanicsStatus.COMPLETED_MECHANICS_ONLY
                for item in self.records
            )
            or decision.upstream_benchmark_labels_previously_opened
            != any(
                item.label_access.label_state
                is LabelState.BENCHMARK_LABELS_PREVIOUSLY_OPENED_AGGREGATES_ONLY
                for item in self.records
            )
        ):
            raise ValueError("evidence_ledger_decision_aggregate_mismatch")
        if decision.raw_source_payloads_opened_by_ledger != sum(
            item.source_access.raw_source_payload_opened_by_ledger for item in self.records
        ):
            raise ValueError("evidence_ledger_source_payload_access_count_mismatch")
        if decision.strongest_authorized_real_empirical_scope is not (
            _strongest_authorized_scope(self.records)
        ):
            raise ValueError("evidence_ledger_strongest_scope_mismatch")
        if any(
            item.authority.release_risk_calibration_eligible
            or item.authority.adaptive_policy_effectiveness_authority
            or item.authority.claim_release_authority
            or item.authority.real_world_effectiveness_authority
            for item in self.records
        ):
            raise ValueError("evidence_ledger_unauthorized_effectiveness_promotion")
        payload = self.model_dump(mode="json", exclude={"ledger_sha256"})
        if hash_canonical(payload) != self.ledger_sha256:
            raise ValueError("evidence_ledger_hash_mismatch")
        return self


_REGISTERED_INPUTS: dict[str, tuple[str, str, str]] = {
    "adaptive_stress": (
        "artifacts/diagnostics/adaptive-stress-study-v1.json",
        "66ed36f25ea8b31d93652594ab8046386df5d864f47ba5ddd3e689b53bf75957",
        "1469e597c7db4a186680898e6d79bdcf6490084bec0d8678fbbda1a8e401ad63",
    ),
    "local_benchmark": (
        "artifacts/benchmarks/local-suite-v1/benchmark-report.json",
        "6af096fcbb367cfc7619489162795fd8d3e3c909ca25ed563130ea2567f114c9",
        "880222ea443b64660cb5af137acad160595e91147b2159816d237b31577e7bbf",
    ),
    "metasyn_retrieval": (
        "artifacts/diagnostics/metasyn-retrieval-study-v1.json",
        "b1570cb45e690e8f66a9250aedf2be768411df1e600a52a1ad05ca7024ac2fd5",
        "09265941ffcc7113167186eecf83a19bf66e1d405a18e4ccdc8c3ebde817a8ff",
    ),
    "metasyn_screening": (
        "artifacts/diagnostics/metasyn-screening-study-v1.json",
        "8853ed4578f10eac54755ec38f3b483a4659e5aa4a32e24486bf24295a365e99",
        "b5fdc31a1b1c3b3430b77b904256712095a626f61c94ca2d611a37bc22400322",
    ),
    "passage_v2_execution_bundle": (
        "data/cache/metasyn/passage-hosted-yield-v2/execution-bundle.json",
        "d75f52f2db6e5a826b3cb8b6303cc0b5a0b8dda42cbd3806ac50b0d1fcd2d4da",
        "f87eddcbcbafc778f18ff85c92c0f914a763d242311a859d9d979ded229b4972",
    ),
    "passage_v2_inventory_ledger": (
        "data/cache/metasyn/passage-hosted-yield-v2/inventory-ledger.json",
        "07dff5339a4d19e95f05c421fe434a7efa8b7801ccf8e3c641ab6601d0596024",
        "aeb824df26b2f9efe4677af85f54fc217f41fe1fe043e1f7ebc9ba68de0b6e2f",
    ),
    "passage_v2_packet_result_01": (
        "data/cache/metasyn/passage-hosted-yield-v2/packet-results/packet-row-02-candidate-01.json",
        "313d84e51d7c21307a6ded318e1a9e143671d158d8970e135de1ed817f5047b9",
        "d01b47432004270d19039da147a4b1188a75a59721e5350343b02af967e63d4a",
    ),
    "passage_v2_packet_result_02": (
        "data/cache/metasyn/passage-hosted-yield-v2/packet-results/packet-row-02-candidate-02.json",
        "4c0cb84962ba9361c785680a494f5981d766cad150a88427b044987b0384a47e",
        "7508423c74fbed9d8d00c3fede37fe910da5923951a904b0a542e012414886a3",
    ),
    "passage_v2_packet_result_03": (
        "data/cache/metasyn/passage-hosted-yield-v2/packet-results/packet-row-03-candidate-01.json",
        "520680a453525c9b86b9a75d9251c8497fafe9cbbe4a5f866566c6bad92194c3",
        "5a17c8f4cfd46b884d06c843ea793a8060c69fdfa45c90e9368b4970e58115ec",
    ),
    "passage_v2_packet_roster": (
        "data/cache/metasyn/passage-hosted-yield-v2/packet-roster.json",
        "ae3ee3ed7d08dcd80754bf2134a9330de9ed8a53fceb8364877267ee95972339",
        "97adae1f1a36da26b462fa1b2bec229dd568449870d8a7ae1b5f3418db2c842f",
    ),
    "passage_v2_packet_smoke_attempt": (
        "data/cache/metasyn/passage-hosted-yield-v2/packet-smoke-attempt.json",
        "48497053993bdabd48c4584edd77fa8b0fd57250f4acef3d65f0a8c6f3ce1a49",
        "63bd3b15b5ba1139a98b9d1590fbb9064f433f7b7a5790ad640aa78c78186a41",
    ),
    "passage_v2_stage_05": (
        "data/cache/metasyn/passage-hosted-yield-v2/stage-checkpoints/05-packet-roster-frozen.json",
        "f07672ce8b496d2bae31ec0be500aee9f47a23592b2d949f6437b378dec9e571",
        "10109c5b8e115a5f742ba99c336c26b590f959b5bda505a66e28f8e285955ac3",
    ),
    "metasyn_synthesis_v1_private": (
        "data/cache/metasyn/synthesis-yield-v1/private-report.json",
        "2eb29e7143342d8aa51afcf290a4634092a10975888775d5a75702e36f41839c",
        "9423772efbf1a23c0b6821a0ee201ae656247c0bbe0efcc2adc1299484252bce",
    ),
    "metasyn_synthesis_v1_public": (
        "artifacts/diagnostics/metasyn-synthesis-yield-v1/summary.json",
        "e9a1fb24c42b3ff751b500af264aba443d7387b5a7467d0c4669d2d05b3a749e",
        "29ab78fa11a3c2393431f32507b04a2241e52031098a55ebab58278dceb9afbb",
    ),
    "metasyn_synthesis_v2_private": (
        "data/cache/metasyn/synthesis-yield-v2/private-report.json",
        "9c3c319022ff1caacff0083118bbc4a2dc5363d3b41cd8afe4c17c1b475789a0",
        "cf9199e559b2abead031030309dbf228f6443797266fb80e083ad42deedab21a",
    ),
    "metasyn_synthesis_v2_public": (
        "artifacts/diagnostics/metasyn-synthesis-yield-v2/summary.json",
        "8de0aecd3e545265df36fdffa8948780a7118616f4ea2853e79034306624e6f1",
        "343373f59bc4e4299a45ed8e8880dbd55fc758a17d288608b9813f6ae0e3d858",
    ),
}

_SEMANTIC_HASH_FIELDS = {
    "adaptive_stress": "artifact_sha256",
    "local_benchmark": "report_payload_sha256",
    "metasyn_retrieval": "public_summary_payload_sha256",
    "metasyn_screening": "public_summary_payload_sha256",
    "passage_v2_execution_bundle": "execution_bundle_sha256",
    "passage_v2_inventory_ledger": "ledger_sha256",
    "passage_v2_packet_result_01": "result_sha256",
    "passage_v2_packet_result_02": "result_sha256",
    "passage_v2_packet_result_03": "result_sha256",
    "passage_v2_packet_roster": "roster_sha256",
    "passage_v2_packet_smoke_attempt": "smoke_sha256",
    "passage_v2_stage_05": "checkpoint_sha256",
    "metasyn_synthesis_v1_private": "report_sha256",
    "metasyn_synthesis_v1_public": "summary_sha256",
    "metasyn_synthesis_v2_private": "report_sha256",
    "metasyn_synthesis_v2_public": "summary_sha256",
}

_IMPLEMENTATION_PATHS = (
    "scripts/run_evidence_boundary_ledger_v1.py",
    "src/literature_multiverse/evidence_boundary_ledger_v1.py",
)


def _required_regular_file(repository_root: Path, relative: str) -> Path:
    root = repository_root.resolve(strict=True)
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise EvidenceBoundaryLedgerError(f"registered_path_invalid:{relative}")
    current = root
    for part in relative_path.parts:
        current /= part
        if current.is_symlink():
            raise EvidenceBoundaryLedgerError(f"registered_path_symlink_forbidden:{relative}")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise EvidenceBoundaryLedgerError(f"registered_artifact_missing:{relative}") from exc
    if not resolved.is_file():
        raise EvidenceBoundaryLedgerError(f"registered_artifact_not_file:{relative}")
    return resolved


_PROTECTED_INPUT_PREFIXES = (
    ("Formatting_Instructions_For_NeurIPS_2026 (2)",),
    ("artifacts", "paper"),
    ("artifacts", "submission"),
    ("docs", "paper"),
    ("paper",),
)


def _optional_input_path(repository_root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise EvidenceBoundaryLedgerError("optional_input_path_not_repository_relative")
    if any(tuple(relative.parts[: len(prefix)]) == prefix for prefix in _PROTECTED_INPUT_PREFIXES):
        raise EvidenceBoundaryLedgerError("optional_input_path_protected")
    return _required_regular_file(repository_root, relative.as_posix())


def _load_registered_json(
    *, repository_root: Path, input_id: str
) -> tuple[dict[str, Any], RegisteredArtifact]:
    try:
        relative, expected_file_hash, expected_semantic_hash = _REGISTERED_INPUTS[input_id]
    except KeyError as exc:
        raise EvidenceBoundaryLedgerError(f"registered_input_unknown:{input_id}") from exc
    path = _required_regular_file(repository_root, relative)
    observed_file_hash = sha256_file(path)
    if observed_file_hash != expected_file_hash:
        raise EvidenceBoundaryLedgerError(f"registered_artifact_file_hash_mismatch:{input_id}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceBoundaryLedgerError(f"registered_artifact_json_invalid:{input_id}") from exc
    if not isinstance(value, dict):
        raise EvidenceBoundaryLedgerError(f"registered_artifact_not_object:{input_id}")
    field = _SEMANTIC_HASH_FIELDS[input_id]
    if value.get(field) != expected_semantic_hash:
        raise EvidenceBoundaryLedgerError(f"registered_artifact_semantic_hash_mismatch:{input_id}")
    return value, RegisteredArtifact(
        path=relative,
        file_sha256=observed_file_hash,
        semantic_hash_field=field,
        semantic_sha256=expected_semantic_hash,
    )


def _verify_self_hash(value: Mapping[str, Any], *, field: str, input_id: str) -> None:
    observed = value.get(field)
    payload = {key: item for key, item in value.items() if key != field}
    if not isinstance(observed, str) or observed != hash_canonical(payload):
        raise EvidenceBoundaryLedgerError(f"artifact_self_hash_mismatch:{input_id}")


def _freeze_record(payload: Mapping[str, Any]) -> EvidenceBoundaryRecord:
    canonical = dict(payload)
    return EvidenceBoundaryRecord.model_validate(
        {**canonical, "record_sha256": hash_canonical(canonical)}
    )


def _not_applicable_typed() -> TypedEffectYieldBoundary:
    return TypedEffectYieldBoundary(
        status=TypedEffectStatus.NOT_APPLICABLE,
        publications_evaluated=None,
        runtime_contract_typed_publications=None,
        release_grade_estimable_publications=None,
        graph_estimates=None,
    )


def _not_applicable_synthesis() -> SynthesisMechanicsBoundary:
    return SynthesisMechanicsBoundary(
        status=SynthesisMechanicsStatus.NOT_APPLICABLE,
        graph_construction_completed_questions=None,
        questions_with_estimable_graph=None,
        synthesis_attempted_groups=None,
        synthesis_completed_groups=None,
        questions_with_completed_synthesis=None,
    )


def _adaptive_stress_record(repository_root: Path) -> EvidenceBoundaryRecord:
    artifact, binding = _load_registered_json(
        repository_root=repository_root, input_id="adaptive_stress"
    )
    validate_adaptive_stress_study_artifact(artifact)
    replayed = build_adaptive_stress_study_artifact(artifact["frozen_config"])
    if replayed != artifact:
        raise EvidenceBoundaryLedgerError("adaptive_stress_external_replay_mismatch")
    independent = int(artifact["summary"]["independent_questions"])
    scenarios = artifact["summary"]["questions_per_scenario"]
    return _freeze_record(
        {
            "record_version": RECORD_VERSION,
            "record_id": "adaptive_stress_simulation_v1",
            "evidence_class": EvidenceClass.SIMULATED,
            "scientific_role": "misspecified adversarial mechanism stress test",
            "registered_artifacts": [binding],
            "validation": ValidationReceipt(
                depth=ValidationDepth.FULL_SYNTHETIC_RECOMPUTATION,
                validator_names=[
                    "build_adaptive_stress_study_artifact",
                    "validate_adaptive_stress_study_artifact",
                ],
                self_hash_validated=True,
                current_source_lineage_validated=True,
            ),
            "runtime_completion": RuntimeCompletionBoundary(
                state="completed_artifact",
                workspace_finalized=None,
                terminal_provider_call_count=0,
                remaining_provider_calls_permitted=None,
                terminal_roster_complete=True,
            ),
            "source_access": SourceAccessBoundary(
                source_payload_state=SourcePayloadState.SYNTHETIC_NOT_APPLICABLE
            ),
            "label_access": LabelAccessBoundary(label_state=LabelState.SYNTHETIC_ORACLE_ONLY),
            "independence": IndependenceBoundary(
                observation_unit="independent_complete_simulated_question",
                observed_unit_count=independent,
                counts_by_partition=dict(sorted(scenarios.items())),
                uncertainty_resampling_unit="complete simulated question",
                repeated_units_across_artifacts=False,
            ),
            "typed_effect_yield": _not_applicable_typed(),
            "synthesis_mechanics": _not_applicable_synthesis(),
            "authority": AuthorityBoundary(
                authority_kind=AuthorityKind.SIMULATION_BEHAVIOR_ONLY,
                authorized_empirical_scope=AuthorizedEmpiricalScope.NONE,
            ),
            "limitations": sorted(
                {
                    "All evidence values, reviewer outcomes, costs, and errors are synthetic.",
                    (
                        "The stress test cannot estimate real released-claim error or "
                        "human efficiency."
                    ),
                    "Simulation operating points are not release-risk calibration.",
                }
            ),
        }
    )


def _validate_retrieval_summary(value: Mapping[str, Any], *, repository_root: Path) -> None:
    _verify_self_hash(value, field="public_summary_payload_sha256", input_id="metasyn_retrieval")
    if (
        value.get("status") != "complete_retrospective_nonpristine"
        or value.get("contains_question_text") is not False
        or value.get("contains_article_text") is not False
        or value.get("contains_per_question_or_per_article_identifiers") is not False
        or value.get("provider_calls") != 0
        or value.get("network_calls") != 0
        or value.get("selection_protocol", {}).get("official_test_evaluated") is not False
        or value.get("access_boundary", {}).get("pristine_final_holdout_eligible") is not False
    ):
        raise EvidenceBoundaryLedgerError("metasyn_retrieval_boundary_invalid")
    source_hashes = value.get("lineage", {}).get("source_code_sha256s")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise EvidenceBoundaryLedgerError("metasyn_retrieval_source_lineage_missing")
    for relative, expected in sorted(source_hashes.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise EvidenceBoundaryLedgerError("metasyn_retrieval_source_lineage_invalid")
        path = _required_regular_file(repository_root, relative)
        if sha256_file(path) != expected:
            raise EvidenceBoundaryLedgerError(
                f"metasyn_retrieval_source_lineage_mismatch:{relative}"
            )


def _local_benchmark_record(
    repository_root: Path,
    retrieval: Mapping[str, Any],
    retrieval_binding: RegisteredArtifact,
) -> EvidenceBoundaryRecord:
    report, report_binding = _load_registered_json(
        repository_root=repository_root, input_id="local_benchmark"
    )
    screening, screening_binding = _load_registered_json(
        repository_root=repository_root, input_id="metasyn_screening"
    )
    _verify_self_hash(report, field="report_payload_sha256", input_id="local_benchmark")
    _verify_self_hash(
        screening,
        field="public_summary_payload_sha256",
        input_id="metasyn_screening",
    )
    try:
        _validate_local_suite(report, root=repository_root)
    except PublicArtifactValidationError as exc:
        raise EvidenceBoundaryLedgerError("local_benchmark_external_replay_failed") from exc
    if report.get("status") != "complete" or report.get("network_calls") != 0:
        raise EvidenceBoundaryLedgerError("local_benchmark_completion_boundary_invalid")
    access = report.get("label_access", {}).get("metasyn", {}).get("access_state", {})
    if not isinstance(access, Mapping) or any(
        not isinstance(state, Mapping)
        or state.get("labels_previously_opened") is not True
        or state.get("pristine_final_holdout_eligible") is not False
        for state in access.values()
    ):
        raise EvidenceBoundaryLedgerError("local_benchmark_label_boundary_invalid")
    development = int(retrieval["dataset_boundary"]["development_reviews"])
    calibration = int(retrieval["dataset_boundary"]["calibration_reviews"])
    return _freeze_record(
        {
            "record_version": RECORD_VERSION,
            "record_id": "local_benchmark_suite_v1",
            "evidence_class": EvidenceClass.REAL_RETROSPECTIVE_NONPRISTINE,
            "scientific_role": "retrospective matched-subset retrieval and screening survival",
            "registered_artifacts": sorted(
                [report_binding, retrieval_binding, screening_binding], key=lambda row: row.path
            ),
            "validation": ValidationReceipt(
                depth=ValidationDepth.AGGREGATE_CROSS_ARTIFACT_REPLAY,
                validator_names=[
                    "_validate_local_suite",
                    "canonical_self_hash_replay",
                ],
                self_hash_validated=True,
                current_source_lineage_validated=True,
            ),
            "runtime_completion": RuntimeCompletionBoundary(
                state="completed_artifact",
                workspace_finalized=None,
                terminal_provider_call_count=0,
                remaining_provider_calls_permitted=None,
                terminal_roster_complete=True,
            ),
            "source_access": SourceAccessBoundary(
                source_payload_state=SourcePayloadState.OPENED_UPSTREAM_AGGREGATE_ONLY_HERE
            ),
            "label_access": LabelAccessBoundary(
                label_state=(LabelState.BENCHMARK_LABELS_PREVIOUSLY_OPENED_AGGREGATES_ONLY)
            ),
            "independence": IndependenceBoundary(
                observation_unit="MetaSyn source-review question",
                observed_unit_count=development + calibration,
                counts_by_partition={
                    "calibration": calibration,
                    "development": development,
                },
                uncertainty_resampling_unit="pre-split MetaSyn review component",
                repeated_units_across_artifacts=True,
            ),
            "typed_effect_yield": _not_applicable_typed(),
            "synthesis_mechanics": _not_applicable_synthesis(),
            "authority": AuthorityBoundary(
                authority_kind=AuthorityKind.RETROSPECTIVE_MATCHED_SUBSET_METRIC_ONLY,
                authorized_empirical_scope=(
                    AuthorizedEmpiricalScope.RETROSPECTIVE_MATCHED_SUBSET_METRICS_ONLY
                ),
            ),
            "limitations": sorted(
                {
                    "Development and calibration labels were previously opened.",
                    (
                        "Matched-paper identifiers are not an exhaustive scientifically "
                        "eligible corpus."
                    ),
                    (
                        "The same questions feed retrieval and screening metrics and must "
                        "not be double-counted."
                    ),
                    (
                        "This is neither a pristine holdout nor an end-to-end "
                        "claim-verification result."
                    ),
                }
            ),
        }
    )


def _retrieval_record(
    repository_root: Path,
) -> tuple[EvidenceBoundaryRecord, dict[str, Any], RegisteredArtifact]:
    summary, binding = _load_registered_json(
        repository_root=repository_root, input_id="metasyn_retrieval"
    )
    _validate_retrieval_summary(summary, repository_root=repository_root)
    development = int(summary["dataset_boundary"]["development_reviews"])
    calibration = int(summary["dataset_boundary"]["calibration_reviews"])
    record = _freeze_record(
        {
            "record_version": RECORD_VERSION,
            "record_id": "metasyn_retrieval_study_v1",
            "evidence_class": EvidenceClass.REAL_RETROSPECTIVE_NONPRISTINE,
            "scientific_role": "retrospective matched-subset lexical Recall@200",
            "registered_artifacts": [binding],
            "validation": ValidationReceipt(
                depth=(ValidationDepth.AGGREGATE_SELF_HASH_AND_CURRENT_SOURCE_LINEAGE),
                validator_names=[
                    "canonical_self_hash_replay",
                    "registered_artifact_identity_check",
                    "source_code_lineage_rehash",
                ],
                self_hash_validated=True,
                current_source_lineage_validated=True,
            ),
            "runtime_completion": RuntimeCompletionBoundary(
                state="completed_artifact",
                workspace_finalized=None,
                terminal_provider_call_count=0,
                remaining_provider_calls_permitted=None,
                terminal_roster_complete=True,
            ),
            "source_access": SourceAccessBoundary(
                source_payload_state=SourcePayloadState.OPENED_UPSTREAM_AGGREGATE_ONLY_HERE
            ),
            "label_access": LabelAccessBoundary(
                label_state=(LabelState.BENCHMARK_LABELS_PREVIOUSLY_OPENED_AGGREGATES_ONLY)
            ),
            "independence": IndependenceBoundary(
                observation_unit="MetaSyn source-review question",
                observed_unit_count=development + calibration,
                counts_by_partition={
                    "calibration": calibration,
                    "development": development,
                },
                uncertainty_resampling_unit="pre-split MetaSyn review component",
                repeated_units_across_artifacts=True,
            ),
            "typed_effect_yield": _not_applicable_typed(),
            "synthesis_mechanics": _not_applicable_synthesis(),
            "authority": AuthorityBoundary(
                authority_kind=AuthorityKind.RETROSPECTIVE_MATCHED_SUBSET_METRIC_ONLY,
                authorized_empirical_scope=(
                    AuthorizedEmpiricalScope.RETROSPECTIVE_MATCHED_SUBSET_METRICS_ONLY
                ),
            ),
            "limitations": sorted(
                {
                    "Calibration evaluated only the development-selected candidate.",
                    "Official test labels were not scored, but all splits were previously opened.",
                    (
                        "Recall is against the released matched-paper subset, not "
                        "exhaustive eligibility."
                    ),
                    (
                        "The aggregate was validated without reopening evaluator labels "
                        "or article payloads."
                    ),
                }
            ),
        }
    )
    return record, summary, binding


def _passage_v2_incomplete_record(repository_root: Path) -> EvidenceBoundaryRecord:
    workspace_relative = Path("data/cache/metasyn/passage-hosted-yield-v2")
    workspace = repository_root / workspace_relative
    loaded: dict[str, tuple[dict[str, Any], RegisteredArtifact]] = {
        input_id: _load_registered_json(repository_root=repository_root, input_id=input_id)
        for input_id in (
            "passage_v2_execution_bundle",
            "passage_v2_inventory_ledger",
            "passage_v2_packet_result_01",
            "passage_v2_packet_result_02",
            "passage_v2_packet_result_03",
            "passage_v2_packet_roster",
            "passage_v2_packet_smoke_attempt",
            "passage_v2_stage_05",
        )
    }
    expected_bundle_sha256 = _REGISTERED_INPUTS["passage_v2_execution_bundle"][2]
    bundle = MetaSynPassageHostedExecutionBundleV2.model_validate(
        loaded["passage_v2_execution_bundle"][0]
    )
    bundle = validate_metasyn_passage_hosted_execution_bundle_v2(
        execution_bundle=bundle,
        repository_root=repository_root,
        external_replay=True,
    )
    if bundle.execution_bundle_sha256 != expected_bundle_sha256:
        raise EvidenceBoundaryLedgerError("passage_v2_bundle_identity_mismatch")

    inventory = InventoryLedgerV2.model_validate(loaded["passage_v2_inventory_ledger"][0])
    replayed_inventory = validate_metasyn_passage_inventory_ledger_v2(
        workspace=workspace, execution_bundle=bundle
    )
    if replayed_inventory != inventory:
        raise EvidenceBoundaryLedgerError("passage_v2_inventory_external_replay_mismatch")
    roster = PacketRosterV2.model_validate(loaded["passage_v2_packet_roster"][0])
    replayed_roster = validate_metasyn_passage_packet_roster_v2(
        workspace=workspace, execution_bundle=bundle
    )
    if replayed_roster != roster:
        raise EvidenceBoundaryLedgerError("passage_v2_roster_external_replay_mismatch")
    if roster.exact_authorization is None:
        raise EvidenceBoundaryLedgerError("passage_v2_packet_authorization_missing")

    smoke = PacketSmokeReceiptV2.model_validate(loaded["passage_v2_packet_smoke_attempt"][0])
    if (
        smoke.execution_bundle_sha256 != bundle.execution_bundle_sha256
        or smoke.packet_roster_sha256 != roster.roster_sha256
        or smoke.status != "failed_gate"
        or smoke.completed_typed_effect_result_sha256 is not None
        or smoke.remaining_packet_calls_permitted
        or len(smoke.ordered_smoke_request_keys) != 3
    ):
        raise EvidenceBoundaryLedgerError("passage_v2_failed_smoke_boundary_invalid")

    request_by_key = {item.request.request_key: item for item in roster.requests}
    result_input_ids = (
        "passage_v2_packet_result_01",
        "passage_v2_packet_result_02",
        "passage_v2_packet_result_03",
    )
    results: list[PacketCallResultV2] = []
    for input_id, request_key in zip(
        result_input_ids, smoke.ordered_smoke_request_keys, strict=True
    ):
        try:
            packet_request = request_by_key[request_key]
        except KeyError as exc:
            raise EvidenceBoundaryLedgerError("passage_v2_smoke_request_not_in_roster") from exc
        result = PacketCallResultV2.model_validate(loaded[input_id][0])
        intent = freeze_hosted_exact_once_intent(
            execution_bundle_sha256=bundle.execution_bundle_sha256,
            phase="packet",
            source_bearing=True,
            context_binding_sha256=packet_request.packet_request_sha256,
            request=packet_request.request,
        )
        replayed_result = validate_metasyn_passage_packet_result_v2(
            workspace=workspace,
            execution_bundle=bundle,
            packet_request=packet_request,
            intent=intent,
            authorization=roster.exact_authorization,
        )
        if replayed_result != result:
            raise EvidenceBoundaryLedgerError("passage_v2_packet_result_external_replay_mismatch")
        results.append(result)
    if [item.result_sha256 for item in results] != smoke.attempted_result_sha256s or any(
        item.authorizes_typed_effect for item in results
    ):
        raise EvidenceBoundaryLedgerError("passage_v2_smoke_result_join_invalid")

    stage = loaded["passage_v2_stage_05"][0]
    _verify_self_hash(stage, field="checkpoint_sha256", input_id="passage_v2_stage_05")
    status = metasyn_passage_hosted_runtime_status_v2(
        repository_root=repository_root,
        workspace=workspace,
        expected_execution_bundle_sha256=expected_bundle_sha256,
    )
    if (
        status.get("current_stage") != "packet_roster_frozen"
        or status.get("stage_ordinal") != 5
        or status.get("checkpoint_sha256") != stage["checkpoint_sha256"]
        or status.get("claim_release_authority") is not False
    ):
        raise EvidenceBoundaryLedgerError("passage_v2_workspace_stage_mismatch")
    forbidden_advanced_paths = (
        "packet-smoke.json",
        "packet-ledger.json",
        "private-yield-report.json",
        "external-validation-receipt.json",
        "stage-checkpoints/06-packet-smoke-passed.json",
        "stage-checkpoints/06-packet-smoke-not-applicable.json",
        "stage-checkpoints/07-packet-roster-terminal.json",
        "stage-checkpoints/08-finalized.json",
        "stage-checkpoints/09-externally-validated.json",
    )
    if any((workspace / relative).exists() for relative in forbidden_advanced_paths):
        raise EvidenceBoundaryLedgerError("passage_v2_workspace_unexpected_advanced_state")
    packet_result_files = sorted(
        path.name for path in (workspace / "packet-results").glob("*.json")
    )
    if packet_result_files != [
        "packet-row-02-candidate-01.json",
        "packet-row-02-candidate-02.json",
        "packet-row-03-candidate-01.json",
    ]:
        raise EvidenceBoundaryLedgerError("passage_v2_packet_result_roster_changed")
    if (
        inventory.validation_status_counts
        != {"inventory_contract_invalid": 10, "inventory_contract_valid": 22}
        or inventory.packet_authorizing_row_count != 11
        or inventory.authorized_candidate_count != 29
        or roster.request_count != 29
    ):
        raise EvidenceBoundaryLedgerError("passage_v2_inventory_yield_boundary_changed")

    bindings = sorted((binding for _, binding in loaded.values()), key=lambda row: row.path)
    return _freeze_record(
        {
            "record_version": RECORD_VERSION,
            "record_id": "metasyn_passage_runtime_v2_failed_smoke",
            "evidence_class": EvidenceClass.REAL_INCOMPLETE_SOURCE_EXECUTION,
            "scientific_role": (
                "exact-once real-source inventory and packet-smoke execution yield; "
                "failed gate, deliberately not finalized"
            ),
            "registered_artifacts": bindings,
            "validation": ValidationReceipt(
                depth=ValidationDepth.INCOMPLETE_EXACT_ONCE_WORKSPACE_REPLAY,
                validator_names=[
                    "metasyn_passage_hosted_runtime_status_v2",
                    "validate_metasyn_passage_hosted_execution_bundle_v2",
                    "validate_metasyn_passage_inventory_ledger_v2",
                    "validate_metasyn_passage_packet_result_v2",
                    "validate_metasyn_passage_packet_roster_v2",
                ],
                self_hash_validated=True,
                current_source_lineage_validated=True,
            ),
            "runtime_completion": RuntimeCompletionBoundary(
                state="packet_roster_frozen_failed_smoke_not_finalized",
                workspace_finalized=False,
                terminal_provider_call_count=43,
                remaining_provider_calls_permitted=False,
                terminal_roster_complete=False,
            ),
            "source_access": SourceAccessBoundary(
                source_payload_state=(
                    SourcePayloadState.OPENED_UPSTREAM_PRIVATE_MECHANICS_ONLY_HERE
                ),
                raw_source_payload_opened_by_ledger=True,
                aggregate_or_mechanics_artifact_contains_raw_article_text=True,
            ),
            "label_access": LabelAccessBoundary(
                label_state=LabelState.REFERENCE_FIELDS_EXPLICITLY_UNOPENED
            ),
            "independence": IndependenceBoundary(
                observation_unit="terminal exact-once provider call (not a scientific unit)",
                observed_unit_count=43,
                counts_by_partition={
                    "inventory": 32,
                    "packet_smoke": 3,
                    "source_free_preflight": 8,
                },
                uncertainty_resampling_unit="none; fixed execution roster",
                repeated_units_across_artifacts=False,
            ),
            "typed_effect_yield": TypedEffectYieldBoundary(
                status=TypedEffectStatus.ZERO_RUNTIME_TYPED_EFFECTS,
                publications_evaluated=3,
                runtime_contract_typed_publications=0,
                release_grade_estimable_publications=0,
                graph_estimates=0,
            ),
            "synthesis_mechanics": _not_applicable_synthesis(),
            "authority": AuthorityBoundary(
                authority_kind=AuthorityKind.REAL_EXECUTION_YIELD_ONLY,
                authorized_empirical_scope=(
                    AuthorizedEmpiricalScope.REAL_EXECUTION_YIELD_MECHANICS_ONLY
                ),
            ),
            "limitations": sorted(
                {
                    "The failed smoke gate forbids every remaining packet call in this v2 run.",
                    "The workspace is frozen at packet_roster_frozen and is not finalized.",
                    "Three packet attempts are a yield diagnostic, not extraction accuracy.",
                    "Reference conclusions, directions, and official test labels remain unopened.",
                    (
                        "No synthesis, calibration, adaptive-policy, or release result "
                        "follows from this run."
                    ),
                }
            ),
        }
    )


def _optional_v3_pre_call_blocker_record(
    *, repository_root: Path, plan_relative_path: Path
) -> EvidenceBoundaryRecord:
    """Replay an optional v3 blocker without calling it provider execution."""

    path = _optional_input_path(repository_root, plan_relative_path)
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceBoundaryLedgerError("v3_pre_call_blocker_plan_json_invalid") from exc
    if not isinstance(raw, dict):
        raise EvidenceBoundaryLedgerError("v3_pre_call_blocker_plan_not_object")

    # Import lazily: the core ledger remains usable before an optional v3 plan is
    # materialized, while an included plan must pass its exact current replay.
    from literature_multiverse.metasyn_passage_packet_rescue_v3 import (
        MetaSynPassagePacketRescuePlanV3,
        validate_metasyn_passage_packet_rescue_plan_v3,
    )

    plan = MetaSynPassagePacketRescuePlanV3.model_validate(raw)
    plan = validate_metasyn_passage_packet_rescue_plan_v3(
        plan=plan,
        repository_root=repository_root,
        external_replay=True,
    )
    blocker = plan.pre_call_blocker
    if (
        plan.provider_calls_made
        or plan.provider_calls_permitted
        or plan.authorization_created
        or blocker.provider_calls_made != 0
        or blocker.authorization_created
        or blocker.calls_permitted
        or blocker.selected_candidate_count != 3
        or blocker.selected_candidate_v2_reachable_count != 0
        or blocker.numeric_boundary_blocked_candidate_count != 3
        or plan.extraction_accuracy_authority
        or plan.scientific_effectiveness_authority
        or plan.synthesis_input_authority
        or plan.claim_release_authority
    ):
        raise EvidenceBoundaryLedgerError("v3_pre_call_blocker_authority_boundary_invalid")
    binding = RegisteredArtifact(
        path=plan_relative_path.as_posix(),
        file_sha256=sha256_file(path),
        semantic_hash_field="plan_sha256",
        semantic_sha256=plan.plan_sha256,
    )
    return _freeze_record(
        {
            "record_version": RECORD_VERSION,
            "record_id": "metasyn_passage_rescue_v3_pre_call_blocker",
            "evidence_class": EvidenceClass.REAL_SOURCE_PREFLIGHT_BLOCKED,
            "scientific_role": (
                "offline actual-fixture pre-call feasibility blocker; zero provider "
                "calls and explicitly not execution"
            ),
            "registered_artifacts": [binding],
            "validation": ValidationReceipt(
                depth=ValidationDepth.AGGREGATE_CROSS_ARTIFACT_REPLAY,
                validator_names=["validate_metasyn_passage_packet_rescue_plan_v3"],
                self_hash_validated=True,
                current_source_lineage_validated=True,
            ),
            "runtime_completion": RuntimeCompletionBoundary(
                state="pre_call_blocked_zero_provider_calls_not_execution",
                workspace_finalized=None,
                terminal_provider_call_count=0,
                remaining_provider_calls_permitted=False,
                terminal_roster_complete=False,
            ),
            "source_access": SourceAccessBoundary(
                source_payload_state=(
                    SourcePayloadState.OPENED_UPSTREAM_PRIVATE_MECHANICS_ONLY_HERE
                ),
                raw_source_payload_opened_by_ledger=True,
                aggregate_or_mechanics_artifact_contains_raw_article_text=True,
            ),
            "label_access": LabelAccessBoundary(
                label_state=LabelState.REFERENCE_FIELDS_EXPLICITLY_UNOPENED
            ),
            "independence": IndependenceBoundary(
                observation_unit=("selected real-source candidate fixture (not a scientific unit)"),
                observed_unit_count=3,
                counts_by_partition={"offline_pre_call_blocked_candidates": 3},
                uncertainty_resampling_unit="none; deterministic exact-fixture replay",
                repeated_units_across_artifacts=True,
            ),
            "typed_effect_yield": _not_applicable_typed(),
            "synthesis_mechanics": _not_applicable_synthesis(),
            "authority": AuthorityBoundary(
                authority_kind=AuthorityKind.REAL_OFFLINE_PREFLIGHT_BLOCKER_ONLY,
                authorized_empirical_scope=(
                    AuthorizedEmpiricalScope.REAL_OFFLINE_PREFLIGHT_BLOCKER_ONLY
                ),
            ),
            "limitations": sorted(
                {
                    "No provider call was authorized or made; this is not execution yield.",
                    "No typed-effect yield or extraction-accuracy denominator exists.",
                    (
                        "The result diagnoses immutable boundary logic for three "
                        "selected candidates only."
                    ),
                    (
                        "All calibration, adaptive-policy, synthesis, and release "
                        "authorities remain false."
                    ),
                }
            ),
        }
    )


def _typed_boundary(summary: Any) -> TypedEffectYieldBoundary:
    typed = int(summary.runtime_contract_typed_publication_count)
    release_grade = int(summary.release_grade_estimable_publication_count)
    status = (
        TypedEffectStatus.ZERO_RUNTIME_TYPED_EFFECTS
        if typed == 0
        else (
            TypedEffectStatus.RELEASE_GRADE_ESTIMATES_PRESENT
            if release_grade > 0
            else TypedEffectStatus.RUNTIME_TYPED_EFFECTS_WITHOUT_RELEASE_GRADE_ESTIMATES
        )
    )
    return TypedEffectYieldBoundary(
        status=status,
        publications_evaluated=int(summary.publication_count),
        runtime_contract_typed_publications=typed,
        release_grade_estimable_publications=release_grade,
        graph_estimates=int(summary.graph_estimate_count),
    )


def _synthesis_boundary(summary: Any) -> SynthesisMechanicsBoundary:
    estimable = int(summary.questions_with_estimable_graph)
    completed = int(summary.synthesis_completed_group_count)
    status = (
        SynthesisMechanicsStatus.GRAPH_BUILT_WITH_ZERO_ESTIMABLE_EFFECTS
        if estimable == 0
        else (
            SynthesisMechanicsStatus.COMPLETED_MECHANICS_ONLY
            if completed > 0
            else SynthesisMechanicsStatus.ESTIMABLE_GRAPH_WITHOUT_COMPLETED_SYNTHESIS
        )
    )
    return SynthesisMechanicsBoundary(
        status=status,
        graph_construction_completed_questions=int(
            summary.graph_construction_completed_question_count
        ),
        questions_with_estimable_graph=estimable,
        synthesis_attempted_groups=int(summary.synthesis_attempted_group_count),
        synthesis_completed_groups=completed,
        questions_with_completed_synthesis=int(summary.questions_with_completed_synthesis),
    )


def _synthesis_v1_record(repository_root: Path) -> EvidenceBoundaryRecord:
    private_raw, private_binding = _load_registered_json(
        repository_root=repository_root, input_id="metasyn_synthesis_v1_private"
    )
    public_raw, public_binding = _load_registered_json(
        repository_root=repository_root, input_id="metasyn_synthesis_v1_public"
    )
    private = MetaSynSynthesisYieldReportV1.model_validate(private_raw)
    public = MetaSynSynthesisYieldPublicSummaryV1.model_validate(public_raw)
    validate_metasyn_synthesis_yield_public_summary(summary=public, report=private)
    require_pipeline_fingerprint_match(
        expected=private.evaluation_pipeline_fingerprint,
        root=repository_root,
    )
    return _freeze_record(
        {
            "record_version": RECORD_VERSION,
            "record_id": "metasyn_synthesis_yield_v1",
            "evidence_class": EvidenceClass.REAL_LABEL_BLIND_MECHANICS,
            "scientific_role": "label-blind typed-graph and synthesis execution yield",
            "registered_artifacts": sorted(
                [private_binding, public_binding], key=lambda row: row.path
            ),
            "validation": ValidationReceipt(
                depth=ValidationDepth.PRIVATE_TO_PUBLIC_EXACT_REPLAY,
                validator_names=[
                    "require_pipeline_fingerprint_match",
                    "validate_metasyn_synthesis_yield_public_summary",
                ],
                self_hash_validated=True,
                current_source_lineage_validated=True,
            ),
            "runtime_completion": RuntimeCompletionBoundary(
                state="finalized_mechanics_report",
                workspace_finalized=True,
                terminal_provider_call_count=0,
                remaining_provider_calls_permitted=False,
                terminal_roster_complete=True,
            ),
            "source_access": SourceAccessBoundary(
                source_payload_state=(
                    SourcePayloadState.OPENED_UPSTREAM_PRIVATE_MECHANICS_ONLY_HERE
                )
            ),
            "label_access": LabelAccessBoundary(
                label_state=LabelState.REFERENCE_FIELDS_EXPLICITLY_UNOPENED
            ),
            "independence": IndependenceBoundary(
                observation_unit="selected MetaSyn review question / independence component",
                observed_unit_count=int(public.question_count),
                counts_by_partition={"calibration_mechanics": int(public.question_count)},
                uncertainty_resampling_unit=(
                    "none; execution-yield census of frozen 10-question roster"
                ),
                repeated_units_across_artifacts=True,
            ),
            "typed_effect_yield": _typed_boundary(public),
            "synthesis_mechanics": _synthesis_boundary(public),
            "authority": AuthorityBoundary(
                authority_kind=AuthorityKind.REAL_EXECUTION_YIELD_ONLY,
                authorized_empirical_scope=(
                    AuthorizedEmpiricalScope.REAL_EXECUTION_YIELD_MECHANICS_ONLY
                ),
            ),
            "limitations": sorted(set(public.caveats)),
        }
    )


def _synthesis_v2_record(repository_root: Path) -> EvidenceBoundaryRecord:
    private_raw, private_binding = _load_registered_json(
        repository_root=repository_root, input_id="metasyn_synthesis_v2_private"
    )
    public_raw, public_binding = _load_registered_json(
        repository_root=repository_root, input_id="metasyn_synthesis_v2_public"
    )
    private = MetaSynSynthesisYieldReportV2.model_validate(private_raw)
    public = MetaSynSynthesisYieldPublicSummaryV2.model_validate(public_raw)
    validate_metasyn_synthesis_yield_v2_public_summary(summary=public, report=private)
    require_pipeline_fingerprint_match(
        expected=private.evaluation_pipeline_fingerprint,
        root=repository_root,
    )
    return _freeze_record(
        {
            "record_version": RECORD_VERSION,
            "record_id": "metasyn_synthesis_yield_v2",
            "evidence_class": EvidenceClass.REAL_LABEL_BLIND_MECHANICS,
            "scientific_role": "hosted label-blind typed-graph and synthesis execution yield",
            "registered_artifacts": sorted(
                [private_binding, public_binding], key=lambda row: row.path
            ),
            "validation": ValidationReceipt(
                depth=ValidationDepth.PRIVATE_TO_PUBLIC_EXACT_REPLAY,
                validator_names=[
                    "require_pipeline_fingerprint_match",
                    "validate_metasyn_synthesis_yield_v2_public_summary",
                ],
                self_hash_validated=True,
                current_source_lineage_validated=True,
            ),
            "runtime_completion": RuntimeCompletionBoundary(
                state="finalized_mechanics_report",
                workspace_finalized=True,
                terminal_provider_call_count=0,
                remaining_provider_calls_permitted=False,
                terminal_roster_complete=True,
            ),
            "source_access": SourceAccessBoundary(
                source_payload_state=(
                    SourcePayloadState.OPENED_UPSTREAM_PRIVATE_MECHANICS_ONLY_HERE
                )
            ),
            "label_access": LabelAccessBoundary(
                label_state=LabelState.REFERENCE_FIELDS_EXPLICITLY_UNOPENED
            ),
            "independence": IndependenceBoundary(
                observation_unit="selected MetaSyn review question / independence component",
                observed_unit_count=int(public.question_count),
                counts_by_partition={"calibration_mechanics": int(public.question_count)},
                uncertainty_resampling_unit=(
                    "none; execution-yield census of frozen 10-question roster"
                ),
                repeated_units_across_artifacts=True,
            ),
            "typed_effect_yield": _typed_boundary(public),
            "synthesis_mechanics": _synthesis_boundary(public),
            "authority": AuthorityBoundary(
                authority_kind=AuthorityKind.REAL_EXECUTION_YIELD_ONLY,
                authorized_empirical_scope=(
                    AuthorizedEmpiricalScope.REAL_EXECUTION_YIELD_MECHANICS_ONLY
                ),
            ),
            "limitations": sorted(set(public.caveats)),
        }
    )


def _question_evaluation_contract_record(
    repository_root: Path,
) -> tuple[EvidenceBoundaryRecord, str]:
    pipeline = compute_question_evaluation_pipeline_fingerprint(root=repository_root)
    module_path = "src/literature_multiverse/question_evaluation.py"
    module_hash = sha256_file(_required_regular_file(repository_root, module_path))
    binding = RegisteredArtifact(
        path=module_path,
        file_sha256=module_hash,
        semantic_hash_field="computed_question_evaluation_pipeline_sha256",
        semantic_sha256=pipeline.pipeline_sha256,
    )
    return (
        _freeze_record(
            {
                "record_version": RECORD_VERSION,
                "record_id": "question_policy_evaluation_contract_v7",
                "evidence_class": EvidenceClass.CONTRACT_ONLY,
                "scientific_role": (
                    "executable complete-question policy-replay contract without an "
                    "included adjudicated evaluation artifact"
                ),
                "registered_artifacts": [binding],
                "validation": ValidationReceipt(
                    depth=ValidationDepth.CONTRACT_FINGERPRINT_ONLY,
                    validator_names=["compute_question_evaluation_pipeline_fingerprint"],
                    self_hash_validated=False,
                    current_source_lineage_validated=True,
                ),
                "runtime_completion": RuntimeCompletionBoundary(
                    state="contract_only_not_executed",
                    workspace_finalized=None,
                    terminal_provider_call_count=0,
                    remaining_provider_calls_permitted=None,
                    terminal_roster_complete=False,
                ),
                "source_access": SourceAccessBoundary(
                    source_payload_state=SourcePayloadState.UNOPENED_CONTRACT_ONLY
                ),
                "label_access": LabelAccessBoundary(label_state=LabelState.NO_LABELS_CONTRACT_ONLY),
                "independence": IndependenceBoundary(
                    observation_unit="complete independent claim question required by contract",
                    observed_unit_count=0,
                    counts_by_partition={},
                    uncertainty_resampling_unit="question-clustered only when records exist",
                    repeated_units_across_artifacts=False,
                ),
                "typed_effect_yield": _not_applicable_typed(),
                "synthesis_mechanics": _not_applicable_synthesis(),
                "authority": AuthorityBoundary(
                    authority_kind=AuthorityKind.NONE_CONTRACT_ONLY,
                    authorized_empirical_scope=AuthorizedEmpiricalScope.NONE,
                ),
                "limitations": sorted(
                    {
                        "Executable safeguards are not empirical evidence that a policy works.",
                        "No expert-adjudicated complete-question benchmark artifact is included.",
                        "No realized human-time policy comparison is included.",
                    }
                ),
            }
        ),
        pipeline.pipeline_sha256,
    )


def _implementation_hashes(repository_root: Path) -> dict[str, str]:
    return {
        relative: sha256_file(_required_regular_file(repository_root, relative))
        for relative in sorted(_IMPLEMENTATION_PATHS)
    }


def build_evidence_boundary_ledger(
    *, repository_root: Path, v3_pre_call_blocker_plan: Path | None = None
) -> EvidenceBoundaryLedgerV1:
    """Externally replay the registered evidence set and derive one strict ledger."""

    root = repository_root.resolve(strict=True)
    adaptive = _adaptive_stress_record(root)
    retrieval, retrieval_summary, retrieval_binding = _retrieval_record(root)
    local = _local_benchmark_record(root, retrieval_summary, retrieval_binding)
    passage_v2 = _passage_v2_incomplete_record(root)
    synthesis_v1 = _synthesis_v1_record(root)
    synthesis_v2 = _synthesis_v2_record(root)
    question_contract, question_pipeline_sha256 = _question_evaluation_contract_record(root)
    optional_v3 = (
        _optional_v3_pre_call_blocker_record(
            repository_root=root,
            plan_relative_path=v3_pre_call_blocker_plan,
        )
        if v3_pre_call_blocker_plan is not None
        else None
    )

    # Six required semantic rows plus one optional pre-call blocker row.  The local
    # suite contains the retrieval artifact, so retrieval is reported as the suite's
    # detailed child rather than counted twice in the top-level decision boundary.
    record_roster = [
        adaptive,
        local,
        passage_v2,
        synthesis_v1,
        synthesis_v2,
        question_contract,
    ]
    if optional_v3 is not None:
        record_roster.append(optional_v3)
    records = sorted(record_roster, key=lambda row: row.record_id)
    if retrieval.authority.authority_kind is not (
        AuthorityKind.RETROSPECTIVE_MATCHED_SUBSET_METRIC_ONLY
    ):
        raise EvidenceBoundaryLedgerError("retrieval_child_authority_boundary_invalid")

    class_counts = Counter(row.evidence_class for row in records)
    implementation_hashes = _implementation_hashes(root)
    registered_input_set = {
        key: {
            "path": value[0],
            "file_sha256": value[1],
            "semantic_hash_field": _SEMANTIC_HASH_FIELDS[key],
            "semantic_sha256": value[2],
        }
        for key, value in sorted(_REGISTERED_INPUTS.items())
    }
    if optional_v3 is not None:
        binding = optional_v3.registered_artifacts[0]
        registered_input_set["optional_v3_pre_call_blocker_plan"] = binding.model_dump(mode="json")
    decision = LedgerDecisionBoundary(
        real_data_records=(
            class_counts[EvidenceClass.REAL_RETROSPECTIVE_NONPRISTINE]
            + class_counts[EvidenceClass.REAL_LABEL_BLIND_MECHANICS]
            + class_counts[EvidenceClass.REAL_INCOMPLETE_SOURCE_EXECUTION]
            + class_counts[EvidenceClass.REAL_SOURCE_PREFLIGHT_BLOCKED]
        ),
        simulated_records=class_counts[EvidenceClass.SIMULATED],
        contract_only_records=class_counts[EvidenceClass.CONTRACT_ONLY],
        raw_source_payloads_opened_by_ledger=sum(
            row.source_access.raw_source_payload_opened_by_ledger for row in records
        ),
        upstream_benchmark_labels_previously_opened=True,
        total_runtime_contract_typed_publications=sum(
            row.typed_effect_yield.runtime_contract_typed_publications or 0 for row in records
        ),
        total_release_grade_estimable_publications=sum(
            row.typed_effect_yield.release_grade_estimable_publications or 0 for row in records
        ),
        any_completed_synthesis_mechanics=any(
            row.synthesis_mechanics.status is SynthesisMechanicsStatus.COMPLETED_MECHANICS_ONLY
            for row in records
        ),
        strongest_authorized_real_empirical_scope=_strongest_authorized_scope(records),
        next_required_authority_gate=NEXT_REQUIRED_AUTHORITY_GATE,
    )
    payload: dict[str, Any] = {
        "ledger_version": LEDGER_VERSION,
        "status": "validated_fail_closed_evidence_boundary",
        "registered_input_set_sha256": hash_canonical(registered_input_set),
        "ledger_implementation_sha256": hash_canonical(implementation_hashes),
        "ledger_implementation_file_sha256s": implementation_hashes,
        "question_evaluation_pipeline_sha256": question_pipeline_sha256,
        "records": records,
        "decision_boundary": decision,
        "prohibited_inferences": PROHIBITED_INFERENCES,
    }
    return EvidenceBoundaryLedgerV1.model_validate(
        {**payload, "ledger_sha256": hash_canonical(payload)}
    )


def validate_evidence_boundary_ledger(
    ledger: EvidenceBoundaryLedgerV1 | Mapping[str, Any],
) -> EvidenceBoundaryLedgerV1:
    """Validate self-hashes and cross-record authority invariants without I/O."""

    return EvidenceBoundaryLedgerV1.model_validate(ledger)


__all__ = [
    "LEDGER_VERSION",
    "NEXT_REQUIRED_AUTHORITY_GATE",
    "PROHIBITED_INFERENCES",
    "AuthorityBoundary",
    "AuthorityKind",
    "AuthorizedEmpiricalScope",
    "EvidenceBoundaryLedgerError",
    "EvidenceBoundaryLedgerV1",
    "EvidenceBoundaryRecord",
    "EvidenceClass",
    "IndependenceBoundary",
    "LabelAccessBoundary",
    "LabelState",
    "LedgerDecisionBoundary",
    "RegisteredArtifact",
    "RuntimeCompletionBoundary",
    "SourceAccessBoundary",
    "SourcePayloadState",
    "SynthesisMechanicsBoundary",
    "SynthesisMechanicsStatus",
    "TypedEffectStatus",
    "TypedEffectYieldBoundary",
    "ValidationDepth",
    "ValidationReceipt",
    "build_evidence_boundary_ledger",
    "validate_evidence_boundary_ledger",
]
