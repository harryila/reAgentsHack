"""Strict, shared data contracts for every pipeline workstream."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    field_validator,
    model_validator,
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SLUG_RE = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")


class ContractModel(BaseModel):
    """Base for closed contracts while retaining useful JSON parsing."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )


class CanonicalDirection(StrEnum):
    INCREASE = "increase"
    NO_EFFECT = "no_effect"
    DECREASE = "decrease"
    MIXED = "mixed"
    UNCLEAR = "unclear"


# This closed, code-owned table is intentionally included by source-tree lineage hashing.
DIRECTION_ALIASES: dict[str, CanonicalDirection] = {
    "increase": CanonicalDirection.INCREASE,
    "no_effect": CanonicalDirection.NO_EFFECT,
    "decrease": CanonicalDirection.DECREASE,
    "mixed": CanonicalDirection.MIXED,
    "unclear": CanonicalDirection.UNCLEAR,
    "positive": CanonicalDirection.INCREASE,
    "negative": CanonicalDirection.DECREASE,
    "null": CanonicalDirection.NO_EFFECT,
    "neutral": CanonicalDirection.NO_EFFECT,
    "indeterminate": CanonicalDirection.UNCLEAR,
}


class DirectionNormalizationError(ValueError):
    """Direction could not be mapped by the closed normalization table."""


def normalize_direction(value: object) -> CanonicalDirection:
    """Normalize only the pre-registered aliases; JSON null is always invalid."""

    if isinstance(value, CanonicalDirection):
        return value
    if value is None:
        raise DirectionNormalizationError("direction_null")
    if not isinstance(value, str):
        raise DirectionNormalizationError("direction_not_string")
    normalized = re.sub(r"[\s-]+", "_", value.strip().casefold())
    try:
        return DIRECTION_ALIASES[normalized]
    except KeyError as exc:
        raise DirectionNormalizationError(f"direction_unknown:{normalized}") from exc


class ContentTier(StrEnum):
    FULL_TEXT = "full_text"
    ABSTRACT_ONLY = "abstract_only"
    UNKNOWN = "unknown"


class PublicationStatus(StrEnum):
    PEER_REVIEWED = "peer_reviewed"
    PREPRINT = "preprint"
    UNKNOWN = "unknown"


class ScreenStatus(StrEnum):
    INCLUDED = "included"
    EXCLUDED = "excluded"


class MapStatus(StrEnum):
    NOT_MAPPED = "not_mapped"
    SUCCESS = "success"
    FAILED = "failed"


class GroundingStatus(StrEnum):
    EXACT = "exact"
    MISSING = "missing"
    MISMATCH = "mismatch"
    UNVERIFIABLE = "unverifiable"


def normalize_doi(value: str) -> str:
    """Normalize DOI URL/prefix variants without inventing an identifier."""

    normalized = value.strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    if not normalized or "/" not in normalized:
        raise ValueError("invalid_doi")
    return normalized


def derive_paper_id(*, doc_id: str, doi: str | None = None, pmid: str | None = None) -> str:
    """Create the immutable local identity using DOI → PMID → document priority."""

    if doi:
        return f"doi:{normalize_doi(doi)}"
    if pmid:
        digits = pmid.strip()
        if not digits.isdigit():
            raise ValueError("invalid_pmid")
        return f"pmid:{digits}"
    doc = doc_id.strip()
    if not doc:
        raise ValueError("empty_doc_id")
    return f"doc:{doc}"


def _aware(value: datetime | None, field_name: str) -> datetime | None:
    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise ValueError(f"timezone_required:{field_name}")
    return value


def _sha256(value: str | None, field_name: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if value is None or not SHA256_RE.fullmatch(value):
        raise ValueError(f"invalid_sha256:{field_name}")
    return value


def _relative_path(value: str | None, field_name: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if value is None or not value:
        raise ValueError(f"missing_path:{field_name}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("./"):
        raise ValueError(f"path_must_be_repository_relative:{field_name}")
    return value


class PaperRecord(ContractModel):
    """One terminal ledger row for every identity-deduplicated paper."""

    paper_id: str
    doc_id: str
    alternate_doc_ids: list[str]
    doi: str | None
    pmid: str | None
    title: str
    first_author: str | None
    pub_year: Annotated[int, Field(ge=1000, le=3000)] | None

    source: str
    article_type: str | None
    query_families: list[str]
    search_result_ids: list[str]
    content_tier: ContentTier
    publication_status: PublicationStatus

    screen_status: ScreenStatus
    screen_reason: str | None
    dedupe_cluster_id: str | None
    dedupe_preferred: bool

    map_status: MapStatus
    eligible: bool | None
    exclusion_reason: str | None
    map_result_id: str | None
    raw_artifact_path: str | None
    raw_finding_count: Annotated[int, Field(ge=0)]
    accepted_finding_count: Annotated[int, Field(ge=0)]
    quarantined_finding_count: Annotated[int, Field(ge=0)]
    failure_code: str | None

    dataset_or_cohort_id: str | None
    prompt_version: str | None
    schema_version: str
    config_sha256: str
    cfghash: str | None
    created_at: datetime

    @field_validator("doi")
    @classmethod
    def validate_doi(cls, value: str | None) -> str | None:
        return normalize_doi(value) if value is not None else None

    @field_validator("pmid")
    @classmethod
    def validate_pmid(cls, value: str | None) -> str | None:
        if value is not None and not value.isdigit():
            raise ValueError("invalid_pmid")
        return value

    @field_validator("alternate_doc_ids")
    @classmethod
    def validate_alternates(cls, value: list[str]) -> list[str]:
        if any(not item for item in value):
            raise ValueError("empty_alternate_doc_id")
        if value != sorted(set(value)):
            raise ValueError("alternate_doc_ids_not_sorted_unique")
        return value

    @field_validator("config_sha256")
    @classmethod
    def validate_config_hash(cls, value: str) -> str:
        return str(_sha256(value, "config_sha256"))

    @field_validator("cfghash")
    @classmethod
    def validate_cfg_hash(cls, value: str | None) -> str | None:
        return _sha256(value, "cfghash", nullable=True)

    @field_validator("raw_artifact_path")
    @classmethod
    def validate_raw_path(cls, value: str | None) -> str | None:
        return _relative_path(value, "raw_artifact_path", nullable=True)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _aware(value, "created_at")  # type: ignore[return-value]

    @model_validator(mode="after")
    def validate_identity_and_terminal_state(self) -> PaperRecord:
        expected_id = derive_paper_id(doc_id=self.doc_id, doi=self.doi, pmid=self.pmid)
        if self.paper_id != expected_id:
            raise ValueError(f"paper_id_identity_mismatch:expected={expected_id}")
        if self.doc_id in self.alternate_doc_ids:
            raise ValueError("preferred_doc_id_repeated_as_alternate")

        if self.screen_status is ScreenStatus.EXCLUDED:
            if self.map_status is not MapStatus.NOT_MAPPED:
                raise ValueError("excluded_paper_must_be_not_mapped")
            if self.screen_reason is None:
                raise ValueError("excluded_paper_requires_screen_reason")
            if self.eligible is not None:
                raise ValueError("excluded_paper_eligibility_must_be_null")
            if any(
                (
                    self.raw_finding_count,
                    self.accepted_finding_count,
                    self.quarantined_finding_count,
                )
            ):
                raise ValueError("excluded_paper_counts_must_be_zero")
            if any(
                value is not None
                for value in (
                    self.map_result_id,
                    self.raw_artifact_path,
                    self.failure_code,
                    self.prompt_version,
                    self.cfghash,
                )
            ):
                raise ValueError("excluded_paper_extraction_fields_must_be_null")
            return self

        if self.screen_reason is not None:
            raise ValueError("included_paper_screen_reason_must_be_null")
        if self.map_status is MapStatus.NOT_MAPPED:
            raise ValueError("included_terminal_paper_must_be_mapped")

        if self.map_status is MapStatus.FAILED:
            if self.failure_code is None:
                raise ValueError("failed_map_requires_failure_code")
            if self.eligible is not None:
                raise ValueError("failed_map_eligibility_must_be_null")
            if any(
                (
                    self.raw_finding_count,
                    self.accepted_finding_count,
                    self.quarantined_finding_count,
                )
            ):
                raise ValueError("failed_map_counts_must_be_zero")
            return self

        required = {
            "map_result_id": self.map_result_id,
            "raw_artifact_path": self.raw_artifact_path,
            "prompt_version": self.prompt_version,
            "cfghash": self.cfghash,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(f"successful_map_missing:{','.join(missing)}")
        if self.failure_code is not None:
            raise ValueError("successful_map_failure_code_must_be_null")
        if self.eligible is None:
            raise ValueError("successful_map_requires_eligibility")
        if self.accepted_finding_count + self.quarantined_finding_count != self.raw_finding_count:
            raise ValueError("successful_map_finding_counts_do_not_reconcile")
        if not self.eligible and self.raw_finding_count != 0:
            raise ValueError("ineligible_paper_must_have_zero_findings")
        if self.eligible and self.exclusion_reason is not None:
            raise ValueError("eligible_paper_exclusion_reason_must_be_null")
        if not self.eligible and self.exclusion_reason is None:
            raise ValueError("ineligible_paper_requires_exclusion_reason")
        return self


def make_finding_id(
    *,
    paper_id: str,
    map_result_id: str,
    array_position: int,
    outcome_name: str,
    timepoint_raw: str | None,
    dose_raw: str | None,
    effect_direction: CanonicalDirection | str,
) -> str:
    """Create the within-map stable finding identifier from the normative tuple."""

    if array_position < 0:
        raise ValueError("negative_array_position")
    direction = normalize_direction(effect_direction).value
    identity_text = "|".join(
        [outcome_name, timepoint_raw or "", dose_raw or "", direction]
    )
    hash8 = hashlib.sha256(identity_text.encode("utf-8")).hexdigest()[:8]
    return f"{paper_id}:{map_result_id}:{array_position}:{hash8}"


type ModeratorScalar = str | int | float | bool | None


class FindingRow(ContractModel):
    """One paper x comparison x outcome x timepoint atomic finding."""

    finding_id: str
    paper_id: str
    doc_id: str
    map_result_id: str
    array_position: Annotated[int, Field(ge=0)]
    prompt_version: str
    schema_version: str
    cfghash: str
    grounding_status: GroundingStatus
    evidence_section: str | None
    section_flagged: bool
    normalization_warnings: list[str]

    study_type: str | None
    species: str | None
    model: str | None
    population_state: str | None
    intervention: str | None
    intervention_class: str | None
    comparator: str | None
    dose_raw: str | None
    duration_raw: str | None
    timing_context: str | None
    outcome_name: Annotated[str, Field(min_length=1)]
    outcome_family: str | None
    timepoint_raw: str | None
    effect_direction: CanonicalDirection
    effect_size_raw: str | None
    p_value: Annotated[float, Field(ge=0, le=1)] | None
    significant: bool | None
    sample_size: Annotated[int, Field(ge=1)] | None
    evidence_quote: str | None
    evidence_lines: list[str] | None
    confidence: Annotated[float, Field(ge=0, le=1)] | None
    moderators: dict[str, ModeratorScalar]

    @field_validator("effect_direction", mode="before")
    @classmethod
    def canonicalize_direction(cls, value: object) -> CanonicalDirection:
        return normalize_direction(value)

    @field_validator("cfghash")
    @classmethod
    def validate_cfghash(cls, value: str) -> str:
        return str(_sha256(value, "cfghash"))

    @field_validator("normalization_warnings")
    @classmethod
    def validate_warnings(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("duplicate_normalization_warning")
        return value

    @field_validator("evidence_lines")
    @classmethod
    def validate_evidence_lines(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(not line for line in value):
            raise ValueError("empty_evidence_line")
        return value

    @model_validator(mode="after")
    def validate_finding_identity(self) -> FindingRow:
        expected = make_finding_id(
            paper_id=self.paper_id,
            map_result_id=self.map_result_id,
            array_position=self.array_position,
            outcome_name=self.outcome_name,
            timepoint_raw=self.timepoint_raw,
            dose_raw=self.dose_raw,
            effect_direction=self.effect_direction,
        )
        if self.finding_id != expected:
            raise ValueError(f"finding_id_identity_mismatch:expected={expected}")
        return self


class ArtifactRef(ContractModel):
    path: str
    sha256: str
    bytes: Annotated[int, Field(ge=0)]
    rows: Annotated[int, Field(ge=0)] | None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return str(_relative_path(value, "artifact.path"))

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return str(_sha256(value, "artifact.sha256"))


class UpstreamRef(ContractModel):
    stage: str
    run_id: str
    run_record_path: str
    run_record_sha256: str

    @field_validator("run_record_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return str(_relative_path(value, "upstream.run_record_path"))

    @field_validator("run_record_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return str(_sha256(value, "upstream.run_record_sha256"))


class RunRecord(ContractModel):
    run_record_version: Literal["1"] = "1"
    run_id: str
    question_id: str
    stage: str
    stage_version: str
    status: Literal["complete", "partial", "failed"]
    completion_mode: Literal["normal", "frozen_incomplete"] = "normal"
    checkpoint_sha256: str | None = None
    started_at: datetime
    completed_at: datetime | None
    code_version: str
    command_argv: list[str]
    config_path: str
    config_sha256: str
    prompt_path: str | None
    prompt_sha256: str | None
    schema_path: str | None
    schema_sha256: str | None
    cfghash: str | None
    upstream: list[UpstreamRef]
    inputs: list[ArtifactRef]
    outputs: list[ArtifactRef]
    external_result_ids: dict[str, list[str]]
    counts: dict[str, Annotated[int, Field(ge=0)]]
    warnings: list[str]

    @field_validator("question_id")
    @classmethod
    def validate_question_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
            raise ValueError("invalid_question_id")
        return value

    @field_validator("started_at", "completed_at")
    @classmethod
    def validate_timestamps(cls, value: datetime | None, info: Any) -> datetime | None:
        return _aware(value, info.field_name)

    @field_validator("config_path", "prompt_path", "schema_path")
    @classmethod
    def validate_paths(cls, value: str | None, info: Any) -> str | None:
        return _relative_path(value, info.field_name, nullable=info.field_name != "config_path")

    @field_validator(
        "config_sha256", "prompt_sha256", "schema_sha256", "cfghash", "checkpoint_sha256"
    )
    @classmethod
    def validate_hashes(cls, value: str | None, info: Any) -> str | None:
        return _sha256(
            value,
            info.field_name,
            nullable=info.field_name != "config_sha256",
        )

    @model_validator(mode="after")
    def validate_run_state(self) -> RunRecord:
        if self.status == "complete" and self.completed_at is None:
            raise ValueError("complete_run_requires_completed_at")
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("run_completed_before_started")
        if self.completion_mode == "frozen_incomplete":
            if self.stage != "s5" or self.status != "complete" or self.checkpoint_sha256 is None:
                raise ValueError("invalid_frozen_incomplete_run_state")
        elif self.checkpoint_sha256 is not None:
            raise ValueError("normal_run_checkpoint_sha256_must_be_null")
        if (self.prompt_path is None) != (self.prompt_sha256 is None):
            raise ValueError("prompt_path_hash_pair_required")
        if (self.schema_path is None) != (self.schema_sha256 is None):
            raise ValueError("schema_path_hash_pair_required")
        if self.cfghash is not None and (self.prompt_sha256 is None or self.schema_sha256 is None):
            raise ValueError("cfghash_requires_prompt_and_schema_hashes")
        return self


class RemapResponse(ContractModel):
    finding_id: str
    value: ModeratorScalar


class RemapReconciliationError(ValueError):
    """Echo-back IDs did not form the exact expected one-to-one set."""


def reconcile_remap_responses(
    expected_finding_ids: set[str], responses: list[RemapResponse]
) -> dict[str, ModeratorScalar]:
    """Validate remap echo-back IDs before any side-table is materialized."""

    observed: dict[str, ModeratorScalar] = {}
    duplicates: set[str] = set()
    for response in responses:
        if response.finding_id in observed:
            duplicates.add(response.finding_id)
        observed[response.finding_id] = response.value
    if duplicates:
        raise RemapReconciliationError(f"remap_duplicate_ids:{','.join(sorted(duplicates))}")
    unknown = set(observed) - expected_finding_ids
    if unknown:
        raise RemapReconciliationError(f"remap_unknown_ids:{','.join(sorted(unknown))}")
    missing = expected_finding_ids - set(observed)
    if missing:
        raise RemapReconciliationError(f"remap_missing_ids:{','.join(sorted(missing))}")
    return observed


class VerificationDecision(ContractModel):
    finding_id: str
    model_status: Literal["agree", "disagree", "unverifiable"]
    adjudication: Literal["none", "accept", "reject"]


class VerificationRecord(ContractModel):
    verification_version: Literal["1"] = "1"
    provider: str
    model: str
    prompt_version: str
    prompt_sha256: str
    requested_finding_ids: list[str]
    decisions: list[VerificationDecision]

    @field_validator("prompt_sha256")
    @classmethod
    def validate_prompt_hash(cls, value: str) -> str:
        return str(_sha256(value, "prompt_sha256"))

    @model_validator(mode="after")
    def reconcile_decisions(self) -> VerificationRecord:
        if len(self.requested_finding_ids) != len(set(self.requested_finding_ids)):
            raise ValueError("duplicate_verification_request_id")
        seen = [decision.finding_id for decision in self.decisions]
        if len(seen) != len(set(seen)):
            raise ValueError("duplicate_verification_decision_id")
        unknown = set(seen) - set(self.requested_finding_ids)
        missing = set(self.requested_finding_ids) - set(seen)
        if unknown:
            raise ValueError(f"unknown_verification_decision:{','.join(sorted(unknown))}")
        if missing:
            raise ValueError(f"missing_verification_decision:{','.join(sorted(missing))}")
        return self


class AuditFieldChecks(ContractModel):
    eligibility: bool
    atomicity: bool
    intervention: bool
    comparator: bool
    outcome: bool
    timepoint: bool
    direction: bool
    quote_support: bool


class AuditDecision(ContractModel):
    finding_id: str
    checks: AuditFieldChecks
    adjudication: Literal["none", "accept", "reject"] = "none"
    error_codes: list[str] = Field(default_factory=list)

    @property
    def correct(self) -> bool:
        return all(self.checks.model_dump().values())


class AuditRecord(ContractModel):
    audit_version: Literal["1"] = "1"
    seed: int
    requested_sample_size: Annotated[int, Field(ge=1)]
    sampled_finding_ids: list[str]
    decisions: list[AuditDecision]
    anchor_results: dict[str, bool]
    correct_count: Annotated[int, Field(ge=0)]
    total_count: Annotated[int, Field(ge=0)]
    wilson_interval: tuple[Annotated[float, Field(ge=0, le=1)], Annotated[float, Field(ge=0, le=1)]]
    error_taxonomy: dict[str, Annotated[int, Field(ge=0)]]
    newly_added_eligible_papers: Annotated[int, Field(ge=0)] | None = None
    newly_added_audit_eligible_papers: Annotated[int, Field(ge=0)] | None = None
    sampled_new_distinct_papers: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def reconcile_audit(self) -> AuditRecord:
        if self.sampled_finding_ids != [decision.finding_id for decision in self.decisions]:
            raise ValueError("audit_decision_order_or_identity_mismatch")
        if len(self.sampled_finding_ids) != len(set(self.sampled_finding_ids)):
            raise ValueError("duplicate_audit_finding_id")
        correct = sum(decision.correct for decision in self.decisions)
        if self.total_count != len(self.decisions) or self.correct_count != correct:
            raise ValueError("audit_counts_do_not_reconcile")
        if self.wilson_interval[0] > self.wilson_interval[1]:
            raise ValueError("invalid_wilson_interval")
        return self


class CheckpointBudgets(ContractModel):
    bootstrap_count: Annotated[int, Field(ge=1)]
    permutation_success_count: Annotated[int, Field(ge=1)]
    permutation_max_attempts: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def validate_attempt_budget(self) -> CheckpointBudgets:
        if self.permutation_max_attempts < self.permutation_success_count:
            raise ValueError("permutation_attempt_budget_below_success_budget")
        return self


class CheckpointResult(ContractModel):
    index: Annotated[int, Field(ge=0)]
    status: Literal["success", "guard_failure"]
    result: dict[str, Any] | None
    error_code: str | None

    @model_validator(mode="after")
    def validate_result(self) -> CheckpointResult:
        if self.status == "success" and (self.result is None or self.error_code is not None):
            raise ValueError("invalid_success_checkpoint_result")
        if self.status == "guard_failure" and (self.result is not None or self.error_code is None):
            raise ValueError("invalid_guard_failure_checkpoint_result")
        return self


class CheckpointArtifactHashes(ContractModel):
    descriptive_inputs: str
    descriptive_outputs: str
    residual_inputs: str
    residual_outputs: str
    evidence_gap_inputs: str
    evidence_gap_outputs: str

    @field_validator("*")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return str(_sha256(value, f"checkpoint.artifact_hashes.{info.field_name}"))


class M4SourceCheckpoint(ContractModel):
    checkpoint_version: Literal["1"] = "1"
    checkpoint_status: Literal["running_snapshot"] = "running_snapshot"
    source_run_id: str
    source_started_at: datetime
    checkpointed_at: datetime
    question_id: str
    config_sha256: str
    code_version: str
    cohort_sha256: str
    g3_gate_sha256: str
    input_hashes: dict[str, str]
    seed: int
    registered_budgets: CheckpointBudgets
    completed_bootstrap_indices: list[int]
    completed_permutation_attempt_indices: list[int]
    successful_permutation_indices: list[int]
    bootstrap_results: list[CheckpointResult]
    permutation_results: list[CheckpointResult]
    guard_failures: list[str]
    artifact_hashes: CheckpointArtifactHashes

    @field_validator("source_started_at", "checkpointed_at")
    @classmethod
    def validate_times(cls, value: datetime, info: Any) -> datetime:
        return _aware(value, info.field_name)  # type: ignore[return-value]

    @field_validator("config_sha256", "cohort_sha256", "g3_gate_sha256")
    @classmethod
    def validate_named_hashes(cls, value: str, info: Any) -> str:
        return str(_sha256(value, info.field_name))

    @field_validator("input_hashes")
    @classmethod
    def validate_input_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("checkpoint_input_hashes_empty")
        for name, digest in value.items():
            _sha256(digest, f"input_hashes.{name}")
        return value

    @field_validator(
        "completed_bootstrap_indices",
        "completed_permutation_attempt_indices",
        "successful_permutation_indices",
    )
    @classmethod
    def validate_indices(cls, value: list[int], info: Any) -> list[int]:
        if any(index < 0 for index in value) or value != sorted(set(value)):
            raise ValueError(f"checkpoint_indices_not_sorted_unique:{info.field_name}")
        return value

    @model_validator(mode="after")
    def reconcile_checkpoint(self) -> M4SourceCheckpoint:
        if self.checkpointed_at < self.source_started_at:
            raise ValueError("checkpoint_before_source_start")
        bootstrap_result_indices = sorted(item.index for item in self.bootstrap_results)
        permutation_result_indices = sorted(item.index for item in self.permutation_results)
        if bootstrap_result_indices != self.completed_bootstrap_indices:
            raise ValueError("checkpoint_bootstrap_indices_do_not_reconcile")
        if permutation_result_indices != self.completed_permutation_attempt_indices:
            raise ValueError("checkpoint_permutation_indices_do_not_reconcile")
        successful = sorted(
            item.index for item in self.permutation_results if item.status == "success"
        )
        if successful != self.successful_permutation_indices:
            raise ValueError("checkpoint_success_indices_do_not_reconcile")
        if len(self.completed_bootstrap_indices) > self.registered_budgets.bootstrap_count:
            raise ValueError("checkpoint_bootstrap_budget_exceeded")
        if (
            len(self.completed_permutation_attempt_indices)
            > self.registered_budgets.permutation_max_attempts
        ):
            raise ValueError("checkpoint_permutation_attempt_budget_exceeded")
        return self

    @property
    def genuinely_incomplete(self) -> bool:
        return (
            len(self.completed_bootstrap_indices) < self.registered_budgets.bootstrap_count
            or (
                len(self.successful_permutation_indices)
                < self.registered_budgets.permutation_success_count
                and len(self.completed_permutation_attempt_indices)
                < self.registered_budgets.permutation_max_attempts
            )
        )


class M4CheckpointNotApplicable(ContractModel):
    status: Literal["not_applicable"] = "not_applicable"
    reason: Literal["m4_completed", "m4_not_run"]


def canonical_model_sha256(model: BaseModel) -> str:
    payload = model.model_dump(mode="json", exclude_none=False)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class M4CheckpointFrozenIncomplete(ContractModel):
    status: Literal["frozen_incomplete"] = "frozen_incomplete"
    source_checkpoint_sha256: str
    checkpoint: M4SourceCheckpoint

    @field_validator("source_checkpoint_sha256")
    @classmethod
    def validate_source_hash(cls, value: str) -> str:
        return str(_sha256(value, "source_checkpoint_sha256"))

    @model_validator(mode="after")
    def validate_wrapper(self) -> M4CheckpointFrozenIncomplete:
        if not self.checkpoint.genuinely_incomplete:
            raise ValueError("checkpoint_not_genuinely_incomplete")
        if canonical_model_sha256(self.checkpoint) != self.source_checkpoint_sha256:
            raise ValueError("checkpoint_source_hash_mismatch")
        return self


type M4Checkpoint = Annotated[
    M4CheckpointNotApplicable | M4CheckpointFrozenIncomplete,
    Field(discriminator="status"),
]
M4_CHECKPOINT_ADAPTER = TypeAdapter(M4Checkpoint)


def validate_frozen_s5_completion(
    run: RunRecord, checkpoint: M4Checkpoint, *, m4_gate_status: str
) -> None:
    """Validate cross-artifact invariants unavailable to ``RunRecord`` alone."""

    if run.completion_mode != "frozen_incomplete":
        if isinstance(checkpoint, M4CheckpointFrozenIncomplete):
            raise ValueError("normal_run_cannot_use_frozen_checkpoint")
        return
    if not isinstance(checkpoint, M4CheckpointFrozenIncomplete):
        raise ValueError("frozen_run_requires_frozen_checkpoint_wrapper")
    if run.checkpoint_sha256 != checkpoint.source_checkpoint_sha256:
        raise ValueError("run_checkpoint_hash_mismatch")
    if m4_gate_status != "incomplete":
        raise ValueError("frozen_run_requires_incomplete_m4_gate")


class StageRunHashes(ContractModel):
    s3: str
    s4: str
    s5: str

    @field_validator("*")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return str(_sha256(value, f"stage_run_sha256s.{info.field_name}"))


class NullableStageRunHashes(ContractModel):
    s3: str | None
    s4: str | None
    s5: str | None

    @field_validator("*")
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        return _sha256(value, f"stage_run_sha256s.{info.field_name}", nullable=True)


class EvidenceHashes(ContractModel):
    g3_gate: str
    audit: str
    verification: str
    headline: str
    baseline: str

    @field_validator("*")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return str(_sha256(value, f"evidence_sha256s.{info.field_name}"))


class NullableEvidenceHashes(ContractModel):
    g3_gate: str | None
    audit: str | None
    verification: str | None
    headline: str | None
    baseline: str | None

    @field_validator("*")
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        return _sha256(value, f"evidence_sha256s.{info.field_name}", nullable=True)


class SelectedRelease(ContractModel):
    corpus_role: Literal["v1", "scaled"]
    release_id: str
    primary_grounded_papers: Annotated[int, Field(ge=0)]
    stage_run_sha256s: StageRunHashes
    evidence_sha256s: EvidenceHashes


ScaledAttemptStatus = Literal["selected", "rejected", "incomplete"]
ScaledFailureCode = Literal[
    "scaled_incomplete",
    "scaled_artifact_integrity_failed",
    "scaled_ledger_reconciliation_failed",
    "scaled_trust_or_offline_validation_failed",
]


class ScaledAttempt(ContractModel):
    status: ScaledAttemptStatus
    failure_code: ScaledFailureCode | None
    last_completed_stage: Literal["s3", "s4", "s5"] | None
    candidate_release_id: str | None
    primary_grounded_papers: Annotated[int, Field(ge=0)] | None
    stage_run_sha256s: NullableStageRunHashes
    evidence_sha256s: NullableEvidenceHashes

    @model_validator(mode="after")
    def validate_attempt(self) -> ScaledAttempt:
        stage = self.stage_run_sha256s
        expected_non_null = {
            None: (False, False, False),
            "s3": (True, False, False),
            "s4": (True, True, False),
            "s5": (True, True, True),
        }[self.last_completed_stage]
        actual_non_null = (stage.s3 is not None, stage.s4 is not None, stage.s5 is not None)
        if actual_non_null != expected_non_null:
            raise ValueError("scaled_attempt_stage_hashes_not_contiguous")

        evidence = self.evidence_sha256s
        s4_ready = stage.s4 is not None
        s5_ready = stage.s5 is not None
        if any(
            value is not None
            for value in (evidence.g3_gate, evidence.audit, evidence.verification)
        ) and not s4_ready:
            raise ValueError("scaled_attempt_g3_evidence_requires_s4")
        if (
            any(value is not None for value in (evidence.headline, evidence.baseline))
            and not s5_ready
        ):
            raise ValueError("scaled_attempt_s5_evidence_requires_s5")

        if self.status == "selected":
            if self.failure_code is not None:
                raise ValueError("selected_scaled_attempt_failure_code_must_be_null")
            if self.last_completed_stage != "s5":
                raise ValueError("selected_scaled_attempt_requires_s5")
            if self.candidate_release_id is None or self.primary_grounded_papers is None:
                raise ValueError("selected_scaled_attempt_requires_identity_and_count")
            if any(value is None for value in evidence.model_dump().values()):
                raise ValueError("selected_scaled_attempt_requires_all_evidence_hashes")
        elif self.failure_code is None:
            raise ValueError("unselected_scaled_attempt_requires_failure_code")
        return self


ReleaseDisposition = Literal[
    "v1_frozen",
    "scaled_promoted",
    "v1_retained_scaled_incomplete",
    "v1_retained_scaled_corrupt",
    "v1_retained_scaled_unreconciled",
    "v1_retained_scaled_unvalidated",
]


def render_release_disclosure(
    disposition: ReleaseDisposition,
    *,
    frozen_v1_primary_papers: int,
    selected_primary_papers: int,
) -> str:
    if disposition == "v1_frozen":
        return (
            f"Release: frozen v1 with {frozen_v1_primary_papers} grounded primary papers; "
            "no scaled candidate was promoted."
        )
    if disposition == "scaled_promoted":
        return (
            f"Release: validated scaled corpus with {selected_primary_papers} grounded primary "
            f"papers, superseding frozen v1 with {frozen_v1_primary_papers}."
        )
    reason = {
        "v1_retained_scaled_incomplete": "it did not complete",
        "v1_retained_scaled_corrupt": "artifact integrity failed",
        "v1_retained_scaled_unreconciled": "ledger reconciliation failed",
        "v1_retained_scaled_unvalidated": "trust or offline release validation did not pass",
    }[disposition]
    return (
        f"Release: frozen v1 with {frozen_v1_primary_papers} grounded primary papers; "
        f"the scaled candidate was not promoted because {reason}."
    )


class ReleaseSelection(ContractModel):
    disposition: ReleaseDisposition
    frozen_v1_primary_papers: Annotated[int, Field(ge=0)]
    selected_release: SelectedRelease
    scaled_attempt: ScaledAttempt | None
    rendered_disclosure: str

    @model_validator(mode="after")
    def validate_state_row(self) -> ReleaseSelection:
        if self.disposition == "v1_frozen":
            if self.scaled_attempt is not None or self.selected_release.corpus_role != "v1":
                raise ValueError("invalid_release_state:v1_frozen")
        elif self.disposition == "scaled_promoted":
            attempt = self.scaled_attempt
            if attempt is None or attempt.status != "selected":
                raise ValueError("invalid_release_state:scaled_promoted")
            if self.selected_release.corpus_role != "scaled":
                raise ValueError("scaled_promoted_selected_role_must_be_scaled")
            if (
                self.selected_release.release_id != attempt.candidate_release_id
                or self.selected_release.primary_grounded_papers != attempt.primary_grounded_papers
                or self.selected_release.stage_run_sha256s.model_dump()
                != attempt.stage_run_sha256s.model_dump()
                or self.selected_release.evidence_sha256s.model_dump()
                != attempt.evidence_sha256s.model_dump()
            ):
                raise ValueError("scaled_promoted_selected_attempt_mismatch")
        else:
            expected = {
                "v1_retained_scaled_incomplete": ("incomplete", "scaled_incomplete"),
                "v1_retained_scaled_corrupt": (
                    "rejected",
                    "scaled_artifact_integrity_failed",
                ),
                "v1_retained_scaled_unreconciled": (
                    "rejected",
                    "scaled_ledger_reconciliation_failed",
                ),
                "v1_retained_scaled_unvalidated": (
                    "rejected",
                    "scaled_trust_or_offline_validation_failed",
                ),
            }[self.disposition]
            attempt = self.scaled_attempt
            if (
                attempt is None
                or (attempt.status, attempt.failure_code) != expected
                or self.selected_release.corpus_role != "v1"
            ):
                raise ValueError(f"invalid_release_state:{self.disposition}")
        if self.selected_release.corpus_role == "v1" and (
            self.selected_release.primary_grounded_papers != self.frozen_v1_primary_papers
        ):
            raise ValueError("selected_v1_count_must_equal_frozen_v1_count")
        expected_disclosure = render_release_disclosure(
            self.disposition,
            frozen_v1_primary_papers=self.frozen_v1_primary_papers,
            selected_primary_papers=self.selected_release.primary_grounded_papers,
        )
        if self.rendered_disclosure != expected_disclosure:
            raise ValueError("release_disclosure_mismatch")
        return self


class ReleaseSelectionDocument(RootModel[ReleaseSelection]):
    """Explicit root wrapper for adapters that expect a document model."""
