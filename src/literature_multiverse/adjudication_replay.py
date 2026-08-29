"""Hash-bound, externally replayable human-adjudication workflow artifacts.

These contracts prove that exact local files agree with the adjudication receipts
used by the decisive trajectory compiler.  The trust registry is operator declared:
it does not provide cryptographic identity, licensure, or expertise verification.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel


class _FrozenExactModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]
AdjudicationDisposition = Literal["no_change", "corrected"]
ReviewerRole = Literal["final_adjudicator", "independent_reviewer", "timekeeper"]


def _strict_relative_path(value: str, field_name: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or value.startswith("./")
        or path.as_posix() != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"adjudication_replay_path_invalid:{field_name}")
    return value


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"adjudication_replay_timestamp_requires_timezone:{field_name}")
    return value


def _self_hash(model: _FrozenExactModel, field_name: str) -> None:
    if hash_canonical(model.model_dump(mode="json", exclude={field_name})) != getattr(
        model, field_name
    ):
        raise ValueError(f"adjudication_replay_self_hash_mismatch:{field_name}")


class ExactAdjudicationArtifactLocatorV1(_FrozenExactModel):
    relative_path: Annotated[str, Field(min_length=1)]
    expected_file_sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _strict_relative_path(value, "artifact")


class OperatorReviewerRosterEntryV1(_FrozenExactModel):
    reviewer_id: Annotated[str, Field(min_length=1)]
    roles: Annotated[list[ReviewerRole], Field(min_length=1)]
    declared_expertise_scope: Annotated[str, Field(min_length=1)]

    @field_validator("roles")
    @classmethod
    def validate_roles(cls, values: list[ReviewerRole]) -> list[ReviewerRole]:
        if values != sorted(set(values)):
            raise ValueError("adjudication_replay_reviewer_roles_not_canonical")
        return values


class AdjudicationOperatorTrustRegistryV1(_FrozenExactModel):
    registry_version: Literal["adjudication-operator-trust-registry-v1"] = (
        "adjudication-operator-trust-registry-v1"
    )
    registry_id: Annotated[str, Field(min_length=1)]
    operator_id: Annotated[str, Field(min_length=1)]
    trust_root_id: Annotated[str, Field(min_length=1)]
    reviewers: Annotated[list[OperatorReviewerRosterEntryV1], Field(min_length=3)]
    trust_semantics: Literal[
        "operator_declared_hash_bound_registry_not_cryptographic_identity_or_expertise_proof"
    ] = "operator_declared_hash_bound_registry_not_cryptographic_identity_or_expertise_proof"
    registry_sha256: Sha256

    @model_validator(mode="after")
    def validate_registry(self) -> AdjudicationOperatorTrustRegistryV1:
        if self.reviewers != sorted(self.reviewers, key=lambda row: row.reviewer_id):
            raise ValueError("adjudication_replay_reviewer_roster_not_canonical")
        if len({row.reviewer_id for row in self.reviewers}) != len(self.reviewers):
            raise ValueError("adjudication_replay_reviewer_duplicate")
        independent = {
            row.reviewer_id for row in self.reviewers if "independent_reviewer" in row.roles
        }
        final = {row.reviewer_id for row in self.reviewers if "final_adjudicator" in row.roles}
        timekeepers = {row.reviewer_id for row in self.reviewers if "timekeeper" in row.roles}
        if len(independent) < 2 or not final or not timekeepers or independent & final:
            raise ValueError("adjudication_replay_operator_role_roster_incomplete")
        _self_hash(self, "registry_sha256")
        return self


def freeze_adjudication_operator_trust_registry_v1(
    *,
    registry_id: str,
    operator_id: str,
    trust_root_id: str,
    reviewers: Sequence[OperatorReviewerRosterEntryV1],
) -> AdjudicationOperatorTrustRegistryV1:
    rows = sorted(reviewers, key=lambda row: row.reviewer_id)
    payload = {
        "registry_version": "adjudication-operator-trust-registry-v1",
        "registry_id": registry_id,
        "operator_id": operator_id,
        "trust_root_id": trust_root_id,
        "reviewers": rows,
        "trust_semantics": (
            "operator_declared_hash_bound_registry_not_cryptographic_identity_or_expertise_proof"
        ),
    }
    return AdjudicationOperatorTrustRegistryV1.model_validate(
        {**payload, "registry_sha256": hash_canonical(payload)}
    )


class AdjudicationProtocolArtifactV1(_FrozenExactModel):
    protocol_version: Literal["decisive-adjudication-protocol-v1"] = (
        "decisive-adjudication-protocol-v1"
    )
    protocol_id: Annotated[str, Field(min_length=1)]
    trust_registry_sha256: Sha256
    minimum_independent_reviewers: Annotated[int, Field(ge=2)] = 2
    distinct_final_adjudicator_required: Literal[True] = True
    blind_to_system_confidence: Literal[True] = True
    blind_to_evaluation_reference_labels: Literal[True] = True
    timing_accounting: Literal["sum_of_individual_active_person_minutes"] = (
        "sum_of_individual_active_person_minutes"
    )
    correction_payload_required_for_every_disposition: Literal[True] = True
    trust_boundary: Literal[
        "workflow_files_are_hash_bound_to_an_operator_registry;_identity_and_expertise_are_not_externally_proven"
    ] = "workflow_files_are_hash_bound_to_an_operator_registry;_identity_and_expertise_are_not_externally_proven"  # noqa: E501


class IndependentReviewerDecisionV1(_FrozenExactModel):
    decision_version: Literal["independent-reviewer-decision-v1"] = (
        "independent-reviewer-decision-v1"
    )
    question_id: Annotated[str, Field(min_length=1)]
    item_id: Annotated[str, Field(min_length=1)]
    reviewer_id: Annotated[str, Field(min_length=1)]
    disposition: AdjudicationDisposition
    submitted_at: datetime
    adjudication_protocol_file_sha256: Sha256
    blinded_to_system_confidence: Literal[True] = True
    blinded_to_evaluation_reference_labels: Literal[True] = True
    decision_rationale: Annotated[str, Field(min_length=1)]

    @field_validator("submitted_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _aware(value, "reviewer_decision_submitted_at")


class ReviewerTimingEvidenceV1(_FrozenExactModel):
    timing_version: Literal["reviewer-timing-evidence-v1"] = "reviewer-timing-evidence-v1"
    question_id: Annotated[str, Field(min_length=1)]
    item_id: Annotated[str, Field(min_length=1)]
    reviewer_id: Annotated[str, Field(min_length=1)]
    observer_id: Annotated[str, Field(min_length=1)]
    started_at: datetime
    completed_at: datetime
    active_person_minutes: Annotated[float, Field(gt=0)]
    timing_source: Literal["operator_recorded_human_timing"] = "operator_recorded_human_timing"
    adjudication_protocol_file_sha256: Sha256

    @field_validator("started_at", "completed_at")
    @classmethod
    def validate_time(cls, value: datetime, info: Any) -> datetime:
        return _aware(value, info.field_name)

    @model_validator(mode="after")
    def validate_timing(self) -> ReviewerTimingEvidenceV1:
        elapsed_minutes = (self.completed_at - self.started_at).total_seconds() / 60.0
        if (
            not math.isfinite(self.active_person_minutes)
            or elapsed_minutes <= 0
            or self.active_person_minutes > elapsed_minutes + 1e-9
        ):
            raise ValueError("adjudication_replay_timing_duration_invalid")
        return self


class ReviewerDecisionDigestV1(_FrozenExactModel):
    reviewer_id: Annotated[str, Field(min_length=1)]
    decision_file_sha256: Sha256


class AdjudicationResolutionArtifactV1(_FrozenExactModel):
    resolution_version: Literal["adjudication-resolution-artifact-v1"] = (
        "adjudication-resolution-artifact-v1"
    )
    question_id: Annotated[str, Field(min_length=1)]
    item_id: Annotated[str, Field(min_length=1)]
    final_adjudicator_id: Annotated[str, Field(min_length=1)]
    independent_decisions: Annotated[list[ReviewerDecisionDigestV1], Field(min_length=2)]
    disposition: AdjudicationDisposition
    corrected_graph_sha256: Sha256 | None = None
    completed_at: datetime
    adjudication_protocol_file_sha256: Sha256
    correction_payload_file_sha256: Sha256
    resolution_rationale: Annotated[str, Field(min_length=1)]

    @field_validator("completed_at")
    @classmethod
    def validate_time(cls, value: datetime) -> datetime:
        return _aware(value, "resolution_completed_at")

    @model_validator(mode="after")
    def validate_resolution(self) -> AdjudicationResolutionArtifactV1:
        if self.independent_decisions != sorted(
            self.independent_decisions, key=lambda row: row.reviewer_id
        ) or len({row.reviewer_id for row in self.independent_decisions}) != len(
            self.independent_decisions
        ):
            raise ValueError("adjudication_replay_resolution_decisions_not_canonical")
        if (self.disposition == "corrected") != (self.corrected_graph_sha256 is not None):
            raise ValueError("adjudication_replay_resolution_correction_mismatch")
        return self


class AdjudicationCorrectionArtifactV1(_FrozenExactModel):
    correction_version: Literal["adjudication-correction-artifact-v1"] = (
        "adjudication-correction-artifact-v1"
    )
    question_id: Annotated[str, Field(min_length=1)]
    item_id: Annotated[str, Field(min_length=1)]
    disposition: AdjudicationDisposition
    corrected_graph_sha256: Sha256 | None = None
    adjudication_protocol_file_sha256: Sha256
    correction_rationale: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def validate_correction(self) -> AdjudicationCorrectionArtifactV1:
        if (self.disposition == "corrected") != (self.corrected_graph_sha256 is not None):
            raise ValueError("adjudication_replay_correction_graph_mismatch")
        return self


class AdjudicationItemReplayV1(_FrozenExactModel):
    item_id: Annotated[str, Field(min_length=1)]
    receipt_sha256s: Annotated[list[Sha256], Field(min_length=1)]
    independent_reviewer_decisions: Annotated[
        list[ExactAdjudicationArtifactLocatorV1], Field(min_length=2)
    ]
    timing_evidence: Annotated[list[ExactAdjudicationArtifactLocatorV1], Field(min_length=3)]
    resolution: ExactAdjudicationArtifactLocatorV1
    correction_payload: ExactAdjudicationArtifactLocatorV1

    @model_validator(mode="after")
    def validate_item(self) -> AdjudicationItemReplayV1:
        if self.receipt_sha256s != sorted(set(self.receipt_sha256s)):
            raise ValueError("adjudication_replay_receipts_not_canonical")
        for field_name in ("independent_reviewer_decisions", "timing_evidence"):
            rows = getattr(self, field_name)
            if rows != sorted(rows, key=lambda row: row.relative_path) or len(
                {row.relative_path for row in rows}
            ) != len(rows):
                raise ValueError(f"adjudication_replay_item_paths_not_canonical:{field_name}")
        paths = [
            *(row.relative_path for row in self.independent_reviewer_decisions),
            *(row.relative_path for row in self.timing_evidence),
            self.resolution.relative_path,
            self.correction_payload.relative_path,
        ]
        if len(paths) != len(set(paths)):
            raise ValueError("adjudication_replay_item_artifact_path_duplicate")
        return self


class AdjudicationReplayPackageV1(_FrozenExactModel):
    package_version: Literal["decisive-adjudication-replay-package-v1"] = (
        "decisive-adjudication-replay-package-v1"
    )
    question_id: Annotated[str, Field(min_length=1)]
    trust_registry: ExactAdjudicationArtifactLocatorV1
    adjudication_protocol: ExactAdjudicationArtifactLocatorV1
    items: Annotated[list[AdjudicationItemReplayV1], Field(min_length=1)]
    package_semantics: Literal[
        "exact_workflow_artifact_replay_against_an_operator_declared_trust_registry_not_external_expertise_proof"
    ] = "exact_workflow_artifact_replay_against_an_operator_declared_trust_registry_not_external_expertise_proof"  # noqa: E501
    package_sha256: Sha256

    @model_validator(mode="after")
    def validate_package(self) -> AdjudicationReplayPackageV1:
        if self.items != sorted(self.items, key=lambda row: row.item_id) or len(
            {row.item_id for row in self.items}
        ) != len(self.items):
            raise ValueError("adjudication_replay_items_not_canonical")
        shared_paths = {
            self.trust_registry.relative_path,
            self.adjudication_protocol.relative_path,
        }
        if len(shared_paths) != 2:
            raise ValueError("adjudication_replay_shared_artifact_path_duplicate")
        item_paths = [
            path
            for item in self.items
            for path in (
                *(row.relative_path for row in item.independent_reviewer_decisions),
                *(row.relative_path for row in item.timing_evidence),
                item.resolution.relative_path,
                item.correction_payload.relative_path,
            )
        ]
        if len(item_paths) != len(set(item_paths)) or shared_paths & set(item_paths):
            raise ValueError("adjudication_replay_package_artifact_path_duplicate")
        _self_hash(self, "package_sha256")
        return self


def freeze_adjudication_replay_package_v1(
    *,
    question_id: str,
    trust_registry: ExactAdjudicationArtifactLocatorV1,
    adjudication_protocol: ExactAdjudicationArtifactLocatorV1,
    items: Sequence[AdjudicationItemReplayV1],
) -> AdjudicationReplayPackageV1:
    rows = sorted(items, key=lambda row: row.item_id)
    payload = {
        "package_version": "decisive-adjudication-replay-package-v1",
        "question_id": question_id,
        "trust_registry": trust_registry,
        "adjudication_protocol": adjudication_protocol,
        "items": rows,
        "package_semantics": (
            "exact_workflow_artifact_replay_against_an_operator_declared_trust_registry_not_external_expertise_proof"
        ),
    }
    return AdjudicationReplayPackageV1.model_validate(
        {**payload, "package_sha256": hash_canonical(payload)}
    )


__all__ = [
    "AdjudicationCorrectionArtifactV1",
    "AdjudicationItemReplayV1",
    "AdjudicationOperatorTrustRegistryV1",
    "AdjudicationProtocolArtifactV1",
    "AdjudicationReplayPackageV1",
    "AdjudicationResolutionArtifactV1",
    "ExactAdjudicationArtifactLocatorV1",
    "IndependentReviewerDecisionV1",
    "OperatorReviewerRosterEntryV1",
    "ReviewerDecisionDigestV1",
    "ReviewerTimingEvidenceV1",
    "freeze_adjudication_operator_trust_registry_v1",
    "freeze_adjudication_replay_package_v1",
]
