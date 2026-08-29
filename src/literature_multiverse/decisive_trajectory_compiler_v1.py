"""Compile real transactional verifier runs into decisive evaluation trajectories.

The decisive claim evaluation deliberately consumes a compact, label-blind
``TrajectoryBundleV1``.  Production audit execution, however, is persisted as one or
more append-only transactional workspaces.  This module is the fail-closed bridge
between those two representations.

For every evaluation question it:

* validates each complete transactional workspace while holding its mutation lock;
* binds the exact bytes and canonical identities of every workspace artifact;
* projects certificate-v5 non-condition states and final certificate-v7 condition
  states into production replay states;
* joins completed adjudications to their realized total-person-minutes and exact
  external correction payloads;
* requires repeated observations of an item or prefix to agree on every field visible
  to the retrospective evaluation;
* runs the decisive policy roster over all available states and retains exactly the
  union of policy-visited prefixes (including the exhaustive canonical path); and
* refuses synthetic, simulated, benchmark-replay, uncalibrated, ungrounded, or
  otherwise incomplete sources.

Reference verdicts are neither accepted nor opened here.  The resulting artifacts
remain candidates for the separately sealed decisive evaluation lifecycle; like that
lifecycle, they intentionally carry no scientific-claim or release authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from literature_multiverse.adjudication_replay import (
    AdjudicationCorrectionArtifactV1,
    AdjudicationOperatorTrustRegistryV1,
    AdjudicationProtocolArtifactV1,
    AdjudicationReplayPackageV1,
    AdjudicationResolutionArtifactV1,
    IndependentReviewerDecisionV1,
    ReviewerDecisionDigestV1,
    ReviewerTimingEvidenceV1,
)
from literature_multiverse.certificate import (
    ConditionVerificationCertificateV6,
    FinalConditionVerificationCertificateV7,
    VerificationCertificate,
)
from literature_multiverse.decisive_claim_evaluation_v1 import (
    DecisiveEvaluationConfigV1,
    DecisivePolicyInputProvenanceV1,
    DecisiveSplitManifestV1,
    FitStageReceiptV1,
    QuestionIdentityV1,
    QuestionTrajectoryV1,
    StudySplit,
    TrajectoryBundleV1,
    _freeze_policy_question_v1,
    freeze_decisive_policy_input_provenance_v1,
    freeze_question_trajectory_v1,
    freeze_trajectory_bundle_v1,
    required_policy_roster_v1,
)
from literature_multiverse.decisive_compilation_lineage_v1 import (
    DecisiveCompilationLineageIdentityV1,
    freeze_decisive_compilation_lineage_identity_v1,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.question_evaluation import (
    AuditCostBasis,
    AuditDisposition,
    BenchmarkEvidenceKind,
    QuestionAuditEvent,
    QuestionReplayState,
    freeze_question_audit_event,
    freeze_question_replay_state_from_certificate,
)
from literature_multiverse.sequential_verification import (
    SequentialActiveCostCheckpointResult,
    SequentialResolutionResult,
    SequentialVerificationState,
)
from literature_multiverse.transactional_audit_workspace_v1 import (
    AuditActiveCostCheckpointReceiptV1,
    AuditAdjudicationCostReceiptV1,
    AuditWorkspaceConfigV1,
    AuditWorkspacePointerV1,
    AuditWorkspaceTransactionMarkerV1,
    _load_workspace,
    _workspace_lock,
)

ROSTER_VERSION = "decisive-trajectory-source-roster-v1"
COMPILATION_VERSION = "decisive-trajectory-compilation-v1"
QUESTION_RECEIPT_VERSION = "decisive-question-trajectory-compilation-v1"
WORKSPACE_BINDING_VERSION = "decisive-transactional-workspace-binding-v1"
ARTIFACT_BINDING_VERSION = "decisive-source-artifact-binding-v1"
ADOPTED_REPLAY_VERSION = "decisive-adopted-replay-binding-v1"
CERTIFICATE_SOURCE_BINDING_VERSION = "decisive-verifier-certificate-source-binding-v1"
COMPILER_COMPONENT_VERSION = "decisive-trajectory-compiler-component-v1"

MODULE_PATH = "src/literature_multiverse/decisive_trajectory_compiler_v1.py"
CLI_PATH = "scripts/run_decisive_trajectory_compiler_v1.py"
_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
_COST_TOLERANCE = 1e-9

_NONEMPIRICAL_METADATA_FLAGS = frozenset(
    {
        "diagnostic_only",
        "fixture",
        "fixture_mode",
        "fixture_only",
        "is_fixture",
        "mechanics_only",
        "non_empirical",
        "planted_simulation",
        "simulated",
        "simulation",
        "simulation_only",
        "synthetic",
        "synthetic_only",
        "synthetic_source_only",
        "test_fixture",
    }
)
_NONEMPIRICAL_METADATA_SCOPES = frozenset(
    {
        "artifact_kind",
        "evidence_kind",
        "label_source",
        "provenance_kind",
        "purpose",
        "scientific_role",
        "source_kind",
    }
)
_NONEMPIRICAL_SCOPE_TOKENS = (
    "diagnostic",
    "fixture",
    "mechanics_only",
    "non-empirical",
    "non_empirical",
    "simulation",
    "synthetic",
)

_COMPONENT_PATHS = (
    MODULE_PATH,
    CLI_PATH,
    "src/literature_multiverse/adjudication_replay.py",
    "src/literature_multiverse/decisive_compilation_lineage_v1.py",
    "src/literature_multiverse/decisive_claim_evaluation_v1.py",
    "src/literature_multiverse/question_evaluation.py",
    "src/literature_multiverse/transactional_audit_workspace_v1.py",
    "src/literature_multiverse/sequential_verification.py",
    "src/literature_multiverse/certificate.py",
)


class DecisiveTrajectoryCompilerV1Error(ValueError):
    """A source cannot safely support a real decisive trajectory bundle."""


class _FrozenExactModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]


def _self_hash(model: _FrozenExactModel, field_name: str) -> None:
    expected = hash_canonical(model.model_dump(mode="json", exclude={field_name}))
    if getattr(model, field_name) != expected:
        raise ValueError(f"decisive_trajectory_compiler_self_hash_mismatch:{field_name}")


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
        raise ValueError(f"decisive_trajectory_compiler_path_invalid:{field_name}")
    return value


def _canonical_time(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_timestamp_requires_timezone"
        )
    rendered = value.isoformat()
    return f"{rendered[:-6]}Z" if rendered.endswith("+00:00") else rendered


class TransactionalWorkspaceLocatorV1(_FrozenExactModel):
    locator_version: Literal["decisive-transactional-workspace-locator-v1"] = (
        "decisive-transactional-workspace-locator-v1"
    )
    relative_path: Annotated[str, Field(min_length=1)]
    expected_workspace_config_sha256: Sha256
    expected_terminal_pointer_sha256: Sha256
    locator_sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _strict_relative_path(value, "workspace")

    @model_validator(mode="after")
    def validate_locator(self) -> TransactionalWorkspaceLocatorV1:
        _self_hash(self, "locator_sha256")
        return self


class AdjudicationReplayPackageLocatorV1(_FrozenExactModel):
    locator_version: Literal["decisive-adjudication-package-locator-v1"] = (
        "decisive-adjudication-package-locator-v1"
    )
    relative_path: Annotated[str, Field(min_length=1)]
    expected_file_sha256: Sha256
    expected_package_sha256: Sha256
    locator_sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _strict_relative_path(value, "adjudication_replay_package")

    @model_validator(mode="after")
    def validate_locator(self) -> AdjudicationReplayPackageLocatorV1:
        _self_hash(self, "locator_sha256")
        return self


class ConditionSetSourceBindingV1(_FrozenExactModel):
    binding_version: Literal["decisive-condition-set-source-binding-v1"] = (
        "decisive-condition-set-source-binding-v1"
    )
    certificate_sha256: Sha256
    relative_path: Annotated[str, Field(min_length=1)]
    expected_file_sha256: Sha256
    condition_set_artifact_sha256: Sha256
    binding_sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _strict_relative_path(value, "condition_set_artifact")

    @model_validator(mode="after")
    def validate_binding(self) -> ConditionSetSourceBindingV1:
        _self_hash(self, "binding_sha256")
        return self


class NormalizedConditionSetArtifactV1(_FrozenExactModel):
    """Semantic identity of one confirmed global condition-dependence result.

    The artifact intentionally excludes certificate, graph, and assessment hashes so
    an independently adjudicated reference can construct the same condition identity.
    Its exact bytes are nevertheless bound by :class:`ConditionSetSourceBindingV1`.
    """

    artifact_version: Literal["decisive-normalized-condition-set-v1"] = (
        "decisive-normalized-condition-set-v1"
    )
    question_id: Annotated[str, Field(min_length=1)]
    condition_target_sha256: Sha256
    selected_moderator: Annotated[str, Field(min_length=1)]
    positive_effect_level: Annotated[str, Field(min_length=1)]
    negative_effect_level: Annotated[str, Field(min_length=1)]
    decision_semantics: Literal[
        "qualitative_predictive_effect_modification_not_causal_interaction"
    ] = "qualitative_predictive_effect_modification_not_causal_interaction"
    artifact_sha256: Sha256

    @model_validator(mode="after")
    def validate_artifact(self) -> NormalizedConditionSetArtifactV1:
        if self.positive_effect_level == self.negative_effect_level:
            raise ValueError("decisive_condition_set_polarity_levels_must_differ")
        _self_hash(self, "artifact_sha256")
        return self


class VerifierCertificateLocatorV1(_FrozenExactModel):
    locator_version: Literal["decisive-verifier-certificate-locator-v1"] = (
        "decisive-verifier-certificate-locator-v1"
    )
    relative_path: Annotated[str, Field(min_length=1)]
    expected_file_sha256: Sha256
    expected_certificate_sha256: Sha256
    locator_sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _strict_relative_path(value, "verifier_certificate")

    @model_validator(mode="after")
    def validate_locator(self) -> VerifierCertificateLocatorV1:
        _self_hash(self, "locator_sha256")
        return self


class QuestionTrajectorySourceV1(_FrozenExactModel):
    source_version: Literal["decisive-question-trajectory-source-v1"] = (
        "decisive-question-trajectory-source-v1"
    )
    question_id: Annotated[str, Field(min_length=1)]
    workspaces: Annotated[list[TransactionalWorkspaceLocatorV1], Field(min_length=1)]
    adjudication_replay_package: AdjudicationReplayPackageLocatorV1
    verifier_certificates: list[VerifierCertificateLocatorV1]
    condition_set_bindings: list[ConditionSetSourceBindingV1]
    source_sha256: Sha256

    @model_validator(mode="after")
    def validate_source(self) -> QuestionTrajectorySourceV1:
        if self.workspaces != sorted(
            self.workspaces,
            key=lambda row: (
                row.relative_path,
                row.expected_workspace_config_sha256,
                row.expected_terminal_pointer_sha256,
            ),
        ):
            raise ValueError("decisive_trajectory_compiler_workspaces_not_canonical")
        if len({row.relative_path for row in self.workspaces}) != len(self.workspaces):
            raise ValueError("decisive_trajectory_compiler_workspace_path_duplicate")
        if self.verifier_certificates != sorted(
            self.verifier_certificates,
            key=lambda row: (row.relative_path, row.expected_certificate_sha256),
        ):
            raise ValueError("decisive_trajectory_compiler_certificates_not_canonical")
        if len({row.relative_path for row in self.verifier_certificates}) != len(
            self.verifier_certificates
        ):
            raise ValueError("decisive_trajectory_compiler_certificate_path_duplicate")
        if self.condition_set_bindings != sorted(
            self.condition_set_bindings, key=lambda row: row.certificate_sha256
        ):
            raise ValueError("decisive_trajectory_compiler_condition_bindings_not_canonical")
        if len({row.certificate_sha256 for row in self.condition_set_bindings}) != len(
            self.condition_set_bindings
        ):
            raise ValueError("decisive_trajectory_compiler_condition_certificate_duplicate")
        _self_hash(self, "source_sha256")
        return self


class DecisiveTrajectorySourceRosterV1(_FrozenExactModel):
    roster_version: Literal["decisive-trajectory-source-roster-v1"] = ROSTER_VERSION
    split_manifest_sha256: Sha256
    questions: Annotated[list[QuestionTrajectorySourceV1], Field(min_length=1)]
    evaluation_reference_labels_present: Literal[False] = False
    source_roster_sha256: Sha256

    @model_validator(mode="after")
    def validate_roster(self) -> DecisiveTrajectorySourceRosterV1:
        if self.questions != sorted(self.questions, key=lambda row: row.question_id):
            raise ValueError("decisive_trajectory_compiler_questions_not_canonical")
        if len({row.question_id for row in self.questions}) != len(self.questions):
            raise ValueError("decisive_trajectory_compiler_question_duplicate")
        _self_hash(self, "source_roster_sha256")
        return self


def freeze_transactional_workspace_locator_v1(
    *,
    relative_path: str,
    expected_workspace_config_sha256: str,
    expected_terminal_pointer_sha256: str,
) -> TransactionalWorkspaceLocatorV1:
    payload = {
        "locator_version": "decisive-transactional-workspace-locator-v1",
        "relative_path": relative_path,
        "expected_workspace_config_sha256": expected_workspace_config_sha256,
        "expected_terminal_pointer_sha256": expected_terminal_pointer_sha256,
    }
    return TransactionalWorkspaceLocatorV1.model_validate(
        {**payload, "locator_sha256": hash_canonical(payload)}
    )


def freeze_adjudication_replay_package_locator_v1(
    *,
    relative_path: str,
    expected_file_sha256: str,
    expected_package_sha256: str,
) -> AdjudicationReplayPackageLocatorV1:
    payload = {
        "locator_version": "decisive-adjudication-package-locator-v1",
        "relative_path": relative_path,
        "expected_file_sha256": expected_file_sha256,
        "expected_package_sha256": expected_package_sha256,
    }
    return AdjudicationReplayPackageLocatorV1.model_validate(
        {**payload, "locator_sha256": hash_canonical(payload)}
    )


def freeze_condition_set_source_binding_v1(
    *,
    certificate_sha256: str,
    relative_path: str,
    expected_file_sha256: str,
    condition_set_artifact_sha256: str,
) -> ConditionSetSourceBindingV1:
    payload = {
        "binding_version": "decisive-condition-set-source-binding-v1",
        "certificate_sha256": certificate_sha256,
        "relative_path": relative_path,
        "expected_file_sha256": expected_file_sha256,
        "condition_set_artifact_sha256": condition_set_artifact_sha256,
    }
    return ConditionSetSourceBindingV1.model_validate(
        {**payload, "binding_sha256": hash_canonical(payload)}
    )


def freeze_normalized_condition_set_artifact_v1(
    *,
    question_id: str,
    condition_target_sha256: str,
    selected_moderator: str,
    positive_effect_level: str,
    negative_effect_level: str,
) -> NormalizedConditionSetArtifactV1:
    payload = {
        "artifact_version": "decisive-normalized-condition-set-v1",
        "question_id": question_id,
        "condition_target_sha256": condition_target_sha256,
        "selected_moderator": selected_moderator,
        "positive_effect_level": positive_effect_level,
        "negative_effect_level": negative_effect_level,
        "decision_semantics": ("qualitative_predictive_effect_modification_not_causal_interaction"),
    }
    return NormalizedConditionSetArtifactV1.model_validate(
        {**payload, "artifact_sha256": hash_canonical(payload)}
    )


def freeze_verifier_certificate_locator_v1(
    *,
    relative_path: str,
    expected_file_sha256: str,
    expected_certificate_sha256: str,
) -> VerifierCertificateLocatorV1:
    payload = {
        "locator_version": "decisive-verifier-certificate-locator-v1",
        "relative_path": relative_path,
        "expected_file_sha256": expected_file_sha256,
        "expected_certificate_sha256": expected_certificate_sha256,
    }
    return VerifierCertificateLocatorV1.model_validate(
        {**payload, "locator_sha256": hash_canonical(payload)}
    )


def freeze_question_trajectory_source_v1(
    *,
    question_id: str,
    workspaces: Sequence[TransactionalWorkspaceLocatorV1],
    adjudication_replay_package: AdjudicationReplayPackageLocatorV1,
    verifier_certificates: Sequence[VerifierCertificateLocatorV1] = (),
    condition_set_bindings: Sequence[ConditionSetSourceBindingV1] = (),
) -> QuestionTrajectorySourceV1:
    rows = sorted(
        workspaces,
        key=lambda row: (
            row.relative_path,
            row.expected_workspace_config_sha256,
            row.expected_terminal_pointer_sha256,
        ),
    )
    certificates = sorted(
        verifier_certificates,
        key=lambda row: (row.relative_path, row.expected_certificate_sha256),
    )
    conditions = sorted(condition_set_bindings, key=lambda row: row.certificate_sha256)
    payload = {
        "source_version": "decisive-question-trajectory-source-v1",
        "question_id": question_id,
        "workspaces": rows,
        "adjudication_replay_package": adjudication_replay_package,
        "verifier_certificates": certificates,
        "condition_set_bindings": conditions,
    }
    return QuestionTrajectorySourceV1.model_validate(
        {**payload, "source_sha256": hash_canonical(payload)}
    )


def freeze_decisive_trajectory_source_roster_v1(
    *,
    split_manifest: DecisiveSplitManifestV1,
    questions: Sequence[QuestionTrajectorySourceV1],
) -> DecisiveTrajectorySourceRosterV1:
    rows = sorted(questions, key=lambda row: row.question_id)
    payload = {
        "roster_version": ROSTER_VERSION,
        "split_manifest_sha256": split_manifest.manifest_sha256,
        "questions": rows,
        "evaluation_reference_labels_present": False,
    }
    return DecisiveTrajectorySourceRosterV1.model_validate(
        {**payload, "source_roster_sha256": hash_canonical(payload)}
    )


class SourceArtifactBindingV1(_FrozenExactModel):
    binding_version: Literal["decisive-source-artifact-binding-v1"] = ARTIFACT_BINDING_VERSION
    relative_path: Annotated[str, Field(min_length=1)]
    artifact_kind: Annotated[str, Field(min_length=1)]
    file_sha256: Sha256
    canonical_sha256: Sha256 | None
    binding_sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _strict_relative_path(value, "bound_artifact")

    @model_validator(mode="after")
    def validate_binding(self) -> SourceArtifactBindingV1:
        _self_hash(self, "binding_sha256")
        return self


class TransactionalWorkspaceSourceBindingV1(_FrozenExactModel):
    binding_version: Literal["decisive-transactional-workspace-binding-v1"] = (
        WORKSPACE_BINDING_VERSION
    )
    locator_sha256: Sha256
    workspace_config_sha256: Sha256
    terminal_pointer_sha256: Sha256
    generation_count: Annotated[int, Field(ge=1)]
    artifact_bindings: Annotated[list[SourceArtifactBindingV1], Field(min_length=1)]
    certificate_sha256s: Annotated[list[Sha256], Field(min_length=1)]
    adjudication_receipt_sha256s: list[Sha256]
    checkpoint_receipt_sha256s: list[Sha256]
    resolution_result_sha256s: list[Sha256]
    checkpoint_result_sha256s: list[Sha256]
    workspace_source_sha256: Sha256

    @model_validator(mode="after")
    def validate_binding(self) -> TransactionalWorkspaceSourceBindingV1:
        if self.artifact_bindings != sorted(
            self.artifact_bindings, key=lambda row: row.relative_path
        ):
            raise ValueError("decisive_trajectory_compiler_artifacts_not_canonical")
        if len({row.relative_path for row in self.artifact_bindings}) != len(
            self.artifact_bindings
        ):
            raise ValueError("decisive_trajectory_compiler_artifact_path_duplicate")
        for field_name in (
            "certificate_sha256s",
            "adjudication_receipt_sha256s",
            "checkpoint_receipt_sha256s",
            "resolution_result_sha256s",
            "checkpoint_result_sha256s",
        ):
            values = getattr(self, field_name)
            if values != sorted(set(values)):
                raise ValueError(
                    f"decisive_trajectory_compiler_hash_roster_not_canonical:{field_name}"
                )
        _self_hash(self, "workspace_source_sha256")
        return self


class VerifierCertificateSourceBindingV1(_FrozenExactModel):
    binding_version: Literal["decisive-verifier-certificate-source-binding-v1"] = (
        CERTIFICATE_SOURCE_BINDING_VERSION
    )
    locator_sha256: Sha256
    artifact_binding: SourceArtifactBindingV1
    certificate_sha256: Sha256
    replay_sha256: Sha256
    source_binding_sha256: Sha256

    @model_validator(mode="after")
    def validate_binding(self) -> VerifierCertificateSourceBindingV1:
        _self_hash(self, "source_binding_sha256")
        return self


class AdjudicationReplayPackageSourceBindingV1(_FrozenExactModel):
    binding_version: Literal["decisive-adjudication-package-source-binding-v1"] = (
        "decisive-adjudication-package-source-binding-v1"
    )
    locator_sha256: Sha256
    question_id: Annotated[str, Field(min_length=1)]
    package_sha256: Sha256
    trust_registry_sha256: Sha256
    adjudication_protocol_file_sha256: Sha256
    package_artifact_binding: SourceArtifactBindingV1
    trust_registry_artifact_binding: SourceArtifactBindingV1
    adjudication_protocol_artifact_binding: SourceArtifactBindingV1
    workflow_artifact_bindings: Annotated[list[SourceArtifactBindingV1], Field(min_length=5)]
    verified_receipt_sha256s: Annotated[list[Sha256], Field(min_length=1)]
    operator_trust_registry_hash_bound: Literal[True] = True
    workflow_artifact_integrity_verified: Literal[True] = True
    cryptographic_reviewer_identity_verified: Literal[False] = False
    external_reviewer_expertise_verified: Literal[False] = False
    trust_semantics: Literal[
        "operator_declared_hash_bound_registry_not_cryptographic_identity_or_expertise_proof"
    ] = "operator_declared_hash_bound_registry_not_cryptographic_identity_or_expertise_proof"
    binding_sha256: Sha256

    @model_validator(mode="after")
    def validate_binding(self) -> AdjudicationReplayPackageSourceBindingV1:
        if self.workflow_artifact_bindings != sorted(
            self.workflow_artifact_bindings, key=lambda row: row.relative_path
        ) or len({row.relative_path for row in self.workflow_artifact_bindings}) != len(
            self.workflow_artifact_bindings
        ):
            raise ValueError("decisive_trajectory_compiler_adjudication_artifacts_not_canonical")
        if self.verified_receipt_sha256s != sorted(set(self.verified_receipt_sha256s)):
            raise ValueError("decisive_trajectory_compiler_adjudication_receipts_not_canonical")
        _self_hash(self, "binding_sha256")
        return self


class AdoptedReplayBindingV1(_FrozenExactModel):
    binding_version: Literal["decisive-adopted-replay-binding-v1"] = ADOPTED_REPLAY_VERSION
    audit_sequence: list[str]
    source_certificate_sha256: Sha256
    source_kind: Literal["transactional_workspace", "standalone_verifier_certificate"]
    source_container_sha256: Sha256
    replay_sha256: Sha256
    semantic_projection_sha256: Sha256
    binding_sha256: Sha256

    @model_validator(mode="after")
    def validate_binding(self) -> AdoptedReplayBindingV1:
        if len(self.audit_sequence) != len(set(self.audit_sequence)):
            raise ValueError("decisive_trajectory_compiler_adopted_prefix_duplicate_item")
        _self_hash(self, "binding_sha256")
        return self


class CompiledAuditEventBindingV1(_FrozenExactModel):
    binding_version: Literal["decisive-compiled-audit-event-binding-v1"] = (
        "decisive-compiled-audit-event-binding-v1"
    )
    item_id: Annotated[str, Field(min_length=1)]
    event_sha256: Sha256
    source_receipt_sha256s: Annotated[list[Sha256], Field(min_length=1)]
    source_resolution_result_sha256s: Annotated[list[Sha256], Field(min_length=1)]
    binding_sha256: Sha256

    @model_validator(mode="after")
    def validate_binding(self) -> CompiledAuditEventBindingV1:
        if self.source_receipt_sha256s != sorted(set(self.source_receipt_sha256s)):
            raise ValueError("decisive_trajectory_compiler_event_receipts_not_canonical")
        if self.source_resolution_result_sha256s != sorted(
            set(self.source_resolution_result_sha256s)
        ):
            raise ValueError("decisive_trajectory_compiler_event_results_not_canonical")
        _self_hash(self, "binding_sha256")
        return self


class QuestionTrajectoryCompilationReceiptV1(_FrozenExactModel):
    receipt_version: Literal["decisive-question-trajectory-compilation-v1"] = (
        QUESTION_RECEIPT_VERSION
    )
    question_id: Annotated[str, Field(min_length=1)]
    question_identity_sha256: Sha256
    source_roster_entry_sha256: Sha256
    workspace_bindings: Annotated[list[TransactionalWorkspaceSourceBindingV1], Field(min_length=1)]
    adjudication_replay_package_binding: AdjudicationReplayPackageSourceBindingV1
    verifier_certificate_bindings: list[VerifierCertificateSourceBindingV1]
    condition_set_artifact_bindings: list[SourceArtifactBindingV1]
    available_prefixes: list[list[str]]
    required_policy_visited_prefixes: list[list[str]]
    adopted_replays: Annotated[list[AdoptedReplayBindingV1], Field(min_length=2)]
    audit_event_bindings: Annotated[list[CompiledAuditEventBindingV1], Field(min_length=1)]
    total_realized_person_minutes: Annotated[float, Field(gt=0)]
    trajectory_sha256: Sha256
    real_source_provenance_verified: Literal[True] = True
    operator_declared_expert_workflow_replayed: Literal[True] = True
    external_reviewer_identity_or_expertise_proven: Literal[False] = False
    evaluation_reference_labels_opened: Literal[False] = False
    scientific_claim_authority: Literal[False] = False
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> QuestionTrajectoryCompilationReceiptV1:
        workspace_receipts = sorted(
            {
                receipt_sha256
                for binding in self.workspace_bindings
                for receipt_sha256 in binding.adjudication_receipt_sha256s
            }
        )
        event_receipts = sorted(
            {
                receipt_sha256
                for binding in self.audit_event_bindings
                for receipt_sha256 in binding.source_receipt_sha256s
            }
        )
        if (
            self.adjudication_replay_package_binding.question_id != self.question_id
            or self.adjudication_replay_package_binding.verified_receipt_sha256s
            != workspace_receipts
            or event_receipts != workspace_receipts
        ):
            raise ValueError(
                "decisive_trajectory_compiler_adjudication_receipt_projection_mismatch"
            )
        if self.workspace_bindings != sorted(
            self.workspace_bindings, key=lambda row: row.workspace_source_sha256
        ):
            raise ValueError("decisive_trajectory_compiler_workspace_bindings_not_canonical")
        if self.verifier_certificate_bindings != sorted(
            self.verifier_certificate_bindings,
            key=lambda row: row.source_binding_sha256,
        ):
            raise ValueError("decisive_trajectory_compiler_certificate_bindings_not_canonical")
        if self.condition_set_artifact_bindings != sorted(
            self.condition_set_artifact_bindings,
            key=lambda row: row.relative_path,
        ):
            raise ValueError("decisive_trajectory_compiler_condition_artifacts_not_canonical")
        if len({row.relative_path for row in self.condition_set_artifact_bindings}) != len(
            self.condition_set_artifact_bindings
        ):
            raise ValueError("decisive_trajectory_compiler_condition_artifact_path_duplicate")
        if self.available_prefixes != sorted(
            self.available_prefixes, key=lambda row: (len(row), row)
        ):
            raise ValueError("decisive_trajectory_compiler_available_prefixes_not_canonical")
        if self.required_policy_visited_prefixes != sorted(
            self.required_policy_visited_prefixes, key=lambda row: (len(row), row)
        ):
            raise ValueError("decisive_trajectory_compiler_required_prefixes_not_canonical")
        if [row.audit_sequence for row in self.adopted_replays] != (
            self.required_policy_visited_prefixes
        ):
            raise ValueError("decisive_trajectory_compiler_adopted_prefix_projection_mismatch")
        if self.audit_event_bindings != sorted(
            self.audit_event_bindings, key=lambda row: row.item_id
        ):
            raise ValueError("decisive_trajectory_compiler_event_bindings_not_canonical")
        _self_hash(self, "receipt_sha256")
        return self


class DecisiveTrajectoryCompilationReceiptV1(_FrozenExactModel):
    compilation_version: Literal["decisive-trajectory-compilation-v1"] = COMPILATION_VERSION
    compiled_at: datetime
    compiler_component_sha256: Sha256
    config_sha256: Sha256
    split_manifest_sha256: Sha256
    development_receipt_sha256: Sha256
    calibration_receipt_sha256: Sha256
    policy_input_provenance_sha256: Sha256
    source_roster_file_sha256: Sha256
    source_roster_sha256: Sha256
    question_receipts: Annotated[list[QuestionTrajectoryCompilationReceiptV1], Field(min_length=1)]
    trajectory_bundle_sha256: Sha256
    trajectory_membership_sha256: Sha256
    compilation_lineage_identity: DecisiveCompilationLineageIdentityV1
    evidence_kind: Literal["real_expert_adjudicated"] = "real_expert_adjudicated"
    union_semantics: Literal[
        "exact_union_of_prefixes_visited_by_the_prespecified_decisive_policy_roster"
    ] = "exact_union_of_prefixes_visited_by_the_prespecified_decisive_policy_roster"
    realized_cost_semantics: Literal[
        "total_person_minutes_across_all_reviewers_and_final_adjudication"
    ] = "total_person_minutes_across_all_reviewers_and_final_adjudication"
    source_locator_portability: Literal[
        "local_relative_paths_replayed_to_exact_content_and_semantic_hashes"
    ] = "local_relative_paths_replayed_to_exact_content_and_semantic_hashes"
    evaluation_reference_labels_opened: Literal[False] = False
    real_empirical_candidate: Literal[True] = True
    scientific_claim_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    compilation_sha256: Sha256

    @field_validator("compiled_at")
    @classmethod
    def validate_compiled_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("decisive_trajectory_compiler_timestamp_requires_timezone")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> DecisiveTrajectoryCompilationReceiptV1:
        if self.question_receipts != sorted(
            self.question_receipts, key=lambda row: row.question_id
        ):
            raise ValueError("decisive_trajectory_compiler_question_receipts_not_canonical")
        if len({row.question_id for row in self.question_receipts}) != len(self.question_receipts):
            raise ValueError("decisive_trajectory_compiler_question_receipt_duplicate")
        expected_lineage = freeze_decisive_compilation_lineage_identity_v1(
            compiler_component_sha256=self.compiler_component_sha256,
            config_sha256=self.config_sha256,
            split_manifest_sha256=self.split_manifest_sha256,
            development_receipt_sha256=self.development_receipt_sha256,
            calibration_receipt_sha256=self.calibration_receipt_sha256,
            source_roster_file_sha256=self.source_roster_file_sha256,
            source_roster_sha256=self.source_roster_sha256,
            trajectory_bundle_sha256=self.trajectory_bundle_sha256,
            trajectory_membership_sha256=self.trajectory_membership_sha256,
            evaluation_question_ids=[row.question_id for row in self.question_receipts],
            question_receipt_sha256s=[row.receipt_sha256 for row in self.question_receipts],
            adjudication_package_binding_sha256s=[
                row.adjudication_replay_package_binding.binding_sha256
                for row in self.question_receipts
            ],
        )
        if self.compilation_lineage_identity != expected_lineage:
            raise ValueError("decisive_trajectory_compiler_compilation_lineage_projection_mismatch")
        _self_hash(self, "compilation_sha256")
        return self


class DecisiveTrajectoryCompilationResultV1(_FrozenExactModel):
    result_version: Literal["decisive-trajectory-compilation-result-v1"] = (
        "decisive-trajectory-compilation-result-v1"
    )
    trajectory_bundle: TrajectoryBundleV1
    compilation_receipt: DecisiveTrajectoryCompilationReceiptV1
    result_sha256: Sha256

    @model_validator(mode="after")
    def validate_result(self) -> DecisiveTrajectoryCompilationResultV1:
        if (
            self.trajectory_bundle.bundle_sha256
            != self.compilation_receipt.trajectory_bundle_sha256
        ):
            raise ValueError("decisive_trajectory_compiler_result_bundle_mismatch")
        _self_hash(self, "result_sha256")
        return self


def compute_decisive_trajectory_compiler_component_sha256_v1(
    repository_root: Path,
) -> str:
    root = repository_root.resolve(strict=True)
    if not root.is_dir():
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_repository_root_not_directory"
        )
    rows: list[dict[str, str]] = []
    for relative in _COMPONENT_PATHS:
        current = root
        for part in PurePosixPath(relative).parts:
            current /= part
            if current.is_symlink():
                raise DecisiveTrajectoryCompilerV1Error(
                    f"decisive_trajectory_compiler_component_symlink:{relative}"
                )
        try:
            resolved = current.resolve(strict=True)
        except FileNotFoundError as exc:
            raise DecisiveTrajectoryCompilerV1Error(
                f"decisive_trajectory_compiler_component_missing:{relative}"
            ) from exc
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise DecisiveTrajectoryCompilerV1Error(
                f"decisive_trajectory_compiler_component_path_invalid:{relative}"
            )
        rows.append({"path": relative, "file_sha256": _sha256_bytes(resolved.read_bytes())})
    return hash_canonical(
        {
            "component_version": COMPILER_COMPONENT_VERSION,
            "files": rows,
        }
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular_no_follow(path: Path, *, label: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise DecisiveTrajectoryCompilerV1Error(
            f"decisive_trajectory_compiler_source_unreadable:{label}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_ARTIFACT_BYTES:
            raise DecisiveTrajectoryCompilerV1Error(
                f"decisive_trajectory_compiler_source_file_invalid:{label}"
            )
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise DecisiveTrajectoryCompilerV1Error(
                    f"decisive_trajectory_compiler_source_short_read:{label}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        final_metadata = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(final_metadata, field_name) != getattr(metadata, field_name)
            for field_name in stable_fields
        ):
            raise DecisiveTrajectoryCompilerV1Error(
                f"decisive_trajectory_compiler_source_changed_during_read:{label}"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _artifact_binding(
    *, relative_path: str, artifact_kind: str, raw: bytes, model: ContractModel | None
) -> SourceArtifactBindingV1:
    payload = {
        "binding_version": ARTIFACT_BINDING_VERSION,
        "relative_path": relative_path,
        "artifact_kind": artifact_kind,
        "file_sha256": _sha256_bytes(raw),
        "canonical_sha256": None if model is None else hash_canonical(model),
    }
    return SourceArtifactBindingV1.model_validate(
        {**payload, "binding_sha256": hash_canonical(payload)}
    )


def _read_model[ModelT: ContractModel](
    path: Path, model_type: type[ModelT], *, relative_path: str, artifact_kind: str
) -> tuple[ModelT, SourceArtifactBindingV1]:
    raw = _read_regular_no_follow(path, label=relative_path)
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("not_object")
        model = model_type.model_validate(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise DecisiveTrajectoryCompilerV1Error(
            f"decisive_trajectory_compiler_source_model_invalid:{relative_path}"
        ) from exc
    return model, _artifact_binding(
        relative_path=relative_path,
        artifact_kind=artifact_kind,
        raw=raw,
        model=model,
    )


type _ProductionCertificate = VerificationCertificate | FinalConditionVerificationCertificateV7


def _read_production_certificate(
    path: Path,
    *,
    relative_path: str,
    artifact_kind: str,
) -> tuple[_ProductionCertificate, SourceArtifactBindingV1]:
    """Read exactly one supported terminal certificate without union coercion."""

    raw = _read_regular_no_follow(path, label=relative_path)
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("not_object")
        version = payload.get("certificate_version")
        if version == "literature-multiverse-verification-v5":
            certificate: _ProductionCertificate = VerificationCertificate.model_validate(payload)
        elif version == "literature-multiverse-condition-verification-v7":
            certificate = FinalConditionVerificationCertificateV7.model_validate(payload)
        else:
            raise ValueError("unsupported_certificate_version")
    except (json.JSONDecodeError, ValueError) as exc:
        raise DecisiveTrajectoryCompilerV1Error(
            f"decisive_trajectory_compiler_source_model_invalid:{relative_path}"
        ) from exc
    return certificate, _artifact_binding(
        relative_path=relative_path,
        artifact_kind=artifact_kind,
        raw=raw,
        model=certificate,
    )


def _normalized_condition_set_from_v7(
    certificate: FinalConditionVerificationCertificateV7,
) -> NormalizedConditionSetArtifactV1:
    """Derive the scoreable condition identity from validated frozen v7 inputs."""

    source = certificate.source_certificate_v6
    model = source.condition_frozen_model
    if (
        model.status != "fitted"
        or model.selected_moderator is None
        or model.frozen_positive_level is None
        or model.frozen_negative_level is None
    ):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_final_condition_model_incomplete"
        )
    return freeze_normalized_condition_set_artifact_v1(
        question_id=source.release_assessment.question_id,
        condition_target_sha256=(source.condition_calibration_projection.condition_target_sha256),
        selected_moderator=model.selected_moderator,
        positive_effect_level=model.frozen_positive_level,
        negative_effect_level=model.frozen_negative_level,
    )


def _snapshot_condition_set_artifact(
    *,
    source_root: Path,
    binding: ConditionSetSourceBindingV1,
) -> tuple[NormalizedConditionSetArtifactV1, SourceArtifactBindingV1]:
    """Replay one normalized condition artifact from exact safe bytes."""

    artifact, artifact_binding = _read_model(
        _resolve_source_path(source_root, binding.relative_path),
        NormalizedConditionSetArtifactV1,
        relative_path=binding.relative_path,
        artifact_kind="normalized_condition_set_artifact",
    )
    if (
        artifact_binding.file_sha256 != binding.expected_file_sha256
        or artifact.artifact_sha256 != binding.condition_set_artifact_sha256
    ):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_condition_set_locator_identity_mismatch"
        )
    return artifact, artifact_binding


def _resolve_source_path(source_root: Path, relative: str) -> Path:
    root = source_root.resolve(strict=True)
    if source_root.is_symlink() or not root.is_dir():
        raise DecisiveTrajectoryCompilerV1Error("decisive_trajectory_compiler_source_root_invalid")
    current = root
    for part in PurePosixPath(_strict_relative_path(relative, "source_locator")).parts:
        current /= part
        if current.is_symlink():
            raise DecisiveTrajectoryCompilerV1Error(
                f"decisive_trajectory_compiler_source_symlink:{relative}"
            )
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise DecisiveTrajectoryCompilerV1Error(
            f"decisive_trajectory_compiler_source_escape:{relative}"
        )
    return resolved


@dataclass(frozen=True)
class _ReplayCandidate:
    certificate: _ProductionCertificate
    replay: QuestionReplayState
    source_kind: Literal["transactional_workspace", "standalone_verifier_certificate"]
    source_container_sha256: str


@dataclass(frozen=True)
class _EventOccurrence:
    receipt: AuditAdjudicationCostReceiptV1
    resolution: SequentialResolutionResult
    authorization_issued_at: datetime


@dataclass(frozen=True)
class _WorkspaceSnapshot:
    config: AuditWorkspaceConfigV1
    terminal_pointer: AuditWorkspacePointerV1
    binding: TransactionalWorkspaceSourceBindingV1
    replay_candidates: tuple[_ReplayCandidate, ...]
    event_occurrences: tuple[_EventOccurrence, ...]


@dataclass(frozen=True)
class _CertificateSnapshot:
    binding: VerifierCertificateSourceBindingV1
    replay_candidate: _ReplayCandidate


def _snapshot_workspace(
    *,
    workspace: Path,
    locator: TransactionalWorkspaceLocatorV1,
) -> _WorkspaceSnapshot:
    with _workspace_lock(workspace):
        root, config, terminal_pointer = _load_workspace(workspace)
        if (
            config.config_sha256 != locator.expected_workspace_config_sha256
            or terminal_pointer.pointer_sha256 != locator.expected_terminal_pointer_sha256
        ):
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_workspace_locator_identity_mismatch"
            )
        allowed_root = {
            "workspace-config.json",
            "current-pointer.json",
            "transaction-marker.json",
            "generations",
        }
        observed_root = {row.name for row in root.iterdir()}
        if observed_root != allowed_root:
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_workspace_root_roster_mismatch"
            )
        artifacts: list[SourceArtifactBindingV1] = []
        parsed_config, binding = _read_model(
            root / "workspace-config.json",
            AuditWorkspaceConfigV1,
            relative_path="workspace-config.json",
            artifact_kind="workspace_config",
        )
        artifacts.append(binding)
        parsed_pointer, binding = _read_model(
            root / "current-pointer.json",
            AuditWorkspacePointerV1,
            relative_path="current-pointer.json",
            artifact_kind="current_pointer",
        )
        artifacts.append(binding)
        marker, binding = _read_model(
            root / "transaction-marker.json",
            AuditWorkspaceTransactionMarkerV1,
            relative_path="transaction-marker.json",
            artifact_kind="transaction_marker",
        )
        artifacts.append(binding)
        if parsed_config != config or parsed_pointer != terminal_pointer:
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_workspace_root_replay_mismatch"
            )
        if marker.committed_pointer_sha256 != terminal_pointer.pointer_sha256:
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_workspace_marker_mismatch"
            )

        generation_root = root / "generations"
        generation_dirs = sorted(generation_root.iterdir(), key=lambda row: row.name)
        if len(generation_dirs) != terminal_pointer.generation + 1:
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_workspace_generation_count_mismatch"
            )
        replay_candidates: list[_ReplayCandidate] = []
        occurrences: list[_EventOccurrence] = []
        certificate_hashes: list[str] = []
        adjudication_hashes: list[str] = []
        checkpoint_hashes: list[str] = []
        resolution_hashes: list[str] = []
        checkpoint_result_hashes: list[str] = []
        authorization_issued_at_by_sha256: dict[str, datetime] = {}

        # The workspace validator already proved the generation chain.  We now bind
        # the exact bytes and retain the typed joins needed by the evaluation.
        for generation, generation_dir in enumerate(generation_dirs):
            if generation_dir.is_symlink() or not generation_dir.is_dir():
                raise DecisiveTrajectoryCompilerV1Error(
                    "decisive_trajectory_compiler_generation_directory_invalid"
                )
            prefix = f"{generation:06d}-"
            if not generation_dir.name.startswith(prefix):
                raise DecisiveTrajectoryCompilerV1Error(
                    "decisive_trajectory_compiler_generation_order_invalid"
                )
            relative_root = f"generations/{generation_dir.name}"
            pointer, pointer_binding = _read_model(
                generation_dir / "generation-pointer.json",
                AuditWorkspacePointerV1,
                relative_path=f"{relative_root}/generation-pointer.json",
                artifact_kind="generation_pointer",
            )
            artifacts.append(pointer_binding)
            if pointer.authorization is not None:
                authorization_issued_at_by_sha256[pointer.authorization.authorization_sha256] = (
                    pointer.authorization.issued_at
                )
            expected_names = {
                "sequential-audit-state.json",
                "verification-certificate.json",
                "verification-certificate.html",
                "state-expectation.json",
                "generation-pointer.json",
            }
            if pointer.authorization is not None:
                expected_names.add("audit-action-authorization.json")
            if generation > 0:
                expected_names.update(
                    {
                        "transition-receipt.json",
                        "preflight-verification-certificate.json",
                        "transition-result.json",
                    }
                )
            observed_names = {row.name for row in generation_dir.iterdir()}
            if observed_names != expected_names:
                raise DecisiveTrajectoryCompilerV1Error(
                    "decisive_trajectory_compiler_generation_artifact_roster_mismatch"
                )

            state, state_binding = _read_model(
                generation_dir / "sequential-audit-state.json",
                SequentialVerificationState,
                relative_path=f"{relative_root}/sequential-audit-state.json",
                artifact_kind="sequential_state",
            )
            artifacts.append(state_binding)
            certificate, certificate_binding = _read_model(
                generation_dir / "verification-certificate.json",
                VerificationCertificate,
                relative_path=f"{relative_root}/verification-certificate.json",
                artifact_kind="verification_certificate",
            )
            artifacts.append(certificate_binding)
            certificate_hashes.append(certificate.certificate_sha256)
            html_path = generation_dir / "verification-certificate.html"
            html_raw = _read_regular_no_follow(
                html_path, label=f"{relative_root}/verification-certificate.html"
            )
            artifacts.append(
                _artifact_binding(
                    relative_path=f"{relative_root}/verification-certificate.html",
                    artifact_kind="verification_certificate_rendering",
                    raw=html_raw,
                    model=None,
                )
            )
            # These models were already validated by _load_workspace.  Binding them
            # again makes the compiler receipt independently enumerate exact inputs.
            for name, model_type, kind in (
                ("state-expectation.json", type(pointer.state_expectation), "state_expectation"),
                (
                    "audit-action-authorization.json",
                    type(pointer.authorization) if pointer.authorization is not None else None,
                    "audit_action_authorization",
                ),
            ):
                if model_type is None:
                    continue
                _, item_binding = _read_model(
                    generation_dir / name,
                    model_type,
                    relative_path=f"{relative_root}/{name}",
                    artifact_kind=kind,
                )
                artifacts.append(item_binding)
            if certificate.sequential_audit_state != state:
                raise DecisiveTrajectoryCompilerV1Error(
                    "decisive_trajectory_compiler_generation_certificate_state_mismatch"
                )

            if generation == 0 or pointer.transition_kind == "adjudicated":
                replay = freeze_question_replay_state_from_certificate(certificate)
                binding = replay.production_binding
                if binding is None:
                    raise DecisiveTrajectoryCompilerV1Error(
                        "decisive_trajectory_compiler_preselection_certificate_required"
                    )
                if binding.evaluated_active_action_item_id is not None:
                    if generation != 0:
                        raise DecisiveTrajectoryCompilerV1Error(
                            "decisive_trajectory_compiler_preselection_certificate_required"
                        )
                else:
                    replay_candidates.append(
                        _ReplayCandidate(
                            certificate=certificate,
                            replay=replay,
                            source_kind="transactional_workspace",
                            source_container_sha256="",  # rebound after source hash is frozen
                        )
                    )

            if generation > 0:
                preflight, preflight_binding = _read_model(
                    generation_dir / "preflight-verification-certificate.json",
                    VerificationCertificate,
                    relative_path=f"{relative_root}/preflight-verification-certificate.json",
                    artifact_kind="preflight_verification_certificate",
                )
                artifacts.append(preflight_binding)
                certificate_hashes.append(preflight.certificate_sha256)
                if pointer.transition_kind == "checkpointed":
                    receipt, receipt_binding = _read_model(
                        generation_dir / "transition-receipt.json",
                        AuditActiveCostCheckpointReceiptV1,
                        relative_path=f"{relative_root}/transition-receipt.json",
                        artifact_kind="active_cost_checkpoint_receipt",
                    )
                    result, result_binding = _read_model(
                        generation_dir / "transition-result.json",
                        SequentialActiveCostCheckpointResult,
                        relative_path=f"{relative_root}/transition-result.json",
                        artifact_kind="active_cost_checkpoint_result",
                    )
                    if receipt.provenance not in {"human_timekeeper", "system_timer"}:
                        raise DecisiveTrajectoryCompilerV1Error(
                            "decisive_trajectory_compiler_benchmark_checkpoint_forbidden"
                        )
                    checkpoint_hashes.append(receipt.receipt_sha256)
                    checkpoint_result_hashes.append(result.result_sha256)
                else:
                    receipt, receipt_binding = _read_model(
                        generation_dir / "transition-receipt.json",
                        AuditAdjudicationCostReceiptV1,
                        relative_path=f"{relative_root}/transition-receipt.json",
                        artifact_kind="adjudication_cost_receipt",
                    )
                    result, result_binding = _read_model(
                        generation_dir / "transition-result.json",
                        SequentialResolutionResult,
                        relative_path=f"{relative_root}/transition-result.json",
                        artifact_kind="sequential_resolution_result",
                    )
                    adjudication_hashes.append(receipt.receipt_sha256)
                    resolution_hashes.append(result.result_sha256)
                    authorization_issued_at = authorization_issued_at_by_sha256.get(
                        receipt.authorization_sha256
                    )
                    if authorization_issued_at is None:
                        raise DecisiveTrajectoryCompilerV1Error(
                            "decisive_trajectory_compiler_adjudication_authorization_missing"
                        )
                    occurrences.append(
                        _EventOccurrence(
                            receipt=receipt,
                            resolution=result,
                            authorization_issued_at=authorization_issued_at,
                        )
                    )
                    evaluated = certificate.production_stop_decision.evaluated_state
                    if evaluated != result.state:
                        raise DecisiveTrajectoryCompilerV1Error(
                            "decisive_trajectory_compiler_resolution_certificate_join_mismatch"
                        )
                artifacts.extend((receipt_binding, result_binding))

        # A second full replay while the lock is still held detects any mutation by a
        # writer that obeys the workspace lock between our initial replay and reads.
        replay_root, replay_config, replay_pointer = _load_workspace(workspace)
        if replay_root != root or replay_config != config or replay_pointer != terminal_pointer:
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_workspace_changed_during_snapshot"
            )
        artifacts.sort(key=lambda row: row.relative_path)
        binding_payload = {
            "binding_version": WORKSPACE_BINDING_VERSION,
            "locator_sha256": locator.locator_sha256,
            "workspace_config_sha256": config.config_sha256,
            "terminal_pointer_sha256": terminal_pointer.pointer_sha256,
            "generation_count": terminal_pointer.generation + 1,
            "artifact_bindings": artifacts,
            "certificate_sha256s": sorted(set(certificate_hashes)),
            "adjudication_receipt_sha256s": sorted(set(adjudication_hashes)),
            "checkpoint_receipt_sha256s": sorted(set(checkpoint_hashes)),
            "resolution_result_sha256s": sorted(set(resolution_hashes)),
            "checkpoint_result_sha256s": sorted(set(checkpoint_result_hashes)),
        }
        source_binding = TransactionalWorkspaceSourceBindingV1.model_validate(
            {
                **binding_payload,
                "workspace_source_sha256": hash_canonical(binding_payload),
            }
        )
        rebound = tuple(
            _ReplayCandidate(
                certificate=row.certificate,
                replay=row.replay,
                source_kind="transactional_workspace",
                source_container_sha256=source_binding.workspace_source_sha256,
            )
            for row in replay_candidates
        )
        return _WorkspaceSnapshot(
            config=config,
            terminal_pointer=terminal_pointer,
            binding=source_binding,
            replay_candidates=rebound,
            event_occurrences=tuple(occurrences),
        )


def _snapshot_adjudication_replay_package(
    *,
    source_root: Path,
    locator: AdjudicationReplayPackageLocatorV1,
    question_id: str,
    occurrences_by_item: Mapping[str, Sequence[_EventOccurrence]],
) -> AdjudicationReplayPackageSourceBindingV1:
    """Replay exact raw workflow files against transactional adjudication receipts."""

    package, package_binding = _read_model(
        _resolve_source_path(source_root, locator.relative_path),
        AdjudicationReplayPackageV1,
        relative_path=locator.relative_path,
        artifact_kind="adjudication_replay_package",
    )
    if (
        package_binding.file_sha256 != locator.expected_file_sha256
        or package.package_sha256 != locator.expected_package_sha256
        or package.question_id != question_id
    ):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_adjudication_package_identity_mismatch"
        )

    def read_bound[ModelT: ContractModel](
        artifact_locator: Any,
        model_type: type[ModelT],
        *,
        artifact_kind: str,
    ) -> tuple[ModelT, SourceArtifactBindingV1]:
        model, binding = _read_model(
            _resolve_source_path(source_root, artifact_locator.relative_path),
            model_type,
            relative_path=artifact_locator.relative_path,
            artifact_kind=artifact_kind,
        )
        if binding.file_sha256 != artifact_locator.expected_file_sha256:
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_adjudication_artifact_hash_mismatch:"
                f"{artifact_locator.relative_path}"
            )
        return model, binding

    registry, registry_binding = read_bound(
        package.trust_registry,
        AdjudicationOperatorTrustRegistryV1,
        artifact_kind="adjudication_operator_trust_registry",
    )
    protocol, protocol_binding = read_bound(
        package.adjudication_protocol,
        AdjudicationProtocolArtifactV1,
        artifact_kind="adjudication_protocol",
    )
    if protocol.trust_registry_sha256 != registry.registry_sha256:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_adjudication_protocol_registry_mismatch"
        )
    roles_by_reviewer = {row.reviewer_id: set(row.roles) for row in registry.reviewers}
    protocol_file_sha256 = protocol_binding.file_sha256
    items = {row.item_id: row for row in package.items}
    if set(items) != set(occurrences_by_item):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_adjudication_package_item_membership_mismatch"
        )

    workflow_bindings: list[SourceArtifactBindingV1] = []
    verified_receipt_sha256s: set[str] = set()
    for item_id, occurrences in sorted(occurrences_by_item.items()):
        item = items[item_id]
        receipt_sha256s = {row.receipt.receipt_sha256 for row in occurrences}
        if set(item.receipt_sha256s) != receipt_sha256s:
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_adjudication_package_receipt_membership_mismatch:"
                f"{item_id}"
            )
        receipts = [row.receipt for row in occurrences]
        if any(row.provenance != "blinded_human" for row in receipts):
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_adjudication_package_blinded_human_required:"
                f"{item_id}"
            )

        decision_rows: list[tuple[IndependentReviewerDecisionV1, SourceArtifactBindingV1]] = [
            read_bound(
                row,
                IndependentReviewerDecisionV1,
                artifact_kind="independent_reviewer_decision",
            )
            for row in item.independent_reviewer_decisions
        ]
        timing_rows: list[tuple[ReviewerTimingEvidenceV1, SourceArtifactBindingV1]] = [
            read_bound(
                row,
                ReviewerTimingEvidenceV1,
                artifact_kind="reviewer_timing_evidence",
            )
            for row in item.timing_evidence
        ]
        resolution, resolution_binding = read_bound(
            item.resolution,
            AdjudicationResolutionArtifactV1,
            artifact_kind="adjudication_resolution",
        )
        correction, correction_binding = read_bound(
            item.correction_payload,
            AdjudicationCorrectionArtifactV1,
            artifact_kind="adjudication_correction_payload",
        )
        workflow_bindings.extend(
            [
                *(row[1] for row in decision_rows),
                *(row[1] for row in timing_rows),
                resolution_binding,
                correction_binding,
            ]
        )

        decisions_by_reviewer = {row.reviewer_id: (row, binding) for row, binding in decision_rows}
        timings_by_reviewer = {row.reviewer_id: row for row, _ in timing_rows}
        if (
            len(decisions_by_reviewer) != len(decision_rows)
            or len(timings_by_reviewer) != len(timing_rows)
            or len(decision_rows) < protocol.minimum_independent_reviewers
        ):
            raise DecisiveTrajectoryCompilerV1Error(
                f"decisive_trajectory_compiler_adjudication_independent_roster_invalid:{item_id}"
            )
        decision_reviewer_ids = set(decisions_by_reviewer)
        if any(
            row.question_id != question_id
            or row.item_id != item_id
            or row.adjudication_protocol_file_sha256 != protocol_file_sha256
            or "independent_reviewer" not in roles_by_reviewer.get(row.reviewer_id, set())
            for row, _ in decision_rows
        ):
            raise DecisiveTrajectoryCompilerV1Error(
                f"decisive_trajectory_compiler_adjudication_decision_membership_mismatch:{item_id}"
            )
        expected_decision_digests = sorted(
            [
                ReviewerDecisionDigestV1(
                    reviewer_id=row.reviewer_id,
                    decision_file_sha256=binding.file_sha256,
                )
                for row, binding in decision_rows
            ],
            key=lambda row: row.reviewer_id,
        )
        if (
            resolution.question_id != question_id
            or resolution.item_id != item_id
            or resolution.independent_decisions != expected_decision_digests
            or resolution.adjudication_protocol_file_sha256 != protocol_file_sha256
            or resolution.final_adjudicator_id in decision_reviewer_ids
            or "final_adjudicator"
            not in roles_by_reviewer.get(resolution.final_adjudicator_id, set())
            or resolution.correction_payload_file_sha256 != correction_binding.file_sha256
        ):
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_adjudication_resolution_membership_mismatch:"
                f"{item_id}"
            )
        participants = decision_reviewer_ids | {resolution.final_adjudicator_id}
        earliest_permitted_start = max(row.authorization_issued_at for row in occurrences)
        if set(timings_by_reviewer) != participants or any(
            timing.question_id != question_id
            or timing.item_id != item_id
            or timing.adjudication_protocol_file_sha256 != protocol_file_sha256
            or "timekeeper" not in roles_by_reviewer.get(timing.observer_id, set())
            or timing.started_at < earliest_permitted_start
            or timing.completed_at > resolution.completed_at
            for timing, _ in timing_rows
        ):
            raise DecisiveTrajectoryCompilerV1Error(
                f"decisive_trajectory_compiler_adjudication_timing_membership_mismatch:{item_id}"
            )
        if (
            any(
                decision.submitted_at != timings_by_reviewer[reviewer_id].completed_at
                for reviewer_id, (decision, _) in decisions_by_reviewer.items()
            )
            or timings_by_reviewer[resolution.final_adjudicator_id].completed_at
            != resolution.completed_at
        ):
            raise DecisiveTrajectoryCompilerV1Error(
                f"decisive_trajectory_compiler_adjudication_timing_join_mismatch:{item_id}"
            )
        realized_minutes = math.fsum(row.active_person_minutes for row, _ in timing_rows)
        if (
            correction.question_id != question_id
            or correction.item_id != item_id
            or correction.adjudication_protocol_file_sha256 != protocol_file_sha256
            or correction.disposition != resolution.disposition
            or correction.corrected_graph_sha256 != resolution.corrected_graph_sha256
        ):
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_adjudication_correction_membership_mismatch:"
                f"{item_id}"
            )
        for receipt in receipts:
            if (
                receipt.item_id != item_id
                or receipt.adjudicator_count != len(participants)
                or receipt.adjudication_protocol_sha256 != protocol_file_sha256
                or receipt.correction_protocol_sha256 != protocol_file_sha256
                or receipt.adjudication_payload_sha256 != resolution_binding.file_sha256
                or receipt.correction_payload_sha256 != correction_binding.file_sha256
                or receipt.disposition.value != resolution.disposition
                or receipt.corrected_graph_sha256 != resolution.corrected_graph_sha256
                or receipt.completed_at != resolution.completed_at
                or not math.isclose(
                    receipt.realized_person_minutes,
                    realized_minutes,
                    rel_tol=1e-12,
                    abs_tol=_COST_TOLERANCE,
                )
            ):
                raise DecisiveTrajectoryCompilerV1Error(
                    "decisive_trajectory_compiler_adjudication_receipt_raw_replay_mismatch:"
                    f"{item_id}:{receipt.receipt_sha256}"
                )
        verified_receipt_sha256s.update(receipt_sha256s)

    workflow_bindings.sort(key=lambda row: row.relative_path)
    payload = {
        "binding_version": "decisive-adjudication-package-source-binding-v1",
        "locator_sha256": locator.locator_sha256,
        "question_id": question_id,
        "package_sha256": package.package_sha256,
        "trust_registry_sha256": registry.registry_sha256,
        "adjudication_protocol_file_sha256": protocol_file_sha256,
        "package_artifact_binding": package_binding,
        "trust_registry_artifact_binding": registry_binding,
        "adjudication_protocol_artifact_binding": protocol_binding,
        "workflow_artifact_bindings": workflow_bindings,
        "verified_receipt_sha256s": sorted(verified_receipt_sha256s),
        "operator_trust_registry_hash_bound": True,
        "workflow_artifact_integrity_verified": True,
        "cryptographic_reviewer_identity_verified": False,
        "external_reviewer_expertise_verified": False,
        "trust_semantics": (
            "operator_declared_hash_bound_registry_not_cryptographic_identity_or_expertise_proof"
        ),
    }
    return AdjudicationReplayPackageSourceBindingV1.model_validate(
        {**payload, "binding_sha256": hash_canonical(payload)}
    )


def _require_real_certificate(
    *, certificate: _ProductionCertificate, identity: QuestionIdentityV1
) -> None:
    if isinstance(certificate, FinalConditionVerificationCertificateV7):
        source: VerificationCertificate | ConditionVerificationCertificateV6 = (
            certificate.source_certificate_v6
        )
        if (
            certificate.release_assessment.question_id != identity.question_id
            or source.production_stop_decision.outcome != "condition_gate_ready"
            or source.condition_confirmation_gate.status != "missing"
            or source.condition_confirmation_assessment is not None
            or source.adaptive_calibration_bundle_v2.label_source != "expert_adjudication"
            or not source.adaptive_calibration_bundle_v2.real_release_eligible
            or not source.adaptive_calibration_bundle_v2.independence_verified
            or source.adaptive_calibration_bundle_v2.selected is None
        ):
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_final_condition_provenance_incomplete"
            )
        release_pipeline_sha256 = source.release_assessment.pipeline_sha256
        adaptive_calibration_valid = True
    else:
        source = certificate
        adaptive = source.adaptive_calibration_bundle
        adaptive_calibration_valid = (
            adaptive is not None and adaptive.label_source == "expert_adjudication"
        )
        release_pipeline_sha256 = source.release_assessment.pipeline_sha256
    manifest = source.claim_manifest
    if (
        source.pipeline_verification.status != "matched"
        or source.pipeline_verification.computed is None
        or release_pipeline_sha256 != identity.pipeline_sha256
        or source.pipeline_verification.computed_pipeline_sha256 != identity.pipeline_sha256
    ):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_pipeline_provenance_incomplete"
        )
    if (
        manifest.get("question_id") != identity.question_id
        or manifest.get("population_id") != identity.population_id
        or manifest.get("domain") != identity.domain
    ):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_question_identity_mismatch"
        )
    assurance = source.corpus.get("provenance_assurance")
    metadata = source.corpus.get("metadata")
    if (
        not isinstance(assurance, Mapping)
        or assurance.get("status") != "source_replayed_native_grounding"
        or assurance.get("release_eligible") is not True
        or source.corpus.get("source_format") != "typed_evidence_grounding_package_json"
        or not isinstance(metadata, Mapping)
        or metadata.get("empirical_evidence") is False
        or metadata.get("purpose") == "offline_integration_test"
    ):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_real_source_provenance_required"
        )
    source_label = source.corpus.get("source_label")
    if isinstance(source_label, str) and source_label.casefold().startswith(
        ("embedded:", "fixture:", "simulation:", "simulation://", "synthetic:")
    ):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_nonempirical_source_marker_forbidden"
        )
    # Native-source replay proves integrity, but it cannot by itself turn an
    # explicitly simulated or fixture-labelled artifact into empirical evidence.
    # Inspect only loader-controlled top-level metadata fields: paper titles and
    # source-manifest contents may legitimately discuss simulations.
    for key, value in metadata.items():
        normalized_key = str(key).casefold().replace("-", "_")
        marker_absent = value is False or value is None or value == 0 or value == ""
        if normalized_key in _NONEMPIRICAL_METADATA_FLAGS and not marker_absent:
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_nonempirical_source_marker_forbidden"
            )
        if normalized_key in _NONEMPIRICAL_METADATA_SCOPES and isinstance(value, str):
            normalized_value = value.casefold().replace(" ", "_")
            if any(token in normalized_value for token in _NONEMPIRICAL_SCOPE_TOKENS):
                raise DecisiveTrajectoryCompilerV1Error(
                    "decisive_trajectory_compiler_nonempirical_source_marker_forbidden"
                )
    if source.complete_corpus_identity.membership_sha256 != identity.corpus_sha256:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_complete_corpus_identity_mismatch"
        )
    papers = sorted({row.paper_id for row in source.source_evidence_graph.publications})
    cohorts = sorted({row.cohort_id for row in source.source_evidence_graph.cohorts})
    if papers != identity.paper_ids or cohorts != identity.cohort_ids:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_evidence_membership_identity_mismatch"
        )
    item_risk = source.item_risk_scoring_receipt
    if (
        item_risk is None
        or item_risk.pipeline_verification.status != "matched"
        or item_risk.calibration_bundle.label_sources != ["expert_adjudication"]
        or not adaptive_calibration_valid
        or source.adaptive_policy_context is None
    ):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_real_calibration_provenance_required"
        )


def _require_certificate_workspace_context(
    *, certificate: _ProductionCertificate, config: AuditWorkspaceConfigV1
) -> None:
    if isinstance(certificate, FinalConditionVerificationCertificateV7):
        source: VerificationCertificate | ConditionVerificationCertificateV6 = (
            certificate.source_certificate_v6
        )
        adaptive_sha256: str | None = None
    else:
        source = certificate
        adaptive = source.adaptive_calibration_bundle
        adaptive_sha256 = None if adaptive is None else adaptive.bundle_sha256
    context = source.adaptive_policy_context
    item_risk = source.item_risk_scoring_receipt
    observed_common = (
        hash_canonical(source.claim_manifest),
        source.corpus_sha256,
        source.source_evidence_graph_sha256,
        source.complete_corpus_identity.membership_sha256,
        source.pipeline_verification.computed_pipeline_sha256,
        source.pipeline_verification.verification_sha256,
        None if context is None else context.policy_context_sha256,
        None if item_risk is None else item_risk.receipt_sha256,
    )
    expected_common = (
        config.claim_manifest_sha256,
        config.source_corpus_sha256,
        config.source_graph_sha256,
        config.complete_corpus_membership_sha256,
        config.pipeline_sha256,
        config.pipeline_verification_sha256,
        config.adaptive_policy_context_sha256,
        config.item_risk_scoring_receipt_sha256,
    )
    if observed_common != expected_common or (
        adaptive_sha256 is not None and adaptive_sha256 != config.adaptive_calibration_bundle_sha256
    ):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_certificate_workspace_context_mismatch"
        )


def _snapshot_verifier_certificate(
    *,
    source_root: Path,
    locator: VerifierCertificateLocatorV1,
) -> _CertificateSnapshot:
    path = _resolve_source_path(source_root, locator.relative_path)
    certificate, artifact_binding = _read_production_certificate(
        path,
        relative_path=locator.relative_path,
        artifact_kind="standalone_verification_certificate",
    )
    if (
        artifact_binding.file_sha256 != locator.expected_file_sha256
        or certificate.certificate_sha256 != locator.expected_certificate_sha256
    ):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_certificate_locator_identity_mismatch"
        )
    replay = freeze_question_replay_state_from_certificate(certificate)
    production = replay.production_binding
    if production is None or production.evaluated_active_action_item_id is not None:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_standalone_preselection_certificate_required"
        )
    payload = {
        "binding_version": CERTIFICATE_SOURCE_BINDING_VERSION,
        "locator_sha256": locator.locator_sha256,
        "artifact_binding": artifact_binding,
        "certificate_sha256": certificate.certificate_sha256,
        "replay_sha256": replay.replay_sha256,
    }
    binding = VerifierCertificateSourceBindingV1.model_validate(
        {**payload, "source_binding_sha256": hash_canonical(payload)}
    )
    return _CertificateSnapshot(
        binding=binding,
        replay_candidate=_ReplayCandidate(
            certificate=certificate,
            replay=replay,
            source_kind="standalone_verifier_certificate",
            source_container_sha256=binding.source_binding_sha256,
        ),
    )


def _event_semantics(occurrence: _EventOccurrence) -> dict[str, Any]:
    receipt = occurrence.receipt
    resolution = occurrence.resolution
    if receipt.provenance not in {"blinded_human", "benchmark_adjudication"}:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_human_adjudication_required"
        )
    if receipt.adjudicator_count < 2:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_two_expert_adjudicators_required"
        )
    if resolution.adjudication.cost_unit != "person_minutes":
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_realized_person_minutes_required"
        )
    corrected = receipt.disposition.value == "corrected"
    return {
        "item_id": receipt.item_id,
        "disposition": "corrected" if corrected else "confirmed",
        "completed_at": receipt.completed_at,
        "realized_minutes": receipt.realized_person_minutes,
        "adjudicator_count": receipt.adjudicator_count,
        "protocol_sha256": receipt.adjudication_protocol_sha256,
        "artifact_sha256": receipt.adjudication_payload_sha256,
        "correction_sha256": receipt.correction_payload_sha256 if corrected else None,
        "external_correction_protocol_sha256": receipt.correction_protocol_sha256,
        "external_correction_payload_sha256": receipt.correction_payload_sha256,
        "correction_disposition": receipt.disposition.value,
        "selected_estimate_before_sha256": (
            resolution.correction_provenance.selected_estimate_before_sha256
        ),
        "selected_estimate_after_sha256": (
            resolution.correction_provenance.selected_estimate_after_sha256
        ),
    }


def _freeze_event_from_semantics(semantics: Mapping[str, Any]) -> QuestionAuditEvent:
    return freeze_question_audit_event(
        item_id=str(semantics["item_id"]),
        disposition=AuditDisposition(str(semantics["disposition"])),
        completed_at=semantics["completed_at"],
        realized_minutes=float(semantics["realized_minutes"]),
        cost_basis=AuditCostBasis.REALIZED_HUMAN_MINUTES,
        adjudicator_count=int(semantics["adjudicator_count"]),
        protocol_sha256=str(semantics["protocol_sha256"]),
        artifact_sha256=str(semantics["artifact_sha256"]),
        correction_sha256=(
            None if semantics["correction_sha256"] is None else str(semantics["correction_sha256"])
        ),
    )


def _replay_semantic_projection(replay: QuestionReplayState) -> dict[str, Any]:
    return {
        "question_id": replay.question_id,
        "pipeline_sha256": replay.pipeline_sha256,
        "audit_sequence": replay.audit_sequence,
        "policy_inputs": replay.policy_inputs,
        "release_status": replay.release_status,
        "claim_classification": replay.claim_classification,
        "release_reasons": replay.release_reasons,
        "graph_sha256": replay.graph_sha256,
        "synthesis_sha256": replay.synthesis_sha256,
    }


def _freeze_adopted_replay(
    *, candidate: _ReplayCandidate, semantic_projection_sha256: str
) -> AdoptedReplayBindingV1:
    payload = {
        "binding_version": ADOPTED_REPLAY_VERSION,
        "audit_sequence": candidate.replay.audit_sequence,
        "source_certificate_sha256": candidate.certificate.certificate_sha256,
        "source_kind": candidate.source_kind,
        "source_container_sha256": candidate.source_container_sha256,
        "replay_sha256": candidate.replay.replay_sha256,
        "semantic_projection_sha256": semantic_projection_sha256,
    }
    return AdoptedReplayBindingV1.model_validate(
        {**payload, "binding_sha256": hash_canonical(payload)}
    )


def _required_policy_prefix_union(
    *, config: DecisiveEvaluationConfigV1, trajectory: QuestionTrajectoryV1
) -> list[tuple[str, ...]]:
    prefixes: set[tuple[str, ...]] = {()}
    for arm in required_policy_roster_v1():
        budgets: Sequence[float | None] = (
            (None,)
            if arm.arm_id == "audit_everything_upper_bound"
            else tuple(config.budgets_minutes_per_question)
        )
        for budget in budgets:
            frozen = _freeze_policy_question_v1(
                trajectory=trajectory,
                arm=arm,
                budget_minutes=budget,
                fixed_count=config.fixed_count,
                random_seed=config.random_seed,
            )
            prefixes.add(tuple(frozen.resolved_item_ids))
            for step in frozen.steps:
                prefixes.add(tuple(step.pre_audit_sequence))
                prefixes.add(tuple(step.post_audit_sequence))
    return sorted(prefixes, key=lambda row: (len(row), row))


def _replace_unfinalized_v5_condition_states(
    replay_candidates: Sequence[_ReplayCandidate],
) -> list[_ReplayCandidate]:
    """Drop gate-ready v5 condition states only when final v7 replaces that prefix.

    A v5 certificate can expose the provisional ``condition_dependent`` category, but
    it necessarily abstains before held-out confirmation.  Treating that category as
    the terminal five-way scientific decision would score an unopened condition gate.
    """

    grouped: dict[tuple[str, ...], list[_ReplayCandidate]] = {}
    for candidate in replay_candidates:
        grouped.setdefault(tuple(candidate.replay.audit_sequence), []).append(candidate)
    retained: list[_ReplayCandidate] = []
    for sequence, candidates in grouped.items():
        provisional_v5 = [
            row
            for row in candidates
            if isinstance(row.certificate, VerificationCertificate)
            and row.replay.claim_classification == "condition_dependent"
        ]
        final_v7 = [
            row
            for row in candidates
            if isinstance(row.certificate, FinalConditionVerificationCertificateV7)
        ]
        if provisional_v5 and not final_v7:
            raise DecisiveTrajectoryCompilerV1Error(
                f"decisive_trajectory_compiler_final_condition_v7_required:{sequence}"
            )
        retained.extend(row for row in candidates if row not in provisional_v5)
    return retained


def _require_single_frozen_v7_policy(
    replay_candidates: Sequence[_ReplayCandidate],
) -> None:
    """Prevent per-prefix substitution of the confirmation-aware calibration."""

    bundle_sha256s = {
        row.certificate.source_certificate_v6.adaptive_calibration_bundle_v2.bundle_sha256
        for row in replay_candidates
        if isinstance(row.certificate, FinalConditionVerificationCertificateV7)
    }
    if len(bundle_sha256s) > 1:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_condition_v7_calibration_changed_across_prefixes"
        )


def _bind_condition_artifacts(
    *,
    candidates: Sequence[_ReplayCandidate],
    declared_bindings: Sequence[ConditionSetSourceBindingV1],
    source_root: Path,
) -> tuple[dict[str, str], list[SourceArtifactBindingV1]]:
    """Bind every final-v7 condition state to an exact normalized artifact."""

    condition_certificates = {
        row.certificate.certificate_sha256: row.certificate
        for row in candidates
        if isinstance(row.certificate, FinalConditionVerificationCertificateV7)
        and row.replay.claim_classification == "condition_dependent"
    }
    declared = {row.certificate_sha256: row for row in declared_bindings}
    if set(declared) != set(condition_certificates):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_condition_set_binding_roster_mismatch"
        )

    condition_sha_by_certificate: dict[str, str] = {}
    artifact_binding_by_path: dict[str, SourceArtifactBindingV1] = {}
    for certificate_sha256, certificate in sorted(condition_certificates.items()):
        declaration = declared[certificate_sha256]
        artifact, artifact_binding = _snapshot_condition_set_artifact(
            source_root=source_root,
            binding=declaration,
        )
        expected = _normalized_condition_set_from_v7(certificate)
        if artifact != expected:
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_condition_set_certificate_mismatch"
            )
        prior = artifact_binding_by_path.get(artifact_binding.relative_path)
        if prior is not None and prior != artifact_binding:
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_condition_artifact_path_disagrees"
            )
        artifact_binding_by_path[artifact_binding.relative_path] = artifact_binding
        condition_sha_by_certificate[certificate_sha256] = artifact.artifact_sha256
    return condition_sha_by_certificate, sorted(
        artifact_binding_by_path.values(), key=lambda row: row.relative_path
    )


def _compile_question(
    *,
    config: DecisiveEvaluationConfigV1,
    identity: QuestionIdentityV1,
    provenance: DecisivePolicyInputProvenanceV1,
    source: QuestionTrajectorySourceV1,
    source_root: Path,
) -> tuple[QuestionTrajectoryV1, QuestionTrajectoryCompilationReceiptV1]:
    if identity.split is not StudySplit.EVALUATION or source.question_id != identity.question_id:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_non_evaluation_question"
        )
    snapshots = [
        _snapshot_workspace(
            workspace=_resolve_source_path(source_root, locator.relative_path),
            locator=locator,
        )
        for locator in source.workspaces
    ]
    certificate_snapshots = [
        _snapshot_verifier_certificate(source_root=source_root, locator=locator)
        for locator in source.verifier_certificates
    ]
    configs = {row.config.config_sha256 for row in snapshots}
    if len(configs) != 1:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_question_workspace_config_mixed"
        )
    source_config = snapshots[0].config
    if (
        source_config.pipeline_sha256 != identity.pipeline_sha256
        or source_config.budget_minutes + _COST_TOLERANCE < max(config.budgets_minutes_per_question)
        or source_config.item_risk_scoring_receipt_sha256 is None
    ):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_workspace_scientific_context_invalid"
        )

    replay_candidates = [row for snapshot in snapshots for row in snapshot.replay_candidates]
    replay_candidates.extend(row.replay_candidate for row in certificate_snapshots)
    if not replay_candidates:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_replay_candidates_empty"
        )
    for candidate in replay_candidates:
        _require_real_certificate(certificate=candidate.certificate, identity=identity)
        _require_certificate_workspace_context(
            certificate=candidate.certificate,
            config=source_config,
        )
        certificate_source = (
            candidate.certificate.source_certificate_v6
            if isinstance(candidate.certificate, FinalConditionVerificationCertificateV7)
            else candidate.certificate
        )
        item_risk = certificate_source.item_risk_scoring_receipt
        legacy_adaptive = (
            None
            if isinstance(candidate.certificate, FinalConditionVerificationCertificateV7)
            else candidate.certificate.adaptive_calibration_bundle
        )
        if (
            item_risk is None
            or (item_risk.receipt_sha256 != source_config.item_risk_scoring_receipt_sha256)
            or (
                legacy_adaptive is not None
                and legacy_adaptive.bundle_sha256
                != source_config.adaptive_calibration_bundle_sha256
            )
        ):
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_workspace_certificate_calibration_mismatch"
            )

    replay_candidates = _replace_unfinalized_v5_condition_states(replay_candidates)
    _require_single_frozen_v7_policy(replay_candidates)
    condition_by_certificate, condition_artifact_bindings = _bind_condition_artifacts(
        candidates=replay_candidates,
        declared_bindings=source.condition_set_bindings,
        source_root=source_root,
    )
    by_sequence: dict[tuple[str, ...], list[_ReplayCandidate]] = {}
    for candidate in replay_candidates:
        by_sequence.setdefault(tuple(candidate.replay.audit_sequence), []).append(candidate)
    adopted_by_sequence: dict[tuple[str, ...], _ReplayCandidate] = {}
    semantic_hash_by_sequence: dict[tuple[str, ...], str] = {}
    condition_by_replay: dict[str, str] = {}
    for sequence, candidates in by_sequence.items():
        semantic_hashes = {
            hash_canonical(_replay_semantic_projection(row.replay)) for row in candidates
        }
        if len(semantic_hashes) != 1:
            raise DecisiveTrajectoryCompilerV1Error(
                f"decisive_trajectory_compiler_duplicate_prefix_disagrees:{identity.question_id}:{sequence}"
            )
        chosen = min(candidates, key=lambda row: row.certificate.certificate_sha256)
        adopted_by_sequence[sequence] = chosen
        semantic_hash_by_sequence[sequence] = next(iter(semantic_hashes))
        condition_hashes: set[str] = set()
        for row in candidates:
            if row.replay.claim_classification == "condition_dependent":
                if not isinstance(row.certificate, FinalConditionVerificationCertificateV7):
                    raise DecisiveTrajectoryCompilerV1Error(
                        "decisive_trajectory_compiler_final_condition_v7_required"
                    )
                condition_hash = condition_by_certificate.get(row.certificate.certificate_sha256)
                if condition_hash is None:
                    raise DecisiveTrajectoryCompilerV1Error(
                        "decisive_trajectory_compiler_condition_set_binding_missing"
                    )
                condition_hashes.add(condition_hash)
        if len(condition_hashes) > 1:
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_duplicate_prefix_condition_set_disagrees"
            )
        if condition_hashes:
            condition_by_replay[chosen.replay.replay_sha256] = next(iter(condition_hashes))

    occurrences_by_item: dict[str, list[_EventOccurrence]] = {}
    for snapshot in snapshots:
        for occurrence in snapshot.event_occurrences:
            occurrences_by_item.setdefault(occurrence.receipt.item_id, []).append(occurrence)
    if not occurrences_by_item:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_completed_adjudications_empty"
        )
    adjudication_package_binding = _snapshot_adjudication_replay_package(
        source_root=source_root,
        locator=source.adjudication_replay_package,
        question_id=identity.question_id,
        occurrences_by_item=occurrences_by_item,
    )
    events: list[QuestionAuditEvent] = []
    event_bindings: list[CompiledAuditEventBindingV1] = []
    for item_id, occurrences in sorted(occurrences_by_item.items()):
        if not any(row.receipt.provenance == "blinded_human" for row in occurrences):
            raise DecisiveTrajectoryCompilerV1Error(
                f"decisive_trajectory_compiler_blinded_human_origin_missing:{item_id}"
            )
        semantics = [_event_semantics(row) for row in occurrences]
        normalized = [
            {**row, "completed_at": _canonical_time(row["completed_at"])} for row in semantics
        ]
        if len({hash_canonical(row) for row in normalized}) != 1:
            raise DecisiveTrajectoryCompilerV1Error(
                f"decisive_trajectory_compiler_repeated_adjudication_disagrees:{item_id}"
            )
        event = _freeze_event_from_semantics(semantics[0])
        events.append(event)
        binding_payload = {
            "binding_version": "decisive-compiled-audit-event-binding-v1",
            "item_id": item_id,
            "event_sha256": event.event_sha256,
            "source_receipt_sha256s": sorted({row.receipt.receipt_sha256 for row in occurrences}),
            "source_resolution_result_sha256s": sorted(
                {row.resolution.result_sha256 for row in occurrences}
            ),
        }
        event_bindings.append(
            CompiledAuditEventBindingV1.model_validate(
                {**binding_payload, "binding_sha256": hash_canonical(binding_payload)}
            )
        )

    baseline = adopted_by_sequence.get(())
    if baseline is None:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_baseline_prefix_missing"
        )
    baseline_ids = [row.item_id for row in baseline.replay.policy_inputs]
    if baseline_ids != sorted(occurrences_by_item):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_audit_event_membership_incomplete"
        )
    canonical_order = {row.item_id: row.canonical_order for row in baseline.replay.policy_inputs}
    canonical_full = tuple(sorted(baseline_ids, key=canonical_order.__getitem__))
    if canonical_full not in adopted_by_sequence:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_exhaustive_canonical_prefix_missing"
        )
    total_minutes = sum(event.realized_minutes for event in events)
    if source_config.budget_minutes + _COST_TOLERANCE < total_minutes:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_source_budget_below_exhaustive_realized_cost"
        )

    all_states = [
        row.replay
        for _, row in sorted(adopted_by_sequence.items(), key=lambda item: (len(item[0]), item[0]))
    ]
    provisional_conditions = {
        state.replay_sha256: condition_by_replay[state.replay_sha256]
        for state in all_states
        if state.claim_classification == "condition_dependent"
    }
    provisional = freeze_question_trajectory_v1(
        question_identity=identity,
        evidence_kind=BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED,
        policy_input_provenance=provenance,
        audit_events=events,
        replay_states=all_states,
        condition_set_artifact_sha256_by_replay_sha256=provisional_conditions,
    )
    try:
        required = _required_policy_prefix_union(config=config, trajectory=provisional)
    except (KeyError, ValueError) as exc:
        raise DecisiveTrajectoryCompilerV1Error(
            f"decisive_trajectory_compiler_policy_visited_prefix_missing:{identity.question_id}"
        ) from exc
    missing = [row for row in required if row not in adopted_by_sequence]
    if missing:
        raise DecisiveTrajectoryCompilerV1Error(
            f"decisive_trajectory_compiler_policy_visited_prefix_missing:{identity.question_id}:{missing}"
        )
    final_states = [adopted_by_sequence[row].replay for row in required]
    final_conditions = {
        state.replay_sha256: condition_by_replay[state.replay_sha256]
        for state in final_states
        if state.claim_classification == "condition_dependent"
    }
    trajectory = freeze_question_trajectory_v1(
        question_identity=identity,
        evidence_kind=BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED,
        policy_input_provenance=provenance,
        audit_events=events,
        replay_states=final_states,
        condition_set_artifact_sha256_by_replay_sha256=final_conditions,
    )
    # Re-run the roster after dropping all unvisited states.  This proves the compact
    # trajectory is sufficient, rather than merely a projection of an overcomplete set.
    if _required_policy_prefix_union(config=config, trajectory=trajectory) != required:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_compact_union_external_replay_mismatch"
        )

    adopted = [
        _freeze_adopted_replay(
            candidate=adopted_by_sequence[sequence],
            semantic_projection_sha256=semantic_hash_by_sequence[sequence],
        )
        for sequence in required
    ]
    workspace_bindings = sorted(
        [row.binding for row in snapshots], key=lambda row: row.workspace_source_sha256
    )
    receipt_payload = {
        "receipt_version": QUESTION_RECEIPT_VERSION,
        "question_id": identity.question_id,
        "question_identity_sha256": identity.identity_sha256,
        "source_roster_entry_sha256": source.source_sha256,
        "workspace_bindings": workspace_bindings,
        "adjudication_replay_package_binding": adjudication_package_binding,
        "verifier_certificate_bindings": sorted(
            [row.binding for row in certificate_snapshots],
            key=lambda row: row.source_binding_sha256,
        ),
        "condition_set_artifact_bindings": condition_artifact_bindings,
        "available_prefixes": [
            list(row) for row in sorted(by_sequence, key=lambda row: (len(row), row))
        ],
        "required_policy_visited_prefixes": [list(row) for row in required],
        "adopted_replays": adopted,
        "audit_event_bindings": sorted(event_bindings, key=lambda row: row.item_id),
        "total_realized_person_minutes": total_minutes,
        "trajectory_sha256": trajectory.trajectory_sha256,
        "real_source_provenance_verified": True,
        "operator_declared_expert_workflow_replayed": True,
        "external_reviewer_identity_or_expertise_proven": False,
        "evaluation_reference_labels_opened": False,
        "scientific_claim_authority": False,
    }
    receipt = QuestionTrajectoryCompilationReceiptV1.model_validate(
        {**receipt_payload, "receipt_sha256": hash_canonical(receipt_payload)}
    )
    return trajectory, receipt


def _verify_source_roster_file(
    *,
    source_roster: DecisiveTrajectorySourceRosterV1,
    source_roster_path: Path,
) -> str:
    raw = _read_regular_no_follow(source_roster_path, label="source_roster")
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("not_object")
        parsed = DecisiveTrajectorySourceRosterV1.model_validate(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_source_roster_file_invalid"
        ) from exc
    if parsed != source_roster:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_source_roster_argument_file_mismatch"
        )
    return _sha256_bytes(raw)


def _require_fit_stage_projection(
    *,
    split_manifest: DecisiveSplitManifestV1,
    development_receipt: FitStageReceiptV1,
    calibration_receipt: FitStageReceiptV1,
) -> None:
    rows_by_split = {
        split: [row for row in split_manifest.identities if row.split is split]
        for split in StudySplit
    }
    expected = (
        (development_receipt, "development_optimizer_fit", StudySplit.DEVELOPMENT),
        (
            calibration_receipt,
            "calibration_policy_and_threshold_freeze",
            StudySplit.CALIBRATION,
        ),
    )
    for receipt, stage, split in expected:
        rows = rows_by_split[split]
        if (
            receipt.stage.value != stage
            or receipt.question_ids != sorted(row.question_id for row in rows)
            or receipt.claim_ids != sorted(row.claim_id for row in rows)
            or receipt.paper_ids != sorted({paper_id for row in rows for paper_id in row.paper_ids})
            or receipt.cohort_ids
            != sorted({cohort_id for row in rows for cohort_id in row.cohort_ids})
            or receipt.pipeline_sha256 != split_manifest.pipeline_sha256
            or receipt.input_manifest_sha256 != split_manifest.manifest_sha256
            or receipt.labels_opened_by_this_stage is not True
            or receipt.evaluation_labels_opened is not False
        ):
            raise DecisiveTrajectoryCompilerV1Error(
                "decisive_trajectory_compiler_fit_stage_projection_mismatch"
            )
    if (
        development_receipt.completed_at > calibration_receipt.completed_at
        or calibration_receipt.frozen_threshold_or_bounds_sha256 is None
    ):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_fit_stage_order_or_freeze_invalid"
        )


def compile_decisive_trajectory_bundle_v1(
    *,
    config: DecisiveEvaluationConfigV1,
    split_manifest: DecisiveSplitManifestV1,
    development_receipt: FitStageReceiptV1,
    calibration_receipt: FitStageReceiptV1,
    source_roster: DecisiveTrajectorySourceRosterV1,
    source_roster_path: Path,
    source_root: Path,
    repository_root: Path,
    compiled_at: datetime,
) -> DecisiveTrajectoryCompilationResultV1:
    """Compile and hash-bind the exact real policy-visited trajectory union."""

    source_roster_file_sha256 = _verify_source_roster_file(
        source_roster=source_roster,
        source_roster_path=source_roster_path,
    )
    if source_roster.split_manifest_sha256 != split_manifest.manifest_sha256:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_source_split_manifest_mismatch"
        )
    if development_receipt.label_source != "expert_adjudication" or (
        calibration_receipt.label_source != "expert_adjudication"
    ):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_real_fit_stage_labels_required"
        )
    _require_fit_stage_projection(
        split_manifest=split_manifest,
        development_receipt=development_receipt,
        calibration_receipt=calibration_receipt,
    )
    provenance = freeze_decisive_policy_input_provenance_v1(
        development_receipt=development_receipt,
        calibration_receipt=calibration_receipt,
    )
    evaluation_identities = {
        row.question_id: row
        for row in split_manifest.identities
        if row.split is StudySplit.EVALUATION
    }
    sources = {row.question_id: row for row in source_roster.questions}
    if set(sources) != set(evaluation_identities):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_evaluation_roster_incomplete"
        )
    trajectories: list[QuestionTrajectoryV1] = []
    question_receipts: list[QuestionTrajectoryCompilationReceiptV1] = []
    for question_id in sorted(evaluation_identities):
        trajectory, question_receipt = _compile_question(
            config=config,
            identity=evaluation_identities[question_id],
            provenance=provenance,
            source=sources[question_id],
            source_root=source_root,
        )
        trajectories.append(trajectory)
        question_receipts.append(question_receipt)
    bundle = freeze_trajectory_bundle_v1(
        split_manifest=split_manifest,
        evidence_kind=BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED,
        trajectories=trajectories,
    )
    component_sha256 = compute_decisive_trajectory_compiler_component_sha256_v1(repository_root)
    compilation_lineage_identity = freeze_decisive_compilation_lineage_identity_v1(
        compiler_component_sha256=component_sha256,
        config_sha256=config.config_sha256,
        split_manifest_sha256=split_manifest.manifest_sha256,
        development_receipt_sha256=development_receipt.receipt_sha256,
        calibration_receipt_sha256=calibration_receipt.receipt_sha256,
        source_roster_file_sha256=source_roster_file_sha256,
        source_roster_sha256=source_roster.source_roster_sha256,
        trajectory_bundle_sha256=bundle.bundle_sha256,
        trajectory_membership_sha256=bundle.trajectory_membership_sha256,
        evaluation_question_ids=[row.question_id for row in question_receipts],
        question_receipt_sha256s=[row.receipt_sha256 for row in question_receipts],
        adjudication_package_binding_sha256s=[
            row.adjudication_replay_package_binding.binding_sha256 for row in question_receipts
        ],
    )
    receipt_payload = {
        "compilation_version": COMPILATION_VERSION,
        "compiled_at": _canonical_time(compiled_at),
        "compiler_component_sha256": component_sha256,
        "config_sha256": config.config_sha256,
        "split_manifest_sha256": split_manifest.manifest_sha256,
        "development_receipt_sha256": development_receipt.receipt_sha256,
        "calibration_receipt_sha256": calibration_receipt.receipt_sha256,
        "policy_input_provenance_sha256": provenance.provenance_sha256,
        "source_roster_file_sha256": source_roster_file_sha256,
        "source_roster_sha256": source_roster.source_roster_sha256,
        "question_receipts": sorted(question_receipts, key=lambda row: row.question_id),
        "trajectory_bundle_sha256": bundle.bundle_sha256,
        "trajectory_membership_sha256": bundle.trajectory_membership_sha256,
        "compilation_lineage_identity": compilation_lineage_identity,
        "evidence_kind": "real_expert_adjudicated",
        "union_semantics": (
            "exact_union_of_prefixes_visited_by_the_prespecified_decisive_policy_roster"
        ),
        "realized_cost_semantics": (
            "total_person_minutes_across_all_reviewers_and_final_adjudication"
        ),
        "source_locator_portability": (
            "local_relative_paths_replayed_to_exact_content_and_semantic_hashes"
        ),
        "evaluation_reference_labels_opened": False,
        "real_empirical_candidate": True,
        "scientific_claim_authority": False,
        "claim_release_authority": False,
    }
    receipt = DecisiveTrajectoryCompilationReceiptV1.model_validate(
        {**receipt_payload, "compilation_sha256": hash_canonical(receipt_payload)}
    )
    result_payload = {
        "result_version": "decisive-trajectory-compilation-result-v1",
        "trajectory_bundle": bundle,
        "compilation_receipt": receipt,
    }
    return DecisiveTrajectoryCompilationResultV1.model_validate(
        {**result_payload, "result_sha256": hash_canonical(result_payload)}
    )


def replay_decisive_trajectory_compilation_v1(
    *,
    expected: DecisiveTrajectoryCompilationResultV1,
    config: DecisiveEvaluationConfigV1,
    split_manifest: DecisiveSplitManifestV1,
    development_receipt: FitStageReceiptV1,
    calibration_receipt: FitStageReceiptV1,
    source_roster: DecisiveTrajectorySourceRosterV1,
    source_roster_path: Path,
    source_root: Path,
    repository_root: Path,
) -> DecisiveTrajectoryCompilationResultV1:
    """Externally rebuild a saved compilation from its exact source workspaces."""

    replayed = compile_decisive_trajectory_bundle_v1(
        config=config,
        split_manifest=split_manifest,
        development_receipt=development_receipt,
        calibration_receipt=calibration_receipt,
        source_roster=source_roster,
        source_roster_path=source_roster_path,
        source_root=source_root,
        repository_root=repository_root,
        compiled_at=expected.compilation_receipt.compiled_at,
    )
    if replayed != expected:
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_external_replay_mismatch"
        )
    return replayed


def write_decisive_trajectory_compilation_v1(
    result: DecisiveTrajectoryCompilationResultV1,
    *,
    bundle_path: Path,
    receipt_path: Path,
) -> None:
    """Write separate decisive-consumable bundle and compiler receipt artifacts."""

    if bundle_path.resolve(strict=False) == receipt_path.resolve(strict=False):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_output_paths_must_differ"
        )
    if any(path.exists() or path.is_symlink() for path in (bundle_path, receipt_path)):
        raise DecisiveTrajectoryCompilerV1Error(
            "decisive_trajectory_compiler_output_must_not_exist"
        )
    # Commit the self-contained result first and the directly consumable trajectory
    # last.  An interrupted two-file write can therefore leave a receipt without a
    # bundle, but never a bare decisive-evaluation bundle without its full lineage.
    atomic_write_json(receipt_path, result, force=False)
    atomic_write_json(bundle_path, result.trajectory_bundle, force=False)


__all__ = [
    "AdjudicationReplayPackageLocatorV1",
    "AdjudicationReplayPackageSourceBindingV1",
    "ConditionSetSourceBindingV1",
    "DecisiveTrajectoryCompilationReceiptV1",
    "DecisiveTrajectoryCompilationResultV1",
    "DecisiveTrajectoryCompilerV1Error",
    "DecisiveTrajectorySourceRosterV1",
    "NormalizedConditionSetArtifactV1",
    "QuestionTrajectoryCompilationReceiptV1",
    "QuestionTrajectorySourceV1",
    "TransactionalWorkspaceLocatorV1",
    "VerifierCertificateLocatorV1",
    "compile_decisive_trajectory_bundle_v1",
    "compute_decisive_trajectory_compiler_component_sha256_v1",
    "freeze_adjudication_replay_package_locator_v1",
    "freeze_condition_set_source_binding_v1",
    "freeze_decisive_trajectory_source_roster_v1",
    "freeze_normalized_condition_set_artifact_v1",
    "freeze_question_trajectory_source_v1",
    "freeze_transactional_workspace_locator_v1",
    "freeze_verifier_certificate_locator_v1",
    "replay_decisive_trajectory_compilation_v1",
    "write_decisive_trajectory_compilation_v1",
]
