"""Transactional, compare-and-swap workspace for release-capable audit advances.

The scientific verifier and sequential audit ledger are intentionally pure.  This
module adds the missing operator boundary: one private canonical workspace, one
locked state pointer, explicit predecessor expectations, typed external cost
receipts, and an atomic resolve/checkpoint-to-certificate commit.  It performs no
network or provider work.

The pointer update is the commit point.  A process writes a durable ``pending``
marker before constructing the next generation.  If it dies before the final
``committed`` marker, the workspace is ambiguous and all later mutations fail
closed rather than guessing whether a human-cost-bearing transition committed.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.adaptive_calibration import AdaptiveCalibrationBundle
from literature_multiverse.audit_session import CorrectionDisposition
from literature_multiverse.certificate import VerificationCertificate, write_certificate_artifacts
from literature_multiverse.evidence_graph import EvidenceGraph
from literature_multiverse.item_risk_artifacts import ItemRiskScoringRunReceipt
from literature_multiverse.lineage import atomic_write_json, hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.pipeline_fingerprint import PipelineFingerprint
from literature_multiverse.sequential_verification import (
    SequentialActiveCostCheckpointResult,
    SequentialResolutionResult,
    SequentialStateExpectation,
    SequentialVerificationState,
    checkpoint_selected_audit_cost,
    freeze_selected_adjudication,
    freeze_state_expectation,
    resolve_selected_audit_candidate,
    resume_sequential_verification_state,
)
from literature_multiverse.verifier import (
    ClaimManifest,
    CorpusLoadResult,
    compute_candidate_runner_sha256,
    compute_synthesis_runner_sha256,
    load_corpus,
    prepare_verification_scientific_state,
    run_verification,
    sequential_candidates_from_prepared_state,
)

_COST_TOLERANCE = 1e-9


class TransactionalAuditWorkspaceError(ValueError):
    """A workspace mutation or replay contract failed closed."""


class StaleTransactionalAuditStateError(TransactionalAuditWorkspaceError):
    """The caller attempted to advance a superseded canonical state."""


class AmbiguousTransactionalAuditWorkspaceError(TransactionalAuditWorkspaceError):
    """A prior transaction cannot safely be classified as committed or absent."""


def _sha256(value: str, field_name: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"invalid_sha256:{field_name}")
    return value


def _optional_sha256(value: str | None, field_name: str) -> str | None:
    if value is not None:
        _sha256(value, field_name)
    return value


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name}_requires_timezone")
    return value


def _finite_cost(value: float, field_name: str, *, positive: bool = False) -> float:
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{field_name}_must_be_finite_{qualifier}")
    return value


def _relative_path(value: str, field_name: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or value.startswith("./")
        or path.as_posix() != value
    ):
        raise ValueError(f"{field_name}_must_be_normalized_relative_path")
    return value


class AuditWorkspaceConfigV1(ContractModel):
    """Frozen scientific and calibration identity shared by every generation."""

    config_version: Literal["transactional-audit-workspace-config-v1"] = (
        "transactional-audit-workspace-config-v1"
    )
    workspace_id: Annotated[str, Field(pattern=r"^audit-workspace-[0-9a-f]{16}$")]
    claim_manifest_sha256: str
    source_corpus_sha256: str
    source_corpus_payload_sha256: str
    source_graph_sha256: str
    complete_corpus_membership_sha256: str
    budget_minutes: Annotated[float, Field(ge=0)]
    cost_unit: Literal["person_minutes"] = "person_minutes"
    pipeline_sha256: str
    pipeline_verification_sha256: str
    adaptive_calibration_bundle_sha256: str
    adaptive_policy_context_sha256: str
    item_risk_scoring_receipt_sha256: str | None = None
    config_sha256: str

    @field_validator(
        "claim_manifest_sha256",
        "source_corpus_sha256",
        "source_corpus_payload_sha256",
        "source_graph_sha256",
        "complete_corpus_membership_sha256",
        "pipeline_sha256",
        "pipeline_verification_sha256",
        "adaptive_calibration_bundle_sha256",
        "adaptive_policy_context_sha256",
        "config_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("item_risk_scoring_receipt_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return _optional_sha256(value, "item_risk_scoring_receipt_sha256")

    @field_validator("budget_minutes")
    @classmethod
    def validate_budget(cls, value: float) -> float:
        return _finite_cost(value, "audit_workspace_budget_minutes")

    @model_validator(mode="after")
    def validate_config(self) -> AuditWorkspaceConfigV1:
        identity_payload = self.model_dump(mode="json", exclude={"workspace_id", "config_sha256"})
        expected_id = f"audit-workspace-{hash_canonical(identity_payload)[:16]}"
        if self.workspace_id != expected_id:
            raise ValueError("audit_workspace_id_mismatch")
        payload = self.model_dump(mode="json", exclude={"config_sha256"})
        if hash_canonical(payload) != self.config_sha256:
            raise ValueError("audit_workspace_config_hash_mismatch")
        return self


class AuditActionAuthorizationV1(ContractModel):
    """Pre-liability proof that the full verifier replay authorized one action."""

    authorization_version: Literal["transactional-audit-action-authorization-v1"] = (
        "transactional-audit-action-authorization-v1"
    )
    workspace_config_sha256: str
    generation: Annotated[int, Field(ge=0)]
    state_expectation: SequentialStateExpectation
    certificate_sha256: str
    session_id: Annotated[str, Field(min_length=1)]
    step: Annotated[int, Field(ge=1)]
    item_id: Annotated[str, Field(min_length=1)]
    action_packet_sha256: str
    issued_at: datetime
    release_blocked: Literal[True] = True
    authorization_sha256: str

    @field_validator(
        "workspace_config_sha256",
        "certificate_sha256",
        "action_packet_sha256",
        "authorization_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("issued_at")
    @classmethod
    def validate_issued_at(cls, value: datetime) -> datetime:
        return _aware(value, "audit_action_authorization_issued_at")

    @model_validator(mode="after")
    def validate_authorization(self) -> AuditActionAuthorizationV1:
        if self.state_expectation.active_action_packet_sha256 != self.action_packet_sha256:
            raise ValueError("audit_authorization_expectation_action_mismatch")
        payload = self.model_dump(mode="json", exclude={"authorization_sha256"})
        if hash_canonical(payload) != self.authorization_sha256:
            raise ValueError("audit_action_authorization_hash_mismatch")
        return self


class AuditWorkspacePointerV1(ContractModel):
    """The one canonical compare-and-swap pointer for a workspace."""

    pointer_version: Literal["transactional-audit-workspace-pointer-v1"] = (
        "transactional-audit-workspace-pointer-v1"
    )
    workspace_id: Annotated[str, Field(pattern=r"^audit-workspace-[0-9a-f]{16}$")]
    workspace_config_sha256: str
    generation: Annotated[int, Field(ge=0)]
    generation_path: str
    state_path: str
    certificate_path: str
    predecessor_pointer_sha256: str | None = None
    transition_kind: Literal["initialized", "checkpointed", "adjudicated"]
    transition_receipt_sha256: str | None = None
    transaction_receipt_sha256s: list[str]
    state_expectation: SequentialStateExpectation
    certificate_sha256: str
    certificate_status: Literal["released", "abstained"]
    authorization: AuditActionAuthorizationV1 | None = None
    pointer_sha256: str

    @field_validator(
        "workspace_config_sha256",
        "certificate_sha256",
        "pointer_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("predecessor_pointer_sha256", "transition_receipt_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None, info: Any) -> str | None:
        return _optional_sha256(value, info.field_name)

    @field_validator("transaction_receipt_sha256s")
    @classmethod
    def validate_receipt_hashes(cls, value: list[str]) -> list[str]:
        for item in value:
            _sha256(item, "transaction_receipt_sha256")
        if len(value) != len(set(value)):
            raise ValueError("audit_workspace_transaction_receipt_duplicate")
        return value

    @field_validator("generation_path", "state_path", "certificate_path")
    @classmethod
    def validate_path(cls, value: str, info: Any) -> str:
        return _relative_path(value, info.field_name)

    @model_validator(mode="after")
    def validate_pointer(self) -> AuditWorkspacePointerV1:
        generation_prefix = f"generations/{self.generation:06d}-"
        if not self.generation_path.startswith(generation_prefix):
            raise ValueError("audit_workspace_generation_path_mismatch")
        if self.state_path != f"{self.generation_path}/sequential-audit-state.json":
            raise ValueError("audit_workspace_state_path_mismatch")
        if self.certificate_path != f"{self.generation_path}/verification-certificate.json":
            raise ValueError("audit_workspace_certificate_path_mismatch")
        if self.state_expectation.state_sha256[:16] not in self.generation_path:
            raise ValueError("audit_workspace_generation_state_hash_missing")
        if self.generation == 0:
            if (
                self.transition_kind != "initialized"
                or self.predecessor_pointer_sha256 is not None
                or self.transition_receipt_sha256 is not None
                or self.transaction_receipt_sha256s
            ):
                raise ValueError("audit_workspace_genesis_pointer_invalid")
        elif (
            self.transition_kind == "initialized"
            or self.predecessor_pointer_sha256 is None
            or self.transition_receipt_sha256 is None
            or not self.transaction_receipt_sha256s
            or self.transaction_receipt_sha256s[-1] != self.transition_receipt_sha256
        ):
            raise ValueError("audit_workspace_transition_pointer_invalid")
        action_sha = self.state_expectation.active_action_packet_sha256
        if (self.authorization is None) != (action_sha is None):
            raise ValueError("audit_workspace_pointer_authorization_presence_mismatch")
        if self.authorization is not None and (
            self.authorization.workspace_config_sha256 != self.workspace_config_sha256
            or self.authorization.generation != self.generation
            or self.authorization.state_expectation != self.state_expectation
            or self.authorization.certificate_sha256 != self.certificate_sha256
            or self.authorization.action_packet_sha256 != action_sha
        ):
            raise ValueError("audit_workspace_pointer_authorization_mismatch")
        if self.certificate_status == "released" and self.authorization is not None:
            raise ValueError("released_audit_workspace_cannot_authorize_action")
        payload = self.model_dump(mode="json", exclude={"pointer_sha256"})
        if hash_canonical(payload) != self.pointer_sha256:
            raise ValueError("audit_workspace_pointer_hash_mismatch")
        return self


class AuditAdjudicationCostReceiptV1(ContractModel):
    """Typed external adjudication and realized-person-minute receipt."""

    receipt_version: Literal["transactional-audit-adjudication-cost-v1"] = (
        "transactional-audit-adjudication-cost-v1"
    )
    workspace_config_sha256: str
    predecessor_pointer_sha256: str
    authorization_sha256: str
    predecessor_expectation_sha256: str
    predecessor_state_sha256: str
    session_id: Annotated[str, Field(min_length=1)]
    step: Annotated[int, Field(ge=1)]
    item_id: Annotated[str, Field(min_length=1)]
    action_packet_sha256: str
    disposition: CorrectionDisposition
    corrected_graph_sha256: str | None = None
    provenance: Literal["blinded_human", "benchmark_adjudication"]
    adjudicator_count: Annotated[int, Field(gt=0)]
    adjudication_protocol_sha256: str
    adjudication_payload_sha256: str
    correction_protocol_sha256: str
    correction_payload_sha256: str
    completed_at: datetime
    realized_person_minutes: Annotated[float, Field(gt=0)]
    cost_unit: Literal["person_minutes"] = "person_minutes"
    receipt_sha256: str

    @field_validator(
        "workspace_config_sha256",
        "predecessor_pointer_sha256",
        "authorization_sha256",
        "predecessor_expectation_sha256",
        "predecessor_state_sha256",
        "action_packet_sha256",
        "adjudication_protocol_sha256",
        "adjudication_payload_sha256",
        "correction_protocol_sha256",
        "correction_payload_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("corrected_graph_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return _optional_sha256(value, "corrected_graph_sha256")

    @field_validator("completed_at")
    @classmethod
    def validate_completed_at(cls, value: datetime) -> datetime:
        return _aware(value, "audit_adjudication_completed_at")

    @field_validator("realized_person_minutes")
    @classmethod
    def validate_realized_cost(cls, value: float) -> float:
        return _finite_cost(value, "audit_adjudication_realized_person_minutes", positive=True)

    @model_validator(mode="after")
    def validate_receipt(self) -> AuditAdjudicationCostReceiptV1:
        if self.disposition is CorrectionDisposition.CORRECTED:
            if self.corrected_graph_sha256 is None:
                raise ValueError("corrected_adjudication_requires_graph_hash")
        elif self.corrected_graph_sha256 is not None:
            raise ValueError("no_change_adjudication_forbids_graph_hash")
        if self.provenance == "blinded_human" and self.adjudicator_count < 2:
            raise ValueError("blinded_human_adjudication_requires_two_adjudicators")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("audit_adjudication_cost_receipt_hash_mismatch")
        return self


class AuditActiveCostCheckpointReceiptV1(ContractModel):
    """Typed cumulative realized-cost observation with no outcome fields."""

    receipt_version: Literal["transactional-audit-active-cost-checkpoint-v1"] = (
        "transactional-audit-active-cost-checkpoint-v1"
    )
    workspace_config_sha256: str
    predecessor_pointer_sha256: str
    authorization_sha256: str
    predecessor_expectation_sha256: str
    predecessor_state_sha256: str
    session_id: Annotated[str, Field(min_length=1)]
    step: Annotated[int, Field(ge=1)]
    item_id: Annotated[str, Field(min_length=1)]
    action_packet_sha256: str
    cumulative_active_person_minutes: Annotated[float, Field(ge=0)]
    cost_unit: Literal["person_minutes"] = "person_minutes"
    observer_id: Annotated[str, Field(min_length=1)]
    provenance: Literal["human_timekeeper", "system_timer", "benchmark_replay"]
    recorded_at: datetime
    receipt_sha256: str

    @field_validator(
        "workspace_config_sha256",
        "predecessor_pointer_sha256",
        "authorization_sha256",
        "predecessor_expectation_sha256",
        "predecessor_state_sha256",
        "action_packet_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("cumulative_active_person_minutes")
    @classmethod
    def validate_cost(cls, value: float) -> float:
        return _finite_cost(value, "audit_checkpoint_cumulative_person_minutes")

    @field_validator("recorded_at")
    @classmethod
    def validate_recorded_at(cls, value: datetime) -> datetime:
        return _aware(value, "audit_checkpoint_recorded_at")

    @model_validator(mode="after")
    def validate_receipt(self) -> AuditActiveCostCheckpointReceiptV1:
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("audit_checkpoint_receipt_hash_mismatch")
        return self


class AuditWorkspaceTransactionMarkerV1(ContractModel):
    """Durable exact-once marker; a surviving pending marker poisons mutation."""

    marker_version: Literal["transactional-audit-marker-v1"] = "transactional-audit-marker-v1"
    status: Literal["committed", "pending"]
    predecessor_pointer_sha256: str | None = None
    intended_generation: Annotated[int, Field(ge=0)]
    transition_kind: Literal["initialized", "checkpointed", "adjudicated"]
    transition_receipt_sha256: str | None = None
    committed_pointer_sha256: str | None = None
    marker_sha256: str

    @field_validator(
        "predecessor_pointer_sha256",
        "transition_receipt_sha256",
        "committed_pointer_sha256",
    )
    @classmethod
    def validate_optional_hash(cls, value: str | None, info: Any) -> str | None:
        return _optional_sha256(value, info.field_name)

    @field_validator("marker_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value, "audit_workspace_marker")

    @model_validator(mode="after")
    def validate_marker(self) -> AuditWorkspaceTransactionMarkerV1:
        if self.status == "pending":
            if self.transition_kind == "initialized" or self.transition_receipt_sha256 is None:
                raise ValueError("audit_workspace_pending_marker_incomplete")
            if self.predecessor_pointer_sha256 is None or self.committed_pointer_sha256 is not None:
                raise ValueError("audit_workspace_pending_marker_state_invalid")
        else:
            if self.committed_pointer_sha256 is None:
                raise ValueError("audit_workspace_committed_marker_missing_pointer")
            if self.transition_kind == "initialized":
                if (
                    self.predecessor_pointer_sha256 is not None
                    or self.transition_receipt_sha256 is not None
                    or self.intended_generation != 0
                ):
                    raise ValueError("audit_workspace_genesis_marker_invalid")
            elif self.predecessor_pointer_sha256 is None or self.transition_receipt_sha256 is None:
                raise ValueError("audit_workspace_committed_marker_incomplete")
        payload = self.model_dump(mode="json", exclude={"marker_sha256"})
        if hash_canonical(payload) != self.marker_sha256:
            raise ValueError("audit_workspace_marker_hash_mismatch")
        return self


class AuditWorkspaceMutationResultV1(ContractModel):
    """Public immutable result of one initialized/checkpointed/adjudicated commit."""

    result_version: Literal["transactional-audit-workspace-result-v1"] = (
        "transactional-audit-workspace-result-v1"
    )
    transition_kind: Literal["initialized", "checkpointed", "adjudicated"]
    config: AuditWorkspaceConfigV1
    previous_pointer_sha256: str | None
    pointer: AuditWorkspacePointerV1
    result_sha256: str

    @field_validator("previous_pointer_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return _optional_sha256(value, "previous_pointer_sha256")

    @field_validator("result_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256(value, "audit_workspace_result")

    @model_validator(mode="after")
    def validate_result(self) -> AuditWorkspaceMutationResultV1:
        if self.transition_kind != self.pointer.transition_kind:
            raise ValueError("audit_workspace_result_transition_mismatch")
        if self.config.config_sha256 != self.pointer.workspace_config_sha256:
            raise ValueError("audit_workspace_result_config_mismatch")
        if self.previous_pointer_sha256 != self.pointer.predecessor_pointer_sha256:
            raise ValueError("audit_workspace_result_predecessor_mismatch")
        payload = self.model_dump(mode="json", exclude={"result_sha256"})
        if hash_canonical(payload) != self.result_sha256:
            raise ValueError("audit_workspace_result_hash_mismatch")
        return self


def _freeze_model(model_type: type[ContractModel], payload: dict[str, Any], hash_field: str) -> Any:
    # Hash the exact JSON-mode representation that the contract validator will
    # replay.  Raw constructor inputs may still contain tuples, enums, datetimes,
    # or nested ContractModel instances whose canonical JSON form differs from the
    # Python input form.
    provisional = model_type.model_construct(**payload, **{hash_field: "0" * 64})
    normalized = provisional.model_dump(mode="json", exclude={hash_field})
    return model_type.model_validate({**normalized, hash_field: hash_canonical(normalized)})


def _freeze_config(
    *,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
    budget_minutes: float,
    certificate: VerificationCertificate,
) -> AuditWorkspaceConfigV1:
    if (
        certificate.adaptive_calibration_bundle is None
        or certificate.adaptive_policy_context is None
    ):
        raise TransactionalAuditWorkspaceError("adaptive_calibration_required_for_workspace")
    identity = {
        "config_version": "transactional-audit-workspace-config-v1",
        "claim_manifest_sha256": hash_canonical(manifest),
        "source_corpus_sha256": corpus.source_sha256,
        "source_corpus_payload_sha256": hash_canonical(corpus.certificate_payload()),
        "source_graph_sha256": hash_canonical(corpus.graph),
        "complete_corpus_membership_sha256": (
            certificate.complete_corpus_identity.membership_sha256
        ),
        "budget_minutes": float(budget_minutes),
        "cost_unit": "person_minutes",
        "pipeline_sha256": certificate.pipeline_verification.computed_pipeline_sha256,
        "pipeline_verification_sha256": (certificate.pipeline_verification.verification_sha256),
        "adaptive_calibration_bundle_sha256": (
            certificate.adaptive_calibration_bundle.bundle_sha256
        ),
        "adaptive_policy_context_sha256": (
            certificate.adaptive_policy_context.policy_context_sha256
        ),
        "item_risk_scoring_receipt_sha256": (
            None
            if certificate.item_risk_scoring_receipt is None
            else certificate.item_risk_scoring_receipt.receipt_sha256
        ),
    }
    if identity["pipeline_sha256"] is None:
        raise TransactionalAuditWorkspaceError("computed_pipeline_identity_missing")
    workspace_id = f"audit-workspace-{hash_canonical(identity)[:16]}"
    payload = {**identity, "workspace_id": workspace_id}
    return _freeze_model(AuditWorkspaceConfigV1, payload, "config_sha256")


def _freeze_authorization(
    *,
    config: AuditWorkspaceConfigV1,
    generation: int,
    state: SequentialVerificationState,
    certificate: VerificationCertificate,
    issued_at: datetime,
) -> AuditActionAuthorizationV1 | None:
    action = state.session.active_action
    if action is None:
        return None
    expectation = freeze_state_expectation(state)
    payload = {
        "authorization_version": "transactional-audit-action-authorization-v1",
        "workspace_config_sha256": config.config_sha256,
        "generation": generation,
        "state_expectation": expectation,
        "certificate_sha256": certificate.certificate_sha256,
        "session_id": state.session.session_id,
        "step": action.step,
        "item_id": action.item_id,
        "action_packet_sha256": action.packet_sha256,
        "issued_at": _aware(issued_at, "audit_action_authorization_issued_at"),
        "release_blocked": True,
    }
    return _freeze_model(AuditActionAuthorizationV1, payload, "authorization_sha256")


def _generation_path(generation: int, state_sha256: str) -> str:
    return f"generations/{generation:06d}-{state_sha256[:16]}"


def _freeze_pointer(
    *,
    config: AuditWorkspaceConfigV1,
    generation: int,
    state: SequentialVerificationState,
    certificate: VerificationCertificate,
    transition_kind: Literal["initialized", "checkpointed", "adjudicated"],
    predecessor: AuditWorkspacePointerV1 | None,
    transition_receipt_sha256: str | None,
    authorization: AuditActionAuthorizationV1 | None,
) -> AuditWorkspacePointerV1:
    expectation = freeze_state_expectation(state)
    generation_path = _generation_path(generation, state.state_sha256)
    receipt_hashes = (
        []
        if predecessor is None
        else [
            *predecessor.transaction_receipt_sha256s,
            str(transition_receipt_sha256),
        ]
    )
    payload = {
        "pointer_version": "transactional-audit-workspace-pointer-v1",
        "workspace_id": config.workspace_id,
        "workspace_config_sha256": config.config_sha256,
        "generation": generation,
        "generation_path": generation_path,
        "state_path": f"{generation_path}/sequential-audit-state.json",
        "certificate_path": f"{generation_path}/verification-certificate.json",
        "predecessor_pointer_sha256": (None if predecessor is None else predecessor.pointer_sha256),
        "transition_kind": transition_kind,
        "transition_receipt_sha256": transition_receipt_sha256,
        "transaction_receipt_sha256s": receipt_hashes,
        "state_expectation": expectation,
        "certificate_sha256": certificate.certificate_sha256,
        "certificate_status": certificate.status,
        "authorization": authorization,
    }
    return _freeze_model(AuditWorkspacePointerV1, payload, "pointer_sha256")


def freeze_audit_adjudication_cost_receipt_v1(
    *,
    config: AuditWorkspaceConfigV1,
    pointer: AuditWorkspacePointerV1,
    disposition: CorrectionDisposition,
    corrected_graph: EvidenceGraph | None,
    provenance: Literal["blinded_human", "benchmark_adjudication"],
    adjudicator_count: int,
    adjudication_protocol_sha256: str,
    adjudication_payload_sha256: str,
    correction_protocol_sha256: str,
    correction_payload_sha256: str,
    completed_at: datetime,
    realized_person_minutes: float,
) -> AuditAdjudicationCostReceiptV1:
    """Freeze one externally completed outcome/cost record for the active action."""

    authorization = pointer.authorization
    if authorization is None:
        raise TransactionalAuditWorkspaceError("audit_receipt_requires_authorized_action")
    payload = {
        "receipt_version": "transactional-audit-adjudication-cost-v1",
        "workspace_config_sha256": config.config_sha256,
        "predecessor_pointer_sha256": pointer.pointer_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "predecessor_expectation_sha256": (pointer.state_expectation.expectation_sha256),
        "predecessor_state_sha256": pointer.state_expectation.state_sha256,
        "session_id": authorization.session_id,
        "step": authorization.step,
        "item_id": authorization.item_id,
        "action_packet_sha256": authorization.action_packet_sha256,
        "disposition": disposition,
        "corrected_graph_sha256": (
            None if corrected_graph is None else hash_canonical(corrected_graph)
        ),
        "provenance": provenance,
        "adjudicator_count": adjudicator_count,
        "adjudication_protocol_sha256": adjudication_protocol_sha256,
        "adjudication_payload_sha256": adjudication_payload_sha256,
        "correction_protocol_sha256": correction_protocol_sha256,
        "correction_payload_sha256": correction_payload_sha256,
        "completed_at": completed_at,
        "realized_person_minutes": realized_person_minutes,
        "cost_unit": "person_minutes",
    }
    return _freeze_model(AuditAdjudicationCostReceiptV1, payload, "receipt_sha256")


def freeze_audit_active_cost_checkpoint_receipt_v1(
    *,
    config: AuditWorkspaceConfigV1,
    pointer: AuditWorkspacePointerV1,
    cumulative_active_person_minutes: float,
    observer_id: str,
    provenance: Literal["human_timekeeper", "system_timer", "benchmark_replay"],
    recorded_at: datetime,
) -> AuditActiveCostCheckpointReceiptV1:
    """Freeze one label-free cumulative cost checkpoint for the active action."""

    authorization = pointer.authorization
    if authorization is None:
        raise TransactionalAuditWorkspaceError("checkpoint_requires_authorized_action")
    payload = {
        "receipt_version": "transactional-audit-active-cost-checkpoint-v1",
        "workspace_config_sha256": config.config_sha256,
        "predecessor_pointer_sha256": pointer.pointer_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "predecessor_expectation_sha256": (pointer.state_expectation.expectation_sha256),
        "predecessor_state_sha256": pointer.state_expectation.state_sha256,
        "session_id": authorization.session_id,
        "step": authorization.step,
        "item_id": authorization.item_id,
        "action_packet_sha256": authorization.action_packet_sha256,
        "cumulative_active_person_minutes": cumulative_active_person_minutes,
        "cost_unit": "person_minutes",
        "observer_id": observer_id,
        "provenance": provenance,
        "recorded_at": recorded_at,
    }
    return _freeze_model(AuditActiveCostCheckpointReceiptV1, payload, "receipt_sha256")


def _freeze_marker(
    *,
    status: Literal["committed", "pending"],
    predecessor: AuditWorkspacePointerV1 | None,
    intended_generation: int,
    transition_kind: Literal["initialized", "checkpointed", "adjudicated"],
    transition_receipt_sha256: str | None,
    committed_pointer_sha256: str | None,
) -> AuditWorkspaceTransactionMarkerV1:
    payload = {
        "marker_version": "transactional-audit-marker-v1",
        "status": status,
        "predecessor_pointer_sha256": (None if predecessor is None else predecessor.pointer_sha256),
        "intended_generation": intended_generation,
        "transition_kind": transition_kind,
        "transition_receipt_sha256": transition_receipt_sha256,
        "committed_pointer_sha256": committed_pointer_sha256,
    }
    return _freeze_model(AuditWorkspaceTransactionMarkerV1, payload, "marker_sha256")


def _freeze_result(
    *,
    transition_kind: Literal["initialized", "checkpointed", "adjudicated"],
    config: AuditWorkspaceConfigV1,
    predecessor: AuditWorkspacePointerV1 | None,
    pointer: AuditWorkspacePointerV1,
) -> AuditWorkspaceMutationResultV1:
    payload = {
        "result_version": "transactional-audit-workspace-result-v1",
        "transition_kind": transition_kind,
        "config": config,
        "previous_pointer_sha256": (None if predecessor is None else predecessor.pointer_sha256),
        "pointer": pointer,
    }
    return _freeze_model(AuditWorkspaceMutationResultV1, payload, "result_sha256")


def _load_model(path: Path, model_type: type[ContractModel], label: str) -> Any:
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label}_must_be_regular_file")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return model_type.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise TransactionalAuditWorkspaceError(f"{label}_invalid:{path}") from exc


def _outer_lock_path(workspace: Path) -> Path:
    if not workspace.name or workspace.name in {".", ".."}:
        raise TransactionalAuditWorkspaceError("audit_workspace_path_invalid")
    parent = workspace.parent.resolve(strict=True)
    if not parent.is_dir():
        raise TransactionalAuditWorkspaceError("audit_workspace_parent_not_directory")
    return parent / f".{workspace.name}.transactional-audit-v1.lock"


@contextmanager
def _workspace_lock(workspace: Path) -> Iterator[None]:
    lock_path = _outer_lock_path(workspace)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _safe_workspace(workspace: Path) -> Path:
    if workspace.is_symlink():
        raise TransactionalAuditWorkspaceError("audit_workspace_symlink_forbidden")
    try:
        resolved = workspace.resolve(strict=True)
    except FileNotFoundError as exc:
        raise TransactionalAuditWorkspaceError("audit_workspace_missing") from exc
    if not resolved.is_dir():
        raise TransactionalAuditWorkspaceError("audit_workspace_not_directory")
    mode = resolved.stat().st_mode & 0o777
    if mode & 0o077:
        raise TransactionalAuditWorkspaceError("audit_workspace_not_private")
    return resolved


def _require_path_absent(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise TransactionalAuditWorkspaceError(f"{label}_unexpected")


def _validate_generation_chain(
    *,
    root: Path,
    config: AuditWorkspaceConfigV1,
    generation_entries: list[Path],
) -> tuple[AuditWorkspacePointerV1, SequentialVerificationState]:
    previous_pointer: AuditWorkspacePointerV1 | None = None
    previous_state: SequentialVerificationState | None = None
    for generation, generation_root in enumerate(generation_entries):
        generation_pointer = _load_model(
            generation_root / "generation-pointer.json",
            AuditWorkspacePointerV1,
            "workspace_generation_pointer",
        )
        if (
            generation_pointer.generation != generation
            or root / generation_pointer.generation_path != generation_root
            or generation_pointer.workspace_id != config.workspace_id
            or generation_pointer.workspace_config_sha256 != config.config_sha256
        ):
            raise TransactionalAuditWorkspaceError(
                "audit_workspace_generation_pointer_context_mismatch"
            )
        state = _load_model(
            root / generation_pointer.state_path,
            SequentialVerificationState,
            "workspace_generation_state",
        )
        certificate = _load_model(
            root / generation_pointer.certificate_path,
            VerificationCertificate,
            "workspace_generation_certificate",
        )
        expectation = _load_model(
            generation_root / "state-expectation.json",
            SequentialStateExpectation,
            "workspace_generation_expectation",
        )
        if (
            expectation != generation_pointer.state_expectation
            or expectation != freeze_state_expectation(state)
            or certificate.certificate_sha256 != generation_pointer.certificate_sha256
            or certificate.status != generation_pointer.certificate_status
            or certificate.sequential_audit_state != state
        ):
            raise TransactionalAuditWorkspaceError(
                "audit_workspace_generation_scientific_artifact_mismatch"
            )
        authorization_path = generation_root / "audit-action-authorization.json"
        if generation_pointer.authorization is None:
            _require_path_absent(
                authorization_path,
                "workspace_generation_authorization",
            )
        else:
            authorization = _load_model(
                authorization_path,
                AuditActionAuthorizationV1,
                "workspace_generation_authorization",
            )
            if authorization != generation_pointer.authorization:
                raise TransactionalAuditWorkspaceError(
                    "audit_workspace_generation_authorization_mismatch"
                )

        receipt_path = generation_root / "transition-receipt.json"
        preflight_path = generation_root / "preflight-verification-certificate.json"
        result_path = generation_root / "transition-result.json"
        if generation == 0:
            if previous_pointer is not None or previous_state is not None:
                raise TransactionalAuditWorkspaceError(
                    "audit_workspace_generation_chain_genesis_mismatch"
                )
            for path, label in (
                (receipt_path, "workspace_genesis_transition_receipt"),
                (preflight_path, "workspace_genesis_preflight"),
                (result_path, "workspace_genesis_transition_result"),
            ):
                _require_path_absent(path, label)
        else:
            assert previous_pointer is not None and previous_state is not None
            if (
                generation_pointer.predecessor_pointer_sha256 != previous_pointer.pointer_sha256
                or generation_pointer.transaction_receipt_sha256s[:-1]
                != previous_pointer.transaction_receipt_sha256s
            ):
                raise TransactionalAuditWorkspaceError("audit_workspace_generation_chain_fork")
            receipt_type: type[
                AuditAdjudicationCostReceiptV1 | AuditActiveCostCheckpointReceiptV1
            ] = (
                AuditActiveCostCheckpointReceiptV1
                if generation_pointer.transition_kind == "checkpointed"
                else AuditAdjudicationCostReceiptV1
            )
            receipt = _load_model(
                receipt_path,
                receipt_type,
                "workspace_generation_transition_receipt",
            )
            if receipt.receipt_sha256 != generation_pointer.transition_receipt_sha256:
                raise TransactionalAuditWorkspaceError(
                    "audit_workspace_generation_receipt_pointer_mismatch"
                )
            _require_receipt_context(
                config=config,
                pointer=previous_pointer,
                state=previous_state,
                receipt=receipt,
            )
            preflight = _load_model(
                preflight_path,
                VerificationCertificate,
                "workspace_generation_preflight_certificate",
            )
            if (
                preflight.sequential_audit_state != previous_state
                or preflight.status != "abstained"
            ):
                raise TransactionalAuditWorkspaceError(
                    "audit_workspace_generation_preflight_state_mismatch"
                )
            if generation_pointer.transition_kind == "checkpointed":
                checkpoint = _load_model(
                    result_path,
                    SequentialActiveCostCheckpointResult,
                    "workspace_generation_checkpoint_result",
                )
                if (
                    checkpoint.previous_state_sha256 != previous_state.state_sha256
                    or checkpoint.state != state
                    or not isinstance(receipt, AuditActiveCostCheckpointReceiptV1)
                    or checkpoint.active_realized_cost != receipt.cumulative_active_person_minutes
                ):
                    raise TransactionalAuditWorkspaceError(
                        "audit_workspace_generation_checkpoint_join_mismatch"
                    )
            else:
                resolution = _load_model(
                    result_path,
                    SequentialResolutionResult,
                    "workspace_generation_resolution_result",
                )
                if not isinstance(receipt, AuditAdjudicationCostReceiptV1):
                    raise TransactionalAuditWorkspaceError(
                        "audit_workspace_generation_resolution_receipt_type_mismatch"
                    )
                adjudication = resolution.adjudication
                correction = resolution.correction_provenance
                if (
                    resolution.previous_state_sha256 != previous_state.state_sha256
                    or state.transitions[: len(resolution.state.transitions)]
                    != resolution.state.transitions
                    or state.graph != resolution.state.graph
                    or state.synthesis != resolution.state.synthesis
                    or (
                        adjudication.provenance,
                        adjudication.adjudicator_count,
                        adjudication.protocol_sha256,
                        adjudication.payload_sha256,
                        adjudication.completed_at,
                        adjudication.realized_cost,
                    )
                    != (
                        receipt.provenance,
                        receipt.adjudicator_count,
                        receipt.adjudication_protocol_sha256,
                        receipt.adjudication_payload_sha256,
                        receipt.completed_at,
                        receipt.realized_person_minutes,
                    )
                    or (
                        correction.disposition,
                        correction.provenance,
                        correction.correction_protocol_sha256,
                        correction.external_correction_payload_sha256,
                    )
                    != (
                        receipt.disposition,
                        receipt.provenance,
                        receipt.correction_protocol_sha256,
                        receipt.correction_payload_sha256,
                    )
                ):
                    raise TransactionalAuditWorkspaceError(
                        "audit_workspace_generation_resolution_join_mismatch"
                    )
        previous_pointer = generation_pointer
        previous_state = state
    if previous_pointer is None or previous_state is None:
        raise AmbiguousTransactionalAuditWorkspaceError("audit_workspace_generation_chain_empty")
    return previous_pointer, previous_state


def _load_workspace(
    workspace: Path,
) -> tuple[Path, AuditWorkspaceConfigV1, AuditWorkspacePointerV1]:
    root = _safe_workspace(workspace)
    config = _load_model(root / "workspace-config.json", AuditWorkspaceConfigV1, "workspace_config")
    pointer = _load_model(
        root / "current-pointer.json", AuditWorkspacePointerV1, "workspace_pointer"
    )
    marker = _load_model(
        root / "transaction-marker.json",
        AuditWorkspaceTransactionMarkerV1,
        "workspace_transaction_marker",
    )
    if marker.status != "committed":
        raise AmbiguousTransactionalAuditWorkspaceError(
            "audit_workspace_prior_transaction_ambiguous"
        )
    if (
        marker.committed_pointer_sha256 != pointer.pointer_sha256
        or marker.intended_generation != pointer.generation
        or marker.transition_kind != pointer.transition_kind
        or marker.predecessor_pointer_sha256 != pointer.predecessor_pointer_sha256
        or marker.transition_receipt_sha256 != pointer.transition_receipt_sha256
    ):
        raise AmbiguousTransactionalAuditWorkspaceError("audit_workspace_marker_pointer_mismatch")
    if (
        config.workspace_id != pointer.workspace_id
        or config.config_sha256 != pointer.workspace_config_sha256
    ):
        raise TransactionalAuditWorkspaceError("audit_workspace_config_pointer_mismatch")
    generation_root = root / "generations"
    if generation_root.is_symlink() or not generation_root.is_dir():
        raise AmbiguousTransactionalAuditWorkspaceError("audit_workspace_generation_root_ambiguous")
    generation_entries = sorted(generation_root.iterdir(), key=lambda child: child.name)
    if any(child.is_symlink() for child in generation_entries):
        raise AmbiguousTransactionalAuditWorkspaceError(
            "audit_workspace_generation_symlink_forbidden"
        )
    if any(not child.is_dir() for child in generation_entries):
        raise AmbiguousTransactionalAuditWorkspaceError(
            "audit_workspace_generation_non_directory_entry"
        )
    observed = [child.name for child in generation_entries]
    expected_prefixes = [f"{index:06d}-" for index in range(pointer.generation + 1)]
    if len(observed) != len(expected_prefixes) or any(
        not name.startswith(prefix)
        for name, prefix in zip(observed, expected_prefixes, strict=True)
    ):
        raise AmbiguousTransactionalAuditWorkspaceError(
            "audit_workspace_generation_roster_ambiguous"
        )
    terminal_pointer, _ = _validate_generation_chain(
        root=root,
        config=config,
        generation_entries=generation_entries,
    )
    if terminal_pointer != pointer:
        raise AmbiguousTransactionalAuditWorkspaceError(
            "audit_workspace_current_pointer_generation_fork"
        )
    return root, config, pointer


def load_transactional_audit_workspace_v1(
    workspace: Path,
) -> tuple[AuditWorkspaceConfigV1, AuditWorkspacePointerV1]:
    """Read and fully validate the canonical workspace under its outer lock."""

    with _workspace_lock(workspace):
        _, config, pointer = _load_workspace(workspace)
        return config, pointer


def _load_pointer_state(
    root: Path, pointer: AuditWorkspacePointerV1
) -> SequentialVerificationState:
    return _load_model(root / pointer.state_path, SequentialVerificationState, "workspace_state")


def _require_expected_pointer(
    pointer: AuditWorkspacePointerV1,
    expected: SequentialStateExpectation,
    expected_pointer_sha256: str,
) -> None:
    if (
        not SHA256_RE.fullmatch(expected_pointer_sha256)
        or expected_pointer_sha256 != pointer.pointer_sha256
    ):
        raise StaleTransactionalAuditStateError("audit_workspace_stale_predecessor_pointer")
    try:
        frozen = SequentialStateExpectation.model_validate(expected.model_dump(mode="json"))
    except (AttributeError, ValueError) as exc:
        raise StaleTransactionalAuditStateError("audit_workspace_expectation_invalid") from exc
    if frozen != pointer.state_expectation:
        raise StaleTransactionalAuditStateError("audit_workspace_stale_expectation")


def _preflight_replay(
    *,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
    budget_minutes: float,
    adaptive_calibration_bundle: AdaptiveCalibrationBundle,
    state: SequentialVerificationState,
    expected_pipeline_fingerprint: PipelineFingerprint | None,
    pipeline_root: Path | None,
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None,
    generated_at: datetime,
) -> VerificationCertificate:
    if manifest.claim_manifest_version == "3":
        raise TransactionalAuditWorkspaceError(
            "transactional_audit_workspace_manifest_v3_not_supported"
        )
    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=budget_minutes,
        adaptive_calibration_bundle=adaptive_calibration_bundle,
        expected_pipeline_fingerprint=expected_pipeline_fingerprint,
        pipeline_root=pipeline_root,
        item_risk_scoring_receipt=item_risk_scoring_receipt,
        sequential_audit_state=state,
        generated_at=generated_at,
    )
    if not isinstance(certificate, VerificationCertificate):
        raise TransactionalAuditWorkspaceError("workspace_replay_certificate_type_invalid")
    if certificate.sequential_audit_state != state:
        raise TransactionalAuditWorkspaceError("workspace_preflight_replay_changed_active_state")
    if state.session.active_action is None:
        raise TransactionalAuditWorkspaceError("workspace_preflight_requires_active_action")
    if certificate.status != "abstained" or (
        "active_audit_action_unresolved" not in certificate.release_assessment.audit.reasons
    ):
        raise TransactionalAuditWorkspaceError(
            "workspace_preflight_active_action_not_release_blocking"
        )
    if certificate.production_stop_decision.outcome != "active_action_in_progress":
        raise TransactionalAuditWorkspaceError("workspace_preflight_stop_decision_mismatch")
    return certificate


def _require_config_match(
    *,
    expected: AuditWorkspaceConfigV1,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
    budget_minutes: float,
    certificate: VerificationCertificate,
) -> None:
    observed = _freeze_config(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=budget_minutes,
        certificate=certificate,
    )
    if observed != expected:
        raise TransactionalAuditWorkspaceError("audit_workspace_replay_config_mismatch")


def _require_receipt_context(
    *,
    config: AuditWorkspaceConfigV1,
    pointer: AuditWorkspacePointerV1,
    state: SequentialVerificationState,
    receipt: AuditAdjudicationCostReceiptV1 | AuditActiveCostCheckpointReceiptV1,
) -> None:
    authorization = pointer.authorization
    action = state.session.active_action
    if authorization is None or action is None:
        raise TransactionalAuditWorkspaceError("audit_workspace_active_authorization_missing")
    if receipt.receipt_sha256 in pointer.transaction_receipt_sha256s:
        raise TransactionalAuditWorkspaceError("audit_workspace_duplicate_receipt")
    observed = (
        receipt.workspace_config_sha256,
        receipt.predecessor_pointer_sha256,
        receipt.authorization_sha256,
        receipt.predecessor_expectation_sha256,
        receipt.predecessor_state_sha256,
        receipt.session_id,
        receipt.step,
        receipt.item_id,
        receipt.action_packet_sha256,
        receipt.cost_unit,
    )
    expected = (
        config.config_sha256,
        pointer.pointer_sha256,
        authorization.authorization_sha256,
        pointer.state_expectation.expectation_sha256,
        state.state_sha256,
        state.session.session_id,
        action.step,
        action.item_id,
        action.packet_sha256,
        state.session.cost_unit,
    )
    if observed != expected:
        raise TransactionalAuditWorkspaceError("audit_workspace_receipt_context_mismatch")
    recorded_at = (
        receipt.completed_at
        if isinstance(receipt, AuditAdjudicationCostReceiptV1)
        else receipt.recorded_at
    )
    if recorded_at < authorization.issued_at or recorded_at < action.selected_at:
        raise TransactionalAuditWorkspaceError("audit_workspace_receipt_predates_authorization")


def _resolve_scientific_state(
    *,
    manifest: ClaimManifest,
    state: SequentialVerificationState,
    receipt: AuditAdjudicationCostReceiptV1,
    corrected_graph: EvidenceGraph | None,
    certificate: VerificationCertificate,
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None,
) -> SequentialResolutionResult:
    if receipt.disposition is CorrectionDisposition.NO_CHANGE:
        if corrected_graph is not None:
            raise TransactionalAuditWorkspaceError("no_change_receipt_forbids_corrected_graph")
        post_graph = state.graph
    else:
        if corrected_graph is None:
            raise TransactionalAuditWorkspaceError("corrected_receipt_requires_corrected_graph")
        if hash_canonical(corrected_graph) != receipt.corrected_graph_sha256:
            raise TransactionalAuditWorkspaceError("corrected_graph_receipt_hash_mismatch")
        post_graph = corrected_graph
    pipeline_verification = certificate.pipeline_verification
    if item_risk_scoring_receipt is not None and (
        item_risk_scoring_receipt.pipeline_verification != pipeline_verification
    ):
        raise TransactionalAuditWorkspaceError("item_risk_scoring_receipt_pipeline_mismatch")
    if any(candidate.risk_bound_sha256 is not None for candidate in state.candidates) and (
        item_risk_scoring_receipt is None
    ):
        raise TransactionalAuditWorkspaceError(
            "artifact_backed_audit_state_requires_scoring_receipt"
        )
    item_bundle = (
        None if item_risk_scoring_receipt is None else item_risk_scoring_receipt.calibration_bundle
    )
    item_candidates = (
        None if item_risk_scoring_receipt is None else list(item_risk_scoring_receipt.candidates)
    )
    action = state.session.active_action
    assert action is not None
    prepared = prepare_verification_scientific_state(
        manifest=manifest,
        graph=post_graph,
        pipeline_verification=pipeline_verification,
        item_risk_calibration_bundle=item_bundle,
        item_risk_candidates=item_candidates,
        resolved_item_ids_for_risk_projection={
            *state.session.resolved_item_ids,
            action.item_id,
        },
    )
    refreshed_candidates = sequential_candidates_from_prepared_state(
        manifest=manifest,
        prepared=prepared,
    )
    expectation = freeze_state_expectation(state)
    adjudication = freeze_selected_adjudication(
        state,
        expected=expectation,
        provenance=receipt.provenance,
        adjudicator_count=receipt.adjudicator_count,
        protocol_sha256=receipt.adjudication_protocol_sha256,
        payload_sha256=receipt.adjudication_payload_sha256,
        completed_at=receipt.completed_at,
        realized_cost=receipt.realized_person_minutes,
    )
    post_graph_sha256 = hash_canonical(post_graph)

    def rerun_synthesis(graph: EvidenceGraph) -> dict[str, Any]:
        if hash_canonical(graph) != post_graph_sha256:
            raise TransactionalAuditWorkspaceError("audit_synthesis_callback_graph_mismatch")
        return prepared.synthesis

    def rerun_candidates(graph: EvidenceGraph, synthesis: Any, session: Any) -> tuple[Any, ...]:
        if (
            hash_canonical(graph) != post_graph_sha256
            or hash_canonical(synthesis) != hash_canonical(prepared.synthesis)
            or session.session_id != state.session.session_id
        ):
            raise TransactionalAuditWorkspaceError("audit_candidate_callback_state_mismatch")
        return refreshed_candidates

    pipeline_sha256 = state.session.pipeline_sha256
    return resolve_selected_audit_candidate(
        state,
        expected=expectation,
        adjudication=adjudication,
        disposition=receipt.disposition,
        corrected_graph=corrected_graph,
        correction_provenance=receipt.provenance,
        correction_protocol_sha256=receipt.correction_protocol_sha256,
        external_correction_payload_sha256=receipt.correction_payload_sha256,
        synthesis_runner_sha256=compute_synthesis_runner_sha256(
            manifest=manifest, pipeline_sha256=pipeline_sha256
        ),
        candidate_runner_sha256=compute_candidate_runner_sha256(
            manifest=manifest, pipeline_sha256=pipeline_sha256
        ),
        rerun_synthesis=rerun_synthesis,
        rerun_candidates=rerun_candidates,
    )


def _terminal_replay(
    *,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
    budget_minutes: float,
    adaptive_calibration_bundle: AdaptiveCalibrationBundle,
    state: SequentialVerificationState,
    expected_pipeline_fingerprint: PipelineFingerprint | None,
    pipeline_root: Path | None,
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None,
    generated_at: datetime,
) -> VerificationCertificate:
    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=budget_minutes,
        adaptive_calibration_bundle=adaptive_calibration_bundle,
        expected_pipeline_fingerprint=expected_pipeline_fingerprint,
        pipeline_root=pipeline_root,
        item_risk_scoring_receipt=item_risk_scoring_receipt,
        sequential_audit_state=state,
        generated_at=generated_at,
    )
    if not isinstance(certificate, VerificationCertificate):
        raise TransactionalAuditWorkspaceError("workspace_terminal_certificate_type_invalid")
    terminal_state = certificate.sequential_audit_state
    if terminal_state is None:
        raise TransactionalAuditWorkspaceError("workspace_terminal_state_missing")
    if not terminal_state.transitions[: len(state.transitions)] == state.transitions:
        raise TransactionalAuditWorkspaceError("workspace_terminal_state_not_descendant")
    if certificate.status == "released" and terminal_state.session.active_action is not None:
        raise TransactionalAuditWorkspaceError("workspace_released_with_active_action")
    return certificate


def _write_generation(
    *,
    root: Path,
    pointer: AuditWorkspacePointerV1,
    state: SequentialVerificationState,
    certificate: VerificationCertificate,
    expectation: SequentialStateExpectation,
    authorization: AuditActionAuthorizationV1 | None,
    transition_receipt: AuditAdjudicationCostReceiptV1 | AuditActiveCostCheckpointReceiptV1 | None,
    preflight_certificate: VerificationCertificate | None,
    transition_result: SequentialResolutionResult | SequentialActiveCostCheckpointResult | None,
) -> None:
    generation_root = root / "generations"
    generation_root.mkdir(mode=0o700, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=generation_root))
    try:
        os.chmod(stage, 0o700)
        atomic_write_json(stage / "sequential-audit-state.json", state)
        write_certificate_artifacts(certificate, stage)
        atomic_write_json(stage / "state-expectation.json", expectation)
        if authorization is not None:
            atomic_write_json(stage / "audit-action-authorization.json", authorization)
        if transition_receipt is not None:
            atomic_write_json(stage / "transition-receipt.json", transition_receipt)
        if preflight_certificate is not None:
            atomic_write_json(
                stage / "preflight-verification-certificate.json",
                preflight_certificate,
            )
        if transition_result is not None:
            atomic_write_json(stage / "transition-result.json", transition_result)
        atomic_write_json(stage / "generation-pointer.json", pointer)
        destination = root / pointer.generation_path
        if destination.exists():
            raise AmbiguousTransactionalAuditWorkspaceError(
                "audit_workspace_generation_destination_exists"
            )
        os.rename(stage, destination)
        descriptor = os.open(generation_root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        if stage.exists():
            # A staging directory is retained deliberately.  Its presence poisons the
            # generation roster on the next invocation instead of hiding ambiguity.
            pass
        raise


def _publish_pointer(
    *,
    root: Path,
    pointer: AuditWorkspacePointerV1,
    predecessor: AuditWorkspacePointerV1,
    transition_kind: Literal["checkpointed", "adjudicated"],
    receipt_sha256: str,
) -> None:
    atomic_write_json(root / "current-pointer.json", pointer, force=True)
    committed = _freeze_marker(
        status="committed",
        predecessor=predecessor,
        intended_generation=pointer.generation,
        transition_kind=transition_kind,
        transition_receipt_sha256=receipt_sha256,
        committed_pointer_sha256=pointer.pointer_sha256,
    )
    atomic_write_json(root / "transaction-marker.json", committed, force=True)


def initialize_transactional_audit_workspace_v1(
    *,
    workspace: Path,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
    budget_minutes: float,
    adaptive_calibration_bundle: AdaptiveCalibrationBundle,
    state: SequentialVerificationState,
    expected_pipeline_fingerprint: PipelineFingerprint | None = None,
    pipeline_root: Path | None = None,
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None = None,
    initialized_at: datetime | None = None,
) -> AuditWorkspaceMutationResultV1:
    """Replay and atomically initialize one pre-liability canonical workspace."""

    initialized_at = _aware(initialized_at or datetime.now(UTC), "audit_workspace_initialized_at")
    current = resume_sequential_verification_state(state)
    if current.session.active_action is not None:
        initialized_at = max(initialized_at, current.session.active_action.selected_at)
    with _workspace_lock(workspace):
        if workspace.exists() or workspace.is_symlink():
            raise TransactionalAuditWorkspaceError("audit_workspace_already_exists")
        certificate = _preflight_replay(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=budget_minutes,
            adaptive_calibration_bundle=adaptive_calibration_bundle,
            state=current,
            expected_pipeline_fingerprint=expected_pipeline_fingerprint,
            pipeline_root=pipeline_root,
            item_risk_scoring_receipt=item_risk_scoring_receipt,
            generated_at=initialized_at,
        )
        config = _freeze_config(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=budget_minutes,
            certificate=certificate,
        )
        authorization = _freeze_authorization(
            config=config,
            generation=0,
            state=current,
            certificate=certificate,
            issued_at=initialized_at,
        )
        assert authorization is not None
        pointer = _freeze_pointer(
            config=config,
            generation=0,
            state=current,
            certificate=certificate,
            transition_kind="initialized",
            predecessor=None,
            transition_receipt_sha256=None,
            authorization=authorization,
        )
        parent = workspace.parent.resolve(strict=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{workspace.name}.init-", dir=parent))
        try:
            os.chmod(stage, 0o700)
            atomic_write_json(stage / "workspace-config.json", config)
            _write_generation(
                root=stage,
                pointer=pointer,
                state=current,
                certificate=certificate,
                expectation=pointer.state_expectation,
                authorization=authorization,
                transition_receipt=None,
                preflight_certificate=None,
                transition_result=None,
            )
            atomic_write_json(stage / "current-pointer.json", pointer)
            marker = _freeze_marker(
                status="committed",
                predecessor=None,
                intended_generation=0,
                transition_kind="initialized",
                transition_receipt_sha256=None,
                committed_pointer_sha256=pointer.pointer_sha256,
            )
            atomic_write_json(stage / "transaction-marker.json", marker)
            os.rename(stage, workspace)
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException:
            if stage.exists():
                # Initialization has no externally visible pointer; removing a private
                # staging directory is safe, but avoid recursive deletion here.  Its
                # unique path cannot be mistaken for the requested workspace.
                pass
            raise
        return _freeze_result(
            transition_kind="initialized",
            config=config,
            predecessor=None,
            pointer=pointer,
        )


def checkpoint_transactional_audit_workspace_v1(
    *,
    workspace: Path,
    expected: SequentialStateExpectation,
    expected_pointer_sha256: str,
    receipt_path: Path,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
    budget_minutes: float,
    adaptive_calibration_bundle: AdaptiveCalibrationBundle,
    expected_pipeline_fingerprint: PipelineFingerprint | None = None,
    pipeline_root: Path | None = None,
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None = None,
) -> AuditWorkspaceMutationResultV1:
    """CAS-commit one label-free cumulative active-cost checkpoint."""

    with _workspace_lock(workspace):
        root, config, pointer = _load_workspace(workspace)
        _require_expected_pointer(pointer, expected, expected_pointer_sha256)
        state = _load_pointer_state(root, pointer)
        action = state.session.active_action
        if action is None:
            raise TransactionalAuditWorkspaceError("checkpoint_requires_active_action")
        generated_at = max(datetime.now(UTC), action.selected_at)
        preflight = _preflight_replay(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=budget_minutes,
            adaptive_calibration_bundle=adaptive_calibration_bundle,
            state=state,
            expected_pipeline_fingerprint=expected_pipeline_fingerprint,
            pipeline_root=pipeline_root,
            item_risk_scoring_receipt=item_risk_scoring_receipt,
            generated_at=generated_at,
        )
        _require_config_match(
            expected=config,
            manifest=manifest,
            corpus=corpus,
            budget_minutes=budget_minutes,
            certificate=preflight,
        )
        # Outcome/cost bytes are deliberately unopened until CAS + full replay pass.
        receipt = _load_model(
            receipt_path,
            AuditActiveCostCheckpointReceiptV1,
            "audit_checkpoint_receipt",
        )
        _require_receipt_context(config=config, pointer=pointer, state=state, receipt=receipt)
        checkpoint = checkpoint_selected_audit_cost(
            state,
            expected=pointer.state_expectation,
            active_realized_cost=receipt.cumulative_active_person_minutes,
        )
        terminal = _terminal_replay(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=budget_minutes,
            adaptive_calibration_bundle=adaptive_calibration_bundle,
            state=checkpoint.state,
            expected_pipeline_fingerprint=expected_pipeline_fingerprint,
            pipeline_root=pipeline_root,
            item_risk_scoring_receipt=item_risk_scoring_receipt,
            generated_at=max(generated_at, receipt.recorded_at),
        )
        terminal_state = terminal.sequential_audit_state
        assert terminal_state is not None
        if (
            terminal_state != checkpoint.state
            or terminal_state.session.active_action is None
            or terminal.status != "abstained"
        ):
            raise TransactionalAuditWorkspaceError(
                "checkpoint_terminal_replay_removed_release_blocker"
            )
        generation = pointer.generation + 1
        authorization = _freeze_authorization(
            config=config,
            generation=generation,
            state=terminal_state,
            certificate=terminal,
            issued_at=receipt.recorded_at,
        )
        assert authorization is not None
        next_pointer = _freeze_pointer(
            config=config,
            generation=generation,
            state=terminal_state,
            certificate=terminal,
            transition_kind="checkpointed",
            predecessor=pointer,
            transition_receipt_sha256=receipt.receipt_sha256,
            authorization=authorization,
        )
        pending = _freeze_marker(
            status="pending",
            predecessor=pointer,
            intended_generation=pointer.generation + 1,
            transition_kind="checkpointed",
            transition_receipt_sha256=receipt.receipt_sha256,
            committed_pointer_sha256=None,
        )
        atomic_write_json(root / "transaction-marker.json", pending, force=True)
        _write_generation(
            root=root,
            pointer=next_pointer,
            state=terminal_state,
            certificate=terminal,
            expectation=next_pointer.state_expectation,
            authorization=authorization,
            transition_receipt=receipt,
            preflight_certificate=preflight,
            transition_result=checkpoint,
        )
        _publish_pointer(
            root=root,
            pointer=next_pointer,
            predecessor=pointer,
            transition_kind="checkpointed",
            receipt_sha256=receipt.receipt_sha256,
        )
        return _freeze_result(
            transition_kind="checkpointed",
            config=config,
            predecessor=pointer,
            pointer=next_pointer,
        )


def advance_transactional_audit_workspace_v1(
    *,
    workspace: Path,
    expected: SequentialStateExpectation,
    expected_pointer_sha256: str,
    receipt_path: Path,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
    budget_minutes: float,
    adaptive_calibration_bundle: AdaptiveCalibrationBundle,
    corrected_corpus_path: Path | None = None,
    expected_pipeline_fingerprint: PipelineFingerprint | None = None,
    pipeline_root: Path | None = None,
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None = None,
) -> AuditWorkspaceMutationResultV1:
    """CAS-resolve, rerun science, assess release/select, and publish one generation."""

    with _workspace_lock(workspace):
        root, config, pointer = _load_workspace(workspace)
        _require_expected_pointer(pointer, expected, expected_pointer_sha256)
        state = _load_pointer_state(root, pointer)
        action = state.session.active_action
        if action is None:
            raise TransactionalAuditWorkspaceError("advance_requires_active_action")
        generated_at = max(datetime.now(UTC), action.selected_at)
        preflight = _preflight_replay(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=budget_minutes,
            adaptive_calibration_bundle=adaptive_calibration_bundle,
            state=state,
            expected_pipeline_fingerprint=expected_pipeline_fingerprint,
            pipeline_root=pipeline_root,
            item_risk_scoring_receipt=item_risk_scoring_receipt,
            generated_at=generated_at,
        )
        _require_config_match(
            expected=config,
            manifest=manifest,
            corpus=corpus,
            budget_minutes=budget_minutes,
            certificate=preflight,
        )
        # Adjudication/cost bytes are deliberately unopened until CAS + replay pass.
        receipt = _load_model(
            receipt_path,
            AuditAdjudicationCostReceiptV1,
            "audit_adjudication_cost_receipt",
        )
        _require_receipt_context(config=config, pointer=pointer, state=state, receipt=receipt)
        if receipt.disposition is CorrectionDisposition.NO_CHANGE:
            if corrected_corpus_path is not None:
                raise TransactionalAuditWorkspaceError("no_change_receipt_forbids_corrected_corpus")
            corrected_graph = None
        else:
            if corrected_corpus_path is None:
                raise TransactionalAuditWorkspaceError(
                    "corrected_receipt_requires_corrected_corpus"
                )
            repository_root = pipeline_root or Path(__file__).resolve().parents[2]
            corrected_graph = load_corpus(
                corrected_corpus_path,
                legacy_settings=manifest.legacy_adapter,
                repository_root=repository_root,
            ).graph
        resolution = _resolve_scientific_state(
            manifest=manifest,
            state=state,
            receipt=receipt,
            corrected_graph=corrected_graph,
            certificate=preflight,
            item_risk_scoring_receipt=item_risk_scoring_receipt,
        )
        terminal = _terminal_replay(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=budget_minutes,
            adaptive_calibration_bundle=adaptive_calibration_bundle,
            state=resolution.state,
            expected_pipeline_fingerprint=expected_pipeline_fingerprint,
            pipeline_root=pipeline_root,
            item_risk_scoring_receipt=item_risk_scoring_receipt,
            generated_at=max(generated_at, receipt.completed_at),
        )
        terminal_state = terminal.sequential_audit_state
        assert terminal_state is not None
        generation = pointer.generation + 1
        authorization = _freeze_authorization(
            config=config,
            generation=generation,
            state=terminal_state,
            certificate=terminal,
            issued_at=max(generated_at, receipt.completed_at),
        )
        next_pointer = _freeze_pointer(
            config=config,
            generation=generation,
            state=terminal_state,
            certificate=terminal,
            transition_kind="adjudicated",
            predecessor=pointer,
            transition_receipt_sha256=receipt.receipt_sha256,
            authorization=authorization,
        )
        pending = _freeze_marker(
            status="pending",
            predecessor=pointer,
            intended_generation=pointer.generation + 1,
            transition_kind="adjudicated",
            transition_receipt_sha256=receipt.receipt_sha256,
            committed_pointer_sha256=None,
        )
        atomic_write_json(root / "transaction-marker.json", pending, force=True)
        _write_generation(
            root=root,
            pointer=next_pointer,
            state=terminal_state,
            certificate=terminal,
            expectation=next_pointer.state_expectation,
            authorization=authorization,
            transition_receipt=receipt,
            preflight_certificate=preflight,
            transition_result=resolution,
        )
        _publish_pointer(
            root=root,
            pointer=next_pointer,
            predecessor=pointer,
            transition_kind="adjudicated",
            receipt_sha256=receipt.receipt_sha256,
        )
        return _freeze_result(
            transition_kind="adjudicated",
            config=config,
            predecessor=pointer,
            pointer=next_pointer,
        )


__all__ = [
    "AmbiguousTransactionalAuditWorkspaceError",
    "AuditActionAuthorizationV1",
    "AuditActiveCostCheckpointReceiptV1",
    "AuditAdjudicationCostReceiptV1",
    "AuditWorkspaceConfigV1",
    "AuditWorkspaceMutationResultV1",
    "AuditWorkspacePointerV1",
    "StaleTransactionalAuditStateError",
    "TransactionalAuditWorkspaceError",
    "advance_transactional_audit_workspace_v1",
    "checkpoint_transactional_audit_workspace_v1",
    "freeze_audit_active_cost_checkpoint_receipt_v1",
    "freeze_audit_adjudication_cost_receipt_v1",
    "initialize_transactional_audit_workspace_v1",
    "load_transactional_audit_workspace_v1",
]
