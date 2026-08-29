"""Fresh exact-once archive lifecycle for the contextual frontier recovery.

The scientific request, source envelope, and trusted evaluator live in
``metasyn_contextual_frontier_recovery_v2``.  This module only owns the mutable
state machine around that frozen core.  It creates a new private workspace,
persists an authorization and intent before transport, permits exactly one
provider attempt, and makes any orphan or ambiguous attempt terminal.

Even a fully grounded typed graph is an observation in this smoke run.  Every
accuracy, mechanics, synthesis, calibration, and release authority remains
false throughout this archive.
"""

from __future__ import annotations

import fcntl
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from pydantic import ConfigDict, Field, model_validator

from literature_multiverse import metasyn_contextual_frontier_recovery_v2 as core
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.metasyn_contextual_frontier_runtime_v1 import (
    MetaSynContextualFrontierClientV1,
    MetaSynContextualFrontierConfigV1,
    MetaSynContextualFrontierProviderResultV1,
    _assert_secret_free,
    _checked_artifact,
    _existing_workspace,
    _fresh_workspace,
    _load_object,
    _persist_json,
    _safe_exception_type,
    _safe_repository_file,
    _safe_request_id,
    _safe_status,
)
from literature_multiverse.models import SHA256_RE, ContractModel

LIFECYCLE_VERSION = "metasyn-contextual-frontier-recovery-lifecycle-v2"
PREPARED_VERSION = "metasyn-contextual-frontier-recovery-lifecycle-prepared-v2"
AUTHORIZATION_VERSION = "metasyn-contextual-frontier-recovery-lifecycle-authorization-v2"
INTENT_VERSION = "metasyn-contextual-frontier-recovery-lifecycle-intent-v2"
RECEIPT_VERSION = "metasyn-contextual-frontier-recovery-lifecycle-receipt-v2"
VALIDATION_VERSION = "metasyn-contextual-frontier-recovery-lifecycle-validation-v2"
INCIDENT_VERSION = "metasyn-contextual-frontier-recovery-lifecycle-incident-v2"
TERMINAL_VERSION = "metasyn-contextual-frontier-recovery-lifecycle-terminal-v2"
STATUS_VERSION = "metasyn-contextual-frontier-recovery-lifecycle-status-v2"
WORKSPACE_VALIDATION_VERSION = (
    "metasyn-contextual-frontier-recovery-lifecycle-workspace-validation-v2"
)

DEFAULT_WORKSPACE = Path("data/cache/metasyn/contextual-frontier-recovery-v2")
LIFECYCLE_SOURCE_PATH = Path(
    "src/literature_multiverse/metasyn_contextual_frontier_recovery_lifecycle_v2.py"
)

_PREPARED = Path("00-prepared.json")
_AUTHORIZED = Path("01-authorized.json")
_TERMINAL = Path("02-terminal.json")
_INTENT = Path("intent.json")
_PROVIDER_RESULT = Path("provider-result.json")
_RECEIPT = Path("provider-receipt.json")
_VALIDATION = Path("validation.json")
_INCIDENT = Path("incident.json")
_LOCK_NAME = ".metasyn-contextual-frontier-recovery-lifecycle-v2.lock"


class MetaSynContextualFrontierRecoveryLifecycleV2Error(ValueError):
    """A recovery lifecycle transition or archive failed closed."""


class _Frozen(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]


def _self_hash(model: _Frozen, field: str, code: str) -> None:
    if getattr(model, field) != hash_canonical(model.model_dump(mode="json", exclude={field})):
        raise ValueError(code)


def _request(plan: core.MetaSynContextualFrontierRecoveryPlanV2) -> Any:
    return plan.request


def _request_key(plan: core.MetaSynContextualFrontierRecoveryPlanV2) -> str:
    return str(_request(plan).request_key)


def _transport_request(plan: core.MetaSynContextualFrontierRecoveryPlanV2) -> Any:
    return _request(plan).transport_request


class MetaSynContextualFrontierRecoveryLifecyclePreparedV2(_Frozen):
    prepared_version: Literal["metasyn-contextual-frontier-recovery-lifecycle-prepared-v2"] = (
        PREPARED_VERSION
    )
    lifecycle_version: Literal["metasyn-contextual-frontier-recovery-lifecycle-v2"] = (
        LIFECYCLE_VERSION
    )
    status: Literal["offline_prepared_zero_provider_calls"] = "offline_prepared_zero_provider_calls"
    plan: core.MetaSynContextualFrontierRecoveryPlanV2
    plan_sha256: Sha256
    core_runtime_pipeline_sha256: Sha256
    lifecycle_source_sha256: Sha256
    lifecycle_pipeline_sha256: Sha256
    provider_calls_made: Literal[0] = 0
    maximum_provider_calls: Literal[1] = 1
    exact_request_retries_permitted: Literal[0] = 0
    graph_construction_mechanics_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    reliability_authority: Literal[False] = False
    generalization_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    prepared_sha256: Sha256

    @model_validator(mode="after")
    def validate_prepared(self) -> MetaSynContextualFrontierRecoveryLifecyclePreparedV2:
        if (
            self.plan_sha256 != self.plan.plan_sha256
            or self.core_runtime_pipeline_sha256 != self.plan.runtime_pipeline_sha256
            or self.lifecycle_pipeline_sha256
            != hash_canonical(
                {
                    "core_runtime_pipeline_sha256": self.core_runtime_pipeline_sha256,
                    "lifecycle_source_sha256": self.lifecycle_source_sha256,
                    "lifecycle_version": self.lifecycle_version,
                }
            )
        ):
            raise ValueError("recovery_lifecycle_v2_prepared_alias_mismatch")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(
            self,
            "prepared_sha256",
            "recovery_lifecycle_v2_prepared_hash_mismatch",
        )
        return self


class MetaSynContextualFrontierRecoveryLifecycleAuthorizationV2(_Frozen):
    authorization_version: Literal[
        "metasyn-contextual-frontier-recovery-lifecycle-authorization-v2"
    ] = AUTHORIZATION_VERSION
    prepared_sha256: Sha256
    plan_sha256: Sha256
    lifecycle_pipeline_sha256: Sha256
    request_key: str
    request_sha256: Sha256
    transport_request_sha256: Sha256
    maximum_provider_attempts: Literal[1] = 1
    maximum_cost_liability_usd_micros: Annotated[int, Field(ge=1)]
    configured_phase_budget_usd_micros: Annotated[int, Field(ge=1)]
    provider_calls_made_before_authorization: Literal[0] = 0
    application_retries_permitted: Literal[0] = 0
    sdk_retries_permitted: Literal[0] = 0
    fallback_requests_permitted: Literal[0] = 0
    orphan_or_ambiguous_attempt_retry_permitted: Literal[False] = False
    graph_construction_mechanics_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    authorization_sha256: Sha256

    @model_validator(mode="after")
    def validate_authorization(
        self,
    ) -> MetaSynContextualFrontierRecoveryLifecycleAuthorizationV2:
        if self.maximum_cost_liability_usd_micros > self.configured_phase_budget_usd_micros:
            raise ValueError("recovery_lifecycle_v2_budget_insufficient")
        _self_hash(
            self,
            "authorization_sha256",
            "recovery_lifecycle_v2_authorization_hash_mismatch",
        )
        return self


class MetaSynContextualFrontierRecoveryLifecycleIntentV2(_Frozen):
    intent_version: Literal["metasyn-contextual-frontier-recovery-lifecycle-intent-v2"] = (
        INTENT_VERSION
    )
    prepared_sha256: Sha256
    plan_sha256: Sha256
    lifecycle_pipeline_sha256: Sha256
    authorization_sha256: Sha256
    request_key: str
    request_sha256: Sha256
    transport_request_sha256: Sha256
    source_bearing: Literal[True] = True
    durable_before_provider_call: Literal[True] = True
    permitted_provider_attempts: Literal[1] = 1
    application_retries_permitted: Literal[0] = 0
    sdk_retries_permitted: Literal[0] = 0
    fallback_requests_permitted: Literal[0] = 0
    orphan_or_ambiguous_attempt_is_terminal: Literal[True] = True
    request_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    attempt_id: Sha256
    graph_construction_mechanics_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    intent_sha256: Sha256

    @model_validator(mode="after")
    def validate_intent(self) -> MetaSynContextualFrontierRecoveryLifecycleIntentV2:
        expected_attempt = hash_canonical(
            {
                "prepared_sha256": self.prepared_sha256,
                "authorization_sha256": self.authorization_sha256,
                "request_sha256": self.request_sha256,
                "transport_request_sha256": self.transport_request_sha256,
                "permitted_provider_attempts": 1,
            }
        )
        if self.attempt_id != expected_attempt:
            raise ValueError("recovery_lifecycle_v2_attempt_id_mismatch")
        _self_hash(self, "intent_sha256", "recovery_lifecycle_v2_intent_hash_mismatch")
        return self


class MetaSynContextualFrontierRecoveryLifecycleReceiptV2(_Frozen):
    receipt_version: Literal["metasyn-contextual-frontier-recovery-lifecycle-receipt-v2"] = (
        RECEIPT_VERSION
    )
    terminal_for_exact_transport_attempt: Literal[True] = True
    prepared_sha256: Sha256
    plan_sha256: Sha256
    lifecycle_pipeline_sha256: Sha256
    authorization_sha256: Sha256
    intent_sha256: Sha256
    attempt_id: Sha256
    request_key: str
    request_sha256: Sha256
    transport_request_sha256: Sha256
    provider_result_sha256: Sha256
    provider_result_artifact_sha256: Sha256
    provider_execution_binding_sha256: Sha256
    credential_archived: Literal[False] = False
    graph_construction_mechanics_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> MetaSynContextualFrontierRecoveryLifecycleReceiptV2:
        expected_binding = hash_canonical(
            {
                "attempt_id": self.attempt_id,
                "lifecycle_pipeline_sha256": self.lifecycle_pipeline_sha256,
                "provider_result_sha256": self.provider_result_sha256,
                "request_sha256": self.request_sha256,
                "transport_request_sha256": self.transport_request_sha256,
            }
        )
        if self.provider_execution_binding_sha256 != expected_binding:
            raise ValueError("recovery_lifecycle_v2_execution_binding_mismatch")
        _self_hash(self, "receipt_sha256", "recovery_lifecycle_v2_receipt_hash_mismatch")
        return self


LifecycleValidationStatus = Literal[
    "typed_graph_mechanics_observed",
    "scientific_abstention",
    "provider_result_failed",
    "contextual_validation_failed_closed",
]


class MetaSynContextualFrontierRecoveryLifecycleValidationV2(_Frozen):
    validation_version: Literal["metasyn-contextual-frontier-recovery-lifecycle-validation-v2"] = (
        VALIDATION_VERSION
    )
    status: LifecycleValidationStatus
    prepared_sha256: Sha256
    plan_sha256: Sha256
    lifecycle_pipeline_sha256: Sha256
    authorization_sha256: Sha256
    intent_sha256: Sha256
    request_sha256: Sha256
    transport_request_sha256: Sha256
    provider_result_sha256: Sha256
    provider_receipt_sha256: Sha256
    provider_execution_binding_sha256: Sha256
    provider_outcome: str
    core_evaluation_sha256: Sha256 | None
    response_sha256: Sha256 | None
    grounding_membership_sha256: Sha256 | None
    grounded_effect_sha256: Sha256 | None
    native_projection_sha256: Sha256 | None
    fresh_native_typed_graph_observed: bool
    failure_code: str | None
    graph_construction_mechanics_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    reliability_authority: Literal[False] = False
    generalization_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    validation_sha256: Sha256

    @model_validator(mode="after")
    def validate_validation(
        self,
    ) -> MetaSynContextualFrontierRecoveryLifecycleValidationV2:
        observed = self.status == "typed_graph_mechanics_observed"
        if (
            observed != self.fresh_native_typed_graph_observed
            or observed != (self.grounding_membership_sha256 is not None)
            or observed != (self.grounded_effect_sha256 is not None)
            or observed != (self.native_projection_sha256 is not None)
            or (self.core_evaluation_sha256 is None) != (self.response_sha256 is None)
        ):
            raise ValueError("recovery_lifecycle_v2_observation_status_mismatch")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(
            self,
            "validation_sha256",
            "recovery_lifecycle_v2_validation_hash_mismatch",
        )
        return self


LifecycleIncidentKind = Literal[
    "orphan_intent_observed_on_resume",
    "provider_call_raised_after_durable_intent",
    "provider_result_invalid_after_return",
    "provider_archive_invalid_on_resume",
]


class MetaSynContextualFrontierRecoveryLifecycleIncidentV2(_Frozen):
    incident_version: Literal["metasyn-contextual-frontier-recovery-lifecycle-incident-v2"] = (
        INCIDENT_VERSION
    )
    status: Literal["terminal_ambiguous_attempt_poison"] = "terminal_ambiguous_attempt_poison"
    incident_kind: LifecycleIncidentKind
    prepared_sha256: Sha256
    plan_sha256: Sha256
    lifecycle_pipeline_sha256: Sha256
    authorization_sha256: Sha256
    intent_sha256: Sha256
    attempt_id: Sha256
    request_key: str
    request_sha256: Sha256
    transport_request_sha256: Sha256
    response_observation: Literal[
        "unknown_after_orphaned_intent",
        "not_observed_by_executor",
        "observed_but_invalid",
    ]
    exception_type: str | None
    http_status: int | None
    provider_request_id: str | None
    possible_provider_attempts: Literal[1] = 1
    retry_this_request_permitted: Literal[False] = False
    fallback_requests_permitted: Literal[0] = 0
    credential_archived: Literal[False] = False
    graph_construction_mechanics_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    incident_sha256: Sha256

    @model_validator(mode="after")
    def validate_incident(self) -> MetaSynContextualFrontierRecoveryLifecycleIncidentV2:
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(self, "incident_sha256", "recovery_lifecycle_v2_incident_hash_mismatch")
        return self


LifecycleTerminalStatus = Literal[
    "typed_graph_mechanics_observed",
    "scientific_abstention",
    "provider_result_failed",
    "contextual_validation_failed_closed",
    "terminal_ambiguous_attempt_poison",
]


class MetaSynContextualFrontierRecoveryLifecycleTerminalReportV2(_Frozen):
    terminal_version: Literal["metasyn-contextual-frontier-recovery-lifecycle-terminal-v2"] = (
        TERMINAL_VERSION
    )
    terminal: Literal[True] = True
    status: LifecycleTerminalStatus
    prepared_sha256: Sha256
    plan_sha256: Sha256
    lifecycle_pipeline_sha256: Sha256
    authorization_sha256: Sha256
    attempted_request_keys: list[str]
    provider_result_sha256: Sha256 | None
    provider_receipt_sha256: Sha256 | None
    validation: MetaSynContextualFrontierRecoveryLifecycleValidationV2 | None
    validation_sha256: Sha256 | None
    incident: MetaSynContextualFrontierRecoveryLifecycleIncidentV2 | None
    incident_sha256: Sha256 | None
    provider_attempt_count_upper_bound: Literal[1] = 1
    provider_receipt_count: Literal[0, 1]
    exact_request_retries_permitted: Literal[0] = 0
    fallback_requests_permitted: Literal[0] = 0
    fresh_native_typed_graph_observed: bool
    graph_construction_mechanics_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    reliability_authority: Literal[False] = False
    generalization_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    report_sha256: Sha256

    @model_validator(mode="after")
    def validate_terminal(
        self,
    ) -> MetaSynContextualFrontierRecoveryLifecycleTerminalReportV2:
        if (self.validation is None) != (self.validation_sha256 is None):
            raise ValueError("recovery_lifecycle_v2_terminal_validation_presence")
        if (self.incident is None) != (self.incident_sha256 is None):
            raise ValueError("recovery_lifecycle_v2_terminal_incident_presence")
        if self.validation is not None and (
            self.validation_sha256 != self.validation.validation_sha256
        ):
            raise ValueError("recovery_lifecycle_v2_terminal_validation_alias")
        if self.incident is not None and self.incident_sha256 != self.incident.incident_sha256:
            raise ValueError("recovery_lifecycle_v2_terminal_incident_alias")
        ambiguous = self.status == "terminal_ambiguous_attempt_poison"
        observed = self.status == "typed_graph_mechanics_observed"
        if (
            ambiguous != (self.incident is not None)
            or ambiguous == (self.validation is not None)
            or observed != self.fresh_native_typed_graph_observed
        ):
            raise ValueError("recovery_lifecycle_v2_terminal_status_mismatch")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(self, "report_sha256", "recovery_lifecycle_v2_terminal_hash_mismatch")
        return self


class MetaSynContextualFrontierRecoveryLifecycleStatusV2(_Frozen):
    status_version: Literal["metasyn-contextual-frontier-recovery-lifecycle-status-v2"] = (
        STATUS_VERSION
    )
    status: Literal["prepared", "authorized", "terminal"]
    prepared_sha256: Sha256
    plan_sha256: Sha256
    lifecycle_pipeline_sha256: Sha256
    authorization_sha256: Sha256 | None
    terminal_report_sha256: Sha256 | None
    intent_count: Literal[0, 1]
    provider_result_count: Literal[0, 1]
    provider_receipt_count: Literal[0, 1]
    validation_count: Literal[0, 1]
    incident_count: Literal[0, 1]
    exact_request_retries_permitted: Literal[0] = 0
    graph_construction_mechanics_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    status_sha256: Sha256

    @model_validator(mode="after")
    def validate_status(self) -> MetaSynContextualFrontierRecoveryLifecycleStatusV2:
        expected = (
            "terminal"
            if self.terminal_report_sha256 is not None
            else "authorized"
            if self.authorization_sha256 is not None
            else "prepared"
        )
        if self.status != expected:
            raise ValueError("recovery_lifecycle_v2_status_mismatch")
        _self_hash(self, "status_sha256", "recovery_lifecycle_v2_status_hash_mismatch")
        return self


class MetaSynContextualFrontierRecoveryLifecycleWorkspaceValidationV2(_Frozen):
    workspace_validation_version: Literal[
        "metasyn-contextual-frontier-recovery-lifecycle-workspace-validation-v2"
    ] = WORKSPACE_VALIDATION_VERSION
    status: MetaSynContextualFrontierRecoveryLifecycleStatusV2
    status_sha256: Sha256
    external_plan_and_lifecycle_source_replayed: bool
    archive_replayed: Literal[True] = True
    workspace_directories_mode_700: Literal[True] = True
    workspace_files_mode_600: Literal[True] = True
    credential_archived: Literal[False] = False
    graph_construction_mechanics_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    reliability_authority: Literal[False] = False
    generalization_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    workspace_validation_sha256: Sha256

    @model_validator(mode="after")
    def validate_workspace(
        self,
    ) -> MetaSynContextualFrontierRecoveryLifecycleWorkspaceValidationV2:
        if self.status_sha256 != self.status.status_sha256:
            raise ValueError("recovery_lifecycle_v2_workspace_status_alias")
        _self_hash(
            self,
            "workspace_validation_sha256",
            "recovery_lifecycle_v2_workspace_validation_hash_mismatch",
        )
        return self


class MetaSynContextualFrontierRecoveryClientProtocolV2(Protocol):
    def generate(
        self, request: core.MetaSynContextualFrontierRecoveryRequestV2
    ) -> MetaSynContextualFrontierProviderResultV1: ...


class MetaSynContextualFrontierRecoveryClientV2:
    """Use the stable zero-retry v1 transport for the frozen recovery request."""

    def __init__(self, transport_config: MetaSynContextualFrontierConfigV1) -> None:
        self._delegate = MetaSynContextualFrontierClientV1(transport_config)

    def generate(
        self, request: core.MetaSynContextualFrontierRecoveryRequestV2
    ) -> MetaSynContextualFrontierProviderResultV1:
        canonical = core.MetaSynContextualFrontierRecoveryRequestV2.model_validate(
            request.model_dump(mode="json")
        )
        return self._delegate.generate(canonical.transport_request)


@contextmanager
def _workspace_lock(workspace: Path) -> Iterator[None]:
    path = workspace / _LOCK_NAME
    if path.is_symlink():
        raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
            "recovery_lifecycle_v2_lock_symlink"
        )
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _load_model(workspace: Path, relative: Path, model: Any, code: str) -> Any:
    return model.model_validate(_load_object(_checked_artifact(workspace, relative), code=code))


def _load_prepared_unlocked(
    workspace: Path,
) -> MetaSynContextualFrontierRecoveryLifecyclePreparedV2:
    return _load_model(
        workspace,
        _PREPARED,
        MetaSynContextualFrontierRecoveryLifecyclePreparedV2,
        "recovery_lifecycle_v2_prepared_invalid",
    )


def _freeze_prepared(
    *,
    repository_root: Path,
    plan: core.MetaSynContextualFrontierRecoveryPlanV2,
) -> MetaSynContextualFrontierRecoveryLifecyclePreparedV2:
    source_sha = sha256_file(_safe_repository_file(repository_root, LIFECYCLE_SOURCE_PATH))
    lifecycle_pipeline_sha = hash_canonical(
        {
            "core_runtime_pipeline_sha256": plan.runtime_pipeline_sha256,
            "lifecycle_source_sha256": source_sha,
            "lifecycle_version": LIFECYCLE_VERSION,
        }
    )
    payload = {
        "prepared_version": PREPARED_VERSION,
        "lifecycle_version": LIFECYCLE_VERSION,
        "status": "offline_prepared_zero_provider_calls",
        "plan": plan,
        "plan_sha256": plan.plan_sha256,
        "core_runtime_pipeline_sha256": plan.runtime_pipeline_sha256,
        "lifecycle_source_sha256": source_sha,
        "lifecycle_pipeline_sha256": lifecycle_pipeline_sha,
        "provider_calls_made": 0,
        "maximum_provider_calls": 1,
        "exact_request_retries_permitted": 0,
        "graph_construction_mechanics_authority": False,
        "extraction_accuracy_authority": False,
        "reliability_authority": False,
        "generalization_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierRecoveryLifecyclePreparedV2.model_validate(
        {**payload, "prepared_sha256": hash_canonical(payload)}
    )


def prepare_metasyn_contextual_frontier_recovery_lifecycle_v2(
    *,
    repository_root: Path,
    workspace: Path = DEFAULT_WORKSPACE,
    config_path: Path = core.DEFAULT_CONFIG_PATH,
) -> core.MetaSynContextualFrontierRecoveryPlanV2:
    """Freeze the recovery core and create a new zero-call private workspace."""

    plan = core.freeze_metasyn_contextual_frontier_recovery_plan_v2(
        repository_root=repository_root, config_path=config_path
    )
    prepared = _freeze_prepared(repository_root=repository_root, plan=plan)
    root = _fresh_workspace(workspace)
    os.chmod(root, 0o700)
    with _workspace_lock(root):
        _persist_json(_checked_artifact(root, _PREPARED), prepared)
    return plan


def load_metasyn_contextual_frontier_recovery_plan_v2(
    *, workspace: Path
) -> core.MetaSynContextualFrontierRecoveryPlanV2:
    root = _existing_workspace(workspace)
    with _workspace_lock(root):
        return _load_prepared_unlocked(root).plan


def _freeze_authorization(
    *,
    prepared: MetaSynContextualFrontierRecoveryLifecyclePreparedV2,
    phase_budget_usd_micros: int,
) -> MetaSynContextualFrontierRecoveryLifecycleAuthorizationV2:
    plan = prepared.plan
    payload = {
        "authorization_version": AUTHORIZATION_VERSION,
        "prepared_sha256": prepared.prepared_sha256,
        "plan_sha256": plan.plan_sha256,
        "lifecycle_pipeline_sha256": prepared.lifecycle_pipeline_sha256,
        "request_key": _request_key(plan),
        "request_sha256": _request(plan).request_sha256,
        "transport_request_sha256": _transport_request(plan).request_sha256,
        "maximum_provider_attempts": 1,
        "maximum_cost_liability_usd_micros": plan.hard_cost_liability_usd_micros,
        "configured_phase_budget_usd_micros": phase_budget_usd_micros,
        "provider_calls_made_before_authorization": 0,
        "application_retries_permitted": 0,
        "sdk_retries_permitted": 0,
        "fallback_requests_permitted": 0,
        "orphan_or_ambiguous_attempt_retry_permitted": False,
        "graph_construction_mechanics_authority": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    try:
        return MetaSynContextualFrontierRecoveryLifecycleAuthorizationV2.model_validate(
            {**payload, "authorization_sha256": hash_canonical(payload)}
        )
    except ValueError as exc:
        raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
            "recovery_lifecycle_v2_phase_budget_insufficient"
        ) from exc


def authorize_metasyn_contextual_frontier_recovery_lifecycle_v2(
    *, workspace: Path, phase_budget_usd_micros: int
) -> MetaSynContextualFrontierRecoveryLifecycleAuthorizationV2:
    root = _existing_workspace(workspace)
    with _workspace_lock(root):
        prepared = _load_prepared_unlocked(root)
        path = _checked_artifact(root, _AUTHORIZED)
        expected = _freeze_authorization(
            prepared=prepared, phase_budget_usd_micros=phase_budget_usd_micros
        )
        if path.exists():
            observed = _load_model(
                root,
                _AUTHORIZED,
                MetaSynContextualFrontierRecoveryLifecycleAuthorizationV2,
                "recovery_lifecycle_v2_authorization_invalid",
            )
            if observed != expected:
                raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
                    "recovery_lifecycle_v2_authorization_drift"
                )
            return observed
        if any(
            _checked_artifact(root, item).exists()
            for item in (_INTENT, _PROVIDER_RESULT, _RECEIPT, _VALIDATION, _INCIDENT, _TERMINAL)
        ):
            raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
                "recovery_lifecycle_v2_authorization_after_attempt_forbidden"
            )
        _persist_json(path, expected)
        return expected


def _load_authorization_unlocked(
    *,
    workspace: Path,
    prepared: MetaSynContextualFrontierRecoveryLifecyclePreparedV2,
) -> MetaSynContextualFrontierRecoveryLifecycleAuthorizationV2:
    authorization = _load_model(
        workspace,
        _AUTHORIZED,
        MetaSynContextualFrontierRecoveryLifecycleAuthorizationV2,
        "recovery_lifecycle_v2_authorization_invalid",
    )
    if (
        authorization.prepared_sha256 != prepared.prepared_sha256
        or authorization.plan_sha256 != prepared.plan_sha256
        or authorization.lifecycle_pipeline_sha256 != prepared.lifecycle_pipeline_sha256
        or authorization.request_sha256 != _request(prepared.plan).request_sha256
        or authorization.transport_request_sha256
        != _transport_request(prepared.plan).request_sha256
    ):
        raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
            "recovery_lifecycle_v2_authorization_binding_mismatch"
        )
    return authorization


def _freeze_intent(
    *,
    prepared: MetaSynContextualFrontierRecoveryLifecyclePreparedV2,
    authorization: MetaSynContextualFrontierRecoveryLifecycleAuthorizationV2,
) -> MetaSynContextualFrontierRecoveryLifecycleIntentV2:
    plan = prepared.plan
    attempt = hash_canonical(
        {
            "prepared_sha256": prepared.prepared_sha256,
            "authorization_sha256": authorization.authorization_sha256,
            "request_sha256": _request(plan).request_sha256,
            "transport_request_sha256": _transport_request(plan).request_sha256,
            "permitted_provider_attempts": 1,
        }
    )
    payload = {
        "intent_version": INTENT_VERSION,
        "prepared_sha256": prepared.prepared_sha256,
        "plan_sha256": plan.plan_sha256,
        "lifecycle_pipeline_sha256": prepared.lifecycle_pipeline_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "request_key": _request_key(plan),
        "request_sha256": _request(plan).request_sha256,
        "transport_request_sha256": _transport_request(plan).request_sha256,
        "source_bearing": True,
        "durable_before_provider_call": True,
        "permitted_provider_attempts": 1,
        "application_retries_permitted": 0,
        "sdk_retries_permitted": 0,
        "fallback_requests_permitted": 0,
        "orphan_or_ambiguous_attempt_is_terminal": True,
        "request_cost_ceiling_usd_micros": plan.hard_cost_liability_usd_micros,
        "attempt_id": attempt,
        "graph_construction_mechanics_authority": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierRecoveryLifecycleIntentV2.model_validate(
        {**payload, "intent_sha256": hash_canonical(payload)}
    )


def _validate_provider_result(
    *,
    plan: core.MetaSynContextualFrontierRecoveryPlanV2,
    value: Any,
) -> MetaSynContextualFrontierProviderResultV1:
    result = MetaSynContextualFrontierProviderResultV1.model_validate(
        value.model_dump(mode="json")
        if isinstance(value, MetaSynContextualFrontierProviderResultV1)
        else value
    )
    transport = _transport_request(plan)
    if (
        result.request_sha256 != transport.request_sha256
        or result.identity_sha256 != transport.identity_sha256
        or result.config_sha256 != transport.config_sha256
        or result.wire_call_surface_sha256 != transport.wire_call_surface_sha256
        or result.original_schema_sha256 != transport.original_schema_sha256
        or result.wire_schema_sha256 != transport.wire_schema_sha256
        or result.model_system_sha256 != transport.model_system_sha256
        or result.prompt_sha256 != transport.prompt_sha256
    ):
        raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
            "recovery_lifecycle_v2_provider_result_binding_mismatch"
        )
    return result


def _freeze_receipt(
    *,
    prepared: MetaSynContextualFrontierRecoveryLifecyclePreparedV2,
    authorization: MetaSynContextualFrontierRecoveryLifecycleAuthorizationV2,
    intent: MetaSynContextualFrontierRecoveryLifecycleIntentV2,
    result: MetaSynContextualFrontierProviderResultV1,
    result_artifact_sha256: str,
) -> MetaSynContextualFrontierRecoveryLifecycleReceiptV2:
    plan = prepared.plan
    binding = hash_canonical(
        {
            "attempt_id": intent.attempt_id,
            "lifecycle_pipeline_sha256": prepared.lifecycle_pipeline_sha256,
            "provider_result_sha256": result.result_sha256,
            "request_sha256": _request(plan).request_sha256,
            "transport_request_sha256": _transport_request(plan).request_sha256,
        }
    )
    payload = {
        "receipt_version": RECEIPT_VERSION,
        "terminal_for_exact_transport_attempt": True,
        "prepared_sha256": prepared.prepared_sha256,
        "plan_sha256": plan.plan_sha256,
        "lifecycle_pipeline_sha256": prepared.lifecycle_pipeline_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "intent_sha256": intent.intent_sha256,
        "attempt_id": intent.attempt_id,
        "request_key": _request_key(plan),
        "request_sha256": _request(plan).request_sha256,
        "transport_request_sha256": _transport_request(plan).request_sha256,
        "provider_result_sha256": result.result_sha256,
        "provider_result_artifact_sha256": result_artifact_sha256,
        "provider_execution_binding_sha256": binding,
        "credential_archived": False,
        "graph_construction_mechanics_authority": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierRecoveryLifecycleReceiptV2.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def _freeze_validation(
    *,
    prepared: MetaSynContextualFrontierRecoveryLifecyclePreparedV2,
    authorization: MetaSynContextualFrontierRecoveryLifecycleAuthorizationV2,
    intent: MetaSynContextualFrontierRecoveryLifecycleIntentV2,
    result: MetaSynContextualFrontierProviderResultV1,
    receipt: MetaSynContextualFrontierRecoveryLifecycleReceiptV2,
) -> MetaSynContextualFrontierRecoveryLifecycleValidationV2:
    evaluation = None
    failure_code = None
    if result.outcome != "completed":
        status: LifecycleValidationStatus = "provider_result_failed"
        failure_code = result.failure_code or result.outcome
    else:
        try:
            if result.parsed_json is None:
                raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
                    "recovery_lifecycle_v2_completed_json_missing"
                )
            evaluation = core.evaluate_metasyn_contextual_frontier_recovery_response_v2(
                plan=prepared.plan,
                raw_response=result.parsed_json,
                provider_execution_binding_sha256=(receipt.provider_execution_binding_sha256),
            )
            if bool(getattr(evaluation, "graph_construction_mechanics_authority", False)):
                raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
                    "recovery_lifecycle_v2_core_authority_must_be_false"
                )
            status = (
                "typed_graph_mechanics_observed"
                if evaluation.status == "typed_graph_mechanics_completed"
                else "scientific_abstention"
            )
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
                raise
            evaluation = None
            status = "contextual_validation_failed_closed"
            failure_code = _safe_exception_type(exc)
    payload = {
        "validation_version": VALIDATION_VERSION,
        "status": status,
        "prepared_sha256": prepared.prepared_sha256,
        "plan_sha256": prepared.plan_sha256,
        "lifecycle_pipeline_sha256": prepared.lifecycle_pipeline_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "intent_sha256": intent.intent_sha256,
        "request_sha256": _request(prepared.plan).request_sha256,
        "transport_request_sha256": _transport_request(prepared.plan).request_sha256,
        "provider_result_sha256": result.result_sha256,
        "provider_receipt_sha256": receipt.receipt_sha256,
        "provider_execution_binding_sha256": receipt.provider_execution_binding_sha256,
        "provider_outcome": result.outcome,
        "core_evaluation_sha256": (
            evaluation.evaluation_sha256 if evaluation is not None else None
        ),
        "response_sha256": (evaluation.response_sha256 if evaluation is not None else None),
        "grounding_membership_sha256": (
            evaluation.grounding_membership_sha256 if evaluation is not None else None
        ),
        "grounded_effect_sha256": (
            evaluation.grounded_effect_sha256 if evaluation is not None else None
        ),
        "native_projection_sha256": (
            evaluation.native_projection_sha256 if evaluation is not None else None
        ),
        "fresh_native_typed_graph_observed": status == "typed_graph_mechanics_observed",
        "failure_code": failure_code,
        "graph_construction_mechanics_authority": False,
        "extraction_accuracy_authority": False,
        "reliability_authority": False,
        "generalization_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierRecoveryLifecycleValidationV2.model_validate(
        {**payload, "validation_sha256": hash_canonical(payload)}
    )


def _freeze_incident(
    *,
    prepared: MetaSynContextualFrontierRecoveryLifecyclePreparedV2,
    authorization: MetaSynContextualFrontierRecoveryLifecycleAuthorizationV2,
    intent: MetaSynContextualFrontierRecoveryLifecycleIntentV2,
    kind: LifecycleIncidentKind,
    response_observation: str,
    exc: BaseException | None = None,
) -> MetaSynContextualFrontierRecoveryLifecycleIncidentV2:
    payload = {
        "incident_version": INCIDENT_VERSION,
        "status": "terminal_ambiguous_attempt_poison",
        "incident_kind": kind,
        "prepared_sha256": prepared.prepared_sha256,
        "plan_sha256": prepared.plan_sha256,
        "lifecycle_pipeline_sha256": prepared.lifecycle_pipeline_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "intent_sha256": intent.intent_sha256,
        "attempt_id": intent.attempt_id,
        "request_key": _request_key(prepared.plan),
        "request_sha256": _request(prepared.plan).request_sha256,
        "transport_request_sha256": _transport_request(prepared.plan).request_sha256,
        "response_observation": response_observation,
        "exception_type": _safe_exception_type(exc) if exc is not None else None,
        "http_status": _safe_status(exc) if exc is not None else None,
        "provider_request_id": _safe_request_id(exc) if exc is not None else None,
        "possible_provider_attempts": 1,
        "retry_this_request_permitted": False,
        "fallback_requests_permitted": 0,
        "credential_archived": False,
        "graph_construction_mechanics_authority": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierRecoveryLifecycleIncidentV2.model_validate(
        {**payload, "incident_sha256": hash_canonical(payload)}
    )


def _freeze_terminal(
    *,
    prepared: MetaSynContextualFrontierRecoveryLifecyclePreparedV2,
    authorization: MetaSynContextualFrontierRecoveryLifecycleAuthorizationV2,
    result: MetaSynContextualFrontierProviderResultV1 | None,
    receipt: MetaSynContextualFrontierRecoveryLifecycleReceiptV2 | None,
    validation: MetaSynContextualFrontierRecoveryLifecycleValidationV2 | None,
    incident: MetaSynContextualFrontierRecoveryLifecycleIncidentV2 | None,
) -> MetaSynContextualFrontierRecoveryLifecycleTerminalReportV2:
    status: LifecycleTerminalStatus = (
        "terminal_ambiguous_attempt_poison" if incident is not None else validation.status  # type: ignore[union-attr]
    )
    payload = {
        "terminal_version": TERMINAL_VERSION,
        "terminal": True,
        "status": status,
        "prepared_sha256": prepared.prepared_sha256,
        "plan_sha256": prepared.plan_sha256,
        "lifecycle_pipeline_sha256": prepared.lifecycle_pipeline_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "attempted_request_keys": [_request_key(prepared.plan)],
        "provider_result_sha256": result.result_sha256 if result is not None else None,
        "provider_receipt_sha256": receipt.receipt_sha256 if receipt is not None else None,
        "validation": validation,
        "validation_sha256": (validation.validation_sha256 if validation is not None else None),
        "incident": incident,
        "incident_sha256": incident.incident_sha256 if incident is not None else None,
        "provider_attempt_count_upper_bound": 1,
        "provider_receipt_count": 1 if receipt is not None else 0,
        "exact_request_retries_permitted": 0,
        "fallback_requests_permitted": 0,
        "fresh_native_typed_graph_observed": bool(
            validation is not None and validation.fresh_native_typed_graph_observed
        ),
        "graph_construction_mechanics_authority": False,
        "extraction_accuracy_authority": False,
        "reliability_authority": False,
        "generalization_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierRecoveryLifecycleTerminalReportV2.model_validate(
        {**payload, "report_sha256": hash_canonical(payload)}
    )


def _write_terminal(
    *, workspace: Path, report: MetaSynContextualFrontierRecoveryLifecycleTerminalReportV2
) -> MetaSynContextualFrontierRecoveryLifecycleTerminalReportV2:
    path = _checked_artifact(workspace, _TERMINAL)
    if path.exists():
        observed = _load_model(
            workspace,
            _TERMINAL,
            MetaSynContextualFrontierRecoveryLifecycleTerminalReportV2,
            "recovery_lifecycle_v2_terminal_invalid",
        )
        if observed != report:
            raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
                "recovery_lifecycle_v2_terminal_drift"
            )
        return observed
    _persist_json(path, report)
    return report


def _terminal_from_incident(
    *,
    workspace: Path,
    prepared: MetaSynContextualFrontierRecoveryLifecyclePreparedV2,
    authorization: MetaSynContextualFrontierRecoveryLifecycleAuthorizationV2,
    intent: MetaSynContextualFrontierRecoveryLifecycleIntentV2,
    kind: LifecycleIncidentKind,
    observation: str,
    exc: BaseException | None = None,
) -> MetaSynContextualFrontierRecoveryLifecycleTerminalReportV2:
    incident = _freeze_incident(
        prepared=prepared,
        authorization=authorization,
        intent=intent,
        kind=kind,
        response_observation=observation,
        exc=exc,
    )
    _persist_json(_checked_artifact(workspace, _INCIDENT), incident)
    return _write_terminal(
        workspace=workspace,
        report=_freeze_terminal(
            prepared=prepared,
            authorization=authorization,
            result=None,
            receipt=None,
            validation=None,
            incident=incident,
        ),
    )


def _load_result_unlocked(
    *, workspace: Path, plan: core.MetaSynContextualFrontierRecoveryPlanV2
) -> MetaSynContextualFrontierProviderResultV1:
    return _validate_provider_result(
        plan=plan,
        value=_load_object(
            _checked_artifact(workspace, _PROVIDER_RESULT),
            code="recovery_lifecycle_v2_provider_result_invalid",
        ),
    )


def _complete_from_result(
    *,
    workspace: Path,
    prepared: MetaSynContextualFrontierRecoveryLifecyclePreparedV2,
    authorization: MetaSynContextualFrontierRecoveryLifecycleAuthorizationV2,
    intent: MetaSynContextualFrontierRecoveryLifecycleIntentV2,
    result: MetaSynContextualFrontierProviderResultV1,
) -> MetaSynContextualFrontierRecoveryLifecycleTerminalReportV2:
    result_path = _checked_artifact(workspace, _PROVIDER_RESULT)
    receipt = _freeze_receipt(
        prepared=prepared,
        authorization=authorization,
        intent=intent,
        result=result,
        result_artifact_sha256=sha256_file(result_path),
    )
    receipt_path = _checked_artifact(workspace, _RECEIPT)
    if receipt_path.exists():
        observed_receipt = _load_model(
            workspace,
            _RECEIPT,
            MetaSynContextualFrontierRecoveryLifecycleReceiptV2,
            "recovery_lifecycle_v2_receipt_invalid",
        )
        if observed_receipt != receipt:
            raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
                "recovery_lifecycle_v2_receipt_replay_mismatch"
            )
    else:
        _persist_json(receipt_path, receipt)
    validation = _freeze_validation(
        prepared=prepared,
        authorization=authorization,
        intent=intent,
        result=result,
        receipt=receipt,
    )
    validation_path = _checked_artifact(workspace, _VALIDATION)
    if validation_path.exists():
        observed_validation = _load_model(
            workspace,
            _VALIDATION,
            MetaSynContextualFrontierRecoveryLifecycleValidationV2,
            "recovery_lifecycle_v2_validation_invalid",
        )
        if observed_validation != validation:
            raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
                "recovery_lifecycle_v2_validation_replay_mismatch"
            )
    else:
        _persist_json(validation_path, validation)
    return _write_terminal(
        workspace=workspace,
        report=_freeze_terminal(
            prepared=prepared,
            authorization=authorization,
            result=result,
            receipt=receipt,
            validation=validation,
            incident=None,
        ),
    )


def execute_metasyn_contextual_frontier_recovery_lifecycle_v2(
    *, workspace: Path, client: MetaSynContextualFrontierRecoveryClientProtocolV2
) -> MetaSynContextualFrontierRecoveryLifecycleTerminalReportV2:
    """Execute at most one authorized request, with no retry or fallback."""

    root = _existing_workspace(workspace)
    with _workspace_lock(root):
        terminal_path = _checked_artifact(root, _TERMINAL)
        if terminal_path.exists():
            return _load_model(
                root,
                _TERMINAL,
                MetaSynContextualFrontierRecoveryLifecycleTerminalReportV2,
                "recovery_lifecycle_v2_terminal_invalid",
            )
        prepared = _load_prepared_unlocked(root)
        authorization = _load_authorization_unlocked(workspace=root, prepared=prepared)
        expected_intent = _freeze_intent(prepared=prepared, authorization=authorization)
        intent_path = _checked_artifact(root, _INTENT)
        result_path = _checked_artifact(root, _PROVIDER_RESULT)
        if intent_path.exists():
            intent = _load_model(
                root,
                _INTENT,
                MetaSynContextualFrontierRecoveryLifecycleIntentV2,
                "recovery_lifecycle_v2_intent_invalid",
            )
            if intent != expected_intent:
                raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
                    "recovery_lifecycle_v2_intent_replay_mismatch"
                )
            if result_path.exists():
                try:
                    result = _load_result_unlocked(workspace=root, plan=prepared.plan)
                    return _complete_from_result(
                        workspace=root,
                        prepared=prepared,
                        authorization=authorization,
                        intent=intent,
                        result=result,
                    )
                except Exception as exc:
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
                        raise
                    return _terminal_from_incident(
                        workspace=root,
                        prepared=prepared,
                        authorization=authorization,
                        intent=intent,
                        kind="provider_archive_invalid_on_resume",
                        observation="observed_but_invalid",
                        exc=exc,
                    )
            return _terminal_from_incident(
                workspace=root,
                prepared=prepared,
                authorization=authorization,
                intent=intent,
                kind="orphan_intent_observed_on_resume",
                observation="unknown_after_orphaned_intent",
            )

        # Provider liability begins only after this private artifact is durable.
        _persist_json(intent_path, expected_intent)
        try:
            raw_result = client.generate(prepared.plan.request)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
                raise
            return _terminal_from_incident(
                workspace=root,
                prepared=prepared,
                authorization=authorization,
                intent=expected_intent,
                kind="provider_call_raised_after_durable_intent",
                observation="not_observed_by_executor",
                exc=exc,
            )
        try:
            result = _validate_provider_result(plan=prepared.plan, value=raw_result)
            _persist_json(result_path, result)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
                raise
            return _terminal_from_incident(
                workspace=root,
                prepared=prepared,
                authorization=authorization,
                intent=expected_intent,
                kind="provider_result_invalid_after_return",
                observation="observed_but_invalid",
                exc=exc,
            )
        return _complete_from_result(
            workspace=root,
            prepared=prepared,
            authorization=authorization,
            intent=expected_intent,
            result=result,
        )


def _status_unlocked(
    workspace: Path,
) -> MetaSynContextualFrontierRecoveryLifecycleStatusV2:
    prepared = _load_prepared_unlocked(workspace)
    authorization = None
    terminal = None
    if _checked_artifact(workspace, _AUTHORIZED).exists():
        authorization = _load_authorization_unlocked(workspace=workspace, prepared=prepared)
    if _checked_artifact(workspace, _TERMINAL).exists():
        terminal = _load_model(
            workspace,
            _TERMINAL,
            MetaSynContextualFrontierRecoveryLifecycleTerminalReportV2,
            "recovery_lifecycle_v2_terminal_invalid",
        )
    counts = {
        "intent_count": int(_checked_artifact(workspace, _INTENT).exists()),
        "provider_result_count": int(_checked_artifact(workspace, _PROVIDER_RESULT).exists()),
        "provider_receipt_count": int(_checked_artifact(workspace, _RECEIPT).exists()),
        "validation_count": int(_checked_artifact(workspace, _VALIDATION).exists()),
        "incident_count": int(_checked_artifact(workspace, _INCIDENT).exists()),
    }
    payload = {
        "status_version": STATUS_VERSION,
        "status": "terminal" if terminal else "authorized" if authorization else "prepared",
        "prepared_sha256": prepared.prepared_sha256,
        "plan_sha256": prepared.plan_sha256,
        "lifecycle_pipeline_sha256": prepared.lifecycle_pipeline_sha256,
        "authorization_sha256": (authorization.authorization_sha256 if authorization else None),
        "terminal_report_sha256": terminal.report_sha256 if terminal else None,
        **counts,
        "exact_request_retries_permitted": 0,
        "graph_construction_mechanics_authority": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierRecoveryLifecycleStatusV2.model_validate(
        {**payload, "status_sha256": hash_canonical(payload)}
    )


def status_metasyn_contextual_frontier_recovery_lifecycle_v2(
    *, workspace: Path = DEFAULT_WORKSPACE
) -> MetaSynContextualFrontierRecoveryLifecycleStatusV2:
    root = _existing_workspace(workspace)
    with _workspace_lock(root):
        return _status_unlocked(root)


def _audit_modes_and_symlinks(workspace: Path) -> None:
    for directory, dirnames, filenames in os.walk(workspace, followlinks=False):
        current = Path(directory)
        if stat.S_IMODE(current.lstat().st_mode) != 0o700:
            raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
                "recovery_lifecycle_v2_directory_mode_invalid"
            )
        for name in [*dirnames, *filenames]:
            path = current / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
                    "recovery_lifecycle_v2_symlink_forbidden"
                )
            if stat.S_ISDIR(mode):
                if stat.S_IMODE(mode) != 0o700:
                    raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
                        "recovery_lifecycle_v2_directory_mode_invalid"
                    )
            elif stat.S_ISREG(mode) and stat.S_IMODE(mode) != 0o600:
                raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
                    "recovery_lifecycle_v2_file_mode_invalid"
                )


def validate_metasyn_contextual_frontier_recovery_lifecycle_v2(
    *,
    repository_root: Path,
    workspace: Path = DEFAULT_WORKSPACE,
    external_replay: bool = True,
) -> MetaSynContextualFrontierRecoveryLifecycleWorkspaceValidationV2:
    """Replay the private archive and optionally rebuild all frozen source bytes."""

    root = _existing_workspace(workspace)
    _audit_modes_and_symlinks(root)
    with _workspace_lock(root):
        prepared = _load_prepared_unlocked(root)
        if external_replay:
            replayed_plan = core.freeze_metasyn_contextual_frontier_recovery_plan_v2(
                repository_root=repository_root
            )
            replayed_prepared = _freeze_prepared(
                repository_root=repository_root, plan=replayed_plan
            )
            if replayed_prepared != prepared:
                raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
                    "recovery_lifecycle_v2_external_replay_mismatch"
                )
        status = _status_unlocked(root)
        # If a terminal archive exists, replay all mutually bound hashes and the
        # trusted evaluator without making any provider call.
        if status.terminal_report_sha256 is not None:
            terminal = _load_model(
                root,
                _TERMINAL,
                MetaSynContextualFrontierRecoveryLifecycleTerminalReportV2,
                "recovery_lifecycle_v2_terminal_invalid",
            )
            authorization = _load_authorization_unlocked(workspace=root, prepared=prepared)
            intent = _load_model(
                root,
                _INTENT,
                MetaSynContextualFrontierRecoveryLifecycleIntentV2,
                "recovery_lifecycle_v2_intent_invalid",
            )
            if terminal.incident is not None:
                observed_incident = _load_model(
                    root,
                    _INCIDENT,
                    MetaSynContextualFrontierRecoveryLifecycleIncidentV2,
                    "recovery_lifecycle_v2_incident_invalid",
                )
                if terminal.incident != observed_incident:
                    raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
                        "recovery_lifecycle_v2_incident_terminal_mismatch"
                    )
            else:
                result = _load_result_unlocked(workspace=root, plan=prepared.plan)
                replayed_terminal = _complete_from_result(
                    workspace=root,
                    prepared=prepared,
                    authorization=authorization,
                    intent=intent,
                    result=result,
                )
                if replayed_terminal != terminal:
                    raise MetaSynContextualFrontierRecoveryLifecycleV2Error(
                        "recovery_lifecycle_v2_terminal_replay_mismatch"
                    )
        payload = {
            "workspace_validation_version": WORKSPACE_VALIDATION_VERSION,
            "status": status,
            "status_sha256": status.status_sha256,
            "external_plan_and_lifecycle_source_replayed": external_replay,
            "archive_replayed": True,
            "workspace_directories_mode_700": True,
            "workspace_files_mode_600": True,
            "credential_archived": False,
            "graph_construction_mechanics_authority": False,
            "extraction_accuracy_authority": False,
            "reliability_authority": False,
            "generalization_authority": False,
            "synthesis_input_authority": False,
            "scientific_synthesis_authority": False,
            "calibration_authority": False,
            "claim_release_authority": False,
        }
        return MetaSynContextualFrontierRecoveryLifecycleWorkspaceValidationV2.model_validate(
            {**payload, "workspace_validation_sha256": hash_canonical(payload)}
        )
