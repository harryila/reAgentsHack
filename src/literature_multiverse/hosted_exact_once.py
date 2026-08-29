"""Generic, staged, exactly-once execution for bounded hosted model requests.

The provider adapter freezes the complete credential-free request.  This module adds
the filesystem transaction boundary: an exact phase roster is cost-authorized before
calls, each intent is durably written before transport, and an orphaned intent is
terminally poisoned rather than retried.  Scientific response validation remains in
the owning pipeline and must replay the provider receipt against its own schema and
source context.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from pydantic import Field, field_validator, model_validator

from literature_multiverse.anthropic_bounded_generation import (
    AnthropicBoundedRequestV1,
    AnthropicBoundedResultV1,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel

EXACT_ONCE_INTENT_VERSION = "hosted-exact-once-intent-v1"
EXACT_ONCE_AUTHORIZATION_VERSION = "hosted-exact-once-cost-authorization-v1"
EXACT_ONCE_RECEIPT_VERSION = "hosted-exact-once-provider-receipt-v1"
EXACT_ONCE_INCIDENT_VERSION = "hosted-exact-once-ambiguity-incident-v1"

CallPhase = Literal[
    "preflight",
    "source_free_preflight",
    "smoke",
    "smoke_inventory",
    "smoke_packet",
    "inventory",
    "packet",
    "evaluation",
]
IncidentKind = Literal[
    "orphan_intent_observed_on_resume",
    "provider_call_raised_after_durable_intent",
    "provider_result_invalid_after_return",
]
ResponseObservation = Literal[
    "unknown_after_orphaned_intent",
    "not_observed_by_executor",
    "observed_but_invalid",
]


class HostedExactOnceError(ValueError):
    """The exact-call roster, cost boundary, or durable state is unsafe."""


class HostedBoundedClientProtocol(Protocol):
    def generate(self, request: AnthropicBoundedRequestV1) -> AnthropicBoundedResultV1:
        """Make exactly one provider attempt and return a closed provider result."""


def _usd_micros(value: Decimal) -> int:
    return int(
        (value * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING)
    )


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HostedExactOnceError("hosted_exact_once_artifact_unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HostedExactOnceError("hosted_exact_once_artifact_invalid") from exc
    if not isinstance(value, dict):
        raise HostedExactOnceError("hosted_exact_once_artifact_not_object")
    return value


def _canonical_workspace(workspace: Path, *, create: bool = False) -> Path:
    if workspace.is_symlink():
        raise HostedExactOnceError("hosted_exact_once_workspace_symlink")
    if create:
        workspace.mkdir(parents=True, exist_ok=True)
    try:
        resolved = workspace.resolve(strict=True)
    except OSError as exc:
        raise HostedExactOnceError("hosted_exact_once_workspace_missing") from exc
    if not resolved.is_dir():
        raise HostedExactOnceError("hosted_exact_once_workspace_not_directory")
    return resolved


@contextmanager
def _workspace_lock(workspace: Path) -> Iterator[None]:
    lock_path = workspace / ".hosted-exact-once.lock"
    if lock_path.is_symlink():
        raise HostedExactOnceError("hosted_exact_once_lock_symlink")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class HostedExactOnceIntentV1(ContractModel):
    intent_version: Literal["hosted-exact-once-intent-v1"] = EXACT_ONCE_INTENT_VERSION
    execution_bundle_sha256: str
    phase: CallPhase
    request_key: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{2,127}$")]
    source_bearing: bool
    context_binding_sha256: str
    request: dict[str, Any]
    request_sha256: str
    provider_identity_sha256: str
    provider_config_sha256: str
    request_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    permitted_provider_attempts: Literal[1] = 1
    application_retries_permitted: Literal[0] = 0
    sdk_retries_permitted: Literal[0] = 0
    orphan_or_ambiguous_attempt_is_terminal: Literal[True] = True
    attempt_id: str
    intent_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "context_binding_sha256",
        "request_sha256",
        "provider_identity_sha256",
        "provider_config_sha256",
        "attempt_id",
        "intent_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"hosted_exact_once_hash_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_intent(self) -> HostedExactOnceIntentV1:
        try:
            request = AnthropicBoundedRequestV1.model_validate(self.request)
        except ValueError as exc:
            raise ValueError("hosted_exact_once_request_invalid") from exc
        if (
            request.request_key != self.request_key
            or request.request_sha256 != self.request_sha256
            or request.identity_sha256 != self.provider_identity_sha256
            or request.config_sha256 != self.provider_config_sha256
            or _usd_micros(request.cost_ceiling.request_cost_ceiling_usd)
            != self.request_cost_ceiling_usd_micros
        ):
            raise ValueError("hosted_exact_once_request_alias_mismatch")
        expected_attempt = hash_canonical(
            {
                "execution_bundle_sha256": self.execution_bundle_sha256,
                "request_sha256": self.request_sha256,
                "context_binding_sha256": self.context_binding_sha256,
                "permitted_provider_attempts": 1,
                "application_retries_permitted": 0,
                "sdk_retries_permitted": 0,
            }
        )
        if self.attempt_id != expected_attempt:
            raise ValueError("hosted_exact_once_attempt_id_mismatch")
        payload = self.model_dump(mode="json", exclude={"intent_sha256"})
        if self.intent_sha256 != hash_canonical(payload):
            raise ValueError("hosted_exact_once_intent_hash_mismatch")
        return self


def freeze_hosted_exact_once_intent(
    *,
    execution_bundle_sha256: str,
    phase: CallPhase,
    source_bearing: bool,
    context_binding_sha256: str,
    request: AnthropicBoundedRequestV1 | Mapping[str, Any],
) -> HostedExactOnceIntentV1:
    canonical = AnthropicBoundedRequestV1.model_validate(request)
    attempt_id = hash_canonical(
        {
            "execution_bundle_sha256": execution_bundle_sha256,
            "request_sha256": canonical.request_sha256,
            "context_binding_sha256": context_binding_sha256,
            "permitted_provider_attempts": 1,
            "application_retries_permitted": 0,
            "sdk_retries_permitted": 0,
        }
    )
    payload: dict[str, Any] = {
        "intent_version": EXACT_ONCE_INTENT_VERSION,
        "execution_bundle_sha256": execution_bundle_sha256,
        "phase": phase,
        "request_key": canonical.request_key,
        "source_bearing": source_bearing,
        "context_binding_sha256": context_binding_sha256,
        "request": canonical.model_dump(mode="json"),
        "request_sha256": canonical.request_sha256,
        "provider_identity_sha256": canonical.identity_sha256,
        "provider_config_sha256": canonical.config_sha256,
        "request_cost_ceiling_usd_micros": _usd_micros(
            canonical.cost_ceiling.request_cost_ceiling_usd
        ),
        "permitted_provider_attempts": 1,
        "application_retries_permitted": 0,
        "sdk_retries_permitted": 0,
        "orphan_or_ambiguous_attempt_is_terminal": True,
        "attempt_id": attempt_id,
    }
    return HostedExactOnceIntentV1.model_validate(
        {**payload, "intent_sha256": hash_canonical(payload)}
    )


class HostedAuthorizedCallV1(ContractModel):
    request_key: str
    intent_sha256: str
    request_sha256: str
    source_bearing: bool
    request_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]

    @field_validator("intent_sha256", "request_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("hosted_exact_once_authorized_call_hash_invalid")
        return value


class HostedExactOnceCostAuthorizationV1(ContractModel):
    authorization_version: Literal[
        "hosted-exact-once-cost-authorization-v1"
    ] = EXACT_ONCE_AUTHORIZATION_VERSION
    execution_bundle_sha256: str
    phase: CallPhase
    authorized_calls: Annotated[list[HostedAuthorizedCallV1], Field(min_length=1)]
    authorized_call_count: Annotated[int, Field(ge=1)]
    authorized_intent_roster_sha256: str
    source_bearing_call_count: Annotated[int, Field(ge=0)]
    cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    configured_phase_budget_usd_micros: Annotated[int, Field(ge=1)]
    provider_calls_made_before_authorization: Literal[0] = 0
    authorization_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "authorized_intent_roster_sha256",
        "authorization_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"hosted_exact_once_hash_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_authorization(self) -> HostedExactOnceCostAuthorizationV1:
        keys = [item.request_key for item in self.authorized_calls]
        if keys != sorted(set(keys)):
            raise ValueError("hosted_exact_once_authorized_calls_not_sorted_unique")
        if self.authorized_call_count != len(self.authorized_calls):
            raise ValueError("hosted_exact_once_authorized_call_count_mismatch")
        if self.source_bearing_call_count != sum(
            item.source_bearing for item in self.authorized_calls
        ):
            raise ValueError("hosted_exact_once_source_call_count_mismatch")
        roster = [item.model_dump(mode="json") for item in self.authorized_calls]
        if self.authorized_intent_roster_sha256 != hash_canonical(roster):
            raise ValueError("hosted_exact_once_authorized_roster_hash_mismatch")
        if self.cost_ceiling_usd_micros != sum(
            item.request_cost_ceiling_usd_micros for item in self.authorized_calls
        ):
            raise ValueError("hosted_exact_once_cost_ceiling_sum_mismatch")
        if self.cost_ceiling_usd_micros > self.configured_phase_budget_usd_micros:
            raise ValueError("hosted_exact_once_cost_ceiling_exceeds_budget")
        payload = self.model_dump(mode="json", exclude={"authorization_sha256"})
        if self.authorization_sha256 != hash_canonical(payload):
            raise ValueError("hosted_exact_once_authorization_hash_mismatch")
        return self


def freeze_hosted_exact_once_cost_authorization(
    *,
    execution_bundle_sha256: str,
    phase: CallPhase,
    intents: Sequence[HostedExactOnceIntentV1 | Mapping[str, Any]],
    configured_phase_budget_usd_micros: int,
) -> HostedExactOnceCostAuthorizationV1:
    canonical = [HostedExactOnceIntentV1.model_validate(item) for item in intents]
    if not canonical or any(
        item.execution_bundle_sha256 != execution_bundle_sha256
        or item.phase != phase
        for item in canonical
    ):
        raise HostedExactOnceError("hosted_exact_once_authorization_context_mismatch")
    calls = sorted(
        (
            HostedAuthorizedCallV1(
                request_key=item.request_key,
                intent_sha256=item.intent_sha256,
                request_sha256=item.request_sha256,
                source_bearing=item.source_bearing,
                request_cost_ceiling_usd_micros=(
                    item.request_cost_ceiling_usd_micros
                ),
            )
            for item in canonical
        ),
        key=lambda item: item.request_key,
    )
    payload: dict[str, Any] = {
        "authorization_version": EXACT_ONCE_AUTHORIZATION_VERSION,
        "execution_bundle_sha256": execution_bundle_sha256,
        "phase": phase,
        "authorized_calls": calls,
        "authorized_call_count": len(calls),
        "authorized_intent_roster_sha256": hash_canonical(
            [item.model_dump(mode="json") for item in calls]
        ),
        "source_bearing_call_count": sum(item.source_bearing for item in calls),
        "cost_ceiling_usd_micros": sum(
            item.request_cost_ceiling_usd_micros for item in calls
        ),
        "configured_phase_budget_usd_micros": configured_phase_budget_usd_micros,
        "provider_calls_made_before_authorization": 0,
    }
    return HostedExactOnceCostAuthorizationV1.model_validate(
        {**payload, "authorization_sha256": hash_canonical(payload)}
    )


class HostedExactOnceProviderReceiptV1(ContractModel):
    receipt_version: Literal[
        "hosted-exact-once-provider-receipt-v1"
    ] = EXACT_ONCE_RECEIPT_VERSION
    terminal: Literal[True] = True
    provider_result_returned: Literal[True] = True
    provider_response_observed: bool
    execution_bundle_sha256: str
    phase: CallPhase
    request_key: str
    source_bearing: bool
    context_binding_sha256: str
    attempt_id: str
    intent_sha256: str
    request_sha256: str
    cost_authorization_sha256: str
    provider_result: AnthropicBoundedResultV1
    provider_result_sha256: str
    request_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    receipt_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "context_binding_sha256",
        "attempt_id",
        "intent_sha256",
        "request_sha256",
        "cost_authorization_sha256",
        "provider_result_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"hosted_exact_once_hash_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> HostedExactOnceProviderReceiptV1:
        if (
            self.provider_result.result_sha256 != self.provider_result_sha256
            or self.provider_result.request_sha256 != self.request_sha256
            or self.provider_response_observed
            != (self.provider_result.response_id is not None)
            or _usd_micros(
                self.provider_result.cost.request_cost_ceiling_usd
            )
            != self.request_cost_ceiling_usd_micros
        ):
            raise ValueError("hosted_exact_once_provider_result_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if self.receipt_sha256 != hash_canonical(payload):
            raise ValueError("hosted_exact_once_receipt_hash_mismatch")
        return self


class HostedExactOnceAmbiguityIncidentV1(ContractModel):
    incident_version: Literal[
        "hosted-exact-once-ambiguity-incident-v1"
    ] = EXACT_ONCE_INCIDENT_VERSION
    status: Literal["terminal_ambiguous_attempt_poison"] = (
        "terminal_ambiguous_attempt_poison"
    )
    incident_kind: IncidentKind
    execution_bundle_sha256: str
    phase: CallPhase
    request_key: str
    source_bearing: bool
    context_binding_sha256: str
    attempt_id: str
    intent_sha256: str
    request_sha256: str
    cost_authorization_sha256: str
    response_observation: ResponseObservation
    observed_provider_result_sha256: str | None = None
    possible_provider_attempts: Literal[1] = 1
    retry_this_request_permitted: Literal[False] = False
    incident_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "context_binding_sha256",
        "attempt_id",
        "intent_sha256",
        "request_sha256",
        "cost_authorization_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"hosted_exact_once_hash_invalid:{info.field_name}")
        return value

    @field_validator("observed_provider_result_sha256", "incident_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(f"hosted_exact_once_hash_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_incident(self) -> HostedExactOnceAmbiguityIncidentV1:
        expected_observation: ResponseObservation = {
            "orphan_intent_observed_on_resume": "unknown_after_orphaned_intent",
            "provider_call_raised_after_durable_intent": "not_observed_by_executor",
            "provider_result_invalid_after_return": "observed_but_invalid",
        }[self.incident_kind]
        if self.response_observation != expected_observation:
            raise ValueError("hosted_exact_once_incident_observation_mismatch")
        if (self.observed_provider_result_sha256 is not None) != (
            self.incident_kind == "provider_result_invalid_after_return"
        ):
            raise ValueError("hosted_exact_once_incident_result_hash_shape_invalid")
        payload = self.model_dump(mode="json", exclude={"incident_sha256"})
        if self.incident_sha256 != hash_canonical(payload):
            raise ValueError("hosted_exact_once_incident_hash_mismatch")
        return self


def _freeze_receipt(
    *,
    intent: HostedExactOnceIntentV1,
    authorization: HostedExactOnceCostAuthorizationV1,
    result: AnthropicBoundedResultV1,
) -> HostedExactOnceProviderReceiptV1:
    canonical_result = AnthropicBoundedResultV1.model_validate(result)
    request = AnthropicBoundedRequestV1.model_validate(intent.request)
    expected_aliases = {
        "compiled_schema_sha256": request.compiled_schema_sha256,
        "config_sha256": request.config_sha256,
        "effect_kind": request.effect_kind,
        "full_acceptance_schema_sha256": request.full_acceptance_schema_sha256,
        "identity_sha256": request.identity_sha256,
        "model_prompt_sha256": request.model_prompt_sha256,
        "model_system_sha256": request.model_system_sha256,
        "original_schema_sha256": request.compiled_schema.original_schema_sha256,
        "output_format_present_in_call": request.output_format_present_in_call,
        "request_sha256": request.request_sha256,
        "schema_kind": request.schema_kind,
        "structured_grammar_enforced_by_provider": (
            request.structured_grammar_enforced_by_provider
        ),
        "transport_mode": request.transport_mode,
        "wire_call_sha256": request.expected_wire_call_sha256,
        "wire_schema_sha256": request.compiled_schema.wire_schema_sha256,
    }
    if any(
        getattr(canonical_result, field_name) != expected
        for field_name, expected in expected_aliases.items()
    ):
        raise HostedExactOnceError("hosted_exact_once_provider_result_request_mismatch")
    payload: dict[str, Any] = {
        "receipt_version": EXACT_ONCE_RECEIPT_VERSION,
        "terminal": True,
        "provider_result_returned": True,
        "provider_response_observed": canonical_result.response_id is not None,
        "execution_bundle_sha256": intent.execution_bundle_sha256,
        "phase": intent.phase,
        "request_key": intent.request_key,
        "source_bearing": intent.source_bearing,
        "context_binding_sha256": intent.context_binding_sha256,
        "attempt_id": intent.attempt_id,
        "intent_sha256": intent.intent_sha256,
        "request_sha256": intent.request_sha256,
        "cost_authorization_sha256": authorization.authorization_sha256,
        "provider_result": canonical_result,
        "provider_result_sha256": canonical_result.result_sha256,
        "request_cost_ceiling_usd_micros": (
            intent.request_cost_ceiling_usd_micros
        ),
    }
    return HostedExactOnceProviderReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def _freeze_incident(
    *,
    intent: HostedExactOnceIntentV1,
    authorization: HostedExactOnceCostAuthorizationV1,
    kind: IncidentKind,
    observed_provider_result_sha256: str | None = None,
) -> HostedExactOnceAmbiguityIncidentV1:
    response_observation: ResponseObservation = {
        "orphan_intent_observed_on_resume": "unknown_after_orphaned_intent",
        "provider_call_raised_after_durable_intent": "not_observed_by_executor",
        "provider_result_invalid_after_return": "observed_but_invalid",
    }[kind]
    payload: dict[str, Any] = {
        "incident_version": EXACT_ONCE_INCIDENT_VERSION,
        "status": "terminal_ambiguous_attempt_poison",
        "incident_kind": kind,
        "execution_bundle_sha256": intent.execution_bundle_sha256,
        "phase": intent.phase,
        "request_key": intent.request_key,
        "source_bearing": intent.source_bearing,
        "context_binding_sha256": intent.context_binding_sha256,
        "attempt_id": intent.attempt_id,
        "intent_sha256": intent.intent_sha256,
        "request_sha256": intent.request_sha256,
        "cost_authorization_sha256": authorization.authorization_sha256,
        "response_observation": response_observation,
        "observed_provider_result_sha256": observed_provider_result_sha256,
        "possible_provider_attempts": 1,
        "retry_this_request_permitted": False,
    }
    return HostedExactOnceAmbiguityIncidentV1.model_validate(
        {**payload, "incident_sha256": hash_canonical(payload)}
    )


def _authorized_call(
    *,
    intent: HostedExactOnceIntentV1,
    authorization: HostedExactOnceCostAuthorizationV1,
) -> HostedAuthorizedCallV1:
    if (
        intent.execution_bundle_sha256 != authorization.execution_bundle_sha256
        or intent.phase != authorization.phase
    ):
        raise HostedExactOnceError("hosted_exact_once_execute_context_mismatch")
    matches = [
        item
        for item in authorization.authorized_calls
        if item.request_key == intent.request_key
    ]
    if len(matches) != 1:
        raise HostedExactOnceError("hosted_exact_once_intent_not_authorized")
    expected = HostedAuthorizedCallV1(
        request_key=intent.request_key,
        intent_sha256=intent.intent_sha256,
        request_sha256=intent.request_sha256,
        source_bearing=intent.source_bearing,
        request_cost_ceiling_usd_micros=intent.request_cost_ceiling_usd_micros,
    )
    if matches[0] != expected:
        raise HostedExactOnceError("hosted_exact_once_authorized_intent_mismatch")
    return matches[0]


def _execute_hosted_exactly_once_locked(
    *,
    workspace: Path,
    intent: HostedExactOnceIntentV1 | Mapping[str, Any],
    authorization: HostedExactOnceCostAuthorizationV1 | Mapping[str, Any],
    client: HostedBoundedClientProtocol,
) -> HostedExactOnceProviderReceiptV1 | HostedExactOnceAmbiguityIncidentV1:
    """Execute or replay one authorized request; an orphan is never retried."""

    canonical_intent = HostedExactOnceIntentV1.model_validate(intent)
    canonical_auth = HostedExactOnceCostAuthorizationV1.model_validate(authorization)
    _authorized_call(intent=canonical_intent, authorization=canonical_auth)
    root = _canonical_workspace(workspace, create=True)
    authorization_dir = root / "cost-authorizations"
    intent_dir = root / "call-intents"
    receipt_dir = root / "provider-receipts"
    incident_dir = root / "ambiguity-incidents"
    for directory in (
        authorization_dir,
        intent_dir,
        receipt_dir,
        incident_dir,
    ):
        if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
            raise HostedExactOnceError("hosted_exact_once_state_directory_unsafe")
        directory.mkdir(exist_ok=True)

    # The complete phase roster and its aggregate ceiling must be durable before
    # the first per-call intent creates any possible provider liability.  A phase
    # authorization is immutable; changing even one request requires a fresh
    # workspace (and therefore a separately identified execution).
    authorization_path = authorization_dir / f"{canonical_auth.phase}.json"
    for existing_path in authorization_dir.iterdir():
        if existing_path.is_symlink() or not existing_path.is_file():
            raise HostedExactOnceError("hosted_exact_once_authorization_artifact_unsafe")
        existing = HostedExactOnceCostAuthorizationV1.model_validate(
            _read_object(existing_path)
        )
        if existing.execution_bundle_sha256 != canonical_auth.execution_bundle_sha256:
            raise HostedExactOnceError("hosted_exact_once_workspace_bundle_mismatch")
    if authorization_path.exists():
        saved_authorization = HostedExactOnceCostAuthorizationV1.model_validate(
            _read_object(authorization_path)
        )
        if saved_authorization != canonical_auth:
            raise HostedExactOnceError("hosted_exact_once_authorization_replay_mismatch")
    else:
        for existing_intent_path in intent_dir.iterdir():
            existing_intent = HostedExactOnceIntentV1.model_validate(
                _read_object(existing_intent_path)
            )
            if existing_intent.phase == canonical_auth.phase:
                raise HostedExactOnceError(
                    "hosted_exact_once_intent_precedes_authorization"
                )
        atomic_write_json(authorization_path, canonical_auth)

    intent_path = intent_dir / f"{canonical_intent.request_key}.json"
    receipt_path = receipt_dir / f"{canonical_intent.request_key}.json"
    incident_path = incident_dir / f"{canonical_intent.request_key}.json"
    if receipt_path.exists() and incident_path.exists():
        raise HostedExactOnceError("hosted_exact_once_two_terminal_outcomes")
    if intent_path.exists():
        saved_intent = HostedExactOnceIntentV1.model_validate(_read_object(intent_path))
        if saved_intent != canonical_intent:
            raise HostedExactOnceError("hosted_exact_once_intent_replay_mismatch")
        if receipt_path.exists():
            receipt = HostedExactOnceProviderReceiptV1.model_validate(
                _read_object(receipt_path)
            )
            replayed = _freeze_receipt(
                intent=canonical_intent,
                authorization=canonical_auth,
                result=receipt.provider_result,
            )
            if receipt != replayed:
                raise HostedExactOnceError("hosted_exact_once_receipt_replay_mismatch")
            return receipt
        if incident_path.exists():
            incident = HostedExactOnceAmbiguityIncidentV1.model_validate(
                _read_object(incident_path)
            )
            replayed_incident = _freeze_incident(
                intent=canonical_intent,
                authorization=canonical_auth,
                kind=incident.incident_kind,
                observed_provider_result_sha256=(
                    incident.observed_provider_result_sha256
                ),
            )
            if incident != replayed_incident:
                raise HostedExactOnceError("hosted_exact_once_incident_replay_mismatch")
            return incident
        incident = _freeze_incident(
            intent=canonical_intent,
            authorization=canonical_auth,
            kind="orphan_intent_observed_on_resume",
        )
        atomic_write_json(incident_path, incident)
        return incident
    if receipt_path.exists() or incident_path.exists():
        raise HostedExactOnceError("hosted_exact_once_outcome_without_intent")

    # This durable write is the irreversible authorization boundary.  There is no
    # retry path between it and the single provider invocation.
    atomic_write_json(intent_path, canonical_intent)
    try:
        request = AnthropicBoundedRequestV1.model_validate(canonical_intent.request)
        result = client.generate(request)
    except (KeyboardInterrupt, SystemExit):  # pragma: no cover - crash semantics
        raise
    except Exception:
        incident = _freeze_incident(
            intent=canonical_intent,
            authorization=canonical_auth,
            kind="provider_call_raised_after_durable_intent",
        )
        atomic_write_json(incident_path, incident)
        return incident
    try:
        receipt = _freeze_receipt(
            intent=canonical_intent,
            authorization=canonical_auth,
            result=result,
        )
    except (KeyboardInterrupt, SystemExit):  # pragma: no cover - crash semantics
        raise
    except Exception:
        result_sha256 = getattr(result, "result_sha256", None)
        if not isinstance(result_sha256, str) or not SHA256_RE.fullmatch(result_sha256):
            # An unhashable/non-contract return is still known to have crossed the
            # provider boundary.  Bind that fact without persisting arbitrary data.
            result_sha256 = hash_canonical(
                {
                    "invalid_return_type": (
                        f"{type(result).__module__}.{type(result).__qualname__}"
                    )
                }
            )
        incident = _freeze_incident(
            intent=canonical_intent,
            authorization=canonical_auth,
            kind="provider_result_invalid_after_return",
            observed_provider_result_sha256=result_sha256,
        )
        atomic_write_json(incident_path, incident)
        return incident
    atomic_write_json(receipt_path, receipt)
    return receipt


def execute_hosted_exactly_once(
    *,
    workspace: Path,
    intent: HostedExactOnceIntentV1 | Mapping[str, Any],
    authorization: HostedExactOnceCostAuthorizationV1 | Mapping[str, Any],
    client: HostedBoundedClientProtocol,
) -> HostedExactOnceProviderReceiptV1 | HostedExactOnceAmbiguityIncidentV1:
    """Serialize workspace access and execute/replay one authorized request."""

    canonical_intent = HostedExactOnceIntentV1.model_validate(intent)
    canonical_authorization = HostedExactOnceCostAuthorizationV1.model_validate(
        authorization
    )
    _authorized_call(
        intent=canonical_intent,
        authorization=canonical_authorization,
    )
    root = _canonical_workspace(workspace, create=True)
    with _workspace_lock(root):
        return _execute_hosted_exactly_once_locked(
            workspace=root,
            intent=canonical_intent,
            authorization=canonical_authorization,
            client=client,
        )


def validate_hosted_exact_once_outcome(
    *,
    workspace: Path,
    intent: HostedExactOnceIntentV1 | Mapping[str, Any],
    authorization: HostedExactOnceCostAuthorizationV1 | Mapping[str, Any],
) -> HostedExactOnceProviderReceiptV1 | HostedExactOnceAmbiguityIncidentV1:
    """Externally replay one terminal call without invoking or mutating a provider."""

    canonical_intent = HostedExactOnceIntentV1.model_validate(intent)
    canonical_auth = HostedExactOnceCostAuthorizationV1.model_validate(authorization)
    _authorized_call(intent=canonical_intent, authorization=canonical_auth)
    root = _canonical_workspace(workspace)
    with _workspace_lock(root):
        authorization_path = (
            root / "cost-authorizations" / f"{canonical_auth.phase}.json"
        )
        intent_path = root / "call-intents" / f"{canonical_intent.request_key}.json"
        receipt_path = (
            root / "provider-receipts" / f"{canonical_intent.request_key}.json"
        )
        incident_path = (
            root / "ambiguity-incidents" / f"{canonical_intent.request_key}.json"
        )
        saved_authorization = HostedExactOnceCostAuthorizationV1.model_validate(
            _read_object(authorization_path)
        )
        saved_intent = HostedExactOnceIntentV1.model_validate(_read_object(intent_path))
        if saved_authorization != canonical_auth:
            raise HostedExactOnceError(
                "hosted_exact_once_authorization_external_replay_mismatch"
            )
        if saved_intent != canonical_intent:
            raise HostedExactOnceError(
                "hosted_exact_once_intent_external_replay_mismatch"
            )
        has_receipt = receipt_path.exists()
        has_incident = incident_path.exists()
        if has_receipt == has_incident:
            raise HostedExactOnceError(
                "hosted_exact_once_terminal_outcome_cardinality_invalid"
            )
        if has_receipt:
            receipt = HostedExactOnceProviderReceiptV1.model_validate(
                _read_object(receipt_path)
            )
            replayed_receipt = _freeze_receipt(
                intent=canonical_intent,
                authorization=canonical_auth,
                result=receipt.provider_result,
            )
            if receipt != replayed_receipt:
                raise HostedExactOnceError(
                    "hosted_exact_once_receipt_external_replay_mismatch"
                )
            return receipt
        incident = HostedExactOnceAmbiguityIncidentV1.model_validate(
            _read_object(incident_path)
        )
        replayed_incident = _freeze_incident(
            intent=canonical_intent,
            authorization=canonical_auth,
            kind=incident.incident_kind,
            observed_provider_result_sha256=incident.observed_provider_result_sha256,
        )
        if incident != replayed_incident:
            raise HostedExactOnceError(
                "hosted_exact_once_incident_external_replay_mismatch"
            )
        return incident


__all__ = [
    "CallPhase",
    "HostedAuthorizedCallV1",
    "HostedBoundedClientProtocol",
    "HostedExactOnceAmbiguityIncidentV1",
    "HostedExactOnceCostAuthorizationV1",
    "HostedExactOnceError",
    "HostedExactOnceIntentV1",
    "HostedExactOnceProviderReceiptV1",
    "execute_hosted_exactly_once",
    "freeze_hosted_exact_once_cost_authorization",
    "freeze_hosted_exact_once_intent",
    "validate_hosted_exact_once_outcome",
]
