"""Durable, label-blind paired execution boundary for Evidence Inference Fable.

This module deliberately does not score benchmark answers.  It accepts the exact
model-visible surfaces corresponding to an already-frozen retrospective roster,
admits whole article pairs against a cumulative hard-liability budget, and gives
every provider attempt an intent-before-transport transaction. Orphaned intents
retain the legacy poison behavior. Caught provider exceptions become terminal
failed requests, are never retried, and do not prevent later requests from running.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from jsonschema.validators import validator_for
from pydantic import ConfigDict, Field, StrictInt, model_validator

from literature_multiverse.evidence_inference_fable_retrospective_v1 import (
    DEFAULT_CONFIG_PATH,
    ArticleBatchRequestV1,
    EvidenceInferenceFableRetrospectivePlanV1,
    ExecutionMode,
    freeze_evidence_inference_fable_retrospective_plan_v1,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel

MODEL = "claude-fable-5"
EFFORT = "high"
SERVICE_TIER = "standard_only"
INPUT_RATE = Decimal(10)
OUTPUT_RATE = Decimal(50)
SDK_VERSION = "0.120.2"

INCIDENT_MESSAGE_MAX_CHARS = 512
INCIDENT_EXCEPTION_TYPE_MAX_CHARS = 160
INCIDENT_REQUEST_ID_MAX_CHARS = 128
INCIDENT_SANITIZATION_POLICY = "bounded-ascii-secret-redaction-v1"

_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?:api[-_ ]?key|authorization|x[-_ ]?api[-_ ]?key|"
    r"access[-_ ]?token|refresh[-_ ]?token|token|secret|password|credential)"
    r"(?:[\"']?\s*[:=]\s*)(?:bearer\s+)?"
    r"(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;]+)"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_KNOWN_SECRET_PREFIX_RE = re.compile(
    r"(?i)\b(?:sk-ant-api\d*|sk-proj|sk-live|sk-test|xox[baprs])[-_A-Za-z0-9]+"
)
_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\."
    r"[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
)
_URL_USERINFO_RE = re.compile(r"(?i)\b(https?://)[^/\s:@]+:[^/\s@]+@")
_LONG_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9._~+/-])(?=[A-Za-z0-9._~+/-]{24,}(?![A-Za-z0-9._~+/-]))"
    r"(?=[A-Za-z0-9._~+/-]*[A-Za-z])(?=[A-Za-z0-9._~+/-]*[0-9])"
    r"[A-Za-z0-9._~+/-]+"
)
_SAFE_REQUEST_ID_RE = re.compile(r"(?:req|request)[-_][A-Za-z0-9._:-]{1,119}")
_SAFE_EXCEPTION_TYPE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,159}")

Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]
Micros = Annotated[StrictInt, Field(ge=0)]


class EvidenceInferenceFablePairedRuntimeError(ValueError):
    """The frozen surface or durable state cannot be executed safely."""


class _Frozen(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False)


def _self_hash(model: _Frozen, field: str, code: str) -> None:
    if getattr(model, field) != hash_canonical(model.model_dump(mode="json", exclude={field})):
        raise ValueError(code)


def _micros(value: Decimal) -> int:
    return int((value * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING))


def _bounded_ascii_text(value: object) -> str:
    """Return single-line ASCII without trusting an exception's string conversion."""

    try:
        text = str(value)
    except Exception:
        return "<unprintable>"
    ascii_text = "".join(character if " " <= character <= "~" else " " for character in text)
    return " ".join(ascii_text.split())


def _sanitize_exception_type(exception: BaseException) -> str:
    cls = type(exception)
    try:
        raw = f"{getattr(cls, '__module__', '')}.{getattr(cls, '__qualname__', '')}".strip(".")
    except Exception:
        return "UnknownException"
    sanitized = re.sub(r"[^A-Za-z0-9_.]", "_", raw)[:INCIDENT_EXCEPTION_TYPE_MAX_CHARS]
    if not sanitized or not re.match(r"[A-Za-z_]", sanitized):
        return "UnknownException"
    return sanitized


def _redact_exception_message(value: object) -> tuple[str, bool]:
    raw = _bounded_ascii_text(value)
    redacted = _SENSITIVE_ASSIGNMENT_RE.sub("[REDACTED]", raw)
    redacted = _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", redacted)
    redacted = _KNOWN_SECRET_PREFIX_RE.sub("[REDACTED]", redacted)
    redacted = _JWT_RE.sub("[REDACTED]", redacted)
    redacted = _URL_USERINFO_RE.sub(r"\1[REDACTED]@", redacted)
    redacted = _LONG_TOKEN_RE.sub("[REDACTED]", redacted)
    if not redacted:
        redacted = "<empty>"
    was_truncated = (
        len(raw) > INCIDENT_MESSAGE_MAX_CHARS
        or len(redacted) > INCIDENT_MESSAGE_MAX_CHARS
    )
    return redacted[:INCIDENT_MESSAGE_MAX_CHARS], was_truncated


def _safe_getattr(value: object, name: str) -> object | None:
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def _exception_http_status(exception: BaseException) -> int | None:
    direct = _safe_getattr(exception, "status_code")
    response = _safe_getattr(exception, "response")
    nested = None if response is None else _safe_getattr(response, "status_code")
    for candidate in (direct, nested):
        if (
            isinstance(candidate, int)
            and not isinstance(candidate, bool)
            and 100 <= candidate <= 599
        ):
            return candidate
    return None


def _sanitize_request_id(value: object | None) -> str | None:
    if value is None:
        return None
    candidate = _bounded_ascii_text(value)
    if not candidate:
        return None
    if len(candidate) <= INCIDENT_REQUEST_ID_MAX_CHARS and _SAFE_REQUEST_ID_RE.fullmatch(candidate):
        return candidate
    return "[REDACTED]"


def _exception_request_id(exception: BaseException) -> str | None:
    direct = _safe_getattr(exception, "request_id")
    if direct is not None:
        return _sanitize_request_id(direct)
    response = _safe_getattr(exception, "response")
    headers = None if response is None else _safe_getattr(response, "headers")
    if headers is not None:
        for header in ("request-id", "x-request-id"):
            try:
                candidate = headers.get(header)  # type: ignore[union-attr]
            except Exception:
                candidate = None
            if candidate is not None:
                return _sanitize_request_id(candidate)
    return None


def _exception_diagnostics(exception: BaseException) -> dict[str, Any]:
    message, was_truncated = _redact_exception_message(exception)
    return {
        "sanitization_policy": INCIDENT_SANITIZATION_POLICY,
        "exception_type": _sanitize_exception_type(exception),
        "http_status_code": _exception_http_status(exception),
        "provider_request_id": _exception_request_id(exception),
        "message_redacted": message,
        "message_was_truncated": was_truncated,
    }


def _wire_kwargs(surface: EvidenceInferenceFableCallSurfaceV1) -> dict[str, Any]:
    return {
        "model": MODEL,
        "max_tokens": surface.max_output_tokens,
        "system": surface.system,
        "messages": [{"role": "user", "content": surface.prompt}],
        "output_config": {
            "effort": EFFORT,
            "format": {"type": "json_schema", "schema": surface.wire_schema},
        },
        "service_tier": SERVICE_TIER,
    }


class EvidenceInferenceFableCallSurfaceV1(_Frozen):
    surface_version: Literal["evidence-inference-fable-call-surface-v1"] = (
        "evidence-inference-fable-call-surface-v1"
    )
    request_key: str
    article_request_sha256: Sha256
    system: str
    prompt: str
    wire_schema: dict[str, Any]
    locked_question_count: Annotated[StrictInt, Field(ge=1)]
    request_hard_liability_usd_micros: Annotated[StrictInt, Field(ge=1)]
    max_output_tokens: Annotated[StrictInt, Field(ge=1, le=32000)]
    model: Literal["claude-fable-5"] = MODEL
    effort: Literal["high"] = EFFORT
    service_tier: Literal["standard_only"] = SERVICE_TIER
    sdk_max_retries: Literal[0] = 0
    application_retries: Literal[0] = 0
    wire_call_sha256: Sha256
    surface_sha256: Sha256

    @model_validator(mode="after")
    def validate_surface(self) -> EvidenceInferenceFableCallSurfaceV1:
        if not self.system or not self.prompt:
            raise ValueError("fable_surface_blank")
        validator_for(self.wire_schema).check_schema(self.wire_schema)
        if self.wire_call_sha256 != hash_canonical(_wire_kwargs(self)):
            raise ValueError("fable_wire_call_hash_mismatch")
        _self_hash(self, "surface_sha256", "fable_surface_hash_mismatch")
        return self


def freeze_evidence_inference_fable_call_surface_v1(
    *, roster_item: ArticleBatchRequestV1, system: str, prompt: str, wire_schema: Mapping[str, Any]
) -> EvidenceInferenceFableCallSurfaceV1:
    schema = dict(wire_schema)
    if (
        hashlib.sha256(system.encode()).hexdigest() != roster_item.system_sha256
        or hashlib.sha256(prompt.encode()).hexdigest() != roster_item.prompt_sha256
        or hash_canonical(schema) != roster_item.wire_schema_sha256
    ):
        raise EvidenceInferenceFablePairedRuntimeError("fable_surface_roster_binding_mismatch")
    base = {
        "surface_version": "evidence-inference-fable-call-surface-v1",
        "request_key": roster_item.request_key,
        "article_request_sha256": roster_item.request_sha256,
        "system": system,
        "prompt": prompt,
        "wire_schema": schema,
        "locked_question_count": roster_item.question_count,
        "request_hard_liability_usd_micros": (
            roster_item.cost.full_context_hard_liability_usd_micros
        ),
        "max_output_tokens": roster_item.max_output_tokens,
        "model": MODEL,
        "effort": EFFORT,
        "service_tier": SERVICE_TIER,
        "sdk_max_retries": 0,
        "application_retries": 0,
    }
    provisional = EvidenceInferenceFableCallSurfaceV1.model_construct(
        **base, wire_call_sha256="0" * 64, surface_sha256="0" * 64
    )
    base["wire_call_sha256"] = hash_canonical(_wire_kwargs(provisional))
    return EvidenceInferenceFableCallSurfaceV1.model_validate(
        {**base, "surface_sha256": hash_canonical(base)}
    )


class EvidenceInferenceFablePreparedRuntimeV1(_Frozen):
    prepared_version: Literal["evidence-inference-fable-prepared-runtime-v1"] = (
        "evidence-inference-fable-prepared-runtime-v1"
    )
    status: Literal["offline_prepared_zero_provider_calls"] = "offline_prepared_zero_provider_calls"
    retrospective_plan_sha256: Sha256
    request_roster_sha256: Sha256
    surfaces: list[EvidenceInferenceFableCallSurfaceV1]
    surface_roster_sha256: Sha256
    pair_count: Annotated[StrictInt, Field(ge=1)]
    labels_opened: Literal[False] = False
    provider_calls_made: Literal[0] = 0
    prepared_sha256: Sha256

    @model_validator(mode="after")
    def validate_prepared(self) -> EvidenceInferenceFablePreparedRuntimeV1:
        if len(
            self.surfaces
        ) != self.pair_count * 2 or self.surface_roster_sha256 != hash_canonical(
            [x.surface_sha256 for x in self.surfaces]
        ):
            raise ValueError("fable_prepared_roster_mismatch")
        _self_hash(self, "prepared_sha256", "fable_prepared_hash_mismatch")
        return self


def freeze_evidence_inference_fable_prepared_runtime_v1(
    *,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    surfaces: Sequence[EvidenceInferenceFableCallSurfaceV1],
) -> EvidenceInferenceFablePreparedRuntimeV1:
    canonical = [EvidenceInferenceFableCallSurfaceV1.model_validate(x) for x in surfaces]
    if [x.request_key for x in canonical] != [x.request_key for x in plan.roster] or [
        x.article_request_sha256 for x in canonical
    ] != [x.request_sha256 for x in plan.roster]:
        raise EvidenceInferenceFablePairedRuntimeError("fable_prepared_plan_roster_mismatch")
    for offset in range(0, len(plan.roster), 2):
        left, right = plan.roster[offset : offset + 2]
        if left.article_id != right.article_id or {left.arm, right.arm} != {"seed", "winner"}:
            raise EvidenceInferenceFablePairedRuntimeError("fable_prepared_pair_invalid")
    payload = {
        "prepared_version": "evidence-inference-fable-prepared-runtime-v1",
        "status": "offline_prepared_zero_provider_calls",
        "retrospective_plan_sha256": plan.plan_sha256,
        "request_roster_sha256": plan.request_roster_sha256,
        "surfaces": canonical,
        "surface_roster_sha256": hash_canonical([x.surface_sha256 for x in canonical]),
        "pair_count": len(canonical) // 2,
        "labels_opened": False,
        "provider_calls_made": 0,
    }
    return EvidenceInferenceFablePreparedRuntimeV1.model_validate(
        {**payload, "prepared_sha256": hash_canonical(payload)}
    )


def reconstruct_evidence_inference_fable_prepared_runtime_v1(
    *,
    repository_root: Path,
    mode: ExecutionMode,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> tuple[
    EvidenceInferenceFableRetrospectivePlanV1,
    EvidenceInferenceFablePreparedRuntimeV1,
]:
    """Rebuild the label-safe plan and its exact model surfaces in one pass."""

    raw_surfaces: list[dict[str, Any]] = []
    plan = freeze_evidence_inference_fable_retrospective_plan_v1(
        repository_root=repository_root,
        mode=mode,
        config_path=config_path,
        _model_surface_sink=raw_surfaces,
    )
    surfaces = [
        freeze_evidence_inference_fable_call_surface_v1(
            roster_item=item,
            system=raw["system"],
            prompt=raw["prompt"],
            wire_schema=raw["wire_schema"],
        )
        for item, raw in zip(plan.roster, raw_surfaces, strict=True)
    ]
    return plan, freeze_evidence_inference_fable_prepared_runtime_v1(plan=plan, surfaces=surfaces)


class EvidenceInferenceFableBudgetAuthorizationV1(_Frozen):
    authorization_version: Literal["evidence-inference-fable-budget-authorization-v1"] = (
        "evidence-inference-fable-budget-authorization-v1"
    )
    prepared_sha256: Sha256
    configured_total_budget_usd_micros: Annotated[StrictInt, Field(ge=1)]
    cumulative_gate_basis: Literal["completed_reported_spend_plus_next_pair_hard_liability"] = (
        "completed_reported_spend_plus_next_pair_hard_liability"
    )
    whole_pair_admission_required: Literal[True] = True
    application_retries_permitted: Literal[0] = 0
    sdk_retries_permitted: Literal[0] = 0
    liability_basis: Literal["full_context_fallback", "certified_provider_token_count"] = (
        "full_context_fallback"
    )
    certified_count_terminal_sha256: Sha256 | None = None
    certified_request_liabilities_usd_micros: dict[str, Annotated[StrictInt, Field(ge=1)]] = Field(
        default_factory=dict
    )
    authorization_sha256: Sha256

    @model_validator(mode="after")
    def validate_auth(self) -> EvidenceInferenceFableBudgetAuthorizationV1:
        certified = self.liability_basis == "certified_provider_token_count"
        if certified != (
            self.certified_count_terminal_sha256 is not None
            and bool(self.certified_request_liabilities_usd_micros)
        ):
            raise ValueError("fable_authorization_certified_liability_shape_invalid")
        _self_hash(self, "authorization_sha256", "fable_authorization_hash_mismatch")
        return self


class EvidenceInferenceFableBudgetAuthorizationV2(_Frozen):
    """Count-certified authorization with an explicit fixed input-token margin.

    V1 remains byte-for-byte stable for external replay.  V2 serializes both the
    provider count result and the transformed liability map so the 1,024-token
    safety margin is visible, hash-bound, and independently checkable against the
    prepared hard bounds.
    """

    authorization_version: Literal[
        "evidence-inference-fable-budget-authorization-v2"
    ] = "evidence-inference-fable-budget-authorization-v2"
    prepared_sha256: Sha256
    configured_total_budget_usd_micros: Annotated[StrictInt, Field(ge=1)]
    cumulative_gate_basis: Literal[
        "completed_reported_spend_plus_next_pair_hard_liability"
    ] = "completed_reported_spend_plus_next_pair_hard_liability"
    whole_pair_admission_required: Literal[True] = True
    application_retries_permitted: Literal[0] = 0
    sdk_retries_permitted: Literal[0] = 0
    liability_basis: Literal[
        "certified_provider_token_count_plus_fixed_input_headroom"
    ] = "certified_provider_token_count_plus_fixed_input_headroom"
    certified_count_terminal_sha256: Sha256
    certified_input_token_headroom_per_request: Literal[1024] = 1024
    input_token_price_usd_micros_per_token: Literal[10] = 10
    liability_transform: Literal[
        "min(certified_count_liability+1024*10,full_context_hard_liability)"
    ] = "min(certified_count_liability+1024*10,full_context_hard_liability)"
    certified_base_request_liabilities_usd_micros: dict[
        str, Annotated[StrictInt, Field(ge=1)]
    ]
    certified_request_liabilities_usd_micros: dict[
        str, Annotated[StrictInt, Field(ge=1)]
    ]
    authorization_sha256: Sha256

    @model_validator(mode="after")
    def validate_auth(self) -> EvidenceInferenceFableBudgetAuthorizationV2:
        base = self.certified_base_request_liabilities_usd_micros
        adjusted = self.certified_request_liabilities_usd_micros
        headroom = (
            self.certified_input_token_headroom_per_request
            * self.input_token_price_usd_micros_per_token
        )
        if (
            not base
            or set(base) != set(adjusted)
            or any(
                adjusted[key] < value or adjusted[key] > value + headroom
                for key, value in base.items()
            )
        ):
            raise ValueError("fable_authorization_v2_liability_shape_invalid")
        _self_hash(self, "authorization_sha256", "fable_authorization_hash_mismatch")
        return self


EvidenceInferenceFableBudgetAuthorizationArtifactV1 = (
    EvidenceInferenceFableBudgetAuthorizationV1
    | EvidenceInferenceFableBudgetAuthorizationV2
)


def parse_evidence_inference_fable_budget_authorization_v1(
    payload: Mapping[str, Any],
) -> EvidenceInferenceFableBudgetAuthorizationArtifactV1:
    """Parse either immutable V1 authorization or explicit-headroom V2."""

    version = payload.get("authorization_version")
    if version == "evidence-inference-fable-budget-authorization-v1":
        return EvidenceInferenceFableBudgetAuthorizationV1.model_validate(payload)
    if version == "evidence-inference-fable-budget-authorization-v2":
        return EvidenceInferenceFableBudgetAuthorizationV2.model_validate(payload)
    raise EvidenceInferenceFablePairedRuntimeError(
        "fable_authorization_version_unknown"
    )


def largest_certified_pair_liability_usd_micros_v1(
    *,
    prepared: EvidenceInferenceFablePreparedRuntimeV1,
    certified_request_liabilities_usd_micros: Mapping[str, int],
) -> int:
    """Return the largest indivisible pair liability in prepared-roster order."""

    if not prepared.surfaces or len(prepared.surfaces) % 2:
        raise EvidenceInferenceFablePairedRuntimeError(
            "fable_certified_liability_pair_roster_invalid"
        )
    hard_bounds = {
        surface.request_key: surface.request_hard_liability_usd_micros
        for surface in prepared.surfaces
    }
    if set(certified_request_liabilities_usd_micros) != set(hard_bounds) or any(
        type(value) is not int or not 1 <= value <= hard_bounds[key]
        for key, value in certified_request_liabilities_usd_micros.items()
    ):
        raise EvidenceInferenceFablePairedRuntimeError(
            "fable_certified_liability_roster_invalid"
        )
    return max(
        sum(
            certified_request_liabilities_usd_micros[surface.request_key]
            for surface in prepared.surfaces[index : index + 2]
        )
        for index in range(0, len(prepared.surfaces), 2)
    )


def freeze_evidence_inference_fable_budget_authorization_v1(
    *,
    prepared: EvidenceInferenceFablePreparedRuntimeV1,
    configured_total_budget_usd_micros: int,
    certified_count_terminal: Mapping[str, Any] | None = None,
) -> EvidenceInferenceFableBudgetAuthorizationV1:
    certified_liabilities: dict[str, int] = {}
    certified_terminal_sha = None
    if certified_count_terminal is not None:
        terminal = dict(certified_count_terminal)
        certified_liabilities = dict(terminal.get("certified_request_liabilities_usd_micros", {}))
        terminal_sha = terminal.get("terminal_sha256")
        expected_keys = {surface.request_key for surface in prepared.surfaces}
        hard_bounds = {
            surface.request_key: surface.request_hard_liability_usd_micros
            for surface in prepared.surfaces
        }
        if (
            terminal.get("status") != "completed_certified"
            or terminal.get("prepared_sha256") != prepared.prepared_sha256
            or not isinstance(terminal_sha, str)
            or terminal_sha
            != hash_canonical(
                {
                    key: value
                    for key, value in terminal.items()
                    if key != "terminal_sha256"
                }
            )
            or set(certified_liabilities) != expected_keys
            or any(
                type(value) is not int or not 1 <= value <= hard_bounds[key]
                for key, value in certified_liabilities.items()
            )
        ):
            raise EvidenceInferenceFablePairedRuntimeError(
                "fable_authorization_count_terminal_invalid"
            )
        largest_pair_liability = largest_certified_pair_liability_usd_micros_v1(
            prepared=prepared,
            certified_request_liabilities_usd_micros=certified_liabilities,
        )
        if configured_total_budget_usd_micros < largest_pair_liability:
            raise EvidenceInferenceFablePairedRuntimeError(
                "fable_budget_below_certified_largest_pair_liability"
            )
        certified_terminal_sha = terminal_sha
    payload = {
        "authorization_version": "evidence-inference-fable-budget-authorization-v1",
        "prepared_sha256": prepared.prepared_sha256,
        "configured_total_budget_usd_micros": configured_total_budget_usd_micros,
        "cumulative_gate_basis": "completed_reported_spend_plus_next_pair_hard_liability",
        "whole_pair_admission_required": True,
        "application_retries_permitted": 0,
        "sdk_retries_permitted": 0,
        "liability_basis": (
            "certified_provider_token_count"
            if certified_count_terminal is not None
            else "full_context_fallback"
        ),
        "certified_count_terminal_sha256": certified_terminal_sha,
        "certified_request_liabilities_usd_micros": certified_liabilities,
    }
    return EvidenceInferenceFableBudgetAuthorizationV1.model_validate(
        {**payload, "authorization_sha256": hash_canonical(payload)}
    )


def freeze_evidence_inference_fable_budget_authorization_v2(
    *,
    prepared: EvidenceInferenceFablePreparedRuntimeV1,
    configured_total_budget_usd_micros: int,
    certified_count_terminal: Mapping[str, Any],
) -> EvidenceInferenceFableBudgetAuthorizationV2:
    """Freeze the fresh-run authorization with a 1,024 input-token margin."""

    terminal = dict(certified_count_terminal)
    base_liabilities = dict(
        terminal.get("certified_request_liabilities_usd_micros", {})
    )
    terminal_sha = terminal.get("terminal_sha256")
    expected_keys = {surface.request_key for surface in prepared.surfaces}
    hard_bounds = {
        surface.request_key: surface.request_hard_liability_usd_micros
        for surface in prepared.surfaces
    }
    if (
        terminal.get("status") != "completed_certified"
        or terminal.get("prepared_sha256") != prepared.prepared_sha256
        or not isinstance(terminal_sha, str)
        or terminal_sha
        != hash_canonical(
            {
                key: value
                for key, value in terminal.items()
                if key != "terminal_sha256"
            }
        )
        or set(base_liabilities) != expected_keys
        or any(
            type(value) is not int or not 1 <= value <= hard_bounds[key]
            for key, value in base_liabilities.items()
        )
    ):
        raise EvidenceInferenceFablePairedRuntimeError(
            "fable_authorization_count_terminal_invalid"
        )
    adjusted_liabilities = {
        key: min(value + 1024 * 10, hard_bounds[key])
        for key, value in base_liabilities.items()
    }
    largest_pair_liability = largest_certified_pair_liability_usd_micros_v1(
        prepared=prepared,
        certified_request_liabilities_usd_micros=adjusted_liabilities,
    )
    if configured_total_budget_usd_micros < largest_pair_liability:
        raise EvidenceInferenceFablePairedRuntimeError(
            "fable_budget_below_certified_largest_pair_liability"
        )
    payload = {
        "authorization_version": "evidence-inference-fable-budget-authorization-v2",
        "prepared_sha256": prepared.prepared_sha256,
        "configured_total_budget_usd_micros": configured_total_budget_usd_micros,
        "cumulative_gate_basis": (
            "completed_reported_spend_plus_next_pair_hard_liability"
        ),
        "whole_pair_admission_required": True,
        "application_retries_permitted": 0,
        "sdk_retries_permitted": 0,
        "liability_basis": (
            "certified_provider_token_count_plus_fixed_input_headroom"
        ),
        "certified_count_terminal_sha256": terminal_sha,
        "certified_input_token_headroom_per_request": 1024,
        "input_token_price_usd_micros_per_token": 10,
        "liability_transform": (
            "min(certified_count_liability+1024*10,full_context_hard_liability)"
        ),
        "certified_base_request_liabilities_usd_micros": base_liabilities,
        "certified_request_liabilities_usd_micros": adjusted_liabilities,
    }
    return EvidenceInferenceFableBudgetAuthorizationV2.model_validate(
        {**payload, "authorization_sha256": hash_canonical(payload)}
    )


class EvidenceInferenceFableIntentV1(_Frozen):
    intent_version: Literal["evidence-inference-fable-intent-v1"] = (
        "evidence-inference-fable-intent-v1"
    )
    prepared_sha256: Sha256
    authorization_sha256: Sha256
    pair_index: Annotated[StrictInt, Field(ge=0)]
    request_key: str
    surface: EvidenceInferenceFableCallSurfaceV1
    cumulative_reported_spend_before_pair_usd_micros: Micros
    pair_hard_liability_usd_micros: Annotated[StrictInt, Field(ge=1)]
    permitted_provider_attempts: Literal[1] = 1
    application_retries_permitted: Literal[0] = 0
    sdk_retries_permitted: Literal[0] = 0
    orphan_or_ambiguous_attempt_is_terminal: Literal[True] = True
    intent_sha256: Sha256

    @model_validator(mode="after")
    def validate_intent(self) -> EvidenceInferenceFableIntentV1:
        if self.request_key != self.surface.request_key:
            raise ValueError("fable_intent_request_alias_mismatch")
        _self_hash(self, "intent_sha256", "fable_intent_hash_mismatch")
        return self


class EvidenceInferenceFableProviderResultV1(_Frozen):
    result_version: Literal["evidence-inference-fable-provider-result-v1"] = (
        "evidence-inference-fable-provider-result-v1"
    )
    request_key: str
    surface_sha256: Sha256
    transport_attempt_count: Literal[1] = 1
    sdk_retry_count: Literal[0] = 0
    outcome: Literal["completed", "failed"]
    response_id: str | None
    response_model: str | None
    parsed_json: dict[str, Any] | None
    input_tokens: Annotated[StrictInt, Field(ge=0)] | None
    output_tokens: Annotated[StrictInt, Field(ge=0)] | None
    reported_cost_usd_micros: Micros | None
    charged_cost_usd_micros: Annotated[StrictInt, Field(ge=1)]
    cost_basis: Literal["reported_usage", "unknown_usage_hard_liability"]
    response_text_sha256: Sha256 | None = None
    failure_code: (
        Literal[
            "response_identity_invalid",
            "response_stop_reason_invalid",
            "response_content_invalid",
            "response_json_invalid",
            "response_schema_invalid",
            "response_usage_invalid",
            "provider_call_raised_after_durable_intent",
            "provider_result_invalid_after_return",
        ]
        | None
    ) = None
    result_sha256: Sha256

    @model_validator(mode="after")
    def validate_result(self) -> EvidenceInferenceFableProviderResultV1:
        reported = self.cost_basis == "reported_usage"
        exact_reported_cost = (
            None
            if self.input_tokens is None or self.output_tokens is None
            else self.input_tokens * 10 + self.output_tokens * 50
        )
        if (
            reported
            != all(
                x is not None
                for x in (
                    self.input_tokens,
                    self.output_tokens,
                    self.reported_cost_usd_micros,
                )
            )
            or (reported and self.charged_cost_usd_micros != self.reported_cost_usd_micros)
            or (reported and self.reported_cost_usd_micros != exact_reported_cost)
            or (
                self.outcome == "completed"
                and (
                    not reported
                    or self.response_id is None
                    or self.response_model != MODEL
                    or self.parsed_json is None
                    or self.failure_code is not None
                )
            )
            or (
                self.outcome == "failed"
                and (self.parsed_json is not None or self.failure_code is None)
            )
        ):
            raise ValueError("fable_result_shape_invalid")
        _self_hash(self, "result_sha256", "fable_result_hash_mismatch")
        return self


class EvidenceInferenceFableClientProtocol(Protocol):
    def generate(
        self, surface: EvidenceInferenceFableCallSurfaceV1
    ) -> EvidenceInferenceFableProviderResultV1: ...


class EvidenceInferenceFableReceiptV1(_Frozen):
    receipt_version: Literal["evidence-inference-fable-receipt-v1"] = (
        "evidence-inference-fable-receipt-v1"
    )
    intent_sha256: Sha256
    request_key: str
    provider_result: EvidenceInferenceFableProviderResultV1
    locked_question_count: Annotated[StrictInt, Field(ge=1)]
    locked_questions_scored_incorrect: Annotated[StrictInt, Field(ge=0)]
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> EvidenceInferenceFableReceiptV1:
        if self.request_key != self.provider_result.request_key:
            raise ValueError("fable_receipt_alias_mismatch")
        expected_incorrect = (
            self.locked_question_count if self.provider_result.outcome == "failed" else 0
        )
        if self.locked_questions_scored_incorrect != expected_incorrect:
            raise ValueError("fable_receipt_locked_question_failure_count_mismatch")
        _self_hash(self, "receipt_sha256", "fable_receipt_hash_mismatch")
        return self


class EvidenceInferenceFableIncidentV1(_Frozen):
    incident_version: Literal["evidence-inference-fable-incident-v1"] = (
        "evidence-inference-fable-incident-v1"
    )
    status: Literal["terminal_ambiguous_attempt_poison"] = "terminal_ambiguous_attempt_poison"
    kind: Literal[
        "orphan_intent_observed_on_resume",
        "provider_call_raised_after_durable_intent",
        "provider_result_invalid_after_return",
    ]
    intent_sha256: Sha256
    request_key: str
    charged_cost_usd_micros: Annotated[StrictInt, Field(ge=1)]
    cost_basis: Literal["unknown_usage_hard_liability"] = "unknown_usage_hard_liability"
    retry_permitted: Literal[False] = False
    incident_sha256: Sha256

    @model_validator(mode="after")
    def validate_incident(self) -> EvidenceInferenceFableIncidentV1:
        _self_hash(self, "incident_sha256", "fable_incident_hash_mismatch")
        return self


class EvidenceInferenceFableIncidentV2(_Frozen):
    """Continuing failed attempt with bounded, sanitized diagnostics.

    V1 remains unchanged so already archived ambiguity incidents retain their
    original hashes and continue to replay exactly.
    """

    incident_version: Literal["evidence-inference-fable-incident-v2"] = (
        "evidence-inference-fable-incident-v2"
    )
    status: Literal["failed_request_archived_continue"] = "failed_request_archived_continue"
    kind: Literal[
        "provider_call_raised_after_durable_intent",
        "provider_result_invalid_after_return",
    ]
    intent_sha256: Sha256
    request_key: str
    charged_cost_usd_micros: Annotated[StrictInt, Field(ge=1)]
    cost_basis: Literal["unknown_usage_hard_liability"] = "unknown_usage_hard_liability"
    retry_permitted: Literal[False] = False
    sanitization_policy: Literal["bounded-ascii-secret-redaction-v1"] = (
        INCIDENT_SANITIZATION_POLICY
    )
    exception_type: Annotated[
        str, Field(min_length=1, max_length=INCIDENT_EXCEPTION_TYPE_MAX_CHARS)
    ]
    http_status_code: Annotated[StrictInt, Field(ge=100, le=599)] | None
    provider_request_id: Annotated[
        str, Field(min_length=1, max_length=INCIDENT_REQUEST_ID_MAX_CHARS)
    ] | None
    message_redacted: Annotated[str, Field(min_length=1, max_length=INCIDENT_MESSAGE_MAX_CHARS)]
    message_was_truncated: bool
    derived_provider_result_sha256: Sha256
    incident_sha256: Sha256

    @model_validator(mode="after")
    def validate_incident(self) -> EvidenceInferenceFableIncidentV2:
        sanitized_message, _ = _redact_exception_message(self.message_redacted)
        if (
            not _SAFE_EXCEPTION_TYPE_RE.fullmatch(self.exception_type)
            or sanitized_message != self.message_redacted
            or (
                self.provider_request_id is not None
                and _sanitize_request_id(self.provider_request_id) != self.provider_request_id
            )
        ):
            raise ValueError("fable_incident_diagnostics_unsafe")
        _self_hash(self, "incident_sha256", "fable_incident_hash_mismatch")
        return self


EvidenceInferenceFableIncidentArtifactV1 = (
    EvidenceInferenceFableIncidentV1 | EvidenceInferenceFableIncidentV2
)


def _provider_exception_failed_result(
    *,
    surface: EvidenceInferenceFableCallSurfaceV1,
    charged_cost_usd_micros: int,
    failure_code: Literal[
        "provider_call_raised_after_durable_intent",
        "provider_result_invalid_after_return",
    ] = "provider_call_raised_after_durable_intent",
) -> EvidenceInferenceFableProviderResultV1:
    payload = {
        "result_version": "evidence-inference-fable-provider-result-v1",
        "request_key": surface.request_key,
        "surface_sha256": surface.surface_sha256,
        "transport_attempt_count": 1,
        "sdk_retry_count": 0,
        "outcome": "failed",
        "response_id": None,
        "response_model": None,
        "parsed_json": None,
        "input_tokens": None,
        "output_tokens": None,
        "reported_cost_usd_micros": None,
        "charged_cost_usd_micros": charged_cost_usd_micros,
        "cost_basis": "unknown_usage_hard_liability",
        "response_text_sha256": None,
        "failure_code": failure_code,
    }
    return EvidenceInferenceFableProviderResultV1.model_validate(
        {**payload, "result_sha256": hash_canonical(payload)}
    )


def _failed_receipt_for_provider_exception(
    *,
    intent: EvidenceInferenceFableIntentV1,
    surface: EvidenceInferenceFableCallSurfaceV1,
    result: EvidenceInferenceFableProviderResultV1,
) -> EvidenceInferenceFableReceiptV1:
    payload = {
        "receipt_version": "evidence-inference-fable-receipt-v1",
        "intent_sha256": intent.intent_sha256,
        "request_key": surface.request_key,
        "provider_result": result,
        "locked_question_count": surface.locked_question_count,
        "locked_questions_scored_incorrect": surface.locked_question_count,
    }
    return EvidenceInferenceFableReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


class EvidenceInferenceFableTerminalV1(_Frozen):
    terminal_version: Literal["evidence-inference-fable-terminal-v1"] = (
        "evidence-inference-fable-terminal-v1"
    )
    status: Literal[
        "completed", "clean_budget_exhaustion_before_next_pair", "terminal_ambiguous_attempt_poison"
    ]
    prepared_sha256: Sha256
    authorization_sha256: Sha256
    completed_request_count: Micros
    completed_pair_count: Micros
    cumulative_reported_spend_usd_micros: Micros
    cumulative_spend_semantics: Literal["reported_usage_or_unknown_usage_hard_liability"] = (
        "reported_usage_or_unknown_usage_hard_liability"
    )
    next_pair_index: Micros
    full_population_score_permitted: bool
    extraction_accuracy_authority: Literal[False] = False
    confirmatory_authority: Literal[False] = False
    synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    terminal_sha256: Sha256

    @model_validator(mode="after")
    def validate_terminal(self) -> EvidenceInferenceFableTerminalV1:
        if self.full_population_score_permitted != (self.status == "completed"):
            raise ValueError("fable_terminal_score_gate_mismatch")
        _self_hash(self, "terminal_sha256", "fable_terminal_hash_mismatch")
        return self


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceInferenceFablePairedRuntimeError("fable_runtime_artifact_unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceInferenceFablePairedRuntimeError("fable_runtime_artifact_invalid") from exc
    if not isinstance(value, dict):
        raise EvidenceInferenceFablePairedRuntimeError("fable_runtime_artifact_not_object")
    return value


def _read_incident(path: Path) -> EvidenceInferenceFableIncidentArtifactV1:
    payload = _read(path)
    version = payload.get("incident_version")
    if version == "evidence-inference-fable-incident-v1":
        return EvidenceInferenceFableIncidentV1.model_validate(payload)
    if version == "evidence-inference-fable-incident-v2":
        return EvidenceInferenceFableIncidentV2.model_validate(payload)
    raise EvidenceInferenceFablePairedRuntimeError("fable_incident_version_unknown")


@contextmanager
def _workspace_lock(workspace: Path) -> Any:
    lock_path = workspace / ".paired-runtime.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "a+b", closefd=True) as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
    finally:
        pass


def prepare_evidence_inference_fable_workspace_v1(
    *, workspace: Path, prepared: EvidenceInferenceFablePreparedRuntimeV1
) -> None:
    if workspace.exists() or workspace.is_symlink():
        raise EvidenceInferenceFablePairedRuntimeError("fable_workspace_must_be_fresh")
    workspace.mkdir(parents=True, mode=0o700)
    atomic_write_json(workspace / "00-prepared.json", prepared)


def authorize_evidence_inference_fable_workspace_v1(
    *,
    workspace: Path,
    authorization: EvidenceInferenceFableBudgetAuthorizationArtifactV1,
) -> None:
    prepared = EvidenceInferenceFablePreparedRuntimeV1.model_validate(
        _read(workspace / "00-prepared.json")
    )
    if authorization.prepared_sha256 != prepared.prepared_sha256:
        raise EvidenceInferenceFablePairedRuntimeError("fable_authorization_prepared_mismatch")
    path = workspace / "01-authorization.json"
    if path.exists():
        if parse_evidence_inference_fable_budget_authorization_v1(
            _read(path)
        ) != authorization:
            raise EvidenceInferenceFablePairedRuntimeError("fable_authorization_replay_mismatch")
        return
    atomic_write_json(path, authorization)


def _terminal(
    prepared: EvidenceInferenceFablePreparedRuntimeV1,
    auth: EvidenceInferenceFableBudgetAuthorizationArtifactV1,
    status: str,
    count: int,
    spend: int,
) -> EvidenceInferenceFableTerminalV1:
    payload = {
        "terminal_version": "evidence-inference-fable-terminal-v1",
        "status": status,
        "prepared_sha256": prepared.prepared_sha256,
        "authorization_sha256": auth.authorization_sha256,
        "completed_request_count": count,
        "completed_pair_count": count // 2,
        "cumulative_reported_spend_usd_micros": spend,
        "cumulative_spend_semantics": ("reported_usage_or_unknown_usage_hard_liability"),
        "next_pair_index": count // 2,
        "full_population_score_permitted": status == "completed",
        "extraction_accuracy_authority": False,
        "confirmatory_authority": False,
        "synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return EvidenceInferenceFableTerminalV1.model_validate(
        {**payload, "terminal_sha256": hash_canonical(payload)}
    )


def _authorized_request_liability(
    authorization: EvidenceInferenceFableBudgetAuthorizationArtifactV1,
    roster_item: ArticleBatchRequestV1,
) -> int:
    if authorization.liability_basis != "full_context_fallback":
        try:
            return authorization.certified_request_liabilities_usd_micros[roster_item.request_key]
        except KeyError as exc:
            raise EvidenceInferenceFablePairedRuntimeError(
                "fable_certified_request_liability_missing"
            ) from exc
    return roster_item.cost.full_context_hard_liability_usd_micros


def _pair_liability_usd_micros(
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    authorization: EvidenceInferenceFableBudgetAuthorizationArtifactV1,
    index: int,
) -> int:
    pair_start = index - (index % 2)
    pair = plan.roster[pair_start : pair_start + 2]
    if len(pair) != 2:
        raise EvidenceInferenceFablePairedRuntimeError("fable_runtime_pair_incomplete")
    return sum(_authorized_request_liability(authorization, item) for item in pair)


def _validate_archived_intent_v1(
    *,
    intent: EvidenceInferenceFableIntentV1,
    prepared: EvidenceInferenceFablePreparedRuntimeV1,
    authorization: EvidenceInferenceFableBudgetAuthorizationArtifactV1,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    surface: EvidenceInferenceFableCallSurfaceV1,
    index: int,
    spend_before_request: int,
) -> None:
    if (
        intent.prepared_sha256 != prepared.prepared_sha256
        or intent.authorization_sha256 != authorization.authorization_sha256
        or intent.pair_index != index // 2
        or intent.request_key != surface.request_key
        or intent.surface != surface
        or intent.cumulative_reported_spend_before_pair_usd_micros != spend_before_request
        or intent.pair_hard_liability_usd_micros
        != _pair_liability_usd_micros(plan, authorization, index)
    ):
        raise EvidenceInferenceFablePairedRuntimeError(
            "fable_archived_intent_external_replay_mismatch"
        )


def _validate_archived_receipt_v1(
    *,
    receipt: EvidenceInferenceFableReceiptV1,
    intent: EvidenceInferenceFableIntentV1,
    authorization: EvidenceInferenceFableBudgetAuthorizationArtifactV1,
    roster_item: ArticleBatchRequestV1,
    surface: EvidenceInferenceFableCallSurfaceV1,
) -> None:
    result = receipt.provider_result
    if (
        receipt.intent_sha256 != intent.intent_sha256
        or receipt.request_key != surface.request_key
        or result.request_key != surface.request_key
        or result.surface_sha256 != surface.surface_sha256
        or receipt.locked_question_count != surface.locked_question_count
        or receipt.locked_question_count != roster_item.question_count
        or (result.output_tokens or 0) > surface.max_output_tokens
        or result.charged_cost_usd_micros
        > _authorized_request_liability(authorization, roster_item)
    ):
        raise EvidenceInferenceFablePairedRuntimeError(
            "fable_archived_receipt_external_replay_mismatch"
        )


def _validate_archived_incident_v1(
    *,
    incident: EvidenceInferenceFableIncidentArtifactV1,
    intent: EvidenceInferenceFableIntentV1,
    authorization: EvidenceInferenceFableBudgetAuthorizationArtifactV1,
    roster_item: ArticleBatchRequestV1,
    surface: EvidenceInferenceFableCallSurfaceV1,
) -> None:
    result_hash_mismatch = False
    if isinstance(incident, EvidenceInferenceFableIncidentV2):
        expected_failed_result = _provider_exception_failed_result(
            surface=surface,
            charged_cost_usd_micros=incident.charged_cost_usd_micros,
            failure_code=incident.kind,
        )
        result_hash_mismatch = (
            incident.derived_provider_result_sha256 != expected_failed_result.result_sha256
        )
    if (
        incident.intent_sha256 != intent.intent_sha256
        or incident.request_key != surface.request_key
        or incident.charged_cost_usd_micros
        != _authorized_request_liability(authorization, roster_item)
        or result_hash_mismatch
    ):
        raise EvidenceInferenceFablePairedRuntimeError(
            "fable_archived_incident_external_replay_mismatch"
        )


def _load_or_materialize_provider_exception_receipt(
    *,
    receipt_path: Path,
    incident: EvidenceInferenceFableIncidentV2,
    intent: EvidenceInferenceFableIntentV1,
    authorization: EvidenceInferenceFableBudgetAuthorizationArtifactV1,
    roster_item: ArticleBatchRequestV1,
    surface: EvidenceInferenceFableCallSurfaceV1,
    materialize_if_missing: bool,
) -> EvidenceInferenceFableReceiptV1:
    result = _provider_exception_failed_result(
        surface=surface,
        charged_cost_usd_micros=incident.charged_cost_usd_micros,
        failure_code=incident.kind,
    )
    expected = _failed_receipt_for_provider_exception(
        intent=intent,
        surface=surface,
        result=result,
    )
    if receipt_path.exists():
        observed = EvidenceInferenceFableReceiptV1.model_validate(_read(receipt_path))
        _validate_archived_receipt_v1(
            receipt=observed,
            intent=intent,
            authorization=authorization,
            roster_item=roster_item,
            surface=surface,
        )
        if observed != expected:
            raise EvidenceInferenceFablePairedRuntimeError(
                "fable_provider_exception_receipt_replay_mismatch"
            )
        return observed
    if not materialize_if_missing:
        raise EvidenceInferenceFablePairedRuntimeError(
            "fable_provider_exception_receipt_missing"
        )
    atomic_write_json(receipt_path, expected)
    return expected


def _ensure_runtime_artifact_directory(path: Path) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise EvidenceInferenceFablePairedRuntimeError(
                "fable_runtime_artifact_directory_unsafe"
            )
        return
    path.mkdir()


def _validate_authorization_prepared_binding_v1(
    *,
    prepared: EvidenceInferenceFablePreparedRuntimeV1,
    authorization: EvidenceInferenceFableBudgetAuthorizationArtifactV1,
) -> None:
    if authorization.prepared_sha256 != prepared.prepared_sha256:
        raise EvidenceInferenceFablePairedRuntimeError(
            "fable_authorization_prepared_mismatch"
        )
    if authorization.liability_basis != "full_context_fallback":
        expected = {
            surface.request_key: surface.request_hard_liability_usd_micros
            for surface in prepared.surfaces
        }
        if set(authorization.certified_request_liabilities_usd_micros) != set(expected) or any(
            value > expected[key]
            for key, value in authorization.certified_request_liabilities_usd_micros.items()
        ):
            raise EvidenceInferenceFablePairedRuntimeError(
                "fable_authorization_certified_liability_binding_mismatch"
            )
        if isinstance(
            authorization, EvidenceInferenceFableBudgetAuthorizationV2
        ) and (
            set(
                authorization.certified_base_request_liabilities_usd_micros
            ) != set(expected) or any(
                authorization.certified_request_liabilities_usd_micros[key]
                != min(
                    value
                    + authorization.certified_input_token_headroom_per_request
                    * authorization.input_token_price_usd_micros_per_token,
                    expected[key],
                )
                for key, value in (
                    authorization.certified_base_request_liabilities_usd_micros.items()
                )
            )
        ):
            raise EvidenceInferenceFablePairedRuntimeError(
                "fable_authorization_v2_headroom_binding_mismatch"
            )


def _execute_evidence_inference_fable_paired_locked_v1(
    *,
    workspace: Path,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    client: EvidenceInferenceFableClientProtocol,
) -> EvidenceInferenceFableTerminalV1:
    prepared = EvidenceInferenceFablePreparedRuntimeV1.model_validate(
        _read(workspace / "00-prepared.json")
    )
    auth = parse_evidence_inference_fable_budget_authorization_v1(
        _read(workspace / "01-authorization.json")
    )
    if prepared.retrospective_plan_sha256 != plan.plan_sha256:
        raise EvidenceInferenceFablePairedRuntimeError("fable_execution_binding_mismatch")
    if freeze_evidence_inference_fable_prepared_runtime_v1(
        plan=plan, surfaces=prepared.surfaces
    ) != prepared:
        raise EvidenceInferenceFablePairedRuntimeError("fable_execution_prepared_replay_mismatch")
    _validate_authorization_prepared_binding_v1(
        prepared=prepared, authorization=auth
    )
    terminal_path = workspace / "02-terminal.json"
    if terminal_path.exists():
        return _validate_evidence_inference_fable_workspace_locked_v1(
            workspace=workspace, plan=plan
        )
    intents = workspace / "intents"
    receipts = workspace / "receipts"
    incidents = workspace / "incidents"
    for path in (intents, receipts, incidents):
        _ensure_runtime_artifact_directory(path)
    count = 0
    spend = 0
    for index, surface in enumerate(prepared.surfaces):
        key = surface.request_key
        ip, rp, xp = intents / f"{key}.json", receipts / f"{key}.json", incidents / f"{key}.json"
        if xp.exists():
            intent = EvidenceInferenceFableIntentV1.model_validate(_read(ip))
            _validate_archived_intent_v1(
                intent=intent,
                prepared=prepared,
                authorization=auth,
                plan=plan,
                surface=surface,
                index=index,
                spend_before_request=spend,
            )
            incident = _read_incident(xp)
            _validate_archived_incident_v1(
                incident=incident,
                intent=intent,
                authorization=auth,
                roster_item=plan.roster[index],
                surface=surface,
            )
            if isinstance(incident, EvidenceInferenceFableIncidentV2):
                receipt = _load_or_materialize_provider_exception_receipt(
                    receipt_path=rp,
                    incident=incident,
                    intent=intent,
                    authorization=auth,
                    roster_item=plan.roster[index],
                    surface=surface,
                    materialize_if_missing=True,
                )
                count += 1
                spend += receipt.provider_result.charged_cost_usd_micros
                continue
            spend += incident.charged_cost_usd_micros
            term = _terminal(prepared, auth, "terminal_ambiguous_attempt_poison", count, spend)
            atomic_write_json(terminal_path, term)
            return term
        if rp.exists():
            intent = EvidenceInferenceFableIntentV1.model_validate(_read(ip))
            _validate_archived_intent_v1(
                intent=intent,
                prepared=prepared,
                authorization=auth,
                plan=plan,
                surface=surface,
                index=index,
                spend_before_request=spend,
            )
            receipt = EvidenceInferenceFableReceiptV1.model_validate(_read(rp))
            _validate_archived_receipt_v1(
                receipt=receipt,
                intent=intent,
                authorization=auth,
                roster_item=plan.roster[index],
                surface=surface,
            )
            if receipt.provider_result.failure_code in {
                "provider_call_raised_after_durable_intent",
                "provider_result_invalid_after_return",
            }:
                raise EvidenceInferenceFablePairedRuntimeError(
                    "fable_provider_exception_incident_missing"
                )
            count += 1
            spend += receipt.provider_result.charged_cost_usd_micros
            continue
        if ip.exists():
            intent = EvidenceInferenceFableIntentV1.model_validate(_read(ip))
            _validate_archived_intent_v1(
                intent=intent,
                prepared=prepared,
                authorization=auth,
                plan=plan,
                surface=surface,
                index=index,
                spend_before_request=spend,
            )
            incident_charge = _authorized_request_liability(auth, plan.roster[index])
            incident_base = {
                "incident_version": "evidence-inference-fable-incident-v1",
                "status": "terminal_ambiguous_attempt_poison",
                "kind": "orphan_intent_observed_on_resume",
                "intent_sha256": intent.intent_sha256,
                "request_key": key,
                "charged_cost_usd_micros": incident_charge,
                "cost_basis": "unknown_usage_hard_liability",
                "retry_permitted": False,
            }
            incident = EvidenceInferenceFableIncidentV1.model_validate(
                {**incident_base, "incident_sha256": hash_canonical(incident_base)}
            )
            atomic_write_json(xp, incident)
            spend += incident_charge
            term = _terminal(prepared, auth, "terminal_ambiguous_attempt_poison", count, spend)
            atomic_write_json(terminal_path, term)
            return term
        if index % 2 == 0:
            liability = _pair_liability_usd_micros(plan, auth, index)
            if spend + liability > auth.configured_total_budget_usd_micros:
                term = _terminal(
                    prepared, auth, "clean_budget_exhaustion_before_next_pair", count, spend
                )
                atomic_write_json(terminal_path, term)
                return term
        else:
            liability = _pair_liability_usd_micros(plan, auth, index)
        intent_base = {
            "intent_version": "evidence-inference-fable-intent-v1",
            "prepared_sha256": prepared.prepared_sha256,
            "authorization_sha256": auth.authorization_sha256,
            "pair_index": index // 2,
            "request_key": key,
            "surface": surface,
            "cumulative_reported_spend_before_pair_usd_micros": spend,
            "pair_hard_liability_usd_micros": liability,
            "permitted_provider_attempts": 1,
            "application_retries_permitted": 0,
            "sdk_retries_permitted": 0,
            "orphan_or_ambiguous_attempt_is_terminal": True,
        }
        intent = EvidenceInferenceFableIntentV1.model_validate(
            {**intent_base, "intent_sha256": hash_canonical(intent_base)}
        )
        atomic_write_json(ip, intent)
        incident_diagnostics: dict[str, Any] | None = None
        try:
            observed = client.generate(surface)
        except Exception as exception:
            incident_kind = "provider_call_raised_after_durable_intent"
            incident_diagnostics = _exception_diagnostics(exception)
        else:
            try:
                result = EvidenceInferenceFableProviderResultV1.model_validate(observed)
                roster_liability = _authorized_request_liability(auth, plan.roster[index])
                if (
                    result.request_key != key
                    or result.surface_sha256 != surface.surface_sha256
                    or (result.output_tokens or 0) > surface.max_output_tokens
                    or result.charged_cost_usd_micros > roster_liability
                ):
                    raise ValueError("provider_result_binding")
            except Exception:
                incident_kind = "provider_result_invalid_after_return"
                incident_diagnostics = {
                    "sanitization_policy": INCIDENT_SANITIZATION_POLICY,
                    "exception_type": "InvalidProviderResult",
                    "http_status_code": None,
                    "provider_request_id": None,
                    "message_redacted": (
                        "Provider result failed local contract validation; "
                        "response contents discarded."
                    ),
                    "message_was_truncated": False,
                }
            else:
                incident_kind = None
        if incident_kind is not None:
            incident_charge = _authorized_request_liability(auth, plan.roster[index])
            if incident_diagnostics is not None:
                result = _provider_exception_failed_result(
                    surface=surface,
                    charged_cost_usd_micros=incident_charge,
                    failure_code=incident_kind,
                )
                incident_base = {
                    "incident_version": "evidence-inference-fable-incident-v2",
                    "status": "failed_request_archived_continue",
                    "kind": incident_kind,
                    "intent_sha256": intent.intent_sha256,
                    "request_key": key,
                    "charged_cost_usd_micros": incident_charge,
                    "cost_basis": "unknown_usage_hard_liability",
                    "retry_permitted": False,
                    **incident_diagnostics,
                    "derived_provider_result_sha256": result.result_sha256,
                }
                incident = EvidenceInferenceFableIncidentV2.model_validate(
                    {**incident_base, "incident_sha256": hash_canonical(incident_base)}
                )
                # Incident-first durability makes a crash before the derived
                # receipt recoverable without another provider attempt.
                atomic_write_json(xp, incident)
                receipt = _failed_receipt_for_provider_exception(
                    intent=intent,
                    surface=surface,
                    result=result,
                )
                atomic_write_json(rp, receipt)
                count += 1
                spend += result.charged_cost_usd_micros
                continue
            else:
                incident_base = {
                    "incident_version": "evidence-inference-fable-incident-v1",
                    "status": "terminal_ambiguous_attempt_poison",
                    "kind": incident_kind,
                    "intent_sha256": intent.intent_sha256,
                    "request_key": key,
                    "charged_cost_usd_micros": incident_charge,
                    "cost_basis": "unknown_usage_hard_liability",
                    "retry_permitted": False,
                }
                incident = EvidenceInferenceFableIncidentV1.model_validate(
                    {**incident_base, "incident_sha256": hash_canonical(incident_base)}
                )
            atomic_write_json(xp, incident)
            spend += incident_charge
            term = _terminal(prepared, auth, "terminal_ambiguous_attempt_poison", count, spend)
            atomic_write_json(terminal_path, term)
            return term
        receipt_base = {
            "receipt_version": "evidence-inference-fable-receipt-v1",
            "intent_sha256": intent.intent_sha256,
            "request_key": key,
            "provider_result": result,
            "locked_question_count": surface.locked_question_count,
            "locked_questions_scored_incorrect": (
                surface.locked_question_count if result.outcome == "failed" else 0
            ),
        }
        receipt = EvidenceInferenceFableReceiptV1.model_validate(
            {**receipt_base, "receipt_sha256": hash_canonical(receipt_base)}
        )
        atomic_write_json(rp, receipt)
        count += 1
        spend += result.charged_cost_usd_micros
    term = _terminal(prepared, auth, "completed", count, spend)
    atomic_write_json(terminal_path, term)
    return term


def execute_evidence_inference_fable_paired_v1(
    *,
    workspace: Path,
    plan: EvidenceInferenceFableRetrospectivePlanV1,
    client: EvidenceInferenceFableClientProtocol,
) -> EvidenceInferenceFableTerminalV1:
    with _workspace_lock(workspace):
        return _execute_evidence_inference_fable_paired_locked_v1(
            workspace=workspace, plan=plan, client=client
        )


def _validate_evidence_inference_fable_workspace_locked_v1(
    *, workspace: Path, plan: EvidenceInferenceFableRetrospectivePlanV1
) -> EvidenceInferenceFableTerminalV1:
    prepared = EvidenceInferenceFablePreparedRuntimeV1.model_validate(
        _read(workspace / "00-prepared.json")
    )
    auth = parse_evidence_inference_fable_budget_authorization_v1(
        _read(workspace / "01-authorization.json")
    )
    terminal = EvidenceInferenceFableTerminalV1.model_validate(
        _read(workspace / "02-terminal.json")
    )
    if (
        prepared.retrospective_plan_sha256 != plan.plan_sha256
        or terminal.prepared_sha256 != prepared.prepared_sha256
        or terminal.authorization_sha256 != auth.authorization_sha256
    ):
        raise EvidenceInferenceFablePairedRuntimeError("fable_external_replay_binding_mismatch")
    if freeze_evidence_inference_fable_prepared_runtime_v1(
        plan=plan, surfaces=prepared.surfaces
    ) != prepared:
        raise EvidenceInferenceFablePairedRuntimeError(
            "fable_external_replay_prepared_mismatch"
        )
    _validate_authorization_prepared_binding_v1(
        prepared=prepared, authorization=auth
    )
    planned_keys = {surface.request_key for surface in prepared.surfaces}
    artifact_keys: dict[str, set[str]] = {}
    for directory_name in ("intents", "receipts", "incidents"):
        directory = workspace / directory_name
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise EvidenceInferenceFablePairedRuntimeError(
                "fable_external_replay_artifact_directory_unsafe"
            )
        observed_names = set()
        for artifact in directory.iterdir():
            if artifact.is_symlink() or not artifact.is_file() or artifact.suffix != ".json":
                raise EvidenceInferenceFablePairedRuntimeError(
                    "fable_external_replay_extra_artifact"
                )
            observed_names.add(artifact.stem)
        if not observed_names.issubset(planned_keys):
            raise EvidenceInferenceFablePairedRuntimeError(
                "fable_external_replay_unknown_request_artifact"
            )
        artifact_keys[directory_name] = observed_names
    count = 0
    spend = 0
    continuing_incident_keys: set[str] = set()
    terminal_incident_key: str | None = None
    for index, surface in enumerate(prepared.surfaces):
        roster_item = plan.roster[index]
        ip = workspace / "intents" / f"{surface.request_key}.json"
        rp = workspace / "receipts" / f"{surface.request_key}.json"
        xp = workspace / "incidents" / f"{surface.request_key}.json"
        if xp.exists():
            intent = EvidenceInferenceFableIntentV1.model_validate(_read(ip))
            _validate_archived_intent_v1(
                intent=intent,
                prepared=prepared,
                authorization=auth,
                plan=plan,
                surface=surface,
                index=index,
                spend_before_request=spend,
            )
            incident = _read_incident(xp)
            _validate_archived_incident_v1(
                incident=incident,
                intent=intent,
                authorization=auth,
                roster_item=roster_item,
                surface=surface,
            )
            if isinstance(incident, EvidenceInferenceFableIncidentV2):
                receipt = _load_or_materialize_provider_exception_receipt(
                    receipt_path=rp,
                    incident=incident,
                    intent=intent,
                    authorization=auth,
                    roster_item=roster_item,
                    surface=surface,
                    materialize_if_missing=False,
                )
                continuing_incident_keys.add(surface.request_key)
                count += 1
                spend += receipt.provider_result.charged_cost_usd_micros
                continue
            if rp.exists():
                raise EvidenceInferenceFablePairedRuntimeError(
                    "fable_poison_incident_receipt_conflict"
                )
            terminal_incident_key = surface.request_key
            spend += incident.charged_cost_usd_micros
            break
        if rp.exists():
            receipt = EvidenceInferenceFableReceiptV1.model_validate(_read(rp))
            intent = EvidenceInferenceFableIntentV1.model_validate(_read(ip))
            _validate_archived_intent_v1(
                intent=intent,
                prepared=prepared,
                authorization=auth,
                plan=plan,
                surface=surface,
                index=index,
                spend_before_request=spend,
            )
            _validate_archived_receipt_v1(
                receipt=receipt,
                intent=intent,
                authorization=auth,
                roster_item=roster_item,
                surface=surface,
            )
            if receipt.provider_result.failure_code in {
                "provider_call_raised_after_durable_intent",
                "provider_result_invalid_after_return",
            }:
                raise EvidenceInferenceFablePairedRuntimeError(
                    "fable_provider_exception_incident_missing"
                )
            count += 1
            spend += receipt.provider_result.charged_cost_usd_micros
        else:
            break
    expected = _terminal(prepared, auth, terminal.status, count, spend)
    if expected != terminal:
        raise EvidenceInferenceFablePairedRuntimeError("fable_external_replay_terminal_mismatch")
    if terminal.status == "completed" and count != len(prepared.surfaces):
        raise EvidenceInferenceFablePairedRuntimeError("fable_completed_terminal_incomplete")
    if terminal.status == "clean_budget_exhaustion_before_next_pair":
        if count % 2 or count >= len(prepared.surfaces):
            raise EvidenceInferenceFablePairedRuntimeError("fable_budget_terminal_position_invalid")
        next_liability = sum(
            _authorized_request_liability(auth, item) for item in plan.roster[count : count + 2]
        )
        if spend + next_liability <= auth.configured_total_budget_usd_micros:
            raise EvidenceInferenceFablePairedRuntimeError(
                "fable_budget_terminal_gate_not_exhausted"
            )
    ordered_keys = [surface.request_key for surface in prepared.surfaces]
    expected_receipts = set(ordered_keys[:count])
    expected_incidents = set(continuing_incident_keys)
    if terminal.status == "terminal_ambiguous_attempt_poison":
        if terminal_incident_key is None or terminal_incident_key != ordered_keys[count]:
            raise EvidenceInferenceFablePairedRuntimeError(
                "fable_poison_terminal_incident_position_mismatch"
            )
        expected_incidents.add(terminal_incident_key)
    elif terminal_incident_key is not None:
        raise EvidenceInferenceFablePairedRuntimeError(
            "fable_nonpoison_terminal_has_poison_incident"
        )
    if (
        artifact_keys.get("receipts", set()) != expected_receipts
        or artifact_keys.get("incidents", set()) != expected_incidents
        or artifact_keys.get("intents", set()) != expected_receipts | expected_incidents
    ):
        raise EvidenceInferenceFablePairedRuntimeError(
            "fable_external_replay_artifact_roster_mismatch"
        )
    return terminal


def validate_evidence_inference_fable_workspace_v1(
    *, workspace: Path, plan: EvidenceInferenceFableRetrospectivePlanV1
) -> EvidenceInferenceFableTerminalV1:
    with _workspace_lock(workspace):
        return _validate_evidence_inference_fable_workspace_locked_v1(
            workspace=workspace, plan=plan
        )


class AnthropicFablePairedClientV1:
    """Optional live adapter; construction is inert and every call has zero SDK retries."""

    def __init__(self, client: Any) -> None:
        self.client = client

    @classmethod
    def from_anthropic_sdk(cls) -> AnthropicFablePairedClientV1:
        """Construct the live SDK boundary with retries and env transport disabled."""

        import anthropic  # type: ignore[import-not-found]

        if str(getattr(anthropic, "__version__", "")) != SDK_VERSION:
            raise EvidenceInferenceFablePairedRuntimeError("fable_sdk_version_mismatch")
        http_client = anthropic.DefaultHttpxClient(
            timeout=600.0, trust_env=False, follow_redirects=False
        )
        return cls(
            anthropic.Anthropic(
                base_url="https://api.anthropic.com",
                default_headers={"anthropic-version": "2023-06-01"},
                http_client=http_client,
                max_retries=0,
                timeout=600.0,
            )
        )

    def generate(
        self, surface: EvidenceInferenceFableCallSurfaceV1
    ) -> EvidenceInferenceFableProviderResultV1:
        response = self.client.messages.create(**_wire_kwargs(surface))
        try:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            if (
                type(input_tokens) is not int
                or type(output_tokens) is not int
                or input_tokens < 0
                or output_tokens < 0
                or output_tokens > surface.max_output_tokens
            ):
                raise ValueError("usage_invalid")
            reported_cost = _micros(
                (Decimal(input_tokens) * INPUT_RATE + Decimal(output_tokens) * OUTPUT_RATE)
                / Decimal(1_000_000)
            )
            charged_cost = reported_cost
            cost_basis = "reported_usage"
            failure_code = None
        except Exception:
            input_tokens = None
            output_tokens = None
            reported_cost = None
            charged_cost = surface.request_hard_liability_usd_micros
            cost_basis = "unknown_usage_hard_liability"
            failure_code = "response_usage_invalid"
        parsed = None
        response_text_sha256 = None
        try:
            text_blocks = [block.text for block in response.content if block.type == "text"]
            if len(text_blocks) == 1 and isinstance(text_blocks[0], str):
                response_text_sha256 = hashlib.sha256(text_blocks[0].encode()).hexdigest()
            if failure_code is not None:
                pass
            elif (
                not isinstance(getattr(response, "id", None), str)
                or not response.id
                or getattr(response, "model", None) != MODEL
            ):
                failure_code = "response_identity_invalid"
            elif getattr(response, "stop_reason", None) != "end_turn":
                failure_code = "response_stop_reason_invalid"
            elif len(text_blocks) != 1:
                failure_code = "response_content_invalid"
            else:
                try:
                    parsed = json.loads(text_blocks[0])
                except (TypeError, ValueError):
                    failure_code = "response_json_invalid"
                else:
                    try:
                        validator_for(surface.wire_schema)(surface.wire_schema).validate(parsed)
                    except Exception:
                        failure_code = "response_schema_invalid"
                        parsed = None
        except Exception:
            failure_code = "response_content_invalid"
        payload = {
            "result_version": "evidence-inference-fable-provider-result-v1",
            "request_key": surface.request_key,
            "surface_sha256": surface.surface_sha256,
            "transport_attempt_count": 1,
            "sdk_retry_count": 0,
            "outcome": "completed" if failure_code is None else "failed",
            "response_id": getattr(response, "id", None),
            "response_model": getattr(response, "model", None),
            "parsed_json": parsed,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reported_cost_usd_micros": reported_cost,
            "charged_cost_usd_micros": charged_cost,
            "cost_basis": cost_basis,
            "response_text_sha256": response_text_sha256,
            "failure_code": failure_code,
        }
        return EvidenceInferenceFableProviderResultV1.model_validate(
            {**payload, "result_sha256": hash_canonical(payload)}
        )


__all__ = [
    name
    for name in globals()
    if name.startswith("EvidenceInferenceFable")
    or name.startswith("freeze_evidence")
    or name.startswith("prepare_evidence")
    or name.startswith("authorize_evidence")
    or name.startswith("execute_evidence")
    or name.startswith("validate_evidence")
    or name == "AnthropicFablePairedClientV1"
]
