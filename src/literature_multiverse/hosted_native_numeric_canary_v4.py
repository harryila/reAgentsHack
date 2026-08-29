"""Source-free, at-most-once prompt-JSON transport canary for numeric pilot v4.

The canary is deliberately isolated from the scientific pilot.  It freezes a tiny
synthetic JSON fixture, authorizes its complete worst-case liability, writes an intent
before transport, and permits at most one provider attempt.  An orphaned intent is
terminally poisoned and never retried.  No source document, benchmark label, or
credential is part of any frozen artifact.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from jsonschema.validators import validator_for
from pydantic import ConfigDict, Field, StrictInt, TypeAdapter, field_validator, model_validator

from literature_multiverse.anthropic_bounded_generation import (
    AnthropicCompiledSchemaV1,
    compile_anthropic_bounded_schema,
    render_anthropic_prompt_json_model_system,
)
from literature_multiverse.lineage import canonical_json_bytes, hash_canonical, sha256_file
from literature_multiverse.models import SHA256_RE, ContractModel

PLAN_VERSION = "hosted-native-numeric-source-free-canary-plan-v4"
AUTHORIZATION_VERSION = "hosted-native-numeric-source-free-canary-authorization-v4"
INTENT_VERSION = "hosted-native-numeric-source-free-canary-intent-v4"
SUCCESS_VERSION = "hosted-native-numeric-source-free-canary-success-v4"
FAILURE_VERSION = "hosted-native-numeric-source-free-canary-failure-v4"
BINDING_VERSION = "hosted-native-numeric-source-free-canary-binding-v4"

MODEL = "claude-fable-5"
MODEL_REVISION = "claude-fable-5"
API_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_API_VERSION = "2023-06-01"
SDK_VERSION = "0.120.2"
EFFORT = "high"
SERVICE_TIER = "standard_only"
TIMEOUT_SECONDS = 600.0
MAX_OUTPUT_TOKENS = 10_240
FIXED_FRAMING_TOKENS = 1_024
INPUT_RATE_USD_PER_MTOK = Decimal("10")
OUTPUT_RATE_USD_PER_MTOK = Decimal("50")

CANARY_HARD_CEILING_USD_MICROS = 600_000
SCIENTIFIC_PHASE_HARD_CEILING_USD_MICROS = 2_400_000
FRESH_V4_HARD_CEILING_USD_MICROS = 3_000_000
PRE_V4_RECONCILED_LIABILITY_USD_MICROS = 58_031_869
PROJECT_AFTER_V4_RESERVATION_USD_MICROS = 61_031_869
PROJECT_HARD_STOP_USD_MICROS = 100_000_000

DEFAULT_WORKSPACE = Path("data/cache/hosted-native-numeric-source-free-canary-v4-live")

CANARY_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "canary": {
            "type": "string",
            "const": "HOSTED_NUMERIC_V4_PROMPT_JSON_OK",
        },
        "ordinal": {"type": "integer", "const": 4},
    },
    "required": ["canary", "ordinal"],
    "additionalProperties": False,
}
CANARY_FIXTURE: dict[str, Any] = {
    "canary": "HOSTED_NUMERIC_V4_PROMPT_JSON_OK",
    "ordinal": 4,
}
BASE_SYSTEM = (
    "Return exactly one JSON object satisfying the schema appended to this system "
    "message. Do not use markdown, commentary, or code fences."
)
BASE_PROMPT = (
    "This is a source-free transport canary. Return exactly this synthetic object: "
    '{"canary":"HOSTED_NUMERIC_V4_PROMPT_JSON_OK","ordinal":4}'
)

_FORBIDDEN_SOURCE_VALUES = (
    "PMC2427034",
    "PMC3104134",
    "2afd321025e677af36ddef3d26a03af1e7197cdac5798996c2415324c436c049",
    "b00a6b52cff19111fe0c7a0e7770e3267f92f4f6e78199c747be7e6c19642205",
    "data/cache/evidence-inference-2.0/txt_files/PMC2427034.txt",
    "data/cache/evidence-inference-2.0/txt_files/PMC3104134.txt",
    "harvest-sha256:2afd321025e677af36ddef3d26a03af1e7197cdac5798996c2415324c436c049",
    "harvest-sha256:b00a6b52cff19111fe0c7a0e7770e3267f92f4f6e78199c747be7e6c19642205",
)
_SECRET_VALUE_RE = re.compile(r"(?i)(?:sk-ant-[A-Za-z0-9_-]{12,}|bearer\s+[A-Za-z0-9._-]{12,})")
_SECRET_KEY_RE = re.compile(
    r"(?i)(?:api[_-]?key|authorization|proxy[_-]?authorization|x[_-]?api[_-]?key)"
)
_SAFE_SCALAR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_EXECUTION_ID_RE = re.compile(
    r"^hosted-native-numeric-canary-v4-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


class HostedNativeNumericCanaryV4Error(ValueError):
    """The source-free request, budget, state, or response failed closed."""


class _Frozen(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]
NonEmpty = Annotated[str, Field(min_length=1)]
TokenCount = Annotated[StrictInt, Field(ge=0)]


def _self_hash(value: ContractModel, field: str, code: str) -> None:
    expected = hash_canonical(value.model_dump(mode="json", exclude={field}))
    if getattr(value, field) != expected:
        raise ValueError(code)


def _sha256_utf8(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _usd_micros(input_tokens: int, output_tokens: int) -> int:
    return input_tokens * int(INPUT_RATE_USD_PER_MTOK) + output_tokens * int(
        OUTPUT_RATE_USD_PER_MTOK
    )


def _new_execution_id() -> str:
    return f"hosted-native-numeric-canary-v4-{uuid.uuid4()}"


def _validate_execution_id(value: str) -> str:
    if not _EXECUTION_ID_RE.fullmatch(value):
        raise ValueError("hosted_numeric_canary_v4_execution_id_invalid")
    parsed = uuid.UUID(value.removeprefix("hosted-native-numeric-canary-v4-"))
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122:
        raise ValueError("hosted_numeric_canary_v4_execution_id_invalid")
    return value


def _assert_no_secret(value: Any) -> None:
    try:
        rendered = canonical_json_bytes(value).decode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_artifact_not_canonical_json"
        ) from exc
    folded = rendered.casefold()
    if _SECRET_VALUE_RE.search(rendered):
        raise HostedNativeNumericCanaryV4Error("hosted_numeric_canary_v4_secret_value_forbidden")
    if any(item.casefold() in folded for item in _FORBIDDEN_SOURCE_VALUES):
        raise HostedNativeNumericCanaryV4Error("hosted_numeric_canary_v4_source_identity_present")


def assert_source_free_canary_payload_v4(value: Any) -> None:
    """Reject credential markers and every frozen source identity/path."""

    try:
        rendered = canonical_json_bytes(value).decode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_source_free_payload_invalid"
        ) from exc
    folded = rendered.casefold()
    if any(item.casefold() in folded for item in _FORBIDDEN_SOURCE_VALUES):
        raise HostedNativeNumericCanaryV4Error("hosted_numeric_canary_v4_source_identity_present")
    if _SECRET_VALUE_RE.search(rendered) or _SECRET_KEY_RE.search(rendered):
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_credential_surface_present"
        )


class CanaryProviderConfigV4(_Frozen):
    provider: Literal["anthropic_first_party_api"] = "anthropic_first_party_api"
    model: Literal["claude-fable-5"] = MODEL
    model_revision: Literal["claude-fable-5"] = MODEL_REVISION
    api_base_url: Literal["https://api.anthropic.com"] = API_BASE_URL
    anthropic_api_version: Literal["2023-06-01"] = ANTHROPIC_API_VERSION
    sdk_name: Literal["anthropic-python"] = "anthropic-python"
    sdk_version: Literal["0.120.2"] = SDK_VERSION
    credential_delivery: Literal["explicit_in_memory_constructor_only"] = (
        "explicit_in_memory_constructor_only"
    )
    effort: Literal["high"] = EFFORT
    service_tier: Literal["standard_only"] = SERVICE_TIER
    timeout_seconds: Literal[600] = 600
    max_output_tokens: Literal[10240] = MAX_OUTPUT_TOKENS
    transport_mode: Literal["prompt_json_schema"] = "prompt_json_schema"
    wire_schema_delivery: Literal["canonical_model_system"] = "canonical_model_system"
    structured_grammar_enforced_by_provider: Literal[False] = False
    output_format_present_in_call: Literal[False] = False
    application_retries: Literal[0] = 0
    sdk_retries: Literal[0] = 0
    provider_attempts: Literal[1] = 1
    http_environment_trust: Literal[False] = False
    follow_redirects: Literal[False] = False
    input_rate_usd_per_million_tokens: Decimal = INPUT_RATE_USD_PER_MTOK
    output_rate_usd_per_million_tokens: Decimal = OUTPUT_RATE_USD_PER_MTOK
    config_sha256: Sha256

    @model_validator(mode="after")
    def validate_config(self) -> CanaryProviderConfigV4:
        if (
            self.input_rate_usd_per_million_tokens != INPUT_RATE_USD_PER_MTOK
            or self.output_rate_usd_per_million_tokens != OUTPUT_RATE_USD_PER_MTOK
        ):
            raise ValueError("hosted_numeric_canary_v4_rate_mismatch")
        _self_hash(self, "config_sha256", "hosted_numeric_canary_v4_config_hash_mismatch")
        return self


def _provider_config() -> CanaryProviderConfigV4:
    payload = {
        "provider": "anthropic_first_party_api",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "api_base_url": API_BASE_URL,
        "anthropic_api_version": ANTHROPIC_API_VERSION,
        "sdk_name": "anthropic-python",
        "sdk_version": SDK_VERSION,
        "credential_delivery": "explicit_in_memory_constructor_only",
        "effort": EFFORT,
        "service_tier": SERVICE_TIER,
        "timeout_seconds": 600,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "transport_mode": "prompt_json_schema",
        "wire_schema_delivery": "canonical_model_system",
        "structured_grammar_enforced_by_provider": False,
        "output_format_present_in_call": False,
        "application_retries": 0,
        "sdk_retries": 0,
        "provider_attempts": 1,
        "http_environment_trust": False,
        "follow_redirects": False,
        "input_rate_usd_per_million_tokens": str(INPUT_RATE_USD_PER_MTOK),
        "output_rate_usd_per_million_tokens": str(OUTPUT_RATE_USD_PER_MTOK),
    }
    return CanaryProviderConfigV4.model_validate(
        {**payload, "config_sha256": hash_canonical(payload)}
    )


class HostedNativeNumericCanaryPlanV4(_Frozen):
    plan_version: Literal["hosted-native-numeric-source-free-canary-plan-v4"] = PLAN_VERSION
    status: Literal["offline_source_free_no_provider_calls"] = (
        "offline_source_free_no_provider_calls"
    )
    execution_id: NonEmpty
    request_key: NonEmpty
    provider_config: CanaryProviderConfigV4
    compiled_schema: AnthropicCompiledSchemaV1
    full_acceptance_schema_sha256: Sha256
    delivered_schema_kind: Literal["original_full_acceptance_schema"] = (
        "original_full_acceptance_schema"
    )
    delivered_schema_sha256: Sha256
    expected_fixture: dict[str, Any]
    expected_fixture_sha256: Sha256
    base_system: NonEmpty
    model_system: NonEmpty
    model_system_sha256: Sha256
    prompt: NonEmpty
    prompt_sha256: Sha256
    wire_request: dict[str, Any]
    wire_request_sha256: Sha256
    request_sha256: Sha256
    context_binding_sha256: Sha256
    source_bearing: Literal[False] = False
    source_free_scan_policy: Literal[
        "exclude-two-pmc-identities-hashes-paths-and-credential-markers-v1"
    ] = "exclude-two-pmc-identities-hashes-paths-and-credential-markers-v1"
    source_free_scan_passed: Literal[True] = True
    source_free_scan_sha256: Sha256
    conservative_input_token_ceiling: Annotated[StrictInt, Field(ge=1)]
    certified_request_liability_usd_micros: Annotated[
        StrictInt, Field(ge=1, le=CANARY_HARD_CEILING_USD_MICROS)
    ]
    canary_hard_ceiling_usd_micros: Literal[600000] = CANARY_HARD_CEILING_USD_MICROS
    scientific_phase_hard_ceiling_usd_micros: Literal[2400000] = (
        SCIENTIFIC_PHASE_HARD_CEILING_USD_MICROS
    )
    fresh_v4_hard_ceiling_usd_micros: Literal[3000000] = FRESH_V4_HARD_CEILING_USD_MICROS
    pre_v4_reconciled_liability_usd_micros: Literal[58031869] = (
        PRE_V4_RECONCILED_LIABILITY_USD_MICROS
    )
    project_after_v4_reservation_usd_micros: Literal[61031869] = (
        PROJECT_AFTER_V4_RESERVATION_USD_MICROS
    )
    project_hard_stop_usd_micros: Literal[100000000] = PROJECT_HARD_STOP_USD_MICROS
    plan_sha256: Sha256

    @field_validator("execution_id")
    @classmethod
    def validate_execution_id(cls, value: str) -> str:
        return _validate_execution_id(value)

    @model_validator(mode="after")
    def validate_plan(self) -> HostedNativeNumericCanaryPlanV4:
        assert_source_free_canary_payload_v4(self.wire_request)
        expected_input = (
            len(self.model_system.encode("utf-8"))
            + len(self.prompt.encode("utf-8"))
            + FIXED_FRAMING_TOKENS
        )
        expected_liability = _usd_micros(expected_input, MAX_OUTPUT_TOKENS)
        scan_payload = {
            "policy": self.source_free_scan_policy,
            "wire_request_sha256": self.wire_request_sha256,
            "source_bearing": False,
            "passed": True,
        }
        if (
            self.expected_fixture != CANARY_FIXTURE
            or self.expected_fixture_sha256 != hash_canonical(CANARY_FIXTURE)
            or self.full_acceptance_schema_sha256 != hash_canonical(CANARY_SCHEMA)
            or self.delivered_schema_sha256 != hash_canonical(CANARY_SCHEMA)
            or self.model_system_sha256 != _sha256_utf8(self.model_system)
            or self.prompt_sha256 != _sha256_utf8(self.prompt)
            or self.wire_request_sha256 != hash_canonical(self.wire_request)
            or self.request_sha256
            != hash_canonical(
                {
                    "execution_id": self.execution_id,
                    "request_key": self.request_key,
                    "wire_request_sha256": self.wire_request_sha256,
                    "provider_config_sha256": self.provider_config.config_sha256,
                }
            )
            or self.source_free_scan_sha256 != hash_canonical(scan_payload)
            or self.conservative_input_token_ceiling != expected_input
            or self.certified_request_liability_usd_micros != expected_liability
            or expected_liability > CANARY_HARD_CEILING_USD_MICROS
            or CANARY_HARD_CEILING_USD_MICROS + SCIENTIFIC_PHASE_HARD_CEILING_USD_MICROS
            != FRESH_V4_HARD_CEILING_USD_MICROS
            or PRE_V4_RECONCILED_LIABILITY_USD_MICROS + FRESH_V4_HARD_CEILING_USD_MICROS
            != PROJECT_AFTER_V4_RESERVATION_USD_MICROS
            or PROJECT_AFTER_V4_RESERVATION_USD_MICROS >= PROJECT_HARD_STOP_USD_MICROS
        ):
            raise ValueError("hosted_numeric_canary_v4_plan_binding_mismatch")
        _self_hash(self, "plan_sha256", "hosted_numeric_canary_v4_plan_hash_mismatch")
        return self


def freeze_hosted_native_numeric_canary_plan_v4(
    *, execution_id: str | None = None
) -> HostedNativeNumericCanaryPlanV4:
    frozen_execution_id = execution_id or _new_execution_id()
    _validate_execution_id(frozen_execution_id)
    config = _provider_config()
    compiled = compile_anthropic_bounded_schema(
        original_schema=CANARY_SCHEMA,
        full_acceptance_schema_sha256=hash_canonical(CANARY_SCHEMA),
    )
    model_system = render_anthropic_prompt_json_model_system(
        base_system=BASE_SYSTEM,
        wire_schema=CANARY_SCHEMA,
    )
    wire_request = {
        "model": MODEL,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": model_system,
        "messages": [{"role": "user", "content": BASE_PROMPT}],
        "output_config": {"effort": EFFORT},
        "service_tier": SERVICE_TIER,
    }
    assert "format" not in wire_request["output_config"]
    assert_source_free_canary_payload_v4(wire_request)
    wire_sha = hash_canonical(wire_request)
    execution_uuid = frozen_execution_id.removeprefix("hosted-native-numeric-canary-v4-").replace(
        "-", ""
    )
    request_key = "source-free-canary-v4-" + execution_uuid
    request_sha = hash_canonical(
        {
            "execution_id": frozen_execution_id,
            "request_key": request_key,
            "wire_request_sha256": wire_sha,
            "provider_config_sha256": config.config_sha256,
        }
    )
    scan_policy = "exclude-two-pmc-identities-hashes-paths-and-credential-markers-v1"
    scan_payload = {
        "policy": scan_policy,
        "wire_request_sha256": wire_sha,
        "source_bearing": False,
        "passed": True,
    }
    input_ceiling = (
        len(model_system.encode("utf-8")) + len(BASE_PROMPT.encode("utf-8")) + FIXED_FRAMING_TOKENS
    )
    liability = _usd_micros(input_ceiling, MAX_OUTPUT_TOKENS)
    if liability > CANARY_HARD_CEILING_USD_MICROS:
        raise HostedNativeNumericCanaryV4Error("hosted_numeric_canary_v4_cost_ceiling_exceeded")
    payload = {
        "plan_version": PLAN_VERSION,
        "status": "offline_source_free_no_provider_calls",
        "execution_id": frozen_execution_id,
        "request_key": request_key,
        "provider_config": config,
        "compiled_schema": compiled,
        "full_acceptance_schema_sha256": hash_canonical(CANARY_SCHEMA),
        "delivered_schema_kind": "original_full_acceptance_schema",
        "delivered_schema_sha256": hash_canonical(CANARY_SCHEMA),
        "expected_fixture": CANARY_FIXTURE,
        "expected_fixture_sha256": hash_canonical(CANARY_FIXTURE),
        "base_system": BASE_SYSTEM,
        "model_system": model_system,
        "model_system_sha256": _sha256_utf8(model_system),
        "prompt": BASE_PROMPT,
        "prompt_sha256": _sha256_utf8(BASE_PROMPT),
        "wire_request": wire_request,
        "wire_request_sha256": wire_sha,
        "request_sha256": request_sha,
        "context_binding_sha256": hash_canonical(
            {
                "role": "source_free_prompt_json_transport_canary",
                "expected_fixture_sha256": hash_canonical(CANARY_FIXTURE),
                "delivered_schema_sha256": hash_canonical(CANARY_SCHEMA),
                "compiled_wire_schema_sha256": compiled.wire_schema_sha256,
            }
        ),
        "source_bearing": False,
        "source_free_scan_policy": scan_policy,
        "source_free_scan_passed": True,
        "source_free_scan_sha256": hash_canonical(scan_payload),
        "conservative_input_token_ceiling": input_ceiling,
        "certified_request_liability_usd_micros": liability,
        "canary_hard_ceiling_usd_micros": CANARY_HARD_CEILING_USD_MICROS,
        "scientific_phase_hard_ceiling_usd_micros": (SCIENTIFIC_PHASE_HARD_CEILING_USD_MICROS),
        "fresh_v4_hard_ceiling_usd_micros": FRESH_V4_HARD_CEILING_USD_MICROS,
        "pre_v4_reconciled_liability_usd_micros": (PRE_V4_RECONCILED_LIABILITY_USD_MICROS),
        "project_after_v4_reservation_usd_micros": (PROJECT_AFTER_V4_RESERVATION_USD_MICROS),
        "project_hard_stop_usd_micros": PROJECT_HARD_STOP_USD_MICROS,
    }
    return HostedNativeNumericCanaryPlanV4.model_validate(
        {**payload, "plan_sha256": hash_canonical(payload)}
    )


class HostedNativeNumericCanaryAuthorizationV4(_Frozen):
    authorization_version: Literal["hosted-native-numeric-source-free-canary-authorization-v4"] = (
        AUTHORIZATION_VERSION
    )
    execution_id: NonEmpty
    plan_sha256: Sha256
    request_key: NonEmpty
    request_sha256: Sha256
    source_bearing_call_count: Literal[0] = 0
    authorized_call_count: Literal[1] = 1
    provider_calls_made_before_authorization: Literal[0] = 0
    certified_request_liability_usd_micros: Annotated[
        StrictInt, Field(ge=1, le=CANARY_HARD_CEILING_USD_MICROS)
    ]
    configured_canary_budget_usd_micros: Literal[600000] = CANARY_HARD_CEILING_USD_MICROS
    fresh_v4_combined_ceiling_usd_micros: Literal[3000000] = FRESH_V4_HARD_CEILING_USD_MICROS
    project_after_combined_reservation_usd_micros: Literal[61031869] = (
        PROJECT_AFTER_V4_RESERVATION_USD_MICROS
    )
    authorization_sha256: Sha256

    @field_validator("execution_id")
    @classmethod
    def validate_execution_id(cls, value: str) -> str:
        return _validate_execution_id(value)

    @model_validator(mode="after")
    def validate_authorization(self) -> HostedNativeNumericCanaryAuthorizationV4:
        if self.certified_request_liability_usd_micros > self.configured_canary_budget_usd_micros:
            raise ValueError("hosted_numeric_canary_v4_authorization_exceeds_budget")
        _self_hash(
            self,
            "authorization_sha256",
            "hosted_numeric_canary_v4_authorization_hash_mismatch",
        )
        return self


def freeze_hosted_native_numeric_canary_authorization_v4(
    plan: HostedNativeNumericCanaryPlanV4 | Mapping[str, Any],
) -> HostedNativeNumericCanaryAuthorizationV4:
    canonical = HostedNativeNumericCanaryPlanV4.model_validate(plan)
    payload = {
        "authorization_version": AUTHORIZATION_VERSION,
        "execution_id": canonical.execution_id,
        "plan_sha256": canonical.plan_sha256,
        "request_key": canonical.request_key,
        "request_sha256": canonical.request_sha256,
        "source_bearing_call_count": 0,
        "authorized_call_count": 1,
        "provider_calls_made_before_authorization": 0,
        "certified_request_liability_usd_micros": (
            canonical.certified_request_liability_usd_micros
        ),
        "configured_canary_budget_usd_micros": CANARY_HARD_CEILING_USD_MICROS,
        "fresh_v4_combined_ceiling_usd_micros": FRESH_V4_HARD_CEILING_USD_MICROS,
        "project_after_combined_reservation_usd_micros": (PROJECT_AFTER_V4_RESERVATION_USD_MICROS),
    }
    return HostedNativeNumericCanaryAuthorizationV4.model_validate(
        {**payload, "authorization_sha256": hash_canonical(payload)}
    )


class HostedNativeNumericCanaryIntentV4(_Frozen):
    intent_version: Literal["hosted-native-numeric-source-free-canary-intent-v4"] = INTENT_VERSION
    execution_id: NonEmpty
    plan_sha256: Sha256
    authorization_sha256: Sha256
    request_key: NonEmpty
    request_sha256: Sha256
    wire_request_sha256: Sha256
    context_binding_sha256: Sha256
    source_bearing: Literal[False] = False
    permitted_provider_attempts: Literal[1] = 1
    application_retries_permitted: Literal[0] = 0
    sdk_retries_permitted: Literal[0] = 0
    intent_durable_before_transport: Literal[True] = True
    orphan_is_terminal: Literal[True] = True
    attempt_id: Sha256
    intent_sha256: Sha256

    @field_validator("execution_id")
    @classmethod
    def validate_execution_id(cls, value: str) -> str:
        return _validate_execution_id(value)

    @model_validator(mode="after")
    def validate_intent(self) -> HostedNativeNumericCanaryIntentV4:
        expected_attempt = hash_canonical(
            {
                "execution_id": self.execution_id,
                "request_sha256": self.request_sha256,
                "authorization_sha256": self.authorization_sha256,
                "context_binding_sha256": self.context_binding_sha256,
                "permitted_provider_attempts": 1,
                "application_retries_permitted": 0,
                "sdk_retries_permitted": 0,
            }
        )
        if self.attempt_id != expected_attempt:
            raise ValueError("hosted_numeric_canary_v4_attempt_id_mismatch")
        _self_hash(self, "intent_sha256", "hosted_numeric_canary_v4_intent_hash_mismatch")
        return self


def freeze_hosted_native_numeric_canary_intent_v4(
    *,
    plan: HostedNativeNumericCanaryPlanV4 | Mapping[str, Any],
    authorization: HostedNativeNumericCanaryAuthorizationV4 | Mapping[str, Any],
) -> HostedNativeNumericCanaryIntentV4:
    canonical_plan = HostedNativeNumericCanaryPlanV4.model_validate(plan)
    canonical_auth = HostedNativeNumericCanaryAuthorizationV4.model_validate(authorization)
    if (
        canonical_auth.execution_id != canonical_plan.execution_id
        or canonical_auth.plan_sha256 != canonical_plan.plan_sha256
        or canonical_auth.request_key != canonical_plan.request_key
        or canonical_auth.request_sha256 != canonical_plan.request_sha256
        or canonical_auth.certified_request_liability_usd_micros
        != canonical_plan.certified_request_liability_usd_micros
    ):
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_authorization_plan_mismatch"
        )
    attempt = hash_canonical(
        {
            "execution_id": canonical_plan.execution_id,
            "request_sha256": canonical_plan.request_sha256,
            "authorization_sha256": canonical_auth.authorization_sha256,
            "context_binding_sha256": canonical_plan.context_binding_sha256,
            "permitted_provider_attempts": 1,
            "application_retries_permitted": 0,
            "sdk_retries_permitted": 0,
        }
    )
    payload = {
        "intent_version": INTENT_VERSION,
        "execution_id": canonical_plan.execution_id,
        "plan_sha256": canonical_plan.plan_sha256,
        "authorization_sha256": canonical_auth.authorization_sha256,
        "request_key": canonical_plan.request_key,
        "request_sha256": canonical_plan.request_sha256,
        "wire_request_sha256": canonical_plan.wire_request_sha256,
        "context_binding_sha256": canonical_plan.context_binding_sha256,
        "source_bearing": False,
        "permitted_provider_attempts": 1,
        "application_retries_permitted": 0,
        "sdk_retries_permitted": 0,
        "intent_durable_before_transport": True,
        "orphan_is_terminal": True,
        "attempt_id": attempt,
    }
    return HostedNativeNumericCanaryIntentV4.model_validate(
        {**payload, "intent_sha256": hash_canonical(payload)}
    )


class CanaryUsageV4(_Frozen):
    input_tokens: TokenCount
    output_tokens: TokenCount
    cache_creation_input_tokens: TokenCount = 0
    cache_read_input_tokens: TokenCount = 0

    @field_validator(
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        mode="before",
    )
    @classmethod
    def reject_bool(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("hosted_numeric_canary_v4_usage_bool_forbidden")
        return value

    @property
    def conservative_input_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens


@dataclass(frozen=True)
class HostedNativeNumericCanaryRawResponseV4:
    response_id: str | None
    response_model: str | None
    stop_reason: str | None
    content_block_count: int
    content_text: str | None
    input_tokens: int | None
    output_tokens: int | None
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class HostedNativeNumericCanaryClientProtocolV4(Protocol):
    def generate(
        self, wire_request: Mapping[str, Any]
    ) -> HostedNativeNumericCanaryRawResponseV4: ...


def _require_sdk() -> Any:
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_anthropic_sdk_missing"
        ) from exc
    observed = str(getattr(anthropic, "__version__", "unknown"))
    if observed != SDK_VERSION:
        raise HostedNativeNumericCanaryV4Error(
            f"hosted_numeric_canary_v4_anthropic_sdk_version_mismatch:{observed}"
        )
    return anthropic


class AnthropicFablePromptJsonCanaryClientV4:
    """Explicit-key, one-attempt Fable adapter with retries and env trust disabled."""

    def __init__(self, *, api_key: str) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise HostedNativeNumericCanaryV4Error(
                "hosted_numeric_canary_v4_anthropic_api_key_missing"
            )
        if os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("ANTHROPIC_CUSTOM_HEADERS"):
            raise HostedNativeNumericCanaryV4Error(
                "hosted_numeric_canary_v4_transport_environment_override_forbidden"
            )
        anthropic = _require_sdk()
        http_client = anthropic.DefaultHttpxClient(
            timeout=TIMEOUT_SECONDS,
            trust_env=False,
            follow_redirects=False,
        )
        self._client = anthropic.Anthropic(
            api_key=api_key,
            base_url=API_BASE_URL,
            default_headers={"anthropic-version": ANTHROPIC_API_VERSION},
            http_client=http_client,
            max_retries=0,
            timeout=TIMEOUT_SECONDS,
        )

    def generate(self, wire_request: Mapping[str, Any]) -> HostedNativeNumericCanaryRawResponseV4:
        response = self._client.messages.create(**dict(wire_request))
        content = list(getattr(response, "content", []))
        text: str | None = None
        if len(content) == 1 and getattr(content[0], "type", None) == "text":
            candidate = getattr(content[0], "text", None)
            text = candidate if isinstance(candidate, str) else None
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
        output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
        cache_creation = (
            getattr(usage, "cache_creation_input_tokens", 0) if usage is not None else 0
        )
        cache_read = getattr(usage, "cache_read_input_tokens", 0) if usage is not None else 0
        return HostedNativeNumericCanaryRawResponseV4(
            response_id=getattr(response, "id", None),
            response_model=getattr(response, "model", None),
            stop_reason=getattr(response, "stop_reason", None),
            content_block_count=len(content),
            content_text=text,
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
            cache_creation_input_tokens=cache_creation if isinstance(cache_creation, int) else 0,
            cache_read_input_tokens=cache_read if isinstance(cache_read, int) else 0,
        )


class HostedNativeNumericCanarySuccessV4(_Frozen):
    terminal_version: Literal["hosted-native-numeric-source-free-canary-success-v4"] = (
        SUCCESS_VERSION
    )
    status: Literal["passed_source_free_prompt_json_canary"] = (
        "passed_source_free_prompt_json_canary"
    )
    execution_id: NonEmpty
    plan_sha256: Sha256
    authorization_sha256: Sha256
    intent_sha256: Sha256
    attempt_id: Sha256
    request_sha256: Sha256
    wire_request_sha256: Sha256
    source_bearing: Literal[False] = False
    provider: Literal["anthropic_first_party_api"] = "anthropic_first_party_api"
    provider_config_sha256: Sha256
    transport_mode: Literal["prompt_json_schema"] = "prompt_json_schema"
    structured_grammar_enforced_by_provider: Literal[False] = False
    output_format_present_in_call: Literal[False] = False
    compiled_schema_sha256: Sha256
    wire_schema_sha256: Sha256
    delivered_schema_kind: Literal["original_full_acceptance_schema"] = (
        "original_full_acceptance_schema"
    )
    delivered_schema_sha256: Sha256
    response_id: NonEmpty
    response_model: Literal["claude-fable-5"] = MODEL
    stop_reason: Literal["end_turn"] = "end_turn"
    content_block_count: Literal[1] = 1
    response_text: NonEmpty
    response_text_sha256: Sha256
    parsed_fixture: dict[str, Any]
    parsed_fixture_sha256: Sha256
    expected_fixture_sha256: Sha256
    full_acceptance_schema_sha256: Sha256
    wire_schema_validated: Literal[True] = True
    delivered_schema_validated: Literal[True] = True
    full_acceptance_schema_validated: Literal[True] = True
    fixture_exact: Literal[True] = True
    usage: CanaryUsageV4
    observed_cost_usd_micros: Annotated[StrictInt, Field(ge=0)]
    certified_request_liability_usd_micros: Annotated[
        StrictInt, Field(ge=1, le=CANARY_HARD_CEILING_USD_MICROS)
    ]
    charged_cost_upper_bound_usd_micros: Annotated[
        StrictInt, Field(ge=1, le=CANARY_HARD_CEILING_USD_MICROS)
    ]
    provider_attempts_observed: Literal[1] = 1
    application_retries: Literal[0] = 0
    sdk_retries: Literal[0] = 0
    retry_permitted: Literal[False] = False
    scientific_authority: Literal[False] = False
    provider_result_sha256: Sha256
    terminal_sha256: Sha256

    @field_validator("execution_id")
    @classmethod
    def validate_execution_id(cls, value: str) -> str:
        return _validate_execution_id(value)

    @model_validator(mode="after")
    def validate_success(self) -> HostedNativeNumericCanarySuccessV4:
        expected_cost = _usd_micros(
            self.usage.conservative_input_tokens,
            self.usage.output_tokens,
        )
        provider_result_payload = {
            "provider": self.provider,
            "provider_config_sha256": self.provider_config_sha256,
            "request_sha256": self.request_sha256,
            "wire_request_sha256": self.wire_request_sha256,
            "transport_mode": self.transport_mode,
            "structured_grammar_enforced_by_provider": (
                self.structured_grammar_enforced_by_provider
            ),
            "output_format_present_in_call": self.output_format_present_in_call,
            "compiled_schema_sha256": self.compiled_schema_sha256,
            "wire_schema_sha256": self.wire_schema_sha256,
            "delivered_schema_kind": self.delivered_schema_kind,
            "delivered_schema_sha256": self.delivered_schema_sha256,
            "full_acceptance_schema_sha256": self.full_acceptance_schema_sha256,
            "response_id": self.response_id,
            "response_model": self.response_model,
            "stop_reason": self.stop_reason,
            "response_text_sha256": self.response_text_sha256,
            "parsed_fixture_sha256": self.parsed_fixture_sha256,
            "expected_fixture_sha256": self.expected_fixture_sha256,
            "wire_schema_validated": self.wire_schema_validated,
            "delivered_schema_validated": self.delivered_schema_validated,
            "full_acceptance_schema_validated": self.full_acceptance_schema_validated,
            "fixture_exact": self.fixture_exact,
            "usage": self.usage,
            "observed_cost_usd_micros": self.observed_cost_usd_micros,
            "certified_request_liability_usd_micros": (self.certified_request_liability_usd_micros),
            "charged_cost_upper_bound_usd_micros": (self.charged_cost_upper_bound_usd_micros),
            "provider_attempts_observed": self.provider_attempts_observed,
            "application_retries": self.application_retries,
            "sdk_retries": self.sdk_retries,
        }
        if (
            self.response_text_sha256 != _sha256_utf8(self.response_text)
            or self.parsed_fixture != CANARY_FIXTURE
            or self.parsed_fixture_sha256 != hash_canonical(CANARY_FIXTURE)
            or self.expected_fixture_sha256 != hash_canonical(CANARY_FIXTURE)
            or self.full_acceptance_schema_sha256 != hash_canonical(CANARY_SCHEMA)
            or self.delivered_schema_sha256 != hash_canonical(CANARY_SCHEMA)
            or self.observed_cost_usd_micros != expected_cost
            or expected_cost > self.certified_request_liability_usd_micros
            or self.charged_cost_upper_bound_usd_micros
            != self.certified_request_liability_usd_micros
            or self.provider_result_sha256 != hash_canonical(provider_result_payload)
        ):
            raise ValueError("hosted_numeric_canary_v4_success_binding_mismatch")
        _assert_no_secret(self.model_dump(mode="json", exclude={"terminal_sha256"}))
        _self_hash(self, "terminal_sha256", "hosted_numeric_canary_v4_terminal_hash_mismatch")
        return self


FailureCode = Literal[
    "orphan_intent_on_resume",
    "provider_http_failure",
    "provider_call_ambiguous_exception",
    "response_id_invalid",
    "response_model_mismatch",
    "response_stop_reason_refusal",
    "response_stop_reason_max_tokens",
    "response_stop_reason_invalid",
    "response_content_invalid",
    "response_usage_invalid",
    "response_json_invalid",
    "response_schema_invalid",
    "response_fixture_mismatch",
]


class HostedNativeNumericCanaryFailureV4(_Frozen):
    terminal_version: Literal["hosted-native-numeric-source-free-canary-failure-v4"] = (
        FAILURE_VERSION
    )
    status: Literal["terminal_failure", "terminal_ambiguous_attempt_poison"]
    execution_id: NonEmpty
    plan_sha256: Sha256
    authorization_sha256: Sha256
    intent_sha256: Sha256
    attempt_id: Sha256
    request_sha256: Sha256
    source_bearing: Literal[False] = False
    failure_code: FailureCode
    exception_type: str | None = None
    provider_http_status: Annotated[StrictInt, Field(ge=400, le=599)] | None = None
    provider_request_id: str | None = None
    provider_attempt_observation: Literal["attempted_once", "unknown_after_orphaned_intent"]
    possible_provider_attempts: Literal[1] = 1
    observed_cost_usd_micros: Annotated[StrictInt, Field(ge=0)] | None = None
    certified_request_liability_usd_micros: Annotated[
        StrictInt, Field(ge=1, le=CANARY_HARD_CEILING_USD_MICROS)
    ]
    charged_cost_upper_bound_usd_micros: Annotated[
        StrictInt, Field(ge=1, le=CANARY_HARD_CEILING_USD_MICROS)
    ]
    application_retries: Literal[0] = 0
    sdk_retries: Literal[0] = 0
    retry_permitted: Literal[False] = False
    scientific_authority: Literal[False] = False
    terminal_sha256: Sha256

    @field_validator("execution_id")
    @classmethod
    def validate_execution_id(cls, value: str) -> str:
        return _validate_execution_id(value)

    @field_validator("exception_type", "provider_request_id")
    @classmethod
    def validate_safe_scalar(cls, value: str | None) -> str | None:
        if value is not None and not _SAFE_SCALAR_RE.fullmatch(value):
            raise ValueError("hosted_numeric_canary_v4_failure_scalar_invalid")
        return value

    @model_validator(mode="after")
    def validate_failure(self) -> HostedNativeNumericCanaryFailureV4:
        orphan = self.failure_code == "orphan_intent_on_resume"
        ambiguous = self.status == "terminal_ambiguous_attempt_poison"
        if (
            orphan != (self.provider_attempt_observation == "unknown_after_orphaned_intent")
            or ambiguous != (orphan or self.failure_code == "provider_call_ambiguous_exception")
            or (self.failure_code == "provider_http_failure")
            != (self.provider_http_status is not None)
            or self.charged_cost_upper_bound_usd_micros
            != self.certified_request_liability_usd_micros
        ):
            raise ValueError("hosted_numeric_canary_v4_failure_shape_invalid")
        _assert_no_secret(self.model_dump(mode="json", exclude={"terminal_sha256"}))
        _self_hash(self, "terminal_sha256", "hosted_numeric_canary_v4_terminal_hash_mismatch")
        return self


CanaryTerminalV4 = HostedNativeNumericCanarySuccessV4 | HostedNativeNumericCanaryFailureV4
_TERMINAL_ADAPTER = TypeAdapter(CanaryTerminalV4)


class HostedNativeNumericCanarySuccessBindingV4(_Frozen):
    binding_version: Literal["hosted-native-numeric-source-free-canary-binding-v4"] = (
        BINDING_VERSION
    )
    terminal_relative_path: Literal["03-terminal.json"] = "03-terminal.json"
    terminal_artifact_sha256: Sha256
    terminal_sha256: Sha256
    plan_sha256: Sha256
    authorization_sha256: Sha256
    intent_sha256: Sha256
    execution_id: NonEmpty
    request_sha256: Sha256
    wire_request_sha256: Sha256
    provider: Literal["anthropic_first_party_api"] = "anthropic_first_party_api"
    response_model: Literal["claude-fable-5"] = MODEL
    transport_mode: Literal["prompt_json_schema"] = "prompt_json_schema"
    effort: Literal["high"] = EFFORT
    service_tier: Literal["standard_only"] = SERVICE_TIER
    max_output_tokens: Literal[10240] = MAX_OUTPUT_TOKENS
    structured_grammar_enforced_by_provider: Literal[False] = False
    output_format_present_in_call: Literal[False] = False
    source_bearing: Literal[False] = False
    provider_attempts_observed: Literal[1] = 1
    application_retries: Literal[0] = 0
    sdk_retries: Literal[0] = 0
    fixture_exact: Literal[True] = True
    provider_config_sha256: Sha256
    compiled_schema_sha256: Sha256
    wire_schema_sha256: Sha256
    delivered_schema_sha256: Sha256
    full_acceptance_schema_sha256: Sha256
    expected_fixture_sha256: Sha256
    provider_result_sha256: Sha256
    certified_request_liability_usd_micros: Annotated[
        StrictInt, Field(ge=1, le=CANARY_HARD_CEILING_USD_MICROS)
    ]
    charged_cost_upper_bound_usd_micros: Annotated[
        StrictInt, Field(ge=1, le=CANARY_HARD_CEILING_USD_MICROS)
    ]
    scientific_authority: Literal[False] = False
    binding_sha256: Sha256

    @field_validator("execution_id")
    @classmethod
    def validate_execution_id(cls, value: str) -> str:
        return _validate_execution_id(value)

    @model_validator(mode="after")
    def validate_binding(self) -> HostedNativeNumericCanarySuccessBindingV4:
        _self_hash(self, "binding_sha256", "hosted_numeric_canary_v4_binding_hash_mismatch")
        return self


def _fresh_workspace(workspace: Path) -> Path:
    absolute = Path(os.path.abspath(workspace))
    parent = absolute.parent
    try:
        resolved_parent = parent.resolve(strict=True)
    except OSError as exc:
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_workspace_parent_missing_or_unsafe"
        ) from exc
    if resolved_parent != parent or parent.is_symlink():
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_workspace_parent_missing_or_unsafe"
        )
    if absolute.exists() or absolute.is_symlink():
        raise HostedNativeNumericCanaryV4Error("hosted_numeric_canary_v4_workspace_must_be_fresh")
    try:
        absolute.mkdir(mode=0o700, exist_ok=False)
        os.chmod(absolute, 0o700)
    except OSError as exc:
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_workspace_create_failed"
        ) from exc
    return absolute


def _existing_workspace(workspace: Path) -> Path:
    absolute = Path(os.path.abspath(workspace))
    try:
        resolved = absolute.resolve(strict=True)
        metadata = absolute.stat(follow_symlinks=False)
    except OSError as exc:
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_workspace_missing_or_unsafe"
        ) from exc
    if (
        resolved != absolute
        or absolute.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_workspace_missing_or_unsafe"
        )
    return absolute


@contextmanager
def _workspace_lock(workspace: Path) -> Iterator[None]:
    lock_path = workspace / ".lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
        ):
            raise HostedNativeNumericCanaryV4Error("hosted_numeric_canary_v4_lock_unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


_ALLOWED_WORKSPACE_FILES = frozenset(
    {".lock", "00-plan.json", "01-authorization.json", "02-intent.json", "03-terminal.json"}
)


def _validate_workspace_entries(workspace: Path) -> None:
    for path in workspace.iterdir():
        if path.name not in _ALLOWED_WORKSPACE_FILES:
            raise HostedNativeNumericCanaryV4Error(
                "hosted_numeric_canary_v4_unexpected_workspace_artifact"
            )
        if path.is_symlink():
            raise HostedNativeNumericCanaryV4Error(
                "hosted_numeric_canary_v4_workspace_symlink_forbidden"
            )


def _atomic_write_json_0600(path: Path, value: Any) -> None:
    serialized = value.model_dump(mode="json") if isinstance(value, ContractModel) else value
    _assert_no_secret(serialized)
    if path.parent.is_symlink() or path.parent.resolve(strict=True) != path.parent:
        raise HostedNativeNumericCanaryV4Error("hosted_numeric_canary_v4_artifact_parent_unsafe")
    payload = canonical_json_bytes(serialized) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_artifact_missing_or_unsafe"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
    ):
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_artifact_missing_or_unsafe"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HostedNativeNumericCanaryV4Error("hosted_numeric_canary_v4_artifact_invalid") from exc
    if not isinstance(value, dict):
        raise HostedNativeNumericCanaryV4Error("hosted_numeric_canary_v4_artifact_not_object")
    _assert_no_secret(value)
    return value


def _write_or_replay(path: Path, value: ContractModel) -> None:
    if path.exists() or path.is_symlink():
        observed = _read_object(path)
        if observed != value.model_dump(mode="json"):
            raise HostedNativeNumericCanaryV4Error(
                "hosted_numeric_canary_v4_artifact_replay_mismatch"
            )
        return
    _atomic_write_json_0600(path, value)


def prepare_hosted_native_numeric_canary_v4(
    *, workspace: Path = DEFAULT_WORKSPACE, execution_id: str | None = None
) -> tuple[HostedNativeNumericCanaryPlanV4, HostedNativeNumericCanaryAuthorizationV4]:
    plan = freeze_hosted_native_numeric_canary_plan_v4(execution_id=execution_id)
    authorization = freeze_hosted_native_numeric_canary_authorization_v4(plan)
    root = _fresh_workspace(workspace)
    with _workspace_lock(root):
        _validate_workspace_entries(root)
        _write_or_replay(root / "00-plan.json", plan)
        _write_or_replay(root / "01-authorization.json", authorization)
    return plan, authorization


def _load_plan_and_authorization(
    workspace: Path,
) -> tuple[HostedNativeNumericCanaryPlanV4, HostedNativeNumericCanaryAuthorizationV4]:
    plan = HostedNativeNumericCanaryPlanV4.model_validate(_read_object(workspace / "00-plan.json"))
    expected_plan = freeze_hosted_native_numeric_canary_plan_v4(execution_id=plan.execution_id)
    if plan != expected_plan:
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_plan_external_replay_mismatch"
        )
    authorization = HostedNativeNumericCanaryAuthorizationV4.model_validate(
        _read_object(workspace / "01-authorization.json")
    )
    expected_auth = freeze_hosted_native_numeric_canary_authorization_v4(plan)
    if authorization != expected_auth:
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_authorization_external_replay_mismatch"
        )
    return plan, authorization


def preflight_hosted_native_numeric_canary_execution_v4(
    *, workspace: Path, expected_plan_sha256: str, expected_authorization_sha256: str
) -> bool:
    """Replay every offline gate; true means the sole provider attempt remains eligible."""

    root = _existing_workspace(workspace)
    with _workspace_lock(root):
        _validate_workspace_entries(root)
        plan, authorization = _load_plan_and_authorization(root)
        if (
            plan.plan_sha256 != expected_plan_sha256
            or authorization.authorization_sha256 != expected_authorization_sha256
        ):
            raise HostedNativeNumericCanaryV4Error(
                "hosted_numeric_canary_v4_execution_anchor_mismatch"
            )
        terminal_path = root / "03-terminal.json"
        intent_path = root / "02-intent.json"
        if terminal_path.exists():
            _validate_terminal_locked(
                workspace=root,
                plan=plan,
                authorization=authorization,
            )
            return False
        if intent_path.exists():
            expected_intent = freeze_hosted_native_numeric_canary_intent_v4(
                plan=plan, authorization=authorization
            )
            observed = HostedNativeNumericCanaryIntentV4.model_validate(_read_object(intent_path))
            if observed != expected_intent:
                raise HostedNativeNumericCanaryV4Error(
                    "hosted_numeric_canary_v4_intent_external_replay_mismatch"
                )
            return False
        return True


def _safe_scalar(value: Any) -> str | None:
    if not isinstance(value, str) or not _SAFE_SCALAR_RE.fullmatch(value):
        return None
    if _SECRET_VALUE_RE.search(value):
        return None
    folded = value.casefold()
    if any(item.casefold() in folded for item in _FORBIDDEN_SOURCE_VALUES):
        return None
    return value


def _strict_json_object(value: str) -> dict[str, Any]:
    def reject_constant(_: str) -> Any:
        raise ValueError("nonfinite_json_constant")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, child in pairs:
            if key in output:
                raise ValueError("duplicate_json_key")
            output[key] = child
        return output

    parsed = json.loads(
        value,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    if not isinstance(parsed, dict):
        raise ValueError("top_level_not_object")
    return parsed


def _failure_terminal(
    *,
    plan: HostedNativeNumericCanaryPlanV4,
    authorization: HostedNativeNumericCanaryAuthorizationV4,
    intent: HostedNativeNumericCanaryIntentV4,
    failure_code: FailureCode,
    exception_type: str | None = None,
    provider_http_status: int | None = None,
    provider_request_id: str | None = None,
    observed_cost_usd_micros: int | None = None,
) -> HostedNativeNumericCanaryFailureV4:
    orphan = failure_code == "orphan_intent_on_resume"
    ambiguous = orphan or failure_code == "provider_call_ambiguous_exception"
    payload = {
        "terminal_version": FAILURE_VERSION,
        "status": ("terminal_ambiguous_attempt_poison" if ambiguous else "terminal_failure"),
        "execution_id": plan.execution_id,
        "plan_sha256": plan.plan_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "intent_sha256": intent.intent_sha256,
        "attempt_id": intent.attempt_id,
        "request_sha256": plan.request_sha256,
        "source_bearing": False,
        "failure_code": failure_code,
        "exception_type": _safe_scalar(exception_type),
        "provider_http_status": provider_http_status,
        "provider_request_id": _safe_scalar(provider_request_id),
        "provider_attempt_observation": (
            "unknown_after_orphaned_intent" if orphan else "attempted_once"
        ),
        "possible_provider_attempts": 1,
        "observed_cost_usd_micros": observed_cost_usd_micros,
        "certified_request_liability_usd_micros": (plan.certified_request_liability_usd_micros),
        "charged_cost_upper_bound_usd_micros": (plan.certified_request_liability_usd_micros),
        "application_retries": 0,
        "sdk_retries": 0,
        "retry_permitted": False,
        "scientific_authority": False,
    }
    return HostedNativeNumericCanaryFailureV4.model_validate(
        {**payload, "terminal_sha256": hash_canonical(payload)}
    )


def _terminal_from_exception(
    *,
    plan: HostedNativeNumericCanaryPlanV4,
    authorization: HostedNativeNumericCanaryAuthorizationV4,
    intent: HostedNativeNumericCanaryIntentV4,
    exc: Exception,
) -> HostedNativeNumericCanaryFailureV4:
    status_raw = getattr(exc, "status_code", None)
    status = (
        status_raw
        if isinstance(status_raw, int)
        and not isinstance(status_raw, bool)
        and 400 <= status_raw <= 599
        else None
    )
    return _failure_terminal(
        plan=plan,
        authorization=authorization,
        intent=intent,
        failure_code=(
            "provider_http_failure" if status is not None else "provider_call_ambiguous_exception"
        ),
        exception_type=type(exc).__name__,
        provider_http_status=status,
        provider_request_id=getattr(exc, "request_id", None),
    )


def _terminal_from_raw(
    *,
    plan: HostedNativeNumericCanaryPlanV4,
    authorization: HostedNativeNumericCanaryAuthorizationV4,
    intent: HostedNativeNumericCanaryIntentV4,
    raw: HostedNativeNumericCanaryRawResponseV4,
) -> CanaryTerminalV4:
    response_id = _safe_scalar(raw.response_id)
    if response_id is None:
        return _failure_terminal(
            plan=plan,
            authorization=authorization,
            intent=intent,
            failure_code="response_id_invalid",
        )
    if raw.response_model != MODEL:
        return _failure_terminal(
            plan=plan,
            authorization=authorization,
            intent=intent,
            failure_code="response_model_mismatch",
            provider_request_id=response_id,
        )
    if raw.stop_reason != "end_turn":
        reason: FailureCode = {
            "refusal": "response_stop_reason_refusal",
            "max_tokens": "response_stop_reason_max_tokens",
        }.get(raw.stop_reason, "response_stop_reason_invalid")  # type: ignore[arg-type]
        return _failure_terminal(
            plan=plan,
            authorization=authorization,
            intent=intent,
            failure_code=reason,
            provider_request_id=response_id,
        )
    if (
        raw.content_block_count != 1
        or not isinstance(raw.content_text, str)
        or not raw.content_text
    ):
        return _failure_terminal(
            plan=plan,
            authorization=authorization,
            intent=intent,
            failure_code="response_content_invalid",
            provider_request_id=response_id,
        )
    try:
        usage = CanaryUsageV4(
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            cache_creation_input_tokens=raw.cache_creation_input_tokens,
            cache_read_input_tokens=raw.cache_read_input_tokens,
        )
    except Exception:
        return _failure_terminal(
            plan=plan,
            authorization=authorization,
            intent=intent,
            failure_code="response_usage_invalid",
            provider_request_id=response_id,
        )
    observed_cost = _usd_micros(usage.conservative_input_tokens, usage.output_tokens)
    if (
        usage.conservative_input_tokens > plan.conservative_input_token_ceiling
        or usage.output_tokens > MAX_OUTPUT_TOKENS
        or observed_cost > plan.certified_request_liability_usd_micros
    ):
        return _failure_terminal(
            plan=plan,
            authorization=authorization,
            intent=intent,
            failure_code="response_usage_invalid",
            provider_request_id=response_id,
            observed_cost_usd_micros=observed_cost,
        )
    try:
        parsed = _strict_json_object(raw.content_text)
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
        return _failure_terminal(
            plan=plan,
            authorization=authorization,
            intent=intent,
            failure_code="response_json_invalid",
            provider_request_id=response_id,
            observed_cost_usd_micros=observed_cost,
        )
    try:
        validator_for(plan.compiled_schema.wire_schema)(plan.compiled_schema.wire_schema).validate(
            parsed
        )
        validator_for(plan.compiled_schema.original_schema)(
            plan.compiled_schema.original_schema
        ).validate(parsed)
        validator_for(CANARY_SCHEMA)(CANARY_SCHEMA).validate(parsed)
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
            raise
        return _failure_terminal(
            plan=plan,
            authorization=authorization,
            intent=intent,
            failure_code="response_schema_invalid",
            provider_request_id=response_id,
            observed_cost_usd_micros=observed_cost,
        )
    if parsed != CANARY_FIXTURE:
        return _failure_terminal(
            plan=plan,
            authorization=authorization,
            intent=intent,
            failure_code="response_fixture_mismatch",
            provider_request_id=response_id,
            observed_cost_usd_micros=observed_cost,
        )
    provider_result_payload = {
        "provider": "anthropic_first_party_api",
        "provider_config_sha256": plan.provider_config.config_sha256,
        "request_sha256": plan.request_sha256,
        "wire_request_sha256": plan.wire_request_sha256,
        "transport_mode": "prompt_json_schema",
        "structured_grammar_enforced_by_provider": False,
        "output_format_present_in_call": False,
        "compiled_schema_sha256": plan.compiled_schema.compiled_schema_sha256,
        "wire_schema_sha256": plan.compiled_schema.wire_schema_sha256,
        "delivered_schema_kind": plan.delivered_schema_kind,
        "delivered_schema_sha256": plan.delivered_schema_sha256,
        "full_acceptance_schema_sha256": plan.full_acceptance_schema_sha256,
        "response_id": response_id,
        "response_model": MODEL,
        "stop_reason": "end_turn",
        "response_text_sha256": _sha256_utf8(raw.content_text),
        "parsed_fixture_sha256": hash_canonical(parsed),
        "expected_fixture_sha256": hash_canonical(CANARY_FIXTURE),
        "wire_schema_validated": True,
        "delivered_schema_validated": True,
        "full_acceptance_schema_validated": True,
        "fixture_exact": True,
        "usage": usage,
        "observed_cost_usd_micros": observed_cost,
        "certified_request_liability_usd_micros": (plan.certified_request_liability_usd_micros),
        "charged_cost_upper_bound_usd_micros": (plan.certified_request_liability_usd_micros),
        "provider_attempts_observed": 1,
        "application_retries": 0,
        "sdk_retries": 0,
    }
    payload = {
        "terminal_version": SUCCESS_VERSION,
        "status": "passed_source_free_prompt_json_canary",
        "execution_id": plan.execution_id,
        "plan_sha256": plan.plan_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "intent_sha256": intent.intent_sha256,
        "attempt_id": intent.attempt_id,
        "request_sha256": plan.request_sha256,
        "wire_request_sha256": plan.wire_request_sha256,
        "source_bearing": False,
        "provider": "anthropic_first_party_api",
        "provider_config_sha256": plan.provider_config.config_sha256,
        "transport_mode": "prompt_json_schema",
        "structured_grammar_enforced_by_provider": False,
        "output_format_present_in_call": False,
        "compiled_schema_sha256": plan.compiled_schema.compiled_schema_sha256,
        "wire_schema_sha256": plan.compiled_schema.wire_schema_sha256,
        "delivered_schema_kind": plan.delivered_schema_kind,
        "delivered_schema_sha256": plan.delivered_schema_sha256,
        "response_id": response_id,
        "response_model": MODEL,
        "stop_reason": "end_turn",
        "content_block_count": 1,
        "response_text": raw.content_text,
        "response_text_sha256": _sha256_utf8(raw.content_text),
        "parsed_fixture": parsed,
        "parsed_fixture_sha256": hash_canonical(parsed),
        "expected_fixture_sha256": hash_canonical(CANARY_FIXTURE),
        "full_acceptance_schema_sha256": hash_canonical(CANARY_SCHEMA),
        "wire_schema_validated": True,
        "delivered_schema_validated": True,
        "full_acceptance_schema_validated": True,
        "fixture_exact": True,
        "usage": usage,
        "observed_cost_usd_micros": observed_cost,
        "certified_request_liability_usd_micros": (plan.certified_request_liability_usd_micros),
        "charged_cost_upper_bound_usd_micros": (plan.certified_request_liability_usd_micros),
        "provider_attempts_observed": 1,
        "application_retries": 0,
        "sdk_retries": 0,
        "retry_permitted": False,
        "scientific_authority": False,
        "provider_result_sha256": hash_canonical(provider_result_payload),
    }
    return HostedNativeNumericCanarySuccessV4.model_validate(
        {**payload, "terminal_sha256": hash_canonical(payload)}
    )


def execute_hosted_native_numeric_canary_v4(
    *,
    workspace: Path,
    expected_plan_sha256: str,
    expected_authorization_sha256: str,
    client: HostedNativeNumericCanaryClientProtocolV4,
) -> CanaryTerminalV4:
    root = _existing_workspace(workspace)
    with _workspace_lock(root):
        _validate_workspace_entries(root)
        plan, authorization = _load_plan_and_authorization(root)
        if (
            plan.plan_sha256 != expected_plan_sha256
            or authorization.authorization_sha256 != expected_authorization_sha256
        ):
            raise HostedNativeNumericCanaryV4Error(
                "hosted_numeric_canary_v4_execution_anchor_mismatch"
            )
        intent = freeze_hosted_native_numeric_canary_intent_v4(
            plan=plan, authorization=authorization
        )
        intent_path = root / "02-intent.json"
        terminal_path = root / "03-terminal.json"
        if terminal_path.exists():
            return _validate_terminal_locked(
                workspace=root,
                plan=plan,
                authorization=authorization,
            )
        if intent_path.exists():
            observed_intent = HostedNativeNumericCanaryIntentV4.model_validate(
                _read_object(intent_path)
            )
            if observed_intent != intent:
                raise HostedNativeNumericCanaryV4Error(
                    "hosted_numeric_canary_v4_intent_external_replay_mismatch"
                )
            terminal = _failure_terminal(
                plan=plan,
                authorization=authorization,
                intent=intent,
                failure_code="orphan_intent_on_resume",
            )
            _write_or_replay(terminal_path, terminal)
            return terminal
        _write_or_replay(intent_path, intent)
        try:
            raw = client.generate(json.loads(canonical_json_bytes(plan.wire_request)))
        except (KeyboardInterrupt, SystemExit):  # pragma: no cover
            raise
        except Exception as exc:
            terminal = _terminal_from_exception(
                plan=plan,
                authorization=authorization,
                intent=intent,
                exc=exc,
            )
        else:
            terminal = _terminal_from_raw(
                plan=plan,
                authorization=authorization,
                intent=intent,
                raw=raw,
            )
        _write_or_replay(terminal_path, terminal)
        return terminal


def _validate_terminal_locked(
    *,
    workspace: Path,
    plan: HostedNativeNumericCanaryPlanV4,
    authorization: HostedNativeNumericCanaryAuthorizationV4,
) -> CanaryTerminalV4:
    intent = freeze_hosted_native_numeric_canary_intent_v4(plan=plan, authorization=authorization)
    observed_intent = HostedNativeNumericCanaryIntentV4.model_validate(
        _read_object(workspace / "02-intent.json")
    )
    if observed_intent != intent:
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_intent_external_replay_mismatch"
        )
    terminal = _TERMINAL_ADAPTER.validate_python(_read_object(workspace / "03-terminal.json"))
    if (
        terminal.execution_id != plan.execution_id
        or terminal.plan_sha256 != plan.plan_sha256
        or terminal.authorization_sha256 != authorization.authorization_sha256
        or terminal.intent_sha256 != intent.intent_sha256
        or terminal.attempt_id != intent.attempt_id
        or terminal.request_sha256 != plan.request_sha256
        or terminal.certified_request_liability_usd_micros
        != plan.certified_request_liability_usd_micros
        or (
            isinstance(terminal, HostedNativeNumericCanarySuccessV4)
            and (
                terminal.wire_request_sha256 != plan.wire_request_sha256
                or terminal.provider_config_sha256 != plan.provider_config.config_sha256
                or terminal.compiled_schema_sha256 != plan.compiled_schema.compiled_schema_sha256
                or terminal.wire_schema_sha256 != plan.compiled_schema.wire_schema_sha256
                or terminal.delivered_schema_sha256 != plan.delivered_schema_sha256
                or terminal.full_acceptance_schema_sha256 != plan.full_acceptance_schema_sha256
                or terminal.expected_fixture_sha256 != plan.expected_fixture_sha256
            )
        )
    ):
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_terminal_external_replay_mismatch"
        )
    return terminal


def validate_hosted_native_numeric_canary_terminal_v4(*, workspace: Path) -> CanaryTerminalV4:
    root = _existing_workspace(workspace)
    with _workspace_lock(root):
        _validate_workspace_entries(root)
        plan, authorization = _load_plan_and_authorization(root)
        return _validate_terminal_locked(
            workspace=root,
            plan=plan,
            authorization=authorization,
        )


def load_successful_hosted_native_numeric_canary_v4(
    *, workspace: Path, expected_terminal_sha256: str | None = None
) -> HostedNativeNumericCanarySuccessBindingV4:
    """Externally replay a passed canary and return its exact file/semantic binding."""

    root = _existing_workspace(workspace)
    with _workspace_lock(root):
        _validate_workspace_entries(root)
        plan, authorization = _load_plan_and_authorization(root)
        terminal = _validate_terminal_locked(
            workspace=root,
            plan=plan,
            authorization=authorization,
        )
        if not isinstance(terminal, HostedNativeNumericCanarySuccessV4):
            raise HostedNativeNumericCanaryV4Error(
                "hosted_numeric_canary_v4_success_terminal_required"
            )
        if (
            expected_terminal_sha256 is not None
            and terminal.terminal_sha256 != expected_terminal_sha256
        ):
            raise HostedNativeNumericCanaryV4Error(
                "hosted_numeric_canary_v4_expected_terminal_hash_mismatch"
            )
        payload = {
            "binding_version": BINDING_VERSION,
            "terminal_relative_path": "03-terminal.json",
            "terminal_artifact_sha256": sha256_file(root / "03-terminal.json"),
            "terminal_sha256": terminal.terminal_sha256,
            "plan_sha256": terminal.plan_sha256,
            "authorization_sha256": terminal.authorization_sha256,
            "intent_sha256": terminal.intent_sha256,
            "execution_id": terminal.execution_id,
            "request_sha256": terminal.request_sha256,
            "wire_request_sha256": terminal.wire_request_sha256,
            "provider": terminal.provider,
            "response_model": terminal.response_model,
            "transport_mode": terminal.transport_mode,
            "effort": EFFORT,
            "service_tier": SERVICE_TIER,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "structured_grammar_enforced_by_provider": False,
            "output_format_present_in_call": False,
            "source_bearing": False,
            "provider_attempts_observed": terminal.provider_attempts_observed,
            "application_retries": terminal.application_retries,
            "sdk_retries": terminal.sdk_retries,
            "fixture_exact": terminal.fixture_exact,
            "provider_config_sha256": terminal.provider_config_sha256,
            "compiled_schema_sha256": terminal.compiled_schema_sha256,
            "wire_schema_sha256": terminal.wire_schema_sha256,
            "delivered_schema_sha256": terminal.delivered_schema_sha256,
            "full_acceptance_schema_sha256": terminal.full_acceptance_schema_sha256,
            "expected_fixture_sha256": terminal.expected_fixture_sha256,
            "provider_result_sha256": terminal.provider_result_sha256,
            "certified_request_liability_usd_micros": (
                terminal.certified_request_liability_usd_micros
            ),
            "charged_cost_upper_bound_usd_micros": (terminal.charged_cost_upper_bound_usd_micros),
            "scientific_authority": False,
        }
        return HostedNativeNumericCanarySuccessBindingV4.model_validate(
            {**payload, "binding_sha256": hash_canonical(payload)}
        )


def require_hosted_native_numeric_canary_binding_v4(
    *,
    workspace: Path,
    expected_binding: HostedNativeNumericCanarySuccessBindingV4 | Mapping[str, Any],
) -> HostedNativeNumericCanarySuccessBindingV4:
    """Reload and recompare every raw canary artifact before a gated stage."""

    expected = HostedNativeNumericCanarySuccessBindingV4.model_validate(expected_binding)
    observed = load_successful_hosted_native_numeric_canary_v4(
        workspace=workspace,
        expected_terminal_sha256=expected.terminal_sha256,
    )
    if observed != expected:
        raise HostedNativeNumericCanaryV4Error(
            "hosted_numeric_canary_v4_binding_external_replay_mismatch"
        )
    return observed


def load_hosted_native_numeric_canary_status_v4(*, workspace: Path) -> dict[str, Any]:
    root = _existing_workspace(workspace)
    with _workspace_lock(root):
        _validate_workspace_entries(root)
        plan, authorization = _load_plan_and_authorization(root)
        if not (root / "03-terminal.json").exists():
            return {
                "status": (
                    "orphan_intent_requires_poison"
                    if (root / "02-intent.json").exists()
                    else "prepared_source_free_no_provider_calls"
                ),
                "execution_id": plan.execution_id,
                "plan_sha256": plan.plan_sha256,
                "authorization_sha256": authorization.authorization_sha256,
                "certified_request_liability_usd_micros": (
                    plan.certified_request_liability_usd_micros
                ),
                "source_bearing": False,
            }
    terminal = validate_hosted_native_numeric_canary_terminal_v4(workspace=root)
    return {
        "status": terminal.status,
        "execution_id": terminal.execution_id,
        "terminal_sha256": terminal.terminal_sha256,
        "certified_request_liability_usd_micros": (terminal.certified_request_liability_usd_micros),
        "charged_cost_upper_bound_usd_micros": (terminal.charged_cost_upper_bound_usd_micros),
        "source_bearing": False,
        "scientific_authority": False,
    }


__all__ = [
    "CANARY_FIXTURE",
    "CANARY_HARD_CEILING_USD_MICROS",
    "DEFAULT_WORKSPACE",
    "AnthropicFablePromptJsonCanaryClientV4",
    "HostedNativeNumericCanaryAuthorizationV4",
    "HostedNativeNumericCanaryFailureV4",
    "HostedNativeNumericCanaryPlanV4",
    "HostedNativeNumericCanaryRawResponseV4",
    "HostedNativeNumericCanarySuccessBindingV4",
    "HostedNativeNumericCanarySuccessV4",
    "HostedNativeNumericCanaryV4Error",
    "assert_source_free_canary_payload_v4",
    "execute_hosted_native_numeric_canary_v4",
    "freeze_hosted_native_numeric_canary_authorization_v4",
    "freeze_hosted_native_numeric_canary_intent_v4",
    "freeze_hosted_native_numeric_canary_plan_v4",
    "load_hosted_native_numeric_canary_status_v4",
    "load_successful_hosted_native_numeric_canary_v4",
    "preflight_hosted_native_numeric_canary_execution_v4",
    "prepare_hosted_native_numeric_canary_v4",
    "require_hosted_native_numeric_canary_binding_v4",
    "validate_hosted_native_numeric_canary_terminal_v4",
]
