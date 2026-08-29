"""Bounded Fable 5 runtime for two contextual-grounding v3 witnesses.

This is an additive successor boundary.  It imports the already-audited Anthropic
schema compiler, but it does not alter or impersonate the immutable Sonnet request
contract.  All model-facing bytes, provider-transformed schema bytes, wire kwargs,
and maximum liability are frozen before a phase can be authorized.  Each intent is
durable before transport; an orphan or ambiguous attempt poisons that exact request
and is never retried.

The runtime is a source-visible mechanics smoke only.  A successful provider output
must pass contextual grounding and fresh native-graph projection in trusted code.
It never grants extraction-accuracy, synthesis-input, scientific, calibration, or
claim-release authority.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from jsonschema.validators import validator_for
from pydantic import ConfigDict, Field, StrictInt, field_validator, model_validator

from literature_multiverse.anthropic_bounded_generation import (
    ANTHROPIC_SDK_VERSION,
    AnthropicCompiledSchemaV1,
    compile_anthropic_bounded_schema,
)
from literature_multiverse.contextual_numeric_grounding_v3 import (
    ContextualGroundedClaimV3,
    ContextualGroundedEffectV3,
    ContextualGroundingFeasibilityReceiptV3,
    ContextualNativeProjectionV3,
    ContextualPacketAbstentionV3,
    ContextualPacketCompletedV3,
    ContextualProviderBindingV3,
    freeze_contextual_grounding_offline_feasibility_suite_v3,
    project_contextual_grounded_outcome_v3,
)
from literature_multiverse.lineage import atomic_write_json as _atomic_write_json
from literature_multiverse.lineage import canonical_json_bytes, hash_canonical, sha256_file
from literature_multiverse.models import SHA256_RE, ContractModel

RUNTIME_VERSION = "metasyn-contextual-frontier-runtime-v1"
CONFIG_VERSION = "metasyn-contextual-frontier-runtime-config-v1"
PLAN_VERSION = "metasyn-contextual-frontier-plan-v1"
REQUEST_VERSION = "metasyn-contextual-frontier-request-v1"
PROVIDER_RESULT_VERSION = "metasyn-contextual-frontier-provider-result-v1"
AUTHORIZATION_VERSION = "metasyn-contextual-frontier-authorization-v1"
INTENT_VERSION = "metasyn-contextual-frontier-intent-v1"
RECEIPT_VERSION = "metasyn-contextual-frontier-provider-receipt-v1"
INCIDENT_VERSION = "metasyn-contextual-frontier-ambiguity-incident-v1"
VALIDATION_VERSION = "metasyn-contextual-frontier-validation-result-v1"
TERMINAL_VERSION = "metasyn-contextual-frontier-terminal-report-v1"

DEFAULT_CONFIG_PATH = Path("configs/benchmarks/metasyn-contextual-frontier-runtime-v1.json")
DEFAULT_WORKSPACE = Path("data/cache/metasyn/contextual-frontier-runtime-v1")
RUNTIME_SOURCE_PATH = Path("src/literature_multiverse/metasyn_contextual_frontier_runtime_v1.py")
CONTEXTUAL_SOURCE_PATH = Path("src/literature_multiverse/contextual_numeric_grounding_v3.py")
COMPILER_SOURCE_PATH = Path("src/literature_multiverse/anthropic_bounded_generation.py")

MODEL = "claude-fable-5"
API_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_API_VERSION = "2023-06-01"
SDK_VERSION = "0.120.2"
EFFORT = "high"
SERVICE_TIER = "standard_only"
INPUT_RATE = Decimal("10")
OUTPUT_RATE = Decimal("50")
FIXED_FRAMING_TOKENS = 2048
MAX_OUTPUT_TOKENS = 32_000
MAX_INPUT_TOKENS = 1_000_000
TIMEOUT_SECONDS = 600.0
PRICING_SOURCE_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
MODEL_SOURCE_URL = "https://platform.claude.com/docs/en/models/overview"
MODEL_ID_SOURCE_URL = (
    "https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions"
)
STRUCTURED_OUTPUTS_SOURCE_URL = (
    "https://platform.claude.com/docs/en/build-with-claude/structured-outputs"
)
SOURCE_VERIFIED_DATE = "2026-08-29"
SYSTEM_PROMPT = (
    "You are a bounded scientific evidence extractor. Return exactly one JSON "
    "object accepted by the terminal schema. Copy only exact source text from the "
    "frozen prompt. Never infer missing values; use the explicit abstention object."
)
PRIMARY_WITNESS = "metasyn-row17-candidate3-binary-primary-endpoint"
FALLBACK_WITNESS = "metasyn-row17-candidate2-binary-symptom-endpoint"
REQUEST_ORDER = (PRIMARY_WITNESS, FALLBACK_WITNESS)

_SECRET_VALUE_RE = re.compile(r"(?i)(?:sk-ant-[A-Za-z0-9_-]+|bearer\s+[A-Za-z0-9._-]+)")
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")


class MetaSynContextualFrontierRuntimeV1Error(ValueError):
    """A frozen request, durable transition, or provider archive failed closed."""


class _Frozen(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]
NonEmpty = Annotated[str, Field(min_length=1)]
TokenCount = Annotated[StrictInt, Field(ge=0)]


def _sha256_utf8(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _usd_micros(value: Decimal) -> int:
    return int((value * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING))


def _contract_json(value: Any) -> Any:
    """Match Pydantic's exact JSON-mode representation before self-hashing."""

    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, ContractModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _contract_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_contract_json(item) for item in value]
    return value


def _self_hash(model: _Frozen, field: str, code: str) -> None:
    observed = getattr(model, field)
    expected = hash_canonical(model.model_dump(mode="json", exclude={field}))
    if observed != expected:
        raise ValueError(code)


def _assert_secret_free(value: Any) -> None:
    """Reject credential-like values and credential-bearing mapping keys."""

    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized = str(key).lower().replace("-", "_")
                if normalized in {
                    "api_key",
                    "anthropic_api_key",
                    "x_api_key",
                    "proxy_authorization",
                }:
                    raise ValueError("metasyn_contextual_frontier_secret_key_forbidden")
                pending.append(child)
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
        elif isinstance(item, str) and _SECRET_VALUE_RE.search(item):
            raise ValueError("metasyn_contextual_frontier_secret_value_forbidden")


def _require_sdk() -> Any:
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise MetaSynContextualFrontierRuntimeV1Error("contextual_frontier_sdk_missing") from exc
    observed = str(getattr(anthropic, "__version__", "unknown"))
    if observed != SDK_VERSION or observed != ANTHROPIC_SDK_VERSION:
        raise MetaSynContextualFrontierRuntimeV1Error(
            f"contextual_frontier_sdk_version_mismatch:{observed}"
        )
    return anthropic


class MetaSynContextualFrontierConfigV1(_Frozen):
    config_version: Literal["metasyn-contextual-frontier-runtime-config-v1"] = CONFIG_VERSION
    model: Literal["claude-fable-5"] = MODEL
    api_base_url: Literal["https://api.anthropic.com"] = API_BASE_URL
    anthropic_api_version: Literal["2023-06-01"] = ANTHROPIC_API_VERSION
    anthropic_sdk_version: Literal["0.120.2"] = SDK_VERSION
    effort: Literal["high"] = EFFORT
    service_tier: Literal["standard_only"] = SERVICE_TIER
    transport_mode: Literal["structured_json_schema"] = "structured_json_schema"
    timeout_seconds: Literal[600.0] = TIMEOUT_SECONDS
    max_output_tokens: Literal[32000] = MAX_OUTPUT_TOKENS
    maximum_input_tokens: Literal[1000000] = MAX_INPUT_TOKENS
    fixed_framing_tokens: Literal[2048] = FIXED_FRAMING_TOKENS
    input_rate_usd_per_million_tokens: Annotated[Decimal, Field(gt=0)] = INPUT_RATE
    output_rate_usd_per_million_tokens: Annotated[Decimal, Field(gt=0)] = OUTPUT_RATE
    maximum_provider_calls: Literal[2] = 2
    request_order: list[
        Literal[
            "metasyn-row17-candidate3-binary-primary-endpoint",
            "metasyn-row17-candidate2-binary-symptom-endpoint",
        ]
    ]
    stop_after_first_fully_grounded_native_typed_graph: Literal[True] = True
    sdk_retries_per_request: Literal[0] = 0
    application_retries_per_request: Literal[0] = 0
    orphan_or_ambiguous_attempt_retry_permitted: Literal[False] = False
    operator_authorized_source_transmission: Literal[True] = True
    pricing_source_url: Literal["https://platform.claude.com/docs/en/about-claude/pricing"] = (
        PRICING_SOURCE_URL
    )
    model_source_url: Literal["https://platform.claude.com/docs/en/models/overview"] = (
        MODEL_SOURCE_URL
    )
    model_id_source_url: Literal[
        "https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions"
    ] = MODEL_ID_SOURCE_URL
    model_id_version_semantics: Literal["canonical_pinned_snapshot_not_evergreen_alias"] = (
        "canonical_pinned_snapshot_not_evergreen_alias"
    )
    structured_outputs_source_url: Literal[
        "https://platform.claude.com/docs/en/build-with-claude/structured-outputs"
    ] = STRUCTURED_OUTPUTS_SOURCE_URL
    token_liability_policy: Literal[
        "full_model_input_context_hard_ceiling_with_known_byte_diagnostic"
    ] = "full_model_input_context_hard_ceiling_with_known_byte_diagnostic"
    source_verified_date: Literal["2026-08-29"] = SOURCE_VERIFIED_DATE
    yield_only_no_accuracy_calibration_synthesis_or_release_authority: Literal[True] = True
    claim_release_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_config(self) -> MetaSynContextualFrontierConfigV1:
        if (
            self.request_order != list(REQUEST_ORDER)
            or self.input_rate_usd_per_million_tokens != INPUT_RATE
            or self.output_rate_usd_per_million_tokens != OUTPUT_RATE
        ):
            raise ValueError("contextual_frontier_request_order_mismatch")
        return self

    @property
    def config_sha256(self) -> str:
        return hash_canonical(self)


def load_metasyn_contextual_frontier_config_v1(
    *, repository_root: Path, config_path: Path = DEFAULT_CONFIG_PATH
) -> MetaSynContextualFrontierConfigV1:
    path = _safe_repository_file(repository_root, config_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynContextualFrontierRuntimeV1Error("contextual_frontier_config_invalid") from exc
    return MetaSynContextualFrontierConfigV1.model_validate(raw)


class MetaSynContextualFrontierProviderIdentityV1(_Frozen):
    identity_version: Literal["metasyn-contextual-frontier-provider-identity-v1"] = (
        "metasyn-contextual-frontier-provider-identity-v1"
    )
    provider: Literal["anthropic_first_party_api"] = "anthropic_first_party_api"
    model: Literal["claude-fable-5"] = MODEL
    model_id_version_semantics: Literal["canonical_pinned_snapshot_not_evergreen_alias"] = (
        "canonical_pinned_snapshot_not_evergreen_alias"
    )
    api_base_url: Literal["https://api.anthropic.com"] = API_BASE_URL
    anthropic_api_version: Literal["2023-06-01"] = ANTHROPIC_API_VERSION
    anthropic_sdk_version: Literal["0.120.2"] = SDK_VERSION
    api_operation: Literal["messages.create"] = "messages.create"
    effort: Literal["high"] = EFFORT
    service_tier: Literal["standard_only"] = SERVICE_TIER
    transport_mode: Literal["structured_json_schema"] = "structured_json_schema"
    output_config_format_present: Literal[True] = True
    sdk_max_retries: Literal[0] = 0
    application_retries: Literal[0] = 0
    requests_per_exact_intent: Literal[1] = 1
    http_environment_trust: Literal[False] = False
    follow_redirects: Literal[False] = False
    environment_base_url_override_permitted: Literal[False] = False
    environment_custom_headers_override_permitted: Literal[False] = False
    timeout_seconds: Literal[600.0] = TIMEOUT_SECONDS
    credential_source: Literal["environment_read_only_by_sdk_not_archived"] = (
        "environment_read_only_by_sdk_not_archived"
    )
    config_sha256: Sha256
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_identity(self) -> MetaSynContextualFrontierProviderIdentityV1:
        _self_hash(self, "identity_sha256", "contextual_frontier_identity_hash_mismatch")
        return self


def freeze_metasyn_contextual_frontier_identity_v1(
    config: MetaSynContextualFrontierConfigV1,
) -> MetaSynContextualFrontierProviderIdentityV1:
    _require_sdk()
    payload = {
        "identity_version": "metasyn-contextual-frontier-provider-identity-v1",
        "provider": "anthropic_first_party_api",
        "model": MODEL,
        "model_id_version_semantics": "canonical_pinned_snapshot_not_evergreen_alias",
        "api_base_url": API_BASE_URL,
        "anthropic_api_version": ANTHROPIC_API_VERSION,
        "anthropic_sdk_version": SDK_VERSION,
        "api_operation": "messages.create",
        "effort": EFFORT,
        "service_tier": SERVICE_TIER,
        "transport_mode": "structured_json_schema",
        "output_config_format_present": True,
        "sdk_max_retries": 0,
        "application_retries": 0,
        "requests_per_exact_intent": 1,
        "http_environment_trust": False,
        "follow_redirects": False,
        "environment_base_url_override_permitted": False,
        "environment_custom_headers_override_permitted": False,
        "timeout_seconds": TIMEOUT_SECONDS,
        "credential_source": "environment_read_only_by_sdk_not_archived",
        "config_sha256": config.config_sha256,
    }
    return MetaSynContextualFrontierProviderIdentityV1.model_validate(
        {**payload, "identity_sha256": hash_canonical(payload)}
    )


class MetaSynContextualFrontierCostCeilingV1(_Frozen):
    cost_version: Literal["metasyn-contextual-frontier-cost-ceiling-v1"] = (
        "metasyn-contextual-frontier-cost-ceiling-v1"
    )
    method: Literal["full_model_input_context_hard_ceiling_with_known_byte_diagnostic"] = (
        "full_model_input_context_hard_ceiling_with_known_byte_diagnostic"
    )
    system_utf8_bytes: TokenCount
    prompt_utf8_bytes: TokenCount
    wire_schema_utf8_bytes: TokenCount
    model_facing_utf8_bytes: TokenCount
    fixed_framing_tokens: Literal[2048] = FIXED_FRAMING_TOKENS
    diagnostic_known_input_token_ceiling: TokenCount
    model_max_input_tokens: Literal[1000000] = MAX_INPUT_TOKENS
    conservative_input_token_ceiling: Literal[1000000] = MAX_INPUT_TOKENS
    max_output_tokens: Literal[32000] = MAX_OUTPUT_TOKENS
    input_rate_usd_per_million_tokens: Annotated[Decimal, Field(gt=0)] = INPUT_RATE
    output_rate_usd_per_million_tokens: Annotated[Decimal, Field(gt=0)] = OUTPUT_RATE
    diagnostic_known_surface_cost_usd: Decimal
    diagnostic_known_surface_cost_usd_micros: Annotated[int, Field(ge=1)]
    request_cost_ceiling_usd: Decimal
    request_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    cost_sha256: Sha256

    @model_validator(mode="after")
    def validate_cost(self) -> MetaSynContextualFrontierCostCeilingV1:
        if self.model_facing_utf8_bytes != (
            self.system_utf8_bytes + self.prompt_utf8_bytes + self.wire_schema_utf8_bytes
        ):
            raise ValueError("contextual_frontier_model_facing_bytes_mismatch")
        if (
            self.input_rate_usd_per_million_tokens != INPUT_RATE
            or self.output_rate_usd_per_million_tokens != OUTPUT_RATE
        ):
            raise ValueError("contextual_frontier_cost_rates_not_pinned")
        diagnostic_input = self.model_facing_utf8_bytes + self.fixed_framing_tokens
        if (
            self.diagnostic_known_input_token_ceiling != diagnostic_input
            or diagnostic_input > self.model_max_input_tokens
            or self.conservative_input_token_ceiling != self.model_max_input_tokens
        ):
            raise ValueError("contextual_frontier_input_ceiling_mismatch")
        expected_cost = (
            Decimal(self.model_max_input_tokens) * INPUT_RATE
            + Decimal(self.max_output_tokens) * OUTPUT_RATE
        ) / Decimal(1_000_000)
        expected_diagnostic_cost = (
            Decimal(diagnostic_input) * INPUT_RATE + Decimal(self.max_output_tokens) * OUTPUT_RATE
        ) / Decimal(1_000_000)
        if (
            self.diagnostic_known_surface_cost_usd != expected_diagnostic_cost
            or self.diagnostic_known_surface_cost_usd_micros
            != _usd_micros(expected_diagnostic_cost)
            or self.request_cost_ceiling_usd != expected_cost
            or self.request_cost_ceiling_usd_micros != _usd_micros(expected_cost)
        ):
            raise ValueError("contextual_frontier_cost_ceiling_mismatch")
        _self_hash(self, "cost_sha256", "contextual_frontier_cost_hash_mismatch")
        return self


def _freeze_cost(
    *, model_system: str, prompt: str, wire_schema: Mapping[str, Any]
) -> MetaSynContextualFrontierCostCeilingV1:
    system_bytes = len(model_system.encode("utf-8"))
    prompt_bytes = len(prompt.encode("utf-8"))
    schema_bytes = len(canonical_json_bytes(wire_schema))
    # Structured outputs inject the exact grammar schema into the model-facing
    # request. Count every schema byte independently rather than hiding it in a
    # token estimate.
    model_facing = system_bytes + prompt_bytes + schema_bytes
    diagnostic_input_ceiling = model_facing + FIXED_FRAMING_TOKENS
    if diagnostic_input_ceiling > MAX_INPUT_TOKENS:
        raise MetaSynContextualFrontierRuntimeV1Error(
            "contextual_frontier_known_input_exceeds_model_context"
        )
    cost = (
        Decimal(MAX_INPUT_TOKENS) * INPUT_RATE + Decimal(MAX_OUTPUT_TOKENS) * OUTPUT_RATE
    ) / Decimal(1_000_000)
    diagnostic_cost = (
        Decimal(diagnostic_input_ceiling) * INPUT_RATE + Decimal(MAX_OUTPUT_TOKENS) * OUTPUT_RATE
    ) / Decimal(1_000_000)
    payload = {
        "cost_version": "metasyn-contextual-frontier-cost-ceiling-v1",
        "method": ("full_model_input_context_hard_ceiling_with_known_byte_diagnostic"),
        "system_utf8_bytes": system_bytes,
        "prompt_utf8_bytes": prompt_bytes,
        "wire_schema_utf8_bytes": schema_bytes,
        "model_facing_utf8_bytes": model_facing,
        "fixed_framing_tokens": FIXED_FRAMING_TOKENS,
        "diagnostic_known_input_token_ceiling": diagnostic_input_ceiling,
        "model_max_input_tokens": MAX_INPUT_TOKENS,
        "conservative_input_token_ceiling": MAX_INPUT_TOKENS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "input_rate_usd_per_million_tokens": INPUT_RATE,
        "output_rate_usd_per_million_tokens": OUTPUT_RATE,
        "diagnostic_known_surface_cost_usd": diagnostic_cost,
        "diagnostic_known_surface_cost_usd_micros": _usd_micros(diagnostic_cost),
        "request_cost_ceiling_usd": cost,
        "request_cost_ceiling_usd_micros": _usd_micros(cost),
    }
    return MetaSynContextualFrontierCostCeilingV1.model_validate(
        {**payload, "cost_sha256": hash_canonical(_contract_json(payload))}
    )


class MetaSynContextualFrontierRequestV1(_Frozen):
    request_version: Literal["metasyn-contextual-frontier-request-v1"] = REQUEST_VERSION
    operation: Literal["messages.create.contextual_grounding_v3"] = (
        "messages.create.contextual_grounding_v3"
    )
    request_key: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{2,127}$")]
    witness_id: Literal[
        "metasyn-row17-candidate3-binary-primary-endpoint",
        "metasyn-row17-candidate2-binary-symptom-endpoint",
    ]
    provider_binding_sha256: Sha256
    model: Literal["claude-fable-5"] = MODEL
    effort: Literal["high"] = EFFORT
    service_tier: Literal["standard_only"] = SERVICE_TIER
    transport_mode: Literal["structured_json_schema"] = "structured_json_schema"
    output_config_format_present: Literal[True] = True
    structured_output_model_supported: Literal[True] = True
    structured_output_schema_sdk_transformed: Literal[True] = True
    structured_outputs_source_url: Literal[
        "https://platform.claude.com/docs/en/build-with-claude/structured-outputs"
    ] = STRUCTURED_OUTPUTS_SOURCE_URL
    config_sha256: Sha256
    identity_sha256: Sha256
    original_schema_sha256: Sha256
    compiled_schema: AnthropicCompiledSchemaV1
    compiled_schema_sha256: Sha256
    wire_schema_sha256: Sha256
    base_system: NonEmpty
    base_system_sha256: Sha256
    model_system: NonEmpty
    model_system_sha256: Sha256
    prompt: NonEmpty
    prompt_sha256: Sha256
    max_output_tokens: Literal[32000] = MAX_OUTPUT_TOKENS
    wire_kwargs: dict[str, Any]
    wire_kwargs_sha256: Sha256
    wire_call_surface_sha256: Sha256
    cost_ceiling: MetaSynContextualFrontierCostCeilingV1
    request_sha256: Sha256

    @model_validator(mode="after")
    def validate_request(self) -> MetaSynContextualFrontierRequestV1:
        if (
            self.compiled_schema_sha256 != self.compiled_schema.compiled_schema_sha256
            or self.original_schema_sha256 != self.compiled_schema.original_schema_sha256
            or self.wire_schema_sha256 != self.compiled_schema.wire_schema_sha256
        ):
            raise ValueError("contextual_frontier_schema_alias_mismatch")
        expected_system = self.base_system
        expected_kwargs = _wire_kwargs(
            model_system=expected_system,
            prompt=self.prompt,
            wire_schema=self.compiled_schema.wire_schema,
        )
        if (
            self.base_system_sha256 != _sha256_utf8(self.base_system)
            or self.model_system != expected_system
            or self.model_system_sha256 != _sha256_utf8(expected_system)
            or self.prompt_sha256 != _sha256_utf8(self.prompt)
            or self.wire_kwargs != expected_kwargs
            or self.wire_kwargs_sha256 != hash_canonical(expected_kwargs)
        ):
            raise ValueError("contextual_frontier_wire_surface_mismatch")
        expected_surface = hash_canonical(
            {
                "api_base_url": API_BASE_URL,
                "anthropic_api_version": ANTHROPIC_API_VERSION,
                "http_environment_trust": False,
                "follow_redirects": False,
                "sdk_max_retries": 0,
                "application_retries": 0,
                "wire_kwargs": expected_kwargs,
            }
        )
        if self.wire_call_surface_sha256 != expected_surface:
            raise ValueError("contextual_frontier_wire_call_surface_hash_mismatch")
        expected_cost = _freeze_cost(
            model_system=expected_system,
            prompt=self.prompt,
            wire_schema=self.compiled_schema.wire_schema,
        )
        if self.cost_ceiling != expected_cost:
            raise ValueError("contextual_frontier_request_cost_replay_mismatch")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(self, "request_sha256", "contextual_frontier_request_hash_mismatch")
        return self


def _wire_kwargs(
    *, model_system: str, prompt: str, wire_schema: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "model": MODEL,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": model_system,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {
            "effort": EFFORT,
            "format": {
                "type": "json_schema",
                "schema": json.loads(canonical_json_bytes(wire_schema)),
            },
        },
        "service_tier": SERVICE_TIER,
    }


def freeze_metasyn_contextual_frontier_request_v1(
    *,
    witness_id: Literal[
        "metasyn-row17-candidate3-binary-primary-endpoint",
        "metasyn-row17-candidate2-binary-symptom-endpoint",
    ],
    provider_binding: ContextualProviderBindingV3,
    config: MetaSynContextualFrontierConfigV1,
    identity: MetaSynContextualFrontierProviderIdentityV1,
) -> MetaSynContextualFrontierRequestV1:
    if identity.config_sha256 != config.config_sha256:
        raise MetaSynContextualFrontierRuntimeV1Error("contextual_frontier_identity_config_drift")
    compiled = compile_anthropic_bounded_schema(
        original_schema=provider_binding.provider_schema,
        full_acceptance_schema_sha256=provider_binding.provider_schema_sha256,
    )
    model_system = SYSTEM_PROMPT
    prompt = provider_binding.rendered_prompt
    kwargs = _wire_kwargs(
        model_system=model_system,
        prompt=prompt,
        wire_schema=compiled.wire_schema,
    )
    surface_sha = hash_canonical(
        {
            "api_base_url": API_BASE_URL,
            "anthropic_api_version": ANTHROPIC_API_VERSION,
            "http_environment_trust": False,
            "follow_redirects": False,
            "sdk_max_retries": 0,
            "application_retries": 0,
            "wire_kwargs": kwargs,
        }
    )
    cost = _freeze_cost(
        model_system=model_system,
        prompt=prompt,
        wire_schema=compiled.wire_schema,
    )
    candidate_index = provider_binding.context.candidate.candidate_index
    payload = {
        "request_version": REQUEST_VERSION,
        "operation": "messages.create.contextual_grounding_v3",
        "request_key": f"row17-candidate{candidate_index}-fable5-high",
        "witness_id": witness_id,
        "provider_binding_sha256": provider_binding.binding_sha256,
        "model": MODEL,
        "effort": EFFORT,
        "service_tier": SERVICE_TIER,
        "transport_mode": "structured_json_schema",
        "output_config_format_present": True,
        "structured_output_model_supported": True,
        "structured_output_schema_sdk_transformed": True,
        "structured_outputs_source_url": STRUCTURED_OUTPUTS_SOURCE_URL,
        "config_sha256": config.config_sha256,
        "identity_sha256": identity.identity_sha256,
        "original_schema_sha256": compiled.original_schema_sha256,
        "compiled_schema": compiled,
        "compiled_schema_sha256": compiled.compiled_schema_sha256,
        "wire_schema_sha256": compiled.wire_schema_sha256,
        "base_system": SYSTEM_PROMPT,
        "base_system_sha256": _sha256_utf8(SYSTEM_PROMPT),
        "model_system": model_system,
        "model_system_sha256": _sha256_utf8(model_system),
        "prompt": prompt,
        "prompt_sha256": _sha256_utf8(prompt),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "wire_kwargs": kwargs,
        "wire_kwargs_sha256": hash_canonical(kwargs),
        "wire_call_surface_sha256": surface_sha,
        "cost_ceiling": cost,
    }
    return MetaSynContextualFrontierRequestV1.model_validate(
        {**payload, "request_sha256": hash_canonical(payload)}
    )


ProviderOutcome = Literal[
    "completed",
    "response_model_mismatch",
    "response_identity_invalid",
    "response_stop_reason_invalid",
    "response_content_invalid",
    "response_usage_invalid",
    "response_json_invalid",
    "response_schema_invalid",
]


class MetaSynContextualFrontierUsageV1(_Frozen):
    input_tokens: TokenCount
    output_tokens: TokenCount
    cache_creation_input_tokens: Literal[0] = 0
    cache_read_input_tokens: Literal[0] = 0

    @field_validator("input_tokens", "output_tokens", mode="before")
    @classmethod
    def reject_bool(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("contextual_frontier_usage_boolean_forbidden")
        return value


class MetaSynContextualFrontierProviderResultV1(_Frozen):
    result_version: Literal["metasyn-contextual-frontier-provider-result-v1"] = (
        PROVIDER_RESULT_VERSION
    )
    request_sha256: Sha256
    identity_sha256: Sha256
    config_sha256: Sha256
    wire_call_surface_sha256: Sha256
    original_schema_sha256: Sha256
    wire_schema_sha256: Sha256
    model_system_sha256: Sha256
    prompt_sha256: Sha256
    transport_attempt_count: Literal[1] = 1
    sdk_retry_count: Literal[0] = 0
    outcome: ProviderOutcome
    response_id: str | None
    response_model: str | None
    stop_reason: str | None
    text: str | None
    text_sha256: Sha256 | None
    parsed_json: dict[str, Any] | None
    parsed_json_sha256: Sha256 | None
    usage: MetaSynContextualFrontierUsageV1 | None
    estimated_cost_usd: Decimal | None
    charged_cost_upper_bound_usd: Decimal
    failure_code: ProviderOutcome | None
    failure_exception_type: str | None = None
    failure_http_status: int | None = None
    failure_provider_request_id: str | None = None
    credential_archived: Literal[False] = False
    result_sha256: Sha256

    @model_validator(mode="after")
    def validate_result(self) -> MetaSynContextualFrontierProviderResultV1:
        if self.outcome == "completed":
            if (
                not self.response_id
                or self.response_model != MODEL
                or self.stop_reason != "end_turn"
                or self.text is None
                or self.parsed_json is None
                or self.usage is None
                or self.estimated_cost_usd is None
                or self.failure_code is not None
            ):
                raise ValueError("contextual_frontier_completed_result_shape_invalid")
        elif self.failure_code != self.outcome:
            raise ValueError("contextual_frontier_failed_result_code_mismatch")
        if (self.text is None) != (self.text_sha256 is None):
            raise ValueError("contextual_frontier_text_hash_presence_mismatch")
        if self.text is not None and self.text_sha256 != _sha256_utf8(self.text):
            raise ValueError("contextual_frontier_text_hash_mismatch")
        if (self.parsed_json is None) != (self.parsed_json_sha256 is None):
            raise ValueError("contextual_frontier_json_hash_presence_mismatch")
        if self.parsed_json is not None and self.parsed_json_sha256 != hash_canonical(
            self.parsed_json
        ):
            raise ValueError("contextual_frontier_json_hash_mismatch")
        if self.usage is not None:
            expected = (
                Decimal(self.usage.input_tokens) * INPUT_RATE
                + Decimal(self.usage.output_tokens) * OUTPUT_RATE
            ) / Decimal(1_000_000)
            if self.estimated_cost_usd != expected:
                raise ValueError("contextual_frontier_reported_cost_mismatch")
        if self.failure_exception_type is not None and not _SAFE_IDENTIFIER_RE.fullmatch(
            self.failure_exception_type
        ):
            raise ValueError("contextual_frontier_exception_type_unsafe")
        if self.failure_provider_request_id is not None and not _SAFE_IDENTIFIER_RE.fullmatch(
            self.failure_provider_request_id
        ):
            raise ValueError("contextual_frontier_provider_request_id_unsafe")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(self, "result_sha256", "contextual_frontier_result_hash_mismatch")
        return self


def freeze_metasyn_contextual_frontier_provider_result_v1(
    *,
    request: MetaSynContextualFrontierRequestV1,
    outcome: ProviderOutcome,
    response_id: str | None,
    response_model: str | None,
    stop_reason: str | None,
    text: str | None,
    parsed_json: Mapping[str, Any] | None,
    usage: MetaSynContextualFrontierUsageV1 | None,
    failure_code: ProviderOutcome | None = None,
    failure_exception_type: str | None = None,
    failure_http_status: int | None = None,
    failure_provider_request_id: str | None = None,
) -> MetaSynContextualFrontierProviderResultV1:
    parsed = dict(parsed_json) if parsed_json is not None else None
    estimated = None
    if usage is not None:
        if (
            usage.input_tokens > request.cost_ceiling.conservative_input_token_ceiling
            or usage.output_tokens > request.max_output_tokens
        ):
            raise MetaSynContextualFrontierRuntimeV1Error(
                "contextual_frontier_usage_exceeds_frozen_ceiling"
            )
        estimated = (
            Decimal(usage.input_tokens) * INPUT_RATE + Decimal(usage.output_tokens) * OUTPUT_RATE
        ) / Decimal(1_000_000)
    payload = {
        "result_version": PROVIDER_RESULT_VERSION,
        "request_sha256": request.request_sha256,
        "identity_sha256": request.identity_sha256,
        "config_sha256": request.config_sha256,
        "wire_call_surface_sha256": request.wire_call_surface_sha256,
        "original_schema_sha256": request.original_schema_sha256,
        "wire_schema_sha256": request.wire_schema_sha256,
        "model_system_sha256": request.model_system_sha256,
        "prompt_sha256": request.prompt_sha256,
        "transport_attempt_count": 1,
        "sdk_retry_count": 0,
        "outcome": outcome,
        "response_id": response_id,
        "response_model": response_model,
        "stop_reason": stop_reason,
        "text": text,
        "text_sha256": _sha256_utf8(text) if text is not None else None,
        "parsed_json": parsed,
        "parsed_json_sha256": hash_canonical(parsed) if parsed is not None else None,
        "usage": usage,
        "estimated_cost_usd": estimated,
        "charged_cost_upper_bound_usd": request.cost_ceiling.request_cost_ceiling_usd,
        "failure_code": failure_code,
        "failure_exception_type": failure_exception_type,
        "failure_http_status": failure_http_status,
        "failure_provider_request_id": failure_provider_request_id,
        "credential_archived": False,
    }
    return MetaSynContextualFrontierProviderResultV1.model_validate(
        {**payload, "result_sha256": hash_canonical(_contract_json(payload))}
    )


def _safe_provider_scalar(value: Any) -> str | None:
    if type(value) is not str or not _SAFE_IDENTIFIER_RE.fullmatch(value):
        return None
    return value


def _strict_json_object(text: str) -> dict[str, Any]:
    def reject_constant(_: str) -> Any:
        raise ValueError("nonfinite_json_forbidden")

    value = json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=lambda pairs: _unique_object(pairs),
    )
    if not isinstance(value, dict):
        raise ValueError("json_root_not_object")
    _assert_secret_free(value)
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


class MetaSynContextualFrontierClientV1:
    """One transport attempt, zero SDK retries, and no credential persistence."""

    def __init__(self, config: MetaSynContextualFrontierConfigV1) -> None:
        self.config = config
        self.identity = freeze_metasyn_contextual_frontier_identity_v1(config)
        self._client: Any | None = None

    def _client_or_create(self) -> Any:
        if self._client is None:
            if os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("ANTHROPIC_CUSTOM_HEADERS"):
                raise MetaSynContextualFrontierRuntimeV1Error(
                    "contextual_frontier_transport_environment_override_forbidden"
                )
            anthropic = _require_sdk()
            http_client = anthropic.DefaultHttpxClient(
                timeout=TIMEOUT_SECONDS,
                trust_env=False,
                follow_redirects=False,
            )
            self._client = anthropic.Anthropic(
                base_url=API_BASE_URL,
                default_headers={"anthropic-version": ANTHROPIC_API_VERSION},
                http_client=http_client,
                max_retries=0,
                timeout=TIMEOUT_SECONDS,
            )
        return self._client

    def generate(
        self, request: MetaSynContextualFrontierRequestV1
    ) -> MetaSynContextualFrontierProviderResultV1:
        canonical = MetaSynContextualFrontierRequestV1.model_validate(
            request.model_dump(mode="json")
        )
        if (
            canonical.config_sha256 != self.config.config_sha256
            or canonical.identity_sha256 != self.identity.identity_sha256
        ):
            raise MetaSynContextualFrontierRuntimeV1Error(
                "contextual_frontier_client_request_identity_mismatch"
            )
        # Transport exceptions intentionally cross this boundary.  The durable
        # executor records them as ambiguous after-intent incidents and never retries.
        response = self._client_or_create().messages.create(**canonical.wire_kwargs)
        response_id = _safe_provider_scalar(getattr(response, "id", None))
        response_model = _safe_provider_scalar(getattr(response, "model", None))
        stop_reason = _safe_provider_scalar(getattr(response, "stop_reason", None))
        try:
            usage_obj = response.usage
            usage = MetaSynContextualFrontierUsageV1(
                input_tokens=usage_obj.input_tokens,
                output_tokens=usage_obj.output_tokens,
                cache_creation_input_tokens=getattr(usage_obj, "cache_creation_input_tokens", 0)
                or 0,
                cache_read_input_tokens=getattr(usage_obj, "cache_read_input_tokens", 0) or 0,
            )
            if (
                usage.input_tokens > canonical.cost_ceiling.conservative_input_token_ceiling
                or usage.output_tokens > canonical.max_output_tokens
            ):
                raise ValueError("usage_outside_bound")
        except Exception:
            return freeze_metasyn_contextual_frontier_provider_result_v1(
                request=canonical,
                outcome="response_usage_invalid",
                response_id=response_id,
                response_model=response_model,
                stop_reason=stop_reason,
                text=None,
                parsed_json=None,
                usage=None,
                failure_code="response_usage_invalid",
            )
        if response_id is None:
            return freeze_metasyn_contextual_frontier_provider_result_v1(
                request=canonical,
                outcome="response_identity_invalid",
                response_id=None,
                response_model=response_model,
                stop_reason=stop_reason,
                text=None,
                parsed_json=None,
                usage=usage,
                failure_code="response_identity_invalid",
            )
        if response_model != MODEL:
            return freeze_metasyn_contextual_frontier_provider_result_v1(
                request=canonical,
                outcome="response_model_mismatch",
                response_id=response_id,
                response_model=response_model,
                stop_reason=stop_reason,
                text=None,
                parsed_json=None,
                usage=usage,
                failure_code="response_model_mismatch",
            )
        if stop_reason != "end_turn":
            return freeze_metasyn_contextual_frontier_provider_result_v1(
                request=canonical,
                outcome="response_stop_reason_invalid",
                response_id=response_id,
                response_model=response_model,
                stop_reason=stop_reason,
                text=None,
                parsed_json=None,
                usage=usage,
                failure_code="response_stop_reason_invalid",
            )
        try:
            blocks = list(response.content or ())
            text_blocks = [block.text for block in blocks if block.type == "text"]
            if (
                len(text_blocks) != 1
                or type(text_blocks[0]) is not str
                or not text_blocks[0]
                or any(
                    block.type not in {"text", "thinking", "redacted_thinking"} for block in blocks
                )
            ):
                raise ValueError("content_shape")
            text = text_blocks[0]
        except Exception:
            return freeze_metasyn_contextual_frontier_provider_result_v1(
                request=canonical,
                outcome="response_content_invalid",
                response_id=response_id,
                response_model=response_model,
                stop_reason=stop_reason,
                text=None,
                parsed_json=None,
                usage=usage,
                failure_code="response_content_invalid",
            )
        try:
            parsed = _strict_json_object(text)
        except Exception:
            return freeze_metasyn_contextual_frontier_provider_result_v1(
                request=canonical,
                outcome="response_json_invalid",
                response_id=response_id,
                response_model=response_model,
                stop_reason=stop_reason,
                text=None,
                parsed_json=None,
                usage=usage,
                failure_code="response_json_invalid",
            )
        try:
            validator_for(canonical.compiled_schema.wire_schema)(
                canonical.compiled_schema.wire_schema
            ).validate(parsed)
            validator_for(canonical.compiled_schema.original_schema)(
                canonical.compiled_schema.original_schema
            ).validate(parsed)
        except Exception:
            return freeze_metasyn_contextual_frontier_provider_result_v1(
                request=canonical,
                outcome="response_schema_invalid",
                response_id=response_id,
                response_model=response_model,
                stop_reason=stop_reason,
                text=text,
                parsed_json=parsed,
                usage=usage,
                failure_code="response_schema_invalid",
            )
        return freeze_metasyn_contextual_frontier_provider_result_v1(
            request=canonical,
            outcome="completed",
            response_id=response_id,
            response_model=response_model,
            stop_reason=stop_reason,
            text=text,
            parsed_json=parsed,
            usage=usage,
        )


class MetaSynContextualFrontierClientProtocol(Protocol):
    def generate(
        self, request: MetaSynContextualFrontierRequestV1
    ) -> MetaSynContextualFrontierProviderResultV1: ...


class MetaSynContextualFrontierRosterItemV1(_Frozen):
    order: Annotated[int, Field(ge=0, le=1)]
    witness_id: Literal[
        "metasyn-row17-candidate3-binary-primary-endpoint",
        "metasyn-row17-candidate2-binary-symptom-endpoint",
    ]
    offline_witness: ContextualGroundingFeasibilityReceiptV3
    offline_witness_sha256: Sha256
    provider_binding_sha256: Sha256
    request: MetaSynContextualFrontierRequestV1
    request_sha256: Sha256
    request_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    item_sha256: Sha256

    @model_validator(mode="after")
    def validate_item(self) -> MetaSynContextualFrontierRosterItemV1:
        if (
            self.witness_id != REQUEST_ORDER[self.order]
            or self.offline_witness.witness_id != self.witness_id
            or self.offline_witness_sha256 != self.offline_witness.receipt_sha256
            or self.provider_binding_sha256 != self.offline_witness.provider_binding.binding_sha256
            or self.request.witness_id != self.witness_id
            or self.request.provider_binding_sha256 != self.provider_binding_sha256
            or self.request_sha256 != self.request.request_sha256
            or self.request_cost_ceiling_usd_micros
            != self.request.cost_ceiling.request_cost_ceiling_usd_micros
        ):
            raise ValueError("contextual_frontier_roster_item_alias_mismatch")
        _self_hash(self, "item_sha256", "contextual_frontier_roster_item_hash_mismatch")
        return self


class MetaSynContextualFrontierPlanV1(_Frozen):
    plan_version: Literal["metasyn-contextual-frontier-plan-v1"] = PLAN_VERSION
    runtime_version: Literal["metasyn-contextual-frontier-runtime-v1"] = RUNTIME_VERSION
    status: Literal["offline_prepared_zero_provider_calls"] = "offline_prepared_zero_provider_calls"
    config: MetaSynContextualFrontierConfigV1
    config_sha256: Sha256
    provider_identity: MetaSynContextualFrontierProviderIdentityV1
    provider_identity_sha256: Sha256
    contextual_suite_sha256: Sha256
    contextual_pipeline_sha256: Sha256
    runtime_pipeline_components: dict[str, str]
    runtime_pipeline_sha256: Sha256
    roster: Annotated[
        list[MetaSynContextualFrontierRosterItemV1], Field(min_length=2, max_length=2)
    ]
    request_roster_sha256: Sha256
    request_count: Literal[2] = 2
    maximum_provider_calls: Literal[2] = 2
    total_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    diagnostic_known_input_token_ceiling_total: Annotated[int, Field(ge=1)]
    diagnostic_known_surface_cost_usd_micros_total: Annotated[int, Field(ge=1)]
    max_output_tokens_rationale: Literal[
        "bounded_32k_allows_high_effort_thinking_plus_full_15_claim_json_without_xhigh_or_max"
    ] = "bounded_32k_allows_high_effort_thinking_plus_full_15_claim_json_without_xhigh_or_max"
    stop_after_first_fully_grounded_native_typed_graph: Literal[True] = True
    fallback_attempted_only_after_primary_non_success: Literal[True] = True
    provider_calls_made: Literal[0] = 0
    credential_opened_or_archived: Literal[False] = False
    official_test_labels_opened: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    plan_sha256: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> MetaSynContextualFrontierPlanV1:
        if (
            self.config_sha256 != self.config.config_sha256
            or self.provider_identity.config_sha256 != self.config_sha256
            or self.provider_identity_sha256 != self.provider_identity.identity_sha256
            or self.runtime_pipeline_sha256 != hash_canonical(self.runtime_pipeline_components)
            or [item.order for item in self.roster] != [0, 1]
            or [item.witness_id for item in self.roster] != list(REQUEST_ORDER)
            or len({item.request_sha256 for item in self.roster}) != 2
            or self.request_roster_sha256
            != hash_canonical([item.item_sha256 for item in self.roster])
            or self.total_cost_ceiling_usd_micros
            != sum(item.request_cost_ceiling_usd_micros for item in self.roster)
            or self.diagnostic_known_input_token_ceiling_total
            != sum(
                item.request.cost_ceiling.diagnostic_known_input_token_ceiling
                for item in self.roster
            )
            or self.diagnostic_known_surface_cost_usd_micros_total
            != sum(
                item.request.cost_ceiling.diagnostic_known_surface_cost_usd_micros
                for item in self.roster
            )
        ):
            raise ValueError("contextual_frontier_plan_alias_mismatch")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(self, "plan_sha256", "contextual_frontier_plan_hash_mismatch")
        return self


def freeze_metasyn_contextual_frontier_plan_v1(
    *, repository_root: Path, config_path: Path = DEFAULT_CONFIG_PATH
) -> MetaSynContextualFrontierPlanV1:
    root = _canonical_repository_root(repository_root)
    config = load_metasyn_contextual_frontier_config_v1(
        repository_root=root, config_path=config_path
    )
    identity = freeze_metasyn_contextual_frontier_identity_v1(config)
    suite = freeze_contextual_grounding_offline_feasibility_suite_v3(repository_root=root)
    pipeline_components = {
        "runtime_source_sha256": sha256_file(_safe_repository_file(root, RUNTIME_SOURCE_PATH)),
        "contextual_source_sha256": sha256_file(
            _safe_repository_file(root, CONTEXTUAL_SOURCE_PATH)
        ),
        "compiler_source_sha256": sha256_file(_safe_repository_file(root, COMPILER_SOURCE_PATH)),
        "config_file_sha256": sha256_file(_safe_repository_file(root, config_path)),
        "config_sha256": config.config_sha256,
        "contextual_pipeline_sha256": suite.pipeline_sha256,
        "anthropic_sdk_version": SDK_VERSION,
        "model": MODEL,
        "effort": EFFORT,
    }
    by_id = {receipt.witness_id: receipt for receipt in suite.receipts}
    roster: list[MetaSynContextualFrontierRosterItemV1] = []
    for order, witness_id in enumerate(REQUEST_ORDER):
        receipt = by_id.get(witness_id)
        if receipt is None:
            raise MetaSynContextualFrontierRuntimeV1Error(
                "contextual_frontier_required_witness_missing"
            )
        request = freeze_metasyn_contextual_frontier_request_v1(
            witness_id=witness_id,  # type: ignore[arg-type]
            provider_binding=receipt.provider_binding,
            config=config,
            identity=identity,
        )
        item_payload = {
            "order": order,
            "witness_id": witness_id,
            "offline_witness": receipt,
            "offline_witness_sha256": receipt.receipt_sha256,
            "provider_binding_sha256": receipt.provider_binding.binding_sha256,
            "request": request,
            "request_sha256": request.request_sha256,
            "request_cost_ceiling_usd_micros": (
                request.cost_ceiling.request_cost_ceiling_usd_micros
            ),
        }
        roster.append(
            MetaSynContextualFrontierRosterItemV1.model_validate(
                {**item_payload, "item_sha256": hash_canonical(item_payload)}
            )
        )
    payload = {
        "plan_version": PLAN_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "status": "offline_prepared_zero_provider_calls",
        "config": config,
        "config_sha256": config.config_sha256,
        "provider_identity": identity,
        "provider_identity_sha256": identity.identity_sha256,
        "contextual_suite_sha256": suite.suite_sha256,
        "contextual_pipeline_sha256": suite.pipeline_sha256,
        "runtime_pipeline_components": pipeline_components,
        "runtime_pipeline_sha256": hash_canonical(pipeline_components),
        "roster": roster,
        "request_roster_sha256": hash_canonical([item.item_sha256 for item in roster]),
        "request_count": 2,
        "maximum_provider_calls": 2,
        "total_cost_ceiling_usd_micros": sum(
            item.request_cost_ceiling_usd_micros for item in roster
        ),
        "diagnostic_known_input_token_ceiling_total": sum(
            item.request.cost_ceiling.diagnostic_known_input_token_ceiling for item in roster
        ),
        "diagnostic_known_surface_cost_usd_micros_total": sum(
            item.request.cost_ceiling.diagnostic_known_surface_cost_usd_micros for item in roster
        ),
        "max_output_tokens_rationale": (
            "bounded_32k_allows_high_effort_thinking_plus_full_15_claim_json_without_xhigh_or_max"
        ),
        "stop_after_first_fully_grounded_native_typed_graph": True,
        "fallback_attempted_only_after_primary_non_success": True,
        "provider_calls_made": 0,
        "credential_opened_or_archived": False,
        "official_test_labels_opened": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_effectiveness_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierPlanV1.model_validate(
        {**payload, "plan_sha256": hash_canonical(payload)}
    )


def _canonical_repository_root(value: Path) -> Path:
    try:
        root = Path(os.path.abspath(value)).resolve(strict=True)
    except OSError as exc:
        raise MetaSynContextualFrontierRuntimeV1Error(
            "contextual_frontier_repository_root_missing"
        ) from exc
    if not root.is_dir() or root.is_symlink():
        raise MetaSynContextualFrontierRuntimeV1Error("contextual_frontier_repository_root_unsafe")
    return root


def _safe_repository_file(root_value: Path, relative: Path) -> Path:
    root = _canonical_repository_root(root_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise MetaSynContextualFrontierRuntimeV1Error("contextual_frontier_path_outside_root")
    candidate = root / relative
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except OSError as exc:
            raise MetaSynContextualFrontierRuntimeV1Error(
                "contextual_frontier_repository_file_missing"
            ) from exc
        if stat.S_ISLNK(mode):
            raise MetaSynContextualFrontierRuntimeV1Error(
                "contextual_frontier_repository_symlink_forbidden"
            )
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise MetaSynContextualFrontierRuntimeV1Error("contextual_frontier_repository_file_unsafe")
    return resolved


def _fresh_workspace(value: Path) -> Path:
    path = Path(os.path.abspath(value))
    if path.exists() or path.is_symlink():
        raise MetaSynContextualFrontierRuntimeV1Error("contextual_frontier_workspace_must_be_fresh")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise MetaSynContextualFrontierRuntimeV1Error(
            "contextual_frontier_workspace_create_race"
        ) from exc
    return _existing_workspace(path)


def _existing_workspace(value: Path) -> Path:
    path = Path(os.path.abspath(value))
    if path.is_symlink():
        raise MetaSynContextualFrontierRuntimeV1Error("contextual_frontier_workspace_symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise MetaSynContextualFrontierRuntimeV1Error(
            "contextual_frontier_workspace_missing"
        ) from exc
    if not resolved.is_dir():
        raise MetaSynContextualFrontierRuntimeV1Error("contextual_frontier_workspace_not_directory")
    return resolved


@contextmanager
def _workspace_lock(workspace: Path) -> Iterator[None]:
    lock_path = workspace / ".metasyn-contextual-frontier-runtime-v1.lock"
    if lock_path.is_symlink():
        raise MetaSynContextualFrontierRuntimeV1Error("contextual_frontier_lock_symlink")
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


def _checked_artifact(workspace: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise MetaSynContextualFrontierRuntimeV1Error("contextual_frontier_artifact_path_invalid")
    cursor = workspace
    for part in relative.parts[:-1]:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise MetaSynContextualFrontierRuntimeV1Error(
                "contextual_frontier_artifact_parent_symlink"
            )
    path = workspace / relative
    if path.is_symlink():
        raise MetaSynContextualFrontierRuntimeV1Error("contextual_frontier_artifact_symlink")
    return path


def _load_object(path: Path, *, code: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MetaSynContextualFrontierRuntimeV1Error(code)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynContextualFrontierRuntimeV1Error(code) from exc
    if not isinstance(value, dict):
        raise MetaSynContextualFrontierRuntimeV1Error(code)
    _assert_secret_free(value)
    return value


def _persist_json(path: Path, value: Any) -> None:
    """Write one credential-free artifact with private directory permissions."""

    parent = path.parent
    missing: list[Path] = []
    cursor = parent
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if cursor.is_symlink():
        raise MetaSynContextualFrontierRuntimeV1Error("contextual_frontier_artifact_parent_symlink")
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
    for directory in [path.parent, *path.parents]:
        if directory == directory.parent:
            break
        if directory.exists() and directory.is_symlink():
            raise MetaSynContextualFrontierRuntimeV1Error(
                "contextual_frontier_artifact_parent_symlink"
            )
    secret_surface = value.model_dump(mode="json") if isinstance(value, ContractModel) else value
    _assert_secret_free(secret_surface)
    _atomic_write_json(path, value)
    os.chmod(path, 0o600)


def prepare_metasyn_contextual_frontier_runtime_v1(
    *,
    repository_root: Path,
    workspace: Path = DEFAULT_WORKSPACE,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> MetaSynContextualFrontierPlanV1:
    plan = freeze_metasyn_contextual_frontier_plan_v1(
        repository_root=repository_root, config_path=config_path
    )
    root = _fresh_workspace(workspace)
    with _workspace_lock(root):
        _persist_json(_checked_artifact(root, Path("00-prepared.json")), plan)
    return plan


def load_metasyn_contextual_frontier_plan_v1(*, workspace: Path) -> MetaSynContextualFrontierPlanV1:
    root = _existing_workspace(workspace)
    return MetaSynContextualFrontierPlanV1.model_validate(
        _load_object(
            _checked_artifact(root, Path("00-prepared.json")),
            code="contextual_frontier_prepared_plan_invalid",
        )
    )


class MetaSynContextualFrontierAuthorizedCallV1(_Frozen):
    order: Annotated[int, Field(ge=0, le=1)]
    request_key: str
    request_sha256: Sha256
    provider_binding_sha256: Sha256
    request_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]


class MetaSynContextualFrontierAuthorizationV1(_Frozen):
    authorization_version: Literal["metasyn-contextual-frontier-authorization-v1"] = (
        AUTHORIZATION_VERSION
    )
    plan_sha256: Sha256
    authorized_calls: Annotated[
        list[MetaSynContextualFrontierAuthorizedCallV1], Field(min_length=2, max_length=2)
    ]
    authorized_call_count: Literal[2] = 2
    authorized_roster_sha256: Sha256
    maximum_provider_attempts: Literal[2] = 2
    maximum_cost_liability_usd_micros: Annotated[int, Field(ge=1)]
    configured_phase_budget_usd_micros: Annotated[int, Field(ge=1)]
    provider_calls_made_before_authorization: Literal[0] = 0
    application_retries_per_request: Literal[0] = 0
    sdk_retries_per_request: Literal[0] = 0
    orphan_or_ambiguous_attempt_retry_permitted: Literal[False] = False
    authorization_sha256: Sha256

    @model_validator(mode="after")
    def validate_authorization(self) -> MetaSynContextualFrontierAuthorizationV1:
        if (
            [item.order for item in self.authorized_calls] != [0, 1]
            or self.authorized_roster_sha256
            != hash_canonical([item.model_dump(mode="json") for item in self.authorized_calls])
            or self.maximum_cost_liability_usd_micros
            != sum(item.request_cost_ceiling_usd_micros for item in self.authorized_calls)
            or self.maximum_cost_liability_usd_micros > self.configured_phase_budget_usd_micros
        ):
            raise ValueError("contextual_frontier_authorization_budget_or_roster_mismatch")
        _self_hash(
            self,
            "authorization_sha256",
            "contextual_frontier_authorization_hash_mismatch",
        )
        return self


def freeze_metasyn_contextual_frontier_authorization_v1(
    *, plan: MetaSynContextualFrontierPlanV1, phase_budget_usd_micros: int
) -> MetaSynContextualFrontierAuthorizationV1:
    canonical = MetaSynContextualFrontierPlanV1.model_validate(plan.model_dump(mode="json"))
    calls = [
        MetaSynContextualFrontierAuthorizedCallV1(
            order=item.order,
            request_key=item.request.request_key,
            request_sha256=item.request_sha256,
            provider_binding_sha256=item.provider_binding_sha256,
            request_cost_ceiling_usd_micros=item.request_cost_ceiling_usd_micros,
        )
        for item in canonical.roster
    ]
    payload = {
        "authorization_version": AUTHORIZATION_VERSION,
        "plan_sha256": canonical.plan_sha256,
        "authorized_calls": calls,
        "authorized_call_count": 2,
        "authorized_roster_sha256": hash_canonical(
            [item.model_dump(mode="json") for item in calls]
        ),
        "maximum_provider_attempts": 2,
        "maximum_cost_liability_usd_micros": (canonical.total_cost_ceiling_usd_micros),
        "configured_phase_budget_usd_micros": phase_budget_usd_micros,
        "provider_calls_made_before_authorization": 0,
        "application_retries_per_request": 0,
        "sdk_retries_per_request": 0,
        "orphan_or_ambiguous_attempt_retry_permitted": False,
    }
    try:
        return MetaSynContextualFrontierAuthorizationV1.model_validate(
            {**payload, "authorization_sha256": hash_canonical(payload)}
        )
    except ValueError as exc:
        raise MetaSynContextualFrontierRuntimeV1Error(
            "contextual_frontier_phase_budget_insufficient"
        ) from exc


def authorize_metasyn_contextual_frontier_runtime_v1(
    *, workspace: Path, phase_budget_usd_micros: int
) -> MetaSynContextualFrontierAuthorizationV1:
    root = _existing_workspace(workspace)
    with _workspace_lock(root):
        plan = load_metasyn_contextual_frontier_plan_v1(workspace=root)
        authorization = freeze_metasyn_contextual_frontier_authorization_v1(
            plan=plan, phase_budget_usd_micros=phase_budget_usd_micros
        )
        path = _checked_artifact(root, Path("01-authorized.json"))
        if path.exists():
            observed = MetaSynContextualFrontierAuthorizationV1.model_validate(
                _load_object(path, code="contextual_frontier_authorization_invalid")
            )
            if observed != authorization:
                raise MetaSynContextualFrontierRuntimeV1Error(
                    "contextual_frontier_authorization_already_differs"
                )
            return observed
        _persist_json(path, authorization)
        return authorization


class MetaSynContextualFrontierIntentV1(_Frozen):
    intent_version: Literal["metasyn-contextual-frontier-intent-v1"] = INTENT_VERSION
    plan_sha256: Sha256
    authorization_sha256: Sha256
    order: Annotated[int, Field(ge=0, le=1)]
    request_key: str
    request_sha256: Sha256
    provider_binding_sha256: Sha256
    source_bearing: Literal[True] = True
    permitted_provider_attempts: Literal[1] = 1
    application_retries_permitted: Literal[0] = 0
    sdk_retries_permitted: Literal[0] = 0
    orphan_or_ambiguous_attempt_is_terminal: Literal[True] = True
    request_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    attempt_id: Sha256
    intent_sha256: Sha256

    @model_validator(mode="after")
    def validate_intent(self) -> MetaSynContextualFrontierIntentV1:
        expected_attempt = hash_canonical(
            {
                "plan_sha256": self.plan_sha256,
                "authorization_sha256": self.authorization_sha256,
                "request_sha256": self.request_sha256,
                "provider_binding_sha256": self.provider_binding_sha256,
                "permitted_provider_attempts": 1,
                "application_retries_permitted": 0,
                "sdk_retries_permitted": 0,
            }
        )
        if self.attempt_id != expected_attempt:
            raise ValueError("contextual_frontier_attempt_id_mismatch")
        _self_hash(self, "intent_sha256", "contextual_frontier_intent_hash_mismatch")
        return self


def _freeze_intent(
    *,
    plan: MetaSynContextualFrontierPlanV1,
    authorization: MetaSynContextualFrontierAuthorizationV1,
    item: MetaSynContextualFrontierRosterItemV1,
) -> MetaSynContextualFrontierIntentV1:
    authorized = authorization.authorized_calls[item.order]
    if (
        authorization.plan_sha256 != plan.plan_sha256
        or authorized.request_sha256 != item.request_sha256
        or authorized.provider_binding_sha256 != item.provider_binding_sha256
    ):
        raise MetaSynContextualFrontierRuntimeV1Error(
            "contextual_frontier_intent_authorization_mismatch"
        )
    attempt_id = hash_canonical(
        {
            "plan_sha256": plan.plan_sha256,
            "authorization_sha256": authorization.authorization_sha256,
            "request_sha256": item.request_sha256,
            "provider_binding_sha256": item.provider_binding_sha256,
            "permitted_provider_attempts": 1,
            "application_retries_permitted": 0,
            "sdk_retries_permitted": 0,
        }
    )
    payload = {
        "intent_version": INTENT_VERSION,
        "plan_sha256": plan.plan_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "order": item.order,
        "request_key": item.request.request_key,
        "request_sha256": item.request_sha256,
        "provider_binding_sha256": item.provider_binding_sha256,
        "source_bearing": True,
        "permitted_provider_attempts": 1,
        "application_retries_permitted": 0,
        "sdk_retries_permitted": 0,
        "orphan_or_ambiguous_attempt_is_terminal": True,
        "request_cost_ceiling_usd_micros": item.request_cost_ceiling_usd_micros,
        "attempt_id": attempt_id,
    }
    return MetaSynContextualFrontierIntentV1.model_validate(
        {**payload, "intent_sha256": hash_canonical(payload)}
    )


class MetaSynContextualFrontierProviderReceiptV1(_Frozen):
    receipt_version: Literal["metasyn-contextual-frontier-provider-receipt-v1"] = RECEIPT_VERSION
    terminal: Literal[True] = True
    plan_sha256: Sha256
    authorization_sha256: Sha256
    intent_sha256: Sha256
    attempt_id: Sha256
    request_key: str
    request_sha256: Sha256
    provider_binding_sha256: Sha256
    provider_result: MetaSynContextualFrontierProviderResultV1
    provider_result_sha256: Sha256
    credential_archived: Literal[False] = False
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> MetaSynContextualFrontierProviderReceiptV1:
        if (
            self.provider_result.request_sha256 != self.request_sha256
            or self.provider_result.result_sha256 != self.provider_result_sha256
        ):
            raise ValueError("contextual_frontier_provider_receipt_alias_mismatch")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(self, "receipt_sha256", "contextual_frontier_receipt_hash_mismatch")
        return self


IncidentKind = Literal[
    "orphan_intent_observed_on_resume",
    "provider_call_raised_after_durable_intent",
    "provider_result_invalid_after_return",
]


class MetaSynContextualFrontierIncidentV1(_Frozen):
    incident_version: Literal["metasyn-contextual-frontier-ambiguity-incident-v1"] = (
        INCIDENT_VERSION
    )
    status: Literal["terminal_ambiguous_attempt_poison"] = "terminal_ambiguous_attempt_poison"
    incident_kind: IncidentKind
    plan_sha256: Sha256
    authorization_sha256: Sha256
    intent_sha256: Sha256
    attempt_id: Sha256
    request_key: str
    request_sha256: Sha256
    provider_binding_sha256: Sha256
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
    credential_archived: Literal[False] = False
    incident_sha256: Sha256

    @model_validator(mode="after")
    def validate_incident(self) -> MetaSynContextualFrontierIncidentV1:
        for value in (self.exception_type, self.provider_request_id):
            if value is not None and not _SAFE_IDENTIFIER_RE.fullmatch(value):
                raise ValueError("contextual_frontier_incident_identifier_unsafe")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(self, "incident_sha256", "contextual_frontier_incident_hash_mismatch")
        return self


def _safe_exception_type(exc: BaseException) -> str:
    value = f"{type(exc).__module__}.{type(exc).__name__}"
    return value if _SAFE_IDENTIFIER_RE.fullmatch(value) else "builtins.Exception"


def _safe_status(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    return value if type(value) is int and 100 <= value <= 599 else None


def _safe_request_id(exc: BaseException) -> str | None:
    value = getattr(exc, "request_id", None)
    return _safe_provider_scalar(value)


def _freeze_incident(
    *,
    kind: IncidentKind,
    intent: MetaSynContextualFrontierIntentV1,
    response_observation: Literal[
        "unknown_after_orphaned_intent",
        "not_observed_by_executor",
        "observed_but_invalid",
    ],
    exc: BaseException | None = None,
) -> MetaSynContextualFrontierIncidentV1:
    payload = {
        "incident_version": INCIDENT_VERSION,
        "status": "terminal_ambiguous_attempt_poison",
        "incident_kind": kind,
        "plan_sha256": intent.plan_sha256,
        "authorization_sha256": intent.authorization_sha256,
        "intent_sha256": intent.intent_sha256,
        "attempt_id": intent.attempt_id,
        "request_key": intent.request_key,
        "request_sha256": intent.request_sha256,
        "provider_binding_sha256": intent.provider_binding_sha256,
        "response_observation": response_observation,
        "exception_type": _safe_exception_type(exc) if exc is not None else None,
        "http_status": _safe_status(exc) if exc is not None else None,
        "provider_request_id": _safe_request_id(exc) if exc is not None else None,
        "possible_provider_attempts": 1,
        "retry_this_request_permitted": False,
        "credential_archived": False,
    }
    return MetaSynContextualFrontierIncidentV1.model_validate(
        {**payload, "incident_sha256": hash_canonical(payload)}
    )


ValidationStatus = Literal[
    "typed_graph_mechanics_completed",
    "scientific_abstention",
    "provider_result_failed",
    "contextual_validation_failed_closed",
]


class MetaSynContextualFrontierValidationResultV1(_Frozen):
    validation_version: Literal["metasyn-contextual-frontier-validation-result-v1"] = (
        VALIDATION_VERSION
    )
    status: ValidationStatus
    plan_sha256: Sha256
    runtime_pipeline_sha256: Sha256
    authorization_sha256: Sha256
    intent_sha256: Sha256
    request_key: str
    request_sha256: Sha256
    witness_id: str
    provider_binding_sha256: Sha256
    provider_receipt_sha256: Sha256
    provider_result_sha256: Sha256
    provider_execution_binding_sha256: Sha256
    provider_outcome: ProviderOutcome
    model_outcome: ContextualPacketCompletedV3 | ContextualPacketAbstentionV3 | None
    model_outcome_sha256: Sha256 | None
    groundings: list[ContextualGroundedClaimV3] | None
    grounding_membership_sha256: Sha256 | None
    grounded_effect: ContextualGroundedEffectV3 | None
    grounded_effect_sha256: Sha256 | None
    contextual_grounding_core_sha256: Sha256 | None
    runtime_grounding_binding_sha256: Sha256 | None
    native_projection: ContextualNativeProjectionV3 | None
    native_projection_sha256: Sha256 | None
    fresh_native_typed_graph_completed: bool
    failure_code: str | None
    credential_archived: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    validation_sha256: Sha256

    @model_validator(mode="after")
    def validate_validation(self) -> MetaSynContextualFrontierValidationResultV1:
        completed_values = (
            self.groundings,
            self.grounding_membership_sha256,
            self.grounded_effect,
            self.grounded_effect_sha256,
            self.contextual_grounding_core_sha256,
            self.runtime_grounding_binding_sha256,
            self.native_projection,
            self.native_projection_sha256,
        )
        if self.status == "typed_graph_mechanics_completed":
            if (
                not self.fresh_native_typed_graph_completed
                or any(value is None for value in completed_values)
                or not isinstance(self.model_outcome, ContextualPacketCompletedV3)
                or self.failure_code is not None
            ):
                raise ValueError("contextual_frontier_success_validation_shape_invalid")
            assert self.groundings is not None
            assert self.grounded_effect is not None
            assert self.native_projection is not None
            if (
                self.grounding_membership_sha256
                != hash_canonical([item.grounding_sha256 for item in self.groundings])
                or self.grounded_effect_sha256 != self.grounded_effect.effect_sha256
                or self.native_projection_sha256 != self.native_projection.projection_sha256
                or self.native_projection.status != "typed_graph_mechanics_completed"
                or self.native_projection.outcome_origin != "runtime_outcome_supplied_by_caller"
                or self.native_projection.fragment is None
                or self.native_projection.fragment.graph is None
                or self.native_projection.extraction_accuracy_authority
                or self.native_projection.synthesis_input_authority
                or self.native_projection.scientific_synthesis_authority
                or self.native_projection.scientific_effectiveness_authority
                or self.native_projection.calibration_authority
                or self.native_projection.claim_release_authority
            ):
                raise ValueError("contextual_frontier_fresh_projection_invalid")
        elif self.fresh_native_typed_graph_completed or any(
            value is not None for value in completed_values
        ):
            raise ValueError("contextual_frontier_non_success_has_grounded_artifact")
        if (self.model_outcome is None) != (self.model_outcome_sha256 is None):
            raise ValueError("contextual_frontier_model_outcome_hash_presence_mismatch")
        if self.model_outcome is not None and self.model_outcome_sha256 != hash_canonical(
            self.model_outcome.model_dump(mode="json")
        ):
            raise ValueError("contextual_frontier_model_outcome_hash_mismatch")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(
            self,
            "validation_sha256",
            "contextual_frontier_validation_hash_mismatch",
        )
        return self


def _provider_execution_binding(
    *,
    plan: MetaSynContextualFrontierPlanV1,
    authorization: MetaSynContextualFrontierAuthorizationV1,
    intent: MetaSynContextualFrontierIntentV1,
    receipt: MetaSynContextualFrontierProviderReceiptV1,
) -> str:
    return hash_canonical(
        {
            "binding_version": "metasyn-contextual-frontier-provider-execution-binding-v1",
            "runtime_pipeline_sha256": plan.runtime_pipeline_sha256,
            "plan_sha256": plan.plan_sha256,
            "authorization_sha256": authorization.authorization_sha256,
            "intent_sha256": intent.intent_sha256,
            "attempt_id": intent.attempt_id,
            "request_sha256": intent.request_sha256,
            "provider_binding_sha256": intent.provider_binding_sha256,
            "provider_receipt_sha256": receipt.receipt_sha256,
            "provider_result_sha256": receipt.provider_result_sha256,
        }
    )


def _process_provider_receipt(
    *,
    plan: MetaSynContextualFrontierPlanV1,
    authorization: MetaSynContextualFrontierAuthorizationV1,
    item: MetaSynContextualFrontierRosterItemV1,
    intent: MetaSynContextualFrontierIntentV1,
    receipt: MetaSynContextualFrontierProviderReceiptV1,
) -> MetaSynContextualFrontierValidationResultV1:
    provider = receipt.provider_result
    execution_binding = _provider_execution_binding(
        plan=plan, authorization=authorization, intent=intent, receipt=receipt
    )
    common: dict[str, Any] = {
        "validation_version": VALIDATION_VERSION,
        "plan_sha256": plan.plan_sha256,
        "runtime_pipeline_sha256": plan.runtime_pipeline_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "intent_sha256": intent.intent_sha256,
        "request_key": item.request.request_key,
        "request_sha256": item.request_sha256,
        "witness_id": item.witness_id,
        "provider_binding_sha256": item.provider_binding_sha256,
        "provider_receipt_sha256": receipt.receipt_sha256,
        "provider_result_sha256": provider.result_sha256,
        "provider_execution_binding_sha256": execution_binding,
        "provider_outcome": provider.outcome,
        "credential_archived": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_effectiveness_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    empty = {
        "groundings": None,
        "grounding_membership_sha256": None,
        "grounded_effect": None,
        "grounded_effect_sha256": None,
        "contextual_grounding_core_sha256": None,
        "runtime_grounding_binding_sha256": None,
        "native_projection": None,
        "native_projection_sha256": None,
        "fresh_native_typed_graph_completed": False,
    }
    if provider.outcome != "completed" or provider.parsed_json is None:
        payload = {
            **common,
            "status": "provider_result_failed",
            "model_outcome": None,
            "model_outcome_sha256": None,
            **empty,
            "failure_code": provider.outcome,
        }
    elif provider.parsed_json.get("packet_status") == "unable_to_complete":
        try:
            abstention = ContextualPacketAbstentionV3.model_validate(provider.parsed_json)
        except ValueError:
            payload = {
                **common,
                "status": "contextual_validation_failed_closed",
                "model_outcome": None,
                "model_outcome_sha256": None,
                **empty,
                "failure_code": "contextual_abstention_contract_invalid",
            }
        else:
            payload = {
                **common,
                "status": "scientific_abstention",
                "model_outcome": abstention,
                "model_outcome_sha256": hash_canonical(abstention.model_dump(mode="json")),
                **empty,
                "failure_code": None,
            }
    else:
        try:
            (
                outcome,
                groundings,
                effect,
                grounding_core_sha256,
                runtime_grounding_binding_sha256,
                projection,
            ) = project_contextual_grounded_outcome_v3(
                fixture_receipt=item.offline_witness,
                raw_outcome=provider.parsed_json,
                runtime_pipeline_sha256=plan.runtime_pipeline_sha256,
                provider_execution_binding_sha256=execution_binding,
            )
            if (
                projection.status != "typed_graph_mechanics_completed"
                or projection.outcome_origin != "runtime_outcome_supplied_by_caller"
                or projection.fragment is None
                or projection.fragment.graph is None
            ):
                raise ValueError("fresh_projection_not_completed")
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
                raise
            payload = {
                **common,
                "status": "contextual_validation_failed_closed",
                "model_outcome": None,
                "model_outcome_sha256": None,
                **empty,
                "failure_code": "contextual_grounding_or_projection_failed_closed",
            }
        else:
            payload = {
                **common,
                "status": "typed_graph_mechanics_completed",
                "model_outcome": outcome,
                "model_outcome_sha256": hash_canonical(outcome.model_dump(mode="json")),
                "groundings": groundings,
                "grounding_membership_sha256": hash_canonical(
                    [grounding.grounding_sha256 for grounding in groundings]
                ),
                "grounded_effect": effect,
                "grounded_effect_sha256": effect.effect_sha256,
                "contextual_grounding_core_sha256": grounding_core_sha256,
                "runtime_grounding_binding_sha256": runtime_grounding_binding_sha256,
                "native_projection": projection,
                "native_projection_sha256": projection.projection_sha256,
                "fresh_native_typed_graph_completed": True,
                "failure_code": None,
            }
    return MetaSynContextualFrontierValidationResultV1.model_validate(
        {**payload, "validation_sha256": hash_canonical(payload)}
    )


TerminalStatus = Literal[
    "typed_graph_smoke_completed",
    "roster_exhausted_without_typed_graph",
    "terminal_ambiguous_attempt_poison",
]


class MetaSynContextualFrontierTerminalReportV1(_Frozen):
    terminal_version: Literal["metasyn-contextual-frontier-terminal-report-v1"] = TERMINAL_VERSION
    terminal: Literal[True] = True
    status: TerminalStatus
    plan_sha256: Sha256
    runtime_pipeline_sha256: Sha256
    authorization_sha256: Sha256
    attempted_request_keys: list[str]
    unattempted_request_keys: list[str]
    unattempted_reason: str | None
    validation_results: list[MetaSynContextualFrontierValidationResultV1]
    validation_membership_sha256: Sha256
    ambiguity_incident: MetaSynContextualFrontierIncidentV1 | None
    successful_request_key: str | None
    provider_attempt_count_upper_bound: Annotated[int, Field(ge=0, le=2)]
    provider_receipt_count: Annotated[int, Field(ge=0, le=2)]
    first_success_stopped_fallback: bool
    credential_archived: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    report_sha256: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> MetaSynContextualFrontierTerminalReportV1:
        validation_keys = [item.request_key for item in self.validation_results]
        if (
            self.attempted_request_keys[: len(validation_keys)] != validation_keys
            or self.validation_membership_sha256
            != hash_canonical([item.validation_sha256 for item in self.validation_results])
            or len(set(self.attempted_request_keys + self.unattempted_request_keys))
            != len(self.attempted_request_keys) + len(self.unattempted_request_keys)
        ):
            raise ValueError("contextual_frontier_terminal_membership_mismatch")
        successes = [
            item
            for item in self.validation_results
            if item.status == "typed_graph_mechanics_completed"
        ]
        if self.status == "typed_graph_smoke_completed":
            if len(successes) != 1 or self.successful_request_key != successes[0].request_key:
                raise ValueError("contextual_frontier_terminal_success_mismatch")
        elif self.successful_request_key is not None or successes:
            raise ValueError("contextual_frontier_terminal_non_success_has_success")
        if (self.status == "terminal_ambiguous_attempt_poison") != (
            self.ambiguity_incident is not None
        ):
            raise ValueError("contextual_frontier_terminal_incident_mismatch")
        if self.ambiguity_incident is not None and (
            not self.attempted_request_keys
            or self.attempted_request_keys[-1] != self.ambiguity_incident.request_key
            or len(self.attempted_request_keys) != len(validation_keys) + 1
        ):
            raise ValueError("contextual_frontier_terminal_incident_attempt_mismatch")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(self, "report_sha256", "contextual_frontier_terminal_hash_mismatch")
        return self


def _terminal_report(
    *,
    plan: MetaSynContextualFrontierPlanV1,
    authorization: MetaSynContextualFrontierAuthorizationV1,
    validations: Sequence[MetaSynContextualFrontierValidationResultV1],
    incident: MetaSynContextualFrontierIncidentV1 | None,
    receipt_count: int,
) -> MetaSynContextualFrontierTerminalReportV1:
    results = list(validations)
    attempted = [item.request_key for item in results]
    if incident is not None:
        attempted.append(incident.request_key)
    all_keys = [item.request.request_key for item in plan.roster]
    unattempted = [key for key in all_keys if key not in attempted]
    success = next(
        (item for item in results if item.status == "typed_graph_mechanics_completed"),
        None,
    )
    if incident is not None:
        status: TerminalStatus = "terminal_ambiguous_attempt_poison"
        reason = "terminal_ambiguous_attempt_no_retry"
    elif success is not None:
        status = "typed_graph_smoke_completed"
        reason = "first_fully_grounded_native_typed_graph_stopped_roster" if unattempted else None
    else:
        status = "roster_exhausted_without_typed_graph"
        reason = None
    payload = {
        "terminal_version": TERMINAL_VERSION,
        "terminal": True,
        "status": status,
        "plan_sha256": plan.plan_sha256,
        "runtime_pipeline_sha256": plan.runtime_pipeline_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "attempted_request_keys": attempted,
        "unattempted_request_keys": unattempted,
        "unattempted_reason": reason,
        "validation_results": results,
        "validation_membership_sha256": hash_canonical(
            [item.validation_sha256 for item in results]
        ),
        "ambiguity_incident": incident,
        "successful_request_key": success.request_key if success is not None else None,
        "provider_attempt_count_upper_bound": len(attempted),
        "provider_receipt_count": receipt_count,
        "first_success_stopped_fallback": bool(success is not None and unattempted),
        "credential_archived": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_effectiveness_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierTerminalReportV1.model_validate(
        {**payload, "report_sha256": hash_canonical(payload)}
    )


def _freeze_provider_receipt(
    *,
    plan: MetaSynContextualFrontierPlanV1,
    authorization: MetaSynContextualFrontierAuthorizationV1,
    item: MetaSynContextualFrontierRosterItemV1,
    intent: MetaSynContextualFrontierIntentV1,
    result: MetaSynContextualFrontierProviderResultV1,
) -> MetaSynContextualFrontierProviderReceiptV1:
    if (
        result.request_sha256 != item.request_sha256
        or result.identity_sha256 != plan.provider_identity_sha256
        or result.config_sha256 != plan.config_sha256
        or result.wire_call_surface_sha256 != item.request.wire_call_surface_sha256
    ):
        raise MetaSynContextualFrontierRuntimeV1Error(
            "contextual_frontier_provider_result_request_mismatch"
        )
    payload = {
        "receipt_version": RECEIPT_VERSION,
        "terminal": True,
        "plan_sha256": plan.plan_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "intent_sha256": intent.intent_sha256,
        "attempt_id": intent.attempt_id,
        "request_key": item.request.request_key,
        "request_sha256": item.request_sha256,
        "provider_binding_sha256": item.provider_binding_sha256,
        "provider_result": result,
        "provider_result_sha256": result.result_sha256,
        "credential_archived": False,
    }
    return MetaSynContextualFrontierProviderReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def _load_authorization(
    *, workspace: Path, plan: MetaSynContextualFrontierPlanV1
) -> MetaSynContextualFrontierAuthorizationV1:
    path = _checked_artifact(workspace, Path("01-authorized.json"))
    authorization = MetaSynContextualFrontierAuthorizationV1.model_validate(
        _load_object(path, code="contextual_frontier_authorization_required")
    )
    expected_calls = [
        (
            item.order,
            item.request.request_key,
            item.request_sha256,
            item.provider_binding_sha256,
            item.request_cost_ceiling_usd_micros,
        )
        for item in plan.roster
    ]
    observed_calls = [
        (
            item.order,
            item.request_key,
            item.request_sha256,
            item.provider_binding_sha256,
            item.request_cost_ceiling_usd_micros,
        )
        for item in authorization.authorized_calls
    ]
    if authorization.plan_sha256 != plan.plan_sha256 or observed_calls != expected_calls:
        raise MetaSynContextualFrontierRuntimeV1Error(
            "contextual_frontier_authorization_plan_mismatch"
        )
    return authorization


def _load_intent(path: Path) -> MetaSynContextualFrontierIntentV1:
    return MetaSynContextualFrontierIntentV1.model_validate(
        _load_object(path, code="contextual_frontier_intent_invalid")
    )


def _load_receipt(path: Path) -> MetaSynContextualFrontierProviderReceiptV1:
    return MetaSynContextualFrontierProviderReceiptV1.model_validate(
        _load_object(path, code="contextual_frontier_provider_receipt_invalid")
    )


def _load_validation(path: Path) -> MetaSynContextualFrontierValidationResultV1:
    return MetaSynContextualFrontierValidationResultV1.model_validate(
        _load_object(path, code="contextual_frontier_validation_result_invalid")
    )


def _load_incident(path: Path) -> MetaSynContextualFrontierIncidentV1:
    return MetaSynContextualFrontierIncidentV1.model_validate(
        _load_object(path, code="contextual_frontier_incident_invalid")
    )


def _write_terminal(
    *, workspace: Path, report: MetaSynContextualFrontierTerminalReportV1
) -> MetaSynContextualFrontierTerminalReportV1:
    path = _checked_artifact(workspace, Path("02-terminal.json"))
    if path.exists():
        observed = MetaSynContextualFrontierTerminalReportV1.model_validate(
            _load_object(path, code="contextual_frontier_terminal_report_invalid")
        )
        if observed != report:
            raise MetaSynContextualFrontierRuntimeV1Error(
                "contextual_frontier_terminal_report_drift"
            )
        return observed
    _persist_json(path, report)
    return report


def execute_metasyn_contextual_frontier_runtime_v1(
    *,
    workspace: Path,
    client: MetaSynContextualFrontierClientProtocol,
) -> MetaSynContextualFrontierTerminalReportV1:
    """Execute at most two authorized calls; never retry an exact request."""

    root = _existing_workspace(workspace)
    with _workspace_lock(root):
        terminal_path = _checked_artifact(root, Path("02-terminal.json"))
        if terminal_path.exists():
            return MetaSynContextualFrontierTerminalReportV1.model_validate(
                _load_object(terminal_path, code="contextual_frontier_terminal_report_invalid")
            )
        plan = load_metasyn_contextual_frontier_plan_v1(workspace=root)
        authorization = _load_authorization(workspace=root, plan=plan)
        validations: list[MetaSynContextualFrontierValidationResultV1] = []
        receipt_count = 0
        for item in plan.roster:
            key = item.request.request_key
            intent_path = _checked_artifact(root, Path("intents") / f"{key}.json")
            receipt_path = _checked_artifact(root, Path("provider-receipts") / f"{key}.json")
            validation_path = _checked_artifact(root, Path("validations") / f"{key}.json")
            incident_path = _checked_artifact(root, Path("incidents") / f"{key}.json")
            expected_intent = _freeze_intent(plan=plan, authorization=authorization, item=item)
            if validation_path.exists():
                if not intent_path.exists() or not receipt_path.exists():
                    raise MetaSynContextualFrontierRuntimeV1Error(
                        "contextual_frontier_validation_without_provider_archive"
                    )
                intent = _load_intent(intent_path)
                receipt = _load_receipt(receipt_path)
                observed = _load_validation(validation_path)
                replayed = _process_provider_receipt(
                    plan=plan,
                    authorization=authorization,
                    item=item,
                    intent=intent,
                    receipt=receipt,
                )
                if intent != expected_intent or observed != replayed:
                    raise MetaSynContextualFrontierRuntimeV1Error(
                        "contextual_frontier_validation_replay_mismatch"
                    )
                validations.append(observed)
                receipt_count += 1
                if observed.fresh_native_typed_graph_completed:
                    return _write_terminal(
                        workspace=root,
                        report=_terminal_report(
                            plan=plan,
                            authorization=authorization,
                            validations=validations,
                            incident=None,
                            receipt_count=receipt_count,
                        ),
                    )
                continue
            if incident_path.exists():
                incident = _load_incident(incident_path)
                if not intent_path.exists() or _load_intent(intent_path) != expected_intent:
                    raise MetaSynContextualFrontierRuntimeV1Error(
                        "contextual_frontier_incident_intent_mismatch"
                    )
                return _write_terminal(
                    workspace=root,
                    report=_terminal_report(
                        plan=plan,
                        authorization=authorization,
                        validations=validations,
                        incident=incident,
                        receipt_count=receipt_count,
                    ),
                )
            if intent_path.exists():
                intent = _load_intent(intent_path)
                if intent != expected_intent:
                    raise MetaSynContextualFrontierRuntimeV1Error(
                        "contextual_frontier_existing_intent_drift"
                    )
                if receipt_path.exists():
                    receipt = _load_receipt(receipt_path)
                    validation = _process_provider_receipt(
                        plan=plan,
                        authorization=authorization,
                        item=item,
                        intent=intent,
                        receipt=receipt,
                    )
                    _persist_json(validation_path, validation)
                    validations.append(validation)
                    receipt_count += 1
                    if validation.fresh_native_typed_graph_completed:
                        return _write_terminal(
                            workspace=root,
                            report=_terminal_report(
                                plan=plan,
                                authorization=authorization,
                                validations=validations,
                                incident=None,
                                receipt_count=receipt_count,
                            ),
                        )
                    continue
                incident = _freeze_incident(
                    kind="orphan_intent_observed_on_resume",
                    intent=intent,
                    response_observation="unknown_after_orphaned_intent",
                )
                _persist_json(incident_path, incident)
                return _write_terminal(
                    workspace=root,
                    report=_terminal_report(
                        plan=plan,
                        authorization=authorization,
                        validations=validations,
                        incident=incident,
                        receipt_count=receipt_count,
                    ),
                )
            # Exact liability begins only after this durable intent exists.
            _persist_json(intent_path, expected_intent)
            try:
                raw_result = client.generate(item.request)
            except Exception as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
                    raise
                incident = _freeze_incident(
                    kind="provider_call_raised_after_durable_intent",
                    intent=expected_intent,
                    response_observation="not_observed_by_executor",
                    exc=exc,
                )
                _persist_json(incident_path, incident)
                return _write_terminal(
                    workspace=root,
                    report=_terminal_report(
                        plan=plan,
                        authorization=authorization,
                        validations=validations,
                        incident=incident,
                        receipt_count=receipt_count,
                    ),
                )
            try:
                result = MetaSynContextualFrontierProviderResultV1.model_validate(
                    raw_result.model_dump(mode="json")
                    if isinstance(raw_result, MetaSynContextualFrontierProviderResultV1)
                    else raw_result
                )
                receipt = _freeze_provider_receipt(
                    plan=plan,
                    authorization=authorization,
                    item=item,
                    intent=expected_intent,
                    result=result,
                )
            except Exception as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
                    raise
                incident = _freeze_incident(
                    kind="provider_result_invalid_after_return",
                    intent=expected_intent,
                    response_observation="observed_but_invalid",
                    exc=exc,
                )
                _persist_json(incident_path, incident)
                return _write_terminal(
                    workspace=root,
                    report=_terminal_report(
                        plan=plan,
                        authorization=authorization,
                        validations=validations,
                        incident=incident,
                        receipt_count=receipt_count,
                    ),
                )
            _persist_json(receipt_path, receipt)
            receipt_count += 1
            validation = _process_provider_receipt(
                plan=plan,
                authorization=authorization,
                item=item,
                intent=expected_intent,
                receipt=receipt,
            )
            _persist_json(validation_path, validation)
            validations.append(validation)
            if validation.fresh_native_typed_graph_completed:
                return _write_terminal(
                    workspace=root,
                    report=_terminal_report(
                        plan=plan,
                        authorization=authorization,
                        validations=validations,
                        incident=None,
                        receipt_count=receipt_count,
                    ),
                )
        return _write_terminal(
            workspace=root,
            report=_terminal_report(
                plan=plan,
                authorization=authorization,
                validations=validations,
                incident=None,
                receipt_count=receipt_count,
            ),
        )


class MetaSynContextualFrontierWorkspaceValidationV1(_Frozen):
    workspace_validation_version: Literal["metasyn-contextual-frontier-workspace-validation-v1"] = (
        "metasyn-contextual-frontier-workspace-validation-v1"
    )
    status: Literal["prepared", "authorized", "terminal"]
    plan: MetaSynContextualFrontierPlanV1
    plan_sha256: Sha256
    authorization: MetaSynContextualFrontierAuthorizationV1 | None
    authorization_sha256: Sha256 | None
    terminal_report: MetaSynContextualFrontierTerminalReportV1 | None
    terminal_report_sha256: Sha256 | None
    intent_count: Annotated[int, Field(ge=0, le=2)]
    provider_receipt_count: Annotated[int, Field(ge=0, le=2)]
    validation_result_count: Annotated[int, Field(ge=0, le=2)]
    ambiguity_incident_count: Annotated[int, Field(ge=0, le=1)]
    external_plan_and_source_replayed: bool
    archive_replayed: Literal[True] = True
    credential_archived: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    workspace_validation_sha256: Sha256

    @model_validator(mode="after")
    def validate_workspace_validation(
        self,
    ) -> MetaSynContextualFrontierWorkspaceValidationV1:
        if self.plan_sha256 != self.plan.plan_sha256:
            raise ValueError("contextual_frontier_workspace_plan_alias_mismatch")
        if (self.authorization is None) != (self.authorization_sha256 is None):
            raise ValueError("contextual_frontier_workspace_auth_presence_mismatch")
        if (
            self.authorization is not None
            and self.authorization_sha256 != self.authorization.authorization_sha256
        ):
            raise ValueError("contextual_frontier_workspace_auth_alias_mismatch")
        if (self.terminal_report is None) != (self.terminal_report_sha256 is None):
            raise ValueError("contextual_frontier_workspace_terminal_presence_mismatch")
        if (
            self.terminal_report is not None
            and self.terminal_report_sha256 != self.terminal_report.report_sha256
        ):
            raise ValueError("contextual_frontier_workspace_terminal_alias_mismatch")
        expected_status = (
            "terminal"
            if self.terminal_report is not None
            else "authorized"
            if self.authorization is not None
            else "prepared"
        )
        if self.status != expected_status:
            raise ValueError("contextual_frontier_workspace_status_mismatch")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(
            self,
            "workspace_validation_sha256",
            "contextual_frontier_workspace_validation_hash_mismatch",
        )
        return self


def _audit_workspace_symlinks(workspace: Path) -> None:
    for directory, dirnames, filenames in os.walk(workspace, followlinks=False):
        current = Path(directory)
        for name in [*dirnames, *filenames]:
            path = current / name
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                raise MetaSynContextualFrontierRuntimeV1Error(
                    "contextual_frontier_workspace_entry_unreadable"
                ) from exc
            if stat.S_ISLNK(mode):
                raise MetaSynContextualFrontierRuntimeV1Error(
                    "contextual_frontier_workspace_symlink_forbidden"
                )


def validate_metasyn_contextual_frontier_runtime_v1(
    *,
    repository_root: Path,
    workspace: Path,
    external_replay: bool = True,
) -> MetaSynContextualFrontierWorkspaceValidationV1:
    """Replay every durable artifact and, by default, every source-plan byte."""

    root = _existing_workspace(workspace)
    _audit_workspace_symlinks(root)
    with _workspace_lock(root):
        plan = load_metasyn_contextual_frontier_plan_v1(workspace=root)
        if external_replay:
            replayed_plan = freeze_metasyn_contextual_frontier_plan_v1(
                repository_root=repository_root
            )
            if replayed_plan != plan:
                raise MetaSynContextualFrontierRuntimeV1Error(
                    "contextual_frontier_external_plan_replay_mismatch"
                )
        authorization_path = _checked_artifact(root, Path("01-authorized.json"))
        authorization = (
            _load_authorization(workspace=root, plan=plan) if authorization_path.exists() else None
        )
        intents: list[MetaSynContextualFrontierIntentV1] = []
        receipts: list[MetaSynContextualFrontierProviderReceiptV1] = []
        validations: list[MetaSynContextualFrontierValidationResultV1] = []
        incidents: list[MetaSynContextualFrontierIncidentV1] = []
        if authorization is None:
            for directory in ("intents", "provider-receipts", "validations", "incidents"):
                if _checked_artifact(root, Path(directory)).exists():
                    raise MetaSynContextualFrontierRuntimeV1Error(
                        "contextual_frontier_provider_archive_before_authorization"
                    )
        else:
            for item in plan.roster:
                key = item.request.request_key
                intent_path = _checked_artifact(root, Path("intents") / f"{key}.json")
                receipt_path = _checked_artifact(root, Path("provider-receipts") / f"{key}.json")
                validation_path = _checked_artifact(root, Path("validations") / f"{key}.json")
                incident_path = _checked_artifact(root, Path("incidents") / f"{key}.json")
                expected_intent = _freeze_intent(plan=plan, authorization=authorization, item=item)
                if intent_path.exists():
                    intent = _load_intent(intent_path)
                    if intent != expected_intent:
                        raise MetaSynContextualFrontierRuntimeV1Error(
                            "contextual_frontier_archived_intent_replay_mismatch"
                        )
                    intents.append(intent)
                elif any(path.exists() for path in (receipt_path, validation_path, incident_path)):
                    raise MetaSynContextualFrontierRuntimeV1Error(
                        "contextual_frontier_archive_without_intent"
                    )

                if receipt_path.exists():
                    if not intent_path.exists() or incident_path.exists():
                        raise MetaSynContextualFrontierRuntimeV1Error(
                            "contextual_frontier_receipt_archive_conflict"
                        )
                    intent = _load_intent(intent_path)
                    receipt = _load_receipt(receipt_path)
                    replayed_receipt = _freeze_provider_receipt(
                        plan=plan,
                        authorization=authorization,
                        item=item,
                        intent=intent,
                        result=receipt.provider_result,
                    )
                    if receipt != replayed_receipt:
                        raise MetaSynContextualFrontierRuntimeV1Error(
                            "contextual_frontier_provider_archive_replay_mismatch"
                        )
                    receipts.append(receipt)
                if validation_path.exists():
                    if not receipt_path.exists():
                        raise MetaSynContextualFrontierRuntimeV1Error(
                            "contextual_frontier_validation_without_receipt"
                        )
                    intent = _load_intent(intent_path)
                    receipt = _load_receipt(receipt_path)
                    validation = _load_validation(validation_path)
                    replayed_validation = _process_provider_receipt(
                        plan=plan,
                        authorization=authorization,
                        item=item,
                        intent=intent,
                        receipt=receipt,
                    )
                    if validation != replayed_validation:
                        raise MetaSynContextualFrontierRuntimeV1Error(
                            "contextual_frontier_grounded_validation_replay_mismatch"
                        )
                    validations.append(validation)
                elif receipt_path.exists():
                    raise MetaSynContextualFrontierRuntimeV1Error(
                        "contextual_frontier_receipt_without_validation"
                    )
                if incident_path.exists():
                    if receipt_path.exists() or validation_path.exists():
                        raise MetaSynContextualFrontierRuntimeV1Error(
                            "contextual_frontier_incident_archive_conflict"
                        )
                    incident = _load_incident(incident_path)
                    if incident.intent_sha256 != expected_intent.intent_sha256:
                        raise MetaSynContextualFrontierRuntimeV1Error(
                            "contextual_frontier_incident_replay_mismatch"
                        )
                    incidents.append(incident)
        terminal_path = _checked_artifact(root, Path("02-terminal.json"))
        terminal = None
        if terminal_path.exists():
            if authorization is None:
                raise MetaSynContextualFrontierRuntimeV1Error(
                    "contextual_frontier_terminal_without_authorization"
                )
            terminal = MetaSynContextualFrontierTerminalReportV1.model_validate(
                _load_object(terminal_path, code="contextual_frontier_terminal_report_invalid")
            )
            if len(incidents) > 1:
                raise MetaSynContextualFrontierRuntimeV1Error(
                    "contextual_frontier_multiple_ambiguity_incidents"
                )
            expected_terminal = _terminal_report(
                plan=plan,
                authorization=authorization,
                validations=validations,
                incident=incidents[0] if incidents else None,
                receipt_count=len(receipts),
            )
            if terminal != expected_terminal:
                raise MetaSynContextualFrontierRuntimeV1Error(
                    "contextual_frontier_terminal_external_replay_mismatch"
                )
        payload = {
            "workspace_validation_version": ("metasyn-contextual-frontier-workspace-validation-v1"),
            "status": (
                "terminal"
                if terminal is not None
                else "authorized"
                if authorization is not None
                else "prepared"
            ),
            "plan": plan,
            "plan_sha256": plan.plan_sha256,
            "authorization": authorization,
            "authorization_sha256": (
                authorization.authorization_sha256 if authorization is not None else None
            ),
            "terminal_report": terminal,
            "terminal_report_sha256": terminal.report_sha256 if terminal else None,
            "intent_count": len(intents),
            "provider_receipt_count": len(receipts),
            "validation_result_count": len(validations),
            "ambiguity_incident_count": len(incidents),
            "external_plan_and_source_replayed": external_replay,
            "archive_replayed": True,
            "credential_archived": False,
            "extraction_accuracy_authority": False,
            "synthesis_input_authority": False,
            "scientific_synthesis_authority": False,
            "scientific_effectiveness_authority": False,
            "calibration_authority": False,
            "claim_release_authority": False,
        }
        return MetaSynContextualFrontierWorkspaceValidationV1.model_validate(
            {
                **payload,
                "workspace_validation_sha256": hash_canonical(payload),
            }
        )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_WORKSPACE",
    "MetaSynContextualFrontierAuthorizationV1",
    "MetaSynContextualFrontierClientProtocol",
    "MetaSynContextualFrontierClientV1",
    "MetaSynContextualFrontierConfigV1",
    "MetaSynContextualFrontierIncidentV1",
    "MetaSynContextualFrontierIntentV1",
    "MetaSynContextualFrontierPlanV1",
    "MetaSynContextualFrontierProviderReceiptV1",
    "MetaSynContextualFrontierProviderResultV1",
    "MetaSynContextualFrontierRequestV1",
    "MetaSynContextualFrontierRuntimeV1Error",
    "MetaSynContextualFrontierTerminalReportV1",
    "MetaSynContextualFrontierUsageV1",
    "MetaSynContextualFrontierValidationResultV1",
    "MetaSynContextualFrontierWorkspaceValidationV1",
    "authorize_metasyn_contextual_frontier_runtime_v1",
    "execute_metasyn_contextual_frontier_runtime_v1",
    "freeze_metasyn_contextual_frontier_authorization_v1",
    "freeze_metasyn_contextual_frontier_plan_v1",
    "freeze_metasyn_contextual_frontier_provider_result_v1",
    "load_metasyn_contextual_frontier_config_v1",
    "load_metasyn_contextual_frontier_plan_v1",
    "prepare_metasyn_contextual_frontier_runtime_v1",
    "validate_metasyn_contextual_frontier_runtime_v1",
]
