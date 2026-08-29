"""Immutable hosted-provider runtime for the label-blind MetaSyn bounded diagnostic.

This module is intentionally separate from the frozen local-Ollama runtime.  It binds a
first-party hosted provider, its SDK/configuration and compiled structured-output schema to
the existing provider-neutral MetaSyn adapter.  Every provider request has a durable intent
before the call, exactly one permitted attempt, no application or SDK retry, and either one
immutable response receipt or one terminal ambiguity incident.  A missing outcome beside a
durable intent is poisoned on resume and is never retried.

The run is yield-only.  It never opens review conclusions, effect directions, aggregate
effects, or test labels, and no runtime failure is converted into a scientific inventory.
"""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
import platform
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from functools import lru_cache
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from jsonschema.validators import validator_for
from pydantic import Field, ValidationError, field_validator, model_validator

from literature_multiverse.anthropic_bounded_generation import (
    AnthropicBoundedConfigV1,
    AnthropicBoundedRequestV1,
    AnthropicBoundedResultV1,
    AnthropicCompiledSchemaV1,
    AnthropicEffectKind,
    AnthropicProviderIdentityV1,
    AnthropicSchemaKind,
    AnthropicTransportMode,
    compile_anthropic_bounded_schema,
    freeze_anthropic_bounded_request,
    freeze_anthropic_provider_identity,
    freeze_anthropic_wire_call_surface,
    project_anthropic_preflight_fixture,
)
from literature_multiverse.lineage import (
    OutputExistsError,
    atomic_write_json,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.metasyn_bounded_adapter import (
    MetaSynBoundedAdapterBundleV1,
    MetaSynBoundedAdapterError,
    MetaSynBoundedPrivateYieldReportV1,
    MetaSynBoundedRowContextV1,
    MetaSynInventoryValidationReceiptV1,
    MetaSynPacketCallV1,
    MetaSynPacketValidationReceiptV1,
    MetaSynPublicationResultV1,
    freeze_metasyn_bounded_adapter_bundle_from_workspace,
    freeze_metasyn_bounded_private_yield_report,
    freeze_metasyn_inventory_validation_receipt,
    freeze_metasyn_packet_call,
    freeze_metasyn_packet_validation_receipt,
    freeze_metasyn_publication_result,
    validate_metasyn_bounded_adapter_bundle_external_replay,
    validate_metasyn_bounded_private_yield_report,
    validate_metasyn_inventory_validation_receipt,
    validate_metasyn_packet_validation_receipt,
    validate_metasyn_publication_result,
)
from literature_multiverse.metasyn_typed_pilot import (
    EXPECTED_SELECTED_COMPONENTS,
    EXPECTED_SELECTED_PAPERS,
    EXPECTED_SELECTED_QUESTIONS,
    compute_metasyn_typed_pilot_pipeline_fingerprint,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_bounded_generation import (
    NativeBoundedGenerationError,
    NativeCandidateDescriptor,
)
from literature_multiverse.native_bounded_schema_v2 import (
    inventory_schema_bundle_v2,
    packet_schema_bundle_v2,
    schema_bundle_receipt_binding_v2,
    schema_v2_contract,
    synthetic_schema_v2_preflight_fingerprint,
    synthetic_schema_v2_preflight_specs,
    validate_inventory_for_row_v2,
    validate_packet_for_candidate_v2,
    validate_raw_payload_against_schema_v2,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    compute_pipeline_fingerprint,
)

RUNTIME_VERSION = "metasyn-bounded-hosted-anthropic-runtime-v2"
CONFIG_VERSION = "metasyn-bounded-hosted-anthropic-config-v1"
EXECUTION_BUNDLE_VERSION = "metasyn-bounded-hosted-execution-bundle-v2"
ATTEMPT_INTENT_VERSION = "metasyn-bounded-hosted-attempt-intent-v2"
CALL_RECEIPT_VERSION = "metasyn-bounded-hosted-call-receipt-v2"
INCIDENT_VERSION = "metasyn-bounded-hosted-ambiguity-incident-v2"
PREFLIGHT_VERSION = "metasyn-bounded-hosted-eight-call-preflight-v2"
SMOKE_VERSION = "metasyn-bounded-hosted-smoke-gate-v2"
ROW_RESULT_VERSION = "metasyn-bounded-hosted-row-result-v2"
LEDGER_VERSION = "metasyn-bounded-hosted-ledger-v2"
PRIVATE_REPORT_VERSION = "metasyn-bounded-hosted-private-yield-report-v2"
COST_AUTHORIZATION_VERSION = (
    "metasyn-bounded-hosted-pre-first-call-cost-authorization-v2"
)
RUNTIME_COMPONENT_VERSION = "5"

DEFAULT_CONFIG_PATH = Path("configs/benchmarks/metasyn-bounded-anthropic-v1.json")
DEFAULT_EXECUTION_WORKSPACE = Path("data/cache/metasyn/bounded-anthropic-yield-v5")
DEFAULT_PILOT_WORKSPACE = Path("data/cache/metasyn/typed-oracle-pilot-v2")

PREFLIGHT_CALL_COUNT = 8
PUBLICATION_COUNT = 32
MAX_CANDIDATES_PER_PUBLICATION = 8
MAX_THEORETICAL_PROVIDER_CALLS = (
    PREFLIGHT_CALL_COUNT + PUBLICATION_COUNT + PUBLICATION_COUNT * MAX_CANDIDATES_PER_PUBLICATION
)
PREFLIGHT_STRUCTURED_CALL_COUNT = 3
PREFLIGHT_PROMPT_JSON_CALL_COUNT = 5
MAX_STRUCTURED_PROVIDER_CALLS = PREFLIGHT_STRUCTURED_CALL_COUNT + PUBLICATION_COUNT
MAX_PROMPT_JSON_PROVIDER_CALLS = (
    PREFLIGHT_PROMPT_JSON_CALL_COUNT
    + PUBLICATION_COUNT * MAX_CANDIDATES_PER_PUBLICATION
)

_MODULE_ENTRYPOINT = "src/literature_multiverse/metasyn_bounded_hosted_runtime.py"
_SCRIPT_ENTRYPOINT = "scripts/run_metasyn_bounded_hosted_runtime.py"
_RUNTIME_ENTRYPOINTS = (_MODULE_ENTRYPOINT, _SCRIPT_ENTRYPOINT)
_RUNTIME_NON_PYTHON_INPUTS = (
    DEFAULT_CONFIG_PATH.as_posix(),
    "prompts/metasyn_candidate_inventory.md",
    "prompts/metasyn_candidate_packet.md",
    "pyproject.toml",
    "uv.lock",
)

CallStage = Literal["preflight", "inventory", "packet"]
CallValidationStatus = Literal[
    "preflight_fixture_valid",
    "inventory_valid_candidates",
    "inventory_valid_no_candidate_non_authorizing",
    "inventory_valid_capacity_or_uncertainty_non_authorizing",
    "packet_completed",
    "packet_unable_to_complete",
    "provider_failure",
    "response_json_missing",
    "wire_schema_invalid",
    "original_provider_schema_invalid",
    "full_schema_invalid",
    "inventory_contract_invalid",
    "packet_contract_invalid",
    "packet_source_grounding_invalid",
    "preflight_fixture_mismatch",
]
IncidentKind = Literal[
    "provider_call_raised_after_durable_intent",
    "orphan_intent_observed_on_resume",
]
RowStatus = Literal[
    "typed_publication_output",
    "adapter_inventory_no_candidate",
    "adapter_inventory_uncertain",
    "adapter_packet_unable",
    "runtime_inventory_blocked",
    "runtime_packet_blocked",
]


class MetaSynHostedRuntimeError(ValueError):
    """The hosted execution state or its immutable lineage is unsafe."""


class HostedBoundedClientProtocol(Protocol):
    """Narrow injected-client boundary used by the runtime and fake tests."""

    def generate(self, request: AnthropicBoundedRequestV1) -> AnthropicBoundedResultV1:
        """Make exactly one SDK request and return a closed provider result."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MetaSynHostedRuntimeError("metasyn_hosted_json_artifact_unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynHostedRuntimeError("metasyn_hosted_json_artifact_invalid") from exc
    if not isinstance(value, dict):
        raise MetaSynHostedRuntimeError("metasyn_hosted_json_artifact_not_object")
    return value


def _canonical_workspace(workspace: Path, *, create: bool = False) -> Path:
    if workspace.is_symlink():
        raise MetaSynHostedRuntimeError("metasyn_hosted_workspace_symlink")
    if create:
        workspace.mkdir(parents=True, exist_ok=True)
    try:
        resolved = workspace.resolve(strict=True)
    except OSError as exc:
        raise MetaSynHostedRuntimeError("metasyn_hosted_workspace_missing") from exc
    if not resolved.is_dir():
        raise MetaSynHostedRuntimeError("metasyn_hosted_workspace_not_directory")
    return resolved


@contextmanager
def _workspace_lock(workspace: Path) -> Iterator[None]:
    lock_path = workspace / ".hosted-runtime.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def metasyn_hosted_runtime_paths(workspace: Path) -> dict[str, Path]:
    return {
        "execution_bundle": workspace / "execution-bundle.json",
        "cost_authorization": workspace / "cost-authorization-receipt.json",
        "preflight": workspace / "preflight-receipt.json",
        "smoke": workspace / "smoke-receipt.json",
        "intents": workspace / "call-intents",
        "receipts": workspace / "call-receipts",
        "incidents": workspace / "call-incidents",
        "row_results": workspace / "row-results",
        "ledger": workspace / "hosted-ledger.json",
        "provider_neutral_yield": workspace / "provider-neutral-yield-report.json",
        "private_report": workspace / "hosted-private-report.json",
    }


def _resolve_local_import(
    *, repository_root: Path, current_path: str, module: str, level: int
) -> str | None:
    current = Path(current_path).with_suffix("")
    if level:
        package_parts = list(current.parts[:-1])
        if level > len(package_parts):
            return None
        module_parts = package_parts[: len(package_parts) - (level - 1)]
        if module:
            module_parts.extend(module.split("."))
        candidates = [
            Path(*module_parts).with_suffix(".py"),
            Path(*module_parts) / "__init__.py",
        ]
    elif module == "literature_multiverse":
        candidates = [Path("src/literature_multiverse/__init__.py")]
    elif module.startswith("literature_multiverse."):
        relative = Path("src", *module.split("."))
        candidates = [relative.with_suffix(".py"), relative / "__init__.py"]
    elif module.startswith("scripts."):
        relative = Path(*module.split("."))
        candidates = [relative.with_suffix(".py")]
    else:
        return None
    for candidate in candidates:
        if (repository_root / candidate).is_file():
            return candidate.as_posix()
    return None


def _runtime_dependency_closure(repository_root: Path) -> list[str]:
    pending = list(_RUNTIME_ENTRYPOINTS)
    observed = {"src/literature_multiverse/__init__.py"}
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        path = repository_root / relative
        if not path.is_file():
            raise MetaSynHostedRuntimeError(f"metasyn_hosted_dependency_missing:{relative}")
        observed.add(relative)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise MetaSynHostedRuntimeError(
                f"metasyn_hosted_dependency_unreadable:{relative}"
            ) from exc
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom):
                modules.append((node.module or "", node.level))
            elif isinstance(node, ast.Import):
                modules.extend((alias.name, 0) for alias in node.names)
            for module, level in modules:
                dependency = _resolve_local_import(
                    repository_root=repository_root,
                    current_path=relative,
                    module=module,
                    level=level,
                )
                if dependency is not None and dependency not in observed:
                    pending.append(dependency)
    return sorted(observed)


class MetaSynHostedUsageV1(ContractModel):
    input_tokens: Annotated[int, Field(ge=0)] = 0
    output_tokens: Annotated[int, Field(ge=0)] = 0
    cache_creation_input_tokens: Annotated[int, Field(ge=0)] = 0
    cache_read_input_tokens: Annotated[int, Field(ge=0)] = 0


class MetaSynHostedCostV1(ContractModel):
    estimated_cost_usd_micros: Annotated[int, Field(ge=0)] = 0
    request_ceiling_usd_micros: Annotated[int, Field(ge=0)] = 0


def _sum_usage(values: Sequence[MetaSynHostedUsageV1]) -> MetaSynHostedUsageV1:
    return MetaSynHostedUsageV1(
        input_tokens=sum(item.input_tokens for item in values),
        output_tokens=sum(item.output_tokens for item in values),
        cache_creation_input_tokens=sum(item.cache_creation_input_tokens for item in values),
        cache_read_input_tokens=sum(item.cache_read_input_tokens for item in values),
    )


def _sum_cost(values: Sequence[MetaSynHostedCostV1]) -> MetaSynHostedCostV1:
    return MetaSynHostedCostV1(
        estimated_cost_usd_micros=sum(item.estimated_cost_usd_micros for item in values),
        request_ceiling_usd_micros=sum(item.request_ceiling_usd_micros for item in values),
    )


def _usd_micros(value: Any) -> int:
    if value is None:
        return 0
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise MetaSynHostedRuntimeError("metasyn_hosted_cost_invalid") from exc
    if not amount.is_finite() or amount < 0:
        raise MetaSynHostedRuntimeError("metasyn_hosted_cost_invalid")
    return int((amount * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING))


def _provider_usage_cost(
    result: AnthropicBoundedResultV1,
) -> tuple[MetaSynHostedUsageV1, MetaSynHostedCostV1]:
    payload = result.model_dump(mode="json")
    raw_usage = payload.get("usage")
    usage = raw_usage if isinstance(raw_usage, Mapping) else {}
    raw_cost = payload.get("cost")
    cost = raw_cost if isinstance(raw_cost, Mapping) else {}

    def integer(*names: str) -> int:
        for name in names:
            value = usage.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return 0

    estimated = next(
        (
            cost[name]
            for name in (
                "estimated_cost_usd",
                "reported_estimated_cost_usd",
                "cost_usd",
            )
            if name in cost
        ),
        None,
    )
    ceiling = next(
        (
            cost[name]
            for name in (
                "request_ceiling_usd",
                "request_cost_ceiling_usd",
                "maximum_request_cost_usd",
                "charged_cost_upper_bound_usd",
                "max_cost_usd",
            )
            if name in cost
        ),
        None,
    )
    return (
        MetaSynHostedUsageV1(
            input_tokens=integer("input_tokens"),
            output_tokens=integer("output_tokens"),
            cache_creation_input_tokens=integer(
                "cache_creation_input_tokens", "cache_creation_tokens"
            ),
            cache_read_input_tokens=integer("cache_read_input_tokens", "cache_read_tokens"),
        ),
        MetaSynHostedCostV1(
            estimated_cost_usd_micros=_usd_micros(estimated),
            request_ceiling_usd_micros=_usd_micros(ceiling),
        ),
    )


class MetaSynHostedRuntimeConfigV1(ContractModel):
    config_version: Literal["metasyn-bounded-hosted-anthropic-config-v1"] = CONFIG_VERSION
    diagnostic_scope: Literal["label_blind_calibration_oracle_corpus_yield_only"] = (
        "label_blind_calibration_oracle_corpus_yield_only"
    )
    runtime_provider: Literal["anthropic_first_party_api"] = "anthropic_first_party_api"
    model: Literal["claude-sonnet-5"] = "claude-sonnet-5"
    timeout_seconds: Annotated[float, Field(gt=0, le=600)]
    input_rate_usd_per_million_tokens: Literal["2"] = "2"
    output_rate_usd_per_million_tokens: Literal["10"] = "10"
    pricing_source_url: Literal["https://platform.claude.com/docs/en/about-claude/pricing"] = (
        "https://platform.claude.com/docs/en/about-claude/pricing"
    )
    pricing_rate_table_sha256: str
    service_tier: Literal["standard_only"] = "standard_only"
    fixed_framing_tokens: Annotated[int, Field(ge=0, le=100_000)]
    system_prompt: Annotated[str, Field(min_length=1, max_length=4000)]
    preflight_max_output_tokens: Annotated[int, Field(ge=1, le=64_000)]
    inventory_max_output_tokens: Annotated[int, Field(ge=1, le=64_000)]
    packet_max_output_tokens: Annotated[int, Field(ge=1, le=64_000)]
    maximum_input_tokens_all_calls: Annotated[int, Field(ge=1, le=100_000_000)]
    maximum_provider_calls: Literal[296] = MAX_THEORETICAL_PROVIDER_CALLS
    maximum_authorized_cost_usd_micros: Annotated[int, Field(ge=1)]
    question_count: Literal[10] = EXPECTED_SELECTED_QUESTIONS
    publication_count: Literal[32] = EXPECTED_SELECTED_PAPERS
    maximum_candidates_per_publication: Literal[8] = MAX_CANDIDATES_PER_PUBLICATION
    preflight_call_count: Literal[8] = PREFLIGHT_CALL_COUNT
    smoke_row_ordinal: Literal[0] = 0
    model_calls_per_request: Literal[1] = 1
    application_retries_per_request: Literal[0] = 0
    sdk_retries_per_request: Literal[0] = 0
    reference_fields_opened: Literal[False] = False
    operator_authorized_source_transmission: Literal[True] = True
    config_sha256: str

    @field_validator("pricing_rate_table_sha256", "config_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_hosted_config_hash_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_config(self) -> MetaSynHostedRuntimeConfigV1:
        rate_table = {
            "model": self.model,
            "pricing_source_url": self.pricing_source_url,
            "service_tier": self.service_tier,
            "input_rate_usd_per_million_tokens": (self.input_rate_usd_per_million_tokens),
            "output_rate_usd_per_million_tokens": (self.output_rate_usd_per_million_tokens),
        }
        if self.pricing_rate_table_sha256 != hash_canonical(rate_table):
            raise ValueError("metasyn_hosted_pricing_rate_table_hash_mismatch")
        expected_output_tokens = (
            self.preflight_call_count * self.preflight_max_output_tokens
            + self.publication_count * self.inventory_max_output_tokens
            + self.publication_count
            * self.maximum_candidates_per_publication
            * self.packet_max_output_tokens
        )
        maximum_input_cost = (
            Decimal(self.maximum_input_tokens_all_calls)
            * Decimal(self.input_rate_usd_per_million_tokens)
            / Decimal(1_000_000)
        )
        maximum_output_cost = (
            Decimal(expected_output_tokens)
            * Decimal(self.output_rate_usd_per_million_tokens)
            / Decimal(1_000_000)
        )
        required_micros = int(
            ((maximum_input_cost + maximum_output_cost) * Decimal(1_000_000)).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        if self.maximum_authorized_cost_usd_micros < required_micros:
            raise ValueError("metasyn_hosted_authorized_cost_below_theoretical_ceiling")
        payload = self.model_dump(mode="json", exclude={"config_sha256"})
        if self.config_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_hosted_config_hash_mismatch")
        return self

    def anthropic_config(self) -> AnthropicBoundedConfigV1:
        return AnthropicBoundedConfigV1(
            model=self.model,
            timeout_seconds=self.timeout_seconds,
        )


def load_metasyn_hosted_runtime_config(
    *, repository_root: Path, config_path: Path = DEFAULT_CONFIG_PATH
) -> tuple[MetaSynHostedRuntimeConfigV1, str]:
    root = repository_root.resolve(strict=True)
    candidate = config_path if config_path.is_absolute() else root / config_path
    if candidate.is_symlink():
        raise MetaSynHostedRuntimeError("metasyn_hosted_config_symlink")
    resolved = candidate.resolve(strict=True)
    if resolved != (root / DEFAULT_CONFIG_PATH).resolve(strict=True):
        raise MetaSynHostedRuntimeError("metasyn_hosted_config_path_not_exact")
    return (
        MetaSynHostedRuntimeConfigV1.model_validate(_read_json_object(resolved)),
        sha256_file(resolved),
    )


def _pilot_downstream_sha(repository_root: Path) -> tuple[str, str]:
    pilot = compute_metasyn_typed_pilot_pipeline_fingerprint(root=repository_root)
    if len(pilot.components) != 1:
        raise MetaSynHostedRuntimeError("metasyn_hosted_pilot_component_count_invalid")
    downstream = pilot.components[0].settings.get("downstream_verifier_pipeline_sha256")
    if not isinstance(downstream, str) or not SHA256_RE.fullmatch(downstream):
        raise MetaSynHostedRuntimeError("metasyn_hosted_downstream_pipeline_missing")
    return pilot.pipeline_sha256, downstream


def compute_metasyn_hosted_runtime_fingerprint(
    *,
    repository_root: Path,
    adapter_pipeline_sha256: str,
    config_sha256: str,
    provider_identity_sha256: str,
    downstream_verifier_pipeline_sha256: str,
) -> PipelineFingerprint:
    for value in (
        adapter_pipeline_sha256,
        config_sha256,
        provider_identity_sha256,
        downstream_verifier_pipeline_sha256,
    ):
        if not SHA256_RE.fullmatch(value):
            raise MetaSynHostedRuntimeError("metasyn_hosted_fingerprint_input_invalid")
    root = repository_root.resolve(strict=True)
    schema_contract = schema_v2_contract()
    component = PipelineComponentSpec(
        component_id="metasyn-bounded-hosted-anthropic-runtime",
        component_version=RUNTIME_COMPONENT_VERSION,
        file_paths=sorted(
            {
                *_runtime_dependency_closure(root),
                *_RUNTIME_NON_PYTHON_INPUTS,
            }
        ),
        settings={
            "adapter_pipeline_sha256": adapter_pipeline_sha256,
            "ambiguity_or_orphan_never_retried": True,
            "application_retries_per_request": 0,
            "config_sha256": config_sha256,
            "downstream_verifier_pipeline_sha256": (downstream_verifier_pipeline_sha256),
            "durable_pre_call_intent_required": True,
            "full_v2_acceptance_after_provider_schema_required": True,
            "installed_dependency_versions": {
                name: distribution_version(name) for name in ("anthropic", "jsonschema", "pydantic")
            },
            "maximum_theoretical_provider_calls": MAX_THEORETICAL_PROVIDER_CALLS,
            "model_calls_per_request": 1,
            "native_schema_v2_contract_sha256": schema_contract["contract_sha256"],
            "platform_machine": platform.machine(),
            "platform_system": platform.system(),
            "provider_identity_sha256": provider_identity_sha256,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "reference_fields_opened": False,
            "sdk_retries_per_request": 0,
            "sequential_full_roster": True,
            "smoke_gate_scientific_correctness_authority": False,
            "synthetic_preflight_call_count": PREFLIGHT_CALL_COUNT,
            "yield_only_no_accuracy_direction_or_release_authority": True,
        },
    )
    return compute_pipeline_fingerprint(root=root, components=[component])


class MetaSynHostedExecutionBundleV1(ContractModel):
    execution_bundle_version: Literal["metasyn-bounded-hosted-execution-bundle-v2"] = (
        EXECUTION_BUNDLE_VERSION
    )
    runtime_version: Literal["metasyn-bounded-hosted-anthropic-runtime-v2"] = RUNTIME_VERSION
    status: Literal["frozen_label_blind_hosted_runtime_no_provider_calls"] = (
        "frozen_label_blind_hosted_runtime_no_provider_calls"
    )
    pilot_workspace_relative: Annotated[str, Field(min_length=1, max_length=2048)]
    config_path: Literal["configs/benchmarks/metasyn-bounded-anthropic-v1.json"] = (
        DEFAULT_CONFIG_PATH.as_posix()
    )
    config_file_sha256: str
    runtime_config: MetaSynHostedRuntimeConfigV1
    config_sha256: str
    anthropic_config: AnthropicBoundedConfigV1
    anthropic_config_sha256: str
    provider_identity: AnthropicProviderIdentityV1
    provider_identity_sha256: str
    adapter_bundle: MetaSynBoundedAdapterBundleV1
    adapter_bundle_sha256: str
    adapter_pipeline_sha256: str
    upstream_pilot_pipeline_sha256: str
    downstream_verifier_pipeline_sha256: str
    native_schema_v2_contract: dict[str, Any]
    native_schema_v2_contract_sha256: str
    schema_v2_preflight_fingerprint: str
    runtime_pipeline_fingerprint: PipelineFingerprint
    runtime_pipeline_sha256: str
    question_count: Literal[10] = EXPECTED_SELECTED_QUESTIONS
    component_count: Literal[10] = EXPECTED_SELECTED_COMPONENTS
    publication_count: Literal[32] = EXPECTED_SELECTED_PAPERS
    question_membership_sha256: str
    component_membership_sha256: str
    row_membership_sha256: str
    smoke_row_ordinal: Literal[0] = 0
    smoke_row_context_sha256: str
    maximum_theoretical_provider_calls: Literal[296] = MAX_THEORETICAL_PROVIDER_CALLS
    maximum_authorized_cost_usd_micros: Annotated[int, Field(ge=1)]
    reference_fields_unopened: Literal[True] = True
    operator_authorized_source_transmission: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    provider_calls_made: Literal[False] = False
    permitted_metrics: Literal["contract_grounding_publication_and_synthesis_input_yield_only"] = (
        "contract_grounding_publication_and_synthesis_input_yield_only"
    )
    execution_bundle_sha256: str

    @field_validator(
        "config_file_sha256",
        "config_sha256",
        "anthropic_config_sha256",
        "provider_identity_sha256",
        "adapter_bundle_sha256",
        "adapter_pipeline_sha256",
        "upstream_pilot_pipeline_sha256",
        "downstream_verifier_pipeline_sha256",
        "native_schema_v2_contract_sha256",
        "schema_v2_preflight_fingerprint",
        "runtime_pipeline_sha256",
        "question_membership_sha256",
        "component_membership_sha256",
        "row_membership_sha256",
        "smoke_row_context_sha256",
        "execution_bundle_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_hosted_bundle_hash_invalid:{info.field_name}")
        return value

    @field_validator("pilot_workspace_relative")
    @classmethod
    def validate_workspace_relative(cls, value: str) -> str:
        path = Path(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != value
            or not value.startswith("data/cache/")
        ):
            raise ValueError("metasyn_hosted_pilot_workspace_path_unsafe")
        return value

    @model_validator(mode="after")
    def validate_bundle(self) -> MetaSynHostedExecutionBundleV1:
        adapter = self.adapter_bundle
        if self.config_sha256 != self.runtime_config.config_sha256:
            raise ValueError("metasyn_hosted_config_hash_alias_mismatch")
        if self.anthropic_config != self.runtime_config.anthropic_config():
            raise ValueError("metasyn_hosted_anthropic_config_mismatch")
        if self.anthropic_config_sha256 != self.anthropic_config.config_sha256:
            raise ValueError("metasyn_hosted_anthropic_config_hash_mismatch")
        if self.provider_identity_sha256 != self.provider_identity.identity_sha256:
            raise ValueError("metasyn_hosted_provider_identity_hash_mismatch")
        if self.adapter_bundle_sha256 != adapter.adapter_bundle_sha256:
            raise ValueError("metasyn_hosted_adapter_bundle_hash_mismatch")
        if self.adapter_pipeline_sha256 != adapter.adapter_pipeline_sha256:
            raise ValueError("metasyn_hosted_adapter_pipeline_hash_mismatch")
        if self.upstream_pilot_pipeline_sha256 != (adapter.upstream_pilot_pipeline_sha256):
            raise ValueError("metasyn_hosted_upstream_pipeline_hash_mismatch")
        current_contract = schema_v2_contract()
        if (
            self.native_schema_v2_contract != current_contract
            or self.native_schema_v2_contract_sha256 != current_contract["contract_sha256"]
        ):
            raise ValueError("metasyn_hosted_native_schema_contract_mismatch")
        if self.schema_v2_preflight_fingerprint != (synthetic_schema_v2_preflight_fingerprint()):
            raise ValueError("metasyn_hosted_preflight_fingerprint_mismatch")
        if self.runtime_pipeline_sha256 != (self.runtime_pipeline_fingerprint.pipeline_sha256):
            raise ValueError("metasyn_hosted_runtime_pipeline_hash_mismatch")
        if len(self.runtime_pipeline_fingerprint.components) != 1:
            raise ValueError("metasyn_hosted_runtime_component_count_mismatch")
        component = self.runtime_pipeline_fingerprint.components[0]
        if component.component_id != "metasyn-bounded-hosted-anthropic-runtime":
            raise ValueError("metasyn_hosted_runtime_component_mismatch")
        expected_settings = {
            "adapter_pipeline_sha256": self.adapter_pipeline_sha256,
            "config_sha256": self.config_sha256,
            "downstream_verifier_pipeline_sha256": (self.downstream_verifier_pipeline_sha256),
            "native_schema_v2_contract_sha256": (self.native_schema_v2_contract_sha256),
            "provider_identity_sha256": self.provider_identity_sha256,
        }
        if any(component.settings.get(key) != value for key, value in expected_settings.items()):
            raise ValueError("metasyn_hosted_runtime_pipeline_setting_mismatch")
        if (
            self.question_count != adapter.question_count
            or self.component_count != adapter.component_count
            or self.publication_count != adapter.publication_count
            or self.question_membership_sha256 != adapter.question_membership_sha256
            or self.component_membership_sha256 != adapter.component_membership_sha256
            or self.row_membership_sha256 != adapter.row_membership_sha256
        ):
            raise ValueError("metasyn_hosted_adapter_roster_alias_mismatch")
        smoke_row = adapter.row_contexts[self.smoke_row_ordinal]
        if self.smoke_row_context_sha256 != smoke_row.row_context_sha256:
            raise ValueError("metasyn_hosted_smoke_row_hash_mismatch")
        if self.maximum_authorized_cost_usd_micros != (
            self.runtime_config.maximum_authorized_cost_usd_micros
        ):
            raise ValueError("metasyn_hosted_cost_budget_alias_mismatch")
        if not self.runtime_config.operator_authorized_source_transmission:
            raise ValueError("metasyn_hosted_source_transmission_not_authorized")
        payload = self.model_dump(mode="json", exclude={"execution_bundle_sha256"})
        if self.execution_bundle_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_hosted_execution_bundle_hash_mismatch")
        return self


def _coerce_metasyn_hosted_execution_bundle(
    value: MetaSynHostedExecutionBundleV1 | Mapping[str, Any],
) -> MetaSynHostedExecutionBundleV1:
    """Validate mappings fully and replay an already-validated instance cheaply.

    Every stage boundary loads the bundle from canonical JSON through the full
    Pydantic contract and then externally replays current code/config.  The same
    frozen object is subsequently passed through intent/receipt factories once
    per provider call.  Re-running the recursively expensive 32-row Pydantic
    graph there does not add a boundary check; it only makes a 296-call run
    quadratic in contract size.  For a typed instance, its outer content hash
    and all critical nested aliases are sufficient to detect post-validation
    mutation before it is reused.
    """

    if not isinstance(value, MetaSynHostedExecutionBundleV1):
        return MetaSynHostedExecutionBundleV1.model_validate(value)
    payload = value.model_dump(mode="json", exclude={"execution_bundle_sha256"})
    if value.execution_bundle_sha256 != hash_canonical(payload):
        raise MetaSynHostedRuntimeError("metasyn_hosted_typed_bundle_hash_mismatch")
    if (
        value.config_sha256 != value.runtime_config.config_sha256
        or value.anthropic_config_sha256 != value.anthropic_config.config_sha256
        or value.provider_identity_sha256 != value.provider_identity.identity_sha256
        or value.adapter_bundle_sha256 != value.adapter_bundle.adapter_bundle_sha256
        or value.adapter_pipeline_sha256 != value.adapter_bundle.adapter_pipeline_sha256
        or value.runtime_pipeline_sha256 != value.runtime_pipeline_fingerprint.pipeline_sha256
    ):
        raise MetaSynHostedRuntimeError("metasyn_hosted_typed_bundle_alias_mismatch")
    return value


def _private_workspace_relative(path: Path, *, repository_root: Path) -> str:
    root = repository_root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise MetaSynHostedRuntimeError("metasyn_hosted_pilot_workspace_symlink")
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise MetaSynHostedRuntimeError("metasyn_hosted_pilot_workspace_unsafe") from exc
    if not resolved.is_dir() or not relative.as_posix().startswith("data/cache/"):
        raise MetaSynHostedRuntimeError("metasyn_hosted_pilot_workspace_not_private")
    return relative.as_posix()


def freeze_metasyn_hosted_execution_bundle(
    *,
    adapter_bundle: MetaSynBoundedAdapterBundleV1 | Mapping[str, Any],
    runtime_config: MetaSynHostedRuntimeConfigV1 | Mapping[str, Any],
    config_file_sha256: str,
    pilot_workspace_relative: str,
    repository_root: Path,
) -> MetaSynHostedExecutionBundleV1:
    root = repository_root.resolve(strict=True)
    adapter = MetaSynBoundedAdapterBundleV1.model_validate(adapter_bundle)
    config = MetaSynHostedRuntimeConfigV1.model_validate(runtime_config)
    current_pilot_sha, downstream_sha = _pilot_downstream_sha(root)
    if current_pilot_sha != adapter.upstream_pilot_pipeline_sha256:
        raise MetaSynHostedRuntimeError("metasyn_hosted_adapter_upstream_stale")
    anthropic_config = config.anthropic_config()
    identity = freeze_anthropic_provider_identity(anthropic_config)
    pipeline = compute_metasyn_hosted_runtime_fingerprint(
        repository_root=root,
        adapter_pipeline_sha256=adapter.adapter_pipeline_sha256,
        config_sha256=config.config_sha256,
        provider_identity_sha256=identity.identity_sha256,
        downstream_verifier_pipeline_sha256=downstream_sha,
    )
    contract = schema_v2_contract()
    payload: dict[str, Any] = {
        "execution_bundle_version": EXECUTION_BUNDLE_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "status": "frozen_label_blind_hosted_runtime_no_provider_calls",
        "pilot_workspace_relative": pilot_workspace_relative,
        "config_path": DEFAULT_CONFIG_PATH.as_posix(),
        "config_file_sha256": config_file_sha256,
        "runtime_config": config,
        "config_sha256": config.config_sha256,
        "anthropic_config": anthropic_config,
        "anthropic_config_sha256": anthropic_config.config_sha256,
        "provider_identity": identity,
        "provider_identity_sha256": identity.identity_sha256,
        "adapter_bundle": adapter,
        "adapter_bundle_sha256": adapter.adapter_bundle_sha256,
        "adapter_pipeline_sha256": adapter.adapter_pipeline_sha256,
        "upstream_pilot_pipeline_sha256": adapter.upstream_pilot_pipeline_sha256,
        "downstream_verifier_pipeline_sha256": downstream_sha,
        "native_schema_v2_contract": contract,
        "native_schema_v2_contract_sha256": contract["contract_sha256"],
        "schema_v2_preflight_fingerprint": (synthetic_schema_v2_preflight_fingerprint()),
        "runtime_pipeline_fingerprint": pipeline,
        "runtime_pipeline_sha256": pipeline.pipeline_sha256,
        "question_count": adapter.question_count,
        "component_count": adapter.component_count,
        "publication_count": adapter.publication_count,
        "question_membership_sha256": adapter.question_membership_sha256,
        "component_membership_sha256": adapter.component_membership_sha256,
        "row_membership_sha256": adapter.row_membership_sha256,
        "smoke_row_ordinal": config.smoke_row_ordinal,
        "smoke_row_context_sha256": (
            adapter.row_contexts[config.smoke_row_ordinal].row_context_sha256
        ),
        "maximum_theoretical_provider_calls": MAX_THEORETICAL_PROVIDER_CALLS,
        "maximum_authorized_cost_usd_micros": (config.maximum_authorized_cost_usd_micros),
        "reference_fields_unopened": True,
        "operator_authorized_source_transmission": True,
        "official_test_labels_opened": False,
        "provider_calls_made": False,
        "permitted_metrics": ("contract_grounding_publication_and_synthesis_input_yield_only"),
    }
    return MetaSynHostedExecutionBundleV1.model_validate(
        {**payload, "execution_bundle_sha256": hash_canonical(payload)}
    )


def prepare_metasyn_hosted_runtime(
    *,
    repository_root: Path,
    pilot_workspace: Path = DEFAULT_PILOT_WORKSPACE,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> MetaSynHostedExecutionBundleV1:
    root = repository_root.resolve(strict=True)
    relative = _private_workspace_relative(pilot_workspace, repository_root=root)
    adapter = freeze_metasyn_bounded_adapter_bundle_from_workspace(
        repository_root=root, workspace=root / relative
    )
    # A second independent replay at the runtime boundary prevents a caller from
    # substituting a merely shape-valid adapter after preparation.
    adapter = validate_metasyn_bounded_adapter_bundle_external_replay(
        adapter_bundle=adapter,
        repository_root=root,
        workspace=root / relative,
    )
    config, file_sha = load_metasyn_hosted_runtime_config(
        repository_root=root, config_path=config_path
    )
    return freeze_metasyn_hosted_execution_bundle(
        adapter_bundle=adapter,
        runtime_config=config,
        config_file_sha256=file_sha,
        pilot_workspace_relative=relative,
        repository_root=root,
    )


def validate_current_metasyn_hosted_execution_bundle(
    *,
    execution_bundle: MetaSynHostedExecutionBundleV1 | Mapping[str, Any],
    repository_root: Path,
    external_replay: bool = True,
) -> MetaSynHostedExecutionBundleV1:
    root = repository_root.resolve(strict=True)
    canonical = MetaSynHostedExecutionBundleV1.model_validate(execution_bundle)
    config, file_sha = load_metasyn_hosted_runtime_config(repository_root=root)
    if config != canonical.runtime_config or file_sha != canonical.config_file_sha256:
        raise MetaSynHostedRuntimeError("metasyn_hosted_current_config_mismatch")
    if external_replay:
        validate_metasyn_bounded_adapter_bundle_external_replay(
            adapter_bundle=canonical.adapter_bundle,
            repository_root=root,
            workspace=root / canonical.pilot_workspace_relative,
        )
    replayed = freeze_metasyn_hosted_execution_bundle(
        adapter_bundle=canonical.adapter_bundle,
        runtime_config=config,
        config_file_sha256=file_sha,
        pilot_workspace_relative=canonical.pilot_workspace_relative,
        repository_root=root,
    )
    if replayed != canonical:
        raise MetaSynHostedRuntimeError("metasyn_hosted_current_bundle_replay_mismatch")
    return canonical


def write_metasyn_hosted_execution_bundle(
    *,
    execution_bundle: MetaSynHostedExecutionBundleV1,
    workspace: Path,
    repository_root: Path,
) -> Path:
    bundle = validate_current_metasyn_hosted_execution_bundle(
        execution_bundle=execution_bundle,
        repository_root=repository_root,
        external_replay=True,
    )
    if workspace.exists() and (
        workspace.is_symlink() or not workspace.is_dir() or any(workspace.iterdir())
    ):
        raise OutputExistsError(workspace.as_posix())
    workspace.mkdir(parents=True, exist_ok=True)
    canonical = _canonical_workspace(workspace)
    output = canonical / "execution-bundle.json"
    atomic_write_json(output, bundle)
    return output


def load_current_metasyn_hosted_execution_bundle(
    *,
    workspace: Path,
    repository_root: Path,
    external_replay: bool = True,
) -> tuple[Path, MetaSynHostedExecutionBundleV1]:
    canonical = _canonical_workspace(workspace)
    bundle = validate_current_metasyn_hosted_execution_bundle(
        execution_bundle=_read_json_object(canonical / "execution-bundle.json"),
        repository_root=repository_root,
        external_replay=external_replay,
    )
    return canonical, bundle


class MetaSynHostedAttemptIntentV1(ContractModel):
    attempt_intent_version: Literal["metasyn-bounded-hosted-attempt-intent-v2"] = (
        ATTEMPT_INTENT_VERSION
    )
    runtime_version: Literal["metasyn-bounded-hosted-anthropic-runtime-v2"] = RUNTIME_VERSION
    status: Literal["durable_pre_call_intent_frozen"] = "durable_pre_call_intent_frozen"
    execution_bundle_sha256: str
    runtime_pipeline_sha256: str
    request_key: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{2,95}$")]
    stage: CallStage
    row_ordinal: Annotated[int, Field(ge=0, lt=32)] | None
    row_context_sha256: str | None
    candidate_index: Annotated[int, Field(ge=1, le=8)] | None
    candidate_sha256: str | None
    inventory_validation_receipt_sha256: str | None
    prompt_sha256: str
    base_system_sha256: str
    model_system_sha256: str
    base_prompt_sha256: str
    model_prompt_sha256: str
    schema_bundle_sha256: str
    provider_schema_sha256: str
    wire_schema_sha256: str
    full_acceptance_schema_sha256: str
    compiled_schema_sha256: str
    request: dict[str, Any]
    request_sha256: str
    provider_identity_sha256: str
    provider_config_sha256: str
    cost_authorization_sha256: str
    schema_kind: Literal["inventory", "packet"]
    effect_kind: AnthropicEffectKind | None
    transport_mode: AnthropicTransportMode
    transport_policy_binding_sha256: str
    structured_grammar_enforced_by_provider: bool
    output_format_present_in_call: bool
    expected_wire_call_sha256: str
    request_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    permitted_call_attempts: Literal[1] = 1
    application_retries_permitted: Literal[0] = 0
    sdk_retries_permitted: Literal[0] = 0
    orphan_or_ambiguous_attempt_is_terminal: Literal[True] = True
    attempt_id: str
    attempt_intent_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "runtime_pipeline_sha256",
        "row_context_sha256",
        "candidate_sha256",
        "inventory_validation_receipt_sha256",
        "prompt_sha256",
        "base_system_sha256",
        "model_system_sha256",
        "base_prompt_sha256",
        "model_prompt_sha256",
        "schema_bundle_sha256",
        "provider_schema_sha256",
        "wire_schema_sha256",
        "full_acceptance_schema_sha256",
        "compiled_schema_sha256",
        "request_sha256",
        "provider_identity_sha256",
        "provider_config_sha256",
        "cost_authorization_sha256",
        "transport_policy_binding_sha256",
        "expected_wire_call_sha256",
        "attempt_id",
        "attempt_intent_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_hosted_intent_hash_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_intent(self) -> MetaSynHostedAttemptIntentV1:
        source_call = self.stage in {"inventory", "packet"}
        if source_call != (self.row_ordinal is not None and self.row_context_sha256 is not None):
            raise ValueError("metasyn_hosted_intent_row_context_mismatch")
        packet = self.stage == "packet"
        if packet != (
            self.candidate_index is not None
            and self.candidate_sha256 is not None
            and self.inventory_validation_receipt_sha256 is not None
        ):
            raise ValueError("metasyn_hosted_intent_packet_context_mismatch")
        try:
            request = AnthropicBoundedRequestV1.model_validate(self.request)
        except ValueError as exc:
            raise ValueError("metasyn_hosted_intent_request_revalidation_failed") from exc
        if self.compiled_schema_sha256 != request.compiled_schema_sha256:
            raise ValueError("metasyn_hosted_intent_compiled_schema_hash_mismatch")
        if self.request_sha256 != request.request_sha256:
            raise ValueError("metasyn_hosted_intent_request_hash_mismatch")
        if request.request_key != self.request_key:
            raise ValueError("metasyn_hosted_intent_request_key_mismatch")
        if request.identity_sha256 != self.provider_identity_sha256:
            raise ValueError("metasyn_hosted_intent_request_identity_mismatch")
        if (
            request.config_sha256 != self.provider_config_sha256
            or request.full_acceptance_schema_sha256
            != self.full_acceptance_schema_sha256
            or request.compiled_schema.original_schema_sha256
            != self.provider_schema_sha256
            or request.compiled_schema.wire_schema_sha256 != self.wire_schema_sha256
        ):
            raise ValueError("metasyn_hosted_intent_request_schema_mismatch")
        if (
            self.prompt_sha256 != _sha256_text(request.prompt)
            or self.base_system_sha256 != request.base_system_sha256
            or self.model_system_sha256 != request.model_system_sha256
            or self.base_prompt_sha256 != request.base_prompt_sha256
            or self.model_prompt_sha256 != request.model_prompt_sha256
        ):
            raise ValueError("metasyn_hosted_intent_request_prompt_mismatch")
        if (
            self.schema_kind != request.schema_kind
            or self.effect_kind != request.effect_kind
            or self.transport_mode != request.transport_mode
            or self.transport_policy_binding_sha256
            != request.transport_policy_binding_sha256
            or self.structured_grammar_enforced_by_provider
            != request.structured_grammar_enforced_by_provider
            or self.output_format_present_in_call
            != request.output_format_present_in_call
            or self.expected_wire_call_sha256 != request.expected_wire_call_sha256
            or self.request_cost_ceiling_usd_micros
            != _usd_micros(request.cost_ceiling.request_cost_ceiling_usd)
        ):
            raise ValueError("metasyn_hosted_intent_transport_alias_mismatch")
        expected_attempt_id = hash_canonical(
            {
                "request_sha256": self.request_sha256,
                "permitted_call_attempts": 1,
                "application_retries_permitted": 0,
                "sdk_retries_permitted": 0,
            }
        )
        if self.attempt_id != expected_attempt_id:
            raise ValueError("metasyn_hosted_attempt_id_mismatch")
        payload = self.model_dump(mode="json", exclude={"attempt_intent_sha256"})
        if self.attempt_intent_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_hosted_attempt_intent_hash_mismatch")
        return self


class MetaSynHostedCallReceiptV1(ContractModel):
    call_receipt_version: Literal["metasyn-bounded-hosted-call-receipt-v2"] = CALL_RECEIPT_VERSION
    runtime_version: Literal["metasyn-bounded-hosted-anthropic-runtime-v2"] = RUNTIME_VERSION
    terminal: Literal[True] = True
    response_observed: Literal[True] = True
    execution_bundle_sha256: str
    request_key: str
    stage: CallStage
    row_ordinal: int | None
    row_context_sha256: str | None
    candidate_index: int | None
    attempt_id: str
    attempt_intent_sha256: str
    request_sha256: str
    provider_identity_sha256: str
    provider_config_sha256: str
    cost_authorization_sha256: str
    schema_bundle_sha256: str
    provider_schema_sha256: str
    wire_schema_sha256: str
    full_acceptance_schema_sha256: str
    compiled_schema_sha256: str
    schema_kind: Literal["inventory", "packet"]
    effect_kind: AnthropicEffectKind | None
    transport_mode: AnthropicTransportMode
    transport_policy_binding_sha256: str
    structured_grammar_enforced_by_provider: bool
    output_format_present_in_call: bool
    base_system_sha256: str
    model_system_sha256: str
    base_prompt_sha256: str
    model_prompt_sha256: str
    expected_wire_call_sha256: str
    request_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    validation_status: CallValidationStatus
    provider_result: AnthropicBoundedResultV1
    provider_result_sha256: str
    accepted_payload: dict[str, Any] | None
    accepted_payload_sha256: str | None
    adapter_validation_receipt: dict[str, Any] | None
    adapter_validation_receipt_sha256: str | None
    terminal_error: str | None
    usage: MetaSynHostedUsageV1
    cost: MetaSynHostedCostV1
    provider_call_attempts: Literal[1] = 1
    application_retries: Literal[0] = 0
    sdk_retries: Literal[0] = 0
    receipt_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "row_context_sha256",
        "attempt_id",
        "attempt_intent_sha256",
        "request_sha256",
        "provider_identity_sha256",
        "provider_config_sha256",
        "cost_authorization_sha256",
        "schema_bundle_sha256",
        "provider_schema_sha256",
        "wire_schema_sha256",
        "full_acceptance_schema_sha256",
        "compiled_schema_sha256",
        "transport_policy_binding_sha256",
        "base_system_sha256",
        "model_system_sha256",
        "base_prompt_sha256",
        "model_prompt_sha256",
        "expected_wire_call_sha256",
        "provider_result_sha256",
        "accepted_payload_sha256",
        "adapter_validation_receipt_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_hosted_receipt_hash_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> MetaSynHostedCallReceiptV1:
        expected_schema_kind = "inventory" if self.stage == "inventory" else (
            "packet" if self.stage == "packet" else self.schema_kind
        )
        if (
            self.schema_kind != expected_schema_kind
            or (self.schema_kind == "inventory") != (self.effect_kind is None)
            or (self.transport_mode == "structured_json_schema")
            != (self.schema_kind == "inventory")
            or self.structured_grammar_enforced_by_provider
            != (self.transport_mode == "structured_json_schema")
            or self.output_format_present_in_call
            != (self.transport_mode == "structured_json_schema")
        ):
            raise ValueError("metasyn_hosted_receipt_transport_shape_invalid")
        if self.provider_result.result_sha256 != self.provider_result_sha256:
            raise ValueError("metasyn_hosted_provider_result_self_hash_mismatch")
        if (
            self.provider_result.request_sha256 != self.request_sha256
            or self.provider_result.identity_sha256 != self.provider_identity_sha256
            or self.provider_result.config_sha256 != self.provider_config_sha256
            or self.provider_result.compiled_schema_sha256 != self.compiled_schema_sha256
            or self.provider_result.original_schema_sha256
            != self.provider_schema_sha256
            or self.provider_result.wire_schema_sha256 != self.wire_schema_sha256
            or self.provider_result.full_acceptance_schema_sha256
            != self.full_acceptance_schema_sha256
            or self.provider_result.schema_kind != self.schema_kind
            or self.provider_result.effect_kind != self.effect_kind
            or self.provider_result.transport_mode != self.transport_mode
            or self.provider_result.structured_grammar_enforced_by_provider
            != self.structured_grammar_enforced_by_provider
            or self.provider_result.output_format_present_in_call
            != self.output_format_present_in_call
            or self.provider_result.model_system_sha256 != self.model_system_sha256
            or self.provider_result.model_prompt_sha256 != self.model_prompt_sha256
            or self.provider_result.wire_call_sha256
            != self.expected_wire_call_sha256
            or _usd_micros(
                self.provider_result.cost.request_cost_ceiling_usd
            )
            != self.request_cost_ceiling_usd_micros
            or self.cost.request_ceiling_usd_micros
            != self.request_cost_ceiling_usd_micros
        ):
            raise ValueError("metasyn_hosted_provider_result_binding_mismatch")
        expected_payload_sha = (
            hash_canonical(self.accepted_payload) if self.accepted_payload is not None else None
        )
        if self.accepted_payload_sha256 != expected_payload_sha:
            raise ValueError("metasyn_hosted_accepted_payload_hash_mismatch")
        expected_adapter_sha = (
            hash_canonical(self.adapter_validation_receipt)
            if self.adapter_validation_receipt is not None
            else None
        )
        if self.adapter_validation_receipt_sha256 != expected_adapter_sha:
            raise ValueError("metasyn_hosted_adapter_receipt_hash_mismatch")
        valid = self.validation_status.startswith("inventory_valid") or (
            self.validation_status
            in {
                "preflight_fixture_valid",
                "packet_completed",
                "packet_unable_to_complete",
            }
        )
        if valid != (self.accepted_payload is not None):
            raise ValueError("metasyn_hosted_accepted_payload_presence_mismatch")
        if (self.terminal_error is None) != valid:
            raise ValueError("metasyn_hosted_terminal_error_presence_mismatch")
        if self.stage == "preflight" and self.adapter_validation_receipt is not None:
            raise ValueError("metasyn_hosted_preflight_adapter_receipt_forbidden")
        if self.stage in {"inventory", "packet"} and valid != (
            self.adapter_validation_receipt is not None
        ):
            raise ValueError("metasyn_hosted_adapter_receipt_presence_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if self.receipt_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_hosted_call_receipt_hash_mismatch")
        return self


class MetaSynHostedAmbiguityIncidentV1(ContractModel):
    incident_version: Literal["metasyn-bounded-hosted-ambiguity-incident-v2"] = INCIDENT_VERSION
    runtime_version: Literal["metasyn-bounded-hosted-anthropic-runtime-v2"] = RUNTIME_VERSION
    status: Literal["terminal_ambiguous_attempt_poison"] = "terminal_ambiguous_attempt_poison"
    incident_kind: IncidentKind
    execution_bundle_sha256: str
    request_key: str
    stage: CallStage
    row_ordinal: int | None
    row_context_sha256: str | None
    candidate_index: int | None
    attempt_id: str
    attempt_intent_sha256: str
    request_sha256: str
    provider_identity_sha256: str
    provider_config_sha256: str
    cost_authorization_sha256: str
    schema_kind: Literal["inventory", "packet"]
    effect_kind: AnthropicEffectKind | None
    transport_mode: AnthropicTransportMode
    transport_policy_binding_sha256: str
    structured_grammar_enforced_by_provider: bool
    output_format_present_in_call: bool
    base_system_sha256: str
    model_system_sha256: str
    base_prompt_sha256: str
    model_prompt_sha256: str
    expected_wire_call_sha256: str
    request_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    response_observed: Literal[False] = False
    possible_provider_call_attempts: Literal[1] = 1
    application_retries_permitted: Literal[0] = 0
    sdk_retries_permitted: Literal[0] = 0
    retry_this_request_permitted: Literal[False] = False
    incident_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "row_context_sha256",
        "attempt_id",
        "attempt_intent_sha256",
        "request_sha256",
        "provider_identity_sha256",
        "provider_config_sha256",
        "cost_authorization_sha256",
        "transport_policy_binding_sha256",
        "base_system_sha256",
        "model_system_sha256",
        "base_prompt_sha256",
        "model_prompt_sha256",
        "expected_wire_call_sha256",
        "incident_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_hosted_incident_hash_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_incident(self) -> MetaSynHostedAmbiguityIncidentV1:
        if (
            (self.schema_kind == "inventory") != (self.effect_kind is None)
            or (self.transport_mode == "structured_json_schema")
            != (self.schema_kind == "inventory")
            or self.structured_grammar_enforced_by_provider
            != (self.transport_mode == "structured_json_schema")
            or self.output_format_present_in_call
            != (self.transport_mode == "structured_json_schema")
        ):
            raise ValueError("metasyn_hosted_incident_transport_shape_invalid")
        payload = self.model_dump(mode="json", exclude={"incident_sha256"})
        if self.incident_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_hosted_incident_hash_mismatch")
        return self


def _request_surface(
    *,
    row: MetaSynBoundedRowContextV1,
    stage: Literal["inventory", "packet"],
    inventory_receipt: MetaSynInventoryValidationReceiptV1 | None = None,
    candidate_index: int | None = None,
) -> tuple[str, dict[str, Any], MetaSynPacketCallV1 | None]:
    if stage == "inventory":
        if inventory_receipt is not None or candidate_index is not None:
            raise MetaSynHostedRuntimeError("metasyn_hosted_inventory_context_invalid")
        bundle = inventory_schema_bundle_v2(
            exposed_line_ids=row.source_row.projection.exposed_line_ids,
            allowed_outcomes=row.allowed_outcomes,
        )
        return row.inventory_prompt, bundle, None
    if inventory_receipt is None or candidate_index is None:
        raise MetaSynHostedRuntimeError("metasyn_hosted_packet_context_missing")
    packet_call = freeze_metasyn_packet_call(
        row=row,
        inventory_receipt=inventory_receipt,
        candidate_index=candidate_index,
    )
    bundle = packet_schema_bundle_v2(
        candidate=packet_call.candidate,
        exposed_line_ids=row.source_row.projection.exposed_line_ids,
        source_locator=row.source_locator,
        allowed_outcomes=row.allowed_outcomes,
        allowed_moderators=row.allowed_moderators,
        allowed_sections=row.allowed_sections,
        outcome_positive_directions=row.outcome_positive_directions,
    )
    return packet_call.rendered_prompt, bundle, packet_call


def _preflight_fixture(spec: Mapping[str, Any]) -> dict[str, Any]:
    provider_schema = spec.get("provider_schema")
    valid_example = spec.get("valid_example")
    full_acceptance_schema = spec.get("full_acceptance_schema")
    if (
        not isinstance(provider_schema, Mapping)
        or not isinstance(valid_example, Mapping)
        or not isinstance(full_acceptance_schema, Mapping)
    ):
        raise MetaSynHostedRuntimeError("metasyn_hosted_preflight_fixture_contract_invalid")
    projected = project_anthropic_preflight_fixture(
        value=valid_example,
        original_schema=provider_schema,
    )
    if not isinstance(projected, dict):
        raise MetaSynHostedRuntimeError("metasyn_hosted_preflight_fixture_not_object")
    try:
        validate_raw_payload_against_schema_v2(
            projected,
            schema=full_acceptance_schema,
        )
    except (NativeBoundedGenerationError, ValueError) as exc:
        raise MetaSynHostedRuntimeError(
            "metasyn_hosted_materialized_preflight_full_schema_invalid"
        ) from exc
    return projected


def _preflight_prompt(spec: Mapping[str, Any]) -> str:
    projected = _preflight_fixture(spec)
    fixture = json.dumps(projected, sort_keys=True, separators=(",", ":"))
    fixture_sha256 = hash_canonical(projected)
    return (
        "This is a source-free structured-output compatibility probe. Emit exactly the "
        "following synthetic JSON fixture, preserving every key, JSON type, string, and "
        "number. Do not add commentary.\n"
        f"PROJECTED_FIXTURE_SHA256={fixture_sha256}\nFIXTURE_JSON={fixture}"
    )


@lru_cache(maxsize=512)
def _compile_provider_schema_cached(
    original_schema_json: str, full_acceptance_schema_sha256: str
) -> AnthropicCompiledSchemaV1:
    """Compile one canonical schema identity once per process.

    The compiler and SDK identities are bound by the runtime pipeline and provider
    identity.  Replaying an unchanged request therefore needs one compilation, not
    another expensive Draft/SDK traversal for every aggregate that references it.
    """

    parsed = json.loads(original_schema_json)
    if not isinstance(parsed, dict):  # pragma: no cover - caller invariant
        raise MetaSynHostedRuntimeError("metasyn_hosted_cached_schema_not_object")
    return compile_anthropic_bounded_schema(
        original_schema=parsed,
        full_acceptance_schema_sha256=full_acceptance_schema_sha256,
    )


@lru_cache(maxsize=512)
def _schema_bundle_binding_cached(schema_bundle_json: str) -> dict[str, Any]:
    parsed = json.loads(schema_bundle_json)
    if not isinstance(parsed, dict):  # pragma: no cover - caller invariant
        raise MetaSynHostedRuntimeError("metasyn_hosted_cached_bundle_not_object")
    return schema_bundle_receipt_binding_v2(parsed)


def _schema_bundle_binding(schema_bundle: Mapping[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(
        schema_bundle,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return _schema_bundle_binding_cached(serialized)


def _max_output_tokens(config: MetaSynHostedRuntimeConfigV1, stage: CallStage) -> int:
    if stage == "preflight":
        return config.preflight_max_output_tokens
    if stage == "inventory":
        return config.inventory_max_output_tokens
    return config.packet_max_output_tokens


def _schema_transport_identity(
    *, stage: CallStage, schema_bundle: Mapping[str, Any]
) -> tuple[AnthropicSchemaKind, AnthropicEffectKind | None]:
    kind = schema_bundle.get("kind")
    if kind not in {"inventory", "packet"}:
        raise MetaSynHostedRuntimeError("metasyn_hosted_schema_kind_invalid")
    if stage == "inventory" and kind != "inventory":
        raise MetaSynHostedRuntimeError("metasyn_hosted_inventory_mode_drift")
    if stage == "packet" and kind != "packet":
        raise MetaSynHostedRuntimeError("metasyn_hosted_packet_mode_drift")
    if kind == "inventory":
        return "inventory", None
    context_binding = schema_bundle.get("context_binding")
    candidate = (
        context_binding.get("candidate")
        if isinstance(context_binding, Mapping)
        else None
    )
    effect_kind = candidate.get("effect_kind") if isinstance(candidate, Mapping) else None
    allowed_effect_kinds = {
        "binary_group_statistics",
        "continuous_group_statistics",
        "direct_confidence_interval",
        "direct_standard_error",
        "direct_variance",
    }
    if effect_kind not in allowed_effect_kinds:
        raise MetaSynHostedRuntimeError("metasyn_hosted_packet_effect_kind_invalid")
    return "packet", effect_kind


def _freeze_provider_request(
    *,
    bundle: MetaSynHostedExecutionBundleV1,
    stage: CallStage,
    request_key: str,
    prompt: str,
    schema_bundle: Mapping[str, Any],
) -> tuple[AnthropicCompiledSchemaV1, AnthropicBoundedRequestV1]:
    binding = _schema_bundle_binding(schema_bundle)
    schema_kind, effect_kind = _schema_transport_identity(
        stage=stage,
        schema_bundle=schema_bundle,
    )
    original_schema_json = json.dumps(
        schema_bundle["provider_schema"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    compiled = _compile_provider_schema_cached(
        original_schema_json,
        binding["full_acceptance_schema_sha256"],
    )
    request = freeze_anthropic_bounded_request(
        operation=stage,
        request_key=request_key,
        prompt=prompt,
        system=bundle.runtime_config.system_prompt,
        compiled_schema=compiled,
        config=bundle.anthropic_config,
        schema_kind=schema_kind,
        effect_kind=effect_kind,
        max_output_tokens=_max_output_tokens(bundle.runtime_config, stage),
        identity=bundle.provider_identity,
    )
    return compiled, request


def freeze_metasyn_hosted_attempt_intent(
    *,
    execution_bundle: MetaSynHostedExecutionBundleV1,
    request_key: str,
    stage: CallStage,
    prompt: str,
    schema_bundle: Mapping[str, Any],
    cost_authorization_sha256: str,
    row_ordinal: int | None = None,
    row: MetaSynBoundedRowContextV1 | None = None,
    inventory_receipt: MetaSynInventoryValidationReceiptV1 | None = None,
    candidate: NativeCandidateDescriptor | None = None,
) -> MetaSynHostedAttemptIntentV1:
    if not SHA256_RE.fullmatch(cost_authorization_sha256):
        raise MetaSynHostedRuntimeError(
            "metasyn_hosted_intent_cost_authorization_hash_invalid"
        )
    bundle = _coerce_metasyn_hosted_execution_bundle(execution_bundle)
    binding = _schema_bundle_binding(schema_bundle)
    compiled, request = _freeze_provider_request(
        bundle=bundle,
        stage=stage,
        request_key=request_key,
        prompt=prompt,
        schema_bundle=schema_bundle,
    )
    source = stage in {"inventory", "packet"}
    if source != (row_ordinal is not None and row is not None):
        raise MetaSynHostedRuntimeError("metasyn_hosted_intent_row_context_invalid")
    packet = stage == "packet"
    if packet != (inventory_receipt is not None and candidate is not None):
        raise MetaSynHostedRuntimeError("metasyn_hosted_intent_packet_context_invalid")
    attempt_seed = {
        "request_sha256": request.request_sha256,
        "permitted_call_attempts": 1,
        "application_retries_permitted": 0,
        "sdk_retries_permitted": 0,
    }
    payload: dict[str, Any] = {
        "attempt_intent_version": ATTEMPT_INTENT_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "status": "durable_pre_call_intent_frozen",
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
        "request_key": request_key,
        "stage": stage,
        "row_ordinal": row_ordinal,
        "row_context_sha256": row.row_context_sha256 if row is not None else None,
        "candidate_index": (candidate.candidate_index if candidate is not None else None),
        "candidate_sha256": (candidate.descriptor_sha256 if candidate is not None else None),
        "inventory_validation_receipt_sha256": (
            inventory_receipt.receipt_sha256 if inventory_receipt is not None else None
        ),
        "prompt_sha256": _sha256_text(prompt),
        "base_system_sha256": request.base_system_sha256,
        "model_system_sha256": request.model_system_sha256,
        "base_prompt_sha256": request.base_prompt_sha256,
        "model_prompt_sha256": request.model_prompt_sha256,
        "schema_bundle_sha256": binding["schema_bundle_sha256"],
        "provider_schema_sha256": binding["provider_schema_sha256"],
        "wire_schema_sha256": request.compiled_schema.wire_schema_sha256,
        "full_acceptance_schema_sha256": binding["full_acceptance_schema_sha256"],
        "compiled_schema_sha256": compiled.compiled_schema_sha256,
        "request": request.model_dump(mode="json"),
        "request_sha256": request.request_sha256,
        "provider_identity_sha256": bundle.provider_identity_sha256,
        "provider_config_sha256": bundle.anthropic_config_sha256,
        "cost_authorization_sha256": cost_authorization_sha256,
        "schema_kind": request.schema_kind,
        "effect_kind": request.effect_kind,
        "transport_mode": request.transport_mode,
        "transport_policy_binding_sha256": (
            request.transport_policy_binding_sha256
        ),
        "structured_grammar_enforced_by_provider": (
            request.structured_grammar_enforced_by_provider
        ),
        "output_format_present_in_call": request.output_format_present_in_call,
        "expected_wire_call_sha256": request.expected_wire_call_sha256,
        "request_cost_ceiling_usd_micros": _usd_micros(
            request.cost_ceiling.request_cost_ceiling_usd
        ),
        "permitted_call_attempts": 1,
        "application_retries_permitted": 0,
        "sdk_retries_permitted": 0,
        "orphan_or_ambiguous_attempt_is_terminal": True,
        "attempt_id": hash_canonical(attempt_seed),
    }
    return MetaSynHostedAttemptIntentV1.model_validate(
        {**payload, "attempt_intent_sha256": hash_canonical(payload)}
    )


def freeze_metasyn_hosted_ambiguity_incident(
    *, intent: MetaSynHostedAttemptIntentV1, incident_kind: IncidentKind
) -> MetaSynHostedAmbiguityIncidentV1:
    intent = MetaSynHostedAttemptIntentV1.model_validate(intent)
    payload: dict[str, Any] = {
        "incident_version": INCIDENT_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "status": "terminal_ambiguous_attempt_poison",
        "incident_kind": incident_kind,
        "execution_bundle_sha256": intent.execution_bundle_sha256,
        "request_key": intent.request_key,
        "stage": intent.stage,
        "row_ordinal": intent.row_ordinal,
        "row_context_sha256": intent.row_context_sha256,
        "candidate_index": intent.candidate_index,
        "attempt_id": intent.attempt_id,
        "attempt_intent_sha256": intent.attempt_intent_sha256,
        "request_sha256": intent.request_sha256,
        "provider_identity_sha256": intent.provider_identity_sha256,
        "provider_config_sha256": intent.provider_config_sha256,
        "cost_authorization_sha256": intent.cost_authorization_sha256,
        "schema_kind": intent.schema_kind,
        "effect_kind": intent.effect_kind,
        "transport_mode": intent.transport_mode,
        "transport_policy_binding_sha256": (
            intent.transport_policy_binding_sha256
        ),
        "structured_grammar_enforced_by_provider": (
            intent.structured_grammar_enforced_by_provider
        ),
        "output_format_present_in_call": intent.output_format_present_in_call,
        "base_system_sha256": intent.base_system_sha256,
        "model_system_sha256": intent.model_system_sha256,
        "base_prompt_sha256": intent.base_prompt_sha256,
        "model_prompt_sha256": intent.model_prompt_sha256,
        "expected_wire_call_sha256": intent.expected_wire_call_sha256,
        "request_cost_ceiling_usd_micros": (
            intent.request_cost_ceiling_usd_micros
        ),
        "response_observed": False,
        "possible_provider_call_attempts": 1,
        "application_retries_permitted": 0,
        "sdk_retries_permitted": 0,
        "retry_this_request_permitted": False,
    }
    return MetaSynHostedAmbiguityIncidentV1.model_validate(
        {**payload, "incident_sha256": hash_canonical(payload)}
    )


def _provider_failure_code(result: AnthropicBoundedResultV1) -> str:
    failure = result.failure
    return failure.code if failure is not None else result.outcome


def _ordered_response_schema_failure(
    *,
    raw: Mapping[str, Any],
    intent: MetaSynHostedAttemptIntentV1,
    full_acceptance_schema: Mapping[str, Any],
) -> Literal[
    "wire_schema_invalid",
    "original_provider_schema_invalid",
    "full_schema_invalid",
] | None:
    """Replay wire → original → full validation without modifying the response."""

    try:
        request = AnthropicBoundedRequestV1.model_validate(intent.request)
    except ValueError as exc:
        raise MetaSynHostedRuntimeError(
            "metasyn_hosted_intent_request_revalidation_failed"
        ) from exc
    schemas = (
        ("wire_schema_invalid", request.compiled_schema.wire_schema),
        (
            "original_provider_schema_invalid",
            request.compiled_schema.original_schema,
        ),
        ("full_schema_invalid", full_acceptance_schema),
    )
    for status, schema in schemas:
        try:
            validator_for(schema)(schema).validate(raw)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
                raise
            return status
    return None


def freeze_metasyn_hosted_call_receipt(
    *,
    execution_bundle: MetaSynHostedExecutionBundleV1,
    intent: MetaSynHostedAttemptIntentV1,
    provider_result: AnthropicBoundedResultV1,
    schema_bundle: Mapping[str, Any],
    row: MetaSynBoundedRowContextV1 | None = None,
    inventory_receipt: MetaSynInventoryValidationReceiptV1 | None = None,
    packet_call: MetaSynPacketCallV1 | None = None,
    preflight_fixture: Mapping[str, Any] | None = None,
) -> MetaSynHostedCallReceiptV1:
    bundle = _coerce_metasyn_hosted_execution_bundle(execution_bundle)
    canonical_intent = MetaSynHostedAttemptIntentV1.model_validate(intent)
    result = AnthropicBoundedResultV1.model_validate(provider_result)
    request = AnthropicBoundedRequestV1.model_validate(canonical_intent.request)
    _, replayed_wire_call_sha256 = freeze_anthropic_wire_call_surface(
        request=request,
        config=bundle.anthropic_config,
    )
    binding = _schema_bundle_binding(schema_bundle)
    if (
        canonical_intent.execution_bundle_sha256 != bundle.execution_bundle_sha256
        or canonical_intent.schema_bundle_sha256 != binding["schema_bundle_sha256"]
        or canonical_intent.provider_schema_sha256 != binding["provider_schema_sha256"]
        or canonical_intent.full_acceptance_schema_sha256
        != binding["full_acceptance_schema_sha256"]
        or result.request_sha256 != canonical_intent.request_sha256
        or replayed_wire_call_sha256 != canonical_intent.expected_wire_call_sha256
        or result.wire_call_sha256 != replayed_wire_call_sha256
        or result.config_sha256 != bundle.anthropic_config_sha256
        or result.identity_sha256 != bundle.provider_identity_sha256
        or _usd_micros(result.cost.request_cost_ceiling_usd)
        != canonical_intent.request_cost_ceiling_usd_micros
    ):
        raise MetaSynHostedRuntimeError("metasyn_hosted_receipt_context_mismatch")

    status: CallValidationStatus
    accepted: dict[str, Any] | None = None
    adapter_receipt: ContractModel | None = None
    terminal_error: str | None = None
    if result.outcome != "completed":
        status = "provider_failure"
        terminal_error = f"provider_failure:{_provider_failure_code(result)}"
    elif not isinstance(result.parsed_json, dict):
        status = "response_json_missing"
        terminal_error = "response_json_missing"
    else:
        raw = result.parsed_json
        ordered_schema_failure = _ordered_response_schema_failure(
            raw=raw,
            intent=canonical_intent,
            full_acceptance_schema=schema_bundle["full_acceptance_schema"],
        )
        if ordered_schema_failure is not None:
            status = ordered_schema_failure
            terminal_error = ordered_schema_failure
        else:
            if canonical_intent.stage == "preflight":
                if preflight_fixture is None:
                    raise MetaSynHostedRuntimeError("metasyn_hosted_preflight_fixture_missing")
                if raw != dict(preflight_fixture):
                    status = "preflight_fixture_mismatch"
                    terminal_error = "preflight_fixture_mismatch"
                else:
                    status = "preflight_fixture_valid"
                    accepted = dict(raw)
            elif canonical_intent.stage == "inventory":
                if row is None:
                    raise MetaSynHostedRuntimeError("metasyn_hosted_inventory_row_missing")
                try:
                    v2 = validate_inventory_for_row_v2(
                        raw,
                        exposed_line_ids=(row.source_row.projection.exposed_line_ids),
                        allowed_outcomes=row.allowed_outcomes,
                    )
                    frozen = freeze_metasyn_inventory_validation_receipt(row=row, value=raw)
                    if frozen.inventory != v2:
                        raise MetaSynHostedRuntimeError("metasyn_hosted_inventory_v2_v1_mismatch")
                except (
                    MetaSynBoundedAdapterError,
                    NativeBoundedGenerationError,
                    ValidationError,
                    ValueError,
                ):
                    status = "inventory_contract_invalid"
                    terminal_error = "inventory_contract_invalid"
                else:
                    status_by_adapter: dict[str, CallValidationStatus] = {
                        "candidates_authorized": "inventory_valid_candidates",
                        "no_candidate_non_authorizing": (
                            "inventory_valid_no_candidate_non_authorizing"
                        ),
                        "capacity_or_uncertainty_non_authorizing": (
                            "inventory_valid_capacity_or_uncertainty_non_authorizing"
                        ),
                    }
                    status = status_by_adapter[frozen.status]
                    accepted = v2.model_dump(mode="json")
                    adapter_receipt = frozen
            else:
                if row is None or inventory_receipt is None or packet_call is None:
                    raise MetaSynHostedRuntimeError(
                        "metasyn_hosted_packet_validation_context_missing"
                    )
                try:
                    v2_packet = validate_packet_for_candidate_v2(
                        raw,
                        candidate=packet_call.candidate,
                        exposed_line_ids=(row.source_row.projection.exposed_line_ids),
                        source_locator=row.source_locator,
                        allowed_outcomes=row.allowed_outcomes,
                        allowed_moderators=row.allowed_moderators,
                        allowed_sections=row.allowed_sections,
                        outcome_positive_directions=(row.outcome_positive_directions),
                    )
                    frozen_packet = freeze_metasyn_packet_validation_receipt(
                        call=packet_call,
                        row=row,
                        inventory_receipt=inventory_receipt,
                        value=raw,
                    )
                    if frozen_packet.packet_payload != v2_packet.model_dump(mode="json"):
                        raise MetaSynHostedRuntimeError("metasyn_hosted_packet_v2_v1_mismatch")
                except MetaSynBoundedAdapterError as exc:
                    if str(exc).startswith("metasyn_packet_quote_"):
                        status = "packet_source_grounding_invalid"
                        terminal_error = "packet_source_grounding_invalid"
                    else:
                        status = "packet_contract_invalid"
                        terminal_error = "packet_contract_invalid"
                except (
                    NativeBoundedGenerationError,
                    ValidationError,
                    ValueError,
                ):
                    status = "packet_contract_invalid"
                    terminal_error = "packet_contract_invalid"
                else:
                    status = (
                        "packet_completed"
                        if frozen_packet.packet_status == "completed"
                        else "packet_unable_to_complete"
                    )
                    accepted = v2_packet.model_dump(mode="json")
                    adapter_receipt = frozen_packet

    usage, cost = _provider_usage_cost(result)
    adapter_payload = (
        adapter_receipt.model_dump(mode="json") if adapter_receipt is not None else None
    )
    payload: dict[str, Any] = {
        "call_receipt_version": CALL_RECEIPT_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "terminal": True,
        "response_observed": True,
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "request_key": canonical_intent.request_key,
        "stage": canonical_intent.stage,
        "row_ordinal": canonical_intent.row_ordinal,
        "row_context_sha256": canonical_intent.row_context_sha256,
        "candidate_index": canonical_intent.candidate_index,
        "attempt_id": canonical_intent.attempt_id,
        "attempt_intent_sha256": canonical_intent.attempt_intent_sha256,
        "request_sha256": canonical_intent.request_sha256,
        "provider_identity_sha256": canonical_intent.provider_identity_sha256,
        "provider_config_sha256": canonical_intent.provider_config_sha256,
        "cost_authorization_sha256": canonical_intent.cost_authorization_sha256,
        "schema_bundle_sha256": canonical_intent.schema_bundle_sha256,
        "provider_schema_sha256": canonical_intent.provider_schema_sha256,
        "wire_schema_sha256": canonical_intent.wire_schema_sha256,
        "full_acceptance_schema_sha256": (canonical_intent.full_acceptance_schema_sha256),
        "compiled_schema_sha256": canonical_intent.compiled_schema_sha256,
        "schema_kind": canonical_intent.schema_kind,
        "effect_kind": canonical_intent.effect_kind,
        "transport_mode": canonical_intent.transport_mode,
        "transport_policy_binding_sha256": (
            canonical_intent.transport_policy_binding_sha256
        ),
        "structured_grammar_enforced_by_provider": (
            canonical_intent.structured_grammar_enforced_by_provider
        ),
        "output_format_present_in_call": (
            canonical_intent.output_format_present_in_call
        ),
        "base_system_sha256": canonical_intent.base_system_sha256,
        "model_system_sha256": canonical_intent.model_system_sha256,
        "base_prompt_sha256": canonical_intent.base_prompt_sha256,
        "model_prompt_sha256": canonical_intent.model_prompt_sha256,
        "expected_wire_call_sha256": canonical_intent.expected_wire_call_sha256,
        "request_cost_ceiling_usd_micros": (
            canonical_intent.request_cost_ceiling_usd_micros
        ),
        "validation_status": status,
        "provider_result": result,
        "provider_result_sha256": result.result_sha256,
        "accepted_payload": accepted,
        "accepted_payload_sha256": (hash_canonical(accepted) if accepted is not None else None),
        "adapter_validation_receipt": adapter_payload,
        "adapter_validation_receipt_sha256": (
            hash_canonical(adapter_payload) if adapter_payload is not None else None
        ),
        "terminal_error": terminal_error,
        "usage": usage,
        "cost": cost,
        "provider_call_attempts": 1,
        "application_retries": 0,
        "sdk_retries": 0,
    }
    return MetaSynHostedCallReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def _outcome_paths(workspace: Path, request_key: str) -> tuple[Path, Path, Path]:
    paths = metasyn_hosted_runtime_paths(workspace)
    return (
        paths["intents"] / f"{request_key}.json",
        paths["receipts"] / f"{request_key}.json",
        paths["incidents"] / f"{request_key}.json",
    )


def _prepare_call_directories(workspace: Path) -> None:
    paths = metasyn_hosted_runtime_paths(workspace)
    for name in ("intents", "receipts", "incidents", "row_results"):
        path = paths[name]
        if path.exists() and (path.is_symlink() or not path.is_dir()):
            raise MetaSynHostedRuntimeError("metasyn_hosted_call_directory_topology_invalid")
        path.mkdir(parents=True, exist_ok=True)


def _current_intent_count(workspace: Path) -> int:
    directory = metasyn_hosted_runtime_paths(workspace)["intents"]
    return len(list(directory.glob("*.json"))) if directory.is_dir() else 0


def _current_intent_cost_ceiling_micros(workspace: Path) -> int:
    directory = metasyn_hosted_runtime_paths(workspace)["intents"]
    if not directory.is_dir():
        return 0
    total = 0
    for path in sorted(directory.glob("*.json")):
        intent = MetaSynHostedAttemptIntentV1.model_validate(_read_json_object(path))
        if path.stem != intent.request_key:
            raise MetaSynHostedRuntimeError(
                "metasyn_hosted_prior_intent_filename_binding_mismatch"
            )
        total += intent.request_cost_ceiling_usd_micros
    return total


_BUDGET_AUTHORITATIVE_INTENT_CACHE: dict[
    tuple[str, str, str, str], MetaSynHostedAttemptIntentV1
] = {}


def _authoritatively_replay_budget_intent(
    *,
    workspace: Path,
    bundle: MetaSynHostedExecutionBundleV1,
    authorization: MetaSynHostedCostAuthorizationReceiptV1,
    intent: MetaSynHostedAttemptIntentV1,
) -> tuple[
    Literal["source_free_preflight", "exact_inventory", "packet_slot_ceiling"],
    AnthropicBoundedRequestV1,
]:
    """Rebuild an intent from the frozen roster instead of trusting its aliases."""

    canonical = MetaSynHostedAttemptIntentV1.model_validate(
        intent.model_dump(mode="json")
    )
    if (
        canonical.execution_bundle_sha256 != bundle.execution_bundle_sha256
        or canonical.cost_authorization_sha256 != authorization.authorization_sha256
    ):
        raise MetaSynHostedRuntimeError(
            "metasyn_hosted_budget_intent_context_mismatch"
        )

    preflight_keys = {f"preflight-{index:02d}": index for index in range(8)}
    inventory_keys = {
        f"row-{index:02d}-inventory": index for index in range(PUBLICATION_COUNT)
    }
    packet_keys = {
        f"row-{row_index:02d}-packet-{candidate_index:02d}": (
            row_index,
            candidate_index,
        )
        for row_index in range(PUBLICATION_COUNT)
        for candidate_index in range(1, MAX_CANDIDATES_PER_PUBLICATION + 1)
    }
    context_sha256 = hash_canonical(None)
    if canonical.request_key in preflight_keys:
        group: Literal[
            "source_free_preflight", "exact_inventory", "packet_slot_ceiling"
        ] = "source_free_preflight"
    elif canonical.request_key in inventory_keys:
        group = "exact_inventory"
    elif canonical.request_key in packet_keys:
        group = "packet_slot_ceiling"
        row_ordinal, _ = packet_keys[canonical.request_key]
        inventory_receipt_path = (
            metasyn_hosted_runtime_paths(workspace)["receipts"]
            / f"row-{row_ordinal:02d}-inventory.json"
        )
        if not inventory_receipt_path.is_file() or inventory_receipt_path.is_symlink():
            raise MetaSynHostedRuntimeError(
                "metasyn_hosted_packet_budget_inventory_receipt_missing"
            )
        context_sha256 = sha256_file(inventory_receipt_path)
    else:
        raise MetaSynHostedRuntimeError(
            "metasyn_hosted_budget_intent_not_in_authorized_roster"
        )

    cache_key = (
        bundle.execution_bundle_sha256,
        authorization.authorization_sha256,
        canonical.attempt_intent_sha256,
        context_sha256,
    )
    expected = _BUDGET_AUTHORITATIVE_INTENT_CACHE.get(cache_key)
    if expected is None:
        if group == "source_free_preflight":
            index = preflight_keys[canonical.request_key]
            spec = synthetic_schema_v2_preflight_specs()[index]
            expected = freeze_metasyn_hosted_attempt_intent(
                execution_bundle=bundle,
                request_key=canonical.request_key,
                stage="preflight",
                prompt=_preflight_prompt(spec),
                schema_bundle=_preflight_schema_bundle(spec),
                cost_authorization_sha256=authorization.authorization_sha256,
            )
        elif group == "exact_inventory":
            row_ordinal = inventory_keys[canonical.request_key]
            row = bundle.adapter_bundle.row_contexts[row_ordinal]
            prompt, schema_bundle, _ = _request_surface(row=row, stage="inventory")
            expected = freeze_metasyn_hosted_attempt_intent(
                execution_bundle=bundle,
                request_key=canonical.request_key,
                stage="inventory",
                prompt=prompt,
                schema_bundle=schema_bundle,
                cost_authorization_sha256=authorization.authorization_sha256,
                row_ordinal=row_ordinal,
                row=row,
            )
        else:
            row_ordinal, candidate_index = packet_keys[canonical.request_key]
            row = bundle.adapter_bundle.row_contexts[row_ordinal]
            inventory_prompt, inventory_bundle, _ = _request_surface(
                row=row,
                stage="inventory",
            )
            inventory_intent = freeze_metasyn_hosted_attempt_intent(
                execution_bundle=bundle,
                request_key=f"row-{row_ordinal:02d}-inventory",
                stage="inventory",
                prompt=inventory_prompt,
                schema_bundle=inventory_bundle,
                cost_authorization_sha256=authorization.authorization_sha256,
                row_ordinal=row_ordinal,
                row=row,
            )
            inventory_outcome = _validate_saved_call(
                workspace=workspace,
                expected_intent=inventory_intent,
                bundle=bundle,
                schema_bundle=inventory_bundle,
                row=row,
            )
            inventory_receipt = _inventory_adapter_receipt(
                inventory_outcome,
                row=row,
            )
            if inventory_receipt is None:
                raise MetaSynHostedRuntimeError(
                    "metasyn_hosted_packet_budget_inventory_not_authorizing"
                )
            candidate = next(
                (
                    item
                    for item in inventory_receipt.inventory.candidates
                    if item.candidate_index == candidate_index
                ),
                None,
            )
            if candidate is None:
                raise MetaSynHostedRuntimeError(
                    "metasyn_hosted_packet_budget_candidate_not_authorized"
                )
            packet_prompt, packet_bundle, packet_call = _request_surface(
                row=row,
                stage="packet",
                inventory_receipt=inventory_receipt,
                candidate_index=candidate_index,
            )
            if packet_call is None:  # pragma: no cover - construction invariant
                raise MetaSynHostedRuntimeError(
                    "metasyn_hosted_packet_budget_call_missing"
                )
            expected = freeze_metasyn_hosted_attempt_intent(
                execution_bundle=bundle,
                request_key=canonical.request_key,
                stage="packet",
                prompt=packet_prompt,
                schema_bundle=packet_bundle,
                cost_authorization_sha256=authorization.authorization_sha256,
                row_ordinal=row_ordinal,
                row=row,
                inventory_receipt=inventory_receipt,
                candidate=packet_call.candidate,
            )
        _BUDGET_AUTHORITATIVE_INTENT_CACHE[cache_key] = expected

    if canonical != expected:
        raise MetaSynHostedRuntimeError(
            "metasyn_hosted_authoritative_budget_intent_replay_mismatch"
        )
    return group, AnthropicBoundedRequestV1.model_validate(canonical.request)


def _assert_authorization_group_ceilings(
    *,
    classified_requests: Sequence[
        tuple[
            Literal[
                "source_free_preflight", "exact_inventory", "packet_slot_ceiling"
            ],
            AnthropicBoundedRequestV1,
            int | None,
        ]
    ],
    authorization: MetaSynHostedCostAuthorizationReceiptV1,
    configured_cost_ceiling_usd_micros: int,
) -> None:
    """Enforce every group, transport, row, and global authorization ceiling."""

    group_stats: dict[str, dict[str, int]] = {
        item.group: {
            "call_count": 0,
            "structured_json_schema_call_count": 0,
            "prompt_json_schema_call_count": 0,
            "conservative_input_token_ceiling": 0,
            "maximum_output_token_ceiling": 0,
            "cost_ceiling_usd_micros": 0,
        }
        for item in authorization.groups
    }
    packet_rows: Counter[int] = Counter()
    for group, request, row_ordinal in classified_requests:
        stats = group_stats[group]
        stats["call_count"] += 1
        stats[f"{request.transport_mode}_call_count"] += 1
        stats["conservative_input_token_ceiling"] += (
            request.cost_ceiling.conservative_input_token_ceiling
        )
        stats["maximum_output_token_ceiling"] += request.max_output_tokens
        stats["cost_ceiling_usd_micros"] += _usd_micros(
            request.cost_ceiling.request_cost_ceiling_usd
        )
        if group == "packet_slot_ceiling":
            if row_ordinal is None:
                raise MetaSynHostedRuntimeError(
                    "metasyn_hosted_packet_budget_row_missing"
                )
            packet_rows[row_ordinal] += 1

    structured = sum(
        stats["structured_json_schema_call_count"]
        for stats in group_stats.values()
    )
    prompt_json = sum(
        stats["prompt_json_schema_call_count"] for stats in group_stats.values()
    )
    if (
        structured > authorization.maximum_structured_json_schema_calls
        or prompt_json > authorization.maximum_prompt_json_schema_calls
    ):
        raise MetaSynHostedRuntimeError(
            "metasyn_hosted_global_transport_mode_ceiling_exceeded"
        )
    for authorized_group in authorization.groups:
        stats = group_stats[authorized_group.group]
        for field in (
            "call_count",
            "structured_json_schema_call_count",
            "prompt_json_schema_call_count",
            "conservative_input_token_ceiling",
            "maximum_output_token_ceiling",
            "cost_ceiling_usd_micros",
        ):
            if stats[field] > getattr(authorized_group, field):
                raise MetaSynHostedRuntimeError(
                    f"metasyn_hosted_authorization_group_{field}_exceeded"
                )
    if any(count > MAX_CANDIDATES_PER_PUBLICATION for count in packet_rows.values()):
        raise MetaSynHostedRuntimeError(
            "metasyn_hosted_packet_per_publication_ceiling_exceeded"
        )
    total_cost = sum(
        stats["cost_ceiling_usd_micros"] for stats in group_stats.values()
    )
    if (
        total_cost > authorization.cost_ceiling_usd_micros
        or total_cost > configured_cost_ceiling_usd_micros
    ):
        raise MetaSynHostedRuntimeError("metasyn_hosted_cost_ceiling_exceeded_pre_call")


def _assert_new_call_within_budget(
    *,
    workspace: Path,
    bundle: MetaSynHostedExecutionBundleV1,
    intent: MetaSynHostedAttemptIntentV1,
    authorization: MetaSynHostedCostAuthorizationReceiptV1,
) -> None:
    if (
        authorization.execution_bundle_sha256 != bundle.execution_bundle_sha256
        or authorization.authorization_sha256 != intent.cost_authorization_sha256
    ):
        raise MetaSynHostedRuntimeError(
            "metasyn_hosted_intent_cost_authorization_mismatch"
        )
    observed = _current_intent_count(workspace)
    if observed >= bundle.maximum_theoretical_provider_calls:
        raise MetaSynHostedRuntimeError("metasyn_hosted_296_call_ceiling_reached")
    prior_intents: list[MetaSynHostedAttemptIntentV1] = []
    intent_directory = metasyn_hosted_runtime_paths(workspace)["intents"]
    if intent_directory.is_dir():
        for path in sorted(intent_directory.glob("*.json")):
            prior = MetaSynHostedAttemptIntentV1.model_validate(
                _read_json_object(path)
            )
            if path.stem != prior.request_key:
                raise MetaSynHostedRuntimeError(
                    "metasyn_hosted_prior_intent_filename_binding_mismatch"
                )
            prior_intents.append(prior)
    all_intents = [*prior_intents, intent]
    if len({item.request_key for item in all_intents}) != len(all_intents):
        raise MetaSynHostedRuntimeError(
            "metasyn_hosted_budget_intent_request_key_duplicate"
        )

    classified: list[
        tuple[
            Literal[
                "source_free_preflight", "exact_inventory", "packet_slot_ceiling"
            ],
            AnthropicBoundedRequestV1,
            int | None,
        ]
    ] = []
    for current in all_intents:
        group, request = _authoritatively_replay_budget_intent(
            workspace=workspace,
            bundle=bundle,
            authorization=authorization,
            intent=current,
        )
        classified.append((group, request, current.row_ordinal))
    _assert_authorization_group_ceilings(
        classified_requests=classified,
        authorization=authorization,
        configured_cost_ceiling_usd_micros=(
            bundle.maximum_authorized_cost_usd_micros
        ),
    )


class MetaSynHostedCostAuthorizationGroupV1(ContractModel):
    group: Literal["source_free_preflight", "exact_inventory", "packet_slot_ceiling"]
    call_count: Annotated[int, Field(ge=1, le=296)]
    structured_json_schema_call_count: Annotated[int, Field(ge=0, le=296)]
    prompt_json_schema_call_count: Annotated[int, Field(ge=0, le=296)]
    conservative_input_token_ceiling: Annotated[int, Field(ge=1)]
    maximum_output_token_ceiling: Annotated[int, Field(ge=1)]
    cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    request_roster_sha256: str
    transport_mode_roster_sha256: str

    @field_validator("request_roster_sha256", "transport_mode_roster_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("metasyn_hosted_cost_group_hash_invalid")
        return value

    @model_validator(mode="after")
    def validate_mode_counts(self) -> MetaSynHostedCostAuthorizationGroupV1:
        if (
            self.structured_json_schema_call_count
            + self.prompt_json_schema_call_count
            != self.call_count
        ):
            raise ValueError("metasyn_hosted_cost_group_mode_count_mismatch")
        return self


class MetaSynHostedCostAuthorizationReceiptV1(ContractModel):
    authorization_version: Literal[
        "metasyn-bounded-hosted-pre-first-call-cost-authorization-v2"
    ] = COST_AUTHORIZATION_VERSION
    status: Literal["authorized_before_first_provider_call"] = (
        "authorized_before_first_provider_call"
    )
    execution_bundle_sha256: str
    runtime_pipeline_sha256: str
    config_sha256: str
    anthropic_config_sha256: str
    provider_identity_sha256: str
    provider_model: Literal["claude-sonnet-5"] = "claude-sonnet-5"
    provider_pricing_table_sha256: str
    provider_pricing_verified_date: Literal["2026-08-28"] = "2026-08-28"
    groups: Annotated[
        list[MetaSynHostedCostAuthorizationGroupV1], Field(min_length=3, max_length=3)
    ]
    maximum_theoretical_provider_calls: Literal[296] = 296
    maximum_structured_json_schema_calls: Literal[35] = MAX_STRUCTURED_PROVIDER_CALLS
    maximum_prompt_json_schema_calls: Literal[261] = MAX_PROMPT_JSON_PROVIDER_CALLS
    conservative_input_token_ceiling: Annotated[int, Field(ge=1)]
    maximum_output_token_ceiling: Annotated[int, Field(ge=1)]
    cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    configured_input_token_ceiling: Annotated[int, Field(ge=1)]
    configured_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    actual_candidate_independent_packet_bound: Literal[True] = True
    packet_bound_method: Literal[
        "per_row_maximum_exact_rendered_packet_request_over_all_effect_families_reused_for_eight_slots"
    ] = (
        "per_row_maximum_exact_rendered_packet_request_over_all_effect_families_"
        "reused_for_eight_slots"
    )
    provider_calls_made_before_authorization: Literal[0] = 0
    authorization_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "runtime_pipeline_sha256",
        "config_sha256",
        "anthropic_config_sha256",
        "provider_identity_sha256",
        "provider_pricing_table_sha256",
        "authorization_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("metasyn_hosted_cost_authorization_hash_invalid")
        return value

    @model_validator(mode="after")
    def validate_authorization(self) -> MetaSynHostedCostAuthorizationReceiptV1:
        expected_groups = [
            "source_free_preflight",
            "exact_inventory",
            "packet_slot_ceiling",
        ]
        if [item.group for item in self.groups] != expected_groups:
            raise ValueError("metasyn_hosted_cost_group_roster_mismatch")
        if [item.call_count for item in self.groups] != [8, 32, 256]:
            raise ValueError("metasyn_hosted_cost_group_call_count_mismatch")
        structured = sum(
            item.structured_json_schema_call_count for item in self.groups
        )
        prompt_json = sum(item.prompt_json_schema_call_count for item in self.groups)
        if (
            structured != self.maximum_structured_json_schema_calls
            or prompt_json != self.maximum_prompt_json_schema_calls
            or [
                (item.structured_json_schema_call_count, item.prompt_json_schema_call_count)
                for item in self.groups
            ]
            != [(3, 5), (32, 0), (0, 256)]
        ):
            raise ValueError("metasyn_hosted_cost_authorization_mode_roster_mismatch")
        if (
            self.conservative_input_token_ceiling
            != sum(item.conservative_input_token_ceiling for item in self.groups)
            or self.maximum_output_token_ceiling
            != sum(item.maximum_output_token_ceiling for item in self.groups)
            or self.cost_ceiling_usd_micros
            != sum(item.cost_ceiling_usd_micros for item in self.groups)
        ):
            raise ValueError("metasyn_hosted_cost_authorization_aggregate_mismatch")
        if (
            self.conservative_input_token_ceiling > self.configured_input_token_ceiling
            or self.cost_ceiling_usd_micros > self.configured_cost_ceiling_usd_micros
        ):
            raise ValueError("metasyn_hosted_cost_authorization_exceeds_config")
        payload = self.model_dump(mode="json", exclude={"authorization_sha256"})
        if self.authorization_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_hosted_cost_authorization_self_hash_mismatch")
        return self


_EFFECT_KINDS = (
    "binary_group_statistics",
    "continuous_group_statistics",
    "direct_confidence_interval",
    "direct_standard_error",
    "direct_variance",
)


def _request_ceiling_tuple(
    request: AnthropicBoundedRequestV1,
) -> tuple[int, int, int, str]:
    ceiling = request.cost_ceiling
    return (
        ceiling.conservative_input_token_ceiling,
        ceiling.max_output_tokens,
        _usd_micros(ceiling.request_cost_ceiling_usd),
        request.request_sha256,
    )


def _freeze_cost_group(
    *,
    group: Literal["source_free_preflight", "exact_inventory", "packet_slot_ceiling"],
    requests: Sequence[AnthropicBoundedRequestV1],
    multiplicity: int = 1,
) -> MetaSynHostedCostAuthorizationGroupV1:
    tuples = [_request_ceiling_tuple(item) for item in requests]
    if not tuples:
        raise MetaSynHostedRuntimeError("metasyn_hosted_cost_group_empty")
    return MetaSynHostedCostAuthorizationGroupV1(
        group=group,
        call_count=len(tuples) * multiplicity,
        structured_json_schema_call_count=(
            sum(item.transport_mode == "structured_json_schema" for item in requests)
            * multiplicity
        ),
        prompt_json_schema_call_count=(
            sum(item.transport_mode == "prompt_json_schema" for item in requests)
            * multiplicity
        ),
        conservative_input_token_ceiling=(sum(item[0] for item in tuples) * multiplicity),
        maximum_output_token_ceiling=(sum(item[1] for item in tuples) * multiplicity),
        cost_ceiling_usd_micros=(sum(item[2] for item in tuples) * multiplicity),
        request_roster_sha256=hash_canonical(
            {
                "request_sha256s": [item[3] for item in tuples],
                "multiplicity": multiplicity,
            }
        ),
        transport_mode_roster_sha256=hash_canonical(
            {
                "transport_modes": [item.transport_mode for item in requests],
                "multiplicity": multiplicity,
            }
        ),
    )


def freeze_metasyn_hosted_cost_authorization(
    *, execution_bundle: MetaSynHostedExecutionBundleV1
) -> MetaSynHostedCostAuthorizationReceiptV1:
    bundle = _coerce_metasyn_hosted_execution_bundle(execution_bundle)
    preflight_requests: list[AnthropicBoundedRequestV1] = []
    for index, spec in enumerate(synthetic_schema_v2_preflight_specs()):
        schema_bundle = _preflight_schema_bundle(spec)
        _, request = _freeze_provider_request(
            bundle=bundle,
            stage="preflight",
            request_key=f"budget-preflight-{index:02d}",
            prompt=_preflight_prompt(spec),
            schema_bundle=schema_bundle,
        )
        preflight_requests.append(request)
    if Counter(item.transport_mode for item in preflight_requests) != Counter(
        {"structured_json_schema": 3, "prompt_json_schema": 5}
    ):
        raise MetaSynHostedRuntimeError(
            "metasyn_hosted_preflight_transport_roster_mismatch"
        )

    inventory_requests: list[AnthropicBoundedRequestV1] = []
    packet_row_ceiling_requests: list[AnthropicBoundedRequestV1] = []
    for row_ordinal, row in enumerate(bundle.adapter_bundle.row_contexts):
        prompt, schema_bundle, _ = _request_surface(row=row, stage="inventory")
        _, request = _freeze_provider_request(
            bundle=bundle,
            stage="inventory",
            request_key=f"budget-row-{row_ordinal:02d}-inventory",
            prompt=prompt,
            schema_bundle=schema_bundle,
        )
        inventory_requests.append(request)
        line_ids = sorted(
            row.source_row.projection.exposed_line_ids,
            key=lambda item: (len(item.encode("utf-8")), item),
            reverse=True,
        )[:4]
        if not line_ids:
            continue
        line_ids.sort()
        outcome_name = max(
            row.allowed_outcomes,
            key=lambda item: (len(item.encode("utf-8")), item),
        )
        row_packet_candidates: list[AnthropicBoundedRequestV1] = []
        for effect_kind in _EFFECT_KINDS:
            inventory_value = {
                "inventory_version": "native-candidate-inventory-v1",
                "inventory_status": "candidates_found",
                "candidates": [
                    {
                        "candidate_index": 1,
                        "outcome_name": outcome_name,
                        "effect_kind": effect_kind,
                        "line_ids": line_ids,
                    }
                ],
                "has_more_or_uncertain": False,
            }
            inventory_receipt = freeze_metasyn_inventory_validation_receipt(
                row=row, value=inventory_value
            )
            packet_prompt, packet_bundle, _ = _request_surface(
                row=row,
                stage="packet",
                inventory_receipt=inventory_receipt,
                candidate_index=1,
            )
            _, packet_request = _freeze_provider_request(
                bundle=bundle,
                stage="packet",
                request_key=(f"budget-row-{row_ordinal:02d}-packet-{effect_kind}"),
                prompt=packet_prompt,
                schema_bundle=packet_bundle,
            )
            row_packet_candidates.append(packet_request)
        packet_row_ceiling_requests.append(
            max(
                row_packet_candidates,
                key=lambda item: (
                    item.cost_ceiling.request_cost_ceiling_usd,
                    item.request_sha256,
                ),
            )
        )
    if len(packet_row_ceiling_requests) != 32:
        raise MetaSynHostedRuntimeError("metasyn_hosted_packet_budget_bound_missing")
    groups = [
        _freeze_cost_group(group="source_free_preflight", requests=preflight_requests),
        _freeze_cost_group(group="exact_inventory", requests=inventory_requests),
        _freeze_cost_group(
            group="packet_slot_ceiling",
            requests=packet_row_ceiling_requests,
            multiplicity=8,
        ),
    ]
    input_ceiling = sum(item.conservative_input_token_ceiling for item in groups)
    output_ceiling = sum(item.maximum_output_token_ceiling for item in groups)
    cost_ceiling = sum(item.cost_ceiling_usd_micros for item in groups)
    payload: dict[str, Any] = {
        "authorization_version": COST_AUTHORIZATION_VERSION,
        "status": "authorized_before_first_provider_call",
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
        "config_sha256": bundle.config_sha256,
        "anthropic_config_sha256": bundle.anthropic_config_sha256,
        "provider_identity_sha256": bundle.provider_identity_sha256,
        "provider_model": bundle.anthropic_config.model,
        "provider_pricing_table_sha256": (bundle.anthropic_config.pricing_table_sha256),
        "provider_pricing_verified_date": (bundle.anthropic_config.source_verified_date),
        "groups": groups,
        "maximum_theoretical_provider_calls": 296,
        "maximum_structured_json_schema_calls": MAX_STRUCTURED_PROVIDER_CALLS,
        "maximum_prompt_json_schema_calls": MAX_PROMPT_JSON_PROVIDER_CALLS,
        "conservative_input_token_ceiling": input_ceiling,
        "maximum_output_token_ceiling": output_ceiling,
        "cost_ceiling_usd_micros": cost_ceiling,
        "configured_input_token_ceiling": (bundle.runtime_config.maximum_input_tokens_all_calls),
        "configured_cost_ceiling_usd_micros": (bundle.maximum_authorized_cost_usd_micros),
        "actual_candidate_independent_packet_bound": True,
        "packet_bound_method": (
            "per_row_maximum_exact_rendered_packet_request_over_all_effect_families_"
            "reused_for_eight_slots"
        ),
        "provider_calls_made_before_authorization": 0,
    }
    try:
        return MetaSynHostedCostAuthorizationReceiptV1.model_validate(
            {**payload, "authorization_sha256": hash_canonical(payload)}
        )
    except ValidationError as exc:
        raise MetaSynHostedRuntimeError(
            "metasyn_hosted_pre_first_call_cost_authorization_failed"
        ) from exc


def validate_metasyn_hosted_cost_authorization(
    *,
    receipt: MetaSynHostedCostAuthorizationReceiptV1 | Mapping[str, Any],
    execution_bundle: MetaSynHostedExecutionBundleV1,
) -> MetaSynHostedCostAuthorizationReceiptV1:
    canonical = MetaSynHostedCostAuthorizationReceiptV1.model_validate(receipt)
    replayed = freeze_metasyn_hosted_cost_authorization(execution_bundle=execution_bundle)
    if replayed != canonical:
        raise MetaSynHostedRuntimeError("metasyn_hosted_cost_authorization_replay_mismatch")
    return canonical


def _execute_once(
    *,
    workspace: Path,
    bundle: MetaSynHostedExecutionBundleV1,
    intent: MetaSynHostedAttemptIntentV1,
    authorization: MetaSynHostedCostAuthorizationReceiptV1,
    client: HostedBoundedClientProtocol,
    schema_bundle: Mapping[str, Any],
    row: MetaSynBoundedRowContextV1 | None = None,
    inventory_receipt: MetaSynInventoryValidationReceiptV1 | None = None,
    packet_call: MetaSynPacketCallV1 | None = None,
    preflight_fixture: Mapping[str, Any] | None = None,
) -> MetaSynHostedCallReceiptV1 | MetaSynHostedAmbiguityIncidentV1:
    bundle = _coerce_metasyn_hosted_execution_bundle(bundle)
    intent = MetaSynHostedAttemptIntentV1.model_validate(
        intent.model_dump(mode="json")
    )
    authorization = MetaSynHostedCostAuthorizationReceiptV1.model_validate(
        authorization.model_dump(mode="json")
    )
    if (
        authorization.execution_bundle_sha256 != bundle.execution_bundle_sha256
        or intent.execution_bundle_sha256 != bundle.execution_bundle_sha256
        or intent.cost_authorization_sha256 != authorization.authorization_sha256
    ):
        raise MetaSynHostedRuntimeError("metasyn_hosted_execute_context_mismatch")
    _prepare_call_directories(workspace)
    intent_path, receipt_path, incident_path = _outcome_paths(workspace, intent.request_key)
    if receipt_path.exists() and incident_path.exists():
        raise MetaSynHostedRuntimeError("metasyn_hosted_call_has_two_outcomes")
    if intent_path.exists():
        saved = MetaSynHostedAttemptIntentV1.model_validate(_read_json_object(intent_path))
        if saved != intent:
            raise MetaSynHostedRuntimeError("metasyn_hosted_intent_replay_mismatch")
        if receipt_path.exists():
            saved_receipt = MetaSynHostedCallReceiptV1.model_validate(
                _read_json_object(receipt_path)
            )
            replayed_receipt = freeze_metasyn_hosted_call_receipt(
                execution_bundle=bundle,
                intent=intent,
                provider_result=saved_receipt.provider_result,
                schema_bundle=schema_bundle,
                row=row,
                inventory_receipt=inventory_receipt,
                packet_call=packet_call,
                preflight_fixture=preflight_fixture,
            )
            if saved_receipt != replayed_receipt:
                raise MetaSynHostedRuntimeError(
                    "metasyn_hosted_saved_receipt_replay_mismatch"
                )
            return saved_receipt
        if incident_path.exists():
            saved_incident = MetaSynHostedAmbiguityIncidentV1.model_validate(
                _read_json_object(incident_path)
            )
            replayed_incident = freeze_metasyn_hosted_ambiguity_incident(
                intent=intent,
                incident_kind=saved_incident.incident_kind,
            )
            if saved_incident != replayed_incident:
                raise MetaSynHostedRuntimeError(
                    "metasyn_hosted_saved_incident_replay_mismatch"
                )
            return saved_incident
        incident = freeze_metasyn_hosted_ambiguity_incident(
            intent=intent,
            incident_kind="orphan_intent_observed_on_resume",
        )
        atomic_write_json(incident_path, incident)
        return incident
    if receipt_path.exists() or incident_path.exists():
        raise MetaSynHostedRuntimeError("metasyn_hosted_outcome_without_intent")
    _assert_new_call_within_budget(
        workspace=workspace,
        bundle=bundle,
        intent=intent,
        authorization=authorization,
    )
    # This fsync-backed immutable write is the authorization boundary.  Nothing
    # between it and ``generate`` may create a second request for this intent.
    atomic_write_json(intent_path, intent)
    try:
        prompt = intent.request.get("prompt")
        if not isinstance(prompt, str):
            raise MetaSynHostedRuntimeError("metasyn_hosted_intent_prompt_missing")
        _, provider_request = _freeze_provider_request(
            bundle=bundle,
            stage=intent.stage,
            request_key=intent.request_key,
            prompt=prompt,
            schema_bundle=schema_bundle,
        )
        if provider_request.model_dump(mode="json") != intent.request:
            raise MetaSynHostedRuntimeError(
                "metasyn_hosted_provider_request_reconstruction_mismatch"
            )
        result = client.generate(provider_request)
    except (KeyboardInterrupt, SystemExit):  # pragma: no cover - crash simulation
        raise
    except Exception:
        incident = freeze_metasyn_hosted_ambiguity_incident(
            intent=intent,
            incident_kind="provider_call_raised_after_durable_intent",
        )
        atomic_write_json(incident_path, incident)
        return incident
    receipt = freeze_metasyn_hosted_call_receipt(
        execution_bundle=bundle,
        intent=intent,
        provider_result=result,
        schema_bundle=schema_bundle,
        row=row,
        inventory_receipt=inventory_receipt,
        packet_call=packet_call,
        preflight_fixture=preflight_fixture,
    )
    atomic_write_json(receipt_path, receipt)
    return receipt


class MetaSynHostedAttemptOutcomeRefV1(ContractModel):
    request_key: str
    stage: CallStage
    schema_kind: Literal["inventory", "packet"]
    effect_kind: AnthropicEffectKind | None
    transport_mode: AnthropicTransportMode
    structured_grammar_enforced_by_provider: bool
    request_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    state: Literal["response", "incident"]
    attempt_id: str
    attempt_intent_sha256: str
    outcome_sha256: str
    validation_status: CallValidationStatus | None
    incident_kind: IncidentKind | None

    @field_validator("attempt_id", "attempt_intent_sha256", "outcome_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("metasyn_hosted_outcome_ref_hash_invalid")
        return value

    @model_validator(mode="after")
    def validate_ref(self) -> MetaSynHostedAttemptOutcomeRefV1:
        response = self.state == "response"
        if response != (self.validation_status is not None and self.incident_kind is None):
            raise ValueError("metasyn_hosted_outcome_ref_state_mismatch")
        if (
            (self.schema_kind == "inventory") != (self.effect_kind is None)
            or (self.transport_mode == "structured_json_schema")
            != (self.schema_kind == "inventory")
            or self.structured_grammar_enforced_by_provider
            != (self.transport_mode == "structured_json_schema")
        ):
            raise ValueError("metasyn_hosted_outcome_ref_transport_shape_invalid")
        return self


def _outcome_ref(
    outcome: MetaSynHostedCallReceiptV1 | MetaSynHostedAmbiguityIncidentV1,
) -> MetaSynHostedAttemptOutcomeRefV1:
    if isinstance(outcome, MetaSynHostedCallReceiptV1):
        return MetaSynHostedAttemptOutcomeRefV1(
            request_key=outcome.request_key,
            stage=outcome.stage,
            schema_kind=outcome.schema_kind,
            effect_kind=outcome.effect_kind,
            transport_mode=outcome.transport_mode,
            structured_grammar_enforced_by_provider=(
                outcome.structured_grammar_enforced_by_provider
            ),
            request_cost_ceiling_usd_micros=(
                outcome.request_cost_ceiling_usd_micros
            ),
            state="response",
            attempt_id=outcome.attempt_id,
            attempt_intent_sha256=outcome.attempt_intent_sha256,
            outcome_sha256=outcome.receipt_sha256,
            validation_status=outcome.validation_status,
            incident_kind=None,
        )
    return MetaSynHostedAttemptOutcomeRefV1(
        request_key=outcome.request_key,
        stage=outcome.stage,
        schema_kind=outcome.schema_kind,
        effect_kind=outcome.effect_kind,
        transport_mode=outcome.transport_mode,
        structured_grammar_enforced_by_provider=(
            outcome.structured_grammar_enforced_by_provider
        ),
        request_cost_ceiling_usd_micros=(
            outcome.request_cost_ceiling_usd_micros
        ),
        state="incident",
        attempt_id=outcome.attempt_id,
        attempt_intent_sha256=outcome.attempt_intent_sha256,
        outcome_sha256=outcome.incident_sha256,
        validation_status=None,
        incident_kind=outcome.incident_kind,
    )


def _durable_intent_liability_surface(
    outcomes: Sequence[MetaSynHostedAttemptOutcomeRefV1],
) -> dict[str, int | str]:
    """Freeze the exact intent roster and its conservative request liability."""

    entries = [
        {
            "request_key": item.request_key,
            "attempt_id": item.attempt_id,
            "attempt_intent_sha256": item.attempt_intent_sha256,
            "schema_kind": item.schema_kind,
            "effect_kind": item.effect_kind,
            "transport_mode": item.transport_mode,
            "outcome_state": item.state,
            "outcome_sha256": item.outcome_sha256,
            "request_cost_ceiling_usd_micros": (
                item.request_cost_ceiling_usd_micros
            ),
        }
        for item in outcomes
    ]
    entries.sort(key=lambda item: str(item["request_key"]))
    if len({str(item["request_key"]) for item in entries}) != len(entries):
        raise ValueError("metasyn_hosted_durable_intent_request_key_duplicate")
    observed = sum(
        item.request_cost_ceiling_usd_micros
        for item in outcomes
        if item.state == "response"
    )
    ambiguous = sum(
        item.request_cost_ceiling_usd_micros
        for item in outcomes
        if item.state == "incident"
    )
    return {
        "durable_intent_count": len(entries),
        "observed_request_ceiling_usd_micros": observed,
        "possible_ambiguous_charge_ceiling_usd_micros": ambiguous,
        "durable_intent_liability_usd_micros": observed + ambiguous,
        "durable_intent_roster_sha256": hash_canonical(entries),
    }


class MetaSynHostedPreflightReceiptV1(ContractModel):
    preflight_version: Literal["metasyn-bounded-hosted-eight-call-preflight-v2"] = PREFLIGHT_VERSION
    execution_bundle_sha256: str
    runtime_pipeline_sha256: str
    provider_identity_sha256: str
    cost_authorization_sha256: str
    schema_v2_preflight_fingerprint: str
    status: Literal["passed", "failed"]
    call_outcomes: Annotated[
        list[MetaSynHostedAttemptOutcomeRefV1], Field(min_length=8, max_length=8)
    ]
    call_outcome_sha256s: Annotated[list[str], Field(min_length=8, max_length=8)]
    passed_call_count: Annotated[int, Field(ge=0, le=8)]
    observed_provider_calls: Annotated[int, Field(ge=0, le=8)]
    possible_ambiguous_provider_calls: Annotated[int, Field(ge=0, le=8)]
    structured_json_schema_calls: Literal[3] = PREFLIGHT_STRUCTURED_CALL_COUNT
    prompt_json_schema_calls: Literal[5] = PREFLIGHT_PROMPT_JSON_CALL_COUNT
    transport_mode_roster_sha256: str
    observed_request_ceiling_usd_micros: Annotated[int, Field(ge=0)]
    possible_ambiguous_charge_ceiling_usd_micros: Annotated[int, Field(ge=0)]
    durable_intent_liability_usd_micros: Annotated[int, Field(ge=1)]
    durable_intent_roster_sha256: str
    cost_authorization_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    configured_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    source_bearing_provider_calls: Literal[0] = 0
    application_retries: Literal[0] = 0
    sdk_retries: Literal[0] = 0
    whole_request_compatibility_only: Literal[True] = True
    scientific_correctness_authority: Literal[False] = False
    usage: MetaSynHostedUsageV1
    cost: MetaSynHostedCostV1
    preflight_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "runtime_pipeline_sha256",
        "provider_identity_sha256",
        "cost_authorization_sha256",
        "schema_v2_preflight_fingerprint",
        "transport_mode_roster_sha256",
        "durable_intent_roster_sha256",
        "preflight_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("metasyn_hosted_preflight_hash_invalid")
        return value

    @model_validator(mode="after")
    def validate_preflight(self) -> MetaSynHostedPreflightReceiptV1:
        keys = [item.request_key for item in self.call_outcomes]
        if keys != [f"preflight-{index:02d}" for index in range(8)]:
            raise ValueError("metasyn_hosted_preflight_call_roster_mismatch")
        hashes = [item.outcome_sha256 for item in self.call_outcomes]
        if self.call_outcome_sha256s != hashes:
            raise ValueError("metasyn_hosted_preflight_outcome_hashes_mismatch")
        passed = sum(
            item.validation_status == "preflight_fixture_valid" for item in self.call_outcomes
        )
        observed = sum(item.state == "response" for item in self.call_outcomes)
        ambiguous = sum(item.state == "incident" for item in self.call_outcomes)
        observed_ceiling = sum(
            item.request_cost_ceiling_usd_micros
            for item in self.call_outcomes
            if item.state == "response"
        )
        modes = [item.transport_mode for item in self.call_outcomes]
        expected_modes = ["structured_json_schema"] * 3 + ["prompt_json_schema"] * 5
        liability = _durable_intent_liability_surface(self.call_outcomes)
        if (
            self.passed_call_count != passed
            or self.observed_provider_calls != observed
            or self.possible_ambiguous_provider_calls != ambiguous
            or observed + ambiguous != 8
            or modes != expected_modes
            or self.structured_json_schema_calls != modes.count(
                "structured_json_schema"
            )
            or self.prompt_json_schema_calls != modes.count("prompt_json_schema")
            or self.transport_mode_roster_sha256 != hash_canonical(modes)
            or self.possible_ambiguous_charge_ceiling_usd_micros
            != sum(
                item.request_cost_ceiling_usd_micros
                for item in self.call_outcomes
                if item.state == "incident"
            )
            or self.cost.request_ceiling_usd_micros != observed_ceiling
            or self.observed_request_ceiling_usd_micros != observed_ceiling
            or self.durable_intent_liability_usd_micros
            != liability["durable_intent_liability_usd_micros"]
            or self.durable_intent_roster_sha256
            != liability["durable_intent_roster_sha256"]
            or self.durable_intent_liability_usd_micros
            != self.observed_request_ceiling_usd_micros
            + self.possible_ambiguous_charge_ceiling_usd_micros
            or self.durable_intent_liability_usd_micros
            > self.cost_authorization_ceiling_usd_micros
            or self.durable_intent_liability_usd_micros
            > self.configured_cost_ceiling_usd_micros
        ):
            raise ValueError("metasyn_hosted_preflight_count_mismatch")
        if (self.status == "passed") != (passed == 8):
            raise ValueError("metasyn_hosted_preflight_status_mismatch")
        payload = self.model_dump(mode="json", exclude={"preflight_sha256"})
        if self.preflight_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_hosted_preflight_self_hash_mismatch")
        return self


def _preflight_schema_bundle(spec: Mapping[str, Any]) -> dict[str, Any]:
    if spec.get("kind") == "inventory":
        bundle = inventory_schema_bundle_v2(
            exposed_line_ids=["SYNTHETIC_LINE"],
            allowed_outcomes=["synthetic_outcome"],
        )
    else:
        effect_kind = spec.get("effect_kind")
        if not isinstance(effect_kind, str):
            raise MetaSynHostedRuntimeError("metasyn_hosted_preflight_effect_kind_missing")
        candidate = NativeCandidateDescriptor(
            candidate_index=1,
            outcome_name="synthetic_outcome",
            effect_kind=effect_kind,
            line_ids=["SYNTHETIC_LINE"],
        )
        bundle = packet_schema_bundle_v2(
            candidate=candidate,
            exposed_line_ids=["SYNTHETIC_LINE"],
            source_locator="synthetic:schema-compilation-only",
            allowed_outcomes=["synthetic_outcome"],
            allowed_moderators=[],
            allowed_sections=["Synthetic"],
            outcome_positive_directions={"synthetic_outcome": "larger synthetic target value"},
        )
    for name in (
        "provider_schema_sha256",
        "full_acceptance_schema_sha256",
        "schema_bundle_sha256",
    ):
        if bundle[name] != spec[name]:
            raise MetaSynHostedRuntimeError(
                f"metasyn_hosted_preflight_schema_binding_mismatch:{name}"
            )
    return bundle


def _freeze_preflight_receipt(
    *,
    bundle: MetaSynHostedExecutionBundleV1,
    authorization: MetaSynHostedCostAuthorizationReceiptV1,
    outcomes: Sequence[MetaSynHostedCallReceiptV1 | MetaSynHostedAmbiguityIncidentV1],
) -> MetaSynHostedPreflightReceiptV1:
    refs = [_outcome_ref(item) for item in outcomes]
    receipts = [item for item in outcomes if isinstance(item, MetaSynHostedCallReceiptV1)]
    passed = sum(item.validation_status == "preflight_fixture_valid" for item in receipts)
    modes = [item.transport_mode for item in refs]
    ambiguous_charge = sum(
        item.request_cost_ceiling_usd_micros
        for item in outcomes
        if isinstance(item, MetaSynHostedAmbiguityIncidentV1)
    )
    liability = _durable_intent_liability_surface(refs)
    if (
        liability["durable_intent_liability_usd_micros"]
        > authorization.groups[0].cost_ceiling_usd_micros
    ):
        raise MetaSynHostedRuntimeError(
            "metasyn_hosted_preflight_liability_exceeds_authorized_group"
        )
    payload: dict[str, Any] = {
        "preflight_version": PREFLIGHT_VERSION,
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
        "provider_identity_sha256": bundle.provider_identity_sha256,
        "cost_authorization_sha256": authorization.authorization_sha256,
        "schema_v2_preflight_fingerprint": (bundle.schema_v2_preflight_fingerprint),
        "status": "passed" if passed == 8 else "failed",
        "call_outcomes": refs,
        "call_outcome_sha256s": [item.outcome_sha256 for item in refs],
        "passed_call_count": passed,
        "observed_provider_calls": len(receipts),
        "possible_ambiguous_provider_calls": len(outcomes) - len(receipts),
        "structured_json_schema_calls": modes.count("structured_json_schema"),
        "prompt_json_schema_calls": modes.count("prompt_json_schema"),
        "transport_mode_roster_sha256": hash_canonical(modes),
        "observed_request_ceiling_usd_micros": (
            liability["observed_request_ceiling_usd_micros"]
        ),
        "possible_ambiguous_charge_ceiling_usd_micros": ambiguous_charge,
        "durable_intent_liability_usd_micros": (
            liability["durable_intent_liability_usd_micros"]
        ),
        "durable_intent_roster_sha256": liability["durable_intent_roster_sha256"],
        "cost_authorization_ceiling_usd_micros": (
            authorization.cost_ceiling_usd_micros
        ),
        "configured_cost_ceiling_usd_micros": (
            bundle.maximum_authorized_cost_usd_micros
        ),
        "source_bearing_provider_calls": 0,
        "application_retries": 0,
        "sdk_retries": 0,
        "whole_request_compatibility_only": True,
        "scientific_correctness_authority": False,
        "usage": _sum_usage([item.usage for item in receipts]),
        "cost": _sum_cost([item.cost for item in receipts]),
    }
    return MetaSynHostedPreflightReceiptV1.model_validate(
        {**payload, "preflight_sha256": hash_canonical(payload)}
    )


def _validate_saved_call(
    *,
    workspace: Path,
    expected_intent: MetaSynHostedAttemptIntentV1,
    bundle: MetaSynHostedExecutionBundleV1,
    schema_bundle: Mapping[str, Any],
    row: MetaSynBoundedRowContextV1 | None = None,
    inventory_receipt: MetaSynInventoryValidationReceiptV1 | None = None,
    packet_call: MetaSynPacketCallV1 | None = None,
    preflight_fixture: Mapping[str, Any] | None = None,
) -> MetaSynHostedCallReceiptV1 | MetaSynHostedAmbiguityIncidentV1:
    intent_path, receipt_path, incident_path = _outcome_paths(
        workspace, expected_intent.request_key
    )
    if not intent_path.is_file():
        raise MetaSynHostedRuntimeError("metasyn_hosted_saved_intent_missing")
    saved_intent = MetaSynHostedAttemptIntentV1.model_validate(_read_json_object(intent_path))
    if saved_intent != expected_intent:
        raise MetaSynHostedRuntimeError("metasyn_hosted_saved_intent_replay_mismatch")
    if receipt_path.exists() == incident_path.exists():
        raise MetaSynHostedRuntimeError("metasyn_hosted_saved_outcome_cardinality_invalid")
    if incident_path.exists():
        incident = MetaSynHostedAmbiguityIncidentV1.model_validate(_read_json_object(incident_path))
        expected = freeze_metasyn_hosted_ambiguity_incident(
            intent=expected_intent, incident_kind=incident.incident_kind
        )
        if incident != expected:
            raise MetaSynHostedRuntimeError("metasyn_hosted_saved_incident_replay_mismatch")
        return incident
    receipt = MetaSynHostedCallReceiptV1.model_validate(_read_json_object(receipt_path))
    replayed = freeze_metasyn_hosted_call_receipt(
        execution_bundle=bundle,
        intent=expected_intent,
        provider_result=receipt.provider_result,
        schema_bundle=schema_bundle,
        row=row,
        inventory_receipt=inventory_receipt,
        packet_call=packet_call,
        preflight_fixture=preflight_fixture,
    )
    if receipt != replayed:
        raise MetaSynHostedRuntimeError("metasyn_hosted_saved_receipt_replay_mismatch")
    return receipt


def validate_metasyn_hosted_preflight(
    *,
    receipt: MetaSynHostedPreflightReceiptV1 | Mapping[str, Any],
    workspace: Path,
    execution_bundle: MetaSynHostedExecutionBundleV1,
) -> MetaSynHostedPreflightReceiptV1:
    canonical = MetaSynHostedPreflightReceiptV1.model_validate(receipt)
    authorization = validate_metasyn_hosted_cost_authorization(
        receipt=_read_json_object(
            metasyn_hosted_runtime_paths(workspace)["cost_authorization"]
        ),
        execution_bundle=execution_bundle,
    )
    specs = synthetic_schema_v2_preflight_specs()
    outcomes: list[MetaSynHostedCallReceiptV1 | MetaSynHostedAmbiguityIncidentV1] = []
    for index, spec in enumerate(specs):
        schema_bundle = _preflight_schema_bundle(spec)
        prompt = _preflight_prompt(spec)
        intent = freeze_metasyn_hosted_attempt_intent(
            execution_bundle=execution_bundle,
            request_key=f"preflight-{index:02d}",
            stage="preflight",
            prompt=prompt,
            schema_bundle=schema_bundle,
            cost_authorization_sha256=authorization.authorization_sha256,
        )
        outcomes.append(
            _validate_saved_call(
                workspace=workspace,
                expected_intent=intent,
                bundle=execution_bundle,
                schema_bundle=schema_bundle,
                preflight_fixture=_preflight_fixture(spec),
            )
        )
    replayed = _freeze_preflight_receipt(
        bundle=execution_bundle,
        authorization=authorization,
        outcomes=outcomes,
    )
    if replayed != canonical:
        raise MetaSynHostedRuntimeError("metasyn_hosted_preflight_replay_mismatch")
    return canonical


def run_metasyn_hosted_preflight(
    *,
    workspace: Path,
    repository_root: Path,
    client: HostedBoundedClientProtocol,
    expected_execution_bundle_sha256: str,
) -> MetaSynHostedPreflightReceiptV1:
    if not SHA256_RE.fullmatch(expected_execution_bundle_sha256):
        raise MetaSynHostedRuntimeError("metasyn_hosted_execution_anchor_invalid")
    canonical_workspace, initial_bundle = load_current_metasyn_hosted_execution_bundle(
        workspace=workspace,
        repository_root=repository_root,
        external_replay=True,
    )
    if initial_bundle.execution_bundle_sha256 != expected_execution_bundle_sha256:
        raise MetaSynHostedRuntimeError("metasyn_hosted_execution_anchor_mismatch")
    with _workspace_lock(canonical_workspace):
        paths = metasyn_hosted_runtime_paths(canonical_workspace)
        if paths["private_report"].exists():
            raise MetaSynHostedRuntimeError("metasyn_hosted_finalized_workspace_is_immutable")
        bundle = validate_current_metasyn_hosted_execution_bundle(
            execution_bundle=_read_json_object(paths["execution_bundle"]),
            repository_root=repository_root,
            external_replay=True,
        )
        if bundle != initial_bundle:
            raise MetaSynHostedRuntimeError("metasyn_hosted_execution_bundle_changed_before_lock")
        if paths["preflight"].exists():
            return validate_metasyn_hosted_preflight(
                receipt=_read_json_object(paths["preflight"]),
                workspace=canonical_workspace,
                execution_bundle=bundle,
            )
        _prepare_call_directories(canonical_workspace)
        if any(path.name.startswith("row-") for path in paths["intents"].glob("*.json")):
            raise MetaSynHostedRuntimeError("metasyn_hosted_source_call_before_preflight")
        # This closed bound is checked before the very first provider call.  Each
        # actual request is then checked against the same cumulative dollar cap.
        if (
            bundle.maximum_theoretical_provider_calls != 296
            or bundle.runtime_config.maximum_provider_calls != 296
            or bundle.runtime_config.maximum_authorized_cost_usd_micros
            != bundle.maximum_authorized_cost_usd_micros
        ):
            raise MetaSynHostedRuntimeError("metasyn_hosted_initial_budget_contract_invalid")
        if paths["cost_authorization"].exists():
            authorization = validate_metasyn_hosted_cost_authorization(
                receipt=_read_json_object(paths["cost_authorization"]),
                execution_bundle=bundle,
            )
        else:
            if _current_intent_count(canonical_workspace) != 0:
                raise MetaSynHostedRuntimeError(
                    "metasyn_hosted_call_intent_precedes_cost_authorization"
                )
            authorization = freeze_metasyn_hosted_cost_authorization(execution_bundle=bundle)
            atomic_write_json(paths["cost_authorization"], authorization)
            authorization = validate_metasyn_hosted_cost_authorization(
                receipt=authorization, execution_bundle=bundle
            )
        outcomes: list[MetaSynHostedCallReceiptV1 | MetaSynHostedAmbiguityIncidentV1] = []
        for index, spec in enumerate(synthetic_schema_v2_preflight_specs()):
            schema_bundle = _preflight_schema_bundle(spec)
            prompt = _preflight_prompt(spec)
            intent = freeze_metasyn_hosted_attempt_intent(
                execution_bundle=bundle,
                request_key=f"preflight-{index:02d}",
                stage="preflight",
                prompt=prompt,
                schema_bundle=schema_bundle,
                cost_authorization_sha256=authorization.authorization_sha256,
            )
            outcomes.append(
                _execute_once(
                    workspace=canonical_workspace,
                    bundle=bundle,
                    intent=intent,
                    authorization=authorization,
                    client=client,
                    schema_bundle=schema_bundle,
                    preflight_fixture=_preflight_fixture(spec),
                )
            )
        receipt = _freeze_preflight_receipt(
            bundle=bundle,
            authorization=authorization,
            outcomes=outcomes,
        )
        atomic_write_json(paths["preflight"], receipt)
        return validate_metasyn_hosted_preflight(
            receipt=receipt,
            workspace=canonical_workspace,
            execution_bundle=bundle,
        )


def _inventory_adapter_receipt(
    outcome: MetaSynHostedCallReceiptV1 | MetaSynHostedAmbiguityIncidentV1,
    *,
    row: MetaSynBoundedRowContextV1,
) -> MetaSynInventoryValidationReceiptV1 | None:
    if not isinstance(outcome, MetaSynHostedCallReceiptV1):
        return None
    if not outcome.validation_status.startswith("inventory_valid"):
        return None
    if outcome.adapter_validation_receipt is None:
        raise MetaSynHostedRuntimeError("metasyn_hosted_inventory_adapter_receipt_missing")
    return validate_metasyn_inventory_validation_receipt(
        receipt=outcome.adapter_validation_receipt,
        row=row,
    )


def _packet_adapter_receipt(
    outcome: MetaSynHostedCallReceiptV1 | MetaSynHostedAmbiguityIncidentV1,
    *,
    row: MetaSynBoundedRowContextV1,
    inventory_receipt: MetaSynInventoryValidationReceiptV1,
) -> MetaSynPacketValidationReceiptV1 | None:
    if not isinstance(outcome, MetaSynHostedCallReceiptV1):
        return None
    if outcome.validation_status not in {
        "packet_completed",
        "packet_unable_to_complete",
    }:
        return None
    if outcome.adapter_validation_receipt is None or outcome.candidate_index is None:
        raise MetaSynHostedRuntimeError("metasyn_hosted_packet_adapter_receipt_missing")
    call = freeze_metasyn_packet_call(
        row=row,
        inventory_receipt=inventory_receipt,
        candidate_index=outcome.candidate_index,
    )
    return validate_metasyn_packet_validation_receipt(
        receipt=outcome.adapter_validation_receipt,
        call=call,
        row=row,
        inventory_receipt=inventory_receipt,
    )


class MetaSynHostedRowResultV1(ContractModel):
    row_result_version: Literal["metasyn-bounded-hosted-row-result-v2"] = ROW_RESULT_VERSION
    row_context_sha256: str
    question_spec_sha256: str
    question_bundle_sha256: str
    source_row_sha256: str
    release_grade_source_grounding_eligible: bool
    status: RowStatus
    blockers: list[str]
    inventory_outcome: MetaSynHostedAttemptOutcomeRefV1
    packet_outcomes: Annotated[list[MetaSynHostedAttemptOutcomeRefV1], Field(max_length=8)]
    inventory_call_receipt_sha256: str | None
    packet_call_receipt_sha256s: list[str]
    adapter_publication_result: MetaSynPublicationResultV1 | None
    adapter_publication_result_sha256: str | None
    observed_provider_calls: Annotated[int, Field(ge=0, le=9)]
    possible_ambiguous_provider_calls: Annotated[int, Field(ge=0, le=9)]
    structured_json_schema_calls: Literal[1] = 1
    prompt_json_schema_calls: Annotated[int, Field(ge=0, le=8)]
    possible_ambiguous_charge_ceiling_usd_micros: Annotated[int, Field(ge=0)]
    typed_finding_count: Annotated[int, Field(ge=0)]
    synthesis_input_eligible: bool
    usage: MetaSynHostedUsageV1
    cost: MetaSynHostedCostV1
    row_result_sha256: str

    @field_validator(
        "row_context_sha256",
        "question_spec_sha256",
        "question_bundle_sha256",
        "source_row_sha256",
        "inventory_call_receipt_sha256",
        "adapter_publication_result_sha256",
        "row_result_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_hosted_row_hash_invalid:{info.field_name}")
        return value

    @field_validator("blockers")
    @classmethod
    def validate_blockers(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("metasyn_hosted_row_blockers_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_row_result(self) -> MetaSynHostedRowResultV1:
        if self.inventory_outcome.stage != "inventory":
            raise ValueError("metasyn_hosted_row_inventory_stage_mismatch")
        if any(item.stage != "packet" for item in self.packet_outcomes):
            raise ValueError("metasyn_hosted_row_packet_stage_mismatch")
        observed = [
            item
            for item in (self.inventory_outcome, *self.packet_outcomes)
            if item.state == "response"
        ]
        ambiguous = [
            item
            for item in (self.inventory_outcome, *self.packet_outcomes)
            if item.state == "incident"
        ]
        if self.observed_provider_calls != len(
            observed
        ) or self.possible_ambiguous_provider_calls != len(ambiguous):
            raise ValueError("metasyn_hosted_row_call_count_mismatch")
        modes = [
            item.transport_mode
            for item in (self.inventory_outcome, *self.packet_outcomes)
        ]
        if (
            self.inventory_outcome.transport_mode != "structured_json_schema"
            or any(
                item.transport_mode != "prompt_json_schema"
                for item in self.packet_outcomes
            )
            or self.structured_json_schema_calls
            != modes.count("structured_json_schema")
            or self.prompt_json_schema_calls != modes.count("prompt_json_schema")
            or self.possible_ambiguous_charge_ceiling_usd_micros
            != sum(
                item.request_cost_ceiling_usd_micros
                for item in (self.inventory_outcome, *self.packet_outcomes)
                if item.state == "incident"
            )
            or self.cost.request_ceiling_usd_micros
            != sum(item.request_cost_ceiling_usd_micros for item in observed)
        ):
            raise ValueError("metasyn_hosted_row_transport_mode_count_mismatch")
        expected_inventory_sha = (
            self.inventory_outcome.outcome_sha256
            if self.inventory_outcome.state == "response"
            else None
        )
        if self.inventory_call_receipt_sha256 != expected_inventory_sha:
            raise ValueError("metasyn_hosted_row_inventory_receipt_hash_mismatch")
        expected_packet_hashes = [
            item.outcome_sha256 for item in self.packet_outcomes if item.state == "response"
        ]
        if self.packet_call_receipt_sha256s != expected_packet_hashes:
            raise ValueError("metasyn_hosted_row_packet_receipt_hashes_mismatch")
        expected_adapter_sha = (
            self.adapter_publication_result.result_sha256
            if self.adapter_publication_result is not None
            else None
        )
        if self.adapter_publication_result_sha256 != expected_adapter_sha:
            raise ValueError("metasyn_hosted_row_adapter_hash_mismatch")
        typed = self.status == "typed_publication_output"
        if typed != (
            self.adapter_publication_result is not None
            and self.adapter_publication_result.status == "typed_publication_output"
        ):
            raise ValueError("metasyn_hosted_row_typed_status_mismatch")
        if typed == bool(self.blockers):
            raise ValueError("metasyn_hosted_row_blocker_presence_mismatch")
        expected_findings = (
            sum(
                len(cohort.findings)
                for study in self.adapter_publication_result.official_output.studies
                for cohort in study.cohorts
            )
            if typed
            and self.adapter_publication_result is not None
            and self.adapter_publication_result.official_output is not None
            else 0
        )
        if self.typed_finding_count != expected_findings:
            raise ValueError("metasyn_hosted_row_finding_count_mismatch")
        if self.synthesis_input_eligible != (
            typed and self.release_grade_source_grounding_eligible
        ):
            raise ValueError("metasyn_hosted_row_synthesis_eligibility_mismatch")
        payload = self.model_dump(mode="json", exclude={"row_result_sha256"})
        if self.row_result_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_hosted_row_result_hash_mismatch")
        return self


def _freeze_row_result(
    *,
    row: MetaSynBoundedRowContextV1,
    inventory_outcome: MetaSynHostedCallReceiptV1 | MetaSynHostedAmbiguityIncidentV1,
    packet_outcomes: Sequence[MetaSynHostedCallReceiptV1 | MetaSynHostedAmbiguityIncidentV1],
) -> MetaSynHostedRowResultV1:
    inventory_receipt = _inventory_adapter_receipt(inventory_outcome, row=row)
    packet_receipts: list[MetaSynPacketValidationReceiptV1] = []
    if inventory_receipt is not None:
        for outcome in packet_outcomes:
            packet = _packet_adapter_receipt(
                outcome,
                row=row,
                inventory_receipt=inventory_receipt,
            )
            if packet is not None:
                packet_receipts.append(packet)

    adapter_result: MetaSynPublicationResultV1 | None = None
    if inventory_receipt is not None:
        adapter_result = freeze_metasyn_publication_result(
            row=row,
            inventory_receipt=inventory_receipt,
            packet_receipts=packet_receipts,
        )
        adapter_result = validate_metasyn_publication_result(result=adapter_result, row=row)

    blockers: set[str] = set()
    if isinstance(inventory_outcome, MetaSynHostedAmbiguityIncidentV1):
        status: RowStatus = "runtime_inventory_blocked"
        blockers.add(f"inventory_incident:{inventory_outcome.incident_kind}")
    elif inventory_receipt is None:
        status = "runtime_inventory_blocked"
        blockers.add(f"inventory_response:{inventory_outcome.validation_status}")
    elif inventory_receipt.status == "no_candidate_non_authorizing":
        status = "adapter_inventory_no_candidate"
        blockers.update(adapter_result.blocking_reasons if adapter_result else [])
    elif inventory_receipt.status == "capacity_or_uncertainty_non_authorizing":
        status = "adapter_inventory_uncertain"
        blockers.update(adapter_result.blocking_reasons if adapter_result else [])
    else:
        for outcome in packet_outcomes:
            candidate_prefix = f"packet_{outcome.candidate_index or 0:02d}"
            if isinstance(outcome, MetaSynHostedAmbiguityIncidentV1):
                blockers.add(f"{candidate_prefix}_incident:{outcome.incident_kind}")
            elif outcome.validation_status not in {
                "packet_completed",
                "packet_unable_to_complete",
            }:
                blockers.add(f"{candidate_prefix}_response:{outcome.validation_status}")
        if adapter_result is not None and (adapter_result.status == "typed_publication_output"):
            status = "typed_publication_output"
            blockers.clear()
        elif adapter_result is not None and (adapter_result.status == "abstained_packet_unable"):
            status = "adapter_packet_unable"
            blockers.update(adapter_result.blocking_reasons)
        else:
            status = "runtime_packet_blocked"
            blockers.add("publication_packet_set_incomplete_or_invalid")
            if adapter_result is not None:
                blockers.update(adapter_result.blocking_reasons)

    receipts = [
        item
        for item in (inventory_outcome, *packet_outcomes)
        if isinstance(item, MetaSynHostedCallReceiptV1)
    ]
    refs = [_outcome_ref(item) for item in packet_outcomes]
    all_outcomes = (inventory_outcome, *packet_outcomes)
    typed_findings = (
        sum(
            len(cohort.findings)
            for study in adapter_result.official_output.studies
            for cohort in study.cohorts
        )
        if status == "typed_publication_output"
        and adapter_result is not None
        and adapter_result.official_output is not None
        else 0
    )
    payload: dict[str, Any] = {
        "row_result_version": ROW_RESULT_VERSION,
        "row_context_sha256": row.row_context_sha256,
        "question_spec_sha256": row.question_spec_sha256,
        "question_bundle_sha256": row.question_bundle_sha256,
        "source_row_sha256": row.source_row_sha256,
        "release_grade_source_grounding_eligible": (
            row.source_row.release_grade_source_grounding_eligible
        ),
        "status": status,
        "blockers": sorted(blockers),
        "inventory_outcome": _outcome_ref(inventory_outcome),
        "packet_outcomes": refs,
        "inventory_call_receipt_sha256": (
            inventory_outcome.receipt_sha256
            if isinstance(inventory_outcome, MetaSynHostedCallReceiptV1)
            else None
        ),
        "packet_call_receipt_sha256s": [
            item.receipt_sha256
            for item in packet_outcomes
            if isinstance(item, MetaSynHostedCallReceiptV1)
        ],
        "adapter_publication_result": adapter_result,
        "adapter_publication_result_sha256": (
            adapter_result.result_sha256 if adapter_result is not None else None
        ),
        "observed_provider_calls": len(receipts),
        "possible_ambiguous_provider_calls": (1 + len(packet_outcomes) - len(receipts)),
        "structured_json_schema_calls": 1,
        "prompt_json_schema_calls": len(packet_outcomes),
        "possible_ambiguous_charge_ceiling_usd_micros": sum(
            item.request_cost_ceiling_usd_micros
            for item in all_outcomes
            if isinstance(item, MetaSynHostedAmbiguityIncidentV1)
        ),
        "typed_finding_count": typed_findings,
        "synthesis_input_eligible": (
            status == "typed_publication_output"
            and row.source_row.release_grade_source_grounding_eligible
        ),
        "usage": _sum_usage([item.usage for item in receipts]),
        "cost": _sum_cost([item.cost for item in receipts]),
    }
    return MetaSynHostedRowResultV1.model_validate(
        {**payload, "row_result_sha256": hash_canonical(payload)}
    )


def _run_or_resume_row(
    *,
    workspace: Path,
    bundle: MetaSynHostedExecutionBundleV1,
    authorization: MetaSynHostedCostAuthorizationReceiptV1,
    row_ordinal: int,
    client: HostedBoundedClientProtocol,
) -> MetaSynHostedRowResultV1:
    row = bundle.adapter_bundle.row_contexts[row_ordinal]
    row_path = metasyn_hosted_runtime_paths(workspace)["row_results"] / (
        f"row-{row_ordinal:02d}.json"
    )
    if row_path.exists():
        return _validate_saved_row_result(
            workspace=workspace,
            bundle=bundle,
            authorization=authorization,
            row_ordinal=row_ordinal,
            result=_read_json_object(row_path),
        )
    prompt, schema_bundle, _ = _request_surface(row=row, stage="inventory")
    inventory_intent = freeze_metasyn_hosted_attempt_intent(
        execution_bundle=bundle,
        request_key=f"row-{row_ordinal:02d}-inventory",
        stage="inventory",
        prompt=prompt,
        schema_bundle=schema_bundle,
        cost_authorization_sha256=authorization.authorization_sha256,
        row_ordinal=row_ordinal,
        row=row,
    )
    inventory_outcome = _execute_once(
        workspace=workspace,
        bundle=bundle,
        intent=inventory_intent,
        authorization=authorization,
        client=client,
        schema_bundle=schema_bundle,
        row=row,
    )
    inventory_receipt = _inventory_adapter_receipt(inventory_outcome, row=row)
    packet_outcomes: list[MetaSynHostedCallReceiptV1 | MetaSynHostedAmbiguityIncidentV1] = []
    if inventory_receipt is not None and inventory_receipt.inventory.authorizes_packet_generation():
        candidates = inventory_receipt.inventory.candidates
        if len(candidates) > MAX_CANDIDATES_PER_PUBLICATION:
            raise MetaSynHostedRuntimeError("metasyn_hosted_candidate_ceiling_exceeded")
        for candidate in candidates:
            packet_prompt, packet_bundle, packet_call = _request_surface(
                row=row,
                stage="packet",
                inventory_receipt=inventory_receipt,
                candidate_index=candidate.candidate_index,
            )
            if packet_call is None:  # pragma: no cover - construction invariant
                raise MetaSynHostedRuntimeError("metasyn_hosted_packet_call_missing")
            intent = freeze_metasyn_hosted_attempt_intent(
                execution_bundle=bundle,
                request_key=(f"row-{row_ordinal:02d}-packet-{candidate.candidate_index:02d}"),
                stage="packet",
                prompt=packet_prompt,
                schema_bundle=packet_bundle,
                cost_authorization_sha256=authorization.authorization_sha256,
                row_ordinal=row_ordinal,
                row=row,
                inventory_receipt=inventory_receipt,
                candidate=candidate,
            )
            packet_outcomes.append(
                _execute_once(
                    workspace=workspace,
                    bundle=bundle,
                    intent=intent,
                    authorization=authorization,
                    client=client,
                    schema_bundle=packet_bundle,
                    row=row,
                    inventory_receipt=inventory_receipt,
                    packet_call=packet_call,
                )
            )
    result = _freeze_row_result(
        row=row,
        inventory_outcome=inventory_outcome,
        packet_outcomes=packet_outcomes,
    )
    atomic_write_json(row_path, result)
    return result


def _validate_saved_row_result(
    *,
    workspace: Path,
    bundle: MetaSynHostedExecutionBundleV1,
    authorization: MetaSynHostedCostAuthorizationReceiptV1,
    row_ordinal: int,
    result: MetaSynHostedRowResultV1 | Mapping[str, Any],
) -> MetaSynHostedRowResultV1:
    canonical = MetaSynHostedRowResultV1.model_validate(result)
    row = bundle.adapter_bundle.row_contexts[row_ordinal]
    if canonical.row_context_sha256 != row.row_context_sha256:
        raise MetaSynHostedRuntimeError("metasyn_hosted_saved_row_roster_mismatch")
    prompt, schema_bundle, _ = _request_surface(row=row, stage="inventory")
    inventory_intent = freeze_metasyn_hosted_attempt_intent(
        execution_bundle=bundle,
        request_key=f"row-{row_ordinal:02d}-inventory",
        stage="inventory",
        prompt=prompt,
        schema_bundle=schema_bundle,
        cost_authorization_sha256=authorization.authorization_sha256,
        row_ordinal=row_ordinal,
        row=row,
    )
    inventory_outcome = _validate_saved_call(
        workspace=workspace,
        expected_intent=inventory_intent,
        bundle=bundle,
        schema_bundle=schema_bundle,
        row=row,
    )
    inventory_receipt = _inventory_adapter_receipt(inventory_outcome, row=row)
    packet_outcomes: list[MetaSynHostedCallReceiptV1 | MetaSynHostedAmbiguityIncidentV1] = []
    if inventory_receipt is not None and inventory_receipt.inventory.authorizes_packet_generation():
        for candidate in inventory_receipt.inventory.candidates:
            packet_prompt, packet_bundle, packet_call = _request_surface(
                row=row,
                stage="packet",
                inventory_receipt=inventory_receipt,
                candidate_index=candidate.candidate_index,
            )
            if packet_call is None:  # pragma: no cover
                raise MetaSynHostedRuntimeError("metasyn_hosted_packet_call_missing")
            intent = freeze_metasyn_hosted_attempt_intent(
                execution_bundle=bundle,
                request_key=(f"row-{row_ordinal:02d}-packet-{candidate.candidate_index:02d}"),
                stage="packet",
                prompt=packet_prompt,
                schema_bundle=packet_bundle,
                cost_authorization_sha256=authorization.authorization_sha256,
                row_ordinal=row_ordinal,
                row=row,
                inventory_receipt=inventory_receipt,
                candidate=candidate,
            )
            packet_outcomes.append(
                _validate_saved_call(
                    workspace=workspace,
                    expected_intent=intent,
                    bundle=bundle,
                    schema_bundle=packet_bundle,
                    row=row,
                    inventory_receipt=inventory_receipt,
                    packet_call=packet_call,
                )
            )
    replayed = _freeze_row_result(
        row=row,
        inventory_outcome=inventory_outcome,
        packet_outcomes=packet_outcomes,
    )
    if replayed != canonical:
        raise MetaSynHostedRuntimeError("metasyn_hosted_saved_row_replay_mismatch")
    return canonical


class MetaSynHostedSmokeGateV1(ContractModel):
    smoke_version: Literal["metasyn-bounded-hosted-smoke-gate-v2"] = SMOKE_VERSION
    execution_bundle_sha256: str
    preflight_sha256: str
    cost_authorization_sha256: str
    row_ordinal: Literal[0] = 0
    row_context_sha256: str
    row_result_sha256: str
    row_status: RowStatus
    status: Literal["passed", "failed"]
    gate_dimensions: Literal["transport_schema_grounding_and_terminal_status_only"] = (
        "transport_schema_grounding_and_terminal_status_only"
    )
    scientific_correctness_evaluated: Literal[False] = False
    extraction_accuracy_evaluated: Literal[False] = False
    direction_agreement_evaluated: Literal[False] = False
    claim_release_authority: Literal[False] = False
    smoke_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "preflight_sha256",
        "cost_authorization_sha256",
        "row_context_sha256",
        "row_result_sha256",
        "smoke_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("metasyn_hosted_smoke_hash_invalid")
        return value

    @model_validator(mode="after")
    def validate_smoke(self) -> MetaSynHostedSmokeGateV1:
        expected = self.row_status not in {
            "runtime_inventory_blocked",
            "runtime_packet_blocked",
        }
        if (self.status == "passed") != expected:
            raise ValueError("metasyn_hosted_smoke_status_mismatch")
        payload = self.model_dump(mode="json", exclude={"smoke_sha256"})
        if self.smoke_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_hosted_smoke_self_hash_mismatch")
        return self


def _freeze_smoke_gate(
    *,
    bundle: MetaSynHostedExecutionBundleV1,
    preflight: MetaSynHostedPreflightReceiptV1,
    authorization: MetaSynHostedCostAuthorizationReceiptV1,
    row_result: MetaSynHostedRowResultV1,
) -> MetaSynHostedSmokeGateV1:
    passed = row_result.status not in {
        "runtime_inventory_blocked",
        "runtime_packet_blocked",
    }
    payload: dict[str, Any] = {
        "smoke_version": SMOKE_VERSION,
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "preflight_sha256": preflight.preflight_sha256,
        "cost_authorization_sha256": authorization.authorization_sha256,
        "row_ordinal": 0,
        "row_context_sha256": row_result.row_context_sha256,
        "row_result_sha256": row_result.row_result_sha256,
        "row_status": row_result.status,
        "status": "passed" if passed else "failed",
        "gate_dimensions": "transport_schema_grounding_and_terminal_status_only",
        "scientific_correctness_evaluated": False,
        "extraction_accuracy_evaluated": False,
        "direction_agreement_evaluated": False,
        "claim_release_authority": False,
    }
    return MetaSynHostedSmokeGateV1.model_validate(
        {**payload, "smoke_sha256": hash_canonical(payload)}
    )


def validate_metasyn_hosted_smoke(
    *,
    receipt: MetaSynHostedSmokeGateV1 | Mapping[str, Any],
    workspace: Path,
    execution_bundle: MetaSynHostedExecutionBundleV1,
    preflight: MetaSynHostedPreflightReceiptV1,
    authorization: MetaSynHostedCostAuthorizationReceiptV1,
) -> MetaSynHostedSmokeGateV1:
    canonical = MetaSynHostedSmokeGateV1.model_validate(receipt)
    row = _validate_saved_row_result(
        workspace=workspace,
        bundle=execution_bundle,
        authorization=authorization,
        row_ordinal=0,
        result=_read_json_object(
            metasyn_hosted_runtime_paths(workspace)["row_results"] / "row-00.json"
        ),
    )
    replayed = _freeze_smoke_gate(
        bundle=execution_bundle,
        preflight=preflight,
        authorization=authorization,
        row_result=row,
    )
    if replayed != canonical:
        raise MetaSynHostedRuntimeError("metasyn_hosted_smoke_replay_mismatch")
    return canonical


def run_metasyn_hosted_smoke(
    *,
    workspace: Path,
    repository_root: Path,
    client: HostedBoundedClientProtocol,
    expected_execution_bundle_sha256: str,
) -> MetaSynHostedSmokeGateV1:
    if not SHA256_RE.fullmatch(expected_execution_bundle_sha256):
        raise MetaSynHostedRuntimeError("metasyn_hosted_execution_anchor_invalid")
    canonical_workspace, initial_bundle = load_current_metasyn_hosted_execution_bundle(
        workspace=workspace,
        repository_root=repository_root,
        external_replay=True,
    )
    if initial_bundle.execution_bundle_sha256 != expected_execution_bundle_sha256:
        raise MetaSynHostedRuntimeError("metasyn_hosted_execution_anchor_mismatch")
    with _workspace_lock(canonical_workspace):
        paths = metasyn_hosted_runtime_paths(canonical_workspace)
        if paths["private_report"].exists():
            raise MetaSynHostedRuntimeError("metasyn_hosted_finalized_workspace_is_immutable")
        bundle = validate_current_metasyn_hosted_execution_bundle(
            execution_bundle=_read_json_object(paths["execution_bundle"]),
            repository_root=repository_root,
            external_replay=True,
        )
        if bundle != initial_bundle:
            raise MetaSynHostedRuntimeError("metasyn_hosted_execution_bundle_changed_before_lock")
        authorization = validate_metasyn_hosted_cost_authorization(
            receipt=_read_json_object(paths["cost_authorization"]),
            execution_bundle=bundle,
        )
        preflight = validate_metasyn_hosted_preflight(
            receipt=_read_json_object(paths["preflight"]),
            workspace=canonical_workspace,
            execution_bundle=bundle,
        )
        if preflight.status != "passed":
            raise MetaSynHostedRuntimeError("metasyn_hosted_smoke_requires_passed_preflight")
        if paths["smoke"].exists():
            return validate_metasyn_hosted_smoke(
                receipt=_read_json_object(paths["smoke"]),
                workspace=canonical_workspace,
                execution_bundle=bundle,
                preflight=preflight,
                authorization=authorization,
            )
        row_result = _run_or_resume_row(
            workspace=canonical_workspace,
            bundle=bundle,
            authorization=authorization,
            row_ordinal=0,
            client=client,
        )
        smoke = _freeze_smoke_gate(
            bundle=bundle,
            preflight=preflight,
            authorization=authorization,
            row_result=row_result,
        )
        atomic_write_json(paths["smoke"], smoke)
        return validate_metasyn_hosted_smoke(
            receipt=smoke,
            workspace=canonical_workspace,
            execution_bundle=bundle,
            preflight=preflight,
            authorization=authorization,
        )


class MetaSynHostedLedgerV1(ContractModel):
    ledger_version: Literal["metasyn-bounded-hosted-ledger-v2"] = LEDGER_VERSION
    status: Literal["complete_32_row_terminal_hosted_ledger"] = (
        "complete_32_row_terminal_hosted_ledger"
    )
    execution_bundle_sha256: str
    runtime_pipeline_sha256: str
    cost_authorization_sha256: str
    preflight_sha256: str
    smoke_sha256: str
    row_membership_sha256: str
    preflight_call_outcomes: Annotated[
        list[MetaSynHostedAttemptOutcomeRefV1], Field(min_length=8, max_length=8)
    ]
    row_results: Annotated[list[MetaSynHostedRowResultV1], Field(min_length=32, max_length=32)]
    row_result_sha256s: Annotated[list[str], Field(min_length=32, max_length=32)]
    row_status_counts: dict[str, Annotated[int, Field(ge=0)]]
    publication_count: Literal[32] = 32
    all_rows_terminal: Literal[True] = True
    observed_source_provider_calls: Annotated[int, Field(ge=0, le=288)]
    possible_ambiguous_source_provider_calls: Annotated[int, Field(ge=0, le=288)]
    observed_preflight_provider_calls: Annotated[int, Field(ge=0, le=8)]
    possible_ambiguous_preflight_provider_calls: Annotated[int, Field(ge=0, le=8)]
    total_provider_call_attempts_or_possible_attempts: Annotated[int, Field(ge=40, le=296)]
    structured_json_schema_calls: Literal[35] = MAX_STRUCTURED_PROVIDER_CALLS
    prompt_json_schema_calls: Annotated[int, Field(ge=5, le=261)]
    maximum_structured_json_schema_calls: Literal[35] = MAX_STRUCTURED_PROVIDER_CALLS
    maximum_prompt_json_schema_calls: Literal[261] = MAX_PROMPT_JSON_PROVIDER_CALLS
    transport_mode_policy: Literal[
        "inventory-structured-json-schema-packet-prompt-json-schema-v1"
    ] = "inventory-structured-json-schema-packet-prompt-json-schema-v1"
    possible_ambiguous_preflight_charge_ceiling_usd_micros: Annotated[
        int, Field(ge=0)
    ]
    observed_preflight_request_ceiling_usd_micros: Annotated[int, Field(ge=0)]
    observed_source_request_ceiling_usd_micros: Annotated[int, Field(ge=0)]
    possible_ambiguous_source_charge_ceiling_usd_micros: Annotated[
        int, Field(ge=0)
    ]
    observed_request_ceiling_usd_micros: Annotated[int, Field(ge=0)]
    possible_ambiguous_charge_ceiling_usd_micros: Annotated[int, Field(ge=0)]
    durable_intent_count: Annotated[int, Field(ge=40, le=296)]
    durable_intent_liability_usd_micros: Annotated[int, Field(ge=1)]
    durable_intent_roster_sha256: str
    cost_authorization_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    configured_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    maximum_theoretical_provider_calls: Literal[296] = 296
    application_retries: Literal[0] = 0
    sdk_retries: Literal[0] = 0
    usage: MetaSynHostedUsageV1
    cost: MetaSynHostedCostV1
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    ledger_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "runtime_pipeline_sha256",
        "cost_authorization_sha256",
        "preflight_sha256",
        "smoke_sha256",
        "row_membership_sha256",
        "durable_intent_roster_sha256",
        "ledger_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("metasyn_hosted_ledger_hash_invalid")
        return value

    @model_validator(mode="after")
    def validate_ledger(self) -> MetaSynHostedLedgerV1:
        row_hashes = [item.row_context_sha256 for item in self.row_results]
        if row_hashes != sorted(set(row_hashes)):
            raise ValueError("metasyn_hosted_ledger_row_roster_invalid")
        if self.row_result_sha256s != [item.row_result_sha256 for item in self.row_results]:
            raise ValueError("metasyn_hosted_ledger_row_hashes_mismatch")
        if self.row_status_counts != dict(
            sorted(Counter(item.status for item in self.row_results).items())
        ):
            raise ValueError("metasyn_hosted_ledger_status_counts_mismatch")
        observed = sum(item.observed_provider_calls for item in self.row_results)
        ambiguous = sum(item.possible_ambiguous_provider_calls for item in self.row_results)
        total = (
            observed
            + ambiguous
            + self.observed_preflight_provider_calls
            + self.possible_ambiguous_preflight_provider_calls
        )
        structured_calls = (
            sum(item.structured_json_schema_calls for item in self.row_results)
            + 3
        )
        prompt_calls = (
            sum(item.prompt_json_schema_calls for item in self.row_results) + 5
        )
        possible_charge = (
            sum(
                item.possible_ambiguous_charge_ceiling_usd_micros
                for item in self.row_results
            )
            + self.possible_ambiguous_preflight_charge_ceiling_usd_micros
        )
        observed_source_ceiling = sum(
            item.cost.request_ceiling_usd_micros for item in self.row_results
        )
        ambiguous_source_ceiling = sum(
            item.possible_ambiguous_charge_ceiling_usd_micros
            for item in self.row_results
        )
        all_outcomes = [
            *self.preflight_call_outcomes,
            *(
                outcome
                for row in self.row_results
                for outcome in (row.inventory_outcome, *row.packet_outcomes)
            ),
        ]
        liability = _durable_intent_liability_surface(all_outcomes)
        if (
            self.observed_source_provider_calls != observed
            or self.possible_ambiguous_source_provider_calls != ambiguous
            or self.total_provider_call_attempts_or_possible_attempts != total
            or self.structured_json_schema_calls != structured_calls
            or self.prompt_json_schema_calls != prompt_calls
            or structured_calls + prompt_calls != total
            or self.possible_ambiguous_charge_ceiling_usd_micros
            != possible_charge
            or self.observed_preflight_request_ceiling_usd_micros
            != sum(
                item.request_cost_ceiling_usd_micros
                for item in self.preflight_call_outcomes
                if item.state == "response"
            )
            or self.observed_source_request_ceiling_usd_micros
            != observed_source_ceiling
            or self.possible_ambiguous_source_charge_ceiling_usd_micros
            != ambiguous_source_ceiling
            or self.observed_request_ceiling_usd_micros
            != self.observed_preflight_request_ceiling_usd_micros
            + self.observed_source_request_ceiling_usd_micros
            or self.possible_ambiguous_charge_ceiling_usd_micros
            != self.possible_ambiguous_preflight_charge_ceiling_usd_micros
            + self.possible_ambiguous_source_charge_ceiling_usd_micros
            or self.observed_request_ceiling_usd_micros
            != liability["observed_request_ceiling_usd_micros"]
            or self.cost.request_ceiling_usd_micros
            != self.observed_request_ceiling_usd_micros
            or self.possible_ambiguous_charge_ceiling_usd_micros
            != liability["possible_ambiguous_charge_ceiling_usd_micros"]
            or self.durable_intent_count != liability["durable_intent_count"]
            or self.durable_intent_liability_usd_micros
            != liability["durable_intent_liability_usd_micros"]
            or self.durable_intent_roster_sha256
            != liability["durable_intent_roster_sha256"]
            or self.durable_intent_liability_usd_micros
            != self.observed_request_ceiling_usd_micros
            + self.possible_ambiguous_charge_ceiling_usd_micros
            or self.durable_intent_liability_usd_micros
            > self.cost_authorization_ceiling_usd_micros
            or self.durable_intent_liability_usd_micros
            > self.configured_cost_ceiling_usd_micros
        ):
            raise ValueError("metasyn_hosted_ledger_call_counts_mismatch")
        payload = self.model_dump(mode="json", exclude={"ledger_sha256"})
        if self.ledger_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_hosted_ledger_self_hash_mismatch")
        return self


def _freeze_hosted_ledger(
    *,
    bundle: MetaSynHostedExecutionBundleV1,
    authorization: MetaSynHostedCostAuthorizationReceiptV1,
    preflight: MetaSynHostedPreflightReceiptV1,
    smoke: MetaSynHostedSmokeGateV1,
    row_results: Sequence[MetaSynHostedRowResultV1],
) -> MetaSynHostedLedgerV1:
    ordered = sorted(row_results, key=lambda item: item.row_context_sha256)
    if len(ordered) != 32:
        raise MetaSynHostedRuntimeError("metasyn_hosted_ledger_requires_32_rows")
    row_usage = _sum_usage([item.usage for item in ordered])
    row_cost = _sum_cost([item.cost for item in ordered])
    payload: dict[str, Any] = {
        "ledger_version": LEDGER_VERSION,
        "status": "complete_32_row_terminal_hosted_ledger",
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
        "cost_authorization_sha256": authorization.authorization_sha256,
        "preflight_sha256": preflight.preflight_sha256,
        "smoke_sha256": smoke.smoke_sha256,
        "row_membership_sha256": bundle.row_membership_sha256,
        "preflight_call_outcomes": preflight.call_outcomes,
        "row_results": ordered,
        "row_result_sha256s": [item.row_result_sha256 for item in ordered],
        "row_status_counts": dict(sorted(Counter(item.status for item in ordered).items())),
        "publication_count": 32,
        "all_rows_terminal": True,
        "observed_source_provider_calls": sum(item.observed_provider_calls for item in ordered),
        "possible_ambiguous_source_provider_calls": sum(
            item.possible_ambiguous_provider_calls for item in ordered
        ),
        "observed_preflight_provider_calls": preflight.observed_provider_calls,
        "possible_ambiguous_preflight_provider_calls": (
            preflight.possible_ambiguous_provider_calls
        ),
        "total_provider_call_attempts_or_possible_attempts": (
            sum(
                item.observed_provider_calls + item.possible_ambiguous_provider_calls
                for item in ordered
            )
            + preflight.observed_provider_calls
            + preflight.possible_ambiguous_provider_calls
        ),
        "structured_json_schema_calls": (
            sum(item.structured_json_schema_calls for item in ordered)
            + preflight.structured_json_schema_calls
        ),
        "prompt_json_schema_calls": (
            sum(item.prompt_json_schema_calls for item in ordered)
            + preflight.prompt_json_schema_calls
        ),
        "maximum_structured_json_schema_calls": MAX_STRUCTURED_PROVIDER_CALLS,
        "maximum_prompt_json_schema_calls": MAX_PROMPT_JSON_PROVIDER_CALLS,
        "transport_mode_policy": (
            "inventory-structured-json-schema-packet-prompt-json-schema-v1"
        ),
        "possible_ambiguous_preflight_charge_ceiling_usd_micros": (
            preflight.possible_ambiguous_charge_ceiling_usd_micros
        ),
        "observed_preflight_request_ceiling_usd_micros": (
            preflight.cost.request_ceiling_usd_micros
        ),
        "observed_source_request_ceiling_usd_micros": (
            row_cost.request_ceiling_usd_micros
        ),
        "possible_ambiguous_source_charge_ceiling_usd_micros": sum(
            item.possible_ambiguous_charge_ceiling_usd_micros
            for item in ordered
        ),
        "possible_ambiguous_charge_ceiling_usd_micros": (
            preflight.possible_ambiguous_charge_ceiling_usd_micros
            + sum(
                item.possible_ambiguous_charge_ceiling_usd_micros
                for item in ordered
            )
        ),
        "observed_request_ceiling_usd_micros": (
            preflight.cost.request_ceiling_usd_micros
            + row_cost.request_ceiling_usd_micros
        ),
        "durable_intent_count": 0,
        "durable_intent_liability_usd_micros": 0,
        "durable_intent_roster_sha256": "",
        "cost_authorization_ceiling_usd_micros": (
            authorization.cost_ceiling_usd_micros
        ),
        "configured_cost_ceiling_usd_micros": (
            bundle.maximum_authorized_cost_usd_micros
        ),
        "maximum_theoretical_provider_calls": 296,
        "application_retries": 0,
        "sdk_retries": 0,
        "usage": _sum_usage([preflight.usage, row_usage]),
        "cost": _sum_cost([preflight.cost, row_cost]),
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
    }
    all_outcomes = [
        *preflight.call_outcomes,
        *(
            outcome
            for row in ordered
            for outcome in (row.inventory_outcome, *row.packet_outcomes)
        ),
    ]
    liability = _durable_intent_liability_surface(all_outcomes)
    payload.update(liability)
    return MetaSynHostedLedgerV1.model_validate(
        {**payload, "ledger_sha256": hash_canonical(payload)}
    )


def _validate_exact_durable_intent_artifact_set(
    *,
    workspace: Path,
    bundle: MetaSynHostedExecutionBundleV1,
    authorization: MetaSynHostedCostAuthorizationReceiptV1,
    outcomes: Sequence[MetaSynHostedAttemptOutcomeRefV1],
) -> dict[str, int | str]:
    """Externally replay the exact intent/outcome files behind final liability."""

    paths = metasyn_hosted_runtime_paths(workspace)

    def artifacts(name: Literal["intents", "receipts", "incidents"]) -> dict[str, Path]:
        directory = paths[name]
        if directory.is_symlink() or not directory.is_dir():
            raise MetaSynHostedRuntimeError(
                "metasyn_hosted_durable_intent_directory_unsafe"
            )
        entries = list(directory.iterdir())
        if any(
            entry.is_symlink()
            or not entry.is_file()
            or entry.suffix != ".json"
            or not entry.stem
            for entry in entries
        ):
            raise MetaSynHostedRuntimeError(
                "metasyn_hosted_durable_intent_artifact_topology_invalid"
            )
        if len({entry.stem for entry in entries}) != len(entries):
            raise MetaSynHostedRuntimeError(
                "metasyn_hosted_durable_intent_artifact_name_duplicate"
            )
        return {entry.stem: entry for entry in entries}

    expected = {item.request_key: item for item in outcomes}
    if len(expected) != len(outcomes):
        raise MetaSynHostedRuntimeError(
            "metasyn_hosted_durable_intent_expected_roster_duplicate"
        )
    intent_files = artifacts("intents")
    receipt_files = artifacts("receipts")
    incident_files = artifacts("incidents")
    expected_receipts = {
        item.request_key for item in outcomes if item.state == "response"
    }
    expected_incidents = {
        item.request_key for item in outcomes if item.state == "incident"
    }
    if (
        set(intent_files) != set(expected)
        or set(receipt_files) != expected_receipts
        or set(incident_files) != expected_incidents
        or set(receipt_files) & set(incident_files)
    ):
        raise MetaSynHostedRuntimeError(
            "metasyn_hosted_durable_intent_artifact_roster_mismatch"
        )

    for request_key, ref in expected.items():
        intent = MetaSynHostedAttemptIntentV1.model_validate(
            _read_json_object(intent_files[request_key])
        )
        if (
            intent.request_key != request_key
            or intent.execution_bundle_sha256 != bundle.execution_bundle_sha256
            or intent.cost_authorization_sha256 != authorization.authorization_sha256
            or intent.attempt_id != ref.attempt_id
            or intent.attempt_intent_sha256 != ref.attempt_intent_sha256
            or intent.request_cost_ceiling_usd_micros
            != ref.request_cost_ceiling_usd_micros
            or intent.schema_kind != ref.schema_kind
            or intent.effect_kind != ref.effect_kind
            or intent.transport_mode != ref.transport_mode
        ):
            raise MetaSynHostedRuntimeError(
                "metasyn_hosted_durable_intent_binding_mismatch"
            )
        if ref.state == "response":
            receipt = MetaSynHostedCallReceiptV1.model_validate(
                _read_json_object(receipt_files[request_key])
            )
            if (
                receipt.request_key != request_key
                or receipt.attempt_id != intent.attempt_id
                or receipt.attempt_intent_sha256 != intent.attempt_intent_sha256
                or receipt.receipt_sha256 != ref.outcome_sha256
                or receipt.request_cost_ceiling_usd_micros
                != intent.request_cost_ceiling_usd_micros
                or receipt.schema_kind != ref.schema_kind
                or receipt.effect_kind != ref.effect_kind
                or receipt.transport_mode != ref.transport_mode
            ):
                raise MetaSynHostedRuntimeError(
                    "metasyn_hosted_durable_receipt_liability_mismatch"
                )
        else:
            incident = MetaSynHostedAmbiguityIncidentV1.model_validate(
                _read_json_object(incident_files[request_key])
            )
            if (
                incident.request_key != request_key
                or incident.attempt_id != intent.attempt_id
                or incident.attempt_intent_sha256 != intent.attempt_intent_sha256
                or incident.incident_sha256 != ref.outcome_sha256
                or incident.request_cost_ceiling_usd_micros
                != intent.request_cost_ceiling_usd_micros
                or incident.schema_kind != ref.schema_kind
                or incident.effect_kind != ref.effect_kind
                or incident.transport_mode != ref.transport_mode
            ):
                raise MetaSynHostedRuntimeError(
                    "metasyn_hosted_durable_incident_liability_mismatch"
                )

    liability = _durable_intent_liability_surface(outcomes)
    if (
        _current_intent_cost_ceiling_micros(workspace)
        != liability["durable_intent_liability_usd_micros"]
        or liability["durable_intent_liability_usd_micros"]
        > authorization.cost_ceiling_usd_micros
        or liability["durable_intent_liability_usd_micros"]
        > bundle.maximum_authorized_cost_usd_micros
    ):
        raise MetaSynHostedRuntimeError(
            "metasyn_hosted_durable_intent_liability_mismatch"
        )
    return liability


def validate_metasyn_hosted_ledger(
    *,
    ledger: MetaSynHostedLedgerV1 | Mapping[str, Any],
    workspace: Path,
    execution_bundle: MetaSynHostedExecutionBundleV1,
    authorization: MetaSynHostedCostAuthorizationReceiptV1,
    preflight: MetaSynHostedPreflightReceiptV1,
    smoke: MetaSynHostedSmokeGateV1,
) -> MetaSynHostedLedgerV1:
    canonical = MetaSynHostedLedgerV1.model_validate(ledger)
    rows = [
        _validate_saved_row_result(
            workspace=workspace,
            bundle=execution_bundle,
            authorization=authorization,
            row_ordinal=index,
            result=_read_json_object(
                metasyn_hosted_runtime_paths(workspace)["row_results"] / f"row-{index:02d}.json"
            ),
        )
        for index in range(32)
    ]
    all_outcomes = [
        *preflight.call_outcomes,
        *(
            outcome
            for row in rows
            for outcome in (row.inventory_outcome, *row.packet_outcomes)
        ),
    ]
    _validate_exact_durable_intent_artifact_set(
        workspace=workspace,
        bundle=execution_bundle,
        authorization=authorization,
        outcomes=all_outcomes,
    )
    replayed = _freeze_hosted_ledger(
        bundle=execution_bundle,
        authorization=authorization,
        preflight=preflight,
        smoke=smoke,
        row_results=rows,
    )
    if replayed != canonical:
        raise MetaSynHostedRuntimeError("metasyn_hosted_ledger_replay_mismatch")
    return canonical


def run_metasyn_hosted_full_roster(
    *,
    workspace: Path,
    repository_root: Path,
    client: HostedBoundedClientProtocol,
    expected_execution_bundle_sha256: str,
) -> MetaSynHostedLedgerV1:
    if not SHA256_RE.fullmatch(expected_execution_bundle_sha256):
        raise MetaSynHostedRuntimeError("metasyn_hosted_execution_anchor_invalid")
    canonical_workspace, initial_bundle = load_current_metasyn_hosted_execution_bundle(
        workspace=workspace,
        repository_root=repository_root,
        external_replay=True,
    )
    if initial_bundle.execution_bundle_sha256 != expected_execution_bundle_sha256:
        raise MetaSynHostedRuntimeError("metasyn_hosted_execution_anchor_mismatch")
    with _workspace_lock(canonical_workspace):
        paths = metasyn_hosted_runtime_paths(canonical_workspace)
        if paths["private_report"].exists():
            raise MetaSynHostedRuntimeError("metasyn_hosted_finalized_workspace_is_immutable")
        bundle = validate_current_metasyn_hosted_execution_bundle(
            execution_bundle=_read_json_object(paths["execution_bundle"]),
            repository_root=repository_root,
            external_replay=True,
        )
        if bundle != initial_bundle:
            raise MetaSynHostedRuntimeError("metasyn_hosted_execution_bundle_changed_before_lock")
        authorization = validate_metasyn_hosted_cost_authorization(
            receipt=_read_json_object(paths["cost_authorization"]),
            execution_bundle=bundle,
        )
        preflight = validate_metasyn_hosted_preflight(
            receipt=_read_json_object(paths["preflight"]),
            workspace=canonical_workspace,
            execution_bundle=bundle,
        )
        if preflight.status != "passed":
            raise MetaSynHostedRuntimeError("metasyn_hosted_full_roster_requires_passed_preflight")
        smoke = validate_metasyn_hosted_smoke(
            receipt=_read_json_object(paths["smoke"]),
            workspace=canonical_workspace,
            execution_bundle=bundle,
            preflight=preflight,
            authorization=authorization,
        )
        if smoke.status != "passed":
            raise MetaSynHostedRuntimeError("metasyn_hosted_full_roster_requires_passed_smoke")
        if paths["ledger"].exists():
            return validate_metasyn_hosted_ledger(
                ledger=_read_json_object(paths["ledger"]),
                workspace=canonical_workspace,
                execution_bundle=bundle,
                authorization=authorization,
                preflight=preflight,
                smoke=smoke,
            )
        rows = [
            _validate_saved_row_result(
                workspace=canonical_workspace,
                bundle=bundle,
                authorization=authorization,
                row_ordinal=0,
                result=_read_json_object(paths["row_results"] / "row-00.json"),
            )
        ]
        for row_ordinal in range(1, 32):
            rows.append(
                _run_or_resume_row(
                    workspace=canonical_workspace,
                    bundle=bundle,
                    authorization=authorization,
                    row_ordinal=row_ordinal,
                    client=client,
                )
            )
        ledger = _freeze_hosted_ledger(
            bundle=bundle,
            authorization=authorization,
            preflight=preflight,
            smoke=smoke,
            row_results=rows,
        )
        atomic_write_json(paths["ledger"], ledger)
        return validate_metasyn_hosted_ledger(
            ledger=ledger,
            workspace=canonical_workspace,
            execution_bundle=bundle,
            authorization=authorization,
            preflight=preflight,
            smoke=smoke,
        )


class MetaSynHostedPrivateYieldReportV1(ContractModel):
    report_version: Literal["metasyn-bounded-hosted-private-yield-report-v2"] = (
        PRIVATE_REPORT_VERSION
    )
    status: Literal["complete_32_row_hosted_yield_only_report"] = (
        "complete_32_row_hosted_yield_only_report"
    )
    execution_bundle_sha256: str
    runtime_pipeline_fingerprint: PipelineFingerprint
    runtime_pipeline_sha256: str
    config_sha256: str
    anthropic_config_sha256: str
    provider_identity_sha256: str
    provider_pricing_table_sha256: str
    adapter_bundle_sha256: str
    downstream_verifier_pipeline_sha256: str
    row_membership_sha256: str
    cost_authorization_sha256: str
    preflight_sha256: str
    smoke_sha256: str
    hosted_ledger_sha256: str
    row_results: Annotated[list[MetaSynHostedRowResultV1], Field(min_length=32, max_length=32)]
    row_result_sha256s: Annotated[list[str], Field(min_length=32, max_length=32)]
    provider_neutral_yield_report: MetaSynBoundedPrivateYieldReportV1 | None
    provider_neutral_yield_report_sha256: str | None
    question_count: Literal[10] = 10
    component_count: Literal[10] = 10
    publication_count: Literal[32] = 32
    row_status_counts: dict[str, Annotated[int, Field(ge=0)]]
    typed_publication_output_count: Annotated[int, Field(ge=0, le=32)]
    release_grade_typed_publication_count: Annotated[int, Field(ge=0, le=32)]
    typed_finding_count: Annotated[int, Field(ge=0)]
    observed_source_provider_calls: Annotated[int, Field(ge=0, le=288)]
    possible_ambiguous_source_provider_calls: Annotated[int, Field(ge=0, le=288)]
    synthetic_preflight_provider_calls: Annotated[int, Field(ge=0, le=8)]
    possible_ambiguous_preflight_provider_calls: Annotated[int, Field(ge=0, le=8)]
    total_provider_call_attempts_or_possible_attempts: Annotated[
        int, Field(ge=40, le=296)
    ]
    structured_json_schema_calls: Literal[35] = MAX_STRUCTURED_PROVIDER_CALLS
    prompt_json_schema_calls: Annotated[int, Field(ge=5, le=261)]
    maximum_structured_json_schema_calls: Literal[35] = MAX_STRUCTURED_PROVIDER_CALLS
    maximum_prompt_json_schema_calls: Literal[261] = MAX_PROMPT_JSON_PROVIDER_CALLS
    transport_mode_policy: Literal[
        "inventory-structured-json-schema-packet-prompt-json-schema-v1"
    ] = "inventory-structured-json-schema-packet-prompt-json-schema-v1"
    observed_preflight_request_ceiling_usd_micros: Annotated[int, Field(ge=0)]
    observed_source_request_ceiling_usd_micros: Annotated[int, Field(ge=0)]
    possible_ambiguous_preflight_charge_ceiling_usd_micros: Annotated[
        int, Field(ge=0)
    ]
    possible_ambiguous_source_charge_ceiling_usd_micros: Annotated[
        int, Field(ge=0)
    ]
    observed_request_ceiling_usd_micros: Annotated[int, Field(ge=0)]
    possible_ambiguous_charge_ceiling_usd_micros: Annotated[int, Field(ge=0)]
    durable_intent_count: Annotated[int, Field(ge=40, le=296)]
    durable_intent_liability_usd_micros: Annotated[int, Field(ge=1)]
    durable_intent_roster_sha256: str
    cost_authorization_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    configured_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    maximum_theoretical_provider_calls: Literal[296] = 296
    application_retries: Literal[0] = 0
    sdk_retries: Literal[0] = 0
    usage: MetaSynHostedUsageV1
    cost: MetaSynHostedCostV1
    operator_authorized_source_transmission: Literal[True] = True
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    direction_agreement_reported: Literal[False] = False
    extraction_accuracy_reported: Literal[False] = False
    claim_release_authority: Literal[False] = False
    permitted_metrics: Literal["contract_grounding_publication_and_synthesis_input_yield_only"] = (
        "contract_grounding_publication_and_synthesis_input_yield_only"
    )
    report_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "runtime_pipeline_sha256",
        "config_sha256",
        "anthropic_config_sha256",
        "provider_identity_sha256",
        "provider_pricing_table_sha256",
        "adapter_bundle_sha256",
        "downstream_verifier_pipeline_sha256",
        "row_membership_sha256",
        "cost_authorization_sha256",
        "preflight_sha256",
        "smoke_sha256",
        "hosted_ledger_sha256",
        "provider_neutral_yield_report_sha256",
        "durable_intent_roster_sha256",
        "report_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_hosted_report_hash_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_report(self) -> MetaSynHostedPrivateYieldReportV1:
        if self.runtime_pipeline_sha256 != (self.runtime_pipeline_fingerprint.pipeline_sha256):
            raise ValueError("metasyn_hosted_report_pipeline_hash_mismatch")
        if self.row_result_sha256s != [item.row_result_sha256 for item in self.row_results]:
            raise ValueError("metasyn_hosted_report_row_hashes_mismatch")
        if self.row_status_counts != dict(
            sorted(Counter(item.status for item in self.row_results).items())
        ):
            raise ValueError("metasyn_hosted_report_status_counts_mismatch")
        expected_provider_sha = (
            self.provider_neutral_yield_report.report_sha256
            if self.provider_neutral_yield_report is not None
            else None
        )
        if self.provider_neutral_yield_report_sha256 != expected_provider_sha:
            raise ValueError("metasyn_hosted_report_provider_yield_hash_mismatch")
        if self.provider_neutral_yield_report is not None and any(
            item.adapter_publication_result is None for item in self.row_results
        ):
            raise ValueError("metasyn_hosted_report_provider_yield_unsafe")
        typed = [item for item in self.row_results if item.status == "typed_publication_output"]
        release_grade = [item for item in typed if item.release_grade_source_grounding_eligible]
        if (
            self.typed_publication_output_count != len(typed)
            or self.release_grade_typed_publication_count != len(release_grade)
            or self.typed_finding_count != sum(item.typed_finding_count for item in typed)
        ):
            raise ValueError("metasyn_hosted_report_yield_counts_mismatch")
        if (
            self.structured_json_schema_calls
            != sum(item.structured_json_schema_calls for item in self.row_results) + 3
            or self.prompt_json_schema_calls
            != sum(item.prompt_json_schema_calls for item in self.row_results) + 5
        ):
            raise ValueError("metasyn_hosted_report_transport_mode_count_mismatch")
        if (
            self.durable_intent_count
            != self.observed_source_provider_calls
            + self.possible_ambiguous_source_provider_calls
            + self.synthetic_preflight_provider_calls
            + self.possible_ambiguous_preflight_provider_calls
            or self.total_provider_call_attempts_or_possible_attempts
            != self.durable_intent_count
            or self.observed_source_request_ceiling_usd_micros
            != sum(item.cost.request_ceiling_usd_micros for item in self.row_results)
            or self.possible_ambiguous_source_charge_ceiling_usd_micros
            != sum(
                item.possible_ambiguous_charge_ceiling_usd_micros
                for item in self.row_results
            )
            or self.observed_request_ceiling_usd_micros
            != self.observed_preflight_request_ceiling_usd_micros
            + self.observed_source_request_ceiling_usd_micros
            or self.possible_ambiguous_charge_ceiling_usd_micros
            != self.possible_ambiguous_preflight_charge_ceiling_usd_micros
            + self.possible_ambiguous_source_charge_ceiling_usd_micros
            or self.cost.request_ceiling_usd_micros
            != self.observed_request_ceiling_usd_micros
            or self.durable_intent_liability_usd_micros
            != self.observed_request_ceiling_usd_micros
            + self.possible_ambiguous_charge_ceiling_usd_micros
            or self.durable_intent_liability_usd_micros
            > self.cost_authorization_ceiling_usd_micros
            or self.durable_intent_liability_usd_micros
            > self.configured_cost_ceiling_usd_micros
        ):
            raise ValueError("metasyn_hosted_report_durable_intent_liability_mismatch")
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        if self.report_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_hosted_report_self_hash_mismatch")
        return self


def _freeze_provider_neutral_yield(
    *,
    bundle: MetaSynHostedExecutionBundleV1,
    rows: Sequence[MetaSynHostedRowResultV1],
) -> MetaSynBoundedPrivateYieldReportV1 | None:
    by_row = {item.row_context_sha256: item for item in rows}
    publication_results: list[MetaSynPublicationResultV1] = []
    for adapter_row in bundle.adapter_bundle.row_contexts:
        row = by_row.get(adapter_row.row_context_sha256)
        if row is None:
            raise MetaSynHostedRuntimeError("metasyn_hosted_provider_yield_row_roster_mismatch")
        if row.adapter_publication_result is None:
            return None
        publication_results.append(row.adapter_publication_result)
    report = freeze_metasyn_bounded_private_yield_report(
        adapter_bundle=bundle.adapter_bundle,
        publication_results=publication_results,
    )
    return validate_metasyn_bounded_private_yield_report(
        report=report, adapter_bundle=bundle.adapter_bundle
    )


def _freeze_hosted_private_report(
    *,
    bundle: MetaSynHostedExecutionBundleV1,
    authorization: MetaSynHostedCostAuthorizationReceiptV1,
    preflight: MetaSynHostedPreflightReceiptV1,
    smoke: MetaSynHostedSmokeGateV1,
    ledger: MetaSynHostedLedgerV1,
    provider_neutral_yield: MetaSynBoundedPrivateYieldReportV1 | None,
) -> MetaSynHostedPrivateYieldReportV1:
    rows = ledger.row_results
    typed = [item for item in rows if item.status == "typed_publication_output"]
    release_grade = [item for item in typed if item.release_grade_source_grounding_eligible]
    payload: dict[str, Any] = {
        "report_version": PRIVATE_REPORT_VERSION,
        "status": "complete_32_row_hosted_yield_only_report",
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "runtime_pipeline_fingerprint": bundle.runtime_pipeline_fingerprint,
        "runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
        "config_sha256": bundle.config_sha256,
        "anthropic_config_sha256": bundle.anthropic_config_sha256,
        "provider_identity_sha256": bundle.provider_identity_sha256,
        "provider_pricing_table_sha256": (bundle.anthropic_config.pricing_table_sha256),
        "adapter_bundle_sha256": bundle.adapter_bundle_sha256,
        "downstream_verifier_pipeline_sha256": (bundle.downstream_verifier_pipeline_sha256),
        "row_membership_sha256": bundle.row_membership_sha256,
        "cost_authorization_sha256": authorization.authorization_sha256,
        "preflight_sha256": preflight.preflight_sha256,
        "smoke_sha256": smoke.smoke_sha256,
        "hosted_ledger_sha256": ledger.ledger_sha256,
        "row_results": rows,
        "row_result_sha256s": [item.row_result_sha256 for item in rows],
        "provider_neutral_yield_report": provider_neutral_yield,
        "provider_neutral_yield_report_sha256": (
            provider_neutral_yield.report_sha256 if provider_neutral_yield is not None else None
        ),
        "question_count": bundle.question_count,
        "component_count": bundle.component_count,
        "publication_count": bundle.publication_count,
        "row_status_counts": ledger.row_status_counts,
        "typed_publication_output_count": len(typed),
        "release_grade_typed_publication_count": len(release_grade),
        "typed_finding_count": sum(item.typed_finding_count for item in typed),
        "observed_source_provider_calls": ledger.observed_source_provider_calls,
        "possible_ambiguous_source_provider_calls": (
            ledger.possible_ambiguous_source_provider_calls
        ),
        "synthetic_preflight_provider_calls": (preflight.observed_provider_calls),
        "possible_ambiguous_preflight_provider_calls": (
            preflight.possible_ambiguous_provider_calls
        ),
        "total_provider_call_attempts_or_possible_attempts": (
            ledger.total_provider_call_attempts_or_possible_attempts
        ),
        "structured_json_schema_calls": ledger.structured_json_schema_calls,
        "prompt_json_schema_calls": ledger.prompt_json_schema_calls,
        "maximum_structured_json_schema_calls": MAX_STRUCTURED_PROVIDER_CALLS,
        "maximum_prompt_json_schema_calls": MAX_PROMPT_JSON_PROVIDER_CALLS,
        "transport_mode_policy": ledger.transport_mode_policy,
        "observed_preflight_request_ceiling_usd_micros": (
            ledger.observed_preflight_request_ceiling_usd_micros
        ),
        "observed_source_request_ceiling_usd_micros": (
            ledger.observed_source_request_ceiling_usd_micros
        ),
        "possible_ambiguous_preflight_charge_ceiling_usd_micros": (
            ledger.possible_ambiguous_preflight_charge_ceiling_usd_micros
        ),
        "possible_ambiguous_source_charge_ceiling_usd_micros": (
            ledger.possible_ambiguous_source_charge_ceiling_usd_micros
        ),
        "observed_request_ceiling_usd_micros": (
            ledger.observed_request_ceiling_usd_micros
        ),
        "possible_ambiguous_charge_ceiling_usd_micros": (
            ledger.possible_ambiguous_charge_ceiling_usd_micros
        ),
        "durable_intent_count": ledger.durable_intent_count,
        "durable_intent_liability_usd_micros": (
            ledger.durable_intent_liability_usd_micros
        ),
        "durable_intent_roster_sha256": ledger.durable_intent_roster_sha256,
        "cost_authorization_ceiling_usd_micros": (
            ledger.cost_authorization_ceiling_usd_micros
        ),
        "configured_cost_ceiling_usd_micros": (
            ledger.configured_cost_ceiling_usd_micros
        ),
        "maximum_theoretical_provider_calls": 296,
        "application_retries": 0,
        "sdk_retries": 0,
        "usage": ledger.usage,
        "cost": ledger.cost,
        "operator_authorized_source_transmission": True,
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "direction_agreement_reported": False,
        "extraction_accuracy_reported": False,
        "claim_release_authority": False,
        "permitted_metrics": ("contract_grounding_publication_and_synthesis_input_yield_only"),
    }
    return MetaSynHostedPrivateYieldReportV1.model_validate(
        {**payload, "report_sha256": hash_canonical(payload)}
    )


def finalize_metasyn_hosted_runtime(
    *,
    workspace: Path,
    repository_root: Path,
    expected_execution_bundle_sha256: str,
) -> MetaSynHostedPrivateYieldReportV1:
    if not SHA256_RE.fullmatch(expected_execution_bundle_sha256):
        raise MetaSynHostedRuntimeError("metasyn_hosted_execution_anchor_invalid")
    canonical_workspace, bundle = load_current_metasyn_hosted_execution_bundle(
        workspace=workspace,
        repository_root=repository_root,
        external_replay=True,
    )
    if bundle.execution_bundle_sha256 != expected_execution_bundle_sha256:
        raise MetaSynHostedRuntimeError("metasyn_hosted_execution_anchor_mismatch")
    with _workspace_lock(canonical_workspace):
        paths = metasyn_hosted_runtime_paths(canonical_workspace)
        if paths["private_report"].exists():
            return validate_finalized_metasyn_hosted_runtime(
                workspace=canonical_workspace,
                repository_root=repository_root,
                expected_execution_bundle_sha256=expected_execution_bundle_sha256,
                _lock_already_held=True,
            )
        authorization = validate_metasyn_hosted_cost_authorization(
            receipt=_read_json_object(paths["cost_authorization"]),
            execution_bundle=bundle,
        )
        preflight = validate_metasyn_hosted_preflight(
            receipt=_read_json_object(paths["preflight"]),
            workspace=canonical_workspace,
            execution_bundle=bundle,
        )
        if preflight.status != "passed":
            raise MetaSynHostedRuntimeError("metasyn_hosted_finalize_requires_passed_preflight")
        smoke = validate_metasyn_hosted_smoke(
            receipt=_read_json_object(paths["smoke"]),
            workspace=canonical_workspace,
            execution_bundle=bundle,
            preflight=preflight,
            authorization=authorization,
        )
        if smoke.status != "passed":
            raise MetaSynHostedRuntimeError("metasyn_hosted_finalize_requires_passed_smoke")
        ledger = validate_metasyn_hosted_ledger(
            ledger=_read_json_object(paths["ledger"]),
            workspace=canonical_workspace,
            execution_bundle=bundle,
            authorization=authorization,
            preflight=preflight,
            smoke=smoke,
        )
        provider_yield = _freeze_provider_neutral_yield(bundle=bundle, rows=ledger.row_results)
        if provider_yield is None:
            if paths["provider_neutral_yield"].exists():
                raise MetaSynHostedRuntimeError("metasyn_hosted_unexpected_provider_yield_artifact")
        elif paths["provider_neutral_yield"].exists():
            saved = validate_metasyn_bounded_private_yield_report(
                report=_read_json_object(paths["provider_neutral_yield"]),
                adapter_bundle=bundle.adapter_bundle,
            )
            if saved != provider_yield:
                raise MetaSynHostedRuntimeError("metasyn_hosted_provider_yield_replay_mismatch")
        else:
            atomic_write_json(paths["provider_neutral_yield"], provider_yield)
        report = _freeze_hosted_private_report(
            bundle=bundle,
            authorization=authorization,
            preflight=preflight,
            smoke=smoke,
            ledger=ledger,
            provider_neutral_yield=provider_yield,
        )
        atomic_write_json(paths["private_report"], report)
        return report


def validate_finalized_metasyn_hosted_runtime(
    *,
    workspace: Path,
    repository_root: Path,
    expected_execution_bundle_sha256: str,
    _lock_already_held: bool = False,
) -> MetaSynHostedPrivateYieldReportV1:
    if not SHA256_RE.fullmatch(expected_execution_bundle_sha256):
        raise MetaSynHostedRuntimeError("metasyn_hosted_execution_anchor_invalid")
    canonical_workspace, bundle = load_current_metasyn_hosted_execution_bundle(
        workspace=workspace,
        repository_root=repository_root,
        external_replay=True,
    )
    if bundle.execution_bundle_sha256 != expected_execution_bundle_sha256:
        raise MetaSynHostedRuntimeError("metasyn_hosted_execution_anchor_mismatch")

    def validate_all() -> MetaSynHostedPrivateYieldReportV1:
        paths = metasyn_hosted_runtime_paths(canonical_workspace)
        authorization = validate_metasyn_hosted_cost_authorization(
            receipt=_read_json_object(paths["cost_authorization"]),
            execution_bundle=bundle,
        )
        preflight = validate_metasyn_hosted_preflight(
            receipt=_read_json_object(paths["preflight"]),
            workspace=canonical_workspace,
            execution_bundle=bundle,
        )
        smoke = validate_metasyn_hosted_smoke(
            receipt=_read_json_object(paths["smoke"]),
            workspace=canonical_workspace,
            execution_bundle=bundle,
            preflight=preflight,
            authorization=authorization,
        )
        ledger = validate_metasyn_hosted_ledger(
            ledger=_read_json_object(paths["ledger"]),
            workspace=canonical_workspace,
            execution_bundle=bundle,
            authorization=authorization,
            preflight=preflight,
            smoke=smoke,
        )
        provider_yield = _freeze_provider_neutral_yield(bundle=bundle, rows=ledger.row_results)
        if provider_yield is None:
            if paths["provider_neutral_yield"].exists():
                raise MetaSynHostedRuntimeError("metasyn_hosted_unexpected_provider_yield_artifact")
        else:
            saved_provider = validate_metasyn_bounded_private_yield_report(
                report=_read_json_object(paths["provider_neutral_yield"]),
                adapter_bundle=bundle.adapter_bundle,
            )
            if saved_provider != provider_yield:
                raise MetaSynHostedRuntimeError("metasyn_hosted_provider_yield_replay_mismatch")
        saved = MetaSynHostedPrivateYieldReportV1.model_validate(
            _read_json_object(paths["private_report"])
        )
        replayed = _freeze_hosted_private_report(
            bundle=bundle,
            authorization=authorization,
            preflight=preflight,
            smoke=smoke,
            ledger=ledger,
            provider_neutral_yield=provider_yield,
        )
        if replayed != saved:
            raise MetaSynHostedRuntimeError("metasyn_hosted_final_report_replay_mismatch")
        return saved

    if _lock_already_held:
        return validate_all()
    with _workspace_lock(canonical_workspace):
        return validate_all()


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_EXECUTION_WORKSPACE",
    "DEFAULT_PILOT_WORKSPACE",
    "MAX_THEORETICAL_PROVIDER_CALLS",
    "MetaSynHostedAmbiguityIncidentV1",
    "MetaSynHostedAttemptIntentV1",
    "MetaSynHostedCallReceiptV1",
    "MetaSynHostedCostAuthorizationReceiptV1",
    "MetaSynHostedExecutionBundleV1",
    "MetaSynHostedLedgerV1",
    "MetaSynHostedPreflightReceiptV1",
    "MetaSynHostedPrivateYieldReportV1",
    "MetaSynHostedRowResultV1",
    "MetaSynHostedRuntimeConfigV1",
    "MetaSynHostedRuntimeError",
    "MetaSynHostedSmokeGateV1",
    "compute_metasyn_hosted_runtime_fingerprint",
    "finalize_metasyn_hosted_runtime",
    "freeze_metasyn_hosted_attempt_intent",
    "freeze_metasyn_hosted_call_receipt",
    "freeze_metasyn_hosted_cost_authorization",
    "freeze_metasyn_hosted_execution_bundle",
    "load_current_metasyn_hosted_execution_bundle",
    "load_metasyn_hosted_runtime_config",
    "metasyn_hosted_runtime_paths",
    "prepare_metasyn_hosted_runtime",
    "run_metasyn_hosted_full_roster",
    "run_metasyn_hosted_preflight",
    "run_metasyn_hosted_smoke",
    "validate_current_metasyn_hosted_execution_bundle",
    "validate_finalized_metasyn_hosted_runtime",
    "validate_metasyn_hosted_cost_authorization",
    "validate_metasyn_hosted_ledger",
    "validate_metasyn_hosted_preflight",
    "validate_metasyn_hosted_smoke",
    "write_metasyn_hosted_execution_bundle",
]
