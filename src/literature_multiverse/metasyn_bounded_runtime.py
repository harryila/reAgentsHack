"""Fail-closed execution runtime for the label-blind MetaSyn bounded pilot.

The provider-neutral adapter freezes all source-bearing prompts and validation contracts.
This module adds the local execution state machine around those contracts: exact runtime
identity, a source-free provider-schema compatibility preflight, durable one-shot intents,
immutable response receipts or ambiguity incidents, resumable full-roster ledgers, and a
yield-only final report.  It never opens MetaSyn conclusions, directions, aggregate effects,
or test labels, and it never converts a failed call into a scientific inventory.
"""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
import platform
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from literature_multiverse.lineage import (
    OutputExistsError,
    atomic_write_json,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.local_ollama import (
    LOCAL_OLLAMA_CLIENT_VERSION,
    LocalOllamaError,
    OllamaClientProtocol,
    OllamaGenerationConfig,
    OllamaGenerationResult,
    OllamaIdentity,
)
from literature_multiverse.metasyn_bounded_adapter import (
    MetaSynBoundedAdapterBundleV1,
    MetaSynBoundedAdapterError,
    MetaSynBoundedRowContextV1,
    MetaSynInventoryValidationReceiptV1,
    MetaSynPacketCallV1,
    MetaSynPacketValidationReceiptV1,
    MetaSynPublicationResultV1,
    freeze_metasyn_bounded_adapter_bundle_from_workspace,
    freeze_metasyn_inventory_validation_receipt,
    freeze_metasyn_packet_call,
    freeze_metasyn_packet_validation_receipt,
    freeze_metasyn_publication_result,
    validate_metasyn_bounded_adapter_bundle_external_replay,
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
    PROVIDER_GRAMMAR_SCOPE_V2,
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

RUNTIME_VERSION = "metasyn-bounded-local-ollama-runtime-v2"
CONFIG_VERSION = "metasyn-bounded-local-ollama-config-v1"
EXECUTION_BUNDLE_VERSION = "metasyn-bounded-execution-bundle-v2"
ATTEMPT_INTENT_VERSION = "metasyn-bounded-attempt-intent-v2"
GENERATION_RECEIPT_VERSION = "metasyn-bounded-generation-receipt-v2"
AMBIGUITY_INCIDENT_VERSION = "metasyn-bounded-ambiguity-incident-v2"
PREFLIGHT_VERSION = (
    "metasyn-bounded-provider-schema-canonical-compatibility-preflight-v4"
)
PREFLIGHT_CALL_RECEIPT_VERSION = "metasyn-bounded-preflight-call-receipt-v4"
PREFLIGHT_FIXTURE_COMPARISON_VERSION = (
    "metasyn-bounded-preflight-declared-default-equivalence-v1"
)
PREFLIGHT_FIXTURE_COMPARISON_MODE = (
    "independent-full-raw-typed-preservation-then-exact-canonical-with-only-"
    "pydantic-declared-default-omissions"
)
PREFLIGHT_PASSED_STATUS = (
    "passed_eight_call_three_inventory_state_provider_schema_canonical_"
    "compatibility_preflight"
)
PREFLIGHT_INVENTORY_STATES = (
    "candidates_found",
    "no_candidate_found",
    "overflow_or_uncertain",
)
PREFLIGHT_CALL_COUNT = 8
LEDGER_VERSION = "metasyn-bounded-prediction-ledger-v1"
ROW_RESULT_VERSION = "metasyn-bounded-runtime-row-result-v1"
PRIVATE_REPORT_VERSION = "metasyn-bounded-runtime-private-yield-report-v1"
PUBLIC_SUMMARY_VERSION = "metasyn-bounded-runtime-public-yield-summary-v1"
RUNTIME_COMPONENT_VERSION = "2"

DEFAULT_CONFIG_PATH = Path("configs/benchmarks/metasyn-bounded-qwen-v1.json")
DEFAULT_EXECUTION_WORKSPACE = Path("data/cache/metasyn/bounded-qwen-yield-v2")
DEFAULT_PILOT_WORKSPACE = Path("data/cache/metasyn/typed-oracle-pilot-v2")

_MODULE_ENTRYPOINT = "src/literature_multiverse/metasyn_bounded_runtime.py"
_SCRIPT_ENTRYPOINT = "scripts/run_metasyn_bounded_runtime.py"
_RUNTIME_DEPENDENCY_ENTRYPOINTS = (_MODULE_ENTRYPOINT, _SCRIPT_ENTRYPOINT)
_RUNTIME_NON_PYTHON_INPUTS = (
    DEFAULT_CONFIG_PATH.as_posix(),
    "prompts/metasyn_candidate_inventory.md",
    "prompts/metasyn_candidate_packet.md",
    "pyproject.toml",
    "uv.lock",
)
_HEX = re.compile(r"^[0-9a-f]{64}$")

InventoryResponseStatus = Literal[
    "inventory_valid_candidates",
    "inventory_valid_no_candidate_non_authorizing",
    "inventory_valid_capacity_or_uncertainty_non_authorizing",
    "generation_truncated",
    "generation_terminal_reason_invalid",
    "response_json_invalid",
    "inventory_contract_invalid",
]
PacketResponseStatus = Literal[
    "packet_completed",
    "packet_unable_to_complete",
    "generation_truncated",
    "generation_terminal_reason_invalid",
    "response_json_invalid",
    "packet_contract_invalid",
    "packet_source_grounding_invalid",
]
GenerationResponseStatus = InventoryResponseStatus | PacketResponseStatus
PreflightResponseStatus = Literal[
    "passed",
    "generation_model_invalid",
    "generation_truncated",
    "generation_terminal_reason_invalid",
    "response_json_invalid",
    "full_acceptance_or_typed_validation_invalid",
    "canonical_semantic_fixture_mismatch",
    "fixture_difference_not_declared_default_omission",
]
IncidentKind = Literal[
    "pre_request_identity_unavailable",
    "pre_request_identity_mismatch",
    "generation_transport_ambiguous",
    "post_response_identity_unavailable",
    "post_response_identity_mismatch",
    "orphan_intent_observed_on_resume",
]


class MetaSynBoundedRuntimeError(ValueError):
    """The current execution context or immutable one-shot ledger is unsafe."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=1)
def _native_schema_v2_contract_json() -> str:
    return _canonical_json_text(schema_v2_contract())


def _current_native_schema_v2_contract() -> dict[str, Any]:
    value = json.loads(_native_schema_v2_contract_json())
    if not isinstance(value, dict):  # pragma: no cover - construction invariant
        raise MetaSynBoundedRuntimeError("metasyn_runtime_schema_v2_contract_invalid")
    return value


@lru_cache(maxsize=1)
def _synthetic_schema_v2_preflight_specs_json() -> str:
    specs = synthetic_schema_v2_preflight_specs()
    if synthetic_schema_v2_preflight_fingerprint() != hash_canonical(specs):
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_schema_v2_preflight_fingerprint_internal_mismatch"
        )
    return _canonical_json_text(specs)


def _current_synthetic_schema_v2_preflight_specs() -> list[dict[str, Any]]:
    value = json.loads(_synthetic_schema_v2_preflight_specs_json())
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise MetaSynBoundedRuntimeError("metasyn_runtime_schema_v2_preflight_specs_invalid")
    return value


def _current_schema_v2_preflight_fingerprint() -> str:
    return hash_canonical(_current_synthetic_schema_v2_preflight_specs())


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


def _runtime_python_dependency_closure(repository_root: Path) -> list[str]:
    pending = list(_RUNTIME_DEPENDENCY_ENTRYPOINTS)
    observed = {"src/literature_multiverse/__init__.py"}
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        source_path = repository_root / relative
        if not source_path.is_file():
            raise MetaSynBoundedRuntimeError(
                f"metasyn_runtime_dependency_missing:{relative}"
            )
        observed.add(relative)
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise MetaSynBoundedRuntimeError(
                f"metasyn_runtime_dependency_unreadable:{relative}"
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


class ExpectedOllamaIdentityV1(ContractModel):
    identity_version: Literal["ollama-local-runtime-identity-v1"]
    client_version: Literal["strict-localhost-ollama-client-v1"]
    ollama_version: Annotated[str, Field(min_length=1, max_length=64)]
    model: Annotated[str, Field(min_length=1, max_length=256)]
    model_digest: str
    parameter_size: Annotated[str, Field(min_length=1, max_length=64)]
    quantization_level: Annotated[str, Field(min_length=1, max_length=64)]
    model_format: Annotated[str, Field(min_length=1, max_length=64)]
    model_family: Annotated[str, Field(min_length=1, max_length=64)]

    @field_validator("model_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("metasyn_runtime_expected_model_digest_invalid")
        return value


class MetaSynBoundedRuntimeConfigV1(ContractModel):
    config_version: Literal["metasyn-bounded-local-ollama-config-v1"] = CONFIG_VERSION
    diagnostic_scope: Literal[
        "label_blind_calibration_oracle_corpus_yield_only"
    ] = "label_blind_calibration_oracle_corpus_yield_only"
    runtime_provider: Literal["local_ollama"] = "local_ollama"
    question_count: Literal[10] = EXPECTED_SELECTED_QUESTIONS
    publication_count: Literal[32] = EXPECTED_SELECTED_PAPERS
    reference_fields_opened: Literal[False] = False
    model_calls_per_request: Literal[1] = 1
    retries_per_request: Literal[0] = 0
    expected_model_identity: ExpectedOllamaIdentityV1
    inventory_generation_config: OllamaGenerationConfig
    packet_generation_config: OllamaGenerationConfig
    config_sha256: str

    @field_validator("config_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("metasyn_runtime_config_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_config(self) -> MetaSynBoundedRuntimeConfigV1:
        inventory = self.inventory_generation_config
        packet = self.packet_generation_config
        expected = self.expected_model_identity
        if inventory.model_copy(update={"num_predict": packet.num_predict}) != packet:
            raise ValueError("metasyn_runtime_generation_configs_diverge_beyond_limit")
        if (
            inventory.model != expected.model
            or inventory.model_digest != expected.model_digest
            or inventory.expected_ollama_version != expected.ollama_version
            or expected.client_version != LOCAL_OLLAMA_CLIENT_VERSION
        ):
            raise ValueError("metasyn_runtime_config_identity_mismatch")
        payload = self.model_dump(mode="json", exclude={"config_sha256"})
        if hash_canonical(payload) != self.config_sha256:
            raise ValueError("metasyn_runtime_config_hash_mismatch")
        return self


def _safe_repo_file(
    *, repository_root: Path, path: Path, expected_relative: Path | None = None
) -> Path:
    root = repository_root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise MetaSynBoundedRuntimeError("metasyn_runtime_repository_file_symlink")
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise MetaSynBoundedRuntimeError("metasyn_runtime_repository_file_unsafe") from exc
    if expected_relative is not None and relative != expected_relative:
        raise MetaSynBoundedRuntimeError("metasyn_runtime_repository_file_not_exact")
    if not resolved.is_file():
        raise MetaSynBoundedRuntimeError("metasyn_runtime_repository_file_missing")
    return resolved


def _read_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MetaSynBoundedRuntimeError("metasyn_runtime_json_artifact_unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynBoundedRuntimeError("metasyn_runtime_json_artifact_invalid") from exc
    if not isinstance(value, dict):
        raise MetaSynBoundedRuntimeError("metasyn_runtime_json_artifact_not_object")
    return value


def load_metasyn_bounded_runtime_config(
    *, repository_root: Path, config_path: Path = DEFAULT_CONFIG_PATH
) -> tuple[MetaSynBoundedRuntimeConfigV1, str]:
    path = _safe_repo_file(
        repository_root=repository_root,
        path=config_path,
        expected_relative=DEFAULT_CONFIG_PATH,
    )
    return MetaSynBoundedRuntimeConfigV1.model_validate(_read_json_object(path)), sha256_file(
        path
    )


def _pilot_downstream_verifier_sha256(repository_root: Path) -> tuple[str, str]:
    pilot = compute_metasyn_typed_pilot_pipeline_fingerprint(root=repository_root)
    if len(pilot.components) != 1:
        raise MetaSynBoundedRuntimeError("metasyn_runtime_pilot_component_count_invalid")
    downstream = pilot.components[0].settings.get("downstream_verifier_pipeline_sha256")
    if not isinstance(downstream, str) or not SHA256_RE.fullmatch(downstream):
        raise MetaSynBoundedRuntimeError("metasyn_runtime_downstream_pipeline_missing")
    return pilot.pipeline_sha256, downstream


def compute_metasyn_bounded_runtime_fingerprint(
    *,
    repository_root: Path,
    adapter_pipeline_sha256: str,
    config_sha256: str,
    downstream_verifier_pipeline_sha256: str,
) -> PipelineFingerprint:
    for value in (
        adapter_pipeline_sha256,
        config_sha256,
        downstream_verifier_pipeline_sha256,
    ):
        if not SHA256_RE.fullmatch(value):
            raise MetaSynBoundedRuntimeError("metasyn_runtime_fingerprint_input_invalid")
    root = repository_root.resolve(strict=True)
    native_v2_contract = _current_native_schema_v2_contract()
    provider_scope = deepcopy(PROVIDER_GRAMMAR_SCOPE_V2)
    preflight_fingerprint = _current_schema_v2_preflight_fingerprint()
    component = PipelineComponentSpec(
        component_id="metasyn-bounded-local-ollama-runtime",
        component_version=RUNTIME_COMPONENT_VERSION,
        file_paths=sorted(
            {
                *_runtime_python_dependency_closure(root),
                *_RUNTIME_NON_PYTHON_INPUTS,
            }
        ),
        settings={
            "adapter_pipeline_sha256": adapter_pipeline_sha256,
            "ambiguity_incident_never_treated_as_model_response": True,
            "canonical_workspace_lock": True,
            "config_sha256": config_sha256,
            "dependency_closure_entrypoints": list(_RUNTIME_DEPENDENCY_ENTRYPOINTS),
            "downstream_verifier_pipeline_sha256": (
                downstream_verifier_pipeline_sha256
            ),
            "full_row_roster_required_for_finalization": True,
            "in_repository_dependency_closure_bound": True,
            "installed_dependency_versions": {
                name: distribution_version(name)
                for name in ("jsonschema", "pydantic")
            },
            "model_calls_per_request": 1,
            "native_schema_v2_contract_sha256": native_v2_contract[
                "contract_sha256"
            ],
            "no_fabricated_inventory_on_runtime_failure": True,
            "platform_machine": platform.machine(),
            "platform_system": platform.system(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "reference_fields_opened": False,
            "retries_per_request": 0,
            "provider_grammar_scope_sha256": hash_canonical(provider_scope),
            "provider_schema_has_scientific_authority": False,
            "raw_response_validated_against_full_schema_v2": True,
            "schema_v2_preflight_fingerprint": preflight_fingerprint,
            "eight_call_three_inventory_state_synthetic_provider_schema_preflight_required": True,
            "preflight_fixture_comparison": (
                PREFLIGHT_FIXTURE_COMPARISON_MODE
            ),
            "preflight_raw_validation_precedes_canonical_comparison": True,
            "preflight_declared_pydantic_default_omission_is_equivalent": True,
            "preflight_nondefault_omission_is_equivalent": False,
            "preflight_scientific_value_or_lexeme_changes_are_equivalent": False,
            "yield_only_no_accuracy_direction_or_release_authority": True,
        },
    )
    return compute_pipeline_fingerprint(root=root, components=[component])


class MetaSynBoundedExecutionBundleV1(ContractModel):
    execution_bundle_version: Literal["metasyn-bounded-execution-bundle-v2"] = (
        EXECUTION_BUNDLE_VERSION
    )
    runtime_version: Literal["metasyn-bounded-local-ollama-runtime-v2"] = (
        RUNTIME_VERSION
    )
    status: Literal["frozen_label_blind_runtime_no_model_calls"] = (
        "frozen_label_blind_runtime_no_model_calls"
    )
    pilot_workspace_relative: Annotated[str, Field(min_length=1, max_length=2048)]
    config_path: Literal["configs/benchmarks/metasyn-bounded-qwen-v1.json"] = (
        DEFAULT_CONFIG_PATH.as_posix()
    )
    config_file_sha256: str
    runtime_config: MetaSynBoundedRuntimeConfigV1
    config_sha256: str
    adapter_bundle: MetaSynBoundedAdapterBundleV1
    adapter_bundle_sha256: str
    adapter_pipeline_sha256: str
    upstream_pilot_pipeline_sha256: str
    downstream_verifier_pipeline_sha256: str
    native_schema_v2_contract: dict[str, Any]
    native_schema_v2_contract_sha256: str
    provider_grammar_scope: dict[str, Any]
    provider_grammar_scope_sha256: str
    schema_v2_preflight_fingerprint: str
    runtime_pipeline_fingerprint: PipelineFingerprint
    runtime_pipeline_sha256: str
    question_count: Literal[10] = EXPECTED_SELECTED_QUESTIONS
    component_count: Literal[10] = EXPECTED_SELECTED_COMPONENTS
    publication_count: Literal[32] = EXPECTED_SELECTED_PAPERS
    question_membership_sha256: str
    component_membership_sha256: str
    row_membership_sha256: str
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    model_calls_made: Literal[False] = False
    permitted_metrics: Literal[
        "contract_grounding_publication_and_synthesis_input_yield_only"
    ] = "contract_grounding_publication_and_synthesis_input_yield_only"
    execution_bundle_sha256: str

    @field_validator(
        "config_file_sha256",
        "config_sha256",
        "adapter_bundle_sha256",
        "adapter_pipeline_sha256",
        "upstream_pilot_pipeline_sha256",
        "downstream_verifier_pipeline_sha256",
        "native_schema_v2_contract_sha256",
        "provider_grammar_scope_sha256",
        "schema_v2_preflight_fingerprint",
        "runtime_pipeline_sha256",
        "question_membership_sha256",
        "component_membership_sha256",
        "row_membership_sha256",
        "execution_bundle_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_runtime_bundle_sha256_invalid:{info.field_name}")
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
            raise ValueError("metasyn_runtime_pilot_workspace_path_unsafe")
        return value

    @model_validator(mode="after")
    def validate_bundle(self) -> MetaSynBoundedExecutionBundleV1:
        adapter = self.adapter_bundle
        if self.config_sha256 != self.runtime_config.config_sha256:
            raise ValueError("metasyn_runtime_config_hash_alias_mismatch")
        if self.adapter_bundle_sha256 != adapter.adapter_bundle_sha256:
            raise ValueError("metasyn_runtime_adapter_bundle_hash_alias_mismatch")
        if self.adapter_pipeline_sha256 != adapter.adapter_pipeline_sha256:
            raise ValueError("metasyn_runtime_adapter_pipeline_hash_alias_mismatch")
        if self.upstream_pilot_pipeline_sha256 != adapter.upstream_pilot_pipeline_sha256:
            raise ValueError("metasyn_runtime_upstream_pipeline_hash_alias_mismatch")
        if self.runtime_pipeline_sha256 != self.runtime_pipeline_fingerprint.pipeline_sha256:
            raise ValueError("metasyn_runtime_pipeline_hash_alias_mismatch")
        current_contract = _current_native_schema_v2_contract()
        if (
            self.native_schema_v2_contract != current_contract
            or self.native_schema_v2_contract_sha256
            != current_contract["contract_sha256"]
        ):
            raise ValueError("metasyn_runtime_native_schema_v2_contract_mismatch")
        if (
            self.provider_grammar_scope != PROVIDER_GRAMMAR_SCOPE_V2
            or self.provider_grammar_scope_sha256
            != hash_canonical(PROVIDER_GRAMMAR_SCOPE_V2)
        ):
            raise ValueError("metasyn_runtime_provider_grammar_scope_mismatch")
        if (
            self.schema_v2_preflight_fingerprint
            != _current_schema_v2_preflight_fingerprint()
        ):
            raise ValueError("metasyn_runtime_schema_v2_preflight_fingerprint_mismatch")
        if len(self.runtime_pipeline_fingerprint.components) != 1:
            raise ValueError("metasyn_runtime_pipeline_component_count_mismatch")
        component = self.runtime_pipeline_fingerprint.components[0]
        if component.component_id != "metasyn-bounded-local-ollama-runtime":
            raise ValueError("metasyn_runtime_pipeline_component_mismatch")
        expected_settings = {
            "adapter_pipeline_sha256": self.adapter_pipeline_sha256,
            "config_sha256": self.config_sha256,
            "downstream_verifier_pipeline_sha256": (
                self.downstream_verifier_pipeline_sha256
            ),
            "native_schema_v2_contract_sha256": (
                self.native_schema_v2_contract_sha256
            ),
            "provider_grammar_scope_sha256": self.provider_grammar_scope_sha256,
            "schema_v2_preflight_fingerprint": (
                self.schema_v2_preflight_fingerprint
            ),
        }
        if any(component.settings.get(key) != value for key, value in expected_settings.items()):
            raise ValueError("metasyn_runtime_pipeline_setting_mismatch")
        if (
            self.question_count != adapter.question_count
            or self.component_count != adapter.component_count
            or self.publication_count != adapter.publication_count
            or self.question_membership_sha256 != adapter.question_membership_sha256
            or self.component_membership_sha256 != adapter.component_membership_sha256
            or self.row_membership_sha256 != adapter.row_membership_sha256
        ):
            raise ValueError("metasyn_runtime_adapter_roster_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"execution_bundle_sha256"})
        if hash_canonical(payload) != self.execution_bundle_sha256:
            raise ValueError("metasyn_runtime_execution_bundle_hash_mismatch")
        return self


def _relative_private_workspace(path: Path, *, repository_root: Path) -> str:
    root = repository_root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    if candidate.is_symlink():
        raise MetaSynBoundedRuntimeError("metasyn_runtime_pilot_workspace_symlink")
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise MetaSynBoundedRuntimeError("metasyn_runtime_pilot_workspace_unsafe") from exc
    if not resolved.is_dir() or not relative.as_posix().startswith("data/cache/"):
        raise MetaSynBoundedRuntimeError("metasyn_runtime_pilot_workspace_not_private")
    return relative.as_posix()


def freeze_metasyn_bounded_execution_bundle(
    *,
    adapter_bundle: MetaSynBoundedAdapterBundleV1 | Mapping[str, Any],
    runtime_config: MetaSynBoundedRuntimeConfigV1 | Mapping[str, Any],
    config_file_sha256: str,
    pilot_workspace_relative: str,
    repository_root: Path,
) -> MetaSynBoundedExecutionBundleV1:
    root = repository_root.resolve(strict=True)
    adapter = MetaSynBoundedAdapterBundleV1.model_validate(adapter_bundle)
    config = MetaSynBoundedRuntimeConfigV1.model_validate(runtime_config)
    current_pilot_sha, downstream_sha = _pilot_downstream_verifier_sha256(root)
    if current_pilot_sha != adapter.upstream_pilot_pipeline_sha256:
        raise MetaSynBoundedRuntimeError("metasyn_runtime_adapter_upstream_stale")
    pipeline = compute_metasyn_bounded_runtime_fingerprint(
        repository_root=root,
        adapter_pipeline_sha256=adapter.adapter_pipeline_sha256,
        config_sha256=config.config_sha256,
        downstream_verifier_pipeline_sha256=downstream_sha,
    )
    native_v2_contract = _current_native_schema_v2_contract()
    provider_scope = deepcopy(PROVIDER_GRAMMAR_SCOPE_V2)
    payload: dict[str, Any] = {
        "execution_bundle_version": EXECUTION_BUNDLE_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "status": "frozen_label_blind_runtime_no_model_calls",
        "pilot_workspace_relative": pilot_workspace_relative,
        "config_path": DEFAULT_CONFIG_PATH.as_posix(),
        "config_file_sha256": config_file_sha256,
        "runtime_config": config,
        "config_sha256": config.config_sha256,
        "adapter_bundle": adapter,
        "adapter_bundle_sha256": adapter.adapter_bundle_sha256,
        "adapter_pipeline_sha256": adapter.adapter_pipeline_sha256,
        "upstream_pilot_pipeline_sha256": adapter.upstream_pilot_pipeline_sha256,
        "downstream_verifier_pipeline_sha256": downstream_sha,
        "native_schema_v2_contract": native_v2_contract,
        "native_schema_v2_contract_sha256": native_v2_contract[
            "contract_sha256"
        ],
        "provider_grammar_scope": provider_scope,
        "provider_grammar_scope_sha256": hash_canonical(provider_scope),
        "schema_v2_preflight_fingerprint": (
            _current_schema_v2_preflight_fingerprint()
        ),
        "runtime_pipeline_fingerprint": pipeline,
        "runtime_pipeline_sha256": pipeline.pipeline_sha256,
        "question_count": adapter.question_count,
        "component_count": adapter.component_count,
        "publication_count": adapter.publication_count,
        "question_membership_sha256": adapter.question_membership_sha256,
        "component_membership_sha256": adapter.component_membership_sha256,
        "row_membership_sha256": adapter.row_membership_sha256,
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "model_calls_made": False,
        "permitted_metrics": (
            "contract_grounding_publication_and_synthesis_input_yield_only"
        ),
    }
    return MetaSynBoundedExecutionBundleV1.model_validate(
        {**payload, "execution_bundle_sha256": hash_canonical(payload)}
    )


def prepare_metasyn_bounded_execution_bundle(
    *,
    repository_root: Path,
    pilot_workspace: Path = DEFAULT_PILOT_WORKSPACE,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> MetaSynBoundedExecutionBundleV1:
    root = repository_root.resolve(strict=True)
    relative_workspace = _relative_private_workspace(
        pilot_workspace, repository_root=root
    )
    adapter = freeze_metasyn_bounded_adapter_bundle_from_workspace(
        repository_root=root, workspace=root / relative_workspace
    )
    config, file_sha = load_metasyn_bounded_runtime_config(
        repository_root=root, config_path=config_path
    )
    return freeze_metasyn_bounded_execution_bundle(
        adapter_bundle=adapter,
        runtime_config=config,
        config_file_sha256=file_sha,
        pilot_workspace_relative=relative_workspace,
        repository_root=root,
    )


def validate_current_metasyn_bounded_execution_bundle(
    *,
    execution_bundle: MetaSynBoundedExecutionBundleV1 | Mapping[str, Any],
    repository_root: Path,
    external_replay: bool = True,
) -> MetaSynBoundedExecutionBundleV1:
    root = repository_root.resolve(strict=True)
    canonical = MetaSynBoundedExecutionBundleV1.model_validate(execution_bundle)
    config, file_sha = load_metasyn_bounded_runtime_config(repository_root=root)
    if config != canonical.runtime_config or file_sha != canonical.config_file_sha256:
        raise MetaSynBoundedRuntimeError("metasyn_runtime_current_config_mismatch")
    pilot_workspace = root / canonical.pilot_workspace_relative
    if external_replay:
        validate_metasyn_bounded_adapter_bundle_external_replay(
            adapter_bundle=canonical.adapter_bundle,
            repository_root=root,
            workspace=pilot_workspace,
        )
    replayed = freeze_metasyn_bounded_execution_bundle(
        adapter_bundle=canonical.adapter_bundle,
        runtime_config=config,
        config_file_sha256=file_sha,
        pilot_workspace_relative=canonical.pilot_workspace_relative,
        repository_root=root,
    )
    if replayed != canonical:
        raise MetaSynBoundedRuntimeError("metasyn_runtime_current_bundle_replay_mismatch")
    return canonical


def _expected_identity(
    config: MetaSynBoundedRuntimeConfigV1,
) -> OllamaIdentity:
    return OllamaIdentity.model_validate(
        config.expected_model_identity.model_dump(mode="json")
    )


def _require_exact_identity(
    identity: OllamaIdentity,
    *,
    runtime_config: MetaSynBoundedRuntimeConfigV1,
    generation_config: OllamaGenerationConfig,
) -> OllamaIdentity:
    canonical = OllamaIdentity.model_validate(identity)
    if canonical != _expected_identity(runtime_config):
        raise MetaSynBoundedRuntimeError("metasyn_runtime_model_identity_mismatch")
    if (
        canonical.model != generation_config.model
        or canonical.model_digest != generation_config.model_digest
        or canonical.ollama_version != generation_config.expected_ollama_version
    ):
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_generation_config_identity_mismatch"
        )
    return canonical


def _strict_json_loads(value: str) -> dict[str, Any]:
    def reject_constant(raw: str) -> None:
        raise ValueError(f"nonfinite_constant:{raw}")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate_object_key")
            result[key] = item
        return result

    try:
        parsed = json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=object_pairs,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise MetaSynBoundedRuntimeError("metasyn_runtime_response_json_invalid") from exc
    if not isinstance(parsed, dict):
        raise MetaSynBoundedRuntimeError("metasyn_runtime_response_json_not_object")
    return parsed


def _row_by_hash(
    bundle: MetaSynBoundedExecutionBundleV1,
) -> dict[str, MetaSynBoundedRowContextV1]:
    return {
        row.row_context_sha256: row
        for row in bundle.adapter_bundle.row_contexts
    }


@lru_cache(maxsize=1024)
def _schema_pair_json(
    kind: Literal["inventory", "packet"], context_json: str
) -> tuple[str, str]:
    context = json.loads(context_json)
    if not isinstance(context, dict):  # pragma: no cover - internal construction
        raise MetaSynBoundedRuntimeError("metasyn_runtime_schema_context_invalid")
    if kind == "inventory":
        schema_bundle = inventory_schema_bundle_v2(
            exposed_line_ids=context["exposed_line_ids"],
            allowed_outcomes=context["allowed_outcomes"],
        )
    else:
        schema_bundle = packet_schema_bundle_v2(
            candidate=NativeCandidateDescriptor.model_validate(context["candidate"]),
            exposed_line_ids=context["exposed_line_ids"],
            source_locator=context["source_locator"],
            allowed_outcomes=context["allowed_outcomes"],
            allowed_moderators=context["allowed_moderators"],
            allowed_sections=context["allowed_sections"],
            outcome_positive_directions=context["outcome_positive_directions"],
        )
    binding = schema_bundle_receipt_binding_v2(schema_bundle)
    return _canonical_json_text(schema_bundle), _canonical_json_text(binding)


def _schema_pair(
    *, kind: Literal["inventory", "packet"], context: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle_json, binding_json = _schema_pair_json(
        kind, _canonical_json_text(dict(context))
    )
    schema_bundle = json.loads(bundle_json)
    binding = json.loads(binding_json)
    if not isinstance(schema_bundle, dict) or not isinstance(binding, dict):
        raise MetaSynBoundedRuntimeError("metasyn_runtime_schema_pair_cache_invalid")
    return schema_bundle, binding


def _request_surface(
    *,
    row: MetaSynBoundedRowContextV1,
    stage: Literal["inventory", "packet"],
    inventory_receipt: MetaSynInventoryValidationReceiptV1 | None = None,
    candidate_index: int | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], MetaSynPacketCallV1 | None]:
    if stage == "inventory":
        if inventory_receipt is not None or candidate_index is not None:
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_inventory_surface_has_packet_context"
            )
        schema_bundle, binding = _schema_pair(
            kind="inventory",
            context={
                "exposed_line_ids": row.source_row.projection.exposed_line_ids,
                "allowed_outcomes": row.allowed_outcomes,
            },
        )
        return row.inventory_prompt, schema_bundle, binding, None
    if inventory_receipt is None or candidate_index is None:
        raise MetaSynBoundedRuntimeError("metasyn_runtime_packet_surface_context_missing")
    call = freeze_metasyn_packet_call(
        row=row,
        inventory_receipt=inventory_receipt,
        candidate_index=candidate_index,
    )
    schema_bundle, binding = _schema_pair(
        kind="packet",
        context={
            "candidate": call.candidate.model_dump(mode="json"),
            "exposed_line_ids": row.source_row.projection.exposed_line_ids,
            "source_locator": row.source_locator,
            "allowed_outcomes": row.allowed_outcomes,
            "allowed_moderators": row.allowed_moderators,
            "allowed_sections": row.allowed_sections,
            "outcome_positive_directions": row.outcome_positive_directions,
        },
    )
    return call.rendered_prompt, schema_bundle, binding, call


class MetaSynAttemptIntentV1(ContractModel):
    attempt_intent_version: Literal["metasyn-bounded-attempt-intent-v2"] = (
        ATTEMPT_INTENT_VERSION
    )
    runtime_version: Literal["metasyn-bounded-local-ollama-runtime-v2"] = (
        RUNTIME_VERSION
    )
    status: Literal["durable_pre_call_intent_frozen"] = (
        "durable_pre_call_intent_frozen"
    )
    execution_bundle_sha256: str
    runtime_pipeline_sha256: str
    adapter_bundle_sha256: str
    row_context_sha256: str
    question_spec_sha256: str
    question_bundle_sha256: str
    stage: Literal["inventory", "packet"]
    candidate_index: int | None
    candidate_sha256: str | None
    inventory_validation_receipt_sha256: str | None
    prompt_sha256: str
    schema_sha256: str
    provider_schema_sha256: str
    full_acceptance_schema_sha256: str
    schema_bundle_sha256: str
    schema_context_binding_sha256: str
    generation_config_sha256: str
    model_identity_sha256: str
    request_sha256: str
    permitted_call_attempts: Literal[1] = 1
    generation_retries_permitted: Literal[0] = 0
    ambiguity_is_terminal_for_this_request: Literal[True] = True
    attempt_id: str
    attempt_intent_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "runtime_pipeline_sha256",
        "adapter_bundle_sha256",
        "row_context_sha256",
        "question_spec_sha256",
        "question_bundle_sha256",
        "candidate_sha256",
        "inventory_validation_receipt_sha256",
        "prompt_sha256",
        "schema_sha256",
        "provider_schema_sha256",
        "full_acceptance_schema_sha256",
        "schema_bundle_sha256",
        "schema_context_binding_sha256",
        "generation_config_sha256",
        "model_identity_sha256",
        "request_sha256",
        "attempt_id",
        "attempt_intent_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_runtime_intent_sha256_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_intent(self) -> MetaSynAttemptIntentV1:
        packet = self.stage == "packet"
        if packet != (
            self.candidate_index is not None
            and self.candidate_sha256 is not None
            and self.inventory_validation_receipt_sha256 is not None
        ):
            raise ValueError("metasyn_runtime_intent_packet_context_mismatch")
        if self.schema_sha256 != self.provider_schema_sha256:
            raise ValueError("metasyn_runtime_intent_provider_schema_alias_mismatch")
        request_payload = {
            "execution_bundle_sha256": self.execution_bundle_sha256,
            "row_context_sha256": self.row_context_sha256,
            "stage": self.stage,
            "candidate_index": self.candidate_index,
            "candidate_sha256": self.candidate_sha256,
            "inventory_validation_receipt_sha256": (
                self.inventory_validation_receipt_sha256
            ),
            "prompt_sha256": self.prompt_sha256,
            "schema_sha256": self.schema_sha256,
            "provider_schema_sha256": self.provider_schema_sha256,
            "full_acceptance_schema_sha256": self.full_acceptance_schema_sha256,
            "schema_bundle_sha256": self.schema_bundle_sha256,
            "schema_context_binding_sha256": self.schema_context_binding_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "model_identity_sha256": self.model_identity_sha256,
        }
        if hash_canonical(request_payload) != self.request_sha256:
            raise ValueError("metasyn_runtime_intent_request_hash_mismatch")
        attempt_payload = {
            "request_sha256": self.request_sha256,
            "permitted_call_attempts": 1,
            "generation_retries_permitted": 0,
        }
        if hash_canonical(attempt_payload) != self.attempt_id:
            raise ValueError("metasyn_runtime_attempt_id_mismatch")
        payload = self.model_dump(mode="json", exclude={"attempt_intent_sha256"})
        if hash_canonical(payload) != self.attempt_intent_sha256:
            raise ValueError("metasyn_runtime_intent_hash_mismatch")
        return self


def freeze_metasyn_attempt_intent(
    *,
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    row: MetaSynBoundedRowContextV1,
    stage: Literal["inventory", "packet"],
    identity: OllamaIdentity,
    inventory_receipt: MetaSynInventoryValidationReceiptV1 | None = None,
    candidate_index: int | None = None,
) -> MetaSynAttemptIntentV1:
    bundle = MetaSynBoundedExecutionBundleV1.model_validate(execution_bundle)
    row = MetaSynBoundedRowContextV1.model_validate(row)
    current_row = _row_by_hash(bundle).get(row.row_context_sha256)
    if current_row != row:
        raise MetaSynBoundedRuntimeError("metasyn_runtime_intent_row_not_in_bundle")
    prompt, _schema_bundle, schema_binding, packet_call = _request_surface(
        row=row,
        stage=stage,
        inventory_receipt=inventory_receipt,
        candidate_index=candidate_index,
    )
    config = (
        bundle.runtime_config.inventory_generation_config
        if stage == "inventory"
        else bundle.runtime_config.packet_generation_config
    )
    identity = _require_exact_identity(
        identity, runtime_config=bundle.runtime_config, generation_config=config
    )
    candidate_sha = packet_call.candidate_sha256 if packet_call is not None else None
    inventory_sha = (
        inventory_receipt.receipt_sha256 if inventory_receipt is not None else None
    )
    request_payload = {
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "row_context_sha256": row.row_context_sha256,
        "stage": stage,
        "candidate_index": candidate_index,
        "candidate_sha256": candidate_sha,
        "inventory_validation_receipt_sha256": inventory_sha,
        "prompt_sha256": _sha256_text(prompt),
        "schema_sha256": schema_binding["provider_schema_sha256"],
        "provider_schema_sha256": schema_binding["provider_schema_sha256"],
        "full_acceptance_schema_sha256": schema_binding[
            "full_acceptance_schema_sha256"
        ],
        "schema_bundle_sha256": schema_binding["schema_bundle_sha256"],
        "schema_context_binding_sha256": schema_binding["context_binding_sha256"],
        "generation_config_sha256": config.config_sha256,
        "model_identity_sha256": identity.identity_sha256,
    }
    request_sha = hash_canonical(request_payload)
    attempt_id = hash_canonical(
        {
            "request_sha256": request_sha,
            "permitted_call_attempts": 1,
            "generation_retries_permitted": 0,
        }
    )
    payload: dict[str, Any] = {
        "attempt_intent_version": ATTEMPT_INTENT_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "status": "durable_pre_call_intent_frozen",
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
        "adapter_bundle_sha256": bundle.adapter_bundle_sha256,
        "row_context_sha256": row.row_context_sha256,
        "question_spec_sha256": row.question_spec_sha256,
        "question_bundle_sha256": row.question_bundle_sha256,
        "stage": stage,
        "candidate_index": candidate_index,
        "candidate_sha256": candidate_sha,
        "inventory_validation_receipt_sha256": inventory_sha,
        **{
            key: request_payload[key]
            for key in (
                "prompt_sha256",
                "schema_sha256",
                "provider_schema_sha256",
                "full_acceptance_schema_sha256",
                "schema_bundle_sha256",
                "schema_context_binding_sha256",
                "generation_config_sha256",
                "model_identity_sha256",
            )
        },
        "request_sha256": request_sha,
        "permitted_call_attempts": 1,
        "generation_retries_permitted": 0,
        "ambiguity_is_terminal_for_this_request": True,
        "attempt_id": attempt_id,
    }
    return MetaSynAttemptIntentV1.model_validate(
        {**payload, "attempt_intent_sha256": hash_canonical(payload)}
    )


def validate_metasyn_attempt_intent(
    *,
    intent: MetaSynAttemptIntentV1 | Mapping[str, Any],
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    row: MetaSynBoundedRowContextV1,
    identity: OllamaIdentity,
    inventory_receipt: MetaSynInventoryValidationReceiptV1 | None = None,
) -> MetaSynAttemptIntentV1:
    canonical = MetaSynAttemptIntentV1.model_validate(intent)
    replayed = freeze_metasyn_attempt_intent(
        execution_bundle=execution_bundle,
        row=row,
        stage=canonical.stage,
        identity=identity,
        inventory_receipt=inventory_receipt,
        candidate_index=canonical.candidate_index,
    )
    if replayed != canonical:
        raise MetaSynBoundedRuntimeError("metasyn_runtime_intent_external_replay_mismatch")
    return canonical


class MetaSynGenerationReceiptV1(ContractModel):
    generation_receipt_version: Literal["metasyn-bounded-generation-receipt-v2"] = (
        GENERATION_RECEIPT_VERSION
    )
    runtime_version: Literal["metasyn-bounded-local-ollama-runtime-v2"] = (
        RUNTIME_VERSION
    )
    stage: Literal["inventory", "packet"]
    status: GenerationResponseStatus
    terminal: Literal[True] = True
    response_observed: Literal[True] = True
    execution_bundle_sha256: str
    row_context_sha256: str
    attempt_id: str
    attempt_intent_sha256: str
    request_sha256: str
    prompt_sha256: str
    schema_sha256: str
    provider_schema_sha256: str
    full_acceptance_schema_sha256: str
    schema_bundle_sha256: str
    schema_context_binding_sha256: str
    generation_config_sha256: str
    model_identity_sha256: str
    candidate_index: int | None
    candidate_sha256: str | None
    inventory_validation_receipt_sha256: str | None
    generation_result: OllamaGenerationResult
    generation_result_sha256: str
    response_text_sha256: str
    adapter_validation_receipt: dict[str, Any] | None
    adapter_validation_receipt_sha256: str | None
    terminal_error: str | None
    generation_call_attempts: Literal[1] = 1
    generation_retries: Literal[0] = 0
    receipt_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "row_context_sha256",
        "attempt_id",
        "attempt_intent_sha256",
        "request_sha256",
        "prompt_sha256",
        "schema_sha256",
        "provider_schema_sha256",
        "full_acceptance_schema_sha256",
        "schema_bundle_sha256",
        "schema_context_binding_sha256",
        "generation_config_sha256",
        "model_identity_sha256",
        "candidate_sha256",
        "inventory_validation_receipt_sha256",
        "generation_result_sha256",
        "response_text_sha256",
        "adapter_validation_receipt_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_runtime_receipt_sha256_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> MetaSynGenerationReceiptV1:
        if self.schema_sha256 != self.provider_schema_sha256:
            raise ValueError("metasyn_runtime_receipt_provider_schema_alias_mismatch")
        if self.generation_result_sha256 != hash_canonical(self.generation_result):
            raise ValueError("metasyn_runtime_generation_result_hash_mismatch")
        if self.response_text_sha256 != _sha256_text(
            self.generation_result.response_text
        ):
            raise ValueError("metasyn_runtime_response_text_hash_mismatch")
        expected_adapter_sha = (
            hash_canonical(self.adapter_validation_receipt)
            if self.adapter_validation_receipt is not None
            else None
        )
        if self.adapter_validation_receipt_sha256 != expected_adapter_sha:
            raise ValueError("metasyn_runtime_adapter_receipt_hash_mismatch")
        valid = self.status.startswith("inventory_valid") or self.status in {
            "packet_completed",
            "packet_unable_to_complete",
        }
        if valid != (self.adapter_validation_receipt is not None):
            raise ValueError("metasyn_runtime_adapter_receipt_presence_mismatch")
        if (self.terminal_error is None) != valid:
            raise ValueError("metasyn_runtime_terminal_error_presence_mismatch")
        packet = self.stage == "packet"
        if packet != (
            self.candidate_index is not None
            and self.candidate_sha256 is not None
            and self.inventory_validation_receipt_sha256 is not None
        ):
            raise ValueError("metasyn_runtime_receipt_packet_context_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("metasyn_runtime_generation_receipt_hash_mismatch")
        return self


class MetaSynAmbiguityIncidentV1(ContractModel):
    ambiguity_incident_version: Literal[
        "metasyn-bounded-ambiguity-incident-v2"
    ] = AMBIGUITY_INCIDENT_VERSION
    runtime_version: Literal["metasyn-bounded-local-ollama-runtime-v2"] = (
        RUNTIME_VERSION
    )
    status: Literal["terminal_ambiguous_attempt_poison"] = (
        "terminal_ambiguous_attempt_poison"
    )
    incident_kind: IncidentKind
    execution_bundle_sha256: str
    row_context_sha256: str
    stage: Literal["inventory", "packet"]
    candidate_index: int | None
    candidate_sha256: str | None
    attempt_id: str
    attempt_intent_sha256: str
    request_sha256: str
    model_identity_sha256: str
    response_observed: Literal[False] = False
    possible_generation_call_attempts: Literal[1] = 1
    generation_retries_permitted: Literal[0] = 0
    retry_this_request_permitted: Literal[False] = False
    incident_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "row_context_sha256",
        "candidate_sha256",
        "attempt_id",
        "attempt_intent_sha256",
        "request_sha256",
        "model_identity_sha256",
        "incident_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_runtime_incident_sha256_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_incident(self) -> MetaSynAmbiguityIncidentV1:
        packet = self.stage == "packet"
        if packet != (
            self.candidate_index is not None and self.candidate_sha256 is not None
        ):
            raise ValueError("metasyn_runtime_incident_packet_context_mismatch")
        payload = self.model_dump(mode="json", exclude={"incident_sha256"})
        if hash_canonical(payload) != self.incident_sha256:
            raise ValueError("metasyn_runtime_incident_hash_mismatch")
        return self


def freeze_metasyn_ambiguity_incident(
    *,
    intent: MetaSynAttemptIntentV1,
    incident_kind: IncidentKind,
) -> MetaSynAmbiguityIncidentV1:
    intent = MetaSynAttemptIntentV1.model_validate(intent)
    payload: dict[str, Any] = {
        "ambiguity_incident_version": AMBIGUITY_INCIDENT_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "status": "terminal_ambiguous_attempt_poison",
        "incident_kind": incident_kind,
        "execution_bundle_sha256": intent.execution_bundle_sha256,
        "row_context_sha256": intent.row_context_sha256,
        "stage": intent.stage,
        "candidate_index": intent.candidate_index,
        "candidate_sha256": intent.candidate_sha256,
        "attempt_id": intent.attempt_id,
        "attempt_intent_sha256": intent.attempt_intent_sha256,
        "request_sha256": intent.request_sha256,
        "model_identity_sha256": intent.model_identity_sha256,
        "response_observed": False,
        "possible_generation_call_attempts": 1,
        "generation_retries_permitted": 0,
        "retry_this_request_permitted": False,
    }
    return MetaSynAmbiguityIncidentV1.model_validate(
        {**payload, "incident_sha256": hash_canonical(payload)}
    )


def _inventory_response_classification(
    *,
    row: MetaSynBoundedRowContextV1,
    result: OllamaGenerationResult,
) -> tuple[InventoryResponseStatus, MetaSynInventoryValidationReceiptV1 | None, str | None]:
    if result.done_reason == "length":
        return "generation_truncated", None, "generation_truncated"
    if result.done_reason != "stop":
        return (
            "generation_terminal_reason_invalid",
            None,
            "generation_terminal_reason_invalid",
        )
    try:
        parsed = _strict_json_loads(result.response_text)
    except MetaSynBoundedRuntimeError:
        return "response_json_invalid", None, "response_json_invalid"
    try:
        v2_inventory = validate_inventory_for_row_v2(
            parsed,
            exposed_line_ids=row.source_row.projection.exposed_line_ids,
            allowed_outcomes=row.allowed_outcomes,
        )
        receipt = freeze_metasyn_inventory_validation_receipt(row=row, value=parsed)
        if receipt.inventory != v2_inventory:
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_inventory_v2_v1_acceptance_mismatch"
            )
    except (MetaSynBoundedAdapterError, NativeBoundedGenerationError, ValidationError, ValueError):
        return "inventory_contract_invalid", None, "inventory_contract_invalid"
    status_by_adapter: dict[str, InventoryResponseStatus] = {
        "candidates_authorized": "inventory_valid_candidates",
        "no_candidate_non_authorizing": (
            "inventory_valid_no_candidate_non_authorizing"
        ),
        "capacity_or_uncertainty_non_authorizing": (
            "inventory_valid_capacity_or_uncertainty_non_authorizing"
        ),
    }
    return status_by_adapter[receipt.status], receipt, None


def _packet_response_classification(
    *,
    row: MetaSynBoundedRowContextV1,
    inventory_receipt: MetaSynInventoryValidationReceiptV1,
    packet_call: MetaSynPacketCallV1,
    result: OllamaGenerationResult,
) -> tuple[PacketResponseStatus, MetaSynPacketValidationReceiptV1 | None, str | None]:
    if result.done_reason == "length":
        return "generation_truncated", None, "generation_truncated"
    if result.done_reason != "stop":
        return (
            "generation_terminal_reason_invalid",
            None,
            "generation_terminal_reason_invalid",
        )
    try:
        parsed = _strict_json_loads(result.response_text)
    except MetaSynBoundedRuntimeError:
        return "response_json_invalid", None, "response_json_invalid"
    try:
        v2_packet = validate_packet_for_candidate_v2(
            parsed,
            candidate=packet_call.candidate,
            exposed_line_ids=row.source_row.projection.exposed_line_ids,
            source_locator=row.source_locator,
            allowed_outcomes=row.allowed_outcomes,
            allowed_moderators=row.allowed_moderators,
            allowed_sections=row.allowed_sections,
            outcome_positive_directions=row.outcome_positive_directions,
        )
        receipt = freeze_metasyn_packet_validation_receipt(
            call=packet_call,
            row=row,
            inventory_receipt=inventory_receipt,
            value=parsed,
        )
        if receipt.packet_payload != v2_packet.model_dump(mode="json"):
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_packet_v2_v1_acceptance_mismatch"
            )
    except MetaSynBoundedAdapterError as exc:
        if str(exc).startswith("metasyn_packet_quote_"):
            return (
                "packet_source_grounding_invalid",
                None,
                "packet_source_grounding_invalid",
            )
        return "packet_contract_invalid", None, "packet_contract_invalid"
    except (NativeBoundedGenerationError, ValidationError, ValueError):
        return "packet_contract_invalid", None, "packet_contract_invalid"
    status: PacketResponseStatus = (
        "packet_completed"
        if receipt.packet_status == "completed"
        else "packet_unable_to_complete"
    )
    return status, receipt, None


def freeze_metasyn_generation_receipt(
    *,
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    row: MetaSynBoundedRowContextV1,
    intent: MetaSynAttemptIntentV1,
    identity: OllamaIdentity,
    generation_result: OllamaGenerationResult,
    inventory_receipt: MetaSynInventoryValidationReceiptV1 | None = None,
) -> MetaSynGenerationReceiptV1:
    bundle = MetaSynBoundedExecutionBundleV1.model_validate(execution_bundle)
    row = MetaSynBoundedRowContextV1.model_validate(row)
    intent = validate_metasyn_attempt_intent(
        intent=intent,
        execution_bundle=bundle,
        row=row,
        identity=identity,
        inventory_receipt=inventory_receipt,
    )
    result = OllamaGenerationResult.model_validate(generation_result)
    config = (
        bundle.runtime_config.inventory_generation_config
        if intent.stage == "inventory"
        else bundle.runtime_config.packet_generation_config
    )
    identity = _require_exact_identity(
        identity, runtime_config=bundle.runtime_config, generation_config=config
    )
    if result.model != config.model:
        raise MetaSynBoundedRuntimeError("metasyn_runtime_generation_result_model_mismatch")
    prompt, _schema_bundle, schema_binding, packet_call = _request_surface(
        row=row,
        stage=intent.stage,
        inventory_receipt=inventory_receipt,
        candidate_index=intent.candidate_index,
    )
    if intent.stage == "inventory":
        status, adapter_receipt, terminal_error = _inventory_response_classification(
            row=row, result=result
        )
    else:
        if inventory_receipt is None or packet_call is None:
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_packet_receipt_context_missing"
            )
        status, adapter_receipt, terminal_error = _packet_response_classification(
            row=row,
            inventory_receipt=inventory_receipt,
            packet_call=packet_call,
            result=result,
        )
    adapter_payload = (
        adapter_receipt.model_dump(mode="json")
        if adapter_receipt is not None
        else None
    )
    payload: dict[str, Any] = {
        "generation_receipt_version": GENERATION_RECEIPT_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "stage": intent.stage,
        "status": status,
        "terminal": True,
        "response_observed": True,
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "row_context_sha256": row.row_context_sha256,
        "attempt_id": intent.attempt_id,
        "attempt_intent_sha256": intent.attempt_intent_sha256,
        "request_sha256": intent.request_sha256,
        "prompt_sha256": _sha256_text(prompt),
        "schema_sha256": schema_binding["provider_schema_sha256"],
        "provider_schema_sha256": schema_binding["provider_schema_sha256"],
        "full_acceptance_schema_sha256": schema_binding[
            "full_acceptance_schema_sha256"
        ],
        "schema_bundle_sha256": schema_binding["schema_bundle_sha256"],
        "schema_context_binding_sha256": schema_binding["context_binding_sha256"],
        "generation_config_sha256": config.config_sha256,
        "model_identity_sha256": identity.identity_sha256,
        "candidate_index": intent.candidate_index,
        "candidate_sha256": intent.candidate_sha256,
        "inventory_validation_receipt_sha256": (
            inventory_receipt.receipt_sha256
            if inventory_receipt is not None
            else None
        ),
        "generation_result": result,
        "generation_result_sha256": hash_canonical(result),
        "response_text_sha256": _sha256_text(result.response_text),
        "adapter_validation_receipt": adapter_payload,
        "adapter_validation_receipt_sha256": (
            hash_canonical(adapter_payload) if adapter_payload is not None else None
        ),
        "terminal_error": terminal_error,
        "generation_call_attempts": 1,
        "generation_retries": 0,
    }
    return MetaSynGenerationReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def validate_metasyn_generation_receipt(
    *,
    receipt: MetaSynGenerationReceiptV1 | Mapping[str, Any],
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    row: MetaSynBoundedRowContextV1,
    intent: MetaSynAttemptIntentV1,
    identity: OllamaIdentity,
    inventory_receipt: MetaSynInventoryValidationReceiptV1 | None = None,
) -> MetaSynGenerationReceiptV1:
    canonical = MetaSynGenerationReceiptV1.model_validate(receipt)
    replayed = freeze_metasyn_generation_receipt(
        execution_bundle=execution_bundle,
        row=row,
        intent=intent,
        identity=identity,
        generation_result=canonical.generation_result,
        inventory_receipt=inventory_receipt,
    )
    if replayed != canonical:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_generation_receipt_external_replay_mismatch"
        )
    return canonical


def _inventory_adapter_receipt(
    receipt: MetaSynGenerationReceiptV1,
    *,
    row: MetaSynBoundedRowContextV1,
) -> MetaSynInventoryValidationReceiptV1 | None:
    if not receipt.status.startswith("inventory_valid"):
        return None
    if receipt.adapter_validation_receipt is None:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_inventory_adapter_receipt_missing"
        )
    return validate_metasyn_inventory_validation_receipt(
        receipt=MetaSynInventoryValidationReceiptV1.model_validate(
            receipt.adapter_validation_receipt
        ),
        row=row,
    )


def _packet_adapter_receipt(
    receipt: MetaSynGenerationReceiptV1,
    *,
    row: MetaSynBoundedRowContextV1,
    inventory_receipt: MetaSynInventoryValidationReceiptV1,
) -> MetaSynPacketValidationReceiptV1 | None:
    if receipt.status not in {"packet_completed", "packet_unable_to_complete"}:
        return None
    if receipt.adapter_validation_receipt is None or receipt.candidate_index is None:
        raise MetaSynBoundedRuntimeError("metasyn_runtime_packet_adapter_receipt_missing")
    call = freeze_metasyn_packet_call(
        row=row,
        inventory_receipt=inventory_receipt,
        candidate_index=receipt.candidate_index,
    )
    return validate_metasyn_packet_validation_receipt(
        receipt=MetaSynPacketValidationReceiptV1.model_validate(
            receipt.adapter_validation_receipt
        ),
        call=call,
        row=row,
        inventory_receipt=inventory_receipt,
    )


def _assert_no_symlink_ancestors(path: Path, *, code: str) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.exists() and candidate.is_symlink():
            raise MetaSynBoundedRuntimeError(code)


def canonical_metasyn_runtime_workspace(
    *, workspace: Path, repository_root: Path, require_exists: bool
) -> Path:
    root = repository_root.resolve(strict=True)
    candidate = workspace if workspace.is_absolute() else root / workspace
    _assert_no_symlink_ancestors(
        candidate, code="metasyn_runtime_workspace_symlink_ancestor_forbidden"
    )
    try:
        resolved = candidate.resolve(strict=require_exists)
        relative = resolved.relative_to(root / "data" / "cache")
    except (OSError, ValueError) as exc:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_workspace_must_be_private_data_cache"
        ) from exc
    if not relative.parts:
        raise MetaSynBoundedRuntimeError("metasyn_runtime_workspace_scope_too_broad")
    if require_exists and not resolved.is_dir():
        raise MetaSynBoundedRuntimeError("metasyn_runtime_workspace_missing")
    return resolved


def metasyn_runtime_paths(workspace: Path) -> dict[str, Path]:
    """Derive the only accepted topology from one canonical workspace."""

    return {
        "execution_bundle": workspace / "execution-bundle.private.json",
        "lock": workspace / ".metasyn-bounded-runtime.lock",
        "preflight_dir": workspace / "schema-preflight",
        "preflight_aggregate": workspace
        / "schema-preflight"
        / "preflight-receipt.json",
        "prediction_dir": workspace / "prediction",
        "attempt_intents": workspace / "prediction" / "pre-call-intents",
        "generation_receipts": workspace / "prediction" / "generation-receipts",
        "ambiguity_incidents": workspace / "prediction" / "ambiguity-incidents",
        "ledger": workspace / "prediction" / "prediction-ledger.json",
        "final_dir": workspace / "final",
        "private_report": workspace / "final" / "private-yield-report.json",
    }


@contextmanager
def _exclusive_workspace_lock(workspace: Path) -> Iterator[None]:
    paths = metasyn_runtime_paths(workspace)
    lock_path = paths["lock"]
    if lock_path.exists() and (lock_path.is_symlink() or not lock_path.is_file()):
        raise MetaSynBoundedRuntimeError("metasyn_runtime_lock_topology_invalid")
    workspace.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MetaSynBoundedRuntimeError("metasyn_runtime_workspace_locked") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _atomic_write_or_validate_exact(
    path: Path, payload: Mapping[str, Any], *, mismatch_code: str
) -> None:
    snapshot = deepcopy(dict(payload))
    if path.exists():
        if _read_json_object(path) != snapshot:
            raise MetaSynBoundedRuntimeError(mismatch_code)
        return
    try:
        atomic_write_json(path, snapshot, force=False)
    except OutputExistsError:
        if _read_json_object(path) != snapshot:
            raise MetaSynBoundedRuntimeError(mismatch_code) from None


def write_metasyn_bounded_execution_bundle(
    *,
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    workspace: Path,
    repository_root: Path,
) -> Path:
    root = repository_root.resolve(strict=True)
    canonical_workspace = canonical_metasyn_runtime_workspace(
        workspace=workspace, repository_root=root, require_exists=False
    )
    if canonical_workspace.exists() and (
        not canonical_workspace.is_dir() or any(canonical_workspace.iterdir())
    ):
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_prepare_requires_fresh_workspace"
        )
    canonical_workspace.mkdir(parents=True, exist_ok=True)
    paths = metasyn_runtime_paths(canonical_workspace)
    bundle = MetaSynBoundedExecutionBundleV1.model_validate(execution_bundle)
    atomic_write_json(
        paths["execution_bundle"], bundle.model_dump(mode="json"), force=False
    )
    return paths["execution_bundle"]


def load_current_metasyn_bounded_execution_bundle(
    *, workspace: Path, repository_root: Path, external_replay: bool = True
) -> tuple[Path, MetaSynBoundedExecutionBundleV1]:
    canonical_workspace = canonical_metasyn_runtime_workspace(
        workspace=workspace, repository_root=repository_root, require_exists=True
    )
    bundle = validate_current_metasyn_bounded_execution_bundle(
        execution_bundle=_read_json_object(
            metasyn_runtime_paths(canonical_workspace)["execution_bundle"]
        ),
        repository_root=repository_root,
        external_replay=external_replay,
    )
    return canonical_workspace, bundle


def _schema_structure(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for child_key, item in sorted(value.items()):
            if child_key == "$id":
                continue
            if child_key == "enum" and isinstance(item, list):
                # Enum literals and cardinality are intentionally erased here. The
                # resulting digest identifies only an object/array/union skeleton; it
                # does not establish compilation of any production enum or cardinality.
                result[child_key] = sorted({type(entry).__name__ for entry in item})
            elif child_key in {"const", "default", "examples"}:
                result[child_key] = type(item).__name__
            else:
                result[child_key] = _schema_structure(item, key=child_key)
        return result
    if isinstance(value, list):
        return [_schema_structure(item, key=key) for item in value]
    return value


def schema_structure_sha256(schema: Mapping[str, Any]) -> str:
    return hash_canonical(_schema_structure(schema))


def _preflight_spec_roster(
    bundle: MetaSynBoundedExecutionBundleV1,
) -> list[dict[str, Any]]:
    """Bind three inventory states and five packet compatibility calls."""

    if (
        bundle.schema_v2_preflight_fingerprint
        != _current_schema_v2_preflight_fingerprint()
    ):
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_preflight_fingerprint_stale"
        )
    base_specs = _current_synthetic_schema_v2_preflight_specs()
    specs: list[dict[str, Any]] = []
    for base in base_specs:
        spec = deepcopy(base)
        if (
            spec["schema"] != spec["provider_schema"]
            or spec["schema_sha256"] != hash_canonical(spec["provider_schema"])
            or spec["provider_schema_sha256"] != spec["schema_sha256"]
            or spec["full_acceptance_schema_sha256"]
            != hash_canonical(spec["full_acceptance_schema"])
            or spec["receipt_binding"]["provider_schema_sha256"]
            != spec["provider_schema_sha256"]
            or spec["receipt_binding"]["full_acceptance_schema_sha256"]
            != spec["full_acceptance_schema_sha256"]
            or spec["receipt_binding"]["schema_bundle_sha256"]
            != spec["schema_bundle_sha256"]
        ):
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_preflight_schema_bundle_mismatch"
            )
        config = (
            bundle.runtime_config.inventory_generation_config
            if spec["kind"] == "inventory"
            else bundle.runtime_config.packet_generation_config
        )
        exact_json = json.dumps(
            spec["valid_example"], sort_keys=True, separators=(",", ":")
        )
        spec.update(
            {
                "config": config,
                "prompt": (
                    "Synthetic provider-schema whole-request compatibility check only; "
                    "no paper, source, claim, protocol, or label. Copy the exact JSON "
                    "object after SYNTHETIC_JSON and return nothing else.\n"
                    f"SYNTHETIC_JSON:{exact_json}"
                ),
                "schema_structure_sha256": schema_structure_sha256(
                    spec["provider_schema"]
                ),
            }
        )
        specs.append(spec)
    if (
        len(specs) != PREFLIGHT_CALL_COUNT
        or sum(spec["kind"] == "inventory" for spec in specs) != 3
        or {
            spec.get("inventory_state")
            for spec in specs
            if spec["kind"] == "inventory"
        }
        != set(PREFLIGHT_INVENTORY_STATES)
        or any(
            spec.get("inventory_state")
            != spec["valid_example"].get("inventory_status")
            for spec in specs
            if spec["kind"] == "inventory"
        )
        or {spec["effect_kind"] for spec in specs if spec["kind"] == "packet"}
        != {
            "binary_group_statistics",
            "continuous_group_statistics",
            "direct_confidence_interval",
            "direct_standard_error",
            "direct_variance",
        }
    ):
        raise MetaSynBoundedRuntimeError("metasyn_runtime_preflight_roster_incomplete")
    return specs


def _validate_preflight_structural_skeleton_coverage(
    *, bundle: MetaSynBoundedExecutionBundleV1, specs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = _preflight_spec_roster(bundle)
    if [dict(item) for item in specs] != expected:
        raise MetaSynBoundedRuntimeError("metasyn_runtime_preflight_roster_mismatch")
    payload = {
        "coverage_version": (
            "metasyn-bounded-provider-schema-family-canonical-coverage-v4"
        ),
        "adapter_bundle_sha256": bundle.adapter_bundle_sha256,
        "schema_v2_preflight_fingerprint": bundle.schema_v2_preflight_fingerprint,
        "provider_schema_sha256s": [
            str(spec["provider_schema_sha256"]) for spec in specs
        ],
        "full_acceptance_schema_sha256s": [
            str(spec["full_acceptance_schema_sha256"]) for spec in specs
        ],
        "schema_bundle_sha256s": [
            str(spec["schema_bundle_sha256"]) for spec in specs
        ],
        "inventory_provider_schema_family_count": 1,
        "inventory_state_fixture_count": len(PREFLIGHT_INVENTORY_STATES),
        "inventory_state_fixtures": [
            str(spec["inventory_state"])
            for spec in specs
            if spec["kind"] == "inventory"
        ],
        "inventory_state_fixture_membership_sha256": hash_canonical(
            [
                str(spec["inventory_state"])
                for spec in specs
                if spec["kind"] == "inventory"
            ]
        ),
        "packet_effect_kind_families": sorted(
            str(spec["effect_kind"])
            for spec in specs
            if spec["kind"] == "packet"
        ),
        "synthetic_schema_family_count": 6,
        "synthetic_call_count": PREFLIGHT_CALL_COUNT,
        "whole_request_compatibility_only": True,
        "provider_keyword_enforcement_validated": False,
        "production_context_schema_compilation_validated": False,
        "production_enum_or_cardinality_compilation_validated": False,
        "canonical_semantic_fixture_equality_required": True,
        "raw_textual_fixture_identity_required": False,
        "declared_pydantic_default_omission_equivalent_after_validation": True,
        "nondefault_omission_equivalent_after_validation": False,
        "scientific_value_or_lexeme_changes_equivalent": False,
    }
    return {**payload, "coverage_sha256": hash_canonical(payload)}


class _AmbiguousCall(RuntimeError):
    def __init__(self, kind: IncidentKind):
        super().__init__(kind)
        self.kind = kind


def _generate_once_with_exact_identity(
    *,
    client: OllamaClientProtocol,
    initial_identity: OllamaIdentity,
    runtime_config: MetaSynBoundedRuntimeConfigV1,
    generation_config: OllamaGenerationConfig,
    prompt: str,
    schema: Mapping[str, Any],
) -> OllamaGenerationResult:
    try:
        before = client.inspect_identity(generation_config)
    except LocalOllamaError as exc:
        raise _AmbiguousCall("pre_request_identity_unavailable") from exc
    try:
        _require_exact_identity(
            before,
            runtime_config=runtime_config,
            generation_config=generation_config,
        )
    except MetaSynBoundedRuntimeError as exc:
        raise _AmbiguousCall("pre_request_identity_mismatch") from exc
    if before != initial_identity:
        raise _AmbiguousCall("pre_request_identity_mismatch")
    try:
        result = client.generate(
            prompt=prompt, output_schema=schema, config=generation_config
        )
    except LocalOllamaError as exc:
        raise _AmbiguousCall("generation_transport_ambiguous") from exc
    try:
        after = client.inspect_identity(generation_config)
    except LocalOllamaError as exc:
        raise _AmbiguousCall("post_response_identity_unavailable") from exc
    try:
        _require_exact_identity(
            after,
            runtime_config=runtime_config,
            generation_config=generation_config,
        )
    except MetaSynBoundedRuntimeError as exc:
        raise _AmbiguousCall("post_response_identity_mismatch") from exc
    if after != initial_identity or result.model != generation_config.model:
        raise _AmbiguousCall("post_response_identity_mismatch")
    return result


def _preflight_file_paths(preflight_dir: Path, call_id: str) -> dict[str, Path]:
    return {
        "intent": preflight_dir / "pre-call-intents" / f"{call_id}.json",
        "receipt": preflight_dir / "call-receipts" / f"{call_id}.json",
        "incident": preflight_dir / "ambiguity-incidents" / f"{call_id}.json",
    }


def _validate_runtime_workspace_topology(
    *, workspace: Path, bundle: MetaSynBoundedExecutionBundleV1
) -> None:
    paths = metasyn_runtime_paths(workspace)
    allowed_root = {
        paths["execution_bundle"].name,
        paths["lock"].name,
        paths["preflight_dir"].name,
        paths["prediction_dir"].name,
        paths["final_dir"].name,
    }
    for entry in workspace.iterdir():
        if entry.name not in allowed_root or entry.is_symlink():
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_workspace_mixing_or_extra"
            )
        should_be_directory = entry.name in {
            paths["preflight_dir"].name,
            paths["prediction_dir"].name,
            paths["final_dir"].name,
        }
        if should_be_directory != entry.is_dir():
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_workspace_entry_type_invalid"
            )
    if paths["preflight_dir"].exists():
        allowed_calls = {
            str(spec["call_id"]) for spec in _preflight_spec_roster(bundle)
        }
        allowed_preflight_root = {
            "pre-call-intents",
            "call-receipts",
            "ambiguity-incidents",
            "preflight-receipt.json",
        }
        for entry in paths["preflight_dir"].iterdir():
            if entry.name not in allowed_preflight_root or entry.is_symlink():
                raise MetaSynBoundedRuntimeError(
                    "metasyn_runtime_preflight_directory_mixing_or_extra"
                )
            if entry.name == "preflight-receipt.json":
                if not entry.is_file():
                    raise MetaSynBoundedRuntimeError(
                        "metasyn_runtime_preflight_aggregate_topology_invalid"
                    )
                continue
            if not entry.is_dir():
                raise MetaSynBoundedRuntimeError(
                    "metasyn_runtime_preflight_subdirectory_topology_invalid"
                )
            for child in entry.iterdir():
                if (
                    child.is_symlink()
                    or not child.is_file()
                    or child.suffix != ".json"
                    or child.stem not in allowed_calls
                ):
                    raise MetaSynBoundedRuntimeError(
                        "metasyn_runtime_preflight_call_mixing_or_extra"
                    )
    if paths["prediction_dir"].exists():
        allowed_prediction_root = {
            "pre-call-intents",
            "generation-receipts",
            "ambiguity-incidents",
            "prediction-ledger.json",
        }
        for entry in paths["prediction_dir"].iterdir():
            if entry.name not in allowed_prediction_root or entry.is_symlink():
                raise MetaSynBoundedRuntimeError(
                    "metasyn_runtime_prediction_directory_mixing_or_extra"
                )
            if entry.name == "prediction-ledger.json":
                if not entry.is_file():
                    raise MetaSynBoundedRuntimeError(
                        "metasyn_runtime_prediction_ledger_topology_invalid"
                    )
            elif not entry.is_dir():
                raise MetaSynBoundedRuntimeError(
                    "metasyn_runtime_prediction_subdirectory_topology_invalid"
                )
    if paths["final_dir"].exists():
        for entry in paths["final_dir"].iterdir():
            if (
                entry.name != "private-yield-report.json"
                or entry.is_symlink()
                or not entry.is_file()
            ):
                raise MetaSynBoundedRuntimeError(
                    "metasyn_runtime_final_directory_mixing_or_extra"
                )


def _freeze_preflight_intent(
    *,
    spec: Mapping[str, Any],
    bundle: MetaSynBoundedExecutionBundleV1,
    identity: OllamaIdentity,
) -> dict[str, Any]:
    config = OllamaGenerationConfig.model_validate(spec["config"])
    payload = {
        "preflight_intent_version": "metasyn-bounded-preflight-intent-v4",
        "status": "durable_source_free_pre_call_intent_frozen",
        "call_id": spec["call_id"],
        "kind": spec["kind"],
        "effect_kind": spec.get("effect_kind"),
        "inventory_state": spec.get("inventory_state"),
        "schema_sha256": spec["schema_sha256"],
        "provider_schema_sha256": spec["provider_schema_sha256"],
        "full_acceptance_schema_sha256": spec[
            "full_acceptance_schema_sha256"
        ],
        "schema_bundle_sha256": spec["schema_bundle_sha256"],
        "schema_context_binding_sha256": spec["receipt_binding"][
            "context_binding_sha256"
        ],
        "schema_structure_sha256": spec["schema_structure_sha256"],
        "native_schema_v2_contract_sha256": (
            bundle.native_schema_v2_contract_sha256
        ),
        "provider_grammar_scope_sha256": bundle.provider_grammar_scope_sha256,
        "schema_v2_preflight_fingerprint": bundle.schema_v2_preflight_fingerprint,
        "prompt_sha256": _sha256_text(str(spec["prompt"])),
        "generation_config_sha256": config.config_sha256,
        "model_identity_sha256": identity.identity_sha256,
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
        "contains_source_text": False,
        "contains_protocol_or_question": False,
        "contains_reference_or_test_labels": False,
        "permitted_call_attempts": 1,
        "generation_retries_permitted": 0,
    }
    return {**payload, "intent_sha256": hash_canonical(payload)}


def _preflight_typed_canonical_payload(
    *, spec: Mapping[str, Any], raw: Mapping[str, Any]
) -> tuple[BaseModel, dict[str, Any]]:
    """Validate one raw fixture through the full stack before canonicalization."""

    validate_raw_payload_against_schema_v2(
        raw, schema=spec["full_acceptance_schema"]
    )
    if spec["kind"] == "inventory":
        typed = validate_inventory_for_row_v2(
            raw,
            exposed_line_ids=["SYNTHETIC_LINE"],
            allowed_outcomes=["synthetic_outcome"],
        )
        if typed.inventory_status != spec.get("inventory_state"):
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_preflight_inventory_state_branch_mismatch"
            )
    else:
        candidate = NativeCandidateDescriptor(
            candidate_index=1,
            outcome_name="synthetic_outcome",
            effect_kind=spec["effect_kind"],
            line_ids=["SYNTHETIC_LINE"],
        )
        typed = validate_packet_for_candidate_v2(
            raw,
            candidate=candidate,
            exposed_line_ids=["SYNTHETIC_LINE"],
            source_locator="synthetic:schema-compilation-only",
            allowed_outcomes=["synthetic_outcome"],
            allowed_moderators=[],
            allowed_sections=["Synthetic"],
            outcome_positive_directions={
                "synthetic_outcome": "larger synthetic target value"
            },
        )
    return typed, typed.model_dump(mode="json")


def _json_value_exact(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _json_value_exact(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_value_exact(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)


def _typed_field_name_for_raw_key(model: BaseModel, raw_key: str) -> str | None:
    for name, field in type(model).model_fields.items():
        aliases = {name}
        for alias in (field.alias, field.serialization_alias):
            if isinstance(alias, str):
                aliases.add(alias)
        if raw_key in aliases:
            return name
    return None


def _declared_default_omission_audit(
    *,
    observed: Any,
    expected: Any,
    expected_typed: Any,
    expected_canonical: Any,
    path: str = "",
) -> tuple[list[str], bool]:
    """Allow only omissions proven to be actual Pydantic field defaults."""

    omitted: list[str] = []
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or not isinstance(
            expected_canonical, Mapping
        ):
            return omitted, False
        equivalent = not (set(observed) - set(expected))
        for key in sorted(expected):
            child_path = f"{path}.{key}" if path else str(key)
            field_name = (
                _typed_field_name_for_raw_key(expected_typed, str(key))
                if isinstance(expected_typed, BaseModel)
                else None
            )
            typed_child = (
                getattr(expected_typed, field_name)
                if field_name is not None
                else (
                    expected_typed.get(key)
                    if isinstance(expected_typed, Mapping)
                    else None
                )
            )
            canonical_child = expected_canonical.get(key)
            if key not in observed:
                if field_name is None or not isinstance(expected_typed, BaseModel):
                    equivalent = False
                    continue
                field = type(expected_typed).model_fields[field_name]
                if field.is_required():
                    equivalent = False
                    continue
                declared_default = field.get_default(call_default_factory=True)
                if typed_child != declared_default or not _json_value_exact(
                    canonical_child, expected[key]
                ):
                    equivalent = False
                    continue
                omitted.append(child_path)
                continue
            child_omitted, child_equivalent = _declared_default_omission_audit(
                observed=observed[key],
                expected=expected[key],
                expected_typed=typed_child,
                expected_canonical=canonical_child,
                path=child_path,
            )
            omitted.extend(child_omitted)
            equivalent = equivalent and child_equivalent
        return omitted, equivalent
    if isinstance(expected, list):
        if (
            not isinstance(observed, list)
            or not isinstance(expected_canonical, list)
            or len(observed) != len(expected)
            or len(expected_canonical) != len(expected)
        ):
            return omitted, False
        equivalent = True
        typed_items = expected_typed if isinstance(expected_typed, list) else []
        if len(typed_items) != len(expected):
            return omitted, False
        for index, expected_item in enumerate(expected):
            child_omitted, child_equivalent = _declared_default_omission_audit(
                observed=observed[index],
                expected=expected_item,
                expected_typed=typed_items[index],
                expected_canonical=expected_canonical[index],
                path=f"{path}[{index}]",
            )
            omitted.extend(child_omitted)
            equivalent = equivalent and child_equivalent
        return omitted, equivalent
    return omitted, _json_value_exact(observed, expected)


def _preflight_comparison_metadata(
    *,
    raw_fixture_equal: bool | None,
    canonical_fixture_equal: bool | None,
    omitted_declared_default_paths: Sequence[str] = (),
) -> dict[str, Any]:
    paths = sorted(str(path) for path in omitted_declared_default_paths)
    if len(paths) != len(set(paths)):
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_preflight_default_omission_paths_duplicate"
        )
    return {
        "fixture_comparison_version": PREFLIGHT_FIXTURE_COMPARISON_VERSION,
        "fixture_comparison_mode": PREFLIGHT_FIXTURE_COMPARISON_MODE,
        "raw_fixture_equal": raw_fixture_equal,
        "canonical_fixture_equal": canonical_fixture_equal,
        "omitted_declared_default_paths": paths,
        "omitted_declared_default_path_count": len(paths),
        "omitted_declared_default_paths_sha256": hash_canonical(paths),
    }


def _assess_preflight_result(
    *, spec: Mapping[str, Any], result: OllamaGenerationResult
) -> tuple[PreflightResponseStatus, dict[str, Any]]:
    def outcome(
        status: PreflightResponseStatus,
        *,
        raw_equal: bool | None = None,
        canonical_equal: bool | None = None,
        omitted: Sequence[str] = (),
    ) -> tuple[PreflightResponseStatus, dict[str, Any]]:
        return status, _preflight_comparison_metadata(
            raw_fixture_equal=raw_equal,
            canonical_fixture_equal=canonical_equal,
            omitted_declared_default_paths=omitted,
        )

    config = OllamaGenerationConfig.model_validate(spec["config"])
    if result.model != config.model:
        return outcome("generation_model_invalid")
    if result.done_reason == "length":
        return outcome("generation_truncated")
    if result.done_reason != "stop":
        return outcome("generation_terminal_reason_invalid")
    try:
        parsed = _strict_json_loads(result.response_text)
    except MetaSynBoundedRuntimeError:
        return outcome("response_json_invalid")
    expected_raw = deepcopy(dict(spec["valid_example"]))
    raw_equal = _json_value_exact(parsed, expected_raw)
    try:
        _, observed_canonical = _preflight_typed_canonical_payload(
            spec=spec, raw=parsed
        )
        expected_typed, expected_canonical = _preflight_typed_canonical_payload(
            spec=spec, raw=expected_raw
        )
    except (NativeBoundedGenerationError, ValidationError, ValueError):
        return outcome(
            "full_acceptance_or_typed_validation_invalid", raw_equal=raw_equal
        )
    canonical_equal = _json_value_exact(observed_canonical, expected_canonical)
    omitted, only_declared_default_omissions = _declared_default_omission_audit(
        observed=parsed,
        expected=expected_raw,
        expected_typed=expected_typed,
        expected_canonical=expected_canonical,
    )
    if not canonical_equal:
        return outcome(
            "canonical_semantic_fixture_mismatch",
            raw_equal=raw_equal,
            canonical_equal=False,
            omitted=omitted,
        )
    if not only_declared_default_omissions:
        return outcome(
            "fixture_difference_not_declared_default_omission",
            raw_equal=raw_equal,
            canonical_equal=True,
            omitted=omitted,
        )
    return outcome(
        "passed",
        raw_equal=raw_equal,
        canonical_equal=True,
        omitted=omitted,
    )


def _classify_preflight_result(
    *, spec: Mapping[str, Any], result: OllamaGenerationResult
) -> PreflightResponseStatus:
    return _assess_preflight_result(spec=spec, result=result)[0]


def _freeze_preflight_call_receipt(
    *,
    spec: Mapping[str, Any],
    intent: Mapping[str, Any],
    identity: OllamaIdentity,
    result: OllamaGenerationResult,
    bundle: MetaSynBoundedExecutionBundleV1,
) -> dict[str, Any]:
    status, comparison = _assess_preflight_result(spec=spec, result=result)
    payload = {
        "preflight_call_receipt_version": PREFLIGHT_CALL_RECEIPT_VERSION,
        "status": status,
        "terminal": True,
        "response_observed": True,
        "call_id": spec["call_id"],
        "kind": spec["kind"],
        "effect_kind": spec.get("effect_kind"),
        "inventory_state": spec.get("inventory_state"),
        "intent_sha256": intent["intent_sha256"],
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
        "schema_sha256": spec["schema_sha256"],
        "provider_schema_sha256": spec["provider_schema_sha256"],
        "full_acceptance_schema_sha256": spec[
            "full_acceptance_schema_sha256"
        ],
        "schema_bundle_sha256": spec["schema_bundle_sha256"],
        "schema_context_binding_sha256": spec["receipt_binding"][
            "context_binding_sha256"
        ],
        "schema_structure_sha256": spec["schema_structure_sha256"],
        "native_schema_v2_contract_sha256": (
            bundle.native_schema_v2_contract_sha256
        ),
        "provider_grammar_scope_sha256": bundle.provider_grammar_scope_sha256,
        "schema_v2_preflight_fingerprint": bundle.schema_v2_preflight_fingerprint,
        "prompt_sha256": _sha256_text(str(spec["prompt"])),
        "generation_config_sha256": spec["config"].config_sha256,
        "model_identity_sha256": identity.identity_sha256,
        "generation_result": result.model_dump(mode="json"),
        "generation_result_sha256": hash_canonical(result),
        "response_text_sha256": _sha256_text(result.response_text),
        **comparison,
        "contains_source_text": False,
        "contains_protocol_or_question": False,
        "contains_reference_or_test_labels": False,
        "terminal_error": None if status == "passed" else status,
        "generation_call_attempts": 1,
        "generation_retries": 0,
    }
    return {**payload, "call_receipt_sha256": hash_canonical(payload)}


def _validate_preflight_call_receipt(
    *,
    receipt: Mapping[str, Any],
    spec: Mapping[str, Any],
    intent: Mapping[str, Any],
    identity: OllamaIdentity,
    bundle: MetaSynBoundedExecutionBundleV1,
) -> dict[str, Any]:
    snapshot = deepcopy(dict(receipt))
    stored_hash = snapshot.pop("call_receipt_sha256", None)
    if stored_hash != hash_canonical(snapshot):
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_preflight_call_receipt_hash_mismatch"
        )
    result = OllamaGenerationResult.model_validate(snapshot.get("generation_result"))
    expected_intent = _freeze_preflight_intent(
        spec=spec, bundle=bundle, identity=identity
    )
    if dict(intent) != expected_intent:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_preflight_intent_replay_mismatch"
        )
    expected = _freeze_preflight_call_receipt(
        spec=spec,
        intent=expected_intent,
        identity=identity,
        result=result,
        bundle=bundle,
    )
    if dict(receipt) != expected:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_preflight_call_receipt_replay_mismatch"
        )
    return expected


def _require_passed_preflight_call_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = deepcopy(dict(receipt))
    if snapshot.get("status") != "passed":
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_preflight_terminal_response_failed:"
            f"{snapshot.get('status')}"
        )
    paths = snapshot.get("omitted_declared_default_paths")
    if (
        snapshot.get("fixture_comparison_version")
        != PREFLIGHT_FIXTURE_COMPARISON_VERSION
        or snapshot.get("fixture_comparison_mode")
        != PREFLIGHT_FIXTURE_COMPARISON_MODE
        or snapshot.get("canonical_fixture_equal") is not True
        or not isinstance(snapshot.get("raw_fixture_equal"), bool)
        or not isinstance(paths, list)
        or paths != sorted(paths)
        or len(paths) != len(set(paths))
        or snapshot.get("omitted_declared_default_path_count") != len(paths)
        or snapshot.get("omitted_declared_default_paths_sha256")
        != hash_canonical(paths)
        or (snapshot["raw_fixture_equal"] is True and paths)
        or (snapshot["raw_fixture_equal"] is False and not paths)
    ):
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_preflight_passed_comparison_metadata_invalid"
        )
    return snapshot


def _preflight_fixture_comparison_summary(
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(receipts) != PREFLIGHT_CALL_COUNT:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_preflight_comparison_receipt_count_invalid"
        )
    canonical = [
        _require_passed_preflight_call_receipt(receipt) for receipt in receipts
    ]
    path_membership = sorted(
        f"{receipt['call_id']}:{path}"
        for receipt in canonical
        for path in receipt["omitted_declared_default_paths"]
    )
    return {
        "fixture_comparison_version": PREFLIGHT_FIXTURE_COMPARISON_VERSION,
        "fixture_comparison_mode": PREFLIGHT_FIXTURE_COMPARISON_MODE,
        "raw_fixture_equal_call_count": sum(
            receipt["raw_fixture_equal"] is True for receipt in canonical
        ),
        "canonical_fixture_equal_call_count": sum(
            receipt["canonical_fixture_equal"] is True for receipt in canonical
        ),
        "declared_default_omission_call_count": sum(
            bool(receipt["omitted_declared_default_paths"])
            for receipt in canonical
        ),
        "omitted_declared_default_path_count": len(path_membership),
        "omitted_declared_default_path_membership_sha256": hash_canonical(
            path_membership
        ),
        "all_raw_differences_are_declared_pydantic_default_omissions": True,
        "all_canonical_fixtures_equal": True,
    }


def _freeze_preflight_incident(
    *,
    spec: Mapping[str, Any],
    intent: Mapping[str, Any],
    kind: IncidentKind,
    bundle: MetaSynBoundedExecutionBundleV1,
) -> dict[str, Any]:
    payload = {
        "preflight_incident_version": "metasyn-bounded-preflight-incident-v4",
        "status": "terminal_ambiguous_preflight_poison",
        "incident_kind": kind,
        "call_id": spec["call_id"],
        "inventory_state": spec.get("inventory_state"),
        "intent_sha256": intent["intent_sha256"],
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "provider_schema_sha256": spec["provider_schema_sha256"],
        "full_acceptance_schema_sha256": spec[
            "full_acceptance_schema_sha256"
        ],
        "schema_bundle_sha256": spec["schema_bundle_sha256"],
        "schema_structure_sha256": spec["schema_structure_sha256"],
        "response_observed": False,
        "possible_generation_call_attempts": 1,
        "retry_this_request_permitted": False,
    }
    return {**payload, "incident_sha256": hash_canonical(payload)}


def validate_metasyn_schema_preflight(
    *,
    receipt: Mapping[str, Any],
    workspace: Path,
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    identity: OllamaIdentity,
) -> dict[str, Any]:
    bundle = MetaSynBoundedExecutionBundleV1.model_validate(execution_bundle)
    identity = _require_exact_identity(
        identity,
        runtime_config=bundle.runtime_config,
        generation_config=bundle.runtime_config.inventory_generation_config,
    )
    specs = _preflight_spec_roster(bundle)
    coverage = _validate_preflight_structural_skeleton_coverage(
        bundle=bundle, specs=specs
    )
    preflight_dir = metasyn_runtime_paths(workspace)["preflight_dir"]
    call_receipts: list[dict[str, Any]] = []
    for spec in specs:
        paths = _preflight_file_paths(preflight_dir, str(spec["call_id"]))
        if paths["incident"].exists():
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_preflight_workspace_poisoned"
            )
        if not paths["intent"].is_file() or not paths["receipt"].is_file():
            raise MetaSynBoundedRuntimeError("metasyn_runtime_preflight_incomplete")
        call_receipts.append(
            _require_passed_preflight_call_receipt(
                _validate_preflight_call_receipt(
                    receipt=_read_json_object(paths["receipt"]),
                    spec=spec,
                    intent=_read_json_object(paths["intent"]),
                    identity=identity,
                    bundle=bundle,
                )
            )
        )
    comparison_summary = _preflight_fixture_comparison_summary(call_receipts)
    payload = {
        "preflight_version": PREFLIGHT_VERSION,
        "status": PREFLIGHT_PASSED_STATUS,
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
        "config_sha256": bundle.config_sha256,
        "adapter_bundle_sha256": bundle.adapter_bundle_sha256,
        "model_identity_sha256": identity.identity_sha256,
        "native_schema_v2_contract_sha256": (
            bundle.native_schema_v2_contract_sha256
        ),
        "provider_grammar_scope_sha256": bundle.provider_grammar_scope_sha256,
        "schema_v2_preflight_fingerprint": bundle.schema_v2_preflight_fingerprint,
        "structural_skeleton_coverage": coverage,
        "structural_skeleton_coverage_sha256": coverage["coverage_sha256"],
        "structural_skeleton_count": len(specs),
        "synthetic_generation_call_attempts": len(specs),
        "call_receipt_sha256s": [
            item["call_receipt_sha256"] for item in call_receipts
        ],
        **comparison_summary,
        "source_bearing_generation_call_attempts": 0,
        "contains_source_text": False,
        "contains_protocol_or_question": False,
        "contains_reference_or_test_labels": False,
        "whole_request_compatibility_only": True,
        "provider_keyword_enforcement_validated": False,
        "production_context_schema_compilation_validated": False,
        "production_enum_or_cardinality_compilation_validated": False,
        "canonical_semantic_fixture_equality_required": True,
        "raw_textual_fixture_identity_required": False,
        "declared_pydantic_default_omission_equivalent_after_validation": True,
        "nondefault_omission_equivalent_after_validation": False,
        "scientific_value_or_lexeme_changes_equivalent": False,
        "generation_retries": 0,
    }
    expected = {**payload, "preflight_sha256": hash_canonical(payload)}
    if dict(receipt) != expected:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_preflight_aggregate_replay_mismatch"
        )
    return expected


def run_metasyn_schema_preflight(
    *,
    workspace: Path,
    repository_root: Path,
    client: OllamaClientProtocol,
) -> dict[str, Any]:
    canonical_workspace, bundle = load_current_metasyn_bounded_execution_bundle(
        workspace=workspace, repository_root=repository_root
    )
    with _exclusive_workspace_lock(canonical_workspace):
        paths = metasyn_runtime_paths(canonical_workspace)
        _validate_runtime_workspace_topology(
            workspace=canonical_workspace, bundle=bundle
        )
        try:
            identity = client.inspect_identity(
                bundle.runtime_config.inventory_generation_config
            )
        except LocalOllamaError as exc:
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_unavailable_before_preflight_no_attempt"
            ) from exc
        identity = _require_exact_identity(
            identity,
            runtime_config=bundle.runtime_config,
            generation_config=bundle.runtime_config.inventory_generation_config,
        )
        if paths["preflight_aggregate"].exists():
            return validate_metasyn_schema_preflight(
                receipt=_read_json_object(paths["preflight_aggregate"]),
                workspace=canonical_workspace,
                execution_bundle=bundle,
                identity=identity,
            )
        specs = _preflight_spec_roster(bundle)
        _validate_preflight_structural_skeleton_coverage(
            bundle=bundle, specs=specs
        )
        for spec in specs:
            call_paths = _preflight_file_paths(
                paths["preflight_dir"], str(spec["call_id"])
            )
            if call_paths["incident"].exists():
                raise MetaSynBoundedRuntimeError(
                    "metasyn_runtime_preflight_workspace_poisoned"
                )
            if call_paths["intent"].exists() and not call_paths["receipt"].exists():
                incident = _freeze_preflight_incident(
                    spec=spec,
                    intent=_read_json_object(call_paths["intent"]),
                    kind="orphan_intent_observed_on_resume",
                    bundle=bundle,
                )
                _atomic_write_or_validate_exact(
                    call_paths["incident"],
                    incident,
                    mismatch_code="metasyn_runtime_preflight_incident_mismatch",
                )
                raise MetaSynBoundedRuntimeError(
                    "metasyn_runtime_preflight_orphan_intent_poisoned"
                )
            if call_paths["receipt"].exists():
                _require_passed_preflight_call_receipt(
                    _validate_preflight_call_receipt(
                        receipt=_read_json_object(call_paths["receipt"]),
                        spec=spec,
                        intent=_read_json_object(call_paths["intent"]),
                        identity=identity,
                        bundle=bundle,
                    )
                )
                continue
            intent = _freeze_preflight_intent(
                spec=spec, bundle=bundle, identity=identity
            )
            _atomic_write_or_validate_exact(
                call_paths["intent"],
                intent,
                mismatch_code="metasyn_runtime_preflight_intent_mismatch",
            )
            try:
                result = _generate_once_with_exact_identity(
                    client=client,
                    initial_identity=identity,
                    runtime_config=bundle.runtime_config,
                    generation_config=spec["config"],
                    prompt=str(spec["prompt"]),
                    schema=spec["schema"],
                )
            except _AmbiguousCall as exc:
                incident = _freeze_preflight_incident(
                    spec=spec,
                    intent=intent,
                    kind=exc.kind,
                    bundle=bundle,
                )
                _atomic_write_or_validate_exact(
                    call_paths["incident"],
                    incident,
                    mismatch_code="metasyn_runtime_preflight_incident_mismatch",
                )
                raise MetaSynBoundedRuntimeError(
                    "metasyn_runtime_preflight_ambiguous_attempt_poisoned"
                ) from exc
            receipt = _freeze_preflight_call_receipt(
                spec=spec,
                intent=intent,
                identity=identity,
                result=result,
                bundle=bundle,
            )
            _atomic_write_or_validate_exact(
                call_paths["receipt"],
                receipt,
                mismatch_code="metasyn_runtime_preflight_receipt_mismatch",
            )
            _require_passed_preflight_call_receipt(receipt)
        specs = _preflight_spec_roster(bundle)
        coverage = _validate_preflight_structural_skeleton_coverage(
            bundle=bundle, specs=specs
        )
        receipts = [
            _read_json_object(
                _preflight_file_paths(paths["preflight_dir"], str(spec["call_id"]))[
                    "receipt"
                ]
            )
            for spec in specs
        ]
        comparison_summary = _preflight_fixture_comparison_summary(receipts)
        payload = {
            "preflight_version": PREFLIGHT_VERSION,
            "status": PREFLIGHT_PASSED_STATUS,
            "execution_bundle_sha256": bundle.execution_bundle_sha256,
            "runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
            "config_sha256": bundle.config_sha256,
            "adapter_bundle_sha256": bundle.adapter_bundle_sha256,
            "model_identity_sha256": identity.identity_sha256,
            "native_schema_v2_contract_sha256": (
                bundle.native_schema_v2_contract_sha256
            ),
            "provider_grammar_scope_sha256": bundle.provider_grammar_scope_sha256,
            "schema_v2_preflight_fingerprint": (
                bundle.schema_v2_preflight_fingerprint
            ),
            "structural_skeleton_coverage": coverage,
            "structural_skeleton_coverage_sha256": coverage["coverage_sha256"],
            "structural_skeleton_count": len(specs),
            "synthetic_generation_call_attempts": len(specs),
            "call_receipt_sha256s": [
                item["call_receipt_sha256"] for item in receipts
            ],
            **comparison_summary,
            "source_bearing_generation_call_attempts": 0,
            "contains_source_text": False,
            "contains_protocol_or_question": False,
            "contains_reference_or_test_labels": False,
            "whole_request_compatibility_only": True,
            "provider_keyword_enforcement_validated": False,
            "production_context_schema_compilation_validated": False,
            "production_enum_or_cardinality_compilation_validated": False,
            "canonical_semantic_fixture_equality_required": True,
            "raw_textual_fixture_identity_required": False,
            "declared_pydantic_default_omission_equivalent_after_validation": True,
            "nondefault_omission_equivalent_after_validation": False,
            "scientific_value_or_lexeme_changes_equivalent": False,
            "generation_retries": 0,
        }
        aggregate = {**payload, "preflight_sha256": hash_canonical(payload)}
        _atomic_write_or_validate_exact(
            paths["preflight_aggregate"],
            aggregate,
            mismatch_code="metasyn_runtime_preflight_aggregate_mismatch",
        )
        return validate_metasyn_schema_preflight(
            receipt=aggregate,
            workspace=canonical_workspace,
            execution_bundle=bundle,
            identity=identity,
        )


def _attempt_artifact_path(directory: Path, attempt_id: str) -> Path:
    if not SHA256_RE.fullmatch(attempt_id):
        raise MetaSynBoundedRuntimeError("metasyn_runtime_attempt_path_id_invalid")
    return directory / f"{attempt_id}.json"


def _validate_incident_for_intent(
    *,
    incident: MetaSynAmbiguityIncidentV1 | Mapping[str, Any],
    intent: MetaSynAttemptIntentV1,
) -> MetaSynAmbiguityIncidentV1:
    canonical = MetaSynAmbiguityIncidentV1.model_validate(incident)
    aliases = {
        "execution_bundle_sha256": intent.execution_bundle_sha256,
        "row_context_sha256": intent.row_context_sha256,
        "stage": intent.stage,
        "candidate_index": intent.candidate_index,
        "candidate_sha256": intent.candidate_sha256,
        "attempt_id": intent.attempt_id,
        "attempt_intent_sha256": intent.attempt_intent_sha256,
        "request_sha256": intent.request_sha256,
        "model_identity_sha256": intent.model_identity_sha256,
    }
    if any(getattr(canonical, key) != value for key, value in aliases.items()):
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_incident_intent_alias_mismatch"
        )
    return canonical


class _PredictionState:
    def __init__(self) -> None:
        self.intents: dict[str, MetaSynAttemptIntentV1] = {}
        self.receipts: dict[str, MetaSynGenerationReceiptV1] = {}
        self.incidents: dict[str, MetaSynAmbiguityIncidentV1] = {}
        self.inventory_attempt_by_row: dict[str, str] = {}
        self.packet_attempt_by_candidate: dict[tuple[str, int], str] = {}


def _load_prediction_state(
    *,
    workspace: Path,
    bundle: MetaSynBoundedExecutionBundleV1,
    identity: OllamaIdentity,
) -> _PredictionState:
    paths = metasyn_runtime_paths(workspace)
    state = _PredictionState()
    expected_attempt_ids: set[str] = set()
    rows_by_hash = _row_by_hash(bundle)
    for row in bundle.adapter_bundle.row_contexts:
        expected_intent = freeze_metasyn_attempt_intent(
            execution_bundle=bundle,
            row=row,
            stage="inventory",
            identity=identity,
        )
        attempt_id = expected_intent.attempt_id
        expected_attempt_ids.add(attempt_id)
        state.inventory_attempt_by_row[row.row_context_sha256] = attempt_id
        intent_path = _attempt_artifact_path(paths["attempt_intents"], attempt_id)
        if intent_path.exists():
            intent = validate_metasyn_attempt_intent(
                intent=_read_json_object(intent_path),
                execution_bundle=bundle,
                row=row,
                identity=identity,
            )
            if intent != expected_intent:
                raise MetaSynBoundedRuntimeError(
                    "metasyn_runtime_inventory_intent_identity_mismatch"
                )
            state.intents[attempt_id] = intent
            receipt_path = _attempt_artifact_path(
                paths["generation_receipts"], attempt_id
            )
            incident_path = _attempt_artifact_path(
                paths["ambiguity_incidents"], attempt_id
            )
            if receipt_path.exists() and incident_path.exists():
                raise MetaSynBoundedRuntimeError(
                    "metasyn_runtime_attempt_has_response_and_incident"
                )
            if receipt_path.exists():
                state.receipts[attempt_id] = validate_metasyn_generation_receipt(
                    receipt=_read_json_object(receipt_path),
                    execution_bundle=bundle,
                    row=row,
                    intent=intent,
                    identity=identity,
                )
            elif incident_path.exists():
                state.incidents[attempt_id] = _validate_incident_for_intent(
                    incident=_read_json_object(incident_path), intent=intent
                )
    for row_hash, inventory_attempt_id in sorted(state.inventory_attempt_by_row.items()):
        inventory_generation = state.receipts.get(inventory_attempt_id)
        if inventory_generation is None:
            continue
        row = rows_by_hash[row_hash]
        adapter_inventory = _inventory_adapter_receipt(
            inventory_generation, row=row
        )
        if (
            adapter_inventory is None
            or not adapter_inventory.inventory.authorizes_packet_generation()
        ):
            continue
        for candidate in adapter_inventory.inventory.candidates:
            expected_intent = freeze_metasyn_attempt_intent(
                execution_bundle=bundle,
                row=row,
                stage="packet",
                identity=identity,
                inventory_receipt=adapter_inventory,
                candidate_index=candidate.candidate_index,
            )
            attempt_id = expected_intent.attempt_id
            expected_attempt_ids.add(attempt_id)
            state.packet_attempt_by_candidate[(row_hash, candidate.candidate_index)] = (
                attempt_id
            )
            intent_path = _attempt_artifact_path(paths["attempt_intents"], attempt_id)
            if not intent_path.exists():
                continue
            intent = validate_metasyn_attempt_intent(
                intent=_read_json_object(intent_path),
                execution_bundle=bundle,
                row=row,
                identity=identity,
                inventory_receipt=adapter_inventory,
            )
            if intent != expected_intent:
                raise MetaSynBoundedRuntimeError(
                    "metasyn_runtime_packet_intent_identity_mismatch"
                )
            state.intents[attempt_id] = intent
            receipt_path = _attempt_artifact_path(
                paths["generation_receipts"], attempt_id
            )
            incident_path = _attempt_artifact_path(
                paths["ambiguity_incidents"], attempt_id
            )
            if receipt_path.exists() and incident_path.exists():
                raise MetaSynBoundedRuntimeError(
                    "metasyn_runtime_attempt_has_response_and_incident"
                )
            if receipt_path.exists():
                state.receipts[attempt_id] = validate_metasyn_generation_receipt(
                    receipt=_read_json_object(receipt_path),
                    execution_bundle=bundle,
                    row=row,
                    intent=intent,
                    identity=identity,
                    inventory_receipt=adapter_inventory,
                )
            elif incident_path.exists():
                state.incidents[attempt_id] = _validate_incident_for_intent(
                    incident=_read_json_object(incident_path), intent=intent
                )
    for directory_key in (
        "attempt_intents",
        "generation_receipts",
        "ambiguity_incidents",
    ):
        directory = paths[directory_key]
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_prediction_directory_topology_invalid"
            )
        observed_ids: set[str] = set()
        for entry in directory.iterdir():
            if (
                entry.is_symlink()
                or not entry.is_file()
                or entry.suffix != ".json"
                or not SHA256_RE.fullmatch(entry.stem)
            ):
                raise MetaSynBoundedRuntimeError(
                    "metasyn_runtime_prediction_directory_mixing_or_extra"
                )
            observed_ids.add(entry.stem)
        if not observed_ids.issubset(expected_attempt_ids):
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_prediction_attempt_not_in_frozen_roster"
            )
        if directory_key != "attempt_intents" and not observed_ids.issubset(
            state.intents
        ):
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_outcome_without_attempt_intent"
            )
    return state


class MetaSynAttemptOutcomeRefV1(ContractModel):
    attempt_id: str
    state: Literal["not_attempted", "intent_without_outcome", "response", "incident"]
    attempt_intent_sha256: str | None
    response_receipt_sha256: str | None
    response_status: str | None
    incident_sha256: str | None
    incident_kind: str | None

    @field_validator(
        "attempt_id",
        "attempt_intent_sha256",
        "response_receipt_sha256",
        "incident_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_runtime_outcome_ref_sha_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_ref(self) -> MetaSynAttemptOutcomeRefV1:
        if self.state == "not_attempted":
            expected = (None, None, None, None, None)
        elif self.state == "intent_without_outcome":
            expected = (self.attempt_intent_sha256, None, None, None, None)
            if expected[0] is None:
                raise ValueError("metasyn_runtime_outcome_ref_intent_missing")
        elif self.state == "response":
            expected = (
                self.attempt_intent_sha256,
                self.response_receipt_sha256,
                self.response_status,
                None,
                None,
            )
            if any(item is None for item in expected[:3]):
                raise ValueError("metasyn_runtime_outcome_ref_response_incomplete")
        else:
            expected = (
                self.attempt_intent_sha256,
                None,
                None,
                self.incident_sha256,
                self.incident_kind,
            )
            if any(item is None for item in (expected[0], expected[3], expected[4])):
                raise ValueError("metasyn_runtime_outcome_ref_incident_incomplete")
        actual = (
            self.attempt_intent_sha256,
            self.response_receipt_sha256,
            self.response_status,
            self.incident_sha256,
            self.incident_kind,
        )
        if actual != expected:
            raise ValueError("metasyn_runtime_outcome_ref_state_mismatch")
        return self


class MetaSynPacketLedgerEntryV1(ContractModel):
    candidate_index: Annotated[int, Field(ge=1)]
    candidate_sha256: str
    outcome: MetaSynAttemptOutcomeRefV1

    @field_validator("candidate_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("metasyn_runtime_packet_ledger_candidate_hash_invalid")
        return value


class MetaSynRowLedgerEntryV1(ContractModel):
    row_context_sha256: str
    question_spec_sha256: str
    question_bundle_sha256: str
    source_row_sha256: str
    independence_component_membership_sha256: str
    source_strength: str
    release_grade_source_grounding_eligible: bool
    terminal: bool
    status: Literal[
        "inventory_pending",
        "inventory_intent_ambiguous_pending_incident",
        "runtime_inventory_blocked",
        "adapter_inventory_non_authorizing",
        "packets_pending",
        "packet_intent_ambiguous_pending_incident",
        "runtime_packet_blocked",
        "adapter_publication_terminal",
    ]
    inventory: MetaSynAttemptOutcomeRefV1
    packets: list[MetaSynPacketLedgerEntryV1]

    @field_validator(
        "row_context_sha256",
        "question_spec_sha256",
        "question_bundle_sha256",
        "source_row_sha256",
        "independence_component_membership_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_runtime_row_ledger_hash_invalid:{info.field_name}")
        return value

    @field_validator("packets")
    @classmethod
    def validate_packets(
        cls, value: list[MetaSynPacketLedgerEntryV1]
    ) -> list[MetaSynPacketLedgerEntryV1]:
        indices = [item.candidate_index for item in value]
        if indices != sorted(set(indices)):
            raise ValueError("metasyn_runtime_row_packet_roster_invalid")
        return value


class MetaSynPredictionLedgerV1(ContractModel):
    ledger_version: Literal["metasyn-bounded-prediction-ledger-v1"] = LEDGER_VERSION
    runtime_version: Literal["metasyn-bounded-local-ollama-runtime-v2"] = (
        RUNTIME_VERSION
    )
    status: Literal["partial_resumable_ledger", "complete_terminal_ledger"]
    execution_bundle_sha256: str
    runtime_pipeline_sha256: str
    config_sha256: str
    adapter_bundle_sha256: str
    preflight_sha256: str
    model_identity_sha256: str
    question_count: Literal[10] = EXPECTED_SELECTED_QUESTIONS
    publication_count: Literal[32] = EXPECTED_SELECTED_PAPERS
    row_membership_sha256: str
    rows: Annotated[list[MetaSynRowLedgerEntryV1], Field(min_length=32, max_length=32)]
    terminal_row_count: Annotated[int, Field(ge=0, le=32)]
    all_rows_terminal: bool
    row_status_counts: dict[str, Annotated[int, Field(ge=0)]]
    inventory_response_status_counts: dict[str, Annotated[int, Field(ge=0)]]
    packet_response_status_counts: dict[str, Annotated[int, Field(ge=0)]]
    ambiguity_incident_kind_counts: dict[str, Annotated[int, Field(ge=0)]]
    observed_source_generation_calls: Annotated[int, Field(ge=0)]
    possible_ambiguous_source_generation_calls: Annotated[int, Field(ge=0)]
    total_possible_source_generation_call_attempts: Annotated[int, Field(ge=0)]
    generation_retries: Literal[0] = 0
    reference_fields_unopened: Literal[True] = True
    direction_or_conclusion_labels_opened: Literal[False] = False
    ledger_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "runtime_pipeline_sha256",
        "config_sha256",
        "adapter_bundle_sha256",
        "preflight_sha256",
        "model_identity_sha256",
        "row_membership_sha256",
        "ledger_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_runtime_ledger_hash_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_ledger(self) -> MetaSynPredictionLedgerV1:
        row_hashes = [row.row_context_sha256 for row in self.rows]
        if row_hashes != sorted(set(row_hashes)):
            raise ValueError("metasyn_runtime_ledger_row_roster_invalid")
        terminal = sum(row.terminal for row in self.rows)
        if self.terminal_row_count != terminal or self.all_rows_terminal != (
            terminal == self.publication_count
        ):
            raise ValueError("metasyn_runtime_ledger_terminal_count_mismatch")
        if self.status != (
            "complete_terminal_ledger"
            if self.all_rows_terminal
            else "partial_resumable_ledger"
        ):
            raise ValueError("metasyn_runtime_ledger_status_mismatch")
        row_counts = dict(sorted(Counter(row.status for row in self.rows).items()))
        inventory_counts = dict(
            sorted(
                Counter(
                    row.inventory.response_status
                    for row in self.rows
                    if row.inventory.response_status is not None
                ).items()
            )
        )
        packet_counts = dict(
            sorted(
                Counter(
                    packet.outcome.response_status
                    for row in self.rows
                    for packet in row.packets
                    if packet.outcome.response_status is not None
                ).items()
            )
        )
        incident_counts = dict(
            sorted(
                Counter(
                    outcome.incident_kind
                    for row in self.rows
                    for outcome in [
                        row.inventory,
                        *(packet.outcome for packet in row.packets),
                    ]
                    if outcome.incident_kind is not None
                ).items()
            )
        )
        observed = sum(
            outcome.state == "response"
            for row in self.rows
            for outcome in [row.inventory, *(packet.outcome for packet in row.packets)]
        )
        ambiguous = sum(
            outcome.state == "incident"
            for row in self.rows
            for outcome in [row.inventory, *(packet.outcome for packet in row.packets)]
        )
        if (
            self.row_status_counts != row_counts
            or self.inventory_response_status_counts != inventory_counts
            or self.packet_response_status_counts != packet_counts
            or self.ambiguity_incident_kind_counts != incident_counts
            or self.observed_source_generation_calls != observed
            or self.possible_ambiguous_source_generation_calls != ambiguous
            or self.total_possible_source_generation_call_attempts
            != observed + ambiguous
        ):
            raise ValueError("metasyn_runtime_ledger_aggregate_mismatch")
        payload = self.model_dump(mode="json", exclude={"ledger_sha256"})
        if hash_canonical(payload) != self.ledger_sha256:
            raise ValueError("metasyn_runtime_ledger_hash_mismatch")
        return self


def _attempt_outcome_ref(
    *, attempt_id: str, state: _PredictionState
) -> MetaSynAttemptOutcomeRefV1:
    intent = state.intents.get(attempt_id)
    receipt = state.receipts.get(attempt_id)
    incident = state.incidents.get(attempt_id)
    if receipt is not None and incident is not None:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_attempt_has_response_and_incident"
        )
    if intent is None:
        payload = {
            "attempt_id": attempt_id,
            "state": "not_attempted",
            "attempt_intent_sha256": None,
            "response_receipt_sha256": None,
            "response_status": None,
            "incident_sha256": None,
            "incident_kind": None,
        }
    elif receipt is not None:
        payload = {
            "attempt_id": attempt_id,
            "state": "response",
            "attempt_intent_sha256": intent.attempt_intent_sha256,
            "response_receipt_sha256": receipt.receipt_sha256,
            "response_status": receipt.status,
            "incident_sha256": None,
            "incident_kind": None,
        }
    elif incident is not None:
        payload = {
            "attempt_id": attempt_id,
            "state": "incident",
            "attempt_intent_sha256": intent.attempt_intent_sha256,
            "response_receipt_sha256": None,
            "response_status": None,
            "incident_sha256": incident.incident_sha256,
            "incident_kind": incident.incident_kind,
        }
    else:
        payload = {
            "attempt_id": attempt_id,
            "state": "intent_without_outcome",
            "attempt_intent_sha256": intent.attempt_intent_sha256,
            "response_receipt_sha256": None,
            "response_status": None,
            "incident_sha256": None,
            "incident_kind": None,
        }
    return MetaSynAttemptOutcomeRefV1.model_validate(payload)


def _row_ledger_entry(
    *,
    row: MetaSynBoundedRowContextV1,
    state: _PredictionState,
) -> MetaSynRowLedgerEntryV1:
    inventory_attempt_id = state.inventory_attempt_by_row[row.row_context_sha256]
    inventory_outcome = _attempt_outcome_ref(
        attempt_id=inventory_attempt_id, state=state
    )
    packet_entries: list[MetaSynPacketLedgerEntryV1] = []
    inventory_generation = state.receipts.get(inventory_attempt_id)
    adapter_inventory = (
        _inventory_adapter_receipt(inventory_generation, row=row)
        if inventory_generation is not None
        else None
    )
    if adapter_inventory is not None and adapter_inventory.inventory.authorizes_packet_generation():
        for candidate in adapter_inventory.inventory.candidates:
            attempt_id = state.packet_attempt_by_candidate[
                (row.row_context_sha256, candidate.candidate_index)
            ]
            packet_entries.append(
                MetaSynPacketLedgerEntryV1(
                    candidate_index=candidate.candidate_index,
                    candidate_sha256=candidate.descriptor_sha256,
                    outcome=_attempt_outcome_ref(attempt_id=attempt_id, state=state),
                )
            )
    if inventory_outcome.state == "not_attempted":
        status = "inventory_pending"
        terminal = False
    elif inventory_outcome.state == "intent_without_outcome":
        status = "inventory_intent_ambiguous_pending_incident"
        terminal = False
    elif inventory_outcome.state == "incident":
        status = "runtime_inventory_blocked"
        terminal = True
    elif inventory_generation is None:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_inventory_response_state_inconsistent"
        )
    elif inventory_generation.status in {
        "generation_truncated",
        "generation_terminal_reason_invalid",
        "response_json_invalid",
        "inventory_contract_invalid",
    }:
        status = "runtime_inventory_blocked"
        terminal = True
    elif inventory_generation.status in {
        "inventory_valid_no_candidate_non_authorizing",
        "inventory_valid_capacity_or_uncertainty_non_authorizing",
    }:
        status = "adapter_inventory_non_authorizing"
        terminal = True
    else:
        packet_outcomes = [item.outcome for item in packet_entries]
        if any(item.state == "intent_without_outcome" for item in packet_outcomes):
            status = "packet_intent_ambiguous_pending_incident"
            terminal = False
        elif any(item.state == "incident" for item in packet_outcomes) or any(
            item.state == "response"
            and item.response_status
            not in {"packet_completed", "packet_unable_to_complete"}
            for item in packet_outcomes
        ):
            status = "runtime_packet_blocked"
            terminal = True
        elif any(
            item.state == "response"
            and item.response_status == "packet_unable_to_complete"
            for item in packet_outcomes
        ) or (
            packet_outcomes
            and all(
                item.state == "response"
                and item.response_status == "packet_completed"
                for item in packet_outcomes
            )
        ):
            status = "adapter_publication_terminal"
            terminal = True
        else:
            status = "packets_pending"
            terminal = False
    return MetaSynRowLedgerEntryV1(
        row_context_sha256=row.row_context_sha256,
        question_spec_sha256=row.question_spec_sha256,
        question_bundle_sha256=row.question_bundle_sha256,
        source_row_sha256=row.source_row_sha256,
        independence_component_membership_sha256=(
            row.independence_component_membership_sha256
        ),
        source_strength=row.source_row.source_projection_strength,
        release_grade_source_grounding_eligible=(
            row.source_row.release_grade_source_grounding_eligible
        ),
        terminal=terminal,
        status=status,
        inventory=inventory_outcome,
        packets=packet_entries,
    )


def freeze_metasyn_prediction_ledger(
    *,
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    preflight_receipt: Mapping[str, Any],
    identity: OllamaIdentity,
    state: _PredictionState,
) -> MetaSynPredictionLedgerV1:
    bundle = MetaSynBoundedExecutionBundleV1.model_validate(execution_bundle)
    identity = _require_exact_identity(
        identity,
        runtime_config=bundle.runtime_config,
        generation_config=bundle.runtime_config.inventory_generation_config,
    )
    if (
        preflight_receipt.get("status") != PREFLIGHT_PASSED_STATUS
        or preflight_receipt.get("execution_bundle_sha256")
        != bundle.execution_bundle_sha256
        or preflight_receipt.get("model_identity_sha256")
        != identity.identity_sha256
    ):
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_ledger_preflight_context_mismatch"
        )
    rows = sorted(
        (
            _row_ledger_entry(row=row, state=state)
            for row in bundle.adapter_bundle.row_contexts
        ),
        key=lambda item: item.row_context_sha256,
    )
    row_counts = dict(sorted(Counter(row.status for row in rows).items()))
    inventory_counts = dict(
        sorted(
            Counter(
                row.inventory.response_status
                for row in rows
                if row.inventory.response_status is not None
            ).items()
        )
    )
    packet_counts = dict(
        sorted(
            Counter(
                packet.outcome.response_status
                for row in rows
                for packet in row.packets
                if packet.outcome.response_status is not None
            ).items()
        )
    )
    incident_counts = dict(
        sorted(
            Counter(
                outcome.incident_kind
                for row in rows
                for outcome in [
                    row.inventory,
                    *(packet.outcome for packet in row.packets),
                ]
                if outcome.incident_kind is not None
            ).items()
        )
    )
    observed_calls = sum(
        outcome.state == "response"
        for row in rows
        for outcome in [row.inventory, *(packet.outcome for packet in row.packets)]
    )
    ambiguous_calls = sum(
        outcome.state == "incident"
        for row in rows
        for outcome in [row.inventory, *(packet.outcome for packet in row.packets)]
    )
    terminal_count = sum(row.terminal for row in rows)
    all_terminal = terminal_count == bundle.publication_count
    payload: dict[str, Any] = {
        "ledger_version": LEDGER_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "status": (
            "complete_terminal_ledger"
            if all_terminal
            else "partial_resumable_ledger"
        ),
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
        "config_sha256": bundle.config_sha256,
        "adapter_bundle_sha256": bundle.adapter_bundle_sha256,
        "preflight_sha256": preflight_receipt["preflight_sha256"],
        "model_identity_sha256": identity.identity_sha256,
        "question_count": bundle.question_count,
        "publication_count": bundle.publication_count,
        "row_membership_sha256": bundle.row_membership_sha256,
        "rows": rows,
        "terminal_row_count": terminal_count,
        "all_rows_terminal": all_terminal,
        "row_status_counts": row_counts,
        "inventory_response_status_counts": inventory_counts,
        "packet_response_status_counts": packet_counts,
        "ambiguity_incident_kind_counts": incident_counts,
        "observed_source_generation_calls": observed_calls,
        "possible_ambiguous_source_generation_calls": ambiguous_calls,
        "total_possible_source_generation_call_attempts": (
            observed_calls + ambiguous_calls
        ),
        "generation_retries": 0,
        "reference_fields_unopened": True,
        "direction_or_conclusion_labels_opened": False,
    }
    return MetaSynPredictionLedgerV1.model_validate(
        {**payload, "ledger_sha256": hash_canonical(payload)}
    )


def validate_metasyn_prediction_ledger(
    *,
    ledger: MetaSynPredictionLedgerV1 | Mapping[str, Any],
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    preflight_receipt: Mapping[str, Any],
    identity: OllamaIdentity,
    state: _PredictionState,
) -> MetaSynPredictionLedgerV1:
    canonical = MetaSynPredictionLedgerV1.model_validate(ledger)
    replayed = freeze_metasyn_prediction_ledger(
        execution_bundle=execution_bundle,
        preflight_receipt=preflight_receipt,
        identity=identity,
        state=state,
    )
    if replayed != canonical:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_prediction_ledger_external_replay_mismatch"
        )
    return canonical


def _outcome_ref_is_prefix(
    previous: MetaSynAttemptOutcomeRefV1,
    current: MetaSynAttemptOutcomeRefV1,
) -> bool:
    if previous.attempt_id != current.attempt_id:
        return False
    if previous.state == "not_attempted":
        return True
    if previous.state == "intent_without_outcome":
        return previous.attempt_intent_sha256 == current.attempt_intent_sha256
    return previous == current


def validate_metasyn_prediction_ledger_prefix(
    *,
    previous: MetaSynPredictionLedgerV1 | Mapping[str, Any],
    current: MetaSynPredictionLedgerV1,
) -> MetaSynPredictionLedgerV1:
    prior = MetaSynPredictionLedgerV1.model_validate(previous)
    static_fields = (
        "execution_bundle_sha256",
        "runtime_pipeline_sha256",
        "config_sha256",
        "adapter_bundle_sha256",
        "preflight_sha256",
        "model_identity_sha256",
        "question_count",
        "publication_count",
        "row_membership_sha256",
    )
    if any(getattr(prior, field) != getattr(current, field) for field in static_fields):
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_prediction_ledger_prefix_context_mismatch"
        )
    current_rows = {row.row_context_sha256: row for row in current.rows}
    if set(current_rows) != {row.row_context_sha256 for row in prior.rows}:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_prediction_ledger_prefix_roster_mismatch"
        )
    for prior_row in prior.rows:
        current_row = current_rows[prior_row.row_context_sha256]
        if not _outcome_ref_is_prefix(prior_row.inventory, current_row.inventory):
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_prediction_ledger_prefix_inventory_mismatch"
            )
        current_packets = {
            item.candidate_index: item for item in current_row.packets
        }
        for prior_packet in prior_row.packets:
            current_packet = current_packets.get(prior_packet.candidate_index)
            if (
                current_packet is None
                or current_packet.candidate_sha256 != prior_packet.candidate_sha256
                or not _outcome_ref_is_prefix(
                    prior_packet.outcome, current_packet.outcome
                )
            ):
                raise MetaSynBoundedRuntimeError(
                    "metasyn_runtime_prediction_ledger_prefix_packet_mismatch"
                )
    return prior


def _write_prediction_ledger(
    *, workspace: Path, ledger: MetaSynPredictionLedgerV1
) -> None:
    atomic_write_json(
        metasyn_runtime_paths(workspace)["ledger"],
        ledger.model_dump(mode="json"),
        force=True,
    )


def _poison_orphan_prediction_intents(
    *, workspace: Path, state: _PredictionState
) -> bool:
    changed = False
    incident_dir = metasyn_runtime_paths(workspace)["ambiguity_incidents"]
    for attempt_id, intent in sorted(state.intents.items()):
        if attempt_id in state.receipts or attempt_id in state.incidents:
            continue
        incident = freeze_metasyn_ambiguity_incident(
            intent=intent, incident_kind="orphan_intent_observed_on_resume"
        )
        _atomic_write_or_validate_exact(
            _attempt_artifact_path(incident_dir, attempt_id),
            incident.model_dump(mode="json"),
            mismatch_code="metasyn_runtime_orphan_incident_mismatch",
        )
        state.incidents[attempt_id] = incident
        changed = True
    return changed


def _persist_prediction_incident(
    *,
    workspace: Path,
    state: _PredictionState,
    intent: MetaSynAttemptIntentV1,
    kind: IncidentKind,
) -> MetaSynAmbiguityIncidentV1:
    incident = freeze_metasyn_ambiguity_incident(intent=intent, incident_kind=kind)
    _atomic_write_or_validate_exact(
        _attempt_artifact_path(
            metasyn_runtime_paths(workspace)["ambiguity_incidents"],
            intent.attempt_id,
        ),
        incident.model_dump(mode="json"),
        mismatch_code="metasyn_runtime_ambiguity_incident_mismatch",
    )
    state.incidents[intent.attempt_id] = incident
    return incident


def _persist_prediction_response(
    *,
    workspace: Path,
    state: _PredictionState,
    receipt: MetaSynGenerationReceiptV1,
) -> None:
    _atomic_write_or_validate_exact(
        _attempt_artifact_path(
            metasyn_runtime_paths(workspace)["generation_receipts"],
            receipt.attempt_id,
        ),
        receipt.model_dump(mode="json"),
        mismatch_code="metasyn_runtime_generation_receipt_mismatch",
    )
    state.receipts[receipt.attempt_id] = receipt


def _persist_prediction_intent(
    *,
    workspace: Path,
    state: _PredictionState,
    intent: MetaSynAttemptIntentV1,
) -> None:
    _atomic_write_or_validate_exact(
        _attempt_artifact_path(
            metasyn_runtime_paths(workspace)["attempt_intents"], intent.attempt_id
        ),
        intent.model_dump(mode="json"),
        mismatch_code="metasyn_runtime_attempt_intent_mismatch",
    )
    state.intents[intent.attempt_id] = intent


def _ensure_packet_attempt_roster(
    *,
    bundle: MetaSynBoundedExecutionBundleV1,
    row: MetaSynBoundedRowContextV1,
    inventory_receipt: MetaSynInventoryValidationReceiptV1,
    identity: OllamaIdentity,
    state: _PredictionState,
) -> None:
    for candidate in inventory_receipt.inventory.candidates:
        expected = freeze_metasyn_attempt_intent(
            execution_bundle=bundle,
            row=row,
            stage="packet",
            identity=identity,
            inventory_receipt=inventory_receipt,
            candidate_index=candidate.candidate_index,
        )
        key = (row.row_context_sha256, candidate.candidate_index)
        prior = state.packet_attempt_by_candidate.setdefault(key, expected.attempt_id)
        if prior != expected.attempt_id:
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_packet_attempt_roster_conflict"
            )


def _current_ledger(
    *,
    bundle: MetaSynBoundedExecutionBundleV1,
    preflight: Mapping[str, Any],
    identity: OllamaIdentity,
    state: _PredictionState,
) -> MetaSynPredictionLedgerV1:
    return freeze_metasyn_prediction_ledger(
        execution_bundle=bundle,
        preflight_receipt=preflight,
        identity=identity,
        state=state,
    )


def run_metasyn_bounded_prediction_stage(
    *,
    workspace: Path,
    repository_root: Path,
    client: OllamaClientProtocol,
    expected_execution_bundle_sha256: str,
    inventory_limit: int | None = None,
    packet_limit: int | None = None,
) -> MetaSynPredictionLedgerV1:
    """Run each missing source request at most once under the frozen full roster."""

    if not SHA256_RE.fullmatch(expected_execution_bundle_sha256):
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_execution_bundle_anchor_invalid"
        )
    for value, code in (
        (inventory_limit, "metasyn_runtime_inventory_limit_invalid"),
        (packet_limit, "metasyn_runtime_packet_limit_invalid"),
    ):
        if value is not None and value < 1:
            raise MetaSynBoundedRuntimeError(code)
    canonical_workspace, initial_bundle = (
        load_current_metasyn_bounded_execution_bundle(
            workspace=workspace, repository_root=repository_root
        )
    )
    if initial_bundle.execution_bundle_sha256 != expected_execution_bundle_sha256:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_execution_bundle_anchor_mismatch"
        )
    with _exclusive_workspace_lock(canonical_workspace):
        paths = metasyn_runtime_paths(canonical_workspace)
        if paths["private_report"].exists():
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_finalized_workspace_is_immutable"
            )
        bundle = validate_current_metasyn_bounded_execution_bundle(
            execution_bundle=_read_json_object(paths["execution_bundle"]),
            repository_root=repository_root,
            external_replay=True,
        )
        if bundle != initial_bundle:
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_execution_bundle_changed_before_lock"
            )
        _validate_runtime_workspace_topology(
            workspace=canonical_workspace, bundle=bundle
        )
        try:
            identity = client.inspect_identity(
                bundle.runtime_config.inventory_generation_config
            )
        except LocalOllamaError as exc:
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_unavailable_before_prediction_no_attempt"
            ) from exc
        identity = _require_exact_identity(
            identity,
            runtime_config=bundle.runtime_config,
            generation_config=bundle.runtime_config.inventory_generation_config,
        )
        preflight = validate_metasyn_schema_preflight(
            receipt=_read_json_object(paths["preflight_aggregate"]),
            workspace=canonical_workspace,
            execution_bundle=bundle,
            identity=identity,
        )
        for directory_key in (
            "attempt_intents",
            "generation_receipts",
            "ambiguity_incidents",
        ):
            directory = paths[directory_key]
            if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
                raise MetaSynBoundedRuntimeError(
                    "metasyn_runtime_prediction_directory_topology_invalid"
                )
            directory.mkdir(parents=True, exist_ok=True)
        state = _load_prediction_state(
            workspace=canonical_workspace, bundle=bundle, identity=identity
        )
        for row in bundle.adapter_bundle.row_contexts:
            inventory_attempt = state.inventory_attempt_by_row[row.row_context_sha256]
            generation = state.receipts.get(inventory_attempt)
            adapter_inventory = (
                _inventory_adapter_receipt(generation, row=row)
                if generation is not None
                else None
            )
            if (
                adapter_inventory is not None
                and adapter_inventory.inventory.authorizes_packet_generation()
            ):
                _ensure_packet_attempt_roster(
                    bundle=bundle,
                    row=row,
                    inventory_receipt=adapter_inventory,
                    identity=identity,
                    state=state,
                )
        current = _current_ledger(
            bundle=bundle,
            preflight=preflight,
            identity=identity,
            state=state,
        )
        if paths["ledger"].exists():
            validate_metasyn_prediction_ledger_prefix(
                previous=_read_json_object(paths["ledger"]), current=current
            )
        if _poison_orphan_prediction_intents(
            workspace=canonical_workspace, state=state
        ):
            current = _current_ledger(
                bundle=bundle,
                preflight=preflight,
                identity=identity,
                state=state,
            )
            _write_prediction_ledger(workspace=canonical_workspace, ledger=current)
        inventory_calls = 0
        stop_after_ambiguity = False
        for row in bundle.adapter_bundle.row_contexts:
            attempt_id = state.inventory_attempt_by_row[row.row_context_sha256]
            if attempt_id in state.receipts or attempt_id in state.incidents:
                continue
            if inventory_limit is not None and inventory_calls >= inventory_limit:
                break
            intent = freeze_metasyn_attempt_intent(
                execution_bundle=bundle,
                row=row,
                stage="inventory",
                identity=identity,
            )
            if intent.attempt_id != attempt_id:
                raise MetaSynBoundedRuntimeError(
                    "metasyn_runtime_inventory_attempt_id_changed"
                )
            _persist_prediction_intent(
                workspace=canonical_workspace, state=state, intent=intent
            )
            prompt, schema_bundle, _, _ = _request_surface(
                row=row,
                stage="inventory",
            )
            inventory_calls += 1
            try:
                result = _generate_once_with_exact_identity(
                    client=client,
                    initial_identity=identity,
                    runtime_config=bundle.runtime_config,
                    generation_config=(
                        bundle.runtime_config.inventory_generation_config
                    ),
                    prompt=prompt,
                    schema=schema_bundle["provider_schema"],
                )
            except _AmbiguousCall as exc:
                _persist_prediction_incident(
                    workspace=canonical_workspace,
                    state=state,
                    intent=intent,
                    kind=exc.kind,
                )
                stop_after_ambiguity = True
            else:
                receipt = freeze_metasyn_generation_receipt(
                    execution_bundle=bundle,
                    row=row,
                    intent=intent,
                    identity=identity,
                    generation_result=result,
                )
                _persist_prediction_response(
                    workspace=canonical_workspace, state=state, receipt=receipt
                )
                adapter_inventory = _inventory_adapter_receipt(receipt, row=row)
                if (
                    adapter_inventory is not None
                    and adapter_inventory.inventory.authorizes_packet_generation()
                ):
                    _ensure_packet_attempt_roster(
                        bundle=bundle,
                        row=row,
                        inventory_receipt=adapter_inventory,
                        identity=identity,
                        state=state,
                    )
            current = _current_ledger(
                bundle=bundle,
                preflight=preflight,
                identity=identity,
                state=state,
            )
            _write_prediction_ledger(workspace=canonical_workspace, ledger=current)
            if stop_after_ambiguity:
                break
        packet_calls = 0
        if not stop_after_ambiguity:
            for row in bundle.adapter_bundle.row_contexts:
                inventory_attempt = state.inventory_attempt_by_row[
                    row.row_context_sha256
                ]
                inventory_generation = state.receipts.get(inventory_attempt)
                if inventory_generation is None:
                    continue
                adapter_inventory = _inventory_adapter_receipt(
                    inventory_generation, row=row
                )
                if (
                    adapter_inventory is None
                    or not adapter_inventory.inventory.authorizes_packet_generation()
                ):
                    continue
                row_already_blocked = False
                for candidate in adapter_inventory.inventory.candidates:
                    attempt_id = state.packet_attempt_by_candidate[
                        (row.row_context_sha256, candidate.candidate_index)
                    ]
                    prior_receipt = state.receipts.get(attempt_id)
                    if attempt_id in state.incidents:
                        row_already_blocked = True
                        break
                    if prior_receipt is not None:
                        if prior_receipt.status != "packet_completed":
                            row_already_blocked = True
                            break
                        continue
                    if row_already_blocked:
                        break
                    if packet_limit is not None and packet_calls >= packet_limit:
                        break
                    intent = freeze_metasyn_attempt_intent(
                        execution_bundle=bundle,
                        row=row,
                        stage="packet",
                        identity=identity,
                        inventory_receipt=adapter_inventory,
                        candidate_index=candidate.candidate_index,
                    )
                    if intent.attempt_id != attempt_id:
                        raise MetaSynBoundedRuntimeError(
                            "metasyn_runtime_packet_attempt_id_changed"
                        )
                    _persist_prediction_intent(
                        workspace=canonical_workspace, state=state, intent=intent
                    )
                    prompt, schema_bundle, _, _ = _request_surface(
                        row=row,
                        stage="packet",
                        inventory_receipt=adapter_inventory,
                        candidate_index=candidate.candidate_index,
                    )
                    packet_calls += 1
                    try:
                        result = _generate_once_with_exact_identity(
                            client=client,
                            initial_identity=identity,
                            runtime_config=bundle.runtime_config,
                            generation_config=(
                                bundle.runtime_config.packet_generation_config
                            ),
                            prompt=prompt,
                            schema=schema_bundle["provider_schema"],
                        )
                    except _AmbiguousCall as exc:
                        _persist_prediction_incident(
                            workspace=canonical_workspace,
                            state=state,
                            intent=intent,
                            kind=exc.kind,
                        )
                        stop_after_ambiguity = True
                        row_already_blocked = True
                    else:
                        receipt = freeze_metasyn_generation_receipt(
                            execution_bundle=bundle,
                            row=row,
                            intent=intent,
                            identity=identity,
                            generation_result=result,
                            inventory_receipt=adapter_inventory,
                        )
                        _persist_prediction_response(
                            workspace=canonical_workspace,
                            state=state,
                            receipt=receipt,
                        )
                        if receipt.status != "packet_completed":
                            row_already_blocked = True
                    current = _current_ledger(
                        bundle=bundle,
                        preflight=preflight,
                        identity=identity,
                        state=state,
                    )
                    _write_prediction_ledger(
                        workspace=canonical_workspace, ledger=current
                    )
                    if row_already_blocked:
                        break
                if stop_after_ambiguity:
                    break
                if packet_limit is not None and packet_calls >= packet_limit:
                    break
        final_ledger = _current_ledger(
            bundle=bundle,
            preflight=preflight,
            identity=identity,
            state=state,
        )
        _write_prediction_ledger(
            workspace=canonical_workspace, ledger=final_ledger
        )
        return validate_metasyn_prediction_ledger(
            ledger=final_ledger,
            execution_bundle=bundle,
            preflight_receipt=preflight,
            identity=identity,
            state=state,
        )


class MetaSynRuntimeRowResultV1(ContractModel):
    row_result_version: Literal["metasyn-bounded-runtime-row-result-v1"] = (
        ROW_RESULT_VERSION
    )
    row_context_sha256: str
    question_spec_sha256: str
    question_bundle_sha256: str
    source_row_sha256: str
    independence_component_membership_sha256: str
    source_strength: str
    release_grade_source_grounding_eligible: bool
    status: Literal[
        "typed_publication_output",
        "adapter_inventory_no_candidate",
        "adapter_inventory_uncertain",
        "adapter_packet_unable",
        "runtime_inventory_blocked",
        "runtime_packet_blocked",
    ]
    runtime_blockers: list[str]
    inventory_attempt: MetaSynAttemptOutcomeRefV1
    packet_attempts: list[MetaSynPacketLedgerEntryV1]
    adapter_publication_result: MetaSynPublicationResultV1 | None
    adapter_publication_result_sha256: str | None
    observed_source_generation_calls: Annotated[int, Field(ge=0)]
    possible_ambiguous_source_generation_calls: Annotated[int, Field(ge=0)]
    typed_finding_count: Annotated[int, Field(ge=0)]
    synthesis_input_eligible: bool
    row_result_sha256: str

    @field_validator(
        "row_context_sha256",
        "question_spec_sha256",
        "question_bundle_sha256",
        "source_row_sha256",
        "independence_component_membership_sha256",
        "adapter_publication_result_sha256",
        "row_result_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_runtime_row_result_hash_invalid:{info.field_name}")
        return value

    @field_validator("runtime_blockers")
    @classmethod
    def validate_blockers(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("metasyn_runtime_row_blockers_not_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> MetaSynRuntimeRowResultV1:
        expected_adapter_sha = (
            self.adapter_publication_result.result_sha256
            if self.adapter_publication_result is not None
            else None
        )
        if self.adapter_publication_result_sha256 != expected_adapter_sha:
            raise ValueError("metasyn_runtime_row_adapter_result_hash_mismatch")
        typed = self.status == "typed_publication_output"
        if typed != (
            self.adapter_publication_result is not None
            and self.adapter_publication_result.status == "typed_publication_output"
        ):
            raise ValueError("metasyn_runtime_row_typed_status_mismatch")
        if typed == bool(self.runtime_blockers):
            raise ValueError("metasyn_runtime_row_blocker_presence_mismatch")
        expected_calls = sum(
            outcome.state == "response"
            for outcome in [
                self.inventory_attempt,
                *(item.outcome for item in self.packet_attempts),
            ]
        )
        expected_ambiguous = sum(
            outcome.state == "incident"
            for outcome in [
                self.inventory_attempt,
                *(item.outcome for item in self.packet_attempts),
            ]
        )
        if (
            self.observed_source_generation_calls != expected_calls
            or self.possible_ambiguous_source_generation_calls != expected_ambiguous
        ):
            raise ValueError("metasyn_runtime_row_call_count_mismatch")
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
            raise ValueError("metasyn_runtime_row_finding_count_mismatch")
        if self.synthesis_input_eligible != (
            typed and self.release_grade_source_grounding_eligible
        ):
            raise ValueError("metasyn_runtime_row_synthesis_eligibility_mismatch")
        payload = self.model_dump(mode="json", exclude={"row_result_sha256"})
        if hash_canonical(payload) != self.row_result_sha256:
            raise ValueError("metasyn_runtime_row_result_hash_mismatch")
        return self


def _runtime_blockers_for_row(row: MetaSynRowLedgerEntryV1) -> list[str]:
    blockers: set[str] = set()
    if row.inventory.state == "incident":
        blockers.add(f"inventory_ambiguous:{row.inventory.incident_kind}")
    elif row.inventory.state == "response" and row.inventory.response_status not in {
        "inventory_valid_candidates",
        "inventory_valid_no_candidate_non_authorizing",
        "inventory_valid_capacity_or_uncertainty_non_authorizing",
    }:
        blockers.add(f"inventory_response:{row.inventory.response_status}")
    for packet in row.packets:
        outcome = packet.outcome
        prefix = f"packet_{packet.candidate_index:02d}"
        if outcome.state == "incident":
            blockers.add(f"{prefix}_ambiguous:{outcome.incident_kind}")
        elif outcome.state == "response" and outcome.response_status != "packet_completed":
            blockers.add(f"{prefix}_response:{outcome.response_status}")
        elif outcome.state == "not_attempted":
            blockers.add(f"{prefix}_not_called_after_terminal_publication_blocker")
        elif outcome.state == "intent_without_outcome":
            blockers.add(f"{prefix}_orphan_intent_unresolved")
    return sorted(blockers)


def _freeze_runtime_row_result(
    *,
    row: MetaSynBoundedRowContextV1,
    ledger_row: MetaSynRowLedgerEntryV1,
    state: _PredictionState,
) -> MetaSynRuntimeRowResultV1:
    if not ledger_row.terminal:
        raise MetaSynBoundedRuntimeError("metasyn_runtime_finalize_row_not_terminal")
    inventory_generation = state.receipts.get(ledger_row.inventory.attempt_id)
    inventory_receipt = (
        _inventory_adapter_receipt(inventory_generation, row=row)
        if inventory_generation is not None
        else None
    )
    adapter_result: MetaSynPublicationResultV1 | None = None
    adapter_assembly_blocker: str | None = None
    valid_packet_receipts: list[MetaSynPacketValidationReceiptV1] = []
    if inventory_receipt is not None:
        for packet_entry in ledger_row.packets:
            generation = state.receipts.get(packet_entry.outcome.attempt_id)
            if generation is None:
                continue
            adapter_packet = _packet_adapter_receipt(
                generation,
                row=row,
                inventory_receipt=inventory_receipt,
            )
            if adapter_packet is not None:
                valid_packet_receipts.append(adapter_packet)
        try:
            adapter_result = freeze_metasyn_publication_result(
                row=row,
                inventory_receipt=inventory_receipt,
                packet_receipts=valid_packet_receipts,
            )
            adapter_result = validate_metasyn_publication_result(
                result=adapter_result, row=row
            )
        except (
            MetaSynBoundedAdapterError,
            NativeBoundedGenerationError,
            ValidationError,
            ValueError,
        ) as exc:
            if not inventory_receipt.inventory.authorizes_packet_generation():
                raise MetaSynBoundedRuntimeError(
                    "metasyn_runtime_non_authorizing_adapter_result_failed"
                ) from exc
            adapter_result = None
            adapter_assembly_blocker = (
                "publication_packet_assembly_conflict_or_invalid"
            )
    blockers = _runtime_blockers_for_row(ledger_row)
    if adapter_assembly_blocker is not None:
        status = "runtime_packet_blocked"
        blockers.append(adapter_assembly_blocker)
    elif ledger_row.status == "runtime_inventory_blocked":
        status = "runtime_inventory_blocked"
    elif ledger_row.status == "runtime_packet_blocked":
        status = "runtime_packet_blocked"
    elif inventory_generation is None or inventory_receipt is None:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_finalize_terminal_inventory_context_missing"
        )
    elif inventory_generation.status == "inventory_valid_no_candidate_non_authorizing":
        status = "adapter_inventory_no_candidate"
        blockers.extend(adapter_result.blocking_reasons if adapter_result else [])
    elif inventory_generation.status == (
        "inventory_valid_capacity_or_uncertainty_non_authorizing"
    ):
        status = "adapter_inventory_uncertain"
        blockers.extend(adapter_result.blocking_reasons if adapter_result else [])
    elif any(
        packet.outcome.response_status == "packet_unable_to_complete"
        for packet in ledger_row.packets
    ):
        status = "adapter_packet_unable"
        blockers.extend(adapter_result.blocking_reasons if adapter_result else [])
    elif adapter_result is not None and adapter_result.status == "typed_publication_output":
        status = "typed_publication_output"
        blockers = []
    else:
        status = "runtime_packet_blocked"
        blockers.append("publication_packet_set_not_fully_authorizing")
        blockers.extend(adapter_result.blocking_reasons if adapter_result else [])
    blockers = sorted(set(blockers))
    packet_attempts = ledger_row.packets
    observed = sum(
        outcome.state == "response"
        for outcome in [
            ledger_row.inventory,
            *(item.outcome for item in packet_attempts),
        ]
    )
    ambiguous = sum(
        outcome.state == "incident"
        for outcome in [
            ledger_row.inventory,
            *(item.outcome for item in packet_attempts),
        ]
    )
    finding_count = (
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
        "independence_component_membership_sha256": (
            row.independence_component_membership_sha256
        ),
        "source_strength": row.source_row.source_projection_strength,
        "release_grade_source_grounding_eligible": (
            row.source_row.release_grade_source_grounding_eligible
        ),
        "status": status,
        "runtime_blockers": blockers,
        "inventory_attempt": ledger_row.inventory,
        "packet_attempts": packet_attempts,
        "adapter_publication_result": adapter_result,
        "adapter_publication_result_sha256": (
            adapter_result.result_sha256 if adapter_result is not None else None
        ),
        "observed_source_generation_calls": observed,
        "possible_ambiguous_source_generation_calls": ambiguous,
        "typed_finding_count": finding_count,
        "synthesis_input_eligible": (
            status == "typed_publication_output"
            and row.source_row.release_grade_source_grounding_eligible
        ),
    }
    return MetaSynRuntimeRowResultV1.model_validate(
        {**payload, "row_result_sha256": hash_canonical(payload)}
    )


class MetaSynBoundedPrivateYieldReportV1(ContractModel):
    report_version: Literal[
        "metasyn-bounded-runtime-private-yield-report-v1"
    ] = PRIVATE_REPORT_VERSION
    status: Literal["complete_32_row_yield_only_runtime_report"] = (
        "complete_32_row_yield_only_runtime_report"
    )
    execution_bundle_sha256: str
    runtime_pipeline_sha256: str
    config_sha256: str
    adapter_bundle_sha256: str
    downstream_verifier_pipeline_sha256: str
    preflight_sha256: str
    prediction_ledger_sha256: str
    model_identity_sha256: str
    question_count: Literal[10] = EXPECTED_SELECTED_QUESTIONS
    component_count: Literal[10] = EXPECTED_SELECTED_COMPONENTS
    publication_count: Literal[32] = EXPECTED_SELECTED_PAPERS
    row_membership_sha256: str
    row_results: Annotated[list[MetaSynRuntimeRowResultV1], Field(min_length=32, max_length=32)]
    row_result_sha256s: Annotated[list[str], Field(min_length=32, max_length=32)]
    row_status_counts: dict[str, Annotated[int, Field(ge=0)]]
    inventory_response_status_counts: dict[str, Annotated[int, Field(ge=0)]]
    packet_response_status_counts: dict[str, Annotated[int, Field(ge=0)]]
    ambiguity_incident_kind_counts: dict[str, Annotated[int, Field(ge=0)]]
    typed_publication_output_count: Annotated[int, Field(ge=0, le=32)]
    release_grade_typed_publication_count: Annotated[int, Field(ge=0, le=32)]
    diagnostic_only_typed_publication_count: Annotated[int, Field(ge=0, le=32)]
    typed_finding_count: Annotated[int, Field(ge=0)]
    questions_with_any_release_grade_typed_publication: Annotated[
        int, Field(ge=0, le=10)
    ]
    synthesis_attempt_input_publication_count: Annotated[int, Field(ge=0, le=32)]
    observed_source_generation_calls: Annotated[int, Field(ge=0)]
    possible_ambiguous_source_generation_calls: Annotated[int, Field(ge=0)]
    total_possible_source_generation_call_attempts: Annotated[int, Field(ge=0)]
    synthetic_preflight_call_attempts: Annotated[int, Field(ge=1)]
    generation_retries: Literal[0] = 0
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    direction_agreement_reported: Literal[False] = False
    extraction_accuracy_reported: Literal[False] = False
    claim_release_authority: Literal[False] = False
    permitted_metrics: Literal[
        "contract_grounding_publication_and_synthesis_input_yield_only"
    ] = "contract_grounding_publication_and_synthesis_input_yield_only"
    synthesis_input_caveat: Literal[
        "typed_full_text_publications_only_not_proof_of_effect_compatibility_or_correctness"
    ] = "typed_full_text_publications_only_not_proof_of_effect_compatibility_or_correctness"
    report_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "runtime_pipeline_sha256",
        "config_sha256",
        "adapter_bundle_sha256",
        "downstream_verifier_pipeline_sha256",
        "preflight_sha256",
        "prediction_ledger_sha256",
        "model_identity_sha256",
        "row_membership_sha256",
        "report_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_runtime_report_hash_invalid:{info.field_name}")
        return value

    @field_validator("row_result_sha256s")
    @classmethod
    def validate_result_hashes(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(
            not SHA256_RE.fullmatch(item) for item in value
        ):
            raise ValueError("metasyn_runtime_report_result_hashes_invalid")
        return value

    @model_validator(mode="after")
    def validate_report(self) -> MetaSynBoundedPrivateYieldReportV1:
        row_hashes = [row.row_context_sha256 for row in self.row_results]
        if row_hashes != sorted(set(row_hashes)):
            raise ValueError("metasyn_runtime_report_row_roster_invalid")
        if self.row_result_sha256s != sorted(
            row.row_result_sha256 for row in self.row_results
        ):
            raise ValueError("metasyn_runtime_report_result_hash_alias_mismatch")
        typed = [
            row for row in self.row_results if row.status == "typed_publication_output"
        ]
        release_grade = [
            row for row in typed if row.release_grade_source_grounding_eligible
        ]
        expected_row_counts = dict(
            sorted(Counter(row.status for row in self.row_results).items())
        )
        expected_inventory_counts = dict(
            sorted(
                Counter(
                    row.inventory_attempt.response_status
                    for row in self.row_results
                    if row.inventory_attempt.response_status is not None
                ).items()
            )
        )
        expected_packet_counts = dict(
            sorted(
                Counter(
                    packet.outcome.response_status
                    for row in self.row_results
                    for packet in row.packet_attempts
                    if packet.outcome.response_status is not None
                ).items()
            )
        )
        expected_incident_counts = dict(
            sorted(
                Counter(
                    outcome.incident_kind
                    for row in self.row_results
                    for outcome in [
                        row.inventory_attempt,
                        *(packet.outcome for packet in row.packet_attempts),
                    ]
                    if outcome.incident_kind is not None
                ).items()
            )
        )
        if (
            self.row_status_counts != expected_row_counts
            or self.inventory_response_status_counts != expected_inventory_counts
            or self.packet_response_status_counts != expected_packet_counts
            or self.ambiguity_incident_kind_counts != expected_incident_counts
        ):
            raise ValueError("metasyn_runtime_report_status_aggregate_mismatch")
        if (
            self.typed_publication_output_count != len(typed)
            or self.release_grade_typed_publication_count != len(release_grade)
            or self.diagnostic_only_typed_publication_count
            != len(typed) - len(release_grade)
            or self.typed_finding_count
            != sum(row.typed_finding_count for row in typed)
            or self.synthesis_attempt_input_publication_count != len(release_grade)
            or self.questions_with_any_release_grade_typed_publication
            != len({row.question_spec_sha256 for row in release_grade})
        ):
            raise ValueError("metasyn_runtime_report_typed_aggregate_mismatch")
        observed = sum(row.observed_source_generation_calls for row in self.row_results)
        ambiguous = sum(
            row.possible_ambiguous_source_generation_calls for row in self.row_results
        )
        if (
            self.observed_source_generation_calls != observed
            or self.possible_ambiguous_source_generation_calls != ambiguous
            or self.total_possible_source_generation_call_attempts
            != observed + ambiguous
        ):
            raise ValueError("metasyn_runtime_report_call_aggregate_mismatch")
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        if hash_canonical(payload) != self.report_sha256:
            raise ValueError("metasyn_runtime_report_hash_mismatch")
        return self


def freeze_metasyn_bounded_private_yield_report(
    *,
    execution_bundle: MetaSynBoundedExecutionBundleV1,
    preflight_receipt: Mapping[str, Any],
    ledger: MetaSynPredictionLedgerV1,
    identity: OllamaIdentity,
    state: _PredictionState,
) -> MetaSynBoundedPrivateYieldReportV1:
    bundle = MetaSynBoundedExecutionBundleV1.model_validate(execution_bundle)
    if not ledger.all_rows_terminal:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_finalize_requires_complete_32_row_ledger"
        )
    rows_by_hash = _row_by_hash(bundle)
    row_results = [
        _freeze_runtime_row_result(
            row=rows_by_hash[ledger_row.row_context_sha256],
            ledger_row=ledger_row,
            state=state,
        )
        for ledger_row in ledger.rows
    ]
    row_results.sort(key=lambda item: item.row_context_sha256)
    row_counts = dict(sorted(Counter(row.status for row in row_results).items()))
    typed = [row for row in row_results if row.status == "typed_publication_output"]
    release_grade = [
        row for row in typed if row.release_grade_source_grounding_eligible
    ]
    question_hashes_with_release = {
        row.question_spec_sha256 for row in release_grade
    }
    payload: dict[str, Any] = {
        "report_version": PRIVATE_REPORT_VERSION,
        "status": "complete_32_row_yield_only_runtime_report",
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
        "config_sha256": bundle.config_sha256,
        "adapter_bundle_sha256": bundle.adapter_bundle_sha256,
        "downstream_verifier_pipeline_sha256": (
            bundle.downstream_verifier_pipeline_sha256
        ),
        "preflight_sha256": preflight_receipt["preflight_sha256"],
        "prediction_ledger_sha256": ledger.ledger_sha256,
        "model_identity_sha256": identity.identity_sha256,
        "question_count": bundle.question_count,
        "component_count": bundle.component_count,
        "publication_count": bundle.publication_count,
        "row_membership_sha256": bundle.row_membership_sha256,
        "row_results": row_results,
        "row_result_sha256s": sorted(row.row_result_sha256 for row in row_results),
        "row_status_counts": row_counts,
        "inventory_response_status_counts": ledger.inventory_response_status_counts,
        "packet_response_status_counts": ledger.packet_response_status_counts,
        "ambiguity_incident_kind_counts": ledger.ambiguity_incident_kind_counts,
        "typed_publication_output_count": len(typed),
        "release_grade_typed_publication_count": len(release_grade),
        "diagnostic_only_typed_publication_count": len(typed) - len(release_grade),
        "typed_finding_count": sum(row.typed_finding_count for row in typed),
        "questions_with_any_release_grade_typed_publication": len(
            question_hashes_with_release
        ),
        "synthesis_attempt_input_publication_count": len(release_grade),
        "observed_source_generation_calls": (
            ledger.observed_source_generation_calls
        ),
        "possible_ambiguous_source_generation_calls": (
            ledger.possible_ambiguous_source_generation_calls
        ),
        "total_possible_source_generation_call_attempts": (
            ledger.total_possible_source_generation_call_attempts
        ),
        "synthetic_preflight_call_attempts": preflight_receipt[
            "synthetic_generation_call_attempts"
        ],
        "generation_retries": 0,
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "direction_agreement_reported": False,
        "extraction_accuracy_reported": False,
        "claim_release_authority": False,
        "permitted_metrics": (
            "contract_grounding_publication_and_synthesis_input_yield_only"
        ),
        "synthesis_input_caveat": (
            "typed_full_text_publications_only_not_proof_of_effect_compatibility_or_"
            "correctness"
        ),
    }
    return MetaSynBoundedPrivateYieldReportV1.model_validate(
        {**payload, "report_sha256": hash_canonical(payload)}
    )


_PUBLIC_CAVEATS = (
    "This is a retrospective, calibration-split, oracle-corpus extraction-yield "
    "diagnostic; it is not a pristine retrieval or end-to-end evaluation.",
    "Review conclusions, directions, aggregate effects, and official test labels remain "
    "unopened, so no direction agreement or conclusion accuracy is reported.",
    "Exact quote and numeric-support validation establishes lexical grounding, not "
    "semantic entailment, treatment/comparator correctness, or extraction accuracy.",
    "Title/abstract-only rows are diagnostic and never count as release-grade synthesis inputs.",
    "Every invalid, truncated, unable, ambiguous, missing, or conflicting request blocks "
    "the publication; no successful subset is treated as a complete extraction.",
    "Typed full-text outputs are only candidate synthesis inputs and do not establish "
    "effect compatibility, scientific correctness, calibration, or claim-release authority.",
)


class MetaSynBoundedPublicYieldSummaryV1(ContractModel):
    summary_version: Literal[
        "metasyn-bounded-runtime-public-yield-summary-v1"
    ] = PUBLIC_SUMMARY_VERSION
    status: Literal["aggregate_only_yield_diagnostic"] = (
        "aggregate_only_yield_diagnostic"
    )
    execution_bundle_sha256: str
    runtime_pipeline_sha256: str
    config_sha256: str
    adapter_bundle_sha256: str
    downstream_verifier_pipeline_sha256: str
    preflight_sha256: str
    prediction_ledger_sha256: str
    private_report_sha256: str
    model_identity_sha256: str
    question_count: Literal[10] = EXPECTED_SELECTED_QUESTIONS
    component_count: Literal[10] = EXPECTED_SELECTED_COMPONENTS
    publication_count: Literal[32] = EXPECTED_SELECTED_PAPERS
    row_status_counts: dict[str, Annotated[int, Field(ge=0)]]
    inventory_response_status_counts: dict[str, Annotated[int, Field(ge=0)]]
    packet_response_status_counts: dict[str, Annotated[int, Field(ge=0)]]
    ambiguity_incident_kind_counts: dict[str, Annotated[int, Field(ge=0)]]
    typed_publication_output_count: Annotated[int, Field(ge=0, le=32)]
    release_grade_typed_publication_count: Annotated[int, Field(ge=0, le=32)]
    diagnostic_only_typed_publication_count: Annotated[int, Field(ge=0, le=32)]
    typed_finding_count: Annotated[int, Field(ge=0)]
    questions_with_any_release_grade_typed_publication: Annotated[
        int, Field(ge=0, le=10)
    ]
    synthesis_attempt_input_publication_count: Annotated[int, Field(ge=0, le=32)]
    observed_source_generation_calls: Annotated[int, Field(ge=0)]
    possible_ambiguous_source_generation_calls: Annotated[int, Field(ge=0)]
    synthetic_preflight_call_attempts: Annotated[int, Field(ge=1)]
    generation_retries: Literal[0] = 0
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    direction_agreement_reported: Literal[False] = False
    extraction_accuracy_reported: Literal[False] = False
    claim_release_authority: Literal[False] = False
    permitted_metrics: Literal[
        "contract_grounding_publication_and_synthesis_input_yield_only"
    ] = "contract_grounding_publication_and_synthesis_input_yield_only"
    caveats: Annotated[list[str], Field(min_length=6, max_length=6)]
    summary_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "runtime_pipeline_sha256",
        "config_sha256",
        "adapter_bundle_sha256",
        "downstream_verifier_pipeline_sha256",
        "preflight_sha256",
        "prediction_ledger_sha256",
        "private_report_sha256",
        "model_identity_sha256",
        "summary_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_runtime_public_hash_invalid:{info.field_name}")
        return value

    @field_validator("row_status_counts")
    @classmethod
    def validate_row_status_keys(cls, value: dict[str, int]) -> dict[str, int]:
        allowed = {
            "typed_publication_output",
            "adapter_inventory_no_candidate",
            "adapter_inventory_uncertain",
            "adapter_packet_unable",
            "runtime_inventory_blocked",
            "runtime_packet_blocked",
        }
        if value != dict(sorted(value.items())) or not set(value).issubset(allowed):
            raise ValueError("metasyn_runtime_public_row_status_counts_invalid")
        return value

    @field_validator("inventory_response_status_counts")
    @classmethod
    def validate_inventory_status_keys(
        cls, value: dict[str, int]
    ) -> dict[str, int]:
        allowed = {
            "inventory_valid_candidates",
            "inventory_valid_no_candidate_non_authorizing",
            "inventory_valid_capacity_or_uncertainty_non_authorizing",
            "generation_truncated",
            "generation_terminal_reason_invalid",
            "response_json_invalid",
            "inventory_contract_invalid",
        }
        if value != dict(sorted(value.items())) or not set(value).issubset(allowed):
            raise ValueError("metasyn_runtime_public_inventory_status_counts_invalid")
        return value

    @field_validator("packet_response_status_counts")
    @classmethod
    def validate_packet_status_keys(cls, value: dict[str, int]) -> dict[str, int]:
        allowed = {
            "packet_completed",
            "packet_unable_to_complete",
            "generation_truncated",
            "generation_terminal_reason_invalid",
            "response_json_invalid",
            "packet_contract_invalid",
            "packet_source_grounding_invalid",
        }
        if value != dict(sorted(value.items())) or not set(value).issubset(allowed):
            raise ValueError("metasyn_runtime_public_packet_status_counts_invalid")
        return value

    @field_validator("ambiguity_incident_kind_counts")
    @classmethod
    def validate_incident_status_keys(cls, value: dict[str, int]) -> dict[str, int]:
        allowed = {
            "pre_request_identity_unavailable",
            "pre_request_identity_mismatch",
            "generation_transport_ambiguous",
            "post_response_identity_unavailable",
            "post_response_identity_mismatch",
            "orphan_intent_observed_on_resume",
        }
        if value != dict(sorted(value.items())) or not set(value).issubset(allowed):
            raise ValueError("metasyn_runtime_public_incident_counts_invalid")
        return value

    @model_validator(mode="after")
    def validate_summary(self) -> MetaSynBoundedPublicYieldSummaryV1:
        if self.caveats != list(_PUBLIC_CAVEATS):
            raise ValueError("metasyn_runtime_public_caveats_mismatch")
        if sum(self.row_status_counts.values()) != self.publication_count:
            raise ValueError("metasyn_runtime_public_row_count_mismatch")
        if self.typed_publication_output_count != (
            self.release_grade_typed_publication_count
            + self.diagnostic_only_typed_publication_count
        ):
            raise ValueError("metasyn_runtime_public_typed_count_mismatch")
        payload = self.model_dump(mode="json", exclude={"summary_sha256"})
        if hash_canonical(payload) != self.summary_sha256:
            raise ValueError("metasyn_runtime_public_summary_hash_mismatch")
        return self


def freeze_metasyn_bounded_public_yield_summary(
    *, report: MetaSynBoundedPrivateYieldReportV1
) -> MetaSynBoundedPublicYieldSummaryV1:
    report = MetaSynBoundedPrivateYieldReportV1.model_validate(report)
    payload: dict[str, Any] = {
        "summary_version": PUBLIC_SUMMARY_VERSION,
        "status": "aggregate_only_yield_diagnostic",
        "execution_bundle_sha256": report.execution_bundle_sha256,
        "runtime_pipeline_sha256": report.runtime_pipeline_sha256,
        "config_sha256": report.config_sha256,
        "adapter_bundle_sha256": report.adapter_bundle_sha256,
        "downstream_verifier_pipeline_sha256": (
            report.downstream_verifier_pipeline_sha256
        ),
        "preflight_sha256": report.preflight_sha256,
        "prediction_ledger_sha256": report.prediction_ledger_sha256,
        "private_report_sha256": report.report_sha256,
        "model_identity_sha256": report.model_identity_sha256,
        "question_count": report.question_count,
        "component_count": report.component_count,
        "publication_count": report.publication_count,
        "row_status_counts": report.row_status_counts,
        "inventory_response_status_counts": report.inventory_response_status_counts,
        "packet_response_status_counts": report.packet_response_status_counts,
        "ambiguity_incident_kind_counts": report.ambiguity_incident_kind_counts,
        "typed_publication_output_count": report.typed_publication_output_count,
        "release_grade_typed_publication_count": (
            report.release_grade_typed_publication_count
        ),
        "diagnostic_only_typed_publication_count": (
            report.diagnostic_only_typed_publication_count
        ),
        "typed_finding_count": report.typed_finding_count,
        "questions_with_any_release_grade_typed_publication": (
            report.questions_with_any_release_grade_typed_publication
        ),
        "synthesis_attempt_input_publication_count": (
            report.synthesis_attempt_input_publication_count
        ),
        "observed_source_generation_calls": report.observed_source_generation_calls,
        "possible_ambiguous_source_generation_calls": (
            report.possible_ambiguous_source_generation_calls
        ),
        "synthetic_preflight_call_attempts": (
            report.synthetic_preflight_call_attempts
        ),
        "generation_retries": 0,
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "direction_agreement_reported": False,
        "extraction_accuracy_reported": False,
        "claim_release_authority": False,
        "permitted_metrics": (
            "contract_grounding_publication_and_synthesis_input_yield_only"
        ),
        "caveats": list(_PUBLIC_CAVEATS),
    }
    return MetaSynBoundedPublicYieldSummaryV1.model_validate(
        {**payload, "summary_sha256": hash_canonical(payload)}
    )


def validate_metasyn_bounded_public_yield_summary(
    *,
    summary: MetaSynBoundedPublicYieldSummaryV1 | Mapping[str, Any],
    report: MetaSynBoundedPrivateYieldReportV1,
) -> MetaSynBoundedPublicYieldSummaryV1:
    canonical = MetaSynBoundedPublicYieldSummaryV1.model_validate(summary)
    replayed = freeze_metasyn_bounded_public_yield_summary(report=report)
    if replayed != canonical:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_public_summary_external_replay_mismatch"
        )
    return canonical


def _load_replayed_prediction_context(
    *,
    workspace: Path,
    bundle: MetaSynBoundedExecutionBundleV1,
) -> tuple[
    OllamaIdentity,
    dict[str, Any],
    _PredictionState,
    MetaSynPredictionLedgerV1,
]:
    paths = metasyn_runtime_paths(workspace)
    identity = _expected_identity(bundle.runtime_config)
    preflight = validate_metasyn_schema_preflight(
        receipt=_read_json_object(paths["preflight_aggregate"]),
        workspace=workspace,
        execution_bundle=bundle,
        identity=identity,
    )
    state = _load_prediction_state(
        workspace=workspace, bundle=bundle, identity=identity
    )
    ledger = validate_metasyn_prediction_ledger(
        ledger=_read_json_object(paths["ledger"]),
        execution_bundle=bundle,
        preflight_receipt=preflight,
        identity=identity,
        state=state,
    )
    return identity, preflight, state, ledger


def finalize_metasyn_bounded_yield_runtime(
    *,
    workspace: Path,
    repository_root: Path,
    expected_execution_bundle_sha256: str,
) -> tuple[
    MetaSynBoundedPrivateYieldReportV1,
    MetaSynBoundedPublicYieldSummaryV1,
]:
    """Freeze the private report and return, but do not persist, its public summary."""

    if not SHA256_RE.fullmatch(expected_execution_bundle_sha256):
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_execution_bundle_anchor_invalid"
        )
    canonical_workspace, initial_bundle = (
        load_current_metasyn_bounded_execution_bundle(
            workspace=workspace, repository_root=repository_root
        )
    )
    if initial_bundle.execution_bundle_sha256 != expected_execution_bundle_sha256:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_execution_bundle_anchor_mismatch"
        )
    with _exclusive_workspace_lock(canonical_workspace):
        paths = metasyn_runtime_paths(canonical_workspace)
        bundle = validate_current_metasyn_bounded_execution_bundle(
            execution_bundle=_read_json_object(paths["execution_bundle"]),
            repository_root=repository_root,
            external_replay=True,
        )
        if bundle != initial_bundle:
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_execution_bundle_changed_before_lock"
            )
        _validate_runtime_workspace_topology(
            workspace=canonical_workspace, bundle=bundle
        )
        identity, preflight, state, ledger = _load_replayed_prediction_context(
            workspace=canonical_workspace, bundle=bundle
        )
        if not ledger.all_rows_terminal:
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_finalize_requires_complete_32_row_ledger"
            )
        report = freeze_metasyn_bounded_private_yield_report(
            execution_bundle=bundle,
            preflight_receipt=preflight,
            ledger=ledger,
            identity=identity,
            state=state,
        )
        _atomic_write_or_validate_exact(
            paths["private_report"],
            report.model_dump(mode="json"),
            mismatch_code="metasyn_runtime_existing_private_report_mismatch",
        )
        summary = freeze_metasyn_bounded_public_yield_summary(report=report)
        return report, summary


def validate_metasyn_bounded_finalized_runtime(
    *,
    workspace: Path,
    repository_root: Path,
    expected_execution_bundle_sha256: str,
) -> tuple[
    MetaSynBoundedPrivateYieldReportV1,
    MetaSynBoundedPublicYieldSummaryV1,
]:
    """Externally replay every private artifact and derive the unmaterialized summary."""

    canonical_workspace, bundle = load_current_metasyn_bounded_execution_bundle(
        workspace=workspace, repository_root=repository_root
    )
    if bundle.execution_bundle_sha256 != expected_execution_bundle_sha256:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_execution_bundle_anchor_mismatch"
        )
    with _exclusive_workspace_lock(canonical_workspace):
        current_bundle = validate_current_metasyn_bounded_execution_bundle(
            execution_bundle=_read_json_object(
                metasyn_runtime_paths(canonical_workspace)["execution_bundle"]
            ),
            repository_root=repository_root,
            external_replay=True,
        )
        if current_bundle != bundle:
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_execution_bundle_changed_before_lock"
            )
        bundle = current_bundle
        _validate_runtime_workspace_topology(
            workspace=canonical_workspace, bundle=bundle
        )
        identity, preflight, state, ledger = _load_replayed_prediction_context(
            workspace=canonical_workspace, bundle=bundle
        )
        expected_report = freeze_metasyn_bounded_private_yield_report(
            execution_bundle=bundle,
            preflight_receipt=preflight,
            ledger=ledger,
            identity=identity,
            state=state,
        )
        stored_report = MetaSynBoundedPrivateYieldReportV1.model_validate(
            _read_json_object(
                metasyn_runtime_paths(canonical_workspace)["private_report"]
            )
        )
        if stored_report != expected_report:
            raise MetaSynBoundedRuntimeError(
                "metasyn_runtime_private_report_external_replay_mismatch"
            )
        summary = freeze_metasyn_bounded_public_yield_summary(report=stored_report)
        return stored_report, summary


def validate_current_metasyn_bounded_public_yield_summary(
    *,
    summary: MetaSynBoundedPublicYieldSummaryV1 | Mapping[str, Any],
    workspace: Path,
    repository_root: Path,
    expected_execution_bundle_sha256: str,
) -> MetaSynBoundedPublicYieldSummaryV1:
    """Require current source/config replay plus the complete private receipt closure."""

    report, expected = validate_metasyn_bounded_finalized_runtime(
        workspace=workspace,
        repository_root=repository_root,
        expected_execution_bundle_sha256=expected_execution_bundle_sha256,
    )
    canonical = validate_metasyn_bounded_public_yield_summary(
        summary=summary, report=report
    )
    if canonical != expected:
        raise MetaSynBoundedRuntimeError(
            "metasyn_runtime_current_public_summary_mismatch"
        )
    return canonical


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_EXECUTION_WORKSPACE",
    "DEFAULT_PILOT_WORKSPACE",
    "ExpectedOllamaIdentityV1",
    "MetaSynAmbiguityIncidentV1",
    "MetaSynAttemptIntentV1",
    "MetaSynAttemptOutcomeRefV1",
    "MetaSynBoundedExecutionBundleV1",
    "MetaSynBoundedPrivateYieldReportV1",
    "MetaSynBoundedPublicYieldSummaryV1",
    "MetaSynBoundedRuntimeConfigV1",
    "MetaSynBoundedRuntimeError",
    "MetaSynGenerationReceiptV1",
    "MetaSynPacketLedgerEntryV1",
    "MetaSynPredictionLedgerV1",
    "MetaSynRowLedgerEntryV1",
    "MetaSynRuntimeRowResultV1",
    "canonical_metasyn_runtime_workspace",
    "compute_metasyn_bounded_runtime_fingerprint",
    "finalize_metasyn_bounded_yield_runtime",
    "freeze_metasyn_ambiguity_incident",
    "freeze_metasyn_attempt_intent",
    "freeze_metasyn_bounded_execution_bundle",
    "freeze_metasyn_bounded_private_yield_report",
    "freeze_metasyn_bounded_public_yield_summary",
    "freeze_metasyn_generation_receipt",
    "freeze_metasyn_prediction_ledger",
    "load_current_metasyn_bounded_execution_bundle",
    "load_metasyn_bounded_runtime_config",
    "metasyn_runtime_paths",
    "prepare_metasyn_bounded_execution_bundle",
    "run_metasyn_bounded_prediction_stage",
    "run_metasyn_schema_preflight",
    "schema_structure_sha256",
    "validate_current_metasyn_bounded_execution_bundle",
    "validate_current_metasyn_bounded_public_yield_summary",
    "validate_metasyn_attempt_intent",
    "validate_metasyn_bounded_finalized_runtime",
    "validate_metasyn_bounded_public_yield_summary",
    "validate_metasyn_generation_receipt",
    "validate_metasyn_prediction_ledger",
    "validate_metasyn_prediction_ledger_prefix",
    "validate_metasyn_schema_preflight",
    "write_metasyn_bounded_execution_bundle",
]
