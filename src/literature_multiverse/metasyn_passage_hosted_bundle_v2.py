"""Offline, label-blind execution bundle for passage-anchored hosted extraction.

This module freezes everything that a later live runtime is allowed to spend against,
but it never constructs a provider client, writes a workspace, or makes a provider
call.  Its only scientific input is the externally replayed v2 extraction-input
surface derived from the immutable v5 source projection.

The bundle deliberately separates three identities:

* exact inventory requests for all 32 frozen rows;
* source-free compatibility probes for three inventory states and five packet
  effect families; and
* conservative per-row packet cost probes over a synthetic maximum-size candidate.

Synthetic fixtures and cost probes have no extraction, accuracy, synthesis, or
claim-release authority.
"""

from __future__ import annotations

import ast
import json
import os
import stat
from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import ROUND_CEILING, Decimal
from functools import lru_cache
from importlib.metadata import version as distribution_version
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from jsonschema.validators import validator_for
from pydantic import ConfigDict, Field, field_validator, model_validator

from literature_multiverse.anthropic_bounded_generation import (
    AnthropicBoundedConfigV1,
    AnthropicBoundedRequestV1,
    AnthropicCompiledSchemaV1,
    AnthropicProviderIdentityV1,
    AnthropicRequestCostCeilingV1,
    compile_anthropic_bounded_schema,
    compute_anthropic_request_cost_ceiling,
    freeze_anthropic_bounded_request,
    freeze_anthropic_provider_identity,
)
from literature_multiverse.effects import EffectFormat
from literature_multiverse.lineage import canonical_json_bytes, hash_canonical, sha256_file
from literature_multiverse.metasyn_candidate_inventory_v2 import (
    MetaSynCandidateInventoryV2,
    MetaSynPassageCandidateV2,
    freeze_metasyn_candidate_inventory_receipt_v2,
    metasyn_candidate_inventory_schema_bundle_v2,
)
from literature_multiverse.metasyn_extraction_inputs_v2 import (
    INVENTORY_PROMPT_PATH,
    PACKET_PROMPT_PATH,
    MetaSynExtractionInputsV2,
    MetaSynExtractionRowInputV2,
    freeze_metasyn_extraction_inputs_v2,
    freeze_metasyn_packet_candidate_input_v2,
    validate_metasyn_extraction_inputs_v2,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_bounded_generation import EffectKind
from literature_multiverse.native_packet_assembly_v2 import (
    PacketAssemblyAnalysisPolicyV2,
    PacketAssemblyProtocolOrientationV2,
    freeze_packet_assembly_analysis_policy_v2,
    freeze_packet_assembly_protocol_orientation_v2,
    replay_metasyn_question_projection_spec_v2,
)
from literature_multiverse.native_packet_grounding_v2 import (
    MAX_IDENTITY_CLAIMS,
    MAX_IDENTITY_TEXT_CHARACTERS,
    PASSAGE_CANDIDATE_BINDING_V2_VERSION,
    PacketGroundingSchemaBundleV2,
    PacketPassageCandidateBindingV2,
    freeze_packet_grounding_schema_bundle_v2,
)
from literature_multiverse.native_question_projection import (
    QuestionProjectionSpecV1,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    compute_pipeline_fingerprint,
)

BUNDLE_COMPONENT_VERSION = "1"
CONFIG_VERSION = "metasyn-passage-hosted-anthropic-config-v2"
EXECUTION_BUNDLE_VERSION = "metasyn-passage-hosted-execution-bundle-v2"
COMPILED_SCHEMA_RECORD_VERSION = "metasyn-passage-compiled-schema-record-v2"
PREFLIGHT_CALL_VERSION = "metasyn-passage-source-free-preflight-call-v2"
INVENTORY_REQUEST_VERSION = "metasyn-passage-inventory-request-v2"
PACKET_COMPILER_GATE_VERSION = "metasyn-passage-packet-compiler-gate-v2"
PACKET_COST_PROBE_VERSION = "metasyn-passage-packet-cost-probe-v2"
PACKET_ROW_COST_VERSION = "metasyn-passage-packet-row-cost-envelope-v2"
ROW_PROTOCOL_ORIENTATION_VERSION = "metasyn-passage-row-protocol-orientation-v2"
COST_GROUP_VERSION = "metasyn-passage-cost-group-v2"
COST_ENVELOPE_VERSION = "metasyn-passage-global-cost-envelope-v2"

DEFAULT_CONFIG_PATH = Path("configs/benchmarks/metasyn-passage-hosted-anthropic-v2.json")
EXPECTED_PUBLICATION_COUNT = 32
EXPECTED_QUESTION_COUNT = 10
EXPECTED_COMPONENT_COUNT = 10
MAX_ACCEPTED_CANDIDATES_PER_ROW = 8
PREFLIGHT_CALL_COUNT = 8
MAX_INVENTORY_CALLS = 32
MAX_PACKET_CALLS = 256
MAX_PROVIDER_CALLS = PREFLIGHT_CALL_COUNT + MAX_INVENTORY_CALLS + MAX_PACKET_CALLS

EFFECT_KINDS: tuple[EffectKind, ...] = (
    "direct_standard_error",
    "direct_variance",
    "direct_confidence_interval",
    "continuous_group_statistics",
    "binary_group_statistics",
)

_BUNDLE_ENTRYPOINTS = (
    "src/literature_multiverse/metasyn_passage_hosted_bundle_v2.py",
    # Assembly is a required downstream acceptance boundary even though this offline
    # component never calls it.  Making it a closure root prevents an execution
    # bundle from remaining current after assembly semantics change.
    "src/literature_multiverse/native_packet_assembly_v2.py",
)
_RUNTIME_ENTRYPOINTS = (
    "src/literature_multiverse/metasyn_passage_hosted_runtime_v2.py",
    "scripts/run_metasyn_passage_hosted_runtime_v2.py",
)
_BUNDLE_NON_PYTHON_FILES = (
    DEFAULT_CONFIG_PATH.as_posix(),
    INVENTORY_PROMPT_PATH.as_posix(),
    PACKET_PROMPT_PATH.as_posix(),
    "pyproject.toml",
    "uv.lock",
)
_INSTALLED_DEPENDENCIES = ("anthropic", "jsonschema", "pyarrow", "pydantic")


class MetaSynPassageHostedBundleV2Error(ValueError):
    """The offline hosted configuration or execution identity failed closed."""


class _ExactContractModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


def _validate_sha256(value: str, field_name: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"metasyn_passage_hosted_v2_hash_invalid:{field_name}")
    return value


def _canonical_root(value: Path) -> Path:
    root = Path(os.path.abspath(value))
    try:
        mode = root.lstat().st_mode
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise MetaSynPassageHostedBundleV2Error(
            "metasyn_passage_hosted_v2_repository_root_unreadable"
        ) from exc
    if stat.S_ISLNK(mode):
        raise MetaSynPassageHostedBundleV2Error(
            "metasyn_passage_hosted_v2_repository_root_symlink_forbidden"
        )
    if not resolved.is_dir():
        raise MetaSynPassageHostedBundleV2Error(
            "metasyn_passage_hosted_v2_repository_root_not_directory"
        )
    return resolved


def _checked_repository_file(*, root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if (
        not relative_path
        or "\\" in relative_path
        or relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
        or relative_path.startswith("./")
        or relative.as_posix() != relative_path
    ):
        raise MetaSynPassageHostedBundleV2Error("metasyn_passage_hosted_v2_file_path_unsafe")
    candidate = root
    for part in relative.parts:
        candidate /= part
        try:
            mode = candidate.lstat().st_mode
        except OSError as exc:
            raise MetaSynPassageHostedBundleV2Error(
                f"metasyn_passage_hosted_v2_file_missing:{relative_path}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise MetaSynPassageHostedBundleV2Error(
                f"metasyn_passage_hosted_v2_file_symlink_forbidden:{relative_path}"
            )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise MetaSynPassageHostedBundleV2Error(
            f"metasyn_passage_hosted_v2_file_missing:{relative_path}"
        ) from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise MetaSynPassageHostedBundleV2Error(
            f"metasyn_passage_hosted_v2_file_not_regular:{relative_path}"
        )
    return resolved


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynPassageHostedBundleV2Error(
            "metasyn_passage_hosted_v2_config_unreadable"
        ) from exc
    if not isinstance(value, dict):
        raise MetaSynPassageHostedBundleV2Error("metasyn_passage_hosted_v2_config_not_object")
    return value


def _usd_micros(value: Decimal) -> int:
    return int((value * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING))


class MetaSynPassageHostedConfigV2(_ExactContractModel):
    """Literal, self-hashed limits and scientific non-authority policy."""

    config_version: Literal["metasyn-passage-hosted-anthropic-config-v2"] = CONFIG_VERSION
    diagnostic_scope: Literal["label_blind_passage_grounding_and_typed_effect_yield_only"] = (
        "label_blind_passage_grounding_and_typed_effect_yield_only"
    )
    runtime_provider: Literal["anthropic_first_party_api"] = "anthropic_first_party_api"
    model: Literal["claude-sonnet-5"] = "claude-sonnet-5"
    timeout_seconds: Literal[600.0] = 600.0
    input_rate_usd_per_million_tokens: Literal["2"] = "2"
    output_rate_usd_per_million_tokens: Literal["10"] = "10"
    pricing_source_url: Literal["https://platform.claude.com/docs/en/about-claude/pricing"] = (
        "https://platform.claude.com/docs/en/about-claude/pricing"
    )
    pricing_rate_table_sha256: str
    service_tier: Literal["standard_only"] = "standard_only"
    fixed_framing_tokens: Literal[1024] = 1024
    system_prompt: Annotated[str, Field(min_length=1, max_length=4000)]

    inventory_max_output_tokens: Literal[32768] = 32768
    packet_max_output_tokens: Literal[65536] = 65536
    maximum_input_tokens_all_calls: Literal[11000000] = 11_000_000
    maximum_provider_calls: Literal[296] = MAX_PROVIDER_CALLS
    maximum_authorized_cost_usd_micros: Literal[210000000] = 210_000_000
    question_count: Literal[10] = EXPECTED_QUESTION_COUNT
    publication_count: Literal[32] = EXPECTED_PUBLICATION_COUNT
    maximum_candidates_per_publication: Literal[8] = MAX_ACCEPTED_CANDIDATES_PER_ROW
    preflight_call_count: Literal[8] = PREFLIGHT_CALL_COUNT
    inventory_call_count: Literal[32] = MAX_INVENTORY_CALLS
    packet_call_ceiling: Literal[256] = MAX_PACKET_CALLS

    model_calls_per_request: Literal[1] = 1
    application_retries_per_request: Literal[0] = 0
    sdk_retries_per_request: Literal[0] = 0
    orphan_or_ambiguous_attempt_retry_permitted: Literal[False] = False
    operator_authorized_source_transmission: Literal[True] = True
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    yield_only_no_accuracy_or_release_authority: Literal[True] = True
    claim_release_authority: Literal[False] = False
    continuous_group_effect_format: Literal["hedges_g"] = "hedges_g"
    binary_group_effect_format: Literal["odds_ratio"] = "odds_ratio"

    inventory_smoke_row_ordinal: Literal[0] = 0
    packet_smoke_priority_row_ordinal: Literal[21] = 21
    packet_smoke_fallback_order: Literal["priority_row_then_row_ordinal_then_candidate_index"] = (
        "priority_row_then_row_ordinal_then_candidate_index"
    )
    packet_smoke_max_already_authorized_calls: Literal[3] = 3
    packet_smoke_pass_condition: Literal[
        "at_least_one_fully_grounded_and_assembled_completed_typed_effect"
    ] = "at_least_one_fully_grounded_and_assembled_completed_typed_effect"
    packet_abstention_semantics: Literal["terminal_valid_but_does_not_pass_packet_spend_gate"] = (
        "terminal_valid_but_does_not_pass_packet_spend_gate"
    )
    remaining_packet_calls_blocked_until_packet_smoke_passes: Literal[True] = True

    packet_capacity_policy: Literal[
        "reduced_canonical_response_shape_with_conservative_utf8_token_proof_v1"
    ] = "reduced_canonical_response_shape_with_conservative_utf8_token_proof_v1"
    packet_native_max_identity_claims: Literal[32] = MAX_IDENTITY_CLAIMS
    packet_accepted_max_identity_claims: Literal[15] = 15
    packet_native_max_identity_text_characters: Literal[512] = MAX_IDENTITY_TEXT_CHARACTERS
    packet_accepted_max_identity_text_characters: Literal[256] = 256
    packet_accepted_max_evidence_quote_characters: Literal[1800] = 1800
    packet_accepted_max_numeric_claims: Literal[24] = 24
    packet_accepted_max_numeric_token_characters: Literal[33] = 33
    packet_accepted_max_effect_format_characters: Literal[64] = 64
    packet_accepted_max_effect_unit_characters: Literal[64] = 64
    packet_accepted_max_timepoint_identity_fields: Literal[2] = 2
    packet_accepted_max_timepoint_identity_characters: Literal[256] = 256
    canonical_json_worst_case_bytes_per_text_character: Literal[6] = 6
    packet_canonical_json_fixed_overhead_bytes: Literal[8192] = 8192
    packet_accepted_canonical_json_utf8_byte_ceiling: Literal[50624] = 50_624
    packet_max_tokens_capacity_proof: Literal[
        "max_output_tokens_gte_one_token_per_accepted_canonical_json_utf8_byte"
    ] = "max_output_tokens_gte_one_token_per_accepted_canonical_json_utf8_byte"
    truncation_or_max_tokens_disposition: Literal[
        "runtime_capacity_failure_not_scientific_abstention"
    ] = "runtime_capacity_failure_not_scientific_abstention"

    inventory_canonical_json_fixed_overhead_bytes: Literal[8192] = 8192
    inventory_accepted_max_candidates_including_sentinel: Literal[9] = 9
    inventory_accepted_max_outcome_quote_characters: Literal[256] = 256
    inventory_accepted_max_outcome_id_characters: Literal[64] = 64
    inventory_accepted_max_passage_ids_per_candidate: Literal[4] = 4
    inventory_passage_id_canonical_json_bytes: Literal[69] = 69
    inventory_accepted_canonical_json_utf8_byte_ceiling: Literal[27956] = 27_956
    config_sha256: str

    @field_validator("pricing_rate_table_sha256", "config_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_config(self) -> MetaSynPassageHostedConfigV2:
        rate_table = {
            "input_rate_usd_per_million_tokens": (self.input_rate_usd_per_million_tokens),
            "model": self.model,
            "output_rate_usd_per_million_tokens": (self.output_rate_usd_per_million_tokens),
            "pricing_source_url": self.pricing_source_url,
            "service_tier": self.service_tier,
        }
        if self.pricing_rate_table_sha256 != hash_canonical(rate_table):
            raise ValueError("metasyn_passage_hosted_v2_pricing_hash_mismatch")
        packet_text_characters = (
            self.packet_accepted_max_evidence_quote_characters
            + self.packet_accepted_max_effect_format_characters
            + self.packet_accepted_max_effect_unit_characters
            + self.packet_accepted_max_numeric_claims
            * self.packet_accepted_max_numeric_token_characters
            + self.packet_accepted_max_identity_claims
            * self.packet_accepted_max_identity_text_characters
            + self.packet_accepted_max_timepoint_identity_fields
            * self.packet_accepted_max_timepoint_identity_characters
        )
        packet_bound = (
            self.packet_canonical_json_fixed_overhead_bytes
            + self.canonical_json_worst_case_bytes_per_text_character * packet_text_characters
        )
        if packet_bound != self.packet_accepted_canonical_json_utf8_byte_ceiling:
            raise ValueError("metasyn_passage_hosted_v2_packet_capacity_bound_mismatch")
        if self.packet_max_output_tokens < packet_bound:
            raise ValueError("metasyn_passage_hosted_v2_packet_output_capacity_unsafe")
        inventory_bound = (
            self.inventory_canonical_json_fixed_overhead_bytes
            + self.canonical_json_worst_case_bytes_per_text_character
            * self.inventory_accepted_max_candidates_including_sentinel
            * (
                self.inventory_accepted_max_outcome_quote_characters
                + self.inventory_accepted_max_outcome_id_characters
            )
            + self.inventory_accepted_max_candidates_including_sentinel
            * self.inventory_accepted_max_passage_ids_per_candidate
            * self.inventory_passage_id_canonical_json_bytes
        )
        if inventory_bound != self.inventory_accepted_canonical_json_utf8_byte_ceiling:
            raise ValueError("metasyn_passage_hosted_v2_inventory_capacity_bound_mismatch")
        if self.inventory_max_output_tokens < inventory_bound:
            raise ValueError("metasyn_passage_hosted_v2_inventory_output_capacity_unsafe")
        theoretical_output_tokens = (
            3 * self.inventory_max_output_tokens
            + 5 * self.packet_max_output_tokens
            + self.inventory_call_count * self.inventory_max_output_tokens
            + self.packet_call_ceiling * self.packet_max_output_tokens
        )
        theoretical_cost = (
            Decimal(self.maximum_input_tokens_all_calls)
            * Decimal(self.input_rate_usd_per_million_tokens)
            + Decimal(theoretical_output_tokens) * Decimal(self.output_rate_usd_per_million_tokens)
        ) / Decimal(1_000_000)
        if self.maximum_authorized_cost_usd_micros < _usd_micros(theoretical_cost):
            raise ValueError("metasyn_passage_hosted_v2_global_budget_below_limits")
        payload = self.model_dump(mode="json", exclude={"config_sha256"})
        if self.config_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_hosted_v2_config_hash_mismatch")
        return self

    def anthropic_config(self) -> AnthropicBoundedConfigV1:
        return AnthropicBoundedConfigV1(
            model=self.model,
            timeout_seconds=self.timeout_seconds,
        )

    def assembly_analysis_policy(self) -> PacketAssemblyAnalysisPolicyV2:
        return freeze_packet_assembly_analysis_policy_v2(
            continuous_group_effect_format=EffectFormat(self.continuous_group_effect_format),
            binary_group_effect_format=EffectFormat(self.binary_group_effect_format),
        )


def load_metasyn_passage_hosted_config_v2(
    *,
    repository_root: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> tuple[MetaSynPassageHostedConfigV2, str]:
    """Load only the literal repository config path and return its file hash."""

    root = _canonical_root(repository_root)
    if config_path != DEFAULT_CONFIG_PATH:
        raise MetaSynPassageHostedBundleV2Error("metasyn_passage_hosted_v2_config_path_not_literal")
    path = _checked_repository_file(root=root, relative_path=DEFAULT_CONFIG_PATH.as_posix())
    try:
        config = MetaSynPassageHostedConfigV2.model_validate(_read_json_object(path))
    except ValueError as exc:
        raise MetaSynPassageHostedBundleV2Error(
            "metasyn_passage_hosted_v2_config_contract_invalid"
        ) from exc
    return config, sha256_file(path)


class CompiledSchemaRecordV2(_ExactContractModel):
    record_version: Literal["metasyn-passage-compiled-schema-record-v2"] = (
        COMPILED_SCHEMA_RECORD_VERSION
    )
    schema_kind: Literal["inventory", "packet"]
    effect_kind: EffectKind | None
    context_binding_sha256: str
    original_schema_sha256: str
    full_acceptance_schema_sha256: str
    compiled_schema: AnthropicCompiledSchemaV1
    compiled_schema_sha256: str
    wire_schema_sha256: str
    wire_optional_parameter_count: Annotated[int, Field(ge=0, le=24)]
    wire_union_parameter_count: Annotated[int, Field(ge=0, le=16)]
    record_sha256: str

    @field_validator(
        "context_binding_sha256",
        "original_schema_sha256",
        "full_acceptance_schema_sha256",
        "compiled_schema_sha256",
        "wire_schema_sha256",
        "record_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_record(self) -> CompiledSchemaRecordV2:
        if (self.schema_kind == "inventory") != (self.effect_kind is None):
            raise ValueError("metasyn_passage_hosted_v2_compiled_schema_shape_invalid")
        aliases = {
            "original_schema_sha256": self.compiled_schema.original_schema_sha256,
            "full_acceptance_schema_sha256": (self.compiled_schema.full_acceptance_schema_sha256),
            "compiled_schema_sha256": self.compiled_schema.compiled_schema_sha256,
            "wire_schema_sha256": self.compiled_schema.wire_schema_sha256,
            "wire_optional_parameter_count": (self.compiled_schema.wire_optional_parameter_count),
            "wire_union_parameter_count": (self.compiled_schema.wire_union_parameter_count),
        }
        if any(getattr(self, key) != expected for key, expected in aliases.items()):
            raise ValueError("metasyn_passage_hosted_v2_compiled_schema_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"record_sha256"})
        if self.record_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_hosted_v2_compiled_schema_hash_mismatch")
        return self


@lru_cache(maxsize=512)
def _compile_schema_cached(
    original_schema_json: str,
    full_acceptance_schema_sha256: str,
) -> AnthropicCompiledSchemaV1:
    parsed = json.loads(original_schema_json)
    if not isinstance(parsed, dict):  # pragma: no cover - caller invariant
        raise MetaSynPassageHostedBundleV2Error(
            "metasyn_passage_hosted_v2_cached_schema_not_object"
        )
    return compile_anthropic_bounded_schema(
        original_schema=parsed,
        full_acceptance_schema_sha256=full_acceptance_schema_sha256,
    )


def _freeze_compiled_schema_record(
    *,
    schema_kind: Literal["inventory", "packet"],
    effect_kind: EffectKind | None,
    context_binding_sha256: str,
    original_schema: Mapping[str, Any],
    full_acceptance_schema_sha256: str,
) -> CompiledSchemaRecordV2:
    original_json = canonical_json_bytes(dict(original_schema)).decode("utf-8")
    compiled = _compile_schema_cached(original_json, full_acceptance_schema_sha256)
    payload = {
        "record_version": COMPILED_SCHEMA_RECORD_VERSION,
        "schema_kind": schema_kind,
        "effect_kind": effect_kind,
        "context_binding_sha256": context_binding_sha256,
        "original_schema_sha256": hash_canonical(dict(original_schema)),
        "full_acceptance_schema_sha256": full_acceptance_schema_sha256,
        "compiled_schema": compiled,
        "compiled_schema_sha256": compiled.compiled_schema_sha256,
        "wire_schema_sha256": compiled.wire_schema_sha256,
        "wire_optional_parameter_count": compiled.wire_optional_parameter_count,
        "wire_union_parameter_count": compiled.wire_union_parameter_count,
    }
    return CompiledSchemaRecordV2.model_validate(
        {**payload, "record_sha256": hash_canonical(payload)}
    )


def _cap_string_nodes(value: Any, maximum: int) -> int:
    changed = 0
    if isinstance(value, dict):
        if value.get("type") == "string" and isinstance(value.get("maxLength"), int):
            if value["maxLength"] > maximum:
                value["maxLength"] = maximum
            changed += 1
        for child in value.values():
            changed += _cap_string_nodes(child, maximum)
    elif isinstance(value, list):
        for child in value:
            changed += _cap_string_nodes(child, maximum)
    return changed


def capacity_limited_packet_schema_v2(
    *,
    schema_bundle: PacketGroundingSchemaBundleV2 | Mapping[str, Any],
    config: MetaSynPassageHostedConfigV2,
) -> dict[str, Any]:
    """Return the explicit provider-accepted subset covered by the token proof."""

    bundle = PacketGroundingSchemaBundleV2.model_validate(
        schema_bundle.model_dump(mode="json")
        if isinstance(schema_bundle, PacketGroundingSchemaBundleV2)
        else schema_bundle
    )
    schema = deepcopy(bundle.model_response_schema)
    observed = {
        "evidence_quote": 0,
        "effect_format_token": 0,
        "effect_unit": 0,
        "numeric_claims": 0,
        "identity_claims": 0,
        "verbatim_identity_text": 0,
        "anchor": 0,
        "raw_label": 0,
    }

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                limits = {
                    "evidence_quote": config.packet_accepted_max_evidence_quote_characters,
                    "effect_format_token": (config.packet_accepted_max_effect_format_characters),
                    "effect_unit": config.packet_accepted_max_effect_unit_characters,
                    "verbatim_identity_text": (config.packet_accepted_max_identity_text_characters),
                    "anchor": config.packet_accepted_max_timepoint_identity_characters,
                    "raw_label": (config.packet_accepted_max_timepoint_identity_characters),
                }
                for name, maximum in limits.items():
                    child = properties.get(name)
                    if isinstance(child, dict):
                        string_nodes = _cap_string_nodes(child, maximum)
                        # Effect-format families may specialize these properties
                        # to a finite enum or to null.  Presence is sufficient in
                        # those cases because the finite values are already below
                        # the declared character bound.
                        observed[name] += (
                            1 if name in {"effect_format_token", "effect_unit"} else string_nodes
                        )
                numeric = properties.get("numeric_claims")
                if isinstance(numeric, dict):
                    numeric["maxItems"] = config.packet_accepted_max_numeric_claims
                    observed["numeric_claims"] += 1
                identities = properties.get("identity_claims")
                if isinstance(identities, dict):
                    identities["maxItems"] = config.packet_accepted_max_identity_claims
                    observed["identity_claims"] += 1
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(schema)
    if any(count == 0 for count in observed.values()):
        missing = ",".join(sorted(key for key, count in observed.items() if count == 0))
        raise MetaSynPassageHostedBundleV2Error(
            f"metasyn_passage_hosted_v2_capacity_schema_target_missing:{missing}"
        )
    try:
        validator = validator_for(schema)
        validator.check_schema(schema)
        validator(schema).validate(bundle.completed_fixture)
        validator(schema).validate(bundle.abstaining_fixture)
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
            raise
        raise MetaSynPassageHostedBundleV2Error(
            "metasyn_passage_hosted_v2_capacity_schema_invalid"
        ) from exc
    return json.loads(canonical_json_bytes(schema))


def maximum_canonical_json_utf8_bytes_for_schema_v2(
    schema: Mapping[str, Any],
) -> int:
    """Bound compact canonical JSON bytes for the closed response schema.

    JSON string content can expand to at most six UTF-8 bytes per schema character
    (for example ``\\u0000``). Closed objects, bounded arrays, local references,
    finite enums, and discriminated unions are supported. Any unbounded or unknown
    shape fails closed instead of relying on a guessed output-token allowance.
    """

    root = json.loads(canonical_json_bytes(dict(schema)))

    def resolve(reference: str) -> dict[str, Any]:
        if not reference.startswith("#/"):
            raise MetaSynPassageHostedBundleV2Error(
                "metasyn_passage_hosted_v2_capacity_reference_nonlocal"
            )
        current: Any = root
        for raw in reference[2:].split("/"):
            token = raw.replace("~1", "/").replace("~0", "~")
            if not isinstance(current, dict) or token not in current:
                raise MetaSynPassageHostedBundleV2Error(
                    "metasyn_passage_hosted_v2_capacity_reference_missing"
                )
            current = current[token]
        if not isinstance(current, dict):
            raise MetaSynPassageHostedBundleV2Error(
                "metasyn_passage_hosted_v2_capacity_reference_not_schema"
            )
        return current

    def bound(node: Any, reference_stack: tuple[str, ...] = ()) -> int:
        if not isinstance(node, dict):
            raise MetaSynPassageHostedBundleV2Error(
                "metasyn_passage_hosted_v2_capacity_schema_node_invalid"
            )
        reference = node.get("$ref")
        if isinstance(reference, str):
            if len(node) != 1:
                raise MetaSynPassageHostedBundleV2Error(
                    "metasyn_passage_hosted_v2_capacity_reference_sibling"
                )
            if reference in reference_stack:
                raise MetaSynPassageHostedBundleV2Error(
                    "metasyn_passage_hosted_v2_capacity_reference_cycle"
                )
            return bound(resolve(reference), (*reference_stack, reference))
        if "const" in node:
            return len(canonical_json_bytes(node["const"]))
        enum = node.get("enum")
        if isinstance(enum, list) and enum:
            return max(len(canonical_json_bytes(item)) for item in enum)
        for keyword in ("oneOf", "anyOf"):
            branches = node.get(keyword)
            if isinstance(branches, list) and branches:
                return max(bound(branch, reference_stack) for branch in branches)
        if "allOf" in node:
            raise MetaSynPassageHostedBundleV2Error(
                "metasyn_passage_hosted_v2_capacity_allof_unsupported"
            )

        schema_type = node.get("type")
        if isinstance(schema_type, list):
            return max(bound({**node, "type": item}, reference_stack) for item in schema_type)
        if schema_type == "null":
            return 4
        if schema_type == "boolean":
            return 5
        if schema_type == "string":
            maximum = node.get("maxLength")
            if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 0:
                raise MetaSynPassageHostedBundleV2Error(
                    "metasyn_passage_hosted_v2_capacity_string_unbounded"
                )
            return 2 + 6 * maximum
        if schema_type in {"integer", "number"}:
            endpoints = [node.get("minimum"), node.get("maximum")]
            if any(value is None for value in endpoints):
                raise MetaSynPassageHostedBundleV2Error(
                    "metasyn_passage_hosted_v2_capacity_number_unbounded"
                )
            return max(len(canonical_json_bytes(value)) for value in endpoints)
        if schema_type == "array" or "items" in node:
            maximum = node.get("maxItems")
            items = node.get("items")
            if (
                not isinstance(maximum, int)
                or isinstance(maximum, bool)
                or maximum < 0
                or not isinstance(items, dict)
            ):
                raise MetaSynPassageHostedBundleV2Error(
                    "metasyn_passage_hosted_v2_capacity_array_unbounded"
                )
            item_bytes = bound(items, reference_stack)
            return 2 + maximum * item_bytes + max(0, maximum - 1)
        properties = node.get("properties")
        if schema_type == "object" or isinstance(properties, dict):
            if not isinstance(properties, dict) or node.get("additionalProperties") is not False:
                raise MetaSynPassageHostedBundleV2Error(
                    "metasyn_passage_hosted_v2_capacity_object_not_closed"
                )
            members = [
                len(canonical_json_bytes(name)) + 1 + bound(child, reference_stack)
                for name, child in sorted(properties.items())
            ]
            return 2 + sum(members) + max(0, len(members) - 1)
        raise MetaSynPassageHostedBundleV2Error(
            "metasyn_passage_hosted_v2_capacity_schema_shape_unsupported"
        )

    return bound(root)


def _synthetic_binding(effect_kind: EffectKind) -> PacketPassageCandidateBindingV2:
    passage_id = f"p2-{hash_canonical({'synthetic_effect_kind': effect_kind})}"
    candidate = MetaSynPassageCandidateV2(
        candidate_index=1,
        canonical_outcome_id="outcome-01",
        outcome_concept_quote="Synthetic outcome",
        effect_kind=effect_kind,
        passage_ids=[passage_id],
    )
    payload = {
        "binding_version": PASSAGE_CANDIDATE_BINDING_V2_VERSION,
        "candidate_index": candidate.candidate_index,
        "candidate_descriptor_sha256": candidate.descriptor_sha256,
        "canonical_outcome_id": candidate.canonical_outcome_id,
        "outcome_concept_quote": candidate.outcome_concept_quote,
        "effect_kind": candidate.effect_kind,
        "passage_ids": candidate.passage_ids,
        "projection_sha256": hash_canonical({"synthetic_projection_for_effect_kind": effect_kind}),
    }
    return PacketPassageCandidateBindingV2.model_validate(
        {**payload, "binding_sha256": hash_canonical(payload)}
    )


class PacketCompilerGateV2(_ExactContractModel):
    gate_version: Literal["metasyn-passage-packet-compiler-gate-v2"] = PACKET_COMPILER_GATE_VERSION
    effect_kind: EffectKind
    synthetic_source_only: Literal[True] = True
    candidate_binding: PacketPassageCandidateBindingV2
    candidate_binding_sha256: str
    native_schema_bundle: PacketGroundingSchemaBundleV2
    native_schema_bundle_sha256: str
    capacity_limited_schema: dict[str, Any]
    capacity_limited_schema_sha256: str
    accepted_canonical_json_utf8_byte_ceiling: Annotated[int, Field(ge=1)]
    compiled_schema_record: CompiledSchemaRecordV2
    compiled_schema_record_sha256: str
    completed_fixture_sha256: str
    abstaining_fixture_sha256: str
    gate_sha256: str

    @field_validator(
        "candidate_binding_sha256",
        "native_schema_bundle_sha256",
        "capacity_limited_schema_sha256",
        "compiled_schema_record_sha256",
        "completed_fixture_sha256",
        "abstaining_fixture_sha256",
        "gate_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_gate(self) -> PacketCompilerGateV2:
        if (
            self.candidate_binding.effect_kind != self.effect_kind
            or self.candidate_binding_sha256 != self.candidate_binding.binding_sha256
            or self.native_schema_bundle_sha256 != self.native_schema_bundle.schema_bundle_sha256
            or self.native_schema_bundle.candidate_binding_sha256 != self.candidate_binding_sha256
            or self.capacity_limited_schema_sha256 != hash_canonical(self.capacity_limited_schema)
            or self.accepted_canonical_json_utf8_byte_ceiling
            != maximum_canonical_json_utf8_bytes_for_schema_v2(self.capacity_limited_schema)
            or self.compiled_schema_record_sha256 != self.compiled_schema_record.record_sha256
            or self.compiled_schema_record.effect_kind != self.effect_kind
            or self.compiled_schema_record.context_binding_sha256 != self.candidate_binding_sha256
            or self.compiled_schema_record.original_schema_sha256
            != self.capacity_limited_schema_sha256
            or self.completed_fixture_sha256 != self.native_schema_bundle.completed_fixture_sha256
            or self.abstaining_fixture_sha256 != self.native_schema_bundle.abstaining_fixture_sha256
        ):
            raise ValueError("metasyn_passage_hosted_v2_packet_gate_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"gate_sha256"})
        if self.gate_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_hosted_v2_packet_gate_hash_mismatch")
        return self


def _freeze_packet_compiler_gates(
    config: MetaSynPassageHostedConfigV2,
) -> list[PacketCompilerGateV2]:
    gates: list[PacketCompilerGateV2] = []
    for effect_kind in EFFECT_KINDS:
        binding = _synthetic_binding(effect_kind)
        native = freeze_packet_grounding_schema_bundle_v2(binding=binding)
        accepted = capacity_limited_packet_schema_v2(
            schema_bundle=native,
            config=config,
        )
        accepted_sha = hash_canonical(accepted)
        accepted_byte_ceiling = maximum_canonical_json_utf8_bytes_for_schema_v2(accepted)
        if accepted_byte_ceiling > (config.packet_accepted_canonical_json_utf8_byte_ceiling):
            raise MetaSynPassageHostedBundleV2Error(
                "metasyn_passage_hosted_v2_packet_schema_exceeds_capacity_proof"
            )
        compiled = _freeze_compiled_schema_record(
            schema_kind="packet",
            effect_kind=effect_kind,
            context_binding_sha256=binding.binding_sha256,
            original_schema=accepted,
            full_acceptance_schema_sha256=accepted_sha,
        )
        payload = {
            "gate_version": PACKET_COMPILER_GATE_VERSION,
            "effect_kind": effect_kind,
            "synthetic_source_only": True,
            "candidate_binding": binding,
            "candidate_binding_sha256": binding.binding_sha256,
            "native_schema_bundle": native,
            "native_schema_bundle_sha256": native.schema_bundle_sha256,
            "capacity_limited_schema": accepted,
            "capacity_limited_schema_sha256": accepted_sha,
            "accepted_canonical_json_utf8_byte_ceiling": accepted_byte_ceiling,
            "compiled_schema_record": compiled,
            "compiled_schema_record_sha256": compiled.record_sha256,
            "completed_fixture_sha256": native.completed_fixture_sha256,
            "abstaining_fixture_sha256": native.abstaining_fixture_sha256,
        }
        gates.append(
            PacketCompilerGateV2.model_validate({**payload, "gate_sha256": hash_canonical(payload)})
        )
    return gates


class RowProtocolOrientationV2(_ExactContractModel):
    row_orientation_version: Literal["metasyn-passage-row-protocol-orientation-v2"] = (
        ROW_PROTOCOL_ORIENTATION_VERSION
    )
    row_ordinal: Annotated[int, Field(ge=0, lt=EXPECTED_PUBLICATION_COUNT)]
    row_key: str
    row_input_sha256: str
    question_surface_sha256: str
    question_surface_question_spec_sha256: str
    protocol: QuestionProjectionSpecV1
    protocol_question_spec_sha256: str
    protocol_projection_spec_sha256: str
    protocol_orientation: PacketAssemblyProtocolOrientationV2
    protocol_orientation_sha256: str
    row_orientation_sha256: str

    @field_validator(
        "row_input_sha256",
        "question_surface_sha256",
        "question_surface_question_spec_sha256",
        "protocol_question_spec_sha256",
        "protocol_projection_spec_sha256",
        "protocol_orientation_sha256",
        "row_orientation_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_orientation(self) -> RowProtocolOrientationV2:
        if (
            self.protocol_orientation_sha256 != self.protocol_orientation.orientation_sha256
            or self.protocol_orientation.question_surface_sha256 != self.question_surface_sha256
            or self.question_surface_question_spec_sha256
            != self.protocol_orientation.question_surface_question_spec_sha256
            or self.protocol_question_spec_sha256
            != self.protocol_orientation.protocol_question_spec_sha256
            or self.protocol_projection_spec_sha256
            != self.protocol_orientation.protocol_projection_spec_sha256
            or self.protocol_question_spec_sha256 != self.protocol.question_spec_sha256
            or self.protocol_projection_spec_sha256 != self.protocol.projection_spec_sha256
        ):
            raise ValueError("metasyn_passage_hosted_v2_protocol_orientation_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"row_orientation_sha256"})
        if self.row_orientation_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_hosted_v2_protocol_orientation_hash_mismatch")
        return self


def _freeze_row_protocol_orientations(
    extraction_inputs: MetaSynExtractionInputsV2,
) -> list[RowProtocolOrientationV2]:
    output: list[RowProtocolOrientationV2] = []
    for row in extraction_inputs.rows:
        protocol = replay_metasyn_question_projection_spec_v2(question_surface=row.question_surface)
        orientation = freeze_packet_assembly_protocol_orientation_v2(
            question_surface=row.question_surface
        )
        if (
            orientation.question_surface_question_spec_sha256 != row.upstream_question_spec_sha256
            or orientation.protocol_question_spec_sha256
            != row.projection_v2.lineage_binding.question_spec_sha256
            or orientation.protocol_question_spec_sha256 != protocol.question_spec_sha256
            or orientation.protocol_projection_spec_sha256 != protocol.projection_spec_sha256
        ):
            raise MetaSynPassageHostedBundleV2Error(
                "metasyn_passage_hosted_v2_protocol_orientation_row_lineage_mismatch"
            )
        payload = {
            "row_orientation_version": ROW_PROTOCOL_ORIENTATION_VERSION,
            "row_ordinal": row.row_ordinal,
            "row_key": row.row_key,
            "row_input_sha256": row.row_input_sha256,
            "question_surface_sha256": row.question_surface_sha256,
            "question_surface_question_spec_sha256": (row.question_surface.question_spec_sha256),
            "protocol": protocol,
            "protocol_question_spec_sha256": protocol.question_spec_sha256,
            "protocol_projection_spec_sha256": protocol.projection_spec_sha256,
            "protocol_orientation": orientation,
            "protocol_orientation_sha256": orientation.orientation_sha256,
        }
        output.append(
            RowProtocolOrientationV2.model_validate(
                {**payload, "row_orientation_sha256": hash_canonical(payload)}
            )
        )
    return output


class InventoryRequestV2(_ExactContractModel):
    inventory_request_version: Literal["metasyn-passage-inventory-request-v2"] = (
        INVENTORY_REQUEST_VERSION
    )
    row_ordinal: Annotated[int, Field(ge=0, lt=EXPECTED_PUBLICATION_COUNT)]
    row_key: str
    row_input_sha256: str
    inventory_input_sha256: str
    rendered_prompt_sha256: str
    inventory_schema_bundle_sha256: str
    compiled_schema_record: CompiledSchemaRecordV2
    compiled_schema_record_sha256: str
    request: AnthropicBoundedRequestV1
    request_sha256: str
    inventory_request_sha256: str

    @field_validator(
        "row_input_sha256",
        "inventory_input_sha256",
        "rendered_prompt_sha256",
        "inventory_schema_bundle_sha256",
        "compiled_schema_record_sha256",
        "request_sha256",
        "inventory_request_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_request(self) -> InventoryRequestV2:
        if (
            self.compiled_schema_record.schema_kind != "inventory"
            or self.compiled_schema_record.effect_kind is not None
            or self.compiled_schema_record_sha256 != self.compiled_schema_record.record_sha256
            or self.request_sha256 != self.request.request_sha256
            or self.request.schema_kind != "inventory"
            or self.request.effect_kind is not None
            or self.request.compiled_schema_sha256
            != self.compiled_schema_record.compiled_schema_sha256
            or self.request.base_prompt_sha256 != self.rendered_prompt_sha256
        ):
            raise ValueError("metasyn_passage_hosted_v2_inventory_request_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"inventory_request_sha256"})
        if self.inventory_request_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_hosted_v2_inventory_request_hash_mismatch")
        return self


def _freeze_inventory_requests(
    *,
    extraction_inputs: MetaSynExtractionInputsV2,
    config: MetaSynPassageHostedConfigV2,
    anthropic_config: AnthropicBoundedConfigV1,
    identity: AnthropicProviderIdentityV1,
) -> list[InventoryRequestV2]:
    records: list[InventoryRequestV2] = []
    for row in extraction_inputs.rows:
        schema_bundle = row.inventory_input.inventory_schema_bundle
        compiled = _freeze_compiled_schema_record(
            schema_kind="inventory",
            effect_kind=None,
            context_binding_sha256=row.inventory_input_sha256,
            original_schema=schema_bundle["provider_schema"],
            full_acceptance_schema_sha256=schema_bundle["full_acceptance_schema_sha256"],
        )
        request = freeze_anthropic_bounded_request(
            operation="metasyn-passage-inventory-v2",
            request_key=f"inventory-row-{row.row_ordinal:02d}",
            prompt=row.inventory_input.rendered_prompt,
            system=config.system_prompt,
            max_output_tokens=config.inventory_max_output_tokens,
            compiled_schema=compiled.compiled_schema,
            config=anthropic_config,
            schema_kind="inventory",
            effect_kind=None,
            identity=identity,
        )
        payload = {
            "inventory_request_version": INVENTORY_REQUEST_VERSION,
            "row_ordinal": row.row_ordinal,
            "row_key": row.row_key,
            "row_input_sha256": row.row_input_sha256,
            "inventory_input_sha256": row.inventory_input_sha256,
            "rendered_prompt_sha256": row.inventory_input.rendered_prompt_sha256,
            "inventory_schema_bundle_sha256": (row.inventory_input.inventory_schema_bundle_sha256),
            "compiled_schema_record": compiled,
            "compiled_schema_record_sha256": compiled.record_sha256,
            "request": request,
            "request_sha256": request.request_sha256,
        }
        records.append(
            InventoryRequestV2.model_validate(
                {**payload, "inventory_request_sha256": hash_canonical(payload)}
            )
        )
    return records


def _source_free_prompt(*, fixture: Mapping[str, Any], label: str) -> str:
    fixture_json = canonical_json_bytes(dict(fixture)).decode("utf-8")
    return (
        "This is a source-free JSON compatibility probe containing no scientific "
        "source or benchmark answer. Emit exactly the supplied synthetic fixture as "
        "one compact JSON object; preserve every key, JSON type, and string. Do not "
        "add commentary.\n"
        f"PROBE_LABEL={label}\n"
        f"FIXTURE_SHA256={hash_canonical(dict(fixture))}\n"
        f"FIXTURE_JSON={fixture_json}"
    )


class SourceFreePreflightCallV2(_ExactContractModel):
    preflight_call_version: Literal["metasyn-passage-source-free-preflight-call-v2"] = (
        PREFLIGHT_CALL_VERSION
    )
    preflight_ordinal: Annotated[int, Field(ge=0, lt=PREFLIGHT_CALL_COUNT)]
    probe_label: str
    schema_kind: Literal["inventory", "packet"]
    effect_kind: EffectKind | None
    source_bearing: Literal[False] = False
    expected_fixture: dict[str, Any]
    expected_fixture_sha256: str
    compiled_schema_record: CompiledSchemaRecordV2
    compiled_schema_record_sha256: str
    request: AnthropicBoundedRequestV1
    request_sha256: str
    preflight_call_sha256: str

    @field_validator(
        "expected_fixture_sha256",
        "compiled_schema_record_sha256",
        "request_sha256",
        "preflight_call_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_call(self) -> SourceFreePreflightCallV2:
        if (
            (self.schema_kind == "inventory") != (self.effect_kind is None)
            or self.expected_fixture_sha256 != hash_canonical(self.expected_fixture)
            or self.compiled_schema_record_sha256 != self.compiled_schema_record.record_sha256
            or self.compiled_schema_record.schema_kind != self.schema_kind
            or self.compiled_schema_record.effect_kind != self.effect_kind
            or self.request_sha256 != self.request.request_sha256
            or self.request.schema_kind != self.schema_kind
            or self.request.effect_kind != self.effect_kind
        ):
            raise ValueError("metasyn_passage_hosted_v2_preflight_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"preflight_call_sha256"})
        if self.preflight_call_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_hosted_v2_preflight_hash_mismatch")
        return self


def _freeze_source_free_preflight_plan(
    *,
    config: MetaSynPassageHostedConfigV2,
    anthropic_config: AnthropicBoundedConfigV1,
    identity: AnthropicProviderIdentityV1,
    packet_gates: Sequence[PacketCompilerGateV2],
) -> list[SourceFreePreflightCallV2]:
    passage_id = f"p2-{hash_canonical('source-free-inventory-passage')}"
    inventory_bundle = metasyn_candidate_inventory_schema_bundle_v2(
        allowed_outcome_ids=["outcome-01"],
        passage_ids=[passage_id],
    )
    inventory_compiled = _freeze_compiled_schema_record(
        schema_kind="inventory",
        effect_kind=None,
        context_binding_sha256=inventory_bundle["schema_bundle_sha256"],
        original_schema=inventory_bundle["provider_schema"],
        full_acceptance_schema_sha256=inventory_bundle["full_acceptance_schema_sha256"],
    )
    inventory_fixtures: list[tuple[str, dict[str, Any]]] = [
        (
            "inventory-candidates-found",
            {
                "inventory_version": "metasyn-passage-candidate-inventory-v2",
                "inventory_status": "candidates_found",
                "candidates": [
                    {
                        "candidate_index": 1,
                        "canonical_outcome_id": "outcome-01",
                        "outcome_concept_quote": "Synthetic outcome",
                        "effect_kind": "direct_confidence_interval",
                        "passage_ids": [passage_id],
                    }
                ],
                "has_more_or_uncertain": False,
            },
        ),
        (
            "inventory-no-candidate",
            {
                "inventory_version": "metasyn-passage-candidate-inventory-v2",
                "inventory_status": "no_candidate_found",
                "candidates": [],
                "has_more_or_uncertain": False,
            },
        ),
        (
            "inventory-overflow-or-uncertain",
            {
                "inventory_version": "metasyn-passage-candidate-inventory-v2",
                "inventory_status": "overflow_or_uncertain",
                "candidates": [],
                "has_more_or_uncertain": True,
            },
        ),
    ]
    plan: list[SourceFreePreflightCallV2] = []
    for ordinal, (label, fixture) in enumerate(inventory_fixtures):
        MetaSynCandidateInventoryV2.model_validate(fixture)
        validator_for(inventory_bundle["full_acceptance_schema"])(
            inventory_bundle["full_acceptance_schema"]
        ).validate(fixture)
        request = freeze_anthropic_bounded_request(
            operation="metasyn-passage-source-free-preflight-v2",
            request_key=f"preflight-{ordinal:02d}-{label}",
            prompt=_source_free_prompt(fixture=fixture, label=label),
            system=config.system_prompt,
            max_output_tokens=config.inventory_max_output_tokens,
            compiled_schema=inventory_compiled.compiled_schema,
            config=anthropic_config,
            schema_kind="inventory",
            effect_kind=None,
            identity=identity,
        )
        payload = {
            "preflight_call_version": PREFLIGHT_CALL_VERSION,
            "preflight_ordinal": ordinal,
            "probe_label": label,
            "schema_kind": "inventory",
            "effect_kind": None,
            "source_bearing": False,
            "expected_fixture": fixture,
            "expected_fixture_sha256": hash_canonical(fixture),
            "compiled_schema_record": inventory_compiled,
            "compiled_schema_record_sha256": inventory_compiled.record_sha256,
            "request": request,
            "request_sha256": request.request_sha256,
        }
        plan.append(
            SourceFreePreflightCallV2.model_validate(
                {**payload, "preflight_call_sha256": hash_canonical(payload)}
            )
        )

    for offset, gate in enumerate(packet_gates, start=3):
        fixture = gate.native_schema_bundle.completed_fixture
        validator_for(gate.capacity_limited_schema)(gate.capacity_limited_schema).validate(fixture)
        label = f"packet-{gate.effect_kind}"
        request_key_label = f"packet-{gate.effect_kind.replace('_', '-')}"
        request = freeze_anthropic_bounded_request(
            operation="metasyn-passage-source-free-preflight-v2",
            request_key=f"preflight-{offset:02d}-{request_key_label}",
            prompt=_source_free_prompt(fixture=fixture, label=label),
            system=config.system_prompt,
            max_output_tokens=config.packet_max_output_tokens,
            compiled_schema=gate.compiled_schema_record.compiled_schema,
            config=anthropic_config,
            schema_kind="packet",
            effect_kind=gate.effect_kind,
            identity=identity,
        )
        payload = {
            "preflight_call_version": PREFLIGHT_CALL_VERSION,
            "preflight_ordinal": offset,
            "probe_label": label,
            "schema_kind": "packet",
            "effect_kind": gate.effect_kind,
            "source_bearing": False,
            "expected_fixture": fixture,
            "expected_fixture_sha256": hash_canonical(fixture),
            "compiled_schema_record": gate.compiled_schema_record,
            "compiled_schema_record_sha256": (gate.compiled_schema_record.record_sha256),
            "request": request,
            "request_sha256": request.request_sha256,
        }
        plan.append(
            SourceFreePreflightCallV2.model_validate(
                {**payload, "preflight_call_sha256": hash_canonical(payload)}
            )
        )
    return plan


def _maximum_protocol_outcome(row: MetaSynExtractionRowInputV2) -> tuple[str, str]:
    candidates: list[tuple[int, str, str]] = []
    for outcome_id, text in row.question_surface.allowed_outcome_text_by_id.items():
        windows = (
            [text]
            if len(text) <= 256
            else [text[index : index + 256] for index in range(len(text) - 255)]
        )
        for quote in windows:
            score = len(canonical_json_bytes({"id": outcome_id, "quote": quote}))
            candidates.append((score, outcome_id, quote))
    if not candidates:  # pragma: no cover - question contract requires outcomes
        raise MetaSynPassageHostedBundleV2Error("metasyn_passage_hosted_v2_cost_outcome_missing")
    _, outcome_id, quote = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return outcome_id, quote


def _maximum_passage_ids(row: MetaSynExtractionRowInputV2) -> list[str]:
    ranked = sorted(
        row.projection_surface.passages,
        key=lambda passage: (
            len(canonical_json_bytes(passage.model_dump(mode="json"))),
            passage.passage_id,
        ),
        reverse=True,
    )
    return sorted(passage.passage_id for passage in ranked[:4])


class PacketEffectCostProbeV2(_ExactContractModel):
    probe_version: Literal["metasyn-passage-packet-cost-probe-v2"] = PACKET_COST_PROBE_VERSION
    effect_kind: EffectKind
    packet_input_sha256: str
    candidate_binding_sha256: str
    rendered_prompt_sha256: str
    rendered_prompt_utf8_bytes: Annotated[int, Field(ge=1)]
    capacity_limited_schema_sha256: str
    compiled_schema_sha256: str
    wire_schema_sha256: str
    cost_ceiling: AnthropicRequestCostCeilingV1
    cost_ceiling_sha256: str
    request_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    probe_sha256: str

    @field_validator(
        "packet_input_sha256",
        "candidate_binding_sha256",
        "rendered_prompt_sha256",
        "capacity_limited_schema_sha256",
        "compiled_schema_sha256",
        "wire_schema_sha256",
        "cost_ceiling_sha256",
        "probe_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_probe(self) -> PacketEffectCostProbeV2:
        if (
            self.cost_ceiling_sha256 != self.cost_ceiling.cost_ceiling_sha256
            or self.request_cost_ceiling_usd_micros
            != _usd_micros(self.cost_ceiling.request_cost_ceiling_usd)
            or self.cost_ceiling.max_output_tokens != 65536
            or self.cost_ceiling.transport_mode != "prompt_json_schema"
        ):
            raise ValueError("metasyn_passage_hosted_v2_packet_cost_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"probe_sha256"})
        if self.probe_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_hosted_v2_packet_cost_hash_mismatch")
        return self


class PacketRowCostEnvelopeV2(_ExactContractModel):
    row_cost_version: Literal["metasyn-passage-packet-row-cost-envelope-v2"] = (
        PACKET_ROW_COST_VERSION
    )
    row_ordinal: Annotated[int, Field(ge=0, lt=EXPECTED_PUBLICATION_COUNT)]
    row_key: str
    row_input_sha256: str
    synthetic_cost_probe_only: Literal[True] = True
    selected_outcome_id: str
    selected_outcome_quote_sha256: str
    selected_passage_ids: Annotated[list[str], Field(min_length=1, max_length=4)]
    evaluated_effect_kinds: list[EffectKind]
    effect_probes: Annotated[list[PacketEffectCostProbeV2], Field(min_length=5, max_length=5)]
    effect_probe_membership_sha256: str
    worst_effect_kind: EffectKind
    worst_probe_sha256: str
    per_call_input_token_ceiling: Annotated[int, Field(ge=1)]
    per_call_max_output_tokens: Annotated[int, Field(ge=1)]
    per_call_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    authorized_slot_multiplicity: Literal[8] = MAX_ACCEPTED_CANDIDATES_PER_ROW
    row_input_token_ceiling: Annotated[int, Field(ge=1)]
    row_output_token_ceiling: Annotated[int, Field(ge=1)]
    row_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    row_cost_sha256: str

    @field_validator(
        "row_input_sha256",
        "selected_outcome_quote_sha256",
        "effect_probe_membership_sha256",
        "worst_probe_sha256",
        "row_cost_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("selected_passage_ids")
    @classmethod
    def validate_passages(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("metasyn_passage_hosted_v2_cost_passages_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_row_cost(self) -> PacketRowCostEnvelopeV2:
        if self.evaluated_effect_kinds != list(EFFECT_KINDS):
            raise ValueError("metasyn_passage_hosted_v2_cost_effect_roster_mismatch")
        if [item.effect_kind for item in self.effect_probes] != list(EFFECT_KINDS):
            raise ValueError("metasyn_passage_hosted_v2_cost_probe_order_mismatch")
        if self.effect_probe_membership_sha256 != hash_canonical(
            [item.probe_sha256 for item in self.effect_probes]
        ):
            raise ValueError("metasyn_passage_hosted_v2_cost_probe_membership_mismatch")
        worst = max(
            self.effect_probes,
            key=lambda item: (
                item.request_cost_ceiling_usd_micros,
                item.cost_ceiling.conservative_input_token_ceiling,
                -EFFECT_KINDS.index(item.effect_kind),
            ),
        )
        if (
            self.worst_effect_kind != worst.effect_kind
            or self.worst_probe_sha256 != worst.probe_sha256
            or self.per_call_input_token_ceiling
            != worst.cost_ceiling.conservative_input_token_ceiling
            or self.per_call_max_output_tokens != worst.cost_ceiling.max_output_tokens
            or self.per_call_cost_ceiling_usd_micros != worst.request_cost_ceiling_usd_micros
            or self.row_input_token_ceiling
            != self.per_call_input_token_ceiling * self.authorized_slot_multiplicity
            or self.row_output_token_ceiling
            != self.per_call_max_output_tokens * self.authorized_slot_multiplicity
            or self.row_cost_ceiling_usd_micros
            != self.per_call_cost_ceiling_usd_micros * self.authorized_slot_multiplicity
        ):
            raise ValueError("metasyn_passage_hosted_v2_row_cost_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"row_cost_sha256"})
        if self.row_cost_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_hosted_v2_row_cost_hash_mismatch")
        return self


def _freeze_packet_row_costs(
    *,
    extraction_inputs: MetaSynExtractionInputsV2,
    config: MetaSynPassageHostedConfigV2,
    anthropic_config: AnthropicBoundedConfigV1,
) -> list[PacketRowCostEnvelopeV2]:
    rows: list[PacketRowCostEnvelopeV2] = []
    for row in extraction_inputs.rows:
        outcome_id, outcome_quote = _maximum_protocol_outcome(row)
        passage_ids = _maximum_passage_ids(row)
        probes: list[PacketEffectCostProbeV2] = []
        for effect_kind in EFFECT_KINDS:
            candidate = MetaSynPassageCandidateV2(
                candidate_index=1,
                canonical_outcome_id=outcome_id,
                outcome_concept_quote=outcome_quote,
                effect_kind=effect_kind,
                passage_ids=passage_ids,
            )
            inventory = MetaSynCandidateInventoryV2(
                inventory_status="candidates_found",
                candidates=[candidate],
                has_more_or_uncertain=False,
            )
            receipt = freeze_metasyn_candidate_inventory_receipt_v2(
                row_context_sha256=row.upstream_row_context_sha256,
                projection_v2_sha256=row.projection_v2_sha256,
                allowed_outcome_text_by_id=(row.question_surface.allowed_outcome_text_by_id),
                passage_text_by_id={
                    passage.passage_id: passage.text for passage in row.projection_surface.passages
                },
                value=inventory,
            )
            packet_input = freeze_metasyn_packet_candidate_input_v2(
                extraction_inputs=extraction_inputs,
                row_ordinal=row.row_ordinal,
                inventory_receipt=receipt,
                candidate_index=1,
            )
            accepted = capacity_limited_packet_schema_v2(
                schema_bundle=packet_input.packet_schema_bundle,
                config=config,
            )
            accepted_sha = hash_canonical(accepted)
            compiled = _freeze_compiled_schema_record(
                schema_kind="packet",
                effect_kind=effect_kind,
                context_binding_sha256=packet_input.candidate_binding_sha256,
                original_schema=accepted,
                full_acceptance_schema_sha256=accepted_sha,
            )
            ceiling = compute_anthropic_request_cost_ceiling(
                config=anthropic_config,
                system=config.system_prompt,
                prompt=packet_input.rendered_prompt,
                wire_schema=compiled.compiled_schema.wire_schema,
                max_output_tokens=config.packet_max_output_tokens,
                transport_mode="prompt_json_schema",
            )
            payload = {
                "probe_version": PACKET_COST_PROBE_VERSION,
                "effect_kind": effect_kind,
                "packet_input_sha256": packet_input.packet_input_sha256,
                "candidate_binding_sha256": packet_input.candidate_binding_sha256,
                "rendered_prompt_sha256": packet_input.rendered_prompt_sha256,
                "rendered_prompt_utf8_bytes": len(packet_input.rendered_prompt.encode("utf-8")),
                "capacity_limited_schema_sha256": accepted_sha,
                "compiled_schema_sha256": compiled.compiled_schema_sha256,
                "wire_schema_sha256": compiled.wire_schema_sha256,
                "cost_ceiling": ceiling,
                "cost_ceiling_sha256": ceiling.cost_ceiling_sha256,
                "request_cost_ceiling_usd_micros": _usd_micros(ceiling.request_cost_ceiling_usd),
            }
            probes.append(
                PacketEffectCostProbeV2.model_validate(
                    {**payload, "probe_sha256": hash_canonical(payload)}
                )
            )
        worst = max(
            probes,
            key=lambda item: (
                item.request_cost_ceiling_usd_micros,
                item.cost_ceiling.conservative_input_token_ceiling,
                -EFFECT_KINDS.index(item.effect_kind),
            ),
        )
        payload = {
            "row_cost_version": PACKET_ROW_COST_VERSION,
            "row_ordinal": row.row_ordinal,
            "row_key": row.row_key,
            "row_input_sha256": row.row_input_sha256,
            "synthetic_cost_probe_only": True,
            "selected_outcome_id": outcome_id,
            "selected_outcome_quote_sha256": hash_canonical(outcome_quote),
            "selected_passage_ids": passage_ids,
            "evaluated_effect_kinds": list(EFFECT_KINDS),
            "effect_probes": probes,
            "effect_probe_membership_sha256": hash_canonical(
                [item.probe_sha256 for item in probes]
            ),
            "worst_effect_kind": worst.effect_kind,
            "worst_probe_sha256": worst.probe_sha256,
            "per_call_input_token_ceiling": (worst.cost_ceiling.conservative_input_token_ceiling),
            "per_call_max_output_tokens": worst.cost_ceiling.max_output_tokens,
            "per_call_cost_ceiling_usd_micros": (worst.request_cost_ceiling_usd_micros),
            "authorized_slot_multiplicity": MAX_ACCEPTED_CANDIDATES_PER_ROW,
            "row_input_token_ceiling": (
                worst.cost_ceiling.conservative_input_token_ceiling
                * MAX_ACCEPTED_CANDIDATES_PER_ROW
            ),
            "row_output_token_ceiling": (
                worst.cost_ceiling.max_output_tokens * MAX_ACCEPTED_CANDIDATES_PER_ROW
            ),
            "row_cost_ceiling_usd_micros": (
                worst.request_cost_ceiling_usd_micros * MAX_ACCEPTED_CANDIDATES_PER_ROW
            ),
        }
        rows.append(
            PacketRowCostEnvelopeV2.model_validate(
                {**payload, "row_cost_sha256": hash_canonical(payload)}
            )
        )
    return rows


class CostGroupV2(_ExactContractModel):
    cost_group_version: Literal["metasyn-passage-cost-group-v2"] = COST_GROUP_VERSION
    group: Literal["source_free_preflight", "inventory", "packet"]
    maximum_calls: Annotated[int, Field(ge=1)]
    conservative_input_token_ceiling: Annotated[int, Field(ge=1)]
    max_output_token_ceiling: Annotated[int, Field(ge=1)]
    cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    request_or_probe_membership_sha256: str
    group_sha256: str

    @field_validator("request_or_probe_membership_sha256", "group_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_group(self) -> CostGroupV2:
        payload = self.model_dump(mode="json", exclude={"group_sha256"})
        if self.group_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_hosted_v2_cost_group_hash_mismatch")
        return self


class GlobalCostEnvelopeV2(_ExactContractModel):
    cost_envelope_version: Literal["metasyn-passage-global-cost-envelope-v2"] = (
        COST_ENVELOPE_VERSION
    )
    liability_policy: Literal[
        "every_authorized_call_or_ambiguous_attempt_charged_at_full_request_ceiling"
    ] = "every_authorized_call_or_ambiguous_attempt_charged_at_full_request_ceiling"
    source_free_preflight: CostGroupV2
    inventory: CostGroupV2
    packet: CostGroupV2
    maximum_provider_calls: Literal[296] = MAX_PROVIDER_CALLS
    conservative_input_token_ceiling: Annotated[int, Field(ge=1)]
    max_output_token_ceiling: Annotated[int, Field(ge=1)]
    cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    configured_maximum_input_tokens: Annotated[int, Field(ge=1)]
    configured_maximum_cost_usd_micros: Annotated[int, Field(ge=1)]
    within_configured_input_limit: Literal[True] = True
    within_configured_cost_limit: Literal[True] = True
    cost_envelope_sha256: str

    @field_validator("cost_envelope_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "cost_envelope_sha256")

    @model_validator(mode="after")
    def validate_envelope(self) -> GlobalCostEnvelopeV2:
        groups = [self.source_free_preflight, self.inventory, self.packet]
        if [item.group for item in groups] != [
            "source_free_preflight",
            "inventory",
            "packet",
        ]:
            raise ValueError("metasyn_passage_hosted_v2_cost_group_order_mismatch")
        if (
            sum(item.maximum_calls for item in groups) != self.maximum_provider_calls
            or sum(item.conservative_input_token_ceiling for item in groups)
            != self.conservative_input_token_ceiling
            or sum(item.max_output_token_ceiling for item in groups)
            != self.max_output_token_ceiling
            or sum(item.cost_ceiling_usd_micros for item in groups) != self.cost_ceiling_usd_micros
            or self.conservative_input_token_ceiling > self.configured_maximum_input_tokens
            or self.cost_ceiling_usd_micros > self.configured_maximum_cost_usd_micros
        ):
            raise ValueError("metasyn_passage_hosted_v2_global_cost_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"cost_envelope_sha256"})
        if self.cost_envelope_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_hosted_v2_global_cost_hash_mismatch")
        return self


def _freeze_cost_group(
    *,
    group: Literal["source_free_preflight", "inventory", "packet"],
    maximum_calls: int,
    input_tokens: int,
    output_tokens: int,
    cost_micros: int,
    membership: Any,
) -> CostGroupV2:
    payload = {
        "cost_group_version": COST_GROUP_VERSION,
        "group": group,
        "maximum_calls": maximum_calls,
        "conservative_input_token_ceiling": input_tokens,
        "max_output_token_ceiling": output_tokens,
        "cost_ceiling_usd_micros": cost_micros,
        "request_or_probe_membership_sha256": hash_canonical(membership),
    }
    return CostGroupV2.model_validate({**payload, "group_sha256": hash_canonical(payload)})


def _freeze_global_cost_envelope(
    *,
    config: MetaSynPassageHostedConfigV2,
    preflight: Sequence[SourceFreePreflightCallV2],
    inventory: Sequence[InventoryRequestV2],
    packet_rows: Sequence[PacketRowCostEnvelopeV2],
) -> GlobalCostEnvelopeV2:
    preflight_group = _freeze_cost_group(
        group="source_free_preflight",
        maximum_calls=len(preflight),
        input_tokens=sum(
            item.request.cost_ceiling.conservative_input_token_ceiling for item in preflight
        ),
        output_tokens=sum(item.request.max_output_tokens for item in preflight),
        cost_micros=sum(
            _usd_micros(item.request.cost_ceiling.request_cost_ceiling_usd) for item in preflight
        ),
        membership=[item.request_sha256 for item in preflight],
    )
    inventory_group = _freeze_cost_group(
        group="inventory",
        maximum_calls=len(inventory),
        input_tokens=sum(
            item.request.cost_ceiling.conservative_input_token_ceiling for item in inventory
        ),
        output_tokens=sum(item.request.max_output_tokens for item in inventory),
        cost_micros=sum(
            _usd_micros(item.request.cost_ceiling.request_cost_ceiling_usd) for item in inventory
        ),
        membership=[item.request_sha256 for item in inventory],
    )
    packet_group = _freeze_cost_group(
        group="packet",
        maximum_calls=sum(item.authorized_slot_multiplicity for item in packet_rows),
        input_tokens=sum(item.row_input_token_ceiling for item in packet_rows),
        output_tokens=sum(item.row_output_token_ceiling for item in packet_rows),
        cost_micros=sum(item.row_cost_ceiling_usd_micros for item in packet_rows),
        membership=[
            {
                "authorized_slot_multiplicity": item.authorized_slot_multiplicity,
                "row_cost_sha256": item.row_cost_sha256,
                "worst_probe_sha256": item.worst_probe_sha256,
            }
            for item in packet_rows
        ],
    )
    groups = [preflight_group, inventory_group, packet_group]
    total_input = sum(item.conservative_input_token_ceiling for item in groups)
    total_output = sum(item.max_output_token_ceiling for item in groups)
    total_cost = sum(item.cost_ceiling_usd_micros for item in groups)
    if total_input > config.maximum_input_tokens_all_calls:
        raise MetaSynPassageHostedBundleV2Error(
            "metasyn_passage_hosted_v2_input_envelope_exceeds_config"
        )
    if total_cost > config.maximum_authorized_cost_usd_micros:
        raise MetaSynPassageHostedBundleV2Error(
            "metasyn_passage_hosted_v2_cost_envelope_exceeds_config"
        )
    payload = {
        "cost_envelope_version": COST_ENVELOPE_VERSION,
        "liability_policy": (
            "every_authorized_call_or_ambiguous_attempt_charged_at_full_request_ceiling"
        ),
        "source_free_preflight": preflight_group,
        "inventory": inventory_group,
        "packet": packet_group,
        "maximum_provider_calls": MAX_PROVIDER_CALLS,
        "conservative_input_token_ceiling": total_input,
        "max_output_token_ceiling": total_output,
        "cost_ceiling_usd_micros": total_cost,
        "configured_maximum_input_tokens": config.maximum_input_tokens_all_calls,
        "configured_maximum_cost_usd_micros": (config.maximum_authorized_cost_usd_micros),
        "within_configured_input_limit": True,
        "within_configured_cost_limit": True,
    }
    return GlobalCostEnvelopeV2.model_validate(
        {**payload, "cost_envelope_sha256": hash_canonical(payload)}
    )


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


def _bundle_python_dependency_closure(repository_root: Path) -> list[str]:
    root = _canonical_root(repository_root)
    runtime_presence = [(root / relative).is_file() for relative in _RUNTIME_ENTRYPOINTS]
    if any(runtime_presence) and not all(runtime_presence):
        raise MetaSynPassageHostedBundleV2Error(
            "metasyn_passage_hosted_v2_runtime_entrypoint_set_incomplete"
        )
    pending = list(_BUNDLE_ENTRYPOINTS)
    if all(runtime_presence):
        pending.extend(_RUNTIME_ENTRYPOINTS)
    observed = {"src/literature_multiverse/__init__.py"}
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        path = _checked_repository_file(root=root, relative_path=relative)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise MetaSynPassageHostedBundleV2Error(
                f"metasyn_passage_hosted_v2_dependency_unreadable:{relative}"
            ) from exc
        observed.add(relative)
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom):
                modules.append((node.module or "", node.level))
            elif isinstance(node, ast.Import):
                modules.extend((alias.name, 0) for alias in node.names)
            for module, level in modules:
                dependency = _resolve_local_import(
                    repository_root=root,
                    current_path=relative,
                    module=module,
                    level=level,
                )
                if dependency is not None and dependency not in observed:
                    pending.append(dependency)
    return sorted(observed)


def compute_metasyn_passage_hosted_bundle_pipeline_fingerprint_v2(
    *,
    repository_root: Path,
    config_sha256: str,
    provider_identity_sha256: str,
    assembly_analysis_policy_sha256: str,
    extraction_inputs: MetaSynExtractionInputsV2,
    protocol_orientation_membership_sha256: str,
    inventory_request_membership_sha256: str,
    packet_compiler_gate_membership_sha256: str,
    preflight_membership_sha256: str,
    packet_row_cost_membership_sha256: str,
    cost_envelope_sha256: str,
) -> PipelineFingerprint:
    """Hash the AST-closed offline implementation and all non-Python inputs."""

    root = _canonical_root(repository_root)
    for name, value in {
        "config_sha256": config_sha256,
        "provider_identity_sha256": provider_identity_sha256,
        "assembly_analysis_policy_sha256": assembly_analysis_policy_sha256,
        "protocol_orientation_membership_sha256": (protocol_orientation_membership_sha256),
        "inventory_request_membership_sha256": inventory_request_membership_sha256,
        "packet_compiler_gate_membership_sha256": (packet_compiler_gate_membership_sha256),
        "preflight_membership_sha256": preflight_membership_sha256,
        "packet_row_cost_membership_sha256": packet_row_cost_membership_sha256,
        "cost_envelope_sha256": cost_envelope_sha256,
    }.items():
        _validate_sha256(value, name)
    files = sorted(set(_bundle_python_dependency_closure(root)) | set(_BUNDLE_NON_PYTHON_FILES))
    runtime_entrypoints_included = all(relative in files for relative in _RUNTIME_ENTRYPOINTS)
    component = PipelineComponentSpec(
        component_id="metasyn-passage-hosted-offline-bundle-v2",
        component_version=BUNDLE_COMPONENT_VERSION,
        file_paths=files,
        settings={
            "application_retries_per_request": 0,
            "assembly_analysis_policy_sha256": assembly_analysis_policy_sha256,
            "assembly_acceptance_boundary_dependency_closed": True,
            "config_sha256": config_sha256,
            "cost_envelope_sha256": cost_envelope_sha256,
            "extraction_inputs_pipeline_sha256": (
                extraction_inputs.extraction_inputs_pipeline_sha256
            ),
            "extraction_inputs_sha256": extraction_inputs.extraction_inputs_sha256,
            "hosted_runtime_entrypoints": (
                sorted(_RUNTIME_ENTRYPOINTS) if runtime_entrypoints_included else []
            ),
            "hosted_runtime_entrypoints_included": runtime_entrypoints_included,
            "installed_dependency_versions": {
                name: distribution_version(name) for name in _INSTALLED_DEPENDENCIES
            },
            "inventory_request_membership_sha256": (inventory_request_membership_sha256),
            "maximum_provider_calls": MAX_PROVIDER_CALLS,
            "official_test_labels_opened": False,
            "operator_authorized_source_transmission": True,
            "packet_compiler_gate_membership_sha256": (packet_compiler_gate_membership_sha256),
            "packet_row_cost_membership_sha256": (packet_row_cost_membership_sha256),
            "preflight_membership_sha256": preflight_membership_sha256,
            "protocol_orientation_membership_sha256": (protocol_orientation_membership_sha256),
            "provider_calls_permitted": False,
            "provider_identity_sha256": provider_identity_sha256,
            "reference_fields_unopened": True,
            "sdk_retries_per_request": 0,
            "source_surface_sha256": (extraction_inputs.upstream_source_surface_sha256),
            "yield_only_no_accuracy_or_release_authority": True,
        },
    )
    return compute_pipeline_fingerprint(root=root, components=[component])


class MetaSynPassageHostedExecutionBundleV2(_ExactContractModel):
    execution_bundle_version: Literal["metasyn-passage-hosted-execution-bundle-v2"] = (
        EXECUTION_BUNDLE_VERSION
    )
    status: Literal["frozen_offline_label_blind_execution_identity_no_provider_calls"] = (
        "frozen_offline_label_blind_execution_identity_no_provider_calls"
    )
    config_path: Literal["configs/benchmarks/metasyn-passage-hosted-anthropic-v2.json"] = (
        DEFAULT_CONFIG_PATH.as_posix()
    )
    config_file_sha256: str
    runtime_config: MetaSynPassageHostedConfigV2
    config_sha256: str
    anthropic_config: AnthropicBoundedConfigV1
    anthropic_config_sha256: str
    provider_identity: AnthropicProviderIdentityV1
    provider_identity_sha256: str
    assembly_analysis_policy: PacketAssemblyAnalysisPolicyV2
    assembly_analysis_policy_sha256: str
    extraction_inputs: MetaSynExtractionInputsV2
    extraction_inputs_sha256: str
    extraction_inputs_pipeline_sha256: str
    upstream_source_surface_sha256: str
    protocol_orientations: Annotated[
        list[RowProtocolOrientationV2], Field(min_length=32, max_length=32)
    ]
    protocol_orientation_membership_sha256: str
    inventory_requests: Annotated[list[InventoryRequestV2], Field(min_length=32, max_length=32)]
    inventory_request_membership_sha256: str
    packet_compiler_gates: Annotated[list[PacketCompilerGateV2], Field(min_length=5, max_length=5)]
    packet_compiler_gate_membership_sha256: str
    source_free_preflight_plan: Annotated[
        list[SourceFreePreflightCallV2], Field(min_length=8, max_length=8)
    ]
    preflight_membership_sha256: str
    packet_row_cost_envelopes: Annotated[
        list[PacketRowCostEnvelopeV2], Field(min_length=32, max_length=32)
    ]
    packet_row_cost_membership_sha256: str
    global_cost_envelope: GlobalCostEnvelopeV2
    cost_envelope_sha256: str
    bundle_pipeline_fingerprint: PipelineFingerprint
    bundle_pipeline_sha256: str
    question_count: Literal[10] = EXPECTED_QUESTION_COUNT
    component_count: Literal[10] = EXPECTED_COMPONENT_COUNT
    publication_count: Literal[32] = EXPECTED_PUBLICATION_COUNT
    maximum_provider_calls: Literal[296] = MAX_PROVIDER_CALLS
    provider_calls_made: Literal[False] = False
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    operator_authorized_source_transmission: Literal[True] = True
    permitted_metrics: Literal["inventory_grounding_assembly_and_typed_effect_yield_only"] = (
        "inventory_grounding_assembly_and_typed_effect_yield_only"
    )
    extraction_accuracy_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    execution_bundle_sha256: str

    @field_validator(
        "config_file_sha256",
        "config_sha256",
        "anthropic_config_sha256",
        "provider_identity_sha256",
        "assembly_analysis_policy_sha256",
        "extraction_inputs_sha256",
        "extraction_inputs_pipeline_sha256",
        "upstream_source_surface_sha256",
        "protocol_orientation_membership_sha256",
        "inventory_request_membership_sha256",
        "packet_compiler_gate_membership_sha256",
        "preflight_membership_sha256",
        "packet_row_cost_membership_sha256",
        "cost_envelope_sha256",
        "bundle_pipeline_sha256",
        "execution_bundle_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_bundle(self) -> MetaSynPassageHostedExecutionBundleV2:
        if (
            self.config_sha256 != self.runtime_config.config_sha256
            or self.anthropic_config != self.runtime_config.anthropic_config()
            or self.anthropic_config_sha256 != self.anthropic_config.config_sha256
            or self.provider_identity_sha256 != self.provider_identity.identity_sha256
            or self.provider_identity.config_sha256 != self.anthropic_config_sha256
            or self.assembly_analysis_policy != (self.runtime_config.assembly_analysis_policy())
            or self.assembly_analysis_policy_sha256
            != self.assembly_analysis_policy.analysis_policy_sha256
            or self.extraction_inputs_sha256 != self.extraction_inputs.extraction_inputs_sha256
            or self.extraction_inputs_pipeline_sha256
            != self.extraction_inputs.extraction_inputs_pipeline_sha256
            or self.upstream_source_surface_sha256
            != self.extraction_inputs.upstream_source_surface_sha256
            or self.cost_envelope_sha256 != self.global_cost_envelope.cost_envelope_sha256
            or self.bundle_pipeline_sha256 != self.bundle_pipeline_fingerprint.pipeline_sha256
        ):
            raise ValueError("metasyn_passage_hosted_v2_bundle_alias_mismatch")
        if [item.row_ordinal for item in self.inventory_requests] != list(range(32)):
            raise ValueError("metasyn_passage_hosted_v2_inventory_roster_mismatch")
        if [item.row_ordinal for item in self.protocol_orientations] != list(range(32)):
            raise ValueError("metasyn_passage_hosted_v2_protocol_orientation_roster_mismatch")
        if [item.effect_kind for item in self.packet_compiler_gates] != list(EFFECT_KINDS):
            raise ValueError("metasyn_passage_hosted_v2_packet_gate_roster_mismatch")
        if [item.preflight_ordinal for item in self.source_free_preflight_plan] != list(range(8)):
            raise ValueError("metasyn_passage_hosted_v2_preflight_roster_mismatch")
        if [item.row_ordinal for item in self.packet_row_cost_envelopes] != list(range(32)):
            raise ValueError("metasyn_passage_hosted_v2_packet_cost_roster_mismatch")
        memberships = {
            "protocol_orientation_membership_sha256": hash_canonical(
                [item.row_orientation_sha256 for item in self.protocol_orientations]
            ),
            "inventory_request_membership_sha256": hash_canonical(
                [item.inventory_request_sha256 for item in self.inventory_requests]
            ),
            "packet_compiler_gate_membership_sha256": hash_canonical(
                [item.gate_sha256 for item in self.packet_compiler_gates]
            ),
            "preflight_membership_sha256": hash_canonical(
                [item.preflight_call_sha256 for item in self.source_free_preflight_plan]
            ),
            "packet_row_cost_membership_sha256": hash_canonical(
                [item.row_cost_sha256 for item in self.packet_row_cost_envelopes]
            ),
        }
        if any(getattr(self, key) != expected for key, expected in memberships.items()):
            raise ValueError("metasyn_passage_hosted_v2_membership_hash_mismatch")
        for row, orientation in zip(
            self.extraction_inputs.rows,
            self.protocol_orientations,
            strict=True,
        ):
            if (
                orientation.row_key != row.row_key
                or orientation.row_input_sha256 != row.row_input_sha256
                or orientation.question_surface_sha256 != row.question_surface_sha256
                or orientation.question_surface_question_spec_sha256
                != row.upstream_question_spec_sha256
                or orientation.protocol_question_spec_sha256
                != row.projection_v2.lineage_binding.question_spec_sha256
            ):
                raise ValueError("metasyn_passage_hosted_v2_protocol_orientation_row_mismatch")
        if (
            self.question_count != self.extraction_inputs.question_count
            or self.component_count != self.extraction_inputs.component_count
            or self.publication_count != self.extraction_inputs.publication_count
            or self.maximum_provider_calls != self.global_cost_envelope.maximum_provider_calls
            or not self.runtime_config.operator_authorized_source_transmission
            or not self.runtime_config.reference_fields_unopened
            or self.runtime_config.official_test_labels_opened
            or not self.runtime_config.yield_only_no_accuracy_or_release_authority
        ):
            raise ValueError("metasyn_passage_hosted_v2_policy_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"execution_bundle_sha256"})
        if self.execution_bundle_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_hosted_v2_bundle_hash_mismatch")
        return self


def freeze_metasyn_passage_hosted_execution_bundle_v2(
    *,
    repository_root: Path,
    extraction_inputs: MetaSynExtractionInputsV2 | Mapping[str, Any] | None = None,
) -> MetaSynPassageHostedExecutionBundleV2:
    """Externally replay inputs and freeze a credential-free execution identity."""

    root = _canonical_root(repository_root)
    config, config_file_sha = load_metasyn_passage_hosted_config_v2(repository_root=root)
    if extraction_inputs is None:
        inputs = freeze_metasyn_extraction_inputs_v2(repository_root=root)
    else:
        inputs = validate_metasyn_extraction_inputs_v2(
            extraction_inputs=extraction_inputs,
            repository_root=root,
            external_replay=True,
        )
    anthropic_config = config.anthropic_config()
    identity = freeze_anthropic_provider_identity(anthropic_config)
    analysis_policy = config.assembly_analysis_policy()
    orientations = _freeze_row_protocol_orientations(inputs)
    inventory = _freeze_inventory_requests(
        extraction_inputs=inputs,
        config=config,
        anthropic_config=anthropic_config,
        identity=identity,
    )
    gates = _freeze_packet_compiler_gates(config)
    preflight = _freeze_source_free_preflight_plan(
        config=config,
        anthropic_config=anthropic_config,
        identity=identity,
        packet_gates=gates,
    )
    packet_costs = _freeze_packet_row_costs(
        extraction_inputs=inputs,
        config=config,
        anthropic_config=anthropic_config,
    )
    inventory_membership = hash_canonical([item.inventory_request_sha256 for item in inventory])
    orientation_membership = hash_canonical([item.row_orientation_sha256 for item in orientations])
    gate_membership = hash_canonical([item.gate_sha256 for item in gates])
    preflight_membership = hash_canonical([item.preflight_call_sha256 for item in preflight])
    packet_cost_membership = hash_canonical([item.row_cost_sha256 for item in packet_costs])
    cost_envelope = _freeze_global_cost_envelope(
        config=config,
        preflight=preflight,
        inventory=inventory,
        packet_rows=packet_costs,
    )
    fingerprint = compute_metasyn_passage_hosted_bundle_pipeline_fingerprint_v2(
        repository_root=root,
        config_sha256=config.config_sha256,
        provider_identity_sha256=identity.identity_sha256,
        assembly_analysis_policy_sha256=analysis_policy.analysis_policy_sha256,
        extraction_inputs=inputs,
        protocol_orientation_membership_sha256=orientation_membership,
        inventory_request_membership_sha256=inventory_membership,
        packet_compiler_gate_membership_sha256=gate_membership,
        preflight_membership_sha256=preflight_membership,
        packet_row_cost_membership_sha256=packet_cost_membership,
        cost_envelope_sha256=cost_envelope.cost_envelope_sha256,
    )
    payload = {
        "execution_bundle_version": EXECUTION_BUNDLE_VERSION,
        "status": "frozen_offline_label_blind_execution_identity_no_provider_calls",
        "config_path": DEFAULT_CONFIG_PATH.as_posix(),
        "config_file_sha256": config_file_sha,
        "runtime_config": config,
        "config_sha256": config.config_sha256,
        "anthropic_config": anthropic_config,
        "anthropic_config_sha256": anthropic_config.config_sha256,
        "provider_identity": identity,
        "provider_identity_sha256": identity.identity_sha256,
        "assembly_analysis_policy": analysis_policy,
        "assembly_analysis_policy_sha256": analysis_policy.analysis_policy_sha256,
        "extraction_inputs": inputs,
        "extraction_inputs_sha256": inputs.extraction_inputs_sha256,
        "extraction_inputs_pipeline_sha256": inputs.extraction_inputs_pipeline_sha256,
        "upstream_source_surface_sha256": inputs.upstream_source_surface_sha256,
        "protocol_orientations": orientations,
        "protocol_orientation_membership_sha256": orientation_membership,
        "inventory_requests": inventory,
        "inventory_request_membership_sha256": inventory_membership,
        "packet_compiler_gates": gates,
        "packet_compiler_gate_membership_sha256": gate_membership,
        "source_free_preflight_plan": preflight,
        "preflight_membership_sha256": preflight_membership,
        "packet_row_cost_envelopes": packet_costs,
        "packet_row_cost_membership_sha256": packet_cost_membership,
        "global_cost_envelope": cost_envelope,
        "cost_envelope_sha256": cost_envelope.cost_envelope_sha256,
        "bundle_pipeline_fingerprint": fingerprint,
        "bundle_pipeline_sha256": fingerprint.pipeline_sha256,
        "question_count": inputs.question_count,
        "component_count": inputs.component_count,
        "publication_count": inputs.publication_count,
        "maximum_provider_calls": MAX_PROVIDER_CALLS,
        "provider_calls_made": False,
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "operator_authorized_source_transmission": True,
        "permitted_metrics": "inventory_grounding_assembly_and_typed_effect_yield_only",
        "extraction_accuracy_authority": False,
        "scientific_effectiveness_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynPassageHostedExecutionBundleV2.model_validate(
        {**payload, "execution_bundle_sha256": hash_canonical(payload)}
    )


def validate_metasyn_passage_hosted_execution_bundle_v2(
    *,
    execution_bundle: MetaSynPassageHostedExecutionBundleV2 | Mapping[str, Any],
    repository_root: Path,
    external_replay: bool = True,
) -> MetaSynPassageHostedExecutionBundleV2:
    """Validate a snapshot and, by default, rebuild every dependency and ceiling."""

    try:
        canonical = MetaSynPassageHostedExecutionBundleV2.model_validate(
            execution_bundle.model_dump(mode="json")
            if isinstance(execution_bundle, MetaSynPassageHostedExecutionBundleV2)
            else execution_bundle
        )
    except ValueError as exc:
        raise MetaSynPassageHostedBundleV2Error(
            "metasyn_passage_hosted_v2_bundle_contract_invalid"
        ) from exc
    if external_replay:
        replayed = freeze_metasyn_passage_hosted_execution_bundle_v2(
            repository_root=repository_root,
            extraction_inputs=canonical.extraction_inputs,
        )
        if replayed != canonical:
            raise MetaSynPassageHostedBundleV2Error(
                "metasyn_passage_hosted_v2_bundle_external_replay_mismatch"
            )
    return canonical


__all__ = [
    "CONFIG_VERSION",
    "DEFAULT_CONFIG_PATH",
    "EFFECT_KINDS",
    "EXECUTION_BUNDLE_VERSION",
    "MAX_PROVIDER_CALLS",
    "CompiledSchemaRecordV2",
    "CostGroupV2",
    "GlobalCostEnvelopeV2",
    "InventoryRequestV2",
    "MetaSynPassageHostedBundleV2Error",
    "MetaSynPassageHostedConfigV2",
    "MetaSynPassageHostedExecutionBundleV2",
    "PacketCompilerGateV2",
    "PacketEffectCostProbeV2",
    "PacketRowCostEnvelopeV2",
    "RowProtocolOrientationV2",
    "SourceFreePreflightCallV2",
    "capacity_limited_packet_schema_v2",
    "compute_metasyn_passage_hosted_bundle_pipeline_fingerprint_v2",
    "freeze_metasyn_passage_hosted_execution_bundle_v2",
    "load_metasyn_passage_hosted_config_v2",
    "maximum_canonical_json_utf8_bytes_for_schema_v2",
    "validate_metasyn_passage_hosted_execution_bundle_v2",
]
