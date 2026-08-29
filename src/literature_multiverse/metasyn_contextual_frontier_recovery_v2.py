"""One-shot Fable 5 recovery for the row-17 500-mg/placebo estimand.

This additive runtime does not modify or retry either immutable frontier-v1
request.  It binds the failed v1 terminal as negative provenance and freezes one
materially different request in a fresh workspace.  The recovery response schema
has an exact object key for every required candidate field.  Trusted code converts
that object to the canonical contextual-grounding-v3 claim list.

The prompt discloses the prespecified estimand and source-visible canonical
nonnumeric tokens.  It deliberately does not disclose the treatment/control event
or total values; those four integers remain an extraction task.  A successful run
is only a source-visible grounding and typed-graph mechanics smoke.  It grants no
accuracy, synthesis-input, scientific, calibration, or release authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from jsonschema import validate as validate_json_schema
from jsonschema.exceptions import ValidationError
from jsonschema.validators import validator_for
from pydantic import ConfigDict, Field, model_validator

from literature_multiverse.anthropic_bounded_generation import (
    AnthropicCompiledSchemaV1,
    compile_anthropic_bounded_schema,
)
from literature_multiverse.contextual_numeric_grounding_v3 import (
    ContextualClaimV3,
    ContextualGroundedClaimV3,
    ContextualGroundedEffectV3,
    ContextualGroundingFeasibilityReceiptV3,
    ContextualNativeProjectionV3,
    ContextualPacketAbstentionV3,
    ContextualPacketCompletedV3,
    ContextualProviderContextV3,
    ContextualSourcePassageV3,
    _freeze_grounded_effect,
    _load_replayed_v2_bundle,
    _runtime_native_projection_from_fixture,
    _source_passage,
    freeze_contextual_grounding_offline_feasibility_suite_v3,
    ground_contextual_claim_v3,
)
from literature_multiverse.lineage import canonical_json_bytes, hash_canonical, sha256_file
from literature_multiverse.metasyn_contextual_frontier_runtime_v1 import (
    ANTHROPIC_API_VERSION,
    API_BASE_URL,
    COMPILER_SOURCE_PATH,
    CONTEXTUAL_SOURCE_PATH,
    EFFORT,
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MODEL,
    SERVICE_TIER,
    STRUCTURED_OUTPUTS_SOURCE_URL,
    MetaSynContextualFrontierConfigV1,
    MetaSynContextualFrontierRequestV1,
    _assert_secret_free,
    _freeze_cost,
    _safe_repository_file,
    _sha256_utf8,
    _wire_kwargs,
    freeze_metasyn_contextual_frontier_identity_v1,
    load_metasyn_contextual_frontier_config_v1,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.schemas import assert_closed_object_schema

RUNTIME_VERSION = "metasyn-contextual-frontier-recovery-v2"
CONFIG_VERSION = "metasyn-contextual-frontier-recovery-config-v2"
PLAN_VERSION = "metasyn-contextual-frontier-recovery-plan-v2"
REQUEST_VERSION = "metasyn-contextual-frontier-recovery-request-v2"
AUTHORIZATION_VERSION = "metasyn-contextual-frontier-recovery-authorization-v2"
INTENT_VERSION = "metasyn-contextual-frontier-recovery-intent-v2"
RECEIPT_VERSION = "metasyn-contextual-frontier-recovery-provider-receipt-v2"
VALIDATION_VERSION = "metasyn-contextual-frontier-recovery-validation-v2"
INCIDENT_VERSION = "metasyn-contextual-frontier-recovery-incident-v2"
TERMINAL_VERSION = "metasyn-contextual-frontier-recovery-terminal-v2"
WORKSPACE_VALIDATION_VERSION = "metasyn-contextual-frontier-recovery-workspace-validation-v2"
RESPONSE_VERSION = "metasyn-contextual-frontier-recovery-response-v2"

DEFAULT_CONFIG_PATH = Path("configs/benchmarks/metasyn-contextual-frontier-recovery-v2.json")
DEFAULT_WORKSPACE = Path("data/cache/metasyn/contextual-frontier-recovery-v2")
RUNTIME_SOURCE_PATH = Path("src/literature_multiverse/metasyn_contextual_frontier_recovery_v2.py")
V1_RUNTIME_SOURCE_PATH = Path("src/literature_multiverse/metasyn_contextual_frontier_runtime_v1.py")
V1_CONFIG_PATH = Path("configs/benchmarks/metasyn-contextual-frontier-runtime-v1.json")
V1_TERMINAL_PATH = Path("data/cache/metasyn/contextual-frontier-runtime-v1/02-terminal.json")

PRIMARY_WITNESS = "metasyn-row17-candidate3-binary-primary-endpoint"
REQUEST_KEY = "row17-candidate3-fable5-high-recovery-v2"
EXPLICIT_ESTIMAND = (
    "fedratinib 500-mg group versus placebo group for the primary end point "
    "spleen response at week 24"
)

EXPECTED_V1_TERMINAL_FILE_SHA256 = (
    "ea3bca6df39aa914d1fede51edbd5abbb39e1624bb6adcb5792946b48411f76d"
)

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_UNSIGNED_INTEGER_RE = re.compile(r"^(?:0|[1-9][0-9]{0,9})$")


class MetaSynContextualFrontierRecoveryV2Error(ValueError):
    """A recovery contract or exact-once transition failed closed."""


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


def _self_hash(model: _Frozen, field: str, code: str) -> None:
    if getattr(model, field) != hash_canonical(model.model_dump(mode="json", exclude={field})):
        raise ValueError(code)


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class RecoveryFieldContractV2(_Frozen):
    field_path: str
    normalization: Literal["verbatim_text", "unsigned_integer", "timepoint_integer"]
    token_policy: Literal["canonical_disclosed", "extract_exact_unsigned_integer_from_source"]
    canonical_token: str | None

    @model_validator(mode="after")
    def validate_contract(self) -> RecoveryFieldContractV2:
        if (self.token_policy == "canonical_disclosed") != (self.canonical_token is not None):
            raise ValueError("recovery_v2_field_token_policy_mismatch")
        return self


_FIELD_SPECS_RAW: list[tuple[str, str, str | None]] = [
    ("cohort.registry_id", "verbatim_text", "NCT01437787"),
    ("comparator_arm.label", "verbatim_text", "placebo group"),
    ("contrast.marker", "verbatim_text", "vs"),
    ("effect.control_events", "unsigned_integer", None),
    ("effect.control_total", "unsigned_integer", None),
    ("effect.treatment_events", "unsigned_integer", None),
    ("effect.treatment_total", "unsigned_integer", None),
    ("finding.endpoint_marker", "verbatim_text", "primary end point"),
    ("finding.outcome_name", "verbatim_text", "spleen response"),
    ("finding.timepoint.anchor", "verbatim_text", "week"),
    ("finding.timepoint.value", "timepoint_integer", "24"),
    ("study.design", "verbatim_text", "Randomized Clinical Trial"),
    ("study.registration_id", "verbatim_text", "NCT01437787"),
    (
        "study.source_label",
        "verbatim_text",
        "Safety and Efficacy of Fedratinib in Patients With Primary or Secondary "
        "Myelofibrosis: A Randomized Clinical Trial.",
    ),
    ("treatment_arm.label", "verbatim_text", "500-mg"),
]


def _field_contract() -> list[RecoveryFieldContractV2]:
    values = [
        RecoveryFieldContractV2(
            field_path=path,
            normalization=normalization,  # type: ignore[arg-type]
            token_policy=(
                "canonical_disclosed"
                if token is not None
                else "extract_exact_unsigned_integer_from_source"
            ),
            canonical_token=token,
        )
        for path, normalization, token in _FIELD_SPECS_RAW
    ]
    if [item.field_path for item in values] != sorted(item.field_path for item in values):
        raise MetaSynContextualFrontierRecoveryV2Error("recovery_v2_field_contract_not_canonical")
    return values


class MetaSynContextualFrontierRecoveryTargetSpecV2(_Frozen):
    target_version: Literal["metasyn-contextual-frontier-recovery-target-v2"] = (
        "metasyn-contextual-frontier-recovery-target-v2"
    )
    recovery_label: Literal["post_hoc_target_conditioned_recovery_smoke"] = (
        "post_hoc_target_conditioned_recovery_smoke"
    )
    estimand: Literal[
        "fedratinib 500-mg group versus placebo group for the primary end point "
        "spleen response at week 24"
    ] = EXPLICIT_ESTIMAND
    fields: list[RecoveryFieldContractV2]
    grounding_only_caller_supplied_field_paths: list[str]
    extraction_scored_field_paths: list[
        Literal[
            "effect.control_events",
            "effect.control_total",
            "effect.treatment_events",
            "effect.treatment_total",
        ]
    ]
    event_count_answers_disclosed_outside_source: Literal[False] = False
    built_from_protocol_and_public_source_constants_not_evaluator_targets: Literal[True] = True
    extraction_accuracy_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    target_sha256: Sha256

    @model_validator(mode="after")
    def validate_target(self) -> MetaSynContextualFrontierRecoveryTargetSpecV2:
        expected = _field_contract()
        extracted = [
            "effect.control_events",
            "effect.control_total",
            "effect.treatment_events",
            "effect.treatment_total",
        ]
        caller = [item.field_path for item in expected if item.canonical_token is not None]
        if (
            self.fields != expected
            or self.extraction_scored_field_paths != extracted
            or self.grounding_only_caller_supplied_field_paths != caller
        ):
            raise ValueError("recovery_v2_target_spec_mismatch")
        _self_hash(self, "target_sha256", "recovery_v2_target_spec_hash_mismatch")
        return self


def freeze_metasyn_contextual_frontier_recovery_target_spec_v2() -> (
    MetaSynContextualFrontierRecoveryTargetSpecV2
):
    fields = _field_contract()
    payload = {
        "target_version": "metasyn-contextual-frontier-recovery-target-v2",
        "recovery_label": "post_hoc_target_conditioned_recovery_smoke",
        "estimand": EXPLICIT_ESTIMAND,
        "fields": fields,
        "grounding_only_caller_supplied_field_paths": [
            item.field_path for item in fields if item.canonical_token is not None
        ],
        "extraction_scored_field_paths": [
            "effect.control_events",
            "effect.control_total",
            "effect.treatment_events",
            "effect.treatment_total",
        ],
        "event_count_answers_disclosed_outside_source": False,
        "built_from_protocol_and_public_source_constants_not_evaluator_targets": True,
        "extraction_accuracy_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierRecoveryTargetSpecV2.model_validate(
        {**payload, "target_sha256": hash_canonical(payload)}
    )


def _response_schema(
    *,
    provider_context: ContextualProviderContextV3,
    target_spec: MetaSynContextualFrontierRecoveryTargetSpecV2,
) -> dict[str, Any]:
    passage_ids = [item.passage_id for item in provider_context.passages]
    target_sha = target_spec.target_sha256
    field_properties: dict[str, Any] = {}
    for item in _field_contract():
        token_schema: dict[str, Any]
        if item.canonical_token is None:
            token_schema = {
                "type": "string",
                "pattern": r"^(?:0|[1-9][0-9]{0,9})$",
            }
        else:
            token_schema = {"const": item.canonical_token, "type": "string"}
        field_properties[item.field_path] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "passage_id": {"enum": passage_ids, "type": "string"},
                "support_quote": {"type": "string", "minLength": 1, "maxLength": 4096},
                "context": {"type": "string", "minLength": 1, "maxLength": 1024},
                "token": token_schema,
                "normalization": {"const": item.normalization, "type": "string"},
            },
            "required": [
                "passage_id",
                "support_quote",
                "context",
                "token",
                "normalization",
            ],
        }
    completed = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "response_version": {"const": RESPONSE_VERSION, "type": "string"},
            "status": {"const": "completed", "type": "string"},
            "target_contract_sha256": {"const": target_sha, "type": "string"},
            "claims_by_field": {
                "type": "object",
                "additionalProperties": False,
                "properties": field_properties,
                "required": sorted(field_properties),
            },
        },
        "required": [
            "response_version",
            "status",
            "target_contract_sha256",
            "claims_by_field",
        ],
    }
    unable = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "response_version": {"const": RESPONSE_VERSION, "type": "string"},
            "status": {"const": "unable_to_complete", "type": "string"},
            "target_contract_sha256": {"const": target_sha, "type": "string"},
            "reason": {
                "enum": [
                    "exact_context_not_unique",
                    "numeric_token_not_exact",
                    "identity_not_exact",
                    "endpoint_not_self_contained",
                    "other_grounding_failure",
                ]
            },
        },
        "required": ["response_version", "status", "target_contract_sha256", "reason"],
    }
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"urn:literature-multiverse:frontier-recovery-v2:{target_sha}",
        "oneOf": [completed, unable],
    }
    validator_for(schema).check_schema(schema)
    assert_closed_object_schema(schema)
    return schema


SYSTEM_PROMPT = (
    "You are a bounded scientific evidence grounding worker. Return exactly one JSON "
    "object accepted by the supplied recovery schema. Follow the prespecified estimand "
    "and exact field contract. Copy source text exactly; never infer missing event counts. "
    "Use unable_to_complete if any required grounding is absent."
)


def _render_prompt(
    *,
    provider_context: ContextualProviderContextV3,
    target_spec: MetaSynContextualFrontierRecoveryTargetSpecV2,
    response_schema_sha256: str,
) -> str:
    contract = {
        "estimand": EXPLICIT_ESTIMAND,
        "required_fields_in_canonical_order": [
            item.model_dump(mode="json") for item in _field_contract()
        ],
        "exact_field_set_only": True,
        "claims_are_keyed_by_field_path": True,
        "trusted_code_converts_keys_to_sorted_claim_list": True,
        "forbidden_extra_effect_fields": [
            "effect.estimate",
            "effect.format",
            "effect.ci_level",
            "effect.ci_lower",
            "effect.ci_upper",
            "effect.treatment_percentage",
            "effect.control_percentage",
        ],
        "event_count_answers_disclosed_outside_source": False,
    }
    passages = [
        {
            "passage_id": item.passage_id,
            "text": item.text,
            "text_sha256": item.text_sha256,
            "section_enums": item.section_enums,
            "passage_lineage_sha256": item.passage_lineage_sha256,
        }
        for item in provider_context.passages
    ]
    instructions = (
        "Contextual recovery contract v2.\n"
        "TARGET ESTIMAND: fedratinib 500-mg group versus placebo group for the primary "
        "end point spleen response at week 24. Do not select the 400-mg arm.\n"
        "Return every and only the required claims_by_field key. For each field copy an "
        "exact support_quote, a local context occurring exactly once in that quote, and a "
        "token occurring exactly once in that context. The four event/total tokens must be "
        "extracted from SOURCE_PASSAGES_JSON; they are intentionally absent from the target "
        "contract. Do not return percentage, confidence-interval, or direct-effect fields. "
        "Do not calculate offsets or normalize punctuation.\n"
        f"CANDIDATE_BINDING_SHA256={provider_context.candidate_binding_sha256}\n"
        f"TARGET_CONTRACT_SHA256={target_spec.target_sha256}\n"
        f"RESPONSE_SCHEMA_SHA256={response_schema_sha256}\n"
        f"FIELD_CONTRACT_JSON={_canonical_json(contract)}\n"
        f"SOURCE_PASSAGES_JSON={_canonical_json(passages)}"
    )
    return instructions


class MetaSynContextualFrontierRecoveryConfigV2(_Frozen):
    config_version: Literal["metasyn-contextual-frontier-recovery-config-v2"] = CONFIG_VERSION
    model: Literal["claude-fable-5"] = MODEL
    effort: Literal["high"] = EFFORT
    transport_mode: Literal["structured_json_schema"] = "structured_json_schema"
    service_tier: Literal["standard_only"] = SERVICE_TIER
    maximum_provider_calls: Literal[1] = 1
    request_key: Literal["row17-candidate3-fable5-high-recovery-v2"] = REQUEST_KEY
    target_witness: Literal["metasyn-row17-candidate3-binary-primary-endpoint"] = PRIMARY_WITNESS
    explicit_estimand: Literal[
        "fedratinib 500-mg group versus placebo group for the primary end point "
        "spleen response at week 24"
    ] = EXPLICIT_ESTIMAND
    sdk_retries_per_request: Literal[0] = 0
    application_retries_per_request: Literal[0] = 0
    orphan_or_ambiguous_attempt_retry_permitted: Literal[False] = False
    predecessor_v1_retry_permitted: Literal[False] = False
    predecessor_workspace_mutation_permitted: Literal[False] = False
    operator_authorized_source_transmission: Literal[True] = True
    hard_liability_policy: Literal["single_request_full_model_input_context_plus_max_output"] = (
        "single_request_full_model_input_context_plus_max_output"
    )
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False

    @property
    def config_sha256(self) -> str:
        return hash_canonical(self)


def load_metasyn_contextual_frontier_recovery_config_v2(
    *, repository_root: Path, config_path: Path = DEFAULT_CONFIG_PATH
) -> MetaSynContextualFrontierRecoveryConfigV2:
    path = _safe_repository_file(repository_root, config_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynContextualFrontierRecoveryV2Error("recovery_v2_config_invalid") from exc
    return MetaSynContextualFrontierRecoveryConfigV2.model_validate(raw)


class MetaSynContextualFrontierPredecessorV1Provenance(_Frozen):
    provenance_version: Literal["metasyn-contextual-frontier-v1-failure-provenance-v2"] = (
        "metasyn-contextual-frontier-v1-failure-provenance-v2"
    )
    terminal_path: Literal["data/cache/metasyn/contextual-frontier-runtime-v1/02-terminal.json"]
    terminal_file_sha256: Sha256
    binding_method: Literal["file_sha256_only_no_semantic_parse"] = (
        "file_sha256_only_no_semantic_parse"
    )
    predecessor_diagnosis: Literal[
        "target_ambiguity_and_validation_surface_mismatch_not_model_weakness"
    ] = "target_ambiguity_and_validation_surface_mismatch_not_model_weakness"
    exact_request_retry_permitted: Literal[False] = False
    predecessor_workspace_mutation_permitted: Literal[False] = False
    claim_release_authority: Literal[False] = False
    provenance_sha256: Sha256

    @model_validator(mode="after")
    def validate_provenance(self) -> MetaSynContextualFrontierPredecessorV1Provenance:
        if self.terminal_file_sha256 != EXPECTED_V1_TERMINAL_FILE_SHA256:
            raise ValueError("recovery_v2_predecessor_provenance_mismatch")
        _self_hash(self, "provenance_sha256", "recovery_v2_predecessor_hash_mismatch")
        return self


def _freeze_predecessor(
    *, repository_root: Path
) -> MetaSynContextualFrontierPredecessorV1Provenance:
    terminal_path = _safe_repository_file(repository_root, V1_TERMINAL_PATH)
    if sha256_file(terminal_path) != EXPECTED_V1_TERMINAL_FILE_SHA256:
        raise MetaSynContextualFrontierRecoveryV2Error("recovery_v2_v1_terminal_file_drift")
    payload = {
        "provenance_version": "metasyn-contextual-frontier-v1-failure-provenance-v2",
        "terminal_path": V1_TERMINAL_PATH.as_posix(),
        "terminal_file_sha256": EXPECTED_V1_TERMINAL_FILE_SHA256,
        "binding_method": "file_sha256_only_no_semantic_parse",
        "predecessor_diagnosis": (
            "target_ambiguity_and_validation_surface_mismatch_not_model_weakness"
        ),
        "exact_request_retry_permitted": False,
        "predecessor_workspace_mutation_permitted": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierPredecessorV1Provenance.model_validate(
        {**payload, "provenance_sha256": hash_canonical(payload)}
    )


class MetaSynContextualFrontierRecoveryRequestV2(_Frozen):
    request_version: Literal["metasyn-contextual-frontier-recovery-request-v2"] = REQUEST_VERSION
    request_key: Literal["row17-candidate3-fable5-high-recovery-v2"] = REQUEST_KEY
    witness_id: Literal["metasyn-row17-candidate3-binary-primary-endpoint"] = PRIMARY_WITNESS
    explicit_estimand: Literal[
        "fedratinib 500-mg group versus placebo group for the primary end point "
        "spleen response at week 24"
    ] = EXPLICIT_ESTIMAND
    recovery_label: Literal["post_hoc_target_conditioned_recovery_smoke"] = (
        "post_hoc_target_conditioned_recovery_smoke"
    )
    target_spec_sha256: Sha256
    required_field_paths: list[str]
    response_schema: dict[str, Any]
    response_schema_sha256: Sha256
    prompt: NonEmpty
    prompt_sha256: Sha256
    model_system: NonEmpty
    model_system_sha256: Sha256
    transport_request: MetaSynContextualFrontierRequestV1
    transport_request_sha256: Sha256
    predecessor_provenance_sha256: Sha256
    material_change_from_v1: Literal[
        "explicit_estimand_exact_keyed_field_contract_no_optional_binary_percent_or_ci_fields"
    ] = "explicit_estimand_exact_keyed_field_contract_no_optional_binary_percent_or_ci_fields"
    event_count_answers_disclosed_outside_source: Literal[False] = False
    exact_request_retry_permitted: Literal[False] = False
    claim_release_authority: Literal[False] = False
    request_sha256: Sha256

    @model_validator(mode="after")
    def validate_request(self) -> MetaSynContextualFrontierRecoveryRequestV2:
        expected_paths = [item.field_path for item in _field_contract()]
        if (
            self.required_field_paths != expected_paths
            or self.response_schema_sha256 != hash_canonical(self.response_schema)
            or self.prompt_sha256 != _sha_text(self.prompt)
            or self.model_system_sha256 != _sha_text(self.model_system)
            or self.transport_request_sha256 != self.transport_request.request_sha256
            or self.transport_request.prompt != self.prompt
            or self.transport_request.model_system != self.model_system
            or self.transport_request.original_schema_sha256 != self.response_schema_sha256
        ):
            raise ValueError("recovery_v2_request_replay_mismatch")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(self, "request_sha256", "recovery_v2_request_hash_mismatch")
        return self


def _freeze_transport_request(
    *,
    provider_context: ContextualProviderContextV3,
    response_schema: Mapping[str, Any],
    prompt: str,
    transport_config: MetaSynContextualFrontierConfigV1,
) -> MetaSynContextualFrontierRequestV1:
    identity = freeze_metasyn_contextual_frontier_identity_v1(transport_config)
    schema = json.loads(canonical_json_bytes(response_schema))
    compiled: AnthropicCompiledSchemaV1 = compile_anthropic_bounded_schema(
        original_schema=schema,
        full_acceptance_schema_sha256=hash_canonical(schema),
    )
    kwargs = _wire_kwargs(
        model_system=SYSTEM_PROMPT,
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
    payload = {
        "request_version": "metasyn-contextual-frontier-request-v1",
        "operation": "messages.create.contextual_grounding_v3",
        "request_key": REQUEST_KEY,
        "witness_id": PRIMARY_WITNESS,
        "provider_binding_sha256": provider_context.context_sha256,
        "model": MODEL,
        "effort": EFFORT,
        "service_tier": SERVICE_TIER,
        "transport_mode": "structured_json_schema",
        "output_config_format_present": True,
        "structured_output_model_supported": True,
        "structured_output_schema_sdk_transformed": True,
        "structured_outputs_source_url": STRUCTURED_OUTPUTS_SOURCE_URL,
        "config_sha256": transport_config.config_sha256,
        "identity_sha256": identity.identity_sha256,
        "original_schema_sha256": compiled.original_schema_sha256,
        "compiled_schema": compiled,
        "compiled_schema_sha256": compiled.compiled_schema_sha256,
        "wire_schema_sha256": compiled.wire_schema_sha256,
        "base_system": SYSTEM_PROMPT,
        "base_system_sha256": _sha256_utf8(SYSTEM_PROMPT),
        "model_system": SYSTEM_PROMPT,
        "model_system_sha256": _sha256_utf8(SYSTEM_PROMPT),
        "prompt": prompt,
        "prompt_sha256": _sha256_utf8(prompt),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "wire_kwargs": kwargs,
        "wire_kwargs_sha256": hash_canonical(kwargs),
        "wire_call_surface_sha256": surface_sha,
        "cost_ceiling": _freeze_cost(
            model_system=SYSTEM_PROMPT,
            prompt=prompt,
            wire_schema=compiled.wire_schema,
        ),
    }
    return MetaSynContextualFrontierRequestV1.model_validate(
        {**payload, "request_sha256": hash_canonical(payload)}
    )


def freeze_metasyn_contextual_frontier_recovery_request_v2(
    *,
    provider_context: ContextualProviderContextV3,
    target_spec: MetaSynContextualFrontierRecoveryTargetSpecV2,
    predecessor: MetaSynContextualFrontierPredecessorV1Provenance,
    transport_config: MetaSynContextualFrontierConfigV1,
) -> MetaSynContextualFrontierRecoveryRequestV2:
    expected_paths = [item.field_path for item in _field_contract()]
    if target_spec.fields != _field_contract():
        raise MetaSynContextualFrontierRecoveryV2Error("recovery_v2_target_spec_drift")
    schema = _response_schema(
        provider_context=provider_context,
        target_spec=target_spec,
    )
    schema_sha = hash_canonical(schema)
    prompt = _render_prompt(
        provider_context=provider_context,
        target_spec=target_spec,
        response_schema_sha256=schema_sha,
    )
    transport = _freeze_transport_request(
        provider_context=provider_context,
        response_schema=schema,
        prompt=prompt,
        transport_config=transport_config,
    )
    payload = {
        "request_version": REQUEST_VERSION,
        "request_key": REQUEST_KEY,
        "witness_id": PRIMARY_WITNESS,
        "explicit_estimand": EXPLICIT_ESTIMAND,
        "recovery_label": "post_hoc_target_conditioned_recovery_smoke",
        "target_spec_sha256": target_spec.target_sha256,
        "required_field_paths": expected_paths,
        "response_schema": schema,
        "response_schema_sha256": schema_sha,
        "prompt": prompt,
        "prompt_sha256": _sha_text(prompt),
        "model_system": SYSTEM_PROMPT,
        "model_system_sha256": _sha_text(SYSTEM_PROMPT),
        "transport_request": transport,
        "transport_request_sha256": transport.request_sha256,
        "predecessor_provenance_sha256": predecessor.provenance_sha256,
        "material_change_from_v1": (
            "explicit_estimand_exact_keyed_field_contract_no_optional_binary_percent_or_ci_fields"
        ),
        "event_count_answers_disclosed_outside_source": False,
        "exact_request_retry_permitted": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierRecoveryRequestV2.model_validate(
        {**payload, "request_sha256": hash_canonical(payload)}
    )


class MetaSynContextualFrontierRecoveryPlanV2(_Frozen):
    plan_version: Literal["metasyn-contextual-frontier-recovery-plan-v2"] = PLAN_VERSION
    runtime_version: Literal["metasyn-contextual-frontier-recovery-v2"] = RUNTIME_VERSION
    status: Literal["offline_prepared_zero_provider_calls"] = "offline_prepared_zero_provider_calls"
    recovery_label: Literal["post_hoc_target_conditioned_recovery_smoke"] = (
        "post_hoc_target_conditioned_recovery_smoke"
    )
    config: MetaSynContextualFrontierRecoveryConfigV2
    config_sha256: Sha256
    transport_profile_config: MetaSynContextualFrontierConfigV1
    transport_profile_config_sha256: Sha256
    predecessor: MetaSynContextualFrontierPredecessorV1Provenance
    predecessor_provenance_sha256: Sha256
    target_spec: MetaSynContextualFrontierRecoveryTargetSpecV2
    target_spec_sha256: Sha256
    provider_context: ContextualProviderContextV3
    provider_context_sha256: Sha256
    evaluator_fixture: ContextualGroundingFeasibilityReceiptV3
    evaluator_fixture_sha256: Sha256
    evaluator_fixture_never_passed_to_request_builder: Literal[True] = True
    evaluator_numeric_targets_model_facing: Literal[False] = False
    provider_visible_source_passages: Annotated[
        list[ContextualSourcePassageV3], Field(min_length=1, max_length=64)
    ]
    provider_visible_passage_membership_sha256: Sha256
    validates_any_provider_visible_citation_not_exact_passage_roster: Literal[True] = True
    request: MetaSynContextualFrontierRecoveryRequestV2
    request_sha256: Sha256
    runtime_pipeline_components: dict[str, str]
    runtime_pipeline_sha256: Sha256
    hard_cost_liability_usd_micros: Annotated[int, Field(ge=1)]
    diagnostic_known_input_token_ceiling: Annotated[int, Field(ge=1)]
    diagnostic_known_surface_cost_usd_micros: Annotated[int, Field(ge=1)]
    provider_calls_made: Literal[0] = 0
    maximum_provider_calls: Literal[1] = 1
    predecessor_requests_retried: Literal[0] = 0
    caller_supplied_fields_excluded_from_extraction_scoring: Literal[True] = True
    graph_construction_mechanics_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    reliability_authority: Literal[False] = False
    generalization_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    plan_sha256: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> MetaSynContextualFrontierRecoveryPlanV2:
        passage_ids = [item.passage_id for item in self.provider_visible_source_passages]
        provider_ids = [item.passage_id for item in self.provider_context.passages]
        cost = self.request.transport_request.cost_ceiling
        if (
            self.config_sha256 != self.config.config_sha256
            or self.transport_profile_config_sha256 != self.transport_profile_config.config_sha256
            or self.predecessor_provenance_sha256 != self.predecessor.provenance_sha256
            or self.target_spec_sha256 != self.target_spec.target_sha256
            or self.provider_context_sha256 != self.provider_context.context_sha256
            or self.evaluator_fixture_sha256 != self.evaluator_fixture.receipt_sha256
            or self.evaluator_fixture.provider_binding.context != self.provider_context
            or self.request_sha256 != self.request.request_sha256
            or self.request.target_spec_sha256 != self.target_spec_sha256
            or self.request.predecessor_provenance_sha256 != self.predecessor_provenance_sha256
            or passage_ids != sorted(set(passage_ids))
            or passage_ids != provider_ids
            or self.provider_visible_passage_membership_sha256
            != hash_canonical(
                [item.passage_sha256 for item in self.provider_visible_source_passages]
            )
            or self.runtime_pipeline_sha256 != hash_canonical(self.runtime_pipeline_components)
            or self.hard_cost_liability_usd_micros != cost.request_cost_ceiling_usd_micros
            or self.diagnostic_known_input_token_ceiling
            != cost.diagnostic_known_input_token_ceiling
            or self.diagnostic_known_surface_cost_usd_micros
            != cost.diagnostic_known_surface_cost_usd_micros
        ):
            raise ValueError("recovery_v2_plan_replay_mismatch")
        if (
            self.hard_cost_liability_usd_micros != 11_600_000
            or cost.model_max_input_tokens != MAX_INPUT_TOKENS
            or cost.max_output_tokens != MAX_OUTPUT_TOKENS
        ):
            raise ValueError("recovery_v2_hard_liability_mismatch")
        # The request builder never receives evaluator material.  As an additional
        # audit, the four evaluator answers may appear in the public source suffix,
        # but not as target/schema constants or instructions.
        prefix = self.request.prompt.split("SOURCE_PASSAGES_JSON=", 1)[0]
        schema_text = _canonical_json(self.request.response_schema)
        for field_path in self.target_spec.extraction_scored_field_paths:
            expected = self.evaluator_fixture.semantic_target.expected_normalized_values[field_path]
            marker = _canonical_json(expected)
            if marker in prefix or f'"const":{marker}' in schema_text:
                raise ValueError("recovery_v2_numeric_evaluator_target_leaked")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(self, "plan_sha256", "recovery_v2_plan_hash_mismatch")
        return self


def freeze_metasyn_contextual_frontier_recovery_plan_v2(
    *,
    repository_root: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> MetaSynContextualFrontierRecoveryPlanV2:
    root = Path(os.path.abspath(repository_root)).resolve(strict=True)
    config = load_metasyn_contextual_frontier_recovery_config_v2(
        repository_root=root, config_path=config_path
    )
    predecessor = _freeze_predecessor(repository_root=root)
    target_spec = freeze_metasyn_contextual_frontier_recovery_target_spec_v2()

    # Evaluator construction is deliberately separate from request construction.
    suite = freeze_contextual_grounding_offline_feasibility_suite_v3(repository_root=root)
    matches = [item for item in suite.receipts if item.witness_id == PRIMARY_WITNESS]
    if len(matches) != 1:
        raise MetaSynContextualFrontierRecoveryV2Error(
            "recovery_v2_primary_evaluator_fixture_missing"
        )
    evaluator_fixture = matches[0]
    provider_context = evaluator_fixture.provider_binding.context

    transport_config = load_metasyn_contextual_frontier_config_v1(
        repository_root=root, config_path=V1_CONFIG_PATH
    )
    request = freeze_metasyn_contextual_frontier_recovery_request_v2(
        provider_context=provider_context,
        target_spec=target_spec,
        predecessor=predecessor,
        transport_config=transport_config,
    )

    # Reconstruct every provider-visible passage from the externally replayed row.
    # Evaluation permits any of these citations; it requires the exact field roster,
    # not the small passage subset used by the old offline fixture.
    bundle = _load_replayed_v2_bundle(root=root)
    row = bundle.extraction_inputs.rows[provider_context.row_ordinal]
    if row.row_input_sha256 != provider_context.row_input_sha256:
        raise MetaSynContextualFrontierRecoveryV2Error("recovery_v2_source_row_drift")
    source_passages = [
        _source_passage(row=row, passage_id=item.passage_id) for item in provider_context.passages
    ]
    source_passages.sort(key=lambda item: item.passage_id)

    pipeline_components = {
        "runtime_source_sha256": sha256_file(_safe_repository_file(root, RUNTIME_SOURCE_PATH)),
        "v1_transport_runtime_source_sha256": sha256_file(
            _safe_repository_file(root, V1_RUNTIME_SOURCE_PATH)
        ),
        "contextual_grounding_source_sha256": sha256_file(
            _safe_repository_file(root, CONTEXTUAL_SOURCE_PATH)
        ),
        "schema_compiler_source_sha256": sha256_file(
            _safe_repository_file(root, COMPILER_SOURCE_PATH)
        ),
        "config_sha256": config.config_sha256,
        "transport_profile_config_sha256": transport_config.config_sha256,
        "predecessor_terminal_file_sha256": predecessor.terminal_file_sha256,
        "target_spec_sha256": target_spec.target_sha256,
        "provider_context_sha256": provider_context.context_sha256,
        "provider_visible_passage_membership_sha256": hash_canonical(
            [item.passage_sha256 for item in source_passages]
        ),
        "response_schema_sha256": request.response_schema_sha256,
        "request_sha256": request.request_sha256,
    }
    cost = request.transport_request.cost_ceiling
    payload = {
        "plan_version": PLAN_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "status": "offline_prepared_zero_provider_calls",
        "recovery_label": "post_hoc_target_conditioned_recovery_smoke",
        "config": config,
        "config_sha256": config.config_sha256,
        "transport_profile_config": transport_config,
        "transport_profile_config_sha256": transport_config.config_sha256,
        "predecessor": predecessor,
        "predecessor_provenance_sha256": predecessor.provenance_sha256,
        "target_spec": target_spec,
        "target_spec_sha256": target_spec.target_sha256,
        "provider_context": provider_context,
        "provider_context_sha256": provider_context.context_sha256,
        "evaluator_fixture": evaluator_fixture,
        "evaluator_fixture_sha256": evaluator_fixture.receipt_sha256,
        "evaluator_fixture_never_passed_to_request_builder": True,
        "evaluator_numeric_targets_model_facing": False,
        "provider_visible_source_passages": source_passages,
        "provider_visible_passage_membership_sha256": hash_canonical(
            [item.passage_sha256 for item in source_passages]
        ),
        "validates_any_provider_visible_citation_not_exact_passage_roster": True,
        "request": request,
        "request_sha256": request.request_sha256,
        "runtime_pipeline_components": pipeline_components,
        "runtime_pipeline_sha256": hash_canonical(pipeline_components),
        "hard_cost_liability_usd_micros": cost.request_cost_ceiling_usd_micros,
        "diagnostic_known_input_token_ceiling": (cost.diagnostic_known_input_token_ceiling),
        "diagnostic_known_surface_cost_usd_micros": (cost.diagnostic_known_surface_cost_usd_micros),
        "provider_calls_made": 0,
        "maximum_provider_calls": 1,
        "predecessor_requests_retried": 0,
        "caller_supplied_fields_excluded_from_extraction_scoring": True,
        "graph_construction_mechanics_authority": False,
        "extraction_accuracy_authority": False,
        "reliability_authority": False,
        "generalization_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierRecoveryPlanV2.model_validate(
        {**payload, "plan_sha256": hash_canonical(payload)}
    )


RecoveryResponse = ContextualPacketCompletedV3 | ContextualPacketAbstentionV3


def convert_metasyn_contextual_frontier_recovery_response_v2(
    *, plan: MetaSynContextualFrontierRecoveryPlanV2, raw: Mapping[str, Any]
) -> RecoveryResponse:
    value = dict(raw)
    try:
        validate_json_schema(value, plan.request.response_schema)
    except ValidationError as exc:
        raise MetaSynContextualFrontierRecoveryV2Error(
            "recovery_v2_response_schema_invalid"
        ) from exc
    if value.get("target_contract_sha256") != plan.target_spec_sha256:
        raise MetaSynContextualFrontierRecoveryV2Error(
            "recovery_v2_response_target_binding_mismatch"
        )
    if value.get("status") == "unable_to_complete":
        return ContextualPacketAbstentionV3(
            candidate_binding_sha256=plan.provider_context.candidate_binding_sha256,
            reason=value["reason"],
        )
    claims_by_field = value.get("claims_by_field")
    if not isinstance(claims_by_field, dict):
        raise MetaSynContextualFrontierRecoveryV2Error("recovery_v2_claim_object_missing")
    claims: list[ContextualClaimV3] = []
    for field_path in plan.request.required_field_paths:
        item = claims_by_field.get(field_path)
        if not isinstance(item, dict):
            raise MetaSynContextualFrontierRecoveryV2Error("recovery_v2_claim_body_invalid")
        claims.append(ContextualClaimV3(field_path=field_path, **item))  # type: ignore[arg-type]
    claims.sort(key=lambda item: (item.field_path, item.passage_id))
    marker = next(item for item in claims if item.field_path == "finding.endpoint_marker")
    scope = (
        "full_text_sections"
        if plan.provider_context.source_content_scope == "full_text_sections"
        else "title_abstract_not_release_grade"
    )
    return ContextualPacketCompletedV3(
        candidate_binding_sha256=plan.provider_context.candidate_binding_sha256,
        canonical_outcome_id=plan.provider_context.candidate.canonical_outcome_id,
        effect_kind="binary_group_statistics",
        endpoint_passage_id=marker.passage_id,
        endpoint_quote=marker.support_quote,
        effect_format_token=None,
        effect_computation="binary_group_statistics_to_odds_ratio_via_existing_harmonizer",
        source_scope_acknowledgement=scope,
        claims=claims,
    )


def _evaluate_completed_response(
    *,
    plan: MetaSynContextualFrontierRecoveryPlanV2,
    outcome: ContextualPacketCompletedV3,
    provider_execution_binding_sha256: str,
) -> tuple[
    list[ContextualGroundedClaimV3],
    ContextualGroundedEffectV3,
    str,
    ContextualNativeProjectionV3,
    dict[str, str],
]:
    provider_map = {item.passage_id: item for item in plan.provider_context.passages}
    passage_map = {item.passage_id: item for item in plan.provider_visible_source_passages}
    if set(provider_map) != set(passage_map):
        raise MetaSynContextualFrontierRecoveryV2Error(
            "recovery_v2_provider_visible_passage_set_drift"
        )
    for passage_id, passage in passage_map.items():
        provider = provider_map[passage_id]
        if (
            passage.passage_text != provider.text
            or passage.passage_text_sha256 != provider.text_sha256
            or passage.passage_lineage_sha256 != provider.passage_lineage_sha256
        ):
            raise MetaSynContextualFrontierRecoveryV2Error(
                "recovery_v2_provider_visible_passage_bytes_drift"
            )
    if [item.field_path for item in outcome.claims] != plan.request.required_field_paths:
        raise MetaSynContextualFrontierRecoveryV2Error("recovery_v2_exact_field_roster_mismatch")
    groundings = [
        ground_contextual_claim_v3(claim=claim, passage=passage_map[claim.passage_id])
        for claim in outcome.claims
    ]
    groundings.sort(key=lambda item: item.claim.field_path)
    effect = _freeze_grounded_effect(
        outcome=outcome,
        groundings=groundings,
        passages=passage_map,
    )
    observed = {item.claim.field_path: item.normalized_value for item in groundings}
    expected = plan.evaluator_fixture.semantic_target.expected_normalized_values
    if observed != expected:
        raise MetaSynContextualFrontierRecoveryV2Error(
            "recovery_v2_evaluator_semantic_target_mismatch"
        )
    extracted = {
        field_path: observed[field_path]
        for field_path in plan.target_spec.extraction_scored_field_paths
    }
    core = {
        "core_version": "metasyn-contextual-frontier-recovery-grounding-core-v2",
        "provider_context_sha256": plan.provider_context_sha256,
        "evaluator_semantic_target_sha256": plan.evaluator_fixture.semantic_target_sha256,
        "model_outcome_sha256": hash_canonical(outcome.model_dump(mode="json")),
        "provider_visible_passage_membership_sha256": (
            plan.provider_visible_passage_membership_sha256
        ),
        "grounding_membership_sha256": hash_canonical(
            [item.grounding_sha256 for item in groundings]
        ),
        "grounded_effect_sha256": effect.effect_sha256,
        "exact_field_roster_not_exact_passage_roster": True,
        "caller_supplied_fields_excluded_from_extraction_scoring": True,
    }
    core_sha = hash_canonical(core)
    projection = _runtime_native_projection_from_fixture(
        fixture_receipt=plan.evaluator_fixture,
        effect=effect,
        groundings=groundings,
        grounding_core_sha256=core_sha,
        runtime_pipeline_sha256=plan.runtime_pipeline_sha256,
        provider_execution_binding_sha256=provider_execution_binding_sha256,
    )
    return groundings, effect, core_sha, projection, extracted


class MetaSynContextualFrontierRecoveryCoreEvaluationV2(_Frozen):
    evaluation_version: Literal["metasyn-contextual-frontier-recovery-core-evaluation-v2"] = (
        "metasyn-contextual-frontier-recovery-core-evaluation-v2"
    )
    recovery_label: Literal["post_hoc_target_conditioned_recovery_smoke"] = (
        "post_hoc_target_conditioned_recovery_smoke"
    )
    status: Literal["typed_graph_mechanics_completed", "scientific_abstention"]
    plan_sha256: Sha256
    runtime_pipeline_sha256: Sha256
    provider_execution_binding_sha256: Sha256
    response: RecoveryResponse
    response_sha256: Sha256
    groundings: list[ContextualGroundedClaimV3] | None
    grounding_membership_sha256: Sha256 | None
    grounded_effect: ContextualGroundedEffectV3 | None
    grounded_effect_sha256: Sha256 | None
    contextual_grounding_core_sha256: Sha256 | None
    native_projection: ContextualNativeProjectionV3 | None
    native_projection_sha256: Sha256 | None
    extracted_numeric_values: dict[str, str] | None
    numeric_extraction_fields_evaluated: Literal[0, 4]
    numeric_evaluator_exact_match: bool | None
    provider_visible_passages_eligible_for_citation: Annotated[int, Field(ge=1)]
    exact_field_roster_not_exact_passage_roster: Literal[True] = True
    caller_supplied_fields_excluded_from_extraction_scoring: Literal[True] = True
    typed_graph_mechanics_observed: bool
    graph_construction_mechanics_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    reliability_authority: Literal[False] = False
    generalization_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    evaluation_sha256: Sha256

    @model_validator(mode="after")
    def validate_evaluation(self) -> MetaSynContextualFrontierRecoveryCoreEvaluationV2:
        if self.response_sha256 != hash_canonical(self.response.model_dump(mode="json")):
            raise ValueError("recovery_v2_core_response_hash_mismatch")
        aliases = (
            (self.groundings is None) == (self.grounding_membership_sha256 is None)
            and (self.grounded_effect is None) == (self.grounded_effect_sha256 is None)
            and (self.native_projection is None) == (self.native_projection_sha256 is None)
        )
        if not aliases:
            raise ValueError("recovery_v2_core_evaluation_presence_mismatch")
        if self.groundings is not None and self.grounding_membership_sha256 != hash_canonical(
            [item.grounding_sha256 for item in self.groundings]
        ):
            raise ValueError("recovery_v2_core_grounding_membership_mismatch")
        if self.grounded_effect is not None and (
            self.grounded_effect_sha256 != self.grounded_effect.effect_sha256
        ):
            raise ValueError("recovery_v2_core_effect_alias_mismatch")
        if self.native_projection is not None and (
            self.native_projection_sha256 != self.native_projection.projection_sha256
        ):
            raise ValueError("recovery_v2_core_projection_alias_mismatch")
        success = self.status == "typed_graph_mechanics_completed"
        if (
            success != self.typed_graph_mechanics_observed
            or success != (self.numeric_evaluator_exact_match is True)
            or success != (self.numeric_extraction_fields_evaluated == 4)
            or success != (self.extracted_numeric_values is not None)
            or success != (self.native_projection is not None)
        ):
            raise ValueError("recovery_v2_core_evaluation_status_mismatch")
        _self_hash(self, "evaluation_sha256", "recovery_v2_core_evaluation_hash_mismatch")
        return self


def evaluate_metasyn_contextual_frontier_recovery_response_v2(
    *,
    plan: MetaSynContextualFrontierRecoveryPlanV2,
    raw_response: Mapping[str, Any],
    provider_execution_binding_sha256: str,
) -> MetaSynContextualFrontierRecoveryCoreEvaluationV2:
    """Convert and evaluate one result without granting empirical authority."""

    response = convert_metasyn_contextual_frontier_recovery_response_v2(plan=plan, raw=raw_response)
    common = {
        "evaluation_version": "metasyn-contextual-frontier-recovery-core-evaluation-v2",
        "recovery_label": "post_hoc_target_conditioned_recovery_smoke",
        "plan_sha256": plan.plan_sha256,
        "runtime_pipeline_sha256": plan.runtime_pipeline_sha256,
        "provider_execution_binding_sha256": provider_execution_binding_sha256,
        "response": response,
        "response_sha256": hash_canonical(response.model_dump(mode="json")),
        "provider_visible_passages_eligible_for_citation": len(
            plan.provider_visible_source_passages
        ),
        "exact_field_roster_not_exact_passage_roster": True,
        "caller_supplied_fields_excluded_from_extraction_scoring": True,
        "extraction_accuracy_authority": False,
        "reliability_authority": False,
        "generalization_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    if isinstance(response, ContextualPacketAbstentionV3):
        payload = {
            **common,
            "status": "scientific_abstention",
            "groundings": None,
            "grounding_membership_sha256": None,
            "grounded_effect": None,
            "grounded_effect_sha256": None,
            "contextual_grounding_core_sha256": None,
            "native_projection": None,
            "native_projection_sha256": None,
            "extracted_numeric_values": None,
            "numeric_extraction_fields_evaluated": 0,
            "numeric_evaluator_exact_match": None,
            "typed_graph_mechanics_observed": False,
            "graph_construction_mechanics_authority": False,
        }
    else:
        groundings, effect, core_sha, projection, extracted = _evaluate_completed_response(
            plan=plan,
            outcome=response,
            provider_execution_binding_sha256=provider_execution_binding_sha256,
        )
        payload = {
            **common,
            "status": "typed_graph_mechanics_completed",
            "groundings": groundings,
            "grounding_membership_sha256": hash_canonical(
                [item.grounding_sha256 for item in groundings]
            ),
            "grounded_effect": effect,
            "grounded_effect_sha256": effect.effect_sha256,
            "contextual_grounding_core_sha256": core_sha,
            "native_projection": projection,
            "native_projection_sha256": projection.projection_sha256,
            "extracted_numeric_values": extracted,
            "numeric_extraction_fields_evaluated": 4,
            "numeric_evaluator_exact_match": True,
            "typed_graph_mechanics_observed": True,
            "graph_construction_mechanics_authority": False,
        }
    return MetaSynContextualFrontierRecoveryCoreEvaluationV2.model_validate(
        {**payload, "evaluation_sha256": hash_canonical(payload)}
    )
