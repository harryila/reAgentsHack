"""Completed-only one-shot recovery after the immutable recovery-v3 failure.

Recovery-v3 returned a hybrid of the completed and abstention branches.  This
post-hoc schema-compatibility smoke removes the abstention branch from the model-
facing grammar while retaining the provider-accepted shared ``claims[]`` item
topology.  Trusted code still requires the exact fifteen-field semantic target.
The run is one call, zero retry, zero fallback, and grants no empirical,
scientific, calibration, synthesis, or release authority.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from jsonschema import validate as validate_json_schema
from jsonschema.exceptions import ValidationError
from jsonschema.validators import validator_for
from pydantic import ConfigDict, Field, model_validator

from literature_multiverse.anthropic_bounded_generation import (
    AnthropicCompiledSchemaV1,
    compile_anthropic_bounded_schema,
)
from literature_multiverse.contextual_numeric_grounding_v3 import (
    ContextualGroundedClaimV3,
    ContextualGroundedEffectV3,
    ContextualNativeProjectionV3,
    ContextualPacketCompletedV3,
    ContextualProviderContextV3,
    _runtime_native_projection_from_fixture,
)
from literature_multiverse.lineage import canonical_json_bytes, hash_canonical, sha256_file
from literature_multiverse.metasyn_contextual_frontier_recovery_v2 import (
    MetaSynContextualFrontierRecoveryPlanV2,
    MetaSynContextualFrontierRecoveryTargetSpecV2,
)
from literature_multiverse.metasyn_contextual_frontier_recovery_v3 import (
    MetaSynContextualFrontierRecoveryCoreEvaluationV3,
    MetaSynContextualFrontierRecoveryPlanV3,
    _load_v1_plan,
    _load_v2_plan,
    evaluate_metasyn_contextual_frontier_recovery_response_v3,
)
from literature_multiverse.metasyn_contextual_frontier_runtime_v1 import (
    ANTHROPIC_API_VERSION,
    API_BASE_URL,
    MetaSynContextualFrontierClientV1,
    MetaSynContextualFrontierConfigV1,
    MetaSynContextualFrontierProviderResultV1,
    MetaSynContextualFrontierRequestV1,
    _assert_secret_free,
    _checked_artifact,
    _existing_workspace,
    _freeze_cost,
    _fresh_workspace,
    _load_object,
    _persist_json,
    _safe_exception_type,
    _safe_request_id,
    _safe_status,
    _sha256_utf8,
    _wire_kwargs,
    _workspace_lock,
)
from literature_multiverse.models import SHA256_RE, ContractModel

RUNTIME_VERSION = "metasyn-contextual-frontier-recovery-v4"
PLAN_VERSION = "metasyn-contextual-frontier-recovery-plan-v4"
CONFIG_VERSION = "metasyn-contextual-frontier-recovery-config-v4"
AUTHORIZATION_VERSION = "metasyn-contextual-frontier-recovery-authorization-v4"
INTENT_VERSION = "metasyn-contextual-frontier-recovery-intent-v4"
RECEIPT_VERSION = "metasyn-contextual-frontier-recovery-receipt-v4"
TERMINAL_VERSION = "metasyn-contextual-frontier-recovery-terminal-v4"
RECOVERY_LABEL = "post_hoc_completed_only_grammar_recovery"
REQUEST_KEY = "row17-candidate3-fable5-high-recovery-v4"
PRIMARY_WITNESS = "metasyn-row17-candidate3-binary-primary-endpoint"

DEFAULT_CONFIG_PATH = Path("configs/benchmarks/metasyn-contextual-frontier-recovery-v4.json")
DEFAULT_WORKSPACE = Path("data/cache/metasyn/contextual-frontier-recovery-v4")
RUNTIME_SOURCE_PATH = Path("src/literature_multiverse/metasyn_contextual_frontier_recovery_v4.py")
V3_PREPARED_PATH = Path("data/cache/metasyn/contextual-frontier-recovery-v3/00-prepared.json")
V3_TERMINAL_PATH = Path("data/cache/metasyn/contextual-frontier-recovery-v3/02-terminal.json")

EXPECTED_V3_REQUEST_SHA256 = "12ddf6a928df1d1089b0d1f8a4c374ea70a55de53ecc6c31a85b1b4518d0dd9a"
EXPECTED_V3_TERMINAL_SHA256 = "4c6b8f686ef6b67370be7f4501d4aeb72c0ea061bcb8c78db68ce0e569eee220"
EXPECTED_V3_TERMINAL_FILE_SHA256 = (
    "52615dbd4d42279768bcc920ade7cd7f38c09dc7b23bb31ab4c65e12bf6e445e"
)


class MetaSynContextualFrontierRecoveryV4Error(ValueError):
    """A recovery-v4 contract or exact-once transition failed closed."""


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


def _root(value: Path) -> Path:
    return value.resolve(strict=True)


def _read_object(root: Path, relative: Path) -> dict[str, Any]:
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise MetaSynContextualFrontierRecoveryV4Error("recovery_v4_path_escape") from exc
    value = json.loads(path.read_text(encoding="utf-8"))
    if path.is_symlink() or not path.is_file() or not isinstance(value, dict):
        raise MetaSynContextualFrontierRecoveryV4Error("recovery_v4_artifact_invalid")
    return value


def _load_v3_plan(root: Path) -> MetaSynContextualFrontierRecoveryPlanV3:
    return MetaSynContextualFrontierRecoveryPlanV3.model_validate(
        _read_object(root, V3_PREPARED_PATH)
    )


class MetaSynContextualFrontierRecoveryConfigV4(_Frozen):
    config_version: Literal["metasyn-contextual-frontier-recovery-config-v4"] = CONFIG_VERSION
    model: Literal["claude-fable-5"] = "claude-fable-5"
    effort: Literal["high"] = "high"
    request_key: Literal["row17-candidate3-fable5-high-recovery-v4"] = REQUEST_KEY
    maximum_provider_calls: Literal[1] = 1
    application_retries_per_request: Literal[0] = 0
    sdk_retries_per_request: Literal[0] = 0
    fallback_requests_permitted: Literal[0] = 0
    predecessor_requests_retry_permitted: Literal[False] = False
    provider_grammar: Literal["completed_branch_only_shared_claims_array"] = (
        "completed_branch_only_shared_claims_array"
    )
    operator_authorized_source_transmission: Literal[True] = True
    post_hoc_completed_only_grammar_recovery: Literal[True] = True
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False

    @property
    def config_sha256(self) -> str:
        return hash_canonical(self)


def load_metasyn_contextual_frontier_recovery_config_v4(
    *, repository_root: Path, config_path: Path = DEFAULT_CONFIG_PATH
) -> MetaSynContextualFrontierRecoveryConfigV4:
    return MetaSynContextualFrontierRecoveryConfigV4.model_validate(
        _read_object(_root(repository_root), config_path)
    )


def _completed_only_schema(
    accepted_request: MetaSynContextualFrontierRequestV1,
) -> dict[str, Any]:
    full = accepted_request.compiled_schema.original_schema
    branches = full.get("oneOf")
    if not isinstance(branches, list) or len(branches) != 2:
        raise MetaSynContextualFrontierRecoveryV4Error("recovery_v4_v1_union_shape_drift")
    completed = deepcopy(branches[0])
    properties = completed.get("properties", {})
    if (
        properties.get("packet_status", {}).get("const") != "completed"
        or "claims" not in properties
        or "reason" in properties
        or any(key in completed for key in ("oneOf", "anyOf", "allOf"))
    ):
        raise MetaSynContextualFrontierRecoveryV4Error("recovery_v4_completed_branch_shape_invalid")
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:literature-multiverse:contextual-frontier-recovery-v4:completed-only",
        **completed,
    }
    validator_for(schema).check_schema(schema)
    return schema


def _schema_complexity(value: Any) -> tuple[int, int, int]:
    properties = enums = unions = 0
    if isinstance(value, dict):
        props = value.get("properties")
        if isinstance(props, dict):
            properties += len(props)
        enum = value.get("enum")
        if isinstance(enum, list):
            enums += len(enum)
        unions += sum(key in value for key in ("oneOf", "anyOf", "allOf"))
        for child in value.values():
            p, e, u = _schema_complexity(child)
            properties += p
            enums += e
            unions += u
    elif isinstance(value, list):
        for child in value:
            p, e, u = _schema_complexity(child)
            properties += p
            enums += e
            unions += u
    return properties, enums, unions


def _render_prompt(
    *,
    provider_context: ContextualProviderContextV3,
    target_spec: MetaSynContextualFrontierRecoveryTargetSpecV2,
) -> str:
    fields = [
        {
            "field_path": item.field_path,
            "normalization": item.normalization,
            "required_token": (
                item.canonical_token
                if item.canonical_token is not None
                else "EXTRACT_EXACT_UNSIGNED_INTEGER_FROM_SOURCE"
            ),
        }
        for item in target_spec.fields
    ]
    passages = [item.model_dump(mode="json") for item in provider_context.passages]
    encoded_fields = json.dumps(fields, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    encoded_passages = json.dumps(
        passages, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return (
        "Post-hoc completed-only grammar recovery v4. Return only the completed object "
        "required by the supplied schema; there is no abstention branch and no reason field.\n"
        "TARGET: fedratinib 500-mg group versus placebo group for the primary end point "
        "spleen response at week 24. Do not select 400-mg or symptom response. Return exactly "
        "the fifteen FIELD_ROSTER_JSON claims once each and no extra field_path. Copy canonical "
        "nonnumeric tokens exactly. Extract the four event/total unsigned integers only from "
        "SOURCE_PASSAGES_JSON; those answers are absent outside the passages. Copy exact quote, "
        "unique local context, and unique token. Trusted code sorts and checks the complete "
        "semantic target.\n"
        f"CANDIDATE_BINDING_SHA256={provider_context.candidate_binding_sha256}\n"
        f"TARGET_CONTRACT_SHA256={target_spec.target_sha256}\n"
        f"FIELD_ROSTER_JSON={encoded_fields}\n"
        f"SOURCE_PASSAGES_JSON={encoded_passages}"
    )


def freeze_metasyn_contextual_frontier_recovery_request_v4(
    *,
    provider_context: ContextualProviderContextV3,
    target_spec: MetaSynContextualFrontierRecoveryTargetSpecV2,
    accepted_v1_request: MetaSynContextualFrontierRequestV1,
    transport_config: MetaSynContextualFrontierConfigV1,
) -> MetaSynContextualFrontierRequestV1:
    accepted = MetaSynContextualFrontierRequestV1.model_validate(
        accepted_v1_request.model_dump(mode="json")
    )
    if accepted.config_sha256 != transport_config.config_sha256:
        raise MetaSynContextualFrontierRecoveryV4Error("recovery_v4_transport_config_drift")
    original = _completed_only_schema(accepted)
    compiled: AnthropicCompiledSchemaV1 = compile_anthropic_bounded_schema(
        original_schema=original,
        full_acceptance_schema_sha256=hash_canonical(original),
    )
    prompt = _render_prompt(provider_context=provider_context, target_spec=target_spec)
    kwargs = _wire_kwargs(
        model_system=accepted.base_system,
        prompt=prompt,
        wire_schema=compiled.wire_schema,
    )
    surface = hash_canonical(
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
    payload = accepted.model_dump(mode="json", exclude={"request_sha256"})
    payload.update(
        {
            "request_key": REQUEST_KEY,
            "provider_binding_sha256": hash_canonical(
                {
                    "recovery_label": RECOVERY_LABEL,
                    "context_sha256": provider_context.context_sha256,
                    "target_sha256": target_spec.target_sha256,
                }
            ),
            "original_schema_sha256": compiled.original_schema_sha256,
            "compiled_schema": compiled,
            "compiled_schema_sha256": compiled.compiled_schema_sha256,
            "wire_schema_sha256": compiled.wire_schema_sha256,
            "prompt": prompt,
            "prompt_sha256": _sha256_utf8(prompt),
            "wire_kwargs": kwargs,
            "wire_kwargs_sha256": hash_canonical(kwargs),
            "wire_call_surface_sha256": surface,
            "cost_ceiling": _freeze_cost(
                model_system=accepted.base_system,
                prompt=prompt,
                wire_schema=compiled.wire_schema,
            ),
        }
    )
    return MetaSynContextualFrontierRequestV1.model_validate(
        {**payload, "request_sha256": hash_canonical(payload)}
    )


class MetaSynContextualFrontierRecoveryPlanV4(_Frozen):
    plan_version: Literal["metasyn-contextual-frontier-recovery-plan-v4"] = PLAN_VERSION
    runtime_version: Literal["metasyn-contextual-frontier-recovery-v4"] = RUNTIME_VERSION
    status: Literal["offline_prepared_zero_provider_calls"] = "offline_prepared_zero_provider_calls"
    recovery_label: Literal["post_hoc_completed_only_grammar_recovery"] = RECOVERY_LABEL
    config: MetaSynContextualFrontierRecoveryConfigV4
    config_sha256: Sha256
    transport_config: MetaSynContextualFrontierConfigV1
    target_spec: MetaSynContextualFrontierRecoveryTargetSpecV2
    provider_context_sha256: Sha256
    provider_visible_passage_count: Literal[16] = 16
    immutable_v3_plan_sha256: Sha256
    immutable_v3_request_sha256: Sha256
    immutable_v3_terminal_sha256: Sha256
    immutable_v3_terminal_file_sha256: Sha256
    v3_failure_diagnosis: Literal[
        "completed_abstention_hybrid_empty_candidate_binding_response_schema_invalid"
    ] = "completed_abstention_hybrid_empty_candidate_binding_response_schema_invalid"
    request: MetaSynContextualFrontierRequestV1
    request_sha256: Sha256
    original_schema_sha256: Sha256
    compiled_schema_sha256: Sha256
    wire_schema_sha256: Sha256
    wire_schema_utf8_bytes: Annotated[int, Field(ge=1)]
    wire_schema_property_slots: Annotated[int, Field(ge=1)]
    wire_schema_enum_values: Annotated[int, Field(ge=1)]
    wire_schema_union_keywords: Literal[0] = 0
    compiler_confirmed: Literal[True] = True
    completed_branch_only: Literal[True] = True
    shared_claims_array_item_topology: Literal[True] = True
    runtime_pipeline_components: dict[str, str]
    runtime_pipeline_sha256: Sha256
    hard_cost_liability_usd_micros: Annotated[int, Field(ge=1)]
    provider_calls_made: Literal[0] = 0
    maximum_provider_calls: Literal[1] = 1
    predecessor_requests_retried: Literal[0] = 0
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
    def validate_plan(self) -> MetaSynContextualFrontierRecoveryPlanV4:
        props, enums, unions = _schema_complexity(self.request.compiled_schema.wire_schema)
        if (
            self.config_sha256 != self.config.config_sha256
            or self.request_sha256 != self.request.request_sha256
            or self.original_schema_sha256 != self.request.original_schema_sha256
            or self.compiled_schema_sha256 != self.request.compiled_schema_sha256
            or self.wire_schema_sha256 != self.request.wire_schema_sha256
            or self.wire_schema_utf8_bytes
            != len(canonical_json_bytes(self.request.compiled_schema.wire_schema))
            or (self.wire_schema_property_slots, self.wire_schema_enum_values, 0)
            != (props, enums, unions)
            or self.runtime_pipeline_sha256 != hash_canonical(self.runtime_pipeline_components)
            or self.hard_cost_liability_usd_micros
            != self.request.cost_ceiling.request_cost_ceiling_usd_micros
            or "reason" in self.request.compiled_schema.original_schema.get("properties", {})
        ):
            raise ValueError("recovery_v4_plan_replay_mismatch")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(self, "plan_sha256", "recovery_v4_plan_hash_mismatch")
        return self


def freeze_metasyn_contextual_frontier_recovery_plan_v4(
    *, repository_root: Path, config_path: Path = DEFAULT_CONFIG_PATH
) -> MetaSynContextualFrontierRecoveryPlanV4:
    root = _root(repository_root)
    config = load_metasyn_contextual_frontier_recovery_config_v4(
        repository_root=root, config_path=config_path
    )
    v1 = _load_v1_plan(root)
    v2 = _load_v2_plan(root)
    v3 = _load_v3_plan(root)
    terminal_raw = _read_object(root, V3_TERMINAL_PATH)
    if (
        sha256_file(root / V3_TERMINAL_PATH) != EXPECTED_V3_TERMINAL_FILE_SHA256
        or terminal_raw.get("terminal_sha256") != EXPECTED_V3_TERMINAL_SHA256
        or terminal_raw.get("status") != "provider_result_failed"
        or terminal_raw.get("failure_code") != "response_schema_invalid"
        or terminal_raw.get("request_sha256") != EXPECTED_V3_REQUEST_SHA256
    ):
        raise MetaSynContextualFrontierRecoveryV4Error("recovery_v4_v3_terminal_drift")
    parsed = terminal_raw.get("provider_receipt", {}).get("provider_result", {}).get("parsed_json")
    if parsed != {
        "candidate_binding_sha256": "",
        "outcome_version": "contextual-packet-model-outcome-v3",
        "packet_status": "completed",
        "reason": "other_grounding_failure",
    }:
        raise MetaSynContextualFrontierRecoveryV4Error("recovery_v4_v3_diagnosis_drift")
    primary = [item.request for item in v1.roster if item.witness_id == PRIMARY_WITNESS]
    if len(primary) != 1:
        raise MetaSynContextualFrontierRecoveryV4Error("recovery_v4_v1_primary_missing")
    request = freeze_metasyn_contextual_frontier_recovery_request_v4(
        provider_context=v2.provider_context,
        target_spec=v2.target_spec,
        accepted_v1_request=primary[0],
        transport_config=v3.transport_config,
    )
    old_requests = {item.request_sha256 for item in v1.roster}
    old_requests.update({v2.request.transport_request_sha256, v3.request_sha256})
    if request.request_sha256 in old_requests:
        raise MetaSynContextualFrontierRecoveryV4Error("recovery_v4_predecessor_retry")
    props, enums, unions = _schema_complexity(request.compiled_schema.wire_schema)
    components = {
        "runtime_source_sha256": sha256_file(root / RUNTIME_SOURCE_PATH),
        "config_sha256": config.config_sha256,
        "provider_context_sha256": v2.provider_context_sha256,
        "target_spec_sha256": v2.target_spec_sha256,
        "v3_plan_sha256": v3.plan_sha256,
        "v3_terminal_sha256": EXPECTED_V3_TERMINAL_SHA256,
        "v3_terminal_file_sha256": EXPECTED_V3_TERMINAL_FILE_SHA256,
        "request_sha256": request.request_sha256,
        "original_schema_sha256": request.original_schema_sha256,
        "compiled_schema_sha256": request.compiled_schema_sha256,
        "wire_schema_sha256": request.wire_schema_sha256,
    }
    payload = {
        "plan_version": PLAN_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "status": "offline_prepared_zero_provider_calls",
        "recovery_label": RECOVERY_LABEL,
        "config": config,
        "config_sha256": config.config_sha256,
        "transport_config": v3.transport_config,
        "target_spec": v2.target_spec,
        "provider_context_sha256": v2.provider_context_sha256,
        "provider_visible_passage_count": len(v2.provider_context.passages),
        "immutable_v3_plan_sha256": v3.plan_sha256,
        "immutable_v3_request_sha256": v3.request_sha256,
        "immutable_v3_terminal_sha256": EXPECTED_V3_TERMINAL_SHA256,
        "immutable_v3_terminal_file_sha256": EXPECTED_V3_TERMINAL_FILE_SHA256,
        "v3_failure_diagnosis": (
            "completed_abstention_hybrid_empty_candidate_binding_response_schema_invalid"
        ),
        "request": request,
        "request_sha256": request.request_sha256,
        "original_schema_sha256": request.original_schema_sha256,
        "compiled_schema_sha256": request.compiled_schema_sha256,
        "wire_schema_sha256": request.wire_schema_sha256,
        "wire_schema_utf8_bytes": len(canonical_json_bytes(request.compiled_schema.wire_schema)),
        "wire_schema_property_slots": props,
        "wire_schema_enum_values": enums,
        "wire_schema_union_keywords": unions,
        "compiler_confirmed": True,
        "completed_branch_only": True,
        "shared_claims_array_item_topology": True,
        "runtime_pipeline_components": components,
        "runtime_pipeline_sha256": hash_canonical(components),
        "hard_cost_liability_usd_micros": request.cost_ceiling.request_cost_ceiling_usd_micros,
        "provider_calls_made": 0,
        "maximum_provider_calls": 1,
        "predecessor_requests_retried": 0,
        "graph_construction_mechanics_authority": False,
        "extraction_accuracy_authority": False,
        "reliability_authority": False,
        "generalization_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierRecoveryPlanV4.model_validate(
        {**payload, "plan_sha256": hash_canonical(payload)}
    )


class MetaSynContextualFrontierRecoveryCoreEvaluationV4(_Frozen):
    evaluation_version: Literal["metasyn-contextual-frontier-recovery-core-evaluation-v4"] = (
        "metasyn-contextual-frontier-recovery-core-evaluation-v4"
    )
    recovery_label: Literal["post_hoc_completed_only_grammar_recovery"] = RECOVERY_LABEL
    status: Literal["typed_graph_mechanics_completed"] = "typed_graph_mechanics_completed"
    plan_sha256: Sha256
    runtime_pipeline_sha256: Sha256
    provider_execution_binding_sha256: Sha256
    v3_evaluator_dependency_sha256: Sha256
    v3_evaluator_plan_sha256: Sha256
    v3_evaluator_runtime_pipeline_sha256: Sha256
    response: ContextualPacketCompletedV3
    response_sha256: Sha256
    groundings: list[ContextualGroundedClaimV3]
    grounding_membership_sha256: Sha256
    grounded_effect: ContextualGroundedEffectV3
    grounded_effect_sha256: Sha256
    contextual_grounding_core_sha256: Sha256
    native_projection: ContextualNativeProjectionV3
    native_projection_sha256: Sha256
    extracted_numeric_values: dict[str, str]
    numeric_extraction_fields_evaluated: Literal[4] = 4
    numeric_evaluator_exact_match: Literal[True] = True
    typed_graph_mechanics_observed: Literal[True] = True
    graph_construction_mechanics_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    reliability_authority: Literal[False] = False
    generalization_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    evaluation_sha256: Sha256

    @model_validator(mode="after")
    def validate_evaluation(self) -> MetaSynContextualFrontierRecoveryCoreEvaluationV4:
        if (
            self.response_sha256 != hash_canonical(self.response.model_dump(mode="json"))
            or self.grounding_membership_sha256
            != hash_canonical([item.grounding_sha256 for item in self.groundings])
            or self.grounded_effect_sha256 != self.grounded_effect.effect_sha256
            or self.native_projection_sha256 != self.native_projection.projection_sha256
            or self.native_projection.runtime_pipeline_sha256 != self.runtime_pipeline_sha256
            or self.native_projection.provider_execution_binding_sha256
            != self.provider_execution_binding_sha256
            or self.native_projection.fragment is None
            or self.native_projection.fragment.pipeline_fingerprint_sha256
            != self.runtime_pipeline_sha256
        ):
            raise ValueError("recovery_v4_evaluation_replay_mismatch")
        _self_hash(self, "evaluation_sha256", "recovery_v4_evaluation_hash_mismatch")
        return self


def evaluate_metasyn_contextual_frontier_recovery_response_v4(
    *,
    repository_root: Path,
    plan: MetaSynContextualFrontierRecoveryPlanV4,
    raw_response: Mapping[str, Any],
    provider_execution_binding_sha256: str,
) -> MetaSynContextualFrontierRecoveryCoreEvaluationV4:
    root = _root(repository_root)
    try:
        validate_json_schema(raw_response, plan.request.compiled_schema.original_schema)
    except ValidationError as exc:
        raise MetaSynContextualFrontierRecoveryV4Error(
            "recovery_v4_completed_only_schema_invalid"
        ) from exc
    v2: MetaSynContextualFrontierRecoveryPlanV2 = _load_v2_plan(root)
    v3 = _load_v3_plan(root)
    dependency: MetaSynContextualFrontierRecoveryCoreEvaluationV3 = (
        evaluate_metasyn_contextual_frontier_recovery_response_v3(
            repository_root=root,
            plan=v3,
            raw_response=raw_response,
            provider_execution_binding_sha256=provider_execution_binding_sha256,
        )
    )
    if (
        dependency.status != "typed_graph_mechanics_completed"
        or dependency.groundings is None
        or dependency.grounded_effect is None
        or dependency.contextual_grounding_core_sha256 is None
        or dependency.extracted_numeric_values is None
        or not isinstance(dependency.response, ContextualPacketCompletedV3)
    ):
        raise MetaSynContextualFrontierRecoveryV4Error("recovery_v4_v3_evaluator_not_completed")
    projection = _runtime_native_projection_from_fixture(
        fixture_receipt=v2.evaluator_fixture,
        effect=dependency.grounded_effect,
        groundings=dependency.groundings,
        grounding_core_sha256=dependency.contextual_grounding_core_sha256,
        runtime_pipeline_sha256=plan.runtime_pipeline_sha256,
        provider_execution_binding_sha256=provider_execution_binding_sha256,
    )
    payload = {
        "evaluation_version": "metasyn-contextual-frontier-recovery-core-evaluation-v4",
        "recovery_label": RECOVERY_LABEL,
        "status": "typed_graph_mechanics_completed",
        "plan_sha256": plan.plan_sha256,
        "runtime_pipeline_sha256": plan.runtime_pipeline_sha256,
        "provider_execution_binding_sha256": provider_execution_binding_sha256,
        "v3_evaluator_dependency_sha256": dependency.evaluation_sha256,
        "v3_evaluator_plan_sha256": dependency.plan_sha256,
        "v3_evaluator_runtime_pipeline_sha256": dependency.runtime_pipeline_sha256,
        "response": dependency.response,
        "response_sha256": dependency.response_sha256,
        "groundings": dependency.groundings,
        "grounding_membership_sha256": dependency.grounding_membership_sha256,
        "grounded_effect": dependency.grounded_effect,
        "grounded_effect_sha256": dependency.grounded_effect_sha256,
        "contextual_grounding_core_sha256": dependency.contextual_grounding_core_sha256,
        "native_projection": projection,
        "native_projection_sha256": projection.projection_sha256,
        "extracted_numeric_values": dependency.extracted_numeric_values,
        "numeric_extraction_fields_evaluated": 4,
        "numeric_evaluator_exact_match": True,
        "typed_graph_mechanics_observed": True,
        "graph_construction_mechanics_authority": False,
        "extraction_accuracy_authority": False,
        "reliability_authority": False,
        "generalization_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_effectiveness_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierRecoveryCoreEvaluationV4.model_validate(
        {**payload, "evaluation_sha256": hash_canonical(payload)}
    )


class MetaSynContextualFrontierRecoveryAuthorizationV4(_Frozen):
    authorization_version: Literal["metasyn-contextual-frontier-recovery-authorization-v4"] = (
        AUTHORIZATION_VERSION
    )
    plan_sha256: Sha256
    request_sha256: Sha256
    request_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    configured_phase_budget_usd_micros: Annotated[int, Field(ge=1)]
    maximum_provider_attempts: Literal[1] = 1
    provider_calls_made_before_authorization: Literal[0] = 0
    exact_request_retries_permitted: Literal[0] = 0
    fallback_requests_permitted: Literal[0] = 0
    authorization_sha256: Sha256

    @model_validator(mode="after")
    def validate_authorization(self) -> MetaSynContextualFrontierRecoveryAuthorizationV4:
        if self.request_cost_ceiling_usd_micros > self.configured_phase_budget_usd_micros:
            raise ValueError("recovery_v4_budget_insufficient")
        _self_hash(self, "authorization_sha256", "recovery_v4_authorization_hash_mismatch")
        return self


class MetaSynContextualFrontierRecoveryIntentV4(_Frozen):
    intent_version: Literal["metasyn-contextual-frontier-recovery-intent-v4"] = INTENT_VERSION
    plan_sha256: Sha256
    authorization_sha256: Sha256
    request_sha256: Sha256
    provider_binding_sha256: Sha256
    request_key: Literal["row17-candidate3-fable5-high-recovery-v4"] = REQUEST_KEY
    permitted_provider_attempts: Literal[1] = 1
    application_retries_permitted: Literal[0] = 0
    sdk_retries_permitted: Literal[0] = 0
    orphan_or_ambiguous_attempt_is_terminal: Literal[True] = True
    attempt_id: Sha256
    intent_sha256: Sha256

    @model_validator(mode="after")
    def validate_intent(self) -> MetaSynContextualFrontierRecoveryIntentV4:
        expected = hash_canonical(
            {
                "plan_sha256": self.plan_sha256,
                "authorization_sha256": self.authorization_sha256,
                "request_sha256": self.request_sha256,
                "provider_binding_sha256": self.provider_binding_sha256,
                "permitted_provider_attempts": 1,
            }
        )
        if self.attempt_id != expected:
            raise ValueError("recovery_v4_attempt_id_mismatch")
        _self_hash(self, "intent_sha256", "recovery_v4_intent_hash_mismatch")
        return self


class MetaSynContextualFrontierRecoveryReceiptV4(_Frozen):
    receipt_version: Literal["metasyn-contextual-frontier-recovery-receipt-v4"] = RECEIPT_VERSION
    plan_sha256: Sha256
    authorization_sha256: Sha256
    intent_sha256: Sha256
    request_sha256: Sha256
    provider_result: MetaSynContextualFrontierProviderResultV1
    provider_result_sha256: Sha256
    credential_archived: Literal[False] = False
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> MetaSynContextualFrontierRecoveryReceiptV4:
        if (
            self.provider_result.request_sha256 != self.request_sha256
            or self.provider_result.result_sha256 != self.provider_result_sha256
        ):
            raise ValueError("recovery_v4_receipt_alias_mismatch")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(self, "receipt_sha256", "recovery_v4_receipt_hash_mismatch")
        return self


TerminalStatusV4 = Literal[
    "typed_graph_mechanics_completed",
    "provider_result_failed",
    "contextual_validation_failed_closed",
    "terminal_ambiguous_attempt_poison",
]


class MetaSynContextualFrontierRecoveryTerminalV4(_Frozen):
    terminal_version: Literal["metasyn-contextual-frontier-recovery-terminal-v4"] = TERMINAL_VERSION
    terminal: Literal[True] = True
    status: TerminalStatusV4
    plan_sha256: Sha256
    authorization_sha256: Sha256
    intent_sha256: Sha256
    request_sha256: Sha256
    provider_attempt_count_upper_bound: Literal[1] = 1
    provider_receipt: MetaSynContextualFrontierRecoveryReceiptV4 | None
    evaluation: MetaSynContextualFrontierRecoveryCoreEvaluationV4 | None
    failure_code: str | None
    exception_type: str | None
    http_status: int | None
    provider_request_id: str | None
    exact_request_retries_permitted: Literal[0] = 0
    predecessor_requests_retried: Literal[0] = 0
    fallback_requests_permitted: Literal[0] = 0
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    terminal_sha256: Sha256

    @model_validator(mode="after")
    def validate_terminal(self) -> MetaSynContextualFrontierRecoveryTerminalV4:
        success = self.status == "typed_graph_mechanics_completed"
        if success != (self.evaluation is not None):
            raise ValueError("recovery_v4_terminal_evaluation_shape_invalid")
        if self.status == "terminal_ambiguous_attempt_poison":
            if self.provider_receipt is not None or self.failure_code is None:
                raise ValueError("recovery_v4_ambiguous_terminal_shape_invalid")
        elif self.provider_receipt is None:
            raise ValueError("recovery_v4_terminal_receipt_missing")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(self, "terminal_sha256", "recovery_v4_terminal_hash_mismatch")
        return self


class MetaSynContextualFrontierRecoveryClientProtocolV4(Protocol):
    def generate(
        self, request: MetaSynContextualFrontierRequestV1
    ) -> MetaSynContextualFrontierProviderResultV1: ...


def _freeze_authorization(
    plan: MetaSynContextualFrontierRecoveryPlanV4, budget: int
) -> MetaSynContextualFrontierRecoveryAuthorizationV4:
    payload = {
        "authorization_version": AUTHORIZATION_VERSION,
        "plan_sha256": plan.plan_sha256,
        "request_sha256": plan.request_sha256,
        "request_cost_ceiling_usd_micros": plan.hard_cost_liability_usd_micros,
        "configured_phase_budget_usd_micros": budget,
        "maximum_provider_attempts": 1,
        "provider_calls_made_before_authorization": 0,
        "exact_request_retries_permitted": 0,
        "fallback_requests_permitted": 0,
    }
    return MetaSynContextualFrontierRecoveryAuthorizationV4.model_validate(
        {**payload, "authorization_sha256": hash_canonical(payload)}
    )


def _freeze_intent(
    plan: MetaSynContextualFrontierRecoveryPlanV4,
    authorization: MetaSynContextualFrontierRecoveryAuthorizationV4,
) -> MetaSynContextualFrontierRecoveryIntentV4:
    if authorization.plan_sha256 != plan.plan_sha256:
        raise MetaSynContextualFrontierRecoveryV4Error("recovery_v4_authorization_plan_drift")
    attempt = hash_canonical(
        {
            "plan_sha256": plan.plan_sha256,
            "authorization_sha256": authorization.authorization_sha256,
            "request_sha256": plan.request_sha256,
            "provider_binding_sha256": plan.request.provider_binding_sha256,
            "permitted_provider_attempts": 1,
        }
    )
    payload = {
        "intent_version": INTENT_VERSION,
        "plan_sha256": plan.plan_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "request_sha256": plan.request_sha256,
        "provider_binding_sha256": plan.request.provider_binding_sha256,
        "request_key": REQUEST_KEY,
        "permitted_provider_attempts": 1,
        "application_retries_permitted": 0,
        "sdk_retries_permitted": 0,
        "orphan_or_ambiguous_attempt_is_terminal": True,
        "attempt_id": attempt,
    }
    return MetaSynContextualFrontierRecoveryIntentV4.model_validate(
        {**payload, "intent_sha256": hash_canonical(payload)}
    )


def _freeze_receipt(
    plan: MetaSynContextualFrontierRecoveryPlanV4,
    authorization: MetaSynContextualFrontierRecoveryAuthorizationV4,
    intent: MetaSynContextualFrontierRecoveryIntentV4,
    result: MetaSynContextualFrontierProviderResultV1,
) -> MetaSynContextualFrontierRecoveryReceiptV4:
    canonical = MetaSynContextualFrontierProviderResultV1.model_validate(
        result.model_dump(mode="json")
    )
    payload = {
        "receipt_version": RECEIPT_VERSION,
        "plan_sha256": plan.plan_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "intent_sha256": intent.intent_sha256,
        "request_sha256": plan.request_sha256,
        "provider_result": canonical,
        "provider_result_sha256": canonical.result_sha256,
        "credential_archived": False,
    }
    return MetaSynContextualFrontierRecoveryReceiptV4.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def _freeze_terminal(
    *,
    status: TerminalStatusV4,
    plan: MetaSynContextualFrontierRecoveryPlanV4,
    authorization: MetaSynContextualFrontierRecoveryAuthorizationV4,
    intent: MetaSynContextualFrontierRecoveryIntentV4,
    receipt: MetaSynContextualFrontierRecoveryReceiptV4 | None = None,
    evaluation: MetaSynContextualFrontierRecoveryCoreEvaluationV4 | None = None,
    failure_code: str | None = None,
    exc: BaseException | None = None,
) -> MetaSynContextualFrontierRecoveryTerminalV4:
    payload = {
        "terminal_version": TERMINAL_VERSION,
        "terminal": True,
        "status": status,
        "plan_sha256": plan.plan_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "intent_sha256": intent.intent_sha256,
        "request_sha256": plan.request_sha256,
        "provider_attempt_count_upper_bound": 1,
        "provider_receipt": receipt,
        "evaluation": evaluation,
        "failure_code": failure_code,
        "exception_type": _safe_exception_type(exc) if exc is not None else None,
        "http_status": _safe_status(exc) if exc is not None else None,
        "provider_request_id": _safe_request_id(exc) if exc is not None else None,
        "exact_request_retries_permitted": 0,
        "predecessor_requests_retried": 0,
        "fallback_requests_permitted": 0,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierRecoveryTerminalV4.model_validate(
        {**payload, "terminal_sha256": hash_canonical(payload)}
    )


def prepare_metasyn_contextual_frontier_recovery_v4(
    *,
    repository_root: Path,
    workspace: Path = DEFAULT_WORKSPACE,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> MetaSynContextualFrontierRecoveryPlanV4:
    plan = freeze_metasyn_contextual_frontier_recovery_plan_v4(
        repository_root=repository_root, config_path=config_path
    )
    root = _fresh_workspace(workspace)
    with _workspace_lock(root):
        _persist_json(_checked_artifact(root, Path("00-prepared.json")), plan)
    return plan


def load_metasyn_contextual_frontier_recovery_plan_v4(
    *, workspace: Path
) -> MetaSynContextualFrontierRecoveryPlanV4:
    root = _existing_workspace(workspace)
    return MetaSynContextualFrontierRecoveryPlanV4.model_validate(
        _load_object(
            _checked_artifact(root, Path("00-prepared.json")),
            code="recovery_v4_prepared_invalid",
        )
    )


def authorize_metasyn_contextual_frontier_recovery_v4(
    *, workspace: Path, phase_budget_usd_micros: int
) -> MetaSynContextualFrontierRecoveryAuthorizationV4:
    root = _existing_workspace(workspace)
    with _workspace_lock(root):
        plan = load_metasyn_contextual_frontier_recovery_plan_v4(workspace=root)
        authorization = _freeze_authorization(plan, phase_budget_usd_micros)
        path = _checked_artifact(root, Path("01-authorized.json"))
        if path.exists():
            observed = MetaSynContextualFrontierRecoveryAuthorizationV4.model_validate(
                _load_object(path, code="recovery_v4_authorization_invalid")
            )
            if observed != authorization:
                raise MetaSynContextualFrontierRecoveryV4Error(
                    "recovery_v4_authorization_replay_mismatch"
                )
            return observed
        _persist_json(path, authorization)
        return authorization


def execute_metasyn_contextual_frontier_recovery_v4(
    *,
    repository_root: Path,
    workspace: Path,
    client: MetaSynContextualFrontierRecoveryClientProtocolV4,
) -> MetaSynContextualFrontierRecoveryTerminalV4:
    root = _existing_workspace(workspace)
    with _workspace_lock(root):
        plan = load_metasyn_contextual_frontier_recovery_plan_v4(workspace=root)
        authorization = MetaSynContextualFrontierRecoveryAuthorizationV4.model_validate(
            _load_object(
                _checked_artifact(root, Path("01-authorized.json")),
                code="recovery_v4_authorization_missing",
            )
        )
        intent = _freeze_intent(plan, authorization)
        intent_path = _checked_artifact(root, Path("intent.json"))
        receipt_path = _checked_artifact(root, Path("provider-receipt.json"))
        terminal_path = _checked_artifact(root, Path("02-terminal.json"))
        if terminal_path.exists():
            return MetaSynContextualFrontierRecoveryTerminalV4.model_validate(
                _load_object(terminal_path, code="recovery_v4_terminal_invalid")
            )
        receipt: MetaSynContextualFrontierRecoveryReceiptV4 | None = None
        if intent_path.exists():
            saved = MetaSynContextualFrontierRecoveryIntentV4.model_validate(
                _load_object(intent_path, code="recovery_v4_intent_invalid")
            )
            if saved != intent:
                raise MetaSynContextualFrontierRecoveryV4Error("recovery_v4_intent_replay_mismatch")
            if not receipt_path.exists():
                terminal = _freeze_terminal(
                    status="terminal_ambiguous_attempt_poison",
                    plan=plan,
                    authorization=authorization,
                    intent=intent,
                    failure_code="orphan_intent_observed_on_resume",
                )
                _persist_json(terminal_path, terminal)
                return terminal
            receipt = MetaSynContextualFrontierRecoveryReceiptV4.model_validate(
                _load_object(receipt_path, code="recovery_v4_receipt_invalid")
            )
        else:
            if receipt_path.exists():
                raise MetaSynContextualFrontierRecoveryV4Error("recovery_v4_receipt_without_intent")
            _persist_json(intent_path, intent)
            try:
                result = client.generate(plan.request)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                terminal = _freeze_terminal(
                    status="terminal_ambiguous_attempt_poison",
                    plan=plan,
                    authorization=authorization,
                    intent=intent,
                    failure_code="provider_call_raised_after_durable_intent",
                    exc=exc,
                )
                _persist_json(terminal_path, terminal)
                return terminal
            receipt = _freeze_receipt(plan, authorization, intent, result)
            _persist_json(receipt_path, receipt)
        assert receipt is not None
        result = receipt.provider_result
        if result.outcome != "completed":
            terminal = _freeze_terminal(
                status="provider_result_failed",
                plan=plan,
                authorization=authorization,
                intent=intent,
                receipt=receipt,
                failure_code=result.failure_code or "provider_result_failed",
            )
        else:
            assert result.parsed_json is not None
            binding = hash_canonical(
                {
                    "plan_sha256": plan.plan_sha256,
                    "authorization_sha256": authorization.authorization_sha256,
                    "intent_sha256": intent.intent_sha256,
                    "receipt_sha256": receipt.receipt_sha256,
                    "provider_result_sha256": receipt.provider_result_sha256,
                }
            )
            try:
                evaluation = evaluate_metasyn_contextual_frontier_recovery_response_v4(
                    repository_root=repository_root,
                    plan=plan,
                    raw_response=result.parsed_json,
                    provider_execution_binding_sha256=binding,
                )
            except Exception as exc:
                terminal = _freeze_terminal(
                    status="contextual_validation_failed_closed",
                    plan=plan,
                    authorization=authorization,
                    intent=intent,
                    receipt=receipt,
                    failure_code=_safe_exception_type(exc),
                )
            else:
                terminal = _freeze_terminal(
                    status="typed_graph_mechanics_completed",
                    plan=plan,
                    authorization=authorization,
                    intent=intent,
                    receipt=receipt,
                    evaluation=evaluation,
                )
        _persist_json(terminal_path, terminal)
        return terminal


def default_metasyn_contextual_frontier_recovery_client_v4(
    plan: MetaSynContextualFrontierRecoveryPlanV4,
) -> MetaSynContextualFrontierClientV1:
    return MetaSynContextualFrontierClientV1(plan.transport_config)
