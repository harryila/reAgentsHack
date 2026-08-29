"""Minimal one-shot recovery using the provider-accepted frontier-v1 grammar.

This is a post-hoc, target-conditioned schema-compatibility smoke.  It never
retries either immutable frontier-v1 request or the immutable recovery-v2
request.  The provider sees the exact compiled array grammar accepted for the
frontier-v1 primary request, but receives a fresh prompt that fixes the estimand
to fedratinib 500 mg versus placebo and discloses the canonical nonnumeric field
tokens.  The four event/total values remain source-only extraction targets.

Trusted code requires the exact fifteen-field roster, sorts claims locally, and
adapts the array to the already-tested recovery-v2 evaluator.  Every empirical,
scientific, calibration, synthesis, and release authority remains false.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from jsonschema import validate as validate_json_schema
from jsonschema.exceptions import ValidationError
from pydantic import ConfigDict, Field, model_validator

from literature_multiverse.contextual_numeric_grounding_v3 import (
    ContextualClaimV3,
    ContextualGroundedClaimV3,
    ContextualGroundedEffectV3,
    ContextualNativeProjectionV3,
    ContextualPacketAbstentionV3,
    ContextualPacketCompletedV3,
    ContextualProviderContextV3,
    _runtime_native_projection_from_fixture,
)
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.metasyn_contextual_frontier_recovery_v2 import (
    RESPONSE_VERSION as V2_RESPONSE_VERSION,
)
from literature_multiverse.metasyn_contextual_frontier_recovery_v2 import (
    MetaSynContextualFrontierRecoveryPlanV2,
    MetaSynContextualFrontierRecoveryTargetSpecV2,
    evaluate_metasyn_contextual_frontier_recovery_response_v2,
)
from literature_multiverse.metasyn_contextual_frontier_runtime_v1 import (
    ANTHROPIC_API_VERSION,
    API_BASE_URL,
    MetaSynContextualFrontierClientV1,
    MetaSynContextualFrontierConfigV1,
    MetaSynContextualFrontierPlanV1,
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

RUNTIME_VERSION = "metasyn-contextual-frontier-recovery-v3"
PLAN_VERSION = "metasyn-contextual-frontier-recovery-plan-v3"
CONFIG_VERSION = "metasyn-contextual-frontier-recovery-config-v3"
AUTHORIZATION_VERSION = "metasyn-contextual-frontier-recovery-authorization-v3"
INTENT_VERSION = "metasyn-contextual-frontier-recovery-intent-v3"
RECEIPT_VERSION = "metasyn-contextual-frontier-recovery-receipt-v3"
TERMINAL_VERSION = "metasyn-contextual-frontier-recovery-terminal-v3"

RECOVERY_LABEL = "post_hoc_target_conditioned_schema_compatibility_recovery"
REQUEST_KEY = "row17-candidate3-fable5-high-recovery-v3"
PRIMARY_WITNESS = "metasyn-row17-candidate3-binary-primary-endpoint"
EXPLICIT_ESTIMAND = (
    "fedratinib 500-mg group versus placebo group for the primary end point "
    "spleen response at week 24"
)

DEFAULT_CONFIG_PATH = Path("configs/benchmarks/metasyn-contextual-frontier-recovery-v3.json")
DEFAULT_WORKSPACE = Path("data/cache/metasyn/contextual-frontier-recovery-v3")
RUNTIME_SOURCE_PATH = Path("src/literature_multiverse/metasyn_contextual_frontier_recovery_v3.py")
V1_PREPARED_PATH = Path("data/cache/metasyn/contextual-frontier-runtime-v1/00-prepared.json")
V1_TERMINAL_PATH = Path("data/cache/metasyn/contextual-frontier-runtime-v1/02-terminal.json")
V2_PREPARED_PATH = Path("data/cache/metasyn/contextual-frontier-recovery-v2/00-prepared.json")
V2_TERMINAL_PATH = Path("data/cache/metasyn/contextual-frontier-recovery-v2/02-terminal.json")

ACCEPTED_V1_REQUEST_SHA256 = "56b6b257bcf6956932a5a8cecc522b0a2f9e1f1b499023c9a2f4c75bf20e562e"
ACCEPTED_ORIGINAL_SCHEMA_SHA256 = "f7d086fa8f01945ac54a34c1e16b9a911417e1b3970bf67b73115844412242c0"
ACCEPTED_COMPILED_SCHEMA_SHA256 = "4261b57af87674350eba3cbbc7205ee97d40b2741e1fd1ed0bcce52cb4b027cd"
ACCEPTED_WIRE_SCHEMA_SHA256 = "fe266be4e582688a8c841a3b4eea6763d6f889627f8ed4b46366d1dde8b7b370"


class MetaSynContextualFrontierRecoveryV3Error(ValueError):
    """A recovery-v3 contract or exact-once transition failed closed."""


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


def _root(repository_root: Path) -> Path:
    return repository_root.resolve(strict=True)


def _read_repository_object(root: Path, relative: Path) -> dict[str, Any]:
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_path_escape") from exc
    if path.is_symlink() or not path.is_file():
        raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_source_artifact_unsafe")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_source_artifact_invalid")
    return value


def _load_v1_plan(root: Path) -> MetaSynContextualFrontierPlanV1:
    raw = _read_repository_object(root, V1_PREPARED_PATH)
    return MetaSynContextualFrontierPlanV1.model_validate(raw.get("plan", raw))


def _load_v2_plan(root: Path) -> MetaSynContextualFrontierRecoveryPlanV2:
    raw = _read_repository_object(root, V2_PREPARED_PATH)
    return MetaSynContextualFrontierRecoveryPlanV2.model_validate(raw.get("plan", raw))


class MetaSynContextualFrontierRecoveryConfigV3(_Frozen):
    config_version: Literal["metasyn-contextual-frontier-recovery-config-v3"] = CONFIG_VERSION
    model: Literal["claude-fable-5"] = "claude-fable-5"
    effort: Literal["high"] = "high"
    request_key: Literal["row17-candidate3-fable5-high-recovery-v3"] = REQUEST_KEY
    maximum_provider_calls: Literal[1] = 1
    application_retries_per_request: Literal[0] = 0
    sdk_retries_per_request: Literal[0] = 0
    fallback_requests_permitted: Literal[0] = 0
    immutable_v1_requests_retry_permitted: Literal[False] = False
    immutable_v2_request_retry_permitted: Literal[False] = False
    operator_authorized_source_transmission: Literal[True] = True
    accepted_original_schema_sha256: Literal[
        "f7d086fa8f01945ac54a34c1e16b9a911417e1b3970bf67b73115844412242c0"
    ] = ACCEPTED_ORIGINAL_SCHEMA_SHA256
    accepted_compiled_schema_sha256: Literal[
        "4261b57af87674350eba3cbbc7205ee97d40b2741e1fd1ed0bcce52cb4b027cd"
    ] = ACCEPTED_COMPILED_SCHEMA_SHA256
    accepted_wire_schema_sha256: Literal[
        "fe266be4e582688a8c841a3b4eea6763d6f889627f8ed4b46366d1dde8b7b370"
    ] = ACCEPTED_WIRE_SCHEMA_SHA256
    post_hoc_target_conditioned: Literal[True] = True
    graph_construction_mechanics_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False

    @property
    def config_sha256(self) -> str:
        return hash_canonical(self)


def load_metasyn_contextual_frontier_recovery_config_v3(
    *, repository_root: Path, config_path: Path = DEFAULT_CONFIG_PATH
) -> MetaSynContextualFrontierRecoveryConfigV3:
    root = _root(repository_root)
    raw = _read_repository_object(root, config_path)
    return MetaSynContextualFrontierRecoveryConfigV3.model_validate(raw)


def _render_prompt(
    *,
    provider_context: ContextualProviderContextV3,
    target_spec: MetaSynContextualFrontierRecoveryTargetSpecV2,
) -> str:
    fields = []
    for item in target_spec.fields:
        fields.append(
            {
                "field_path": item.field_path,
                "normalization": item.normalization,
                "required_token": (
                    item.canonical_token
                    if item.canonical_token is not None
                    else "EXTRACT_EXACT_UNSIGNED_INTEGER_FROM_SOURCE"
                ),
            }
        )
    passages = [item.model_dump(mode="json") for item in provider_context.passages]
    field_json = json.dumps(fields, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    passage_json = json.dumps(passages, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return (
        "Post-hoc target-conditioned schema-compatibility recovery v3.\n"
        "TARGET ESTIMAND: fedratinib 500-mg group versus placebo group for the primary "
        "end point spleen response at week 24. Do not select the 400-mg arm or the "
        "secondary symptom-response endpoint.\n"
        "Return the completed object in the supplied array schema only if you can provide "
        "exactly the fifteen FIELD_ROSTER_JSON claims, once each, with no extra field_path. "
        "Use each canonical nonnumeric token exactly as written. For the four event/total "
        "fields, extract the exact unsigned integer from SOURCE_PASSAGES_JSON; their answers "
        "are intentionally not provided outside those passages. Copy support_quote exactly, "
        "then choose a local context occurring exactly once in the quote and a token occurring "
        "exactly once in that context. Trusted code sorts claims and verifies the exact field "
        "set and semantic target. If any required claim cannot be grounded, return the schema's "
        "unable_to_complete object.\n"
        f"TARGET_CONTRACT_SHA256={target_spec.target_sha256}\n"
        f"PROVIDER_CONTEXT_SHA256={provider_context.context_sha256}\n"
        f"FIELD_ROSTER_JSON={field_json}\n"
        f"SOURCE_PASSAGES_JSON={passage_json}"
    )


def freeze_metasyn_contextual_frontier_recovery_request_v3(
    *,
    provider_context: ContextualProviderContextV3,
    target_spec: MetaSynContextualFrontierRecoveryTargetSpecV2,
    accepted_v1_request: MetaSynContextualFrontierRequestV1,
    transport_config: MetaSynContextualFrontierConfigV1,
) -> MetaSynContextualFrontierRequestV1:
    """Freeze a new prompt on the exact provider-accepted v1 compiled grammar."""

    accepted = MetaSynContextualFrontierRequestV1.model_validate(
        accepted_v1_request.model_dump(mode="json")
    )
    if (
        accepted.request_sha256 != ACCEPTED_V1_REQUEST_SHA256
        or accepted.original_schema_sha256 != ACCEPTED_ORIGINAL_SCHEMA_SHA256
        or accepted.compiled_schema_sha256 != ACCEPTED_COMPILED_SCHEMA_SHA256
        or accepted.wire_schema_sha256 != ACCEPTED_WIRE_SCHEMA_SHA256
        or accepted.config_sha256 != transport_config.config_sha256
        or accepted.witness_id != PRIMARY_WITNESS
    ):
        raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_accepted_template_drift")
    if len(target_spec.fields) != 15 or [item.field_path for item in target_spec.fields] != sorted(
        item.field_path for item in target_spec.fields
    ):
        raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_target_field_roster_invalid")
    prompt = _render_prompt(provider_context=provider_context, target_spec=target_spec)
    kwargs = _wire_kwargs(
        model_system=accepted.base_system,
        prompt=prompt,
        wire_schema=accepted.compiled_schema.wire_schema,
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
    payload = accepted.model_dump(mode="json", exclude={"request_sha256"})
    payload.update(
        {
            "request_key": REQUEST_KEY,
            "provider_binding_sha256": hash_canonical(
                {
                    "recovery_label": RECOVERY_LABEL,
                    "provider_context_sha256": provider_context.context_sha256,
                    "target_spec_sha256": target_spec.target_sha256,
                }
            ),
            "prompt": prompt,
            "prompt_sha256": _sha256_utf8(prompt),
            "wire_kwargs": kwargs,
            "wire_kwargs_sha256": hash_canonical(kwargs),
            "wire_call_surface_sha256": surface_sha,
            "cost_ceiling": _freeze_cost(
                model_system=accepted.base_system,
                prompt=prompt,
                wire_schema=accepted.compiled_schema.wire_schema,
            ),
        }
    )
    request = MetaSynContextualFrontierRequestV1.model_validate(
        {**payload, "request_sha256": hash_canonical(payload)}
    )
    if (
        request.request_sha256 == accepted.request_sha256
        or request.prompt_sha256 == accepted.prompt_sha256
    ):
        raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_not_materially_fresh")
    return request


class MetaSynContextualFrontierRecoveryPlanV3(_Frozen):
    plan_version: Literal["metasyn-contextual-frontier-recovery-plan-v3"] = PLAN_VERSION
    runtime_version: Literal["metasyn-contextual-frontier-recovery-v3"] = RUNTIME_VERSION
    status: Literal["offline_prepared_zero_provider_calls"] = "offline_prepared_zero_provider_calls"
    recovery_label: Literal["post_hoc_target_conditioned_schema_compatibility_recovery"] = (
        RECOVERY_LABEL
    )
    config: MetaSynContextualFrontierRecoveryConfigV3
    config_sha256: Sha256
    transport_config: MetaSynContextualFrontierConfigV1
    transport_config_sha256: Sha256
    target_spec: MetaSynContextualFrontierRecoveryTargetSpecV2
    target_spec_sha256: Sha256
    provider_context_sha256: Sha256
    provider_visible_passage_count: Literal[16] = 16
    accepted_v1_request_sha256: Literal[
        "56b6b257bcf6956932a5a8cecc522b0a2f9e1f1b499023c9a2f4c75bf20e562e"
    ] = ACCEPTED_V1_REQUEST_SHA256
    accepted_original_schema_sha256: Sha256
    accepted_compiled_schema_sha256: Sha256
    accepted_wire_schema_sha256: Sha256
    immutable_v1_prepared_file_sha256: Sha256
    immutable_v1_terminal_file_sha256: Sha256
    immutable_v2_prepared_file_sha256: Sha256
    immutable_v2_terminal_file_sha256: Sha256
    request: MetaSynContextualFrontierRequestV1
    request_sha256: Sha256
    runtime_pipeline_components: dict[str, str]
    runtime_pipeline_sha256: Sha256
    hard_cost_liability_usd_micros: Annotated[int, Field(ge=1)]
    provider_calls_made: Literal[0] = 0
    maximum_provider_calls: Literal[1] = 1
    predecessor_requests_retried: Literal[0] = 0
    evaluator_fixture_passed_to_request_builder: Literal[False] = False
    numeric_targets_disclosed_outside_source_passages: Literal[False] = False
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
    def validate_plan(self) -> MetaSynContextualFrontierRecoveryPlanV3:
        if (
            self.config_sha256 != self.config.config_sha256
            or self.transport_config_sha256 != self.transport_config.config_sha256
            or self.target_spec_sha256 != self.target_spec.target_sha256
            or self.request_sha256 != self.request.request_sha256
            or self.request.original_schema_sha256 != ACCEPTED_ORIGINAL_SCHEMA_SHA256
            or self.request.compiled_schema_sha256 != ACCEPTED_COMPILED_SCHEMA_SHA256
            or self.request.wire_schema_sha256 != ACCEPTED_WIRE_SCHEMA_SHA256
            or self.runtime_pipeline_sha256 != hash_canonical(self.runtime_pipeline_components)
            or self.hard_cost_liability_usd_micros
            != self.request.cost_ceiling.request_cost_ceiling_usd_micros
        ):
            raise ValueError("recovery_v3_plan_replay_mismatch")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(self, "plan_sha256", "recovery_v3_plan_hash_mismatch")
        return self


def freeze_metasyn_contextual_frontier_recovery_plan_v3(
    *, repository_root: Path, config_path: Path = DEFAULT_CONFIG_PATH
) -> MetaSynContextualFrontierRecoveryPlanV3:
    root = _root(repository_root)
    config = load_metasyn_contextual_frontier_recovery_config_v3(
        repository_root=root, config_path=config_path
    )
    v1_plan = _load_v1_plan(root)
    v2_plan = _load_v2_plan(root)
    primary = [item.request for item in v1_plan.roster if item.witness_id == PRIMARY_WITNESS]
    if len(primary) != 1:
        raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_v1_primary_missing")
    request = freeze_metasyn_contextual_frontier_recovery_request_v3(
        provider_context=v2_plan.provider_context,
        target_spec=v2_plan.target_spec,
        accepted_v1_request=primary[0],
        transport_config=v2_plan.transport_profile_config,
    )
    if request.request_sha256 in {item.request_sha256 for item in v1_plan.roster}:
        raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_retries_v1_request")
    if request.request_sha256 == v2_plan.request.transport_request_sha256:
        raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_retries_v2_request")
    components = {
        "runtime_source_sha256": sha256_file(root / RUNTIME_SOURCE_PATH),
        "config_sha256": config.config_sha256,
        "transport_config_sha256": v2_plan.transport_profile_config.config_sha256,
        "target_spec_sha256": v2_plan.target_spec_sha256,
        "provider_context_sha256": v2_plan.provider_context_sha256,
        "accepted_original_schema_sha256": ACCEPTED_ORIGINAL_SCHEMA_SHA256,
        "accepted_compiled_schema_sha256": ACCEPTED_COMPILED_SCHEMA_SHA256,
        "accepted_wire_schema_sha256": ACCEPTED_WIRE_SCHEMA_SHA256,
        "request_sha256": request.request_sha256,
        "immutable_v1_prepared_file_sha256": sha256_file(root / V1_PREPARED_PATH),
        "immutable_v1_terminal_file_sha256": sha256_file(root / V1_TERMINAL_PATH),
        "immutable_v2_prepared_file_sha256": sha256_file(root / V2_PREPARED_PATH),
        "immutable_v2_terminal_file_sha256": sha256_file(root / V2_TERMINAL_PATH),
    }
    payload = {
        "plan_version": PLAN_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "status": "offline_prepared_zero_provider_calls",
        "recovery_label": RECOVERY_LABEL,
        "config": config,
        "config_sha256": config.config_sha256,
        "transport_config": v2_plan.transport_profile_config,
        "transport_config_sha256": v2_plan.transport_profile_config.config_sha256,
        "target_spec": v2_plan.target_spec,
        "target_spec_sha256": v2_plan.target_spec_sha256,
        "provider_context_sha256": v2_plan.provider_context_sha256,
        "provider_visible_passage_count": len(v2_plan.provider_context.passages),
        "accepted_v1_request_sha256": ACCEPTED_V1_REQUEST_SHA256,
        "accepted_original_schema_sha256": ACCEPTED_ORIGINAL_SCHEMA_SHA256,
        "accepted_compiled_schema_sha256": ACCEPTED_COMPILED_SCHEMA_SHA256,
        "accepted_wire_schema_sha256": ACCEPTED_WIRE_SCHEMA_SHA256,
        "immutable_v1_prepared_file_sha256": components["immutable_v1_prepared_file_sha256"],
        "immutable_v1_terminal_file_sha256": components["immutable_v1_terminal_file_sha256"],
        "immutable_v2_prepared_file_sha256": components["immutable_v2_prepared_file_sha256"],
        "immutable_v2_terminal_file_sha256": components["immutable_v2_terminal_file_sha256"],
        "request": request,
        "request_sha256": request.request_sha256,
        "runtime_pipeline_components": components,
        "runtime_pipeline_sha256": hash_canonical(components),
        "hard_cost_liability_usd_micros": request.cost_ceiling.request_cost_ceiling_usd_micros,
        "provider_calls_made": 0,
        "maximum_provider_calls": 1,
        "predecessor_requests_retried": 0,
        "evaluator_fixture_passed_to_request_builder": False,
        "numeric_targets_disclosed_outside_source_passages": False,
        "graph_construction_mechanics_authority": False,
        "extraction_accuracy_authority": False,
        "reliability_authority": False,
        "generalization_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierRecoveryPlanV3.model_validate(
        {**payload, "plan_sha256": hash_canonical(payload)}
    )


def _adapt_array_response_to_v2(
    *,
    plan: MetaSynContextualFrontierRecoveryPlanV3,
    v2_plan: MetaSynContextualFrontierRecoveryPlanV2,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(raw)
    status = value.get("packet_status")
    if v2_plan.provider_context_sha256 != plan.provider_context_sha256:
        raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_v2_context_drift")
    if status == "unable_to_complete":
        return {
            "response_version": V2_RESPONSE_VERSION,
            "status": "unable_to_complete",
            "target_contract_sha256": plan.target_spec_sha256,
            "reason": value.get("reason"),
        }
    if status != "completed":
        raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_packet_status_invalid")
    if (
        value.get("candidate_binding_sha256") != v2_plan.provider_context.candidate_binding_sha256
        or value.get("canonical_outcome_id")
        != v2_plan.provider_context.candidate.canonical_outcome_id
        or value.get("effect_kind") != "binary_group_statistics"
        or value.get("effect_format_token") is not None
        or value.get("effect_computation")
        != "binary_group_statistics_to_odds_ratio_via_existing_harmonizer"
    ):
        raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_semantic_header_mismatch")
    claims_raw = value.get("claims")
    if not isinstance(claims_raw, list):
        raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_claims_not_array")
    claims = [ContextualClaimV3.model_validate(item) for item in claims_raw]
    claims.sort(key=lambda item: (item.field_path, item.passage_id))
    required = [item.field_path for item in plan.target_spec.fields]
    observed = [item.field_path for item in claims]
    if len(claims) != 15 or observed != required:
        raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_exact_field_set_mismatch")
    by_contract = {item.field_path: item for item in plan.target_spec.fields}
    keyed: dict[str, Any] = {}
    for claim in claims:
        contract = by_contract[claim.field_path]
        if claim.normalization != contract.normalization:
            raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_normalization_mismatch")
        if contract.canonical_token is not None and claim.token != contract.canonical_token:
            raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_canonical_token_mismatch")
        if contract.canonical_token is None:
            token = claim.token
            valid_unsigned = token == "0" or (
                1 <= len(token) <= 10
                and token[0] in "123456789"
                and (not token[1:] or (token[1:].isascii() and token[1:].isdigit()))
            )
            if not valid_unsigned:
                raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_numeric_token_invalid")
        keyed[claim.field_path] = claim.model_dump(mode="json", exclude={"field_path"})
    return {
        "response_version": V2_RESPONSE_VERSION,
        "status": "completed",
        "target_contract_sha256": plan.target_spec_sha256,
        "claims_by_field": keyed,
    }


def evaluate_metasyn_contextual_frontier_recovery_response_v3(
    *,
    repository_root: Path,
    plan: MetaSynContextualFrontierRecoveryPlanV3,
    raw_response: Mapping[str, Any],
    provider_execution_binding_sha256: str,
) -> MetaSynContextualFrontierRecoveryCoreEvaluationV3:
    root = _root(repository_root)
    v2_plan = _load_v2_plan(root)
    if (
        sha256_file(root / V2_PREPARED_PATH) != plan.immutable_v2_prepared_file_sha256
        or v2_plan.provider_context_sha256 != plan.provider_context_sha256
        or v2_plan.target_spec_sha256 != plan.target_spec_sha256
    ):
        raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_v2_evaluator_drift")
    try:
        validate_json_schema(raw_response, plan.request.compiled_schema.original_schema)
    except ValidationError as exc:
        raise MetaSynContextualFrontierRecoveryV3Error(
            "recovery_v3_accepted_array_schema_invalid"
        ) from exc
    adapted = _adapt_array_response_to_v2(plan=plan, v2_plan=v2_plan, raw=raw_response)
    dependency = evaluate_metasyn_contextual_frontier_recovery_response_v2(
        plan=v2_plan,
        raw_response=adapted,
        provider_execution_binding_sha256=provider_execution_binding_sha256,
    )
    projection: ContextualNativeProjectionV3 | None = None
    if dependency.status == "typed_graph_mechanics_completed":
        if (
            dependency.groundings is None
            or dependency.grounded_effect is None
            or dependency.contextual_grounding_core_sha256 is None
        ):
            raise MetaSynContextualFrontierRecoveryV3Error(
                "recovery_v3_dependency_success_artifacts_missing"
            )
        projection = _runtime_native_projection_from_fixture(
            fixture_receipt=v2_plan.evaluator_fixture,
            effect=dependency.grounded_effect,
            groundings=dependency.groundings,
            grounding_core_sha256=dependency.contextual_grounding_core_sha256,
            runtime_pipeline_sha256=plan.runtime_pipeline_sha256,
            provider_execution_binding_sha256=provider_execution_binding_sha256,
        )
    payload = {
        "evaluation_version": "metasyn-contextual-frontier-recovery-core-evaluation-v3",
        "recovery_label": RECOVERY_LABEL,
        "status": dependency.status,
        "plan_sha256": plan.plan_sha256,
        "runtime_pipeline_sha256": plan.runtime_pipeline_sha256,
        "provider_execution_binding_sha256": provider_execution_binding_sha256,
        "v2_evaluator_dependency_sha256": dependency.evaluation_sha256,
        "v2_evaluator_plan_sha256": dependency.plan_sha256,
        "v2_evaluator_runtime_pipeline_sha256": dependency.runtime_pipeline_sha256,
        "response": dependency.response,
        "response_sha256": dependency.response_sha256,
        "groundings": dependency.groundings,
        "grounding_membership_sha256": dependency.grounding_membership_sha256,
        "grounded_effect": dependency.grounded_effect,
        "grounded_effect_sha256": dependency.grounded_effect_sha256,
        "contextual_grounding_core_sha256": dependency.contextual_grounding_core_sha256,
        "native_projection": projection,
        "native_projection_sha256": (
            projection.projection_sha256 if projection is not None else None
        ),
        "extracted_numeric_values": dependency.extracted_numeric_values,
        "numeric_extraction_fields_evaluated": (dependency.numeric_extraction_fields_evaluated),
        "numeric_evaluator_exact_match": dependency.numeric_evaluator_exact_match,
        "provider_visible_passages_eligible_for_citation": (
            dependency.provider_visible_passages_eligible_for_citation
        ),
        "exact_field_roster_not_exact_passage_roster": True,
        "caller_supplied_fields_excluded_from_extraction_scoring": True,
        "typed_graph_mechanics_observed": dependency.typed_graph_mechanics_observed,
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
    return MetaSynContextualFrontierRecoveryCoreEvaluationV3.model_validate(
        {**payload, "evaluation_sha256": hash_canonical(payload)}
    )


RecoveryResponseV3 = ContextualPacketCompletedV3 | ContextualPacketAbstentionV3


class MetaSynContextualFrontierRecoveryCoreEvaluationV3(_Frozen):
    evaluation_version: Literal["metasyn-contextual-frontier-recovery-core-evaluation-v3"] = (
        "metasyn-contextual-frontier-recovery-core-evaluation-v3"
    )
    recovery_label: Literal["post_hoc_target_conditioned_schema_compatibility_recovery"] = (
        RECOVERY_LABEL
    )
    status: Literal["typed_graph_mechanics_completed", "scientific_abstention"]
    plan_sha256: Sha256
    runtime_pipeline_sha256: Sha256
    provider_execution_binding_sha256: Sha256
    v2_evaluator_dependency_sha256: Sha256
    v2_evaluator_plan_sha256: Sha256
    v2_evaluator_runtime_pipeline_sha256: Sha256
    response: RecoveryResponseV3
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
    scientific_effectiveness_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    evaluation_sha256: Sha256

    @model_validator(mode="after")
    def validate_evaluation(
        self,
    ) -> MetaSynContextualFrontierRecoveryCoreEvaluationV3:
        if self.response_sha256 != hash_canonical(self.response.model_dump(mode="json")):
            raise ValueError("recovery_v3_response_hash_mismatch")
        if self.v2_evaluator_dependency_sha256 == self.evaluation_sha256:
            raise ValueError("recovery_v3_dependency_self_reference")
        aliases = (
            (self.groundings is None) == (self.grounding_membership_sha256 is None)
            and (self.grounded_effect is None) == (self.grounded_effect_sha256 is None)
            and (self.native_projection is None) == (self.native_projection_sha256 is None)
        )
        if not aliases:
            raise ValueError("recovery_v3_evaluation_presence_mismatch")
        if self.groundings is not None and self.grounding_membership_sha256 != hash_canonical(
            [item.grounding_sha256 for item in self.groundings]
        ):
            raise ValueError("recovery_v3_grounding_membership_mismatch")
        if self.grounded_effect is not None and (
            self.grounded_effect_sha256 != self.grounded_effect.effect_sha256
        ):
            raise ValueError("recovery_v3_grounded_effect_alias_mismatch")
        success = self.status == "typed_graph_mechanics_completed"
        if (
            success != self.typed_graph_mechanics_observed
            or success != (self.numeric_evaluator_exact_match is True)
            or success != (self.numeric_extraction_fields_evaluated == 4)
            or success != (self.extracted_numeric_values is not None)
            or success != (self.native_projection is not None)
        ):
            raise ValueError("recovery_v3_evaluation_status_mismatch")
        if success:
            assert self.native_projection is not None
            assert self.native_projection_sha256 is not None
            if (
                self.native_projection_sha256 != self.native_projection.projection_sha256
                or self.native_projection.runtime_pipeline_sha256 != self.runtime_pipeline_sha256
                or self.native_projection.provider_execution_binding_sha256
                != self.provider_execution_binding_sha256
                or self.native_projection.fragment is None
                or self.native_projection.fragment.pipeline_fingerprint_sha256
                != self.runtime_pipeline_sha256
            ):
                raise ValueError("recovery_v3_projection_runtime_binding_mismatch")
        _self_hash(self, "evaluation_sha256", "recovery_v3_evaluation_hash_mismatch")
        return self


class MetaSynContextualFrontierRecoveryAuthorizationV3(_Frozen):
    authorization_version: Literal["metasyn-contextual-frontier-recovery-authorization-v3"] = (
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
    def validate_authorization(self) -> MetaSynContextualFrontierRecoveryAuthorizationV3:
        if self.request_cost_ceiling_usd_micros > self.configured_phase_budget_usd_micros:
            raise ValueError("recovery_v3_budget_insufficient")
        _self_hash(self, "authorization_sha256", "recovery_v3_authorization_hash_mismatch")
        return self


class MetaSynContextualFrontierRecoveryIntentV3(_Frozen):
    intent_version: Literal["metasyn-contextual-frontier-recovery-intent-v3"] = INTENT_VERSION
    plan_sha256: Sha256
    authorization_sha256: Sha256
    request_key: Literal["row17-candidate3-fable5-high-recovery-v3"] = REQUEST_KEY
    request_sha256: Sha256
    provider_binding_sha256: Sha256
    source_bearing: Literal[True] = True
    permitted_provider_attempts: Literal[1] = 1
    application_retries_permitted: Literal[0] = 0
    sdk_retries_permitted: Literal[0] = 0
    orphan_or_ambiguous_attempt_is_terminal: Literal[True] = True
    attempt_id: Sha256
    intent_sha256: Sha256

    @model_validator(mode="after")
    def validate_intent(self) -> MetaSynContextualFrontierRecoveryIntentV3:
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
            raise ValueError("recovery_v3_attempt_id_mismatch")
        _self_hash(self, "intent_sha256", "recovery_v3_intent_hash_mismatch")
        return self


class MetaSynContextualFrontierRecoveryReceiptV3(_Frozen):
    receipt_version: Literal["metasyn-contextual-frontier-recovery-receipt-v3"] = RECEIPT_VERSION
    plan_sha256: Sha256
    authorization_sha256: Sha256
    intent_sha256: Sha256
    request_sha256: Sha256
    provider_result: MetaSynContextualFrontierProviderResultV1
    provider_result_sha256: Sha256
    credential_archived: Literal[False] = False
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> MetaSynContextualFrontierRecoveryReceiptV3:
        if (
            self.provider_result.request_sha256 != self.request_sha256
            or self.provider_result.result_sha256 != self.provider_result_sha256
        ):
            raise ValueError("recovery_v3_receipt_alias_mismatch")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(self, "receipt_sha256", "recovery_v3_receipt_hash_mismatch")
        return self


TerminalStatus = Literal[
    "typed_graph_mechanics_completed",
    "scientific_abstention",
    "provider_result_failed",
    "contextual_validation_failed_closed",
    "terminal_ambiguous_attempt_poison",
]


class MetaSynContextualFrontierRecoveryTerminalV3(_Frozen):
    terminal_version: Literal["metasyn-contextual-frontier-recovery-terminal-v3"] = TERMINAL_VERSION
    terminal: Literal[True] = True
    status: TerminalStatus
    plan_sha256: Sha256
    authorization_sha256: Sha256
    intent_sha256: Sha256
    request_sha256: Sha256
    provider_attempt_count_upper_bound: Literal[1] = 1
    provider_receipt: MetaSynContextualFrontierRecoveryReceiptV3 | None
    evaluation: MetaSynContextualFrontierRecoveryCoreEvaluationV3 | None
    failure_code: str | None
    exception_type: str | None
    http_status: int | None
    provider_request_id: str | None
    exact_request_retries_permitted: Literal[0] = 0
    predecessor_requests_retried: Literal[0] = 0
    fallback_requests_permitted: Literal[0] = 0
    graph_construction_mechanics_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    reliability_authority: Literal[False] = False
    generalization_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    terminal_sha256: Sha256

    @model_validator(mode="after")
    def validate_terminal(self) -> MetaSynContextualFrontierRecoveryTerminalV3:
        if self.status in {"typed_graph_mechanics_completed", "scientific_abstention"}:
            if (
                self.provider_receipt is None
                or self.evaluation is None
                or self.failure_code is not None
            ):
                raise ValueError("recovery_v3_success_terminal_shape_invalid")
        elif self.status == "provider_result_failed":
            if (
                self.provider_receipt is None
                or self.evaluation is not None
                or self.failure_code is None
            ):
                raise ValueError("recovery_v3_provider_failure_shape_invalid")
        elif self.status == "contextual_validation_failed_closed":
            if (
                self.provider_receipt is None
                or self.evaluation is not None
                or self.failure_code is None
            ):
                raise ValueError("recovery_v3_validation_failure_shape_invalid")
        elif self.provider_receipt is not None or self.evaluation is not None:
            raise ValueError("recovery_v3_ambiguous_terminal_shape_invalid")
        _assert_secret_free(self.model_dump(mode="json"))
        _self_hash(self, "terminal_sha256", "recovery_v3_terminal_hash_mismatch")
        return self


class MetaSynContextualFrontierRecoveryClientProtocolV3(Protocol):
    def generate(
        self, request: MetaSynContextualFrontierRequestV1
    ) -> MetaSynContextualFrontierProviderResultV1: ...


def _freeze_authorization(
    *, plan: MetaSynContextualFrontierRecoveryPlanV3, phase_budget_usd_micros: int
) -> MetaSynContextualFrontierRecoveryAuthorizationV3:
    payload = {
        "authorization_version": AUTHORIZATION_VERSION,
        "plan_sha256": plan.plan_sha256,
        "request_sha256": plan.request_sha256,
        "request_cost_ceiling_usd_micros": plan.hard_cost_liability_usd_micros,
        "configured_phase_budget_usd_micros": phase_budget_usd_micros,
        "maximum_provider_attempts": 1,
        "provider_calls_made_before_authorization": 0,
        "exact_request_retries_permitted": 0,
        "fallback_requests_permitted": 0,
    }
    return MetaSynContextualFrontierRecoveryAuthorizationV3.model_validate(
        {**payload, "authorization_sha256": hash_canonical(payload)}
    )


def _freeze_intent(
    *,
    plan: MetaSynContextualFrontierRecoveryPlanV3,
    authorization: MetaSynContextualFrontierRecoveryAuthorizationV3,
) -> MetaSynContextualFrontierRecoveryIntentV3:
    if (
        authorization.plan_sha256 != plan.plan_sha256
        or authorization.request_sha256 != plan.request_sha256
    ):
        raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_authorization_plan_mismatch")
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
        "request_key": REQUEST_KEY,
        "request_sha256": plan.request_sha256,
        "provider_binding_sha256": plan.request.provider_binding_sha256,
        "source_bearing": True,
        "permitted_provider_attempts": 1,
        "application_retries_permitted": 0,
        "sdk_retries_permitted": 0,
        "orphan_or_ambiguous_attempt_is_terminal": True,
        "attempt_id": attempt,
    }
    return MetaSynContextualFrontierRecoveryIntentV3.model_validate(
        {**payload, "intent_sha256": hash_canonical(payload)}
    )


def _freeze_receipt(
    *,
    plan: MetaSynContextualFrontierRecoveryPlanV3,
    authorization: MetaSynContextualFrontierRecoveryAuthorizationV3,
    intent: MetaSynContextualFrontierRecoveryIntentV3,
    result: MetaSynContextualFrontierProviderResultV1,
) -> MetaSynContextualFrontierRecoveryReceiptV3:
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
    return MetaSynContextualFrontierRecoveryReceiptV3.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def _freeze_terminal(
    *,
    status: TerminalStatus,
    plan: MetaSynContextualFrontierRecoveryPlanV3,
    authorization: MetaSynContextualFrontierRecoveryAuthorizationV3,
    intent: MetaSynContextualFrontierRecoveryIntentV3,
    receipt: MetaSynContextualFrontierRecoveryReceiptV3 | None = None,
    evaluation: MetaSynContextualFrontierRecoveryCoreEvaluationV3 | None = None,
    failure_code: str | None = None,
    exc: BaseException | None = None,
) -> MetaSynContextualFrontierRecoveryTerminalV3:
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
        "graph_construction_mechanics_authority": False,
        "extraction_accuracy_authority": False,
        "reliability_authority": False,
        "generalization_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierRecoveryTerminalV3.model_validate(
        {**payload, "terminal_sha256": hash_canonical(payload)}
    )


def prepare_metasyn_contextual_frontier_recovery_v3(
    *,
    repository_root: Path,
    workspace: Path = DEFAULT_WORKSPACE,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> MetaSynContextualFrontierRecoveryPlanV3:
    plan = freeze_metasyn_contextual_frontier_recovery_plan_v3(
        repository_root=repository_root, config_path=config_path
    )
    root = _fresh_workspace(workspace)
    with _workspace_lock(root):
        _persist_json(_checked_artifact(root, Path("00-prepared.json")), plan)
    return plan


def load_metasyn_contextual_frontier_recovery_plan_v3(
    *, workspace: Path
) -> MetaSynContextualFrontierRecoveryPlanV3:
    root = _existing_workspace(workspace)
    return MetaSynContextualFrontierRecoveryPlanV3.model_validate(
        _load_object(
            _checked_artifact(root, Path("00-prepared.json")),
            code="recovery_v3_prepared_invalid",
        )
    )


def authorize_metasyn_contextual_frontier_recovery_v3(
    *, workspace: Path, phase_budget_usd_micros: int
) -> MetaSynContextualFrontierRecoveryAuthorizationV3:
    root = _existing_workspace(workspace)
    with _workspace_lock(root):
        plan = load_metasyn_contextual_frontier_recovery_plan_v3(workspace=root)
        authorization = _freeze_authorization(
            plan=plan, phase_budget_usd_micros=phase_budget_usd_micros
        )
        path = _checked_artifact(root, Path("01-authorized.json"))
        if path.exists():
            observed = MetaSynContextualFrontierRecoveryAuthorizationV3.model_validate(
                _load_object(path, code="recovery_v3_authorization_invalid")
            )
            if observed != authorization:
                raise MetaSynContextualFrontierRecoveryV3Error(
                    "recovery_v3_authorization_replay_mismatch"
                )
            return observed
        _persist_json(path, authorization)
        return authorization


def execute_metasyn_contextual_frontier_recovery_v3(
    *,
    repository_root: Path,
    workspace: Path,
    client: MetaSynContextualFrontierRecoveryClientProtocolV3,
) -> MetaSynContextualFrontierRecoveryTerminalV3:
    root = _existing_workspace(workspace)
    repository = _root(repository_root)
    with _workspace_lock(root):
        plan = load_metasyn_contextual_frontier_recovery_plan_v3(workspace=root)
        authorization = MetaSynContextualFrontierRecoveryAuthorizationV3.model_validate(
            _load_object(
                _checked_artifact(root, Path("01-authorized.json")),
                code="recovery_v3_authorization_missing_or_invalid",
            )
        )
        intent = _freeze_intent(plan=plan, authorization=authorization)
        intent_path = _checked_artifact(root, Path("intent.json"))
        receipt_path = _checked_artifact(root, Path("provider-receipt.json"))
        terminal_path = _checked_artifact(root, Path("02-terminal.json"))
        if terminal_path.exists():
            return MetaSynContextualFrontierRecoveryTerminalV3.model_validate(
                _load_object(terminal_path, code="recovery_v3_terminal_invalid")
            )
        receipt: MetaSynContextualFrontierRecoveryReceiptV3 | None = None
        if intent_path.exists():
            saved_intent = MetaSynContextualFrontierRecoveryIntentV3.model_validate(
                _load_object(intent_path, code="recovery_v3_intent_invalid")
            )
            if saved_intent != intent:
                raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_intent_replay_mismatch")
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
            receipt = MetaSynContextualFrontierRecoveryReceiptV3.model_validate(
                _load_object(receipt_path, code="recovery_v3_receipt_invalid")
            )
        else:
            if receipt_path.exists():
                raise MetaSynContextualFrontierRecoveryV3Error("recovery_v3_receipt_without_intent")
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
            receipt = _freeze_receipt(
                plan=plan,
                authorization=authorization,
                intent=intent,
                result=result,
            )
            _persist_json(receipt_path, receipt)
        assert receipt is not None
        if receipt.provider_result.outcome != "completed":
            terminal = _freeze_terminal(
                status="provider_result_failed",
                plan=plan,
                authorization=authorization,
                intent=intent,
                receipt=receipt,
                failure_code=receipt.provider_result.failure_code or "provider_result_failed",
            )
        else:
            parsed = receipt.provider_result.parsed_json
            assert parsed is not None
            execution_binding = hash_canonical(
                {
                    "plan_sha256": plan.plan_sha256,
                    "authorization_sha256": authorization.authorization_sha256,
                    "intent_sha256": intent.intent_sha256,
                    "receipt_sha256": receipt.receipt_sha256,
                    "provider_result_sha256": receipt.provider_result_sha256,
                }
            )
            try:
                evaluation = evaluate_metasyn_contextual_frontier_recovery_response_v3(
                    repository_root=repository,
                    plan=plan,
                    raw_response=parsed,
                    provider_execution_binding_sha256=execution_binding,
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
                    status=evaluation.status,
                    plan=plan,
                    authorization=authorization,
                    intent=intent,
                    receipt=receipt,
                    evaluation=evaluation,
                )
        _persist_json(terminal_path, terminal)
        return terminal


def default_metasyn_contextual_frontier_recovery_client_v3(
    plan: MetaSynContextualFrontierRecoveryPlanV3,
) -> MetaSynContextualFrontierClientV1:
    return MetaSynContextualFrontierClientV1(plan.transport_config)
