from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from tests.private_cache_support import (
    HOSTED_ADAPTER_STALE_CODES,
    TYPED_PILOT_STALE_CODES,
    require_private_cache,
    skip_when_historical_replay_is_stale,
)

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_bounded_hosted_runtime import MetaSynHostedRuntimeError
from literature_multiverse.metasyn_passage_hosted_bundle_v2 import (
    EFFECT_KINDS,
    MAX_PROVIDER_CALLS,
    MetaSynPassageHostedBundleV2Error,
    MetaSynPassageHostedExecutionBundleV2,
    freeze_metasyn_passage_hosted_execution_bundle_v2,
    load_metasyn_passage_hosted_config_v2,
    validate_metasyn_passage_hosted_execution_bundle_v2,
)
from literature_multiverse.metasyn_typed_pilot import MetaSynTypedPilotError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle() -> MetaSynPassageHostedExecutionBundleV2:
    root = require_private_cache(
        "data/cache/metasyn/passage-hosted-yield-v2",
        "data/cache/metasyn/bounded-anthropic-yield-v5",
        "data/cache/metasyn/typed-oracle-pilot-v2",
    )
    return skip_when_historical_replay_is_stale(
        lambda: freeze_metasyn_passage_hosted_execution_bundle_v2(repository_root=root),
        stale_errors=(MetaSynTypedPilotError, MetaSynHostedRuntimeError),
        stale_codes=TYPED_PILOT_STALE_CODES | HOSTED_ADAPTER_STALE_CODES,
    )


def _property_values(value: Any, property_name: str, keyword: str) -> list[Any]:
    observed: list[Any] = []
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            property_schema = properties.get(property_name)
            if isinstance(property_schema, dict) and keyword in property_schema:
                observed.append(property_schema[keyword])
            if isinstance(property_schema, dict):
                observed.extend(_keyword_values(property_schema, keyword))
        for child in value.values():
            observed.extend(_property_values(child, property_name, keyword))
    elif isinstance(value, list):
        for child in value:
            observed.extend(_property_values(child, property_name, keyword))
    return observed


def _keyword_values(value: Any, keyword: str) -> list[Any]:
    observed: list[Any] = []
    if isinstance(value, dict):
        if keyword in value:
            observed.append(value[keyword])
        for child in value.values():
            observed.extend(_keyword_values(child, keyword))
    elif isinstance(value, list):
        for child in value:
            observed.extend(_keyword_values(child, keyword))
    return observed


def _first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    """Return the first canonical path for actionable replay diagnostics."""

    if type(left) is not type(right):
        return f"{path}:type:{type(left).__name__}!={type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path}:keys:{sorted(set(left) ^ set(right))}"
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}:length:{len(left)}!={len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(
                left_item,
                right_item,
                f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    return None if left == right else f"{path}:{left!r}!={right!r}"


def test_literal_config_freezes_spend_and_scientific_boundaries() -> None:
    config, file_sha256 = load_metasyn_passage_hosted_config_v2(repository_root=ROOT)

    assert len(file_sha256) == 64
    assert config.maximum_provider_calls == 296
    assert config.maximum_input_tokens_all_calls == 11_000_000
    assert config.maximum_authorized_cost_usd_micros == 210_000_000
    assert config.application_retries_per_request == 0
    assert config.sdk_retries_per_request == 0
    assert config.operator_authorized_source_transmission is True
    assert config.reference_fields_unopened is True
    assert config.official_test_labels_opened is False
    assert config.claim_release_authority is False
    assert config.inventory_smoke_row_ordinal == 0
    assert config.packet_smoke_priority_row_ordinal == 21
    assert config.packet_smoke_max_already_authorized_calls == 3
    assert (
        config.packet_abstention_semantics == "terminal_valid_but_does_not_pass_packet_spend_gate"
    )
    assert (
        config.truncation_or_max_tokens_disposition
        == "runtime_capacity_failure_not_scientific_abstention"
    )
    assert config.packet_accepted_canonical_json_utf8_byte_ceiling == 50_624
    assert config.packet_max_output_tokens == 65_536
    assert config.packet_max_output_tokens >= (
        config.packet_accepted_canonical_json_utf8_byte_ceiling
    )
    assert config.packet_accepted_max_identity_claims < (config.packet_native_max_identity_claims)
    assert config.packet_accepted_max_identity_text_characters < (
        config.packet_native_max_identity_text_characters
    )
    assert config.assembly_analysis_policy().analysis_policy_sha256 == (
        "a5e15982cf6c00760607385b7075478039261dc745c6dbf57891b679bc1756fb"
    )


def test_config_loader_rejects_nonliteral_path() -> None:
    with pytest.raises(
        MetaSynPassageHostedBundleV2Error,
        match="config_path_not_literal",
    ):
        load_metasyn_passage_hosted_config_v2(
            repository_root=ROOT,
            config_path=Path("configs/benchmarks/metasyn-bounded-anthropic-v1.json"),
        )


@pytest.mark.private_cache
def test_real_input_and_inventory_rosters_are_exact(
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    assert len(bundle.extraction_inputs.rows) == 32
    assert len(bundle.protocol_orientations) == 32
    assert len(bundle.inventory_requests) == 32
    assert [item.row_ordinal for item in bundle.inventory_requests] == list(range(32))
    assert all(
        item.request.schema_kind == "inventory"
        and item.request.effect_kind is None
        and item.request.transport_mode == "structured_json_schema"
        and item.request.max_output_tokens == 32_768
        and item.compiled_schema_record.wire_optional_parameter_count <= 24
        and item.compiled_schema_record.wire_union_parameter_count <= 16
        for item in bundle.inventory_requests
    )
    assert bundle.inventory_request_membership_sha256 == hash_canonical(
        [item.inventory_request_sha256 for item in bundle.inventory_requests]
    )
    for row, request in zip(
        bundle.extraction_inputs.rows,
        bundle.inventory_requests,
        strict=True,
    ):
        assert request.row_input_sha256 == row.row_input_sha256
        assert request.inventory_input_sha256 == row.inventory_input_sha256
        assert request.rendered_prompt_sha256 == (row.inventory_input.rendered_prompt_sha256)
        assert request.inventory_schema_bundle_sha256 == (
            row.inventory_input.inventory_schema_bundle_sha256
        )
    for row, frozen in zip(
        bundle.extraction_inputs.rows,
        bundle.protocol_orientations,
        strict=True,
    ):
        assert frozen.row_input_sha256 == row.row_input_sha256
        assert frozen.question_surface_sha256 == row.question_surface_sha256
        assert frozen.question_surface_question_spec_sha256 == (row.upstream_question_spec_sha256)
        assert frozen.protocol_question_spec_sha256 == frozen.protocol.question_spec_sha256
        assert frozen.protocol_question_spec_sha256 == (
            row.projection_v2.lineage_binding.question_spec_sha256
        )
        assert frozen.protocol_projection_spec_sha256 == (frozen.protocol.projection_spec_sha256)
        assert frozen.protocol_orientation.question_surface_question_spec_sha256 == (
            frozen.question_surface_question_spec_sha256
        )
        assert frozen.protocol_orientation.protocol_question_spec_sha256 == (
            frozen.protocol_question_spec_sha256
        )
        assert frozen.protocol_orientation.protocol_projection_spec_sha256 == (
            frozen.protocol_projection_spec_sha256
        )
        assert frozen.protocol_orientation.relation_kind == (row.question_surface.relation_kind)
        assert frozen.protocol_orientation.treatment_arm_role.value == (
            "intervention" if row.question_surface.relation_kind == "intervention" else "exposure"
        )
        assert frozen.protocol_orientation.comparator_arm_role.value == "comparator"
    assert {
        relation: sum(
            frozen.protocol_orientation.relation_kind == relation
            for frozen in bundle.protocol_orientations
        )
        for relation in ("intervention", "exposure")
    } == {"intervention": 22, "exposure": 10}


@pytest.mark.private_cache
def test_all_five_packet_schemas_compile_with_reduced_capacity_and_replay(
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    assert [item.effect_kind for item in bundle.packet_compiler_gates] == list(EFFECT_KINDS)
    assert [
        item.compiled_schema_record.wire_optional_parameter_count
        for item in bundle.packet_compiler_gates
    ] == [8, 8, 8, 8, 8]
    assert [
        item.compiled_schema_record.wire_union_parameter_count
        for item in bundle.packet_compiler_gates
    ] == [7, 7, 7, 7, 6]
    assert [
        item.accepted_canonical_json_utf8_byte_ceiling for item in bundle.packet_compiler_gates
    ] == [46_602, 46_602, 46_602, 46_220, 45_838]
    for gate in bundle.packet_compiler_gates:
        assert gate.accepted_canonical_json_utf8_byte_ceiling <= 50_624
        assert gate.compiled_schema_record.original_schema_sha256 == (
            gate.capacity_limited_schema_sha256
        )
        assert gate.compiled_schema_record.full_acceptance_schema_sha256 == (
            gate.capacity_limited_schema_sha256
        )
        assert 15 in _property_values(
            gate.capacity_limited_schema,
            "identity_claims",
            "maxItems",
        )
        assert 256 in _property_values(
            gate.capacity_limited_schema,
            "verbatim_identity_text",
            "maxLength",
        )

    reparsed = MetaSynPassageHostedExecutionBundleV2.model_validate(bundle.model_dump(mode="json"))
    assert reparsed.packet_compiler_gates == bundle.packet_compiler_gates


@pytest.mark.private_cache
def test_source_free_preflight_plan_has_three_plus_five_exact_calls(
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    plan = bundle.source_free_preflight_plan
    assert len(plan) == 8
    assert [item.preflight_ordinal for item in plan] == list(range(8))
    assert [item.schema_kind for item in plan] == ["inventory"] * 3 + ["packet"] * 5
    assert [item.effect_kind for item in plan[3:]] == list(EFFECT_KINDS)
    assert all(item.source_bearing is False for item in plan)
    assert all("source-free JSON compatibility probe" in item.request.prompt for item in plan)
    assert all(item.request.transport_mode == "structured_json_schema" for item in plan[:3])
    assert all(item.request.transport_mode == "prompt_json_schema" for item in plan[3:])
    assert [item.request.request_key for item in plan[3:]] == [
        f"preflight-{ordinal:02d}-packet-{effect_kind.replace('_', '-')}"
        for ordinal, effect_kind in enumerate(EFFECT_KINDS, start=3)
    ]
    assert bundle.preflight_membership_sha256 == hash_canonical(
        [item.preflight_call_sha256 for item in plan]
    )


@pytest.mark.private_cache
def test_packet_cost_probes_cover_every_row_effect_and_slot(
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    rows = bundle.packet_row_cost_envelopes
    assert len(rows) == 32
    assert [item.row_ordinal for item in rows] == list(range(32))
    assert all(item.authorized_slot_multiplicity == 8 for item in rows)
    assert all(
        [probe.effect_kind for probe in item.effect_probes] == list(EFFECT_KINDS) for item in rows
    )
    assert sum(item.authorized_slot_multiplicity for item in rows) == 256
    assert (
        max(probe.rendered_prompt_utf8_bytes for row in rows for probe in row.effect_probes)
        == 41_858
    )


@pytest.mark.private_cache
def test_global_cost_envelope_is_exact_and_inside_literal_limits(
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    envelope = bundle.global_cost_envelope
    assert envelope.maximum_provider_calls == MAX_PROVIDER_CALLS == 296
    assert [
        envelope.source_free_preflight.maximum_calls,
        envelope.inventory.maximum_calls,
        envelope.packet.maximum_calls,
    ] == [8, 32, 256]
    assert envelope.conservative_input_token_ceiling == 10_572_414
    assert envelope.max_output_token_ceiling == 18_251_776
    assert envelope.cost_ceiling_usd_micros == 203_662_588
    assert envelope.conservative_input_token_ceiling <= 11_000_000
    assert envelope.cost_ceiling_usd_micros <= 210_000_000


@pytest.mark.private_cache
def test_pipeline_fingerprint_is_dependency_closed(
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    assert len(bundle.bundle_pipeline_fingerprint.components) == 1
    component = bundle.bundle_pipeline_fingerprint.components[0]
    paths = {item.path for item in component.files}
    assert {
        "configs/benchmarks/metasyn-passage-hosted-anthropic-v2.json",
        "prompts/metasyn_candidate_inventory_v2.md",
        "prompts/metasyn_candidate_packet_v2.md",
        "src/literature_multiverse/metasyn_extraction_inputs_v2.py",
        "src/literature_multiverse/metasyn_passage_hosted_bundle_v2.py",
        "src/literature_multiverse/native_packet_assembly_v2.py",
        "src/literature_multiverse/native_packet_grounding_v2.py",
        "pyproject.toml",
        "uv.lock",
    }.issubset(paths)
    assert component.settings["provider_calls_permitted"] is False
    assert component.settings["official_test_labels_opened"] is False
    assert component.settings["reference_fields_unopened"] is True
    runtime_paths = {
        "src/literature_multiverse/metasyn_passage_hosted_runtime_v2.py",
        "scripts/run_metasyn_passage_hosted_runtime_v2.py",
    }
    runtime_exists = all((ROOT / path).is_file() for path in runtime_paths)
    assert component.settings["hosted_runtime_entrypoints_included"] is runtime_exists
    assert component.settings["hosted_runtime_entrypoints"] == (
        sorted(runtime_paths) if runtime_exists else []
    )
    if runtime_exists:
        assert runtime_paths.issubset(paths)


@pytest.mark.private_cache
def test_bundle_external_replay_is_byte_exact(
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    replayed = freeze_metasyn_passage_hosted_execution_bundle_v2(
        repository_root=ROOT,
        extraction_inputs=bundle.extraction_inputs,
    )
    difference = _first_difference(
        bundle.model_dump(mode="json"),
        replayed.model_dump(mode="json"),
    )
    assert difference is None, difference
    assert (
        validate_metasyn_passage_hosted_execution_bundle_v2(
            execution_bundle=bundle,
            repository_root=ROOT,
            external_replay=False,
        )
        == bundle
    )


@pytest.mark.private_cache
def test_bundle_self_hash_and_literal_policy_fail_closed(
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    tampered = bundle.model_dump(mode="json")
    tampered["execution_bundle_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="bundle_hash_mismatch"):
        MetaSynPassageHostedExecutionBundleV2.model_validate(tampered)

    tampered = bundle.model_dump(mode="json")
    tampered["claim_release_authority"] = True
    with pytest.raises(ValidationError):
        MetaSynPassageHostedExecutionBundleV2.model_validate(tampered)
