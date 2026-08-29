from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError as PydanticValidationError

from literature_multiverse.anthropic_bounded_generation import (
    ANTHROPIC_API_BASE_URL,
    ANTHROPIC_API_VERSION,
    ANTHROPIC_MODEL,
    ANTHROPIC_PRICING_TABLE_SHA256,
    ANTHROPIC_SCHEMA_COMPILER_VERSION,
    ANTHROPIC_SCHEMA_MAX_INLINED_DEPTH,
    ANTHROPIC_SCHEMA_MAX_INLINED_NODES,
    ANTHROPIC_SCHEMA_MAX_INLINED_UTF8_BYTES,
    ANTHROPIC_SCHEMA_MAX_NULLABLE_OPTIONAL_PROMOTIONS,
    ANTHROPIC_SCHEMA_MAX_OPTIONAL_PARAMETERS,
    ANTHROPIC_SCHEMA_MAX_REFERENCE_EXPANSIONS,
    ANTHROPIC_SCHEMA_MAX_UNION_PARAMETERS,
    ANTHROPIC_SCHEMA_NULLABLE_OPTIONAL_REQUIRED_TARGET,
    ANTHROPIC_SDK_VERSION,
    AnthropicBoundedClient,
    AnthropicBoundedConfigV1,
    AnthropicBoundedGenerationError,
    AnthropicCompiledSchemaV1,
    AnthropicEffectKind,
    AnthropicSchemaKind,
    adapt_anthropic_nullable_optional_properties,
    annotate_anthropic_literal_types,
    compile_anthropic_bounded_schema,
    compute_anthropic_request_cost_ceiling,
    count_anthropic_optional_parameters,
    count_anthropic_union_parameters,
    freeze_anthropic_bounded_request,
    freeze_anthropic_provider_identity,
    freeze_anthropic_wire_call_surface,
    inline_anthropic_local_references,
    project_anthropic_preflight_fixture,
)
from literature_multiverse.lineage import canonical_json_bytes, hash_canonical
from literature_multiverse.native_bounded_schema_v2 import (
    synthetic_schema_v2_preflight_specs,
)

_FULL_HASH = "1" * 64
_SIMPLE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "answer": {"enum": ["yes", "no"]},
        "count": {"const": 1},
    },
    "required": ["answer", "count"],
    "additionalProperties": False,
}
_ANY_OF_DEFS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:literature-multiverse:test:anyof-defs",
    "$defs": {
        "positive/value": {
            "type": "object",
            "properties": {"status": {"const": "positive"}},
            "required": ["status"],
            "additionalProperties": False,
        },
        "negative~value": {
            "type": "object",
            "properties": {"status": {"const": "negative"}},
            "required": ["status"],
            "additionalProperties": False,
        },
    },
    "anyOf": [
        {"$ref": "#/$defs/positive~1value"},
        {"$ref": "#/$defs/negative~0value"},
    ],
}
_OPTIONAL_NULLABLE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "additionalProperties": False,
}


class FakeMessages:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class FakeTransportError(Exception):
    status_code = 503
    request_id = "req_safe-123"


def _response(
    text: str = '{"answer":"yes","count":1}',
    *,
    model: str = ANTHROPIC_MODEL,
    stop_reason: str = "end_turn",
    response_id: str | None = "msg_test",
    content: list[object] | None = None,
    input_tokens: Any = 100,
    output_tokens: Any = 20,
    cache_creation_input_tokens: Any = 0,
    cache_read_input_tokens: Any = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=response_id,
        model=model,
        stop_reason=stop_reason,
        content=(
            content
            if content is not None
            else [
                SimpleNamespace(type="thinking", thinking="private"),
                SimpleNamespace(type="text", text=text),
            ]
        ),
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        ),
    )


def _frozen_inputs(
    *,
    response: object | None = None,
    error: Exception | None = None,
    schema: Mapping[str, Any] | None = None,
    schema_kind: AnthropicSchemaKind = "inventory",
    effect_kind: AnthropicEffectKind | None = None,
) -> tuple[
    AnthropicBoundedConfigV1,
    AnthropicCompiledSchemaV1,
    Any,
    FakeMessages,
    AnthropicBoundedClient,
]:
    config = AnthropicBoundedConfigV1(timeout_seconds=120)
    compiled = compile_anthropic_bounded_schema(
        original_schema=schema or _SIMPLE_SCHEMA,
        full_acceptance_schema_sha256=_FULL_HASH,
    )
    identity = freeze_anthropic_provider_identity(config)
    request = freeze_anthropic_bounded_request(
        operation="source-free-preflight",
        request_key="fixture-01",
        prompt="Return the fixture JSON.",
        system="Return exactly one JSON object matching the supplied schema.",
        max_output_tokens=256,
        compiled_schema=compiled,
        config=config,
        schema_kind=schema_kind,
        effect_kind=effect_kind,
        identity=identity,
    )
    messages = FakeMessages(response=response or _response(), error=error)
    client = AnthropicBoundedClient(config)
    # Test-only in-memory transport seam; production construction has no client
    # injection parameter and always builds the pinned first-party transport.
    client._client = SimpleNamespace(messages=messages)
    return config, compiled, request, messages, client


def test_all_eight_v2_preflight_schemas_compile_with_immutable_hashes() -> None:
    expected = {
        "00-inventory-no_candidate_found-v2": (
            "bfe29cae9a8c218aa43940dda66ec819c0aa4af3a7ac3622a1e963b9ac67fc2c",
            "82e9b56d265c2b3fddbb2722642afc2f7c8cf9f17aec0c1e1e90d60f2a7c0744",
            "919d4fd7febd4c79e79db58deb695e10ddf147d64cac45891f38d585e4f73c93",
            "eb50b2fdcadb243bdcbf6cb8c8e0dfdf4785ac15baf5a2b7d6a8a80d384cb170",
        ),
        "01-inventory-candidates_found-v2": (
            "bfe29cae9a8c218aa43940dda66ec819c0aa4af3a7ac3622a1e963b9ac67fc2c",
            "82e9b56d265c2b3fddbb2722642afc2f7c8cf9f17aec0c1e1e90d60f2a7c0744",
            "919d4fd7febd4c79e79db58deb695e10ddf147d64cac45891f38d585e4f73c93",
            "eb50b2fdcadb243bdcbf6cb8c8e0dfdf4785ac15baf5a2b7d6a8a80d384cb170",
        ),
        "02-inventory-overflow_or_uncertain-v2": (
            "bfe29cae9a8c218aa43940dda66ec819c0aa4af3a7ac3622a1e963b9ac67fc2c",
            "82e9b56d265c2b3fddbb2722642afc2f7c8cf9f17aec0c1e1e90d60f2a7c0744",
            "919d4fd7febd4c79e79db58deb695e10ddf147d64cac45891f38d585e4f73c93",
            "eb50b2fdcadb243bdcbf6cb8c8e0dfdf4785ac15baf5a2b7d6a8a80d384cb170",
        ),
        "03-packet-binary_group_statistics-v2": (
            "627cb7b55a16668b46d1ea3f33116c2c3d172524c9a3426eff3730bc60811db4",
            "b8576d1e525618f2dbf588443447e26d153b6e9736f850e2e0a5fa114906239c",
            "ec5353d9e46093d59eaded223bddbb5241fefe2f57a59cf43324a6e3415adebe",
            "d602bfc1730d43dca9cda4fd94b60786158d50198b6fdb8c87a4b698883bcad9",
        ),
        "04-packet-continuous_group_statistics-v2": (
            "e0027977e8d6169269fc822b49475c45e922f5cb12ce0ca684ccfc5e54dfdaaf",
            "8587bed5101dc330a42f6ccdcb19eb929c3a76ab3f364d8794b859e528bb4090",
            "1d74154e80eeef5ffdeb45210e02f6de624f4988d5f5b874165d4234dfe3313c",
            "c3841eb2f9209046195144b22fd6916702ce3aeb7623b8a9252e154d91696d60",
        ),
        "05-packet-direct_confidence_interval-v2": (
            "804822ab9533115b770ee6ac5b5a194160e7c00f3ea2ca15a1079d225c434c73",
            "badff11ca314bafaca74425a041db7bddfba7ea35f2635b642cfc69a0cbc2bcc",
            "2e27ca46b09d6abebb453e47316212df537009c6d252dc1abc152f77521d9d35",
            "0213f166bcb4d27962cfc5946566e4f074736301a1c5d3ec1e7ae1e139200ed2",
        ),
        "06-packet-direct_standard_error-v2": (
            "17764caa9cc126b94534252bf596768d297abe281e18c20ed7193caf4e69936b",
            "e261638a5f4f92e36cc9bc7ab1196c10ddb17f8870ed78ca6d7816ded66c3662",
            "195d5c5a7adb755350f7c8b18af2854d1315067693c90eacebb9a123e3387391",
            "ffeda512d6640d70762ac3ba15fa04d5f4a1edd31d5d5728e30b6a58cea12f27",
        ),
        "07-packet-direct_variance-v2": (
            "00ee7d2b6c9a5f69ce3b5543e7885034a7d9938ff06f10545f8319153f80b2e1",
            "32571fff43f9b3551c20d9f4a52be2ce86206a327833b5ca3629f567b2a43a01",
            "d6c8cf07d1c3bbbca0f40c298208a29cf0599cb2e34104fb8bb0d58312c4d4a2",
            "0e7fcf082398a4207798ab89de318d5392d2ed5a854b30a7eab52d1164745e8a",
        ),
    }
    specs = synthetic_schema_v2_preflight_specs()
    assert len(specs) == 8
    expected_optional_counts = {
        "00-inventory-no_candidate_found-v2": (0, 0, 0, 0, 0, 0),
        "01-inventory-candidates_found-v2": (0, 0, 0, 0, 0, 0),
        "02-inventory-overflow_or_uncertain-v2": (0, 0, 0, 0, 0, 0),
        "03-packet-binary_group_statistics-v2": (25, 10, 8, 15, 19, 11),
        "04-packet-continuous_group_statistics-v2": (26, 10, 9, 16, 20, 11),
        "05-packet-direct_confidence_interval-v2": (26, 10, 9, 16, 20, 11),
        "06-packet-direct_standard_error-v2": (26, 10, 9, 16, 20, 11),
        "07-packet-direct_variance-v2": (26, 10, 9, 16, 20, 11),
    }

    for spec in specs:
        original_before = deepcopy(spec["provider_schema"])
        compiled = compile_anthropic_bounded_schema(
            original_schema=spec["provider_schema"],
            full_acceptance_schema_sha256=spec["full_acceptance_schema_sha256"],
        )
        assert spec["provider_schema"] == original_before
        Draft202012Validator.check_schema(compiled.original_schema)
        Draft202012Validator.check_schema(compiled.literal_annotated_schema)
        Draft202012Validator.check_schema(compiled.wire_schema)
        wire_bytes = canonical_json_bytes(compiled.wire_schema)
        assert b'"$defs"' not in wire_bytes
        assert b'"$ref"' not in wire_bytes
        assert b"#/$defs/" not in wire_bytes
        assert b"#/definitions/" not in wire_bytes
        before, promoted, stripped, after, pre_union, post_union = (
            expected_optional_counts[spec["call_id"]]
        )
        assert compiled.pre_promotion_optional_parameter_count == before
        assert compiled.nullable_optional_promotion_count == promoted
        assert compiled.nullable_optional_null_stripping_count == stripped
        assert compiled.post_promotion_optional_parameter_count == after
        assert compiled.wire_optional_parameter_count == after
        assert compiled.pre_adaptation_union_parameter_count == pre_union
        assert compiled.post_adaptation_union_parameter_count == post_union
        assert compiled.wire_union_parameter_count == post_union
        assert after <= ANTHROPIC_SCHEMA_MAX_OPTIONAL_PARAMETERS
        assert post_union <= ANTHROPIC_SCHEMA_MAX_UNION_PARAMETERS
        assert len(compiled.nullable_optional_promotion_paths) == len(
            set(compiled.nullable_optional_promotion_paths)
        )
        assert compiled.nullable_optional_promotion_paths_sha256 == hash_canonical(
            compiled.nullable_optional_promotion_paths
        )
        assert compiled.nullable_optional_null_stripping_paths_sha256 == hash_canonical(
            compiled.nullable_optional_null_stripping_paths
        )
        assert compiled.nullable_optional_candidate_paths == sorted(
            compiled.nullable_optional_promotion_paths
            + compiled.nullable_optional_null_stripping_paths
        )
        assert [
            proof.path for proof in compiled.nullable_optional_adaptation_proofs
        ] == (
            compiled.nullable_optional_promotion_paths
            + compiled.nullable_optional_null_stripping_paths
        )
        assert (
            compiled.original_schema_sha256,
            compiled.literal_annotated_schema_sha256,
            compiled.wire_schema_sha256,
            compiled.compiled_schema_sha256,
        ) == expected[spec["call_id"]]
        Draft202012Validator(compiled.original_schema).validate(spec["valid_example"])
        Draft202012Validator(compiled.literal_annotated_schema).validate(
            spec["valid_example"]
        )


def test_v6_retains_v4_ref_inlining_without_changing_schema_authority() -> None:
    original_before = deepcopy(_ANY_OF_DEFS_SCHEMA)
    inlined = inline_anthropic_local_references(_ANY_OF_DEFS_SCHEMA)
    compiled = compile_anthropic_bounded_schema(
        original_schema=_ANY_OF_DEFS_SCHEMA,
        full_acceptance_schema_sha256=_FULL_HASH,
    )

    assert ANTHROPIC_SCHEMA_COMPILER_VERSION == "anthropic-literal-type-compiler-v7"
    assert original_before == _ANY_OF_DEFS_SCHEMA
    assert "$defs" in compiled.original_schema
    assert "$defs" in compiled.literal_annotated_schema
    assert "$defs" not in inlined
    assert inlined["$id"] == _ANY_OF_DEFS_SCHEMA["$id"]
    assert '"$defs"' not in json.dumps(compiled.wire_schema)
    assert '"$ref"' not in json.dumps(compiled.wire_schema)
    assert "anyOf" in compiled.wire_schema
    assert compiled.wire_schema_scientific_authority == "none"
    assert compiled.local_response_validation_schema == "original_schema"
    assert compiled.full_acceptance_schema_sha256 == _FULL_HASH

    original_validator = Draft202012Validator(_ANY_OF_DEFS_SCHEMA)
    inlined_validator = Draft202012Validator(inlined)
    for candidate in (
        {"status": "positive"},
        {"status": "negative"},
        {"status": "unknown"},
        {"status": "positive", "extra": True},
        {},
        None,
    ):
        assert original_validator.is_valid(candidate) == inlined_validator.is_valid(
            candidate
        )

    replayed = AnthropicCompiledSchemaV1.model_validate_json(
        compiled.model_dump_json()
    )
    assert replayed.compiled_schema_sha256 == compiled.compiled_schema_sha256


def test_v4_reference_audit_ignores_instance_literals_and_property_names() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "$ref": {"type": "string"},
            "literal": {
                "const": {"$ref": "this-is-instance-data-not-a-schema-reference"}
            },
        },
        "required": ["$ref", "literal"],
        "additionalProperties": False,
    }
    compiled = compile_anthropic_bounded_schema(
        original_schema=schema,
        full_acceptance_schema_sha256=_FULL_HASH,
    )

    Draft202012Validator(compiled.original_schema).validate(
        {
            "$ref": "ordinary property value",
            "literal": {"$ref": "this-is-instance-data-not-a-schema-reference"},
        }
    )
    assert compiled.wire_schema


def test_v6_hybrid_nullable_policy_projects_only_preflight_fixtures() -> None:
    expected_promotions = [0, 0, 0, 10, 10, 10, 10, 10]
    expected_stripped = [0, 0, 0, 8, 9, 9, 9, 9]

    def without_null_values(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: without_null_values(item)
                for key, item in value.items()
                if item is not None
            }
        if isinstance(value, list):
            return [without_null_values(item) for item in value]
        return value

    for spec, expected_promoted, expected_removed in zip(
        synthetic_schema_v2_preflight_specs(),
        expected_promotions,
        expected_stripped,
        strict=True,
    ):
        inlined = inline_anthropic_local_references(
            annotate_anthropic_literal_types(spec["provider_schema"])
        )
        adapted = adapt_anthropic_nullable_optional_properties(inlined)
        compiled = compile_anthropic_bounded_schema(
            original_schema=spec["provider_schema"],
            full_acceptance_schema_sha256=spec["full_acceptance_schema_sha256"],
        )
        dematerialized = without_null_values(spec["valid_example"])
        projected_from_missing = project_anthropic_preflight_fixture(
            value=dematerialized,
            original_schema=spec["provider_schema"],
        )
        projected_from_explicit_null = project_anthropic_preflight_fixture(
            value=spec["valid_example"],
            original_schema=spec["provider_schema"],
        )

        assert compiled.nullable_optional_promotion_count == expected_promoted
        assert compiled.nullable_optional_null_stripping_count == expected_removed
        assert count_anthropic_optional_parameters(adapted) == (
            compiled.post_promotion_optional_parameter_count
        )
        assert Draft202012Validator(spec["provider_schema"]).is_valid(dematerialized)
        assert projected_from_missing == projected_from_explicit_null
        for projected in (projected_from_missing, projected_from_explicit_null):
            assert Draft202012Validator(spec["provider_schema"]).is_valid(projected)
            assert Draft202012Validator(spec["full_acceptance_schema"]).is_valid(
                projected
            )
            assert Draft202012Validator(adapted).is_valid(projected)
            assert Draft202012Validator(compiled.wire_schema).is_valid(projected)
        assert str(projected_from_missing).count("None") == expected_promoted

    packet_compiled = [
        compile_anthropic_bounded_schema(
            original_schema=spec["provider_schema"],
            full_acceptance_schema_sha256=spec["full_acceptance_schema_sha256"],
        )
        for spec in synthetic_schema_v2_preflight_specs()[3:]
    ]
    expected_common_suffixes = (
        "/properties/contrast/properties/estimand",
        "/properties/effect/properties/equivalence_margin",
        "/properties/effect/properties/reported_p_value",
        "/properties/finding/properties/analysis_population",
        "/properties/finding/properties/timepoint/properties/value",
        "/properties/finding/properties/timepoint/properties/lower",
        "/properties/finding/properties/timepoint/properties/upper",
        "/properties/finding/properties/timepoint/properties/unit",
        "/properties/finding/properties/timepoint/properties/raw_label",
    )
    for index, compiled in enumerate(packet_compiled):
        expected_last = (
            "/properties/finding/properties/timepoint/properties/anchor"
            if index == 0
            else "/properties/effect/properties/unit"
        )
        assert all(
            path.endswith(suffix)
            for path, suffix in zip(
                compiled.nullable_optional_promotion_paths,
                (*expected_common_suffixes, expected_last),
                strict=True,
            )
        )


def test_v6_exact_optional_and_union_regression_preserves_all_fields() -> None:
    nullable_properties = {
        f"field_{index:02d}": {
            "anyOf": [{"type": "string"}, {"type": "null"}]
        }
        for index in range(25)
    }
    nullable_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": nullable_properties,
        "additionalProperties": False,
    }
    compiled = compile_anthropic_bounded_schema(
        original_schema=nullable_schema,
        full_acceptance_schema_sha256=_FULL_HASH,
    )
    projected = project_anthropic_preflight_fixture(
        value={}, original_schema=nullable_schema
    )

    assert compiled.pre_promotion_optional_parameter_count == 25
    assert compiled.nullable_optional_promotion_count == 10
    assert compiled.nullable_optional_null_stripping_count == 15
    assert compiled.post_promotion_optional_parameter_count == 15
    assert compiled.wire_optional_parameter_count == 15
    assert compiled.wire_union_parameter_count == 10
    assert set(projected) == {f"field_{index:02d}" for index in range(10)}
    assert set(projected.values()) == {None}
    assert set(compiled.wire_schema["properties"]) == set(nullable_properties)
    assert Draft202012Validator(nullable_schema).is_valid(projected)
    assert Draft202012Validator(compiled.wire_schema).is_valid(projected)

    nonnullable_schema = deepcopy(nullable_schema)
    nonnullable_schema["properties"] = {
        name: {"type": "string"} for name in nullable_properties
    }
    with pytest.raises(
        AnthropicBoundedGenerationError,
        match="optional_parameter_limit_exceeded:25",
    ):
        compile_anthropic_bounded_schema(
            original_schema=nonnullable_schema,
            full_acceptance_schema_sha256=_FULL_HASH,
        )


def test_v6_null_stripping_supported_subset_and_nested_paths_fail_closed() -> None:
    def schema_with_last(last_schema: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                **{
                    f"field_{index:02d}": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    }
                    for index in range(10)
                },
                "field_10": deepcopy(dict(last_schema)),
            },
            "additionalProperties": False,
        }

    for supported in (
        {"anyOf": [{"type": "null"}, {"type": "string", "minLength": 1}]},
        {"oneOf": [{"type": "null"}, {"type": "string", "minLength": 1}]},
        {"type": ["string", "null"], "minLength": 1},
    ):
        schema = schema_with_last(supported)
        compiled = compile_anthropic_bounded_schema(
            original_schema=schema,
            full_acceptance_schema_sha256=_FULL_HASH,
        )
        projected = project_anthropic_preflight_fixture(
            value={"field_10": None}, original_schema=schema
        )
        projected_nonnull = project_anthropic_preflight_fixture(
            value={"field_10": "kept"}, original_schema=schema
        )
        assert "field_10" not in projected
        assert projected_nonnull["field_10"] == "kept"
        assert Draft202012Validator(schema).is_valid(projected)
        assert Draft202012Validator(compiled.wire_schema).is_valid(projected)
        assert Draft202012Validator(compiled.wire_schema).is_valid(projected_nonnull)
        proof = compiled.nullable_optional_adaptation_proofs[-1]
        assert proof.action == "keep_optional_strip_null"
        assert proof.accepts_null_before is True
        assert proof.accepts_null_after is False

    for unsupported in (
        {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "minLength": 1,
        },
        {
            "anyOf": [
                {"type": ["string", "null"]},
                {"type": "null"},
            ]
        },
        {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"},
                {"type": "null"},
            ]
        },
        {"type": ["string", "integer", "null"]},
    ):
        with pytest.raises(AnthropicBoundedGenerationError, match="nullable_strip"):
            compile_anthropic_bounded_schema(
                original_schema=schema_with_last(unsupported),
                full_acceptance_schema_sha256=_FULL_HASH,
            )

    nested_nullable = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "nested": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    }
                },
                "additionalProperties": False,
            },
            {"type": "null"},
        ]
    }
    nested_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            **{
                f"a_{index:02d}": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                }
                for index in range(10)
            },
            "z_container": nested_nullable,
        },
        "additionalProperties": False,
    }
    nested_compiled = compile_anthropic_bounded_schema(
        original_schema=nested_schema,
        full_acceptance_schema_sha256=_FULL_HASH,
    )
    assert nested_compiled.nullable_optional_null_stripping_count == 2
    assert Draft202012Validator(nested_compiled.wire_schema).is_valid({
        f"a_{index:02d}": None for index in range(10)
    })


def test_v6_enforces_exact_provider_optional_and_union_cap_boundaries() -> None:
    union_property = {"anyOf": [{"type": "string"}, {"type": "integer"}]}

    def required_union_schema(count: int) -> dict[str, Any]:
        names = [f"union_{index:02d}" for index in range(count)]
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {name: deepcopy(union_property) for name in names},
            "required": names,
            "additionalProperties": False,
        }

    boundary = compile_anthropic_bounded_schema(
        original_schema=required_union_schema(16),
        full_acceptance_schema_sha256=_FULL_HASH,
    )
    assert boundary.wire_union_parameter_count == 16
    with pytest.raises(
        AnthropicBoundedGenerationError,
        match="union_parameter_limit_exceeded:17",
    ):
        compile_anthropic_bounded_schema(
            original_schema=required_union_schema(17),
            full_acceptance_schema_sha256=_FULL_HASH,
        )

    optional_boundary = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            f"optional_{index:02d}": {"type": "string"} for index in range(24)
        },
        "additionalProperties": False,
    }
    assert compile_anthropic_bounded_schema(
        original_schema=optional_boundary,
        full_acceptance_schema_sha256=_FULL_HASH,
    ).wire_optional_parameter_count == 24
    assert count_anthropic_union_parameters(
        {
            "anyOf": [{"type": "string"}, {"type": "integer"}],
            "properties": {
                "nested": {
                    "oneOf": [{"type": "string"}, {"type": "integer"}]
                }
            },
        }
    ) == 1


def test_v4_inlining_is_validation_equivalent_on_all_eight_frozen_schemas() -> None:
    for spec in synthetic_schema_v2_preflight_specs():
        original = spec["provider_schema"]
        inlined = inline_anthropic_local_references(
            annotate_anthropic_literal_types(original)
        )
        original_validator = Draft202012Validator(original)
        inlined_validator = Draft202012Validator(inlined)
        valid = deepcopy(spec["valid_example"])
        candidates: list[Any] = [valid, None, {}, [], "invalid-root-type"]
        if isinstance(valid, dict):
            with_extra = deepcopy(valid)
            with_extra["v4_unexpected_root_property"] = True
            candidates.append(with_extra)
            if valid:
                without_first = deepcopy(valid)
                without_first.pop(sorted(valid)[0])
                candidates.append(without_first)
        for candidate in candidates:
            assert original_validator.is_valid(
                candidate
            ) == inlined_validator.is_valid(candidate)


@pytest.mark.parametrize(
    ("schema", "error"),
    [
        (
            {
                "$defs": {"loop": {"$ref": "#/$defs/loop"}},
                "anyOf": [{"$ref": "#/$defs/loop"}],
            },
            "reference_cycle",
        ),
        (
            {
                "$defs": {
                    "a": {"$ref": "#/$defs/b"},
                    "b": {"$ref": "#/$defs/a"},
                },
                "anyOf": [{"$ref": "#/$defs/a"}],
            },
            "reference_cycle",
        ),
        (
            {"anyOf": [{"$ref": "#/$defs/missing"}]},
            "reference_unresolved",
        ),
        (
            {
                "$defs": {"value": {"type": "string"}},
                "anyOf": [{"$ref": "#/$defs/value", "type": "string"}],
            },
            "siblings_forbidden",
        ),
        (
            {"anyOf": [{"$dynamicRef": "#/$defs/value"}]},
            "dynamic_or_recursive_reference_forbidden",
        ),
        (
            {
                "$defs": {
                    "value": {
                        "$id": "urn:scope-change",
                        "type": "string",
                    }
                },
                "anyOf": [{"$ref": "#/$defs/value"}],
            },
            "nested_reference_scope_forbidden",
        ),
        (
            {
                "$defs": {"unused": {"$ref": "#/$defs/unused"}},
                "type": "string",
            },
            "reference_cycle",
        ),
        (
            {
                "$defs": {
                    "value": {"type": "string"},
                    "unused": {"$ref": "#/$defs/value", "description": "unsafe"},
                },
                "type": "string",
            },
            "siblings_forbidden",
        ),
        (
            {
                "$defs": {
                    "unused": {"$recursiveAnchor": True, "type": "string"}
                },
                "type": "string",
            },
            "nested_reference_scope_forbidden",
        ),
    ],
)
def test_v4_reference_inlining_rejects_unsafe_graphs(
    schema: dict[str, Any], error: str
) -> None:
    with pytest.raises(AnthropicBoundedGenerationError, match=error):
        inline_anthropic_local_references(schema)


def test_v4_pointer_decoding_and_expansion_limits_are_fail_closed() -> None:
    with pytest.raises(AnthropicBoundedGenerationError, match="pointer_escape_invalid"):
        inline_anthropic_local_references(
            {
                "$defs": {"bad~2escape": {"type": "string"}},
                "anyOf": [{"$ref": "#/$defs/bad~2escape"}],
            }
        )
    with pytest.raises(AnthropicBoundedGenerationError, match="percent_encoding"):
        inline_anthropic_local_references(
            {
                "$defs": {"encoded/name": {"type": "string"}},
                "anyOf": [{"$ref": "#/$defs/encoded%2Fname"}],
            }
        )

    with (
        patch(
            "literature_multiverse.anthropic_bounded_generation."
            "ANTHROPIC_SCHEMA_MAX_REFERENCE_EXPANSIONS",
            1,
        ),
        pytest.raises(
            AnthropicBoundedGenerationError, match="expansion_limit_exceeded"
        ),
    ):
        inline_anthropic_local_references(
            {
                "$defs": {"value": {"type": "string"}},
                "anyOf": [
                    {"$ref": "#/$defs/value"},
                    {"$ref": "#/$defs/value"},
                ],
            }
        )

    exactly_two = {
        "$defs": {"value": {"type": "string"}},
        "anyOf": [
            {"$ref": "#/$defs/value"},
            {"$ref": "#/$defs/value"},
        ],
    }
    with patch(
        "literature_multiverse.anthropic_bounded_generation."
        "ANTHROPIC_SCHEMA_MAX_REFERENCE_EXPANSIONS",
        2,
    ):
        assert inline_anthropic_local_references(exactly_two)["anyOf"] == [
            {"type": "string"},
            {"type": "string"},
        ]

    with (
        patch(
            "literature_multiverse.anthropic_bounded_generation."
            "ANTHROPIC_SCHEMA_MAX_INLINED_NODES",
            2,
        ),
        pytest.raises(AnthropicBoundedGenerationError, match="node_limit_exceeded"),
    ):
        inline_anthropic_local_references(
            {
                "type": "object",
                "properties": {
                    "one": {"type": "string"},
                    "two": {"type": "string"},
                },
            }
        )

    with (
        patch(
            "literature_multiverse.anthropic_bounded_generation."
            "ANTHROPIC_SCHEMA_MAX_INLINED_UTF8_BYTES",
            64,
        ),
        pytest.raises(
            AnthropicBoundedGenerationError, match="input_byte_limit_exceeded"
        ),
    ):
        inline_anthropic_local_references(
            {"type": "string", "description": "x" * 100}
        )

    identity = freeze_anthropic_provider_identity(
        AnthropicBoundedConfigV1(timeout_seconds=120)
    )
    assert identity.schema_max_reference_expansions == (
        ANTHROPIC_SCHEMA_MAX_REFERENCE_EXPANSIONS
    )
    assert identity.schema_max_inlined_nodes == ANTHROPIC_SCHEMA_MAX_INLINED_NODES
    assert identity.schema_max_inlined_depth == ANTHROPIC_SCHEMA_MAX_INLINED_DEPTH
    assert identity.schema_max_inlined_utf8_bytes == (
        ANTHROPIC_SCHEMA_MAX_INLINED_UTF8_BYTES
    )
    assert identity.schema_max_nullable_optional_promotions == (
        ANTHROPIC_SCHEMA_MAX_NULLABLE_OPTIONAL_PROMOTIONS
    )
    assert identity.schema_max_optional_parameters == (
        ANTHROPIC_SCHEMA_MAX_OPTIONAL_PARAMETERS
    )
    assert identity.schema_nullable_optional_required_target == (
        ANTHROPIC_SCHEMA_NULLABLE_OPTIONAL_REQUIRED_TARGET
    )
    assert identity.schema_max_union_parameters == ANTHROPIC_SCHEMA_MAX_UNION_PARAMETERS
    assert identity.response_schema_validation_order == (
        "wire-then-original-then-runtime-full-acceptance"
    )


def test_literal_compiler_adds_only_implied_types_and_rejects_heterogeneity() -> None:
    original = {
        "oneOf": [
            {"const": None},
            {"enum": [1, 2, 3]},
            {"enum": ["a", "b"]},
            {"type": "string", "enum": ["fixed"]},
        ]
    }
    before = deepcopy(original)
    annotated = annotate_anthropic_literal_types(original)

    assert original == before
    assert [branch["type"] for branch in annotated["oneOf"]] == [
        "null",
        "integer",
        "string",
        "string",
    ]
    without_annotations = deepcopy(annotated)
    for branch in without_annotations["oneOf"][:3]:
        branch.pop("type")
    assert without_annotations == original

    for bad in (
        {"enum": [1, "1"]},
        {"enum": [1, 1.0]},
        {"const": 1, "enum": ["1"]},
        {"enum": []},
    ):
        with pytest.raises(AnthropicBoundedGenerationError):
            annotate_anthropic_literal_types(bad)


def test_compiler_rejects_invalid_schema_and_hash_and_contract_is_frozen() -> None:
    with pytest.raises(AnthropicBoundedGenerationError, match="original_schema_invalid"):
        compile_anthropic_bounded_schema(
            original_schema={"type": "definitely-not-a-json-type"},
            full_acceptance_schema_sha256=_FULL_HASH,
        )
    with pytest.raises(AnthropicBoundedGenerationError, match="hash_invalid"):
        compile_anthropic_bounded_schema(
            original_schema=_SIMPLE_SCHEMA,
            full_acceptance_schema_sha256="not-a-hash",
        )

    compiled = compile_anthropic_bounded_schema(
        original_schema=_SIMPLE_SCHEMA,
        full_acceptance_schema_sha256=_FULL_HASH,
    )
    with pytest.raises(PydanticValidationError):
        compiled.wire_schema_sha256 = "2" * 64  # type: ignore[misc]

    for unresolved in (
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "#/$defs/missing",
        },
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "https://schemas.invalid/remote.json",
        },
    ):
        with pytest.raises(AnthropicBoundedGenerationError, match="reference"):
            compile_anthropic_bounded_schema(
                original_schema=unresolved,
                full_acceptance_schema_sha256=_FULL_HASH,
            )


def test_coherently_rehashed_compiled_schema_must_replay_exact_compiler() -> None:
    compiled = compile_anthropic_bounded_schema(
        original_schema=_SIMPLE_SCHEMA,
        full_acceptance_schema_sha256=_FULL_HASH,
    )
    payload = compiled.model_dump(mode="json")
    payload["wire_schema"]["description"] = "coherently forged provider grammar"
    payload["wire_schema_sha256"] = hash_canonical(payload["wire_schema"])

    with pytest.raises(PydanticValidationError, match="wire_schema_replay_mismatch"):
        AnthropicCompiledSchemaV1.model_validate(payload)

    promoted = compile_anthropic_bounded_schema(
        original_schema=_OPTIONAL_NULLABLE_SCHEMA,
        full_acceptance_schema_sha256=_FULL_HASH,
    )
    payload = promoted.model_dump(mode="json")
    payload["nullable_optional_promotion_paths"] = []
    payload["nullable_optional_promotion_paths_sha256"] = hash_canonical([])
    payload["nullable_optional_promotion_count"] = 0
    with pytest.raises(PydanticValidationError, match="nullable_"):
        AnthropicCompiledSchemaV1.model_validate(payload)

    hybrid_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            f"field_{index:02d}": {
                "anyOf": [{"type": "string"}, {"type": "null"}]
            }
            for index in range(11)
        },
        "additionalProperties": False,
    }
    hybrid = compile_anthropic_bounded_schema(
        original_schema=hybrid_schema,
        full_acceptance_schema_sha256=_FULL_HASH,
    )
    payload = hybrid.model_dump(mode="json")
    payload["nullable_optional_null_stripping_paths"] = []
    payload["nullable_optional_null_stripping_paths_sha256"] = hash_canonical([])
    payload["nullable_optional_null_stripping_count"] = 0
    with pytest.raises(PydanticValidationError, match="nullable_"):
        AnthropicCompiledSchemaV1.model_validate(payload)


def test_default_client_constructor_pins_sdk_zero_retries_and_timeout() -> None:
    config = AnthropicBoundedConfigV1(timeout_seconds=123)
    client = AnthropicBoundedClient(config)
    sentinel = object()
    http_sentinel = object()

    with (
        patch(
            "anthropic.DefaultHttpxClient", return_value=http_sentinel
        ) as http_constructor,
        patch("anthropic.Anthropic", return_value=sentinel) as constructor,
    ):
        assert client._client_or_create() is sentinel

    http_constructor.assert_called_once_with(
        timeout=123.0, trust_env=False, follow_redirects=False
    )
    constructor.assert_called_once_with(
        base_url=ANTHROPIC_API_BASE_URL,
        default_headers={"anthropic-version": ANTHROPIC_API_VERSION},
        http_client=http_sentinel,
        max_retries=0,
        timeout=123.0,
    )
    assert client._client_or_create() is sentinel
    assert client.identity.anthropic_sdk_version == ANTHROPIC_SDK_VERSION
    assert client.identity.transport_attempts_per_request == 1
    assert client.identity.application_retry_count == 0


def test_success_makes_exactly_one_call_and_forwards_only_frozen_controls() -> None:
    config, compiled, request, messages, client = _frozen_inputs()

    result = client.generate(request)

    assert result.outcome == "completed"
    assert result.parsed_json == {"answer": "yes", "count": 1}
    assert result.response_model == ANTHROPIC_MODEL
    assert result.usage is not None and result.usage.input_tokens == 100
    assert result.cost.estimated_cost_usd == Decimal("0.0004")
    assert len(messages.calls) == 1
    call = messages.calls[0]
    assert call == {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 256,
        "system": request.system,
        "messages": [{"role": "user", "content": request.prompt}],
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": compiled.wire_schema},
        },
        "service_tier": "standard_only",
    }
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(call)
    assert result.wire_call_sha256 == hash_canonical(
        {
            "api_base_url": ANTHROPIC_API_BASE_URL,
            "anthropic_api_version": ANTHROPIC_API_VERSION,
            "environment_transport_overrides_permitted": False,
            "http_environment_trust": False,
            "follow_redirects": False,
            "transport_mode": "structured_json_schema",
            "request_kwargs": call,
        }
    )
    assert result.sdk_retry_count == 0
    assert result.transport_attempt_count == 1
    assert config.thinking_mode == "provider_default_adaptive"


def test_packet_prompt_json_mode_embeds_one_schema_and_omits_format() -> None:
    config, compiled, request, messages, client = _frozen_inputs(
        schema_kind="packet",
        effect_kind="binary_group_statistics",
    )

    result = client.generate(request)

    assert result.outcome == "completed"
    assert request.transport_mode == "prompt_json_schema"
    assert request.prompt == request.model_prompt
    assert request.system != request.model_system
    schema_json = canonical_json_bytes(compiled.wire_schema).decode("utf-8")
    assert request.model_system.endswith(schema_json)
    assert request.model_system.count(schema_json) == 1
    assert request.cost_ceiling.structured_format_schema_utf8_bytes == 0
    assert request.cost_ceiling.embedded_system_schema_utf8_bytes == len(
        schema_json.encode("utf-8")
    )
    assert request.cost_ceiling.model_facing_input_utf8_bytes == (
        len(request.model_system.encode("utf-8"))
        + len(request.model_prompt.encode("utf-8"))
    )
    kwargs, wire_call_sha256 = freeze_anthropic_wire_call_surface(
        request=request,
        config=config,
    )
    assert kwargs == messages.calls[0]
    assert kwargs["output_config"] == {"effort": "low"}
    assert "format" not in kwargs["output_config"]
    assert kwargs["messages"] == [{"role": "user", "content": request.prompt}]
    assert wire_call_sha256 == request.expected_wire_call_sha256
    assert result.wire_call_sha256 == wire_call_sha256
    assert result.response_text_sha256 is not None
    assert result.parsed_json_sha256 == hash_canonical(result.parsed_json)


def test_coherently_rehashed_wrong_transport_mode_fails_closed() -> None:
    _, _, request, messages, client = _frozen_inputs(
        schema_kind="packet",
        effect_kind="direct_variance",
    )
    payload = request.model_dump(mode="json")
    payload["transport_mode"] = "structured_json_schema"
    payload["request_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "request_sha256"}
    )

    with pytest.raises(PydanticValidationError, match="transport_mode_drift"):
        client.generate(type(request).model_validate(payload))
    assert messages.calls == []


def test_response_validation_is_wire_then_original_without_post_fill() -> None:
    _, _, request, messages, client = _frozen_inputs(
        schema=_OPTIONAL_NULLABLE_SCHEMA,
        response=_response(text="{}"),
    )
    missing_required_null = client.generate(request)

    assert missing_required_null.outcome == "response_wire_schema_invalid"
    assert missing_required_null.failure is not None
    assert missing_required_null.failure.detail == "response_json_failed_wire_schema"
    assert missing_required_null.parsed_json == {}
    assert len(messages.calls) == 1

    _, _, request, messages, client = _frozen_inputs(
        schema=_OPTIONAL_NULLABLE_SCHEMA,
        response=_response(text='{"note":null}'),
    )
    explicit_null = client.generate(request)
    assert explicit_null.outcome == "completed"
    assert explicit_null.parsed_json == {"note": None}
    assert len(messages.calls) == 1

    overlapping_one_of = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "oneOf": [
            {"type": "number"},
            {"type": "number", "minimum": 0},
        ],
    }
    _, _, request, messages, client = _frozen_inputs(
        schema=overlapping_one_of,
        response=_response(text="1"),
    )
    original_failure = client.generate(request)
    assert Draft202012Validator(request.compiled_schema.wire_schema).is_valid(1)
    assert not Draft202012Validator(request.compiled_schema.original_schema).is_valid(1)
    assert original_failure.outcome == "response_schema_invalid"
    assert original_failure.failure is not None
    assert (
        original_failure.failure.detail
        == "response_json_failed_original_provider_schema"
    )
    assert len(messages.calls) == 1


@pytest.mark.parametrize(
    ("response", "outcome"),
    [
        (_response(response_id=None), "response_identity_invalid"),
        (_response(model="claude-wrong-model"), "response_model_mismatch"),
        (_response(stop_reason="max_tokens"), "response_stop_reason_invalid"),
        (
            _response(content=[SimpleNamespace(type="tool_use", input={})]),
            "response_content_invalid",
        ),
        (_response(text="not-json"), "response_json_invalid"),
        (
            _response(text='{"answer":"yes","count":2}'),
            "response_schema_invalid",
        ),
        (_response(text='{"answer":"yes","answer":"no","count":1}'), "response_json_invalid"),
        (_response(text='{"answer":"yes","count":NaN}'), "response_json_invalid"),
        (_response(text='{"answer":"yes","count":1e9999}'), "response_json_invalid"),
        (_response(text='{"answer":"\\ud800","count":1}'), "response_json_invalid"),
        (_response(input_tokens=10**9), "response_usage_invalid"),
        (_response(cache_creation_input_tokens=1), "response_usage_invalid"),
    ],
)
def test_representative_response_failures_make_exactly_one_call(
    response: object, outcome: str
) -> None:
    _, _, request, messages, client = _frozen_inputs(response=response)

    result = client.generate(request)

    assert result.outcome == outcome
    assert result.failure is not None and result.failure.code == outcome
    assert result.transport_attempt_count == 1
    assert result.sdk_retry_count == 0
    assert len(messages.calls) == 1


@pytest.mark.parametrize(
    "text",
    [
        '{"answer":"' + chr(0xD800) + '","count":1}',
        '{"answer":"yes","count":' + "[" * 100 + "1" + "]" * 100 + "}",
    ],
)
def test_noncanonical_or_deep_json_is_terminal_without_rearchiving_text(
    text: str,
) -> None:
    _, _, request, messages, client = _frozen_inputs(response=_response(text=text))

    result = client.generate(request)

    assert result.outcome == "response_json_invalid"
    assert result.text is None
    assert result.parsed_json is None
    assert len(messages.calls) == 1


def test_transport_failure_is_generic_secret_free_and_not_retried() -> None:
    error = FakeTransportError(
        "api_key=sk-ant-do-not-archive Authorization: Bearer also-secret"
    )
    _, _, request, messages, client = _frozen_inputs(error=error)

    result = client.generate(request)

    serialized = result.model_dump_json()
    assert result.outcome == "transport_failed"
    assert result.failure is not None
    assert result.failure.exception_type == "FakeTransportError"
    assert result.failure.http_status == 503
    assert result.failure.provider_request_id == "req_safe-123"
    assert result.cost.basis == "unknown_request_ceiling"
    assert result.cost.estimated_cost_usd is None
    assert result.cost.charged_cost_upper_bound_usd is None
    assert "sk-ant-do-not-archive" not in serialized
    assert "also-secret" not in serialized
    assert len(messages.calls) == 1


def test_cost_ceiling_counts_every_utf8_byte_schema_framing_and_output() -> None:
    config = AnthropicBoundedConfigV1(timeout_seconds=120)
    wire_schema = {"type": "object", "description": "μ", "additionalProperties": False}
    system = "systém"
    prompt = "paper → JSON"
    ceiling = compute_anthropic_request_cost_ceiling(
        config=config,
        system=system,
        prompt=prompt,
        wire_schema=wire_schema,
        max_output_tokens=777,
    )

    expected_input = (
        len(system.encode("utf-8"))
        + len(prompt.encode("utf-8"))
        + len(canonical_json_bytes(wire_schema))
        + 1024
    )
    expected_cost = (
        Decimal(expected_input) * Decimal(2) + Decimal(777) * Decimal(10)
    ) / Decimal(1_000_000)
    assert ceiling.conservative_input_token_ceiling == expected_input
    assert ceiling.request_cost_ceiling_usd == expected_cost
    assert (
        ceiling.token_bound_method
        == "one_token_per_model_facing_utf8_byte_plus_fixed_framing"
    )


def test_request_preserves_exact_prompt_and_system_bytes_including_whitespace() -> None:
    config = AnthropicBoundedConfigV1(timeout_seconds=120)
    compiled = compile_anthropic_bounded_schema(
        original_schema=_SIMPLE_SCHEMA,
        full_acceptance_schema_sha256=_FULL_HASH,
    )
    system = " system with intentional boundary spaces \n"
    prompt = "line one\nline two\n"
    request = freeze_anthropic_bounded_request(
        operation="exact-text",
        request_key="one",
        prompt=prompt,
        system=system,
        max_output_tokens=100,
        compiled_schema=compiled,
        config=config,
        schema_kind="inventory",
        effect_kind=None,
    )

    assert request.system == system
    assert request.prompt == prompt
    assert request.cost_ceiling.base_system_utf8_bytes == len(system.encode("utf-8"))
    assert request.cost_ceiling.base_prompt_utf8_bytes == len(prompt.encode("utf-8"))


def test_coherently_rehashed_cost_or_prompt_tampering_is_rejected_pre_call() -> None:
    _, _, request, messages, client = _frozen_inputs()
    payload = request.model_dump(mode="json")
    payload["prompt"] = "x" * 5000
    payload["request_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "request_sha256"}
    )

    with pytest.raises(PydanticValidationError, match="model_surface_binding_mismatch"):
        client.generate(type(request).model_validate(payload))
    assert messages.calls == []


def test_transport_environment_redirects_are_rejected_without_constructing_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, request, _, _ = _frozen_inputs()
    client = AnthropicBoundedClient(AnthropicBoundedConfigV1(timeout_seconds=120))
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://redirect.invalid")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "anthropic-version:evil")

    with patch("anthropic.Anthropic") as constructor:
        result = client.generate(request)

    assert result.outcome == "transport_failed"
    assert constructor.call_count == 0


def test_missing_usage_and_adversarial_error_metadata_are_terminal_and_sanitized() -> None:
    response = _response()
    del response.usage
    _, _, request, messages, client = _frozen_inputs(response=response)
    result = client.generate(request)
    assert result.outcome == "response_usage_invalid"
    assert len(messages.calls) == 1

    error = FakeTransportError("secret")
    error.request_id = "req_sk-ant-super-secret"
    _, _, request, _, client = _frozen_inputs(error=error)
    result = client.generate(request)
    assert result.failure is not None
    assert result.failure.provider_request_id is None


def test_provider_controlled_getter_failures_are_terminal_results() -> None:
    class RaisingMetadata:
        def __init__(self) -> None:
            self.model = ANTHROPIC_MODEL
            self.stop_reason = "end_turn"
            self.content = [
                SimpleNamespace(type="text", text='{"answer":"yes","count":1}')
            ]
            self.usage = SimpleNamespace(input_tokens=100, output_tokens=20)

        @property
        def id(self) -> str:
            raise RuntimeError("provider metadata unavailable")

    _, _, request, messages, client = _frozen_inputs(response=RaisingMetadata())
    result = client.generate(request)
    assert result.outcome == "response_identity_invalid"
    assert len(messages.calls) == 1

    class RaisingContent:
        id = "msg_test"
        model = ANTHROPIC_MODEL
        stop_reason = "end_turn"
        usage = SimpleNamespace(input_tokens=100, output_tokens=20)

        @property
        def content(self) -> list[object]:
            raise RuntimeError("provider content unavailable")

    _, _, request, messages, client = _frozen_inputs(response=RaisingContent())
    result = client.generate(request)
    assert result.outcome == "response_content_invalid"
    assert len(messages.calls) == 1

    class RaisingErrorMetadata(Exception):
        @property
        def status_code(self) -> int:
            raise RuntimeError("bad status getter")

        @property
        def request_id(self) -> str:
            raise RuntimeError("bad request-id getter")

    _, _, request, messages, client = _frozen_inputs(
        error=RaisingErrorMetadata("transport failed")
    )
    result = client.generate(request)
    assert result.outcome == "transport_failed"
    assert result.failure is not None
    assert result.failure.http_status is None
    assert result.failure.provider_request_id is None
    assert len(messages.calls) == 1

    error = FakeTransportError("secret")
    error.status_code = 999
    error.request_id = "sk-ant-super-secret"
    _, _, request, messages, client = _frozen_inputs(error=error)
    result = client.generate(request)
    assert result.outcome == "transport_failed"
    assert result.failure is not None
    assert result.failure.http_status is None
    assert result.failure.provider_request_id is None
    assert "sk-ant" not in result.model_dump_json().casefold()
    assert len(messages.calls) == 1


def test_models_requests_and_results_are_credential_free_and_hash_bound() -> None:
    config, compiled, request, _, client = _frozen_inputs()
    identity = client.identity
    result = client.generate(request)

    assert config.config_sha256 == (
        "6814b7cee4b43fbd1c7d9717c2243907e48f2631a23b0ae57de82eb0cea7c5e5"
    )
    assert identity.identity_sha256 == (
        "fab8f686ea0f80039b636781957a597660cde7bc96a451159950db91d5ba5063"
    )
    assert config.pricing_table_sha256 == ANTHROPIC_PRICING_TABLE_SHA256
    assert ANTHROPIC_PRICING_TABLE_SHA256 == (
        "af9d91d0ff6293255f941211d090a389cf7226238c2cfb5eba943405e4a7e2e6"
    )
    assert compiled.original_schema_sha256 == hash_canonical(_SIMPLE_SCHEMA)
    assert request.request_sha256 == hash_canonical(
        request.model_dump(mode="json", exclude={"request_sha256"})
    )
    assert result.result_sha256 == hash_canonical(
        result.model_dump(mode="json", exclude={"result_sha256"})
    )

    serialized = "\n".join(
        model.model_dump_json()
        for model in (config, identity, compiled, request, result)
    ).casefold()
    assert "api_key" not in serialized
    assert "authorization" not in serialized
    assert "bearer " not in serialized
    assert "sk-ant-" not in serialized
    assert "credential_defined_not_archived" in serialized
    for model in (config, identity, compiled, request, result):
        assert "api_key" not in json.dumps(model.model_json_schema()).casefold()


def test_nested_schema_mutation_is_rejected_before_any_call() -> None:
    _, _, request, messages, client = _frozen_inputs()
    request.compiled_schema.wire_schema["unexpected"] = True

    with pytest.raises(
        AnthropicBoundedGenerationError, match="request_revalidation_failed"
    ):
        client.generate(request)

    assert messages.calls == []
