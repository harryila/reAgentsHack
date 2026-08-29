"""Strict one-attempt Anthropic boundary for bounded native generation.

This module is intentionally independent of the frozen local-Ollama runtime.  It
compiles a provider-facing schema without changing the local acceptance schema,
freezes every request input before transport, applies a conservative pre-request
cost ceiling, and makes exactly one Anthropic SDK call.  It never reads or stores a
credential and it contains no retry loop.

The transformed wire schema is a generation aid only.  Successful responses are
validated against the original provider schema here; the caller must still validate
the raw JSON against the separately hash-bound full acceptance schema.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal
from typing import Annotated, Any, Literal

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from pydantic import ConfigDict, Field, StrictInt, field_validator, model_validator

from literature_multiverse.lineage import canonical_json_bytes, hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel

ANTHROPIC_BOUNDED_CONTRACT_VERSION = "anthropic-bounded-generation-v1"
ANTHROPIC_SCHEMA_COMPILER_VERSION = "anthropic-literal-type-compiler-v7"
ANTHROPIC_SDK_VERSION = "0.120.2"
ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_MODEL = "claude-sonnet-5"
ANTHROPIC_API_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_INPUT_RATE_USD_PER_MTOK = Decimal("2")
ANTHROPIC_OUTPUT_RATE_USD_PER_MTOK = Decimal("10")
ANTHROPIC_FIXED_FRAMING_TOKENS = 1024
ANTHROPIC_PRICING_SOURCE_URL = (
    "https://platform.claude.com/docs/en/about-claude/pricing"
)
ANTHROPIC_MODEL_SOURCE_URL = (
    "https://platform.claude.com/docs/en/models/sonnet-5/whats-new-sonnet-5"
)
ANTHROPIC_SOURCE_VERIFIED_DATE = "2026-08-28"
ANTHROPIC_PINNED_PRICING_TABLE_V1: dict[str, Any] = {
    "currency": "USD",
    "model": ANTHROPIC_MODEL,
    "api_base_url": ANTHROPIC_API_BASE_URL,
    "service_tier": "standard_only",
    "unit": "one_million_tokens",
    "input_rate": "2",
    "output_rate": "10",
    "pricing_source_url": ANTHROPIC_PRICING_SOURCE_URL,
    "model_source_url": ANTHROPIC_MODEL_SOURCE_URL,
    "source_verified_date": ANTHROPIC_SOURCE_VERIFIED_DATE,
}
ANTHROPIC_PRICING_TABLE_SHA256 = hash_canonical(ANTHROPIC_PINNED_PRICING_TABLE_V1)

_MAX_OUTPUT_TOKENS = 128_000
_MAX_TEXT_CHARACTERS = 20_000_000
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 1_000_000
_MAX_FAILURE_TYPE_CHARACTERS = 160
_SAFE_EXCEPTION_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,159}$")

# These limits are part of the v4 compiler algorithm.  Reference expansion can
# otherwise turn a small recursive or highly shared schema into an unbounded wire
# payload before the provider sees it.  They are deliberately much larger than
# every schema in the frozen MetaSyn roster while still failing closed on an
# accidental expansion bomb.
ANTHROPIC_SCHEMA_MAX_REFERENCE_EXPANSIONS = 10_000
ANTHROPIC_SCHEMA_MAX_INLINED_NODES = 100_000
ANTHROPIC_SCHEMA_MAX_INLINED_DEPTH = 128
ANTHROPIC_SCHEMA_MAX_INLINED_UTF8_BYTES = 5_000_000
ANTHROPIC_SCHEMA_MAX_NULLABLE_OPTIONAL_PROMOTIONS = 10_000
ANTHROPIC_SCHEMA_NULLABLE_OPTIONAL_REQUIRED_TARGET = 10
ANTHROPIC_SCHEMA_MAX_OPTIONAL_PARAMETERS = 24
ANTHROPIC_SCHEMA_MAX_UNION_PARAMETERS = 16
ANTHROPIC_NULLABLE_RETAIN_PRIORITY_V1 = (
    "/properties/contrast/properties/estimand",
    "/properties/effect/properties/equivalence_margin",
    "/properties/effect/properties/reported_p_value",
    "/properties/finding/properties/analysis_population",
    "/properties/finding/properties/timepoint/properties/value",
    "/properties/finding/properties/timepoint/properties/lower",
    "/properties/finding/properties/timepoint/properties/upper",
    "/properties/finding/properties/timepoint/properties/unit",
    "/properties/finding/properties/timepoint/properties/raw_label",
    "/properties/effect/properties/unit",
    "/properties/finding/properties/timepoint/properties/anchor",
)
ANTHROPIC_NULLABLE_RETAIN_PRIORITY_SHA256 = hash_canonical(
    ANTHROPIC_NULLABLE_RETAIN_PRIORITY_V1
)
ANTHROPIC_PROMPT_JSON_SYSTEM_ENVELOPE_VERSION = (
    "anthropic-prompt-json-system-envelope-v1"
)
ANTHROPIC_PROMPT_JSON_SYSTEM_PREAMBLE = (
    "Return exactly one JSON object and no other text. Never use Markdown fences. "
    "The object must validate against the canonical JSON Schema below."
)
ANTHROPIC_PROMPT_JSON_SYSTEM_ENVELOPE_SHA256 = hash_canonical(
    {
        "version": ANTHROPIC_PROMPT_JSON_SYSTEM_ENVELOPE_VERSION,
        "preamble": ANTHROPIC_PROMPT_JSON_SYSTEM_PREAMBLE,
        "schema_hash_label": "WIRE_SCHEMA_SHA256",
        "schema_bytes_label": "WIRE_SCHEMA_UTF8_BYTES",
        "terminal_label": "OUTPUT_SCHEMA_JSON_FOLLOWS_TO_END_OF_SYSTEM",
        "schema_is_terminal_bytes": True,
    }
)

_SCHEMA_MAPPING_KEYWORDS = (
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
)
_SCHEMA_SINGLE_KEYWORDS = (
    "items",
    "additionalItems",
    "additionalProperties",
    "unevaluatedItems",
    "unevaluatedProperties",
    "contains",
    "propertyNames",
    "contentSchema",
    "if",
    "then",
    "else",
    "not",
)
_SCHEMA_SEQUENCE_KEYWORDS = ("allOf", "anyOf", "oneOf", "prefixItems")
_SCHEMA_MIXED_MAPPING_KEYWORDS = ("dependencies",)

AnthropicOutcome = Literal[
    "completed",
    "transport_failed",
    "response_identity_invalid",
    "response_model_mismatch",
    "response_stop_reason_invalid",
    "response_content_invalid",
    "response_usage_invalid",
    "response_json_invalid",
    "response_wire_schema_invalid",
    "response_schema_invalid",
]
AnthropicFailureCode = Literal[
    "transport_failed",
    "response_identity_invalid",
    "response_model_mismatch",
    "response_stop_reason_invalid",
    "response_content_invalid",
    "response_usage_invalid",
    "response_json_invalid",
    "response_wire_schema_invalid",
    "response_schema_invalid",
]
AnthropicTransportMode = Literal["structured_json_schema", "prompt_json_schema"]
AnthropicSchemaKind = Literal["inventory", "packet"]
AnthropicEffectKind = Literal[
    "binary_group_statistics",
    "continuous_group_statistics",
    "direct_confidence_interval",
    "direct_standard_error",
    "direct_variance",
]


class AnthropicBoundedGenerationError(ValueError):
    """A schema, request, or pre-transport provider binding is unsafe."""


def _sha256_utf8(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _FrozenContract(ContractModel):
    """Closed, assignment-frozen provider contract."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        # Prompts and response text are byte-bound scientific inputs/outputs.
        str_strip_whitespace=False,
        use_enum_values=False,
        frozen=True,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]
PositiveRate = Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
TokenCount = Annotated[StrictInt, Field(ge=0)]


def _require_exact_sdk() -> Any:
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - required project dependency
        raise AnthropicBoundedGenerationError("anthropic_sdk_missing") from exc
    observed = str(getattr(anthropic, "__version__", "unknown"))
    if observed != ANTHROPIC_SDK_VERSION:
        raise AnthropicBoundedGenerationError(
            f"anthropic_sdk_version_mismatch:{observed}"
        )
    return anthropic


def _validate_json_schema(schema: Mapping[str, Any], *, code: str) -> None:
    try:
        schema_class = validator_for(schema)
        schema_class.check_schema(schema)
    except (SchemaError, TypeError, ValueError) as exc:
        raise AnthropicBoundedGenerationError(code) from exc
    _validate_local_schema_references(schema, code=code)


def _validate_local_schema_references(
    schema: Mapping[str, Any], *, code: str
) -> None:
    """Validate references at schema positions, never inside instance literals."""

    try:
        _audit_anthropic_reference_graph(schema)
    except AnthropicBoundedGenerationError as exc:
        raise AnthropicBoundedGenerationError(f"{code}_{exc}") from exc


def _decode_local_json_pointer(reference: str) -> tuple[str, ...]:
    """Decode the supported local RFC 6901 subset without URI ambiguity.

    URI percent-encoding is intentionally outside compiler v4.  Rejecting it is
    safer and deterministic: two textual references can never alias only after a
    hidden URI-decoding step.
    """

    if reference == "#":
        return ()
    if not reference.startswith("#/"):
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_reference_nonlocal_forbidden"
        )
    if "%" in reference:
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_reference_percent_encoding_forbidden"
        )

    decoded: list[str] = []
    for raw_token in reference[2:].split("/"):
        token: list[str] = []
        index = 0
        while index < len(raw_token):
            character = raw_token[index]
            if character != "~":
                token.append(character)
                index += 1
                continue
            if index + 1 >= len(raw_token) or raw_token[index + 1] not in {"0", "1"}:
                raise AnthropicBoundedGenerationError(
                    "anthropic_schema_reference_pointer_escape_invalid"
                )
            token.append("~" if raw_token[index + 1] == "0" else "/")
            index += 2
        decoded.append("".join(token))
    return tuple(decoded)


def _iter_structural_schema_children(
    schema: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Return only child schema objects, excluding literal instance data."""

    children_out: list[Mapping[str, Any]] = []
    for keyword in _SCHEMA_MAPPING_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, Mapping):
            children_out.extend(
                child for child in children.values() if isinstance(child, Mapping)
            )
    for keyword in _SCHEMA_MIXED_MAPPING_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, Mapping):
            # Draft-07 property dependencies are string arrays; only mapping
            # values are schema dependencies.
            children_out.extend(
                child for child in children.values() if isinstance(child, Mapping)
            )
    for keyword in _SCHEMA_SINGLE_KEYWORDS:
        child = schema.get(keyword)
        if isinstance(child, Mapping):
            children_out.append(child)
    for keyword in _SCHEMA_SEQUENCE_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, list):
            children_out.extend(child for child in children if isinstance(child, Mapping))
    return children_out


def _resolve_local_json_pointer(
    root: Mapping[str, Any], pointer: tuple[str, ...]
) -> Mapping[str, Any]:
    current: Any = root
    for token in pointer:
        if isinstance(current, Mapping):
            if token not in current:
                raise AnthropicBoundedGenerationError(
                    "anthropic_schema_reference_unresolved"
                )
            current = current[token]
            continue
        if isinstance(current, list):
            if (
                not token.isascii()
                or not token.isdigit()
                or (len(token) > 1 and token.startswith("0"))
            ):
                raise AnthropicBoundedGenerationError(
                    "anthropic_schema_reference_unresolved"
                )
            item_index = int(token)
            if item_index >= len(current):
                raise AnthropicBoundedGenerationError(
                    "anthropic_schema_reference_unresolved"
                )
            current = current[item_index]
            continue
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_reference_unresolved"
        )
    if not isinstance(current, Mapping):
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_reference_target_not_object"
        )
    return current


def _structural_references_within(
    schema: Mapping[str, Any],
) -> set[tuple[str, ...]]:
    references: set[tuple[str, ...]] = set()
    pending: list[Mapping[str, Any]] = [schema]
    while pending:
        node = pending.pop()
        reference = node.get("$ref")
        if isinstance(reference, str):
            references.add(_decode_local_json_pointer(reference))
        pending.extend(_iter_structural_schema_children(node))
    return references


def _audit_anthropic_reference_graph(schema: Mapping[str, Any]) -> None:
    """Globally audit used and unused schema positions before any expansion."""

    root = schema
    referenced_pointers: set[tuple[str, ...]] = set()
    pending: list[tuple[Mapping[str, Any], int]] = [(schema, 0)]
    audited_nodes = 0
    while pending:
        node, depth = pending.pop()
        audited_nodes += 1
        if audited_nodes > ANTHROPIC_SCHEMA_MAX_INLINED_NODES:
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_reference_node_limit_exceeded"
            )
        if depth > ANTHROPIC_SCHEMA_MAX_INLINED_DEPTH:
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_reference_depth_limit_exceeded"
            )
        if "$dynamicRef" in node or "$recursiveRef" in node:
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_dynamic_or_recursive_reference_forbidden"
            )
        if depth > 0 and any(
            keyword in node
            for keyword in ("$id", "$anchor", "$dynamicAnchor", "$recursiveAnchor")
        ):
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_nested_reference_scope_forbidden"
            )
        if "$ref" in node:
            if set(node) != {"$ref"}:
                raise AnthropicBoundedGenerationError(
                    "anthropic_schema_reference_siblings_forbidden"
                )
            reference = node["$ref"]
            if not isinstance(reference, str):
                raise AnthropicBoundedGenerationError(
                    "anthropic_schema_reference_not_string"
                )
            pointer = _decode_local_json_pointer(reference)
            _resolve_local_json_pointer(root, pointer)
            referenced_pointers.add(pointer)
        pending.extend(
            (child, depth + 1) for child in _iter_structural_schema_children(node)
        )

    # Build a graph between referenced targets and reject every cycle, including
    # one reachable only through an unused definition container.
    adjacency: dict[tuple[str, ...], set[tuple[str, ...]]] = {}
    pointers_to_audit = list(referenced_pointers)
    while pointers_to_audit:
        pointer = pointers_to_audit.pop()
        if pointer in adjacency:
            continue
        target = _resolve_local_json_pointer(root, pointer)
        outgoing = _structural_references_within(target)
        for destination in outgoing:
            _resolve_local_json_pointer(root, destination)
        adjacency[pointer] = outgoing
        pointers_to_audit.extend(outgoing - adjacency.keys())

    state: dict[tuple[str, ...], Literal["visiting", "visited"]] = {}

    def visit_pointer(pointer: tuple[str, ...], *, depth: int) -> None:
        if depth > ANTHROPIC_SCHEMA_MAX_INLINED_DEPTH:
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_reference_depth_limit_exceeded"
            )
        observed = state.get(pointer)
        if observed == "visiting":
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_reference_cycle"
            )
        if observed == "visited":
            return
        state[pointer] = "visiting"
        for destination in sorted(adjacency.get(pointer, set())):
            visit_pointer(destination, depth=depth + 1)
        state[pointer] = "visited"

    for pointer in sorted(adjacency):
        visit_pointer(pointer, depth=0)


def _audit_no_stale_reference_metadata(schema: Mapping[str, Any]) -> None:
    """Forbid provider annotations that still point at removed definitions."""

    pending: list[Mapping[str, Any]] = [schema]
    while pending:
        node = pending.pop()
        if "discriminator" in node:
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_discriminator_not_removed"
            )
        for keyword in ("description", "title", "$comment"):
            value = node.get(keyword)
            if isinstance(value, str) and any(
                marker in value for marker in ("#/$defs/", "#/definitions/")
            ):
                raise AnthropicBoundedGenerationError(
                    "anthropic_schema_stale_reference_metadata"
                )
        pending.extend(_iter_structural_schema_children(node))


def inline_anthropic_local_references(
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministically inline local ``$ref`` nodes for Anthropic's grammar.

    The input remains the authoritative JSON Schema.  This provider-only stage
    expands ref-only local JSON-pointer nodes, removes definition containers, and
    rejects every case where expansion could alter or ambiguously approximate the
    original semantics.  In particular, cycles, unresolved pointers, ref siblings,
    dynamic/recursive references, nested identifier scopes, and expansion bombs
    fail before the SDK or network boundary.
    """

    try:
        root = json.loads(canonical_json_bytes(dict(schema)))
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_reference_input_not_canonical_json"
        ) from exc
    if not isinstance(root, dict):  # pragma: no cover - Mapping contract guard
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_reference_input_not_object"
        )
    if len(canonical_json_bytes(root)) > ANTHROPIC_SCHEMA_MAX_INLINED_UTF8_BYTES:
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_reference_input_byte_limit_exceeded"
        )
    _audit_anthropic_reference_graph(root)

    expansion_count = 0
    schema_node_count = 0

    def visit(
        raw: Mapping[str, Any],
        *,
        reference_stack: tuple[tuple[str, ...], ...],
        depth: int,
    ) -> dict[str, Any]:
        nonlocal expansion_count, schema_node_count
        if depth > ANTHROPIC_SCHEMA_MAX_INLINED_DEPTH:
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_reference_depth_limit_exceeded"
            )
        schema_node_count += 1
        if schema_node_count > ANTHROPIC_SCHEMA_MAX_INLINED_NODES:
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_reference_node_limit_exceeded"
            )

        if "$dynamicRef" in raw or "$recursiveRef" in raw:
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_dynamic_or_recursive_reference_forbidden"
            )
        if depth > 0 and any(
            keyword in raw
            for keyword in ("$id", "$anchor", "$dynamicAnchor", "$recursiveAnchor")
        ):
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_nested_reference_scope_forbidden"
            )
        if "$ref" in raw:
            if set(raw) != {"$ref"}:
                raise AnthropicBoundedGenerationError(
                    "anthropic_schema_reference_siblings_forbidden"
                )
            reference = raw["$ref"]
            if not isinstance(reference, str):
                raise AnthropicBoundedGenerationError(
                    "anthropic_schema_reference_not_string"
                )
            pointer = _decode_local_json_pointer(reference)
            if pointer in reference_stack:
                raise AnthropicBoundedGenerationError(
                    "anthropic_schema_reference_cycle"
                )
            expansion_count += 1
            if expansion_count > ANTHROPIC_SCHEMA_MAX_REFERENCE_EXPANSIONS:
                raise AnthropicBoundedGenerationError(
                    "anthropic_schema_reference_expansion_limit_exceeded"
                )
            return visit(
                _resolve_local_json_pointer(root, pointer),
                reference_stack=(*reference_stack, pointer),
                depth=depth + 1,
            )

        # Definition containers are resolution-only inputs.  Keeping them on an
        # ``anyOf`` schema is the exact provider rejection fixed by compiler v4.
        node = {
            str(key): deepcopy(value)
            for key, value in raw.items()
            if key not in {"$defs", "definitions", "discriminator"}
        }
        for keyword in _SCHEMA_MAPPING_KEYWORDS:
            if keyword in {"$defs", "definitions"}:
                continue
            children = raw.get(keyword)
            if isinstance(children, Mapping):
                node[keyword] = {
                    str(key): visit(
                        child,
                        reference_stack=reference_stack,
                        depth=depth + 1,
                    )
                    if isinstance(child, Mapping)
                    else deepcopy(child)
                    for key, child in children.items()
                }
        for keyword in _SCHEMA_MIXED_MAPPING_KEYWORDS:
            children = raw.get(keyword)
            if isinstance(children, Mapping):
                node[keyword] = {
                    str(key): visit(
                        child,
                        reference_stack=reference_stack,
                        depth=depth + 1,
                    )
                    if isinstance(child, Mapping)
                    else deepcopy(child)
                    for key, child in children.items()
                }
        for keyword in _SCHEMA_SINGLE_KEYWORDS:
            child = raw.get(keyword)
            if isinstance(child, Mapping):
                node[keyword] = visit(
                    child,
                    reference_stack=reference_stack,
                    depth=depth + 1,
                )
        for keyword in _SCHEMA_SEQUENCE_KEYWORDS:
            children = raw.get(keyword)
            if isinstance(children, list):
                node[keyword] = [
                    visit(
                        child,
                        reference_stack=reference_stack,
                        depth=depth + 1,
                    )
                    if isinstance(child, Mapping)
                    else deepcopy(child)
                    for child in children
                ]
        return node

    inlined = visit(root, reference_stack=(), depth=0)

    # Audit schema positions rather than literal values under enum/const/default.
    pending: list[Mapping[str, Any]] = [inlined]
    while pending:
        node = pending.pop()
        if any(
            keyword in node
            for keyword in ("$ref", "$dynamicRef", "$recursiveRef", "$defs", "definitions")
        ):
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_reference_postcondition_failed"
            )
        for keyword in _SCHEMA_MAPPING_KEYWORDS:
            children = node.get(keyword)
            if isinstance(children, Mapping):
                pending.extend(
                    child for child in children.values() if isinstance(child, Mapping)
                )
        for keyword in _SCHEMA_MIXED_MAPPING_KEYWORDS:
            children = node.get(keyword)
            if isinstance(children, Mapping):
                pending.extend(
                    child for child in children.values() if isinstance(child, Mapping)
                )
        for keyword in _SCHEMA_SINGLE_KEYWORDS:
            child = node.get(keyword)
            if isinstance(child, Mapping):
                pending.append(child)
        for keyword in _SCHEMA_SEQUENCE_KEYWORDS:
            children = node.get(keyword)
            if isinstance(children, list):
                pending.extend(child for child in children if isinstance(child, Mapping))

    _audit_no_stale_reference_metadata(inlined)

    try:
        encoded = canonical_json_bytes(inlined)
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:  # pragma: no cover
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_reference_output_not_canonical_json"
        ) from exc
    if len(encoded) > ANTHROPIC_SCHEMA_MAX_INLINED_UTF8_BYTES:
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_reference_output_byte_limit_exceeded"
        )
    return json.loads(encoded)


def _literal_json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_literal_nonfinite"
            )
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    raise AnthropicBoundedGenerationError("anthropic_schema_literal_type_unsupported")


def annotate_anthropic_literal_types(
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Add only a homogeneous JSON type already implied by ``enum``/``const``.

    Anthropic SDK 0.120.2 requires a type (or a composition keyword) on each
    transformable node.  JSON Schema literal-only branches need no explicit type,
    so this compiler adds the redundant annotation.  A heterogeneous untyped enum,
    or conflicting untyped enum and const, is rejected instead of guessed.
    """

    def visit(raw: Mapping[str, Any]) -> dict[str, Any]:
        node = deepcopy(dict(raw))
        if "type" not in node:
            inferred: set[str] = set()
            if "const" in node:
                inferred.add(_literal_json_type(node["const"]))
            if "enum" in node:
                raw_enum = node["enum"]
                if not isinstance(raw_enum, list) or not raw_enum:
                    raise AnthropicBoundedGenerationError(
                        "anthropic_schema_untyped_enum_empty_or_invalid"
                    )
                inferred.update(_literal_json_type(value) for value in raw_enum)
            if len(inferred) > 1:
                raise AnthropicBoundedGenerationError(
                    "anthropic_schema_literal_types_heterogeneous"
                )
            if inferred:
                node["type"] = next(iter(inferred))

        for keyword in _SCHEMA_MAPPING_KEYWORDS:
            children = node.get(keyword)
            if isinstance(children, Mapping):
                node[keyword] = {
                    str(key): visit(child)
                    if isinstance(child, Mapping)
                    else deepcopy(child)
                    for key, child in children.items()
                }
        for keyword in _SCHEMA_MIXED_MAPPING_KEYWORDS:
            children = node.get(keyword)
            if isinstance(children, Mapping):
                node[keyword] = {
                    str(key): visit(child)
                    if isinstance(child, Mapping)
                    else deepcopy(child)
                    for key, child in children.items()
                }
        for keyword in _SCHEMA_SINGLE_KEYWORDS:
            child = node.get(keyword)
            if isinstance(child, Mapping):
                node[keyword] = visit(child)
        for keyword in _SCHEMA_SEQUENCE_KEYWORDS:
            children = node.get(keyword)
            if isinstance(children, list):
                node[keyword] = [
                    visit(child) if isinstance(child, Mapping) else deepcopy(child)
                    for child in children
                ]
        return node

    return visit(schema)


def _encode_json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _schema_accepts_null(schema: Any) -> bool:
    if type(schema) is bool:
        return schema
    if not isinstance(schema, Mapping):
        return False
    try:
        return validator_for(schema)(schema).is_valid(None)
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
            raise
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_nullable_optional_proof_failed"
        ) from exc


def _count_optional_parameters(schema: Mapping[str, Any]) -> int:
    count = 0
    pending: list[Mapping[str, Any]] = [schema]
    while pending:
        node = pending.pop()
        properties = node.get("properties")
        if isinstance(properties, Mapping):
            required = node.get("required", [])
            required_names = set(required) if isinstance(required, list) else set()
            count += len(set(properties) - required_names)
        pending.extend(_iter_structural_schema_children(node))
    return count


def count_anthropic_optional_parameters(schema: Mapping[str, Any]) -> int:
    """Count syntactically optional object properties as Anthropic does."""

    try:
        canonical = json.loads(canonical_json_bytes(dict(schema)))
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_optional_count_input_not_canonical_json"
        ) from exc
    _validate_json_schema(canonical, code="anthropic_schema_optional_count_invalid")
    return _count_optional_parameters(canonical)


def _count_union_parameters(schema: Mapping[str, Any]) -> int:
    """Count property schemas using a type array or ``anyOf``.

    This is the provider's documented structural unit for the union-parameter cap,
    not a semantic count of every schema node that happens to accept multiple types.
    """

    count = 0
    pending: list[Mapping[str, Any]] = [schema]
    while pending:
        node = pending.pop()
        properties = node.get("properties")
        if isinstance(properties, Mapping):
            for child in properties.values():
                if isinstance(child, Mapping) and (
                    isinstance(child.get("type"), list)
                    or isinstance(child.get("anyOf"), list)
                    or isinstance(child.get("oneOf"), list)
                ):
                    count += 1
        pending.extend(_iter_structural_schema_children(node))
    return count


def count_anthropic_union_parameters(schema: Mapping[str, Any]) -> int:
    """Count provider union-bearing property schemas after schema validation."""

    try:
        canonical = json.loads(canonical_json_bytes(dict(schema)))
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_union_count_input_not_canonical_json"
        ) from exc
    _validate_json_schema(canonical, code="anthropic_schema_union_count_invalid")
    return _count_union_parameters(canonical)


def _collect_nullable_optional_property_paths(
    schema: Mapping[str, Any],
) -> list[str]:
    paths: list[str] = []
    node_count = 0

    def visit(raw: Mapping[str, Any], *, path: str, depth: int) -> None:
        nonlocal node_count
        if depth > ANTHROPIC_SCHEMA_MAX_INLINED_DEPTH:
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_nullable_adaptation_depth_limit_exceeded"
            )
        node_count += 1
        if node_count > ANTHROPIC_SCHEMA_MAX_INLINED_NODES:
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_nullable_adaptation_node_limit_exceeded"
            )
        properties = raw.get("properties")
        if isinstance(properties, Mapping):
            required_raw = raw.get("required", [])
            required = set(required_raw) if isinstance(required_raw, list) else set()
            for name in sorted(properties):
                if name not in required and _schema_accepts_null(properties[name]):
                    paths.append(
                        f"{path}/properties/{_encode_json_pointer_token(str(name))}"
                    )
        for keyword in _SCHEMA_MAPPING_KEYWORDS:
            children = raw.get(keyword)
            if isinstance(children, Mapping):
                for name, child in children.items():
                    if isinstance(child, Mapping):
                        visit(
                            child,
                            path=(
                                f"{path}/{keyword}/"
                                f"{_encode_json_pointer_token(str(name))}"
                            ),
                            depth=depth + 1,
                        )
        for keyword in _SCHEMA_MIXED_MAPPING_KEYWORDS:
            children = raw.get(keyword)
            if isinstance(children, Mapping):
                for name, child in children.items():
                    if isinstance(child, Mapping):
                        visit(
                            child,
                            path=(
                                f"{path}/{keyword}/"
                                f"{_encode_json_pointer_token(str(name))}"
                            ),
                            depth=depth + 1,
                        )
        for keyword in _SCHEMA_SINGLE_KEYWORDS:
            child = raw.get(keyword)
            if isinstance(child, Mapping):
                visit(child, path=f"{path}/{keyword}", depth=depth + 1)
        for keyword in _SCHEMA_SEQUENCE_KEYWORDS:
            children = raw.get(keyword)
            if isinstance(children, list):
                for index, child in enumerate(children):
                    if isinstance(child, Mapping):
                        visit(
                            child,
                            path=f"{path}/{keyword}/{index}",
                            depth=depth + 1,
                        )

    visit(schema, path="#", depth=0)
    paths.sort()
    if len(paths) != len(set(paths)):
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_nullable_adaptation_duplicate_path"
        )
    if len(paths) > ANTHROPIC_SCHEMA_MAX_NULLABLE_OPTIONAL_PROMOTIONS:
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_nullable_adaptation_limit_exceeded"
        )
    return paths


_NULL_BRANCH_ANNOTATION_KEYS = frozenset(
    {
        "type",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "$comment",
    }
)
_NULLABLE_WRAPPER_ANNOTATION_KEYS = _NULL_BRANCH_ANNOTATION_KEYS.difference({"type"})


def _is_structurally_null_only_branch(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("type") == "null"
        and set(value).issubset(_NULL_BRANCH_ANNOTATION_KEYS)
    )


def _null_stripping_strategy(
    schema: Mapping[str, Any],
) -> Literal[
    "remove-null-from-single-nonnull-type-array",
    "collapse-exact-two-branch-anyof",
    "collapse-exact-two-branch-oneof",
]:
    raw_type = schema.get("type")
    if isinstance(raw_type, list) and "null" in raw_type:
        return "remove-null-from-single-nonnull-type-array"
    if isinstance(schema.get("anyOf"), list):
        return "collapse-exact-two-branch-anyof"
    if isinstance(schema.get("oneOf"), list):
        return "collapse-exact-two-branch-oneof"
    raise AnthropicBoundedGenerationError(
        "anthropic_schema_nullable_strip_shape_unsupported"
    )


def _strip_explicit_null_acceptance(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only a structurally isolated null alternative, or fail closed."""

    original = deepcopy(dict(schema))
    candidate: dict[str, Any] | None = None
    raw_type = original.get("type")
    if isinstance(raw_type, list) and "null" in raw_type:
        remaining_types = [value for value in raw_type if value != "null"]
        if len(remaining_types) != 1 or len(remaining_types) != len(
            set(remaining_types)
        ):
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_nullable_strip_type_array_unsafe"
            )
        candidate = deepcopy(original)
        candidate["type"] = remaining_types[0]
    else:
        for keyword in ("anyOf", "oneOf"):
            branches = original.get(keyword)
            if not isinstance(branches, list):
                continue
            if set(original).difference({keyword}).difference(
                _NULLABLE_WRAPPER_ANNOTATION_KEYS
            ):
                raise AnthropicBoundedGenerationError(
                    "anthropic_schema_nullable_strip_semantic_sibling_unsupported"
                )
            null_branches = [
                branch for branch in branches if _is_structurally_null_only_branch(branch)
            ]
            nonnull_branches = [
                branch for branch in branches if not _is_structurally_null_only_branch(branch)
            ]
            if len(null_branches) != 1 or len(nonnull_branches) != 1:
                raise AnthropicBoundedGenerationError(
                    "anthropic_schema_nullable_strip_branch_shape_unsupported"
                )
            nonnull = nonnull_branches[0]
            if not isinstance(nonnull, Mapping) or _schema_accepts_null(nonnull):
                raise AnthropicBoundedGenerationError(
                    "anthropic_schema_nullable_strip_nonnull_branch_invalid"
                )
            candidate = deepcopy(dict(nonnull))
            for name, value in original.items():
                if name == keyword:
                    continue
                if name in candidate and candidate[name] != value:
                    raise AnthropicBoundedGenerationError(
                        "anthropic_schema_nullable_strip_sibling_collision"
                    )
                candidate[name] = deepcopy(value)
            break
    if candidate is None:
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_nullable_strip_shape_unsupported"
        )
    _validate_json_schema(candidate, code="anthropic_schema_nullable_strip_output_invalid")
    if not _schema_accepts_null(original) or _schema_accepts_null(candidate):
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_nullable_strip_postcondition_failed"
        )
    return json.loads(canonical_json_bytes(candidate))


def _adapt_anthropic_nullable_optional_properties(
    schema: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    list[str],
    list[str],
    list[AnthropicNullableAdaptationProofV1],
]:
    """Apply the provider-only v6 hybrid nullable/optional grammar subset."""

    try:
        canonical = json.loads(canonical_json_bytes(dict(schema)))
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_nullable_adaptation_input_not_canonical_json"
        ) from exc
    _validate_json_schema(
        canonical, code="anthropic_schema_nullable_adaptation_input_invalid"
    )
    nullable_paths = _collect_nullable_optional_property_paths(canonical)

    def retain_rank(path: str) -> tuple[int, int, str]:
        for index, suffix in enumerate(ANTHROPIC_NULLABLE_RETAIN_PRIORITY_V1):
            if path.endswith(suffix):
                return (0, index, path)
        return (1, 0, path)

    retain_order = sorted(nullable_paths, key=retain_rank)
    promotion_paths = retain_order[
        :ANTHROPIC_SCHEMA_NULLABLE_OPTIONAL_REQUIRED_TARGET
    ]
    stripped_paths = sorted(
        set(nullable_paths).difference(promotion_paths)
    )
    promotion_set = set(promotion_paths)
    stripped_set = set(stripped_paths)
    proof_by_path: dict[str, AnthropicNullableAdaptationProofV1] = {}
    node_count = 0

    def visit(raw: Mapping[str, Any], *, path: str, depth: int) -> dict[str, Any]:
        nonlocal node_count
        if depth > ANTHROPIC_SCHEMA_MAX_INLINED_DEPTH:
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_nullable_adaptation_depth_limit_exceeded"
            )
        node_count += 1
        if node_count > ANTHROPIC_SCHEMA_MAX_INLINED_NODES:
            raise AnthropicBoundedGenerationError(
                "anthropic_schema_nullable_adaptation_node_limit_exceeded"
            )
        node = deepcopy(dict(raw))
        properties = raw.get("properties")
        if isinstance(properties, Mapping):
            required_raw = raw.get("required", [])
            required = set(required_raw) if isinstance(required_raw, list) else set()
            required_before = set(required)
            adapted_properties: dict[str, Any] = {}
            for name, child in properties.items():
                child_path = (
                    f"{path}/properties/{_encode_json_pointer_token(str(name))}"
                )
                if child_path in promotion_set:
                    required.add(str(name))
                adapted_properties[str(name)] = (
                    visit(child, path=child_path, depth=depth + 1)
                    if isinstance(child, Mapping)
                    else deepcopy(child)
                )
                if child_path in stripped_set:
                    adapted_child = adapted_properties[str(name)]
                    if not isinstance(adapted_child, Mapping):  # pragma: no cover
                        raise AnthropicBoundedGenerationError(
                            "anthropic_schema_nullable_strip_child_not_object"
                        )
                    adapted_properties[str(name)] = (
                        _strip_explicit_null_acceptance(adapted_child)
                    )
                if child_path in promotion_set or child_path in stripped_set:
                    original_child = properties[name]
                    provider_child = adapted_properties[str(name)]
                    if not isinstance(original_child, Mapping) or not isinstance(
                        provider_child, Mapping
                    ):
                        raise AnthropicBoundedGenerationError(
                            "anthropic_schema_nullable_adaptation_child_not_object"
                        )
                    action: Literal[
                        "require_nullable", "keep_optional_strip_null"
                    ] = (
                        "require_nullable"
                        if child_path in promotion_set
                        else "keep_optional_strip_null"
                    )
                    proof_by_path[child_path] = AnthropicNullableAdaptationProofV1(
                        path=child_path,
                        action=action,
                        strategy=(
                            "required-membership-only"
                            if action == "require_nullable"
                            else _null_stripping_strategy(original_child)
                        ),
                        original_property_schema_sha256=hash_canonical(original_child),
                        provider_property_schema_sha256=hash_canonical(provider_child),
                        required_before=str(name) in required_before,
                        required_after=str(name) in required,
                        accepts_null_before=_schema_accepts_null(original_child),
                        accepts_null_after=_schema_accepts_null(provider_child),
                    )
            node["properties"] = adapted_properties
            if promotion_set.intersection(
                f"{path}/properties/{_encode_json_pointer_token(str(name))}"
                for name in properties
            ):
                node["required"] = sorted(required)

        for keyword in _SCHEMA_MAPPING_KEYWORDS:
            if keyword == "properties":
                continue
            children = raw.get(keyword)
            if isinstance(children, Mapping):
                node[keyword] = {
                    str(name): visit(
                        child,
                        path=(
                            f"{path}/{keyword}/"
                            f"{_encode_json_pointer_token(str(name))}"
                        ),
                        depth=depth + 1,
                    )
                    if isinstance(child, Mapping)
                    else deepcopy(child)
                    for name, child in children.items()
                }
        for keyword in _SCHEMA_MIXED_MAPPING_KEYWORDS:
            children = raw.get(keyword)
            if isinstance(children, Mapping):
                node[keyword] = {
                    str(name): visit(
                        child,
                        path=(
                            f"{path}/{keyword}/"
                            f"{_encode_json_pointer_token(str(name))}"
                        ),
                        depth=depth + 1,
                    )
                    if isinstance(child, Mapping)
                    else deepcopy(child)
                    for name, child in children.items()
                }
        for keyword in _SCHEMA_SINGLE_KEYWORDS:
            child = raw.get(keyword)
            if isinstance(child, Mapping):
                node[keyword] = visit(
                    child,
                    path=f"{path}/{keyword}",
                    depth=depth + 1,
                )
        for keyword in _SCHEMA_SEQUENCE_KEYWORDS:
            children = raw.get(keyword)
            if isinstance(children, list):
                node[keyword] = [
                    visit(
                        child,
                        path=f"{path}/{keyword}/{index}",
                        depth=depth + 1,
                    )
                    if isinstance(child, Mapping)
                    else deepcopy(child)
                    for index, child in enumerate(children)
                ]
        return node

    adapted = visit(canonical, path="#", depth=0)
    _validate_json_schema(
        adapted, code="anthropic_schema_nullable_adaptation_output_invalid"
    )
    optional_count = _count_optional_parameters(adapted)
    if optional_count > ANTHROPIC_SCHEMA_MAX_OPTIONAL_PARAMETERS:
        raise AnthropicBoundedGenerationError(
            f"anthropic_schema_optional_parameter_limit_exceeded:{optional_count}"
        )
    proof_paths = promotion_paths + stripped_paths
    if set(proof_by_path) != set(proof_paths):
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_nullable_adaptation_proof_roster_mismatch"
        )
    return (
        json.loads(canonical_json_bytes(adapted)),
        promotion_paths,
        stripped_paths,
        [proof_by_path[path] for path in proof_paths],
    )


def adapt_anthropic_nullable_optional_properties(
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the provider-only v6 hybrid grammar without mutating input."""

    adapted, _, _, _ = _adapt_anthropic_nullable_optional_properties(schema)
    return adapted


def promote_anthropic_nullable_optional_properties(
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility name for the v6 hybrid provider-only adaptation."""

    return adapt_anthropic_nullable_optional_properties(schema)


def project_anthropic_preflight_fixture(
    *, value: Any, original_schema: Mapping[str, Any]
) -> Any:
    """Project a known preflight fixture only; never repair a live response."""

    try:
        original = json.loads(canonical_json_bytes(dict(original_schema)))
        canonical_value = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise AnthropicBoundedGenerationError(
            "anthropic_preflight_projection_input_not_canonical_json"
        ) from exc
    _validate_json_schema(original, code="anthropic_preflight_projection_schema_invalid")
    try:
        validator_for(original)(original).validate(canonical_value)
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
            raise
        raise AnthropicBoundedGenerationError(
            "anthropic_preflight_projection_input_schema_invalid"
        ) from exc

    annotated = annotate_anthropic_literal_types(original)
    inlined = inline_anthropic_local_references(annotated)
    adapted, promotion_paths, stripped_paths, _ = (
        _adapt_anthropic_nullable_optional_properties(inlined)
    )
    promotion_set = set(promotion_paths)
    stripped_set = set(stripped_paths)
    visited_nodes = 0

    def project(
        raw_value: Any,
        schema: Mapping[str, Any],
        *,
        path: str,
        depth: int,
    ) -> Any:
        nonlocal visited_nodes
        if depth > _MAX_JSON_DEPTH:
            raise AnthropicBoundedGenerationError(
                "anthropic_preflight_projection_depth_limit_exceeded"
            )
        visited_nodes += 1
        if visited_nodes > _MAX_JSON_NODES:
            raise AnthropicBoundedGenerationError(
                "anthropic_preflight_projection_node_limit_exceeded"
            )
        result = deepcopy(raw_value)

        properties = schema.get("properties")
        if isinstance(result, dict) and isinstance(properties, Mapping):
            for name in sorted(properties):
                child_schema = properties[name]
                child_path = (
                    f"{path}/properties/{_encode_json_pointer_token(str(name))}"
                )
                if child_path in promotion_set and name not in result:
                    result[name] = None
                elif child_path in stripped_set and result.get(name) is None:
                    result.pop(name, None)
                if name in result and isinstance(child_schema, Mapping):
                    result[name] = project(
                        result[name],
                        child_schema,
                        path=child_path,
                        depth=depth + 1,
                    )

        prefix_items = schema.get("prefixItems")
        if isinstance(result, list) and isinstance(prefix_items, list):
            for index, child_schema in enumerate(prefix_items[: len(result)]):
                if isinstance(child_schema, Mapping):
                    result[index] = project(
                        result[index],
                        child_schema,
                        path=f"{path}/prefixItems/{index}",
                        depth=depth + 1,
                    )
        items = schema.get("items")
        if isinstance(result, list) and isinstance(items, Mapping):
            prefix_count = len(prefix_items) if isinstance(prefix_items, list) else 0
            for index in range(prefix_count, len(result)):
                result[index] = project(
                    result[index],
                    items,
                    path=f"{path}/items",
                    depth=depth + 1,
                )

        for keyword in ("oneOf", "anyOf"):
            branches = schema.get(keyword)
            if not isinstance(branches, list):
                continue
            valid_branches = [
                (index, branch)
                for index, branch in enumerate(branches)
                if isinstance(branch, Mapping)
                and validator_for(branch)(branch).is_valid(result)
            ]
            if not valid_branches:
                raise AnthropicBoundedGenerationError(
                    "anthropic_preflight_projection_branch_missing"
                )
            if keyword == "oneOf" and len(valid_branches) != 1:
                raise AnthropicBoundedGenerationError(
                    "anthropic_preflight_projection_oneof_ambiguous"
                )
            candidates = [
                project(
                    result,
                    branch,
                    path=f"{path}/{keyword}/{index}",
                    depth=depth + 1,
                )
                for index, branch in valid_branches
            ]
            candidate_hashes = {hash_canonical(candidate) for candidate in candidates}
            if len(candidate_hashes) != 1:
                raise AnthropicBoundedGenerationError(
                    "anthropic_preflight_projection_anyof_ambiguous"
                )
            result = candidates[0]

        branches = schema.get("allOf")
        if isinstance(branches, list):
            for index, branch in enumerate(branches):
                if isinstance(branch, Mapping):
                    result = project(
                        result,
                        branch,
                        path=f"{path}/allOf/{index}",
                        depth=depth + 1,
                    )
        conditional = schema.get("if")
        if isinstance(conditional, Mapping):
            selected = (
                "then"
                if validator_for(conditional)(conditional).is_valid(result)
                else "else"
            )
            branch = schema.get(selected)
            if isinstance(branch, Mapping):
                result = project(
                    result,
                    branch,
                    path=f"{path}/{selected}",
                    depth=depth + 1,
                )
        dependent = schema.get("dependentSchemas")
        if isinstance(result, dict) and isinstance(dependent, Mapping):
            for name in sorted(set(result) & set(dependent)):
                branch = dependent[name]
                if isinstance(branch, Mapping):
                    result = project(
                        result,
                        branch,
                        path=(
                            f"{path}/dependentSchemas/"
                            f"{_encode_json_pointer_token(str(name))}"
                        ),
                        depth=depth + 1,
                    )
        return result

    projected = project(canonical_value, inlined, path="#", depth=0)
    try:
        validator_for(original)(original).validate(projected)
        validator_for(adapted)(adapted).validate(projected)
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
            raise
        raise AnthropicBoundedGenerationError(
            "anthropic_preflight_projection_postcondition_failed"
        ) from exc
    return json.loads(canonical_json_bytes(projected))


def materialize_anthropic_nullable_optionals(
    *, value: Any, original_schema: Mapping[str, Any]
) -> Any:
    """Compatibility wrapper for preflight-only fixture projection."""

    return project_anthropic_preflight_fixture(
        value=value, original_schema=original_schema
    )


class AnthropicBoundedConfigV1(_FrozenContract):
    """Price- and behavior-pinned settings; credentials are deliberately absent."""

    config_version: Literal["anthropic-bounded-config-v1"] = (
        "anthropic-bounded-config-v1"
    )
    model: Literal["claude-sonnet-5"] = ANTHROPIC_MODEL
    api_base_url: Literal["https://api.anthropic.com"] = ANTHROPIC_API_BASE_URL
    timeout_seconds: Annotated[float, Field(gt=0, le=600, allow_inf_nan=False)]
    input_rate_usd_per_million_tokens: PositiveRate = (
        ANTHROPIC_INPUT_RATE_USD_PER_MTOK
    )
    output_rate_usd_per_million_tokens: PositiveRate = (
        ANTHROPIC_OUTPUT_RATE_USD_PER_MTOK
    )
    fixed_framing_tokens: Literal[1024] = ANTHROPIC_FIXED_FRAMING_TOKENS
    effort: Literal["low"] = "low"
    service_tier: Literal["standard_only"] = "standard_only"
    thinking_mode: Literal["provider_default_adaptive"] = (
        "provider_default_adaptive"
    )
    max_tokens_includes_adaptive_thinking: Literal[True] = True
    sampling_parameters: Literal["provider_defaults_only"] = (
        "provider_defaults_only"
    )
    pricing_source_url: Literal[
        "https://platform.claude.com/docs/en/about-claude/pricing"
    ] = ANTHROPIC_PRICING_SOURCE_URL
    model_source_url: Literal[
        "https://platform.claude.com/docs/en/models/sonnet-5/whats-new-sonnet-5"
    ] = ANTHROPIC_MODEL_SOURCE_URL
    source_verified_date: Literal["2026-08-28"] = ANTHROPIC_SOURCE_VERIFIED_DATE
    pricing_table_sha256: Sha256 = ANTHROPIC_PRICING_TABLE_SHA256

    @model_validator(mode="after")
    def validate_rates(self) -> AnthropicBoundedConfigV1:
        if self.input_rate_usd_per_million_tokens != ANTHROPIC_INPUT_RATE_USD_PER_MTOK:
            raise ValueError("anthropic_input_rate_not_pinned")
        if self.output_rate_usd_per_million_tokens != ANTHROPIC_OUTPUT_RATE_USD_PER_MTOK:
            raise ValueError("anthropic_output_rate_not_pinned")
        if self.pricing_table_sha256 != ANTHROPIC_PRICING_TABLE_SHA256:
            raise ValueError("anthropic_pricing_table_hash_not_pinned")
        return self

    @property
    def config_sha256(self) -> str:
        return hash_canonical(self)


class AnthropicProviderIdentityV1(_FrozenContract):
    identity_version: Literal["anthropic-provider-identity-v1"] = (
        "anthropic-provider-identity-v1"
    )
    provider: Literal["anthropic"] = "anthropic"
    model: Literal["claude-sonnet-5"] = ANTHROPIC_MODEL
    api_base_url: Literal["https://api.anthropic.com"] = ANTHROPIC_API_BASE_URL
    environment_base_url_override_permitted: Literal[False] = False
    environment_custom_headers_override_permitted: Literal[False] = False
    http_environment_trust: Literal[False] = False
    follow_redirects: Literal[False] = False
    client_injection_permitted: Literal[False] = False
    anthropic_version_header: Literal["2023-06-01"] = ANTHROPIC_API_VERSION
    api_operation: Literal["messages.create"] = "messages.create"
    anthropic_api_version: Literal["2023-06-01"] = ANTHROPIC_API_VERSION
    anthropic_sdk_version: Literal["0.120.2"] = ANTHROPIC_SDK_VERSION
    schema_compiler_version: Literal["anthropic-literal-type-compiler-v7"] = (
        ANTHROPIC_SCHEMA_COMPILER_VERSION
    )
    schema_reference_resolution: Literal[
        "deterministic-local-json-pointer-inline-v1"
    ] = "deterministic-local-json-pointer-inline-v1"
    schema_max_reference_expansions: Literal[10000] = (
        ANTHROPIC_SCHEMA_MAX_REFERENCE_EXPANSIONS
    )
    schema_max_inlined_nodes: Literal[100000] = ANTHROPIC_SCHEMA_MAX_INLINED_NODES
    schema_max_inlined_depth: Literal[128] = ANTHROPIC_SCHEMA_MAX_INLINED_DEPTH
    schema_max_inlined_utf8_bytes: Literal[5000000] = (
        ANTHROPIC_SCHEMA_MAX_INLINED_UTF8_BYTES
    )
    schema_nullable_optional_policy: Literal[
        "hybrid-require-ten-nullable-optionals-strip-null-from-remainder-v1"
    ] = "hybrid-require-ten-nullable-optionals-strip-null-from-remainder-v1"
    schema_max_nullable_optional_promotions: Literal[10000] = (
        ANTHROPIC_SCHEMA_MAX_NULLABLE_OPTIONAL_PROMOTIONS
    )
    schema_nullable_optional_required_target: Literal[10] = (
        ANTHROPIC_SCHEMA_NULLABLE_OPTIONAL_REQUIRED_TARGET
    )
    schema_nullable_retain_priority_sha256: Literal[
        "e44a8430b4c719ba41935ef889b93c980fc631e959473581750ca6214cc9ff58"
    ] = ANTHROPIC_NULLABLE_RETAIN_PRIORITY_SHA256
    schema_max_optional_parameters: Literal[24] = (
        ANTHROPIC_SCHEMA_MAX_OPTIONAL_PARAMETERS
    )
    schema_max_union_parameters: Literal[16] = ANTHROPIC_SCHEMA_MAX_UNION_PARAMETERS
    response_schema_validation_order: Literal[
        "wire-then-original-then-runtime-full-acceptance"
    ] = "wire-then-original-then-runtime-full-acceptance"
    transport_mode_policy: Literal[
        "inventory-structured-json-schema-packet-prompt-json-schema-v1"
    ] = "inventory-structured-json-schema-packet-prompt-json-schema-v1"
    prompt_json_system_envelope_version: Literal[
        "anthropic-prompt-json-system-envelope-v1"
    ] = (
        ANTHROPIC_PROMPT_JSON_SYSTEM_ENVELOPE_VERSION
    )
    prompt_json_system_envelope_sha256: Sha256 = (
        ANTHROPIC_PROMPT_JSON_SYSTEM_ENVELOPE_SHA256
    )
    prompt_json_output_config: Literal["effort-only-no-format"] = (
        "effort-only-no-format"
    )
    sdk_max_retries: Literal[0] = 0
    application_retry_count: Literal[0] = 0
    transport_attempts_per_request: Literal[1] = 1
    timeout_seconds: Annotated[float, Field(gt=0, le=600, allow_inf_nan=False)]
    effort: Literal["low"] = "low"
    service_tier: Literal["standard_only"] = "standard_only"
    thinking_mode: Literal["provider_default_adaptive"] = (
        "provider_default_adaptive"
    )
    sampling_parameters: Literal["provider_defaults_only"] = (
        "provider_defaults_only"
    )
    rate_limit_scope: Literal["credential_defined_not_archived"] = (
        "credential_defined_not_archived"
    )
    config_sha256: Sha256

    @model_validator(mode="after")
    def validate_transport_policy(self) -> AnthropicProviderIdentityV1:
        if (
            self.prompt_json_system_envelope_sha256
            != ANTHROPIC_PROMPT_JSON_SYSTEM_ENVELOPE_SHA256
        ):
            raise ValueError("anthropic_prompt_json_system_envelope_hash_mismatch")
        return self

    @property
    def identity_sha256(self) -> str:
        return hash_canonical(self)


def freeze_anthropic_provider_identity(
    config: AnthropicBoundedConfigV1,
) -> AnthropicProviderIdentityV1:
    """Freeze the exact offline SDK/model identity without reading a credential."""

    _require_exact_sdk()
    return AnthropicProviderIdentityV1(
        timeout_seconds=config.timeout_seconds,
        config_sha256=config.config_sha256,
    )


class AnthropicNullableAdaptationProofV1(_FrozenContract):
    path: str
    action: Literal["require_nullable", "keep_optional_strip_null"]
    strategy: Literal[
        "required-membership-only",
        "remove-null-from-single-nonnull-type-array",
        "collapse-exact-two-branch-anyof",
        "collapse-exact-two-branch-oneof",
    ]
    original_property_schema_sha256: Sha256
    provider_property_schema_sha256: Sha256
    required_before: bool
    required_after: bool
    accepts_null_before: bool
    accepts_null_after: bool

    @model_validator(mode="after")
    def validate_proof(self) -> AnthropicNullableAdaptationProofV1:
        if not self.path.startswith("#/"):
            raise ValueError("anthropic_nullable_adaptation_proof_path_invalid")
        if not self.accepts_null_before or self.required_before:
            raise ValueError("anthropic_nullable_adaptation_proof_input_invalid")
        if self.action == "require_nullable":
            if (
                self.strategy != "required-membership-only"
                or
                not self.required_after
                or not self.accepts_null_after
                or self.original_property_schema_sha256
                != self.provider_property_schema_sha256
            ):
                raise ValueError("anthropic_nullable_promotion_proof_invalid")
        elif (
            self.strategy == "required-membership-only"
            or self.required_after
            or self.accepts_null_after
            or self.original_property_schema_sha256
            == self.provider_property_schema_sha256
        ):
            raise ValueError("anthropic_nullable_null_stripping_proof_invalid")
        return self


class AnthropicCompiledSchemaV1(_FrozenContract):
    compiled_schema_version: Literal["anthropic-compiled-schema-v2"] = (
        "anthropic-compiled-schema-v2"
    )
    compiler_version: Literal["anthropic-literal-type-compiler-v7"] = (
        ANTHROPIC_SCHEMA_COMPILER_VERSION
    )
    anthropic_sdk_version: Literal["0.120.2"] = ANTHROPIC_SDK_VERSION
    original_schema: dict[str, Any]
    original_schema_sha256: Sha256
    literal_annotated_schema: dict[str, Any]
    literal_annotated_schema_sha256: Sha256
    provider_hybrid_schema_sha256: Sha256
    nullable_optional_promotion_paths: list[str]
    nullable_optional_promotion_paths_sha256: Sha256
    nullable_optional_promotion_count: Annotated[StrictInt, Field(ge=0, le=10000)]
    nullable_optional_null_stripping_paths: list[str]
    nullable_optional_null_stripping_paths_sha256: Sha256
    nullable_optional_null_stripping_count: Annotated[
        StrictInt, Field(ge=0, le=10000)
    ]
    nullable_optional_path_partition_sha256: Sha256
    nullable_optional_candidate_paths: list[str]
    nullable_optional_candidate_paths_sha256: Sha256
    nullable_optional_candidate_count: Annotated[StrictInt, Field(ge=0, le=10000)]
    nullable_optional_adaptation_proofs: list[AnthropicNullableAdaptationProofV1]
    nullable_optional_adaptation_proofs_sha256: Sha256
    nullable_retain_priority_sha256: Literal[
        "e44a8430b4c719ba41935ef889b93c980fc631e959473581750ca6214cc9ff58"
    ] = ANTHROPIC_NULLABLE_RETAIN_PRIORITY_SHA256
    pre_promotion_optional_parameter_count: Annotated[StrictInt, Field(ge=0)]
    post_promotion_optional_parameter_count: Annotated[StrictInt, Field(ge=0, le=24)]
    pre_adaptation_union_parameter_count: Annotated[StrictInt, Field(ge=0)]
    post_adaptation_union_parameter_count: Annotated[StrictInt, Field(ge=0, le=16)]
    wire_schema: dict[str, Any]
    wire_schema_sha256: Sha256
    wire_optional_parameter_count: Annotated[StrictInt, Field(ge=0, le=24)]
    wire_union_parameter_count: Annotated[StrictInt, Field(ge=0, le=16)]
    provider_optional_parameter_limit: Literal[24] = ANTHROPIC_SCHEMA_MAX_OPTIONAL_PARAMETERS
    provider_union_parameter_limit: Literal[16] = ANTHROPIC_SCHEMA_MAX_UNION_PARAMETERS
    provider_nullable_optional_required_target: Literal[10] = (
        ANTHROPIC_SCHEMA_NULLABLE_OPTIONAL_REQUIRED_TARGET
    )
    provider_grammar_delta: Literal[
        "first-ten-nullable-optionals-required-nullable-remainder-optional-nonnull"
    ] = "first-ten-nullable-optionals-required-nullable-remainder-optional-nonnull"
    nullable_null_stripping_proof: Literal[
        "exact-isolated-null-branch-or-single-nonnull-type-array-rewrite-v1"
    ] = (
        "exact-isolated-null-branch-or-single-nonnull-type-array-rewrite-v1"
    )
    full_acceptance_schema_sha256: Sha256
    wire_schema_scientific_authority: Literal["none"] = "none"
    local_response_validation_schema: Literal["original_schema"] = "original_schema"
    response_schema_validation_order: Literal[
        "wire_schema_then_original_schema_then_downstream_full_acceptance"
    ] = "wire_schema_then_original_schema_then_downstream_full_acceptance"
    downstream_raw_acceptance_required: Literal[True] = True

    @model_validator(mode="after")
    def validate_schema_hashes(self) -> AnthropicCompiledSchemaV1:
        for value, expected, code in (
            (
                self.original_schema,
                self.original_schema_sha256,
                "anthropic_original_schema_hash_mismatch",
            ),
            (
                self.literal_annotated_schema,
                self.literal_annotated_schema_sha256,
                "anthropic_annotated_schema_hash_mismatch",
            ),
            (
                self.wire_schema,
                self.wire_schema_sha256,
                "anthropic_wire_schema_hash_mismatch",
            ),
        ):
            if hash_canonical(value) != expected:
                raise ValueError(code)
        _validate_json_schema(
            self.original_schema, code="anthropic_original_schema_invalid"
        )
        _validate_json_schema(
            self.literal_annotated_schema,
            code="anthropic_annotated_schema_invalid",
        )
        _validate_json_schema(self.wire_schema, code="anthropic_wire_schema_invalid")
        _audit_no_stale_reference_metadata(self.wire_schema)
        if (
            len(self.nullable_optional_promotion_paths)
            != len(set(self.nullable_optional_promotion_paths))
            or any(
                not path.startswith("#/")
                for path in self.nullable_optional_promotion_paths
            )
        ):
            raise ValueError("anthropic_nullable_promotion_paths_invalid")
        if (
            self.nullable_optional_promotion_count
            != len(self.nullable_optional_promotion_paths)
            or self.nullable_optional_promotion_paths_sha256
            != hash_canonical(self.nullable_optional_promotion_paths)
        ):
            raise ValueError("anthropic_nullable_promotion_path_binding_mismatch")
        if (
            self.nullable_optional_null_stripping_paths
            != sorted(set(self.nullable_optional_null_stripping_paths))
            or any(
                not path.startswith("#/")
                for path in self.nullable_optional_null_stripping_paths
            )
        ):
            raise ValueError("anthropic_nullable_null_stripping_paths_invalid")
        if (
            self.nullable_optional_null_stripping_count
            != len(self.nullable_optional_null_stripping_paths)
            or self.nullable_optional_null_stripping_paths_sha256
            != hash_canonical(self.nullable_optional_null_stripping_paths)
            or set(self.nullable_optional_promotion_paths).intersection(
                self.nullable_optional_null_stripping_paths
            )
        ):
            raise ValueError("anthropic_nullable_null_stripping_path_binding_mismatch")
        path_partition = {
            "required_nullable": self.nullable_optional_promotion_paths,
            "optional_nonnull": self.nullable_optional_null_stripping_paths,
        }
        if self.nullable_optional_path_partition_sha256 != hash_canonical(
            path_partition
        ):
            raise ValueError("anthropic_nullable_path_partition_hash_mismatch")
        candidate_paths = sorted(
            self.nullable_optional_promotion_paths
            + self.nullable_optional_null_stripping_paths
        )
        if (
            self.nullable_optional_candidate_paths != candidate_paths
            or self.nullable_optional_candidate_count != len(candidate_paths)
            or self.nullable_optional_candidate_paths_sha256
            != hash_canonical(candidate_paths)
        ):
            raise ValueError("anthropic_nullable_candidate_path_binding_mismatch")
        expected_proof_paths = (
            self.nullable_optional_promotion_paths
            + self.nullable_optional_null_stripping_paths
        )
        proof_payload = [
            item.model_dump(mode="json")
            for item in self.nullable_optional_adaptation_proofs
        ]
        if (
            [item.path for item in self.nullable_optional_adaptation_proofs]
            != expected_proof_paths
            or self.nullable_optional_adaptation_proofs_sha256
            != hash_canonical(proof_payload)
        ):
            raise ValueError("anthropic_nullable_adaptation_proof_binding_mismatch")
        replayed_original = json.loads(canonical_json_bytes(self.original_schema))
        replayed_annotated = annotate_anthropic_literal_types(replayed_original)
        if hash_canonical(replayed_annotated) != self.literal_annotated_schema_sha256:
            raise ValueError("anthropic_annotated_schema_replay_mismatch")
        try:
            replayed_inlined = inline_anthropic_local_references(replayed_annotated)
            (
                replayed_adapted,
                replayed_promotions,
                replayed_stripped,
                replayed_proofs,
            ) = (
                _adapt_anthropic_nullable_optional_properties(replayed_inlined)
            )
            replayed_wire = _require_exact_sdk().transform_schema(
                deepcopy(replayed_adapted)
            )
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
                raise
            raise ValueError("anthropic_wire_schema_replay_failed") from exc
        if (
            hash_canonical(replayed_adapted) != self.provider_hybrid_schema_sha256
            or replayed_promotions != self.nullable_optional_promotion_paths
            or replayed_stripped != self.nullable_optional_null_stripping_paths
            or replayed_proofs != self.nullable_optional_adaptation_proofs
            or _count_optional_parameters(replayed_inlined)
            != self.pre_promotion_optional_parameter_count
            or _count_optional_parameters(replayed_adapted)
            != self.post_promotion_optional_parameter_count
            or _count_union_parameters(replayed_inlined)
            != self.pre_adaptation_union_parameter_count
            or _count_union_parameters(replayed_adapted)
            != self.post_adaptation_union_parameter_count
            or _count_optional_parameters(self.wire_schema)
            != self.wire_optional_parameter_count
            or _count_union_parameters(self.wire_schema)
            != self.wire_union_parameter_count
        ):
            raise ValueError("anthropic_nullable_adaptation_replay_mismatch")
        if hash_canonical(replayed_wire) != self.wire_schema_sha256:
            raise ValueError("anthropic_wire_schema_replay_mismatch")
        return self

    @property
    def compiled_schema_sha256(self) -> str:
        return hash_canonical(self)


def compile_anthropic_bounded_schema(
    *,
    original_schema: Mapping[str, Any],
    full_acceptance_schema_sha256: str,
) -> AnthropicCompiledSchemaV1:
    """Compile and hash all three schema stages without mutating the input."""

    if not SHA256_RE.fullmatch(full_acceptance_schema_sha256):
        raise AnthropicBoundedGenerationError(
            "anthropic_full_acceptance_schema_hash_invalid"
        )
    try:
        # Canonical key insertion order is part of the compiler algorithm because
        # the SDK derives set-like ``required`` arrays by iterating object keys.
        # This keeps replay stable after fsync-backed, sort-key JSON persistence.
        original = json.loads(canonical_json_bytes(dict(original_schema)))
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise AnthropicBoundedGenerationError(
            "anthropic_original_schema_not_canonical_json"
        ) from exc
    _validate_json_schema(original, code="anthropic_original_schema_invalid")
    annotated = annotate_anthropic_literal_types(original)
    _validate_json_schema(annotated, code="anthropic_annotated_schema_invalid")
    inlined = inline_anthropic_local_references(annotated)
    _validate_json_schema(inlined, code="anthropic_inlined_schema_invalid")
    pre_promotion_optional_count = _count_optional_parameters(inlined)
    pre_adaptation_union_count = _count_union_parameters(inlined)
    provider_adapted, promotion_paths, stripped_paths, adaptation_proofs = (
        _adapt_anthropic_nullable_optional_properties(inlined)
    )
    post_promotion_optional_count = _count_optional_parameters(provider_adapted)
    post_adaptation_union_count = _count_union_parameters(provider_adapted)
    if post_adaptation_union_count > ANTHROPIC_SCHEMA_MAX_UNION_PARAMETERS:
        raise AnthropicBoundedGenerationError(
            "anthropic_schema_union_parameter_limit_exceeded:"
            f"{post_adaptation_union_count}"
        )
    anthropic = _require_exact_sdk()
    try:
        transformed = anthropic.transform_schema(deepcopy(provider_adapted))
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
            raise
        raise AnthropicBoundedGenerationError(
            "anthropic_wire_schema_transform_failed"
        ) from exc
    if not isinstance(transformed, dict):  # pragma: no cover - SDK contract guard
        raise AnthropicBoundedGenerationError(
            "anthropic_wire_schema_transform_not_object"
        )
    wire = deepcopy(transformed)
    _validate_json_schema(wire, code="anthropic_wire_schema_invalid")
    _audit_no_stale_reference_metadata(wire)
    wire_optional_count = _count_optional_parameters(wire)
    wire_union_count = _count_union_parameters(wire)
    if wire_optional_count > ANTHROPIC_SCHEMA_MAX_OPTIONAL_PARAMETERS:
        raise AnthropicBoundedGenerationError(
            f"anthropic_wire_optional_parameter_limit_exceeded:{wire_optional_count}"
        )
    if wire_union_count > ANTHROPIC_SCHEMA_MAX_UNION_PARAMETERS:
        raise AnthropicBoundedGenerationError(
            f"anthropic_wire_union_parameter_limit_exceeded:{wire_union_count}"
        )
    path_partition = {
        "required_nullable": promotion_paths,
        "optional_nonnull": stripped_paths,
    }
    candidate_paths = sorted(promotion_paths + stripped_paths)
    adaptation_proof_payload = [
        item.model_dump(mode="json") for item in adaptation_proofs
    ]
    return AnthropicCompiledSchemaV1(
        original_schema=original,
        original_schema_sha256=hash_canonical(original),
        literal_annotated_schema=annotated,
        literal_annotated_schema_sha256=hash_canonical(annotated),
        provider_hybrid_schema_sha256=hash_canonical(provider_adapted),
        nullable_optional_promotion_paths=promotion_paths,
        nullable_optional_promotion_paths_sha256=hash_canonical(promotion_paths),
        nullable_optional_promotion_count=len(promotion_paths),
        nullable_optional_null_stripping_paths=stripped_paths,
        nullable_optional_null_stripping_paths_sha256=hash_canonical(stripped_paths),
        nullable_optional_null_stripping_count=len(stripped_paths),
        nullable_optional_path_partition_sha256=hash_canonical(path_partition),
        nullable_optional_candidate_paths=candidate_paths,
        nullable_optional_candidate_paths_sha256=hash_canonical(candidate_paths),
        nullable_optional_candidate_count=len(candidate_paths),
        nullable_optional_adaptation_proofs=adaptation_proofs,
        nullable_optional_adaptation_proofs_sha256=hash_canonical(
            adaptation_proof_payload
        ),
        pre_promotion_optional_parameter_count=pre_promotion_optional_count,
        post_promotion_optional_parameter_count=post_promotion_optional_count,
        pre_adaptation_union_parameter_count=pre_adaptation_union_count,
        post_adaptation_union_parameter_count=post_adaptation_union_count,
        wire_schema=wire,
        wire_schema_sha256=hash_canonical(wire),
        wire_optional_parameter_count=wire_optional_count,
        wire_union_parameter_count=wire_union_count,
        full_acceptance_schema_sha256=full_acceptance_schema_sha256,
    )


def _transport_mode_for_schema_kind(
    *, schema_kind: AnthropicSchemaKind, effect_kind: AnthropicEffectKind | None
) -> AnthropicTransportMode:
    if schema_kind == "inventory":
        if effect_kind is not None:
            raise AnthropicBoundedGenerationError(
                "anthropic_inventory_effect_kind_forbidden"
            )
        return "structured_json_schema"
    if effect_kind is None:
        raise AnthropicBoundedGenerationError("anthropic_packet_effect_kind_missing")
    return "prompt_json_schema"


def render_anthropic_prompt_json_model_system(
    *, base_system: str, wire_schema: Mapping[str, Any]
) -> str:
    """Append one canonical wire schema as the terminal system-message bytes."""

    if not isinstance(base_system, str) or not base_system.strip():
        raise AnthropicBoundedGenerationError("anthropic_base_system_blank")
    try:
        canonical_schema = canonical_json_bytes(dict(wire_schema)).decode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise AnthropicBoundedGenerationError(
            "anthropic_prompt_json_wire_schema_not_canonical"
        ) from exc
    _validate_json_schema(
        json.loads(canonical_schema), code="anthropic_prompt_json_wire_schema_invalid"
    )
    schema_sha256 = hash_canonical(wire_schema)
    forbidden_markers = (
        "PROMPT_JSON_SYSTEM_ENVELOPE=",
        "PROMPT_JSON_SYSTEM_ENVELOPE_SHA256=",
        "WIRE_SCHEMA_SHA256=",
        "WIRE_SCHEMA_UTF8_BYTES=",
        "OUTPUT_SCHEMA_JSON_FOLLOWS_TO_END_OF_SYSTEM",
    )
    if any(marker in base_system for marker in forbidden_markers):
        raise AnthropicBoundedGenerationError(
            "anthropic_prompt_json_base_system_marker_collision"
        )
    if canonical_schema in base_system:
        raise AnthropicBoundedGenerationError(
            "anthropic_prompt_json_base_system_schema_collision"
        )
    rendered = (
        f"{base_system}\n\n{ANTHROPIC_PROMPT_JSON_SYSTEM_PREAMBLE}\n"
        "PROMPT_JSON_SYSTEM_ENVELOPE="
        f"{ANTHROPIC_PROMPT_JSON_SYSTEM_ENVELOPE_VERSION}\n"
        "PROMPT_JSON_SYSTEM_ENVELOPE_SHA256="
        f"{ANTHROPIC_PROMPT_JSON_SYSTEM_ENVELOPE_SHA256}\n"
        f"WIRE_SCHEMA_SHA256={schema_sha256}\n"
        f"WIRE_SCHEMA_UTF8_BYTES={len(canonical_schema.encode('utf-8'))}\n"
        "OUTPUT_SCHEMA_JSON_FOLLOWS_TO_END_OF_SYSTEM\n"
        f"{canonical_schema}"
    )
    if not rendered.endswith(canonical_schema) or rendered.count(canonical_schema) != 1:
        raise AnthropicBoundedGenerationError(
            "anthropic_prompt_json_system_envelope_postcondition_failed"
        )
    return rendered


def _build_anthropic_wire_call_surface(
    *,
    model: str,
    max_output_tokens: int,
    model_system: str,
    model_prompt: str,
    transport_mode: AnthropicTransportMode,
    wire_schema: Mapping[str, Any],
    effort: str,
    service_tier: str,
) -> tuple[dict[str, Any], str]:
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_output_tokens,
        "system": model_system,
        "messages": [{"role": "user", "content": model_prompt}],
        "output_config": {"effort": effort},
        "service_tier": service_tier,
    }
    if transport_mode == "structured_json_schema":
        kwargs["output_config"]["format"] = {
            "type": "json_schema",
            "schema": deepcopy(dict(wire_schema)),
        }
    wire_call_sha256 = hash_canonical(
        {
            "api_base_url": ANTHROPIC_API_BASE_URL,
            "anthropic_api_version": ANTHROPIC_API_VERSION,
            "environment_transport_overrides_permitted": False,
            "http_environment_trust": False,
            "follow_redirects": False,
            "transport_mode": transport_mode,
            "request_kwargs": kwargs,
        }
    )
    return kwargs, wire_call_sha256


class AnthropicRequestCostCeilingV1(_FrozenContract):
    cost_ceiling_version: Literal["anthropic-request-cost-ceiling-v2"] = (
        "anthropic-request-cost-ceiling-v2"
    )
    token_bound_method: Literal[
        "one_token_per_model_facing_utf8_byte_plus_fixed_framing"
    ] = (
        "one_token_per_model_facing_utf8_byte_plus_fixed_framing"
    )
    transport_mode: AnthropicTransportMode
    base_system_utf8_bytes: TokenCount
    model_system_utf8_bytes: TokenCount
    base_prompt_utf8_bytes: TokenCount
    model_prompt_utf8_bytes: TokenCount
    canonical_wire_schema_utf8_bytes: TokenCount
    structured_format_schema_utf8_bytes: TokenCount
    embedded_system_schema_utf8_bytes: TokenCount
    model_facing_input_utf8_bytes: TokenCount
    fixed_framing_tokens: Literal[1024] = ANTHROPIC_FIXED_FRAMING_TOKENS
    conservative_input_token_ceiling: TokenCount
    max_output_tokens: Annotated[StrictInt, Field(ge=1, le=_MAX_OUTPUT_TOKENS)]
    input_rate_usd_per_million_tokens: PositiveRate
    output_rate_usd_per_million_tokens: PositiveRate
    request_cost_ceiling_usd: Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]

    @model_validator(mode="after")
    def validate_ceiling(self) -> AnthropicRequestCostCeilingV1:
        if (
            self.input_rate_usd_per_million_tokens
            != ANTHROPIC_INPUT_RATE_USD_PER_MTOK
            or self.output_rate_usd_per_million_tokens
            != ANTHROPIC_OUTPUT_RATE_USD_PER_MTOK
        ):
            raise ValueError("anthropic_request_cost_rates_not_pinned")
        if self.transport_mode == "structured_json_schema":
            if (
                self.model_system_utf8_bytes != self.base_system_utf8_bytes
                or
                self.model_prompt_utf8_bytes != self.base_prompt_utf8_bytes
                or self.structured_format_schema_utf8_bytes
                != self.canonical_wire_schema_utf8_bytes
                or self.embedded_system_schema_utf8_bytes != 0
            ):
                raise ValueError("anthropic_structured_cost_surface_invalid")
        elif (
            self.structured_format_schema_utf8_bytes != 0
            or self.embedded_system_schema_utf8_bytes
            != self.canonical_wire_schema_utf8_bytes
            or self.model_prompt_utf8_bytes != self.base_prompt_utf8_bytes
            or self.model_system_utf8_bytes
            < self.base_system_utf8_bytes + self.canonical_wire_schema_utf8_bytes
        ):
            raise ValueError("anthropic_prompt_json_cost_surface_invalid")
        expected_model_facing = (
            self.model_system_utf8_bytes
            + self.model_prompt_utf8_bytes
            + self.structured_format_schema_utf8_bytes
        )
        if self.model_facing_input_utf8_bytes != expected_model_facing:
            raise ValueError("anthropic_model_facing_byte_count_mismatch")
        expected_input = expected_model_facing + self.fixed_framing_tokens
        if self.conservative_input_token_ceiling != expected_input:
            raise ValueError("anthropic_input_token_ceiling_mismatch")
        expected_cost = (
            Decimal(expected_input) * self.input_rate_usd_per_million_tokens
            + Decimal(self.max_output_tokens)
            * self.output_rate_usd_per_million_tokens
        ) / Decimal(1_000_000)
        if self.request_cost_ceiling_usd != expected_cost:
            raise ValueError("anthropic_request_cost_ceiling_mismatch")
        return self

    @property
    def cost_ceiling_sha256(self) -> str:
        return hash_canonical(self)


def compute_anthropic_request_cost_ceiling(
    *,
    config: AnthropicBoundedConfigV1,
    system: str,
    prompt: str,
    wire_schema: Mapping[str, Any],
    max_output_tokens: int,
    transport_mode: AnthropicTransportMode = "structured_json_schema",
) -> AnthropicRequestCostCeilingV1:
    """Compute the frozen pre-call ceiling from every model-facing byte surface."""

    if isinstance(max_output_tokens, bool) or not 1 <= max_output_tokens <= _MAX_OUTPUT_TOKENS:
        raise AnthropicBoundedGenerationError("anthropic_max_output_tokens_invalid")
    if transport_mode not in {"structured_json_schema", "prompt_json_schema"}:
        raise AnthropicBoundedGenerationError("anthropic_transport_mode_invalid")
    base_system_bytes = len(system.encode("utf-8"))
    base_prompt_bytes = len(prompt.encode("utf-8"))
    schema_bytes = len(canonical_json_bytes(wire_schema))
    model_system = (
        system
        if transport_mode == "structured_json_schema"
        else render_anthropic_prompt_json_model_system(
            base_system=system, wire_schema=wire_schema
        )
    )
    model_prompt = prompt
    model_system_bytes = len(model_system.encode("utf-8"))
    model_prompt_bytes = len(model_prompt.encode("utf-8"))
    structured_schema_bytes = (
        schema_bytes if transport_mode == "structured_json_schema" else 0
    )
    embedded_system_schema_bytes = (
        schema_bytes if transport_mode == "prompt_json_schema" else 0
    )
    model_facing_bytes = (
        model_system_bytes + model_prompt_bytes + structured_schema_bytes
    )
    input_ceiling = model_facing_bytes + config.fixed_framing_tokens
    cost = (
        Decimal(input_ceiling) * config.input_rate_usd_per_million_tokens
        + Decimal(max_output_tokens) * config.output_rate_usd_per_million_tokens
    ) / Decimal(1_000_000)
    return AnthropicRequestCostCeilingV1(
        transport_mode=transport_mode,
        base_system_utf8_bytes=base_system_bytes,
        model_system_utf8_bytes=model_system_bytes,
        base_prompt_utf8_bytes=base_prompt_bytes,
        model_prompt_utf8_bytes=model_prompt_bytes,
        canonical_wire_schema_utf8_bytes=schema_bytes,
        structured_format_schema_utf8_bytes=structured_schema_bytes,
        embedded_system_schema_utf8_bytes=embedded_system_schema_bytes,
        model_facing_input_utf8_bytes=model_facing_bytes,
        conservative_input_token_ceiling=input_ceiling,
        max_output_tokens=max_output_tokens,
        input_rate_usd_per_million_tokens=config.input_rate_usd_per_million_tokens,
        output_rate_usd_per_million_tokens=config.output_rate_usd_per_million_tokens,
        request_cost_ceiling_usd=cost,
    )


class AnthropicBoundedRequestV1(_FrozenContract):
    request_version: Literal["anthropic-bounded-request-v2"] = (
        "anthropic-bounded-request-v2"
    )
    operation: Annotated[str, Field(min_length=1, max_length=128)]
    request_key: Annotated[str, Field(min_length=1, max_length=256)]
    model: Literal["claude-sonnet-5"] = ANTHROPIC_MODEL
    identity_sha256: Sha256
    config_sha256: Sha256
    compiled_schema: AnthropicCompiledSchemaV1
    compiled_schema_sha256: Sha256
    full_acceptance_schema_sha256: Sha256
    schema_kind: AnthropicSchemaKind
    effect_kind: AnthropicEffectKind | None
    transport_mode: AnthropicTransportMode
    transport_policy_binding_sha256: Sha256
    prompt_json_system_envelope_sha256: Sha256
    system: Annotated[str, Field(min_length=1, max_length=_MAX_TEXT_CHARACTERS)]
    base_system_sha256: Sha256
    model_system: Annotated[str, Field(min_length=1, max_length=_MAX_TEXT_CHARACTERS)]
    model_system_sha256: Sha256
    prompt: Annotated[str, Field(min_length=1, max_length=_MAX_TEXT_CHARACTERS)]
    base_prompt_sha256: Sha256
    model_prompt: Annotated[str, Field(min_length=1, max_length=_MAX_TEXT_CHARACTERS)]
    model_prompt_sha256: Sha256
    wire_schema_delivery: Literal[
        "structured_output_config_format", "canonical_model_system"
    ]
    structured_grammar_enforced_by_provider: bool
    output_format_present_in_call: bool
    expected_wire_call_sha256: Sha256
    max_output_tokens: Annotated[StrictInt, Field(ge=1, le=_MAX_OUTPUT_TOKENS)]
    cost_ceiling: AnthropicRequestCostCeilingV1
    request_sha256: Sha256

    @field_validator(
        "operation",
        "request_key",
        "system",
        "model_system",
        "prompt",
        "model_prompt",
    )
    @classmethod
    def require_non_whitespace_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("anthropic_request_text_blank")
        return value

    @model_validator(mode="after")
    def validate_bindings(self) -> AnthropicBoundedRequestV1:
        if self.compiled_schema_sha256 != self.compiled_schema.compiled_schema_sha256:
            raise ValueError("anthropic_request_compiled_schema_hash_mismatch")
        if (
            self.full_acceptance_schema_sha256
            != self.compiled_schema.full_acceptance_schema_sha256
        ):
            raise ValueError("anthropic_request_full_acceptance_hash_mismatch")
        try:
            expected_mode = _transport_mode_for_schema_kind(
                schema_kind=self.schema_kind,
                effect_kind=self.effect_kind,
            )
        except AnthropicBoundedGenerationError as exc:
            raise ValueError("anthropic_request_schema_transport_invalid") from exc
        if self.transport_mode != expected_mode:
            raise ValueError("anthropic_request_transport_mode_drift")
        expected_delivery = (
            "structured_output_config_format"
            if expected_mode == "structured_json_schema"
            else "canonical_model_system"
        )
        if self.wire_schema_delivery != expected_delivery:
            raise ValueError("anthropic_request_wire_schema_delivery_mismatch")
        structured = expected_mode == "structured_json_schema"
        if (
            self.structured_grammar_enforced_by_provider != structured
            or self.output_format_present_in_call != structured
        ):
            raise ValueError("anthropic_request_format_presence_mismatch")
        expected_model_system = (
            self.system
            if expected_mode == "structured_json_schema"
            else render_anthropic_prompt_json_model_system(
                base_system=self.system,
                wire_schema=self.compiled_schema.wire_schema,
            )
        )
        expected_model_prompt = self.prompt
        if (
            self.base_system_sha256 != _sha256_utf8(self.system)
            or self.model_system != expected_model_system
            or self.model_system_sha256 != _sha256_utf8(expected_model_system)
            or
            self.base_prompt_sha256 != _sha256_utf8(self.prompt)
            or self.model_prompt != expected_model_prompt
            or self.model_prompt_sha256 != _sha256_utf8(expected_model_prompt)
            or self.prompt_json_system_envelope_sha256
            != ANTHROPIC_PROMPT_JSON_SYSTEM_ENVELOPE_SHA256
        ):
            raise ValueError("anthropic_request_model_surface_binding_mismatch")
        _, expected_wire_call_sha256 = _build_anthropic_wire_call_surface(
            model=self.model,
            max_output_tokens=self.max_output_tokens,
            model_system=self.model_system,
            model_prompt=self.model_prompt,
            transport_mode=self.transport_mode,
            wire_schema=self.compiled_schema.wire_schema,
            effort="low",
            service_tier="standard_only",
        )
        if self.expected_wire_call_sha256 != expected_wire_call_sha256:
            raise ValueError("anthropic_request_wire_call_hash_mismatch")
        expected_policy_binding = hash_canonical(
            {
                "policy": (
                    "inventory-structured-json-schema-packet-prompt-json-schema-v1"
                ),
                "schema_kind": self.schema_kind,
                "effect_kind": self.effect_kind,
                "transport_mode": self.transport_mode,
                "compiled_schema_sha256": self.compiled_schema_sha256,
                "wire_schema_sha256": self.compiled_schema.wire_schema_sha256,
            }
        )
        if self.transport_policy_binding_sha256 != expected_policy_binding:
            raise ValueError("anthropic_request_transport_policy_binding_mismatch")
        if self.max_output_tokens != self.cost_ceiling.max_output_tokens:
            raise ValueError("anthropic_request_output_ceiling_mismatch")
        schema_bytes = len(canonical_json_bytes(self.compiled_schema.wire_schema))
        expected_structured_bytes = (
            schema_bytes if expected_mode == "structured_json_schema" else 0
        )
        expected_embedded_bytes = (
            schema_bytes if expected_mode == "prompt_json_schema" else 0
        )
        if (
            self.cost_ceiling.transport_mode != expected_mode
            or self.cost_ceiling.base_system_utf8_bytes
            != len(self.system.encode("utf-8"))
            or self.cost_ceiling.model_system_utf8_bytes
            != len(self.model_system.encode("utf-8"))
            or self.cost_ceiling.base_prompt_utf8_bytes
            != len(self.prompt.encode("utf-8"))
            or self.cost_ceiling.model_prompt_utf8_bytes
            != len(self.model_prompt.encode("utf-8"))
            or self.cost_ceiling.canonical_wire_schema_utf8_bytes != schema_bytes
            or self.cost_ceiling.structured_format_schema_utf8_bytes
            != expected_structured_bytes
            or self.cost_ceiling.embedded_system_schema_utf8_bytes
            != expected_embedded_bytes
        ):
            raise ValueError("anthropic_request_cost_ceiling_replay_mismatch")
        expected = hash_canonical(
            self.model_dump(mode="json", exclude={"request_sha256"})
        )
        if self.request_sha256 != expected:
            raise ValueError("anthropic_request_hash_mismatch")
        return self


def freeze_anthropic_bounded_request(
    *,
    operation: str,
    request_key: str,
    prompt: str,
    system: str,
    max_output_tokens: int,
    compiled_schema: AnthropicCompiledSchemaV1,
    config: AnthropicBoundedConfigV1,
    schema_kind: AnthropicSchemaKind,
    effect_kind: AnthropicEffectKind | None,
    identity: AnthropicProviderIdentityV1 | None = None,
) -> AnthropicBoundedRequestV1:
    """Freeze a complete, credential-free request before any transport call."""

    frozen_identity = identity or freeze_anthropic_provider_identity(config)
    if frozen_identity.config_sha256 != config.config_sha256:
        raise AnthropicBoundedGenerationError(
            "anthropic_request_identity_config_mismatch"
        )
    transport_mode = _transport_mode_for_schema_kind(
        schema_kind=schema_kind,
        effect_kind=effect_kind,
    )
    model_system = (
        system
        if transport_mode == "structured_json_schema"
        else render_anthropic_prompt_json_model_system(
            base_system=system,
            wire_schema=compiled_schema.wire_schema,
        )
    )
    model_prompt = prompt
    ceiling = compute_anthropic_request_cost_ceiling(
        config=config,
        system=system,
        prompt=prompt,
        wire_schema=compiled_schema.wire_schema,
        max_output_tokens=max_output_tokens,
        transport_mode=transport_mode,
    )
    _, expected_wire_call_sha256 = _build_anthropic_wire_call_surface(
        model=config.model,
        max_output_tokens=max_output_tokens,
        model_system=model_system,
        model_prompt=model_prompt,
        transport_mode=transport_mode,
        wire_schema=compiled_schema.wire_schema,
        effort=config.effort,
        service_tier=config.service_tier,
    )
    transport_policy_binding = hash_canonical(
        {
            "policy": (
                "inventory-structured-json-schema-packet-prompt-json-schema-v1"
            ),
            "schema_kind": schema_kind,
            "effect_kind": effect_kind,
            "transport_mode": transport_mode,
            "compiled_schema_sha256": compiled_schema.compiled_schema_sha256,
            "wire_schema_sha256": compiled_schema.wire_schema_sha256,
        }
    )
    payload: dict[str, Any] = {
        "operation": operation,
        "request_key": request_key,
        "model": config.model,
        "identity_sha256": frozen_identity.identity_sha256,
        "config_sha256": config.config_sha256,
        "compiled_schema": compiled_schema.model_dump(mode="json"),
        "compiled_schema_sha256": compiled_schema.compiled_schema_sha256,
        "full_acceptance_schema_sha256": (
            compiled_schema.full_acceptance_schema_sha256
        ),
        "schema_kind": schema_kind,
        "effect_kind": effect_kind,
        "transport_mode": transport_mode,
        "transport_policy_binding_sha256": transport_policy_binding,
        "prompt_json_system_envelope_sha256": (
            ANTHROPIC_PROMPT_JSON_SYSTEM_ENVELOPE_SHA256
        ),
        "system": system,
        "base_system_sha256": _sha256_utf8(system),
        "model_system": model_system,
        "model_system_sha256": _sha256_utf8(model_system),
        "prompt": prompt,
        "base_prompt_sha256": _sha256_utf8(prompt),
        "model_prompt": model_prompt,
        "model_prompt_sha256": _sha256_utf8(model_prompt),
        "wire_schema_delivery": (
            "structured_output_config_format"
            if transport_mode == "structured_json_schema"
            else "canonical_model_system"
        ),
        "structured_grammar_enforced_by_provider": (
            transport_mode == "structured_json_schema"
        ),
        "output_format_present_in_call": (
            transport_mode == "structured_json_schema"
        ),
        "expected_wire_call_sha256": expected_wire_call_sha256,
        "max_output_tokens": max_output_tokens,
        "cost_ceiling": ceiling.model_dump(mode="json"),
    }
    payload["request_sha256"] = hash_canonical(
        {
            "request_version": "anthropic-bounded-request-v2",
            **payload,
        }
    )
    return AnthropicBoundedRequestV1(**payload)


class AnthropicUsageV1(_FrozenContract):
    usage_version: Literal["anthropic-usage-v1"] = "anthropic-usage-v1"
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
    def reject_boolean_tokens(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("anthropic_usage_boolean_invalid")
        return value


class AnthropicCostV1(_FrozenContract):
    cost_version: Literal["anthropic-reported-cost-v1"] = (
        "anthropic-reported-cost-v1"
    )
    basis: Literal["reported_standard_usage", "unknown_request_ceiling"]
    input_rate_usd_per_million_tokens: PositiveRate
    output_rate_usd_per_million_tokens: PositiveRate
    estimated_cost_usd: Annotated[Decimal, Field(ge=0, allow_inf_nan=False)] | None
    request_cost_ceiling_usd: Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
    charged_cost_upper_bound_usd: (
        Annotated[Decimal, Field(gt=0, allow_inf_nan=False)] | None
    )

    @model_validator(mode="after")
    def validate_cost(self) -> AnthropicCostV1:
        if (
            self.input_rate_usd_per_million_tokens
            != ANTHROPIC_INPUT_RATE_USD_PER_MTOK
            or self.output_rate_usd_per_million_tokens
            != ANTHROPIC_OUTPUT_RATE_USD_PER_MTOK
        ):
            raise ValueError("anthropic_result_cost_rates_not_pinned")
        if self.basis == "reported_standard_usage":
            if (
                self.estimated_cost_usd is None
                or self.charged_cost_upper_bound_usd
                != self.request_cost_ceiling_usd
            ):
                raise ValueError("anthropic_reported_cost_shape_invalid")
        elif (
            self.estimated_cost_usd is not None
            or self.charged_cost_upper_bound_usd is not None
        ):
            raise ValueError("anthropic_unknown_cost_shape_invalid")
        return self


class AnthropicFailureV1(_FrozenContract):
    failure_version: Literal["anthropic-failure-v1"] = "anthropic-failure-v1"
    code: AnthropicFailureCode
    exception_type: Annotated[
        str, Field(min_length=1, max_length=_MAX_FAILURE_TYPE_CHARACTERS)
    ] | None = None
    http_status: Annotated[StrictInt, Field(ge=100, le=599)] | None = None
    provider_request_id: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    detail: Literal[
        "transport_attempt_failed",
        "response_id_missing",
        "response_model_did_not_match_frozen_request",
        "response_did_not_end_with_end_turn",
        "response_content_shape_invalid",
        "reported_usage_invalid_or_outside_request_bound",
        "response_text_was_not_strict_json",
        "response_json_failed_wire_schema",
        "response_json_failed_original_provider_schema",
    ]


class AnthropicBoundedResultV1(_FrozenContract):
    result_version: Literal["anthropic-bounded-result-v2"] = (
        "anthropic-bounded-result-v2"
    )
    provider: Literal["anthropic"] = "anthropic"
    request_sha256: Sha256
    identity_sha256: Sha256
    config_sha256: Sha256
    compiled_schema_sha256: Sha256
    original_schema_sha256: Sha256
    wire_schema_sha256: Sha256
    full_acceptance_schema_sha256: Sha256
    schema_kind: AnthropicSchemaKind
    effect_kind: AnthropicEffectKind | None
    transport_mode: AnthropicTransportMode
    structured_grammar_enforced_by_provider: bool
    output_format_present_in_call: bool
    model_system_sha256: Sha256
    model_prompt_sha256: Sha256
    wire_call_sha256: Sha256
    transport_attempt_count: Literal[1] = 1
    sdk_retry_count: Literal[0] = 0
    outcome: AnthropicOutcome
    response_id: Annotated[str, Field(min_length=1, max_length=256)] | None
    response_model: Annotated[str, Field(min_length=1, max_length=256)] | None
    stop_reason: Annotated[str, Field(min_length=1, max_length=128)] | None
    text: Annotated[str, Field(max_length=_MAX_TEXT_CHARACTERS)] | None
    response_text_sha256: Sha256 | None
    parsed_json: Any | None
    parsed_json_sha256: Sha256 | None
    usage: AnthropicUsageV1 | None
    cost: AnthropicCostV1
    failure: AnthropicFailureV1 | None
    result_sha256: Sha256

    @model_validator(mode="after")
    def validate_result(self) -> AnthropicBoundedResultV1:
        structured = self.transport_mode == "structured_json_schema"
        if (
            self.structured_grammar_enforced_by_provider != structured
            or self.output_format_present_in_call != structured
            or (self.schema_kind == "inventory") != structured
            or (self.effect_kind is None) != structured
        ):
            raise ValueError("anthropic_result_transport_mode_shape_invalid")
        if self.outcome == "completed":
            if self.failure is not None or self.parsed_json is None:
                raise ValueError("anthropic_completed_result_shape_invalid")
            if self.stop_reason != "end_turn" or self.response_model != ANTHROPIC_MODEL:
                raise ValueError("anthropic_completed_response_identity_invalid")
            if self.usage is None or self.text is None or self.response_id is None:
                raise ValueError("anthropic_completed_response_fields_missing")
        elif self.failure is None or self.failure.code != self.outcome:
            raise ValueError("anthropic_failed_result_shape_invalid")
        expected_text_sha256 = _sha256_utf8(self.text) if self.text is not None else None
        expected_json_sha256 = (
            hash_canonical(self.parsed_json) if self.parsed_json is not None else None
        )
        if (
            self.response_text_sha256 != expected_text_sha256
            or self.parsed_json_sha256 != expected_json_sha256
        ):
            raise ValueError("anthropic_result_raw_canonical_hash_mismatch")
        if self.usage is None:
            if self.cost.basis != "unknown_request_ceiling":
                raise ValueError("anthropic_result_unknown_usage_cost_mismatch")
        else:
            expected_cost = (
                Decimal(self.usage.input_tokens)
                * ANTHROPIC_INPUT_RATE_USD_PER_MTOK
                + Decimal(self.usage.output_tokens)
                * ANTHROPIC_OUTPUT_RATE_USD_PER_MTOK
            ) / Decimal(1_000_000)
            if (
                self.cost.basis != "reported_standard_usage"
                or self.cost.estimated_cost_usd != expected_cost
                or expected_cost > self.cost.request_cost_ceiling_usd
            ):
                raise ValueError("anthropic_result_reported_cost_mismatch")
        expected = hash_canonical(
            self.model_dump(mode="json", exclude={"result_sha256"})
        )
        if self.result_sha256 != expected:
            raise ValueError("anthropic_result_hash_mismatch")
        return self


def _strict_json_loads(value: str) -> Any:
    def reject_constant(raw: str) -> Any:
        raise ValueError(f"nonfinite:{raw}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise ValueError("duplicate_json_key")
            output[key] = item
        return output

    parsed = json.loads(
        value,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )

    pending: list[tuple[Any, int]] = [(parsed, 0)]
    observed_nodes = 0
    while pending:
        item, depth = pending.pop()
        observed_nodes += 1
        if observed_nodes > _MAX_JSON_NODES:
            raise ValueError("json_node_limit_exceeded")
        if depth > _MAX_JSON_DEPTH:
            raise ValueError("json_depth_limit_exceeded")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite_json_number")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
    try:
        canonical_json_bytes(parsed)
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise ValueError("json_value_not_canonical_utf8") from exc
    return parsed


def _safe_exception_type(exc: Exception) -> str:
    raw = type(exc).__name__[:_MAX_FAILURE_TYPE_CHARACTERS]
    lowered = raw.casefold()
    return (
        raw
        if _SAFE_EXCEPTION_TYPE.fullmatch(raw)
        and not any(marker in lowered for marker in ("sk-ant", "bearer", "api_key"))
        else "ProviderException"
    )


def _safe_status(exc: Exception) -> int | None:
    try:
        value = getattr(exc, "status_code", None)
    except Exception:
        return None
    return (
        value
        if type(value) is int
        and 100 <= value <= 599
        else None
    )


def _safe_request_id(exc: Exception) -> str | None:
    try:
        value = getattr(exc, "request_id", None)
    except Exception:
        return None
    if (
        type(value) is not str
        or not re.fullmatch(r"req_[A-Za-z0-9_-]{1,196}", value)
        or any(
            marker in value.casefold()
            for marker in ("sk-ant", "bearer", "api_key", "authorization")
        )
    ):
        return None
    return value


def _provider_getattr(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name, default)
    except Exception:
        return default


def _safe_response_id(value: Any) -> str | None:
    if type(value) is not str:
        return None
    return value if re.fullmatch(r"msg_[A-Za-z0-9_-]{1,252}", value) else None


def _safe_response_model(value: Any) -> str | None:
    return ANTHROPIC_MODEL if type(value) is str and value == ANTHROPIC_MODEL else None


def _safe_stop_reason(value: Any) -> str | None:
    return (
        value
        if type(value) is str
        and value
        in {
            "end_turn",
            "max_tokens",
            "pause_turn",
            "refusal",
            "stop_sequence",
            "tool_use",
        }
        else None
    )


def _parse_usage(response: Any, request: AnthropicBoundedRequestV1) -> AnthropicUsageV1:
    try:
        raw = response.usage
        input_tokens = raw.input_tokens
        output_tokens = raw.output_tokens
        cache_creation_input_tokens = getattr(
            raw, "cache_creation_input_tokens", 0
        )
        cache_read_input_tokens = getattr(raw, "cache_read_input_tokens", 0)
    except Exception as exc:
        raise ValueError("anthropic_usage_fields_unreadable") from exc
    usage = AnthropicUsageV1(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens or 0,
        cache_read_input_tokens=cache_read_input_tokens or 0,
    )
    # This boundary does not request prompt caching and pins only standard rates.
    if usage.cache_creation_input_tokens or usage.cache_read_input_tokens:
        raise ValueError("anthropic_unrequested_cache_usage")
    if (
        usage.input_tokens
        > request.cost_ceiling.conservative_input_token_ceiling
        or usage.output_tokens > request.max_output_tokens
    ):
        raise ValueError("anthropic_reported_usage_exceeds_ceiling")
    return usage


def _reported_cost(
    usage: AnthropicUsageV1,
    *,
    request: AnthropicBoundedRequestV1,
    config: AnthropicBoundedConfigV1,
) -> AnthropicCostV1:
    estimated = (
        Decimal(usage.input_tokens) * config.input_rate_usd_per_million_tokens
        + Decimal(usage.output_tokens) * config.output_rate_usd_per_million_tokens
    ) / Decimal(1_000_000)
    return AnthropicCostV1(
        basis="reported_standard_usage",
        input_rate_usd_per_million_tokens=config.input_rate_usd_per_million_tokens,
        output_rate_usd_per_million_tokens=config.output_rate_usd_per_million_tokens,
        estimated_cost_usd=estimated,
        request_cost_ceiling_usd=request.cost_ceiling.request_cost_ceiling_usd,
        charged_cost_upper_bound_usd=request.cost_ceiling.request_cost_ceiling_usd,
    )


def _unknown_cost(
    request: AnthropicBoundedRequestV1,
    config: AnthropicBoundedConfigV1,
) -> AnthropicCostV1:
    return AnthropicCostV1(
        basis="unknown_request_ceiling",
        input_rate_usd_per_million_tokens=config.input_rate_usd_per_million_tokens,
        output_rate_usd_per_million_tokens=config.output_rate_usd_per_million_tokens,
        estimated_cost_usd=None,
        request_cost_ceiling_usd=request.cost_ceiling.request_cost_ceiling_usd,
        charged_cost_upper_bound_usd=None,
    )


def freeze_anthropic_wire_call_surface(
    *, request: AnthropicBoundedRequestV1, config: AnthropicBoundedConfigV1
) -> tuple[dict[str, Any], str]:
    """Replay the exact credential-free SDK kwargs and their immutable hash."""

    canonical_request = AnthropicBoundedRequestV1.model_validate(
        request.model_dump(mode="json")
    )
    if (
        canonical_request.config_sha256 != config.config_sha256
        or canonical_request.model != config.model
    ):
        raise AnthropicBoundedGenerationError(
            "anthropic_wire_call_request_config_mismatch"
        )
    kwargs, wire_call_sha256 = _build_anthropic_wire_call_surface(
        model=config.model,
        max_output_tokens=canonical_request.max_output_tokens,
        model_system=canonical_request.model_system,
        model_prompt=canonical_request.model_prompt,
        transport_mode=canonical_request.transport_mode,
        wire_schema=canonical_request.compiled_schema.wire_schema,
        effort=config.effort,
        service_tier=config.service_tier,
    )
    if wire_call_sha256 != canonical_request.expected_wire_call_sha256:
        raise AnthropicBoundedGenerationError(
            "anthropic_wire_call_request_hash_mismatch"
        )
    return kwargs, wire_call_sha256


class AnthropicBoundedClient:
    """Exactly-one-call SDK client; failed outcomes are returned, never retried."""

    def __init__(
        self,
        config: AnthropicBoundedConfigV1,
    ) -> None:
        self.config = config
        self.identity = freeze_anthropic_provider_identity(config)
        self._client: Any | None = None

    def _client_or_create(self) -> Any:
        if self._client is None:
            if os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get(
                "ANTHROPIC_CUSTOM_HEADERS"
            ):
                raise AnthropicBoundedGenerationError(
                    "anthropic_transport_environment_override_forbidden"
                )
            anthropic = _require_exact_sdk()
            http_client = anthropic.DefaultHttpxClient(
                timeout=self.config.timeout_seconds,
                trust_env=False,
                follow_redirects=False,
            )
            self._client = anthropic.Anthropic(
                base_url=ANTHROPIC_API_BASE_URL,
                default_headers={"anthropic-version": ANTHROPIC_API_VERSION},
                http_client=http_client,
                max_retries=0,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    def _freeze_result(
        self,
        *,
        request: AnthropicBoundedRequestV1,
        wire_call_sha256: str,
        outcome: AnthropicOutcome,
        response_id: str | None,
        response_model: str | None,
        stop_reason: str | None,
        text: str | None,
        parsed_json: Any | None,
        usage: AnthropicUsageV1 | None,
        cost: AnthropicCostV1,
        failure: AnthropicFailureV1 | None,
    ) -> AnthropicBoundedResultV1:
        payload: dict[str, Any] = {
            "request_sha256": request.request_sha256,
            "identity_sha256": request.identity_sha256,
            "config_sha256": request.config_sha256,
            "compiled_schema_sha256": request.compiled_schema_sha256,
            "original_schema_sha256": (
                request.compiled_schema.original_schema_sha256
            ),
            "wire_schema_sha256": request.compiled_schema.wire_schema_sha256,
            "full_acceptance_schema_sha256": (
                request.full_acceptance_schema_sha256
            ),
            "schema_kind": request.schema_kind,
            "effect_kind": request.effect_kind,
            "transport_mode": request.transport_mode,
            "structured_grammar_enforced_by_provider": (
                request.transport_mode == "structured_json_schema"
            ),
            "output_format_present_in_call": (
                request.transport_mode == "structured_json_schema"
            ),
            "model_system_sha256": request.model_system_sha256,
            "model_prompt_sha256": request.model_prompt_sha256,
            "wire_call_sha256": wire_call_sha256,
            "outcome": outcome,
            "response_id": response_id,
            "response_model": response_model,
            "stop_reason": stop_reason,
            "text": text,
            "response_text_sha256": _sha256_utf8(text) if text is not None else None,
            "parsed_json": parsed_json,
            "parsed_json_sha256": (
                hash_canonical(parsed_json) if parsed_json is not None else None
            ),
            "usage": usage.model_dump(mode="json") if usage is not None else None,
            "cost": cost.model_dump(mode="json"),
            "failure": failure.model_dump(mode="json") if failure is not None else None,
        }
        payload["result_sha256"] = hash_canonical(
            {
                "result_version": "anthropic-bounded-result-v2",
                "provider": "anthropic",
                "transport_attempt_count": 1,
                "sdk_retry_count": 0,
                **payload,
            }
        )
        return AnthropicBoundedResultV1(**payload)

    def generate(self, request: AnthropicBoundedRequestV1) -> AnthropicBoundedResultV1:
        """Perform one Messages request and return one immutable terminal result."""

        # Pydantic's frozen setting prevents field assignment but nested JSON objects
        # are ordinary containers.  Revalidation catches any post-freeze nested edit
        # before a transport call can be authorized.
        try:
            request = AnthropicBoundedRequestV1.model_validate(
                request.model_dump(mode="json")
            )
        except ValueError as exc:
            raise AnthropicBoundedGenerationError(
                "anthropic_client_request_revalidation_failed"
            ) from exc
        if request.config_sha256 != self.config.config_sha256:
            raise AnthropicBoundedGenerationError(
                "anthropic_client_request_config_mismatch"
            )
        if request.identity_sha256 != self.identity.identity_sha256:
            raise AnthropicBoundedGenerationError(
                "anthropic_client_request_identity_mismatch"
            )
        if request.model != self.config.model:
            raise AnthropicBoundedGenerationError(
                "anthropic_client_request_model_mismatch"
            )

        kwargs, wire_call_sha256 = freeze_anthropic_wire_call_surface(
            request=request,
            config=self.config,
        )
        try:
            response = self._client_or_create().messages.create(**kwargs)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
                raise
            failure = AnthropicFailureV1(
                code="transport_failed",
                exception_type=_safe_exception_type(exc),
                http_status=_safe_status(exc),
                provider_request_id=_safe_request_id(exc),
                detail="transport_attempt_failed",
            )
            return self._freeze_result(
                request=request,
                wire_call_sha256=wire_call_sha256,
                outcome="transport_failed",
                response_id=None,
                response_model=None,
                stop_reason=None,
                text=None,
                parsed_json=None,
                usage=None,
                cost=_unknown_cost(request, self.config),
                failure=failure,
            )

        response_id = _safe_response_id(_provider_getattr(response, "id"))
        response_model = _safe_response_model(_provider_getattr(response, "model"))
        stop_reason = _safe_stop_reason(_provider_getattr(response, "stop_reason"))

        try:
            usage = _parse_usage(response, request)
            cost = _reported_cost(usage, request=request, config=self.config)
        except Exception:
            usage = None
            cost = _unknown_cost(request, self.config)
            failure = AnthropicFailureV1(
                code="response_usage_invalid",
                detail="reported_usage_invalid_or_outside_request_bound",
            )
            return self._freeze_result(
                request=request,
                wire_call_sha256=wire_call_sha256,
                outcome="response_usage_invalid",
                response_id=response_id,
                response_model=response_model,
                stop_reason=stop_reason,
                text=None,
                parsed_json=None,
                usage=None,
                cost=cost,
                failure=failure,
            )

        if not response_id or not response_id.strip():
            failure = AnthropicFailureV1(
                code="response_identity_invalid", detail="response_id_missing"
            )
            return self._freeze_result(
                request=request,
                wire_call_sha256=wire_call_sha256,
                outcome="response_identity_invalid",
                response_id=None,
                response_model=response_model,
                stop_reason=stop_reason,
                text=None,
                parsed_json=None,
                usage=usage,
                cost=cost,
                failure=failure,
            )
        if response_model != request.model:
            failure = AnthropicFailureV1(
                code="response_model_mismatch",
                detail="response_model_did_not_match_frozen_request",
            )
            return self._freeze_result(
                request=request,
                wire_call_sha256=wire_call_sha256,
                outcome="response_model_mismatch",
                response_id=response_id,
                response_model=response_model,
                stop_reason=stop_reason,
                text=None,
                parsed_json=None,
                usage=usage,
                cost=cost,
                failure=failure,
            )
        if stop_reason != "end_turn":
            failure = AnthropicFailureV1(
                code="response_stop_reason_invalid",
                detail="response_did_not_end_with_end_turn",
            )
            return self._freeze_result(
                request=request,
                wire_call_sha256=wire_call_sha256,
                outcome="response_stop_reason_invalid",
                response_id=response_id,
                response_model=response_model,
                stop_reason=stop_reason,
                text=None,
                parsed_json=None,
                usage=usage,
                cost=cost,
                failure=failure,
            )

        content_access_valid = True
        try:
            blocks = list(response.content or ())
            block_types = []
            text_blocks = []
            for block in blocks:
                block_type = block.type
                if type(block_type) is not str:
                    raise ValueError("anthropic_response_block_type_invalid")
                block_types.append(block_type)
                if block_type == "text":
                    block_text = block.text
                    if type(block_text) is not str:
                        raise ValueError("anthropic_response_block_text_invalid")
                    text_blocks.append(block_text)
        except Exception:
            content_access_valid = False
            blocks = []
            block_types = []
            text_blocks = []
        allowed_types = {"text", "thinking", "redacted_thinking"}
        if (
            not content_access_valid
            or len(text_blocks) != 1
            or type(text_blocks[0]) is not str
            or not text_blocks[0]
            or len(text_blocks[0]) > _MAX_TEXT_CHARACTERS
            or any(block_type not in allowed_types for block_type in block_types)
        ):
            failure = AnthropicFailureV1(
                code="response_content_invalid",
                detail="response_content_shape_invalid",
            )
            return self._freeze_result(
                request=request,
                wire_call_sha256=wire_call_sha256,
                outcome="response_content_invalid",
                response_id=response_id,
                response_model=response_model,
                stop_reason=stop_reason,
                text=None,
                parsed_json=None,
                usage=usage,
                cost=cost,
                failure=failure,
            )
        text = text_blocks[0]
        try:
            parsed = _strict_json_loads(text)
        except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
            failure = AnthropicFailureV1(
                code="response_json_invalid",
                detail="response_text_was_not_strict_json",
            )
            return self._freeze_result(
                request=request,
                wire_call_sha256=wire_call_sha256,
                outcome="response_json_invalid",
                response_id=response_id,
                response_model=response_model,
                stop_reason=stop_reason,
                # Invalid provider text is intentionally not re-archived: it may
                # contain non-UTF-8 lone surrogates and has no scientific authority.
                text=None,
                parsed_json=None,
                usage=usage,
                cost=cost,
                failure=failure,
            )
        try:
            validator_for(request.compiled_schema.wire_schema)(
                request.compiled_schema.wire_schema
            ).validate(parsed)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
                raise
            failure = AnthropicFailureV1(
                code="response_wire_schema_invalid",
                detail="response_json_failed_wire_schema",
            )
            return self._freeze_result(
                request=request,
                wire_call_sha256=wire_call_sha256,
                outcome="response_wire_schema_invalid",
                response_id=response_id,
                response_model=response_model,
                stop_reason=stop_reason,
                text=text,
                parsed_json=parsed,
                usage=usage,
                cost=cost,
                failure=failure,
            )
        try:
            validator_for(request.compiled_schema.original_schema)(
                request.compiled_schema.original_schema
            ).validate(parsed)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):  # pragma: no cover
                raise
            failure = AnthropicFailureV1(
                code="response_schema_invalid",
                detail="response_json_failed_original_provider_schema",
            )
            return self._freeze_result(
                request=request,
                wire_call_sha256=wire_call_sha256,
                outcome="response_schema_invalid",
                response_id=response_id,
                response_model=response_model,
                stop_reason=stop_reason,
                text=text,
                parsed_json=parsed,
                usage=usage,
                cost=cost,
                failure=failure,
            )
        return self._freeze_result(
            request=request,
            wire_call_sha256=wire_call_sha256,
            outcome="completed",
            response_id=response_id,
            response_model=response_model,
            stop_reason=stop_reason,
            text=text,
            parsed_json=parsed,
            usage=usage,
            cost=cost,
            failure=None,
        )


__all__ = [
    "ANTHROPIC_API_BASE_URL",
    "ANTHROPIC_API_VERSION",
    "ANTHROPIC_BOUNDED_CONTRACT_VERSION",
    "ANTHROPIC_FIXED_FRAMING_TOKENS",
    "ANTHROPIC_INPUT_RATE_USD_PER_MTOK",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_MODEL_SOURCE_URL",
    "ANTHROPIC_OUTPUT_RATE_USD_PER_MTOK",
    "ANTHROPIC_PINNED_PRICING_TABLE_V1",
    "ANTHROPIC_PRICING_SOURCE_URL",
    "ANTHROPIC_PRICING_TABLE_SHA256",
    "ANTHROPIC_PROMPT_JSON_SYSTEM_ENVELOPE_SHA256",
    "ANTHROPIC_PROMPT_JSON_SYSTEM_ENVELOPE_VERSION",
    "ANTHROPIC_SCHEMA_COMPILER_VERSION",
    "ANTHROPIC_SCHEMA_MAX_INLINED_DEPTH",
    "ANTHROPIC_SCHEMA_MAX_INLINED_NODES",
    "ANTHROPIC_SCHEMA_MAX_INLINED_UTF8_BYTES",
    "ANTHROPIC_SCHEMA_MAX_NULLABLE_OPTIONAL_PROMOTIONS",
    "ANTHROPIC_SCHEMA_MAX_OPTIONAL_PARAMETERS",
    "ANTHROPIC_SCHEMA_MAX_REFERENCE_EXPANSIONS",
    "ANTHROPIC_SCHEMA_MAX_UNION_PARAMETERS",
    "ANTHROPIC_SCHEMA_NULLABLE_OPTIONAL_REQUIRED_TARGET",
    "ANTHROPIC_SDK_VERSION",
    "AnthropicBoundedClient",
    "AnthropicBoundedConfigV1",
    "AnthropicBoundedGenerationError",
    "AnthropicBoundedRequestV1",
    "AnthropicBoundedResultV1",
    "AnthropicCompiledSchemaV1",
    "AnthropicCostV1",
    "AnthropicEffectKind",
    "AnthropicFailureV1",
    "AnthropicProviderIdentityV1",
    "AnthropicRequestCostCeilingV1",
    "AnthropicSchemaKind",
    "AnthropicTransportMode",
    "AnthropicUsageV1",
    "adapt_anthropic_nullable_optional_properties",
    "annotate_anthropic_literal_types",
    "compile_anthropic_bounded_schema",
    "compute_anthropic_request_cost_ceiling",
    "count_anthropic_optional_parameters",
    "count_anthropic_union_parameters",
    "freeze_anthropic_bounded_request",
    "freeze_anthropic_provider_identity",
    "freeze_anthropic_wire_call_surface",
    "inline_anthropic_local_references",
    "materialize_anthropic_nullable_optionals",
    "project_anthropic_preflight_fixture",
    "promote_anthropic_nullable_optional_properties",
    "render_anthropic_prompt_json_model_system",
]
