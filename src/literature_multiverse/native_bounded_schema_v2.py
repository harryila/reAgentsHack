"""Isolated v2 generation-schema hardening for bounded native packets.

This module is deliberately not imported by :mod:`native_bounded_generation`.
The frozen Antiox v1 run must remain replayable against the exact schema that was
used for its calls.  V2 transforms a v1 schema for a future generation attempt; it
does not repair, coerce, or promote a v1 model output.

The official Pydantic/native extraction contracts remain the acceptance authority.
JSON Schema can reject many malformed generations earlier, but it cannot prove
cross-field numeric equality, source entailment, or scientific correctness.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, TypedDict

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.native_bounded_generation import (
    PACKET_VERSION,
    NativeBoundedGenerationError,
    NativeCandidateDescriptor,
    NativeCandidateInventory,
    NativeCandidatePacketOutcome,
    assert_bounded_generation_schema,
    inventory_generation_schema,
    packet_generation_schema,
    validate_inventory_for_row,
    validate_packet_for_candidate,
)
from literature_multiverse.schemas import assert_closed_object_schema

PACKET_GENERATION_SCHEMA_V2 = "native-candidate-packet-generation-schema-v2"
INVENTORY_GENERATION_SCHEMA_V2 = "native-candidate-inventory-generation-schema-v2"
PACKET_PROVIDER_SCHEMA_V2 = "native-candidate-packet-provider-schema-v2"
INVENTORY_PROVIDER_SCHEMA_V2 = "native-candidate-inventory-provider-schema-v2"
SCHEMA_BUNDLE_V2 = "native-bounded-generation-schema-bundle-v2"

# These patterns mirror the closed v1 lexical grammars.  They are deliberately
# attached to JSON *strings*: generation never converts a model-authored number to
# a scientific value or manufactures a source token.
_ABSOLUTE_END_PATTERN = r"(?![\s\S])$"
_DECIMAL_LEXEME_BODY = (
    r"-?(?:(?:0|[1-9][0-9]{0,12})(?:\.[0-9]{1,12})?|\.[0-9]{1,12})"
    r"(?:[eE][+-]?(?:0|[1-9][0-9]{0,2}))?"
)
_NONNEGATIVE_DECIMAL_LEXEME_BODY = (
    r"(?:(?:0|[1-9][0-9]{0,12})(?:\.[0-9]{1,12})?|\.[0-9]{1,12})"
    r"(?:[eE][+-]?(?:0|[1-9][0-9]{0,2}))?"
)
_ZERO_DECIMAL_LEXEME_BODY = (
    r"-?(?:0(?:\.0{1,12})?|\.0{1,12})"
    r"(?:[eE][+-]?(?:0|[1-9][0-9]{0,2}))?"
)
DECIMAL_LEXEME_PATTERN = rf"^{_DECIMAL_LEXEME_BODY}{_ABSOLUTE_END_PATTERN}"
NONNEGATIVE_DECIMAL_LEXEME_PATTERN = rf"^{_NONNEGATIVE_DECIMAL_LEXEME_BODY}{_ABSOLUTE_END_PATTERN}"
ZERO_DECIMAL_LEXEME_PATTERN = rf"^{_ZERO_DECIMAL_LEXEME_BODY}{_ABSOLUTE_END_PATTERN}"
PERCENT_DECIMAL_TOKEN_PATTERN = rf"^{_DECIMAL_LEXEME_BODY}%{_ABSOLUTE_END_PATTERN}"
PROVIDER_DECIMAL_OR_PERCENT_TOKEN_PATTERN = rf"^{_DECIMAL_LEXEME_BODY}%?{_ABSOLUTE_END_PATTERN}"
UNSIGNED_COUNT_PATTERN = rf"^(?:0|[1-9][0-9]{{0,8}}|1000000000){_ABSOLUTE_END_PATTERN}"
POSITIVE_COUNT_PATTERN = rf"^(?:[1-9][0-9]{{0,8}}|1000000000){_ABSOLUTE_END_PATTERN}"
AT_LEAST_TWO_COUNT_PATTERN = rf"^(?:[2-9]|[1-9][0-9]{{1,8}}|1000000000){_ABSOLUTE_END_PATTERN}"
QUOTE_OFFSET_PATTERN = rf"^(?:0|[1-9][0-9]{{0,2}}|1[0-7][0-9]{{2}}|1800){_ABSOLUTE_END_PATTERN}"

_PERCENT_NORMALIZABLE_PATHS = {
    "effect.ci_level",
    "effect.reported_p_value",
}
_HEADER_NUMERIC_PATHS = {
    "cohort.total_sample_size",
    "treatment_arm.sample_size",
    "comparator_arm.sample_size",
}
_TIMEPOINT_NUMERIC_PATHS = {
    "finding.timepoint.value",
    "finding.timepoint.lower",
    "finding.timepoint.upper",
}
_COMMON_EFFECT_NUMERIC_PATHS = {
    "effect.reported_p_value",
    "effect.equivalence_margin",
}
_REQUIRED_EFFECT_NUMERIC_PATHS: dict[str, tuple[str, ...]] = {
    "direct_standard_error": ("effect.estimate", "effect.standard_error"),
    "direct_variance": ("effect.estimate", "effect.variance"),
    "direct_confidence_interval": (
        "effect.estimate",
        "effect.ci_lower",
        "effect.ci_upper",
        "effect.ci_level",
    ),
    "continuous_group_statistics": (
        "effect.treatment_mean",
        "effect.treatment_sd",
        "effect.treatment_n",
        "effect.control_mean",
        "effect.control_sd",
        "effect.control_n",
    ),
    "binary_group_statistics": (
        "effect.treatment_events",
        "effect.treatment_total",
        "effect.control_events",
        "effect.control_total",
    ),
}
_EFFECT_DEFINITION_NAMES = {
    "direct_standard_error": "DirectStandardErrorEffect",
    "direct_variance": "DirectVarianceEffect",
    "direct_confidence_interval": "DirectConfidenceIntervalEffect",
    "continuous_group_statistics": "ContinuousGroupEffect",
    "binary_group_statistics": "BinaryGroupEffect",
}
_DIRECT_FORMATS = {
    "mean_difference",
    "cohens_d",
    "hedges_g",
    "odds_ratio",
    "log_odds_ratio",
    "risk_ratio",
    "log_risk_ratio",
}
_CONTINUOUS_FORMATS = {"mean_difference", "cohens_d", "hedges_g"}
_BINARY_FORMATS = {
    "odds_ratio",
    "log_odds_ratio",
    "risk_ratio",
    "log_risk_ratio",
}
_POSITIVE_RATIO_FORMATS = {"odds_ratio", "risk_ratio"}
_EFFECT_KINDS = tuple(sorted(_REQUIRED_EFFECT_NUMERIC_PATHS))

# Standard Draft 2020-12 has no portable cross-instance comparison primitive.
# These checks therefore stay in deterministic acceptance postvalidation even after
# v2 rejects lexical/shape failures during generation.
UNAVOIDABLE_POSTVALIDATION_INVARIANTS: tuple[str, ...] = (
    "canonical_lexicographic_order_for_model_authored_lists",
    "inventory_descriptor_signatures_are_unique_across_candidate_indices",
    "decimal_value_domains_and_magnitude_bounds",
    "timepoint_range_lower_strictly_less_than_upper",
    "treatment_and_comparator_arm_keys_are_distinct",
    "confidence_interval_order_and_estimate_membership",
    "binary_event_counts_do_not_exceed_totals",
    "numeric_support_set_equals_all_emitted_numeric_leaves",
    "numeric_support_paths_are_canonically_ordered",
    "each_numeric_support_start_is_less_than_end_and_source_spans_are_unique",
    "numeric_support_token_is_exact_verbatim_quote_slice",
    "numeric_support_token_boundaries_exclude_larger_expressions",
    "numeric_support_decimal_equals_emitted_decimal",
    "source_token_semantically_supports_the_named_scientific_field",
)

# Provider grammar compilation is not an acceptance boundary.  In particular,
# llama.cpp-family JSON-Schema converters have versions that skip these keywords.
# The raw Draft 2020-12 validation pass below remains mandatory even when an Ollama
# request accepts the schema.
PROVIDER_GRAMMAR_SCOPE_V2: dict[str, Any] = {
    "provider_grammar_enforcement_assumed": False,
    "eight_call_preflight_proves": "whole_schema_request_compatibility_only",
    "eight_call_preflight_does_not_prove": (
        "keyword_level_grammar_enforcement_or_extraction_yield"
    ),
    "known_potentially_skipped_keywords": [
        "contains",
        "maxContains",
        "minContains",
        "not",
        "pattern",
        "prefixItems",
        "uniqueItems",
    ],
    "pattern_feature_support_requires_target_version_probe": True,
    "provider_pattern_policy": (
        "omit_all_pattern_keywords_for_ollama_0_15_1_grammar_compatibility"
    ),
    "provider_pattern_omission_reason": (
        "llama_cpp_schema_to_grammar_lookaround_and_regex_shorthand_crash_risk"
    ),
    "provider_pattern_lexical_enforcement": (
        "mandatory_full_acceptance_raw_draft202012_validation_only"
    ),
    "raw_draft202012_validation_before_pydantic_is_required": True,
    "provider_schema_scientific_authority": "none",
    "provider_schema_is_intentionally_wider_than_full_acceptance": True,
    "inventory_provider_state_coherence": (
        "three_closed_oneOf_branches_preserve_empty_found_and_overflow_states"
    ),
    "provider_only_simplifications": [
        "arm_role_side_assignment",
        "direct_ratio_positive_domain",
        "inventory_count_specific_index_specialization_and_descriptor_signature_uniqueness",
        "numeric_support_normalization_path_coupling",
        "string_lexeme_patterns",
        "timepoint_kind_dependent_shape",
    ],
}


SchemaCoverage = Literal["enforced", "partial", "postvalidation_only"]


class ValidatorCoverage(TypedDict):
    validator: str
    coverage: SchemaCoverage
    schema_enforcement: str
    remaining_postvalidation: str


# An explicit audit of every custom validator reachable from a completed packet.
# Tests pin this roster so a new validator cannot be silently assumed to be covered.
PACKET_VALIDATOR_COVERAGE_V2: tuple[ValidatorCoverage, ...] = (
    {
        "validator": "BoundedStudyHeader.validate_registration_ids",
        "coverage": "partial",
        "schema_enforcement": "uniqueItems rejects duplicate complete strings",
        "remaining_postvalidation": "lexicographic canonical order",
    },
    {
        "validator": "BoundedCohortHeader.validate_identity_lists",
        "coverage": "partial",
        "schema_enforcement": "uniqueItems rejects duplicate complete strings",
        "remaining_postvalidation": "lexicographic canonical order",
    },
    {
        "validator": "BoundedCohortHeader.validate_total_sample_size",
        "coverage": "enforced",
        "schema_enforcement": "closed unsigned 1..1000000000 string pattern",
        "remaining_postvalidation": "Pydantic remains acceptance authority",
    },
    {
        "validator": "BoundedArm.validate_sample_size",
        "coverage": "enforced",
        "schema_enforcement": "closed unsigned 1..1000000000 string pattern",
        "remaining_postvalidation": "Pydantic remains acceptance authority",
    },
    {
        "validator": "BoundedTimepoint.validate_finite",
        "coverage": "partial",
        "schema_enforcement": "nonnegative decimal-string lexical grammar",
        "remaining_postvalidation": "Decimal finiteness and 1e6 magnitude bound",
    },
    {
        "validator": "BoundedTimepoint.validate_official_shape",
        "coverage": "partial",
        "schema_enforcement": "four closed kind-discriminated shape branches",
        "remaining_postvalidation": "range lower < upper",
    },
    {
        "validator": "BoundedEvidence.validate_line_ids",
        "coverage": "enforced",
        "schema_enforcement": "candidate-bound exact array const preserves membership and order",
        "remaining_postvalidation": "Pydantic remains acceptance authority",
    },
    {
        "validator": "BoundedEffectCommon.validate_reported_p_value",
        "coverage": "partial",
        "schema_enforcement": "decimal-string lexical grammar",
        "remaining_postvalidation": "numeric interval [0,1]",
    },
    {
        "validator": "BoundedEffectCommon.validate_equivalence_margin",
        "coverage": "partial",
        "schema_enforcement": "decimal-string grammar plus lexical-zero rejection",
        "remaining_postvalidation": "Decimal positivity and magnitude",
    },
    {
        "validator": "BoundedEffectCommon.validate_moderators",
        "coverage": "partial",
        "schema_enforcement": "allowed names and at most one object per name",
        "remaining_postvalidation": "lexicographic canonical order",
    },
    {
        "validator": "DirectStandardErrorEffect.validate_values_and_format",
        "coverage": "partial",
        "schema_enforcement": (
            "compatible formats, decimal estimate, positive standard error, positive ratio estimate"
        ),
        "remaining_postvalidation": "Decimal magnitude and domain authority",
    },
    {
        "validator": "DirectVarianceEffect.validate_values_and_format",
        "coverage": "partial",
        "schema_enforcement": (
            "compatible formats, decimal estimate, positive variance, positive ratio estimate"
        ),
        "remaining_postvalidation": "Decimal magnitude and domain authority",
    },
    {
        "validator": "DirectConfidenceIntervalEffect.validate_interval",
        "coverage": "partial",
        "schema_enforcement": "compatible formats and typed decimal fields; positive ratio fields",
        "remaining_postvalidation": "CI level range, lower < upper, and estimate inside interval",
    },
    {
        "validator": "ContinuousGroupEffect.validate_statistics_and_format",
        "coverage": "partial",
        "schema_enforcement": "compatible formats, typed means, positive SDs, n >= 2",
        "remaining_postvalidation": "Decimal magnitude authority",
    },
    {
        "validator": "BinaryGroupEffect.validate_counts",
        "coverage": "partial",
        "schema_enforcement": "compatible formats and bounded unsigned count strings",
        "remaining_postvalidation": "events <= totals",
    },
    {
        "validator": "BoundedNumericSupport.validate_offsets_and_token",
        "coverage": "partial",
        "schema_enforcement": "identity/percent token branches and bounded unsigned offset strings",
        "remaining_postvalidation": "quote_start < quote_end",
    },
    {
        "validator": "NativeCandidatePacket.validate_packet_references",
        "coverage": "partial",
        "schema_enforcement": "role-specific arms and effect-specific required support paths",
        "remaining_postvalidation": (
            "distinct arm keys and all cross-field/source-support invariants"
        ),
    },
    {
        "validator": "_validate_numeric_support",
        "coverage": "partial",
        "schema_enforcement": (
            "effect-specific paths, one receipt per path, required effect receipts"
        ),
        "remaining_postvalidation": (
            "complete optional field set, canonical order, unique spans, quote "
            "slicing, token boundaries, and exact Decimal equality"
        ),
    },
)


INVENTORY_VALIDATOR_COVERAGE_V2: tuple[ValidatorCoverage, ...] = (
    {
        "validator": "NativeCandidateDescriptor.validate_line_ids",
        "coverage": "partial",
        "schema_enforcement": ("exposed-line membership, bounded length, and duplicate rejection"),
        "remaining_postvalidation": "lexicographic canonical order",
    },
    {
        "validator": "NativeCandidateInventory.validate_inventory",
        "coverage": "partial",
        "schema_enforcement": (
            "raw Draft 2020-12 contiguous indices and status/candidate/uncertainty shape"
        ),
        "remaining_postvalidation": (
            "duplicate scientific descriptor signatures across different indices"
        ),
    },
)


def _definition(schema: Mapping[str, Any], name: str) -> dict[str, Any]:
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict) or not isinstance(definitions.get(name), dict):
        raise NativeBoundedGenerationError(f"native_schema_v2_definition_missing:{name}")
    return definitions[name]


def _completed_definition_name(schema: Mapping[str, Any]) -> str:
    definitions = schema.get("$defs")
    if not isinstance(definitions, Mapping):
        raise NativeBoundedGenerationError("native_schema_v2_definitions_missing")
    names = [str(name) for name in definitions if str(name).startswith("NativeCandidatePacket_")]
    if len(names) != 1:
        raise NativeBoundedGenerationError("native_schema_v2_completed_definition_ambiguous")
    return names[0]


def _assert_v1_schema_candidate_binding(
    schema: Mapping[str, Any], *, candidate: NativeCandidateDescriptor
) -> None:
    """Fail closed unless the frozen v1 schema encodes this exact candidate."""

    completed = _definition(schema, _completed_definition_name(schema))
    unable = _definition(schema, "NativeCandidateUnableToComplete")
    for definition in (completed, unable):
        index_enum = _property(definition, "candidate_index").get("enum")
        if index_enum != [candidate.candidate_index]:
            raise NativeBoundedGenerationError("native_schema_v2_candidate_index_binding_mismatch")

    outcome_enum = _property(_definition(schema, "BoundedFindingHeader"), "outcome_name").get(
        "enum"
    )
    if outcome_enum != [candidate.outcome_name]:
        raise NativeBoundedGenerationError("native_schema_v2_candidate_outcome_binding_mismatch")

    line_schema = _property(_definition(schema, "BoundedEvidence"), "line_ids")
    items = line_schema.get("items")
    line_enum = items.get("enum") if isinstance(items, Mapping) else None
    if line_enum != candidate.line_ids:
        raise NativeBoundedGenerationError("native_schema_v2_candidate_line_binding_mismatch")

    expected_effect_definition = _EFFECT_DEFINITION_NAMES[candidate.effect_kind]
    effect_reference = _property(completed, "effect").get("$ref")
    if effect_reference != f"#/$defs/{expected_effect_definition}":
        raise NativeBoundedGenerationError(
            "native_schema_v2_candidate_effect_kind_binding_mismatch"
        )
    effect = _definition(schema, expected_effect_definition)
    effect_kind = _property(effect, "effect_kind")
    encoded_kind = effect_kind.get("const")
    if encoded_kind is None:
        encoded_kind = effect_kind.get("default")
    if encoded_kind != candidate.effect_kind:
        raise NativeBoundedGenerationError(
            "native_schema_v2_candidate_effect_kind_binding_mismatch"
        )


def _require_properties(definition: dict[str, Any], *names: str) -> None:
    required = definition.setdefault("required", [])
    if not isinstance(required, list):
        raise NativeBoundedGenerationError("native_schema_v2_required_invalid")
    definition["required"] = sorted(set(required).union(names))


def _property(definition: Mapping[str, Any], name: str) -> dict[str, Any]:
    properties = definition.get("properties")
    if not isinstance(properties, Mapping) or not isinstance(properties.get(name), dict):
        raise NativeBoundedGenerationError(f"native_schema_v2_property_missing:{name}")
    return properties[name]


def _set_const(definition: dict[str, Any], name: str, value: Any) -> None:
    prop = _property(definition, name)
    prop.pop("enum", None)
    prop["const"] = value


def _walk_string_nodes(value: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if isinstance(value, dict):
        raw_type = value.get("type")
        if raw_type == "string" or (isinstance(raw_type, list) and "string" in raw_type):
            output.append(value)
        for item in value.values():
            output.extend(_walk_string_nodes(item))
    elif isinstance(value, list):
        for item in value:
            output.extend(_walk_string_nodes(item))
    return output


def _set_string_pattern(
    definition: dict[str, Any],
    name: str,
    pattern: str,
    *,
    forbid_zero: bool = False,
) -> None:
    nodes = _walk_string_nodes(_property(definition, name))
    if not nodes:
        raise NativeBoundedGenerationError(f"native_schema_v2_string_property_missing:{name}")
    for node in nodes:
        node["pattern"] = pattern
        if forbid_zero:
            node["not"] = {"pattern": ZERO_DECIMAL_LEXEME_PATTERN}


def _non_null_string_schema(
    property_schema: Mapping[str, Any],
    *,
    pattern: str,
    forbid_zero: bool = False,
) -> dict[str, Any]:
    nodes = _walk_string_nodes(property_schema)
    if len(nodes) != 1:
        raise NativeBoundedGenerationError("native_schema_v2_non_null_string_schema_ambiguous")
    output = deepcopy(nodes[0])
    output["pattern"] = pattern
    if forbid_zero:
        output["not"] = {"pattern": ZERO_DECIMAL_LEXEME_PATTERN}
    return output


def _set_enum(definition: dict[str, Any], name: str, values: Sequence[str]) -> None:
    prop = _property(definition, name)
    prop.clear()
    prop.update({"type": "string", "enum": sorted(set(values))})


def _add_unique_items(definition: dict[str, Any], name: str) -> None:
    prop = _property(definition, name)
    prop["uniqueItems"] = True


def _closed_timepoint_branches(base: Mapping[str, Any]) -> dict[str, Any]:
    properties = base.get("properties")
    if not isinstance(properties, Mapping):
        raise NativeBoundedGenerationError("native_schema_v2_timepoint_properties_missing")

    def branch(kind: str) -> dict[str, Any]:
        output = deepcopy(dict(base))
        output.pop("title", None)
        output["properties"]["kind"] = {"type": "string", "const": kind}
        return output

    exact = branch("exact")
    _require_properties(exact, "kind", "value", "unit")
    exact["properties"]["value"] = _non_null_string_schema(
        properties["value"], pattern=NONNEGATIVE_DECIMAL_LEXEME_PATTERN
    )
    exact["properties"]["lower"] = {"type": "null"}
    exact["properties"]["upper"] = {"type": "null"}
    exact["properties"]["unit"] = {"$ref": "#/$defs/TimeUnit"}

    range_branch = branch("range")
    _require_properties(range_branch, "kind", "lower", "upper", "unit")
    range_branch["properties"]["value"] = {"type": "null"}
    range_branch["properties"]["lower"] = _non_null_string_schema(
        properties["lower"], pattern=NONNEGATIVE_DECIMAL_LEXEME_PATTERN
    )
    range_branch["properties"]["upper"] = _non_null_string_schema(
        properties["upper"], pattern=NONNEGATIVE_DECIMAL_LEXEME_PATTERN
    )
    range_branch["properties"]["unit"] = {"$ref": "#/$defs/TimeUnit"}

    reported_text = branch("reported_text")
    _require_properties(reported_text, "kind", "raw_label")
    for name in ("value", "lower", "upper", "unit"):
        reported_text["properties"][name] = {"type": "null"}
    raw_label_nodes = _walk_string_nodes(reported_text["properties"]["raw_label"])
    if not raw_label_nodes:
        raise NativeBoundedGenerationError("native_schema_v2_timepoint_raw_label_missing")
    reported_text["properties"]["raw_label"] = deepcopy(raw_label_nodes[0])

    not_reported = branch("not_reported")
    _require_properties(not_reported, "kind")
    for name in ("value", "lower", "upper", "unit", "anchor", "raw_label"):
        not_reported["properties"][name] = {"type": "null"}

    return {"oneOf": [exact, range_branch, reported_text, not_reported]}


def _harden_header_definitions(schema: dict[str, Any]) -> None:
    study = _definition(schema, "BoundedStudyHeader")
    _add_unique_items(study, "registration_ids")

    cohort = _definition(schema, "BoundedCohortHeader")
    for name in ("source_labels", "registry_ids", "dataset_ids"):
        _add_unique_items(cohort, name)
    _set_string_pattern(cohort, "total_sample_size", POSITIVE_COUNT_PATTERN)

    arm = _definition(schema, "BoundedArm")
    _set_string_pattern(arm, "sample_size", POSITIVE_COUNT_PATTERN)


def _harden_arm_roles(schema: dict[str, Any], completed: dict[str, Any]) -> None:
    definitions = schema["$defs"]
    base = deepcopy(_definition(schema, "BoundedArm"))
    treatment = deepcopy(base)
    comparator = deepcopy(base)
    _set_enum(treatment, "role", ("intervention", "exposure"))
    _set_enum(comparator, "role", ("comparator", "control"))
    definitions["BoundedTreatmentArmV2"] = treatment
    definitions["BoundedComparatorArmV2"] = comparator
    completed["properties"]["treatment_arm"] = {"$ref": "#/$defs/BoundedTreatmentArmV2"}
    completed["properties"]["comparator_arm"] = {"$ref": "#/$defs/BoundedComparatorArmV2"}


def _harden_effect_definition(schema: dict[str, Any], *, effect_kind: str) -> None:
    name = _EFFECT_DEFINITION_NAMES[effect_kind]
    effect = _definition(schema, name)
    _require_properties(effect, "effect_kind")
    _set_const(effect, "effect_kind", effect_kind)
    _set_string_pattern(effect, "reported_p_value", NONNEGATIVE_DECIMAL_LEXEME_PATTERN)
    _set_string_pattern(
        effect,
        "equivalence_margin",
        NONNEGATIVE_DECIMAL_LEXEME_PATTERN,
        forbid_zero=True,
    )
    moderators = _property(effect, "moderators")
    moderators["uniqueItems"] = True

    if effect_kind == "direct_standard_error":
        _set_string_pattern(effect, "estimate", DECIMAL_LEXEME_PATTERN)
        _set_string_pattern(
            effect,
            "standard_error",
            NONNEGATIVE_DECIMAL_LEXEME_PATTERN,
            forbid_zero=True,
        )
        allowed_formats = _DIRECT_FORMATS
    elif effect_kind == "direct_variance":
        _set_string_pattern(effect, "estimate", DECIMAL_LEXEME_PATTERN)
        _set_string_pattern(
            effect,
            "variance",
            NONNEGATIVE_DECIMAL_LEXEME_PATTERN,
            forbid_zero=True,
        )
        allowed_formats = _DIRECT_FORMATS
    elif effect_kind == "direct_confidence_interval":
        for field_name in ("estimate", "ci_lower", "ci_upper", "ci_level"):
            _set_string_pattern(effect, field_name, DECIMAL_LEXEME_PATTERN)
        _set_string_pattern(
            effect,
            "ci_level",
            NONNEGATIVE_DECIMAL_LEXEME_PATTERN,
            forbid_zero=True,
        )
        allowed_formats = _DIRECT_FORMATS
    elif effect_kind == "continuous_group_statistics":
        for field_name in ("treatment_mean", "control_mean"):
            _set_string_pattern(effect, field_name, DECIMAL_LEXEME_PATTERN)
        for field_name in ("treatment_sd", "control_sd"):
            _set_string_pattern(
                effect,
                field_name,
                NONNEGATIVE_DECIMAL_LEXEME_PATTERN,
                forbid_zero=True,
            )
        for field_name in ("treatment_n", "control_n"):
            _set_string_pattern(effect, field_name, AT_LEAST_TWO_COUNT_PATTERN)
        allowed_formats = _CONTINUOUS_FORMATS
    elif effect_kind == "binary_group_statistics":
        for field_name in ("treatment_events", "control_events"):
            _set_string_pattern(effect, field_name, UNSIGNED_COUNT_PATTERN)
        for field_name in ("treatment_total", "control_total"):
            _set_string_pattern(effect, field_name, POSITIVE_COUNT_PATTERN)
        allowed_formats = _BINARY_FORMATS
    else:  # pragma: no cover - candidate Pydantic contract closes this enum
        raise NativeBoundedGenerationError(f"native_schema_v2_effect_kind_unknown:{effect_kind}")
    _set_enum(effect, "effect_format", sorted(allowed_formats))

    if effect_kind.startswith("direct_"):
        non_ratio = deepcopy(effect)
        ratio = deepcopy(effect)
        _set_enum(non_ratio, "effect_format", sorted(allowed_formats - _POSITIVE_RATIO_FORMATS))
        _set_enum(ratio, "effect_format", sorted(_POSITIVE_RATIO_FORMATS))
        ratio_fields = ["estimate"]
        if effect_kind == "direct_confidence_interval":
            ratio_fields.extend(("ci_lower", "ci_upper"))
        for field_name in ratio_fields:
            _set_string_pattern(
                ratio, field_name, NONNEGATIVE_DECIMAL_LEXEME_PATTERN, forbid_zero=True
            )
        schema["$defs"][name] = {"oneOf": [non_ratio, ratio]}


def _numeric_support_branches(
    base: Mapping[str, Any], *, allowed_paths: Sequence[str]
) -> dict[str, Any]:
    identity = deepcopy(dict(base))
    _require_properties(identity, "normalization")
    _set_const(identity, "normalization", "identity")
    _set_enum(identity, "field_path", allowed_paths)
    _set_string_pattern(identity, "verbatim_token", DECIMAL_LEXEME_PATTERN)
    _set_string_pattern(identity, "quote_start", QUOTE_OFFSET_PATTERN)
    _set_string_pattern(identity, "quote_end", QUOTE_OFFSET_PATTERN)

    percent_paths = sorted(set(allowed_paths).intersection(_PERCENT_NORMALIZABLE_PATHS))
    branches = [identity]
    if percent_paths:
        percent = deepcopy(dict(base))
        _require_properties(percent, "normalization")
        _set_const(percent, "normalization", "percent_to_proportion")
        _set_enum(percent, "field_path", percent_paths)
        _set_string_pattern(percent, "verbatim_token", PERCENT_DECIMAL_TOKEN_PATTERN)
        _set_string_pattern(percent, "quote_start", QUOTE_OFFSET_PATTERN)
        _set_string_pattern(percent, "quote_end", QUOTE_OFFSET_PATTERN)
        branches.append(percent)
    return {"oneOf": branches}


def _specialized_support_definition(
    generic: Mapping[str, Any], *, field_path: str
) -> dict[str, Any]:
    branches = generic.get("oneOf")
    if not isinstance(branches, list):
        raise NativeBoundedGenerationError("native_schema_v2_numeric_support_branches_missing")
    selected: list[dict[str, Any]] = []
    for branch in branches:
        if not isinstance(branch, dict):
            continue
        normalization = _property(branch, "normalization").get("const")
        if (
            normalization == "percent_to_proportion"
            and field_path not in _PERCENT_NORMALIZABLE_PATHS
        ):
            continue
        clone = deepcopy(branch)
        _set_const(clone, "field_path", field_path)
        selected.append(clone)
    if not selected:
        raise NativeBoundedGenerationError(
            f"native_schema_v2_numeric_support_path_missing:{field_path}"
        )
    return {"oneOf": selected}


def _support_definition_name(field_path: str) -> str:
    return "BoundedNumericSupportV2__" + re.sub(r"[^A-Za-z0-9]+", "_", field_path)


def _harden_numeric_support(
    schema: dict[str, Any], *, effect_kind: str, completed: dict[str, Any]
) -> None:
    required_paths = _REQUIRED_EFFECT_NUMERIC_PATHS[effect_kind]
    allowed_paths = sorted(
        set(required_paths)
        | _HEADER_NUMERIC_PATHS
        | _TIMEPOINT_NUMERIC_PATHS
        | _COMMON_EFFECT_NUMERIC_PATHS
    )
    base = deepcopy(_definition(schema, "BoundedNumericSupport"))
    generic = _numeric_support_branches(base, allowed_paths=allowed_paths)
    schema["$defs"]["BoundedNumericSupport"] = generic

    support_array = _property(completed, "numeric_support")
    support_array["items"] = {"$ref": "#/$defs/BoundedNumericSupport"}
    support_array["allOf"] = []
    for field_path in allowed_paths:
        definition_name = _support_definition_name(field_path)
        schema["$defs"][definition_name] = _specialized_support_definition(
            generic, field_path=field_path
        )
        support_array["allOf"].append(
            {
                "contains": {"$ref": f"#/$defs/{definition_name}"},
                "minContains": 1 if field_path in required_paths else 0,
                "maxContains": 1,
            }
        )


def _harden_moderator_uniqueness(schema: dict[str, Any], *, effect_kind: str) -> None:
    effect_name = _EFFECT_DEFINITION_NAMES[effect_kind]
    effect_wrapper = _definition(schema, effect_name)
    effect_branches = effect_wrapper.get("oneOf", [effect_wrapper])
    moderator = deepcopy(_definition(schema, "BoundedModerator"))
    names = _property(moderator, "name").get("enum", [])
    if not isinstance(names, list):
        return
    for name in names:
        if not isinstance(name, str):
            continue
        definition_name = "BoundedModeratorV2__" + re.sub(r"[^A-Za-z0-9]+", "_", name)
        specialized = deepcopy(moderator)
        _set_const(specialized, "name", name)
        schema["$defs"][definition_name] = specialized
        for effect_branch in effect_branches:
            if not isinstance(effect_branch, dict):
                continue
            moderators = _property(effect_branch, "moderators")
            moderators.setdefault("allOf", []).append(
                {
                    "contains": {"$ref": f"#/$defs/{definition_name}"},
                    "minContains": 0,
                    "maxContains": 1,
                }
            )


def upgrade_packet_schema_v1_to_v2(
    schema_v1: Mapping[str, Any], *, candidate: NativeCandidateDescriptor
) -> dict[str, Any]:
    """Strengthen one exact v1 packet schema without changing its source context."""

    schema = deepcopy(dict(schema_v1))
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise NativeBoundedGenerationError("native_schema_v2_input_schema_invalid") from exc

    _assert_v1_schema_candidate_binding(schema, candidate=candidate)

    completed_name = _completed_definition_name(schema)
    completed = _definition(schema, completed_name)
    unable = _definition(schema, "NativeCandidateUnableToComplete")
    for definition, status in ((completed, "completed"), (unable, "unable_to_complete")):
        _require_properties(definition, "packet_version", "packet_status")
        _set_const(definition, "packet_version", PACKET_VERSION)
        _set_const(definition, "packet_status", status)

    _harden_header_definitions(schema)
    schema["$defs"]["BoundedTimepoint"] = _closed_timepoint_branches(
        _definition(schema, "BoundedTimepoint")
    )

    evidence = _definition(schema, "BoundedEvidence")
    line_ids = _property(evidence, "line_ids")
    line_ids["const"] = list(candidate.line_ids)
    line_ids["minItems"] = len(candidate.line_ids)
    line_ids["maxItems"] = len(candidate.line_ids)
    line_ids["uniqueItems"] = True

    _harden_effect_definition(schema, effect_kind=candidate.effect_kind)
    _harden_arm_roles(schema, completed)
    _harden_numeric_support(schema, effect_kind=candidate.effect_kind, completed=completed)
    _harden_moderator_uniqueness(schema, effect_kind=candidate.effect_kind)

    context_sha256 = hash_canonical(
        {
            "v1_schema": schema_v1,
            "candidate": candidate.model_dump(mode="json"),
        }
    )
    schema["$id"] = (
        "urn:literature-multiverse:native-candidate-packet:generation-schema-v2:"
        f"{candidate.effect_kind}:{context_sha256[:24]}"
    )
    schema["x-literature-multiverse-generation-schema-version"] = PACKET_GENERATION_SCHEMA_V2
    assert_closed_object_schema(schema)
    assert_bounded_generation_schema(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:  # pragma: no cover - unit tests exercise all kinds
        raise NativeBoundedGenerationError("native_schema_v2_output_schema_invalid") from exc
    return schema


def packet_generation_schema_v2(
    *,
    candidate: NativeCandidateDescriptor,
    exposed_line_ids: Sequence[str],
    source_locator: str,
    allowed_outcomes: Sequence[str],
    allowed_moderators: Sequence[str] = (),
    allowed_sections: Sequence[str] = ("FigureTable", "Methods", "Results"),
    outcome_positive_directions: Mapping[str, str],
) -> dict[str, Any]:
    """Build v2 from the exact context-bound v1 generation schema."""

    schema_v1 = packet_generation_schema(
        candidate=candidate,
        exposed_line_ids=exposed_line_ids,
        source_locator=source_locator,
        allowed_outcomes=allowed_outcomes,
        allowed_moderators=allowed_moderators,
        allowed_sections=allowed_sections,
        outcome_positive_directions=outcome_positive_directions,
    )
    return upgrade_packet_schema_v1_to_v2(schema_v1, candidate=candidate)


def _harden_inventory_state_and_indices(schema: dict[str, Any]) -> None:
    definitions = schema["$defs"]
    descriptor = deepcopy(_definition(schema, "NativeCandidateDescriptor"))
    for index in range(1, 10):
        specialized = deepcopy(descriptor)
        index_schema = _property(specialized, "candidate_index")
        index_schema.clear()
        index_schema.update({"type": "integer", "enum": [index]})
        definitions[f"NativeCandidateDescriptorV2Index{index}"] = specialized

    candidates = _property(schema, "candidates")
    count_branches: list[dict[str, Any]] = []
    for count in range(10):
        count_branch: dict[str, Any] = {
            "type": "array",
            "minItems": count,
            "maxItems": count,
            "items": False,
        }
        if count:
            count_branch["prefixItems"] = [
                {"$ref": f"#/$defs/NativeCandidateDescriptorV2Index{index}"}
                for index in range(1, count + 1)
            ]
        count_branches.append(count_branch)
    candidates.clear()
    candidates.update({"type": "array", "maxItems": 9, "oneOf": count_branches})

    root_object = {
        key: deepcopy(schema[key])
        for key in ("type", "title", "additionalProperties", "properties", "required")
        if key in schema
    }

    def state_branch(*, status: str, has_more: bool, minimum: int, maximum: int) -> dict[str, Any]:
        branch = deepcopy(root_object)
        _set_const(branch, "inventory_status", status)
        _set_const(branch, "has_more_or_uncertain", has_more)
        branch_candidates = _property(branch, "candidates")
        branch_candidates["minItems"] = minimum
        branch_candidates["maxItems"] = maximum
        return branch

    schema.pop("type", None)
    schema.pop("title", None)
    schema.pop("additionalProperties", None)
    schema.pop("properties", None)
    schema.pop("required", None)
    schema["oneOf"] = [
        state_branch(status="candidates_found", has_more=False, minimum=1, maximum=8),
        state_branch(status="no_candidate_found", has_more=False, minimum=0, maximum=0),
        state_branch(status="overflow_or_uncertain", has_more=True, minimum=0, maximum=9),
    ]


def inventory_generation_schema_v2(
    *, exposed_line_ids: Sequence[str], allowed_outcomes: Sequence[str]
) -> dict[str, Any]:
    """Add safe lexical/uniqueness constraints to the value-free inventory schema."""

    schema = inventory_generation_schema(
        exposed_line_ids=exposed_line_ids, allowed_outcomes=allowed_outcomes
    )
    _require_properties(schema, "inventory_version")
    _set_const(schema, "inventory_version", "native-candidate-inventory-v1")
    descriptor = _definition(schema, "NativeCandidateDescriptor")
    line_ids = _property(descriptor, "line_ids")
    line_ids["uniqueItems"] = True
    _harden_inventory_state_and_indices(schema)
    context_sha256 = hash_canonical(
        {
            "line_ids": sorted(set(exposed_line_ids)),
            "outcomes": sorted(set(allowed_outcomes)),
        }
    )
    schema["$id"] = (
        "urn:literature-multiverse:native-candidate-inventory:generation-schema-v2:"
        f"{context_sha256[:24]}"
    )
    schema["x-literature-multiverse-generation-schema-version"] = INVENTORY_GENERATION_SCHEMA_V2
    assert_closed_object_schema(schema)
    assert_bounded_generation_schema(schema)
    Draft202012Validator.check_schema(schema)
    return schema


_PROVIDER_UNSUPPORTED_SCHEMA_KEYWORDS = {
    "contains",
    "maxContains",
    "minContains",
    "not",
    "pattern",
    "prefixItems",
    "uniqueItems",
}
_PROVIDER_OMIT_METADATA_KEYWORDS = {"default", "description", "examples", "title"}


def _compact_provider_keywords(value: Any, *, parent_keyword: str | None = None) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            is_schema_keyword = parent_keyword not in {
                "$defs",
                "properties",
                "patternProperties",
            }
            if key in _PROVIDER_UNSUPPORTED_SCHEMA_KEYWORDS and is_schema_keyword:
                continue
            # ``description``, ``title``, and similar words are valid authored
            # property/$defs names as well as annotation keywords.  Only remove
            # them while traversing an actual schema node; deleting them from a
            # ``properties`` mapping silently changes the accepted object shape.
            if key in _PROVIDER_OMIT_METADATA_KEYWORDS and is_schema_keyword:
                continue
            compacted = _compact_provider_keywords(item, parent_keyword=key)
            if key in {"allOf", "anyOf", "oneOf"} and isinstance(compacted, list):
                compacted = [entry for entry in compacted if entry not in ({}, True)]
                if not compacted:
                    continue
            output[key] = compacted
        return output
    if isinstance(value, list):
        return [_compact_provider_keywords(item, parent_keyword=parent_keyword) for item in value]
    return deepcopy(value)


def _remove_redundant_provider_constraints(
    value: Any, *, parent_keyword: str | None = None
) -> None:
    """Shrink exact-value nodes without changing their Draft acceptance set."""

    if isinstance(value, dict):
        is_schema_node = parent_keyword not in {
            "$defs",
            "patternProperties",
            "properties",
        }
        if is_schema_node and "const" in value:
            for key in (
                "enum",
                "items",
                "maxItems",
                "maxLength",
                "maximum",
                "minItems",
                "minLength",
                "minimum",
                "pattern",
                "type",
            ):
                value.pop(key, None)
        elif is_schema_node and isinstance(value.get("enum"), list) and value["enum"]:
            enum_values = value["enum"]
            enum_types = {type(item) for item in enum_values}
            declared_type = value.get("type")
            if len(enum_types) == 1 and declared_type in {
                "boolean",
                "integer",
                "null",
                "number",
                "string",
            }:
                value.pop("type", None)
            if enum_types == {str}:
                lengths = [len(item) for item in enum_values]
                if min(lengths) >= value.get("minLength", 0):
                    value.pop("minLength", None)
                maximum = value.get("maxLength")
                if isinstance(maximum, int) and max(lengths) <= maximum:
                    value.pop("maxLength", None)
            if enum_types.issubset({int}) and not enum_types.issubset({bool}):
                minimum = value.get("minimum")
                maximum = value.get("maximum")
                if isinstance(minimum, (int, float)) and min(enum_values) >= minimum:
                    value.pop("minimum", None)
                if isinstance(maximum, (int, float)) and max(enum_values) <= maximum:
                    value.pop("maximum", None)
        if is_schema_node and value.get("type") == "array" and value.get("minItems") == 0:
            value.pop("minItems", None)
        for key, item in value.items():
            _remove_redundant_provider_constraints(item, parent_keyword=key)
    elif isinstance(value, list):
        for item in value:
            _remove_redundant_provider_constraints(item, parent_keyword=parent_keyword)


def _referenced_definition_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, Mapping):
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            names.add(reference.removeprefix("#/$defs/"))
        for key, item in value.items():
            if key != "$defs":
                names.update(_referenced_definition_names(item))
    elif isinstance(value, list):
        for item in value:
            names.update(_referenced_definition_names(item))
    return names


def _prune_unreachable_definitions(schema: dict[str, Any]) -> None:
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        return
    pending = list(_referenced_definition_names(schema))
    reachable: set[str] = set()
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        definition = definitions.get(name)
        if not isinstance(definition, dict):
            raise NativeBoundedGenerationError(
                f"native_schema_v2_provider_definition_missing:{name}"
            )
        reachable.add(name)
        pending.extend(_referenced_definition_names(definition) - reachable)
    schema["$defs"] = {name: definitions[name] for name in sorted(reachable)}


def _collapse_direct_effect_for_provider(schema: dict[str, Any], *, effect_kind: str) -> None:
    if not effect_kind.startswith("direct_"):
        return
    name = _EFFECT_DEFINITION_NAMES[effect_kind]
    wrapper = _definition(schema, name)
    branches = wrapper.get("oneOf")
    if not isinstance(branches, list) or len(branches) != 2:
        raise NativeBoundedGenerationError(
            "native_schema_v2_provider_direct_effect_branches_invalid"
        )
    effect = deepcopy(branches[0])
    _set_enum(effect, "effect_format", sorted(_DIRECT_FORMATS))
    _set_string_pattern(effect, "estimate", DECIMAL_LEXEME_PATTERN)
    if effect_kind == "direct_confidence_interval":
        for field_name in ("ci_lower", "ci_upper"):
            _set_string_pattern(effect, field_name, DECIMAL_LEXEME_PATTERN)
    schema["$defs"][name] = effect


def _collapse_arm_roles_for_provider(schema: dict[str, Any]) -> None:
    """Share one arm definition where side-specific roles need postvalidation."""

    completed = _definition(schema, _completed_definition_name(schema))
    for field_name in ("treatment_arm", "comparator_arm"):
        completed["properties"][field_name] = {"$ref": "#/$defs/BoundedArm"}


def _collapse_timepoint_for_provider(schema: dict[str, Any]) -> None:
    """Keep lexical assistance without duplicating four full acceptance branches."""

    wrapper = _definition(schema, "BoundedTimepoint")
    branches = wrapper.get("oneOf")
    if not isinstance(branches, list) or len(branches) != 4:
        raise NativeBoundedGenerationError("native_schema_v2_provider_timepoint_branches_invalid")
    exact = deepcopy(branches[0])
    exact["properties"]["kind"] = {
        "type": "string",
        "enum": ["exact", "not_reported", "range", "reported_text"],
    }
    numeric_string = deepcopy(_property(branches[0], "value"))
    for field_name in ("value", "lower", "upper"):
        exact["properties"][field_name] = {"anyOf": [deepcopy(numeric_string), {"type": "null"}]}
    exact["properties"]["unit"] = {"anyOf": [{"$ref": "#/$defs/TimeUnit"}, {"type": "null"}]}
    exact["required"] = ["kind"]
    schema["$defs"]["BoundedTimepoint"] = exact


def _collapse_numeric_support_for_provider(schema: dict[str, Any]) -> None:
    """Use one lexical support shape; full Draft validation restores coupling."""

    wrapper = _definition(schema, "BoundedNumericSupport")
    branches = wrapper.get("oneOf")
    if not isinstance(branches, list) or not branches:
        raise NativeBoundedGenerationError(
            "native_schema_v2_provider_numeric_support_branches_invalid"
        )
    support = deepcopy(branches[0])
    allowed_paths: set[str] = set()
    normalizations: set[str] = set()
    for branch in branches:
        if not isinstance(branch, Mapping):
            continue
        field_paths = _property(branch, "field_path").get("enum", [])
        if isinstance(field_paths, list):
            allowed_paths.update(str(item) for item in field_paths)
        normalization = _property(branch, "normalization").get("const")
        if isinstance(normalization, str):
            normalizations.add(normalization)
    _set_enum(support, "field_path", sorted(allowed_paths))
    _set_enum(support, "normalization", sorted(normalizations))
    _set_string_pattern(
        support,
        "verbatim_token",
        PROVIDER_DECIMAL_OR_PERCENT_TOKEN_PATTERN,
    )
    schema["$defs"]["BoundedNumericSupport"] = support


def _packet_provider_schema_from_full(
    full: Mapping[str, Any], *, candidate: NativeCandidateDescriptor
) -> dict[str, Any]:
    compactable = deepcopy(dict(full))
    _collapse_direct_effect_for_provider(compactable, effect_kind=candidate.effect_kind)
    _collapse_arm_roles_for_provider(compactable)
    _collapse_timepoint_for_provider(compactable)
    _collapse_numeric_support_for_provider(compactable)
    return _finalize_provider_schema(
        compactable,
        full_acceptance_schema=full,
        provider_version=PACKET_PROVIDER_SCHEMA_V2,
        provider_id_prefix=(
            "urn:literature-multiverse:native-candidate-packet:provider-schema-v2:"
        ),
    )


def _inventory_provider_schema_from_full(full: Mapping[str, Any]) -> dict[str, Any]:
    compactable = deepcopy(dict(full))
    state_branches = compactable.get("oneOf")
    if not isinstance(state_branches, list) or len(state_branches) != 3:
        raise NativeBoundedGenerationError(
            "native_schema_v2_provider_inventory_state_branches_invalid"
        )
    inventory = deepcopy(state_branches[0])
    _set_enum(
        inventory,
        "inventory_status",
        ("candidates_found", "no_candidate_found", "overflow_or_uncertain"),
    )
    has_more = _property(inventory, "has_more_or_uncertain")
    has_more.clear()
    has_more["type"] = "boolean"
    candidates = _property(inventory, "candidates")
    candidates.clear()
    candidates.update(
        {
            "type": "array",
            "minItems": 0,
            "maxItems": 9,
            "items": {"$ref": "#/$defs/NativeCandidateDescriptor"},
        }
    )

    # Keep the three inventory states coherent in the provider grammar without
    # reintroducing prefixItems/uniqueItems or any regex feature.  Each branch is
    # a complete closed object because a property-only conditional branch would
    # either be open or reject the other authored properties under
    # additionalProperties:false.  The full schema remains the authority for
    # contiguous indices and descriptor-signature uniqueness.
    provider_states: list[dict[str, Any]] = []
    for status, has_more_or_uncertain, minimum, maximum in (
        ("candidates_found", False, 1, 8),
        ("no_candidate_found", False, 0, 0),
        ("overflow_or_uncertain", True, 0, 9),
    ):
        branch = deepcopy(inventory)
        _set_const(branch, "inventory_status", status)
        _set_const(branch, "has_more_or_uncertain", has_more_or_uncertain)
        branch_candidates = _property(branch, "candidates")
        branch_candidates["minItems"] = minimum
        branch_candidates["maxItems"] = maximum
        provider_states.append(branch)
    for key in ("oneOf", "type", "title", "additionalProperties", "properties", "required"):
        compactable.pop(key, None)
    compactable["oneOf"] = provider_states
    return _finalize_provider_schema(
        compactable,
        full_acceptance_schema=full,
        provider_version=INVENTORY_PROVIDER_SCHEMA_V2,
        provider_id_prefix=(
            "urn:literature-multiverse:native-candidate-inventory:provider-schema-v2:"
        ),
    )


def _finalize_provider_schema(
    provider_source_schema: Mapping[str, Any],
    *,
    full_acceptance_schema: Mapping[str, Any],
    provider_version: str,
    provider_id_prefix: str,
) -> dict[str, Any]:
    acceptance_sha256 = hash_canonical(full_acceptance_schema)
    provider = _compact_provider_keywords(provider_source_schema)
    provider["$id"] = f"{provider_id_prefix}{acceptance_sha256[:24]}"
    provider["x-literature-multiverse-generation-schema-version"] = provider_version
    provider["x-literature-multiverse-full-acceptance-schema-sha256"] = acceptance_sha256
    _remove_redundant_provider_constraints(provider)
    _prune_unreachable_definitions(provider)
    assert_closed_object_schema(provider)
    assert_bounded_generation_schema(provider)
    Draft202012Validator.check_schema(provider)
    return provider


def packet_provider_schema_v2(
    *,
    candidate: NativeCandidateDescriptor,
    exposed_line_ids: Sequence[str],
    source_locator: str,
    allowed_outcomes: Sequence[str],
    allowed_moderators: Sequence[str] = (),
    allowed_sections: Sequence[str] = ("FigureTable", "Methods", "Results"),
    outcome_positive_directions: Mapping[str, str],
) -> dict[str, Any]:
    """Return the compact, zero-authority schema sent to the model provider."""

    full = packet_generation_schema_v2(
        candidate=candidate,
        exposed_line_ids=exposed_line_ids,
        source_locator=source_locator,
        allowed_outcomes=allowed_outcomes,
        allowed_moderators=allowed_moderators,
        allowed_sections=allowed_sections,
        outcome_positive_directions=outcome_positive_directions,
    )
    return _packet_provider_schema_from_full(full, candidate=candidate)


def inventory_provider_schema_v2(
    *, exposed_line_ids: Sequence[str], allowed_outcomes: Sequence[str]
) -> dict[str, Any]:
    """Return the compact, zero-authority inventory provider schema."""

    full = inventory_generation_schema_v2(
        exposed_line_ids=exposed_line_ids,
        allowed_outcomes=allowed_outcomes,
    )
    return _inventory_provider_schema_from_full(full)


def _schema_bundle_v2(
    *,
    kind: Literal["inventory", "packet"],
    full_acceptance_schema: Mapping[str, Any],
    provider_schema: Mapping[str, Any],
    context_binding: Mapping[str, Any],
) -> dict[str, Any]:
    full = deepcopy(dict(full_acceptance_schema))
    provider = deepcopy(dict(provider_schema))
    full_sha256 = hash_canonical(full)
    provider_sha256 = hash_canonical(provider)
    if provider.get("x-literature-multiverse-full-acceptance-schema-sha256") != full_sha256:
        raise NativeBoundedGenerationError(
            "native_schema_v2_provider_full_acceptance_binding_mismatch"
        )
    payload = {
        "schema_bundle_version": SCHEMA_BUNDLE_V2,
        "kind": kind,
        "full_acceptance_schema_version": full.get(
            "x-literature-multiverse-generation-schema-version"
        ),
        "full_acceptance_schema_sha256": full_sha256,
        "provider_schema_version": provider.get(
            "x-literature-multiverse-generation-schema-version"
        ),
        "provider_schema_sha256": provider_sha256,
        "context_binding": deepcopy(dict(context_binding)),
        "context_binding_sha256": hash_canonical(context_binding),
        "schema_sent_to_provider": "provider_schema",
        "raw_response_validation_schema": "full_acceptance_schema",
        "provider_schema_scientific_authority": "none",
        "full_acceptance_schema": full,
        "provider_schema": provider,
    }
    return {**payload, "schema_bundle_sha256": hash_canonical(payload)}


def schema_bundle_receipt_binding_v2(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Replay an exact bundle and return its dual-hash intent/receipt binding."""

    expected_bundle_keys = {
        "context_binding",
        "context_binding_sha256",
        "full_acceptance_schema",
        "full_acceptance_schema_sha256",
        "full_acceptance_schema_version",
        "kind",
        "provider_schema",
        "provider_schema_scientific_authority",
        "provider_schema_sha256",
        "provider_schema_version",
        "raw_response_validation_schema",
        "schema_bundle_sha256",
        "schema_bundle_version",
        "schema_sent_to_provider",
    }
    if set(bundle) != expected_bundle_keys:
        raise NativeBoundedGenerationError("native_schema_v2_bundle_keys_mismatch")
    payload = {
        key: deepcopy(value) for key, value in bundle.items() if key != "schema_bundle_sha256"
    }
    if payload.get("schema_bundle_version") != SCHEMA_BUNDLE_V2:
        raise NativeBoundedGenerationError("native_schema_v2_bundle_version_mismatch")
    if bundle.get("schema_bundle_sha256") != hash_canonical(payload):
        raise NativeBoundedGenerationError("native_schema_v2_bundle_hash_mismatch")
    full = payload.get("full_acceptance_schema")
    provider = payload.get("provider_schema")
    context = payload.get("context_binding")
    if not isinstance(full, Mapping) or not isinstance(provider, Mapping):
        raise NativeBoundedGenerationError("native_schema_v2_bundle_schema_missing")
    if not isinstance(context, Mapping):
        raise NativeBoundedGenerationError("native_schema_v2_bundle_context_missing")
    kind = payload.get("kind")
    identities = {
        "packet": {
            "full_version": PACKET_GENERATION_SCHEMA_V2,
            "provider_version": PACKET_PROVIDER_SCHEMA_V2,
            "full_id_prefix": (
                "urn:literature-multiverse:native-candidate-packet:generation-schema-v2:"
            ),
            "full_id_suffix_pattern": (
                "(?:" + "|".join(re.escape(item) for item in _EFFECT_KINDS) + "):[0-9a-f]{24}"
            ),
            "provider_id_prefix": (
                "urn:literature-multiverse:native-candidate-packet:provider-schema-v2:"
            ),
        },
        "inventory": {
            "full_version": INVENTORY_GENERATION_SCHEMA_V2,
            "provider_version": INVENTORY_PROVIDER_SCHEMA_V2,
            "full_id_prefix": (
                "urn:literature-multiverse:native-candidate-inventory:generation-schema-v2:"
            ),
            "full_id_suffix_pattern": "[0-9a-f]{24}",
            "provider_id_prefix": (
                "urn:literature-multiverse:native-candidate-inventory:provider-schema-v2:"
            ),
        },
    }
    identity = identities.get(kind) if isinstance(kind, str) else None
    if identity is None:
        raise NativeBoundedGenerationError("native_schema_v2_bundle_kind_mismatch")
    fixed_fields = {
        "schema_sent_to_provider": "provider_schema",
        "raw_response_validation_schema": "full_acceptance_schema",
        "provider_schema_scientific_authority": "none",
        "full_acceptance_schema_version": identity["full_version"],
        "provider_schema_version": identity["provider_version"],
    }
    for field_name, expected in fixed_fields.items():
        if payload.get(field_name) != expected:
            raise NativeBoundedGenerationError(
                f"native_schema_v2_bundle_fixed_field_mismatch:{field_name}"
            )
    for schema_name, schema, version, id_prefix, id_suffix_pattern in (
        (
            "full",
            full,
            identity["full_version"],
            identity["full_id_prefix"],
            identity["full_id_suffix_pattern"],
        ),
        (
            "provider",
            provider,
            identity["provider_version"],
            identity["provider_id_prefix"],
            "[0-9a-f]{24}",
        ),
    ):
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise NativeBoundedGenerationError(
                f"native_schema_v2_bundle_{schema_name}_draft_marker_mismatch"
            )
        if schema.get("x-literature-multiverse-generation-schema-version") != version:
            raise NativeBoundedGenerationError(
                f"native_schema_v2_bundle_{schema_name}_version_mismatch"
            )
        schema_id = schema.get("$id")
        if (
            not isinstance(schema_id, str)
            or re.fullmatch(re.escape(id_prefix) + id_suffix_pattern, schema_id) is None
        ):
            raise NativeBoundedGenerationError(f"native_schema_v2_bundle_{schema_name}_id_mismatch")
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise NativeBoundedGenerationError(
                f"native_schema_v2_bundle_{schema_name}_schema_invalid"
            ) from exc
    if payload.get("full_acceptance_schema_sha256") != hash_canonical(full):
        raise NativeBoundedGenerationError("native_schema_v2_bundle_full_hash_mismatch")
    if payload.get("provider_schema_sha256") != hash_canonical(provider):
        raise NativeBoundedGenerationError("native_schema_v2_bundle_provider_hash_mismatch")
    if payload.get("context_binding_sha256") != hash_canonical(context):
        raise NativeBoundedGenerationError("native_schema_v2_bundle_context_hash_mismatch")
    if (
        provider.get("x-literature-multiverse-full-acceptance-schema-sha256")
        != payload["full_acceptance_schema_sha256"]
    ):
        raise NativeBoundedGenerationError("native_schema_v2_bundle_provider_full_binding_mismatch")

    packet_context_keys = {
        "candidate",
        "exposed_line_ids",
        "source_locator",
        "allowed_outcomes",
        "allowed_moderators",
        "allowed_sections",
        "outcome_positive_directions",
    }
    inventory_context_keys = {"exposed_line_ids", "allowed_outcomes"}
    try:
        if kind == "packet":
            if set(context) != packet_context_keys:
                raise NativeBoundedGenerationError(
                    "native_schema_v2_bundle_packet_context_keys_mismatch"
                )
            expected_bundle = packet_schema_bundle_v2(
                candidate=NativeCandidateDescriptor.model_validate(context["candidate"]),
                exposed_line_ids=context["exposed_line_ids"],
                source_locator=context["source_locator"],
                allowed_outcomes=context["allowed_outcomes"],
                allowed_moderators=context["allowed_moderators"],
                allowed_sections=context["allowed_sections"],
                outcome_positive_directions=context["outcome_positive_directions"],
            )
        else:
            if set(context) != inventory_context_keys:
                raise NativeBoundedGenerationError(
                    "native_schema_v2_bundle_inventory_context_keys_mismatch"
                )
            expected_bundle = inventory_schema_bundle_v2(
                exposed_line_ids=context["exposed_line_ids"],
                allowed_outcomes=context["allowed_outcomes"],
            )
    except NativeBoundedGenerationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeBoundedGenerationError("native_schema_v2_bundle_context_invalid") from exc
    if dict(bundle) != expected_bundle:
        raise NativeBoundedGenerationError("native_schema_v2_bundle_replay_mismatch")
    return {
        "schema_bundle_version": SCHEMA_BUNDLE_V2,
        "schema_bundle_sha256": bundle["schema_bundle_sha256"],
        "kind": kind,
        "full_acceptance_schema_version": payload["full_acceptance_schema_version"],
        "full_acceptance_schema_sha256": payload["full_acceptance_schema_sha256"],
        "provider_schema_version": payload["provider_schema_version"],
        "provider_schema_sha256": payload["provider_schema_sha256"],
        "context_binding_sha256": payload["context_binding_sha256"],
        "provider_schema_scientific_authority": "none",
    }


def packet_schema_bundle_v2(
    *,
    candidate: NativeCandidateDescriptor,
    exposed_line_ids: Sequence[str],
    source_locator: str,
    allowed_outcomes: Sequence[str],
    allowed_moderators: Sequence[str] = (),
    allowed_sections: Sequence[str] = ("FigureTable", "Methods", "Results"),
    outcome_positive_directions: Mapping[str, str],
) -> dict[str, Any]:
    """Build the paired provider/full packet schemas for one frozen context."""

    full = packet_generation_schema_v2(
        candidate=candidate,
        exposed_line_ids=exposed_line_ids,
        source_locator=source_locator,
        allowed_outcomes=allowed_outcomes,
        allowed_moderators=allowed_moderators,
        allowed_sections=allowed_sections,
        outcome_positive_directions=outcome_positive_directions,
    )
    provider = _packet_provider_schema_from_full(full, candidate=candidate)
    return _schema_bundle_v2(
        kind="packet",
        full_acceptance_schema=full,
        provider_schema=provider,
        context_binding={
            "candidate": candidate.model_dump(mode="json"),
            "exposed_line_ids": list(exposed_line_ids),
            "source_locator": source_locator,
            "allowed_outcomes": list(allowed_outcomes),
            "allowed_moderators": list(allowed_moderators),
            "allowed_sections": list(allowed_sections),
            "outcome_positive_directions": dict(outcome_positive_directions),
        },
    )


def inventory_schema_bundle_v2(
    *, exposed_line_ids: Sequence[str], allowed_outcomes: Sequence[str]
) -> dict[str, Any]:
    """Build the paired provider/full inventory schemas for one frozen context."""

    full = inventory_generation_schema_v2(
        exposed_line_ids=exposed_line_ids,
        allowed_outcomes=allowed_outcomes,
    )
    provider = _inventory_provider_schema_from_full(full)
    return _schema_bundle_v2(
        kind="inventory",
        full_acceptance_schema=full,
        provider_schema=provider,
        context_binding={
            "exposed_line_ids": list(exposed_line_ids),
            "allowed_outcomes": list(allowed_outcomes),
        },
    )


def schema_v2_contract() -> dict[str, Any]:
    """Return the stable, content-free integration contract and its fingerprint."""

    payload = {
        "contract_version": "native-bounded-generation-schema-v2-contract-v2",
        "packet_generation_schema_version": PACKET_GENERATION_SCHEMA_V2,
        "inventory_generation_schema_version": INVENTORY_GENERATION_SCHEMA_V2,
        "packet_provider_schema_version": PACKET_PROVIDER_SCHEMA_V2,
        "inventory_provider_schema_version": INVENTORY_PROVIDER_SCHEMA_V2,
        "schema_bundle_version": SCHEMA_BUNDLE_V2,
        "accepted_payload_packet_version": PACKET_VERSION,
        "supported_effect_kinds": list(_EFFECT_KINDS),
        "decimal_lexeme_pattern": DECIMAL_LEXEME_PATTERN,
        "nonnegative_decimal_lexeme_pattern": NONNEGATIVE_DECIMAL_LEXEME_PATTERN,
        "unsigned_count_pattern": UNSIGNED_COUNT_PATTERN,
        "positive_count_pattern": POSITIVE_COUNT_PATTERN,
        "at_least_two_count_pattern": AT_LEAST_TWO_COUNT_PATTERN,
        "quote_offset_pattern": QUOTE_OFFSET_PATTERN,
        "required_effect_numeric_support_paths": {
            key: list(value) for key, value in sorted(_REQUIRED_EFFECT_NUMERIC_PATHS.items())
        },
        "candidate_line_ids_are_exact_array_const": True,
        "timepoint_kinds_are_closed_discriminated_branches": True,
        "scientific_number_coercion_or_fabrication_permitted": False,
        "v1_runtime_imports_v2": False,
        "acceptance_authority": "native_bounded_generation_v1_postvalidation",
        "raw_syntactic_acceptance_authority": "full_acceptance_schema_draft202012",
        "provider_schema_scientific_authority": "none",
        "provider_output_must_not_be_validated_only_against_provider_schema": True,
        "intent_and_receipt_required_schema_hashes": [
            "provider_schema_sha256",
            "full_acceptance_schema_sha256",
        ],
        "schema_sent_to_provider": "provider_schema",
        "raw_response_validation_schema": "full_acceptance_schema",
        "raw_generation_validation_order": [
            "json_parse_without_normalization",
            "draft202012_schema_v2",
            "pydantic_and_scientific_postvalidation_v1",
            "raw_to_canonical_authored_value_preservation",
        ],
        "provider_grammar_scope": deepcopy(PROVIDER_GRAMMAR_SCOPE_V2),
        "packet_validator_coverage": [dict(item) for item in PACKET_VALIDATOR_COVERAGE_V2],
        "inventory_validator_coverage": [dict(item) for item in INVENTORY_VALIDATOR_COVERAGE_V2],
        "unavoidable_postvalidation_invariants": list(UNAVOIDABLE_POSTVALIDATION_INVARIANTS),
    }
    return {**payload, "contract_sha256": hash_canonical(payload)}


def synthetic_completed_packet_example(effect_kind: str) -> dict[str, Any]:
    """Build a source-free completed example that exercises one v2 effect branch."""

    common: dict[str, Any] = {
        "effect_kind": effect_kind,
        "reported_p_value": None,
        "reported_significance": "not_reported",
        "equivalence_conclusion": "not_tested",
        "equivalence_margin": None,
        "moderators": [],
        "extraction_method": "reported",
    }
    if effect_kind == "direct_standard_error":
        effect = {
            **common,
            "effect_format": "mean_difference",
            "estimate": "0.5",
            "standard_error": "0.2",
            "unit": None,
        }
        values = {"effect.estimate": "0.5", "effect.standard_error": "0.2"}
    elif effect_kind == "direct_variance":
        effect = {
            **common,
            "effect_format": "mean_difference",
            "estimate": "0.5",
            "variance": "0.04",
            "unit": None,
        }
        values = {"effect.estimate": "0.5", "effect.variance": "0.04"}
    elif effect_kind == "direct_confidence_interval":
        effect = {
            **common,
            "effect_format": "mean_difference",
            "estimate": "0.5",
            "ci_lower": "0.1",
            "ci_upper": "0.9",
            "ci_level": "0.95",
            "unit": None,
        }
        values = {
            "effect.ci_level": "0.95",
            "effect.ci_lower": "0.1",
            "effect.ci_upper": "0.9",
            "effect.estimate": "0.5",
        }
    elif effect_kind == "continuous_group_statistics":
        effect = {
            **common,
            "effect_format": "mean_difference",
            "treatment_mean": "1.5",
            "treatment_sd": "0.5",
            "treatment_n": "21",
            "control_mean": "1.0",
            "control_sd": "0.4",
            "control_n": "22",
            "unit": None,
        }
        values = {
            "effect.control_mean": "1.0",
            "effect.control_n": "22",
            "effect.control_sd": "0.4",
            "effect.treatment_mean": "1.5",
            "effect.treatment_n": "21",
            "effect.treatment_sd": "0.5",
        }
    elif effect_kind == "binary_group_statistics":
        effect = {
            **common,
            "effect_format": "risk_ratio",
            "treatment_events": "10",
            "treatment_total": "21",
            "control_events": "5",
            "control_total": "22",
        }
        values = {
            "effect.control_events": "5",
            "effect.control_total": "22",
            "effect.treatment_events": "10",
            "effect.treatment_total": "21",
        }
    else:
        raise NativeBoundedGenerationError(
            f"native_schema_v2_synthetic_effect_kind_unknown:{effect_kind}"
        )

    quote = "; ".join(f"{path}={token}" for path, token in sorted(values.items()))
    numeric_support: list[dict[str, Any]] = []
    for path, token in sorted(values.items()):
        start = quote.index(f"{path}={token}") + len(path) + 1
        numeric_support.append(
            {
                "field_path": path,
                "verbatim_token": token,
                "normalization": "identity",
                "quote_start": str(start),
                "quote_end": str(start + len(token)),
            }
        )
    return {
        "packet_version": PACKET_VERSION,
        "packet_status": "completed",
        "candidate_index": 1,
        "study": {
            "key": "synthetic-study",
            "source_label": "Synthetic schema compilation study",
            "design": None,
            "registration_ids": [],
        },
        "cohort": {
            "key": "synthetic-cohort",
            "source_labels": ["Synthetic schema compilation cohort"],
            "registry_ids": [],
            "dataset_ids": [],
            "population_description": None,
            "recruitment_period": None,
            "total_sample_size": None,
        },
        "treatment_arm": {
            "key": "synthetic-treatment",
            "label": "Synthetic treatment",
            "role": "intervention",
            "description": None,
            "sample_size": None,
        },
        "comparator_arm": {
            "key": "synthetic-control",
            "label": "Synthetic control",
            "role": "control",
            "description": None,
            "sample_size": None,
        },
        "contrast": {
            "key": "synthetic-target",
            "label": "Synthetic treatment versus control",
            "estimand": None,
            "positive_direction_means": "larger synthetic target value",
        },
        "finding": {
            "key": "synthetic-finding",
            "outcome_name": "synthetic_outcome",
            "timepoint": {
                "kind": "not_reported",
                "value": None,
                "lower": None,
                "upper": None,
                "unit": None,
                "anchor": None,
                "raw_label": None,
            },
            "analysis_population": None,
        },
        "effect": effect,
        "evidence": {
            "source_locator": "synthetic:schema-compilation-only",
            "quote": quote,
            "section": "Synthetic",
            "line_ids": ["SYNTHETIC_LINE"],
        },
        "numeric_support": numeric_support,
    }


def synthetic_schema_v2_preflight_specs() -> list[dict[str, Any]]:
    """Return three inventory plus five packet compatibility calls.

    All examples are real-source-free and claim-free.  The inventory examples
    exercise every coherent state branch.  The packet examples are completed
    synthetic fixtures so every effect branch and its lexical constraints must
    compile and validate before a real run is permitted.
    """

    inventory_bundle = inventory_schema_bundle_v2(
        exposed_line_ids=["SYNTHETIC_LINE"],
        allowed_outcomes=["synthetic_outcome"],
    )
    inventory_binding = schema_bundle_receipt_binding_v2(inventory_bundle)
    synthetic_candidate = {
        "candidate_index": 1,
        "outcome_name": "synthetic_outcome",
        "effect_kind": "direct_standard_error",
        "line_ids": ["SYNTHETIC_LINE"],
    }
    inventory_examples = (
        (
            "no_candidate_found",
            {
                "inventory_version": "native-candidate-inventory-v1",
                "inventory_status": "no_candidate_found",
                "candidates": [],
                "has_more_or_uncertain": False,
            },
        ),
        (
            "candidates_found",
            {
                "inventory_version": "native-candidate-inventory-v1",
                "inventory_status": "candidates_found",
                "candidates": [deepcopy(synthetic_candidate)],
                "has_more_or_uncertain": False,
            },
        ),
        (
            "overflow_or_uncertain",
            {
                "inventory_version": "native-candidate-inventory-v1",
                "inventory_status": "overflow_or_uncertain",
                "candidates": [deepcopy(synthetic_candidate)],
                "has_more_or_uncertain": True,
            },
        ),
    )
    specs: list[dict[str, Any]] = []
    for index, (inventory_state, inventory_example) in enumerate(inventory_examples):
        specs.append(
            {
                "call_id": f"{index:02d}-inventory-{inventory_state}-v2",
                "kind": "inventory",
                "effect_kind": None,
                "inventory_state": inventory_state,
                "schema": inventory_bundle["provider_schema"],
                "schema_sha256": inventory_bundle["provider_schema_sha256"],
                "provider_schema": inventory_bundle["provider_schema"],
                "provider_schema_sha256": inventory_bundle["provider_schema_sha256"],
                "full_acceptance_schema": inventory_bundle["full_acceptance_schema"],
                "full_acceptance_schema_sha256": inventory_bundle["full_acceptance_schema_sha256"],
                "schema_bundle_sha256": inventory_bundle["schema_bundle_sha256"],
                "receipt_binding": inventory_binding,
                "valid_example": inventory_example,
                "valid_example_sha256": hash_canonical(inventory_example),
                "contains_real_source_or_claim_content": False,
                "contains_only_synthetic_numeric_fixture": False,
            }
        )
    for index, effect_kind in enumerate(_EFFECT_KINDS, start=3):
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
        example = synthetic_completed_packet_example(effect_kind)
        specs.append(
            {
                "call_id": f"{index:02d}-packet-{effect_kind}-v2",
                "kind": "packet",
                "effect_kind": effect_kind,
                "inventory_state": None,
                "schema": bundle["provider_schema"],
                "schema_sha256": bundle["provider_schema_sha256"],
                "provider_schema": bundle["provider_schema"],
                "provider_schema_sha256": bundle["provider_schema_sha256"],
                "full_acceptance_schema": bundle["full_acceptance_schema"],
                "full_acceptance_schema_sha256": bundle["full_acceptance_schema_sha256"],
                "schema_bundle_sha256": bundle["schema_bundle_sha256"],
                "receipt_binding": schema_bundle_receipt_binding_v2(bundle),
                "valid_example": example,
                "valid_example_sha256": hash_canonical(example),
                "contains_real_source_or_claim_content": False,
                "contains_only_synthetic_numeric_fixture": True,
            }
        )
    return specs


def synthetic_schema_v2_preflight_fingerprint() -> str:
    """Fingerprint the exact eight source-free schemas and expected examples."""

    return hash_canonical(synthetic_schema_v2_preflight_specs())


def _selected_completed_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    name = _completed_definition_name(schema)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": deepcopy(schema["$defs"]),
        "$ref": f"#/$defs/{name}",
    }


def _redacted_path(path: Sequence[Any]) -> str:
    output = ["*" if isinstance(item, int) else str(item) for item in path]
    return ".".join(output) or "$"


def _schema_failure_reasons(schema: Mapping[str, Any], payload: Mapping[str, Any]) -> set[str]:
    reasons: set[str] = set()

    def leaves(error: Any) -> list[Any]:
        if not error.context:
            return [error]
        output: list[Any] = []
        for child in error.context:
            output.extend(leaves(child))
        return output

    def selected_union_context(error: Any) -> list[Any]:
        """Select the discriminator-compatible oneOf/anyOf branch.

        Reporting every failed alternative would falsely claim, for example, that
        an ``exact`` timepoint also failed the ``reported_text`` kind const.  The
        selection uses only schema validators and paths; no instance value enters
        the returned aggregate report.
        """

        groups: dict[int, list[Any]] = {}
        for child in error.context:
            relative = list(child.relative_schema_path)
            branch_index = relative[0] if relative and isinstance(relative[0], int) else -1
            groups.setdefault(branch_index, []).append(child)
        if len(groups) <= 1:
            return list(error.context)

        discriminator_fields = {"kind", "effect_format", "normalization"}

        def score(item: tuple[int, list[Any]]) -> tuple[int, int, int]:
            branch_index, branch_errors = item
            leaf_errors = [leaf for err in branch_errors for leaf in leaves(err)]
            discriminator_errors = sum(
                1
                for leaf in leaf_errors
                if leaf.validator in {"const", "enum"}
                and list(leaf.absolute_path)
                and str(list(leaf.absolute_path)[-1]) in discriminator_fields
            )
            return discriminator_errors, len(leaf_errors), branch_index

        return min(groups.items(), key=score)[1]

    def collect(error: Any) -> None:
        if error.context:
            children = (
                selected_union_context(error)
                if error.validator in {"oneOf", "anyOf"}
                else list(error.context)
            )
            for child in children:
                collect(child)
            return
        path = _redacted_path(list(error.absolute_path))
        validator = str(error.validator or "unknown")
        if validator == "required" and isinstance(error.validator_value, list):
            instance = error.instance if isinstance(error.instance, Mapping) else {}
            missing = [
                item
                for item in error.validator_value
                if isinstance(item, str) and item not in instance
            ]
            if missing:
                for item in missing:
                    reasons.add(f"required:{path}.{item}")
                return
        reasons.add(f"{validator}:{path}")

    for validation_error in Draft202012Validator(schema).iter_errors(payload):
        collect(validation_error)
    return reasons


def validate_raw_payload_against_schema_v2(
    value: Any, *, schema: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the unnormalized JSON object before any ContractModel parsing.

    The exception includes only schema validator/path names.  It never includes an
    instance value, source quote, or model response fragment.
    """

    if not isinstance(value, dict):
        raise NativeBoundedGenerationError("native_schema_v2_raw_payload_not_object")
    marker = schema.get("x-literature-multiverse-generation-schema-version")
    recognized_markers = {
        PACKET_GENERATION_SCHEMA_V2: (
            "urn:literature-multiverse:native-candidate-packet:generation-schema-v2:"
        ),
        INVENTORY_GENERATION_SCHEMA_V2: (
            "urn:literature-multiverse:native-candidate-inventory:generation-schema-v2:"
        ),
    }
    expected_id_prefix = recognized_markers.get(marker)
    if expected_id_prefix is None:
        raise NativeBoundedGenerationError(
            "native_schema_v2_raw_validation_marker_missing_or_unknown"
        )
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise NativeBoundedGenerationError("native_schema_v2_raw_validation_draft_marker_mismatch")
    schema_id = schema.get("$id")
    if not isinstance(schema_id, str) or not schema_id.startswith(expected_id_prefix):
        raise NativeBoundedGenerationError("native_schema_v2_raw_validation_id_mismatch")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise NativeBoundedGenerationError(
            "native_schema_v2_raw_validation_schema_invalid"
        ) from exc
    payload = deepcopy(value)
    reasons = _schema_failure_reasons(schema, payload)
    if reasons:
        raise NativeBoundedGenerationError(
            "native_schema_v2_raw_validation_error:" + ",".join(sorted(reasons))
        )
    return payload


def assert_raw_to_canonical_preservation(
    raw: Any, canonical: Any, *, _path: tuple[Any, ...] = ()
) -> None:
    """Reject any Pydantic normalization of a model-authored value.

    Canonical models may add omitted default-valued object properties.  Every value
    that the model actually authored, including each list element, must retain the
    exact JSON type and value at the exact same path.
    """

    path = _redacted_path(_path)
    if isinstance(raw, dict):
        if not isinstance(canonical, Mapping):
            raise NativeBoundedGenerationError(
                f"native_schema_v2_raw_canonical_type_changed:{path}"
            )
        for key, value in raw.items():
            if key not in canonical:
                raise NativeBoundedGenerationError(
                    f"native_schema_v2_raw_canonical_path_missing:{path}.{key}"
                )
            assert_raw_to_canonical_preservation(value, canonical[key], _path=(*_path, key))
        return
    if isinstance(raw, list):
        if not isinstance(canonical, list) or len(raw) != len(canonical):
            raise NativeBoundedGenerationError(
                f"native_schema_v2_raw_canonical_list_changed:{path}"
            )
        for index, (raw_item, canonical_item) in enumerate(zip(raw, canonical, strict=True)):
            assert_raw_to_canonical_preservation(raw_item, canonical_item, _path=(*_path, index))
        return
    if type(raw) is not type(canonical):
        raise NativeBoundedGenerationError(f"native_schema_v2_raw_canonical_type_changed:{path}")
    if raw != canonical:
        raise NativeBoundedGenerationError(f"native_schema_v2_raw_canonical_value_changed:{path}")


def validate_inventory_for_row_v2(
    value: Any,
    *,
    exposed_line_ids: Sequence[str],
    allowed_outcomes: Sequence[str],
) -> NativeCandidateInventory:
    """Apply raw Draft v2, v1 authority, then no-normalization preservation."""

    schema = inventory_generation_schema_v2(
        exposed_line_ids=exposed_line_ids,
        allowed_outcomes=allowed_outcomes,
    )
    raw = validate_raw_payload_against_schema_v2(value, schema=schema)
    inventory = validate_inventory_for_row(
        raw,
        exposed_line_ids=exposed_line_ids,
        allowed_outcomes=allowed_outcomes,
    )
    assert_raw_to_canonical_preservation(raw, inventory.model_dump(mode="json"))
    return inventory


def validate_packet_for_candidate_v2(
    value: Any,
    *,
    candidate: NativeCandidateDescriptor,
    exposed_line_ids: Sequence[str],
    source_locator: str,
    allowed_outcomes: Sequence[str],
    allowed_moderators: Sequence[str],
    allowed_sections: Sequence[str] = ("FigureTable", "Methods", "Results"),
    outcome_positive_directions: Mapping[str, str],
) -> NativeCandidatePacketOutcome:
    """Apply the mandatory v2-to-v1 fail-closed packet validation sequence."""

    schema = packet_generation_schema_v2(
        candidate=candidate,
        exposed_line_ids=exposed_line_ids,
        source_locator=source_locator,
        allowed_outcomes=allowed_outcomes,
        allowed_moderators=allowed_moderators,
        allowed_sections=allowed_sections,
        outcome_positive_directions=outcome_positive_directions,
    )
    raw = validate_raw_payload_against_schema_v2(value, schema=schema)
    packet = validate_packet_for_candidate(
        raw,
        candidate=candidate,
        exposed_line_ids=exposed_line_ids,
        source_locator=source_locator,
        allowed_outcomes=allowed_outcomes,
        allowed_moderators=allowed_moderators,
        allowed_sections=allowed_sections,
        outcome_positive_directions=outcome_positive_directions,
    )
    assert_raw_to_canonical_preservation(raw, packet.model_dump(mode="json"))
    return packet


def _administrative_packet_envelope(
    payload: Mapping[str, Any], *, candidate: NativeCandidateDescriptor
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Inject only frozen contract discriminators; never touch scientific values."""

    output = deepcopy(dict(payload))
    inserted: list[str] = []
    for name, value in (
        ("packet_version", PACKET_VERSION),
        ("packet_status", "completed"),
    ):
        if name not in output:
            output[name] = value
            inserted.append(name)
    effect = output.get("effect")
    if isinstance(effect, dict) and "effect_kind" not in effect:
        effect["effect_kind"] = candidate.effect_kind
        inserted.append("effect.effect_kind")
    return output, tuple(sorted(inserted))


def audit_saved_v1_packet_receipts(workspace: Path) -> dict[str, Any]:
    """Aggregate-only v2 countervalidation of frozen v1 parsed packet outputs.

    The returned object contains no row keys, source text, quotes, model responses,
    emitted values, or per-packet records.  It is diagnostic evidence only and can
    never authorize promotion of a v1 output.
    """

    receipt_paths = sorted((workspace / "packet-receipts").glob("*/*.json"))
    if not receipt_paths:
        raise NativeBoundedGenerationError("native_schema_v2_saved_packet_receipts_missing")
    reason_counts: Counter[str] = Counter()
    injection_counts: Counter[str] = Counter()
    parsed_hashes: list[str] = []
    valid_after_envelope = 0
    for path in receipt_paths:
        receipt = json.loads(path.read_text())
        if not isinstance(receipt, dict):
            raise NativeBoundedGenerationError("native_schema_v2_saved_packet_receipt_not_object")
        parsed = receipt.get("parsed_output")
        candidate_payload = receipt.get("candidate")
        schema_v1 = receipt.get("schema")
        if not isinstance(parsed, dict) or not isinstance(schema_v1, dict):
            raise NativeBoundedGenerationError("native_schema_v2_saved_packet_payload_missing")
        candidate = NativeCandidateDescriptor.model_validate(candidate_payload)
        parsed_sha256 = hash_canonical(parsed)
        if receipt.get("parsed_output_sha256") != parsed_sha256:
            raise NativeBoundedGenerationError(
                "native_schema_v2_saved_packet_payload_hash_mismatch"
            )
        schema_v2 = upgrade_packet_schema_v1_to_v2(schema_v1, candidate=candidate)
        envelope, inserted = _administrative_packet_envelope(parsed, candidate=candidate)
        injection_counts.update(inserted)
        reasons = _schema_failure_reasons(_selected_completed_schema(schema_v2), envelope)
        if reasons:
            reason_counts.update(reasons)
        else:
            valid_after_envelope += 1
        parsed_hashes.append(parsed_sha256)

    packet_count = len(receipt_paths)
    payload = {
        "audit_version": "native-bounded-v1-saved-output-schema-v2-audit-v1",
        "source_contract": "native-bounded-two-stage-generation-v1",
        "candidate_generation_schema": PACKET_GENERATION_SCHEMA_V2,
        "scope": "aggregate_only_countervalidation_of_frozen_parsed_outputs",
        "packet_count": packet_count,
        "invalid_after_administrative_envelope_count": (packet_count - valid_after_envelope),
        "valid_after_administrative_envelope_count": valid_after_envelope,
        "all_saved_outputs_rejected": valid_after_envelope == 0,
        "administrative_fields_injected_counts": dict(sorted(injection_counts.items())),
        "schema_failure_reason_counts": dict(sorted(reason_counts.items())),
        "source_payload_set_sha256": hash_canonical(sorted(parsed_hashes)),
        "contains_source_or_response_content": False,
        "contains_scientific_values": False,
        "authorizes_v1_output_repair_or_promotion": False,
        "scientific_claim_authority": "none",
    }
    return {**payload, "audit_sha256": hash_canonical(payload)}


__all__ = [
    "DECIMAL_LEXEME_PATTERN",
    "INVENTORY_GENERATION_SCHEMA_V2",
    "INVENTORY_PROVIDER_SCHEMA_V2",
    "INVENTORY_VALIDATOR_COVERAGE_V2",
    "PACKET_GENERATION_SCHEMA_V2",
    "PACKET_PROVIDER_SCHEMA_V2",
    "PACKET_VALIDATOR_COVERAGE_V2",
    "PROVIDER_DECIMAL_OR_PERCENT_TOKEN_PATTERN",
    "PROVIDER_GRAMMAR_SCOPE_V2",
    "SCHEMA_BUNDLE_V2",
    "UNAVOIDABLE_POSTVALIDATION_INVARIANTS",
    "assert_raw_to_canonical_preservation",
    "audit_saved_v1_packet_receipts",
    "inventory_generation_schema_v2",
    "inventory_provider_schema_v2",
    "inventory_schema_bundle_v2",
    "packet_generation_schema_v2",
    "packet_provider_schema_v2",
    "packet_schema_bundle_v2",
    "schema_bundle_receipt_binding_v2",
    "schema_v2_contract",
    "synthetic_completed_packet_example",
    "synthetic_schema_v2_preflight_fingerprint",
    "synthetic_schema_v2_preflight_specs",
    "upgrade_packet_schema_v1_to_v2",
    "validate_inventory_for_row_v2",
    "validate_packet_for_candidate_v2",
    "validate_raw_payload_against_schema_v2",
]
