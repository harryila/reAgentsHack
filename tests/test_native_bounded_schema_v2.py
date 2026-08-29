from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

import literature_multiverse.native_bounded_schema_v2 as schema_v2_module
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.native_bounded_generation import (
    PACKET_MODELS,
    BoundedArm,
    BoundedCohortHeader,
    BoundedContrast,
    BoundedEvidence,
    BoundedFindingHeader,
    BoundedNumericSupport,
    BoundedStudyHeader,
    BoundedTimepoint,
    DirectStandardErrorEffect,
    NativeBoundedGenerationError,
    NativeCandidateDescriptor,
    NativeCandidatePacket,
    inventory_generation_schema,
    packet_generation_schema,
    validate_inventory_for_row,
    validate_packet_for_candidate,
)
from literature_multiverse.native_bounded_schema_v2 import (
    INVENTORY_PROVIDER_SCHEMA_V2,
    INVENTORY_VALIDATOR_COVERAGE_V2,
    PACKET_GENERATION_SCHEMA_V2,
    PACKET_PROVIDER_SCHEMA_V2,
    PACKET_VALIDATOR_COVERAGE_V2,
    PROVIDER_GRAMMAR_SCOPE_V2,
    SCHEMA_BUNDLE_V2,
    audit_saved_v1_packet_receipts,
    inventory_generation_schema_v2,
    inventory_provider_schema_v2,
    inventory_schema_bundle_v2,
    packet_generation_schema_v2,
    packet_provider_schema_v2,
    packet_schema_bundle_v2,
    schema_bundle_receipt_binding_v2,
    schema_v2_contract,
    synthetic_schema_v2_preflight_fingerprint,
    synthetic_schema_v2_preflight_specs,
    upgrade_packet_schema_v1_to_v2,
    validate_packet_for_candidate_v2,
    validate_raw_payload_against_schema_v2,
)

OUTCOME = "synthetic_outcome"
LINES = ["L10", "L20"]
LOCATOR = "synthetic:bounded-schema-v2"
DIRECTION = {OUTCOME: "larger synthetic target value"}


def _candidate(*, effect_kind: str = "direct_standard_error") -> NativeCandidateDescriptor:
    return NativeCandidateDescriptor(
        candidate_index=1,
        outcome_name=OUTCOME,
        effect_kind=effect_kind,
        line_ids=LINES,
    )


def _schema(candidate: NativeCandidateDescriptor) -> dict[str, Any]:
    return packet_generation_schema_v2(
        candidate=candidate,
        exposed_line_ids=LINES,
        source_locator=LOCATOR,
        allowed_outcomes=[OUTCOME],
        allowed_moderators=[],
        allowed_sections=["Results"],
        outcome_positive_directions=DIRECTION,
    )


def _provider_schema(candidate: NativeCandidateDescriptor) -> dict[str, Any]:
    return packet_provider_schema_v2(
        candidate=candidate,
        exposed_line_ids=LINES,
        source_locator=LOCATOR,
        allowed_outcomes=[OUTCOME],
        allowed_moderators=[],
        allowed_sections=["Results"],
        outcome_positive_directions=DIRECTION,
    )


def _support(path: str, token: str, quote: str) -> BoundedNumericSupport:
    start = quote.index(token)
    return BoundedNumericSupport(
        field_path=path,
        verbatim_token=token,
        normalization="identity",
        quote_start=str(start),
        quote_end=str(start + len(token)),
    )


def _valid_packet(
    timepoint: BoundedTimepoint | None = None,
) -> tuple[NativeCandidateDescriptor, dict[str, Any]]:
    candidate = _candidate()
    quote = "At follow-up, values were 8, 12, 0.5, and 0.2."
    actual_timepoint = timepoint or BoundedTimepoint(kind="not_reported")
    numeric_support = [
        _support("effect.estimate", "0.5", quote),
        _support("effect.standard_error", "0.2", quote),
    ]
    if actual_timepoint.value is not None:
        numeric_support.append(_support("finding.timepoint.value", actual_timepoint.value, quote))
    if actual_timepoint.lower is not None:
        numeric_support.append(_support("finding.timepoint.lower", actual_timepoint.lower, quote))
    if actual_timepoint.upper is not None:
        numeric_support.append(_support("finding.timepoint.upper", actual_timepoint.upper, quote))
    packet = NativeCandidatePacket[DirectStandardErrorEffect](
        candidate_index=1,
        study=BoundedStudyHeader(
            key="study-1",
            source_label="Synthetic study",
            design=None,
            registration_ids=[],
        ),
        cohort=BoundedCohortHeader(
            key="cohort-1",
            source_labels=["Synthetic cohort"],
            registry_ids=[],
            dataset_ids=[],
            population_description=None,
            recruitment_period=None,
            total_sample_size=None,
        ),
        treatment_arm=BoundedArm(
            key="treatment",
            label="Synthetic treatment",
            role="intervention",
            description=None,
            sample_size=None,
        ),
        comparator_arm=BoundedArm(
            key="control",
            label="Synthetic control",
            role="control",
            description=None,
            sample_size=None,
        ),
        contrast=BoundedContrast(
            key="target",
            label="treatment versus control",
            estimand=None,
            positive_direction_means=DIRECTION[OUTCOME],
        ),
        finding=BoundedFindingHeader(
            key="finding-1",
            outcome_name=OUTCOME,
            timepoint=actual_timepoint,
            analysis_population=None,
        ),
        effect=DirectStandardErrorEffect(
            effect_format="mean_difference",
            estimate="0.5",
            standard_error="0.2",
            unit=None,
        ),
        evidence=BoundedEvidence(
            source_locator=LOCATOR,
            quote=quote,
            section="Results",
            line_ids=LINES,
        ),
        numeric_support=sorted(numeric_support, key=lambda item: item.field_path),
    )
    return candidate, packet.model_dump(mode="json")


def _validate_v1_authority(candidate: NativeCandidateDescriptor, payload: dict[str, Any]) -> None:
    validate_packet_for_candidate(
        payload,
        candidate=candidate,
        exposed_line_ids=LINES,
        source_locator=LOCATOR,
        allowed_outcomes=[OUTCOME],
        allowed_moderators=[],
        allowed_sections=["Results"],
        outcome_positive_directions=DIRECTION,
    )


def _completed_effect_payload(
    effect_kind: str,
) -> tuple[NativeCandidateDescriptor, dict[str, Any]]:
    _, payload = _valid_packet()
    candidate = _candidate(effect_kind=effect_kind)
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
            "effect.estimate": "0.5",
            "effect.ci_lower": "0.1",
            "effect.ci_upper": "0.9",
            "effect.ci_level": "0.95",
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
            "effect.treatment_mean": "1.5",
            "effect.treatment_sd": "0.5",
            "effect.treatment_n": "21",
            "effect.control_mean": "1.0",
            "effect.control_sd": "0.4",
            "effect.control_n": "22",
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
            "effect.treatment_events": "10",
            "effect.treatment_total": "21",
            "effect.control_events": "5",
            "effect.control_total": "22",
        }
    else:  # pragma: no cover - parameterized from the closed production mapping
        raise AssertionError(effect_kind)

    quote = "; ".join(f"{path}={token}" for path, token in sorted(values.items()))
    support: list[dict[str, str]] = []
    for path, token in sorted(values.items()):
        start = quote.index(f"{path}={token}") + len(path) + 1
        support.append(
            {
                "field_path": path,
                "verbatim_token": token,
                "normalization": "identity",
                "quote_start": str(start),
                "quote_end": str(start + len(token)),
            }
        )
    payload["effect"] = effect
    payload["evidence"]["quote"] = quote
    payload["numeric_support"] = support
    return candidate, payload


def test_v2_contract_and_eight_source_free_preflight_examples_are_replayable() -> None:
    contract = schema_v2_contract()
    unsigned = {key: value for key, value in contract.items() if key != "contract_sha256"}
    assert contract["contract_sha256"] == hash_canonical(unsigned)
    assert contract["packet_generation_schema_version"] == PACKET_GENERATION_SCHEMA_V2
    assert contract["packet_provider_schema_version"] == PACKET_PROVIDER_SCHEMA_V2
    assert contract["inventory_provider_schema_version"] == INVENTORY_PROVIDER_SCHEMA_V2
    assert contract["schema_bundle_version"] == SCHEMA_BUNDLE_V2
    assert contract["scientific_number_coercion_or_fabrication_permitted"] is False
    assert contract["v1_runtime_imports_v2"] is False
    assert contract["provider_grammar_scope"] == PROVIDER_GRAMMAR_SCOPE_V2
    assert PROVIDER_GRAMMAR_SCOPE_V2["provider_grammar_enforcement_assumed"] is False
    assert contract["provider_schema_scientific_authority"] == "none"
    assert contract["schema_sent_to_provider"] == "provider_schema"
    assert contract["raw_response_validation_schema"] == "full_acceptance_schema"
    assert contract["raw_generation_validation_order"][1] == "draft202012_schema_v2"
    assert PROVIDER_GRAMMAR_SCOPE_V2["provider_pattern_policy"] == (
        "omit_all_pattern_keywords_for_ollama_0_15_1_grammar_compatibility"
    )
    assert PROVIDER_GRAMMAR_SCOPE_V2["provider_pattern_lexical_enforcement"] == (
        "mandatory_full_acceptance_raw_draft202012_validation_only"
    )
    assert PROVIDER_GRAMMAR_SCOPE_V2["eight_call_preflight_proves"] == (
        "whole_schema_request_compatibility_only"
    )
    assert "pattern" in PROVIDER_GRAMMAR_SCOPE_V2["known_potentially_skipped_keywords"]
    assert "string_lexeme_patterns" in PROVIDER_GRAMMAR_SCOPE_V2["provider_only_simplifications"]

    specs = synthetic_schema_v2_preflight_specs()
    assert len(specs) == 8
    assert {item["effect_kind"] for item in specs[3:]} == set(PACKET_MODELS)
    assert [item["inventory_state"] for item in specs] == [
        "no_candidate_found",
        "candidates_found",
        "overflow_or_uncertain",
        None,
        None,
        None,
        None,
        None,
    ]
    assert synthetic_schema_v2_preflight_fingerprint() == hash_canonical(specs)
    assert {spec["call_id"]: spec["full_acceptance_schema_sha256"] for spec in specs} == {
        "00-inventory-no_candidate_found-v2": (
            "9955eb0a31de3aec8a4d2f98690feae024059b052534e2034857c0bef0f61785"
        ),
        "01-inventory-candidates_found-v2": (
            "9955eb0a31de3aec8a4d2f98690feae024059b052534e2034857c0bef0f61785"
        ),
        "02-inventory-overflow_or_uncertain-v2": (
            "9955eb0a31de3aec8a4d2f98690feae024059b052534e2034857c0bef0f61785"
        ),
        "03-packet-binary_group_statistics-v2": (
            "4edc1fa0543d53f30489c301c27e771c227572698a71dd46fc962a5ecb5d7ae3"
        ),
        "04-packet-continuous_group_statistics-v2": (
            "4f0ee53800a7a00b74c068b9487257507a69892f691be3fc2231bb832d1a7a07"
        ),
        "05-packet-direct_confidence_interval-v2": (
            "98d0c428b10368b5015888b6b83b6b43a089caab8f31adb3c2e572f443bc5e93"
        ),
        "06-packet-direct_standard_error-v2": (
            "9357dc23e43c88d953c252343c58d945910dfb49c956b8d814fd8d9c0a7bbad6"
        ),
        "07-packet-direct_variance-v2": (
            "a856bfe85e2d9699d146ec2d178378bd55517cd2813bcc6a5b18b0fc8c29d1f2"
        ),
    }
    for spec in specs:
        assert spec["contains_real_source_or_claim_content"] is False
        if spec["kind"] == "inventory":
            assert spec["inventory_state"] == spec["valid_example"]["inventory_status"]
        else:
            assert spec["inventory_state"] is None
        assert spec["schema_sha256"] == hash_canonical(spec["schema"])
        assert spec["schema"] == spec["provider_schema"]
        assert spec["provider_schema_sha256"] == hash_canonical(spec["provider_schema"])
        assert _all_schema_patterns(spec["provider_schema"]) == []
        assert spec["full_acceptance_schema_sha256"] == hash_canonical(
            spec["full_acceptance_schema"]
        )
        assert spec["receipt_binding"]["provider_schema_sha256"] == spec["provider_schema_sha256"]
        assert (
            spec["receipt_binding"]["full_acceptance_schema_sha256"]
            == spec["full_acceptance_schema_sha256"]
        )
        assert spec["receipt_binding"]["schema_bundle_sha256"] == spec["schema_bundle_sha256"]
        assert spec["valid_example_sha256"] == hash_canonical(spec["valid_example"])
        for schema in (spec["provider_schema"], spec["full_acceptance_schema"]):
            Draft202012Validator.check_schema(schema)
            assert not list(Draft202012Validator(schema).iter_errors(spec["valid_example"]))
        if spec["kind"] == "packet":
            assert spec["contains_only_synthetic_numeric_fixture"] is True
            assert _all_schema_patterns(spec["full_acceptance_schema"])
            candidate = NativeCandidateDescriptor(
                candidate_index=1,
                outcome_name="synthetic_outcome",
                effect_kind=spec["effect_kind"],
                line_ids=["SYNTHETIC_LINE"],
            )
            validate_packet_for_candidate(
                spec["valid_example"],
                candidate=candidate,
                exposed_line_ids=["SYNTHETIC_LINE"],
                source_locator="synthetic:schema-compilation-only",
                allowed_outcomes=["synthetic_outcome"],
                allowed_moderators=[],
                allowed_sections=["Synthetic"],
                outcome_positive_directions={"synthetic_outcome": "larger synthetic target value"},
            )


def _all_mapping_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_mapping_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_mapping_keys(item) for item in value))
    return set()


def _all_schema_patterns(value: Any) -> list[str]:
    if isinstance(value, dict):
        output = [value["pattern"]] if isinstance(value.get("pattern"), str) else []
        return output + [
            pattern for item in value.values() for pattern in _all_schema_patterns(item)
        ]
    if isinstance(value, list):
        return [pattern for item in value for pattern in _all_schema_patterns(item)]
    return []


def _canonical_json_size(value: Any) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":")))


def test_provider_annotation_pruning_preserves_property_and_definition_name_collisions() -> None:
    collision_names = {"default", "description", "examples", "pattern", "title"}
    source = {
        "type": "object",
        "description": "root annotation",
        "title": "root annotation",
        "default": {},
        "examples": [],
        "properties": {
            name: {"type": "string", "description": "field annotation"}
            for name in sorted(collision_names)
        },
        "$defs": {name: {"type": "string"} for name in sorted(collision_names)},
    }
    compact = schema_v2_module._compact_provider_keywords(source)
    assert collision_names.isdisjoint(compact)
    assert set(compact["properties"]) == collision_names
    assert set(compact["$defs"]) == collision_names
    assert all("description" not in node for node in compact["properties"].values())


@pytest.mark.parametrize("effect_kind", sorted(PACKET_MODELS))
def test_compact_packet_provider_schema_is_deterministic_bound_and_dual_valid(
    effect_kind: str,
) -> None:
    candidate, payload = _completed_effect_payload(effect_kind)
    full_before = _schema(candidate)
    full_sha256 = hash_canonical(full_before)
    provider = _provider_schema(candidate)
    assert provider == _provider_schema(candidate)
    assert provider["x-literature-multiverse-generation-schema-version"] == (
        PACKET_PROVIDER_SCHEMA_V2
    )
    assert provider["x-literature-multiverse-full-acceptance-schema-sha256"] == (full_sha256)
    assert hash_canonical(_schema(candidate)) == full_sha256
    assert not list(Draft202012Validator(provider).iter_errors(payload))
    assert not list(Draft202012Validator(full_before).iter_errors(payload))

    schema_v1 = packet_generation_schema(
        candidate=candidate,
        exposed_line_ids=LINES,
        source_locator=LOCATOR,
        allowed_outcomes=[OUTCOME],
        allowed_moderators=[],
        allowed_sections=["Results"],
        outcome_positive_directions=DIRECTION,
    )
    assert _canonical_json_size(provider) < 0.9 * _canonical_json_size(schema_v1)
    assert _canonical_json_size(provider) < 0.35 * _canonical_json_size(full_before)

    forbidden = set(PROVIDER_GRAMMAR_SCOPE_V2["known_potentially_skipped_keywords"])
    assert _all_mapping_keys(provider).isdisjoint(forbidden)
    assert _all_schema_patterns(provider) == []


def test_compact_inventory_provider_schema_is_deterministic_small_and_dual_valid() -> None:
    context = {"exposed_line_ids": LINES, "allowed_outcomes": [OUTCOME]}
    full = inventory_generation_schema_v2(**context)
    provider = inventory_provider_schema_v2(**context)
    schema_v1 = inventory_generation_schema(**context)
    example = {
        "inventory_version": "native-candidate-inventory-v1",
        "inventory_status": "no_candidate_found",
        "candidates": [],
        "has_more_or_uncertain": False,
    }
    assert provider == inventory_provider_schema_v2(**context)
    assert provider["x-literature-multiverse-generation-schema-version"] == (
        INVENTORY_PROVIDER_SCHEMA_V2
    )
    assert provider["x-literature-multiverse-full-acceptance-schema-sha256"] == (
        hash_canonical(full)
    )
    assert not list(Draft202012Validator(provider).iter_errors(example))
    assert not list(Draft202012Validator(full).iter_errors(example))
    # Three complete closed branches cost slightly more than the flattened v1
    # object, but remain a tiny provider request and far smaller than full v2.
    assert _canonical_json_size(provider) < 2_500
    assert _canonical_json_size(provider) < 0.125 * _canonical_json_size(full)
    assert _canonical_json_size(schema_v1) < _canonical_json_size(provider)
    forbidden = set(PROVIDER_GRAMMAR_SCOPE_V2["known_potentially_skipped_keywords"])
    assert _all_mapping_keys(provider).isdisjoint(forbidden)
    assert _all_schema_patterns(provider) == []


def _synthetic_inventory_candidates(count: int) -> list[dict[str, Any]]:
    signatures = [
        (effect_kind, line_ids)
        for effect_kind in sorted(PACKET_MODELS)
        for line_ids in (["L10"], ["L20"], LINES)
    ]
    return [
        {
            "candidate_index": index,
            "outcome_name": OUTCOME,
            "effect_kind": effect_kind,
            "line_ids": line_ids,
        }
        for index, (effect_kind, line_ids) in enumerate(signatures[:count], start=1)
    ]


@pytest.mark.parametrize(
    ("inventory_status", "has_more_or_uncertain", "candidate_count"),
    [
        (status, has_more, count)
        for status in (
            "candidates_found",
            "no_candidate_found",
            "overflow_or_uncertain",
        )
        for has_more in (False, True)
        for count in range(10)
    ],
)
def test_inventory_provider_exhaustively_preserves_all_three_state_branches(
    inventory_status: str,
    has_more_or_uncertain: bool,
    candidate_count: int,
) -> None:
    payload = {
        "inventory_version": "native-candidate-inventory-v1",
        "inventory_status": inventory_status,
        "candidates": _synthetic_inventory_candidates(candidate_count),
        "has_more_or_uncertain": has_more_or_uncertain,
    }
    expected_valid = (
        (
            inventory_status == "candidates_found"
            and not has_more_or_uncertain
            and 1 <= candidate_count <= 8
        )
        or (
            inventory_status == "no_candidate_found"
            and not has_more_or_uncertain
            and candidate_count == 0
        )
        or (
            inventory_status == "overflow_or_uncertain"
            and has_more_or_uncertain
            and 0 <= candidate_count <= 9
        )
    )
    context = {"exposed_line_ids": LINES, "allowed_outcomes": [OUTCOME]}
    provider_errors = list(
        Draft202012Validator(inventory_provider_schema_v2(**context)).iter_errors(payload)
    )
    full_errors = list(
        Draft202012Validator(inventory_generation_schema_v2(**context)).iter_errors(payload)
    )
    assert (not provider_errors) is expected_valid
    assert (not full_errors) is expected_valid


def test_inventory_provider_state_branches_are_closed_bounded_and_llama_safe() -> None:
    provider = inventory_provider_schema_v2(exposed_line_ids=LINES, allowed_outcomes=[OUTCOME])
    branches = provider["oneOf"]
    assert len(branches) == 3
    observed = {
        (
            branch["properties"]["inventory_status"]["const"],
            branch["properties"]["has_more_or_uncertain"]["const"],
            branch["properties"]["candidates"].get("minItems", 0),
            branch["properties"]["candidates"]["maxItems"],
        )
        for branch in branches
    }
    assert observed == {
        ("candidates_found", False, 1, 8),
        ("no_candidate_found", False, 0, 0),
        ("overflow_or_uncertain", True, 0, 9),
    }
    assert all(branch["additionalProperties"] is False for branch in branches)
    assert _all_schema_patterns(provider) == []
    forbidden = set(PROVIDER_GRAMMAR_SCOPE_V2["known_potentially_skipped_keywords"])
    assert _all_mapping_keys(provider).isdisjoint(forbidden)
    assert PROVIDER_GRAMMAR_SCOPE_V2["inventory_provider_state_coherence"] == (
        "three_closed_oneOf_branches_preserve_empty_found_and_overflow_states"
    )
    assert (
        "inventory_count_specific_index_specialization_and_descriptor_signature_uniqueness"
        in (PROVIDER_GRAMMAR_SCOPE_V2["provider_only_simplifications"])
    )
    assert (
        "inventory_index_and_state_cross_field_coherence"
        not in (PROVIDER_GRAMMAR_SCOPE_V2["provider_only_simplifications"])
    )
    Draft202012Validator.check_schema(provider)


@pytest.mark.parametrize(
    ("effect_kind", "field_family", "invalid_lexeme"),
    [
        ("direct_standard_error", "effect.estimate", "not-a-decimal"),
        ("direct_standard_error", "effect.standard_error", "-0.2"),
        ("direct_standard_error", "effect.reported_p_value", "not-a-p-value"),
        ("direct_standard_error", "numeric_support.quote_start", "x"),
        ("direct_confidence_interval", "effect.ci_level", "not-a-percent"),
        ("binary_group_statistics", "effect.treatment_events", "x"),
    ],
)
def test_provider_pattern_omission_is_explicitly_wider_but_full_lexical_authority_rejects(
    effect_kind: str,
    field_family: str,
    invalid_lexeme: str,
) -> None:
    candidate, payload = _completed_effect_payload(effect_kind)
    if field_family == "numeric_support.quote_start":
        payload["numeric_support"][0]["quote_start"] = invalid_lexeme
    else:
        parent, field_name = field_family.split(".")
        payload[parent][field_name] = invalid_lexeme

    assert not list(Draft202012Validator(_provider_schema(candidate)).iter_errors(payload))
    assert list(Draft202012Validator(_schema(candidate)).iter_errors(payload))


def test_provider_pattern_omission_widens_timepoint_lexeme_only_before_full_validation() -> None:
    candidate, payload = _valid_packet(BoundedTimepoint(kind="exact", value="8", unit="week"))
    payload["finding"]["timepoint"]["value"] = "not-a-timepoint-number"

    assert not list(Draft202012Validator(_provider_schema(candidate)).iter_errors(payload))
    assert list(Draft202012Validator(_schema(candidate)).iter_errors(payload))


def test_provider_schema_keeps_exact_candidate_context_but_has_zero_acceptance_authority() -> None:
    candidate, payload = _completed_effect_payload("direct_standard_error")
    provider = _provider_schema(candidate)
    mutations = [
        ("candidate_index", 2),
        ("finding.outcome_name", "different_outcome"),
        ("effect.effect_kind", "direct_variance"),
        ("evidence.source_locator", "synthetic:different-source"),
        ("evidence.section", "Methods"),
        ("evidence.line_ids", list(reversed(LINES))),
        ("contrast.positive_direction_means", "opposite direction"),
    ]
    validator = Draft202012Validator(provider)
    for dotted_path, replacement in mutations:
        attack = deepcopy(payload)
        target = attack
        parts = dotted_path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = replacement
        assert list(validator.iter_errors(attack)), dotted_path

    intentionally_wider = deepcopy(payload)
    intentionally_wider["finding"]["timepoint"]["kind"] = "exact"
    assert not list(validator.iter_errors(intentionally_wider))
    assert list(Draft202012Validator(_schema(candidate)).iter_errors(intentionally_wider))
    with pytest.raises(NativeBoundedGenerationError, match="marker_missing_or_unknown"):
        validate_raw_payload_against_schema_v2(payload, schema=provider)


def test_schema_bundles_bind_both_hashes_and_fail_closed_on_tampering() -> None:
    candidate = _candidate()
    bundle = packet_schema_bundle_v2(
        candidate=candidate,
        exposed_line_ids=LINES,
        source_locator=LOCATOR,
        allowed_outcomes=[OUTCOME],
        allowed_moderators=[],
        allowed_sections=["Results"],
        outcome_positive_directions=DIRECTION,
    )
    assert bundle == packet_schema_bundle_v2(
        candidate=candidate,
        exposed_line_ids=LINES,
        source_locator=LOCATOR,
        allowed_outcomes=[OUTCOME],
        allowed_moderators=[],
        allowed_sections=["Results"],
        outcome_positive_directions=DIRECTION,
    )
    binding = schema_bundle_receipt_binding_v2(bundle)
    assert binding["schema_bundle_version"] == SCHEMA_BUNDLE_V2
    assert binding["provider_schema_sha256"] == hash_canonical(bundle["provider_schema"])
    assert binding["full_acceptance_schema_sha256"] == hash_canonical(
        bundle["full_acceptance_schema"]
    )
    assert binding["provider_schema_scientific_authority"] == "none"

    inventory_bundle = inventory_schema_bundle_v2(
        exposed_line_ids=LINES,
        allowed_outcomes=[OUTCOME],
    )
    assert schema_bundle_receipt_binding_v2(inventory_bundle)["kind"] == "inventory"

    tampered = deepcopy(bundle)
    tampered["provider_schema"]["$id"] += ":tampered"
    with pytest.raises(NativeBoundedGenerationError, match="bundle_hash_mismatch"):
        schema_bundle_receipt_binding_v2(tampered)


def test_prechange_pattern_bound_provider_bundle_and_contract_are_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_unsupported = set(schema_v2_module._PROVIDER_UNSUPPORTED_SCHEMA_KEYWORDS)
    assert "pattern" in current_unsupported
    monkeypatch.setattr(
        schema_v2_module,
        "_PROVIDER_UNSUPPORTED_SCHEMA_KEYWORDS",
        current_unsupported - {"pattern"},
    )
    stale_bundle = packet_schema_bundle_v2(
        candidate=NativeCandidateDescriptor(
            candidate_index=1,
            outcome_name="synthetic_outcome",
            effect_kind="binary_group_statistics",
            line_ids=["SYNTHETIC_LINE"],
        ),
        exposed_line_ids=["SYNTHETIC_LINE"],
        source_locator="synthetic:schema-compilation-only",
        allowed_outcomes=["synthetic_outcome"],
        allowed_moderators=[],
        allowed_sections=["Synthetic"],
        outcome_positive_directions={"synthetic_outcome": "larger synthetic target value"},
    )
    assert stale_bundle["provider_schema_sha256"] == (
        "d5150b381d5818f83715d774776dcc91d081a4928e61f85f8cf4da2fd1cd3598"
    )
    assert stale_bundle["full_acceptance_schema_sha256"] == (
        "4edc1fa0543d53f30489c301c27e771c227572698a71dd46fc962a5ecb5d7ae3"
    )
    assert stale_bundle["schema_bundle_sha256"] == (
        "66b6bcc7992426c11883091c218696052c7bb4a53eeae7d369c58f6e773b5435"
    )

    monkeypatch.setattr(
        schema_v2_module,
        "_PROVIDER_UNSUPPORTED_SCHEMA_KEYWORDS",
        current_unsupported,
    )
    current_bundle = packet_schema_bundle_v2(
        candidate=NativeCandidateDescriptor.model_validate(
            stale_bundle["context_binding"]["candidate"]
        ),
        exposed_line_ids=stale_bundle["context_binding"]["exposed_line_ids"],
        source_locator=stale_bundle["context_binding"]["source_locator"],
        allowed_outcomes=stale_bundle["context_binding"]["allowed_outcomes"],
        allowed_moderators=stale_bundle["context_binding"]["allowed_moderators"],
        allowed_sections=stale_bundle["context_binding"]["allowed_sections"],
        outcome_positive_directions=stale_bundle["context_binding"]["outcome_positive_directions"],
    )
    assert (
        current_bundle["full_acceptance_schema_sha256"]
        == (stale_bundle["full_acceptance_schema_sha256"])
    )
    assert current_bundle["provider_schema_sha256"] != (stale_bundle["provider_schema_sha256"])
    assert current_bundle["schema_bundle_sha256"] != stale_bundle["schema_bundle_sha256"]
    with pytest.raises(NativeBoundedGenerationError, match="bundle_replay_mismatch"):
        schema_bundle_receipt_binding_v2(stale_bundle)

    assert schema_v2_contract()["contract_sha256"] != (
        "1269114f975b42bb44d6f78e37c592c2a69957de6fbee47225a99b2c8ec83a5f"
    )
    assert synthetic_schema_v2_preflight_fingerprint() != (
        "5757c7a2bc9373ac3a96f1e78d8f6e9f84045fa34677f156fc64199eb53ac809"
    )


def _recompute_outer_bundle_hash(bundle: dict[str, Any]) -> None:
    payload = {key: value for key, value in bundle.items() if key != "schema_bundle_sha256"}
    bundle["schema_bundle_sha256"] = hash_canonical(payload)


def test_inventory_bundle_replay_rejects_pre_hardening_flattened_state_schema() -> None:
    bundle = inventory_schema_bundle_v2(exposed_line_ids=LINES, allowed_outcomes=[OUTCOME])
    stale = deepcopy(bundle)
    provider = stale["provider_schema"]
    flattened = deepcopy(provider["oneOf"][0])
    flattened["properties"]["inventory_status"] = {
        "enum": [
            "candidates_found",
            "no_candidate_found",
            "overflow_or_uncertain",
        ]
    }
    flattened["properties"]["has_more_or_uncertain"] = {"type": "boolean"}
    flattened["properties"]["candidates"].pop("minItems", None)
    flattened["properties"]["candidates"]["maxItems"] = 9
    provider.pop("oneOf")
    provider.update(flattened)
    Draft202012Validator.check_schema(provider)
    contradictory = {
        "inventory_version": "native-candidate-inventory-v1",
        "inventory_status": "overflow_or_uncertain",
        "candidates": _synthetic_inventory_candidates(1),
        "has_more_or_uncertain": False,
    }
    assert not list(Draft202012Validator(provider).iter_errors(contradictory))
    assert list(Draft202012Validator(bundle["provider_schema"]).iter_errors(contradictory))
    stale["provider_schema_sha256"] = hash_canonical(provider)
    _recompute_outer_bundle_hash(stale)
    with pytest.raises(NativeBoundedGenerationError, match="bundle_replay_mismatch"):
        schema_bundle_receipt_binding_v2(stale)


def test_schema_bundle_rejects_self_consistent_red_team_forgery() -> None:
    candidate = _candidate()
    forged = packet_schema_bundle_v2(
        candidate=candidate,
        exposed_line_ids=LINES,
        source_locator=LOCATOR,
        allowed_outcomes=[OUTCOME],
        allowed_moderators=[],
        allowed_sections=["Results"],
        outcome_positive_directions=DIRECTION,
    )
    full_sha256 = forged["full_acceptance_schema_sha256"]
    forged["kind"] = "evil"
    forged["schema_sent_to_provider"] = "full_acceptance_schema"
    forged["raw_response_validation_schema"] = "provider_schema"
    forged["provider_schema_version"] = "evil"
    forged["provider_schema"] = {
        "x-literature-multiverse-full-acceptance-schema-sha256": full_sha256
    }
    forged["provider_schema_sha256"] = hash_canonical(forged["provider_schema"])
    _recompute_outer_bundle_hash(forged)
    with pytest.raises(NativeBoundedGenerationError, match="bundle_kind_mismatch"):
        schema_bundle_receipt_binding_v2(forged)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("schema_sent_to_provider", "full_acceptance_schema"),
        ("raw_response_validation_schema", "provider_schema"),
        ("provider_schema_scientific_authority", "acceptance"),
        ("provider_schema_version", "wrong-provider-version"),
        ("full_acceptance_schema_version", "wrong-full-version"),
    ],
)
def test_schema_bundle_rejects_rehashed_fixed_field_tampering(
    field_name: str, replacement: str
) -> None:
    bundle = inventory_schema_bundle_v2(
        exposed_line_ids=LINES,
        allowed_outcomes=[OUTCOME],
    )
    bundle[field_name] = replacement
    _recompute_outer_bundle_hash(bundle)
    with pytest.raises(NativeBoundedGenerationError, match="fixed_field_mismatch"):
        schema_bundle_receipt_binding_v2(bundle)


def test_schema_bundle_rejects_rehashed_context_schema_and_dual_hash_tampering() -> None:
    bundle = packet_schema_bundle_v2(
        candidate=_candidate(),
        exposed_line_ids=LINES,
        source_locator=LOCATOR,
        allowed_outcomes=[OUTCOME],
        allowed_moderators=[],
        allowed_sections=["Results"],
        outcome_positive_directions=DIRECTION,
    )

    context_attack = deepcopy(bundle)
    context_attack["context_binding"]["source_locator"] = "synthetic:different-source"
    context_attack["context_binding_sha256"] = hash_canonical(context_attack["context_binding"])
    _recompute_outer_bundle_hash(context_attack)
    with pytest.raises(NativeBoundedGenerationError, match="bundle_replay_mismatch"):
        schema_bundle_receipt_binding_v2(context_attack)

    schema_attack = deepcopy(bundle)
    schema_attack["provider_schema"]["$id"] += ":forged"
    schema_attack["provider_schema_sha256"] = hash_canonical(schema_attack["provider_schema"])
    _recompute_outer_bundle_hash(schema_attack)
    with pytest.raises(NativeBoundedGenerationError, match="provider_id_mismatch"):
        schema_bundle_receipt_binding_v2(schema_attack)

    dual_hash_attack = deepcopy(bundle)
    (
        dual_hash_attack["full_acceptance_schema_sha256"],
        dual_hash_attack["provider_schema_sha256"],
    ) = (
        dual_hash_attack["provider_schema_sha256"],
        dual_hash_attack["full_acceptance_schema_sha256"],
    )
    _recompute_outer_bundle_hash(dual_hash_attack)
    with pytest.raises(NativeBoundedGenerationError, match="full_hash_mismatch"):
        schema_bundle_receipt_binding_v2(dual_hash_attack)


@pytest.mark.parametrize("effect_kind", sorted(PACKET_MODELS))
def test_every_completed_effect_kind_passes_v1_authority_and_v2_schema(
    effect_kind: str,
) -> None:
    candidate, payload = _completed_effect_payload(effect_kind)
    _validate_v1_authority(candidate, payload)
    assert not list(Draft202012Validator(_schema(candidate)).iter_errors(payload))


@pytest.mark.parametrize(
    "timepoint",
    [
        BoundedTimepoint(kind="exact", value="8", unit="week"),
        BoundedTimepoint(kind="range", lower="8", upper="12", unit="week"),
        BoundedTimepoint(kind="reported_text", raw_label="end of study"),
        BoundedTimepoint(kind="not_reported"),
    ],
)
def test_completed_valid_v1_fixtures_pass_all_v2_timepoint_branches(
    timepoint: BoundedTimepoint,
) -> None:
    candidate, payload = _valid_packet(timepoint)
    _validate_v1_authority(candidate, payload)
    schema = _schema(candidate)
    assert not list(Draft202012Validator(schema).iter_errors(payload))


def test_v2_rejects_observed_structural_failure_classes_before_pydantic() -> None:
    candidate, valid = _valid_packet(BoundedTimepoint(kind="exact", value="8", unit="week"))
    validator = Draft202012Validator(_schema(candidate))

    attacks: list[dict[str, Any]] = []
    missing_discriminator = deepcopy(valid)
    missing_discriminator.pop("packet_status")
    attacks.append(missing_discriminator)

    timepoint_with_unit_in_value = deepcopy(valid)
    timepoint_with_unit_in_value["finding"]["timepoint"]["value"] = "8 weeks"
    attacks.append(timepoint_with_unit_in_value)

    coerced_number = deepcopy(valid)
    coerced_number["effect"]["estimate"] = 0.5
    attacks.append(coerced_number)

    malformed_offset = deepcopy(valid)
    malformed_offset["numeric_support"][0]["quote_start"] = "offset 10"
    attacks.append(malformed_offset)

    reordered_or_wrong_lines = deepcopy(valid)
    reordered_or_wrong_lines["evidence"]["line_ids"] = list(reversed(LINES))
    attacks.append(reordered_or_wrong_lines)

    wrong_effect_support = deepcopy(valid)
    wrong_effect_support["numeric_support"][0]["field_path"] = "effect.variance"
    attacks.append(wrong_effect_support)

    missing_required_effect_support = deepcopy(valid)
    missing_required_effect_support["numeric_support"] = [
        item
        for item in missing_required_effect_support["numeric_support"]
        if item["field_path"] != "effect.standard_error"
    ]
    attacks.append(missing_required_effect_support)

    assert attacks
    assert all(list(validator.iter_errors(attack)) for attack in attacks)


@pytest.mark.parametrize(
    ("effect_kind", "field_name"),
    [
        ("direct_standard_error", "standard_error"),
        ("direct_variance", "variance"),
        ("continuous_group_statistics", "treatment_sd"),
        ("continuous_group_statistics", "control_sd"),
    ],
)
def test_positive_only_effect_fields_reject_negative_lexemes(
    effect_kind: str, field_name: str
) -> None:
    candidate, payload = _completed_effect_payload(effect_kind)
    payload["effect"][field_name] = "-0.2"
    assert list(Draft202012Validator(_schema(candidate)).iter_errors(payload))


@pytest.mark.parametrize("field_name", ["equivalence_margin", "reported_p_value"])
def test_nonnegative_optional_effect_fields_reject_negative_lexemes(
    field_name: str,
) -> None:
    candidate, payload = _completed_effect_payload("direct_standard_error")
    payload["effect"][field_name] = "-0.1"
    assert list(Draft202012Validator(_schema(candidate)).iter_errors(payload))


def test_exact_and_range_required_timepoint_values_reject_null() -> None:
    exact_candidate, exact = _valid_packet(BoundedTimepoint(kind="exact", value="8", unit="week"))
    exact["finding"]["timepoint"]["value"] = None
    assert list(Draft202012Validator(_schema(exact_candidate)).iter_errors(exact))

    range_candidate, ranged = _valid_packet(
        BoundedTimepoint(kind="range", lower="8", upper="12", unit="week")
    )
    for field_name in ("lower", "upper"):
        attack = deepcopy(ranged)
        attack["finding"]["timepoint"][field_name] = None
        assert list(Draft202012Validator(_schema(range_candidate)).iter_errors(attack))


def _schema_patterns(value: Any) -> list[str]:
    output: list[str] = []
    if isinstance(value, dict):
        if isinstance(value.get("pattern"), str):
            output.append(value["pattern"])
        for item in value.values():
            output.extend(_schema_patterns(item))
    elif isinstance(value, list):
        for item in value:
            output.extend(_schema_patterns(item))
    return output


def test_every_v2_pattern_has_provider_anchor_shape_and_absolute_draft_end() -> None:
    schemas = [inventory_generation_schema_v2(exposed_line_ids=LINES, allowed_outcomes=[OUTCOME])]
    schemas.extend(_schema(_candidate(effect_kind=kind)) for kind in PACKET_MODELS)
    patterns = [pattern for schema in schemas for pattern in _schema_patterns(schema)]
    assert patterns
    assert all(pattern.startswith("^") and pattern.endswith("$") for pattern in patterns)
    assert all(r"(?![\s\S])$" in pattern for pattern in patterns)


@pytest.mark.parametrize(
    "invalid_lexeme",
    [
        " 0.5",
        "0.5 ",
        "\t0.5",
        "0.5\t",
        "0.5\n",
        "0.5\r",
        "0.5\r\n",
        "0.5\u2028",
        "0.5\u2029",
    ],
)
def test_raw_decimal_schema_rejects_whitespace_and_line_terminators_before_pydantic(
    invalid_lexeme: str,
) -> None:
    candidate, payload = _completed_effect_payload("direct_standard_error")
    schema = _schema(candidate)
    payload["effect"]["estimate"] = invalid_lexeme
    with pytest.raises(NativeBoundedGenerationError, match="raw_validation_error"):
        validate_raw_payload_against_schema_v2(payload, schema=schema)


@pytest.mark.parametrize("suffix", [" ", "\t", "\n", "\r", "\r\n", "\u2028"])
def test_raw_count_offset_and_percent_families_reject_trailing_whitespace(
    suffix: str,
) -> None:
    count_candidate, count_payload = _completed_effect_payload("continuous_group_statistics")
    count_payload["effect"]["treatment_n"] += suffix
    with pytest.raises(NativeBoundedGenerationError, match="raw_validation_error"):
        validate_raw_payload_against_schema_v2(count_payload, schema=_schema(count_candidate))

    offset_candidate, offset_payload = _completed_effect_payload("direct_standard_error")
    offset_payload["numeric_support"][0]["quote_start"] += suffix
    with pytest.raises(NativeBoundedGenerationError, match="raw_validation_error"):
        validate_raw_payload_against_schema_v2(offset_payload, schema=_schema(offset_candidate))

    percent_candidate, percent_payload = _completed_effect_payload("direct_confidence_interval")
    ci_level_support = next(
        item
        for item in percent_payload["numeric_support"]
        if item["field_path"] == "effect.ci_level"
    )
    ci_level_support["normalization"] = "percent_to_proportion"
    ci_level_support["verbatim_token"] = "95%" + suffix
    with pytest.raises(NativeBoundedGenerationError, match="raw_validation_error"):
        validate_raw_payload_against_schema_v2(percent_payload, schema=_schema(percent_candidate))


def test_raw_validator_returns_an_unchanged_copy_without_string_normalization() -> None:
    candidate, payload = _completed_effect_payload("direct_standard_error")
    validated = validate_raw_payload_against_schema_v2(payload, schema=_schema(candidate))
    assert validated == payload
    assert validated is not payload


def test_raw_v2_api_rejects_v1_arbitrary_and_marker_tampered_schemas() -> None:
    candidate, payload = _completed_effect_payload("direct_standard_error")
    schema_v1 = packet_generation_schema(
        candidate=candidate,
        exposed_line_ids=LINES,
        source_locator=LOCATOR,
        allowed_outcomes=[OUTCOME],
        allowed_moderators=[],
        allowed_sections=["Results"],
        outcome_positive_directions=DIRECTION,
    )
    with pytest.raises(NativeBoundedGenerationError, match="marker_missing_or_unknown"):
        validate_raw_payload_against_schema_v2(payload, schema=schema_v1)
    with pytest.raises(NativeBoundedGenerationError, match="marker_missing_or_unknown"):
        validate_raw_payload_against_schema_v2(payload, schema={"type": "object"})

    schema_v2 = _schema(candidate)
    wrong_draft = deepcopy(schema_v2)
    wrong_draft["$schema"] = "https://json-schema.org/draft/2019-09/schema"
    with pytest.raises(NativeBoundedGenerationError, match="draft_marker_mismatch"):
        validate_raw_payload_against_schema_v2(payload, schema=wrong_draft)

    wrong_id = deepcopy(schema_v2)
    wrong_id["$id"] = "urn:literature-multiverse:stale-schema"
    with pytest.raises(NativeBoundedGenerationError, match="id_mismatch"):
        validate_raw_payload_against_schema_v2(payload, schema=wrong_id)


@pytest.mark.parametrize(
    ("dotted_path", "replacement", "allowed_moderators"),
    [
        ("study.source_label", " Synthetic study ", []),
        ("effect.unit", " synthetic-unit ", []),
        ("evidence.quote", None, []),
        ("effect.moderators", [{"name": "age", "value": " adult "}], ["age"]),
    ],
)
def test_v2_integration_rejects_pydantic_free_text_normalization(
    dotted_path: str,
    replacement: Any,
    allowed_moderators: list[str],
) -> None:
    candidate, payload = _completed_effect_payload("direct_standard_error")
    if dotted_path == "evidence.quote":
        replacement = payload["evidence"]["quote"] + " "
    target = payload
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = replacement
    schema = packet_generation_schema_v2(
        candidate=candidate,
        exposed_line_ids=LINES,
        source_locator=LOCATOR,
        allowed_outcomes=[OUTCOME],
        allowed_moderators=allowed_moderators,
        allowed_sections=["Results"],
        outcome_positive_directions=DIRECTION,
    )
    assert not list(Draft202012Validator(schema).iter_errors(payload))
    with pytest.raises(NativeBoundedGenerationError, match="raw_canonical_value_changed"):
        validate_packet_for_candidate_v2(
            payload,
            candidate=candidate,
            exposed_line_ids=LINES,
            source_locator=LOCATOR,
            allowed_outcomes=[OUTCOME],
            allowed_moderators=allowed_moderators,
            allowed_sections=["Results"],
            outcome_positive_directions=DIRECTION,
        )


def test_v1_schema_upgrade_fails_closed_on_every_candidate_binding_mismatch() -> None:
    candidate = _candidate()
    schema_v1 = packet_generation_schema(
        candidate=candidate,
        exposed_line_ids=LINES,
        source_locator=LOCATOR,
        allowed_outcomes=[OUTCOME],
        allowed_moderators=[],
        allowed_sections=["Results"],
        outcome_positive_directions=DIRECTION,
    )
    mismatches = [
        NativeCandidateDescriptor(
            candidate_index=2,
            outcome_name=OUTCOME,
            effect_kind="direct_standard_error",
            line_ids=LINES,
        ),
        NativeCandidateDescriptor(
            candidate_index=1,
            outcome_name="different_outcome",
            effect_kind="direct_standard_error",
            line_ids=LINES,
        ),
        NativeCandidateDescriptor(
            candidate_index=1,
            outcome_name=OUTCOME,
            effect_kind="direct_standard_error",
            line_ids=["L10"],
        ),
        NativeCandidateDescriptor(
            candidate_index=1,
            outcome_name=OUTCOME,
            effect_kind="direct_variance",
            line_ids=LINES,
        ),
    ]
    for mismatch in mismatches:
        with pytest.raises(NativeBoundedGenerationError, match="binding_mismatch"):
            upgrade_packet_schema_v1_to_v2(schema_v1, candidate=mismatch)


def test_candidate_context_and_unable_branch_are_exactly_bound() -> None:
    candidate, payload = _completed_effect_payload("direct_standard_error")
    validator = Draft202012Validator(_schema(candidate))
    mutations = [
        ("candidate_index", 2),
        ("finding.outcome_name", "different_outcome"),
        ("effect.effect_kind", "direct_variance"),
        ("evidence.source_locator", "synthetic:different-source"),
        ("evidence.section", "Methods"),
        ("evidence.line_ids", ["L20", "L10"]),
        ("contrast.positive_direction_means", "opposite direction"),
    ]
    for dotted_path, replacement in mutations:
        attack = deepcopy(payload)
        target = attack
        parts = dotted_path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = replacement
        assert list(validator.iter_errors(attack)), dotted_path

    unable = {
        "packet_version": "native-candidate-packet-v1",
        "packet_status": "unable_to_complete",
        "candidate_index": 1,
        "reason": "capacity_or_other_uncertainty",
    }
    assert not list(validator.iter_errors(unable))
    for field_name, replacement in (
        ("packet_version", "wrong-version"),
        ("packet_status", "completed"),
        ("candidate_index", 2),
    ):
        attack = deepcopy(unable)
        attack[field_name] = replacement
        assert list(validator.iter_errors(attack))
    unable_with_scientific_field = {**unable, "effect": payload["effect"]}
    assert list(validator.iter_errors(unable_with_scientific_field))


def test_cross_field_scientific_invariants_deliberately_remain_postvalidation() -> None:
    candidate, valid = _valid_packet()
    schema_validator = Draft202012Validator(_schema(candidate))

    same_arm_key = deepcopy(valid)
    same_arm_key["comparator_arm"]["key"] = same_arm_key["treatment_arm"]["key"]
    assert not list(schema_validator.iter_errors(same_arm_key))
    with pytest.raises(ValueError, match="bounded_packet_arms_not_distinct"):
        _validate_v1_authority(candidate, same_arm_key)

    unsupported_emitted_value = deepcopy(valid)
    unsupported_emitted_value["effect"]["estimate"] = "0.6"
    assert not list(schema_validator.iter_errors(unsupported_emitted_value))
    with pytest.raises(ValueError, match="numeric_support_value_mismatch"):
        _validate_v1_authority(candidate, unsupported_emitted_value)

    range_candidate, range_payload = _valid_packet(
        BoundedTimepoint(kind="range", lower="8", upper="12", unit="week")
    )
    reversed_range = deepcopy(range_payload)
    reversed_range["finding"]["timepoint"]["lower"] = "12"
    reversed_range["finding"]["timepoint"]["upper"] = "8"
    assert not list(Draft202012Validator(_schema(range_candidate)).iter_errors(reversed_range))
    with pytest.raises(ValueError, match="timepoint_range_not_ordered"):
        _validate_v1_authority(range_candidate, reversed_range)


def test_inventory_v2_enforces_version_and_uniqueness_but_not_string_sorting() -> None:
    schema_v1 = inventory_generation_schema(exposed_line_ids=LINES, allowed_outcomes=[OUTCOME])
    schema_v2 = inventory_generation_schema_v2(exposed_line_ids=LINES, allowed_outcomes=[OUTCOME])
    assert schema_v1 != schema_v2
    payload = {
        "inventory_version": "native-candidate-inventory-v1",
        "inventory_status": "candidates_found",
        "candidates": [
            {
                "candidate_index": 1,
                "outcome_name": OUTCOME,
                "effect_kind": "direct_standard_error",
                "line_ids": LINES,
            }
        ],
        "has_more_or_uncertain": False,
    }
    validator = Draft202012Validator(schema_v2)
    assert not list(validator.iter_errors(payload))

    duplicate = deepcopy(payload)
    duplicate["candidates"][0]["line_ids"] = ["L10", "L10"]
    assert list(validator.iter_errors(duplicate))

    noncontiguous = deepcopy(payload)
    noncontiguous["candidates"][0]["candidate_index"] = 2
    assert list(validator.iter_errors(noncontiguous))

    empty_found = deepcopy(payload)
    empty_found["candidates"] = []
    assert list(validator.iter_errors(empty_found))

    populated_empty_state = deepcopy(payload)
    populated_empty_state["inventory_status"] = "no_candidate_found"
    assert list(validator.iter_errors(populated_empty_state))

    false_overflow = deepcopy(payload)
    false_overflow["inventory_status"] = "overflow_or_uncertain"
    assert list(validator.iter_errors(false_overflow))

    unsorted = deepcopy(payload)
    unsorted["candidates"][0]["line_ids"] = list(reversed(LINES))
    assert not list(validator.iter_errors(unsorted))
    with pytest.raises(ValueError, match="line_ids_not_sorted_unique"):
        validate_inventory_for_row(unsorted, exposed_line_ids=LINES, allowed_outcomes=[OUTCOME])

    duplicate_signature = deepcopy(payload)
    duplicate_signature["candidates"].append(
        {
            **deepcopy(duplicate_signature["candidates"][0]),
            "candidate_index": 2,
        }
    )
    assert not list(validator.iter_errors(duplicate_signature))
    with pytest.raises(ValueError, match="descriptor_signature_duplicate"):
        validate_inventory_for_row(
            duplicate_signature,
            exposed_line_ids=LINES,
            allowed_outcomes=[OUTCOME],
        )


def test_validator_coverage_roster_matches_every_reachable_custom_validator() -> None:
    source_path = (
        Path(__file__).parents[1] / "src" / "literature_multiverse" / "native_bounded_generation.py"
    )
    tree = ast.parse(source_path.read_text())
    reachable_classes = {
        "BoundedStudyHeader",
        "BoundedCohortHeader",
        "BoundedArm",
        "BoundedTimepoint",
        "BoundedEvidence",
        "BoundedEffectCommon",
        "DirectStandardErrorEffect",
        "DirectVarianceEffect",
        "DirectConfidenceIntervalEffect",
        "ContinuousGroupEffect",
        "BinaryGroupEffect",
        "BoundedNumericSupport",
        "NativeCandidatePacket",
    }
    discovered: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in reachable_classes:
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorator_names = {
                decorator.func.id
                for decorator in item.decorator_list
                if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name)
            }
            if decorator_names.intersection({"field_validator", "model_validator"}):
                discovered.add(f"{node.name}.{item.name}")
    discovered.add("_validate_numeric_support")
    declared = {item["validator"] for item in PACKET_VALIDATOR_COVERAGE_V2}
    assert declared == discovered
    assert {item["coverage"] for item in PACKET_VALIDATOR_COVERAGE_V2}.issubset(
        {"enforced", "partial", "postvalidation_only"}
    )
    assert {item["validator"] for item in INVENTORY_VALIDATOR_COVERAGE_V2} == {
        "NativeCandidateDescriptor.validate_line_ids",
        "NativeCandidateInventory.validate_inventory",
    }
    assert "native_bounded_schema_v2" not in source_path.read_text()


def test_aggregate_saved_output_audit_never_returns_content(tmp_path: Path) -> None:
    candidate, valid = _valid_packet(BoundedTimepoint(kind="exact", value="8", unit="week"))
    malformed = deepcopy(valid)
    malformed["finding"]["timepoint"]["value"] = "eight weeks"
    malformed["numeric_support"][0]["quote_start"] = "beginning"
    malformed.pop("packet_status")
    malformed.pop("packet_version")
    malformed["effect"].pop("effect_kind")
    schema_v1 = packet_generation_schema(
        candidate=candidate,
        exposed_line_ids=LINES,
        source_locator=LOCATOR,
        allowed_outcomes=[OUTCOME],
        allowed_moderators=[],
        allowed_sections=["Results"],
        outcome_positive_directions=DIRECTION,
    )
    receipt = {
        "candidate": candidate.model_dump(mode="json"),
        "schema": schema_v1,
        "parsed_output": malformed,
        "parsed_output_sha256": hash_canonical(malformed),
    }
    receipt_path = tmp_path / "packet-receipts" / "row" / "01.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt))

    report = audit_saved_v1_packet_receipts(tmp_path)
    assert report["packet_count"] == 1
    assert report["invalid_after_administrative_envelope_count"] == 1
    assert report["all_saved_outputs_rejected"] is True
    assert report["contains_source_or_response_content"] is False
    assert report["contains_scientific_values"] is False
    assert report["authorizes_v1_output_repair_or_promotion"] is False

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    report_keys = keys(report)
    for forbidden in ("response_text", "parsed_output", "evidence", "quote", "row_key"):
        assert forbidden not in report_keys


_FROZEN_ANTIOX_WORKSPACE = (
    Path(__file__).parents[1] / "data" / "cache" / "native-antiox-bounded-v1-final-v1"
)


@pytest.mark.skipif(
    not _FROZEN_ANTIOX_WORKSPACE.exists(),
    reason="ignored frozen Antiox v1 diagnostic workspace is not present",
)
def test_all_33_frozen_antiox_v1_packets_fail_v2_for_explicit_schema_reasons() -> None:
    report = audit_saved_v1_packet_receipts(_FROZEN_ANTIOX_WORKSPACE)
    assert report["packet_count"] == 33
    assert report["invalid_after_administrative_envelope_count"] == 33
    assert report["valid_after_administrative_envelope_count"] == 0
    assert report["all_saved_outputs_rejected"] is True
    reasons = report["schema_failure_reason_counts"]
    assert reasons["pattern:numeric_support.*.quote_start"] == 33
    assert reasons["pattern:numeric_support.*.quote_end"] == 33
    assert (
        reasons["pattern:finding.timepoint.value"] + reasons["type:finding.timepoint.value"] == 33
    )
    assert reasons["pattern:effect.estimate"] >= 8
    assert report["contains_source_or_response_content"] is False
    assert report["contains_scientific_values"] is False
    assert report["authorizes_v1_output_repair_or_promotion"] is False
