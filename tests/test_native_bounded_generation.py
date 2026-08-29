from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.native_bounded_generation import (
    INVENTORY_SENTINEL_CAP,
    PACKET_MODELS,
    BoundedArm,
    BoundedCohortHeader,
    BoundedContrast,
    BoundedEvidence,
    BoundedFindingHeader,
    BoundedNumericSupport,
    BoundedStudyHeader,
    BoundedTimepoint,
    ContinuousGroupEffect,
    DirectConfidenceIntervalEffect,
    DirectStandardErrorEffect,
    NativeBoundedGenerationError,
    NativeCandidateDescriptor,
    NativeCandidateInventory,
    NativeCandidatePacket,
    NativeCandidateUnableToComplete,
    assemble_candidate_packets,
    assert_bounded_generation_schema,
    inventory_generation_schema,
    packet_generation_schema,
    validate_inventory_for_row,
    validate_packet_for_candidate,
)
from literature_multiverse.native_extraction import (
    native_publication_extraction_json_schema,
)

OUTCOMES = ["aerobic_capacity", "muscle_strength"]
LINES = ["L10", "L20", "L30", "L40"]
LOCATOR = "json:data/cache/source.json#/document"
POSITIVE_DIRECTIONS = {
    "aerobic_capacity": "larger beneficial adaptation with supplement",
    "muscle_strength": "larger beneficial adaptation with supplement",
}


def _candidate(
    index: int,
    *,
    outcome: str = "aerobic_capacity",
    line_id: str | None = None,
) -> NativeCandidateDescriptor:
    return NativeCandidateDescriptor(
        candidate_index=index,
        outcome_name=outcome,
        effect_kind="direct_standard_error",
        line_ids=[line_id or LINES[index - 1]],
    )


def _inventory(count: int) -> NativeCandidateInventory:
    return NativeCandidateInventory(
        inventory_status="candidates_found",
        candidates=[
            _candidate(
                index,
                outcome="aerobic_capacity" if index % 2 else "muscle_strength",
                line_id=f"L{index * 10}",
            )
            for index in range(1, count + 1)
        ],
        has_more_or_uncertain=False,
    )


def _packet(
    candidate: NativeCandidateDescriptor,
    *,
    study_key: str = "study-1",
    cohort_key: str = "cohort-1",
    treatment_label: str = "Supplement",
) -> NativeCandidatePacket[DirectStandardErrorEffect]:
    quote = "The adjusted difference was 0.5 (SE 0.2)."
    return NativeCandidatePacket[DirectStandardErrorEffect](
        candidate_index=candidate.candidate_index,
        study=BoundedStudyHeader(
            key=study_key,
            source_label=f"Study {study_key}",
            design="parallel controlled trial",
            registration_ids=[],
        ),
        cohort=BoundedCohortHeader(
            key=cohort_key,
            source_labels=[f"Cohort {cohort_key}"],
            registry_ids=[],
            dataset_ids=[],
            population_description=None,
            recruitment_period=None,
            total_sample_size=None,
        ),
        treatment_arm=BoundedArm(
            key="supplement",
            label=treatment_label,
            role="intervention",
            sample_size=None,
        ),
        comparator_arm=BoundedArm(
            key="control",
            label="Placebo",
            role="control",
            sample_size=None,
        ),
        contrast=BoundedContrast(
            key="target",
            label="supplement_vs_placebo",
            estimand="between-group difference in training adaptation",
            positive_direction_means="larger beneficial adaptation with supplement",
        ),
        finding=BoundedFindingHeader(
            key=f"finding-{candidate.candidate_index}",
            outcome_name=candidate.outcome_name,
            timepoint=BoundedTimepoint(
                kind="not_reported",
            ),
            analysis_population=None,
        ),
        effect=DirectStandardErrorEffect(
            effect_format="mean_difference",
            estimate="0.5",
            standard_error="0.2",
            unit="mL/kg/min",
        ),
        evidence=BoundedEvidence(
            source_locator=LOCATOR,
            quote=quote,
            section="Results",
            line_ids=candidate.line_ids,
        ),
        numeric_support=[
            BoundedNumericSupport(
                field_path="effect.estimate",
                verbatim_token="0.5",
                quote_start=str(quote.index("0.5")),
                quote_end=str(quote.index("0.5") + len("0.5")),
            ),
            BoundedNumericSupport(
                field_path="effect.standard_error",
                verbatim_token="0.2",
                quote_start=str(quote.index("0.2")),
                quote_end=str(quote.index("0.2") + len("0.2")),
            ),
        ],
    )


def _assemble(
    inventory: NativeCandidateInventory,
    packets: list[NativeCandidatePacket[Any] | NativeCandidateUnableToComplete],
    *,
    locator: str = LOCATOR,
) -> Any:
    return assemble_candidate_packets(
        inventory=inventory,
        packets=packets,
        exposed_line_ids=LINES,
        source_locator=locator,
        allowed_outcomes=OUTCOMES,
        allowed_moderators=[],
        outcome_positive_directions=POSITIVE_DIRECTIONS,
    )


def _all_schema_nodes(value: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if isinstance(value, dict):
        output.append(value)
        for item in value.values():
            output.extend(_all_schema_nodes(item))
    elif isinstance(value, list):
        for item in value:
            output.extend(_all_schema_nodes(item))
    return output


def _numeric_support(
    *,
    path: str,
    token: str,
    quote: str,
    occurrence: int = 0,
    normalization: str = "identity",
) -> dict[str, str]:
    start = -1
    for _ in range(occurrence + 1):
        start = quote.index(token, start + 1)
    return {
        "field_path": path,
        "verbatim_token": token,
        "normalization": normalization,
        "quote_start": str(start),
        "quote_end": str(start + len(token)),
    }


def test_model_facing_schemas_are_closed_regex_free_and_mechanically_bounded() -> None:
    schemas = [
        inventory_generation_schema(
            exposed_line_ids=LINES,
            allowed_outcomes=OUTCOMES,
        )
    ]
    for kind in PACKET_MODELS:
        descriptor = NativeCandidateDescriptor(
            candidate_index=1,
            outcome_name="aerobic_capacity",
            effect_kind=kind,
            line_ids=["L10"],
        )
        schemas.append(
            packet_generation_schema(
                candidate=descriptor,
                exposed_line_ids=LINES,
                source_locator=LOCATOR,
                allowed_outcomes=OUTCOMES,
                outcome_positive_directions=POSITIVE_DIRECTIONS,
            )
        )

    for schema in schemas:
        Draft202012Validator.check_schema(schema)
        assert_bounded_generation_schema(schema)
        assert all("pattern" not in node for node in _all_schema_nodes(schema))
        for node in _all_schema_nodes(schema):
            raw_type = node.get("type")
            types = {raw_type} if isinstance(raw_type, str) else set(raw_type or [])
            if "array" in types:
                assert type(node.get("maxItems")) is int
            if "object" in types:
                assert node.get("additionalProperties") is False
            if "string" in types and "enum" not in node and "const" not in node:
                assert type(node.get("maxLength")) is int
            if types.intersection({"integer", "number"}):
                assert node.get("enum")


def test_official_native_schema_remains_the_unchanged_acceptance_authority() -> None:
    schema = native_publication_extraction_json_schema()

    assert schema["$id"] == "urn:literature-multiverse:native-publication-extraction:v1"
    assert hash_canonical(schema) == (
        "8913bfa2846c6f45cb27789c3ab47199c38322ae0954c847f77bcb10750a3d65"
    )


def test_inventory_cap_hit_is_saturation_not_completeness() -> None:
    inventory = _inventory(INVENTORY_SENTINEL_CAP)

    assert inventory.inventory_status == "candidates_found"
    assert inventory.has_more_or_uncertain is False
    assert inventory.authorizes_packet_generation() is False
    assert inventory.blocking_status() == (
        "inventory_capacity_or_uncertainty_non_authorizing"
    )
    with pytest.raises(
        NativeBoundedGenerationError,
        match="inventory_capacity_or_uncertainty_non_authorizing",
    ):
        _assemble(inventory, [])


@pytest.mark.parametrize(
    "payload",
    [
        {
            "inventory_status": "candidates_found",
            "candidates": [],
            "has_more_or_uncertain": False,
        },
        {
            "inventory_status": "no_candidate_found",
            "candidates": [_candidate(1).model_dump(mode="json")],
            "has_more_or_uncertain": False,
        },
        {
            "inventory_status": "overflow_or_uncertain",
            "candidates": [],
            "has_more_or_uncertain": False,
        },
        {
            "inventory_status": "candidates_found",
            "candidates": [
                _candidate(2).model_dump(mode="json"),
                _candidate(1).model_dump(mode="json"),
            ],
            "has_more_or_uncertain": False,
        },
    ],
)
def test_inventory_status_and_cardinality_truth_table_fails_closed(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        NativeCandidateInventory.model_validate(payload)


def test_inventory_row_validation_rejects_unknown_anchors_outcomes_and_zero_projection() -> None:
    unknown_line = _inventory(1).model_dump(mode="json")
    unknown_line["candidates"][0]["line_ids"] = ["L999"]
    with pytest.raises(NativeBoundedGenerationError, match="json_schema_validation_error"):
        validate_inventory_for_row(
            unknown_line,
            exposed_line_ids=LINES,
            allowed_outcomes=OUTCOMES,
        )

    unknown_outcome = _inventory(1).model_dump(mode="json")
    unknown_outcome["candidates"][0]["outcome_name"] = "unfrozen_outcome"
    with pytest.raises(NativeBoundedGenerationError, match="json_schema_validation_error"):
        validate_inventory_for_row(
            unknown_outcome,
            exposed_line_ids=LINES,
            allowed_outcomes=OUTCOMES,
        )

    with pytest.raises(NativeBoundedGenerationError, match="json_schema_validation_error"):
        validate_inventory_for_row(
            _inventory(1),
            exposed_line_ids=[],
            allowed_outcomes=OUTCOMES,
        )


def test_packet_validator_crossbinds_exact_candidate_and_source_context() -> None:
    candidate = _candidate(1)
    packet = _packet(candidate)
    validated = validate_packet_for_candidate(
        packet,
        candidate=candidate,
        exposed_line_ids=LINES,
        source_locator=LOCATOR,
        allowed_outcomes=OUTCOMES,
        allowed_moderators=[],
        outcome_positive_directions=POSITIVE_DIRECTIONS,
    )
    assert validated == packet

    tampered = packet.model_dump(mode="json")
    tampered["evidence"]["line_ids"] = ["L20"]
    with pytest.raises(NativeBoundedGenerationError, match="json_schema_validation_error"):
        validate_packet_for_candidate(
            tampered,
            candidate=candidate,
            exposed_line_ids=LINES,
            source_locator=LOCATOR,
            allowed_outcomes=OUTCOMES,
            allowed_moderators=[],
            outcome_positive_directions=POSITIVE_DIRECTIONS,
        )


def test_multi_study_multi_cohort_packets_round_trip_without_loss() -> None:
    candidates = [
        _candidate(1, line_id="L10"),
        _candidate(2, outcome="muscle_strength", line_id="L20"),
        _candidate(3, line_id="L30"),
    ]
    inventory = NativeCandidateInventory(
        inventory_status="candidates_found",
        candidates=candidates,
        has_more_or_uncertain=False,
    )
    packets = [
        _packet(candidates[0], study_key="study-a", cohort_key="cohort-a"),
        _packet(candidates[1], study_key="study-a", cohort_key="cohort-b"),
        _packet(candidates[2], study_key="study-b", cohort_key="cohort-a"),
    ]

    official = _assemble(inventory, packets)

    assert official.extraction_schema_version == "native-publication-extraction-v1"
    assert official.status == "estimable"
    assert [study.key for study in official.studies] == ["study-a", "study-b"]
    assert [[cohort.key for cohort in study.cohorts] for study in official.studies] == [
        ["cohort-a", "cohort-b"],
        ["cohort-a"],
    ]
    assert sum(
        len(cohort.findings) for study in official.studies for cohort in study.cohorts
    ) == len(candidates)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "duplicate", "unsorted"],
)
def test_assembler_requires_exact_ordered_candidate_bijection(mutation: str) -> None:
    inventory = _inventory(2)
    packets = [_packet(candidate) for candidate in inventory.candidates]
    if mutation == "missing":
        packets = packets[:1]
    elif mutation == "extra":
        packets.append(_packet(_candidate(3, line_id="L30")))
    elif mutation == "duplicate":
        packets[1] = packets[0]
    else:
        packets.reverse()

    with pytest.raises(NativeBoundedGenerationError, match="membership_mismatch"):
        _assemble(inventory, packets)


def test_assembler_rejects_conflicting_shared_node_metadata_whole_publication() -> None:
    inventory = _inventory(2)
    packets = [_packet(candidate) for candidate in inventory.candidates]
    conflicting = deepcopy(packets[1].model_dump(mode="json"))
    conflicting["treatment_arm"]["label"] = "Conflicting arm label"
    packets[1] = NativeCandidatePacket[DirectStandardErrorEffect].model_validate(
        conflicting
    )

    with pytest.raises(NativeBoundedGenerationError, match="arm_metadata_conflict"):
        _assemble(inventory, packets)


def test_packet_generation_schema_singletons_candidate_source_and_lineage() -> None:
    candidate = _candidate(1)
    schema = packet_generation_schema(
        candidate=candidate,
        exposed_line_ids=LINES,
        source_locator=LOCATOR,
        allowed_outcomes=OUTCOMES,
        outcome_positive_directions=POSITIVE_DIRECTIONS,
    )
    nodes = _all_schema_nodes(schema)

    assert any(node.get("enum") == [1] for node in nodes)
    assert any(node.get("enum") == [LOCATOR] for node in nodes)
    assert any(node.get("enum") == candidate.line_ids for node in nodes)
    assert any(node.get("enum") == [candidate.outcome_name] for node in nodes)


def test_strict_json_schema_prevents_pydantic_coercion() -> None:
    payload = _inventory(1).model_dump(mode="json")
    payload["has_more_or_uncertain"] = "false"
    with pytest.raises(NativeBoundedGenerationError, match="json_schema_validation_error"):
        validate_inventory_for_row(
            payload,
            exposed_line_ids=LINES,
            allowed_outcomes=OUTCOMES,
        )

    candidate = _candidate(1)
    packet = _packet(candidate).model_dump(mode="json")
    packet["candidate_index"] = "1"
    with pytest.raises(NativeBoundedGenerationError, match="json_schema_validation_error"):
        validate_packet_for_candidate(
            packet,
            candidate=candidate,
            exposed_line_ids=LINES,
            source_locator=LOCATOR,
            allowed_outcomes=OUTCOMES,
            allowed_moderators=[],
            outcome_positive_directions=POSITIVE_DIRECTIONS,
        )


def test_duplicate_candidate_descriptor_signature_is_rejected() -> None:
    with pytest.raises(ValueError, match="descriptor_signature_duplicate"):
        NativeCandidateInventory(
            inventory_status="candidates_found",
            candidates=[_candidate(1), _candidate(2, line_id="L10")],
            has_more_or_uncertain=False,
        )


def test_assembler_revalidates_exact_source_context() -> None:
    inventory = _inventory(1)
    packet = _packet(inventory.candidates[0])

    with pytest.raises(NativeBoundedGenerationError, match="json_schema_validation_error"):
        _assemble(inventory, [packet], locator="json:another-source#/document")


@pytest.mark.parametrize("lexeme", ["1e999", "1000000000001", "NaN", "Infinity"])
def test_scientific_numeric_lexemes_are_finite_and_magnitude_bounded(
    lexeme: str,
) -> None:
    with pytest.raises(ValueError):
        DirectStandardErrorEffect(
            effect_format="mean_difference",
            estimate=lexeme,
            standard_error="0.2",
        )


def test_numeric_support_rejects_wrong_estimate_even_with_exact_quote_and_se() -> None:
    packet = _packet(_candidate(1)).model_dump(mode="json")
    packet["effect"]["estimate"] = "0.7"

    with pytest.raises(
        ValueError, match=r"numeric_support_value_mismatch:effect\.estimate"
    ):
        NativeCandidatePacket[DirectStandardErrorEffect].model_validate(packet)


def test_numeric_support_covers_every_emitted_scientific_numeric_leaf() -> None:
    candidate = _candidate(1)
    packet = _packet(candidate).model_dump(mode="json")
    quote = (
        "N 40 treatment 20 control 20 week 8 estimate 0.5 SE 0.2 "
        "p .05 margin .1."
    )
    packet["cohort"]["total_sample_size"] = "40"
    packet["treatment_arm"]["sample_size"] = "20"
    packet["comparator_arm"]["sample_size"] = "20"
    packet["finding"]["timepoint"] = {
        "kind": "exact",
        "value": "8",
        "lower": None,
        "upper": None,
        "unit": "week",
        "anchor": None,
        "raw_label": None,
    }
    packet["effect"]["reported_p_value"] = ".05"
    packet["effect"]["equivalence_margin"] = ".1"
    packet["evidence"]["quote"] = quote
    packet["numeric_support"] = sorted(
        [
            _numeric_support(path="cohort.total_sample_size", token="40", quote=quote),
            _numeric_support(
                path="treatment_arm.sample_size",
                token="20",
                quote=quote,
            ),
            _numeric_support(
                path="comparator_arm.sample_size",
                token="20",
                quote=quote,
                occurrence=1,
            ),
            _numeric_support(
                path="finding.timepoint.value", token="8", quote=quote
            ),
            _numeric_support(path="effect.estimate", token="0.5", quote=quote),
            _numeric_support(
                path="effect.standard_error", token="0.2", quote=quote
            ),
            _numeric_support(
                path="effect.reported_p_value", token=".05", quote=quote
            ),
            _numeric_support(
                path="effect.equivalence_margin", token=".1", quote=quote
            ),
        ],
        key=lambda item: item["field_path"],
    )

    validated = NativeCandidatePacket[DirectStandardErrorEffect].model_validate(packet)
    assert validated.effect.reported_p_value == ".05"
    incomplete = deepcopy(packet)
    incomplete["numeric_support"] = incomplete["numeric_support"][:-1]
    with pytest.raises(ValueError, match="numeric_support_field_set_mismatch"):
        NativeCandidatePacket[DirectStandardErrorEffect].model_validate(incomplete)


def test_percent_normalization_is_field_specific_and_deterministic() -> None:
    candidate = NativeCandidateDescriptor(
        candidate_index=1,
        outcome_name="aerobic_capacity",
        effect_kind="direct_confidence_interval",
        line_ids=["L10"],
    )
    packet = _packet(candidate).model_dump(mode="json")
    quote = "Estimate 0.5 with 95% CI 0.1 to 0.9."
    packet["effect"] = {
        "effect_kind": "direct_confidence_interval",
        "effect_format": "mean_difference",
        "estimate": "0.5",
        "ci_lower": "0.1",
        "ci_upper": "0.9",
        "ci_level": "0.95",
        "unit": "mL/kg/min",
        "reported_p_value": None,
        "reported_significance": "not_reported",
        "equivalence_conclusion": "not_tested",
        "equivalence_margin": None,
        "moderators": [],
        "extraction_method": "reported",
    }
    packet["evidence"]["quote"] = quote
    packet["numeric_support"] = sorted(
        [
            _numeric_support(path="effect.estimate", token="0.5", quote=quote),
            _numeric_support(path="effect.ci_lower", token="0.1", quote=quote),
            _numeric_support(path="effect.ci_upper", token="0.9", quote=quote),
            _numeric_support(
                path="effect.ci_level",
                token="95%",
                quote=quote,
                normalization="percent_to_proportion",
            ),
        ],
        key=lambda item: item["field_path"],
    )
    validated = NativeCandidatePacket[DirectConfidenceIntervalEffect].model_validate(
        packet
    )
    assert validated.effect.ci_level == "0.95"

    attack = deepcopy(packet)
    support = next(
        item
        for item in attack["numeric_support"]
        if item["field_path"] == "effect.estimate"
    )
    attack["evidence"]["quote"] += " Unrelated 50%."
    support["verbatim_token"] = "50%"
    support["normalization"] = "percent_to_proportion"
    support["quote_start"] = str(attack["evidence"]["quote"].index("50%"))
    support["quote_end"] = str(int(support["quote_start"]) + len("50%"))
    with pytest.raises(ValueError, match="percent_normalization_forbidden"):
        NativeCandidatePacket[DirectConfidenceIntervalEffect].model_validate(attack)


@pytest.mark.parametrize(
    ("quote", "token", "expected_error"),
    [
        ("Estimate \u22120.5 with SE 0.2.", "0.5", "token_boundary"),
        ("Estimate 1,000 with SE 0.2.", "1", "token_boundary"),
        ("Estimate 20% with SE 0.2.", "20", "unlicensed_percent_token"),
        ("Estimate <0.5 with SE 0.2.", "0.5", "token_boundary"),
        ("Reported p < .05 with SE 0.2.", ".05", "token_boundary"),
        ("Estimate 20 % with SE 0.2.", "20", "unlicensed_percent_token"),
        ("Estimate 1 000 with SE 0.2.", "1", "token_boundary"),
        ("Estimate 1\u202f000 with SE 0.2.", "1", "token_boundary"),
        ("Estimate 10\u207b3 with SE 0.2.", "3", "token_boundary"),
    ],
)
def test_numeric_support_rejects_sign_grouping_and_unit_marker_bypasses(
    quote: str,
    token: str,
    expected_error: str,
) -> None:
    packet = _packet(_candidate(1)).model_dump(mode="json")
    packet["effect"]["estimate"] = token
    packet["evidence"]["quote"] = quote
    packet["numeric_support"] = sorted(
        [
            _numeric_support(path="effect.estimate", token=token, quote=quote),
            _numeric_support(
                path="effect.standard_error", token="0.2", quote=quote
            ),
        ],
        key=lambda item: item["field_path"],
    )
    with pytest.raises(ValueError, match=expected_error):
        NativeCandidatePacket[DirectStandardErrorEffect].model_validate(packet)


def test_percent_normalization_requires_percent_bearing_exact_slice() -> None:
    quote = "There were 95 participants; CI level was not reported."
    with pytest.raises(ValueError, match="percent_marker_missing"):
        BoundedNumericSupport(
            field_path="effect.ci_level",
            verbatim_token="95",
            normalization="percent_to_proportion",
            quote_start=str(quote.index("95")),
            quote_end=str(quote.index("95") + 2),
        )


def test_one_source_occurrence_cannot_support_two_numeric_fields() -> None:
    packet = _packet(_candidate(1)).model_dump(mode="json")
    packet["effect"]["standard_error"] = "0.5"
    packet["numeric_support"][1] = deepcopy(packet["numeric_support"][0])
    packet["numeric_support"][1]["field_path"] = "effect.standard_error"
    with pytest.raises(ValueError, match="source_span_reused"):
        NativeCandidatePacket[DirectStandardErrorEffect].model_validate(packet)


@pytest.mark.parametrize(
    "quote",
    [
        "Estimate 1, with SE 0.2.",
        "Estimate 1. SE was 0.2.",
        "Estimate = 1; SE was 0.2.",
    ],
)
def test_ordinary_numeric_punctuation_and_explicit_equality_remain_supported(
    quote: str,
) -> None:
    packet = _packet(_candidate(1)).model_dump(mode="json")
    packet["effect"]["estimate"] = "1"
    packet["evidence"]["quote"] = quote
    packet["numeric_support"] = sorted(
        [
            _numeric_support(path="effect.estimate", token="1", quote=quote),
            _numeric_support(
                path="effect.standard_error", token="0.2", quote=quote
            ),
        ],
        key=lambda item: item["field_path"],
    )
    NativeCandidatePacket[DirectStandardErrorEffect].model_validate(packet)


def test_effect_kind_format_and_arm_orientation_fail_closed() -> None:
    with pytest.raises(ValueError, match="continuous_effect_format_incompatible"):
        ContinuousGroupEffect(
            effect_format="odds_ratio",
            treatment_mean="1",
            treatment_sd="1",
            treatment_n="20",
            control_mean="0",
            control_sd="1",
            control_n="20",
        )

    packet = _packet(_candidate(1)).model_dump(mode="json")
    packet["treatment_arm"]["role"] = "control"
    with pytest.raises(ValueError, match="treatment_arm_role_invalid"):
        NativeCandidatePacket[DirectStandardErrorEffect].model_validate(packet)

    with pytest.raises(ValueError, match="direct_ratio_value_not_positive"):
        DirectStandardErrorEffect(
            effect_format="odds_ratio",
            estimate="-1",
            standard_error="0.2",
        )
    DirectStandardErrorEffect(
        effect_format="log_odds_ratio",
        estimate="-1",
        standard_error="0.2",
    )


def test_unable_to_complete_is_schema_valid_terminal_and_never_assembles() -> None:
    inventory = _inventory(1)
    candidate = inventory.candidates[0]
    unable = NativeCandidateUnableToComplete(
        candidate_index=1,
        reason="insufficient_numeric_support",
    )
    schema = packet_generation_schema(
        candidate=candidate,
        exposed_line_ids=LINES,
        source_locator=LOCATOR,
        allowed_outcomes=OUTCOMES,
        outcome_positive_directions=POSITIVE_DIRECTIONS,
    )
    Draft202012Validator(schema).validate(unable.model_dump(mode="json"))
    validated = validate_packet_for_candidate(
        unable,
        candidate=candidate,
        exposed_line_ids=LINES,
        source_locator=LOCATOR,
        allowed_outcomes=OUTCOMES,
        allowed_moderators=[],
        outcome_positive_directions=POSITIVE_DIRECTIONS,
    )
    assert isinstance(validated, NativeCandidateUnableToComplete)
    with pytest.raises(NativeBoundedGenerationError, match="unable_to_complete"):
        _assemble(inventory, [unable])


def test_title_abstract_projection_sections_are_context_bound_not_antiox_hardcoded() -> None:
    candidate = _candidate(1)
    packet = _packet(candidate).model_dump(mode="json")
    packet["evidence"]["section"] = "Abstract"
    validated = validate_packet_for_candidate(
        packet,
        candidate=candidate,
        exposed_line_ids=LINES,
        source_locator=LOCATOR,
        allowed_outcomes=OUTCOMES,
        allowed_moderators=[],
        allowed_sections=["Title", "Abstract"],
        outcome_positive_directions=POSITIVE_DIRECTIONS,
    )
    assert validated.evidence.section == "Abstract"
