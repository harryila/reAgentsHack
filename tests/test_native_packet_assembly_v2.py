from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any

import pytest

import literature_multiverse.native_packet_assembly_v2 as assembly_v2
from literature_multiverse.effects import EffectFormat
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_candidate_inventory_v2 import (
    MetaSynPassageCandidateV2,
)
from literature_multiverse.metasyn_extraction_inputs_v2 import (
    MetaSynExtractionQuestionSurfaceV2,
)
from literature_multiverse.metasyn_projection_v2 import (
    FrozenMetaSynProjectionV2,
    freeze_metasyn_projection_v2,
    freeze_projection_v2_lineage_binding,
)
from literature_multiverse.native_grounding import ResolvedNativeSource, ResolvedSourceLine
from literature_multiverse.native_packet_assembly_v2 import (
    GroundedBinaryGroupEffectV2,
    GroundedContinuousGroupEffectV2,
    GroundedDirectConfidenceIntervalEffectV2,
    GroundedDirectStandardErrorEffectV2,
    GroundedDirectVarianceEffectV2,
    NativePacketAssemblyAbstentionV2,
    NativePacketAssemblyCompletedV2,
    NativePacketAssemblyV2Error,
    PacketAssemblyAnalysisPolicyV2,
    PacketAssemblyProtocolOrientationV2,
    freeze_packet_assembly_analysis_policy_v2,
    freeze_packet_assembly_protocol_orientation_v2,
)
from literature_multiverse.native_packet_assembly_v2 import (
    assemble_native_packet_v2 as _assemble_native_packet_v2_impl,
)
from literature_multiverse.native_packet_assembly_v2 import (
    validate_native_packet_assembly_v2 as _validate_native_packet_assembly_v2_impl,
)
from literature_multiverse.native_packet_grounding_v2 import (
    MODEL_OUTCOME_V2_VERSION,
    PacketGroundingCompletedReceiptV2,
    freeze_passage_packet_candidate_binding_v2,
    freeze_passage_packet_grounding_receipt_v2,
)
from literature_multiverse.native_question_projection import (
    QuestionProjectionSpecV1,
    freeze_question_projection_spec,
    project_resolved_source_for_question,
)

COMMON = (
    "Study Alpha, Main cohort, Arm A, Placebo, Active comparison, "
    "Blood pressure: "
)
EFFECT_CASES: dict[str, dict[str, Any]] = {
    "direct_standard_error": {
        "quote": COMMON + "Hedges g estimate 0.50 and standard error 0.10.",
        "format_token": "Hedges g",
        "unit": None,
        "numeric": {
            "effect.estimate": "0.50",
            "effect.standard_error": "0.10",
        },
        "effect_class": GroundedDirectStandardErrorEffectV2,
        "expected_format": EffectFormat.HEDGES_G,
        "extraction_method": "reported",
    },
    "direct_variance": {
        "quote": COMMON + "Hedges g estimate 0.60 and variance 0.04.",
        "format_token": "Hedges g",
        "unit": None,
        "numeric": {
            "effect.estimate": "0.60",
            "effect.variance": "0.04",
        },
        "effect_class": GroundedDirectVarianceEffectV2,
        "expected_format": EffectFormat.HEDGES_G,
        "extraction_method": "reported",
    },
    "direct_confidence_interval": {
        "quote": (
            COMMON
            + "Hedges g estimate -0.50 with 95% CI from -0.80 to -0.20."
        ),
        "format_token": "Hedges g",
        "unit": None,
        "numeric": {
            "effect.ci_level": ("95%", "percent_to_proportion"),
            "effect.ci_lower": "-0.80",
            "effect.ci_upper": "-0.20",
            "effect.estimate": "-0.50",
        },
        "effect_class": GroundedDirectConfidenceIntervalEffectV2,
        "expected_format": EffectFormat.HEDGES_G,
        "extraction_method": "reported",
    },
    "continuous_group_statistics": {
        "quote": (
            COMMON
            + "treatment mean 10.1, SD 2.3, n 40; control mean 12.2, "
            "SD 3.4, n 50; units mmHg."
        ),
        "format_token": None,
        "unit": "mmHg",
        "numeric": {
            "effect.control_mean": "12.2",
            "effect.control_n": "50",
            "effect.control_sd": "3.4",
            "effect.treatment_mean": "10.1",
            "effect.treatment_n": "40",
            "effect.treatment_sd": "2.3",
        },
        "effect_class": GroundedContinuousGroupEffectV2,
        "expected_format": EffectFormat.MEAN_DIFFERENCE,
        "extraction_method": "computed_from_reported_statistics",
    },
    "binary_group_statistics": {
        "quote": (
            COMMON
            + "treatment events 8 of 40; control events 12 of 50."
        ),
        "format_token": None,
        "unit": None,
        "numeric": {
            "effect.control_events": "12",
            "effect.control_total": "50",
            "effect.treatment_events": "8",
            "effect.treatment_total": "40",
        },
        "effect_class": GroundedBinaryGroupEffectV2,
        "expected_format": EffectFormat.ODDS_RATIO,
        "extraction_method": "computed_from_reported_statistics",
    },
}


def _protocol(
    *,
    population: str = "adults",
) -> QuestionProjectionSpecV1:
    return freeze_question_projection_spec(
        question_id="synthetic-assembly-question",
        population=population,
        intervention_or_exposure="Arm A",
        comparison="Placebo",
        outcome_texts=["Blood pressure"],
        treatment_role="intervention_or_exposure",
        comparator_role="comparator",
        contrast_estimand=(
            "between_group_effect_intervention_or_exposure_vs_comparator_on_reported_measure"
        ),
        positive_direction_means_by_outcome={
            "Blood pressure": "higher values indicate greater blood pressure"
        },
    )


def _orientation(
    protocol: QuestionProjectionSpecV1,
    *,
    relation_kind: str = "intervention",
) -> PacketAssemblyProtocolOrientationV2:
    outcomes = protocol.question_fields.outcomes
    outcome_text = {item.outcome_id: item.outcome_text for item in outcomes}
    direction = {
        item.outcome_id: item.positive_direction_means for item in outcomes
    }
    payload = {
        "question_surface_version": "metasyn-extraction-question-surface-v2",
        "question_id": protocol.question_id,
        "question_spec_sha256": protocol.question_spec_sha256,
        "research_question": "What is the between-group effect?",
        "population": protocol.question_fields.population,
        "relation_kind": relation_kind,
        "intervention_or_exposure": (
            protocol.question_fields.intervention_or_exposure
        ),
        "comparison": protocol.question_fields.comparison,
        "treatment_role": protocol.question_fields.treatment_role,
        "comparator_role": protocol.question_fields.comparator_role,
        "contrast_orientation": "intervention_or_exposure_minus_comparator",
        "contrast_estimand": protocol.question_fields.contrast_estimand,
        "inclusion_criteria": None,
        "exclusion_criteria": None,
        "allowed_outcome_ids": sorted(outcome_text),
        "allowed_outcome_text_by_id": dict(sorted(outcome_text.items())),
        "raw_positive_direction_meaning_by_outcome_id": dict(
            sorted(direction.items())
        ),
        "outcome_membership_sha256": hash_canonical(outcome_text),
    }
    surface = MetaSynExtractionQuestionSurfaceV2.model_validate(
        {**payload, "question_surface_sha256": hash_canonical(payload)}
    )
    return freeze_packet_assembly_protocol_orientation_v2(
        question_surface=surface
    )


def assemble_native_packet_v2(**kwargs: Any) -> Any:
    kwargs.setdefault("protocol_orientation", _orientation(kwargs["protocol"]))
    return _assemble_native_packet_v2_impl(**kwargs)


def validate_native_packet_assembly_v2(**kwargs: Any) -> Any:
    kwargs.setdefault("protocol_orientation", _orientation(kwargs["protocol"]))
    return _validate_native_packet_assembly_v2_impl(**kwargs)


def _source(*quotes: str) -> ResolvedNativeSource:
    lines: list[ResolvedSourceLine] = []
    char_cursor = 0
    byte_cursor = 0
    for line_number, quote in enumerate(quotes, start=1):
        char_end = char_cursor + len(quote)
        byte_end = byte_cursor + len(quote.encode("utf-8"))
        lines.append(
            ResolvedSourceLine(
                line_id=f"L{line_number}",
                line_number=line_number,
                section="Results",
                text=quote,
                char_start=char_cursor,
                char_end=char_end,
                utf8_byte_start=byte_cursor,
                utf8_byte_end=byte_end,
            )
        )
        char_cursor = char_end + 1
        byte_cursor = byte_end + 1
    return ResolvedNativeSource(
        source_kind="metasyn_parquet_row",
        artifact_path="data/cache/synthetic-assembly.parquet",
        artifact_sha256=hashlib.sha256(b"synthetic-assembly").hexdigest(),
        source_locator="metasyn://synthetic-assembly/1",
        source_payload_sha256=hash_canonical({"quotes": list(quotes)}),
        source_text="\n".join(quotes),
        lines=lines,
    )


def _projection(
    protocol: QuestionProjectionSpecV1, *quotes: str
) -> FrozenMetaSynProjectionV2:
    upstream = project_resolved_source_for_question(
        row_id="synthetic-assembly:1",
        source=_source(*quotes),
        spec=protocol,
    )
    lineage = freeze_projection_v2_lineage_binding(
        upstream_execution_bundle_sha256="a" * 64,
        upstream_row_context_sha256="b" * 64,
        upstream_source_row_sha256="c" * 64,
        projection=upstream,
    )
    return freeze_metasyn_projection_v2(
        projection=upstream,
        lineage_binding=lineage,
    )


def _passage_id(projection: FrozenMetaSynProjectionV2, quote: str) -> str:
    matches = [
        item.passage_anchor
        for item in projection.passages
        if item.selection_status == "selected" and item.text == quote
    ]
    assert len(matches) == 1
    return matches[0]


def _candidate(
    *,
    projection: FrozenMetaSynProjectionV2,
    quote: str,
    effect_kind: str,
    index: int = 1,
) -> MetaSynPassageCandidateV2:
    return MetaSynPassageCandidateV2(
        candidate_index=index,
        canonical_outcome_id="outcome-01",
        outcome_concept_quote="Blood pressure",
        effect_kind=effect_kind,
        passage_ids=[_passage_id(projection, quote)],
    )


def _identity_claims(*, cohort_label: str = "Main cohort") -> list[dict[str, str]]:
    values = {
        "study.source_label": "Study Alpha",
        "cohort.source_label": cohort_label,
        "treatment_arm.label": "Arm A",
        "comparator_arm.label": "Placebo",
        "contrast.label": "Active comparison",
    }
    return [
        {"field_path": field_path, "verbatim_identity_text": value}
        for field_path, value in sorted(values.items())
    ]


def _numeric_claims(values: dict[str, Any]) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    for field_path, raw in sorted(values.items()):
        token, normalization = (
            raw if isinstance(raw, tuple) else (raw, "identity")
        )
        claims.append(
            {
                "field_path": field_path,
                "verbatim_numeric_token": token,
                "normalization": normalization,
            }
        )
    return claims


def _grounding(
    *,
    candidate: MetaSynPassageCandidateV2,
    projection: FrozenMetaSynProjectionV2,
    case: dict[str, Any],
    numeric: dict[str, Any] | None = None,
    cohort_label: str = "Main cohort",
    effect_unit: str | object | None = ...,
) -> PacketGroundingCompletedReceiptV2:
    binding = freeze_passage_packet_candidate_binding_v2(
        candidate=candidate,
        projection=projection,
    )
    unit = case["unit"] if effect_unit is ... else effect_unit
    outcome = {
        "outcome_version": MODEL_OUTCOME_V2_VERSION,
        "packet_status": "completed",
        "candidate_binding_sha256": binding.binding_sha256,
        "evidence_quote": case["quote"],
        "effect_format_token": case["format_token"],
        "effect_unit": unit,
        "numeric_claims": _numeric_claims(
            case["numeric"] if numeric is None else numeric
        ),
        "identity_claims": _identity_claims(cohort_label=cohort_label),
        "timepoint": {"kind": "not_reported"},
    }
    receipt = freeze_passage_packet_grounding_receipt_v2(
        model_outcome=outcome,
        candidate=candidate,
        projection=projection,
    )
    assert isinstance(receipt, PacketGroundingCompletedReceiptV2)
    return receipt


def _policy(
    *, continuous: EffectFormat = EffectFormat.MEAN_DIFFERENCE
) -> PacketAssemblyAnalysisPolicyV2:
    return freeze_packet_assembly_analysis_policy_v2(
        continuous_group_effect_format=continuous,
        binary_group_effect_format=EffectFormat.ODDS_RATIO,
    )


@pytest.mark.parametrize("effect_kind", tuple(EFFECT_CASES))
def test_all_effect_families_assemble_complete_grounded_typed_effects(
    effect_kind: str,
) -> None:
    case = EFFECT_CASES[effect_kind]
    protocol = _protocol()
    projection = _projection(protocol, case["quote"])
    candidate = _candidate(
        projection=projection,
        quote=case["quote"],
        effect_kind=effect_kind,
    )
    grounding = _grounding(
        candidate=candidate,
        projection=projection,
        case=case,
    )

    first = assemble_native_packet_v2(
        candidate=candidate,
        projection=projection,
        protocol=protocol,
        analysis_policy=_policy(),
        grounding_receipt=grounding,
    )
    second = assemble_native_packet_v2(
        candidate=candidate,
        projection=projection,
        protocol=protocol,
        analysis_policy=_policy(),
        grounding_receipt=grounding,
    )

    assert isinstance(first, NativePacketAssemblyCompletedV2)
    assert first == second
    assert isinstance(first.typed_effect.effect, case["effect_class"])
    assert first.typed_effect.effect.effect_format is case["expected_format"]
    assert first.typed_effect.extraction_method == case["extraction_method"]
    assert first.authorizes_typed_effect is True
    assert first.authorizes_native_candidate_packet is False
    assert first.native_candidate_packet is None
    assert first.typed_effect.reported_significance_coverage == (
        "not_extracted_from_selected_support"
    )
    assert validate_native_packet_assembly_v2(
        assembly=first,
        candidate=candidate,
        projection=projection,
        protocol=protocol,
        analysis_policy=_policy(),
        grounding_receipt=grounding,
    ) == first


def test_generic_frozen_protocol_uses_hash_bound_relation_kind_for_arm_role() -> None:
    case = EFFECT_CASES["direct_standard_error"]
    protocol = _protocol()
    projection = _projection(protocol, case["quote"])
    candidate = _candidate(
        projection=projection,
        quote=case["quote"],
        effect_kind="direct_standard_error",
    )
    grounding = _grounding(
        candidate=candidate,
        projection=projection,
        case=case,
    )
    intervention = _orientation(protocol, relation_kind="intervention")
    exposure = _orientation(protocol, relation_kind="exposure")

    result = assemble_native_packet_v2(
        candidate=candidate,
        projection=projection,
        protocol=protocol,
        protocol_orientation=exposure,
        analysis_policy=_policy(),
        grounding_receipt=grounding,
    )

    assert isinstance(result, NativePacketAssemblyCompletedV2)
    assert result.typed_effect.treatment_arm_role.value == "exposure"
    assert result.protocol_orientation == exposure
    assert intervention.orientation_sha256 != exposure.orientation_sha256
    assert result.typed_effect.protocol_orientation_sha256 == (
        exposure.orientation_sha256
    )


def test_incomplete_numeric_core_returns_explicit_non_authorizing_abstention() -> None:
    case = EFFECT_CASES["direct_standard_error"]
    protocol = _protocol()
    projection = _projection(protocol, case["quote"])
    candidate = _candidate(
        projection=projection,
        quote=case["quote"],
        effect_kind="direct_standard_error",
    )
    grounding = _grounding(
        candidate=candidate,
        projection=projection,
        case=case,
        numeric={"effect.estimate": "0.50"},
    )

    result = assemble_native_packet_v2(
        candidate=candidate,
        projection=projection,
        protocol=protocol,
        analysis_policy=_policy(),
        grounding_receipt=grounding,
    )

    assert isinstance(result, NativePacketAssemblyAbstentionV2)
    assert result.primary_blocker == "effect_numeric_field_set_incompatible"
    assert result.missing_field_paths == ["effect.standard_error"]
    assert result.authorizes_typed_effect is False


def test_continuous_mean_difference_requires_grounded_unit_but_standardized_does_not() -> None:
    case = EFFECT_CASES["continuous_group_statistics"]
    unitless_quote = case["quote"].replace("; units mmHg", "")
    unitless_case = {**case, "quote": unitless_quote, "unit": None}
    protocol = _protocol()
    projection = _projection(protocol, unitless_quote)
    candidate = _candidate(
        projection=projection,
        quote=unitless_quote,
        effect_kind="continuous_group_statistics",
    )
    grounding = _grounding(
        candidate=candidate,
        projection=projection,
        case=unitless_case,
    )

    mean_difference = assemble_native_packet_v2(
        candidate=candidate,
        projection=projection,
        protocol=protocol,
        analysis_policy=_policy(continuous=EffectFormat.MEAN_DIFFERENCE),
        grounding_receipt=grounding,
    )
    standardized = assemble_native_packet_v2(
        candidate=candidate,
        projection=projection,
        protocol=protocol,
        analysis_policy=_policy(continuous=EffectFormat.HEDGES_G),
        grounding_receipt=grounding,
    )

    assert isinstance(mean_difference, NativePacketAssemblyAbstentionV2)
    assert mean_difference.primary_blocker == "effect_contract_invalid"
    assert isinstance(standardized, NativePacketAssemblyCompletedV2)
    assert standardized.typed_effect.effect.effect_format is EffectFormat.HEDGES_G
    assert standardized.typed_effect.effect.unit is None


def test_protocol_hash_and_orientation_binding_mismatches_abstain() -> None:
    case = EFFECT_CASES["direct_variance"]
    protocol = _protocol()
    projection = _projection(protocol, case["quote"])
    candidate = _candidate(
        projection=projection,
        quote=case["quote"],
        effect_kind="direct_variance",
    )
    grounding = _grounding(
        candidate=candidate,
        projection=projection,
        case=case,
    )

    wrong_hash = assemble_native_packet_v2(
        candidate=candidate,
        projection=projection,
        protocol=_protocol(population="children"),
        analysis_policy=_policy(),
        grounding_receipt=grounding,
    )
    mismatched_orientation = assemble_native_packet_v2(
        candidate=candidate,
        projection=projection,
        protocol=protocol,
        protocol_orientation=_orientation(_protocol(population="children")),
        analysis_policy=_policy(),
        grounding_receipt=grounding,
    )

    assert isinstance(wrong_hash, NativePacketAssemblyAbstentionV2)
    assert wrong_hash.primary_blocker == "question_spec_hash_mismatch"
    assert isinstance(mismatched_orientation, NativePacketAssemblyAbstentionV2)
    assert (
        mismatched_orientation.primary_blocker
        == "protocol_orientation_binding_mismatch"
    )


def test_grounding_abstention_propagates_as_non_authorizing_assembly() -> None:
    case = EFFECT_CASES["direct_standard_error"]
    protocol = _protocol()
    projection = _projection(protocol, case["quote"])
    candidate = _candidate(
        projection=projection,
        quote=case["quote"],
        effect_kind="direct_standard_error",
    )
    binding = freeze_passage_packet_candidate_binding_v2(
        candidate=candidate,
        projection=projection,
    )
    grounding = freeze_passage_packet_grounding_receipt_v2(
        model_outcome={
            "outcome_version": MODEL_OUTCOME_V2_VERSION,
            "packet_status": "unable_to_complete",
            "candidate_binding_sha256": binding.binding_sha256,
            "reason": "source_support_incomplete",
        },
        candidate=candidate,
        projection=projection,
    )

    result = assemble_native_packet_v2(
        candidate=candidate,
        projection=projection,
        protocol=protocol,
        analysis_policy=_policy(),
        grounding_receipt=grounding,
    )

    assert isinstance(result, NativePacketAssemblyAbstentionV2)
    assert result.primary_blocker == "grounding_abstained"
    assert result.authorizes_typed_effect is False


def test_coherent_assembly_tamper_fails_external_replay() -> None:
    case = EFFECT_CASES["direct_variance"]
    protocol = _protocol()
    projection = _projection(protocol, case["quote"])
    candidate = _candidate(
        projection=projection,
        quote=case["quote"],
        effect_kind="direct_variance",
    )
    grounding = _grounding(
        candidate=candidate,
        projection=projection,
        case=case,
    )
    assembly = assemble_native_packet_v2(
        candidate=candidate,
        projection=projection,
        protocol=protocol,
        analysis_policy=_policy(),
        grounding_receipt=grounding,
    )
    assert isinstance(assembly, NativePacketAssemblyCompletedV2)

    tampered = assembly.model_dump(mode="json")
    typed = tampered["typed_effect"]
    typed["effect"]["estimate"] = "0.70"
    typed["typed_effect_sha256"] = hash_canonical(
        {key: value for key, value in typed.items() if key != "typed_effect_sha256"}
    )
    tampered["typed_effect_sha256"] = typed["typed_effect_sha256"]
    tampered["assembly_receipt_sha256"] = hash_canonical(
        {
            key: value
            for key, value in tampered.items()
            if key != "assembly_receipt_sha256"
        }
    )
    with pytest.raises(
        NativePacketAssemblyV2Error,
        match="external_replay_mismatch",
    ):
        validate_native_packet_assembly_v2(
            assembly=tampered,
            candidate=candidate,
            projection=projection,
            protocol=protocol,
            analysis_policy=_policy(),
            grounding_receipt=grounding,
        )


def test_entity_keys_join_same_cohort_split_distinct_cohort_and_publication() -> None:
    first_case = EFFECT_CASES["direct_standard_error"]
    second_quote = first_case["quote"].replace("0.50", "0.70").replace(
        "0.10", "0.20"
    )
    second_case = {
        **first_case,
        "quote": second_quote,
        "numeric": {
            "effect.estimate": "0.70",
            "effect.standard_error": "0.20",
        },
    }
    third_quote = second_quote.replace("Main cohort", "Secondary cohort").replace(
        "0.70", "0.90"
    )
    third_case = {
        **second_case,
        "quote": third_quote,
        "numeric": {
            "effect.estimate": "0.90",
            "effect.standard_error": "0.20",
        },
    }
    protocol = _protocol()
    projection = _projection(
        protocol,
        first_case["quote"],
        second_quote,
        third_quote,
    )

    def assembled(
        case: dict[str, Any], *, index: int, cohort_label: str
    ) -> NativePacketAssemblyCompletedV2:
        candidate = _candidate(
            projection=projection,
            quote=case["quote"],
            effect_kind="direct_standard_error",
            index=index,
        )
        grounding = _grounding(
            candidate=candidate,
            projection=projection,
            case=case,
            cohort_label=cohort_label,
        )
        result = assemble_native_packet_v2(
            candidate=candidate,
            projection=projection,
            protocol=protocol,
            analysis_policy=_policy(),
            grounding_receipt=grounding,
        )
        assert isinstance(result, NativePacketAssemblyCompletedV2)
        return result

    first = assembled(first_case, index=1, cohort_label="Main cohort")
    same_cohort = assembled(second_case, index=2, cohort_label="Main cohort")
    distinct_cohort = assembled(
        third_case,
        index=3,
        cohort_label="Secondary cohort",
    )

    assert first.typed_effect.study_key == same_cohort.typed_effect.study_key
    assert first.typed_effect.cohort_key == same_cohort.typed_effect.cohort_key
    assert first.typed_effect.treatment_arm_key == (
        same_cohort.typed_effect.treatment_arm_key
    )
    assert first.typed_effect.contrast_key == same_cohort.typed_effect.contrast_key
    assert first.typed_effect.finding_key != same_cohort.typed_effect.finding_key
    assert first.typed_effect.study_key == distinct_cohort.typed_effect.study_key
    assert first.typed_effect.cohort_key != distinct_cohort.typed_effect.cohort_key
    assert first.typed_effect.treatment_arm_key != (
        distinct_cohort.typed_effect.treatment_arm_key
    )

    other_projection = _projection(protocol, first_case["quote"], "Extra source row text.")
    other_candidate = _candidate(
        projection=other_projection,
        quote=first_case["quote"],
        effect_kind="direct_standard_error",
    )
    other_grounding = _grounding(
        candidate=other_candidate,
        projection=other_projection,
        case=first_case,
    )
    other = assemble_native_packet_v2(
        candidate=other_candidate,
        projection=other_projection,
        protocol=protocol,
        analysis_policy=_policy(),
        grounding_receipt=other_grounding,
    )
    assert isinstance(other, NativePacketAssemblyCompletedV2)
    assert first.typed_effect.study_key != other.typed_effect.study_key


def test_analysis_policy_hash_tamper_fails_closed() -> None:
    policy = _policy()
    tampered = deepcopy(policy.model_dump(mode="json"))
    tampered["binary_group_effect_format"] = "risk_ratio"
    with pytest.raises(ValueError, match="self_hash_mismatch"):
        PacketAssemblyAnalysisPolicyV2.model_validate(tampered)


def test_public_analysis_policy_api_is_explicitly_exported() -> None:
    assert "PacketAssemblyAnalysisPolicyV2" in assembly_v2.__all__
    assert "freeze_packet_assembly_analysis_policy_v2" in assembly_v2.__all__
    from_wire_values = freeze_packet_assembly_analysis_policy_v2(
        continuous_group_effect_format="mean_difference",
        binary_group_effect_format="odds_ratio",
    )
    assert from_wire_values.continuous_group_effect_format is (
        EffectFormat.MEAN_DIFFERENCE
    )
