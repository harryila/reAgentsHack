from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

import pytest

from literature_multiverse.anthropic_bounded_generation import (
    compile_anthropic_bounded_schema,
)
from literature_multiverse.effects import EffectFormat
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_candidate_inventory_v2 import (
    MetaSynPassageCandidateV2,
)
from literature_multiverse.metasyn_projection_v2 import (
    FrozenMetaSynProjectionV2,
    freeze_metasyn_projection_v2,
    freeze_projection_v2_lineage_binding,
)
from literature_multiverse.native_bounded_generation import (
    NativeCandidateDescriptor,
)
from literature_multiverse.native_grounding import ResolvedNativeSource, ResolvedSourceLine
from literature_multiverse.native_packet_grounding_v2 import (
    MODEL_OUTCOME_V2_VERSION,
    NORMALIZATION_POLICY_V2_SHA256,
    NativePacketGroundingV2Error,
    PacketGroundingAbstentionReceiptV2,
    PacketGroundingCompletedReceiptV2,
    PacketPassageCandidateBindingV2,
    freeze_packet_candidate_binding_v2,
    freeze_packet_grounding_receipt_v2,
    freeze_packet_grounding_schema_bundle_v2,
    freeze_passage_packet_candidate_binding_v2,
    freeze_passage_packet_grounding_receipt_v2,
    validate_packet_grounding_receipt_v2,
    validate_passage_packet_grounding_receipt_v2,
)
from literature_multiverse.native_question_projection import (
    FrozenSourceProjectionV1,
    freeze_question_projection_spec,
    project_resolved_source_for_question,
)

BASE_QUOTE = (
    "Trial \u03b1 compared Arm A with placebo at week 8 using Hedges g: "
    "estimate -0.50, "
    "95% CI -0.80 to -0.20."
)


def _source(*sections_and_text: tuple[str, str]) -> ResolvedNativeSource:
    lines: list[ResolvedSourceLine] = []
    char_cursor = 0
    byte_cursor = 0
    for line_number, (section, text) in enumerate(sections_and_text, start=1):
        char_end = char_cursor + len(text)
        byte_end = byte_cursor + len(text.encode("utf-8"))
        lines.append(
            ResolvedSourceLine(
                line_id=f"L{line_number}",
                line_number=line_number,
                section=section,
                text=text,
                char_start=char_cursor,
                char_end=char_end,
                utf8_byte_start=byte_cursor,
                utf8_byte_end=byte_end,
            )
        )
        char_cursor = char_end + 1
        byte_cursor = byte_end + 1
    source_text = "\n".join(text for _, text in sections_and_text)
    return ResolvedNativeSource(
        source_kind="metasyn_parquet_row",
        artifact_path="data/cache/private-fixture.parquet",
        artifact_sha256=hashlib.sha256(b"fixture-artifact").hexdigest(),
        source_locator="metasyn://private-fixture/1",
        source_payload_sha256=hash_canonical(
            {"sections": list(sections_and_text)}
        ),
        source_text=source_text,
        lines=lines,
    )


def _projection(*texts: str) -> FrozenSourceProjectionV1:
    source = _source(*(tuple(("Results", text)) for text in texts))
    spec = freeze_question_projection_spec(
        question_id="private-fixture-question",
        population="adults",
        intervention_or_exposure="Arm A",
        comparison="placebo",
        outcome_texts=["Blood pressure"],
        treatment_role="intervention_or_exposure",
        comparator_role="comparator",
        contrast_estimand="not_prespecified_in_protocol_metadata",
    )
    return project_resolved_source_for_question(
        row_id="metasyn-private-fixture:1", source=source, spec=spec
    )


def _candidate(*line_ids: str) -> NativeCandidateDescriptor:
    return NativeCandidateDescriptor(
        candidate_index=1,
        outcome_name="outcome-01",
        effect_kind="direct_confidence_interval",
        line_ids=list(line_ids or ("L1",)),
    )


def _passage_projection(*texts: str) -> FrozenMetaSynProjectionV2:
    upstream = _projection(*texts)
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


def _passage_candidate(
    projection: FrozenMetaSynProjectionV2,
    *,
    effect_kind: str = "direct_confidence_interval",
) -> MetaSynPassageCandidateV2:
    return MetaSynPassageCandidateV2(
        candidate_index=1,
        canonical_outcome_id="outcome-01",
        outcome_concept_quote="Blood pressure",
        effect_kind=effect_kind,
        passage_ids=[projection.selected_passage_anchors[0]],
    )


def _binding(
    projection: FrozenSourceProjectionV1, candidate: NativeCandidateDescriptor | None = None
):
    return freeze_packet_candidate_binding_v2(
        candidate=candidate or _candidate(), projection=projection
    )


def _completed_outcome(
    binding_sha256: str,
    *,
    quote: str = BASE_QUOTE,
    study_label: str = "Trial \u03b1",
) -> dict[str, Any]:
    return {
        "outcome_version": MODEL_OUTCOME_V2_VERSION,
        "packet_status": "completed",
        "candidate_binding_sha256": binding_sha256,
        "evidence_quote": quote,
        "effect_format_token": "Hedges g",
        "effect_unit": None,
        "numeric_claims": [
            {
                "field_path": "effect.ci_level",
                "verbatim_numeric_token": "95%",
                "normalization": "percent_to_proportion",
            },
            {
                "field_path": "effect.ci_lower",
                "verbatim_numeric_token": "-0.80",
                "normalization": "identity",
            },
            {
                "field_path": "effect.ci_upper",
                "verbatim_numeric_token": "-0.20",
                "normalization": "identity",
            },
            {
                "field_path": "effect.estimate",
                "verbatim_numeric_token": "-0.50",
                "normalization": "identity",
            },
            {
                "field_path": "finding.timepoint.value",
                "verbatim_numeric_token": "8",
                "normalization": "identity",
            },
        ],
        "identity_claims": [
            {
                "field_path": "comparator_arm.label",
                "verbatim_identity_text": "placebo",
            },
            {
                "field_path": "study.source_label",
                "verbatim_identity_text": study_label,
            },
            {
                "field_path": "treatment_arm.label",
                "verbatim_identity_text": "Arm A",
            },
        ],
        "timepoint": {
            "kind": "exact",
            "unit": "week",
            "anchor": "week",
            "raw_label": None,
        },
    }


def _minimal_completed(
    binding_sha256: str, *, quote: str, token: str = "0.5"
) -> dict[str, Any]:
    return {
        "outcome_version": MODEL_OUTCOME_V2_VERSION,
        "packet_status": "completed",
        "candidate_binding_sha256": binding_sha256,
        "evidence_quote": quote,
        "effect_format_token": "Hedges g",
        "effect_unit": None,
        "numeric_claims": [
            {
                "field_path": "effect.estimate",
                "verbatim_numeric_token": token,
                "normalization": "identity",
            }
        ],
        "identity_claims": [
            {
                "field_path": "study.source_label",
                "verbatim_identity_text": "Trial \u03b1",
            }
        ],
        "timepoint": {"kind": "not_reported"},
    }


def test_schema_is_offset_free_candidate_bound_and_has_both_fixtures() -> None:
    projection = _projection(BASE_QUOTE)
    binding = _binding(projection)
    bundle = freeze_packet_grounding_schema_bundle_v2(binding=binding)
    serialized = json.dumps(bundle.model_response_schema, sort_keys=True)

    assert "quote_start" not in serialized
    assert "quote_end" not in serialized
    assert "token_start" not in serialized
    assert bundle.model_response_schema[
        "x-literature-multiverse-model-authored-offsets"
    ] is False
    assert binding.binding_sha256 in serialized
    assert bundle.completed_fixture["packet_status"] == "completed"
    assert bundle.abstaining_fixture["packet_status"] == "unable_to_complete"
    assert bundle.fixtures_are_synthetic is True
    assert bundle.scientific_authority is False


def test_all_effect_family_schemas_compile_for_anthropic_without_weakening() -> None:
    projection = _projection(BASE_QUOTE)
    passage_projection = _passage_projection(BASE_QUOTE)
    effect_kinds = (
        "direct_standard_error",
        "direct_variance",
        "direct_confidence_interval",
        "continuous_group_statistics",
        "binary_group_statistics",
    )
    for effect_kind in effect_kinds:
        candidate = NativeCandidateDescriptor(
            candidate_index=1,
            outcome_name="outcome-01",
            effect_kind=effect_kind,
            line_ids=["L1"],
        )
        bundle = freeze_packet_grounding_schema_bundle_v2(
            binding=_binding(projection, candidate)
        )
        compiled = compile_anthropic_bounded_schema(
            original_schema=bundle.model_response_schema,
            full_acceptance_schema_sha256=bundle.model_response_schema_sha256,
        )

        assert compiled.full_acceptance_schema_sha256 == (
            bundle.model_response_schema_sha256
        )
        assert compiled.wire_optional_parameter_count <= 24
        assert compiled.wire_union_parameter_count <= 16

        passage_candidate = _passage_candidate(
            passage_projection,
            effect_kind=effect_kind,
        )
        passage_binding = freeze_passage_packet_candidate_binding_v2(
            candidate=passage_candidate,
            projection=passage_projection,
        )
        passage_bundle = freeze_packet_grounding_schema_bundle_v2(
            binding=passage_binding
        )
        passage_compiled = compile_anthropic_bounded_schema(
            original_schema=passage_bundle.model_response_schema,
            full_acceptance_schema_sha256=(
                passage_bundle.model_response_schema_sha256
            ),
        )

        assert passage_compiled.full_acceptance_schema_sha256 == (
            passage_bundle.model_response_schema_sha256
        )
        assert passage_compiled.wire_optional_parameter_count <= 24
        assert passage_compiled.wire_union_parameter_count <= 16


def test_completed_receipt_derives_exact_character_and_utf8_offsets() -> None:
    projection = _projection(BASE_QUOTE)
    candidate = _candidate()
    binding = _binding(projection, candidate)
    outcome = _completed_outcome(binding.binding_sha256)

    first = freeze_packet_grounding_receipt_v2(
        model_outcome=outcome, candidate=candidate, projection=projection
    )
    second = freeze_packet_grounding_receipt_v2(
        model_outcome=outcome, candidate=candidate, projection=projection
    )

    assert isinstance(first, PacketGroundingCompletedReceiptV2)
    assert first == second
    assert first.claim_release_authority is False
    assert first.evidence_receipt.evidence_quote == BASE_QUOTE
    assert len(first.numeric_receipts) == 5
    by_path = {
        item.normalization_receipt.field_path: item for item in first.numeric_receipts
    }
    estimate = by_path["effect.estimate"]
    assert (
        BASE_QUOTE[estimate.token_start_in_quote : estimate.token_end_exclusive_in_quote]
        == "-0.50"
    )
    assert estimate.token_source_utf8_byte_start > estimate.token_source_char_start
    ci_level = by_path["effect.ci_level"].normalization_receipt
    assert ci_level.normalization == "percent_to_proportion"
    assert ci_level.normalized_numeric_lexeme == "0.95"
    assert ci_level.normalization_policy_sha256 == NORMALIZATION_POLICY_V2_SHA256
    assert validate_packet_grounding_receipt_v2(
        receipt=first,
        model_outcome=outcome,
        candidate=candidate,
        projection=projection,
    ) == first


def test_direct_mean_difference_format_and_unit_are_exactly_grounded() -> None:
    quote = (
        "Trial Alpha reports a mean difference estimate 2.0 mmHg with "
        "standard error 0.5."
    )
    projection = _projection(quote)
    candidate = NativeCandidateDescriptor(
        candidate_index=1,
        outcome_name="outcome-01",
        effect_kind="direct_standard_error",
        line_ids=["L1"],
    )
    binding = _binding(projection, candidate)
    outcome = {
        "outcome_version": MODEL_OUTCOME_V2_VERSION,
        "packet_status": "completed",
        "candidate_binding_sha256": binding.binding_sha256,
        "evidence_quote": quote,
        "effect_format_token": "mean difference",
        "effect_unit": "mmHg",
        "numeric_claims": [
            {
                "field_path": "effect.estimate",
                "verbatim_numeric_token": "2.0",
                "normalization": "identity",
            },
            {
                "field_path": "effect.standard_error",
                "verbatim_numeric_token": "0.5",
                "normalization": "identity",
            },
        ],
        "identity_claims": [],
        "timepoint": {"kind": "not_reported"},
    }

    receipt = freeze_packet_grounding_receipt_v2(
        model_outcome=outcome,
        candidate=candidate,
        projection=projection,
    )

    assert isinstance(receipt, PacketGroundingCompletedReceiptV2)
    assert receipt.effect_format_receipt is not None
    assert receipt.effect_format_receipt.effect_format is EffectFormat.MEAN_DIFFERENCE
    assert receipt.effect_format_receipt.verbatim_effect_format_token == (
        "mean difference"
    )
    unit_receipts = [
        item for item in receipt.identity_receipts if item.field_path == "effect.unit"
    ]
    assert len(unit_receipts) == 1
    assert unit_receipts[0].verbatim_identity_text == "mmHg"

    missing_unit = deepcopy(outcome)
    missing_unit["effect_unit"] = None
    with pytest.raises(
        NativePacketGroundingV2Error,
        match="effect_format_unit_shape_mismatch",
    ):
        freeze_packet_grounding_receipt_v2(
            model_outcome=missing_unit,
            candidate=candidate,
            projection=projection,
        )


def test_abstention_is_value_free_hashed_and_replayable() -> None:
    projection = _projection(BASE_QUOTE)
    candidate = _candidate()
    binding = _binding(projection, candidate)
    outcome = {
        "outcome_version": MODEL_OUTCOME_V2_VERSION,
        "packet_status": "unable_to_complete",
        "candidate_binding_sha256": binding.binding_sha256,
        "reason": "source_support_incomplete",
    }

    receipt = freeze_packet_grounding_receipt_v2(
        model_outcome=outcome, candidate=candidate, projection=projection
    )

    assert isinstance(receipt, PacketGroundingAbstentionReceiptV2)
    assert receipt.status == "unable_to_complete"
    assert receipt.claim_release_authority is False
    serialized = receipt.model_dump_json()
    assert "evidence_quote" not in serialized
    assert "numeric_claims" not in serialized
    assert validate_packet_grounding_receipt_v2(
        receipt=receipt,
        model_outcome=outcome,
        candidate=candidate,
        projection=projection,
    ) == receipt


def test_model_cannot_submit_offsets_or_swap_candidate_binding() -> None:
    projection = _projection(BASE_QUOTE)
    candidate = _candidate()
    binding = _binding(projection, candidate)
    with_offset = _completed_outcome(binding.binding_sha256)
    with_offset["numeric_claims"][0]["quote_start"] = "0"

    with pytest.raises(
        NativePacketGroundingV2Error,
        match="model_outcome_schema_invalid",
    ):
        freeze_packet_grounding_receipt_v2(
            model_outcome=with_offset, candidate=candidate, projection=projection
        )

    swapped = _completed_outcome("0" * 64)
    with pytest.raises(
        NativePacketGroundingV2Error,
        match="model_outcome_schema_invalid",
    ):
        freeze_packet_grounding_receipt_v2(
            model_outcome=swapped, candidate=candidate, projection=projection
        )


def test_not_reported_timepoint_forbids_all_details_and_numeric_paths() -> None:
    projection = _projection(BASE_QUOTE)
    candidate = _candidate()
    binding = _binding(projection, candidate)
    invalid = _minimal_completed(
        binding.binding_sha256, quote=BASE_QUOTE, token="-0.50"
    )
    invalid["timepoint"] = {"kind": "not_reported", "anchor": "week"}

    with pytest.raises(
        NativePacketGroundingV2Error,
        match="model_outcome_schema_invalid",
    ):
        freeze_packet_grounding_receipt_v2(
            model_outcome=invalid, candidate=candidate, projection=projection
        )

    wrong_numeric_shape = _completed_outcome(binding.binding_sha256)
    wrong_numeric_shape["timepoint"] = {"kind": "not_reported"}
    with pytest.raises(
        NativePacketGroundingV2Error,
        match="model_outcome_contract_invalid",
    ):
        freeze_packet_grounding_receipt_v2(
            model_outcome=wrong_numeric_shape,
            candidate=candidate,
            projection=projection,
        )


def test_identity_text_must_be_exactly_present_in_frozen_projection() -> None:
    projection = _projection(BASE_QUOTE)
    candidate = _candidate()
    binding = _binding(projection, candidate)
    absent = _completed_outcome(binding.binding_sha256, study_label="Ghost Trial")

    with pytest.raises(
        NativePacketGroundingV2Error,
        match=r"identity_text_absent:study\.source_label",
    ):
        freeze_packet_grounding_receipt_v2(
            model_outcome=absent, candidate=candidate, projection=projection
        )


def test_quote_and_each_numeric_token_must_be_unique_without_repair() -> None:
    duplicate_quote = "Trial \u03b1 reports Hedges g estimate 0.5."
    projection = _projection(duplicate_quote, duplicate_quote)
    candidate = _candidate("L1", "L2")
    binding = _binding(projection, candidate)
    duplicate_quote_outcome = _minimal_completed(
        binding.binding_sha256, quote=duplicate_quote
    )

    with pytest.raises(
        NativePacketGroundingV2Error, match="evidence_quote_not_unique"
    ):
        freeze_packet_grounding_receipt_v2(
            model_outcome=duplicate_quote_outcome,
            candidate=candidate,
            projection=projection,
        )

    duplicate_token_quote = "Trial \u03b1 reports Hedges g 0.5 and repeats 0.5."
    projection = _projection(duplicate_token_quote)
    candidate = _candidate()
    binding = _binding(projection, candidate)
    duplicate_token_outcome = _minimal_completed(
        binding.binding_sha256, quote=duplicate_token_quote
    )
    with pytest.raises(
        NativePacketGroundingV2Error,
        match=r"numeric_token_not_unique:effect\.estimate",
    ):
        freeze_packet_grounding_receipt_v2(
            model_outcome=duplicate_token_outcome,
            candidate=candidate,
            projection=projection,
        )


def test_percent_normalization_is_explicit_field_limited_and_hashed() -> None:
    quote = "Trial \u03b1 reports Hedges g 95% as the estimate."
    projection = _projection(quote)
    candidate = _candidate()
    binding = _binding(projection, candidate)
    forbidden = _minimal_completed(binding.binding_sha256, quote=quote, token="95%")
    forbidden["numeric_claims"][0]["normalization"] = "percent_to_proportion"
    with pytest.raises(
        NativePacketGroundingV2Error,
        match="model_outcome_contract_invalid",
    ):
        freeze_packet_grounding_receipt_v2(
            model_outcome=forbidden, candidate=candidate, projection=projection
        )

    implicit = _minimal_completed(binding.binding_sha256, quote=quote, token="95%")
    with pytest.raises(
        NativePacketGroundingV2Error,
        match="model_outcome_contract_invalid",
    ):
        freeze_packet_grounding_receipt_v2(
            model_outcome=implicit, candidate=candidate, projection=projection
        )


def test_candidate_must_bind_only_exposed_lines_and_allowed_outcome() -> None:
    projection = _projection(BASE_QUOTE)
    with pytest.raises(
        NativePacketGroundingV2Error, match="candidate_line_not_exposed"
    ):
        _binding(projection, _candidate("L2"))
    wrong_outcome = NativeCandidateDescriptor(
        candidate_index=1,
        outcome_name="outcome-99",
        effect_kind="direct_confidence_interval",
        line_ids=["L1"],
    )
    with pytest.raises(
        NativePacketGroundingV2Error, match="candidate_outcome_not_allowed"
    ):
        _binding(projection, wrong_outcome)


def test_simple_and_coherently_rehashed_receipt_tampering_fail_closed() -> None:
    text = BASE_QUOTE + " Trial \u03b1 was registered prospectively."
    projection = _projection(text)
    candidate = _candidate()
    binding = _binding(projection, candidate)
    outcome = _completed_outcome(binding.binding_sha256, quote=BASE_QUOTE)
    receipt = freeze_packet_grounding_receipt_v2(
        model_outcome=outcome, candidate=candidate, projection=projection
    )
    assert isinstance(receipt, PacketGroundingCompletedReceiptV2)

    simple = receipt.model_dump(mode="json")
    simple["numeric_receipts"][0]["normalization_receipt"][
        "normalization_policy_sha256"
    ] = "0" * 64
    with pytest.raises(
        NativePacketGroundingV2Error, match="saved_receipt_invalid"
    ):
        validate_packet_grounding_receipt_v2(
            receipt=simple,
            model_outcome=outcome,
            candidate=candidate,
            projection=projection,
        )

    coherent = receipt.model_dump(mode="json")
    identity = coherent["identity_receipts"][1]
    identity["occurrence_count"] += 1
    identity["identity_receipt_sha256"] = hash_canonical(
        {
            key: value
            for key, value in identity.items()
            if key != "identity_receipt_sha256"
        }
    )
    coherent["receipt_sha256"] = hash_canonical(
        {key: value for key, value in coherent.items() if key != "receipt_sha256"}
    )
    with pytest.raises(
        NativePacketGroundingV2Error,
        match="external_replay_mismatch",
    ):
        validate_packet_grounding_receipt_v2(
            receipt=coherent,
            model_outcome=outcome,
            candidate=candidate,
            projection=projection,
        )


def test_raw_model_strings_are_not_trimmed_or_casefolded() -> None:
    projection = _projection(BASE_QUOTE)
    candidate = _candidate()
    binding = _binding(projection, candidate)
    padded = _completed_outcome(binding.binding_sha256)
    padded["identity_claims"][1]["verbatim_identity_text"] = " Trial \u03b1"
    with pytest.raises(
        NativePacketGroundingV2Error,
        match="model_outcome_contract_invalid",
    ):
        freeze_packet_grounding_receipt_v2(
            model_outcome=padded, candidate=candidate, projection=projection
        )

    case_changed = _completed_outcome(
        binding.binding_sha256, study_label="trial \u03b1"
    )
    with pytest.raises(
        NativePacketGroundingV2Error,
        match=r"identity_text_absent:study\.source_label",
    ):
        freeze_packet_grounding_receipt_v2(
            model_outcome=case_changed,
            candidate=candidate,
            projection=projection,
        )


def test_replay_rejects_model_outcome_substitution() -> None:
    projection = _projection(BASE_QUOTE)
    candidate = _candidate()
    binding = _binding(projection, candidate)
    outcome = _completed_outcome(binding.binding_sha256)
    receipt = freeze_packet_grounding_receipt_v2(
        model_outcome=outcome, candidate=candidate, projection=projection
    )
    substituted = deepcopy(outcome)
    substituted["identity_claims"][0]["verbatim_identity_text"] = "placebo "

    with pytest.raises(NativePacketGroundingV2Error):
        validate_packet_grounding_receipt_v2(
            receipt=receipt,
            model_outcome=substituted,
            candidate=candidate,
            projection=projection,
        )


def test_passage_candidate_binding_and_grounding_are_end_to_end_replayable() -> None:
    projection = _passage_projection(BASE_QUOTE)
    candidate = _passage_candidate(projection)
    binding = freeze_passage_packet_candidate_binding_v2(
        candidate=candidate,
        projection=projection,
    )
    outcome = _completed_outcome(binding.binding_sha256)

    receipt = freeze_passage_packet_grounding_receipt_v2(
        model_outcome=outcome,
        candidate=candidate,
        projection=projection,
    )

    assert isinstance(binding, PacketPassageCandidateBindingV2)
    assert binding.passage_ids == candidate.passage_ids
    assert isinstance(receipt, PacketGroundingCompletedReceiptV2)
    assert receipt.candidate_binding == binding
    assert receipt.evidence_receipt.passage_anchor == candidate.passage_ids[0]
    assert receipt.evidence_receipt.source_origin_sha256 is not None
    assert receipt.evidence_receipt.source_occurrence_count == 1
    assert (
        receipt.evidence_receipt.quote_source_utf8_byte_end_exclusive
        > receipt.evidence_receipt.quote_source_char_end_exclusive
    )
    assert validate_passage_packet_grounding_receipt_v2(
        receipt=receipt,
        model_outcome=outcome,
        candidate=candidate,
        projection=projection,
    ) == receipt


def test_passage_candidate_abstention_remains_value_free_and_replayable() -> None:
    projection = _passage_projection(BASE_QUOTE)
    candidate = _passage_candidate(projection)
    binding = freeze_passage_packet_candidate_binding_v2(
        candidate=candidate,
        projection=projection,
    )
    outcome = {
        "outcome_version": MODEL_OUTCOME_V2_VERSION,
        "packet_status": "unable_to_complete",
        "candidate_binding_sha256": binding.binding_sha256,
        "reason": "source_support_incomplete",
    }
    receipt = freeze_passage_packet_grounding_receipt_v2(
        model_outcome=outcome,
        candidate=candidate,
        projection=projection,
    )

    assert isinstance(receipt, PacketGroundingAbstentionReceiptV2)
    assert "evidence_quote" not in receipt.model_dump_json()
    assert validate_passage_packet_grounding_receipt_v2(
        receipt=receipt,
        model_outcome=outcome,
        candidate=candidate,
        projection=projection,
    ) == receipt


def test_passage_candidate_rejects_unexposed_anchor_and_identity() -> None:
    projection = _passage_projection(BASE_QUOTE)
    missing = MetaSynPassageCandidateV2(
        candidate_index=1,
        canonical_outcome_id="outcome-01",
        outcome_concept_quote="Blood pressure",
        effect_kind="direct_confidence_interval",
        passage_ids=["p2-" + "f" * 64],
    )
    with pytest.raises(
        NativePacketGroundingV2Error,
        match="candidate_passage_not_exposed",
    ):
        freeze_passage_packet_candidate_binding_v2(
            candidate=missing,
            projection=projection,
        )

    candidate = _passage_candidate(projection)
    binding = freeze_passage_packet_candidate_binding_v2(
        candidate=candidate,
        projection=projection,
    )
    absent_identity = _completed_outcome(
        binding.binding_sha256,
        study_label="Ghost Trial",
    )
    with pytest.raises(
        NativePacketGroundingV2Error,
        match=r"identity_text_absent:study\.source_label",
    ):
        freeze_passage_packet_grounding_receipt_v2(
            model_outcome=absent_identity,
            candidate=candidate,
            projection=projection,
        )


def test_passage_receipt_records_deduplicated_source_occurrences_and_tamper() -> None:
    projection = _passage_projection(BASE_QUOTE, BASE_QUOTE)
    candidate = _passage_candidate(projection)
    binding = freeze_passage_packet_candidate_binding_v2(
        candidate=candidate,
        projection=projection,
    )
    outcome = _completed_outcome(binding.binding_sha256)
    receipt = freeze_passage_packet_grounding_receipt_v2(
        model_outcome=outcome,
        candidate=candidate,
        projection=projection,
    )
    assert isinstance(receipt, PacketGroundingCompletedReceiptV2)
    assert receipt.evidence_receipt.source_occurrence_count == 2

    tampered = receipt.model_dump(mode="json")
    evidence = tampered["evidence_receipt"]
    evidence["passage_anchor"] = "p2-" + "f" * 64
    evidence["evidence_receipt_sha256"] = hash_canonical(
        {
            key: value
            for key, value in evidence.items()
            if key != "evidence_receipt_sha256"
        }
    )
    tampered["receipt_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    )
    with pytest.raises(
        NativePacketGroundingV2Error,
        match="saved_receipt_invalid",
    ):
        validate_passage_packet_grounding_receipt_v2(
            receipt=tampered,
            model_outcome=outcome,
            candidate=candidate,
            projection=projection,
        )
