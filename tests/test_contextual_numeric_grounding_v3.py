from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.private_cache_support import (
    HOSTED_ADAPTER_STALE_CODES,
    TYPED_PILOT_STALE_CODES,
    require_private_cache,
    skip_when_historical_replay_is_stale,
)

from literature_multiverse.contextual_numeric_grounding_v3 import (
    ContextualClaimV3,
    ContextualGroundedClaimV3,
    ContextualGroundedEffectV3,
    ContextualGroundingOfflineFeasibilitySuiteV3,
    ContextualNumericGroundingV3Error,
    ContextualSourcePassageV3,
    freeze_contextual_grounding_offline_feasibility_suite_v3,
    freeze_contextual_provider_binding_v3,
    ground_contextual_claim_v3,
    ground_contextual_outcome_v3,
    project_contextual_grounded_outcome_v3,
    validate_contextual_grounding_offline_feasibility_suite_v3,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_bounded_hosted_runtime import MetaSynHostedRuntimeError
from literature_multiverse.metasyn_typed_pilot import MetaSynTypedPilotError

ROOT = Path(__file__).resolve().parents[1]
SUITE_ARTIFACT = (
    ROOT / "artifacts/diagnostics/contextual-grounding-offline-feasibility-suite-v3.json"
)


@pytest.fixture(scope="module")
def suite() -> ContextualGroundingOfflineFeasibilitySuiteV3:
    return ContextualGroundingOfflineFeasibilitySuiteV3.model_validate(
        json.loads(SUITE_ARTIFACT.read_text(encoding="utf-8"))
    )


@pytest.mark.private_cache
def test_live_freeze_contextual_grounding_offline_feasibility_suite_v3_replays_tracked_artifact_or_is_stale(  # noqa: E501
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
) -> None:
    require_private_cache(
        "data/cache/metasyn/passage-hosted-yield-v2",
        "data/cache/metasyn/bounded-anthropic-yield-v5",
    )
    rebuilt = skip_when_historical_replay_is_stale(
        lambda: freeze_contextual_grounding_offline_feasibility_suite_v3(repository_root=ROOT),
        stale_errors=(MetaSynTypedPilotError, MetaSynHostedRuntimeError),
        stale_codes=TYPED_PILOT_STALE_CODES | HOSTED_ADAPTER_STALE_CODES,
    )
    assert rebuilt.suite_sha256 == suite.suite_sha256


def _receipt(
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
    row: int,
    candidate: int | None = None,
):
    return next(
        item
        for item in suite.receipts
        if item.row_ordinal == row
        and (
            candidate is None
            or item.provider_binding.context.candidate.candidate_index == candidate
        )
    )


def test_external_replay_freezes_three_non_authorizing_witnesses(
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
) -> None:
    validated = validate_contextual_grounding_offline_feasibility_suite_v3(
        suite=suite,
        repository_root=ROOT,
        external_replay=False,
    )
    assert validated == suite
    assert suite.offline_witness_count == 3
    assert suite.contextual_grounding_completed_count == 3
    assert suite.typed_graph_mechanics_completed_count == 2
    assert not suite.provider_calls_made
    assert not suite.extraction_accuracy_authority
    assert not suite.scientific_effectiveness_authority
    assert not suite.calibration_authority
    assert not suite.claim_release_authority


def test_row16_preserves_unicode_range_delimiter_without_sign_interpretation(
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
) -> None:
    receipt = _receipt(suite, 16)
    effect = receipt.grounded_effect
    assert effect.estimate == "134.4"
    assert effect.ci_lower == "18.0"
    assert effect.ci_upper == "1005"
    assert effect.ci_level == "0.95"
    assert effect.unicode_range is not None
    assert effect.unicode_range.exact_range_text == "18.0\u20131005"
    assert effect.unicode_range.delimiter == "\u2013"
    assert not effect.unicode_range.delimiter_interpreted_as_numeric_sign
    assert not effect.unicode_range.punctuation_normalized
    assert receipt.native_projection.status == "blocked_missing_exact_identity"


def test_duplicate_numeric_token_is_disambiguated_by_unique_local_context(
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
) -> None:
    receipt = _receipt(suite, 17, 3)
    passage = next(
        item
        for item in receipt.passages
        if item.passage_id == receipt.model_outcome.endpoint_passage_id
    )
    # 96 occurs for the 400-mg group and again for placebo; the local context binds
    # the selected placebo total without claiming global token uniqueness.
    assert passage.passage_text.count("96") == 2
    control_total = next(
        item for item in receipt.groundings if item.claim.field_path == "effect.control_total"
    )
    assert control_total.claim.context == "vs 1 of 96 ("
    assert control_total.claim.context.count("96") == 1
    assert (
        passage.passage_text[
            control_total.token_char_start_in_passage : (
                control_total.token_char_end_exclusive_in_passage
            )
        ]
        == "96"
    )


def test_context_forgery_and_token_context_mismatch_fail_closed() -> None:
    with pytest.raises(ValidationError, match="context_not_unique_in_support_quote"):
        ContextualClaimV3(
            field_path="effect.estimate",
            passage_id="p2-" + "0" * 64,
            support_quote="odds ratio 134.4",
            context="risk ratio 134.4",
            token="134.4",
            normalization="decimal_identity",
        )
    with pytest.raises(ValidationError, match="token_not_unique_in_context"):
        ContextualClaimV3(
            field_path="effect.estimate",
            passage_id="p2-" + "0" * 64,
            support_quote="odds ratio 134.4",
            context="odds ratio 134.4",
            token="1005",
            normalization="decimal_identity",
        )


def test_unicode_dash_cannot_be_smuggled_into_numeric_token(
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
) -> None:
    receipt = _receipt(suite, 16)
    lower = next(item for item in receipt.groundings if item.claim.field_path == "effect.ci_lower")
    passage = next(item for item in receipt.passages if item.passage_id == lower.claim.passage_id)
    smuggled = ContextualClaimV3(
        field_path="effect.ci_upper",
        passage_id=lower.claim.passage_id,
        support_quote=lower.claim.support_quote,
        context="18.0\u20131005",
        token="\u20131005",
        normalization="decimal_identity",
    )
    with pytest.raises(ContextualNumericGroundingV3Error, match="non_ascii_numeric_sign"):
        ground_contextual_claim_v3(claim=smuggled, passage=passage)


@pytest.mark.parametrize(
    ("field_path", "ambiguous_character"),
    [
        ("effect.ci_lower", "\u2013"),
        ("effect.ci_upper", "\u2013"),
        ("effect.ci_lower", "\u2212"),
        ("effect.ci_upper", "\u2010"),
        ("effect.ci_upper", "\u2011"),
        ("effect.ci_upper", "\u2014"),
    ],
)
def test_unicode_minus_hyphen_and_dash_are_never_numeric_signs(
    field_path: str,
    ambiguous_character: str,
) -> None:
    token = ambiguous_character + "1005"
    text = "reported interval boundary " + token + "."
    passage_payload = {
        "passage_version": "contextual-source-passage-v3",
        "passage_id": "p2-" + "1" * 64,
        "passage_text": text,
        "passage_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "passage_lineage_sha256": "2" * 64,
        "source_text_sha256": "3" * 64,
        "source_locator": "synthetic:test-only-unicode-sign",
        "section": "Results",
        "line_id": "L1",
        "source_char_start": 0,
        "source_char_end_exclusive": len(text),
        "source_utf8_byte_start": 0,
        "source_utf8_byte_end_exclusive": len(text.encode("utf-8")),
        "exact_source_occurrence_count": 1,
        "single_exact_origin": True,
    }
    passage = ContextualSourcePassageV3.model_validate(
        {
            **passage_payload,
            "passage_sha256": hash_canonical(passage_payload),
        }
    )
    claim = ContextualClaimV3(
        field_path=field_path,
        passage_id=passage.passage_id,
        support_quote=text,
        context="boundary " + token + ".",
        token=token,
        normalization="decimal_identity",
    )
    with pytest.raises(ContextualNumericGroundingV3Error, match="non_ascii_numeric_sign"):
        ground_contextual_claim_v3(claim=claim, passage=passage)


def test_trusted_char_and_utf8_offsets_point_to_exact_source_bytes(
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
) -> None:
    receipt = _receipt(suite, 16)
    passage_map = {item.passage_id: item for item in receipt.passages}
    for grounding in receipt.groundings:
        passage = passage_map[grounding.claim.passage_id]
        assert (
            passage.passage_text[
                grounding.token_char_start_in_passage : (
                    grounding.token_char_end_exclusive_in_passage
                )
            ]
            == grounding.claim.token
        )
        byte_start = grounding.token_source_utf8_byte_start - passage.source_utf8_byte_start
        byte_end = grounding.token_source_utf8_byte_end_exclusive - passage.source_utf8_byte_start
        assert passage.passage_text.encode("utf-8")[byte_start:byte_end].decode() == (
            grounding.claim.token
        )


def test_offset_tamper_is_rejected_by_intrinsic_and_receipt_replay(
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
) -> None:
    receipt = _receipt(suite, 16)
    grounding = receipt.groundings[0]
    raw = grounding.model_dump(mode="json")
    raw["token_char_end_exclusive_in_passage"] += 1
    raw["grounding_sha256"] = hash_canonical(
        {key: value for key, value in raw.items() if key != "grounding_sha256"}
    )
    with pytest.raises(ValidationError, match="grounded_char_lengths_mismatch"):
        ContextualGroundedClaimV3.model_validate(raw)


def test_source_drift_rejected_even_with_coherently_rehashed_source_passage(
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
) -> None:
    receipt = _receipt(suite, 16)
    passages = [item.model_dump(mode="json") for item in receipt.passages]
    first = passages[0]
    first["passage_text"] = first["passage_text"].replace("134.4", "134.5")
    first["passage_text_sha256"] = hashlib.sha256(first["passage_text"].encode("utf-8")).hexdigest()
    first["source_char_end_exclusive"] = first["source_char_start"] + len(first["passage_text"])
    first["source_utf8_byte_end_exclusive"] = first["source_utf8_byte_start"] + len(
        first["passage_text"].encode("utf-8")
    )
    first["passage_sha256"] = hash_canonical(
        {key: value for key, value in first.items() if key != "passage_sha256"}
    )
    drifted = [ContextualSourcePassageV3.model_validate(item) for item in passages]
    with pytest.raises(ContextualNumericGroundingV3Error, match="source_passage_drift"):
        ground_contextual_outcome_v3(
            provider_binding=receipt.provider_binding,
            raw_outcome=receipt.model_outcome,
            passages=drifted,
        )


def test_endpoint_effect_format_and_timepoint_mismatches_fail_closed(
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
) -> None:
    receipt = _receipt(suite, 16)
    passages = receipt.passages

    endpoint = receipt.model_outcome.model_dump(mode="json")
    endpoint["endpoint_quote"] = endpoint["endpoint_quote"].replace(
        "primary endpoint", "secondary endpoint"
    )
    with pytest.raises(ContextualNumericGroundingV3Error, match="model_outcome_invalid"):
        ground_contextual_outcome_v3(
            provider_binding=receipt.provider_binding,
            raw_outcome=endpoint,
            passages=passages,
        )

    effect_format = receipt.model_outcome.model_dump(mode="json")
    effect_format["effect_format_token"] = "risk ratio"
    with pytest.raises(ContextualNumericGroundingV3Error, match="model_outcome_invalid"):
        ground_contextual_outcome_v3(
            provider_binding=receipt.provider_binding,
            raw_outcome=effect_format,
            passages=passages,
        )

    timepoint = receipt.model_outcome.model_dump(mode="json")
    claim = next(
        item for item in timepoint["claims"] if item["field_path"] == "finding.timepoint.anchor"
    )
    claim["token"] = "at"
    with pytest.raises(ContextualNumericGroundingV3Error, match="timepoint_semantics_mismatch"):
        ground_contextual_outcome_v3(
            provider_binding=receipt.provider_binding,
            raw_outcome=timepoint,
            passages=passages,
        )


def test_candidate_semantic_substitution_rejects_other_exact_source_numbers(
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
) -> None:
    direct = _receipt(suite, 16)
    wrong_direct = direct.model_outcome.model_dump(mode="json")
    estimate = next(
        item for item in wrong_direct["claims"] if item["field_path"] == "effect.estimate"
    )
    estimate["context"] = "primary endpoint, 41.9% of"
    estimate["token"] = "41.9"
    with pytest.raises(
        ContextualNumericGroundingV3Error,
        match="candidate_semantic_substitution",
    ):
        ground_contextual_outcome_v3(
            provider_binding=direct.provider_binding,
            raw_outcome=wrong_direct,
            passages=direct.passages,
        )

    binary = _receipt(suite, 17, 3)
    wrong_binary = binary.model_outcome.model_dump(mode="json")
    treatment_events = next(
        item for item in wrong_binary["claims"] if item["field_path"] == "effect.treatment_events"
    )
    treatment_events["context"] = "35 of 96 ("
    treatment_events["token"] = "35"
    with pytest.raises(
        ContextualNumericGroundingV3Error,
        match="binary_pair_not_exact:treatment",
    ):
        ground_contextual_outcome_v3(
            provider_binding=binary.provider_binding,
            raw_outcome=wrong_binary,
            passages=binary.passages,
        )


def test_binary_pair_coherence_rejects_same_total_from_wrong_arm_occurrence(
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
) -> None:
    binary = _receipt(suite, 17, 3)
    wrong_binary = binary.model_outcome.model_dump(mode="json")
    control_total = next(
        item for item in wrong_binary["claims"] if item["field_path"] == "effect.control_total"
    )
    assert control_total["token"] == "96"
    control_total["context"] = "35 of 96 ("

    with pytest.raises(
        ContextualNumericGroundingV3Error,
        match="binary_pair_not_exact:control",
    ):
        ground_contextual_outcome_v3(
            provider_binding=binary.provider_binding,
            raw_outcome=wrong_binary,
            passages=binary.passages,
        )


def test_direct_odds_ratio_and_interval_bounds_must_all_be_strictly_positive(
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
) -> None:
    effect = _receipt(suite, 16).grounded_effect.model_dump(mode="json")
    effect["ci_lower"] = "-18.0"
    effect["effect_sha256"] = hash_canonical(
        {key: value for key, value in effect.items() if key != "effect_sha256"}
    )
    with pytest.raises(ValidationError, match="direct_interval_invalid"):
        ContextualGroundedEffectV3.model_validate(effect)


def test_provider_schema_and_prompt_are_closed_deterministic_and_zero_call(
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
) -> None:
    for receipt in suite.receipts:
        replayed = freeze_contextual_provider_binding_v3(context=receipt.provider_binding.context)
        assert replayed == receipt.provider_binding
        assert not replayed.provider_calls_made
        for branch in replayed.provider_schema["oneOf"]:
            assert branch["additionalProperties"] is False
        completed = replayed.provider_schema["oneOf"][0]
        assert completed["properties"]["claims"]["items"]["additionalProperties"] is False


def test_row17_native_graph_and_quantitative_mechanics_are_exact_but_non_authorizing(
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
) -> None:
    receipt = _receipt(suite, 17, 3)
    projection = receipt.native_projection
    assert projection.status == "typed_graph_mechanics_completed"
    assert projection.fragment is not None
    assert projection.fragment.graph is not None
    estimate = projection.fragment.graph.outcome_estimates[0]
    assert estimate.effect.treatment_events == 39
    assert estimate.effect.treatment_total == 97
    assert estimate.effect.control_events == 1
    assert estimate.effect.control_total == 96
    assert estimate.effect.effect_format.value == "odds_ratio"
    assert projection.harmonization_result["status"] == "estimable"
    assert projection.title_abstract_only_not_release_grade
    assert not projection.synthesis_input_authority
    assert not projection.scientific_synthesis_authority
    assert not projection.claim_release_authority


def test_fresh_outcome_public_projection_builds_runtime_graph_without_authority(
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
) -> None:
    fixture = _receipt(suite, 17, 3)
    runtime_pipeline_sha256 = "a" * 64
    provider_execution_binding_sha256 = "b" * 64
    outcome, groundings, effect, core_sha256, runtime_binding_sha256, projection = (
        project_contextual_grounded_outcome_v3(
            fixture_receipt=fixture,
            raw_outcome=fixture.model_outcome.model_dump(mode="json"),
            runtime_pipeline_sha256=runtime_pipeline_sha256,
            provider_execution_binding_sha256=provider_execution_binding_sha256,
        )
    )
    assert outcome == fixture.model_outcome
    assert groundings == fixture.groundings
    assert effect == fixture.grounded_effect
    assert core_sha256 == fixture.grounding_core_sha256
    assert projection.status == "typed_graph_mechanics_completed"
    assert projection.outcome_origin == "runtime_outcome_supplied_by_caller"
    assert projection.runtime_pipeline_sha256 == runtime_pipeline_sha256
    assert projection.provider_execution_binding_sha256 == provider_execution_binding_sha256
    assert projection.contextual_grounding_core_sha256 == core_sha256
    assert projection.runtime_grounding_binding_sha256 == runtime_binding_sha256
    assert projection.fragment is not None
    assert projection.fragment.graph is not None
    assert projection.fragment.pipeline_fingerprint_sha256 == runtime_pipeline_sha256
    assert projection.fragment.extraction_context_sha256 == runtime_binding_sha256
    assert projection.fragment.grounding_receipt_sha256 == runtime_binding_sha256
    estimate = projection.fragment.graph.outcome_estimates[0]
    assert estimate.effect.treatment_events == 39
    assert estimate.effect.control_events == 1
    assert "provider_execution_not_attested_by_contextual_grounding_v3" in (projection.blockers)
    assert not projection.extraction_accuracy_authority
    assert not projection.synthesis_input_authority
    assert not projection.scientific_synthesis_authority
    assert not projection.scientific_effectiveness_authority
    assert not projection.calibration_authority
    assert not projection.claim_release_authority

    with pytest.raises(
        ContextualNumericGroundingV3Error,
        match="runtime_projection_sha256_invalid",
    ):
        project_contextual_grounded_outcome_v3(
            fixture_receipt=fixture,
            raw_outcome=fixture.model_outcome,
            runtime_pipeline_sha256="not-a-sha256",
            provider_execution_binding_sha256=provider_execution_binding_sha256,
        )


def test_row17_candidate2_is_a_distinct_exact_binary_fallback(
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
) -> None:
    receipt = _receipt(suite, 17, 2)
    effect = receipt.grounded_effect
    assert effect.treatment_events == 31
    assert effect.treatment_total == 91
    assert effect.control_events == 6
    assert effect.control_total == 85
    assert effect.outcome_name == "Symptom response"
    assert effect.treatment_arm_label == "500-mg"
    assert effect.comparator_arm_label == "placebo"
    assert receipt.native_projection.status == "typed_graph_mechanics_completed"
    assert receipt.candidate_binding_sha256 != _receipt(suite, 17, 3).candidate_binding_sha256
    assert not receipt.native_projection.synthesis_input_authority


def test_authority_escalation_is_rejected_even_when_suite_hash_is_recomputed(
    suite: ContextualGroundingOfflineFeasibilitySuiteV3,
) -> None:
    raw = deepcopy(suite.model_dump(mode="json"))
    raw["claim_release_authority"] = True
    raw["suite_sha256"] = hash_canonical(
        {key: value for key, value in raw.items() if key != "suite_sha256"}
    )
    with pytest.raises(ValidationError):
        ContextualGroundingOfflineFeasibilitySuiteV3.model_validate(raw)
