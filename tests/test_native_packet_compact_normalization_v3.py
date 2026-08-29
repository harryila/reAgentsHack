from __future__ import annotations

from copy import deepcopy

import pytest

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.native_packet_compact_normalization_v3 import (
    NativePacketCompactNormalizationReceiptV3,
    NativePacketCompactNormalizationV3Error,
    freeze_native_packet_compact_normalization_receipt_v3,
    validate_native_packet_compact_normalization_receipt_v3,
)
from literature_multiverse.native_packet_grounding_v2 import MODEL_OUTCOME_V2_VERSION

BINDING = "a" * 64


def _compact_abstention() -> dict[str, object]:
    return {
        "candidate_binding_sha256": BINDING,
        "reason": "source_support_incomplete",
    }


def _compact_completed() -> dict[str, object]:
    return {
        "candidate_binding_sha256": BINDING,
        "evidence_quote": "Trial Alpha reported Hedges g 0.50 (95% CI 0.20 to 0.80).",
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
                "verbatim_numeric_token": "0.20",
                "normalization": "identity",
            },
            {
                "field_path": "effect.ci_upper",
                "verbatim_numeric_token": "0.80",
                "normalization": "identity",
            },
            {
                "field_path": "effect.estimate",
                "verbatim_numeric_token": "0.50",
                "normalization": "identity",
            },
        ],
        "identity_claims": [
            {
                "field_path": "study.source_label",
                "verbatim_identity_text": "Trial Alpha",
            }
        ],
        "timepoint": {"kind": "not_reported"},
    }


@pytest.mark.parametrize(
    ("raw", "branch"),
    [
        (_compact_abstention(), "unable_to_complete"),
        (_compact_completed(), "completed"),
    ],
)
def test_only_absent_invariant_constants_are_inserted_and_replay_is_idempotent(
    raw: dict[str, object], branch: str
) -> None:
    receipt = freeze_native_packet_compact_normalization_receipt_v3(
        raw_model_outcome=raw,
        expected_candidate_binding_sha256=BINDING,
    )

    assert receipt.raw_candidate_binding_already_matched is True
    assert receipt.raw_model_outcome == raw
    assert receipt.absent_invariant_fields == ["outcome_version", "packet_status"]
    assert [item.field_name for item in receipt.insertions] == [
        "outcome_version",
        "packet_status",
    ]
    assert receipt.normalized_model_outcome == {
        **raw,
        "outcome_version": MODEL_OUTCOME_V2_VERSION,
        "packet_status": branch,
    }
    assert receipt.noninvariant_payload_unchanged is True
    assert receipt.scientific_values_derived_or_changed is False
    assert receipt.normalization_idempotent is True

    replayed = validate_native_packet_compact_normalization_receipt_v3(
        receipt=receipt,
        raw_model_outcome=raw,
        expected_candidate_binding_sha256=BINDING,
    )
    assert replayed == receipt

    second = freeze_native_packet_compact_normalization_receipt_v3(
        raw_model_outcome=receipt.normalized_model_outcome,
        expected_candidate_binding_sha256=BINDING,
    )
    assert second.normalized_model_outcome == receipt.normalized_model_outcome
    assert second.absent_invariant_fields == []
    assert second.insertions == []


def test_conflicts_binding_mismatch_extra_fields_and_scientific_repair_fail_closed() -> None:
    cases = []
    mismatch = _compact_abstention()
    mismatch["candidate_binding_sha256"] = "b" * 64
    cases.append(mismatch)
    conflict = _compact_abstention()
    conflict["packet_status"] = "completed"
    cases.append(conflict)
    extra = _compact_abstention()
    extra["helpful_repair"] = "not permitted"
    cases.append(extra)
    incomplete = _compact_completed()
    incomplete.pop("numeric_claims")
    cases.append(incomplete)

    for raw in cases:
        with pytest.raises(NativePacketCompactNormalizationV3Error):
            freeze_native_packet_compact_normalization_receipt_v3(
                raw_model_outcome=raw,
                expected_candidate_binding_sha256=BINDING,
            )


def test_hidden_defaults_and_type_coercion_are_not_accepted() -> None:
    raw = _compact_completed()
    numeric_claims = raw["numeric_claims"]
    assert isinstance(numeric_claims, list)
    numeric_claims[0]["verbatim_numeric_token"] = 95  # type: ignore[index]

    with pytest.raises(NativePacketCompactNormalizationV3Error):
        freeze_native_packet_compact_normalization_receipt_v3(
            raw_model_outcome=raw,
            expected_candidate_binding_sha256=BINDING,
        )


def test_external_replay_rejects_a_receipt_for_different_raw_bytes() -> None:
    raw = _compact_abstention()
    receipt = freeze_native_packet_compact_normalization_receipt_v3(
        raw_model_outcome=raw,
        expected_candidate_binding_sha256=BINDING,
    )
    different = deepcopy(raw)
    different["reason"] = "candidate_ambiguous"

    with pytest.raises(
        NativePacketCompactNormalizationV3Error,
        match="external_replay_mismatch",
    ):
        validate_native_packet_compact_normalization_receipt_v3(
            receipt=receipt,
            raw_model_outcome=different,
            expected_candidate_binding_sha256=BINDING,
        )


def _coherently_rehash_receipt(payload: dict[str, object]) -> dict[str, object]:
    raw = payload["raw_model_outcome"]
    normalized = payload["normalized_model_outcome"]
    payload["raw_model_outcome_sha256"] = hash_canonical(raw)
    payload["normalized_model_outcome_sha256"] = hash_canonical(normalized)
    payload["receipt_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    return payload


@pytest.mark.parametrize(
    "forgery",
    [
        "changed_noninvariant_payload",
        "wrong_inserted_constant",
        "branch_kind_conflict",
        "removed_normalized_scientific_field",
    ],
)
def test_coherently_rehashed_forged_receipt_fails_intrinsic_replay(forgery: str) -> None:
    receipt = freeze_native_packet_compact_normalization_receipt_v3(
        raw_model_outcome=_compact_abstention(),
        expected_candidate_binding_sha256=BINDING,
    )
    payload = receipt.model_dump(mode="json")
    normalized = payload["normalized_model_outcome"]
    insertions = payload["insertions"]
    assert isinstance(normalized, dict)
    assert isinstance(insertions, list)
    if forgery == "changed_noninvariant_payload":
        normalized["reason"] = "candidate_ambiguous"
    elif forgery == "wrong_inserted_constant":
        insertions[0]["inserted_value"] = "forged"  # type: ignore[index]
    elif forgery == "branch_kind_conflict":
        payload["branch"] = "completed"
        payload["normalized_full_contract_kind"] = "completed"
    else:
        normalized.pop("reason")
    _coherently_rehash_receipt(payload)

    with pytest.raises(ValueError, match="intrinsic"):
        NativePacketCompactNormalizationReceiptV3.model_validate(payload)
