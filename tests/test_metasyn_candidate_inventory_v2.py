from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_candidate_inventory_v2 import (
    MetaSynCandidateInventoryReceiptV2,
    MetaSynCandidateInventoryV2,
    MetaSynCandidateInventoryV2Error,
    freeze_metasyn_candidate_inventory_receipt_v2,
    metasyn_candidate_inventory_schema_bundle_v2,
    validate_metasyn_candidate_inventory_receipt_v2,
    validate_metasyn_candidate_inventory_v2,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
PASSAGE_A = "p2-" + "1" * 64
PASSAGE_B = "p2-" + "2" * 64
OUTCOMES = {
    "outcome-01": (
        "Spleen volume reduction, total symptom score reduction, anemia events, "
        "and thrombocytopenia events"
    )
}
PASSAGES = {
    PASSAGE_A: "Spleen volume response was reported for treatment versus placebo.",
    PASSAGE_B: "The total symptom score response was reported at week 24.",
}


def _candidate(
    *,
    index: int = 1,
    concept: str = "Spleen volume reduction",
    effect_kind: str = "direct_confidence_interval",
    passage_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "candidate_index": index,
        "canonical_outcome_id": "outcome-01",
        "outcome_concept_quote": concept,
        "effect_kind": effect_kind,
        "passage_ids": passage_ids or [PASSAGE_A],
    }


def _inventory(candidates: list[dict[str, object]]) -> dict[str, object]:
    return {
        "inventory_version": "metasyn-passage-candidate-inventory-v2",
        "inventory_status": "candidates_found",
        "candidates": candidates,
        "has_more_or_uncertain": False,
    }


def test_passage_anchors_separate_targets_that_shared_a_v1_line() -> None:
    value = _inventory(
        [
            _candidate(),
            _candidate(
                index=2,
                concept="total symptom score reduction",
                effect_kind="binary_group_statistics",
                passage_ids=[PASSAGE_B],
            ),
        ]
    )

    canonical = validate_metasyn_candidate_inventory_v2(
        value,
        allowed_outcome_text_by_id=OUTCOMES,
        passage_text_by_id=PASSAGES,
    )

    assert canonical.authorizes_packet_generation() is True
    assert len({item.descriptor_sha256 for item in canonical.candidates}) == 2


def test_multiple_representations_of_one_target_fail_closed() -> None:
    value = _inventory(
        [
            _candidate(),
            _candidate(index=2, effect_kind="binary_group_statistics"),
        ]
    )

    with pytest.raises(ValidationError, match="statistical_target_duplicate"):
        MetaSynCandidateInventoryV2.model_validate(value)


def test_outcome_concept_must_be_an_exact_protocol_quote() -> None:
    value = _inventory([_candidate(concept="overall survival")])

    with pytest.raises(
        MetaSynCandidateInventoryV2Error,
        match="outcome_concept_not_exact_protocol_quote",
    ):
        validate_metasyn_candidate_inventory_v2(
            value,
            allowed_outcome_text_by_id=OUTCOMES,
            passage_text_by_id=PASSAGES,
        )


def test_inventory_receipt_exactly_binds_prompt_surfaces_and_rejects_tamper() -> None:
    receipt = freeze_metasyn_candidate_inventory_receipt_v2(
        row_context_sha256=HASH_A,
        projection_v2_sha256=HASH_B,
        allowed_outcome_text_by_id=OUTCOMES,
        passage_text_by_id=PASSAGES,
        value=_inventory([_candidate()]),
    )

    assert receipt.status == "candidates_authorized"
    assert receipt.inventory_sha256 == hash_canonical(receipt.inventory)
    assert (
        validate_metasyn_candidate_inventory_receipt_v2(
            receipt,
            row_context_sha256=HASH_A,
            projection_v2_sha256=HASH_B,
            allowed_outcome_text_by_id=OUTCOMES,
            passage_text_by_id=PASSAGES,
        )
        == receipt
    )

    tampered = receipt.model_dump(mode="json")
    tampered["passage_membership_sha256"] = "c" * 64
    with pytest.raises(ValidationError, match="receipt_hash_mismatch"):
        MetaSynCandidateInventoryReceiptV2.model_validate(tampered)

    with pytest.raises(
        MetaSynCandidateInventoryV2Error,
        match="external_replay_mismatch",
    ):
        validate_metasyn_candidate_inventory_receipt_v2(
            receipt,
            row_context_sha256=HASH_A,
            projection_v2_sha256="d" * 64,
            allowed_outcome_text_by_id=OUTCOMES,
            passage_text_by_id=PASSAGES,
        )


def test_inventory_schema_binds_allowed_outcomes_and_passages() -> None:
    bundle = metasyn_candidate_inventory_schema_bundle_v2(
        allowed_outcome_ids=["outcome-01"],
        passage_ids=[PASSAGE_A, PASSAGE_B],
    )

    candidate = bundle["full_acceptance_schema"]["$defs"][
        "MetaSynPassageCandidateV2"
    ]["properties"]
    assert candidate["canonical_outcome_id"]["enum"] == ["outcome-01"]
    assert candidate["passage_ids"]["items"]["enum"] == [PASSAGE_A, PASSAGE_B]

    tampered = deepcopy(bundle)
    tampered["context_binding"]["passage_ids"].append("p2-" + "9" * 64)
    tampered_payload = {
        key: value for key, value in tampered.items() if key != "schema_bundle_sha256"
    }
    assert hash_canonical(tampered_payload) != bundle["schema_bundle_sha256"]


def test_uncertainty_never_authorizes_packets() -> None:
    uncertain = {
        "inventory_version": "metasyn-passage-candidate-inventory-v2",
        "inventory_status": "overflow_or_uncertain",
        "candidates": [_candidate()],
        "has_more_or_uncertain": True,
    }

    canonical = validate_metasyn_candidate_inventory_v2(
        uncertain,
        allowed_outcome_text_by_id=OUTCOMES,
        passage_text_by_id=PASSAGES,
    )

    assert canonical.authorizes_packet_generation() is False
