"""Passage-anchored, value-free MetaSyn candidate inventory contract.

V1 keyed candidates by a coarse source line, canonical outcome, and effect family.
Several distinct passage-level estimands therefore collapsed to the same descriptor.
V2 makes the passage anchor and an exact protocol outcome-concept quote part of the
scientific unit while deliberately excluding numerical effect values.  Multiple
numerical representations of the same passage-level target remain one candidate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_bounded_generation import (
    INVENTORY_SENTINEL_CAP,
    EffectKind,
)

INVENTORY_V2_VERSION = "metasyn-passage-candidate-inventory-v2"
INVENTORY_V2_RECEIPT_VERSION = "metasyn-passage-candidate-inventory-receipt-v2"
SCHEMA_BUNDLE_VERSION = "metasyn-passage-inventory-schema-bundle-v2"


class MetaSynCandidateInventoryV2Error(ValueError):
    """A model inventory cannot be joined safely to its frozen prompt surface."""


class MetaSynPassageCandidateV2(ContractModel):
    """One value-free statistical target anchored below the coarse-line level."""

    candidate_index: Annotated[int, Field(ge=1, le=INVENTORY_SENTINEL_CAP)]
    canonical_outcome_id: Annotated[str, Field(min_length=1, max_length=64)]
    outcome_concept_quote: Annotated[str, Field(min_length=1, max_length=256)]
    effect_kind: EffectKind
    passage_ids: Annotated[
        list[Annotated[str, Field(pattern=r"^p2-[0-9a-f]{64}$")]],
        Field(min_length=1, max_length=4),
    ]

    @field_validator("passage_ids")
    @classmethod
    def validate_passage_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("metasyn_inventory_v2_passage_ids_not_sorted_unique")
        return value

    @property
    def statistical_target_signature(self) -> tuple[str, str, tuple[str, ...]]:
        """Exclude representation family so one estimand cannot be double counted."""

        return (
            self.canonical_outcome_id,
            " ".join(self.outcome_concept_quote.casefold().split()),
            tuple(self.passage_ids),
        )

    @property
    def descriptor_sha256(self) -> str:
        return hash_canonical(self)


class MetaSynCandidateInventoryV2(ContractModel):
    inventory_version: Literal["metasyn-passage-candidate-inventory-v2"] = (
        INVENTORY_V2_VERSION
    )
    inventory_status: Literal[
        "candidates_found",
        "no_candidate_found",
        "overflow_or_uncertain",
    ]
    candidates: Annotated[
        list[MetaSynPassageCandidateV2], Field(max_length=INVENTORY_SENTINEL_CAP)
    ]
    has_more_or_uncertain: bool

    @model_validator(mode="after")
    def validate_inventory(self) -> MetaSynCandidateInventoryV2:
        indices = [candidate.candidate_index for candidate in self.candidates]
        if indices != list(range(1, len(indices) + 1)):
            raise ValueError("metasyn_inventory_v2_indices_not_contiguous")
        signatures = [
            candidate.statistical_target_signature for candidate in self.candidates
        ]
        if len(signatures) != len(set(signatures)):
            raise ValueError("metasyn_inventory_v2_statistical_target_duplicate")
        expected_order = sorted(
            self.candidates,
            key=lambda candidate: (
                candidate.passage_ids[0],
                candidate.canonical_outcome_id,
                " ".join(candidate.outcome_concept_quote.casefold().split()),
                candidate.effect_kind,
            ),
        )
        if self.candidates != expected_order:
            raise ValueError("metasyn_inventory_v2_candidates_not_canonical")
        if self.inventory_status == "candidates_found":
            if (
                not self.candidates
                or self.has_more_or_uncertain
                or len(self.candidates) >= INVENTORY_SENTINEL_CAP
            ):
                raise ValueError("metasyn_inventory_v2_found_state_invalid")
        elif self.inventory_status == "no_candidate_found":
            if self.candidates or self.has_more_or_uncertain:
                raise ValueError("metasyn_inventory_v2_empty_state_invalid")
        elif not self.has_more_or_uncertain:
            raise ValueError("metasyn_inventory_v2_overflow_requires_uncertainty")
        return self

    def authorizes_packet_generation(self) -> bool:
        return (
            self.inventory_status == "candidates_found"
            and not self.has_more_or_uncertain
            and 0 < len(self.candidates) < INVENTORY_SENTINEL_CAP
        )


class MetaSynCandidateInventoryReceiptV2(ContractModel):
    receipt_version: Literal[
        "metasyn-passage-candidate-inventory-receipt-v2"
    ] = INVENTORY_V2_RECEIPT_VERSION
    row_context_sha256: str
    projection_v2_sha256: str
    allowed_outcome_membership_sha256: str
    passage_membership_sha256: str
    inventory: MetaSynCandidateInventoryV2
    inventory_sha256: str
    status: Literal[
        "candidates_authorized",
        "no_candidate_non_authorizing",
        "capacity_or_uncertainty_non_authorizing",
    ]
    receipt_sha256: str

    @field_validator(
        "row_context_sha256",
        "projection_v2_sha256",
        "allowed_outcome_membership_sha256",
        "passage_membership_sha256",
        "inventory_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"metasyn_inventory_v2_hash_invalid:{info.field_name}")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> MetaSynCandidateInventoryReceiptV2:
        if self.inventory_sha256 != hash_canonical(self.inventory):
            raise ValueError("metasyn_inventory_v2_payload_hash_mismatch")
        expected_status = (
            "candidates_authorized"
            if self.inventory.authorizes_packet_generation()
            else "no_candidate_non_authorizing"
            if self.inventory.inventory_status == "no_candidate_found"
            else "capacity_or_uncertainty_non_authorizing"
        )
        if self.status != expected_status:
            raise ValueError("metasyn_inventory_v2_receipt_status_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if self.receipt_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_inventory_v2_receipt_hash_mismatch")
        return self


def _canonical_outcome_map(
    value: Mapping[str, str] | Sequence[tuple[str, str]],
) -> dict[str, str]:
    items = value.items() if isinstance(value, Mapping) else value
    output: dict[str, str] = {}
    for outcome_id, text in items:
        if (
            not isinstance(outcome_id, str)
            or not outcome_id
            or len(outcome_id) > 64
            or not isinstance(text, str)
            or not text.strip()
            or len(text) > 4096
            or outcome_id in output
        ):
            raise MetaSynCandidateInventoryV2Error(
                "metasyn_inventory_v2_allowed_outcome_invalid"
            )
        output[outcome_id] = text.strip()
    if not output:
        raise MetaSynCandidateInventoryV2Error(
            "metasyn_inventory_v2_allowed_outcomes_empty"
        )
    return dict(sorted(output.items()))


def _canonical_passages(value: Mapping[str, str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for passage_id, text in value.items():
        if (
            not isinstance(passage_id, str)
            or len(passage_id) != 67
            or not passage_id.startswith("p2-")
            or any(character not in "0123456789abcdef" for character in passage_id[3:])
            or not isinstance(text, str)
            or not text.strip()
            or passage_id in output
        ):
            raise MetaSynCandidateInventoryV2Error(
                "metasyn_inventory_v2_passage_surface_invalid"
            )
        output[passage_id] = text
    if not output:
        raise MetaSynCandidateInventoryV2Error(
            "metasyn_inventory_v2_passage_surface_empty"
        )
    return dict(sorted(output.items()))


def validate_metasyn_candidate_inventory_v2(
    value: MetaSynCandidateInventoryV2 | Mapping[str, Any],
    *,
    allowed_outcome_text_by_id: Mapping[str, str] | Sequence[tuple[str, str]],
    passage_text_by_id: Mapping[str, str],
) -> MetaSynCandidateInventoryV2:
    """Validate every value-free descriptor against its exact frozen surfaces."""

    outcomes = _canonical_outcome_map(allowed_outcome_text_by_id)
    passages = _canonical_passages(passage_text_by_id)
    try:
        inventory = MetaSynCandidateInventoryV2.model_validate(value)
    except ValueError as exc:
        raise MetaSynCandidateInventoryV2Error(
            "metasyn_inventory_v2_contract_invalid"
        ) from exc
    for candidate in inventory.candidates:
        outcome_text = outcomes.get(candidate.canonical_outcome_id)
        if outcome_text is None:
            raise MetaSynCandidateInventoryV2Error(
                "metasyn_inventory_v2_outcome_not_allowed"
            )
        if candidate.outcome_concept_quote not in outcome_text:
            raise MetaSynCandidateInventoryV2Error(
                "metasyn_inventory_v2_outcome_concept_not_exact_protocol_quote"
            )
        if any(passage_id not in passages for passage_id in candidate.passage_ids):
            raise MetaSynCandidateInventoryV2Error(
                "metasyn_inventory_v2_passage_not_allowed"
            )
    return inventory


def freeze_metasyn_candidate_inventory_receipt_v2(
    *,
    row_context_sha256: str,
    projection_v2_sha256: str,
    allowed_outcome_text_by_id: Mapping[str, str] | Sequence[tuple[str, str]],
    passage_text_by_id: Mapping[str, str],
    value: MetaSynCandidateInventoryV2 | Mapping[str, Any],
) -> MetaSynCandidateInventoryReceiptV2:
    outcomes = _canonical_outcome_map(allowed_outcome_text_by_id)
    passages = _canonical_passages(passage_text_by_id)
    inventory = validate_metasyn_candidate_inventory_v2(
        value,
        allowed_outcome_text_by_id=outcomes,
        passage_text_by_id=passages,
    )
    status = (
        "candidates_authorized"
        if inventory.authorizes_packet_generation()
        else "no_candidate_non_authorizing"
        if inventory.inventory_status == "no_candidate_found"
        else "capacity_or_uncertainty_non_authorizing"
    )
    payload: dict[str, Any] = {
        "receipt_version": INVENTORY_V2_RECEIPT_VERSION,
        "row_context_sha256": row_context_sha256,
        "projection_v2_sha256": projection_v2_sha256,
        "allowed_outcome_membership_sha256": hash_canonical(outcomes),
        "passage_membership_sha256": hash_canonical(passages),
        "inventory": inventory,
        "inventory_sha256": hash_canonical(inventory),
        "status": status,
    }
    return MetaSynCandidateInventoryReceiptV2.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def validate_metasyn_candidate_inventory_receipt_v2(
    receipt: MetaSynCandidateInventoryReceiptV2 | Mapping[str, Any],
    *,
    row_context_sha256: str,
    projection_v2_sha256: str,
    allowed_outcome_text_by_id: Mapping[str, str] | Sequence[tuple[str, str]],
    passage_text_by_id: Mapping[str, str],
) -> MetaSynCandidateInventoryReceiptV2:
    """Rebuild the receipt from its immutable question and passage surfaces."""

    canonical = MetaSynCandidateInventoryReceiptV2.model_validate(receipt)
    replayed = freeze_metasyn_candidate_inventory_receipt_v2(
        row_context_sha256=row_context_sha256,
        projection_v2_sha256=projection_v2_sha256,
        allowed_outcome_text_by_id=allowed_outcome_text_by_id,
        passage_text_by_id=passage_text_by_id,
        value=canonical.inventory,
    )
    if canonical != replayed:
        raise MetaSynCandidateInventoryV2Error(
            "metasyn_inventory_v2_receipt_external_replay_mismatch"
        )
    return canonical


def metasyn_candidate_inventory_schema_bundle_v2(
    *, allowed_outcome_ids: Sequence[str], passage_ids: Sequence[str]
) -> dict[str, Any]:
    """Return provider/full schemas plus exact context binding for one row."""

    outcomes = sorted(set(allowed_outcome_ids))
    passages = sorted(set(passage_ids))
    if not outcomes or outcomes != list(allowed_outcome_ids):
        raise MetaSynCandidateInventoryV2Error(
            "metasyn_inventory_v2_schema_outcomes_not_sorted_unique"
        )
    if not passages or passages != list(passage_ids):
        raise MetaSynCandidateInventoryV2Error(
            "metasyn_inventory_v2_schema_passages_not_sorted_unique"
        )
    full_schema = TypeAdapter(MetaSynCandidateInventoryV2).json_schema()

    def constrain(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                outcome = properties.get("canonical_outcome_id")
                if isinstance(outcome, dict):
                    outcome["enum"] = outcomes
                candidate_passages = properties.get("passage_ids")
                if isinstance(candidate_passages, dict):
                    items = candidate_passages.get("items")
                    if isinstance(items, dict):
                        items["enum"] = passages
            for child in node.values():
                constrain(child)
        elif isinstance(node, list):
            for child in node:
                constrain(child)

    constrain(full_schema)
    provider_schema = dict(full_schema)
    valid_example = {
        "inventory_version": INVENTORY_V2_VERSION,
        "inventory_status": "no_candidate_found",
        "candidates": [],
        "has_more_or_uncertain": False,
    }
    payload = {
        "schema_bundle_version": SCHEMA_BUNDLE_VERSION,
        "kind": "inventory",
        "context_binding": {
            "allowed_outcome_ids": outcomes,
            "passage_ids": passages,
        },
        "provider_schema": provider_schema,
        "provider_schema_sha256": hash_canonical(provider_schema),
        "full_acceptance_schema": full_schema,
        "full_acceptance_schema_sha256": hash_canonical(full_schema),
        "valid_example": valid_example,
        "valid_example_sha256": hash_canonical(valid_example),
    }
    return {**payload, "schema_bundle_sha256": hash_canonical(payload)}


__all__ = [
    "MetaSynCandidateInventoryReceiptV2",
    "MetaSynCandidateInventoryV2",
    "MetaSynCandidateInventoryV2Error",
    "MetaSynPassageCandidateV2",
    "freeze_metasyn_candidate_inventory_receipt_v2",
    "metasyn_candidate_inventory_schema_bundle_v2",
    "validate_metasyn_candidate_inventory_receipt_v2",
    "validate_metasyn_candidate_inventory_v2",
]
