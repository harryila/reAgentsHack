"""Fail-closed normalization of omitted packet-contract invariants.

The Anthropic packet wire schema used by the immutable v2 run represented literal
constants in descriptions and did not require them on the wire.  A compact model
response can therefore be scientifically complete while omitting ``outcome_version``
or ``packet_status``.  This additive adapter restores only those two deterministic
constants.  It never repairs scientific content, identifiers, nested discriminators,
or malformed shapes.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal

from pydantic import ConfigDict, TypeAdapter, field_validator, model_validator

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_packet_grounding_v2 import (
    MODEL_OUTCOME_V2_VERSION,
    PacketGroundingModelOutcomeV2,
)

COMPACT_NORMALIZATION_V3_VERSION = "native-packet-compact-normalization-v3"
COMPACT_NORMALIZATION_POLICY_V3_VERSION = "native-packet-compact-invariant-policy-v3"
NORMALIZABLE_INVARIANT_FIELDS = ("outcome_version", "packet_status")

_COMPLETED_REQUIRED_NONINVARIANT_FIELDS = frozenset(
    {
        "candidate_binding_sha256",
        "evidence_quote",
        "effect_format_token",
        "effect_unit",
        "numeric_claims",
        "identity_claims",
        "timepoint",
    }
)
_COMPLETED_ALLOWED_FIELDS = _COMPLETED_REQUIRED_NONINVARIANT_FIELDS | frozenset(
    NORMALIZABLE_INVARIANT_FIELDS
)
_ABSTENTION_REQUIRED_NONINVARIANT_FIELDS = frozenset({"candidate_binding_sha256", "reason"})
_ABSTENTION_ALLOWED_FIELDS = _ABSTENTION_REQUIRED_NONINVARIANT_FIELDS | frozenset(
    NORMALIZABLE_INVARIANT_FIELDS
)
_MODEL_OUTCOME_ADAPTER = TypeAdapter(PacketGroundingModelOutcomeV2)


class NativePacketCompactNormalizationV3Error(ValueError):
    """A compact output cannot be expanded without altering scientific content."""


class _FrozenExactModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


def _validate_sha256(value: str, field_name: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"native_packet_compact_v3_hash_invalid:{field_name}")
    return value


class NativePacketInvariantInsertionV3(_FrozenExactModel):
    field_name: Literal["outcome_version", "packet_status"]
    operation: Literal["insert_absent_invariant_constant"] = "insert_absent_invariant_constant"
    inserted_value: str
    scientific_value_derived_or_changed: Literal[False] = False


class NativePacketCompactNormalizationReceiptV3(_FrozenExactModel):
    normalization_version: Literal["native-packet-compact-normalization-v3"] = (
        COMPACT_NORMALIZATION_V3_VERSION
    )
    policy_version: Literal["native-packet-compact-invariant-policy-v3"] = (
        COMPACT_NORMALIZATION_POLICY_V3_VERSION
    )
    status: Literal["expanded_only_absent_invariant_constants_full_v2_contract_valid"] = (
        "expanded_only_absent_invariant_constants_full_v2_contract_valid"
    )
    expected_candidate_binding_sha256: str
    raw_candidate_binding_sha256: str
    raw_candidate_binding_already_matched: Literal[True] = True
    raw_model_outcome: dict[str, Any]
    raw_model_outcome_sha256: str
    raw_field_names: list[str]
    branch: Literal["completed", "unable_to_complete"]
    absent_invariant_fields: list[Literal["outcome_version", "packet_status"]]
    insertions: list[NativePacketInvariantInsertionV3]
    normalized_model_outcome: dict[str, Any]
    normalized_model_outcome_sha256: str
    normalized_full_contract_kind: Literal["completed", "abstention"]
    normalized_full_contract_valid: Literal[True] = True
    normalization_idempotent: Literal[True] = True
    only_absent_invariant_constants_inserted: Literal[True] = True
    noninvariant_payload_unchanged: Literal[True] = True
    scientific_values_derived_or_changed: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    receipt_sha256: str

    @field_validator(
        "expected_candidate_binding_sha256",
        "raw_candidate_binding_sha256",
        "raw_model_outcome_sha256",
        "normalized_model_outcome_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("raw_field_names")
    @classmethod
    def validate_raw_fields(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("native_packet_compact_v3_raw_fields_not_canonical")
        return value

    @field_validator("absent_invariant_fields")
    @classmethod
    def validate_absent_fields(
        cls, value: list[Literal["outcome_version", "packet_status"]]
    ) -> list[Literal["outcome_version", "packet_status"]]:
        expected_order = [
            field_name for field_name in NORMALIZABLE_INVARIANT_FIELDS if field_name in set(value)
        ]
        if value != expected_order:
            raise ValueError("native_packet_compact_v3_absent_fields_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> NativePacketCompactNormalizationReceiptV3:
        if self.expected_candidate_binding_sha256 != self.raw_candidate_binding_sha256:
            raise ValueError("native_packet_compact_v3_candidate_binding_mismatch")
        if self.raw_model_outcome_sha256 != hash_canonical(self.raw_model_outcome):
            raise ValueError("native_packet_compact_v3_raw_hash_mismatch")
        if self.normalized_model_outcome_sha256 != hash_canonical(self.normalized_model_outcome):
            raise ValueError("native_packet_compact_v3_normalized_hash_mismatch")
        if self.raw_field_names != sorted(self.raw_model_outcome):
            raise ValueError("native_packet_compact_v3_raw_field_membership_mismatch")
        insertion_fields = [item.field_name for item in self.insertions]
        if insertion_fields != self.absent_invariant_fields:
            raise ValueError("native_packet_compact_v3_insertion_membership_mismatch")
        expected_normalized, expected_branch, expected_absent = _expand(
            raw_model_outcome=self.raw_model_outcome,
            expected_candidate_binding_sha256=self.expected_candidate_binding_sha256,
        )
        expected_kind = "completed" if expected_branch == "completed" else "abstention"
        expected_inserted_values = {
            "outcome_version": MODEL_OUTCOME_V2_VERSION,
            "packet_status": expected_branch,
        }
        if (
            self.branch != expected_branch
            or self.absent_invariant_fields != expected_absent
            or self.normalized_model_outcome != expected_normalized
            or self.normalized_full_contract_kind != expected_kind
            or any(
                item.inserted_value != expected_inserted_values[item.field_name]
                for item in self.insertions
            )
        ):
            raise ValueError("native_packet_compact_v3_intrinsic_replay_mismatch")
        second, second_branch, second_absent = _expand(
            raw_model_outcome=self.normalized_model_outcome,
            expected_candidate_binding_sha256=self.expected_candidate_binding_sha256,
        )
        if second != self.normalized_model_outcome or second_branch != self.branch or second_absent:
            raise ValueError("native_packet_compact_v3_intrinsic_idempotence_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if self.receipt_sha256 != hash_canonical(payload):
            raise ValueError("native_packet_compact_v3_receipt_hash_mismatch")
        return self


def _canonical_raw(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativePacketCompactNormalizationV3Error(
            "native_packet_compact_v3_raw_outcome_not_object"
        )
    raw = deepcopy(dict(value))
    if not raw or any(not isinstance(key, str) for key in raw):
        raise NativePacketCompactNormalizationV3Error(
            "native_packet_compact_v3_raw_outcome_keys_invalid"
        )
    return raw


def _branch(raw: Mapping[str, Any]) -> Literal["completed", "unable_to_complete"]:
    observed = raw.get("packet_status")
    if observed is not None:
        if observed not in {"completed", "unable_to_complete"}:
            raise NativePacketCompactNormalizationV3Error(
                "native_packet_compact_v3_packet_status_conflict"
            )
        return observed
    has_abstention_shape = "reason" in raw
    completed_shape_fields = _COMPLETED_REQUIRED_NONINVARIANT_FIELDS - {"candidate_binding_sha256"}
    has_completed_shape = bool(completed_shape_fields & set(raw))
    if has_abstention_shape == has_completed_shape:
        raise NativePacketCompactNormalizationV3Error("native_packet_compact_v3_branch_ambiguous")
    return "unable_to_complete" if has_abstention_shape else "completed"


def _expand(
    *,
    raw_model_outcome: Mapping[str, Any],
    expected_candidate_binding_sha256: str,
) -> tuple[
    dict[str, Any],
    Literal["completed", "unable_to_complete"],
    list[Literal["outcome_version", "packet_status"]],
]:
    expected_binding = _validate_sha256(
        expected_candidate_binding_sha256, "expected_candidate_binding_sha256"
    )
    raw = _canonical_raw(raw_model_outcome)
    if raw.get("candidate_binding_sha256") != expected_binding:
        raise NativePacketCompactNormalizationV3Error(
            "native_packet_compact_v3_raw_candidate_binding_mismatch"
        )
    branch = _branch(raw)
    allowed = _COMPLETED_ALLOWED_FIELDS if branch == "completed" else _ABSTENTION_ALLOWED_FIELDS
    required = (
        _COMPLETED_REQUIRED_NONINVARIANT_FIELDS
        if branch == "completed"
        else _ABSTENTION_REQUIRED_NONINVARIANT_FIELDS
    )
    extra = set(raw) - allowed
    missing_scientific = required - set(raw)
    if extra:
        raise NativePacketCompactNormalizationV3Error(
            "native_packet_compact_v3_extra_normalization_forbidden:" + ",".join(sorted(extra))
        )
    if missing_scientific:
        raise NativePacketCompactNormalizationV3Error(
            "native_packet_compact_v3_noninvariant_field_missing:"
            + ",".join(sorted(missing_scientific))
        )
    expected_constants = {
        "outcome_version": MODEL_OUTCOME_V2_VERSION,
        "packet_status": branch,
    }
    for field_name, expected in expected_constants.items():
        if field_name in raw and raw[field_name] != expected:
            raise NativePacketCompactNormalizationV3Error(
                f"native_packet_compact_v3_invariant_conflict:{field_name}"
            )
    absent: list[Literal["outcome_version", "packet_status"]] = [
        field_name  # type: ignore[misc]
        for field_name in NORMALIZABLE_INVARIANT_FIELDS
        if field_name not in raw
    ]
    normalized = deepcopy(raw)
    for field_name in absent:
        normalized[field_name] = expected_constants[field_name]
    try:
        parsed = _MODEL_OUTCOME_ADAPTER.validate_python(normalized)
    except ValueError as exc:
        raise NativePacketCompactNormalizationV3Error(
            "native_packet_compact_v3_full_contract_invalid"
        ) from exc
    canonical = parsed.model_dump(mode="json")
    if canonical != normalized:
        raise NativePacketCompactNormalizationV3Error(
            "native_packet_compact_v3_hidden_default_or_coercion_forbidden"
        )
    return canonical, branch, absent


def freeze_native_packet_compact_normalization_receipt_v3(
    *,
    raw_model_outcome: Mapping[str, Any],
    expected_candidate_binding_sha256: str,
) -> NativePacketCompactNormalizationReceiptV3:
    """Expand only absent top-level literal constants and freeze the proof."""

    raw = _canonical_raw(raw_model_outcome)
    normalized, branch, absent = _expand(
        raw_model_outcome=raw,
        expected_candidate_binding_sha256=expected_candidate_binding_sha256,
    )
    second, second_branch, second_absent = _expand(
        raw_model_outcome=normalized,
        expected_candidate_binding_sha256=expected_candidate_binding_sha256,
    )
    if second != normalized or second_branch != branch or second_absent:
        raise NativePacketCompactNormalizationV3Error("native_packet_compact_v3_not_idempotent")
    expected_constants = {
        "outcome_version": MODEL_OUTCOME_V2_VERSION,
        "packet_status": branch,
    }
    insertions = [
        NativePacketInvariantInsertionV3(
            field_name=field_name,
            inserted_value=expected_constants[field_name],
        )
        for field_name in absent
    ]
    payload: dict[str, Any] = {
        "normalization_version": COMPACT_NORMALIZATION_V3_VERSION,
        "policy_version": COMPACT_NORMALIZATION_POLICY_V3_VERSION,
        "status": ("expanded_only_absent_invariant_constants_full_v2_contract_valid"),
        "expected_candidate_binding_sha256": expected_candidate_binding_sha256,
        "raw_candidate_binding_sha256": raw["candidate_binding_sha256"],
        "raw_candidate_binding_already_matched": True,
        "raw_model_outcome": raw,
        "raw_model_outcome_sha256": hash_canonical(raw),
        "raw_field_names": sorted(raw),
        "branch": branch,
        "absent_invariant_fields": absent,
        "insertions": insertions,
        "normalized_model_outcome": normalized,
        "normalized_model_outcome_sha256": hash_canonical(normalized),
        "normalized_full_contract_kind": ("completed" if branch == "completed" else "abstention"),
        "normalized_full_contract_valid": True,
        "normalization_idempotent": True,
        "only_absent_invariant_constants_inserted": True,
        "noninvariant_payload_unchanged": True,
        "scientific_values_derived_or_changed": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return NativePacketCompactNormalizationReceiptV3.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


def validate_native_packet_compact_normalization_receipt_v3(
    *,
    receipt: NativePacketCompactNormalizationReceiptV3 | Mapping[str, Any],
    raw_model_outcome: Mapping[str, Any],
    expected_candidate_binding_sha256: str,
) -> NativePacketCompactNormalizationReceiptV3:
    """Replay a compact-normalization receipt from the untouched raw mapping."""

    try:
        canonical = NativePacketCompactNormalizationReceiptV3.model_validate(
            receipt.model_dump(mode="json")
            if isinstance(receipt, NativePacketCompactNormalizationReceiptV3)
            else receipt
        )
    except ValueError as exc:
        raise NativePacketCompactNormalizationV3Error(
            "native_packet_compact_v3_receipt_contract_invalid"
        ) from exc
    replayed = freeze_native_packet_compact_normalization_receipt_v3(
        raw_model_outcome=raw_model_outcome,
        expected_candidate_binding_sha256=expected_candidate_binding_sha256,
    )
    if replayed != canonical:
        raise NativePacketCompactNormalizationV3Error(
            "native_packet_compact_v3_receipt_external_replay_mismatch"
        )
    return canonical


__all__ = [
    "COMPACT_NORMALIZATION_POLICY_V3_VERSION",
    "COMPACT_NORMALIZATION_V3_VERSION",
    "NORMALIZABLE_INVARIANT_FIELDS",
    "NativePacketCompactNormalizationReceiptV3",
    "NativePacketCompactNormalizationV3Error",
    "NativePacketInvariantInsertionV3",
    "freeze_native_packet_compact_normalization_receipt_v3",
    "validate_native_packet_compact_normalization_receipt_v3",
]
