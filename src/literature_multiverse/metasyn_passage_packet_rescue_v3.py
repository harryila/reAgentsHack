"""Additive, label-blind rescue of the immutable MetaSyn packet smoke.

This module does two deliberately separate things:

* it externally replays the immutable v2 workspace and freezes a forensic proof
  that its three packet outputs were valid scientific abstentions after restoring
  only omitted wire-schema constants; and
* it freezes and executes at most three fresh, exactly-once exploratory requests
  selected only from v2 candidates that were never attempted.

Nothing here changes v2 artifacts, repairs inventory outputs, measures accuracy,
authorizes synthesis, or authorizes claim release.
"""

from __future__ import annotations

import ast
import fcntl
import json
import os
import re
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from importlib.metadata import version as distribution_version
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Protocol

from pydantic import ConfigDict, Field, TypeAdapter, field_validator, model_validator

import literature_multiverse.native_packet_grounding_v2 as native_packet_grounding_v2
from literature_multiverse.anthropic_bounded_generation import (
    AnthropicBoundedRequestV1,
    AnthropicBoundedResultV1,
    freeze_anthropic_bounded_request,
)
from literature_multiverse.hosted_exact_once import (
    HostedExactOnceAmbiguityIncidentV1,
    HostedExactOnceCostAuthorizationV1,
    HostedExactOnceIntentV1,
    HostedExactOnceProviderReceiptV1,
    execute_hosted_exactly_once,
    freeze_hosted_exact_once_cost_authorization,
    freeze_hosted_exact_once_intent,
    validate_hosted_exact_once_outcome,
)
from literature_multiverse.lineage import (
    atomic_write_json,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.metasyn_candidate_inventory_v2 import (
    MetaSynCandidateInventoryV2Error,
    validate_metasyn_candidate_inventory_v2,
)
from literature_multiverse.metasyn_extraction_inputs_v2 import (
    MetaSynPacketCandidateInputV2,
)
from literature_multiverse.metasyn_passage_hosted_bundle_v2 import (
    MetaSynPassageHostedExecutionBundleV2,
    validate_metasyn_passage_hosted_execution_bundle_v2,
)
from literature_multiverse.metasyn_passage_hosted_runtime_v2 import (
    InventoryLedgerV2,
    PacketCallResultV2,
    PacketRequestV2,
    PacketRosterV2,
    PacketSmokeReceiptV2,
    metasyn_passage_hosted_runtime_status_v2,
    validate_metasyn_passage_inventory_ledger_v2,
    validate_metasyn_passage_packet_result_v2,
    validate_metasyn_passage_packet_roster_v2,
    validate_metasyn_passage_preflight_v2,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_packet_assembly_v2 import (
    NativePacketAssemblyAbstentionV2,
    NativePacketAssemblyCompletedV2,
    NativePacketAssemblyOutcomeV2,
    assemble_native_packet_v2,
    replay_metasyn_question_projection_spec_v2,
    validate_native_packet_assembly_v2,
)
from literature_multiverse.native_packet_compact_normalization_v3 import (
    NativePacketCompactNormalizationReceiptV3,
    freeze_native_packet_compact_normalization_receipt_v3,
    validate_native_packet_compact_normalization_receipt_v3,
)
from literature_multiverse.native_packet_grounding_v2 import (
    PacketGroundingAbstentionReceiptV2,
    PacketGroundingCompletedReceiptV2,
    PacketGroundingReceiptV2,
    freeze_passage_packet_grounding_receipt_v2,
    validate_passage_packet_grounding_receipt_v2,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    compute_pipeline_fingerprint,
)

RESCUE_CONFIG_VERSION = "metasyn-passage-packet-rescue-config-v3"
RESCUE_PLAN_VERSION = "metasyn-passage-packet-rescue-plan-v3"
V2_REPLAY_SNAPSHOT_VERSION = "metasyn-passage-v2-replay-snapshot-v3"
V2_FORENSIC_RECEIPT_VERSION = "metasyn-passage-v2-compact-forensic-receipt-v3"
RESCUE_RESULT_VERSION = "metasyn-passage-packet-rescue-result-v3"
RESCUE_SMOKE_VERSION = "metasyn-passage-packet-rescue-smoke-v3"
RESCUE_REPORT_VERSION = "metasyn-passage-packet-rescue-report-v3"
RESCUE_VALIDATION_VERSION = "metasyn-passage-packet-rescue-validation-v3"
RESCUE_CHECKPOINT_VERSION = "metasyn-passage-packet-rescue-checkpoint-v3"
RESCUE_PRE_CALL_BLOCKER_VERSION = "metasyn-passage-packet-rescue-pre-call-blocker-v3"
RESCUE_PIPELINE_COMPONENT_VERSION = "1"

DEFAULT_CONFIG_PATH = Path("configs/benchmarks/metasyn-passage-packet-rescue-v3.json")
DEFAULT_V2_WORKSPACE = Path("data/cache/metasyn/passage-hosted-yield-v2")
DEFAULT_RESCUE_WORKSPACE = Path("data/cache/metasyn/passage-packet-rescue-v3")

EXPECTED_V2_EXECUTION_BUNDLE_SHA256 = (
    "f87eddcbcbafc778f18ff85c92c0f914a763d242311a859d9d979ded229b4972"
)
EXPECTED_V2_INVENTORY_LEDGER_SHA256 = (
    "aeb824df26b2f9efe4677af85f54fc217f41fe1fe043e1f7ebc9ba68de0b6e2f"
)
EXPECTED_V2_PACKET_ROSTER_SHA256 = (
    "97adae1f1a36da26b462fa1b2bec229dd568449870d8a7ae1b5f3418db2c842f"
)
EXPECTED_V2_FAILED_SMOKE_SHA256 = "63bd3b15b5ba1139a98b9d1590fbb9064f433f7b7a5790ad640aa78c78186a41"
EXPECTED_V2_PROVIDER_RECEIPT_COUNT = 43
EXPECTED_V2_PACKET_ATTEMPT_COUNT = 3
EXPECTED_V2_INVENTORY_INVALID_COUNT = 10
EXPECTED_FUTURE_REPRESENTATIONAL_RECOVERY_ROWS = 9
EXPECTED_FUTURE_REPRESENTATIONAL_RECOVERY_CANDIDATES = 42
MAXIMUM_RESCUE_SMOKE_CALLS = 3

_CI_LITERAL_PATTERN = r"(?i)(?:\bconfidence\s+interval\b|\b95\s*%\s*ci\b|\b95%ci\b)"
_RATIO_LITERAL_PATTERN = (
    r"(?i)\b(?:odds\s+ratio|hazard\s+ratio|rate\s+ratio|risk\s+ratio|"
    r"relative\s+risk|or|hr|rr)\b"
)
_CI_LITERAL_RE = re.compile(_CI_LITERAL_PATTERN)
_RATIO_LITERAL_RE = re.compile(_RATIO_LITERAL_PATTERN)
_EN_DASH = "\N{EN DASH}"

_INVENTORY_REPRESENTATIONAL_ROWS = frozenset({9, 18, 19, 21, 25, 26, 27, 29, 30})
_INVENTORY_SCIENTIFICALLY_INVALID_ROWS = frozenset({13})

_PYTHON_FINGERPRINT_SEEDS = (
    "src/literature_multiverse/native_packet_compact_normalization_v3.py",
    "src/literature_multiverse/metasyn_passage_packet_rescue_v3.py",
    "scripts/run_metasyn_passage_packet_rescue_v3.py",
)
_NON_PYTHON_FINGERPRINT_FILES = (
    DEFAULT_CONFIG_PATH.as_posix(),
    "pyproject.toml",
    "uv.lock",
)
_INSTALLED_DEPENDENCIES = ("anthropic", "jsonschema", "pydantic")


class MetaSynPassagePacketRescueV3Error(ValueError):
    """The additive rescue cannot proceed without weakening a frozen gate."""


class HostedClientProtocol(Protocol):
    def generate(self, request: AnthropicBoundedRequestV1) -> AnthropicBoundedResultV1:
        """Make the one provider attempt authorized for ``request``."""


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
        raise ValueError(f"metasyn_packet_rescue_v3_hash_invalid:{field_name}")
    return value


def _usd_micros(value: Decimal) -> int:
    return int((value * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING))


class MetaSynPassagePacketRescueConfigV3(_FrozenExactModel):
    config_version: Literal["metasyn-passage-packet-rescue-config-v3"] = RESCUE_CONFIG_VERSION
    diagnostic_scope: Literal["post_hoc_label_blind_exploratory_typed_effect_yield_only"]
    expected_v2_execution_bundle_sha256: Literal[
        "f87eddcbcbafc778f18ff85c92c0f914a763d242311a859d9d979ded229b4972"
    ]
    expected_v2_inventory_ledger_sha256: Literal[
        "aeb824df26b2f9efe4677af85f54fc217f41fe1fe043e1f7ebc9ba68de0b6e2f"
    ]
    expected_v2_packet_roster_sha256: Literal[
        "97adae1f1a36da26b462fa1b2bec229dd568449870d8a7ae1b5f3418db2c842f"
    ]
    expected_v2_failed_smoke_sha256: Literal[
        "63bd3b15b5ba1139a98b9d1590fbb9064f433f7b7a5790ad640aa78c78186a41"
    ]
    expected_v2_provider_receipt_count: Literal[43]
    maximum_smoke_calls: Literal[3]
    configured_cost_cap_usd_micros: Literal[2200000]
    operation: Literal["metasyn-passage-packet-rescue-v3"]
    request_key_prefix: Literal["rescue-v3"]
    packet_phase: Literal["smoke_packet"]
    selection_rule: Literal[
        "unattempted_then_release_grade_full_text_then_direct_confidence_interval_then_literal_ci_and_ratio_then_prompt_rank_row_candidate"
    ]
    compact_normalizable_invariant_fields: list[Literal["outcome_version", "packet_status"]]
    application_retries_per_request: Literal[0]
    sdk_retries_per_request: Literal[0]
    orphan_or_ambiguous_attempt_retry_permitted: Literal[False]
    pre_call_zero_yield_blocker_required: Literal[True]
    provider_calls_permitted: Literal[False]
    authorization_created: Literal[False]
    inventory_normalization_permitted: Literal[False]
    reference_fields_unopened: Literal[True]
    official_test_labels_opened: Literal[False]
    accuracy_authority: Literal[False]
    synthesis_input_authority: Literal[False]
    claim_release_authority: Literal[False]
    config_sha256: str

    @field_validator("config_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "config_sha256")

    @field_validator("compact_normalizable_invariant_fields")
    @classmethod
    def validate_normalizable_fields(
        cls, value: list[Literal["outcome_version", "packet_status"]]
    ) -> list[Literal["outcome_version", "packet_status"]]:
        if value != ["outcome_version", "packet_status"]:
            raise ValueError("metasyn_packet_rescue_v3_normalizable_fields_mismatch")
        return value

    @model_validator(mode="after")
    def validate_config(self) -> MetaSynPassagePacketRescueConfigV3:
        payload = self.model_dump(mode="json", exclude={"config_sha256"})
        if self.config_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_packet_rescue_v3_config_hash_mismatch")
        return self


def _canonical_repository_root(value: Path) -> Path:
    root = Path(os.path.abspath(value))
    try:
        mode = root.lstat().st_mode
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_repository_root_unreadable"
        ) from exc
    if stat.S_ISLNK(mode) or not resolved.is_dir():
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_repository_root_unsafe")
    return resolved


def _checked_file(*, root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if (
        not relative_path
        or "\\" in relative_path
        or relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
        or relative.as_posix() != relative_path
    ):
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_file_path_unsafe")
    candidate = root
    for part in relative.parts:
        candidate /= part
        try:
            mode = candidate.lstat().st_mode
        except OSError as exc:
            raise MetaSynPassagePacketRescueV3Error(
                f"metasyn_packet_rescue_v3_file_missing:{relative_path}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise MetaSynPassagePacketRescueV3Error(
                f"metasyn_packet_rescue_v3_file_symlink:{relative_path}"
            )
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise MetaSynPassagePacketRescueV3Error(
            f"metasyn_packet_rescue_v3_file_not_regular:{relative_path}"
        )
    return resolved


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_artifact_unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_artifact_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_artifact_not_object")
    return value


def load_metasyn_passage_packet_rescue_config_v3(
    *, repository_root: Path
) -> tuple[MetaSynPassagePacketRescueConfigV3, str]:
    root = _canonical_repository_root(repository_root)
    path = _checked_file(root=root, relative_path=DEFAULT_CONFIG_PATH.as_posix())
    try:
        config = MetaSynPassagePacketRescueConfigV3.model_validate(_read_object(path))
    except ValueError as exc:
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_config_invalid") from exc
    return config, sha256_file(path)


class RescueArtifactBindingV3(_FrozenExactModel):
    relative_path: str
    sha256: str
    utf8_bytes: Annotated[int, Field(ge=1)]

    @field_validator("sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "sha256")


class V2AttemptedPacketIdentityV3(_FrozenExactModel):
    request_key: str
    row_ordinal: Annotated[int, Field(ge=0, lt=32)]
    candidate_index: Annotated[int, Field(ge=1, le=8)]
    packet_request_sha256: str
    request_sha256: str
    intent_sha256: str
    provider_receipt_sha256: str
    provider_result_sha256: str
    packet_input_sha256: str
    candidate_descriptor_sha256: str
    candidate_binding_sha256: str
    scientific_request_signature_sha256: str

    @field_validator(
        "packet_request_sha256",
        "request_sha256",
        "intent_sha256",
        "provider_receipt_sha256",
        "provider_result_sha256",
        "packet_input_sha256",
        "candidate_descriptor_sha256",
        "candidate_binding_sha256",
        "scientific_request_signature_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)


InventoryFailureClassV3 = Literal[
    "candidates_not_canonical",
    "passage_ids_not_sorted_unique",
    "outcome_concept_not_exact_protocol_quote",
]


class V2InventoryFailureForensicV3(_FrozenExactModel):
    row_ordinal: Annotated[int, Field(ge=0, lt=32)]
    row_key: str
    provider_receipt_sha256: str
    provider_result_sha256: str
    raw_inventory_sha256: str
    failure_class: InventoryFailureClassV3
    inventory_normalized_for_current_rescue: Literal[False] = False
    current_rescue_candidate_authority: Literal[False] = False
    future_lossless_representational_canonicalization_evaluated: Literal[True] = True
    future_representational_canonicalization_revalidates: bool
    future_representational_candidate_count: Annotated[int, Field(ge=0, le=72)]

    @field_validator("provider_receipt_sha256", "provider_result_sha256", "raw_inventory_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)


class MetaSynV2ReplaySnapshotV3(_FrozenExactModel):
    snapshot_version: Literal["metasyn-passage-v2-replay-snapshot-v3"] = V2_REPLAY_SNAPSHOT_VERSION
    status: Literal["all_current_v2_base_artifacts_externally_replayed_before_new_liability"]
    execution_bundle_sha256: str
    inventory_ledger_sha256: str
    packet_roster_sha256: str
    failed_smoke_sha256: str
    v2_current_stage: Literal["packet_roster_frozen"]
    json_artifacts: Annotated[list[RescueArtifactBindingV3], Field(min_length=1)]
    json_artifact_membership_sha256: str
    json_artifact_count: Annotated[int, Field(ge=1)]
    provider_receipt_count: Literal[43]
    provider_receipt_membership_sha256: str
    provider_intent_membership_sha256: str
    provider_result_membership_sha256: str
    attempted_packet_requests: Annotated[
        list[V2AttemptedPacketIdentityV3], Field(min_length=3, max_length=3)
    ]
    attempted_packet_request_membership_sha256: str
    attempted_packet_intent_membership_sha256: str
    attempted_packet_provider_result_membership_sha256: str
    attempted_packet_scientific_request_membership_sha256: str
    inventory_failures: Annotated[
        list[V2InventoryFailureForensicV3], Field(min_length=10, max_length=10)
    ]
    inventory_failure_membership_sha256: str
    future_representational_recovery_row_count: Literal[9]
    future_representational_recovery_candidate_count: Literal[42]
    scientifically_invalid_inventory_row_count: Literal[1]
    inventory_normalization_performed: Literal[False] = False
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    provider_calls_made_by_replay: Literal[0] = 0
    snapshot_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "inventory_ledger_sha256",
        "packet_roster_sha256",
        "failed_smoke_sha256",
        "json_artifact_membership_sha256",
        "provider_receipt_membership_sha256",
        "provider_intent_membership_sha256",
        "provider_result_membership_sha256",
        "attempted_packet_request_membership_sha256",
        "attempted_packet_intent_membership_sha256",
        "attempted_packet_provider_result_membership_sha256",
        "attempted_packet_scientific_request_membership_sha256",
        "inventory_failure_membership_sha256",
        "snapshot_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_snapshot(self) -> MetaSynV2ReplaySnapshotV3:
        if self.json_artifacts != sorted(self.json_artifacts, key=lambda item: item.relative_path):
            raise ValueError("metasyn_packet_rescue_v3_artifact_manifest_not_canonical")
        if len({item.relative_path for item in self.json_artifacts}) != len(self.json_artifacts):
            raise ValueError("metasyn_packet_rescue_v3_artifact_manifest_duplicate_path")
        if self.json_artifact_count != len(self.json_artifacts):
            raise ValueError("metasyn_packet_rescue_v3_artifact_count_mismatch")
        if self.json_artifact_membership_sha256 != hash_canonical(
            [item.model_dump(mode="json") for item in self.json_artifacts]
        ):
            raise ValueError("metasyn_packet_rescue_v3_artifact_membership_mismatch")
        if [item.request_key for item in self.attempted_packet_requests] != [
            "packet-row-02-candidate-01",
            "packet-row-02-candidate-02",
            "packet-row-03-candidate-01",
        ]:
            raise ValueError("metasyn_packet_rescue_v3_attempted_roster_mismatch")
        if (
            self.attempted_packet_request_membership_sha256
            != hash_canonical(
                [item.packet_request_sha256 for item in self.attempted_packet_requests]
            )
            or self.attempted_packet_intent_membership_sha256
            != hash_canonical([item.intent_sha256 for item in self.attempted_packet_requests])
            or self.attempted_packet_provider_result_membership_sha256
            != hash_canonical(
                [item.provider_result_sha256 for item in self.attempted_packet_requests]
            )
            or self.attempted_packet_scientific_request_membership_sha256
            != hash_canonical(
                [
                    item.scientific_request_signature_sha256
                    for item in self.attempted_packet_requests
                ]
            )
        ):
            raise ValueError("metasyn_packet_rescue_v3_attempted_membership_mismatch")
        if [item.row_ordinal for item in self.inventory_failures] != [
            9,
            13,
            18,
            19,
            21,
            25,
            26,
            27,
            29,
            30,
        ]:
            raise ValueError("metasyn_packet_rescue_v3_inventory_failure_roster_mismatch")
        if self.inventory_failure_membership_sha256 != hash_canonical(
            [item.model_dump(mode="json") for item in self.inventory_failures]
        ):
            raise ValueError("metasyn_packet_rescue_v3_inventory_failure_membership_mismatch")
        if (
            sum(
                item.future_representational_canonicalization_revalidates
                for item in self.inventory_failures
            )
            != self.future_representational_recovery_row_count
            or sum(item.future_representational_candidate_count for item in self.inventory_failures)
            != self.future_representational_recovery_candidate_count
            or sum(
                not item.future_representational_canonicalization_revalidates
                for item in self.inventory_failures
            )
            != self.scientifically_invalid_inventory_row_count
        ):
            raise ValueError("metasyn_packet_rescue_v3_inventory_recovery_count_mismatch")
        payload = self.model_dump(mode="json", exclude={"snapshot_sha256"})
        if self.snapshot_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_packet_rescue_v3_snapshot_hash_mismatch")
        return self


class V2CompactSmokeForensicItemV3(_FrozenExactModel):
    request_key: str
    row_ordinal: Annotated[int, Field(ge=0, lt=32)]
    candidate_index: Annotated[int, Field(ge=1, le=8)]
    packet_request_sha256: str
    provider_receipt_sha256: str
    provider_result_sha256: str
    response_text_sha256: str
    candidate_binding_sha256: str
    raw_candidate_binding_already_matched: Literal[True] = True
    normalization_receipt: NativePacketCompactNormalizationReceiptV3
    normalization_receipt_sha256: str
    grounding_abstention_receipt: PacketGroundingAbstentionReceiptV2
    grounding_abstention_receipt_sha256: str
    forensic_item_sha256: str

    @field_validator(
        "packet_request_sha256",
        "provider_receipt_sha256",
        "provider_result_sha256",
        "response_text_sha256",
        "candidate_binding_sha256",
        "normalization_receipt_sha256",
        "grounding_abstention_receipt_sha256",
        "forensic_item_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_item(self) -> V2CompactSmokeForensicItemV3:
        if (
            self.normalization_receipt_sha256 != self.normalization_receipt.receipt_sha256
            or self.grounding_abstention_receipt_sha256
            != self.grounding_abstention_receipt.receipt_sha256
            or self.normalization_receipt.raw_candidate_binding_sha256
            != self.candidate_binding_sha256
        ):
            raise ValueError("metasyn_packet_rescue_v3_forensic_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"forensic_item_sha256"})
        if self.forensic_item_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_packet_rescue_v3_forensic_item_hash_mismatch")
        return self


class MetaSynV2CompactSmokeForensicReceiptV3(_FrozenExactModel):
    forensic_version: Literal["metasyn-passage-v2-compact-forensic-receipt-v3"] = (
        V2_FORENSIC_RECEIPT_VERSION
    )
    status: Literal[
        "three_raw_bindings_matched_and_constant_expansion_yielded_three_valid_abstentions"
    ]
    v2_replay_snapshot_sha256: str
    v2_execution_bundle_sha256: str
    v2_failed_smoke_sha256: str
    raw_output_count: Literal[3]
    raw_candidate_binding_match_count: Literal[3]
    valid_grounding_abstention_count: Literal[3]
    completed_typed_effect_count: Literal[0]
    items: Annotated[list[V2CompactSmokeForensicItemV3], Field(min_length=3, max_length=3)]
    item_membership_sha256: str
    raw_outputs_omitted_only_normalizable_invariant_constants: Literal[True] = True
    deterministic_expansion_idempotent: Literal[True] = True
    original_v2_smoke_status: Literal["failed_gate"] = "failed_gate"
    original_v2_remaining_packet_calls_permitted: Literal[False] = False
    normalized_abstentions_do_not_pass_gate: Literal[True] = True
    v2_failed_gate_semantics_changed: Literal[False] = False
    inventory_contract_failures_separate_from_compact_constant_bug: Literal[True] = True
    inventory_normalization_performed: Literal[False] = False
    forensic_post_hoc_only: Literal[True] = True
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    forensic_receipt_sha256: str

    @field_validator(
        "v2_replay_snapshot_sha256",
        "v2_execution_bundle_sha256",
        "v2_failed_smoke_sha256",
        "item_membership_sha256",
        "forensic_receipt_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_receipt(self) -> MetaSynV2CompactSmokeForensicReceiptV3:
        if self.item_membership_sha256 != hash_canonical(
            [item.forensic_item_sha256 for item in self.items]
        ):
            raise ValueError("metasyn_packet_rescue_v3_forensic_membership_mismatch")
        payload = self.model_dump(mode="json", exclude={"forensic_receipt_sha256"})
        if self.forensic_receipt_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_packet_rescue_v3_forensic_hash_mismatch")
        return self


class MetaSynRescueSelectionFeaturesV3(_FrozenExactModel):
    source_content_scope: Literal["full_text_sections", "title_abstract"]
    release_grade_source_grounding_eligible: bool
    projection_selection_complete: bool
    source_strength_blocker_count: Annotated[int, Field(ge=0)]
    effect_kind: str
    direct_confidence_interval: bool
    literal_confidence_interval_visible: bool
    literal_ratio_measure_visible: bool
    minimum_prompt_rank: Annotated[int, Field(ge=1)]
    candidate_passage_text_sha256: str
    deterministic_feature_vector: list[int]
    feature_sha256: str

    @field_validator("candidate_passage_text_sha256", "feature_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_features(self) -> MetaSynRescueSelectionFeaturesV3:
        expected = [
            int(self.release_grade_source_grounding_eligible),
            int(self.source_content_scope == "full_text_sections"),
            int(self.projection_selection_complete),
            int(self.source_strength_blocker_count == 0),
            int(self.direct_confidence_interval),
            int(self.literal_confidence_interval_visible),
            int(self.literal_ratio_measure_visible),
        ]
        if self.deterministic_feature_vector != expected:
            raise ValueError("metasyn_packet_rescue_v3_feature_vector_mismatch")
        payload = self.model_dump(mode="json", exclude={"feature_sha256"})
        if self.feature_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_packet_rescue_v3_feature_hash_mismatch")
        return self


SelectionDispositionV3 = Literal[
    "eligible_ranked",
    "excluded_previously_attempted",
    "excluded_not_release_grade_full_text",
    "excluded_not_direct_confidence_interval",
    "excluded_literal_ci_or_ratio_not_visible",
]


class MetaSynRescueCandidateAuditV3(_FrozenExactModel):
    row_ordinal: Annotated[int, Field(ge=0, lt=32)]
    candidate_index: Annotated[int, Field(ge=1, le=8)]
    row_key: str
    packet_request_sha256: str
    packet_input_sha256: str
    candidate_descriptor_sha256: str
    candidate_binding_sha256: str
    scientific_request_signature_sha256: str
    features: MetaSynRescueSelectionFeaturesV3
    disposition: SelectionDispositionV3
    eligible_rank: Annotated[int, Field(ge=1)] | None
    selected_for_smoke: bool
    audit_sha256: str

    @field_validator(
        "packet_request_sha256",
        "packet_input_sha256",
        "candidate_descriptor_sha256",
        "candidate_binding_sha256",
        "scientific_request_signature_sha256",
        "audit_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_audit(self) -> MetaSynRescueCandidateAuditV3:
        if (self.eligible_rank is not None) != (self.disposition == "eligible_ranked"):
            raise ValueError("metasyn_packet_rescue_v3_eligible_rank_shape_invalid")
        if self.selected_for_smoke and self.eligible_rank not in {1, 2, 3}:
            raise ValueError("metasyn_packet_rescue_v3_smoke_selection_invalid")
        payload = self.model_dump(mode="json", exclude={"audit_sha256"})
        if self.audit_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_packet_rescue_v3_audit_hash_mismatch")
        return self


class MetaSynPassageRescueRequestV3(_FrozenExactModel):
    rescue_request_version: Literal["metasyn-passage-packet-rescue-request-v3"] = (
        "metasyn-passage-packet-rescue-request-v3"
    )
    eligible_rank: Annotated[int, Field(ge=1, le=3)]
    row_ordinal: Annotated[int, Field(ge=0, lt=32)]
    candidate_index: Annotated[int, Field(ge=1, le=8)]
    source_packet_request: PacketRequestV2
    source_packet_request_sha256: str
    candidate_binding_sha256: str
    scientific_request_signature_sha256: str
    selection_feature_sha256: str
    request: AnthropicBoundedRequestV1
    request_sha256: str
    request_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    fresh_request_key: Literal[True] = True
    fresh_intent_domain: Literal[True] = True
    previously_attempted_scientific_request: Literal[False] = False
    rescue_request_sha256: str

    @field_validator(
        "source_packet_request_sha256",
        "candidate_binding_sha256",
        "scientific_request_signature_sha256",
        "selection_feature_sha256",
        "request_sha256",
        "rescue_request_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_request(self) -> MetaSynPassageRescueRequestV3:
        if (
            self.source_packet_request_sha256 != self.source_packet_request.packet_request_sha256
            or self.candidate_binding_sha256
            != self.source_packet_request.packet_input.candidate_binding_sha256
            or self.request_sha256 != self.request.request_sha256
            or self.row_ordinal != self.source_packet_request.row_ordinal
            or self.candidate_index != self.source_packet_request.candidate_index
            or self.request.operation != "metasyn-passage-packet-rescue-v3"
            or not self.request.request_key.startswith("rescue-v3-")
            or self.request_cost_ceiling_usd_micros
            != _usd_micros(self.request.cost_ceiling.request_cost_ceiling_usd)
        ):
            raise ValueError("metasyn_packet_rescue_v3_request_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"rescue_request_sha256"})
        if self.rescue_request_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_packet_rescue_v3_request_hash_mismatch")
        return self


class MetaSynPassageRescuePreCallBlockerItemV3(_FrozenExactModel):
    blocker_item_version: Literal["metasyn-passage-packet-rescue-pre-call-blocker-item-v3"] = (
        "metasyn-passage-packet-rescue-pre-call-blocker-item-v3"
    )
    selected_rank: Annotated[int, Field(ge=1, le=3)]
    row_ordinal: Annotated[int, Field(ge=0, lt=32)]
    candidate_index: Annotated[int, Field(ge=1, le=8)]
    rescue_request_sha256: str
    source_packet_request_sha256: str
    candidate_binding_sha256: str
    candidate_passage_ids: list[str]
    evidence_quote: str
    evidence_quote_sha256: str
    ci_lower_token: str
    ci_upper_token: str
    exact_range_separator: Literal["\N{EN DASH}"]
    exact_range_separator_code_point: Literal["U+2013"] = "U+2013"
    exact_range_expression: str
    effect_format_token: str
    exact_effect_format_alias_supported_by_v2: bool
    constructed_source_copied_probe: dict[str, Any]
    constructed_source_copied_probe_sha256: str
    full_v2_model_contract_valid: bool
    compact_normalization_receipt: NativePacketCompactNormalizationReceiptV3 | None
    compact_normalization_receipt_sha256: str | None
    immutable_v2_first_failure_code: str
    immutable_v2_first_failure_code_path: str
    upper_token_v2_valid_occurrence_count: Literal[0]
    static_blocker_codes: list[str]
    fails_only_at_immutable_numeric_boundary_gate: bool
    numerical_effect_values_copied_exactly_from_source: Literal[True] = True
    provider_call_required_to_establish_blocker: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    blocker_item_sha256: str

    @field_validator(
        "rescue_request_sha256",
        "source_packet_request_sha256",
        "candidate_binding_sha256",
        "evidence_quote_sha256",
        "constructed_source_copied_probe_sha256",
        "blocker_item_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("compact_normalization_receipt_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else _validate_sha256(value, "compact_normalization_receipt_sha256")
        )

    @model_validator(mode="after")
    def validate_blocker_item(self) -> MetaSynPassageRescuePreCallBlockerItemV3:
        expected_by_candidate = {
            (16, 1): {
                "rank": 1,
                "lower": "18.0",
                "upper": "1005",
                "format": "odds ratio",
                "format_supported": True,
                "failure": "packet_grounding_v2_numeric_token_absent:effect.ci_upper",
                "path": (
                    "native_packet_grounding_v2.freeze_passage_packet_grounding_receipt_v2"
                    "->_freeze_numeric_receipts->_numeric_token_occurrences"
                    "->_assert_numeric_token_boundary"
                ),
                "blockers": ["numeric_upper_token_rejected_after_en_dash"],
            },
            (20, 1): {
                "rank": 2,
                "lower": "0.62",
                "upper": "0.92",
                "format": "rate ratio",
                "format_supported": False,
                "failure": "packet_grounding_v2_effect_format_alias_unsupported",
                "path": (
                    "native_packet_grounding_v2.PacketGroundingModelCompletedV2"
                    ".effect_format_token->_effect_format_from_exact_token"
                ),
                "blockers": [
                    "exact_effect_format_alias_unsupported_by_v2",
                    "numeric_upper_token_rejected_after_en_dash",
                ],
            },
            (20, 2): {
                "rank": 3,
                "lower": "0.65",
                "upper": "1.34",
                "format": "RR",
                "format_supported": False,
                "failure": "packet_grounding_v2_effect_format_alias_unsupported",
                "path": (
                    "native_packet_grounding_v2.PacketGroundingModelCompletedV2"
                    ".effect_format_token->_effect_format_from_exact_token"
                ),
                "blockers": [
                    "exact_effect_format_alias_unsupported_by_v2",
                    "numeric_upper_token_rejected_after_en_dash",
                ],
            },
        }
        expected = expected_by_candidate.get((self.row_ordinal, self.candidate_index))
        numeric_claims = {
            item.get("field_path"): item.get("verbatim_numeric_token")
            for item in self.constructed_source_copied_probe.get("numeric_claims", [])
            if isinstance(item, dict)
        }
        receipt_expected = bool(expected and expected["format_supported"])
        if (
            expected is None
            or self.selected_rank != expected["rank"]
            or self.ci_lower_token != expected["lower"]
            or self.ci_upper_token != expected["upper"]
            or self.effect_format_token != expected["format"]
            or self.exact_effect_format_alias_supported_by_v2 != expected["format_supported"]
            or self.immutable_v2_first_failure_code != expected["failure"]
            or self.immutable_v2_first_failure_code_path != expected["path"]
            or self.static_blocker_codes != expected["blockers"]
            or self.evidence_quote_sha256 != hash_canonical(self.evidence_quote)
            or self.constructed_source_copied_probe_sha256
            != hash_canonical(self.constructed_source_copied_probe)
            or self.full_v2_model_contract_valid != receipt_expected
            or (self.compact_normalization_receipt is not None) != receipt_expected
            or (self.compact_normalization_receipt_sha256 is not None) != receipt_expected
            or (
                self.compact_normalization_receipt is not None
                and (
                    self.compact_normalization_receipt_sha256
                    != self.compact_normalization_receipt.receipt_sha256
                    or self.compact_normalization_receipt.raw_model_outcome
                    != self.constructed_source_copied_probe
                    or self.compact_normalization_receipt.expected_candidate_binding_sha256
                    != self.candidate_binding_sha256
                )
            )
            or self.constructed_source_copied_probe.get("candidate_binding_sha256")
            != self.candidate_binding_sha256
            or self.constructed_source_copied_probe.get("evidence_quote") != self.evidence_quote
            or self.constructed_source_copied_probe.get("effect_format_token")
            != self.effect_format_token
            or numeric_claims.get("effect.ci_lower") != self.ci_lower_token
            or numeric_claims.get("effect.ci_upper") != self.ci_upper_token
            or self.exact_range_expression
            != f"{self.ci_lower_token}{self.exact_range_separator}{self.ci_upper_token}"
            or self.exact_range_expression not in self.evidence_quote
            or self.static_blocker_codes != sorted(set(self.static_blocker_codes))
        ):
            raise ValueError("metasyn_packet_rescue_v3_blocker_item_alias_mismatch")
        expected_only_boundary = self.static_blocker_codes == [
            "numeric_upper_token_rejected_after_en_dash"
        ]
        if self.fails_only_at_immutable_numeric_boundary_gate != expected_only_boundary:
            raise ValueError("metasyn_packet_rescue_v3_blocker_only_boundary_mismatch")
        payload = self.model_dump(mode="json", exclude={"blocker_item_sha256"})
        if self.blocker_item_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_packet_rescue_v3_blocker_item_hash_mismatch")
        return self


class MetaSynPassageRescuePreCallBlockerV3(_FrozenExactModel):
    blocker_version: Literal["metasyn-passage-packet-rescue-pre-call-blocker-v3"] = (
        RESCUE_PRE_CALL_BLOCKER_VERSION
    )
    status: Literal["selected_v2_candidates_proven_unreachable_before_provider_liability"]
    selected_request_membership_sha256: str
    items: Annotated[
        list[MetaSynPassageRescuePreCallBlockerItemV3], Field(min_length=3, max_length=3)
    ]
    item_membership_sha256: str
    selected_candidate_count: Literal[3]
    selected_candidate_v2_reachable_count: Literal[0]
    numeric_boundary_blocked_candidate_count: Literal[3]
    unsupported_exact_effect_format_candidate_count: Literal[2]
    row16_completed_contract_probe_fails_only_at_numeric_boundary: Literal[True] = True
    provider_calls_made: Literal[0] = 0
    authorization_created: Literal[False] = False
    calls_permitted: Literal[False] = False
    provider_cost_liability_usd_micros: Literal[0] = 0
    additive_successor_required: Literal[True] = True
    v2_artifacts_or_semantics_changed: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    blocker_sha256: str

    @field_validator(
        "selected_request_membership_sha256",
        "item_membership_sha256",
        "blocker_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_blocker(self) -> MetaSynPassageRescuePreCallBlockerV3:
        if [(item.row_ordinal, item.candidate_index) for item in self.items] != [
            (16, 1),
            (20, 1),
            (20, 2),
        ]:
            raise ValueError("metasyn_packet_rescue_v3_blocker_roster_mismatch")
        if [item.selected_rank for item in self.items] != [1, 2, 3]:
            raise ValueError("metasyn_packet_rescue_v3_blocker_rank_mismatch")
        if self.item_membership_sha256 != hash_canonical(
            [item.blocker_item_sha256 for item in self.items]
        ):
            raise ValueError("metasyn_packet_rescue_v3_blocker_membership_mismatch")
        if self.selected_request_membership_sha256 != hash_canonical(
            [item.rescue_request_sha256 for item in self.items]
        ):
            raise ValueError("metasyn_packet_rescue_v3_blocker_request_membership_mismatch")
        if sum(item.fails_only_at_immutable_numeric_boundary_gate for item in self.items) != 1:
            raise ValueError("metasyn_packet_rescue_v3_blocker_boundary_count_mismatch")
        if (
            sum(not item.exact_effect_format_alias_supported_by_v2 for item in self.items)
            != self.unsupported_exact_effect_format_candidate_count
        ):
            raise ValueError("metasyn_packet_rescue_v3_blocker_format_count_mismatch")
        payload = self.model_dump(mode="json", exclude={"blocker_sha256"})
        if self.blocker_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_packet_rescue_v3_blocker_hash_mismatch")
        return self


class MetaSynPassagePacketRescuePlanV3(_FrozenExactModel):
    plan_version: Literal["metasyn-passage-packet-rescue-plan-v3"] = RESCUE_PLAN_VERSION
    status: Literal["frozen_post_hoc_label_blind_unattempted_candidate_plan_pre_call_blocked"]
    config: MetaSynPassagePacketRescueConfigV3
    config_sha256: str
    config_file_sha256: str
    v2_replay_snapshot: MetaSynV2ReplaySnapshotV3
    v2_replay_snapshot_sha256: str
    v2_forensic_receipt: MetaSynV2CompactSmokeForensicReceiptV3
    v2_forensic_receipt_sha256: str
    pipeline_fingerprint: PipelineFingerprint
    pipeline_sha256: str
    candidate_audits: Annotated[list[MetaSynRescueCandidateAuditV3], Field(min_length=1)]
    candidate_audit_membership_sha256: str
    eligible_candidate_audit_sha256s: Annotated[list[str], Field(min_length=3)]
    eligible_candidate_membership_sha256: str
    requests: Annotated[list[MetaSynPassageRescueRequestV3], Field(min_length=3, max_length=3)]
    request_membership_sha256: str
    request_count: Literal[3]
    pre_call_blocker: MetaSynPassageRescuePreCallBlockerV3
    pre_call_blocker_sha256: str
    provider_calls_permitted: Literal[False] = False
    authorization_created: Literal[False] = False
    conservative_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    configured_cost_cap_usd_micros: Literal[2200000]
    attempted_v2_request_membership_sha256: str
    attempted_v2_intent_membership_sha256: str
    attempted_v2_provider_result_membership_sha256: str
    attempted_v2_scientific_request_membership_sha256: str
    all_requests_unattempted_under_scientific_signature: Literal[True] = True
    provider_calls_made: Literal[False] = False
    post_hoc_exploratory: Literal[True] = True
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    plan_sha256: str

    @field_validator(
        "config_sha256",
        "config_file_sha256",
        "v2_replay_snapshot_sha256",
        "v2_forensic_receipt_sha256",
        "pipeline_sha256",
        "candidate_audit_membership_sha256",
        "eligible_candidate_membership_sha256",
        "request_membership_sha256",
        "pre_call_blocker_sha256",
        "attempted_v2_request_membership_sha256",
        "attempted_v2_intent_membership_sha256",
        "attempted_v2_provider_result_membership_sha256",
        "attempted_v2_scientific_request_membership_sha256",
        "plan_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("eligible_candidate_audit_sha256s")
    @classmethod
    def validate_eligible_hashes(cls, value: list[str]) -> list[str]:
        for item in value:
            _validate_sha256(item, "eligible_candidate_audit_sha256s")
        if len(value) != len(set(value)):
            raise ValueError("metasyn_packet_rescue_v3_eligible_hashes_duplicate")
        return value

    @model_validator(mode="after")
    def validate_plan(self) -> MetaSynPassagePacketRescuePlanV3:
        if (
            self.config_sha256 != self.config.config_sha256
            or self.v2_replay_snapshot_sha256 != self.v2_replay_snapshot.snapshot_sha256
            or self.v2_forensic_receipt_sha256 != self.v2_forensic_receipt.forensic_receipt_sha256
            or self.v2_forensic_receipt.v2_replay_snapshot_sha256 != self.v2_replay_snapshot_sha256
            or self.pipeline_sha256 != self.pipeline_fingerprint.pipeline_sha256
            or self.pre_call_blocker_sha256 != self.pre_call_blocker.blocker_sha256
            or self.pre_call_blocker.selected_request_membership_sha256
            != self.request_membership_sha256
        ):
            raise ValueError("metasyn_packet_rescue_v3_plan_alias_mismatch")
        if [(item.row_ordinal, item.candidate_index) for item in self.candidate_audits] != sorted(
            (item.row_ordinal, item.candidate_index) for item in self.candidate_audits
        ):
            raise ValueError("metasyn_packet_rescue_v3_candidate_audits_not_canonical")
        if self.candidate_audit_membership_sha256 != hash_canonical(
            [item.audit_sha256 for item in self.candidate_audits]
        ):
            raise ValueError("metasyn_packet_rescue_v3_candidate_membership_mismatch")
        eligible = sorted(
            (item for item in self.candidate_audits if item.eligible_rank is not None),
            key=lambda item: item.eligible_rank or 0,
        )
        if [item.eligible_rank for item in eligible] != list(range(1, len(eligible) + 1)):
            raise ValueError("metasyn_packet_rescue_v3_eligible_ranks_not_contiguous")
        if self.eligible_candidate_audit_sha256s != [item.audit_sha256 for item in eligible]:
            raise ValueError("metasyn_packet_rescue_v3_eligible_roster_mismatch")
        if self.eligible_candidate_membership_sha256 != hash_canonical(
            self.eligible_candidate_audit_sha256s
        ):
            raise ValueError("metasyn_packet_rescue_v3_eligible_membership_mismatch")
        if [item.eligible_rank for item in self.requests] != [1, 2, 3]:
            raise ValueError("metasyn_packet_rescue_v3_request_rank_mismatch")
        if self.request_membership_sha256 != hash_canonical(
            [item.rescue_request_sha256 for item in self.requests]
        ):
            raise ValueError("metasyn_packet_rescue_v3_request_membership_mismatch")
        if self.conservative_cost_ceiling_usd_micros != sum(
            item.request_cost_ceiling_usd_micros for item in self.requests
        ):
            raise ValueError("metasyn_packet_rescue_v3_cost_sum_mismatch")
        if self.conservative_cost_ceiling_usd_micros > self.configured_cost_cap_usd_micros:
            raise ValueError("metasyn_packet_rescue_v3_cost_cap_exceeded")
        attempted = {
            item.scientific_request_signature_sha256
            for item in self.v2_replay_snapshot.attempted_packet_requests
        }
        if any(item.scientific_request_signature_sha256 in attempted for item in self.requests):
            raise ValueError("metasyn_packet_rescue_v3_attempted_scientific_overlap")
        payload = self.model_dump(mode="json", exclude={"plan_sha256"})
        if self.plan_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_packet_rescue_v3_plan_hash_mismatch")
        return self


@dataclass(frozen=True)
class _V2ReplayContext:
    bundle: MetaSynPassageHostedExecutionBundleV2
    inventory_ledger: InventoryLedgerV2
    packet_roster: PacketRosterV2
    smoke: PacketSmokeReceiptV2
    attempted_outcomes: tuple[HostedExactOnceProviderReceiptV1, ...]
    attempted_results: tuple[PacketCallResultV2, ...]
    snapshot: MetaSynV2ReplaySnapshotV3


def _canonical_existing_workspace(value: Path, *, name: str) -> Path:
    if value.is_symlink():
        raise MetaSynPassagePacketRescueV3Error(
            f"metasyn_packet_rescue_v3_{name}_workspace_symlink"
        )
    try:
        resolved = value.resolve(strict=True)
    except OSError as exc:
        raise MetaSynPassagePacketRescueV3Error(
            f"metasyn_packet_rescue_v3_{name}_workspace_missing"
        ) from exc
    if not resolved.is_dir():
        raise MetaSynPassagePacketRescueV3Error(
            f"metasyn_packet_rescue_v3_{name}_workspace_not_directory"
        )
    return resolved


def _assert_tree_no_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_upstream_symlink_forbidden"
            )


def _artifact_manifest(root: Path) -> list[RescueArtifactBindingV3]:
    output: list[RescueArtifactBindingV3] = []
    for path in sorted(root.rglob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_upstream_artifact_unsafe"
            )
        raw = path.read_bytes()
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_upstream_json_invalid"
            ) from exc
        if not isinstance(value, dict):
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_upstream_json_not_object"
            )
        output.append(
            RescueArtifactBindingV3(
                relative_path=path.relative_to(root).as_posix(),
                sha256=sha256_file(path),
                utf8_bytes=len(raw),
            )
        )
    return output


def _scientific_request_signature(request: PacketRequestV2) -> str:
    packet = request.packet_input
    return hash_canonical(
        {
            "row_ordinal": request.row_ordinal,
            "row_key": request.row_key,
            "row_input_sha256": packet.row_input_sha256,
            "packet_input_sha256": packet.packet_input_sha256,
            "candidate_descriptor_sha256": packet.candidate_descriptor_sha256,
            "candidate_binding_sha256": packet.candidate_binding_sha256,
            "projection_surface_sha256": packet.projection_surface_sha256,
            "candidate_passage_surface_sha256": packet.candidate_passage_surface_sha256,
            "rendered_prompt_sha256": packet.rendered_prompt_sha256,
            "compiled_schema_sha256": request.compiled_schema_sha256,
            "effect_kind": packet.candidate.effect_kind,
        }
    )


def _v2_packet_intent(
    *, bundle: MetaSynPassageHostedExecutionBundleV2, request: PacketRequestV2
) -> HostedExactOnceIntentV1:
    return freeze_hosted_exact_once_intent(
        execution_bundle_sha256=bundle.execution_bundle_sha256,
        phase="packet",
        source_bearing=True,
        context_binding_sha256=request.packet_request_sha256,
        request=request.request,
    )


def _exception_chain_text(exc: BaseException) -> str:
    values: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        values.append(str(current))
        current = current.__cause__ or current.__context__
    return "\n".join(values)


def _classify_inventory_failure(
    *, raw: Mapping[str, Any], bundle: MetaSynPassageHostedExecutionBundleV2, row_ordinal: int
) -> InventoryFailureClassV3:
    row = bundle.extraction_inputs.rows[row_ordinal]
    try:
        validate_metasyn_candidate_inventory_v2(
            raw,
            allowed_outcome_text_by_id=row.question_surface.allowed_outcome_text_by_id,
            passage_text_by_id={
                passage.passage_id: passage.text for passage in row.projection_surface.passages
            },
        )
    except (ValueError, MetaSynCandidateInventoryV2Error) as exc:
        detail = _exception_chain_text(exc)
        if "metasyn_inventory_v2_candidates_not_canonical" in detail:
            return "candidates_not_canonical"
        if "metasyn_inventory_v2_passage_ids_not_sorted_unique" in detail:
            return "passage_ids_not_sorted_unique"
        if "metasyn_inventory_v2_outcome_concept_not_exact_protocol_quote" in detail:
            return "outcome_concept_not_exact_protocol_quote"
        raise MetaSynPassagePacketRescueV3Error(
            f"metasyn_packet_rescue_v3_unknown_inventory_failure:{row_ordinal}"
        ) from exc
    raise MetaSynPassagePacketRescueV3Error(
        f"metasyn_packet_rescue_v3_expected_inventory_failure_missing:{row_ordinal}"
    )


def _representational_inventory_replay(
    *, raw: Mapping[str, Any], bundle: MetaSynPassageHostedExecutionBundleV2, row_ordinal: int
) -> tuple[bool, int]:
    candidate_values = raw.get("candidates")
    if not isinstance(candidate_values, list):
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_inventory_candidates_not_list"
        )
    canonicalized = deepcopy(dict(raw))
    candidates: list[dict[str, Any]] = []
    for value in candidate_values:
        if not isinstance(value, Mapping):
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_inventory_candidate_not_object"
            )
        candidate = deepcopy(dict(value))
        passages = candidate.get("passage_ids")
        if not isinstance(passages, list) or any(not isinstance(item, str) for item in passages):
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_inventory_passages_invalid"
            )
        candidate["passage_ids"] = sorted(set(passages))
        candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            item["passage_ids"][0],
            item["canonical_outcome_id"],
            " ".join(item["outcome_concept_quote"].casefold().split()),
            item["effect_kind"],
        )
    )
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidate_index"] = index
    canonicalized["candidates"] = candidates
    row = bundle.extraction_inputs.rows[row_ordinal]
    try:
        validate_metasyn_candidate_inventory_v2(
            canonicalized,
            allowed_outcome_text_by_id=row.question_surface.allowed_outcome_text_by_id,
            passage_text_by_id={
                passage.passage_id: passage.text for passage in row.projection_surface.passages
            },
        )
    except (ValueError, MetaSynCandidateInventoryV2Error):
        return False, len(candidates)
    return True, len(candidates)


def _freeze_inventory_failures(
    *,
    v2_workspace: Path,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    ledger: InventoryLedgerV2,
) -> list[V2InventoryFailureForensicV3]:
    invalid_results = [
        item for item in ledger.results if item.validation_status == "inventory_contract_invalid"
    ]
    if len(invalid_results) != EXPECTED_V2_INVENTORY_INVALID_COUNT:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_inventory_invalid_count_mismatch"
        )
    output: list[V2InventoryFailureForensicV3] = []
    for result in invalid_results:
        path = (
            v2_workspace
            / "provider-state/provider-receipts"
            / f"inventory-row-{result.row_ordinal:02d}.json"
        )
        receipt = HostedExactOnceProviderReceiptV1.model_validate(_read_object(path))
        raw = receipt.provider_result.parsed_json
        if receipt.provider_result.outcome != "completed" or not isinstance(raw, Mapping):
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_inventory_failure_not_completed_object"
            )
        failure = _classify_inventory_failure(
            raw=raw, bundle=bundle, row_ordinal=result.row_ordinal
        )
        future_valid, candidate_count = _representational_inventory_replay(
            raw=raw, bundle=bundle, row_ordinal=result.row_ordinal
        )
        output.append(
            V2InventoryFailureForensicV3(
                row_ordinal=result.row_ordinal,
                row_key=result.row_key,
                provider_receipt_sha256=receipt.receipt_sha256,
                provider_result_sha256=receipt.provider_result_sha256,
                raw_inventory_sha256=hash_canonical(dict(raw)),
                failure_class=failure,
                future_representational_canonicalization_revalidates=future_valid,
                future_representational_candidate_count=(candidate_count if future_valid else 0),
            )
        )
    revalidated_rows = {
        item.row_ordinal
        for item in output
        if item.future_representational_canonicalization_revalidates
    }
    if revalidated_rows != _INVENTORY_REPRESENTATIONAL_ROWS:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_inventory_representational_rows_mismatch"
        )
    scientifically_invalid_rows = {
        item.row_ordinal
        for item in output
        if not item.future_representational_canonicalization_revalidates
    }
    if scientifically_invalid_rows != _INVENTORY_SCIENTIFICALLY_INVALID_ROWS:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_inventory_scientific_rows_mismatch"
        )
    recovered = sum(item.future_representational_candidate_count for item in output)
    if recovered != EXPECTED_FUTURE_REPRESENTATIONAL_RECOVERY_CANDIDATES:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_inventory_recovered_candidate_count_mismatch"
        )
    return output


def _replay_v2_base(*, repository_root: Path, v2_workspace: Path) -> _V2ReplayContext:
    root = _canonical_repository_root(repository_root)
    workspace = _canonical_existing_workspace(v2_workspace, name="v2")
    _assert_tree_no_symlinks(workspace)
    bundle = MetaSynPassageHostedExecutionBundleV2.model_validate(
        _read_object(workspace / "execution-bundle.json")
    )
    bundle = validate_metasyn_passage_hosted_execution_bundle_v2(
        execution_bundle=bundle, repository_root=root, external_replay=True
    )
    if bundle.execution_bundle_sha256 != EXPECTED_V2_EXECUTION_BUNDLE_SHA256:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_v2_bundle_anchor_mismatch"
        )
    status = metasyn_passage_hosted_runtime_status_v2(
        repository_root=root,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )
    if status["current_stage"] != "packet_roster_frozen":
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_v2_stage_mismatch")
    validate_metasyn_passage_preflight_v2(workspace=workspace, execution_bundle=bundle)
    ledger = validate_metasyn_passage_inventory_ledger_v2(
        workspace=workspace, execution_bundle=bundle
    )
    roster = validate_metasyn_passage_packet_roster_v2(workspace=workspace, execution_bundle=bundle)
    smoke = PacketSmokeReceiptV2.model_validate(
        _read_object(workspace / "packet-smoke-attempt.json")
    )
    if (
        ledger.ledger_sha256 != EXPECTED_V2_INVENTORY_LEDGER_SHA256
        or roster.roster_sha256 != EXPECTED_V2_PACKET_ROSTER_SHA256
        or smoke.smoke_sha256 != EXPECTED_V2_FAILED_SMOKE_SHA256
        or smoke.status != "failed_gate"
        or smoke.remaining_packet_calls_permitted
        or smoke.completed_typed_effect_result_sha256 is not None
        or len(smoke.ordered_smoke_request_keys) != EXPECTED_V2_PACKET_ATTEMPT_COUNT
    ):
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_v2_anchor_or_gate_mismatch"
        )
    request_by_key = {item.request.request_key: item for item in roster.requests}
    if set(smoke.ordered_smoke_request_keys) - set(request_by_key):
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_v2_attempt_not_in_roster")
    if roster.exact_authorization is None:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_v2_packet_authorization_missing"
        )
    attempted_outcomes: list[HostedExactOnceProviderReceiptV1] = []
    attempted_results: list[PacketCallResultV2] = []
    attempted_identities: list[V2AttemptedPacketIdentityV3] = []
    for index, request_key in enumerate(smoke.ordered_smoke_request_keys):
        request = request_by_key[request_key]
        intent = _v2_packet_intent(bundle=bundle, request=request)
        outcome = validate_hosted_exact_once_outcome(
            workspace=workspace / "provider-state",
            intent=intent,
            authorization=roster.exact_authorization,
        )
        if not isinstance(outcome, HostedExactOnceProviderReceiptV1):
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_v2_packet_terminal_not_receipt"
            )
        result = validate_metasyn_passage_packet_result_v2(
            workspace=workspace,
            execution_bundle=bundle,
            packet_request=request,
            intent=intent,
            authorization=roster.exact_authorization,
        )
        if (
            result.result_sha256 != smoke.attempted_result_sha256s[index]
            or result.validation_status != "grounding_invalid"
            or outcome.provider_result.outcome != "completed"
        ):
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_v2_packet_result_mismatch"
            )
        signature = _scientific_request_signature(request)
        attempted_outcomes.append(outcome)
        attempted_results.append(result)
        attempted_identities.append(
            V2AttemptedPacketIdentityV3(
                request_key=request_key,
                row_ordinal=request.row_ordinal,
                candidate_index=request.candidate_index,
                packet_request_sha256=request.packet_request_sha256,
                request_sha256=request.request_sha256,
                intent_sha256=intent.intent_sha256,
                provider_receipt_sha256=outcome.receipt_sha256,
                provider_result_sha256=outcome.provider_result_sha256,
                packet_input_sha256=request.packet_input_sha256,
                candidate_descriptor_sha256=(request.packet_input.candidate_descriptor_sha256),
                candidate_binding_sha256=request.packet_input.candidate_binding_sha256,
                scientific_request_signature_sha256=signature,
            )
        )
    provider_receipt_dir = workspace / "provider-state/provider-receipts"
    provider_intent_dir = workspace / "provider-state/call-intents"
    receipt_paths = sorted(provider_receipt_dir.glob("*.json"))
    intent_paths = sorted(provider_intent_dir.glob("*.json"))
    if (
        len(receipt_paths) != EXPECTED_V2_PROVIDER_RECEIPT_COUNT
        or len(intent_paths) != EXPECTED_V2_PROVIDER_RECEIPT_COUNT
    ):
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_v2_provider_file_count_mismatch"
        )
    receipts = [
        HostedExactOnceProviderReceiptV1.model_validate(_read_object(path))
        for path in receipt_paths
    ]
    provider_results = [item.provider_result_sha256 for item in receipts]
    failures = _freeze_inventory_failures(v2_workspace=workspace, bundle=bundle, ledger=ledger)
    artifacts = _artifact_manifest(workspace)
    snapshot_payload: dict[str, Any] = {
        "snapshot_version": V2_REPLAY_SNAPSHOT_VERSION,
        "status": ("all_current_v2_base_artifacts_externally_replayed_before_new_liability"),
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "inventory_ledger_sha256": ledger.ledger_sha256,
        "packet_roster_sha256": roster.roster_sha256,
        "failed_smoke_sha256": smoke.smoke_sha256,
        "v2_current_stage": status["current_stage"],
        "json_artifacts": artifacts,
        "json_artifact_membership_sha256": hash_canonical(
            [item.model_dump(mode="json") for item in artifacts]
        ),
        "json_artifact_count": len(artifacts),
        "provider_receipt_count": len(receipt_paths),
        "provider_receipt_membership_sha256": hash_canonical(
            [sha256_file(path) for path in receipt_paths]
        ),
        "provider_intent_membership_sha256": hash_canonical(
            [sha256_file(path) for path in intent_paths]
        ),
        "provider_result_membership_sha256": hash_canonical(provider_results),
        "attempted_packet_requests": attempted_identities,
        "attempted_packet_request_membership_sha256": hash_canonical(
            [item.packet_request_sha256 for item in attempted_identities]
        ),
        "attempted_packet_intent_membership_sha256": hash_canonical(
            [item.intent_sha256 for item in attempted_identities]
        ),
        "attempted_packet_provider_result_membership_sha256": hash_canonical(
            [item.provider_result_sha256 for item in attempted_identities]
        ),
        "attempted_packet_scientific_request_membership_sha256": hash_canonical(
            [item.scientific_request_signature_sha256 for item in attempted_identities]
        ),
        "inventory_failures": failures,
        "inventory_failure_membership_sha256": hash_canonical(
            [item.model_dump(mode="json") for item in failures]
        ),
        "future_representational_recovery_row_count": sum(
            item.future_representational_canonicalization_revalidates for item in failures
        ),
        "future_representational_recovery_candidate_count": sum(
            item.future_representational_candidate_count for item in failures
        ),
        "scientifically_invalid_inventory_row_count": sum(
            not item.future_representational_canonicalization_revalidates for item in failures
        ),
        "inventory_normalization_performed": False,
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "provider_calls_made_by_replay": 0,
    }
    snapshot = MetaSynV2ReplaySnapshotV3.model_validate(
        {**snapshot_payload, "snapshot_sha256": hash_canonical(snapshot_payload)}
    )
    return _V2ReplayContext(
        bundle=bundle,
        inventory_ledger=ledger,
        packet_roster=roster,
        smoke=smoke,
        attempted_outcomes=tuple(attempted_outcomes),
        attempted_results=tuple(attempted_results),
        snapshot=snapshot,
    )


def _freeze_v2_forensic_receipt(
    *, context: _V2ReplayContext
) -> MetaSynV2CompactSmokeForensicReceiptV3:
    request_by_key = {item.request.request_key: item for item in context.packet_roster.requests}
    items: list[V2CompactSmokeForensicItemV3] = []
    for request_key, outcome in zip(
        context.smoke.ordered_smoke_request_keys,
        context.attempted_outcomes,
        strict=True,
    ):
        request = request_by_key[request_key]
        raw = outcome.provider_result.parsed_json
        if not isinstance(raw, Mapping) or outcome.provider_result.response_text_sha256 is None:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_v2_compact_output_missing"
            )
        normalization = freeze_native_packet_compact_normalization_receipt_v3(
            raw_model_outcome=raw,
            expected_candidate_binding_sha256=(request.packet_input.candidate_binding_sha256),
        )
        normalization = validate_native_packet_compact_normalization_receipt_v3(
            receipt=normalization,
            raw_model_outcome=raw,
            expected_candidate_binding_sha256=(request.packet_input.candidate_binding_sha256),
        )
        grounding = freeze_passage_packet_grounding_receipt_v2(
            model_outcome=normalization.normalized_model_outcome,
            candidate=request.packet_input.candidate,
            projection=context.bundle.extraction_inputs.rows[request.row_ordinal].projection_v2,
        )
        grounding = validate_passage_packet_grounding_receipt_v2(
            receipt=grounding,
            model_outcome=normalization.normalized_model_outcome,
            candidate=request.packet_input.candidate,
            projection=context.bundle.extraction_inputs.rows[request.row_ordinal].projection_v2,
        )
        if not isinstance(grounding, PacketGroundingAbstentionReceiptV2):
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_v2_compact_not_abstention"
            )
        item_payload: dict[str, Any] = {
            "request_key": request_key,
            "row_ordinal": request.row_ordinal,
            "candidate_index": request.candidate_index,
            "packet_request_sha256": request.packet_request_sha256,
            "provider_receipt_sha256": outcome.receipt_sha256,
            "provider_result_sha256": outcome.provider_result_sha256,
            "response_text_sha256": outcome.provider_result.response_text_sha256,
            "candidate_binding_sha256": request.packet_input.candidate_binding_sha256,
            "raw_candidate_binding_already_matched": True,
            "normalization_receipt": normalization,
            "normalization_receipt_sha256": normalization.receipt_sha256,
            "grounding_abstention_receipt": grounding,
            "grounding_abstention_receipt_sha256": grounding.receipt_sha256,
        }
        items.append(
            V2CompactSmokeForensicItemV3.model_validate(
                {
                    **item_payload,
                    "forensic_item_sha256": hash_canonical(item_payload),
                }
            )
        )
    if any(item.normalization_receipt.branch != "unable_to_complete" for item in items):
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_v2_forensic_branch_mismatch"
        )
    payload: dict[str, Any] = {
        "forensic_version": V2_FORENSIC_RECEIPT_VERSION,
        "status": (
            "three_raw_bindings_matched_and_constant_expansion_yielded_three_valid_abstentions"
        ),
        "v2_replay_snapshot_sha256": context.snapshot.snapshot_sha256,
        "v2_execution_bundle_sha256": context.bundle.execution_bundle_sha256,
        "v2_failed_smoke_sha256": context.smoke.smoke_sha256,
        "raw_output_count": len(items),
        "raw_candidate_binding_match_count": len(items),
        "valid_grounding_abstention_count": len(items),
        "completed_typed_effect_count": 0,
        "items": items,
        "item_membership_sha256": hash_canonical([item.forensic_item_sha256 for item in items]),
        "raw_outputs_omitted_only_normalizable_invariant_constants": True,
        "deterministic_expansion_idempotent": True,
        "original_v2_smoke_status": "failed_gate",
        "original_v2_remaining_packet_calls_permitted": False,
        "normalized_abstentions_do_not_pass_gate": True,
        "v2_failed_gate_semantics_changed": False,
        "inventory_contract_failures_separate_from_compact_constant_bug": True,
        "inventory_normalization_performed": False,
        "forensic_post_hoc_only": True,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynV2CompactSmokeForensicReceiptV3.model_validate(
        {**payload, "forensic_receipt_sha256": hash_canonical(payload)}
    )


def _selection_features(request: PacketRequestV2) -> MetaSynRescueSelectionFeaturesV3:
    packet = request.packet_input
    strength = packet.candidate_passage_surface.source_strength
    passages = packet.candidate_passage_surface.passages
    exact_visible_text = "\n".join(item.text for item in passages)
    payload: dict[str, Any] = {
        "source_content_scope": strength.source_content_scope,
        "release_grade_source_grounding_eligible": (
            strength.release_grade_source_grounding_eligible
        ),
        "projection_selection_complete": strength.projection_v2_selection_complete,
        "source_strength_blocker_count": len(strength.source_strength_blockers),
        "effect_kind": packet.candidate.effect_kind,
        "direct_confidence_interval": (
            packet.candidate.effect_kind == "direct_confidence_interval"
        ),
        "literal_confidence_interval_visible": bool(_CI_LITERAL_RE.search(exact_visible_text)),
        "literal_ratio_measure_visible": bool(_RATIO_LITERAL_RE.search(exact_visible_text)),
        "minimum_prompt_rank": min(item.prompt_rank for item in passages),
        "candidate_passage_text_sha256": hash_canonical(
            [
                {
                    "passage_id": item.passage_id,
                    "prompt_rank": item.prompt_rank,
                    "text": item.text,
                    "text_sha256": item.text_sha256,
                }
                for item in passages
            ]
        ),
    }
    payload["deterministic_feature_vector"] = [
        int(payload["release_grade_source_grounding_eligible"]),
        int(payload["source_content_scope"] == "full_text_sections"),
        int(payload["projection_selection_complete"]),
        int(payload["source_strength_blocker_count"] == 0),
        int(payload["direct_confidence_interval"]),
        int(payload["literal_confidence_interval_visible"]),
        int(payload["literal_ratio_measure_visible"]),
    ]
    return MetaSynRescueSelectionFeaturesV3.model_validate(
        {**payload, "feature_sha256": hash_canonical(payload)}
    )


def _selection_disposition(
    *,
    request: PacketRequestV2,
    features: MetaSynRescueSelectionFeaturesV3,
    attempted_packet_inputs: set[str],
    attempted_bindings: set[str],
    attempted_signatures: set[str],
) -> SelectionDispositionV3:
    signature = _scientific_request_signature(request)
    if (
        request.packet_input_sha256 in attempted_packet_inputs
        or request.packet_input.candidate_binding_sha256 in attempted_bindings
        or signature in attempted_signatures
    ):
        return "excluded_previously_attempted"
    if not (
        features.release_grade_source_grounding_eligible
        and features.source_content_scope == "full_text_sections"
        and features.projection_selection_complete
        and features.source_strength_blocker_count == 0
    ):
        return "excluded_not_release_grade_full_text"
    if not features.direct_confidence_interval:
        return "excluded_not_direct_confidence_interval"
    if not (
        features.literal_confidence_interval_visible and features.literal_ratio_measure_visible
    ):
        return "excluded_literal_ci_or_ratio_not_visible"
    return "eligible_ranked"


def _freeze_candidate_audits(*, context: _V2ReplayContext) -> list[MetaSynRescueCandidateAuditV3]:
    attempted = context.snapshot.attempted_packet_requests
    attempted_packet_inputs = {item.packet_input_sha256 for item in attempted}
    attempted_bindings = {item.candidate_binding_sha256 for item in attempted}
    attempted_signatures = {item.scientific_request_signature_sha256 for item in attempted}
    provisional: list[
        tuple[
            PacketRequestV2,
            MetaSynRescueSelectionFeaturesV3,
            SelectionDispositionV3,
            str,
        ]
    ] = []
    for request in context.packet_roster.requests:
        features = _selection_features(request)
        signature = _scientific_request_signature(request)
        disposition = _selection_disposition(
            request=request,
            features=features,
            attempted_packet_inputs=attempted_packet_inputs,
            attempted_bindings=attempted_bindings,
            attempted_signatures=attempted_signatures,
        )
        provisional.append((request, features, disposition, signature))
    eligible = sorted(
        (item for item in provisional if item[2] == "eligible_ranked"),
        key=lambda item: (
            item[1].minimum_prompt_rank,
            item[0].row_ordinal,
            item[0].candidate_index,
            item[0].packet_input.candidate_binding_sha256,
        ),
    )
    eligible_rank = {
        request.packet_request_sha256: index
        for index, (request, _, _, _) in enumerate(eligible, start=1)
    }
    if len(eligible) < MAXIMUM_RESCUE_SMOKE_CALLS:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_too_few_eligible_candidates"
        )
    output: list[MetaSynRescueCandidateAuditV3] = []
    for request, features, disposition, signature in provisional:
        rank = eligible_rank.get(request.packet_request_sha256)
        payload: dict[str, Any] = {
            "row_ordinal": request.row_ordinal,
            "candidate_index": request.candidate_index,
            "row_key": request.row_key,
            "packet_request_sha256": request.packet_request_sha256,
            "packet_input_sha256": request.packet_input_sha256,
            "candidate_descriptor_sha256": (request.packet_input.candidate_descriptor_sha256),
            "candidate_binding_sha256": (request.packet_input.candidate_binding_sha256),
            "scientific_request_signature_sha256": signature,
            "features": features,
            "disposition": disposition,
            "eligible_rank": rank,
            "selected_for_smoke": rank in {1, 2, 3},
        }
        output.append(
            MetaSynRescueCandidateAuditV3.model_validate(
                {**payload, "audit_sha256": hash_canonical(payload)}
            )
        )
    return output


def _resolve_local_import(
    *, repository_root: Path, current_path: str, module: str, level: int
) -> str | None:
    if level:
        if not current_path.startswith("src/literature_multiverse/"):
            return None
        package_parts = PurePosixPath(current_path).with_suffix("").parts[:-1]
        if level > len(package_parts):
            return None
        prefix = list(package_parts[: len(package_parts) - level + 1])
        module_parts = module.split(".") if module else []
        candidate_parts = prefix + module_parts
    else:
        if not module.startswith("literature_multiverse"):
            return None
        candidate_parts = ["src", *module.split(".")]
    candidate = "/".join(candidate_parts)
    choices = [f"{candidate}.py", f"{candidate}/__init__.py"]
    existing = [value for value in choices if (repository_root / value).is_file()]
    if len(existing) > 1:
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_dependency_ambiguous")
    return existing[0] if existing else None


def _python_dependency_closure(repository_root: Path) -> list[str]:
    pending = list(_PYTHON_FINGERPRINT_SEEDS)
    observed: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        path = _checked_file(root=repository_root, relative_path=relative)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise MetaSynPassagePacketRescueV3Error(
                f"metasyn_packet_rescue_v3_dependency_unreadable:{relative}"
            ) from exc
        observed.add(relative)
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom):
                modules.append((node.module or "", node.level))
            elif isinstance(node, ast.Import):
                modules.extend((alias.name, 0) for alias in node.names)
            for module, level in modules:
                dependency = _resolve_local_import(
                    repository_root=repository_root,
                    current_path=relative,
                    module=module,
                    level=level,
                )
                if dependency is not None and dependency not in observed:
                    pending.append(dependency)
    return sorted(observed)


def compute_metasyn_passage_packet_rescue_pipeline_fingerprint_v3(
    *,
    repository_root: Path,
    config_sha256: str,
    v2_replay_snapshot_sha256: str,
    v2_forensic_receipt_sha256: str,
    pre_call_blocker_sha256: str,
) -> PipelineFingerprint:
    root = _canonical_repository_root(repository_root)
    for field_name, value in {
        "config_sha256": config_sha256,
        "v2_replay_snapshot_sha256": v2_replay_snapshot_sha256,
        "v2_forensic_receipt_sha256": v2_forensic_receipt_sha256,
        "pre_call_blocker_sha256": pre_call_blocker_sha256,
    }.items():
        _validate_sha256(value, field_name)
    files = sorted(set(_python_dependency_closure(root)) | set(_NON_PYTHON_FINGERPRINT_FILES))
    if not set(_PYTHON_FINGERPRINT_SEEDS).issubset(files):
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_entrypoints_not_fingerprinted"
        )
    component = PipelineComponentSpec(
        component_id="metasyn-passage-packet-rescue-v3",
        component_version=RESCUE_PIPELINE_COMPONENT_VERSION,
        file_paths=files,
        settings={
            "application_retries_per_request": 0,
            "compact_normalizable_invariant_fields": [
                "outcome_version",
                "packet_status",
            ],
            "config_sha256": config_sha256,
            "expected_v2_execution_bundle_sha256": (EXPECTED_V2_EXECUTION_BUNDLE_SHA256),
            "expected_v2_failed_smoke_sha256": EXPECTED_V2_FAILED_SMOKE_SHA256,
            "expected_v2_inventory_ledger_sha256": (EXPECTED_V2_INVENTORY_LEDGER_SHA256),
            "expected_v2_packet_roster_sha256": EXPECTED_V2_PACKET_ROSTER_SHA256,
            "installed_dependency_versions": {
                name: distribution_version(name) for name in _INSTALLED_DEPENDENCIES
            },
            "inventory_normalization_permitted": False,
            "pre_call_blocker_sha256": pre_call_blocker_sha256,
            "provider_calls_permitted": False,
            "authorization_created": False,
            "literal_ci_pattern": _CI_LITERAL_PATTERN,
            "literal_ratio_pattern": _RATIO_LITERAL_PATTERN,
            "maximum_rescue_smoke_calls": MAXIMUM_RESCUE_SMOKE_CALLS,
            "official_test_labels_opened": False,
            "post_hoc_exploratory": True,
            "provider_calls_permitted_by_fingerprint_freeze": False,
            "reference_fields_unopened": True,
            "sdk_retries_per_request": 0,
            "v2_forensic_receipt_sha256": v2_forensic_receipt_sha256,
            "v2_replay_snapshot_sha256": v2_replay_snapshot_sha256,
            "yield_only_no_accuracy_or_release_authority": True,
        },
    )
    return compute_pipeline_fingerprint(root=root, components=[component])


def _freeze_rescue_requests(
    *,
    config: MetaSynPassagePacketRescueConfigV3,
    context: _V2ReplayContext,
    audits: Sequence[MetaSynRescueCandidateAuditV3],
) -> list[MetaSynPassageRescueRequestV3]:
    audit_by_source_sha = {item.packet_request_sha256: item for item in audits}
    request_by_source_sha = {
        item.packet_request_sha256: item for item in context.packet_roster.requests
    }
    selected = sorted(
        (item for item in audits if item.selected_for_smoke),
        key=lambda item: item.eligible_rank or 0,
    )
    if [item.eligible_rank for item in selected] != [1, 2, 3]:
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_selected_rank_mismatch")
    prior_keys = {item.request.request_key for item in context.packet_roster.requests} | {
        item.request_key for item in context.snapshot.attempted_packet_requests
    }
    output: list[MetaSynPassageRescueRequestV3] = []
    for audit in selected:
        source = request_by_source_sha[audit.packet_request_sha256]
        if audit != audit_by_source_sha[source.packet_request_sha256]:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_selected_audit_alias_mismatch"
            )
        key = (
            f"{config.request_key_prefix}-row-{source.row_ordinal:02d}"
            f"-candidate-{source.candidate_index:02d}"
        )
        if key in prior_keys:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_request_key_not_fresh"
            )
        request = freeze_anthropic_bounded_request(
            operation=config.operation,
            request_key=key,
            prompt=source.packet_input.rendered_prompt,
            system=context.bundle.runtime_config.system_prompt,
            max_output_tokens=source.request.max_output_tokens,
            compiled_schema=source.compiled_schema,
            config=context.bundle.anthropic_config,
            schema_kind="packet",
            effect_kind=source.packet_input.candidate.effect_kind,
            identity=context.bundle.provider_identity,
        )
        payload: dict[str, Any] = {
            "rescue_request_version": ("metasyn-passage-packet-rescue-request-v3"),
            "eligible_rank": audit.eligible_rank,
            "row_ordinal": source.row_ordinal,
            "candidate_index": source.candidate_index,
            "source_packet_request": source,
            "source_packet_request_sha256": source.packet_request_sha256,
            "candidate_binding_sha256": source.packet_input.candidate_binding_sha256,
            "scientific_request_signature_sha256": (audit.scientific_request_signature_sha256),
            "selection_feature_sha256": audit.features.feature_sha256,
            "request": request,
            "request_sha256": request.request_sha256,
            "request_cost_ceiling_usd_micros": _usd_micros(
                request.cost_ceiling.request_cost_ceiling_usd
            ),
            "fresh_request_key": True,
            "fresh_intent_domain": True,
            "previously_attempted_scientific_request": False,
        }
        output.append(
            MetaSynPassageRescueRequestV3.model_validate(
                {
                    **payload,
                    "rescue_request_sha256": hash_canonical(payload),
                }
            )
        )
    return output


def _pre_call_completed_probe(
    request: MetaSynPassageRescueRequestV3,
) -> tuple[dict[str, Any], str, str, str]:
    key = (request.row_ordinal, request.candidate_index)
    passage_surface = request.source_packet_request.packet_input.candidate_passage_surface
    if len(passage_surface.passages) != 1:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_blocker_candidate_passage_cardinality"
        )
    quote = passage_surface.passages[0].text
    if key == (16, 1):
        lower, upper, format_token = "18.0", "1005", "odds ratio"
        identities = {
            "cohort.source_label": "Thirty-six patients",
            "comparator_arm.label": "placebo-treated patients",
            "contrast.label": (
                "ruxolitinib-treated patients achieved a ≥35% reduction in spleen volume "
                "at week 24 compared with 0.7% of placebo-treated patients"
            ),
            "study.source_label": "EFFICACY",
            "treatment_arm.label": "ruxolitinib-treated patients",
        }
        timepoint: dict[str, Any] = {
            "kind": "reported_text",
            "raw_label": "at week 24 compared with",
            "anchor": None,
        }
        estimate = "134.4"
    elif key == (20, 1):
        lower, upper, format_token = "0.62", "0.92", "rate ratio"
        identities = {
            "cohort.source_label": "patients with colorectal cancer",
            "comparator_arm.label": "non-users",
            "contrast.label": "users as compared to non-users",
            "study.source_label": "present observational population-based study",
            "treatment_arm.label": "aspirin use after diagnosis",
        }
        timepoint = {
            "kind": "reported_text",
            "raw_label": "after diagnosis",
            "anchor": None,
        }
        estimate = "0.75"
    elif key == (20, 2):
        lower, upper, format_token = "0.65", "1.34", "RR"
        identities = {
            "cohort.source_label": "patients with rectal cancer",
            "comparator_arm.label": "non-users",
            "contrast.label": "users as compared to non-users",
            "study.source_label": "present observational population-based study",
            "treatment_arm.label": "frequent users of aspirin",
        }
        timepoint = {"kind": "not_reported"}
        estimate = "0.94"
    else:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_blocker_selected_roster_changed"
        )
    range_expression = f"{lower}{_EN_DASH}{upper}"
    if range_expression not in quote:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_blocker_range_expression_absent"
        )
    numeric = {
        "effect.ci_level": ("95%", "percent_to_proportion"),
        "effect.ci_lower": (lower, "identity"),
        "effect.ci_upper": (upper, "identity"),
        "effect.estimate": (estimate, "identity"),
    }
    raw: dict[str, Any] = {
        "candidate_binding_sha256": request.candidate_binding_sha256,
        "evidence_quote": quote,
        "effect_format_token": format_token,
        "effect_unit": None,
        "numeric_claims": [
            {
                "field_path": field_path,
                "verbatim_numeric_token": token,
                "normalization": normalization,
            }
            for field_path, (token, normalization) in sorted(numeric.items())
        ],
        "identity_claims": [
            {"field_path": field_path, "verbatim_identity_text": value}
            for field_path, value in sorted(identities.items())
        ],
        "timepoint": timepoint,
    }
    return raw, lower, upper, format_token


def _freeze_pre_call_blocker(
    *,
    context: _V2ReplayContext,
    requests: Sequence[MetaSynPassageRescueRequestV3],
) -> MetaSynPassageRescuePreCallBlockerV3:
    if [(item.row_ordinal, item.candidate_index) for item in requests] != [
        (16, 1),
        (20, 1),
        (20, 2),
    ]:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_blocker_selected_roster_changed"
        )
    items: list[MetaSynPassageRescuePreCallBlockerItemV3] = []
    for request in requests:
        raw, lower, upper, format_token = _pre_call_completed_probe(request)
        quote = raw["evidence_quote"]
        if not isinstance(quote, str):  # pragma: no cover - constructed above
            raise AssertionError("constructed quote must be text")
        upper_occurrences = native_packet_grounding_v2._numeric_token_occurrences(
            quote=quote,
            token=upper,
            normalization="identity",
            field_path="effect.ci_upper",
        )
        if upper_occurrences:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_blocker_upper_token_unexpectedly_reachable"
            )
        try:
            native_packet_grounding_v2._effect_format_from_exact_token(format_token)
        except ValueError:
            format_supported = False
        else:
            format_supported = True
        normalization: NativePacketCompactNormalizationReceiptV3 | None
        row = context.bundle.extraction_inputs.rows[request.row_ordinal]
        if format_supported:
            normalization = freeze_native_packet_compact_normalization_receipt_v3(
                raw_model_outcome=raw,
                expected_candidate_binding_sha256=request.candidate_binding_sha256,
            )
            try:
                freeze_passage_packet_grounding_receipt_v2(
                    model_outcome=normalization.normalized_model_outcome,
                    candidate=request.source_packet_request.packet_input.candidate,
                    projection=row.projection_v2,
                )
            except ValueError as exc:
                first_failure = str(exc)
            else:
                raise MetaSynPassagePacketRescueV3Error(
                    "metasyn_packet_rescue_v3_blocker_probe_unexpectedly_grounded"
                )
        else:
            normalization = None
            try:
                freeze_native_packet_compact_normalization_receipt_v3(
                    raw_model_outcome=raw,
                    expected_candidate_binding_sha256=request.candidate_binding_sha256,
                )
            except ValueError as exc:
                exception_text = _exception_chain_text(exc)
                if "packet_grounding_v2_effect_format_alias_unsupported" not in exception_text:
                    raise MetaSynPassagePacketRescueV3Error(
                        "metasyn_packet_rescue_v3_blocker_format_failure_changed:" + exception_text
                    ) from exc
                first_failure = "packet_grounding_v2_effect_format_alias_unsupported"
            else:
                raise MetaSynPassagePacketRescueV3Error(
                    "metasyn_packet_rescue_v3_blocker_format_probe_unexpectedly_valid"
                )
        static_blockers = ["numeric_upper_token_rejected_after_en_dash"]
        if not format_supported:
            static_blockers.append("exact_effect_format_alias_unsupported_by_v2")
        static_blockers.sort()
        expected_failure = (
            "packet_grounding_v2_numeric_token_absent:effect.ci_upper"
            if format_supported
            else "packet_grounding_v2_effect_format_alias_unsupported"
        )
        if first_failure != expected_failure:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_blocker_first_failure_changed:" + first_failure
            )
        code_path = (
            "native_packet_grounding_v2.freeze_passage_packet_grounding_receipt_v2"
            "->_freeze_numeric_receipts->_numeric_token_occurrences"
            "->_assert_numeric_token_boundary"
            if format_supported
            else "native_packet_grounding_v2.PacketGroundingModelCompletedV2"
            ".effect_format_token->_effect_format_from_exact_token"
        )
        item_payload: dict[str, Any] = {
            "blocker_item_version": ("metasyn-passage-packet-rescue-pre-call-blocker-item-v3"),
            "selected_rank": request.eligible_rank,
            "row_ordinal": request.row_ordinal,
            "candidate_index": request.candidate_index,
            "rescue_request_sha256": request.rescue_request_sha256,
            "source_packet_request_sha256": request.source_packet_request_sha256,
            "candidate_binding_sha256": request.candidate_binding_sha256,
            "candidate_passage_ids": (
                request.source_packet_request.packet_input.candidate.passage_ids
            ),
            "evidence_quote": quote,
            "evidence_quote_sha256": hash_canonical(quote),
            "ci_lower_token": lower,
            "ci_upper_token": upper,
            "exact_range_separator": _EN_DASH,
            "exact_range_separator_code_point": "U+2013",
            "exact_range_expression": f"{lower}{_EN_DASH}{upper}",
            "effect_format_token": format_token,
            "exact_effect_format_alias_supported_by_v2": format_supported,
            "constructed_source_copied_probe": raw,
            "constructed_source_copied_probe_sha256": hash_canonical(raw),
            "full_v2_model_contract_valid": format_supported,
            "compact_normalization_receipt": normalization,
            "compact_normalization_receipt_sha256": (
                normalization.receipt_sha256 if normalization is not None else None
            ),
            "immutable_v2_first_failure_code": first_failure,
            "immutable_v2_first_failure_code_path": code_path,
            "upper_token_v2_valid_occurrence_count": len(upper_occurrences),
            "static_blocker_codes": static_blockers,
            "fails_only_at_immutable_numeric_boundary_gate": format_supported,
            "numerical_effect_values_copied_exactly_from_source": True,
            "provider_call_required_to_establish_blocker": False,
            "synthesis_input_authority": False,
            "claim_release_authority": False,
        }
        items.append(
            MetaSynPassageRescuePreCallBlockerItemV3.model_validate(
                {
                    **item_payload,
                    "blocker_item_sha256": hash_canonical(item_payload),
                }
            )
        )
    request_membership = hash_canonical([item.rescue_request_sha256 for item in requests])
    payload: dict[str, Any] = {
        "blocker_version": RESCUE_PRE_CALL_BLOCKER_VERSION,
        "status": "selected_v2_candidates_proven_unreachable_before_provider_liability",
        "selected_request_membership_sha256": request_membership,
        "items": items,
        "item_membership_sha256": hash_canonical([item.blocker_item_sha256 for item in items]),
        "selected_candidate_count": len(items),
        "selected_candidate_v2_reachable_count": 0,
        "numeric_boundary_blocked_candidate_count": len(items),
        "unsupported_exact_effect_format_candidate_count": sum(
            not item.exact_effect_format_alias_supported_by_v2 for item in items
        ),
        "row16_completed_contract_probe_fails_only_at_numeric_boundary": True,
        "provider_calls_made": 0,
        "authorization_created": False,
        "calls_permitted": False,
        "provider_cost_liability_usd_micros": 0,
        "additive_successor_required": True,
        "v2_artifacts_or_semantics_changed": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynPassageRescuePreCallBlockerV3.model_validate(
        {**payload, "blocker_sha256": hash_canonical(payload)}
    )


def freeze_metasyn_passage_packet_rescue_plan_v3(
    *,
    repository_root: Path,
    v2_workspace: Path = DEFAULT_V2_WORKSPACE,
) -> MetaSynPassagePacketRescuePlanV3:
    """Externally replay v2 and freeze a credential-free additive rescue plan."""

    root = _canonical_repository_root(repository_root)
    config, config_file_sha256 = load_metasyn_passage_packet_rescue_config_v3(repository_root=root)
    context = _replay_v2_base(repository_root=root, v2_workspace=v2_workspace)
    forensic = _freeze_v2_forensic_receipt(context=context)
    audits = _freeze_candidate_audits(context=context)
    requests = _freeze_rescue_requests(config=config, context=context, audits=audits)
    blocker = _freeze_pre_call_blocker(context=context, requests=requests)
    cost_ceiling = sum(item.request_cost_ceiling_usd_micros for item in requests)
    if cost_ceiling > config.configured_cost_cap_usd_micros:
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_plan_cost_cap_exceeded")
    fingerprint = compute_metasyn_passage_packet_rescue_pipeline_fingerprint_v3(
        repository_root=root,
        config_sha256=config.config_sha256,
        v2_replay_snapshot_sha256=context.snapshot.snapshot_sha256,
        v2_forensic_receipt_sha256=forensic.forensic_receipt_sha256,
        pre_call_blocker_sha256=blocker.blocker_sha256,
    )
    eligible = sorted(
        (item for item in audits if item.eligible_rank is not None),
        key=lambda item: item.eligible_rank or 0,
    )
    snapshot = context.snapshot
    payload: dict[str, Any] = {
        "plan_version": RESCUE_PLAN_VERSION,
        "status": ("frozen_post_hoc_label_blind_unattempted_candidate_plan_pre_call_blocked"),
        "config": config,
        "config_sha256": config.config_sha256,
        "config_file_sha256": config_file_sha256,
        "v2_replay_snapshot": snapshot,
        "v2_replay_snapshot_sha256": snapshot.snapshot_sha256,
        "v2_forensic_receipt": forensic,
        "v2_forensic_receipt_sha256": forensic.forensic_receipt_sha256,
        "pipeline_fingerprint": fingerprint,
        "pipeline_sha256": fingerprint.pipeline_sha256,
        "candidate_audits": audits,
        "candidate_audit_membership_sha256": hash_canonical([item.audit_sha256 for item in audits]),
        "eligible_candidate_audit_sha256s": [item.audit_sha256 for item in eligible],
        "eligible_candidate_membership_sha256": hash_canonical(
            [item.audit_sha256 for item in eligible]
        ),
        "requests": requests,
        "request_membership_sha256": hash_canonical(
            [item.rescue_request_sha256 for item in requests]
        ),
        "request_count": len(requests),
        "pre_call_blocker": blocker,
        "pre_call_blocker_sha256": blocker.blocker_sha256,
        "provider_calls_permitted": False,
        "authorization_created": False,
        "conservative_cost_ceiling_usd_micros": cost_ceiling,
        "configured_cost_cap_usd_micros": config.configured_cost_cap_usd_micros,
        "attempted_v2_request_membership_sha256": (
            snapshot.attempted_packet_request_membership_sha256
        ),
        "attempted_v2_intent_membership_sha256": (
            snapshot.attempted_packet_intent_membership_sha256
        ),
        "attempted_v2_provider_result_membership_sha256": (
            snapshot.attempted_packet_provider_result_membership_sha256
        ),
        "attempted_v2_scientific_request_membership_sha256": (
            snapshot.attempted_packet_scientific_request_membership_sha256
        ),
        "all_requests_unattempted_under_scientific_signature": True,
        "provider_calls_made": False,
        "post_hoc_exploratory": True,
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "extraction_accuracy_authority": False,
        "scientific_effectiveness_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynPassagePacketRescuePlanV3.model_validate(
        {**payload, "plan_sha256": hash_canonical(payload)}
    )


def validate_metasyn_passage_packet_rescue_plan_v3(
    *,
    plan: MetaSynPassagePacketRescuePlanV3 | Mapping[str, Any],
    repository_root: Path,
    v2_workspace: Path = DEFAULT_V2_WORKSPACE,
    external_replay: bool = True,
) -> MetaSynPassagePacketRescuePlanV3:
    try:
        canonical = MetaSynPassagePacketRescuePlanV3.model_validate(
            plan.model_dump(mode="json")
            if isinstance(plan, MetaSynPassagePacketRescuePlanV3)
            else plan
        )
    except ValueError as exc:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_plan_contract_invalid"
        ) from exc
    if external_replay:
        replayed = freeze_metasyn_passage_packet_rescue_plan_v3(
            repository_root=repository_root, v2_workspace=v2_workspace
        )
        if replayed != canonical:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_plan_external_replay_mismatch"
            )
    return canonical


RescueRuntimeStageV3 = Literal[
    "prepared",
    "authorized",
    "smoke_passed",
    "smoke_failed",
    "finalized",
    "externally_validated",
]
_STAGE_ORDINAL: dict[RescueRuntimeStageV3, int] = {
    "prepared": 0,
    "authorized": 1,
    "smoke_passed": 2,
    "smoke_failed": 2,
    "finalized": 3,
    "externally_validated": 4,
}
_STAGE_FILENAMES: dict[int, tuple[str, ...]] = {
    0: ("00-prepared.json",),
    1: ("01-authorized.json",),
    2: ("02-smoke-passed.json", "02-smoke-failed.json"),
    3: ("03-finalized.json",),
    4: ("04-externally-validated.json",),
}


class MetaSynPassageRescueStageCheckpointV3(_FrozenExactModel):
    checkpoint_version: Literal["metasyn-passage-packet-rescue-checkpoint-v3"] = (
        RESCUE_CHECKPOINT_VERSION
    )
    plan_sha256: str
    stage: RescueRuntimeStageV3
    stage_ordinal: Annotated[int, Field(ge=0, le=4)]
    previous_checkpoint_sha256: str | None
    artifacts: Annotated[list[RescueArtifactBindingV3], Field(min_length=1)]
    artifact_membership_sha256: str
    checkpoint_sha256: str

    @field_validator("plan_sha256", "artifact_membership_sha256", "checkpoint_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("previous_checkpoint_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return None if value is None else _validate_sha256(value, "previous_checkpoint_sha256")

    @model_validator(mode="after")
    def validate_checkpoint(self) -> MetaSynPassageRescueStageCheckpointV3:
        if self.stage_ordinal != _STAGE_ORDINAL[self.stage]:
            raise ValueError("metasyn_packet_rescue_v3_checkpoint_ordinal_mismatch")
        if self.artifacts != sorted(self.artifacts, key=lambda item: item.relative_path):
            raise ValueError("metasyn_packet_rescue_v3_checkpoint_artifacts_not_canonical")
        if self.artifact_membership_sha256 != hash_canonical(
            [item.model_dump(mode="json") for item in self.artifacts]
        ):
            raise ValueError("metasyn_packet_rescue_v3_checkpoint_membership_mismatch")
        payload = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        if self.checkpoint_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_packet_rescue_v3_checkpoint_hash_mismatch")
        return self


class MetaSynPassageRescueAuthorizationV3(_FrozenExactModel):
    authorization_version: Literal["metasyn-passage-packet-rescue-authorization-v3"] = (
        "metasyn-passage-packet-rescue-authorization-v3"
    )
    status: Literal["exact_three_call_cost_authorization_persisted_before_liability"]
    plan_sha256: str
    request_membership_sha256: str
    exact_authorization: HostedExactOnceCostAuthorizationV1
    exact_authorization_sha256: str
    authorized_call_count: Literal[3]
    conservative_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    configured_cost_cap_usd_micros: Literal[2200000]
    provider_calls_made_before_authorization: Literal[0] = 0
    retries_per_request: Literal[0] = 0
    authorization_receipt_sha256: str

    @field_validator(
        "plan_sha256",
        "request_membership_sha256",
        "exact_authorization_sha256",
        "authorization_receipt_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_authorization(self) -> MetaSynPassageRescueAuthorizationV3:
        if (
            self.exact_authorization_sha256 != self.exact_authorization.authorization_sha256
            or self.exact_authorization.execution_bundle_sha256 != self.plan_sha256
            or self.exact_authorization.phase != "smoke_packet"
            or self.authorized_call_count != self.exact_authorization.authorized_call_count
            or self.conservative_cost_ceiling_usd_micros
            != self.exact_authorization.cost_ceiling_usd_micros
            or self.conservative_cost_ceiling_usd_micros > self.configured_cost_cap_usd_micros
        ):
            raise ValueError("metasyn_packet_rescue_v3_authorization_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"authorization_receipt_sha256"})
        if self.authorization_receipt_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_packet_rescue_v3_authorization_hash_mismatch")
        return self


RescueValidationStatusV3 = Literal[
    "typed_effect_completed",
    "grounding_abstained",
    "assembly_abstained",
    "grounding_invalid",
    "assembly_invalid",
    "provider_runtime_failure",
    "exact_once_terminal_incident",
]
RescueTerminalOutcomeV3 = HostedExactOnceProviderReceiptV1 | HostedExactOnceAmbiguityIncidentV1
_RESCUE_TERMINAL_ADAPTER = TypeAdapter(RescueTerminalOutcomeV3)
_GROUNDING_ADAPTER = TypeAdapter(PacketGroundingReceiptV2)
_ASSEMBLY_ADAPTER = TypeAdapter(NativePacketAssemblyOutcomeV2)


class MetaSynPassageRescueResultV3(_FrozenExactModel):
    result_version: Literal["metasyn-passage-packet-rescue-result-v3"] = RESCUE_RESULT_VERSION
    plan_sha256: str
    rescue_request_sha256: str
    request_key: str
    row_ordinal: Annotated[int, Field(ge=0, lt=32)]
    candidate_index: Annotated[int, Field(ge=1, le=8)]
    packet_input: MetaSynPacketCandidateInputV2
    packet_input_sha256: str
    terminal: RescueTerminalOutcomeV3
    terminal_sha256: str
    validation_status: RescueValidationStatusV3
    compact_normalization_receipt: NativePacketCompactNormalizationReceiptV3 | None
    compact_normalization_receipt_sha256: str | None
    grounding_receipt: PacketGroundingReceiptV2 | None
    grounding_receipt_sha256: str | None
    assembly_receipt: NativePacketAssemblyOutcomeV2 | None
    assembly_receipt_sha256: str | None
    authorizes_typed_effect: bool
    standard_packet_input_grounding_assembly_triple_persisted: bool
    bridge_v2_single_terminal_shape_compatible: bool
    complete_v2_authorized_candidate_terminal_roster: Literal[False] = False
    bridge_v2_full_corpus_input_ready: Literal[False] = False
    runtime_failure_is_not_scientific_abstention: Literal[True] = True
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    result_sha256: str

    @field_validator(
        "plan_sha256",
        "rescue_request_sha256",
        "packet_input_sha256",
        "terminal_sha256",
        "result_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator(
        "compact_normalization_receipt_sha256",
        "grounding_receipt_sha256",
        "assembly_receipt_sha256",
    )
    @classmethod
    def validate_optional_hashes(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_result(self) -> MetaSynPassageRescueResultV3:
        terminal_hash = (
            self.terminal.receipt_sha256
            if isinstance(self.terminal, HostedExactOnceProviderReceiptV1)
            else self.terminal.incident_sha256
        )
        if (
            self.packet_input_sha256 != self.packet_input.packet_input_sha256
            or self.packet_input.row_ordinal != self.row_ordinal
            or self.packet_input.candidate.candidate_index != self.candidate_index
            or self.terminal_sha256 != terminal_hash
            or self.compact_normalization_receipt_sha256
            != (
                self.compact_normalization_receipt.receipt_sha256
                if self.compact_normalization_receipt
                else None
            )
            or self.grounding_receipt_sha256
            != (self.grounding_receipt.receipt_sha256 if self.grounding_receipt else None)
            or self.assembly_receipt_sha256
            != (self.assembly_receipt.assembly_receipt_sha256 if self.assembly_receipt else None)
        ):
            raise ValueError("metasyn_packet_rescue_v3_result_alias_mismatch")
        if (
            self.terminal.execution_bundle_sha256 != self.plan_sha256
            or self.terminal.request_key != self.request_key
            or self.terminal.context_binding_sha256 != self.rescue_request_sha256
        ):
            raise ValueError("metasyn_packet_rescue_v3_terminal_context_mismatch")
        is_incident = isinstance(self.terminal, HostedExactOnceAmbiguityIncidentV1)
        provider_completed = (
            isinstance(self.terminal, HostedExactOnceProviderReceiptV1)
            and self.terminal.provider_result.outcome == "completed"
        )
        if self.compact_normalization_receipt is not None:
            if (
                not provider_completed
                or not isinstance(self.terminal, HostedExactOnceProviderReceiptV1)
                or not isinstance(self.terminal.provider_result.parsed_json, Mapping)
            ):
                raise ValueError("metasyn_packet_rescue_v3_normalization_terminal_mismatch")
            validate_native_packet_compact_normalization_receipt_v3(
                receipt=self.compact_normalization_receipt,
                raw_model_outcome=self.terminal.provider_result.parsed_json,
                expected_candidate_binding_sha256=self.packet_input.candidate_binding_sha256,
            )
        grounding_completed = isinstance(self.grounding_receipt, PacketGroundingCompletedReceiptV2)
        grounding_abstained = isinstance(self.grounding_receipt, PacketGroundingAbstentionReceiptV2)
        assembly_completed = isinstance(self.assembly_receipt, NativePacketAssemblyCompletedV2)
        assembly_abstained = isinstance(self.assembly_receipt, NativePacketAssemblyAbstentionV2)
        if grounding_completed and (
            self.compact_normalization_receipt is None
            or self.compact_normalization_receipt.branch != "completed"
        ):
            raise ValueError("metasyn_packet_rescue_v3_completed_grounding_branch_mismatch")
        if grounding_abstained and (
            self.compact_normalization_receipt is None
            or self.compact_normalization_receipt.branch != "unable_to_complete"
        ):
            raise ValueError("metasyn_packet_rescue_v3_abstention_grounding_branch_mismatch")
        if assembly_completed and not grounding_completed:
            raise ValueError("metasyn_packet_rescue_v3_completed_assembly_grounding_mismatch")
        if assembly_abstained and not (grounding_completed or grounding_abstained):
            raise ValueError("metasyn_packet_rescue_v3_abstention_assembly_grounding_mismatch")
        expected_status: RescueValidationStatusV3
        if is_incident:
            if any(
                item is not None
                for item in (
                    self.compact_normalization_receipt,
                    self.grounding_receipt,
                    self.assembly_receipt,
                )
            ):
                raise ValueError("metasyn_packet_rescue_v3_incident_scientific_receipt_present")
            expected_status = "exact_once_terminal_incident"
        elif not provider_completed:
            if any(
                item is not None
                for item in (
                    self.compact_normalization_receipt,
                    self.grounding_receipt,
                    self.assembly_receipt,
                )
            ):
                raise ValueError("metasyn_packet_rescue_v3_runtime_failure_receipt_present")
            expected_status = "provider_runtime_failure"
        elif assembly_completed:
            expected_status = "typed_effect_completed"
        elif assembly_abstained and grounding_abstained:
            expected_status = "grounding_abstained"
        elif assembly_abstained and grounding_completed:
            expected_status = "assembly_abstained"
        elif self.grounding_receipt is not None and self.assembly_receipt is None:
            expected_status = "assembly_invalid"
        elif self.grounding_receipt is None and self.assembly_receipt is None:
            expected_status = "grounding_invalid"
        else:
            raise ValueError("metasyn_packet_rescue_v3_result_receipt_shape_invalid")
        if self.validation_status != expected_status:
            raise ValueError("metasyn_packet_rescue_v3_validation_status_shape_mismatch")
        if self.authorizes_typed_effect != assembly_completed:
            raise ValueError("metasyn_packet_rescue_v3_typed_authority_mismatch")
        triple = self.grounding_receipt is not None and self.assembly_receipt is not None
        if (
            self.standard_packet_input_grounding_assembly_triple_persisted != triple
            or self.bridge_v2_single_terminal_shape_compatible != triple
        ):
            raise ValueError("metasyn_packet_rescue_v3_terminal_triple_mismatch")
        payload = self.model_dump(mode="json", exclude={"result_sha256"})
        if self.result_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_packet_rescue_v3_result_hash_mismatch")
        return self


class MetaSynPassageRescueSmokeReceiptV3(_FrozenExactModel):
    smoke_version: Literal["metasyn-passage-packet-rescue-smoke-v3"] = RESCUE_SMOKE_VERSION
    status: Literal["passed", "failed_gate"]
    plan_sha256: str
    authorization_receipt_sha256: str
    ordered_authorized_request_keys: Annotated[list[str], Field(min_length=3, max_length=3)]
    attempted_request_keys: Annotated[list[str], Field(min_length=1, max_length=3)]
    results: Annotated[list[MetaSynPassageRescueResultV3], Field(min_length=1, max_length=3)]
    result_membership_sha256: str
    completed_typed_effect_result_sha256: str | None
    typed_effect_count: Annotated[int, Field(ge=0, le=1)]
    valid_abstention_does_not_pass: Literal[True] = True
    compact_normalization_only_absent_invariants: Literal[True] = True
    retries_per_request: Literal[0] = 0
    remaining_calls_under_this_smoke_authorization_permitted: Literal[False] = False
    future_additive_full_roster_extension_possible: Literal[True] = True
    complete_v2_authorized_candidate_terminal_roster: Literal[False] = False
    bridge_v2_full_corpus_input_ready: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    smoke_sha256: str

    @field_validator(
        "plan_sha256",
        "authorization_receipt_sha256",
        "result_membership_sha256",
        "smoke_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("completed_typed_effect_result_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else _validate_sha256(value, "completed_typed_effect_result_sha256")
        )

    @model_validator(mode="after")
    def validate_smoke(self) -> MetaSynPassageRescueSmokeReceiptV3:
        if self.attempted_request_keys != [item.request_key for item in self.results]:
            raise ValueError("metasyn_packet_rescue_v3_smoke_result_order_mismatch")
        if (
            self.attempted_request_keys
            != self.ordered_authorized_request_keys[: len(self.attempted_request_keys)]
        ):
            raise ValueError("metasyn_packet_rescue_v3_smoke_not_authorized_prefix")
        if len(set(self.attempted_request_keys)) != len(self.attempted_request_keys):
            raise ValueError("metasyn_packet_rescue_v3_smoke_duplicate_terminal")
        if any(item.plan_sha256 != self.plan_sha256 for item in self.results):
            raise ValueError("metasyn_packet_rescue_v3_smoke_result_plan_mismatch")
        incident_indexes = [
            index
            for index, item in enumerate(self.results)
            if isinstance(item.terminal, HostedExactOnceAmbiguityIncidentV1)
        ]
        if incident_indexes and incident_indexes != [len(self.results) - 1]:
            raise ValueError("metasyn_packet_rescue_v3_smoke_incident_not_terminal")
        if self.result_membership_sha256 != hash_canonical(
            [item.result_sha256 for item in self.results]
        ):
            raise ValueError("metasyn_packet_rescue_v3_smoke_membership_mismatch")
        typed = [item for item in self.results if item.authorizes_typed_effect]
        if self.typed_effect_count != len(typed):
            raise ValueError("metasyn_packet_rescue_v3_smoke_typed_count_mismatch")
        if self.status == "passed":
            if (
                len(typed) != 1
                or self.completed_typed_effect_result_sha256 != typed[0].result_sha256
                or not self.results[-1].authorizes_typed_effect
            ):
                raise ValueError("metasyn_packet_rescue_v3_smoke_pass_shape_invalid")
        elif typed or self.completed_typed_effect_result_sha256 is not None:
            raise ValueError("metasyn_packet_rescue_v3_smoke_failure_shape_invalid")
        payload = self.model_dump(mode="json", exclude={"smoke_sha256"})
        if self.smoke_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_packet_rescue_v3_smoke_hash_mismatch")
        return self


class MetaSynPassageRescueFinalReportV3(_FrozenExactModel):
    report_version: Literal["metasyn-passage-packet-rescue-report-v3"] = RESCUE_REPORT_VERSION
    status: Literal["complete_post_hoc_exploratory_yield_report_no_accuracy_or_release_authority"]
    plan_sha256: str
    pipeline_sha256: str
    v2_replay_snapshot_sha256: str
    v2_forensic_receipt_sha256: str
    authorization_receipt_sha256: str
    smoke_sha256: str
    smoke_status: Literal["passed", "failed_gate"]
    attempted_call_count: Annotated[int, Field(ge=1, le=3)]
    terminal_result_membership_sha256: str
    typed_effect_count: Annotated[int, Field(ge=0, le=1)]
    grounding_abstention_count: Annotated[int, Field(ge=0, le=3)]
    conservative_attempt_liability_usd_micros: Annotated[int, Field(ge=1)]
    configured_cost_cap_usd_micros: Literal[2200000]
    standard_completed_terminal_triples_persisted: bool
    complete_v2_authorized_candidate_terminal_roster: Literal[False] = False
    bridge_v2_full_corpus_input_ready: Literal[False] = False
    future_additive_full_roster_extension_possible: Literal[True] = True
    post_hoc_exploratory: Literal[True] = True
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    report_sha256: str

    @field_validator(
        "plan_sha256",
        "pipeline_sha256",
        "v2_replay_snapshot_sha256",
        "v2_forensic_receipt_sha256",
        "authorization_receipt_sha256",
        "smoke_sha256",
        "terminal_result_membership_sha256",
        "report_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_report(self) -> MetaSynPassageRescueFinalReportV3:
        if self.conservative_attempt_liability_usd_micros > self.configured_cost_cap_usd_micros:
            raise ValueError("metasyn_packet_rescue_v3_report_cost_cap_exceeded")
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        if self.report_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_packet_rescue_v3_report_hash_mismatch")
        return self


class MetaSynPassageRescueExternalValidationV3(_FrozenExactModel):
    validation_version: Literal["metasyn-passage-packet-rescue-validation-v3"] = (
        RESCUE_VALIDATION_VERSION
    )
    status: Literal["v2_base_plan_authorization_terminals_and_report_externally_replayed"]
    plan_sha256: str
    report_sha256: str
    exact_terminal_count: Annotated[int, Field(ge=1, le=3)]
    completed_typed_effect_count: Annotated[int, Field(ge=0, le=1)]
    provider_calls_made_by_validation: Literal[0] = 0
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    claim_release_authority: Literal[False] = False
    validation_sha256: str

    @field_validator("plan_sha256", "report_sha256", "validation_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_receipt(self) -> MetaSynPassageRescueExternalValidationV3:
        payload = self.model_dump(mode="json", exclude={"validation_sha256"})
        if self.validation_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_packet_rescue_v3_validation_hash_mismatch")
        return self


def _runtime_paths(workspace: Path) -> dict[str, Path]:
    return {
        "plan": workspace / "rescue-plan.json",
        "forensic": workspace / "v2-compact-forensic-receipt.json",
        "blocker": workspace / "pre-call-zero-yield-blocker.json",
        "authorization": workspace / "rescue-authorization.json",
        "provider": workspace / "provider-state",
        "results": workspace / "results",
        "grounding": workspace / "grounding-receipts",
        "assembly": workspace / "assembly-receipts",
        "smoke": workspace / "rescue-smoke.json",
        "report": workspace / "final-report.json",
        "validation": workspace / "external-validation.json",
        "stages": workspace / "stage-checkpoints",
    }


def _create_fresh_workspace(value: Path) -> Path:
    path = Path(os.path.abspath(value))
    if path.exists() or path.is_symlink():
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_workspace_not_fresh")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_workspace_parent_invalid")
    path.mkdir(mode=0o700)
    return path.resolve(strict=True)


@contextmanager
def _runtime_lock(workspace: Path) -> Iterator[None]:
    lock_path = workspace / ".metasyn-passage-packet-rescue-v3.lock"
    if lock_path.is_symlink():
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_lock_symlink")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_or_replay(path: Path, value: ContractModel) -> None:
    if path.exists():
        saved = _read_object(path)
        if saved != value.model_dump(mode="json"):
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_artifact_replay_mismatch"
            )
        return
    atomic_write_json(path, value)


def _workspace_binding(workspace: Path, path: Path) -> RescueArtifactBindingV3:
    if path.is_symlink() or not path.is_file():
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_stage_artifact_unsafe")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(workspace):
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_stage_artifact_outside_workspace"
        )
    raw = path.read_bytes()
    return RescueArtifactBindingV3(
        relative_path=resolved.relative_to(workspace).as_posix(),
        sha256=sha256_file(path),
        utf8_bytes=len(raw),
    )


def _checkpoint_path(workspace: Path, stage: RescueRuntimeStageV3) -> Path:
    ordinal = _STAGE_ORDINAL[stage]
    filename = next(
        name
        for name in _STAGE_FILENAMES[ordinal]
        if stage.replace("_", "-") in name or ordinal not in {2}
    )
    return _runtime_paths(workspace)["stages"] / filename


def _load_stage_chain(workspace: Path) -> list[MetaSynPassageRescueStageCheckpointV3]:
    directory = _runtime_paths(workspace)["stages"]
    if directory.is_symlink() or not directory.is_dir():
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_stage_directory_unsafe")
    actual = {path.name for path in directory.glob("*.json")}
    chain: list[MetaSynPassageRescueStageCheckpointV3] = []
    for ordinal in range(5):
        matches = actual.intersection(_STAGE_FILENAMES[ordinal])
        if not matches:
            if any(actual.intersection(_STAGE_FILENAMES[later]) for later in range(ordinal + 1, 5)):
                raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_stage_gap")
            break
        if len(matches) != 1:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_stage_branch_ambiguous"
            )
        path = directory / next(iter(matches))
        checkpoint = MetaSynPassageRescueStageCheckpointV3.model_validate(_read_object(path))
        expected_previous = chain[-1].checkpoint_sha256 if chain else None
        if (
            checkpoint.stage_ordinal != ordinal
            or checkpoint.previous_checkpoint_sha256 != expected_previous
        ):
            raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_stage_chain_mismatch")
        for binding in checkpoint.artifacts:
            artifact_path = workspace / binding.relative_path
            if _workspace_binding(workspace, artifact_path) != binding:
                raise MetaSynPassagePacketRescueV3Error(
                    "metasyn_packet_rescue_v3_stage_artifact_tamper"
                )
        chain.append(checkpoint)
    allowed = {name for ordinal in range(len(chain)) for name in _STAGE_FILENAMES[ordinal]}
    if actual - allowed:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_unexpected_stage_artifact"
        )
    return chain


def _write_checkpoint(
    *,
    workspace: Path,
    plan_sha256: str,
    stage: RescueRuntimeStageV3,
    artifact_paths: Sequence[Path],
) -> MetaSynPassageRescueStageCheckpointV3:
    chain = _load_stage_chain(workspace)
    ordinal = _STAGE_ORDINAL[stage]
    if len(chain) != ordinal:
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_stage_advance_invalid")
    artifacts = sorted(
        (_workspace_binding(workspace, path) for path in artifact_paths),
        key=lambda item: item.relative_path,
    )
    payload: dict[str, Any] = {
        "checkpoint_version": RESCUE_CHECKPOINT_VERSION,
        "plan_sha256": plan_sha256,
        "stage": stage,
        "stage_ordinal": ordinal,
        "previous_checkpoint_sha256": (chain[-1].checkpoint_sha256 if chain else None),
        "artifacts": artifacts,
        "artifact_membership_sha256": hash_canonical(
            [item.model_dump(mode="json") for item in artifacts]
        ),
    }
    checkpoint = MetaSynPassageRescueStageCheckpointV3.model_validate(
        {**payload, "checkpoint_sha256": hash_canonical(payload)}
    )
    atomic_write_json(_checkpoint_path(workspace, stage), checkpoint)
    return checkpoint


def _load_plan(
    *, repository_root: Path, v2_workspace: Path, workspace: Path
) -> MetaSynPassagePacketRescuePlanV3:
    plan = MetaSynPassagePacketRescuePlanV3.model_validate(
        _read_object(_runtime_paths(workspace)["plan"])
    )
    return validate_metasyn_passage_packet_rescue_plan_v3(
        plan=plan,
        repository_root=repository_root,
        v2_workspace=v2_workspace,
        external_replay=True,
    )


def _rescue_intents(
    plan: MetaSynPassagePacketRescuePlanV3,
) -> list[HostedExactOnceIntentV1]:
    return [
        freeze_hosted_exact_once_intent(
            execution_bundle_sha256=plan.plan_sha256,
            phase="smoke_packet",
            source_bearing=True,
            context_binding_sha256=item.rescue_request_sha256,
            request=item.request,
        )
        for item in plan.requests
    ]


def _freeze_authorization(
    plan: MetaSynPassagePacketRescuePlanV3,
) -> MetaSynPassageRescueAuthorizationV3:
    exact = freeze_hosted_exact_once_cost_authorization(
        execution_bundle_sha256=plan.plan_sha256,
        phase="smoke_packet",
        intents=_rescue_intents(plan),
        configured_phase_budget_usd_micros=(plan.configured_cost_cap_usd_micros),
    )
    payload: dict[str, Any] = {
        "authorization_version": ("metasyn-passage-packet-rescue-authorization-v3"),
        "status": ("exact_three_call_cost_authorization_persisted_before_liability"),
        "plan_sha256": plan.plan_sha256,
        "request_membership_sha256": plan.request_membership_sha256,
        "exact_authorization": exact,
        "exact_authorization_sha256": exact.authorization_sha256,
        "authorized_call_count": exact.authorized_call_count,
        "conservative_cost_ceiling_usd_micros": exact.cost_ceiling_usd_micros,
        "configured_cost_cap_usd_micros": plan.configured_cost_cap_usd_micros,
        "provider_calls_made_before_authorization": 0,
        "retries_per_request": 0,
    }
    return MetaSynPassageRescueAuthorizationV3.model_validate(
        {
            **payload,
            "authorization_receipt_sha256": hash_canonical(payload),
        }
    )


def _terminal_sha256(outcome: RescueTerminalOutcomeV3) -> str:
    return (
        outcome.receipt_sha256
        if isinstance(outcome, HostedExactOnceProviderReceiptV1)
        else outcome.incident_sha256
    )


def _process_rescue_outcome(
    *,
    plan: MetaSynPassagePacketRescuePlanV3,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    rescue_request: MetaSynPassageRescueRequestV3,
    outcome: RescueTerminalOutcomeV3,
) -> MetaSynPassageRescueResultV3:
    normalization: NativePacketCompactNormalizationReceiptV3 | None = None
    grounding: PacketGroundingReceiptV2 | None = None
    assembly: NativePacketAssemblyOutcomeV2 | None = None
    if isinstance(outcome, HostedExactOnceAmbiguityIncidentV1):
        status: RescueValidationStatusV3 = "exact_once_terminal_incident"
    elif outcome.provider_result.outcome != "completed":
        status = "provider_runtime_failure"
    elif not isinstance(outcome.provider_result.parsed_json, Mapping):
        status = "grounding_invalid"
    else:
        try:
            normalization = freeze_native_packet_compact_normalization_receipt_v3(
                raw_model_outcome=outcome.provider_result.parsed_json,
                expected_candidate_binding_sha256=(rescue_request.candidate_binding_sha256),
            )
            row = bundle.extraction_inputs.rows[rescue_request.row_ordinal]
            grounding = freeze_passage_packet_grounding_receipt_v2(
                model_outcome=normalization.normalized_model_outcome,
                candidate=rescue_request.source_packet_request.packet_input.candidate,
                projection=row.projection_v2,
            )
        except (ValueError, TypeError):
            grounding = None
            status = "grounding_invalid"
        else:
            try:
                protocol = replay_metasyn_question_projection_spec_v2(
                    question_surface=row.question_surface
                )
                assembly = assemble_native_packet_v2(
                    candidate=rescue_request.source_packet_request.packet_input.candidate,
                    projection=row.projection_v2,
                    protocol=protocol,
                    protocol_orientation=(
                        bundle.protocol_orientations[
                            rescue_request.row_ordinal
                        ].protocol_orientation
                    ),
                    analysis_policy=bundle.assembly_analysis_policy,
                    grounding_receipt=grounding,
                )
            except (ValueError, TypeError):
                assembly = None
                status = "assembly_invalid"
            else:
                if isinstance(assembly, NativePacketAssemblyCompletedV2):
                    status = "typed_effect_completed"
                elif isinstance(grounding, PacketGroundingAbstentionReceiptV2):
                    status = "grounding_abstained"
                else:
                    status = "assembly_abstained"
    payload: dict[str, Any] = {
        "result_version": RESCUE_RESULT_VERSION,
        "plan_sha256": plan.plan_sha256,
        "rescue_request_sha256": rescue_request.rescue_request_sha256,
        "request_key": rescue_request.request.request_key,
        "row_ordinal": rescue_request.row_ordinal,
        "candidate_index": rescue_request.candidate_index,
        "packet_input": rescue_request.source_packet_request.packet_input,
        "packet_input_sha256": (rescue_request.source_packet_request.packet_input_sha256),
        "terminal": outcome,
        "terminal_sha256": _terminal_sha256(outcome),
        "validation_status": status,
        "compact_normalization_receipt": normalization,
        "compact_normalization_receipt_sha256": (
            normalization.receipt_sha256 if normalization else None
        ),
        "grounding_receipt": grounding,
        "grounding_receipt_sha256": grounding.receipt_sha256 if grounding else None,
        "assembly_receipt": assembly,
        "assembly_receipt_sha256": (assembly.assembly_receipt_sha256 if assembly else None),
        "authorizes_typed_effect": status == "typed_effect_completed",
        "standard_packet_input_grounding_assembly_triple_persisted": (
            grounding is not None and assembly is not None
        ),
        "bridge_v2_single_terminal_shape_compatible": (
            grounding is not None and assembly is not None
        ),
        "complete_v2_authorized_candidate_terminal_roster": False,
        "bridge_v2_full_corpus_input_ready": False,
        "runtime_failure_is_not_scientific_abstention": True,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynPassageRescueResultV3.model_validate(
        {**payload, "result_sha256": hash_canonical(payload)}
    )


def _result_path(workspace: Path, request_key: str) -> Path:
    return _runtime_paths(workspace)["results"] / f"{request_key}.json"


def _grounding_path(workspace: Path, request_key: str) -> Path:
    return _runtime_paths(workspace)["grounding"] / f"{request_key}.json"


def _assembly_path(workspace: Path, request_key: str) -> Path:
    return _runtime_paths(workspace)["assembly"] / f"{request_key}.json"


def _persist_result_artifacts(*, workspace: Path, result: MetaSynPassageRescueResultV3) -> None:
    if result.grounding_receipt is not None:
        _write_or_replay(
            _grounding_path(workspace, result.request_key),
            result.grounding_receipt,
        )
    if result.assembly_receipt is not None:
        _write_or_replay(
            _assembly_path(workspace, result.request_key),
            result.assembly_receipt,
        )
    _write_or_replay(_result_path(workspace, result.request_key), result)


def _validate_result_artifacts(
    *,
    workspace: Path,
    plan: MetaSynPassagePacketRescuePlanV3,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    rescue_request: MetaSynPassageRescueRequestV3,
    outcome: RescueTerminalOutcomeV3,
) -> MetaSynPassageRescueResultV3:
    saved = MetaSynPassageRescueResultV3.model_validate(
        _read_object(_result_path(workspace, rescue_request.request.request_key))
    )
    expected = _process_rescue_outcome(
        plan=plan,
        bundle=bundle,
        rescue_request=rescue_request,
        outcome=outcome,
    )
    if saved != expected:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_result_external_replay_mismatch"
        )
    if saved.compact_normalization_receipt is not None:
        if not isinstance(outcome, HostedExactOnceProviderReceiptV1) or not isinstance(
            outcome.provider_result.parsed_json, Mapping
        ):
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_normalization_terminal_shape_invalid"
            )
        validate_native_packet_compact_normalization_receipt_v3(
            receipt=saved.compact_normalization_receipt,
            raw_model_outcome=outcome.provider_result.parsed_json,
            expected_candidate_binding_sha256=(rescue_request.candidate_binding_sha256),
        )
    if saved.grounding_receipt is not None:
        grounding = _GROUNDING_ADAPTER.validate_python(
            _read_object(_grounding_path(workspace, saved.request_key))
        )
        row = bundle.extraction_inputs.rows[saved.row_ordinal]
        grounding = validate_passage_packet_grounding_receipt_v2(
            receipt=grounding,
            model_outcome=(
                saved.compact_normalization_receipt.normalized_model_outcome
                if saved.compact_normalization_receipt
                else {}
            ),
            candidate=rescue_request.source_packet_request.packet_input.candidate,
            projection=row.projection_v2,
        )
        if grounding != saved.grounding_receipt:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_grounding_external_replay_mismatch"
            )
    elif _grounding_path(workspace, saved.request_key).exists():
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_unexpected_grounding_artifact"
        )
    if saved.assembly_receipt is not None:
        assembly = _ASSEMBLY_ADAPTER.validate_python(
            _read_object(_assembly_path(workspace, saved.request_key))
        )
        row = bundle.extraction_inputs.rows[saved.row_ordinal]
        protocol = replay_metasyn_question_projection_spec_v2(question_surface=row.question_surface)
        assembly = validate_native_packet_assembly_v2(
            assembly=assembly,
            candidate=rescue_request.source_packet_request.packet_input.candidate,
            projection=row.projection_v2,
            protocol=protocol,
            protocol_orientation=(
                bundle.protocol_orientations[rescue_request.row_ordinal].protocol_orientation
            ),
            analysis_policy=bundle.assembly_analysis_policy,
            grounding_receipt=saved.grounding_receipt,
        )
        if assembly != saved.assembly_receipt:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_assembly_external_replay_mismatch"
            )
    elif _assembly_path(workspace, saved.request_key).exists():
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_unexpected_assembly_artifact"
        )
    return saved


def _freeze_smoke_receipt(
    *,
    plan: MetaSynPassagePacketRescuePlanV3,
    authorization: MetaSynPassageRescueAuthorizationV3,
    results: Sequence[MetaSynPassageRescueResultV3],
) -> MetaSynPassageRescueSmokeReceiptV3:
    completed = next((item for item in results if item.authorizes_typed_effect), None)
    payload: dict[str, Any] = {
        "smoke_version": RESCUE_SMOKE_VERSION,
        "status": "passed" if completed else "failed_gate",
        "plan_sha256": plan.plan_sha256,
        "authorization_receipt_sha256": (authorization.authorization_receipt_sha256),
        "ordered_authorized_request_keys": [item.request.request_key for item in plan.requests],
        "attempted_request_keys": [item.request_key for item in results],
        "results": list(results),
        "result_membership_sha256": hash_canonical([item.result_sha256 for item in results]),
        "completed_typed_effect_result_sha256": (completed.result_sha256 if completed else None),
        "typed_effect_count": int(completed is not None),
        "valid_abstention_does_not_pass": True,
        "compact_normalization_only_absent_invariants": True,
        "retries_per_request": 0,
        "remaining_calls_under_this_smoke_authorization_permitted": False,
        "future_additive_full_roster_extension_possible": True,
        "complete_v2_authorized_candidate_terminal_roster": False,
        "bridge_v2_full_corpus_input_ready": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynPassageRescueSmokeReceiptV3.model_validate(
        {**payload, "smoke_sha256": hash_canonical(payload)}
    )


def prepare_metasyn_passage_packet_rescue_v3(
    *,
    repository_root: Path,
    workspace: Path = DEFAULT_RESCUE_WORKSPACE,
    v2_workspace: Path = DEFAULT_V2_WORKSPACE,
) -> MetaSynPassagePacketRescuePlanV3:
    root = _canonical_repository_root(repository_root)
    plan = freeze_metasyn_passage_packet_rescue_plan_v3(
        repository_root=root, v2_workspace=v2_workspace
    )
    ws = _create_fresh_workspace(workspace)
    with _runtime_lock(ws):
        paths = _runtime_paths(ws)
        for name in ("provider", "results", "grounding", "assembly", "stages"):
            paths[name].mkdir(mode=0o700)
        _write_or_replay(paths["plan"], plan)
        _write_or_replay(paths["forensic"], plan.v2_forensic_receipt)
        _write_or_replay(paths["blocker"], plan.pre_call_blocker)
        replayed = _load_plan(repository_root=root, v2_workspace=v2_workspace, workspace=ws)
        if replayed != plan:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_written_plan_replay_mismatch"
            )
        _write_checkpoint(
            workspace=ws,
            plan_sha256=plan.plan_sha256,
            stage="prepared",
            artifact_paths=[paths["plan"], paths["forensic"], paths["blocker"]],
        )
    return plan


def authorize_metasyn_passage_packet_rescue_v3(
    *,
    repository_root: Path,
    workspace: Path,
    v2_workspace: Path = DEFAULT_V2_WORKSPACE,
    expected_plan_sha256: str,
) -> MetaSynPassageRescueAuthorizationV3:
    root = _canonical_repository_root(repository_root)
    ws = _canonical_existing_workspace(workspace, name="rescue")
    with _runtime_lock(ws):
        chain = _load_stage_chain(ws)
        if not chain or chain[0].plan_sha256 != expected_plan_sha256:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_expected_plan_mismatch"
            )
        plan = _load_plan(repository_root=root, v2_workspace=v2_workspace, workspace=ws)
        paths = _runtime_paths(ws)
        if not plan.provider_calls_permitted or not plan.pre_call_blocker.calls_permitted:
            if paths["authorization"].exists() or list(paths["provider"].rglob("*.json")):
                raise MetaSynPassagePacketRescueV3Error(
                    "metasyn_packet_rescue_v3_blocked_plan_has_provider_liability_artifact"
                )
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_pre_call_zero_yield_blocker"
            )
        if len(chain) > 1:
            return MetaSynPassageRescueAuthorizationV3.model_validate(
                _read_object(paths["authorization"])
            )
        provider_files = list(paths["provider"].rglob("*.json"))
        if provider_files:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_provider_state_precedes_authorization"
            )
        authorization = _freeze_authorization(plan)
        _write_or_replay(paths["authorization"], authorization)
        provider_auth_dir = paths["provider"] / "cost-authorizations"
        provider_auth_dir.mkdir(mode=0o700)
        provider_auth_path = provider_auth_dir / "smoke_packet.json"
        atomic_write_json(provider_auth_path, authorization.exact_authorization)
        _write_checkpoint(
            workspace=ws,
            plan_sha256=plan.plan_sha256,
            stage="authorized",
            artifact_paths=[paths["authorization"], provider_auth_path],
        )
        return authorization


def run_metasyn_passage_packet_rescue_smoke_v3(
    *,
    repository_root: Path,
    workspace: Path,
    v2_workspace: Path = DEFAULT_V2_WORKSPACE,
    expected_plan_sha256: str,
    client: HostedClientProtocol,
) -> MetaSynPassageRescueSmokeReceiptV3:
    root = _canonical_repository_root(repository_root)
    ws = _canonical_existing_workspace(workspace, name="rescue")
    with _runtime_lock(ws):
        chain = _load_stage_chain(ws)
        if not chain or chain[0].plan_sha256 != expected_plan_sha256:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_expected_plan_mismatch"
            )
        plan = _load_plan(repository_root=root, v2_workspace=v2_workspace, workspace=ws)
        if not plan.provider_calls_permitted or not plan.pre_call_blocker.calls_permitted:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_pre_call_zero_yield_blocker"
            )
        if len(chain) < 2:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_smoke_requires_authorization"
            )
        paths = _runtime_paths(ws)
        if len(chain) > 2:
            return MetaSynPassageRescueSmokeReceiptV3.model_validate(_read_object(paths["smoke"]))
        context = _replay_v2_base(repository_root=root, v2_workspace=v2_workspace)
        authorization = MetaSynPassageRescueAuthorizationV3.model_validate(
            _read_object(paths["authorization"])
        )
        expected_authorization = _freeze_authorization(plan)
        if authorization != expected_authorization:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_authorization_external_replay_mismatch"
            )
        intents = _rescue_intents(plan)
        results: list[MetaSynPassageRescueResultV3] = []
        for rescue_request, intent in zip(plan.requests, intents, strict=True):
            outcome = execute_hosted_exactly_once(
                workspace=paths["provider"],
                intent=intent,
                authorization=authorization.exact_authorization,
                client=client,
            )
            outcome = validate_hosted_exact_once_outcome(
                workspace=paths["provider"],
                intent=intent,
                authorization=authorization.exact_authorization,
            )
            result = _process_rescue_outcome(
                plan=plan,
                bundle=context.bundle,
                rescue_request=rescue_request,
                outcome=outcome,
            )
            _persist_result_artifacts(workspace=ws, result=result)
            results.append(result)
            if isinstance(outcome, HostedExactOnceAmbiguityIncidentV1):
                break
            if result.authorizes_typed_effect:
                break
        smoke = _freeze_smoke_receipt(plan=plan, authorization=authorization, results=results)
        _write_or_replay(paths["smoke"], smoke)
        result_artifacts = [_result_path(ws, item.request_key) for item in results]
        result_artifacts.extend(
            _grounding_path(ws, item.request_key)
            for item in results
            if item.grounding_receipt is not None
        )
        result_artifacts.extend(
            _assembly_path(ws, item.request_key)
            for item in results
            if item.assembly_receipt is not None
        )
        _write_checkpoint(
            workspace=ws,
            plan_sha256=plan.plan_sha256,
            stage="smoke_passed" if smoke.status == "passed" else "smoke_failed",
            artifact_paths=[paths["smoke"], *result_artifacts],
        )
        return smoke


def _freeze_final_report(
    *,
    plan: MetaSynPassagePacketRescuePlanV3,
    authorization: MetaSynPassageRescueAuthorizationV3,
    smoke: MetaSynPassageRescueSmokeReceiptV3,
) -> MetaSynPassageRescueFinalReportV3:
    liability = sum(
        next(
            request.request_cost_ceiling_usd_micros
            for request in plan.requests
            if request.request.request_key == result.request_key
        )
        for result in smoke.results
    )
    payload: dict[str, Any] = {
        "report_version": RESCUE_REPORT_VERSION,
        "status": ("complete_post_hoc_exploratory_yield_report_no_accuracy_or_release_authority"),
        "plan_sha256": plan.plan_sha256,
        "pipeline_sha256": plan.pipeline_sha256,
        "v2_replay_snapshot_sha256": plan.v2_replay_snapshot_sha256,
        "v2_forensic_receipt_sha256": plan.v2_forensic_receipt_sha256,
        "authorization_receipt_sha256": (authorization.authorization_receipt_sha256),
        "smoke_sha256": smoke.smoke_sha256,
        "smoke_status": smoke.status,
        "attempted_call_count": len(smoke.results),
        "terminal_result_membership_sha256": smoke.result_membership_sha256,
        "typed_effect_count": smoke.typed_effect_count,
        "grounding_abstention_count": sum(
            item.validation_status == "grounding_abstained" for item in smoke.results
        ),
        "conservative_attempt_liability_usd_micros": liability,
        "configured_cost_cap_usd_micros": plan.configured_cost_cap_usd_micros,
        "standard_completed_terminal_triples_persisted": (
            smoke.typed_effect_count == 1
            and all(
                item.standard_packet_input_grounding_assembly_triple_persisted
                for item in smoke.results
                if item.authorizes_typed_effect
            )
        ),
        "complete_v2_authorized_candidate_terminal_roster": False,
        "bridge_v2_full_corpus_input_ready": False,
        "future_additive_full_roster_extension_possible": True,
        "post_hoc_exploratory": True,
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "extraction_accuracy_authority": False,
        "scientific_effectiveness_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynPassageRescueFinalReportV3.model_validate(
        {**payload, "report_sha256": hash_canonical(payload)}
    )


def _load_authorization(
    *, workspace: Path, plan: MetaSynPassagePacketRescuePlanV3
) -> MetaSynPassageRescueAuthorizationV3:
    authorization = MetaSynPassageRescueAuthorizationV3.model_validate(
        _read_object(_runtime_paths(workspace)["authorization"])
    )
    expected = _freeze_authorization(plan)
    if authorization != expected:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_authorization_external_replay_mismatch"
        )
    provider_auth = HostedExactOnceCostAuthorizationV1.model_validate(
        _read_object(
            _runtime_paths(workspace)["provider"] / "cost-authorizations/smoke_packet.json"
        )
    )
    if provider_auth != authorization.exact_authorization:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_provider_authorization_mismatch"
        )
    return authorization


def finalize_metasyn_passage_packet_rescue_v3(
    *,
    repository_root: Path,
    workspace: Path,
    v2_workspace: Path = DEFAULT_V2_WORKSPACE,
    expected_plan_sha256: str,
) -> MetaSynPassageRescueFinalReportV3:
    root = _canonical_repository_root(repository_root)
    ws = _canonical_existing_workspace(workspace, name="rescue")
    with _runtime_lock(ws):
        chain = _load_stage_chain(ws)
        if len(chain) < 3 or chain[0].plan_sha256 != expected_plan_sha256:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_finalize_requires_terminal_smoke"
            )
        paths = _runtime_paths(ws)
        if len(chain) > 3:
            return MetaSynPassageRescueFinalReportV3.model_validate(_read_object(paths["report"]))
        plan = _load_plan(repository_root=root, v2_workspace=v2_workspace, workspace=ws)
        authorization = _load_authorization(workspace=ws, plan=plan)
        smoke = MetaSynPassageRescueSmokeReceiptV3.model_validate(_read_object(paths["smoke"]))
        report = _freeze_final_report(plan=plan, authorization=authorization, smoke=smoke)
        _write_or_replay(paths["report"], report)
        _write_checkpoint(
            workspace=ws,
            plan_sha256=plan.plan_sha256,
            stage="finalized",
            artifact_paths=[paths["report"]],
        )
        return report


def _validate_provider_file_sets(
    *,
    workspace: Path,
    smoke: MetaSynPassageRescueSmokeReceiptV3,
) -> None:
    provider = _runtime_paths(workspace)["provider"]
    attempted = set(smoke.attempted_request_keys)
    intent_dir = provider / "call-intents"
    receipt_dir = provider / "provider-receipts"
    incident_dir = provider / "ambiguity-incidents"
    intent_keys = (
        {path.stem for path in intent_dir.glob("*.json")} if intent_dir.exists() else set()
    )
    receipt_keys = (
        {path.stem for path in receipt_dir.glob("*.json")} if receipt_dir.exists() else set()
    )
    incident_keys = (
        {path.stem for path in incident_dir.glob("*.json")} if incident_dir.exists() else set()
    )
    if (
        intent_keys != attempted
        or receipt_keys & incident_keys
        or receipt_keys | incident_keys != attempted
    ):
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_exact_once_file_set_mismatch"
        )
    authorization_files = {path.name for path in (provider / "cost-authorizations").glob("*.json")}
    if authorization_files != {"smoke_packet.json"}:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_authorization_file_set_mismatch"
        )


def _externally_replay_finalized(
    *,
    repository_root: Path,
    workspace: Path,
    v2_workspace: Path,
    expected_plan_sha256: str,
) -> tuple[
    MetaSynPassagePacketRescuePlanV3,
    MetaSynPassageRescueSmokeReceiptV3,
    MetaSynPassageRescueFinalReportV3,
]:
    plan = _load_plan(
        repository_root=repository_root,
        v2_workspace=v2_workspace,
        workspace=workspace,
    )
    if plan.plan_sha256 != expected_plan_sha256:
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_expected_plan_mismatch")
    context = _replay_v2_base(repository_root=repository_root, v2_workspace=v2_workspace)
    authorization = _load_authorization(workspace=workspace, plan=plan)
    saved_smoke = MetaSynPassageRescueSmokeReceiptV3.model_validate(
        _read_object(_runtime_paths(workspace)["smoke"])
    )
    intents_by_key = {intent.request_key: intent for intent in _rescue_intents(plan)}
    requests_by_key = {item.request.request_key: item for item in plan.requests}
    results: list[MetaSynPassageRescueResultV3] = []
    for request_key in saved_smoke.attempted_request_keys:
        request = requests_by_key.get(request_key)
        intent = intents_by_key.get(request_key)
        if request is None or intent is None:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_smoke_request_not_authorized"
            )
        outcome = validate_hosted_exact_once_outcome(
            workspace=_runtime_paths(workspace)["provider"],
            intent=intent,
            authorization=authorization.exact_authorization,
        )
        result = _validate_result_artifacts(
            workspace=workspace,
            plan=plan,
            bundle=context.bundle,
            rescue_request=request,
            outcome=outcome,
        )
        results.append(result)
    expected_smoke = _freeze_smoke_receipt(plan=plan, authorization=authorization, results=results)
    if saved_smoke != expected_smoke:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_smoke_external_replay_mismatch"
        )
    _validate_provider_file_sets(workspace=workspace, smoke=saved_smoke)
    expected_result_files = {f"{item.request_key}.json" for item in results}
    actual_result_files = {
        path.name for path in _runtime_paths(workspace)["results"].glob("*.json")
    }
    if actual_result_files != expected_result_files:
        raise MetaSynPassagePacketRescueV3Error("metasyn_packet_rescue_v3_result_file_set_mismatch")
    expected_grounding_files = {
        f"{item.request_key}.json" for item in results if item.grounding_receipt is not None
    }
    actual_grounding_files = {
        path.name for path in _runtime_paths(workspace)["grounding"].glob("*.json")
    }
    if actual_grounding_files != expected_grounding_files:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_grounding_file_set_mismatch"
        )
    expected_assembly_files = {
        f"{item.request_key}.json" for item in results if item.assembly_receipt is not None
    }
    actual_assembly_files = {
        path.name for path in _runtime_paths(workspace)["assembly"].glob("*.json")
    }
    if actual_assembly_files != expected_assembly_files:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_assembly_file_set_mismatch"
        )
    saved_report = MetaSynPassageRescueFinalReportV3.model_validate(
        _read_object(_runtime_paths(workspace)["report"])
    )
    expected_report = _freeze_final_report(
        plan=plan, authorization=authorization, smoke=saved_smoke
    )
    if saved_report != expected_report:
        raise MetaSynPassagePacketRescueV3Error(
            "metasyn_packet_rescue_v3_report_external_replay_mismatch"
        )
    return plan, saved_smoke, saved_report


def validate_finalized_metasyn_passage_packet_rescue_v3(
    *,
    repository_root: Path,
    workspace: Path,
    v2_workspace: Path = DEFAULT_V2_WORKSPACE,
    expected_plan_sha256: str,
    mark_externally_validated: bool = True,
) -> MetaSynPassageRescueExternalValidationV3:
    root = _canonical_repository_root(repository_root)
    ws = _canonical_existing_workspace(workspace, name="rescue")
    with _runtime_lock(ws):
        chain = _load_stage_chain(ws)
        if len(chain) < 4 or chain[0].plan_sha256 != expected_plan_sha256:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_validation_requires_finalized"
            )
        plan, smoke, report = _externally_replay_finalized(
            repository_root=root,
            workspace=ws,
            v2_workspace=v2_workspace,
            expected_plan_sha256=expected_plan_sha256,
        )
        payload: dict[str, Any] = {
            "validation_version": RESCUE_VALIDATION_VERSION,
            "status": ("v2_base_plan_authorization_terminals_and_report_externally_replayed"),
            "plan_sha256": plan.plan_sha256,
            "report_sha256": report.report_sha256,
            "exact_terminal_count": len(smoke.results),
            "completed_typed_effect_count": smoke.typed_effect_count,
            "provider_calls_made_by_validation": 0,
            "reference_fields_unopened": True,
            "official_test_labels_opened": False,
            "claim_release_authority": False,
        }
        validation = MetaSynPassageRescueExternalValidationV3.model_validate(
            {**payload, "validation_sha256": hash_canonical(payload)}
        )
        path = _runtime_paths(ws)["validation"]
        if len(chain) > 4:
            saved = MetaSynPassageRescueExternalValidationV3.model_validate(_read_object(path))
            if saved != validation:
                raise MetaSynPassagePacketRescueV3Error(
                    "metasyn_packet_rescue_v3_validation_receipt_mismatch"
                )
            return saved
        if mark_externally_validated:
            _write_or_replay(path, validation)
            _write_checkpoint(
                workspace=ws,
                plan_sha256=plan.plan_sha256,
                stage="externally_validated",
                artifact_paths=[path],
            )
        return validation


def metasyn_passage_packet_rescue_status_v3(
    *,
    repository_root: Path,
    workspace: Path,
    v2_workspace: Path = DEFAULT_V2_WORKSPACE,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    root = _canonical_repository_root(repository_root)
    ws = _canonical_existing_workspace(workspace, name="rescue")
    with _runtime_lock(ws):
        chain = _load_stage_chain(ws)
        if not chain or chain[0].plan_sha256 != expected_plan_sha256:
            raise MetaSynPassagePacketRescueV3Error(
                "metasyn_packet_rescue_v3_expected_plan_mismatch"
            )
        plan = _load_plan(repository_root=root, v2_workspace=v2_workspace, workspace=ws)
        output: dict[str, Any] = {
            "runtime_version": "metasyn-passage-packet-rescue-runtime-v3",
            "plan_sha256": plan.plan_sha256,
            "pipeline_sha256": plan.pipeline_sha256,
            "current_stage": chain[-1].stage,
            "stage_ordinal": chain[-1].stage_ordinal,
            "checkpoint_sha256": chain[-1].checkpoint_sha256,
            "maximum_smoke_calls": len(plan.requests),
            "conservative_cost_ceiling_usd_micros": (plan.conservative_cost_ceiling_usd_micros),
            "pre_call_blocker_sha256": plan.pre_call_blocker_sha256,
            "provider_calls_permitted": False,
            "authorization_created": False,
            "provider_cost_liability_usd_micros": 0,
            "post_hoc_exploratory": True,
            "extraction_accuracy_authority": False,
            "synthesis_input_authority": False,
            "claim_release_authority": False,
        }
        if _runtime_paths(ws)["smoke"].exists():
            smoke = MetaSynPassageRescueSmokeReceiptV3.model_validate(
                _read_object(_runtime_paths(ws)["smoke"])
            )
            output.update(
                {
                    "smoke_status": smoke.status,
                    "attempted_call_count": len(smoke.results),
                    "typed_effect_count": smoke.typed_effect_count,
                    "bridge_v2_full_corpus_input_ready": False,
                }
            )
        return output


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_RESCUE_WORKSPACE",
    "DEFAULT_V2_WORKSPACE",
    "EXPECTED_V2_EXECUTION_BUNDLE_SHA256",
    "EXPECTED_V2_FAILED_SMOKE_SHA256",
    "EXPECTED_V2_INVENTORY_LEDGER_SHA256",
    "EXPECTED_V2_PACKET_ROSTER_SHA256",
    "MAXIMUM_RESCUE_SMOKE_CALLS",
    "MetaSynPassagePacketRescueConfigV3",
    "MetaSynPassagePacketRescuePlanV3",
    "MetaSynPassagePacketRescueV3Error",
    "MetaSynPassageRescueAuthorizationV3",
    "MetaSynPassageRescueExternalValidationV3",
    "MetaSynPassageRescueFinalReportV3",
    "MetaSynPassageRescuePreCallBlockerItemV3",
    "MetaSynPassageRescuePreCallBlockerV3",
    "MetaSynPassageRescueResultV3",
    "MetaSynPassageRescueSmokeReceiptV3",
    "MetaSynV2CompactSmokeForensicReceiptV3",
    "MetaSynV2ReplaySnapshotV3",
    "authorize_metasyn_passage_packet_rescue_v3",
    "compute_metasyn_passage_packet_rescue_pipeline_fingerprint_v3",
    "finalize_metasyn_passage_packet_rescue_v3",
    "freeze_metasyn_passage_packet_rescue_plan_v3",
    "load_metasyn_passage_packet_rescue_config_v3",
    "metasyn_passage_packet_rescue_status_v3",
    "prepare_metasyn_passage_packet_rescue_v3",
    "run_metasyn_passage_packet_rescue_smoke_v3",
    "validate_finalized_metasyn_passage_packet_rescue_v3",
    "validate_metasyn_passage_packet_rescue_plan_v3",
]
