"""Exact-once source-visible binary packet mechanics smoke.

This additive v4 diagnostic is intentionally narrow.  It externally replays the
immutable v2 run (including all 43 provider receipts and its failed packet gate),
then targets two previously-unattempted binary candidates whose *contract
feasibility* can be proven from exact source literals before any new provider
liability is created.  The diagnostic measures typed-packet mechanics yield only.
It has no extraction-accuracy, synthesis-input, or claim-release authority.

The v2 artifacts and scientific contracts are never repaired or weakened.  Live
outputs pass through the strict compact normalizer v3 and then the unchanged v2
grounding and assembly functions.  Every possible provider attempt is governed by
the generic durable exact-once executor: the complete two-call ceiling is persisted
before the first intent, application/SDK retries are zero, and ambiguity poisons
the request and stops the stage.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from pydantic import ConfigDict, Field, TypeAdapter, field_validator, model_validator

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
from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file
from literature_multiverse.metasyn_extraction_inputs_v2 import MetaSynPacketCandidateInputV2
from literature_multiverse.metasyn_passage_hosted_bundle_v2 import (
    MetaSynPassageHostedExecutionBundleV2,
)
from literature_multiverse.metasyn_passage_hosted_runtime_v2 import PacketRequestV2
from literature_multiverse.metasyn_passage_packet_rescue_v3 import (
    EXPECTED_V2_EXECUTION_BUNDLE_SHA256,
    EXPECTED_V2_FAILED_SMOKE_SHA256,
    EXPECTED_V2_INVENTORY_LEDGER_SHA256,
    EXPECTED_V2_PACKET_ROSTER_SHA256,
    EXPECTED_V2_PROVIDER_RECEIPT_COUNT,
    _python_dependency_closure,
    _replay_v2_base,
    _scientific_request_signature,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_packet_assembly_v2 import (
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
    PacketGroundingModelCompletedV2,
    PacketGroundingReceiptV2,
    freeze_passage_packet_grounding_receipt_v2,
    validate_passage_packet_grounding_receipt_v2,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    compute_pipeline_fingerprint,
)

BINARY_SMOKE_CONFIG_VERSION = "metasyn-passage-binary-mechanics-config-v4"
BINARY_SMOKE_WITNESS_VERSION = "metasyn-passage-binary-mechanics-witness-v4"
BINARY_SMOKE_REQUEST_VERSION = "metasyn-passage-binary-mechanics-request-v4"
BINARY_SMOKE_PLAN_VERSION = "metasyn-passage-binary-mechanics-plan-v4"
BINARY_SMOKE_AUTHORIZATION_VERSION = "metasyn-passage-binary-mechanics-authorization-v4"
BINARY_SMOKE_RESULT_VERSION = "metasyn-passage-binary-mechanics-result-v4"
BINARY_SMOKE_RECEIPT_VERSION = "metasyn-passage-binary-mechanics-smoke-v4"
BINARY_SMOKE_REPORT_VERSION = "metasyn-passage-binary-mechanics-report-v4"
BINARY_SMOKE_VALIDATION_VERSION = "metasyn-passage-binary-mechanics-validation-v4"
BINARY_SMOKE_CHECKPOINT_VERSION = "metasyn-passage-binary-mechanics-checkpoint-v4"

DEFAULT_CONFIG_PATH = Path("configs/benchmarks/metasyn-passage-binary-mechanics-smoke-v4.json")
DEFAULT_V2_WORKSPACE = Path("data/cache/metasyn/passage-hosted-yield-v2")
DEFAULT_BINARY_SMOKE_WORKSPACE = Path("data/cache/metasyn/passage-binary-mechanics-smoke-v4")

MAXIMUM_BINARY_SMOKE_CALLS = 2
SELECTED_CANDIDATES: tuple[tuple[int, int], ...] = ((17, 2), (17, 3))
_OPERATION = "metasyn-passage-binary-mechanics-smoke-v4"
_REQUEST_KEY_PREFIX = "binary-v4"
_PHASE: Literal["smoke_packet"] = "smoke_packet"

_NEW_FINGERPRINT_FILES = (
    "src/literature_multiverse/metasyn_passage_binary_mechanics_smoke_v4.py",
    "scripts/run_metasyn_passage_binary_mechanics_smoke_v4.py",
    DEFAULT_CONFIG_PATH.as_posix(),
)


class MetaSynPassageBinaryMechanicsSmokeV4Error(ValueError):
    """A v4 artifact or exact-once transition is unsafe."""


class HostedClientProtocol(Protocol):
    def generate(self, request: AnthropicBoundedRequestV1) -> AnthropicBoundedResultV1:
        """Make the sole authorized attempt for one request."""


class _FrozenExactModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]


def _validate_self_hash(model: _FrozenExactModel, field_name: str) -> None:
    payload = model.model_dump(mode="json", exclude={field_name})
    if getattr(model, field_name) != hash_canonical(payload):
        raise ValueError(f"binary_mechanics_v4_self_hash_mismatch:{field_name}")


def _usd_micros(value: Decimal) -> int:
    return int((value * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING))


def _canonical_root(value: Path) -> Path:
    root = Path(os.path.abspath(value))
    try:
        mode = root.lstat().st_mode
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_repository_root_unreadable"
        ) from exc
    if stat.S_ISLNK(mode) or not resolved.is_dir():
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_repository_root_unsafe"
        )
    return resolved


def _checked_file(*, root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise MetaSynPassageBinaryMechanicsSmokeV4Error("binary_mechanics_v4_relative_path_unsafe")
    candidate = root
    for part in relative.parts:
        candidate /= part
        try:
            mode = candidate.lstat().st_mode
        except OSError as exc:
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                f"binary_mechanics_v4_file_missing:{relative.as_posix()}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                f"binary_mechanics_v4_file_symlink:{relative.as_posix()}"
            )
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise MetaSynPassageBinaryMechanicsSmokeV4Error("binary_mechanics_v4_file_not_regular")
    return resolved


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MetaSynPassageBinaryMechanicsSmokeV4Error("binary_mechanics_v4_artifact_unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_artifact_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise MetaSynPassageBinaryMechanicsSmokeV4Error("binary_mechanics_v4_artifact_not_object")
    return value


class MetaSynPassageBinaryMechanicsConfigV4(_FrozenExactModel):
    config_version: Literal["metasyn-passage-binary-mechanics-config-v4"] = (
        BINARY_SMOKE_CONFIG_VERSION
    )
    diagnostic_scope: Literal[
        "post_hoc_label_blind_source_visible_binary_typed_effect_mechanics_yield_only"
    ]
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
    selected_candidate_coordinates: list[Literal["17:2", "17:3"]]
    maximum_smoke_calls: Literal[2]
    configured_cost_cap_usd_micros: Literal[1425336]
    operation: Literal["metasyn-passage-binary-mechanics-smoke-v4"]
    request_key_prefix: Literal["binary-v4"]
    packet_phase: Literal["smoke_packet"]
    selection_rule: Literal[
        "exact_unattempted_row17_binary_candidates_with_pre_call_source_literal_completed_witnesses"
    ]
    compact_normalizer_version: Literal["native-packet-compact-normalization-v3"]
    application_retries_per_request: Literal[0]
    sdk_retries_per_request: Literal[0]
    orphan_or_ambiguous_attempt_retry_permitted: Literal[False]
    stop_after_first_typed_effect: Literal[True]
    stop_and_poison_on_ambiguity: Literal[True]
    inventory_normalization_permitted: Literal[False]
    reference_fields_unopened: Literal[True]
    official_test_labels_opened: Literal[False]
    extraction_accuracy_authority: Literal[False]
    synthesis_input_authority: Literal[False]
    claim_release_authority: Literal[False]
    config_sha256: Sha256

    @field_validator("selected_candidate_coordinates")
    @classmethod
    def validate_coordinates(cls, value: list[str]) -> list[str]:
        if value != ["17:2", "17:3"]:
            raise ValueError("binary_mechanics_v4_candidate_coordinates_changed")
        return value

    @model_validator(mode="after")
    def validate_config(self) -> MetaSynPassageBinaryMechanicsConfigV4:
        _validate_self_hash(self, "config_sha256")
        return self


def load_metasyn_passage_binary_mechanics_config_v4(
    *, repository_root: Path
) -> tuple[MetaSynPassageBinaryMechanicsConfigV4, str]:
    root = _canonical_root(repository_root)
    path = _checked_file(root=root, relative=DEFAULT_CONFIG_PATH)
    try:
        config = MetaSynPassageBinaryMechanicsConfigV4.model_validate(_read_object(path))
    except ValueError as exc:
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_config_invalid"
        ) from exc
    return config, sha256_file(path)


class MetaSynBinaryFeasibilityWitnessV4(_FrozenExactModel):
    witness_version: Literal["metasyn-passage-binary-mechanics-witness-v4"] = (
        BINARY_SMOKE_WITNESS_VERSION
    )
    row_ordinal: Literal[17]
    candidate_index: Literal[2, 3]
    source_packet_request_sha256: Sha256
    packet_input_sha256: Sha256
    candidate_descriptor_sha256: Sha256
    candidate_binding_sha256: Sha256
    v2_scientific_request_signature_sha256: Sha256
    previously_attempted_in_immutable_v2: Literal[False]
    full_source_literal_completed_model_outcome: PacketGroundingModelCompletedV2
    full_source_literal_completed_model_outcome_sha256: Sha256
    grounding_receipt: PacketGroundingCompletedReceiptV2
    grounding_receipt_sha256: Sha256
    assembly_receipt: NativePacketAssemblyCompletedV2
    assembly_receipt_sha256: Sha256
    exact_source_literals_only: Literal[True]
    model_or_provider_called_for_witness: Literal[False]
    hidden_or_reference_labels_opened: Literal[False]
    relaxed_parsing_used: Literal[False]
    mechanics_feasibility_only: Literal[True]
    extraction_accuracy_authority: Literal[False]
    synthesis_input_authority: Literal[False]
    claim_release_authority: Literal[False]
    witness_sha256: Sha256

    @model_validator(mode="after")
    def validate_witness(self) -> MetaSynBinaryFeasibilityWitnessV4:
        if (
            self.full_source_literal_completed_model_outcome_sha256
            != hash_canonical(self.full_source_literal_completed_model_outcome)
            or self.grounding_receipt_sha256 != self.grounding_receipt.receipt_sha256
            or self.assembly_receipt_sha256 != self.assembly_receipt.assembly_receipt_sha256
        ):
            raise ValueError("binary_mechanics_v4_witness_receipt_hash_mismatch")
        if (
            self.full_source_literal_completed_model_outcome.candidate_binding_sha256
            != self.candidate_binding_sha256
            or self.grounding_receipt.candidate_binding.binding_sha256
            != self.candidate_binding_sha256
            or self.assembly_receipt.grounding_receipt_sha256 != self.grounding_receipt_sha256
            or not self.assembly_receipt.authorizes_typed_effect
        ):
            raise ValueError("binary_mechanics_v4_witness_alias_mismatch")
        _validate_self_hash(self, "witness_sha256")
        return self


class MetaSynBinaryMechanicsRequestV4(_FrozenExactModel):
    request_version: Literal["metasyn-passage-binary-mechanics-request-v4"] = (
        BINARY_SMOKE_REQUEST_VERSION
    )
    ordinal: Annotated[int, Field(ge=1, le=2)]
    row_ordinal: Literal[17]
    candidate_index: Literal[2, 3]
    source_packet_request: PacketRequestV2
    source_packet_request_sha256: Sha256
    packet_input_sha256: Sha256
    candidate_binding_sha256: Sha256
    v2_scientific_request_signature_sha256: Sha256
    feasibility_witness_sha256: Sha256
    request: AnthropicBoundedRequestV1
    request_sha256: Sha256
    request_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    fresh_operation_domain: Literal[True]
    fresh_request_key_domain: Literal[True]
    fresh_exact_once_intent_domain: Literal[True]
    previously_attempted_in_immutable_v2: Literal[False]
    request_receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_request(self) -> MetaSynBinaryMechanicsRequestV4:
        if (
            self.row_ordinal != self.source_packet_request.row_ordinal
            or self.candidate_index != self.source_packet_request.candidate_index
            or self.source_packet_request_sha256 != self.source_packet_request.packet_request_sha256
            or self.packet_input_sha256 != self.source_packet_request.packet_input_sha256
            or self.candidate_binding_sha256
            != self.source_packet_request.packet_input.candidate_binding_sha256
            or self.request_sha256 != self.request.request_sha256
            or self.request.operation != _OPERATION
            or self.request.request_key
            != f"{_REQUEST_KEY_PREFIX}-row-17-candidate-{self.candidate_index:02d}"
            or self.request_cost_ceiling_usd_micros
            != _usd_micros(self.request.cost_ceiling.request_cost_ceiling_usd)
        ):
            raise ValueError("binary_mechanics_v4_request_alias_mismatch")
        _validate_self_hash(self, "request_receipt_sha256")
        return self


class MetaSynPassageBinaryMechanicsPlanV4(_FrozenExactModel):
    plan_version: Literal["metasyn-passage-binary-mechanics-plan-v4"] = BINARY_SMOKE_PLAN_VERSION
    status: Literal[
        "frozen_post_hoc_source_visible_binary_mechanics_plan_before_provider_liability"
    ]
    config: MetaSynPassageBinaryMechanicsConfigV4
    config_sha256: Sha256
    config_file_sha256: Sha256
    v2_replay_snapshot_sha256: Sha256
    v2_execution_bundle_sha256: Sha256
    v2_inventory_ledger_sha256: Sha256
    v2_packet_roster_sha256: Sha256
    v2_failed_smoke_sha256: Sha256
    v2_provider_receipt_count: Literal[43]
    v2_failed_gate_preserved: Literal[True]
    feasibility_witnesses: Annotated[
        list[MetaSynBinaryFeasibilityWitnessV4], Field(min_length=2, max_length=2)
    ]
    feasibility_witness_membership_sha256: Sha256
    requests: Annotated[list[MetaSynBinaryMechanicsRequestV4], Field(min_length=2, max_length=2)]
    request_membership_sha256: Sha256
    request_count: Literal[2]
    pipeline_fingerprint: PipelineFingerprint
    pipeline_sha256: Sha256
    conservative_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    configured_cost_cap_usd_micros: Literal[1425336]
    all_candidates_previously_unattempted_in_v2: Literal[True]
    exact_cost_authorization_created: Literal[False]
    provider_calls_made: Literal[False]
    post_hoc_mechanics_yield_only: Literal[True]
    hidden_or_reference_labels_opened: Literal[False]
    official_test_labels_opened: Literal[False]
    extraction_accuracy_authority: Literal[False]
    synthesis_input_authority: Literal[False]
    claim_release_authority: Literal[False]
    plan_sha256: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> MetaSynPassageBinaryMechanicsPlanV4:
        if (
            self.config_sha256 != self.config.config_sha256
            or self.pipeline_sha256 != self.pipeline_fingerprint.pipeline_sha256
            or [(item.row_ordinal, item.candidate_index) for item in self.requests]
            != list(SELECTED_CANDIDATES)
            or [item.ordinal for item in self.requests] != [1, 2]
            or [(item.row_ordinal, item.candidate_index) for item in self.feasibility_witnesses]
            != list(SELECTED_CANDIDATES)
            or self.feasibility_witness_membership_sha256
            != hash_canonical([item.witness_sha256 for item in self.feasibility_witnesses])
            or self.request_membership_sha256
            != hash_canonical([item.request_receipt_sha256 for item in self.requests])
            or self.conservative_cost_ceiling_usd_micros
            != sum(item.request_cost_ceiling_usd_micros for item in self.requests)
            or self.conservative_cost_ceiling_usd_micros > self.configured_cost_cap_usd_micros
        ):
            raise ValueError("binary_mechanics_v4_plan_alias_mismatch")
        witness_by_key = {
            (item.row_ordinal, item.candidate_index): item for item in self.feasibility_witnesses
        }
        if any(
            item.feasibility_witness_sha256
            != witness_by_key[(item.row_ordinal, item.candidate_index)].witness_sha256
            for item in self.requests
        ):
            raise ValueError("binary_mechanics_v4_request_witness_mismatch")
        _validate_self_hash(self, "plan_sha256")
        return self


def _source_literal_outcome(
    *, candidate_index: int, candidate_binding_sha256: str
) -> PacketGroundingModelCompletedV2:
    title = (
        "Safety and Efficacy of Fedratinib in Patients With Primary or Secondary "
        "Myelofibrosis: A Randomized Clinical Trial."
    )
    if candidate_index == 2:
        evidence_quote = (
            "31 of 91 (34% [95% CI, 24%-44%]), and 6 of 85 (7% [95% CI, "
            "2%-13%]) in the fedratinib 400-mg, 500-mg, and placebo groups, "
            "respectively (P\u2009<\u2009.001)."
        )
        values = {
            "effect.control_events": "6",
            "effect.control_total": "85",
            "effect.treatment_events": "31",
            "effect.treatment_total": "91",
        }
        contrast = "500-mg, and placebo groups"
    elif candidate_index == 3:
        evidence_quote = (
            "39 of 97 (40% [95% CI, 30%-50%]) patients in the fedratinib "
            "400-mg and 500-mg groups, vs 1 of 96"
        )
        values = {
            "effect.control_events": "1",
            "effect.control_total": "96",
            "effect.treatment_events": "39",
            "effect.treatment_total": "97",
        }
        contrast = "groups, vs"
    else:  # pragma: no cover - closed caller roster
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_witness_candidate_unknown"
        )
    payload: dict[str, Any] = {
        "outcome_version": "native-packet-grounding-model-outcome-v2",
        "packet_status": "completed",
        "candidate_binding_sha256": candidate_binding_sha256,
        "evidence_quote": evidence_quote,
        "effect_format_token": None,
        "effect_unit": None,
        "numeric_claims": [
            {
                "field_path": field_path,
                "verbatim_numeric_token": token,
                "normalization": "identity",
            }
            for field_path, token in sorted(values.items())
        ],
        "identity_claims": [
            {
                "field_path": "cohort.source_label",
                "verbatim_identity_text": "NCT01437787",
            },
            {
                "field_path": "comparator_arm.label",
                "verbatim_identity_text": "placebo",
            },
            {
                "field_path": "contrast.label",
                "verbatim_identity_text": contrast,
            },
            {
                "field_path": "study.source_label",
                "verbatim_identity_text": title,
            },
            {
                "field_path": "treatment_arm.label",
                "verbatim_identity_text": "500-mg",
            },
        ],
        "timepoint": {
            "kind": "reported_text",
            "raw_label": "week 24",
            "anchor": None,
        },
    }
    return PacketGroundingModelCompletedV2.model_validate(payload)


def _freeze_witness(
    *,
    context: Any,
    source_request: PacketRequestV2,
) -> MetaSynBinaryFeasibilityWitnessV4:
    key = (source_request.row_ordinal, source_request.candidate_index)
    attempted = {
        (item.row_ordinal, item.candidate_index)
        for item in context.snapshot.attempted_packet_requests
    }
    if key in attempted or key not in SELECTED_CANDIDATES:
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_candidate_not_unattempted"
        )
    row = context.bundle.extraction_inputs.rows[source_request.row_ordinal]
    outcome = _source_literal_outcome(
        candidate_index=source_request.candidate_index,
        candidate_binding_sha256=(source_request.packet_input.candidate_binding_sha256),
    )
    grounding = freeze_passage_packet_grounding_receipt_v2(
        model_outcome=outcome.model_dump(mode="json"),
        candidate=source_request.packet_input.candidate,
        projection=row.projection_v2,
    )
    if not isinstance(grounding, PacketGroundingCompletedReceiptV2):
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_witness_grounding_not_completed"
        )
    protocol = replay_metasyn_question_projection_spec_v2(question_surface=row.question_surface)
    assembly = assemble_native_packet_v2(
        candidate=source_request.packet_input.candidate,
        projection=row.projection_v2,
        protocol=protocol,
        protocol_orientation=context.bundle.protocol_orientations[
            source_request.row_ordinal
        ].protocol_orientation,
        analysis_policy=context.bundle.assembly_analysis_policy,
        grounding_receipt=grounding,
    )
    if not isinstance(assembly, NativePacketAssemblyCompletedV2):
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_witness_assembly_not_completed"
        )
    payload: dict[str, Any] = {
        "witness_version": BINARY_SMOKE_WITNESS_VERSION,
        "row_ordinal": source_request.row_ordinal,
        "candidate_index": source_request.candidate_index,
        "source_packet_request_sha256": source_request.packet_request_sha256,
        "packet_input_sha256": source_request.packet_input_sha256,
        "candidate_descriptor_sha256": (source_request.packet_input.candidate_descriptor_sha256),
        "candidate_binding_sha256": (source_request.packet_input.candidate_binding_sha256),
        "v2_scientific_request_signature_sha256": (_scientific_request_signature(source_request)),
        "previously_attempted_in_immutable_v2": False,
        "full_source_literal_completed_model_outcome": outcome,
        "full_source_literal_completed_model_outcome_sha256": hash_canonical(outcome),
        "grounding_receipt": grounding,
        "grounding_receipt_sha256": grounding.receipt_sha256,
        "assembly_receipt": assembly,
        "assembly_receipt_sha256": assembly.assembly_receipt_sha256,
        "exact_source_literals_only": True,
        "model_or_provider_called_for_witness": False,
        "hidden_or_reference_labels_opened": False,
        "relaxed_parsing_used": False,
        "mechanics_feasibility_only": True,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynBinaryFeasibilityWitnessV4.model_validate(
        {**payload, "witness_sha256": hash_canonical(payload)}
    )


def _freeze_requests(
    *,
    context: Any,
    witnesses: Sequence[MetaSynBinaryFeasibilityWitnessV4],
) -> list[MetaSynBinaryMechanicsRequestV4]:
    source_by_key = {
        (item.row_ordinal, item.candidate_index): item for item in context.packet_roster.requests
    }
    witness_by_key = {(item.row_ordinal, item.candidate_index): item for item in witnesses}
    prior_request_keys = {item.request.request_key for item in context.packet_roster.requests} | {
        item.request_key for item in context.snapshot.attempted_packet_requests
    }
    attempted_signatures = {
        item.scientific_request_signature_sha256
        for item in context.snapshot.attempted_packet_requests
    }
    output: list[MetaSynBinaryMechanicsRequestV4] = []
    for ordinal, key in enumerate(SELECTED_CANDIDATES, start=1):
        source = source_by_key.get(key)
        witness = witness_by_key.get(key)
        if source is None or witness is None:
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_selected_candidate_missing"
            )
        signature = _scientific_request_signature(source)
        if signature in attempted_signatures:
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_scientific_request_previously_attempted"
            )
        request_key = (
            f"{_REQUEST_KEY_PREFIX}-row-{source.row_ordinal:02d}"
            f"-candidate-{source.candidate_index:02d}"
        )
        if request_key in prior_request_keys:
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_request_key_not_fresh"
            )
        request = freeze_anthropic_bounded_request(
            operation=_OPERATION,
            request_key=request_key,
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
            "request_version": BINARY_SMOKE_REQUEST_VERSION,
            "ordinal": ordinal,
            "row_ordinal": source.row_ordinal,
            "candidate_index": source.candidate_index,
            "source_packet_request": source,
            "source_packet_request_sha256": source.packet_request_sha256,
            "packet_input_sha256": source.packet_input_sha256,
            "candidate_binding_sha256": (source.packet_input.candidate_binding_sha256),
            "v2_scientific_request_signature_sha256": signature,
            "feasibility_witness_sha256": witness.witness_sha256,
            "request": request,
            "request_sha256": request.request_sha256,
            "request_cost_ceiling_usd_micros": _usd_micros(
                request.cost_ceiling.request_cost_ceiling_usd
            ),
            "fresh_operation_domain": True,
            "fresh_request_key_domain": True,
            "fresh_exact_once_intent_domain": True,
            "previously_attempted_in_immutable_v2": False,
        }
        output.append(
            MetaSynBinaryMechanicsRequestV4.model_validate(
                {
                    **payload,
                    "request_receipt_sha256": hash_canonical(payload),
                }
            )
        )
    return output


def _compute_pipeline_fingerprint(
    *, repository_root: Path, config_sha256: str, v2_snapshot_sha256: str
) -> PipelineFingerprint:
    files = sorted(set(_python_dependency_closure(repository_root)) | set(_NEW_FINGERPRINT_FILES))
    component = PipelineComponentSpec(
        component_id="metasyn-passage-binary-mechanics-smoke-v4",
        component_version="1",
        file_paths=files,
        settings={
            "application_retries_per_request": 0,
            "sdk_retries_per_request": 0,
            "maximum_calls": 2,
            "selected_candidates": ["17:2", "17:3"],
            "operation": _OPERATION,
            "request_key_prefix": _REQUEST_KEY_PREFIX,
            "phase": _PHASE,
            "compact_normalizer": "native-packet-compact-normalization-v3",
            "v2_grounding_unchanged": True,
            "v2_assembly_unchanged": True,
            "config_sha256": config_sha256,
            "v2_replay_snapshot_sha256": v2_snapshot_sha256,
            "post_hoc_mechanics_yield_only": True,
            "extraction_accuracy_authority": False,
            "synthesis_input_authority": False,
            "claim_release_authority": False,
        },
    )
    return compute_pipeline_fingerprint(root=repository_root, components=[component])


def freeze_metasyn_passage_binary_mechanics_plan_v4(
    *,
    repository_root: Path,
    v2_workspace: Path = DEFAULT_V2_WORKSPACE,
) -> MetaSynPassageBinaryMechanicsPlanV4:
    """Replay immutable v2 and freeze both witnesses and fresh requests."""

    root = _canonical_root(repository_root)
    config, config_file_sha256 = load_metasyn_passage_binary_mechanics_config_v4(
        repository_root=root
    )
    context = _replay_v2_base(
        repository_root=root,
        v2_workspace=v2_workspace,
    )
    snapshot = context.snapshot
    if (
        snapshot.execution_bundle_sha256 != EXPECTED_V2_EXECUTION_BUNDLE_SHA256
        or snapshot.inventory_ledger_sha256 != EXPECTED_V2_INVENTORY_LEDGER_SHA256
        or snapshot.packet_roster_sha256 != EXPECTED_V2_PACKET_ROSTER_SHA256
        or snapshot.failed_smoke_sha256 != EXPECTED_V2_FAILED_SMOKE_SHA256
        or snapshot.provider_receipt_count != EXPECTED_V2_PROVIDER_RECEIPT_COUNT
        or context.smoke.status != "failed_gate"
        or context.smoke.remaining_packet_calls_permitted
    ):
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_v2_replay_anchor_mismatch"
        )
    source_by_key = {
        (item.row_ordinal, item.candidate_index): item for item in context.packet_roster.requests
    }
    witnesses = [
        _freeze_witness(context=context, source_request=source_by_key[key])
        for key in SELECTED_CANDIDATES
    ]
    requests = _freeze_requests(context=context, witnesses=witnesses)
    cost_ceiling = sum(item.request_cost_ceiling_usd_micros for item in requests)
    if cost_ceiling > config.configured_cost_cap_usd_micros:
        raise MetaSynPassageBinaryMechanicsSmokeV4Error("binary_mechanics_v4_cost_cap_exceeded")
    pipeline = _compute_pipeline_fingerprint(
        repository_root=root,
        config_sha256=config.config_sha256,
        v2_snapshot_sha256=snapshot.snapshot_sha256,
    )
    payload: dict[str, Any] = {
        "plan_version": BINARY_SMOKE_PLAN_VERSION,
        "status": (
            "frozen_post_hoc_source_visible_binary_mechanics_plan_before_provider_liability"
        ),
        "config": config,
        "config_sha256": config.config_sha256,
        "config_file_sha256": config_file_sha256,
        "v2_replay_snapshot_sha256": snapshot.snapshot_sha256,
        "v2_execution_bundle_sha256": snapshot.execution_bundle_sha256,
        "v2_inventory_ledger_sha256": snapshot.inventory_ledger_sha256,
        "v2_packet_roster_sha256": snapshot.packet_roster_sha256,
        "v2_failed_smoke_sha256": snapshot.failed_smoke_sha256,
        "v2_provider_receipt_count": snapshot.provider_receipt_count,
        "v2_failed_gate_preserved": True,
        "feasibility_witnesses": witnesses,
        "feasibility_witness_membership_sha256": hash_canonical(
            [item.witness_sha256 for item in witnesses]
        ),
        "requests": requests,
        "request_membership_sha256": hash_canonical(
            [item.request_receipt_sha256 for item in requests]
        ),
        "request_count": len(requests),
        "pipeline_fingerprint": pipeline,
        "pipeline_sha256": pipeline.pipeline_sha256,
        "conservative_cost_ceiling_usd_micros": cost_ceiling,
        "configured_cost_cap_usd_micros": config.configured_cost_cap_usd_micros,
        "all_candidates_previously_unattempted_in_v2": True,
        "exact_cost_authorization_created": False,
        "provider_calls_made": False,
        "post_hoc_mechanics_yield_only": True,
        "hidden_or_reference_labels_opened": False,
        "official_test_labels_opened": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynPassageBinaryMechanicsPlanV4.model_validate(
        {**payload, "plan_sha256": hash_canonical(payload)}
    )


def validate_metasyn_passage_binary_mechanics_plan_v4(
    *,
    plan: MetaSynPassageBinaryMechanicsPlanV4 | Mapping[str, Any],
    repository_root: Path,
    v2_workspace: Path = DEFAULT_V2_WORKSPACE,
    external_replay: bool = True,
) -> MetaSynPassageBinaryMechanicsPlanV4:
    try:
        canonical = MetaSynPassageBinaryMechanicsPlanV4.model_validate(plan)
    except ValueError as exc:
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_plan_contract_invalid"
        ) from exc
    if external_replay:
        replayed = freeze_metasyn_passage_binary_mechanics_plan_v4(
            repository_root=repository_root,
            v2_workspace=v2_workspace,
        )
        if replayed != canonical:
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_plan_external_replay_mismatch"
            )
    return canonical


def _intents(plan: MetaSynPassageBinaryMechanicsPlanV4) -> list[HostedExactOnceIntentV1]:
    return [
        freeze_hosted_exact_once_intent(
            execution_bundle_sha256=plan.plan_sha256,
            phase=_PHASE,
            source_bearing=True,
            context_binding_sha256=item.request_receipt_sha256,
            request=item.request,
        )
        for item in plan.requests
    ]


class MetaSynBinaryMechanicsAuthorizationV4(_FrozenExactModel):
    authorization_version: Literal["metasyn-passage-binary-mechanics-authorization-v4"] = (
        BINARY_SMOKE_AUTHORIZATION_VERSION
    )
    status: Literal["exact_two_call_cost_authorization_persisted_before_liability"]
    plan_sha256: Sha256
    request_membership_sha256: Sha256
    exact_authorization: HostedExactOnceCostAuthorizationV1
    exact_authorization_sha256: Sha256
    authorized_call_count: Literal[2]
    conservative_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    configured_cost_cap_usd_micros: Literal[1425336]
    provider_calls_made_before_authorization: Literal[0]
    application_retries_per_request: Literal[0]
    sdk_retries_per_request: Literal[0]
    ambiguity_is_terminal: Literal[True]
    authorization_receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_authorization(self) -> MetaSynBinaryMechanicsAuthorizationV4:
        if (
            self.exact_authorization_sha256 != self.exact_authorization.authorization_sha256
            or self.exact_authorization.execution_bundle_sha256 != self.plan_sha256
            or self.exact_authorization.phase != _PHASE
            or self.authorized_call_count != self.exact_authorization.authorized_call_count
            or self.conservative_cost_ceiling_usd_micros
            != self.exact_authorization.cost_ceiling_usd_micros
            or self.conservative_cost_ceiling_usd_micros > self.configured_cost_cap_usd_micros
        ):
            raise ValueError("binary_mechanics_v4_authorization_alias_mismatch")
        _validate_self_hash(self, "authorization_receipt_sha256")
        return self


def _freeze_authorization(
    plan: MetaSynPassageBinaryMechanicsPlanV4,
) -> MetaSynBinaryMechanicsAuthorizationV4:
    exact = freeze_hosted_exact_once_cost_authorization(
        execution_bundle_sha256=plan.plan_sha256,
        phase=_PHASE,
        intents=_intents(plan),
        configured_phase_budget_usd_micros=plan.configured_cost_cap_usd_micros,
    )
    payload: dict[str, Any] = {
        "authorization_version": BINARY_SMOKE_AUTHORIZATION_VERSION,
        "status": "exact_two_call_cost_authorization_persisted_before_liability",
        "plan_sha256": plan.plan_sha256,
        "request_membership_sha256": plan.request_membership_sha256,
        "exact_authorization": exact,
        "exact_authorization_sha256": exact.authorization_sha256,
        "authorized_call_count": exact.authorized_call_count,
        "conservative_cost_ceiling_usd_micros": exact.cost_ceiling_usd_micros,
        "configured_cost_cap_usd_micros": plan.configured_cost_cap_usd_micros,
        "provider_calls_made_before_authorization": 0,
        "application_retries_per_request": 0,
        "sdk_retries_per_request": 0,
        "ambiguity_is_terminal": True,
    }
    return MetaSynBinaryMechanicsAuthorizationV4.model_validate(
        {
            **payload,
            "authorization_receipt_sha256": hash_canonical(payload),
        }
    )


BinaryTerminalV4 = HostedExactOnceProviderReceiptV1 | HostedExactOnceAmbiguityIncidentV1
_TERMINAL_ADAPTER = TypeAdapter(BinaryTerminalV4)
_GROUNDING_ADAPTER = TypeAdapter(PacketGroundingReceiptV2)
_ASSEMBLY_ADAPTER = TypeAdapter(NativePacketAssemblyOutcomeV2)

BinaryValidationStatusV4 = Literal[
    "typed_effect_completed",
    "grounding_abstained",
    "assembly_abstained",
    "grounding_invalid",
    "assembly_invalid",
    "provider_runtime_failure",
    "exact_once_terminal_incident",
]


class MetaSynBinaryMechanicsResultV4(_FrozenExactModel):
    result_version: Literal["metasyn-passage-binary-mechanics-result-v4"] = (
        BINARY_SMOKE_RESULT_VERSION
    )
    plan_sha256: Sha256
    request_receipt_sha256: Sha256
    request_key: str
    row_ordinal: Literal[17]
    candidate_index: Literal[2, 3]
    packet_input: MetaSynPacketCandidateInputV2
    packet_input_sha256: Sha256
    terminal: BinaryTerminalV4
    terminal_sha256: Sha256
    validation_status: BinaryValidationStatusV4
    compact_normalization_receipt: NativePacketCompactNormalizationReceiptV3 | None
    compact_normalization_receipt_sha256: Sha256 | None
    grounding_receipt: PacketGroundingReceiptV2 | None
    grounding_receipt_sha256: Sha256 | None
    assembly_receipt: NativePacketAssemblyOutcomeV2 | None
    assembly_receipt_sha256: Sha256 | None
    authorizes_typed_effect: bool
    standard_packet_input_grounding_assembly_triple_persisted: bool
    v2_failed_gate_preserved: Literal[True]
    post_hoc_mechanics_yield_only: Literal[True]
    extraction_accuracy_authority: Literal[False]
    synthesis_input_authority: Literal[False]
    claim_release_authority: Literal[False]
    result_sha256: Sha256

    @model_validator(mode="after")
    def validate_result(self) -> MetaSynBinaryMechanicsResultV4:
        terminal_hash = (
            self.terminal.receipt_sha256
            if isinstance(self.terminal, HostedExactOnceProviderReceiptV1)
            else self.terminal.incident_sha256
        )
        expected_completed = self.validation_status == "typed_effect_completed"
        if (
            self.packet_input_sha256 != self.packet_input.packet_input_sha256
            or self.packet_input.row_ordinal != self.row_ordinal
            or self.packet_input.candidate.candidate_index != self.candidate_index
            or self.terminal_sha256 != terminal_hash
            or self.terminal.execution_bundle_sha256 != self.plan_sha256
            or self.terminal.request_key != self.request_key
            or self.terminal.context_binding_sha256 != self.request_receipt_sha256
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
            or self.authorizes_typed_effect != expected_completed
            or self.standard_packet_input_grounding_assembly_triple_persisted
            != (self.grounding_receipt is not None and self.assembly_receipt is not None)
        ):
            raise ValueError("binary_mechanics_v4_result_alias_mismatch")
        _validate_self_hash(self, "result_sha256")
        return self


def _terminal_sha256(outcome: BinaryTerminalV4) -> str:
    return (
        outcome.receipt_sha256
        if isinstance(outcome, HostedExactOnceProviderReceiptV1)
        else outcome.incident_sha256
    )


def _process_outcome(
    *,
    plan: MetaSynPassageBinaryMechanicsPlanV4,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    binary_request: MetaSynBinaryMechanicsRequestV4,
    outcome: BinaryTerminalV4,
) -> MetaSynBinaryMechanicsResultV4:
    normalization: NativePacketCompactNormalizationReceiptV3 | None = None
    grounding: PacketGroundingReceiptV2 | None = None
    assembly: NativePacketAssemblyOutcomeV2 | None = None
    if isinstance(outcome, HostedExactOnceAmbiguityIncidentV1):
        status: BinaryValidationStatusV4 = "exact_once_terminal_incident"
    elif outcome.provider_result.outcome != "completed":
        status = "provider_runtime_failure"
    elif not isinstance(outcome.provider_result.parsed_json, Mapping):
        status = "grounding_invalid"
    else:
        try:
            normalization = freeze_native_packet_compact_normalization_receipt_v3(
                raw_model_outcome=outcome.provider_result.parsed_json,
                expected_candidate_binding_sha256=(binary_request.candidate_binding_sha256),
            )
            row = bundle.extraction_inputs.rows[binary_request.row_ordinal]
            grounding = freeze_passage_packet_grounding_receipt_v2(
                model_outcome=normalization.normalized_model_outcome,
                candidate=binary_request.source_packet_request.packet_input.candidate,
                projection=row.projection_v2,
            )
        except (TypeError, ValueError):
            normalization = None
            grounding = None
            status = "grounding_invalid"
        else:
            try:
                protocol = replay_metasyn_question_projection_spec_v2(
                    question_surface=row.question_surface
                )
                assembly = assemble_native_packet_v2(
                    candidate=(binary_request.source_packet_request.packet_input.candidate),
                    projection=row.projection_v2,
                    protocol=protocol,
                    protocol_orientation=bundle.protocol_orientations[
                        binary_request.row_ordinal
                    ].protocol_orientation,
                    analysis_policy=bundle.assembly_analysis_policy,
                    grounding_receipt=grounding,
                )
            except (TypeError, ValueError):
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
        "result_version": BINARY_SMOKE_RESULT_VERSION,
        "plan_sha256": plan.plan_sha256,
        "request_receipt_sha256": binary_request.request_receipt_sha256,
        "request_key": binary_request.request.request_key,
        "row_ordinal": binary_request.row_ordinal,
        "candidate_index": binary_request.candidate_index,
        "packet_input": binary_request.source_packet_request.packet_input,
        "packet_input_sha256": binary_request.packet_input_sha256,
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
        "v2_failed_gate_preserved": True,
        "post_hoc_mechanics_yield_only": True,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynBinaryMechanicsResultV4.model_validate(
        {**payload, "result_sha256": hash_canonical(payload)}
    )


class MetaSynBinaryMechanicsSmokeReceiptV4(_FrozenExactModel):
    smoke_version: Literal["metasyn-passage-binary-mechanics-smoke-v4"] = (
        BINARY_SMOKE_RECEIPT_VERSION
    )
    status: Literal["passed", "failed_gate", "terminal_ambiguous_poison"]
    plan_sha256: Sha256
    authorization_receipt_sha256: Sha256
    ordered_authorized_request_keys: Annotated[list[str], Field(min_length=2, max_length=2)]
    attempted_request_keys: Annotated[list[str], Field(min_length=1, max_length=2)]
    results: Annotated[list[MetaSynBinaryMechanicsResultV4], Field(min_length=1, max_length=2)]
    result_membership_sha256: Sha256
    attempted_call_count: Annotated[int, Field(ge=1, le=2)]
    typed_effect_count: Annotated[int, Field(ge=0, le=1)]
    valid_abstention_does_not_pass: Literal[True]
    stop_after_first_typed_effect: Literal[True]
    stopped_and_poisoned_on_ambiguity: bool
    application_retries_per_request: Literal[0]
    sdk_retries_per_request: Literal[0]
    remaining_calls_under_this_authorization_permitted: Literal[False]
    post_hoc_mechanics_yield_only: Literal[True]
    extraction_accuracy_authority: Literal[False]
    synthesis_input_authority: Literal[False]
    claim_release_authority: Literal[False]
    smoke_sha256: Sha256

    @model_validator(mode="after")
    def validate_smoke(self) -> MetaSynBinaryMechanicsSmokeReceiptV4:
        ambiguous = any(
            item.validation_status == "exact_once_terminal_incident" for item in self.results
        )
        typed = sum(item.authorizes_typed_effect for item in self.results)
        expected_status = (
            "terminal_ambiguous_poison" if ambiguous else "passed" if typed else "failed_gate"
        )
        if (
            self.attempted_request_keys != [item.request_key for item in self.results]
            or self.result_membership_sha256
            != hash_canonical([item.result_sha256 for item in self.results])
            or self.attempted_call_count != len(self.results)
            or self.typed_effect_count != typed
            or self.status != expected_status
            or self.stopped_and_poisoned_on_ambiguity != ambiguous
            or typed > 1
        ):
            raise ValueError("binary_mechanics_v4_smoke_alias_mismatch")
        _validate_self_hash(self, "smoke_sha256")
        return self


def _freeze_smoke(
    *,
    plan: MetaSynPassageBinaryMechanicsPlanV4,
    authorization: MetaSynBinaryMechanicsAuthorizationV4,
    results: Sequence[MetaSynBinaryMechanicsResultV4],
) -> MetaSynBinaryMechanicsSmokeReceiptV4:
    ambiguous = any(item.validation_status == "exact_once_terminal_incident" for item in results)
    typed = sum(item.authorizes_typed_effect for item in results)
    status = "terminal_ambiguous_poison" if ambiguous else "passed" if typed else "failed_gate"
    payload: dict[str, Any] = {
        "smoke_version": BINARY_SMOKE_RECEIPT_VERSION,
        "status": status,
        "plan_sha256": plan.plan_sha256,
        "authorization_receipt_sha256": (authorization.authorization_receipt_sha256),
        "ordered_authorized_request_keys": [item.request.request_key for item in plan.requests],
        "attempted_request_keys": [item.request_key for item in results],
        "results": list(results),
        "result_membership_sha256": hash_canonical([item.result_sha256 for item in results]),
        "attempted_call_count": len(results),
        "typed_effect_count": typed,
        "valid_abstention_does_not_pass": True,
        "stop_after_first_typed_effect": True,
        "stopped_and_poisoned_on_ambiguity": ambiguous,
        "application_retries_per_request": 0,
        "sdk_retries_per_request": 0,
        "remaining_calls_under_this_authorization_permitted": False,
        "post_hoc_mechanics_yield_only": True,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynBinaryMechanicsSmokeReceiptV4.model_validate(
        {**payload, "smoke_sha256": hash_canonical(payload)}
    )


class BinaryArtifactBindingV4(_FrozenExactModel):
    relative_path: str
    sha256: Sha256
    utf8_bytes: Annotated[int, Field(ge=1)]


BinaryStageV4 = Literal[
    "prepared",
    "authorized",
    "smoke_passed",
    "smoke_failed",
    "smoke_ambiguous_poison",
    "finalized",
    "externally_validated",
]
_STAGE_ORDINAL: dict[BinaryStageV4, int] = {
    "prepared": 0,
    "authorized": 1,
    "smoke_passed": 2,
    "smoke_failed": 2,
    "smoke_ambiguous_poison": 2,
    "finalized": 3,
    "externally_validated": 4,
}
_STAGE_FILENAMES: dict[int, tuple[str, ...]] = {
    0: ("00-prepared.json",),
    1: ("01-authorized.json",),
    2: (
        "02-smoke-passed.json",
        "02-smoke-failed.json",
        "02-smoke-ambiguous-poison.json",
    ),
    3: ("03-finalized.json",),
    4: ("04-externally-validated.json",),
}


class MetaSynBinaryMechanicsCheckpointV4(_FrozenExactModel):
    checkpoint_version: Literal["metasyn-passage-binary-mechanics-checkpoint-v4"] = (
        BINARY_SMOKE_CHECKPOINT_VERSION
    )
    plan_sha256: Sha256
    stage: BinaryStageV4
    stage_ordinal: Annotated[int, Field(ge=0, le=4)]
    previous_checkpoint_sha256: Sha256 | None
    artifacts: Annotated[list[BinaryArtifactBindingV4], Field(min_length=1)]
    artifact_membership_sha256: Sha256
    checkpoint_sha256: Sha256

    @model_validator(mode="after")
    def validate_checkpoint(self) -> MetaSynBinaryMechanicsCheckpointV4:
        if (
            self.stage_ordinal != _STAGE_ORDINAL[self.stage]
            or self.artifacts != sorted(self.artifacts, key=lambda item: item.relative_path)
            or len({item.relative_path for item in self.artifacts}) != len(self.artifacts)
            or self.artifact_membership_sha256
            != hash_canonical([item.model_dump(mode="json") for item in self.artifacts])
        ):
            raise ValueError("binary_mechanics_v4_checkpoint_alias_mismatch")
        _validate_self_hash(self, "checkpoint_sha256")
        return self


def _runtime_paths(workspace: Path) -> dict[str, Path]:
    return {
        "plan": workspace / "binary-mechanics-plan.json",
        "witnesses": workspace / "feasibility-witnesses.json",
        "authorization": workspace / "binary-mechanics-authorization.json",
        "provider": workspace / "provider-state",
        "packet_inputs": workspace / "packet-inputs",
        "normalization": workspace / "compact-normalization-receipts",
        "grounding": workspace / "grounding-receipts",
        "assembly": workspace / "assembly-receipts",
        "results": workspace / "results",
        "smoke": workspace / "binary-mechanics-smoke.json",
        "report": workspace / "final-report.json",
        "validation": workspace / "external-validation.json",
        "stages": workspace / "stage-checkpoints",
    }


def _create_fresh_workspace(value: Path) -> Path:
    path = Path(os.path.abspath(value))
    if path.exists() or path.is_symlink():
        raise MetaSynPassageBinaryMechanicsSmokeV4Error("binary_mechanics_v4_workspace_not_fresh")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_workspace_parent_missing"
        ) from exc
    if not parent.is_dir():
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_workspace_parent_invalid"
        )
    path.mkdir(mode=0o700)
    return path.resolve(strict=True)


def _canonical_workspace(value: Path) -> Path:
    if value.is_symlink():
        raise MetaSynPassageBinaryMechanicsSmokeV4Error("binary_mechanics_v4_workspace_symlink")
    try:
        path = value.resolve(strict=True)
    except OSError as exc:
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_workspace_missing"
        ) from exc
    if not path.is_dir():
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_workspace_not_directory"
        )
    return path


@contextmanager
def _runtime_lock(workspace: Path) -> Iterator[None]:
    lock_path = workspace / ".metasyn-passage-binary-mechanics-v4.lock"
    if lock_path.is_symlink():
        raise MetaSynPassageBinaryMechanicsSmokeV4Error("binary_mechanics_v4_lock_symlink")
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
        if _read_object(path) != value.model_dump(mode="json"):
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_artifact_replay_mismatch"
            )
        return
    atomic_write_json(path, value)


def _artifact_binding(workspace: Path, path: Path) -> BinaryArtifactBindingV4:
    if path.is_symlink() or not path.is_file():
        raise MetaSynPassageBinaryMechanicsSmokeV4Error("binary_mechanics_v4_stage_artifact_unsafe")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(workspace):
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_stage_artifact_outside_workspace"
        )
    raw = path.read_bytes()
    return BinaryArtifactBindingV4(
        relative_path=resolved.relative_to(workspace).as_posix(),
        sha256=sha256_file(path),
        utf8_bytes=len(raw),
    )


def _checkpoint_path(workspace: Path, stage: BinaryStageV4) -> Path:
    ordinal = _STAGE_ORDINAL[stage]
    if ordinal == 2:
        filename = {
            "smoke_passed": "02-smoke-passed.json",
            "smoke_failed": "02-smoke-failed.json",
            "smoke_ambiguous_poison": "02-smoke-ambiguous-poison.json",
        }[stage]
    else:
        filename = _STAGE_FILENAMES[ordinal][0]
    return _runtime_paths(workspace)["stages"] / filename


def _load_stage_chain(workspace: Path) -> list[MetaSynBinaryMechanicsCheckpointV4]:
    directory = _runtime_paths(workspace)["stages"]
    if directory.is_symlink() or not directory.is_dir():
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_stage_directory_unsafe"
        )
    actual = {path.name for path in directory.glob("*.json")}
    chain: list[MetaSynBinaryMechanicsCheckpointV4] = []
    for ordinal in range(5):
        matches = actual.intersection(_STAGE_FILENAMES[ordinal])
        if not matches:
            if any(actual.intersection(_STAGE_FILENAMES[later]) for later in range(ordinal + 1, 5)):
                raise MetaSynPassageBinaryMechanicsSmokeV4Error("binary_mechanics_v4_stage_gap")
            break
        if len(matches) != 1:
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_stage_branch_ambiguous"
            )
        checkpoint = MetaSynBinaryMechanicsCheckpointV4.model_validate(
            _read_object(directory / next(iter(matches)))
        )
        previous = chain[-1].checkpoint_sha256 if chain else None
        if checkpoint.stage_ordinal != ordinal or checkpoint.previous_checkpoint_sha256 != previous:
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_stage_chain_mismatch"
            )
        for binding in checkpoint.artifacts:
            if _artifact_binding(workspace, workspace / binding.relative_path) != binding:
                raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                    "binary_mechanics_v4_stage_artifact_tamper"
                )
        chain.append(checkpoint)
    allowed = {name for ordinal in range(len(chain)) for name in _STAGE_FILENAMES[ordinal]}
    if actual - allowed:
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_unexpected_stage_artifact"
        )
    return chain


def _write_checkpoint(
    *,
    workspace: Path,
    plan_sha256: str,
    stage: BinaryStageV4,
    artifacts: Sequence[Path],
) -> MetaSynBinaryMechanicsCheckpointV4:
    chain = _load_stage_chain(workspace)
    ordinal = _STAGE_ORDINAL[stage]
    if len(chain) != ordinal:
        raise MetaSynPassageBinaryMechanicsSmokeV4Error("binary_mechanics_v4_stage_advance_invalid")
    bindings = sorted(
        (_artifact_binding(workspace, path) for path in artifacts),
        key=lambda item: item.relative_path,
    )
    payload: dict[str, Any] = {
        "checkpoint_version": BINARY_SMOKE_CHECKPOINT_VERSION,
        "plan_sha256": plan_sha256,
        "stage": stage,
        "stage_ordinal": ordinal,
        "previous_checkpoint_sha256": (chain[-1].checkpoint_sha256 if chain else None),
        "artifacts": bindings,
        "artifact_membership_sha256": hash_canonical(
            [item.model_dump(mode="json") for item in bindings]
        ),
    }
    checkpoint = MetaSynBinaryMechanicsCheckpointV4.model_validate(
        {**payload, "checkpoint_sha256": hash_canonical(payload)}
    )
    atomic_write_json(_checkpoint_path(workspace, stage), checkpoint)
    return checkpoint


def _load_plan(
    *, repository_root: Path, workspace: Path, v2_workspace: Path
) -> MetaSynPassageBinaryMechanicsPlanV4:
    saved = MetaSynPassageBinaryMechanicsPlanV4.model_validate(
        _read_object(_runtime_paths(workspace)["plan"])
    )
    return validate_metasyn_passage_binary_mechanics_plan_v4(
        plan=saved,
        repository_root=repository_root,
        v2_workspace=v2_workspace,
        external_replay=True,
    )


class MetaSynBinaryMechanicsWitnessSetV4(_FrozenExactModel):
    witness_set_version: Literal["metasyn-passage-binary-mechanics-witness-set-v4"] = (
        "metasyn-passage-binary-mechanics-witness-set-v4"
    )
    plan_sha256: Sha256
    witnesses: Annotated[list[MetaSynBinaryFeasibilityWitnessV4], Field(min_length=2, max_length=2)]
    witness_membership_sha256: Sha256
    mechanics_feasibility_only: Literal[True]
    extraction_accuracy_authority: Literal[False]
    synthesis_input_authority: Literal[False]
    claim_release_authority: Literal[False]
    witness_set_sha256: Sha256

    @model_validator(mode="after")
    def validate_set(self) -> MetaSynBinaryMechanicsWitnessSetV4:
        if self.witness_membership_sha256 != hash_canonical(
            [item.witness_sha256 for item in self.witnesses]
        ):
            raise ValueError("binary_mechanics_v4_witness_set_membership_mismatch")
        _validate_self_hash(self, "witness_set_sha256")
        return self


def _freeze_witness_set(
    plan: MetaSynPassageBinaryMechanicsPlanV4,
) -> MetaSynBinaryMechanicsWitnessSetV4:
    payload: dict[str, Any] = {
        "witness_set_version": "metasyn-passage-binary-mechanics-witness-set-v4",
        "plan_sha256": plan.plan_sha256,
        "witnesses": plan.feasibility_witnesses,
        "witness_membership_sha256": plan.feasibility_witness_membership_sha256,
        "mechanics_feasibility_only": True,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynBinaryMechanicsWitnessSetV4.model_validate(
        {**payload, "witness_set_sha256": hash_canonical(payload)}
    )


def prepare_metasyn_passage_binary_mechanics_smoke_v4(
    *,
    repository_root: Path,
    workspace: Path = DEFAULT_BINARY_SMOKE_WORKSPACE,
    v2_workspace: Path = DEFAULT_V2_WORKSPACE,
) -> MetaSynPassageBinaryMechanicsPlanV4:
    root = _canonical_root(repository_root)
    plan = freeze_metasyn_passage_binary_mechanics_plan_v4(
        repository_root=root,
        v2_workspace=v2_workspace,
    )
    ws = _create_fresh_workspace(workspace)
    with _runtime_lock(ws):
        paths = _runtime_paths(ws)
        for name in (
            "provider",
            "packet_inputs",
            "normalization",
            "grounding",
            "assembly",
            "results",
            "stages",
        ):
            paths[name].mkdir(mode=0o700)
        witness_set = _freeze_witness_set(plan)
        _write_or_replay(paths["plan"], plan)
        _write_or_replay(paths["witnesses"], witness_set)
        if (
            _load_plan(
                repository_root=root,
                workspace=ws,
                v2_workspace=v2_workspace,
            )
            != plan
        ):
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_written_plan_replay_mismatch"
            )
        _write_checkpoint(
            workspace=ws,
            plan_sha256=plan.plan_sha256,
            stage="prepared",
            artifacts=[paths["plan"], paths["witnesses"]],
        )
    return plan


def authorize_metasyn_passage_binary_mechanics_smoke_v4(
    *,
    repository_root: Path,
    workspace: Path,
    v2_workspace: Path = DEFAULT_V2_WORKSPACE,
    expected_plan_sha256: str,
) -> MetaSynBinaryMechanicsAuthorizationV4:
    root = _canonical_root(repository_root)
    ws = _canonical_workspace(workspace)
    with _runtime_lock(ws):
        chain = _load_stage_chain(ws)
        if not chain or chain[0].plan_sha256 != expected_plan_sha256:
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_expected_plan_mismatch"
            )
        plan = _load_plan(repository_root=root, workspace=ws, v2_workspace=v2_workspace)
        paths = _runtime_paths(ws)
        if len(chain) > 1:
            return MetaSynBinaryMechanicsAuthorizationV4.model_validate(
                _read_object(paths["authorization"])
            )
        if list(paths["provider"].rglob("*.json")):
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_provider_state_precedes_authorization"
            )
        authorization = _freeze_authorization(plan)
        # Both the rich receipt and the exact executor's phase roster are durable
        # before any call intent can be written.
        _write_or_replay(paths["authorization"], authorization)
        provider_auth_dir = paths["provider"] / "cost-authorizations"
        provider_auth_dir.mkdir(mode=0o700)
        provider_auth_path = provider_auth_dir / f"{_PHASE}.json"
        atomic_write_json(provider_auth_path, authorization.exact_authorization)
        _write_checkpoint(
            workspace=ws,
            plan_sha256=plan.plan_sha256,
            stage="authorized",
            artifacts=[paths["authorization"], provider_auth_path],
        )
        return authorization


def _artifact_path(directory: Path, request_key: str) -> Path:
    return directory / f"{request_key}.json"


def _persist_result(*, workspace: Path, result: MetaSynBinaryMechanicsResultV4) -> list[Path]:
    paths = _runtime_paths(workspace)
    written: list[Path] = []
    packet_path = _artifact_path(paths["packet_inputs"], result.request_key)
    _write_or_replay(packet_path, result.packet_input)
    written.append(packet_path)
    if result.compact_normalization_receipt is not None:
        normalization_path = _artifact_path(paths["normalization"], result.request_key)
        _write_or_replay(normalization_path, result.compact_normalization_receipt)
        written.append(normalization_path)
    if result.grounding_receipt is not None:
        grounding_path = _artifact_path(paths["grounding"], result.request_key)
        _write_or_replay(grounding_path, result.grounding_receipt)
        written.append(grounding_path)
    if result.assembly_receipt is not None:
        assembly_path = _artifact_path(paths["assembly"], result.request_key)
        _write_or_replay(assembly_path, result.assembly_receipt)
        written.append(assembly_path)
    result_path = _artifact_path(paths["results"], result.request_key)
    _write_or_replay(result_path, result)
    written.append(result_path)
    return written


def run_metasyn_passage_binary_mechanics_smoke_v4(
    *,
    repository_root: Path,
    workspace: Path,
    v2_workspace: Path = DEFAULT_V2_WORKSPACE,
    expected_plan_sha256: str,
    client: HostedClientProtocol,
) -> MetaSynBinaryMechanicsSmokeReceiptV4:
    root = _canonical_root(repository_root)
    ws = _canonical_workspace(workspace)
    with _runtime_lock(ws):
        chain = _load_stage_chain(ws)
        if not chain or chain[0].plan_sha256 != expected_plan_sha256:
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_expected_plan_mismatch"
            )
        paths = _runtime_paths(ws)
        if len(chain) < 2:
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_smoke_requires_authorization"
            )
        if len(chain) > 2:
            return MetaSynBinaryMechanicsSmokeReceiptV4.model_validate(_read_object(paths["smoke"]))
        plan = _load_plan(repository_root=root, workspace=ws, v2_workspace=v2_workspace)
        authorization = MetaSynBinaryMechanicsAuthorizationV4.model_validate(
            _read_object(paths["authorization"])
        )
        if authorization != _freeze_authorization(plan):
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_authorization_external_replay_mismatch"
            )
        context = _replay_v2_base(
            repository_root=root,
            v2_workspace=v2_workspace,
        )
        intents = _intents(plan)
        results: list[MetaSynBinaryMechanicsResultV4] = []
        result_artifacts: list[Path] = []
        for binary_request, intent in zip(plan.requests, intents, strict=True):
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
            result = _process_outcome(
                plan=plan,
                bundle=context.bundle,
                binary_request=binary_request,
                outcome=outcome,
            )
            result_artifacts.extend(_persist_result(workspace=ws, result=result))
            results.append(result)
            if isinstance(outcome, HostedExactOnceAmbiguityIncidentV1):
                break
            if result.authorizes_typed_effect:
                break
        smoke = _freeze_smoke(
            plan=plan,
            authorization=authorization,
            results=results,
        )
        _write_or_replay(paths["smoke"], smoke)
        stage: BinaryStageV4 = {
            "passed": "smoke_passed",
            "failed_gate": "smoke_failed",
            "terminal_ambiguous_poison": "smoke_ambiguous_poison",
        }[smoke.status]
        _write_checkpoint(
            workspace=ws,
            plan_sha256=plan.plan_sha256,
            stage=stage,
            artifacts=[paths["smoke"], *result_artifacts],
        )
        return smoke


def _validate_saved_result(
    *,
    workspace: Path,
    plan: MetaSynPassageBinaryMechanicsPlanV4,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    binary_request: MetaSynBinaryMechanicsRequestV4,
    outcome: BinaryTerminalV4,
) -> MetaSynBinaryMechanicsResultV4:
    paths = _runtime_paths(workspace)
    saved = MetaSynBinaryMechanicsResultV4.model_validate(
        _read_object(_artifact_path(paths["results"], binary_request.request.request_key))
    )
    expected = _process_outcome(
        plan=plan,
        bundle=bundle,
        binary_request=binary_request,
        outcome=outcome,
    )
    if saved != expected:
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_result_external_replay_mismatch"
        )
    packet_input = MetaSynPacketCandidateInputV2.model_validate(
        _read_object(_artifact_path(paths["packet_inputs"], saved.request_key))
    )
    if packet_input != saved.packet_input:
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_packet_input_external_replay_mismatch"
        )
    if saved.compact_normalization_receipt is not None:
        if not isinstance(outcome, HostedExactOnceProviderReceiptV1) or not isinstance(
            outcome.provider_result.parsed_json, Mapping
        ):
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_normalization_terminal_shape_invalid"
            )
        normalization = NativePacketCompactNormalizationReceiptV3.model_validate(
            _read_object(_artifact_path(paths["normalization"], saved.request_key))
        )
        normalization = validate_native_packet_compact_normalization_receipt_v3(
            receipt=normalization,
            raw_model_outcome=outcome.provider_result.parsed_json,
            expected_candidate_binding_sha256=binary_request.candidate_binding_sha256,
        )
        if normalization != saved.compact_normalization_receipt:
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_normalization_external_replay_mismatch"
            )
    if saved.grounding_receipt is not None:
        grounding = _GROUNDING_ADAPTER.validate_python(
            _read_object(_artifact_path(paths["grounding"], saved.request_key))
        )
        row = bundle.extraction_inputs.rows[saved.row_ordinal]
        grounding = validate_passage_packet_grounding_receipt_v2(
            receipt=grounding,
            model_outcome=(
                saved.compact_normalization_receipt.normalized_model_outcome
                if saved.compact_normalization_receipt
                else {}
            ),
            candidate=binary_request.source_packet_request.packet_input.candidate,
            projection=row.projection_v2,
        )
        if grounding != saved.grounding_receipt:
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_grounding_external_replay_mismatch"
            )
    if saved.assembly_receipt is not None:
        assembly = _ASSEMBLY_ADAPTER.validate_python(
            _read_object(_artifact_path(paths["assembly"], saved.request_key))
        )
        row = bundle.extraction_inputs.rows[saved.row_ordinal]
        protocol = replay_metasyn_question_projection_spec_v2(question_surface=row.question_surface)
        assembly = validate_native_packet_assembly_v2(
            assembly=assembly,
            candidate=binary_request.source_packet_request.packet_input.candidate,
            projection=row.projection_v2,
            protocol=protocol,
            protocol_orientation=bundle.protocol_orientations[
                saved.row_ordinal
            ].protocol_orientation,
            analysis_policy=bundle.assembly_analysis_policy,
            grounding_receipt=saved.grounding_receipt,
        )
        if assembly != saved.assembly_receipt:
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_assembly_external_replay_mismatch"
            )
    return saved


class MetaSynBinaryMechanicsFinalReportV4(_FrozenExactModel):
    report_version: Literal["metasyn-passage-binary-mechanics-report-v4"] = (
        BINARY_SMOKE_REPORT_VERSION
    )
    status: Literal[
        "complete_post_hoc_binary_mechanics_yield_report_no_accuracy_or_release_authority"
    ]
    plan_sha256: Sha256
    pipeline_sha256: Sha256
    authorization_receipt_sha256: Sha256
    smoke_sha256: Sha256
    smoke_status: Literal["passed", "failed_gate", "terminal_ambiguous_poison"]
    authorized_call_count: Literal[2]
    attempted_call_count: Annotated[int, Field(ge=1, le=2)]
    typed_effect_count: Annotated[int, Field(ge=0, le=1)]
    completed_standard_triple_count: Annotated[int, Field(ge=0, le=2)]
    ambiguity_incident_count: Annotated[int, Field(ge=0, le=1)]
    attempted_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    authorized_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    remaining_authorized_calls_permitted: Literal[False]
    v2_failed_gate_preserved: Literal[True]
    post_hoc_mechanics_yield_only: Literal[True]
    extraction_accuracy_authority: Literal[False]
    synthesis_input_authority: Literal[False]
    claim_release_authority: Literal[False]
    report_sha256: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> MetaSynBinaryMechanicsFinalReportV4:
        if self.attempted_cost_ceiling_usd_micros > self.authorized_cost_ceiling_usd_micros:
            raise ValueError("binary_mechanics_v4_report_cost_alias_mismatch")
        _validate_self_hash(self, "report_sha256")
        return self


def _freeze_report(
    *,
    plan: MetaSynPassageBinaryMechanicsPlanV4,
    authorization: MetaSynBinaryMechanicsAuthorizationV4,
    smoke: MetaSynBinaryMechanicsSmokeReceiptV4,
) -> MetaSynBinaryMechanicsFinalReportV4:
    ceiling_by_key = {
        item.request.request_key: item.request_cost_ceiling_usd_micros for item in plan.requests
    }
    payload: dict[str, Any] = {
        "report_version": BINARY_SMOKE_REPORT_VERSION,
        "status": (
            "complete_post_hoc_binary_mechanics_yield_report_no_accuracy_or_release_authority"
        ),
        "plan_sha256": plan.plan_sha256,
        "pipeline_sha256": plan.pipeline_sha256,
        "authorization_receipt_sha256": authorization.authorization_receipt_sha256,
        "smoke_sha256": smoke.smoke_sha256,
        "smoke_status": smoke.status,
        "authorized_call_count": authorization.authorized_call_count,
        "attempted_call_count": smoke.attempted_call_count,
        "typed_effect_count": smoke.typed_effect_count,
        "completed_standard_triple_count": sum(
            item.standard_packet_input_grounding_assembly_triple_persisted for item in smoke.results
        ),
        "ambiguity_incident_count": sum(
            item.validation_status == "exact_once_terminal_incident" for item in smoke.results
        ),
        "attempted_cost_ceiling_usd_micros": sum(
            ceiling_by_key[key] for key in smoke.attempted_request_keys
        ),
        "authorized_cost_ceiling_usd_micros": (authorization.conservative_cost_ceiling_usd_micros),
        "remaining_authorized_calls_permitted": False,
        "v2_failed_gate_preserved": True,
        "post_hoc_mechanics_yield_only": True,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynBinaryMechanicsFinalReportV4.model_validate(
        {**payload, "report_sha256": hash_canonical(payload)}
    )


def _replay_runtime_results(
    *,
    repository_root: Path,
    workspace: Path,
    v2_workspace: Path,
    plan: MetaSynPassageBinaryMechanicsPlanV4,
    authorization: MetaSynBinaryMechanicsAuthorizationV4,
    smoke: MetaSynBinaryMechanicsSmokeReceiptV4,
) -> list[MetaSynBinaryMechanicsResultV4]:
    context = _replay_v2_base(
        repository_root=repository_root,
        v2_workspace=v2_workspace,
    )
    request_by_key = {item.request.request_key: item for item in plan.requests}
    intent_by_key = {item.request_key: item for item in _intents(plan)}
    replayed: list[MetaSynBinaryMechanicsResultV4] = []
    for request_key in smoke.attempted_request_keys:
        binary_request = request_by_key[request_key]
        intent = intent_by_key[request_key]
        outcome = validate_hosted_exact_once_outcome(
            workspace=_runtime_paths(workspace)["provider"],
            intent=intent,
            authorization=authorization.exact_authorization,
        )
        replayed.append(
            _validate_saved_result(
                workspace=workspace,
                plan=plan,
                bundle=context.bundle,
                binary_request=binary_request,
                outcome=outcome,
            )
        )
    if replayed != smoke.results:
        raise MetaSynPassageBinaryMechanicsSmokeV4Error(
            "binary_mechanics_v4_smoke_result_external_replay_mismatch"
        )
    return replayed


def finalize_metasyn_passage_binary_mechanics_smoke_v4(
    *,
    repository_root: Path,
    workspace: Path,
    v2_workspace: Path = DEFAULT_V2_WORKSPACE,
    expected_plan_sha256: str,
) -> MetaSynBinaryMechanicsFinalReportV4:
    root = _canonical_root(repository_root)
    ws = _canonical_workspace(workspace)
    with _runtime_lock(ws):
        chain = _load_stage_chain(ws)
        if len(chain) < 3 or chain[0].plan_sha256 != expected_plan_sha256:
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_finalize_requires_terminal_smoke"
            )
        paths = _runtime_paths(ws)
        if len(chain) > 3:
            return MetaSynBinaryMechanicsFinalReportV4.model_validate(_read_object(paths["report"]))
        plan = _load_plan(repository_root=root, workspace=ws, v2_workspace=v2_workspace)
        authorization = MetaSynBinaryMechanicsAuthorizationV4.model_validate(
            _read_object(paths["authorization"])
        )
        smoke = MetaSynBinaryMechanicsSmokeReceiptV4.model_validate(_read_object(paths["smoke"]))
        _replay_runtime_results(
            repository_root=root,
            workspace=ws,
            v2_workspace=v2_workspace,
            plan=plan,
            authorization=authorization,
            smoke=smoke,
        )
        report = _freeze_report(
            plan=plan,
            authorization=authorization,
            smoke=smoke,
        )
        _write_or_replay(paths["report"], report)
        _write_checkpoint(
            workspace=ws,
            plan_sha256=plan.plan_sha256,
            stage="finalized",
            artifacts=[paths["report"]],
        )
        return report


class MetaSynBinaryMechanicsValidationV4(_FrozenExactModel):
    validation_version: Literal["metasyn-passage-binary-mechanics-validation-v4"] = (
        BINARY_SMOKE_VALIDATION_VERSION
    )
    status: Literal["externally_replayed_complete"]
    plan_sha256: Sha256
    pipeline_sha256: Sha256
    authorization_receipt_sha256: Sha256
    smoke_sha256: Sha256
    report_sha256: Sha256
    validated_result_sha256s: Annotated[list[Sha256], Field(min_length=1, max_length=2)]
    immutable_v2_replayed_again: Literal[True]
    provider_calls_made_by_validation: Literal[0]
    all_saved_triples_replayed: Literal[True]
    extraction_accuracy_authority: Literal[False]
    synthesis_input_authority: Literal[False]
    claim_release_authority: Literal[False]
    validation_sha256: Sha256

    @model_validator(mode="after")
    def validate_validation(self) -> MetaSynBinaryMechanicsValidationV4:
        _validate_self_hash(self, "validation_sha256")
        return self


def validate_finalized_metasyn_passage_binary_mechanics_smoke_v4(
    *,
    repository_root: Path,
    workspace: Path,
    v2_workspace: Path = DEFAULT_V2_WORKSPACE,
    expected_plan_sha256: str,
) -> MetaSynBinaryMechanicsValidationV4:
    root = _canonical_root(repository_root)
    ws = _canonical_workspace(workspace)
    with _runtime_lock(ws):
        chain = _load_stage_chain(ws)
        if len(chain) < 4 or chain[0].plan_sha256 != expected_plan_sha256:
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_validate_requires_finalized"
            )
        paths = _runtime_paths(ws)
        if len(chain) > 4:
            return MetaSynBinaryMechanicsValidationV4.model_validate(
                _read_object(paths["validation"])
            )
        plan = _load_plan(repository_root=root, workspace=ws, v2_workspace=v2_workspace)
        authorization = MetaSynBinaryMechanicsAuthorizationV4.model_validate(
            _read_object(paths["authorization"])
        )
        if authorization != _freeze_authorization(plan):
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_validation_authorization_mismatch"
            )
        smoke = MetaSynBinaryMechanicsSmokeReceiptV4.model_validate(_read_object(paths["smoke"]))
        results = _replay_runtime_results(
            repository_root=root,
            workspace=ws,
            v2_workspace=v2_workspace,
            plan=plan,
            authorization=authorization,
            smoke=smoke,
        )
        if smoke != _freeze_smoke(plan=plan, authorization=authorization, results=results):
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_validation_smoke_mismatch"
            )
        report = MetaSynBinaryMechanicsFinalReportV4.model_validate(_read_object(paths["report"]))
        if report != _freeze_report(plan=plan, authorization=authorization, smoke=smoke):
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_validation_report_mismatch"
            )
        payload: dict[str, Any] = {
            "validation_version": BINARY_SMOKE_VALIDATION_VERSION,
            "status": "externally_replayed_complete",
            "plan_sha256": plan.plan_sha256,
            "pipeline_sha256": plan.pipeline_sha256,
            "authorization_receipt_sha256": (authorization.authorization_receipt_sha256),
            "smoke_sha256": smoke.smoke_sha256,
            "report_sha256": report.report_sha256,
            "validated_result_sha256s": [item.result_sha256 for item in results],
            "immutable_v2_replayed_again": True,
            "provider_calls_made_by_validation": 0,
            "all_saved_triples_replayed": True,
            "extraction_accuracy_authority": False,
            "synthesis_input_authority": False,
            "claim_release_authority": False,
        }
        validation = MetaSynBinaryMechanicsValidationV4.model_validate(
            {**payload, "validation_sha256": hash_canonical(payload)}
        )
        _write_or_replay(paths["validation"], validation)
        _write_checkpoint(
            workspace=ws,
            plan_sha256=plan.plan_sha256,
            stage="externally_validated",
            artifacts=[paths["validation"]],
        )
        return validation


def metasyn_passage_binary_mechanics_smoke_status_v4(
    *,
    repository_root: Path,
    workspace: Path,
    v2_workspace: Path = DEFAULT_V2_WORKSPACE,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    root = _canonical_root(repository_root)
    ws = _canonical_workspace(workspace)
    with _runtime_lock(ws):
        chain = _load_stage_chain(ws)
        if not chain or chain[0].plan_sha256 != expected_plan_sha256:
            raise MetaSynPassageBinaryMechanicsSmokeV4Error(
                "binary_mechanics_v4_expected_plan_mismatch"
            )
        plan = _load_plan(repository_root=root, workspace=ws, v2_workspace=v2_workspace)
        paths = _runtime_paths(ws)
        return {
            "status_version": "metasyn-passage-binary-mechanics-status-v4",
            "plan_sha256": plan.plan_sha256,
            "pipeline_sha256": plan.pipeline_sha256,
            "current_stage": chain[-1].stage,
            "stage_ordinal": chain[-1].stage_ordinal,
            "authorization_persisted": paths["authorization"].is_file(),
            "smoke_terminal": paths["smoke"].is_file(),
            "finalized": paths["report"].is_file(),
            "externally_validated": paths["validation"].is_file(),
            "maximum_provider_calls": MAXIMUM_BINARY_SMOKE_CALLS,
            "post_hoc_mechanics_yield_only": True,
            "extraction_accuracy_authority": False,
            "synthesis_input_authority": False,
            "claim_release_authority": False,
        }


__all__ = [
    "DEFAULT_BINARY_SMOKE_WORKSPACE",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_V2_WORKSPACE",
    "MAXIMUM_BINARY_SMOKE_CALLS",
    "MetaSynBinaryFeasibilityWitnessV4",
    "MetaSynBinaryMechanicsAuthorizationV4",
    "MetaSynBinaryMechanicsFinalReportV4",
    "MetaSynBinaryMechanicsResultV4",
    "MetaSynBinaryMechanicsSmokeReceiptV4",
    "MetaSynBinaryMechanicsValidationV4",
    "MetaSynPassageBinaryMechanicsPlanV4",
    "MetaSynPassageBinaryMechanicsSmokeV4Error",
    "authorize_metasyn_passage_binary_mechanics_smoke_v4",
    "finalize_metasyn_passage_binary_mechanics_smoke_v4",
    "freeze_metasyn_passage_binary_mechanics_plan_v4",
    "metasyn_passage_binary_mechanics_smoke_status_v4",
    "prepare_metasyn_passage_binary_mechanics_smoke_v4",
    "run_metasyn_passage_binary_mechanics_smoke_v4",
    "validate_finalized_metasyn_passage_binary_mechanics_smoke_v4",
    "validate_metasyn_passage_binary_mechanics_plan_v4",
]
