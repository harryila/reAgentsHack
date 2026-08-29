"""Fresh, staged, exactly-once runtime for passage-hosted MetaSyn extraction.

The offline execution bundle is the only spend authority.  This module materializes
that bundle in a new private workspace, authorizes each exact call roster before its
first possible provider liability, and then executes the two source-bearing passes:
all 32 value-free inventories followed by the dynamically authorized packet roster.

The runtime is deliberately yield-only.  Provider completion is never treated as
scientific completion: inventory values are replayed against their exact question and
passage surfaces, packet values are grounded locally, and typed effects are assembled
under the frozen analysis/orientation policies.  Runtime failures (especially
``max_tokens``) never become scientific abstentions.  Every durable provider attempt
is handled by :mod:`literature_multiverse.hosted_exact_once`; an orphan or ambiguous
attempt is terminal and cannot be retried.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from decimal import ROUND_CEILING, Decimal
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Protocol

from pydantic import ConfigDict, Field, TypeAdapter, field_validator, model_validator

from literature_multiverse.anthropic_bounded_generation import (
    AnthropicBoundedRequestV1,
    AnthropicBoundedResultV1,
    AnthropicCompiledSchemaV1,
    compile_anthropic_bounded_schema,
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
    MetaSynCandidateInventoryReceiptV2,
    freeze_metasyn_candidate_inventory_receipt_v2,
    validate_metasyn_candidate_inventory_receipt_v2,
)
from literature_multiverse.metasyn_extraction_inputs_v2 import (
    MetaSynPacketCandidateInputV2,
    freeze_metasyn_packet_candidate_input_v2,
    validate_metasyn_packet_candidate_input_v2,
)
from literature_multiverse.metasyn_passage_hosted_bundle_v2 import (
    DEFAULT_CONFIG_PATH,
    MetaSynPassageHostedExecutionBundleV2,
    capacity_limited_packet_schema_v2,
    freeze_metasyn_passage_hosted_execution_bundle_v2,
    validate_metasyn_passage_hosted_execution_bundle_v2,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_packet_assembly_v2 import (
    NativePacketAssemblyCompletedV2,
    NativePacketAssemblyOutcomeV2,
    assemble_native_packet_v2,
    replay_metasyn_question_projection_spec_v2,
    validate_native_packet_assembly_v2,
)
from literature_multiverse.native_packet_grounding_v2 import (
    NativePacketGroundingV2Error,
    PacketGroundingAbstentionReceiptV2,
    PacketGroundingReceiptV2,
    freeze_passage_packet_grounding_receipt_v2,
    validate_passage_packet_grounding_receipt_v2,
)
from literature_multiverse.native_question_projection import (
    QuestionProjectionSpecV1,
)

RUNTIME_VERSION = "metasyn-passage-hosted-runtime-v2"
STAGE_CHECKPOINT_VERSION = "metasyn-passage-hosted-stage-checkpoint-v2"
GLOBAL_AUTHORIZATION_VERSION = "metasyn-passage-hosted-global-cost-authorization-v2"
PREFLIGHT_RECEIPT_VERSION = "metasyn-passage-hosted-preflight-receipt-v2"
INVENTORY_RESULT_VERSION = "metasyn-passage-hosted-inventory-result-v2"
INVENTORY_LEDGER_VERSION = "metasyn-passage-hosted-inventory-ledger-v2"
PACKET_REQUEST_VERSION = "metasyn-passage-hosted-packet-request-v2"
PACKET_ROSTER_VERSION = "metasyn-passage-hosted-packet-roster-v2"
PACKET_RESULT_VERSION = "metasyn-passage-hosted-packet-result-v2"
PACKET_SMOKE_VERSION = "metasyn-passage-hosted-packet-smoke-v2"
PACKET_LEDGER_VERSION = "metasyn-passage-hosted-packet-ledger-v2"
FINAL_REPORT_VERSION = "metasyn-passage-hosted-private-yield-report-v2"
EXTERNAL_VALIDATION_VERSION = "metasyn-passage-hosted-external-validation-v2"

DEFAULT_EXECUTION_WORKSPACE = Path("data/cache/metasyn/passage-hosted-yield-v2")

RuntimeStage = Literal[
    "prepared",
    "globally_cost_authorized",
    "preflight_passed",
    "inventory_smoke_passed",
    "inventory_roster_terminal",
    "packet_roster_frozen",
    "packet_smoke_passed",
    "packet_smoke_not_applicable",
    "packet_roster_terminal",
    "finalized",
    "externally_validated",
]

_STAGE_ORDINAL: dict[RuntimeStage, int] = {
    "prepared": 0,
    "globally_cost_authorized": 1,
    "preflight_passed": 2,
    "inventory_smoke_passed": 3,
    "inventory_roster_terminal": 4,
    "packet_roster_frozen": 5,
    "packet_smoke_passed": 6,
    "packet_smoke_not_applicable": 6,
    "packet_roster_terminal": 7,
    "finalized": 8,
    "externally_validated": 9,
}
_STAGE_FILENAMES: dict[int, tuple[str, ...]] = {
    0: ("00-prepared.json",),
    1: ("01-globally-cost-authorized.json",),
    2: ("02-preflight-passed.json",),
    3: ("03-inventory-smoke-passed.json",),
    4: ("04-inventory-roster-terminal.json",),
    5: ("05-packet-roster-frozen.json",),
    6: ("06-packet-smoke-passed.json", "06-packet-smoke-not-applicable.json"),
    7: ("07-packet-roster-terminal.json",),
    8: ("08-finalized.json",),
    9: ("09-externally-validated.json",),
}

InventoryValidationStatus = Literal[
    "inventory_contract_valid",
    "runtime_capacity_failure",
    "provider_runtime_failure",
    "exact_once_terminal_incident",
    "inventory_contract_invalid",
]
PacketValidationStatus = Literal[
    "typed_effect_completed",
    "grounding_abstained",
    "assembly_abstained",
    "grounding_invalid",
    "assembly_invalid",
    "runtime_capacity_failure",
    "provider_runtime_failure",
    "exact_once_terminal_incident",
]


class MetaSynPassageHostedRuntimeV2Error(ValueError):
    """The private runtime workspace cannot be advanced without weakening a gate."""


class HostedClientProtocol(Protocol):
    def generate(self, request: AnthropicBoundedRequestV1) -> AnthropicBoundedResultV1:
        """Make the one provider attempt authorized for ``request``."""


class _ExactModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


def _validate_hash(value: str, field_name: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"metasyn_passage_runtime_v2_hash_invalid:{field_name}")
    return value


def _usd_micros(value: Decimal) -> int:
    return int((value * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING))


def _canonical_repository_root(value: Path) -> Path:
    try:
        root = Path(os.path.abspath(value)).resolve(strict=True)
    except OSError as exc:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_repository_root_unreadable"
        ) from exc
    if not root.is_dir() or root.is_symlink():
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_repository_root_unsafe"
        )
    return root


def _canonical_existing_workspace(value: Path) -> Path:
    path = Path(os.path.abspath(value))
    if path.is_symlink():
        raise MetaSynPassageHostedRuntimeV2Error("metasyn_passage_runtime_v2_workspace_symlink")
    try:
        root = path.resolve(strict=True)
    except OSError as exc:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_workspace_missing"
        ) from exc
    if not root.is_dir():
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_workspace_not_directory"
        )
    return root


def _create_fresh_workspace(value: Path) -> Path:
    path = Path(os.path.abspath(value))
    if path.exists() or path.is_symlink():
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_workspace_must_be_fresh"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_workspace_create_race"
        ) from exc
    return _canonical_existing_workspace(path)


@contextmanager
def _runtime_lock(workspace: Path) -> Iterator[None]:
    lock_path = workspace / ".metasyn-passage-hosted-runtime-v2.lock"
    if lock_path.is_symlink():
        raise MetaSynPassageHostedRuntimeV2Error("metasyn_passage_runtime_v2_lock_symlink")
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


def _runtime_paths(workspace: Path) -> dict[str, Path]:
    return {
        "bundle": workspace / "execution-bundle.json",
        "global_authorization": workspace / "run-cost-authorization.json",
        "stages": workspace / "stage-checkpoints",
        "provider": workspace / "provider-state",
        "validation": workspace / "validation-receipts",
        "preflight_calls": workspace / "validation-receipts" / "preflight-calls",
        "preflight": workspace / "validation-receipts" / "preflight.json",
        "inventory_results": workspace / "inventory-results",
        "inventory_receipts": workspace / "inventory-receipts",
        "inventory_ledger": workspace / "inventory-ledger.json",
        "packet_roster": workspace / "packet-roster.json",
        "packet_results": workspace / "packet-results",
        "packet_grounding": workspace / "packet-grounding-receipts",
        "packet_assembly": workspace / "packet-assembly-receipts",
        "packet_smoke": workspace / "packet-smoke.json",
        "packet_smoke_attempt": workspace / "packet-smoke-attempt.json",
        "packet_ledger": workspace / "packet-ledger.json",
        "final_report": workspace / "private-yield-report.json",
        "external_validation": workspace / "external-validation-receipt.json",
    }


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MetaSynPassageHostedRuntimeV2Error("metasyn_passage_runtime_v2_artifact_unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_artifact_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise MetaSynPassageHostedRuntimeV2Error("metasyn_passage_runtime_v2_artifact_not_object")
    return value


def _ensure_directory(path: Path) -> None:
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_state_directory_unsafe"
        )
    path.mkdir(exist_ok=True)


def _write_or_replay(path: Path, value: _ExactModel | ContractModel) -> None:
    if path.exists():
        if _read_object(path) != value.model_dump(mode="json"):
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_artifact_replay_mismatch"
            )
        return
    atomic_write_json(path, value)


class RuntimeArtifactBindingV2(_ExactModel):
    relative_path: Annotated[str, Field(min_length=1, max_length=512)]
    file_sha256: str

    @field_validator("relative_path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or "\\" in value
            or ".." in path.parts
            or "." in path.parts
            or path.as_posix() != value
        ):
            raise ValueError("metasyn_passage_runtime_v2_artifact_path_unsafe")
        return value

    @field_validator("file_sha256")
    @classmethod
    def validate_sha(cls, value: str) -> str:
        return _validate_hash(value, "file_sha256")


class RuntimeStageCheckpointV2(_ExactModel):
    checkpoint_version: Literal["metasyn-passage-hosted-stage-checkpoint-v2"] = (
        STAGE_CHECKPOINT_VERSION
    )
    runtime_version: Literal["metasyn-passage-hosted-runtime-v2"] = RUNTIME_VERSION
    execution_bundle_sha256: str
    stage: RuntimeStage
    stage_ordinal: Annotated[int, Field(ge=0, le=9)]
    previous_checkpoint_sha256: str | None
    artifacts: Annotated[list[RuntimeArtifactBindingV2], Field(min_length=1)]
    artifact_membership_sha256: str
    checkpoint_sha256: str

    @field_validator("execution_bundle_sha256", "artifact_membership_sha256", "checkpoint_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_hash(value, info.field_name)

    @field_validator("previous_checkpoint_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return None if value is None else _validate_hash(value, "previous_checkpoint_sha256")

    @model_validator(mode="after")
    def validate_checkpoint(self) -> RuntimeStageCheckpointV2:
        if self.stage_ordinal != _STAGE_ORDINAL[self.stage]:
            raise ValueError("metasyn_passage_runtime_v2_stage_ordinal_mismatch")
        if (self.stage_ordinal == 0) != (self.previous_checkpoint_sha256 is None):
            raise ValueError("metasyn_passage_runtime_v2_stage_predecessor_shape_invalid")
        paths = [item.relative_path for item in self.artifacts]
        if paths != sorted(set(paths)):
            raise ValueError("metasyn_passage_runtime_v2_stage_artifacts_not_canonical")
        if self.artifact_membership_sha256 != hash_canonical(
            [item.model_dump(mode="json") for item in self.artifacts]
        ):
            raise ValueError("metasyn_passage_runtime_v2_stage_artifact_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"checkpoint_sha256"})
        if self.checkpoint_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_runtime_v2_stage_hash_mismatch")
        return self


def _artifact_binding(*, workspace: Path, path: Path) -> RuntimeArtifactBindingV2:
    try:
        relative = path.relative_to(workspace).as_posix()
    except ValueError as exc:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_artifact_outside_workspace"
        ) from exc
    return RuntimeArtifactBindingV2(relative_path=relative, file_sha256=sha256_file(path))


def _checkpoint_path(workspace: Path, stage: RuntimeStage) -> Path:
    ordinal = _STAGE_ORDINAL[stage]
    filename = f"{ordinal:02d}-{stage.replace('_', '-')}.json"
    return _runtime_paths(workspace)["stages"] / filename


def _load_stage_chain(workspace: Path) -> list[RuntimeStageCheckpointV2]:
    stage_dir = _runtime_paths(workspace)["stages"]
    if stage_dir.is_symlink() or not stage_dir.is_dir():
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_stage_directory_missing"
        )
    observed: list[RuntimeStageCheckpointV2] = []
    previous: RuntimeStageCheckpointV2 | None = None
    for ordinal in range(10):
        candidates = [stage_dir / name for name in _STAGE_FILENAMES[ordinal]]
        present = [path for path in candidates if path.exists()]
        if len(present) > 1:
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_stage_branch_cardinality_invalid"
            )
        if not present:
            if any(
                (stage_dir / name).exists()
                for later in range(ordinal + 1, 10)
                for name in _STAGE_FILENAMES[later]
            ):
                raise MetaSynPassageHostedRuntimeV2Error("metasyn_passage_runtime_v2_stage_gap")
            break
        checkpoint = RuntimeStageCheckpointV2.model_validate(_read_object(present[0]))
        if checkpoint.stage_ordinal != ordinal:
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_stage_filename_mismatch"
            )
        expected_previous = None if previous is None else previous.checkpoint_sha256
        if checkpoint.previous_checkpoint_sha256 != expected_previous:
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_stage_chain_mismatch"
            )
        for artifact in checkpoint.artifacts:
            artifact_path = workspace / artifact.relative_path
            if sha256_file(artifact_path) != artifact.file_sha256:
                raise MetaSynPassageHostedRuntimeV2Error(
                    "metasyn_passage_runtime_v2_stage_artifact_tamper"
                )
        observed.append(checkpoint)
        previous = checkpoint
    extras = {
        path.name
        for path in stage_dir.iterdir()
        if path.name not in {name for names in _STAGE_FILENAMES.values() for name in names}
    }
    if extras:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_unknown_stage_artifact"
        )
    return observed


def _write_checkpoint(
    *,
    workspace: Path,
    bundle_sha256: str,
    stage: RuntimeStage,
    artifact_paths: Sequence[Path],
) -> RuntimeStageCheckpointV2:
    chain = _load_stage_chain(workspace)
    ordinal = _STAGE_ORDINAL[stage]
    if len(chain) > ordinal:
        existing = chain[ordinal]
        if existing.stage != stage:
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_stage_branch_mismatch"
            )
        return existing
    if len(chain) != ordinal:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_stage_transition_invalid"
        )
    artifacts = sorted(
        (_artifact_binding(workspace=workspace, path=path) for path in artifact_paths),
        key=lambda item: item.relative_path,
    )
    payload = {
        "checkpoint_version": STAGE_CHECKPOINT_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "execution_bundle_sha256": bundle_sha256,
        "stage": stage,
        "stage_ordinal": ordinal,
        "previous_checkpoint_sha256": (chain[-1].checkpoint_sha256 if chain else None),
        "artifacts": artifacts,
        "artifact_membership_sha256": hash_canonical(
            [item.model_dump(mode="json") for item in artifacts]
        ),
    }
    checkpoint = RuntimeStageCheckpointV2.model_validate(
        {**payload, "checkpoint_sha256": hash_canonical(payload)}
    )
    _write_or_replay(_checkpoint_path(workspace, stage), checkpoint)
    return checkpoint


def _require_stage(workspace: Path, minimum_ordinal: int) -> list[RuntimeStageCheckpointV2]:
    chain = _load_stage_chain(workspace)
    if len(chain) <= minimum_ordinal:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_required_stage_missing"
        )
    return chain


def _load_bundle(
    *, workspace: Path, repository_root: Path, expected_sha256: str
) -> MetaSynPassageHostedExecutionBundleV2:
    _validate_hash(expected_sha256, "expected_execution_bundle_sha256")
    bundle = validate_metasyn_passage_hosted_execution_bundle_v2(
        execution_bundle=_read_object(_runtime_paths(workspace)["bundle"]),
        repository_root=repository_root,
        external_replay=True,
    )
    if bundle.execution_bundle_sha256 != expected_sha256:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_execution_anchor_mismatch"
        )
    chain = _load_stage_chain(workspace)
    if any(item.execution_bundle_sha256 != expected_sha256 for item in chain):
        raise MetaSynPassageHostedRuntimeV2Error("metasyn_passage_runtime_v2_stage_bundle_mismatch")
    return bundle


class GlobalRunCostAuthorizationV2(_ExactModel):
    authorization_version: Literal["metasyn-passage-hosted-global-cost-authorization-v2"] = (
        GLOBAL_AUTHORIZATION_VERSION
    )
    status: Literal["pre_first_provider_call_global_cost_authorized"] = (
        "pre_first_provider_call_global_cost_authorized"
    )
    execution_bundle_sha256: str
    cost_envelope_sha256: str
    maximum_provider_calls: Literal[296] = 296
    conservative_input_token_ceiling: Annotated[int, Field(ge=1)]
    max_output_token_ceiling: Annotated[int, Field(ge=1)]
    cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    configured_cost_limit_usd_micros: Annotated[int, Field(ge=1)]
    provider_calls_made_before_authorization: Literal[0] = 0
    source_transmission_operator_authorized: Literal[True] = True
    retries_per_request: Literal[0] = 0
    authorization_sha256: str

    @field_validator("execution_bundle_sha256", "cost_envelope_sha256", "authorization_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_hash(value, info.field_name)

    @model_validator(mode="after")
    def validate_authorization(self) -> GlobalRunCostAuthorizationV2:
        if self.cost_ceiling_usd_micros > self.configured_cost_limit_usd_micros:
            raise ValueError("metasyn_passage_runtime_v2_global_cost_exceeds_limit")
        payload = self.model_dump(mode="json", exclude={"authorization_sha256"})
        if self.authorization_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_runtime_v2_global_authorization_hash_mismatch")
        return self


def _freeze_global_authorization(
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> GlobalRunCostAuthorizationV2:
    envelope = bundle.global_cost_envelope
    payload = {
        "authorization_version": GLOBAL_AUTHORIZATION_VERSION,
        "status": "pre_first_provider_call_global_cost_authorized",
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "cost_envelope_sha256": envelope.cost_envelope_sha256,
        "maximum_provider_calls": envelope.maximum_provider_calls,
        "conservative_input_token_ceiling": envelope.conservative_input_token_ceiling,
        "max_output_token_ceiling": envelope.max_output_token_ceiling,
        "cost_ceiling_usd_micros": envelope.cost_ceiling_usd_micros,
        "configured_cost_limit_usd_micros": envelope.configured_maximum_cost_usd_micros,
        "provider_calls_made_before_authorization": 0,
        "source_transmission_operator_authorized": True,
        "retries_per_request": 0,
    }
    return GlobalRunCostAuthorizationV2.model_validate(
        {**payload, "authorization_sha256": hash_canonical(payload)}
    )


def _preflight_intents(
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> list[HostedExactOnceIntentV1]:
    return [
        freeze_hosted_exact_once_intent(
            execution_bundle_sha256=bundle.execution_bundle_sha256,
            phase="source_free_preflight",
            source_bearing=False,
            context_binding_sha256=item.preflight_call_sha256,
            request=item.request,
        )
        for item in bundle.source_free_preflight_plan
    ]


def _inventory_intents(
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> list[HostedExactOnceIntentV1]:
    return [
        freeze_hosted_exact_once_intent(
            execution_bundle_sha256=bundle.execution_bundle_sha256,
            phase="inventory",
            source_bearing=True,
            context_binding_sha256=item.inventory_request_sha256,
            request=item.request,
        )
        for item in bundle.inventory_requests
    ]


def _persist_exact_authorization(
    *,
    provider_workspace: Path,
    authorization: HostedExactOnceCostAuthorizationV1,
) -> None:
    _ensure_directory(provider_workspace)
    auth_dir = provider_workspace / "cost-authorizations"
    _ensure_directory(auth_dir)
    path = auth_dir / f"{authorization.phase}.json"
    if path.exists():
        saved = HostedExactOnceCostAuthorizationV1.model_validate(_read_object(path))
        if saved != authorization:
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_exact_authorization_replay_mismatch"
            )
        return
    intent_dir = provider_workspace / "call-intents"
    if intent_dir.exists():
        if intent_dir.is_symlink() or not intent_dir.is_dir():
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_exact_intent_directory_unsafe"
            )
        for intent_path in intent_dir.glob("*.json"):
            intent = HostedExactOnceIntentV1.model_validate(_read_object(intent_path))
            if intent.phase == authorization.phase:
                raise MetaSynPassageHostedRuntimeV2Error(
                    "metasyn_passage_runtime_v2_intent_precedes_phase_authorization"
                )
    atomic_write_json(path, authorization)


def _freeze_phase_authorization(
    *,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    phase: Literal["source_free_preflight", "inventory", "packet"],
    intents: Sequence[HostedExactOnceIntentV1],
    configured_budget_usd_micros: int,
) -> HostedExactOnceCostAuthorizationV1:
    return freeze_hosted_exact_once_cost_authorization(
        execution_bundle_sha256=bundle.execution_bundle_sha256,
        phase=phase,
        intents=intents,
        configured_phase_budget_usd_micros=configured_budget_usd_micros,
    )


class TerminalCallRefV2(_ExactModel):
    request_key: str
    intent_sha256: str
    authorization_sha256: str
    terminal_kind: Literal["provider_receipt", "ambiguity_incident"]
    terminal_sha256: str
    provider_outcome: str | None
    request_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]

    @field_validator("intent_sha256", "authorization_sha256", "terminal_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_hash(value, info.field_name)


def _terminal_ref(
    *,
    intent: HostedExactOnceIntentV1,
    authorization: HostedExactOnceCostAuthorizationV1,
    outcome: HostedExactOnceProviderReceiptV1 | HostedExactOnceAmbiguityIncidentV1,
) -> TerminalCallRefV2:
    if isinstance(outcome, HostedExactOnceProviderReceiptV1):
        return TerminalCallRefV2(
            request_key=intent.request_key,
            intent_sha256=intent.intent_sha256,
            authorization_sha256=authorization.authorization_sha256,
            terminal_kind="provider_receipt",
            terminal_sha256=outcome.receipt_sha256,
            provider_outcome=outcome.provider_result.outcome,
            request_cost_ceiling_usd_micros=intent.request_cost_ceiling_usd_micros,
        )
    return TerminalCallRefV2(
        request_key=intent.request_key,
        intent_sha256=intent.intent_sha256,
        authorization_sha256=authorization.authorization_sha256,
        terminal_kind="ambiguity_incident",
        terminal_sha256=outcome.incident_sha256,
        provider_outcome=None,
        request_cost_ceiling_usd_micros=intent.request_cost_ceiling_usd_micros,
    )


def _is_capacity_failure(result: AnthropicBoundedResultV1) -> bool:
    return result.outcome == "response_stop_reason_invalid" and result.stop_reason in {
        "max_tokens",
        "model_context_window_exceeded",
    }


class PreflightCallValidationV2(_ExactModel):
    preflight_ordinal: Annotated[int, Field(ge=0, lt=8)]
    preflight_call_sha256: str
    expected_fixture_sha256: str
    terminal: TerminalCallRefV2
    validation_status: Literal[
        "fixture_exact",
        "fixture_mismatch",
        "runtime_capacity_failure",
        "provider_runtime_failure",
        "exact_once_terminal_incident",
    ]
    validation_sha256: str

    @field_validator("preflight_call_sha256", "expected_fixture_sha256", "validation_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_hash(value, info.field_name)

    @model_validator(mode="after")
    def validate_self_hash(self) -> PreflightCallValidationV2:
        payload = self.model_dump(mode="json", exclude={"validation_sha256"})
        if self.validation_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_runtime_v2_preflight_call_hash_mismatch")
        return self


class PreflightReceiptV2(_ExactModel):
    receipt_version: Literal["metasyn-passage-hosted-preflight-receipt-v2"] = (
        PREFLIGHT_RECEIPT_VERSION
    )
    status: Literal["passed_all_eight_source_free_calls"] = "passed_all_eight_source_free_calls"
    execution_bundle_sha256: str
    exact_authorization_sha256: str
    calls: Annotated[list[PreflightCallValidationV2], Field(min_length=8, max_length=8)]
    call_membership_sha256: str
    source_bearing_call_count: Literal[0] = 0
    extraction_accuracy_authority: Literal[False] = False
    receipt_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "exact_authorization_sha256",
        "call_membership_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_hash(value, info.field_name)

    @model_validator(mode="after")
    def validate_receipt(self) -> PreflightReceiptV2:
        if [item.preflight_ordinal for item in self.calls] != list(range(8)):
            raise ValueError("metasyn_passage_runtime_v2_preflight_order_mismatch")
        if any(item.validation_status != "fixture_exact" for item in self.calls):
            raise ValueError("metasyn_passage_runtime_v2_preflight_not_all_passed")
        if self.call_membership_sha256 != hash_canonical(
            [item.validation_sha256 for item in self.calls]
        ):
            raise ValueError("metasyn_passage_runtime_v2_preflight_membership_mismatch")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if self.receipt_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_runtime_v2_preflight_hash_mismatch")
        return self


class InventoryCallResultV2(_ExactModel):
    result_version: Literal["metasyn-passage-hosted-inventory-result-v2"] = INVENTORY_RESULT_VERSION
    execution_bundle_sha256: str
    row_ordinal: Annotated[int, Field(ge=0, lt=32)]
    row_key: str
    inventory_request_sha256: str
    terminal: TerminalCallRefV2
    validation_status: InventoryValidationStatus
    inventory_receipt: MetaSynCandidateInventoryReceiptV2 | None
    inventory_receipt_sha256: str | None
    authorizes_packet_generation: bool
    runtime_failure_is_not_scientific_abstention: Literal[True] = True
    result_sha256: str

    @field_validator("execution_bundle_sha256", "inventory_request_sha256", "result_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_hash(value, info.field_name)

    @field_validator("inventory_receipt_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return None if value is None else _validate_hash(value, "inventory_receipt_sha256")

    @model_validator(mode="after")
    def validate_result(self) -> InventoryCallResultV2:
        has_receipt = self.inventory_receipt is not None
        if has_receipt != (self.validation_status == "inventory_contract_valid"):
            raise ValueError("metasyn_passage_runtime_v2_inventory_receipt_shape_invalid")
        if has_receipt:
            assert self.inventory_receipt is not None
            if self.inventory_receipt_sha256 != self.inventory_receipt.receipt_sha256:
                raise ValueError("metasyn_passage_runtime_v2_inventory_receipt_hash_mismatch")
            expected_authority = self.inventory_receipt.status == "candidates_authorized"
            if self.authorizes_packet_generation != expected_authority:
                raise ValueError("metasyn_passage_runtime_v2_inventory_authority_mismatch")
        elif self.inventory_receipt_sha256 is not None or self.authorizes_packet_generation:
            raise ValueError("metasyn_passage_runtime_v2_inventory_nonreceipt_authority")
        payload = self.model_dump(mode="json", exclude={"result_sha256"})
        if self.result_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_runtime_v2_inventory_result_hash_mismatch")
        return self


class InventoryLedgerV2(_ExactModel):
    ledger_version: Literal["metasyn-passage-hosted-inventory-ledger-v2"] = INVENTORY_LEDGER_VERSION
    status: Literal["all_32_inventory_calls_terminal"] = "all_32_inventory_calls_terminal"
    execution_bundle_sha256: str
    exact_authorization_sha256: str
    results: Annotated[list[InventoryCallResultV2], Field(min_length=32, max_length=32)]
    result_membership_sha256: str
    validation_status_counts: dict[str, Annotated[int, Field(ge=0)]]
    packet_authorizing_row_count: Annotated[int, Field(ge=0, le=32)]
    authorized_candidate_count: Annotated[int, Field(ge=0, le=256)]
    all_calls_terminal: Literal[True] = True
    extraction_accuracy_authority: Literal[False] = False
    ledger_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "exact_authorization_sha256",
        "result_membership_sha256",
        "ledger_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_hash(value, info.field_name)

    @model_validator(mode="after")
    def validate_ledger(self) -> InventoryLedgerV2:
        if [item.row_ordinal for item in self.results] != list(range(32)):
            raise ValueError("metasyn_passage_runtime_v2_inventory_ledger_order_mismatch")
        if self.result_membership_sha256 != hash_canonical(
            [item.result_sha256 for item in self.results]
        ):
            raise ValueError("metasyn_passage_runtime_v2_inventory_ledger_membership_mismatch")
        counts = dict(sorted(Counter(item.validation_status for item in self.results).items()))
        if self.validation_status_counts != counts:
            raise ValueError("metasyn_passage_runtime_v2_inventory_counts_mismatch")
        authorized = [item for item in self.results if item.authorizes_packet_generation]
        candidate_count = sum(
            len(item.inventory_receipt.inventory.candidates)
            for item in authorized
            if item.inventory_receipt is not None
        )
        if (
            self.packet_authorizing_row_count != len(authorized)
            or self.authorized_candidate_count != candidate_count
        ):
            raise ValueError("metasyn_passage_runtime_v2_inventory_authorized_count_mismatch")
        payload = self.model_dump(mode="json", exclude={"ledger_sha256"})
        if self.ledger_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_runtime_v2_inventory_ledger_hash_mismatch")
        return self


class PacketRequestV2(_ExactModel):
    packet_request_version: Literal["metasyn-passage-hosted-packet-request-v2"] = (
        PACKET_REQUEST_VERSION
    )
    row_ordinal: Annotated[int, Field(ge=0, lt=32)]
    candidate_index: Annotated[int, Field(ge=1, le=8)]
    row_key: str
    inventory_receipt_sha256: str
    packet_input: MetaSynPacketCandidateInputV2
    packet_input_sha256: str
    capacity_limited_schema_sha256: str
    compiled_schema: AnthropicCompiledSchemaV1
    compiled_schema_sha256: str
    request: AnthropicBoundedRequestV1
    request_sha256: str
    row_cost_envelope_sha256: str
    within_frozen_row_cost_envelope: Literal[True] = True
    packet_request_sha256: str

    @field_validator(
        "inventory_receipt_sha256",
        "packet_input_sha256",
        "capacity_limited_schema_sha256",
        "compiled_schema_sha256",
        "request_sha256",
        "row_cost_envelope_sha256",
        "packet_request_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_hash(value, info.field_name)

    @model_validator(mode="after")
    def validate_request(self) -> PacketRequestV2:
        if (
            self.packet_input_sha256 != self.packet_input.packet_input_sha256
            or self.compiled_schema_sha256 != self.compiled_schema.compiled_schema_sha256
            or self.request_sha256 != self.request.request_sha256
            or self.request.compiled_schema != self.compiled_schema
            or self.request.schema_kind != "packet"
            or self.request.effect_kind != self.packet_input.candidate.effect_kind
            or self.request.base_prompt_sha256 != self.packet_input.rendered_prompt_sha256
            or self.row_ordinal != self.packet_input.row_ordinal
            or self.candidate_index != self.packet_input.candidate.candidate_index
        ):
            raise ValueError("metasyn_passage_runtime_v2_packet_request_alias_mismatch")
        payload = self.model_dump(mode="json", exclude={"packet_request_sha256"})
        if self.packet_request_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_runtime_v2_packet_request_hash_mismatch")
        return self


class PacketRosterV2(_ExactModel):
    roster_version: Literal["metasyn-passage-hosted-packet-roster-v2"] = PACKET_ROSTER_VERSION
    status: Literal["exact_dynamic_roster_frozen_from_replayed_inventories"] = (
        "exact_dynamic_roster_frozen_from_replayed_inventories"
    )
    execution_bundle_sha256: str
    inventory_ledger_sha256: str
    requests: Annotated[list[PacketRequestV2], Field(max_length=256)]
    request_count: Annotated[int, Field(ge=0, le=256)]
    request_membership_sha256: str
    exact_authorization: HostedExactOnceCostAuthorizationV1 | None
    exact_authorization_sha256: str | None
    zero_roster_disposition: Literal["not_applicable"] | None
    roster_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "inventory_ledger_sha256",
        "request_membership_sha256",
        "roster_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_hash(value, info.field_name)

    @field_validator("exact_authorization_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return None if value is None else _validate_hash(value, "exact_authorization_sha256")

    @model_validator(mode="after")
    def validate_roster(self) -> PacketRosterV2:
        order = [(item.row_ordinal, item.candidate_index) for item in self.requests]
        if order != sorted(set(order)):
            raise ValueError("metasyn_passage_runtime_v2_packet_roster_not_canonical")
        if self.request_count != len(self.requests):
            raise ValueError("metasyn_passage_runtime_v2_packet_roster_count_mismatch")
        if self.request_membership_sha256 != hash_canonical(
            [item.packet_request_sha256 for item in self.requests]
        ):
            raise ValueError("metasyn_passage_runtime_v2_packet_roster_membership_mismatch")
        if self.requests:
            if (
                self.exact_authorization is None
                or self.exact_authorization_sha256 != self.exact_authorization.authorization_sha256
                or self.exact_authorization.phase != "packet"
                or self.zero_roster_disposition is not None
            ):
                raise ValueError("metasyn_passage_runtime_v2_packet_authorization_shape_invalid")
        elif (
            self.exact_authorization is not None
            or self.exact_authorization_sha256 is not None
            or self.zero_roster_disposition != "not_applicable"
        ):
            raise ValueError("metasyn_passage_runtime_v2_zero_roster_shape_invalid")
        payload = self.model_dump(mode="json", exclude={"roster_sha256"})
        if self.roster_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_runtime_v2_packet_roster_hash_mismatch")
        return self


class PacketCallResultV2(_ExactModel):
    result_version: Literal["metasyn-passage-hosted-packet-result-v2"] = PACKET_RESULT_VERSION
    execution_bundle_sha256: str
    packet_request_sha256: str
    row_ordinal: Annotated[int, Field(ge=0, lt=32)]
    candidate_index: Annotated[int, Field(ge=1, le=8)]
    terminal: TerminalCallRefV2
    validation_status: PacketValidationStatus
    grounding_receipt: PacketGroundingReceiptV2 | None
    grounding_receipt_sha256: str | None
    assembly_receipt: NativePacketAssemblyOutcomeV2 | None
    assembly_receipt_sha256: str | None
    authorizes_typed_effect: bool
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    runtime_failure_is_not_scientific_abstention: Literal[True] = True
    result_sha256: str

    @field_validator("execution_bundle_sha256", "packet_request_sha256", "result_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_hash(value, info.field_name)

    @field_validator("grounding_receipt_sha256", "assembly_receipt_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _validate_hash(value, info.field_name)

    @model_validator(mode="after")
    def validate_result(self) -> PacketCallResultV2:
        grounding_statuses = {
            "typed_effect_completed",
            "grounding_abstained",
            "assembly_abstained",
            "assembly_invalid",
        }
        assembly_statuses = {
            "typed_effect_completed",
            "grounding_abstained",
            "assembly_abstained",
        }
        if (self.grounding_receipt is not None) != (self.validation_status in grounding_statuses):
            raise ValueError("metasyn_passage_runtime_v2_packet_grounding_shape_invalid")
        if (self.assembly_receipt is not None) != (self.validation_status in assembly_statuses):
            raise ValueError("metasyn_passage_runtime_v2_packet_assembly_shape_invalid")
        if self.grounding_receipt is not None and self.grounding_receipt_sha256 != (
            self.grounding_receipt.receipt_sha256
        ):
            raise ValueError("metasyn_passage_runtime_v2_packet_grounding_hash_mismatch")
        if self.grounding_receipt is None and self.grounding_receipt_sha256 is not None:
            raise ValueError("metasyn_passage_runtime_v2_packet_grounding_hash_unexpected")
        if self.assembly_receipt is not None and self.assembly_receipt_sha256 != (
            self.assembly_receipt.assembly_receipt_sha256
        ):
            raise ValueError("metasyn_passage_runtime_v2_packet_assembly_hash_mismatch")
        if self.assembly_receipt is None and self.assembly_receipt_sha256 is not None:
            raise ValueError("metasyn_passage_runtime_v2_packet_assembly_hash_unexpected")
        expected_authority = self.validation_status == "typed_effect_completed"
        if self.authorizes_typed_effect != expected_authority:
            raise ValueError("metasyn_passage_runtime_v2_packet_typed_authority_mismatch")
        payload = self.model_dump(mode="json", exclude={"result_sha256"})
        if self.result_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_runtime_v2_packet_result_hash_mismatch")
        return self


class PacketSmokeReceiptV2(_ExactModel):
    smoke_version: Literal["metasyn-passage-hosted-packet-smoke-v2"] = PACKET_SMOKE_VERSION
    execution_bundle_sha256: str
    packet_roster_sha256: str
    status: Literal["passed", "not_applicable", "failed_gate"]
    ordered_smoke_request_keys: Annotated[list[str], Field(max_length=3)]
    attempted_result_sha256s: Annotated[list[str], Field(max_length=3)]
    completed_typed_effect_result_sha256: str | None
    valid_abstention_does_not_pass: Literal[True] = True
    remaining_packet_calls_permitted: bool
    smoke_sha256: str

    @field_validator("execution_bundle_sha256", "packet_roster_sha256", "smoke_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_hash(value, info.field_name)

    @field_validator("completed_typed_effect_result_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return (
            None if value is None else _validate_hash(value, "completed_typed_effect_result_sha256")
        )

    @model_validator(mode="after")
    def validate_smoke(self) -> PacketSmokeReceiptV2:
        if len(self.ordered_smoke_request_keys) != len(self.attempted_result_sha256s):
            raise ValueError("metasyn_passage_runtime_v2_smoke_attempt_alias_mismatch")
        if self.status == "passed":
            if self.completed_typed_effect_result_sha256 is None or not (
                self.remaining_packet_calls_permitted
            ):
                raise ValueError("metasyn_passage_runtime_v2_smoke_pass_shape_invalid")
        elif self.status == "not_applicable":
            if (
                self.ordered_smoke_request_keys
                or self.completed_typed_effect_result_sha256 is not None
                or not self.remaining_packet_calls_permitted
            ):
                raise ValueError("metasyn_passage_runtime_v2_smoke_na_shape_invalid")
        elif self.completed_typed_effect_result_sha256 is not None or (
            self.remaining_packet_calls_permitted
        ):
            raise ValueError("metasyn_passage_runtime_v2_smoke_failure_shape_invalid")
        payload = self.model_dump(mode="json", exclude={"smoke_sha256"})
        if self.smoke_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_runtime_v2_smoke_hash_mismatch")
        return self


class PacketLedgerV2(_ExactModel):
    ledger_version: Literal["metasyn-passage-hosted-packet-ledger-v2"] = PACKET_LEDGER_VERSION
    status: Literal["exact_dynamic_packet_roster_terminal"] = "exact_dynamic_packet_roster_terminal"
    execution_bundle_sha256: str
    packet_roster_sha256: str
    packet_smoke_sha256: str
    results: Annotated[list[PacketCallResultV2], Field(max_length=256)]
    result_membership_sha256: str
    validation_status_counts: dict[str, Annotated[int, Field(ge=0)]]
    typed_effect_count: Annotated[int, Field(ge=0, le=256)]
    all_calls_terminal: Literal[True] = True
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    ledger_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "packet_roster_sha256",
        "packet_smoke_sha256",
        "result_membership_sha256",
        "ledger_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_hash(value, info.field_name)

    @model_validator(mode="after")
    def validate_ledger(self) -> PacketLedgerV2:
        order = [(item.row_ordinal, item.candidate_index) for item in self.results]
        if order != sorted(set(order)):
            raise ValueError("metasyn_passage_runtime_v2_packet_ledger_order_mismatch")
        if self.result_membership_sha256 != hash_canonical(
            [item.result_sha256 for item in self.results]
        ):
            raise ValueError("metasyn_passage_runtime_v2_packet_ledger_membership_mismatch")
        counts = dict(sorted(Counter(item.validation_status for item in self.results).items()))
        if self.validation_status_counts != counts:
            raise ValueError("metasyn_passage_runtime_v2_packet_counts_mismatch")
        if self.typed_effect_count != sum(item.authorizes_typed_effect for item in self.results):
            raise ValueError("metasyn_passage_runtime_v2_typed_effect_count_mismatch")
        payload = self.model_dump(mode="json", exclude={"ledger_sha256"})
        if self.ledger_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_runtime_v2_packet_ledger_hash_mismatch")
        return self


class PrivateYieldReportV2(_ExactModel):
    report_version: Literal["metasyn-passage-hosted-private-yield-report-v2"] = FINAL_REPORT_VERSION
    status: Literal["complete_yield_only_no_accuracy_or_release_authority"] = (
        "complete_yield_only_no_accuracy_or_release_authority"
    )
    execution_bundle_sha256: str
    bundle_pipeline_sha256: str
    extraction_inputs_sha256: str
    inventory_ledger_sha256: str
    packet_roster_sha256: str
    packet_smoke_sha256: str
    packet_ledger_sha256: str
    inventory_call_count: Literal[32] = 32
    packet_call_count: Annotated[int, Field(ge=0, le=256)]
    total_provider_attempts_or_possible_attempts: Annotated[int, Field(ge=40, le=296)]
    source_bearing_attempts_or_possible_attempts: Annotated[int, Field(ge=32, le=288)]
    typed_effect_count: Annotated[int, Field(ge=0, le=256)]
    inventory_validation_status_counts: dict[str, Annotated[int, Field(ge=0)]]
    packet_validation_status_counts: dict[str, Annotated[int, Field(ge=0)]]
    reported_usage_receipt_count: Annotated[int, Field(ge=0, le=296)]
    reported_input_tokens: Annotated[int, Field(ge=0)]
    reported_output_tokens: Annotated[int, Field(ge=0)]
    reported_estimated_cost_usd_micros: Annotated[int, Field(ge=0)]
    conservative_attempt_liability_usd_micros: Annotated[int, Field(ge=1)]
    configured_global_cost_ceiling_usd_micros: Annotated[int, Field(ge=1)]
    reference_fields_unopened: Literal[True] = True
    official_test_labels_opened: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    report_sha256: str

    @field_validator(
        "execution_bundle_sha256",
        "bundle_pipeline_sha256",
        "extraction_inputs_sha256",
        "inventory_ledger_sha256",
        "packet_roster_sha256",
        "packet_smoke_sha256",
        "packet_ledger_sha256",
        "report_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_hash(value, info.field_name)

    @model_validator(mode="after")
    def validate_report(self) -> PrivateYieldReportV2:
        if self.total_provider_attempts_or_possible_attempts != 40 + self.packet_call_count:
            raise ValueError("metasyn_passage_runtime_v2_total_attempt_count_mismatch")
        if self.source_bearing_attempts_or_possible_attempts != 32 + self.packet_call_count:
            raise ValueError("metasyn_passage_runtime_v2_source_attempt_count_mismatch")
        if self.conservative_attempt_liability_usd_micros > (
            self.configured_global_cost_ceiling_usd_micros
        ):
            raise ValueError("metasyn_passage_runtime_v2_realized_liability_exceeds_global")
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        if self.report_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_runtime_v2_report_hash_mismatch")
        return self


class ExternalValidationReceiptV2(_ExactModel):
    validation_version: Literal["metasyn-passage-hosted-external-validation-v2"] = (
        EXTERNAL_VALIDATION_VERSION
    )
    status: Literal["all_current_artifacts_externally_replayed"] = (
        "all_current_artifacts_externally_replayed"
    )
    execution_bundle_sha256: str
    final_report_sha256: str
    stage_count_before_validation: Literal[9] = 9
    exact_terminal_outcome_count: Annotated[int, Field(ge=40, le=296)]
    saved_grounding_receipt_count: Annotated[int, Field(ge=0, le=256)]
    saved_assembly_receipt_count: Annotated[int, Field(ge=0, le=256)]
    provider_calls_made_by_validation: Literal[0] = 0
    validation_sha256: str

    @field_validator("execution_bundle_sha256", "final_report_sha256", "validation_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _validate_hash(value, info.field_name)

    @model_validator(mode="after")
    def validate_receipt(self) -> ExternalValidationReceiptV2:
        payload = self.model_dump(mode="json", exclude={"validation_sha256"})
        if self.validation_sha256 != hash_canonical(payload):
            raise ValueError("metasyn_passage_runtime_v2_external_validation_hash_mismatch")
        return self


_GROUNDING_ADAPTER = TypeAdapter(PacketGroundingReceiptV2)
_ASSEMBLY_ADAPTER = TypeAdapter(NativePacketAssemblyOutcomeV2)


def prepare_metasyn_passage_hosted_runtime_v2(
    *,
    repository_root: Path,
    workspace: Path = DEFAULT_EXECUTION_WORKSPACE,
) -> MetaSynPassageHostedExecutionBundleV2:
    """Create a never-before-used private workspace and write the replayed bundle."""

    root = _canonical_repository_root(repository_root)
    bundle = freeze_metasyn_passage_hosted_execution_bundle_v2(repository_root=root)
    bundle = validate_metasyn_passage_hosted_execution_bundle_v2(
        execution_bundle=bundle,
        repository_root=root,
        external_replay=True,
    )
    canonical_workspace = _create_fresh_workspace(workspace)
    with _runtime_lock(canonical_workspace):
        paths = _runtime_paths(canonical_workspace)
        for name in (
            "stages",
            "provider",
            "validation",
            "preflight_calls",
            "inventory_results",
            "inventory_receipts",
            "packet_results",
            "packet_grounding",
            "packet_assembly",
        ):
            _ensure_directory(paths[name])
        atomic_write_json(paths["bundle"], bundle)
        replayed = validate_metasyn_passage_hosted_execution_bundle_v2(
            execution_bundle=_read_object(paths["bundle"]),
            repository_root=root,
            external_replay=True,
        )
        if replayed != bundle:
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_written_bundle_replay_mismatch"
            )
        _write_checkpoint(
            workspace=canonical_workspace,
            bundle_sha256=bundle.execution_bundle_sha256,
            stage="prepared",
            artifact_paths=[paths["bundle"]],
        )
    return bundle


def authorize_metasyn_passage_hosted_runtime_v2(
    *,
    repository_root: Path,
    workspace: Path,
    expected_execution_bundle_sha256: str,
) -> GlobalRunCostAuthorizationV2:
    """Persist the global envelope and exact eight-call preflight authorization."""

    root = _canonical_repository_root(repository_root)
    ws = _canonical_existing_workspace(workspace)
    with _runtime_lock(ws):
        bundle = _load_bundle(
            workspace=ws,
            repository_root=root,
            expected_sha256=expected_execution_bundle_sha256,
        )
        chain = _require_stage(ws, 0)
        paths = _runtime_paths(ws)
        if len(chain) > 1:
            return GlobalRunCostAuthorizationV2.model_validate(
                _read_object(paths["global_authorization"])
            )
        provider_files = [
            path
            for directory in (
                paths["provider"] / "call-intents",
                paths["provider"] / "provider-receipts",
                paths["provider"] / "ambiguity-incidents",
            )
            if directory.exists()
            for path in directory.glob("*.json")
        ]
        if provider_files:
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_provider_state_precedes_global_authorization"
            )
        global_auth = _freeze_global_authorization(bundle)
        _write_or_replay(paths["global_authorization"], global_auth)
        intents = _preflight_intents(bundle)
        phase_auth = _freeze_phase_authorization(
            bundle=bundle,
            phase="source_free_preflight",
            intents=intents,
            configured_budget_usd_micros=(
                bundle.global_cost_envelope.source_free_preflight.cost_ceiling_usd_micros
            ),
        )
        _persist_exact_authorization(provider_workspace=paths["provider"], authorization=phase_auth)
        _write_checkpoint(
            workspace=ws,
            bundle_sha256=bundle.execution_bundle_sha256,
            stage="globally_cost_authorized",
            artifact_paths=[
                paths["global_authorization"],
                paths["provider"] / "cost-authorizations" / "source_free_preflight.json",
            ],
        )
        return global_auth


def _preflight_call_validation(
    *,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    ordinal: int,
    intent: HostedExactOnceIntentV1,
    authorization: HostedExactOnceCostAuthorizationV1,
    outcome: HostedExactOnceProviderReceiptV1 | HostedExactOnceAmbiguityIncidentV1,
) -> PreflightCallValidationV2:
    spec = bundle.source_free_preflight_plan[ordinal]
    if isinstance(outcome, HostedExactOnceAmbiguityIncidentV1):
        status = "exact_once_terminal_incident"
    elif _is_capacity_failure(outcome.provider_result):
        status = "runtime_capacity_failure"
    elif outcome.provider_result.outcome != "completed":
        status = "provider_runtime_failure"
    elif outcome.provider_result.parsed_json != spec.expected_fixture:
        status = "fixture_mismatch"
    else:
        status = "fixture_exact"
    payload = {
        "preflight_ordinal": ordinal,
        "preflight_call_sha256": spec.preflight_call_sha256,
        "expected_fixture_sha256": spec.expected_fixture_sha256,
        "terminal": _terminal_ref(intent=intent, authorization=authorization, outcome=outcome),
        "validation_status": status,
    }
    return PreflightCallValidationV2.model_validate(
        {**payload, "validation_sha256": hash_canonical(payload)}
    )


def run_metasyn_passage_source_free_preflight_v2(
    *,
    repository_root: Path,
    workspace: Path,
    expected_execution_bundle_sha256: str,
    client: HostedClientProtocol,
) -> PreflightReceiptV2:
    root = _canonical_repository_root(repository_root)
    ws = _canonical_existing_workspace(workspace)
    with _runtime_lock(ws):
        bundle = _load_bundle(
            workspace=ws,
            repository_root=root,
            expected_sha256=expected_execution_bundle_sha256,
        )
        chain = _require_stage(ws, 1)
        paths = _runtime_paths(ws)
        if len(chain) > 2:
            return validate_metasyn_passage_preflight_v2(workspace=ws, execution_bundle=bundle)
        intents = _preflight_intents(bundle)
        authorization = _freeze_phase_authorization(
            bundle=bundle,
            phase="source_free_preflight",
            intents=intents,
            configured_budget_usd_micros=(
                bundle.global_cost_envelope.source_free_preflight.cost_ceiling_usd_micros
            ),
        )
        _persist_exact_authorization(
            provider_workspace=paths["provider"], authorization=authorization
        )
        validations: list[PreflightCallValidationV2] = []
        for ordinal, intent in enumerate(intents):
            outcome = execute_hosted_exactly_once(
                workspace=paths["provider"],
                intent=intent,
                authorization=authorization,
                client=client,
            )
            replayed = validate_hosted_exact_once_outcome(
                workspace=paths["provider"],
                intent=intent,
                authorization=authorization,
            )
            if outcome != replayed:
                raise MetaSynPassageHostedRuntimeV2Error(
                    "metasyn_passage_runtime_v2_preflight_exact_replay_mismatch"
                )
            validation = _preflight_call_validation(
                bundle=bundle,
                ordinal=ordinal,
                intent=intent,
                authorization=authorization,
                outcome=outcome,
            )
            path = paths["preflight_calls"] / f"preflight-{ordinal:02d}.json"
            _write_or_replay(path, validation)
            validations.append(validation)
        failed = [item for item in validations if item.validation_status != "fixture_exact"]
        if failed:
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_preflight_gate_failed:"
                + ",".join(item.validation_status for item in failed)
            )
        payload = {
            "receipt_version": PREFLIGHT_RECEIPT_VERSION,
            "status": "passed_all_eight_source_free_calls",
            "execution_bundle_sha256": bundle.execution_bundle_sha256,
            "exact_authorization_sha256": authorization.authorization_sha256,
            "calls": validations,
            "call_membership_sha256": hash_canonical(
                [item.validation_sha256 for item in validations]
            ),
            "source_bearing_call_count": 0,
            "extraction_accuracy_authority": False,
        }
        receipt = PreflightReceiptV2.model_validate(
            {**payload, "receipt_sha256": hash_canonical(payload)}
        )
        _write_or_replay(paths["preflight"], receipt)
        validate_metasyn_passage_preflight_v2(workspace=ws, execution_bundle=bundle)
        _write_checkpoint(
            workspace=ws,
            bundle_sha256=bundle.execution_bundle_sha256,
            stage="preflight_passed",
            artifact_paths=[paths["preflight"]],
        )
        return receipt


def validate_metasyn_passage_preflight_v2(
    *, workspace: Path, execution_bundle: MetaSynPassageHostedExecutionBundleV2
) -> PreflightReceiptV2:
    ws = _canonical_existing_workspace(workspace)
    paths = _runtime_paths(ws)
    receipt = PreflightReceiptV2.model_validate(_read_object(paths["preflight"]))
    intents = _preflight_intents(execution_bundle)
    authorization = _freeze_phase_authorization(
        bundle=execution_bundle,
        phase="source_free_preflight",
        intents=intents,
        configured_budget_usd_micros=(
            execution_bundle.global_cost_envelope.source_free_preflight.cost_ceiling_usd_micros
        ),
    )
    validations: list[PreflightCallValidationV2] = []
    for ordinal, intent in enumerate(intents):
        outcome = validate_hosted_exact_once_outcome(
            workspace=paths["provider"], intent=intent, authorization=authorization
        )
        expected = _preflight_call_validation(
            bundle=execution_bundle,
            ordinal=ordinal,
            intent=intent,
            authorization=authorization,
            outcome=outcome,
        )
        saved = PreflightCallValidationV2.model_validate(
            _read_object(paths["preflight_calls"] / f"preflight-{ordinal:02d}.json")
        )
        if saved != expected:
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_preflight_call_external_replay_mismatch"
            )
        validations.append(saved)
    payload = {
        "receipt_version": PREFLIGHT_RECEIPT_VERSION,
        "status": "passed_all_eight_source_free_calls",
        "execution_bundle_sha256": execution_bundle.execution_bundle_sha256,
        "exact_authorization_sha256": authorization.authorization_sha256,
        "calls": validations,
        "call_membership_sha256": hash_canonical([item.validation_sha256 for item in validations]),
        "source_bearing_call_count": 0,
        "extraction_accuracy_authority": False,
    }
    expected_receipt = PreflightReceiptV2.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )
    if receipt != expected_receipt:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_preflight_external_replay_mismatch"
        )
    return receipt


def _inventory_result(
    *,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    row_ordinal: int,
    intent: HostedExactOnceIntentV1,
    authorization: HostedExactOnceCostAuthorizationV1,
    outcome: HostedExactOnceProviderReceiptV1 | HostedExactOnceAmbiguityIncidentV1,
) -> InventoryCallResultV2:
    row = bundle.extraction_inputs.rows[row_ordinal]
    request_spec = bundle.inventory_requests[row_ordinal]
    receipt: MetaSynCandidateInventoryReceiptV2 | None = None
    if isinstance(outcome, HostedExactOnceAmbiguityIncidentV1):
        status: InventoryValidationStatus = "exact_once_terminal_incident"
    elif _is_capacity_failure(outcome.provider_result):
        status = "runtime_capacity_failure"
    elif outcome.provider_result.outcome != "completed":
        status = "provider_runtime_failure"
    else:
        try:
            if not isinstance(outcome.provider_result.parsed_json, Mapping):
                raise ValueError("inventory_not_object")
            receipt = freeze_metasyn_candidate_inventory_receipt_v2(
                row_context_sha256=row.upstream_row_context_sha256,
                projection_v2_sha256=row.projection_v2_sha256,
                allowed_outcome_text_by_id=row.question_surface.allowed_outcome_text_by_id,
                passage_text_by_id={
                    item.passage_id: item.text for item in row.projection_surface.passages
                },
                value=outcome.provider_result.parsed_json,
            )
            status = "inventory_contract_valid"
        except (ValueError, TypeError):
            receipt = None
            status = "inventory_contract_invalid"
    payload = {
        "result_version": INVENTORY_RESULT_VERSION,
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "row_ordinal": row_ordinal,
        "row_key": row.row_key,
        "inventory_request_sha256": request_spec.inventory_request_sha256,
        "terminal": _terminal_ref(intent=intent, authorization=authorization, outcome=outcome),
        "validation_status": status,
        "inventory_receipt": receipt,
        "inventory_receipt_sha256": receipt.receipt_sha256 if receipt else None,
        "authorizes_packet_generation": bool(
            receipt is not None and receipt.status == "candidates_authorized"
        ),
        "runtime_failure_is_not_scientific_abstention": True,
    }
    return InventoryCallResultV2.model_validate(
        {**payload, "result_sha256": hash_canonical(payload)}
    )


def _inventory_result_path(workspace: Path, row_ordinal: int) -> Path:
    return _runtime_paths(workspace)["inventory_results"] / f"row-{row_ordinal:02d}.json"


def _inventory_receipt_path(workspace: Path, row_ordinal: int) -> Path:
    return _runtime_paths(workspace)["inventory_receipts"] / f"row-{row_ordinal:02d}.json"


def _run_inventory_row(
    *,
    workspace: Path,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    row_ordinal: int,
    intent: HostedExactOnceIntentV1,
    authorization: HostedExactOnceCostAuthorizationV1,
    client: HostedClientProtocol,
) -> InventoryCallResultV2:
    paths = _runtime_paths(workspace)
    outcome = execute_hosted_exactly_once(
        workspace=paths["provider"],
        intent=intent,
        authorization=authorization,
        client=client,
    )
    outcome = validate_hosted_exact_once_outcome(
        workspace=paths["provider"], intent=intent, authorization=authorization
    )
    result = _inventory_result(
        bundle=bundle,
        row_ordinal=row_ordinal,
        intent=intent,
        authorization=authorization,
        outcome=outcome,
    )
    if result.inventory_receipt is not None:
        _write_or_replay(_inventory_receipt_path(workspace, row_ordinal), result.inventory_receipt)
    _write_or_replay(_inventory_result_path(workspace, row_ordinal), result)
    return validate_metasyn_passage_inventory_result_v2(
        workspace=workspace,
        execution_bundle=bundle,
        row_ordinal=row_ordinal,
        authorization=authorization,
        intent=intent,
    )


def validate_metasyn_passage_inventory_result_v2(
    *,
    workspace: Path,
    execution_bundle: MetaSynPassageHostedExecutionBundleV2,
    row_ordinal: int,
    authorization: HostedExactOnceCostAuthorizationV1,
    intent: HostedExactOnceIntentV1,
) -> InventoryCallResultV2:
    saved = InventoryCallResultV2.model_validate(
        _read_object(_inventory_result_path(workspace, row_ordinal))
    )
    outcome = validate_hosted_exact_once_outcome(
        workspace=_runtime_paths(workspace)["provider"],
        intent=intent,
        authorization=authorization,
    )
    expected = _inventory_result(
        bundle=execution_bundle,
        row_ordinal=row_ordinal,
        intent=intent,
        authorization=authorization,
        outcome=outcome,
    )
    if saved != expected:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_inventory_result_external_replay_mismatch"
        )
    if saved.inventory_receipt is not None:
        row = execution_bundle.extraction_inputs.rows[row_ordinal]
        receipt = MetaSynCandidateInventoryReceiptV2.model_validate(
            _read_object(_inventory_receipt_path(workspace, row_ordinal))
        )
        receipt = validate_metasyn_candidate_inventory_receipt_v2(
            receipt,
            row_context_sha256=row.upstream_row_context_sha256,
            projection_v2_sha256=row.projection_v2_sha256,
            allowed_outcome_text_by_id=row.question_surface.allowed_outcome_text_by_id,
            passage_text_by_id={
                item.passage_id: item.text for item in row.projection_surface.passages
            },
        )
        if receipt != saved.inventory_receipt:
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_inventory_receipt_external_replay_mismatch"
            )
    elif _inventory_receipt_path(workspace, row_ordinal).exists():
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_unexpected_inventory_receipt"
        )
    return saved


def _inventory_authorization(
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> tuple[list[HostedExactOnceIntentV1], HostedExactOnceCostAuthorizationV1]:
    intents = _inventory_intents(bundle)
    authorization = _freeze_phase_authorization(
        bundle=bundle,
        phase="inventory",
        intents=intents,
        configured_budget_usd_micros=(
            bundle.global_cost_envelope.inventory.cost_ceiling_usd_micros
        ),
    )
    return intents, authorization


def run_metasyn_passage_inventory_smoke_v2(
    *,
    repository_root: Path,
    workspace: Path,
    expected_execution_bundle_sha256: str,
    client: HostedClientProtocol,
) -> InventoryCallResultV2:
    root = _canonical_repository_root(repository_root)
    ws = _canonical_existing_workspace(workspace)
    with _runtime_lock(ws):
        bundle = _load_bundle(
            workspace=ws,
            repository_root=root,
            expected_sha256=expected_execution_bundle_sha256,
        )
        chain = _require_stage(ws, 2)
        intents, authorization = _inventory_authorization(bundle)
        _persist_exact_authorization(
            provider_workspace=_runtime_paths(ws)["provider"],
            authorization=authorization,
        )
        if len(chain) > 3:
            return validate_metasyn_passage_inventory_result_v2(
                workspace=ws,
                execution_bundle=bundle,
                row_ordinal=0,
                authorization=authorization,
                intent=intents[0],
            )
        result = _run_inventory_row(
            workspace=ws,
            bundle=bundle,
            row_ordinal=0,
            intent=intents[0],
            authorization=authorization,
            client=client,
        )
        if result.validation_status != "inventory_contract_valid":
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_inventory_smoke_failed:" + result.validation_status
            )
        _write_checkpoint(
            workspace=ws,
            bundle_sha256=bundle.execution_bundle_sha256,
            stage="inventory_smoke_passed",
            artifact_paths=[
                _inventory_result_path(ws, 0),
                _runtime_paths(ws)["provider"] / "cost-authorizations" / "inventory.json",
            ],
        )
        return result


def _freeze_inventory_ledger(
    *,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    authorization: HostedExactOnceCostAuthorizationV1,
    results: Sequence[InventoryCallResultV2],
) -> InventoryLedgerV2:
    ordered = list(results)
    counts = dict(sorted(Counter(item.validation_status for item in ordered).items()))
    authorized = [item for item in ordered if item.authorizes_packet_generation]
    payload = {
        "ledger_version": INVENTORY_LEDGER_VERSION,
        "status": "all_32_inventory_calls_terminal",
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "exact_authorization_sha256": authorization.authorization_sha256,
        "results": ordered,
        "result_membership_sha256": hash_canonical([item.result_sha256 for item in ordered]),
        "validation_status_counts": counts,
        "packet_authorizing_row_count": len(authorized),
        "authorized_candidate_count": sum(
            len(item.inventory_receipt.inventory.candidates)
            for item in authorized
            if item.inventory_receipt is not None
        ),
        "all_calls_terminal": True,
        "extraction_accuracy_authority": False,
    }
    return InventoryLedgerV2.model_validate({**payload, "ledger_sha256": hash_canonical(payload)})


def run_metasyn_passage_inventory_roster_v2(
    *,
    repository_root: Path,
    workspace: Path,
    expected_execution_bundle_sha256: str,
    client: HostedClientProtocol,
) -> InventoryLedgerV2:
    root = _canonical_repository_root(repository_root)
    ws = _canonical_existing_workspace(workspace)
    with _runtime_lock(ws):
        bundle = _load_bundle(
            workspace=ws,
            repository_root=root,
            expected_sha256=expected_execution_bundle_sha256,
        )
        chain = _require_stage(ws, 3)
        if len(chain) > 4:
            return validate_metasyn_passage_inventory_ledger_v2(
                workspace=ws, execution_bundle=bundle
            )
        intents, authorization = _inventory_authorization(bundle)
        _persist_exact_authorization(
            provider_workspace=_runtime_paths(ws)["provider"],
            authorization=authorization,
        )
        results = [
            validate_metasyn_passage_inventory_result_v2(
                workspace=ws,
                execution_bundle=bundle,
                row_ordinal=0,
                authorization=authorization,
                intent=intents[0],
            )
        ]
        for row_ordinal in range(1, 32):
            results.append(
                _run_inventory_row(
                    workspace=ws,
                    bundle=bundle,
                    row_ordinal=row_ordinal,
                    intent=intents[row_ordinal],
                    authorization=authorization,
                    client=client,
                )
            )
        ledger = _freeze_inventory_ledger(
            bundle=bundle, authorization=authorization, results=results
        )
        _write_or_replay(_runtime_paths(ws)["inventory_ledger"], ledger)
        validate_metasyn_passage_inventory_ledger_v2(workspace=ws, execution_bundle=bundle)
        _write_checkpoint(
            workspace=ws,
            bundle_sha256=bundle.execution_bundle_sha256,
            stage="inventory_roster_terminal",
            artifact_paths=[_runtime_paths(ws)["inventory_ledger"]],
        )
        return ledger


def validate_metasyn_passage_inventory_ledger_v2(
    *,
    workspace: Path,
    execution_bundle: MetaSynPassageHostedExecutionBundleV2,
) -> InventoryLedgerV2:
    ws = _canonical_existing_workspace(workspace)
    saved = InventoryLedgerV2.model_validate(_read_object(_runtime_paths(ws)["inventory_ledger"]))
    intents, authorization = _inventory_authorization(execution_bundle)
    results = [
        validate_metasyn_passage_inventory_result_v2(
            workspace=ws,
            execution_bundle=execution_bundle,
            row_ordinal=index,
            authorization=authorization,
            intent=intents[index],
        )
        for index in range(32)
    ]
    expected = _freeze_inventory_ledger(
        bundle=execution_bundle, authorization=authorization, results=results
    )
    if saved != expected:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_inventory_ledger_external_replay_mismatch"
        )
    expected_result_files = {f"row-{index:02d}.json" for index in range(32)}
    actual_result_files = {
        path.name for path in _runtime_paths(ws)["inventory_results"].glob("*.json")
    }
    if actual_result_files != expected_result_files:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_inventory_result_file_set_mismatch"
        )
    return saved


def _freeze_packet_request(
    *,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    inventory_result: InventoryCallResultV2,
    candidate_index: int,
) -> PacketRequestV2:
    if inventory_result.inventory_receipt is None:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_packet_request_without_inventory_receipt"
        )
    packet_input = freeze_metasyn_packet_candidate_input_v2(
        extraction_inputs=bundle.extraction_inputs,
        row_ordinal=inventory_result.row_ordinal,
        inventory_receipt=inventory_result.inventory_receipt,
        candidate_index=candidate_index,
    )
    accepted_schema = capacity_limited_packet_schema_v2(
        schema_bundle=packet_input.packet_schema_bundle,
        config=bundle.runtime_config,
    )
    accepted_sha = hash_canonical(accepted_schema)
    compiled = compile_anthropic_bounded_schema(
        original_schema=accepted_schema,
        full_acceptance_schema_sha256=accepted_sha,
    )
    request = freeze_anthropic_bounded_request(
        operation="metasyn-passage-packet-v2",
        request_key=(
            f"packet-row-{inventory_result.row_ordinal:02d}-candidate-{candidate_index:02d}"
        ),
        prompt=packet_input.rendered_prompt,
        system=bundle.runtime_config.system_prompt,
        max_output_tokens=bundle.runtime_config.packet_max_output_tokens,
        compiled_schema=compiled,
        config=bundle.anthropic_config,
        schema_kind="packet",
        effect_kind=packet_input.candidate.effect_kind,
        identity=bundle.provider_identity,
    )
    cost_envelope = bundle.packet_row_cost_envelopes[inventory_result.row_ordinal]
    request_cost_micros = _usd_micros(request.cost_ceiling.request_cost_ceiling_usd)
    if (
        request.cost_ceiling.conservative_input_token_ceiling
        > cost_envelope.per_call_input_token_ceiling
        or request.max_output_tokens > cost_envelope.per_call_max_output_tokens
        or request_cost_micros > cost_envelope.per_call_cost_ceiling_usd_micros
    ):
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_dynamic_packet_exceeds_row_envelope"
        )
    payload = {
        "packet_request_version": PACKET_REQUEST_VERSION,
        "row_ordinal": inventory_result.row_ordinal,
        "candidate_index": candidate_index,
        "row_key": inventory_result.row_key,
        "inventory_receipt_sha256": inventory_result.inventory_receipt.receipt_sha256,
        "packet_input": packet_input,
        "packet_input_sha256": packet_input.packet_input_sha256,
        "capacity_limited_schema_sha256": accepted_sha,
        "compiled_schema": compiled,
        "compiled_schema_sha256": compiled.compiled_schema_sha256,
        "request": request,
        "request_sha256": request.request_sha256,
        "row_cost_envelope_sha256": cost_envelope.row_cost_sha256,
        "within_frozen_row_cost_envelope": True,
    }
    return PacketRequestV2.model_validate(
        {**payload, "packet_request_sha256": hash_canonical(payload)}
    )


def _packet_intents(
    *, bundle: MetaSynPassageHostedExecutionBundleV2, requests: Sequence[PacketRequestV2]
) -> list[HostedExactOnceIntentV1]:
    return [
        freeze_hosted_exact_once_intent(
            execution_bundle_sha256=bundle.execution_bundle_sha256,
            phase="packet",
            source_bearing=True,
            context_binding_sha256=item.packet_request_sha256,
            request=item.request,
        )
        for item in requests
    ]


def freeze_metasyn_passage_packet_roster_v2(
    *,
    repository_root: Path,
    workspace: Path,
    expected_execution_bundle_sha256: str,
) -> PacketRosterV2:
    root = _canonical_repository_root(repository_root)
    ws = _canonical_existing_workspace(workspace)
    with _runtime_lock(ws):
        bundle = _load_bundle(
            workspace=ws,
            repository_root=root,
            expected_sha256=expected_execution_bundle_sha256,
        )
        chain = _require_stage(ws, 4)
        paths = _runtime_paths(ws)
        if len(chain) > 5:
            return validate_metasyn_passage_packet_roster_v2(workspace=ws, execution_bundle=bundle)
        inventory_ledger = validate_metasyn_passage_inventory_ledger_v2(
            workspace=ws, execution_bundle=bundle
        )
        requests: list[PacketRequestV2] = []
        for result in inventory_ledger.results:
            if not result.authorizes_packet_generation:
                continue
            assert result.inventory_receipt is not None
            for candidate in result.inventory_receipt.inventory.candidates:
                requests.append(
                    _freeze_packet_request(
                        bundle=bundle,
                        inventory_result=result,
                        candidate_index=candidate.candidate_index,
                    )
                )
        intents = _packet_intents(bundle=bundle, requests=requests)
        authorization = None
        if intents:
            authorization = _freeze_phase_authorization(
                bundle=bundle,
                phase="packet",
                intents=intents,
                configured_budget_usd_micros=(
                    bundle.global_cost_envelope.packet.cost_ceiling_usd_micros
                ),
            )
            _persist_exact_authorization(
                provider_workspace=paths["provider"], authorization=authorization
            )
        payload = {
            "roster_version": PACKET_ROSTER_VERSION,
            "status": "exact_dynamic_roster_frozen_from_replayed_inventories",
            "execution_bundle_sha256": bundle.execution_bundle_sha256,
            "inventory_ledger_sha256": inventory_ledger.ledger_sha256,
            "requests": requests,
            "request_count": len(requests),
            "request_membership_sha256": hash_canonical(
                [item.packet_request_sha256 for item in requests]
            ),
            "exact_authorization": authorization,
            "exact_authorization_sha256": (
                authorization.authorization_sha256 if authorization else None
            ),
            "zero_roster_disposition": None if requests else "not_applicable",
        }
        roster = PacketRosterV2.model_validate(
            {**payload, "roster_sha256": hash_canonical(payload)}
        )
        _write_or_replay(paths["packet_roster"], roster)
        validate_metasyn_passage_packet_roster_v2(workspace=ws, execution_bundle=bundle)
        artifacts = [paths["packet_roster"]]
        if authorization is not None:
            artifacts.append(paths["provider"] / "cost-authorizations" / "packet.json")
        _write_checkpoint(
            workspace=ws,
            bundle_sha256=bundle.execution_bundle_sha256,
            stage="packet_roster_frozen",
            artifact_paths=artifacts,
        )
        return roster


def validate_metasyn_passage_packet_roster_v2(
    *,
    workspace: Path,
    execution_bundle: MetaSynPassageHostedExecutionBundleV2,
) -> PacketRosterV2:
    ws = _canonical_existing_workspace(workspace)
    saved = PacketRosterV2.model_validate(_read_object(_runtime_paths(ws)["packet_roster"]))
    inventory_ledger = validate_metasyn_passage_inventory_ledger_v2(
        workspace=ws, execution_bundle=execution_bundle
    )
    requests: list[PacketRequestV2] = []
    for result in inventory_ledger.results:
        if not result.authorizes_packet_generation:
            continue
        assert result.inventory_receipt is not None
        for candidate in result.inventory_receipt.inventory.candidates:
            request = _freeze_packet_request(
                bundle=execution_bundle,
                inventory_result=result,
                candidate_index=candidate.candidate_index,
            )
            validate_metasyn_packet_candidate_input_v2(
                packet_input=request.packet_input,
                extraction_inputs=execution_bundle.extraction_inputs,
                inventory_receipt=result.inventory_receipt,
            )
            requests.append(request)
    intents = _packet_intents(bundle=execution_bundle, requests=requests)
    authorization = (
        _freeze_phase_authorization(
            bundle=execution_bundle,
            phase="packet",
            intents=intents,
            configured_budget_usd_micros=(
                execution_bundle.global_cost_envelope.packet.cost_ceiling_usd_micros
            ),
        )
        if intents
        else None
    )
    payload = {
        "roster_version": PACKET_ROSTER_VERSION,
        "status": "exact_dynamic_roster_frozen_from_replayed_inventories",
        "execution_bundle_sha256": execution_bundle.execution_bundle_sha256,
        "inventory_ledger_sha256": inventory_ledger.ledger_sha256,
        "requests": requests,
        "request_count": len(requests),
        "request_membership_sha256": hash_canonical(
            [item.packet_request_sha256 for item in requests]
        ),
        "exact_authorization": authorization,
        "exact_authorization_sha256": (
            authorization.authorization_sha256 if authorization else None
        ),
        "zero_roster_disposition": None if requests else "not_applicable",
    }
    expected = PacketRosterV2.model_validate({**payload, "roster_sha256": hash_canonical(payload)})
    if saved != expected:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_packet_roster_external_replay_mismatch"
        )
    if authorization is not None:
        path = _runtime_paths(ws)["provider"] / "cost-authorizations" / "packet.json"
        persisted = HostedExactOnceCostAuthorizationV1.model_validate(_read_object(path))
        if persisted != authorization:
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_packet_authorization_external_replay_mismatch"
            )
    elif (_runtime_paths(ws)["provider"] / "cost-authorizations" / "packet.json").exists():
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_zero_roster_has_authorization"
        )
    return saved


def _protocol_from_packet_input(
    *,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    packet_request: PacketRequestV2,
) -> QuestionProjectionSpecV1:
    """Rebuild the exact generic-role protocol bound by projection lineage."""

    row = bundle.extraction_inputs.rows[packet_request.row_ordinal]
    orientation = bundle.protocol_orientations[packet_request.row_ordinal]
    replayed_protocol = replay_metasyn_question_projection_spec_v2(
        question_surface=row.question_surface
    )
    protocol = orientation.protocol
    if (
        orientation.row_input_sha256 != row.row_input_sha256
        or orientation.question_surface_sha256 != row.question_surface_sha256
        or orientation.question_surface_question_spec_sha256 != row.upstream_question_spec_sha256
        or replayed_protocol != protocol
        or orientation.protocol_question_spec_sha256 != protocol.question_spec_sha256
        or orientation.protocol_projection_spec_sha256 != protocol.projection_spec_sha256
        or orientation.protocol_orientation.protocol_question_spec_sha256
        != protocol.question_spec_sha256
        or orientation.protocol_orientation.protocol_projection_spec_sha256
        != protocol.projection_spec_sha256
        or protocol.question_spec_sha256 != row.projection_v2.lineage_binding.question_spec_sha256
    ):
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_protocol_orientation_binding_mismatch"
        )
    return protocol


def _packet_result_path(workspace: Path, request: PacketRequestV2) -> Path:
    return _runtime_paths(workspace)["packet_results"] / f"{request.request.request_key}.json"


def _packet_grounding_path(workspace: Path, request: PacketRequestV2) -> Path:
    return _runtime_paths(workspace)["packet_grounding"] / f"{request.request.request_key}.json"


def _packet_assembly_path(workspace: Path, request: PacketRequestV2) -> Path:
    return _runtime_paths(workspace)["packet_assembly"] / f"{request.request.request_key}.json"


def _packet_result_from_outcome(
    *,
    workspace: Path,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    packet_request: PacketRequestV2,
    intent: HostedExactOnceIntentV1,
    authorization: HostedExactOnceCostAuthorizationV1,
    outcome: HostedExactOnceProviderReceiptV1 | HostedExactOnceAmbiguityIncidentV1,
) -> PacketCallResultV2:
    grounding: PacketGroundingReceiptV2 | None = None
    assembly: NativePacketAssemblyOutcomeV2 | None = None
    if isinstance(outcome, HostedExactOnceAmbiguityIncidentV1):
        status: PacketValidationStatus = "exact_once_terminal_incident"
    elif _is_capacity_failure(outcome.provider_result):
        status = "runtime_capacity_failure"
    elif outcome.provider_result.outcome != "completed":
        status = "provider_runtime_failure"
    else:
        try:
            if not isinstance(outcome.provider_result.parsed_json, Mapping):
                raise NativePacketGroundingV2Error("packet_grounding_v2_model_outcome_not_object")
            grounding = freeze_passage_packet_grounding_receipt_v2(
                model_outcome=outcome.provider_result.parsed_json,
                candidate=packet_request.packet_input.candidate,
                projection=(
                    bundle.extraction_inputs.rows[packet_request.row_ordinal].projection_v2
                ),
            )
        except (ValueError, TypeError, NativePacketGroundingV2Error):
            status = "grounding_invalid"
        else:
            try:
                protocol = _protocol_from_packet_input(bundle=bundle, packet_request=packet_request)
                assembly = assemble_native_packet_v2(
                    candidate=packet_request.packet_input.candidate,
                    projection=(
                        bundle.extraction_inputs.rows[packet_request.row_ordinal].projection_v2
                    ),
                    protocol=protocol,
                    protocol_orientation=(
                        bundle.protocol_orientations[
                            packet_request.row_ordinal
                        ].protocol_orientation
                    ),
                    analysis_policy=bundle.assembly_analysis_policy,
                    grounding_receipt=grounding,
                )
            except (ValueError, TypeError):
                status = "assembly_invalid"
            else:
                if isinstance(assembly, NativePacketAssemblyCompletedV2):
                    status = "typed_effect_completed"
                elif isinstance(grounding, PacketGroundingAbstentionReceiptV2):
                    status = "grounding_abstained"
                else:
                    status = "assembly_abstained"
    payload = {
        "result_version": PACKET_RESULT_VERSION,
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "packet_request_sha256": packet_request.packet_request_sha256,
        "row_ordinal": packet_request.row_ordinal,
        "candidate_index": packet_request.candidate_index,
        "terminal": _terminal_ref(intent=intent, authorization=authorization, outcome=outcome),
        "validation_status": status,
        "grounding_receipt": grounding,
        "grounding_receipt_sha256": grounding.receipt_sha256 if grounding else None,
        "assembly_receipt": assembly,
        "assembly_receipt_sha256": (assembly.assembly_receipt_sha256 if assembly else None),
        "authorizes_typed_effect": status == "typed_effect_completed",
        "synthesis_input_authority": False,
        "claim_release_authority": False,
        "runtime_failure_is_not_scientific_abstention": True,
    }
    return PacketCallResultV2.model_validate({**payload, "result_sha256": hash_canonical(payload)})


def _run_packet_call(
    *,
    workspace: Path,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    packet_request: PacketRequestV2,
    intent: HostedExactOnceIntentV1,
    authorization: HostedExactOnceCostAuthorizationV1,
    client: HostedClientProtocol,
) -> PacketCallResultV2:
    outcome = execute_hosted_exactly_once(
        workspace=_runtime_paths(workspace)["provider"],
        intent=intent,
        authorization=authorization,
        client=client,
    )
    outcome = validate_hosted_exact_once_outcome(
        workspace=_runtime_paths(workspace)["provider"],
        intent=intent,
        authorization=authorization,
    )
    result = _packet_result_from_outcome(
        workspace=workspace,
        bundle=bundle,
        packet_request=packet_request,
        intent=intent,
        authorization=authorization,
        outcome=outcome,
    )
    if result.grounding_receipt is not None:
        _write_or_replay(
            _packet_grounding_path(workspace, packet_request),
            result.grounding_receipt,
        )
    if result.assembly_receipt is not None:
        _write_or_replay(_packet_assembly_path(workspace, packet_request), result.assembly_receipt)
    _write_or_replay(_packet_result_path(workspace, packet_request), result)
    return validate_metasyn_passage_packet_result_v2(
        workspace=workspace,
        execution_bundle=bundle,
        packet_request=packet_request,
        intent=intent,
        authorization=authorization,
    )


def validate_metasyn_passage_packet_result_v2(
    *,
    workspace: Path,
    execution_bundle: MetaSynPassageHostedExecutionBundleV2,
    packet_request: PacketRequestV2,
    intent: HostedExactOnceIntentV1,
    authorization: HostedExactOnceCostAuthorizationV1,
) -> PacketCallResultV2:
    saved = PacketCallResultV2.model_validate(
        _read_object(_packet_result_path(workspace, packet_request))
    )
    outcome = validate_hosted_exact_once_outcome(
        workspace=_runtime_paths(workspace)["provider"],
        intent=intent,
        authorization=authorization,
    )
    expected = _packet_result_from_outcome(
        workspace=workspace,
        bundle=execution_bundle,
        packet_request=packet_request,
        intent=intent,
        authorization=authorization,
        outcome=outcome,
    )
    if saved != expected:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_packet_result_external_replay_mismatch"
        )
    if saved.grounding_receipt is not None:
        grounding = _GROUNDING_ADAPTER.validate_python(
            _read_object(_packet_grounding_path(workspace, packet_request))
        )
        grounding = validate_passage_packet_grounding_receipt_v2(
            receipt=grounding,
            model_outcome=grounding.model_outcome.model_dump(mode="json"),
            candidate=packet_request.packet_input.candidate,
            projection=(
                execution_bundle.extraction_inputs.rows[packet_request.row_ordinal].projection_v2
            ),
        )
        if grounding != saved.grounding_receipt:
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_grounding_external_replay_mismatch"
            )
    elif _packet_grounding_path(workspace, packet_request).exists():
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_unexpected_grounding_artifact"
        )
    if saved.assembly_receipt is not None:
        assembly = _ASSEMBLY_ADAPTER.validate_python(
            _read_object(_packet_assembly_path(workspace, packet_request))
        )
        protocol = _protocol_from_packet_input(
            bundle=execution_bundle, packet_request=packet_request
        )
        assembly = validate_native_packet_assembly_v2(
            assembly=assembly,
            candidate=packet_request.packet_input.candidate,
            projection=(
                execution_bundle.extraction_inputs.rows[packet_request.row_ordinal].projection_v2
            ),
            protocol=protocol,
            protocol_orientation=(
                execution_bundle.protocol_orientations[
                    packet_request.row_ordinal
                ].protocol_orientation
            ),
            analysis_policy=execution_bundle.assembly_analysis_policy,
            grounding_receipt=saved.grounding_receipt,
        )
        if assembly != saved.assembly_receipt:
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_assembly_external_replay_mismatch"
            )
    elif _packet_assembly_path(workspace, packet_request).exists():
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_unexpected_assembly_artifact"
        )
    return saved


def _smoke_order(roster: PacketRosterV2, priority_row: int) -> list[PacketRequestV2]:
    return sorted(
        roster.requests,
        key=lambda item: (
            0 if item.row_ordinal == priority_row else 1,
            item.row_ordinal,
            item.candidate_index,
        ),
    )


def _freeze_smoke(
    *,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    roster: PacketRosterV2,
    status: Literal["passed", "not_applicable", "failed_gate"],
    ordered_requests: Sequence[PacketRequestV2],
    results: Sequence[PacketCallResultV2],
    completed: PacketCallResultV2 | None,
) -> PacketSmokeReceiptV2:
    payload = {
        "smoke_version": PACKET_SMOKE_VERSION,
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "packet_roster_sha256": roster.roster_sha256,
        "status": status,
        "ordered_smoke_request_keys": [item.request.request_key for item in ordered_requests],
        "attempted_result_sha256s": [item.result_sha256 for item in results],
        "completed_typed_effect_result_sha256": completed.result_sha256 if completed else None,
        "valid_abstention_does_not_pass": True,
        "remaining_packet_calls_permitted": status in {"passed", "not_applicable"},
    }
    return PacketSmokeReceiptV2.model_validate({**payload, "smoke_sha256": hash_canonical(payload)})


def run_metasyn_passage_packet_smoke_v2(
    *,
    repository_root: Path,
    workspace: Path,
    expected_execution_bundle_sha256: str,
    client: HostedClientProtocol,
) -> PacketSmokeReceiptV2:
    root = _canonical_repository_root(repository_root)
    ws = _canonical_existing_workspace(workspace)
    with _runtime_lock(ws):
        bundle = _load_bundle(
            workspace=ws,
            repository_root=root,
            expected_sha256=expected_execution_bundle_sha256,
        )
        chain = _require_stage(ws, 5)
        roster = validate_metasyn_passage_packet_roster_v2(workspace=ws, execution_bundle=bundle)
        paths = _runtime_paths(ws)
        if len(chain) > 6:
            return PacketSmokeReceiptV2.model_validate(_read_object(paths["packet_smoke"]))
        if not roster.requests:
            smoke = _freeze_smoke(
                bundle=bundle,
                roster=roster,
                status="not_applicable",
                ordered_requests=[],
                results=[],
                completed=None,
            )
            _write_or_replay(paths["packet_smoke"], smoke)
            _write_checkpoint(
                workspace=ws,
                bundle_sha256=bundle.execution_bundle_sha256,
                stage="packet_smoke_not_applicable",
                artifact_paths=[paths["packet_smoke"]],
            )
            return smoke
        assert roster.exact_authorization is not None
        ordered = _smoke_order(roster, bundle.runtime_config.packet_smoke_priority_row_ordinal)[
            : bundle.runtime_config.packet_smoke_max_already_authorized_calls
        ]
        intent_by_key = {
            item.request_key: item
            for item in _packet_intents(bundle=bundle, requests=roster.requests)
        }
        results: list[PacketCallResultV2] = []
        completed: PacketCallResultV2 | None = None
        attempted_requests: list[PacketRequestV2] = []
        for request in ordered:
            attempted_requests.append(request)
            result = _run_packet_call(
                workspace=ws,
                bundle=bundle,
                packet_request=request,
                intent=intent_by_key[request.request.request_key],
                authorization=roster.exact_authorization,
                client=client,
            )
            results.append(result)
            if result.authorizes_typed_effect:
                completed = result
                break
        smoke = _freeze_smoke(
            bundle=bundle,
            roster=roster,
            status="passed" if completed else "failed_gate",
            ordered_requests=attempted_requests,
            results=results,
            completed=completed,
        )
        if completed is None:
            _write_or_replay(paths["packet_smoke_attempt"], smoke)
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_packet_smoke_failed_no_typed_effect"
            )
        _write_or_replay(paths["packet_smoke"], smoke)
        _write_checkpoint(
            workspace=ws,
            bundle_sha256=bundle.execution_bundle_sha256,
            stage="packet_smoke_passed",
            artifact_paths=[paths["packet_smoke"]],
        )
        return smoke


def _freeze_packet_ledger(
    *,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    roster: PacketRosterV2,
    smoke: PacketSmokeReceiptV2,
    results: Sequence[PacketCallResultV2],
) -> PacketLedgerV2:
    ordered = sorted(results, key=lambda item: (item.row_ordinal, item.candidate_index))
    payload = {
        "ledger_version": PACKET_LEDGER_VERSION,
        "status": "exact_dynamic_packet_roster_terminal",
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "packet_roster_sha256": roster.roster_sha256,
        "packet_smoke_sha256": smoke.smoke_sha256,
        "results": ordered,
        "result_membership_sha256": hash_canonical([item.result_sha256 for item in ordered]),
        "validation_status_counts": dict(
            sorted(Counter(item.validation_status for item in ordered).items())
        ),
        "typed_effect_count": sum(item.authorizes_typed_effect for item in ordered),
        "all_calls_terminal": True,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return PacketLedgerV2.model_validate({**payload, "ledger_sha256": hash_canonical(payload)})


def run_metasyn_passage_packet_roster_v2(
    *,
    repository_root: Path,
    workspace: Path,
    expected_execution_bundle_sha256: str,
    client: HostedClientProtocol,
) -> PacketLedgerV2:
    root = _canonical_repository_root(repository_root)
    ws = _canonical_existing_workspace(workspace)
    with _runtime_lock(ws):
        bundle = _load_bundle(
            workspace=ws,
            repository_root=root,
            expected_sha256=expected_execution_bundle_sha256,
        )
        chain = _require_stage(ws, 6)
        if len(chain) > 7:
            return validate_metasyn_passage_packet_ledger_v2(workspace=ws, execution_bundle=bundle)
        roster = validate_metasyn_passage_packet_roster_v2(workspace=ws, execution_bundle=bundle)
        smoke = PacketSmokeReceiptV2.model_validate(
            _read_object(_runtime_paths(ws)["packet_smoke"])
        )
        if smoke.status not in {"passed", "not_applicable"}:
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_remaining_packets_blocked_by_smoke"
            )
        results: list[PacketCallResultV2] = []
        if roster.requests:
            assert roster.exact_authorization is not None
            intent_by_key = {
                item.request_key: item
                for item in _packet_intents(bundle=bundle, requests=roster.requests)
            }
            for request in roster.requests:
                results.append(
                    _run_packet_call(
                        workspace=ws,
                        bundle=bundle,
                        packet_request=request,
                        intent=intent_by_key[request.request.request_key],
                        authorization=roster.exact_authorization,
                        client=client,
                    )
                )
        ledger = _freeze_packet_ledger(bundle=bundle, roster=roster, smoke=smoke, results=results)
        _write_or_replay(_runtime_paths(ws)["packet_ledger"], ledger)
        validate_metasyn_passage_packet_ledger_v2(workspace=ws, execution_bundle=bundle)
        _write_checkpoint(
            workspace=ws,
            bundle_sha256=bundle.execution_bundle_sha256,
            stage="packet_roster_terminal",
            artifact_paths=[_runtime_paths(ws)["packet_ledger"]],
        )
        return ledger


def validate_metasyn_passage_packet_ledger_v2(
    *,
    workspace: Path,
    execution_bundle: MetaSynPassageHostedExecutionBundleV2,
) -> PacketLedgerV2:
    ws = _canonical_existing_workspace(workspace)
    roster = validate_metasyn_passage_packet_roster_v2(
        workspace=ws, execution_bundle=execution_bundle
    )
    smoke = PacketSmokeReceiptV2.model_validate(_read_object(_runtime_paths(ws)["packet_smoke"]))
    results: list[PacketCallResultV2] = []
    if roster.requests:
        assert roster.exact_authorization is not None
        intent_by_key = {
            item.request_key: item
            for item in _packet_intents(bundle=execution_bundle, requests=roster.requests)
        }
        for request in roster.requests:
            results.append(
                validate_metasyn_passage_packet_result_v2(
                    workspace=ws,
                    execution_bundle=execution_bundle,
                    packet_request=request,
                    intent=intent_by_key[request.request.request_key],
                    authorization=roster.exact_authorization,
                )
            )
    expected = _freeze_packet_ledger(
        bundle=execution_bundle, roster=roster, smoke=smoke, results=results
    )
    saved = PacketLedgerV2.model_validate(_read_object(_runtime_paths(ws)["packet_ledger"]))
    if saved != expected:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_packet_ledger_external_replay_mismatch"
        )
    expected_files = {item.request.request_key + ".json" for item in roster.requests}
    actual_files = {path.name for path in _runtime_paths(ws)["packet_results"].glob("*.json")}
    if actual_files != expected_files:
        raise MetaSynPassageHostedRuntimeV2Error(
            "metasyn_passage_runtime_v2_packet_result_file_set_mismatch"
        )
    return saved


def _all_terminal_outcomes(
    *,
    workspace: Path,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    roster: PacketRosterV2,
) -> list[HostedExactOnceProviderReceiptV1 | HostedExactOnceAmbiguityIncidentV1]:
    paths = _runtime_paths(workspace)
    outcomes: list[HostedExactOnceProviderReceiptV1 | HostedExactOnceAmbiguityIncidentV1] = []
    preflight_intents = _preflight_intents(bundle)
    preflight_auth = _freeze_phase_authorization(
        bundle=bundle,
        phase="source_free_preflight",
        intents=preflight_intents,
        configured_budget_usd_micros=(
            bundle.global_cost_envelope.source_free_preflight.cost_ceiling_usd_micros
        ),
    )
    outcomes.extend(
        validate_hosted_exact_once_outcome(
            workspace=paths["provider"], intent=item, authorization=preflight_auth
        )
        for item in preflight_intents
    )
    inventory_intents, inventory_auth = _inventory_authorization(bundle)
    outcomes.extend(
        validate_hosted_exact_once_outcome(
            workspace=paths["provider"], intent=item, authorization=inventory_auth
        )
        for item in inventory_intents
    )
    if roster.requests:
        assert roster.exact_authorization is not None
        outcomes.extend(
            validate_hosted_exact_once_outcome(
                workspace=paths["provider"],
                intent=item,
                authorization=roster.exact_authorization,
            )
            for item in _packet_intents(bundle=bundle, requests=roster.requests)
        )
    return outcomes


def _freeze_final_report(
    *,
    workspace: Path,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    inventory: InventoryLedgerV2,
    roster: PacketRosterV2,
    smoke: PacketSmokeReceiptV2,
    packets: PacketLedgerV2,
) -> PrivateYieldReportV2:
    outcomes = _all_terminal_outcomes(workspace=workspace, bundle=bundle, roster=roster)
    receipts = [item for item in outcomes if isinstance(item, HostedExactOnceProviderReceiptV1)]
    usages = [item.provider_result.usage for item in receipts if item.provider_result.usage]
    estimated_costs = [
        item.provider_result.cost.estimated_cost_usd
        for item in receipts
        if item.provider_result.cost.estimated_cost_usd is not None
    ]
    payload = {
        "report_version": FINAL_REPORT_VERSION,
        "status": "complete_yield_only_no_accuracy_or_release_authority",
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "bundle_pipeline_sha256": bundle.bundle_pipeline_sha256,
        "extraction_inputs_sha256": bundle.extraction_inputs_sha256,
        "inventory_ledger_sha256": inventory.ledger_sha256,
        "packet_roster_sha256": roster.roster_sha256,
        "packet_smoke_sha256": smoke.smoke_sha256,
        "packet_ledger_sha256": packets.ledger_sha256,
        "inventory_call_count": 32,
        "packet_call_count": roster.request_count,
        "total_provider_attempts_or_possible_attempts": len(outcomes),
        "source_bearing_attempts_or_possible_attempts": 32 + roster.request_count,
        "typed_effect_count": packets.typed_effect_count,
        "inventory_validation_status_counts": inventory.validation_status_counts,
        "packet_validation_status_counts": packets.validation_status_counts,
        "reported_usage_receipt_count": len(usages),
        "reported_input_tokens": sum(item.input_tokens for item in usages),
        "reported_output_tokens": sum(item.output_tokens for item in usages),
        "reported_estimated_cost_usd_micros": sum(_usd_micros(item) for item in estimated_costs),
        "conservative_attempt_liability_usd_micros": sum(
            (
                item.request_cost_ceiling_usd_micros
                if isinstance(item, HostedExactOnceProviderReceiptV1)
                else next(
                    intent.request_cost_ceiling_usd_micros
                    for intent in (
                        *_preflight_intents(bundle),
                        *_inventory_intents(bundle),
                        *_packet_intents(bundle=bundle, requests=roster.requests),
                    )
                    if intent.request_key == item.request_key
                )
            )
            for item in outcomes
        ),
        "configured_global_cost_ceiling_usd_micros": (
            bundle.global_cost_envelope.cost_ceiling_usd_micros
        ),
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "extraction_accuracy_authority": False,
        "scientific_effectiveness_authority": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return PrivateYieldReportV2.model_validate(
        {**payload, "report_sha256": hash_canonical(payload)}
    )


def finalize_metasyn_passage_hosted_runtime_v2(
    *,
    repository_root: Path,
    workspace: Path,
    expected_execution_bundle_sha256: str,
) -> PrivateYieldReportV2:
    root = _canonical_repository_root(repository_root)
    ws = _canonical_existing_workspace(workspace)
    with _runtime_lock(ws):
        bundle = _load_bundle(
            workspace=ws,
            repository_root=root,
            expected_sha256=expected_execution_bundle_sha256,
        )
        chain = _require_stage(ws, 7)
        if len(chain) > 8:
            return PrivateYieldReportV2.model_validate(
                _read_object(_runtime_paths(ws)["final_report"])
            )
        inventory = validate_metasyn_passage_inventory_ledger_v2(
            workspace=ws, execution_bundle=bundle
        )
        roster = validate_metasyn_passage_packet_roster_v2(workspace=ws, execution_bundle=bundle)
        smoke = PacketSmokeReceiptV2.model_validate(
            _read_object(_runtime_paths(ws)["packet_smoke"])
        )
        packets = validate_metasyn_passage_packet_ledger_v2(workspace=ws, execution_bundle=bundle)
        report = _freeze_final_report(
            workspace=ws,
            bundle=bundle,
            inventory=inventory,
            roster=roster,
            smoke=smoke,
            packets=packets,
        )
        _write_or_replay(_runtime_paths(ws)["final_report"], report)
        _write_checkpoint(
            workspace=ws,
            bundle_sha256=bundle.execution_bundle_sha256,
            stage="finalized",
            artifact_paths=[_runtime_paths(ws)["final_report"]],
        )
        return report


def validate_finalized_metasyn_passage_hosted_runtime_v2(
    *,
    repository_root: Path,
    workspace: Path,
    expected_execution_bundle_sha256: str,
    mark_externally_validated: bool = True,
) -> ExternalValidationReceiptV2:
    """Replay every bundle, stage, exact outcome, grounding and assembly artifact."""

    root = _canonical_repository_root(repository_root)
    ws = _canonical_existing_workspace(workspace)
    with _runtime_lock(ws):
        bundle = _load_bundle(
            workspace=ws,
            repository_root=root,
            expected_sha256=expected_execution_bundle_sha256,
        )
        chain = _require_stage(ws, 8)
        inventory = validate_metasyn_passage_inventory_ledger_v2(
            workspace=ws, execution_bundle=bundle
        )
        roster = validate_metasyn_passage_packet_roster_v2(workspace=ws, execution_bundle=bundle)
        smoke = PacketSmokeReceiptV2.model_validate(
            _read_object(_runtime_paths(ws)["packet_smoke"])
        )
        packets = validate_metasyn_passage_packet_ledger_v2(workspace=ws, execution_bundle=bundle)
        expected_report = _freeze_final_report(
            workspace=ws,
            bundle=bundle,
            inventory=inventory,
            roster=roster,
            smoke=smoke,
            packets=packets,
        )
        saved_report = PrivateYieldReportV2.model_validate(
            _read_object(_runtime_paths(ws)["final_report"])
        )
        if saved_report != expected_report:
            raise MetaSynPassageHostedRuntimeV2Error(
                "metasyn_passage_runtime_v2_final_report_external_replay_mismatch"
            )
        payload = {
            "validation_version": EXTERNAL_VALIDATION_VERSION,
            "status": "all_current_artifacts_externally_replayed",
            "execution_bundle_sha256": bundle.execution_bundle_sha256,
            "final_report_sha256": saved_report.report_sha256,
            "stage_count_before_validation": 9,
            "exact_terminal_outcome_count": 40 + roster.request_count,
            "saved_grounding_receipt_count": sum(
                item.grounding_receipt is not None for item in packets.results
            ),
            "saved_assembly_receipt_count": sum(
                item.assembly_receipt is not None for item in packets.results
            ),
            "provider_calls_made_by_validation": 0,
        }
        receipt = ExternalValidationReceiptV2.model_validate(
            {**payload, "validation_sha256": hash_canonical(payload)}
        )
        path = _runtime_paths(ws)["external_validation"]
        if len(chain) > 9:
            saved = ExternalValidationReceiptV2.model_validate(_read_object(path))
            if saved != receipt:
                raise MetaSynPassageHostedRuntimeV2Error(
                    "metasyn_passage_runtime_v2_external_validation_replay_mismatch"
                )
            return saved
        if mark_externally_validated:
            _write_or_replay(path, receipt)
            _write_checkpoint(
                workspace=ws,
                bundle_sha256=bundle.execution_bundle_sha256,
                stage="externally_validated",
                artifact_paths=[path],
            )
        return receipt


def metasyn_passage_hosted_runtime_status_v2(
    *, repository_root: Path, workspace: Path, expected_execution_bundle_sha256: str
) -> dict[str, Any]:
    root = _canonical_repository_root(repository_root)
    ws = _canonical_existing_workspace(workspace)
    with _runtime_lock(ws):
        bundle = _load_bundle(
            workspace=ws,
            repository_root=root,
            expected_sha256=expected_execution_bundle_sha256,
        )
        chain = _load_stage_chain(ws)
        return {
            "runtime_version": RUNTIME_VERSION,
            "execution_bundle_sha256": bundle.execution_bundle_sha256,
            "current_stage": chain[-1].stage,
            "stage_ordinal": chain[-1].stage_ordinal,
            "checkpoint_sha256": chain[-1].checkpoint_sha256,
            "provider_calls_made": chain[-1].stage_ordinal >= 2,
            "claim_release_authority": False,
        }


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_EXECUTION_WORKSPACE",
    "ExternalValidationReceiptV2",
    "GlobalRunCostAuthorizationV2",
    "InventoryCallResultV2",
    "InventoryLedgerV2",
    "MetaSynPassageHostedRuntimeV2Error",
    "PacketCallResultV2",
    "PacketLedgerV2",
    "PacketRequestV2",
    "PacketRosterV2",
    "PacketSmokeReceiptV2",
    "PreflightReceiptV2",
    "PrivateYieldReportV2",
    "authorize_metasyn_passage_hosted_runtime_v2",
    "finalize_metasyn_passage_hosted_runtime_v2",
    "freeze_metasyn_passage_packet_roster_v2",
    "metasyn_passage_hosted_runtime_status_v2",
    "prepare_metasyn_passage_hosted_runtime_v2",
    "run_metasyn_passage_inventory_roster_v2",
    "run_metasyn_passage_inventory_smoke_v2",
    "run_metasyn_passage_packet_roster_v2",
    "run_metasyn_passage_packet_smoke_v2",
    "run_metasyn_passage_source_free_preflight_v2",
    "validate_finalized_metasyn_passage_hosted_runtime_v2",
    "validate_metasyn_passage_inventory_ledger_v2",
    "validate_metasyn_passage_packet_ledger_v2",
    "validate_metasyn_passage_packet_roster_v2",
    "validate_metasyn_passage_preflight_v2",
]
