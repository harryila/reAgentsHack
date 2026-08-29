"""Exact-wire reuse overlay for the frozen full Evidence Inference comparison.

The ordinary paired runtime remains the authority for intent-before-transport,
pairwise budget admission, receipts, and terminal replay.  This module adds a
separate, hash-bound overlay which may satisfy a target request from an already
validated source attempt only when the complete provider wire call is identical.

Source workspaces are read-only.  A reused request still receives a normal target
intent and receipt, but the overlay records that no provider attempt was made in
the target workspace.  A legacy ambiguous source attempt is inherited as a
continuing failed request and is never retried.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, StrictInt, model_validator

from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    INCIDENT_SANITIZATION_POLICY,
    EvidenceInferenceFableBudgetAuthorizationV1,
    EvidenceInferenceFableCallSurfaceV1,
    EvidenceInferenceFableClientProtocol,
    EvidenceInferenceFableIncidentArtifactV1,
    EvidenceInferenceFableIncidentV1,
    EvidenceInferenceFableIncidentV2,
    EvidenceInferenceFableIntentV1,
    EvidenceInferenceFablePreparedRuntimeV1,
    EvidenceInferenceFableProviderResultV1,
    EvidenceInferenceFableReceiptV1,
    EvidenceInferenceFableTerminalV1,
    execute_evidence_inference_fable_paired_v1,
    validate_evidence_inference_fable_workspace_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_v1 import (
    EvidenceInferenceFableRetrospectivePlanV1,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel

Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]
Count = Annotated[StrictInt, Field(ge=0)]
PositiveCount = Annotated[StrictInt, Field(ge=1)]
Micros = Annotated[StrictInt, Field(ge=0)]

REUSE_DIRECTORY = "full-reuse-v1"
REUSE_PLAN_FILE = "00-adoption-plan.json"
REUSE_TERMINAL_FILE = "02-reuse-terminal.json"
EXPECTED_ADOPTED_RECEIPTS = 20
EXPECTED_INHERITED_AMBIGUITIES = 1
EXPECTED_FULL_REQUESTS = 382


class EvidenceInferenceFableFullReuseError(ValueError):
    """The immutable-source reuse overlay failed closed."""


class _Frozen(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False)


def _self_hash(model: _Frozen, field: str, code: str) -> None:
    if getattr(model, field) != hash_canonical(
        model.model_dump(mode="json", exclude={field})
    ):
        raise ValueError(code)


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceInferenceFableFullReuseError("fable_reuse_artifact_unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceInferenceFableFullReuseError(
            "fable_reuse_artifact_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise EvidenceInferenceFableFullReuseError(
            "fable_reuse_artifact_not_object"
        )
    return value


@contextmanager
def _reuse_lock(workspace: Path) -> Any:
    descriptor = os.open(
        workspace / ".full-reuse-v1.lock",
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "a+b", closefd=True) as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
    finally:
        pass


SourceSlot = Literal["poisoned_pilot_v1", "recovery_pilot_v2"]
AdoptionKind = Literal["terminal_receipt", "inherited_ambiguous_failure"]


@dataclass(frozen=True)
class EvidenceInferenceFableReuseSourceV1:
    """A caller-supplied immutable source; its path is never serialized."""

    slot: SourceSlot
    plan: EvidenceInferenceFableRetrospectivePlanV1
    workspace: Path


class EvidenceInferenceFableReuseSourceBindingV1(_Frozen):
    slot: SourceSlot
    plan_sha256: Sha256
    prepared_sha256: Sha256
    authorization_sha256: Sha256
    terminal_sha256: Sha256
    terminal_status: Literal["completed", "terminal_ambiguous_attempt_poison"]
    source_paths_serialized: Literal[False] = False
    source_workspace_mutation_permitted: Literal[False] = False


class EvidenceInferenceFableReuseEntryV1(_Frozen):
    entry_version: Literal["evidence-inference-fable-full-reuse-entry-v1"] = (
        "evidence-inference-fable-full-reuse-entry-v1"
    )
    adoption_kind: AdoptionKind
    target_execution_index: Count
    target_request_key: str
    target_surface_sha256: Sha256
    wire_call_sha256: Sha256
    locked_question_count: PositiveCount
    source_slot: SourceSlot
    source_plan_sha256: Sha256
    source_prepared_sha256: Sha256
    source_authorization_sha256: Sha256
    source_terminal_sha256: Sha256
    source_intent_sha256: Sha256
    source_request_key: str
    source_surface_sha256: Sha256
    source_receipt_sha256: Sha256 | None
    source_provider_result_sha256: Sha256 | None
    source_incident_sha256: Sha256 | None
    source_charged_cost_usd_micros: PositiveCount
    source_retry_permitted: Literal[False] = False
    target_provider_attempts_permitted_for_entry: Literal[0] = 0
    entry_sha256: Sha256

    @model_validator(mode="after")
    def validate_entry(self) -> EvidenceInferenceFableReuseEntryV1:
        receipt_shape = (
            self.source_receipt_sha256 is not None
            and self.source_provider_result_sha256 is not None
            and self.source_incident_sha256 is None
        )
        ambiguity_shape = (
            self.source_receipt_sha256 is None
            and self.source_provider_result_sha256 is None
            and self.source_incident_sha256 is not None
        )
        if (self.adoption_kind == "terminal_receipt") != receipt_shape or (
            self.adoption_kind == "inherited_ambiguous_failure"
        ) != ambiguity_shape:
            raise ValueError("fable_reuse_entry_source_shape_invalid")
        if Path(self.target_request_key).name != self.target_request_key or Path(
            self.source_request_key
        ).name != self.source_request_key:
            raise ValueError("fable_reuse_entry_request_key_unsafe")
        _self_hash(self, "entry_sha256", "fable_reuse_entry_hash_mismatch")
        return self


class EvidenceInferenceFableFullReusePlanV1(_Frozen):
    plan_version: Literal["evidence-inference-fable-full-reuse-plan-v1"] = (
        "evidence-inference-fable-full-reuse-plan-v1"
    )
    full_plan_sha256: Sha256
    full_prepared_sha256: Sha256
    full_authorization_sha256: Sha256
    configured_total_budget_usd_micros: PositiveCount
    full_request_count: Literal[382] = EXPECTED_FULL_REQUESTS
    source_bindings: list[EvidenceInferenceFableReuseSourceBindingV1]
    entries: list[EvidenceInferenceFableReuseEntryV1]
    adopted_terminal_receipt_count: Literal[20] = EXPECTED_ADOPTED_RECEIPTS
    inherited_ambiguous_failure_count: Literal[1] = EXPECTED_INHERITED_AMBIGUITIES
    maximum_new_provider_attempt_count: Literal[361] = (
        EXPECTED_FULL_REQUESTS
        - EXPECTED_ADOPTED_RECEIPTS
        - EXPECTED_INHERITED_AMBIGUITIES
    )
    exact_wire_match_required: Literal[True] = True
    source_workspaces_immutable: Literal[True] = True
    inherited_ambiguity_retry_permitted: Literal[False] = False
    labels_opened: Literal[False] = False
    provider_calls_made_while_planning: Literal[0] = 0
    scientific_claim_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    plan_sha256: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> EvidenceInferenceFableFullReusePlanV1:
        if (
            [binding.slot for binding in self.source_bindings]
            != ["poisoned_pilot_v1", "recovery_pilot_v2"]
            or [entry.target_execution_index for entry in self.entries]
            != sorted(entry.target_execution_index for entry in self.entries)
            or len({entry.target_request_key for entry in self.entries})
            != len(self.entries)
            or len({entry.wire_call_sha256 for entry in self.entries})
            != len(self.entries)
            or sum(
                entry.adoption_kind == "terminal_receipt" for entry in self.entries
            )
            != self.adopted_terminal_receipt_count
            or sum(
                entry.adoption_kind == "inherited_ambiguous_failure"
                for entry in self.entries
            )
            != self.inherited_ambiguous_failure_count
            or len(self.entries) + self.maximum_new_provider_attempt_count
            != self.full_request_count
        ):
            raise ValueError("fable_reuse_plan_roster_invalid")
        _self_hash(self, "plan_sha256", "fable_reuse_plan_hash_mismatch")
        return self


class EvidenceInferenceFableReuseRecordV1(_Frozen):
    record_version: Literal["evidence-inference-fable-full-reuse-record-v1"] = (
        "evidence-inference-fable-full-reuse-record-v1"
    )
    adoption_plan_sha256: Sha256
    entry_sha256: Sha256
    adoption_kind: AdoptionKind
    source_slot: SourceSlot
    source_terminal_sha256: Sha256
    source_intent_sha256: Sha256
    source_receipt_sha256: Sha256 | None
    source_provider_result_sha256: Sha256 | None
    source_incident_sha256: Sha256 | None
    target_authorization_sha256: Sha256
    target_request_key: str
    target_surface_sha256: Sha256
    wire_call_sha256: Sha256
    target_intent_sha256: Sha256
    target_provider_result_sha256: Sha256
    expected_target_receipt_sha256: Sha256
    expected_target_incident_sha256: Sha256 | None
    target_provider_attempt_count: Literal[0] = 0
    source_attempt_retry_permitted: Literal[False] = False
    locked_questions_scored_incorrect: Count
    charged_cost_usd_micros: PositiveCount
    record_sha256: Sha256

    @model_validator(mode="after")
    def validate_record(self) -> EvidenceInferenceFableReuseRecordV1:
        if (
            (self.adoption_kind == "terminal_receipt")
            != (
                self.source_receipt_sha256 is not None
                and self.source_provider_result_sha256 is not None
                and self.source_incident_sha256 is None
                and self.expected_target_incident_sha256 is None
            )
            or (self.adoption_kind == "inherited_ambiguous_failure")
            != (
                self.source_receipt_sha256 is None
                and self.source_provider_result_sha256 is None
                and self.source_incident_sha256 is not None
                and self.expected_target_incident_sha256 is not None
            )
        ):
            raise ValueError("fable_reuse_record_shape_invalid")
        _self_hash(self, "record_sha256", "fable_reuse_record_hash_mismatch")
        return self


class EvidenceInferenceFableFullReuseTerminalV1(_Frozen):
    terminal_version: Literal["evidence-inference-fable-full-reuse-terminal-v1"] = (
        "evidence-inference-fable-full-reuse-terminal-v1"
    )
    adoption_plan_sha256: Sha256
    target_runtime_terminal_sha256: Sha256
    target_runtime_status: Literal[
        "completed",
        "clean_budget_exhaustion_before_next_pair",
        "terminal_ambiguous_attempt_poison",
    ]
    target_completed_request_count: Count
    realized_adopted_terminal_receipt_count: Count
    realized_inherited_ambiguous_failure_count: Count
    new_provider_attempt_count: Count
    maximum_new_provider_attempt_count: Literal[361] = 361
    target_accounted_spend_usd_micros: Micros
    adopted_source_accounted_spend_usd_micros: Micros
    new_provider_accounted_spend_usd_micros: Micros
    source_provider_attempts_reused: Count
    inherited_ambiguous_attempts_retried: Literal[0] = 0
    target_provider_attempts_for_adopted_entries: Literal[0] = 0
    full_population_score_permitted: bool
    scoring_requires_this_reuse_terminal: Literal[True] = True
    scientific_claim_authority: Literal[False] = False
    confirmatory_gepa_improvement_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    terminal_sha256: Sha256

    @model_validator(mode="after")
    def validate_terminal(self) -> EvidenceInferenceFableFullReuseTerminalV1:
        if (
            self.new_provider_attempt_count > self.maximum_new_provider_attempt_count
            or self.source_provider_attempts_reused
            != self.realized_adopted_terminal_receipt_count
            + self.realized_inherited_ambiguous_failure_count
            or self.target_accounted_spend_usd_micros
            != self.adopted_source_accounted_spend_usd_micros
            + self.new_provider_accounted_spend_usd_micros
            or self.full_population_score_permitted
            != (
                self.target_runtime_status == "completed"
                and self.realized_adopted_terminal_receipt_count
                == EXPECTED_ADOPTED_RECEIPTS
                and self.realized_inherited_ambiguous_failure_count
                == EXPECTED_INHERITED_AMBIGUITIES
                and self.new_provider_attempt_count
                == self.maximum_new_provider_attempt_count
            )
        ):
            raise ValueError("fable_reuse_terminal_counts_invalid")
        _self_hash(self, "terminal_sha256", "fable_reuse_terminal_hash_mismatch")
        return self


@dataclass(frozen=True)
class _SourceState:
    source: EvidenceInferenceFableReuseSourceV1
    prepared: EvidenceInferenceFablePreparedRuntimeV1
    authorization: EvidenceInferenceFableBudgetAuthorizationV1
    terminal: EvidenceInferenceFableTerminalV1
    intents: Mapping[str, EvidenceInferenceFableIntentV1]
    receipts: Mapping[str, EvidenceInferenceFableReceiptV1]
    incidents: Mapping[str, EvidenceInferenceFableIncidentArtifactV1]


def _artifact_map(path: Path, model: Any) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.is_symlink() or not path.is_dir():
        raise EvidenceInferenceFableFullReuseError(
            "fable_reuse_source_artifact_directory_unsafe"
        )
    result: dict[str, Any] = {}
    for artifact in path.iterdir():
        if artifact.is_symlink() or not artifact.is_file() or artifact.suffix != ".json":
            raise EvidenceInferenceFableFullReuseError(
                "fable_reuse_source_artifact_extra"
            )
        result[artifact.stem] = model.model_validate(_read_object(artifact))
    return result


def _incident_map(path: Path) -> dict[str, EvidenceInferenceFableIncidentArtifactV1]:
    if not path.exists():
        return {}
    if path.is_symlink() or not path.is_dir():
        raise EvidenceInferenceFableFullReuseError(
            "fable_reuse_source_incident_directory_unsafe"
        )
    result: dict[str, EvidenceInferenceFableIncidentArtifactV1] = {}
    for artifact in path.iterdir():
        payload = _read_object(artifact)
        version = payload.get("incident_version")
        if version == "evidence-inference-fable-incident-v1":
            incident: EvidenceInferenceFableIncidentArtifactV1 = (
                EvidenceInferenceFableIncidentV1.model_validate(payload)
            )
        elif version == "evidence-inference-fable-incident-v2":
            incident = EvidenceInferenceFableIncidentV2.model_validate(payload)
        else:
            raise EvidenceInferenceFableFullReuseError(
                "fable_reuse_source_incident_version_unknown"
            )
        result[artifact.stem] = incident
    return result


def _load_source(source: EvidenceInferenceFableReuseSourceV1) -> _SourceState:
    root = source.workspace
    if root.is_symlink() or not root.is_dir():
        raise EvidenceInferenceFableFullReuseError("fable_reuse_source_workspace_unsafe")
    terminal = validate_evidence_inference_fable_workspace_v1(
        workspace=root, plan=source.plan
    )
    prepared = EvidenceInferenceFablePreparedRuntimeV1.model_validate(
        _read_object(root / "00-prepared.json")
    )
    authorization = EvidenceInferenceFableBudgetAuthorizationV1.model_validate(
        _read_object(root / "01-authorization.json")
    )
    intents = _artifact_map(root / "intents", EvidenceInferenceFableIntentV1)
    receipts = _artifact_map(root / "receipts", EvidenceInferenceFableReceiptV1)
    incidents = _incident_map(root / "incidents")
    return _SourceState(
        source=source,
        prepared=prepared,
        authorization=authorization,
        terminal=terminal,
        intents=intents,
        receipts=receipts,
        incidents=incidents,
    )


def _entry_payload(
    *,
    target_index: int,
    target_surface: EvidenceInferenceFableCallSurfaceV1,
    state: _SourceState,
    source_intent: EvidenceInferenceFableIntentV1,
    source_receipt: EvidenceInferenceFableReceiptV1 | None,
    source_incident: EvidenceInferenceFableIncidentV1 | None,
) -> dict[str, Any]:
    if (source_receipt is None) == (source_incident is None):
        raise EvidenceInferenceFableFullReuseError(
            "fable_reuse_source_candidate_shape_invalid"
        )
    source_surface = source_intent.surface
    exact_wire_identity = (
        "claude-fable-5",
        source_surface.effort,
        source_surface.service_tier,
        source_surface.max_output_tokens,
        source_surface.system,
        source_surface.prompt,
        source_surface.wire_schema,
    )
    target_wire_identity = (
        "claude-fable-5",
        target_surface.effort,
        target_surface.service_tier,
        target_surface.max_output_tokens,
        target_surface.system,
        target_surface.prompt,
        target_surface.wire_schema,
    )
    if (
        source_surface.wire_call_sha256 != target_surface.wire_call_sha256
        or exact_wire_identity != target_wire_identity
        or source_surface.locked_question_count != target_surface.locked_question_count
    ):
        raise EvidenceInferenceFableFullReuseError(
            "fable_reuse_exact_wire_or_question_count_mismatch"
        )
    result = None if source_receipt is None else source_receipt.provider_result
    incident = source_incident
    return {
        "entry_version": "evidence-inference-fable-full-reuse-entry-v1",
        "adoption_kind": (
            "terminal_receipt"
            if source_receipt is not None
            else "inherited_ambiguous_failure"
        ),
        "target_execution_index": target_index,
        "target_request_key": target_surface.request_key,
        "target_surface_sha256": target_surface.surface_sha256,
        "wire_call_sha256": target_surface.wire_call_sha256,
        "locked_question_count": target_surface.locked_question_count,
        "source_slot": state.source.slot,
        "source_plan_sha256": state.source.plan.plan_sha256,
        "source_prepared_sha256": state.prepared.prepared_sha256,
        "source_authorization_sha256": state.authorization.authorization_sha256,
        "source_terminal_sha256": state.terminal.terminal_sha256,
        "source_intent_sha256": source_intent.intent_sha256,
        "source_request_key": source_intent.request_key,
        "source_surface_sha256": source_intent.surface.surface_sha256,
        "source_receipt_sha256": (
            None if source_receipt is None else source_receipt.receipt_sha256
        ),
        "source_provider_result_sha256": (
            None if result is None else result.result_sha256
        ),
        "source_incident_sha256": (
            None if incident is None else incident.incident_sha256
        ),
        "source_charged_cost_usd_micros": (
            result.charged_cost_usd_micros
            if result is not None
            else incident.charged_cost_usd_micros
        ),
        "source_retry_permitted": False,
        "target_provider_attempts_permitted_for_entry": 0,
    }


def freeze_evidence_inference_fable_full_reuse_plan_v1(
    *,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    full_prepared: EvidenceInferenceFablePreparedRuntimeV1,
    full_authorization: EvidenceInferenceFableBudgetAuthorizationV1,
    sources: list[EvidenceInferenceFableReuseSourceV1],
) -> EvidenceInferenceFableFullReusePlanV1:
    """Freeze the only exact-wire overlaps from validated immutable sources."""

    if (
        full_plan.mode != "full_paired"
        or full_prepared.retrospective_plan_sha256 != full_plan.plan_sha256
        or full_authorization.prepared_sha256 != full_prepared.prepared_sha256
        or len(full_prepared.surfaces) != EXPECTED_FULL_REQUESTS
        or [source.slot for source in sources]
        != ["poisoned_pilot_v1", "recovery_pilot_v2"]
    ):
        raise EvidenceInferenceFableFullReuseError(
            "fable_reuse_full_or_source_binding_invalid"
        )
    states = [_load_source(source) for source in sources]
    candidates: dict[
        str,
        tuple[
            _SourceState,
            EvidenceInferenceFableIntentV1,
            EvidenceInferenceFableReceiptV1 | None,
            EvidenceInferenceFableIncidentV1 | None,
        ],
    ] = {}
    for state in states:
        for request_key, intent in state.intents.items():
            receipt = state.receipts.get(request_key)
            incident_artifact = state.incidents.get(request_key)
            source_incident = (
                incident_artifact
                if isinstance(incident_artifact, EvidenceInferenceFableIncidentV1)
                else None
            )
            if receipt is None and source_incident is None:
                continue
            wire = intent.surface.wire_call_sha256
            if wire in candidates:
                raise EvidenceInferenceFableFullReuseError(
                    "fable_reuse_duplicate_source_wire_candidate"
                )
            candidates[wire] = (state, intent, receipt, source_incident)

    entries: list[EvidenceInferenceFableReuseEntryV1] = []
    for index, surface in enumerate(full_prepared.surfaces):
        candidate = candidates.get(surface.wire_call_sha256)
        if candidate is None:
            continue
        state, intent, receipt, incident = candidate
        payload = _entry_payload(
            target_index=index,
            target_surface=surface,
            state=state,
            source_intent=intent,
            source_receipt=receipt,
            source_incident=incident,
        )
        entries.append(
            EvidenceInferenceFableReuseEntryV1.model_validate(
                {**payload, "entry_sha256": hash_canonical(payload)}
            )
        )

    bindings = [
        EvidenceInferenceFableReuseSourceBindingV1(
            slot=state.source.slot,
            plan_sha256=state.source.plan.plan_sha256,
            prepared_sha256=state.prepared.prepared_sha256,
            authorization_sha256=state.authorization.authorization_sha256,
            terminal_sha256=state.terminal.terminal_sha256,
            terminal_status=state.terminal.status,
            source_paths_serialized=False,
            source_workspace_mutation_permitted=False,
        )
        for state in states
    ]
    payload = {
        "plan_version": "evidence-inference-fable-full-reuse-plan-v1",
        "full_plan_sha256": full_plan.plan_sha256,
        "full_prepared_sha256": full_prepared.prepared_sha256,
        "full_authorization_sha256": full_authorization.authorization_sha256,
        "configured_total_budget_usd_micros": (
            full_authorization.configured_total_budget_usd_micros
        ),
        "full_request_count": EXPECTED_FULL_REQUESTS,
        "source_bindings": bindings,
        "entries": entries,
        "adopted_terminal_receipt_count": EXPECTED_ADOPTED_RECEIPTS,
        "inherited_ambiguous_failure_count": EXPECTED_INHERITED_AMBIGUITIES,
        "maximum_new_provider_attempt_count": 361,
        "exact_wire_match_required": True,
        "source_workspaces_immutable": True,
        "inherited_ambiguity_retry_permitted": False,
        "labels_opened": False,
        "provider_calls_made_while_planning": 0,
        "scientific_claim_authority": False,
        "claim_release_authority": False,
    }
    try:
        return EvidenceInferenceFableFullReusePlanV1.model_validate(
            {**payload, "plan_sha256": hash_canonical(payload)}
        )
    except ValueError as exc:
        raise EvidenceInferenceFableFullReuseError(
            "fable_reuse_expected_20_receipts_1_ambiguity_361_new_calls"
        ) from exc


def prepare_evidence_inference_fable_full_reuse_v1(
    *, workspace: Path, adoption_plan: EvidenceInferenceFableFullReusePlanV1
) -> None:
    """Install only the label-free plan sidecar in an unstarted target workspace."""

    if workspace.is_symlink() or not workspace.is_dir():
        raise EvidenceInferenceFableFullReuseError("fable_reuse_target_workspace_unsafe")
    prepared = EvidenceInferenceFablePreparedRuntimeV1.model_validate(
        _read_object(workspace / "00-prepared.json")
    )
    authorization = EvidenceInferenceFableBudgetAuthorizationV1.model_validate(
        _read_object(workspace / "01-authorization.json")
    )
    if (
        prepared.prepared_sha256 != adoption_plan.full_prepared_sha256
        or authorization.authorization_sha256
        != adoption_plan.full_authorization_sha256
        or (workspace / "02-terminal.json").exists()
        or any(
            (workspace / name).exists() for name in ("intents", "receipts", "incidents")
        )
    ):
        raise EvidenceInferenceFableFullReuseError(
            "fable_reuse_target_not_fresh_or_binding_mismatch"
        )
    root = workspace / REUSE_DIRECTORY
    plan_path = root / REUSE_PLAN_FILE
    if root.exists():
        if root.is_symlink() or not root.is_dir() or not plan_path.is_file():
            raise EvidenceInferenceFableFullReuseError(
                "fable_reuse_directory_replay_unsafe"
            )
        archived = EvidenceInferenceFableFullReusePlanV1.model_validate(
            _read_object(plan_path)
        )
        if archived != adoption_plan:
            raise EvidenceInferenceFableFullReuseError(
                "fable_reuse_plan_replay_mismatch"
            )
        return
    root.mkdir(mode=0o700)
    (root / "records").mkdir(mode=0o700)
    atomic_write_json(plan_path, adoption_plan)


def _target_liability(
    *,
    authorization: EvidenceInferenceFableBudgetAuthorizationV1,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    entry: EvidenceInferenceFableReuseEntryV1,
) -> int:
    if authorization.liability_basis == "certified_provider_token_count":
        try:
            return authorization.certified_request_liabilities_usd_micros[
                entry.target_request_key
            ]
        except KeyError as exc:
            raise EvidenceInferenceFableFullReuseError(
                "fable_reuse_target_liability_missing"
            ) from exc
    return full_plan.roster[
        entry.target_execution_index
    ].cost.full_context_hard_liability_usd_micros


@dataclass(frozen=True)
class _DerivedAdoption:
    result: EvidenceInferenceFableProviderResultV1
    receipt: EvidenceInferenceFableReceiptV1
    incident: EvidenceInferenceFableIncidentV2 | None
    record: EvidenceInferenceFableReuseRecordV1


def _derive_adoption(
    *,
    entry: EvidenceInferenceFableReuseEntryV1,
    adoption_plan: EvidenceInferenceFableFullReusePlanV1,
    target_intent: EvidenceInferenceFableIntentV1,
    target_surface: EvidenceInferenceFableCallSurfaceV1,
    target_authorization: EvidenceInferenceFableBudgetAuthorizationV1,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    source_states: Mapping[SourceSlot, _SourceState],
) -> _DerivedAdoption:
    state = source_states[entry.source_slot]
    source_intent = state.intents.get(entry.source_request_key)
    if (
        source_intent is None
        or source_intent.intent_sha256 != entry.source_intent_sha256
        or target_intent.request_key != entry.target_request_key
        or target_intent.surface != target_surface
        or target_surface.surface_sha256 != entry.target_surface_sha256
        or target_surface.wire_call_sha256 != entry.wire_call_sha256
        or source_intent.surface.wire_call_sha256 != entry.wire_call_sha256
        or target_intent.authorization_sha256
        != target_authorization.authorization_sha256
    ):
        raise EvidenceInferenceFableFullReuseError(
            "fable_reuse_target_or_source_intent_binding_mismatch"
        )
    target_incident: EvidenceInferenceFableIncidentV2 | None = None
    if entry.adoption_kind == "terminal_receipt":
        source_receipt = state.receipts.get(entry.source_request_key)
        if (
            source_receipt is None
            or source_receipt.receipt_sha256 != entry.source_receipt_sha256
            or source_receipt.provider_result.result_sha256
            != entry.source_provider_result_sha256
        ):
            raise EvidenceInferenceFableFullReuseError(
                "fable_reuse_source_receipt_binding_mismatch"
            )
        source_payload = source_receipt.provider_result.model_dump(
            mode="json", exclude={"request_key", "surface_sha256", "result_sha256"}
        )
        result_payload = {
            "request_key": target_surface.request_key,
            "surface_sha256": target_surface.surface_sha256,
            **source_payload,
        }
        result = EvidenceInferenceFableProviderResultV1.model_validate(
            {**result_payload, "result_sha256": hash_canonical(result_payload)}
        )
    else:
        source_incident = state.incidents.get(entry.source_request_key)
        liability = _target_liability(
            authorization=target_authorization,
            full_plan=full_plan,
            entry=entry,
        )
        if (
            not isinstance(source_incident, EvidenceInferenceFableIncidentV1)
            or source_incident.incident_sha256 != entry.source_incident_sha256
            or source_incident.retry_permitted
            or source_incident.charged_cost_usd_micros != liability
            or entry.source_charged_cost_usd_micros != liability
        ):
            raise EvidenceInferenceFableFullReuseError(
                "fable_reuse_source_ambiguity_or_liability_mismatch"
            )
        result_payload = {
            "result_version": "evidence-inference-fable-provider-result-v1",
            "request_key": target_surface.request_key,
            "surface_sha256": target_surface.surface_sha256,
            "transport_attempt_count": 1,
            "sdk_retry_count": 0,
            "outcome": "failed",
            "response_id": None,
            "response_model": None,
            "parsed_json": None,
            "input_tokens": None,
            "output_tokens": None,
            "reported_cost_usd_micros": None,
            "charged_cost_usd_micros": liability,
            "cost_basis": "unknown_usage_hard_liability",
            "response_text_sha256": None,
            "failure_code": "provider_call_raised_after_durable_intent",
        }
        result = EvidenceInferenceFableProviderResultV1.model_validate(
            {**result_payload, "result_sha256": hash_canonical(result_payload)}
        )
        incident_payload = {
            "incident_version": "evidence-inference-fable-incident-v2",
            "status": "failed_request_archived_continue",
            "kind": "provider_call_raised_after_durable_intent",
            "intent_sha256": target_intent.intent_sha256,
            "request_key": target_surface.request_key,
            "charged_cost_usd_micros": liability,
            "cost_basis": "unknown_usage_hard_liability",
            "retry_permitted": False,
            "sanitization_policy": INCIDENT_SANITIZATION_POLICY,
            "exception_type": "InheritedSourceAmbiguousAttempt",
            "http_status_code": None,
            "provider_request_id": None,
            "message_redacted": (
                "Inherited exact-wire ambiguity; provider call was not retried."
            ),
            "message_was_truncated": False,
            "derived_provider_result_sha256": result.result_sha256,
        }
        target_incident = EvidenceInferenceFableIncidentV2.model_validate(
            {
                **incident_payload,
                "incident_sha256": hash_canonical(incident_payload),
            }
        )
    receipt_payload = {
        "receipt_version": "evidence-inference-fable-receipt-v1",
        "intent_sha256": target_intent.intent_sha256,
        "request_key": target_surface.request_key,
        "provider_result": result,
        "locked_question_count": target_surface.locked_question_count,
        "locked_questions_scored_incorrect": (
            target_surface.locked_question_count if result.outcome == "failed" else 0
        ),
    }
    receipt = EvidenceInferenceFableReceiptV1.model_validate(
        {**receipt_payload, "receipt_sha256": hash_canonical(receipt_payload)}
    )
    record_payload = {
        "record_version": "evidence-inference-fable-full-reuse-record-v1",
        "adoption_plan_sha256": adoption_plan.plan_sha256,
        "entry_sha256": entry.entry_sha256,
        "adoption_kind": entry.adoption_kind,
        "source_slot": entry.source_slot,
        "source_terminal_sha256": entry.source_terminal_sha256,
        "source_intent_sha256": entry.source_intent_sha256,
        "source_receipt_sha256": entry.source_receipt_sha256,
        "source_provider_result_sha256": entry.source_provider_result_sha256,
        "source_incident_sha256": entry.source_incident_sha256,
        "target_authorization_sha256": target_authorization.authorization_sha256,
        "target_request_key": entry.target_request_key,
        "target_surface_sha256": entry.target_surface_sha256,
        "wire_call_sha256": entry.wire_call_sha256,
        "target_intent_sha256": target_intent.intent_sha256,
        "target_provider_result_sha256": result.result_sha256,
        "expected_target_receipt_sha256": receipt.receipt_sha256,
        "expected_target_incident_sha256": (
            None if target_incident is None else target_incident.incident_sha256
        ),
        "target_provider_attempt_count": 0,
        "source_attempt_retry_permitted": False,
        "locked_questions_scored_incorrect": (
            receipt.locked_questions_scored_incorrect
        ),
        "charged_cost_usd_micros": result.charged_cost_usd_micros,
    }
    record = EvidenceInferenceFableReuseRecordV1.model_validate(
        {**record_payload, "record_sha256": hash_canonical(record_payload)}
    )
    return _DerivedAdoption(
        result=result, receipt=receipt, incident=target_incident, record=record
    )


class _ReuseAwareClient:
    def __init__(
        self,
        *,
        workspace: Path,
        full_plan: EvidenceInferenceFableRetrospectivePlanV1,
        prepared: EvidenceInferenceFablePreparedRuntimeV1,
        authorization: EvidenceInferenceFableBudgetAuthorizationV1,
        adoption_plan: EvidenceInferenceFableFullReusePlanV1,
        source_states: Mapping[SourceSlot, _SourceState],
        delegate: EvidenceInferenceFableClientProtocol,
    ) -> None:
        self.workspace = workspace
        self.full_plan = full_plan
        self.prepared = prepared
        self.authorization = authorization
        self.adoption_plan = adoption_plan
        self.source_states = source_states
        self.delegate = delegate
        self.entries = {entry.target_request_key: entry for entry in adoption_plan.entries}

    def generate(
        self, surface: EvidenceInferenceFableCallSurfaceV1
    ) -> EvidenceInferenceFableProviderResultV1:
        entry = self.entries.get(surface.request_key)
        if entry is None:
            return self.delegate.generate(surface)
        intent = EvidenceInferenceFableIntentV1.model_validate(
            _read_object(self.workspace / "intents" / f"{surface.request_key}.json")
        )
        derived = _derive_adoption(
            entry=entry,
            adoption_plan=self.adoption_plan,
            target_intent=intent,
            target_surface=surface,
            target_authorization=self.authorization,
            full_plan=self.full_plan,
            source_states=self.source_states,
        )
        _write_or_validate_reuse_record(
            workspace=self.workspace, record=derived.record
        )
        if derived.incident is not None:
            incident_path = self.workspace / "incidents" / f"{surface.request_key}.json"
            if incident_path.exists():
                if EvidenceInferenceFableIncidentV2.model_validate(
                    _read_object(incident_path)
                ) != derived.incident:
                    raise EvidenceInferenceFableFullReuseError(
                        "fable_reuse_target_incident_replay_mismatch"
                    )
            else:
                atomic_write_json(incident_path, derived.incident)
        return derived.result


def _write_or_validate_reuse_record(
    *, workspace: Path, record: EvidenceInferenceFableReuseRecordV1
) -> None:
    path = workspace / REUSE_DIRECTORY / "records" / f"{record.target_request_key}.json"
    if path.exists():
        if EvidenceInferenceFableReuseRecordV1.model_validate(
            _read_object(path)
        ) != record:
            raise EvidenceInferenceFableFullReuseError(
                "fable_reuse_record_replay_mismatch"
            )
        return
    atomic_write_json(path, record)


def _recover_adoption_orphans(
    *,
    workspace: Path,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    prepared: EvidenceInferenceFablePreparedRuntimeV1,
    authorization: EvidenceInferenceFableBudgetAuthorizationV1,
    adoption_plan: EvidenceInferenceFableFullReusePlanV1,
    source_states: Mapping[SourceSlot, _SourceState],
) -> None:
    """Repair only zero-call adoption orphans; a real-call orphan still poisons."""

    surfaces = {surface.request_key: surface for surface in prepared.surfaces}
    for entry in adoption_plan.entries:
        intent_path = workspace / "intents" / f"{entry.target_request_key}.json"
        receipt_path = workspace / "receipts" / f"{entry.target_request_key}.json"
        incident_path = workspace / "incidents" / f"{entry.target_request_key}.json"
        record_path = (
            workspace / REUSE_DIRECTORY / "records" / f"{entry.target_request_key}.json"
        )
        existing = [
            intent_path.exists(),
            receipt_path.exists(),
            incident_path.exists(),
            record_path.exists(),
        ]
        if not any(existing):
            continue
        if not intent_path.exists():
            raise EvidenceInferenceFableFullReuseError(
                "fable_reuse_target_artifact_without_intent"
            )
        intent = EvidenceInferenceFableIntentV1.model_validate(
            _read_object(intent_path)
        )
        derived = _derive_adoption(
            entry=entry,
            adoption_plan=adoption_plan,
            target_intent=intent,
            target_surface=surfaces[entry.target_request_key],
            target_authorization=authorization,
            full_plan=full_plan,
            source_states=source_states,
        )
        if not record_path.exists() and (receipt_path.exists() or incident_path.exists()):
            raise EvidenceInferenceFableFullReuseError(
                "fable_reuse_target_result_without_prior_sidecar"
            )
        _write_or_validate_reuse_record(workspace=workspace, record=derived.record)
        if derived.incident is not None:
            if incident_path.exists():
                if EvidenceInferenceFableIncidentV2.model_validate(
                    _read_object(incident_path)
                ) != derived.incident:
                    raise EvidenceInferenceFableFullReuseError(
                        "fable_reuse_recovered_incident_mismatch"
                    )
            else:
                atomic_write_json(incident_path, derived.incident)
        elif incident_path.exists():
            raise EvidenceInferenceFableFullReuseError(
                "fable_reuse_receipt_adoption_has_incident"
            )
        if receipt_path.exists():
            if EvidenceInferenceFableReceiptV1.model_validate(
                _read_object(receipt_path)
            ) != derived.receipt:
                raise EvidenceInferenceFableFullReuseError(
                    "fable_reuse_recovered_receipt_mismatch"
                )
        else:
            atomic_write_json(receipt_path, derived.receipt)


def _validated_context(
    *,
    workspace: Path,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    sources: list[EvidenceInferenceFableReuseSourceV1],
) -> tuple[
    EvidenceInferenceFablePreparedRuntimeV1,
    EvidenceInferenceFableBudgetAuthorizationV1,
    EvidenceInferenceFableFullReusePlanV1,
    dict[SourceSlot, _SourceState],
]:
    prepared = EvidenceInferenceFablePreparedRuntimeV1.model_validate(
        _read_object(workspace / "00-prepared.json")
    )
    authorization = EvidenceInferenceFableBudgetAuthorizationV1.model_validate(
        _read_object(workspace / "01-authorization.json")
    )
    archived_plan = EvidenceInferenceFableFullReusePlanV1.model_validate(
        _read_object(workspace / REUSE_DIRECTORY / REUSE_PLAN_FILE)
    )
    expected = freeze_evidence_inference_fable_full_reuse_plan_v1(
        full_plan=full_plan,
        full_prepared=prepared,
        full_authorization=authorization,
        sources=sources,
    )
    if archived_plan != expected:
        raise EvidenceInferenceFableFullReuseError(
            "fable_reuse_plan_external_replay_mismatch"
        )
    states = {source.slot: _load_source(source) for source in sources}
    return prepared, authorization, archived_plan, states


def _freeze_reuse_terminal(
    *,
    workspace: Path,
    adoption_plan: EvidenceInferenceFableFullReusePlanV1,
    target_terminal: EvidenceInferenceFableTerminalV1,
) -> EvidenceInferenceFableFullReuseTerminalV1:
    records_path = workspace / REUSE_DIRECTORY / "records"
    records = _artifact_map(records_path, EvidenceInferenceFableReuseRecordV1)
    intents = _artifact_map(workspace / "intents", EvidenceInferenceFableIntentV1)
    receipts = _artifact_map(workspace / "receipts", EvidenceInferenceFableReceiptV1)
    adopted_cost = 0
    terminal_receipts = 0
    inherited = 0
    for key, record in records.items():
        receipt = receipts.get(key)
        if receipt is None or receipt.receipt_sha256 != record.expected_target_receipt_sha256:
            raise EvidenceInferenceFableFullReuseError(
                "fable_reuse_terminal_record_receipt_mismatch"
            )
        adopted_cost += receipt.provider_result.charged_cost_usd_micros
        if record.adoption_kind == "terminal_receipt":
            terminal_receipts += 1
        else:
            inherited += 1
    new_attempts = len(intents) - len(records)
    payload = {
        "terminal_version": "evidence-inference-fable-full-reuse-terminal-v1",
        "adoption_plan_sha256": adoption_plan.plan_sha256,
        "target_runtime_terminal_sha256": target_terminal.terminal_sha256,
        "target_runtime_status": target_terminal.status,
        "target_completed_request_count": target_terminal.completed_request_count,
        "realized_adopted_terminal_receipt_count": terminal_receipts,
        "realized_inherited_ambiguous_failure_count": inherited,
        "new_provider_attempt_count": new_attempts,
        "maximum_new_provider_attempt_count": 361,
        "target_accounted_spend_usd_micros": (
            target_terminal.cumulative_reported_spend_usd_micros
        ),
        "adopted_source_accounted_spend_usd_micros": adopted_cost,
        "new_provider_accounted_spend_usd_micros": (
            target_terminal.cumulative_reported_spend_usd_micros - adopted_cost
        ),
        "source_provider_attempts_reused": len(records),
        "inherited_ambiguous_attempts_retried": 0,
        "target_provider_attempts_for_adopted_entries": 0,
        "full_population_score_permitted": (
            target_terminal.status == "completed"
            and terminal_receipts == EXPECTED_ADOPTED_RECEIPTS
            and inherited == EXPECTED_INHERITED_AMBIGUITIES
            and new_attempts == 361
        ),
        "scoring_requires_this_reuse_terminal": True,
        "scientific_claim_authority": False,
        "confirmatory_gepa_improvement_authority": False,
        "claim_release_authority": False,
    }
    if payload["new_provider_accounted_spend_usd_micros"] < 0:
        raise EvidenceInferenceFableFullReuseError(
            "fable_reuse_terminal_adopted_spend_exceeds_total"
        )
    return EvidenceInferenceFableFullReuseTerminalV1.model_validate(
        {**payload, "terminal_sha256": hash_canonical(payload)}
    )


def _validate_realized_records(
    *,
    workspace: Path,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    prepared: EvidenceInferenceFablePreparedRuntimeV1,
    authorization: EvidenceInferenceFableBudgetAuthorizationV1,
    adoption_plan: EvidenceInferenceFableFullReusePlanV1,
    source_states: Mapping[SourceSlot, _SourceState],
    target_terminal: EvidenceInferenceFableTerminalV1,
) -> None:
    records = _artifact_map(
        workspace / REUSE_DIRECTORY / "records", EvidenceInferenceFableReuseRecordV1
    )
    expected_entries = {
        entry.target_request_key: entry
        for entry in adoption_plan.entries
        if entry.target_execution_index < target_terminal.completed_request_count
    }
    if set(records) != set(expected_entries):
        raise EvidenceInferenceFableFullReuseError(
            "fable_reuse_realized_record_roster_mismatch"
        )
    surfaces = {surface.request_key: surface for surface in prepared.surfaces}
    for key, entry in expected_entries.items():
        intent = EvidenceInferenceFableIntentV1.model_validate(
            _read_object(workspace / "intents" / f"{key}.json")
        )
        derived = _derive_adoption(
            entry=entry,
            adoption_plan=adoption_plan,
            target_intent=intent,
            target_surface=surfaces[key],
            target_authorization=authorization,
            full_plan=full_plan,
            source_states=source_states,
        )
        if records[key] != derived.record or EvidenceInferenceFableReceiptV1.model_validate(
            _read_object(workspace / "receipts" / f"{key}.json")
        ) != derived.receipt:
            raise EvidenceInferenceFableFullReuseError(
                "fable_reuse_realized_artifact_external_replay_mismatch"
            )
        incident_path = workspace / "incidents" / f"{key}.json"
        if derived.incident is None:
            if incident_path.exists():
                raise EvidenceInferenceFableFullReuseError(
                    "fable_reuse_unexpected_target_incident"
                )
        elif EvidenceInferenceFableIncidentV2.model_validate(
            _read_object(incident_path)
        ) != derived.incident:
            raise EvidenceInferenceFableFullReuseError(
                "fable_reuse_inherited_incident_external_replay_mismatch"
            )


def execute_evidence_inference_fable_full_reuse_v1(
    *,
    workspace: Path,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    sources: list[EvidenceInferenceFableReuseSourceV1],
    delegate: EvidenceInferenceFableClientProtocol,
) -> EvidenceInferenceFableFullReuseTerminalV1:
    """Run/replay the full roster while delegating at most 361 new calls."""

    with _reuse_lock(workspace):
        prepared, authorization, adoption_plan, states = _validated_context(
            workspace=workspace, full_plan=full_plan, sources=sources
        )
        _recover_adoption_orphans(
            workspace=workspace,
            full_plan=full_plan,
            prepared=prepared,
            authorization=authorization,
            adoption_plan=adoption_plan,
            source_states=states,
        )
        client = _ReuseAwareClient(
            workspace=workspace,
            full_plan=full_plan,
            prepared=prepared,
            authorization=authorization,
            adoption_plan=adoption_plan,
            source_states=states,
            delegate=delegate,
        )
        target_terminal = execute_evidence_inference_fable_paired_v1(
            workspace=workspace, plan=full_plan, client=client
        )
        _validate_realized_records(
            workspace=workspace,
            full_plan=full_plan,
            prepared=prepared,
            authorization=authorization,
            adoption_plan=adoption_plan,
            source_states=states,
            target_terminal=target_terminal,
        )
        reuse_terminal = _freeze_reuse_terminal(
            workspace=workspace,
            adoption_plan=adoption_plan,
            target_terminal=target_terminal,
        )
        path = workspace / REUSE_DIRECTORY / REUSE_TERMINAL_FILE
        if path.exists():
            if EvidenceInferenceFableFullReuseTerminalV1.model_validate(
                _read_object(path)
            ) != reuse_terminal:
                raise EvidenceInferenceFableFullReuseError(
                    "fable_reuse_terminal_replay_mismatch"
                )
        else:
            atomic_write_json(path, reuse_terminal)
        return reuse_terminal


def validate_evidence_inference_fable_full_reuse_v1(
    *,
    workspace: Path,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    sources: list[EvidenceInferenceFableReuseSourceV1],
) -> EvidenceInferenceFableFullReuseTerminalV1:
    """Externally replay source, target, adoption records, and both terminals."""

    with _reuse_lock(workspace):
        prepared, authorization, adoption_plan, states = _validated_context(
            workspace=workspace, full_plan=full_plan, sources=sources
        )
        target_terminal = validate_evidence_inference_fable_workspace_v1(
            workspace=workspace, plan=full_plan
        )
        _validate_realized_records(
            workspace=workspace,
            full_plan=full_plan,
            prepared=prepared,
            authorization=authorization,
            adoption_plan=adoption_plan,
            source_states=states,
            target_terminal=target_terminal,
        )
        expected = _freeze_reuse_terminal(
            workspace=workspace,
            adoption_plan=adoption_plan,
            target_terminal=target_terminal,
        )
        observed = EvidenceInferenceFableFullReuseTerminalV1.model_validate(
            _read_object(workspace / REUSE_DIRECTORY / REUSE_TERMINAL_FILE)
        )
        if observed != expected:
            raise EvidenceInferenceFableFullReuseError(
                "fable_reuse_terminal_external_replay_mismatch"
            )
        return observed


def require_evidence_inference_fable_full_reuse_scoring_v1(
    *,
    workspace: Path,
    full_plan: EvidenceInferenceFableRetrospectivePlanV1,
    sources: list[EvidenceInferenceFableReuseSourceV1],
) -> EvidenceInferenceFableFullReuseTerminalV1:
    """Fail closed unless scoring is backed by the completed reuse lineage."""

    terminal = validate_evidence_inference_fable_full_reuse_v1(
        workspace=workspace, full_plan=full_plan, sources=sources
    )
    if not terminal.full_population_score_permitted:
        raise EvidenceInferenceFableFullReuseError(
            "fable_reuse_full_scoring_prerequisite_not_satisfied"
        )
    return terminal
