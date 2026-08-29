"""Label-blind continuation and descriptive composite for a poisoned Fable run.

The original runtime deliberately stops forever after an ambiguous provider attempt.
This module does not weaken that rule.  It externally replays the poisoned workspace,
freezes a separate continuation containing only requests for which no durable intent
exists, and later joins the two immutable lineages.  The ambiguous request is retained
as an intention-to-evaluate failure with zero credit on every locked question.

The resulting composite is suitable for descriptive pilot scoring only.  In
particular, it cannot satisfy the predeclared clean-mechanics gate: changing that gate
after observing a transport incident would hide the very operational failure the pilot
was intended to detect.
"""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from pydantic import ConfigDict, Field, StrictInt, model_validator

from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    EvidenceInferenceFableBudgetAuthorizationV1,
    EvidenceInferenceFableCallSurfaceV1,
    EvidenceInferenceFableIncidentV1,
    EvidenceInferenceFableIntentV1,
    EvidenceInferenceFablePreparedRuntimeV1,
    EvidenceInferenceFableProviderResultV1,
    EvidenceInferenceFableReceiptV1,
    EvidenceInferenceFableTerminalV1,
    validate_evidence_inference_fable_workspace_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_v1 import (
    ArticleBatchRequestV1,
    EvidenceInferenceFableRetrospectivePlanV1,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel

Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]
Micros = Annotated[StrictInt, Field(ge=0)]
PositiveMicros = Annotated[StrictInt, Field(ge=1)]
Count = Annotated[StrictInt, Field(ge=0)]


class EvidenceInferenceFableContinuationError(ValueError):
    """A source replay, continuation, or composite invariant failed closed."""


class _Frozen(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=False,
    )


def _self_hash(model: _Frozen, field: str, code: str) -> None:
    if getattr(model, field) != hash_canonical(
        model.model_dump(mode="json", exclude={field})
    ):
        raise ValueError(code)


def _read_object(path: Path, *, root: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceInferenceFableContinuationError(
            "fable_continuation_artifact_unsafe"
        )
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidenceInferenceFableContinuationError(
            "fable_continuation_artifact_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise EvidenceInferenceFableContinuationError(
            "fable_continuation_artifact_not_object"
        )
    return value


def _ensure_directory(path: Path) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise EvidenceInferenceFableContinuationError(
                "fable_continuation_directory_unsafe"
            )
        return
    path.mkdir(mode=0o700)


@contextmanager
def _lock(workspace: Path) -> Any:
    descriptor = os.open(
        workspace / ".continuation.lock",
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(descriptor, "a+b", closefd=True) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def _authorized_liability(
    authorization: EvidenceInferenceFableBudgetAuthorizationV1,
    request: ArticleBatchRequestV1,
) -> int:
    if authorization.liability_basis == "certified_provider_token_count":
        try:
            return authorization.certified_request_liabilities_usd_micros[
                request.request_key
            ]
        except KeyError as exc:
            raise EvidenceInferenceFableContinuationError(
                "fable_continuation_source_liability_missing"
            ) from exc
    return request.cost.full_context_hard_liability_usd_micros


class EvidenceInferenceFableContinuationRequestV1(_Frozen):
    original_execution_index: Count
    request_key: str
    article_request_sha256: Sha256
    surface: EvidenceInferenceFableCallSurfaceV1
    authorized_hard_liability_usd_micros: PositiveMicros
    request_sha256: Sha256

    @model_validator(mode="after")
    def validate_request(self) -> EvidenceInferenceFableContinuationRequestV1:
        if (
            self.request_key != self.surface.request_key
            or self.article_request_sha256 != self.surface.article_request_sha256
        ):
            raise ValueError("fable_continuation_request_alias_mismatch")
        _self_hash(
            self,
            "request_sha256",
            "fable_continuation_request_hash_mismatch",
        )
        return self


class EvidenceInferenceFableForcedFailureV1(_Frozen):
    request_key: str
    article_request_sha256: Sha256
    original_execution_index: Count
    locked_question_count: Annotated[StrictInt, Field(ge=1)]
    source_intent_sha256: Sha256
    source_incident_sha256: Sha256
    failure_basis: Literal[
        "ambiguous_provider_attempt_intention_to_evaluate_failure"
    ] = "ambiguous_provider_attempt_intention_to_evaluate_failure"
    structured_output_credit_for_all_locked_questions: Literal[0] = 0
    direction_credit_for_all_locked_questions: Literal[0] = 0
    exact_grounding_credit_for_all_locked_questions: Literal[0] = 0
    retry_permitted: Literal[False] = False
    failure_sha256: Sha256

    @model_validator(mode="after")
    def validate_failure(self) -> EvidenceInferenceFableForcedFailureV1:
        _self_hash(
            self,
            "failure_sha256",
            "fable_continuation_forced_failure_hash_mismatch",
        )
        return self


class EvidenceInferenceFableContinuationPlanV1(_Frozen):
    plan_version: Literal["evidence-inference-fable-continuation-plan-v1"] = (
        "evidence-inference-fable-continuation-plan-v1"
    )
    status: Literal["offline_prepared_zero_provider_calls"] = (
        "offline_prepared_zero_provider_calls"
    )
    retrospective_plan_sha256: Sha256
    source_prepared_sha256: Sha256
    source_authorization_sha256: Sha256
    source_terminal_sha256: Sha256
    source_snapshot_sha256: Sha256
    source_completed_request_count: Count
    source_completed_receipt_sha256s: dict[str, Sha256]
    forced_failure: EvidenceInferenceFableForcedFailureV1
    continuation_requests: list[EvidenceInferenceFableContinuationRequestV1]
    continuation_request_roster_sha256: Sha256
    original_request_count: Annotated[StrictInt, Field(ge=2)]
    continuation_request_count: Annotated[StrictInt, Field(ge=1)]
    prior_workspace_mutation_permitted: Literal[False] = False
    ambiguous_request_retry_permitted: Literal[False] = False
    descriptive_scoring_only: Literal[True] = True
    clean_mechanics_gate_authority: Literal[False] = False
    confirmatory_authority: Literal[False] = False
    scientific_claim_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    labels_opened: Literal[False] = False
    provider_calls_made: Literal[0] = 0
    plan_sha256: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> EvidenceInferenceFableContinuationPlanV1:
        indices = [item.original_execution_index for item in self.continuation_requests]
        if (
            self.continuation_request_count != len(self.continuation_requests)
            or indices != sorted(indices)
            or len(set(indices)) != len(indices)
            or len({item.request_key for item in self.continuation_requests}) != len(indices)
            or self.forced_failure.original_execution_index
            != self.source_completed_request_count
            or self.original_request_count
            != self.source_completed_request_count + 1 + self.continuation_request_count
            or self.continuation_request_roster_sha256
            != hash_canonical([item.request_sha256 for item in self.continuation_requests])
        ):
            raise ValueError("fable_continuation_plan_roster_mismatch")
        _self_hash(self, "plan_sha256", "fable_continuation_plan_hash_mismatch")
        return self


@dataclass(frozen=True)
class _SourceReplay:
    prepared: EvidenceInferenceFablePreparedRuntimeV1
    authorization: EvidenceInferenceFableBudgetAuthorizationV1
    terminal: EvidenceInferenceFableTerminalV1
    incident_intent: EvidenceInferenceFableIntentV1
    incident: EvidenceInferenceFableIncidentV1
    completed_receipts: tuple[EvidenceInferenceFableReceiptV1, ...]
    completed_intents: tuple[EvidenceInferenceFableIntentV1, ...]
    snapshot_sha256: str


def _replay_poisoned_source(
    *,
    source_workspace: Path,
    retrospective_plan: EvidenceInferenceFableRetrospectivePlanV1,
) -> _SourceReplay:
    workspace = source_workspace.resolve(strict=True)
    terminal = validate_evidence_inference_fable_workspace_v1(
        workspace=workspace,
        plan=retrospective_plan,
    )
    if terminal.status != "terminal_ambiguous_attempt_poison":
        raise EvidenceInferenceFableContinuationError(
            "fable_continuation_requires_poisoned_source"
        )
    prepared = EvidenceInferenceFablePreparedRuntimeV1.model_validate(
        _read_object(workspace / "00-prepared.json", root=workspace)
    )
    authorization = EvidenceInferenceFableBudgetAuthorizationV1.model_validate(
        _read_object(workspace / "01-authorization.json", root=workspace)
    )
    incident_index = terminal.completed_request_count
    if incident_index >= len(retrospective_plan.roster) - 1:
        raise EvidenceInferenceFableContinuationError(
            "fable_continuation_source_has_no_unattempted_requests"
        )
    completed_receipts: list[EvidenceInferenceFableReceiptV1] = []
    completed_intents: list[EvidenceInferenceFableIntentV1] = []
    for request in retrospective_plan.roster[:incident_index]:
        completed_intents.append(
            EvidenceInferenceFableIntentV1.model_validate(
                _read_object(
                    workspace / "intents" / f"{request.request_key}.json",
                    root=workspace,
                )
            )
        )
        completed_receipts.append(
            EvidenceInferenceFableReceiptV1.model_validate(
                _read_object(
                    workspace / "receipts" / f"{request.request_key}.json",
                    root=workspace,
                )
            )
        )
    poisoned_request = retrospective_plan.roster[incident_index]
    incident_intent = EvidenceInferenceFableIntentV1.model_validate(
        _read_object(
            workspace / "intents" / f"{poisoned_request.request_key}.json",
            root=workspace,
        )
    )
    incident = EvidenceInferenceFableIncidentV1.model_validate(
        _read_object(
            workspace / "incidents" / f"{poisoned_request.request_key}.json",
            root=workspace,
        )
    )
    snapshot = {
        "prepared_sha256": prepared.prepared_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "terminal_sha256": terminal.terminal_sha256,
        "completed_intent_sha256s": [item.intent_sha256 for item in completed_intents],
        "completed_receipt_sha256s": [item.receipt_sha256 for item in completed_receipts],
        "incident_intent_sha256": incident_intent.intent_sha256,
        "incident_sha256": incident.incident_sha256,
    }
    return _SourceReplay(
        prepared=prepared,
        authorization=authorization,
        terminal=terminal,
        incident_intent=incident_intent,
        incident=incident,
        completed_receipts=tuple(completed_receipts),
        completed_intents=tuple(completed_intents),
        snapshot_sha256=hash_canonical(snapshot),
    )


def freeze_evidence_inference_fable_continuation_plan_v1(
    *,
    source_workspace: Path,
    retrospective_plan: EvidenceInferenceFableRetrospectivePlanV1,
) -> EvidenceInferenceFableContinuationPlanV1:
    """Freeze only never-attempted requests after externally replaying the source."""

    replay = _replay_poisoned_source(
        source_workspace=source_workspace,
        retrospective_plan=retrospective_plan,
    )
    index = replay.terminal.completed_request_count
    poisoned = retrospective_plan.roster[index]
    forced_base = {
        "request_key": poisoned.request_key,
        "article_request_sha256": poisoned.request_sha256,
        "original_execution_index": index,
        "locked_question_count": poisoned.question_count,
        "source_intent_sha256": replay.incident_intent.intent_sha256,
        "source_incident_sha256": replay.incident.incident_sha256,
        "failure_basis": "ambiguous_provider_attempt_intention_to_evaluate_failure",
        "structured_output_credit_for_all_locked_questions": 0,
        "direction_credit_for_all_locked_questions": 0,
        "exact_grounding_credit_for_all_locked_questions": 0,
        "retry_permitted": False,
    }
    forced = EvidenceInferenceFableForcedFailureV1.model_validate(
        {**forced_base, "failure_sha256": hash_canonical(forced_base)}
    )
    continuation: list[EvidenceInferenceFableContinuationRequestV1] = []
    for request, surface in zip(
        retrospective_plan.roster[index + 1 :],
        replay.prepared.surfaces[index + 1 :],
        strict=True,
    ):
        base = {
            "original_execution_index": request.execution_index,
            "request_key": request.request_key,
            "article_request_sha256": request.request_sha256,
            "surface": surface,
            "authorized_hard_liability_usd_micros": _authorized_liability(
                replay.authorization, request
            ),
        }
        continuation.append(
            EvidenceInferenceFableContinuationRequestV1.model_validate(
                {**base, "request_sha256": hash_canonical(base)}
            )
        )
    completed_hashes = {
        request.request_key: receipt.receipt_sha256
        for request, receipt in zip(
            retrospective_plan.roster[:index], replay.completed_receipts, strict=True
        )
    }
    payload = {
        "plan_version": "evidence-inference-fable-continuation-plan-v1",
        "status": "offline_prepared_zero_provider_calls",
        "retrospective_plan_sha256": retrospective_plan.plan_sha256,
        "source_prepared_sha256": replay.prepared.prepared_sha256,
        "source_authorization_sha256": replay.authorization.authorization_sha256,
        "source_terminal_sha256": replay.terminal.terminal_sha256,
        "source_snapshot_sha256": replay.snapshot_sha256,
        "source_completed_request_count": index,
        "source_completed_receipt_sha256s": completed_hashes,
        "forced_failure": forced,
        "continuation_requests": continuation,
        "continuation_request_roster_sha256": hash_canonical(
            [item.request_sha256 for item in continuation]
        ),
        "original_request_count": retrospective_plan.request_count,
        "continuation_request_count": len(continuation),
        "prior_workspace_mutation_permitted": False,
        "ambiguous_request_retry_permitted": False,
        "descriptive_scoring_only": True,
        "clean_mechanics_gate_authority": False,
        "confirmatory_authority": False,
        "scientific_claim_authority": False,
        "claim_release_authority": False,
        "labels_opened": False,
        "provider_calls_made": 0,
    }
    return EvidenceInferenceFableContinuationPlanV1.model_validate(
        {**payload, "plan_sha256": hash_canonical(payload)}
    )


class EvidenceInferenceFableContinuationAuthorizationV1(_Frozen):
    authorization_version: Literal[
        "evidence-inference-fable-continuation-authorization-v1"
    ] = "evidence-inference-fable-continuation-authorization-v1"
    continuation_plan_sha256: Sha256
    configured_total_budget_usd_micros: PositiveMicros
    required_total_hard_liability_usd_micros: PositiveMicros
    whole_continuation_liability_admitted_before_first_attempt: Literal[True] = True
    permitted_attempts_per_request: Literal[1] = 1
    application_retries_permitted: Literal[0] = 0
    sdk_retries_permitted: Literal[0] = 0
    ambiguous_request_retry_permitted: Literal[False] = False
    authorization_sha256: Sha256

    @model_validator(mode="after")
    def validate_authorization(
        self,
    ) -> EvidenceInferenceFableContinuationAuthorizationV1:
        if (
            self.configured_total_budget_usd_micros
            < self.required_total_hard_liability_usd_micros
        ):
            raise ValueError("fable_continuation_budget_below_full_liability")
        _self_hash(
            self,
            "authorization_sha256",
            "fable_continuation_authorization_hash_mismatch",
        )
        return self


def freeze_evidence_inference_fable_continuation_authorization_v1(
    *,
    continuation_plan: EvidenceInferenceFableContinuationPlanV1,
    configured_total_budget_usd_micros: int,
) -> EvidenceInferenceFableContinuationAuthorizationV1:
    required = sum(
        item.authorized_hard_liability_usd_micros
        for item in continuation_plan.continuation_requests
    )
    payload = {
        "authorization_version": (
            "evidence-inference-fable-continuation-authorization-v1"
        ),
        "continuation_plan_sha256": continuation_plan.plan_sha256,
        "configured_total_budget_usd_micros": configured_total_budget_usd_micros,
        "required_total_hard_liability_usd_micros": required,
        "whole_continuation_liability_admitted_before_first_attempt": True,
        "permitted_attempts_per_request": 1,
        "application_retries_permitted": 0,
        "sdk_retries_permitted": 0,
        "ambiguous_request_retry_permitted": False,
    }
    return EvidenceInferenceFableContinuationAuthorizationV1.model_validate(
        {**payload, "authorization_sha256": hash_canonical(payload)}
    )


class EvidenceInferenceFableContinuationIntentV1(_Frozen):
    intent_version: Literal["evidence-inference-fable-continuation-intent-v1"] = (
        "evidence-inference-fable-continuation-intent-v1"
    )
    continuation_plan_sha256: Sha256
    continuation_authorization_sha256: Sha256
    source_terminal_sha256: Sha256
    continuation_index: Count
    request: EvidenceInferenceFableContinuationRequestV1
    cumulative_charged_spend_before_request_usd_micros: Micros
    permitted_provider_attempts: Literal[1] = 1
    application_retries_permitted: Literal[0] = 0
    sdk_retries_permitted: Literal[0] = 0
    intent_sha256: Sha256

    @model_validator(mode="after")
    def validate_intent(self) -> EvidenceInferenceFableContinuationIntentV1:
        _self_hash(self, "intent_sha256", "fable_continuation_intent_hash_mismatch")
        return self


class EvidenceInferenceFableContinuationReceiptV1(_Frozen):
    receipt_version: Literal["evidence-inference-fable-continuation-receipt-v1"] = (
        "evidence-inference-fable-continuation-receipt-v1"
    )
    intent_sha256: Sha256
    request_key: str
    provider_result: EvidenceInferenceFableProviderResultV1
    locked_question_count: Annotated[StrictInt, Field(ge=1)]
    locked_questions_scored_incorrect: Count
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> EvidenceInferenceFableContinuationReceiptV1:
        expected = (
            self.locked_question_count if self.provider_result.outcome == "failed" else 0
        )
        if (
            self.request_key != self.provider_result.request_key
            or self.locked_questions_scored_incorrect != expected
        ):
            raise ValueError("fable_continuation_receipt_alias_mismatch")
        _self_hash(
            self,
            "receipt_sha256",
            "fable_continuation_receipt_hash_mismatch",
        )
        return self


class EvidenceInferenceFableContinuationIncidentV1(_Frozen):
    incident_version: Literal["evidence-inference-fable-continuation-incident-v1"] = (
        "evidence-inference-fable-continuation-incident-v1"
    )
    status: Literal["terminal_ambiguous_attempt_poison"] = (
        "terminal_ambiguous_attempt_poison"
    )
    kind: Literal[
        "orphan_intent_observed_on_resume",
        "provider_call_raised_after_durable_intent",
        "provider_result_invalid_after_return",
    ]
    intent_sha256: Sha256
    request_key: str
    charged_cost_usd_micros: PositiveMicros
    retry_permitted: Literal[False] = False
    incident_sha256: Sha256

    @model_validator(mode="after")
    def validate_incident(self) -> EvidenceInferenceFableContinuationIncidentV1:
        _self_hash(
            self,
            "incident_sha256",
            "fable_continuation_incident_hash_mismatch",
        )
        return self


class EvidenceInferenceFableContinuationTerminalV1(_Frozen):
    terminal_version: Literal["evidence-inference-fable-continuation-terminal-v1"] = (
        "evidence-inference-fable-continuation-terminal-v1"
    )
    status: Literal["completed", "terminal_ambiguous_attempt_poison"]
    continuation_plan_sha256: Sha256
    continuation_authorization_sha256: Sha256
    source_terminal_sha256: Sha256
    completed_request_count: Count
    cumulative_charged_spend_usd_micros: Micros
    next_continuation_index: Count
    descriptive_composite_materialization_permitted: bool
    clean_mechanics_gate_authority: Literal[False] = False
    inferential_authority: Literal[False] = False
    scientific_claim_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    terminal_sha256: Sha256

    @model_validator(mode="after")
    def validate_terminal(self) -> EvidenceInferenceFableContinuationTerminalV1:
        if self.descriptive_composite_materialization_permitted != (
            self.status == "completed"
        ):
            raise ValueError("fable_continuation_terminal_permission_mismatch")
        _self_hash(
            self,
            "terminal_sha256",
            "fable_continuation_terminal_hash_mismatch",
        )
        return self


class EvidenceInferenceFableContinuationClientProtocol(Protocol):
    def generate(
        self, surface: EvidenceInferenceFableCallSurfaceV1
    ) -> EvidenceInferenceFableProviderResultV1: ...


def prepare_evidence_inference_fable_continuation_workspace_v1(
    *,
    workspace: Path,
    continuation_plan: EvidenceInferenceFableContinuationPlanV1,
) -> None:
    if workspace.exists() or workspace.is_symlink():
        raise EvidenceInferenceFableContinuationError(
            "fable_continuation_workspace_must_be_fresh"
        )
    workspace.mkdir(parents=True, mode=0o700)
    atomic_write_json(workspace / "00-continuation-plan.json", continuation_plan)


def authorize_evidence_inference_fable_continuation_workspace_v1(
    *,
    workspace: Path,
    authorization: EvidenceInferenceFableContinuationAuthorizationV1,
) -> None:
    root = workspace.resolve(strict=True)
    plan = EvidenceInferenceFableContinuationPlanV1.model_validate(
        _read_object(root / "00-continuation-plan.json", root=root)
    )
    if authorization.continuation_plan_sha256 != plan.plan_sha256:
        raise EvidenceInferenceFableContinuationError(
            "fable_continuation_authorization_plan_mismatch"
        )
    path = root / "01-continuation-authorization.json"
    if path.exists():
        existing = EvidenceInferenceFableContinuationAuthorizationV1.model_validate(
            _read_object(path, root=root)
        )
        if existing != authorization:
            raise EvidenceInferenceFableContinuationError(
                "fable_continuation_authorization_replay_mismatch"
            )
        return
    atomic_write_json(path, authorization)


def _terminal(
    *,
    plan: EvidenceInferenceFableContinuationPlanV1,
    authorization: EvidenceInferenceFableContinuationAuthorizationV1,
    status: Literal["completed", "terminal_ambiguous_attempt_poison"],
    count: int,
    spend: int,
) -> EvidenceInferenceFableContinuationTerminalV1:
    payload = {
        "terminal_version": "evidence-inference-fable-continuation-terminal-v1",
        "status": status,
        "continuation_plan_sha256": plan.plan_sha256,
        "continuation_authorization_sha256": authorization.authorization_sha256,
        "source_terminal_sha256": plan.source_terminal_sha256,
        "completed_request_count": count,
        "cumulative_charged_spend_usd_micros": spend,
        "next_continuation_index": count,
        "descriptive_composite_materialization_permitted": status == "completed",
        "clean_mechanics_gate_authority": False,
        "inferential_authority": False,
        "scientific_claim_authority": False,
        "claim_release_authority": False,
    }
    return EvidenceInferenceFableContinuationTerminalV1.model_validate(
        {**payload, "terminal_sha256": hash_canonical(payload)}
    )


def _validate_plan_against_source(
    *,
    continuation_plan: EvidenceInferenceFableContinuationPlanV1,
    source_workspace: Path,
    retrospective_plan: EvidenceInferenceFableRetrospectivePlanV1,
) -> None:
    regenerated = freeze_evidence_inference_fable_continuation_plan_v1(
        source_workspace=source_workspace,
        retrospective_plan=retrospective_plan,
    )
    if regenerated != continuation_plan:
        raise EvidenceInferenceFableContinuationError(
            "fable_continuation_source_replay_mismatch"
        )


def _validate_intent(
    *,
    intent: EvidenceInferenceFableContinuationIntentV1,
    plan: EvidenceInferenceFableContinuationPlanV1,
    authorization: EvidenceInferenceFableContinuationAuthorizationV1,
    index: int,
    spend: int,
) -> None:
    request = plan.continuation_requests[index]
    if (
        intent.continuation_plan_sha256 != plan.plan_sha256
        or intent.continuation_authorization_sha256 != authorization.authorization_sha256
        or intent.source_terminal_sha256 != plan.source_terminal_sha256
        or intent.continuation_index != index
        or intent.request != request
        or intent.cumulative_charged_spend_before_request_usd_micros != spend
    ):
        raise EvidenceInferenceFableContinuationError(
            "fable_continuation_archived_intent_mismatch"
        )


def _validate_receipt(
    *,
    receipt: EvidenceInferenceFableContinuationReceiptV1,
    intent: EvidenceInferenceFableContinuationIntentV1,
    request: EvidenceInferenceFableContinuationRequestV1,
) -> None:
    result = receipt.provider_result
    if (
        receipt.intent_sha256 != intent.intent_sha256
        or receipt.request_key != request.request_key
        or result.surface_sha256 != request.surface.surface_sha256
        or receipt.locked_question_count != request.surface.locked_question_count
        or (result.output_tokens or 0) > request.surface.max_output_tokens
        or result.charged_cost_usd_micros
        > request.authorized_hard_liability_usd_micros
    ):
        raise EvidenceInferenceFableContinuationError(
            "fable_continuation_archived_receipt_mismatch"
        )


def _validate_incident(
    *,
    incident: EvidenceInferenceFableContinuationIncidentV1,
    intent: EvidenceInferenceFableContinuationIntentV1,
    request: EvidenceInferenceFableContinuationRequestV1,
) -> None:
    if (
        incident.intent_sha256 != intent.intent_sha256
        or incident.request_key != request.request_key
        or incident.charged_cost_usd_micros
        != request.authorized_hard_liability_usd_micros
    ):
        raise EvidenceInferenceFableContinuationError(
            "fable_continuation_archived_incident_mismatch"
        )


def _validate_workspace_locked(
    *,
    workspace: Path,
    source_workspace: Path,
    retrospective_plan: EvidenceInferenceFableRetrospectivePlanV1,
) -> EvidenceInferenceFableContinuationTerminalV1:
    root = workspace.resolve(strict=True)
    plan = EvidenceInferenceFableContinuationPlanV1.model_validate(
        _read_object(root / "00-continuation-plan.json", root=root)
    )
    authorization = EvidenceInferenceFableContinuationAuthorizationV1.model_validate(
        _read_object(root / "01-continuation-authorization.json", root=root)
    )
    terminal = EvidenceInferenceFableContinuationTerminalV1.model_validate(
        _read_object(root / "02-continuation-terminal.json", root=root)
    )
    if (
        plan.retrospective_plan_sha256 != retrospective_plan.plan_sha256
        or authorization.continuation_plan_sha256 != plan.plan_sha256
        or terminal.continuation_plan_sha256 != plan.plan_sha256
        or terminal.continuation_authorization_sha256
        != authorization.authorization_sha256
        or terminal.source_terminal_sha256 != plan.source_terminal_sha256
    ):
        raise EvidenceInferenceFableContinuationError(
            "fable_continuation_terminal_binding_mismatch"
        )
    _validate_plan_against_source(
        continuation_plan=plan,
        source_workspace=source_workspace,
        retrospective_plan=retrospective_plan,
    )
    planned_keys = {item.request_key for item in plan.continuation_requests}
    observed: dict[str, set[str]] = {}
    for directory_name in ("intents", "receipts", "incidents"):
        directory = root / directory_name
        names: set[str] = set()
        if directory.exists():
            if directory.is_symlink() or not directory.is_dir():
                raise EvidenceInferenceFableContinuationError(
                    "fable_continuation_directory_unsafe"
                )
            for path in directory.iterdir():
                if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                    raise EvidenceInferenceFableContinuationError(
                        "fable_continuation_extra_artifact"
                    )
                names.add(path.stem)
        if not names.issubset(planned_keys):
            raise EvidenceInferenceFableContinuationError(
                "fable_continuation_unknown_artifact"
            )
        observed[directory_name] = names
    count = 0
    spend = 0
    incident_key: str | None = None
    for index, request in enumerate(plan.continuation_requests):
        ip = root / "intents" / f"{request.request_key}.json"
        rp = root / "receipts" / f"{request.request_key}.json"
        xp = root / "incidents" / f"{request.request_key}.json"
        if rp.exists():
            intent = EvidenceInferenceFableContinuationIntentV1.model_validate(
                _read_object(ip, root=root)
            )
            _validate_intent(
                intent=intent,
                plan=plan,
                authorization=authorization,
                index=index,
                spend=spend,
            )
            receipt = EvidenceInferenceFableContinuationReceiptV1.model_validate(
                _read_object(rp, root=root)
            )
            _validate_receipt(receipt=receipt, intent=intent, request=request)
            count += 1
            spend += receipt.provider_result.charged_cost_usd_micros
        elif xp.exists():
            intent = EvidenceInferenceFableContinuationIntentV1.model_validate(
                _read_object(ip, root=root)
            )
            _validate_intent(
                intent=intent,
                plan=plan,
                authorization=authorization,
                index=index,
                spend=spend,
            )
            incident = EvidenceInferenceFableContinuationIncidentV1.model_validate(
                _read_object(xp, root=root)
            )
            _validate_incident(incident=incident, intent=intent, request=request)
            spend += incident.charged_cost_usd_micros
            incident_key = request.request_key
            break
        else:
            break
    expected_status: Literal["completed", "terminal_ambiguous_attempt_poison"] = (
        "terminal_ambiguous_attempt_poison" if incident_key is not None else "completed"
    )
    if expected_status == "completed" and count != len(plan.continuation_requests):
        raise EvidenceInferenceFableContinuationError(
            "fable_continuation_completed_terminal_incomplete"
        )
    expected = _terminal(
        plan=plan,
        authorization=authorization,
        status=expected_status,
        count=count,
        spend=spend,
    )
    if terminal != expected:
        raise EvidenceInferenceFableContinuationError(
            "fable_continuation_terminal_replay_mismatch"
        )
    ordered = [item.request_key for item in plan.continuation_requests]
    expected_receipts = set(ordered[:count])
    expected_incidents = {incident_key} if incident_key is not None else set()
    if (
        observed["receipts"] != expected_receipts
        or observed["incidents"] != expected_incidents
        or observed["intents"] != expected_receipts | expected_incidents
    ):
        raise EvidenceInferenceFableContinuationError(
            "fable_continuation_artifact_roster_mismatch"
        )
    return terminal


def validate_evidence_inference_fable_continuation_workspace_v1(
    *,
    workspace: Path,
    source_workspace: Path,
    retrospective_plan: EvidenceInferenceFableRetrospectivePlanV1,
) -> EvidenceInferenceFableContinuationTerminalV1:
    with _lock(workspace):
        return _validate_workspace_locked(
            workspace=workspace,
            source_workspace=source_workspace,
            retrospective_plan=retrospective_plan,
        )


def execute_evidence_inference_fable_continuation_v1(
    *,
    workspace: Path,
    source_workspace: Path,
    retrospective_plan: EvidenceInferenceFableRetrospectivePlanV1,
    client: EvidenceInferenceFableContinuationClientProtocol,
) -> EvidenceInferenceFableContinuationTerminalV1:
    """Attempt only the suffix with no durable source intent; never retry any intent."""

    with _lock(workspace):
        root = workspace.resolve(strict=True)
        plan = EvidenceInferenceFableContinuationPlanV1.model_validate(
            _read_object(root / "00-continuation-plan.json", root=root)
        )
        authorization = EvidenceInferenceFableContinuationAuthorizationV1.model_validate(
            _read_object(root / "01-continuation-authorization.json", root=root)
        )
        if authorization.continuation_plan_sha256 != plan.plan_sha256:
            raise EvidenceInferenceFableContinuationError(
                "fable_continuation_authorization_plan_mismatch"
            )
        _validate_plan_against_source(
            continuation_plan=plan,
            source_workspace=source_workspace,
            retrospective_plan=retrospective_plan,
        )
        terminal_path = root / "02-continuation-terminal.json"
        if terminal_path.exists():
            return _validate_workspace_locked(
                workspace=root,
                source_workspace=source_workspace,
                retrospective_plan=retrospective_plan,
            )
        for directory_name in ("intents", "receipts", "incidents"):
            _ensure_directory(root / directory_name)
        count = 0
        spend = 0
        for index, request in enumerate(plan.continuation_requests):
            ip = root / "intents" / f"{request.request_key}.json"
            rp = root / "receipts" / f"{request.request_key}.json"
            xp = root / "incidents" / f"{request.request_key}.json"
            if xp.exists():
                intent = EvidenceInferenceFableContinuationIntentV1.model_validate(
                    _read_object(ip, root=root)
                )
                _validate_intent(
                    intent=intent,
                    plan=plan,
                    authorization=authorization,
                    index=index,
                    spend=spend,
                )
                incident = EvidenceInferenceFableContinuationIncidentV1.model_validate(
                    _read_object(xp, root=root)
                )
                _validate_incident(incident=incident, intent=intent, request=request)
                spend += incident.charged_cost_usd_micros
                terminal = _terminal(
                    plan=plan,
                    authorization=authorization,
                    status="terminal_ambiguous_attempt_poison",
                    count=count,
                    spend=spend,
                )
                atomic_write_json(terminal_path, terminal)
                return terminal
            if rp.exists():
                intent = EvidenceInferenceFableContinuationIntentV1.model_validate(
                    _read_object(ip, root=root)
                )
                _validate_intent(
                    intent=intent,
                    plan=plan,
                    authorization=authorization,
                    index=index,
                    spend=spend,
                )
                receipt = EvidenceInferenceFableContinuationReceiptV1.model_validate(
                    _read_object(rp, root=root)
                )
                _validate_receipt(receipt=receipt, intent=intent, request=request)
                count += 1
                spend += receipt.provider_result.charged_cost_usd_micros
                continue
            if ip.exists():
                intent = EvidenceInferenceFableContinuationIntentV1.model_validate(
                    _read_object(ip, root=root)
                )
                _validate_intent(
                    intent=intent,
                    plan=plan,
                    authorization=authorization,
                    index=index,
                    spend=spend,
                )
                incident_kind = "orphan_intent_observed_on_resume"
            else:
                intent_base = {
                    "intent_version": "evidence-inference-fable-continuation-intent-v1",
                    "continuation_plan_sha256": plan.plan_sha256,
                    "continuation_authorization_sha256": authorization.authorization_sha256,
                    "source_terminal_sha256": plan.source_terminal_sha256,
                    "continuation_index": index,
                    "request": request,
                    "cumulative_charged_spend_before_request_usd_micros": spend,
                    "permitted_provider_attempts": 1,
                    "application_retries_permitted": 0,
                    "sdk_retries_permitted": 0,
                }
                intent = EvidenceInferenceFableContinuationIntentV1.model_validate(
                    {**intent_base, "intent_sha256": hash_canonical(intent_base)}
                )
                atomic_write_json(ip, intent)
                try:
                    observed = client.generate(request.surface)
                except Exception:
                    incident_kind = "provider_call_raised_after_durable_intent"
                else:
                    try:
                        result = EvidenceInferenceFableProviderResultV1.model_validate(
                            observed
                        )
                        if (
                            result.request_key != request.request_key
                            or result.surface_sha256 != request.surface.surface_sha256
                            or (result.output_tokens or 0)
                            > request.surface.max_output_tokens
                            or result.charged_cost_usd_micros
                            > request.authorized_hard_liability_usd_micros
                        ):
                            raise ValueError("continuation_provider_result_binding")
                    except Exception:
                        incident_kind = "provider_result_invalid_after_return"
                    else:
                        incident_kind = None
            if incident_kind is not None:
                incident_base = {
                    "incident_version": "evidence-inference-fable-continuation-incident-v1",
                    "status": "terminal_ambiguous_attempt_poison",
                    "kind": incident_kind,
                    "intent_sha256": intent.intent_sha256,
                    "request_key": request.request_key,
                    "charged_cost_usd_micros": request.authorized_hard_liability_usd_micros,
                    "retry_permitted": False,
                }
                incident = EvidenceInferenceFableContinuationIncidentV1.model_validate(
                    {**incident_base, "incident_sha256": hash_canonical(incident_base)}
                )
                atomic_write_json(xp, incident)
                spend += incident.charged_cost_usd_micros
                terminal = _terminal(
                    plan=plan,
                    authorization=authorization,
                    status="terminal_ambiguous_attempt_poison",
                    count=count,
                    spend=spend,
                )
                atomic_write_json(terminal_path, terminal)
                return terminal
            receipt_base = {
                "receipt_version": "evidence-inference-fable-continuation-receipt-v1",
                "intent_sha256": intent.intent_sha256,
                "request_key": request.request_key,
                "provider_result": result,
                "locked_question_count": request.surface.locked_question_count,
                "locked_questions_scored_incorrect": (
                    request.surface.locked_question_count
                    if result.outcome == "failed"
                    else 0
                ),
            }
            receipt = EvidenceInferenceFableContinuationReceiptV1.model_validate(
                {**receipt_base, "receipt_sha256": hash_canonical(receipt_base)}
            )
            atomic_write_json(rp, receipt)
            count += 1
            spend += result.charged_cost_usd_micros
        _validate_plan_against_source(
            continuation_plan=plan,
            source_workspace=source_workspace,
            retrospective_plan=retrospective_plan,
        )
        terminal = _terminal(
            plan=plan,
            authorization=authorization,
            status="completed",
            count=count,
            spend=spend,
        )
        atomic_write_json(terminal_path, terminal)
        return terminal


class EvidenceInferenceFableCompositeRecordV1(_Frozen):
    original_execution_index: Count
    request_key: str
    article_request_sha256: Sha256
    origin: Literal[
        "source_receipt",
        "source_ambiguous_intention_to_evaluate_failure",
        "continuation_receipt",
    ]
    source_artifact_sha256: Sha256
    locked_question_count: Annotated[StrictInt, Field(ge=1)]
    provider_outcome: Literal[
        "provider_response",
        "provider_response_unusable",
        "transport_failed_or_ambiguous",
    ]
    parsed_batch: dict[str, Any] | None
    input_tokens: Count | None
    output_tokens: Count | None
    accounted_cost_usd_micros: Micros
    forced_zero_structured_output_direction_and_grounding: bool
    retry_permitted: Literal[False] = False
    record_sha256: Sha256

    @model_validator(mode="after")
    def validate_record(self) -> EvidenceInferenceFableCompositeRecordV1:
        if self.provider_outcome == "provider_response":
            valid = (
                self.parsed_batch is not None
                and self.input_tokens is not None
                and self.output_tokens is not None
                and not self.forced_zero_structured_output_direction_and_grounding
            )
        elif self.provider_outcome == "provider_response_unusable":
            valid = (
                self.parsed_batch is None
                and self.forced_zero_structured_output_direction_and_grounding
            )
        else:
            valid = (
                self.origin
                == "source_ambiguous_intention_to_evaluate_failure"
                and self.parsed_batch is None
                and self.input_tokens is None
                and self.output_tokens is None
                and self.forced_zero_structured_output_direction_and_grounding
            )
        if not valid:
            raise ValueError("fable_composite_record_shape_invalid")
        _self_hash(self, "record_sha256", "fable_composite_record_hash_mismatch")
        return self


class EvidenceInferenceFableDescriptiveCompositeV1(_Frozen):
    composite_version: Literal["evidence-inference-fable-descriptive-composite-v1"] = (
        "evidence-inference-fable-descriptive-composite-v1"
    )
    status: Literal["complete_descriptive_intention_to_evaluate_roster"] = (
        "complete_descriptive_intention_to_evaluate_roster"
    )
    retrospective_plan_sha256: Sha256
    source_terminal_sha256: Sha256
    source_snapshot_sha256: Sha256
    continuation_plan_sha256: Sha256
    continuation_terminal_sha256: Sha256
    records: list[EvidenceInferenceFableCompositeRecordV1]
    record_roster_sha256: Sha256
    request_count: Annotated[StrictInt, Field(ge=2)]
    provider_receipt_count: Count
    ambiguous_attempt_failure_count: Literal[1] = 1
    ambiguous_locked_question_failure_count: Annotated[StrictInt, Field(ge=1)]
    intention_to_evaluate_denominator_preserved: Literal[True] = True
    descriptive_full_roster_scoring_permitted: Literal[True] = True
    clean_mechanics_gate_authority: Literal[False] = False
    mechanics_reliability_claim_permitted: Literal[False] = False
    inferential_effect_claim_permitted: Literal[False] = False
    confirmatory_authority: Literal[False] = False
    scientific_claim_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    gate_blocker: Literal[
        "observed_ambiguous_transport_attempt_invalidates_clean_mechanics_pass"
    ] = "observed_ambiguous_transport_attempt_invalidates_clean_mechanics_pass"
    labels_opened: Literal[False] = False
    provider_calls_made_by_composite_materialization: Literal[0] = 0
    composite_sha256: Sha256

    @model_validator(mode="after")
    def validate_composite(self) -> EvidenceInferenceFableDescriptiveCompositeV1:
        if (
            self.request_count != len(self.records)
            or self.provider_receipt_count != self.request_count - 1
            or [item.original_execution_index for item in self.records]
            != list(range(self.request_count))
            or len({item.request_key for item in self.records}) != self.request_count
            or sum(
                item.provider_outcome == "transport_failed_or_ambiguous"
                for item in self.records
            )
            != 1
            or self.record_roster_sha256
            != hash_canonical([item.record_sha256 for item in self.records])
        ):
            raise ValueError("fable_composite_roster_mismatch")
        _self_hash(self, "composite_sha256", "fable_composite_hash_mismatch")
        return self


def _record_from_result(
    *,
    index: int,
    request: ArticleBatchRequestV1,
    origin: Literal["source_receipt", "continuation_receipt"],
    artifact_sha256: str,
    result: EvidenceInferenceFableProviderResultV1,
) -> EvidenceInferenceFableCompositeRecordV1:
    completed = result.outcome == "completed"
    payload = {
        "original_execution_index": index,
        "request_key": request.request_key,
        "article_request_sha256": request.request_sha256,
        "origin": origin,
        "source_artifact_sha256": artifact_sha256,
        "locked_question_count": request.question_count,
        "provider_outcome": (
            "provider_response" if completed else "provider_response_unusable"
        ),
        "parsed_batch": result.parsed_json if completed else None,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "accounted_cost_usd_micros": result.charged_cost_usd_micros,
        "forced_zero_structured_output_direction_and_grounding": not completed,
        "retry_permitted": False,
    }
    return EvidenceInferenceFableCompositeRecordV1.model_validate(
        {**payload, "record_sha256": hash_canonical(payload)}
    )


def freeze_evidence_inference_fable_descriptive_composite_v1(
    *,
    source_workspace: Path,
    continuation_workspace: Path,
    retrospective_plan: EvidenceInferenceFableRetrospectivePlanV1,
) -> EvidenceInferenceFableDescriptiveCompositeV1:
    """Join immutable lineages without converting the result into a gate pass."""

    source_before = _replay_poisoned_source(
        source_workspace=source_workspace,
        retrospective_plan=retrospective_plan,
    )
    continuation_root = continuation_workspace.resolve(strict=True)
    continuation_terminal = validate_evidence_inference_fable_continuation_workspace_v1(
        workspace=continuation_root,
        source_workspace=source_workspace,
        retrospective_plan=retrospective_plan,
    )
    if (
        continuation_terminal.status != "completed"
        or not continuation_terminal.descriptive_composite_materialization_permitted
    ):
        raise EvidenceInferenceFableContinuationError(
            "fable_composite_requires_completed_continuation"
        )
    continuation_plan = EvidenceInferenceFableContinuationPlanV1.model_validate(
        _read_object(
            continuation_root / "00-continuation-plan.json",
            root=continuation_root,
        )
    )
    records: list[EvidenceInferenceFableCompositeRecordV1] = []
    for index, (request, receipt) in enumerate(
        zip(
            retrospective_plan.roster[: source_before.terminal.completed_request_count],
            source_before.completed_receipts,
            strict=True,
        )
    ):
        records.append(
            _record_from_result(
                index=index,
                request=request,
                origin="source_receipt",
                artifact_sha256=receipt.receipt_sha256,
                result=receipt.provider_result,
            )
        )
    failed_index = source_before.terminal.completed_request_count
    failed_request = retrospective_plan.roster[failed_index]
    forced_payload = {
        "original_execution_index": failed_index,
        "request_key": failed_request.request_key,
        "article_request_sha256": failed_request.request_sha256,
        "origin": "source_ambiguous_intention_to_evaluate_failure",
        "source_artifact_sha256": source_before.incident.incident_sha256,
        "locked_question_count": failed_request.question_count,
        "provider_outcome": "transport_failed_or_ambiguous",
        "parsed_batch": None,
        "input_tokens": None,
        "output_tokens": None,
        "accounted_cost_usd_micros": source_before.incident.charged_cost_usd_micros,
        "forced_zero_structured_output_direction_and_grounding": True,
        "retry_permitted": False,
    }
    records.append(
        EvidenceInferenceFableCompositeRecordV1.model_validate(
            {**forced_payload, "record_sha256": hash_canonical(forced_payload)}
        )
    )
    for request in continuation_plan.continuation_requests:
        original = retrospective_plan.roster[request.original_execution_index]
        receipt = EvidenceInferenceFableContinuationReceiptV1.model_validate(
            _read_object(
                continuation_root / "receipts" / f"{request.request_key}.json",
                root=continuation_root,
            )
        )
        records.append(
            _record_from_result(
                index=request.original_execution_index,
                request=original,
                origin="continuation_receipt",
                artifact_sha256=receipt.receipt_sha256,
                result=receipt.provider_result,
            )
        )
    source_after = _replay_poisoned_source(
        source_workspace=source_workspace,
        retrospective_plan=retrospective_plan,
    )
    if source_after.snapshot_sha256 != source_before.snapshot_sha256:
        raise EvidenceInferenceFableContinuationError(
            "fable_composite_source_changed_during_materialization"
        )
    payload = {
        "composite_version": "evidence-inference-fable-descriptive-composite-v1",
        "status": "complete_descriptive_intention_to_evaluate_roster",
        "retrospective_plan_sha256": retrospective_plan.plan_sha256,
        "source_terminal_sha256": source_before.terminal.terminal_sha256,
        "source_snapshot_sha256": source_before.snapshot_sha256,
        "continuation_plan_sha256": continuation_plan.plan_sha256,
        "continuation_terminal_sha256": continuation_terminal.terminal_sha256,
        "records": records,
        "record_roster_sha256": hash_canonical([item.record_sha256 for item in records]),
        "request_count": retrospective_plan.request_count,
        "provider_receipt_count": retrospective_plan.request_count - 1,
        "ambiguous_attempt_failure_count": 1,
        "ambiguous_locked_question_failure_count": failed_request.question_count,
        "intention_to_evaluate_denominator_preserved": True,
        "descriptive_full_roster_scoring_permitted": True,
        "clean_mechanics_gate_authority": False,
        "mechanics_reliability_claim_permitted": False,
        "inferential_effect_claim_permitted": False,
        "confirmatory_authority": False,
        "scientific_claim_authority": False,
        "claim_release_authority": False,
        "gate_blocker": (
            "observed_ambiguous_transport_attempt_invalidates_clean_mechanics_pass"
        ),
        "labels_opened": False,
        "provider_calls_made_by_composite_materialization": 0,
    }
    return EvidenceInferenceFableDescriptiveCompositeV1.model_validate(
        {**payload, "composite_sha256": hash_canonical(payload)}
    )


def write_evidence_inference_fable_descriptive_composite_v1(
    *, path: Path, composite: EvidenceInferenceFableDescriptiveCompositeV1
) -> None:
    """Persist only to a new path; never replace a prior runtime artifact."""

    if path.exists() or path.is_symlink():
        raise EvidenceInferenceFableContinuationError(
            "fable_composite_output_must_be_fresh"
        )
    atomic_write_json(path, composite)
