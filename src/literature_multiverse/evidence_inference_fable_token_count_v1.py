"""Optional exact-once token-count preflight for frozen Fable paired requests."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from pydantic import ConfigDict, Field, StrictInt, model_validator

from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    EvidenceInferenceFablePreparedRuntimeV1,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel

Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]


class EvidenceInferenceFableTokenCountError(ValueError):
    pass


class _Frozen(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False)


def _self_hash(value: _Frozen, field: str) -> None:
    if getattr(value, field) != hash_canonical(value.model_dump(mode="json", exclude={field})):
        raise ValueError("fable_token_count_self_hash_mismatch")


class EvidenceInferenceFableCountAuthorizationV1(_Frozen):
    authorization_version: Literal["evidence-inference-fable-count-authorization-v1"] = (
        "evidence-inference-fable-count-authorization-v1"
    )
    prepared_sha256: Sha256
    surface_roster_sha256: Sha256
    authorized_request_keys: list[str]
    authorized_roster_sha256: Sha256
    whole_pair_authorization: Literal[True] = True
    maximum_count_calls: Annotated[StrictInt, Field(ge=2)]
    sdk_retries_permitted: Literal[0] = 0
    application_retries_permitted: Literal[0] = 0
    authorization_sha256: Sha256

    @model_validator(mode="after")
    def validate_auth(self) -> EvidenceInferenceFableCountAuthorizationV1:
        if (
            len(self.authorized_request_keys) != self.maximum_count_calls
            or self.maximum_count_calls % 2
            or len(set(self.authorized_request_keys)) != self.maximum_count_calls
            or self.authorized_roster_sha256 != hash_canonical(self.authorized_request_keys)
        ):
            raise ValueError("fable_token_count_authorization_roster_invalid")
        _self_hash(self, "authorization_sha256")
        return self


def freeze_evidence_inference_fable_count_authorization_v1(
    prepared: EvidenceInferenceFablePreparedRuntimeV1,
) -> EvidenceInferenceFableCountAuthorizationV1:
    keys = [surface.request_key for surface in prepared.surfaces]
    payload = {
        "authorization_version": "evidence-inference-fable-count-authorization-v1",
        "prepared_sha256": prepared.prepared_sha256,
        "surface_roster_sha256": prepared.surface_roster_sha256,
        "authorized_request_keys": keys,
        "authorized_roster_sha256": hash_canonical(keys),
        "whole_pair_authorization": True,
        "maximum_count_calls": len(keys),
        "sdk_retries_permitted": 0,
        "application_retries_permitted": 0,
    }
    return EvidenceInferenceFableCountAuthorizationV1.model_validate(
        {**payload, "authorization_sha256": hash_canonical(payload)}
    )


class EvidenceInferenceFableCountIntentV1(_Frozen):
    intent_version: Literal["evidence-inference-fable-count-intent-v1"] = (
        "evidence-inference-fable-count-intent-v1"
    )
    authorization_sha256: Sha256
    request_key: str
    surface_sha256: Sha256
    wire_call_sha256: Sha256
    model: Literal["claude-fable-5"]
    schema_sha256: Sha256
    pair_index: Annotated[StrictInt, Field(ge=0)]
    permitted_attempts: Literal[1] = 1
    retries_permitted: Literal[0] = 0
    intent_sha256: Sha256

    @model_validator(mode="after")
    def validate_intent(self) -> EvidenceInferenceFableCountIntentV1:
        _self_hash(self, "intent_sha256")
        return self


class EvidenceInferenceFableCountReceiptV1(_Frozen):
    receipt_version: Literal["evidence-inference-fable-count-receipt-v1"] = (
        "evidence-inference-fable-count-receipt-v1"
    )
    intent_sha256: Sha256
    request_key: str
    surface_sha256: Sha256
    wire_call_sha256: Sha256
    counted_input_tokens: Annotated[StrictInt, Field(ge=1, le=1000000)]
    count_attempts: Literal[1] = 1
    sdk_retries: Literal[0] = 0
    tightened_request_liability_usd_micros: Annotated[StrictInt, Field(ge=1)]
    receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> EvidenceInferenceFableCountReceiptV1:
        _self_hash(self, "receipt_sha256")
        return self


class EvidenceInferenceFableCountIncidentV1(_Frozen):
    incident_version: Literal["evidence-inference-fable-count-incident-v1"] = (
        "evidence-inference-fable-count-incident-v1"
    )
    status: Literal["terminal_ambiguous_count_poison"] = "terminal_ambiguous_count_poison"
    request_key: str
    intent_sha256: Sha256
    kind: Literal["orphan_count_intent", "count_call_raised", "count_result_invalid"]
    retry_permitted: Literal[False] = False
    incident_sha256: Sha256

    @model_validator(mode="after")
    def validate_incident(self) -> EvidenceInferenceFableCountIncidentV1:
        _self_hash(self, "incident_sha256")
        return self


class EvidenceInferenceFableCountTerminalV1(_Frozen):
    terminal_version: Literal["evidence-inference-fable-count-terminal-v1"] = (
        "evidence-inference-fable-count-terminal-v1"
    )
    status: Literal["completed_certified", "terminal_ambiguous_count_poison"]
    prepared_sha256: Sha256
    authorization_sha256: Sha256
    receipt_sha256s: list[Sha256]
    certified_request_liabilities_usd_micros: dict[str, Annotated[StrictInt, Field(ge=1)]]
    certified_total_liability_usd_micros: Annotated[StrictInt, Field(ge=0)]
    full_context_fallback_preserved: Literal[True] = True
    labels_opened: Literal[False] = False
    terminal_sha256: Sha256

    @model_validator(mode="after")
    def validate_terminal(self) -> EvidenceInferenceFableCountTerminalV1:
        if self.certified_total_liability_usd_micros != sum(
            self.certified_request_liabilities_usd_micros.values()
        ):
            raise ValueError("fable_token_count_terminal_sum_mismatch")
        if self.status != "completed_certified" and self.certified_request_liabilities_usd_micros:
            raise ValueError("fable_token_count_partial_certification_forbidden")
        _self_hash(self, "terminal_sha256")
        return self


class EvidenceInferenceFableTokenCounterProtocol(Protocol):
    def count_tokens(self, wire_kwargs: Mapping[str, Any]) -> int: ...


class AnthropicFableTokenCounterV1:
    """Optional live count adapter with an explicitly zero-retry SDK client."""

    def __init__(self, client: Any) -> None:
        self.client = client

    @classmethod
    def from_anthropic_sdk(cls) -> AnthropicFableTokenCounterV1:
        import anthropic  # type: ignore[import-not-found]

        if str(getattr(anthropic, "__version__", "")) != "0.120.2":
            raise EvidenceInferenceFableTokenCountError("fable_token_count_sdk_drift")
        http_client = anthropic.DefaultHttpxClient(
            timeout=600.0, trust_env=False, follow_redirects=False
        )
        return cls(
            anthropic.Anthropic(
                base_url="https://api.anthropic.com",
                default_headers={"anthropic-version": "2023-06-01"},
                http_client=http_client,
                max_retries=0,
                timeout=600.0,
            )
        )

    def count_tokens(self, wire_kwargs: Mapping[str, Any]) -> int:
        response = self.client.messages.count_tokens(**dict(wire_kwargs))
        return response.input_tokens


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceInferenceFableTokenCountError("fable_token_count_artifact_unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceInferenceFableTokenCountError(
            "fable_token_count_artifact_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise EvidenceInferenceFableTokenCountError("fable_token_count_artifact_invalid")
    return value


@contextmanager
def _lock(workspace: Path) -> Any:
    workspace.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(workspace / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def _intent(
    auth: EvidenceInferenceFableCountAuthorizationV1, surface: Any, index: int
) -> EvidenceInferenceFableCountIntentV1:
    payload = {
        "intent_version": "evidence-inference-fable-count-intent-v1",
        "authorization_sha256": auth.authorization_sha256,
        "request_key": surface.request_key,
        "surface_sha256": surface.surface_sha256,
        "wire_call_sha256": surface.wire_call_sha256,
        "model": surface.model,
        "schema_sha256": hash_canonical(surface.wire_schema),
        "pair_index": index // 2,
        "permitted_attempts": 1,
        "retries_permitted": 0,
    }
    return EvidenceInferenceFableCountIntentV1.model_validate(
        {**payload, "intent_sha256": hash_canonical(payload)}
    )


def _validate_count_receipt_replay_v1(
    *,
    authorization: EvidenceInferenceFableCountAuthorizationV1,
    surface: Any,
    index: int,
    intent: EvidenceInferenceFableCountIntentV1,
    receipt: EvidenceInferenceFableCountReceiptV1,
) -> None:
    expected_intent = _intent(authorization, surface, index)
    expected_liability = (
        receipt.counted_input_tokens * 10 + surface.max_output_tokens * 50
    )
    if (
        intent != expected_intent
        or receipt.intent_sha256 != intent.intent_sha256
        or receipt.request_key != surface.request_key
        or receipt.surface_sha256 != surface.surface_sha256
        or receipt.wire_call_sha256 != surface.wire_call_sha256
        or receipt.tightened_request_liability_usd_micros != expected_liability
        or expected_liability > surface.request_hard_liability_usd_micros
    ):
        raise EvidenceInferenceFableTokenCountError(
            "fable_token_count_receipt_binding_mismatch"
        )


def execute_evidence_inference_fable_token_count_v1(
    *,
    workspace: Path,
    prepared: EvidenceInferenceFablePreparedRuntimeV1,
    authorization: EvidenceInferenceFableCountAuthorizationV1,
    counter: EvidenceInferenceFableTokenCounterProtocol,
) -> EvidenceInferenceFableCountTerminalV1:
    if authorization.prepared_sha256 != prepared.prepared_sha256:
        raise EvidenceInferenceFableTokenCountError("fable_token_count_binding_mismatch")
    if (
        authorization.surface_roster_sha256 != prepared.surface_roster_sha256
        or authorization.authorized_request_keys
        != [surface.request_key for surface in prepared.surfaces]
    ):
        raise EvidenceInferenceFableTokenCountError(
            "fable_token_count_authorization_surface_roster_mismatch"
        )
    with _lock(workspace):
        auth_path = workspace / "authorization.json"
        if auth_path.exists():
            if (
                EvidenceInferenceFableCountAuthorizationV1.model_validate(_read(auth_path))
                != authorization
            ):
                raise EvidenceInferenceFableTokenCountError(
                    "fable_token_count_auth_replay_mismatch"
                )
        else:
            atomic_write_json(auth_path, authorization)
        terminal_path = workspace / "terminal.json"
        if terminal_path.exists():
            return validate_evidence_inference_fable_token_count_v1(
                workspace=workspace, prepared=prepared, _already_locked=True
            )
        for name in ("intents", "receipts", "incidents"):
            directory = workspace / name
            if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
                raise EvidenceInferenceFableTokenCountError(
                    "fable_token_count_artifact_directory_unsafe"
                )
            directory.mkdir(exist_ok=True)
        request_keys = [surface.request_key for surface in prepared.surfaces]
        receipt_paths = list((workspace / "receipts").iterdir())
        if any(
            path.is_symlink()
            or not path.is_file()
            or path.suffix != ".json"
            or path.stem not in request_keys
            for path in receipt_paths
        ):
            raise EvidenceInferenceFableTokenCountError(
                "fable_token_count_receipt_prefix_invalid"
            )
        receipt_keys = {path.stem for path in receipt_paths}
        if receipt_keys != set(request_keys[: len(receipt_keys)]):
            raise EvidenceInferenceFableTokenCountError(
                "fable_token_count_receipt_prefix_invalid"
            )
        for index, surface in enumerate(prepared.surfaces[: len(receipt_keys)]):
            intent = EvidenceInferenceFableCountIntentV1.model_validate(
                _read(workspace / "intents" / f"{surface.request_key}.json")
            )
            receipt = EvidenceInferenceFableCountReceiptV1.model_validate(
                _read(workspace / "receipts" / f"{surface.request_key}.json")
            )
            _validate_count_receipt_replay_v1(
                authorization=authorization,
                surface=surface,
                index=index,
                intent=intent,
                receipt=receipt,
            )
        receipts: list[EvidenceInferenceFableCountReceiptV1] = []
        for index, surface in enumerate(prepared.surfaces):
            intent = _intent(authorization, surface, index)
            ip = workspace / "intents" / f"{surface.request_key}.json"
            rp = workspace / "receipts" / f"{surface.request_key}.json"
            xp = workspace / "incidents" / f"{surface.request_key}.json"
            if rp.exists():
                receipt = EvidenceInferenceFableCountReceiptV1.model_validate(
                    _read(rp)
                )
                intent = EvidenceInferenceFableCountIntentV1.model_validate(
                    _read(ip)
                )
                _validate_count_receipt_replay_v1(
                    authorization=authorization,
                    surface=surface,
                    index=index,
                    intent=intent,
                    receipt=receipt,
                )
                receipts.append(receipt)
                continue
            kind = None
            if ip.exists():
                kind = "orphan_count_intent"
            else:
                atomic_write_json(ip, intent)
                try:
                    count = counter.count_tokens(
                        {
                            "model": surface.model,
                            "system": surface.system,
                            "messages": [{"role": "user", "content": surface.prompt}],
                            "output_config": {
                                "effort": "high",
                                "format": {"type": "json_schema", "schema": surface.wire_schema},
                            },
                        }
                    )
                except Exception:
                    kind = "count_call_raised"
                else:
                    if type(count) is not int or not 1 <= count <= 1000000:
                        kind = "count_result_invalid"
                    else:
                        liability = count * 10 + surface.max_output_tokens * 50
                        base = {
                            "receipt_version": "evidence-inference-fable-count-receipt-v1",
                            "intent_sha256": intent.intent_sha256,
                            "request_key": surface.request_key,
                            "surface_sha256": surface.surface_sha256,
                            "wire_call_sha256": surface.wire_call_sha256,
                            "counted_input_tokens": count,
                            "count_attempts": 1,
                            "sdk_retries": 0,
                            "tightened_request_liability_usd_micros": liability,
                        }
                        receipt = EvidenceInferenceFableCountReceiptV1.model_validate(
                            {**base, "receipt_sha256": hash_canonical(base)}
                        )
                        atomic_write_json(rp, receipt)
                        receipts.append(receipt)
            if kind is not None:
                base = {
                    "incident_version": "evidence-inference-fable-count-incident-v1",
                    "status": "terminal_ambiguous_count_poison",
                    "request_key": surface.request_key,
                    "intent_sha256": intent.intent_sha256,
                    "kind": kind,
                    "retry_permitted": False,
                }
                incident = EvidenceInferenceFableCountIncidentV1.model_validate(
                    {**base, "incident_sha256": hash_canonical(base)}
                )
                atomic_write_json(xp, incident)
                payload = {
                    "terminal_version": "evidence-inference-fable-count-terminal-v1",
                    "status": "terminal_ambiguous_count_poison",
                    "prepared_sha256": prepared.prepared_sha256,
                    "authorization_sha256": authorization.authorization_sha256,
                    "receipt_sha256s": [],
                    "certified_request_liabilities_usd_micros": {},
                    "certified_total_liability_usd_micros": 0,
                    "full_context_fallback_preserved": True,
                    "labels_opened": False,
                }
                terminal = EvidenceInferenceFableCountTerminalV1.model_validate(
                    {**payload, "terminal_sha256": hash_canonical(payload)}
                )
                atomic_write_json(terminal_path, terminal)
                return terminal
        liabilities = {r.request_key: r.tightened_request_liability_usd_micros for r in receipts}
        payload = {
            "terminal_version": "evidence-inference-fable-count-terminal-v1",
            "status": "completed_certified",
            "prepared_sha256": prepared.prepared_sha256,
            "authorization_sha256": authorization.authorization_sha256,
            "receipt_sha256s": [r.receipt_sha256 for r in receipts],
            "certified_request_liabilities_usd_micros": liabilities,
            "certified_total_liability_usd_micros": sum(liabilities.values()),
            "full_context_fallback_preserved": True,
            "labels_opened": False,
        }
        terminal = EvidenceInferenceFableCountTerminalV1.model_validate(
            {**payload, "terminal_sha256": hash_canonical(payload)}
        )
        atomic_write_json(terminal_path, terminal)
        return terminal


def validate_evidence_inference_fable_token_count_v1(
    *,
    workspace: Path,
    prepared: EvidenceInferenceFablePreparedRuntimeV1,
    _already_locked: bool = False,
) -> EvidenceInferenceFableCountTerminalV1:
    def replay() -> EvidenceInferenceFableCountTerminalV1:
        auth = EvidenceInferenceFableCountAuthorizationV1.model_validate(
            _read(workspace / "authorization.json")
        )
        terminal = EvidenceInferenceFableCountTerminalV1.model_validate(
            _read(workspace / "terminal.json")
        )
        if (
            auth.prepared_sha256 != prepared.prepared_sha256
            or auth.surface_roster_sha256 != prepared.surface_roster_sha256
            or auth.authorized_request_keys
            != [surface.request_key for surface in prepared.surfaces]
            or terminal.authorization_sha256 != auth.authorization_sha256
        ):
            raise EvidenceInferenceFableTokenCountError("fable_token_count_replay_binding_mismatch")
        keys = [s.request_key for s in prepared.surfaces]
        receipt_files = list((workspace / "receipts").glob("*.json"))
        incident_files = list((workspace / "incidents").glob("*.json"))
        intent_files = list((workspace / "intents").glob("*.json"))
        if any(
            path.stem not in keys or path.is_symlink()
            for path in receipt_files + incident_files + intent_files
        ):
            raise EvidenceInferenceFableTokenCountError("fable_token_count_extra_artifact")
        receipts = [
            EvidenceInferenceFableCountReceiptV1.model_validate(
                _read(workspace / "receipts" / f"{key}.json")
            )
            for key in keys
            if (workspace / "receipts" / f"{key}.json").exists()
        ]
        receipt_keys = [receipt.request_key for receipt in receipts]
        for receipt in receipts:
            offset = keys.index(receipt.request_key)
            surface = prepared.surfaces[offset]
            intent = EvidenceInferenceFableCountIntentV1.model_validate(
                _read(workspace / "intents" / f"{receipt.request_key}.json")
            )
            _validate_count_receipt_replay_v1(
                authorization=auth,
                surface=surface,
                index=offset,
                intent=intent,
                receipt=receipt,
            )
        if terminal.status == "completed_certified":
            if (
                len(receipts) != len(keys)
                or terminal.receipt_sha256s != [r.receipt_sha256 for r in receipts]
                or terminal.certified_request_liabilities_usd_micros
                != {r.request_key: r.tightened_request_liability_usd_micros for r in receipts}
                or incident_files
                or {path.stem for path in intent_files} != set(keys)
            ):
                raise EvidenceInferenceFableTokenCountError(
                    "fable_token_count_complete_replay_mismatch"
                )
        elif (
            len(incident_files) != 1
            or terminal.receipt_sha256s
            or terminal.certified_request_liabilities_usd_micros
            or receipt_keys != keys[: len(receipt_keys)]
            or {path.stem for path in intent_files} != set(keys[: len(receipt_keys) + 1])
        ):
            raise EvidenceInferenceFableTokenCountError(
                "fable_token_count_incident_replay_mismatch"
            )
        else:
            incident = EvidenceInferenceFableCountIncidentV1.model_validate(
                _read(incident_files[0])
            )
            expected_key = keys[len(receipt_keys)]
            intent = EvidenceInferenceFableCountIntentV1.model_validate(
                _read(workspace / "intents" / f"{expected_key}.json")
            )
            if (
                incident.request_key != expected_key
                or incident.intent_sha256 != intent.intent_sha256
            ):
                raise EvidenceInferenceFableTokenCountError(
                    "fable_token_count_incident_binding_mismatch"
                )
        return terminal

    if _already_locked:
        return replay()
    with _lock(workspace):
        return replay()


__all__ = [
    name
    for name in globals()
    if name.startswith("EvidenceInferenceFable")
    or name.startswith("freeze_evidence")
    or name.startswith("execute_evidence")
    or name.startswith("validate_evidence")
]
