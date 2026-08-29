from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from threading import Event
from typing import Any

import pytest
from pydantic import ValidationError

from literature_multiverse.anthropic_bounded_generation import (
    ANTHROPIC_INPUT_RATE_USD_PER_MTOK,
    ANTHROPIC_OUTPUT_RATE_USD_PER_MTOK,
    AnthropicBoundedConfigV1,
    AnthropicBoundedRequestV1,
    AnthropicBoundedResultV1,
    AnthropicCostV1,
    AnthropicUsageV1,
    compile_anthropic_bounded_schema,
    freeze_anthropic_bounded_request,
)
from literature_multiverse.hosted_exact_once import (
    HostedExactOnceAmbiguityIncidentV1,
    HostedExactOnceError,
    HostedExactOnceProviderReceiptV1,
    execute_hosted_exactly_once,
    freeze_hosted_exact_once_cost_authorization,
    freeze_hosted_exact_once_intent,
    validate_hosted_exact_once_outcome,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical

EXECUTION_SHA = "a" * 64
CONTEXT_SHA = "b" * 64
_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def _request(*, request_key: str = "preflight-one") -> AnthropicBoundedRequestV1:
    config = AnthropicBoundedConfigV1(timeout_seconds=30)
    compiled = compile_anthropic_bounded_schema(
        original_schema=_SCHEMA,
        full_acceptance_schema_sha256=hash_canonical(_SCHEMA),
    )
    return freeze_anthropic_bounded_request(
        operation="hosted-exact-once-test",
        request_key=request_key,
        prompt="Return the requested JSON object.",
        system="Return JSON only.",
        max_output_tokens=128,
        compiled_schema=compiled,
        config=config,
        schema_kind="inventory",
        effect_kind=None,
    )


def _result(request: AnthropicBoundedRequestV1) -> AnthropicBoundedResultV1:
    parsed = {"ok": True}
    text = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    usage = AnthropicUsageV1(input_tokens=1, output_tokens=1)
    estimated = (
        Decimal(usage.input_tokens) * ANTHROPIC_INPUT_RATE_USD_PER_MTOK
        + Decimal(usage.output_tokens) * ANTHROPIC_OUTPUT_RATE_USD_PER_MTOK
    ) / Decimal(1_000_000)
    cost = AnthropicCostV1(
        basis="reported_standard_usage",
        input_rate_usd_per_million_tokens=ANTHROPIC_INPUT_RATE_USD_PER_MTOK,
        output_rate_usd_per_million_tokens=ANTHROPIC_OUTPUT_RATE_USD_PER_MTOK,
        estimated_cost_usd=estimated,
        request_cost_ceiling_usd=request.cost_ceiling.request_cost_ceiling_usd,
        charged_cost_upper_bound_usd=request.cost_ceiling.request_cost_ceiling_usd,
    )
    payload: dict[str, Any] = {
        "result_version": "anthropic-bounded-result-v2",
        "provider": "anthropic",
        "request_sha256": request.request_sha256,
        "identity_sha256": request.identity_sha256,
        "config_sha256": request.config_sha256,
        "compiled_schema_sha256": request.compiled_schema_sha256,
        "original_schema_sha256": request.compiled_schema.original_schema_sha256,
        "wire_schema_sha256": request.compiled_schema.wire_schema_sha256,
        "full_acceptance_schema_sha256": request.full_acceptance_schema_sha256,
        "schema_kind": request.schema_kind,
        "effect_kind": request.effect_kind,
        "transport_mode": request.transport_mode,
        "structured_grammar_enforced_by_provider": (
            request.structured_grammar_enforced_by_provider
        ),
        "output_format_present_in_call": request.output_format_present_in_call,
        "model_system_sha256": request.model_system_sha256,
        "model_prompt_sha256": request.model_prompt_sha256,
        "wire_call_sha256": request.expected_wire_call_sha256,
        "transport_attempt_count": 1,
        "sdk_retry_count": 0,
        "outcome": "completed",
        "response_id": "msg_fake_exact_once",
        "response_model": "claude-sonnet-5",
        "stop_reason": "end_turn",
        "text": text,
        "response_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "parsed_json": parsed,
        "parsed_json_sha256": hash_canonical(parsed),
        "usage": usage,
        "cost": cost,
        "failure": None,
    }
    return AnthropicBoundedResultV1.model_validate(
        {**payload, "result_sha256": hash_canonical(payload)}
    )


class FakeClient:
    def __init__(
        self,
        *,
        raises: bool = False,
        required_artifact: Path | None = None,
    ) -> None:
        self.raises = raises
        self.required_artifact = required_artifact
        self.calls: list[str] = []

    def generate(self, request: AnthropicBoundedRequestV1) -> AnthropicBoundedResultV1:
        if self.required_artifact is not None:
            assert self.required_artifact.is_file()
        self.calls.append(request.request_sha256)
        if self.raises:
            raise RuntimeError("provider failure text must not be archived")
        return _result(request)


class WrongAliasClient(FakeClient):
    def generate(self, request: AnthropicBoundedRequestV1) -> AnthropicBoundedResultV1:
        self.calls.append(request.request_sha256)
        payload = _result(request).model_dump(mode="json")
        payload["identity_sha256"] = "c" * 64
        payload["result_sha256"] = hash_canonical(
            {key: value for key, value in payload.items() if key != "result_sha256"}
        )
        return AnthropicBoundedResultV1.model_validate(payload)


class BlockingClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.entered = Event()
        self.release = Event()

    def generate(self, request: AnthropicBoundedRequestV1) -> AnthropicBoundedResultV1:
        self.calls.append(request.request_sha256)
        self.entered.set()
        assert self.release.wait(timeout=5)
        return _result(request)


def _intent_and_authorization(*, source_bearing: bool = False) -> tuple[Any, Any]:
    request = _request()
    intent = freeze_hosted_exact_once_intent(
        execution_bundle_sha256=EXECUTION_SHA,
        phase="preflight",
        source_bearing=source_bearing,
        context_binding_sha256=CONTEXT_SHA,
        request=request,
    )
    authorization = freeze_hosted_exact_once_cost_authorization(
        execution_bundle_sha256=EXECUTION_SHA,
        phase="preflight",
        intents=[intent],
        configured_phase_budget_usd_micros=1_000_000,
    )
    return intent, authorization


def test_completed_request_replays_without_a_second_provider_call(
    tmp_path: Path,
) -> None:
    intent, authorization = _intent_and_authorization()
    client = FakeClient()
    workspace = tmp_path / "exact-once"

    first = execute_hosted_exactly_once(
        workspace=workspace,
        intent=intent,
        authorization=authorization,
        client=client,
    )
    second = execute_hosted_exactly_once(
        workspace=workspace,
        intent=intent,
        authorization=authorization,
        client=client,
    )

    assert isinstance(first, HostedExactOnceProviderReceiptV1)
    assert first == second
    assert len(client.calls) == 1
    assert first.provider_result.parsed_json == {"ok": True}


def test_orphaned_durable_intent_is_poisoned_without_call(tmp_path: Path) -> None:
    intent, authorization = _intent_and_authorization(source_bearing=True)
    workspace = tmp_path / "orphan"
    (workspace / "call-intents").mkdir(parents=True)
    (workspace / "cost-authorizations").mkdir(parents=True)
    atomic_write_json(
        workspace / "cost-authorizations" / "preflight.json",
        authorization,
    )
    atomic_write_json(workspace / "call-intents" / "preflight-one.json", intent)
    client = FakeClient()

    outcome = execute_hosted_exactly_once(
        workspace=workspace,
        intent=intent,
        authorization=authorization,
        client=client,
    )

    assert isinstance(outcome, HostedExactOnceAmbiguityIncidentV1)
    assert outcome.incident_kind == "orphan_intent_observed_on_resume"
    assert outcome.response_observation == "unknown_after_orphaned_intent"
    assert client.calls == []


def test_provider_exception_is_terminal_and_never_retried(tmp_path: Path) -> None:
    intent, authorization = _intent_and_authorization()
    workspace = tmp_path / "provider-exception"
    client = FakeClient(raises=True)

    first = execute_hosted_exactly_once(
        workspace=workspace,
        intent=intent,
        authorization=authorization,
        client=client,
    )
    second = execute_hosted_exactly_once(
        workspace=workspace,
        intent=intent,
        authorization=authorization,
        client=client,
    )

    assert isinstance(first, HostedExactOnceAmbiguityIncidentV1)
    assert first == second
    assert first.incident_kind == "provider_call_raised_after_durable_intent"
    assert first.response_observation == "not_observed_by_executor"
    assert len(client.calls) == 1
    assert "failure text" not in first.model_dump_json()


def test_authorization_is_exact_and_fails_closed_on_budget_or_intent_drift(
    tmp_path: Path,
) -> None:
    intent, authorization = _intent_and_authorization()

    with pytest.raises(ValidationError, match="exceeds_budget"):
        freeze_hosted_exact_once_cost_authorization(
            execution_bundle_sha256=EXECUTION_SHA,
            phase="preflight",
            intents=[intent],
            configured_phase_budget_usd_micros=1,
        )

    changed = freeze_hosted_exact_once_intent(
        execution_bundle_sha256=EXECUTION_SHA,
        phase="preflight",
        source_bearing=True,
        context_binding_sha256=CONTEXT_SHA,
        request=_request(),
    )
    unauthorized_workspace = tmp_path / "unauthorized-intent"
    with pytest.raises(HostedExactOnceError, match="authorized_intent_mismatch"):
        execute_hosted_exactly_once(
            workspace=unauthorized_workspace,
            intent=changed,
            authorization=authorization,
            client=FakeClient(),
        )
    assert not unauthorized_workspace.exists()


def test_cost_authorization_is_durable_before_provider_call(tmp_path: Path) -> None:
    intent, authorization = _intent_and_authorization(source_bearing=True)
    workspace = tmp_path / "durable-authorization"
    authorization_path = workspace / "cost-authorizations" / "preflight.json"
    client = FakeClient(required_artifact=authorization_path)

    outcome = execute_hosted_exactly_once(
        workspace=workspace,
        intent=intent,
        authorization=authorization,
        client=client,
    )

    assert isinstance(outcome, HostedExactOnceProviderReceiptV1)
    saved = json.loads(authorization_path.read_text(encoding="utf-8"))
    assert saved == authorization.model_dump(mode="json")
    assert len(client.calls) == 1


def test_saved_authorization_tamper_fails_without_provider_call(
    tmp_path: Path,
) -> None:
    intent, authorization = _intent_and_authorization()
    workspace = tmp_path / "authorization-tamper"
    authorization_path = workspace / "cost-authorizations" / "preflight.json"
    authorization_path.parent.mkdir(parents=True)
    changed = authorization.model_dump(mode="json")
    changed["configured_phase_budget_usd_micros"] += 1
    changed["authorization_sha256"] = hash_canonical(
        {key: value for key, value in changed.items() if key != "authorization_sha256"}
    )
    atomic_write_json(authorization_path, changed)
    client = FakeClient()

    with pytest.raises(
        HostedExactOnceError,
        match="authorization_replay_mismatch",
    ):
        execute_hosted_exactly_once(
            workspace=workspace,
            intent=intent,
            authorization=authorization,
            client=client,
        )

    assert client.calls == []


def test_provider_result_request_alias_drift_is_terminally_poisoned(
    tmp_path: Path,
) -> None:
    intent, authorization = _intent_and_authorization(source_bearing=True)
    workspace = tmp_path / "wrong-provider-alias"
    client = WrongAliasClient()

    first = execute_hosted_exactly_once(
        workspace=workspace,
        intent=intent,
        authorization=authorization,
        client=client,
    )
    second = execute_hosted_exactly_once(
        workspace=workspace,
        intent=intent,
        authorization=authorization,
        client=client,
    )

    assert isinstance(first, HostedExactOnceAmbiguityIncidentV1)
    assert first == second
    assert first.incident_kind == "provider_result_invalid_after_return"
    assert first.response_observation == "observed_but_invalid"
    assert first.observed_provider_result_sha256 is not None
    assert len(client.calls) == 1


def test_terminal_outcomes_are_externally_replayed_without_a_client(
    tmp_path: Path,
) -> None:
    intent, authorization = _intent_and_authorization()
    workspace = tmp_path / "external-replay"
    expected = execute_hosted_exactly_once(
        workspace=workspace,
        intent=intent,
        authorization=authorization,
        client=FakeClient(),
    )

    replayed = validate_hosted_exact_once_outcome(
        workspace=workspace,
        intent=intent,
        authorization=authorization,
    )

    assert replayed == expected


def test_external_replay_rejects_coherently_rehashed_receipt_drift(
    tmp_path: Path,
) -> None:
    intent, authorization = _intent_and_authorization()
    workspace = tmp_path / "external-replay-tamper"
    execute_hosted_exactly_once(
        workspace=workspace,
        intent=intent,
        authorization=authorization,
        client=FakeClient(),
    )
    receipt_path = workspace / "provider-receipts" / "preflight-one.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_bearing"] = True
    receipt["receipt_sha256"] = hash_canonical(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(
        HostedExactOnceError,
        match="receipt_external_replay_mismatch",
    ):
        validate_hosted_exact_once_outcome(
            workspace=workspace,
            intent=intent,
            authorization=authorization,
        )


def test_concurrent_callers_share_one_provider_attempt(tmp_path: Path) -> None:
    intent, authorization = _intent_and_authorization(source_bearing=True)
    workspace = tmp_path / "concurrent"
    client = BlockingClient()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            execute_hosted_exactly_once,
            workspace=workspace,
            intent=intent,
            authorization=authorization,
            client=client,
        )
        assert client.entered.wait(timeout=5)
        second = executor.submit(
            execute_hosted_exactly_once,
            workspace=workspace,
            intent=intent,
            authorization=authorization,
            client=client,
        )
        client.release.set()
        first_outcome = first.result(timeout=5)
        second_outcome = second.result(timeout=5)

    assert isinstance(first_outcome, HostedExactOnceProviderReceiptV1)
    assert second_outcome == first_outcome
    assert len(client.calls) == 1
