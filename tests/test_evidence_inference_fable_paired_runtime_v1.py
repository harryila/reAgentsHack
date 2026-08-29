from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    AnthropicFablePairedClientV1,
    EvidenceInferenceFablePairedRuntimeError,
    EvidenceInferenceFableProviderResultV1,
    EvidenceInferenceFableTerminalV1,
    authorize_evidence_inference_fable_workspace_v1,
    execute_evidence_inference_fable_paired_v1,
    freeze_evidence_inference_fable_budget_authorization_v1,
    freeze_evidence_inference_fable_budget_authorization_v2,
    freeze_evidence_inference_fable_call_surface_v1,
    freeze_evidence_inference_fable_prepared_runtime_v1,
    prepare_evidence_inference_fable_workspace_v1,
    reconstruct_evidence_inference_fable_prepared_runtime_v1,
    validate_evidence_inference_fable_workspace_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_v1 import (
    ArticleBatchRequestV1,
    EvidenceInferenceFableRetrospectivePlanV1,
)
from literature_multiverse.evidence_inference_fable_token_count_v1 import (
    EvidenceInferenceFableCountTerminalV1,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
ROOT = Path(__file__).resolve().parents[1]


def _fixture(hard_liability: int = 100) -> tuple[Any, Any]:
    items = []
    surfaces = []
    system = "Return JSON only."
    for index, (article, arm) in enumerate(
        [("PMC1", "seed"), ("PMC1", "winner"), ("PMC2", "winner"), ("PMC2", "seed")]
    ):
        prompt = f"article {article} arm {arm}"
        request_sha = hashlib.sha256(f"request-{index}".encode()).hexdigest()
        cost = SimpleNamespace(
            full_context_hard_liability_usd_micros=hard_liability
        )
        item = ArticleBatchRequestV1.model_construct(
            execution_index=index,
            request_key=f"request-{index}",
            article_id=article,
            arm=arm,
            question_count=1,
            system_sha256=hashlib.sha256(system.encode()).hexdigest(),
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            wire_schema_sha256=hash_canonical(SCHEMA),
            max_output_tokens=8192,
            request_sha256=request_sha,
            cost=cost,
        )
        items.append(item)
        surfaces.append(
            freeze_evidence_inference_fable_call_surface_v1(
                roster_item=item, system=system, prompt=prompt, wire_schema=SCHEMA
            )
        )
    plan = EvidenceInferenceFableRetrospectivePlanV1.model_construct(
        plan_sha256="a" * 64, request_roster_sha256="b" * 64, roster=items
    )
    prepared = freeze_evidence_inference_fable_prepared_runtime_v1(plan=plan, surfaces=surfaces)
    return plan, prepared


def _result(surface: Any) -> EvidenceInferenceFableProviderResultV1:
    payload = {
        "result_version": "evidence-inference-fable-provider-result-v1",
        "request_key": surface.request_key,
        "surface_sha256": surface.surface_sha256,
        "transport_attempt_count": 1,
        "sdk_retry_count": 0,
        "outcome": "completed",
        "response_id": f"fake-{surface.request_key}",
        "response_model": "claude-fable-5",
        "parsed_json": {"ok": True},
        "input_tokens": 1,
        "output_tokens": 1,
        "reported_cost_usd_micros": 60,
        "charged_cost_usd_micros": 60,
        "cost_basis": "reported_usage",
        "response_text_sha256": None,
        "failure_code": None,
    }
    return EvidenceInferenceFableProviderResultV1.model_validate(
        {**payload, "result_sha256": hash_canonical(payload)}
    )


def _certified_count_terminal(
    prepared: Any, liability: int = 100
) -> dict[str, Any]:
    liabilities = {
        surface.request_key: liability for surface in prepared.surfaces
    }
    payload = {
        "terminal_version": "evidence-inference-fable-count-terminal-v1",
        "status": "completed_certified",
        "prepared_sha256": prepared.prepared_sha256,
        "authorization_sha256": "c" * 64,
        "receipt_sha256s": [f"{index:x}" * 64 for index in range(1, 5)],
        "certified_request_liabilities_usd_micros": liabilities,
        "certified_total_liability_usd_micros": sum(liabilities.values()),
        "full_context_fallback_preserved": True,
        "labels_opened": False,
    }
    return EvidenceInferenceFableCountTerminalV1.model_validate(
        {**payload, "terminal_sha256": hash_canonical(payload)}
    ).model_dump(mode="json")


class FakeClient:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    def generate(self, surface: Any) -> EvidenceInferenceFableProviderResultV1:
        self.calls.append(surface.request_key)
        if self.fail:
            raise RuntimeError("not archived")
        return _result(surface)


def _workspace(tmp_path: Path, budget: int) -> tuple[Any, Any, Path]:
    plan, prepared = _fixture()
    auth = freeze_evidence_inference_fable_budget_authorization_v1(
        prepared=prepared, configured_total_budget_usd_micros=budget
    )
    workspace = tmp_path / "runtime"
    prepare_evidence_inference_fable_workspace_v1(workspace=workspace, prepared=prepared)
    authorize_evidence_inference_fable_workspace_v1(workspace=workspace, authorization=auth)
    return plan, auth, workspace


def test_complete_pairs_replay_without_more_calls(tmp_path: Path) -> None:
    plan, _, workspace = _workspace(tmp_path, 1000)
    client = FakeClient()
    first = execute_evidence_inference_fable_paired_v1(
        workspace=workspace, plan=plan, client=client
    )
    second = execute_evidence_inference_fable_paired_v1(
        workspace=workspace, plan=plan, client=client
    )
    assert (
        first
        == second
        == validate_evidence_inference_fable_workspace_v1(workspace=workspace, plan=plan)
    )
    assert first.status == "completed"
    assert first.completed_pair_count == 2
    assert client.calls == ["request-0", "request-1", "request-2", "request-3"]


def test_cumulative_gate_stops_before_next_whole_pair(tmp_path: Path) -> None:
    plan, _, workspace = _workspace(tmp_path, 300)
    client = FakeClient()
    terminal = execute_evidence_inference_fable_paired_v1(
        workspace=workspace, plan=plan, client=client
    )
    assert terminal.status == "clean_budget_exhaustion_before_next_pair"
    assert terminal.completed_pair_count == 1
    assert client.calls == ["request-0", "request-1"]


def test_certified_budget_uses_largest_pair_and_never_overshoots(tmp_path: Path) -> None:
    plan, prepared = _fixture()
    certified_terminal = _certified_count_terminal(prepared)
    with pytest.raises(
        EvidenceInferenceFablePairedRuntimeError,
        match="budget_below_certified_largest_pair_liability",
    ):
        freeze_evidence_inference_fable_budget_authorization_v1(
            prepared=prepared,
            configured_total_budget_usd_micros=199,
            certified_count_terminal=certified_terminal,
        )

    # Each pair has a 200-micro-dollar certified liability and the whole roster
    # has a 400-micro-dollar liability. A 300-micro-dollar sequential budget is
    # therefore useful and safe even though it is below the roster-wide sum.
    auth = freeze_evidence_inference_fable_budget_authorization_v1(
        prepared=prepared,
        configured_total_budget_usd_micros=300,
        certified_count_terminal=certified_terminal,
    )
    workspace = tmp_path / "certified-runtime"
    prepare_evidence_inference_fable_workspace_v1(workspace=workspace, prepared=prepared)
    authorize_evidence_inference_fable_workspace_v1(
        workspace=workspace, authorization=auth
    )
    client = FakeClient()
    terminal = execute_evidence_inference_fable_paired_v1(
        workspace=workspace, plan=plan, client=client
    )

    assert auth.configured_total_budget_usd_micros == 300 < 400
    assert terminal.status == "clean_budget_exhaustion_before_next_pair"
    assert terminal.completed_pair_count == 1
    assert terminal.cumulative_reported_spend_usd_micros == 120 <= 300
    assert client.calls == ["request-0", "request-1"]
    assert not (workspace / "intents" / "request-2.json").exists()
    assert terminal == validate_evidence_inference_fable_workspace_v1(
        workspace=workspace, plan=plan
    )


def test_v2_authorization_hash_binds_fixed_1024_token_headroom_and_pair_gate(
    tmp_path: Path,
) -> None:
    plan, prepared = _fixture(hard_liability=20_000)
    certified_terminal = _certified_count_terminal(prepared, liability=500)
    v1 = freeze_evidence_inference_fable_budget_authorization_v1(
        prepared=prepared,
        configured_total_budget_usd_micros=21_480,
        certified_count_terminal=certified_terminal,
    )
    auth = freeze_evidence_inference_fable_budget_authorization_v2(
        prepared=prepared,
        configured_total_budget_usd_micros=21_480,
        certified_count_terminal=certified_terminal,
    )

    assert v1.authorization_version.endswith("authorization-v1")
    assert "certified_input_token_headroom_per_request" not in v1.model_dump(
        mode="json"
    )
    assert auth.certified_input_token_headroom_per_request == 1024
    assert auth.input_token_price_usd_micros_per_token == 10
    assert set(auth.certified_base_request_liabilities_usd_micros.values()) == {
        500
    }
    assert set(auth.certified_request_liabilities_usd_micros.values()) == {
        10_740
    }
    assert auth.authorization_sha256 == hash_canonical(
        auth.model_dump(mode="json", exclude={"authorization_sha256"})
    )

    workspace = tmp_path / "headroom-runtime"
    prepare_evidence_inference_fable_workspace_v1(
        workspace=workspace, prepared=prepared
    )
    authorize_evidence_inference_fable_workspace_v1(
        workspace=workspace, authorization=auth
    )
    client = FakeClient()
    terminal = execute_evidence_inference_fable_paired_v1(
        workspace=workspace, plan=plan, client=client
    )

    # The first pair is admitted against 2 * 10,740, then the second pair is
    # refused because 120 actual micros + 21,480 liability exceeds the cap.
    assert terminal.status == "clean_budget_exhaustion_before_next_pair"
    assert terminal.cumulative_reported_spend_usd_micros == 120
    assert client.calls == ["request-0", "request-1"]
    assert terminal == validate_evidence_inference_fable_workspace_v1(
        workspace=workspace, plan=plan
    )


def test_v2_invalid_return_charges_headroom_liability_and_continues(
    tmp_path: Path,
) -> None:
    plan, prepared = _fixture(hard_liability=20_000)
    auth = freeze_evidence_inference_fable_budget_authorization_v2(
        prepared=prepared,
        configured_total_budget_usd_micros=99_000_000,
        certified_count_terminal=_certified_count_terminal(
            prepared, liability=500
        ),
    )
    workspace = tmp_path / "headroom-invalid-runtime"
    prepare_evidence_inference_fable_workspace_v1(
        workspace=workspace, prepared=prepared
    )
    authorize_evidence_inference_fable_workspace_v1(
        workspace=workspace, authorization=auth
    )

    class InvalidClient(FakeClient):
        def generate(self, surface: Any) -> Any:
            self.calls.append(surface.request_key)
            return {"discarded": "invalid-provider-return"}

    client = InvalidClient()
    terminal = execute_evidence_inference_fable_paired_v1(
        workspace=workspace, plan=plan, client=client
    )
    incident = json.loads(
        (workspace / "incidents" / "request-0.json").read_text(encoding="utf-8")
    )
    assert terminal.status == "completed"
    assert terminal.cumulative_reported_spend_usd_micros == 4 * 10_740
    assert incident["charged_cost_usd_micros"] == 10_740
    assert "invalid-provider-return" not in json.dumps(incident)
    assert client.calls == [f"request-{index}" for index in range(4)]


def test_exception_after_intent_is_failed_continued_and_never_retried(tmp_path: Path) -> None:
    plan, _, workspace = _workspace(tmp_path, 1000)
    client = FakeClient(fail=True)
    first = execute_evidence_inference_fable_paired_v1(
        workspace=workspace, plan=plan, client=client
    )
    second = execute_evidence_inference_fable_paired_v1(
        workspace=workspace, plan=plan, client=client
    )
    assert first == second
    assert first.status == "completed"
    assert first.cumulative_reported_spend_usd_micros == 400
    assert client.calls == ["request-0", "request-1", "request-2", "request-3"]
    for index in range(4):
        receipt = json.loads(
            (workspace / "receipts" / f"request-{index}.json").read_text(encoding="utf-8")
        )
        assert receipt["locked_questions_scored_incorrect"] == 1
        assert (
            receipt["provider_result"]["failure_code"]
            == "provider_call_raised_after_durable_intent"
        )


def test_provider_exception_incident_archives_only_sanitized_bounded_metadata(
    tmp_path: Path,
) -> None:
    plan, _, workspace = _workspace(tmp_path, 1000)
    fake_api_key = "sk-ant-api03-FAKESECRET0123456789"
    fake_bearer = "bearer-token-FAKESECRET0123456789"

    class ProviderTransportError(RuntimeError):
        status_code = 429
        request_id = "req_01SAFE123"

    class DiagnosticClient(FakeClient):
        def generate(self, surface: Any) -> EvidenceInferenceFableProviderResultV1:
            self.calls.append(surface.request_key)
            raise ProviderTransportError(
                f"api_key={fake_api_key} Authorization: Bearer {fake_bearer} " + "x" * 700
            )

    client = DiagnosticClient()
    terminal = execute_evidence_inference_fable_paired_v1(
        workspace=workspace, plan=plan, client=client
    )
    incident_path = workspace / "incidents" / "request-0.json"
    incident_text = incident_path.read_text(encoding="utf-8")
    incident = json.loads(incident_text)

    assert terminal.status == "completed"
    assert client.calls == ["request-0", "request-1", "request-2", "request-3"]
    assert incident["incident_version"] == "evidence-inference-fable-incident-v2"
    assert incident["exception_type"].endswith("ProviderTransportError")
    assert incident["http_status_code"] == 429
    assert incident["provider_request_id"] == "req_01SAFE123"
    assert incident["message_was_truncated"] is True
    assert len(incident["message_redacted"]) <= 512
    assert fake_api_key not in incident_text
    assert fake_bearer not in incident_text
    assert "[REDACTED]" in incident["message_redacted"]
    assert (
        validate_evidence_inference_fable_workspace_v1(workspace=workspace, plan=plan)
        == terminal
    )


def test_incident_first_crash_recovers_failed_receipt_without_provider_retry(
    tmp_path: Path,
) -> None:
    plan, _, workspace = _workspace(tmp_path, 1000)
    execute_evidence_inference_fable_paired_v1(
        workspace=workspace, plan=plan, client=FakeClient(fail=True)
    )
    # Recreate the only durable state possible between the incident-first and
    # derived-receipt writes. All paths are isolated pytest artifacts.
    (workspace / "02-terminal.json").unlink()
    (workspace / "receipts" / "request-0.json").unlink()

    client = FakeClient()
    terminal = execute_evidence_inference_fable_paired_v1(
        workspace=workspace, plan=plan, client=client
    )
    receipt = json.loads(
        (workspace / "receipts" / "request-0.json").read_text(encoding="utf-8")
    )
    assert terminal.status == "completed"
    assert receipt["locked_questions_scored_incorrect"] == 1
    assert client.calls == []


def test_external_replay_rejects_rehashed_unsanitized_incident_diagnostics(
    tmp_path: Path,
) -> None:
    plan, _, workspace = _workspace(tmp_path, 1000)
    client = FakeClient(fail=True)
    execute_evidence_inference_fable_paired_v1(
        workspace=workspace, plan=plan, client=client
    )
    incident_path = workspace / "incidents" / "request-0.json"
    incident = json.loads(incident_path.read_text(encoding="utf-8"))
    incident["message_redacted"] = "api_key=sk-ant-api03-FAKESECRET0123456789"
    incident["incident_sha256"] = hash_canonical(
        {key: value for key, value in incident.items() if key != "incident_sha256"}
    )
    incident_path.write_text(json.dumps(incident), encoding="utf-8")

    with pytest.raises(ValueError, match="fable_incident_diagnostics_unsafe"):
        validate_evidence_inference_fable_workspace_v1(workspace=workspace, plan=plan)


def test_invalid_return_after_intent_is_zero_credit_incident_and_continues(
    tmp_path: Path,
) -> None:
    plan, _, workspace = _workspace(tmp_path, 1000)

    class InvalidClient(FakeClient):
        def generate(self, surface: Any) -> Any:
            self.calls.append(surface.request_key)
            return {"not": "a provider result"}

    client = InvalidClient()
    terminal = execute_evidence_inference_fable_paired_v1(
        workspace=workspace, plan=plan, client=client
    )
    incident = json.loads(
        (workspace / "incidents" / "request-0.json").read_text(encoding="utf-8")
    )
    receipt = json.loads(
        (workspace / "receipts" / "request-0.json").read_text(encoding="utf-8")
    )
    assert terminal.status == "completed"
    assert incident["incident_version"] == "evidence-inference-fable-incident-v2"
    assert incident["kind"] == "provider_result_invalid_after_return"
    assert incident["retry_permitted"] is False
    assert "not" not in json.dumps(incident)
    assert receipt["locked_questions_scored_incorrect"] == 1
    assert receipt["provider_result"]["parsed_json"] is None
    assert client.calls == ["request-0", "request-1", "request-2", "request-3"]
    assert (
        validate_evidence_inference_fable_workspace_v1(workspace=workspace, plan=plan)
        == terminal
    )


def test_external_replay_accepts_legacy_v1_provider_exception_poison(
    tmp_path: Path,
) -> None:
    plan, _, workspace = _workspace(tmp_path, 1000)

    class InvalidClient(FakeClient):
        def generate(self, surface: Any) -> Any:
            self.calls.append(surface.request_key)
            return {"not": "a provider result"}

    execute_evidence_inference_fable_paired_v1(
        workspace=workspace, plan=plan, client=InvalidClient()
    )
    for request_index in range(4):
        receipt = workspace / "receipts" / f"request-{request_index}.json"
        if receipt.exists():
            receipt.unlink()
        if request_index:
            (workspace / "intents" / f"request-{request_index}.json").unlink()
            (workspace / "incidents" / f"request-{request_index}.json").unlink()
    (workspace / "02-terminal.json").unlink()
    incident_path = workspace / "incidents" / "request-0.json"
    incident_v2 = json.loads(incident_path.read_text(encoding="utf-8"))
    incident = {
        "incident_version": "evidence-inference-fable-incident-v1",
        "status": "terminal_ambiguous_attempt_poison",
        "kind": "provider_call_raised_after_durable_intent",
        "intent_sha256": incident_v2["intent_sha256"],
        "request_key": "request-0",
        "charged_cost_usd_micros": 100,
        "cost_basis": "unknown_usage_hard_liability",
        "retry_permitted": False,
    }
    atomic_write_json(
        incident_path,
        {**incident, "incident_sha256": hash_canonical(incident)},
        force=True,
    )
    terminal_payload = {
        "terminal_version": "evidence-inference-fable-terminal-v1",
        "status": "terminal_ambiguous_attempt_poison",
        "prepared_sha256": _fixture()[1].prepared_sha256,
        "authorization_sha256": json.loads(
            (workspace / "01-authorization.json").read_text(encoding="utf-8")
        )["authorization_sha256"],
        "completed_request_count": 0,
        "completed_pair_count": 0,
        "cumulative_reported_spend_usd_micros": 100,
        "cumulative_spend_semantics": (
            "reported_usage_or_unknown_usage_hard_liability"
        ),
        "next_pair_index": 0,
        "full_population_score_permitted": False,
        "extraction_accuracy_authority": False,
        "confirmatory_authority": False,
        "synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    terminal = EvidenceInferenceFableTerminalV1.model_validate(
        {
            **terminal_payload,
            "terminal_sha256": hash_canonical(terminal_payload),
        }
    )
    atomic_write_json(workspace / "02-terminal.json", terminal)

    assert terminal.status == "terminal_ambiguous_attempt_poison"
    assert (
        validate_evidence_inference_fable_workspace_v1(workspace=workspace, plan=plan)
        == terminal
    )


def test_orphaned_intent_is_poisoned_without_provider_call(tmp_path: Path) -> None:
    plan, _, workspace = _workspace(tmp_path, 1000)
    # Simulate a process death after the durable intent and before a receipt.
    from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
        EvidenceInferenceFableIntentV1,
    )
    from literature_multiverse.lineage import atomic_write_json

    prepared = _fixture()[1]
    auth = freeze_evidence_inference_fable_budget_authorization_v1(
        prepared=prepared, configured_total_budget_usd_micros=1000
    )
    surface = prepared.surfaces[0]
    base = {
        "intent_version": "evidence-inference-fable-intent-v1",
        "prepared_sha256": prepared.prepared_sha256,
        "authorization_sha256": auth.authorization_sha256,
        "pair_index": 0,
        "request_key": surface.request_key,
        "surface": surface,
        "cumulative_reported_spend_before_pair_usd_micros": 0,
        "pair_hard_liability_usd_micros": 200,
        "permitted_provider_attempts": 1,
        "application_retries_permitted": 0,
        "sdk_retries_permitted": 0,
        "orphan_or_ambiguous_attempt_is_terminal": True,
    }
    intent = EvidenceInferenceFableIntentV1.model_validate(
        {**base, "intent_sha256": hash_canonical(base)}
    )
    (workspace / "intents").mkdir()
    atomic_write_json(workspace / "intents" / "request-0.json", intent)
    client = FakeClient()
    terminal = execute_evidence_inference_fable_paired_v1(
        workspace=workspace, plan=plan, client=client
    )
    assert terminal.status == "terminal_ambiguous_attempt_poison"
    assert terminal.cumulative_reported_spend_usd_micros == 100
    assert client.calls == []


def test_reported_usage_cost_must_equal_exact_token_cost() -> None:
    _, prepared = _fixture()
    payload = _result(prepared.surfaces[0]).model_dump(
        mode="json", exclude={"result_sha256"}
    )
    payload["reported_cost_usd_micros"] = 1
    payload["charged_cost_usd_micros"] = 1
    with pytest.raises(ValidationError, match="fable_result_shape_invalid"):
        EvidenceInferenceFableProviderResultV1.model_validate(
            {**payload, "result_sha256": hash_canonical(payload)}
        )


def test_external_replay_rejects_self_hashed_archived_intent_drift(
    tmp_path: Path,
) -> None:
    plan, _, workspace = _workspace(tmp_path, 1000)
    execute_evidence_inference_fable_paired_v1(
        workspace=workspace, plan=plan, client=FakeClient()
    )
    intent_path = workspace / "intents" / "request-0.json"
    intent = json.loads(intent_path.read_text())
    intent["pair_hard_liability_usd_micros"] = 199
    intent["intent_sha256"] = hash_canonical(
        {key: value for key, value in intent.items() if key != "intent_sha256"}
    )
    intent_path.write_text(json.dumps(intent))
    receipt_path = workspace / "receipts" / "request-0.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["intent_sha256"] = intent["intent_sha256"]
    receipt["receipt_sha256"] = hash_canonical(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="archived_intent_external_replay_mismatch"):
        validate_evidence_inference_fable_workspace_v1(
            workspace=workspace, plan=plan
        )


def test_symlinked_runtime_artifact_directory_fails_before_call(
    tmp_path: Path,
) -> None:
    plan, _, workspace = _workspace(tmp_path, 1000)
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "intents").symlink_to(outside, target_is_directory=True)
    client = FakeClient()
    with pytest.raises(ValueError, match="runtime_artifact_directory_unsafe"):
        execute_evidence_inference_fable_paired_v1(
            workspace=workspace, plan=plan, client=client
        )
    assert client.calls == []


def test_dynamic_caps_and_fable_high_wire_surface_are_bound() -> None:
    _, prepared = _fixture()
    assert all(surface.max_output_tokens == 8192 for surface in prepared.surfaces)
    assert all(
        surface.model == "claude-fable-5" and surface.effort == "high"
        for surface in prepared.surfaces
    )
    assert all(surface.sdk_max_retries == 0 for surface in prepared.surfaces)


def test_malformed_structured_response_is_cost_bearing_failed_receipt(
    tmp_path: Path,
) -> None:
    plan, prepared = _fixture()
    for item in plan.roster:
        item.cost.full_context_hard_liability_usd_micros = 200
    response = SimpleNamespace(
        id="fake-malformed",
        model="claude-fable-5",
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="not-json")],
        usage=SimpleNamespace(input_tokens=3, output_tokens=2),
    )
    messages = SimpleNamespace(create=lambda **_: response)
    client = AnthropicFablePairedClientV1(SimpleNamespace(messages=messages))
    result = client.generate(prepared.surfaces[0])
    auth = freeze_evidence_inference_fable_budget_authorization_v1(
        prepared=prepared, configured_total_budget_usd_micros=1000
    )
    workspace = tmp_path / "failed-receipt"
    prepare_evidence_inference_fable_workspace_v1(workspace=workspace, prepared=prepared)
    authorize_evidence_inference_fable_workspace_v1(workspace=workspace, authorization=auth)
    terminal = execute_evidence_inference_fable_paired_v1(
        workspace=workspace, plan=plan, client=client
    )
    receipt = (workspace / "receipts" / "request-0.json").read_text()
    assert result.outcome == "failed"
    assert result.failure_code == "response_json_invalid"
    assert result.reported_cost_usd_micros == 130
    assert terminal.status == "completed"
    assert '"locked_questions_scored_incorrect":1' in receipt


def test_external_replay_rejects_unknown_artifact(tmp_path: Path) -> None:
    plan, _, workspace = _workspace(tmp_path, 1000)
    execute_evidence_inference_fable_paired_v1(workspace=workspace, plan=plan, client=FakeClient())
    (workspace / "receipts" / "unknown.json").write_text("{}")
    with pytest.raises(ValueError, match="unknown_request_artifact"):
        validate_evidence_inference_fable_workspace_v1(workspace=workspace, plan=plan)


def test_exact_surface_reconstruction_is_label_safe(monkeypatch: Any) -> None:
    original_text = Path.read_text
    original_bytes = Path.read_bytes

    def guard(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.suffix == ".jsonl" or path.name == "annotations_merged.csv":
            raise AssertionError("benchmark label payload opened")
        method = original_bytes if kwargs.pop("binary", False) else original_text
        return method(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guard)
    monkeypatch.setattr(
        Path, "read_bytes", lambda path, *args, **kwargs: guard(path, *args, binary=True, **kwargs)
    )
    plan, prepared = reconstruct_evidence_inference_fable_prepared_runtime_v1(
        repository_root=ROOT, mode="pilot30_paired"
    )
    assert len(prepared.surfaces) == plan.request_count == 14
    assert [surface.max_output_tokens for surface in prepared.surfaces] == [
        item.max_output_tokens for item in plan.roster
    ]


def test_workspace_lock_serializes_competing_executors(tmp_path: Path) -> None:
    plan, _, workspace = _workspace(tmp_path, 1000)

    class BlockingClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.entered = Event()
            self.release = Event()

        def generate(self, surface: Any) -> EvidenceInferenceFableProviderResultV1:
            self.calls.append(surface.request_key)
            if len(self.calls) == 1:
                self.entered.set()
                assert self.release.wait(5)
            return _result(surface)

    client = BlockingClient()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            execute_evidence_inference_fable_paired_v1,
            workspace=workspace,
            plan=plan,
            client=client,
        )
        assert client.entered.wait(5)
        second = pool.submit(
            execute_evidence_inference_fable_paired_v1,
            workspace=workspace,
            plan=plan,
            client=client,
        )
        client.release.set()
        assert first.result() == second.result()
    assert client.calls == ["request-0", "request-1", "request-2", "request-3"]


def test_live_factory_sets_sdk_max_retries_zero(monkeypatch: Any) -> None:
    observed: dict[str, Any] = {}

    def anthropic_factory(**kwargs: Any) -> Any:
        observed.update(kwargs)
        return SimpleNamespace(messages=SimpleNamespace())

    fake_sdk = SimpleNamespace(
        __version__="0.120.2",
        DefaultHttpxClient=lambda **kwargs: SimpleNamespace(**kwargs),
        Anthropic=anthropic_factory,
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake_sdk)
    AnthropicFablePairedClientV1.from_anthropic_sdk()
    assert observed["max_retries"] == 0
    assert observed["http_client"].trust_env is False
    assert observed["http_client"].follow_redirects is False


def test_returned_response_with_missing_usage_is_failed_at_hard_liability() -> None:
    _, prepared = _fixture()
    surface = prepared.surfaces[0]
    response = SimpleNamespace(
        id=None,
        model=None,
        stop_reason=None,
        content=[],
        usage=None,
    )
    client = AnthropicFablePairedClientV1(
        SimpleNamespace(messages=SimpleNamespace(create=lambda **_: response))
    )
    result = client.generate(surface)
    assert result.outcome == "failed"
    assert result.failure_code == "response_usage_invalid"
    assert result.cost_basis == "unknown_usage_hard_liability"
    assert result.charged_cost_usd_micros == surface.request_hard_liability_usd_micros
