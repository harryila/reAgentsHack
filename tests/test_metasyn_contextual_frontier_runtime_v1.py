from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

import literature_multiverse.metasyn_contextual_frontier_runtime_v1 as runtime
from literature_multiverse.metasyn_contextual_frontier_runtime_v1 import (
    MetaSynContextualFrontierPlanV1,
    MetaSynContextualFrontierProviderResultV1,
    MetaSynContextualFrontierRuntimeV1Error,
    MetaSynContextualFrontierUsageV1,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def plan() -> MetaSynContextualFrontierPlanV1:
    cached = os.environ.get("LM_FRONTIER_TEST_PLAN")
    if cached:
        return MetaSynContextualFrontierPlanV1.model_validate(
            json.loads(Path(cached).read_text(encoding="utf-8"))
        )
    return runtime.freeze_metasyn_contextual_frontier_plan_v1(repository_root=ROOT)


def _completed(
    request: runtime.MetaSynContextualFrontierRequestV1,
    parsed: dict[str, Any],
) -> MetaSynContextualFrontierProviderResultV1:
    text = json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return runtime.freeze_metasyn_contextual_frontier_provider_result_v1(
        request=request,
        outcome="completed",
        response_id="msg_fake_" + request.request_key,
        response_model="claude-fable-5",
        stop_reason="end_turn",
        text=text,
        parsed_json=parsed,
        usage=MetaSynContextualFrontierUsageV1(input_tokens=100, output_tokens=100),
    )


def _failed(
    request: runtime.MetaSynContextualFrontierRequestV1,
    outcome: runtime.ProviderOutcome,
) -> MetaSynContextualFrontierProviderResultV1:
    return runtime.freeze_metasyn_contextual_frontier_provider_result_v1(
        request=request,
        outcome=outcome,
        response_id="msg_fake_" + request.request_key,
        response_model=(
            "claude-not-the-frozen-model"
            if outcome == "response_model_mismatch"
            else "claude-fable-5"
        ),
        stop_reason="end_turn",
        text=None,
        parsed_json=None,
        usage=MetaSynContextualFrontierUsageV1(input_tokens=100, output_tokens=10),
        failure_code=outcome,
    )


class FakeClient:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def generate(
        self, request: runtime.MetaSynContextualFrontierRequestV1
    ) -> MetaSynContextualFrontierProviderResultV1:
        with self._lock:
            self.calls.append(request.request_key)
        value = self.values[request.request_key]
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, MetaSynContextualFrontierProviderResultV1):
            return value
        return _completed(request, value)


class InvalidReturnClient:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.calls: list[str] = []

    def generate(self, request: runtime.MetaSynContextualFrontierRequestV1) -> Any:
        self.calls.append(request.request_key)
        return self.value


def _prepare(tmp_path: Path, plan: MetaSynContextualFrontierPlanV1) -> Path:
    workspace = runtime._fresh_workspace(tmp_path / "frontier")
    runtime._persist_json(workspace / "00-prepared.json", plan)
    return workspace


def _authorize(workspace: Path, plan: MetaSynContextualFrontierPlanV1) -> None:
    runtime.authorize_metasyn_contextual_frontier_runtime_v1(
        workspace=workspace,
        phase_budget_usd_micros=plan.total_cost_ceiling_usd_micros,
    )


def _fixture_outputs(plan: MetaSynContextualFrontierPlanV1) -> dict[str, dict[str, Any]]:
    return {
        item.request.request_key: item.offline_witness.model_outcome.model_dump(mode="json")
        for item in plan.roster
    }


def _abstention(item: runtime.MetaSynContextualFrontierRosterItemV1) -> dict[str, Any]:
    return {
        "outcome_version": "contextual-packet-model-outcome-v3",
        "packet_status": "unable_to_complete",
        "candidate_binding_sha256": item.provider_binding_sha256,
        "reason": "other_grounding_failure",
    }


def test_fable_high_structured_wire_and_full_context_liability_are_exact(
    plan: MetaSynContextualFrontierPlanV1,
) -> None:
    assert plan.provider_identity.model == "claude-fable-5"
    assert plan.provider_identity.effort == "high"
    assert plan.provider_identity.transport_mode == "structured_json_schema"
    assert plan.total_cost_ceiling_usd_micros == 23_200_000
    assert plan.diagnostic_known_surface_cost_usd_micros_total < 23_200_000
    assert all(
        item.request.cost_ceiling.model_max_input_tokens == 1_000_000 for item in plan.roster
    )
    for item in plan.roster:
        call = item.request.wire_kwargs
        assert call["model"] == "claude-fable-5"
        assert call["output_config"]["effort"] == "high"
        assert call["output_config"]["format"] == {
            "type": "json_schema",
            "schema": item.request.compiled_schema.wire_schema,
        }
        assert call["service_tier"] == "standard_only"
        assert item.request.compiled_schema.wire_optional_parameter_count <= 24
        assert item.request.compiled_schema.wire_union_parameter_count <= 16


def test_binary_pair_coherence_is_present_in_frozen_primary_and_fallback(
    plan: MetaSynContextualFrontierPlanV1,
) -> None:
    primary, fallback = plan.roster
    primary_claims = {
        claim.field_path: claim for claim in primary.offline_witness.model_outcome.claims
    }
    fallback_claims = {
        claim.field_path: claim for claim in fallback.offline_witness.model_outcome.claims
    }
    assert primary_claims["effect.control_events"].context == "vs 1 of 96 ("
    assert primary_claims["effect.control_total"].context == "vs 1 of 96 ("
    assert fallback_claims["effect.control_events"].context == "and 6 of 85 ("
    assert fallback_claims["effect.control_total"].context == "and 6 of 85 ("


def test_budget_precheck_and_no_call_before_authorization(
    tmp_path: Path, plan: MetaSynContextualFrontierPlanV1
) -> None:
    workspace = _prepare(tmp_path, plan)
    client = FakeClient(_fixture_outputs(plan))
    with pytest.raises(MetaSynContextualFrontierRuntimeV1Error):
        runtime.authorize_metasyn_contextual_frontier_runtime_v1(
            workspace=workspace,
            phase_budget_usd_micros=plan.total_cost_ceiling_usd_micros - 1,
        )
    with pytest.raises(MetaSynContextualFrontierRuntimeV1Error):
        runtime.execute_metasyn_contextual_frontier_runtime_v1(workspace=workspace, client=client)
    assert client.calls == []


def test_first_fully_grounded_graph_stops_fallback_and_binds_runtime_provenance(
    tmp_path: Path, plan: MetaSynContextualFrontierPlanV1
) -> None:
    workspace = _prepare(tmp_path, plan)
    _authorize(workspace, plan)
    client = FakeClient(_fixture_outputs(plan))
    report = runtime.execute_metasyn_contextual_frontier_runtime_v1(
        workspace=workspace, client=client
    )
    assert report.status == "typed_graph_smoke_completed"
    assert client.calls == [plan.roster[0].request.request_key]
    assert report.unattempted_request_keys == [plan.roster[1].request.request_key]
    assert report.first_success_stopped_fallback
    validation = report.validation_results[0]
    projection = validation.native_projection
    assert projection is not None and projection.fragment is not None
    assert projection.runtime_pipeline_sha256 == plan.runtime_pipeline_sha256
    assert (
        projection.provider_execution_binding_sha256 == validation.provider_execution_binding_sha256
    )
    assert (
        projection.runtime_grounding_binding_sha256 == validation.runtime_grounding_binding_sha256
    )
    assert projection.fragment.pipeline_fingerprint_sha256 == plan.runtime_pipeline_sha256
    assert (
        projection.fragment.extraction_context_sha256 == validation.runtime_grounding_binding_sha256
    )
    assert not report.claim_release_authority


def test_scientific_abstention_uses_fallback_and_second_success_is_terminal(
    tmp_path: Path, plan: MetaSynContextualFrontierPlanV1
) -> None:
    workspace = _prepare(tmp_path, plan)
    _authorize(workspace, plan)
    values = _fixture_outputs(plan)
    values[plan.roster[0].request.request_key] = _abstention(plan.roster[0])
    client = FakeClient(values)
    report = runtime.execute_metasyn_contextual_frontier_runtime_v1(
        workspace=workspace, client=client
    )
    assert report.status == "typed_graph_smoke_completed"
    assert [item.status for item in report.validation_results] == [
        "scientific_abstention",
        "typed_graph_mechanics_completed",
    ]
    assert client.calls == [item.request.request_key for item in plan.roster]


@pytest.mark.parametrize(
    "first_failure",
    ["response_json_invalid", "response_schema_invalid", "response_model_mismatch"],
)
def test_malformed_schema_and_model_failures_remain_runtime_failures_then_fallback(
    tmp_path: Path,
    plan: MetaSynContextualFrontierPlanV1,
    first_failure: runtime.ProviderOutcome,
) -> None:
    workspace = _prepare(tmp_path, plan)
    _authorize(workspace, plan)
    values: dict[str, Any] = _fixture_outputs(plan)
    values[plan.roster[0].request.request_key] = _failed(plan.roster[0].request, first_failure)
    report = runtime.execute_metasyn_contextual_frontier_runtime_v1(
        workspace=workspace, client=FakeClient(values)
    )
    assert [item.status for item in report.validation_results] == [
        "provider_result_failed",
        "typed_graph_mechanics_completed",
    ]
    assert report.validation_results[0].failure_code == first_failure


def test_provider_exception_is_terminal_ambiguous_and_never_retried(
    tmp_path: Path, plan: MetaSynContextualFrontierPlanV1
) -> None:
    workspace = _prepare(tmp_path, plan)
    _authorize(workspace, plan)
    client = FakeClient(
        {plan.roster[0].request.request_key: RuntimeError("synthetic secret-free failure")}
    )
    first = runtime.execute_metasyn_contextual_frontier_runtime_v1(
        workspace=workspace, client=client
    )
    second = runtime.execute_metasyn_contextual_frontier_runtime_v1(
        workspace=workspace, client=client
    )
    assert first == second
    assert first.status == "terminal_ambiguous_attempt_poison"
    assert first.ambiguity_incident is not None
    assert not first.ambiguity_incident.retry_this_request_permitted
    assert client.calls == [plan.roster[0].request.request_key]


def test_orphan_intent_on_resume_is_poisoned_without_transport(
    tmp_path: Path, plan: MetaSynContextualFrontierPlanV1
) -> None:
    workspace = _prepare(tmp_path, plan)
    _authorize(workspace, plan)
    authorization = runtime._load_authorization(workspace=workspace, plan=plan)
    intent = runtime._freeze_intent(plan=plan, authorization=authorization, item=plan.roster[0])
    runtime._persist_json(workspace / "intents" / f"{intent.request_key}.json", intent)
    client = FakeClient(_fixture_outputs(plan))
    report = runtime.execute_metasyn_contextual_frontier_runtime_v1(
        workspace=workspace, client=client
    )
    assert report.status == "terminal_ambiguous_attempt_poison"
    assert report.ambiguity_incident is not None
    assert report.ambiguity_incident.incident_kind == "orphan_intent_observed_on_resume"
    assert client.calls == []


def test_invalid_secret_bearing_return_is_sanitized_and_never_archived(
    tmp_path: Path, plan: MetaSynContextualFrontierPlanV1
) -> None:
    workspace = _prepare(tmp_path, plan)
    _authorize(workspace, plan)
    synthetic_secret = "sk-ant-SYNTHETIC-DO-NOT-ARCHIVE"
    client = InvalidReturnClient({"authorization": synthetic_secret})
    report = runtime.execute_metasyn_contextual_frontier_runtime_v1(
        workspace=workspace, client=client
    )
    assert report.status == "terminal_ambiguous_attempt_poison"
    archived = "".join(
        path.read_text(encoding="utf-8") for path in workspace.rglob("*.json") if path.is_file()
    )
    assert synthetic_secret not in archived
    assert 'authorization":"sk-ant' not in archived


def test_archive_tampering_and_workspace_symlinks_fail_closed(
    tmp_path: Path, plan: MetaSynContextualFrontierPlanV1
) -> None:
    workspace = _prepare(tmp_path, plan)
    _authorize(workspace, plan)
    runtime.execute_metasyn_contextual_frontier_runtime_v1(
        workspace=workspace, client=FakeClient(_fixture_outputs(plan))
    )
    validated = runtime.validate_metasyn_contextual_frontier_runtime_v1(
        repository_root=ROOT, workspace=workspace, external_replay=False
    )
    assert validated.status == "terminal"
    receipt_path = next((workspace / "provider-receipts").glob("*.json"))
    raw = json.loads(receipt_path.read_text(encoding="utf-8"))
    raw["provider_result_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((MetaSynContextualFrontierRuntimeV1Error, ValueError)):
        runtime.validate_metasyn_contextual_frontier_runtime_v1(
            repository_root=ROOT, workspace=workspace, external_replay=False
        )

    target = tmp_path / "real-workspace"
    target.mkdir()
    symlink = tmp_path / "workspace-link"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(MetaSynContextualFrontierRuntimeV1Error):
        runtime.load_metasyn_contextual_frontier_plan_v1(workspace=symlink)


def test_concurrent_executors_share_one_exact_attempt(
    tmp_path: Path, plan: MetaSynContextualFrontierPlanV1
) -> None:
    workspace = _prepare(tmp_path, plan)
    _authorize(workspace, plan)
    client = FakeClient(_fixture_outputs(plan))
    reports: list[runtime.MetaSynContextualFrontierTerminalReportV1] = []

    def execute() -> None:
        reports.append(
            runtime.execute_metasyn_contextual_frontier_runtime_v1(
                workspace=workspace, client=client
            )
        )

    threads = [threading.Thread(target=execute) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert len(reports) == 2
    assert reports[0] == reports[1]
    assert client.calls == [plan.roster[0].request.request_key]
