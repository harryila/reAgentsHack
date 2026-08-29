from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from threading import Lock
from typing import Any

import pytest

import literature_multiverse.metasyn_passage_hosted_runtime_v2 as runtime
from literature_multiverse.anthropic_bounded_generation import (
    ANTHROPIC_INPUT_RATE_USD_PER_MTOK,
    ANTHROPIC_OUTPUT_RATE_USD_PER_MTOK,
    AnthropicBoundedRequestV1,
    AnthropicBoundedResultV1,
    AnthropicCostV1,
    AnthropicFailureV1,
    AnthropicUsageV1,
)
from literature_multiverse.hosted_exact_once import (
    HostedExactOnceCostAuthorizationV1,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_passage_hosted_bundle_v2 import (
    MetaSynPassageHostedExecutionBundleV2,
    freeze_metasyn_passage_hosted_execution_bundle_v2,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle() -> MetaSynPassageHostedExecutionBundleV2:
    return freeze_metasyn_passage_hosted_execution_bundle_v2(repository_root=ROOT)


def _patch_bundle_replay(
    monkeypatch: pytest.MonkeyPatch,
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    def freeze(**_: Any) -> MetaSynPassageHostedExecutionBundleV2:
        return bundle

    def validate(*, execution_bundle: Any, **_: Any) -> MetaSynPassageHostedExecutionBundleV2:
        canonical = MetaSynPassageHostedExecutionBundleV2.model_validate(
            execution_bundle.model_dump(mode="json")
            if isinstance(execution_bundle, MetaSynPassageHostedExecutionBundleV2)
            else execution_bundle
        )
        assert canonical == bundle
        return canonical

    monkeypatch.setattr(runtime, "freeze_metasyn_passage_hosted_execution_bundle_v2", freeze)
    monkeypatch.setattr(runtime, "validate_metasyn_passage_hosted_execution_bundle_v2", validate)


def _completed_result(
    request: AnthropicBoundedRequestV1, parsed: dict[str, Any]
) -> AnthropicBoundedResultV1:
    text = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    usage = AnthropicUsageV1(input_tokens=11, output_tokens=7)
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
        "response_id": "msg_fake_" + request.request_key,
        "response_model": request.model,
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


def _max_tokens_result(request: AnthropicBoundedRequestV1) -> AnthropicBoundedResultV1:
    usage = AnthropicUsageV1(input_tokens=11, output_tokens=request.max_output_tokens)
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
    failure = AnthropicFailureV1(
        code="response_stop_reason_invalid",
        detail="response_did_not_end_with_end_turn",
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
        "outcome": "response_stop_reason_invalid",
        "response_id": "msg_fake_max_tokens_" + request.request_key,
        "response_model": request.model,
        "stop_reason": "max_tokens",
        "text": None,
        "response_text_sha256": None,
        "parsed_json": None,
        "parsed_json_sha256": None,
        "usage": usage,
        "cost": cost,
        "failure": failure,
    }
    return AnthropicBoundedResultV1.model_validate(
        {**payload, "result_sha256": hash_canonical(payload)}
    )


class FakeClient:
    def __init__(self, bundle: MetaSynPassageHostedExecutionBundleV2) -> None:
        self.bundle = bundle
        self.responses = {
            item.request.request_key: item.expected_fixture
            for item in bundle.source_free_preflight_plan
        }
        self.calls: list[str] = []
        self.max_tokens_keys: set[str] = set()
        self.required_authorization_paths: dict[str, Path] = {}
        self._lock = Lock()

    def generate(self, request: AnthropicBoundedRequestV1) -> AnthropicBoundedResultV1:
        required = self.required_authorization_paths.get(request.request_key)
        if required is not None:
            assert required.is_file()
        with self._lock:
            self.calls.append(request.request_key)
        if request.request_key in self.max_tokens_keys:
            return _max_tokens_result(request)
        parsed = self.responses.get(request.request_key)
        if parsed is None and request.schema_kind == "inventory":
            parsed = {
                "inventory_version": "metasyn-passage-candidate-inventory-v2",
                "inventory_status": "no_candidate_found",
                "candidates": [],
                "has_more_or_uncertain": False,
            }
        if parsed is None:
            raise AssertionError(f"missing fake response:{request.request_key}")
        return _completed_result(request, parsed)


def _prepare(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> tuple[Path, FakeClient]:
    _patch_bundle_replay(monkeypatch, bundle)
    workspace = tmp_path / "fresh-runtime"
    prepared = runtime.prepare_metasyn_passage_hosted_runtime_v2(
        repository_root=ROOT, workspace=workspace
    )
    assert prepared == bundle
    return workspace, FakeClient(bundle)


def _authorize_and_preflight(
    *,
    workspace: Path,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    client: FakeClient,
) -> None:
    runtime.authorize_metasyn_passage_hosted_runtime_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )
    runtime.run_metasyn_passage_source_free_preflight_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
        client=client,
    )


def test_zero_packet_roster_reaches_externally_validated_yield_only_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    workspace, client = _prepare(tmp_path=tmp_path, monkeypatch=monkeypatch, bundle=bundle)
    runtime.authorize_metasyn_passage_hosted_runtime_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )
    assert client.calls == []
    preflight_auth = HostedExactOnceCostAuthorizationV1.model_validate(
        json.loads(
            (
                workspace / "provider-state/cost-authorizations/source_free_preflight.json"
            ).read_text()
        )
    )
    assert preflight_auth.authorized_call_count == 8

    runtime.run_metasyn_passage_source_free_preflight_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
        client=client,
    )
    assert len(client.calls) == 8
    runtime.run_metasyn_passage_source_free_preflight_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
        client=client,
    )
    assert len(client.calls) == 8

    inventory_auth_path = workspace / "provider-state/cost-authorizations/inventory.json"
    client.required_authorization_paths["inventory-row-00"] = inventory_auth_path
    runtime.run_metasyn_passage_inventory_smoke_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
        client=client,
    )
    inventory_auth = HostedExactOnceCostAuthorizationV1.model_validate(
        json.loads(inventory_auth_path.read_text())
    )
    assert inventory_auth.authorized_call_count == 32
    assert inventory_auth.source_bearing_call_count == 32
    inventory = runtime.run_metasyn_passage_inventory_roster_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
        client=client,
    )
    assert len(client.calls) == 40
    assert inventory.validation_status_counts == {"inventory_contract_valid": 32}

    roster = runtime.freeze_metasyn_passage_packet_roster_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )
    assert roster.request_count == 0
    assert not (workspace / "provider-state/cost-authorizations/packet.json").exists()
    smoke = runtime.run_metasyn_passage_packet_smoke_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
        client=client,
    )
    assert smoke.status == "not_applicable"
    packet_ledger = runtime.run_metasyn_passage_packet_roster_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
        client=client,
    )
    assert packet_ledger.results == []
    report = runtime.finalize_metasyn_passage_hosted_runtime_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )
    assert report.total_provider_attempts_or_possible_attempts == 40
    assert report.typed_effect_count == 0
    assert report.extraction_accuracy_authority is False
    assert report.synthesis_input_authority is False
    validation = runtime.validate_finalized_metasyn_passage_hosted_runtime_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )
    assert validation.exact_terminal_outcome_count == 40
    status = runtime.metasyn_passage_hosted_runtime_status_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )
    assert status["current_stage"] == "externally_validated"


def test_concurrent_preflight_is_serialized_and_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    workspace, client = _prepare(tmp_path=tmp_path, monkeypatch=monkeypatch, bundle=bundle)
    runtime.authorize_metasyn_passage_hosted_runtime_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )

    def run() -> str:
        return runtime.run_metasyn_passage_source_free_preflight_v2(
            repository_root=ROOT,
            workspace=workspace,
            expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
            client=client,
        ).receipt_sha256

    with ThreadPoolExecutor(max_workers=2) as executor:
        values = list(executor.map(lambda _: run(), range(2)))
    assert values[0] == values[1]
    assert len(client.calls) == 8


def test_tampered_stage_bound_artifact_fails_before_any_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    workspace, client = _prepare(tmp_path=tmp_path, monkeypatch=monkeypatch, bundle=bundle)
    runtime.authorize_metasyn_passage_hosted_runtime_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )
    path = workspace / "run-cost-authorization.json"
    value = json.loads(path.read_text())
    value["configured_cost_limit_usd_micros"] += 1
    value["authorization_sha256"] = hash_canonical(
        {key: item for key, item in value.items() if key != "authorization_sha256"}
    )
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        runtime.MetaSynPassageHostedRuntimeV2Error,
        match="stage_artifact_tamper",
    ):
        runtime.run_metasyn_passage_source_free_preflight_v2(
            repository_root=ROOT,
            workspace=workspace,
            expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
            client=client,
        )
    assert client.calls == []


def test_max_tokens_is_runtime_capacity_failure_not_scientific_abstention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    workspace, client = _prepare(tmp_path=tmp_path, monkeypatch=monkeypatch, bundle=bundle)
    _authorize_and_preflight(workspace=workspace, bundle=bundle, client=client)
    client.max_tokens_keys.add("inventory-row-00")
    with pytest.raises(
        runtime.MetaSynPassageHostedRuntimeV2Error,
        match="inventory_smoke_failed:runtime_capacity_failure",
    ):
        runtime.run_metasyn_passage_inventory_smoke_v2(
            repository_root=ROOT,
            workspace=workspace,
            expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
            client=client,
        )
    result = runtime.InventoryCallResultV2.model_validate(
        json.loads((workspace / "inventory-results/row-00.json").read_text())
    )
    assert result.validation_status == "runtime_capacity_failure"
    assert result.inventory_receipt is None
    assert result.authorizes_packet_generation is False
    assert result.runtime_failure_is_not_scientific_abstention is True
    status = runtime.metasyn_passage_hosted_runtime_status_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )
    assert status["current_stage"] == "preflight_passed"


def test_packet_abstention_is_valid_but_cannot_open_spend_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    workspace, client = _prepare(tmp_path=tmp_path, monkeypatch=monkeypatch, bundle=bundle)
    _authorize_and_preflight(workspace=workspace, bundle=bundle, client=client)
    row = bundle.extraction_inputs.rows[21]
    passage = row.projection_surface.passages[0]
    outcome_id = row.question_surface.allowed_outcome_ids[0]
    outcome_text = row.question_surface.allowed_outcome_text_by_id[outcome_id]
    client.responses["inventory-row-21"] = {
        "inventory_version": "metasyn-passage-candidate-inventory-v2",
        "inventory_status": "candidates_found",
        "candidates": [
            {
                "candidate_index": 1,
                "canonical_outcome_id": outcome_id,
                "outcome_concept_quote": outcome_text[: min(40, len(outcome_text))],
                "effect_kind": "direct_standard_error",
                "passage_ids": [passage.passage_id],
            }
        ],
        "has_more_or_uncertain": False,
    }
    runtime.run_metasyn_passage_inventory_smoke_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
        client=client,
    )
    runtime.run_metasyn_passage_inventory_roster_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
        client=client,
    )
    roster = runtime.freeze_metasyn_passage_packet_roster_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )
    assert roster.request_count == 1
    packet = roster.requests[0]
    client.responses[packet.request.request_key] = (
        packet.packet_input.packet_schema_bundle.abstaining_fixture
    )
    with pytest.raises(
        runtime.MetaSynPassageHostedRuntimeV2Error,
        match="packet_smoke_failed_no_typed_effect",
    ):
        runtime.run_metasyn_passage_packet_smoke_v2(
            repository_root=ROOT,
            workspace=workspace,
            expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
            client=client,
        )
    result = runtime.PacketCallResultV2.model_validate(
        json.loads(
            (workspace / "packet-results" / f"{packet.request.request_key}.json").read_text()
        )
    )
    assert result.validation_status == "grounding_abstained"
    assert result.authorizes_typed_effect is False
    smoke = runtime.PacketSmokeReceiptV2.model_validate(
        json.loads((workspace / "packet-smoke-attempt.json").read_text())
    )
    assert smoke.status == "failed_gate"
    assert smoke.remaining_packet_calls_permitted is False
    status = runtime.metasyn_passage_hosted_runtime_status_v2(
        repository_root=ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )
    assert status["current_stage"] == "packet_roster_frozen"
