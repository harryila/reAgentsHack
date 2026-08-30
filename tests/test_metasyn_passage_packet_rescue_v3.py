from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.private_cache_support import (
    HOSTED_ADAPTER_STALE_CODES,
    TYPED_PILOT_STALE_CODES,
    require_private_cache,
    skip_when_historical_replay_is_stale,
)

import literature_multiverse.metasyn_passage_packet_rescue_v3 as rescue
from literature_multiverse.anthropic_bounded_generation import (
    ANTHROPIC_INPUT_RATE_USD_PER_MTOK,
    ANTHROPIC_OUTPUT_RATE_USD_PER_MTOK,
    AnthropicBoundedRequestV1,
    AnthropicBoundedResultV1,
    AnthropicCostV1,
    AnthropicUsageV1,
)
from literature_multiverse.hosted_exact_once import execute_hosted_exactly_once
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_bounded_hosted_runtime import MetaSynHostedRuntimeError
from literature_multiverse.metasyn_passage_hosted_bundle_v2 import (
    MetaSynPassageHostedExecutionBundleV2,
)
from literature_multiverse.metasyn_passage_packet_rescue_v3 import (
    MetaSynPassagePacketRescuePlanV3,
    MetaSynPassagePacketRescueV3Error,
    MetaSynPassageRescuePreCallBlockerItemV3,
    MetaSynPassageRescueResultV3,
    MetaSynPassageRescueSmokeReceiptV3,
)
from literature_multiverse.metasyn_typed_pilot import MetaSynTypedPilotError

ROOT = Path(__file__).resolve().parents[1]
V2_WORKSPACE = ROOT / "data/cache/metasyn/passage-hosted-yield-v2"


@pytest.fixture(scope="module")
def plan() -> MetaSynPassagePacketRescuePlanV3:
    # The real replay reaches, transitively, the private bounded-anthropic-yield-v5
    # bundle and (through its adapter/typed-pilot rebuild) typed-oracle-pilot-v2;
    # all three must be present or this fixture skips rather than erroring.
    require_private_cache(
        "data/cache/metasyn/passage-hosted-yield-v2",
        "data/cache/metasyn/bounded-anthropic-yield-v5",
        "data/cache/metasyn/typed-oracle-pilot-v2",
    )
    return skip_when_historical_replay_is_stale(
        lambda: rescue.freeze_metasyn_passage_packet_rescue_plan_v3(
            repository_root=ROOT,
            v2_workspace=V2_WORKSPACE,
        ),
        stale_errors=(MetaSynTypedPilotError, MetaSynHostedRuntimeError),
        stale_codes=TYPED_PILOT_STALE_CODES | HOSTED_ADAPTER_STALE_CODES,
    )


@pytest.fixture(scope="module")
def bundle() -> MetaSynPassageHostedExecutionBundleV2:
    require_private_cache("data/cache/metasyn/passage-hosted-yield-v2")
    return MetaSynPassageHostedExecutionBundleV2.model_validate(
        json.loads((V2_WORKSPACE / "execution-bundle.json").read_text(encoding="utf-8"))
    )


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


class FakeClient:
    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def generate(self, request: AnthropicBoundedRequestV1) -> AnthropicBoundedResultV1:
        self.calls.append(request.request_key)
        return _completed_result(request, self.responses[request.request_key])


class RaisingClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, request: AnthropicBoundedRequestV1) -> AnthropicBoundedResultV1:
        self.calls.append(request.request_key)
        raise RuntimeError("synthetic transport boundary failure")


def _compact_abstention(binding: str) -> dict[str, Any]:
    return {
        "candidate_binding_sha256": binding,
        "reason": "source_support_incomplete",
    }


def _patch_fast_replay(
    monkeypatch: pytest.MonkeyPatch,
    *,
    plan: MetaSynPassagePacketRescuePlanV3,
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    monkeypatch.setattr(
        rescue,
        "freeze_metasyn_passage_packet_rescue_plan_v3",
        lambda **_: plan,
    )
    monkeypatch.setattr(rescue, "_load_plan", lambda **_: plan)
    monkeypatch.setattr(
        rescue,
        "_replay_v2_base",
        lambda **_: SimpleNamespace(bundle=bundle),
    )


def _prepare(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan: MetaSynPassagePacketRescuePlanV3,
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> Path:
    _patch_fast_replay(monkeypatch, plan=plan, bundle=bundle)
    workspace = tmp_path / "rescue"
    assert (
        rescue.prepare_metasyn_passage_packet_rescue_v3(
            repository_root=ROOT,
            workspace=workspace,
            v2_workspace=V2_WORKSPACE,
        )
        == plan
    )
    return workspace


def _local_contract_result(
    *,
    tmp_path: Path,
    plan: MetaSynPassagePacketRescuePlanV3,
    bundle: MetaSynPassageHostedExecutionBundleV2,
    request_index: int,
    client: FakeClient | RaisingClient,
) -> MetaSynPassageRescueResultV3:
    request = plan.requests[request_index]
    intent = rescue._rescue_intents(plan)[request_index]
    authorization = rescue._freeze_authorization(plan)
    outcome = execute_hosted_exactly_once(
        workspace=tmp_path / f"provider-contract-{request_index}",
        intent=intent,
        authorization=authorization.exact_authorization,
        client=client,
    )
    return rescue._process_rescue_outcome(
        plan=plan,
        bundle=bundle,
        rescue_request=request,
        outcome=outcome,
    )


@pytest.mark.private_cache
def test_real_v2_replay_forensics_selection_and_zero_yield_blocker_are_exact(
    plan: MetaSynPassagePacketRescuePlanV3,
) -> None:
    snapshot = plan.v2_replay_snapshot
    forensic = plan.v2_forensic_receipt
    blocker = plan.pre_call_blocker

    assert snapshot.execution_bundle_sha256 == rescue.EXPECTED_V2_EXECUTION_BUNDLE_SHA256
    assert snapshot.inventory_ledger_sha256 == rescue.EXPECTED_V2_INVENTORY_LEDGER_SHA256
    assert snapshot.packet_roster_sha256 == rescue.EXPECTED_V2_PACKET_ROSTER_SHA256
    assert snapshot.failed_smoke_sha256 == rescue.EXPECTED_V2_FAILED_SMOKE_SHA256
    assert snapshot.provider_receipt_count == 43
    assert len(snapshot.attempted_packet_requests) == 3
    assert forensic.raw_candidate_binding_match_count == 3
    assert forensic.valid_grounding_abstention_count == 3
    assert forensic.completed_typed_effect_count == 0
    assert all(item.normalization_receipt.normalization_idempotent for item in forensic.items)
    assert all(item.raw_candidate_binding_already_matched for item in forensic.items)
    assert all(
        item.normalization_receipt.absent_invariant_fields == ["outcome_version"]
        for item in forensic.items
    )
    assert forensic.original_v2_smoke_status == "failed_gate"
    assert forensic.v2_failed_gate_semantics_changed is False

    failure_by_row = {item.row_ordinal: item.failure_class for item in snapshot.inventory_failures}
    assert failure_by_row[9] == "passage_ids_not_sorted_unique"
    assert failure_by_row[13] == "outcome_concept_not_exact_protocol_quote"
    assert all(
        value == "candidates_not_canonical"
        for row, value in failure_by_row.items()
        if row not in {9, 13}
    )
    assert snapshot.future_representational_recovery_row_count == 9
    assert snapshot.future_representational_recovery_candidate_count == 42
    assert snapshot.scientifically_invalid_inventory_row_count == 1
    assert snapshot.inventory_normalization_performed is False

    eligible = sorted(
        (item for item in plan.candidate_audits if item.eligible_rank is not None),
        key=lambda item: item.eligible_rank or 0,
    )
    assert [(item.row_ordinal, item.candidate_index) for item in eligible] == [
        (16, 1),
        (20, 1),
        (20, 2),
        (20, 4),
        (20, 3),
    ]
    assert [(item.row_ordinal, item.candidate_index) for item in plan.requests] == [
        (16, 1),
        (20, 1),
        (20, 2),
    ]
    assert plan.conservative_cost_ceiling_usd_micros == 2_172_408
    assert plan.provider_calls_permitted is False
    assert plan.authorization_created is False

    assert blocker.selected_candidate_count == 3
    assert blocker.selected_candidate_v2_reachable_count == 0
    assert blocker.numeric_boundary_blocked_candidate_count == 3
    assert blocker.unsupported_exact_effect_format_candidate_count == 2
    assert blocker.provider_calls_made == 0
    assert blocker.authorization_created is False
    assert blocker.calls_permitted is False
    assert blocker.provider_cost_liability_usd_micros == 0
    assert [item.immutable_v2_first_failure_code for item in blocker.items] == [
        "packet_grounding_v2_numeric_token_absent:effect.ci_upper",
        "packet_grounding_v2_effect_format_alias_unsupported",
        "packet_grounding_v2_effect_format_alias_unsupported",
    ]
    assert blocker.items[0].fails_only_at_immutable_numeric_boundary_gate is True
    assert all(item.upper_token_v2_valid_occurrence_count == 0 for item in blocker.items)
    assert all(item.exact_range_separator_code_point == "U+2013" for item in blocker.items)
    assert plan.reference_fields_unopened is True
    assert plan.official_test_labels_opened is False
    assert plan.extraction_accuracy_authority is False
    assert plan.synthesis_input_authority is False
    assert plan.claim_release_authority is False


@pytest.mark.private_cache
def test_selector_rejects_attempted_scientific_signature_even_under_fresh_key(
    plan: MetaSynPassagePacketRescuePlanV3,
) -> None:
    selected = plan.requests[0]
    source = selected.source_packet_request
    features = rescue._selection_features(source)
    signature = rescue._scientific_request_signature(source)
    assert signature == selected.scientific_request_signature_sha256
    assert selected.request.request_key != source.request.request_key
    assert (
        rescue._selection_disposition(
            request=source,
            features=features,
            attempted_packet_inputs=set(),
            attempted_bindings=set(),
            attempted_signatures={signature},
        )
        == "excluded_previously_attempted"
    )


@pytest.mark.private_cache
def test_coherent_plan_and_blocker_tamper_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    plan: MetaSynPassagePacketRescuePlanV3,
) -> None:
    plan_payload = plan.model_dump(mode="json")
    plan_payload["config_file_sha256"] = "b" * 64
    plan_payload["plan_sha256"] = hash_canonical(
        {key: value for key, value in plan_payload.items() if key != "plan_sha256"}
    )
    tampered_plan = MetaSynPassagePacketRescuePlanV3.model_validate(plan_payload)
    monkeypatch.setattr(
        rescue,
        "freeze_metasyn_passage_packet_rescue_plan_v3",
        lambda **_: plan,
    )
    with pytest.raises(MetaSynPassagePacketRescueV3Error, match="external_replay_mismatch"):
        rescue.validate_metasyn_passage_packet_rescue_plan_v3(
            plan=tampered_plan,
            repository_root=ROOT,
            v2_workspace=V2_WORKSPACE,
            external_replay=True,
        )

    item_payload = plan.pre_call_blocker.items[0].model_dump(mode="json")
    item_payload["immutable_v2_first_failure_code"] = "forged_gate"
    item_payload["blocker_item_sha256"] = hash_canonical(
        {key: value for key, value in item_payload.items() if key != "blocker_item_sha256"}
    )
    with pytest.raises(ValueError, match="blocker_item_alias_mismatch"):
        MetaSynPassageRescuePreCallBlockerItemV3.model_validate(item_payload)


@pytest.mark.private_cache
def test_prepare_persists_blocker_and_authorize_and_smoke_make_zero_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan: MetaSynPassagePacketRescuePlanV3,
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    workspace = _prepare(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        plan=plan,
        bundle=bundle,
    )
    saved_blocker = json.loads(
        (workspace / "pre-call-zero-yield-blocker.json").read_text(encoding="utf-8")
    )
    assert saved_blocker == plan.pre_call_blocker.model_dump(mode="json")
    client = FakeClient({})
    with pytest.raises(MetaSynPassagePacketRescueV3Error, match="pre_call_zero_yield_blocker"):
        rescue.authorize_metasyn_passage_packet_rescue_v3(
            repository_root=ROOT,
            workspace=workspace,
            v2_workspace=V2_WORKSPACE,
            expected_plan_sha256=plan.plan_sha256,
        )
    with pytest.raises(MetaSynPassagePacketRescueV3Error, match="pre_call_zero_yield_blocker"):
        rescue.run_metasyn_passage_packet_rescue_smoke_v3(
            repository_root=ROOT,
            workspace=workspace,
            v2_workspace=V2_WORKSPACE,
            expected_plan_sha256=plan.plan_sha256,
            client=client,
        )
    assert client.calls == []
    assert not (workspace / "rescue-authorization.json").exists()
    assert list((workspace / "provider-state").rglob("*.json")) == []
    status = rescue.metasyn_passage_packet_rescue_status_v3(
        repository_root=ROOT,
        workspace=workspace,
        v2_workspace=V2_WORKSPACE,
        expected_plan_sha256=plan.plan_sha256,
    )
    assert status["current_stage"] == "prepared"
    assert status["provider_calls_permitted"] is False
    assert status["authorization_created"] is False
    assert status["provider_cost_liability_usd_micros"] == 0


@pytest.mark.private_cache
def test_result_and_smoke_intrinsic_shapes_reject_coherent_authority_forgery(
    tmp_path: Path,
    plan: MetaSynPassagePacketRescuePlanV3,
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    request = plan.requests[0]
    client = FakeClient(
        {request.request.request_key: _compact_abstention(request.candidate_binding_sha256)}
    )
    result = _local_contract_result(
        tmp_path=tmp_path,
        plan=plan,
        bundle=bundle,
        request_index=0,
        client=client,
    )
    assert result.validation_status == "grounding_abstained"
    assert result.authorizes_typed_effect is False
    forged = result.model_dump(mode="json")
    forged["validation_status"] = "typed_effect_completed"
    forged["authorizes_typed_effect"] = True
    forged["result_sha256"] = hash_canonical(
        {key: value for key, value in forged.items() if key != "result_sha256"}
    )
    with pytest.raises(ValueError, match="validation_status_shape_mismatch"):
        MetaSynPassageRescueResultV3.model_validate(forged)

    authorization = rescue._freeze_authorization(plan)
    smoke = rescue._freeze_smoke_receipt(
        plan=plan,
        authorization=authorization,
        results=[result],
    )
    mismatched = result.model_copy(update={"plan_sha256": "b" * 64})
    invalid_smoke = smoke.model_copy(update={"results": [mismatched]})
    with pytest.raises(ValueError, match="smoke_result_plan_mismatch"):
        invalid_smoke.validate_smoke()


@pytest.mark.private_cache
def test_incident_followed_by_a_later_result_is_rejected_intrinsically(
    tmp_path: Path,
    plan: MetaSynPassagePacketRescuePlanV3,
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    incident_client = RaisingClient()
    incident = _local_contract_result(
        tmp_path=tmp_path / "incident",
        plan=plan,
        bundle=bundle,
        request_index=0,
        client=incident_client,
    )
    second_request = plan.requests[1]
    second_client = FakeClient(
        {
            second_request.request.request_key: _compact_abstention(
                second_request.candidate_binding_sha256
            )
        }
    )
    later = _local_contract_result(
        tmp_path=tmp_path / "later",
        plan=plan,
        bundle=bundle,
        request_index=1,
        client=second_client,
    )
    authorization = rescue._freeze_authorization(plan)
    payload: dict[str, Any] = {
        "smoke_version": rescue.RESCUE_SMOKE_VERSION,
        "status": "failed_gate",
        "plan_sha256": plan.plan_sha256,
        "authorization_receipt_sha256": authorization.authorization_receipt_sha256,
        "ordered_authorized_request_keys": [item.request.request_key for item in plan.requests],
        "attempted_request_keys": [incident.request_key, later.request_key],
        "results": [incident, later],
        "result_membership_sha256": hash_canonical([incident.result_sha256, later.result_sha256]),
        "completed_typed_effect_result_sha256": None,
        "typed_effect_count": 0,
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
    with pytest.raises(ValueError, match="smoke_incident_not_terminal"):
        MetaSynPassageRescueSmokeReceiptV3.model_validate(
            {**payload, "smoke_sha256": hash_canonical(payload)}
        )
    assert incident_client.calls == [plan.requests[0].request.request_key]
    assert second_client.calls == [plan.requests[1].request.request_key]


@pytest.mark.private_cache
def test_prepared_blocker_artifact_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan: MetaSynPassagePacketRescuePlanV3,
    bundle: MetaSynPassageHostedExecutionBundleV2,
) -> None:
    workspace = _prepare(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        plan=plan,
        bundle=bundle,
    )
    blocker_path = workspace / "pre-call-zero-yield-blocker.json"
    blocker_path.write_text(
        blocker_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(MetaSynPassagePacketRescueV3Error, match="stage_artifact_tamper"):
        rescue.metasyn_passage_packet_rescue_status_v3(
            repository_root=ROOT,
            workspace=workspace,
            v2_workspace=V2_WORKSPACE,
            expected_plan_sha256=plan.plan_sha256,
        )


def test_symlink_and_nonfresh_workspace_are_rejected(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(MetaSynPassagePacketRescueV3Error, match="workspace_not_fresh"):
        rescue._create_fresh_workspace(existing)

    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(MetaSynPassagePacketRescueV3Error, match="workspace_not_fresh"):
        rescue._create_fresh_workspace(symlink)
