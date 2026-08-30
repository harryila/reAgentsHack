from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from argparse import Namespace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import scripts.run_metasyn_bounded_hosted_runtime as hosted_cli
from jsonschema import Draft202012Validator
from tests.private_cache_support import (
    HOSTED_ADAPTER_STALE_CODES,
    TYPED_PILOT_STALE_CODES,
    require_private_cache,
    skip_when_historical_replay_is_stale,
)

import literature_multiverse.metasyn_bounded_hosted_runtime as runtime
from literature_multiverse.anthropic_bounded_generation import (
    ANTHROPIC_INPUT_RATE_USD_PER_MTOK,
    ANTHROPIC_OUTPUT_RATE_USD_PER_MTOK,
    AnthropicBoundedRequestV1,
    AnthropicBoundedResultV1,
    AnthropicCostV1,
    AnthropicUsageV1,
)
from literature_multiverse.lineage import (
    OutputExistsError,
    atomic_write_json,
    canonical_json_bytes,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.metasyn_bounded_adapter import (
    MetaSynBoundedAdapterBundleV1,
    freeze_metasyn_inventory_validation_receipt,
)
from literature_multiverse.metasyn_bounded_hosted_runtime import (
    MAX_THEORETICAL_PROVIDER_CALLS,
    MetaSynHostedCallReceiptV1,
    MetaSynHostedCostAuthorizationGroupV1,
    MetaSynHostedCostAuthorizationReceiptV1,
    MetaSynHostedExecutionBundleV1,
    MetaSynHostedRuntimeError,
    _assert_authorization_group_ceilings,
    _assert_new_call_within_budget,
    _freeze_provider_request,
    _preflight_fixture,
    _preflight_prompt,
    _preflight_schema_bundle,
    _request_surface,
    finalize_metasyn_hosted_runtime,
    freeze_metasyn_hosted_attempt_intent,
    freeze_metasyn_hosted_cost_authorization,
    freeze_metasyn_hosted_execution_bundle,
    load_metasyn_hosted_runtime_config,
    metasyn_hosted_runtime_paths,
    run_metasyn_hosted_full_roster,
    run_metasyn_hosted_preflight,
    run_metasyn_hosted_smoke,
    validate_finalized_metasyn_hosted_runtime,
    write_metasyn_hosted_execution_bundle,
)
from literature_multiverse.metasyn_typed_pilot import MetaSynTypedPilotError
from literature_multiverse.native_bounded_schema_v2 import (
    synthetic_schema_v2_preflight_specs,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Every test in this module reaches the private local cache transitively through
# `hosted_bundle` (directly, or via `runtime_workspace`), so the whole module is
# marked rather than repeating the marker on each test.
pytestmark = pytest.mark.private_cache


@pytest.fixture(scope="session")
def hosted_bundle() -> MetaSynHostedExecutionBundleV1:
    root = require_private_cache(
        "data/cache/metasyn/bounded-qwen-yield-v2-attempt-06/execution-bundle.private.json",
        "data/cache/metasyn/typed-oracle-pilot-v2",
    )
    frozen_local_runtime = json.loads(
        (
            root / "data/cache/metasyn/bounded-qwen-yield-v2-attempt-06/"
            "execution-bundle.private.json"
        ).read_text(encoding="utf-8")
    )
    adapter_bundle = MetaSynBoundedAdapterBundleV1.model_validate(
        frozen_local_runtime["adapter_bundle"]
    )
    config, config_file_sha = load_metasyn_hosted_runtime_config(repository_root=root)
    # E16 (restoring this frozen v2 attempt against the current typed-oracle pilot
    # identity) is declined by the operator: `_pilot_downstream_sha` is never
    # patched here. The historical adapter is expected to be stale against the
    # current pilot pipeline identity; that expectation is pinned explicitly by
    # `test_historical_qwen_attempt06_adapter_is_stale_only_in_upstream_pilot_identity`
    # below. Here, the documented stale code is converted into a skip so the rest
    # of this module's tests (which only care about hosted-runtime mechanics, not
    # pilot-identity staleness) can still exercise a frozen bundle when available.
    return skip_when_historical_replay_is_stale(
        lambda: freeze_metasyn_hosted_execution_bundle(
            adapter_bundle=adapter_bundle,
            runtime_config=config,
            config_file_sha256=config_file_sha,
            pilot_workspace_relative="data/cache/metasyn/typed-oracle-pilot-v2",
            repository_root=root,
        ),
        stale_errors=(MetaSynTypedPilotError, MetaSynHostedRuntimeError),
        stale_codes=TYPED_PILOT_STALE_CODES | HOSTED_ADAPTER_STALE_CODES,
    )


def _authorization(
    bundle: MetaSynHostedExecutionBundleV1,
) -> MetaSynHostedCostAuthorizationReceiptV1:
    groups = [
        MetaSynHostedCostAuthorizationGroupV1(
            group="source_free_preflight",
            call_count=8,
            structured_json_schema_call_count=3,
            prompt_json_schema_call_count=5,
            conservative_input_token_ceiling=500_000,
            maximum_output_token_ceiling=32_768,
            cost_ceiling_usd_micros=1_000_000,
            request_roster_sha256=hash_canonical("preflight"),
            transport_mode_roster_sha256=hash_canonical(
                ["structured_json_schema"] * 3 + ["prompt_json_schema"] * 5
            ),
        ),
        MetaSynHostedCostAuthorizationGroupV1(
            group="exact_inventory",
            call_count=32,
            structured_json_schema_call_count=32,
            prompt_json_schema_call_count=0,
            conservative_input_token_ceiling=2_000_000,
            maximum_output_token_ceiling=24_576,
            cost_ceiling_usd_micros=3_000_000,
            request_roster_sha256=hash_canonical("inventory"),
            transport_mode_roster_sha256=hash_canonical(
                ["structured_json_schema"] * 32
            ),
        ),
        MetaSynHostedCostAuthorizationGroupV1(
            group="packet_slot_ceiling",
            call_count=256,
            structured_json_schema_call_count=0,
            prompt_json_schema_call_count=256,
            conservative_input_token_ceiling=4_500_000,
            maximum_output_token_ceiling=524_288,
            cost_ceiling_usd_micros=16_000_000,
            request_roster_sha256=hash_canonical("packet"),
            transport_mode_roster_sha256=hash_canonical(
                ["prompt_json_schema"] * 256
            ),
        ),
    ]
    payload: dict[str, Any] = {
        "authorization_version": ("metasyn-bounded-hosted-pre-first-call-cost-authorization-v2"),
        "status": "authorized_before_first_provider_call",
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
        "config_sha256": bundle.config_sha256,
        "anthropic_config_sha256": bundle.anthropic_config_sha256,
        "provider_identity_sha256": bundle.provider_identity_sha256,
        "provider_model": "claude-sonnet-5",
        "provider_pricing_table_sha256": (bundle.anthropic_config.pricing_table_sha256),
        "provider_pricing_verified_date": "2026-08-28",
        "groups": groups,
        "maximum_theoretical_provider_calls": 296,
        "maximum_structured_json_schema_calls": 35,
        "maximum_prompt_json_schema_calls": 261,
        "conservative_input_token_ceiling": 7_000_000,
        "maximum_output_token_ceiling": 581_632,
        "cost_ceiling_usd_micros": 20_000_000,
        "configured_input_token_ceiling": (bundle.runtime_config.maximum_input_tokens_all_calls),
        "configured_cost_ceiling_usd_micros": (bundle.maximum_authorized_cost_usd_micros),
        "actual_candidate_independent_packet_bound": True,
        "packet_bound_method": (
            "per_row_maximum_exact_rendered_packet_request_over_all_effect_families_"
            "reused_for_eight_slots"
        ),
        "provider_calls_made_before_authorization": 0,
    }
    return MetaSynHostedCostAuthorizationReceiptV1.model_validate(
        {**payload, "authorization_sha256": hash_canonical(payload)}
    )


@pytest.fixture()
def runtime_workspace(
    tmp_path: Path,
    hosted_bundle: MetaSynHostedExecutionBundleV1,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, MetaSynHostedExecutionBundleV1]:
    import literature_multiverse.metasyn_bounded_hosted_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "validate_metasyn_bounded_adapter_bundle_external_replay",
        lambda **kwargs: MetaSynBoundedAdapterBundleV1.model_validate(kwargs["adapter_bundle"]),
    )
    # Stage-machine tests exercise immutable artifact replay many times.  Keep
    # those tests focused on that boundary instead of repeatedly recomputing the
    # repository-wide code fingerprint.  The dedicated test below still runs
    # the real current-bundle validator once.
    monkeypatch.setattr(
        runtime,
        "validate_current_metasyn_hosted_execution_bundle",
        lambda **kwargs: MetaSynHostedExecutionBundleV1.model_validate(kwargs["execution_bundle"]),
    )
    frozen_authorization = _authorization(hosted_bundle)
    monkeypatch.setattr(
        runtime,
        "freeze_metasyn_hosted_cost_authorization",
        lambda **_: frozen_authorization,
    )
    workspace = tmp_path / "hosted-runtime"
    write_metasyn_hosted_execution_bundle(
        execution_bundle=hosted_bundle,
        workspace=workspace,
        repository_root=REPOSITORY_ROOT,
    )
    return workspace, hosted_bundle


def test_hosted_bundle_is_closed_and_current(
    hosted_bundle: MetaSynHostedExecutionBundleV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import literature_multiverse.metasyn_bounded_hosted_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "validate_metasyn_bounded_adapter_bundle_external_replay",
        lambda **kwargs: MetaSynBoundedAdapterBundleV1.model_validate(kwargs["adapter_bundle"]),
    )
    assert (
        runtime.validate_current_metasyn_hosted_execution_bundle(
            execution_bundle=hosted_bundle,
            repository_root=REPOSITORY_ROOT,
            external_replay=True,
        )
        == hosted_bundle
    )


class FakeHostedClient:
    def __init__(self) -> None:
        self.requests: list[AnthropicBoundedRequestV1] = []

    def generate(self, request: AnthropicBoundedRequestV1) -> AnthropicBoundedResultV1:
        self.requests.append(request)
        parsed = self.response_payload(request)
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
            request_cost_ceiling_usd=(request.cost_ceiling.request_cost_ceiling_usd),
            charged_cost_upper_bound_usd=(request.cost_ceiling.request_cost_ceiling_usd),
        )
        payload: dict[str, Any] = {
            "result_version": "anthropic-bounded-result-v2",
            "provider": "anthropic",
            "request_sha256": request.request_sha256,
            "identity_sha256": request.identity_sha256,
            "config_sha256": request.config_sha256,
            "compiled_schema_sha256": request.compiled_schema_sha256,
            "original_schema_sha256": (request.compiled_schema.original_schema_sha256),
            "wire_schema_sha256": request.compiled_schema.wire_schema_sha256,
            "full_acceptance_schema_sha256": (request.full_acceptance_schema_sha256),
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
            "response_id": f"msg_fake_{len(self.requests)}",
            "response_model": "claude-sonnet-5",
            "stop_reason": "end_turn",
            "text": text,
            "response_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "parsed_json": parsed,
            "parsed_json_sha256": hash_canonical(parsed),
            "usage": usage,
            "cost": cost,
            "failure": None,
        }
        return AnthropicBoundedResultV1.model_validate(
            {**payload, "result_sha256": hash_canonical(payload)}
        )

    def response_payload(self, request: AnthropicBoundedRequestV1) -> dict[str, Any]:
        if "FIXTURE_JSON=" in request.prompt:
            return json.loads(request.prompt.split("FIXTURE_JSON=", 1)[1])
        return {
            "inventory_version": "native-candidate-inventory-v1",
            "inventory_status": "no_candidate_found",
            "candidates": [],
            "has_more_or_uncertain": False,
        }


class InvalidInventoryClient(FakeHostedClient):
    def response_payload(self, request: AnthropicBoundedRequestV1) -> dict[str, Any]:
        if "FIXTURE_JSON=" in request.prompt:
            return super().response_payload(request)
        return {}


def _run_preflight(
    workspace: Path,
    bundle: MetaSynHostedExecutionBundleV1,
    client: FakeHostedClient,
) -> Any:
    return run_metasyn_hosted_preflight(
        workspace=workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )


def test_complete_staged_runtime_resumes_without_repeating_calls(
    runtime_workspace: tuple[Path, MetaSynHostedExecutionBundleV1],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, bundle = runtime_workspace
    client = FakeHostedClient()
    preflight = _run_preflight(workspace, bundle, client)
    assert bundle.execution_bundle_version.endswith("-v2")
    assert bundle.runtime_version.endswith("-v2")
    assert preflight.preflight_version.endswith("-v2")
    assert preflight.status == "passed"
    assert preflight.structured_json_schema_calls == 3
    assert preflight.prompt_json_schema_calls == 5
    assert len(client.requests) == 8

    smoke = run_metasyn_hosted_smoke(
        workspace=workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )
    assert smoke.status == "passed"
    assert len(client.requests) == 9

    ledger = run_metasyn_hosted_full_roster(
        workspace=workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )
    assert ledger.publication_count == 32
    assert ledger.ledger_version.endswith("-v2")
    assert ledger.total_provider_call_attempts_or_possible_attempts == 40
    assert ledger.structured_json_schema_calls == 35
    assert ledger.prompt_json_schema_calls == 5
    assert ledger.durable_intent_count == 40
    assert ledger.durable_intent_count == (
        ledger.total_provider_call_attempts_or_possible_attempts
    )
    assert ledger.observed_request_ceiling_usd_micros == (
        ledger.cost.request_ceiling_usd_micros
    )
    assert ledger.durable_intent_liability_usd_micros == (
        ledger.observed_request_ceiling_usd_micros
        + ledger.possible_ambiguous_charge_ceiling_usd_micros
    )
    assert ledger.durable_intent_liability_usd_micros <= (
        ledger.cost_authorization_ceiling_usd_micros
    )
    assert ledger.durable_intent_liability_usd_micros <= (
        ledger.configured_cost_ceiling_usd_micros
    )
    assert len(client.requests) == 40
    assert all(item.status == "adapter_inventory_no_candidate" for item in ledger.row_results)

    # Every completed command is an immutable replay, never a second provider call.
    assert _run_preflight(workspace, bundle, client) == preflight
    assert (
        run_metasyn_hosted_smoke(
            workspace=workspace,
            repository_root=REPOSITORY_ROOT,
            client=client,
            expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
        )
        == smoke
    )
    assert (
        run_metasyn_hosted_full_roster(
            workspace=workspace,
            repository_root=REPOSITORY_ROOT,
            client=client,
            expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
        )
        == ledger
    )
    assert len(client.requests) == 40

    report = finalize_metasyn_hosted_runtime(
        workspace=workspace,
        repository_root=REPOSITORY_ROOT,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )
    assert report.provider_neutral_yield_report is not None
    assert report.report_version.endswith("-v2")
    assert report.typed_publication_output_count == 0
    assert report.maximum_theoretical_provider_calls == 296
    assert report.structured_json_schema_calls == 35
    assert report.prompt_json_schema_calls == 5
    assert report.durable_intent_count == ledger.durable_intent_count
    assert (
        report.durable_intent_liability_usd_micros
        == ledger.durable_intent_liability_usd_micros
    )
    assert report.durable_intent_roster_sha256 == (
        ledger.durable_intent_roster_sha256
    )
    assert (
        validate_finalized_metasyn_hosted_runtime(
            workspace=workspace,
            repository_root=REPOSITORY_ROOT,
            expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
        )
        == report
    )
    assert len(client.requests) == 40

    args = Namespace(
        repository_root=REPOSITORY_ROOT,
        workspace=workspace,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )
    monkeypatch.setattr(hosted_cli, "_live_client", lambda *_args, **_kwargs: client)
    monkeypatch.setattr(
        hosted_cli, "run_metasyn_hosted_preflight", lambda **_kwargs: preflight
    )
    monkeypatch.setattr(
        hosted_cli, "run_metasyn_hosted_full_roster", lambda **_kwargs: ledger
    )
    monkeypatch.setattr(
        hosted_cli, "finalize_metasyn_hosted_runtime", lambda **_kwargs: report
    )
    monkeypatch.setattr(
        hosted_cli, "validate_finalized_metasyn_hosted_runtime", lambda **_kwargs: report
    )
    summaries = (
        hosted_cli._preflight(args),
        hosted_cli._full(args),
        hosted_cli._finalize(args),
        hosted_cli._validate_final(args),
    )
    required_audit_fields = {
        "structured_json_schema_calls",
        "prompt_json_schema_calls",
        "maximum_structured_json_schema_calls",
        "maximum_prompt_json_schema_calls",
        "transport_mode_policy",
        "cost_authorization_sha256",
        "observed_request_ceiling_usd_micros",
        "possible_ambiguous_charge_ceiling_usd_micros",
        "durable_intent_liability_usd_micros",
        "cost_authorization_ceiling_usd_micros",
        "configured_cost_ceiling_usd_micros",
        "durable_intent_roster_sha256",
    }
    assert all(required_audit_fields <= set(summary) for summary in summaries)
    assert summaries[0]["exact_request_roster_size"] == 8
    assert summaries[0]["new_provider_call_attempts_this_invocation"] == 0
    assert summaries[0]["reused_terminal_outcomes_this_invocation"] == 8
    assert summaries[0]["new_terminal_incidents_this_invocation"] == 0

    spec = synthetic_schema_v2_preflight_specs()[0]
    extra_intent = freeze_metasyn_hosted_attempt_intent(
        execution_bundle=bundle,
        request_key="unreferenced-intent",
        stage="preflight",
        prompt=_preflight_prompt(spec),
        schema_bundle=_preflight_schema_bundle(spec),
        cost_authorization_sha256=_authorization(bundle).authorization_sha256,
    )
    atomic_write_json(
        metasyn_hosted_runtime_paths(workspace)["intents"]
        / "unreferenced-intent.json",
        extra_intent,
    )
    with pytest.raises(
        MetaSynHostedRuntimeError,
        match="durable_intent_artifact_roster_mismatch",
    ):
        validate_finalized_metasyn_hosted_runtime(
            workspace=workspace,
            repository_root=REPOSITORY_ROOT,
            expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
        )


def test_orphaned_intent_is_poisoned_and_never_retried(
    runtime_workspace: tuple[Path, MetaSynHostedExecutionBundleV1],
) -> None:
    workspace, bundle = runtime_workspace
    paths = metasyn_hosted_runtime_paths(workspace)
    atomic_write_json(paths["cost_authorization"], _authorization(bundle))
    paths["intents"].mkdir(parents=True)
    paths["receipts"].mkdir(parents=True)
    paths["incidents"].mkdir(parents=True)
    paths["row_results"].mkdir(parents=True)
    spec = synthetic_schema_v2_preflight_specs()[0]
    schema_bundle = _preflight_schema_bundle(spec)
    intent = freeze_metasyn_hosted_attempt_intent(
        execution_bundle=bundle,
        request_key="preflight-00",
        stage="preflight",
        prompt=_preflight_prompt(spec),
        schema_bundle=schema_bundle,
        cost_authorization_sha256=_authorization(bundle).authorization_sha256,
    )
    atomic_write_json(paths["intents"] / "preflight-00.json", intent)

    client = FakeHostedClient()
    receipt = _run_preflight(workspace, bundle, client)
    assert receipt.status == "failed"
    assert receipt.possible_ambiguous_provider_calls == 1
    assert (
        receipt.possible_ambiguous_charge_ceiling_usd_micros
        == intent.request_cost_ceiling_usd_micros
    )
    incident_payload = json.loads(
        (paths["incidents"] / "preflight-00.json").read_text(encoding="utf-8")
    )
    assert incident_payload["transport_mode"] == "structured_json_schema"
    assert (
        incident_payload["request_cost_ceiling_usd_micros"]
        == intent.request_cost_ceiling_usd_micros
    )
    assert len(client.requests) == 7
    assert _run_preflight(workspace, bundle, client) == receipt
    assert len(client.requests) == 7


def test_receipt_rejects_coherently_rehashed_provider_wire_and_cost_drift(
    runtime_workspace: tuple[Path, MetaSynHostedExecutionBundleV1],
) -> None:
    workspace, bundle = runtime_workspace
    client = FakeHostedClient()
    assert _run_preflight(workspace, bundle, client).status == "passed"
    receipt_path = metasyn_hosted_runtime_paths(workspace)["receipts"] / "preflight-00.json"
    original = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload = json.loads(json.dumps(original))

    provider_result = payload["provider_result"]
    provider_result["wire_call_sha256"] = "f" * 64
    provider_result["result_sha256"] = hash_canonical(
        {
            key: value
            for key, value in provider_result.items()
            if key != "result_sha256"
        }
    )
    payload["provider_result_sha256"] = provider_result["result_sha256"]
    payload["receipt_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )

    with pytest.raises(ValueError, match="provider_result_binding_mismatch"):
        MetaSynHostedCallReceiptV1.model_validate(payload)

    payload = json.loads(json.dumps(original))
    provider_result = payload["provider_result"]
    provider_result["cost"]["request_cost_ceiling_usd"] = "1"
    provider_result["cost"]["charged_cost_upper_bound_usd"] = "1"
    provider_result["result_sha256"] = hash_canonical(
        {
            key: value
            for key, value in provider_result.items()
            if key != "result_sha256"
        }
    )
    payload["provider_result_sha256"] = provider_result["result_sha256"]
    payload["receipt_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="provider_result_binding_mismatch"):
        MetaSynHostedCallReceiptV1.model_validate(payload)


def test_invalid_inventory_is_a_terminal_row_result_and_never_retried(
    runtime_workspace: tuple[Path, MetaSynHostedExecutionBundleV1],
) -> None:
    workspace, bundle = runtime_workspace
    preflight_client = FakeHostedClient()
    assert _run_preflight(workspace, bundle, preflight_client).status == "passed"

    client = InvalidInventoryClient()
    smoke = run_metasyn_hosted_smoke(
        workspace=workspace,
        repository_root=REPOSITORY_ROOT,
        client=client,
        expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
    )
    assert smoke.status == "failed"
    assert smoke.row_status == "runtime_inventory_blocked"
    assert len(client.requests) == 1
    row_payload = json.loads(
        (metasyn_hosted_runtime_paths(workspace)["row_results"] / "row-00.json").read_text()
    )
    assert row_payload["adapter_publication_result"] is None
    assert row_payload["observed_provider_calls"] == 1
    assert row_payload["blockers"] == ["inventory_response:wire_schema_invalid"]

    assert (
        run_metasyn_hosted_smoke(
            workspace=workspace,
            repository_root=REPOSITORY_ROOT,
            client=client,
            expected_execution_bundle_sha256=bundle.execution_bundle_sha256,
        )
        == smoke
    )
    assert len(client.requests) == 1


def test_tampered_cost_authorization_blocks_before_any_call(
    runtime_workspace: tuple[Path, MetaSynHostedExecutionBundleV1],
) -> None:
    workspace, bundle = runtime_workspace
    expected = _authorization(bundle)
    payload = expected.model_dump(mode="json", exclude={"authorization_sha256"})
    payload["groups"][0]["request_roster_sha256"] = "f" * 64
    tampered = MetaSynHostedCostAuthorizationReceiptV1.model_validate(
        {**payload, "authorization_sha256": hash_canonical(payload)}
    )
    atomic_write_json(metasyn_hosted_runtime_paths(workspace)["cost_authorization"], tampered)
    client = FakeHostedClient()
    with pytest.raises(MetaSynHostedRuntimeError, match="cost_authorization_replay_mismatch"):
        _run_preflight(workspace, bundle, client)
    assert not client.requests


def test_exact_296_call_ceiling_rejects_297th_before_client(
    runtime_workspace: tuple[Path, MetaSynHostedExecutionBundleV1],
) -> None:
    workspace, bundle = runtime_workspace
    paths = metasyn_hosted_runtime_paths(workspace)
    paths["intents"].mkdir(parents=True)
    for index in range(MAX_THEORETICAL_PROVIDER_CALLS):
        (paths["intents"] / f"occupied-{index:03d}.json").touch()
    spec = synthetic_schema_v2_preflight_specs()[0]
    schema_bundle = _preflight_schema_bundle(spec)
    intent = freeze_metasyn_hosted_attempt_intent(
        execution_bundle=bundle,
        request_key="preflight-00",
        stage="preflight",
        prompt=_preflight_prompt(spec),
        schema_bundle=schema_bundle,
        cost_authorization_sha256=_authorization(bundle).authorization_sha256,
    )
    with pytest.raises(MetaSynHostedRuntimeError, match="296_call_ceiling"):
        _assert_new_call_within_budget(
            workspace=workspace,
            bundle=bundle,
            intent=intent,
            authorization=_authorization(bundle),
        )


def test_subgroup_cost_ceiling_blocks_next_call_with_global_spare(
    runtime_workspace: tuple[Path, MetaSynHostedExecutionBundleV1],
) -> None:
    workspace, bundle = runtime_workspace
    base_authorization = _authorization(bundle)
    paths = metasyn_hosted_runtime_paths(workspace)
    paths["intents"].mkdir(parents=True)
    specs = synthetic_schema_v2_preflight_specs()
    probe = freeze_metasyn_hosted_attempt_intent(
        execution_bundle=bundle,
        request_key="preflight-00",
        stage="preflight",
        prompt=_preflight_prompt(specs[0]),
        schema_bundle=_preflight_schema_bundle(specs[0]),
        cost_authorization_sha256=base_authorization.authorization_sha256,
    )
    authorization_payload = base_authorization.model_dump(
        mode="json", exclude={"authorization_sha256"}
    )
    released = (
        authorization_payload["groups"][0]["cost_ceiling_usd_micros"]
        - probe.request_cost_ceiling_usd_micros
    )
    assert released > 0
    authorization_payload["groups"][0]["cost_ceiling_usd_micros"] -= released
    authorization_payload["groups"][2]["cost_ceiling_usd_micros"] += released
    authorization = MetaSynHostedCostAuthorizationReceiptV1.model_validate(
        {
            **authorization_payload,
            "authorization_sha256": hash_canonical(authorization_payload),
        }
    )
    prior = freeze_metasyn_hosted_attempt_intent(
        execution_bundle=bundle,
        request_key="preflight-00",
        stage="preflight",
        prompt=_preflight_prompt(specs[0]),
        schema_bundle=_preflight_schema_bundle(specs[0]),
        cost_authorization_sha256=authorization.authorization_sha256,
    )
    proposed = freeze_metasyn_hosted_attempt_intent(
        execution_bundle=bundle,
        request_key="preflight-01",
        stage="preflight",
        prompt=_preflight_prompt(specs[1]),
        schema_bundle=_preflight_schema_bundle(specs[1]),
        cost_authorization_sha256=authorization.authorization_sha256,
    )
    atomic_write_json(paths["intents"] / "preflight-00.json", prior)

    with pytest.raises(
        MetaSynHostedRuntimeError,
        match="authorization_group_cost_ceiling_usd_micros_exceeded",
    ):
        _assert_new_call_within_budget(
            workspace=workspace,
            bundle=bundle,
            intent=proposed,
            authorization=authorization,
        )


def test_group_and_transport_ceilings_reject_33rd_inventory_36th_structured_and_257th_packet(
    hosted_bundle: MetaSynHostedExecutionBundleV1,
) -> None:
    authorization = _authorization(hosted_bundle)
    row = hosted_bundle.adapter_bundle.row_contexts[0]
    inventory_prompt, inventory_bundle, _ = _request_surface(
        row=row,
        stage="inventory",
    )
    _, inventory_request = _freeze_provider_request(
        bundle=hosted_bundle,
        stage="inventory",
        request_key="group-count-inventory",
        prompt=inventory_prompt,
        schema_bundle=inventory_bundle,
    )
    packet_spec = synthetic_schema_v2_preflight_specs()[3]
    _, packet_mode_request = _freeze_provider_request(
        bundle=hosted_bundle,
        stage="preflight",
        request_key="group-count-packet-mode",
        prompt=_preflight_prompt(packet_spec),
        schema_bundle=_preflight_schema_bundle(packet_spec),
    )

    with pytest.raises(
        MetaSynHostedRuntimeError,
        match="global_transport_mode_ceiling_exceeded",
    ):
        _assert_authorization_group_ceilings(
            classified_requests=[
                *[
                    ("source_free_preflight", inventory_request, None)
                    for _ in range(3)
                ],
                *[
                    ("exact_inventory", inventory_request, index % 32)
                    for index in range(33)
                ],
            ],
            authorization=authorization,
            configured_cost_ceiling_usd_micros=(
                hosted_bundle.maximum_authorized_cost_usd_micros
            ),
        )
    with pytest.raises(
        MetaSynHostedRuntimeError,
        match="authorization_group_call_count_exceeded",
    ):
        _assert_authorization_group_ceilings(
            classified_requests=[
                ("exact_inventory", inventory_request, index % 32)
                for index in range(33)
            ],
            authorization=authorization,
            configured_cost_ceiling_usd_micros=(
                hosted_bundle.maximum_authorized_cost_usd_micros
            ),
        )
    with pytest.raises(
        MetaSynHostedRuntimeError,
        match="authorization_group_call_count_exceeded",
    ):
        _assert_authorization_group_ceilings(
            classified_requests=[
                ("packet_slot_ceiling", packet_mode_request, index % 32)
                for index in range(257)
            ],
            authorization=authorization,
            configured_cost_ceiling_usd_micros=(
                hosted_bundle.maximum_authorized_cost_usd_micros
            ),
        )
    with pytest.raises(
        MetaSynHostedRuntimeError,
        match="packet_per_publication_ceiling_exceeded",
    ):
        _assert_authorization_group_ceilings(
            classified_requests=[
                ("packet_slot_ceiling", packet_mode_request, 0)
                for _ in range(9)
            ],
            authorization=authorization,
            configured_cost_ceiling_usd_micros=(
                hosted_bundle.maximum_authorized_cost_usd_micros
            ),
        )


def test_coherent_transport_substitution_fails_authoritative_roster_replay(
    runtime_workspace: tuple[Path, MetaSynHostedExecutionBundleV1],
) -> None:
    workspace, bundle = runtime_workspace
    authorization = _authorization(bundle)
    structured_spec = synthetic_schema_v2_preflight_specs()[0]
    substituted = freeze_metasyn_hosted_attempt_intent(
        execution_bundle=bundle,
        request_key="preflight-03",
        stage="preflight",
        prompt=_preflight_prompt(structured_spec),
        schema_bundle=_preflight_schema_bundle(structured_spec),
        cost_authorization_sha256=authorization.authorization_sha256,
    )
    with pytest.raises(
        MetaSynHostedRuntimeError,
        match="authoritative_budget_intent_replay_mismatch",
    ):
        _assert_new_call_within_budget(
            workspace=workspace,
            bundle=bundle,
            intent=substituted,
            authorization=authorization,
        )


def test_bundle_and_artifacts_contain_no_labels_or_credentials(
    runtime_workspace: tuple[Path, MetaSynHostedExecutionBundleV1],
) -> None:
    workspace, bundle = runtime_workspace
    serialized = json.dumps(bundle.model_dump(mode="json"), sort_keys=True).casefold()
    for forbidden in (
        "api_key",
        "authorization",
        "bearer ",
        "sk-ant-",
        "conclusion_summary",
        "effect_direction",
        "reference_verdict",
    ):
        assert forbidden not in serialized
    config_payload = json.loads(
        (REPOSITORY_ROOT / "configs/benchmarks/metasyn-bounded-anthropic-v1.json").read_text()
    )
    assert "api_key" not in json.dumps(config_payload).casefold()
    assert bundle.maximum_theoretical_provider_calls == 8 + 32 + 32 * 8
    assert sha256_file(REPOSITORY_ROOT / "configs/benchmarks/metasyn-bounded-anthropic-v1.json")
    assert not metasyn_hosted_runtime_paths(workspace)["preflight"].exists()


def test_legacy_hosted_v1_artifact_literals_are_rejected(
    hosted_bundle: MetaSynHostedExecutionBundleV1,
) -> None:
    payload = hosted_bundle.model_dump(mode="json")
    payload["execution_bundle_version"] = (
        "metasyn-bounded-hosted-execution-bundle-v1"
    )
    payload["runtime_version"] = "metasyn-bounded-hosted-anthropic-runtime-v1"
    with pytest.raises(ValueError):
        MetaSynHostedExecutionBundleV1.model_validate(payload)


def test_prepare_workspace_never_overwrites_existing_artifacts(
    runtime_workspace: tuple[Path, MetaSynHostedExecutionBundleV1],
) -> None:
    workspace, bundle = runtime_workspace
    with pytest.raises(OutputExistsError):
        write_metasyn_hosted_execution_bundle(
            execution_bundle=bundle,
            workspace=workspace,
            repository_root=REPOSITORY_ROOT,
        )


def test_cli_validation_is_offline_and_live_stage_requires_explicit_flag(
    runtime_workspace: tuple[Path, MetaSynHostedExecutionBundleV1],
) -> None:
    workspace, bundle = runtime_workspace
    script = REPOSITORY_ROOT / "scripts/run_metasyn_bounded_hosted_runtime.py"
    environment = {key: value for key, value in os.environ.items() if key != "ANTHROPIC_API_KEY"}
    validated = subprocess.run(
        [
            sys.executable,
            str(script),
            "validate-bundle",
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--workspace",
            str(workspace),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["provider_calls_made"] is False

    blocked = subprocess.run(
        [
            sys.executable,
            str(script),
            "preflight",
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--workspace",
            str(workspace),
            "--expected-execution-bundle-sha256",
            bundle.execution_bundle_sha256,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert blocked.returncode != 0
    assert "metasyn_hosted_live_flag_required" in blocked.stderr
    assert not metasyn_hosted_runtime_paths(workspace)["cost_authorization"].exists()

    missing_key = subprocess.run(
        [
            sys.executable,
            str(script),
            "preflight",
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--workspace",
            str(workspace),
            "--expected-execution-bundle-sha256",
            bundle.execution_bundle_sha256,
            "--live",
            "--env-file",
            str(workspace / "does-not-exist.env"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert missing_key.returncode != 0
    assert "metasyn_hosted_anthropic_api_key_missing_pre_call" in missing_key.stderr
    assert not metasyn_hosted_runtime_paths(workspace)["cost_authorization"].exists()


def test_v7_preflight_prompts_bind_projected_wire_valid_fixtures_and_modes(
    hosted_bundle: MetaSynHostedExecutionBundleV1,
) -> None:
    expected_counts = [
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (10, 8, 15, 11),
        (10, 9, 16, 11),
        (10, 9, 16, 11),
        (10, 9, 16, 11),
        (10, 9, 16, 11),
    ]
    for index, (spec, expected) in enumerate(
        zip(synthetic_schema_v2_preflight_specs(), expected_counts, strict=True)
    ):
        fixture = _preflight_fixture(spec)
        prompt = _preflight_prompt(spec)
        _, request = _freeze_provider_request(
            bundle=hosted_bundle,
            stage="preflight",
            request_key=f"materialized-preflight-{index:02d}",
            prompt=prompt,
            schema_bundle=_preflight_schema_bundle(spec),
        )
        canonical_fixture = json.dumps(fixture, sort_keys=True, separators=(",", ":"))
        assert prompt.endswith(f"FIXTURE_JSON={canonical_fixture}")
        assert f"PROJECTED_FIXTURE_SHA256={hash_canonical(fixture)}" in prompt
        assert (
            request.compiled_schema.nullable_optional_promotion_count,
            request.compiled_schema.nullable_optional_null_stripping_count,
            request.compiled_schema.wire_optional_parameter_count,
            request.compiled_schema.wire_union_parameter_count,
        ) == expected
        assert Draft202012Validator(spec["provider_schema"]).is_valid(fixture)
        assert Draft202012Validator(spec["full_acceptance_schema"]).is_valid(fixture)
        assert Draft202012Validator(request.compiled_schema.wire_schema).is_valid(
            fixture
        )
        expected_mode = (
            "structured_json_schema" if index < 3 else "prompt_json_schema"
        )
        assert request.transport_mode == expected_mode
        assert request.output_format_present_in_call == (index < 3)
        assert request.model_prompt == prompt
        if index < 3:
            assert request.model_system == request.system
        else:
            wire_json = canonical_json_bytes(
                request.compiled_schema.wire_schema
            ).decode("utf-8")
            assert request.model_system.endswith(wire_json)
            assert request.model_system.count(wire_json) == 1


def test_v7_all_frozen_rows_and_effect_families_fit_caps_and_mode_policy(
    hosted_bundle: MetaSynHostedExecutionBundleV1,
) -> None:
    effect_kinds = (
        "binary_group_statistics",
        "continuous_group_statistics",
        "direct_confidence_interval",
        "direct_standard_error",
        "direct_variance",
    )
    observed = 0
    for row_ordinal, row in enumerate(hosted_bundle.adapter_bundle.row_contexts):
        inventory_prompt, inventory_bundle, _ = _request_surface(
            row=row,
            stage="inventory",
        )
        _, inventory_request = _freeze_provider_request(
            bundle=hosted_bundle,
            stage="inventory",
            request_key=f"grammar-row-{row_ordinal:02d}-inventory",
            prompt=inventory_prompt,
            schema_bundle=inventory_bundle,
        )
        assert inventory_request.compiled_schema.wire_optional_parameter_count == 0
        assert inventory_request.compiled_schema.wire_union_parameter_count == 0
        assert inventory_request.transport_mode == "structured_json_schema"
        assert inventory_request.output_format_present_in_call is True

        line_id = sorted(row.source_row.projection.exposed_line_ids)[0]
        outcome_name = sorted(row.allowed_outcomes)[0]
        for effect_kind in effect_kinds:
            inventory_receipt = freeze_metasyn_inventory_validation_receipt(
                row=row,
                value={
                    "inventory_version": "native-candidate-inventory-v1",
                    "inventory_status": "candidates_found",
                    "candidates": [
                        {
                            "candidate_index": 1,
                            "outcome_name": outcome_name,
                            "effect_kind": effect_kind,
                            "line_ids": [line_id],
                        }
                    ],
                    "has_more_or_uncertain": False,
                },
            )
            packet_prompt, packet_bundle, _ = _request_surface(
                row=row,
                stage="packet",
                inventory_receipt=inventory_receipt,
                candidate_index=1,
            )
            _, packet_request = _freeze_provider_request(
                bundle=hosted_bundle,
                stage="packet",
                request_key=(
                    f"grammar-row-{row_ordinal:02d}-packet-{effect_kind}"
                ),
                prompt=packet_prompt,
                schema_bundle=packet_bundle,
            )
            expected_optional = 15 if effect_kind == effect_kinds[0] else 16
            assert (
                packet_request.compiled_schema.wire_optional_parameter_count
                == expected_optional
            )
            assert packet_request.compiled_schema.wire_union_parameter_count == 11
            assert packet_request.transport_mode == "prompt_json_schema"
            assert packet_request.output_format_present_in_call is False
            assert packet_request.model_prompt == packet_prompt
            observed += 1
    assert observed == 32 * len(effect_kinds)


def test_v5_real_all_roster_cost_authorization_binds_35_261_under_cap(
    hosted_bundle: MetaSynHostedExecutionBundleV1,
) -> None:
    authorization = freeze_metasyn_hosted_cost_authorization(
        execution_bundle=hosted_bundle
    )

    assert authorization.maximum_theoretical_provider_calls == 296
    assert authorization.maximum_structured_json_schema_calls == 35
    assert authorization.maximum_prompt_json_schema_calls == 261
    assert [
        (
            item.structured_json_schema_call_count,
            item.prompt_json_schema_call_count,
        )
        for item in authorization.groups
    ] == [(3, 5), (32, 0), (0, 256)]
    assert authorization.cost_ceiling_usd_micros <= 20_000_000
    assert (
        authorization.cost_ceiling_usd_micros
        <= hosted_bundle.maximum_authorized_cost_usd_micros
    )


def test_all_preflight_and_inventory_requests_are_credential_free(
    hosted_bundle: MetaSynHostedExecutionBundleV1,
) -> None:
    requests: list[AnthropicBoundedRequestV1] = []
    for index, spec in enumerate(synthetic_schema_v2_preflight_specs()):
        _, request = _freeze_provider_request(
            bundle=hosted_bundle,
            stage="preflight",
            request_key=f"test-preflight-{index:02d}",
            prompt=_preflight_prompt(spec),
            schema_bundle=_preflight_schema_bundle(spec),
        )
        requests.append(request)
    serialized = json.dumps(
        [item.model_dump(mode="json") for item in requests], sort_keys=True
    ).casefold()
    assert "api_key" not in serialized
    assert "authorization" not in serialized
    assert "sk-ant-" not in serialized


def test_historical_qwen_attempt06_adapter_is_stale_only_in_upstream_pilot_identity() -> None:
    root = require_private_cache(
        "data/cache/metasyn/bounded-qwen-yield-v2-attempt-06/execution-bundle.private.json",
        "data/cache/metasyn/typed-oracle-pilot-v2",
    )
    frozen = json.loads(
        (
            root / "data/cache/metasyn/bounded-qwen-yield-v2-attempt-06/"
            "execution-bundle.private.json"
        ).read_text(encoding="utf-8")
    )
    adapter = MetaSynBoundedAdapterBundleV1.model_validate(frozen["adapter_bundle"])
    current_pilot_sha, _downstream = runtime._pilot_downstream_sha(root)
    assert current_pilot_sha != adapter.upstream_pilot_pipeline_sha256
    config, config_sha = load_metasyn_hosted_runtime_config(repository_root=root)
    with pytest.raises(MetaSynHostedRuntimeError, match="metasyn_hosted_adapter_upstream_stale"):
        freeze_metasyn_hosted_execution_bundle(
            adapter_bundle=adapter,
            runtime_config=config,
            config_file_sha256=config_sha,
            pilot_workspace_relative="data/cache/metasyn/typed-oracle-pilot-v2",
            repository_root=root,
        )
