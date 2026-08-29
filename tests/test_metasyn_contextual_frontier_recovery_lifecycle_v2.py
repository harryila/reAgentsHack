from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import literature_multiverse.metasyn_contextual_frontier_recovery_lifecycle_v2 as lifecycle
import literature_multiverse.metasyn_contextual_frontier_recovery_v2 as core
from literature_multiverse.metasyn_contextual_frontier_runtime_v1 import (
    MODEL,
    MetaSynContextualFrontierUsageV1,
    _persist_json,
    freeze_metasyn_contextual_frontier_provider_result_v1,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def plan() -> core.MetaSynContextualFrontierRecoveryPlanV2:
    cached = os.environ.get("LM_FRONTIER_RECOVERY_V2_TEST_PLAN")
    if cached:
        raw = json.loads(Path(cached).read_text(encoding="utf-8"))
        if "plan" in raw:
            raw = raw["plan"]
        return core.MetaSynContextualFrontierRecoveryPlanV2.model_validate(raw)
    return core.freeze_metasyn_contextual_frontier_recovery_plan_v2(repository_root=ROOT)


def _workspace(
    *, tmp_path: Path, plan: core.MetaSynContextualFrontierRecoveryPlanV2
) -> tuple[
    Path,
    lifecycle.MetaSynContextualFrontierRecoveryLifecyclePreparedV2,
    lifecycle.MetaSynContextualFrontierRecoveryLifecycleAuthorizationV2,
]:
    workspace = tmp_path / "recovery"
    workspace.mkdir(mode=0o700)
    prepared = lifecycle._freeze_prepared(repository_root=ROOT, plan=plan)
    _persist_json(workspace / "00-prepared.json", prepared)
    authorization = lifecycle.authorize_metasyn_contextual_frontier_recovery_lifecycle_v2(
        workspace=workspace,
        phase_budget_usd_micros=plan.hard_cost_liability_usd_micros,
    )
    return workspace, prepared, authorization


def _completed_raw(
    plan: core.MetaSynContextualFrontierRecoveryPlanV2,
) -> dict[str, Any]:
    return {
        "response_version": "metasyn-contextual-frontier-recovery-response-v2",
        "status": "completed",
        "target_contract_sha256": plan.target_spec_sha256,
        "claims_by_field": {
            claim.field_path: {
                "passage_id": claim.passage_id,
                "support_quote": claim.support_quote,
                "context": claim.context,
                "token": claim.token,
                "normalization": claim.normalization,
            }
            for claim in plan.evaluator_fixture.model_outcome.claims
        },
    }


class _CountingClient:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls = 0

    def generate(self, request: Any) -> Any:
        self.calls += 1
        assert request.request_key == core.REQUEST_KEY
        return self.result


class _ForbiddenClient:
    calls = 0

    def generate(self, request: Any) -> Any:  # pragma: no cover - must not run
        self.calls += 1
        raise AssertionError("orphaned exact request must never be retried")


def _assert_authorities_false(value: Any) -> None:
    grant_suffix = "_authority"
    found: list[bool] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                if key.endswith(grant_suffix):
                    found.append(item)
                visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(value.model_dump(mode="json"))
    assert found
    assert all(item is False for item in found)


def test_mock_success_is_archived_once_and_never_recalled(
    tmp_path: Path, plan: core.MetaSynContextualFrontierRecoveryPlanV2
) -> None:
    workspace, _, _ = _workspace(tmp_path=tmp_path, plan=plan)
    raw = _completed_raw(plan)
    result = freeze_metasyn_contextual_frontier_provider_result_v1(
        request=plan.request.transport_request,
        outcome="completed",
        response_id="msg_test_recovery_v2",
        response_model=MODEL,
        stop_reason="end_turn",
        text=json.dumps(raw, sort_keys=True, separators=(",", ":")),
        parsed_json=raw,
        usage=MetaSynContextualFrontierUsageV1(input_tokens=100, output_tokens=100),
    )
    client = _CountingClient(result)

    first = lifecycle.execute_metasyn_contextual_frontier_recovery_lifecycle_v2(
        workspace=workspace, client=client
    )
    second = lifecycle.execute_metasyn_contextual_frontier_recovery_lifecycle_v2(
        workspace=workspace, client=client
    )

    assert first == second
    assert client.calls == 1
    assert first.status == "typed_graph_mechanics_observed"
    assert first.fresh_native_typed_graph_observed
    assert first.provider_result_sha256 == result.result_sha256
    assert first.provider_receipt_sha256 is not None
    assert first.validation_sha256 is not None
    assert first.report_sha256
    assert stat_mode(workspace) == 0o700
    assert all(stat_mode(path) == 0o600 for path in workspace.iterdir() if path.is_file())
    _assert_authorities_false(first)

    validation = lifecycle.validate_metasyn_contextual_frontier_recovery_lifecycle_v2(
        repository_root=ROOT,
        workspace=workspace,
        external_replay=False,
    )
    assert validation.status.status == "terminal"
    assert validation.status.provider_result_count == 1
    assert validation.status.provider_receipt_count == 1
    _assert_authorities_false(validation)


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_orphaned_durable_intent_is_terminal_without_provider_retry(
    tmp_path: Path, plan: core.MetaSynContextualFrontierRecoveryPlanV2
) -> None:
    workspace, prepared, authorization = _workspace(tmp_path=tmp_path, plan=plan)
    intent = lifecycle._freeze_intent(prepared=prepared, authorization=authorization)
    _persist_json(workspace / "intent.json", intent)
    client = _ForbiddenClient()

    first = lifecycle.execute_metasyn_contextual_frontier_recovery_lifecycle_v2(
        workspace=workspace, client=client
    )
    second = lifecycle.execute_metasyn_contextual_frontier_recovery_lifecycle_v2(
        workspace=workspace, client=client
    )

    assert first == second
    assert client.calls == 0
    assert first.status == "terminal_ambiguous_attempt_poison"
    assert first.incident is not None
    assert first.incident.incident_kind == "orphan_intent_observed_on_resume"
    assert not first.incident.retry_this_request_permitted
    assert first.provider_result_sha256 is None
    assert first.provider_receipt_sha256 is None
    _assert_authorities_false(first)
