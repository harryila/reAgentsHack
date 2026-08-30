from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from tests.private_cache_support import require_private_cache

from literature_multiverse.lineage import canonical_json_bytes
from literature_multiverse.metasyn_contextual_frontier_recovery_v2 import (
    MetaSynContextualFrontierRecoveryPlanV2,
)
from literature_multiverse.metasyn_contextual_frontier_recovery_v3 import (
    ACCEPTED_COMPILED_SCHEMA_SHA256,
    ACCEPTED_ORIGINAL_SCHEMA_SHA256,
    ACCEPTED_WIRE_SCHEMA_SHA256,
    MetaSynContextualFrontierRecoveryV3Error,
    authorize_metasyn_contextual_frontier_recovery_v3,
    evaluate_metasyn_contextual_frontier_recovery_response_v3,
    execute_metasyn_contextual_frontier_recovery_v3,
    freeze_metasyn_contextual_frontier_recovery_plan_v3,
    freeze_metasyn_contextual_frontier_recovery_request_v3,
    prepare_metasyn_contextual_frontier_recovery_v3,
)
from literature_multiverse.metasyn_contextual_frontier_runtime_v1 import (
    MetaSynContextualFrontierPlanV1,
    MetaSynContextualFrontierProviderResultV1,
    MetaSynContextualFrontierUsageV1,
    freeze_metasyn_contextual_frontier_provider_result_v1,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_v1() -> MetaSynContextualFrontierPlanV1:
    raw = json.loads(
        (ROOT / "data/cache/metasyn/contextual-frontier-runtime-v1/00-prepared.json").read_text(
            encoding="utf-8"
        )
    )
    return MetaSynContextualFrontierPlanV1.model_validate(raw.get("plan", raw))


def _load_v2() -> MetaSynContextualFrontierRecoveryPlanV2:
    raw = json.loads(
        (ROOT / "data/cache/metasyn/contextual-frontier-recovery-v2/00-prepared.json").read_text(
            encoding="utf-8"
        )
    )
    return MetaSynContextualFrontierRecoveryPlanV2.model_validate(raw.get("plan", raw))


@pytest.fixture(scope="module")
def v1_plan() -> MetaSynContextualFrontierPlanV1:
    require_private_cache("data/cache/metasyn/contextual-frontier-runtime-v1/00-prepared.json")
    return _load_v1()


@pytest.fixture(scope="module")
def v2_plan() -> MetaSynContextualFrontierRecoveryPlanV2:
    require_private_cache("data/cache/metasyn/contextual-frontier-recovery-v2/00-prepared.json")
    return _load_v2()


@pytest.fixture(scope="module")
def plan():
    require_private_cache(
        "data/cache/metasyn/contextual-frontier-runtime-v1/00-prepared.json",
        "data/cache/metasyn/contextual-frontier-recovery-v2/00-prepared.json",
    )
    return freeze_metasyn_contextual_frontier_recovery_plan_v3(repository_root=ROOT)


@pytest.mark.private_cache
def test_reuses_exact_provider_accepted_v1_array_grammar_byte_for_byte(
    plan: Any, v1_plan: MetaSynContextualFrontierPlanV1
) -> None:
    accepted = v1_plan.roster[0].request
    assert plan.request.compiled_schema == accepted.compiled_schema
    assert canonical_json_bytes(
        plan.request.compiled_schema.original_schema
    ) == canonical_json_bytes(accepted.compiled_schema.original_schema)
    assert plan.request.original_schema_sha256 == ACCEPTED_ORIGINAL_SCHEMA_SHA256
    assert plan.request.compiled_schema_sha256 == ACCEPTED_COMPILED_SCHEMA_SHA256
    assert plan.request.wire_schema_sha256 == ACCEPTED_WIRE_SCHEMA_SHA256
    claims_schema = plan.request.compiled_schema.original_schema["oneOf"][0]["properties"]["claims"]
    assert claims_schema["type"] == "array"
    assert claims_schema["items"]["properties"]["field_path"]["enum"]


@pytest.mark.private_cache
def test_fresh_prompt_fixes_estimand_and_hides_four_numeric_targets(
    plan: Any, v1_plan: MetaSynContextualFrontierPlanV1, v2_plan: Any
) -> None:
    prompt = plan.request.prompt
    assert "TARGET ESTIMAND: fedratinib 500-mg group versus placebo group" in prompt
    assert "Do not select the 400-mg arm" in prompt
    assert "exactly the fifteen" in prompt
    prefix, passage_json = prompt.split("SOURCE_PASSAGES_JSON=", 1)
    roster_json = prefix.split("FIELD_ROSTER_JSON=", 1)[1].split("\n", 1)[0]
    roster = json.loads(roster_json)
    assert len(roster) == 15
    numeric = {
        "effect.control_events",
        "effect.control_total",
        "effect.treatment_events",
        "effect.treatment_total",
    }
    by_path = {item["field_path"]: item for item in roster}
    assert set(by_path) == {item.field_path for item in plan.target_spec.fields}
    assert all(
        by_path[field]["required_token"] == "EXTRACT_EXACT_UNSIGNED_INTEGER_FROM_SOURCE"
        for field in numeric
    )
    assert all(
        item.canonical_token is None
        for item in plan.target_spec.fields
        if item.field_path in numeric
    )
    assert len(json.loads(passage_json)) == 16
    assert plan.request.request_sha256 not in {item.request_sha256 for item in v1_plan.roster}
    assert plan.request.request_sha256 != v2_plan.request.transport_request_sha256
    assert plan.predecessor_requests_retried == 0


def test_request_builder_has_no_evaluator_or_numeric_answer_parameter() -> None:
    parameters = inspect.signature(
        freeze_metasyn_contextual_frontier_recovery_request_v3
    ).parameters
    assert set(parameters) == {
        "provider_context",
        "target_spec",
        "accepted_v1_request",
        "transport_config",
    }
    assert all("evaluator" not in name and "answer" not in name for name in parameters)


@pytest.mark.private_cache
def test_local_adapter_sorts_then_requires_exact_roster_and_semantics(
    plan: Any, v2_plan: MetaSynContextualFrontierRecoveryPlanV2
) -> None:
    raw = v2_plan.evaluator_fixture.model_outcome.model_dump(mode="json")
    raw["claims"] = list(reversed(raw["claims"]))
    evaluation = evaluate_metasyn_contextual_frontier_recovery_response_v3(
        repository_root=ROOT,
        plan=plan,
        raw_response=raw,
        provider_execution_binding_sha256="a" * 64,
    )
    assert evaluation.status == "typed_graph_mechanics_completed"
    assert evaluation.numeric_extraction_fields_evaluated == 4
    assert evaluation.plan_sha256 == plan.plan_sha256
    assert evaluation.runtime_pipeline_sha256 == plan.runtime_pipeline_sha256
    assert evaluation.v2_evaluator_plan_sha256 == v2_plan.plan_sha256
    assert evaluation.v2_evaluator_dependency_sha256 != evaluation.evaluation_sha256
    assert evaluation.native_projection is not None
    assert evaluation.native_projection.runtime_pipeline_sha256 == plan.runtime_pipeline_sha256
    assert evaluation.native_projection.fragment is not None
    assert (
        evaluation.native_projection.fragment.pipeline_fingerprint_sha256
        == plan.runtime_pipeline_sha256
    )
    assert not evaluation.extraction_accuracy_authority
    assert not evaluation.claim_release_authority

    wrong = json.loads(json.dumps(raw))
    treatment = next(
        item for item in wrong["claims"] if item["field_path"] == "treatment_arm.label"
    )
    treatment["token"] = "400-mg"
    with pytest.raises(
        MetaSynContextualFrontierRecoveryV3Error,
        match="recovery_v3_canonical_token_mismatch",
    ):
        evaluate_metasyn_contextual_frontier_recovery_response_v3(
            repository_root=ROOT,
            plan=plan,
            raw_response=wrong,
            provider_execution_binding_sha256="b" * 64,
        )

    duplicate = json.loads(json.dumps(raw))
    duplicate["claims"][0] = duplicate["claims"][1]
    with pytest.raises(
        MetaSynContextualFrontierRecoveryV3Error,
        match="recovery_v3_exact_field_set_mismatch",
    ):
        evaluate_metasyn_contextual_frontier_recovery_response_v3(
            repository_root=ROOT,
            plan=plan,
            raw_response=duplicate,
            provider_execution_binding_sha256="c" * 64,
        )


class _FakeClient:
    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.calls = 0

    def generate(self, request: Any) -> MetaSynContextualFrontierProviderResultV1:
        self.calls += 1
        return freeze_metasyn_contextual_frontier_provider_result_v1(
            request=request,
            outcome="completed",
            response_id="msg_recovery_v3_test",
            response_model="claude-fable-5",
            stop_reason="end_turn",
            text=json.dumps(self.raw, sort_keys=True),
            parsed_json=self.raw,
            usage=MetaSynContextualFrontierUsageV1(input_tokens=100, output_tokens=100),
        )


@pytest.mark.private_cache
def test_mock_execution_is_exactly_once_and_replays_terminal(
    tmp_path: Path, v2_plan: MetaSynContextualFrontierRecoveryPlanV2
) -> None:
    require_private_cache(
        "data/cache/metasyn/contextual-frontier-runtime-v1/00-prepared.json",
        "data/cache/metasyn/contextual-frontier-recovery-v2/00-prepared.json",
    )
    workspace = tmp_path / "recovery-v3"
    plan = prepare_metasyn_contextual_frontier_recovery_v3(
        repository_root=ROOT, workspace=workspace
    )
    authorize_metasyn_contextual_frontier_recovery_v3(
        workspace=workspace,
        phase_budget_usd_micros=plan.hard_cost_liability_usd_micros,
    )
    client = _FakeClient(v2_plan.evaluator_fixture.model_outcome.model_dump(mode="json"))
    first = execute_metasyn_contextual_frontier_recovery_v3(
        repository_root=ROOT, workspace=workspace, client=client
    )
    second = execute_metasyn_contextual_frontier_recovery_v3(
        repository_root=ROOT, workspace=workspace, client=client
    )
    assert first == second
    assert first.status == "typed_graph_mechanics_completed"
    assert first.provider_attempt_count_upper_bound == 1
    assert first.exact_request_retries_permitted == 0
    assert first.predecessor_requests_retried == 0
    assert client.calls == 1
    assert (workspace / "intent.json").is_file()
    assert (workspace / "provider-receipt.json").is_file()
    assert (workspace / "02-terminal.json").is_file()
