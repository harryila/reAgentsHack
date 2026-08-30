from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.private_cache_support import require_private_cache

from literature_multiverse.metasyn_contextual_frontier_recovery_v2 import (
    MetaSynContextualFrontierRecoveryPlanV2,
)
from literature_multiverse.metasyn_contextual_frontier_recovery_v4 import (
    EXPECTED_V3_TERMINAL_FILE_SHA256,
    EXPECTED_V3_TERMINAL_SHA256,
    authorize_metasyn_contextual_frontier_recovery_v4,
    execute_metasyn_contextual_frontier_recovery_v4,
    freeze_metasyn_contextual_frontier_recovery_plan_v4,
    prepare_metasyn_contextual_frontier_recovery_v4,
)
from literature_multiverse.metasyn_contextual_frontier_runtime_v1 import (
    MetaSynContextualFrontierProviderResultV1,
    MetaSynContextualFrontierUsageV1,
    freeze_metasyn_contextual_frontier_provider_result_v1,
)

ROOT = Path(__file__).resolve().parents[1]

# freeze_metasyn_contextual_frontier_recovery_plan_v4 replays the immutable v1
# roster, the v2 recovery plan, and the v3 recovery plan/terminal to bind the
# recovery-v4 request; every test in this module reaches that call path.
_V4_REPLAY_PATHS = (
    "data/cache/metasyn/contextual-frontier-runtime-v1/00-prepared.json",
    "data/cache/metasyn/contextual-frontier-recovery-v2/00-prepared.json",
    "data/cache/metasyn/contextual-frontier-recovery-v3/00-prepared.json",
    "data/cache/metasyn/contextual-frontier-recovery-v3/02-terminal.json",
)

pytestmark = pytest.mark.private_cache


def _v2() -> MetaSynContextualFrontierRecoveryPlanV2:
    raw = json.loads(
        (ROOT / "data/cache/metasyn/contextual-frontier-recovery-v2/00-prepared.json").read_text(
            encoding="utf-8"
        )
    )
    return MetaSynContextualFrontierRecoveryPlanV2.model_validate(raw["plan"])


def test_plan_is_completed_only_compiled_shared_array_and_binds_v3_failure() -> None:
    require_private_cache(*_V4_REPLAY_PATHS)
    plan = freeze_metasyn_contextual_frontier_recovery_plan_v4(repository_root=ROOT)
    schema = plan.request.compiled_schema.original_schema
    assert not any(key in schema for key in ("oneOf", "anyOf", "allOf"))
    assert schema["properties"]["packet_status"]["const"] == "completed"
    assert "reason" not in schema["properties"]
    assert schema["properties"]["claims"]["type"] == "array"
    assert schema["properties"]["claims"]["items"]["type"] == "object"
    assert plan.wire_schema_union_keywords == 0
    assert plan.wire_schema_utf8_bytes == 3654
    assert plan.wire_schema_property_slots == 17
    assert plan.wire_schema_enum_values == 41
    assert plan.compiler_confirmed
    assert plan.immutable_v3_terminal_sha256 == EXPECTED_V3_TERMINAL_SHA256
    assert plan.immutable_v3_terminal_file_sha256 == EXPECTED_V3_TERMINAL_FILE_SHA256
    assert plan.predecessor_requests_retried == 0
    assert not plan.claim_release_authority


class _FakeClient:
    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.calls = 0

    def generate(self, request: Any) -> MetaSynContextualFrontierProviderResultV1:
        self.calls += 1
        return freeze_metasyn_contextual_frontier_provider_result_v1(
            request=request,
            outcome="completed",
            response_id="msg_recovery_v4_test",
            response_model="claude-fable-5",
            stop_reason="end_turn",
            text=json.dumps(self.raw, sort_keys=True),
            parsed_json=self.raw,
            usage=MetaSynContextualFrontierUsageV1(input_tokens=100, output_tokens=100),
        )


def test_mock_exact_once_rebinds_evaluation_and_projection_to_v4(
    tmp_path: Path,
) -> None:
    require_private_cache(*_V4_REPLAY_PATHS)
    v2 = _v2()
    workspace = tmp_path / "recovery-v4"
    plan = prepare_metasyn_contextual_frontier_recovery_v4(
        repository_root=ROOT, workspace=workspace
    )
    authorize_metasyn_contextual_frontier_recovery_v4(
        workspace=workspace,
        phase_budget_usd_micros=plan.hard_cost_liability_usd_micros,
    )
    client = _FakeClient(v2.evaluator_fixture.model_outcome.model_dump(mode="json"))
    first = execute_metasyn_contextual_frontier_recovery_v4(
        repository_root=ROOT, workspace=workspace, client=client
    )
    second = execute_metasyn_contextual_frontier_recovery_v4(
        repository_root=ROOT, workspace=workspace, client=client
    )
    assert first == second
    assert client.calls == 1
    assert first.status == "typed_graph_mechanics_completed"
    assert first.evaluation is not None
    assert first.evaluation.plan_sha256 == plan.plan_sha256
    assert first.evaluation.runtime_pipeline_sha256 == plan.runtime_pipeline_sha256
    assert first.evaluation.native_projection.runtime_pipeline_sha256 == (
        plan.runtime_pipeline_sha256
    )
    assert first.evaluation.native_projection.fragment is not None
    assert (
        first.evaluation.native_projection.fragment.pipeline_fingerprint_sha256
        == plan.runtime_pipeline_sha256
    )
    assert first.exact_request_retries_permitted == 0
    assert first.predecessor_requests_retried == 0
    assert not first.claim_release_authority
