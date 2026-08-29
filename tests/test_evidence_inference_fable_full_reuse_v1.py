from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from literature_multiverse.evidence_inference_fable_full_reuse_v1 import (
    EvidenceInferenceFableReuseSourceV1,
    execute_evidence_inference_fable_full_reuse_v1,
    freeze_evidence_inference_fable_full_reuse_plan_v1,
    prepare_evidence_inference_fable_full_reuse_v1,
    require_evidence_inference_fable_full_reuse_scoring_v1,
    validate_evidence_inference_fable_full_reuse_v1,
)
from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    EvidenceInferenceFableBudgetAuthorizationV1,
    EvidenceInferenceFableCallSurfaceV1,
    EvidenceInferenceFablePreparedRuntimeV1,
    EvidenceInferenceFableProviderResultV1,
    authorize_evidence_inference_fable_workspace_v1,
    prepare_evidence_inference_fable_workspace_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_v1 import (
    EvidenceInferenceFableRetrospectivePlanV1,
)
from literature_multiverse.lineage import hash_canonical

ROOT = Path(__file__).resolve().parents[1]
FULL_PLAN_PATH = (
    ROOT
    / "artifacts/diagnostics/evidence-inference/fable-retrospective-full-plan-v1.json"
)
FULL_WORKSPACE = ROOT / "data/cache/evidence-inference-fable-retrospective-full-live-v1"
SOURCE_PLAN_PATH = (
    ROOT / "artifacts/diagnostics/evidence-inference/fable-retrospective-pilot30-plan-v1.json"
)
SOURCE_WORKSPACE = ROOT / "data/cache/evidence-inference-fable-retrospective-pilot-live-v1"
RECOVERY_PLAN_PATH = (
    ROOT
    / "artifacts/diagnostics/evidence-inference/"
    "fable-retrospective-pilot30-recovery-v2-plan-v1.json"
)
RECOVERY_WORKSPACE = (
    ROOT / "data/cache/evidence-inference-fable-retrospective-pilot-recovery-v2-live"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _context() -> tuple[
    EvidenceInferenceFableRetrospectivePlanV1,
    EvidenceInferenceFablePreparedRuntimeV1,
    EvidenceInferenceFableBudgetAuthorizationV1,
    list[EvidenceInferenceFableReuseSourceV1],
]:
    full_plan = EvidenceInferenceFableRetrospectivePlanV1.model_validate(
        _read(FULL_PLAN_PATH)
    )
    prepared = EvidenceInferenceFablePreparedRuntimeV1.model_validate(
        _read(FULL_WORKSPACE / "00-prepared.json")
    )
    authorization = EvidenceInferenceFableBudgetAuthorizationV1.model_validate(
        _read(FULL_WORKSPACE / "01-authorization.json")
    )
    source_plan = EvidenceInferenceFableRetrospectivePlanV1.model_validate(
        _read(SOURCE_PLAN_PATH)
    )
    recovery_plan = EvidenceInferenceFableRetrospectivePlanV1.model_validate(
        _read(RECOVERY_PLAN_PATH)
    )
    sources = [
        EvidenceInferenceFableReuseSourceV1(
            "poisoned_pilot_v1", source_plan, SOURCE_WORKSPACE
        ),
        EvidenceInferenceFableReuseSourceV1(
            "recovery_pilot_v2", recovery_plan, RECOVERY_WORKSPACE
        ),
    ]
    return full_plan, prepared, authorization, sources


def _source_json_hashes() -> dict[str, str]:
    paths = sorted(SOURCE_WORKSPACE.rglob("*.json")) + sorted(
        RECOVERY_WORKSPACE.rglob("*.json")
    )
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


class _FailedButTerminalDelegate:
    def __init__(self) -> None:
        self.wire_calls: list[str] = []

    def generate(
        self, surface: EvidenceInferenceFableCallSurfaceV1
    ) -> EvidenceInferenceFableProviderResultV1:
        self.wire_calls.append(surface.wire_call_sha256)
        payload = {
            "result_version": "evidence-inference-fable-provider-result-v1",
            "request_key": surface.request_key,
            "surface_sha256": surface.surface_sha256,
            "transport_attempt_count": 1,
            "sdk_retry_count": 0,
            "outcome": "failed",
            "response_id": "offline-fake-response",
            "response_model": "claude-fable-5",
            "parsed_json": None,
            "input_tokens": 1,
            "output_tokens": 0,
            "reported_cost_usd_micros": 10,
            "charged_cost_usd_micros": 10,
            "cost_basis": "reported_usage",
            "response_text_sha256": None,
            "failure_code": "response_content_invalid",
        }
        return EvidenceInferenceFableProviderResultV1.model_validate(
            {**payload, "result_sha256": hash_canonical(payload)}
        )


def test_actual_reuse_plan_is_exactly_20_plus_1_plus_361() -> None:
    full_plan, prepared, authorization, sources = _context()
    plan = freeze_evidence_inference_fable_full_reuse_plan_v1(
        full_plan=full_plan,
        full_prepared=prepared,
        full_authorization=authorization,
        sources=sources,
    )

    assert len(plan.entries) == 21
    assert plan.adopted_terminal_receipt_count == 20
    assert plan.inherited_ambiguous_failure_count == 1
    assert plan.maximum_new_provider_attempt_count == 361
    assert sum(
        entry.locked_question_count
        for entry in plan.entries
        if entry.adoption_kind == "inherited_ambiguous_failure"
    ) == 15


def test_reuse_wrapper_makes_only_361_delegate_calls_and_replays(
    tmp_path: Path,
) -> None:
    full_plan, prepared, authorization, sources = _context()
    adoption_plan = freeze_evidence_inference_fable_full_reuse_plan_v1(
        full_plan=full_plan,
        full_prepared=prepared,
        full_authorization=authorization,
        sources=sources,
    )
    source_before = _source_json_hashes()
    workspace = tmp_path / "full-reuse"
    prepare_evidence_inference_fable_workspace_v1(
        workspace=workspace, prepared=prepared
    )
    authorize_evidence_inference_fable_workspace_v1(
        workspace=workspace, authorization=authorization
    )
    prepare_evidence_inference_fable_full_reuse_v1(
        workspace=workspace, adoption_plan=adoption_plan
    )
    delegate = _FailedButTerminalDelegate()

    terminal = execute_evidence_inference_fable_full_reuse_v1(
        workspace=workspace,
        full_plan=full_plan,
        sources=sources,
        delegate=delegate,
    )

    assert terminal.full_population_score_permitted
    assert terminal.realized_adopted_terminal_receipt_count == 20
    assert terminal.realized_inherited_ambiguous_failure_count == 1
    assert terminal.new_provider_attempt_count == 361
    assert len(delegate.wire_calls) == 361
    assert len(set(delegate.wire_calls)) == 361
    assert not {
        entry.wire_call_sha256 for entry in adoption_plan.entries
    }.intersection(delegate.wire_calls)
    assert _source_json_hashes() == source_before

    replay = validate_evidence_inference_fable_full_reuse_v1(
        workspace=workspace, full_plan=full_plan, sources=sources
    )
    scoring_gate = require_evidence_inference_fable_full_reuse_scoring_v1(
        workspace=workspace, full_plan=full_plan, sources=sources
    )
    assert replay == terminal == scoring_gate

    replay_delegate = _FailedButTerminalDelegate()
    assert (
        execute_evidence_inference_fable_full_reuse_v1(
            workspace=workspace,
            full_plan=full_plan,
            sources=sources,
            delegate=replay_delegate,
        )
        == terminal
    )
    assert replay_delegate.wire_calls == []
