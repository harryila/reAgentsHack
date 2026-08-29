from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from literature_multiverse.evidence_inference_fable_continuation_v1 import (
    EvidenceInferenceFableContinuationError,
    authorize_evidence_inference_fable_continuation_workspace_v1,
    execute_evidence_inference_fable_continuation_v1,
    freeze_evidence_inference_fable_continuation_authorization_v1,
    freeze_evidence_inference_fable_continuation_plan_v1,
    freeze_evidence_inference_fable_descriptive_composite_v1,
    prepare_evidence_inference_fable_continuation_workspace_v1,
    validate_evidence_inference_fable_continuation_workspace_v1,
)
from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    EvidenceInferenceFableIncidentV1,
    EvidenceInferenceFableIntentV1,
    EvidenceInferenceFableProviderResultV1,
    EvidenceInferenceFableReceiptV1,
    EvidenceInferenceFableTerminalV1,
    authorize_evidence_inference_fable_workspace_v1,
    freeze_evidence_inference_fable_budget_authorization_v1,
    freeze_evidence_inference_fable_call_surface_v1,
    freeze_evidence_inference_fable_prepared_runtime_v1,
    prepare_evidence_inference_fable_workspace_v1,
    validate_evidence_inference_fable_workspace_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_v1 import (
    ArticleBatchRequestV1,
    EvidenceInferenceFableRetrospectivePlanV1,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def _fixture() -> tuple[Any, Any]:
    items = []
    surfaces = []
    system = "Return JSON only."
    for index, (article, arm) in enumerate(
        [("PMC1", "seed"), ("PMC1", "winner"), ("PMC2", "winner"), ("PMC2", "seed")]
    ):
        prompt = f"article {article} arm {arm}"
        request_sha = hashlib.sha256(f"request-{index}".encode()).hexdigest()
        cost = SimpleNamespace(full_context_hard_liability_usd_micros=100)
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
                roster_item=item,
                system=system,
                prompt=prompt,
                wire_schema=SCHEMA,
            )
        )
    plan = EvidenceInferenceFableRetrospectivePlanV1.model_construct(
        plan_sha256="a" * 64,
        request_roster_sha256="b" * 64,
        request_count=4,
        roster=items,
    )
    prepared = freeze_evidence_inference_fable_prepared_runtime_v1(
        plan=plan, surfaces=surfaces
    )
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


def _intent(*, prepared: Any, authorization: Any, index: int, spend: int) -> Any:
    surface = prepared.surfaces[index]
    payload = {
        "intent_version": "evidence-inference-fable-intent-v1",
        "prepared_sha256": prepared.prepared_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "pair_index": index // 2,
        "request_key": surface.request_key,
        "surface": surface,
        "cumulative_reported_spend_before_pair_usd_micros": spend,
        "pair_hard_liability_usd_micros": 200,
        "permitted_provider_attempts": 1,
        "application_retries_permitted": 0,
        "sdk_retries_permitted": 0,
        "orphan_or_ambiguous_attempt_is_terminal": True,
    }
    return EvidenceInferenceFableIntentV1.model_validate(
        {**payload, "intent_sha256": hash_canonical(payload)}
    )


def _poisoned_source(tmp_path: Path) -> tuple[Any, Path]:
    plan, prepared = _fixture()
    authorization = freeze_evidence_inference_fable_budget_authorization_v1(
        prepared=prepared,
        configured_total_budget_usd_micros=1000,
    )
    workspace = tmp_path / "source"
    prepare_evidence_inference_fable_workspace_v1(
        workspace=workspace, prepared=prepared
    )
    authorize_evidence_inference_fable_workspace_v1(
        workspace=workspace, authorization=authorization
    )
    for name in ("intents", "receipts", "incidents"):
        (workspace / name).mkdir()
    spend = 0
    for index in range(2):
        intent = _intent(
            prepared=prepared,
            authorization=authorization,
            index=index,
            spend=spend,
        )
        result = _result(prepared.surfaces[index])
        receipt_payload = {
            "receipt_version": "evidence-inference-fable-receipt-v1",
            "intent_sha256": intent.intent_sha256,
            "request_key": prepared.surfaces[index].request_key,
            "provider_result": result,
            "locked_question_count": 1,
            "locked_questions_scored_incorrect": 0,
        }
        receipt = EvidenceInferenceFableReceiptV1.model_validate(
            {**receipt_payload, "receipt_sha256": hash_canonical(receipt_payload)}
        )
        atomic_write_json(
            workspace / "intents" / f"request-{index}.json", intent
        )
        atomic_write_json(
            workspace / "receipts" / f"request-{index}.json", receipt
        )
        spend += 60
    poison_intent = _intent(
        prepared=prepared,
        authorization=authorization,
        index=2,
        spend=spend,
    )
    incident_payload = {
        "incident_version": "evidence-inference-fable-incident-v1",
        "status": "terminal_ambiguous_attempt_poison",
        "kind": "provider_call_raised_after_durable_intent",
        "intent_sha256": poison_intent.intent_sha256,
        "request_key": "request-2",
        "charged_cost_usd_micros": 100,
        "cost_basis": "unknown_usage_hard_liability",
        "retry_permitted": False,
    }
    incident = EvidenceInferenceFableIncidentV1.model_validate(
        {**incident_payload, "incident_sha256": hash_canonical(incident_payload)}
    )
    atomic_write_json(workspace / "intents" / "request-2.json", poison_intent)
    atomic_write_json(workspace / "incidents" / "request-2.json", incident)
    terminal_payload = {
        "terminal_version": "evidence-inference-fable-terminal-v1",
        "status": "terminal_ambiguous_attempt_poison",
        "prepared_sha256": prepared.prepared_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "completed_request_count": 2,
        "completed_pair_count": 1,
        "cumulative_reported_spend_usd_micros": 220,
        "cumulative_spend_semantics": "reported_usage_or_unknown_usage_hard_liability",
        "next_pair_index": 1,
        "full_population_score_permitted": False,
        "extraction_accuracy_authority": False,
        "confirmatory_authority": False,
        "synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    terminal = EvidenceInferenceFableTerminalV1.model_validate(
        {**terminal_payload, "terminal_sha256": hash_canonical(terminal_payload)}
    )
    atomic_write_json(workspace / "02-terminal.json", terminal)
    assert (
        validate_evidence_inference_fable_workspace_v1(
            workspace=workspace, plan=plan
        )
        == terminal
    )
    return plan, workspace


class FakeClient:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    def generate(self, surface: Any) -> EvidenceInferenceFableProviderResultV1:
        self.calls.append(surface.request_key)
        if self.fail:
            raise RuntimeError("transport state unknown")
        return _result(surface)


def _continuation(tmp_path: Path) -> tuple[Any, Path, Path]:
    plan, source = _poisoned_source(tmp_path)
    continuation_plan = freeze_evidence_inference_fable_continuation_plan_v1(
        source_workspace=source,
        retrospective_plan=plan,
    )
    workspace = tmp_path / "continuation"
    prepare_evidence_inference_fable_continuation_workspace_v1(
        workspace=workspace,
        continuation_plan=continuation_plan,
    )
    authorization = freeze_evidence_inference_fable_continuation_authorization_v1(
        continuation_plan=continuation_plan,
        configured_total_budget_usd_micros=500,
    )
    authorize_evidence_inference_fable_continuation_workspace_v1(
        workspace=workspace,
        authorization=authorization,
    )
    return plan, source, workspace


def test_continuation_attempts_only_requests_without_source_intent(tmp_path: Path) -> None:
    plan, source, workspace = _continuation(tmp_path)
    client = FakeClient()
    terminal = execute_evidence_inference_fable_continuation_v1(
        workspace=workspace,
        source_workspace=source,
        retrospective_plan=plan,
        client=client,
    )
    assert terminal.status == "completed"
    assert client.calls == ["request-3"]
    assert not (workspace / "intents" / "request-2.json").exists()
    assert terminal == validate_evidence_inference_fable_continuation_workspace_v1(
        workspace=workspace,
        source_workspace=source,
        retrospective_plan=plan,
    )


def test_composite_preserves_denominator_but_cannot_pass_mechanics_gate(
    tmp_path: Path,
) -> None:
    plan, source, workspace = _continuation(tmp_path)
    execute_evidence_inference_fable_continuation_v1(
        workspace=workspace,
        source_workspace=source,
        retrospective_plan=plan,
        client=FakeClient(),
    )
    composite = freeze_evidence_inference_fable_descriptive_composite_v1(
        source_workspace=source,
        continuation_workspace=workspace,
        retrospective_plan=plan,
    )
    assert [item.origin for item in composite.records] == [
        "source_receipt",
        "source_receipt",
        "source_ambiguous_intention_to_evaluate_failure",
        "continuation_receipt",
    ]
    failed = composite.records[2]
    assert failed.provider_outcome == "transport_failed_or_ambiguous"
    assert failed.forced_zero_structured_output_direction_and_grounding is True
    assert composite.descriptive_full_roster_scoring_permitted is True
    assert composite.intention_to_evaluate_denominator_preserved is True
    assert composite.clean_mechanics_gate_authority is False
    assert composite.mechanics_reliability_claim_permitted is False
    assert composite.inferential_effect_claim_permitted is False


def test_replay_never_resubmits_completed_continuation(tmp_path: Path) -> None:
    plan, source, workspace = _continuation(tmp_path)
    first_client = FakeClient()
    first = execute_evidence_inference_fable_continuation_v1(
        workspace=workspace,
        source_workspace=source,
        retrospective_plan=plan,
        client=first_client,
    )
    replay_client = FakeClient()
    second = execute_evidence_inference_fable_continuation_v1(
        workspace=workspace,
        source_workspace=source,
        retrospective_plan=plan,
        client=replay_client,
    )
    assert first == second
    assert first_client.calls == ["request-3"]
    assert replay_client.calls == []


def test_new_ambiguous_attempt_is_terminal_and_never_retried(tmp_path: Path) -> None:
    plan, source, workspace = _continuation(tmp_path)
    client = FakeClient(fail=True)
    first = execute_evidence_inference_fable_continuation_v1(
        workspace=workspace,
        source_workspace=source,
        retrospective_plan=plan,
        client=client,
    )
    second = execute_evidence_inference_fable_continuation_v1(
        workspace=workspace,
        source_workspace=source,
        retrospective_plan=plan,
        client=client,
    )
    assert first == second
    assert first.status == "terminal_ambiguous_attempt_poison"
    assert client.calls == ["request-3"]
    with pytest.raises(
        EvidenceInferenceFableContinuationError,
        match="composite_requires_completed_continuation",
    ):
        freeze_evidence_inference_fable_descriptive_composite_v1(
            source_workspace=source,
            continuation_workspace=workspace,
            retrospective_plan=plan,
        )


def test_source_semantic_artifacts_are_not_modified(tmp_path: Path) -> None:
    plan, source, workspace = _continuation(tmp_path)
    before = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*.json")
    }
    execute_evidence_inference_fable_continuation_v1(
        workspace=workspace,
        source_workspace=source,
        retrospective_plan=plan,
        client=FakeClient(),
    )
    freeze_evidence_inference_fable_descriptive_composite_v1(
        source_workspace=source,
        continuation_workspace=workspace,
        retrospective_plan=plan,
    )
    after = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*.json")
    }
    assert after == before


def test_authorization_requires_liability_for_all_never_attempted_requests(
    tmp_path: Path,
) -> None:
    plan, source = _poisoned_source(tmp_path)
    continuation = freeze_evidence_inference_fable_continuation_plan_v1(
        source_workspace=source,
        retrospective_plan=plan,
    )
    with pytest.raises(ValueError, match="budget_below_full_liability"):
        freeze_evidence_inference_fable_continuation_authorization_v1(
            continuation_plan=continuation,
            configured_total_budget_usd_micros=99,
        )
