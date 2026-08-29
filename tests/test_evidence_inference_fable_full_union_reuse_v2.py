from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import scripts.run_evidence_inference_fable_full_union_reuse_v2 as harness

from literature_multiverse.evidence_inference_fable_full_reuse_v1 import (
    EvidenceInferenceFableReuseSourceV1,
)
from literature_multiverse.evidence_inference_fable_full_union_reuse_v2 import (
    EvidenceInferenceFableUnionSourceV2,
    execute_evidence_inference_fable_full_union_v2,
    freeze_evidence_inference_fable_full_union_plan_v2,
    freeze_evidence_inference_fable_full_union_scoring_lineage_v2,
    prepare_evidence_inference_fable_full_union_v2,
    require_evidence_inference_fable_full_union_scoring_v2,
    validate_evidence_inference_fable_full_union_v2,
)
from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    EvidenceInferenceFableBudgetAuthorizationV2,
    EvidenceInferenceFablePreparedRuntimeV1,
    EvidenceInferenceFableProviderResultV1,
    authorize_evidence_inference_fable_workspace_v1,
    freeze_evidence_inference_fable_budget_authorization_v2,
    largest_certified_pair_liability_usd_micros_v1,
    prepare_evidence_inference_fable_workspace_v1,
)
from literature_multiverse.evidence_inference_fable_retrospective_scoring_v1 import (
    replay_terminal_scoring_receipts_v1,
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
PILOT_PLAN_PATH = (
    ROOT
    / "artifacts/diagnostics/evidence-inference/fable-retrospective-pilot30-plan-v1.json"
)
RECOVERY_PLAN_PATH = (
    ROOT
    / "artifacts/diagnostics/evidence-inference/"
    "fable-retrospective-pilot30-recovery-v2-plan-v1.json"
)
FULL_V2_WORKSPACE = (
    ROOT / "data/cache/evidence-inference-fable-retrospective-full-live-v2"
)
PILOT_WORKSPACE = (
    ROOT / "data/cache/evidence-inference-fable-retrospective-pilot-live-v1"
)
RECOVERY_WORKSPACE = (
    ROOT
    / "data/cache/evidence-inference-fable-retrospective-pilot-recovery-v2-live"
)
FULL_COUNT_TERMINAL = (
    ROOT
    / "data/cache/evidence-inference-fable-retrospective-full-token-count-live-v1/terminal.json"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _context() -> tuple[
    EvidenceInferenceFableRetrospectivePlanV1,
    EvidenceInferenceFablePreparedRuntimeV1,
    EvidenceInferenceFableBudgetAuthorizationV2,
    list[EvidenceInferenceFableUnionSourceV2],
]:
    full_plan = EvidenceInferenceFableRetrospectivePlanV1.model_validate(
        _read(FULL_PLAN_PATH)
    )
    prepared = EvidenceInferenceFablePreparedRuntimeV1.model_validate(
        _read(FULL_V2_WORKSPACE / "00-prepared.json")
    )
    authorization = freeze_evidence_inference_fable_budget_authorization_v2(
        prepared=prepared,
        configured_total_budget_usd_micros=99_000_000,
        certified_count_terminal=_read(FULL_COUNT_TERMINAL),
    )
    pilot_plan = EvidenceInferenceFableRetrospectivePlanV1.model_validate(
        _read(PILOT_PLAN_PATH)
    )
    recovery_plan = EvidenceInferenceFableRetrospectivePlanV1.model_validate(
        _read(RECOVERY_PLAN_PATH)
    )
    pilot_reuse_source = EvidenceInferenceFableReuseSourceV1(
        "poisoned_pilot_v1", pilot_plan, PILOT_WORKSPACE
    )
    recovery_reuse_source = EvidenceInferenceFableReuseSourceV1(
        "recovery_pilot_v2", recovery_plan, RECOVERY_WORKSPACE
    )
    sources = [
        EvidenceInferenceFableUnionSourceV2(
            "poisoned_full_v2",
            full_plan,
            FULL_V2_WORKSPACE,
            (pilot_reuse_source, recovery_reuse_source),
        ),
        EvidenceInferenceFableUnionSourceV2(
            "poisoned_pilot_v1", pilot_plan, PILOT_WORKSPACE
        ),
        EvidenceInferenceFableUnionSourceV2(
            "recovery_pilot_v2", recovery_plan, RECOVERY_WORKSPACE
        ),
    ]
    return full_plan, prepared, authorization, sources


def _source_file_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for workspace in (FULL_V2_WORKSPACE, PILOT_WORKSPACE, RECOVERY_WORKSPACE):
        for path in sorted(candidate for candidate in workspace.rglob("*") if candidate.is_file()):
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def test_real_priority_union_is_exactly_22_plus_2_plus_358_and_read_only() -> None:
    full_plan, prepared, authorization, sources = _context()
    source_before = _source_file_hashes()

    union = freeze_evidence_inference_fable_full_union_plan_v2(
        full_plan=full_plan,
        full_prepared=prepared,
        full_authorization=authorization,
        sources=sources,
    )

    assert len(union.entries) == 24
    assert union.adopted_terminal_receipt_count == 22
    assert union.inherited_ambiguous_failure_count == 2
    assert union.maximum_new_provider_attempt_count == 358
    assert len(union.entries) + union.maximum_new_provider_attempt_count == 382
    assert union.configured_total_budget_usd_micros == 99_000_000
    assert authorization.configured_total_budget_usd_micros == 99_000_000
    assert (
        sum(authorization.certified_request_liabilities_usd_micros.values())
        == 234_938_730
    )
    assert (
        largest_certified_pair_liability_usd_micros_v1(
            prepared=prepared,
            certified_request_liabilities_usd_micros=(
                authorization.certified_request_liabilities_usd_micros
            ),
        )
        == 2_647_910
    )
    assert union.source_priority == [
        "poisoned_full_v2",
        "poisoned_pilot_v1",
        "recovery_pilot_v2",
    ]
    assert union.source_workspaces_immutable
    assert union.shadowed_lower_priority_candidate_count == 8
    assert union.transitively_reused_nested_record_count == 8
    assert union.provider_calls_made_while_planning == 0
    assert union.labels_opened is False

    ambiguities = [
        entry
        for entry in union.entries
        if entry.adoption_kind == "inherited_ambiguous_failure"
    ]
    assert len(ambiguities) == 2
    assert {entry.source_incident_kind for entry in ambiguities} == {
        "provider_call_raised_after_durable_intent",
        "provider_result_invalid_after_return",
    }
    assert all(entry.source_retry_permitted is False for entry in ambiguities)
    assert all(
        entry.target_provider_attempts_permitted_for_entry == 0
        for entry in ambiguities
    )
    assert union.inherited_ambiguity_retry_permitted is False
    assert _source_file_hashes() == source_before


def test_cli_defaults_target_fresh_v4_and_full_v2_as_first_source() -> None:
    args = harness._parser().parse_args(
        [
            "prepare",
            "--expected-full-plan-sha256",
            "0" * 64,
            "--expected-authorization-sha256",
            "1" * 64,
        ]
    )

    assert args.workspace == harness.DEFAULT_TARGET_WORKSPACE
    assert args.workspace.name.endswith("full-live-v4")
    assert args.full_v2_source_workspace == harness.DEFAULT_FULL_V2_SOURCE_WORKSPACE


def test_offline_completed_union_replays_and_binds_scoring_lineage(
    tmp_path: Path,
) -> None:
    full_plan, prepared, authorization, sources = _context()
    source_before = _source_file_hashes()
    workspace = tmp_path / "full-union-runtime"
    prepare_evidence_inference_fable_workspace_v1(
        workspace=workspace, prepared=prepared
    )
    authorize_evidence_inference_fable_workspace_v1(
        workspace=workspace, authorization=authorization
    )
    union_plan = freeze_evidence_inference_fable_full_union_plan_v2(
        full_plan=full_plan,
        full_prepared=prepared,
        full_authorization=authorization,
        sources=sources,
    )
    prepare_evidence_inference_fable_full_union_v2(
        workspace=workspace, union_plan=union_plan
    )

    class OfflineClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, surface: Any) -> EvidenceInferenceFableProviderResultV1:
            self.calls.append(surface.request_key)
            payload = {
                "result_version": "evidence-inference-fable-provider-result-v1",
                "request_key": surface.request_key,
                "surface_sha256": surface.surface_sha256,
                "transport_attempt_count": 1,
                "sdk_retry_count": 0,
                "outcome": "completed",
                "response_id": f"offline-{len(self.calls)}",
                "response_model": "claude-fable-5",
                "parsed_json": {},
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

    client = OfflineClient()
    terminal = execute_evidence_inference_fable_full_union_v2(
        workspace=workspace,
        full_plan=full_plan,
        sources=sources,
        delegate=client,
    )
    assert terminal.target_runtime_status == "completed"
    assert terminal.full_population_score_permitted
    assert terminal.realized_adopted_terminal_receipt_count == 22
    assert terminal.realized_inherited_ambiguous_failure_count == 2
    assert terminal.new_provider_attempt_count == len(client.calls) == 358
    assert terminal.target_provider_attempts_for_adopted_entries == 0
    assert terminal.inherited_ambiguous_attempts_retried == 0
    assert terminal == validate_evidence_inference_fable_full_union_v2(
        workspace=workspace, full_plan=full_plan, sources=sources
    )
    assert terminal == require_evidence_inference_fable_full_union_scoring_v2(
        workspace=workspace, full_plan=full_plan, sources=sources
    )
    scoring_terminal, scoring_receipts = replay_terminal_scoring_receipts_v1(
        plan=full_plan, runtime_workspace=workspace
    )
    assert scoring_terminal.terminal_sha256 == terminal.target_runtime_terminal_sha256
    assert len(scoring_receipts) == 382
    assert (
        sum(
            receipt.accounted_cost_basis
            == "certified_provider_token_plus_headroom_liability_unknown_usage"
            for receipt in scoring_receipts
        )
        == 2
    )

    lineage = freeze_evidence_inference_fable_full_union_scoring_lineage_v2(
        union_terminal=terminal,
        completion_certificate_sha256="a" * 64,
        private_report_sha256="b" * 64,
        public_summary_sha256="c" * 64,
    )
    assert lineage.union_terminal_sha256 == terminal.terminal_sha256
    assert (
        lineage.target_runtime_terminal_sha256
        == terminal.target_runtime_terminal_sha256
    )
    assert lineage.lineage_sha256 == hash_canonical(
        lineage.model_dump(mode="json", exclude={"lineage_sha256"})
    )
    assert _source_file_hashes() == source_before


def test_live_cli_stops_before_paths_environment_or_provider_without_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_context(_args: object) -> Any:
        raise AssertionError("paths opened before explicit live flag")

    def forbidden_environment(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("environment opened before explicit live flag")

    def forbidden_provider(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("provider constructed before explicit live flag")

    monkeypatch.setattr(harness, "_context", forbidden_context)
    monkeypatch.setattr(harness, "load_live_environment", forbidden_environment)
    monkeypatch.setattr(
        harness.AnthropicFablePairedClientV1,
        "from_anthropic_sdk",
        forbidden_provider,
    )
    with pytest.raises(
        harness.EvidenceInferenceFableFullUnionHarnessError,
        match="live_flag_required",
    ):
        harness.main(
            [
                "run",
                "--expected-full-plan-sha256",
                "0" * 64,
                "--expected-authorization-sha256",
                "1" * 64,
                "--expected-union-plan-sha256",
                "2" * 64,
            ]
        )
