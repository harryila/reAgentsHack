from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from literature_multiverse.contextual_numeric_grounding_v3 import (
    ContextualGroundingOfflineFeasibilitySuiteV3,
    project_contextual_grounded_outcome_v3,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_contextual_frontier_runtime_v1 import (
    MetaSynContextualFrontierTerminalReportV1,
    MetaSynContextualFrontierValidationResultV1,
)
from literature_multiverse.postlive_contextual_join_v1 import (
    PostLiveContextualCertificateV1,
    PostLiveContextualJoinV1Error,
    freeze_postlive_contextual_certificate_v1,
)

ROOT = Path(__file__).resolve().parents[1]
OFFLINE_SUITE = (
    ROOT / "artifacts/diagnostics/contextual-grounding-offline-feasibility-suite-v3.json"
)
NOW = datetime(2026, 8, 29, 8, 0, tzinfo=UTC)


def _terminal_fixture() -> MetaSynContextualFrontierTerminalReportV1:
    suite = ContextualGroundingOfflineFeasibilitySuiteV3.model_validate(
        json.loads(OFFLINE_SUITE.read_text(encoding="utf-8"))
    )
    fixture = next(
        item
        for item in suite.receipts
        if item.witness_id == "metasyn-row17-candidate3-binary-primary-endpoint"
    )
    runtime_pipeline_sha256 = "1" * 64
    provider_execution_binding_sha256 = "2" * 64
    (
        outcome,
        groundings,
        effect,
        grounding_core_sha256,
        runtime_grounding_binding_sha256,
        projection,
    ) = project_contextual_grounded_outcome_v3(
        fixture_receipt=fixture,
        raw_outcome=fixture.model_outcome,
        runtime_pipeline_sha256=runtime_pipeline_sha256,
        provider_execution_binding_sha256=provider_execution_binding_sha256,
    )
    validation_payload: dict[str, Any] = {
        "validation_version": "metasyn-contextual-frontier-validation-result-v1",
        "status": "typed_graph_mechanics_completed",
        "plan_sha256": "3" * 64,
        "runtime_pipeline_sha256": runtime_pipeline_sha256,
        "authorization_sha256": "4" * 64,
        "intent_sha256": "5" * 64,
        "request_key": "fixture-request",
        "request_sha256": "6" * 64,
        "witness_id": fixture.witness_id,
        "provider_binding_sha256": fixture.provider_binding_sha256,
        "provider_receipt_sha256": "7" * 64,
        "provider_result_sha256": "8" * 64,
        "provider_execution_binding_sha256": provider_execution_binding_sha256,
        "provider_outcome": "completed",
        "model_outcome": outcome,
        "model_outcome_sha256": hash_canonical(outcome.model_dump(mode="json")),
        "groundings": groundings,
        "grounding_membership_sha256": hash_canonical(
            [item.grounding_sha256 for item in groundings]
        ),
        "grounded_effect": effect,
        "grounded_effect_sha256": effect.effect_sha256,
        "contextual_grounding_core_sha256": grounding_core_sha256,
        "runtime_grounding_binding_sha256": runtime_grounding_binding_sha256,
        "native_projection": projection,
        "native_projection_sha256": projection.projection_sha256,
        "fresh_native_typed_graph_completed": True,
        "failure_code": None,
        "credential_archived": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_effectiveness_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    validation = MetaSynContextualFrontierValidationResultV1.model_validate(
        {
            **validation_payload,
            "validation_sha256": hash_canonical(validation_payload),
        }
    )
    terminal_payload: dict[str, Any] = {
        "terminal_version": "metasyn-contextual-frontier-terminal-report-v1",
        "terminal": True,
        "status": "typed_graph_smoke_completed",
        "plan_sha256": "3" * 64,
        "runtime_pipeline_sha256": runtime_pipeline_sha256,
        "authorization_sha256": "4" * 64,
        "attempted_request_keys": ["fixture-request"],
        "unattempted_request_keys": ["fallback-request"],
        "unattempted_reason": "first_fully_grounded_native_typed_graph_stopped_roster",
        "validation_results": [validation],
        "validation_membership_sha256": hash_canonical([validation.validation_sha256]),
        "ambiguity_incident": None,
        "successful_request_key": "fixture-request",
        "provider_attempt_count_upper_bound": 1,
        "provider_receipt_count": 1,
        "first_success_stopped_fallback": True,
        "credential_archived": False,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_effectiveness_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return MetaSynContextualFrontierTerminalReportV1.model_validate(
        {**terminal_payload, "report_sha256": hash_canonical(terminal_payload)}
    )


def _certificate(**kwargs: Any) -> PostLiveContextualCertificateV1:
    return freeze_postlive_contextual_certificate_v1(
        terminal_report=_terminal_fixture(),
        runtime_workspace_validation_sha256="9" * 64,
        generated_at=NOW,
        target_direction="increase",
        **kwargs,
    )


def test_success_joins_real_graph_counterfactuals_without_authority() -> None:
    certificate = _certificate()

    assert certificate.status == "mechanics_completed_non_authorizing"
    assert certificate.synthesis == (certificate.audit_mechanics.sequential_state.synthesis)
    assert len(certificate.evidence_graph.publications) == 1
    assert len(certificate.evidence_graph.outcome_estimates) == 1
    assert certificate.condition_mechanics.status == "not_scientifically_defined"
    assert certificate.audit_mechanics.status == "scheduler_ready_no_audit_selected"
    assert len(certificate.audit_mechanics.audit_candidates) == 1
    assert len(certificate.audit_mechanics.priority_records) == 1
    assert (
        certificate.audit_mechanics.priority_records[0]["probability_influence"]
        == certificate.audit_mechanics.current_candidates[0].priority
    )
    assert certificate.audit_mechanics.sequential_state.transitions == []
    assert certificate.audit_mechanics.sequential_state.session.active_action is None
    assert not certificate.audit_mechanics.item_error_calibration_performed
    assert not certificate.audit_mechanics.human_cost_measurement_performed
    assert not certificate.release_authorizing
    assert not certificate.claim_release_authority
    assert "title_or_abstract_only_not_release_grade" in certificate.blockers
    assert "single_publication_mechanics_only" in certificate.blockers


def test_explicit_moderator_runs_but_cannot_confirm_condition() -> None:
    certificate = _certificate(prespecified_moderators=["dose"])

    assert certificate.condition_mechanics.analysis_executed
    assert certificate.condition_mechanics.status == "executed_insufficient"
    assert certificate.condition_mechanics.analysis is not None
    assert certificate.condition_mechanics.analysis["status"] == "insufficient"
    assert not certificate.condition_mechanics.held_out_confirmation_performed
    assert not certificate.condition_claim_authority


def test_terminal_nested_hash_tamper_fails_closed() -> None:
    terminal = _terminal_fixture().model_dump(mode="json")
    terminal["validation_results"][0]["provider_result_sha256"] = "a" * 64

    with pytest.raises(
        PostLiveContextualJoinV1Error,
        match="postlive_terminal_contract_or_hash_invalid",
    ):
        freeze_postlive_contextual_certificate_v1(
            terminal_report=terminal,
            runtime_workspace_validation_sha256="9" * 64,
            generated_at=NOW,
            target_direction="increase",
        )


def test_certificate_authority_tamper_fails_even_with_recomputed_hash() -> None:
    raw = _certificate().model_dump(mode="json")
    raw["release_authorizing"] = True
    raw["certificate_sha256"] = hash_canonical(
        {key: value for key, value in raw.items() if key != "certificate_sha256"}
    )

    with pytest.raises(ValidationError):
        PostLiveContextualCertificateV1.model_validate(raw)


def test_required_blocker_removal_fails_even_with_recomputed_hash() -> None:
    raw = _certificate().model_dump(mode="json")
    raw["blockers"].remove("title_or_abstract_only_not_release_grade")
    raw["certificate_sha256"] = hash_canonical(
        {key: value for key, value in raw.items() if key != "certificate_sha256"}
    )

    with pytest.raises(ValidationError, match="required_blocker_missing"):
        PostLiveContextualCertificateV1.model_validate(raw)


def test_nonsuccess_terminal_is_rejected() -> None:
    terminal = _terminal_fixture().model_dump(mode="json")
    terminal["status"] = "roster_exhausted_without_typed_graph"
    terminal["successful_request_key"] = None
    terminal["report_sha256"] = hash_canonical(
        {key: value for key, value in terminal.items() if key != "report_sha256"}
    )

    with pytest.raises(PostLiveContextualJoinV1Error):
        freeze_postlive_contextual_certificate_v1(
            terminal_report=terminal,
            runtime_workspace_validation_sha256="9" * 64,
            generated_at=NOW,
            target_direction="increase",
        )


def test_moderator_order_and_timezone_are_closed_contract_inputs() -> None:
    with pytest.raises(PostLiveContextualJoinV1Error, match="sorted_unique"):
        _certificate(prespecified_moderators=["z", "a"])

    with pytest.raises(ValueError, match="timezone_required"):
        freeze_postlive_contextual_certificate_v1(
            terminal_report=_terminal_fixture(),
            runtime_workspace_validation_sha256="9" * 64,
            generated_at=datetime(2026, 8, 29),
            target_direction="increase",
        )


def test_certificate_byte_level_tamper_breaks_self_hash() -> None:
    raw = deepcopy(_certificate().model_dump(mode="json"))
    raw["outcome_name"] = f"{raw['outcome_name']} altered"

    with pytest.raises(ValidationError, match="self_hash_mismatch"):
        PostLiveContextualCertificateV1.model_validate(raw)


def test_status_cli_keeps_non_authorizing_boundary(tmp_path: Path) -> None:
    certificate_path = tmp_path / "certificate.json"
    certificate_path.write_text(
        json.dumps(_certificate().model_dump(mode="json")),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_postlive_contextual_join_v1.py"),
            "status",
            "--input",
            str(certificate_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    status = json.loads(completed.stdout)

    assert status["status"] == "mechanics_completed_non_authorizing"
    assert status["audit_candidate_count"] == 1
    assert not status["audit_action_selected"]
    assert not status["item_error_calibration_performed"]
    assert not status["release_authorizing"]
