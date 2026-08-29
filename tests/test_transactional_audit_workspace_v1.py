from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from scripts.run_transactional_audit_workspace_v1 import main as workspace_cli_main

from literature_multiverse.adaptive_calibration import (
    AdaptiveCalibrationBundle,
    calibrate_adaptive_first_release,
    complete_corpus_identity_from_certificate_v5,
    fit_adaptive_development,
    freeze_adaptive_policy_arm_trajectory,
    freeze_adaptive_preselection_state,
    freeze_complete_corpus_identity,
    freeze_policy_visible_question_trajectory,
    freeze_prospective_adaptive_candidate,
    freeze_question_reference_verdict,
    join_labeled_question_trajectory,
    preselection_state_from_certificate_v5,
)
from literature_multiverse.audit_session import CorrectionDisposition
from literature_multiverse.certificate import VerificationCertificate
from literature_multiverse.claim_release import CLAIM_RELEASE_RISK_FEATURE_NAMES
from literature_multiverse.evidence_graph import EvidenceGraph
from literature_multiverse.lineage import atomic_write_json, hash_canonical
from literature_multiverse.pipeline_fingerprint import PipelineFingerprint
from literature_multiverse.sequential_verification import SequentialVerificationState
from literature_multiverse.transactional_audit_workspace_v1 import (
    AmbiguousTransactionalAuditWorkspaceError,
    AuditWorkspaceTransactionMarkerV1,
    StaleTransactionalAuditStateError,
    TransactionalAuditWorkspaceError,
    _freeze_marker,
    advance_transactional_audit_workspace_v1,
    checkpoint_transactional_audit_workspace_v1,
    freeze_audit_active_cost_checkpoint_receipt_v1,
    freeze_audit_adjudication_cost_receipt_v1,
    initialize_transactional_audit_workspace_v1,
    load_transactional_audit_workspace_v1,
)
from literature_multiverse.verifier import (
    ClaimManifest,
    CorpusLoadResult,
    build_offline_fixture,
    build_verifier_adaptive_policy_context,
    compute_verifier_pipeline_fingerprint,
    load_corpus,
    run_verification,
)


@dataclass(frozen=True)
class WorkspaceContractFixture:
    repository_root: Path
    manifest: ClaimManifest
    corpus: CorpusLoadResult
    fingerprint: PipelineFingerprint
    bundle: AdaptiveCalibrationBundle
    active_state: SequentialVerificationState
    manifest_path: Path
    corpus_path: Path
    fingerprint_path: Path
    bundle_path: Path
    state_path: Path


def _adaptive_release_contract(
    *,
    manifest: ClaimManifest,
    preselection_certificate: VerificationCertificate,
    pipeline_sha256: str,
    budget_minutes: float,
) -> AdaptiveCalibrationBundle:
    context = build_verifier_adaptive_policy_context(
        manifest=manifest,
        pipeline_sha256=pipeline_sha256,
        budget_minutes=budget_minutes,
        policy_arm_id="production-adaptive",
    )
    projected = preselection_state_from_certificate_v5(preselection_certificate)

    def labeled_row(index: int, split: str):
        state = freeze_adaptive_preselection_state(
            prefix_index=0,
            audit_prefix_item_ids=[],
            audit_prefix_cost_minutes=0,
            scheduler_state_sha256=f"{index + 10:x}".rjust(64, "0"),
            evidence_graph_sha256=f"{index + 20:x}".rjust(64, "0"),
            synthesis_sha256=f"{index + 30:x}".rjust(64, "0"),
            non_calibration_assessment_sha256=f"{index + 40:x}".rjust(64, "0"),
            non_calibration_gates_passed=True,
            non_calibration_blocking_reasons=[],
            claim_decision=projected.claim_decision,
            score_features={
                name: projected.score_features[name] for name in CLAIM_RELEASE_RISK_FEATURE_NAMES
            },
        )
        arm = freeze_adaptive_policy_arm_trajectory(
            policy_arm_id=context.policy_arm_id,
            policy_context_sha256=context.policy_context_sha256,
            states=[state],
            terminal_reason="all_items_resolved",
            terminal_candidates=[],
            terminal_source_candidate_input_sha256=hash_canonical([]),
            terminal_remaining_budget_minutes=context.budget_minutes,
        )
        complete_corpus = freeze_complete_corpus_identity(
            corpus_id=f"calibration-corpus-{index}",
            corpus_source_sha256=f"{index + 50:x}".rjust(64, "0"),
            corpus_cutoff=manifest.protocol.corpus_cutoff,
            publication_ids=[f"calibration-publication-{index}"],
            source_manifest_sha256=f"{index + 60:x}".rjust(64, "0"),
        )
        visible = freeze_policy_visible_question_trajectory(
            question_id=f"calibration-question-{index}",
            split=split,  # type: ignore[arg-type]
            population_id=manifest.population_id,
            domain=manifest.domain,
            corpus=complete_corpus,
            arms=[arm],
        )
        reference = freeze_question_reference_verdict(
            question_id=visible.question_id,
            verdict=projected.claim_decision,
            label_source="expert_adjudication",
            adjudication_protocol_sha256="7" * 64,
            adjudication_artifact_sha256=f"{index + 80:x}".rjust(64, "0"),
        )
        return join_labeled_question_trajectory(visible=visible, reference=reference)

    development = [labeled_row(index, "development") for index in range(1, 5)]
    calibration = [labeled_row(index, "calibration") for index in range(5, 9)]
    development_freeze = fit_adaptive_development(
        development,
        policy_contexts=[context],
        alpha=0.99,
        delta=0.5,
        calibration_visible_trajectories=[row.visible for row in calibration],
        candidate_thresholds={context.policy_arm_id: [1.0]},
        seed=11,
    )
    bundle = calibrate_adaptive_first_release(development_freeze, calibration)
    freeze_prospective_adaptive_candidate(
        question_id=manifest.question_id,
        population_id=manifest.population_id,
        domain=manifest.domain,
        policy_arm_id=context.policy_arm_id,
        policy_context_sha256=context.policy_context_sha256,
        corpus=complete_corpus_identity_from_certificate_v5(preselection_certificate),
        observed_states=[projected],
    )
    return bundle


@pytest.fixture(scope="module")
def contract(tmp_path_factory: pytest.TempPathFactory) -> WorkspaceContractFixture:
    root = Path(__file__).resolve().parents[1]
    artifact_root = tmp_path_factory.mktemp("transactional-audit-contract")
    manifest, embedded = build_offline_fixture()
    manifest_path = artifact_root / "claim.json"
    corpus_path = artifact_root / "corpus.json"
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        corpus_path,
        {
            "corpus_bundle_version": "verification-corpus-bundle-v1",
            "corpus_id": manifest.question_id,
            "graph": embedded.graph,
            "eligibility": [row.model_dump(mode="json") for row in embedded.eligibility],
            "metadata": {"empirical_evidence": False, "purpose": "test_fixture"},
        },
    )
    corpus = load_corpus(
        corpus_path,
        legacy_settings=manifest.legacy_adapter,
        repository_root=root,
    )
    fingerprint = compute_verifier_pipeline_fingerprint(root=root)
    shadow = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=root,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )
    bundle = _adaptive_release_contract(
        manifest=manifest,
        preselection_certificate=shadow,
        pipeline_sha256=fingerprint.pipeline_sha256,
        budget_minutes=30,
    )
    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        adaptive_calibration_bundle=bundle,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=root,
        generated_at=datetime(2026, 8, 27, 12, 1, tzinfo=UTC),
    )
    state = certificate.sequential_audit_state
    assert state is not None and state.session.active_action is not None
    fingerprint_path = artifact_root / "pipeline-fingerprint.json"
    bundle_path = artifact_root / "adaptive-calibration.json"
    state_path = artifact_root / "sequential-audit-state.json"
    atomic_write_json(fingerprint_path, fingerprint)
    atomic_write_json(bundle_path, bundle)
    atomic_write_json(state_path, state)
    return WorkspaceContractFixture(
        repository_root=root,
        manifest=manifest,
        corpus=corpus,
        fingerprint=fingerprint,
        bundle=bundle,
        active_state=state,
        manifest_path=manifest_path,
        corpus_path=corpus_path,
        fingerprint_path=fingerprint_path,
        bundle_path=bundle_path,
        state_path=state_path,
    )


def _inputs(contract: WorkspaceContractFixture) -> dict[str, Any]:
    return {
        "manifest": contract.manifest,
        "corpus": contract.corpus,
        "budget_minutes": 30,
        "adaptive_calibration_bundle": contract.bundle,
        "expected_pipeline_fingerprint": contract.fingerprint,
        "pipeline_root": contract.repository_root,
    }


def _initialize(
    contract: WorkspaceContractFixture,
    workspace: Path,
):
    return initialize_transactional_audit_workspace_v1(
        workspace=workspace,
        state=contract.active_state,
        initialized_at=datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
        **_inputs(contract),
    )


def _write_receipt(path: Path, receipt: Any) -> None:
    atomic_write_json(path, receipt)


def _checkpoint_receipt(initialized, *, minutes: float, offset: int = 1):
    authorization = initialized.pointer.authorization
    assert authorization is not None
    return freeze_audit_active_cost_checkpoint_receipt_v1(
        config=initialized.config,
        pointer=initialized.pointer,
        cumulative_active_person_minutes=minutes,
        observer_id="review-clock-01",
        provenance="system_timer",
        recorded_at=authorization.issued_at + timedelta(minutes=offset),
    )


def _adjudication_receipt(
    initialized,
    *,
    cost: float,
    corrected_graph: EvidenceGraph | None = None,
):
    authorization = initialized.pointer.authorization
    assert authorization is not None
    return freeze_audit_adjudication_cost_receipt_v1(
        config=initialized.config,
        pointer=initialized.pointer,
        disposition=(
            CorrectionDisposition.NO_CHANGE
            if corrected_graph is None
            else CorrectionDisposition.CORRECTED
        ),
        corrected_graph=corrected_graph,
        provenance="benchmark_adjudication",
        adjudicator_count=1,
        adjudication_protocol_sha256="a" * 64,
        adjudication_payload_sha256="b" * 64,
        correction_protocol_sha256="c" * 64,
        correction_payload_sha256="d" * 64,
        completed_at=authorization.issued_at + timedelta(minutes=4),
        realized_person_minutes=cost,
    )


def test_init_and_checkpoint_publish_complete_private_generations(
    contract: WorkspaceContractFixture,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "audit-workspace"
    initialized = _initialize(contract, workspace)

    assert initialized.pointer.generation == 0
    assert initialized.pointer.authorization is not None
    assert workspace.stat().st_mode & 0o077 == 0
    assert (workspace / initialized.pointer.state_path).is_file()
    assert (workspace / initialized.pointer.certificate_path).is_file()
    with pytest.raises(
        StaleTransactionalAuditStateError,
        match="stale_predecessor_pointer",
    ):
        checkpoint_transactional_audit_workspace_v1(
            workspace=workspace,
            expected=initialized.pointer.state_expectation,
            expected_pointer_sha256="0" * 64,
            receipt_path=tmp_path / "must-not-be-opened.json",
            **_inputs(contract),
        )

    receipt = _checkpoint_receipt(initialized, minutes=2)
    receipt_path = tmp_path / "checkpoint-receipt.json"
    _write_receipt(receipt_path, receipt)
    checkpointed = checkpoint_transactional_audit_workspace_v1(
        workspace=workspace,
        expected=initialized.pointer.state_expectation,
        expected_pointer_sha256=initialized.pointer.pointer_sha256,
        receipt_path=receipt_path,
        **_inputs(contract),
    )

    assert checkpointed.pointer.generation == 1
    assert checkpointed.pointer.transition_kind == "checkpointed"
    assert checkpointed.pointer.certificate_status == "abstained"
    assert checkpointed.pointer.authorization is not None
    assert checkpointed.pointer.transaction_receipt_sha256s == [receipt.receipt_sha256]
    state = SequentialVerificationState.model_validate_json(
        (workspace / checkpointed.pointer.state_path).read_text(encoding="utf-8")
    )
    assert state.session.active_realized_cost == 2
    assert state.session.active_action is not None
    with pytest.raises(StaleTransactionalAuditStateError, match="stale_expectation"):
        checkpoint_transactional_audit_workspace_v1(
            workspace=workspace,
            expected=initialized.pointer.state_expectation,
            expected_pointer_sha256=checkpointed.pointer.pointer_sha256,
            receipt_path=tmp_path / "must-not-be-opened.json",
            **_inputs(contract),
        )


def test_advance_is_atomic_scientific_rerun_and_auto_selection(
    contract: WorkspaceContractFixture,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "audit-workspace"
    initialized = _initialize(contract, workspace)
    previous_action = initialized.pointer.authorization
    assert previous_action is not None
    receipt = _adjudication_receipt(initialized, cost=3)
    receipt_path = tmp_path / "adjudication-receipt.json"
    _write_receipt(receipt_path, receipt)

    advanced = advance_transactional_audit_workspace_v1(
        workspace=workspace,
        expected=initialized.pointer.state_expectation,
        expected_pointer_sha256=initialized.pointer.pointer_sha256,
        receipt_path=receipt_path,
        **_inputs(contract),
    )

    assert advanced.pointer.generation == 1
    assert advanced.pointer.transition_kind == "adjudicated"
    assert advanced.pointer.transition_receipt_sha256 == receipt.receipt_sha256
    assert advanced.pointer.authorization is not None
    assert advanced.pointer.authorization.item_id != previous_action.item_id
    generation = workspace / advanced.pointer.generation_path
    assert (generation / "preflight-verification-certificate.json").is_file()
    assert (generation / "transition-result.json").is_file()
    state = SequentialVerificationState.model_validate_json(
        (workspace / advanced.pointer.state_path).read_text(encoding="utf-8")
    )
    assert previous_action.item_id in state.session.resolved_item_ids
    assert state.session.current_realized_cost == 3
    assert state.session.active_action is not None
    with pytest.raises(TransactionalAuditWorkspaceError, match="duplicate_receipt"):
        advance_transactional_audit_workspace_v1(
            workspace=workspace,
            expected=advanced.pointer.state_expectation,
            expected_pointer_sha256=advanced.pointer.pointer_sha256,
            receipt_path=receipt_path,
            **_inputs(contract),
        )


def test_corrected_advance_binds_loaded_graph_and_changes_synthesis(
    contract: WorkspaceContractFixture,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "audit-workspace"
    initialized = _initialize(contract, workspace)
    authorization = initialized.pointer.authorization
    assert authorization is not None
    payload = contract.active_state.graph.model_dump(mode="json")
    selected = next(
        estimate
        for estimate in payload["outcome_estimates"]
        if estimate["estimate_id"] == authorization.item_id
    )
    selected["effect"]["estimate"] = 0.5
    corrected_graph = EvidenceGraph.model_validate(payload)
    corrected_path = tmp_path / "corrected-evidence-graph.json"
    atomic_write_json(corrected_path, corrected_graph)
    receipt = _adjudication_receipt(
        initialized,
        cost=4,
        corrected_graph=corrected_graph,
    )
    receipt_path = tmp_path / "corrected-adjudication-receipt.json"
    _write_receipt(receipt_path, receipt)

    advanced = advance_transactional_audit_workspace_v1(
        workspace=workspace,
        expected=initialized.pointer.state_expectation,
        expected_pointer_sha256=initialized.pointer.pointer_sha256,
        receipt_path=receipt_path,
        corrected_corpus_path=corrected_path,
        **_inputs(contract),
    )
    state = SequentialVerificationState.model_validate_json(
        (workspace / advanced.pointer.state_path).read_text(encoding="utf-8")
    )

    assert state.graph == corrected_graph
    assert hash_canonical(state.synthesis) != hash_canonical(contract.active_state.synthesis)


def test_invalid_cost_or_tampered_receipt_never_publishes_or_poisons(
    contract: WorkspaceContractFixture,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "audit-workspace"
    initialized = _initialize(contract, workspace)
    receipt = _checkpoint_receipt(initialized, minutes=2)
    receipt_path = tmp_path / "checkpoint.json"
    _write_receipt(receipt_path, receipt)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["observer_id"] = "tampered-observer"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TransactionalAuditWorkspaceError, match="checkpoint_receipt_invalid"):
        checkpoint_transactional_audit_workspace_v1(
            workspace=workspace,
            expected=initialized.pointer.state_expectation,
            expected_pointer_sha256=initialized.pointer.pointer_sha256,
            receipt_path=receipt_path,
            **_inputs(contract),
        )
    _, pointer = load_transactional_audit_workspace_v1(workspace)
    assert pointer == initialized.pointer

    over_budget = _checkpoint_receipt(initialized, minutes=31, offset=2)
    over_budget_path = tmp_path / "over-budget.json"
    _write_receipt(over_budget_path, over_budget)
    with pytest.raises(ValueError, match="budget"):
        checkpoint_transactional_audit_workspace_v1(
            workspace=workspace,
            expected=initialized.pointer.state_expectation,
            expected_pointer_sha256=initialized.pointer.pointer_sha256,
            receipt_path=over_budget_path,
            **_inputs(contract),
        )
    valid = _checkpoint_receipt(initialized, minutes=3, offset=3)
    valid_path = tmp_path / "valid-checkpoint.json"
    _write_receipt(valid_path, valid)
    committed = checkpoint_transactional_audit_workspace_v1(
        workspace=workspace,
        expected=initialized.pointer.state_expectation,
        expected_pointer_sha256=initialized.pointer.pointer_sha256,
        receipt_path=valid_path,
        **_inputs(contract),
    )
    assert committed.pointer.generation == 1
    decreased = _checkpoint_receipt(committed, minutes=2, offset=1)
    decreased_path = tmp_path / "decreased-checkpoint.json"
    _write_receipt(decreased_path, decreased)
    with pytest.raises(ValueError, match="cannot_decrease"):
        checkpoint_transactional_audit_workspace_v1(
            workspace=workspace,
            expected=committed.pointer.state_expectation,
            expected_pointer_sha256=committed.pointer.pointer_sha256,
            receipt_path=decreased_path,
            **_inputs(contract),
        )
    _, unchanged = load_transactional_audit_workspace_v1(workspace)
    assert unchanged == committed.pointer


def test_outer_lock_allows_exactly_one_concurrent_cas_advance(
    contract: WorkspaceContractFixture,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "audit-workspace"
    initialized = _initialize(contract, workspace)
    receipt = _adjudication_receipt(initialized, cost=3)
    receipt_path = tmp_path / "adjudication.json"
    _write_receipt(receipt_path, receipt)

    def attempt():
        try:
            return advance_transactional_audit_workspace_v1(
                workspace=workspace,
                expected=initialized.pointer.state_expectation,
                expected_pointer_sha256=initialized.pointer.pointer_sha256,
                receipt_path=receipt_path,
                **_inputs(contract),
            )
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: attempt(), range(2)))

    successes = [item for item in outcomes if not isinstance(item, Exception)]
    failures = [item for item in outcomes if isinstance(item, Exception)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], StaleTransactionalAuditStateError)
    _, pointer = load_transactional_audit_workspace_v1(workspace)
    assert pointer.generation == 1


def test_pending_marker_and_pointer_tamper_fail_closed(
    contract: WorkspaceContractFixture,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ambiguous-workspace"
    initialized = _initialize(contract, workspace)
    receipt = _checkpoint_receipt(initialized, minutes=2)
    pending = _freeze_marker(
        status="pending",
        predecessor=initialized.pointer,
        intended_generation=1,
        transition_kind="checkpointed",
        transition_receipt_sha256=receipt.receipt_sha256,
        committed_pointer_sha256=None,
    )
    atomic_write_json(workspace / "transaction-marker.json", pending, force=True)
    with pytest.raises(
        AmbiguousTransactionalAuditWorkspaceError,
        match="prior_transaction_ambiguous",
    ):
        load_transactional_audit_workspace_v1(workspace)

    other = tmp_path / "tampered-workspace"
    second = _initialize(contract, other)
    pointer_path = other / "current-pointer.json"
    pointer_payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer_payload["certificate_sha256"] = "0" * 64
    pointer_path.write_text(json.dumps(pointer_payload), encoding="utf-8")
    with pytest.raises(TransactionalAuditWorkspaceError, match="workspace_pointer_invalid"):
        load_transactional_audit_workspace_v1(other)
    assert second.pointer.generation == 0


def test_committed_generation_history_is_replayed_and_tamper_evident(
    contract: WorkspaceContractFixture,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "history-workspace"
    initialized = _initialize(contract, workspace)
    receipt = _checkpoint_receipt(initialized, minutes=2)
    receipt_path = tmp_path / "history-checkpoint.json"
    _write_receipt(receipt_path, receipt)
    checkpointed = checkpoint_transactional_audit_workspace_v1(
        workspace=workspace,
        expected=initialized.pointer.state_expectation,
        expected_pointer_sha256=initialized.pointer.pointer_sha256,
        receipt_path=receipt_path,
        **_inputs(contract),
    )
    stored_receipt_path = (
        workspace / checkpointed.pointer.generation_path / "transition-receipt.json"
    )
    payload = json.loads(stored_receipt_path.read_text(encoding="utf-8"))
    payload["observer_id"] = "post-commit-tamper"
    stored_receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        TransactionalAuditWorkspaceError,
        match="generation_transition_receipt_invalid",
    ):
        load_transactional_audit_workspace_v1(workspace)


def test_manifest_v3_is_rejected_before_workspace_creation(
    contract: WorkspaceContractFixture,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "condition-workspace"
    with pytest.raises(
        TransactionalAuditWorkspaceError,
        match="manifest_v3_not_supported",
    ):
        initialize_transactional_audit_workspace_v1(
            workspace=workspace,
            manifest=SimpleNamespace(claim_manifest_version="3"),  # type: ignore[arg-type]
            corpus=contract.corpus,
            budget_minutes=30,
            adaptive_calibration_bundle=contract.bundle,
            state=contract.active_state,
            expected_pipeline_fingerprint=contract.fingerprint,
            pipeline_root=contract.repository_root,
            initialized_at=datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
        )
    assert not workspace.exists()


def test_standalone_cli_executes_init_checkpoint_and_advance(
    contract: WorkspaceContractFixture,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "cli-workspace"
    common = [
        "--workspace",
        str(workspace),
        "--claim",
        str(contract.manifest_path),
        "--corpus",
        str(contract.corpus_path),
        "--budget-minutes",
        "30",
        "--adaptive-calibration",
        str(contract.bundle_path),
        "--pipeline-fingerprint",
        str(contract.fingerprint_path),
        "--pipeline-root",
        str(contract.repository_root),
    ]
    assert (
        workspace_cli_main(
            [
                "init",
                *common,
                "--state",
                str(contract.state_path),
                "--initialized-at",
                "2026-08-27T12:02:00Z",
            ]
        )
        == 0
    )
    initialized_summary = json.loads(capsys.readouterr().out)
    assert initialized_summary["generation"] == 0
    config, pointer = load_transactional_audit_workspace_v1(workspace)
    checkpoint = freeze_audit_active_cost_checkpoint_receipt_v1(
        config=config,
        pointer=pointer,
        cumulative_active_person_minutes=2,
        observer_id="cli-timer",
        provenance="system_timer",
        recorded_at=pointer.authorization.issued_at + timedelta(minutes=1),  # type: ignore[union-attr]
    )
    checkpoint_path = tmp_path / "cli-checkpoint.json"
    _write_receipt(checkpoint_path, checkpoint)
    expectation_path = workspace / pointer.generation_path / "state-expectation.json"
    assert (
        workspace_cli_main(
            [
                "checkpoint",
                *common,
                "--expected",
                str(expectation_path),
                "--expected-pointer-sha256",
                pointer.pointer_sha256,
                "--checkpoint-receipt",
                str(checkpoint_path),
            ]
        )
        == 0
    )
    checkpoint_summary = json.loads(capsys.readouterr().out)
    assert checkpoint_summary["generation"] == 1

    config, pointer = load_transactional_audit_workspace_v1(workspace)
    authorization = pointer.authorization
    assert authorization is not None
    adjudication = freeze_audit_adjudication_cost_receipt_v1(
        config=config,
        pointer=pointer,
        disposition=CorrectionDisposition.NO_CHANGE,
        corrected_graph=None,
        provenance="benchmark_adjudication",
        adjudicator_count=1,
        adjudication_protocol_sha256="a" * 64,
        adjudication_payload_sha256="b" * 64,
        correction_protocol_sha256="c" * 64,
        correction_payload_sha256="d" * 64,
        completed_at=authorization.issued_at + timedelta(minutes=2),
        realized_person_minutes=3,
    )
    adjudication_path = tmp_path / "cli-adjudication.json"
    _write_receipt(adjudication_path, adjudication)
    expectation_path = workspace / pointer.generation_path / "state-expectation.json"
    assert (
        workspace_cli_main(
            [
                "advance",
                *common,
                "--expected",
                str(expectation_path),
                "--expected-pointer-sha256",
                pointer.pointer_sha256,
                "--adjudication-receipt",
                str(adjudication_path),
            ]
        )
        == 0
    )
    advanced_summary = json.loads(capsys.readouterr().out)
    assert advanced_summary["generation"] == 2
    assert advanced_summary["transition_kind"] == "adjudicated"


def test_marker_model_rejects_hash_tamper() -> None:
    payload = {
        "marker_version": "transactional-audit-marker-v1",
        "status": "pending",
        "predecessor_pointer_sha256": "1" * 64,
        "intended_generation": 2,
        "transition_kind": "checkpointed",
        "transition_receipt_sha256": "2" * 64,
        "committed_pointer_sha256": None,
        "marker_sha256": "3" * 64,
    }
    with pytest.raises(ValueError, match="marker_hash_mismatch"):
        AuditWorkspaceTransactionMarkerV1.model_validate(payload)
