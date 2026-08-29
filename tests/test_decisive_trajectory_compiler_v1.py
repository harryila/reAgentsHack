from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from itertools import permutations
from pathlib import Path
from typing import Any

import pytest
from scripts.run_decisive_trajectory_compiler_v1 import main as compiler_cli_main

from literature_multiverse.adjudication_replay import (
    AdjudicationCorrectionArtifactV1,
    AdjudicationItemReplayV1,
    AdjudicationProtocolArtifactV1,
    AdjudicationReplayPackageV1,
    AdjudicationResolutionArtifactV1,
    ExactAdjudicationArtifactLocatorV1,
    IndependentReviewerDecisionV1,
    OperatorReviewerRosterEntryV1,
    ReviewerDecisionDigestV1,
    ReviewerTimingEvidenceV1,
    freeze_adjudication_operator_trust_registry_v1,
    freeze_adjudication_replay_package_v1,
)
from literature_multiverse.audit_session import CorrectionDisposition
from literature_multiverse.certificate import (
    FinalConditionVerificationCertificateV7,
    VerificationCertificate,
)
from literature_multiverse.decisive_claim_evaluation_v1 import (
    DecisiveEvaluationConfigV1,
    DecisiveSplitManifestV1,
    FitStageReceiptV1,
    _fixture_hash,
    _fixture_state,
    assess_decisive_evaluation_readiness_v1,
    freeze_decisive_evaluation_config_v1,
    freeze_decisive_policy_input_provenance_v1,
    freeze_decisive_split_manifest_v1,
    freeze_fit_stage_receipt_v1,
    freeze_question_identity_v1,
    freeze_question_trajectory_v1,
)
from literature_multiverse.decisive_compilation_lineage_v1 import (
    DecisiveCompilationLineageV1Error,
    replay_decisive_compilation_lineage_v1,
)
from literature_multiverse.decisive_trajectory_compiler_v1 import (
    DecisiveTrajectoryCompilationResultV1,
    DecisiveTrajectoryCompilerV1Error,
    DecisiveTrajectorySourceRosterV1,
    _bind_condition_artifacts,
    _replace_unfinalized_v5_condition_states,
    _ReplayCandidate,
    _require_real_certificate,
    _required_policy_prefix_union,
    _snapshot_verifier_certificate,
    compile_decisive_trajectory_bundle_v1,
    freeze_adjudication_replay_package_locator_v1,
    freeze_condition_set_source_binding_v1,
    freeze_decisive_trajectory_source_roster_v1,
    freeze_normalized_condition_set_artifact_v1,
    freeze_question_trajectory_source_v1,
    freeze_transactional_workspace_locator_v1,
    freeze_verifier_certificate_locator_v1,
    replay_decisive_trajectory_compilation_v1,
)
from literature_multiverse.evidence_graph import EvidenceGraph
from literature_multiverse.lineage import atomic_write_json, hash_canonical
from literature_multiverse.question_evaluation import (
    AuditCostBasis,
    AuditDisposition,
    BenchmarkEvidenceKind,
    freeze_question_audit_event,
    freeze_question_replay_state_from_certificate,
)
from literature_multiverse.transactional_audit_workspace_v1 import (
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
    CorpusProvenanceAssurance,
    build_offline_fixture,
    compute_verifier_pipeline_fingerprint,
    finalize_condition_verification,
    run_verification,
)
from test_unified_verifier import (  # type: ignore[import-not-found]
    _adaptive_release_contract,
    _artifact_backed_item_risk_contract,
    _condition_runtime_fixture,
    _source_replayed_fixture_contract,
)

_T0 = datetime(2026, 8, 29, 9, tzinfo=UTC)


@pytest.fixture(scope="module")
def final_condition_v7_certificate() -> FinalConditionVerificationCertificateV7:
    (
        manifest,
        corpus,
        fingerprint,
        plan,
        development,
        model,
        assessment,
        bundle,
        item_risk_receipt,
    ) = _condition_runtime_fixture()
    source = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        adaptive_calibration_bundle_v2=bundle,
        condition_plan=plan,
        condition_development_graph=development,
        condition_frozen_model=model,
        item_risk_scoring_receipt=item_risk_receipt,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=Path(__file__).resolve().parents[1],
        generated_at=_T0,
    )
    return finalize_condition_verification(
        source_certificate=source,
        condition_confirmation_assessment=assessment,
        generated_at=_T0,
    )


def _single_estimate_graph(graph: EvidenceGraph) -> EvidenceGraph:
    estimate = graph.outcome_estimates[0]
    contrast = next(row for row in graph.contrasts if row.contrast_id == estimate.contrast_id)
    cohort = next(row for row in graph.cohorts if row.cohort_id == contrast.cohort_id)
    study = next(row for row in graph.studies if row.study_id == cohort.study_id)
    publication_ids = set(study.publication_ids)
    paper_ids = {
        row.paper_id for row in graph.publications if row.publication_id in publication_ids
    }
    assert estimate.effect.paper_id in paper_ids
    arm_ids = {contrast.treatment_arm_id, contrast.comparator_arm_id}
    span_ids = set(estimate.evidence_span_ids)
    return EvidenceGraph(
        publications=[row for row in graph.publications if row.publication_id in publication_ids],
        studies=[study],
        cohorts=[cohort],
        arms=[row for row in graph.arms if row.arm_id in arm_ids],
        contrasts=[contrast],
        outcome_estimates=[estimate],
        evidence_spans=[row for row in graph.evidence_spans if row.span_id in span_ids],
    )


@dataclass(frozen=True)
class RealCompilerContract:
    repository_root: Path
    source_root: Path
    workspace: Path
    config: DecisiveEvaluationConfigV1
    split_manifest: DecisiveSplitManifestV1
    development_receipt: FitStageReceiptV1
    calibration_receipt: FitStageReceiptV1
    source_roster: DecisiveTrajectorySourceRosterV1
    source_roster_path: Path
    adjudication_package_path: Path
    adjudication_artifact_paths: dict[str, Path]
    manifest: ClaimManifest
    corpus: CorpusLoadResult
    baseline_certificate: VerificationCertificate


@pytest.fixture(scope="module")
def real_contract(tmp_path_factory: pytest.TempPathFactory) -> RealCompilerContract:
    repository_root = Path(__file__).resolve().parents[1]
    root = tmp_path_factory.mktemp("decisive-trajectory-compiler-real")
    source_root = root / "sources"
    source_root.mkdir()
    workspace_parent = source_root / "workspaces"
    workspace_parent.mkdir()
    workspace = workspace_parent / "evaluation-question"

    original_manifest, embedded = build_offline_fixture()
    graph = _single_estimate_graph(embedded.graph)
    selected_paper_ids = {row.paper_id for row in graph.publications}
    eligibility = tuple(row for row in embedded.eligibility if row.paper_id in selected_paper_ids)
    manifest = original_manifest.model_copy(
        update={
            "question_id": "real-compiler-evaluation-question",
            "population_id": "prospective-biomedical-population",
            "domain": "clinical-medicine",
            "release": original_manifest.release.model_copy(
                update={"require_prediction_interval_stability": False}
            ),
            "audit_guard": original_manifest.audit_guard.model_copy(
                update={
                    "block_counterfactual_conclusion_flips": False,
                    "max_unresolved_item_cell_ucl_sum": 0.001,
                    "max_unresolved_expected_claim_loss": 10.0,
                    "max_unresolved_item_influence": 1.0,
                }
            ),
        }
    )
    provisional = replace(
        embedded,
        corpus_id=manifest.question_id,
        graph=graph,
        eligibility=eligibility,
        source_sha256=hash_canonical({"real_compiler_source_graph": graph}),
    )
    fingerprint = compute_verifier_pipeline_fingerprint(root=repository_root)
    replay_sha256 = hash_canonical({"real_compiler_grounding_replay": graph})
    replayed_eligibility, metadata, extraction_context = _source_replayed_fixture_contract(
        manifest=manifest,
        fixture=provisional,
        pipeline_sha256=fingerprint.pipeline_sha256,
        replay_sha256=replay_sha256,
    )
    corpus = CorpusLoadResult(
        corpus_id=manifest.question_id,
        source_label="/frozen/native/real-compiler-package.json",
        source_format="typed_evidence_grounding_package_json",
        source_sha256=provisional.source_sha256,
        graph=graph,
        eligibility=replayed_eligibility,
        adapter_issues=(),
        metadata=metadata,
        provenance_assurance=CorpusProvenanceAssurance(
            status="source_replayed_native_grounding",
            reason="Exactly replayed native grounding package for contract testing.",
            replay_sha256=replay_sha256,
        ),
        extraction_context=extraction_context,
    )
    _, _, item_risk_receipt = _artifact_backed_item_risk_contract(
        manifest=manifest,
        corpus=corpus,
        fingerprint=fingerprint,
        repository_root=repository_root,
    )
    shadow = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=repository_root,
        item_risk_scoring_receipt=item_risk_receipt,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=_T0,
    )
    adaptive_bundle, _ = _adaptive_release_contract(
        manifest=manifest,
        preselection_certificate=shadow,
        pipeline_sha256=fingerprint.pipeline_sha256,
        budget_minutes=30,
    )
    baseline_certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        adaptive_calibration_bundle=adaptive_bundle,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=repository_root,
        item_risk_scoring_receipt=item_risk_receipt,
        generated_at=_T0 + timedelta(minutes=1),
    )
    active_state = baseline_certificate.sequential_audit_state
    assert active_state is not None and active_state.session.active_action is not None
    certificate_root = source_root / "certificates"
    certificate_root.mkdir()
    baseline_certificate_path = certificate_root / "baseline.json"
    atomic_write_json(baseline_certificate_path, baseline_certificate)
    common = {
        "manifest": manifest,
        "corpus": corpus,
        "budget_minutes": 30.0,
        "adaptive_calibration_bundle": adaptive_bundle,
        "expected_pipeline_fingerprint": fingerprint,
        "pipeline_root": repository_root,
        "item_risk_scoring_receipt": item_risk_receipt,
    }
    initialized = initialize_transactional_audit_workspace_v1(
        workspace=workspace,
        state=active_state,
        initialized_at=_T0 + timedelta(minutes=2),
        **common,
    )
    authorization = initialized.pointer.authorization
    assert authorization is not None
    checkpoint = freeze_audit_active_cost_checkpoint_receipt_v1(
        config=initialized.config,
        pointer=initialized.pointer,
        cumulative_active_person_minutes=1.0,
        observer_id="human-review-clock",
        provenance="system_timer",
        recorded_at=authorization.issued_at + timedelta(minutes=1),
    )
    checkpoint_path = root / "checkpoint.json"
    atomic_write_json(checkpoint_path, checkpoint)
    checkpointed = checkpoint_transactional_audit_workspace_v1(
        workspace=workspace,
        expected=initialized.pointer.state_expectation,
        expected_pointer_sha256=initialized.pointer.pointer_sha256,
        receipt_path=checkpoint_path,
        **common,
    )
    checkpoint_authorization = checkpointed.pointer.authorization
    assert checkpoint_authorization is not None

    adjudication_root = source_root / "adjudication"
    adjudication_root.mkdir()
    adjudication_artifact_paths: dict[str, Path] = {}

    def freeze_raw(name: str, value: Any) -> ExactAdjudicationArtifactLocatorV1:
        path = adjudication_root / f"{name}.json"
        atomic_write_json(path, value)
        adjudication_artifact_paths[name] = path
        return ExactAdjudicationArtifactLocatorV1(
            relative_path=f"adjudication/{name}.json",
            expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    registry = freeze_adjudication_operator_trust_registry_v1(
        registry_id="operator-registry-2026-08-29",
        operator_id="literature-multiverse-study-operator",
        trust_root_id="prespecified-local-operator-trust-root",
        reviewers=[
            OperatorReviewerRosterEntryV1(
                reviewer_id="reviewer-1",
                roles=["independent_reviewer"],
                declared_expertise_scope="clinical evidence synthesis",
            ),
            OperatorReviewerRosterEntryV1(
                reviewer_id="reviewer-2",
                roles=["independent_reviewer"],
                declared_expertise_scope="clinical evidence synthesis",
            ),
            OperatorReviewerRosterEntryV1(
                reviewer_id="resolver-1",
                roles=["final_adjudicator", "timekeeper"],
                declared_expertise_scope="clinical evidence adjudication",
            ),
        ],
    )
    registry_locator = freeze_raw("trust-registry", registry)
    protocol = AdjudicationProtocolArtifactV1(
        protocol_id="decisive-human-audit-protocol-2026-08-29",
        trust_registry_sha256=registry.registry_sha256,
    )
    protocol_locator = freeze_raw("protocol", protocol)
    protocol_sha256 = protocol_locator.expected_file_sha256
    item_id = checkpoint_authorization.item_id
    started_at = checkpoint_authorization.issued_at
    reviewer_1_completed = started_at + timedelta(minutes=1)
    reviewer_2_completed = started_at + timedelta(minutes=1, seconds=15)
    resolution_completed = started_at + timedelta(minutes=3)
    decision_1 = IndependentReviewerDecisionV1(
        question_id=manifest.question_id,
        item_id=item_id,
        reviewer_id="reviewer-1",
        disposition="no_change",
        submitted_at=reviewer_1_completed,
        adjudication_protocol_file_sha256=protocol_sha256,
        decision_rationale="Source value and grounding agree with the extraction.",
    )
    decision_2 = IndependentReviewerDecisionV1(
        question_id=manifest.question_id,
        item_id=item_id,
        reviewer_id="reviewer-2",
        disposition="corrected",
        submitted_at=reviewer_2_completed,
        adjudication_protocol_file_sha256=protocol_sha256,
        decision_rationale="Independent review raised a discrepancy for resolution.",
    )
    decision_locators = sorted(
        [
            freeze_raw("decision-reviewer-1", decision_1),
            freeze_raw("decision-reviewer-2", decision_2),
        ],
        key=lambda row: row.relative_path,
    )
    timing_locators = sorted(
        [
            freeze_raw(
                "timing-reviewer-1",
                ReviewerTimingEvidenceV1(
                    question_id=manifest.question_id,
                    item_id=item_id,
                    reviewer_id="reviewer-1",
                    observer_id="resolver-1",
                    started_at=started_at,
                    completed_at=reviewer_1_completed,
                    active_person_minutes=0.75,
                    adjudication_protocol_file_sha256=protocol_sha256,
                ),
            ),
            freeze_raw(
                "timing-reviewer-2",
                ReviewerTimingEvidenceV1(
                    question_id=manifest.question_id,
                    item_id=item_id,
                    reviewer_id="reviewer-2",
                    observer_id="resolver-1",
                    started_at=started_at,
                    completed_at=reviewer_2_completed,
                    active_person_minutes=0.75,
                    adjudication_protocol_file_sha256=protocol_sha256,
                ),
            ),
            freeze_raw(
                "timing-resolver-1",
                ReviewerTimingEvidenceV1(
                    question_id=manifest.question_id,
                    item_id=item_id,
                    reviewer_id="resolver-1",
                    observer_id="resolver-1",
                    started_at=reviewer_2_completed,
                    completed_at=resolution_completed,
                    active_person_minutes=1.5,
                    adjudication_protocol_file_sha256=protocol_sha256,
                ),
            ),
        ],
        key=lambda row: row.relative_path,
    )
    correction = AdjudicationCorrectionArtifactV1(
        question_id=manifest.question_id,
        item_id=item_id,
        disposition="no_change",
        corrected_graph_sha256=None,
        adjudication_protocol_file_sha256=protocol_sha256,
        correction_rationale="Final adjudication retained the extracted evidence unchanged.",
    )
    correction_locator = freeze_raw("correction", correction)
    decision_file_sha_by_reviewer = {
        "reviewer-1": next(
            row.expected_file_sha256
            for row in decision_locators
            if row.relative_path.endswith("decision-reviewer-1.json")
        ),
        "reviewer-2": next(
            row.expected_file_sha256
            for row in decision_locators
            if row.relative_path.endswith("decision-reviewer-2.json")
        ),
    }
    resolution = AdjudicationResolutionArtifactV1(
        question_id=manifest.question_id,
        item_id=item_id,
        final_adjudicator_id="resolver-1",
        independent_decisions=[
            ReviewerDecisionDigestV1(
                reviewer_id=reviewer_id,
                decision_file_sha256=decision_file_sha_by_reviewer[reviewer_id],
            )
            for reviewer_id in sorted(decision_file_sha_by_reviewer)
        ],
        disposition="no_change",
        corrected_graph_sha256=None,
        completed_at=resolution_completed,
        adjudication_protocol_file_sha256=protocol_sha256,
        correction_payload_file_sha256=correction_locator.expected_file_sha256,
        resolution_rationale="The final adjudicator resolved the discrepancy as no change.",
    )
    resolution_locator = freeze_raw("resolution", resolution)
    adjudication = freeze_audit_adjudication_cost_receipt_v1(
        config=checkpointed.config,
        pointer=checkpointed.pointer,
        disposition=CorrectionDisposition.NO_CHANGE,
        corrected_graph=None,
        provenance="blinded_human",
        adjudicator_count=3,
        adjudication_protocol_sha256=protocol_sha256,
        adjudication_payload_sha256=resolution_locator.expected_file_sha256,
        correction_protocol_sha256=protocol_sha256,
        correction_payload_sha256=correction_locator.expected_file_sha256,
        completed_at=resolution_completed,
        realized_person_minutes=3.0,
    )
    adjudication_path = root / "adjudication.json"
    atomic_write_json(adjudication_path, adjudication)
    advanced = advance_transactional_audit_workspace_v1(
        workspace=workspace,
        expected=checkpointed.pointer.state_expectation,
        expected_pointer_sha256=checkpointed.pointer.pointer_sha256,
        receipt_path=adjudication_path,
        **common,
    )
    assert advanced.pointer.generation == 2
    package = freeze_adjudication_replay_package_v1(
        question_id=manifest.question_id,
        trust_registry=registry_locator,
        adjudication_protocol=protocol_locator,
        items=[
            AdjudicationItemReplayV1(
                item_id=item_id,
                receipt_sha256s=[adjudication.receipt_sha256],
                independent_reviewer_decisions=decision_locators,
                timing_evidence=timing_locators,
                resolution=resolution_locator,
                correction_payload=correction_locator,
            )
        ],
    )
    adjudication_package_path = adjudication_root / "package.json"
    atomic_write_json(adjudication_package_path, package)
    adjudication_artifact_paths["package"] = adjudication_package_path
    adjudication_package_locator = freeze_adjudication_replay_package_locator_v1(
        relative_path="adjudication/package.json",
        expected_file_sha256=hashlib.sha256(adjudication_package_path.read_bytes()).hexdigest(),
        expected_package_sha256=package.package_sha256,
    )

    config = freeze_decisive_evaluation_config_v1(
        budgets_minutes_per_question=(5.0, 15.0, 30.0),
        bootstrap_draws=100,
    )
    pipeline_sha256 = fingerprint.pipeline_sha256
    development_identity = freeze_question_identity_v1(
        split="development",
        question_id="real-compiler-development-question",
        claim_id="real-compiler-development-claim",
        domain=manifest.domain,
        population_id=manifest.population_id,
        pipeline_sha256=pipeline_sha256,
        corpus_sha256="1" * 64,
        paper_ids=["development-paper"],
        cohort_ids=["development-cohort"],
    )
    calibration_identity = freeze_question_identity_v1(
        split="calibration",
        question_id="real-compiler-calibration-question",
        claim_id="real-compiler-calibration-claim",
        domain=manifest.domain,
        population_id=manifest.population_id,
        pipeline_sha256=pipeline_sha256,
        corpus_sha256="2" * 64,
        paper_ids=["calibration-paper"],
        cohort_ids=["calibration-cohort"],
    )
    evaluation_identity = freeze_question_identity_v1(
        split="evaluation",
        question_id=manifest.question_id,
        claim_id="real-compiler-evaluation-claim",
        domain=manifest.domain,
        population_id=manifest.population_id,
        pipeline_sha256=pipeline_sha256,
        corpus_sha256=baseline_certificate.complete_corpus_identity.membership_sha256,
        paper_ids=sorted(row.paper_id for row in graph.publications),
        cohort_ids=sorted(row.cohort_id for row in graph.cohorts),
    )
    split_manifest = freeze_decisive_split_manifest_v1(
        identities=[development_identity, calibration_identity, evaluation_identity],
        split_salt_sha256="3" * 64,
    )
    development_receipt = freeze_fit_stage_receipt_v1(
        stage="development_optimizer_fit",
        identities=[development_identity],
        pipeline_sha256=pipeline_sha256,
        input_manifest_sha256=split_manifest.manifest_sha256,
        label_source="expert_adjudication",
        frozen_optimizer_or_policy_sha256="4" * 64,
        frozen_threshold_or_bounds_sha256=None,
        completed_at=_T0 - timedelta(minutes=2),
    )
    calibration_receipt = freeze_fit_stage_receipt_v1(
        stage="calibration_policy_and_threshold_freeze",
        identities=[calibration_identity],
        pipeline_sha256=pipeline_sha256,
        input_manifest_sha256=split_manifest.manifest_sha256,
        label_source="expert_adjudication",
        frozen_optimizer_or_policy_sha256="5" * 64,
        frozen_threshold_or_bounds_sha256="6" * 64,
        completed_at=_T0 - timedelta(minutes=1),
    )
    workspace_config, pointer = load_transactional_audit_workspace_v1(workspace)
    locator = freeze_transactional_workspace_locator_v1(
        relative_path="workspaces/evaluation-question",
        expected_workspace_config_sha256=workspace_config.config_sha256,
        expected_terminal_pointer_sha256=pointer.pointer_sha256,
    )
    certificate_locator = freeze_verifier_certificate_locator_v1(
        relative_path="certificates/baseline.json",
        expected_file_sha256=hashlib.sha256(baseline_certificate_path.read_bytes()).hexdigest(),
        expected_certificate_sha256=baseline_certificate.certificate_sha256,
    )
    source = freeze_question_trajectory_source_v1(
        question_id=manifest.question_id,
        workspaces=[locator],
        adjudication_replay_package=adjudication_package_locator,
        verifier_certificates=[certificate_locator],
    )
    source_roster = freeze_decisive_trajectory_source_roster_v1(
        split_manifest=split_manifest,
        questions=[source],
    )
    source_roster_path = root / "source-roster.json"
    atomic_write_json(source_roster_path, source_roster)
    return RealCompilerContract(
        repository_root=repository_root,
        source_root=source_root,
        workspace=workspace,
        config=config,
        split_manifest=split_manifest,
        development_receipt=development_receipt,
        calibration_receipt=calibration_receipt,
        source_roster=source_roster,
        source_roster_path=source_roster_path,
        adjudication_package_path=adjudication_package_path,
        adjudication_artifact_paths=adjudication_artifact_paths,
        manifest=manifest,
        corpus=corpus,
        baseline_certificate=baseline_certificate,
    )


def _compile(contract: RealCompilerContract) -> DecisiveTrajectoryCompilationResultV1:
    return compile_decisive_trajectory_bundle_v1(
        config=contract.config,
        split_manifest=contract.split_manifest,
        development_receipt=contract.development_receipt,
        calibration_receipt=contract.calibration_receipt,
        source_roster=contract.source_roster,
        source_roster_path=contract.source_roster_path,
        source_root=contract.source_root,
        repository_root=contract.repository_root,
        compiled_at=_T0 + timedelta(hours=1),
    )


def _overwrite_model_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _contract_with_rebound_adjudication_artifact(
    contract: RealCompilerContract,
    *,
    tmp_path: Path,
    artifact_name: str,
    replacement_artifact: Any,
) -> RealCompilerContract:
    """Copy the source tree and intentionally re-freeze one raw workflow file.

    Rebinding both the raw-file locator and outer package/roster hashes exercises
    semantic replay rather than merely exercising the first byte-hash guard.
    """

    copied_root = tmp_path / "sources"
    shutil.copytree(contract.source_root, copied_root)
    relative_artifact = contract.adjudication_artifact_paths[artifact_name].relative_to(
        contract.source_root
    )
    artifact_path = copied_root / relative_artifact
    _overwrite_model_json(artifact_path, replacement_artifact)
    replacement_locator = ExactAdjudicationArtifactLocatorV1(
        relative_path=relative_artifact.as_posix(),
        expected_file_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    )

    package_path = copied_root / contract.adjudication_package_path.relative_to(
        contract.source_root
    )
    package = AdjudicationReplayPackageV1.model_validate_json(
        package_path.read_text(encoding="utf-8")
    )
    item = package.items[0]

    def rebound(locator: ExactAdjudicationArtifactLocatorV1) -> ExactAdjudicationArtifactLocatorV1:
        return (
            replacement_locator
            if locator.relative_path == relative_artifact.as_posix()
            else locator
        )

    rebound_item = AdjudicationItemReplayV1(
        item_id=item.item_id,
        receipt_sha256s=item.receipt_sha256s,
        independent_reviewer_decisions=[
            rebound(row) for row in item.independent_reviewer_decisions
        ],
        timing_evidence=[rebound(row) for row in item.timing_evidence],
        resolution=rebound(item.resolution),
        correction_payload=rebound(item.correction_payload),
    )
    rebound_package = freeze_adjudication_replay_package_v1(
        question_id=package.question_id,
        trust_registry=package.trust_registry,
        adjudication_protocol=package.adjudication_protocol,
        items=[rebound_item],
    )
    _overwrite_model_json(package_path, rebound_package)
    package_locator = freeze_adjudication_replay_package_locator_v1(
        relative_path=contract.adjudication_package_path.relative_to(
            contract.source_root
        ).as_posix(),
        expected_file_sha256=hashlib.sha256(package_path.read_bytes()).hexdigest(),
        expected_package_sha256=rebound_package.package_sha256,
    )
    original_source = contract.source_roster.questions[0]
    rebound_source = freeze_question_trajectory_source_v1(
        question_id=original_source.question_id,
        workspaces=original_source.workspaces,
        adjudication_replay_package=package_locator,
        verifier_certificates=original_source.verifier_certificates,
        condition_set_bindings=original_source.condition_set_bindings,
    )
    rebound_roster = freeze_decisive_trajectory_source_roster_v1(
        split_manifest=contract.split_manifest,
        questions=[rebound_source],
    )
    rebound_roster_path = tmp_path / "source-roster.json"
    atomic_write_json(rebound_roster_path, rebound_roster)
    copied_artifacts = {
        name: copied_root / path.relative_to(contract.source_root)
        for name, path in contract.adjudication_artifact_paths.items()
    }
    return replace(
        contract,
        source_root=copied_root,
        workspace=copied_root / contract.workspace.relative_to(contract.source_root),
        source_roster=rebound_roster,
        source_roster_path=rebound_roster_path,
        adjudication_package_path=package_path,
        adjudication_artifact_paths=copied_artifacts,
    )


def test_real_workspace_compiles_exact_policy_prefix_union_and_replays(
    real_contract: RealCompilerContract,
) -> None:
    result = _compile(real_contract)
    trajectory = result.trajectory_bundle.trajectories[0]
    question_receipt = result.compilation_receipt.question_receipts[0]
    item_id = trajectory.audit_events[0].item_id

    assert result.trajectory_bundle.evidence_kind == "real_expert_adjudicated"
    assert [row.audit_sequence for row in trajectory.replay_states] == [[], [item_id]]
    assert question_receipt.available_prefixes == [[], [item_id]]
    assert question_receipt.required_policy_visited_prefixes == [[], [item_id]]
    assert question_receipt.total_realized_person_minutes == 3.0
    assert trajectory.audit_events[0].cost_basis is AuditCostBasis.REALIZED_HUMAN_MINUTES
    assert trajectory.audit_events[0].adjudicator_count == 3
    assert (
        trajectory.audit_events[0].artifact_sha256
        == hashlib.sha256(
            real_contract.adjudication_artifact_paths["resolution"].read_bytes()
        ).hexdigest()
    )
    assert (
        question_receipt.adjudication_replay_package_binding.operator_trust_registry_hash_bound
        is True
    )
    assert (
        question_receipt.adjudication_replay_package_binding.external_reviewer_expertise_verified
        is False
    )
    assert question_receipt.operator_declared_expert_workflow_replayed is True
    assert question_receipt.external_reviewer_identity_or_expertise_proven is False
    assert question_receipt.workspace_bindings[0].checkpoint_receipt_sha256s
    assert question_receipt.workspace_bindings[0].resolution_result_sha256s
    assert result.compilation_receipt.evaluation_reference_labels_opened is False
    assert result.compilation_receipt.scientific_claim_authority is False
    assert result.compilation_receipt.claim_release_authority is False
    assert (
        replay_decisive_trajectory_compilation_v1(
            expected=result,
            config=real_contract.config,
            split_manifest=real_contract.split_manifest,
            development_receipt=real_contract.development_receipt,
            calibration_receipt=real_contract.calibration_receipt,
            source_roster=real_contract.source_roster,
            source_roster_path=real_contract.source_roster_path,
            source_root=real_contract.source_root,
            repository_root=real_contract.repository_root,
        )
        == result
    )


def test_real_bundle_requires_exact_compiler_lineage_before_label_lifecycle(
    real_contract: RealCompilerContract,
    tmp_path: Path,
) -> None:
    result = _compile(real_contract)
    result_path = tmp_path / "compilation-result.json"
    atomic_write_json(result_path, result)
    proof = replay_decisive_compilation_lineage_v1(
        config=real_contract.config,
        split_manifest=real_contract.split_manifest,
        development_receipt=real_contract.development_receipt,
        calibration_receipt=real_contract.calibration_receipt,
        trajectory_bundle=result.trajectory_bundle,
        compiler_result_path=result_path,
        source_roster_path=real_contract.source_roster_path,
        source_root=real_contract.source_root,
        repository_root=real_contract.repository_root,
    )
    assert proof.lineage_identity == result.compilation_receipt.compilation_lineage_identity
    assert proof.compiler_result_sha256 == result.result_sha256
    assert proof.bare_trajectory_bundle_is_not_sufficient is True
    assert proof.external_reviewer_identity_or_expertise_proven is False

    byte_changed_path = tmp_path / "compilation-result-byte-changed.json"
    byte_changed_path.write_bytes(result_path.read_bytes() + b"\n")
    byte_changed_proof = replay_decisive_compilation_lineage_v1(
        config=real_contract.config,
        split_manifest=real_contract.split_manifest,
        development_receipt=real_contract.development_receipt,
        calibration_receipt=real_contract.calibration_receipt,
        trajectory_bundle=result.trajectory_bundle,
        compiler_result_path=byte_changed_path,
        source_roster_path=real_contract.source_roster_path,
        source_root=real_contract.source_root,
        repository_root=real_contract.repository_root,
    )
    assert byte_changed_proof.compiler_result_sha256 == proof.compiler_result_sha256
    assert byte_changed_proof.compiler_result_file_sha256 != proof.compiler_result_file_sha256
    assert byte_changed_proof != proof

    bare = assess_decisive_evaluation_readiness_v1(
        config=real_contract.config,
        repository_root=real_contract.repository_root,
        assessed_at=_T0 + timedelta(hours=2),
        split_manifest=real_contract.split_manifest,
        development_receipt=real_contract.development_receipt,
        calibration_receipt=real_contract.calibration_receipt,
        trajectory_bundle=result.trajectory_bundle,
    )
    assert bare.status == "blocked"
    assert bare.real_scored_run_candidate is False
    assert {
        "missing_trajectory_compilation_result",
        "missing_trajectory_compilation_source_roster",
        "missing_trajectory_compilation_source_root",
    } <= set(bare.blockers)


def test_compiler_lineage_result_symlink_fails_closed(
    real_contract: RealCompilerContract,
    tmp_path: Path,
) -> None:
    result = _compile(real_contract)
    target = tmp_path / "compilation-result-target.json"
    atomic_write_json(target, result)
    link = tmp_path / "compilation-result.json"
    link.symlink_to(target.name)
    with pytest.raises(
        DecisiveCompilationLineageV1Error,
        match="source_symlink:trajectory_compilation_result",
    ):
        replay_decisive_compilation_lineage_v1(
            config=real_contract.config,
            split_manifest=real_contract.split_manifest,
            development_receipt=real_contract.development_receipt,
            calibration_receipt=real_contract.calibration_receipt,
            trajectory_bundle=result.trajectory_bundle,
            compiler_result_path=link,
            source_roster_path=real_contract.source_roster_path,
            source_root=real_contract.source_root,
            repository_root=real_contract.repository_root,
        )


def test_self_rehashed_compiler_lineage_projection_forgery_is_rejected(
    real_contract: RealCompilerContract,
) -> None:
    payload = _compile(real_contract).model_dump(mode="json")
    receipt = payload["compilation_receipt"]
    lineage = receipt["compilation_lineage_identity"]
    lineage["source_roster_sha256"] = "f" * 64
    lineage_without_hash = {
        key: value for key, value in lineage.items() if key != "identity_sha256"
    }
    lineage["identity_sha256"] = hash_canonical(lineage_without_hash)
    receipt_without_hash = {
        key: value for key, value in receipt.items() if key != "compilation_sha256"
    }
    receipt["compilation_sha256"] = hash_canonical(receipt_without_hash)
    result_without_hash = {key: value for key, value in payload.items() if key != "result_sha256"}
    payload["result_sha256"] = hash_canonical(result_without_hash)
    with pytest.raises(
        ValueError,
        match="compilation_lineage_projection_mismatch",
    ):
        DecisiveTrajectoryCompilationResultV1.model_validate(payload)


def test_cli_compiles_and_validates_without_opening_reference_labels(
    real_contract: RealCompilerContract,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.json"
    split_path = tmp_path / "split.json"
    development_path = tmp_path / "development.json"
    calibration_path = tmp_path / "calibration.json"
    for path, value in (
        (config_path, real_contract.config),
        (split_path, real_contract.split_manifest),
        (development_path, real_contract.development_receipt),
        (calibration_path, real_contract.calibration_receipt),
    ):
        atomic_write_json(path, value)
    roster_path = tmp_path / "source-roster.json"
    assert (
        compiler_cli_main(
            [
                "freeze-roster",
                "--split-manifest",
                str(split_path),
                "--source-root",
                str(real_contract.source_root),
                "--workspace",
                (f"{real_contract.manifest.question_id}=workspaces/evaluation-question"),
                "--adjudication-package",
                f"{real_contract.manifest.question_id}=adjudication/package.json",
                "--certificate",
                f"{real_contract.manifest.question_id}=certificates/baseline.json",
                "--output",
                str(roster_path),
            ]
        )
        == 0
    )
    assert '"evaluation_reference_labels_present": false' in capsys.readouterr().out
    bundle_path = tmp_path / "trajectory-bundle.json"
    receipt_path = tmp_path / "compilation-receipt.json"
    common = [
        "--config",
        str(config_path),
        "--split-manifest",
        str(split_path),
        "--development-receipt",
        str(development_path),
        "--calibration-receipt",
        str(calibration_path),
        "--source-roster",
        str(roster_path),
        "--source-root",
        str(real_contract.source_root),
        "--repository-root",
        str(real_contract.repository_root),
    ]
    assert (
        compiler_cli_main(
            [
                "compile",
                *common,
                "--compiled-at",
                (_T0 + timedelta(hours=1)).isoformat(),
                "--output-bundle",
                str(bundle_path),
                "--output-receipt",
                str(receipt_path),
            ]
        )
        == 0
    )
    assert '"evaluation_reference_labels_opened": false' in capsys.readouterr().out
    assert (
        compiler_cli_main(
            [
                "validate",
                *common,
                "--receipt",
                str(receipt_path),
            ]
        )
        == 0
    )
    assert '"external_replay": "passed"' in capsys.readouterr().out


def test_cli_rejects_real_roster_without_adjudication_replay_package(
    real_contract: RealCompilerContract,
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "split.json"
    atomic_write_json(split_path, real_contract.split_manifest)
    with pytest.raises(
        ValueError,
        match="evaluation_workspace_roster_incomplete",
    ):
        compiler_cli_main(
            [
                "freeze-roster",
                "--split-manifest",
                str(split_path),
                "--source-root",
                str(real_contract.source_root),
                "--workspace",
                f"{real_contract.manifest.question_id}=workspaces/evaluation-question",
                "--certificate",
                f"{real_contract.manifest.question_id}=certificates/baseline.json",
                "--output",
                str(tmp_path / "forbidden-roster.json"),
            ]
        )


@pytest.mark.parametrize("relative_path", [".", "adjudication/decision.json\n"])
def test_adjudication_artifact_locator_rejects_ambiguous_or_control_paths(
    relative_path: str,
) -> None:
    with pytest.raises(ValueError, match="adjudication_replay_path_invalid"):
        ExactAdjudicationArtifactLocatorV1(
            relative_path=relative_path,
            expected_file_sha256="a" * 64,
        )


def test_raw_adjudication_byte_forgery_fails_closed(
    real_contract: RealCompilerContract,
    tmp_path: Path,
) -> None:
    copied_root = tmp_path / "sources"
    shutil.copytree(real_contract.source_root, copied_root)
    decision_path = copied_root / real_contract.adjudication_artifact_paths[
        "decision-reviewer-1"
    ].relative_to(real_contract.source_root)
    decision_path.write_text(
        decision_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        DecisiveTrajectoryCompilerV1Error,
        match="adjudication_artifact_hash_mismatch",
    ):
        _compile(replace(real_contract, source_root=copied_root))


def test_adjudication_workflow_symlink_fails_closed(
    real_contract: RealCompilerContract,
    tmp_path: Path,
) -> None:
    copied_root = tmp_path / "sources"
    shutil.copytree(real_contract.source_root, copied_root)
    decision_path = copied_root / real_contract.adjudication_artifact_paths[
        "decision-reviewer-1"
    ].relative_to(real_contract.source_root)
    target_path = decision_path.with_name("decision-reviewer-1-target.json")
    decision_path.rename(target_path)
    decision_path.symlink_to(target_path.name)
    with pytest.raises(
        DecisiveTrajectoryCompilerV1Error,
        match="source_symlink",
    ):
        _compile(replace(real_contract, source_root=copied_root))


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    [
        ("question_id", "forged-question"),
        ("item_id", "forged-item"),
        ("reviewer_id", "unregistered-reviewer"),
    ],
)
def test_rebound_decision_membership_and_reviewer_roster_forgery_fails_closed(
    real_contract: RealCompilerContract,
    tmp_path: Path,
    field_name: str,
    forged_value: str,
) -> None:
    source = IndependentReviewerDecisionV1.model_validate_json(
        real_contract.adjudication_artifact_paths["decision-reviewer-1"].read_text(encoding="utf-8")
    )
    forged = source.model_copy(update={field_name: forged_value})
    rebound = _contract_with_rebound_adjudication_artifact(
        real_contract,
        tmp_path=tmp_path,
        artifact_name="decision-reviewer-1",
        replacement_artifact=forged,
    )
    with pytest.raises(
        DecisiveTrajectoryCompilerV1Error,
        match="adjudication_decision_membership_mismatch",
    ):
        _compile(rebound)


def test_rebound_resolution_cannot_override_receipt_payload_hash(
    real_contract: RealCompilerContract,
    tmp_path: Path,
) -> None:
    source = AdjudicationResolutionArtifactV1.model_validate_json(
        real_contract.adjudication_artifact_paths["resolution"].read_text(encoding="utf-8")
    )
    forged = source.model_copy(
        update={"resolution_rationale": "Operator-rebound but receipt-unbound resolution."}
    )
    rebound = _contract_with_rebound_adjudication_artifact(
        real_contract,
        tmp_path=tmp_path,
        artifact_name="resolution",
        replacement_artifact=forged,
    )
    with pytest.raises(
        DecisiveTrajectoryCompilerV1Error,
        match="adjudication_receipt_raw_replay_mismatch",
    ):
        _compile(rebound)


def test_reviewer_timing_cannot_predate_action_authorization(
    real_contract: RealCompilerContract,
    tmp_path: Path,
) -> None:
    source = ReviewerTimingEvidenceV1.model_validate_json(
        real_contract.adjudication_artifact_paths["timing-reviewer-1"].read_text(encoding="utf-8")
    )
    forged = source.model_copy(update={"started_at": source.started_at - timedelta(minutes=1)})
    rebound = _contract_with_rebound_adjudication_artifact(
        real_contract,
        tmp_path=tmp_path,
        artifact_name="timing-reviewer-1",
        replacement_artifact=forged,
    )
    with pytest.raises(
        DecisiveTrajectoryCompilerV1Error,
        match="adjudication_timing_membership_mismatch",
    ):
        _compile(rebound)


def test_rebound_timing_sum_must_match_transaction_receipt(
    real_contract: RealCompilerContract,
    tmp_path: Path,
) -> None:
    source = ReviewerTimingEvidenceV1.model_validate_json(
        real_contract.adjudication_artifact_paths["timing-reviewer-1"].read_text(encoding="utf-8")
    )
    forged = source.model_copy(update={"active_person_minutes": 0.5})
    rebound = _contract_with_rebound_adjudication_artifact(
        real_contract,
        tmp_path=tmp_path,
        artifact_name="timing-reviewer-1",
        replacement_artifact=forged,
    )
    with pytest.raises(
        DecisiveTrajectoryCompilerV1Error,
        match="adjudication_receipt_raw_replay_mismatch",
    ):
        _compile(rebound)


def test_embedded_synthetic_certificate_is_rejected_for_real_compilation(
    real_contract: RealCompilerContract,
) -> None:
    manifest, embedded = build_offline_fixture()
    fingerprint = compute_verifier_pipeline_fingerprint(root=real_contract.repository_root)
    certificate = run_verification(
        manifest=manifest,
        corpus=embedded,
        budget_minutes=30,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=real_contract.repository_root,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=_T0,
    )
    identity = real_contract.split_manifest.identities[-1]
    with pytest.raises(
        DecisiveTrajectoryCompilerV1Error,
        match=r"real_source_provenance_required|question_identity_mismatch",
    ):
        _require_real_certificate(certificate=certificate, identity=identity)


@pytest.mark.parametrize(
    ("metadata_key", "metadata_value"),
    [
        ("simulation", True),
        ("synthetic_source_only", True),
        ("fixture_mode", True),
        ("purpose", "planted_simulation_mechanics_only"),
        ("evidence_kind", "diagnostic_proxy"),
    ],
)
def test_explicit_nonempirical_markers_cannot_hide_behind_native_replay(
    real_contract: RealCompilerContract,
    metadata_key: str,
    metadata_value: object,
) -> None:
    certificate = real_contract.baseline_certificate
    corpus = dict(certificate.corpus)
    corpus["metadata"] = {
        **certificate.corpus["metadata"],
        metadata_key: metadata_value,
    }
    tainted = certificate.model_copy(update={"corpus": corpus})
    identity = next(
        row
        for row in real_contract.split_manifest.identities
        if row.question_id == real_contract.manifest.question_id
    )
    with pytest.raises(
        DecisiveTrajectoryCompilerV1Error,
        match="nonempirical_source_marker_forbidden",
    ):
        _require_real_certificate(certificate=tainted, identity=identity)


def test_roster_and_workspace_byte_tamper_fail_external_replay(
    real_contract: RealCompilerContract,
    tmp_path: Path,
) -> None:
    result = _compile(real_contract)
    changed_roster_path = tmp_path / "changed-source-roster.json"
    changed_roster_path.write_text(
        real_contract.source_roster_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        DecisiveTrajectoryCompilerV1Error,
        match="external_replay_mismatch",
    ):
        replay_decisive_trajectory_compilation_v1(
            expected=result,
            config=real_contract.config,
            split_manifest=real_contract.split_manifest,
            development_receipt=real_contract.development_receipt,
            calibration_receipt=real_contract.calibration_receipt,
            source_roster=real_contract.source_roster,
            source_roster_path=changed_roster_path,
            source_root=real_contract.source_root,
            repository_root=real_contract.repository_root,
        )

    # A copied source with one changed rendering byte remains a valid transactional
    # scientific workspace but cannot replay the prior exact source binding.
    copied_root = tmp_path / "sources"
    shutil.copytree(real_contract.source_root, copied_root)
    copied_workspace = copied_root / "workspaces" / "evaluation-question"
    terminal_generation = sorted((copied_workspace / "generations").iterdir())[-1]
    html_path = terminal_generation / "verification-certificate.html"
    html_path.write_text(html_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(
        DecisiveTrajectoryCompilerV1Error,
        match="external_replay_mismatch",
    ):
        replay_decisive_trajectory_compilation_v1(
            expected=result,
            config=real_contract.config,
            split_manifest=real_contract.split_manifest,
            development_receipt=real_contract.development_receipt,
            calibration_receipt=real_contract.calibration_receipt,
            source_roster=real_contract.source_roster,
            source_roster_path=real_contract.source_roster_path,
            source_root=copied_root,
            repository_root=real_contract.repository_root,
        )


def test_policy_union_is_compact_and_missing_visited_branch_fails_closed() -> None:
    pipeline_sha256 = _fixture_hash("pipeline")
    development = freeze_question_identity_v1(
        split="development",
        question_id="union-development",
        claim_id="union-development-claim",
        domain="union-domain",
        population_id="union-population",
        pipeline_sha256=pipeline_sha256,
        corpus_sha256="1" * 64,
        paper_ids=["union-development-paper"],
        cohort_ids=["union-development-cohort"],
    )
    calibration = freeze_question_identity_v1(
        split="calibration",
        question_id="union-calibration",
        claim_id="union-calibration-claim",
        domain="union-domain",
        population_id="union-population",
        pipeline_sha256=pipeline_sha256,
        corpus_sha256="2" * 64,
        paper_ids=["union-calibration-paper"],
        cohort_ids=["union-calibration-cohort"],
    )
    identity = freeze_question_identity_v1(
        split="evaluation",
        question_id="union-evaluation",
        claim_id="union-evaluation-claim",
        domain="union-domain",
        population_id="union-population",
        pipeline_sha256=pipeline_sha256,
        corpus_sha256="3" * 64,
        paper_ids=["union-evaluation-paper"],
        cohort_ids=["union-evaluation-cohort"],
    )
    development_receipt = freeze_fit_stage_receipt_v1(
        stage="development_optimizer_fit",
        identities=[development],
        pipeline_sha256=pipeline_sha256,
        input_manifest_sha256="4" * 64,
        label_source="planted_simulation",
        frozen_optimizer_or_policy_sha256="5" * 64,
        frozen_threshold_or_bounds_sha256=None,
        completed_at=_T0,
    )
    calibration_receipt = freeze_fit_stage_receipt_v1(
        stage="calibration_policy_and_threshold_freeze",
        identities=[calibration],
        pipeline_sha256=pipeline_sha256,
        input_manifest_sha256="4" * 64,
        label_source="planted_simulation",
        frozen_optimizer_or_policy_sha256="6" * 64,
        frozen_threshold_or_bounds_sha256="7" * 64,
        completed_at=_T0 + timedelta(minutes=1),
    )
    provenance = freeze_decisive_policy_input_provenance_v1(
        development_receipt=development_receipt,
        calibration_receipt=calibration_receipt,
    )
    item_ids = [f"{identity.question_id}-{suffix}" for suffix in ("a", "b", "c")]
    events = [
        freeze_question_audit_event(
            item_id=item_id,
            disposition=(
                AuditDisposition.CORRECTED if item_id.endswith("-a") else AuditDisposition.CONFIRMED
            ),
            completed_at=_T0 + timedelta(minutes=index),
            realized_minutes=float(4 - index),
            cost_basis=AuditCostBasis.SIMULATED_MINUTES,
            adjudicator_count=1,
            protocol_sha256="8" * 64,
            artifact_sha256=hash_canonical({"union_event": item_id}),
            correction_sha256=(
                hash_canonical({"union_correction": item_id}) if item_id.endswith("-a") else None
            ),
        )
        for index, item_id in enumerate(item_ids, start=1)
    ]
    states = [
        _fixture_state(
            question_id=identity.question_id,
            audit_sequence=sequence,
            item_ids=item_ids,
            truth="supported",
        )
        for length in range(4)
        for sequence in permutations(item_ids, length)
    ]
    trajectory = freeze_question_trajectory_v1(
        question_identity=identity,
        evidence_kind=BenchmarkEvidenceKind.SIMULATION,
        policy_input_provenance=provenance,
        audit_events=events,
        replay_states=states,
    )
    config = freeze_decisive_evaluation_config_v1(
        budgets_minutes_per_question=(2.0, 4.0, 6.0),
        bootstrap_draws=100,
    )
    required = _required_policy_prefix_union(config=config, trajectory=trajectory)
    all_prefixes = {tuple(row.audit_sequence) for row in states}
    assert set(required) < all_prefixes
    assert () in required
    assert tuple(item_ids) in required
    removable = next(row for row in required if row not in {(), tuple(item_ids)})
    incomplete = freeze_question_trajectory_v1(
        question_identity=identity,
        evidence_kind=BenchmarkEvidenceKind.SIMULATION,
        policy_input_provenance=provenance,
        audit_events=events,
        replay_states=[row for row in states if tuple(row.audit_sequence) != removable],
    )
    with pytest.raises(
        ValueError,
        match=r"replay_prefix_missing|post_audit_rerun_missing",
    ):
        _required_policy_prefix_union(config=config, trajectory=incomplete)


def test_final_condition_v7_replays_with_exact_normalized_artifact(
    tmp_path: Path,
    final_condition_v7_certificate: FinalConditionVerificationCertificateV7,
) -> None:
    source_root = tmp_path / "sources"
    certificate_root = source_root / "certificates"
    condition_root = source_root / "conditions"
    certificate_root.mkdir(parents=True)
    condition_root.mkdir()
    certificate_path = certificate_root / "final-v7.json"
    atomic_write_json(certificate_path, final_condition_v7_certificate)
    locator = freeze_verifier_certificate_locator_v1(
        relative_path="certificates/final-v7.json",
        expected_file_sha256=hashlib.sha256(certificate_path.read_bytes()).hexdigest(),
        expected_certificate_sha256=(final_condition_v7_certificate.certificate_sha256),
    )
    snapshot = _snapshot_verifier_certificate(
        source_root=source_root,
        locator=locator,
    )
    assert isinstance(
        snapshot.replay_candidate.certificate,
        FinalConditionVerificationCertificateV7,
    )
    assert snapshot.replay_candidate.replay.claim_classification == "condition_dependent"

    source = final_condition_v7_certificate.source_certificate_v6
    model = source.condition_frozen_model
    assert model.selected_moderator is not None
    assert model.frozen_positive_level is not None
    assert model.frozen_negative_level is not None
    artifact = freeze_normalized_condition_set_artifact_v1(
        question_id=source.release_assessment.question_id,
        condition_target_sha256=(source.condition_calibration_projection.condition_target_sha256),
        selected_moderator=model.selected_moderator,
        positive_effect_level=model.frozen_positive_level,
        negative_effect_level=model.frozen_negative_level,
    )
    artifact_path = condition_root / "normalized.json"
    atomic_write_json(artifact_path, artifact)
    declaration = freeze_condition_set_source_binding_v1(
        certificate_sha256=final_condition_v7_certificate.certificate_sha256,
        relative_path="conditions/normalized.json",
        expected_file_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        condition_set_artifact_sha256=artifact.artifact_sha256,
    )
    by_certificate, bindings = _bind_condition_artifacts(
        candidates=[snapshot.replay_candidate],
        declared_bindings=[declaration],
        source_root=source_root,
    )
    assert by_certificate == {
        final_condition_v7_certificate.certificate_sha256: artifact.artifact_sha256
    }
    assert len(bindings) == 1
    assert bindings[0].file_sha256 == declaration.expected_file_sha256

    symlink_path = condition_root / "normalized-link.json"
    symlink_path.symlink_to(artifact_path)
    symlink_declaration = freeze_condition_set_source_binding_v1(
        certificate_sha256=final_condition_v7_certificate.certificate_sha256,
        relative_path="conditions/normalized-link.json",
        expected_file_sha256=declaration.expected_file_sha256,
        condition_set_artifact_sha256=artifact.artifact_sha256,
    )
    with pytest.raises(
        DecisiveTrajectoryCompilerV1Error,
        match="source_symlink",
    ):
        _bind_condition_artifacts(
            candidates=[snapshot.replay_candidate],
            declared_bindings=[symlink_declaration],
            source_root=source_root,
        )


def test_final_condition_artifact_rejects_forged_semantics_and_declared_hash(
    tmp_path: Path,
    final_condition_v7_certificate: FinalConditionVerificationCertificateV7,
) -> None:
    source_root = tmp_path / "sources"
    condition_root = source_root / "conditions"
    condition_root.mkdir(parents=True)
    source = final_condition_v7_certificate.source_certificate_v6
    model = source.condition_frozen_model
    assert model.selected_moderator is not None
    assert model.frozen_positive_level is not None
    assert model.frozen_negative_level is not None
    forged = freeze_normalized_condition_set_artifact_v1(
        question_id=source.release_assessment.question_id,
        condition_target_sha256=(source.condition_calibration_projection.condition_target_sha256),
        selected_moderator=model.selected_moderator,
        positive_effect_level=f"{model.frozen_positive_level}-forged",
        negative_effect_level=model.frozen_negative_level,
    )
    artifact_path = condition_root / "forged.json"
    atomic_write_json(artifact_path, forged)
    file_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    replay = freeze_question_replay_state_from_certificate(final_condition_v7_certificate)
    candidate = _ReplayCandidate(
        certificate=final_condition_v7_certificate,
        replay=replay,
        source_kind="standalone_verifier_certificate",
        source_container_sha256="a" * 64,
    )
    forged_binding = freeze_condition_set_source_binding_v1(
        certificate_sha256=final_condition_v7_certificate.certificate_sha256,
        relative_path="conditions/forged.json",
        expected_file_sha256=file_sha256,
        condition_set_artifact_sha256=forged.artifact_sha256,
    )
    with pytest.raises(
        DecisiveTrajectoryCompilerV1Error,
        match="condition_set_certificate_mismatch",
    ):
        _bind_condition_artifacts(
            candidates=[candidate],
            declared_bindings=[forged_binding],
            source_root=source_root,
        )

    false_hash_binding = freeze_condition_set_source_binding_v1(
        certificate_sha256=final_condition_v7_certificate.certificate_sha256,
        relative_path="conditions/forged.json",
        expected_file_sha256=file_sha256,
        condition_set_artifact_sha256="f" * 64,
    )
    with pytest.raises(
        DecisiveTrajectoryCompilerV1Error,
        match="condition_set_locator_identity_mismatch",
    ):
        _bind_condition_artifacts(
            candidates=[candidate],
            declared_bindings=[false_hash_binding],
            source_root=source_root,
        )


def test_v5_only_condition_state_requires_final_v7(
    real_contract: RealCompilerContract,
) -> None:
    replay = freeze_question_replay_state_from_certificate(
        real_contract.baseline_certificate
    ).model_copy(update={"claim_classification": "condition_dependent"})
    candidate = _ReplayCandidate(
        certificate=real_contract.baseline_certificate,
        replay=replay,
        source_kind="standalone_verifier_certificate",
        source_container_sha256="b" * 64,
    )
    with pytest.raises(
        DecisiveTrajectoryCompilerV1Error,
        match="final_condition_v7_required",
    ):
        _replace_unfinalized_v5_condition_states([candidate])
