from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from literature_multiverse.adaptive_calibration import (
    AdaptivePolicyContext,
    assess_adaptive_release_candidate,
    calibrate_adaptive_first_release,
    fit_adaptive_development,
    freeze_adaptive_policy_arm_trajectory,
    freeze_adaptive_policy_context,
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
from literature_multiverse.item_risk_artifacts import ItemRiskScoringRunReceipt
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.sequential_verification import (
    freeze_selected_adjudication,
    freeze_state_expectation,
    resolve_selected_audit_candidate,
)
from literature_multiverse.verifier import (
    ClaimManifest,
    CorpusLoadResult,
    CorpusProvenanceAssurance,
    VerificationContractError,
    build_offline_fixture,
    compute_candidate_runner_sha256,
    compute_synthesis_runner_sha256,
    compute_verifier_pipeline_fingerprint,
    prepare_verification_scientific_state,
    run_verification,
    sequential_candidates_from_prepared_state,
)

# These fixture builders are shared only to keep this adversarial test focused on
# certificate/release integrity instead of duplicating native-manifest and risk-receipt
# construction.  The production code under test is imported only from src/ above.
from test_unified_verifier import (  # type: ignore[import-not-found]
    _adaptive_release_contract,
    _artifact_backed_item_risk_contract,
    _release_calibration_bundle,
    _source_replayed_fixture_contract,
)

_ROOT = Path(__file__).resolve().parents[1]
_BUDGET = 30.0


def _release_eligible_fixture() -> tuple[
    ClaimManifest,
    CorpusLoadResult,
    object,
    ItemRiskScoringRunReceipt,
]:
    manifest, fixture = build_offline_fixture()
    manifest = manifest.model_copy(
        update={
            "release": manifest.release.model_copy(
                update={"require_prediction_interval_stability": False}
            ),
            "audit_guard": manifest.audit_guard.model_copy(
                update={
                    "block_counterfactual_conclusion_flips": False,
                    "max_unresolved_item_cell_ucl_sum": 1.0,
                    "max_unresolved_expected_claim_loss": 10.0,
                    "max_unresolved_item_influence": 1.0,
                }
            ),
        }
    )
    fingerprint = compute_verifier_pipeline_fingerprint(root=_ROOT)
    replay_sha256 = "e" * 64
    eligibility, metadata, extraction_context = _source_replayed_fixture_contract(
        manifest=manifest,
        fixture=fixture,
        pipeline_sha256=fingerprint.pipeline_sha256,
        replay_sha256=replay_sha256,
    )
    corpus = type(fixture)(
        corpus_id=manifest.question_id,
        source_label="/frozen/native/package.json",
        source_format="typed_evidence_grounding_package_json",
        source_sha256=fixture.source_sha256,
        graph=fixture.graph,
        eligibility=eligibility,
        adapter_issues=(),
        metadata=metadata,
        extraction_context=extraction_context,
        provenance_assurance=CorpusProvenanceAssurance(
            status="source_replayed_native_grounding",
            reason="Fixture emulates a successfully replayed native package.",
            replay_sha256=replay_sha256,
        ),
    )
    _, _, scoring_receipt = _artifact_backed_item_risk_contract(
        manifest=manifest,
        corpus=corpus,
        fingerprint=fingerprint,
        repository_root=_ROOT,
    )
    return manifest, corpus, fingerprint, scoring_receipt


def _resolve_one_no_change(
    *,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
    fingerprint: object,
    scoring_receipt: ItemRiskScoringRunReceipt,
    initial: VerificationCertificate,
):
    state = initial.sequential_audit_state
    assert state is not None and state.session.active_action is not None
    prepared = prepare_verification_scientific_state(
        manifest=manifest,
        graph=state.graph,
        pipeline_verification=scoring_receipt.pipeline_verification,
        item_risk_calibration_bundle=scoring_receipt.calibration_bundle,
        item_risk_candidates=list(scoring_receipt.candidates),
    )
    refreshed_candidates = sequential_candidates_from_prepared_state(
        manifest=manifest,
        prepared=prepared,
    )
    expected = freeze_state_expectation(state)
    adjudication = freeze_selected_adjudication(
        state,
        expected=expected,
        provenance="benchmark_adjudication",
        adjudicator_count=1,
        protocol_sha256="a" * 64,
        payload_sha256="b" * 64,
        completed_at=datetime(2026, 8, 27, 12, 1, tzinfo=UTC),
        realized_cost=1.0,
    )
    result = resolve_selected_audit_candidate(
        state,
        expected=expected,
        adjudication=adjudication,
        disposition=CorrectionDisposition.NO_CHANGE,
        corrected_graph=None,
        correction_provenance="benchmark_adjudication",
        correction_protocol_sha256="c" * 64,
        external_correction_payload_sha256="d" * 64,
        synthesis_runner_sha256=compute_synthesis_runner_sha256(
            manifest=manifest,
            pipeline_sha256=scoring_receipt.pipeline_verification.expected_pipeline_sha256,
        ),
        candidate_runner_sha256=compute_candidate_runner_sha256(
            manifest=manifest,
            pipeline_sha256=scoring_receipt.pipeline_verification.expected_pipeline_sha256,
        ),
        rerun_synthesis=lambda _graph: prepared.synthesis,
        rerun_candidates=lambda _graph, _synthesis, _session: refreshed_candidates,
    )
    return result.state


def _released_adaptive_certificate() -> tuple[
    ClaimManifest,
    CorpusLoadResult,
    object,
    ItemRiskScoringRunReceipt,
    VerificationCertificate,
    VerificationCertificate,
]:
    manifest, corpus, fingerprint, scoring_receipt = _release_eligible_fixture()
    fixed_bundle = _release_calibration_bundle(
        manifest=manifest,
        pipeline_sha256=scoring_receipt.pipeline_verification.expected_pipeline_sha256,
    )
    preselection = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=_BUDGET,
        frozen_calibration_bundle=fixed_bundle,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=_ROOT,
        item_risk_scoring_receipt=scoring_receipt,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )
    evaluated = preselection.production_stop_decision.evaluated_state
    assert evaluated is not None and evaluated.session.active_action is None
    bundle, _ = _adaptive_release_contract(
        manifest=manifest,
        preselection_certificate=preselection,
        pipeline_sha256=scoring_receipt.pipeline_verification.expected_pipeline_sha256,
        budget_minutes=_BUDGET,
    )
    released = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=_BUDGET,
        adaptive_calibration_bundle=bundle,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=_ROOT,
        item_risk_scoring_receipt=scoring_receipt,
        generated_at=datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
    )
    assert released.status == "released"
    return manifest, corpus, fingerprint, scoring_receipt, preselection, released


def _adaptive_contract_for_context(
    *,
    manifest: ClaimManifest,
    preselection: VerificationCertificate,
    context: AdaptivePolicyContext,
):
    projected = preselection_state_from_certificate_v5(preselection)

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
            score_features=projected.score_features,
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
        corpus = freeze_complete_corpus_identity(
            corpus_id=f"context-redteam-corpus-{index}",
            corpus_source_sha256=f"{index + 50:x}".rjust(64, "0"),
            corpus_cutoff=manifest.protocol.corpus_cutoff,
            publication_ids=[f"context-redteam-publication-{index}"],
            source_manifest_sha256=f"{index + 60:x}".rjust(64, "0"),
        )
        visible = freeze_policy_visible_question_trajectory(
            question_id=f"context-redteam-question-{index}",
            split=split,  # type: ignore[arg-type]
            population_id=manifest.population_id,
            domain=manifest.domain,
            corpus=corpus,
            arms=[arm],
        )
        reference = freeze_question_reference_verdict(
            question_id=f"context-redteam-question-{index}",
            verdict=projected.claim_decision,
            label_source="expert_adjudication",
            adjudication_protocol_sha256="7" * 64,
            adjudication_artifact_sha256=f"{index + 80:x}".rjust(64, "0"),
        )
        return join_labeled_question_trajectory(visible=visible, reference=reference)

    development = [labeled_row(index, "development") for index in range(1, 5)]
    calibration = [labeled_row(index, "calibration") for index in range(5, 9)]
    freeze = fit_adaptive_development(
        development,
        policy_contexts=[context],
        alpha=0.99,
        delta=0.5,
        calibration_visible_trajectories=[row.visible for row in calibration],
        candidate_thresholds={context.policy_arm_id: [1.0]},
        seed=11,
    )
    bundle = calibrate_adaptive_first_release(freeze, calibration)
    candidate = freeze_prospective_adaptive_candidate(
        question_id=manifest.question_id,
        population_id=manifest.population_id,
        domain=manifest.domain,
        policy_arm_id=context.policy_arm_id,
        policy_context_sha256=context.policy_context_sha256,
        corpus=preselection.complete_corpus_identity,
        observed_states=[projected],
    )
    return bundle, candidate, assess_adaptive_release_candidate(candidate, bundle)


def _rehash(payload: dict[str, object], field: str) -> None:
    payload[field] = hash_canonical(
        {key: value for key, value in payload.items() if key != field}
    )


def test_run_verification_rejects_rewritten_prior_adaptive_prefix_history() -> None:
    manifest, corpus, fingerprint, scoring_receipt = _release_eligible_fixture()
    initial = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=_BUDGET,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=_ROOT,
        item_risk_scoring_receipt=scoring_receipt,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )
    resolved = _resolve_one_no_change(
        manifest=manifest,
        corpus=corpus,
        fingerprint=fingerprint,
        scoring_receipt=scoring_receipt,
        initial=initial,
    )
    current = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=_BUDGET,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=_ROOT,
        item_risk_scoring_receipt=scoring_receipt,
        sequential_audit_state=resolved,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
    )
    evaluated = current.production_stop_decision.evaluated_state
    assert evaluated is not None and evaluated.session.active_action is None
    bundle, _ = _adaptive_release_contract(
        manifest=manifest,
        preselection_certificate=current,
        prior_preselection_certificates=(initial,),
        pipeline_sha256=scoring_receipt.pipeline_verification.expected_pipeline_sha256,
        budget_minutes=_BUDGET,
    )

    # Adaptive calibration must be present at the first selection. A caller cannot
    # retrofit a prospective trajectory over a previously non-adaptive audit path.
    with pytest.raises(
        VerificationContractError,
        match="adaptive_calibration_cannot_activate_after_state_genesis",
    ):
        run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=_BUDGET,
            adaptive_calibration_bundle=bundle,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=_ROOT,
            item_risk_scoring_receipt=scoring_receipt,
            sequential_audit_state=evaluated,
            generated_at=datetime(2026, 8, 27, 12, 3, tzinfo=UTC),
        )


def test_certificate_rejects_rehashed_adaptive_context_with_wrong_budget() -> None:
    manifest, _, _, _, _preselection, released = _released_adaptive_certificate()
    context = released.adaptive_policy_context
    assert context is not None
    wrong_context = freeze_adaptive_policy_context(
        policy_arm_id=context.policy_arm_id,
        population_id=context.population_id,
        pipeline_sha256=context.pipeline_sha256,
        allocation_policy=context.allocation_policy,
        budget_minutes=_BUDGET + 1.0,
        release_config=context.release_config,
        audit_config=context.audit_config,
        target_semantics=context.target_semantics,
        corpus_protocol_context=context.corpus_protocol_context,
        score_feature_names=context.score_feature_names,
    )
    bundle, candidate, assessment = _adaptive_contract_for_context(
        manifest=manifest,
        preselection=released,
        context=wrong_context,
    )
    assert assessment.status == "released"

    forged = deepcopy(released.model_dump(mode="json"))
    forged["adaptive_policy_context"] = wrong_context.model_dump(mode="json")
    forged["adaptive_calibration_bundle"] = bundle.model_dump(mode="json")
    forged["adaptive_release_candidate"] = candidate.model_dump(mode="json")
    forged["adaptive_prospective_assessment"] = assessment.model_dump(mode="json")

    release = forged["release_assessment"]
    assert isinstance(release, dict)
    calibration = release["calibration"]
    assert isinstance(calibration, dict)
    calibration.update(
        {
            "status": assessment.status,
            "reason": assessment.reason,
            "frozen_bundle_sha256": bundle.bundle_sha256,
            "release_candidate_sha256": candidate.candidate_sha256,
            "prospective_assessment_sha256": assessment.assessment_sha256,
            "policy_context_sha256": wrong_context.policy_context_sha256,
            "scalar_risk_score": assessment.scalar_risk_score,
            "threshold": assessment.threshold,
            "label_source": bundle.label_source,
            "guarantee_scope": assessment.guarantee_scope,
        }
    )
    _rehash(release, "decision_sha256")

    stop = forged["production_stop_decision"]
    assert isinstance(stop, dict)
    stop["release_assessment"] = deepcopy(release)
    _rehash(stop, "decision_sha256")

    lineage = forged["lineage"]
    assert isinstance(lineage, list)
    adaptive_stage = lineage[6]
    assert isinstance(adaptive_stage, dict)
    adaptive_stage["input_sha256s"] = dict(
        sorted(
            {
                "adaptive_calibration_bundle": bundle.bundle_sha256,
                "adaptive_policy_context": wrong_context.policy_context_sha256,
                "adaptive_release_candidate": candidate.candidate_sha256,
                "complete_corpus_identity": candidate.corpus.membership_sha256,
            }.items()
        )
    )
    adaptive_stage["output_sha256s"] = {
        "adaptive_prospective_assessment": assessment.assessment_sha256
    }

    release_stage = lineage[7]
    assert isinstance(release_stage, dict)
    release_inputs = hash_canonical(
        {
            "audit_candidates": forged["audit_candidates"],
            "audit_receipts": release["audit"]["resolution_receipts"],
            "sequential_audit_state": forged["sequential_audit_state"],
            "budget_minutes": release["audit"]["budget"],
            "complete_corpus_identity": forged["complete_corpus_identity"],
            "item_risk_scoring_receipt": forged["item_risk_scoring_receipt"],
            "adaptive_policy_context": forged["adaptive_policy_context"],
            "adaptive_calibration_bundle": forged["adaptive_calibration_bundle"],
            "adaptive_release_candidate": forged["adaptive_release_candidate"],
            "adaptive_prospective_assessment": forged[
                "adaptive_prospective_assessment"
            ],
            "pipeline_sha256": release["pipeline_sha256"],
            "production_stop_decision_sha256": stop["decision_sha256"],
            "target": release["target"],
        }
    )
    release_stage["input_sha256s"] = {"release_inputs": release_inputs}
    release_stage["output_sha256s"] = {"release_decision": release["decision_sha256"]}

    verification = forged["pipeline_verification"]
    assert isinstance(verification, dict)
    receipt = forged["item_risk_scoring_receipt"]
    assert isinstance(receipt, dict)
    complete = forged["complete_corpus_identity"]
    assert isinstance(complete, dict)
    run_identity = hash_canonical(
        {
            "claim_manifest_sha256": forged["claim_manifest_sha256"],
            "corpus_sha256": forged["corpus_sha256"],
            "source_evidence_graph_sha256": forged[
                "source_evidence_graph_sha256"
            ],
            "evidence_graph_sha256": forged["evidence_graph_sha256"],
            "release_decision_sha256": release["decision_sha256"],
            "pipeline_verification_sha256": verification["verification_sha256"],
            "production_stop_decision_sha256": stop["decision_sha256"],
            "complete_corpus_membership_sha256": complete["membership_sha256"],
            "item_risk_scoring_receipt_sha256": receipt["receipt_sha256"],
            "adaptive_policy_context_sha256": wrong_context.policy_context_sha256,
            "adaptive_calibration_bundle_sha256": bundle.bundle_sha256,
            "adaptive_release_candidate_sha256": candidate.candidate_sha256,
            "adaptive_prospective_assessment_sha256": assessment.assessment_sha256,
        }
    )
    forged["run_id"] = f"verify-{run_identity[:16]}"
    _rehash(forged, "certificate_sha256")

    with pytest.raises(
        ValidationError,
        match="adaptive_policy_context_manifest_mismatch",
    ):
        VerificationCertificate.model_validate(forged)
