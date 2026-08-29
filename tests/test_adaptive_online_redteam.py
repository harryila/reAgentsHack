from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import literature_multiverse.claim_release as claim_release_module
import literature_multiverse.verifier as verifier_module
from literature_multiverse.adaptive_calibration import (
    calibrate_adaptive_first_release,
    fit_adaptive_development,
    freeze_adaptive_policy_arm_trajectory,
    freeze_adaptive_preselection_state,
    freeze_complete_corpus_identity,
    freeze_policy_visible_question_trajectory,
    freeze_preselection_state_from_production_components,
    freeze_prospective_adaptive_candidate,
    freeze_question_reference_verdict,
    join_labeled_question_trajectory,
    preselection_state_from_certificate_v5,
)
from literature_multiverse.audit_session import CorrectionDisposition
from literature_multiverse.certificate import VerificationCertificate
from literature_multiverse.claim_release import (
    ClaimReleaseContractError,
    assess_claim_release,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.sequential_verification import (
    SequentialVerificationContractError,
    freeze_selected_adjudication,
    freeze_state_expectation,
    resolve_selected_audit_candidate,
    select_next_audit_candidate,
)
from literature_multiverse.verifier import (
    CorpusLoadResult,
    CorpusProvenanceAssurance,
    VerificationContractError,
    build_offline_fixture,
    build_verifier_adaptive_policy_context,
    compute_candidate_runner_sha256,
    compute_synthesis_runner_sha256,
    compute_verifier_pipeline_fingerprint,
    prepare_verification_scientific_state,
    run_verification,
    sequential_candidates_from_prepared_state,
)

# These builders produce a release-eligible, source-replayed fixture and a complete
# expert-labelled adaptive calibration bundle. Keeping the adversarial operations in
# this file makes it clear that only public production contracts are being attacked.
from test_certificate_v5_redteam import (  # type: ignore[import-not-found]
    _adaptive_contract_for_context,
)
from test_unified_verifier import (  # type: ignore[import-not-found]
    _adaptive_release_contract,
    _artifact_backed_item_risk_contract,
    _source_replayed_fixture_contract,
)

_ROOT = Path(__file__).resolve().parents[1]
_BUDGET = 30.0
_T0 = datetime(2026, 8, 27, 12, tzinfo=UTC)


def _prospective_fixture():
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
    corpus = CorpusLoadResult(
        corpus_id=manifest.question_id,
        source_label="/frozen/native/package.json",
        source_format="typed_evidence_grounding_package_json",
        source_sha256=fixture.source_sha256,
        graph=fixture.graph,
        eligibility=eligibility,
        adapter_issues=(),
        metadata=metadata,
        provenance_assurance=CorpusProvenanceAssurance(
            status="source_replayed_native_grounding",
            reason="Fixture emulates a successfully replayed native package.",
            replay_sha256=replay_sha256,
        ),
        extraction_context=extraction_context,
    )
    _, _, scoring_receipt = _artifact_backed_item_risk_contract(
        manifest=manifest,
        corpus=corpus,
        fingerprint=fingerprint,
        repository_root=_ROOT,
    )
    calibration_shadow = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=_BUDGET,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=_ROOT,
        item_risk_scoring_receipt=scoring_receipt,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=_T0,
    )
    bundle, _ = _adaptive_release_contract(
        manifest=manifest,
        preselection_certificate=calibration_shadow,
        pipeline_sha256=fingerprint.pipeline_sha256,
        budget_minutes=_BUDGET,
    )
    context = build_verifier_adaptive_policy_context(
        manifest=manifest,
        pipeline_sha256=fingerprint.pipeline_sha256,
        budget_minutes=_BUDGET,
        policy_arm_id="production-adaptive",
    )
    assert context in bundle.development_freeze.policy_contexts
    preselection = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=_BUDGET,
        adaptive_calibration_bundle=bundle,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=_ROOT,
        item_risk_scoring_receipt=scoring_receipt,
        generated_at=datetime(2026, 8, 27, 12, 0, 30, tzinfo=UTC),
    )
    evaluated = preselection.production_stop_decision.evaluated_state
    assert preselection.status == "released"
    assert evaluated is not None
    assert evaluated.session.active_action is None
    assert evaluated.adaptive_policy_context_sha256 == context.policy_context_sha256
    assert evaluated.adaptive_calibration_bundle_sha256 == bundle.bundle_sha256
    return (
        manifest,
        corpus,
        fingerprint,
        scoring_receipt,
        preselection,
        evaluated,
        bundle,
        context,
    )


def _select_with_checkpoint(*, state, checkpoint, bundle, context):
    return select_next_audit_candidate(
        state,
        expected=freeze_state_expectation(state),
        selected_at=datetime(2026, 8, 27, 12, 1, tzinfo=UTC),
        adaptive_preselection_state=checkpoint,
        adaptive_policy_context_sha256=context.policy_context_sha256,
        adaptive_calibration_bundle_sha256=bundle.bundle_sha256,
    )


def _multi_arm_bundle(
    *,
    manifest,
    preselection,
    contexts,
    alpha: float,
):
    projected = preselection_state_from_certificate_v5(preselection)

    def labeled_row(index: int, split: str):
        arms = []
        for arm_index, context in enumerate(contexts):
            state = freeze_adaptive_preselection_state(
                prefix_index=0,
                audit_prefix_item_ids=[],
                audit_prefix_cost_minutes=0,
                scheduler_state_sha256=(
                    f"{index * 10 + arm_index + 10:x}".rjust(64, "0")
                ),
                evidence_graph_sha256=(
                    f"{index * 10 + arm_index + 20:x}".rjust(64, "0")
                ),
                synthesis_sha256=(
                    f"{index * 10 + arm_index + 30:x}".rjust(64, "0")
                ),
                non_calibration_assessment_sha256=(
                    f"{index * 10 + arm_index + 40:x}".rjust(64, "0")
                ),
                non_calibration_gates_passed=True,
                non_calibration_blocking_reasons=[],
                claim_decision=projected.claim_decision,
                score_features=projected.score_features,
            )
            arms.append(
                freeze_adaptive_policy_arm_trajectory(
                    policy_arm_id=context.policy_arm_id,
                    policy_context_sha256=context.policy_context_sha256,
                    states=[state],
                    terminal_reason="all_items_resolved",
                    terminal_candidates=[],
                    terminal_source_candidate_input_sha256=hash_canonical([]),
                    terminal_remaining_budget_minutes=context.budget_minutes,
                )
            )
        corpus = freeze_complete_corpus_identity(
            corpus_id=f"multi-arm-corpus-{index}",
            corpus_source_sha256=f"{index + 100:x}".rjust(64, "0"),
            corpus_cutoff=manifest.protocol.corpus_cutoff,
            publication_ids=[f"multi-arm-publication-{index}"],
            source_manifest_sha256=f"{index + 200:x}".rjust(64, "0"),
        )
        visible = freeze_policy_visible_question_trajectory(
            question_id=f"multi-arm-question-{index}",
            split=split,  # type: ignore[arg-type]
            population_id=manifest.population_id,
            domain=manifest.domain,
            corpus=corpus,
            arms=arms,
        )
        reference = freeze_question_reference_verdict(
            question_id=visible.question_id,
            verdict=projected.claim_decision,
            label_source="expert_adjudication",
            adjudication_protocol_sha256="7" * 64,
            adjudication_artifact_sha256=f"{index + 300:x}".rjust(64, "0"),
        )
        return join_labeled_question_trajectory(
            visible=visible,
            reference=reference,
        )

    development = [labeled_row(index, "development") for index in range(1, 5)]
    calibration = [labeled_row(index, "calibration") for index in range(5, 9)]
    development_freeze = fit_adaptive_development(
        development,
        policy_contexts=contexts,
        calibration_visible_trajectories=[row.visible for row in calibration],
        alpha=alpha,
        delta=0.5,
        candidate_thresholds={context.policy_arm_id: [1.0] for context in contexts},
        seed=91,
    )
    return calibrate_adaptive_first_release(
        development_freeze,
        calibration,
    )


def test_verifier_rejects_semantically_forged_initial_checkpoint() -> None:
    (
        manifest,
        corpus,
        fingerprint,
        scoring_receipt,
        preselection,
        evaluated,
        bundle,
        context,
    ) = _prospective_fixture()
    true_state = preselection_state_from_certificate_v5(preselection)
    forged_state = freeze_adaptive_preselection_state(
        prefix_index=true_state.prefix_index,
        audit_prefix_item_ids=true_state.audit_prefix_item_ids,
        audit_prefix_cost_minutes=true_state.audit_prefix_cost_minutes,
        scheduler_state_sha256=true_state.scheduler_state_sha256,
        evidence_graph_sha256=true_state.evidence_graph_sha256,
        synthesis_sha256=true_state.synthesis_sha256,
        non_calibration_assessment_sha256="f" * 64,
        non_calibration_gates_passed=False,
        non_calibration_blocking_reasons=["forged:pretend-not-release-eligible"],
        claim_decision="forged-decision",
        score_features=true_state.score_features,
    )
    selected = _select_with_checkpoint(
        state=evaluated,
        checkpoint=forged_state,
        bundle=bundle,
        context=context,
    )

    # A self-hashed checkpoint is not enough: production must independently replay
    # its gate ledger, decision, features, and non-calibration assessment hash.
    with pytest.raises(
        VerificationContractError,
        match=r"adaptive_.*checkpoint.*mismatch|adaptive_.*history.*mismatch",
    ):
        run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=_BUDGET,
            adaptive_calibration_bundle=bundle,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=_ROOT,
            item_risk_scoring_receipt=scoring_receipt,
            sequential_audit_state=selected.state,
            generated_at=datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
        )


def test_active_action_cannot_exist_after_checkpoint_already_earned_release() -> None:
    (
        manifest,
        corpus,
        fingerprint,
        scoring_receipt,
        preselection,
        evaluated,
        bundle,
        context,
    ) = _prospective_fixture()
    qualifying_state = preselection_state_from_certificate_v5(preselection)
    selected = _select_with_checkpoint(
        state=evaluated,
        checkpoint=qualifying_state,
        bundle=bundle,
        context=context,
    )

    # This exact prefix releases under the frozen policy. Selecting an action after
    # it is a direct violation of the calibrated first-release stopping rule.
    with pytest.raises(
        VerificationContractError,
        match=(
            r"continued_after_first_release|active_action_after_first_release|"
            r"action_selected_after_qualifying_release"
        ),
    ):
        run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=_BUDGET,
            adaptive_calibration_bundle=bundle,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=_ROOT,
            item_risk_scoring_receipt=scoring_receipt,
            sequential_audit_state=selected.state,
            generated_at=datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
        )


def test_adaptive_bundle_cannot_be_omitted_or_switched_on_active_resume() -> None:
    (
        manifest,
        corpus,
        fingerprint,
        scoring_receipt,
        preselection,
        evaluated,
        bundle,
        context,
    ) = _prospective_fixture()
    true_state = preselection_state_from_certificate_v5(preselection)
    # Keep this checkpoint genuinely non-release-eligible so this test isolates
    # bundle identity, rather than the continued-after-release guard above.
    blocked_state = freeze_adaptive_preselection_state(
        prefix_index=true_state.prefix_index,
        audit_prefix_item_ids=true_state.audit_prefix_item_ids,
        audit_prefix_cost_minutes=true_state.audit_prefix_cost_minutes,
        scheduler_state_sha256=true_state.scheduler_state_sha256,
        evidence_graph_sha256=true_state.evidence_graph_sha256,
        synthesis_sha256=true_state.synthesis_sha256,
        non_calibration_assessment_sha256=true_state.non_calibration_assessment_sha256,
        non_calibration_gates_passed=False,
        non_calibration_blocking_reasons=["redteam:blocked"],
        claim_decision=true_state.claim_decision,
        score_features=true_state.score_features,
    )
    selected = _select_with_checkpoint(
        state=evaluated,
        checkpoint=blocked_state,
        bundle=bundle,
        context=context,
    )
    common = {
        "manifest": manifest,
        "corpus": corpus,
        "budget_minutes": _BUDGET,
        "expected_pipeline_fingerprint": fingerprint,
        "pipeline_root": _ROOT,
        "item_risk_scoring_receipt": scoring_receipt,
        "sequential_audit_state": selected.state,
        "generated_at": datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
    }
    with pytest.raises(
        VerificationContractError,
        match="adaptive_calibration_bundle_required_for_existing_history",
    ):
        run_verification(**common)

    alternate_bundle, _, _ = _adaptive_contract_for_context(
        manifest=manifest,
        preselection=preselection,
        context=context,
    )
    assert alternate_bundle.bundle_sha256 != bundle.bundle_sha256
    with pytest.raises(
        VerificationContractError,
        match="adaptive_selection_history_calibration_identity_mismatch",
    ):
        run_verification(
            **common,
            adaptive_calibration_bundle=alternate_bundle,
        )


def test_prefix_zero_release_state_cannot_downgrade_to_no_calibration() -> None:
    (
        manifest,
        corpus,
        fingerprint,
        scoring_receipt,
        _,
        evaluated,
        bundle,
        _,
    ) = _prospective_fixture()
    released = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=_BUDGET,
        adaptive_calibration_bundle=bundle,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=_ROOT,
        item_risk_scoring_receipt=scoring_receipt,
        sequential_audit_state=evaluated,
        generated_at=datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
    )
    state = released.sequential_audit_state
    assert released.status == "released"
    assert state is not None
    assert state.session.active_action is None
    assert state.transitions == []

    # A release at prefix zero creates no selection transition. Resuming that same
    # state without its calibration bundle must still be a forbidden policy downgrade.
    with pytest.raises(
        VerificationContractError,
        match=r"adaptive_calibration_bundle_required|calibration_policy_downgrade",
    ):
        run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=_BUDGET,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=_ROOT,
            item_risk_scoring_receipt=scoring_receipt,
            sequential_audit_state=state,
            generated_at=datetime(2026, 8, 27, 12, 3, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("scheduler_state_sha256", "1" * 64, "scheduler_state_hash_mismatch"),
        ("audit_prefix_item_ids", ["forged-item"], "prefix_identity_mismatch"),
        ("audit_prefix_cost_minutes", 1.0, "prefix_cost_mismatch"),
        ("evidence_graph_sha256", "2" * 64, "evidence_graph_hash_mismatch"),
        ("synthesis_sha256", "3" * 64, "synthesis_hash_mismatch"),
    ],
)
def test_selection_rejects_rehashed_structural_checkpoint_mutations(
    field: str,
    value: object,
    error: str,
) -> None:
    *_, preselection, evaluated, bundle, context = _prospective_fixture()
    true_state = preselection_state_from_certificate_v5(preselection)
    payload = true_state.model_dump(mode="python")
    payload[field] = value
    if field == "audit_prefix_item_ids":
        payload["prefix_index"] = 1
    payload.pop("state_version")
    payload.pop("state_sha256")
    forged = freeze_adaptive_preselection_state(**payload)

    with pytest.raises(SequentialVerificationContractError, match=error):
        _select_with_checkpoint(
            state=evaluated,
            checkpoint=forged,
            bundle=bundle,
            context=context,
        )


def test_certificate_candidate_history_is_verifier_owned() -> None:
    (
        manifest,
        corpus,
        fingerprint,
        scoring_receipt,
        preselection,
        evaluated,
        bundle,
        context,
    ) = _prospective_fixture()
    true_state = preselection_state_from_certificate_v5(preselection)
    selected = _select_with_checkpoint(
        state=evaluated,
        checkpoint=true_state,
        bundle=bundle,
        context=context,
    )

    # If production rejects continued-after-release, it also cannot emit a forged
    # candidate prefix into a certificate that reparses as independently valid.
    with pytest.raises(VerificationContractError):
        certificate = run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=_BUDGET,
            adaptive_calibration_bundle=bundle,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=_ROOT,
            item_risk_scoring_receipt=scoring_receipt,
            sequential_audit_state=selected.state,
            generated_at=datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
        )
        VerificationCertificate.model_validate(certificate.model_dump(mode="json"))


def test_certificate_independently_rejects_active_action_after_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        manifest,
        corpus,
        fingerprint,
        scoring_receipt,
        preselection,
        evaluated,
        bundle,
        context,
    ) = _prospective_fixture()
    qualifying_state = preselection_state_from_certificate_v5(preselection)
    selected = _select_with_checkpoint(
        state=evaluated,
        checkpoint=qualifying_state,
        bundle=bundle,
        context=context,
    )

    # Bypass only the verifier's first defense to exercise the certificate as an
    # independent verifier. All later assessment and certificate replay calls use
    # the real frozen adaptive policy.
    real_assessor = verifier_module.assess_adaptive_release_candidate
    calls = 0

    def bypass_first_guard(candidate, calibration_bundle):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(status="abstained")
        return real_assessor(candidate, calibration_bundle)

    monkeypatch.setattr(
        verifier_module,
        "assess_adaptive_release_candidate",
        bypass_first_guard,
    )
    monkeypatch.setattr(
        claim_release_module,
        "assess_adaptive_release_candidate",
        lambda _candidate, _bundle: SimpleNamespace(status="abstained"),
    )
    with pytest.raises(
        (ValidationError, VerificationContractError),
        match=r"adaptive_.*active.*release|adaptive_.*qualifying.*release",
    ):
        certificate = run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=_BUDGET,
            adaptive_calibration_bundle=bundle,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=_ROOT,
            item_risk_scoring_receipt=scoring_receipt,
            sequential_audit_state=selected.state,
            generated_at=datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
        )
        assert certificate.adaptive_prospective_assessment is not None
        assert certificate.adaptive_prospective_assessment.status == "released"
        VerificationCertificate.model_validate(certificate.model_dump(mode="json"))


def test_certificate_independently_rejects_forged_prior_checkpoint_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        manifest,
        corpus,
        fingerprint,
        scoring_receipt,
        preselection,
        evaluated,
        bundle,
        context,
    ) = _prospective_fixture()
    true_state = preselection_state_from_certificate_v5(preselection)
    forged_state = freeze_adaptive_preselection_state(
        prefix_index=true_state.prefix_index,
        audit_prefix_item_ids=true_state.audit_prefix_item_ids,
        audit_prefix_cost_minutes=true_state.audit_prefix_cost_minutes,
        scheduler_state_sha256=true_state.scheduler_state_sha256,
        evidence_graph_sha256=true_state.evidence_graph_sha256,
        synthesis_sha256=true_state.synthesis_sha256,
        non_calibration_assessment_sha256="f" * 64,
        non_calibration_gates_passed=False,
        non_calibration_blocking_reasons=["forged:pretend-not-release-eligible"],
        claim_decision="forged-decision",
        score_features=true_state.score_features,
    )
    selected = _select_with_checkpoint(
        state=evaluated,
        checkpoint=forged_state,
        bundle=bundle,
        context=context,
    )

    # Bypass only production's semantic replay once. Certificate validation must not
    # reduce to "candidate equals the same caller-authored checkpoint in the ledger."
    real_projector = verifier_module.freeze_preselection_state_from_production_components
    calls = 0

    def bypass_first_projection(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return forged_state
        return real_projector(**kwargs)

    monkeypatch.setattr(
        verifier_module,
        "freeze_preselection_state_from_production_components",
        bypass_first_projection,
    )
    with pytest.raises(
        (ValidationError, VerificationContractError),
        match=r"adaptive_.*checkpoint.*mismatch|adaptive_.*history.*mismatch",
    ):
        certificate = run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=_BUDGET,
            adaptive_calibration_bundle=bundle,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=_ROOT,
            item_risk_scoring_receipt=scoring_receipt,
            sequential_audit_state=selected.state,
            generated_at=datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
        )
        VerificationCertificate.model_validate(certificate.model_dump(mode="json"))


def test_selected_calibrated_arm_disambiguates_matching_multi_arm_bundle() -> None:
    (
        manifest,
        corpus,
        fingerprint,
        scoring_receipt,
        _,
        _,
        bundle,
        _,
    ) = _prospective_fixture()

    # The calibrated selected candidate, rather than caller input, owns the deployed
    # arm. This is a smoke test for fail-closed arm derivation on the verifier path.
    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=_BUDGET,
        adaptive_calibration_bundle=bundle,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=_ROOT,
        item_risk_scoring_receipt=scoring_receipt,
        generated_at=datetime(2026, 8, 27, 12, 3, tzinfo=UTC),
    )
    assert bundle.selected is not None
    assert certificate.adaptive_policy_context is not None
    assert certificate.adaptive_policy_context.policy_arm_id == (
        bundle.selected.candidate.policy_arm_id
    )


def test_selected_candidate_disambiguates_two_matching_production_arms() -> None:
    (
        manifest,
        corpus,
        fingerprint,
        scoring_receipt,
        preselection,
        _,
        _,
        _,
    ) = _prospective_fixture()
    contexts = [
        build_verifier_adaptive_policy_context(
            manifest=manifest,
            pipeline_sha256=fingerprint.pipeline_sha256,
            budget_minutes=_BUDGET,
            policy_arm_id=arm_id,
        )
        for arm_id in ("production-a", "production-b")
    ]
    bundle = _multi_arm_bundle(
        manifest=manifest,
        preselection=preselection,
        contexts=contexts,
        alpha=0.99,
    )
    assert bundle.selected is not None

    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=_BUDGET,
        adaptive_calibration_bundle=bundle,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=_ROOT,
        item_risk_scoring_receipt=scoring_receipt,
        generated_at=datetime(2026, 8, 27, 12, 3, tzinfo=UTC),
    )
    assert certificate.adaptive_policy_context is not None
    assert certificate.adaptive_policy_context.policy_arm_id == (
        bundle.selected.candidate.policy_arm_id
    )


def test_abstain_all_bundle_with_two_matching_arms_is_rejected_as_ambiguous() -> None:
    (
        manifest,
        corpus,
        fingerprint,
        scoring_receipt,
        preselection,
        evaluated,
        _,
        _,
    ) = _prospective_fixture()
    contexts = [
        build_verifier_adaptive_policy_context(
            manifest=manifest,
            pipeline_sha256=fingerprint.pipeline_sha256,
            budget_minutes=_BUDGET,
            policy_arm_id=arm_id,
        )
        for arm_id in ("production-a", "production-b")
    ]
    bundle = _multi_arm_bundle(
        manifest=manifest,
        preselection=preselection,
        contexts=contexts,
        alpha=1e-6,
    )
    assert bundle.status == "abstain_all"

    with pytest.raises(
        VerificationContractError,
        match="adaptive_calibration_abstain_all_policy_context_ambiguous",
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


def test_claim_release_active_path_rejects_mismatched_candidate_identity() -> None:
    (
        manifest,
        _,
        _,
        scoring_receipt,
        preselection,
        evaluated,
        bundle,
        context,
    ) = _prospective_fixture()
    checkpoint = preselection_state_from_certificate_v5(preselection)
    selected = _select_with_checkpoint(
        state=evaluated,
        checkpoint=checkpoint,
        bundle=bundle,
        context=context,
    )
    prepared = prepare_verification_scientific_state(
        manifest=manifest,
        graph=selected.state.graph,
        pipeline_verification=scoring_receipt.pipeline_verification,
        item_risk_calibration_bundle=scoring_receipt.calibration_bundle,
        item_risk_candidates=list(scoring_receipt.candidates),
    )
    forged_candidate = freeze_prospective_adaptive_candidate(
        question_id="wrong-question",
        population_id="wrong-population",
        domain="wrong-domain",
        policy_arm_id=context.policy_arm_id,
        policy_context_sha256=context.policy_context_sha256,
        corpus=preselection.complete_corpus_identity,
        observed_states=[checkpoint],
    )

    # The active-action blocker must not bypass basic adaptive identity and ledger
    # checks; otherwise a standalone release assessment certifies unrelated lineage.
    with pytest.raises(
        ClaimReleaseContractError,
        match=(
            r"adaptive_release_candidate_.*mismatch|"
            r"adaptive_release_requires_unified_verifier_history_replay"
        ),
    ):
        assess_claim_release(
            graph=selected.state.graph,
            question_id=manifest.question_id,
            population_id=manifest.population_id,
            domain=manifest.domain,
            pipeline_sha256=scoring_receipt.pipeline_verification.expected_pipeline_sha256,
            target=prepared.target,
            audit_candidates=list(prepared.audit_candidates),
            claim_model=prepared.claim_model,
            audit_resolution_receipts=[],
            audit_budget=_BUDGET,
            frozen_calibration_bundle=None,
            adaptive_calibration_bundle=bundle,
            adaptive_release_candidate=forged_candidate,
            config=manifest.release,
            audit_guard_config=manifest.audit_guard.to_runtime(),
            sequential_audit_state=selected.state,
        )


def test_claim_release_rejects_forged_prior_prefix_before_releasing() -> None:
    (
        manifest,
        _,
        fingerprint,
        scoring_receipt,
        preselection,
        evaluated,
        bundle,
        context,
    ) = _prospective_fixture()
    true_prefix_zero = preselection_state_from_certificate_v5(preselection)
    forged_prefix_zero = freeze_adaptive_preselection_state(
        prefix_index=0,
        audit_prefix_item_ids=[],
        audit_prefix_cost_minutes=0,
        scheduler_state_sha256=true_prefix_zero.scheduler_state_sha256,
        evidence_graph_sha256=true_prefix_zero.evidence_graph_sha256,
        synthesis_sha256=true_prefix_zero.synthesis_sha256,
        non_calibration_assessment_sha256="f" * 64,
        non_calibration_gates_passed=False,
        non_calibration_blocking_reasons=["forged:pretend-not-release-eligible"],
        claim_decision=true_prefix_zero.claim_decision,
        score_features=true_prefix_zero.score_features,
    )
    selected = _select_with_checkpoint(
        state=evaluated,
        checkpoint=forged_prefix_zero,
        bundle=bundle,
        context=context,
    )
    prepared = prepare_verification_scientific_state(
        manifest=manifest,
        graph=selected.state.graph,
        pipeline_verification=scoring_receipt.pipeline_verification,
        item_risk_calibration_bundle=scoring_receipt.calibration_bundle,
        item_risk_candidates=list(scoring_receipt.candidates),
    )
    refreshed_candidates = sequential_candidates_from_prepared_state(
        manifest=manifest,
        prepared=prepared,
    )
    expected = freeze_state_expectation(selected.state)
    adjudication = freeze_selected_adjudication(
        selected.state,
        expected=expected,
        provenance="redteam_adjudication",
        adjudicator_count=1,
        protocol_sha256="a" * 64,
        payload_sha256="b" * 64,
        completed_at=datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
        realized_cost=1.0,
    )
    resolved = resolve_selected_audit_candidate(
        selected.state,
        expected=expected,
        adjudication=adjudication,
        disposition=CorrectionDisposition.NO_CHANGE,
        corrected_graph=None,
        correction_provenance="redteam_adjudication",
        correction_protocol_sha256="c" * 64,
        external_correction_payload_sha256="d" * 64,
        synthesis_runner_sha256=compute_synthesis_runner_sha256(
            manifest=manifest,
            pipeline_sha256=fingerprint.pipeline_sha256,
        ),
        candidate_runner_sha256=compute_candidate_runner_sha256(
            manifest=manifest,
            pipeline_sha256=fingerprint.pipeline_sha256,
        ),
        rerun_synthesis=lambda _graph: prepared.synthesis,
        rerun_candidates=lambda _graph, _synthesis, _session: refreshed_candidates,
    )

    common = {
        "graph": resolved.state.graph,
        "question_id": manifest.question_id,
        "population_id": manifest.population_id,
        "domain": manifest.domain,
        "pipeline_sha256": fingerprint.pipeline_sha256,
        "target": prepared.target,
        "audit_candidates": list(prepared.audit_candidates),
        "claim_model": prepared.claim_model,
        "audit_resolution_receipts": [],
        "audit_budget": _BUDGET,
        "external_noncalibration_blocking_reasons": [],
        "config": manifest.release,
        "audit_guard_config": manifest.audit_guard.to_runtime(),
        "sequential_audit_state": resolved.state,
    }
    shadow = assess_claim_release(
        **common,
        frozen_calibration_bundle=None,
    )
    true_current = freeze_preselection_state_from_production_components(
        sequential_state=resolved.state,
        release_assessment=shadow,
        blocking_adapter_reasons=[],
    )
    candidate = freeze_prospective_adaptive_candidate(
        question_id=manifest.question_id,
        population_id=manifest.population_id,
        domain=manifest.domain,
        policy_arm_id=context.policy_arm_id,
        policy_context_sha256=context.policy_context_sha256,
        corpus=preselection.complete_corpus_identity,
        observed_states=[forged_prefix_zero, true_current],
    )

    # The current prefix is exact and would release, but the earlier checkpoint was
    # caller-authored. The lower-level release API must verify the whole ledger too.
    with pytest.raises(
        ClaimReleaseContractError,
        match=(
            r"adaptive_.*history.*mismatch|adaptive_.*checkpoint.*mismatch|"
            r"adaptive_release_requires_unified_verifier_history_replay"
        ),
    ):
        assess_claim_release(
            **common,
            frozen_calibration_bundle=None,
            adaptive_calibration_bundle=bundle,
            adaptive_release_candidate=candidate,
        )
