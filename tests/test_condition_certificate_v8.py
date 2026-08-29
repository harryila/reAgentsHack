from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from literature_multiverse.certificate import (
    ConditionVerificationCertificateV6,
    ConditionVerificationCertificateV8,
    FinalConditionVerificationCertificateV9,
)
from literature_multiverse.composed_corpus_identity import (
    COMPLETE_CORPUS_IDENTITY_V3_VERSION,
    CompleteCorpusIdentityV3,
    freeze_complete_corpus_identity_v2,
)
from literature_multiverse.condition_confirmation import (
    confirm_condition_dependence,
    fit_condition_confirmation_model,
    freeze_condition_confirmation_config,
    freeze_condition_confirmation_target,
    materialize_condition_confirmation_inputs,
    prepare_condition_confirmation_plan,
)
from literature_multiverse.corpus_pipeline_composition import (
    build_composed_calibration_pipeline_verification_v1,
    build_corpus_ingress_projection_v1,
    build_corpus_pipeline_composition_join_v1,
)
from literature_multiverse.corpus_pipeline_composition_runtime import (
    EXTERNAL_REPLAY_RECEIPT_VERSION,
    HOSTED_INGRESS_INTERFACE,
    CorpusPipelineCompositionExternalReplayReceiptV1,
    join_policy_pipeline_components,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.pipeline_fingerprint import (
    compute_pipeline_fingerprint,
    require_pipeline_fingerprint_match,
)
from literature_multiverse.verifier import (
    COMPOSITION_PIPELINE_IDENTITY_BASIS,
    complete_corpus_identity_for_adaptive_calibration,
    compute_verifier_pipeline_fingerprint,
    finalize_condition_verification,
    run_verification,
    verifier_pipeline_components,
)
from test_unified_verifier import (
    _artifact_backed_item_risk_contract,
    _condition_runtime_fixture,
    _confirmation_aware_noncondition_bundle,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return hash_canonical({"condition-v8-test": label})


def _synthetic_receipt_and_corpus(corpus: Any) -> tuple[
    CorpusPipelineCompositionExternalReplayReceiptV1,
    Any,
]:
    verifier_components = verifier_pipeline_components()
    verifier_fingerprint = compute_verifier_pipeline_fingerprint(root=ROOT)
    verifier_proof = require_pipeline_fingerprint_match(
        expected=verifier_fingerprint,
        root=ROOT,
        current_components=verifier_components,
    )
    join_components = join_policy_pipeline_components()
    join_fingerprint = compute_pipeline_fingerprint(
        root=ROOT,
        components=list(join_components),
    )
    join_proof = require_pipeline_fingerprint_match(
        expected=join_fingerprint,
        root=ROOT,
        current_components=join_components,
    )
    metadata = deepcopy(corpus.metadata)
    graph_sha256 = hash_canonical(corpus.graph)
    metadata.update(
        {
            "grounding_package_sha256": _sha("grounding-package"),
            "typed_evidence_corpus_sha256": _sha("typed-corpus"),
            "grounding_validation_sha256": _sha("grounding-validation"),
            "cohort_reconciliation_receipt_sha256": _sha("reconciliation"),
            "reconciled_graph_sha256": graph_sha256,
        }
    )
    corpus = replace(corpus, metadata=metadata)
    source_manifest = metadata["native_source_manifest"]
    ingress = build_corpus_ingress_projection_v1(
        ingress_interface=HOSTED_INGRESS_INTERFACE,
        corpus_id=corpus.corpus_id,
        question_id=corpus.corpus_id,
        corpus_cutoff=metadata["native_corpus_cutoff"],
        corpus_source_sha256=corpus.source_sha256,
        grounding_package_sha256=metadata["grounding_package_sha256"],
        typed_corpus_sha256=metadata["typed_evidence_corpus_sha256"],
        extraction_pipeline_sha256=verifier_fingerprint.pipeline_sha256,
        extraction_pipeline_verification_sha256=verifier_proof.verification_sha256,
        source_manifest_sha256=metadata["source_manifest_sha256"],
        source_membership_sha256=hash_canonical(source_manifest["records"]),
        question_config_sha256=metadata["question_config_sha256"],
        extraction_context_sha256=metadata["extraction_context_sha256"],
        extraction_context_receipt_sha256=(
            metadata["extraction_context_receipt_sha256"]
        ),
        hosted_run_sha256=_sha("hosted-run"),
        terminal_call_membership_sha256=_sha("terminal-calls"),
        terminal_fragment_membership_sha256=(
            metadata["terminal_fragment_membership_sha256"]
        ),
        grounding_validation_sha256=metadata["grounding_validation_sha256"],
        grounding_replay_sha256=metadata["grounding_replay_sha256"],
        cohort_reconciliation_receipt_sha256=(
            metadata["cohort_reconciliation_receipt_sha256"]
        ),
        reconciled_graph_sha256=graph_sha256,
        effective_graph_sha256=graph_sha256,
        hosted_bridge_receipt_sha256=_sha("bridge"),
    )
    join = build_corpus_pipeline_composition_join_v1(
        extraction_pipeline_verification=verifier_proof,
        verifier_core_pipeline_verification=verifier_proof,
        join_policy_pipeline_verification=join_proof,
        corpus_ingress=ingress,
    )
    composed_proof = build_composed_calibration_pipeline_verification_v1(
        repository_root=ROOT,
        verifier_core_components=verifier_components,
        join_policy_components=join_components,
        ingress_interface=HOSTED_INGRESS_INTERFACE,
        extraction_pipeline_verification=verifier_proof,
        verifier_core_pipeline_verification=verifier_proof,
        join_policy_pipeline_verification=join_proof,
    )
    composed = composed_proof.computed
    assert composed is not None
    payload: dict[str, Any] = {
        "receipt_version": EXTERNAL_REPLAY_RECEIPT_VERSION,
        "external_replay_completed": True,
        "ingress_interface": HOSTED_INGRESS_INTERFACE,
        "grounding_package_file_sha256": _sha("package-file"),
        "hosted_bridge_receipt_file_sha256": _sha("bridge-file"),
        "corpus_source_sha256": corpus.source_sha256,
        "grounding_package_sha256": ingress.grounding_package_sha256,
        "typed_corpus_sha256": ingress.typed_corpus_sha256,
        "hosted_bridge_receipt_sha256": ingress.hosted_bridge_receipt_sha256,
        "grounding_replay_sha256": ingress.grounding_replay_sha256,
        "effective_graph_sha256": ingress.effective_graph_sha256,
        "extraction_pipeline_verification_sha256": verifier_proof.verification_sha256,
        "verifier_core_pipeline_verification_sha256": verifier_proof.verification_sha256,
        "join_policy_pipeline_verification_sha256": join_proof.verification_sha256,
        "composed_pipeline_fingerprint": composed,
        "composed_pipeline_verification": composed_proof,
        "composed_pipeline_sha256": composed.pipeline_sha256,
        "composed_pipeline_verification_sha256": composed_proof.verification_sha256,
        "calibration_pipeline_sha256": composed.pipeline_sha256,
        "release_pipeline_sha256": composed.pipeline_sha256,
        "composition_join": join,
        "composition_join_sha256": join.join_sha256,
        "corpus_ingress_projection_sha256": ingress.ingress_projection_sha256,
        "public_corpus_loader_match_completed": True,
        "scientific_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    receipt = CorpusPipelineCompositionExternalReplayReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )
    return receipt, corpus


def _complete_v3(
    *,
    membership_v1: Any,
    receipt: CorpusPipelineCompositionExternalReplayReceiptV1,
) -> CompleteCorpusIdentityV3:
    complete_v2 = freeze_complete_corpus_identity_v2(
        complete_corpus_membership_v1=membership_v1,
        corpus_pipeline_join=receipt.composition_join,
        composed_pipeline_verification=receipt.composed_pipeline_verification,
    )
    payload: dict[str, Any] = {
        "identity_version": COMPLETE_CORPUS_IDENTITY_V3_VERSION,
        "complete_corpus_identity_v2": complete_v2,
        "external_replay_receipt": receipt,
        "corpus_id": complete_v2.corpus_id,
        "corpus_source_sha256": complete_v2.corpus_source_sha256,
        "corpus_cutoff": complete_v2.corpus_cutoff,
        "membership_basis": complete_v2.membership_basis,
        "publication_ids": complete_v2.publication_ids,
        "source_manifest_sha256": complete_v2.source_manifest_sha256,
        "membership_sha256": complete_v2.membership_sha256,
        "membership_composition_v2_sha256": complete_v2.membership_composition_sha256,
        "external_replay_receipt_sha256": receipt.receipt_sha256,
        "corpus_pipeline_join_sha256": receipt.composition_join_sha256,
        "corpus_ingress_projection_sha256": receipt.corpus_ingress_projection_sha256,
        "extraction_pipeline_sha256": receipt.composition_join.extraction_pipeline_sha256,
        "calibration_pipeline_sha256": receipt.calibration_pipeline_sha256,
        "release_pipeline_sha256": receipt.release_pipeline_sha256,
        "composed_pipeline_verification_sha256": (
            receipt.composed_pipeline_verification_sha256
        ),
        "extraction_accuracy_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_claim_truth_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
        "release_authorizing": False,
    }
    return CompleteCorpusIdentityV3.model_validate(
        {**payload, "membership_composition_sha256": hash_canonical(payload)}
    )


@pytest.fixture(scope="module")
def condition_v8_certificate() -> ConditionVerificationCertificateV8:
    manifest, base_corpus, *_ = _condition_runtime_fixture()
    receipt, corpus = _synthetic_receipt_and_corpus(base_corpus)
    complete_v1 = complete_corpus_identity_for_adaptive_calibration(
        manifest=manifest,
        corpus=corpus,
    )
    global_target = manifest.global_condition_target
    extraction_context = corpus.extraction_context
    assert global_target is not None and extraction_context is not None
    target = freeze_condition_confirmation_target(
        question_id=manifest.question_id,
        claim_spec_sha256=global_target.target_sha256,
        question_config_sha256=extraction_context.question_config_sha256,
        corpus_snapshot_sha256=complete_v1.membership_sha256,
        corpus_cutoff=manifest.protocol.corpus_cutoff,
        outcome_name=global_target.outcome_name,
        contrast_label=global_target.contrast_label,
        contrast_estimand=global_target.estimand,
        positive_direction_means=global_target.positive_direction_means,
        treatment_role=global_target.treatment_role,
        comparator_role=global_target.comparator_role,
        measure=global_target.measure,
        moderator_names=global_target.moderator_names,
    )
    roster, development, _confirmation, materialization = (
        materialize_condition_confirmation_inputs(
            full_graph=corpus.graph,
            target=target,
        )
    )
    pipeline_sha256 = receipt.composed_pipeline_sha256
    plan = prepare_condition_confirmation_plan(
        target=target,
        config=freeze_condition_confirmation_config(),
        roster=roster,
        materialization_receipt=materialization,
        pipeline_sha256=pipeline_sha256,
        external_freeze_anchor="test:condition-v8-composed-pipeline",
    )
    model = fit_condition_confirmation_model(
        plan,
        development,
        current_pipeline_sha256=pipeline_sha256,
    )
    bundle = _confirmation_aware_noncondition_bundle(
        manifest=manifest,
        pipeline_sha256=pipeline_sha256,
        budget_minutes=30,
    )
    _, _, item_risk_receipt = _artifact_backed_item_risk_contract(
        manifest=manifest,
        corpus=replace(corpus, graph=development),
        fingerprint=receipt.composed_pipeline_fingerprint,
        repository_root=ROOT,
        calibration_unit_count=1000,
    )
    complete_v3 = _complete_v3(
        membership_v1=complete_v1,
        receipt=receipt,
    )
    composed_corpus = replace(
        corpus,
        composition_external_replay_receipt=receipt,
    )
    with (
        patch(
            "literature_multiverse.verifier._composition_pipeline_identity",
            return_value=(
                receipt.composed_pipeline_verification,
                COMPOSITION_PIPELINE_IDENTITY_BASIS,
                receipt,
            ),
        ),
        patch(
            "literature_multiverse.verifier.freeze_complete_corpus_identity_v3",
            return_value=complete_v3,
        ),
    ):
        source = run_verification(
            manifest=manifest,
            corpus=composed_corpus,
            budget_minutes=30,
            adaptive_calibration_bundle_v2=bundle,
            condition_plan=plan,
            condition_development_graph=development,
            condition_frozen_model=model,
            item_risk_scoring_receipt=item_risk_receipt,
            expected_pipeline_fingerprint=receipt.composed_pipeline_fingerprint,
            pipeline_root=ROOT,
            composition_grounding_package_path=ROOT / "synthetic-package.json",
            composition_hosted_bridge_receipt_path=(
                ROOT / "synthetic-bridge.json"
            ),
            generated_at=datetime(2026, 8, 29, 14, tzinfo=UTC),
        )
    assert isinstance(source, ConditionVerificationCertificateV8)
    return source


def test_condition_v8_replays_v6_and_preserves_outcome_firewall(
    condition_v8_certificate: ConditionVerificationCertificateV8,
) -> None:
    source = condition_v8_certificate
    assert source.certificate_version == (
        "literature-multiverse-condition-verification-v8"
    )
    assert source.status == "abstained"
    assert source.condition_confirmation_assessment is None
    assert source.condition_confirmation_gate.status == "missing"
    assert "adapter:corpus_pipeline_identity_mismatch" not in source.reasons
    assert source.pipeline_verification == (
        source.composition_external_replay_receipt.composed_pipeline_verification
    )
    assert ConditionVerificationCertificateV8.model_validate(
        source.model_dump(mode="json")
    ) == source


def test_condition_v8_rejects_receipt_alias_tamper(
    condition_v8_certificate: ConditionVerificationCertificateV8,
) -> None:
    payload = condition_v8_certificate.model_dump(mode="json")
    payload["composition_external_replay_receipt_sha256"] = "0" * 64
    unsigned = {
        key: value for key, value in payload.items() if key != "certificate_sha256"
    }
    payload["certificate_sha256"] = hash_canonical(unsigned)
    with pytest.raises(ValidationError, match="composition_alias_mismatch"):
        ConditionVerificationCertificateV8.model_validate(payload)


def test_condition_v8_rejects_manifest_corpus_policy_binding_alias_tamper(
    condition_v8_certificate: ConditionVerificationCertificateV8,
) -> None:
    payload = condition_v8_certificate.model_dump(mode="json")
    payload["manifest_corpus_policy_binding_v3_sha256"] = "0" * 64
    unsigned = {
        key: value for key, value in payload.items() if key != "certificate_sha256"
    }
    payload["certificate_sha256"] = hash_canonical(unsigned)
    with pytest.raises(ValidationError, match="composition_alias_mismatch"):
        ConditionVerificationCertificateV8.model_validate(payload)


def test_condition_v8_cannot_downgrade_to_v6(
    condition_v8_certificate: ConditionVerificationCertificateV8,
) -> None:
    with pytest.raises(ValidationError):
        ConditionVerificationCertificateV6.model_validate(
            condition_v8_certificate.model_dump(mode="json")
        )


@pytest.fixture(scope="module")
def condition_v9_certificate(
    condition_v8_certificate: ConditionVerificationCertificateV8,
) -> FinalConditionVerificationCertificateV9:
    source = condition_v8_certificate
    assert source.production_stop_decision.outcome == "condition_gate_ready"
    pipeline_sha256 = source.pipeline_verification.computed_pipeline_sha256
    assert pipeline_sha256 is not None
    assessment = confirm_condition_dependence(
        plan=source.condition_plan,
        model=source.condition_frozen_model,
        full_graph=source.current_full_evidence_graph,
        current_pipeline_sha256=pipeline_sha256,
    )
    final = finalize_condition_verification(
        source_certificate=source,
        condition_confirmation_assessment=assessment,
        generated_at=datetime(2026, 8, 29, 15, tzinfo=UTC),
    )
    assert isinstance(final, FinalConditionVerificationCertificateV9)
    return final


def test_condition_v9_binds_v8_before_opening_terminal_outcome(
    condition_v9_certificate: FinalConditionVerificationCertificateV9,
) -> None:
    final = condition_v9_certificate
    source = final.source_certificate_v8
    assert final.certificate_version == (
        "literature-multiverse-condition-verification-v9"
    )
    assert final.source_v8_certificate_sha256 == source.certificate_sha256
    assert final.v6_common_contract_replay_sha256 == (
        source.v6_common_contract_replay_sha256
    )
    assert final.composition_terminal_join.source_v8_certificate_sha256 == (
        source.certificate_sha256
    )
    assert final.composition_terminal_join.source_v8_composition_receipt_sha256 == (
        source.composition_external_replay_receipt_sha256
    )
    assert FinalConditionVerificationCertificateV9.model_validate(
        final.model_dump(mode="json")
    ) == final


def test_condition_v9_rejects_source_v8_hash_substitution(
    condition_v9_certificate: FinalConditionVerificationCertificateV9,
) -> None:
    payload = condition_v9_certificate.model_dump(mode="json")
    payload["source_v8_certificate_sha256"] = "0" * 64
    unsigned = {
        key: value for key, value in payload.items() if key != "certificate_sha256"
    }
    payload["certificate_sha256"] = hash_canonical(unsigned)
    with pytest.raises(ValidationError, match="source_v8_certificate_sha256_alias_mismatch"):
        FinalConditionVerificationCertificateV9.model_validate(payload)
