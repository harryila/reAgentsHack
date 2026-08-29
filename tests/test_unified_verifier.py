from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import scripts.calibrate_adaptive_release as adaptive_calibration_cli
from pydantic import ValidationError

import literature_multiverse.cli as cli_module
from literature_multiverse.adaptive_calibration import (
    AdaptiveCalibrationBundle,
    AdaptiveCalibrationBundleV2,
    AdaptiveCalibrationError,
    ConditionCalibrationCollectionSourceRosterV1,
    ProspectiveAdaptiveReleaseCandidate,
    calibrate_adaptive_first_release,
    calibrate_confirmation_aware_first_release,
    complete_corpus_identity_from_certificate_v5,
    fit_adaptive_development,
    fit_adaptive_development_v2,
    freeze_adaptive_independence_identity_v2,
    freeze_adaptive_policy_arm_trajectory,
    freeze_adaptive_preselection_state,
    freeze_adaptive_target_semantics_v2,
    freeze_complete_corpus_identity,
    freeze_condition_calibration_collection_source_roster_v1,
    freeze_confirmation_aware_arm_trajectory,
    freeze_gate_complete_calibration_roster_v2,
    freeze_policy_visible_question_trajectory,
    freeze_policy_visible_question_trajectory_v2,
    freeze_prospective_adaptive_candidate,
    freeze_question_reference_verdict,
    freeze_question_reference_verdict_v2,
    join_condition_calibration_assessment_receipts,
    join_labeled_question_trajectory,
    join_labeled_question_trajectory_v2,
    policy_visible_trajectory_from_certificate_v5_sequence,
    preselection_state_from_certificate_v5,
)
from literature_multiverse.audit_session import CorrectionDisposition
from literature_multiverse.calibration import (
    FrozenCalibrationBundle,
    RiskExample,
    freeze_calibration_bundle,
)
from literature_multiverse.certificate import (
    ConditionCalibrationAssessmentReceiptV1,
    ConditionCalibrationCollectionSourceV1,
    ConditionVerificationCertificateV6,
    FinalConditionVerificationCertificateV7,
    VerificationCertificate,
    freeze_condition_calibration_assessment_receipt_v1,
)
from literature_multiverse.claim_release import CLAIM_RELEASE_RISK_FEATURE_NAMES
from literature_multiverse.claim_semantics import (
    ClaimDirection,
    ConditionOperator,
    ConditionPredicate,
    MeaningfulEffectThreshold,
    freeze_claim_target_v2,
    freeze_global_condition_dependence_target,
)
from literature_multiverse.cli import main as cli_main
from literature_multiverse.condition_confirmation import (
    confirm_condition_dependence,
    fit_condition_confirmation_model,
    freeze_condition_confirmation_config,
    freeze_condition_confirmation_target,
    materialize_condition_confirmation_inputs,
    prepare_condition_confirmation_plan,
)
from literature_multiverse.config import QuestionConfig, load_config_for_question
from literature_multiverse.effects import EffectEvidence, HarmonizedMeasure
from literature_multiverse.evidence_graph import (
    ArmRole,
    CohortIdentity,
    EvidenceGraph,
    GraphAdapterContext,
    PublicationIdentity,
    adapt_effect_evidence,
)
from literature_multiverse.item_risk_artifacts import ItemRiskScoringRunReceipt
from literature_multiverse.item_risk_calibration import (
    ItemRiskCalibrationBundle,
    ItemRiskCandidate,
    calibrate_item_risk_bounds,
    make_fixed_risk_bin_family,
    score_item_risk_bound,
    seal_item_risk_calibration_unit,
    seal_item_risk_candidate,
    seal_shift_assessment,
)
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.models import FindingRow, make_finding_id
from literature_multiverse.native_extraction import (
    NativeSourceManifest,
    NativeSourceRecord,
    native_extraction_prompt_replacements,
    native_publication_extraction_json_schema,
)
from literature_multiverse.native_grounding import (
    NativeEvaluationSchemaArtifact,
    NativeExtractionArtifactDigest,
    NativeExtractionExecutionContext,
    NativeRenderedPromptArtifact,
    freeze_native_extraction_execution_context,
    freeze_native_provider_execution_receipt,
)
from literature_multiverse.pipeline_fingerprint import (
    PipelineFingerprint,
    verify_pipeline_fingerprint,
)
from literature_multiverse.prompting import render_prompt_text
from literature_multiverse.question_evaluation import (
    QuestionReplayState,
    freeze_question_replay_state_from_certificate,
)
from literature_multiverse.sequential_verification import (
    SequentialVerificationState,
    create_sequential_verification_state,
    freeze_current_audit_candidate,
    freeze_selected_adjudication,
    freeze_state_expectation,
    resolve_selected_audit_candidate,
)
from literature_multiverse.typed_extraction import SourceDocumentArtifact
from literature_multiverse.verifier import (
    AuditGuardConfig,
    AuditPolicyConfig,
    ClaimManifest,
    CorpusEligibilityRecord,
    CorpusLoadResult,
    CorpusProvenanceAssurance,
    LegacyAdapterConfig,
    ScientificClaim,
    VerificationContractError,
    VerificationProtocol,
    adapt_legacy_findings,
    build_offline_fixture,
    build_verifier_adaptive_policy_context,
    complete_corpus_identity_for_adaptive_calibration,
    compute_candidate_runner_sha256,
    compute_synthesis_runner_sha256,
    compute_verification_policy_sha256,
    compute_verifier_pipeline_fingerprint,
    finalize_condition_verification,
    load_corpus,
    prepare_verification_scientific_state,
    run_condition_calibration_collection,
    run_verification,
    sequential_candidates_from_prepared_state,
    validate_condition_calibration_assessment_receipt_external_replay,
    validate_condition_calibration_collection_source_external_replay,
)


def _fixture_extraction_context(
    *,
    manifest: ClaimManifest,
    source_manifest: NativeSourceManifest,
    pipeline_sha256: str,
    corpus_cutoff: str,
) -> NativeExtractionExecutionContext:
    """Build a fully valid v4 context for source-replayed verifier fixtures."""

    repository_root = Path(__file__).resolve().parents[1]
    base = load_config_for_question(
        "antiox-training",
        root=repository_root,
        require_locked=True,
    ).model_dump(mode="json")
    base["question_id"] = manifest.question_id
    base["research_question"] = manifest.claim.statement
    base["target_relation"]["outcome"] = manifest.claim.outcome_name
    base["eligibility"]["include"] = list(manifest.protocol.inclusion_criteria)
    base["eligibility"]["exclude"] = list(manifest.protocol.exclusion_criteria)
    base["outcomes"] = {
        "primary_family": manifest.claim.outcome_name,
        "family_map": {manifest.claim.outcome_name: manifest.claim.outcome_name},
        "included_primary_endpoints": [manifest.claim.outcome_name],
        "endpoint_direction_overrides": {},
        "endpoint_map": {},
    }
    if manifest.global_condition_target is not None:
        existing = {row["name"] for row in base["moderators"]}
        base["moderators"].extend(
            {
                "name": name,
                "type": "categorical",
                "source": "topic",
                "role": "tested",
                "kind": "paper_constant",
                "permutation": "paper",
                "paper_summary": None,
                "display_name": name,
                "allowed_values": ["high", "low"],
                "bins": None,
            }
            for name in manifest.global_condition_target.moderator_names
            if name not in existing
        )
    base["variant_b"]["primary_endpoints"] = [manifest.claim.outcome_name]
    base["anchor_papers"] = []
    # Locked configs require a non-empty anchor list; it is never consulted by the
    # verifier compatibility gate and these are explicitly synthetic fixtures.
    base["anchor_papers"] = [
        {
            "paper_id": "fixture-anchor",
            "expected_eligible": True,
            "expected_finding_count": 1,
            "expected_directions": ["increase"],
            "notes": "Synthetic verifier contract fixture.",
        }
    ]
    base["demo"]["fixture_mode"] = False
    config = QuestionConfig.model_validate(base)

    template_path = repository_root / "prompts/native_extraction.md"
    rendered_prompt, prompt_version = render_prompt_text(
        template_path.read_text(encoding="utf-8"),
        native_extraction_prompt_replacements(config),
    )
    schema = native_publication_extraction_json_schema()
    execution_id = "fixture-paperclip-execution"
    provider_receipt = freeze_native_provider_execution_receipt(
        execution_id=execution_id,
        execution_mode="paperclip_archived",
        provider_id="fixture-paperclip",
        model_id="fixture-model",
        runtime_id="fixture-runtime",
        runtime_version="1",
        raw_call_ledger={"fixture": True, "source_manifest": hash_canonical(source_manifest)},
        call_count=1,
    )
    return freeze_native_extraction_execution_context(
        extraction_mode="paperclip_archived",
        question_config=config,
        pipeline_fingerprint_sha256=pipeline_sha256,
        rendered_prompts=[
            NativeRenderedPromptArtifact(
                prompt_id="fixture-native-prompt",
                renderer_id="repository-native-extraction-v1",
                prompt_version=prompt_version,
                template_path="prompts/native_extraction.md",
                template_sha256=sha256_file(template_path),
                rendered_prompt=rendered_prompt,
                rendered_prompt_sha256=hashlib.sha256(
                    rendered_prompt.encode("utf-8")
                ).hexdigest(),
            )
        ],
        evaluation_schemas=[
            NativeEvaluationSchemaArtifact(
                schema_id="native-official-postvalidation",
                role="official_postvalidation",
                schema_payload=schema,
                schema_sha256=hash_canonical(schema),
            )
        ],
        provider_execution_receipts=[provider_receipt],
        input_artifacts=[
            NativeExtractionArtifactDigest(
                artifact_id="fixture-source-manifest",
                role="source_manifest_input",
                sha256=hash_canonical(source_manifest),
                hash_basis="canonical_json",
            ),
            NativeExtractionArtifactDigest(
                artifact_id="fixture-map-output",
                role="map_output",
                sha256=hash_canonical({"fixture": "map-output"}),
                hash_basis="canonical_json",
                execution_ids=[execution_id],
            ),
            NativeExtractionArtifactDigest(
                artifact_id="fixture-provider-receipt",
                role="provider_execution_receipt",
                sha256=provider_receipt.receipt_sha256,
                hash_basis="canonical_json",
                execution_ids=[execution_id],
            ),
        ],
        source_manifest_content_sha256=hash_canonical(source_manifest),
        source_manifest_records=len(source_manifest.records),
        corpus_cutoff=corpus_cutoff,
    )


def _source_replayed_fixture_contract(
    *,
    manifest: ClaimManifest,
    fixture: object,
    pipeline_sha256: str,
    replay_sha256: str,
    corpus_cutoff: str | None = None,
) -> tuple[
    tuple[object, ...],
    dict[str, object],
    NativeExtractionExecutionContext,
]:
    graph = fixture.graph  # type: ignore[attr-defined]
    records: list[NativeSourceRecord] = []
    source_by_paper: dict[str, SourceDocumentArtifact] = {}
    for index, publication in enumerate(graph.publications, start=1):
        assert publication.doc_id is not None
        source = SourceDocumentArtifact(
            artifact_path=f"data/raw/fixture-source-{index}.json",
            sha256=hash_canonical({"fixture_source_document": index}),
            media_type="application/json",
            source_locator=(
                f"json:data/raw/fixture-source-{index}.json#/"
                f"{publication.doc_id}"
            ),
        )
        source_by_paper[publication.paper_id] = source
        records.append(
            NativeSourceRecord(
                doc_id=publication.doc_id,
                publication=publication,
                source_document=source,
            )
        )
    source_manifest = NativeSourceManifest(
        question_id=manifest.question_id,
        records=sorted(records, key=lambda row: row.doc_id),
    )
    effective_cutoff = corpus_cutoff or manifest.protocol.corpus_cutoff
    extraction_context = _fixture_extraction_context(
        manifest=manifest,
        source_manifest=source_manifest,
        pipeline_sha256=pipeline_sha256,
        corpus_cutoff=effective_cutoff,
    )
    terminal_membership = [
        {
            "fragment_sha256": hash_canonical(
                {"fixture_terminal_fragment": record.publication.publication_id}
            ),
            "paper_id": record.publication.paper_id,
            "publication_id": record.publication.publication_id,
            "status": "estimable",
        }
        for record in source_manifest.records
    ]
    eligibility = tuple(
        row.model_copy(update={"source": source_by_paper[row.paper_id].source_locator})
        for row in fixture.eligibility  # type: ignore[attr-defined]
    )
    metadata: dict[str, object] = {
        "grounding_package_version": "typed-evidence-grounding-package-v4",
        "grounding_replay_sha256": replay_sha256,
        "native_corpus_cutoff": effective_cutoff,
        "native_source_manifest": source_manifest.model_dump(mode="json"),
        "pipeline_fingerprint_sha256": pipeline_sha256,
        "source_manifest_membership_bound": True,
        "source_manifest_records": len(records),
        "source_manifest_sha256": hash_canonical(source_manifest),
        "terminal_fragment_membership": terminal_membership,
        "terminal_fragment_membership_sha256": hash_canonical(terminal_membership),
        "terminal_fragment_records": len(terminal_membership),
        "extraction_context_sha256": extraction_context.context_sha256,
        "extraction_context_receipt_sha256": "d" * 64,
        "replayed_extraction_context_receipt_sha256": "d" * 64,
        "question_config_sha256": extraction_context.question_config_sha256,
        "rendered_prompt_sha256s": sorted(
            prompt.rendered_prompt_sha256
            for prompt in extraction_context.rendered_prompts
        ),
        "evaluation_schema_sha256s": sorted(
            schema.schema_sha256
            for schema in extraction_context.evaluation_schemas
        ),
        "provider_execution_receipts": [
            {
                "call_count": receipt.call_count,
                "execution_identity_sha256": receipt.execution_identity_sha256,
                "execution_mode": receipt.execution_mode,
                "model_id": receipt.model_id,
                "model_revision": receipt.model_revision,
                "provider_id": receipt.provider_id,
                "receipt_sha256": receipt.receipt_sha256,
                "runtime_id": receipt.runtime_id,
                "runtime_version": receipt.runtime_version,
            }
            for receipt in extraction_context.provider_execution_receipts
        ],
    }
    return eligibility, metadata, extraction_context


def _artifact_backed_item_risk_contract(
    *,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
    fingerprint: PipelineFingerprint,
    repository_root: Path,
    bind_source_snapshot: bool = True,
    calibration_unit_count: int = 3,
) -> tuple[
    ItemRiskCalibrationBundle,
    list[ItemRiskCandidate],
    ItemRiskScoringRunReceipt,
]:
    verification = verify_pipeline_fingerprint(
        expected=fingerprint,
        root=repository_root,
    )
    score_model_sha256 = "a" * 64
    adjudication_protocol_sha256 = "b" * 64
    family = make_fixed_risk_bin_family(
        edges=[0.0, 1.0],
        score_name="fixture-extraction-risk",
        score_model_sha256=score_model_sha256,
        definition_source="prespecified",
        definition_artifact_sha256="c" * 64,
    )
    if calibration_unit_count < 3:
        raise ValueError("fixture_item_risk_calibration_requires_three_units")
    splits = ["development", *("calibration" for _ in range(calibration_unit_count - 1))]
    units = [
        seal_item_risk_calibration_unit(
            split=split,
            item_id=f"calibration-item-{index}",
            question_id=f"calibration-question-{index}",
            paper_id=f"calibration-paper-{index}",
            population_id=manifest.population_id,
            domain=manifest.domain,
            pipeline_sha256=fingerprint.pipeline_sha256,
            score_model_sha256=score_model_sha256,
            score_input_sha256=f"{index:x}".rjust(64, "0"),
            risk_score=0.1,
            observed_error=False,
            label_source="expert_adjudication",
            adjudication_protocol_sha256=adjudication_protocol_sha256,
            adjudication_artifact_sha256=f"{index + 20:x}".rjust(64, "0"),
        )
        for index, split in enumerate(splits, start=1)
    ]
    bundle = calibrate_item_risk_bounds(
        units,
        pipeline_verification=verification,
        bin_family=family,
        familywise_delta=0.05,
        sampling_protocol_sha256="d" * 64,
        error_event_definition="Any material evidence-item extraction error.",
        shift_detector_id="fixture-shift-detector",
        shift_detector_sha256="e" * 64,
    )
    shift = seal_shift_assessment(
        bundle=bundle,
        candidate_population_id=manifest.population_id,
        candidate_domain=manifest.domain,
        status="no_shift_detected",
        assessment_input_sha256="f" * 64,
        assessment_artifact_sha256="1" * 64,
    )
    candidates = sorted(
        [
            seal_item_risk_candidate(
                item_id=estimate.estimate_id,
                question_id=manifest.question_id,
                paper_id=estimate.effect.paper_id,
                population_id=manifest.population_id,
                domain=manifest.domain,
                pipeline_sha256=fingerprint.pipeline_sha256,
                score_model_sha256=score_model_sha256,
                score_input_sha256=(
                    hash_canonical(estimate) if bind_source_snapshot else "9" * 64
                ),
                risk_score=0.1,
                shift_assessment=shift,
            )
            for estimate in corpus.graph.outcome_estimates
        ],
        key=lambda candidate: candidate.item_id,
    )
    bounds = [
        score_item_risk_bound(
            candidate=candidate,
            bundle=bundle,
            pipeline_verification=verification,
        )
        for candidate in candidates
    ]
    receipt_payload = {
        "receipt_version": "item-risk-scoring-run-v2",
        "calibration_run_file_sha256": hash_canonical(bundle),
        "calibration_run_receipt_sha256": hash_canonical(
            {"fixture": "calibration-run", "bundle": bundle}
        ),
        "calibration_bundle_sha256": bundle.bundle_sha256,
        "calibration_bundle": bundle,
        "expected_pipeline_file_sha256": hash_canonical(fingerprint),
        "shift_run_file_sha256": hash_canonical(shift),
        "shift_run_receipt_sha256": hash_canonical(
            {"fixture": "shift-run", "assessment": shift}
        ),
        "candidate_input_file_sha256": hash_canonical(candidates),
        "candidate_count": len(candidates),
        "pipeline_verification": verification,
        "candidates": candidates,
        "candidate_sha256s": [row.candidate_sha256 for row in candidates],
        "bounds": bounds,
        "access_order": [
            "calibration_run_receipt_opened",
            "expected_pipeline_fingerprint_opened",
            "pipeline_fingerprint_recomputed_and_matched",
            "shift_assessment_receipt_opened",
            "prospective_candidates_opened",
            "risk_bounds_scored",
        ],
    }
    receipt = ItemRiskScoringRunReceipt.model_validate(
        {
            **receipt_payload,
            "receipt_sha256": hash_canonical(receipt_payload),
        }
    )
    return bundle, candidates, receipt


def _release_calibration_bundle(
    *,
    manifest: ClaimManifest,
    pipeline_sha256: str,
) -> FrozenCalibrationBundle:
    rows: list[RiskExample] = []
    for index in range(8):
        unsupported = index >= 4
        rows.append(
            RiskExample(
                question_id=f"development-{index}",
                split="development",
                population_id=manifest.population_id,
                domain=manifest.domain,
                pipeline_sha256=pipeline_sha256,
                paper_ids=[f"development-paper-{index}"],
                features={
                    name: float(unsupported)
                    for name in CLAIM_RELEASE_RISK_FEATURE_NAMES
                },
                unsupported_claim=unsupported,
                label_source="benchmark_annotation",
            )
        )
    for index in range(4):
        rows.append(
            RiskExample(
                question_id=f"calibration-{index}",
                split="calibration",
                population_id=manifest.population_id,
                domain=manifest.domain,
                pipeline_sha256=pipeline_sha256,
                paper_ids=[f"calibration-paper-{index}"],
                features={name: 0.0 for name in CLAIM_RELEASE_RISK_FEATURE_NAMES},
                unsupported_claim=False,
                label_source="benchmark_annotation",
            )
        )
    return freeze_calibration_bundle(
        rows,
        alpha=0.99,
        delta=0.5,
        seed=3,
        candidate_thresholds=[1.0],
    )


def _adaptive_release_contract(
    *,
    manifest: ClaimManifest,
    preselection_certificate: VerificationCertificate,
    pipeline_sha256: str,
    budget_minutes: float,
    prior_preselection_certificates: tuple[VerificationCertificate, ...] = (),
) -> tuple[AdaptiveCalibrationBundle, ProspectiveAdaptiveReleaseCandidate]:
    """Small complete-question contract for verifier integration tests."""

    context = build_verifier_adaptive_policy_context(
        manifest=manifest,
        pipeline_sha256=pipeline_sha256,
        budget_minutes=budget_minutes,
        policy_arm_id="production-adaptive",
    )
    projected_states = [
        preselection_state_from_certificate_v5(certificate)
        for certificate in (*prior_preselection_certificates, preselection_certificate)
    ]
    projected_states.sort(key=lambda state: state.prefix_index)
    projected = projected_states[-1]

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
            corpus=corpus,
            arms=[arm],
        )
        reference = freeze_question_reference_verdict(
            question_id=f"calibration-question-{index}",
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
    bundle = calibrate_adaptive_first_release(
        development_freeze,
        calibration,
    )
    candidate = freeze_prospective_adaptive_candidate(
        question_id=manifest.question_id,
        population_id=manifest.population_id,
        domain=manifest.domain,
        policy_arm_id=context.policy_arm_id,
        policy_context_sha256=context.policy_context_sha256,
        corpus=complete_corpus_identity_from_certificate_v5(
            preselection_certificate
        ),
        observed_states=projected_states,
    )
    return bundle, candidate


def _confirmation_aware_noncondition_bundle(
    *,
    manifest: ClaimManifest,
    pipeline_sha256: str,
    budget_minutes: float,
) -> AdaptiveCalibrationBundleV2:
    """Build a tiny v2 contract used to exercise the manifest-v3 runtime.

    These fixtures intentionally contain no condition-dependent calibration
    releases.  The scientific integration test therefore does not treat this bundle
    as evidence for a real release; it is only a complete, self-replaying policy
    artifact for the v6/v7 fail-closed path.
    """

    context = build_verifier_adaptive_policy_context(
        manifest=manifest,
        pipeline_sha256=pipeline_sha256,
        budget_minutes=budget_minutes,
        policy_arm_id="production-adaptive-condition",
    )

    def visible(index: int, split: str):
        question_id = f"condition-calibration-{index}"
        identity = freeze_adaptive_independence_identity_v2(
            strong_components=[
                {
                    "doi": [f"10.7777/condition-calibration-{index}"],
                    "pmid": [str(8_000_000 + index)],
                    "registry_id": [f"clinicaltrials.gov:NCT{8_000_000 + index:08d}"],
                    "dataset_id": [f"zenodo.org:condition-calibration-{index}"],
                }
            ]
        )
        semantics = freeze_adaptive_target_semantics_v2(
            question_id=question_id,
            claim_spec_sha256=hash_canonical({"calibration_claim": index}),
            global_condition_target_sha256=hash_canonical(
                {"calibration_global_target": index}
            ),
        )
        complete_corpus = freeze_complete_corpus_identity(
            corpus_id=question_id,
            corpus_source_sha256=hash_canonical({"calibration_corpus": index}),
            corpus_cutoff=manifest.protocol.corpus_cutoff,
            publication_ids=[f"calibration-publication-{index}"],
            source_manifest_sha256=hash_canonical(
                {"calibration_source_manifest": index}
            ),
        )
        state = freeze_adaptive_preselection_state(
            prefix_index=0,
            audit_prefix_item_ids=[],
            audit_prefix_cost_minutes=0,
            scheduler_state_sha256=hash_canonical({"calibration_state": index}),
            evidence_graph_sha256=hash_canonical({"calibration_graph": index}),
            synthesis_sha256=hash_canonical({"calibration_synthesis": index}),
            non_calibration_assessment_sha256=hash_canonical(
                {"calibration_assessment": index}
            ),
            non_calibration_gates_passed=True,
            non_calibration_blocking_reasons=[],
            claim_decision="supported",
            score_features={
                name: 0.0 for name in CLAIM_RELEASE_RISK_FEATURE_NAMES
            },
        )
        arm = freeze_adaptive_policy_arm_trajectory(
            policy_arm_id=context.policy_arm_id,
            policy_context_sha256=context.policy_context_sha256,
            states=[state],
            terminal_reason="all_items_resolved",
            terminal_candidates=[],
            terminal_source_candidate_input_sha256=hash_canonical([]),
            terminal_remaining_budget_minutes=budget_minutes,
        )
        base_visible = freeze_policy_visible_question_trajectory(
            question_id=question_id,
            split=split,
            population_id=context.population_id,
            domain=manifest.domain,
            corpus=complete_corpus,
            arms=[arm],
        )
        wrapped = freeze_confirmation_aware_arm_trajectory(
            base_arm=arm,
        )
        return freeze_policy_visible_question_trajectory_v2(
            base_visible=base_visible,
            target_semantics=semantics,
            independence_identity=identity,
            arms=[wrapped],
        )

    development_visible = [visible(1, "development"), visible(2, "development")]
    calibration_visible = [visible(3, "calibration"), visible(4, "calibration")]

    def reference(row):
        return freeze_question_reference_verdict_v2(
            question_id=row.base_visible.question_id,
            verdict="supported",
            target_semantics=row.target_semantics,
            label_source="expert_adjudication",
            adjudicator_count=2,
            adjudication_protocol_sha256="7" * 64,
            adjudication_artifact_sha256=hash_canonical(
                {"calibration_reference": row.base_visible.question_id}
            ),
        )

    development = [
        join_labeled_question_trajectory_v2(
            visible=row,
            reference=reference(row),
        )
        for row in development_visible
    ]
    freeze = fit_adaptive_development_v2(
        development,
        policy_contexts=[context],
        calibration_visible_trajectories=calibration_visible,
        alpha=0.99,
        delta=0.5,
        candidate_thresholds={context.policy_arm_id: [1.0]},
        seed=29,
    )
    gate_complete = [
        join_condition_calibration_assessment_receipts(
            visible=row,
            calibration_roster=freeze.calibration_roster,
            calibration_assessment_receipts=[],
        )
        for row in calibration_visible
    ]
    roster = freeze_gate_complete_calibration_roster_v2(
        development_freeze=freeze,
        trajectories=gate_complete,
    )
    return calibrate_confirmation_aware_first_release(
        freeze,
        roster,
        [reference(row) for row in calibration_visible],
    )


def _condition_runtime_fixture():
    question_id = "synthetic-condition-verifier"
    estimand = "between-group standardized difference in performance"
    graphs: list[EvidenceGraph] = []
    for index in range(150):
        level = "high" if index % 2 == 0 else "low"
        estimate = 0.8 if level == "high" else -0.8
        suffix = f"cv-{index:03d}"
        evidence = EffectEvidence(
            paper_id=f"paper-{suffix}",
            finding_id=f"finding-{suffix}",
            outcome="performance",
            contrast="intervention_vs_control",
            effect_format="hedges_g",
            estimate=estimate,
            standard_error=0.10,
            moderators={"dose": level},
            provenance={
                "source_locator": f"paper-{suffix}.pdf#page=4",
                "source_quote": f"The standardized effect was {estimate}.",
            },
        )
        context = GraphAdapterContext(
            publication=PublicationIdentity(
                publication_id=f"publication-{suffix}",
                paper_id=f"paper-{suffix}",
                doc_id=f"document-{suffix}",
                doi=f"10.9998/{suffix}",
                pmid=str(7_000_000 + index),
                title=f"Synthetic condition source {index}",
            ),
            study_id=f"study-{suffix}",
            cohort_identity=CohortIdentity(
                cohort_id=f"cohort-{suffix}",
                basis="reviewer_reconciled",
                source_labels=[f"source cohort {index}"],
                rationale="Synthetic independent identity for verifier tests.",
            ),
            treatment_arm_id=f"arm-{suffix}-treatment",
            comparator_arm_id=f"arm-{suffix}-control",
            contrast_id=f"contrast-{suffix}",
            contrast_label="intervention_vs_control",
            positive_direction_means="higher performance under intervention",
            treatment_label="intervention",
            comparator_label="control",
        )
        payload = adapt_effect_evidence(evidence, context=context).graph.model_dump(
            mode="json"
        )
        payload["contrasts"][0]["estimand"] = estimand
        payload["studies"][0]["registration_ids"] = [f"NCT{index:08d}"]
        graphs.append(EvidenceGraph.model_validate(payload))
    graph = EvidenceGraph(
        publications=[row for item in graphs for row in item.publications],
        studies=[row for item in graphs for row in item.studies],
        cohorts=[row for item in graphs for row in item.cohorts],
        arms=[row for item in graphs for row in item.arms],
        contrasts=[row for item in graphs for row in item.contrasts],
        outcome_estimates=[row for item in graphs for row in item.outcome_estimates],
        evidence_spans=[row for item in graphs for row in item.evidence_spans],
    )
    global_target = freeze_global_condition_dependence_target(
        claim_id="synthetic-global-condition-verifier",
        reference_direction=ClaimDirection.INCREASE,
        outcome_name="performance",
        contrast_label="intervention_vs_control",
        estimand=estimand,
        positive_direction_means="higher performance under intervention",
        treatment_role=ArmRole.INTERVENTION,
        comparator_role=ArmRole.COMPARATOR,
        measure=HarmonizedMeasure.STANDARDIZED_MEAN_DIFFERENCE,
        moderator_names=["dose"],
    )
    manifest = ClaimManifest.model_validate(
        {
            "claim_manifest_version": "3",
            "question_id": question_id,
            "population_id": "synthetic-condition-population",
            "domain": "synthetic-condition-domain",
            "claim": {
                "statement": "The intervention effect depends on dose.",
                "direction": "increase",
                "outcome_name": "performance",
                "estimand": estimand,
            },
            "global_condition_target": global_target.model_dump(mode="json"),
            "protocol": {
                "corpus_cutoff": "2026-08-01T00:00:00Z",
                "inclusion_criteria": ["all synthetic condition studies"],
                "exclusion_criteria": [],
            },
            "release": {
                "require_explicit_timepoint": False,
                "require_prediction_interval_stability": False,
                "prespecified_condition_moderators": ["dose"],
            },
            "audit_guard": {
                "max_unresolved_item_influence": 1.0,
                "max_unresolved_expected_claim_loss": 100.0,
                "block_counterfactual_conclusion_flips": False,
                "require_calibrated_item_scores": True,
                "require_item_cell_rate_ucls": True,
                "max_unresolved_item_cell_ucl_sum": 1.0,
            },
        }
    )
    repository_root = Path(__file__).resolve().parents[1]
    fingerprint = compute_verifier_pipeline_fingerprint(root=repository_root)
    eligibility = tuple(
        CorpusEligibilityRecord(
            paper_id=publication.paper_id,
            title=publication.title,
            status="included",
            reason="Synthetic condition verifier fixture.",
        )
        for publication in graph.publications
    )
    provisional = CorpusLoadResult(
        corpus_id=question_id,
        source_label="embedded:condition-runtime-fixture",
        source_format="embedded_synthetic_fixture",
        source_sha256=hash_canonical({"condition_graph": graph}),
        graph=graph,
        eligibility=eligibility,
        adapter_issues=(),
        metadata={"empirical_evidence": False},
        provenance_assurance=CorpusProvenanceAssurance(
            status="embedded_synthetic_fixture",
            reason="Fixture construction only.",
        ),
    )
    replay_sha256 = hash_canonical({"condition_replay": graph})
    replayed_eligibility, metadata, extraction_context = (
        _source_replayed_fixture_contract(
            manifest=manifest,
            fixture=provisional,
            pipeline_sha256=fingerprint.pipeline_sha256,
            replay_sha256=replay_sha256,
        )
    )
    corpus = CorpusLoadResult(
        corpus_id=question_id,
        source_label="/frozen/native/condition-package.json",
        source_format="typed_evidence_grounding_package_json",
        source_sha256=provisional.source_sha256,
        graph=graph,
        eligibility=replayed_eligibility,
        adapter_issues=(),
        metadata=metadata,
        extraction_context=extraction_context,
        provenance_assurance=CorpusProvenanceAssurance(
            status="source_replayed_native_grounding",
            reason="Fixture emulates an exactly replayed v4 native package.",
            replay_sha256=replay_sha256,
        ),
    )
    complete_identity = complete_corpus_identity_for_adaptive_calibration(
        manifest=manifest,
        corpus=corpus,
    )
    target = freeze_condition_confirmation_target(
        question_id=question_id,
        claim_spec_sha256=global_target.target_sha256,
        question_config_sha256=extraction_context.question_config_sha256,
        corpus_snapshot_sha256=complete_identity.membership_sha256,
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
    roster, development, _confirmation, receipt = (
        materialize_condition_confirmation_inputs(full_graph=graph, target=target)
    )
    plan = prepare_condition_confirmation_plan(
        target=target,
        config=freeze_condition_confirmation_config(),
        roster=roster,
        materialization_receipt=receipt,
        pipeline_sha256=fingerprint.pipeline_sha256,
        external_freeze_anchor="git:condition-runtime-fixture-v1",
    )
    model = fit_condition_confirmation_model(
        plan,
        development,
        current_pipeline_sha256=fingerprint.pipeline_sha256,
    )
    _, _, item_risk_receipt = _artifact_backed_item_risk_contract(
        manifest=manifest,
        corpus=replace(corpus, graph=development),
        fingerprint=fingerprint,
        repository_root=repository_root,
        calibration_unit_count=1000,
    )
    assessment = confirm_condition_dependence(
        plan=plan,
        model=model,
        full_graph=graph,
        current_pipeline_sha256=fingerprint.pipeline_sha256,
    )
    bundle = _confirmation_aware_noncondition_bundle(
        manifest=manifest,
        pipeline_sha256=fingerprint.pipeline_sha256,
        budget_minutes=30,
    )
    return (
        manifest,
        corpus,
        fingerprint,
        plan,
        development,
        model,
        assessment,
        bundle,
        item_risk_receipt,
    )


@pytest.fixture(scope="module")
def condition_runtime_fixture():
    return _condition_runtime_fixture()


@pytest.fixture(scope="module")
def condition_collection_source_fixture(condition_runtime_fixture):
    (
        manifest,
        corpus,
        fingerprint,
        plan,
        development,
        model,
        _assessment,
        _bundle,
        item_risk_receipt,
    ) = condition_runtime_fixture
    policy_context = build_verifier_adaptive_policy_context(
        manifest=manifest,
        pipeline_sha256=fingerprint.pipeline_sha256,
        budget_minutes=30,
        policy_arm_id="condition-collection-arm",
    )
    return run_condition_calibration_collection(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        collection_split="calibration",
        adaptive_policy_context=policy_context,
        condition_plan=plan,
        condition_development_graph=development,
        condition_frozen_model=model,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=Path(__file__).resolve().parents[1],
        item_risk_scoring_receipt=item_risk_receipt,
        generated_at=datetime(2026, 8, 28, 13, tzinfo=UTC),
    )


@pytest.fixture(scope="module")
def condition_v6_source_fixture(condition_runtime_fixture):
    (
        manifest,
        corpus,
        fingerprint,
        plan,
        development,
        model,
        _assessment,
        bundle,
        item_risk_receipt,
    ) = condition_runtime_fixture
    return run_verification(
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
        generated_at=datetime(2026, 8, 28, 12, tzinfo=UTC),
    )


def _materially_corrected_fixture_state(
    *,
    forged_synthesis: bool = False,
    forged_candidates: bool = False,
    no_change: bool = False,
):
    manifest, corpus = build_offline_fixture()
    repository_root = Path(__file__).resolve().parents[1]
    fingerprint = compute_verifier_pipeline_fingerprint(root=repository_root)
    generated_at = datetime(2026, 8, 27, 12, tzinfo=UTC)
    initial = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=repository_root,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=generated_at,
    )
    state = initial.sequential_audit_state
    assert state is not None
    action = state.session.active_action
    assert action is not None
    corrected_payload = state.graph.model_dump(mode="json")
    if not no_change:
        selected = next(
            row
            for row in corrected_payload["outcome_estimates"]
            if row["estimate_id"] == action.item_id
        )
        selected["effect"]["estimate"] = float(selected["effect"]["estimate"]) + 0.75
    corrected_graph = EvidenceGraph.model_validate(corrected_payload)
    verification = verify_pipeline_fingerprint(
        expected=fingerprint,
        root=repository_root,
    )
    prepared = prepare_verification_scientific_state(
        manifest=manifest,
        graph=corrected_graph,
        pipeline_verification=verification,
    )
    expected_candidates = sequential_candidates_from_prepared_state(
        manifest=manifest,
        prepared=prepared,
    )
    expectation = freeze_state_expectation(state)
    adjudication = freeze_selected_adjudication(
        state,
        expected=expectation,
        provenance="benchmark_adjudication",
        adjudicator_count=1,
        protocol_sha256="a" * 64,
        payload_sha256="b" * 64,
        completed_at=datetime(2026, 8, 27, 12, 1, tzinfo=UTC),
        realized_cost=4.5,
    )

    def rerun_synthesis(_graph):
        if forged_synthesis:
            return {"mode": "forged-but-internally-hash-bound", "status": "ok"}
        return prepared.synthesis

    def rerun_candidates(_graph, _synthesis, _session):
        if forged_candidates:
            return state.candidates
        return expected_candidates

    result = resolve_selected_audit_candidate(
        state,
        expected=expectation,
        adjudication=adjudication,
        disposition=(
            CorrectionDisposition.NO_CHANGE
            if no_change
            else CorrectionDisposition.CORRECTED
        ),
        corrected_graph=None if no_change else corrected_graph,
        correction_provenance="benchmark_adjudication",
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
        rerun_synthesis=rerun_synthesis,
        rerun_candidates=rerun_candidates,
    )
    return manifest, corpus, fingerprint, initial, result.state


def test_offline_fixture_runs_complete_fail_closed_path_and_freezes_hashes() -> None:
    manifest, corpus = build_offline_fixture()
    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )

    assert certificate.status == "abstained"
    assert certificate.release_assessment.calibration.status == "not_run"
    assert len(certificate.counterfactual_reruns) == 3
    assert len(certificate.release_assessment.audit.ranking) == 3
    assert all(
        row["scenario"] == "leave_one_out_actual_synthesis_rerun"
        for row in certificate.counterfactual_reruns
    )
    assert certificate.evidence_graph_sha256 == (
        certificate.release_assessment.evidence_graph_sha256
    )
    assert certificate.corpus["provenance_assurance"] == {
        "assurance_version": "corpus-provenance-assurance-v1",
        "reason": (
            "Deterministic embedded synthetic fixture authorized only for mechanical "
            "integration testing; it is not empirical evidence."
        ),
        "release_eligible": False,
        "replay_sha256": None,
        "status": "embedded_synthetic_fixture",
    }
    assert "adapter:unverified_source_provenance" not in certificate.reasons
    assert "adapter:embedded_synthetic_fixture_not_empirical" in certificate.reasons

    tampered = certificate.model_dump(mode="json")
    tampered["corpus"]["metadata"]["purpose"] = "tampered"
    with pytest.raises(
        ValidationError,
        match="verification_certificate_embedded_fixture_identity_invalid",
    ):
        VerificationCertificate.model_validate(tampered)


def test_budget_numeric_spelling_does_not_change_frozen_run_identity() -> None:
    manifest, corpus = build_offline_fixture()
    generated_at = datetime(2026, 8, 27, 12, tzinfo=UTC)

    integer_budget = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        generated_at=generated_at,
    )
    float_budget = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30.0,
        generated_at=generated_at,
    )

    assert integer_budget.run_id == float_budget.run_id
    assert integer_budget.certificate_sha256 == float_budget.certificate_sha256


def test_certificate_rejects_rehashed_internal_release_contradictions() -> None:
    manifest, corpus = build_offline_fixture()
    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )

    def rehash(payload: dict[str, object]) -> dict[str, object]:
        unsigned = {key: value for key, value in payload.items() if key != "certificate_sha256"}
        return {**unsigned, "certificate_sha256": hash_canonical(unsigned)}

    assert VerificationCertificate.model_validate_json(certificate.model_dump_json()) == (
        certificate
    )

    wrong_status = certificate.model_dump(mode="json")
    wrong_status["status"] = "released"
    with pytest.raises(
        ValidationError,
        match="unverified_corpus_cannot_have_released_certificate",
    ):
        VerificationCertificate.model_validate(rehash(wrong_status))

    missing_reason = certificate.model_dump(mode="json")
    missing_reason["reasons"] = []
    with pytest.raises(
        ValidationError,
        match="verification_certificate_reason_ledger_mismatch",
    ):
        VerificationCertificate.model_validate(rehash(missing_reason))

    forged_release_input = certificate.model_dump(mode="json")
    forged_release_input["lineage"][-1]["input_sha256s"]["release_inputs"] = (
        "f" * 64
    )
    with pytest.raises(
        ValidationError,
        match="verification_certificate_release_lineage_mismatch",
    ):
        VerificationCertificate.model_validate(rehash(forged_release_input))

    escalated_assurance = certificate.model_dump(mode="json")
    escalated_assurance["corpus"]["provenance_assurance"]["release_eligible"] = True
    with pytest.raises(
        ValidationError,
        match="verification_certificate_corpus_assurance_escalation",
    ):
        VerificationCertificate.model_validate(rehash(escalated_assurance))

    wrong_run = certificate.model_dump(mode="json")
    wrong_run["run_id"] = "verify-0000000000000000"
    with pytest.raises(
        ValidationError,
        match="verification_certificate_run_identity_mismatch",
    ):
        VerificationCertificate.model_validate(rehash(wrong_run))

    misleading_lineage = certificate.model_dump(mode="json")
    misleading_lineage["lineage"][4]["method"] = "declared-without-rerun"
    with pytest.raises(
        ValidationError,
        match="verification_certificate_counterfactual_lineage_mismatch",
    ):
        VerificationCertificate.model_validate(rehash(misleading_lineage))

    wrong_cutoff = certificate.model_dump(mode="json")
    wrong_cutoff["corpus"]["declared_corpus_cutoff"] = "different-corpus-cutoff"
    with pytest.raises(
        ValidationError,
        match="verification_certificate_corpus_cutoff_mismatch",
    ):
        VerificationCertificate.model_validate(rehash(wrong_cutoff))

    missing_stop_blocker = certificate.model_dump(mode="json")
    stop_decision = missing_stop_blocker["production_stop_decision"]
    stop_decision["blocking_adapter_reasons"] = []
    stop_decision["decision_sha256"] = hash_canonical(
        {
            key: value
            for key, value in stop_decision.items()
            if key != "decision_sha256"
        }
    )
    with pytest.raises(
        ValidationError,
        match="verification_certificate_production_stop_blocker_mismatch",
    ):
        VerificationCertificate.model_validate(rehash(missing_stop_blocker))

    stale_stop_state = certificate.model_dump(mode="json")
    stop_decision = stale_stop_state["production_stop_decision"]
    selection_result = stop_decision["selection_result"]
    selection_result["previous_state_sha256"] = "0" * 64
    selection_result["result_sha256"] = hash_canonical(
        {
            key: value
            for key, value in selection_result.items()
            if key != "result_sha256"
        }
    )
    stop_decision["decision_sha256"] = hash_canonical(
        {
            key: value
            for key, value in stop_decision.items()
            if key != "decision_sha256"
        }
    )
    with pytest.raises(
        ValidationError,
        match="selection_result_transition_mismatch",
    ):
        VerificationCertificate.model_validate(rehash(stale_stop_state))

    wrong_stop_action = certificate.model_dump(mode="json")
    stop_decision = wrong_stop_action["production_stop_decision"]
    selection_result = stop_decision["selection_result"]
    selected_item = selection_result["action"]["item_id"]
    selection_result["candidate"] = next(
        row
        for row in stop_decision["evaluated_state"]["candidates"]
        if row["item_id"] != selected_item
    )
    selection_result["result_sha256"] = hash_canonical(
        {
            key: value
            for key, value in selection_result.items()
            if key != "result_sha256"
        }
    )
    stop_decision["decision_sha256"] = hash_canonical(
        {
            key: value
            for key, value in stop_decision.items()
            if key != "decision_sha256"
        }
    )
    with pytest.raises(
        ValidationError,
        match="selection_result_transition_mismatch",
    ):
        VerificationCertificate.model_validate(rehash(wrong_stop_action))

    assert certificate.sequential_audit_state is not None
    assert certificate.sequential_audit_state.session.active_action is not None
    hidden_active_audit = certificate.model_dump(mode="json")
    assessment = hidden_active_audit["release_assessment"]
    assessment["audit"]["reasons"].remove("active_audit_action_unresolved")
    assessment["reasons"].remove("audit:active_audit_action_unresolved")
    unsigned_assessment = {
        key: value for key, value in assessment.items() if key != "decision_sha256"
    }
    assessment["decision_sha256"] = hash_canonical(unsigned_assessment)
    hidden_active_audit["reasons"].remove("audit:active_audit_action_unresolved")
    hidden_active_audit["lineage"][4]["output_sha256s"]["release_decision"] = assessment[
        "decision_sha256"
    ]
    run_identity = hash_canonical(
        {
            "claim_manifest_sha256": hidden_active_audit["claim_manifest_sha256"],
            "corpus_sha256": hidden_active_audit["corpus_sha256"],
            "evidence_graph_sha256": hidden_active_audit["evidence_graph_sha256"],
            "release_decision_sha256": assessment["decision_sha256"],
            "pipeline_verification_sha256": hidden_active_audit["pipeline_verification"][
                "verification_sha256"
            ],
        }
    )
    hidden_active_audit["run_id"] = f"verify-{run_identity[:16]}"
    with pytest.raises(
        ValidationError,
        match="verification_certificate_active_audit_gate_mismatch",
    ):
        VerificationCertificate.model_validate(rehash(hidden_active_audit))

    predates_audit = certificate.model_dump(mode="json")
    predates_audit["generated_at"] = "2020-01-01T00:00:00Z"
    with pytest.raises(
        ValidationError,
        match="verification_certificate_predates_audit_state",
    ):
        VerificationCertificate.model_validate(rehash(predates_audit))

    invalid_adapter_issue = certificate.model_dump(mode="json")
    invalid_adapter_issue["adapter_issues"] = [
        {
            "code": "fabricated",
            "detail": "Invalid severity must not enter the reason ledger.",
            "finding_id": None,
            "paper_id": None,
            "severity": "block-ish",
        }
    ]
    with pytest.raises(
        ValidationError,
        match="verification_certificate_adapter_issue_invalid",
    ):
        VerificationCertificate.model_validate(rehash(invalid_adapter_issue))


def test_sequential_state_is_bound_to_complete_claim_manifest_context() -> None:
    manifest, corpus = build_offline_fixture()
    initial = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )
    assert initial.sequential_audit_state is not None
    assert initial.sequential_audit_state.session.policy_sha256 == (
        compute_verification_policy_sha256(manifest)
    )

    # The statement does not affect synthesis or candidate identities. Without a
    # complete manifest binding this valid state could therefore be spliced into a
    # different claim certificate while retaining every scientific-state hash.
    changed_manifest = manifest.model_copy(
        update={
            "claim": manifest.claim.model_copy(
                update={"statement": "A different AI-generated scientific claim."}
            )
        }
    )
    with pytest.raises(
        VerificationContractError,
        match="sequential_audit_state_claim_manifest_context_mismatch",
    ):
        run_verification(
            manifest=changed_manifest,
            corpus=corpus,
            budget_minutes=30,
            sequential_audit_state=initial.sequential_audit_state,
            generated_at=datetime(2026, 8, 27, 13, tzinfo=UTC),
        )


def test_uncalibrated_default_cannot_create_or_select_release_audit_state() -> None:
    manifest, corpus = build_offline_fixture()

    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )

    assert certificate.sequential_audit_state is None
    assert certificate.production_stop_decision.outcome == (
        "adaptive_calibration_required_before_audit_genesis"
    )
    assert "adapter:adaptive_calibration_required_before_audit_genesis" in (
        certificate.reasons
    )


def test_uncalibrated_opt_in_is_permanently_analysis_only() -> None:
    manifest, corpus = build_offline_fixture()
    repository_root = Path(__file__).resolve().parents[1]
    fingerprint = compute_verifier_pipeline_fingerprint(root=repository_root)
    analysis = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=repository_root,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )
    assert analysis.sequential_audit_state is not None
    assert analysis.sequential_audit_state.session.active_action is not None
    assert "adapter:uncalibrated_sequential_audit_analysis_only" in analysis.reasons

    bundle, _ = _adaptive_release_contract(
        manifest=manifest,
        preselection_certificate=analysis,
        pipeline_sha256=fingerprint.pipeline_sha256,
        budget_minutes=30,
    )
    with pytest.raises(
        VerificationContractError,
        match="adaptive_calibration_cannot_activate_after_state_genesis",
    ):
        run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=30,
            adaptive_calibration_bundle=bundle,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=repository_root,
            sequential_audit_state=analysis.sequential_audit_state,
            generated_at=datetime(2026, 8, 27, 13, tzinfo=UTC),
        )


def test_standalone_uncalibrated_selection_requires_analysis_opt_in(
    tmp_path: Path,
) -> None:
    manifest, corpus = build_offline_fixture()
    analysis = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )
    assert analysis.sequential_audit_state is not None
    state_path = tmp_path / "state.json"
    state_path.write_text(
        analysis.sequential_audit_state.model_dump_json(indent=2),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="uncalibrated_audit_selection_requires_analysis_only_opt_in",
    ):
        cli_main(
            [
                "audit-select",
                "--state",
                str(state_path),
                "--output-dir",
                str(tmp_path / "selection"),
            ]
        )


def test_source_replayed_corpus_from_different_pipeline_is_release_blocked() -> None:
    manifest, fixture = build_offline_fixture()
    replay_sha256 = "e" * 64
    eligibility, metadata, extraction_context = _source_replayed_fixture_contract(
        manifest=manifest,
        fixture=fixture,
        pipeline_sha256="f" * 64,
        replay_sha256=replay_sha256,
    )
    mismatched = type(fixture)(
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
    assert mismatched.provenance_release_eligible() is True

    certificate = run_verification(
        manifest=manifest,
        corpus=mismatched,
        budget_minutes=30,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )

    assert certificate.status == "abstained"
    assert "adapter:corpus_pipeline_identity_mismatch" in certificate.reasons
    assert any(
        issue["severity"] == "blocking"
        and issue["code"] == "corpus_pipeline_identity_mismatch"
        for issue in certificate.adapter_issues
    )

    forged = certificate.model_dump(mode="json", exclude={"certificate_sha256"})
    forged["adapter_issues"] = [
        issue
        for issue in forged["adapter_issues"]
        if issue["code"] != "corpus_pipeline_identity_mismatch"
    ]
    forged["reasons"] = [
        reason
        for reason in forged["reasons"]
        if reason != "adapter:corpus_pipeline_identity_mismatch"
    ]
    with pytest.raises(
        ValidationError,
        match="verification_certificate_corpus_pipeline_mismatch_requires_blocker",
    ):
        VerificationCertificate.model_validate(
            {**forged, "certificate_sha256": hash_canonical(forged)}
        )

    truncated_membership = certificate.model_dump(
        mode="json", exclude={"certificate_sha256"}
    )
    native_manifest = truncated_membership["corpus"]["metadata"][
        "native_source_manifest"
    ]
    native_manifest["records"].pop()
    truncated_membership["corpus"]["metadata"]["source_manifest_records"] -= 1
    truncated_membership["corpus"]["metadata"]["source_manifest_sha256"] = (
        hash_canonical(native_manifest)
    )
    with pytest.raises(
        ValidationError,
        match="verification_certificate_native_source_eligibility_membership_mismatch",
    ):
        VerificationCertificate.model_validate(
            {
                **truncated_membership,
                "certificate_sha256": hash_canonical(truncated_membership),
            }
        )

    omitted_eligibility = certificate.model_dump(
        mode="json", exclude={"certificate_sha256"}
    )
    omitted_eligibility["corpus"]["eligibility"].pop()
    with pytest.raises(
        ValidationError,
        match="verification_certificate_native_source_eligibility_membership_mismatch",
    ):
        VerificationCertificate.model_validate(
            {
                **omitted_eligibility,
                "certificate_sha256": hash_canonical(omitted_eligibility),
            }
        )

    for mutate_terminal_membership in (
        lambda rows: rows.pop(),
        lambda rows: rows.append(dict(rows[0])),
    ):
        invalid_terminal_membership = certificate.model_dump(
            mode="json", exclude={"certificate_sha256"}
        )
        terminal_membership = invalid_terminal_membership["corpus"]["metadata"][
            "terminal_fragment_membership"
        ]
        mutate_terminal_membership(terminal_membership)
        invalid_terminal_membership["corpus"]["metadata"][
            "terminal_fragment_records"
        ] = len(terminal_membership)
        invalid_terminal_membership["corpus"]["metadata"][
            "terminal_fragment_membership_sha256"
        ] = hash_canonical(terminal_membership)
        with pytest.raises(
            ValidationError,
            match="verification_certificate_terminal_fragment_membership_mismatch",
        ):
            VerificationCertificate.model_validate(
                {
                    **invalid_terminal_membership,
                    "certificate_sha256": hash_canonical(invalid_terminal_membership),
                }
            )

    redirected_source = certificate.model_dump(
        mode="json", exclude={"certificate_sha256"}
    )
    native_manifest = redirected_source["corpus"]["metadata"][
        "native_source_manifest"
    ]
    native_manifest["records"][0]["source_document"]["source_locator"] = (
        "json:data/raw/other.json#/forged"
    )
    redirected_source["corpus"]["metadata"]["source_manifest_sha256"] = (
        hash_canonical(native_manifest)
    )
    with pytest.raises(
        ValidationError,
        match="verification_certificate_native_source_eligibility_mismatch",
    ):
        VerificationCertificate.model_validate(
            {
                **redirected_source,
                "certificate_sha256": hash_canonical(redirected_source),
            }
        )


def test_source_replayed_corpus_question_identity_mismatch_is_release_blocked() -> None:
    manifest, fixture = build_offline_fixture()
    repository_root = Path(__file__).resolve().parents[1]
    current_pipeline = compute_verifier_pipeline_fingerprint(root=repository_root)
    replay_sha256 = "e" * 64
    eligibility, metadata, extraction_context = _source_replayed_fixture_contract(
        manifest=manifest,
        fixture=fixture,
        pipeline_sha256=current_pipeline.pipeline_sha256,
        replay_sha256=replay_sha256,
    )
    mismatched = type(fixture)(
        corpus_id="different-review-question",
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

    certificate = run_verification(
        manifest=manifest,
        corpus=mismatched,
        budget_minutes=30,
        expected_pipeline_fingerprint=current_pipeline,
        pipeline_root=repository_root,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )

    assert certificate.status == "abstained"
    assert "adapter:corpus_question_identity_mismatch" in certificate.reasons
    forged = certificate.model_dump(mode="json", exclude={"certificate_sha256"})
    forged["adapter_issues"] = [
        issue
        for issue in forged["adapter_issues"]
        if issue["code"] != "corpus_question_identity_mismatch"
    ]
    forged["reasons"] = [
        reason
        for reason in forged["reasons"]
        if reason != "adapter:corpus_question_identity_mismatch"
    ]
    with pytest.raises(
        ValidationError,
        match="verification_certificate_corpus_question_mismatch_requires_blocker",
    ):
        VerificationCertificate.model_validate(
            {**forged, "certificate_sha256": hash_canonical(forged)}
        )


def test_membership_bound_native_corpus_cutoff_mismatch_is_release_blocked() -> None:
    manifest, fixture = build_offline_fixture()
    repository_root = Path(__file__).resolve().parents[1]
    current_pipeline = compute_verifier_pipeline_fingerprint(root=repository_root)
    replay_sha256 = "e" * 64
    eligibility, metadata, extraction_context = _source_replayed_fixture_contract(
        manifest=manifest,
        fixture=fixture,
        pipeline_sha256=current_pipeline.pipeline_sha256,
        replay_sha256=replay_sha256,
        corpus_cutoff="different-frozen-corpus-v2",
    )
    mismatched = type(fixture)(
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

    certificate = run_verification(
        manifest=manifest,
        corpus=mismatched,
        budget_minutes=30,
        expected_pipeline_fingerprint=current_pipeline,
        pipeline_root=repository_root,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )

    assert certificate.status == "abstained"
    assert "adapter:corpus_cutoff_identity_mismatch" in certificate.reasons
    forged = certificate.model_dump(mode="json", exclude={"certificate_sha256"})
    forged["adapter_issues"] = [
        issue
        for issue in forged["adapter_issues"]
        if issue["code"] != "corpus_cutoff_identity_mismatch"
    ]
    forged["reasons"] = [
        reason
        for reason in forged["reasons"]
        if reason != "adapter:corpus_cutoff_identity_mismatch"
    ]
    with pytest.raises(
        ValidationError,
        match="verification_certificate_native_cutoff_mismatch_requires_blocker",
    ):
        VerificationCertificate.model_validate(
            {**forged, "certificate_sha256": hash_canonical(forged)}
        )


def test_cli_fixture_writes_self_contained_json_and_html(tmp_path, capsys) -> None:
    output = tmp_path / "certificate"
    result = cli_main(
        [
            "verify",
            "--fixture",
            "--budget-minutes",
            "30",
            "--analysis-only-uncalibrated-audit",
            "--output-dir",
            str(output),
        ]
    )

    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    payload = json.loads((output / "verification-certificate.json").read_text())
    html = (output / "verification-certificate.html").read_text()
    certificate = VerificationCertificate.model_validate(payload)
    assert summary["certificate_sha256"] == certificate.certificate_sha256
    assert certificate.run_id in html
    assert certificate.certificate_sha256 in html
    assert "embedded_synthetic_fixture" in html
    assert "Complete normative JSON payload" in html
    assert "<script" not in html


def test_cli_freezes_and_reverifies_computed_pipeline(tmp_path, capsys) -> None:
    fingerprint_path = tmp_path / "pipeline.json"
    assert cli_main(["fingerprint", "--output", str(fingerprint_path)]) == 0
    fingerprint_summary = json.loads(capsys.readouterr().out)

    output = tmp_path / "verified"
    assert (
        cli_main(
            [
                "verify",
                "--fixture",
                "--budget-minutes",
                "30",
                "--pipeline-fingerprint",
                str(fingerprint_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()
    certificate = VerificationCertificate.model_validate_json(
        (output / "verification-certificate.json").read_text()
    )
    assert certificate.pipeline_verification.status == "matched"
    assert (
        certificate.pipeline_verification.computed_pipeline_sha256
        == fingerprint_summary["pipeline_sha256"]
    )


def test_claim_manifest_cannot_self_declare_calibrated_item_probabilities() -> None:
    with pytest.raises(
        ValidationError,
        match="claim_manifest_probability_basis_must_be_heuristic",
    ):
        AuditPolicyConfig(probability_basis="calibrated_upper_bound")


def test_claim_manifest_rejects_ambiguous_legacy_guard_risk_aliases() -> None:
    with pytest.raises(ValidationError, match="max_residual_decision_risk"):
        AuditGuardConfig.model_validate({"max_residual_decision_risk": 0.05})

    with pytest.raises(
        ValidationError,
        match="require_error_probability_upper_bounds",
    ):
        AuditGuardConfig.model_validate(
            {"require_error_probability_upper_bounds": True}
        )


def test_v2_qualified_claim_runs_exact_condition_and_magnitude_path() -> None:
    original_manifest, original_corpus = build_offline_fixture()
    graph = original_corpus.graph.model_copy(
        update={
            "outcome_estimates": [
                estimate.model_copy(
                    update={
                        "effect": estimate.effect.model_copy(
                            update={"moderators": {"setting": "laboratory"}}
                        )
                    }
                )
                for estimate in original_corpus.graph.outcome_estimates
            ]
        }
    )
    target = freeze_claim_target_v2(
        claim_id="offline-qualified-fixture",
        direction=ClaimDirection.INCREASE,
        outcome_name="fixture_outcome",
        conditions=[
            ConditionPredicate(
                moderator="setting",
                operator=ConditionOperator.EQUALS,
                value="laboratory",
            )
        ],
        meaningful_effect_threshold=MeaningfulEffectThreshold(
            minimum_magnitude=0.1,
            measure=HarmonizedMeasure.STANDARDIZED_MEAN_DIFFERENCE,
        ),
    )
    payload = original_manifest.model_dump(mode="json")
    payload.update(
        {
            "claim_manifest_version": "2",
            "qualified_target": target.model_dump(mode="json"),
            "release": {
                **payload["release"],
                "prespecified_condition_moderators": ["setting"],
            },
        }
    )
    manifest = ClaimManifest.model_validate(payload)
    corpus = type(original_corpus)(
        corpus_id=original_corpus.corpus_id,
        source_label=original_corpus.source_label,
        source_format=original_corpus.source_format,
        source_sha256=original_corpus.source_sha256,
        graph=graph,
        eligibility=original_corpus.eligibility,
        adapter_issues=original_corpus.adapter_issues,
        metadata=original_corpus.metadata,
        provenance_assurance=original_corpus.provenance_assurance,
    )

    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )

    assert certificate.release_assessment.assessment_version == (
        "prospective-qualified-claim-release-v2"
    )
    assert certificate.release_assessment.target.claim_sha256 == target.claim_sha256
    assert all(
        row["scenario"] == "leave_one_out_actual_qualified_synthesis_rerun"
        for row in certificate.counterfactual_reruns
    )
    assert certificate.status == "abstained"


def test_v3_global_condition_target_is_explicit_and_distinct_from_v2() -> None:
    original, _ = build_offline_fixture()
    target = freeze_global_condition_dependence_target(
        claim_id="offline-global-condition-dependence",
        reference_direction=ClaimDirection.INCREASE,
        outcome_name="fixture_outcome",
        contrast_label="intervention_vs_control",
        estimand=original.claim.estimand,
        positive_direction_means="higher fixture outcome under intervention",
        treatment_role=ArmRole.INTERVENTION,
        comparator_role=ArmRole.COMPARATOR,
        measure=HarmonizedMeasure.STANDARDIZED_MEAN_DIFFERENCE,
        moderator_names=["setting"],
    )
    payload = original.model_dump(mode="json")
    payload.update(
        {
            "claim_manifest_version": "3",
            "global_condition_target": target.model_dump(mode="json"),
            "release": {
                **payload["release"],
                "prespecified_condition_moderators": ["setting"],
            },
        }
    )

    manifest = ClaimManifest.model_validate(payload)

    assert manifest.global_condition_target == target
    assert manifest.qualified_target is None
    swapped = payload.copy()
    swapped["claim_manifest_version"] = "2"
    with pytest.raises(
        ValidationError,
        match="v2_claim_manifest_requires_qualified_target",
    ):
        ClaimManifest.model_validate(swapped)


def test_v3_freezes_outcome_free_v6_then_finalizes_exact_source_v7(
    condition_runtime_fixture,
    condition_v6_source_fixture,
) -> None:
    (
        _manifest,
        _corpus,
        _fingerprint,
        _plan,
        _development,
        _model,
        assessment,
        bundle,
        _item_risk_receipt,
    ) = condition_runtime_fixture
    generated_at = datetime(2026, 8, 28, 12, tzinfo=UTC)
    source = condition_v6_source_fixture

    assert isinstance(source, ConditionVerificationCertificateV6)
    assert source.status == "abstained"
    assert source.condition_confirmation_assessment is None
    assert source.condition_confirmation_gate.status == "missing"
    assert source.release_assessment.terminal_gate_deferred is True
    assert source.production_stop_decision.outcome == "condition_gate_ready"
    assert source.condition_gate_invocation_proof is not None
    source_sha256 = source.certificate_sha256
    source_payload = source.model_dump(mode="json")

    final = finalize_condition_verification(
        source_certificate=source,
        condition_confirmation_assessment=assessment,
        generated_at=generated_at,
    )

    assert isinstance(final, FinalConditionVerificationCertificateV7)
    assert final.source_v6_certificate_sha256 == source_sha256
    assert final.source_certificate_v6.model_dump(mode="json") == source_payload
    assert final.condition_confirmation_assessment == assessment
    assert final.condition_confirmation_gate.status == assessment.status
    assert final.terminal_gate_result.source_v6_certificate_sha256 == source_sha256
    assert (
        final.terminal_gate_result.source_v6_decision_sha256
        == source.release_assessment.decision_sha256
    )
    # This fixture's calibration questions intentionally contain no condition
    # releases. Confirmation-aware v2 must therefore remain fail-closed.
    assert bundle.selected is None
    assert final.status == "abstained"
    replay = freeze_question_replay_state_from_certificate(final)
    assert replay.release_assessment_sha256 == final.release_assessment.decision_sha256
    assert QuestionReplayState.model_validate(replay.model_dump(mode="json")) == replay

    substituted = source.model_dump(mode="json", exclude={"certificate_sha256"})
    substituted["claim_manifest"]["population_id"] = "substituted-population"
    substituted["claim_manifest_sha256"] = hash_canonical(
        substituted["claim_manifest"]
    )
    with pytest.raises(ValidationError, match="condition_v6_pipeline_verification_mismatch"):
        ConditionVerificationCertificateV6.model_validate(
            {**substituted, "certificate_sha256": hash_canonical(substituted)}
        )
    substituted_question = source.model_dump(
        mode="json", exclude={"certificate_sha256"}
    )
    substituted_question["claim_manifest"]["question_id"] = "substituted-question"
    substituted_question["claim_manifest_sha256"] = hash_canonical(
        substituted_question["claim_manifest"]
    )
    with pytest.raises(ValidationError):
        ConditionVerificationCertificateV6.model_validate(
            {
                **substituted_question,
                "certificate_sha256": hash_canonical(substituted_question),
            }
        )
    substituted_corpus = source.model_dump(
        mode="json", exclude={"certificate_sha256"}
    )
    substituted_corpus["corpus"]["corpus_id"] = "substituted-corpus"
    with pytest.raises(ValidationError):
        ConditionVerificationCertificateV6.model_validate(
            {
                **substituted_corpus,
                "certificate_sha256": hash_canonical(substituted_corpus),
            }
        )
    with pytest.raises(
        VerificationContractError,
        match="condition_v7_generated_at_precedes_source_v6",
    ):
        finalize_condition_verification(
            source_certificate=source,
            condition_confirmation_assessment=assessment,
            generated_at=source.generated_at - timedelta(microseconds=1),
        )


def test_v3_finalizer_rejects_preinvocation_and_assessment_or_source_tamper(
    condition_runtime_fixture,
) -> None:
    (
        manifest,
        corpus,
        fingerprint,
        plan,
        development,
        model,
        assessment,
        _bundle,
        item_risk_receipt,
    ) = condition_runtime_fixture
    strict_manifest = manifest.model_copy(
        update={"audit_guard": AuditGuardConfig()}
    )
    strict_bundle = _confirmation_aware_noncondition_bundle(
        manifest=strict_manifest,
        pipeline_sha256=fingerprint.pipeline_sha256,
        budget_minutes=30,
    )
    nonterminal = run_verification(
        manifest=strict_manifest,
        corpus=corpus,
        budget_minutes=30,
        adaptive_calibration_bundle_v2=strict_bundle,
        condition_plan=plan,
        condition_development_graph=development,
        condition_frozen_model=model,
        item_risk_scoring_receipt=item_risk_receipt,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=Path(__file__).resolve().parents[1],
        generated_at=datetime(2026, 8, 28, 12, tzinfo=UTC),
    )
    assert isinstance(nonterminal, ConditionVerificationCertificateV6)
    assert nonterminal.production_stop_decision.outcome != "condition_gate_ready"
    with pytest.raises(
        VerificationContractError,
        match="condition_v7_source_not_outcome_free_gate_ready",
    ):
        finalize_condition_verification(
            source_certificate=nonterminal,
            condition_confirmation_assessment=assessment,
        )

    tampered_assessment = assessment.model_copy(
        update={"pipeline_sha256": "f" * 64}
    )
    source = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        adaptive_calibration_bundle_v2=_bundle,
        condition_plan=plan,
        condition_development_graph=development,
        condition_frozen_model=model,
        item_risk_scoring_receipt=item_risk_receipt,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=Path(__file__).resolve().parents[1],
        generated_at=datetime(2026, 8, 28, 12, tzinfo=UTC),
    )
    with pytest.raises(
        VerificationContractError,
        match="condition_v7_finalizer_input_integrity_changed",
    ):
        finalize_condition_verification(
            source_certificate=source,
            condition_confirmation_assessment=tampered_assessment,
        )

    tampered_source = source.model_copy(
        update={"certificate_sha256": "f" * 64}
    )
    with pytest.raises(VerificationContractError):
        finalize_condition_verification(
            source_certificate=tampered_source,
            condition_confirmation_assessment=assessment,
        )


def test_verify_public_cli_lazily_finalizes_outcome_free_v6_to_v7(
    condition_runtime_fixture,
    condition_v6_source_fixture,
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
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
    ) = condition_runtime_fixture
    source = condition_v6_source_fixture
    repository_root = Path(__file__).resolve().parents[1]
    inputs = {
        "claim": (tmp_path / "claim-v3.json", manifest),
        "bundle": (tmp_path / "adaptive-calibration-v2.json", bundle),
        "plan": (tmp_path / "condition-plan.json", plan),
        "development": (tmp_path / "condition-development-graph.json", development),
        "model": (tmp_path / "condition-model.json", model),
        "assessment": (tmp_path / "condition-assessment.json", assessment),
        "fingerprint": (tmp_path / "pipeline-fingerprint.json", fingerprint),
        "item-risk": (tmp_path / "item-risk-scoring-receipt.json", item_risk_receipt),
    }
    for path, value in inputs.values():
        path.write_text(value.model_dump_json(indent=2))

    monkeypatch.setattr(cli_module, "load_corpus", lambda *_args, **_kwargs: corpus)
    monkeypatch.setattr(cli_module, "run_verification", lambda **_kwargs: source)
    output_dir = tmp_path / "verification-v7"
    assert (
        cli_main(
            [
                "verify",
                "--claim",
                str(inputs["claim"][0]),
                "--corpus",
                str(tmp_path / "native-package.json"),
                "--budget-minutes",
                "30",
                "--condition-adaptive-calibration",
                str(inputs["bundle"][0]),
                "--condition-plan",
                str(inputs["plan"][0]),
                "--condition-development-graph",
                str(inputs["development"][0]),
                "--condition-model",
                str(inputs["model"][0]),
                "--condition-assessment",
                str(inputs["assessment"][0]),
                "--pipeline-fingerprint",
                str(inputs["fingerprint"][0]),
                "--pipeline-root",
                str(repository_root),
                "--item-risk-scoring-receipt",
                str(inputs["item-risk"][0]),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    certificate_payload = json.loads(
        (output_dir / "verification-certificate.json").read_text()
    )
    assert summary["certificate_version"] == (
        "literature-multiverse-condition-verification-v7"
    )
    assert certificate_payload["source_v6_certificate_sha256"] == (
        source.certificate_sha256
    )
    assert certificate_payload["condition_confirmation_assessment"][
        "assessment_sha256"
    ] == assessment.assessment_sha256
    assert certificate_payload["terminal_gate_result"][
        "source_v6_certificate_sha256"
    ] == source.certificate_sha256
    assert (output_dir / "verification-certificate.html").is_file()


def test_run_verification_refuses_outcome_bearing_assessment_input(
    condition_runtime_fixture,
) -> None:
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
    ) = (
        condition_runtime_fixture
    )
    with pytest.raises(
        VerificationContractError,
        match="condition_terminal_assessment_requires_dedicated_finalizer",
    ):
        run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=30,
            adaptive_calibration_bundle_v2=bundle,
            condition_plan=plan,
            condition_development_graph=development,
            condition_frozen_model=model,
            condition_confirmation_assessment=assessment,
            item_risk_scoring_receipt=item_risk_receipt,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=Path(__file__).resolve().parents[1],
        )


def test_condition_calibration_collection_source_and_receipt_exact_replay(
    condition_runtime_fixture,
    condition_collection_source_fixture,
) -> None:
    (
        _manifest,
        _corpus,
        _fingerprint,
        _plan,
        _development,
        _model,
        assessment,
        _bundle,
        _item_risk_receipt,
    ) = condition_runtime_fixture
    repository_root = Path(__file__).resolve().parents[1]
    source = condition_collection_source_fixture

    assert isinstance(source, ConditionCalibrationCollectionSourceV1)
    assert source.status == "abstained"
    assert source.collection_decision.outcome == "condition_gate_ready"
    assert source.policy_visible_question_trajectory is not None
    assert source.adaptive_calibration_bundle_unavailable is True
    source_roster = freeze_condition_calibration_collection_source_roster_v1(
        [source]
    )
    receipt = freeze_condition_calibration_assessment_receipt_v1(
        collection_source_roster=source_roster,
        collection_source=source,
        condition_confirmation_assessment=assessment,
    )
    assert isinstance(receipt, ConditionCalibrationAssessmentReceiptV1)
    assert receipt.calibration_gate_result.status == assessment.status
    assert receipt.collection_source_sha256 == source.collection_source_sha256
    assert (
        validate_condition_calibration_assessment_receipt_external_replay(
            receipt,
            collection_source_roster=source_roster,
            pipeline_root=repository_root,
        )
        == receipt
    )
    assert (
        ConditionCalibrationAssessmentReceiptV1.model_validate(
            receipt.model_dump(mode="json")
        )
        == receipt
    )

    tampered = source.model_dump(mode="json", exclude={"collection_source_sha256"})
    tampered["claim_manifest"]["population_id"] = "substituted-population"
    tampered["claim_manifest_sha256"] = hash_canonical(tampered["claim_manifest"])
    with pytest.raises(
        ValidationError,
        match="condition_collection_scientific_context_mismatch",
    ):
        ConditionCalibrationCollectionSourceV1.model_validate(
            {**tampered, "collection_source_sha256": hash_canonical(tampered)}
        )

    substituted_question = source.model_dump(
        mode="json", exclude={"collection_source_sha256"}
    )
    substituted_question["claim_manifest"]["question_id"] = "substituted-question"
    substituted_question["claim_manifest_sha256"] = hash_canonical(
        substituted_question["claim_manifest"]
    )
    with pytest.raises(
        ValidationError,
        match="condition_collection_scientific_context_mismatch",
    ):
        ConditionCalibrationCollectionSourceV1.model_validate(
            {
                **substituted_question,
                "collection_source_sha256": hash_canonical(substituted_question),
            }
        )

    substituted_corpus = source.model_dump(
        mode="json", exclude={"collection_source_sha256"}
    )
    substituted_corpus["corpus"]["corpus_id"] = "substituted-corpus"
    with pytest.raises(
        ValidationError,
        match="condition_collection_scientific_context_mismatch",
    ):
        ConditionCalibrationCollectionSourceV1.model_validate(
            {
                **substituted_corpus,
                "collection_source_sha256": hash_canonical(substituted_corpus),
            }
        )

    excluded = source.model_dump(mode="json", exclude={"collection_source_sha256"})
    excluded["corpus"]["eligibility"][0]["status"] = "excluded"
    with pytest.raises(
        VerificationContractError,
        match="condition_collection_external_eligibility_membership_mismatch",
    ):
        validate_condition_calibration_collection_source_external_replay(
            ConditionCalibrationCollectionSourceV1.model_validate(
                {**excluded, "collection_source_sha256": hash_canonical(excluded)}
            ),
            pipeline_root=repository_root,
        )

    malformed_terminal = source.model_dump(
        mode="json", exclude={"collection_source_sha256"}
    )
    terminal_rows = malformed_terminal["corpus"]["metadata"][
        "terminal_fragment_membership"
    ]
    terminal_rows[0]["unexpected"] = "coherently-rehashed"
    malformed_terminal["corpus"]["metadata"][
        "terminal_fragment_membership_sha256"
    ] = hash_canonical(terminal_rows)
    with pytest.raises(
        VerificationContractError,
        match="condition_collection_external_terminal_membership_invalid",
    ):
        validate_condition_calibration_collection_source_external_replay(
            ConditionCalibrationCollectionSourceV1.model_validate(
                {
                    **malformed_terminal,
                    "collection_source_sha256": hash_canonical(malformed_terminal),
                }
            ),
            pipeline_root=repository_root,
        )


def test_condition_calibration_receipt_forbids_development_outcome_opening(
    condition_runtime_fixture,
) -> None:
    (
        manifest,
        corpus,
        fingerprint,
        plan,
        development,
        model,
        _assessment,
        _bundle,
        item_risk_receipt,
    ) = condition_runtime_fixture
    source = run_condition_calibration_collection(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        collection_split="development",
        adaptive_policy_context=build_verifier_adaptive_policy_context(
            manifest=manifest,
            pipeline_sha256=fingerprint.pipeline_sha256,
            budget_minutes=30,
            policy_arm_id="condition-development-collection-arm",
        ),
        condition_plan=plan,
        condition_development_graph=development,
        condition_frozen_model=model,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=Path(__file__).resolve().parents[1],
        item_risk_scoring_receipt=item_risk_receipt,
        generated_at=datetime(2026, 8, 28, 13, tzinfo=UTC),
    )
    with pytest.raises(AdaptiveCalibrationError, match="source_split_mismatch"):
        freeze_condition_calibration_collection_source_roster_v1([source])


def test_condition_collection_and_finalize_public_cli_roundtrip(
    condition_runtime_fixture,
    condition_collection_source_fixture,
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    (
        manifest,
        corpus,
        fingerprint,
        plan,
        development,
        model,
        assessment,
        _bundle,
        item_risk_receipt,
    ) = condition_runtime_fixture
    source = condition_collection_source_fixture
    repository_root = Path(__file__).resolve().parents[1]

    claim_path = tmp_path / "claim-v3.json"
    plan_path = tmp_path / "condition-plan.json"
    development_path = tmp_path / "condition-development-graph.json"
    model_path = tmp_path / "condition-model.json"
    policy_path = tmp_path / "condition-policy-context.json"
    fingerprint_path = tmp_path / "pipeline-fingerprint.json"
    item_risk_path = tmp_path / "item-risk-scoring-receipt.json"
    assessment_path = tmp_path / "condition-assessment.json"
    claim_path.write_text(manifest.model_dump_json(indent=2))
    plan_path.write_text(plan.model_dump_json(indent=2))
    development_path.write_text(development.model_dump_json(indent=2))
    model_path.write_text(model.model_dump_json(indent=2))
    policy_path.write_text(source.adaptive_policy_context.model_dump_json(indent=2))
    fingerprint_path.write_text(fingerprint.model_dump_json(indent=2))
    item_risk_path.write_text(item_risk_receipt.model_dump_json(indent=2))
    assessment_path.write_text(assessment.model_dump_json(indent=2))

    monkeypatch.setattr(cli_module, "load_corpus", lambda *_args, **_kwargs: corpus)
    monkeypatch.setattr(
        cli_module,
        "run_condition_calibration_collection",
        lambda **_kwargs: source,
    )
    collection_dir = tmp_path / "collection-output"
    assert (
        cli_main(
            [
                "condition-collect",
                "--claim",
                str(claim_path),
                "--corpus",
                str(tmp_path / "native-package.json"),
                "--budget-minutes",
                "30",
                "--split",
                "calibration",
                "--policy-context",
                str(policy_path),
                "--condition-plan",
                str(plan_path),
                "--condition-development-graph",
                str(development_path),
                "--condition-model",
                str(model_path),
                "--pipeline-fingerprint",
                str(fingerprint_path),
                "--pipeline-root",
                str(repository_root),
                "--item-risk-scoring-receipt",
                str(item_risk_path),
                "--output-dir",
                str(collection_dir),
            ]
        )
        == 0
    )
    collection_summary = json.loads(capsys.readouterr().out)
    source_path = collection_dir / "condition-calibration-collection-source.json"
    assert collection_summary["collection_source_sha256"] == source.collection_source_sha256
    assert source_path.is_file()

    sources_jsonl = tmp_path / "condition-collection-sources.jsonl"
    sources_jsonl.write_text(source_path.read_text().strip() + "\n")
    roster_path = tmp_path / "condition-collection-source-roster.json"
    assert (
        adaptive_calibration_cli.main(
            [
                "freeze-collection-sources-v2",
                "--collection-sources",
                str(sources_jsonl),
                "--output",
                str(roster_path),
            ]
        )
        == 0
    )
    roster_summary = json.loads(capsys.readouterr().out)
    roster = ConditionCalibrationCollectionSourceRosterV1.model_validate_json(
        roster_path.read_text()
    )
    assert roster_summary["condition_assessments_opened"] is False
    assert roster_summary["source_roster_sha256"] == roster.source_roster_sha256
    assert roster_summary["source_membership_sha256"] == roster.source_membership_sha256

    receipt_path = tmp_path / "condition-calibration-assessment-receipt.json"
    assert (
        cli_main(
            [
                "condition-finalize-calibration",
                "--source-roster",
                str(roster_path),
                "--expected-source-roster-sha256",
                roster.source_roster_sha256,
                "--expected-source-membership-sha256",
                roster.source_membership_sha256,
                "--source",
                str(source_path),
                "--condition-assessment",
                str(assessment_path),
                "--output",
                str(receipt_path),
            ]
        )
        == 0
    )
    receipt_summary = json.loads(capsys.readouterr().out)
    receipt_payload = json.loads(receipt_path.read_text())
    assert receipt_summary["status"] == assessment.status
    assert receipt_payload["source_anchor"] == roster.source_anchors[0].model_dump(
        mode="json"
    )
    assert receipt_payload["source_roster_sha256"] == roster.source_roster_sha256
    assert (
        receipt_payload["source_membership_sha256"]
        == roster.source_membership_sha256
    )


def test_condition_finalize_cli_preflights_output_before_any_input_open(
    monkeypatch,
    tmp_path,
) -> None:
    output = tmp_path / "existing-receipt.json"
    output.write_text("already frozen")
    opened: list[str] = []

    def forbidden_source(_path):
        opened.append("source")
        raise AssertionError("source must not open after output collision")

    def forbidden_roster(_path):
        opened.append("roster")
        raise AssertionError("roster must not open after output collision")

    def forbidden_assessment(_path):
        opened.append("assessment")
        raise AssertionError("assessment must not open after output collision")

    monkeypatch.setattr(cli_module, "_condition_collection_source", forbidden_source)
    monkeypatch.setattr(
        cli_module,
        "_condition_collection_source_roster",
        forbidden_roster,
    )
    monkeypatch.setattr(
        cli_module,
        "_condition_confirmation_assessment",
        forbidden_assessment,
    )
    with pytest.raises(FileExistsError, match="receipt_output_exists"):
        cli_main(
            [
                "condition-finalize-calibration",
                "--source-roster",
                str(tmp_path / "source-roster.json"),
                "--expected-source-roster-sha256",
                "0" * 64,
                "--expected-source-membership-sha256",
                "1" * 64,
                "--source",
                str(tmp_path / "source.json"),
                "--condition-assessment",
                str(tmp_path / "assessment.json"),
                "--output",
                str(output),
            ]
        )
    assert opened == []


def test_condition_finalize_cli_checks_roster_anchor_before_source_or_assessment(
    monkeypatch,
    tmp_path,
) -> None:
    opened: list[str] = []
    roster = SimpleNamespace(
        source_roster_sha256="a" * 64,
        source_membership_sha256="b" * 64,
    )
    monkeypatch.setattr(
        cli_module,
        "_condition_collection_source_roster",
        lambda _path: roster,
    )

    def forbidden_source(_path):
        opened.append("source")
        raise AssertionError("source must not open before roster anchor matches")

    def forbidden_assessment(_path):
        opened.append("assessment")
        raise AssertionError("assessment must not open before roster anchor matches")

    monkeypatch.setattr(cli_module, "_condition_collection_source", forbidden_source)
    monkeypatch.setattr(
        cli_module,
        "_condition_confirmation_assessment",
        forbidden_assessment,
    )
    with pytest.raises(ValueError, match="roster_external_anchor_mismatch"):
        cli_main(
            [
                "condition-finalize-calibration",
                "--source-roster",
                str(tmp_path / "source-roster.json"),
                "--expected-source-roster-sha256",
                "0" * 64,
                "--expected-source-membership-sha256",
                "1" * 64,
                "--source",
                str(tmp_path / "source.json"),
                "--condition-assessment",
                str(tmp_path / "assessment.json"),
                "--output",
                str(tmp_path / "receipt.json"),
            ]
        )
    assert opened == []


def test_condition_finalize_cli_never_opens_assessment_for_nongate_source(
    monkeypatch,
    tmp_path,
) -> None:
    opened: list[str] = []
    source = SimpleNamespace(
        collection_decision=SimpleNamespace(
            outcome="no_feasible_action",
            condition_gate_invocation_proof=None,
        ),
        policy_visible_question_trajectory=object(),
    )
    roster = SimpleNamespace(
        source_roster_sha256="0" * 64,
        source_membership_sha256="1" * 64,
    )
    monkeypatch.setattr(
        cli_module,
        "_condition_collection_source_roster",
        lambda _path: roster,
    )
    monkeypatch.setattr(
        cli_module,
        "_condition_collection_source",
        lambda _path: source,
    )
    monkeypatch.setattr(
        cli_module,
        "match_validated_condition_calibration_collection_source_membership_v1",
        lambda **_kwargs: None,
    )

    def forbidden_assessment(_path):
        opened.append("assessment")
        raise AssertionError("nongate source cannot open held-out assessment")

    monkeypatch.setattr(
        cli_module,
        "_condition_confirmation_assessment",
        forbidden_assessment,
    )
    with pytest.raises(ValueError, match="source_not_gate_ready"):
        cli_main(
            [
                "condition-finalize-calibration",
                "--source-roster",
                str(tmp_path / "source-roster.json"),
                "--expected-source-roster-sha256",
                "0" * 64,
                "--expected-source-membership-sha256",
                "1" * 64,
                "--source",
                str(tmp_path / "source.json"),
                "--condition-assessment",
                str(tmp_path / "assessment.json"),
                "--output",
                str(tmp_path / "receipt.json"),
            ]
        )
    assert opened == []


def test_item_risk_probabilities_require_artifact_backed_bounds() -> None:
    manifest, corpus = build_offline_fixture()
    repository_root = Path(__file__).resolve().parents[1]
    fingerprint = compute_verifier_pipeline_fingerprint(root=repository_root)
    bundle, candidates, scoring_receipt = _artifact_backed_item_risk_contract(
        manifest=manifest,
        corpus=corpus,
        fingerprint=fingerprint,
        repository_root=repository_root,
    )
    _, _, stale_scoring_receipt = _artifact_backed_item_risk_contract(
        manifest=manifest,
        corpus=corpus,
        fingerprint=fingerprint,
        repository_root=repository_root,
        bind_source_snapshot=False,
    )
    with pytest.raises(
        VerificationContractError,
        match="item_risk_scoring_receipt_source_snapshot_mismatch",
    ):
        run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=30,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=repository_root,
            item_risk_scoring_receipt=stale_scoring_receipt,
            generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
        )

    with pytest.raises(
        VerificationContractError,
        match="detached_item_risk_inputs_forbidden_use_scoring_receipt_v2",
    ):
        run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=30,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=repository_root,
            item_risk_calibration_bundle=bundle,
            item_risk_candidates=candidates,
            generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
        )

    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=repository_root,
        item_risk_scoring_receipt=scoring_receipt,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )

    assert {bound.status for bound in certificate.item_risk_bounds} == {
        "cell_rate_ucl_available"
    }
    assert all(bound.usable_for_release is False for bound in certificate.item_risk_bounds)
    assert certificate.item_risk_scoring_receipt == scoring_receipt
    assert "item_risk_bounds" not in certificate.model_dump(mode="json")
    assert {row.probability_basis for row in certificate.release_assessment.audit.ranking} == {
        "calibrated_cell_rate_ucl"
    }


def test_inconclusive_full_release_state_selects_even_when_audit_guard_is_eligible() -> None:
    manifest, corpus = build_offline_fixture()
    manifest = manifest.model_copy(
        update={
            "audit_guard": manifest.audit_guard.model_copy(
                update={
                    "block_counterfactual_conclusion_flips": False,
                    "max_unresolved_item_cell_ucl_sum": 1.0,
                    "max_unresolved_expected_claim_loss": 10.0,
                    "max_unresolved_item_influence": 1.0,
                }
            )
        }
    )
    repository_root = Path(__file__).resolve().parents[1]
    fingerprint = compute_verifier_pipeline_fingerprint(root=repository_root)
    _, _, scoring_receipt = _artifact_backed_item_risk_contract(
        manifest=manifest,
        corpus=corpus,
        fingerprint=fingerprint,
        repository_root=repository_root,
    )

    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=repository_root,
        item_risk_scoring_receipt=scoring_receipt,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )

    # The residual-risk audit gate itself permits downstream assessment, but the
    # synthesis is inconclusive and the fixture has a corpus blocker. Production
    # must therefore continue with the next feasible action instead of treating the
    # audit sub-gate as if it were the complete release decision.
    assert certificate.release_assessment.evidence.classification == "inconclusive"
    assert certificate.sequential_audit_state is not None
    assert certificate.sequential_audit_state.session.active_action is not None
    assert "active_audit_action_unresolved" in certificate.release_assessment.audit.reasons
    decision = certificate.production_stop_decision
    assert decision.outcome == "selected_next_action"
    assert decision.full_release_eligible is False
    assert decision.selection_result is not None
    assert decision.release_assessment.audit.status == "eligible"
    assert "active_audit_action_unresolved" not in decision.release_assessment.audit.reasons
    assert decision.selection_result.state == certificate.sequential_audit_state


def test_full_release_eligible_initial_state_stops_before_any_audit_action() -> None:
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
    repository_root = Path(__file__).resolve().parents[1]
    fingerprint = compute_verifier_pipeline_fingerprint(root=repository_root)
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
    (
        _,
        _,
        item_risk_scoring_receipt,
    ) = _artifact_backed_item_risk_contract(
        manifest=manifest,
        corpus=corpus,
        fingerprint=fingerprint,
        repository_root=repository_root,
    )
    fixed_state_bundle = _release_calibration_bundle(
        manifest=manifest,
        pipeline_sha256=fingerprint.pipeline_sha256,
    )

    fixed_state_attempt = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        frozen_calibration_bundle=fixed_state_bundle,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=repository_root,
        item_risk_scoring_receipt=item_risk_scoring_receipt,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )
    assert fixed_state_attempt.status == "abstained"
    assert fixed_state_attempt.release_assessment.calibration.reason == (
        "fixed_state_calibration_invalid_for_adaptive_trajectory"
    )
    evaluated_state = fixed_state_attempt.production_stop_decision.evaluated_state
    assert evaluated_state is not None
    adaptive_bundle, adaptive_candidate = _adaptive_release_contract(
        manifest=manifest,
        preselection_certificate=fixed_state_attempt,
        pipeline_sha256=fingerprint.pipeline_sha256,
        budget_minutes=30,
    )

    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        adaptive_calibration_bundle=adaptive_bundle,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=repository_root,
        item_risk_scoring_receipt=item_risk_scoring_receipt,
        generated_at=datetime(2026, 8, 27, 12, 1, tzinfo=UTC),
    )

    assert certificate.status == "released"
    assert certificate.release_assessment.calibration.calibration_contract == (
        "adaptive-first-release-trajectory-v1"
    )
    assert certificate.release_assessment.status.value == "released"
    assert certificate.sequential_audit_state is not None
    assert certificate.sequential_audit_state.session.active_action is None
    assert certificate.sequential_audit_state.session.selected_item_ids == ()
    assert certificate.release_assessment.audit.spent == 0
    assert certificate.complete_corpus_identity == adaptive_candidate.corpus
    assert certificate.item_risk_scoring_receipt == item_risk_scoring_receipt
    assert certificate.adaptive_calibration_bundle == adaptive_bundle
    assert certificate.adaptive_release_candidate is not None
    assert (
        certificate.adaptive_release_candidate.policy_context_sha256
        == adaptive_candidate.policy_context_sha256
    )
    assert (
        certificate.adaptive_release_candidate.observed_states[0].score_features
        == adaptive_candidate.observed_states[0].score_features
    )
    assert certificate.adaptive_policy_context is not None
    assert certificate.adaptive_prospective_assessment is not None
    assert (
        certificate.adaptive_prospective_assessment.assessment_sha256
        == certificate.release_assessment.calibration.prospective_assessment_sha256
    )
    assert certificate.adaptive_prospective_assessment.status == "released"
    decision = certificate.production_stop_decision
    assert decision.outcome == "stopped_released"
    assert decision.full_release_eligible is True
    assert decision.selection_result is None
    assert decision.release_assessment == certificate.release_assessment

    missing_adaptive_artifact = certificate.model_dump(mode="json")
    missing_adaptive_artifact["adaptive_prospective_assessment"] = None
    unsigned = {
        key: value
        for key, value in missing_adaptive_artifact.items()
        if key != "certificate_sha256"
    }
    with pytest.raises(ValidationError, match="adaptive_lineage_incomplete"):
        VerificationCertificate.model_validate(
            {**unsigned, "certificate_sha256": hash_canonical(unsigned)}
        )


def test_materially_corrected_state_releases_only_after_normal_gates() -> None:
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
    source_payload = fixture.graph.model_dump(mode="json")
    outlier_id = source_payload["outcome_estimates"][0]["estimate_id"]
    source_payload["outcome_estimates"][0]["effect"]["estimate"] = -1.5
    source_graph = EvidenceGraph.model_validate(source_payload)
    repository_root = Path(__file__).resolve().parents[1]
    fingerprint = compute_verifier_pipeline_fingerprint(root=repository_root)
    replay_sha256 = "e" * 64
    source_fixture = replace(fixture, graph=source_graph)
    eligibility, metadata, extraction_context = _source_replayed_fixture_contract(
        manifest=manifest,
        fixture=source_fixture,
        pipeline_sha256=fingerprint.pipeline_sha256,
        replay_sha256=replay_sha256,
    )
    corpus = type(fixture)(
        corpus_id=manifest.question_id,
        source_label="/frozen/native/package.json",
        source_format="typed_evidence_grounding_package_json",
        source_sha256=hash_canonical(
            {"fixture": "corrected-release-source", "graph": source_graph}
        ),
        graph=source_graph,
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
    item_bundle, item_candidates, item_scoring_receipt = (
        _artifact_backed_item_risk_contract(
            manifest=manifest,
            corpus=corpus,
            fingerprint=fingerprint,
            repository_root=repository_root,
        )
    )
    fixed_release_bundle = _release_calibration_bundle(
        manifest=manifest,
        pipeline_sha256=fingerprint.pipeline_sha256,
    )
    calibration_shadow = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=repository_root,
        item_risk_scoring_receipt=item_scoring_receipt,
        allow_uncalibrated_sequential_analysis=True,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )
    calibration_start_state = (
        calibration_shadow.production_stop_decision.evaluated_state
    )
    assert calibration_start_state is not None
    assert calibration_start_state.session.active_action is None
    adaptive_bundle, _ = _adaptive_release_contract(
        manifest=manifest,
        preselection_certificate=calibration_shadow,
        pipeline_sha256=fingerprint.pipeline_sha256,
        budget_minutes=30,
    )
    initial = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        adaptive_calibration_bundle=adaptive_bundle,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=repository_root,
        item_risk_scoring_receipt=item_scoring_receipt,
        generated_at=datetime(2026, 8, 27, 12, 0, 30, tzinfo=UTC),
    )
    assert initial.status == "abstained"
    assert initial.release_assessment.evidence.classification == "inconclusive"
    state = initial.sequential_audit_state
    assert state is not None
    action = state.session.active_action
    assert action is not None
    assert action.item_id == outlier_id

    corrected_payload = state.graph.model_dump(mode="json")
    selected = next(
        row
        for row in corrected_payload["outcome_estimates"]
        if row["estimate_id"] == action.item_id
    )
    selected["effect"]["estimate"] = 0.5
    corrected_graph = EvidenceGraph.model_validate(corrected_payload)
    verification = verify_pipeline_fingerprint(
        expected=fingerprint,
        root=repository_root,
    )
    prepared = prepare_verification_scientific_state(
        manifest=manifest,
        graph=corrected_graph,
        pipeline_verification=verification,
        item_risk_calibration_bundle=item_bundle,
        item_risk_candidates=item_candidates,
        resolved_item_ids_for_risk_projection={action.item_id},
    )
    refreshed_candidates = sequential_candidates_from_prepared_state(
        manifest=manifest,
        prepared=prepared,
    )
    expectation = freeze_state_expectation(state)
    adjudication = freeze_selected_adjudication(
        state,
        expected=expectation,
        provenance="benchmark_adjudication",
        adjudicator_count=1,
        protocol_sha256="1" * 64,
        payload_sha256="2" * 64,
        completed_at=datetime(2026, 8, 27, 12, 1, tzinfo=UTC),
        realized_cost=4.0,
    )
    resolution = resolve_selected_audit_candidate(
        state,
        expected=expectation,
        adjudication=adjudication,
        disposition=CorrectionDisposition.CORRECTED,
        corrected_graph=corrected_graph,
        correction_provenance="benchmark_adjudication",
        correction_protocol_sha256="3" * 64,
        external_correction_payload_sha256="4" * 64,
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

    with pytest.raises(
        VerificationContractError,
        match="adaptive_calibration_bundle_required_for_existing_history",
    ):
        run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=30,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=repository_root,
            item_risk_scoring_receipt=item_scoring_receipt,
            sequential_audit_state=resolution.state,
            generated_at=datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
        )
    with pytest.raises(
        VerificationContractError,
        match="adaptive_calibration_bundle_required_for_existing_history",
    ):
        run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=30,
            frozen_calibration_bundle=fixed_release_bundle,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=repository_root,
            item_risk_scoring_receipt=item_scoring_receipt,
            sequential_audit_state=resolution.state,
            generated_at=datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
        )
    released = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=30,
        adaptive_calibration_bundle=adaptive_bundle,
        expected_pipeline_fingerprint=fingerprint,
        pipeline_root=repository_root,
        item_risk_scoring_receipt=item_scoring_receipt,
        sequential_audit_state=resolution.state,
        generated_at=datetime(2026, 8, 27, 12, 3, tzinfo=UTC),
    )
    assert released.status == "released"
    assert released.reasons == []
    assert released.release_assessment.evidence.classification == "supported"
    assert released.release_assessment.audit.status == "eligible"
    assert released.release_assessment.calibration.status == "released"
    assert released.source_evidence_graph == source_graph
    assert released.evidence_graph == corrected_graph
    assert released.production_stop_decision.outcome == "stopped_released"


def test_complete_v5_certificate_sequence_projects_real_adaptive_trajectory() -> None:
    manifest, corpus = build_offline_fixture()
    repository_root = Path(__file__).resolve().parents[1]
    fingerprint = compute_verifier_pipeline_fingerprint(root=repository_root)
    budget_minutes = 30.0
    certificates = [
        run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=budget_minutes,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=repository_root,
            allow_uncalibrated_sequential_analysis=True,
            generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
        )
    ]
    policy_context = build_verifier_adaptive_policy_context(
        manifest=manifest,
        pipeline_sha256=fingerprint.pipeline_sha256,
        budget_minutes=budget_minutes,
        policy_arm_id="production-adaptive-policy",
    )

    with pytest.raises(
        AdaptiveCalibrationError,
        match="adaptive_projection_terminal_reason_not_proven",
    ):
        policy_visible_trajectory_from_certificate_v5_sequence(
            certificates,
            split="development",
            policy_context=policy_context,
            terminal_reason="all_items_resolved",
        )

    for index in range(1, 5):
        previous = certificates[-1]
        state = previous.sequential_audit_state
        assert state is not None
        action = state.session.active_action
        if action is None:
            break
        verification = verify_pipeline_fingerprint(
            expected=fingerprint,
            root=repository_root,
        )
        prepared = prepare_verification_scientific_state(
            manifest=manifest,
            graph=state.graph,
            pipeline_verification=verification,
        )
        refreshed_candidates = sequential_candidates_from_prepared_state(
            manifest=manifest,
            prepared=prepared,
        )
        expectation = freeze_state_expectation(state)
        adjudication = freeze_selected_adjudication(
            state,
            expected=expectation,
            provenance="benchmark_adjudication",
            adjudicator_count=1,
            protocol_sha256="a" * 64,
            payload_sha256=f"{index + 10:x}".rjust(64, "0"),
            completed_at=datetime(2026, 8, 27, 12, index * 2 - 1, tzinfo=UTC),
            realized_cost=1.0,
        )
        resolution = resolve_selected_audit_candidate(
            state,
            expected=expectation,
            adjudication=adjudication,
            disposition=CorrectionDisposition.NO_CHANGE,
            corrected_graph=None,
            correction_provenance="benchmark_adjudication",
            correction_protocol_sha256="c" * 64,
            external_correction_payload_sha256=f"{index + 20:x}".rjust(64, "0"),
            synthesis_runner_sha256=compute_synthesis_runner_sha256(
                manifest=manifest,
                pipeline_sha256=fingerprint.pipeline_sha256,
            ),
            candidate_runner_sha256=compute_candidate_runner_sha256(
                manifest=manifest,
                pipeline_sha256=fingerprint.pipeline_sha256,
            ),
            rerun_synthesis=lambda _graph, result=prepared.synthesis: result,
            rerun_candidates=(
                lambda _graph, _synthesis, _session, result=refreshed_candidates: result
            ),
        )
        certificates.append(
            run_verification(
                manifest=manifest,
                corpus=corpus,
                budget_minutes=budget_minutes,
                expected_pipeline_fingerprint=fingerprint,
                pipeline_root=repository_root,
                sequential_audit_state=resolution.state,
                allow_uncalibrated_sequential_analysis=True,
                generated_at=datetime(2026, 8, 27, 12, index * 2, tzinfo=UTC),
            )
        )

    final_state = certificates[-1].production_stop_decision.evaluated_state
    assert final_state is not None
    assert final_state.session.active_action is None
    assert set(final_state.session.resolved_item_ids) == {
        candidate.item_id for candidate in final_state.candidates
    }
    trajectory = policy_visible_trajectory_from_certificate_v5_sequence(
        certificates,
        split="development",
        policy_context=policy_context,
        terminal_reason="all_items_resolved",
    )
    arm = trajectory.arms[0]
    assert arm.completeness_basis == "validated_v5_certificate_sequence"
    assert arm.source_certificate_sha256s == [
        certificate.certificate_sha256 for certificate in certificates
    ]
    assert arm.terminal_decision_sha256 == (
        certificates[-1].production_stop_decision.decision_sha256
    )
    assert arm.terminal_proof.resolved_item_ids == list(
        final_state.session.resolved_item_ids
    )


def test_cli_sequential_audit_charges_realized_minutes_and_resumes(tmp_path, capsys) -> None:
    manifest, _ = build_offline_fixture()
    claim_path = tmp_path / "claim.json"
    claim_path.write_text(manifest.model_dump_json(indent=2))
    initial_dir = tmp_path / "initial"
    assert (
        cli_main(
            [
                "verify",
                "--fixture",
                "--budget-minutes",
                "30",
                "--analysis-only-uncalibrated-audit",
                "--output-dir",
                str(initial_dir),
            ]
        )
        == 0
    )
    initial_summary = json.loads(capsys.readouterr().out)
    assert initial_summary["selected_audit_item_id"] is not None
    initial_certificate = VerificationCertificate.model_validate_json(
        (initial_dir / "verification-certificate.json").read_text()
    )
    assert initial_certificate.sequential_audit_state is not None
    assert initial_certificate.sequential_audit_state.session.active_action is not None
    assert "active_audit_action_unresolved" in (
        initial_certificate.release_assessment.audit.reasons
    )
    assert initial_certificate.status == "abstained"

    evidence_files = {}
    for name in (
        "adjudication-protocol",
        "adjudication-payload",
        "correction-protocol",
        "correction-payload",
    ):
        path = tmp_path / f"{name}.txt"
        path.write_text(f"immutable {name}\n")
        evidence_files[name] = path
    resolved_dir = tmp_path / "resolved"
    assert (
        cli_main(
            [
                "audit-resolve",
                "--state",
                str(initial_dir / "sequential-audit-state.json"),
                "--claim",
                str(claim_path),
                "--disposition",
                "no_change",
                "--adjudication-protocol",
                str(evidence_files["adjudication-protocol"]),
                "--adjudication-payload",
                str(evidence_files["adjudication-payload"]),
                "--correction-protocol",
                str(evidence_files["correction-protocol"]),
                "--correction-payload",
                str(evidence_files["correction-payload"]),
                "--provenance",
                "benchmark_adjudication",
                "--adjudicator-count",
                "1",
                "--realized-minutes",
                "4.5",
                "--output-dir",
                str(resolved_dir),
            ]
        )
        == 0
    )
    resolved_summary = json.loads(capsys.readouterr().out)
    assert resolved_summary["cumulative_realized_minutes"] == 4.5

    terminal_dir = tmp_path / "terminal"
    assert (
        cli_main(
            [
                "verify",
                "--fixture",
                "--budget-minutes",
                "30",
                "--audit-state",
                str(resolved_dir / "sequential-audit-state.json"),
                "--analysis-only-uncalibrated-audit",
                "--output-dir",
                str(terminal_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    terminal = VerificationCertificate.model_validate_json(
        (terminal_dir / "verification-certificate.json").read_text()
    )
    assert terminal.release_assessment.audit.cost_basis == "realized_session"
    assert terminal.release_assessment.audit.spent == 4.5
    assert len(terminal.release_assessment.audit.resolution_receipts_v2) == 1
    assert terminal.sequential_audit_state is not None
    resumed_action = terminal.sequential_audit_state.session.active_action
    assert resumed_action is not None
    assert resumed_action.item_id != initial_summary["selected_audit_item_id"]
    assert len(terminal.sequential_audit_state.session.selected_item_ids) == 2
    assert "active_audit_action_unresolved" in terminal.release_assessment.audit.reasons
    resumed_decision = terminal.production_stop_decision
    assert resumed_decision.outcome == "selected_next_action"
    assert resumed_decision.evaluated_state is not None
    assert resumed_decision.evaluated_state.session.active_action is None
    assert len(resumed_decision.evaluated_state.session.resolved_item_ids) == 1
    assert resumed_decision.selection_result is not None
    assert resumed_decision.selection_result.action == resumed_action

    selected_dir = tmp_path / "selected-again"
    assert (
        cli_main(
            [
                "audit-select",
                "--state",
                str(resolved_dir / "sequential-audit-state.json"),
                "--analysis-only-uncalibrated-audit",
                "--output-dir",
                str(selected_dir),
            ]
        )
        == 0
    )
    next_summary = json.loads(capsys.readouterr().out)
    assert next_summary["item_id"] != initial_summary["selected_audit_item_id"]
    assert next_summary["remaining_budget"] == 25.5


@pytest.mark.parametrize("remove_selected", [False, True])
def test_cli_material_correction_reenters_verifier_with_source_current_lineage(
    tmp_path,
    capsys,
    remove_selected: bool,
) -> None:
    manifest, source_corpus = build_offline_fixture()
    repository_root = Path(__file__).resolve().parents[1]
    fingerprint = compute_verifier_pipeline_fingerprint(root=repository_root)
    _, _, item_risk_receipt = _artifact_backed_item_risk_contract(
        manifest=manifest,
        corpus=source_corpus,
        fingerprint=fingerprint,
        repository_root=repository_root,
    )
    item_risk_receipt_path = tmp_path / "item-risk-scoring-receipt.json"
    item_risk_receipt_path.write_text(item_risk_receipt.model_dump_json(indent=2))
    claim_path = tmp_path / "claim.json"
    claim_path.write_text(manifest.model_dump_json(indent=2))
    initial_dir = tmp_path / "initial"
    assert cli_main(
        [
            "verify",
            "--fixture",
            "--budget-minutes",
            "30",
            "--analysis-only-uncalibrated-audit",
            "--item-risk-scoring-receipt",
            str(item_risk_receipt_path),
            "--output-dir",
            str(initial_dir),
        ]
    ) == 0
    capsys.readouterr()
    initial_certificate = VerificationCertificate.model_validate_json(
        (initial_dir / "verification-certificate.json").read_text()
    )
    initial_state = initial_certificate.sequential_audit_state
    assert initial_state is not None
    action = initial_state.session.active_action
    assert action is not None

    corrected_graph = initial_state.graph.model_dump(mode="json")
    selected = next(
        row
        for row in corrected_graph["outcome_estimates"]
        if row["estimate_id"] == action.item_id
    )
    if remove_selected:
        corrected_graph["outcome_estimates"] = [
            row
            for row in corrected_graph["outcome_estimates"]
            if row["estimate_id"] != action.item_id
        ]
    else:
        original_estimate = selected["effect"]["estimate"]
        selected["effect"]["estimate"] = float(original_estimate) + 0.75
    corrected_path = tmp_path / "corrected-evidence-graph.json"
    corrected_path.write_text(json.dumps(corrected_graph, sort_keys=True))
    evidence_files: dict[str, Path] = {}
    for name in (
        "adjudication-protocol",
        "adjudication-payload",
        "correction-protocol",
        "correction-payload",
    ):
        path = tmp_path / f"{name}.txt"
        path.write_text(f"immutable material {name}\n")
        evidence_files[name] = path

    resolved_dir = tmp_path / "resolved"
    assert cli_main(
        [
            "audit-resolve",
            "--state",
            str(initial_dir / "sequential-audit-state.json"),
            "--claim",
            str(claim_path),
            "--disposition",
            "corrected",
            "--corrected-corpus",
            str(corrected_path),
            "--adjudication-protocol",
            str(evidence_files["adjudication-protocol"]),
            "--adjudication-payload",
            str(evidence_files["adjudication-payload"]),
            "--correction-protocol",
            str(evidence_files["correction-protocol"]),
            "--correction-payload",
            str(evidence_files["correction-payload"]),
            "--provenance",
            "benchmark_adjudication",
            "--adjudicator-count",
            "1",
            "--realized-minutes",
            "4.5",
            "--item-risk-scoring-receipt",
            str(item_risk_receipt_path),
            "--output-dir",
            str(resolved_dir),
        ]
    ) == 0
    capsys.readouterr()

    terminal_dir = tmp_path / "terminal"
    assert cli_main(
        [
            "verify",
            "--fixture",
            "--budget-minutes",
            "30",
            "--audit-state",
            str(resolved_dir / "sequential-audit-state.json"),
            "--analysis-only-uncalibrated-audit",
            "--item-risk-scoring-receipt",
            str(item_risk_receipt_path),
            "--output-dir",
            str(terminal_dir),
        ]
    ) == 0
    capsys.readouterr()
    terminal = VerificationCertificate.model_validate_json(
        (terminal_dir / "verification-certificate.json").read_text()
    )
    assert terminal.certificate_version == "literature-multiverse-verification-v5"
    assert terminal.source_evidence_graph == source_corpus.graph
    assert terminal.source_evidence_graph_sha256 == hash_canonical(source_corpus.graph)
    assert terminal.evidence_graph_sha256 != terminal.source_evidence_graph_sha256
    assert terminal.synthesis_sha256 != initial_certificate.synthesis_sha256
    assert terminal.release_assessment.evidence_graph_sha256 == (
        terminal.evidence_graph_sha256
    )
    assert terminal.release_assessment.synthesis_sha256 == terminal.synthesis_sha256
    assert terminal.sequential_audit_state is not None
    correction_transitions = [
        transition
        for transition in terminal.sequential_audit_state.transitions
        if transition.transition_kind == "correction"
    ]
    assert len(correction_transitions) == 1
    correction = correction_transitions[0]
    assert correction.receipt is not None
    assert correction.receipt.realized_cost == 4.5
    assert correction.post_graph_sha256 == terminal.evidence_graph_sha256
    assert terminal.sequential_audit_state.session.historical_realized_cost == 4.5
    assert terminal.current_state_hashes["source_evidence_graph"] == (
        terminal.source_evidence_graph_sha256
    )
    assert terminal.current_state_hashes["current_evidence_graph"] == (
        terminal.evidence_graph_sha256
    )
    assert (
        VerificationCertificate.model_validate_json(terminal.model_dump_json())
        == terminal
    )


def test_item_risk_projection_rejects_changed_unresolved_estimate() -> None:
    manifest, corpus = build_offline_fixture()
    repository_root = Path(__file__).resolve().parents[1]
    fingerprint = compute_verifier_pipeline_fingerprint(root=repository_root)
    verification = verify_pipeline_fingerprint(
        expected=fingerprint,
        root=repository_root,
    )
    bundle, candidates, _ = _artifact_backed_item_risk_contract(
        manifest=manifest,
        corpus=corpus,
        fingerprint=fingerprint,
        repository_root=repository_root,
    )
    selected_id, unresolved_id = sorted(
        estimate.estimate_id for estimate in corpus.graph.outcome_estimates
    )[:2]
    payload = corpus.graph.model_dump(mode="json")
    unresolved = next(
        estimate
        for estimate in payload["outcome_estimates"]
        if estimate["estimate_id"] == unresolved_id
    )
    unresolved["effect"]["estimate"] = float(
        unresolved["effect"]["estimate"]
    ) + 0.25
    changed = EvidenceGraph.model_validate(payload)

    with pytest.raises(
        VerificationContractError,
        match=f"item_risk_projection_changed_unresolved_items:{unresolved_id}",
    ):
        prepare_verification_scientific_state(
            manifest=manifest,
            graph=changed,
            pipeline_verification=verification,
            item_risk_calibration_bundle=bundle,
            item_risk_candidates=candidates,
            resolved_item_ids_for_risk_projection={selected_id},
        )


@pytest.mark.parametrize(
    ("forged_synthesis", "forged_candidates", "expected_error"),
    [
        (
            True,
            False,
            "sequential_audit_correction_1_synthesis_recomputation_mismatch",
        ),
        (
            False,
            True,
            "sequential_audit_correction_1_candidate_recomputation_mismatch",
        ),
    ],
)
def test_production_recomputes_corrected_synthesis_and_candidates(
    forged_synthesis: bool,
    forged_candidates: bool,
    expected_error: str,
) -> None:
    manifest, corpus, fingerprint, _, state = _materially_corrected_fixture_state(
        forged_synthesis=forged_synthesis,
        forged_candidates=forged_candidates,
    )

    with pytest.raises(VerificationContractError, match=expected_error):
        run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=30,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=Path(__file__).resolve().parents[1],
            sequential_audit_state=state,
            generated_at=datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
        )


def test_corrected_state_rejects_stale_source_budget_and_cost_unit() -> None:
    manifest, corpus, fingerprint, initial, state = _materially_corrected_fixture_state()
    repository_root = Path(__file__).resolve().parents[1]
    stale_source_payload = corpus.graph.model_dump(mode="json")
    stale_source_payload["outcome_estimates"][0]["effect"]["estimate"] += 0.01
    stale_source = replace(
        corpus,
        graph=EvidenceGraph.model_validate(stale_source_payload),
    )
    with pytest.raises(
        VerificationContractError,
        match="sequential_audit_source_evidence_graph_mismatch",
    ):
        run_verification(
            manifest=manifest,
            corpus=stale_source,
            budget_minutes=30,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=repository_root,
            sequential_audit_state=state,
            generated_at=datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
        )

    with pytest.raises(
        VerificationContractError,
        match="sequential_audit_state_budget_mismatch",
    ):
        run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=29,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=repository_root,
            sequential_audit_state=state,
            generated_at=datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
        )

    preselection = initial.production_stop_decision.evaluated_state
    assert preselection is not None
    wrong_cost_candidates = [
        freeze_current_audit_candidate(
            item_id=candidate.item_id,
            priority=candidate.priority,
            estimated_cost=candidate.estimated_cost,
            cost_unit="reviewer_minutes",
            scientific_candidate_sha256=candidate.scientific_candidate_sha256,
            counterfactual_synthesis_sha256=(
                candidate.counterfactual_synthesis_sha256
            ),
            eligible=candidate.eligible,
            ineligibility_reasons=candidate.ineligibility_reasons,
            risk_bound_sha256=candidate.risk_bound_sha256,
        )
        for candidate in preselection.candidates
    ]
    wrong_cost_unit_state = create_sequential_verification_state(
        session_id="wrong-cost-unit-session",
        created_at=preselection.session.created_at,
        pipeline_sha256=preselection.session.pipeline_sha256,
        policy_sha256=preselection.session.policy_sha256,
        budget=30,
        cost_unit="reviewer_minutes",
        graph=preselection.graph,
        synthesis=preselection.synthesis,
        candidates=wrong_cost_candidates,
    )
    with pytest.raises(
        VerificationContractError,
        match="sequential_audit_state_cost_unit_mismatch",
    ):
        run_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=30,
            expected_pipeline_fingerprint=fingerprint,
            pipeline_root=repository_root,
            sequential_audit_state=wrong_cost_unit_state,
            generated_at=datetime(2026, 8, 27, 12, 2, tzinfo=UTC),
        )


def test_transition_ledger_rejects_reorder_receipt_predecessor_and_drop() -> None:
    _, _, _, _, corrected_state = _materially_corrected_fixture_state()

    reordered = corrected_state.model_dump(mode="json", exclude={"state_sha256"})
    reordered["transitions"] = list(reversed(reordered["transitions"]))
    with pytest.raises(
        ValidationError,
        match="sequential_transition_predecessor_state_mismatch",
    ):
        SequentialVerificationState.model_validate(
            {**reordered, "state_sha256": hash_canonical(reordered)}
        )

    bad_predecessor = corrected_state.model_dump(
        mode="json", exclude={"state_sha256"}
    )
    correction = bad_predecessor["transitions"][-1]
    correction["previous_state_sha256"] = "f" * 64
    correction["transition_sha256"] = hash_canonical(
        {
            key: value
            for key, value in correction.items()
            if key != "transition_sha256"
        }
    )
    with pytest.raises(
        ValidationError,
        match="sequential_transition_predecessor_state_mismatch",
    ):
        SequentialVerificationState.model_validate(
            {**bad_predecessor, "state_sha256": hash_canonical(bad_predecessor)}
        )

    bad_receipt = corrected_state.model_dump(mode="json", exclude={"state_sha256"})
    receipt = bad_receipt["transitions"][-1]["receipt"]
    receipt["realized_cost"] += 0.25
    with pytest.raises(ValidationError, match="receipt_cumulative_realized_cost_mismatch"):
        SequentialVerificationState.model_validate(
            {**bad_receipt, "state_sha256": hash_canonical(bad_receipt)}
        )

    _, _, _, _, no_change_state = _materially_corrected_fixture_state(no_change=True)
    dropped = no_change_state.model_dump(mode="json", exclude={"state_sha256"})
    dropped["transitions"].pop()
    with pytest.raises(
        ValidationError,
        match="sequential_transition_final_session_mismatch",
    ):
        SequentialVerificationState.model_validate(
            {**dropped, "state_sha256": hash_canonical(dropped)}
        )


def test_cli_active_cost_checkpoint_exhausts_budget_and_resume_abstains(
    tmp_path,
    capsys,
) -> None:
    initial_dir = tmp_path / "initial"
    assert (
        cli_main(
            [
                "verify",
                "--fixture",
                "--budget-minutes",
                "30",
                "--analysis-only-uncalibrated-audit",
                "--output-dir",
                str(initial_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    initial_state = json.loads(
        (initial_dir / "sequential-audit-state.json").read_text()
    )

    with pytest.raises(ValueError, match="active_realized_cost_exceeds_budget"):
        cli_main(
            [
                "audit-checkpoint",
                "--state",
                str(initial_dir / "sequential-audit-state.json"),
                "--active-realized-minutes",
                "30.1",
                "--output-dir",
                str(tmp_path / "invalid-checkpoint"),
            ]
        )
    assert not (tmp_path / "invalid-checkpoint").exists()

    checkpoint_dir = tmp_path / "checkpoint"
    assert (
        cli_main(
            [
                "audit-checkpoint",
                "--state",
                str(initial_dir / "sequential-audit-state.json"),
                "--active-realized-minutes",
                "30",
                "--output-dir",
                str(checkpoint_dir),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    checkpoint_state = json.loads(
        (checkpoint_dir / "sequential-audit-state.json").read_text()
    )
    checkpoint_receipt = json.loads(
        (checkpoint_dir / "audit-active-cost-checkpoint.json").read_text()
    )

    assert summary["active_realized_minutes"] == 30.0
    assert summary["cumulative_realized_minutes"] == 30.0
    assert summary["historical_realized_minutes"] == 0.0
    assert summary["remaining_budget"] == 0.0
    assert summary["release_blocked_by_active_action"] is True
    assert checkpoint_state["graph_sha256"] == initial_state["graph_sha256"]
    assert checkpoint_state["synthesis_sha256"] == initial_state["synthesis_sha256"]
    assert checkpoint_state["session"]["resolved_item_ids"] == []
    assert checkpoint_state["session"]["active_action"] is not None
    assert checkpoint_receipt["result_sha256"] == hash_canonical(
        {
            key: value
            for key, value in checkpoint_receipt.items()
            if key != "result_sha256"
        }
    )

    resumed_dir = tmp_path / "resumed"
    assert (
        cli_main(
            [
                "verify",
                "--fixture",
                "--budget-minutes",
                "30",
                "--audit-state",
                str(checkpoint_dir / "sequential-audit-state.json"),
                "--output-dir",
                str(resumed_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    certificate = VerificationCertificate.model_validate_json(
        (resumed_dir / "verification-certificate.json").read_text()
    )
    assert certificate.status == "abstained"
    assert certificate.release_assessment.audit.spent == 30.0
    assert certificate.release_assessment.audit.resolved_item_ids == []
    assert "active_audit_action_unresolved" in (
        certificate.release_assessment.audit.reasons
    )


def _legacy_finding() -> FindingRow:
    finding_id = make_finding_id(
        paper_id="doc:legacy-1",
        map_result_id="map-legacy-1",
        array_position=0,
        outcome_name="performance",
        timepoint_raw="4 weeks",
        dose_raw=None,
        effect_direction="increase",
    )
    return FindingRow(
        finding_id=finding_id,
        paper_id="doc:legacy-1",
        doc_id="legacy-1",
        map_result_id="map-legacy-1",
        array_position=0,
        prompt_version="legacy-test-v1",
        schema_version="1",
        cfghash="a" * 64,
        grounding_status="exact",
        evidence_section="Results",
        section_flagged=False,
        normalization_warnings=[],
        study_type="randomized trial",
        species="human",
        model=None,
        population_state="healthy",
        intervention="intervention",
        intervention_class=None,
        comparator="control",
        dose_raw=None,
        duration_raw="4 weeks",
        timing_context="post intervention",
        outcome_name="performance",
        outcome_family="performance",
        timepoint_raw="4 weeks",
        effect_direction="increase",
        effect_size_raw=None,
        p_value=None,
        significant=True,
        sample_size=20,
        evidence_quote="Performance increased after four weeks.",
        evidence_lines=["L1"],
        confidence=0.8,
        moderators={},
    )


def test_legacy_findings_are_connected_but_cannot_silently_release() -> None:
    graph, issues = adapt_legacy_findings([_legacy_finding()], settings=LegacyAdapterConfig())
    manifest = ClaimManifest(
        question_id="legacy-verification-test",
        population_id="legacy-test-population",
        domain="synthetic",
        claim=ScientificClaim(
            statement="The intervention increases performance.",
            direction="increase",
            outcome_name="performance",
        ),
        protocol=VerificationProtocol(corpus_cutoff="legacy-fixture-v1"),
    )
    _, fixture_corpus = build_offline_fixture()
    corpus = type(fixture_corpus)(
        corpus_id="legacy-fixture",
        source_label="embedded:legacy-fixture",
        source_format="legacy_findings_test",
        source_sha256="b" * 64,
        graph=graph,
        eligibility=(),
        adapter_issues=issues,
        metadata={},
        provenance_assurance=fixture_corpus.provenance_assurance.model_copy(
            update={
                "status": "unverified_source_provenance",
                "reason": (
                    "Legacy categorical findings were not replayed against native sources."
                ),
            }
        ),
    )

    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=10,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )

    assert certificate.status == "abstained"
    assert certificate.release_assessment.evidence.classification == "not_evaluable"
    assert "adapter:legacy_effect_not_quantitatively_interpretable" in certificate.reasons
    assert "adapter:unresolved_cohort_identity" in certificate.reasons


def test_typed_graph_json_is_a_supported_corpus_input(tmp_path) -> None:
    manifest, fixture = build_offline_fixture()
    graph_path = tmp_path / "evidence_graph.json"
    graph_path.write_text(fixture.graph.model_dump_json(indent=2))

    loaded = load_corpus(graph_path, legacy_settings=manifest.legacy_adapter)

    assert loaded.source_format == "evidence_graph_json"
    assert loaded.graph == fixture.graph
    assert all(item.status == "included" for item in loaded.eligibility)
    assert loaded.provenance_assurance.status == "unverified_source_provenance"
    assert loaded.provenance_release_eligible() is False
    assert any(
        issue.code == "unverified_source_provenance"
        and issue.severity.value == "blocking"
        for issue in loaded.adapter_issues
    )

    certificate = run_verification(
        manifest=manifest,
        corpus=loaded,
        budget_minutes=30,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )
    assert certificate.status == "abstained"
    assert "adapter:unverified_source_provenance" in certificate.reasons
    assert certificate.corpus["provenance_assurance"]["release_eligible"] is False


def test_graph_bundle_cannot_downgrade_source_provenance_blocker(tmp_path) -> None:
    manifest, fixture = build_offline_fixture()
    bundle_path = tmp_path / "corpus-bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "corpus_bundle_version": "diagnostic-v1",
                "corpus_id": "unverified-bundle",
                "graph": fixture.graph.model_dump(mode="json"),
                "adapter_issues": [
                    {
                        "severity": "warning",
                        "code": "unverified_source_provenance",
                        "detail": "Untrusted input attempted to downgrade the release blocker.",
                    }
                ],
            }
        )
    )

    loaded = load_corpus(bundle_path, legacy_settings=manifest.legacy_adapter)

    provenance_issues = [
        issue
        for issue in loaded.adapter_issues
        if issue.code == "unverified_source_provenance"
    ]
    assert len(provenance_issues) == 1
    assert provenance_issues[0].severity.value == "blocking"

    forged = type(loaded)(
        corpus_id=loaded.corpus_id,
        source_label=loaded.source_label,
        source_format=loaded.source_format,
        source_sha256=loaded.source_sha256,
        graph=loaded.graph,
        eligibility=loaded.eligibility,
        adapter_issues=(),
        metadata=loaded.metadata,
        provenance_assurance=fixture.provenance_assurance,
    )
    certificate = run_verification(
        manifest=manifest,
        corpus=forged,
        budget_minutes=30,
        generated_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
    )
    assert "adapter:unverified_source_provenance" in certificate.reasons
