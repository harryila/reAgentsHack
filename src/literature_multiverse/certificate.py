"""Hash-bound, self-contained verification certificate artifacts.

The JSON certificate is the normative artifact.  The HTML file is a dependency-free
human-readable rendering of that exact JSON payload; it never fetches remote assets or
executes JavaScript.  A certificate can therefore be archived, inspected, and verified
without the application that created it.
"""

from __future__ import annotations

import html
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.adaptive_calibration import (
    AdaptiveCalibrationBundle,
    AdaptiveCalibrationBundleV2,
    AdaptiveCalibrationError,
    AdaptiveIndependenceIdentityV2,
    AdaptivePolicyContext,
    AdaptivePreselectionState,
    AdaptiveProspectiveAssessment,
    AdaptiveProspectiveAssessmentV2,
    AdaptiveTargetSemanticsBindingV2,
    CompleteCorpusIdentity,
    ConditionCalibrationCollectionSourceAnchorV1,
    ConditionCalibrationCollectionSourceRosterV1,
    ConditionCalibrationGateResultV1,
    ConditionCalibrationProjectionV1,
    ConditionConfirmationGateAssessmentV1,
    ConditionGateInvocationProofV2,
    ConditionTerminalGateResultV2,
    ConfirmationAwareReleaseQualificationProofV2,
    PolicyVisibleQuestionTrajectoryV2,
    ProspectiveAdaptiveReleaseCandidate,
    ProspectiveAdaptiveReleaseCandidateV2,
    assess_adaptive_release_candidate,
    assess_confirmation_aware_adaptive_release_candidate,
    freeze_adaptive_policy_context,
    freeze_condition_calibration_gate_result_v1,
    freeze_condition_confirmation_gate_assessment,
    freeze_condition_terminal_gate_result_v2,
    freeze_confirmation_aware_release_qualification_proof_v2,
    freeze_preselection_state_from_production_components,
    freeze_prospective_adaptive_candidate_v2,
    validate_adaptive_calibration_bundle_integrity,
    validate_adaptive_calibration_bundle_v2_integrity,
)
from literature_multiverse.claim_release import (
    CLAIM_RELEASE_RISK_FEATURE_NAMES,
    ClaimReleaseAssessment,
    ConditionClaimReleaseAssessmentV1,
    QualifiedClaimReleaseAssessment,
)
from literature_multiverse.composed_corpus_identity import (
    CompleteCorpusIdentityV3,
    ManifestCorpusPolicyBindingV3,
    freeze_manifest_corpus_policy_binding_v3,
    validate_complete_corpus_identity_v3,
    validate_manifest_corpus_policy_binding_v3,
)
from literature_multiverse.condition_confirmation import (
    ConditionConfirmationAssessmentV1,
    ConditionConfirmationError,
    ConditionConfirmationFrozenModelV1,
    ConditionConfirmationPlanV1,
    validate_condition_confirmation_assessment,
    validate_condition_confirmation_model,
)
from literature_multiverse.corpus_pipeline_composition_runtime import (
    CorpusPipelineCompositionExternalReplayReceiptV1,
)
from literature_multiverse.evidence_graph import EvidenceGraph
from literature_multiverse.independence_identity import StrongIndependenceIdentityV1
from literature_multiverse.item_risk_artifacts import ItemRiskScoringRunReceipt
from literature_multiverse.item_risk_calibration import RiskBound
from literature_multiverse.lineage import (
    atomic_write_json,
    atomic_write_text,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_extraction import NativeSourceManifest
from literature_multiverse.pipeline_fingerprint import PipelineFingerprintVerification
from literature_multiverse.production_policy import (
    PRODUCTION_STOPPING_RULE,
    should_stop_at_full_frozen_release,
)
from literature_multiverse.sequential_verification import (
    SequentialSelectionResult,
    SequentialVerificationContractError,
    SequentialVerificationState,
    adaptive_preselection_history_from_state,
    resume_sequential_verification_state,
    selection_predecessor_states_from_state,
)


class CertificateLineageStage(ContractModel):
    """One deterministic hand-off in the unified verifier."""

    stage: Annotated[str, Field(min_length=1)]
    input_sha256s: dict[str, str]
    output_sha256s: dict[str, str]
    method: Annotated[str, Field(min_length=1)]

    @field_validator("input_sha256s", "output_sha256s")
    @classmethod
    def validate_hash_mapping(cls, value: dict[str, str]) -> dict[str, str]:
        if value != dict(sorted(value.items())):
            raise ValueError("certificate_lineage_hashes_must_be_sorted")
        if any(not SHA256_RE.fullmatch(digest) for digest in value.values()):
            raise ValueError("certificate_lineage_hash_invalid")
        return value


class ProductionStopDecision(ContractModel):
    """The exact preselection release decision and any resulting transition.

    The final certificate commonly contains an active action and therefore an
    abstaining *postselection* assessment.  Persisting the no-active prefix and its
    complete assessment makes it independently checkable that production selected
    only after failing the same release rule used by retrospective evaluation.
    """

    decision_version: Literal["production-stop-decision-v1"] = (
        "production-stop-decision-v1"
    )
    stopping_rule: Literal["stop_at_first_full_frozen_release_eligible_state"] = (
        "stop_at_first_full_frozen_release_eligible_state"
    )
    evaluated_state: SequentialVerificationState | None
    release_assessment: ClaimReleaseAssessment | QualifiedClaimReleaseAssessment
    blocking_adapter_reasons: list[str]
    full_release_eligible: bool
    outcome: Literal[
        "stopped_released",
        "selected_next_action",
        "active_action_in_progress",
        "no_feasible_action",
        "legacy_receipt_assessment",
        "adaptive_calibration_required_before_audit_genesis",
        "adaptive_calibration_required_before_selection",
    ]
    selection_result: SequentialSelectionResult | None
    decision_sha256: str

    @field_validator("decision_sha256")
    @classmethod
    def validate_decision_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("production_stop_decision_sha256_invalid")
        return value

    @field_validator("blocking_adapter_reasons")
    @classmethod
    def validate_blockers(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(
            not reason.startswith("adapter:") or len(reason) == len("adapter:")
            for reason in value
        ):
            raise ValueError("production_stop_adapter_blockers_invalid")
        return value

    @model_validator(mode="after")
    def validate_decision(self) -> ProductionStopDecision:
        if self.stopping_rule != PRODUCTION_STOPPING_RULE:
            raise ValueError("production_stop_rule_identity_mismatch")
        active_action = (
            self.evaluated_state is not None
            and self.evaluated_state.session.active_action is not None
        )
        expected_release = should_stop_at_full_frozen_release(
            release_status=self.release_assessment.status.value,
            blocking_reasons=self.blocking_adapter_reasons,
            active_action=active_action,
        )
        if self.full_release_eligible != expected_release:
            raise ValueError("production_stop_full_release_eligibility_mismatch")
        assessment_state_sha256 = (
            self.release_assessment.audit.sequential_state_sha256
        )
        if self.evaluated_state is None:
            if assessment_state_sha256 is not None:
                raise ValueError("production_stop_evaluated_state_missing")
            if self.outcome not in {
                "legacy_receipt_assessment",
                "adaptive_calibration_required_before_audit_genesis",
            } or self.selection_result is not None:
                raise ValueError("production_stop_legacy_outcome_mismatch")
        else:
            if assessment_state_sha256 != self.evaluated_state.state_sha256:
                raise ValueError("production_stop_assessment_state_mismatch")
            if self.outcome == "stopped_released":
                if not expected_release or active_action or self.selection_result is not None:
                    raise ValueError("production_stop_released_outcome_mismatch")
            elif self.outcome == "selected_next_action":
                result = self.selection_result
                if expected_release or active_action or result is None:
                    raise ValueError("production_stop_selection_outcome_mismatch")
                if result.previous_state_sha256 != self.evaluated_state.state_sha256:
                    raise ValueError("production_stop_selection_stale_state")
            elif self.outcome == "active_action_in_progress":
                if not active_action or self.selection_result is not None:
                    raise ValueError("production_stop_active_action_outcome_mismatch")
            elif self.outcome == "no_feasible_action":
                if expected_release or active_action or self.selection_result is not None:
                    raise ValueError("production_stop_no_feasible_outcome_mismatch")
            elif self.outcome == "adaptive_calibration_required_before_selection":
                if expected_release or active_action or self.selection_result is not None:
                    raise ValueError("production_stop_calibration_required_outcome_mismatch")
            else:
                raise ValueError("production_stop_stateful_outcome_invalid")
        payload = self.model_dump(mode="json", exclude={"decision_sha256"})
        if hash_canonical(payload) != self.decision_sha256:
            raise ValueError("production_stop_decision_hash_mismatch")
        return self


def freeze_production_stop_decision(
    *,
    evaluated_state: SequentialVerificationState | None,
    release_assessment: ClaimReleaseAssessment | QualifiedClaimReleaseAssessment,
    blocking_adapter_reasons: list[str],
    outcome: Literal[
        "stopped_released",
        "selected_next_action",
        "active_action_in_progress",
        "no_feasible_action",
        "legacy_receipt_assessment",
        "adaptive_calibration_required_before_audit_genesis",
        "adaptive_calibration_required_before_selection",
    ],
    selection_result: SequentialSelectionResult | None = None,
) -> ProductionStopDecision:
    """Freeze one label-free production stopping decision."""

    payload: dict[str, Any] = {
        "decision_version": "production-stop-decision-v1",
        "stopping_rule": PRODUCTION_STOPPING_RULE,
        "evaluated_state": evaluated_state,
        "release_assessment": release_assessment,
        "blocking_adapter_reasons": sorted(set(blocking_adapter_reasons)),
        "full_release_eligible": should_stop_at_full_frozen_release(
            release_status=release_assessment.status.value,
            blocking_reasons=blocking_adapter_reasons,
            active_action=(
                evaluated_state is not None
                and evaluated_state.session.active_action is not None
            ),
        ),
        "outcome": outcome,
        "selection_result": selection_result,
    }
    return ProductionStopDecision.model_validate(
        {**payload, "decision_sha256": hash_canonical(payload)}
    )


class VerificationCertificate(ContractModel):
    """Complete frozen record of one claim-verification run."""

    certificate_version: Literal["literature-multiverse-verification-v5"] = (
        "literature-multiverse-verification-v5"
    )
    run_id: Annotated[str, Field(pattern=r"^verify-[0-9a-f]{16}$")]
    generated_at: datetime
    status: Literal["released", "abstained"]
    reasons: list[str]
    claim_manifest: dict[str, Any]
    claim_manifest_sha256: str
    corpus: dict[str, Any]
    corpus_sha256: str
    source_evidence_graph: EvidenceGraph
    source_evidence_graph_sha256: str
    evidence_graph: EvidenceGraph
    evidence_graph_sha256: str
    adapter_issues: list[dict[str, Any]]
    synthesis: dict[str, Any]
    synthesis_sha256: str
    counterfactual_reruns: list[dict[str, Any]]
    audit_candidates: list[dict[str, Any]]
    release_assessment: ClaimReleaseAssessment | QualifiedClaimReleaseAssessment
    pipeline_verification: PipelineFingerprintVerification
    complete_corpus_identity: CompleteCorpusIdentity
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None
    adaptive_policy_context: AdaptivePolicyContext | None
    adaptive_calibration_bundle: AdaptiveCalibrationBundle | None
    adaptive_release_candidate: ProspectiveAdaptiveReleaseCandidate | None
    adaptive_prospective_assessment: AdaptiveProspectiveAssessment | None
    sequential_audit_state: SequentialVerificationState | None
    production_stop_decision: ProductionStopDecision
    lineage: list[CertificateLineageStage]
    certificate_sha256: str
    interpretation: Literal[
        "literature-support verification under the declared corpus; not scientific truth"
    ] = "literature-support verification under the declared corpus; not scientific truth"

    @field_validator(
        "claim_manifest_sha256",
        "corpus_sha256",
        "source_evidence_graph_sha256",
        "evidence_graph_sha256",
        "synthesis_sha256",
        "certificate_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("verification_certificate_sha256_invalid")
        return value

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("verification_certificate_reasons_must_be_sorted_unique")
        return value

    @property
    def current_state_hashes(self) -> dict[str, str]:
        """Exact source/current hashes for evaluation and certificate consumers."""

        transitions = (
            list(self.sequential_audit_state.transitions)
            if self.sequential_audit_state is not None
            else []
        )
        correction_receipts = [
            transition.receipt
            for transition in transitions
            if transition.transition_kind == "correction"
        ]
        return dict(
            sorted(
                {
                    "audit_correction_receipts": hash_canonical(correction_receipts),
                    "audit_transition_ledger": hash_canonical(transitions),
                    "current_evidence_graph": self.evidence_graph_sha256,
                    "current_synthesis": self.synthesis_sha256,
                    "source_evidence_graph": self.source_evidence_graph_sha256,
                }.items()
            )
        )

    @property
    def item_risk_bounds(self) -> list[RiskBound]:
        """Recomputable scheduling-only bounds carried by the sealed v2 receipt."""

        receipt = self.item_risk_scoring_receipt
        return [] if receipt is None else list(receipt.bounds)

    @model_validator(mode="after")
    def validate_integrity(self) -> VerificationCertificate:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("verification_certificate_generated_at_requires_timezone")
        if hash_canonical(self.claim_manifest) != self.claim_manifest_sha256:
            raise ValueError("verification_certificate_claim_manifest_hash_mismatch")
        if self.corpus.get("source_sha256") != self.corpus_sha256:
            raise ValueError("verification_certificate_corpus_source_hash_mismatch")
        if (
            hash_canonical(self.source_evidence_graph)
            != self.source_evidence_graph_sha256
        ):
            raise ValueError(
                "verification_certificate_source_evidence_graph_hash_mismatch"
            )
        if hash_canonical(self.evidence_graph) != self.evidence_graph_sha256:
            raise ValueError("verification_certificate_evidence_graph_hash_mismatch")
        if hash_canonical(self.synthesis) != self.synthesis_sha256:
            raise ValueError("verification_certificate_synthesis_hash_mismatch")
        adapter_issue_keys: list[tuple[str, str, str]] = []
        expected_issue_fields = {
            "code",
            "detail",
            "finding_id",
            "paper_id",
            "severity",
        }
        for issue in self.adapter_issues:
            if (
                not isinstance(issue, dict)
                or set(issue) != expected_issue_fields
                or issue.get("severity") not in {"blocking", "warning"}
                or not isinstance(issue.get("code"), str)
                or not issue["code"].strip()
                or not isinstance(issue.get("detail"), str)
                or not issue["detail"].strip()
                or not (
                    issue.get("paper_id") is None
                    or isinstance(issue.get("paper_id"), str)
                )
                or not (
                    issue.get("finding_id") is None
                    or isinstance(issue.get("finding_id"), str)
                )
            ):
                raise ValueError("verification_certificate_adapter_issue_invalid")
            adapter_issue_keys.append(
                (
                    str(issue.get("finding_id") or ""),
                    str(issue.get("paper_id") or ""),
                    issue["code"],
                )
            )
        if adapter_issue_keys != sorted(set(adapter_issue_keys)):
            raise ValueError(
                "verification_certificate_adapter_issues_not_sorted_unique"
            )
        if (
            self.release_assessment.evidence_graph_sha256
            != self.evidence_graph_sha256
            or self.release_assessment.synthesis_sha256 != self.synthesis_sha256
        ):
            raise ValueError("verification_certificate_release_science_mismatch")
        if self.release_assessment.question_id != self.claim_manifest.get("question_id"):
            raise ValueError("verification_certificate_release_claim_identity_mismatch")
        manifest_protocol = self.claim_manifest.get("protocol")
        if (
            not isinstance(manifest_protocol, dict)
            or self.corpus.get("declared_corpus_cutoff")
            != manifest_protocol.get("corpus_cutoff")
        ):
            raise ValueError("verification_certificate_corpus_cutoff_mismatch")
        verification = self.pipeline_verification
        if (
            verification.status != "matched"
            or verification.computed_pipeline_sha256 != self.release_assessment.pipeline_sha256
        ):
            raise ValueError("verification_certificate_pipeline_not_matched")
        candidate_ids = [row.get("item_id") for row in self.audit_candidates]
        if (
            any(not isinstance(item_id, str) or not item_id for item_id in candidate_ids)
            or candidate_ids != sorted(set(candidate_ids))
            or candidate_ids != self.release_assessment.audit.candidate_item_ids
        ):
            raise ValueError("verification_certificate_audit_candidate_identity_mismatch")
        counterfactual_ids = [row.get("item_id") for row in self.counterfactual_reruns]
        if counterfactual_ids != candidate_ids:
            raise ValueError("verification_certificate_counterfactual_identity_mismatch")
        scoring_receipt = self.item_risk_scoring_receipt
        if scoring_receipt is not None:
            try:
                scoring_receipt = ItemRiskScoringRunReceipt.model_validate(
                    scoring_receipt.model_dump(mode="json")
                )
            except ValueError as exc:
                raise ValueError(
                    "verification_certificate_item_risk_receipt_invalid"
                ) from exc
            bound_ids = [bound.item_id for bound in scoring_receipt.bounds]
            receipt_candidate_ids = [
                candidate.item_id for candidate in scoring_receipt.candidates
            ]
            expected_receipt_ids = candidate_ids
            if self.sequential_audit_state is not None:
                state = self.sequential_audit_state
                expected_receipt_ids = sorted(
                    candidate.item_id for candidate in state.initial_candidates
                )
                removed_current_ids = set(expected_receipt_ids) - set(candidate_ids)
                if (
                    not set(candidate_ids) <= set(expected_receipt_ids)
                    or not removed_current_ids
                    <= set(state.session.resolved_item_ids)
                ):
                    raise ValueError(
                        "verification_certificate_item_risk_projection_mismatch"
                    )
            if (
                bound_ids != expected_receipt_ids
                or receipt_candidate_ids != expected_receipt_ids
            ):
                raise ValueError("verification_certificate_item_risk_identity_mismatch")
            source_estimates = {
                estimate.estimate_id: estimate
                for estimate in self.source_evidence_graph.outcome_estimates
            }
            if any(
                candidate.score_input_sha256
                != hash_canonical(source_estimates[candidate.item_id])
                for candidate in scoring_receipt.candidates
            ):
                raise ValueError(
                    "verification_certificate_item_risk_source_snapshot_mismatch"
                )
            if scoring_receipt.pipeline_verification != verification or any(
                bound.pipeline_sha256 != self.release_assessment.pipeline_sha256
                or bound.pipeline_verification_sha256
                != verification.verification_sha256
                for bound in scoring_receipt.bounds
            ):
                raise ValueError("verification_certificate_item_risk_pipeline_mismatch")
        if self.sequential_audit_state is not None:
            state = self.sequential_audit_state
            audit_timestamps = [state.session.created_at]
            if state.session.active_action is not None:
                audit_timestamps.append(state.session.active_action.selected_at)
            audit_timestamps.extend(
                step.adjudication.completed_at for step in state.session.steps
            )
            if any(timestamp > self.generated_at for timestamp in audit_timestamps):
                raise ValueError("verification_certificate_predates_audit_state")
            if (
                state.session.pipeline_sha256 != self.release_assessment.pipeline_sha256
                or state.initial_graph != self.source_evidence_graph
                or state.graph_sha256 != self.evidence_graph_sha256
                or state.synthesis_sha256 != self.synthesis_sha256
            ):
                raise ValueError("verification_certificate_sequential_state_mismatch")
            expected_policy_sha256 = hash_canonical(
                {
                    "policy_context_version": "verification-manifest-context-v1",
                    "claim_manifest": self.claim_manifest,
                }
            )
            if state.session.policy_sha256 != expected_policy_sha256:
                raise ValueError(
                    "verification_certificate_sequential_claim_context_mismatch"
                )
            if (
                state.state_sha256
                != self.release_assessment.audit.sequential_state_sha256
                or state.session.budget != self.release_assessment.audit.budget
                or sorted(candidate.item_id for candidate in state.candidates)
                != self.release_assessment.audit.candidate_item_ids
            ):
                raise ValueError("verification_certificate_sequential_audit_mismatch")
            active_reason = "active_audit_action_unresolved"
            has_active_action = state.session.active_action is not None
            audit_records_active = active_reason in self.release_assessment.audit.reasons
            assessment_records_active = (
                f"audit:{active_reason}" in self.release_assessment.reasons
            )
            if has_active_action != audit_records_active or (
                has_active_action != assessment_records_active
            ):
                raise ValueError(
                    "verification_certificate_active_audit_gate_mismatch"
                )
            if has_active_action and self.release_assessment.audit.status != "blocked":
                raise ValueError(
                    "verification_certificate_active_audit_cannot_be_release_eligible"
                )
        else:
            if self.release_assessment.audit.sequential_state_sha256 is not None:
                raise ValueError("verification_certificate_sequential_state_missing")
            if self.source_evidence_graph != self.evidence_graph:
                raise ValueError(
                    "verification_certificate_unreplayed_graph_correction"
                )
        correction_transitions = (
            [
                transition
                for transition in self.sequential_audit_state.transitions
                if transition.transition_kind == "correction"
            ]
            if self.sequential_audit_state is not None
            else []
        )
        if not correction_transitions and (
            self.source_evidence_graph != self.evidence_graph
        ):
            raise ValueError(
                "verification_certificate_zero_correction_graph_mismatch"
            )
        assurance = self.corpus.get("provenance_assurance")
        if not isinstance(assurance, dict):
            raise ValueError("verification_certificate_corpus_assurance_missing")
        if assurance.get("assurance_version") != "corpus-provenance-assurance-v1":
            raise ValueError("verification_certificate_corpus_assurance_version_invalid")
        release_eligible = assurance.get("release_eligible")
        if not isinstance(release_eligible, bool):
            raise ValueError("verification_certificate_corpus_assurance_eligibility_invalid")
        assurance_status = assurance.get("status")
        replay_sha256 = assurance.get("replay_sha256")
        reason = assurance.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("verification_certificate_corpus_assurance_reason_invalid")
        metadata = self.corpus.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("verification_certificate_corpus_metadata_invalid")
        corpus_eligibility = self.corpus.get("eligibility")
        native_manifest: NativeSourceManifest | None = None
        declares_native_membership = (
            metadata.get("grounding_package_version")
            in {
                "typed-evidence-grounding-package-v3",
                "typed-evidence-grounding-package-v4",
            }
            or metadata.get("source_manifest_membership_bound") is True
        )
        if declares_native_membership:
            manifest_payload = metadata.get("native_source_manifest")
            if not isinstance(manifest_payload, dict):
                raise ValueError(
                    "verification_certificate_native_source_manifest_missing"
                )
            try:
                native_manifest = NativeSourceManifest.model_validate(manifest_payload)
            except ValueError as exc:
                raise ValueError(
                    "verification_certificate_native_source_manifest_invalid"
                ) from exc
            if hash_canonical(native_manifest) != metadata.get("source_manifest_sha256"):
                raise ValueError(
                    "verification_certificate_native_source_manifest_hash_mismatch"
                )
            if native_manifest.question_id != self.claim_manifest.get("question_id"):
                raise ValueError(
                    "verification_certificate_native_source_manifest_question_mismatch"
                )
            if not isinstance(corpus_eligibility, list):
                raise ValueError(
                    "verification_certificate_native_source_manifest_eligibility_missing"
                )
            eligibility_by_paper = {
                row.get("paper_id"): row
                for row in corpus_eligibility
                if isinstance(row, dict) and isinstance(row.get("paper_id"), str)
            }
            if len(eligibility_by_paper) != len(corpus_eligibility):
                raise ValueError(
                    "verification_certificate_native_source_manifest_eligibility_duplicate"
                )
            manifest_by_publication = {
                record.publication.publication_id: record
                for record in native_manifest.records
            }
            manifest_paper_ids = {
                record.publication.paper_id for record in native_manifest.records
            }
            if set(eligibility_by_paper) != manifest_paper_ids:
                raise ValueError(
                    "verification_certificate_native_source_eligibility_membership_mismatch"
                )
            terminal_membership = metadata.get("terminal_fragment_membership")
            if not isinstance(terminal_membership, list) or any(
                not isinstance(row, dict)
                or set(row)
                != {"fragment_sha256", "paper_id", "publication_id", "status"}
                or not isinstance(row.get("fragment_sha256"), str)
                or SHA256_RE.fullmatch(row["fragment_sha256"]) is None
                or not isinstance(row.get("paper_id"), str)
                or not row["paper_id"]
                or not isinstance(row.get("publication_id"), str)
                or not row["publication_id"]
                or row.get("status") not in {"estimable", "non_estimable"}
                for row in terminal_membership
            ):
                raise ValueError(
                    "verification_certificate_terminal_fragment_membership_invalid"
                )
            terminal_by_publication = {
                row["publication_id"]: row for row in terminal_membership
            }
            if (
                len(terminal_by_publication) != len(terminal_membership)
                or set(terminal_by_publication) != set(manifest_by_publication)
                or metadata.get("terminal_fragment_records")
                != len(terminal_membership)
                or metadata.get("terminal_fragment_membership_sha256")
                != hash_canonical(terminal_membership)
            ):
                raise ValueError(
                    "verification_certificate_terminal_fragment_membership_mismatch"
                )
            for publication_id, record in manifest_by_publication.items():
                if (
                    terminal_by_publication[publication_id]["paper_id"]
                    != record.publication.paper_id
                ):
                    raise ValueError(
                        "verification_certificate_terminal_fragment_paper_mismatch"
                    )
            publications_by_id = {
                publication.publication_id: publication
                for publication in self.source_evidence_graph.publications
            }
            if set(publications_by_id) != set(manifest_by_publication):
                raise ValueError(
                    "verification_certificate_native_source_manifest_membership_mismatch"
                )
            for record in native_manifest.records:
                publication = publications_by_id.get(record.publication.publication_id)
                if publication is not None and record.publication.model_dump(
                    mode="json"
                ) != publication.model_dump(mode="json"):
                    raise ValueError(
                        "verification_certificate_native_source_publication_mismatch"
                    )
                eligibility_row = eligibility_by_paper.get(record.publication.paper_id)
                if (
                    eligibility_row is None
                    or eligibility_row.get("status") != "included"
                    or eligibility_row.get("source")
                    != record.source_document.source_locator
                ):
                    raise ValueError(
                        "verification_certificate_native_source_eligibility_mismatch"
                    )
        rendered_prompt_sha256s = metadata.get("rendered_prompt_sha256s")
        evaluation_schema_sha256s = metadata.get("evaluation_schema_sha256s")
        provider_execution_receipts = metadata.get("provider_execution_receipts")
        execution_context_complete = (
            isinstance(metadata.get("extraction_context_sha256"), str)
            and SHA256_RE.fullmatch(metadata["extraction_context_sha256"]) is not None
            and isinstance(metadata.get("extraction_context_receipt_sha256"), str)
            and SHA256_RE.fullmatch(metadata["extraction_context_receipt_sha256"])
            is not None
            and metadata.get("extraction_context_receipt_sha256")
            == metadata.get("replayed_extraction_context_receipt_sha256")
            and isinstance(metadata.get("question_config_sha256"), str)
            and SHA256_RE.fullmatch(metadata["question_config_sha256"]) is not None
            and isinstance(rendered_prompt_sha256s, list)
            and rendered_prompt_sha256s == sorted(set(rendered_prompt_sha256s))
            and bool(rendered_prompt_sha256s)
            and all(
                isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
                for value in rendered_prompt_sha256s
            )
            and isinstance(evaluation_schema_sha256s, list)
            and evaluation_schema_sha256s == sorted(set(evaluation_schema_sha256s))
            and bool(evaluation_schema_sha256s)
            and all(
                isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
                for value in evaluation_schema_sha256s
            )
            and isinstance(provider_execution_receipts, list)
            and bool(provider_execution_receipts)
            and all(
                isinstance(receipt, dict)
                and isinstance(receipt.get("call_count"), int)
                and receipt["call_count"] >= 1
                and isinstance(receipt.get("execution_identity_sha256"), str)
                and SHA256_RE.fullmatch(receipt["execution_identity_sha256"])
                is not None
                and isinstance(receipt.get("receipt_sha256"), str)
                and SHA256_RE.fullmatch(receipt["receipt_sha256"]) is not None
                and receipt.get("execution_mode")
                in {
                    "paperclip_archived",
                    "paperclip_live",
                    "ollama_local",
                    "hosted_exact_once",
                }
                and all(
                    isinstance(receipt.get(field), str)
                    and bool(receipt[field].strip())
                    for field in (
                        "model_id",
                        "provider_id",
                        "runtime_id",
                        "runtime_version",
                    )
                )
                for receipt in provider_execution_receipts
            )
        )
        native_membership_complete = (
            metadata.get("grounding_package_version")
            == "typed-evidence-grounding-package-v4"
            and metadata.get("source_manifest_membership_bound") is True
            and isinstance(metadata.get("source_manifest_sha256"), str)
            and SHA256_RE.fullmatch(metadata["source_manifest_sha256"]) is not None
            and native_manifest is not None
            and isinstance(corpus_eligibility, list)
            and metadata.get("source_manifest_records") == len(corpus_eligibility)
            and isinstance(metadata.get("terminal_fragment_membership"), list)
            and metadata.get("terminal_fragment_records") == len(corpus_eligibility)
            and hash_canonical(metadata["terminal_fragment_membership"])
            == metadata.get("terminal_fragment_membership_sha256")
            and isinstance(metadata.get("native_corpus_cutoff"), str)
            and bool(metadata["native_corpus_cutoff"].strip())
            and execution_context_complete
        )
        complete_identity = self.complete_corpus_identity
        expected_publication_ids = sorted(
            publication.publication_id
            for publication in self.source_evidence_graph.publications
        )
        expected_membership_basis = "frozen_corpus_publications"
        expected_source_manifest_sha256 = None
        if native_manifest is not None:
            expected_membership_basis = "source_manifest"
            expected_source_manifest_sha256 = metadata.get("source_manifest_sha256")
        if (
            complete_identity.corpus_id != self.corpus.get("corpus_id")
            or complete_identity.corpus_source_sha256 != self.corpus_sha256
            or complete_identity.corpus_cutoff
            != self.corpus.get("declared_corpus_cutoff")
            or complete_identity.membership_basis != expected_membership_basis
            or complete_identity.publication_ids != expected_publication_ids
            or complete_identity.source_manifest_sha256
            != expected_source_manifest_sha256
        ):
            raise ValueError(
                "verification_certificate_complete_corpus_identity_mismatch"
            )
        adaptive_fields = (
            self.adaptive_policy_context,
            self.adaptive_calibration_bundle,
            self.adaptive_release_candidate,
            self.adaptive_prospective_assessment,
        )
        if any(value is not None for value in adaptive_fields) and not all(
            value is not None for value in adaptive_fields
        ):
            raise ValueError("verification_certificate_adaptive_lineage_incomplete")
        if all(value is not None for value in adaptive_fields):
            assert self.adaptive_policy_context is not None
            assert self.adaptive_calibration_bundle is not None
            assert self.adaptive_release_candidate is not None
            assert self.adaptive_prospective_assessment is not None
            try:
                adaptive_bundle = validate_adaptive_calibration_bundle_integrity(
                    self.adaptive_calibration_bundle
                )
                recomputed_adaptive_assessment = assess_adaptive_release_candidate(
                    self.adaptive_release_candidate,
                    adaptive_bundle,
                )
            except AdaptiveCalibrationError as exc:
                raise ValueError(
                    "verification_certificate_adaptive_replay_invalid"
                ) from exc
            matching_context = next(
                (
                    context
                    for context in adaptive_bundle.development_freeze.policy_contexts
                    if context.policy_arm_id
                    == self.adaptive_release_candidate.policy_arm_id
                ),
                None,
            )
            manifest_claim = self.claim_manifest.get("claim")
            manifest_release = self.claim_manifest.get("release")
            manifest_audit = self.claim_manifest.get("audit")
            manifest_audit_guard = self.claim_manifest.get("audit_guard")
            manifest_protocol = self.claim_manifest.get("protocol")
            if not all(
                isinstance(value, dict)
                for value in (
                    manifest_claim,
                    manifest_release,
                    manifest_audit,
                    manifest_audit_guard,
                    manifest_protocol,
                )
            ):
                raise ValueError(
                    "verification_certificate_adaptive_manifest_context_incomplete"
                )
            assert isinstance(manifest_claim, dict)
            assert isinstance(manifest_release, dict)
            assert isinstance(manifest_audit, dict)
            assert isinstance(manifest_audit_guard, dict)
            assert isinstance(manifest_protocol, dict)
            if matching_context is not None:
                try:
                    expected_adaptive_context = freeze_adaptive_policy_context(
                        policy_arm_id=matching_context.policy_arm_id,
                        population_id=str(self.claim_manifest.get("population_id", "")),
                        pipeline_sha256=self.release_assessment.pipeline_sha256,
                        allocation_policy={
                            "name": manifest_release.get("audit_allocation_policy"),
                            "seed": manifest_release.get("audit_seed"),
                        },
                        budget_minutes=self.release_assessment.audit.budget,
                        release_config=manifest_release,
                        audit_config={
                            "scheduler_inputs": manifest_audit,
                            "release_guard": manifest_audit_guard,
                        },
                        target_semantics={
                            "claim_manifest_version": self.claim_manifest.get(
                                "claim_manifest_version"
                            ),
                            "claim_direction": manifest_claim.get("direction"),
                            "outcome_name": manifest_claim.get("outcome_name"),
                            "contrast_id": manifest_claim.get("contrast_id"),
                            "estimand": manifest_claim.get("estimand"),
                            "qualified_target": self.claim_manifest.get(
                                "qualified_target"
                            ),
                            "global_condition_target": self.claim_manifest.get(
                                "global_condition_target"
                            ),
                            "decision_loss": (
                                "released_claim_decision_differs_from_reference_verdict"
                            ),
                        },
                        corpus_protocol_context=manifest_protocol,
                        score_feature_names=CLAIM_RELEASE_RISK_FEATURE_NAMES,
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "verification_certificate_adaptive_manifest_context_invalid"
                    ) from exc
            else:
                expected_adaptive_context = None
            if matching_context is None:
                raise ValueError(
                    "verification_certificate_adaptive_policy_context_missing"
                )
            if matching_context != expected_adaptive_context:
                raise ValueError(
                    "verification_certificate_adaptive_policy_context_manifest_mismatch"
                )
            if (
                self.adaptive_policy_context != matching_context
                or self.adaptive_release_candidate.policy_context_sha256
                != matching_context.policy_context_sha256
                or self.adaptive_release_candidate.corpus != complete_identity
                or self.adaptive_release_candidate.question_id
                != self.claim_manifest.get("question_id")
                or self.adaptive_release_candidate.population_id
                != self.claim_manifest.get("population_id")
                or self.adaptive_release_candidate.domain
                != self.claim_manifest.get("domain")
                or self.adaptive_prospective_assessment
                != recomputed_adaptive_assessment
            ):
                raise ValueError(
                    "verification_certificate_adaptive_replay_lineage_mismatch"
                )
        if assurance_status == "source_replayed_native_grounding":
            expected_release_eligible = (
                isinstance(replay_sha256, str)
                and SHA256_RE.fullmatch(replay_sha256) is not None
                and self.corpus.get("source_format")
                == "typed_evidence_grounding_package_json"
                and metadata.get("grounding_replay_sha256") == replay_sha256
                and native_membership_complete
            )
        elif assurance_status == "embedded_synthetic_fixture":
            valid_fixture_identity = (
                replay_sha256 is None
                and self.corpus.get("source_format") == "embedded_synthetic_fixture"
                and self.corpus.get("source_label")
                == "embedded:offline-verifier-fixture-v1"
                and metadata.get("empirical_evidence") is False
                and metadata.get("purpose") == "offline_integration_test"
            )
            if not valid_fixture_identity:
                raise ValueError(
                    "verification_certificate_embedded_fixture_identity_invalid"
                )
            expected_release_eligible = False
        elif assurance_status == "unverified_source_provenance":
            if replay_sha256 is not None:
                raise ValueError(
                    "verification_certificate_unverified_corpus_forbids_replay_hash"
                )
            expected_release_eligible = False
        else:
            raise ValueError("verification_certificate_corpus_assurance_status_invalid")
        if release_eligible != expected_release_eligible:
            raise ValueError("verification_certificate_corpus_assurance_escalation")
        if assurance_status == "source_replayed_native_grounding" and (
            metadata.get("pipeline_fingerprint_sha256")
            != self.release_assessment.pipeline_sha256
        ) and not any(
            issue.get("severity") == "blocking"
            and issue.get("code") == "corpus_pipeline_identity_mismatch"
            for issue in self.adapter_issues
        ):
            raise ValueError(
                "verification_certificate_corpus_pipeline_mismatch_requires_blocker"
            )
        if (
            assurance_status == "source_replayed_native_grounding"
            and self.corpus.get("corpus_id") != self.claim_manifest.get("question_id")
        ) and not any(
            issue.get("severity") == "blocking"
            and issue.get("code") == "corpus_question_identity_mismatch"
            for issue in self.adapter_issues
        ):
            raise ValueError(
                "verification_certificate_corpus_question_mismatch_requires_blocker"
            )
        if assurance_status == "source_replayed_native_grounding" and (
            not native_membership_complete
        ) and not any(
            issue.get("severity") == "blocking"
            and issue.get("code") == "native_source_manifest_membership_unbound"
            for issue in self.adapter_issues
        ):
            raise ValueError(
                "verification_certificate_native_membership_requires_blocker"
            )
        if (
            assurance_status == "source_replayed_native_grounding"
            and native_membership_complete
            and metadata.get("native_corpus_cutoff")
            != self.corpus.get("declared_corpus_cutoff")
        ) and not any(
            issue.get("severity") == "blocking"
            and issue.get("code") == "corpus_cutoff_identity_mismatch"
            for issue in self.adapter_issues
        ):
            raise ValueError(
                "verification_certificate_native_cutoff_mismatch_requires_blocker"
            )
        if self.status == "released" and not release_eligible:
            raise ValueError("unverified_corpus_cannot_have_released_certificate")
        if not release_eligible:
            expected_blocker = (
                "embedded_synthetic_fixture_not_empirical"
                if assurance_status == "embedded_synthetic_fixture"
                else "unverified_source_provenance"
            )
            if not any(
                issue.get("severity") == "blocking"
                and issue.get("code") == expected_blocker
                for issue in self.adapter_issues
            ):
                raise ValueError("unverified_corpus_requires_provenance_blocker")
        expected_graph_counts = {
            "cohorts": len(self.source_evidence_graph.cohorts),
            "estimates": len(self.source_evidence_graph.outcome_estimates),
            "publications": len(self.source_evidence_graph.publications),
            "studies": len(self.source_evidence_graph.studies),
        }
        if self.corpus.get("graph_counts") != expected_graph_counts:
            raise ValueError("verification_certificate_corpus_graph_counts_mismatch")
        eligibility = self.corpus.get("eligibility")
        if not isinstance(eligibility, list) or any(
            not isinstance(row, dict)
            or row.get("status") not in {"included", "excluded", "pending"}
            for row in eligibility
        ):
            raise ValueError("verification_certificate_corpus_eligibility_invalid")
        expected_eligibility_counts = {
            status: sum(row["status"] == status for row in eligibility)
            for status in ("excluded", "included", "pending")
        }
        if self.corpus.get("eligibility_counts") != expected_eligibility_counts:
            raise ValueError("verification_certificate_corpus_eligibility_counts_mismatch")
        blocking_adapter_reasons = {
            f"adapter:{issue.get('code')}"
            for issue in self.adapter_issues
            if issue.get("severity") == "blocking"
            and isinstance(issue.get("code"), str)
            and issue.get("code")
        }
        decision = self.production_stop_decision
        if decision.blocking_adapter_reasons != sorted(blocking_adapter_reasons):
            raise ValueError("verification_certificate_production_stop_blocker_mismatch")
        preselection = decision.release_assessment
        if (
            preselection.question_id != self.release_assessment.question_id
            or preselection.pipeline_sha256 != self.release_assessment.pipeline_sha256
            or preselection.evidence_graph_sha256 != self.evidence_graph_sha256
            or preselection.synthesis_sha256 != self.synthesis_sha256
        ):
            raise ValueError("verification_certificate_production_stop_science_mismatch")
        calibration = preselection.calibration
        adaptive_ledger_history = ()
        adaptive_ledger_context_sha256 = None
        adaptive_ledger_bundle_sha256 = None
        if self.sequential_audit_state is not None:
            try:
                (
                    adaptive_ledger_history,
                    adaptive_ledger_context_sha256,
                    adaptive_ledger_bundle_sha256,
                ) = adaptive_preselection_history_from_state(
                    self.sequential_audit_state
                )
            except SequentialVerificationContractError as exc:
                raise ValueError(
                    "verification_certificate_adaptive_history_invalid"
                ) from exc
        if self.adaptive_prospective_assessment is not None:
            assert self.adaptive_calibration_bundle is not None
            assert self.adaptive_release_candidate is not None
            assert self.adaptive_policy_context is not None
            adaptive_assessment = self.adaptive_prospective_assessment
            adaptive_evaluated_state = decision.evaluated_state
            if self.sequential_audit_state is None or adaptive_evaluated_state is None:
                raise ValueError(
                    "verification_certificate_adaptive_sequential_state_missing"
                )
            selection_transitions_exist = any(
                transition.transition_kind == "selection"
                for transition in self.sequential_audit_state.transitions
            )
            if selection_transitions_exist and not adaptive_ledger_history:
                raise ValueError(
                    "verification_certificate_adaptive_history_started_midstream"
                )
            if (
                adaptive_ledger_context_sha256
                != self.adaptive_policy_context.policy_context_sha256
                or adaptive_ledger_bundle_sha256
                != self.adaptive_calibration_bundle.bundle_sha256
            ):
                raise ValueError(
                    "verification_certificate_adaptive_history_identity_mismatch"
                )
            try:
                predecessor_states = selection_predecessor_states_from_state(
                    self.sequential_audit_state
                )
            except SequentialVerificationContractError as exc:
                raise ValueError(
                    "verification_certificate_adaptive_history_snapshot_invalid"
                ) from exc
            if len(predecessor_states) != len(adaptive_ledger_history):
                raise ValueError(
                    "verification_certificate_adaptive_history_snapshot_count_mismatch"
                )
            # Import lazily to avoid the module-level verifier/certificate cycle.  A
            # v5 certificate is independently validated from its embedded manifest,
            # pipeline verification, item-risk receipt, and predecessor states; it
            # does not trust semantically caller-authored checkpoint fields merely
            # because their unkeyed hashes are internally consistent.
            from literature_multiverse.verifier import (
                ClaimManifest,
                VerificationContractError,
                recompute_verifier_adaptive_preselection_checkpoint,
            )

            try:
                certificate_manifest = ClaimManifest.model_validate(
                    self.claim_manifest
                )
            except ValueError as exc:
                raise ValueError(
                    "verification_certificate_adaptive_manifest_invalid"
                ) from exc
            for checkpoint_index, (checkpoint, predecessor_state) in enumerate(
                zip(
                    adaptive_ledger_history,
                    predecessor_states,
                    strict=True,
                )
            ):
                try:
                    recomputed_checkpoint = (
                        recompute_verifier_adaptive_preselection_checkpoint(
                            manifest=certificate_manifest,
                            state=predecessor_state,
                            pipeline_verification=self.pipeline_verification,
                            item_risk_scoring_receipt=(
                                self.item_risk_scoring_receipt
                            ),
                            blocking_adapter_reasons=(
                                decision.blocking_adapter_reasons
                            ),
                        )
                    )
                except VerificationContractError as exc:
                    raise ValueError(
                        "verification_certificate_adaptive_checkpoint_scientific_"
                        f"replay_failed:{checkpoint_index}"
                    ) from exc
                if recomputed_checkpoint != checkpoint:
                    raise ValueError(
                        "verification_certificate_adaptive_checkpoint_semantics_"
                        f"mismatch:{checkpoint_index}"
                    )
            if self.sequential_audit_state.session.active_action is None:
                try:
                    expected_current_adaptive_state = (
                        freeze_preselection_state_from_production_components(
                            sequential_state=adaptive_evaluated_state,
                            release_assessment=preselection,
                            blocking_adapter_reasons=decision.blocking_adapter_reasons,
                        )
                    )
                except AdaptiveCalibrationError as exc:
                    raise ValueError(
                        "verification_certificate_adaptive_current_state_invalid"
                    ) from exc
                expected_observed_states = [
                    *adaptive_ledger_history,
                    expected_current_adaptive_state,
                ]
            else:
                if not adaptive_ledger_history:
                    raise ValueError(
                        "verification_certificate_adaptive_active_action_checkpoint_missing"
                    )
                if adaptive_assessment.status == "released":
                    raise ValueError(
                        "verification_certificate_adaptive_action_selected_after_qualifying_release"
                    )
                expected_observed_states = list(adaptive_ledger_history)
            if (
                self.adaptive_release_candidate.observed_states
                != expected_observed_states
            ):
                raise ValueError(
                    "verification_certificate_adaptive_history_proof_mismatch"
                )
            common_calibration_mismatch = (
                calibration.calibration_contract
                != "adaptive-first-release-trajectory-v1"
                or calibration.frozen_bundle_sha256
                != self.adaptive_calibration_bundle.bundle_sha256
                or calibration.release_candidate_sha256
                != self.adaptive_release_candidate.candidate_sha256
                or calibration.policy_context_sha256
                != self.adaptive_policy_context.policy_context_sha256
                or calibration.label_source
                != self.adaptive_calibration_bundle.label_source
                or calibration.guarantee_scope != adaptive_assessment.guarantee_scope
                or adaptive_assessment.question_id != preselection.question_id
            )
            if adaptive_evaluated_state.session.active_action is None:
                assessment_calibration_mismatch = (
                    calibration.prospective_assessment_sha256
                    != adaptive_assessment.assessment_sha256
                    or calibration.scalar_risk_score
                    != adaptive_assessment.scalar_risk_score
                    or calibration.threshold != adaptive_assessment.threshold
                    or calibration.status != adaptive_assessment.status
                    or calibration.reason != adaptive_assessment.reason
                )
            else:
                active_payload = {
                    "adaptive_active_action_block_version": "1",
                    "bundle_sha256": self.adaptive_calibration_bundle.bundle_sha256,
                    "candidate_sha256": self.adaptive_release_candidate.candidate_sha256,
                    "sequential_state_sha256": adaptive_evaluated_state.state_sha256,
                }
                assessment_calibration_mismatch = (
                    calibration.status != "abstained"
                    or calibration.reason
                    != "active_audit_action_unresolved_before_calibration"
                    or calibration.prospective_assessment_sha256
                    != hash_canonical(active_payload)
                    or calibration.scalar_risk_score is not None
                    or calibration.threshold is not None
                )
            if common_calibration_mismatch or assessment_calibration_mismatch:
                raise ValueError(
                    "verification_certificate_adaptive_release_assessment_mismatch"
                )
        elif calibration.calibration_contract == (
            "adaptive-first-release-trajectory-v1"
        ):
            raise ValueError(
                "verification_certificate_adaptive_release_artifacts_missing"
            )
        elif (
            adaptive_ledger_history
            or adaptive_ledger_context_sha256 is not None
            or adaptive_ledger_bundle_sha256 is not None
        ):
            raise ValueError(
                "verification_certificate_adaptive_history_artifacts_missing"
            )
        selection = decision.selection_result
        evaluated_state = decision.evaluated_state
        if selection is not None:
            if (
                evaluated_state is None
                or self.sequential_audit_state != selection.state
                or selection.state.graph != evaluated_state.graph
                or selection.state.synthesis != evaluated_state.synthesis
                or selection.state.candidates != evaluated_state.candidates
                or selection.action.selection_state_sha256
                != evaluated_state.session.session_sha256
                or selection.state.session.previous_session_sha256
                != evaluated_state.session.session_sha256
                or selection.action.selected_at != self.generated_at
            ):
                raise ValueError(
                    "verification_certificate_production_stop_transition_mismatch"
                )
        else:
            if self.sequential_audit_state != evaluated_state:
                raise ValueError(
                    "verification_certificate_production_stop_final_state_mismatch"
                )
            if self.release_assessment != preselection:
                raise ValueError(
                    "verification_certificate_production_stop_assessment_mismatch"
                )
            if (
                self.sequential_audit_state is not None
                and self.sequential_audit_state.session.active_action is not None
                and self.sequential_audit_state.session.active_action.selected_at
                >= self.generated_at
            ):
                raise ValueError(
                    "verification_certificate_new_active_action_requires_transition"
                )
        if decision.full_release_eligible != (self.status == "released"):
            raise ValueError("verification_certificate_production_stop_status_mismatch")
        expected_reasons = sorted(
            set(self.release_assessment.reasons) | blocking_adapter_reasons
        )
        if self.reasons != expected_reasons:
            raise ValueError("verification_certificate_reason_ledger_mismatch")
        expected_status = (
            "released"
            if self.release_assessment.status.value == "released" and not expected_reasons
            else "abstained"
        )
        if self.status != expected_status:
            raise ValueError("verification_certificate_status_gate_mismatch")
        expected_run_identity = hash_canonical(
            {
                "claim_manifest_sha256": self.claim_manifest_sha256,
                "corpus_sha256": self.corpus_sha256,
                "source_evidence_graph_sha256": self.source_evidence_graph_sha256,
                "evidence_graph_sha256": self.evidence_graph_sha256,
                "release_decision_sha256": self.release_assessment.decision_sha256,
                "pipeline_verification_sha256": verification.verification_sha256,
                "production_stop_decision_sha256": decision.decision_sha256,
                "complete_corpus_membership_sha256": (
                    self.complete_corpus_identity.membership_sha256
                ),
                "item_risk_scoring_receipt_sha256": (
                    None
                    if self.item_risk_scoring_receipt is None
                    else self.item_risk_scoring_receipt.receipt_sha256
                ),
                "adaptive_policy_context_sha256": (
                    None
                    if self.adaptive_policy_context is None
                    else self.adaptive_policy_context.policy_context_sha256
                ),
                "adaptive_calibration_bundle_sha256": (
                    None
                    if self.adaptive_calibration_bundle is None
                    else self.adaptive_calibration_bundle.bundle_sha256
                ),
                "adaptive_release_candidate_sha256": (
                    None
                    if self.adaptive_release_candidate is None
                    else self.adaptive_release_candidate.candidate_sha256
                ),
                "adaptive_prospective_assessment_sha256": (
                    None
                    if self.adaptive_prospective_assessment is None
                    else self.adaptive_prospective_assessment.assessment_sha256
                ),
            }
        )
        if self.run_id != f"verify-{expected_run_identity[:16]}":
            raise ValueError("verification_certificate_run_identity_mismatch")
        expected_stage_names = [
            "pipeline_identity_verification",
            "corpus_to_evidence_graph",
            "audit_correction_replay",
            "evidence_graph_to_synthesis",
            "counterfactual_verification_priority",
            "item_risk_scoring",
            "adaptive_first_release_calibration",
            "risk_controlled_release",
        ]
        if [stage.stage for stage in self.lineage] != expected_stage_names:
            raise ValueError("verification_certificate_lineage_stage_mismatch")
        (
            pipeline_stage,
            corpus_stage,
            audit_replay_stage,
            synthesis_stage,
            counterfactual_stage,
            item_risk_stage,
            adaptive_stage,
            release_stage,
        ) = self.lineage
        if (
            pipeline_stage.input_sha256s
            != {"expected_pipeline": verification.expected_pipeline_sha256}
            or pipeline_stage.output_sha256s
            != {"pipeline_verification": verification.verification_sha256}
            or pipeline_stage.method
            != "recomputed-explicit-file-and-settings-fingerprint"
        ):
            raise ValueError("verification_certificate_pipeline_lineage_mismatch")
        if (
            corpus_stage.input_sha256s
            != {
                "claim_manifest": self.claim_manifest_sha256,
                "corpus_source": self.corpus_sha256,
            }
            or corpus_stage.output_sha256s
            != {"source_evidence_graph": self.source_evidence_graph_sha256}
            or corpus_stage.method
            != f"{self.corpus.get('source_format')}:closed-corpus-adapter"
        ):
            raise ValueError("verification_certificate_corpus_lineage_mismatch")
        audit_transitions = (
            list(self.sequential_audit_state.transitions)
            if self.sequential_audit_state is not None
            else []
        )
        audit_correction_receipts = [
            transition.receipt
            for transition in audit_transitions
            if transition.transition_kind == "correction"
        ]
        expected_audit_outputs = {
            "audit_correction_receipts": hash_canonical(audit_correction_receipts),
            "audit_transition_ledger": hash_canonical(audit_transitions),
            "evidence_graph": self.evidence_graph_sha256,
        }
        if self.sequential_audit_state is not None:
            expected_audit_outputs["sequential_audit_state"] = (
                self.sequential_audit_state.state_sha256
            )
        if (
            audit_replay_stage.input_sha256s
            != {"source_evidence_graph": self.source_evidence_graph_sha256}
            or audit_replay_stage.output_sha256s
            != dict(sorted(expected_audit_outputs.items()))
            or audit_replay_stage.method
            != "hash-chained-sequential-audit-correction-replay-v1"
        ):
            raise ValueError(
                "verification_certificate_audit_correction_lineage_mismatch"
            )
        if (
            synthesis_stage.input_sha256s
            != {"evidence_graph": self.evidence_graph_sha256}
            or synthesis_stage.output_sha256s
            != {"synthesis": self.synthesis_sha256}
            or synthesis_stage.method != str(self.synthesis.get("mode", "insufficient"))
        ):
            raise ValueError("verification_certificate_synthesis_lineage_mismatch")
        if (
            counterfactual_stage.input_sha256s
            != {
                "evidence_graph": self.evidence_graph_sha256,
                "synthesis": self.synthesis_sha256,
            }
            or counterfactual_stage.output_sha256s
            != {
                "audit_candidates": hash_canonical(self.audit_candidates),
                "counterfactual_reruns": hash_canonical(self.counterfactual_reruns),
            }
            or counterfactual_stage.method != "actual-leave-one-out-synthesis-reruns"
        ):
            raise ValueError("verification_certificate_counterfactual_lineage_mismatch")
        expected_item_receipt_sha256 = (
            hash_canonical(None)
            if self.item_risk_scoring_receipt is None
            else self.item_risk_scoring_receipt.receipt_sha256
        )
        expected_item_method = (
            "not-supplied"
            if self.item_risk_scoring_receipt is None
            else "recomputed-self-contained-item-risk-scoring-run-v2"
        )
        if (
            item_risk_stage.input_sha256s
            != {
                "audit_candidates": hash_canonical(self.audit_candidates),
                "pipeline_verification": verification.verification_sha256,
            }
            or item_risk_stage.output_sha256s
            != {"item_risk_scoring_receipt": expected_item_receipt_sha256}
            or item_risk_stage.method != expected_item_method
        ):
            raise ValueError("verification_certificate_item_risk_lineage_mismatch")
        expected_adaptive_assessment_sha256 = (
            hash_canonical(None)
            if self.adaptive_prospective_assessment is None
            else self.adaptive_prospective_assessment.assessment_sha256
        )
        expected_adaptive_inputs = {
            "adaptive_calibration_bundle": (
                hash_canonical(None)
                if self.adaptive_calibration_bundle is None
                else self.adaptive_calibration_bundle.bundle_sha256
            ),
            "adaptive_policy_context": (
                hash_canonical(None)
                if self.adaptive_policy_context is None
                else self.adaptive_policy_context.policy_context_sha256
            ),
            "adaptive_release_candidate": (
                hash_canonical(None)
                if self.adaptive_release_candidate is None
                else self.adaptive_release_candidate.candidate_sha256
            ),
            "complete_corpus_identity": (
                self.complete_corpus_identity.membership_sha256
            ),
        }
        expected_adaptive_method = (
            "not-supplied"
            if self.adaptive_prospective_assessment is None
            else "recomputed-complete-question-first-release-trajectory-v1"
        )
        if (
            adaptive_stage.input_sha256s
            != dict(sorted(expected_adaptive_inputs.items()))
            or adaptive_stage.output_sha256s
            != {
                "adaptive_prospective_assessment": (
                    expected_adaptive_assessment_sha256
                )
            }
            or adaptive_stage.method != expected_adaptive_method
        ):
            raise ValueError("verification_certificate_adaptive_lineage_mismatch")
        expected_release_method = (
            "prospective-qualified-claim-release-v2"
            if self.release_assessment.assessment_version
            == "prospective-qualified-claim-release-v2"
            else "prospective-claim-release-v2"
        )
        expected_release_input_sha256 = hash_canonical(
            {
                "audit_candidates": self.audit_candidates,
                "audit_receipts": self.release_assessment.audit.resolution_receipts,
                "sequential_audit_state": self.sequential_audit_state,
                "budget_minutes": self.release_assessment.audit.budget,
                "complete_corpus_identity": self.complete_corpus_identity,
                "item_risk_scoring_receipt": self.item_risk_scoring_receipt,
                "adaptive_policy_context": self.adaptive_policy_context,
                "adaptive_calibration_bundle": self.adaptive_calibration_bundle,
                "adaptive_release_candidate": self.adaptive_release_candidate,
                "adaptive_prospective_assessment": (
                    self.adaptive_prospective_assessment
                ),
                "pipeline_sha256": self.release_assessment.pipeline_sha256,
                "production_stop_decision_sha256": decision.decision_sha256,
                "target": self.release_assessment.target,
            }
        )
        if (
            release_stage.input_sha256s
            != {"release_inputs": expected_release_input_sha256}
            or release_stage.output_sha256s
            != {"release_decision": self.release_assessment.decision_sha256}
            or release_stage.method != expected_release_method
        ):
            raise ValueError("verification_certificate_release_lineage_mismatch")
        payload = self.model_dump(mode="json", exclude={"certificate_sha256"})
        if hash_canonical(payload) != self.certificate_sha256:
            raise ValueError("verification_certificate_hash_mismatch")
        if self.status == "released" and self.reasons:
            raise ValueError("released_verification_certificate_cannot_have_reasons")
        if self.status == "abstained" and not self.reasons:
            raise ValueError("abstained_verification_certificate_requires_reason")
        return self


class CertificateArtifacts(ContractModel):
    """Paths and byte hashes written by :func:`write_certificate_artifacts`."""

    json_path: str
    json_sha256: str
    html_path: str
    html_sha256: str

    @field_validator("json_sha256", "html_sha256")
    @classmethod
    def validate_artifact_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("verification_artifact_sha256_invalid")
        return value


def freeze_verification_certificate(
    *,
    generated_at: datetime,
    status: Literal["released", "abstained"],
    reasons: list[str],
    claim_manifest: dict[str, Any],
    corpus: dict[str, Any],
    corpus_sha256: str,
    source_evidence_graph: EvidenceGraph,
    evidence_graph: EvidenceGraph,
    adapter_issues: list[dict[str, Any]],
    synthesis: dict[str, Any],
    counterfactual_reruns: list[dict[str, Any]],
    audit_candidates: list[dict[str, Any]],
    release_assessment: ClaimReleaseAssessment | QualifiedClaimReleaseAssessment,
    lineage: list[CertificateLineageStage],
    pipeline_verification: PipelineFingerprintVerification,
    complete_corpus_identity: CompleteCorpusIdentity,
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None,
    adaptive_calibration_bundle: AdaptiveCalibrationBundle | None,
    adaptive_release_candidate: ProspectiveAdaptiveReleaseCandidate | None,
    sequential_audit_state: SequentialVerificationState | None,
    production_stop_decision: ProductionStopDecision,
) -> VerificationCertificate:
    """Freeze and self-hash one complete verification result."""

    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("verification_certificate_generated_at_requires_timezone")
    generated_at_json = generated_at.isoformat()
    if generated_at_json.endswith("+00:00"):
        generated_at_json = f"{generated_at_json[:-6]}Z"
    claim_hash = hash_canonical(claim_manifest)
    source_graph_hash = hash_canonical(source_evidence_graph)
    graph_hash = hash_canonical(evidence_graph)
    synthesis_hash = hash_canonical(synthesis)
    if (adaptive_calibration_bundle is None) != (
        adaptive_release_candidate is None
    ):
        raise ValueError(
            "verification_certificate_adaptive_bundle_candidate_pair_required"
        )
    adaptive_policy_context = None
    adaptive_prospective_assessment = None
    if adaptive_calibration_bundle is not None:
        assert adaptive_release_candidate is not None
        adaptive_calibration_bundle = validate_adaptive_calibration_bundle_integrity(
            adaptive_calibration_bundle
        )
        adaptive_policy_context = next(
            (
                context
                for context in adaptive_calibration_bundle.development_freeze.policy_contexts
                if context.policy_arm_id == adaptive_release_candidate.policy_arm_id
            ),
            None,
        )
        if adaptive_policy_context is None:
            raise ValueError(
                "verification_certificate_adaptive_policy_context_missing"
            )
        adaptive_prospective_assessment = assess_adaptive_release_candidate(
            adaptive_release_candidate,
            adaptive_calibration_bundle,
        )
    run_identity = hash_canonical(
        {
            "claim_manifest_sha256": claim_hash,
            "corpus_sha256": corpus_sha256,
            "source_evidence_graph_sha256": source_graph_hash,
            "evidence_graph_sha256": graph_hash,
            "release_decision_sha256": release_assessment.decision_sha256,
            "pipeline_verification_sha256": pipeline_verification.verification_sha256,
            "production_stop_decision_sha256": (
                production_stop_decision.decision_sha256
            ),
            "complete_corpus_membership_sha256": (
                complete_corpus_identity.membership_sha256
            ),
            "item_risk_scoring_receipt_sha256": (
                None
                if item_risk_scoring_receipt is None
                else item_risk_scoring_receipt.receipt_sha256
            ),
            "adaptive_policy_context_sha256": (
                None
                if adaptive_policy_context is None
                else adaptive_policy_context.policy_context_sha256
            ),
            "adaptive_calibration_bundle_sha256": (
                None
                if adaptive_calibration_bundle is None
                else adaptive_calibration_bundle.bundle_sha256
            ),
            "adaptive_release_candidate_sha256": (
                None
                if adaptive_release_candidate is None
                else adaptive_release_candidate.candidate_sha256
            ),
            "adaptive_prospective_assessment_sha256": (
                None
                if adaptive_prospective_assessment is None
                else adaptive_prospective_assessment.assessment_sha256
            ),
        }
    )
    payload: dict[str, Any] = {
        "certificate_version": "literature-multiverse-verification-v5",
        "run_id": f"verify-{run_identity[:16]}",
        "generated_at": generated_at_json,
        "status": status,
        "reasons": sorted(set(reasons)),
        "claim_manifest": claim_manifest,
        "claim_manifest_sha256": claim_hash,
        "corpus": corpus,
        "corpus_sha256": corpus_sha256,
        "source_evidence_graph": source_evidence_graph,
        "source_evidence_graph_sha256": source_graph_hash,
        "evidence_graph": evidence_graph,
        "evidence_graph_sha256": graph_hash,
        "adapter_issues": adapter_issues,
        "synthesis": synthesis,
        "synthesis_sha256": synthesis_hash,
        "counterfactual_reruns": counterfactual_reruns,
        "audit_candidates": audit_candidates,
        "release_assessment": release_assessment,
        "pipeline_verification": pipeline_verification,
        "complete_corpus_identity": complete_corpus_identity,
        "item_risk_scoring_receipt": item_risk_scoring_receipt,
        "adaptive_policy_context": adaptive_policy_context,
        "adaptive_calibration_bundle": adaptive_calibration_bundle,
        "adaptive_release_candidate": adaptive_release_candidate,
        "adaptive_prospective_assessment": adaptive_prospective_assessment,
        "sequential_audit_state": sequential_audit_state,
        "production_stop_decision": production_stop_decision,
        "lineage": lineage,
        "interpretation": (
            "literature-support verification under the declared corpus; not scientific truth"
        ),
    }
    return VerificationCertificate.model_validate(
        {**payload, "certificate_sha256": hash_canonical(payload)}
    )


_V8_COMPOSITION_STAGE = "corpus_pipeline_composition_external_replay"
_V8_PIPELINE_IDENTITY_BASIS = (
    "externally_replayed_extraction_verifier_composition-v1"
)


def _verification_v8_shadow_corpus(
    *,
    corpus: dict[str, Any],
    receipt: CorpusPipelineCompositionExternalReplayReceiptV1,
) -> dict[str, Any]:
    """Build the v5-validation view without overwriting normative v8 lineage."""

    shadow = deepcopy(corpus)
    metadata = shadow.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("verification_v8_corpus_metadata_invalid")
    extraction_pipeline_sha256 = receipt.composition_join.extraction_pipeline_sha256
    if metadata.get("pipeline_fingerprint_sha256") != extraction_pipeline_sha256:
        raise ValueError("verification_v8_corpus_extraction_pipeline_alias_mismatch")
    metadata["pipeline_fingerprint_sha256"] = receipt.composed_pipeline_sha256
    return shadow


def _verification_v8_composition_stage(
    *,
    receipt: CorpusPipelineCompositionExternalReplayReceiptV1,
    complete_corpus_identity_v3: CompleteCorpusIdentityV3,
    manifest_corpus_policy_binding_v3: ManifestCorpusPolicyBindingV3,
) -> CertificateLineageStage:
    return CertificateLineageStage(
        stage=_V8_COMPOSITION_STAGE,
        input_sha256s=dict(
            sorted(
                {
                    "corpus_source": receipt.corpus_source_sha256,
                    "extraction_pipeline_verification": (
                        receipt.extraction_pipeline_verification_sha256
                    ),
                    "grounding_package": receipt.grounding_package_sha256,
                    "hosted_bridge_receipt": receipt.hosted_bridge_receipt_sha256,
                    "verifier_core_pipeline_verification": (
                        receipt.verifier_core_pipeline_verification_sha256
                    ),
                }.items()
            )
        ),
        output_sha256s=dict(
            sorted(
                {
                    "complete_corpus_identity_v3": (
                        complete_corpus_identity_v3.membership_composition_sha256
                    ),
                    "composed_pipeline_verification": (
                        receipt.composed_pipeline_verification_sha256
                    ),
                    "external_replay_receipt": receipt.receipt_sha256,
                    "manifest_corpus_policy_binding_v3": (
                        manifest_corpus_policy_binding_v3
                        .manifest_corpus_policy_binding_sha256
                    ),
                }.items()
            )
        ),
        method="full-current-byte-package-bridge-loader-replay-v1",
    )


def _certificate_manifest_policy_sha256(claim_manifest: dict[str, Any]) -> str:
    """Mirror the verifier's closed manifest-context policy identity."""

    return hash_canonical(
        {
            "policy_context_version": "verification-manifest-context-v1",
            "claim_manifest": claim_manifest,
        }
    )


def _require_verification_v8_receipt_corpus_aliases(
    *,
    corpus: dict[str, Any],
    source_evidence_graph: EvidenceGraph,
    corpus_sha256: str,
    receipt: CorpusPipelineCompositionExternalReplayReceiptV1,
) -> None:
    metadata = corpus.get("metadata")
    ingress = receipt.composition_join.corpus_ingress
    if not isinstance(metadata, dict):
        raise ValueError("verification_v8_corpus_metadata_invalid")
    aliases = {
        "corpus_id": (corpus.get("corpus_id"), ingress.corpus_id),
        "corpus_source_sha256": (corpus_sha256, receipt.corpus_source_sha256),
        "source_payload_sha256": (corpus.get("source_sha256"), corpus_sha256),
        "grounding_package_sha256": (
            metadata.get("grounding_package_sha256"),
            receipt.grounding_package_sha256,
        ),
        "typed_corpus_sha256": (
            metadata.get("typed_evidence_corpus_sha256"),
            receipt.typed_corpus_sha256,
        ),
        "extraction_pipeline_sha256": (
            metadata.get("pipeline_fingerprint_sha256"),
            ingress.extraction_pipeline_sha256,
        ),
        "source_manifest_sha256": (
            metadata.get("source_manifest_sha256"),
            ingress.source_manifest_sha256,
        ),
        "grounding_replay_sha256": (
            metadata.get("grounding_replay_sha256"),
            receipt.grounding_replay_sha256,
        ),
        "effective_graph_sha256": (
            hash_canonical(source_evidence_graph),
            receipt.effective_graph_sha256,
        ),
    }
    for field, (observed, expected) in aliases.items():
        if observed != expected:
            raise ValueError(f"verification_v8_{field}_alias_mismatch")
    if corpus.get("pipeline_identity_basis") != _V8_PIPELINE_IDENTITY_BASIS:
        raise ValueError("verification_v8_pipeline_identity_basis_invalid")


class VerificationCertificateV8(VerificationCertificate):
    """Join-aware ordinary certificate using a composed calibration pipeline.

    V8 retains all ordinary v5 scientific, audit, calibration, and stopping-rule
    checks.  Its only semantic change is that a fully externally replayed
    extraction/verifier composition may replace the legacy direct pipeline-equality
    blocker.  No other adapter issue is removed or weakened.
    """

    certificate_version: Literal["literature-multiverse-verification-v8"] = (
        "literature-multiverse-verification-v8"
    )
    run_id: Annotated[str, Field(pattern=r"^verify-v8-[0-9a-f]{16}$")]
    composition_external_replay_receipt: (
        CorpusPipelineCompositionExternalReplayReceiptV1
    )
    composition_external_replay_receipt_sha256: str
    complete_corpus_identity_v3: CompleteCorpusIdentityV3
    complete_corpus_membership_v3_sha256: str
    manifest_corpus_policy_binding_v3: ManifestCorpusPolicyBindingV3
    manifest_corpus_policy_binding_v3_sha256: str
    cleared_adapter_issue_code: Literal["corpus_pipeline_identity_mismatch"] = (
        "corpus_pipeline_identity_mismatch"
    )
    v5_common_contract_replay_sha256: str

    @field_validator(
        "composition_external_replay_receipt_sha256",
        "complete_corpus_membership_v3_sha256",
        "manifest_corpus_policy_binding_v3_sha256",
        "v5_common_contract_replay_sha256",
    )
    @classmethod
    def validate_v8_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("verification_v8_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_integrity(self) -> VerificationCertificateV8:
        try:
            receipt = CorpusPipelineCompositionExternalReplayReceiptV1.model_validate(
                self.composition_external_replay_receipt.model_dump(mode="json")
            )
            complete_v3 = validate_complete_corpus_identity_v3(
                self.complete_corpus_identity_v3
            )
            binding_v3 = validate_manifest_corpus_policy_binding_v3(
                self.manifest_corpus_policy_binding_v3
            )
        except (AttributeError, ValueError) as exc:
            raise ValueError("verification_v8_composition_contract_invalid") from exc
        if (
            self.composition_external_replay_receipt_sha256
            != receipt.receipt_sha256
            or self.complete_corpus_membership_v3_sha256
            != complete_v3.membership_composition_sha256
            or self.manifest_corpus_policy_binding_v3_sha256
            != binding_v3.manifest_corpus_policy_binding_sha256
            or binding_v3.complete_corpus_identity_v3 != complete_v3
            or binding_v3.claim_manifest_sha256 != self.claim_manifest_sha256
            or binding_v3.policy_sha256
            != _certificate_manifest_policy_sha256(self.claim_manifest)
            or complete_v3.external_replay_receipt != receipt
            or complete_v3.complete_corpus_identity_v2.complete_corpus_membership_v1
            != self.complete_corpus_identity
            or self.pipeline_verification != receipt.composed_pipeline_verification
            or self.release_assessment.pipeline_sha256
            != receipt.release_pipeline_sha256
        ):
            raise ValueError("verification_v8_composition_alias_mismatch")
        _require_verification_v8_receipt_corpus_aliases(
            corpus=self.corpus,
            source_evidence_graph=self.source_evidence_graph,
            corpus_sha256=self.corpus_sha256,
            receipt=receipt,
        )
        mismatch_reason = "adapter:corpus_pipeline_identity_mismatch"
        if mismatch_reason in self.reasons or any(
            issue.get("code") == self.cleared_adapter_issue_code
            for issue in self.adapter_issues
        ):
            raise ValueError("verification_v8_legacy_pipeline_blocker_not_cleared")
        if len(self.lineage) != 9 or self.lineage[-1] != _verification_v8_composition_stage(
            receipt=receipt,
            complete_corpus_identity_v3=complete_v3,
            manifest_corpus_policy_binding_v3=binding_v3,
        ):
            raise ValueError("verification_v8_composition_lineage_mismatch")

        shadow_corpus = _verification_v8_shadow_corpus(
            corpus=self.corpus,
            receipt=receipt,
        )
        try:
            shadow = freeze_verification_certificate(
                generated_at=self.generated_at,
                status=self.status,
                reasons=self.reasons,
                claim_manifest=self.claim_manifest,
                corpus=shadow_corpus,
                corpus_sha256=self.corpus_sha256,
                source_evidence_graph=self.source_evidence_graph,
                evidence_graph=self.evidence_graph,
                adapter_issues=self.adapter_issues,
                synthesis=self.synthesis,
                counterfactual_reruns=self.counterfactual_reruns,
                audit_candidates=self.audit_candidates,
                release_assessment=self.release_assessment,
                lineage=self.lineage[:-1],
                pipeline_verification=self.pipeline_verification,
                complete_corpus_identity=self.complete_corpus_identity,
                item_risk_scoring_receipt=self.item_risk_scoring_receipt,
                adaptive_calibration_bundle=self.adaptive_calibration_bundle,
                adaptive_release_candidate=self.adaptive_release_candidate,
                sequential_audit_state=self.sequential_audit_state,
                production_stop_decision=self.production_stop_decision,
            )
        except ValueError as exc:
            raise ValueError("verification_v8_v5_common_contract_replay_failed") from exc
        if self.v5_common_contract_replay_sha256 != shadow.certificate_sha256:
            raise ValueError("verification_v8_common_contract_replay_hash_mismatch")
        v8_exclusions = {
            "certificate_version",
            "run_id",
            "corpus",
            "lineage",
            "certificate_sha256",
            "composition_external_replay_receipt",
            "composition_external_replay_receipt_sha256",
            "complete_corpus_identity_v3",
            "complete_corpus_membership_v3_sha256",
            "manifest_corpus_policy_binding_v3",
            "manifest_corpus_policy_binding_v3_sha256",
            "cleared_adapter_issue_code",
            "v5_common_contract_replay_sha256",
        }
        shadow_exclusions = {
            "certificate_version",
            "run_id",
            "corpus",
            "lineage",
            "certificate_sha256",
        }
        if self.model_dump(mode="json", exclude=v8_exclusions) != shadow.model_dump(
            mode="json", exclude=shadow_exclusions
        ):
            raise ValueError("verification_v8_common_contract_projection_mismatch")
        run_identity = hash_canonical(
            {
                "v5_common_contract_replay_sha256": shadow.certificate_sha256,
                "composition_external_replay_receipt_sha256": receipt.receipt_sha256,
                "complete_corpus_membership_v3_sha256": (
                    complete_v3.membership_composition_sha256
                ),
                "manifest_corpus_policy_binding_v3_sha256": (
                    binding_v3.manifest_corpus_policy_binding_sha256
                ),
            }
        )
        if self.run_id != f"verify-v8-{run_identity[:16]}":
            raise ValueError("verification_v8_run_identity_mismatch")
        payload = self.model_dump(mode="json", exclude={"certificate_sha256"})
        if self.certificate_sha256 != hash_canonical(payload):
            raise ValueError("verification_v8_certificate_hash_mismatch")
        return self


def freeze_verification_certificate_v8(
    *,
    generated_at: datetime,
    status: Literal["released", "abstained"],
    reasons: list[str],
    claim_manifest: dict[str, Any],
    corpus: dict[str, Any],
    corpus_sha256: str,
    source_evidence_graph: EvidenceGraph,
    evidence_graph: EvidenceGraph,
    adapter_issues: list[dict[str, Any]],
    synthesis: dict[str, Any],
    counterfactual_reruns: list[dict[str, Any]],
    audit_candidates: list[dict[str, Any]],
    release_assessment: ClaimReleaseAssessment | QualifiedClaimReleaseAssessment,
    lineage: list[CertificateLineageStage],
    pipeline_verification: PipelineFingerprintVerification,
    complete_corpus_identity: CompleteCorpusIdentity,
    complete_corpus_identity_v3: CompleteCorpusIdentityV3,
    composition_external_replay_receipt: (
        CorpusPipelineCompositionExternalReplayReceiptV1
    ),
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None,
    adaptive_calibration_bundle: AdaptiveCalibrationBundle | None,
    adaptive_release_candidate: ProspectiveAdaptiveReleaseCandidate | None,
    sequential_audit_state: SequentialVerificationState | None,
    production_stop_decision: ProductionStopDecision,
) -> VerificationCertificateV8:
    """Freeze v8 after replaying the complete ordinary v5 contract projection."""

    receipt = CorpusPipelineCompositionExternalReplayReceiptV1.model_validate(
        composition_external_replay_receipt.model_dump(mode="json")
    )
    complete_v3 = validate_complete_corpus_identity_v3(complete_corpus_identity_v3)
    if (
        complete_v3.external_replay_receipt != receipt
        or complete_v3.complete_corpus_identity_v2.complete_corpus_membership_v1
        != complete_corpus_identity
    ):
        raise ValueError("verification_v8_complete_corpus_input_mismatch")
    binding_v3 = freeze_manifest_corpus_policy_binding_v3(
        claim_manifest_sha256=hash_canonical(claim_manifest),
        complete_corpus_identity_v3=complete_v3,
        policy_sha256=_certificate_manifest_policy_sha256(claim_manifest),
    )
    if any(stage.stage == _V8_COMPOSITION_STAGE for stage in lineage):
        raise ValueError("verification_v8_duplicate_composition_lineage_stage")
    if any(
        issue.get("code") == "corpus_pipeline_identity_mismatch"
        for issue in adapter_issues
    ) or "adapter:corpus_pipeline_identity_mismatch" in reasons:
        raise ValueError("verification_v8_pipeline_mismatch_blocker_must_be_removed")
    shadow = freeze_verification_certificate(
        generated_at=generated_at,
        status=status,
        reasons=reasons,
        claim_manifest=claim_manifest,
        corpus=_verification_v8_shadow_corpus(corpus=corpus, receipt=receipt),
        corpus_sha256=corpus_sha256,
        source_evidence_graph=source_evidence_graph,
        evidence_graph=evidence_graph,
        adapter_issues=adapter_issues,
        synthesis=synthesis,
        counterfactual_reruns=counterfactual_reruns,
        audit_candidates=audit_candidates,
        release_assessment=release_assessment,
        lineage=lineage,
        pipeline_verification=pipeline_verification,
        complete_corpus_identity=complete_corpus_identity,
        item_risk_scoring_receipt=item_risk_scoring_receipt,
        adaptive_calibration_bundle=adaptive_calibration_bundle,
        adaptive_release_candidate=adaptive_release_candidate,
        sequential_audit_state=sequential_audit_state,
        production_stop_decision=production_stop_decision,
    )
    payload = shadow.model_dump(mode="json", exclude={"certificate_sha256"})
    payload.update(
        {
            "certificate_version": "literature-multiverse-verification-v8",
            "corpus": corpus,
            "lineage": [
                *lineage,
                _verification_v8_composition_stage(
                    receipt=receipt,
                    complete_corpus_identity_v3=complete_v3,
                    manifest_corpus_policy_binding_v3=binding_v3,
                ),
            ],
            "composition_external_replay_receipt": receipt,
            "composition_external_replay_receipt_sha256": receipt.receipt_sha256,
            "complete_corpus_identity_v3": complete_v3,
            "complete_corpus_membership_v3_sha256": (
                complete_v3.membership_composition_sha256
            ),
            "manifest_corpus_policy_binding_v3": binding_v3,
            "manifest_corpus_policy_binding_v3_sha256": (
                binding_v3.manifest_corpus_policy_binding_sha256
            ),
            "cleared_adapter_issue_code": "corpus_pipeline_identity_mismatch",
            "v5_common_contract_replay_sha256": shadow.certificate_sha256,
        }
    )
    run_identity = hash_canonical(
        {
            "v5_common_contract_replay_sha256": shadow.certificate_sha256,
            "composition_external_replay_receipt_sha256": receipt.receipt_sha256,
            "complete_corpus_membership_v3_sha256": (
                complete_v3.membership_composition_sha256
            ),
            "manifest_corpus_policy_binding_v3_sha256": (
                binding_v3.manifest_corpus_policy_binding_sha256
            ),
        }
    )
    payload["run_id"] = f"verify-v8-{run_identity[:16]}"
    return VerificationCertificateV8.model_validate(
        {**payload, "certificate_sha256": hash_canonical(payload)}
    )


class ConditionProductionStopDecisionV2(ContractModel):
    """Outcome-free scheduler decision made before terminal outcomes are opened."""

    decision_version: Literal["condition-production-stop-decision-v2"] = (
        "condition-production-stop-decision-v2"
    )
    stopping_rule: Literal[
        "invoke_condition_gate_at_first_nonconfirmation_eligible_or_scheduler_terminal_state"
    ] = (
        "invoke_condition_gate_at_first_nonconfirmation_eligible_or_scheduler_terminal_state"
    )
    evaluated_state: SequentialVerificationState
    release_assessment: ConditionClaimReleaseAssessmentV1
    blocking_adapter_reasons: list[str]
    outcome: Literal[
        "condition_gate_ready",
        "selected_next_action",
        "active_action_in_progress",
        "no_feasible_action",
        "condition_context_blocked",
    ]
    selection_result: SequentialSelectionResult | None = None
    condition_gate_invocation_proof: ConditionGateInvocationProofV2 | None = None
    decision_sha256: str

    @field_validator("decision_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("condition_stop_decision_sha256_invalid")
        return value

    @field_validator("blocking_adapter_reasons")
    @classmethod
    def validate_blockers(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(
            not item.startswith("adapter:") for item in value
        ):
            raise ValueError("condition_stop_adapter_blockers_invalid")
        return value

    @model_validator(mode="after")
    def validate_decision(self) -> ConditionProductionStopDecisionV2:
        state = self.evaluated_state
        assessment = self.release_assessment
        if assessment.audit.sequential_state_sha256 != state.state_sha256:
            raise ValueError("condition_stop_assessment_state_mismatch")
        active = state.session.active_action is not None
        feasible = sorted(
            candidate.item_id
            for candidate in state.candidates
            if candidate.item_id not in set(state.session.resolved_item_ids)
            and candidate.eligible
            and candidate.estimated_cost <= state.session.remaining_budget + 1e-9
        )
        if self.outcome == "condition_gate_ready":
            proof = self.condition_gate_invocation_proof
            if (
                active
                or self.selection_result is not None
                or proof is None
                or proof.terminal_preselection_state.scheduler_state_sha256
                != state.state_sha256
                or proof.condition_projection
                != assessment.condition_calibration_projection
                or not assessment.terminal_gate_deferred
            ):
                raise ValueError("condition_stop_gate_ready_mismatch")
        elif self.outcome == "selected_next_action":
            result = self.selection_result
            if (
                active
                or result is None
                or self.condition_gate_invocation_proof is not None
                or result.previous_state_sha256 != state.state_sha256
            ):
                raise ValueError("condition_stop_selection_mismatch")
        elif self.outcome == "active_action_in_progress":
            if (
                not active
                or self.selection_result is not None
                or self.condition_gate_invocation_proof is not None
            ):
                raise ValueError("condition_stop_active_action_mismatch")
        elif self.outcome == "no_feasible_action":
            if (
                active
                or feasible
                or self.selection_result is not None
                or self.condition_gate_invocation_proof is not None
            ):
                raise ValueError("condition_stop_no_feasible_action_mismatch")
        elif (
            active
            or self.selection_result is not None
            or self.condition_gate_invocation_proof is not None
        ):
            raise ValueError("condition_stop_context_blocked_mismatch")
        payload = self.model_dump(mode="json", exclude={"decision_sha256"})
        if hash_canonical(payload) != self.decision_sha256:
            raise ValueError("condition_stop_decision_hash_mismatch")
        return self


def freeze_condition_production_stop_decision_v2(
    *,
    evaluated_state: SequentialVerificationState,
    release_assessment: ConditionClaimReleaseAssessmentV1,
    blocking_adapter_reasons: list[str],
    outcome: Literal[
        "condition_gate_ready",
        "selected_next_action",
        "active_action_in_progress",
        "no_feasible_action",
        "condition_context_blocked",
    ],
    selection_result: SequentialSelectionResult | None = None,
    condition_gate_invocation_proof: ConditionGateInvocationProofV2 | None = None,
) -> ConditionProductionStopDecisionV2:
    payload: dict[str, Any] = {
        "decision_version": "condition-production-stop-decision-v2",
        "stopping_rule": (
            "invoke_condition_gate_at_first_nonconfirmation_eligible_or_"
            "scheduler_terminal_state"
        ),
        "evaluated_state": evaluated_state,
        "release_assessment": release_assessment,
        "blocking_adapter_reasons": sorted(set(blocking_adapter_reasons)),
        "outcome": outcome,
        "selection_result": selection_result,
        "condition_gate_invocation_proof": condition_gate_invocation_proof,
    }
    return ConditionProductionStopDecisionV2.model_validate(
        {**payload, "decision_sha256": hash_canonical(payload)}
    )


class ConditionCalibrationCollectionDecisionV1(ContractModel):
    """Outcome-free scheduler decision used only to collect calibration data."""

    decision_version: Literal["condition-calibration-collection-decision-v1"] = (
        "condition-calibration-collection-decision-v1"
    )
    evaluated_state: SequentialVerificationState
    terminal_preselection_state: AdaptivePreselectionState | None
    outcome: Literal[
        "selected_next_action",
        "active_action_in_progress",
        "condition_gate_ready",
        "condition_context_blocked",
        "no_feasible_action",
    ]
    selection_result: SequentialSelectionResult | None
    condition_gate_invocation_proof: ConditionGateInvocationProofV2 | None
    reasons: list[str]
    decision_sha256: str

    @field_validator("decision_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("condition_collection_decision_sha256_invalid")
        return value

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, value: list[str]) -> list[str]:
        if not value or value != sorted(set(value)) or any(not item for item in value):
            raise ValueError("condition_collection_decision_reasons_invalid")
        return value

    @model_validator(mode="after")
    def validate_decision(self) -> ConditionCalibrationCollectionDecisionV1:
        try:
            state = resume_sequential_verification_state(self.evaluated_state)
        except SequentialVerificationContractError as exc:
            raise ValueError("condition_collection_decision_state_invalid") from exc
        preselection = self.terminal_preselection_state
        invocation = self.condition_gate_invocation_proof
        selection = self.selection_result
        if self.outcome == "condition_gate_ready":
            if (
                state.session.active_action is not None
                or preselection is None
                or not preselection.non_calibration_gates_passed
                or invocation is None
                or selection is not None
                or invocation.terminal_preselection_state != preselection
            ):
                raise ValueError("condition_collection_gate_ready_contract_mismatch")
        elif self.outcome == "selected_next_action":
            if (
                state.session.active_action is not None
                or preselection is None
                or preselection.non_calibration_gates_passed
                or invocation is not None
                or selection is None
                or selection.previous_state_sha256 != state.state_sha256
            ):
                raise ValueError("condition_collection_selection_contract_mismatch")
        elif self.outcome == "active_action_in_progress":
            if (
                state.session.active_action is None
                or preselection is not None
                or invocation is not None
                or selection is not None
            ):
                raise ValueError("condition_collection_active_action_contract_mismatch")
        elif (
            selection is not None
            or invocation is not None
            or state.session.active_action is not None
        ):
            raise ValueError("condition_collection_terminal_contract_mismatch")
        payload = self.model_dump(mode="json", exclude={"decision_sha256"})
        if hash_canonical(payload) != self.decision_sha256:
            raise ValueError("condition_collection_decision_hash_mismatch")
        return self


def freeze_condition_calibration_collection_decision_v1(
    *,
    evaluated_state: SequentialVerificationState,
    terminal_preselection_state: AdaptivePreselectionState | None,
    outcome: Literal[
        "selected_next_action",
        "active_action_in_progress",
        "condition_gate_ready",
        "condition_context_blocked",
        "no_feasible_action",
    ],
    selection_result: SequentialSelectionResult | None,
    condition_gate_invocation_proof: ConditionGateInvocationProofV2 | None,
    reasons: list[str],
) -> ConditionCalibrationCollectionDecisionV1:
    payload: dict[str, Any] = {
        "decision_version": "condition-calibration-collection-decision-v1",
        "evaluated_state": evaluated_state,
        "terminal_preselection_state": terminal_preselection_state,
        "outcome": outcome,
        "selection_result": selection_result,
        "condition_gate_invocation_proof": condition_gate_invocation_proof,
        "reasons": sorted(set(reasons)),
    }
    return ConditionCalibrationCollectionDecisionV1.model_validate(
        {**payload, "decision_sha256": hash_canonical(payload)}
    )


class ConditionCalibrationCollectionSourceV1(ContractModel):
    """Always-abstained, pre-bundle source for confirmation calibration.

    The object contains enough frozen science and scheduler state to replay the
    policy-visible trajectory.  It contains no held-out confirmation assessment,
    reference verdict, calibration bundle, release qualification, or release result.
    """

    certificate_version: Literal[
        "condition-calibration-collection-source-v1"
    ] = "condition-calibration-collection-source-v1"
    run_id: Annotated[str, Field(pattern=r"^condition-collection-[0-9a-f]{16}$")]
    generated_at: datetime
    status: Literal["abstained"] = "abstained"
    collection_split: Literal["development", "calibration"]
    reasons: list[str]
    claim_manifest: dict[str, Any]
    claim_manifest_sha256: str
    corpus: dict[str, Any]
    corpus_sha256: str
    source_evidence_graph: EvidenceGraph
    source_evidence_graph_sha256: str
    current_full_evidence_graph: EvidenceGraph
    current_full_evidence_graph_sha256: str
    development_evidence_graph: EvidenceGraph
    development_evidence_graph_sha256: str
    adapter_issues: list[dict[str, Any]]
    synthesis: dict[str, Any]
    synthesis_sha256: str
    audit_candidates: list[dict[str, Any]]
    pipeline_verification: PipelineFingerprintVerification
    complete_corpus_identity: CompleteCorpusIdentity
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None
    condition_plan: ConditionConfirmationPlanV1
    condition_frozen_model: ConditionConfirmationFrozenModelV1
    condition_calibration_projection: ConditionCalibrationProjectionV1
    condition_target_semantics: AdaptiveTargetSemanticsBindingV2
    condition_independence_identity: StrongIndependenceIdentityV1
    adaptive_policy_context: AdaptivePolicyContext
    online_preselection_states: list[AdaptivePreselectionState]
    policy_visible_question_trajectory: PolicyVisibleQuestionTrajectoryV2 | None
    sequential_audit_state: SequentialVerificationState
    collection_decision: ConditionCalibrationCollectionDecisionV1
    lineage: list[CertificateLineageStage]
    confirmation_partition_unopened_by_online_policy: Literal[True] = True
    condition_assessment_unopened: Literal[True] = True
    reference_labels_unopened: Literal[True] = True
    adaptive_calibration_bundle_unavailable: Literal[True] = True
    collection_source_sha256: str

    @field_validator(
        "claim_manifest_sha256",
        "corpus_sha256",
        "source_evidence_graph_sha256",
        "current_full_evidence_graph_sha256",
        "development_evidence_graph_sha256",
        "synthesis_sha256",
        "collection_source_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("condition_collection_source_sha256_invalid")
        return value

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, value: list[str]) -> list[str]:
        if not value or value != sorted(set(value)) or any(not item for item in value):
            raise ValueError("condition_collection_source_reasons_invalid")
        return value

    @model_validator(mode="after")
    def validate_source(self) -> ConditionCalibrationCollectionSourceV1:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("condition_collection_generated_at_requires_timezone")
        if hash_canonical(self.claim_manifest) != self.claim_manifest_sha256:
            raise ValueError("condition_collection_manifest_hash_mismatch")
        if self.corpus.get("source_sha256") != self.corpus_sha256:
            raise ValueError("condition_collection_corpus_hash_mismatch")
        graph_checks = (
            (self.source_evidence_graph, self.source_evidence_graph_sha256),
            (self.current_full_evidence_graph, self.current_full_evidence_graph_sha256),
            (self.development_evidence_graph, self.development_evidence_graph_sha256),
        )
        if any(hash_canonical(graph) != digest for graph, digest in graph_checks):
            raise ValueError("condition_collection_graph_hash_mismatch")
        if hash_canonical(self.synthesis) != self.synthesis_sha256:
            raise ValueError("condition_collection_synthesis_hash_mismatch")
        pipeline_sha256 = self.pipeline_verification.computed_pipeline_sha256
        plan = self.condition_plan
        projection = self.condition_calibration_projection
        manifest_question_id = self.claim_manifest.get("question_id")
        manifest_population_id = self.claim_manifest.get("population_id")
        manifest_domain = self.claim_manifest.get("domain")
        manifest_protocol = self.claim_manifest.get("protocol")
        manifest_cutoff = (
            manifest_protocol.get("corpus_cutoff")
            if isinstance(manifest_protocol, dict)
            else None
        )
        manifest_target = self.claim_manifest.get("global_condition_target")
        manifest_target_sha256 = (
            manifest_target.get("target_sha256")
            if isinstance(manifest_target, dict)
            else None
        )
        if (
            self.pipeline_verification.status != "matched"
            or pipeline_sha256 is None
            or plan.pipeline_sha256 != pipeline_sha256
            or plan.full_graph_sha256 != self.current_full_evidence_graph_sha256
            or plan.development_graph_sha256
            != self.development_evidence_graph_sha256
            or projection.plan_sha256 != plan.plan_sha256
            or projection.full_graph_sha256 != plan.full_graph_sha256
            or projection.development_graph_sha256 != plan.development_graph_sha256
            or projection.pipeline_sha256 != pipeline_sha256
            or projection.target_semantics != self.condition_target_semantics
            or projection.independence_identity_sha256
            != self.condition_independence_identity.independence_identity_sha256
            or self.adaptive_policy_context.pipeline_sha256 != pipeline_sha256
            or self.claim_manifest.get("claim_manifest_version") != "3"
            or manifest_question_id != projection.question_id
            or manifest_question_id != self.condition_target_semantics.question_id
            or manifest_population_id != self.adaptive_policy_context.population_id
            or manifest_target_sha256 != projection.condition_target_sha256
            or self.complete_corpus_identity.corpus_id != self.corpus.get("corpus_id")
            or self.complete_corpus_identity.corpus_source_sha256 != self.corpus_sha256
            or self.complete_corpus_identity.corpus_cutoff != manifest_cutoff
            or projection.corpus_snapshot_sha256
            != self.complete_corpus_identity.membership_sha256
            or projection.corpus_cutoff != manifest_cutoff
        ):
            raise ValueError("condition_collection_scientific_context_mismatch")
        try:
            validate_condition_confirmation_model(
                plan=plan,
                development_graph=self.development_evidence_graph,
                model=self.condition_frozen_model,
                current_pipeline_sha256=pipeline_sha256,
            )
            state = resume_sequential_verification_state(self.sequential_audit_state)
        except (ConditionConfirmationError, SequentialVerificationContractError) as exc:
            raise ValueError("condition_collection_replay_failed") from exc
        decision = self.collection_decision
        expected_state = (
            decision.selection_result.state
            if decision.outcome == "selected_next_action"
            and decision.selection_result is not None
            else decision.evaluated_state
        )
        if state != expected_state or state.graph != self.development_evidence_graph:
            raise ValueError("condition_collection_sequential_state_mismatch")
        gate_ready = decision.outcome == "condition_gate_ready"
        complete_trajectory = decision.outcome in {
            "condition_gate_ready",
            "condition_context_blocked",
            "no_feasible_action",
        }
        visible = self.policy_visible_question_trajectory
        if complete_trajectory != (visible is not None):
            raise ValueError("condition_collection_visible_trajectory_presence_mismatch")
        if complete_trajectory:
            assert visible is not None
            invocation = decision.condition_gate_invocation_proof
            if gate_ready and invocation is None:
                raise ValueError("condition_collection_gate_ready_invocation_missing")
            selected_arms = [
                arm
                for arm in visible.arms
                if arm.base_arm.policy_arm_id
                == self.adaptive_policy_context.policy_arm_id
            ]
            if (
                visible.base_visible.question_id != projection.question_id
                or visible.base_visible.split != self.collection_split
                or visible.base_visible.population_id != manifest_population_id
                or visible.base_visible.domain != manifest_domain
                or visible.base_visible.corpus != self.complete_corpus_identity
                or visible.target_semantics != self.condition_target_semantics
                or visible.independence_identity
                != self.condition_independence_identity
                or len(selected_arms) != 1
                or selected_arms[0].base_arm.policy_context_sha256
                != self.adaptive_policy_context.policy_context_sha256
                or selected_arms[0].base_arm.states
                != self.online_preselection_states
                or selected_arms[0].condition_gate_invocation_proof != invocation
                or selected_arms[0].terminal_condition_projection != projection
            ):
                raise ValueError("condition_collection_visible_trajectory_mismatch")
        payload = self.model_dump(mode="json", exclude={"collection_source_sha256"})
        if hash_canonical(payload) != self.collection_source_sha256:
            raise ValueError("condition_collection_source_hash_mismatch")
        return self

    @property
    def question_id(self) -> str:
        return self.condition_calibration_projection.question_id

    @property
    def policy_arm_id(self) -> str:
        return self.adaptive_policy_context.policy_arm_id

    @property
    def collection_source_decision_sha256(self) -> str:
        return self.collection_decision.decision_sha256


def freeze_condition_calibration_collection_source_v1(
    *,
    generated_at: datetime,
    collection_split: Literal["development", "calibration"],
    reasons: list[str],
    claim_manifest: dict[str, Any],
    corpus: dict[str, Any],
    corpus_sha256: str,
    source_evidence_graph: EvidenceGraph,
    current_full_evidence_graph: EvidenceGraph,
    development_evidence_graph: EvidenceGraph,
    adapter_issues: list[dict[str, Any]],
    synthesis: dict[str, Any],
    audit_candidates: list[dict[str, Any]],
    pipeline_verification: PipelineFingerprintVerification,
    complete_corpus_identity: CompleteCorpusIdentity,
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None,
    condition_plan: ConditionConfirmationPlanV1,
    condition_frozen_model: ConditionConfirmationFrozenModelV1,
    condition_calibration_projection: ConditionCalibrationProjectionV1,
    condition_target_semantics: AdaptiveTargetSemanticsBindingV2,
    condition_independence_identity: AdaptiveIndependenceIdentityV2,
    adaptive_policy_context: AdaptivePolicyContext,
    online_preselection_states: list[AdaptivePreselectionState],
    policy_visible_question_trajectory: PolicyVisibleQuestionTrajectoryV2 | None,
    sequential_audit_state: SequentialVerificationState,
    collection_decision: ConditionCalibrationCollectionDecisionV1,
    lineage: list[CertificateLineageStage],
) -> ConditionCalibrationCollectionSourceV1:
    """Freeze one outcome-free, never-releasable arm-level collection source."""

    claim_manifest_sha256 = hash_canonical(claim_manifest)
    source_graph_sha256 = hash_canonical(source_evidence_graph)
    current_full_sha256 = hash_canonical(current_full_evidence_graph)
    development_sha256 = hash_canonical(development_evidence_graph)
    synthesis_sha256 = hash_canonical(synthesis)
    run_identity = hash_canonical(
        {
            "claim_manifest_sha256": claim_manifest_sha256,
            "collection_split": collection_split,
            "condition_projection_sha256": (
                condition_calibration_projection.projection_sha256
            ),
            "decision_sha256": collection_decision.decision_sha256,
            "policy_context_sha256": adaptive_policy_context.policy_context_sha256,
            "visible_trajectory_sha256": (
                None
                if policy_visible_question_trajectory is None
                else policy_visible_question_trajectory.trajectory_sha256
            ),
        }
    )
    payload: dict[str, Any] = {
        "certificate_version": "condition-calibration-collection-source-v1",
        "run_id": f"condition-collection-{run_identity[:16]}",
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "status": "abstained",
        "collection_split": collection_split,
        "reasons": sorted(set(reasons)),
        "claim_manifest": claim_manifest,
        "claim_manifest_sha256": claim_manifest_sha256,
        "corpus": corpus,
        "corpus_sha256": corpus_sha256,
        "source_evidence_graph": source_evidence_graph,
        "source_evidence_graph_sha256": source_graph_sha256,
        "current_full_evidence_graph": current_full_evidence_graph,
        "current_full_evidence_graph_sha256": current_full_sha256,
        "development_evidence_graph": development_evidence_graph,
        "development_evidence_graph_sha256": development_sha256,
        "adapter_issues": adapter_issues,
        "synthesis": synthesis,
        "synthesis_sha256": synthesis_sha256,
        "audit_candidates": audit_candidates,
        "pipeline_verification": pipeline_verification,
        "complete_corpus_identity": complete_corpus_identity,
        "item_risk_scoring_receipt": item_risk_scoring_receipt,
        "condition_plan": condition_plan,
        "condition_frozen_model": condition_frozen_model,
        "condition_calibration_projection": condition_calibration_projection,
        "condition_target_semantics": condition_target_semantics,
        "condition_independence_identity": condition_independence_identity,
        "adaptive_policy_context": adaptive_policy_context,
        "online_preselection_states": online_preselection_states,
        "policy_visible_question_trajectory": policy_visible_question_trajectory,
        "sequential_audit_state": sequential_audit_state,
        "collection_decision": collection_decision,
        "lineage": lineage,
        "confirmation_partition_unopened_by_online_policy": True,
        "condition_assessment_unopened": True,
        "reference_labels_unopened": True,
        "adaptive_calibration_bundle_unavailable": True,
    }
    source = ConditionCalibrationCollectionSourceV1.model_validate(
        {**payload, "collection_source_sha256": hash_canonical(payload)}
    )
    try:
        from literature_multiverse.verifier import (
            validate_condition_calibration_collection_source_external_replay,
        )

        return validate_condition_calibration_collection_source_external_replay(source)
    except (ImportError, ValueError) as exc:
        raise ValueError(
            f"condition_collection_source_external_replay_failed:{exc}"
        ) from exc


def validate_condition_calibration_collection_source_anchor_v1(
    *,
    source_anchor: ConditionCalibrationCollectionSourceAnchorV1,
    collection_source: ConditionCalibrationCollectionSourceV1,
) -> ConditionCalibrationCollectionSourceAnchorV1:
    """Recompute one content-silent anchor from its exact replayable source."""

    source = collection_source
    anchor = source_anchor
    visible = source.policy_visible_question_trajectory
    if visible is None:
        raise ValueError("condition_collection_source_roster_visible_missing")
    if (
        anchor.question_id != source.question_id
        or anchor.policy_arm_id != source.policy_arm_id
        or anchor.policy_context_sha256
        != source.adaptive_policy_context.policy_context_sha256
        or anchor.visible_trajectory_sha256 != visible.trajectory_sha256
        or anchor.collection_source_sha256 != source.collection_source_sha256
        or anchor.collection_source_decision_sha256
        != source.collection_decision.decision_sha256
    ):
        raise ValueError("condition_collection_source_roster_anchor_mismatch")
    return anchor


def match_validated_condition_calibration_collection_source_membership_v1(
    *,
    collection_source_roster: ConditionCalibrationCollectionSourceRosterV1,
    collection_source: ConditionCalibrationCollectionSourceV1,
    expected_source_roster_sha256: str,
    expected_source_membership_sha256: str,
) -> ConditionCalibrationCollectionSourceAnchorV1:
    """Match objects already replayed within the same public boundary.

    This avoids repeating scientific replay after the caller has just validated both
    typed inputs.  Callers receiving authored JSON must use
    :func:`validate_condition_calibration_collection_source_membership_v1` instead.
    """

    roster = collection_source_roster
    source = collection_source
    if (
        roster.source_roster_sha256 != expected_source_roster_sha256
        or roster.source_membership_sha256 != expected_source_membership_sha256
    ):
        raise ValueError("condition_collection_source_roster_external_anchor_mismatch")
    source_payload = source.model_dump(mode="json")
    matching_payloads = [
        payload
        for payload in roster.collection_sources
        if payload.get("collection_source_sha256")
        == source.collection_source_sha256
    ]
    if matching_payloads != [source_payload]:
        raise ValueError("condition_collection_source_roster_payload_mismatch")
    matching_anchors = [
        anchor
        for anchor in roster.source_anchors
        if anchor.question_id == source.question_id
        and anchor.policy_arm_id == source.policy_arm_id
    ]
    if len(matching_anchors) != 1:
        raise ValueError("condition_collection_source_roster_anchor_missing")
    return validate_condition_calibration_collection_source_anchor_v1(
        source_anchor=matching_anchors[0],
        collection_source=source,
    )


def validate_condition_calibration_collection_source_membership_v1(
    *,
    collection_source_roster: ConditionCalibrationCollectionSourceRosterV1,
    collection_source: ConditionCalibrationCollectionSourceV1,
    expected_source_roster_sha256: str,
    expected_source_membership_sha256: str,
) -> ConditionCalibrationCollectionSourceAnchorV1:
    """Replay authored inputs, then prove exact pre-outcome roster membership."""

    try:
        roster = ConditionCalibrationCollectionSourceRosterV1.model_validate(
            collection_source_roster.model_dump(mode="json")
        )
        source = ConditionCalibrationCollectionSourceV1.model_validate(
            collection_source.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise ValueError("condition_collection_source_roster_integrity_changed") from exc
    return match_validated_condition_calibration_collection_source_membership_v1(
        collection_source_roster=roster,
        collection_source=source,
        expected_source_roster_sha256=expected_source_roster_sha256,
        expected_source_membership_sha256=expected_source_membership_sha256,
    )


class ConditionCalibrationAssessmentReceiptV1(ContractModel):
    """Replayed held-out condition assessment for one immutable collection source."""

    receipt_version: Literal["condition-calibration-assessment-receipt-v1"] = (
        "condition-calibration-assessment-receipt-v1"
    )
    source_anchor: ConditionCalibrationCollectionSourceAnchorV1
    source_roster_sha256: str
    source_membership_sha256: str
    collection_source: ConditionCalibrationCollectionSourceV1
    collection_source_sha256: str
    collection_source_decision_sha256: str
    condition_confirmation_assessment: ConditionConfirmationAssessmentV1
    condition_confirmation_gate: ConditionConfirmationGateAssessmentV1
    calibration_gate_result: ConditionCalibrationGateResultV1
    reference_labels_unopened: Literal[True] = True
    receipt_sha256: str

    @field_validator(
        "collection_source_sha256",
        "collection_source_decision_sha256",
        "source_roster_sha256",
        "source_membership_sha256",
        "receipt_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("condition_collection_receipt_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> ConditionCalibrationAssessmentReceiptV1:
        source = self.collection_source
        validate_condition_calibration_collection_source_anchor_v1(
            source_anchor=self.source_anchor,
            collection_source=source,
        )
        if (
            source.collection_split != "calibration"
            or
            source.collection_decision.outcome != "condition_gate_ready"
            or source.policy_visible_question_trajectory is None
            or self.collection_source_sha256 != source.collection_source_sha256
            or self.collection_source_decision_sha256
            != source.collection_decision.decision_sha256
        ):
            raise ValueError("condition_collection_receipt_source_not_gate_ready")
        invocation = source.collection_decision.condition_gate_invocation_proof
        assert invocation is not None
        pipeline_sha256 = source.pipeline_verification.computed_pipeline_sha256
        assert pipeline_sha256 is not None
        try:
            assessment = validate_condition_confirmation_assessment(
                plan=source.condition_plan,
                model=source.condition_frozen_model,
                full_graph=source.current_full_evidence_graph,
                assessment=self.condition_confirmation_assessment,
                current_pipeline_sha256=pipeline_sha256,
            )
        except (ConditionConfirmationError, ValueError) as exc:
            raise ValueError("condition_collection_receipt_assessment_replay_failed") from exc
        expected_gate = freeze_condition_confirmation_gate_assessment(
            provisional_claim_decision="condition_dependent",
            status=assessment.status,
            reasons=assessment.reasons,
            condition_projection_sha256=(
                source.condition_calibration_projection.projection_sha256
            ),
            target_sha256=(
                source.condition_calibration_projection.condition_target_sha256
            ),
            plan_sha256=source.condition_plan.plan_sha256,
            config_sha256=source.condition_plan.config_sha256,
            model_sha256=source.condition_frozen_model.model_sha256,
            assessment_sha256=assessment.assessment_sha256,
        )
        expected_result = freeze_condition_calibration_gate_result_v1(
            question_id=source.question_id,
            policy_arm_id=source.policy_arm_id,
            condition_gate_invocation_proof=invocation,
            gate_assessment=expected_gate,
            collection_source_sha256=source.collection_source_sha256,
            collection_source_decision_sha256=(
                source.collection_decision.decision_sha256
            ),
        )
        if (
            assessment != self.condition_confirmation_assessment
            or expected_gate != self.condition_confirmation_gate
            or expected_result != self.calibration_gate_result
        ):
            raise ValueError("condition_collection_receipt_terminal_join_mismatch")
        try:
            from literature_multiverse.verifier import (
                validate_condition_calibration_collection_source_external_replay,
            )

            validate_condition_calibration_collection_source_external_replay(source)
        except (ImportError, ValueError) as exc:
            raise ValueError(
                f"condition_collection_receipt_source_replay_failed:{exc}"
            ) from exc
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if hash_canonical(payload) != self.receipt_sha256:
            raise ValueError("condition_collection_receipt_hash_mismatch")
        return self

    @property
    def question_id(self) -> str:
        return self.collection_source.question_id

    @property
    def policy_arm_id(self) -> str:
        return self.collection_source.policy_arm_id

    @property
    def policy_context_sha256(self) -> str:
        return self.collection_source.adaptive_policy_context.policy_context_sha256

    @property
    def target_semantics_sha256(self) -> str:
        return self.collection_source.condition_target_semantics.target_semantics_sha256

    @property
    def independence_identity_sha256(self) -> str:
        return (
            self.collection_source.condition_independence_identity
            .independence_identity_sha256
        )

    @property
    def terminal_state_sha256(self) -> str:
        invocation = self.collection_source.collection_decision.condition_gate_invocation_proof
        assert invocation is not None
        return invocation.terminal_preselection_state_sha256

    @property
    def condition_gate_invocation_proof(self) -> ConditionGateInvocationProofV2:
        invocation = self.collection_source.collection_decision.condition_gate_invocation_proof
        assert invocation is not None
        return invocation

    @property
    def condition_gate_invocation_proof_sha256(self) -> str:
        return self.condition_gate_invocation_proof.proof_sha256

    @property
    def condition_calibration_projection(self) -> ConditionCalibrationProjectionV1:
        return self.collection_source.condition_calibration_projection

    @property
    def condition_calibration_projection_sha256(self) -> str:
        return self.condition_calibration_projection.projection_sha256

    @property
    def policy_visible_question_trajectory(self) -> PolicyVisibleQuestionTrajectoryV2:
        visible = self.collection_source.policy_visible_question_trajectory
        assert visible is not None
        return visible


def freeze_condition_calibration_assessment_receipt_v1(
    *,
    collection_source_roster: ConditionCalibrationCollectionSourceRosterV1,
    collection_source: ConditionCalibrationCollectionSourceV1,
    condition_confirmation_assessment: ConditionConfirmationAssessmentV1,
) -> ConditionCalibrationAssessmentReceiptV1:
    # The outcome-free roster is validated and membership is proven before the
    # assessment object is inspected.  The CLI mirrors this ordering at file-open
    # time so the physical held-out-outcome firewall is independently auditable.
    try:
        roster = ConditionCalibrationCollectionSourceRosterV1.model_validate(
            collection_source_roster.model_dump(mode="json")
        )
        source = ConditionCalibrationCollectionSourceV1.model_validate(
            collection_source.model_dump(mode="json")
        )
        source_anchor = match_validated_condition_calibration_collection_source_membership_v1(
            collection_source_roster=roster,
            collection_source=source,
            expected_source_roster_sha256=roster.source_roster_sha256,
            expected_source_membership_sha256=roster.source_membership_sha256,
        )
    except (AttributeError, ValueError) as exc:
        raise ValueError("condition_collection_receipt_input_integrity_changed") from exc
    try:
        assessment = ConditionConfirmationAssessmentV1.model_validate(
            condition_confirmation_assessment.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise ValueError("condition_collection_receipt_assessment_integrity_changed") from exc
    if (
        source.collection_split != "calibration"
        or
        source.collection_decision.outcome != "condition_gate_ready"
        or source.policy_visible_question_trajectory is None
    ):
        raise ValueError("condition_collection_receipt_source_not_gate_ready")
    invocation = source.collection_decision.condition_gate_invocation_proof
    assert invocation is not None
    pipeline_sha256 = source.pipeline_verification.computed_pipeline_sha256
    assert pipeline_sha256 is not None
    try:
        assessment = validate_condition_confirmation_assessment(
            plan=source.condition_plan,
            model=source.condition_frozen_model,
            full_graph=source.current_full_evidence_graph,
            assessment=assessment,
            current_pipeline_sha256=pipeline_sha256,
        )
    except (ConditionConfirmationError, ValueError) as exc:
        raise ValueError("condition_collection_receipt_assessment_replay_failed") from exc
    gate = freeze_condition_confirmation_gate_assessment(
        provisional_claim_decision="condition_dependent",
        status=assessment.status,
        reasons=assessment.reasons,
        condition_projection_sha256=(
            source.condition_calibration_projection.projection_sha256
        ),
        target_sha256=source.condition_calibration_projection.condition_target_sha256,
        plan_sha256=source.condition_plan.plan_sha256,
        config_sha256=source.condition_plan.config_sha256,
        model_sha256=source.condition_frozen_model.model_sha256,
        assessment_sha256=assessment.assessment_sha256,
    )
    result = freeze_condition_calibration_gate_result_v1(
        question_id=source.question_id,
        policy_arm_id=source.policy_arm_id,
        condition_gate_invocation_proof=invocation,
        gate_assessment=gate,
        collection_source_sha256=source.collection_source_sha256,
        collection_source_decision_sha256=source.collection_decision.decision_sha256,
    )
    payload: dict[str, Any] = {
        "receipt_version": "condition-calibration-assessment-receipt-v1",
        "source_anchor": source_anchor,
        "source_roster_sha256": roster.source_roster_sha256,
        "source_membership_sha256": roster.source_membership_sha256,
        "collection_source": source,
        "collection_source_sha256": source.collection_source_sha256,
        "collection_source_decision_sha256": source.collection_decision.decision_sha256,
        "condition_confirmation_assessment": assessment,
        "condition_confirmation_gate": gate,
        "calibration_gate_result": result,
        "reference_labels_unopened": True,
    }
    return ConditionCalibrationAssessmentReceiptV1.model_validate(
        {**payload, "receipt_sha256": hash_canonical(payload)}
    )


class ConditionVerificationCertificateV6(ContractModel):
    """Immutable outcome-free source certificate before the terminal gate join.

    A v6 certificate is always abstained and is frozen before any held-out
    confirmation outcome is opened.  Its gate is therefore necessarily ``missing``
    and its assessment necessarily absent.  A dedicated finalizer may join an exact
    v6 to a replayed held-out assessment and produce a distinct v7 artifact; the
    online pipeline is never rerun after that outcome opens.
    """

    certificate_version: Literal[
        "literature-multiverse-condition-verification-v6"
    ] = "literature-multiverse-condition-verification-v6"
    run_id: Annotated[str, Field(pattern=r"^verify-condition-v6-[0-9a-f]{16}$")]
    generated_at: datetime
    status: Literal["abstained"] = "abstained"
    reasons: list[str]
    claim_manifest: dict[str, Any]
    claim_manifest_sha256: str
    corpus: dict[str, Any]
    corpus_sha256: str
    source_evidence_graph: EvidenceGraph
    source_evidence_graph_sha256: str
    current_full_evidence_graph: EvidenceGraph
    current_full_evidence_graph_sha256: str
    development_evidence_graph: EvidenceGraph
    development_evidence_graph_sha256: str
    adapter_issues: list[dict[str, Any]]
    synthesis: dict[str, Any]
    synthesis_sha256: str
    counterfactual_reruns: list[dict[str, Any]]
    audit_candidates: list[dict[str, Any]]
    release_assessment: ConditionClaimReleaseAssessmentV1
    pipeline_verification: PipelineFingerprintVerification
    complete_corpus_identity: CompleteCorpusIdentity
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None
    condition_plan: ConditionConfirmationPlanV1
    condition_frozen_model: ConditionConfirmationFrozenModelV1
    condition_confirmation_assessment: None = None
    condition_calibration_projection: ConditionCalibrationProjectionV1
    condition_confirmation_gate: ConditionConfirmationGateAssessmentV1
    condition_target_semantics: AdaptiveTargetSemanticsBindingV2
    condition_independence_identity: StrongIndependenceIdentityV1
    condition_gate_invocation_proof: ConditionGateInvocationProofV2 | None
    release_qualification_proof: (
        ConfirmationAwareReleaseQualificationProofV2 | None
    )
    adaptive_policy_context: AdaptivePolicyContext
    adaptive_calibration_bundle_v2: AdaptiveCalibrationBundleV2
    adaptive_release_candidate_v2: ProspectiveAdaptiveReleaseCandidateV2
    adaptive_prospective_assessment_v2: AdaptiveProspectiveAssessmentV2
    sequential_audit_state: SequentialVerificationState
    production_stop_decision: ConditionProductionStopDecisionV2
    lineage: list[CertificateLineageStage]
    certificate_sha256: str
    interpretation: Literal[
        "outcome-free abstained immutable source awaiting a held-out predictive "
        "association gate; not a final release, causal proof, scientific truth, or "
        "domain-shift guarantee"
    ] = (
        "outcome-free abstained immutable source awaiting a held-out predictive "
        "association gate; not a final release, causal proof, scientific truth, or "
        "domain-shift guarantee"
    )

    @field_validator(
        "claim_manifest_sha256",
        "corpus_sha256",
        "source_evidence_graph_sha256",
        "current_full_evidence_graph_sha256",
        "development_evidence_graph_sha256",
        "synthesis_sha256",
        "certificate_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("condition_v6_sha256_invalid")
        return value

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, value: list[str]) -> list[str]:
        if not value or value != sorted(set(value)) or any(not item for item in value):
            raise ValueError("condition_v6_reasons_invalid")
        return value

    @model_validator(mode="after")
    def validate_certificate(self) -> ConditionVerificationCertificateV6:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("condition_v6_generated_at_requires_timezone")
        if hash_canonical(self.claim_manifest) != self.claim_manifest_sha256:
            raise ValueError("condition_v6_manifest_hash_mismatch")
        if self.corpus.get("source_sha256") != self.corpus_sha256:
            raise ValueError("condition_v6_corpus_hash_mismatch")
        graph_checks = {
            "source": (
                self.source_evidence_graph,
                self.source_evidence_graph_sha256,
            ),
            "current_full": (
                self.current_full_evidence_graph,
                self.current_full_evidence_graph_sha256,
            ),
            "development": (
                self.development_evidence_graph,
                self.development_evidence_graph_sha256,
            ),
        }
        for label, (graph, digest) in graph_checks.items():
            if hash_canonical(graph) != digest:
                raise ValueError(f"condition_v6_{label}_graph_hash_mismatch")
        if hash_canonical(self.synthesis) != self.synthesis_sha256:
            raise ValueError("condition_v6_synthesis_hash_mismatch")
        pipeline_sha256 = self.pipeline_verification.computed_pipeline_sha256
        manifest_question_id = self.claim_manifest.get("question_id")
        manifest_population_id = self.claim_manifest.get("population_id")
        manifest_protocol = self.claim_manifest.get("protocol")
        manifest_cutoff = (
            manifest_protocol.get("corpus_cutoff")
            if isinstance(manifest_protocol, dict)
            else None
        )
        manifest_target = self.claim_manifest.get("global_condition_target")
        manifest_target_sha256 = (
            manifest_target.get("target_sha256")
            if isinstance(manifest_target, dict)
            else None
        )
        if (
            self.pipeline_verification.status != "matched"
            or pipeline_sha256 is None
            or pipeline_sha256 != self.release_assessment.pipeline_sha256
            or self.claim_manifest.get("claim_manifest_version") != "3"
            or manifest_question_id != self.release_assessment.question_id
            or manifest_question_id != self.condition_target_semantics.question_id
            or manifest_population_id != self.adaptive_policy_context.population_id
            or manifest_target_sha256
            != self.condition_calibration_projection.condition_target_sha256
            or self.complete_corpus_identity.corpus_id != self.corpus.get("corpus_id")
            or self.complete_corpus_identity.corpus_source_sha256 != self.corpus_sha256
            or self.complete_corpus_identity.corpus_cutoff != manifest_cutoff
            or self.condition_calibration_projection.corpus_snapshot_sha256
            != self.complete_corpus_identity.membership_sha256
            or self.condition_calibration_projection.corpus_cutoff != manifest_cutoff
        ):
            raise ValueError("condition_v6_pipeline_verification_mismatch")
        plan = self.condition_plan
        projection = self.condition_calibration_projection
        gate = self.condition_confirmation_gate
        if (
            plan.pipeline_sha256 != pipeline_sha256
            or plan.full_graph_sha256 != self.current_full_evidence_graph_sha256
            or plan.development_graph_sha256
            != self.development_evidence_graph_sha256
            or projection.plan_sha256 != plan.plan_sha256
            or projection.full_graph_sha256 != plan.full_graph_sha256
            or projection.development_graph_sha256
            != plan.development_graph_sha256
            or projection.pipeline_sha256 != pipeline_sha256
            or self.release_assessment.condition_calibration_projection
            != projection
            or self.release_assessment.condition_confirmation_gate != gate
            or self.release_assessment.evidence_graph_sha256
            != self.development_evidence_graph_sha256
            or self.release_assessment.synthesis_sha256 != self.synthesis_sha256
        ):
            raise ValueError("condition_v6_scientific_context_mismatch")
        if (
            projection.target_semantics != self.condition_target_semantics
            or projection.independence_identity_sha256
            != self.condition_independence_identity.independence_identity_sha256
        ):
            raise ValueError("condition_v6_target_or_independence_mismatch")
        try:
            recomputed_model = validate_condition_confirmation_model(
                plan=plan,
                development_graph=self.development_evidence_graph,
                model=self.condition_frozen_model,
                current_pipeline_sha256=pipeline_sha256,
            )
        except (ConditionConfirmationError, ValueError) as exc:
            raise ValueError("condition_v6_model_replay_failed") from exc
        if recomputed_model != self.condition_frozen_model:
            raise ValueError("condition_v6_model_mismatch")
        if (
            gate.status != "missing"
            or self.condition_confirmation_assessment is not None
            or not self.release_assessment.terminal_gate_deferred
        ):
            raise ValueError("condition_v6_must_be_outcome_free")
        try:
            bundle = validate_adaptive_calibration_bundle_v2_integrity(
                self.adaptive_calibration_bundle_v2
            )
        except AdaptiveCalibrationError as exc:
            raise ValueError("condition_v6_adaptive_bundle_invalid") from exc
        if self.adaptive_policy_context not in (
            bundle.development_freeze.base_freeze.policy_contexts
        ):
            raise ValueError("condition_v6_policy_context_not_frozen")
        candidate = self.adaptive_release_candidate_v2
        if (
            candidate.base_candidate.policy_context_sha256
            != self.adaptive_policy_context.policy_context_sha256
            or candidate.base_candidate.corpus != self.complete_corpus_identity
            or candidate.target_semantics != self.condition_target_semantics
            or candidate.independence_identity
            != self.condition_independence_identity
            or candidate.condition_projection != projection
            or candidate.condition_gate_invocation_proof
            != self.condition_gate_invocation_proof
            or candidate.release_qualification_proof
            != self.release_qualification_proof
            or candidate.terminal_gate_result is not None
        ):
            raise ValueError("condition_v6_adaptive_candidate_lineage_mismatch")
        expected_adaptive = assess_confirmation_aware_adaptive_release_candidate(
            candidate,
            bundle,
        )
        if (
            expected_adaptive != self.adaptive_prospective_assessment_v2
            or expected_adaptive.status != "abstained"
        ):
            raise ValueError("condition_v6_adaptive_assessment_mismatch")
        invocation = self.condition_gate_invocation_proof
        qualification = self.release_qualification_proof
        if invocation is None:
            if qualification is not None or gate.status != "missing":
                raise ValueError("condition_v6_terminal_artifact_order_mismatch")
        else:
            if (
                invocation.condition_projection != projection
                or self.production_stop_decision.condition_gate_invocation_proof
                != invocation
            ):
                raise ValueError("condition_v6_invocation_mismatch")
            if qualification is not None:
                expected_qualification = (
                    freeze_confirmation_aware_release_qualification_proof_v2(
                        question_id=self.release_assessment.question_id,
                        policy_arm_id=self.adaptive_policy_context.policy_arm_id,
                        condition_gate_invocation_proof=invocation,
                        bundle=bundle,
                    )
                )
                if qualification != expected_qualification:
                    raise ValueError("condition_v6_qualification_mismatch")
        if self.production_stop_decision.release_assessment.pipeline_sha256 != pipeline_sha256:
            raise ValueError("condition_v6_production_decision_pipeline_mismatch")
        expected_reasons = sorted(
            set(self.release_assessment.reasons)
            | set(self.production_stop_decision.blocking_adapter_reasons)
        )
        if self.reasons != expected_reasons:
            raise ValueError("condition_v6_reason_ledger_mismatch")
        payload = self.model_dump(mode="json", exclude={"certificate_sha256"})
        if hash_canonical(payload) != self.certificate_sha256:
            raise ValueError("condition_v6_certificate_hash_mismatch")
        return self


def freeze_condition_verification_certificate_v6(
    *,
    generated_at: datetime,
    reasons: list[str],
    claim_manifest: dict[str, Any],
    corpus: dict[str, Any],
    corpus_sha256: str,
    source_evidence_graph: EvidenceGraph,
    current_full_evidence_graph: EvidenceGraph,
    development_evidence_graph: EvidenceGraph,
    adapter_issues: list[dict[str, Any]],
    synthesis: dict[str, Any],
    counterfactual_reruns: list[dict[str, Any]],
    audit_candidates: list[dict[str, Any]],
    release_assessment: ConditionClaimReleaseAssessmentV1,
    pipeline_verification: PipelineFingerprintVerification,
    complete_corpus_identity: CompleteCorpusIdentity,
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None,
    condition_plan: ConditionConfirmationPlanV1,
    condition_frozen_model: ConditionConfirmationFrozenModelV1,
    condition_calibration_projection: ConditionCalibrationProjectionV1,
    condition_confirmation_gate: ConditionConfirmationGateAssessmentV1,
    condition_target_semantics: AdaptiveTargetSemanticsBindingV2,
    condition_independence_identity: AdaptiveIndependenceIdentityV2,
    condition_gate_invocation_proof: ConditionGateInvocationProofV2 | None,
    release_qualification_proof: (
        ConfirmationAwareReleaseQualificationProofV2 | None
    ),
    adaptive_policy_context: AdaptivePolicyContext,
    adaptive_calibration_bundle_v2: AdaptiveCalibrationBundleV2,
    adaptive_release_candidate_v2: ProspectiveAdaptiveReleaseCandidateV2,
    adaptive_prospective_assessment_v2: AdaptiveProspectiveAssessmentV2,
    sequential_audit_state: SequentialVerificationState,
    production_stop_decision: ConditionProductionStopDecisionV2,
    lineage: list[CertificateLineageStage],
) -> ConditionVerificationCertificateV6:
    generated = generated_at.isoformat().replace("+00:00", "Z")
    claim_manifest_sha256 = hash_canonical(claim_manifest)
    source_graph_sha256 = hash_canonical(source_evidence_graph)
    current_full_sha256 = hash_canonical(current_full_evidence_graph)
    development_sha256 = hash_canonical(development_evidence_graph)
    synthesis_sha256 = hash_canonical(synthesis)
    run_identity = hash_canonical(
        {
            "claim_manifest_sha256": claim_manifest_sha256,
            "current_full_evidence_graph_sha256": current_full_sha256,
            "development_evidence_graph_sha256": development_sha256,
            "release_decision_sha256": release_assessment.decision_sha256,
            "production_stop_decision_sha256": production_stop_decision.decision_sha256,
            "condition_gate_assessment_sha256": (
                condition_confirmation_gate.gate_assessment_sha256
            ),
            "adaptive_candidate_sha256": adaptive_release_candidate_v2.candidate_sha256,
        }
    )
    payload: dict[str, Any] = {
        "certificate_version": "literature-multiverse-condition-verification-v6",
        "run_id": f"verify-condition-v6-{run_identity[:16]}",
        "generated_at": generated,
        "status": "abstained",
        "reasons": sorted(set(reasons)),
        "claim_manifest": claim_manifest,
        "claim_manifest_sha256": claim_manifest_sha256,
        "corpus": corpus,
        "corpus_sha256": corpus_sha256,
        "source_evidence_graph": source_evidence_graph,
        "source_evidence_graph_sha256": source_graph_sha256,
        "current_full_evidence_graph": current_full_evidence_graph,
        "current_full_evidence_graph_sha256": current_full_sha256,
        "development_evidence_graph": development_evidence_graph,
        "development_evidence_graph_sha256": development_sha256,
        "adapter_issues": adapter_issues,
        "synthesis": synthesis,
        "synthesis_sha256": synthesis_sha256,
        "counterfactual_reruns": counterfactual_reruns,
        "audit_candidates": audit_candidates,
        "release_assessment": release_assessment,
        "pipeline_verification": pipeline_verification,
        "complete_corpus_identity": complete_corpus_identity,
        "item_risk_scoring_receipt": item_risk_scoring_receipt,
        "condition_plan": condition_plan,
        "condition_frozen_model": condition_frozen_model,
        "condition_confirmation_assessment": None,
        "condition_calibration_projection": condition_calibration_projection,
        "condition_confirmation_gate": condition_confirmation_gate,
        "condition_target_semantics": condition_target_semantics,
        "condition_independence_identity": condition_independence_identity,
        "condition_gate_invocation_proof": condition_gate_invocation_proof,
        "release_qualification_proof": release_qualification_proof,
        "adaptive_policy_context": adaptive_policy_context,
        "adaptive_calibration_bundle_v2": adaptive_calibration_bundle_v2,
        "adaptive_release_candidate_v2": adaptive_release_candidate_v2,
        "adaptive_prospective_assessment_v2": adaptive_prospective_assessment_v2,
        "sequential_audit_state": sequential_audit_state,
        "production_stop_decision": production_stop_decision,
        "lineage": lineage,
        "interpretation": (
            "outcome-free abstained immutable source awaiting a held-out predictive "
            "association gate; not a final release, causal proof, scientific truth, or "
            "domain-shift guarantee"
        ),
    }
    return ConditionVerificationCertificateV6.model_validate(
        {**payload, "certificate_sha256": hash_canonical(payload)}
    )


class ConditionVerificationCertificateV8(ConditionVerificationCertificateV6):
    """Outcome-free condition source bound to an externally replayed composition."""

    certificate_version: Literal[
        "literature-multiverse-condition-verification-v8"
    ] = "literature-multiverse-condition-verification-v8"
    run_id: Annotated[str, Field(pattern=r"^verify-condition-v8-[0-9a-f]{16}$")]
    composition_external_replay_receipt: (
        CorpusPipelineCompositionExternalReplayReceiptV1
    )
    composition_external_replay_receipt_sha256: str
    complete_corpus_identity_v3: CompleteCorpusIdentityV3
    complete_corpus_membership_v3_sha256: str
    manifest_corpus_policy_binding_v3: ManifestCorpusPolicyBindingV3
    manifest_corpus_policy_binding_v3_sha256: str
    cleared_adapter_issue_code: Literal["corpus_pipeline_identity_mismatch"] = (
        "corpus_pipeline_identity_mismatch"
    )
    v6_common_contract_replay_sha256: str

    @field_validator(
        "composition_external_replay_receipt_sha256",
        "complete_corpus_membership_v3_sha256",
        "manifest_corpus_policy_binding_v3_sha256",
        "v6_common_contract_replay_sha256",
    )
    @classmethod
    def validate_v8_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("condition_v8_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_certificate(self) -> ConditionVerificationCertificateV8:
        try:
            receipt = CorpusPipelineCompositionExternalReplayReceiptV1.model_validate(
                self.composition_external_replay_receipt.model_dump(mode="json")
            )
            complete_v3 = validate_complete_corpus_identity_v3(
                self.complete_corpus_identity_v3
            )
            binding_v3 = validate_manifest_corpus_policy_binding_v3(
                self.manifest_corpus_policy_binding_v3
            )
        except (AttributeError, ValueError) as exc:
            raise ValueError("condition_v8_composition_contract_invalid") from exc
        if (
            self.composition_external_replay_receipt_sha256
            != receipt.receipt_sha256
            or self.complete_corpus_membership_v3_sha256
            != complete_v3.membership_composition_sha256
            or self.manifest_corpus_policy_binding_v3_sha256
            != binding_v3.manifest_corpus_policy_binding_sha256
            or binding_v3.complete_corpus_identity_v3 != complete_v3
            or binding_v3.claim_manifest_sha256 != self.claim_manifest_sha256
            or binding_v3.policy_sha256
            != _certificate_manifest_policy_sha256(self.claim_manifest)
            or complete_v3.external_replay_receipt != receipt
            or complete_v3.complete_corpus_identity_v2.complete_corpus_membership_v1
            != self.complete_corpus_identity
            or self.pipeline_verification != receipt.composed_pipeline_verification
            or self.release_assessment.pipeline_sha256
            != receipt.release_pipeline_sha256
        ):
            raise ValueError("condition_v8_composition_alias_mismatch")
        _require_verification_v8_receipt_corpus_aliases(
            corpus=self.corpus,
            source_evidence_graph=self.source_evidence_graph,
            corpus_sha256=self.corpus_sha256,
            receipt=receipt,
        )
        mismatch_reason = "adapter:corpus_pipeline_identity_mismatch"
        if mismatch_reason in self.reasons or any(
            issue.get("code") == self.cleared_adapter_issue_code
            for issue in self.adapter_issues
        ):
            raise ValueError("condition_v8_legacy_pipeline_blocker_not_cleared")
        if len(self.lineage) != 4 or self.lineage[-1] != _verification_v8_composition_stage(
            receipt=receipt,
            complete_corpus_identity_v3=complete_v3,
            manifest_corpus_policy_binding_v3=binding_v3,
        ):
            raise ValueError("condition_v8_composition_lineage_mismatch")
        try:
            shadow = freeze_condition_verification_certificate_v6(
                generated_at=self.generated_at,
                reasons=self.reasons,
                claim_manifest=self.claim_manifest,
                corpus=_verification_v8_shadow_corpus(
                    corpus=self.corpus,
                    receipt=receipt,
                ),
                corpus_sha256=self.corpus_sha256,
                source_evidence_graph=self.source_evidence_graph,
                current_full_evidence_graph=self.current_full_evidence_graph,
                development_evidence_graph=self.development_evidence_graph,
                adapter_issues=self.adapter_issues,
                synthesis=self.synthesis,
                counterfactual_reruns=self.counterfactual_reruns,
                audit_candidates=self.audit_candidates,
                release_assessment=self.release_assessment,
                pipeline_verification=self.pipeline_verification,
                complete_corpus_identity=self.complete_corpus_identity,
                item_risk_scoring_receipt=self.item_risk_scoring_receipt,
                condition_plan=self.condition_plan,
                condition_frozen_model=self.condition_frozen_model,
                condition_calibration_projection=(
                    self.condition_calibration_projection
                ),
                condition_confirmation_gate=self.condition_confirmation_gate,
                condition_target_semantics=self.condition_target_semantics,
                condition_independence_identity=(
                    self.condition_independence_identity
                ),
                condition_gate_invocation_proof=(
                    self.condition_gate_invocation_proof
                ),
                release_qualification_proof=self.release_qualification_proof,
                adaptive_policy_context=self.adaptive_policy_context,
                adaptive_calibration_bundle_v2=self.adaptive_calibration_bundle_v2,
                adaptive_release_candidate_v2=self.adaptive_release_candidate_v2,
                adaptive_prospective_assessment_v2=(
                    self.adaptive_prospective_assessment_v2
                ),
                sequential_audit_state=self.sequential_audit_state,
                production_stop_decision=self.production_stop_decision,
                lineage=self.lineage[:-1],
            )
        except ValueError as exc:
            raise ValueError("condition_v8_v6_common_contract_replay_failed") from exc
        if self.v6_common_contract_replay_sha256 != shadow.certificate_sha256:
            raise ValueError("condition_v8_common_contract_replay_hash_mismatch")
        v8_exclusions = {
            "certificate_version",
            "run_id",
            "corpus",
            "lineage",
            "certificate_sha256",
            "composition_external_replay_receipt",
            "composition_external_replay_receipt_sha256",
            "complete_corpus_identity_v3",
            "complete_corpus_membership_v3_sha256",
            "manifest_corpus_policy_binding_v3",
            "manifest_corpus_policy_binding_v3_sha256",
            "cleared_adapter_issue_code",
            "v6_common_contract_replay_sha256",
        }
        shadow_exclusions = {
            "certificate_version",
            "run_id",
            "corpus",
            "lineage",
            "certificate_sha256",
        }
        if self.model_dump(mode="json", exclude=v8_exclusions) != shadow.model_dump(
            mode="json", exclude=shadow_exclusions
        ):
            raise ValueError("condition_v8_common_contract_projection_mismatch")
        run_identity = hash_canonical(
            {
                "v6_common_contract_replay_sha256": shadow.certificate_sha256,
                "composition_external_replay_receipt_sha256": receipt.receipt_sha256,
                "complete_corpus_membership_v3_sha256": (
                    complete_v3.membership_composition_sha256
                ),
                "manifest_corpus_policy_binding_v3_sha256": (
                    binding_v3.manifest_corpus_policy_binding_sha256
                ),
            }
        )
        if self.run_id != f"verify-condition-v8-{run_identity[:16]}":
            raise ValueError("condition_v8_run_identity_mismatch")
        payload = self.model_dump(mode="json", exclude={"certificate_sha256"})
        if self.certificate_sha256 != hash_canonical(payload):
            raise ValueError("condition_v8_certificate_hash_mismatch")
        return self


def freeze_condition_verification_certificate_v8(
    *,
    generated_at: datetime,
    reasons: list[str],
    claim_manifest: dict[str, Any],
    corpus: dict[str, Any],
    corpus_sha256: str,
    source_evidence_graph: EvidenceGraph,
    current_full_evidence_graph: EvidenceGraph,
    development_evidence_graph: EvidenceGraph,
    adapter_issues: list[dict[str, Any]],
    synthesis: dict[str, Any],
    counterfactual_reruns: list[dict[str, Any]],
    audit_candidates: list[dict[str, Any]],
    release_assessment: ConditionClaimReleaseAssessmentV1,
    pipeline_verification: PipelineFingerprintVerification,
    complete_corpus_identity: CompleteCorpusIdentity,
    complete_corpus_identity_v3: CompleteCorpusIdentityV3,
    composition_external_replay_receipt: (
        CorpusPipelineCompositionExternalReplayReceiptV1
    ),
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None,
    condition_plan: ConditionConfirmationPlanV1,
    condition_frozen_model: ConditionConfirmationFrozenModelV1,
    condition_calibration_projection: ConditionCalibrationProjectionV1,
    condition_confirmation_gate: ConditionConfirmationGateAssessmentV1,
    condition_target_semantics: AdaptiveTargetSemanticsBindingV2,
    condition_independence_identity: AdaptiveIndependenceIdentityV2,
    condition_gate_invocation_proof: ConditionGateInvocationProofV2 | None,
    release_qualification_proof: ConfirmationAwareReleaseQualificationProofV2 | None,
    adaptive_policy_context: AdaptivePolicyContext,
    adaptive_calibration_bundle_v2: AdaptiveCalibrationBundleV2,
    adaptive_release_candidate_v2: ProspectiveAdaptiveReleaseCandidateV2,
    adaptive_prospective_assessment_v2: AdaptiveProspectiveAssessmentV2,
    sequential_audit_state: SequentialVerificationState,
    production_stop_decision: ConditionProductionStopDecisionV2,
    lineage: list[CertificateLineageStage],
) -> ConditionVerificationCertificateV8:
    """Freeze outcome-free condition v8 via the exact v6 common projection."""

    receipt = CorpusPipelineCompositionExternalReplayReceiptV1.model_validate(
        composition_external_replay_receipt.model_dump(mode="json")
    )
    complete_v3 = validate_complete_corpus_identity_v3(complete_corpus_identity_v3)
    if (
        complete_v3.external_replay_receipt != receipt
        or complete_v3.complete_corpus_identity_v2.complete_corpus_membership_v1
        != complete_corpus_identity
    ):
        raise ValueError("condition_v8_complete_corpus_input_mismatch")
    binding_v3 = freeze_manifest_corpus_policy_binding_v3(
        claim_manifest_sha256=hash_canonical(claim_manifest),
        complete_corpus_identity_v3=complete_v3,
        policy_sha256=_certificate_manifest_policy_sha256(claim_manifest),
    )
    if any(stage.stage == _V8_COMPOSITION_STAGE for stage in lineage):
        raise ValueError("condition_v8_duplicate_composition_lineage_stage")
    if any(
        issue.get("code") == "corpus_pipeline_identity_mismatch"
        for issue in adapter_issues
    ) or "adapter:corpus_pipeline_identity_mismatch" in reasons:
        raise ValueError("condition_v8_pipeline_mismatch_blocker_must_be_removed")
    shadow = freeze_condition_verification_certificate_v6(
        generated_at=generated_at,
        reasons=reasons,
        claim_manifest=claim_manifest,
        corpus=_verification_v8_shadow_corpus(corpus=corpus, receipt=receipt),
        corpus_sha256=corpus_sha256,
        source_evidence_graph=source_evidence_graph,
        current_full_evidence_graph=current_full_evidence_graph,
        development_evidence_graph=development_evidence_graph,
        adapter_issues=adapter_issues,
        synthesis=synthesis,
        counterfactual_reruns=counterfactual_reruns,
        audit_candidates=audit_candidates,
        release_assessment=release_assessment,
        pipeline_verification=pipeline_verification,
        complete_corpus_identity=complete_corpus_identity,
        item_risk_scoring_receipt=item_risk_scoring_receipt,
        condition_plan=condition_plan,
        condition_frozen_model=condition_frozen_model,
        condition_calibration_projection=condition_calibration_projection,
        condition_confirmation_gate=condition_confirmation_gate,
        condition_target_semantics=condition_target_semantics,
        condition_independence_identity=condition_independence_identity,
        condition_gate_invocation_proof=condition_gate_invocation_proof,
        release_qualification_proof=release_qualification_proof,
        adaptive_policy_context=adaptive_policy_context,
        adaptive_calibration_bundle_v2=adaptive_calibration_bundle_v2,
        adaptive_release_candidate_v2=adaptive_release_candidate_v2,
        adaptive_prospective_assessment_v2=adaptive_prospective_assessment_v2,
        sequential_audit_state=sequential_audit_state,
        production_stop_decision=production_stop_decision,
        lineage=lineage,
    )
    payload = shadow.model_dump(mode="json", exclude={"certificate_sha256"})
    payload.update(
        {
            "certificate_version": (
                "literature-multiverse-condition-verification-v8"
            ),
            "corpus": corpus,
            "lineage": [
                *lineage,
                _verification_v8_composition_stage(
                    receipt=receipt,
                    complete_corpus_identity_v3=complete_v3,
                    manifest_corpus_policy_binding_v3=binding_v3,
                ),
            ],
            "composition_external_replay_receipt": receipt,
            "composition_external_replay_receipt_sha256": receipt.receipt_sha256,
            "complete_corpus_identity_v3": complete_v3,
            "complete_corpus_membership_v3_sha256": (
                complete_v3.membership_composition_sha256
            ),
            "manifest_corpus_policy_binding_v3": binding_v3,
            "manifest_corpus_policy_binding_v3_sha256": (
                binding_v3.manifest_corpus_policy_binding_sha256
            ),
            "cleared_adapter_issue_code": "corpus_pipeline_identity_mismatch",
            "v6_common_contract_replay_sha256": shadow.certificate_sha256,
        }
    )
    run_identity = hash_canonical(
        {
            "v6_common_contract_replay_sha256": shadow.certificate_sha256,
            "composition_external_replay_receipt_sha256": receipt.receipt_sha256,
            "complete_corpus_membership_v3_sha256": (
                complete_v3.membership_composition_sha256
            ),
            "manifest_corpus_policy_binding_v3_sha256": (
                binding_v3.manifest_corpus_policy_binding_sha256
            ),
        }
    )
    payload["run_id"] = f"verify-condition-v8-{run_identity[:16]}"
    return ConditionVerificationCertificateV8.model_validate(
        {**payload, "certificate_sha256": hash_canonical(payload)}
    )


class FinalConditionReleaseAssessmentV1(ContractModel):
    """Exact final v7 join of immutable v6 science and v2 calibrated release."""

    assessment_version: Literal["final-condition-release-v1"] = (
        "final-condition-release-v1"
    )
    question_id: Annotated[str, Field(min_length=1)]
    source_v6_certificate_sha256: str
    source_v6_decision_sha256: str
    terminal_gate_result_sha256: str
    adaptive_bundle_sha256: str
    policy_context_sha256: str
    selected_threshold_candidate_sha256: str | None
    adaptive_candidate_sha256: str
    adaptive_prospective_assessment_sha256: str
    released_claim_decision: Literal["condition_dependent"] | None = None
    status: Literal["released", "abstained"]
    reasons: list[str]
    decision_sha256: str
    release_semantics: Literal[
        "held-out predictive association plus confirmation-aware calibrated "
        "literature-support release; not causal proof or scientific truth"
    ] = (
        "held-out predictive association plus confirmation-aware calibrated "
        "literature-support release; not causal proof or scientific truth"
    )

    @field_validator(
        "source_v6_certificate_sha256",
        "source_v6_decision_sha256",
        "terminal_gate_result_sha256",
        "adaptive_bundle_sha256",
        "policy_context_sha256",
        "selected_threshold_candidate_sha256",
        "adaptive_candidate_sha256",
        "adaptive_prospective_assessment_sha256",
        "decision_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("final_condition_release_sha256_invalid")
        return value

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item for item in value):
            raise ValueError("final_condition_release_reasons_invalid")
        return value

    @model_validator(mode="after")
    def validate_decision(self) -> FinalConditionReleaseAssessmentV1:
        released = self.status == "released"
        if released != (self.released_claim_decision == "condition_dependent"):
            raise ValueError("final_condition_release_decision_status_mismatch")
        if released == bool(self.reasons):
            raise ValueError("final_condition_release_reason_status_mismatch")
        payload = self.model_dump(mode="json", exclude={"decision_sha256"})
        if hash_canonical(payload) != self.decision_sha256:
            raise ValueError("final_condition_release_decision_hash_mismatch")
        return self


def freeze_final_condition_release_assessment_v1(
    *,
    source_certificate: ConditionVerificationCertificateV6,
    terminal_gate_result: ConditionTerminalGateResultV2,
    adaptive_candidate: ProspectiveAdaptiveReleaseCandidateV2,
    adaptive_assessment: AdaptiveProspectiveAssessmentV2,
) -> FinalConditionReleaseAssessmentV1:
    released = (
        terminal_gate_result.passed
        and adaptive_assessment.status == "released"
        and adaptive_assessment.released_claim_decision == "condition_dependent"
    )
    reasons = [] if released else [f"adaptive:{adaptive_assessment.reason}"]
    payload: dict[str, Any] = {
        "assessment_version": "final-condition-release-v1",
        "question_id": source_certificate.release_assessment.question_id,
        "source_v6_certificate_sha256": source_certificate.certificate_sha256,
        "source_v6_decision_sha256": (
            source_certificate.release_assessment.decision_sha256
        ),
        "terminal_gate_result_sha256": terminal_gate_result.result_sha256,
        "adaptive_bundle_sha256": (
            source_certificate.adaptive_calibration_bundle_v2.bundle_sha256
        ),
        "policy_context_sha256": (
            source_certificate.adaptive_policy_context.policy_context_sha256
        ),
        "selected_threshold_candidate_sha256": (
            adaptive_assessment.threshold_candidate_sha256
        ),
        "adaptive_candidate_sha256": adaptive_candidate.candidate_sha256,
        "adaptive_prospective_assessment_sha256": (
            adaptive_assessment.assessment_sha256
        ),
        "released_claim_decision": "condition_dependent" if released else None,
        "status": "released" if released else "abstained",
        "reasons": reasons,
        "release_semantics": (
            "held-out predictive association plus confirmation-aware calibrated "
            "literature-support release; not causal proof or scientific truth"
        ),
    }
    return FinalConditionReleaseAssessmentV1.model_validate(
        {**payload, "decision_sha256": hash_canonical(payload)}
    )


class FinalConditionVerificationCertificateV7(ContractModel):
    """Final certificate formed only after v6 has an immutable source hash."""

    certificate_version: Literal[
        "literature-multiverse-condition-verification-v7"
    ] = "literature-multiverse-condition-verification-v7"
    run_id: Annotated[str, Field(pattern=r"^verify-condition-v7-[0-9a-f]{16}$")]
    generated_at: datetime
    status: Literal["released", "abstained"]
    reasons: list[str]
    source_certificate_v6: ConditionVerificationCertificateV6
    source_v6_certificate_sha256: str
    condition_confirmation_assessment: ConditionConfirmationAssessmentV1
    condition_confirmation_gate: ConditionConfirmationGateAssessmentV1
    terminal_gate_result: ConditionTerminalGateResultV2
    terminal_gate_result_sha256: str
    adaptive_release_candidate_v2: ProspectiveAdaptiveReleaseCandidateV2
    adaptive_prospective_assessment_v2: AdaptiveProspectiveAssessmentV2
    release_assessment: FinalConditionReleaseAssessmentV1
    certificate_sha256: str
    interpretation: Literal[
        "risk-controlled literature-support decision for a predictive association; "
        "not causal proof, scientific truth, or domain-shift robustness"
    ] = (
        "risk-controlled literature-support decision for a predictive association; "
        "not causal proof, scientific truth, or domain-shift robustness"
    )

    @field_validator(
        "source_v6_certificate_sha256",
        "terminal_gate_result_sha256",
        "certificate_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("condition_v7_sha256_invalid")
        return value

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item for item in value):
            raise ValueError("condition_v7_reasons_invalid")
        return value

    @model_validator(mode="after")
    def validate_certificate(self) -> FinalConditionVerificationCertificateV7:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("condition_v7_generated_at_requires_timezone")
        try:
            source = ConditionVerificationCertificateV6.model_validate(
                self.source_certificate_v6.model_dump(mode="json")
            )
        except ValueError as exc:
            raise ValueError("condition_v7_source_v6_replay_failed") from exc
        if self.generated_at < source.generated_at:
            raise ValueError("condition_v7_generated_at_precedes_source_v6")
        gate_result = self.terminal_gate_result
        invocation = source.condition_gate_invocation_proof
        if (
            invocation is None
            or source.production_stop_decision.outcome != "condition_gate_ready"
            or source.condition_confirmation_gate.status != "missing"
            or source.condition_confirmation_assessment is not None
        ):
            raise ValueError("condition_v7_source_not_outcome_free_gate_ready")
        pipeline_sha256 = source.pipeline_verification.computed_pipeline_sha256
        assert pipeline_sha256 is not None
        try:
            assessment = validate_condition_confirmation_assessment(
                plan=source.condition_plan,
                model=source.condition_frozen_model,
                full_graph=source.current_full_evidence_graph,
                assessment=self.condition_confirmation_assessment,
                current_pipeline_sha256=pipeline_sha256,
            )
        except (ConditionConfirmationError, ValueError) as exc:
            raise ValueError("condition_v7_assessment_replay_failed") from exc
        expected_gate = freeze_condition_confirmation_gate_assessment(
            provisional_claim_decision="condition_dependent",
            status=assessment.status,
            reasons=assessment.reasons,
            condition_projection_sha256=(
                source.condition_calibration_projection.projection_sha256
            ),
            target_sha256=(
                source.condition_calibration_projection.condition_target_sha256
            ),
            plan_sha256=source.condition_plan.plan_sha256,
            config_sha256=source.condition_plan.config_sha256,
            model_sha256=source.condition_frozen_model.model_sha256,
            assessment_sha256=assessment.assessment_sha256,
        )
        if (
            assessment != self.condition_confirmation_assessment
            or self.condition_confirmation_gate != expected_gate
        ):
            raise ValueError("condition_v7_gate_assessment_mismatch")
        expected_gate_result = freeze_condition_terminal_gate_result_v2(
            question_id=source.release_assessment.question_id,
            policy_arm_id=source.adaptive_policy_context.policy_arm_id,
            condition_gate_invocation_proof=invocation,
            gate_assessment=expected_gate,
            source_v6_certificate_sha256=source.certificate_sha256,
            source_v6_decision_sha256=source.release_assessment.decision_sha256,
        )
        if (
            self.source_v6_certificate_sha256 != source.certificate_sha256
            or self.terminal_gate_result_sha256 != gate_result.result_sha256
            or gate_result != expected_gate_result
            or gate_result.source_v6_certificate_sha256 != source.certificate_sha256
            or gate_result.source_v6_decision_sha256
            != source.release_assessment.decision_sha256
            or gate_result.gate_assessment != expected_gate
            or gate_result.condition_projection != source.condition_calibration_projection
            or gate_result.condition_gate_invocation_proof_sha256
            != invocation.proof_sha256
        ):
            raise ValueError("condition_v7_source_gate_lineage_mismatch")
        expected_candidate = freeze_prospective_adaptive_candidate_v2(
            base_candidate=source.adaptive_release_candidate_v2.base_candidate,
            target_semantics=source.condition_target_semantics,
            independence_identity=source.condition_independence_identity,
            condition_projection=source.condition_calibration_projection,
            condition_gate_invocation_proof=(
                source.condition_gate_invocation_proof
            ),
            release_qualification_proof=source.release_qualification_proof,
            terminal_gate_result=gate_result,
        )
        if expected_candidate != self.adaptive_release_candidate_v2:
            raise ValueError("condition_v7_candidate_substitution")
        expected_assessment = assess_confirmation_aware_adaptive_release_candidate(
            expected_candidate,
            source.adaptive_calibration_bundle_v2,
        )
        if expected_assessment != self.adaptive_prospective_assessment_v2:
            raise ValueError("condition_v7_adaptive_assessment_mismatch")
        expected_release = freeze_final_condition_release_assessment_v1(
            source_certificate=source,
            terminal_gate_result=gate_result,
            adaptive_candidate=expected_candidate,
            adaptive_assessment=expected_assessment,
        )
        if expected_release != self.release_assessment:
            raise ValueError("condition_v7_release_assessment_mismatch")
        if (
            self.status != expected_release.status
            or self.reasons != expected_release.reasons
        ):
            raise ValueError("condition_v7_status_mismatch")
        payload = self.model_dump(mode="json", exclude={"certificate_sha256"})
        if hash_canonical(payload) != self.certificate_sha256:
            raise ValueError("condition_v7_certificate_hash_mismatch")
        return self


def freeze_final_condition_verification_certificate_v7(
    *,
    generated_at: datetime,
    source_certificate: ConditionVerificationCertificateV6,
    condition_confirmation_assessment: ConditionConfirmationAssessmentV1,
) -> FinalConditionVerificationCertificateV7:
    """Join an exact outcome-free v6 to one replayed held-out assessment.

    This is the only supported v7 constructor.  It accepts neither caller-authored
    gate objects nor terminal-result hashes, so a held-out outcome cannot rewrite the
    online prefix, action roster, qualification proof, bundle, or source decision.
    """

    try:
        source_certificate = ConditionVerificationCertificateV6.model_validate(
            source_certificate.model_dump(mode="json")
        )
        condition_confirmation_assessment = (
            ConditionConfirmationAssessmentV1.model_validate(
                condition_confirmation_assessment.model_dump(mode="json")
            )
        )
    except (AttributeError, ValueError) as exc:
        raise ValueError("condition_v7_finalizer_input_integrity_changed") from exc
    invocation = source_certificate.condition_gate_invocation_proof
    if (
        invocation is None
        or source_certificate.production_stop_decision.outcome
        != "condition_gate_ready"
        or source_certificate.condition_confirmation_gate.status != "missing"
        or source_certificate.condition_confirmation_assessment is not None
    ):
        raise ValueError("condition_v7_source_not_outcome_free_gate_ready")
    pipeline_sha256 = source_certificate.pipeline_verification.computed_pipeline_sha256
    assert pipeline_sha256 is not None
    try:
        replayed_assessment = validate_condition_confirmation_assessment(
            plan=source_certificate.condition_plan,
            model=source_certificate.condition_frozen_model,
            full_graph=source_certificate.current_full_evidence_graph,
            assessment=condition_confirmation_assessment,
            current_pipeline_sha256=pipeline_sha256,
        )
    except (ConditionConfirmationError, ValueError) as exc:
        raise ValueError("condition_v7_assessment_replay_failed") from exc
    gate = freeze_condition_confirmation_gate_assessment(
        provisional_claim_decision="condition_dependent",
        status=replayed_assessment.status,
        reasons=replayed_assessment.reasons,
        condition_projection_sha256=(
            source_certificate.condition_calibration_projection.projection_sha256
        ),
        target_sha256=(
            source_certificate.condition_calibration_projection.condition_target_sha256
        ),
        plan_sha256=source_certificate.condition_plan.plan_sha256,
        config_sha256=source_certificate.condition_plan.config_sha256,
        model_sha256=source_certificate.condition_frozen_model.model_sha256,
        assessment_sha256=replayed_assessment.assessment_sha256,
    )
    terminal_gate_result = freeze_condition_terminal_gate_result_v2(
        question_id=source_certificate.release_assessment.question_id,
        policy_arm_id=source_certificate.adaptive_policy_context.policy_arm_id,
        condition_gate_invocation_proof=invocation,
        gate_assessment=gate,
        source_v6_certificate_sha256=source_certificate.certificate_sha256,
        source_v6_decision_sha256=(
            source_certificate.release_assessment.decision_sha256
        ),
    )
    candidate = freeze_prospective_adaptive_candidate_v2(
        base_candidate=source_certificate.adaptive_release_candidate_v2.base_candidate,
        target_semantics=source_certificate.condition_target_semantics,
        independence_identity=source_certificate.condition_independence_identity,
        condition_projection=source_certificate.condition_calibration_projection,
        condition_gate_invocation_proof=(
            source_certificate.condition_gate_invocation_proof
        ),
        release_qualification_proof=source_certificate.release_qualification_proof,
        terminal_gate_result=terminal_gate_result,
    )
    adaptive_assessment = assess_confirmation_aware_adaptive_release_candidate(
        candidate,
        source_certificate.adaptive_calibration_bundle_v2,
    )
    release = freeze_final_condition_release_assessment_v1(
        source_certificate=source_certificate,
        terminal_gate_result=terminal_gate_result,
        adaptive_candidate=candidate,
        adaptive_assessment=adaptive_assessment,
    )
    generated = generated_at.isoformat().replace("+00:00", "Z")
    identity = hash_canonical(
        {
            "source_v6_certificate_sha256": source_certificate.certificate_sha256,
            "condition_confirmation_assessment_sha256": (
                replayed_assessment.assessment_sha256
            ),
            "terminal_gate_result_sha256": terminal_gate_result.result_sha256,
            "release_decision_sha256": release.decision_sha256,
        }
    )
    payload: dict[str, Any] = {
        "certificate_version": "literature-multiverse-condition-verification-v7",
        "run_id": f"verify-condition-v7-{identity[:16]}",
        "generated_at": generated,
        "status": release.status,
        "reasons": release.reasons,
        "source_certificate_v6": source_certificate,
        "source_v6_certificate_sha256": source_certificate.certificate_sha256,
        "condition_confirmation_assessment": replayed_assessment,
        "condition_confirmation_gate": gate,
        "terminal_gate_result": terminal_gate_result,
        "terminal_gate_result_sha256": terminal_gate_result.result_sha256,
        "adaptive_release_candidate_v2": candidate,
        "adaptive_prospective_assessment_v2": adaptive_assessment,
        "release_assessment": release,
        "interpretation": (
            "risk-controlled literature-support decision for a predictive association; "
            "not causal proof, scientific truth, or domain-shift robustness"
        ),
    }
    return FinalConditionVerificationCertificateV7.model_validate(
        {**payload, "certificate_sha256": hash_canonical(payload)}
    )


def _condition_v8_shadow_v6(
    source: ConditionVerificationCertificateV8,
) -> ConditionVerificationCertificateV6:
    """Replay the exact mature v6 projection embedded by a condition-v8 source."""

    source = ConditionVerificationCertificateV8.model_validate(
        source.model_dump(mode="json")
    )
    receipt = source.composition_external_replay_receipt
    shadow = freeze_condition_verification_certificate_v6(
        generated_at=source.generated_at,
        reasons=source.reasons,
        claim_manifest=source.claim_manifest,
        corpus=_verification_v8_shadow_corpus(
            corpus=source.corpus,
            receipt=receipt,
        ),
        corpus_sha256=source.corpus_sha256,
        source_evidence_graph=source.source_evidence_graph,
        current_full_evidence_graph=source.current_full_evidence_graph,
        development_evidence_graph=source.development_evidence_graph,
        adapter_issues=source.adapter_issues,
        synthesis=source.synthesis,
        counterfactual_reruns=source.counterfactual_reruns,
        audit_candidates=source.audit_candidates,
        release_assessment=source.release_assessment,
        pipeline_verification=source.pipeline_verification,
        complete_corpus_identity=source.complete_corpus_identity,
        item_risk_scoring_receipt=source.item_risk_scoring_receipt,
        condition_plan=source.condition_plan,
        condition_frozen_model=source.condition_frozen_model,
        condition_calibration_projection=source.condition_calibration_projection,
        condition_confirmation_gate=source.condition_confirmation_gate,
        condition_target_semantics=source.condition_target_semantics,
        condition_independence_identity=source.condition_independence_identity,
        condition_gate_invocation_proof=source.condition_gate_invocation_proof,
        release_qualification_proof=source.release_qualification_proof,
        adaptive_policy_context=source.adaptive_policy_context,
        adaptive_calibration_bundle_v2=source.adaptive_calibration_bundle_v2,
        adaptive_release_candidate_v2=source.adaptive_release_candidate_v2,
        adaptive_prospective_assessment_v2=(
            source.adaptive_prospective_assessment_v2
        ),
        sequential_audit_state=source.sequential_audit_state,
        production_stop_decision=source.production_stop_decision,
        lineage=source.lineage[:-1],
    )
    if shadow.certificate_sha256 != source.v6_common_contract_replay_sha256:
        raise ValueError("condition_v8_shadow_v6_hash_mismatch")
    return shadow


class ConditionCompositionTerminalJoinV1(ContractModel):
    """Exact bridge from immutable condition v8 to its mature v7 replay witness."""

    join_version: Literal["condition-composition-terminal-join-v1"] = (
        "condition-composition-terminal-join-v1"
    )
    source_v8_certificate_sha256: str
    source_v8_decision_sha256: str
    source_v8_composition_receipt_sha256: str
    source_v8_complete_corpus_membership_v3_sha256: str
    v6_common_contract_replay_sha256: str
    v7_common_contract_replay_sha256: str
    terminal_gate_result_sha256: str
    release_decision_sha256: str
    join_sha256: str

    @field_validator(
        "source_v8_certificate_sha256",
        "source_v8_decision_sha256",
        "source_v8_composition_receipt_sha256",
        "source_v8_complete_corpus_membership_v3_sha256",
        "v6_common_contract_replay_sha256",
        "v7_common_contract_replay_sha256",
        "terminal_gate_result_sha256",
        "release_decision_sha256",
        "join_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("condition_composition_terminal_join_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_join(self) -> ConditionCompositionTerminalJoinV1:
        payload = self.model_dump(mode="json", exclude={"join_sha256"})
        if self.join_sha256 != hash_canonical(payload):
            raise ValueError("condition_composition_terminal_join_hash_mismatch")
        return self


def freeze_condition_composition_terminal_join_v1(
    *,
    source_v8: ConditionVerificationCertificateV8,
    shadow_v6: ConditionVerificationCertificateV6,
    shadow_v7: FinalConditionVerificationCertificateV7,
) -> ConditionCompositionTerminalJoinV1:
    payload: dict[str, Any] = {
        "join_version": "condition-composition-terminal-join-v1",
        "source_v8_certificate_sha256": source_v8.certificate_sha256,
        "source_v8_decision_sha256": source_v8.release_assessment.decision_sha256,
        "source_v8_composition_receipt_sha256": (
            source_v8.composition_external_replay_receipt_sha256
        ),
        "source_v8_complete_corpus_membership_v3_sha256": (
            source_v8.complete_corpus_membership_v3_sha256
        ),
        "v6_common_contract_replay_sha256": shadow_v6.certificate_sha256,
        "v7_common_contract_replay_sha256": shadow_v7.certificate_sha256,
        "terminal_gate_result_sha256": shadow_v7.terminal_gate_result_sha256,
        "release_decision_sha256": shadow_v7.release_assessment.decision_sha256,
    }
    return ConditionCompositionTerminalJoinV1.model_validate(
        {**payload, "join_sha256": hash_canonical(payload)}
    )


class FinalConditionVerificationCertificateV9(ContractModel):
    """Terminal held-out join over an immutable receipt-bound condition-v8 source."""

    certificate_version: Literal[
        "literature-multiverse-condition-verification-v9"
    ] = "literature-multiverse-condition-verification-v9"
    run_id: Annotated[str, Field(pattern=r"^verify-condition-v9-[0-9a-f]{16}$")]
    generated_at: datetime
    status: Literal["released", "abstained"]
    reasons: list[str]
    source_certificate_v8: ConditionVerificationCertificateV8
    source_v8_certificate_sha256: str
    v6_common_contract_replay_sha256: str
    v7_common_contract_replay_sha256: str
    condition_confirmation_assessment: ConditionConfirmationAssessmentV1
    condition_confirmation_gate: ConditionConfirmationGateAssessmentV1
    terminal_gate_result: ConditionTerminalGateResultV2
    terminal_gate_result_sha256: str
    adaptive_release_candidate_v2: ProspectiveAdaptiveReleaseCandidateV2
    adaptive_prospective_assessment_v2: AdaptiveProspectiveAssessmentV2
    release_assessment: FinalConditionReleaseAssessmentV1
    composition_terminal_join: ConditionCompositionTerminalJoinV1
    composition_terminal_join_sha256: str
    certificate_sha256: str
    interpretation: Literal[
        "risk-controlled literature-support decision for a predictive association "
        "under an externally replayed composed pipeline; not causal proof, scientific "
        "truth, or domain-shift robustness"
    ] = (
        "risk-controlled literature-support decision for a predictive association "
        "under an externally replayed composed pipeline; not causal proof, scientific "
        "truth, or domain-shift robustness"
    )

    @field_validator(
        "source_v8_certificate_sha256",
        "v6_common_contract_replay_sha256",
        "v7_common_contract_replay_sha256",
        "terminal_gate_result_sha256",
        "composition_terminal_join_sha256",
        "certificate_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("condition_v9_sha256_invalid")
        return value

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item for item in value):
            raise ValueError("condition_v9_reasons_invalid")
        return value

    @model_validator(mode="after")
    def validate_certificate(self) -> FinalConditionVerificationCertificateV9:
        try:
            source = ConditionVerificationCertificateV8.model_validate(
                self.source_certificate_v8.model_dump(mode="json")
            )
            shadow_v6 = _condition_v8_shadow_v6(source)
            shadow_v7 = freeze_final_condition_verification_certificate_v7(
                generated_at=self.generated_at,
                source_certificate=shadow_v6,
                condition_confirmation_assessment=(
                    self.condition_confirmation_assessment
                ),
            )
            join = freeze_condition_composition_terminal_join_v1(
                source_v8=source,
                shadow_v6=shadow_v6,
                shadow_v7=shadow_v7,
            )
        except (AttributeError, ValueError) as exc:
            raise ValueError("condition_v9_external_replay_failed") from exc
        if self.generated_at < source.generated_at:
            raise ValueError("condition_v9_generated_at_precedes_source_v8")
        aliases: tuple[tuple[str, Any, Any], ...] = (
            (
                "source_v8_certificate_sha256",
                self.source_v8_certificate_sha256,
                source.certificate_sha256,
            ),
            (
                "v6_common_contract_replay_sha256",
                self.v6_common_contract_replay_sha256,
                shadow_v6.certificate_sha256,
            ),
            (
                "v7_common_contract_replay_sha256",
                self.v7_common_contract_replay_sha256,
                shadow_v7.certificate_sha256,
            ),
            (
                "condition_confirmation_gate",
                self.condition_confirmation_gate,
                shadow_v7.condition_confirmation_gate,
            ),
            (
                "terminal_gate_result",
                self.terminal_gate_result,
                shadow_v7.terminal_gate_result,
            ),
            (
                "terminal_gate_result_sha256",
                self.terminal_gate_result_sha256,
                shadow_v7.terminal_gate_result_sha256,
            ),
            (
                "adaptive_release_candidate_v2",
                self.adaptive_release_candidate_v2,
                shadow_v7.adaptive_release_candidate_v2,
            ),
            (
                "adaptive_prospective_assessment_v2",
                self.adaptive_prospective_assessment_v2,
                shadow_v7.adaptive_prospective_assessment_v2,
            ),
            (
                "release_assessment",
                self.release_assessment,
                shadow_v7.release_assessment,
            ),
            ("status", self.status, shadow_v7.status),
            ("reasons", self.reasons, shadow_v7.reasons),
            ("composition_terminal_join", self.composition_terminal_join, join),
            (
                "composition_terminal_join_sha256",
                self.composition_terminal_join_sha256,
                join.join_sha256,
            ),
        )
        for field, observed, expected in aliases:
            if observed != expected:
                raise ValueError(f"condition_v9_{field}_alias_mismatch")
        run_identity = hash_canonical(
            {
                "source_v8_certificate_sha256": source.certificate_sha256,
                "v7_common_contract_replay_sha256": shadow_v7.certificate_sha256,
                "composition_terminal_join_sha256": join.join_sha256,
            }
        )
        if self.run_id != f"verify-condition-v9-{run_identity[:16]}":
            raise ValueError("condition_v9_run_identity_mismatch")
        payload = self.model_dump(mode="json", exclude={"certificate_sha256"})
        if self.certificate_sha256 != hash_canonical(payload):
            raise ValueError("condition_v9_certificate_hash_mismatch")
        return self


def freeze_final_condition_verification_certificate_v9(
    *,
    generated_at: datetime,
    source_certificate: ConditionVerificationCertificateV8,
    condition_confirmation_assessment: ConditionConfirmationAssessmentV1,
) -> FinalConditionVerificationCertificateV9:
    """Open one terminal outcome only after receipt-bound v8 is immutable."""

    source = ConditionVerificationCertificateV8.model_validate(
        source_certificate.model_dump(mode="json")
    )
    shadow_v6 = _condition_v8_shadow_v6(source)
    shadow_v7 = freeze_final_condition_verification_certificate_v7(
        generated_at=generated_at,
        source_certificate=shadow_v6,
        condition_confirmation_assessment=condition_confirmation_assessment,
    )
    join = freeze_condition_composition_terminal_join_v1(
        source_v8=source,
        shadow_v6=shadow_v6,
        shadow_v7=shadow_v7,
    )
    payload: dict[str, Any] = {
        "certificate_version": "literature-multiverse-condition-verification-v9",
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "status": shadow_v7.status,
        "reasons": shadow_v7.reasons,
        "source_certificate_v8": source,
        "source_v8_certificate_sha256": source.certificate_sha256,
        "v6_common_contract_replay_sha256": shadow_v6.certificate_sha256,
        "v7_common_contract_replay_sha256": shadow_v7.certificate_sha256,
        "condition_confirmation_assessment": (
            shadow_v7.condition_confirmation_assessment
        ),
        "condition_confirmation_gate": shadow_v7.condition_confirmation_gate,
        "terminal_gate_result": shadow_v7.terminal_gate_result,
        "terminal_gate_result_sha256": shadow_v7.terminal_gate_result_sha256,
        "adaptive_release_candidate_v2": shadow_v7.adaptive_release_candidate_v2,
        "adaptive_prospective_assessment_v2": (
            shadow_v7.adaptive_prospective_assessment_v2
        ),
        "release_assessment": shadow_v7.release_assessment,
        "composition_terminal_join": join,
        "composition_terminal_join_sha256": join.join_sha256,
        "interpretation": (
            "risk-controlled literature-support decision for a predictive association "
            "under an externally replayed composed pipeline; not causal proof, scientific "
            "truth, or domain-shift robustness"
        ),
    }
    run_identity = hash_canonical(
        {
            "source_v8_certificate_sha256": source.certificate_sha256,
            "v7_common_contract_replay_sha256": shadow_v7.certificate_sha256,
            "composition_terminal_join_sha256": join.join_sha256,
        }
    )
    payload["run_id"] = f"verify-condition-v9-{run_identity[:16]}"
    return FinalConditionVerificationCertificateV9.model_validate(
        {**payload, "certificate_sha256": hash_canonical(payload)}
    )


def _cell(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return html.escape(str(value))


def _eligibility_rows(certificate: VerificationCertificate) -> str:
    rows = certificate.corpus.get("eligibility", [])
    if not isinstance(rows, list) or not rows:
        return '<tr><td colspan="4">No paper-level eligibility ledger was supplied.</td></tr>'
    rendered: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rendered.append(
            "<tr>"
            f"<td>{_cell(row.get('paper_id'))}</td>"
            f"<td>{_cell(row.get('title'))}</td>"
            f"<td>{_cell(row.get('status'))}</td>"
            f"<td>{_cell(row.get('reason'))}</td>"
            "</tr>"
        )
    return "".join(rendered)


def _estimate_rows(certificate: VerificationCertificate) -> str:
    span_by_id = {span.span_id: span for span in certificate.evidence_graph.evidence_spans}
    rows: list[str] = []
    for estimate in certificate.evidence_graph.outcome_estimates:
        first_span = span_by_id[estimate.evidence_span_ids[0]]
        rows.append(
            "<tr>"
            f"<td>{_cell(estimate.estimate_id)}</td>"
            f"<td>{_cell(estimate.effect.paper_id)}</td>"
            f"<td>{_cell(estimate.outcome_name)}</td>"
            f"<td>{_cell(estimate.effect.estimate)}</td>"
            f"<td>{_cell(estimate.effect.effect_format)}</td>"
            f"<td>{_cell(first_span.source_locator)}</td>"
            f"<td>{_cell(first_span.quote)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _audit_rows(certificate: VerificationCertificate) -> str:
    rows: list[str] = []
    for item in certificate.release_assessment.audit.ranking:
        rows.append(
            "<tr>"
            f"<td>{item.rank}</td>"
            f"<td>{_cell(item.item_id)}</td>"
            f"<td>{_cell(item.selected_for_audit)}</td>"
            f"<td>{_cell(item.resolved_before_release)}</td>"
            f"<td>{item.probability_influence:.6f}</td>"
            f"<td>{item.expected_claim_loss_reduction_per_cost:.6f}</td>"
            "</tr>"
        )
    if not rows:
        return '<tr><td colspan="6">No matching evidence items were available to rank.</td></tr>'
    return "".join(rows)


def _render_condition_certificate_html(
    certificate: ConditionVerificationCertificateV6
    | ConditionVerificationCertificateV8
    | FinalConditionVerificationCertificateV7
    | FinalConditionVerificationCertificateV9,
) -> str:
    """Render the v6/v7 condition lineage without opening remote resources."""

    if isinstance(certificate, ConditionVerificationCertificateV6):
        source = certificate
    elif isinstance(certificate, FinalConditionVerificationCertificateV9):
        source = certificate.source_certificate_v8
    else:
        source = certificate.source_certificate_v6
    gate = (
        source.condition_confirmation_gate
        if isinstance(certificate, ConditionVerificationCertificateV6)
        else certificate.condition_confirmation_gate
    )
    reason_items = "".join(
        f"<li>{_cell(reason)}</li>" for reason in certificate.reasons
    ) or "<li>All declared release gates passed.</li>"
    canonical = json.dumps(
        certificate.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    status_class = "released" if certificate.status == "released" else "abstained"
    terminal_result = (
        None
        if isinstance(certificate, ConditionVerificationCertificateV6)
        else certificate.terminal_gate_result
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Condition verification certificate — {html.escape(certificate.run_id)}</title>
  <style>
    :root {{ color-scheme: light dark; --ok:#0b6b42; --stop:#9b2c2c; --line:#8b8b8b55; }}
    body {{ font-family:ui-sans-serif,system-ui,sans-serif; line-height:1.45; margin:0 auto;
            max-width:1100px; padding:2rem; }}
    .banner {{ border-left:.55rem solid var(--stop); padding:1rem 1.2rem; background:#9b2c2c12; }}
    .banner.released {{ border-color:var(--ok); background:#0b6b4212; }}
    table {{ width:100%; border-collapse:collapse; }}
    th,td {{ border:1px solid var(--line); padding:.5rem; text-align:left; vertical-align:top; }}
    pre {{ overflow:auto; max-height:48rem; border:1px solid var(--line); padding:1rem; }}
    .hash {{ overflow-wrap:anywhere; }}
  </style>
</head>
<body>
  <div class="banner {status_class}">
    <h1>{html.escape(certificate.status.upper())}</h1>
    <p>Predictive literature-association verification; not causal proof or scientific truth.</p>
  </div>
  <h2>Decision and blockers</h2><ul>{reason_items}</ul>
  <table>
    <tr><th>Certificate contract</th><td>{_cell(certificate.certificate_version)}</td></tr>
    <tr><th>Held-out confirmation status</th><td>{_cell(gate.status)}</td></tr>
    <tr><th>Scientific gate passed</th><td>{_cell(gate.scientific_gate_passed)}</td></tr>
    <tr><th>Development graph SHA-256</th><td class="hash">
      {_cell(source.development_evidence_graph_sha256)}</td></tr>
    <tr><th>Plan SHA-256</th><td class="hash">{_cell(source.condition_plan.plan_sha256)}</td></tr>
    <tr><th>Outcome-firewall receipt SHA-256</th><td class="hash">
      {_cell(source.condition_calibration_projection.firewall_receipt_sha256)}</td></tr>
    <tr><th>Source certificate SHA-256</th><td class="hash">
      {_cell(source.certificate_sha256)}</td></tr>
    <tr><th>Terminal gate-result SHA-256</th><td class="hash">
      {_cell(None if terminal_result is None else terminal_result.result_sha256)}</td></tr>
    <tr><th>Certificate SHA-256</th><td class="hash">
      {_cell(certificate.certificate_sha256)}</td></tr>
  </table>
  <details><summary><strong>Complete normative JSON payload</strong></summary>
    <pre>{html.escape(canonical)}</pre>
  </details>
</body>
</html>
"""


def render_certificate_html(
    certificate: VerificationCertificate
    | VerificationCertificateV8
    | ConditionVerificationCertificateV6
    | ConditionVerificationCertificateV8
    | FinalConditionVerificationCertificateV7
    | FinalConditionVerificationCertificateV9,
) -> str:
    """Render a static HTML view containing the complete canonical certificate JSON."""

    if isinstance(
        certificate,
        (
            ConditionVerificationCertificateV6,
            ConditionVerificationCertificateV8,
            FinalConditionVerificationCertificateV7,
            FinalConditionVerificationCertificateV9,
        ),
    ):
        return _render_condition_certificate_html(certificate)

    manifest_claim = certificate.claim_manifest.get("claim", {})
    if not isinstance(manifest_claim, dict):
        manifest_claim = {}
    evidence = certificate.release_assessment.evidence
    evidence_classification = getattr(evidence, "classification", None)
    if evidence_classification is None:
        evidence_classification = getattr(evidence, "state", "unknown")
    evidence_n_papers = getattr(
        evidence,
        "n_papers",
        len(certificate.release_assessment.paper_ids),
    )
    provenance_assurance = certificate.corpus.get("provenance_assurance", {})
    if not isinstance(provenance_assurance, dict):
        provenance_assurance = {}
    reason_items = "".join(f"<li>{_cell(reason)}</li>" for reason in certificate.reasons)
    if not reason_items:
        reason_items = "<li>All declared gates passed.</li>"
    canonical = json.dumps(
        certificate.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    status_class = "released" if certificate.status == "released" else "abstained"
    stop_decision = certificate.production_stop_decision
    selected_action = (
        stop_decision.selection_result.action
        if stop_decision.selection_result is not None
        else None
    )
    item_risk_receipt_sha256 = (
        None
        if certificate.item_risk_scoring_receipt is None
        else certificate.item_risk_scoring_receipt.receipt_sha256
    )
    adaptive_bundle_sha256 = (
        None
        if certificate.adaptive_calibration_bundle is None
        else certificate.adaptive_calibration_bundle.bundle_sha256
    )
    adaptive_assessment_sha256 = (
        None
        if certificate.adaptive_prospective_assessment is None
        else certificate.adaptive_prospective_assessment.assessment_sha256
    )
    adaptive_simultaneous_upper_risk = (
        None
        if certificate.adaptive_calibration_bundle is None
        or certificate.adaptive_calibration_bundle.selected is None
        else certificate.adaptive_calibration_bundle.selected.simultaneous_upper_risk
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Verification certificate — {html.escape(certificate.run_id)}</title>
  <style>
    :root {{ color-scheme: light dark; --ok:#0b6b42; --stop:#9b2c2c; --line:#8b8b8b55; }}
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; line-height:1.45; margin:0 auto;
            max-width:1200px; padding:2rem; }}
    .banner {{ border-left:.55rem solid var(--stop); padding:1rem 1.2rem; background:#9b2c2c12; }}
    .banner.released {{ border-color:var(--ok); background:#0b6b4212; }}
    h1,h2 {{ line-height:1.15; }} h2 {{ margin-top:2rem; }}
    table {{ width:100%; border-collapse:collapse; font-size:.9rem; }}
    th,td {{ border:1px solid var(--line); padding:.5rem; text-align:left; vertical-align:top; }}
    th {{ background:#80808014; }}
    code,pre {{ font-family:ui-monospace, SFMono-Regular, monospace; }}
    pre {{ overflow:auto; max-height:42rem; border:1px solid var(--line); padding:1rem; }}
    .hash {{ overflow-wrap:anywhere; }} .muted {{ opacity:.75; }}
  </style>
</head>
<body>
  <div class="banner {status_class}">
    <h1>{html.escape(certificate.status.upper())}</h1>
    <p><strong>{_cell(manifest_claim.get("statement"))}</strong></p>
    <p>This decision verifies support in the declared literature corpus; it is not a claim of
       scientific truth.</p>
  </div>

  <h2>Decision</h2>
  <ul>{reason_items}</ul>
  <table>
    <tr><th>Certificate contract</th><td>{_cell(certificate.certificate_version)}</td></tr>
    <tr><th>Evidence classification</th><td>{_cell(evidence_classification)}</td></tr>
    <tr><th>Synthesis mode</th><td>{_cell(evidence.mode)}</td></tr>
    <tr><th>Papers synthesized</th><td>{evidence_n_papers}</td></tr>
    <tr><th>Audit budget / spent</th><td>{certificate.release_assessment.audit.budget:g} /
      {certificate.release_assessment.audit.spent:g}
      {_cell(certificate.release_assessment.audit.cost_unit)}</td></tr>
    <tr><th>Calibration</th><td>{_cell(certificate.release_assessment.calibration.status)} —
      {_cell(certificate.release_assessment.calibration.reason)}</td></tr>
    <tr><th>Adaptive simultaneous upper reference-loss bound</th><td>
      {_cell(adaptive_simultaneous_upper_risk)}</td></tr>
    <tr><th>Corpus provenance</th><td>{_cell(provenance_assurance.get("status"))} —
      {_cell(provenance_assurance.get("reason"))}</td></tr>
    <tr><th>Provenance release-eligible</th><td>
      {_cell(provenance_assurance.get("release_eligible"))}</td></tr>
    <tr><th>Production stopping rule</th><td>{_cell(stop_decision.stopping_rule)}</td></tr>
    <tr><th>Preselection decision</th><td>{_cell(stop_decision.outcome)};
      full release-eligible: {_cell(stop_decision.full_release_eligible)}</td></tr>
    <tr><th>Action opened by this run</th><td>
      {_cell(selected_action.item_id if selected_action is not None else None)}</td></tr>
    <tr><th>Stop-decision SHA-256</th><td class="hash">
      {_cell(stop_decision.decision_sha256)}</td></tr>
    <tr><th>Complete-corpus membership SHA-256</th><td class="hash">
      {_cell(certificate.complete_corpus_identity.membership_sha256)}</td></tr>
    <tr><th>Item-risk scoring receipt SHA-256</th><td class="hash">
      {_cell(item_risk_receipt_sha256)}</td></tr>
    <tr><th>Adaptive bundle SHA-256</th><td class="hash">
      {_cell(adaptive_bundle_sha256)}</td></tr>
    <tr><th>Adaptive prospective assessment SHA-256</th><td class="hash">
      {_cell(adaptive_assessment_sha256)}</td></tr>
  </table>

  <h2>Corpus eligibility</h2>
  <table><thead><tr><th>Paper</th><th>Title</th><th>Status</th><th>Reason</th></tr></thead>
    <tbody>{_eligibility_rows(certificate)}</tbody></table>

  <h2>Evidence graph</h2>
  <table><thead><tr><th>Estimate</th><th>Paper</th><th>Outcome</th><th>Value</th>
    <th>Format</th><th>Source</th><th>Grounding</th></tr></thead>
    <tbody>{_estimate_rows(certificate)}</tbody></table>

  <h2>Ranked verification actions</h2>
  <table><thead><tr><th>Rank</th><th>Evidence item</th><th>Selected</th><th>Resolved</th>
    <th>Influence</th><th>Expected loss reduction / minute</th></tr></thead>
    <tbody>{_audit_rows(certificate)}</tbody></table>

  <h2>Integrity</h2>
  <p class="hash"><strong>Certificate SHA-256:</strong> {certificate.certificate_sha256}</p>
  <p class="hash"><strong>Graph SHA-256:</strong> {certificate.evidence_graph_sha256}</p>
  <p class="hash"><strong>Synthesis SHA-256:</strong> {certificate.synthesis_sha256}</p>

  <details><summary><strong>Complete normative JSON payload</strong></summary>
    <pre>{html.escape(canonical)}</pre>
  </details>
  <p class="muted">Generated {html.escape(certificate.generated_at.isoformat())};
    no remote assets.</p>
</body>
</html>
"""


def write_certificate_artifacts(
    certificate: VerificationCertificate
    | VerificationCertificateV8
    | ConditionVerificationCertificateV6
    | ConditionVerificationCertificateV8
    | FinalConditionVerificationCertificateV7
    | FinalConditionVerificationCertificateV9,
    output_dir: Path,
    *,
    force: bool = False,
) -> CertificateArtifacts:
    """Atomically write the normative JSON and its static HTML rendering."""

    json_path = output_dir / "verification-certificate.json"
    html_path = output_dir / "verification-certificate.html"
    if not force:
        existing = [path.as_posix() for path in (json_path, html_path) if path.exists()]
        if existing:
            raise FileExistsError(f"verification_certificate_outputs_exist:{existing}")
    rendered_html = render_certificate_html(certificate)
    atomic_write_json(json_path, certificate, force=force)
    atomic_write_text(html_path, rendered_html, force=force)
    return CertificateArtifacts(
        json_path=json_path.as_posix(),
        json_sha256=sha256_file(json_path),
        html_path=html_path.as_posix(),
        html_sha256=sha256_file(html_path),
    )


__all__ = [
    "CertificateArtifacts",
    "CertificateLineageStage",
    "ConditionCalibrationAssessmentReceiptV1",
    "ConditionCalibrationCollectionDecisionV1",
    "ConditionCalibrationCollectionSourceV1",
    "ConditionCompositionTerminalJoinV1",
    "ConditionProductionStopDecisionV2",
    "ConditionVerificationCertificateV6",
    "ConditionVerificationCertificateV8",
    "FinalConditionReleaseAssessmentV1",
    "FinalConditionVerificationCertificateV7",
    "FinalConditionVerificationCertificateV9",
    "ProductionStopDecision",
    "VerificationCertificate",
    "VerificationCertificateV8",
    "freeze_condition_calibration_assessment_receipt_v1",
    "freeze_condition_calibration_collection_decision_v1",
    "freeze_condition_calibration_collection_source_v1",
    "freeze_condition_composition_terminal_join_v1",
    "freeze_condition_production_stop_decision_v2",
    "freeze_condition_verification_certificate_v6",
    "freeze_condition_verification_certificate_v8",
    "freeze_final_condition_release_assessment_v1",
    "freeze_final_condition_verification_certificate_v7",
    "freeze_final_condition_verification_certificate_v9",
    "freeze_production_stop_decision",
    "freeze_verification_certificate",
    "freeze_verification_certificate_v8",
    "match_validated_condition_calibration_collection_source_membership_v1",
    "render_certificate_html",
    "validate_condition_calibration_collection_source_anchor_v1",
    "validate_condition_calibration_collection_source_membership_v1",
    "write_certificate_artifacts",
]
