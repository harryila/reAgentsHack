"""Unified, provider-free orchestration for scientific claim verification.

This module owns the supported path from a claim manifest and frozen corpus artifact to
the evidence graph, statistical synthesis, graph-derived audit priorities, fail-closed
release assessment, and verification certificate.  Network retrieval and model-backed
extraction remain replaceable upstream adapters; this boundary consumes their frozen
outputs and never opens the corpus during a run.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from literature_multiverse.adaptive_calibration import (
    AdaptiveCalibrationBundle,
    AdaptiveCalibrationBundleV2,
    AdaptiveCalibrationError,
    AdaptiveIndependenceIdentityV2,
    AdaptivePolicyArmTrajectory,
    AdaptivePolicyContext,
    AdaptivePreselectionState,
    AdaptiveProspectiveAssessmentV2,
    AdaptiveTargetSemanticsBindingV2,
    AdaptiveTerminalAuditCandidate,
    CompleteCorpusIdentity,
    ConditionCalibrationCollectionSourceRosterV1,
    ConditionCalibrationProjectionV1,
    ConditionConfirmationGateAssessmentV1,
    ConditionGateInvocationProofV2,
    ConfirmationAwareReleaseQualificationProofV2,
    PolicyVisibleQuestionTrajectoryV2,
    ProspectiveAdaptiveReleaseCandidate,
    ProspectiveAdaptiveReleaseCandidateV2,
    adaptive_independence_identity_from_condition_plan_v1,
    assess_adaptive_release_candidate,
    assess_confirmation_aware_adaptive_release_candidate,
    freeze_adaptive_policy_arm_trajectory,
    freeze_adaptive_policy_context,
    freeze_adaptive_target_semantics_v2,
    freeze_complete_corpus_identity,
    freeze_condition_calibration_projection,
    freeze_condition_confirmation_gate_assessment,
    freeze_condition_gate_invocation_proof_v2,
    freeze_confirmation_aware_arm_trajectory,
    freeze_confirmation_aware_release_qualification_proof_v2,
    freeze_policy_visible_question_trajectory,
    freeze_policy_visible_question_trajectory_v2,
    freeze_preselection_state_from_production_components,
    freeze_prospective_adaptive_candidate,
    freeze_prospective_adaptive_candidate_v2,
    validate_adaptive_calibration_bundle_integrity,
    validate_adaptive_calibration_bundle_v2_integrity,
)
from literature_multiverse.budgeted_verification import (
    AuditCandidate,
    ClaimModel,
    ProbabilityBasis,
    ReleaseGuardConfig,
    ScenarioKind,
)
from literature_multiverse.calibration import FrozenCalibrationBundle
from literature_multiverse.certificate import (
    CertificateLineageStage,
    ConditionCalibrationCollectionSourceV1,
    ConditionVerificationCertificateV6,
    FinalConditionVerificationCertificateV7,
    VerificationCertificate,
    freeze_condition_calibration_collection_decision_v1,
    freeze_condition_calibration_collection_source_v1,
    freeze_condition_production_stop_decision_v2,
    freeze_condition_verification_certificate_v6,
    freeze_final_condition_verification_certificate_v7,
    freeze_production_stop_decision,
    freeze_verification_certificate,
)
from literature_multiverse.claim_release import (
    CLAIM_RELEASE_RISK_FEATURE_NAMES,
    AuditResolutionReceipt,
    ClaimReleaseConfig,
    ClaimTarget,
    ConditionClaimReleaseAssessmentV1,
    TargetDirection,
    _assess_claim_release_after_verifier_history_replay,
    _assess_qualified_claim_release_after_verifier_history_replay,
    assess_claim_release,
    assess_global_condition_claim_release_source,
    assess_qualified_claim_release,
    classify_qualified_synthesis_evidence,
)
from literature_multiverse.claim_semantics import (
    ClaimTargetV2,
    GlobalConditionDependenceTargetV1,
    QualifiedClaimVerdict,
)
from literature_multiverse.condition_confirmation import (
    ConditionConfirmationAssessmentV1,
    ConditionConfirmationError,
    ConditionConfirmationFrozenModelV1,
    ConditionConfirmationPlanV1,
    fit_condition_confirmation_model,
    freeze_condition_confirmation_target,
    materialize_condition_confirmation_inputs,
    partition_evidence_graph,
    prepare_condition_confirmation_plan,
    validate_condition_confirmation_model,
)
from literature_multiverse.effects import EffectEvidence
from literature_multiverse.evidence_graph import (
    AdapterIssueSeverity,
    CohortIdentity,
    CohortIdentityBasis,
    EvidenceGraph,
    GraphAdapterContext,
    OutcomeTimepoint,
    PublicationIdentity,
    adapt_effect_evidence,
    adapt_finding_row,
)
from literature_multiverse.item_risk_artifacts import ItemRiskScoringRunReceipt
from literature_multiverse.item_risk_calibration import (
    ItemRiskCalibrationBundle,
    ItemRiskCalibrationError,
    ItemRiskCandidate,
    RiskBound,
    score_item_risk_bound,
    verified_audit_cell_rate_ucl_fields,
)
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.meta_analysis import (
    build_graph_counterfactual_audit_plan,
    synthesize_evidence_graph,
)
from literature_multiverse.models import SHA256_RE, ContractModel, FindingRow
from literature_multiverse.native_extraction import NativeSourceManifest
from literature_multiverse.native_grounding import (
    NativeExtractionExecutionContext,
    NativeGroundingError,
    TypedEvidenceGroundingPackage,
    reverify_typed_evidence_grounding_package,
)
from literature_multiverse.normalize import map_outcome_family
from literature_multiverse.pipeline_fingerprint import (
    PipelineComponentSpec,
    PipelineFingerprint,
    PipelineFingerprintVerification,
    compute_pipeline_fingerprint,
    require_pipeline_fingerprint_match,
)
from literature_multiverse.production_policy import should_stop_at_full_frozen_release
from literature_multiverse.records import read_parquet_records
from literature_multiverse.sequential_verification import (
    CurrentAuditCandidate,
    SequentialVerificationContractError,
    SequentialVerificationState,
    adaptive_preselection_history_from_state,
    create_sequential_verification_state,
    current_candidates_from_audit_candidates,
    freeze_state_expectation,
    resume_sequential_verification_state,
    select_next_audit_candidate,
    selection_predecessor_states_from_state,
)
from literature_multiverse.typed_extraction import (
    FragmentStatus,
    TypedEvidenceCorpus,
)


class VerificationContractError(ValueError):
    """The unified verifier cannot safely complete the requested run."""


Scalar = str | int | float | bool | None


class ScientificClaim(ContractModel):
    """The AI-generated statement whose literature support is being assessed."""

    statement: Annotated[str, Field(min_length=1)]
    direction: TargetDirection
    outcome_name: Annotated[str, Field(min_length=1)]
    contrast_id: Annotated[str, Field(min_length=1)] | None = None
    estimand: str | None = None
    conditions: dict[str, Scalar] = Field(default_factory=dict)

    @field_validator("conditions")
    @classmethod
    def validate_conditions(cls, value: dict[str, Scalar]) -> dict[str, Scalar]:
        if any(not key.strip() for key in value):
            raise ValueError("claim_condition_name_empty")
        for item in value.values():
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("claim_condition_value_nonfinite")
        return value


class VerificationProtocol(ContractModel):
    """Frozen search/screening boundary associated with the claim."""

    corpus_cutoff: Annotated[str, Field(min_length=1)]
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("inclusion_criteria", "exclusion_criteria")
    @classmethod
    def validate_criteria(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("verification_protocol_criterion_empty")
        if len(value) != len(set(value)):
            raise ValueError("verification_protocol_criteria_not_unique")
        return value


class AuditPolicyConfig(ContractModel):
    """Prospective ranking inputs; costs are total person-minutes, never labels."""

    error_probability: Annotated[float, Field(ge=0, le=1)] = 0.5
    probability_basis: ProbabilityBasis = ProbabilityBasis.HEURISTIC
    probability_source: Annotated[str, Field(min_length=1)] = (
        "unvalidated-default-error-probability"
    )
    verification_minutes_per_item: Annotated[float, Field(gt=0)] = 10.0
    item_error_probabilities: dict[str, float] = Field(default_factory=dict)
    item_verification_minutes: dict[str, float] = Field(default_factory=dict)
    decision_threshold: Annotated[float, Field(gt=0, lt=1)] = 0.5

    @model_validator(mode="after")
    def validate_item_overrides(self) -> AuditPolicyConfig:
        if self.probability_basis is not ProbabilityBasis.HEURISTIC:
            raise ValueError(
                "claim_manifest_probability_basis_must_be_heuristic;"
                "calibration_requires_external_proof"
            )
        if any(not item_id.strip() for item_id in self.item_error_probabilities):
            raise ValueError("audit_item_error_probability_id_empty")
        if any(
            not math.isfinite(value) or not 0 <= value <= 1
            for value in self.item_error_probabilities.values()
        ):
            raise ValueError("audit_item_error_probability_invalid")
        if any(not item_id.strip() for item_id in self.item_verification_minutes):
            raise ValueError("audit_item_verification_cost_id_empty")
        if any(
            not math.isfinite(value) or value <= 0
            for value in self.item_verification_minutes.values()
        ):
            raise ValueError("audit_item_verification_cost_invalid")
        return self


class LegacyAdapterConfig(ContractModel):
    """Explicit orientation used when lifting the old categorical findings ledger."""

    contrast_label: Annotated[str, Field(min_length=1)] = "intervention_vs_comparator"
    positive_direction_means: Annotated[str, Field(min_length=1)] = (
        "higher outcome under the intervention"
    )
    treatment_label: Annotated[str, Field(min_length=1)] = "intervention"
    comparator_label: Annotated[str, Field(min_length=1)] = "comparator"


class AuditGuardConfig(ContractModel):
    """Serializable view of :class:`ReleaseGuardConfig`.

    Legacy probability/risk aliases are intentionally rejected by the strict model.
    A cell-average rate UCL is not an itemwise error probability or a claim-level
    residual-risk bound.
    """

    max_unresolved_item_influence: Annotated[float, Field(ge=0, le=1)] = 0.05
    max_unresolved_expected_claim_loss: Annotated[float, Field(ge=0)] = 0.05
    block_counterfactual_conclusion_flips: bool = True
    require_calibrated_item_scores: bool = True
    require_item_cell_rate_ucls: bool = True
    max_unresolved_item_cell_ucl_sum: Annotated[float, Field(ge=0, le=1)] = 0.05

    def to_runtime(self) -> ReleaseGuardConfig:
        return ReleaseGuardConfig(**self.model_dump(mode="python"))


class ClaimManifest(ContractModel):
    """Closed YAML/JSON input contract for ``lm verify``."""

    claim_manifest_version: Literal["1", "2", "3"] = "1"
    question_id: Annotated[str, Field(pattern=r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")]
    population_id: Annotated[str, Field(min_length=1)]
    domain: Annotated[str, Field(min_length=1)]
    claim: ScientificClaim
    qualified_target: ClaimTargetV2 | None = None
    global_condition_target: GlobalConditionDependenceTargetV1 | None = None
    protocol: VerificationProtocol
    audit: AuditPolicyConfig = Field(default_factory=AuditPolicyConfig)
    legacy_adapter: LegacyAdapterConfig = Field(default_factory=LegacyAdapterConfig)
    release: ClaimReleaseConfig = Field(default_factory=ClaimReleaseConfig)
    audit_guard: AuditGuardConfig = Field(default_factory=AuditGuardConfig)
    pipeline_sha256: str | None = None

    @field_validator("pipeline_sha256")
    @classmethod
    def validate_pipeline_hash(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("claim_manifest_pipeline_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_condition_contract(self) -> ClaimManifest:
        if self.claim_manifest_version == "1":
            if self.qualified_target is not None or self.global_condition_target is not None:
                raise ValueError("v1_claim_manifest_forbids_typed_target")
            if self.claim.conditions:
                raise ValueError("claim_conditions_require_v2_typed_qualified_target")
            return self
        if self.claim_manifest_version == "2":
            if self.qualified_target is None:
                raise ValueError("v2_claim_manifest_requires_qualified_target")
            if self.global_condition_target is not None:
                raise ValueError("v2_claim_manifest_forbids_global_condition_target")
            if self.claim.conditions:
                raise ValueError("v2_claim_uses_qualified_target_conditions_only")
            target = self.qualified_target
            if (
                target.direction.value != self.claim.direction.value
                or target.outcome_name != self.claim.outcome_name
                or target.contrast_id != self.claim.contrast_id
            ):
                raise ValueError("qualified_target_claim_identity_mismatch")
            condition_names = sorted(condition.moderator for condition in target.conditions)
            if self.release.prespecified_condition_moderators != condition_names:
                raise ValueError(
                    "qualified_target_conditions_must_match_prespecified_condition_moderators"
                )
            return self

        if self.qualified_target is not None:
            raise ValueError("v3_claim_manifest_forbids_qualified_target")
        if self.global_condition_target is None:
            raise ValueError("v3_claim_manifest_requires_global_condition_target")
        if self.claim.conditions:
            raise ValueError("v3_global_condition_target_forbids_claim_conditions")
        target = self.global_condition_target
        if (
            target.reference_direction.value != self.claim.direction.value
            or target.outcome_name != self.claim.outcome_name
            or target.contrast_id != self.claim.contrast_id
            or self.claim.estimand is None
            or target.estimand != self.claim.estimand
        ):
            raise ValueError("global_condition_target_claim_identity_mismatch")
        if self.release.prespecified_condition_moderators != target.moderator_names:
            raise ValueError(
                "global_condition_target_moderators_must_match_release_family"
            )
        return self


class CorpusEligibilityRecord(ContractModel):
    """One retrieved paper and its terminal screening state."""

    paper_id: Annotated[str, Field(min_length=1)]
    title: str | None = None
    status: Literal["included", "excluded", "pending"]
    reason: Annotated[str, Field(min_length=1)]
    source: str | None = None


class CorpusAdapterIssue(ContractModel):
    """A graph-conversion caveat tied to its source evidence item."""

    severity: AdapterIssueSeverity
    code: Annotated[str, Field(min_length=1)]
    detail: Annotated[str, Field(min_length=1)]
    paper_id: str | None = None
    finding_id: str | None = None


class CorpusProvenanceAssurance(ContractModel):
    """Replay assurance attached to a corpus before release is considered."""

    assurance_version: Literal["corpus-provenance-assurance-v1"] = "corpus-provenance-assurance-v1"
    status: Literal[
        "source_replayed_native_grounding",
        "embedded_synthetic_fixture",
        "unverified_source_provenance",
    ]
    reason: Annotated[str, Field(min_length=1)]
    replay_sha256: str | None = None

    @field_validator("replay_sha256")
    @classmethod
    def validate_replay_hash(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("corpus_provenance_replay_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_assurance(self) -> CorpusProvenanceAssurance:
        if self.status == "source_replayed_native_grounding":
            if self.replay_sha256 is None:
                raise ValueError("source_replayed_corpus_requires_replay_sha256")
        elif self.replay_sha256 is not None:
            raise ValueError("non_replayed_corpus_forbids_replay_sha256")
        return self

    @property
    def release_eligible(self) -> bool:
        """Whether this assurance class can proceed to all other release gates."""

        return self.status == "source_replayed_native_grounding"


@dataclass(frozen=True, slots=True)
class CorpusLoadResult:
    """Frozen corpus payload ready for orchestration."""

    corpus_id: str
    source_label: str
    source_format: str
    source_sha256: str
    graph: EvidenceGraph
    eligibility: tuple[CorpusEligibilityRecord, ...]
    adapter_issues: tuple[CorpusAdapterIssue, ...]
    metadata: dict[str, Any]
    provenance_assurance: CorpusProvenanceAssurance
    extraction_context: NativeExtractionExecutionContext | None = None

    def embedded_fixture_identity_valid(self) -> bool:
        """Recognize only the loader-owned deterministic integration fixture."""

        return (
            self.provenance_assurance.status == "embedded_synthetic_fixture"
            and self.source_format == "embedded_synthetic_fixture"
            and self.source_label == "embedded:offline-verifier-fixture-v1"
            and self.metadata.get("empirical_evidence") is False
            and self.metadata.get("purpose") == "offline_integration_test"
        )

    def provenance_release_eligible(self) -> bool:
        """Require the assurance class to agree with the loader-controlled source kind."""

        assurance = self.provenance_assurance
        if not assurance.release_eligible:
            return False
        if assurance.status == "source_replayed_native_grounding":
            terminal_membership = self.metadata.get("terminal_fragment_membership")
            return (
                self.source_format == "typed_evidence_grounding_package_json"
                and self.metadata.get("grounding_replay_sha256") == assurance.replay_sha256
                and self.metadata.get("grounding_package_version")
                == "typed-evidence-grounding-package-v4"
                and self.metadata.get("source_manifest_membership_bound") is True
                and isinstance(self.metadata.get("source_manifest_sha256"), str)
                and SHA256_RE.fullmatch(self.metadata["source_manifest_sha256"]) is not None
                and isinstance(self.metadata.get("native_source_manifest"), dict)
                and hash_canonical(self.metadata["native_source_manifest"])
                == self.metadata["source_manifest_sha256"]
                and self.metadata.get("source_manifest_records") == len(self.eligibility)
                and isinstance(terminal_membership, list)
                and self.metadata.get("terminal_fragment_records")
                == len(self.eligibility)
                and hash_canonical(terminal_membership)
                == self.metadata.get("terminal_fragment_membership_sha256")
                and isinstance(self.metadata.get("native_corpus_cutoff"), str)
                and bool(self.metadata["native_corpus_cutoff"].strip())
                and self.extraction_context is not None
                and self.metadata.get("extraction_context_sha256")
                == self.extraction_context.context_sha256
                and self.metadata.get("extraction_context_receipt_sha256")
                == self.metadata.get("replayed_extraction_context_receipt_sha256")
            )
        return False

    def certificate_payload(self) -> dict[str, Any]:
        included = sum(item.status == "included" for item in self.eligibility)
        excluded = sum(item.status == "excluded" for item in self.eligibility)
        pending = sum(item.status == "pending" for item in self.eligibility)
        assurance = self.provenance_assurance
        if (
            assurance.status == "embedded_synthetic_fixture"
            and not self.embedded_fixture_identity_valid()
        ):
            assurance = CorpusProvenanceAssurance(
                status="unverified_source_provenance",
                reason=(
                    "The claimed embedded-fixture assurance does not match the exact "
                    "loader-owned fixture identity."
                ),
            )
        return {
            "corpus_id": self.corpus_id,
            "source_label": self.source_label,
            "source_format": self.source_format,
            "source_sha256": self.source_sha256,
            "provenance_assurance": {
                **assurance.model_dump(mode="json"),
                "release_eligible": self.provenance_release_eligible(),
            },
            "eligibility": [item.model_dump(mode="json") for item in self.eligibility],
            "eligibility_counts": {
                "excluded": excluded,
                "included": included,
                "pending": pending,
            },
            "graph_counts": {
                "cohorts": len(self.graph.cohorts),
                "estimates": len(self.graph.outcome_estimates),
                "publications": len(self.graph.publications),
                "studies": len(self.graph.studies),
            },
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class PreparedVerificationScientificState:
    """Recomputed scientific state used by both release and audit correction paths."""

    target: ClaimTarget
    claim_model: ClaimModel
    audit_candidates: tuple[AuditCandidate, ...]
    counterfactuals: tuple[dict[str, Any], ...]
    synthesis: dict[str, Any]
    item_risk_bounds: tuple[RiskBound, ...]


@dataclass(frozen=True, slots=True)
class PreparedConditionVerificationContext:
    """Validated development-only inputs for one manifest-v3 online trajectory."""

    plan: ConditionConfirmationPlanV1
    development_graph: EvidenceGraph
    frozen_model: ConditionConfirmationFrozenModelV1
    target_semantics: AdaptiveTargetSemanticsBindingV2
    independence_identity: AdaptiveIndependenceIdentityV2
    projection: ConditionCalibrationProjectionV1
    ordinary_blocking_reasons: tuple[str, ...]


def prepare_verification_scientific_state(
    *,
    manifest: ClaimManifest,
    graph: EvidenceGraph,
    pipeline_verification: PipelineFingerprintVerification,
    item_risk_calibration_bundle: ItemRiskCalibrationBundle | None = None,
    item_risk_candidates: list[ItemRiskCandidate] | None = None,
    resolved_item_ids_for_risk_projection: set[str] | None = None,
) -> PreparedVerificationScientificState:
    """Rerun qualified synthesis, item risk, and every graph counterfactual."""

    target = ClaimTarget(
        direction=manifest.claim.direction,
        outcome_name=manifest.claim.outcome_name,
        contrast_id=manifest.claim.contrast_id,
    )
    scope_ids = _audit_scope_ids(
        graph=graph,
        target=target,
        qualified_target=manifest.qualified_target,
        config=manifest.release,
    )
    projected_risk_candidates = item_risk_candidates
    resolved_risk_ids = resolved_item_ids_for_risk_projection or set()
    if item_risk_candidates is not None and resolved_risk_ids:
        projected_risk_candidates = _project_item_risk_candidates_after_audit_corrections(
            graph=graph,
            expected_item_ids=scope_ids,
            candidates=item_risk_candidates,
            resolved_item_ids=resolved_risk_ids,
        )
    probability_overrides, item_risk_bounds = _artifact_backed_item_probabilities(
        manifest=manifest,
        graph=graph,
        expected_item_ids=scope_ids,
        pipeline_verification=pipeline_verification,
        bundle=item_risk_calibration_bundle,
        candidates=projected_risk_candidates,
        stale_candidate_item_ids=resolved_risk_ids & set(scope_ids),
    )
    claim_model, candidates, counterfactuals, synthesis = build_graph_counterfactual_audits(
        graph,
        target=target,
        release_config=manifest.release,
        policy=manifest.audit,
        qualified_target=manifest.qualified_target,
        global_condition_target=manifest.global_condition_target,
        probability_overrides=probability_overrides,
    )
    return PreparedVerificationScientificState(
        target=target,
        claim_model=claim_model,
        audit_candidates=tuple(candidates),
        counterfactuals=tuple(counterfactuals),
        synthesis=synthesis,
        item_risk_bounds=tuple(item_risk_bounds),
    )


def load_claim_manifest(path: Path) -> ClaimManifest:
    """Load a closed claim YAML/JSON document without resolving external references."""

    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise VerificationContractError(f"claim_manifest_unreadable:{path}") from exc
    if not isinstance(payload, dict):
        raise VerificationContractError("claim_manifest_must_be_object")
    return ClaimManifest.model_validate(payload)


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode()).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _json_plain(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if type(value).__name__ == "NAType":
        return None
    if isinstance(value, dict):
        return {str(key): _json_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_plain(item) for item in value]
    return value


def _finding_from_flat_record(record: dict[str, Any]) -> FindingRow:
    payload = {key: _json_plain(value) for key, value in record.items()}
    if "moderators" not in payload:
        moderator_keys = sorted(key for key in payload if key.startswith("mod__"))
        payload["moderators"] = {
            key.removeprefix("mod__"): payload.pop(key) for key in moderator_keys
        }
    return FindingRow.model_validate(payload)


def _paper_metadata(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(record["paper_id"]): {key: _json_plain(value) for key, value in record.items()}
        for record in records
        if record.get("paper_id")
    }


def _eligibility_from_papers(
    records: list[dict[str, Any]],
) -> tuple[CorpusEligibilityRecord, ...]:
    output: list[CorpusEligibilityRecord] = []
    for raw in records:
        record = {key: _json_plain(value) for key, value in raw.items()}
        eligible = record.get("eligible")
        screen_status = str(record.get("screen_status") or "")
        if eligible is True:
            status: Literal["included", "excluded", "pending"] = "included"
            reason = "eligible_after_screening_and_extraction"
        elif eligible is False:
            status = "excluded"
            reason = str(record.get("exclusion_reason") or "ineligible_after_mapping")
        elif screen_status == "excluded":
            status = "excluded"
            reason = str(record.get("screen_reason") or "excluded_during_screening")
        else:
            status = "pending"
            reason = "terminal_eligibility_not_available"
        output.append(
            CorpusEligibilityRecord(
                paper_id=str(record["paper_id"]),
                title=(str(record["title"]) if record.get("title") else None),
                status=status,
                reason=reason,
                source=(str(record["source"]) if record.get("source") else None),
            )
        )
    return tuple(sorted(output, key=lambda item: item.paper_id))


def _merge_graphs(graphs: list[EvidenceGraph]) -> EvidenceGraph:
    if not graphs:
        raise VerificationContractError("legacy_corpus_contains_no_findings")

    def merge(field: str, identity: str) -> list[Any]:
        values: dict[str, Any] = {}
        for graph in graphs:
            for item in getattr(graph, field):
                key = str(getattr(item, identity))
                if key in values and values[key] != item:
                    raise VerificationContractError(
                        f"evidence_graph_identity_collision:{field}:{key}"
                    )
                values[key] = item
        return [values[key] for key in sorted(values)]

    return EvidenceGraph(
        publications=merge("publications", "publication_id"),
        studies=merge("studies", "study_id"),
        cohorts=merge("cohorts", "cohort_id"),
        arms=merge("arms", "arm_id"),
        contrasts=merge("contrasts", "contrast_id"),
        outcome_estimates=merge("outcome_estimates", "estimate_id"),
        evidence_spans=merge("evidence_spans", "span_id"),
    )


def adapt_legacy_findings(
    findings: list[FindingRow],
    *,
    settings: LegacyAdapterConfig,
    papers: dict[str, dict[str, Any]] | None = None,
) -> tuple[EvidenceGraph, tuple[CorpusAdapterIssue, ...]]:
    """Lift the old categorical ledger without inventing cohorts or effect sizes."""

    papers = papers or {}
    graphs: list[EvidenceGraph] = []
    issues: list[CorpusAdapterIssue] = []
    for finding in sorted(findings, key=lambda item: item.finding_id):
        item_key = _stable_id("legacy", finding.finding_id)
        paper_key = _stable_id("publication", finding.paper_id)
        metadata = papers.get(finding.paper_id, {})
        publication = PublicationIdentity(
            publication_id=paper_key,
            paper_id=finding.paper_id,
            doc_id=finding.doc_id,
            doi=metadata.get("doi"),
            pmid=(str(metadata["pmid"]) if metadata.get("pmid") else None),
            title=(str(metadata["title"]) if metadata.get("title") else None),
            publication_year=(
                int(metadata["pub_year"]) if metadata.get("pub_year") is not None else None
            ),
        )
        context = GraphAdapterContext(
            publication=publication,
            study_id=f"study-{item_key}",
            cohort_identity=CohortIdentity(
                cohort_id=f"cohort-{item_key}",
                basis=CohortIdentityBasis.LEGACY_PLACEHOLDER,
                rationale=(
                    "The legacy FindingRow does not encode a reconciled participant/sample "
                    "identity; human reconciliation is required before synthesis."
                ),
            ),
            treatment_arm_id=f"arm-treatment-{item_key}",
            comparator_arm_id=f"arm-comparator-{item_key}",
            contrast_id=f"contrast-{item_key}",
            contrast_label=settings.contrast_label,
            positive_direction_means=settings.positive_direction_means,
            treatment_label=finding.intervention or settings.treatment_label,
            comparator_label=finding.comparator or settings.comparator_label,
        )
        result = adapt_finding_row(finding, context=context)
        graphs.append(result.graph)
        issues.extend(
            CorpusAdapterIssue(
                severity=issue.severity,
                code=issue.code,
                detail=issue.detail,
                paper_id=finding.paper_id,
                finding_id=finding.finding_id,
            )
            for issue in result.issues
        )
    return _merge_graphs(graphs), tuple(
        sorted(issues, key=lambda item: (item.finding_id or "", item.code))
    )


def _graph_eligibility(graph: EvidenceGraph) -> tuple[CorpusEligibilityRecord, ...]:
    return tuple(
        CorpusEligibilityRecord(
            paper_id=publication.paper_id,
            title=publication.title,
            status="included",
            reason="present_in_typed_evidence_graph",
        )
        for publication in sorted(graph.publications, key=lambda item: item.paper_id)
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationContractError(f"corpus_json_unreadable:{path}") from exc
    if not isinstance(payload, dict):
        raise VerificationContractError("corpus_json_must_be_object")
    return payload


def _read_jsonl_findings(path: Path) -> list[FindingRow]:
    findings: list[FindingRow] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise VerificationContractError(f"corpus_jsonl_unreadable:{path}") from exc
    for position, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerificationContractError(f"corpus_jsonl_invalid_json:line={position}") from exc
        if not isinstance(payload, dict):
            raise VerificationContractError(f"corpus_jsonl_row_not_object:line={position}")
        findings.append(_finding_from_flat_record(payload))
    return findings


def load_corpus(
    path: Path,
    *,
    legacy_settings: LegacyAdapterConfig,
    repository_root: Path | None = None,
) -> CorpusLoadResult:
    """Load a typed graph/bundle or conservatively adapt normalized legacy findings."""

    source_path = path
    paper_path: Path | None = None
    if path.is_dir():
        typed_package_path = path / "typed_evidence_grounding_package.json"
        typed_corpus_path = path / "typed_evidence_corpus.json"
        graph_path = path / "evidence_graph.json"
        findings_path = path / "findings.parquet"
        if typed_package_path.is_file():
            source_path = typed_package_path
        elif typed_corpus_path.is_file():
            source_path = typed_corpus_path
        elif graph_path.is_file():
            source_path = graph_path
        elif findings_path.is_file():
            source_path = findings_path
        else:
            raise VerificationContractError(
                "corpus_directory_requires_typed_evidence_grounding_package_"
                "typed_evidence_corpus_"
                "evidence_graph_json_or_findings_parquet"
            )
        candidate_papers = path / "papers.parquet"
        paper_path = candidate_papers if candidate_papers.is_file() else None
    elif not path.is_file():
        raise VerificationContractError(f"corpus_path_not_found:{path}")

    paper_records = read_parquet_records(paper_path) if paper_path is not None else []
    paper_metadata = _paper_metadata(paper_records)
    eligibility = _eligibility_from_papers(paper_records)
    source_hash_parts = {"evidence": sha256_file(source_path)}
    if paper_path is not None:
        source_hash_parts["papers"] = sha256_file(paper_path)
    source_sha256 = hash_canonical(source_hash_parts)
    provenance_assurance: CorpusProvenanceAssurance | None = None
    extraction_context: NativeExtractionExecutionContext | None = None

    suffix = source_path.suffix.casefold()
    if suffix == ".json":
        payload = _read_json_object(source_path)
        typed_corpus: TypedEvidenceCorpus | None = None
        grounding_package: TypedEvidenceGroundingPackage | None = None
        if payload.get("package_version") in {
            "typed-evidence-grounding-package-v1",
            "typed-evidence-grounding-package-v2",
            "typed-evidence-grounding-package-v3",
            "typed-evidence-grounding-package-v4",
        }:
            grounding_package = TypedEvidenceGroundingPackage.model_validate(payload)
            if repository_root is None:
                raise VerificationContractError(
                    "typed_evidence_grounding_package_requires_repository_root"
                )
            try:
                replay = reverify_typed_evidence_grounding_package(
                    package=grounding_package,
                    repository_root=repository_root,
                )
            except NativeGroundingError as exc:
                raise VerificationContractError(
                    f"typed_evidence_grounding_replay_failed:{exc}"
                ) from exc
            typed_corpus = grounding_package.corpus
        elif payload.get("corpus_version") in {
            "typed-evidence-corpus-v2",
            "typed-evidence-corpus-v3",
        }:
            typed_corpus = TypedEvidenceCorpus.model_validate(payload)
            if typed_corpus.estimable_publication_ids:
                raise VerificationContractError(
                    "estimable_typed_evidence_corpus_requires_grounding_package"
                )
        if typed_corpus is not None:
            graph = typed_corpus.graph
            corpus_id = typed_corpus.question_id
            source_format = (
                "typed_evidence_grounding_package_json"
                if grounding_package is not None
                else "typed_evidence_corpus_json"
            )
            eligibility = tuple(
                CorpusEligibilityRecord(
                    paper_id=fragment.paper_id,
                    status="included",
                    reason=(
                        "native_typed_effect_extracted"
                        if fragment.status is FragmentStatus.ESTIMABLE
                        else f"included_but_{fragment.non_estimability_reason.value}"
                    ),
                    source=fragment.source_document.source_locator,
                )
                for fragment in typed_corpus.fragments
            )
            issues = tuple(
                CorpusAdapterIssue(
                    severity=AdapterIssueSeverity.BLOCKING,
                    code=issue.code,
                    detail=issue.detail,
                    paper_id=issue.paper_id,
                )
                for issue in typed_corpus.issues
            )
            bundle_metadata = {
                "typed_evidence_corpus_sha256": typed_corpus.corpus_sha256,
                "pipeline_fingerprint_sha256": (typed_corpus.pipeline_fingerprint_sha256),
                "publication_fragments": len(typed_corpus.fragments),
                "estimable_publications": len(typed_corpus.estimable_publication_ids),
                "non_estimable_publications": len(typed_corpus.non_estimable_publication_ids),
                "terminal_fragment_membership": [
                    {
                        "fragment_sha256": fragment.fragment_sha256,
                        "paper_id": fragment.paper_id,
                        "publication_id": fragment.publication_id,
                        "status": fragment.status.value,
                    }
                    for fragment in typed_corpus.fragments
                ],
                "terminal_fragment_membership_sha256": hash_canonical(
                    [
                        {
                            "fragment_sha256": fragment.fragment_sha256,
                            "paper_id": fragment.paper_id,
                            "publication_id": fragment.publication_id,
                            "status": fragment.status.value,
                        }
                        for fragment in typed_corpus.fragments
                    ]
                ),
                "terminal_fragment_records": len(typed_corpus.fragments),
            }
            if grounding_package is not None:
                reconciliation = grounding_package.cohort_reconciliation
                if reconciliation is not None:
                    assert reconciliation.reconciled_graph is not None
                    graph = reconciliation.reconciled_graph
                    bundle_metadata.update(
                        {
                            "cohort_reconciliation_status": reconciliation.status.value,
                            "cohort_reconciliation_receipt_sha256": (reconciliation.receipt_sha256),
                            "cross_publication_identity_assurance_complete": (
                                reconciliation.cross_publication_identity_assurance_complete
                            ),
                            "reconciled_graph_sha256": (reconciliation.reconciled_graph_sha256),
                            "merged_study_groups": reconciliation.merged_study_groups,
                            "merged_cohort_groups": reconciliation.merged_cohort_groups,
                            "reconciliation_candidate_components": len(reconciliation.candidates),
                            "reconciliation_issue_codes": sorted(
                                {item.code for item in reconciliation.issues}
                            ),
                        }
                    )
                    if not reconciliation.cross_publication_identity_assurance_complete:
                        issues = (
                            *issues,
                            CorpusAdapterIssue(
                                severity=AdapterIssueSeverity.BLOCKING,
                                code=("cross_publication_cohort_reconciliation_incomplete"),
                                detail=(
                                    "Exact strong identifiers were reconciled where safe, "
                                    "but complete cross-publication study/cohort identity "
                                    "requires an external reviewer partition."
                                ),
                            ),
                        )
                else:
                    bundle_metadata.update(
                        {
                            "cohort_reconciliation_status": "absent",
                            "cohort_reconciliation_receipt_sha256": None,
                            "cross_publication_identity_assurance_complete": (
                                len(graph.publications) == 1
                            ),
                            "merged_study_groups": 0,
                            "merged_cohort_groups": 0,
                        }
                    )
                    if len(graph.publications) > 1:
                        issues = (
                            *issues,
                            CorpusAdapterIssue(
                                severity=AdapterIssueSeverity.BLOCKING,
                                code="cross_publication_cohort_reconciliation_absent",
                                detail=(
                                    "This legacy native package has multiple publications "
                                    "but no hash-bound cohort reconciliation receipt."
                                ),
                            ),
                        )
                bundle_metadata.update(
                    {
                        "grounding_package_version": grounding_package.package_version,
                        "grounding_package_sha256": grounding_package.package_sha256,
                        "grounding_validation_sha256": (
                            grounding_package.grounding_validation.validation_sha256
                        ),
                        "grounding_receipts": len(grounding_package.grounding_receipts),
                        "grounding_replay_sha256": replay.replay_sha256,
                        "source_manifest_membership_bound": (
                            grounding_package.package_version
                            in {
                                "typed-evidence-grounding-package-v3",
                                "typed-evidence-grounding-package-v4",
                            }
                        ),
                        "source_manifest_sha256": grounding_package.source_manifest_sha256,
                        "native_source_manifest": (
                            grounding_package.source_manifest.model_dump(mode="json")
                            if grounding_package.source_manifest is not None
                            else None
                        ),
                        "source_manifest_records": (
                            len(grounding_package.source_manifest.records)
                            if grounding_package.source_manifest is not None
                            else 0
                        ),
                        "native_corpus_cutoff": grounding_package.corpus_cutoff,
                    }
                )
                context_receipt = grounding_package.extraction_context_receipt
                if context_receipt is not None:
                    extraction_context = context_receipt.execution_context
                    bundle_metadata.update(
                        {
                            "extraction_context_sha256": (
                                extraction_context.context_sha256
                            ),
                            "extraction_context_receipt_sha256": (
                                context_receipt.receipt_sha256
                            ),
                            "replayed_extraction_context_receipt_sha256": (
                                replay.extraction_context_receipt_sha256
                            ),
                            "question_config_sha256": (
                                extraction_context.question_config_sha256
                            ),
                            "rendered_prompt_sha256s": (
                                replay.rendered_prompt_sha256s
                            ),
                            "evaluation_schema_sha256s": (
                                replay.evaluation_schema_sha256s
                            ),
                            "provider_execution_receipts": [
                                {
                                    "call_count": receipt.call_count,
                                    "execution_identity_sha256": (
                                        receipt.execution_identity_sha256
                                    ),
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
                    )
                if grounding_package.package_version in {
                    "typed-evidence-grounding-package-v1",
                    "typed-evidence-grounding-package-v2",
                }:
                    issues = (
                        *issues,
                        CorpusAdapterIssue(
                            severity=AdapterIssueSeverity.BLOCKING,
                            code="native_source_manifest_membership_unbound",
                            detail=(
                                "This legacy grounding package replays its supplied fragments "
                                "but does not bind the complete native source manifest and "
                                "corpus cutoff. It is analysis-only."
                            ),
                        ),
                    )
                if grounding_package.package_version == (
                    "typed-evidence-grounding-package-v3"
                ):
                    issues = (
                        *issues,
                        CorpusAdapterIssue(
                            severity=AdapterIssueSeverity.BLOCKING,
                            code="native_extraction_context_unbound",
                            detail=(
                                "This legacy v3 package binds corpus membership and source "
                                "grounding but not the exact question config, rendered prompts, "
                                "schemas, provider/model receipts, or extraction input artifacts."
                            ),
                        ),
                    )
                provenance_assurance = CorpusProvenanceAssurance(
                    status="source_replayed_native_grounding",
                    reason=(
                        "Every receipt was replayed against current source bytes and every "
                        "receipt-linked fragment was deterministically rebuilt."
                    ),
                    replay_sha256=replay.replay_sha256,
                )
        elif "graph_schema_version" in payload:
            graph = EvidenceGraph.model_validate(payload)
            bundle_metadata: dict[str, Any] = {}
            issues: tuple[CorpusAdapterIssue, ...] = ()
            corpus_id = path.name
            source_format = "evidence_graph_json"
        elif "graph" in payload:
            allowed = {
                "adapter_issues",
                "corpus_bundle_version",
                "corpus_id",
                "eligibility",
                "graph",
                "metadata",
            }
            extras = sorted(set(payload) - allowed)
            if extras:
                raise VerificationContractError(f"corpus_bundle_unknown_keys:{extras}")
            graph = EvidenceGraph.model_validate(payload["graph"])
            corpus_id = str(payload.get("corpus_id") or path.name)
            source_format = "verification_corpus_bundle_json"
            raw_metadata = payload.get("metadata", {})
            if not isinstance(raw_metadata, dict):
                raise VerificationContractError("corpus_bundle_metadata_must_be_object")
            bundle_metadata = raw_metadata
            raw_issues = payload.get("adapter_issues", [])
            if not isinstance(raw_issues, list):
                raise VerificationContractError("corpus_bundle_adapter_issues_must_be_list")
            issues = tuple(CorpusAdapterIssue.model_validate(item) for item in raw_issues)
            if "eligibility" in payload:
                raw_eligibility = payload["eligibility"]
                if not isinstance(raw_eligibility, list):
                    raise VerificationContractError("corpus_bundle_eligibility_must_be_list")
                eligibility = tuple(
                    sorted(
                        (CorpusEligibilityRecord.model_validate(item) for item in raw_eligibility),
                        key=lambda item: item.paper_id,
                    )
                )
        else:
            raise VerificationContractError("corpus_json_requires_graph_schema_or_graph_bundle")
    elif suffix == ".jsonl":
        findings = _read_jsonl_findings(source_path)
        graph, issues = adapt_legacy_findings(
            findings,
            settings=legacy_settings,
            papers=paper_metadata,
        )
        corpus_id = path.name
        source_format = "legacy_findings_jsonl"
        bundle_metadata = {"legacy_finding_count": len(findings)}
    elif suffix == ".parquet":
        findings = [
            _finding_from_flat_record(record) for record in read_parquet_records(source_path)
        ]
        graph, issues = adapt_legacy_findings(
            findings,
            settings=legacy_settings,
            papers=paper_metadata,
        )
        corpus_id = path.name
        source_format = "legacy_findings_parquet"
        bundle_metadata = {"legacy_finding_count": len(findings)}
    else:
        raise VerificationContractError(f"corpus_format_not_supported:{suffix}")

    if not eligibility:
        eligibility = _graph_eligibility(graph)
    if provenance_assurance is None:
        provenance_assurance = CorpusProvenanceAssurance(
            status="unverified_source_provenance",
            reason=(
                f"The {source_format} input was not source-replayed through the native "
                "grounding-package boundary; analysis is allowed but release is blocked."
            ),
        )
    if not provenance_assurance.release_eligible:
        issues = tuple(
            sorted(
                (
                    *(issue for issue in issues if issue.code != "unverified_source_provenance"),
                    CorpusAdapterIssue(
                        severity=AdapterIssueSeverity.BLOCKING,
                        code="unverified_source_provenance",
                        detail=provenance_assurance.reason,
                    ),
                ),
                key=lambda issue: (issue.finding_id or "", issue.paper_id or "", issue.code),
            )
        )
    return CorpusLoadResult(
        corpus_id=corpus_id,
        source_label=path.as_posix(),
        source_format=source_format,
        source_sha256=source_sha256,
        graph=graph,
        eligibility=eligibility,
        adapter_issues=issues,
        metadata=bundle_metadata,
        provenance_assurance=provenance_assurance,
        extraction_context=extraction_context,
    )


def _disagreement_score(estimate: Any, direction: TargetDirection) -> float:
    numeric = estimate.effect.estimate
    if numeric is not None:
        if numeric == 0:
            return 0.5
        supports = numeric > 0 if direction is TargetDirection.INCREASE else numeric < 0
        return 0.0 if supports else 1.0
    legacy = estimate.legacy_reported_direction
    if legacy is None:
        return 0.5
    if legacy.value in {"mixed", "unclear", "no_effect"}:
        return 0.5
    supports = legacy.value == direction.value
    return 0.0 if supports else 1.0


def _native_condition_value_compatible(value: Any, moderator: Any) -> bool:
    """Check one literal claim/extraction value against the locked moderator type."""

    if moderator.type == "categorical":
        return type(value) is str and value in set(moderator.allowed_values or [])
    if moderator.type == "bool":
        return type(value) is bool and value in set(moderator.allowed_values or [])
    if moderator.type == "int":
        return type(value) is int
    if moderator.type == "float":
        return type(value) in {int, float} and math.isfinite(float(value))
    return False


def _native_claim_config_compatibility_issues(
    *,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
) -> list[CorpusAdapterIssue]:
    """Fail closed when a v4 extraction context does not license the claim semantics."""

    context = corpus.extraction_context
    if context is None:
        return []
    config = context.question_config
    issues: list[CorpusAdapterIssue] = []

    def block(code: str, detail: str, *, finding_id: str | None = None) -> None:
        issues.append(
            CorpusAdapterIssue(
                severity=AdapterIssueSeverity.BLOCKING,
                code=code,
                detail=detail,
                finding_id=finding_id,
            )
        )

    if config.question_id != manifest.question_id:
        block(
            "native_claim_question_config_mismatch",
            "The embedded locked question config does not match the claim question_id.",
        )
    if sorted(manifest.protocol.inclusion_criteria) != sorted(config.eligibility.include):
        block(
            "native_protocol_inclusion_config_mismatch",
            "The claim protocol inclusion criteria are not the exact locked extraction criteria.",
        )
    if sorted(manifest.protocol.exclusion_criteria) != sorted(config.eligibility.exclude):
        block(
            "native_protocol_exclusion_config_mismatch",
            "The claim protocol exclusion criteria are not the exact locked extraction criteria.",
        )

    outcome = manifest.claim.outcome_name
    canonical_endpoint = map_outcome_family(outcome, config.outcomes.endpoint_map) or outcome
    primary_family = config.outcomes.primary_family
    configured_family = map_outcome_family(canonical_endpoint, config.outcomes.family_map)
    outcome_allowed = (
        outcome == primary_family
        or (
            canonical_endpoint in config.outcomes.included_primary_endpoints
            and configured_family == primary_family
        )
    )
    if not outcome_allowed:
        block(
            "native_claim_outcome_config_mismatch",
            "The claim outcome is not licensed by the locked primary endpoint registry.",
        )

    if manifest.claim.estimand is None or not manifest.claim.estimand.strip():
        block(
            "native_claim_estimand_missing",
            "A release-capable native claim must state the exact estimand used by its contrast.",
        )
    else:
        target_estimates = [
            estimate
            for estimate in corpus.graph.outcome_estimates
            if estimate.outcome_name == outcome
            and (
                manifest.claim.contrast_id is None
                or estimate.contrast_id == manifest.claim.contrast_id
            )
        ]
        contrast_by_id = {
            contrast.contrast_id: contrast for contrast in corpus.graph.contrasts
        }
        targeted_contrasts = [
            contrast_by_id[estimate.contrast_id]
            for estimate in target_estimates
            if estimate.contrast_id in contrast_by_id
        ]
        if targeted_contrasts and any(
            contrast.estimand != manifest.claim.estimand
            for contrast in targeted_contrasts
        ):
            block(
                "native_claim_estimand_graph_mismatch",
                "At least one target contrast does not carry the claim's exact estimand.",
            )

    moderator_by_name = {moderator.name: moderator for moderator in config.moderators}
    if manifest.qualified_target is not None:
        for condition in manifest.qualified_target.conditions:
            moderator = moderator_by_name.get(condition.moderator)
            if moderator is None:
                block(
                    "native_claim_condition_not_configured",
                    f"Claim condition {condition.moderator!r} is absent from the locked config.",
                )
                continue
            if moderator.role != "tested":
                block(
                    "native_claim_condition_not_prespecified_tested",
                    f"Claim condition {condition.moderator!r} was not a tested moderator.",
                )
            values: list[Any]
            if condition.operator.value == "equals":
                values = [condition.value]
            elif condition.operator.value == "in":
                values = list(condition.values)
            else:
                values = [condition.lower, condition.upper]
            if any(
                value is None or not _native_condition_value_compatible(value, moderator)
                for value in values
            ):
                block(
                    "native_claim_condition_value_config_mismatch",
                    f"Claim condition {condition.moderator!r} violates its locked type/domain.",
                )
    if manifest.global_condition_target is not None:
        for moderator_name in manifest.global_condition_target.moderator_names:
            moderator = moderator_by_name.get(moderator_name)
            if moderator is None:
                block(
                    "native_global_condition_moderator_not_configured",
                    (
                        f"Global condition moderator {moderator_name!r} is absent "
                        "from the locked extraction config."
                    ),
                )
            elif moderator.role != "tested":
                block(
                    "native_global_condition_moderator_not_prespecified_tested",
                    (
                        f"Global condition moderator {moderator_name!r} was not "
                        "prospectively designated as tested."
                    ),
                )

    for estimate in corpus.graph.outcome_estimates:
        mapped_endpoint = (
            map_outcome_family(estimate.outcome_name, config.outcomes.endpoint_map)
            or estimate.outcome_name
        )
        mapped_family = map_outcome_family(mapped_endpoint, config.outcomes.family_map)
        if mapped_endpoint not in config.outcomes.included_primary_endpoints or (
            mapped_family != primary_family
        ):
            block(
                "native_extracted_outcome_config_mismatch",
                "An extracted estimate is outside the locked primary endpoint registry.",
                finding_id=estimate.estimate_id,
            )
        for name, value in estimate.effect.moderators.items():
            moderator = moderator_by_name.get(name)
            if moderator is None or not _native_condition_value_compatible(value, moderator):
                block(
                    "native_extracted_moderator_config_mismatch",
                    "An extracted moderator name/value violates the locked config.",
                    finding_id=estimate.estimate_id,
                )
    return issues


def _qualified_verdict_score(
    verdict: QualifiedClaimVerdict,
    target: ClaimTargetV2,
) -> float:
    """Map a meaningful-effect margin to a bounded sensitivity score.

    This is used only to rank counterfactual audit influence.  It is deliberately
    identified as a score, not a posterior probability or a truth probability.
    """

    if verdict.decision_margin is None:
        return 0.0
    scale = target.meaningful_effect_threshold.minimum_magnitude
    standardized = max(-40.0, min(40.0, verdict.decision_margin / scale))
    return 1.0 / (1.0 + math.exp(-standardized))


def build_graph_counterfactual_audits(
    graph: EvidenceGraph,
    *,
    target: ClaimTarget,
    release_config: ClaimReleaseConfig,
    policy: AuditPolicyConfig,
    qualified_target: ClaimTargetV2 | None = None,
    global_condition_target: GlobalConditionDependenceTargetV1 | None = None,
    probability_overrides: dict[str, tuple[float, ProbabilityBasis, str]] | None = None,
) -> tuple[ClaimModel, list[AuditCandidate], list[dict[str, Any]], dict[str, Any]]:
    """Rerun the real synthesis after removing each matching evidence estimate."""

    if qualified_target is not None and global_condition_target is not None:
        raise VerificationContractError("multiple_typed_claim_targets_forbidden")

    synthesis_kwargs: dict[str, Any] = {
        "outcome_name": target.outcome_name,
        "contrast_id": target.contrast_id,
        "require_explicit_timepoint": release_config.require_explicit_timepoint,
        "confidence_level": release_config.confidence_level,
        "assumed_within_cohort_correlation": (release_config.assumed_within_cohort_correlation),
        "prespecified_moderators": release_config.prespecified_condition_moderators,
        "condition_familywise_alpha": release_config.condition_familywise_alpha,
        "condition_min_cohorts_per_level": (release_config.condition_min_cohorts_per_level),
        "qualified_target": qualified_target,
    }
    baseline_synthesis = synthesize_evidence_graph(graph, **synthesis_kwargs)
    if qualified_target is None:
        selected_ids = {
            estimate.estimate_id
            for estimate in graph.outcome_estimates
            if estimate.outcome_name == target.outcome_name
            and (target.contrast_id is None or estimate.contrast_id == target.contrast_id)
        }
    else:
        qualified = baseline_synthesis.get("qualified_claim")
        if not isinstance(qualified, dict):
            raise VerificationContractError("qualified_synthesis_selection_missing")
        raw_matched = qualified.get("matched_estimate_ids")
        if not isinstance(raw_matched, list) or any(
            not isinstance(item, str) for item in raw_matched
        ):
            raise VerificationContractError("qualified_synthesis_selection_invalid")
        selected_ids = set(raw_matched)
    selected = sorted(
        (estimate for estimate in graph.outcome_estimates if estimate.estimate_id in selected_ids),
        key=lambda item: item.estimate_id,
    )
    unknown_error_overrides = sorted(set(policy.item_error_probabilities) - selected_ids)
    unknown_cost_overrides = sorted(set(policy.item_verification_minutes) - selected_ids)
    if unknown_error_overrides or unknown_cost_overrides:
        raise VerificationContractError(
            "audit_item_override_identity_unknown:"
            f"errors={unknown_error_overrides}:costs={unknown_cost_overrides}"
        )
    if probability_overrides is not None and set(probability_overrides) != selected_ids:
        raise VerificationContractError("item_risk_probability_identity_mismatch")
    if not selected:
        return (
            ClaimModel(
                intercept=0.0,
                decision_threshold=policy.decision_threshold,
                claim_id=f"{target.outcome_name}-{target.direction.value}",
            ),
            [],
            [],
            baseline_synthesis,
        )

    if probability_overrides is None:
        errors = {
            estimate.estimate_id: policy.item_error_probabilities.get(
                estimate.estimate_id, policy.error_probability
            )
            for estimate in selected
        }
        probability_bases = {
            estimate.estimate_id: policy.probability_basis for estimate in selected
        }
        probability_sources = {
            estimate.estimate_id: policy.probability_source for estimate in selected
        }
    else:
        errors = {item_id: probability_overrides[item_id][0] for item_id in selected_ids}
        probability_bases = {item_id: probability_overrides[item_id][1] for item_id in selected_ids}
        probability_sources = {
            item_id: probability_overrides[item_id][2] for item_id in selected_ids
        }
    costs = {
        estimate.estimate_id: policy.item_verification_minutes.get(
            estimate.estimate_id, policy.verification_minutes_per_item
        )
        for estimate in selected
    }
    disagreement = {
        estimate.estimate_id: _disagreement_score(estimate, target.direction)
        for estimate in selected
    }
    if qualified_target is not None:
        baseline_verdict = classify_qualified_synthesis_evidence(
            baseline_synthesis,
            target=qualified_target,
            require_prediction_interval_stability=(
                release_config.require_prediction_interval_stability
            ),
        )
        baseline_score = _qualified_verdict_score(
            baseline_verdict,
            qualified_target,
        )
        claim_model = ClaimModel(
            intercept=0.0,
            decision_threshold=0.5,
            claim_id=qualified_target.claim_id,
        )
        candidates: list[AuditCandidate] = []
        counterfactual_rows: list[dict[str, Any]] = []
        for estimate in selected:
            item_id = estimate.estimate_id
            counterfactual = synthesize_evidence_graph(
                graph,
                excluded_estimate_ids=[item_id],
                **synthesis_kwargs,
            )
            verdict = classify_qualified_synthesis_evidence(
                counterfactual,
                target=qualified_target,
                require_prediction_interval_stability=(
                    release_config.require_prediction_interval_stability
                ),
            )
            counterfactual_score = _qualified_verdict_score(
                verdict,
                qualified_target,
            )
            candidates.append(
                AuditCandidate(
                    item_id=item_id,
                    baseline_contribution=0.0,
                    counterfactual_contribution=0.0,
                    error_probability=errors[item_id],
                    probability_basis=probability_bases[item_id],
                    probability_source=probability_sources[item_id],
                    verification_cost=costs[item_id],
                    cost_unit="person_minutes",
                    disagreement_score=disagreement[item_id],
                    scenario_kind=ScenarioKind.LEAVE_ONE_OUT,
                    scenario_source=(
                        "actual_condition_qualified_synthesis_leave_one_out_rerun;"
                        "sensitivity_scenario_not_oracle_correction"
                    ),
                    baseline_decision_score=baseline_score,
                    counterfactual_decision_score=counterfactual_score,
                    decision_score_source=("scaled_meaningful_effect_margin_not_truth_probability"),
                    baseline_decision=baseline_verdict.synthesis_gate_passed,
                    counterfactual_decision=verdict.synthesis_gate_passed,
                )
            )
            counterfactual_rows.append(
                {
                    "baseline_decision": baseline_verdict.model_dump(mode="json"),
                    "baseline_synthesis_sha256": hash_canonical(baseline_synthesis),
                    "counterfactual_decision": verdict.model_dump(mode="json"),
                    "counterfactual_synthesis": counterfactual,
                    "counterfactual_synthesis_sha256": hash_canonical(counterfactual),
                    "item_id": item_id,
                    "scenario": "leave_one_out_actual_qualified_synthesis_rerun",
                }
            )
        return claim_model, candidates, counterfactual_rows, baseline_synthesis

    if global_condition_target is not None:
        expected_moderators = global_condition_target.moderator_names

        def exploratory_decision(value: dict[str, Any]) -> bool:
            analysis = value.get("condition_analysis")
            if not isinstance(analysis, dict):
                raise VerificationContractError(
                    "global_condition_development_analysis_missing"
                )
            analyses = analysis.get("analyses")
            if not isinstance(analyses, list):
                raise VerificationContractError(
                    "global_condition_development_analysis_invalid"
                )
            observed = sorted(
                str(row.get("moderator"))
                for row in analyses
                if isinstance(row, dict) and isinstance(row.get("moderator"), str)
            )
            if observed != expected_moderators:
                raise VerificationContractError(
                    "global_condition_development_moderator_family_mismatch"
                )
            return (
                analysis.get("status")
                == "exploratory_qualitative_condition_signal"
            )

        baseline_decision = exploratory_decision(baseline_synthesis)
        claim_model = ClaimModel(
            intercept=0.0,
            decision_threshold=0.5,
            claim_id=global_condition_target.claim_id,
        )
        candidates = []
        counterfactual_rows = []
        for estimate in selected:
            item_id = estimate.estimate_id
            counterfactual = synthesize_evidence_graph(
                graph,
                excluded_estimate_ids=[item_id],
                **synthesis_kwargs,
            )
            counterfactual_decision = exploratory_decision(counterfactual)
            candidates.append(
                AuditCandidate(
                    item_id=item_id,
                    baseline_contribution=0.0,
                    counterfactual_contribution=0.0,
                    error_probability=errors[item_id],
                    probability_basis=probability_bases[item_id],
                    probability_source=probability_sources[item_id],
                    verification_cost=costs[item_id],
                    cost_unit="person_minutes",
                    disagreement_score=disagreement[item_id],
                    scenario_kind=ScenarioKind.LEAVE_ONE_OUT,
                    scenario_source=(
                        "actual_development_condition_synthesis_leave_one_out_rerun;"
                        "heldout_confirmation_outcomes_firewalled;"
                        "sensitivity_scenario_not_oracle_correction"
                    ),
                    baseline_decision_score=float(baseline_decision),
                    counterfactual_decision_score=float(counterfactual_decision),
                    decision_score_source=(
                        "binary_development_condition_signal_not_truth_probability"
                    ),
                    baseline_decision=baseline_decision,
                    counterfactual_decision=counterfactual_decision,
                )
            )
            counterfactual_rows.append(
                {
                    "baseline_decision": {
                        "classification": "exploratory_condition_signal",
                        "qualifies": baseline_decision,
                    },
                    "baseline_synthesis_sha256": hash_canonical(baseline_synthesis),
                    "counterfactual_decision": {
                        "classification": "exploratory_condition_signal",
                        "qualifies": counterfactual_decision,
                    },
                    "counterfactual_synthesis": counterfactual,
                    "counterfactual_synthesis_sha256": hash_canonical(counterfactual),
                    "item_id": item_id,
                    "scenario": (
                        "leave_one_out_actual_development_condition_synthesis_rerun"
                    ),
                }
            )
        return claim_model, candidates, counterfactual_rows, baseline_synthesis

    plan = build_graph_counterfactual_audit_plan(
        graph,
        outcome_name=target.outcome_name,
        target_direction=target.direction.value,
        error_probabilities=errors,
        verification_costs=costs,
        probability_basis=probability_bases,
        probability_source="artifact-specific-item-risk-source",
        cost_unit="person_minutes",
        disagreement_scores=disagreement,
        contrast_id=target.contrast_id,
        require_explicit_timepoint=release_config.require_explicit_timepoint,
        require_prediction_interval_stability=(
            release_config.require_prediction_interval_stability
        ),
        confidence_level=release_config.confidence_level,
        assumed_within_cohort_correlation=(release_config.assumed_within_cohort_correlation),
        prespecified_moderators=release_config.prespecified_condition_moderators,
        condition_familywise_alpha=release_config.condition_familywise_alpha,
        condition_min_cohorts_per_level=(release_config.condition_min_cohorts_per_level),
        claim_id=f"{target.outcome_name}-{target.direction.value}",
    )
    plan_candidates = [
        replace(
            candidate,
            probability_source=probability_sources[candidate.item_id],
        )
        for candidate in plan.candidates
    ]
    counterfactual_rows = [
        {
            "baseline_decision": plan.baseline_decision.model_dump(mode="json"),
            "baseline_synthesis_sha256": hash_canonical(plan.baseline_synthesis),
            "counterfactual_decision": plan.counterfactual_decisions[candidate.item_id].model_dump(
                mode="json"
            ),
            "counterfactual_synthesis": plan.counterfactual_syntheses[candidate.item_id],
            "counterfactual_synthesis_sha256": hash_canonical(
                plan.counterfactual_syntheses[candidate.item_id]
            ),
            "item_id": candidate.item_id,
            "scenario": "leave_one_out_actual_synthesis_rerun",
        }
        for candidate in plan_candidates
    ]
    return (
        plan.claim_model,
        plan_candidates,
        counterfactual_rows,
        plan.baseline_synthesis,
    )


def verifier_pipeline_components() -> tuple[PipelineComponentSpec, ...]:
    """Return the explicit code/prompt surface defining the supported verifier."""

    return (
        PipelineComponentSpec(
            component_id="runtime-contract",
            component_version="3",
            file_paths=[
                "pyproject.toml",
                "src/literature_multiverse/__init__.py",
                "src/literature_multiverse/config.py",
                "src/literature_multiverse/lineage.py",
                "src/literature_multiverse/models.py",
                "src/literature_multiverse/paths.py",
                "src/literature_multiverse/records.py",
                "src/literature_multiverse/schemas.py",
                "uv.lock",
            ],
            settings={
                "dependency_lock_bound": True,
                "installed_dependency_versions": {
                    name: distribution_version(name)
                    for name in (
                        "jsonschema",
                        "numpy",
                        "pandas",
                        "pyarrow",
                        "pydantic",
                        "PyYAML",
                        "scikit-learn",
                        "scipy",
                    )
                },
                "platform_machine": platform.machine(),
                "platform_system": platform.system(),
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
                "shared_contract_helpers_bound": True,
            },
        ),
        PipelineComponentSpec(
            component_id="native-extraction",
            component_version="10",
            file_paths=[
                "configs/benchmarks/native-antiox-bounded-v1.json",
                "prompts/native_candidate_inventory.md",
                "prompts/native_candidate_packet.md",
                "prompts/native_extraction.md",
                "scripts/build_native_source_manifest.py",
                "scripts/build_typed_evidence_corpus.py",
                "scripts/reconcile_native_cohorts.py",
                "scripts/run_native_bounded_ollama_diagnostic.py",
                "scripts/run_native_ollama_diagnostic.py",
                "scripts/s3_extract_typed.py",
                "src/literature_multiverse/cohort_reconciliation.py",
                "src/literature_multiverse/extract.py",
                "src/literature_multiverse/grounding.py",
                "src/literature_multiverse/live.py",
                "src/literature_multiverse/local_ollama.py",
                "src/literature_multiverse/metasyn_benchmark.py",
                "src/literature_multiverse/metasyn_retrieval.py",
                "src/literature_multiverse/native_bounded_generation.py",
                "src/literature_multiverse/native_bounded_ollama_diagnostic.py",
                "src/literature_multiverse/native_extraction.py",
                "src/literature_multiverse/native_grounding.py",
                "src/literature_multiverse/native_ollama_diagnostic.py",
                "src/literature_multiverse/normalize.py",
                "src/literature_multiverse/paperclip_cli.py",
                "src/literature_multiverse/prompting.py",
                "src/literature_multiverse/source_manifest_bridge.py",
                "src/literature_multiverse/typed_extraction.py",
            ],
            settings={
                "contract": "publication-fragment-v3",
                "estimable_requires_exact_grounding_receipt": True,
                "grounding_package_contract": "typed-evidence-grounding-package-v4",
                "complete_source_manifest_and_corpus_cutoff_required": True,
                "cross_publication_reconciliation_receipt_required": True,
                "exact_extraction_execution_context_required": True,
                "in_repository_dependency_closure_bound": True,
                "native_extraction_entry_points": [
                    "scripts/run_native_bounded_ollama_diagnostic.py",
                    "scripts/run_native_ollama_diagnostic.py",
                    "scripts/s3_extract_typed.py",
                ],
                "bounded_generation_contract": "native-bounded-two-stage-generation-v1",
                "bounded_pre_call_intent_required": True,
                "bounded_exact_source_projection_quote_grounding_required": True,
                "bounded_whole_publication_assembly": "all_or_nothing",
                "free_text_identity_auto_merge": False,
            },
        ),
        PipelineComponentSpec(
            component_id="scientific-synthesis",
            component_version="3",
            file_paths=[
                "scripts/confirm_condition_dependence.py",
                "src/literature_multiverse/claim_semantics.py",
                "src/literature_multiverse/condition_confirmation.py",
                "src/literature_multiverse/effects.py",
                "src/literature_multiverse/evidence_graph.py",
                "src/literature_multiverse/independence_identity.py",
                "src/literature_multiverse/meta_analysis.py",
            ],
            settings={
                "cohort_unit": "explicit",
                "global_condition_dependence": (
                    "prespecified-development-fit-plus-heldout-confirmation"
                ),
                "heldout_confirmation_unit": "authority-linked-independence-component",
                "qualified_claims": True,
                "same_corpus_moderator_analysis": "exploratory_only",
            },
        ),
        PipelineComponentSpec(
            component_id="verification-release",
            component_version="9",
            file_paths=[
                "scripts/build_condition_calibration_trajectory.py",
                "scripts/build_question_replay_state.py",
                "scripts/calibrate_adaptive_release.py",
                "scripts/calibrate_item_risk.py",
                "scripts/evaluate_question_benchmark.py",
                "src/literature_multiverse/adaptive_calibration.py",
                "src/literature_multiverse/audit_session.py",
                "src/literature_multiverse/budgeted_verification.py",
                "src/literature_multiverse/calibration.py",
                "src/literature_multiverse/certificate.py",
                "src/literature_multiverse/claim_release.py",
                "src/literature_multiverse/cli.py",
                "src/literature_multiverse/condition_trajectory_builder.py",
                "src/literature_multiverse/item_risk_artifacts.py",
                "src/literature_multiverse/item_risk_calibration.py",
                "src/literature_multiverse/pipeline_fingerprint.py",
                "src/literature_multiverse/production_policy.py",
                "src/literature_multiverse/question_evaluation.py",
                "src/literature_multiverse/sequential_verification.py",
                "src/literature_multiverse/verifier.py",
            ],
            settings={
                "adaptive_calibration_unit": "independent_complete_question_trajectory",
                "adaptive_calibration_roster": "preregistered_label_free_complete_questions",
                "adaptive_stopping_rule": "first_full_release_from_prefix_zero",
                "audit_cost_unit": "total_person_minutes",
                "certificate_contract": (
                    "legacy-v5-or-prebundle-collection-source-receipt-then-"
                    "immutable-production-v6-to-final-v7"
                ),
                "condition_calibration_contract": (
                    "prebundle-collection-source-external-replay-to-receipt-roster-"
                    "then-confirmation-aware-complete-question-trajectory-v2"
                ),
                "condition_calibration_outcome_opening": (
                    "frozen-source-roster-membership-before-assessment"
                ),
                "condition_multi_arm_trajectory_construction": (
                    "independent-single-arm-pass-then-canonical-builder-then-shared-trajectory-pass"
                ),
                "condition_outcome_firewall": "development-only-online-policy",
                "fixed_state_calibration_scope": "single_decision_only",
                "item_risk_contract": "self-contained-scoring-receipt-v2",
                "in_repository_dependency_closure_bound": True,
                "release_contract": "adaptive-first-release-v5",
                "corrected_item_risk_projection": (
                    "source-receipt-to-unchanged-unresolved-items"
                ),
                "uncalibrated_selection_requires_analysis_opt_in": True,
            },
        ),
    )


def compute_verifier_pipeline_fingerprint(*, root: Path | None = None) -> PipelineFingerprint:
    """Hash the actual supported implementation rather than trusting a declared ID."""

    repository_root = root or Path(__file__).resolve().parents[2]
    return compute_pipeline_fingerprint(
        root=repository_root,
        components=verifier_pipeline_components(),
    )


def _pipeline_identity(
    manifest: ClaimManifest,
    *,
    expected: PipelineFingerprint | None,
    root: Path | None,
) -> tuple[PipelineFingerprintVerification, str]:
    repository_root = root or Path(__file__).resolve().parents[2]
    frozen = expected or compute_verifier_pipeline_fingerprint(root=repository_root)
    verification = require_pipeline_fingerprint_match(
        expected=frozen,
        root=repository_root,
    )
    if (
        manifest.pipeline_sha256 is not None
        and manifest.pipeline_sha256 != verification.computed_pipeline_sha256
    ):
        raise VerificationContractError(
            "claim_manifest_pipeline_sha256_does_not_match_computed_pipeline"
        )
    basis = (
        "verified_expected_pipeline_artifact"
        if expected is not None
        else "computed_and_self_verified_at_run_start"
    )
    return verification, basis


def _audit_scope_ids(
    *,
    graph: EvidenceGraph,
    target: ClaimTarget,
    qualified_target: ClaimTargetV2 | None,
    config: ClaimReleaseConfig,
) -> list[str]:
    if qualified_target is None:
        return sorted(
            estimate.estimate_id
            for estimate in graph.outcome_estimates
            if estimate.outcome_name == target.outcome_name
            and (target.contrast_id is None or estimate.contrast_id == target.contrast_id)
        )
    synthesis = synthesize_evidence_graph(
        graph,
        outcome_name=qualified_target.outcome_name,
        contrast_id=qualified_target.contrast_id,
        require_explicit_timepoint=config.require_explicit_timepoint,
        confidence_level=config.confidence_level,
        assumed_within_cohort_correlation=config.assumed_within_cohort_correlation,
        prespecified_moderators=config.prespecified_condition_moderators,
        condition_familywise_alpha=config.condition_familywise_alpha,
        condition_min_cohorts_per_level=config.condition_min_cohorts_per_level,
        qualified_target=qualified_target,
    )
    qualified = synthesis.get("qualified_claim")
    if not isinstance(qualified, dict):
        raise VerificationContractError("qualified_synthesis_selection_missing")
    matched = qualified.get("matched_estimate_ids")
    if not isinstance(matched, list) or any(not isinstance(item, str) for item in matched):
        raise VerificationContractError("qualified_synthesis_selection_invalid")
    return sorted(matched)


def _artifact_backed_item_probabilities(
    *,
    manifest: ClaimManifest,
    graph: EvidenceGraph,
    expected_item_ids: list[str],
    pipeline_verification: PipelineFingerprintVerification,
    bundle: ItemRiskCalibrationBundle | None,
    candidates: list[ItemRiskCandidate] | None,
    stale_candidate_item_ids: set[str] | None = None,
) -> tuple[
    dict[str, tuple[float, ProbabilityBasis, str]] | None,
    list[RiskBound],
]:
    if (bundle is None) != (candidates is None):
        raise VerificationContractError("item_risk_bundle_and_candidates_must_be_supplied_together")
    if bundle is None or candidates is None:
        return None, []
    if manifest.audit.item_error_probabilities:
        raise VerificationContractError(
            "manifest_item_error_probabilities_conflict_with_item_risk_artifacts"
        )
    try:
        rows = [
            ItemRiskCandidate.model_validate(candidate.model_dump(mode="json"))
            for candidate in candidates
        ]
    except (AttributeError, ValueError) as exc:
        raise VerificationContractError("item_risk_candidate_contract_invalid") from exc
    rows.sort(key=lambda candidate: candidate.item_id)
    item_ids = [candidate.item_id for candidate in rows]
    if item_ids != expected_item_ids:
        raise VerificationContractError(
            "item_risk_candidate_identity_mismatch:"
            f"expected={expected_item_ids}:observed={item_ids}"
        )
    estimates = {estimate.estimate_id: estimate for estimate in graph.outcome_estimates}
    stale_ids = stale_candidate_item_ids or set()
    if not stale_ids <= set(expected_item_ids):
        raise VerificationContractError("item_risk_stale_candidate_identity_unknown")
    overrides: dict[str, tuple[float, ProbabilityBasis, str]] = {}
    bounds: list[RiskBound] = []
    for candidate in rows:
        estimate = estimates.get(candidate.item_id)
        if estimate is None:
            raise VerificationContractError(
                f"item_risk_candidate_evidence_unknown:{candidate.item_id}"
            )
        if (
            candidate.question_id != manifest.question_id
            or candidate.paper_id != estimate.effect.paper_id
            or candidate.population_id != manifest.population_id
            or candidate.domain != manifest.domain
            or candidate.pipeline_sha256 != pipeline_verification.computed_pipeline_sha256
        ):
            raise VerificationContractError(
                f"item_risk_candidate_scope_mismatch:{candidate.item_id}"
            )
        if (
            candidate.item_id not in stale_ids
            and candidate.score_input_sha256 != hash_canonical(estimate)
        ):
            raise VerificationContractError(
                f"item_risk_candidate_source_snapshot_mismatch:{candidate.item_id}"
            )
        try:
            bound = score_item_risk_bound(
                candidate=candidate,
                bundle=bundle,
                pipeline_verification=pipeline_verification,
            )
        except ItemRiskCalibrationError as exc:
            raise VerificationContractError(
                f"item_risk_scoring_failed:{candidate.item_id}:{exc}"
            ) from exc
        bounds.append(bound)
        if bound.status == "cell_rate_ucl_available":
            fields = verified_audit_cell_rate_ucl_fields(
                bound=bound,
                bundle=bundle,
                pipeline_verification=pipeline_verification,
            )
            overrides[candidate.item_id] = (
                float(fields["item_cell_rate_ucl"]),
                ProbabilityBasis.CALIBRATED_CELL_RATE_UCL,
                (
                    f"{fields['rate_source']}:scheduling-only:"
                    f"{fields['estimand']}"
                ),
            )
        else:
            overrides[candidate.item_id] = (
                candidate.risk_score,
                ProbabilityBasis.HEURISTIC,
                (
                    f"uncertified-item-risk-score:{candidate.score_model_sha256}:"
                    f"{bound.status}:{bound.risk_bound_sha256}"
                ),
            )
    return overrides, bounds


def _project_item_risk_candidates_after_audit_corrections(
    *,
    graph: EvidenceGraph,
    expected_item_ids: list[str],
    candidates: list[ItemRiskCandidate],
    resolved_item_ids: set[str],
) -> list[ItemRiskCandidate]:
    """Project a source-bound scoring receipt onto unchanged unresolved evidence.

    A correction may modify or remove only the selected estimate.  Its original
    risk record remains cryptographic history but is scheduling-irrelevant after
    resolution.  Every unresolved estimate must retain the exact original source
    snapshot; new or changed unresolved items fail closed.
    """

    by_id = {candidate.item_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise VerificationContractError("item_risk_candidate_identity_duplicate")
    expected = set(expected_item_ids)
    candidate_ids = set(by_id)
    missing = sorted(expected - candidate_ids)
    if missing:
        raise VerificationContractError(
            "item_risk_projection_new_unscored_items:" + ",".join(missing)
        )
    removed_unresolved = sorted((candidate_ids - expected) - resolved_item_ids)
    if removed_unresolved:
        raise VerificationContractError(
            "item_risk_projection_removed_unresolved_items:"
            + ",".join(removed_unresolved)
        )
    estimates = {estimate.estimate_id: estimate for estimate in graph.outcome_estimates}
    changed_unresolved = sorted(
        item_id
        for item_id in expected - resolved_item_ids
        if item_id not in estimates
        or by_id[item_id].score_input_sha256 != hash_canonical(estimates[item_id])
    )
    if changed_unresolved:
        raise VerificationContractError(
            "item_risk_projection_changed_unresolved_items:"
            + ",".join(changed_unresolved)
        )
    return [by_id[item_id] for item_id in sorted(expected)]


def sequential_candidates_from_prepared_state(
    *,
    manifest: ClaimManifest,
    prepared: PreparedVerificationScientificState,
) -> tuple[CurrentAuditCandidate, ...]:
    """Adapt recomputed audit candidates to the resumable scheduler contract."""

    if not prepared.audit_candidates:
        return ()
    counterfactual_hashes = {
        str(row["item_id"]): str(row["counterfactual_synthesis_sha256"])
        for row in prepared.counterfactuals
    }
    risk_hashes = {bound.item_id: bound.risk_bound_sha256 for bound in prepared.item_risk_bounds}
    return current_candidates_from_audit_candidates(
        prepared.audit_candidates,
        prepared.claim_model,
        policy=manifest.release.audit_allocation_policy,
        counterfactual_synthesis_sha256s=counterfactual_hashes,
        risk_bound_sha256s=risk_hashes,
        seed=manifest.release.audit_seed,
    )


def compute_verification_policy_sha256(manifest: ClaimManifest) -> str:
    """Bind a sequential audit session to the complete frozen claim contract.

    The generic audit ledger calls this value a ``policy_sha256``, but the verifier
    must bind more than the ranking policy. Claim identity, target conditions,
    corpus cutoff, adapter settings, and every release/audit setting can all change
    the meaning of a graph or correction. Hashing the complete normalized manifest
    prevents a valid state from being resumed or corrected under a different claim
    whose numerical synthesis happens to look identical.
    """

    return hash_canonical(
        {
            "policy_context_version": "verification-manifest-context-v1",
            "claim_manifest": manifest.model_dump(mode="json"),
        }
    )


def complete_corpus_identity_for_adaptive_calibration(
    *,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
) -> CompleteCorpusIdentity:
    """Bind overlap checks to every frozen publication, never target estimates only."""

    native_manifest = corpus.metadata.get("native_source_manifest")
    source_manifest_sha256 = corpus.metadata.get("source_manifest_sha256")
    if isinstance(native_manifest, dict):
        records = native_manifest.get("records")
        if not isinstance(records, list) or not isinstance(source_manifest_sha256, str):
            raise VerificationContractError(
                "adaptive_calibration_native_source_manifest_invalid"
            )
        if hash_canonical(native_manifest) != source_manifest_sha256:
            raise VerificationContractError(
                "adaptive_calibration_native_source_manifest_hash_mismatch"
            )
        try:
            publication_ids = sorted(
                str(record["publication"]["publication_id"])
                for record in records
            )
        except (KeyError, TypeError) as exc:
            raise VerificationContractError(
                "adaptive_calibration_native_publication_membership_invalid"
            ) from exc
        graph_publication_ids = sorted(
            publication.publication_id for publication in corpus.graph.publications
        )
        if publication_ids != graph_publication_ids:
            raise VerificationContractError(
                "adaptive_calibration_complete_publication_membership_mismatch"
            )
    else:
        publication_ids = sorted(
            publication.publication_id for publication in corpus.graph.publications
        )
        source_manifest_sha256 = None
    return freeze_complete_corpus_identity(
        corpus_id=corpus.corpus_id,
        corpus_source_sha256=corpus.source_sha256,
        corpus_cutoff=manifest.protocol.corpus_cutoff,
        publication_ids=publication_ids,
        source_manifest_sha256=source_manifest_sha256,
    )


def build_verifier_adaptive_policy_context(
    *,
    manifest: ClaimManifest,
    pipeline_sha256: str,
    budget_minutes: float,
    policy_arm_id: str,
) -> AdaptivePolicyContext:
    """Project the exact verifier manifest into the calibrated policy context."""

    target_semantics: dict[str, Any] = {
        "claim_manifest_version": manifest.claim_manifest_version,
        "claim_direction": manifest.claim.direction.value,
        "outcome_name": manifest.claim.outcome_name,
        "contrast_id": manifest.claim.contrast_id,
        "estimand": manifest.claim.estimand,
        "qualified_target": (
            None
            if manifest.qualified_target is None
            else manifest.qualified_target.model_dump(mode="json")
        ),
        "global_condition_target": (
            None
            if manifest.global_condition_target is None
            else {
                "target_sha256": manifest.global_condition_target.target_sha256,
                "semantics": (
                    "prospectively frozen global condition-dependence target"
                ),
            }
        ),
        "decision_loss": "released_claim_decision_differs_from_reference_verdict",
    }
    return freeze_adaptive_policy_context(
        policy_arm_id=policy_arm_id,
        population_id=manifest.population_id,
        pipeline_sha256=pipeline_sha256,
        allocation_policy={
            "name": manifest.release.audit_allocation_policy.value,
            "seed": manifest.release.audit_seed,
        },
        budget_minutes=budget_minutes,
        release_config=manifest.release.model_dump(mode="json"),
        audit_config={
            "scheduler_inputs": manifest.audit.model_dump(mode="json"),
            "release_guard": manifest.audit_guard.model_dump(mode="json"),
        },
        target_semantics=target_semantics,
        corpus_protocol_context=manifest.protocol.model_dump(mode="json"),
        score_feature_names=CLAIM_RELEASE_RISK_FEATURE_NAMES,
    )


def _derive_verifier_adaptive_policy_context(
    *,
    manifest: ClaimManifest,
    pipeline_sha256: str,
    budget_minutes: float,
    bundle: AdaptiveCalibrationBundle,
) -> AdaptivePolicyContext:
    """Select the one calibrated arm that exactly matches deployed production.

    A prospective caller supplies only the immutable calibration bundle. The
    verifier recomputes every candidate context from its own manifest, pipeline,
    and budget and will not accept a caller-authored arm choice. A calibrated
    bundle's selected threshold fixes the arm. An abstain-all bundle is usable
    only when exactly one frozen arm matches the deployed verifier.
    """

    matches = [
        context
        for context in bundle.development_freeze.policy_contexts
        if context
        == build_verifier_adaptive_policy_context(
            manifest=manifest,
            pipeline_sha256=pipeline_sha256,
            budget_minutes=budget_minutes,
            policy_arm_id=context.policy_arm_id,
        )
    ]
    if bundle.selected is not None:
        selected_arm_id = bundle.selected.candidate.policy_arm_id
        selected_matches = [
            context for context in matches if context.policy_arm_id == selected_arm_id
        ]
        if len(selected_matches) != 1:
            raise VerificationContractError(
                "adaptive_calibration_selected_policy_context_mismatch"
            )
        return selected_matches[0]
    if len(matches) != 1:
        raise VerificationContractError(
            "adaptive_calibration_abstain_all_policy_context_ambiguous"
        )
    return matches[0]


def _derive_verifier_adaptive_policy_context_v2(
    *,
    manifest: ClaimManifest,
    pipeline_sha256: str,
    budget_minutes: float,
    bundle: AdaptiveCalibrationBundleV2,
) -> AdaptivePolicyContext:
    """Select the exact manifest-v3 arm from a confirmation-aware bundle."""

    try:
        bundle = validate_adaptive_calibration_bundle_v2_integrity(bundle)
    except AdaptiveCalibrationError as exc:
        raise VerificationContractError(
            f"condition_adaptive_calibration_bundle_invalid:{exc}"
        ) from exc
    contexts = bundle.development_freeze.base_freeze.policy_contexts
    matches = [
        context
        for context in contexts
        if context
        == build_verifier_adaptive_policy_context(
            manifest=manifest,
            pipeline_sha256=pipeline_sha256,
            budget_minutes=budget_minutes,
            policy_arm_id=context.policy_arm_id,
        )
    ]
    if bundle.selected is not None:
        selected_arm = bundle.selected.candidate.policy_arm_id
        matches = [row for row in matches if row.policy_arm_id == selected_arm]
    if len(matches) != 1:
        raise VerificationContractError(
            "condition_adaptive_calibration_policy_context_mismatch"
        )
    return matches[0]


def compute_synthesis_runner_sha256(
    *, manifest: ClaimManifest, pipeline_sha256: str
) -> str:
    """Identity of the deterministic production synthesis rerun."""

    return hash_canonical(
        {
            "pipeline_sha256": pipeline_sha256,
            "release_config": manifest.release,
            "qualified_target": manifest.qualified_target,
            "global_condition_target": manifest.global_condition_target,
            "runner": "prepare_verification_scientific_state:synthesis",
        }
    )


def compute_candidate_runner_sha256(
    *, manifest: ClaimManifest, pipeline_sha256: str
) -> str:
    """Identity of the deterministic production counterfactual rerun."""

    return hash_canonical(
        {
            "audit_policy": manifest.audit,
            "pipeline_sha256": pipeline_sha256,
            "release_config": manifest.release,
            "runner": "prepare_verification_scientific_state:counterfactuals",
        }
    )


def prepare_condition_verification_context(
    *,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
    complete_corpus_identity: CompleteCorpusIdentity,
    plan: ConditionConfirmationPlanV1,
    development_graph: EvidenceGraph,
    frozen_model: ConditionConfirmationFrozenModelV1,
    pipeline_sha256: str,
    current_full_graph: EvidenceGraph | None = None,
) -> PreparedConditionVerificationContext:
    """Recompute the complete manifest-v3 development/confirmation boundary.

    The full graph is opened here only by the contract-validation layer to replay the
    custodian partition and content-silent receipt.  The returned online context
    exposes only the development graph; synthesis, item-risk scoring, counterfactuals,
    and scheduling never receive the confirmation partition.
    """

    if manifest.claim_manifest_version != "3" or manifest.global_condition_target is None:
        raise VerificationContractError("condition_context_requires_manifest_v3")
    if corpus.extraction_context is None:
        raise VerificationContractError(
            "condition_context_requires_v4_native_extraction_context"
        )
    try:
        plan = ConditionConfirmationPlanV1.model_validate(plan.model_dump(mode="json"))
        full_graph = EvidenceGraph.model_validate(
            (current_full_graph or corpus.graph).model_dump(mode="json")
        )
        development_graph = EvidenceGraph.model_validate(
            development_graph.model_dump(mode="json")
        )
        frozen_model = ConditionConfirmationFrozenModelV1.model_validate(
            frozen_model.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise VerificationContractError("condition_context_input_tampered") from exc
    target = manifest.global_condition_target
    expected_target = freeze_condition_confirmation_target(
        question_id=manifest.question_id,
        claim_spec_sha256=target.target_sha256,
        question_config_sha256=(
            corpus.extraction_context.question_config_sha256
        ),
        corpus_snapshot_sha256=complete_corpus_identity.membership_sha256,
        corpus_cutoff=manifest.protocol.corpus_cutoff,
        outcome_name=target.outcome_name,
        claim_contrast_id=target.contrast_id,
        contrast_label=target.contrast_label,
        contrast_estimand=target.estimand,
        positive_direction_means=target.positive_direction_means,
        treatment_role=target.treatment_role,
        comparator_role=target.comparator_role,
        measure=target.measure,
        unit=target.unit,
        moderator_names=target.moderator_names,
    )
    if plan.target != expected_target:
        raise VerificationContractError("condition_plan_manifest_target_mismatch")
    if plan.pipeline_sha256 != pipeline_sha256:
        raise VerificationContractError("condition_plan_pipeline_mismatch")
    if plan.full_graph_sha256 != hash_canonical(full_graph):
        raise VerificationContractError("condition_plan_full_graph_mismatch")
    try:
        roster, expected_development, _confirmation, receipt = (
            materialize_condition_confirmation_inputs(
                full_graph=full_graph,
                target=expected_target,
            )
        )
    except ConditionConfirmationError as exc:
        raise VerificationContractError(
            f"condition_materialization_replay_failed:{exc}"
        ) from exc
    if (
        plan.roster != roster
        or plan.materialization_receipt != receipt
        or plan.development_graph_sha256 != hash_canonical(expected_development)
        or development_graph != expected_development
    ):
        raise VerificationContractError(
            "condition_materialization_or_development_graph_mismatch"
        )
    try:
        frozen_model = validate_condition_confirmation_model(
            plan=plan,
            development_graph=development_graph,
            model=frozen_model,
            current_pipeline_sha256=pipeline_sha256,
        )
        target_semantics = freeze_adaptive_target_semantics_v2(
            question_id=manifest.question_id,
            claim_spec_sha256=target.target_sha256,
            global_condition_target_sha256=target.target_sha256,
        )
        independence_identity = adaptive_independence_identity_from_condition_plan_v1(
            plan
        )
        projection = freeze_condition_calibration_projection(
            question_id=manifest.question_id,
            target_semantics=target_semantics,
            independence_identity=independence_identity,
            question_config_sha256=expected_target.question_config_sha256,
            corpus_snapshot_sha256=complete_corpus_identity.membership_sha256,
            corpus_cutoff=manifest.protocol.corpus_cutoff,
            plan_sha256=plan.plan_sha256,
            materialization_receipt_sha256=plan.materialization_receipt_sha256,
            full_graph_sha256=plan.full_graph_sha256,
            development_graph_sha256=plan.development_graph_sha256,
            confirmation_graph_sha256=plan.confirmation_graph_sha256,
            development_partition_sha256=(
                plan.development_partition.partition_sha256
            ),
            confirmation_partition_sha256=(
                plan.confirmation_partition.partition_sha256
            ),
            confirmation_config_sha256=plan.config_sha256,
            pipeline_sha256=pipeline_sha256,
            synthesis_runner_sha256=compute_synthesis_runner_sha256(
                manifest=manifest,
                pipeline_sha256=pipeline_sha256,
            ),
            candidate_runner_sha256=compute_candidate_runner_sha256(
                manifest=manifest,
                pipeline_sha256=pipeline_sha256,
            ),
            prespecified_moderator_names=target.moderator_names,
        )
    except (AdaptiveCalibrationError, ConditionConfirmationError, ValueError) as exc:
        raise VerificationContractError(
            f"condition_context_projection_failed:{exc}"
        ) from exc
    blockers: list[str] = []
    if plan.status != "ready":
        blockers.append("condition_confirmation_plan_insufficient")
    if frozen_model.status != "fitted":
        blockers.append("condition_confirmation_development_model_insufficient")
    if independence_identity.verification_status != "verified":
        blockers.append("condition_confirmation_strong_independence_unverified")
    return PreparedConditionVerificationContext(
        plan=plan,
        development_graph=development_graph,
        frozen_model=frozen_model,
        target_semantics=target_semantics,
        independence_identity=independence_identity,
        projection=projection,
        ordinary_blocking_reasons=tuple(sorted(set(blockers))),
    )


def _condition_full_graph_from_development_state(
    *,
    source_full_graph: EvidenceGraph,
    development_graph: EvidenceGraph,
    plan: ConditionConfirmationPlanV1,
) -> EvidenceGraph:
    """Join corrected development evidence to the never-online confirmation split."""

    confirmation_graph = partition_evidence_graph(
        source_full_graph,
        plan.confirmation_partition,
    )
    return _merge_graphs([development_graph, confirmation_graph])


def _rebuild_condition_context_from_graphs(
    *,
    manifest: ClaimManifest,
    complete_corpus_identity: CompleteCorpusIdentity,
    source_full_graph: EvidenceGraph,
    development_graph: EvidenceGraph,
    template_plan: ConditionConfirmationPlanV1,
    pipeline_sha256: str,
    exact_current_full_graph: EvidenceGraph | None = None,
) -> tuple[PreparedConditionVerificationContext, EvidenceGraph]:
    """Rebuild one outcome-free condition context from frozen graph bytes alone."""

    current_full_graph = (
        exact_current_full_graph
        if exact_current_full_graph is not None
        else _condition_full_graph_from_development_state(
            source_full_graph=source_full_graph,
            development_graph=development_graph,
            plan=template_plan,
        )
    )
    target = manifest.global_condition_target
    if target is None:
        raise VerificationContractError("condition_rebuild_requires_global_target")
    expected_target = freeze_condition_confirmation_target(
        question_id=manifest.question_id,
        claim_spec_sha256=target.target_sha256,
        question_config_sha256=template_plan.target.question_config_sha256,
        corpus_snapshot_sha256=complete_corpus_identity.membership_sha256,
        corpus_cutoff=manifest.protocol.corpus_cutoff,
        outcome_name=target.outcome_name,
        claim_contrast_id=target.contrast_id,
        contrast_label=target.contrast_label,
        contrast_estimand=target.estimand,
        positive_direction_means=target.positive_direction_means,
        treatment_role=target.treatment_role,
        comparator_role=target.comparator_role,
        measure=target.measure,
        unit=target.unit,
        moderator_names=target.moderator_names,
    )
    if template_plan.target != expected_target:
        raise VerificationContractError("condition_rebuild_target_mismatch")
    try:
        roster, expected_development, _confirmation, receipt = (
            materialize_condition_confirmation_inputs(
                full_graph=current_full_graph,
                target=expected_target,
            )
        )
        if expected_development != development_graph:
            raise VerificationContractError(
                "condition_historical_development_partition_mismatch"
            )
        rebuilt_plan = prepare_condition_confirmation_plan(
            target=expected_target,
            config=template_plan.config,
            roster=roster,
            materialization_receipt=receipt,
            pipeline_sha256=pipeline_sha256,
            external_freeze_anchor=template_plan.external_freeze_anchor,
        )
        rebuilt_model = fit_condition_confirmation_model(
            rebuilt_plan,
            development_graph,
            current_pipeline_sha256=pipeline_sha256,
        )
        target_semantics = freeze_adaptive_target_semantics_v2(
            question_id=manifest.question_id,
            claim_spec_sha256=target.target_sha256,
            global_condition_target_sha256=target.target_sha256,
        )
        independence_identity = adaptive_independence_identity_from_condition_plan_v1(
            rebuilt_plan
        )
        projection = freeze_condition_calibration_projection(
            question_id=manifest.question_id,
            target_semantics=target_semantics,
            independence_identity=independence_identity,
            question_config_sha256=expected_target.question_config_sha256,
            corpus_snapshot_sha256=complete_corpus_identity.membership_sha256,
            corpus_cutoff=manifest.protocol.corpus_cutoff,
            plan_sha256=rebuilt_plan.plan_sha256,
            materialization_receipt_sha256=(
                rebuilt_plan.materialization_receipt_sha256
            ),
            full_graph_sha256=rebuilt_plan.full_graph_sha256,
            development_graph_sha256=rebuilt_plan.development_graph_sha256,
            confirmation_graph_sha256=rebuilt_plan.confirmation_graph_sha256,
            development_partition_sha256=(
                rebuilt_plan.development_partition.partition_sha256
            ),
            confirmation_partition_sha256=(
                rebuilt_plan.confirmation_partition.partition_sha256
            ),
            confirmation_config_sha256=rebuilt_plan.config_sha256,
            pipeline_sha256=pipeline_sha256,
            synthesis_runner_sha256=compute_synthesis_runner_sha256(
                manifest=manifest,
                pipeline_sha256=pipeline_sha256,
            ),
            candidate_runner_sha256=compute_candidate_runner_sha256(
                manifest=manifest,
                pipeline_sha256=pipeline_sha256,
            ),
            prespecified_moderator_names=target.moderator_names,
        )
    except (AdaptiveCalibrationError, ConditionConfirmationError, ValueError) as exc:
        raise VerificationContractError(
            f"condition_historical_context_rebuild_failed:{exc}"
        ) from exc
    blockers: list[str] = []
    if rebuilt_plan.status != "ready":
        blockers.append("condition_confirmation_plan_insufficient")
    if rebuilt_model.status != "fitted":
        blockers.append("condition_confirmation_development_model_insufficient")
    if independence_identity.verification_status != "verified":
        blockers.append("condition_confirmation_strong_independence_unverified")
    return (
        PreparedConditionVerificationContext(
            plan=rebuilt_plan,
            development_graph=development_graph,
            frozen_model=rebuilt_model,
            target_semantics=target_semantics,
            independence_identity=independence_identity,
            projection=projection,
            ordinary_blocking_reasons=tuple(sorted(set(blockers))),
        ),
        current_full_graph,
    )


def _rebuild_condition_context_for_development_state(
    *,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
    complete_corpus_identity: CompleteCorpusIdentity,
    development_graph: EvidenceGraph,
    template_plan: ConditionConfirmationPlanV1,
    pipeline_sha256: str,
) -> tuple[PreparedConditionVerificationContext, EvidenceGraph]:
    """Recompute a historical plan/model without opening terminal assessment labels."""

    source_development = partition_evidence_graph(
        corpus.graph,
        template_plan.development_partition,
    )
    return _rebuild_condition_context_from_graphs(
        manifest=manifest,
        complete_corpus_identity=complete_corpus_identity,
        source_full_graph=corpus.graph,
        development_graph=development_graph,
        template_plan=template_plan,
        pipeline_sha256=pipeline_sha256,
        exact_current_full_graph=(
            corpus.graph if development_graph == source_development else None
        ),
    )


def _require_exact_prepared_sequential_artifacts(
    *,
    label: str,
    state_synthesis: dict[str, Any],
    state_candidates: list[CurrentAuditCandidate],
    prepared: PreparedVerificationScientificState,
    manifest: ClaimManifest,
) -> None:
    expected_candidates = list(
        sequential_candidates_from_prepared_state(
            manifest=manifest,
            prepared=prepared,
        )
    )
    if state_synthesis != prepared.synthesis:
        raise VerificationContractError(
            f"sequential_audit_{label}_synthesis_recomputation_mismatch"
        )
    if state_candidates != expected_candidates:
        raise VerificationContractError(
            f"sequential_audit_{label}_candidate_recomputation_mismatch"
        )


def _replay_corrected_sequential_science(
    *,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
    state: SequentialVerificationState,
    pipeline_verification: PipelineFingerprintVerification,
    budget_minutes: float,
    item_risk_calibration_bundle: ItemRiskCalibrationBundle | None,
    item_risk_candidates: list[ItemRiskCandidate] | None,
) -> PreparedVerificationScientificState:
    """Recompute every correction from the source graph and reject injected state.

    Generic state validation replays selection, checkpoint, receipt, and cost
    transitions. This production bridge additionally reruns the actual verifier
    synthesis and counterfactual builders for the source state and every scientific
    correction before allowing the final state into release assessment.
    """

    try:
        current = resume_sequential_verification_state(state)
    except SequentialVerificationContractError as exc:
        raise VerificationContractError(
            f"sequential_audit_transition_replay_failed:{exc}"
        ) from exc
    pipeline_sha256 = pipeline_verification.computed_pipeline_sha256
    if pipeline_sha256 is None:
        raise VerificationContractError("computed_pipeline_identity_missing")
    if current.session.pipeline_sha256 != pipeline_sha256:
        raise VerificationContractError("sequential_audit_state_pipeline_mismatch")
    if current.session.policy_sha256 != compute_verification_policy_sha256(manifest):
        raise VerificationContractError(
            "sequential_audit_state_claim_manifest_context_mismatch"
        )
    if not math.isclose(
        current.session.budget,
        float(budget_minutes),
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise VerificationContractError("sequential_audit_state_budget_mismatch")
    if current.session.cost_unit != "person_minutes":
        raise VerificationContractError("sequential_audit_state_cost_unit_mismatch")
    if current.initial_graph != corpus.graph:
        raise VerificationContractError(
            "sequential_audit_source_evidence_graph_mismatch"
        )

    prepared = prepare_verification_scientific_state(
        manifest=manifest,
        graph=corpus.graph,
        pipeline_verification=pipeline_verification,
        item_risk_calibration_bundle=item_risk_calibration_bundle,
        item_risk_candidates=item_risk_candidates,
    )
    _require_exact_prepared_sequential_artifacts(
        label="initial",
        state_synthesis=current.initial_synthesis,
        state_candidates=current.initial_candidates,
        prepared=prepared,
        manifest=manifest,
    )
    synthesis_runner_sha256 = compute_synthesis_runner_sha256(
        manifest=manifest,
        pipeline_sha256=pipeline_sha256,
    )
    candidate_runner_sha256 = compute_candidate_runner_sha256(
        manifest=manifest,
        pipeline_sha256=pipeline_sha256,
    )
    correction_index = 0
    resolved_item_ids: set[str] = set()
    for transition in current.transitions:
        if transition.transition_kind != "correction":
            continue
        correction_index += 1
        assert transition.action is not None
        resolved_item_ids.add(transition.action.item_id)
        assert transition.post_graph is not None
        assert transition.post_synthesis is not None
        assert transition.post_candidates is not None
        assert transition.correction_provenance is not None
        provenance = transition.correction_provenance
        if (
            provenance.synthesis_runner_sha256 != synthesis_runner_sha256
            or provenance.candidate_runner_sha256 != candidate_runner_sha256
        ):
            raise VerificationContractError(
                "sequential_audit_correction_runner_identity_mismatch:"
                f"{correction_index}"
            )
        try:
            prepared = prepare_verification_scientific_state(
                manifest=manifest,
                graph=transition.post_graph,
                pipeline_verification=pipeline_verification,
                item_risk_calibration_bundle=item_risk_calibration_bundle,
                item_risk_candidates=item_risk_candidates,
                resolved_item_ids_for_risk_projection=resolved_item_ids,
            )
        except VerificationContractError as exc:
            raise VerificationContractError(
                "sequential_audit_correction_science_replay_failed:"
                f"{correction_index}:{exc}"
            ) from exc
        _require_exact_prepared_sequential_artifacts(
            label=f"correction_{correction_index}",
            state_synthesis=transition.post_synthesis,
            state_candidates=transition.post_candidates,
            prepared=prepared,
            manifest=manifest,
        )
    _require_exact_prepared_sequential_artifacts(
        label="current",
        state_synthesis=current.synthesis,
        state_candidates=current.candidates,
        prepared=prepared,
        manifest=manifest,
    )
    return prepared


def recompute_verifier_adaptive_preselection_checkpoint(
    *,
    manifest: ClaimManifest,
    state: SequentialVerificationState,
    pipeline_verification: PipelineFingerprintVerification,
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None,
    blocking_adapter_reasons: list[str],
) -> AdaptivePreselectionState:
    """Recompute one historical checkpoint from its complete scientific state.

    This is also called lazily by certificate-v5 validation. It deliberately reruns
    synthesis, counterfactual audit candidates, all non-calibration release gates,
    and the risk-feature projection instead of trusting checkpoint-authored semantic
    fields merely because their unkeyed hashes are self-consistent.
    """

    try:
        current = resume_sequential_verification_state(state)
    except SequentialVerificationContractError as exc:
        raise VerificationContractError(
            f"adaptive_checkpoint_state_replay_failed:{exc}"
        ) from exc
    if current.session.active_action is not None:
        raise VerificationContractError(
            "adaptive_checkpoint_recompute_requires_preselection_state"
        )
    pipeline_sha256 = pipeline_verification.computed_pipeline_sha256
    if pipeline_sha256 is None:
        raise VerificationContractError("computed_pipeline_identity_missing")
    if (
        current.session.pipeline_sha256 != pipeline_sha256
        or current.session.policy_sha256
        != compute_verification_policy_sha256(manifest)
    ):
        raise VerificationContractError(
            "adaptive_checkpoint_verifier_context_mismatch"
        )
    item_risk_bundle = None
    item_risk_candidates = None
    if item_risk_scoring_receipt is not None:
        if item_risk_scoring_receipt.pipeline_verification != pipeline_verification:
            raise VerificationContractError(
                "adaptive_checkpoint_item_risk_pipeline_mismatch"
            )
        item_risk_bundle = item_risk_scoring_receipt.calibration_bundle
        item_risk_candidates = list(item_risk_scoring_receipt.candidates)
    prepared = prepare_verification_scientific_state(
        manifest=manifest,
        graph=current.graph,
        pipeline_verification=pipeline_verification,
        item_risk_calibration_bundle=item_risk_bundle,
        item_risk_candidates=item_risk_candidates,
        resolved_item_ids_for_risk_projection=set(
            current.session.resolved_item_ids
        ),
    )
    _require_exact_prepared_sequential_artifacts(
        label="adaptive_checkpoint",
        state_synthesis=current.synthesis,
        state_candidates=current.candidates,
        prepared=prepared,
        manifest=manifest,
    )
    assessment_kwargs = {
        "graph": current.graph,
        "question_id": manifest.question_id,
        "population_id": manifest.population_id,
        "domain": manifest.domain,
        "pipeline_sha256": pipeline_sha256,
        "audit_candidates": list(prepared.audit_candidates),
        "claim_model": prepared.claim_model,
        "audit_resolution_receipts": [],
        "audit_budget": current.session.budget,
        "frozen_calibration_bundle": None,
        "adaptive_calibration_bundle": None,
        "adaptive_release_candidate": None,
        "external_noncalibration_blocking_reasons": blocking_adapter_reasons,
        "config": manifest.release,
        "audit_guard_config": manifest.audit_guard.to_runtime(),
        "sequential_audit_state": current,
    }
    if manifest.qualified_target is None:
        assessment = assess_claim_release(
            target=prepared.target,
            **assessment_kwargs,
        )
    else:
        assessment = assess_qualified_claim_release(
            target=manifest.qualified_target,
            **assessment_kwargs,
        )
    try:
        return freeze_preselection_state_from_production_components(
            sequential_state=current,
            release_assessment=assessment,
            blocking_adapter_reasons=blocking_adapter_reasons,
        )
    except AdaptiveCalibrationError as exc:
        raise VerificationContractError(
            f"adaptive_checkpoint_projection_failed:{exc}"
        ) from exc


def _create_initial_sequential_audit_state(
    *,
    manifest: ClaimManifest,
    graph: EvidenceGraph,
    synthesis: dict[str, Any],
    candidates: list[AuditCandidate],
    claim_model: ClaimModel,
    counterfactuals: list[dict[str, Any]],
    item_risk_bounds: list[RiskBound],
    pipeline_sha256: str,
    budget_minutes: float,
    created_at: datetime,
    adaptive_policy_context_sha256: str | None = None,
    adaptive_calibration_bundle_sha256: str | None = None,
) -> SequentialVerificationState:
    prepared = PreparedVerificationScientificState(
        target=ClaimTarget(
            direction=manifest.claim.direction,
            outcome_name=manifest.claim.outcome_name,
            contrast_id=manifest.claim.contrast_id,
        ),
        claim_model=claim_model,
        audit_candidates=tuple(candidates),
        counterfactuals=tuple(counterfactuals),
        synthesis=synthesis,
        item_risk_bounds=tuple(item_risk_bounds),
    )
    current_candidates = sequential_candidates_from_prepared_state(
        manifest=manifest,
        prepared=prepared,
    )
    policy_sha256 = compute_verification_policy_sha256(manifest)
    identity = hash_canonical(
        {
            "question_id": manifest.question_id,
            "graph_sha256": hash_canonical(graph),
            "pipeline_sha256": pipeline_sha256,
            "policy_sha256": policy_sha256,
            "budget_minutes": float(budget_minutes),
        }
    )
    state = create_sequential_verification_state(
        session_id=f"verify-session-{identity[:16]}",
        created_at=created_at,
        pipeline_sha256=pipeline_sha256,
        policy_sha256=policy_sha256,
        budget=budget_minutes,
        cost_unit="person_minutes",
        graph=graph,
        synthesis=synthesis,
        candidates=current_candidates,
        adaptive_policy_context_sha256=adaptive_policy_context_sha256,
        adaptive_calibration_bundle_sha256=adaptive_calibration_bundle_sha256,
    )
    # Selection is deliberately deferred until ``run_verification`` has evaluated
    # every release gate against this exact frozen no-active-action state.  This is
    # the same temporal contract used by the question-level policy evaluator: stop
    # at the first *full* release-eligible state, otherwise open one next action.
    return state


def _condition_missing_gate(
    context: PreparedConditionVerificationContext,
) -> ConditionConfirmationGateAssessmentV1:
    return freeze_condition_confirmation_gate_assessment(
        provisional_claim_decision="condition_dependent",
        status="missing",
        reasons=["condition_confirmation_required"],
        condition_projection_sha256=context.projection.projection_sha256,
        target_sha256=context.projection.condition_target_sha256,
        plan_sha256=context.plan.plan_sha256,
        config_sha256=context.plan.config_sha256,
    )


def _condition_terminal_actions(
    state: SequentialVerificationState,
) -> list[AdaptiveTerminalAuditCandidate]:
    return [
        AdaptiveTerminalAuditCandidate(
            item_id=candidate.item_id,
            eligible=candidate.eligible,
            estimated_cost_minutes=candidate.estimated_cost,
            source_candidate_sha256=candidate.candidate_sha256,
        )
        for candidate in sorted(state.candidates, key=lambda row: row.item_id)
    ]


def _condition_corpus_issues(
    *,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
    pipeline_sha256: str,
) -> list[CorpusAdapterIssue]:
    issues = list(corpus.adapter_issues)
    if not corpus.provenance_release_eligible():
        issues.append(
            CorpusAdapterIssue(
                severity=AdapterIssueSeverity.BLOCKING,
                code="unverified_source_provenance",
                detail=(
                    "Manifest-v3 condition verification requires a fully replayed v4 "
                    "native grounding package and exact extraction context."
                ),
            )
        )
    if corpus.metadata.get("pipeline_fingerprint_sha256") != pipeline_sha256:
        issues.append(
            CorpusAdapterIssue(
                severity=AdapterIssueSeverity.BLOCKING,
                code="corpus_pipeline_identity_mismatch",
                detail="The native corpus and current computed verifier pipeline differ.",
            )
        )
    if corpus.corpus_id != manifest.question_id:
        issues.append(
            CorpusAdapterIssue(
                severity=AdapterIssueSeverity.BLOCKING,
                code="corpus_question_identity_mismatch",
                detail="The native corpus question does not match manifest v3.",
            )
        )
    if corpus.metadata.get("native_corpus_cutoff") != manifest.protocol.corpus_cutoff:
        issues.append(
            CorpusAdapterIssue(
                severity=AdapterIssueSeverity.BLOCKING,
                code="corpus_cutoff_identity_mismatch",
                detail="The membership-bound corpus cutoff does not match manifest v3.",
            )
        )
    issues.extend(_native_claim_config_compatibility_issues(manifest=manifest, corpus=corpus))
    if any(row.status == "pending" for row in corpus.eligibility):
        issues.append(
            CorpusAdapterIssue(
                severity=AdapterIssueSeverity.BLOCKING,
                code="eligibility_screening_incomplete",
                detail="Every frozen corpus record requires a terminal inclusion decision.",
            )
        )
    unique = {
        (issue.finding_id or "", issue.paper_id or "", issue.code): issue
        for issue in issues
    }
    return [unique[key] for key in sorted(unique)]


def _run_condition_verification(
    *,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
    budget_minutes: float,
    adaptive_calibration_bundle_v2: AdaptiveCalibrationBundleV2,
    condition_plan: ConditionConfirmationPlanV1,
    condition_development_graph: EvidenceGraph,
    condition_frozen_model: ConditionConfirmationFrozenModelV1,
    expected_pipeline_fingerprint: PipelineFingerprint | None,
    pipeline_root: Path | None,
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None,
    sequential_audit_state: SequentialVerificationState | None,
    generated_at: datetime,
) -> ConditionVerificationCertificateV6:
    """Execute manifest v3 without exposing confirmation outcomes to online policy."""

    pipeline_verification, pipeline_basis = _pipeline_identity(
        manifest,
        expected=expected_pipeline_fingerprint,
        root=pipeline_root,
    )
    pipeline_sha256 = pipeline_verification.computed_pipeline_sha256
    if pipeline_sha256 is None:
        raise VerificationContractError("computed_pipeline_identity_missing")
    try:
        bundle_v2 = validate_adaptive_calibration_bundle_v2_integrity(
            adaptive_calibration_bundle_v2
        )
    except AdaptiveCalibrationError as exc:
        raise VerificationContractError(
            f"condition_adaptive_calibration_bundle_invalid:{exc}"
        ) from exc
    complete_identity = complete_corpus_identity_for_adaptive_calibration(
        manifest=manifest,
        corpus=corpus,
    )
    policy_context = _derive_verifier_adaptive_policy_context_v2(
        manifest=manifest,
        pipeline_sha256=pipeline_sha256,
        budget_minutes=budget_minutes,
        bundle=bundle_v2,
    )
    try:
        parsed_plan = ConditionConfirmationPlanV1.model_validate(
            condition_plan.model_dump(mode="json")
        )
        supplied_development = EvidenceGraph.model_validate(
            condition_development_graph.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise VerificationContractError("condition_runtime_input_tampered") from exc

    state = None
    if sequential_audit_state is not None:
        try:
            state = resume_sequential_verification_state(sequential_audit_state)
        except SequentialVerificationContractError as exc:
            raise VerificationContractError(
                f"condition_sequential_state_invalid:{exc}"
            ) from exc
        if (
            state.session.pipeline_sha256 != pipeline_sha256
            or state.session.policy_sha256
            != compute_verification_policy_sha256(manifest)
            or not math.isclose(
                state.session.budget,
                budget_minutes,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or state.adaptive_policy_context_sha256
            != policy_context.policy_context_sha256
            or state.adaptive_calibration_bundle_sha256 != bundle_v2.bundle_sha256
        ):
            raise VerificationContractError(
                "condition_sequential_state_context_mismatch"
            )
        if supplied_development != state.graph:
            raise VerificationContractError(
                "condition_supplied_development_graph_not_current_state"
            )
        source_development = partition_evidence_graph(
            corpus.graph,
            parsed_plan.development_partition,
        )
        if state.initial_graph != source_development:
            raise VerificationContractError(
                "condition_sequential_initial_development_graph_mismatch"
            )
        current_full_graph = _condition_full_graph_from_development_state(
            source_full_graph=corpus.graph,
            development_graph=state.graph,
            plan=parsed_plan,
        )
    else:
        current_full_graph = corpus.graph
    context = prepare_condition_verification_context(
        manifest=manifest,
        corpus=corpus,
        complete_corpus_identity=complete_identity,
        plan=parsed_plan,
        development_graph=supplied_development,
        frozen_model=condition_frozen_model,
        pipeline_sha256=pipeline_sha256,
        current_full_graph=current_full_graph,
    )

    item_risk_bundle = None
    item_risk_candidates = None
    if item_risk_scoring_receipt is not None:
        try:
            item_risk_scoring_receipt = ItemRiskScoringRunReceipt.model_validate(
                item_risk_scoring_receipt.model_dump(mode="json")
            )
        except (AttributeError, ValueError) as exc:
            raise VerificationContractError(
                "condition_item_risk_scoring_receipt_invalid"
            ) from exc
        if item_risk_scoring_receipt.pipeline_verification != pipeline_verification:
            raise VerificationContractError(
                "condition_item_risk_pipeline_mismatch"
            )
        source_estimates = {
            row.estimate_id: row
            for row in (
                state.initial_graph if state is not None else context.development_graph
            ).outcome_estimates
        }
        stale = sorted(
            candidate.item_id
            for candidate in item_risk_scoring_receipt.candidates
            if candidate.item_id not in source_estimates
            or candidate.score_input_sha256
            != hash_canonical(source_estimates[candidate.item_id])
        )
        if stale:
            raise VerificationContractError(
                f"condition_item_risk_source_snapshot_mismatch:{stale}"
            )
        item_risk_bundle = item_risk_scoring_receipt.calibration_bundle
        item_risk_candidates = list(item_risk_scoring_receipt.candidates)

    prepared = prepare_verification_scientific_state(
        manifest=manifest,
        graph=context.development_graph,
        pipeline_verification=pipeline_verification,
        item_risk_calibration_bundle=item_risk_bundle,
        item_risk_candidates=item_risk_candidates,
        resolved_item_ids_for_risk_projection=(
            set() if state is None else set(state.session.resolved_item_ids)
        ),
    )
    if state is None:
        state = _create_initial_sequential_audit_state(
            manifest=manifest,
            graph=context.development_graph,
            synthesis=prepared.synthesis,
            candidates=list(prepared.audit_candidates),
            claim_model=prepared.claim_model,
            counterfactuals=list(prepared.counterfactuals),
            item_risk_bounds=list(prepared.item_risk_bounds),
            pipeline_sha256=pipeline_sha256,
            budget_minutes=budget_minutes,
            created_at=generated_at,
            adaptive_policy_context_sha256=policy_context.policy_context_sha256,
            adaptive_calibration_bundle_sha256=bundle_v2.bundle_sha256,
        )
    else:
        _require_exact_prepared_sequential_artifacts(
            label="condition_current",
            state_synthesis=state.synthesis,
            state_candidates=state.candidates,
            prepared=prepared,
            manifest=manifest,
        )

    corpus_issues = _condition_corpus_issues(
        manifest=manifest,
        corpus=corpus,
        pipeline_sha256=pipeline_sha256,
    )
    blocking_adapter_reasons = sorted(
        f"adapter:{issue.code}"
        for issue in corpus_issues
        if issue.severity is AdapterIssueSeverity.BLOCKING
    )
    missing_gate = _condition_missing_gate(context)

    def assess_condition_state(
        checkpoint_state: SequentialVerificationState,
        checkpoint_context: PreparedConditionVerificationContext,
        checkpoint_prepared: PreparedVerificationScientificState,
        gate: ConditionConfirmationGateAssessmentV1,
    ) -> ConditionClaimReleaseAssessmentV1:
        return assess_global_condition_claim_release_source(
            graph=checkpoint_context.development_graph,
            question_id=manifest.question_id,
            population_id=manifest.population_id,
            domain=manifest.domain,
            pipeline_sha256=pipeline_sha256,
            target=manifest.global_condition_target,
            condition_calibration_projection=checkpoint_context.projection,
            condition_confirmation_gate=gate,
            audit_candidates=list(checkpoint_prepared.audit_candidates),
            claim_model=checkpoint_prepared.claim_model,
            audit_resolution_receipts=[],
            audit_budget=budget_minutes,
            condition_noncalibration_reasons=(
                checkpoint_context.ordinary_blocking_reasons
            ),
            external_noncalibration_blocking_reasons=blocking_adapter_reasons,
            config=manifest.release,
            audit_guard_config=manifest.audit_guard.to_runtime(),
            sequential_audit_state=checkpoint_state,
        )

    try:
        history, history_context_sha, history_bundle_sha = (
            adaptive_preselection_history_from_state(state)
        )
        predecessor_states = selection_predecessor_states_from_state(state)
    except SequentialVerificationContractError as exc:
        raise VerificationContractError(
            f"condition_adaptive_history_invalid:{exc}"
        ) from exc
    if (
        history_context_sha != policy_context.policy_context_sha256
        or history_bundle_sha != bundle_v2.bundle_sha256
        or len(history) != len(predecessor_states)
    ):
        raise VerificationContractError("condition_adaptive_history_identity_mismatch")
    for index, (checkpoint, predecessor) in enumerate(
        zip(history, predecessor_states, strict=True)
    ):
        historical_context, _historical_full = (
            _rebuild_condition_context_for_development_state(
                manifest=manifest,
                corpus=corpus,
                complete_corpus_identity=complete_identity,
                development_graph=predecessor.graph,
                template_plan=context.plan,
                pipeline_sha256=pipeline_sha256,
            )
        )
        historical_prepared = prepare_verification_scientific_state(
            manifest=manifest,
            graph=predecessor.graph,
            pipeline_verification=pipeline_verification,
            item_risk_calibration_bundle=item_risk_bundle,
            item_risk_candidates=item_risk_candidates,
            resolved_item_ids_for_risk_projection=set(
                predecessor.session.resolved_item_ids
            ),
        )
        _require_exact_prepared_sequential_artifacts(
            label=f"condition_history_{index}",
            state_synthesis=predecessor.synthesis,
            state_candidates=predecessor.candidates,
            prepared=historical_prepared,
            manifest=manifest,
        )
        historical_assessment = assess_condition_state(
            predecessor,
            historical_context,
            historical_prepared,
            _condition_missing_gate(historical_context),
        )
        try:
            recomputed_checkpoint = freeze_preselection_state_from_production_components(
                sequential_state=predecessor,
                release_assessment=historical_assessment,
                blocking_adapter_reasons=blocking_adapter_reasons,
            )
        except AdaptiveCalibrationError as exc:
            raise VerificationContractError(
                f"condition_history_projection_failed:{index}:{exc}"
            ) from exc
        if recomputed_checkpoint != checkpoint:
            raise VerificationContractError(
                f"condition_history_checkpoint_mismatch:{index}"
            )

    current_preselection = None
    if state.session.active_action is None:
        preselection_assessment = assess_condition_state(
            state,
            context,
            prepared,
            missing_gate,
        )
        try:
            current_preselection = freeze_preselection_state_from_production_components(
                sequential_state=state,
                release_assessment=preselection_assessment,
                blocking_adapter_reasons=blocking_adapter_reasons,
            )
        except AdaptiveCalibrationError as exc:
            raise VerificationContractError(
                f"condition_current_projection_failed:{exc}"
            ) from exc
        observed_states = [*history, current_preselection]
    else:
        preselection_assessment = assess_condition_state(
            state,
            context,
            prepared,
            missing_gate,
        )
        if not history:
            raise VerificationContractError(
                "condition_active_action_missing_selection_checkpoint"
            )
        observed_states = list(history)
    base_candidate = freeze_prospective_adaptive_candidate(
        question_id=manifest.question_id,
        population_id=manifest.population_id,
        domain=manifest.domain,
        policy_arm_id=policy_context.policy_arm_id,
        policy_context_sha256=policy_context.policy_context_sha256,
        corpus=complete_identity,
        observed_states=observed_states,
    )

    invocation: ConditionGateInvocationProofV2 | None = None
    qualification: ConfirmationAwareReleaseQualificationProofV2 | None = None
    selection_result = None
    evaluated_state = state
    hard_context_blockers = bool(
        blocking_adapter_reasons or context.ordinary_blocking_reasons
    )
    if state.session.active_action is not None:
        stop_outcome = "active_action_in_progress"
    elif current_preselection is not None and current_preselection.non_calibration_gates_passed:
        invocation = freeze_condition_gate_invocation_proof_v2(
            terminal_preselection_state=current_preselection,
            condition_projection=context.projection,
            source_candidate_input_sha256=state.candidate_input_sha256,
            available_actions=_condition_terminal_actions(state),
            remaining_budget_minutes=state.session.remaining_budget,
        )
        try:
            qualification = freeze_confirmation_aware_release_qualification_proof_v2(
                question_id=manifest.question_id,
                policy_arm_id=policy_context.policy_arm_id,
                condition_gate_invocation_proof=invocation,
                bundle=bundle_v2,
            )
        except AdaptiveCalibrationError as exc:
            if str(exc) not in {
                "confirmation_v2_release_qualification_bundle_abstains_all",
                "confirmation_v2_release_qualification_risk_above_threshold",
            }:
                raise VerificationContractError(
                    f"condition_release_qualification_failed:{exc}"
                ) from exc
        stop_outcome = "condition_gate_ready"
    elif hard_context_blockers:
        stop_outcome = "condition_context_blocked"
    elif state.session.status.value == "active" and state.session.remaining_budget > 0:
        try:
            selection_result = select_next_audit_candidate(
                state,
                expected=freeze_state_expectation(state),
                selected_at=generated_at,
                adaptive_preselection_state=current_preselection,
                adaptive_policy_context_sha256=policy_context.policy_context_sha256,
                adaptive_calibration_bundle_sha256=bundle_v2.bundle_sha256,
            )
        except SequentialVerificationContractError as exc:
            if str(exc) != "no_eligible_candidate_fits_remaining_budget":
                raise VerificationContractError(
                    f"condition_audit_selection_failed:{exc}"
                ) from exc
            stop_outcome = "no_feasible_action"
        else:
            stop_outcome = "selected_next_action"
            state = selection_result.state
    else:
        stop_outcome = "no_feasible_action"

    gate = missing_gate
    source_assessment = assess_condition_state(
        evaluated_state,
        context,
        prepared,
        gate,
    )
    production_stop = freeze_condition_production_stop_decision_v2(
        evaluated_state=evaluated_state,
        release_assessment=preselection_assessment,
        blocking_adapter_reasons=blocking_adapter_reasons,
        outcome=stop_outcome,
        selection_result=selection_result,
        condition_gate_invocation_proof=invocation,
    )
    candidate_v2: ProspectiveAdaptiveReleaseCandidateV2 = (
        freeze_prospective_adaptive_candidate_v2(
            base_candidate=base_candidate,
            target_semantics=context.target_semantics,
            independence_identity=context.independence_identity,
            condition_projection=context.projection,
            condition_gate_invocation_proof=invocation,
            release_qualification_proof=qualification,
        )
    )
    adaptive_assessment_v2: AdaptiveProspectiveAssessmentV2 = (
        assess_confirmation_aware_adaptive_release_candidate(candidate_v2, bundle_v2)
    )
    candidate_payload = [asdict(candidate) for candidate in prepared.audit_candidates]
    lineage = [
        CertificateLineageStage(
            stage="condition_outcome_firewall",
            input_sha256s=dict(
                sorted(
                    {
                        "full_graph": hash_canonical(current_full_graph),
                        "materialization_receipt": (
                            context.plan.materialization_receipt_sha256
                        ),
                    }.items()
                )
            ),
            output_sha256s={
                "development_graph": context.plan.development_graph_sha256,
                "firewall_receipt": context.projection.firewall_receipt_sha256,
            },
            method="custodian-materialization-replay-development-only-online-v1",
        ),
        CertificateLineageStage(
            stage="condition_online_synthesis_and_audit",
            input_sha256s={
                "development_graph": context.plan.development_graph_sha256
            },
            output_sha256s=dict(
                sorted(
                    {
                        "candidate_input": hash_canonical(candidate_payload),
                        "synthesis": hash_canonical(prepared.synthesis),
                    }.items()
                )
            ),
            method="development-only-synthesis-counterfactual-audit-v1",
        ),
        CertificateLineageStage(
            stage="condition_terminal_gate",
            input_sha256s={
                "invocation_proof": (
                    hash_canonical(None) if invocation is None else invocation.proof_sha256
                )
            },
            output_sha256s={
                "gate_assessment": gate.gate_assessment_sha256
            },
            method=(
                "deferred-unopened"
            ),
        ),
    ]
    corpus_payload = corpus.certificate_payload()
    corpus_payload["declared_corpus_cutoff"] = manifest.protocol.corpus_cutoff
    corpus_payload["pipeline_identity_basis"] = pipeline_basis
    reasons = sorted(
        set(source_assessment.reasons) | set(blocking_adapter_reasons)
    )
    source_v6 = freeze_condition_verification_certificate_v6(
        generated_at=generated_at,
        reasons=reasons,
        claim_manifest=manifest.model_dump(mode="json"),
        corpus=corpus_payload,
        corpus_sha256=corpus.source_sha256,
        source_evidence_graph=corpus.graph,
        current_full_evidence_graph=current_full_graph,
        development_evidence_graph=context.development_graph,
        adapter_issues=[row.model_dump(mode="json") for row in corpus_issues],
        synthesis=prepared.synthesis,
        counterfactual_reruns=list(prepared.counterfactuals),
        audit_candidates=candidate_payload,
        release_assessment=source_assessment,
        pipeline_verification=pipeline_verification,
        complete_corpus_identity=complete_identity,
        item_risk_scoring_receipt=item_risk_scoring_receipt,
        condition_plan=context.plan,
        condition_frozen_model=context.frozen_model,
        condition_calibration_projection=context.projection,
        condition_confirmation_gate=gate,
        condition_target_semantics=context.target_semantics,
        condition_independence_identity=context.independence_identity,
        condition_gate_invocation_proof=invocation,
        release_qualification_proof=qualification,
        adaptive_policy_context=policy_context,
        adaptive_calibration_bundle_v2=bundle_v2,
        adaptive_release_candidate_v2=candidate_v2,
        adaptive_prospective_assessment_v2=adaptive_assessment_v2,
        sequential_audit_state=state,
        production_stop_decision=production_stop,
        lineage=lineage,
    )
    return source_v6


def run_condition_calibration_collection(
    *,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
    budget_minutes: float,
    collection_split: Literal["development", "calibration"],
    adaptive_policy_context: AdaptivePolicyContext,
    condition_plan: ConditionConfirmationPlanV1,
    condition_development_graph: EvidenceGraph,
    condition_frozen_model: ConditionConfirmationFrozenModelV1,
    policy_visible_question_trajectory: PolicyVisibleQuestionTrajectoryV2 | None = None,
    expected_pipeline_fingerprint: PipelineFingerprint | None = None,
    pipeline_root: Path | None = None,
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None = None,
    sequential_audit_state: SequentialVerificationState | None = None,
    generated_at: datetime | None = None,
) -> ConditionCalibrationCollectionSourceV1:
    """Run one pre-bundle, outcome-free condition-calibration policy arm.

    This path never accepts or emits confirmation outcomes, reference labels, a v2
    calibration bundle, a release qualification, or a release decision.  It uses the
    ordinary threshold-blind scheduler and always returns an abstained, type-distinct
    collection source.  Once the first non-confirmation-eligible prefix is reached,
    its exact trajectory may be joined to a held-out assessment only by
    :func:`freeze_condition_calibration_assessment_receipt_v1`.
    """

    if manifest.claim_manifest_version != "3":
        raise VerificationContractError(
            "condition_collection_requires_manifest_v3"
        )
    if not math.isfinite(budget_minutes) or budget_minutes < 0:
        raise VerificationContractError(
            "condition_collection_budget_minutes_invalid"
        )
    generated_at = generated_at or datetime.now(UTC)
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise VerificationContractError(
            "condition_collection_generated_at_requires_timezone"
        )
    pipeline_verification, pipeline_basis = _pipeline_identity(
        manifest,
        expected=expected_pipeline_fingerprint,
        root=pipeline_root,
    )
    pipeline_sha256 = pipeline_verification.computed_pipeline_sha256
    if pipeline_sha256 is None:
        raise VerificationContractError("computed_pipeline_identity_missing")
    try:
        policy_context = AdaptivePolicyContext.model_validate(
            adaptive_policy_context.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise VerificationContractError(
            "condition_collection_policy_context_invalid"
        ) from exc
    expected_policy_context = build_verifier_adaptive_policy_context(
        manifest=manifest,
        pipeline_sha256=pipeline_sha256,
        budget_minutes=float(budget_minutes),
        policy_arm_id=policy_context.policy_arm_id,
    )
    if policy_context != expected_policy_context:
        raise VerificationContractError(
            "condition_collection_policy_context_mismatch"
        )
    complete_identity = complete_corpus_identity_for_adaptive_calibration(
        manifest=manifest,
        corpus=corpus,
    )
    try:
        parsed_plan = ConditionConfirmationPlanV1.model_validate(
            condition_plan.model_dump(mode="json")
        )
        supplied_development = EvidenceGraph.model_validate(
            condition_development_graph.model_dump(mode="json")
        )
    except (AttributeError, ValueError) as exc:
        raise VerificationContractError(
            "condition_collection_runtime_input_tampered"
        ) from exc

    state: SequentialVerificationState | None = None
    if sequential_audit_state is not None:
        try:
            state = resume_sequential_verification_state(sequential_audit_state)
        except SequentialVerificationContractError as exc:
            raise VerificationContractError(
                f"condition_collection_sequential_state_invalid:{exc}"
            ) from exc
        if (
            state.session.pipeline_sha256 != pipeline_sha256
            or state.session.policy_sha256
            != compute_verification_policy_sha256(manifest)
            or not math.isclose(
                state.session.budget,
                float(budget_minutes),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or state.adaptive_policy_context_sha256 is not None
            or state.adaptive_calibration_bundle_sha256 is not None
        ):
            raise VerificationContractError(
                "condition_collection_sequential_state_context_mismatch"
            )
        if supplied_development != state.graph:
            raise VerificationContractError(
                "condition_collection_supplied_development_not_current_state"
            )
        source_development = partition_evidence_graph(
            corpus.graph,
            parsed_plan.development_partition,
        )
        if state.initial_graph != source_development:
            raise VerificationContractError(
                "condition_collection_initial_development_graph_mismatch"
            )
        current_full_graph = _condition_full_graph_from_development_state(
            source_full_graph=corpus.graph,
            development_graph=state.graph,
            plan=parsed_plan,
        )
    else:
        current_full_graph = corpus.graph
    context = prepare_condition_verification_context(
        manifest=manifest,
        corpus=corpus,
        complete_corpus_identity=complete_identity,
        plan=parsed_plan,
        development_graph=supplied_development,
        frozen_model=condition_frozen_model,
        pipeline_sha256=pipeline_sha256,
        current_full_graph=current_full_graph,
    )

    item_risk_bundle = None
    item_risk_candidates = None
    if item_risk_scoring_receipt is not None:
        try:
            item_risk_scoring_receipt = ItemRiskScoringRunReceipt.model_validate(
                item_risk_scoring_receipt.model_dump(mode="json")
            )
        except (AttributeError, ValueError) as exc:
            raise VerificationContractError(
                "condition_collection_item_risk_receipt_invalid"
            ) from exc
        if item_risk_scoring_receipt.pipeline_verification != pipeline_verification:
            raise VerificationContractError(
                "condition_collection_item_risk_pipeline_mismatch"
            )
        source_estimates = {
            row.estimate_id: row
            for row in (
                state.initial_graph if state is not None else context.development_graph
            ).outcome_estimates
        }
        stale = sorted(
            candidate.item_id
            for candidate in item_risk_scoring_receipt.candidates
            if candidate.item_id not in source_estimates
            or candidate.score_input_sha256
            != hash_canonical(source_estimates[candidate.item_id])
        )
        if stale:
            raise VerificationContractError(
                f"condition_collection_item_risk_source_snapshot_mismatch:{stale}"
            )
        item_risk_bundle = item_risk_scoring_receipt.calibration_bundle
        item_risk_candidates = list(item_risk_scoring_receipt.candidates)

    prepared = prepare_verification_scientific_state(
        manifest=manifest,
        graph=context.development_graph,
        pipeline_verification=pipeline_verification,
        item_risk_calibration_bundle=item_risk_bundle,
        item_risk_candidates=item_risk_candidates,
        resolved_item_ids_for_risk_projection=(
            set() if state is None else set(state.session.resolved_item_ids)
        ),
    )
    if state is None:
        state = _create_initial_sequential_audit_state(
            manifest=manifest,
            graph=context.development_graph,
            synthesis=prepared.synthesis,
            candidates=list(prepared.audit_candidates),
            claim_model=prepared.claim_model,
            counterfactuals=list(prepared.counterfactuals),
            item_risk_bounds=list(prepared.item_risk_bounds),
            pipeline_sha256=pipeline_sha256,
            budget_minutes=float(budget_minutes),
            created_at=generated_at,
        )
    else:
        _require_exact_prepared_sequential_artifacts(
            label="condition_collection_current",
            state_synthesis=state.synthesis,
            state_candidates=state.candidates,
            prepared=prepared,
            manifest=manifest,
        )

    corpus_issues = _condition_corpus_issues(
        manifest=manifest,
        corpus=corpus,
        pipeline_sha256=pipeline_sha256,
    )
    blocking_adapter_reasons = sorted(
        f"adapter:{issue.code}"
        for issue in corpus_issues
        if issue.severity is AdapterIssueSeverity.BLOCKING
    )

    def assess_condition_state(
        checkpoint_state: SequentialVerificationState,
        checkpoint_context: PreparedConditionVerificationContext,
        checkpoint_prepared: PreparedVerificationScientificState,
    ) -> ConditionClaimReleaseAssessmentV1:
        return assess_global_condition_claim_release_source(
            graph=checkpoint_context.development_graph,
            question_id=manifest.question_id,
            population_id=manifest.population_id,
            domain=manifest.domain,
            pipeline_sha256=pipeline_sha256,
            target=manifest.global_condition_target,
            condition_calibration_projection=checkpoint_context.projection,
            condition_confirmation_gate=_condition_missing_gate(checkpoint_context),
            audit_candidates=list(checkpoint_prepared.audit_candidates),
            claim_model=checkpoint_prepared.claim_model,
            audit_resolution_receipts=[],
            audit_budget=float(budget_minutes),
            condition_noncalibration_reasons=(
                checkpoint_context.ordinary_blocking_reasons
            ),
            external_noncalibration_blocking_reasons=blocking_adapter_reasons,
            config=manifest.release,
            audit_guard_config=manifest.audit_guard.to_runtime(),
            sequential_audit_state=checkpoint_state,
        )

    try:
        predecessor_states = selection_predecessor_states_from_state(state)
    except SequentialVerificationContractError as exc:
        raise VerificationContractError(
            f"condition_collection_history_invalid:{exc}"
        ) from exc
    online_states: list[AdaptivePreselectionState] = []
    for index, predecessor in enumerate(predecessor_states):
        historical_context, _ = _rebuild_condition_context_for_development_state(
            manifest=manifest,
            corpus=corpus,
            complete_corpus_identity=complete_identity,
            development_graph=predecessor.graph,
            template_plan=context.plan,
            pipeline_sha256=pipeline_sha256,
        )
        historical_prepared = prepare_verification_scientific_state(
            manifest=manifest,
            graph=predecessor.graph,
            pipeline_verification=pipeline_verification,
            item_risk_calibration_bundle=item_risk_bundle,
            item_risk_candidates=item_risk_candidates,
            resolved_item_ids_for_risk_projection=set(
                predecessor.session.resolved_item_ids
            ),
        )
        _require_exact_prepared_sequential_artifacts(
            label=f"condition_collection_history_{index}",
            state_synthesis=predecessor.synthesis,
            state_candidates=predecessor.candidates,
            prepared=historical_prepared,
            manifest=manifest,
        )
        try:
            online_states.append(
                freeze_preselection_state_from_production_components(
                    sequential_state=predecessor,
                    release_assessment=assess_condition_state(
                        predecessor,
                        historical_context,
                        historical_prepared,
                    ),
                    blocking_adapter_reasons=blocking_adapter_reasons,
                )
            )
        except AdaptiveCalibrationError as exc:
            raise VerificationContractError(
                f"condition_collection_history_projection_failed:{index}:{exc}"
            ) from exc

    current_preselection: AdaptivePreselectionState | None = None
    current_assessment = assess_condition_state(state, context, prepared)
    if state.session.active_action is None:
        try:
            current_preselection = freeze_preselection_state_from_production_components(
                sequential_state=state,
                release_assessment=current_assessment,
                blocking_adapter_reasons=blocking_adapter_reasons,
            )
        except AdaptiveCalibrationError as exc:
            raise VerificationContractError(
                f"condition_collection_current_projection_failed:{exc}"
            ) from exc
        online_states.append(current_preselection)
    elif not online_states:
        raise VerificationContractError(
            "condition_collection_active_action_missing_predecessor"
        )

    invocation: ConditionGateInvocationProofV2 | None = None
    selection_result = None
    evaluated_state = state
    hard_context_blockers = bool(
        blocking_adapter_reasons or context.ordinary_blocking_reasons
    )
    if state.session.active_action is not None:
        outcome: Literal[
            "selected_next_action",
            "active_action_in_progress",
            "condition_gate_ready",
            "condition_context_blocked",
            "no_feasible_action",
        ] = "active_action_in_progress"
    elif current_preselection is not None and current_preselection.non_calibration_gates_passed:
        invocation = freeze_condition_gate_invocation_proof_v2(
            terminal_preselection_state=current_preselection,
            condition_projection=context.projection,
            source_candidate_input_sha256=state.candidate_input_sha256,
            available_actions=_condition_terminal_actions(state),
            remaining_budget_minutes=state.session.remaining_budget,
        )
        outcome = "condition_gate_ready"
    elif hard_context_blockers:
        outcome = "condition_context_blocked"
    elif state.session.status.value == "active" and state.session.remaining_budget > 0:
        try:
            selection_result = select_next_audit_candidate(
                state,
                expected=freeze_state_expectation(state),
                selected_at=generated_at,
            )
        except SequentialVerificationContractError as exc:
            if str(exc) != "no_eligible_candidate_fits_remaining_budget":
                raise VerificationContractError(
                    f"condition_collection_audit_selection_failed:{exc}"
                ) from exc
            outcome = "no_feasible_action"
        else:
            outcome = "selected_next_action"
            state = selection_result.state
    else:
        outcome = "no_feasible_action"

    decision_reasons = {
        "active_action_in_progress": "condition_collection_action_in_progress",
        "condition_context_blocked": "condition_collection_context_blocked",
        "condition_gate_ready": "condition_collection_gate_ready_always_abstained",
        "no_feasible_action": "condition_collection_no_feasible_action",
        "selected_next_action": "condition_collection_action_selected",
    }
    collection_decision = freeze_condition_calibration_collection_decision_v1(
        evaluated_state=evaluated_state,
        terminal_preselection_state=current_preselection,
        outcome=outcome,
        selection_result=selection_result,
        condition_gate_invocation_proof=invocation,
        reasons=[decision_reasons[outcome]],
    )

    visible = policy_visible_question_trajectory
    if outcome in {
        "condition_gate_ready",
        "condition_context_blocked",
        "no_feasible_action",
    }:
        terminal_actions = _condition_terminal_actions(evaluated_state)
        if outcome == "condition_gate_ready":
            assert invocation is not None
            terminal_reason = invocation.terminal_reason
        elif outcome == "condition_context_blocked":
            terminal_reason = "nonconfirmation_context_blocked"
        else:
            unresolved = [
                action
                for action in terminal_actions
                if action.item_id
                not in set(evaluated_state.session.resolved_item_ids)
            ]
            if not unresolved:
                terminal_reason = "all_items_resolved"
            elif math.isclose(
                evaluated_state.session.remaining_budget,
                0.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                terminal_reason = "budget_exhausted"
            else:
                terminal_reason = "no_feasible_action"
        arm: AdaptivePolicyArmTrajectory = freeze_adaptive_policy_arm_trajectory(
            policy_arm_id=policy_context.policy_arm_id,
            policy_context_sha256=policy_context.policy_context_sha256,
            states=online_states,
            terminal_reason=terminal_reason,
            terminal_candidates=terminal_actions,
            terminal_source_candidate_input_sha256=(
                evaluated_state.candidate_input_sha256
            ),
            terminal_remaining_budget_minutes=(
                evaluated_state.session.remaining_budget
            ),
            terminal_nonconfirmation_blocking_reasons=(
                []
                if outcome != "condition_context_blocked"
                else current_preselection.non_calibration_blocking_reasons
                if current_preselection is not None
                else []
            ),
            terminal_condition_projection=(
                context.projection
                if outcome == "condition_gate_ready"
                else None
            ),
            terminal_condition_invocation_proof=invocation,
        )
        wrapped_arm = freeze_confirmation_aware_arm_trajectory(
            base_arm=arm,
            terminal_condition_projection=context.projection,
        )
        if visible is None:
            base_visible = freeze_policy_visible_question_trajectory(
                question_id=manifest.question_id,
                split=collection_split,
                population_id=manifest.population_id,
                domain=manifest.domain,
                corpus=complete_identity,
                arms=[arm],
            )
            visible = freeze_policy_visible_question_trajectory_v2(
                base_visible=base_visible,
                target_semantics=context.target_semantics,
                independence_identity=context.independence_identity,
                arms=[wrapped_arm],
            )
        else:
            try:
                visible = PolicyVisibleQuestionTrajectoryV2.model_validate(
                    visible.model_dump(mode="json")
                )
            except (AttributeError, ValueError) as exc:
                raise VerificationContractError(
                    "condition_collection_visible_trajectory_invalid"
                ) from exc
    elif visible is not None:
        raise VerificationContractError(
            "condition_collection_visible_trajectory_requires_complete_terminal_state"
        )

    candidate_payload = [asdict(candidate) for candidate in prepared.audit_candidates]
    lineage = [
        CertificateLineageStage(
            stage="condition_collection_outcome_firewall",
            input_sha256s={
                "full_graph": hash_canonical(current_full_graph),
                "materialization_receipt": (
                    context.plan.materialization_receipt_sha256
                ),
            },
            output_sha256s={
                "development_graph": context.plan.development_graph_sha256,
                "firewall_receipt": context.projection.firewall_receipt_sha256,
            },
            method="custodian-materialization-replay-development-only-online-v1",
        ),
        CertificateLineageStage(
            stage="condition_collection_scheduler",
            input_sha256s={
                "candidate_input": evaluated_state.candidate_input_sha256,
                "development_graph": context.plan.development_graph_sha256,
            },
            output_sha256s={
                "collection_decision": collection_decision.decision_sha256,
                "trajectory": (
                    hash_canonical(None)
                    if visible is None
                    else visible.trajectory_sha256
                ),
            },
            method="threshold-blind-prebundle-condition-collection-v1",
        ),
    ]
    corpus_payload = corpus.certificate_payload()
    corpus_payload["declared_corpus_cutoff"] = manifest.protocol.corpus_cutoff
    corpus_payload["pipeline_identity_basis"] = pipeline_basis
    source_reasons = sorted(
        set(current_assessment.reasons)
        | set(blocking_adapter_reasons)
        | {"condition_calibration_collection_source_never_release_eligible"}
    )
    return freeze_condition_calibration_collection_source_v1(
        generated_at=generated_at,
        collection_split=collection_split,
        reasons=source_reasons,
        claim_manifest=manifest.model_dump(mode="json"),
        corpus=corpus_payload,
        corpus_sha256=corpus.source_sha256,
        source_evidence_graph=corpus.graph,
        current_full_evidence_graph=current_full_graph,
        development_evidence_graph=context.development_graph,
        adapter_issues=[row.model_dump(mode="json") for row in corpus_issues],
        synthesis=prepared.synthesis,
        audit_candidates=candidate_payload,
        pipeline_verification=pipeline_verification,
        complete_corpus_identity=complete_identity,
        item_risk_scoring_receipt=item_risk_scoring_receipt,
        condition_plan=context.plan,
        condition_frozen_model=context.frozen_model,
        condition_calibration_projection=context.projection,
        condition_target_semantics=context.target_semantics,
        condition_independence_identity=context.independence_identity,
        adaptive_policy_context=policy_context,
        online_preselection_states=online_states,
        policy_visible_question_trajectory=visible,
        sequential_audit_state=state,
        collection_decision=collection_decision,
        lineage=lineage,
    )


def validate_condition_calibration_collection_source_external_replay(
    source: ConditionCalibrationCollectionSourceV1,
    *,
    pipeline_root: Path | None = None,
) -> ConditionCalibrationCollectionSourceV1:
    """Rerun one already-parsed collection source without trusting authored hashes.

    Structural model validation checks the self-hashed artifact without rerunning
    science.  Public source factories/loaders, source-roster validation, and receipt
    validation call this boundary explicitly once per authored input.  This function
    therefore operates on the already parsed object and never hides a nested replay.
    """

    computed = source.pipeline_verification.computed
    if computed is None:
        raise VerificationContractError(
            "condition_collection_external_pipeline_artifact_missing"
        )
    repository_root = pipeline_root or Path(__file__).resolve().parents[2]
    try:
        replayed_pipeline = require_pipeline_fingerprint_match(
            expected=computed,
            root=repository_root,
        )
    except ValueError as exc:
        raise VerificationContractError(
            f"condition_collection_external_pipeline_replay_failed:{exc}"
        ) from exc
    if replayed_pipeline != source.pipeline_verification:
        raise VerificationContractError(
            "condition_collection_external_pipeline_verification_mismatch"
        )
    pipeline_sha256 = replayed_pipeline.computed_pipeline_sha256
    assert pipeline_sha256 is not None
    try:
        manifest = ClaimManifest.model_validate(source.claim_manifest)
    except ValueError as exc:
        raise VerificationContractError(
            "condition_collection_external_manifest_invalid"
        ) from exc
    if manifest.claim_manifest_version != "3":
        raise VerificationContractError(
            "condition_collection_external_manifest_version_mismatch"
        )

    metadata = source.corpus.get("metadata")
    native_manifest = (
        metadata.get("native_source_manifest")
        if isinstance(metadata, dict)
        else None
    )
    source_manifest_sha256 = (
        metadata.get("source_manifest_sha256")
        if isinstance(metadata, dict)
        else None
    )
    provenance = source.corpus.get("provenance_assurance")
    eligibility_payload = source.corpus.get("eligibility")
    terminal_membership = (
        metadata.get("terminal_fragment_membership")
        if isinstance(metadata, dict)
        else None
    )
    if (
        source.corpus.get("corpus_id") != manifest.question_id
        or source.corpus.get("source_sha256") != source.corpus_sha256
        or source.corpus.get("source_format")
        != "typed_evidence_grounding_package_json"
        or not isinstance(provenance, dict)
        or provenance.get("status") != "source_replayed_native_grounding"
        or provenance.get("release_eligible") is not True
        or not isinstance(metadata, dict)
        or metadata.get("grounding_package_version")
        != "typed-evidence-grounding-package-v4"
        or metadata.get("pipeline_fingerprint_sha256") != pipeline_sha256
        or metadata.get("native_corpus_cutoff")
        != manifest.protocol.corpus_cutoff
        or metadata.get("grounding_replay_sha256") != provenance.get("replay_sha256")
        or metadata.get("source_manifest_membership_bound") is not True
        or metadata.get("question_config_sha256")
        != source.condition_plan.target.question_config_sha256
        or metadata.get("extraction_context_sha256") in {None, ""}
        or metadata.get("extraction_context_receipt_sha256")
        != metadata.get("replayed_extraction_context_receipt_sha256")
        or not isinstance(eligibility_payload, list)
        or not isinstance(terminal_membership, list)
    ):
        raise VerificationContractError(
            "condition_collection_external_native_corpus_contract_mismatch"
        )
    if isinstance(native_manifest, dict):
        try:
            parsed_native_manifest = NativeSourceManifest.model_validate(native_manifest)
        except ValueError as exc:
            raise VerificationContractError(
                "condition_collection_external_source_manifest_invalid"
            ) from exc
        records = native_manifest.get("records")
        if (
            not isinstance(records, list)
            or not isinstance(source_manifest_sha256, str)
            or hash_canonical(native_manifest) != source_manifest_sha256
            or parsed_native_manifest.question_id != manifest.question_id
        ):
            raise VerificationContractError(
                "condition_collection_external_source_manifest_invalid"
            )
        try:
            publication_ids = sorted(
                str(record["publication"]["publication_id"])
                for record in records
            )
        except (KeyError, TypeError) as exc:
            raise VerificationContractError(
                "condition_collection_external_source_membership_invalid"
            ) from exc
        manifest_publications = sorted(
            (record.publication for record in parsed_native_manifest.records),
            key=lambda row: row.publication_id,
        )
        graph_publications = sorted(
            source.source_evidence_graph.publications,
            key=lambda row: row.publication_id,
        )
        if manifest_publications != graph_publications:
            raise VerificationContractError(
                "condition_collection_external_manifest_graph_publication_mismatch"
            )
    else:
        raise VerificationContractError(
            "condition_collection_external_native_source_manifest_missing"
        )
    try:
        eligibility = [
            CorpusEligibilityRecord.model_validate(row)
            for row in eligibility_payload
        ]
    except ValueError as exc:
        raise VerificationContractError(
            "condition_collection_external_eligibility_invalid"
        ) from exc
    eligibility_by_paper = {row.paper_id: row for row in eligibility}
    if (
        len(eligibility_by_paper) != len(eligibility)
        or sorted(eligibility_by_paper)
        != sorted(record.publication.paper_id for record in parsed_native_manifest.records)
        or any(
            eligibility_by_paper[record.publication.paper_id].status != "included"
            or eligibility_by_paper[record.publication.paper_id].source
            != record.source_document.source_locator
            for record in parsed_native_manifest.records
        )
    ):
        raise VerificationContractError(
            "condition_collection_external_eligibility_membership_mismatch"
        )
    terminal_expected = sorted(
        (
            record.publication.paper_id,
            record.publication.publication_id,
        )
        for record in parsed_native_manifest.records
    )
    if any(
        not isinstance(row, dict)
        or set(row) != {"fragment_sha256", "paper_id", "publication_id", "status"}
        or not isinstance(row.get("fragment_sha256"), str)
        or SHA256_RE.fullmatch(row["fragment_sha256"]) is None
        or not isinstance(row.get("paper_id"), str)
        or not row["paper_id"]
        or not isinstance(row.get("publication_id"), str)
        or not row["publication_id"]
        or row.get("status") not in {"estimable", "non_estimable"}
        for row in terminal_membership
    ):
        raise VerificationContractError(
            "condition_collection_external_terminal_membership_invalid"
        )
    try:
        terminal_observed = sorted(
            (str(row["paper_id"]), str(row["publication_id"]))
            for row in terminal_membership
        )
    except (KeyError, TypeError) as exc:
        raise VerificationContractError(
            "condition_collection_external_terminal_membership_invalid"
        ) from exc
    if (
        terminal_observed != terminal_expected
        or len(terminal_membership) != len(terminal_expected)
        or metadata.get("terminal_fragment_records") != len(terminal_membership)
        or hash_canonical(terminal_membership)
        != metadata.get("terminal_fragment_membership_sha256")
        or metadata.get("source_manifest_records")
        != len(parsed_native_manifest.records)
    ):
        raise VerificationContractError(
            "condition_collection_external_terminal_membership_mismatch"
        )
    expected_graph_counts = {
        "cohorts": len(source.source_evidence_graph.cohorts),
        "estimates": len(source.source_evidence_graph.outcome_estimates),
        "publications": len(source.source_evidence_graph.publications),
        "studies": len(source.source_evidence_graph.studies),
    }
    expected_eligibility_counts = {
        "excluded": sum(row.status == "excluded" for row in eligibility),
        "included": sum(row.status == "included" for row in eligibility),
        "pending": sum(row.status == "pending" for row in eligibility),
    }
    if (
        source.corpus.get("graph_counts") != expected_graph_counts
        or source.corpus.get("eligibility_counts") != expected_eligibility_counts
    ):
        raise VerificationContractError(
            "condition_collection_external_corpus_count_mismatch"
        )
    expected_identity = freeze_complete_corpus_identity(
        corpus_id=str(source.corpus.get("corpus_id") or ""),
        corpus_source_sha256=source.corpus_sha256,
        corpus_cutoff=manifest.protocol.corpus_cutoff,
        publication_ids=publication_ids,
        source_manifest_sha256=source_manifest_sha256,
    )
    if expected_identity != source.complete_corpus_identity:
        raise VerificationContractError(
            "condition_collection_external_complete_corpus_mismatch"
        )
    expected_policy = build_verifier_adaptive_policy_context(
        manifest=manifest,
        pipeline_sha256=pipeline_sha256,
        budget_minutes=source.adaptive_policy_context.budget_minutes,
        policy_arm_id=source.adaptive_policy_context.policy_arm_id,
    )
    if expected_policy != source.adaptive_policy_context:
        raise VerificationContractError(
            "condition_collection_external_policy_context_mismatch"
        )
    context, current_full = _rebuild_condition_context_from_graphs(
        manifest=manifest,
        complete_corpus_identity=expected_identity,
        source_full_graph=source.source_evidence_graph,
        development_graph=source.development_evidence_graph,
        template_plan=source.condition_plan,
        pipeline_sha256=pipeline_sha256,
        exact_current_full_graph=source.current_full_evidence_graph,
    )
    if (
        current_full != source.current_full_evidence_graph
        or context.plan != source.condition_plan
        or context.frozen_model != source.condition_frozen_model
        or context.projection != source.condition_calibration_projection
        or context.target_semantics != source.condition_target_semantics
        or context.independence_identity != source.condition_independence_identity
    ):
        raise VerificationContractError(
            "condition_collection_external_scientific_context_mismatch"
        )

    item_risk_bundle = None
    item_risk_candidates = None
    if source.item_risk_scoring_receipt is not None:
        receipt = source.item_risk_scoring_receipt
        if receipt.pipeline_verification != replayed_pipeline:
            raise VerificationContractError(
                "condition_collection_external_item_risk_pipeline_mismatch"
            )
        source_estimates = {
            row.estimate_id: row
            for row in partition_evidence_graph(
                source.source_evidence_graph,
                source.condition_plan.development_partition,
            ).outcome_estimates
        }
        stale = sorted(
            candidate.item_id
            for candidate in receipt.candidates
            if candidate.item_id not in source_estimates
            or candidate.score_input_sha256
            != hash_canonical(source_estimates[candidate.item_id])
        )
        if stale:
            raise VerificationContractError(
                "condition_collection_external_item_risk_snapshot_mismatch:"
                + ",".join(stale)
            )
        item_risk_bundle = receipt.calibration_bundle
        item_risk_candidates = list(receipt.candidates)

    state = resume_sequential_verification_state(source.sequential_audit_state)
    if (
        state.session.pipeline_sha256 != pipeline_sha256
        or state.session.policy_sha256
        != compute_verification_policy_sha256(manifest)
        or state.adaptive_policy_context_sha256 is not None
        or state.adaptive_calibration_bundle_sha256 is not None
        or state.graph != source.development_evidence_graph
        or state.initial_graph
        != partition_evidence_graph(
            source.source_evidence_graph,
            source.condition_plan.development_partition,
        )
    ):
        raise VerificationContractError(
            "condition_collection_external_sequential_context_mismatch"
        )
    prepared = prepare_verification_scientific_state(
        manifest=manifest,
        graph=source.development_evidence_graph,
        pipeline_verification=replayed_pipeline,
        item_risk_calibration_bundle=item_risk_bundle,
        item_risk_candidates=item_risk_candidates,
        resolved_item_ids_for_risk_projection=set(state.session.resolved_item_ids),
    )
    _require_exact_prepared_sequential_artifacts(
        label="condition_collection_external_current",
        state_synthesis=state.synthesis,
        state_candidates=state.candidates,
        prepared=prepared,
        manifest=manifest,
    )
    if (
        prepared.synthesis != source.synthesis
        or [asdict(row) for row in prepared.audit_candidates]
        != source.audit_candidates
    ):
        raise VerificationContractError(
            "condition_collection_external_science_payload_mismatch"
        )

    blocking_adapter_reasons = sorted(
        f"adapter:{issue.get('code')!s}"
        for issue in source.adapter_issues
        if issue.get("severity") == AdapterIssueSeverity.BLOCKING.value
    )

    def assess_checkpoint(
        checkpoint_state: SequentialVerificationState,
        checkpoint_context: PreparedConditionVerificationContext,
        checkpoint_prepared: PreparedVerificationScientificState,
    ) -> ConditionClaimReleaseAssessmentV1:
        return assess_global_condition_claim_release_source(
            graph=checkpoint_context.development_graph,
            question_id=manifest.question_id,
            population_id=manifest.population_id,
            domain=manifest.domain,
            pipeline_sha256=pipeline_sha256,
            target=manifest.global_condition_target,
            condition_calibration_projection=checkpoint_context.projection,
            condition_confirmation_gate=_condition_missing_gate(checkpoint_context),
            audit_candidates=list(checkpoint_prepared.audit_candidates),
            claim_model=checkpoint_prepared.claim_model,
            audit_resolution_receipts=[],
            audit_budget=state.session.budget,
            condition_noncalibration_reasons=(
                checkpoint_context.ordinary_blocking_reasons
            ),
            external_noncalibration_blocking_reasons=blocking_adapter_reasons,
            config=manifest.release,
            audit_guard_config=manifest.audit_guard.to_runtime(),
            sequential_audit_state=checkpoint_state,
        )

    predecessor_states = selection_predecessor_states_from_state(state)
    online_states: list[AdaptivePreselectionState] = []
    for index, predecessor in enumerate(predecessor_states):
        historical_context, _ = _rebuild_condition_context_from_graphs(
            manifest=manifest,
            complete_corpus_identity=expected_identity,
            source_full_graph=source.source_evidence_graph,
            development_graph=predecessor.graph,
            template_plan=source.condition_plan,
            pipeline_sha256=pipeline_sha256,
            exact_current_full_graph=(
                source.source_evidence_graph
                if predecessor.graph == state.initial_graph
                else None
            ),
        )
        historical_prepared = prepare_verification_scientific_state(
            manifest=manifest,
            graph=predecessor.graph,
            pipeline_verification=replayed_pipeline,
            item_risk_calibration_bundle=item_risk_bundle,
            item_risk_candidates=item_risk_candidates,
            resolved_item_ids_for_risk_projection=set(
                predecessor.session.resolved_item_ids
            ),
        )
        _require_exact_prepared_sequential_artifacts(
            label=f"condition_collection_external_history_{index}",
            state_synthesis=predecessor.synthesis,
            state_candidates=predecessor.candidates,
            prepared=historical_prepared,
            manifest=manifest,
        )
        online_states.append(
            freeze_preselection_state_from_production_components(
                sequential_state=predecessor,
                release_assessment=assess_checkpoint(
                    predecessor,
                    historical_context,
                    historical_prepared,
                ),
                blocking_adapter_reasons=blocking_adapter_reasons,
            )
        )
    decision = source.collection_decision
    evaluated = decision.evaluated_state
    current_preselection: AdaptivePreselectionState | None = None
    current_assessment = assess_checkpoint(evaluated, context, prepared)
    if evaluated.session.active_action is None:
        current_preselection = freeze_preselection_state_from_production_components(
            sequential_state=evaluated,
            release_assessment=current_assessment,
            blocking_adapter_reasons=blocking_adapter_reasons,
        )
        if not online_states or online_states[-1] != current_preselection:
            online_states.append(current_preselection)
    if online_states != source.online_preselection_states:
        raise VerificationContractError(
            "condition_collection_external_online_trajectory_mismatch"
        )

    invocation: ConditionGateInvocationProofV2 | None = None
    selection_result = None
    hard_context_blockers = bool(
        blocking_adapter_reasons or context.ordinary_blocking_reasons
    )
    if evaluated.session.active_action is not None:
        expected_outcome = "active_action_in_progress"
    elif current_preselection is not None and current_preselection.non_calibration_gates_passed:
        invocation = freeze_condition_gate_invocation_proof_v2(
            terminal_preselection_state=current_preselection,
            condition_projection=context.projection,
            source_candidate_input_sha256=evaluated.candidate_input_sha256,
            available_actions=_condition_terminal_actions(evaluated),
            remaining_budget_minutes=evaluated.session.remaining_budget,
        )
        expected_outcome = "condition_gate_ready"
    elif hard_context_blockers:
        expected_outcome = "condition_context_blocked"
    elif evaluated.session.status.value == "active" and evaluated.session.remaining_budget > 0:
        try:
            selection_result = select_next_audit_candidate(
                evaluated,
                expected=freeze_state_expectation(evaluated),
                selected_at=source.generated_at,
            )
        except SequentialVerificationContractError as exc:
            if str(exc) != "no_eligible_candidate_fits_remaining_budget":
                raise VerificationContractError(
                    f"condition_collection_external_selection_failed:{exc}"
                ) from exc
            expected_outcome = "no_feasible_action"
        else:
            expected_outcome = "selected_next_action"
    else:
        expected_outcome = "no_feasible_action"
    reason_by_outcome = {
        "active_action_in_progress": "condition_collection_action_in_progress",
        "condition_context_blocked": "condition_collection_context_blocked",
        "condition_gate_ready": "condition_collection_gate_ready_always_abstained",
        "no_feasible_action": "condition_collection_no_feasible_action",
        "selected_next_action": "condition_collection_action_selected",
    }
    expected_decision = freeze_condition_calibration_collection_decision_v1(
        evaluated_state=evaluated,
        terminal_preselection_state=current_preselection,
        outcome=expected_outcome,
        selection_result=selection_result,
        condition_gate_invocation_proof=invocation,
        reasons=[reason_by_outcome[expected_outcome]],
    )
    if expected_decision != decision:
        raise VerificationContractError(
            "condition_collection_external_decision_mismatch"
        )
    expected_final_state = (
        selection_result.state
        if expected_outcome == "selected_next_action" and selection_result is not None
        else evaluated
    )
    if expected_final_state != state:
        raise VerificationContractError(
            "condition_collection_external_final_state_mismatch"
        )
    expected_reasons = sorted(
        set(current_assessment.reasons)
        | set(blocking_adapter_reasons)
        | {"condition_calibration_collection_source_never_release_eligible"}
    )
    if source.reasons != expected_reasons:
        raise VerificationContractError(
            "condition_collection_external_reason_ledger_mismatch"
        )
    expected_lineage = [
        CertificateLineageStage(
            stage="condition_collection_outcome_firewall",
            input_sha256s={
                "full_graph": hash_canonical(source.current_full_evidence_graph),
                "materialization_receipt": (
                    context.plan.materialization_receipt_sha256
                ),
            },
            output_sha256s={
                "development_graph": context.plan.development_graph_sha256,
                "firewall_receipt": context.projection.firewall_receipt_sha256,
            },
            method="custodian-materialization-replay-development-only-online-v1",
        ),
        CertificateLineageStage(
            stage="condition_collection_scheduler",
            input_sha256s={
                "candidate_input": evaluated.candidate_input_sha256,
                "development_graph": context.plan.development_graph_sha256,
            },
            output_sha256s={
                "collection_decision": decision.decision_sha256,
                "trajectory": (
                    hash_canonical(None)
                    if source.policy_visible_question_trajectory is None
                    else source.policy_visible_question_trajectory.trajectory_sha256
                ),
            },
            method="threshold-blind-prebundle-condition-collection-v1",
        ),
    ]
    if source.lineage != expected_lineage:
        raise VerificationContractError(
            "condition_collection_external_lineage_mismatch"
        )
    expected_run_identity = hash_canonical(
        {
            "claim_manifest_sha256": source.claim_manifest_sha256,
            "collection_split": source.collection_split,
            "condition_projection_sha256": context.projection.projection_sha256,
            "decision_sha256": decision.decision_sha256,
            "policy_context_sha256": expected_policy.policy_context_sha256,
            "visible_trajectory_sha256": (
                None
                if source.policy_visible_question_trajectory is None
                else source.policy_visible_question_trajectory.trajectory_sha256
            ),
        }
    )
    if source.run_id != f"condition-collection-{expected_run_identity[:16]}":
        raise VerificationContractError(
            "condition_collection_external_run_identity_mismatch"
        )
    return source


def validate_condition_calibration_assessment_receipt_external_replay(
    receipt: Any,
    *,
    collection_source_roster: ConditionCalibrationCollectionSourceRosterV1
    | dict[str, Any]
    | None = None,
    pipeline_root: Path | None = None,
) -> Any:
    """Canonical public replay boundary consumed by adaptive v2 calibration."""

    from literature_multiverse.certificate import (
        ConditionCalibrationAssessmentReceiptV1,
        match_validated_condition_calibration_collection_source_membership_v1,
    )

    try:
        canonical = ConditionCalibrationAssessmentReceiptV1.model_validate(
            receipt.model_dump(mode="json")
            if hasattr(receipt, "model_dump")
            else receipt
        )
    except (AttributeError, ValueError) as exc:
        raise VerificationContractError(
            "condition_collection_receipt_external_contract_invalid"
        ) from exc
    default_root = Path(__file__).resolve().parents[2]
    if pipeline_root is not None and pipeline_root.resolve() != default_root.resolve():
        validate_condition_calibration_collection_source_external_replay(
            canonical.collection_source,
            pipeline_root=pipeline_root,
        )
    if collection_source_roster is not None:
        try:
            roster = ConditionCalibrationCollectionSourceRosterV1.model_validate(
                collection_source_roster.model_dump(mode="json")
                if hasattr(collection_source_roster, "model_dump")
                else collection_source_roster
            )
            anchor = match_validated_condition_calibration_collection_source_membership_v1(
                collection_source_roster=roster,
                collection_source=canonical.collection_source,
                expected_source_roster_sha256=canonical.source_roster_sha256,
                expected_source_membership_sha256=(
                    canonical.source_membership_sha256
                ),
            )
        except (AttributeError, ValueError) as exc:
            raise VerificationContractError(
                "condition_collection_receipt_external_roster_mismatch"
            ) from exc
        if anchor != canonical.source_anchor:
            raise VerificationContractError(
                "condition_collection_receipt_external_anchor_mismatch"
            )
    return canonical


def finalize_condition_verification(
    *,
    source_certificate: ConditionVerificationCertificateV6,
    condition_confirmation_assessment: ConditionConfirmationAssessmentV1,
    generated_at: datetime | None = None,
) -> FinalConditionVerificationCertificateV7:
    """Join a gate-ready, outcome-free v6 to one exact held-out assessment.

    This boundary deliberately has no corpus, manifest, state, or calibration input:
    all online artifacts are taken from and independently replayed out of the immutable
    source certificate.  The held-out assessment can therefore affect only the typed
    terminal gate and final calibrated decision, never retrieval, synthesis, scoring,
    audit selection, or its own threshold.
    """

    generated_at = generated_at or datetime.now(UTC)
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise VerificationContractError(
            "condition_finalizer_generated_at_requires_timezone"
        )
    try:
        return freeze_final_condition_verification_certificate_v7(
            generated_at=generated_at,
            source_certificate=source_certificate,
            condition_confirmation_assessment=(
                condition_confirmation_assessment
            ),
        )
    except (AttributeError, ValueError) as exc:
        raise VerificationContractError(
            f"condition_finalizer_rejected:{exc}"
        ) from exc


def run_verification(
    *,
    manifest: ClaimManifest,
    corpus: CorpusLoadResult,
    budget_minutes: float,
    frozen_calibration_bundle: FrozenCalibrationBundle | None = None,
    adaptive_calibration_bundle: AdaptiveCalibrationBundle | None = None,
    adaptive_calibration_bundle_v2: AdaptiveCalibrationBundleV2 | None = None,
    condition_plan: ConditionConfirmationPlanV1 | None = None,
    condition_development_graph: EvidenceGraph | None = None,
    condition_frozen_model: ConditionConfirmationFrozenModelV1 | None = None,
    condition_confirmation_assessment: ConditionConfirmationAssessmentV1 | None = None,
    audit_resolution_receipts: list[AuditResolutionReceipt] | None = None,
    expected_pipeline_fingerprint: PipelineFingerprint | None = None,
    pipeline_root: Path | None = None,
    item_risk_scoring_receipt: ItemRiskScoringRunReceipt | None = None,
    item_risk_calibration_bundle: ItemRiskCalibrationBundle | None = None,
    item_risk_candidates: list[ItemRiskCandidate] | None = None,
    sequential_audit_state: SequentialVerificationState | None = None,
    allow_uncalibrated_sequential_analysis: bool = False,
    generated_at: datetime | None = None,
) -> (
    VerificationCertificate
    | ConditionVerificationCertificateV6
):
    """Execute the complete frozen-corpus verifier and freeze its certificate."""

    if not math.isfinite(budget_minutes) or budget_minutes < 0:
        raise VerificationContractError("verification_budget_minutes_invalid")
    if type(allow_uncalibrated_sequential_analysis) is not bool:
        raise VerificationContractError(
            "allow_uncalibrated_sequential_analysis_must_be_boolean"
        )
    if allow_uncalibrated_sequential_analysis and (
        adaptive_calibration_bundle is not None or audit_resolution_receipts
    ):
        raise VerificationContractError(
            "uncalibrated_sequential_analysis_conflicts_with_release_inputs"
        )
    budget_minutes = float(budget_minutes)
    generated_at = generated_at or datetime.now(UTC)
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise VerificationContractError("verification_generated_at_requires_timezone")
    condition_inputs = (
        adaptive_calibration_bundle_v2,
        condition_plan,
        condition_development_graph,
        condition_frozen_model,
        condition_confirmation_assessment,
    )
    if manifest.claim_manifest_version == "3":
        if condition_confirmation_assessment is not None:
            raise VerificationContractError(
                "condition_terminal_assessment_requires_dedicated_finalizer"
            )
        if (
            adaptive_calibration_bundle_v2 is None
            or condition_plan is None
            or condition_development_graph is None
            or condition_frozen_model is None
        ):
            raise VerificationContractError(
                "manifest_v3_requires_condition_plan_development_model_and_v2_calibration"
            )
        if (
            frozen_calibration_bundle is not None
            or adaptive_calibration_bundle is not None
            or audit_resolution_receipts
            or allow_uncalibrated_sequential_analysis
            or item_risk_calibration_bundle is not None
            or item_risk_candidates is not None
        ):
            raise VerificationContractError(
                "manifest_v3_forbids_legacy_or_detached_release_inputs"
            )
        return _run_condition_verification(
            manifest=manifest,
            corpus=corpus,
            budget_minutes=budget_minutes,
            adaptive_calibration_bundle_v2=adaptive_calibration_bundle_v2,
            condition_plan=condition_plan,
            condition_development_graph=condition_development_graph,
            condition_frozen_model=condition_frozen_model,
            expected_pipeline_fingerprint=expected_pipeline_fingerprint,
            pipeline_root=pipeline_root,
            item_risk_scoring_receipt=item_risk_scoring_receipt,
            sequential_audit_state=sequential_audit_state,
            generated_at=generated_at,
        )
    if any(value is not None for value in condition_inputs):
        raise VerificationContractError(
            "condition_verification_inputs_require_manifest_v3"
        )
    receipts = sorted(
        (
            AuditResolutionReceipt.model_validate(receipt)
            for receipt in (audit_resolution_receipts or [])
        ),
        key=lambda receipt: receipt.item_id,
    )
    pipeline_verification, pipeline_basis = _pipeline_identity(
        manifest,
        expected=expected_pipeline_fingerprint,
        root=pipeline_root,
    )
    pipeline_sha256 = pipeline_verification.computed_pipeline_sha256
    if pipeline_sha256 is None:
        raise VerificationContractError("computed_pipeline_identity_missing")
    if item_risk_scoring_receipt is not None:
        if (
            item_risk_calibration_bundle is not None
            or item_risk_candidates is not None
        ):
            raise VerificationContractError(
                "item_risk_scoring_receipt_conflicts_with_detached_inputs"
            )
        try:
            item_risk_scoring_receipt = ItemRiskScoringRunReceipt.model_validate(
                item_risk_scoring_receipt.model_dump(mode="json")
            )
        except (AttributeError, ValueError) as exc:
            raise VerificationContractError(
                "item_risk_scoring_receipt_invalid"
            ) from exc
        if item_risk_scoring_receipt.pipeline_verification != pipeline_verification:
            raise VerificationContractError(
                "item_risk_scoring_receipt_pipeline_mismatch"
            )
        item_risk_calibration_bundle = (
            item_risk_scoring_receipt.calibration_bundle
        )
        item_risk_candidates = list(item_risk_scoring_receipt.candidates)
    elif (
        item_risk_calibration_bundle is not None or item_risk_candidates is not None
    ):
        raise VerificationContractError(
            "detached_item_risk_inputs_forbidden_use_scoring_receipt_v2"
        )
    complete_corpus_identity = complete_corpus_identity_for_adaptive_calibration(
        manifest=manifest,
        corpus=corpus,
    )
    if item_risk_scoring_receipt is not None:
        source_estimates = {
            estimate.estimate_id: estimate
            for estimate in corpus.graph.outcome_estimates
        }
        stale_item_ids = sorted(
            candidate.item_id
            for candidate in item_risk_scoring_receipt.candidates
            if candidate.item_id not in source_estimates
            or candidate.score_input_sha256
            != hash_canonical(source_estimates[candidate.item_id])
        )
        if stale_item_ids:
            raise VerificationContractError(
                "item_risk_scoring_receipt_source_snapshot_mismatch:"
                f"{stale_item_ids}"
            )
    adaptive_policy_context_for_certificate: AdaptivePolicyContext | None = None
    if frozen_calibration_bundle is not None and adaptive_calibration_bundle is not None:
        raise VerificationContractError("multiple_claim_calibration_bundles_supplied")
    if adaptive_calibration_bundle is not None:
        try:
            adaptive_calibration_bundle = (
                validate_adaptive_calibration_bundle_integrity(
                    adaptive_calibration_bundle
                )
            )
        except AdaptiveCalibrationError as exc:
            raise VerificationContractError(
                f"adaptive_calibration_bundle_invalid:{exc}"
            ) from exc
        adaptive_policy_context_for_certificate = (
            _derive_verifier_adaptive_policy_context(
                manifest=manifest,
                pipeline_sha256=pipeline_sha256,
                budget_minutes=budget_minutes,
                bundle=adaptive_calibration_bundle,
            )
        )
    if (
        sequential_audit_state is not None
        and sequential_audit_state.session.policy_sha256
        != compute_verification_policy_sha256(manifest)
    ):
        raise VerificationContractError(
            "sequential_audit_state_claim_manifest_context_mismatch"
        )
    corpus_issues = list(corpus.adapter_issues)
    if corpus.provenance_assurance.status == "source_replayed_native_grounding" and not (
        corpus.metadata.get("grounding_package_version")
        == "typed-evidence-grounding-package-v4"
        and corpus.metadata.get("source_manifest_membership_bound") is True
        and isinstance(corpus.metadata.get("source_manifest_sha256"), str)
        and SHA256_RE.fullmatch(corpus.metadata["source_manifest_sha256"]) is not None
        and isinstance(corpus.metadata.get("native_source_manifest"), dict)
        and hash_canonical(corpus.metadata["native_source_manifest"])
        == corpus.metadata["source_manifest_sha256"]
        and corpus.metadata.get("source_manifest_records") == len(corpus.eligibility)
        and isinstance(corpus.metadata.get("terminal_fragment_membership"), list)
        and corpus.metadata.get("terminal_fragment_records") == len(corpus.eligibility)
        and hash_canonical(corpus.metadata["terminal_fragment_membership"])
        == corpus.metadata.get("terminal_fragment_membership_sha256")
        and isinstance(corpus.metadata.get("native_corpus_cutoff"), str)
        and bool(corpus.metadata["native_corpus_cutoff"].strip())
        and corpus.extraction_context is not None
        and corpus.metadata.get("extraction_context_sha256")
        == corpus.extraction_context.context_sha256
        and corpus.metadata.get("extraction_context_receipt_sha256")
        == corpus.metadata.get("replayed_extraction_context_receipt_sha256")
    ):
        corpus_issues = [
            issue
            for issue in corpus_issues
            if issue.code != "native_source_manifest_membership_unbound"
        ]
        corpus_issues.append(
            CorpusAdapterIssue(
                severity=AdapterIssueSeverity.BLOCKING,
                code="native_source_manifest_membership_unbound",
                detail=(
                    "The source-replayed corpus does not bind the complete v4 native source, "
                    "fragment, cutoff, and exact extraction-execution context contract."
                ),
            )
        )
    if corpus.embedded_fixture_identity_valid():
        corpus_issues = [
            issue
            for issue in corpus_issues
            if issue.code
            not in {
                "embedded_synthetic_fixture_not_empirical",
                "unverified_source_provenance",
            }
        ]
        corpus_issues.append(
            CorpusAdapterIssue(
                severity=AdapterIssueSeverity.BLOCKING,
                code="embedded_synthetic_fixture_not_empirical",
                detail=(
                    "The embedded deterministic fixture exercises verifier mechanics "
                    "but is synthetic and cannot release an empirical scientific claim."
                ),
            )
        )
    elif not corpus.provenance_release_eligible():
        corpus_issues = [
            issue for issue in corpus_issues if issue.code != "unverified_source_provenance"
        ]
        detail = (
            corpus.provenance_assurance.reason
            if not corpus.provenance_assurance.release_eligible
            else (
                "The declared corpus-provenance assurance does not agree with the "
                "loader-controlled source format or replay identity."
            )
        )
        corpus_issues.append(
            CorpusAdapterIssue(
                severity=AdapterIssueSeverity.BLOCKING,
                code="unverified_source_provenance",
                detail=detail,
            )
        )
    if (
        corpus.provenance_assurance.status == "source_replayed_native_grounding"
        and corpus.metadata.get("pipeline_fingerprint_sha256") != pipeline_sha256
    ):
        corpus_issues = [
            issue for issue in corpus_issues if issue.code != "corpus_pipeline_identity_mismatch"
        ]
        corpus_issues.append(
            CorpusAdapterIssue(
                severity=AdapterIssueSeverity.BLOCKING,
                code="corpus_pipeline_identity_mismatch",
                detail=(
                    "The source-replayed native extraction corpus was produced by a "
                    "different computed pipeline than the verifier and calibration context."
                ),
            )
        )
    if (
        corpus.provenance_assurance.status == "source_replayed_native_grounding"
        and corpus.corpus_id != manifest.question_id
    ):
        corpus_issues = [
            issue for issue in corpus_issues if issue.code != "corpus_question_identity_mismatch"
        ]
        corpus_issues.append(
            CorpusAdapterIssue(
                severity=AdapterIssueSeverity.BLOCKING,
                code="corpus_question_identity_mismatch",
                detail=(
                    "The source-replayed native corpus question_id does not match the "
                    "claim manifest question_id."
                ),
            )
        )
    if (
        corpus.provenance_assurance.status == "source_replayed_native_grounding"
        and corpus.metadata.get("source_manifest_membership_bound") is True
        and corpus.metadata.get("native_corpus_cutoff")
        != manifest.protocol.corpus_cutoff
    ):
        corpus_issues = [
            issue for issue in corpus_issues if issue.code != "corpus_cutoff_identity_mismatch"
        ]
        corpus_issues.append(
            CorpusAdapterIssue(
                severity=AdapterIssueSeverity.BLOCKING,
                code="corpus_cutoff_identity_mismatch",
                detail=(
                    "The membership-bound native corpus cutoff does not match the claim "
                    "manifest protocol cutoff."
                ),
            )
        )
    if corpus.provenance_assurance.status == "source_replayed_native_grounding":
        corpus_issues.extend(
            _native_claim_config_compatibility_issues(
                manifest=manifest,
                corpus=corpus,
            )
        )
    if receipts:
        corpus_issues.append(
            CorpusAdapterIssue(
                severity=AdapterIssueSeverity.BLOCKING,
                code="legacy_static_audit_receipts_analysis_only",
                detail=(
                    "Detached static audit receipts are retained for legacy analysis "
                    "only; they are not a calibrated sequential release trajectory."
                ),
            )
        )
    elif adaptive_calibration_bundle is None:
        existing_nonadaptive_state = (
            sequential_audit_state is not None
            and sequential_audit_state.adaptive_policy_context_sha256 is None
        )
        if allow_uncalibrated_sequential_analysis or existing_nonadaptive_state:
            corpus_issues.append(
                CorpusAdapterIssue(
                    severity=AdapterIssueSeverity.BLOCKING,
                    code="uncalibrated_sequential_audit_analysis_only",
                    detail=(
                        "This sequential audit was explicitly created without a frozen "
                        "adaptive calibration commitment and can never become a release "
                        "trajectory."
                    ),
                )
            )
        else:
            corpus_issues.append(
                CorpusAdapterIssue(
                    severity=AdapterIssueSeverity.BLOCKING,
                    code="adaptive_calibration_required_before_audit_genesis",
                    detail=(
                        "A release-capable sequential audit must bind its adaptive "
                        "calibration bundle before the audit state is created."
                    ),
                )
            )
    corpus_issues.sort(key=lambda issue: (issue.finding_id or "", issue.paper_id or "", issue.code))
    source_prepared = prepare_verification_scientific_state(
        manifest=manifest,
        graph=corpus.graph,
        pipeline_verification=pipeline_verification,
        item_risk_calibration_bundle=item_risk_calibration_bundle,
        item_risk_candidates=item_risk_candidates,
    )
    if (
        item_risk_scoring_receipt is not None
        and list(source_prepared.item_risk_bounds)
        != item_risk_scoring_receipt.bounds
    ):
        raise VerificationContractError(
            "item_risk_scoring_receipt_bound_replay_mismatch"
        )
    if sequential_audit_state is not None and receipts:
        raise VerificationContractError(
            "sequential_audit_state_conflicts_with_v1_resolution_receipts"
        )
    prepared = source_prepared
    effective_graph = corpus.graph
    if sequential_audit_state is not None:
        prepared = _replay_corrected_sequential_science(
            manifest=manifest,
            corpus=corpus,
            state=sequential_audit_state,
            pipeline_verification=pipeline_verification,
            budget_minutes=budget_minutes,
            item_risk_calibration_bundle=item_risk_calibration_bundle,
            item_risk_candidates=item_risk_candidates,
        )
        effective_graph = sequential_audit_state.graph
    target = prepared.target
    claim_model = prepared.claim_model
    candidates = list(prepared.audit_candidates)
    counterfactuals = list(prepared.counterfactuals)
    synthesis = prepared.synthesis
    item_risk_bounds = list(prepared.item_risk_bounds)
    effective_sequential_state = sequential_audit_state
    if effective_sequential_state is None and not receipts and (
        adaptive_calibration_bundle is not None
        or allow_uncalibrated_sequential_analysis
    ):
        effective_sequential_state = _create_initial_sequential_audit_state(
            manifest=manifest,
            graph=corpus.graph,
            synthesis=synthesis,
            candidates=candidates,
            claim_model=claim_model,
            counterfactuals=counterfactuals,
            item_risk_bounds=item_risk_bounds,
            pipeline_sha256=pipeline_sha256,
            budget_minutes=budget_minutes,
            created_at=generated_at,
            adaptive_policy_context_sha256=(
                None
                if adaptive_policy_context_for_certificate is None
                else adaptive_policy_context_for_certificate.policy_context_sha256
            ),
            adaptive_calibration_bundle_sha256=(
                None
                if adaptive_calibration_bundle is None
                else adaptive_calibration_bundle.bundle_sha256
            ),
        )
    blocking_adapter_reasons = sorted(
        {
            f"adapter:{issue.code}"
            for issue in corpus_issues
            if issue.severity is AdapterIssueSeverity.BLOCKING
        }
    )

    def assess_current_frozen_state(
        state: SequentialVerificationState | None,
        *,
        scientific_graph: EvidenceGraph,
        scientific_state: PreparedVerificationScientificState,
        fixed_bundle: FrozenCalibrationBundle | None,
        adaptive_bundle: AdaptiveCalibrationBundle | None,
        adaptive_candidate: ProspectiveAdaptiveReleaseCandidate | None,
    ) -> Any:
        if manifest.qualified_target is None:
            assessor = (
                _assess_claim_release_after_verifier_history_replay
                if adaptive_bundle is not None
                else assess_claim_release
            )
            return assessor(
                graph=scientific_graph,
                question_id=manifest.question_id,
                population_id=manifest.population_id,
                domain=manifest.domain,
                pipeline_sha256=pipeline_sha256,
                target=scientific_state.target,
                audit_candidates=list(scientific_state.audit_candidates),
                claim_model=scientific_state.claim_model,
                audit_resolution_receipts=receipts,
                audit_budget=budget_minutes,
                frozen_calibration_bundle=fixed_bundle,
                adaptive_calibration_bundle=adaptive_bundle,
                adaptive_release_candidate=adaptive_candidate,
                external_noncalibration_blocking_reasons=blocking_adapter_reasons,
                config=manifest.release,
                audit_guard_config=manifest.audit_guard.to_runtime(),
                sequential_audit_state=state,
            )
        qualified_assessor = (
            _assess_qualified_claim_release_after_verifier_history_replay
            if adaptive_bundle is not None
            else assess_qualified_claim_release
        )
        return qualified_assessor(
            graph=scientific_graph,
            question_id=manifest.question_id,
            population_id=manifest.population_id,
            domain=manifest.domain,
            pipeline_sha256=pipeline_sha256,
            target=manifest.qualified_target,
            audit_candidates=list(scientific_state.audit_candidates),
            claim_model=scientific_state.claim_model,
            audit_resolution_receipts=receipts,
            audit_budget=budget_minutes,
            frozen_calibration_bundle=fixed_bundle,
            adaptive_calibration_bundle=adaptive_bundle,
            adaptive_release_candidate=adaptive_candidate,
            external_noncalibration_blocking_reasons=blocking_adapter_reasons,
            config=manifest.release,
            audit_guard_config=manifest.audit_guard.to_runtime(),
            sequential_audit_state=state,
        )

    adaptive_release_candidate: ProspectiveAdaptiveReleaseCandidate | None = None
    adaptive_current_preselection_state = None
    adaptive_history = ()
    adaptive_history_context_sha256 = None
    adaptive_history_bundle_sha256 = None
    selection_predecessor_states = ()
    if effective_sequential_state is not None:
        try:
            (
                adaptive_history,
                adaptive_history_context_sha256,
                adaptive_history_bundle_sha256,
            ) = adaptive_preselection_history_from_state(
                effective_sequential_state
            )
            selection_predecessor_states = (
                selection_predecessor_states_from_state(
                    effective_sequential_state
                )
            )
        except SequentialVerificationContractError as exc:
            raise VerificationContractError(
                f"adaptive_selection_history_invalid:{exc}"
            ) from exc
    selection_transitions_exist = bool(
        effective_sequential_state is not None
        and any(
            transition.transition_kind == "selection"
            for transition in effective_sequential_state.transitions
        )
    )
    state_has_adaptive_commitment = (
        adaptive_history_context_sha256 is not None
        or adaptive_history_bundle_sha256 is not None
    )
    if adaptive_calibration_bundle is None:
        if state_has_adaptive_commitment:
            raise VerificationContractError(
                "adaptive_calibration_bundle_required_for_existing_history"
            )
    else:
        if effective_sequential_state is None:
            raise VerificationContractError(
                "adaptive_calibration_requires_sequential_production_state"
            )
        assert adaptive_policy_context_for_certificate is not None
        if not state_has_adaptive_commitment:
            raise VerificationContractError(
                "adaptive_calibration_cannot_activate_after_state_genesis"
            )
        if selection_transitions_exist and not adaptive_history:
            raise VerificationContractError(
                "adaptive_calibration_cannot_activate_after_nonadaptive_selection"
            )
        if (
            adaptive_history_context_sha256
            != adaptive_policy_context_for_certificate.policy_context_sha256
            or adaptive_history_bundle_sha256
            != adaptive_calibration_bundle.bundle_sha256
        ):
            raise VerificationContractError(
                "adaptive_selection_history_calibration_identity_mismatch"
            )
        if len(selection_predecessor_states) != len(adaptive_history):
            raise VerificationContractError(
                "adaptive_selection_history_snapshot_count_mismatch"
            )
        for index, (checkpoint, historical_state) in enumerate(
            zip(
                adaptive_history,
                selection_predecessor_states,
                strict=True,
            )
        ):
            try:
                recomputed_checkpoint = (
                    recompute_verifier_adaptive_preselection_checkpoint(
                        manifest=manifest,
                        state=historical_state,
                        pipeline_verification=pipeline_verification,
                        item_risk_scoring_receipt=item_risk_scoring_receipt,
                        blocking_adapter_reasons=blocking_adapter_reasons,
                    )
                )
            except VerificationContractError as exc:
                raise VerificationContractError(
                    "adaptive_selection_checkpoint_scientific_replay_failed:"
                    f"{index}:{exc}"
                ) from exc
            if recomputed_checkpoint != checkpoint:
                raise VerificationContractError(
                    "adaptive_selection_checkpoint_assessment_mismatch:"
                    f"{index}"
                )
        if effective_sequential_state.session.active_action is None:
            shadow_assessment = assess_current_frozen_state(
                effective_sequential_state,
                scientific_graph=effective_graph,
                scientific_state=prepared,
                fixed_bundle=None,
                adaptive_bundle=None,
                adaptive_candidate=None,
            )
            try:
                adaptive_current_preselection_state = (
                    freeze_preselection_state_from_production_components(
                        sequential_state=effective_sequential_state,
                        release_assessment=shadow_assessment,
                        blocking_adapter_reasons=blocking_adapter_reasons,
                    )
                )
            except AdaptiveCalibrationError as exc:
                raise VerificationContractError(
                    f"adaptive_current_preselection_projection_failed:{exc}"
                ) from exc
            observed_states = [
                *adaptive_history,
                adaptive_current_preselection_state,
            ]
        else:
            if not adaptive_history:
                raise VerificationContractError(
                    "adaptive_active_action_missing_selection_checkpoint"
                )
            observed_states = list(adaptive_history)
        try:
            adaptive_release_candidate = freeze_prospective_adaptive_candidate(
                question_id=manifest.question_id,
                population_id=manifest.population_id,
                domain=manifest.domain,
                policy_arm_id=adaptive_policy_context_for_certificate.policy_arm_id,
                policy_context_sha256=(
                    adaptive_policy_context_for_certificate.policy_context_sha256
                ),
                corpus=complete_corpus_identity,
                observed_states=observed_states,
            )
        except AdaptiveCalibrationError as exc:
            raise VerificationContractError(
                f"adaptive_release_candidate_derivation_failed:{exc}"
            ) from exc
        try:
            derived_adaptive_assessment = assess_adaptive_release_candidate(
                adaptive_release_candidate,
                adaptive_calibration_bundle,
            )
        except AdaptiveCalibrationError as exc:
            raise VerificationContractError(
                f"adaptive_release_candidate_replay_failed:{exc}"
            ) from exc
        if (
            effective_sequential_state.session.active_action is not None
            and derived_adaptive_assessment.status == "released"
        ):
            raise VerificationContractError(
                "adaptive_active_action_selected_after_qualifying_release"
            )

    # Production uses the literal prespecified stopping rule evaluated by the
    # question benchmark: on every no-active-action state, freeze the *complete*
    # release assessment before deciding whether another human action may start.
    # Audit-guard eligibility alone is insufficient because synthesis,
    # calibration, or corpus provenance can still require abstention.
    assessment = assess_current_frozen_state(
        effective_sequential_state,
        scientific_graph=effective_graph,
        scientific_state=prepared,
        fixed_bundle=frozen_calibration_bundle,
        adaptive_bundle=adaptive_calibration_bundle,
        adaptive_candidate=adaptive_release_candidate,
    )
    evaluated_sequential_state = effective_sequential_state
    preselection_assessment = assessment
    full_release_eligible = should_stop_at_full_frozen_release(
        release_status=assessment.status.value,
        blocking_reasons=blocking_adapter_reasons,
        active_action=(
            effective_sequential_state is not None
            and effective_sequential_state.session.active_action is not None
        ),
    )
    selection_result = None
    if effective_sequential_state is None:
        production_stop_outcome = (
            "legacy_receipt_assessment"
            if receipts
            else "adaptive_calibration_required_before_audit_genesis"
        )
    elif full_release_eligible:
        production_stop_outcome = "stopped_released"
    elif effective_sequential_state.session.active_action is not None:
        production_stop_outcome = "active_action_in_progress"
    elif adaptive_calibration_bundle is None and not (
        allow_uncalibrated_sequential_analysis
    ):
        production_stop_outcome = "adaptive_calibration_required_before_selection"
    elif (
        effective_sequential_state.session.status.value == "active"
        and effective_sequential_state.session.remaining_budget > 0
    ):
        try:
            selection_result = select_next_audit_candidate(
                effective_sequential_state,
                expected=freeze_state_expectation(effective_sequential_state),
                selected_at=generated_at,
                adaptive_preselection_state=adaptive_current_preselection_state,
                adaptive_policy_context_sha256=(
                    None
                    if adaptive_policy_context_for_certificate is None
                    else adaptive_policy_context_for_certificate.policy_context_sha256
                ),
                adaptive_calibration_bundle_sha256=(
                    None
                    if adaptive_calibration_bundle is None
                    else adaptive_calibration_bundle.bundle_sha256
                ),
            )
        except SequentialVerificationContractError as exc:
            if str(exc) != "no_eligible_candidate_fits_remaining_budget":
                raise VerificationContractError(
                    f"sequential_audit_selection_failed:{exc}"
                ) from exc
            production_stop_outcome = "no_feasible_action"
        else:
            production_stop_outcome = "selected_next_action"
            effective_sequential_state = selection_result.state
            # The newly active action is itself a release blocker. Persist the
            # assessment of the post-selection state in the certificate while the
            # preselection assessment remains control-flow-only and label-free.
            assessment = assess_current_frozen_state(
                effective_sequential_state,
                scientific_graph=effective_graph,
                scientific_state=prepared,
                fixed_bundle=frozen_calibration_bundle,
                adaptive_bundle=adaptive_calibration_bundle,
                adaptive_candidate=adaptive_release_candidate,
            )
    else:
        production_stop_outcome = "no_feasible_action"
    production_stop_decision = freeze_production_stop_decision(
        evaluated_state=evaluated_sequential_state,
        release_assessment=preselection_assessment,
        blocking_adapter_reasons=blocking_adapter_reasons,
        outcome=production_stop_outcome,
        selection_result=selection_result,
    )
    synthesis_hash = hash_canonical(synthesis)
    if assessment.synthesis_sha256 != synthesis_hash:
        raise VerificationContractError("orchestrator_release_synthesis_hash_mismatch")

    reasons = list(assessment.reasons)
    reasons.extend(blocking_adapter_reasons)
    reasons = sorted(set(reasons))
    status: Literal["released", "abstained"] = (
        "released" if assessment.status.value == "released" and not reasons else "abstained"
    )
    manifest_payload = manifest.model_dump(mode="json")
    manifest_hash = hash_canonical(manifest_payload)
    source_graph_hash = hash_canonical(corpus.graph)
    graph_hash = hash_canonical(effective_graph)
    candidate_payload = [asdict(candidate) for candidate in candidates]
    candidate_hash = hash_canonical(candidate_payload)
    counterfactual_hash = hash_canonical(counterfactuals)
    audit_transitions = (
        list(effective_sequential_state.transitions)
        if effective_sequential_state is not None
        else []
    )
    correction_receipts = [
        transition.receipt
        for transition in audit_transitions
        if transition.transition_kind == "correction"
    ]
    audit_transition_hash = hash_canonical(audit_transitions)
    correction_receipt_hash = hash_canonical(correction_receipts)
    adaptive_prospective_assessment = None
    if adaptive_calibration_bundle is not None:
        assert adaptive_release_candidate is not None
        adaptive_prospective_assessment = assess_adaptive_release_candidate(
            adaptive_release_candidate,
            adaptive_calibration_bundle,
        )
    release_input_hash = hash_canonical(
        {
            "audit_candidates": candidate_payload,
            "audit_receipts": receipts,
            "sequential_audit_state": effective_sequential_state,
            "budget_minutes": float(budget_minutes),
            "complete_corpus_identity": complete_corpus_identity,
            "item_risk_scoring_receipt": item_risk_scoring_receipt,
            "adaptive_policy_context": adaptive_policy_context_for_certificate,
            "adaptive_calibration_bundle": adaptive_calibration_bundle,
            "adaptive_release_candidate": adaptive_release_candidate,
            "adaptive_prospective_assessment": adaptive_prospective_assessment,
            "pipeline_sha256": pipeline_sha256,
            "production_stop_decision_sha256": (
                production_stop_decision.decision_sha256
            ),
            "target": manifest.qualified_target or target,
        }
    )
    lineage = [
        CertificateLineageStage(
            stage="pipeline_identity_verification",
            input_sha256s={"expected_pipeline": pipeline_verification.expected_pipeline_sha256},
            output_sha256s={"pipeline_verification": pipeline_verification.verification_sha256},
            method="recomputed-explicit-file-and-settings-fingerprint",
        ),
        CertificateLineageStage(
            stage="corpus_to_evidence_graph",
            input_sha256s=dict(
                sorted(
                    {
                        "claim_manifest": manifest_hash,
                        "corpus_source": corpus.source_sha256,
                    }.items()
                )
            ),
            output_sha256s={"source_evidence_graph": source_graph_hash},
            method=f"{corpus.source_format}:closed-corpus-adapter",
        ),
        CertificateLineageStage(
            stage="audit_correction_replay",
            input_sha256s={"source_evidence_graph": source_graph_hash},
            output_sha256s=dict(
                sorted(
                    {
                        "audit_correction_receipts": correction_receipt_hash,
                        "audit_transition_ledger": audit_transition_hash,
                        "evidence_graph": graph_hash,
                        **(
                            {
                                "sequential_audit_state": (
                                    effective_sequential_state.state_sha256
                                )
                            }
                            if effective_sequential_state is not None
                            else {}
                        ),
                    }.items()
                )
            ),
            method="hash-chained-sequential-audit-correction-replay-v1",
        ),
        CertificateLineageStage(
            stage="evidence_graph_to_synthesis",
            input_sha256s={"evidence_graph": graph_hash},
            output_sha256s={"synthesis": synthesis_hash},
            method=str(synthesis.get("mode", "insufficient")),
        ),
        CertificateLineageStage(
            stage="counterfactual_verification_priority",
            input_sha256s=dict(
                sorted(
                    {
                        "evidence_graph": graph_hash,
                        "synthesis": synthesis_hash,
                    }.items()
                )
            ),
            output_sha256s={
                "audit_candidates": candidate_hash,
                "counterfactual_reruns": counterfactual_hash,
            },
            method="actual-leave-one-out-synthesis-reruns",
        ),
        CertificateLineageStage(
            stage="item_risk_scoring",
            input_sha256s={
                "audit_candidates": candidate_hash,
                "pipeline_verification": pipeline_verification.verification_sha256,
            },
            output_sha256s={
                "item_risk_scoring_receipt": (
                    hash_canonical(None)
                    if item_risk_scoring_receipt is None
                    else item_risk_scoring_receipt.receipt_sha256
                )
            },
            method=(
                "not-supplied"
                if item_risk_scoring_receipt is None
                else "recomputed-self-contained-item-risk-scoring-run-v2"
            ),
        ),
        CertificateLineageStage(
            stage="adaptive_first_release_calibration",
            input_sha256s=dict(
                sorted(
                    {
                        "adaptive_calibration_bundle": (
                            hash_canonical(None)
                            if adaptive_calibration_bundle is None
                            else adaptive_calibration_bundle.bundle_sha256
                        ),
                        "adaptive_policy_context": (
                            hash_canonical(None)
                            if adaptive_policy_context_for_certificate is None
                            else adaptive_policy_context_for_certificate.policy_context_sha256
                        ),
                        "adaptive_release_candidate": (
                            hash_canonical(None)
                            if adaptive_release_candidate is None
                            else adaptive_release_candidate.candidate_sha256
                        ),
                        "complete_corpus_identity": (
                            complete_corpus_identity.membership_sha256
                        ),
                    }.items()
                )
            ),
            output_sha256s={
                "adaptive_prospective_assessment": (
                    hash_canonical(None)
                    if adaptive_prospective_assessment is None
                    else adaptive_prospective_assessment.assessment_sha256
                )
            },
            method=(
                "not-supplied"
                if adaptive_prospective_assessment is None
                else "recomputed-complete-question-first-release-trajectory-v1"
            ),
        ),
        CertificateLineageStage(
            stage="risk_controlled_release",
            input_sha256s={"release_inputs": release_input_hash},
            output_sha256s={"release_decision": assessment.decision_sha256},
            method=(
                "prospective-qualified-claim-release-v2"
                if manifest.qualified_target is not None
                else "prospective-claim-release-v2"
            ),
        ),
    ]
    corpus_payload = corpus.certificate_payload()
    corpus_payload["declared_corpus_cutoff"] = manifest.protocol.corpus_cutoff
    corpus_payload["pipeline_identity_basis"] = pipeline_basis
    return freeze_verification_certificate(
        generated_at=generated_at,
        status=status,
        reasons=reasons,
        claim_manifest=manifest_payload,
        corpus=corpus_payload,
        corpus_sha256=corpus.source_sha256,
        source_evidence_graph=corpus.graph,
        evidence_graph=effective_graph,
        adapter_issues=[issue.model_dump(mode="json") for issue in corpus_issues],
        synthesis=synthesis,
        counterfactual_reruns=counterfactuals,
        audit_candidates=candidate_payload,
        release_assessment=assessment,
        lineage=lineage,
        pipeline_verification=pipeline_verification,
        complete_corpus_identity=complete_corpus_identity,
        item_risk_scoring_receipt=item_risk_scoring_receipt,
        adaptive_calibration_bundle=adaptive_calibration_bundle,
        adaptive_release_candidate=adaptive_release_candidate,
        sequential_audit_state=effective_sequential_state,
        production_stop_decision=production_stop_decision,
    )


def build_offline_fixture() -> tuple[ClaimManifest, CorpusLoadResult]:
    """Construct a deterministic three-study fixture with no provider dependencies."""

    fixture_estimand = "between-group standardized mean difference at 4 weeks"
    graphs: list[EvidenceGraph] = []
    for index, estimate in enumerate((0.45, 0.62, 0.51), start=1):
        suffix = str(index)
        context = GraphAdapterContext(
            publication=PublicationIdentity(
                publication_id=f"fixture-publication-{suffix}",
                paper_id=f"fixture-paper-{suffix}",
                doc_id=f"fixture-document-{suffix}",
                title=f"Offline verifier fixture study {suffix}",
            ),
            study_id=f"fixture-study-{suffix}",
            cohort_identity=CohortIdentity(
                cohort_id=f"fixture-cohort-{suffix}",
                basis=CohortIdentityBasis.REVIEWER_RECONCILED,
                rationale="Deterministic fixture identity; not an empirical study.",
            ),
            treatment_arm_id=f"fixture-treatment-{suffix}",
            comparator_arm_id=f"fixture-control-{suffix}",
            contrast_id=f"fixture-contrast-{suffix}",
            contrast_label="intervention_vs_control",
            positive_direction_means="higher fixture outcome under intervention",
            treatment_label="fixture intervention",
            comparator_label="fixture control",
            timepoint=OutcomeTimepoint(kind="exact", value=4, unit="week"),
        )
        effect = EffectEvidence(
            paper_id=f"fixture-paper-{suffix}",
            finding_id=f"fixture-finding-{suffix}",
            outcome="fixture_outcome",
            contrast="intervention_vs_control",
            effect_format="hedges_g",
            availability="available",
            estimate=estimate,
            standard_error=0.08,
            reported_significance="significant",
            provenance={
                "source_locator": f"fixture-paper-{suffix}#page=1",
                "source_quote": f"The planted standardized estimate was {estimate}.",
            },
        )
        graphs.append(adapt_effect_evidence(effect, context=context).graph)
    graph_payload = _merge_graphs(graphs).model_dump(mode="json")
    for contrast in graph_payload["contrasts"]:
        contrast["estimand"] = fixture_estimand
    graph = EvidenceGraph.model_validate(graph_payload)
    manifest = ClaimManifest(
        question_id="offline-verifier-fixture",
        population_id="offline-fixture-population",
        domain="synthetic",
        claim=ScientificClaim(
            statement="The fixture intervention increases the fixture outcome.",
            direction=TargetDirection.INCREASE,
            outcome_name="fixture_outcome",
            estimand=fixture_estimand,
        ),
        protocol=VerificationProtocol(
            corpus_cutoff="deterministic-offline-fixture-v1",
            inclusion_criteria=["all three generated fixture studies"],
            exclusion_criteria=[],
        ),
    )
    source_hash = hash_canonical(
        {"fixture": "unified-verifier-v1", "graph": graph.model_dump(mode="json")}
    )
    corpus = CorpusLoadResult(
        corpus_id="offline-verifier-fixture-v1",
        source_label="embedded:offline-verifier-fixture-v1",
        source_format="embedded_synthetic_fixture",
        source_sha256=source_hash,
        graph=graph,
        eligibility=_graph_eligibility(graph),
        adapter_issues=(),
        metadata={"empirical_evidence": False, "purpose": "offline_integration_test"},
        provenance_assurance=CorpusProvenanceAssurance(
            status="embedded_synthetic_fixture",
            reason=(
                "Deterministic embedded synthetic fixture authorized only for mechanical "
                "integration testing; it is not empirical evidence."
            ),
        ),
    )
    return manifest, corpus


__all__ = [
    "AuditGuardConfig",
    "AuditPolicyConfig",
    "ClaimManifest",
    "CorpusAdapterIssue",
    "CorpusEligibilityRecord",
    "CorpusLoadResult",
    "CorpusProvenanceAssurance",
    "LegacyAdapterConfig",
    "PreparedVerificationScientificState",
    "ScientificClaim",
    "VerificationContractError",
    "VerificationProtocol",
    "adapt_legacy_findings",
    "build_graph_counterfactual_audits",
    "build_offline_fixture",
    "compute_candidate_runner_sha256",
    "compute_synthesis_runner_sha256",
    "compute_verification_policy_sha256",
    "compute_verifier_pipeline_fingerprint",
    "finalize_condition_verification",
    "load_claim_manifest",
    "load_corpus",
    "prepare_verification_scientific_state",
    "run_condition_calibration_collection",
    "run_verification",
    "sequential_candidates_from_prepared_state",
    "validate_condition_calibration_assessment_receipt_external_replay",
    "validate_condition_calibration_collection_source_external_replay",
    "verifier_pipeline_components",
]
