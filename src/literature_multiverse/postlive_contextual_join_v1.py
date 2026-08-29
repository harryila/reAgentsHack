"""Join one validated contextual live result to verifier mechanics.

This module is deliberately narrow.  It accepts only a fully replayed successful
contextual-frontier workspace, recomputes the graph synthesis and every leave-one-out
counterfactual, and freezes scheduler-ready audit inputs.  It does not manufacture a
claim-level calibration set, a human cost measurement, an adjudication, a condition
hypothesis, or release authority.

The default audit weights and costs are explicit unit proxies.  They exist only so
the existing influence scheduler can expose the next evidence item mechanically;
they are not empirical probabilities or minutes.  No audit action is selected and
no correction is applied by this join.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.budgeted_verification import (
    AllocationPolicy,
    ProbabilityBasis,
    rank_candidates,
)
from literature_multiverse.evidence_graph import EvidenceGraph
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.meta_analysis import (
    GraphCounterfactualAuditPlan,
    build_graph_counterfactual_audit_plan,
    synthesize_evidence_graph,
)
from literature_multiverse.metasyn_contextual_frontier_runtime_v1 import (
    MetaSynContextualFrontierTerminalReportV1,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.sequential_verification import (
    CurrentAuditCandidate,
    SequentialVerificationState,
    create_sequential_verification_state,
    current_candidates_from_audit_candidates,
    resume_sequential_verification_state,
)

JOIN_VERSION = "postlive-contextual-join-v1"
CERTIFICATE_VERSION = "postlive-contextual-non-authorizing-certificate-v1"
UNIT_RISK_SOURCE = (
    "fixed_unity_heuristic_weight_for_leave_one_out_influence_only;"
    "not_an_empirical_error_probability_or_calibration_bound"
)
UNIT_COST = 1.0
UNIT_COST_UNIT = "unmeasured_relative_unit_cost_proxy"
AUDIT_POLICY = AllocationPolicy.INFLUENCE_ONLY


class PostLiveContextualJoinV1Error(ValueError):
    """The runtime or join artifact violated the closed mechanics contract."""


def _sha256(value: str, field_name: str) -> str:
    if SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"postlive_contextual_join_sha256_invalid:{field_name}")
    return value


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"postlive_contextual_join_timezone_required:{field_name}")
    return value


def _datetime_json(value: datetime) -> str:
    rendered = _aware(value, "datetime_json").isoformat()
    return f"{rendered[:-6]}Z" if rendered.endswith("+00:00") else rendered


def _self_hash(model: ContractModel, field_name: str) -> None:
    payload = model.model_dump(mode="json", exclude={field_name})
    if hash_canonical(payload) != getattr(model, field_name):
        raise ValueError(f"postlive_contextual_join_self_hash_mismatch:{field_name}")


def _canonical_moderators(values: Sequence[str]) -> list[str]:
    moderators = [value.strip() for value in values]
    if any(not value for value in moderators):
        raise PostLiveContextualJoinV1Error("postlive_condition_moderator_empty")
    if moderators != sorted(set(moderators)):
        raise PostLiveContextualJoinV1Error("postlive_condition_moderators_must_be_sorted_unique")
    return moderators


class PostLiveConditionMechanicsV1(ContractModel):
    """Exact same-graph condition mechanics, never held-out confirmation."""

    condition_version: Literal["postlive-condition-mechanics-v1"] = (
        "postlive-condition-mechanics-v1"
    )
    status: Literal[
        "not_scientifically_defined",
        "executed_insufficient",
        "executed_exploratory",
    ]
    prespecified_moderators: list[str]
    analysis: dict[str, Any] | None
    analysis_sha256: str | None
    reason: str
    analysis_executed: bool
    held_out_confirmation_performed: Literal[False] = False
    condition_claim_authority: Literal[False] = False
    condition_sha256: str

    @field_validator("analysis_sha256", "condition_sha256")
    @classmethod
    def validate_hash(cls, value: str | None, info: Any) -> str | None:
        if value is not None:
            return _sha256(value, info.field_name)
        return value

    @field_validator("prespecified_moderators")
    @classmethod
    def validate_moderators(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item for item in value):
            raise ValueError("postlive_condition_moderators_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_condition(self) -> PostLiveConditionMechanicsV1:
        if (self.analysis is None) != (self.analysis_sha256 is None):
            raise ValueError("postlive_condition_analysis_hash_presence_mismatch")
        if self.analysis is not None and self.analysis_sha256 != hash_canonical(self.analysis):
            raise ValueError("postlive_condition_analysis_hash_mismatch")
        if self.status == "not_scientifically_defined":
            if self.prespecified_moderators or self.analysis is not None or self.analysis_executed:
                raise ValueError("postlive_undefined_condition_has_analysis")
        elif (
            not self.prespecified_moderators or self.analysis is None or not self.analysis_executed
        ):
            raise ValueError("postlive_executed_condition_missing_inputs")
        if (
            self.status == "executed_insufficient"
            and self.analysis is not None
            and self.analysis.get("status") != "insufficient"
        ):
            raise ValueError("postlive_condition_insufficient_status_mismatch")
        if (
            self.status == "executed_exploratory"
            and self.analysis is not None
            and self.analysis.get("status") != "ok"
        ):
            raise ValueError("postlive_condition_exploratory_status_mismatch")
        _self_hash(self, "condition_sha256")
        return self


class PostLiveAuditMechanicsV1(ContractModel):
    """Leave-one-out reruns plus an unresolved scheduler genesis state."""

    audit_version: Literal["postlive-audit-mechanics-v1"] = "postlive-audit-mechanics-v1"
    status: Literal["scheduler_ready_no_audit_selected"] = "scheduler_ready_no_audit_selected"
    policy: Literal["influence_only"] = "influence_only"
    probability_basis: Literal["heuristic"] = "heuristic"
    probability_source: Literal[
        "fixed_unity_heuristic_weight_for_leave_one_out_influence_only;not_an_empirical_error_probability_or_calibration_bound"
    ] = UNIT_RISK_SOURCE
    unit_error_weight: Literal[1.0] = 1.0
    unit_cost: Literal[1.0] = 1.0
    cost_unit: Literal["unmeasured_relative_unit_cost_proxy"] = UNIT_COST_UNIT
    baseline_decision: dict[str, Any]
    baseline_decision_sha256: str
    audit_candidates: list[dict[str, Any]]
    audit_candidate_membership_sha256: str
    priority_records: list[dict[str, Any]]
    priority_record_membership_sha256: str
    counterfactual_syntheses: dict[str, dict[str, Any]]
    counterfactual_synthesis_membership_sha256: str
    counterfactual_decisions: dict[str, dict[str, Any]]
    counterfactual_decision_membership_sha256: str
    current_candidates: Annotated[list[CurrentAuditCandidate], Field(min_length=1)]
    current_candidate_membership_sha256: str
    sequential_state: SequentialVerificationState
    sequential_state_sha256: str
    item_error_calibration_performed: Literal[False] = False
    human_cost_measurement_performed: Literal[False] = False
    audit_action_selected: Literal[False] = False
    human_adjudication_count: Literal[0] = 0
    correction_count: Literal[0] = 0
    release_risk_bound_available: Literal[False] = False
    audit_sha256: str

    @field_validator(
        "baseline_decision_sha256",
        "audit_candidate_membership_sha256",
        "priority_record_membership_sha256",
        "counterfactual_synthesis_membership_sha256",
        "counterfactual_decision_membership_sha256",
        "current_candidate_membership_sha256",
        "sequential_state_sha256",
        "audit_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_audit(self) -> PostLiveAuditMechanicsV1:
        candidate_ids = [str(item.get("item_id", "")) for item in self.audit_candidates]
        current_ids = [item.item_id for item in self.current_candidates]
        priority_ids = [str(item.get("item_id", "")) for item in self.priority_records]
        if (
            not candidate_ids
            or candidate_ids != sorted(set(candidate_ids))
            or current_ids != candidate_ids
            or priority_ids != candidate_ids
            or set(self.counterfactual_syntheses) != set(candidate_ids)
            or set(self.counterfactual_decisions) != set(candidate_ids)
        ):
            raise ValueError("postlive_audit_candidate_identity_mismatch")
        checks = (
            (self.baseline_decision_sha256, self.baseline_decision),
            (self.audit_candidate_membership_sha256, self.audit_candidates),
            (self.priority_record_membership_sha256, self.priority_records),
            (
                self.counterfactual_synthesis_membership_sha256,
                self.counterfactual_syntheses,
            ),
            (
                self.counterfactual_decision_membership_sha256,
                self.counterfactual_decisions,
            ),
            (
                self.current_candidate_membership_sha256,
                [item.candidate_sha256 for item in self.current_candidates],
            ),
        )
        if any(observed != hash_canonical(value) for observed, value in checks):
            raise ValueError("postlive_audit_nested_hash_mismatch")
        resumed = resume_sequential_verification_state(self.sequential_state)
        if (
            resumed.state_sha256 != self.sequential_state_sha256
            or resumed.transitions
            or resumed.session.active_action is not None
            or resumed.session.selected_item_ids
        ):
            raise ValueError("postlive_audit_scheduler_state_not_unselected_genesis")
        _self_hash(self, "audit_sha256")
        return self


class PostLiveContextualCertificateV1(ContractModel):
    """Self-contained diagnostic certificate that can never authorize release."""

    certificate_version: Literal["postlive-contextual-non-authorizing-certificate-v1"] = (
        CERTIFICATE_VERSION
    )
    generated_at: datetime
    status: Literal["mechanics_completed_non_authorizing"] = "mechanics_completed_non_authorizing"
    target_direction: Literal["increase", "decrease"]
    runtime_workspace_validation_sha256: str
    terminal_report_sha256: str
    terminal_status: Literal["typed_graph_smoke_completed"] = "typed_graph_smoke_completed"
    successful_validation_sha256: str
    provider_receipt_sha256: str
    provider_result_sha256: str
    provider_execution_binding_sha256: str
    contextual_grounding_core_sha256: str
    runtime_grounding_binding_sha256: str
    native_projection_sha256: str
    fragment_sha256: str
    runtime_pipeline_sha256: str
    integration_pipeline_sha256: str
    evidence_graph: EvidenceGraph
    evidence_graph_sha256: str
    outcome_name: str
    contrast_id: str
    synthesis: dict[str, Any]
    synthesis_sha256: str
    condition_mechanics: PostLiveConditionMechanicsV1
    audit_mechanics: PostLiveAuditMechanicsV1
    blockers: Annotated[list[str], Field(min_length=1)]
    title_abstract_only: Literal[True] = True
    single_publication: Literal[True] = True
    complete_corpus: Literal[False] = False
    provider_call_success_is_not_accuracy_evidence: Literal[True] = True
    graph_construction_mechanics_authority: Literal[True] = True
    extraction_accuracy_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    condition_claim_authority: Literal[False] = False
    adaptive_effectiveness_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    release_authorizing: Literal[False] = False
    certificate_sha256: str

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _aware(value, "generated_at")

    @field_validator(
        "runtime_workspace_validation_sha256",
        "terminal_report_sha256",
        "successful_validation_sha256",
        "provider_receipt_sha256",
        "provider_result_sha256",
        "provider_execution_binding_sha256",
        "contextual_grounding_core_sha256",
        "runtime_grounding_binding_sha256",
        "native_projection_sha256",
        "fragment_sha256",
        "runtime_pipeline_sha256",
        "integration_pipeline_sha256",
        "evidence_graph_sha256",
        "synthesis_sha256",
        "certificate_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("blockers")
    @classmethod
    def validate_blockers(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item for item in value):
            raise ValueError("postlive_certificate_blockers_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_certificate(self) -> PostLiveContextualCertificateV1:
        required = {
            "calibration_not_performed",
            "complete_corpus_not_available",
            "extraction_accuracy_not_evaluated",
            "human_adjudication_absent",
            "human_verification_cost_unmeasured",
            "single_publication_mechanics_only",
            "title_or_abstract_only_not_release_grade",
        }
        if not required <= set(self.blockers):
            raise ValueError("postlive_certificate_required_blocker_missing")
        if (
            len(self.evidence_graph.publications) != 1
            or self.evidence_graph_sha256 != hash_canonical(self.evidence_graph)
            or self.synthesis_sha256 != hash_canonical(self.synthesis)
            or self.audit_mechanics.sequential_state.graph_sha256 != self.evidence_graph_sha256
            or self.audit_mechanics.sequential_state.synthesis_sha256 != self.synthesis_sha256
        ):
            raise ValueError("postlive_certificate_scientific_artifact_hash_mismatch")
        _self_hash(self, "certificate_sha256")
        return self


def build_postlive_condition_mechanics_v1(
    *,
    moderators: Sequence[str],
    synthesis: Mapping[str, Any],
) -> PostLiveConditionMechanicsV1:
    """Freeze source-neutral condition mechanics for an already replayed synthesis."""

    canonical = _canonical_moderators(moderators)
    analysis = synthesis.get("condition_analysis")
    if not canonical:
        if analysis is not None:
            raise PostLiveContextualJoinV1Error(
                "postlive_condition_analysis_without_prespecification"
            )
        payload: dict[str, Any] = {
            "condition_version": "postlive-condition-mechanics-v1",
            "status": "not_scientifically_defined",
            "prespecified_moderators": [],
            "analysis": None,
            "analysis_sha256": None,
            "reason": (
                "no_prespecified_moderator_target;condition_claim_not_inferred_from_one_result"
            ),
            "analysis_executed": False,
            "held_out_confirmation_performed": False,
            "condition_claim_authority": False,
        }
    else:
        if not isinstance(analysis, Mapping):
            raise PostLiveContextualJoinV1Error("postlive_prespecified_condition_analysis_missing")
        frozen_analysis = dict(analysis)
        status = (
            "executed_insufficient"
            if frozen_analysis.get("status") == "insufficient"
            else "executed_exploratory"
        )
        payload = {
            "condition_version": "postlive-condition-mechanics-v1",
            "status": status,
            "prespecified_moderators": canonical,
            "analysis": frozen_analysis,
            "analysis_sha256": hash_canonical(frozen_analysis),
            "reason": ("same_graph_prespecified_mechanics_only;held_out_confirmation_absent"),
            "analysis_executed": True,
            "held_out_confirmation_performed": False,
            "condition_claim_authority": False,
        }
    return PostLiveConditionMechanicsV1.model_validate(
        {**payload, "condition_sha256": hash_canonical(payload)}
    )


def _condition_mechanics(
    *,
    moderators: Sequence[str],
    synthesis: Mapping[str, Any],
) -> PostLiveConditionMechanicsV1:
    """Compatibility wrapper for the original v1 terminal join."""

    return build_postlive_condition_mechanics_v1(
        moderators=moderators,
        synthesis=synthesis,
    )


def _audit_candidate_payload(plan: GraphCounterfactualAuditPlan) -> list[dict[str, Any]]:
    payloads = [asdict(candidate) for candidate in plan.candidates]
    payloads.sort(key=lambda item: str(item["item_id"]))
    return payloads


def build_postlive_audit_mechanics_from_source_identity_v1(
    *,
    created_at: datetime,
    source_identity_sha256: str,
    integration_pipeline_sha256: str,
    graph: EvidenceGraph,
    outcome_name: str,
    contrast_id: str,
    target_direction: Literal["increase", "decrease"],
    moderators: Sequence[str],
    synthesis: Mapping[str, Any],
) -> PostLiveAuditMechanicsV1:
    """Freeze source-neutral audit mechanics for an already validated graph.

    ``source_identity_sha256`` names the exact upstream artifact that licensed the
    mechanics run.  It need not be, and must not be represented as, a v1 runtime
    terminal hash when another truthful adapter supplies the graph.
    """

    _sha256(source_identity_sha256, "source_identity_sha256")
    item_ids = sorted(
        estimate.estimate_id
        for estimate in graph.outcome_estimates
        if estimate.outcome_name == outcome_name and estimate.contrast_id == contrast_id
    )
    if not item_ids:
        raise PostLiveContextualJoinV1Error("postlive_audit_no_matching_estimate")
    plan = build_graph_counterfactual_audit_plan(
        graph,
        outcome_name=outcome_name,
        contrast_id=contrast_id,
        target_direction=target_direction,
        error_probabilities={item_id: 1.0 for item_id in item_ids},
        verification_costs={item_id: UNIT_COST for item_id in item_ids},
        probability_basis=ProbabilityBasis.HEURISTIC,
        probability_source=UNIT_RISK_SOURCE,
        cost_unit=UNIT_COST_UNIT,
        disagreement_scores={item_id: 0.0 for item_id in item_ids},
        require_explicit_timepoint=True,
        require_prediction_interval_stability=True,
        confidence_level=0.95,
        assumed_within_cohort_correlation=1.0,
        prespecified_moderators=tuple(moderators),
        claim_id=(f"postlive-{source_identity_sha256[:16]}-{outcome_name}-{target_direction}"),
    )
    if plan.baseline_synthesis != dict(synthesis):
        raise PostLiveContextualJoinV1Error("postlive_audit_baseline_synthesis_replay_mismatch")
    candidate_payloads = _audit_candidate_payload(plan)
    priority_records = [
        asdict(item)
        for item in rank_candidates(
            plan.candidates,
            plan.claim_model,
            policy=AUDIT_POLICY,
        )
    ]
    counterfactual_syntheses = {
        item_id: value for item_id, value in sorted(plan.counterfactual_syntheses.items())
    }
    counterfactual_decisions = {
        item_id: value.model_dump(mode="json")
        for item_id, value in sorted(plan.counterfactual_decisions.items())
    }
    current = current_candidates_from_audit_candidates(
        plan.candidates,
        plan.claim_model,
        policy=AUDIT_POLICY,
        counterfactual_synthesis_sha256s={
            item_id: hash_canonical(value) for item_id, value in counterfactual_syntheses.items()
        },
    )
    policy_sha256 = hash_canonical(
        {
            "policy_version": "postlive-influence-only-unit-proxy-v1",
            "policy": AUDIT_POLICY.value,
            "probability_basis": ProbabilityBasis.HEURISTIC.value,
            "probability_source": UNIT_RISK_SOURCE,
            "unit_error_weight": 1.0,
            "unit_cost": UNIT_COST,
            "cost_unit": UNIT_COST_UNIT,
            "selection_performed": False,
        }
    )
    sequential = create_sequential_verification_state(
        session_id=f"postlive-{source_identity_sha256[:24]}",
        created_at=_aware(created_at, "created_at"),
        pipeline_sha256=integration_pipeline_sha256,
        policy_sha256=policy_sha256,
        budget=UNIT_COST,
        cost_unit=UNIT_COST_UNIT,
        graph=graph,
        synthesis=synthesis,
        candidates=current,
    )
    baseline_decision = plan.baseline_decision.model_dump(mode="json")
    payload = {
        "audit_version": "postlive-audit-mechanics-v1",
        "status": "scheduler_ready_no_audit_selected",
        "policy": AUDIT_POLICY.value,
        "probability_basis": ProbabilityBasis.HEURISTIC.value,
        "probability_source": UNIT_RISK_SOURCE,
        "unit_error_weight": 1.0,
        "unit_cost": UNIT_COST,
        "cost_unit": UNIT_COST_UNIT,
        "baseline_decision": baseline_decision,
        "baseline_decision_sha256": hash_canonical(baseline_decision),
        "audit_candidates": candidate_payloads,
        "audit_candidate_membership_sha256": hash_canonical(candidate_payloads),
        "priority_records": priority_records,
        "priority_record_membership_sha256": hash_canonical(priority_records),
        "counterfactual_syntheses": counterfactual_syntheses,
        "counterfactual_synthesis_membership_sha256": hash_canonical(counterfactual_syntheses),
        "counterfactual_decisions": counterfactual_decisions,
        "counterfactual_decision_membership_sha256": hash_canonical(counterfactual_decisions),
        "current_candidates": list(current),
        "current_candidate_membership_sha256": hash_canonical(
            [item.candidate_sha256 for item in current]
        ),
        "sequential_state": sequential,
        "sequential_state_sha256": sequential.state_sha256,
        "item_error_calibration_performed": False,
        "human_cost_measurement_performed": False,
        "audit_action_selected": False,
        "human_adjudication_count": 0,
        "correction_count": 0,
        "release_risk_bound_available": False,
    }
    return PostLiveAuditMechanicsV1.model_validate(
        {**payload, "audit_sha256": hash_canonical(payload)}
    )


def _audit_mechanics(
    *,
    created_at: datetime,
    terminal: MetaSynContextualFrontierTerminalReportV1,
    integration_pipeline_sha256: str,
    graph: EvidenceGraph,
    outcome_name: str,
    contrast_id: str,
    target_direction: Literal["increase", "decrease"],
    moderators: Sequence[str],
    synthesis: Mapping[str, Any],
) -> PostLiveAuditMechanicsV1:
    return build_postlive_audit_mechanics_from_source_identity_v1(
        created_at=created_at,
        source_identity_sha256=terminal.report_sha256,
        integration_pipeline_sha256=integration_pipeline_sha256,
        graph=graph,
        outcome_name=outcome_name,
        contrast_id=contrast_id,
        target_direction=target_direction,
        moderators=moderators,
        synthesis=synthesis,
    )


def freeze_postlive_contextual_certificate_v1(
    *,
    terminal_report: MetaSynContextualFrontierTerminalReportV1 | Mapping[str, Any],
    runtime_workspace_validation_sha256: str,
    generated_at: datetime,
    target_direction: Literal["increase", "decrease"],
    prespecified_moderators: Sequence[str] = (),
) -> PostLiveContextualCertificateV1:
    """Revalidate and join one successful runtime terminal report.

    ``runtime_workspace_validation_sha256`` must identify the external full-workspace
    replay performed by :func:`build_postlive_contextual_certificate_from_workspace_v1`.
    The lower-level entry point remains useful for deterministic fixture tests, but it
    does not itself inspect the workspace.
    """

    _sha256(runtime_workspace_validation_sha256, "runtime_workspace_validation_sha256")
    created = _aware(generated_at, "generated_at")
    moderators = _canonical_moderators(prespecified_moderators)
    raw_terminal = (
        terminal_report.model_dump(mode="json")
        if isinstance(terminal_report, MetaSynContextualFrontierTerminalReportV1)
        else terminal_report
    )
    try:
        terminal = MetaSynContextualFrontierTerminalReportV1.model_validate(raw_terminal)
    except ValueError as exc:
        raise PostLiveContextualJoinV1Error("postlive_terminal_contract_or_hash_invalid") from exc
    successes = [
        item
        for item in terminal.validation_results
        if item.status == "typed_graph_mechanics_completed"
    ]
    if (
        terminal.status != "typed_graph_smoke_completed"
        or len(successes) != 1
        or terminal.successful_request_key != successes[0].request_key
    ):
        raise PostLiveContextualJoinV1Error(
            "postlive_requires_exactly_one_typed_graph_terminal_success"
        )
    success = successes[0]
    projection = success.native_projection
    if (
        not success.fresh_native_typed_graph_completed
        or projection is None
        or projection.status != "typed_graph_mechanics_completed"
        or projection.outcome_origin != "runtime_outcome_supplied_by_caller"
        or projection.fragment is None
        or projection.fragment.graph is None
        or not projection.title_abstract_only_not_release_grade
        or "single_publication_mechanics_only" not in projection.blockers
        or "title_or_abstract_only_not_release_grade" not in projection.blockers
    ):
        raise PostLiveContextualJoinV1Error(
            "postlive_runtime_projection_not_eligible_for_mechanics_join"
        )
    graph = EvidenceGraph.model_validate(projection.fragment.graph.model_dump(mode="json"))
    if (
        len(graph.publications) != 1
        or len(graph.outcome_estimates) != 1
        or len(graph.cohorts) != 1
        or len(graph.contrasts) != 1
    ):
        raise PostLiveContextualJoinV1Error(
            "postlive_join_requires_single_publication_single_estimate_smoke_graph"
        )
    estimate = graph.outcome_estimates[0]
    synthesis = synthesize_evidence_graph(
        graph,
        outcome_name=estimate.outcome_name,
        contrast_id=estimate.contrast_id,
        require_explicit_timepoint=True,
        confidence_level=0.95,
        assumed_within_cohort_correlation=1.0,
        prespecified_moderators=moderators,
    )
    if not moderators and synthesis != projection.quantitative_mechanics_result:
        raise PostLiveContextualJoinV1Error("postlive_projection_synthesis_replay_mismatch")
    condition = _condition_mechanics(moderators=moderators, synthesis=synthesis)
    integration_pipeline_sha256 = hash_canonical(
        {
            "integration_version": JOIN_VERSION,
            "runtime_pipeline_sha256": terminal.runtime_pipeline_sha256,
            "target_direction": target_direction,
            "prespecified_moderators": moderators,
            "synthesis_contract": {
                "require_explicit_timepoint": True,
                "confidence_level": 0.95,
                "assumed_within_cohort_correlation": 1.0,
            },
            "audit_policy": AUDIT_POLICY.value,
            "audit_probability_source": UNIT_RISK_SOURCE,
            "audit_cost_unit": UNIT_COST_UNIT,
        }
    )
    audit = _audit_mechanics(
        created_at=created,
        terminal=terminal,
        integration_pipeline_sha256=integration_pipeline_sha256,
        graph=graph,
        outcome_name=estimate.outcome_name,
        contrast_id=estimate.contrast_id,
        target_direction=target_direction,
        moderators=moderators,
        synthesis=synthesis,
    )
    blockers = sorted(
        {
            *projection.blockers,
            "complete_corpus_not_available",
            "human_adjudication_absent",
            "human_verification_cost_unmeasured",
            "item_error_probability_not_calibrated",
            "release_risk_bound_unavailable",
            "same_graph_condition_analysis_not_confirmatory",
        }
    )
    assert success.contextual_grounding_core_sha256 is not None
    assert success.runtime_grounding_binding_sha256 is not None
    payload = {
        "certificate_version": CERTIFICATE_VERSION,
        "generated_at": _datetime_json(created),
        "status": "mechanics_completed_non_authorizing",
        "target_direction": target_direction,
        "runtime_workspace_validation_sha256": runtime_workspace_validation_sha256,
        "terminal_report_sha256": terminal.report_sha256,
        "terminal_status": terminal.status,
        "successful_validation_sha256": success.validation_sha256,
        "provider_receipt_sha256": success.provider_receipt_sha256,
        "provider_result_sha256": success.provider_result_sha256,
        "provider_execution_binding_sha256": (success.provider_execution_binding_sha256),
        "contextual_grounding_core_sha256": success.contextual_grounding_core_sha256,
        "runtime_grounding_binding_sha256": success.runtime_grounding_binding_sha256,
        "native_projection_sha256": projection.projection_sha256,
        "fragment_sha256": projection.fragment.fragment_sha256,
        "runtime_pipeline_sha256": terminal.runtime_pipeline_sha256,
        "integration_pipeline_sha256": integration_pipeline_sha256,
        "evidence_graph": graph,
        "evidence_graph_sha256": hash_canonical(graph),
        "outcome_name": estimate.outcome_name,
        "contrast_id": estimate.contrast_id,
        "synthesis": synthesis,
        "synthesis_sha256": hash_canonical(synthesis),
        "condition_mechanics": condition,
        "audit_mechanics": audit,
        "blockers": blockers,
        "title_abstract_only": True,
        "single_publication": True,
        "complete_corpus": False,
        "provider_call_success_is_not_accuracy_evidence": True,
        "graph_construction_mechanics_authority": True,
        "extraction_accuracy_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "condition_claim_authority": False,
        "adaptive_effectiveness_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
        "release_authorizing": False,
    }
    return PostLiveContextualCertificateV1.model_validate(
        {**payload, "certificate_sha256": hash_canonical(payload)}
    )


def build_postlive_contextual_certificate_from_workspace_v1(
    *,
    repository_root: Path,
    runtime_workspace: Path,
    generated_at: datetime,
    target_direction: Literal["increase", "decrease"],
    prespecified_moderators: Sequence[str] = (),
    external_replay: bool = True,
) -> PostLiveContextualCertificateV1:
    """Full-workspace replay followed by the non-authorizing mechanics join."""

    # Imported lazily because the runtime and this additive join are developed as
    # separate new-only artifacts.  The public name is frozen by the runtime contract.
    from literature_multiverse.metasyn_contextual_frontier_runtime_v1 import (
        validate_metasyn_contextual_frontier_runtime_v1,
    )

    replay = validate_metasyn_contextual_frontier_runtime_v1(
        repository_root=repository_root,
        workspace=runtime_workspace,
        external_replay=external_replay,
    )
    if replay.status != "terminal" or replay.terminal_report is None:
        raise PostLiveContextualJoinV1Error("postlive_runtime_workspace_has_no_terminal_report")
    if not replay.external_plan_and_source_replayed:
        raise PostLiveContextualJoinV1Error("postlive_runtime_external_replay_required")
    return freeze_postlive_contextual_certificate_v1(
        terminal_report=replay.terminal_report,
        runtime_workspace_validation_sha256=replay.workspace_validation_sha256,
        generated_at=generated_at,
        target_direction=target_direction,
        prespecified_moderators=prespecified_moderators,
    )


def validate_postlive_contextual_certificate_v1(
    *,
    certificate: PostLiveContextualCertificateV1 | Mapping[str, Any],
    repository_root: Path,
    runtime_workspace: Path,
    external_replay: bool = True,
) -> PostLiveContextualCertificateV1:
    """Rebuild the certificate from the runtime workspace and reject any drift."""

    raw = (
        certificate.model_dump(mode="json")
        if isinstance(certificate, PostLiveContextualCertificateV1)
        else certificate
    )
    try:
        observed = PostLiveContextualCertificateV1.model_validate(raw)
    except ValueError as exc:
        raise PostLiveContextualJoinV1Error(
            "postlive_certificate_contract_or_hash_invalid"
        ) from exc
    expected = build_postlive_contextual_certificate_from_workspace_v1(
        repository_root=repository_root,
        runtime_workspace=runtime_workspace,
        generated_at=observed.generated_at,
        target_direction=observed.target_direction,
        prespecified_moderators=observed.condition_mechanics.prespecified_moderators,
        external_replay=external_replay,
    )
    if observed != expected:
        raise PostLiveContextualJoinV1Error("postlive_certificate_external_replay_mismatch")
    return observed


__all__ = [
    "CERTIFICATE_VERSION",
    "JOIN_VERSION",
    "PostLiveAuditMechanicsV1",
    "PostLiveConditionMechanicsV1",
    "PostLiveContextualCertificateV1",
    "PostLiveContextualJoinV1Error",
    "build_postlive_audit_mechanics_from_source_identity_v1",
    "build_postlive_condition_mechanics_v1",
    "build_postlive_contextual_certificate_from_workspace_v1",
    "freeze_postlive_contextual_certificate_v1",
    "validate_postlive_contextual_certificate_v1",
]
