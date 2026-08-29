"""Join an offline recovery-v4 repair artifact to non-authorizing mechanics.

This adapter deliberately does not accept a recovery runtime workspace or a v1
terminal report.  Its only upstream input is a self-hashed post-hoc artifact whose
immutable lineage records the failed-closed recovery-v4 terminal and whose native
projection is rebound to a distinct offline source-repair pipeline.

The result is evidence-graph, synthesis, condition, and counterfactual-audit
*mechanics*.  Post-hoc repair is an explicit blocker.  No field in this artifact
licenses extraction accuracy, scientific synthesis, calibration, or claim release.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.evidence_graph import EvidenceGraph
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.meta_analysis import synthesize_evidence_graph
from literature_multiverse.metasyn_contextual_frontier_recovery_v4 import (
    MetaSynContextualFrontierRecoveryCoreEvaluationV4,
)
from literature_multiverse.metasyn_contextual_frontier_recovery_v4_posthoc_v1 import (
    MetaSynContextualFrontierRecoveryV4PosthocArtifactV1,
    freeze_metasyn_contextual_frontier_recovery_v4_posthoc_artifact_v1,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.postlive_contextual_join_v1 import (
    AUDIT_POLICY,
    UNIT_COST_UNIT,
    UNIT_RISK_SOURCE,
    PostLiveAuditMechanicsV1,
    PostLiveConditionMechanicsV1,
    build_postlive_audit_mechanics_from_source_identity_v1,
    build_postlive_condition_mechanics_v1,
)

JOIN_VERSION = "postlive-recovery-v4-join-v1"
SOURCE_ARTIFACT_VERSION = (
    "metasyn-contextual-frontier-recovery-v4-posthoc-artifact-v1"
)
SOURCE_RECOVERY_LABEL = "post_hoc_syntactic_canonicalization"
SOURCE_STATUS = "typed_graph_mechanics_completed"
SOURCE_V4_TERMINAL_STATUS = "contextual_validation_failed_closed"

EXPECTED_V4_PLAN_SHA256 = (
    "5b504b4f7bad1742ec6a773289141835715f05bb63454ab3e61026f25bd012c8"
)
EXPECTED_V4_PLAN_FILE_SHA256 = (
    "936ac25e5ffdead2b5246126c2ff38632e8999adb5b45850e17753e94e707b86"
)
EXPECTED_V4_TERMINAL_SHA256 = (
    "9c1bd812915afd5527585c389559f6716f868c02bbd8be1faf360cf8de158986"
)
EXPECTED_V4_TERMINAL_FILE_SHA256 = (
    "e9a87352f85b9aafa826abdc59ccc575dfe818f1adf5040d94b1111a01d47d3f"
)
EXPECTED_V4_RECEIPT_SHA256 = (
    "8ebe84f61a2136e79e06fb3fc2dd6b02462b5213df892fd0830840c571405bef"
)
EXPECTED_V4_RECEIPT_FILE_SHA256 = (
    "c6d6b80dbf3dc83843fd86f1cd6199b5461c4a2aa4e001a191e32da77b6dd0e8"
)
EXPECTED_PROVIDER_RESULT_SHA256 = (
    "4cf8d252b609c528e6638c5cbf57431639f2331cfee366cafa4ed6a17e148165"
)
EXPECTED_ORIGINAL_RESPONSE_SHA256 = (
    "ee3229492abcc677e540d91522d6b0536e37ab2e06962a11c898ef50e6ff48a9"
)

_SOURCE_FALSE_AUTHORITIES = (
    "graph_construction_mechanics_authority",
    "extraction_accuracy_authority",
    "reliability_authority",
    "generalization_authority",
    "synthesis_input_authority",
    "scientific_synthesis_authority",
    "scientific_effectiveness_authority",
    "calibration_authority",
    "claim_release_authority",
)
_SOURCE_TRUE_INVARIANTS = (
    "raw_schema_validated_before_repair",
    "exact_roster_and_order_validated_before_repair",
    "no_target_or_gold_token_injection",
    "immutable_transform_ledger",
    "endpoint_quote_ascii_provider_to_exact_u2009_source",
    "endpoint_source_match_unique",
    "endpoint_marker_only_passage_id_change",
    "grounding_uniqueness_and_offsets_recomputed",
    "binary_pair_structure_and_arm_source_spans_verified",
    "hidden_numeric_equality_not_consulted_by_repair",
    "nested_projection_graph_mechanics_flag_dependency_only",
    "field_set_unchanged",
    "tokens_unchanged",
    "normalizations_unchanged",
    "numeric_values_unchanged",
    "arm_outcome_semantics_unchanged",
    "source_span_only_changes_validated",
    "endpoint_quote_unicode_whitespace_equivalent",
)
_REQUIRED_SOURCE_BLOCKERS = {
    "post_hoc_source_span_repair",
}
_REQUIRED_JOIN_BLOCKERS = {
    "calibration_not_performed",
    "complete_corpus_not_available",
    "extraction_accuracy_not_evaluated",
    "human_adjudication_absent",
    "human_verification_cost_unmeasured",
    "item_error_probability_not_calibrated",
    "post_hoc_source_span_repair",
    "post_hoc_syntactic_canonicalization",
    "release_risk_bound_unavailable",
    "same_graph_condition_analysis_not_confirmatory",
    "single_publication_mechanics_only",
    "source_v4_terminal_failed_closed",
    "title_or_abstract_only_not_release_grade",
}


class PostLiveRecoveryV4JoinV1Error(ValueError):
    """The repaired source or non-authorizing join contract failed closed."""


def _sha256(value: str, field_name: str) -> str:
    if SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"postlive_recovery_v4_sha256_invalid:{field_name}")
    return value


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"postlive_recovery_v4_timezone_required:{field_name}")
    return value


def _datetime_json(value: datetime) -> str:
    rendered = _aware(value, "datetime_json").isoformat()
    return f"{rendered[:-6]}Z" if rendered.endswith("+00:00") else rendered


def _self_hash(model: ContractModel, field_name: str) -> None:
    if hash_canonical(model.model_dump(mode="json", exclude={field_name})) != getattr(
        model, field_name
    ):
        raise ValueError(f"postlive_recovery_v4_self_hash_mismatch:{field_name}")


def _canonical_moderators(values: Sequence[str]) -> list[str]:
    moderators = [value.strip() for value in values]
    if any(not value for value in moderators) or moderators != sorted(set(moderators)):
        raise PostLiveRecoveryV4JoinV1Error(
            "postlive_recovery_v4_moderators_must_be_sorted_unique"
        )
    return moderators


class PostLiveSourceRepairChangeV1(ContractModel):
    """One immutable entry from the upstream source-span repair ledger."""

    json_pointer: str
    change_kind: Literal[
        "minimal_local_context",
        "unicode_whitespace_exact_source_quote",
        "endpoint_marker_passage_binding",
        "endpoint_marker_quote_binding",
    ]
    before_sha256: str
    after_sha256: str

    @field_validator("before_sha256", "after_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)


@dataclass(frozen=True)
class _PosthocSourceView:
    artifact_sha256: str
    artifact_file_sha256: str
    canonicalizer_provider_calls_made: int
    upstream_v4_provider_attempt_count: int
    upstream_v4_provider_response_completed: bool
    plan_sha256: str
    plan_file_sha256: str
    terminal_sha256: str
    terminal_file_sha256: str
    receipt_sha256: str
    receipt_file_sha256: str
    provider_result_sha256: str
    original_response_sha256: str
    canonicalized_response_sha256: str
    provider_execution_binding_sha256: str
    canonicalizer_source_sha256: str
    canonicalization_pipeline_sha256: str
    changes: tuple[PostLiveSourceRepairChangeV1, ...]
    change_membership_sha256: str
    blockers: tuple[str, ...]
    evaluation: MetaSynContextualFrontierRecoveryCoreEvaluationV4


class PostLiveRecoveryV4JoinArtifactV1(ContractModel):
    """A repaired-source mechanics join with no scientific or release authority."""

    join_version: Literal["postlive-recovery-v4-join-v1"] = JOIN_VERSION
    generated_at: datetime
    status: Literal["composed_offline_mechanics_completed_non_authorizing"] = (
        "composed_offline_mechanics_completed_non_authorizing"
    )
    target_direction: Literal["increase", "decrease"]
    source_artifact_version: Literal[
        "metasyn-contextual-frontier-recovery-v4-posthoc-artifact-v1"
    ] = SOURCE_ARTIFACT_VERSION
    source_recovery_label: Literal["post_hoc_syntactic_canonicalization"] = (
        SOURCE_RECOVERY_LABEL
    )
    source_status: Literal["typed_graph_mechanics_completed"] = SOURCE_STATUS
    source_v4_terminal_status: Literal["contextual_validation_failed_closed"] = (
        SOURCE_V4_TERMINAL_STATUS
    )
    source_v4_runtime_workspace_success: Literal[False] = False
    source_offline_only: Literal[True] = True
    canonicalizer_provider_calls_made: Literal[0] = 0
    upstream_v4_provider_attempt_count: Literal[1] = 1
    upstream_v4_provider_response_completed: Literal[True] = True
    source_posthoc_external_replay_performed: bool
    source_posthoc_external_replay_sha256: str | None
    source_posthoc_artifact_sha256: str
    source_posthoc_artifact_file_sha256: str
    immutable_v4_plan_sha256: str
    immutable_v4_plan_file_sha256: str
    immutable_v4_terminal_sha256: str
    immutable_v4_terminal_file_sha256: str
    immutable_v4_receipt_sha256: str
    immutable_v4_receipt_file_sha256: str
    provider_result_sha256: str
    original_response_sha256: str
    canonicalized_response_sha256: str
    provider_execution_binding_sha256: str
    canonicalizer_source_sha256: str
    canonicalization_pipeline_sha256: str
    source_repair_changes: list[PostLiveSourceRepairChangeV1]
    source_repair_change_membership_sha256: str
    source_evaluation_sha256: str
    contextual_grounding_core_sha256: str
    runtime_grounding_binding_sha256: str
    native_projection_sha256: str
    fragment_sha256: str
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
    post_hoc_syntactic_canonicalization: Literal[True] = True
    post_hoc_source_span_repair: Literal[True] = True
    typed_graph_mechanics_observed: Literal[True] = True
    title_abstract_only: Literal[True] = True
    single_publication: Literal[True] = True
    complete_corpus: Literal[False] = False
    provider_call_success_is_not_accuracy_evidence: Literal[True] = True
    composed_offline_pipeline_mechanics_only: Literal[True] = True
    graph_construction_mechanics_authority: Literal[False] = False
    extraction_accuracy_authority: Literal[False] = False
    reliability_authority: Literal[False] = False
    generalization_authority: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    scientific_synthesis_authority: Literal[False] = False
    scientific_effectiveness_authority: Literal[False] = False
    condition_claim_authority: Literal[False] = False
    adaptive_effectiveness_authority: Literal[False] = False
    calibration_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    release_authorizing: Literal[False] = False
    artifact_sha256: str

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: datetime) -> datetime:
        return _aware(value, "generated_at")

    @field_validator(
        "source_posthoc_artifact_sha256",
        "source_posthoc_artifact_file_sha256",
        "immutable_v4_plan_sha256",
        "immutable_v4_plan_file_sha256",
        "immutable_v4_terminal_sha256",
        "immutable_v4_terminal_file_sha256",
        "immutable_v4_receipt_sha256",
        "immutable_v4_receipt_file_sha256",
        "provider_result_sha256",
        "original_response_sha256",
        "canonicalized_response_sha256",
        "provider_execution_binding_sha256",
        "canonicalizer_source_sha256",
        "canonicalization_pipeline_sha256",
        "source_repair_change_membership_sha256",
        "source_evaluation_sha256",
        "contextual_grounding_core_sha256",
        "runtime_grounding_binding_sha256",
        "native_projection_sha256",
        "fragment_sha256",
        "integration_pipeline_sha256",
        "evidence_graph_sha256",
        "synthesis_sha256",
        "artifact_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("source_posthoc_external_replay_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None, info: Any) -> str | None:
        return _sha256(value, info.field_name) if value is not None else None

    @field_validator("blockers")
    @classmethod
    def validate_blockers(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not item for item in value):
            raise ValueError("postlive_recovery_v4_blockers_not_canonical")
        return value

    @field_validator("source_repair_changes")
    @classmethod
    def validate_repair_ledger(
        cls, value: list[PostLiveSourceRepairChangeV1]
    ) -> list[PostLiveSourceRepairChangeV1]:
        pointers = [item.json_pointer for item in value]
        if not value or pointers != sorted(set(pointers)):
            raise ValueError("postlive_recovery_v4_repair_ledger_not_canonical")
        return value

    @model_validator(mode="after")
    def validate_artifact(self) -> PostLiveRecoveryV4JoinArtifactV1:
        if not set(self.blockers) >= _REQUIRED_JOIN_BLOCKERS:
            raise ValueError("postlive_recovery_v4_required_blocker_missing")
        if self.source_posthoc_external_replay_performed != (
            self.source_posthoc_external_replay_sha256 is not None
        ):
            raise ValueError("postlive_recovery_v4_external_replay_alias_mismatch")
        if (
            self.source_repair_change_membership_sha256
            != hash_canonical(
                [item.model_dump(mode="json") for item in self.source_repair_changes]
            )
            or len(self.evidence_graph.publications) != 1
            or len(self.evidence_graph.cohorts) != 1
            or len(self.evidence_graph.contrasts) != 1
            or len(self.evidence_graph.outcome_estimates) != 1
            or self.evidence_graph_sha256 != hash_canonical(self.evidence_graph)
            or self.synthesis_sha256 != hash_canonical(self.synthesis)
            or self.audit_mechanics.sequential_state.graph_sha256
            != self.evidence_graph_sha256
            or self.audit_mechanics.sequential_state.synthesis_sha256
            != self.synthesis_sha256
        ):
            raise ValueError("postlive_recovery_v4_nested_artifact_mismatch")
        _self_hash(self, "artifact_sha256")
        return self


def _source_string(source: Mapping[str, Any], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str):
        raise PostLiveRecoveryV4JoinV1Error(f"postlive_recovery_v4_source_missing:{key}")
    return value


def _source_view(
    *, posthoc_artifact: Mapping[str, Any], artifact_file_sha256: str
) -> _PosthocSourceView:
    """Validate only the narrow, immutable source interface needed by the join."""

    _sha256(artifact_file_sha256, "source_posthoc_artifact_file_sha256")
    artifact_sha256 = _source_string(posthoc_artifact, "artifact_sha256")
    if artifact_sha256 != hash_canonical(
        {key: value for key, value in posthoc_artifact.items() if key != "artifact_sha256"}
    ):
        raise PostLiveRecoveryV4JoinV1Error("postlive_recovery_v4_source_self_hash_mismatch")
    expected_literals: dict[str, Any] = {
        "artifact_version": SOURCE_ARTIFACT_VERSION,
        "recovery_label": SOURCE_RECOVERY_LABEL,
        "status": SOURCE_STATUS,
        "offline_only": True,
        "canonicalizer_provider_calls_made": 0,
        "upstream_v4_provider_attempt_count": 1,
        "upstream_v4_provider_response_completed": True,
    }
    if any(posthoc_artifact.get(key) != value for key, value in expected_literals.items()):
        raise PostLiveRecoveryV4JoinV1Error("postlive_recovery_v4_source_contract_mismatch")
    if any(posthoc_artifact.get(key) is not False for key in _SOURCE_FALSE_AUTHORITIES):
        raise PostLiveRecoveryV4JoinV1Error("postlive_recovery_v4_source_authority_not_false")
    if any(posthoc_artifact.get(key) is not True for key in _SOURCE_TRUE_INVARIANTS):
        raise PostLiveRecoveryV4JoinV1Error("postlive_recovery_v4_source_invariant_not_true")

    pinned = {
        "immutable_v4_plan_sha256": EXPECTED_V4_PLAN_SHA256,
        "immutable_v4_plan_file_sha256": EXPECTED_V4_PLAN_FILE_SHA256,
        "immutable_v4_terminal_sha256": EXPECTED_V4_TERMINAL_SHA256,
        "immutable_v4_terminal_file_sha256": EXPECTED_V4_TERMINAL_FILE_SHA256,
        "immutable_v4_receipt_sha256": EXPECTED_V4_RECEIPT_SHA256,
        "immutable_v4_receipt_file_sha256": EXPECTED_V4_RECEIPT_FILE_SHA256,
        "provider_result_sha256": EXPECTED_PROVIDER_RESULT_SHA256,
        "original_response_sha256": EXPECTED_ORIGINAL_RESPONSE_SHA256,
    }
    if any(posthoc_artifact.get(key) != value for key, value in pinned.items()):
        raise PostLiveRecoveryV4JoinV1Error("postlive_recovery_v4_immutable_lineage_mismatch")

    raw_changes = posthoc_artifact.get("canonicalization_changes")
    raw_blockers = posthoc_artifact.get("blockers")
    if not isinstance(raw_changes, list) or not raw_changes:
        raise PostLiveRecoveryV4JoinV1Error("postlive_recovery_v4_source_repair_ledger_missing")
    if (
        not isinstance(raw_blockers, list)
        or raw_blockers != sorted(set(raw_blockers))
        or not set(raw_blockers) >= _REQUIRED_SOURCE_BLOCKERS
    ):
        raise PostLiveRecoveryV4JoinV1Error("postlive_recovery_v4_source_repair_blocker_missing")
    try:
        changes = tuple(PostLiveSourceRepairChangeV1.model_validate(item) for item in raw_changes)
        evaluation = MetaSynContextualFrontierRecoveryCoreEvaluationV4.model_validate(
            posthoc_artifact.get("evaluation")
        )
    except ValueError as exc:
        raise PostLiveRecoveryV4JoinV1Error(
            "postlive_recovery_v4_source_nested_contract_invalid"
        ) from exc

    change_membership_sha256 = _source_string(
        posthoc_artifact, "canonicalization_change_membership_sha256"
    )
    canonicalized_response = posthoc_artifact.get("canonicalized_response")
    canonicalized_response_sha256 = _source_string(
        posthoc_artifact, "canonicalized_response_sha256"
    )
    evaluation_sha256 = _source_string(posthoc_artifact, "evaluation_sha256")
    provider_execution_binding_sha256 = _source_string(
        posthoc_artifact, "provider_execution_binding_sha256"
    )
    canonicalization_pipeline_sha256 = _source_string(
        posthoc_artifact, "canonicalization_pipeline_sha256"
    )
    if (
        change_membership_sha256
        != hash_canonical([item.model_dump(mode="json") for item in changes])
        or canonicalized_response_sha256 != hash_canonical(canonicalized_response)
        or evaluation_sha256 != evaluation.evaluation_sha256
        or evaluation.response.model_dump(mode="json") != canonicalized_response
        or evaluation.response_sha256 != canonicalized_response_sha256
        or evaluation.plan_sha256 != EXPECTED_V4_PLAN_SHA256
        or evaluation.provider_execution_binding_sha256
        != provider_execution_binding_sha256
        or evaluation.runtime_pipeline_sha256 != canonicalization_pipeline_sha256
        or evaluation.native_projection.runtime_pipeline_sha256
        != canonicalization_pipeline_sha256
        or evaluation.native_projection.status != "typed_graph_mechanics_completed"
        or evaluation.native_projection.fragment is None
        or evaluation.native_projection.fragment.graph is None
    ):
        raise PostLiveRecoveryV4JoinV1Error("postlive_recovery_v4_source_replay_mismatch")
    return _PosthocSourceView(
        artifact_sha256=artifact_sha256,
        artifact_file_sha256=artifact_file_sha256,
        canonicalizer_provider_calls_made=0,
        upstream_v4_provider_attempt_count=1,
        upstream_v4_provider_response_completed=True,
        plan_sha256=EXPECTED_V4_PLAN_SHA256,
        plan_file_sha256=EXPECTED_V4_PLAN_FILE_SHA256,
        terminal_sha256=EXPECTED_V4_TERMINAL_SHA256,
        terminal_file_sha256=EXPECTED_V4_TERMINAL_FILE_SHA256,
        receipt_sha256=EXPECTED_V4_RECEIPT_SHA256,
        receipt_file_sha256=EXPECTED_V4_RECEIPT_FILE_SHA256,
        provider_result_sha256=EXPECTED_PROVIDER_RESULT_SHA256,
        original_response_sha256=EXPECTED_ORIGINAL_RESPONSE_SHA256,
        canonicalized_response_sha256=canonicalized_response_sha256,
        provider_execution_binding_sha256=provider_execution_binding_sha256,
        canonicalizer_source_sha256=_source_string(
            posthoc_artifact, "canonicalizer_source_sha256"
        ),
        canonicalization_pipeline_sha256=canonicalization_pipeline_sha256,
        changes=changes,
        change_membership_sha256=change_membership_sha256,
        blockers=tuple(raw_blockers),
        evaluation=evaluation,
    )


def freeze_postlive_recovery_v4_join_artifact_v1(
    *,
    posthoc_artifact: Mapping[str, Any],
    posthoc_artifact_file_sha256: str,
    generated_at: datetime,
    target_direction: Literal["increase", "decrease"],
    prespecified_moderators: Sequence[str] = (),
    source_posthoc_external_replay_sha256: str | None = None,
) -> PostLiveRecoveryV4JoinArtifactV1:
    """Compose the actual repaired native projection with verifier mechanics."""

    created = _aware(generated_at, "generated_at")
    moderators = _canonical_moderators(prespecified_moderators)
    if source_posthoc_external_replay_sha256 is not None:
        _sha256(
            source_posthoc_external_replay_sha256,
            "source_posthoc_external_replay_sha256",
        )
    source = _source_view(
        posthoc_artifact=posthoc_artifact,
        artifact_file_sha256=posthoc_artifact_file_sha256,
    )
    projection = source.evaluation.native_projection
    if (
        projection.outcome_origin != "runtime_outcome_supplied_by_caller"
        or projection.fragment is None
        or projection.fragment.graph is None
        or projection.contextual_grounding_core_sha256 is None
        or projection.runtime_grounding_binding_sha256 is None
        or not projection.title_abstract_only_not_release_grade
    ):
        raise PostLiveRecoveryV4JoinV1Error(
            "postlive_recovery_v4_projection_not_eligible_for_mechanics_join"
        )
    graph = EvidenceGraph.model_validate(projection.fragment.graph.model_dump(mode="json"))
    if (
        len(graph.publications) != 1
        or len(graph.cohorts) != 1
        or len(graph.contrasts) != 1
        or len(graph.outcome_estimates) != 1
    ):
        raise PostLiveRecoveryV4JoinV1Error(
            "postlive_recovery_v4_requires_single_publication_single_estimate_graph"
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
        raise PostLiveRecoveryV4JoinV1Error(
            "postlive_recovery_v4_projection_synthesis_replay_mismatch"
        )
    condition = build_postlive_condition_mechanics_v1(
        moderators=moderators,
        synthesis=synthesis,
    )
    integration_pipeline_sha256 = hash_canonical(
        {
            "integration_version": JOIN_VERSION,
            "source_posthoc_artifact_sha256": source.artifact_sha256,
            "source_canonicalization_pipeline_sha256": (
                source.canonicalization_pipeline_sha256
            ),
            "source_repair_change_membership_sha256": source.change_membership_sha256,
            "source_evaluation_sha256": source.evaluation.evaluation_sha256,
            "source_native_projection_sha256": projection.projection_sha256,
            "source_posthoc_external_replay_sha256": (
                source_posthoc_external_replay_sha256
            ),
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
            "scientific_and_release_authority": False,
        }
    )
    audit = build_postlive_audit_mechanics_from_source_identity_v1(
        created_at=created,
        source_identity_sha256=source.artifact_sha256,
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
            *source.blockers,
            "complete_corpus_not_available",
            "human_adjudication_absent",
            "human_verification_cost_unmeasured",
            "item_error_probability_not_calibrated",
            "post_hoc_source_span_repair",
            "post_hoc_syntactic_canonicalization",
            "release_risk_bound_unavailable",
            "same_graph_condition_analysis_not_confirmatory",
            "source_v4_terminal_failed_closed",
            *(
                []
                if source_posthoc_external_replay_sha256 is not None
                else ["source_posthoc_external_replay_not_performed"]
            ),
        }
    )
    payload = {
        "join_version": JOIN_VERSION,
        "generated_at": _datetime_json(created),
        "status": "composed_offline_mechanics_completed_non_authorizing",
        "target_direction": target_direction,
        "source_artifact_version": SOURCE_ARTIFACT_VERSION,
        "source_recovery_label": SOURCE_RECOVERY_LABEL,
        "source_status": SOURCE_STATUS,
        "source_v4_terminal_status": SOURCE_V4_TERMINAL_STATUS,
        "source_v4_runtime_workspace_success": False,
        "source_offline_only": True,
        "canonicalizer_provider_calls_made": source.canonicalizer_provider_calls_made,
        "upstream_v4_provider_attempt_count": source.upstream_v4_provider_attempt_count,
        "upstream_v4_provider_response_completed": (
            source.upstream_v4_provider_response_completed
        ),
        "source_posthoc_external_replay_performed": (
            source_posthoc_external_replay_sha256 is not None
        ),
        "source_posthoc_external_replay_sha256": (
            source_posthoc_external_replay_sha256
        ),
        "source_posthoc_artifact_sha256": source.artifact_sha256,
        "source_posthoc_artifact_file_sha256": source.artifact_file_sha256,
        "immutable_v4_plan_sha256": source.plan_sha256,
        "immutable_v4_plan_file_sha256": source.plan_file_sha256,
        "immutable_v4_terminal_sha256": source.terminal_sha256,
        "immutable_v4_terminal_file_sha256": source.terminal_file_sha256,
        "immutable_v4_receipt_sha256": source.receipt_sha256,
        "immutable_v4_receipt_file_sha256": source.receipt_file_sha256,
        "provider_result_sha256": source.provider_result_sha256,
        "original_response_sha256": source.original_response_sha256,
        "canonicalized_response_sha256": source.canonicalized_response_sha256,
        "provider_execution_binding_sha256": source.provider_execution_binding_sha256,
        "canonicalizer_source_sha256": source.canonicalizer_source_sha256,
        "canonicalization_pipeline_sha256": source.canonicalization_pipeline_sha256,
        "source_repair_changes": list(source.changes),
        "source_repair_change_membership_sha256": source.change_membership_sha256,
        "source_evaluation_sha256": source.evaluation.evaluation_sha256,
        "contextual_grounding_core_sha256": (
            source.evaluation.contextual_grounding_core_sha256
        ),
        "runtime_grounding_binding_sha256": projection.runtime_grounding_binding_sha256,
        "native_projection_sha256": projection.projection_sha256,
        "fragment_sha256": projection.fragment.fragment_sha256,
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
        "post_hoc_syntactic_canonicalization": True,
        "post_hoc_source_span_repair": True,
        "typed_graph_mechanics_observed": True,
        "title_abstract_only": True,
        "single_publication": True,
        "complete_corpus": False,
        "provider_call_success_is_not_accuracy_evidence": True,
        "composed_offline_pipeline_mechanics_only": True,
        "graph_construction_mechanics_authority": False,
        "extraction_accuracy_authority": False,
        "reliability_authority": False,
        "generalization_authority": False,
        "synthesis_input_authority": False,
        "scientific_synthesis_authority": False,
        "scientific_effectiveness_authority": False,
        "condition_claim_authority": False,
        "adaptive_effectiveness_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
        "release_authorizing": False,
    }
    return PostLiveRecoveryV4JoinArtifactV1.model_validate(
        {**payload, "artifact_sha256": hash_canonical(payload)}
    )


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PostLiveRecoveryV4JoinV1Error("postlive_recovery_v4_source_file_unsafe")
    resolved = path.resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostLiveRecoveryV4JoinV1Error(
            "postlive_recovery_v4_source_file_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise PostLiveRecoveryV4JoinV1Error("postlive_recovery_v4_source_not_object")
    return value


def build_postlive_recovery_v4_join_from_artifact_v1(
    *,
    repository_root: Path,
    posthoc_artifact_path: Path,
    generated_at: datetime,
    target_direction: Literal["increase", "decrease"],
    prespecified_moderators: Sequence[str] = (),
    immutable_v4_workspace: Path | None = None,
) -> PostLiveRecoveryV4JoinArtifactV1:
    """Externally replay one frozen post-hoc artifact, then build the join."""

    raw = _read_object(posthoc_artifact_path)
    resolved = posthoc_artifact_path.resolve(strict=True)
    try:
        observed = MetaSynContextualFrontierRecoveryV4PosthocArtifactV1.model_validate(raw)
        expected = freeze_metasyn_contextual_frontier_recovery_v4_posthoc_artifact_v1(
            repository_root=repository_root,
            immutable_workspace=immutable_v4_workspace,
        )
    except ValueError as exc:
        raise PostLiveRecoveryV4JoinV1Error(
            "postlive_recovery_v4_posthoc_external_replay_failed"
        ) from exc
    if observed != expected:
        raise PostLiveRecoveryV4JoinV1Error(
            "postlive_recovery_v4_posthoc_external_replay_mismatch"
        )
    artifact_file_sha256 = sha256_file(resolved)
    external_replay_sha256 = hash_canonical(
        {
            "replay_version": "postlive-recovery-v4-posthoc-external-replay-v1",
            "source_posthoc_artifact_sha256": observed.artifact_sha256,
            "source_posthoc_artifact_file_sha256": artifact_file_sha256,
            "canonicalizer_source_sha256": observed.canonicalizer_source_sha256,
            "canonicalization_pipeline_sha256": observed.canonicalization_pipeline_sha256,
            "immutable_v4_plan_sha256": observed.immutable_v4_plan_sha256,
            "immutable_v4_terminal_sha256": observed.immutable_v4_terminal_sha256,
            "immutable_v4_receipt_sha256": observed.immutable_v4_receipt_sha256,
            "exact_source_artifact_equality": True,
        }
    )
    return freeze_postlive_recovery_v4_join_artifact_v1(
        posthoc_artifact=observed.model_dump(mode="json"),
        posthoc_artifact_file_sha256=artifact_file_sha256,
        generated_at=generated_at,
        target_direction=target_direction,
        prespecified_moderators=prespecified_moderators,
        source_posthoc_external_replay_sha256=external_replay_sha256,
    )


def validate_postlive_recovery_v4_join_artifact_v1(
    *,
    artifact: PostLiveRecoveryV4JoinArtifactV1 | Mapping[str, Any],
    repository_root: Path,
    posthoc_artifact_path: Path,
    immutable_v4_workspace: Path | None = None,
) -> PostLiveRecoveryV4JoinArtifactV1:
    """Rebuild from the repaired source artifact and reject any join drift."""

    raw = (
        artifact.model_dump(mode="json")
        if isinstance(artifact, PostLiveRecoveryV4JoinArtifactV1)
        else artifact
    )
    try:
        observed = PostLiveRecoveryV4JoinArtifactV1.model_validate(raw)
    except ValueError as exc:
        raise PostLiveRecoveryV4JoinV1Error(
            "postlive_recovery_v4_join_contract_or_hash_invalid"
        ) from exc
    expected = build_postlive_recovery_v4_join_from_artifact_v1(
        repository_root=repository_root,
        posthoc_artifact_path=posthoc_artifact_path,
        generated_at=observed.generated_at,
        target_direction=observed.target_direction,
        prespecified_moderators=observed.condition_mechanics.prespecified_moderators,
        immutable_v4_workspace=immutable_v4_workspace,
    )
    if observed != expected:
        raise PostLiveRecoveryV4JoinV1Error("postlive_recovery_v4_join_replay_mismatch")
    return observed


__all__ = [
    "JOIN_VERSION",
    "PostLiveRecoveryV4JoinArtifactV1",
    "PostLiveRecoveryV4JoinV1Error",
    "PostLiveSourceRepairChangeV1",
    "build_postlive_recovery_v4_join_from_artifact_v1",
    "freeze_postlive_recovery_v4_join_artifact_v1",
    "validate_postlive_recovery_v4_join_artifact_v1",
]
