from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import ValidationError

from literature_multiverse.condition_confirmation import (
    BOOTSTRAP_PROTOCOL,
    SPLIT_SALT,
    ConditionConfirmationAssessmentV1,
    ConditionConfirmationError,
    ConditionConfirmationFrozenModelV1,
    ConditionConfirmationMaterializationReceiptV1,
    ConditionConfirmationPlanV1,
    LabelFreeGraphRosterV1,
    RosterArmV1,
    RosterCohortV1,
    RosterContrastV1,
    RosterEstimateV1,
    RosterPublicationV1,
    RosterSpanV1,
    RosterStudyV1,
    confirm_condition_dependence,
    derive_condition_components,
    fit_condition_confirmation_model,
    freeze_condition_confirmation_config,
    freeze_condition_confirmation_target,
    freeze_label_free_graph_roster,
    materialize_condition_confirmation_inputs,
    prepare_condition_confirmation_plan,
    validate_condition_confirmation_assessment,
    validate_condition_confirmation_materialization,
    validate_condition_confirmation_model,
)
from literature_multiverse.effects import EffectEvidence, HarmonizedMeasure
from literature_multiverse.evidence_graph import (
    CohortIdentity,
    EvidenceGraph,
    GraphAdapterContext,
    PublicationIdentity,
    adapt_effect_evidence,
)
from literature_multiverse.independence_identity import (
    authority_identity_set_sha256,
    parse_canonical_authority_identity,
)
from literature_multiverse.lineage import hash_canonical

PIPELINE_SHA256 = "a" * 64
CLAIM_SHA256 = "b" * 64
QUESTION_CONFIG_SHA256 = "c" * 64
CORPUS_SNAPSHOT_SHA256 = "d" * 64
EXTERNAL_ANCHOR = "git:0123456789abcdef@2026-08-28T12:00:00Z"


@dataclass(frozen=True)
class ConfirmationCase:
    graph: EvidenceGraph
    roster: LabelFreeGraphRosterV1
    plan: ConditionConfirmationPlanV1
    development_graph: EvidenceGraph
    confirmation_graph: EvidenceGraph
    materialization_receipt: ConditionConfirmationMaterializationReceiptV1
    model: ConditionConfirmationFrozenModelV1


def _single_graph(index: int, *, level: str, estimate: float) -> EvidenceGraph:
    suffix = f"cc-{index:03d}"
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
            "source_quote": f"The standardized effect estimate was {estimate}.",
        },
    )
    context = GraphAdapterContext(
        publication=PublicationIdentity(
            publication_id=f"publication-{suffix}",
            paper_id=f"paper-{suffix}",
            doc_id=f"document-{suffix}",
            doi=f"10.9999/{suffix}",
            pmid=str(9_000_000 + index),
            title=f"Synthetic source {index}",
        ),
        study_id=f"study-{suffix}",
        cohort_identity=CohortIdentity(
            cohort_id=f"cohort-{suffix}",
            basis="reviewer_reconciled",
            source_labels=[f"source cohort {index}"],
            rationale="Synthetic independent cohort identity for contract tests.",
        ),
        treatment_arm_id=f"arm-{suffix}-treatment",
        comparator_arm_id=f"arm-{suffix}-control",
        contrast_id=f"contrast-{suffix}",
        contrast_label="intervention_vs_control",
        positive_direction_means="higher performance under intervention",
        treatment_label="intervention",
        comparator_label="control",
    )
    payload = adapt_effect_evidence(evidence, context=context).graph.model_dump(mode="json")
    payload["contrasts"][0]["estimand"] = (
        "between-group standardized difference in performance"
    )
    payload["studies"][0]["registration_ids"] = [f"NCT{index:08d}"]
    return EvidenceGraph.model_validate(payload)


def _merge_graphs(graphs: list[EvidenceGraph]) -> EvidenceGraph:
    return EvidenceGraph(
        publications=[row for graph in graphs for row in graph.publications],
        studies=[row for graph in graphs for row in graph.studies],
        cohorts=[row for graph in graphs for row in graph.cohorts],
        arms=[row for graph in graphs for row in graph.arms],
        contrasts=[row for graph in graphs for row in graph.contrasts],
        outcome_estimates=[row for graph in graphs for row in graph.outcome_estimates],
        evidence_spans=[row for graph in graphs for row in graph.evidence_spans],
    )


def _target():
    return freeze_condition_confirmation_target(
        question_id="synthetic-condition-confirmation",
        claim_spec_sha256=CLAIM_SHA256,
        question_config_sha256=QUESTION_CONFIG_SHA256,
        corpus_snapshot_sha256=CORPUS_SNAPSHOT_SHA256,
        corpus_cutoff="2026-08-01T00:00:00Z",
        outcome_name="performance",
        contrast_label="intervention_vs_control",
        contrast_estimand="between-group standardized difference in performance",
        positive_direction_means="higher performance under intervention",
        treatment_role="intervention",
        comparator_role="comparator",
        measure=HarmonizedMeasure.STANDARDIZED_MEAN_DIFFERENCE,
        moderator_names=["dose"],
    )


def _roster(graph: EvidenceGraph) -> LabelFreeGraphRosterV1:
    publication_by_paper = {row.paper_id: row for row in graph.publications}
    cohort_by_id = {row.cohort_id: row for row in graph.cohorts}
    contrast_by_id = {row.contrast_id: row for row in graph.contrasts}
    return freeze_label_free_graph_roster(
        source_graph_sha256=hash_canonical(graph),
        publications=[
            RosterPublicationV1(
                publication_id=row.publication_id,
                paper_id=row.paper_id,
                doi=row.doi,
                pmid=row.pmid,
                doc_id=row.doc_id,
            )
            for row in graph.publications
        ],
        studies=[
            RosterStudyV1(
                study_id=row.study_id,
                publication_ids=row.publication_ids,
                registration_ids=row.registration_ids,
            )
            for row in graph.studies
        ],
        cohorts=[
            RosterCohortV1(
                cohort_id=row.cohort_id,
                study_id=row.study_id,
                identity_basis=row.identity.basis,
                registry_ids=row.identity.registry_ids,
                dataset_ids=row.identity.dataset_ids,
            )
            for row in graph.cohorts
        ],
        arms=[
            RosterArmV1(
                arm_id=row.arm_id,
                cohort_id=row.cohort_id,
                role=row.role,
            )
            for row in graph.arms
        ],
        contrasts=[
            RosterContrastV1(
                contrast_id=row.contrast_id,
                cohort_id=row.cohort_id,
                treatment_arm_id=row.treatment_arm_id,
                comparator_arm_id=row.comparator_arm_id,
                label=row.label,
                estimand=row.estimand,
                positive_direction_means=row.positive_direction_means,
            )
            for row in graph.contrasts
        ],
        estimates=[
            RosterEstimateV1(
                estimate_id=row.estimate_id,
                publication_id=publication_by_paper[row.effect.paper_id].publication_id,
                study_id=cohort_by_id[
                    contrast_by_id[row.contrast_id].cohort_id
                ].study_id,
                cohort_id=contrast_by_id[row.contrast_id].cohort_id,
                contrast_id=row.contrast_id,
                target_scope=(
                    row.outcome_name == "performance"
                    and contrast_by_id[row.contrast_id].label
                    == "intervention_vs_control"
                ),
                moderator_values={"dose": row.effect.moderators.get("dose")},
            )
            for row in graph.outcome_estimates
        ],
        spans=[
            RosterSpanV1(span_id=row.span_id, publication_id=row.publication_id)
            for row in graph.evidence_spans
        ],
    )


def _case_from_graph(graph: EvidenceGraph) -> ConfirmationCase:
    target = _target()
    config = freeze_condition_confirmation_config()
    roster, development_graph, confirmation_graph, materialization_receipt = (
        materialize_condition_confirmation_inputs(
            full_graph=graph,
            target=target,
        )
    )
    plan = prepare_condition_confirmation_plan(
        target=target,
        config=config,
        roster=roster,
        materialization_receipt=materialization_receipt,
        pipeline_sha256=PIPELINE_SHA256,
        external_freeze_anchor=EXTERNAL_ANCHOR,
    )
    model = fit_condition_confirmation_model(
        plan,
        development_graph,
        current_pipeline_sha256=PIPELINE_SHA256,
    )
    return ConfirmationCase(
        graph=graph,
        roster=roster,
        plan=plan,
        development_graph=development_graph,
        confirmation_graph=confirmation_graph,
        materialization_receipt=materialization_receipt,
        model=model,
    )


@pytest.fixture(scope="module")
def confirmation_case() -> ConfirmationCase:
    graph = _merge_graphs(
        [
            _single_graph(
                index,
                level="high" if index % 2 == 0 else "low",
                estimate=0.8 if index % 2 == 0 else -0.8,
            )
            for index in range(150)
        ]
    )
    case = _case_from_graph(graph)
    assert case.plan.status == "ready"
    assert case.model.status == "fitted"
    return case


def _rehash(payload: dict[str, Any], hash_field: str) -> dict[str, Any]:
    payload[hash_field] = hash_canonical(
        {key: value for key, value in payload.items() if key != hash_field}
    )
    return payload


def test_prepare_roster_is_strictly_outcome_blind() -> None:
    schema = LabelFreeGraphRosterV1.model_json_schema()
    estimate_properties = schema["$defs"]["RosterEstimateV1"]["properties"]
    for forbidden in (
        "expected_status",
        "outcome",
        "outcome_name",
        "effect",
        "effect_format",
        "availability",
        "estimate",
        "standard_error",
        "variance",
        "sampling_variance",
        "confidence_interval",
        "p_value",
        "reported_significance",
    ):
        assert forbidden not in estimate_properties

    graph = _single_graph(999, level="high", estimate=0.8)
    roster = _roster(graph)
    forbidden_values = {
        "expected_status": "compatible_quantitative",
        "effect_format": "hedges_g",
        "availability": "available",
        "estimate": 0.8,
        "standard_error": 0.1,
        "variance": 0.01,
        "reported_significance": "significant",
    }
    for field, value in forbidden_values.items():
        payload = roster.model_dump(mode="json")
        payload["estimates"][0][field] = value
        payload = _rehash(payload, "roster_sha256")
        with pytest.raises(ValidationError, match="extra_forbidden"):
            LabelFreeGraphRosterV1.model_validate(payload)


def test_custodian_materialization_receipt_is_content_silent_and_exactly_replays(
    confirmation_case: ConfirmationCase,
) -> None:
    case = confirmation_case
    receipt_payload = case.materialization_receipt.model_dump(mode="json")
    assert receipt_payload["effect_outcome_uncertainty_values_embedded"] is False
    assert receipt_payload["full_graph_outcomes_opened_by_custodian"] is True
    forbidden_keys = {
        "effect",
        "estimate",
        "standard_error",
        "variance",
        "sampling_variance",
        "confidence_interval",
        "p_value",
        "moderator_values",
        "source_quote",
        "source_locator",
    }
    assert forbidden_keys.isdisjoint(receipt_payload)
    replay = validate_condition_confirmation_materialization(
        full_graph=case.graph,
        target=case.plan.target,
        roster=case.roster,
        development_graph=case.development_graph,
        confirmation_graph=case.confirmation_graph,
        receipt=case.materialization_receipt,
    )
    assert replay == case.materialization_receipt

    tampered = case.materialization_receipt.model_dump(mode="json")
    tampered["development_graph_sha256"] = "9" * 64
    tampered = _rehash(tampered, "receipt_sha256")
    forged = ConditionConfirmationMaterializationReceiptV1.model_validate(tampered)
    with pytest.raises(
        ConditionConfirmationError,
        match="materialization_recomputation_mismatch",
    ):
        validate_condition_confirmation_materialization(
            full_graph=case.graph,
            target=case.plan.target,
            roster=case.roster,
            development_graph=case.development_graph,
            confirmation_graph=case.confirmation_graph,
            receipt=forged,
        )


def test_component_split_uses_exact_strong_identity_and_hash_protocol() -> None:
    graph = _merge_graphs(
        [
            _single_graph(1001, level="high", estimate=0.8),
            _single_graph(1002, level="low", estimate=-0.8),
        ]
    )
    roster = _roster(graph)
    payload = roster.model_dump(mode="json")
    payload["studies"][0]["registration_ids"] = ["NCT87654321"]
    payload["studies"][1]["registration_ids"] = ["nct87654321"]
    payload = _rehash(payload, "roster_sha256")
    joined = LabelFreeGraphRosterV1.model_validate(payload)

    assignments = derive_condition_components(
        joined,
        question_id=_target().question_id,
    )

    assert len(assignments) == 1
    assignment = assignments[0]
    assert (
        "join-only-v1:registry:nct87654321" in assignment.strong_identity_tokens
    )
    expected_component = hash_canonical(
        {
            "publication_ids": assignment.publication_ids,
            "paper_ids": assignment.paper_ids,
            "study_ids": assignment.study_ids,
            "cohort_ids": assignment.cohort_ids,
            "strong_identity_tokens": assignment.strong_identity_tokens,
        }
    )
    assert assignment.component_id == expected_component
    expected_split_identity = authority_identity_set_sha256(
        assignment.split_identity_tokens
    )
    assert assignment.split_identity_sha256 == expected_split_identity
    expected_assignment = hashlib.sha256(
        SPLIT_SALT.encode()
        + b"\0"
        + _target().question_id.encode()
        + b"\0"
        + assignment.split_identity_sha256.encode()
    ).hexdigest()
    assert assignment.assignment_sha256 == expected_assignment
    expected_split = (
        "confirmation"
        if int.from_bytes(bytes.fromhex(expected_assignment)[:8], "big") % 3 == 0
        else "development"
    )
    assert assignment.split == expected_split
    assert any(
        parse_canonical_authority_identity(token).kind == "trial_registry"
        for token in assignment.split_identity_tokens
    )


def test_split_assignment_is_invariant_to_graph_local_identity_renaming() -> None:
    roster = _roster(_single_graph(1100, level="high", estimate=0.8))
    original = derive_condition_components(
        roster,
        question_id=_target().question_id,
    )[0]
    payload = roster.model_dump(mode="json")
    old_publication_id = payload["publications"][0]["publication_id"]
    new_publication_id = "publication-renamed-after-freeze-attempt"
    payload["publications"][0]["publication_id"] = new_publication_id
    payload["publications"][0]["paper_id"] = "paper-renamed-after-freeze-attempt"
    payload["publications"][0]["doc_id"] = "document-renamed-after-freeze-attempt"
    payload["studies"][0]["publication_ids"] = [new_publication_id]
    payload["estimates"][0]["publication_id"] = new_publication_id
    payload["spans"][0]["publication_id"] = new_publication_id
    assert old_publication_id != new_publication_id
    renamed_roster = LabelFreeGraphRosterV1.model_validate(
        _rehash(payload, "roster_sha256")
    )
    renamed = derive_condition_components(
        renamed_roster,
        question_id=_target().question_id,
    )[0]
    assert renamed.component_id != original.component_id
    assert renamed.split_identity_tokens == original.split_identity_tokens
    assert renamed.split_identity_sha256 == original.split_identity_sha256
    assert renamed.assignment_sha256 == original.assignment_sha256
    assert renamed.split == original.split


def test_plan_recomputes_authority_split_closure_instead_of_trusting_digests(
    confirmation_case: ConfirmationCase,
) -> None:
    payload = confirmation_case.plan.model_dump(mode="json")
    first = payload["component_assignments"][0]
    second = payload["component_assignments"][1]
    first["split_identity_tokens"] = second["split_identity_tokens"]
    first["split_identity_sha256"] = authority_identity_set_sha256(
        first["split_identity_tokens"]
    )
    raw = (
        SPLIT_SALT.encode()
        + b"\0"
        + confirmation_case.plan.target.question_id.encode()
        + b"\0"
        + first["split_identity_sha256"].encode()
    )
    first["assignment_sha256"] = hashlib.sha256(raw).hexdigest()
    first["split"] = (
        "confirmation"
        if int.from_bytes(bytes.fromhex(first["assignment_sha256"])[:8], "big") % 3
        == 0
        else "development"
    )
    payload = _rehash(payload, "plan_sha256")
    with pytest.raises(ValidationError, match="plan_component_assignment_mismatch"):
        ConditionConfirmationPlanV1.model_validate(payload)


def test_target_component_without_authority_identity_is_release_ineligible(
    confirmation_case: ConfirmationCase,
) -> None:
    payload = confirmation_case.graph.model_dump(mode="json")
    payload["publications"][0]["doi"] = None
    payload["publications"][0]["pmid"] = None
    graph = EvidenceGraph.model_validate(payload)
    roster, _, _, receipt = materialize_condition_confirmation_inputs(
        full_graph=graph,
        target=confirmation_case.plan.target,
    )
    plan = prepare_condition_confirmation_plan(
        target=confirmation_case.plan.target,
        config=confirmation_case.plan.config,
        roster=roster,
        materialization_receipt=receipt,
        pipeline_sha256=PIPELINE_SHA256,
        external_freeze_anchor=EXTERNAL_ANCHOR,
    )
    assert plan.status == "insufficient"
    assert any(
        "target_component_publication_lacks_authority_identity" in reason
        for reason in plan.insufficiency_reasons
    )


def test_authority_aliases_union_components_before_split() -> None:
    graph = _merge_graphs(
        [
            _single_graph(1201, level="high", estimate=0.8),
            _single_graph(1202, level="high", estimate=0.7),
        ]
    )
    payload = graph.model_dump(mode="json")
    payload["studies"][0]["registration_ids"] = ["NCT87650001"]
    payload["studies"][1]["registration_ids"] = [
        "clinicaltrials.gov:NCT87650001"
    ]
    roster = _roster(EvidenceGraph.model_validate(payload))
    assignments = derive_condition_components(
        roster,
        question_id=_target().question_id,
    )
    assert len(assignments) == 1
    assignment = assignments[0]
    trial_tokens = [
        token
        for token in assignment.split_identity_tokens
        if parse_canonical_authority_identity(token).kind == "trial_registry"
    ]
    assert len(trial_tokens) == 1
    assert assignment.authority_identity_conflict_sha256s == []


def test_conflicting_authority_aliases_make_target_component_ineligible(
    confirmation_case: ConfirmationCase,
) -> None:
    payload = confirmation_case.graph.model_dump(mode="json")
    payload["studies"][0]["registration_ids"] = [
        "clinicaltrials.gov:NCT87650002"
    ]
    payload["studies"][1]["registration_ids"] = [
        "conflicting-registry.example:NCT87650002"
    ]
    case = _case_from_graph(EvidenceGraph.model_validate(payload))
    assert case.plan.status == "insufficient"
    assert any(
        "target_component_authority_identity_conflict" in reason
        for reason in case.plan.insufficiency_reasons
    )


def test_join_only_multi_report_linkage_cannot_certify_independence(
    confirmation_case: ConfirmationCase,
) -> None:
    payload = confirmation_case.graph.model_dump(mode="json")
    for study in payload["studies"][:2]:
        study["registration_ids"] = sorted(
            [*study["registration_ids"], "unscoped-shared-report-link"]
        )
    case = _case_from_graph(EvidenceGraph.model_validate(payload))
    assert case.plan.status == "insufficient"
    assert any(
        "target_component_lacks_all_report_authority_linkage" in reason
        for reason in case.plan.insufficiency_reasons
    )


def test_target_component_without_cohort_level_authority_linkage_is_ineligible(
    confirmation_case: ConfirmationCase,
) -> None:
    payload = confirmation_case.graph.model_dump(mode="json")
    payload["studies"][0]["registration_ids"] = []
    case = _case_from_graph(EvidenceGraph.model_validate(payload))
    assert case.plan.status == "insufficient"
    assert any(
        "target_component_lacks_all_report_authority_linkage" in reason
        for reason in case.plan.insufficiency_reasons
    )


def test_plan_model_and_all_cross_bindings_are_self_hashed(
    confirmation_case: ConfirmationCase,
) -> None:
    case = confirmation_case
    assert case.plan.freeze_state == "confirmation_outcomes_unopened"
    assert case.plan.claim_spec_sha256 == CLAIM_SHA256
    assert case.plan.question_config_sha256 == QUESTION_CONFIG_SHA256
    assert case.plan.corpus_snapshot_sha256 == CORPUS_SNAPSHOT_SHA256
    assert case.plan.external_freeze_anchor == EXTERNAL_ANCHOR
    assert case.model.freeze_state == "confirmation_outcomes_unopened"
    assert case.model.selected_moderator == "dose"
    assert case.model.frozen_positive_level == "high"
    assert case.model.frozen_negative_level == "low"
    assert "confirmation" not in inspect.signature(
        fit_condition_confirmation_model
    ).parameters
    assert "test" not in inspect.signature(fit_condition_confirmation_model).parameters

    validated = validate_condition_confirmation_model(
        plan=case.plan,
        development_graph=case.development_graph,
        model=case.model,
        current_pipeline_sha256=PIPELINE_SHA256,
    )
    assert validated == case.model


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("claim_spec_sha256", "1" * 64),
        ("question_config_sha256", "2" * 64),
        ("corpus_snapshot_sha256", "3" * 64),
        ("corpus_cutoff", "2025-01-01T00:00:00Z"),
    ],
)
def test_claim_config_corpus_and_cutoff_crossbinding_tamper_is_rejected(
    confirmation_case: ConfirmationCase,
    field: str,
    replacement: str,
) -> None:
    payload = confirmation_case.plan.model_dump(mode="json")
    payload["target"][field] = replacement
    payload["target"] = _rehash(payload["target"], "target_sha256")
    payload["target_sha256"] = payload["target"]["target_sha256"]
    payload = _rehash(payload, "plan_sha256")
    with pytest.raises(
        ValidationError,
        match="plan_claim_corpus_binding_mismatch",
    ):
        ConditionConfirmationPlanV1.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "replacement", "expected"),
    [
        (
            "development_graph_sha256",
            "8" * 64,
            "materialization_graph_hash_mismatch",
        ),
        ("target_sha256", "7" * 64, "materialization_target_roster_mismatch"),
    ],
)
def test_semantically_rehashed_materialization_receipt_tamper_is_rejected_by_plan(
    confirmation_case: ConfirmationCase,
    field: str,
    replacement: str,
    expected: str,
) -> None:
    payload = confirmation_case.plan.model_dump(mode="json")
    payload["materialization_receipt"][field] = replacement
    payload["materialization_receipt"] = _rehash(
        payload["materialization_receipt"],
        "receipt_sha256",
    )
    payload["materialization_receipt_sha256"] = payload[
        "materialization_receipt"
    ]["receipt_sha256"]
    payload = _rehash(payload, "plan_sha256")
    with pytest.raises(ValidationError, match=expected):
        ConditionConfirmationPlanV1.model_validate(payload)


def test_ambiguous_or_mismatched_contrast_label_mapping_fails_plan_closed(
    confirmation_case: ConfirmationCase,
) -> None:
    case = confirmation_case
    payload = case.graph.model_dump(mode="json")
    for contrast in payload["contrasts"]:
        contrast["label"] = "different_prespecified_contrast"
    for estimate in payload["outcome_estimates"]:
        estimate["effect"]["contrast"] = "different_prespecified_contrast"
    graph = EvidenceGraph.model_validate(payload)
    roster, _, _, receipt = materialize_condition_confirmation_inputs(
        full_graph=graph,
        target=case.plan.target,
    )
    plan = prepare_condition_confirmation_plan(
        target=case.plan.target,
        config=case.plan.config,
        roster=roster,
        materialization_receipt=receipt,
        pipeline_sha256=PIPELINE_SHA256,
        external_freeze_anchor=EXTERNAL_ANCHOR,
    )
    assert plan.status == "insufficient"
    assert any(
        "contrast_label_mapping_not_exactly_one" in reason
        for reason in plan.insufficiency_reasons
    )


@pytest.mark.parametrize(
    ("field", "expected_reason"),
    [
        ("estimand", "target_contrast_estimand_mapping_not_exactly_one"),
        (
            "positive_direction_means",
            "target_positive_direction_mapping_not_exactly_one",
        ),
        ("treatment_role", "target_treatment_role_mapping_not_exactly_one"),
        ("comparator_role", "target_comparator_role_mapping_not_exactly_one"),
    ],
)
def test_estimand_direction_and_arm_orientation_drift_fail_plan_closed(
    confirmation_case: ConfirmationCase,
    field: str,
    expected_reason: str,
) -> None:
    case = confirmation_case
    payload = case.graph.model_dump(mode="json")
    contrast = payload["contrasts"][0]
    if field == "estimand":
        contrast["estimand"] = "a different estimand"
    elif field == "positive_direction_means":
        contrast["positive_direction_means"] = "the opposite scientific direction"
    else:
        arm_field = (
            "treatment_arm_id" if field == "treatment_role" else "comparator_arm_id"
        )
        arm_id = contrast[arm_field]
        next(row for row in payload["arms"] if row["arm_id"] == arm_id)["role"] = (
            "other"
        )
    graph = EvidenceGraph.model_validate(payload)
    roster, _, _, receipt = materialize_condition_confirmation_inputs(
        full_graph=graph,
        target=case.plan.target,
    )
    plan = prepare_condition_confirmation_plan(
        target=case.plan.target,
        config=case.plan.config,
        roster=roster,
        materialization_receipt=receipt,
        pipeline_sha256=PIPELINE_SHA256,
        external_freeze_anchor=EXTERNAL_ANCHOR,
    )
    assert plan.status == "insufficient"
    assert any(expected_reason in reason for reason in plan.insufficiency_reasons)


def test_development_uses_one_conservative_unit_per_independence_component() -> None:
    payload = _merge_graphs(
        [
            _single_graph(
                index + 3000,
                level="high" if index % 2 == 0 else "low",
                estimate=0.8 if index % 2 == 0 else -0.8,
            )
            for index in range(180)
        ]
    ).model_dump(mode="json")
    for index, study in enumerate(payload["studies"]):
        level = "high" if index % 2 == 0 else "low"
        base = 91_000_000 if level == "high" else 91_100_000
        study["registration_ids"] = [f"NCT{base + index // 4:08d}"]
    graph = EvidenceGraph.model_validate(payload)
    case = _case_from_graph(graph)
    assert case.plan.status == "ready"
    assert case.model.status == "fitted"
    assert case.model.unconditional is not None
    development_components = len(case.plan.development_partition.component_ids)
    assert development_components < len(case.plan.development_partition.cohort_ids)
    assert (
        case.model.unconditional.development_component_count
        == development_components
    )
    assert case.model.conditional is not None
    assert sum(
        row.development_component_count
        for row in case.model.conditional.level_parameters
    ) == development_components


def test_connected_duplicate_cohorts_cannot_inflate_development_support() -> None:
    payload = _merge_graphs(
        [
            _single_graph(
                index + 4000,
                level="high" if index % 2 == 0 else "low",
                estimate=0.8 if index % 2 == 0 else -0.8,
            )
            for index in range(150)
        ]
    ).model_dump(mode="json")
    target_study_ids = {
        row["study_id"] for row in payload["cohorts"][:90]
    }
    for index, study in enumerate(payload["studies"]):
        if study["study_id"] in target_study_ids:
            study["registration_ids"] = [
                "NCT92000001" if index % 2 == 0 else "NCT92000002"
            ]
    for estimate in payload["outcome_estimates"][90:]:
        estimate["outcome_name"] = "out_of_scope_performance"
        estimate["effect"]["outcome"] = "out_of_scope_performance"
    graph = EvidenceGraph.model_validate(payload)
    roster, _, _, receipt = materialize_condition_confirmation_inputs(
        full_graph=graph,
        target=_target(),
    )
    plan = prepare_condition_confirmation_plan(
        target=_target(),
        config=freeze_condition_confirmation_config(),
        roster=roster,
        materialization_receipt=receipt,
        pipeline_sha256=PIPELINE_SHA256,
        external_freeze_anchor=EXTERNAL_ANCHOR,
    )
    target_estimate_ids = {
        row.estimate_id for row in roster.estimates if row.target_scope
    }
    target_components = [
        assignment
        for assignment in plan.component_assignments
        if target_estimate_ids.intersection(assignment.estimate_ids)
    ]
    assert len(target_estimate_ids) == 90
    assert len(target_components) == 2
    assert plan.status == "insufficient"
    assert any(
        "development_component_level_sparse" in reason
        or "development_moderator_has_fewer_than_two_levels" in reason
        for reason in plan.insufficiency_reasons
    )


def test_conflicting_levels_within_one_component_fail_before_development_fit(
    confirmation_case: ConfirmationCase,
) -> None:
    payload = confirmation_case.graph.model_dump(mode="json")
    payload["studies"][0]["registration_ids"] = ["NCT93000001"]
    payload["studies"][1]["registration_ids"] = ["nct93000001"]
    case = _case_from_graph(EvidenceGraph.model_validate(payload))
    assert case.plan.status == "insufficient"
    assert case.model.status == "insufficient"
    assert any(
        "moderator_conflict_within_independence_component" in reason
        for reason in case.plan.insufficiency_reasons
    )


def test_heldout_confirmation_passes_all_three_gates_and_replays_exactly(
    confirmation_case: ConfirmationCase,
) -> None:
    case = confirmation_case
    assessment = confirm_condition_dependence(
        plan=case.plan,
        model=case.model,
        full_graph=case.graph,
        current_pipeline_sha256=PIPELINE_SHA256,
    )

    assert assessment.status == "confirmed"
    assert assessment.reasons == []
    assert assessment.overlap_checks.passed is True
    assert assessment.brier_comparison is not None
    assert assessment.brier_comparison.passed is True
    assert assessment.brier_comparison.bootstrap_replicates == 10_000
    seed = int.from_bytes(
        hashlib.sha256(
            case.plan.plan_sha256.encode()
            + b"\0"
            + case.model.model_sha256.encode()
            + b"\0"
            + BOOTSTRAP_PROTOCOL.encode()
        ).digest()[:8],
        "big",
    )
    assert assessment.brier_comparison.seed == seed
    assert assessment.polarity_replication is not None
    assert assessment.polarity_replication.passed is True
    assert assessment.polarity_replication.positive.confidence_level == 0.975
    assert assessment.confirmation_omnibus_passed is True
    assert assessment.claim_spec_sha256 == CLAIM_SHA256

    replay = validate_condition_confirmation_assessment(
        plan=case.plan,
        model=case.model,
        full_graph=case.graph,
        assessment=assessment,
        current_pipeline_sha256=PIPELINE_SHA256,
    )
    assert replay == assessment


def test_semantically_rehashed_model_and_assessment_tampering_is_rejected(
    confirmation_case: ConfirmationCase,
) -> None:
    case = confirmation_case
    model_payload = case.model.model_dump(mode="json")
    model_payload["development_effect_input_sha256"] = "f" * 64
    model_payload = _rehash(model_payload, "model_sha256")
    forged_model = ConditionConfirmationFrozenModelV1.model_validate(model_payload)
    with pytest.raises(
        ConditionConfirmationError,
        match="model_recomputation_mismatch",
    ):
        validate_condition_confirmation_model(
            plan=case.plan,
            development_graph=case.development_graph,
            model=forged_model,
            current_pipeline_sha256=PIPELINE_SHA256,
        )

    assessment = confirm_condition_dependence(
        plan=case.plan,
        model=case.model,
        full_graph=case.graph,
        current_pipeline_sha256=PIPELINE_SHA256,
    )
    assessment_payload = assessment.model_dump(mode="json")
    assessment_payload["predictions"][0]["estimate"] = 0.7
    assessment_payload = _rehash(assessment_payload, "assessment_sha256")
    forged_assessment = ConditionConfirmationAssessmentV1.model_validate(
        assessment_payload
    )
    with pytest.raises(
        ConditionConfirmationError,
        match="assessment_recomputation_mismatch",
    ):
        validate_condition_confirmation_assessment(
            plan=case.plan,
            model=case.model,
            full_graph=case.graph,
            assessment=forged_assessment,
            current_pipeline_sha256=PIPELINE_SHA256,
        )


def test_pipeline_and_graph_hash_mismatch_fail_before_scientific_replay(
    confirmation_case: ConfirmationCase,
) -> None:
    case = confirmation_case
    with pytest.raises(ConditionConfirmationError, match="pipeline_hash_mismatch"):
        fit_condition_confirmation_model(
            case.plan,
            case.development_graph,
            current_pipeline_sha256="e" * 64,
        )
    with pytest.raises(ConditionConfirmationError, match="pipeline_hash_mismatch"):
        confirm_condition_dependence(
            plan=case.plan,
            model=case.model,
            full_graph=case.graph,
            current_pipeline_sha256="e" * 64,
        )

    graph_payload = case.graph.model_dump(mode="json")
    graph_payload["outcome_estimates"][0]["effect"]["estimate"] = 0.75
    tampered_graph = EvidenceGraph.model_validate(graph_payload)
    with pytest.raises(ConditionConfirmationError, match="full_graph_hash_mismatch"):
        confirm_condition_dependence(
            plan=case.plan,
            model=case.model,
            full_graph=tampered_graph,
            current_pipeline_sha256=PIPELINE_SHA256,
        )


@pytest.mark.parametrize("failure_mode", ["non_estimable", "directional_only"])
def test_confirmation_effects_that_are_not_strictly_quantitative_fail_closed(
    confirmation_case: ConfirmationCase,
    failure_mode: str,
) -> None:
    base = confirmation_case
    confirmation_estimate_id = base.plan.confirmation_partition.estimate_ids[0]
    graph_payload = base.graph.model_dump(mode="json")
    row = next(
        value
        for value in graph_payload["outcome_estimates"]
        if value["estimate_id"] == confirmation_estimate_id
    )
    row["effect"]["standard_error"] = None
    if failure_mode == "non_estimable":
        row["effect"]["availability"] = "missing"
        row["effect"]["estimate"] = None
    graph = EvidenceGraph.model_validate(graph_payload)
    case = _case_from_graph(graph)
    assert case.plan.status == "ready"
    assert case.model.status == "fitted"

    assessment = confirm_condition_dependence(
        plan=case.plan,
        model=case.model,
        full_graph=case.graph,
        current_pipeline_sha256=PIPELINE_SHA256,
    )

    assert assessment.status == "insufficient"
    assert any(failure_mode in reason for reason in assessment.reasons)


def test_exact_zero_and_unseen_confirmation_level_fail_closed(
    confirmation_case: ConfirmationCase,
) -> None:
    base = confirmation_case
    confirmation_estimate_id = base.plan.confirmation_partition.estimate_ids[0]
    for field, value, expected_reason in (
        ("estimate", 0.0, "exact_zero_sign_ambiguous"),
        ("moderator", "unseen", "moderator_level_unseen"),
    ):
        graph_payload = base.graph.model_dump(mode="json")
        row = next(
            item
            for item in graph_payload["outcome_estimates"]
            if item["estimate_id"] == confirmation_estimate_id
        )
        if field == "estimate":
            row["effect"]["estimate"] = value
        else:
            row["effect"]["moderators"]["dose"] = value
        graph = EvidenceGraph.model_validate(graph_payload)
        case = _case_from_graph(graph)
        assert case.model.status == "fitted"
        assessment = confirm_condition_dependence(
            plan=case.plan,
            model=case.model,
            full_graph=case.graph,
            current_pipeline_sha256=PIPELINE_SHA256,
        )
        assert assessment.status == "insufficient"
        assert any(expected_reason in reason for reason in assessment.reasons)


def test_legacy_target_identity_and_sparse_split_are_insufficient() -> None:
    graph = _merge_graphs(
        [
            _single_graph(
                index + 2000,
                level="high" if index % 2 == 0 else "low",
                estimate=0.8 if index % 2 == 0 else -0.8,
            )
            for index in range(12)
        ]
    )
    payload = graph.model_dump(mode="json")
    payload["cohorts"][0]["identity"]["basis"] = "legacy_placeholder"
    graph = EvidenceGraph.model_validate(payload)
    roster, _, _, receipt = materialize_condition_confirmation_inputs(
        full_graph=graph,
        target=_target(),
    )
    plan = prepare_condition_confirmation_plan(
        target=_target(),
        config=freeze_condition_confirmation_config(),
        roster=roster,
        materialization_receipt=receipt,
        pipeline_sha256=PIPELINE_SHA256,
        external_freeze_anchor=EXTERNAL_ANCHOR,
    )
    assert plan.status == "insufficient"
    assert any("legacy_placeholder" in reason for reason in plan.insufficiency_reasons)
    assert any("below_minimum" in reason for reason in plan.insufficiency_reasons)


def test_full_partition_retains_out_of_scope_nodes_without_counting_them_as_test_support(
    confirmation_case: ConfirmationCase,
) -> None:
    base = confirmation_case
    payload = base.roster.model_dump(mode="json")
    row = payload["estimates"][0]
    row["target_scope"] = False
    row["moderator_values"] = {}
    payload = _rehash(payload, "roster_sha256")
    roster = LabelFreeGraphRosterV1.model_validate(payload)
    assignments = derive_condition_components(roster, question_id=_target().question_id)
    assigned_estimates = {
        estimate_id for assignment in assignments for estimate_id in assignment.estimate_ids
    }
    assert row["estimate_id"] in assigned_estimates
