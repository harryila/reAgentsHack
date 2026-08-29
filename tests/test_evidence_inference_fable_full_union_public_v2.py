from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest
import scripts.run_evidence_inference_fable_full_union_reuse_v2 as harness

from literature_multiverse.evidence_inference_fable_full_union_reuse_v2 import (
    EvidenceInferenceFableFullUnionPlanV2,
    EvidenceInferenceFableFullUnionReuseError,
    EvidenceInferenceFableFullUnionTerminalV2,
    EvidenceInferenceFableUnionEntryV2,
    EvidenceInferenceFableUnionSourceBindingV2,
    freeze_evidence_inference_fable_full_union_failure_burden_v2,
    freeze_evidence_inference_fable_full_union_scoring_lineage_v2,
    materialize_evidence_inference_fable_full_union_public_evaluation_v2,
    project_evidence_inference_fable_full_union_public_evaluation_v2,
)
from literature_multiverse.evidence_inference_fable_paired_runtime_v1 import (
    INCIDENT_SANITIZATION_POLICY,
    EvidenceInferenceFableIncidentV2,
    EvidenceInferenceFableProviderResultV1,
    EvidenceInferenceFableReceiptV1,
    EvidenceInferenceFableTerminalV1,
)
from literature_multiverse.evidence_inference_fable_retrospective_inference_v1 import (
    MetricBootstrapEstimateV1,
    PairedArticleClusterBootstrapV1,
)
from literature_multiverse.evidence_inference_fable_retrospective_scoring_v1 import (
    ArmAggregateV1,
    PublicPairedSummaryV1,
)
from literature_multiverse.evidence_inference_fable_retrospective_v1 import (
    EvidenceInferenceFableRetrospectivePlanV1,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical

ROOT = Path(__file__).resolve().parents[1]
FULL_PLAN = ROOT / "artifacts/diagnostics/evidence-inference/fable-retrospective-full-plan-v1.json"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _full_plan() -> EvidenceInferenceFableRetrospectivePlanV1:
    return EvidenceInferenceFableRetrospectivePlanV1.model_validate(
        json.loads(FULL_PLAN.read_text(encoding="utf-8"))
    )


def _ambiguity_indexes(plan: EvidenceInferenceFableRetrospectivePlanV1) -> tuple[int, int]:
    winners = [
        (index, request.question_count)
        for index, request in enumerate(plan.roster)
        if request.arm == "winner"
    ]
    return next(
        (first[0], second[0])
        for first, second in combinations(winners, 2)
        if first[1] + second[1] == 16
    )


def _union_plan(
    full: EvidenceInferenceFableRetrospectivePlanV1,
) -> EvidenceInferenceFableFullUnionPlanV2:
    ambiguous = set(_ambiguity_indexes(full))
    terminal_indexes = [index for index in range(len(full.roster)) if index not in ambiguous][:22]
    selected = sorted([*terminal_indexes, *ambiguous])
    entries = []
    nested_receipts = set(terminal_indexes[:8])
    for index in selected:
        request = full.roster[index]
        is_ambiguous = index in ambiguous
        payload = {
            "entry_version": "evidence-inference-fable-full-union-entry-v2",
            "adoption_kind": (
                "inherited_ambiguous_failure" if is_ambiguous else "terminal_receipt"
            ),
            "target_execution_index": index,
            "target_request_key": request.request_key,
            "target_surface_sha256": _sha(f"target-surface-{index}"),
            "wire_call_sha256": _sha(f"wire-{index}"),
            "locked_question_count": request.question_count,
            "source_slot": "poisoned_full_v2",
            "source_priority": 0,
            "source_plan_sha256": full.plan_sha256,
            "source_prepared_sha256": _sha("source-prepared"),
            "source_authorization_sha256": _sha("source-authorization"),
            "source_terminal_sha256": _sha("source-terminal"),
            "source_nested_reuse_terminal_sha256": _sha("nested-terminal"),
            "source_nested_reuse_record_sha256": (
                _sha(f"nested-record-{index}") if index in nested_receipts else None
            ),
            "source_intent_sha256": _sha(f"source-intent-{index}"),
            "source_request_key": request.request_key,
            "source_surface_sha256": _sha(f"source-surface-{index}"),
            "source_receipt_sha256": (None if is_ambiguous else _sha(f"source-receipt-{index}")),
            "source_provider_result_sha256": (
                None if is_ambiguous else _sha(f"source-result-{index}")
            ),
            "source_incident_sha256": (_sha(f"source-incident-{index}") if is_ambiguous else None),
            "source_incident_kind": (
                "provider_call_raised_after_durable_intent" if is_ambiguous else None
            ),
            "source_charged_cost_usd_micros": 1_000 + index,
            "source_retry_permitted": False,
            "target_provider_attempts_permitted_for_entry": 0,
        }
        entries.append(
            EvidenceInferenceFableUnionEntryV2.model_validate(
                {**payload, "entry_sha256": hash_canonical(payload)}
            )
        )
    bindings = [
        EvidenceInferenceFableUnionSourceBindingV2(
            slot=slot,
            priority=priority,
            plan_sha256=full.plan_sha256,
            prepared_sha256=_sha(f"prepared-{priority}"),
            authorization_sha256=_sha(f"authorization-{priority}"),
            terminal_sha256=_sha(f"terminal-{priority}"),
            terminal_status=(
                "completed" if slot == "recovery_pilot_v2" else "terminal_ambiguous_attempt_poison"
            ),
            nested_reuse_terminal_sha256=(
                _sha("nested-terminal") if slot == "poisoned_full_v2" else None
            ),
            source_paths_serialized=False,
            source_workspace_mutation_permitted=False,
        )
        for priority, slot in enumerate(
            ["poisoned_full_v2", "poisoned_pilot_v1", "recovery_pilot_v2"]
        )
    ]
    payload = {
        "plan_version": "evidence-inference-fable-full-union-plan-v2",
        "full_plan_sha256": full.plan_sha256,
        "full_prepared_sha256": _sha("target-prepared"),
        "full_authorization_sha256": _sha("target-authorization"),
        "configured_total_budget_usd_micros": 99_000_000,
        "full_request_count": 382,
        "source_priority": [
            "poisoned_full_v2",
            "poisoned_pilot_v1",
            "recovery_pilot_v2",
        ],
        "source_bindings": bindings,
        "entries": entries,
        "adopted_terminal_receipt_count": 22,
        "inherited_ambiguous_failure_count": 2,
        "maximum_new_provider_attempt_count": 358,
        "shadowed_lower_priority_candidate_count": 8,
        "transitively_reused_nested_record_count": 8,
        "exact_wire_hash_and_deep_call_equality_required": True,
        "source_workspaces_immutable": True,
        "inherited_ambiguity_retry_permitted": False,
        "labels_opened": False,
        "provider_calls_made_while_planning": 0,
        "scientific_claim_authority": False,
        "claim_release_authority": False,
    }
    return EvidenceInferenceFableFullUnionPlanV2.model_validate(
        {**payload, "plan_sha256": hash_canonical(payload)}
    )


def _target_terminal() -> EvidenceInferenceFableTerminalV1:
    payload = {
        "terminal_version": "evidence-inference-fable-terminal-v1",
        "status": "completed",
        "prepared_sha256": _sha("target-prepared"),
        "authorization_sha256": _sha("target-authorization"),
        "completed_request_count": 382,
        "completed_pair_count": 191,
        "cumulative_reported_spend_usd_micros": 10_000,
        "cumulative_spend_semantics": ("reported_usage_or_unknown_usage_hard_liability"),
        "next_pair_index": 191,
        "full_population_score_permitted": True,
        "extraction_accuracy_authority": False,
        "confirmatory_authority": False,
        "synthesis_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
    }
    return EvidenceInferenceFableTerminalV1.model_validate(
        {**payload, "terminal_sha256": hash_canonical(payload)}
    )


def _terminal(
    union: EvidenceInferenceFableFullUnionPlanV2,
    target: EvidenceInferenceFableTerminalV1,
) -> EvidenceInferenceFableFullUnionTerminalV2:
    payload = {
        "terminal_version": "evidence-inference-fable-full-union-terminal-v2",
        "union_plan_sha256": union.plan_sha256,
        "target_runtime_terminal_sha256": target.terminal_sha256,
        "target_runtime_status": "completed",
        "target_completed_request_count": 382,
        "realized_adopted_terminal_receipt_count": 22,
        "realized_inherited_ambiguous_failure_count": 2,
        "new_provider_attempt_count": 358,
        "maximum_new_provider_attempt_count": 358,
        "target_accounted_spend_usd_micros": 10_000,
        "adopted_target_accounted_spend_usd_micros": 3_000,
        "new_provider_accounted_spend_usd_micros": 7_000,
        "source_terminal_artifact_lineage_count": 24,
        "inherited_ambiguous_attempts_retried": 0,
        "target_provider_attempts_for_adopted_entries": 0,
        "full_population_score_permitted": True,
        "scoring_requires_this_union_terminal": True,
        "scientific_claim_authority": False,
        "confirmatory_gepa_improvement_authority": False,
        "claim_release_authority": False,
    }
    return EvidenceInferenceFableFullUnionTerminalV2.model_validate(
        {**payload, "terminal_sha256": hash_canonical(payload)}
    )


def _runtime_failures(
    full: EvidenceInferenceFableRetrospectivePlanV1,
    union: EvidenceInferenceFableFullUnionPlanV2,
) -> tuple[dict[str, EvidenceInferenceFableReceiptV1], dict[str, EvidenceInferenceFableIncidentV2]]:
    inherited = {
        entry.target_request_key
        for entry in union.entries
        if entry.adoption_kind == "inherited_ambiguous_failure"
    }
    union_keys = {entry.target_request_key for entry in union.entries}
    new_incident = next(
        request.request_key
        for request in full.roster
        if request.arm == "seed"
        and request.question_count == 15
        and request.request_key not in union_keys
    )
    reserved = {*inherited, new_incident}
    nonincident_failed = {
        next(
            request.request_key
            for request in full.roster
            if request.arm == arm and request.request_key not in reserved
        )
        for arm in ("seed", "winner")
    }
    incident_keys = {*inherited, new_incident}
    receipts: dict[str, EvidenceInferenceFableReceiptV1] = {}
    incidents: dict[str, EvidenceInferenceFableIncidentV2] = {}
    for request in full.roster:
        key = request.request_key
        intent_sha = _sha(f"target-intent-{key}")
        if key in incident_keys:
            result_payload = {
                "result_version": "evidence-inference-fable-provider-result-v1",
                "request_key": key,
                "surface_sha256": _sha(f"target-surface-{key}"),
                "transport_attempt_count": 1,
                "sdk_retry_count": 0,
                "outcome": "failed",
                "response_id": None,
                "response_model": None,
                "parsed_json": None,
                "input_tokens": None,
                "output_tokens": None,
                "reported_cost_usd_micros": None,
                "charged_cost_usd_micros": 1_000,
                "cost_basis": "unknown_usage_hard_liability",
                "response_text_sha256": None,
                "failure_code": "provider_call_raised_after_durable_intent",
            }
        elif key in nonincident_failed:
            result_payload = {
                "result_version": "evidence-inference-fable-provider-result-v1",
                "request_key": key,
                "surface_sha256": _sha(f"target-surface-{key}"),
                "transport_attempt_count": 1,
                "sdk_retry_count": 0,
                "outcome": "failed",
                "response_id": f"failed-{request.execution_index}",
                "response_model": "claude-fable-5",
                "parsed_json": None,
                "input_tokens": 1,
                "output_tokens": 1,
                "reported_cost_usd_micros": 60,
                "charged_cost_usd_micros": 60,
                "cost_basis": "reported_usage",
                "response_text_sha256": None,
                "failure_code": "response_stop_reason_invalid",
            }
        else:
            result_payload = {
                "result_version": "evidence-inference-fable-provider-result-v1",
                "request_key": key,
                "surface_sha256": _sha(f"target-surface-{key}"),
                "transport_attempt_count": 1,
                "sdk_retry_count": 0,
                "outcome": "completed",
                "response_id": f"completed-{request.execution_index}",
                "response_model": "claude-fable-5",
                "parsed_json": {},
                "input_tokens": 1,
                "output_tokens": 1,
                "reported_cost_usd_micros": 60,
                "charged_cost_usd_micros": 60,
                "cost_basis": "reported_usage",
                "response_text_sha256": None,
                "failure_code": None,
            }
        result = EvidenceInferenceFableProviderResultV1.model_validate(
            {**result_payload, "result_sha256": hash_canonical(result_payload)}
        )
        receipt_payload = {
            "receipt_version": "evidence-inference-fable-receipt-v1",
            "intent_sha256": intent_sha,
            "request_key": key,
            "provider_result": result,
            "locked_question_count": request.question_count,
            "locked_questions_scored_incorrect": (
                request.question_count if result.outcome == "failed" else 0
            ),
        }
        receipts[key] = EvidenceInferenceFableReceiptV1.model_validate(
            {**receipt_payload, "receipt_sha256": hash_canonical(receipt_payload)}
        )
        if key in incident_keys:
            incident_payload = {
                "incident_version": "evidence-inference-fable-incident-v2",
                "status": "failed_request_archived_continue",
                "kind": "provider_call_raised_after_durable_intent",
                "intent_sha256": intent_sha,
                "request_key": key,
                "charged_cost_usd_micros": 1_000,
                "cost_basis": "unknown_usage_hard_liability",
                "retry_permitted": False,
                "sanitization_policy": INCIDENT_SANITIZATION_POLICY,
                "exception_type": "SyntheticProviderFailure",
                "http_status_code": 400 if key == new_incident else None,
                "provider_request_id": None,
                "message_redacted": "Synthetic sanitized failure.",
                "message_was_truncated": False,
                "derived_provider_result_sha256": result.result_sha256,
            }
            incidents[key] = EvidenceInferenceFableIncidentV2.model_validate(
                {**incident_payload, "incident_sha256": hash_canonical(incident_payload)}
            )
    return receipts, incidents


def _public_summary(
    full: EvidenceInferenceFableRetrospectivePlanV1,
    terminal: EvidenceInferenceFableFullUnionTerminalV2,
) -> PublicPairedSummaryV1:
    metric_names = [
        "direction_accuracy",
        "structured_output_reliability",
        "exact_grounding_reliability",
    ]
    arm = ArmAggregateV1(
        requests=191,
        question_evaluations=524,
        valid_batch_requests=191,
        provider_outcome_counts={"provider_response": 191},
        primary_failure_counts={"success": 524},
        metric_success_counts={name: 524 for name in metric_names},
        metric_rates={name: 1.0 for name in metric_names},
        conditional_grounding_denominator=524,
        conditional_exact_grounding_numerator=524,
        conditional_exact_grounding_rate=1.0,
        usage_reported_requests=191,
        usage_missing_requests=0,
        input_tokens=1,
        output_tokens=1,
        accounted_cost_usd_micros=60,
    )
    estimates = [
        MetricBootstrapEstimateV1(
            metric=name,
            seed_success_count=524,
            winner_success_count=524,
            denominator=524,
            seed_rate=Decimal("1.000000000"),
            winner_rate=Decimal("1.000000000"),
            winner_minus_seed_difference=Decimal("0.000000000"),
            percentile_95_lower=Decimal("0.000000000"),
            percentile_95_upper=Decimal("0.000000000"),
        )
        for name in metric_names
    ]
    bootstrap_payload = {
        "bootstrap_version": "evidence-inference-fable-paired-article-bootstrap-v1",
        "inference_pipeline_version": ("evidence-inference-fable-retrospective-inference-v1"),
        "plan_sha256": full.plan_sha256,
        "scoring_completion_binding_sha256": _sha("scoring-binding"),
        "runtime_terminal_sha256": terminal.target_runtime_terminal_sha256,
        "scoring_completion_certificate_sha256": _sha("certificate"),
        "private_scored_rows_sha256": _sha("private-rows"),
        "scoring_artifact_sha256": _sha("scoring-artifact"),
        "mode": "full_paired",
        "population": "full_test",
        "article_cluster_count": 191,
        "question_count": 524,
        "cluster_score_membership_sha256": _sha("cluster-membership"),
        "replicates": 20_000,
        "seed": 20_260_829,
        "sampling_algorithm": ("sha256_counter_modulo_n_resample_n_articles_with_replacement-v1"),
        "interval_method": ("paired_article_cluster_percentile_95_nearest_rank_no_interpolation"),
        "lower_order_index_zero_based": 499,
        "upper_order_index_zero_based": 19_499,
        "primary_metrics": metric_names,
        "estimates": estimates,
        "all_reference_labels_historically_opened": True,
        "interpretation": ("exploratory_cross_model_transfer_on_historically_opened_test"),
        "exploratory_interval_reporting_permitted": True,
        "pilot_mechanics_only_no_inferential_authority": False,
        "confirmatory_gepa_improvement_authority": False,
        "pristine_holdout_authority": False,
        "calibration_authority": False,
        "scientific_claim_authority": False,
        "claim_release_authority": False,
        "provider_calls_made_by_bootstrap": 0,
        "credentials_opened_by_bootstrap": False,
        "network_opened_by_bootstrap": False,
        "benchmark_rows_or_labels_opened_by_bootstrap": False,
    }
    bootstrap = PairedArticleClusterBootstrapV1.model_validate(
        {
            **bootstrap_payload,
            "bootstrap_sha256": hash_canonical(bootstrap_payload),
        }
    )
    public_payload = {
        "public_summary_version": "evidence-inference-fable-public-paired-summary-v1",
        "scoring_version": "evidence-inference-fable-retrospective-scoring-v1",
        "status": "aggregate_only_exploratory_retrospective_paired_score",
        "private_report_sha256": _sha("private-report"),
        "completion_certificate_sha256": _sha("certificate"),
        "plan_sha256": full.plan_sha256,
        "runtime_terminal_sha256": terminal.target_runtime_terminal_sha256,
        "population": "full_test",
        "examples": 524,
        "articles": 191,
        "requests": 382,
        "arms": {"seed": arm, "winner": arm},
        "paired_article_cluster_bootstrap": bootstrap,
        "contains_article_or_question_text": False,
        "contains_article_or_example_identifiers": False,
        "contains_reference_or_per_example_labels": False,
        "contains_raw_or_per_example_predictions": False,
        "contains_evidence_quotes_or_line_references": False,
        "contains_absolute_paths": False,
        "all_reference_labels_historically_opened": True,
        "exploratory_cross_model_transfer_only": True,
        "confirmatory_gepa_improvement_claim_permitted": False,
        "gepa_optimization_improvement_authority": False,
        "scientific_effectiveness_authority": False,
        "generalization_authority": False,
        "eligibility_metric_claim_authority": False,
        "calibration_authority": False,
        "claim_release_authority": False,
        "required_caveats": [
            "historically_opened_test_not_pristine_or_confirmatory",
            "cross_model_and_article_batched_interface_transfer_only",
            "formal_exact_grounding_is_not_semantic_entailment",
            "all_retained_examples_are_eligibility_positive",
        ],
    }
    return PublicPairedSummaryV1.model_validate(
        {
            **public_payload,
            "public_summary_sha256": hash_canonical(public_payload),
        }
    )


def _inputs() -> tuple[Any, ...]:
    full = _full_plan()
    union = _union_plan(full)
    target = _target_terminal()
    terminal = _terminal(union, target)
    receipts, incidents = _runtime_failures(full, union)
    burden = freeze_evidence_inference_fable_full_union_failure_burden_v2(
        full_plan=full,
        union_plan=union,
        union_terminal=terminal,
        target_terminal=target,
        receipts=receipts,
        incidents=incidents,
    )
    public = _public_summary(full, terminal)
    lineage = freeze_evidence_inference_fable_full_union_scoring_lineage_v2(
        union_terminal=terminal,
        completion_certificate_sha256=public.completion_certificate_sha256,
        private_report_sha256=public.private_report_sha256,
        public_summary_sha256=public.public_summary_sha256,
    )
    return full, union, terminal, public, lineage, burden


def test_public_union_projection_is_aggregate_only_hash_bound_and_fresh(
    tmp_path: Path,
) -> None:
    full, union, terminal, public, lineage, burden = _inputs()
    evaluation = project_evidence_inference_fable_full_union_public_evaluation_v2(
        full_plan=full,
        union_plan=union,
        union_terminal=terminal,
        public_summary=public,
        union_scoring_lineage=lineage,
        failure_burden=burden,
    )
    assert evaluation.inherited_failure_request_count_by_arm == {
        "seed": 0,
        "winner": 2,
    }
    assert evaluation.inherited_failure_locked_question_count_by_arm == {
        "seed": 0,
        "winner": 16,
    }
    assert evaluation.union_terminal_sha256 == terminal.terminal_sha256
    assert evaluation.public_summary_sha256 == public.public_summary_sha256
    assert evaluation.union_scoring_lineage_sha256 == lineage.lineage_sha256
    assert evaluation.target_incident_count == 3
    assert evaluation.target_incident_locked_question_count == 31
    assert evaluation.target_incident_request_count_by_arm == {"seed": 1, "winner": 2}
    assert evaluation.target_incident_locked_question_count_by_arm == {
        "seed": 15,
        "winner": 16,
    }
    assert evaluation.new_runtime_incident_request_count_by_arm == {
        "seed": 1,
        "winner": 0,
    }
    assert evaluation.new_runtime_incident_locked_question_count == 15
    assert evaluation.new_runtime_incident_locked_question_count_by_arm == {
        "seed": 15,
        "winner": 0,
    }
    assert evaluation.all_forced_zero_request_count == 5
    assert evaluation.scientific_effectiveness_authority is False
    assert evaluation.claim_release_authority is False
    serialized = json.dumps(evaluation.model_dump(mode="json"), sort_keys=True)
    assert "PMC" not in serialized
    assert "ei2-prompt" not in serialized
    assert '"predictions"' not in serialized
    assert '"request_key"' not in serialized
    assert "Synthetic sanitized failure" not in serialized

    output = tmp_path / "public-union-evaluation.json"
    assert (
        materialize_evidence_inference_fable_full_union_public_evaluation_v2(
            evaluation=evaluation, output_path=output
        )
        == output
    )
    with pytest.raises(
        EvidenceInferenceFableFullUnionReuseError,
        match="target_not_fresh_or_safe",
    ):
        materialize_evidence_inference_fable_full_union_public_evaluation_v2(
            evaluation=evaluation, output_path=output
        )


def test_public_union_projection_rejects_score_lineage_mismatch() -> None:
    full, union, terminal, public, lineage, burden = _inputs()
    payload = lineage.model_dump(mode="json", exclude={"lineage_sha256"})
    payload["public_summary_sha256"] = _sha("different-public")
    mismatched = type(lineage).model_validate(
        {**payload, "lineage_sha256": hash_canonical(payload)}
    )
    with pytest.raises(
        EvidenceInferenceFableFullUnionReuseError,
        match="source_binding_invalid",
    ):
        project_evidence_inference_fable_full_union_public_evaluation_v2(
            full_plan=full,
            union_plan=union,
            union_terminal=terminal,
            public_summary=public,
            union_scoring_lineage=mismatched,
            failure_burden=burden,
        )


def test_project_public_cli_writes_sidecar_after_validated_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full, union, terminal, public, lineage, burden = _inputs()
    workspace = tmp_path / "runtime"
    private = workspace / "private"
    private.mkdir(parents=True)
    public_root = tmp_path / "artifacts" / "diagnostics" / "evidence-inference"
    public_root.mkdir(parents=True)
    public_path = tmp_path / "full-summary.json"
    lineage_path = private / "lineage.json"
    output = public_root / "union-evaluation.json"
    atomic_write_json(public_path, public)
    atomic_write_json(lineage_path, lineage)

    monkeypatch.setattr(
        harness,
        "_context",
        lambda _args: (workspace, full, None, None, [], union),
    )
    monkeypatch.setattr(
        harness,
        "validate_evidence_inference_fable_full_union_v2",
        lambda **_kwargs: terminal,
    )
    monkeypatch.setattr(
        harness,
        "derive_evidence_inference_fable_full_union_failure_burden_v2",
        lambda **_kwargs: burden,
    )
    assert (
        harness.main(
            [
                "project-public",
                "--repository-root",
                str(tmp_path),
                "--expected-full-plan-sha256",
                full.plan_sha256,
                "--expected-authorization-sha256",
                _sha("authorization"),
                "--expected-union-plan-sha256",
                union.plan_sha256,
                "--public-summary",
                str(public_path),
                "--union-scoring-lineage",
                str(lineage_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    observed = json.loads(output.read_text(encoding="utf-8"))
    assert observed["evaluation_sha256"]
    assert observed["public_summary_sha256"] == public.public_summary_sha256
    assert observed["union_scoring_lineage_sha256"] == lineage.lineage_sha256
    assert observed["target_incident_count"] == 3


def test_project_public_cli_paths_fail_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "runtime"
    (workspace / "private").mkdir(parents=True)
    outside_lineage = tmp_path / "outside-lineage.json"
    atomic_write_json(outside_lineage, {"fixture": True})
    with pytest.raises(
        harness.EvidenceInferenceFableFullUnionHarnessError,
        match="outside_private_namespace",
    ):
        harness._safe_private_lineage(
            root=tmp_path,
            workspace=workspace,
            requested=outside_lineage,
        )

    (tmp_path / "artifacts" / "diagnostics" / "evidence-inference").mkdir(parents=True)
    (tmp_path / "not-public").mkdir()
    with pytest.raises(
        harness.EvidenceInferenceFableFullUnionHarnessError,
        match="public_output_path_escape",
    ):
        harness._safe_public_output(
            root=tmp_path,
            requested=tmp_path / "not-public" / "evaluation.json",
        )
