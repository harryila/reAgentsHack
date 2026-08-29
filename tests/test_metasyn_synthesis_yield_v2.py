from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import scripts.run_metasyn_synthesis_yield_v2 as synthesis_v2_cli
from pydantic import ValidationError
from tests.test_metasyn_synthesis_yield import (
    REPOSITORY_ROOT,
    SyntheticInputs,
    _runtime_row,
    _sha,
)
from tests.test_metasyn_synthesis_yield import (
    synthetic_inputs as _v1_synthetic_inputs_fixture,
)

from literature_multiverse.lineage import OutputExistsError, hash_canonical
from literature_multiverse.metasyn_bounded_adapter import (
    MetaSynBoundedAdapterError,
    MetaSynBoundedPrivateYieldReportV1,
    freeze_metasyn_bounded_private_yield_report,
)
from literature_multiverse.metasyn_bounded_hosted_runtime import (
    MetaSynHostedAttemptOutcomeRefV1,
    MetaSynHostedCostV1,
    MetaSynHostedPrivateYieldReportV1,
    MetaSynHostedRowResultV1,
    MetaSynHostedUsageV1,
)
from literature_multiverse.metasyn_synthesis_yield_v2 import (
    MetaSynSynthesisYieldV2Error,
    freeze_metasyn_synthesis_yield_v2_public_summary,
    freeze_metasyn_synthesis_yield_v2_report,
    validate_metasyn_synthesis_yield_v2_public_summary,
    validate_metasyn_synthesis_yield_v2_report,
)


@pytest.fixture(scope="module")
def synthetic_inputs() -> Any:
    """Reuse the v1 10-question/32-row builder and its source cleanup."""

    iterator = _v1_synthetic_inputs_fixture.__wrapped__()
    value = next(iterator)
    try:
        yield value
    finally:
        with pytest.raises(StopIteration):
            next(iterator)


def _provider_report(
    inputs: SyntheticInputs, *, zero_yield: bool
) -> MetaSynBoundedPrivateYieldReportV1:
    runtime_rows = {
        row.row_context_sha256: row for row in inputs.runtime.row_results
    }
    publication_results = []
    for row in inputs.adapter.row_contexts:
        runtime_row = runtime_rows[row.row_context_sha256]
        result = None if zero_yield else runtime_row.adapter_publication_result
        if result is None:
            result = _runtime_row(row=row, mode="no_candidate").adapter_publication_result
        assert result is not None
        publication_results.append(result)
    return freeze_metasyn_bounded_private_yield_report(
        adapter_bundle=inputs.adapter,
        publication_results=publication_results,
    )


def _hosted_report(
    inputs: SyntheticInputs, *, zero_yield: bool
) -> MetaSynHostedPrivateYieldReportV1:
    # Transport receipts remain covered in the hosted-runtime suite. This fixture is
    # nevertheless a fully validated hosted report rather than a model_construct stub.
    provider = _provider_report(inputs, zero_yield=zero_yield)
    results = {
        result.row_context_sha256: result for result in provider.publication_results
    }
    hosted_rows: list[MetaSynHostedRowResultV1] = []
    for row in inputs.adapter.row_contexts:
        result = results[row.row_context_sha256]
        status = {
            "typed_publication_output": "typed_publication_output",
            "abstained_inventory_no_candidate": "adapter_inventory_no_candidate",
            "abstained_inventory_uncertain": "adapter_inventory_uncertain",
            "abstained_packet_set_incomplete": "adapter_packet_unable",
            "abstained_packet_unable": "adapter_packet_unable",
        }[result.status]
        validation_status = (
            "inventory_valid_no_candidate_non_authorizing"
            if result.status == "abstained_inventory_no_candidate"
            else (
                "inventory_valid_capacity_or_uncertainty_non_authorizing"
                if result.status == "abstained_inventory_uncertain"
                else "inventory_valid_candidates"
            )
        )
        inventory_outcome = MetaSynHostedAttemptOutcomeRefV1(
            request_key=f"inventory-{row.row_context_sha256}",
            stage="inventory",
            schema_kind="inventory",
            effect_kind=None,
            transport_mode="structured_json_schema",
            structured_grammar_enforced_by_provider=True,
            request_cost_ceiling_usd_micros=1,
            state="response",
            attempt_id=_sha(f"attempt:{row.row_context_sha256}"),
            attempt_intent_sha256=_sha(f"intent:{row.row_context_sha256}"),
            outcome_sha256=_sha(f"outcome:{row.row_context_sha256}"),
            validation_status=validation_status,
            incident_kind=None,
        )
        row_payload = {
            "row_result_version": "metasyn-bounded-hosted-row-result-v2",
            "row_context_sha256": row.row_context_sha256,
            "question_spec_sha256": row.question_spec_sha256,
            "question_bundle_sha256": row.question_bundle_sha256,
            "source_row_sha256": row.source_row_sha256,
            "release_grade_source_grounding_eligible": (
                row.source_row.release_grade_source_grounding_eligible
            ),
            "status": status,
            "blockers": (
                [] if status == "typed_publication_output" else result.blocking_reasons
            ),
            "inventory_outcome": inventory_outcome,
            "packet_outcomes": [],
            "inventory_call_receipt_sha256": inventory_outcome.outcome_sha256,
            "packet_call_receipt_sha256s": [],
            "adapter_publication_result": result,
            "adapter_publication_result_sha256": result.result_sha256,
            "observed_provider_calls": 1,
            "possible_ambiguous_provider_calls": 0,
            "structured_json_schema_calls": 1,
            "prompt_json_schema_calls": 0,
            "possible_ambiguous_charge_ceiling_usd_micros": 0,
            "typed_finding_count": (
                sum(
                    len(cohort.findings)
                    for study in result.official_output.studies
                    for cohort in study.cohorts
                )
                if result.official_output is not None
                else 0
            ),
            "synthesis_input_eligible": (
                status == "typed_publication_output"
                and row.source_row.release_grade_source_grounding_eligible
            ),
            "usage": MetaSynHostedUsageV1(),
            "cost": MetaSynHostedCostV1(request_ceiling_usd_micros=1),
        }
        hosted_rows.append(
            MetaSynHostedRowResultV1.model_validate(
                {**row_payload, "row_result_sha256": hash_canonical(row_payload)}
            )
        )
    hosted_rows.sort(key=lambda item: item.row_context_sha256)
    typed = [item for item in hosted_rows if item.status == "typed_publication_output"]
    report_payload = {
        "report_version": "metasyn-bounded-hosted-private-yield-report-v2",
        "status": "complete_32_row_hosted_yield_only_report",
        "execution_bundle_sha256": _sha("hosted-execution-bundle"),
        "runtime_pipeline_fingerprint": inputs.prepare.pilot_pipeline_fingerprint,
        "runtime_pipeline_sha256": inputs.prepare.pilot_pipeline_sha256,
        "config_sha256": _sha("hosted-config"),
        "anthropic_config_sha256": _sha("anthropic-config"),
        "provider_identity_sha256": _sha("provider-identity"),
        "provider_pricing_table_sha256": _sha("provider-pricing"),
        "adapter_bundle_sha256": inputs.adapter.adapter_bundle_sha256,
        "downstream_verifier_pipeline_sha256": (
            inputs.prepare.downstream_verifier_pipeline_sha256
        ),
        "row_membership_sha256": inputs.adapter.row_membership_sha256,
        "cost_authorization_sha256": _sha("cost-authorization"),
        "preflight_sha256": _sha("preflight"),
        "smoke_sha256": _sha("smoke"),
        "hosted_ledger_sha256": _sha("hosted-ledger"),
        "row_results": hosted_rows,
        "row_result_sha256s": [item.row_result_sha256 for item in hosted_rows],
        "provider_neutral_yield_report": provider,
        "provider_neutral_yield_report_sha256": provider.report_sha256,
        "question_count": 10,
        "component_count": 10,
        "publication_count": 32,
        "row_status_counts": dict(
            sorted(Counter(item.status for item in hosted_rows).items())
        ),
        "typed_publication_output_count": len(typed),
        "release_grade_typed_publication_count": sum(
            item.release_grade_source_grounding_eligible for item in typed
        ),
        "typed_finding_count": sum(item.typed_finding_count for item in typed),
        "observed_source_provider_calls": 32,
        "possible_ambiguous_source_provider_calls": 0,
        "synthetic_preflight_provider_calls": 8,
        "possible_ambiguous_preflight_provider_calls": 0,
        "total_provider_call_attempts_or_possible_attempts": 40,
        "structured_json_schema_calls": 35,
        "prompt_json_schema_calls": 5,
        "maximum_structured_json_schema_calls": 35,
        "maximum_prompt_json_schema_calls": 261,
        "transport_mode_policy": (
            "inventory-structured-json-schema-packet-prompt-json-schema-v1"
        ),
        "observed_preflight_request_ceiling_usd_micros": 8,
        "observed_source_request_ceiling_usd_micros": 32,
        "possible_ambiguous_preflight_charge_ceiling_usd_micros": 0,
        "possible_ambiguous_source_charge_ceiling_usd_micros": 0,
        "observed_request_ceiling_usd_micros": 40,
        "possible_ambiguous_charge_ceiling_usd_micros": 0,
        "durable_intent_count": 40,
        "durable_intent_liability_usd_micros": 40,
        "durable_intent_roster_sha256": _sha("durable-intent-roster"),
        "cost_authorization_ceiling_usd_micros": 5_000_000,
        "configured_cost_ceiling_usd_micros": 20_000_000,
        "maximum_theoretical_provider_calls": 296,
        "application_retries": 0,
        "sdk_retries": 0,
        "usage": MetaSynHostedUsageV1(),
        "cost": MetaSynHostedCostV1(request_ceiling_usd_micros=40),
        "operator_authorized_source_transmission": True,
        "reference_fields_unopened": True,
        "official_test_labels_opened": False,
        "direction_agreement_reported": False,
        "extraction_accuracy_reported": False,
        "claim_release_authority": False,
        "permitted_metrics": (
            "contract_grounding_publication_and_synthesis_input_yield_only"
        ),
    }
    return MetaSynHostedPrivateYieldReportV1.model_validate(
        {**report_payload, "report_sha256": hash_canonical(report_payload)}
    )


_UNCHANGED_PROVIDER = object()


def _rebuild_hosted_report(
    report: MetaSynHostedPrivateYieldReportV1,
    *,
    rows: list[MetaSynHostedRowResultV1] | None = None,
    provider: MetaSynBoundedPrivateYieldReportV1 | object | None = (
        _UNCHANGED_PROVIDER
    ),
) -> MetaSynHostedPrivateYieldReportV1:
    selected_rows = list(report.row_results if rows is None else rows)
    selected_provider = (
        report.provider_neutral_yield_report
        if provider is _UNCHANGED_PROVIDER
        else provider
    )
    assert selected_provider is None or isinstance(
        selected_provider, MetaSynBoundedPrivateYieldReportV1
    )
    typed = [
        item for item in selected_rows if item.status == "typed_publication_output"
    ]
    payload = report.model_dump(mode="python", exclude={"report_sha256"})
    payload.update(
        {
            "row_results": selected_rows,
            "row_result_sha256s": [item.row_result_sha256 for item in selected_rows],
            "provider_neutral_yield_report": selected_provider,
            "provider_neutral_yield_report_sha256": (
                selected_provider.report_sha256
                if selected_provider is not None
                else None
            ),
            "row_status_counts": dict(
                sorted(Counter(item.status for item in selected_rows).items())
            ),
            "typed_publication_output_count": len(typed),
            "release_grade_typed_publication_count": sum(
                item.release_grade_source_grounding_eligible for item in typed
            ),
            "typed_finding_count": sum(item.typed_finding_count for item in typed),
            "observed_source_provider_calls": sum(
                item.observed_provider_calls for item in selected_rows
            ),
            "possible_ambiguous_source_provider_calls": sum(
                item.possible_ambiguous_provider_calls for item in selected_rows
            ),
        }
    )
    return MetaSynHostedPrivateYieldReportV1.model_validate(
        {**payload, "report_sha256": hash_canonical(payload)}
    )


@pytest.fixture(scope="module")
def typed_hosted_report(
    synthetic_inputs: SyntheticInputs,
) -> MetaSynHostedPrivateYieldReportV1:
    return _hosted_report(synthetic_inputs, zero_yield=False)


def _public_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        nested = set().union(*(_public_keys(item) for item in value.values()))
        return {str(key).casefold() for key in value} | nested
    if isinstance(value, list):
        return set().union(*(_public_keys(item) for item in value)) if value else set()
    return set()


def test_typed_full_text_path_blocks_unsupported_synthesis_units(
    synthetic_inputs: SyntheticInputs,
    typed_hosted_report: MetaSynHostedPrivateYieldReportV1,
) -> None:
    report = freeze_metasyn_synthesis_yield_v2_report(
        repository_root=REPOSITORY_ROOT,
        hosted_runtime_report=typed_hosted_report,
        adapter_bundle=synthetic_inputs.adapter,
        prepare_bundle=synthetic_inputs.prepare,
    )

    assert report.publication_count == 32
    assert len(report.row_views) == 32
    assert report.provider_neutral_yield_report_present is True
    assert report.runtime_contract_typed_publication_count == 7
    assert report.original_source_grounding_attempt_count == 7
    assert report.release_grade_estimable_publication_count == 5
    assert report.synthesis_input_group_count == 0
    assert report.synthesis_attempted_group_count == 0
    assert report.synthesis_completed_group_count == 0
    assert report.synthesis_completion_mode_counts == {}
    assert report.synthesis_unit_authorization_receipt_count == (
        report.compatibility_group_count
    )
    assert report.authorized_synthesis_input_group_count == 0
    assert report.compatibility_group_count == report.graph_estimate_count
    assert report.compatibility_groups_are_provisional_single_estimate_strata is True
    assert all(
        len(group.estimate_ids) == 1
        for question in report.question_reports
        for group in question.compatibility_groups
    )
    invented_within_paper = next(
        receipt
        for receipt in report.synthesis_unit_authorization_receipts
        if "within_publication_cohort_independence_not_source_authorized"
        in receipt.issues
    )
    assert invented_within_paper.authorizes_synthesis_input is False
    assert invented_within_paper.structural_claim_support_receipt_sha256s == []
    assert invented_within_paper.pairwise_independence_support_receipt_sha256s == []
    component = report.evaluation_pipeline_fingerprint.components[0]
    bound_paths = {item.path for item in component.files}
    assert component.component_id == "metasyn-hosted-synthesis-yield"
    assert {
        "src/literature_multiverse/metasyn_synthesis_yield_v2.py",
        "src/literature_multiverse/metasyn_bounded_hosted_runtime.py",
        "src/literature_multiverse/metasyn_synthesis_yield.py",
        "src/literature_multiverse/native_grounding.py",
        "src/literature_multiverse/cohort_reconciliation.py",
        "src/literature_multiverse/meta_analysis.py",
    }.issubset(bound_paths)
    assert component.settings["hosted_runtime_private_report_sha256"] == (
        typed_hosted_report.report_sha256
    )
    assert component.settings["provider_neutral_yield_report_sha256"] == (
        typed_hosted_report.provider_neutral_yield_report_sha256
    )
    assert component.settings["synthesis_unit_authorization_contract"] == (
        "metasyn-synthesis-unit-authorization-receipt-v2"
    )
    source_failed = next(
        publication
        for question in report.question_reports
        for publication in question.publication_records
        if publication.stage == "original_source_grounding_failed"
    )
    assert source_failed.original_source_grounding_receipt is not None
    assert source_failed.original_source_grounding_authorized is False
    assert source_failed.terminal_fragment.status.value == "non_estimable"
    assert validate_metasyn_synthesis_yield_v2_report(
        report=report,
        repository_root=REPOSITORY_ROOT,
        hosted_runtime_report=typed_hosted_report,
        adapter_bundle=synthetic_inputs.adapter,
        prepare_bundle=synthetic_inputs.prepare,
    ) == report


def test_zero_yield_keeps_exact_terminal_accounting(
    synthetic_inputs: SyntheticInputs,
) -> None:
    hosted = _hosted_report(synthetic_inputs, zero_yield=True)
    report = freeze_metasyn_synthesis_yield_v2_report(
        repository_root=REPOSITORY_ROOT,
        hosted_runtime_report=hosted,
        adapter_bundle=synthetic_inputs.adapter,
        prepare_bundle=synthetic_inputs.prepare,
    )
    public = freeze_metasyn_synthesis_yield_v2_public_summary(report=report)

    assert report.terminal_fragment_count == 32
    assert report.publication_stage_counts == {
        "runtime_terminal_fragment_excluded": 32
    }
    assert report.runtime_contract_typed_publication_count == 0
    assert report.graph_estimate_count == 0
    assert report.compatibility_group_count == 0
    assert report.synthesis_attempted_group_count == 0
    assert report.synthesis_completed_group_count == 0
    assert public.synthesis_private_report_sha256 == report.report_sha256


def test_missing_provider_aggregate_never_fabricates_an_inventory(
    synthetic_inputs: SyntheticInputs,
) -> None:
    hosted = _hosted_report(synthetic_inputs, zero_yield=True)
    first_payload = hosted.row_results[0].model_dump(
        mode="python", exclude={"row_result_sha256"}
    )
    first_payload.update(
        {
            "status": "runtime_inventory_blocked",
            "blockers": ["inventory_transport_or_contract_failure"],
            "adapter_publication_result": None,
            "adapter_publication_result_sha256": None,
        }
    )
    first = MetaSynHostedRowResultV1.model_validate(
        {**first_payload, "row_result_sha256": hash_canonical(first_payload)}
    )
    hosted = _rebuild_hosted_report(
        hosted,
        rows=[first, *hosted.row_results[1:]],
        provider=None,
    )
    report = freeze_metasyn_synthesis_yield_v2_report(
        repository_root=REPOSITORY_ROOT,
        hosted_runtime_report=hosted,
        adapter_bundle=synthetic_inputs.adapter,
        prepare_bundle=synthetic_inputs.prepare,
    )
    public = freeze_metasyn_synthesis_yield_v2_public_summary(report=report)

    assert report.provider_neutral_yield_report_present is False
    assert report.provider_neutral_yield_report_sha256 is None
    assert report.terminal_fragment_count == 32
    assert report.publication_stage_counts == {
        "runtime_terminal_fragment_excluded": 32
    }
    assert public.provider_neutral_yield_report_present is False
    assert public.provider_neutral_yield_report_sha256 is None


def test_all_adapter_results_require_provider_aggregate(
    synthetic_inputs: SyntheticInputs,
    typed_hosted_report: MetaSynHostedPrivateYieldReportV1,
) -> None:
    hosted = _rebuild_hosted_report(
        typed_hosted_report,
        provider=None,
    )

    with pytest.raises(
        MetaSynSynthesisYieldV2Error,
        match="provider_report_completeness_mismatch",
    ):
        freeze_metasyn_synthesis_yield_v2_report(
            repository_root=REPOSITORY_ROOT,
            hosted_runtime_report=hosted,
            adapter_bundle=synthetic_inputs.adapter,
            prepare_bundle=synthetic_inputs.prepare,
        )


def test_public_summary_has_no_identifiers_and_binds_private_report(
    synthetic_inputs: SyntheticInputs,
    typed_hosted_report: MetaSynHostedPrivateYieldReportV1,
) -> None:
    report = freeze_metasyn_synthesis_yield_v2_report(
        repository_root=REPOSITORY_ROOT,
        hosted_runtime_report=typed_hosted_report,
        adapter_bundle=synthetic_inputs.adapter,
        prepare_bundle=synthetic_inputs.prepare,
    )
    public = freeze_metasyn_synthesis_yield_v2_public_summary(report=report)
    payload = public.model_dump(mode="json")
    forbidden = {
        "question_id",
        "publication_id",
        "paper_id",
        "study_id",
        "cohort_id",
        "estimate_id",
        "source_locator",
        "quote",
        "conclusion_summary",
        "effect_direction",
    }

    assert not _public_keys(payload).intersection(forbidden)
    serialized = json.dumps(payload, sort_keys=True)
    assert "metasyn-publication:" not in serialized
    assert "parquet:" not in serialized
    private_values = {
        question.question_id
        for question in report.question_reports
    } | {
        value
        for question in report.question_reports
        for publication in question.publication_records
        for value in (
            publication.publication_id,
            publication.paper_id,
            publication.source_locator,
        )
    } | {
        estimate.estimate_id
        for question in report.question_reports
        for estimate in question.terminal_corpus.graph.outcome_estimates
    }
    assert not any(value in serialized for value in private_values)
    assert "synthesis_unit_authorization_receipts" not in payload
    assert payload["structural_synthesis_authorization_required"] is True
    assert payload["structural_authorization_details_public"] is False
    assert payload["compatibility_groups_are_provisional_single_estimate_strata"] is True
    assert public.hosted_runtime_private_report_sha256 == (
        typed_hosted_report.report_sha256
    )
    assert public.hosted_runtime_pipeline_sha256 == (
        typed_hosted_report.runtime_pipeline_sha256
    )
    assert public.hosted_execution_bundle_sha256 == (
        typed_hosted_report.execution_bundle_sha256
    )
    assert public.provider_neutral_yield_report_sha256 == (
        typed_hosted_report.provider_neutral_yield_report_sha256
    )
    assert public.adapter_bundle_sha256 == (
        synthetic_inputs.adapter.adapter_bundle_sha256
    )
    assert public.prepare_bundle_sha256 == (
        synthetic_inputs.prepare.prepare_bundle_sha256
    )
    assert public.downstream_verifier_pipeline_sha256 == (
        synthetic_inputs.prepare.downstream_verifier_pipeline_sha256
    )
    assert public.row_view_membership_sha256 == report.row_view_membership_sha256
    assert public.synthesis_private_report_sha256 == report.report_sha256
    assert public.official_test_labels_opened is False
    assert public.direction_agreement_reported is False
    assert public.extraction_accuracy_reported is False
    assert public.calibration_authority is False
    assert public.claim_release_authority is False
    assert validate_metasyn_synthesis_yield_v2_public_summary(
        summary=public, report=report
    ) == public

    tampered = payload | {"synthesis_private_report_sha256": "0" * 64}
    tampered["summary_sha256"] = hash_canonical(
        {
            key: value
            for key, value in tampered.items()
            if key != "summary_sha256"
        }
    )
    with pytest.raises(
        MetaSynSynthesisYieldV2Error,
        match="public_external_replay_mismatch",
    ):
        validate_metasyn_synthesis_yield_v2_public_summary(
            summary=tampered, report=report
        )


def test_private_tampering_is_rejected(
    synthetic_inputs: SyntheticInputs,
    typed_hosted_report: MetaSynHostedPrivateYieldReportV1,
) -> None:
    report = freeze_metasyn_synthesis_yield_v2_report(
        repository_root=REPOSITORY_ROOT,
        hosted_runtime_report=typed_hosted_report,
        adapter_bundle=synthetic_inputs.adapter,
        prepare_bundle=synthetic_inputs.prepare,
    )
    payload = report.model_dump(mode="json")
    payload["row_views"][0]["hosted_row_result_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="row_view_hash_mismatch"):
        type(report).model_validate(payload)


def test_context_join_rejects_coherently_rehashed_runtime_status_swap(
    synthetic_inputs: SyntheticInputs,
    typed_hosted_report: MetaSynHostedPrivateYieldReportV1,
) -> None:
    report = freeze_metasyn_synthesis_yield_v2_report(
        repository_root=REPOSITORY_ROOT,
        hosted_runtime_report=typed_hosted_report,
        adapter_bundle=synthetic_inputs.adapter,
        prepare_bundle=synthetic_inputs.prepare,
    )
    payload = report.model_dump(mode="json")
    question = payload["question_reports"][0]
    record = question["publication_records"][0]
    record["runtime_status"] = "coherently-forged-runtime-status"
    record["record_sha256"] = hash_canonical(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )
    question["publication_record_sha256s"] = sorted(
        item["record_sha256"] for item in question["publication_records"]
    )
    question["question_report_sha256"] = hash_canonical(
        {
            key: value
            for key, value in question.items()
            if key != "question_report_sha256"
        }
    )
    payload["question_report_sha256s"] = sorted(
        item["question_report_sha256"] for item in payload["question_reports"]
    )
    payload["report_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "report_sha256"}
    )

    with pytest.raises(ValidationError, match="context_join_mismatch"):
        type(report).model_validate(payload)


def test_authorization_receipt_tamper_rejects_after_coherent_rehash(
    synthetic_inputs: SyntheticInputs,
    typed_hosted_report: MetaSynHostedPrivateYieldReportV1,
) -> None:
    report = freeze_metasyn_synthesis_yield_v2_report(
        repository_root=REPOSITORY_ROOT,
        hosted_runtime_report=typed_hosted_report,
        adapter_bundle=synthetic_inputs.adapter,
        prepare_bundle=synthetic_inputs.prepare,
    )
    payload = report.model_dump(mode="json")
    receipt = payload["synthesis_unit_authorization_receipts"][0]
    receipt["issues"] = sorted([*receipt["issues"], "forged_support_assertion"])
    receipt["receipt_sha256"] = hash_canonical(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    payload["synthesis_unit_authorization_receipt_sha256s"] = sorted(
        item["receipt_sha256"]
        for item in payload["synthesis_unit_authorization_receipts"]
    )
    payload["report_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "report_sha256"}
    )

    with pytest.raises(ValidationError, match="authorization_receipt_replay_mismatch"):
        type(report).model_validate(payload)


def test_provider_neutral_report_tampering_fails_before_derivation(
    synthetic_inputs: SyntheticInputs,
    typed_hosted_report: MetaSynHostedPrivateYieldReportV1,
) -> None:
    provider = typed_hosted_report.provider_neutral_yield_report
    assert provider is not None
    payload = provider.model_dump(mode="json")
    payload["adapter_bundle_sha256"] = "0" * 64
    payload["report_sha256"] = hash_canonical(
        {key: value for key, value in payload.items() if key != "report_sha256"}
    )
    forged_provider = MetaSynBoundedPrivateYieldReportV1.model_validate(payload)
    forged_hosted = _rebuild_hosted_report(
        typed_hosted_report,
        provider=forged_provider,
    )
    with pytest.raises(
        MetaSynBoundedAdapterError, match=r"yield|adapter|provider"
    ):
        freeze_metasyn_synthesis_yield_v2_report(
            repository_root=REPOSITORY_ROOT,
            hosted_runtime_report=forged_hosted,
            adapter_bundle=synthetic_inputs.adapter,
            prepare_bundle=synthetic_inputs.prepare,
        )


def test_cli_refuses_existing_fixed_output_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / synthesis_v2_cli.PRIVATE_OUTPUT_RELATIVE
    private.parent.mkdir(parents=True)
    private.write_text("do-not-overwrite", encoding="utf-8")

    def forbidden_evaluate(**_: Any) -> tuple[Any, Any]:
        raise AssertionError("evaluation must not start when an output exists")

    monkeypatch.setattr(
        synthesis_v2_cli,
        "evaluate_current_metasyn_synthesis_yield_v2",
        forbidden_evaluate,
    )
    with pytest.raises(OutputExistsError):
        synthesis_v2_cli.main(
            [
                "--repository-root",
                str(tmp_path),
                "--hosted-runtime-workspace",
                "hosted",
                "--pilot-workspace",
                "pilot",
                "--expected-execution-bundle-sha256",
                "a" * 64,
            ]
        )
    assert private.read_text(encoding="utf-8") == "do-not-overwrite"


def test_dirfd_writer_installs_canonical_pair_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    private = repository_root / synthesis_v2_cli.PRIVATE_OUTPUT_RELATIVE
    public = repository_root / synthesis_v2_cli.PUBLIC_OUTPUT_RELATIVE
    repository_root_fd = synthesis_v2_cli._open_repository_root(repository_root)
    try:
        synthesis_v2_cli._write_output_pair_no_overwrite(
            repository_root_fd=repository_root_fd,
            private_output=private,
            private_value={"z": 2, "a": 1},
            public_output=public,
            public_value={"public": True},
        )
        with pytest.raises(OutputExistsError):
            synthesis_v2_cli._write_output_pair_no_overwrite(
                repository_root_fd=repository_root_fd,
                private_output=private,
                private_value={"replacement": True},
                public_output=public,
                public_value={"replacement": True},
            )
    finally:
        os.close(repository_root_fd)

    assert private.read_bytes() == b'{"a":1,"z":2}\n'
    assert public.read_bytes() == b'{"public":true}\n'


def test_cli_dirfd_writer_rejects_artifact_ancestor_swap_after_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    artifacts = repository_root / "artifacts"
    artifacts.mkdir()
    displaced_artifacts = repository_root / "artifacts-before-swap"
    outside = tmp_path / "outside"
    outside.mkdir()

    def synthetic_evaluate(**_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"private": True}, {"public": True}

    monkeypatch.setattr(
        synthesis_v2_cli,
        "evaluate_current_metasyn_synthesis_yield_v2",
        synthetic_evaluate,
    )
    monkeypatch.setattr(
        synthesis_v2_cli,
        "validate_metasyn_synthesis_yield_v2_public_summary",
        lambda **_: None,
    )
    original_install = synthesis_v2_cli._install_staged_no_overwrite_at
    swapped = False

    def swap_artifacts_then_install(**kwargs: Any) -> Any:
        nonlocal swapped
        if kwargs["relative"] == synthesis_v2_cli.PUBLIC_OUTPUT_RELATIVE:
            assert not swapped
            artifacts.rename(displaced_artifacts)
            artifacts.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_install(**kwargs)

    monkeypatch.setattr(
        synthesis_v2_cli,
        "_install_staged_no_overwrite_at",
        swap_artifacts_then_install,
    )

    with pytest.raises(
        ValueError,
        match="metasyn_synthesis_v2_v2_public_output_symlink_forbidden",
    ):
        synthesis_v2_cli.main(
            [
                "--repository-root",
                str(repository_root),
                "--hosted-runtime-workspace",
                "hosted",
                "--pilot-workspace",
                "pilot",
                "--expected-execution-bundle-sha256",
                "a" * 64,
            ]
        )

    assert swapped is True
    assert list(outside.iterdir()) == []
    assert not (
        repository_root / synthesis_v2_cli.PRIVATE_OUTPUT_RELATIVE
    ).exists()
    assert not (
        displaced_artifacts
        / Path(*synthesis_v2_cli.PUBLIC_OUTPUT_RELATIVE.parts[1:])
    ).exists()
    assert not list(repository_root.rglob("*.tmp"))
    assert not list(displaced_artifacts.rglob("*.tmp"))
