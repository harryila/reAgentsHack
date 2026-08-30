from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.private_cache_support import require_private_cache

import literature_multiverse.metasyn_contextual_frontier_v1_failure_audit as failure
from literature_multiverse.lineage import hash_canonical

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / failure.DEFAULT_V1_WORKSPACE
OUTPUT = ROOT / failure.DEFAULT_OUTPUT_PATH

pytestmark = pytest.mark.private_cache


@pytest.fixture(scope="module")
def audit() -> failure.MetaSynContextualFrontierV1FailureAudit:
    require_private_cache("data/cache/metasyn/contextual-frontier-runtime-v1")
    return failure.freeze_metasyn_contextual_frontier_v1_failure_audit(
        repository_root=ROOT,
        v1_workspace=WORKSPACE,
    )


def test_completed_provider_calls_and_costs_are_bound_without_reclassification(
    audit: failure.MetaSynContextualFrontierV1FailureAudit,
) -> None:
    assert audit.terminal_status == "roster_exhausted_without_typed_graph"
    assert audit.provider_receipt_count == 2
    assert audit.structured_completed_response_count == 2
    assert audit.typed_graph_completed_response_count == 0
    assert audit.total_input_tokens == 14_850
    assert audit.total_output_tokens == 14_717
    assert audit.total_estimated_cost_usd_micros == 884_350
    assert audit.total_charged_cost_ceiling_usd_micros == 23_200_000
    assert [item.estimated_cost_usd_micros for item in audit.request_audits] == [
        504_480,
        379_870,
    ]
    assert all(item.provider_outcome == "completed" for item in audit.request_audits)
    assert all(item.stop_reason == "end_turn" for item in audit.request_audits)
    assert all(item.structured_json_completed for item in audit.request_audits)
    assert audit.provider_or_transport_failure_explains_result is False
    assert audit.structured_json_failure_explains_result is False


def test_sort_only_replay_proves_order_canonicalization_is_insufficient(
    audit: failure.MetaSynContextualFrontierV1FailureAudit,
) -> None:
    by_key = {item.request_key: item for item in audit.request_audits}
    assert by_key["row17-candidate3-fable5-high"].sort_only_failure_code == (
        "contextual_grounding_v3_binary_contract_mismatch"
    )
    assert by_key["row17-candidate2-fable5-high"].sort_only_failure_code == (
        "contextual_grounding_v3_endpoint_marker_not_exact"
    )
    assert all(not item.raw_claim_order_canonical for item in audit.request_audits)
    assert all(not item.sort_only_salvage_succeeded for item in audit.request_audits)
    assert all(not item.canonicalization_only_recovery_supported for item in audit.request_audits)
    assert audit.canonical_sort_alone_salvages_any_response is False


def test_schema_and_fixture_surfaces_explain_post_schema_rejections(
    audit: failure.MetaSynContextualFrontierV1FailureAudit,
) -> None:
    primary, symptom = audit.request_audits
    assert primary.missing_required_field_paths == ("cohort.registry_id",)
    assert primary.extra_nonbinary_field_paths == (
        "effect.ci_level",
        "effect.ci_lower",
        "effect.ci_upper",
        "effect.estimate",
    )
    assert symptom.missing_required_field_paths == ()
    assert symptom.extra_nonbinary_field_paths == ()
    assert all(
        not item.provider_schema_enforced_exact_binary_field_set for item in audit.request_audits
    )
    assert all(not item.provider_schema_enforced_claim_order for item in audit.request_audits)
    assert [item.provider_visible_passage_count for item in audit.request_audits] == [16, 16]
    assert [item.fixture_selected_passage_count for item in audit.request_audits] == [4, 3]
    assert all(item.fixture_passage_surface_is_strict_subset for item in audit.request_audits)
    assert all(
        item.predicted_provider_visible_but_fixture_excluded_passage_ids
        for item in audit.request_audits
    )
    assert primary.endpoint_quote_exact_in_provider_visible_passage is False
    assert symptom.endpoint_quote_exact_in_provider_visible_passage is False
    assert "contrast.marker" in symptom.nonexact_support_quote_field_paths


def test_arm_ambiguity_is_not_repaired_by_leaking_expected_values(
    audit: failure.MetaSynContextualFrontierV1FailureAudit,
) -> None:
    primary, symptom = audit.request_audits
    assert (primary.predicted_treatment_arm, primary.predicted_treatment_events) == (
        "fedratinib 400-mg",
        "35",
    )
    assert (
        primary.code_owned_target_treatment_arm,
        primary.code_owned_target_treatment_events,
    ) == (
        "500-mg",
        "39",
    )
    assert (symptom.predicted_treatment_arm, symptom.predicted_treatment_events) == (
        "400-mg",
        "33",
    )
    assert (
        symptom.code_owned_target_treatment_arm,
        symptom.code_owned_target_treatment_events,
    ) == (
        "500-mg",
        "31",
    )
    assert all(not item.upstream_protocol_prespecified_exact_dose for item in audit.request_audits)
    assert all(item.source_exposes_multiple_active_doses for item in audit.request_audits)
    assert all(
        item.single_contrast_target_ambiguous_without_protocol_arm for item in audit.request_audits
    )
    assert audit.expected_numeric_values_exposed_to_provider is False
    assert audit.exposing_expected_numeric_values_would_invalidate_independent_extraction_claim
    assert audit.exact_arm_may_be_provider_input_only_if_prespecified_upstream
    assert audit.otherwise_all_source_supported_contrasts_must_be_enumerated


def test_artifact_is_hash_bound_non_authorizing_and_externally_replayable(
    audit: failure.MetaSynContextualFrontierV1FailureAudit,
) -> None:
    assert len(audit.bound_artifacts) == 6
    assert audit.new_provider_calls_made == 0
    assert audit.immutable_v1_preserved is True
    assert audit.extraction_accuracy_authority is False
    assert audit.synthesis_input_authority is False
    assert audit.scientific_synthesis_authority is False
    assert audit.scientific_effectiveness_authority is False
    assert audit.calibration_authority is False
    assert audit.claim_release_authority is False
    assert all(not item.claim_release_authority for item in audit.request_audits)
    assert (
        hash_canonical(audit.model_dump(mode="json", exclude={"audit_sha256"}))
        == audit.audit_sha256
    )

    saved = failure.MetaSynContextualFrontierV1FailureAudit.model_validate(
        json.loads(OUTPUT.read_text(encoding="utf-8"))
    )
    assert saved == audit
    assert (
        failure.validate_metasyn_contextual_frontier_v1_failure_audit(
            audit=saved,
            repository_root=ROOT,
            v1_workspace=WORKSPACE,
            external_replay=True,
        )
        == audit
    )


def test_coherently_rehashed_claim_cannot_grant_authority(
    audit: failure.MetaSynContextualFrontierV1FailureAudit,
) -> None:
    payload = audit.model_dump(mode="json")
    payload["claim_release_authority"] = True
    body = {key: value for key, value in payload.items() if key != "audit_sha256"}
    payload["audit_sha256"] = hash_canonical(body)
    with pytest.raises(ValueError):
        failure.MetaSynContextualFrontierV1FailureAudit.model_validate(payload)
