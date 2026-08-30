from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from tests.private_cache_support import (
    HOSTED_ADAPTER_STALE_CODES,
    TYPED_PILOT_STALE_CODES,
    require_private_cache,
    skip_when_historical_replay_is_stale,
)

import literature_multiverse.metasyn_passage_offline_feasibility_audit_v1 as offline
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_bounded_hosted_runtime import MetaSynHostedRuntimeError
from literature_multiverse.metasyn_typed_pilot import MetaSynTypedPilotError

ROOT = Path(__file__).resolve().parents[1]
V2_WORKSPACE = ROOT / "data/cache/metasyn/passage-hosted-yield-v2"
AUDIT_ARTIFACT_PATH = (
    ROOT / "artifacts/diagnostics/metasyn-passage-offline-feasibility-audit-v1.json"
)

EXPECTED_COORDINATES = [
    "03:02",
    "03:03",
    "03:04",
    "10:01",
    "14:01",
    "15:01",
    "15:02",
    "15:03",
    "15:04",
    "16:01",
    "16:02",
    "16:03",
    "16:04",
    "16:05",
    "17:01",
    "17:02",
    "17:03",
    "20:01",
    "20:02",
    "20:03",
    "20:04",
    "22:01",
    "23:01",
    "28:01",
    "28:02",
    "28:03",
]


@pytest.fixture(scope="module")
def audit() -> offline.MetaSynPassageOfflineFeasibilityAuditV1:
    return offline.MetaSynPassageOfflineFeasibilityAuditV1.model_validate(
        json.loads(AUDIT_ARTIFACT_PATH.read_text(encoding="utf-8"))
    )


@pytest.mark.private_cache
def test_live_freeze_metasyn_passage_offline_feasibility_audit_v1_replays_tracked_artifact_or_is_stale(  # noqa: E501
    audit: offline.MetaSynPassageOfflineFeasibilityAuditV1,
) -> None:
    require_private_cache("data/cache/metasyn/passage-hosted-yield-v2")
    rebuilt = skip_when_historical_replay_is_stale(
        lambda: offline.freeze_metasyn_passage_offline_feasibility_audit_v1(repository_root=ROOT),
        stale_errors=(MetaSynTypedPilotError, MetaSynHostedRuntimeError),
        stale_codes=TYPED_PILOT_STALE_CODES | HOSTED_ADAPTER_STALE_CODES,
    )
    assert rebuilt.audit_sha256 == audit.audit_sha256


def test_real_external_v2_replay_covers_the_exact_unattempted_roster(
    audit: offline.MetaSynPassageOfflineFeasibilityAuditV1,
) -> None:
    assert audit.v2_execution_bundle_sha256 == offline.EXPECTED_V2_EXECUTION_BUNDLE_SHA256
    assert audit.v2_inventory_ledger_sha256 == offline.EXPECTED_V2_INVENTORY_LEDGER_SHA256
    assert audit.v2_packet_roster_sha256 == offline.EXPECTED_V2_PACKET_ROSTER_SHA256
    assert audit.v2_failed_smoke_sha256 == offline.EXPECTED_V2_FAILED_SMOKE_SHA256
    assert audit.v2_provider_receipt_count == 43
    assert audit.v2_attempted_packet_count == 3
    assert audit.audited_unattempted_candidate_count == 26
    assert [item.coordinate for item in audit.candidate_audits] == EXPECTED_COORDINATES
    assert all(not item.previously_attempted_in_v2 for item in audit.candidate_audits)
    assert audit.v2_failed_gate_preserved is True


def test_every_full_source_passage_and_candidate_target_is_hash_bound(
    audit: offline.MetaSynPassageOfflineFeasibilityAuditV1,
) -> None:
    for item in audit.candidate_audits:
        assert item.full_quote_policy_applied is True
        assert item.substring_quote_used is False
        assert item.cross_passage_join_used is False
        assert item.candidate_endpoint_and_contrast_held_fixed is True
        assert item.candidate_target_mutated is False
        assert len(item.candidate_passage_ids) == len(item.full_exact_candidate_passage_texts)
        assert item.full_exact_candidate_passage_text_sha256s == [
            hashlib.sha256(text.encode("utf-8")).hexdigest()
            for text in item.full_exact_candidate_passage_texts
        ]
        assert len(item.full_exact_candidate_passage_lineage_sha256s) == len(
            item.candidate_passage_ids
        )
        assert item.question_surface_sha256
        assert item.row_protocol_orientation_sha256
        assert item.assembly_analysis_policy_sha256
        assert item.candidate_binding_sha256
        assert item.scientific_request_signature_sha256

    by_coordinate = {item.coordinate: item for item in audit.candidate_audits}
    assert by_coordinate["17:02"].full_exact_candidate_passage_text_sha256s == [
        "625172b190c130f778efe1bbe398a48ca7e838e6f63b1d702ccca961971dd8bb"
    ]
    assert by_coordinate["17:03"].full_exact_candidate_passage_text_sha256s == [
        "edd2191f8f2cefe305497f05ca929b5e2e05439abc4c6eb1ea069c262bc53a25"
    ]


def test_blocker_taxonomy_is_mutually_exclusive_exhaustive_and_zero_yield(
    audit: offline.MetaSynPassageOfflineFeasibilityAuditV1,
) -> None:
    assert audit.blocker_family_counts == {
        "continuous_core_incomplete": 5,
        "binary_core_or_contrast_incomplete": 9,
        "numeric_token_ambiguity": 2,
        "direct_ci_contract_unreachable": 7,
        "misrouted_non_ci": 3,
    }
    assert sum(audit.blocker_family_counts.values()) == 26
    assert len({item.coordinate for item in audit.candidate_audits}) == 26
    assert all(item.blocker_codes for item in audit.candidate_audits)
    assert all(
        not item.full_self_contained_completed_quote_available for item in audit.candidate_audits
    )
    assert all(
        not item.all_required_numeric_tokens_unique_under_unchanged_v2
        for item in audit.candidate_audits
    )
    assert all(not item.unchanged_v2_grounding_completed for item in audit.candidate_audits)
    assert all(not item.unchanged_v2_assembly_completed for item in audit.candidate_audits)
    assert all(not item.reachable_typed_effect for item in audit.candidate_audits)
    assert audit.reachable_candidate_count == 0
    assert audit.ranked_reachable_candidates == []


def test_full_quote_probes_freeze_the_unchanged_v2_failure_boundary(
    audit: offline.MetaSynPassageOfflineFeasibilityAuditV1,
) -> None:
    by_coordinate = {item.coordinate: item for item in audit.candidate_audits}
    expected = {
        "16:01": "packet_grounding_v2_numeric_token_absent:effect.ci_upper",
        "17:02": ("packet_grounding_v2_numeric_token_not_unique:effect.treatment_total"),
        "17:03": "packet_grounding_v2_numeric_token_not_unique:effect.control_total",
        "20:01": "packet_grounding_v2_effect_format_alias_unsupported",
        "20:02": "packet_grounding_v2_effect_format_alias_unsupported",
        "20:03": "packet_grounding_v2_effect_format_alias_unsupported",
        "20:04": "packet_grounding_v2_effect_format_alias_unsupported",
        "22:01": "packet_grounding_v2_effect_format_alias_unsupported",
    }
    for coordinate, fragment in expected.items():
        item = by_coordinate[coordinate]
        assert item.probe_model_outcome is not None
        assert item.probe_model_outcome_sha256 == hash_canonical(item.probe_model_outcome)
        assert any(fragment in line for line in item.probe_failure_chain)
    assert all(
        not item.probe_failure_chain
        for item in audit.candidate_audits
        if item.coordinate not in expected
    )


def test_partial_row17_witnesses_are_withdrawn_and_confer_no_authority(
    audit: offline.MetaSynPassageOfflineFeasibilityAuditV1,
) -> None:
    assert [item.coordinate for item in audit.withdrawn_partial_witnesses] == [
        "17:02",
        "17:03",
    ]
    assert all(
        item.withdrawal_status == "withdrawn_after_full_source_sentence_self_containment_review"
        for item in audit.withdrawn_partial_witnesses
    )
    assert all(
        not item.observed_partial_receipt_bytes_retained
        for item in audit.withdrawn_partial_witnesses
    )
    assert all(
        not item.observed_partial_receipts_revalidation_authority
        for item in audit.withdrawn_partial_witnesses
    )
    assert all(
        item.discarded_partial_quote_not_reused for item in audit.withdrawn_partial_witnesses
    )
    assert all(not item.source_feasibility_authority for item in audit.withdrawn_partial_witnesses)
    assert all(not item.extraction_accuracy_authority for item in audit.withdrawn_partial_witnesses)
    assert all(not item.synthesis_input_authority for item in audit.withdrawn_partial_witnesses)
    assert all(not item.claim_release_authority for item in audit.withdrawn_partial_witnesses)


def test_zero_yield_is_narrow_and_never_a_general_extractability_claim(
    audit: offline.MetaSynPassageOfflineFeasibilityAuditV1,
) -> None:
    assert audit.zero_yield_scope == (
        "unchanged_v2_grounder_and_assembler_with_one_entire_exact_candidate_passage"
    )
    assert audit.general_extractability_evaluated is False
    assert audit.zero_reachable_does_not_imply_zero_extractable_in_general is True
    assert audit.provider_calls_made == 0
    assert audit.hidden_or_reference_labels_opened is False
    assert audit.official_test_labels_opened is False
    assert audit.extraction_accuracy_authority is False
    assert audit.synthesis_input_authority is False
    assert audit.claim_release_authority is False
    assert all(item.provider_calls_made == 0 for item in audit.candidate_audits)
    assert (
        sum(
            item.source_strength.source_content_scope == "title_abstract"
            for item in audit.candidate_audits
        )
        == 12
    )
    assert (
        sum(
            item.source_strength.release_grade_source_grounding_eligible
            for item in audit.candidate_audits
        )
        == 14
    )


def test_unrehashed_source_or_taxonomy_tamper_is_rejected(
    audit: offline.MetaSynPassageOfflineFeasibilityAuditV1,
) -> None:
    payload = audit.model_dump(mode="json")
    payload["candidate_audits"][0]["full_exact_candidate_passage_texts"][0] += "x"
    with pytest.raises(ValueError):
        offline.MetaSynPassageOfflineFeasibilityAuditV1.model_validate(payload)

    payload = audit.model_dump(mode="json")
    payload["candidate_audits"][0]["blocker_family"] = "numeric_token_ambiguity"
    with pytest.raises(ValueError):
        offline.MetaSynPassageOfflineFeasibilityAuditV1.model_validate(payload)


def test_coherently_rehashed_source_tamper_fails_external_replay(
    audit: offline.MetaSynPassageOfflineFeasibilityAuditV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = audit.model_dump(mode="json")
    candidate = payload["candidate_audits"][0]
    candidate["full_exact_candidate_passage_texts"][0] += "x"
    candidate["full_exact_candidate_passage_text_sha256s"][0] = hashlib.sha256(
        candidate["full_exact_candidate_passage_texts"][0].encode("utf-8")
    ).hexdigest()
    candidate_body = {
        key: value for key, value in candidate.items() if key != "candidate_audit_sha256"
    }
    candidate["candidate_audit_sha256"] = hash_canonical(candidate_body)
    payload["candidate_audit_membership_sha256"] = hash_canonical(
        [item["candidate_audit_sha256"] for item in payload["candidate_audits"]]
    )
    audit_body = {key: value for key, value in payload.items() if key != "audit_sha256"}
    payload["audit_sha256"] = hash_canonical(audit_body)
    tampered = offline.MetaSynPassageOfflineFeasibilityAuditV1.model_validate(payload)

    monkeypatch.setattr(
        offline,
        "freeze_metasyn_passage_offline_feasibility_audit_v1",
        lambda **_: audit,
    )
    with pytest.raises(
        offline.MetaSynPassageOfflineFeasibilityAuditV1Error,
        match="external_replay_mismatch",
    ):
        offline.validate_metasyn_passage_offline_feasibility_audit_v1(
            audit=tampered,
            repository_root=ROOT,
            v2_workspace=V2_WORKSPACE,
            external_replay=True,
        )


def test_diagnostic_writer_is_idempotent_and_rejects_collision(
    audit: offline.MetaSynPassageOfflineFeasibilityAuditV1,
    tmp_path: Path,
) -> None:
    output = tmp_path / "diagnostics" / "audit.json"
    assert offline.write_metasyn_passage_offline_feasibility_audit_v1(
        audit=audit,
        output_path=output,
    ) == output.resolve(strict=True)
    assert offline.write_metasyn_passage_offline_feasibility_audit_v1(
        audit=audit,
        output_path=output,
    ) == output.resolve(strict=True)

    different = audit.model_copy(update={"audit_sha256": "0" * 64})
    with pytest.raises(
        offline.MetaSynPassageOfflineFeasibilityAuditV1Error,
        match="output_replay_mismatch",
    ):
        offline.write_metasyn_passage_offline_feasibility_audit_v1(
            audit=different,
            output_path=output,
        )


def test_pipeline_fingerprint_closes_over_all_new_runtime_files(
    audit: offline.MetaSynPassageOfflineFeasibilityAuditV1,
) -> None:
    files = {
        item.path for component in audit.pipeline_fingerprint.components for item in component.files
    }
    assert {
        "configs/benchmarks/metasyn-passage-offline-feasibility-audit-v1.json",
        "scripts/run_metasyn_passage_offline_feasibility_audit_v1.py",
        "src/literature_multiverse/metasyn_passage_offline_feasibility_audit_v1.py",
    } <= files
    assert audit.pipeline_sha256 == audit.pipeline_fingerprint.pipeline_sha256
