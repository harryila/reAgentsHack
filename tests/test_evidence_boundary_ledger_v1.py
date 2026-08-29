from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from scripts import run_evidence_boundary_ledger_v1 as runner

import literature_multiverse.evidence_boundary_ledger_v1 as ledger_module
from literature_multiverse.evidence_boundary_ledger_v1 import (
    AuthorityKind,
    AuthorizedEmpiricalScope,
    EvidenceBoundaryLedgerError,
    EvidenceBoundaryLedgerV1,
    EvidenceBoundaryRecord,
    EvidenceClass,
    LabelState,
    SourcePayloadState,
    TypedEffectStatus,
    build_evidence_boundary_ledger,
    validate_evidence_boundary_ledger,
)
from literature_multiverse.lineage import OutputExistsError, hash_canonical, sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
V3_PLAN = Path("data/cache/metasyn/passage-packet-rescue-v3/rescue-plan.json")


@pytest.fixture(scope="module")
def ledger() -> EvidenceBoundaryLedgerV1:
    return build_evidence_boundary_ledger(
        repository_root=REPOSITORY_ROOT,
        v3_pre_call_blocker_plan=V3_PLAN,
    )


def test_ledger_separates_evidence_classes_and_keeps_every_authority_fail_closed(
    ledger: EvidenceBoundaryLedgerV1,
) -> None:
    assert [row.record_id for row in ledger.records] == sorted(
        row.record_id for row in ledger.records
    )
    assert {row.evidence_class for row in ledger.records} == {
        EvidenceClass.CONTRACT_ONLY,
        EvidenceClass.REAL_INCOMPLETE_SOURCE_EXECUTION,
        EvidenceClass.REAL_LABEL_BLIND_MECHANICS,
        EvidenceClass.REAL_RETROSPECTIVE_NONPRISTINE,
        EvidenceClass.REAL_SOURCE_PREFLIGHT_BLOCKED,
        EvidenceClass.SIMULATED,
    }
    assert ledger.decision_boundary.real_data_records == 5
    assert ledger.decision_boundary.simulated_records == 1
    assert ledger.decision_boundary.contract_only_records == 1
    assert ledger.decision_boundary.human_expert_adjudication_present is False
    assert ledger.decision_boundary.complete_independent_human_adjudicated_questions == 0
    assert ledger.decision_boundary.release_risk_calibration_eligible is False
    assert ledger.decision_boundary.adaptive_policy_effectiveness_authority is False
    assert ledger.decision_boundary.claim_release_authority is False
    assert all(
        not row.authority.real_world_effectiveness_authority
        and not row.authority.extraction_accuracy_authority
        and not row.authority.scientific_synthesis_accuracy_authority
        and not row.authority.release_risk_calibration_eligible
        and not row.authority.adaptive_policy_effectiveness_authority
        and not row.authority.claim_release_authority
        for row in ledger.records
    )


def test_v2_failed_smoke_is_real_incomplete_execution_not_a_finalized_run(
    ledger: EvidenceBoundaryLedgerV1,
) -> None:
    row = next(
        item
        for item in ledger.records
        if item.record_id == "metasyn_passage_runtime_v2_failed_smoke"
    )
    assert row.evidence_class is EvidenceClass.REAL_INCOMPLETE_SOURCE_EXECUTION
    assert row.runtime_completion.state == ("packet_roster_frozen_failed_smoke_not_finalized")
    assert row.runtime_completion.workspace_finalized is False
    assert row.runtime_completion.remaining_provider_calls_permitted is False
    assert row.runtime_completion.terminal_roster_complete is False
    assert row.runtime_completion.terminal_provider_call_count == 43
    assert row.typed_effect_yield.status is TypedEffectStatus.ZERO_RUNTIME_TYPED_EFFECTS
    assert row.typed_effect_yield.runtime_contract_typed_publications == 0
    assert row.source_access.raw_source_payload_opened_by_ledger is True
    assert row.label_access.label_state is LabelState.REFERENCE_FIELDS_EXPLICITLY_UNOPENED
    assert row.authority.authority_kind is AuthorityKind.REAL_EXECUTION_YIELD_ONLY
    assert ledger.decision_boundary.raw_source_payloads_opened_by_ledger == 2


def test_simulation_contract_and_nonpristine_boundaries_are_explicit(
    ledger: EvidenceBoundaryLedgerV1,
) -> None:
    by_id = {row.record_id: row for row in ledger.records}
    simulation = by_id["adaptive_stress_simulation_v1"]
    assert simulation.evidence_class is EvidenceClass.SIMULATED
    assert simulation.independence.observed_unit_count == 1120
    assert simulation.independence.observation_unit == ("independent_complete_simulated_question")
    assert simulation.source_access.source_payload_state is (
        SourcePayloadState.SYNTHETIC_NOT_APPLICABLE
    )
    assert simulation.authority.authorized_empirical_scope is AuthorizedEmpiricalScope.NONE

    contract = by_id["question_policy_evaluation_contract_v7"]
    assert contract.evidence_class is EvidenceClass.CONTRACT_ONLY
    assert contract.independence.observed_unit_count == 0
    assert contract.label_access.label_state is LabelState.NO_LABELS_CONTRACT_ONLY
    assert contract.authority.authority_kind is AuthorityKind.NONE_CONTRACT_ONLY

    local = by_id["local_benchmark_suite_v1"]
    assert local.evidence_class is EvidenceClass.REAL_RETROSPECTIVE_NONPRISTINE
    assert local.independence.counts_by_partition == {
        "calibration": 161,
        "development": 158,
    }
    assert local.label_access.label_state is (
        LabelState.BENCHMARK_LABELS_PREVIOUSLY_OPENED_AGGREGATES_ONLY
    )
    assert local.label_access.human_expert_adjudication_present is False


def test_zero_typed_effects_and_zero_synthesis_are_not_promoted(
    ledger: EvidenceBoundaryLedgerV1,
) -> None:
    assert ledger.decision_boundary.total_runtime_contract_typed_publications == 0
    assert ledger.decision_boundary.total_release_grade_estimable_publications == 0
    assert ledger.decision_boundary.any_completed_synthesis_mechanics is False
    synthesis_rows = [
        row for row in ledger.records if row.record_id.startswith("metasyn_synthesis_yield")
    ]
    assert len(synthesis_rows) == 2
    assert all(row.typed_effect_yield.graph_estimates == 0 for row in synthesis_rows)
    assert all(row.synthesis_mechanics.synthesis_completed_groups == 0 for row in synthesis_rows)


def test_ledger_and_record_hashes_are_deterministic_and_tamper_evident(
    ledger: EvidenceBoundaryLedgerV1,
) -> None:
    assert validate_evidence_boundary_ledger(ledger.model_dump(mode="json")) == ledger
    assert ledger.ledger_sha256 == hash_canonical(
        ledger.model_dump(mode="json", exclude={"ledger_sha256"})
    )
    assert all(
        row.record_sha256 == hash_canonical(row.model_dump(mode="json", exclude={"record_sha256"}))
        for row in ledger.records
    )

    tampered = copy.deepcopy(ledger.model_dump(mode="json"))
    old_hash = tampered["ledger_implementation_sha256"]
    tampered["ledger_implementation_sha256"] = old_hash[:-1] + ("0" if old_hash[-1] != "0" else "1")
    with pytest.raises(ValueError, match="evidence_ledger_hash_mismatch"):
        validate_evidence_boundary_ledger(tampered)


def test_even_rehashed_simulation_or_mechanics_cannot_gain_effectiveness_authority(
    ledger: EvidenceBoundaryLedgerV1,
) -> None:
    tampered = copy.deepcopy(ledger.model_dump(mode="json"))
    tampered["decision_boundary"]["adaptive_policy_effectiveness_authority"] = True
    payload = {key: value for key, value in tampered.items() if key != "ledger_sha256"}
    tampered["ledger_sha256"] = hash_canonical(payload)
    with pytest.raises(ValueError):
        EvidenceBoundaryLedgerV1.model_validate(tampered)

    tampered = copy.deepcopy(ledger.model_dump(mode="json"))
    row = next(
        item for item in tampered["records"] if item["record_id"] == "adaptive_stress_simulation_v1"
    )
    row["authority"]["claim_release_authority"] = True
    row_payload = {key: value for key, value in row.items() if key != "record_sha256"}
    row["record_sha256"] = hash_canonical(row_payload)
    ledger_payload = {key: value for key, value in tampered.items() if key != "ledger_sha256"}
    tampered["ledger_sha256"] = hash_canonical(ledger_payload)
    with pytest.raises(ValueError):
        EvidenceBoundaryLedgerV1.model_validate(tampered)


def test_schema_represents_optional_v3_pre_call_blocker_without_calling_it_execution(
    ledger: EvidenceBoundaryLedgerV1,
) -> None:
    actual = next(
        row
        for row in ledger.records
        if row.record_id == "metasyn_passage_rescue_v3_pre_call_blocker"
    )
    assert actual.evidence_class is EvidenceClass.REAL_SOURCE_PREFLIGHT_BLOCKED
    assert actual.registered_artifacts[0].semantic_sha256 == (
        "6a135a7e6cde328b92d89128f27d9475ecbbf1e97d35bae7b9a17876751a9a36"
    )
    assert actual.runtime_completion.state == ("pre_call_blocked_zero_provider_calls_not_execution")
    assert actual.runtime_completion.terminal_provider_call_count == 0
    assert actual.typed_effect_yield.status is TypedEffectStatus.NOT_APPLICABLE
    assert actual.authority.authority_kind is (AuthorityKind.REAL_OFFLINE_PREFLIGHT_BLOCKER_ONLY)
    assert actual.authority.claim_release_authority is False

    source = next(
        row for row in ledger.records if row.record_id == "question_policy_evaluation_contract_v7"
    )
    payload = source.model_dump(mode="json", exclude={"record_sha256"})
    payload.update(
        {
            "record_id": "metasyn_passage_rescue_v3_pre_call_blocker",
            "evidence_class": EvidenceClass.REAL_SOURCE_PREFLIGHT_BLOCKED,
            "scientific_role": "offline actual-fixture pre-call blocker",
            "runtime_completion": {
                "state": "pre_call_blocked_zero_provider_calls_not_execution",
                "workspace_finalized": None,
                "terminal_provider_call_count": 0,
                "remaining_provider_calls_permitted": False,
                "terminal_roster_complete": False,
            },
            "validation": {
                "depth": "aggregate_cross_artifact_replay",
                "validator_names": ["validate_metasyn_passage_packet_rescue_plan_v3"],
                "exact_replay_match": True,
                "self_hash_validated": True,
                "current_source_lineage_validated": True,
                "raw_empirical_payload_recomputed": False,
                "raw_evaluator_or_human_labels_opened": False,
            },
            "source_access": {
                "source_payload_state": (
                    SourcePayloadState.OPENED_UPSTREAM_PRIVATE_MECHANICS_ONLY_HERE
                ),
                "raw_source_payload_opened_by_ledger": True,
                "aggregate_or_mechanics_artifact_contains_raw_article_text": True,
                "reference_fields_opened_by_ledger": False,
            },
            "label_access": {
                "label_state": LabelState.REFERENCE_FIELDS_EXPLICITLY_UNOPENED,
                "human_expert_adjudication_present": False,
                "human_expert_adjudication_opened_by_ledger": False,
                "complete_human_adjudicated_question_count": 0,
            },
            "independence": {
                "observation_unit": "selected real-source candidate fixture",
                "observed_unit_count": 3,
                "counts_by_partition": {"offline_pre_call_blocked_candidates": 3},
                "uncertainty_resampling_unit": "none; deterministic exact-fixture replay",
                "complete_independent_claim_question_count": 0,
                "repeated_units_across_artifacts": True,
            },
            "authority": {
                "authority_kind": AuthorityKind.REAL_OFFLINE_PREFLIGHT_BLOCKER_ONLY,
                "authorized_empirical_scope": (
                    AuthorizedEmpiricalScope.REAL_OFFLINE_PREFLIGHT_BLOCKER_ONLY
                ),
                "real_world_effectiveness_authority": False,
                "extraction_accuracy_authority": False,
                "scientific_synthesis_accuracy_authority": False,
                "release_risk_calibration_eligible": False,
                "adaptive_policy_effectiveness_authority": False,
                "claim_release_authority": False,
            },
        }
    )
    record = EvidenceBoundaryRecord.model_validate(
        {**payload, "record_sha256": hash_canonical(payload)}
    )
    assert record.runtime_completion.terminal_provider_call_count == 0
    assert record.typed_effect_yield.status is TypedEffectStatus.NOT_APPLICABLE

    invalid = record.model_dump(mode="json", exclude={"record_sha256"})
    invalid["typed_effect_yield"] = {
        "status": "zero_runtime_typed_effects",
        "publications_evaluated": 0,
        "runtime_contract_typed_publications": 0,
        "release_grade_estimable_publications": 0,
        "graph_estimates": 0,
    }
    with pytest.raises(ValueError, match="typed_status_class_mismatch"):
        EvidenceBoundaryRecord.model_validate({**invalid, "record_sha256": hash_canonical(invalid)})


def test_rehashed_scope_overclaims_and_runtime_shape_contradictions_fail_intrinsically(
    ledger: EvidenceBoundaryLedgerV1,
) -> None:
    tampered = copy.deepcopy(ledger.model_dump(mode="json"))
    local = next(
        row for row in tampered["records"] if row["record_id"] == "local_benchmark_suite_v1"
    )
    local["authority"]["authorized_empirical_scope"] = (
        AuthorizedEmpiricalScope.REAL_EXECUTION_YIELD_MECHANICS_ONLY
    )
    local["record_sha256"] = hash_canonical(
        {key: value for key, value in local.items() if key != "record_sha256"}
    )
    tampered["ledger_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "ledger_sha256"}
    )
    with pytest.raises(ValueError, match="authority_scope_kind_mismatch"):
        EvidenceBoundaryLedgerV1.model_validate(tampered)

    tampered = copy.deepcopy(ledger.model_dump(mode="json"))
    tampered["decision_boundary"]["strongest_authorized_real_empirical_scope"] = (
        AuthorizedEmpiricalScope.RETROSPECTIVE_MATCHED_SUBSET_METRICS_ONLY
    )
    tampered["ledger_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "ledger_sha256"}
    )
    with pytest.raises(ValueError, match="strongest_scope_mismatch"):
        EvidenceBoundaryLedgerV1.model_validate(tampered)

    tampered = copy.deepcopy(ledger.model_dump(mode="json"))
    tampered["decision_boundary"]["next_required_authority_gate"] = (
        "two questions are enough for effectiveness"
    )
    tampered["ledger_sha256"] = hash_canonical(
        {key: value for key, value in tampered.items() if key != "ledger_sha256"}
    )
    with pytest.raises(ValueError, match="next_authority_gate_mismatch"):
        EvidenceBoundaryLedgerV1.model_validate(tampered)

    completed = next(
        row for row in ledger.records if row.runtime_completion.state == "completed_artifact"
    ).model_dump(mode="json", exclude={"record_sha256"})
    completed["runtime_completion"]["remaining_provider_calls_permitted"] = False
    with pytest.raises(ValueError, match="completed_artifact_state_invalid"):
        EvidenceBoundaryRecord.model_validate(
            {**completed, "record_sha256": hash_canonical(completed)}
        )


def test_registered_inputs_exclude_protected_and_hidden_label_paths(
    ledger: EvidenceBoundaryLedgerV1,
) -> None:
    paths = {artifact.path for row in ledger.records for artifact in row.registered_artifacts}
    assert not any("evaluator_labels" in path or "hidden" in path for path in paths)
    assert not any(
        path.startswith(
            (
                "Formatting_Instructions_For_NeurIPS_2026 (2)/",
                "artifacts/paper/",
                "artifacts/submission/",
                "docs/paper/",
                "paper/",
            )
        )
        for path in paths
    )


def test_registered_artifact_loader_fails_closed_on_missing_and_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_id = "temporary_ledger_fixture"
    relative = "fixture.json"
    semantic = "a" * 64
    path = tmp_path / relative
    path.write_text(json.dumps({"artifact_sha256": semantic}), encoding="utf-8")
    original_hash = sha256_file(path)
    monkeypatch.setitem(
        ledger_module._REGISTERED_INPUTS,
        input_id,
        (relative, original_hash, semantic),
    )
    monkeypatch.setitem(
        ledger_module._SEMANTIC_HASH_FIELDS,
        input_id,
        "artifact_sha256",
    )
    path.write_text(json.dumps({"artifact_sha256": "b" * 64}), encoding="utf-8")
    with pytest.raises(
        EvidenceBoundaryLedgerError,
        match="registered_artifact_file_hash_mismatch",
    ):
        ledger_module._load_registered_json(
            repository_root=tmp_path,
            input_id=input_id,
        )

    path.unlink()
    with pytest.raises(EvidenceBoundaryLedgerError, match="registered_artifact_missing"):
        ledger_module._load_registered_json(
            repository_root=tmp_path,
            input_id=input_id,
        )


def test_optional_input_rejects_protected_paper_and_submission_paths() -> None:
    for relative in (
        Path("artifacts/paper/diagnostic.json"),
        Path("artifacts/submission/bundle.json"),
        Path("docs/paper/draft.json"),
        Path("Formatting_Instructions_For_NeurIPS_2026 (2)/plan.json"),
    ):
        with pytest.raises(EvidenceBoundaryLedgerError, match="optional_input_path_protected"):
            ledger_module._optional_input_path(REPOSITORY_ROOT, relative)


def test_cli_writes_new_output_once_without_overwriting(
    ledger: EvidenceBoundaryLedgerV1,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "ledger.json"
    monkeypatch.setattr(runner, "build_evidence_boundary_ledger", lambda **_: ledger)
    assert (
        runner.main(
            [
                "build",
                "--repository-root",
                REPOSITORY_ROOT.as_posix(),
                "--output",
                output.as_posix(),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["ledger_sha256"] == ledger.ledger_sha256
    assert validate_evidence_boundary_ledger(json.loads(output.read_text())) == ledger
    with pytest.raises(OutputExistsError):
        runner.main(
            [
                "build",
                "--repository-root",
                REPOSITORY_ROOT.as_posix(),
                "--output",
                output.as_posix(),
            ]
        )
