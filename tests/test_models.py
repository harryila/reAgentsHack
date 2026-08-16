from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from literature_multiverse.models import (
    M4_CHECKPOINT_ADAPTER,
    AuditDecision,
    AuditFieldChecks,
    AuditRecord,
    CanonicalDirection,
    DirectionNormalizationError,
    EvidenceHashes,
    FindingRow,
    M4CheckpointFrozenIncomplete,
    M4CheckpointNotApplicable,
    NullableEvidenceHashes,
    NullableStageRunHashes,
    PaperRecord,
    ReleaseSelection,
    RemapReconciliationError,
    RemapResponse,
    RunRecord,
    ScaledAttempt,
    SelectedRelease,
    StageRunHashes,
    VerificationDecision,
    VerificationRecord,
    canonical_model_sha256,
    derive_paper_id,
    make_finding_id,
    normalize_direction,
    reconcile_remap_responses,
    render_release_disclosure,
    validate_frozen_s5_completion,
)


def test_paper_id_priority_and_normalization() -> None:
    assert derive_paper_id(doc_id="doc", doi="https://doi.org/10.1/ABC", pmid="12") == (
        "doi:10.1/abc"
    )
    assert derive_paper_id(doc_id="doc", pmid="0012") == "pmid:0012"
    assert derive_paper_id(doc_id="doc") == "doc:doc"


def test_successful_zero_finding_and_ineligible_papers_remain_terminal(
    successful_paper_payload: dict,
) -> None:
    eligible_zero = deepcopy(successful_paper_payload)
    eligible_zero.update(
        raw_finding_count=0,
        accepted_finding_count=0,
        quarantined_finding_count=0,
    )
    assert PaperRecord.model_validate(eligible_zero).eligible is True

    ineligible = deepcopy(eligible_zero)
    ineligible.update(eligible=False, exclusion_reason="wrong population")
    record = PaperRecord.model_validate(ineligible)
    assert record.map_status.value == "success"
    assert record.raw_finding_count == 0


def test_excluded_and_failed_terminal_invariants(successful_paper_payload: dict) -> None:
    excluded = deepcopy(successful_paper_payload)
    excluded.update(
        screen_status="excluded",
        screen_reason="review",
        map_status="not_mapped",
        eligible=None,
        map_result_id=None,
        raw_artifact_path=None,
        raw_finding_count=0,
        accepted_finding_count=0,
        quarantined_finding_count=0,
        prompt_version=None,
        cfghash=None,
    )
    assert PaperRecord.model_validate(excluded).map_status.value == "not_mapped"

    failed = deepcopy(successful_paper_payload)
    failed.update(
        map_status="failed",
        eligible=None,
        map_result_id=None,
        raw_finding_count=0,
        accepted_finding_count=0,
        quarantined_finding_count=0,
        failure_code="MAP_TIMEOUT",
    )
    assert PaperRecord.model_validate(failed).failure_code == "MAP_TIMEOUT"

    for mutation, message in (
        ({"map_status": "success"}, "excluded_paper_must_be_not_mapped"),
        ({"accepted_finding_count": 1}, "excluded_paper_counts_must_be_zero"),
    ):
        invalid = deepcopy(excluded)
        invalid.update(mutation)
        with pytest.raises(ValidationError, match=message):
            PaperRecord.model_validate(invalid)


def test_success_counts_must_reconcile_and_failure_has_zero_counts(
    successful_paper_payload: dict,
) -> None:
    mismatch = deepcopy(successful_paper_payload)
    mismatch["quarantined_finding_count"] = 1
    with pytest.raises(ValidationError, match="finding_counts_do_not_reconcile"):
        PaperRecord.model_validate(mismatch)

    failed = deepcopy(successful_paper_payload)
    failed.update(map_status="failed", eligible=None, failure_code="MAP_FAILED")
    with pytest.raises(ValidationError, match="failed_map_counts_must_be_zero"):
        PaperRecord.model_validate(failed)


def test_alternate_ids_are_sorted_unique_and_preferred_is_not_repeated(
    successful_paper_payload: dict,
) -> None:
    duplicate = deepcopy(successful_paper_payload)
    duplicate["alternate_doc_ids"] = ["z", "z"]
    with pytest.raises(ValidationError, match="not_sorted_unique"):
        PaperRecord.model_validate(duplicate)

    preferred = deepcopy(successful_paper_payload)
    preferred["alternate_doc_ids"] = ["doc-1"]
    with pytest.raises(ValidationError, match="preferred_doc_id_repeated"):
        PaperRecord.model_validate(preferred)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("increase", CanonicalDirection.INCREASE),
        (" positive ", CanonicalDirection.INCREASE),
        ("negative", CanonicalDirection.DECREASE),
        ("null", CanonicalDirection.NO_EFFECT),
        ("No Effect", CanonicalDirection.NO_EFFECT),
        ("neutral", CanonicalDirection.NO_EFFECT),
        ("indeterminate", CanonicalDirection.UNCLEAR),
    ],
)
def test_direction_closed_aliases(source: str, expected: CanonicalDirection) -> None:
    assert normalize_direction(source) is expected


@pytest.mark.parametrize("source", [None, "improved", "harmful", 1])
def test_direction_null_and_unknown_values_are_rejected(source: object) -> None:
    with pytest.raises(DirectionNormalizationError):
        normalize_direction(source)


def test_finding_identity_is_deterministic_and_map_scoped(finding_payload: dict) -> None:
    first = FindingRow.model_validate(finding_payload)
    assert first.finding_id == FindingRow.model_validate(deepcopy(finding_payload)).finding_id

    other_map = deepcopy(finding_payload)
    other_map["map_result_id"] = "map-2"
    other_map["finding_id"] = make_finding_id(
        paper_id=other_map["paper_id"],
        map_result_id=other_map["map_result_id"],
        array_position=other_map["array_position"],
        outcome_name=other_map["outcome_name"],
        timepoint_raw=other_map["timepoint_raw"],
        dose_raw=other_map["dose_raw"],
        effect_direction=other_map["effect_direction"],
    )
    assert FindingRow.model_validate(other_map).finding_id != first.finding_id

    forged = deepcopy(finding_payload)
    forged["finding_id"] = "forged"
    with pytest.raises(ValidationError, match="finding_id_identity_mismatch"):
        FindingRow.model_validate(forged)


def test_remap_echo_back_rejects_duplicate_unknown_and_missing_ids() -> None:
    expected = {"a", "b"}
    with pytest.raises(RemapReconciliationError, match="duplicate"):
        reconcile_remap_responses(
            expected,
            [RemapResponse(finding_id="a", value=1), RemapResponse(finding_id="a", value=2)],
        )
    with pytest.raises(RemapReconciliationError, match="unknown"):
        reconcile_remap_responses(
            expected,
            [RemapResponse(finding_id="a", value=1), RemapResponse(finding_id="c", value=2)],
        )
    with pytest.raises(RemapReconciliationError, match="missing"):
        reconcile_remap_responses(expected, [RemapResponse(finding_id="a", value=None)])
    assert reconcile_remap_responses(
        expected,
        [RemapResponse(finding_id="a", value=None), RemapResponse(finding_id="b", value="x")],
    ) == {"a": None, "b": "x"}


def test_verification_requires_one_immutable_model_decision_per_request(hash64: str) -> None:
    record = VerificationRecord(
        provider="anthropic",
        model="fixture-verifier",
        prompt_version="1",
        prompt_sha256=hash64,
        requested_finding_ids=["a", "b"],
        decisions=[
            VerificationDecision(finding_id="a", model_status="agree", adjudication="none"),
            VerificationDecision(
                finding_id="b", model_status="disagree", adjudication="accept"
            ),
        ],
    )
    assert record.decisions[1].model_status == "disagree"
    with pytest.raises(ValidationError, match="missing_verification_decision"):
        VerificationRecord.model_validate(
            {**record.model_dump(), "decisions": [record.decisions[0].model_dump()]}
        )


def test_audit_counts_recompute_from_all_required_checks() -> None:
    good = AuditDecision(
        finding_id="a",
        checks=AuditFieldChecks(
            eligibility=True,
            atomicity=True,
            intervention=True,
            comparator=True,
            outcome=True,
            timepoint=True,
            direction=True,
            quote_support=True,
        ),
    )
    bad_payload = good.model_dump()
    bad_payload["finding_id"] = "b"
    bad_payload["checks"]["direction"] = False
    bad = AuditDecision.model_validate(bad_payload)
    audit = AuditRecord(
        seed=1,
        requested_sample_size=2,
        sampled_finding_ids=["a", "b"],
        decisions=[good, bad],
        anchor_results={"anchor": True},
        correct_count=1,
        total_count=2,
        wilson_interval=(0.1, 0.9),
        error_taxonomy={"direction": 1},
    )
    assert audit.correct_count == 1


def _run_payload(hash64: str) -> dict:
    return {
        "run_id": "run-1",
        "question_id": "fixture-b-incomplete",
        "stage": "s5",
        "stage_version": "1",
        "status": "complete",
        "completion_mode": "frozen_incomplete",
        "checkpoint_sha256": hash64,
        "started_at": datetime(2026, 8, 15, 12, 0, tzinfo=UTC),
        "completed_at": datetime(2026, 8, 15, 12, 5, tzinfo=UTC),
        "code_version": f"dirty:{hash64}",
        "command_argv": ["scripts/s5_analyze.py"],
        "config_path": "configs/questions/fixture-b-incomplete.yaml",
        "config_sha256": hash64,
        "prompt_path": None,
        "prompt_sha256": None,
        "schema_path": None,
        "schema_sha256": None,
        "cfghash": None,
        "upstream": [],
        "inputs": [],
        "outputs": [],
        "external_result_ids": {},
        "counts": {},
        "warnings": [],
    }


def test_frozen_incomplete_run_requires_exact_wrapper_and_gate(
    hash64: str, source_checkpoint
) -> None:
    source_hash = canonical_model_sha256(source_checkpoint)
    wrapper = M4CheckpointFrozenIncomplete(
        source_checkpoint_sha256=source_hash,
        checkpoint=source_checkpoint,
    )
    run = RunRecord.model_validate({**_run_payload(hash64), "checkpoint_sha256": source_hash})
    validate_frozen_s5_completion(run, wrapper, m4_gate_status="incomplete")
    with pytest.raises(ValueError, match="incomplete_m4_gate"):
        validate_frozen_s5_completion(run, wrapper, m4_gate_status="complete")
    with pytest.raises(ValueError, match="frozen_checkpoint_wrapper"):
        validate_frozen_s5_completion(
            run,
            M4CheckpointNotApplicable(reason="m4_completed"),
            m4_gate_status="incomplete",
        )
    with pytest.raises(ValidationError, match="normal_run_checkpoint_sha256_must_be_null"):
        RunRecord.model_validate(
            {**_run_payload(hash64), "completion_mode": "normal"}
        )


def test_checkpoint_union_rejects_forged_source_hash(source_checkpoint) -> None:
    wrapper = {
        "status": "frozen_incomplete",
        "source_checkpoint_sha256": "0" * 64,
        "checkpoint": source_checkpoint.model_dump(mode="json"),
    }
    with pytest.raises(ValidationError, match="checkpoint_source_hash_mismatch"):
        M4_CHECKPOINT_ADAPTER.validate_python(wrapper)
    assert M4_CHECKPOINT_ADAPTER.validate_python(
        {"status": "not_applicable", "reason": "m4_not_run"}
    ).status == "not_applicable"


def _selected_release(hash64: str, *, role: str, papers: int, release_id: str) -> SelectedRelease:
    return SelectedRelease(
        corpus_role=role,
        release_id=release_id,
        primary_grounded_papers=papers,
        stage_run_sha256s=StageRunHashes(s3=hash64, s4=hash64, s5=hash64),
        evidence_sha256s=EvidenceHashes(
            g3_gate=hash64,
            audit=hash64,
            verification=hash64,
            headline=hash64,
            baseline=hash64,
        ),
    )


def _attempt(
    hash64: str,
    *,
    status: str,
    failure_code: str | None,
    selected: bool = False,
) -> ScaledAttempt:
    stage = NullableStageRunHashes(s3=hash64, s4=hash64, s5=hash64)
    evidence = NullableEvidenceHashes(
        g3_gate=hash64,
        audit=hash64,
        verification=hash64,
        headline=hash64,
        baseline=hash64,
    )
    return ScaledAttempt(
        status=status,
        failure_code=failure_code,
        last_completed_stage="s5",
        candidate_release_id="scaled-release",
        primary_grounded_papers=40,
        stage_run_sha256s=stage,
        evidence_sha256s=evidence,
    )


@pytest.mark.parametrize(
    ("disposition", "attempt_status", "failure_code"),
    [
        ("v1_retained_scaled_incomplete", "incomplete", "scaled_incomplete"),
        ("v1_retained_scaled_corrupt", "rejected", "scaled_artifact_integrity_failed"),
        (
            "v1_retained_scaled_unreconciled",
            "rejected",
            "scaled_ledger_reconciliation_failed",
        ),
        (
            "v1_retained_scaled_unvalidated",
            "rejected",
            "scaled_trust_or_offline_validation_failed",
        ),
    ],
)
def test_release_selection_accepts_each_literal_retained_v1_state(
    hash64: str, disposition: str, attempt_status: str, failure_code: str
) -> None:
    selected = _selected_release(hash64, role="v1", papers=30, release_id="v1-release")
    record = ReleaseSelection(
        disposition=disposition,
        frozen_v1_primary_papers=30,
        selected_release=selected,
        scaled_attempt=_attempt(
            hash64,
            status=attempt_status,
            failure_code=failure_code,
        ),
        rendered_disclosure=render_release_disclosure(
            disposition,
            frozen_v1_primary_papers=30,
            selected_primary_papers=30,
        ),
    )
    assert record.selected_release.corpus_role == "v1"


def test_release_selection_accepts_frozen_and_exact_scaled_promotion(hash64: str) -> None:
    v1 = _selected_release(hash64, role="v1", papers=30, release_id="v1-release")
    frozen = ReleaseSelection(
        disposition="v1_frozen",
        frozen_v1_primary_papers=30,
        selected_release=v1,
        scaled_attempt=None,
        rendered_disclosure=render_release_disclosure(
            "v1_frozen",
            frozen_v1_primary_papers=30,
            selected_primary_papers=30,
        ),
    )
    assert frozen.scaled_attempt is None

    attempt = _attempt(hash64, status="selected", failure_code=None, selected=True)
    scaled = _selected_release(
        hash64,
        role="scaled",
        papers=40,
        release_id="scaled-release",
    )
    promoted = ReleaseSelection(
        disposition="scaled_promoted",
        frozen_v1_primary_papers=30,
        selected_release=scaled,
        scaled_attempt=attempt,
        rendered_disclosure=render_release_disclosure(
            "scaled_promoted",
            frozen_v1_primary_papers=30,
            selected_primary_papers=40,
        ),
    )
    assert promoted.selected_release.release_id == promoted.scaled_attempt.candidate_release_id


def test_release_selection_rejects_wrong_state_or_free_text_disclosure(hash64: str) -> None:
    v1 = _selected_release(hash64, role="v1", papers=30, release_id="v1-release")
    corrupt = _attempt(
        hash64,
        status="rejected",
        failure_code="scaled_artifact_integrity_failed",
    )
    with pytest.raises(ValidationError, match="invalid_release_state"):
        ReleaseSelection(
            disposition="v1_retained_scaled_unvalidated",
            frozen_v1_primary_papers=30,
            selected_release=v1,
            scaled_attempt=corrupt,
            rendered_disclosure="wrong",
        )
    with pytest.raises(ValidationError, match="release_disclosure_mismatch"):
        ReleaseSelection(
            disposition="v1_retained_scaled_corrupt",
            frozen_v1_primary_papers=30,
            selected_release=v1,
            scaled_attempt=corrupt,
            rendered_disclosure="free text",
        )
