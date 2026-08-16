from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from literature_multiverse.extract import RawFinding
from literature_multiverse.normalize import (
    NormalizationError,
    apply_patches,
    compute_quality_metrics,
    ground_evidence,
    normalize_direction,
    normalize_raw_finding,
    normalized_frames,
    primary_cohort,
    reconcile_verification,
    write_processed_ledgers,
)


def _paper(paper_id: str = "doc:p1", *, accepted: int = 4, quarantined: int = 1) -> dict:
    return {
        "paper_id": paper_id,
        "screen_status": "included",
        "map_status": "success",
        "eligible": True,
        "accepted_finding_count": accepted,
        "quarantined_finding_count": quarantined,
        "raw_finding_count": accepted + quarantined,
        "pub_year": None,
    }


def _finding(
    finding_id: str,
    *,
    grounding: str = "exact",
    section_flagged: bool = False,
    direction: str = "increase",
    family: str = "primary",
    tuple_suffix: str = "a",
) -> dict:
    return {
        "finding_id": finding_id,
        "paper_id": "doc:p1",
        "grounding_status": grounding,
        "section_flagged": section_flagged,
        "effect_direction": direction,
        "outcome_family": family,
        "prompt_version": f"prompt-{tuple_suffix}",
        "schema_version": "1",
        "cfghash": tuple_suffix * 64,
        "array_position": 0,
        "sample_size": None,
        "moderators": {"dose": "high"},
    }


def test_grounding_normalizes_unicode_whitespace_and_multiline_ranges() -> None:
    result = ground_evidence(
        "Dose was 5 μg and\nresponse increased.",
        ["L10-L11"],
        {
            "L10": {"text": "Dose was 5 μg", "section": "Results"},
            "L11": {"text": "and   response increased.", "section": "Results"},
        },
    )
    assert result.status == "exact"
    assert result.evidence_section == "Results"
    assert result.section_flagged is False


@pytest.mark.parametrize(
    ("quote", "lines", "source", "expected"),
    [
        (None, ["L1"], {"L1": "text"}, "missing"),
        ("quote", [], {"L1": "quote"}, "missing"),
        ("quote", ["L1"], None, "unverifiable"),
        ("quote", ["L2"], {"L1": "quote"}, "unverifiable"),
        ("quote", ["L1"], {"L1": {"text": "different", "section": "Results"}}, "mismatch"),
    ],
)
def test_grounding_failure_states(quote, lines, source, expected: str) -> None:
    assert ground_evidence(quote, lines, source).status == expected


@pytest.mark.parametrize("section", ["Abstract", "Discussion", "Conclusion", "References", None])
def test_banned_or_unknown_sections_are_flagged(section: str | None) -> None:
    result = ground_evidence("supported", ["L1"], {"L1": {"text": "supported", "section": section}})
    assert result.status == "exact"
    assert result.section_flagged is True


def test_direction_contract_maps_only_closed_aliases() -> None:
    assert normalize_direction(" positive ") == "increase"
    assert normalize_direction("null") == "no_effect"
    with pytest.raises(NormalizationError, match="FINDING_DIRECTION_JSON_NULL"):
        normalize_direction(None)
    with pytest.raises(NormalizationError, match="FINDING_DIRECTION_UNKNOWN"):
        normalize_direction("beneficial")


def test_strict_extraction_requires_nullable_keys_and_rejects_extras() -> None:
    raw = RawFinding(
        paper_id="doc:p1",
        doc_id="p1",
        map_result_id="m1",
        array_position=0,
        payload={"outcome_name": "x", "effect_direction": "increase", "junk": "value"},
    )
    accepted, rejected = normalize_raw_finding(
        raw,
        prompt_version="p1",
        schema_version="1",
        cfghash="a" * 64,
        require_all_keys=True,
    )
    assert accepted is None
    assert rejected is not None
    assert rejected.reason_code == "FINDING_EXTRA_FIELDS"

    raw_without_extra = RawFinding(
        paper_id=raw.paper_id,
        doc_id=raw.doc_id,
        map_result_id=raw.map_result_id,
        array_position=raw.array_position,
        payload={"outcome_name": "x", "effect_direction": "increase"},
    )
    accepted, rejected = normalize_raw_finding(
        raw_without_extra,
        prompt_version="p1",
        schema_version="1",
        cfghash="a" * 64,
        require_all_keys=True,
    )
    assert accepted is None
    assert rejected is not None
    assert rejected.reason_code == "FINDING_REQUIRED_KEYS_MISSING"


def test_patches_require_exact_single_match_old_value_and_reason() -> None:
    rows = [{"finding_id": "f1", "confidence": 0.5}, {"finding_id": "f2", "confidence": 0.5}]
    updated, audit = apply_patches(
        rows,
        [
            {
                "selector": {"finding_id": "f1"},
                "field": "confidence",
                "expected_old_value": 0.5,
                "value": 0.8,
                "reason": "human source check",
            }
        ],
    )
    assert updated[0]["confidence"] == 0.8
    assert rows[0]["confidence"] == 0.5
    assert audit[0]["old_value"] == 0.5

    for patch, code in [
        (
            {
                "selector": {"confidence": 0.5},
                "field": "confidence",
                "expected_old_value": 0.5,
                "value": 0.8,
                "reason": "ambiguous",
            },
            "PATCH_SELECTOR_CARDINALITY",
        ),
        (
            {
                "selector": {"finding_id": "f1"},
                "field": "confidence",
                "expected_old_value": 0.1,
                "value": 0.8,
                "reason": "wrong old",
            },
            "PATCH_OLD_VALUE_MISMATCH",
        ),
        (
            {
                "selector": {"finding_id": "f1"},
                "field": "paper_id",
                "expected_old_value": None,
                "value": "doc:other",
                "reason": "immutable join key",
            },
            "PATCH_IMMUTABLE_FIELD_FORBIDDEN",
        ),
    ]:
        with pytest.raises(NormalizationError) as exc_info:
            apply_patches(rows, [patch])
        assert exc_info.value.code == code


def test_verification_and_quality_denominators_follow_contract() -> None:
    findings = [
        _finding("f1"),
        _finding("f2", section_flagged=True, direction="mixed"),
        _finding("f3", grounding="mismatch", direction="decrease"),
        _finding("f4", family="secondary"),
    ]
    verification = {
        "requested_finding_ids": ["f1", "f2", "f4"],
        "decisions": [
            {"finding_id": "f1", "model_status": "disagree", "adjudication": "accept"},
            {"finding_id": "f2", "model_status": "agree", "adjudication": "none"},
            {"finding_id": "f4", "model_status": "agree", "adjudication": "none"},
        ],
    }

    metrics = compute_quality_metrics([_paper()], findings, verification, primary_family="primary")

    assert metrics["grounded"] == {"numerator": 2, "denominator": 3, "fraction": 2 / 3}
    assert metrics["quarantine"] == {"numerator": 1, "denominator": 5, "fraction": 0.2}
    assert metrics["cross_model_agreement"] == {
        "numerator": 2,
        "denominator": 3,
        "fraction": 2 / 3,
    }
    assert metrics["mixed_or_unclear_exclusion"]["fraction"] == 1 / 3
    assert metrics["section_flagged_exclusion"]["fraction"] == 1 / 3
    # Human acceptance controls inclusion, but never improves model agreement.
    assert metrics["verification_exclusion"]["fraction"] == 0
    assert [
        row["finding_id"]
        for row in primary_cohort([_paper()], findings, verification, primary_family="primary")
    ] == ["f1"]


def test_verification_duplicate_unknown_and_missing_ids_fail() -> None:
    findings = [_finding("f1")]
    valid = {"requested_finding_ids": ["f1"]}
    cases = [
        (
            {
                **valid,
                "decisions": [
                    {"finding_id": "f1", "model_status": "agree", "adjudication": "none"},
                    {"finding_id": "f1", "model_status": "agree", "adjudication": "none"},
                ],
            },
            "VERIFICATION_DUPLICATE_ID",
        ),
        (
            {
                "requested_finding_ids": ["f1", "other"],
                "decisions": [
                    {"finding_id": "f1", "model_status": "agree", "adjudication": "none"}
                ],
            },
            "VERIFICATION_REQUEST_SET_MISMATCH",
        ),
        ({**valid, "decisions": []}, "VERIFICATION_MISSING_IDS"),
    ]
    for verification, code in cases:
        with pytest.raises(NormalizationError) as exc_info:
            reconcile_verification(findings, verification)
        assert exc_info.value.code == code


def test_zero_quarantine_denominator_is_null() -> None:
    metrics = compute_quality_metrics(
        [],
        [],
        {"requested_finding_ids": [], "decisions": []},
        primary_family="primary",
    )
    assert metrics["quarantine"] == {"numerator": 0, "denominator": 0, "fraction": None}


def test_s4_frames_use_nullable_ints_flatten_moderators_and_reject_mixing() -> None:
    papers_frame, findings_frame = normalized_frames(
        [_paper()],
        [_finding("f1")],
        moderator_names=["dose"],
        moderator_types={"dose": "categorical"},
    )
    assert str(papers_frame["pub_year"].dtype) == "Int64"
    assert str(findings_frame["sample_size"].dtype) == "Int64"
    assert findings_frame.loc[0, "mod__dose"] == "high"
    assert str(findings_frame["mod__dose"].dtype) == "string"

    with pytest.raises(NormalizationError, match="S4_MIXED_EXTRACTION_TUPLES"):
        normalized_frames([_paper()], [_finding("f1"), _finding("f2", tuple_suffix="b")])
    orphan = _finding("f1")
    orphan["paper_id"] = "doc:missing"
    with pytest.raises(NormalizationError, match="S4_ORPHAN_FINDINGS"):
        normalized_frames([_paper()], [orphan])

    _, empty_findings = normalized_frames([_paper(accepted=0, quarantined=0)], [])
    assert "finding_id" in empty_findings.columns
    assert str(empty_findings["sample_size"].dtype) == "Int64"


def test_processed_parquet_write_refuses_overwrite(tmp_path: Path) -> None:
    papers_path = tmp_path / "papers.parquet"
    findings_path = tmp_path / "findings.parquet"
    assert write_processed_ledgers(
        [_paper()], [_finding("f1")], papers_path=papers_path, findings_path=findings_path
    ) == (1, 1)
    assert len(pd.read_parquet(papers_path)) == 1
    with pytest.raises(NormalizationError, match="S4_OUTPUT_EXISTS"):
        write_processed_ledgers(
            [_paper()],
            [_finding("f1")],
            papers_path=papers_path,
            findings_path=findings_path,
        )


def test_endpoint_map_canonicalizes_outcome_name_before_family_mapping() -> None:
    from literature_multiverse.extract import RawFinding
    from literature_multiverse.normalize import normalize_raw_finding

    raw = RawFinding(
        doc_id="PMCTEST01",
        paper_id="doc:PMCTEST01",
        map_result_id="m_test",
        array_position=0,
        payload={
            "outcome_name": "two-legged knee extension 1 RM",
            "effect_direction": "no_effect",
            "evidence_quote": None,
            "evidence_lines": None,
        },
    )
    row, rejected = normalize_raw_finding(
        raw,
        prompt_version="test-v1",
        schema_version="1",
        cfghash="0" * 64,
        outcome_family_map={"muscle_strength": "functional_adaptation"},
        outcome_endpoint_map={r"re:knee extension|1\s?RM": "muscle_strength"},
    )
    assert rejected is None
    assert row is not None
    assert row["outcome_name"] == "muscle_strength"
    assert row["outcome_family"] == "functional_adaptation"
    warnings = row["normalization_warnings"]
    assert any(warning.startswith("outcome_name_canonicalized:") for warning in warnings)


def test_unmatched_outcome_name_is_preserved_verbatim() -> None:
    from literature_multiverse.extract import RawFinding
    from literature_multiverse.normalize import normalize_raw_finding

    raw = RawFinding(
        doc_id="PMCTEST02",
        paper_id="doc:PMCTEST02",
        map_result_id="m_test",
        array_position=0,
        payload={
            "outcome_name": "serum hepcidin",
            "effect_direction": "increase",
            "evidence_quote": None,
            "evidence_lines": None,
        },
    )
    row, rejected = normalize_raw_finding(
        raw,
        prompt_version="test-v1",
        schema_version="1",
        cfghash="0" * 64,
        outcome_endpoint_map={r"re:knee extension": "muscle_strength"},
    )
    assert rejected is None
    assert row is not None
    assert row["outcome_name"] == "serum hepcidin"


def test_non_line_locator_tokens_are_dropped_with_warning() -> None:
    """Observed live 2026-08-15: citations like ["L28", "L34", "Table 3"]."""
    from literature_multiverse.normalize import ground_evidence

    source = {
        f"L{number}": {
            "line_id": f"L{number}",
            "text": "VO2max did not differ between groups." if number == 28 else "filler",
            "section": "Results",
        }
        for number in range(28, 35)
    }
    result = ground_evidence(
        "VO2max did not differ between groups.", ["L28", "L34", "Table 3"], source
    )
    assert result.status == "exact"
    assert result.section_flagged is False
    assert result.dropped_line_tokens == ("Table 3",)

    # All-invalid citations remain unverifiable.
    bad = ground_evidence("VO2max did not differ between groups.", ["Table 3"], source)
    assert bad.status == "unverifiable"


def test_exposure_terms_quarantine_non_exposure_arm() -> None:
    """2026-08-16 census audit: a factorial trial's placebo+exercise arm row is not a
    finding of the target relation."""
    from literature_multiverse.extract import RawFinding
    from literature_multiverse.normalize import normalize_raw_finding

    def raw(intervention: str | None, intervention_class: str | None) -> RawFinding:
        return RawFinding(
            paper_id="doc:PMCX",
            doc_id="PMCX",
            map_result_id="m_x",
            array_position=0,
            payload={
                "outcome_name": "muscle_strength",
                "effect_direction": "increase",
                "intervention": intervention,
                "intervention_class": intervention_class,
                "evidence_quote": None,
                "evidence_lines": None,
            },
        )

    _, rejected = normalize_raw_finding(
        raw("Placebo combined with resistance training (PLB+RT)", "exercise_only"),
        prompt_version="v",
        schema_version="1",
        cfghash="c",
        exposure_terms=["vitamin", "VES"],
    )
    assert rejected is not None
    assert rejected.reason_code == "FINDING_INTERVENTION_LACKS_EXPOSURE"

    record, rejected = normalize_raw_finding(
        raw(
            "Vitamin E supplementation combined with resistance training",
            "vitamin_e_and_exercise",
        ),
        prompt_version="v",
        schema_version="1",
        cfghash="c",
        exposure_terms=["vitamin", "VES"],
    )
    assert rejected is None and record is not None

    # No intervention text at all -> the rule does not fire.
    record, rejected = normalize_raw_finding(
        raw(None, None),
        prompt_version="v",
        schema_version="1",
        cfghash="c",
        exposure_terms=["vitamin"],
    )
    assert rejected is None and record is not None
