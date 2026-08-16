from __future__ import annotations

import pytest

from literature_multiverse.audit import (
    build_audit_artifact,
    compute_quality_metrics,
    select_audit_sample,
)
from literature_multiverse.grounding import (
    GroundingContractError,
    expand_line_reference,
    ground_evidence,
    is_primary_headline_row,
    normalize_evidence_text,
    reconcile_verification,
)


def test_normalizes_unicode_whitespace_and_multiline_ranges() -> None:
    lines = {
        "L18": {"text": "Vitamin\u00a0C reduced", "section": "Results"},
        "L19": {"text": "fatigue after training.", "section": "Results"},
        "L20": {"text": "Secondary sentence.", "section": "Results"},
    }
    result = ground_evidence(
        "Vitamin C reduced\n fatigue after training.",
        ["L18-L19"],
        lines,
    )
    assert expand_line_reference("L18\u2013L20") == (18, 19, 20)
    assert normalize_evidence_text("A\u00a0 B\nC") == "A B C"
    assert result["grounding_status"] == "exact"
    assert result["resolved_line_numbers"] == [18, 19]
    assert result["evidence_section"] == "Results"
    assert result["section_flagged"] is False


@pytest.mark.parametrize(
    ("quote", "refs", "content", "accessible", "expected"),
    [
        (None, ["L1"], {1: "text"}, True, "missing"),
        ("", ["L1"], {1: "text"}, True, "missing"),
        ("text", [], {1: "text"}, True, "missing"),
        ("other", ["L1"], {1: "text"}, True, "mismatch"),
        ("text", ["L2"], {1: "text"}, True, "unverifiable"),
        ("text", ["L1"], None, False, "unverifiable"),
    ],
)
def test_grounding_statuses_are_closed(
    quote: str | None,
    refs: list[str],
    content: dict[int, str] | None,
    accessible: bool,
    expected: str,
) -> None:
    result = ground_evidence(quote, refs, content, source_accessible=accessible)
    assert result["grounding_status"] == expected


@pytest.mark.parametrize(
    "section", ["Abstract", "Discussion", "Conclusion", "Conclusions", "References", None]
)
def test_banned_and_unknown_sections_are_flagged(section: str | None) -> None:
    result = ground_evidence(
        "supported",
        ["L1"],
        {1: {"text": "supported", "section": section}},
    )
    assert result["grounding_status"] == "exact"
    assert result["section_flagged"] is True


def test_any_banned_section_flags_a_mixed_range() -> None:
    result = ground_evidence(
        "first second",
        ["L1-L2"],
        {
            1: {"text": "first", "section": "Results"},
            2: {"text": "second", "section": "Discussion"},
        },
    )
    assert result["evidence_section"] == "multiple"
    assert result["section_flagged"] is True


def test_verification_reconciliation_preserves_model_rate_and_adjudicates_inclusion() -> None:
    reconciled = reconcile_verification(
        ["f1", "f2", "f3"],
        [
            {"finding_id": "f1", "model_status": "agree", "adjudication": "none"},
            {"finding_id": "f2", "model_status": "disagree", "adjudication": "accept"},
            {"finding_id": "f3", "model_status": "unverifiable", "adjudication": "reject"},
        ],
    )
    assert reconciled["cross_model_agreement"] == pytest.approx(1 / 3)
    assert reconciled["primary_included"] == 2
    assert reconciled["primary_excluded"] == 1
    assert is_primary_headline_row(
        {
            "effect_direction": "increase",
            "outcome_family": "primary",
            "grounding_status": "exact",
            "section_flagged": False,
        },
        reconciled["decisions"][1],
        primary_family="primary",
    )


@pytest.mark.parametrize(
    "decisions",
    [
        [{"finding_id": "f1", "model_status": "agree", "adjudication": "none"}],
        [
            {"finding_id": "f1", "model_status": "agree", "adjudication": "none"},
            {"finding_id": "f1", "model_status": "agree", "adjudication": "none"},
        ],
        [
            {"finding_id": "f1", "model_status": "agree", "adjudication": "none"},
            {"finding_id": "unknown", "model_status": "agree", "adjudication": "none"},
        ],
    ],
)
def test_missing_duplicate_and_unknown_verifier_ids_fail(decisions: list[dict[str, str]]) -> None:
    with pytest.raises(GroundingContractError):
        reconcile_verification(["f1", "f2"], decisions)


def test_invalid_line_reference_fails_loudly() -> None:
    with pytest.raises(GroundingContractError):
        expand_line_reference("L4-L2")


def test_quality_denominators_recompute_exactly_and_adjudication_never_improves_model_rate() -> (
    None
):
    papers = [
        {
            "paper_id": "p1",
            "screen_status": "included",
            "map_status": "success",
            "eligible": True,
            "accepted_finding_count": 4,
            "quarantined_finding_count": 1,
        },
        {
            "paper_id": "p2",
            "screen_status": "included",
            "map_status": "success",
            "eligible": True,
            "accepted_finding_count": 2,
            "quarantined_finding_count": 0,
        },
    ]

    def finding(
        finding_id: str,
        paper_id: str,
        direction: str,
        grounding: str,
        *,
        family: str = "primary",
        flagged: bool = False,
    ) -> dict[str, object]:
        return {
            "finding_id": finding_id,
            "paper_id": paper_id,
            "outcome_family": family,
            "effect_direction": direction,
            "grounding_status": grounding,
            "section_flagged": flagged,
        }

    findings = [
        finding("f1", "p1", "increase", "exact"),
        finding("f2", "p1", "decrease", "exact"),
        finding("f3", "p1", "no_effect", "exact"),
        finding("f4", "p1", "mixed", "exact", flagged=True),
        finding("f5", "p2", "unclear", "mismatch"),
        finding("f6", "p2", "increase", "exact", family="secondary"),
    ]
    verification = [
        {"finding_id": "f1", "model_status": "agree", "adjudication": "none"},
        {"finding_id": "f2", "model_status": "disagree", "adjudication": "accept"},
        {"finding_id": "f3", "model_status": "unverifiable", "adjudication": "reject"},
        {"finding_id": "f4", "model_status": "agree", "adjudication": "none"},
        {"finding_id": "f6", "model_status": "disagree", "adjudication": "accept"},
    ]
    quality = compute_quality_metrics(papers, findings, verification, primary_family="primary")
    assert quality["grounded_fraction"] == {
        "numerator": 4,
        "denominator": 5,
        "value": 0.8,
    }
    assert quality["quarantine_fraction"]["numerator"] == 1
    assert quality["quarantine_fraction"]["denominator"] == 7
    assert quality["quarantine_fraction"]["value"] == pytest.approx(1 / 7)
    assert quality["cross_model_agreement"] == {
        "numerator": 2,
        "denominator": 5,
        "value": 0.4,
    }
    assert quality["mixed_or_unclear_fraction"]["value"] == pytest.approx(2 / 5)
    assert quality["section_flagged_fraction"]["value"] == pytest.approx(1 / 5)
    assert quality["verification_excluded_fraction"] == {
        "numerator": 1,
        "denominator": 3,
        "value": pytest.approx(1 / 3),
    }


def test_fixed_seed_audit_sample_meets_scaled_distinct_new_paper_quota() -> None:
    rows = [
        {
            "finding_id": f"f{index}",
            "paper_id": f"p{index}",
            "effect_direction": ("increase", "no_effect", "decrease")[index % 3],
            "outcome_name": ("speed", "strength")[index % 2],
        }
        for index in range(25)
    ]
    first = select_audit_sample(
        rows,
        seed=5,
        new_paper_ids=[f"p{index}" for index in range(12)],
        minimum_new_distinct_papers=10,
    )
    second = select_audit_sample(
        rows,
        seed=5,
        new_paper_ids=[f"p{index}" for index in range(12)],
        minimum_new_distinct_papers=10,
    )
    assert first == second
    assert first["sample_size"] == 20
    assert first["sampled_new_distinct_papers"] >= 10
    assert first["distinct_papers"] == 20

    all_pass = {
        name: True
        for name in (
            "eligibility",
            "atomicity",
            "intervention",
            "comparator",
            "outcome",
            "timepoint",
            "direction",
            "quote_support",
        )
    }
    decisions = [
        {"finding_id": finding_id, "fields": dict(all_pass)} for finding_id in first["finding_ids"]
    ]
    decisions[0]["fields"]["direction"] = False
    artifact = build_audit_artifact(first["finding_ids"], decisions)
    assert artifact["audit_correct"] == 19
    assert artifact["audit_total"] == 20
    assert artifact["passes_17_of_20"] is True
    assert artifact["error_taxonomy"] == {"direction": 1}


def test_two_element_citation_is_an_inclusive_range() -> None:
    """Observed live 2026-08-15: the extractor cites contiguous passages as [start, end]."""
    from literature_multiverse.grounding import ground_evidence

    content = {
        "L58": {
            "text": "The improvement was significant for groups A and B,",
            "section": "Results",
        },
        "L59": {"text": "while no such difference in VO2max was found", "section": "Results"},
        "L60": {"text": "for the combined group.", "section": "Results"},
        "L200": {"text": "unrelated distant line", "section": "Results"},
    }
    grounded = ground_evidence(
        "while no such difference in VO2max was found",
        ["L58", "L60"],
        content,
    )
    assert grounded["grounding_status"] == "exact"
    assert grounded["resolved_line_numbers"] == [58, 59, 60]

    # A distant pair keeps the literal two-line meaning (no silent spanning); the
    # relocation rule then recovers the uniquely-located quote with an audit trail.
    distant = ground_evidence(
        "while no such difference in VO2max was found",
        ["L58", "L200"],
        content,
    )
    assert distant["grounding_status"] == "exact"
    assert distant["resolved_line_numbers"] == [59]
    assert distant["relocated_from_line_numbers"] == [58, 200]


def test_glyph_spacing_differences_still_ground_exactly() -> None:
    """Observed live: provider text spaces out special glyphs ('V ˙ O 2max')."""
    from literature_multiverse.grounding import ground_evidence

    content = {
        "L58": {
            "text": "improvement in  V ˙ O 2max  after HIIT for vitamin C,",
            "section": "Results",
        },
        "L59": {
            "text": "respectively while no such difference in  V ˙ O 2max  was found",
            "section": "Results",
        },
    }
    grounded = ground_evidence(
        "while no such difference in V˙O2max was found",
        ["L58", "L59"],
        content,
    )
    assert grounded["grounding_status"] == "exact"


def test_ellipsis_spliced_quotes_ground_segmentwise() -> None:
    """Observed live: extractors splice long sentences with '...' in quotes."""
    from literature_multiverse.grounding import quote_content_contained

    cited = (
        "however, the as group had higher increases in arm lean mass (d = 0.74), "
        "skeletal muscle mass index (d = 0.71), handgrip strength (d = 0.51), and knee "
        "extension strength (d = 0.89) than the pla group."
    )
    assert quote_content_contained(
        "however, the as group had higher increases in... handgrip strength (d = 0.51)...",
        cited,
    )
    # Order matters: segments may not match out of sequence.
    assert not quote_content_contained(
        "handgrip strength (d = 0.51)... higher increases in arm lean mass",
        cited,
    )
    assert not quote_content_contained("a sentence that is not present", cited)


def test_colon_suffixed_line_reference_strips_to_line_id() -> None:
    """Observed live 2026-08-15: citations like "L18: However, the AS group ..."."""
    from literature_multiverse.grounding import expand_line_reference, strip_line_reference_suffix

    assert strip_line_reference_suffix("L18: However, the AS group had higher") == "L18"
    assert expand_line_reference("L18: However, the AS group had higher") == (18,)
    assert expand_line_reference("L12-L14: some inline text") == (12, 13, 14)
    # A plain time-like or prose string must not silently become a citation.
    assert strip_line_reference_suffix("Results: significant") == "Results: significant"


def test_relocate_quote_unique_region_recovers_wrong_citation() -> None:
    from literature_multiverse.grounding import ground_evidence

    content = {
        "L10": {"text": "Participants received supplements in a double-blind manner.",
                "section": "Methods"},
        "L20": {
            "text": "For 1RM, a significant main effect of time was observed",
            "section": "Results",
        },
        "L21": {"text": "and a significant time-group interaction emerged.", "section": "Results"},
    }
    result = ground_evidence(
        "For 1RM, a significant main effect of time was observed and a significant "
        "time-group interaction emerged.",
        ["L10"],
        content,
    )
    assert result["grounding_status"] == "exact"
    assert result["resolved_line_numbers"] == [20, 21]
    assert result["relocated_from_line_numbers"] == [10]
    assert result["evidence_section"] == "Results"
    assert result["section_flagged"] is False


def test_relocate_quote_ambiguous_regions_stay_mismatch() -> None:
    from literature_multiverse.grounding import ground_evidence

    content = {
        "L5": {"text": "The treatment increased strength markedly.", "section": "Results"},
        "L50": {"text": "The treatment increased strength markedly.", "section": "Discussion"},
    }
    result = ground_evidence("The treatment increased strength markedly.", ["L9"], content)
    # L9 is absent -> unverifiable, so cite an existing wrong line instead.
    result = ground_evidence("The treatment increased strength markedly.", ["L5"], content)
    assert result["grounding_status"] == "exact"  # direct hit is untouched
    result = ground_evidence("The treatment increased strength markedly.", ["L6"], content)
    assert result["grounding_status"] == "unverifiable"


def test_relocate_quote_two_copies_never_relocates() -> None:
    from literature_multiverse.grounding import ground_evidence

    content = {
        "L5": {"text": "The treatment increased strength markedly.", "section": "Results"},
        "L6": {"text": "Unrelated methods text.", "section": "Methods"},
        "L50": {"text": "The treatment increased strength markedly.", "section": "Discussion"},
    }
    result = ground_evidence("The treatment increased strength markedly.", ["L6"], content)
    assert result["grounding_status"] == "mismatch"
    assert "relocated_from_line_numbers" not in result


def test_normalize_path_relocation_rebinds_lines_and_warns() -> None:
    from literature_multiverse.normalize import expand_evidence_lines, ground_evidence

    assert expand_evidence_lines(["L18: quoted text here"]) == ["L18"]

    source = {
        "L10": {"line_id": "L10", "text": "Dosing was 800 IU daily.", "section": "Methods"},
        "L64": {
            "line_id": "L64",
            "text": "1RM strength increased significantly in the placebo group only.",
            "section": "Results",
        },
    }
    result = ground_evidence(
        "1RM strength increased significantly in the placebo group only.", ["L10"], source
    )
    assert result.status == "exact"
    assert result.relocated_line_ids == ("L64",)
    assert result.evidence_section == "Results"


def test_flagged_citation_refines_to_clean_subset() -> None:
    """Observed live 2026-08-15: an abstract line co-cited beside the Results line that
    already contains the whole quote must not flag the finding."""
    from literature_multiverse.grounding import ground_evidence

    content = {
        "L10": {"text": "Vitamins attenuated VO2max gains after training.", "section": "Abstract"},
        "L50": {
            "text": "On the other hand, vitamins significantly attenuated VO2max by 3.11.",
            "section": "Results",
        },
    }
    result = ground_evidence(
        "vitamins significantly attenuated VO2max by 3.11.", ["L10", "L50"], content
    )
    assert result["grounding_status"] == "exact"
    assert result["section_flagged"] is False
    assert result["resolved_line_numbers"] == [50]
    assert result["refined_from_line_numbers"] == [10, 50]
    assert result["evidence_section"] == "Results"

    # A quote that NEEDS the banned line stays flagged.
    needs_both = ground_evidence(
        "Vitamins attenuated VO2max gains after training. On the other hand, vitamins "
        "significantly attenuated VO2max by 3.11.",
        ["L10", "L50"],
        content,
    )
    assert needs_both["grounding_status"] == "exact"
    assert needs_both["section_flagged"] is True
    assert "refined_from_line_numbers" not in needs_both


def test_normalize_path_flag_refinement_rebinds_and_warns() -> None:
    from literature_multiverse.normalize import ground_evidence

    source = {
        "L10": {"line_id": "L10", "text": "Summary of gains.", "section": "Abstract"},
        "L50": {
            "line_id": "L50",
            "text": "Strength increased by 12 percent in the placebo group.",
            "section": "Results",
        },
    }
    result = ground_evidence(
        "Strength increased by 12 percent in the placebo group.", ["L10", "L50"], source
    )
    assert result.status == "exact"
    assert result.section_flagged is False
    assert result.refined_line_ids == ("L50",)
