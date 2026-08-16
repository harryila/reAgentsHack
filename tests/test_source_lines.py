"""Section-derivation contracts for provider content.lines parsing."""

from __future__ import annotations

import pytest

from literature_multiverse.source_lines import SourceLinesParseError, parse_content_lines

SAMPLE = "\n".join(
    [
        "L1: A Randomized Trial of Something",
        "L2: Journal of Examples (2024)",
        "L3: Alice Author, Bob Author",
        "L4: Background:  We asked a question.",
        "L5: Results:  We found an answer.",
        "L6: 1. Introduction",
        "L7: Prior work exists.",
        "L8: 2. Materials and methods",
        "L9: We recruited participants.",
        "L10: 2.1. Outcomes",
        "L11: We measured strength.",
        "L12: 3. Results",
        "L13: Strength increased (P = .01).",
        "L14: 4. Discussion",
        "L15: This aligns with prior work.",
        "L16: 5. Conclusion",
        "L17: It worked.",
        "L18: Acknowledgments",
        "L19: We thank everyone.",
        "L20: References",
        "L21: 1. Prior A, et al. An earlier trial. J Ex 2019.",
        "L22: Methods",
        "L23: Supplement text that echoes a heading.",
    ]
)


def test_sections_resolve_and_backmatter_is_terminal() -> None:
    parsed = parse_content_lines(SAMPLE)
    assert parsed["L1"]["section"] == "Abstract"
    # Deliberate semantic (matches the frozen G1 contract): a structured-abstract line
    # prefixed "Results:" reports the paper's own measured result and is labeled Results;
    # quote⇒direction verification remains the correctness backstop for such citations.
    assert parsed["L5"]["section"] == "Results"
    assert parsed["L7"]["section"] == "Introduction"
    assert parsed["L9"]["section"] == "Methods"
    assert parsed["L11"]["section"] == "Methods"  # 2.x inherits the major-2 mapping
    assert parsed["L13"]["section"] == "Results"
    assert parsed["L15"]["section"] == "Discussion"
    assert parsed["L17"]["section"] == "Conclusion"
    # Once back matter starts, nothing re-opens a body section — not even an echoed
    # "Methods" heading inside a supplement.
    for number in (19, 21, 22, 23):
        assert parsed[f"L{number}"]["section"] == "References"


def test_provider_banners_are_skipped_and_empty_input_fails() -> None:
    parsed = parse_content_lines("[~19414 tokens total]\n" + SAMPLE)
    assert parsed["L13"]["section"] == "Results"
    with pytest.raises(SourceLinesParseError):
        parse_content_lines("no numbered lines here")


def test_captions_get_result_bearing_label_and_references_never_reopen_body() -> None:
    """Observed live (PMC3303472): figure captions serialize after the Discussion, and
    numbered reference entries whose titles contain section keywords ("an introduction")
    were relabeling the bibliography as body text."""
    from literature_multiverse.source_lines import parse_content_lines

    raw = "\n".join(
        [
            "L1: Some Trial Title",
            "L2: Abstract",
            "L3: We studied things.",
            "L4: 2. Results",
            "L5: VO2max decreased in the vitamin group.",
            "L6: Discussion",
            "L7: We believe this happened because reasons.",
            "L8: Fig. 3 Change in maximal oxygen consumption rate. The vitamin group"
            " decreased significantly.",
            "L9: 2 Packer L Cadenas E Davies KJ Free radicals and exercise: an introduction"
            " Free Radic Biol Med 2008 44 123",
            "L10: 3 Alessio HM Exercise-induced oxidative stress Med Sci Sports Exerc 1993 25 218",
        ]
    )
    parsed = parse_content_lines(raw)
    assert parsed["L5"]["section"] == "Results"
    assert parsed["L7"]["section"] == "Discussion"
    assert parsed["L8"]["section"] == "FigureTable"
    # Reference entries inherit the (banned) Discussion instead of re-opening body
    # sections through their leading numbers or title keywords.
    assert parsed["L9"]["section"] == "Discussion"
    assert parsed["L10"]["section"] == "Discussion"
