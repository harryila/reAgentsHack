from __future__ import annotations

import pytest

from literature_multiverse.contradictions import (
    ContradictionContractError,
    find_contradictions,
    residual_summary,
)
from literature_multiverse.evidence_gaps import build_evidence_gap_grid


def _paper(paper_id: str) -> dict[str, object]:
    return {
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "doi": f"10.1/{paper_id}",
        "doc_id": f"doc-{paper_id}",
    }


def _finding(
    finding_id: str,
    paper_id: str,
    direction: str,
    *,
    dose: str = "high",
    endpoint: str = "fatigue",
) -> dict[str, object]:
    return {
        "finding_id": finding_id,
        "paper_id": paper_id,
        "effect_direction": direction,
        "outcome_family": "performance",
        "outcome_name": endpoint,
        "intervention_class": "vitamin",
        "comparator": "placebo",
        "population_state": "trained",
        "timing_context": "pre",
        "timepoint_raw": "week 4",
        "mod__dose": dose,
        "grounding_status": "exact",
        "section_flagged": False,
        "evidence_quote": f"quote {finding_id}",
        "evidence_lines": ["L10-L11"],
        "evidence_section": "Results",
    }


def test_contradictions_require_opposite_directions_and_preserve_both_citations() -> None:
    papers = [_paper("p1"), _paper("p2"), _paper("p3")]
    rows = [
        _finding("f1", "p1", "increase"),
        _finding("f2", "p2", "decrease"),
        _finding("f3", "p3", "no_effect"),
    ]
    pairs = find_contradictions(rows, papers, primary_family="performance")
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["shared_context_count"] >= 2
    assert pair["left_citation"]["finding_id"] == "f1"
    assert pair["right_citation"]["finding_id"] == "f2"
    assert pair["left_citation"]["title"] == "Paper p1"
    assert pair["right_citation"]["title"] == "Paper p2"
    assert pair["distance_components"]
    summary = residual_summary(pairs)
    assert summary["pair_count"] == 1
    assert summary["top_pair_id"] == pair["pair_id"]
    assert "both source passages" in summary["rendered_sentence"]


def test_same_paper_secondary_family_and_section_flagged_rows_do_not_pair() -> None:
    papers = [_paper("p1"), _paper("p2")]
    left = _finding("f1", "p1", "increase")
    same_paper = _finding("f2", "p1", "decrease")
    secondary = _finding("f3", "p2", "decrease")
    secondary["outcome_family"] = "secondary"
    flagged = _finding("f4", "p2", "decrease")
    flagged["section_flagged"] = True
    assert (
        find_contradictions(
            [left, same_paper, secondary, flagged], papers, primary_family="performance"
        )
        == []
    )
    empty = residual_summary([])
    assert empty["top_pair_id"] is None
    assert "empty residual view" in empty["rendered_sentence"]


def test_orphan_citation_fails_instead_of_rendering() -> None:
    with pytest.raises(ContradictionContractError):
        find_contradictions(
            [_finding("f1", "missing", "increase")],
            [_paper("p1")],
            primary_family="performance",
        )


def test_evidence_gap_grid_keeps_zero_cells_and_exact_status_boundaries() -> None:
    rows = [_finding("a1", "pa", "increase", dose="one")]
    rows.extend(
        _finding(f"c{i}", f"pc{i}", "increase" if i < 3 else "decrease", dose="five")
        for i in range(5)
    )
    grid = build_evidence_gap_grid(
        rows,
        axis_levels={"dose": ["zero", "one", "five"]},
        primary_endpoints=["fatigue"],
    )
    assert len(grid) == 3
    by_level = {row["axis_values"]["dose"]: row for row in grid}
    assert by_level["zero"]["status"] == "empty"
    assert by_level["zero"]["n_findings"] == 0
    assert by_level["zero"]["paper_entropy"] is None
    assert by_level["one"]["status"] == "sparse"
    assert by_level["one"]["n_papers_grounded"] == 1
    assert by_level["one"]["paper_entropy"] is None
    assert by_level["five"]["status"] == "supported"
    assert by_level["five"]["n_papers_grounded"] == 5
    assert by_level["five"]["paper_entropy"] is not None


def test_evidence_grid_is_frozen_cartesian_not_observation_inferred() -> None:
    rows = [_finding("f1", "p1", "increase", dose="low", endpoint="fatigue")]
    rows[0]["mod__timing"] = "pre"
    grid = build_evidence_gap_grid(
        rows,
        axis_levels={"dose": ["low", "high"], "timing": ["pre", "post"]},
        primary_endpoints=["fatigue", "strength"],
    )
    assert len(grid) == 8
    assert sum(row["n_findings"] == 0 for row in grid) == 7
    assert all(
        set(row)
        == {
            "cell_id",
            "primary_endpoint",
            "axis_values",
            "n_papers_total",
            "n_papers_grounded",
            "n_findings",
            "grounded_fraction",
            "classifiable_fraction",
            "paper_entropy",
            "status",
        }
        for row in grid
    )
