from __future__ import annotations

import math

import pytest

from literature_multiverse.disagreement import (
    bootstrap_disagreement,
    evaluate_g2,
    evaluate_g3,
    normalized_entropy,
    paper_balanced_finding_summary,
    paper_modal_summary,
)


def _row(paper: str, direction: str) -> dict[str, str]:
    return {"paper_id": paper, "effect_direction": direction}


def test_hand_computable_entropy_extremes() -> None:
    unanimous = paper_balanced_finding_summary([_row("p1", "increase"), _row("p2", "increase")])
    balanced = paper_balanced_finding_summary(
        [_row("p1", "increase"), _row("p2", "no_effect"), _row("p3", "decrease")]
    )
    assert unanimous["normalized_entropy"] == pytest.approx(0.0)
    assert balanced["normalized_entropy"] == pytest.approx(1.0)
    assert normalized_entropy({"a": 1, "b": 1, "c": 1}, denominator_classes=3) == pytest.approx(1.0)


def test_duplicating_all_rows_from_one_paper_does_not_change_paper_balance() -> None:
    rows = [
        _row("p1", "increase"),
        _row("p1", "decrease"),
        _row("p2", "no_effect"),
    ]
    duplicated = rows + [dict(rows[0]), dict(rows[1])] * 7
    before = paper_balanced_finding_summary(rows)
    after = paper_balanced_finding_summary(duplicated)
    assert before["class_proportions"] == pytest.approx(after["class_proportions"])
    assert before["normalized_entropy"] == pytest.approx(after["normalized_entropy"])


def test_paper_modal_tie_is_excluded_primary_and_unresolved_in_log4_sensitivity() -> None:
    summary = paper_modal_summary(
        [
            _row("p1", "increase"),
            _row("p1", "decrease"),
            _row("p2", "increase"),
            _row("p3", "no_effect"),
            _row("p4", "decrease"),
        ]
    )
    assert summary["paper_modes"]["p1"] == "mixed"
    assert summary["n_papers_tied"] == 1
    assert summary["primary"]["class_counts"] == {
        "increase": 1,
        "no_effect": 1,
        "decrease": 1,
    }
    sensitivity = summary["unresolved_sensitivity"]
    assert sensitivity["class_counts"]["unresolved"] == 1
    assert sensitivity["denominator_log_classes"] == 4
    assert sensitivity["normalized_entropy"] == pytest.approx(1.0)
    assert sensitivity["eligible_for_primary_gate"] is False


def test_bootstrap_is_fixed_seed_and_retains_registered_draw_count() -> None:
    rows = [
        _row("p1", "increase"),
        _row("p2", "no_effect"),
        _row("p3", "decrease"),
        _row("p4", "increase"),
    ]
    first = bootstrap_disagreement(rows, n_bootstraps=50, seed=7)
    second = bootstrap_disagreement(rows, n_bootstraps=50, seed=7)
    assert first == second
    assert len(first["paper_entropy"]["draws"]) == 50
    assert first["paper_entropy"]["invalid_draws"] >= 0


def test_g2_upper_bound_can_pass_while_g3_lower_bound_fails() -> None:
    g2 = evaluate_g2(
        entropy_point=0.35,
        entropy_interval_90=(0.10, 0.45),
        distinct_papers_by_direction={"increase": 4, "no_effect": 2, "decrease": 1},
        relation_purity=0.8,
        estimated_usable_primaries=44,
    )
    g3 = evaluate_g3(
        audit_correct=20,
        audit_total=20,
        anchors_passed=2,
        anchors_total=2,
        cross_model_agreement=0.9,
        quarantine_fraction=0.02,
        g1b_passed=True,
        paper_entropy_interval_90=(0.39, 0.7),
        classifiable_papers=25,
        distinct_papers_by_direction={"increase": 10, "no_effect": 10, "decrease": 5},
    )
    assert g2["passed"] is True
    assert g3["trust_passed"] is True
    assert g3["story_passed"] is False
    assert g3["action"] == "select_variant_b_story"


def test_g3_zero_quarantine_denominator_cannot_pass_trust() -> None:
    result = evaluate_g3(
        audit_correct=20,
        audit_total=20,
        anchors_passed=1,
        anchors_total=1,
        cross_model_agreement=1.0,
        quarantine_fraction=None,
        g1b_passed=True,
        paper_entropy_interval_90=(0.5, 0.8),
        classifiable_papers=30,
        distinct_papers_by_direction={"increase": 10, "no_effect": 10, "decrease": 10},
    )
    assert result["trust_passed"] is False
    assert result["action"] == "block_release"


def test_empty_entropy_mass_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalized_entropy({"increase": 0.0}, denominator_classes=3)
    assert math.log(3) > 0
