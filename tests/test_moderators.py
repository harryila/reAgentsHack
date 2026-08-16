from __future__ import annotations

import numpy as np
import pytest

from literature_multiverse.moderators import (
    OTHER_LEVEL,
    align_and_floor_probabilities,
    evaluate_all_moderators,
    evaluate_moderator,
    feasible_grouped_splits,
    laplace_prior,
    pool_rare_levels,
)


def _row(paper: str, direction: str, level: str | None) -> dict[str, object]:
    return {
        "finding_id": f"{paper}:{direction}:{level}",
        "paper_id": paper,
        "effect_direction": direction,
        "mod__regime": level,
        "mod__null_mod": "constant" if level is not None else None,
    }


def _informative_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(6):
        rows.append(_row(f"inc-{index}", "increase", "high"))
        rows.append(_row(f"dec-{index}", "decrease", "low"))
        rows.append(_row(f"nul-{index}", "no_effect", "middle"))
    return rows


def test_grouped_cv_never_crosses_papers_and_scores_same_test_subset() -> None:
    result = evaluate_moderator(_informative_rows(), "regime", seed=17)
    assert result["status"] == "eligible"
    assert result["k"] >= 3
    assert result["delta_ll"] > 0
    for fold in result["folds"]:
        assert set(fold["train_papers"]).isdisjoint(fold["test_papers"])
        assert fold["n_test_papers"] == pytest.approx(fold["test_weight_sum"])
        assert fold["model_log_loss"] >= 0
        assert fold["baseline_log_loss"] >= 0


def test_probability_alignment_floors_absent_canonical_column() -> None:
    aligned = align_and_floor_probabilities(
        np.asarray([[0.9, 0.1], [0.2, 0.8]]), ["increase", "decrease"]
    )
    assert aligned.shape == (2, 3)
    assert np.all(aligned[:, 1] > 0)
    assert np.allclose(aligned.sum(axis=1), 1.0)
    prior = laplace_prior(["increase", "decrease"], [1.0, 1.0])
    assert prior.shape == (3,)
    assert prior.sum() == pytest.approx(1.0)


def test_binary_subset_still_uses_full_probabilities_but_rare_class_blocks_cv() -> None:
    binary = [_row(f"i{index}", "increase", "high") for index in range(5)] + [
        _row(f"d{index}", "decrease", "low") for index in range(5)
    ]
    binary_result = evaluate_moderator(binary, "regime", seed=3)
    assert binary_result["status"] == "eligible"
    assert binary_result["labels_observed"] == ["increase", "decrease"]

    labels = ["increase"] * 5 + ["decrease"]
    papers = [f"p{index}" for index in range(6)]
    split = feasible_grouped_splits(labels, papers, seed=3)
    assert split["status"] == "insufficient_for_cv"
    assert split["splits"] == []


def test_outcome_correlated_missingness_cannot_create_same_subset_gain() -> None:
    rows = [_row(f"i{index}", "increase", "reported") for index in range(5)] + [
        _row(f"d{index}", "decrease", None) for index in range(5)
    ]
    result = evaluate_moderator(rows, "regime", seed=5)
    assert result["n_papers"] == 5
    assert result["status"] == "insufficient_for_cv"
    assert result["delta_ll"] is None


def test_duplicating_fifteen_finding_paper_is_numerically_invariant() -> None:
    rows = _informative_rows()
    target = rows[0]
    duplicated = rows + [dict(target, finding_id=f"duplicate-{index}") for index in range(14)]
    before = evaluate_moderator(rows, "regime", seed=11)
    after = evaluate_moderator(duplicated, "regime", seed=11)
    assert before["k"] == after["k"]
    assert before["delta_ll"] == pytest.approx(after["delta_ll"], abs=1e-10)
    assert before["positive_folds"] == after["positive_folds"]


def test_rare_level_repeated_in_one_paper_is_still_pooled() -> None:
    values = ["rare"] * 15 + ["common"] * 3
    papers = ["one-paper"] * 15 + ["p2", "p3", "p4"]
    pooled = pool_rare_levels(values, papers)
    assert pooled["values"][:15] == [OTHER_LEVEL] * 15
    assert pooled["pooled_original_levels"] == ["rare"]
    assert pooled["distinct_papers_by_original_level"]["rare"] == 1


def test_all_configured_moderators_remain_in_output() -> None:
    results = evaluate_all_moderators(
        _informative_rows(),
        [
            {"name": "regime", "kind": "paper_constant", "role": "tested", "permutation": "paper"},
            {
                "name": "null_mod",
                "kind": "paper_constant",
                "role": "tested",
                "permutation": "paper",
            },
            {"name": "within", "kind": "within_paper", "role": "tested", "permutation": "none"},
        ],
        seed=13,
    )
    assert [result["moderator"] for result in results] == ["regime", "null_mod", "within"]
    assert results[1]["status"] == "insufficient_for_cv"
    assert results[2]["status"] == "descriptive_only"
