from __future__ import annotations

from literature_multiverse.meta_simulation import (
    simulate_meta_replicate,
    summarize_meta_simulations,
)


def test_meta_simulation_is_deterministic() -> None:
    first = simulate_meta_replicate(seed=17, moderator_effect=0.35, papers_per_level=8)
    second = simulate_meta_replicate(seed=17, moderator_effect=0.35, papers_per_level=8)

    assert first == second


def test_precision_confounding_exposes_significance_vote_failure() -> None:
    null_rows = [
        simulate_meta_replicate(
            seed=100 + index,
            moderator_effect=0.0,
            papers_per_level=20,
            heldout_papers_per_level=20,
        )
        for index in range(40)
    ]
    moderator_rows = [
        simulate_meta_replicate(
            seed=200 + index,
            moderator_effect=0.35,
            papers_per_level=20,
            heldout_papers_per_level=20,
        )
        for index in range(40)
    ]
    summary = summarize_meta_simulations(null_rows, moderator_rows, alpha=0.05)
    null = summary["null_moderator"]
    planted = summary["planted_moderator"]

    assert null["meta_detection_rate"] < null["significance_vote_detection_rate"]
    assert planted["meta_detection_rate"] > planted["significance_vote_detection_rate"]
    assert null["meta_mean_heldout_brier"] < null["significance_vote_mean_heldout_brier"]
    assert (
        planted["meta_mean_heldout_brier"]
        < planted["significance_vote_mean_heldout_brier"]
    )

