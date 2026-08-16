from __future__ import annotations

import json
from copy import deepcopy

import pytest

from literature_multiverse.resampling import (
    ResamplingContractError,
    build_variant_a_headline,
    build_variant_b_headline,
    canonical_sha256,
    checkpoint_is_genuinely_incomplete,
    comparison_statistics,
    evaluate_m4_gate,
    frozen_checkpoint_wrapper,
    paper_permutation,
    summarize_westfall_young,
    write_content_addressed_checkpoint,
)


def _row(paper: str, direction: str, regime: str) -> dict[str, object]:
    return {
        "finding_id": f"{paper}:{direction}:{regime}",
        "paper_id": paper,
        "effect_direction": direction,
        "mod__regime": regime,
    }


def _material_rows() -> list[dict[str, object]]:
    rows = [_row(f"high-{index}", "increase", "high") for index in range(5)]
    rows.extend(_row(f"low-{index}", "decrease", "low") for index in range(4))
    rows.append(_row("low-4", "increase", "low"))
    return rows


def _passing_result() -> dict[str, object]:
    comparison = comparison_statistics(
        _material_rows(), "mod__regime", config_level_order=["high", "low"]
    )
    return {
        "moderator": "regime",
        "k": 5,
        "delta_ll": 0.2,
        "positive_folds": 5,
        "westfall_young_p": 0.03,
        "narrated_level_support": [5, 5],
        "comparison": comparison,
        "stability": {
            "pattern_fraction": 0.8,
            "eligible_fraction": 0.9,
            "top3_fraction": 0.85,
            "n_bootstraps": 200,
        },
        "all_valid_sensitivity": {
            "positive_gain": True,
            "directions_preserved": True,
        },
    }


def test_hand_calculated_same_subset_q_p_d_and_contrast() -> None:
    result = comparison_statistics(
        _material_rows(), "mod__regime", config_level_order=["high", "low"]
    )
    assert result["eligible"] is True
    assert result["coverage_papers"] == 1.0
    assert result["global_mode"] == "increase"
    assert result["agreement_q"] == pytest.approx(0.6)
    assert result["agreement_p"] == pytest.approx(0.9)
    assert result["absolute_gain"] == pytest.approx(0.3)
    assert result["contrast"] == {
        "level_a": "high",
        "direction_a": "increase",
        "n_papers_a": 5,
        "level_b": "low",
        "direction_b": "decrease",
        "n_papers_b": 5,
    }


def test_tied_global_or_regime_mode_is_not_headline_eligible() -> None:
    global_tie = [_row(f"h{i}", "increase", "high") for i in range(5)] + [
        _row(f"l{i}", "decrease", "low") for i in range(5)
    ]
    assert comparison_statistics(global_tie, "mod__regime")["reason"] == "global_mode_tied"

    regime_tie = _material_rows()
    regime_tie.extend(
        [
            _row("middle-0", "increase", "middle"),
            _row("middle-1", "increase", "middle"),
            _row("middle-2", "decrease", "middle"),
            _row("middle-3", "decrease", "middle"),
            _row("middle-4", "no_effect", "middle"),
        ]
    )
    result = comparison_statistics(
        regime_tie, "mod__regime", config_level_order=["high", "low", "middle"]
    )
    assert result["reason"] == "regime_mode_tied"


def test_paper_permutation_preserves_missingness_and_paper_constancy() -> None:
    rows = [
        _row("p1", "increase", "high"),
        dict(_row("p1", "increase", "high"), finding_id="p1:second"),
        _row("p2", "decrease", "low"),
        _row("p3", "no_effect", None),
    ]
    permuted = paper_permutation(rows, "mod__regime", seed=2)
    assert permuted[0]["mod__regime"] == permuted[1]["mod__regime"]
    assert permuted[3]["mod__regime"] is None
    assert sorted([permuted[0]["mod__regime"], permuted[2]["mod__regime"]]) == ["high", "low"]


def test_westfall_young_uses_complete_family_max_and_add_one_formula() -> None:
    observed = {"signal": 0.5, "null": 0.3}
    permutations = [{"signal": 0.1, "null": 0.6 if index < 9 else 0.1} for index in range(100)]
    result = summarize_westfall_young(observed, permutations)
    assert result["status"] == "complete"
    assert result["success_count"] == 100
    assert result["p_values"]["signal"]["raw"] == pytest.approx(1 / 101)
    assert result["p_values"]["signal"]["westfall_young"] == pytest.approx(10 / 101)
    assert result["p_values"]["signal"]["westfall_young"] >= result["p_values"]["signal"]["raw"]


def test_fewer_than_100_successes_after_125_attempts_is_completed_failed_rule() -> None:
    scores = [{"signal": 0.0}] * 99 + [None] * 26
    result = summarize_westfall_young({"signal": 0.2}, scores)
    assert result["status"] == "complete"
    assert result["reason"] == "insufficient_successes"
    assert result["attempt_count"] == 125
    assert result["success_count"] == 99
    assert result["p_values"]["signal"] == {"raw": None, "westfall_young": None}


def test_every_m4_rule_is_atomic_and_any_flip_selects_variant_b() -> None:
    passing = _passing_result()
    gate = evaluate_m4_gate([passing], config_order=["regime"])
    assert gate["selected_variant"] == "A"
    assert gate["selected_moderator"] == "regime"

    flipped = deepcopy(passing)
    flipped["comparison"]["absolute_gain"] = 0.099
    failed = evaluate_m4_gate([flipped], config_order=["regime"])
    assert failed["selected_variant"] == "B"
    assert failed["selection_reason"] == "m4_no_moderator"
    assert "material_gain" in failed["moderators"][0]["failed_rules"]

    interrupted = evaluate_m4_gate(
        [passing], config_order=["regime"], completion_status="incomplete"
    )
    assert interrupted["status"] == "incomplete"
    assert interrupted["selected_variant"] == "B"
    assert interrupted["moderators"][0]["passed"] is None


def test_headline_numbers_are_copied_from_same_rows_and_literal_branch() -> None:
    passing = _passing_result()
    headline = build_variant_a_headline(passing, moderator_display_name="dose regime")
    assert headline["narrative_variant"] == "A"
    assert headline["global_baseline"]["agreement_q"] == pytest.approx(0.6)
    assert headline["within_regime"]["absolute_gain"] == pytest.approx(0.3)
    assert "(+30 points)" in headline["rendered_sentence"]
    assert "p=0.030" in headline["rendered_sentence"]

    residuals = {"pair_count": 0, "top_pair_id": None, "rendered_sentence": "empty"}
    b_headline = build_variant_b_headline(
        selection_reason="m4_no_moderator",
        disagreement={
            "n_papers": 20,
            "n_findings": 25,
            "paper_entropy": 0.5,
            "interval_90": [0.4, 0.6],
        },
        residuals=residuals,
        sparse_or_empty_cells=3,
        total_cells=8,
    )
    assert b_headline["narrative_variant"] == "B"
    assert b_headline["m4"]["status"] == "failed"
    assert (
        "no pre-specified moderator passed every pre-registered" in b_headline["rendered_sentence"]
    )


def _checkpoint() -> dict[str, object]:
    digest = "a" * 64
    return {
        "checkpoint_version": "1",
        "checkpoint_status": "running_snapshot",
        "source_run_id": "run-1",
        "started_at": "2026-08-15T12:00:00-07:00",
        "checkpointed_at": "2026-08-15T12:10:00-07:00",
        "question_id": "fixture-b-incomplete",
        "config_sha256": digest,
        "code_version": "test",
        "cohort_sha256": digest,
        "g3_sha256": digest,
        "input_sha256s": {"findings": digest},
        "seed": 20260815,
        "budgets": {
            "bootstrap_count": 200,
            "permutation_successes": 100,
            "permutation_max_attempts": 125,
        },
        "completed_bootstrap_indices": list(range(25)),
        "bootstrap_results": [{"draw_index": index} for index in range(25)],
        "completed_permutation_attempt_indices": list(range(25)),
        "permutation_results": [{"attempt_index": index} for index in range(25)],
        "successful_permutation_indices": list(range(25)),
        "guard_failures": [],
        "artifact_sha256s": {"descriptive": digest, "residuals": digest, "gaps": digest},
    }


def test_checkpoint_is_content_addressed_atomic_and_wrapper_hash_is_separate(tmp_path) -> None:
    checkpoint = _checkpoint()
    assert checkpoint_is_genuinely_incomplete(checkpoint)
    path = write_content_addressed_checkpoint(checkpoint, tmp_path)
    assert path.name == f"{canonical_sha256(checkpoint)}.json"
    assert json.loads(path.read_text()) == checkpoint
    assert write_content_addressed_checkpoint(checkpoint, tmp_path) == path
    wrapper = frozen_checkpoint_wrapper(checkpoint)
    assert wrapper["source_checkpoint_sha256"] == canonical_sha256(wrapper["checkpoint"])
    assert canonical_sha256(wrapper) != wrapper["source_checkpoint_sha256"]


def test_terminal_checkpoint_cannot_take_incomplete_finalization_path() -> None:
    checkpoint = _checkpoint()
    checkpoint["completed_bootstrap_indices"] = list(range(200))
    checkpoint["bootstrap_results"] = [{"draw_index": index} for index in range(200)]
    checkpoint["completed_permutation_attempt_indices"] = list(range(100))
    checkpoint["permutation_results"] = [{"attempt_index": index} for index in range(100)]
    checkpoint["successful_permutation_indices"] = list(range(100))
    assert checkpoint_is_genuinely_incomplete(checkpoint) is False
    with pytest.raises(ResamplingContractError):
        frozen_checkpoint_wrapper(checkpoint)
