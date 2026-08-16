"""Paper-aware resampling, same-subset headline statistics, and the exact M4 gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from literature_multiverse.disagreement import CANONICAL_LABELS, paper_balanced_weights

CHECKPOINT_KEYS = frozenset(
    {
        "checkpoint_version",
        "checkpoint_status",
        "source_run_id",
        "started_at",
        "checkpointed_at",
        "question_id",
        "config_sha256",
        "code_version",
        "cohort_sha256",
        "g3_sha256",
        "input_sha256s",
        "seed",
        "budgets",
        "completed_bootstrap_indices",
        "bootstrap_results",
        "completed_permutation_attempt_indices",
        "permutation_results",
        "successful_permutation_indices",
        "guard_failures",
        "artifact_sha256s",
    }
)


class ResamplingContractError(ValueError):
    """Raised when a registered resampling or M4 contract is violated."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize an object using the checkpoint/source hashing convention."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def paper_permutation(
    rows: Sequence[Mapping[str, Any]],
    moderator_key: str,
    *,
    seed: int,
    attempt_index: int = 0,
) -> list[dict[str, Any]]:
    """Shuffle a paper-constant value among non-missing papers and replicate to findings."""

    by_paper: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        paper_id = row.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            raise ResamplingContractError("paper permutation requires paper_id")
        by_paper[paper_id].append(row)
    values: dict[str, Any] = {}
    missing: set[str] = set()
    for paper_id, paper_rows in by_paper.items():
        observed = {
            row.get(moderator_key) for row in paper_rows if row.get(moderator_key) is not None
        }
        if len(observed) > 1:
            raise ResamplingContractError(f"{moderator_key} is not paper-constant for {paper_id}")
        if observed:
            values[paper_id] = next(iter(observed))
        else:
            missing.add(paper_id)
    nonmissing_papers = sorted(values)
    shuffled_values = [values[paper_id] for paper_id in nonmissing_papers]
    random.Random(f"{seed}:{attempt_index}:{moderator_key}").shuffle(shuffled_values)
    permuted = dict(zip(nonmissing_papers, shuffled_values, strict=True))
    output: list[dict[str, Any]] = []
    for row in rows:
        clone = dict(row)
        paper_id = clone["paper_id"]
        clone[moderator_key] = None if paper_id in missing else permuted[paper_id]
        output.append(clone)
    return output


def summarize_westfall_young(
    observed_scores: Mapping[str, float],
    permutation_scores: Sequence[Mapping[str, float] | None],
    *,
    required_successes: int = 100,
    max_attempts: int = 125,
) -> dict[str, Any]:
    """Compute add-one raw and single-step max-statistic adjusted p-values."""

    if not observed_scores:
        raise ResamplingContractError("Westfall-Young family cannot be empty")
    if required_successes < 1 or max_attempts < required_successes:
        raise ResamplingContractError("invalid permutation budget")
    family = tuple(observed_scores)
    if any(not math.isfinite(float(score)) for score in observed_scores.values()):
        raise ResamplingContractError("observed permutation scores must be finite")
    successes: list[Mapping[str, float]] = []
    failures: list[dict[str, Any]] = []
    attempts = 0
    for attempt, result in enumerate(permutation_scores[:max_attempts]):
        attempts += 1
        if (
            result is None
            or set(result) != set(family)
            or any(not math.isfinite(float(result[name])) for name in family)
        ):
            failures.append({"attempt_index": attempt, "reason": "invalid_family_result"})
            continue
        successes.append(result)
        if len(successes) == required_successes:
            break
    complete_budget = len(successes) == required_successes or attempts == max_attempts
    if not complete_budget:
        status = "incomplete"
        reason = "permutation_interrupted"
    else:
        status = "complete"
        reason = None if len(successes) == required_successes else "insufficient_successes"
    p_values: dict[str, dict[str, float | None]] = {}
    for name, observed in observed_scores.items():
        if len(successes) < required_successes:
            p_values[name] = {"raw": None, "westfall_young": None}
            continue
        raw_exceedances = sum(result[name] >= observed for result in successes)
        adjusted_exceedances = sum(max(result.values()) >= observed for result in successes)
        denominator = len(successes) + 1
        p_values[name] = {
            "raw": (1 + raw_exceedances) / denominator,
            "westfall_young": (1 + adjusted_exceedances) / denominator,
        }
    return {
        "status": status,
        "reason": reason,
        "required_successes": required_successes,
        "max_attempts": max_attempts,
        "attempt_count": attempts,
        "success_count": len(successes),
        "guard_failures": failures,
        "family": list(family),
        "p_values": p_values,
    }


def paper_bootstrap_sample(
    rows: Sequence[Mapping[str, Any]], *, seed: int, draw_index: int
) -> list[dict[str, Any]]:
    """Sample whole papers with replacement, retaining source group identity separately."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        paper_id = row.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            raise ResamplingContractError("paper bootstrap requires paper_id")
        grouped[paper_id].append(row)
    paper_ids = sorted(grouped)
    if not paper_ids:
        raise ResamplingContractError("paper bootstrap requires at least one paper")
    rng = random.Random(f"{seed}:{draw_index}:paper-bootstrap")
    selected = [rng.choice(paper_ids) for _ in paper_ids]
    output: list[dict[str, Any]] = []
    for occurrence, source_paper in enumerate(selected):
        instance_id = f"{source_paper}#bootstrap#{occurrence}"
        for row in grouped[source_paper]:
            clone = dict(row)
            clone["source_paper_id"] = source_paper
            clone["paper_id"] = instance_id
            output.append(clone)
    return output


def run_paper_bootstraps(
    rows: Sequence[Mapping[str, Any]],
    evaluator: Callable[[Sequence[Mapping[str, Any]], int], Mapping[str, Any]],
    *,
    n_bootstraps: int = 200,
    seed: int = 20260815,
    checkpoint_every: int = 25,
    checkpoint_callback: Callable[[int, Sequence[Mapping[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    """Run every registered draw; guarded failures remain in the denominator."""

    if n_bootstraps < 1 or checkpoint_every < 1:
        raise ResamplingContractError("bootstrap and checkpoint budgets must be positive")
    results: list[dict[str, Any]] = []
    for draw_index in range(n_bootstraps):
        try:
            sample = paper_bootstrap_sample(rows, seed=seed, draw_index=draw_index)
            value = dict(evaluator(sample, draw_index))
            results.append({"draw_index": draw_index, "status": "success", "result": value})
        except Exception as exc:  # Guarded scientific draw, deliberately serialized.
            results.append(
                {
                    "draw_index": draw_index,
                    "status": "error",
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
        if checkpoint_callback and (draw_index + 1) % checkpoint_every == 0:
            checkpoint_callback(draw_index + 1, results)
    return {
        "status": "complete",
        "seed": seed,
        "n_bootstraps": n_bootstraps,
        "success_count": sum(result["status"] == "success" for result in results),
        "error_count": sum(result["status"] == "error" for result in results),
        "draws": results,
    }


def _unique_weighted_mode(
    rows: Sequence[Mapping[str, Any]], weights: Sequence[float]
) -> tuple[str | None, float, dict[str, float]]:
    mass = {label: 0.0 for label in CANONICAL_LABELS}
    for row, weight in zip(rows, weights, strict=True):
        mass[row["effect_direction"]] += weight
    maximum = max(mass.values())
    winners = [
        label for label in CANONICAL_LABELS if math.isclose(mass[label], maximum, abs_tol=1e-12)
    ]
    return (winners[0] if len(winners) == 1 else None, maximum, mass)


def comparison_statistics(
    rows: Sequence[Mapping[str, Any]],
    moderator_key: str,
    *,
    config_level_order: Sequence[Any] | None = None,
    minimum_level_papers: int = 5,
) -> dict[str, Any]:
    """Compute exact same-subset Q, P, D, unique modes, and support-max contrast."""

    primary_rows = [row for row in rows if row.get("effect_direction") in CANONICAL_LABELS]
    all_papers = {row.get("paper_id") for row in primary_rows if row.get("paper_id")}
    support: dict[Any, set[str]] = defaultdict(set)
    for row in primary_rows:
        value = row.get(moderator_key)
        if value is not None and value != "__OTHER__":
            support[value].add(row["paper_id"])
    supported_levels = {
        level for level, papers in support.items() if len(papers) >= minimum_level_papers
    }
    subset = [row for row in primary_rows if row.get(moderator_key) in supported_levels]
    if not subset:
        return {
            "eligible": False,
            "reason": "no_supported_levels",
            "n_findings": 0,
            "n_papers": 0,
            "coverage_papers": 0.0,
            "supported_levels": [],
        }
    weights = paper_balanced_weights(subset)
    global_mode, global_mass, class_mass = _unique_weighted_mode(subset, weights)
    total_weight = sum(weights)
    level_modes: dict[Any, dict[str, Any]] = {}
    for level in supported_levels:
        indices = [index for index, row in enumerate(subset) if row[moderator_key] == level]
        level_rows = [subset[index] for index in indices]
        level_weights = [weights[index] for index in indices]
        mode, mode_mass, mass = _unique_weighted_mode(level_rows, level_weights)
        level_modes[level] = {
            "modal_direction": mode,
            "unique": mode is not None,
            "n_papers": len({row["paper_id"] for row in level_rows}),
            "n_findings": len(level_rows),
            "agreement": mode_mass / sum(level_weights),
            "class_mass": mass,
        }
    q_value = global_mass / total_weight
    all_regimes_unique = all(item["unique"] for item in level_modes.values())
    p_numerator = 0.0
    if all_regimes_unique:
        for row, weight in zip(subset, weights, strict=True):
            if row["effect_direction"] == level_modes[row[moderator_key]]["modal_direction"]:
                p_numerator += weight
        p_value = p_numerator / total_weight
        d_value = p_value - q_value
    else:
        p_value = None
        d_value = None

    default_order = list(config_level_order or [])
    for level in sorted(supported_levels, key=str):
        if level not in default_order:
            default_order.append(level)
    order_index = {level: index for index, level in enumerate(default_order)}
    candidates: list[tuple[int, int, int, Any, Any]] = []
    for left_index, left in enumerate(default_order):
        if left not in level_modes or not level_modes[left]["unique"]:
            continue
        for right in default_order[left_index + 1 :]:
            if right not in level_modes or not level_modes[right]["unique"]:
                continue
            if level_modes[left]["modal_direction"] == level_modes[right]["modal_direction"]:
                continue
            combined = level_modes[left]["n_papers"] + level_modes[right]["n_papers"]
            candidates.append((-combined, order_index[left], order_index[right], left, right))
    candidates.sort()
    contrast = None
    if candidates:
        _, _, _, level_a, level_b = candidates[0]
        contrast = {
            "level_a": level_a,
            "direction_a": level_modes[level_a]["modal_direction"],
            "n_papers_a": level_modes[level_a]["n_papers"],
            "level_b": level_b,
            "direction_b": level_modes[level_b]["modal_direction"],
            "n_papers_b": level_modes[level_b]["n_papers"],
        }
    return {
        "eligible": global_mode is not None and all_regimes_unique and contrast is not None,
        "reason": (
            "global_mode_tied"
            if global_mode is None
            else "regime_mode_tied"
            if not all_regimes_unique
            else "no_differently_directed_regimes"
            if contrast is None
            else None
        ),
        "n_findings": len(subset),
        "n_papers": len({row["paper_id"] for row in subset}),
        "coverage_papers": len({row["paper_id"] for row in subset}) / len(all_papers)
        if all_papers
        else 0.0,
        "supported_levels": [level for level in default_order if level in supported_levels],
        "support_papers": {str(level): len(support[level]) for level in supported_levels},
        "global_mode": global_mode,
        "global_class_mass": class_mass,
        "agreement_q": q_value,
        "level_modes": {str(level): value for level, value in level_modes.items()},
        "agreement_p": p_value,
        "absolute_gain": d_value,
        "contrast": contrast,
    }


def bootstrap_contrast_stability(
    rows: Sequence[Mapping[str, Any]],
    moderator_key: str,
    expected_contrast: Mapping[str, Any],
    *,
    config_level_order: Sequence[Any] | None = None,
    n_bootstraps: int = 200,
    seed: int = 20260815,
    minimum_level_papers: int = 5,
) -> dict[str, Any]:
    """Measure recurrence over all draws, retaining ineligible resamples in the denominator."""

    expected = (
        expected_contrast.get("level_a"),
        expected_contrast.get("direction_a"),
        expected_contrast.get("level_b"),
        expected_contrast.get("direction_b"),
    )

    def evaluate(sample: Sequence[Mapping[str, Any]], _: int) -> Mapping[str, Any]:
        comparison = comparison_statistics(
            sample,
            moderator_key,
            config_level_order=config_level_order,
            minimum_level_papers=minimum_level_papers,
        )
        contrast = comparison.get("contrast") or {}
        observed = (
            contrast.get("level_a"),
            contrast.get("direction_a"),
            contrast.get("level_b"),
            contrast.get("direction_b"),
        )
        reversed_observed = (observed[2], observed[3], observed[0], observed[1])
        matches = observed == expected or reversed_observed == expected
        material = (
            comparison.get("absolute_gain") is not None and comparison["absolute_gain"] >= 0.10
        )
        return {
            "eligible": bool(comparison.get("eligible")),
            "reason": comparison.get("reason"),
            "contrast_matches": matches,
            "material": material,
            "pattern_recurred": bool(comparison.get("eligible") and matches and material),
            "absolute_gain": comparison.get("absolute_gain"),
        }

    battery = run_paper_bootstraps(rows, evaluate, n_bootstraps=n_bootstraps, seed=seed)
    successful_results = [
        draw["result"] for draw in battery["draws"] if draw["status"] == "success"
    ]
    eligible_count = sum(result["eligible"] for result in successful_results)
    pattern_count = sum(result["pattern_recurred"] for result in successful_results)
    return {
        "n_bootstraps": n_bootstraps,
        "pattern_count": pattern_count,
        "pattern_fraction": pattern_count / n_bootstraps,
        "eligible_count": eligible_count,
        "eligible_fraction": eligible_count / n_bootstraps,
        "error_count": battery["error_count"],
        "draws": battery["draws"],
    }


def _gate_rule(name: str, observed: Any, threshold: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "observed": observed, "threshold": threshold, "passed": bool(passed)}


def _unevaluated_m4_moderators(
    moderator_results: Sequence[Mapping[str, Any]], reason: str
) -> list[dict[str, Any]]:
    thresholds = (
        ("feasible_k", ">=3"),
        ("paper_coverage", ">=0.60"),
        ("narrated_level_support", "every narrated level >=5 papers"),
        ("mean_delta_ll", ">=0.02"),
        ("positive_folds", ">=ceil(0.6*k)"),
        ("westfall_young_p", "<0.10"),
        ("unique_global_mode", "unique"),
        ("different_unique_regime_modes", "two differently directed unique modes"),
        ("material_gain", ">=0.10"),
        ("bootstrap_pattern", ">=0.60 of all 200"),
        ("all_valid_positive_gain", "true"),
        ("all_valid_directions", "true"),
    )
    return [
        {
            "moderator": result.get("moderator"),
            "rules": [
                {
                    "name": name,
                    "observed": None,
                    "threshold": threshold,
                    "passed": None,
                    "reason": reason,
                }
                for name, threshold in thresholds
            ],
            "failed_rules": [],
            "passed": None,
            "status": "not_evaluated",
        }
        for result in moderator_results
    ]


def evaluate_m4_gate(
    moderator_results: Sequence[Mapping[str, Any]],
    *,
    config_order: Sequence[str],
    seed: int = 20260815,
    completion_status: str = "complete",
    g3_story_passed: bool = True,
) -> dict[str, Any]:
    """Apply every §6.3 rule atomically and select exactly one A/B branch."""

    if not g3_story_passed:
        return {
            "status": "not_run",
            "reason": "g3_story_not_viable",
            "selected_moderator": None,
            "selection_reason": "g3_story_not_viable",
            "selected_variant": "B",
            "seed": seed,
            "moderators": _unevaluated_m4_moderators(
                moderator_results, "g3_story_not_viable"
            ),
        }
    if completion_status == "incomplete":
        return {
            "status": "incomplete",
            "reason": "m4_incomplete",
            "selected_moderator": None,
            "selection_reason": "m4_incomplete",
            "selected_variant": "B",
            "seed": seed,
            "moderators": _unevaluated_m4_moderators(
                moderator_results, "m4_incomplete"
            ),
        }
    if completion_status != "complete":
        raise ResamplingContractError(f"invalid M4 completion status: {completion_status!r}")

    order = {name: index for index, name in enumerate(config_order)}
    evaluated: list[dict[str, Any]] = []
    passing: list[Mapping[str, Any]] = []
    for result in moderator_results:
        name = result.get("moderator")
        if name not in order:
            raise ResamplingContractError(f"M4 result is outside locked family: {name!r}")
        k = result.get("k")
        comparison = result.get("comparison") or {}
        stability = result.get("stability") or {}
        sensitivity = result.get("all_valid_sensitivity") or {}
        positive_required = math.ceil(0.6 * k) if isinstance(k, int) else None
        narrated_support = result.get("narrated_level_support", [])
        rules = [
            _gate_rule("feasible_k", k, ">=3", isinstance(k, int) and k >= 3),
            _gate_rule(
                "paper_coverage",
                comparison.get("coverage_papers"),
                ">=0.60",
                comparison.get("coverage_papers") is not None
                and comparison.get("coverage_papers") >= 0.60,
            ),
            _gate_rule(
                "narrated_level_support",
                narrated_support,
                "every narrated level >=5 papers",
                bool(narrated_support) and all(int(value) >= 5 for value in narrated_support),
            ),
            _gate_rule(
                "mean_delta_ll",
                result.get("delta_ll"),
                ">=0.02",
                result.get("delta_ll") is not None and result.get("delta_ll") >= 0.02,
            ),
            _gate_rule(
                "positive_folds",
                result.get("positive_folds"),
                f">={positive_required}",
                positive_required is not None
                and result.get("positive_folds") is not None
                and result.get("positive_folds") >= positive_required,
            ),
            _gate_rule(
                "westfall_young_p",
                result.get("westfall_young_p"),
                "<0.10",
                result.get("westfall_young_p") is not None
                and result.get("westfall_young_p") < 0.10,
            ),
            _gate_rule(
                "unique_global_mode",
                comparison.get("global_mode"),
                "unique",
                comparison.get("global_mode") is not None,
            ),
            _gate_rule(
                "different_unique_regime_modes",
                comparison.get("contrast"),
                "two differently directed unique modes",
                bool(comparison.get("eligible") and comparison.get("contrast")),
            ),
            _gate_rule(
                "material_gain",
                comparison.get("absolute_gain"),
                ">=0.10",
                comparison.get("absolute_gain") is not None
                and comparison.get("absolute_gain") >= 0.10,
            ),
            _gate_rule(
                "bootstrap_pattern",
                stability.get("pattern_fraction"),
                ">=0.60 of all 200",
                stability.get("n_bootstraps") == 200
                and stability.get("pattern_fraction") is not None
                and stability.get("pattern_fraction") >= 0.60,
            ),
            _gate_rule(
                "all_valid_positive_gain",
                sensitivity.get("positive_gain"),
                "true",
                sensitivity.get("positive_gain") is True,
            ),
            _gate_rule(
                "all_valid_directions",
                sensitivity.get("directions_preserved"),
                "true",
                sensitivity.get("directions_preserved") is True,
            ),
        ]
        passed = all(rule["passed"] for rule in rules)
        evaluated.append(
            {
                "moderator": name,
                "rules": rules,
                "failed_rules": [rule["name"] for rule in rules if not rule["passed"]],
                "passed": passed,
                "status": "passed" if passed else "failed",
            }
        )
        if passed:
            passing.append(result)
    if passing:
        winner = min(
            passing,
            key=lambda result: (
                -float(result["delta_ll"]),
                float(result["westfall_young_p"]),
                order[str(result["moderator"])],
            ),
        )
        variant = "A"
        selected = winner["moderator"]
        selection_reason = "m4_moderator_passed"
    else:
        variant = "B"
        selected = None
        selection_reason = "m4_no_moderator"
    return {
        "status": "complete",
        "reason": None,
        "selected_moderator": selected,
        "selection_reason": selection_reason,
        "selected_variant": variant,
        "seed": seed,
        "moderators": evaluated,
    }


def _whole_percent(value: float) -> str:
    rounded = (Decimal(str(value)) * Decimal(100)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return str(int(rounded))


def build_variant_a_headline(
    result: Mapping[str, Any], *, moderator_display_name: str | None = None
) -> dict[str, Any]:
    """Build the literal Variant-A headline branch and pre-approved sentence."""

    comparison = result["comparison"]
    stability = result["stability"]
    contrast = comparison["contrast"]
    moderator = str(result["moderator"])
    display = moderator_display_name or moderator
    sentence = (
        f"Among {comparison['n_papers']} papers ({comparison['n_findings']} grounded directional "
        f"findings) reporting {display}, one global direction matches "
        f"{_whole_percent(comparison['agreement_q'])}% of paper-balanced findings; "
        f"regime-specific directions match {_whole_percent(comparison['agreement_p'])}% "
        f"(+{_whole_percent(comparison['absolute_gain'])} points), changing from "
        f"{contrast['direction_a']} for {contrast['level_a']} to {contrast['direction_b']} for "
        f"{contrast['level_b']}. The same material contrast appeared in "
        f"{_whole_percent(stability['pattern_fraction'])}% of paper bootstraps "
        f"(Westfall\u2013Young adjusted p={float(result['westfall_young_p']):.3f})."
    )
    return {
        "narrative_variant": "A",
        "cohort_definition": "primary_grounded_unflagged",
        "analysis_labels": list(CANONICAL_LABELS),
        "comparison_subset": {
            "n_findings": comparison["n_findings"],
            "n_papers": comparison["n_papers"],
            "coverage_papers": comparison["coverage_papers"],
        },
        "global_baseline": {
            "modal_direction": comparison["global_mode"],
            "agreement_q": comparison["agreement_q"],
        },
        "within_regime": {
            "agreement_p": comparison["agreement_p"],
            "absolute_gain": comparison["absolute_gain"],
        },
        "contrast": contrast,
        "moderator": {
            "name": moderator,
            "k": result["k"],
            "delta_ll": result["delta_ll"],
            "positive_folds": result["positive_folds"],
            "westfall_young_p": result["westfall_young_p"],
        },
        "stability": stability,
        "rendered_sentence": sentence,
    }


def build_variant_b_headline(
    *,
    selection_reason: str,
    disagreement: Mapping[str, Any],
    residuals: Mapping[str, Any],
    sparse_or_empty_cells: int,
    total_cells: int,
    m4_failures: Sequence[Mapping[str, Any]] = (),
    global_baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the literal Variant-B branch from one of three approved clauses."""

    clauses = {
        "g3_story_not_viable": (
            "the corpus did not meet the pre-registered disagreement-support gate for a "
            "conditional-pattern claim"
        ),
        "m4_no_moderator": (
            "no pre-specified moderator passed every pre-registered inference, stability, "
            "support, sensitivity, and materiality rule"
        ),
        "m4_incomplete": (
            "the pre-registered moderator-inference battery did not complete before freeze"
        ),
    }
    if selection_reason not in clauses:
        raise ResamplingContractError(f"invalid Variant-B selection reason: {selection_reason!r}")
    interval = disagreement["interval_90"]
    sentence = (
        "In our retrieved, grounded corpus, paper-level directional disagreement is "
        f"{_whole_percent(disagreement['paper_entropy'])}% (90% bootstrap interval "
        f"{_whole_percent(interval[0])}\u2013{_whole_percent(interval[1])}%), but "
        f"{clauses[selection_reason]}. {residuals['pair_count']} grounded opposite-direction "
        f"paper pairs remain, and {sparse_or_empty_cells} of {total_cells} pre-registered "
        "evidence cells contain fewer than five papers."
    )
    m4_status = {
        "g3_story_not_viable": "not_run",
        "m4_no_moderator": "failed",
        "m4_incomplete": "incomplete",
    }[selection_reason]
    headline: dict[str, Any] = {
        "narrative_variant": "B",
        "selection_reason": selection_reason,
        "cohort_definition": "primary_grounded_unflagged",
        "analysis_labels": list(CANONICAL_LABELS),
        "disagreement": dict(disagreement),
        "m4": {"status": m4_status, "failures": list(m4_failures)},
        "residuals": dict(residuals),
        "evidence_gaps": {
            "sparse_or_empty_cells": sparse_or_empty_cells,
            "total_cells": total_cells,
        },
        "rendered_sentence": sentence,
    }
    if global_baseline is not None:
        headline["global_baseline"] = dict(global_baseline)
    return headline


def validate_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Validate the complete, closed running-snapshot checkpoint object."""

    keys = set(checkpoint)
    if keys != CHECKPOINT_KEYS:
        raise ResamplingContractError(
            f"checkpoint keys mismatch: missing={sorted(CHECKPOINT_KEYS - keys)}, "
            f"extra={sorted(keys - CHECKPOINT_KEYS)}"
        )
    if (
        checkpoint["checkpoint_version"] != "1"
        or checkpoint["checkpoint_status"] != "running_snapshot"
    ):
        raise ResamplingContractError("checkpoint version/status is invalid")
    budgets = checkpoint["budgets"]
    if not isinstance(budgets, Mapping) or set(budgets) != {
        "bootstrap_count",
        "permutation_successes",
        "permutation_max_attempts",
    }:
        raise ResamplingContractError("checkpoint budgets are invalid")
    if (
        budgets["bootstrap_count"] < 1
        or budgets["permutation_successes"] < 1
        or budgets["permutation_max_attempts"] < budgets["permutation_successes"]
    ):
        raise ResamplingContractError("checkpoint budget values are invalid")
    bootstrap_indices = list(checkpoint["completed_bootstrap_indices"])
    permutation_indices = list(checkpoint["completed_permutation_attempt_indices"])
    success_indices = list(checkpoint["successful_permutation_indices"])
    if bootstrap_indices != sorted(set(bootstrap_indices)) or any(
        index < 0 or index >= budgets["bootstrap_count"] for index in bootstrap_indices
    ):
        raise ResamplingContractError("checkpoint bootstrap index set is invalid")
    if permutation_indices != sorted(set(permutation_indices)) or any(
        index < 0 or index >= budgets["permutation_max_attempts"] for index in permutation_indices
    ):
        raise ResamplingContractError("checkpoint permutation index set is invalid")
    if success_indices != sorted(set(success_indices)) or not set(success_indices).issubset(
        permutation_indices
    ):
        raise ResamplingContractError("checkpoint successful permutation indices are invalid")
    if len(checkpoint["bootstrap_results"]) != len(bootstrap_indices):
        raise ResamplingContractError("checkpoint bootstrap results do not match indices")
    if len(checkpoint["permutation_results"]) != len(permutation_indices):
        raise ResamplingContractError("checkpoint permutation results do not match indices")
    for hash_field in ("config_sha256", "cohort_sha256", "g3_sha256"):
        value = checkpoint[hash_field]
        if not isinstance(value, str) or len(value) != 64:
            raise ResamplingContractError(f"checkpoint {hash_field} is invalid")


def checkpoint_is_genuinely_incomplete(checkpoint: Mapping[str, Any]) -> bool:
    validate_checkpoint(checkpoint)
    budgets = checkpoint["budgets"]
    bootstraps_unfinished = (
        len(checkpoint["completed_bootstrap_indices"]) < budgets["bootstrap_count"]
    )
    permutations_terminal = (
        len(checkpoint["successful_permutation_indices"]) >= budgets["permutation_successes"]
        or len(checkpoint["completed_permutation_attempt_indices"])
        >= budgets["permutation_max_attempts"]
    )
    return bootstraps_unfinished or not permutations_terminal


def frozen_checkpoint_wrapper(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Embed an unfinished source object and bind its canonical-object hash."""

    validate_checkpoint(checkpoint)
    if not checkpoint_is_genuinely_incomplete(checkpoint):
        raise ResamplingContractError("terminal checkpoint cannot be finalized as incomplete")
    return {
        "status": "frozen_incomplete",
        "source_checkpoint_sha256": canonical_sha256(checkpoint),
        "checkpoint": dict(checkpoint),
    }


def write_content_addressed_checkpoint(
    checkpoint: Mapping[str, Any], directory: str | Path
) -> Path:
    """Atomically write a canonical checkpoint under its content hash."""

    validate_checkpoint(checkpoint)
    payload = canonical_json_bytes(checkpoint)
    digest = hashlib.sha256(payload).hexdigest()
    destination_directory = Path(directory)
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / f"{digest}.json"
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ResamplingContractError("content-addressed checkpoint path has different bytes")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{digest}.", suffix=".tmp", dir=destination_directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


__all__ = [
    "CHECKPOINT_KEYS",
    "ResamplingContractError",
    "bootstrap_contrast_stability",
    "build_variant_a_headline",
    "build_variant_b_headline",
    "canonical_json_bytes",
    "canonical_sha256",
    "checkpoint_is_genuinely_incomplete",
    "comparison_statistics",
    "evaluate_m4_gate",
    "frozen_checkpoint_wrapper",
    "paper_bootstrap_sample",
    "paper_permutation",
    "run_paper_bootstraps",
    "summarize_westfall_young",
    "validate_checkpoint",
    "write_content_addressed_checkpoint",
]
