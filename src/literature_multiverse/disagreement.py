"""Paper-balanced disagreement summaries and pure G2/G3 gate functions."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

CANONICAL_LABELS = ("increase", "no_effect", "decrease")
UNRESOLVED_LABELS = (*CANONICAL_LABELS, "unresolved")


class DisagreementContractError(ValueError):
    """Raised when a disagreement calculation receives incoherent rows."""


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    paper_key: str,
    label_key: str,
    labels: Sequence[str],
) -> None:
    allowed = set(labels)
    for position, row in enumerate(rows):
        paper_id = row.get(paper_key)
        label = row.get(label_key)
        if not isinstance(paper_id, str) or not paper_id:
            raise DisagreementContractError(f"row {position} has no valid {paper_key}")
        if label not in allowed:
            raise DisagreementContractError(f"row {position} has non-analysis label {label!r}")


def paper_balanced_weights(
    rows: Sequence[Mapping[str, Any]],
    *,
    paper_key: str = "paper_id",
) -> list[float]:
    """Return ``1 / findings-from-paper`` weights, preserving input order."""

    counts = Counter(row.get(paper_key) for row in rows)
    if None in counts or "" in counts:
        raise DisagreementContractError(f"every row requires a non-empty {paper_key}")
    return [1.0 / counts[row[paper_key]] for row in rows]


def normalized_entropy(proportions: Mapping[str, float], *, denominator_classes: int) -> float:
    """Shannon entropy divided by a fixed ``log(denominator_classes)``."""

    if denominator_classes < 2:
        raise DisagreementContractError("entropy denominator needs at least two classes")
    values = list(proportions.values())
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise DisagreementContractError("entropy proportions must be finite and non-negative")
    total = sum(values)
    if total <= 0:
        raise DisagreementContractError("entropy is undefined for zero mass")
    probabilities = [value / total for value in values if value > 0]
    entropy = -sum(value * math.log(value) for value in probabilities)
    return entropy / math.log(denominator_classes)


def paper_balanced_finding_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    paper_key: str = "paper_id",
    label_key: str = "effect_direction",
    labels: Sequence[str] = CANONICAL_LABELS,
) -> dict[str, Any]:
    """Summarize finding directions with each distinct paper contributing unit mass."""

    _validate_rows(rows, paper_key=paper_key, label_key=label_key, labels=labels)
    if not rows:
        raise DisagreementContractError("finding disagreement requires at least one row")
    weights = paper_balanced_weights(rows, paper_key=paper_key)
    mass = {label: 0.0 for label in labels}
    raw_counts = {label: 0 for label in labels}
    papers_by_label: dict[str, set[str]] = {label: set() for label in labels}
    for row, weight in zip(rows, weights, strict=True):
        label = row[label_key]
        mass[label] += weight
        raw_counts[label] += 1
        papers_by_label[label].add(row[paper_key])
    total_mass = sum(mass.values())
    proportions = {label: mass[label] / total_mass for label in labels}
    max_mass = max(mass.values())
    modes = [label for label in labels if math.isclose(mass[label], max_mass, abs_tol=1e-12)]
    return {
        "n_findings": len(rows),
        "n_papers": len({row[paper_key] for row in rows}),
        "raw_class_counts": raw_counts,
        "distinct_papers_by_class": {label: len(papers_by_label[label]) for label in labels},
        "paper_balanced_mass": mass,
        "class_proportions": proportions,
        "normalized_entropy": normalized_entropy(
            proportions, denominator_classes=len(CANONICAL_LABELS)
        ),
        "majority": {
            "modal_direction": modes[0] if len(modes) == 1 else None,
            "unique": len(modes) == 1,
            "agreement": max_mass / total_mass,
        },
    }


def _paper_modes(
    rows: Sequence[Mapping[str, Any]],
    *,
    paper_key: str,
    label_key: str,
) -> tuple[dict[str, str], dict[str, Counter[str]]]:
    by_paper: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_paper[row[paper_key]][row[label_key]] += 1
    modes: dict[str, str] = {}
    for paper_id, counts in by_paper.items():
        maximum = max(counts.values())
        winners = [label for label in CANONICAL_LABELS if counts[label] == maximum]
        modes[paper_id] = winners[0] if len(winners) == 1 else "mixed"
    return modes, by_paper


def paper_modal_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    paper_key: str = "paper_id",
    label_key: str = "effect_direction",
) -> dict[str, Any]:
    """Compute primary three-class and fixed-log(4) unresolved paper-mode summaries."""

    _validate_rows(
        rows,
        paper_key=paper_key,
        label_key=label_key,
        labels=CANONICAL_LABELS,
    )
    if not rows:
        raise DisagreementContractError("paper-modal disagreement requires at least one row")
    modes, _ = _paper_modes(rows, paper_key=paper_key, label_key=label_key)
    primary_counts = Counter(mode for mode in modes.values() if mode != "mixed")
    tied = sum(mode == "mixed" for mode in modes.values())
    classifiable = sum(primary_counts.values())
    primary_entropy = (
        normalized_entropy(
            {label: float(primary_counts[label]) for label in CANONICAL_LABELS},
            denominator_classes=3,
        )
        if classifiable
        else None
    )
    unresolved_counts = {
        **{label: primary_counts[label] for label in CANONICAL_LABELS},
        "unresolved": tied,
    }
    unresolved_entropy = normalized_entropy(
        {label: float(unresolved_counts[label]) for label in UNRESOLVED_LABELS},
        denominator_classes=4,
    )
    return {
        "n_papers_total": len(modes),
        "n_papers_classifiable": classifiable,
        "n_papers_tied": tied,
        "tie_fraction": tied / len(modes),
        "paper_modes": modes,
        "primary": {
            "class_counts": {label: primary_counts[label] for label in CANONICAL_LABELS},
            "normalized_entropy": primary_entropy,
            "denominator_log_classes": 3,
        },
        "unresolved_sensitivity": {
            "class_counts": unresolved_counts,
            "normalized_entropy": unresolved_entropy,
            "denominator_log_classes": 4,
            "eligible_for_primary_gate": False,
        },
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise DisagreementContractError("quantile requires values")
    if not 0 <= probability <= 1:
        raise DisagreementContractError("quantile probability must be in [0, 1]")
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_disagreement(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_bootstraps: int = 1000,
    seed: int = 20260815,
    paper_key: str = "paper_id",
    label_key: str = "effect_direction",
) -> dict[str, Any]:
    """Bootstrap whole papers and return fixed-seed 90% entropy intervals.

    A sampled paper occurrence receives a synthetic group ID so repeated draws contribute
    the intended bootstrap multiplicity while findings within each occurrence remain balanced.
    Degenerate paper-modal draws are retained as ``None`` and reported, not silently imputed.
    """

    if n_bootstraps < 1:
        raise DisagreementContractError("n_bootstraps must be positive")
    _validate_rows(
        rows,
        paper_key=paper_key,
        label_key=label_key,
        labels=CANONICAL_LABELS,
    )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[paper_key]].append(row)
    paper_ids = sorted(grouped)
    if not paper_ids:
        raise DisagreementContractError("bootstrap requires at least one paper")
    rng = random.Random(seed)
    finding_draws: list[float] = []
    modal_draws: list[float | None] = []
    for _ in range(n_bootstraps):
        sampled_ids = [rng.choice(paper_ids) for _ in paper_ids]
        sampled_rows: list[dict[str, Any]] = []
        for occurrence, original_id in enumerate(sampled_ids):
            synthetic_id = f"{original_id}#bootstrap#{occurrence}"
            for source_row in grouped[original_id]:
                row = dict(source_row)
                row[paper_key] = synthetic_id
                sampled_rows.append(row)
        finding_draws.append(
            paper_balanced_finding_summary(sampled_rows, paper_key=paper_key, label_key=label_key)[
                "normalized_entropy"
            ]
        )
        modal_draws.append(
            paper_modal_summary(sampled_rows, paper_key=paper_key, label_key=label_key)["primary"][
                "normalized_entropy"
            ]
        )
    valid_modal = [value for value in modal_draws if value is not None]
    return {
        "seed": seed,
        "n_bootstraps": n_bootstraps,
        "finding_entropy": {
            "interval_90": [_quantile(finding_draws, 0.05), _quantile(finding_draws, 0.95)],
            "draws": finding_draws,
            "invalid_draws": 0,
        },
        "paper_entropy": {
            "interval_90": (
                [_quantile(valid_modal, 0.05), _quantile(valid_modal, 0.95)]
                if valid_modal
                else [None, None]
            ),
            "draws": modal_draws,
            "invalid_draws": len(modal_draws) - len(valid_modal),
        },
    }


def _rule(
    name: str,
    observed: Any,
    threshold: Any,
    passed: bool,
    failure_code: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "threshold": threshold,
        "passed": bool(passed),
        "failure_code": None if passed else failure_code,
    }


def evaluate_g2(
    *,
    entropy_point: float | None,
    entropy_interval_90: Sequence[float | None],
    distinct_papers_by_direction: Mapping[str, int],
    relation_purity: float | None,
    estimated_usable_primaries: float,
    topic_support_passed: bool = True,
) -> dict[str, Any]:
    """Evaluate the permissive topic-viability gate without interpreting it as G3."""

    upper = entropy_interval_90[1] if len(entropy_interval_90) == 2 else None
    directions_with_two = sum(
        int(distinct_papers_by_direction.get(label, 0)) >= 2 for label in CANONICAL_LABELS
    )
    rules = [
        _rule(
            "entropy_point",
            entropy_point,
            ">=0.30",
            entropy_point is not None and entropy_point >= 0.30,
            "g2_entropy_point_low",
        ),
        _rule(
            "entropy_upper_90",
            upper,
            ">=0.40",
            upper is not None and upper >= 0.40,
            "g2_entropy_upper_low",
        ),
        _rule(
            "direction_support",
            directions_with_two,
            ">=2 directions with >=2 papers",
            directions_with_two >= 2,
            "g2_direction_support_low",
        ),
        _rule(
            "relation_purity",
            relation_purity,
            ">=0.70",
            relation_purity is not None and relation_purity >= 0.70,
            "g2_relation_purity_low_or_undefined",
        ),
        _rule(
            "estimated_usable_primaries",
            estimated_usable_primaries,
            ">=40",
            estimated_usable_primaries >= 40,
            "g2_usable_primaries_low",
        ),
        _rule(
            "topic_specific_support",
            topic_support_passed,
            "true",
            topic_support_passed,
            "g2_topic_support_failed",
        ),
    ]
    return {
        "gate": "G2",
        "passed": all(rule["passed"] for rule in rules),
        "rules": rules,
        "failure_codes": [rule["failure_code"] for rule in rules if not rule["passed"]],
    }


def evaluate_g3(
    *,
    audit_correct: int,
    audit_total: int,
    anchors_passed: int,
    anchors_total: int,
    cross_model_agreement: float | None,
    quarantine_fraction: float | None,
    g1b_passed: bool,
    paper_entropy_interval_90: Sequence[float | None],
    classifiable_papers: int,
    distinct_papers_by_direction: Mapping[str, int],
    audit_candidate_count: int | None = None,
    audit_failed_ids_in_cohort: Sequence[str] = (),
    release_set_cross_model_agreement: float | None = None,
) -> dict[str, Any]:
    """Evaluate G3 trust and story independently and return the required action."""

    lower = paper_entropy_interval_90[0] if len(paper_entropy_interval_90) == 2 else None
    # Sampling mode (the design's original rule) bounds the error rate of UNAUDITED
    # rows: 20 sampled, at most 3 wrong.  When the whole candidate pool is 20 rows or
    # fewer, the audit is a census — there are no unaudited rows, so the trust demand
    # becomes exact: every row that remains in the released cohort must have passed its
    # audit (rows that failed and were excluded by verification do not count against it).
    census = (
        audit_candidate_count is not None
        and audit_candidate_count <= 20
        and audit_total == audit_candidate_count
        and audit_total > 0
    )
    if census:
        audit_rule = _rule(
            "human_audit",
            {
                "correct": audit_correct,
                "total": audit_total,
                "failed_ids_in_cohort": list(audit_failed_ids_in_cohort),
            },
            "census: no audit-failed row remains in the cohort",
            not audit_failed_ids_in_cohort,
            "g3_audit_failed",
        )
    else:
        audit_rule = _rule(
            "human_audit",
            {"correct": audit_correct, "total": audit_total},
            ">=17/20 with exactly 20",
            audit_total == 20 and audit_correct >= 17,
            "g3_audit_failed",
        )
    # In census mode the trust gate certifies the RELEASE SET (census-audited primary
    # rows minus audit-excluded ones): the ≥0.85 bar applies to model agreement on those
    # rows.  The full-corpus agreement over every exact-grounded finding stays in the
    # rule payload (and in quality metrics) as the transparency figure — many secondary
    # rows carry within-group-only quotes that honestly cannot verify a between-group
    # direction, which is reported, not gated (2026-08-16 execution amendment).
    if census and release_set_cross_model_agreement is not None:
        cross_model_rule = _rule(
            "cross_model_agreement",
            {
                "release_set": release_set_cross_model_agreement,
                "all_exact_grounded": cross_model_agreement,
            },
            ">=0.85 on the census release set",
            release_set_cross_model_agreement >= 0.85,
            "g3_verification_failed",
        )
    else:
        cross_model_rule = _rule(
            "cross_model_agreement",
            cross_model_agreement,
            ">=0.85",
            cross_model_agreement is not None and cross_model_agreement >= 0.85,
            "g3_verification_failed",
        )
    trust_rules = [
        audit_rule,
        _rule(
            "anchors",
            {"passed": anchors_passed, "total": anchors_total},
            "all pass",
            anchors_total > 0 and anchors_passed == anchors_total,
            "g3_anchor_failed",
        ),
        cross_model_rule,
        _rule(
            "quarantine_fraction",
            quarantine_fraction,
            "<=0.10 and defined",
            quarantine_fraction is not None and quarantine_fraction <= 0.10,
            "g3_quarantine_failed_or_undefined",
        ),
        _rule("g1b", g1b_passed, "true", g1b_passed, "g3_g1b_failed"),
    ]
    directions_with_five = sum(
        int(distinct_papers_by_direction.get(label, 0)) >= 5 for label in CANONICAL_LABELS
    )
    story_rules = [
        _rule(
            "paper_entropy_lower_90",
            lower,
            ">=0.40",
            lower is not None and lower >= 0.40,
            "g3_entropy_lower_low",
        ),
        _rule(
            "classifiable_papers",
            classifiable_papers,
            ">=20",
            classifiable_papers >= 20,
            "g3_classifiable_papers_low",
        ),
        _rule(
            "direction_support",
            directions_with_five,
            ">=2 directions with >=5 papers",
            directions_with_five >= 2,
            "g3_direction_support_low",
        ),
    ]
    trust_passed = all(rule["passed"] for rule in trust_rules)
    story_passed = all(rule["passed"] for rule in story_rules)
    if not trust_passed:
        action = "block_release"
    elif not story_passed:
        action = "select_variant_b_story"
    else:
        action = "run_m4"
    return {
        "gate": "G3",
        "trust_passed": trust_passed,
        "story_passed": story_passed,
        "trust_rules": trust_rules,
        "story_rules": story_rules,
        "action": action,
        "failure_codes": [
            rule["failure_code"] for rule in (*trust_rules, *story_rules) if not rule["passed"]
        ],
    }


__all__ = [
    "CANONICAL_LABELS",
    "UNRESOLVED_LABELS",
    "DisagreementContractError",
    "bootstrap_disagreement",
    "evaluate_g2",
    "evaluate_g3",
    "normalized_entropy",
    "paper_balanced_finding_summary",
    "paper_balanced_weights",
    "paper_modal_summary",
]
