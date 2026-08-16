"""Paper-balanced, grouped-CV moderator evaluation.

This module owns the pre-registered §6.2 estimator.  It operates on records rather than
foundation models so the same implementation is used for parquet rows and fixtures.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder

from literature_multiverse.disagreement import CANONICAL_LABELS, paper_balanced_weights

OTHER_LEVEL = "__OTHER__"
PROBABILITY_FLOOR = 1e-3


class ModeratorContractError(ValueError):
    """Raised when a moderator evaluation violates the registered analysis contract."""


def _value(row: Mapping[str, Any], name: str) -> Any:
    flattened = f"mod__{name}"
    if flattened in row:
        return row[flattened]
    moderators = row.get("moderators")
    if isinstance(moderators, Mapping) and name in moderators:
        return moderators[name]
    return row.get(name)


def _not_missing(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def align_and_floor_probabilities(
    probabilities: np.ndarray,
    model_classes: Sequence[str],
    *,
    labels: Sequence[str] = CANONICAL_LABELS,
    floor: float = PROBABILITY_FLOOR,
) -> np.ndarray:
    """Align arbitrary estimator columns to the canonical label array and renormalize."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(model_classes):
        raise ModeratorContractError("probability matrix and model classes do not align")
    if not 0 < floor < 1 / len(labels):
        raise ModeratorContractError("probability floor is outside its valid range")
    if set(model_classes) - set(labels):
        raise ModeratorContractError("model emitted a non-canonical class")
    aligned = np.zeros((values.shape[0], len(labels)), dtype=float)
    class_index = {label: index for index, label in enumerate(model_classes)}
    for output_index, label in enumerate(labels):
        if label in class_index:
            aligned[:, output_index] = values[:, class_index[label]]
    aligned = np.maximum(aligned, floor)
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def laplace_prior(
    labels_observed: Sequence[str],
    weights: Sequence[float],
    *,
    labels: Sequence[str] = CANONICAL_LABELS,
    alpha: float = 1.0,
) -> np.ndarray:
    """Return the fold-local weighted canonical-label Laplace prior."""

    if alpha <= 0:
        raise ModeratorContractError("Laplace alpha must be positive")
    if len(labels_observed) != len(weights):
        raise ModeratorContractError("labels and weights differ in length")
    counts = {label: float(alpha) for label in labels}
    for label, weight in zip(labels_observed, weights, strict=True):
        if label not in counts:
            raise ModeratorContractError(f"non-canonical analysis label: {label!r}")
        if weight < 0 or not math.isfinite(weight):
            raise ModeratorContractError("weights must be finite and non-negative")
        counts[label] += float(weight)
    total = sum(counts.values())
    return np.asarray([counts[label] / total for label in labels], dtype=float)


def weighted_log_loss(
    labels_observed: Sequence[str],
    probabilities: np.ndarray,
    weights: Sequence[float],
    *,
    labels: Sequence[str] = CANONICAL_LABELS,
) -> float:
    """Natural-log loss on a full canonical probability array."""

    values = np.asarray(probabilities, dtype=float)
    weight_values = np.asarray(weights, dtype=float)
    if values.shape != (len(labels_observed), len(labels)):
        raise ModeratorContractError("probability array has the wrong shape")
    if weight_values.shape != (len(labels_observed),) or weight_values.sum() <= 0:
        raise ModeratorContractError("score weights are invalid")
    label_index = {label: index for index, label in enumerate(labels)}
    try:
        chosen = np.asarray(
            [
                values[row_index, label_index[label]]
                for row_index, label in enumerate(labels_observed)
            ]
        )
    except KeyError as exc:
        raise ModeratorContractError(f"non-canonical score label: {exc.args[0]!r}") from exc
    return float(-np.average(np.log(chosen), weights=weight_values))


def pool_rare_levels(
    values: Sequence[Any],
    paper_ids: Sequence[str],
    *,
    minimum_distinct_papers: int = 3,
) -> dict[str, Any]:
    """Pool levels using distinct-paper support only, never finding-row frequency."""

    if len(values) != len(paper_ids):
        raise ModeratorContractError("moderator values and paper IDs differ in length")
    support: dict[Any, set[str]] = defaultdict(set)
    for value, paper_id in zip(values, paper_ids, strict=True):
        if _not_missing(value):
            support[value].add(paper_id)
    rare = {value for value, papers in support.items() if len(papers) < minimum_distinct_papers}
    pooled = [OTHER_LEVEL if value in rare else value for value in values]
    return {
        "values": pooled,
        "distinct_papers_by_original_level": {
            str(level): len(papers)
            for level, papers in sorted(support.items(), key=lambda item: str(item[0]))
        },
        "pooled_original_levels": sorted(str(value) for value in rare),
    }


def feasible_grouped_splits(
    labels_observed: Sequence[str],
    paper_ids: Sequence[str],
    *,
    seed: int,
    max_folds: int = 5,
) -> dict[str, Any]:
    """Find the largest grouped split whose train and test sets contain every subset label."""

    if len(labels_observed) != len(paper_ids) or not labels_observed:
        raise ModeratorContractError("split labels/groups must be non-empty and aligned")
    represented = tuple(label for label in CANONICAL_LABELS if label in set(labels_observed))
    if len(represented) < 2:
        return {"k": None, "status": "insufficient_for_cv", "splits": [], "labels": represented}
    papers_by_class = {
        label: {
            paper
            for paper, observed in zip(paper_ids, labels_observed, strict=True)
            if observed == label
        }
        for label in represented
    }
    start = min(max_folds, min(len(papers) for papers in papers_by_class.values()))
    if start < 2:
        return {"k": None, "status": "insufficient_for_cv", "splits": [], "labels": represented}

    # One paper-class record makes split assignment invariant to duplicated findings while
    # still using the registered StratifiedGroupKFold implementation.
    split_y: list[str] = []
    split_groups: list[str] = []
    for paper_id in sorted(set(paper_ids)):
        for label in represented:
            if any(
                group == paper_id and observed == label
                for group, observed in zip(paper_ids, labels_observed, strict=True)
            ):
                split_y.append(label)
                split_groups.append(paper_id)
    split_x = np.zeros((len(split_y), 1), dtype=float)
    row_groups = np.asarray(paper_ids, dtype=object)
    for k in range(start, 1, -1):
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        candidate: list[tuple[np.ndarray, np.ndarray]] = []
        valid = True
        for split_train, split_test in splitter.split(split_x, split_y, groups=split_groups):
            train_papers = {split_groups[index] for index in split_train}
            test_papers = {split_groups[index] for index in split_test}
            train = np.flatnonzero(np.isin(row_groups, list(train_papers)))
            test = np.flatnonzero(np.isin(row_groups, list(test_papers)))
            if train_papers & test_papers:
                raise ModeratorContractError("grouped CV placed one paper in both fold sides")
            if set(np.asarray(labels_observed, dtype=object)[train]) != set(represented):
                valid = False
                break
            if set(np.asarray(labels_observed, dtype=object)[test]) != set(represented):
                valid = False
                break
            candidate.append((train, test))
        if valid and len(candidate) == k:
            return {
                "k": k,
                "status": "eligible" if k >= 3 else "exploratory",
                "splits": candidate,
                "labels": represented,
            }
    return {"k": None, "status": "insufficient_for_cv", "splits": [], "labels": represented}


def _weighted_mutual_information(
    levels: Sequence[Any], labels_observed: Sequence[str], weights: Sequence[float]
) -> float:
    total = float(sum(weights))
    joint: Counter[tuple[Any, str]] = Counter()
    level_mass: Counter[Any] = Counter()
    label_mass: Counter[str] = Counter()
    for level, label, weight in zip(levels, labels_observed, weights, strict=True):
        joint[(level, label)] += weight
        level_mass[level] += weight
        label_mass[label] += weight
    information = 0.0
    for (level, label), mass in joint.items():
        p_joint = mass / total
        information += p_joint * math.log(
            p_joint / ((level_mass[level] / total) * (label_mass[label] / total))
        )
    return information


def evaluate_moderator(
    rows: Sequence[Mapping[str, Any]],
    moderator_name: str,
    *,
    seed: int = 20260815,
    max_folds: int = 5,
    kind: str = "paper_constant",
    inferential_key: str | None = None,
) -> dict[str, Any]:
    """Run the exact same-subset baseline/moderator grouped-CV comparison."""

    if kind not in {"paper_constant", "within_paper"}:
        raise ModeratorContractError(f"invalid moderator kind: {kind!r}")
    if kind == "within_paper" and inferential_key is None:
        return {
            "moderator": moderator_name,
            "status": "descriptive_only",
            "reason": "within_paper_summary_missing",
            "k": None,
            "delta_ll": None,
            "folds": [],
        }
    source_name = inferential_key or moderator_name
    subset = [
        dict(row)
        for row in rows
        if row.get("effect_direction") in CANONICAL_LABELS
        and isinstance(row.get("paper_id"), str)
        and _not_missing(_value(row, source_name))
    ]
    all_papers = {row.get("paper_id") for row in rows if isinstance(row.get("paper_id"), str)}
    subset_papers = {row["paper_id"] for row in subset}
    coverage_findings = len(subset) / len(rows) if rows else 0.0
    coverage_papers = len(subset_papers) / len(all_papers) if all_papers else 0.0
    base = {
        "moderator": moderator_name,
        "inferential_key": source_name,
        "n_findings": len(subset),
        "n_papers": len(subset_papers),
        "coverage_findings": coverage_findings,
        "coverage_papers": coverage_papers,
    }
    if not subset:
        return {
            **base,
            "status": "insufficient_for_cv",
            "reason": "no_nonmissing_rows",
            "k": None,
            "delta_ll": None,
            "folds": [],
        }

    labels_observed = [str(row["effect_direction"]) for row in subset]
    paper_ids = [str(row["paper_id"]) for row in subset]
    original_values = [_value(row, source_name) for row in subset]
    pooled = pool_rare_levels(original_values, paper_ids)
    levels = pooled["values"]
    narrated_levels = {value for value in levels if value != OTHER_LEVEL}
    support_by_level_class: dict[str, dict[str, int]] = {}
    for level in sorted(set(original_values), key=str):
        level_key = str(level)
        support_by_level_class[level_key] = {
            label: len(
                {
                    paper_id
                    for value, paper_id, observed in zip(
                        original_values, paper_ids, labels_observed, strict=True
                    )
                    if value == level and observed == label
                }
            )
            for label in CANONICAL_LABELS
        }
        support_by_level_class[level_key]["total"] = len(
            {
                paper_id
                for value, paper_id in zip(original_values, paper_ids, strict=True)
                if value == level
            }
        )
    base.update(
        {
            "support_by_level_class": support_by_level_class,
            "pooled_original_levels": pooled["pooled_original_levels"],
            "model_levels": sorted(str(level) for level in set(levels)),
            "narratable_levels": sorted(str(level) for level in narrated_levels),
        }
    )
    if len(set(levels)) < 2:
        return {
            **base,
            "status": "insufficient_for_cv",
            "reason": "fewer_than_two_model_levels",
            "k": None,
            "delta_ll": None,
            "folds": [],
        }

    split_result = feasible_grouped_splits(
        labels_observed, paper_ids, seed=seed, max_folds=max_folds
    )
    if not split_result["splits"]:
        return {
            **base,
            "status": split_result["status"],
            "reason": "no_all_class_grouped_split",
            "k": split_result["k"],
            "labels_observed": list(split_result["labels"]),
            "delta_ll": None,
            "folds": [],
        }

    x = np.asarray(levels, dtype=object).reshape(-1, 1)
    y = np.asarray(labels_observed, dtype=object)
    folds: list[dict[str, Any]] = []
    for fold_index, (train, test) in enumerate(split_result["splits"]):
        train_rows = [subset[index] for index in train]
        test_rows = [subset[index] for index in test]
        train_weights = np.asarray(paper_balanced_weights(train_rows), dtype=float)
        test_weights = np.asarray(paper_balanced_weights(test_rows), dtype=float)
        encoder = OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False)
        train_x = encoder.fit_transform(x[train])
        test_x = encoder.transform(x[test])
        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            class_weight=None,
            random_state=seed,
        )
        model.fit(train_x, y[train], sample_weight=train_weights)
        model_probabilities = align_and_floor_probabilities(
            model.predict_proba(test_x), model.classes_
        )
        prior = laplace_prior(y[train].tolist(), train_weights.tolist())
        baseline_probabilities = np.repeat(prior.reshape(1, -1), len(test), axis=0)
        model_loss = weighted_log_loss(y[test].tolist(), model_probabilities, test_weights)
        baseline_loss = weighted_log_loss(y[test].tolist(), baseline_probabilities, test_weights)
        folds.append(
            {
                "fold": fold_index,
                "train_indices": train.tolist(),
                "test_indices": test.tolist(),
                "train_papers": sorted({paper_ids[index] for index in train}),
                "test_papers": sorted({paper_ids[index] for index in test}),
                "n_train_papers": len({paper_ids[index] for index in train}),
                "n_test_papers": len({paper_ids[index] for index in test}),
                "model_log_loss": model_loss,
                "baseline_log_loss": baseline_loss,
                "delta_ll": baseline_loss - model_loss,
                "test_weight_sum": float(test_weights.sum()),
            }
        )
    deltas = [fold["delta_ll"] for fold in folds]
    full_weights = paper_balanced_weights(subset)
    return {
        **base,
        "status": split_result["status"],
        "reason": None,
        "k": split_result["k"],
        "labels_observed": list(split_result["labels"]),
        "delta_ll": float(np.mean(deltas)),
        "delta_ll_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        "positive_folds": sum(delta > 0 for delta in deltas),
        "folds": folds,
        "descriptive_mi": _weighted_mutual_information(levels, labels_observed, full_weights),
    }


def _spec_value(spec: Any, key: str, default: Any = None) -> Any:
    if isinstance(spec, Mapping):
        return spec.get(key, default)
    return getattr(spec, key, default)


def evaluate_all_moderators(
    rows: Sequence[Mapping[str, Any]],
    moderator_specs: Sequence[Any],
    *,
    seed: int = 20260815,
    max_folds: int = 5,
) -> list[dict[str, Any]]:
    """Evaluate every configured moderator in locked order, retaining failed rows."""

    output: list[dict[str, Any]] = []
    names: set[str] = set()
    for spec in moderator_specs:
        name = _spec_value(spec, "name")
        if not isinstance(name, str) or not name or name in names:
            raise ModeratorContractError("moderator specs require unique non-empty names")
        names.add(name)
        kind = _spec_value(spec, "kind", "paper_constant")
        paper_summary = _spec_value(spec, "paper_summary")
        inferential_key = (
            f"{name}__paper_summary" if kind == "within_paper" and paper_summary else None
        )
        result = evaluate_moderator(
            rows,
            name,
            seed=seed,
            max_folds=max_folds,
            kind=kind,
            inferential_key=inferential_key,
        )
        result["role"] = _spec_value(spec, "role", "tested")
        result["permutation"] = _spec_value(spec, "permutation", "none")
        output.append(result)
    return output


__all__ = [
    "OTHER_LEVEL",
    "PROBABILITY_FLOOR",
    "ModeratorContractError",
    "align_and_floor_probabilities",
    "evaluate_all_moderators",
    "evaluate_moderator",
    "feasible_grouped_splits",
    "laplace_prior",
    "pool_rare_levels",
    "weighted_log_loss",
]
