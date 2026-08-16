"""Human-renderable, paper-weighted shallow decision trees."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from literature_multiverse.disagreement import CANONICAL_LABELS, paper_balanced_weights
from literature_multiverse.moderators import (
    align_and_floor_probabilities,
    feasible_grouped_splits,
    laplace_prior,
    weighted_log_loss,
)
from literature_multiverse.resampling import paper_bootstrap_sample

NOT_REPORTED = "__NOT_REPORTED__"


class TreeContractError(ValueError):
    """Raised when the supporting-tree contract cannot be satisfied."""


def _raw_value(row: Mapping[str, Any], name: str) -> Any:
    flattened = f"mod__{name}"
    if flattened in row:
        return row[flattened]
    moderators = row.get("moderators")
    if isinstance(moderators, Mapping) and name in moderators:
        return moderators[name]
    return row.get(name)


def _encoded_value(value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return NOT_REPORTED
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return NOT_REPORTED
        return format(value, ".15g")
    return str(value)


def _matrix(rows: Sequence[Mapping[str, Any]], moderator_names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[_encoded_value(_raw_value(row, name)) for name in moderator_names] for row in rows],
        dtype=object,
    )


def _feature_names(encoder: OneHotEncoder, moderator_names: Sequence[str]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for name, categories in zip(moderator_names, encoder.categories_, strict=True):
        output.extend({"moderator": name, "level": str(category)} for category in categories)
    return output


def _fit_full_tree(
    rows: Sequence[Mapping[str, Any]],
    moderator_names: Sequence[str],
    *,
    seed: int,
    max_depth: int,
) -> tuple[DecisionTreeClassifier, OneHotEncoder, np.ndarray, list[dict[str, str]]]:
    x_raw = _matrix(rows, moderator_names)
    encoder = OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False)
    x = encoder.fit_transform(x_raw)
    y = np.asarray([row["effect_direction"] for row in rows], dtype=object)
    weights = np.asarray(paper_balanced_weights(rows), dtype=float)
    classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=seed,
        class_weight=None,
    )
    classifier.fit(x, y, sample_weight=weights)
    return classifier, encoder, x, _feature_names(encoder, moderator_names)


def evaluate_tree_cv(
    rows: Sequence[Mapping[str, Any]],
    moderator_names: Sequence[str],
    *,
    seed: int = 20260815,
    max_depth: int = 3,
    max_folds: int = 5,
) -> dict[str, Any]:
    """Score a depth-limited tree against the identical fold-local Laplace baseline."""

    if not 1 <= max_depth <= 3:
        raise TreeContractError("tree max_depth must be between 1 and 3")
    subset = [
        dict(row)
        for row in rows
        if row.get("effect_direction") in CANONICAL_LABELS
        and isinstance(row.get("paper_id"), str)
        and row.get("paper_id")
    ]
    if not subset or not moderator_names:
        return {
            "status": "insufficient_for_cv",
            "reason": "no_rows_or_moderators",
            "k": None,
            "delta_ll": None,
            "folds": [],
        }
    labels = [str(row["effect_direction"]) for row in subset]
    paper_ids = [str(row["paper_id"]) for row in subset]
    split = feasible_grouped_splits(labels, paper_ids, seed=seed, max_folds=max_folds)
    if not split["splits"]:
        return {
            "status": split["status"],
            "reason": "no_all_class_grouped_split",
            "k": split["k"],
            "delta_ll": None,
            "folds": [],
        }
    raw_x = _matrix(subset, moderator_names)
    y = np.asarray(labels, dtype=object)
    folds: list[dict[str, Any]] = []
    for fold_index, (train, test) in enumerate(split["splits"]):
        encoder = OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False)
        train_x = encoder.fit_transform(raw_x[train])
        test_x = encoder.transform(raw_x[test])
        train_rows = [subset[index] for index in train]
        test_rows = [subset[index] for index in test]
        train_weights = np.asarray(paper_balanced_weights(train_rows), dtype=float)
        test_weights = np.asarray(paper_balanced_weights(test_rows), dtype=float)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
        model.fit(train_x, y[train], sample_weight=train_weights)
        probabilities = align_and_floor_probabilities(model.predict_proba(test_x), model.classes_)
        prior = laplace_prior(y[train].tolist(), train_weights.tolist())
        baseline = np.repeat(prior.reshape(1, -1), len(test), axis=0)
        model_loss = weighted_log_loss(y[test].tolist(), probabilities, test_weights)
        baseline_loss = weighted_log_loss(y[test].tolist(), baseline, test_weights)
        folds.append(
            {
                "fold": fold_index,
                "train_papers": sorted({paper_ids[index] for index in train}),
                "test_papers": sorted({paper_ids[index] for index in test}),
                "model_log_loss": model_loss,
                "baseline_log_loss": baseline_loss,
                "delta_ll": baseline_loss - model_loss,
            }
        )
    deltas = [fold["delta_ll"] for fold in folds]
    return {
        "status": split["status"],
        "reason": None,
        "k": split["k"],
        "delta_ll": float(np.mean(deltas)),
        "positive_folds": sum(delta > 0 for delta in deltas),
        "folds": folds,
    }


def build_tree_artifact(
    rows: Sequence[Mapping[str, Any]],
    moderator_names: Sequence[str],
    *,
    seed: int = 20260815,
    max_depth: int = 3,
    supporting: bool = False,
) -> dict[str, Any]:
    """Fit a tree and emit a stable JSON renderer spec, never a serialized plot."""

    if not 1 <= max_depth <= 3:
        raise TreeContractError("tree max_depth must be between 1 and 3")
    if not moderator_names or len(set(moderator_names)) != len(moderator_names):
        raise TreeContractError("moderator_names must be non-empty and unique")
    subset = [
        dict(row)
        for row in rows
        if row.get("effect_direction") in CANONICAL_LABELS
        and isinstance(row.get("paper_id"), str)
        and row.get("paper_id")
    ]
    if len({row["effect_direction"] for row in subset}) < 2:
        return {
            "status": "not_run",
            "reason": "fewer_than_two_classes",
            "nodes": [],
            "max_depth": max_depth,
            "cv": evaluate_tree_cv(subset, moderator_names, seed=seed, max_depth=max_depth),
        }
    classifier, _encoder, encoded, features = _fit_full_tree(
        subset, moderator_names, seed=seed, max_depth=max_depth
    )
    decision_path = classifier.decision_path(encoded).tocsc()
    weights = paper_balanced_weights(subset)
    label_order = list(CANONICAL_LABELS)
    flat_nodes: list[dict[str, Any]] = []

    def emit(node_id: int, depth: int, *, missingness_path: bool) -> None:
        row_indices = decision_path[:, node_id].nonzero()[0].tolist()
        node_rows = [subset[index] for index in row_indices]
        node_weights = [weights[index] for index in row_indices]
        class_mass = {label: 0.0 for label in label_order}
        for row, weight in zip(node_rows, node_weights, strict=True):
            class_mass[row["effect_direction"]] += weight
        maximum = max(class_mass.values()) if class_mass else 0.0
        winners = [
            label
            for label in label_order
            if math.isclose(class_mass[label], maximum, rel_tol=0.0, abs_tol=1e-12)
        ]
        left = int(classifier.tree_.children_left[node_id])
        right = int(classifier.tree_.children_right[node_id])
        is_leaf = left == right
        n_papers = len({row["paper_id"] for row in node_rows})
        base: dict[str, Any] = {
            "node_id": node_id,
            "depth": depth,
            "type": "leaf" if is_leaf else "branch",
            "n_findings": len(node_rows),
            "n_papers": n_papers,
            "paper_balanced_class_mass": class_mass,
            "modal_direction": winners[0] if len(winners) == 1 else None,
            "missingness_path": missingness_path,
        }
        if is_leaf:
            greyed = n_papers < 5
            base.update(
                {
                    "greyed": greyed,
                    "grey_reason": "fewer_than_5_distinct_papers" if greyed else None,
                    "narratable": not greyed and not missingness_path,
                }
            )
            flat_nodes.append(base)
            return
        feature_index = int(classifier.tree_.feature[node_id])
        feature = features[feature_index]
        split_is_missingness = feature["level"] == NOT_REPORTED
        base.update(
            {
                "moderator": feature["moderator"],
                "level": feature["level"],
                "display_level": ("not reported" if split_is_missingness else feature["level"]),
                "threshold": float(classifier.tree_.threshold[node_id]),
                "left_child": left,
                "right_child": right,
                "left_condition": "is not level",
                "right_condition": "is level",
                "missingness_split": split_is_missingness,
                "narratable": not split_is_missingness and not missingness_path,
            }
        )
        flat_nodes.append(base)
        emit(left, depth + 1, missingness_path=missingness_path)
        emit(
            right,
            depth + 1,
            missingness_path=missingness_path or split_is_missingness,
        )

    emit(0, 0, missingness_path=False)
    cv = evaluate_tree_cv(subset, moderator_names, seed=seed, max_depth=max_depth)
    root = flat_nodes[0]
    return {
        "status": "supporting" if supporting else "exploratory",
        "reason": None,
        "seed": seed,
        "max_depth": max_depth,
        "n_findings": len(subset),
        "n_papers": len({row["paper_id"] for row in subset}),
        "feature_names": list(moderator_names),
        "root_split": (
            {"moderator": root["moderator"], "level": root["level"]}
            if root["type"] == "branch"
            else None
        ),
        "cv": cv,
        "nodes": sorted(flat_nodes, key=lambda node: node["node_id"]),
    }


def root_split_bootstrap_frequency(
    rows: Sequence[Mapping[str, Any]],
    moderator_names: Sequence[str],
    *,
    expected_moderator: str,
    n_bootstraps: int = 200,
    seed: int = 20260815,
    max_depth: int = 3,
) -> dict[str, Any]:
    """Return the all-draw frequency with which the root uses the expected moderator."""

    if expected_moderator not in moderator_names:
        raise TreeContractError("expected root moderator is outside the configured tree family")
    matches = 0
    errors: list[dict[str, Any]] = []
    roots: list[dict[str, Any] | None] = []
    for draw_index in range(n_bootstraps):
        try:
            sample = paper_bootstrap_sample(rows, seed=seed, draw_index=draw_index)
            eligible = [row for row in sample if row.get("effect_direction") in CANONICAL_LABELS]
            if len({row["effect_direction"] for row in eligible}) < 2:
                roots.append(None)
                continue
            classifier, _, _, features = _fit_full_tree(
                eligible, moderator_names, seed=seed, max_depth=max_depth
            )
            feature_index = int(classifier.tree_.feature[0])
            if feature_index < 0:
                root = None
            else:
                feature = features[feature_index]
                root = {"moderator": feature["moderator"], "level": feature["level"]}
            roots.append(root)
            matches += bool(root and root["moderator"] == expected_moderator)
        except Exception as exc:  # Guarded registered draw.
            roots.append(None)
            errors.append(
                {
                    "draw_index": draw_index,
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
    return {
        "seed": seed,
        "n_bootstraps": n_bootstraps,
        "expected_moderator": expected_moderator,
        "match_count": matches,
        "frequency": matches / n_bootstraps,
        "error_count": len(errors),
        "errors": errors,
        "root_splits": roots,
    }


__all__ = [
    "NOT_REPORTED",
    "TreeContractError",
    "build_tree_artifact",
    "evaluate_tree_cv",
    "root_split_bootstrap_frequency",
]
