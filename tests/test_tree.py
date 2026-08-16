from __future__ import annotations

import pytest

from literature_multiverse.tree import (
    NOT_REPORTED,
    TreeContractError,
    build_tree_artifact,
    evaluate_tree_cv,
    root_split_bootstrap_frequency,
)


def _row(paper: str, direction: str, regime: str | None) -> dict[str, object]:
    return {
        "finding_id": f"{paper}:{direction}",
        "paper_id": paper,
        "effect_direction": direction,
        "mod__regime": regime,
    }


def _rows(per_class: int = 6) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(per_class):
        rows.append(_row(f"inc-{index}", "increase", "high"))
        rows.append(_row(f"dec-{index}", "decrease", "low"))
        rows.append(_row(f"nul-{index}", "no_effect", None))
    return rows


def test_tree_is_depth_limited_human_json_and_grouped_cv() -> None:
    artifact = build_tree_artifact(_rows(), ["regime"], seed=9, max_depth=3, supporting=True)
    assert artifact["status"] == "supporting"
    assert artifact["max_depth"] == 3
    assert artifact["nodes"]
    assert max(node["depth"] for node in artifact["nodes"]) <= 3
    assert all(isinstance(node, dict) for node in artifact["nodes"])
    assert artifact["cv"]["k"] >= 3
    for fold in artifact["cv"]["folds"]:
        assert set(fold["train_papers"]).isdisjoint(fold["test_papers"])


def test_not_reported_is_explicit_visually_distinct_and_not_narrated() -> None:
    artifact = build_tree_artifact(_rows(), ["regime"], seed=9)
    missing_nodes = [
        node
        for node in artifact["nodes"]
        if node.get("level") == NOT_REPORTED or node.get("missingness_path")
    ]
    assert missing_nodes
    assert any(node.get("display_level") == "not reported" for node in missing_nodes)
    assert all(node["narratable"] is False for node in missing_nodes)


def test_leaf_audit_greys_fewer_than_five_distinct_papers() -> None:
    rows = [_row(f"i-{index}", "increase", "high") for index in range(3)]
    rows.extend(_row(f"d-{index}", "decrease", "low") for index in range(3))
    artifact = build_tree_artifact(rows, ["regime"], seed=4)
    leaves = [node for node in artifact["nodes"] if node["type"] == "leaf"]
    assert leaves
    assert all(node["greyed"] for node in leaves)
    assert all(node["grey_reason"] == "fewer_than_5_distinct_papers" for node in leaves)
    assert all(node["narratable"] is False for node in leaves)


def test_root_split_bootstrap_frequency_keeps_all_draws_in_denominator() -> None:
    result = root_split_bootstrap_frequency(
        _rows(),
        ["regime"],
        expected_moderator="regime",
        n_bootstraps=25,
        seed=12,
    )
    assert result["n_bootstraps"] == 25
    assert len(result["root_splits"]) == 25
    assert result["frequency"] == pytest.approx(result["match_count"] / 25)
    assert result["frequency"] > 0.5


def test_tree_rejects_depth_above_registered_maximum() -> None:
    with pytest.raises(TreeContractError):
        build_tree_artifact(_rows(), ["regime"], max_depth=4)
    cv = evaluate_tree_cv([], ["regime"])
    assert cv["status"] == "insufficient_for_cv"
