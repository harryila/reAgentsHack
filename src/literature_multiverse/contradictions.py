"""Residual opposite-direction pair discovery with citation-preserving output."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_CONTEXT_FIELDS = (
    "intervention_class",
    "comparator",
    "population_state",
    "timing_context",
    "timepoint_raw",
    "outcome_name",
)
OPPOSITE = {("increase", "decrease"), ("decrease", "increase")}


class ContradictionContractError(ValueError):
    """Raised when residual pairs cannot be reconciled to their sources."""


def _present(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _canonical(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _citation(row: Mapping[str, Any], paper: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": paper["paper_id"],
        "finding_id": row["finding_id"],
        "title": paper.get("title"),
        "doi": paper.get("doi"),
        "doc_id": paper.get("doc_id"),
        "evidence_quote": row.get("evidence_quote"),
        "evidence_lines": list(row.get("evidence_lines") or []),
        "evidence_section": row.get("evidence_section"),
    }


def find_contradictions(
    findings: Sequence[Mapping[str, Any]],
    papers: Sequence[Mapping[str, Any]],
    *,
    primary_family: str,
    context_fields: Sequence[str] = DEFAULT_CONTEXT_FIELDS,
    minimum_shared_fields: int = 2,
    numeric_ranges: Mapping[str, tuple[float, float]] | None = None,
    require_primary_grounded: bool = True,
) -> list[dict[str, Any]]:
    """Return deterministic cross-paper increase/decrease residual pairs.

    Distance is a declared Gower-like average over fields reported by both rows:
    categorical components are 0/1 and numeric components are range-scaled.  Eligibility
    additionally requires at least ``minimum_shared_fields`` exact shared values.
    """

    if minimum_shared_fields < 1:
        raise ContradictionContractError("minimum_shared_fields must be positive")
    paper_by_id: dict[str, Mapping[str, Any]] = {}
    for paper in papers:
        paper_id = paper.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            raise ContradictionContractError("paper ledger contains invalid paper_id")
        if paper_id in paper_by_id:
            raise ContradictionContractError(f"duplicate paper_id: {paper_id}")
        paper_by_id[paper_id] = paper

    eligible: list[Mapping[str, Any]] = []
    finding_ids: set[str] = set()
    for row in findings:
        finding_id = row.get("finding_id")
        paper_id = row.get("paper_id")
        if not isinstance(finding_id, str) or not finding_id:
            raise ContradictionContractError("finding requires finding_id")
        if finding_id in finding_ids:
            raise ContradictionContractError(f"duplicate finding_id: {finding_id}")
        finding_ids.add(finding_id)
        if paper_id not in paper_by_id:
            raise ContradictionContractError(f"orphan finding: {finding_id}")
        if row.get("outcome_family") != primary_family:
            continue
        if row.get("effect_direction") not in {"increase", "decrease"}:
            continue
        if require_primary_grounded and (
            row.get("grounding_status") != "exact" or bool(row.get("section_flagged"))
        ):
            continue
        eligible.append(row)

    ranges = dict(numeric_ranges or {})
    results: list[dict[str, Any]] = []
    for left_index, left in enumerate(eligible):
        for right in eligible[left_index + 1 :]:
            if left["paper_id"] == right["paper_id"]:
                continue
            if (left["effect_direction"], right["effect_direction"]) not in OPPOSITE:
                continue
            components: list[dict[str, Any]] = []
            shared: list[str] = []
            for field in context_fields:
                left_value = left.get(field)
                right_value = right.get(field)
                if not (_present(left_value) and _present(right_value)):
                    continue
                if (
                    field in ranges
                    and isinstance(left_value, (int, float))
                    and isinstance(right_value, (int, float))
                ):
                    lower, upper = ranges[field]
                    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
                        raise ContradictionContractError(f"invalid numeric range for {field}")
                    distance = min(
                        1.0, abs(float(left_value) - float(right_value)) / (upper - lower)
                    )
                    equal = math.isclose(distance, 0.0, abs_tol=1e-12)
                else:
                    equal = _canonical(left_value) == _canonical(right_value)
                    distance = 0.0 if equal else 1.0
                if equal:
                    shared.append(field)
                components.append(
                    {
                        "field": field,
                        "left": left_value,
                        "right": right_value,
                        "distance": distance,
                        "shared": equal,
                    }
                )
            if len(shared) < minimum_shared_fields or not components:
                continue
            ordered_ids = sorted([left["finding_id"], right["finding_id"]])
            pair_id = "pair:" + hashlib.sha256("\x1f".join(ordered_ids).encode()).hexdigest()[:16]
            distance = sum(component["distance"] for component in components) / len(components)
            left_paper = paper_by_id[left["paper_id"]]
            right_paper = paper_by_id[right["paper_id"]]
            results.append(
                {
                    "pair_id": pair_id,
                    "outcome_family": primary_family,
                    "left_direction": left["effect_direction"],
                    "right_direction": right["effect_direction"],
                    "shared_context_fields": shared,
                    "shared_context_count": len(shared),
                    "distance": distance,
                    "distance_components": components,
                    "left_citation": _citation(left, left_paper),
                    "right_citation": _citation(right, right_paper),
                }
            )
    return sorted(results, key=lambda item: (item["distance"], item["pair_id"]))


def residual_summary(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the exact pre-approved residual sentence branch."""

    pair_count = len(pairs)
    if pair_count:
        return {
            "pair_count": pair_count,
            "top_pair_id": pairs[0]["pair_id"],
            "rendered_sentence": (
                "This top pair remains opposite despite matching on at least two recorded "
                "conditions; both source passages and line references are shown."
            ),
        }
    return {
        "pair_count": 0,
        "top_pair_id": None,
        "rendered_sentence": (
            "No grounded opposite-direction pair met the matching rule; the empty residual "
            "view is part of the result."
        ),
    }


__all__ = [
    "DEFAULT_CONTEXT_FIELDS",
    "ContradictionContractError",
    "find_contradictions",
    "residual_summary",
]
