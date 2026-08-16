"""Config-frozen evidence-gap grid construction."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

from literature_multiverse.disagreement import CANONICAL_LABELS, normalized_entropy


class EvidenceGapContractError(ValueError):
    """Raised when the frozen evidence grid or rows are invalid."""


def _axis_value(row: Mapping[str, Any], name: str) -> Any:
    flattened = f"mod__{name}"
    if flattened in row:
        return row[flattened]
    moderators = row.get("moderators")
    if isinstance(moderators, Mapping) and name in moderators:
        return moderators[name]
    return row.get(name)


def _paper_modal_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    by_paper: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        direction = row.get("effect_direction")
        if direction in CANONICAL_LABELS:
            by_paper[row["paper_id"]][direction] += 1
    modes: Counter[str] = Counter()
    for counts in by_paper.values():
        maximum = max(counts.values())
        winners = [label for label in CANONICAL_LABELS if counts[label] == maximum]
        if len(winners) == 1:
            modes[winners[0]] += 1
    return modes


def build_evidence_gap_grid(
    findings: Sequence[Mapping[str, Any]],
    *,
    axis_levels: Mapping[str, Sequence[Any]],
    primary_endpoints: Sequence[str],
) -> list[dict[str, Any]]:
    """Emit the full frozen axis-level combination x endpoint Cartesian product."""

    if not axis_levels:
        raise EvidenceGapContractError("at least one Variant-B axis is required")
    if not primary_endpoints or len(set(primary_endpoints)) != len(primary_endpoints):
        raise EvidenceGapContractError("primary_endpoints must be non-empty and unique")
    axis_names = list(axis_levels)
    for name, levels in axis_levels.items():
        if not levels or len({_stable_value(level) for level in levels}) != len(levels):
            raise EvidenceGapContractError(f"axis {name!r} levels must be non-empty and unique")
        if any(level is None for level in levels):
            raise EvidenceGapContractError("missing is not a scientific evidence-grid level")

    finding_ids: set[str] = set()
    for row in findings:
        finding_id = row.get("finding_id")
        paper_id = row.get("paper_id")
        if not isinstance(finding_id, str) or not finding_id:
            raise EvidenceGapContractError("every evidence-gap row needs finding_id")
        if finding_id in finding_ids:
            raise EvidenceGapContractError(f"duplicate finding_id: {finding_id}")
        finding_ids.add(finding_id)
        if not isinstance(paper_id, str) or not paper_id:
            raise EvidenceGapContractError(f"finding {finding_id} needs paper_id")

    output: list[dict[str, Any]] = []
    level_products = product(*(axis_levels[name] for name in axis_names))
    for combination in level_products:
        axis_values = dict(zip(axis_names, combination, strict=True))
        for endpoint in primary_endpoints:
            cell_rows = [
                row
                for row in findings
                if row.get("outcome_name") == endpoint
                and all(
                    _stable_value(_axis_value(row, name)) == _stable_value(value)
                    for name, value in axis_values.items()
                )
            ]
            grounded_rows = [
                row
                for row in cell_rows
                if row.get("grounding_status") == "exact" and not bool(row.get("section_flagged"))
            ]
            classifiable_rows = [
                row for row in grounded_rows if row.get("effect_direction") in CANONICAL_LABELS
            ]
            total_papers = len({row["paper_id"] for row in cell_rows})
            grounded_papers = len({row["paper_id"] for row in grounded_rows})
            classifiable_papers = len({row["paper_id"] for row in classifiable_rows})
            modal_counts = _paper_modal_counts(classifiable_rows)
            represented_directions = sum(modal_counts[label] > 0 for label in CANONICAL_LABELS)
            entropy = None
            if (
                classifiable_papers >= 3
                and represented_directions >= 2
                and sum(modal_counts.values())
            ):
                entropy = normalized_entropy(
                    {label: float(modal_counts[label]) for label in CANONICAL_LABELS},
                    denominator_classes=3,
                )
            if grounded_papers == 0:
                status = "empty"
            elif grounded_papers < 5:
                status = "sparse"
            else:
                status = "supported"
            canonical_cell = json.dumps(
                {"primary_endpoint": endpoint, "axis_values": axis_values},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            output.append(
                {
                    "cell_id": "cell:" + hashlib.sha256(canonical_cell.encode()).hexdigest()[:16],
                    "primary_endpoint": endpoint,
                    "axis_values": axis_values,
                    "n_papers_total": total_papers,
                    "n_papers_grounded": grounded_papers,
                    "n_findings": len(cell_rows),
                    "grounded_fraction": len(grounded_rows) / len(cell_rows) if cell_rows else None,
                    "classifiable_fraction": (
                        len(classifiable_rows) / len(grounded_rows) if grounded_rows else None
                    ),
                    "paper_entropy": entropy,
                    "status": status,
                }
            )
    return output


def _stable_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


__all__ = ["EvidenceGapContractError", "build_evidence_gap_grid"]
