"""Deterministic human-audit sampling and exact quality denominators."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from literature_multiverse.grounding import reconcile_verification, verification_allows_primary

DEFAULT_AUDIT_FIELDS = (
    "eligibility",
    "atomicity",
    "intervention",
    "comparator",
    "outcome",
    "timepoint",
    "direction",
    "quote_support",
)
PRIMARY_DIRECTIONS = frozenset({"increase", "no_effect", "decrease"})


class AuditContractError(ValueError):
    """Raised when an audit or quality artifact cannot reconcile exactly."""


def wilson_interval(
    correct: int, total: int, *, z: float = 1.959963984540054
) -> list[float | None]:
    """Return a two-sided Wilson score interval (95% by default)."""

    if total < 0 or correct < 0 or correct > total:
        raise AuditContractError("audit counts must satisfy 0 <= correct <= total")
    if total == 0:
        return [None, None]
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _finding_id(row: Mapping[str, Any]) -> str:
    finding_id = row.get("finding_id")
    if not isinstance(finding_id, str) or not finding_id:
        raise AuditContractError("every audit candidate requires a finding_id")
    return finding_id


def audit_candidate_rows(
    papers: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    *,
    primary_family: str,
) -> list[dict[str, Any]]:
    """The audit-eligible pool: exact-grounded, unflagged, primary-family 3-class rows
    from successfully mapped eligible papers.  Shared by the sample freezer and the G3
    census check so both always see the same candidate count."""

    eligible_papers = {
        str(paper["paper_id"])
        for paper in papers
        if paper.get("screen_status") == "included"
        and paper.get("map_status") == "success"
        and paper.get("eligible") is True
    }
    return [
        dict(row)
        for row in findings
        if str(row.get("paper_id")) in eligible_papers
        and row.get("outcome_family") == primary_family
        and row.get("effect_direction") in PRIMARY_DIRECTIONS
        and row.get("grounding_status") == "exact"
        and not bool(row.get("section_flagged"))
        and not bool(row.get("quarantined", False))
    ]


def select_audit_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_size: int = 20,
    seed: int = 20260815,
    new_paper_ids: Sequence[str] = (),
    minimum_new_distinct_papers: int = 0,
    direction_key: str = "effect_direction",
    endpoint_key: str = "outcome_name",
) -> dict[str, Any]:
    """Select a fixed-seed sample balanced over paper, direction, and endpoint.

    For a scaled audit, ``minimum_new_distinct_papers`` is satisfied first with one row
    per new paper before the ordinary balanced fill.  Sampling never repeats a finding.
    """

    if sample_size < 1:
        raise AuditContractError("sample_size must be positive")
    candidates = [dict(row) for row in rows]
    ids = [_finding_id(row) for row in candidates]
    if len(ids) != len(set(ids)):
        raise AuditContractError("audit candidate finding IDs must be unique")
    if len(candidates) < sample_size:
        raise AuditContractError(
            f"audit requires {sample_size} candidates, found {len(candidates)}"
        )
    new_set = set(new_paper_ids)
    available_new_papers = {
        row.get("paper_id") for row in candidates if row.get("paper_id") in new_set
    }
    required_new = min(minimum_new_distinct_papers, sample_size)
    if len(available_new_papers) < required_new:
        raise AuditContractError(
            f"scaled audit requires {required_new} new distinct papers, found "
            f"{len(available_new_papers)}"
        )

    rng = random.Random(seed)
    order = list(range(len(candidates)))
    rng.shuffle(order)
    random_rank = {index: rank for rank, index in enumerate(order)}
    selected: list[int] = []
    selected_ids: set[str] = set()
    paper_counts: Counter[str] = Counter()
    stratum_counts: Counter[tuple[Any, Any]] = Counter()

    def choose(pool: Sequence[int], *, new_distinct_only: bool = False) -> int:
        eligible: list[int] = []
        for index in pool:
            row = candidates[index]
            if _finding_id(row) in selected_ids:
                continue
            paper_id = row.get("paper_id")
            if not isinstance(paper_id, str) or not paper_id:
                raise AuditContractError("every audit candidate requires paper_id")
            if new_distinct_only and (paper_id not in new_set or paper_counts[paper_id]):
                continue
            eligible.append(index)
        if not eligible:
            raise AuditContractError("audit balancing constraints exhausted the candidate pool")
        return min(
            eligible,
            key=lambda index: (
                paper_counts[candidates[index]["paper_id"]],
                stratum_counts[
                    (candidates[index].get(direction_key), candidates[index].get(endpoint_key))
                ],
                random_rank[index],
                _finding_id(candidates[index]),
            ),
        )

    for _ in range(required_new):
        index = choose(order, new_distinct_only=True)
        selected.append(index)
        row = candidates[index]
        selected_ids.add(_finding_id(row))
        paper_counts[row["paper_id"]] += 1
        stratum_counts[(row.get(direction_key), row.get(endpoint_key))] += 1
    while len(selected) < sample_size:
        index = choose(order)
        selected.append(index)
        row = candidates[index]
        selected_ids.add(_finding_id(row))
        paper_counts[row["paper_id"]] += 1
        stratum_counts[(row.get(direction_key), row.get(endpoint_key))] += 1

    sample = [candidates[index] for index in selected]
    sampled_new_papers = {row["paper_id"] for row in sample if row["paper_id"] in new_set}
    return {
        "seed": seed,
        "sample_size": sample_size,
        "finding_ids": [_finding_id(row) for row in sample],
        "rows": sample,
        "distinct_papers": len({row["paper_id"] for row in sample}),
        "sampled_new_distinct_papers": len(sampled_new_papers),
        "required_new_distinct_papers": required_new,
    }


def build_audit_artifact(
    sampled_finding_ids: Sequence[str],
    decisions: Sequence[Mapping[str, Any]],
    *,
    required_fields: Sequence[str] = DEFAULT_AUDIT_FIELDS,
) -> dict[str, Any]:
    """Reconcile human decisions and count a row correct only when every field passes."""

    sampled = list(sampled_finding_ids)
    if len(sampled) != len(set(sampled)):
        raise AuditContractError("sampled finding IDs must be unique")
    by_id: dict[str, Mapping[str, Any]] = {}
    for decision in decisions:
        finding_id = decision.get("finding_id")
        if finding_id in by_id:
            raise AuditContractError(f"duplicate audit decision: {finding_id}")
        if finding_id not in sampled:
            raise AuditContractError(f"unknown audit decision: {finding_id}")
        fields = decision.get("fields")
        if not isinstance(fields, Mapping):
            raise AuditContractError(f"audit decision {finding_id} requires fields")
        missing = set(required_fields) - fields.keys()
        extra = fields.keys() - set(required_fields)
        if missing or extra:
            raise AuditContractError(
                f"audit decision {finding_id} field mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        if any(not isinstance(fields[name], bool) for name in required_fields):
            raise AuditContractError(f"audit decision {finding_id} fields must be booleans")
        by_id[str(finding_id)] = decision
    missing_ids = set(sampled) - by_id.keys()
    if missing_ids:
        raise AuditContractError("missing audit decisions: " + ", ".join(sorted(missing_ids)))

    ordered: list[dict[str, Any]] = []
    taxonomy: Counter[str] = Counter()
    correct = 0
    for finding_id in sampled:
        source = by_id[finding_id]
        fields = dict(source["fields"])
        row_correct = all(fields.values())
        correct += row_correct
        for name, passed in fields.items():
            if not passed:
                taxonomy[name] += 1
        ordered.append(
            {
                "finding_id": finding_id,
                "fields": fields,
                "correct": row_correct,
                "adjudication": source.get("adjudication"),
                "notes": source.get("notes"),
            }
        )
    total = len(ordered)
    return {
        "decisions": ordered,
        "audit_correct": correct,
        "audit_total": total,
        "wilson_interval_95": wilson_interval(correct, total),
        "error_taxonomy": dict(sorted(taxonomy.items())),
        "passes_17_of_20": total == 20 and correct >= 17,
    }


def compute_quality_metrics(
    papers: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    verification_decisions: Sequence[Mapping[str, Any]],
    *,
    primary_family: str,
) -> dict[str, Any]:
    """Recompute the exact §4.5 quality and exclusion denominators from ledgers."""

    paper_by_id: dict[str, Mapping[str, Any]] = {}
    for paper in papers:
        paper_id = paper.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            raise AuditContractError("paper ledger contains an invalid paper_id")
        if paper_id in paper_by_id:
            raise AuditContractError(f"duplicate paper ledger ID: {paper_id}")
        paper_by_id[paper_id] = paper
    finding_ids: set[str] = set()
    for finding in findings:
        finding_id = _finding_id(finding)
        if finding_id in finding_ids:
            raise AuditContractError(f"duplicate finding ledger ID: {finding_id}")
        finding_ids.add(finding_id)
        if finding.get("paper_id") not in paper_by_id:
            raise AuditContractError(f"orphan finding: {finding_id}")

    successful_included = {
        paper_id
        for paper_id, paper in paper_by_id.items()
        if paper.get("screen_status") == "included" and paper.get("map_status") == "success"
    }
    eligible = {
        paper_id
        for paper_id in successful_included
        if paper_by_id[paper_id].get("eligible") is True
    }
    accepted_all = [
        finding
        for finding in findings
        if finding.get("paper_id") in eligible and not bool(finding.get("quarantined", False))
    ]
    accepted_primary = [
        finding for finding in accepted_all if finding.get("outcome_family") == primary_family
    ]
    accepted_denominator = len(accepted_primary)
    exact_primary = [
        finding for finding in accepted_primary if finding.get("grounding_status") == "exact"
    ]
    # Verification is requested for every exact-grounded accepted finding, including
    # secondary-family rows. The headline exclusion fraction below narrows back to the
    # canonical, unflagged primary-family denominator.
    exact_all = [finding for finding in accepted_all if finding.get("grounding_status") == "exact"]

    accepted_raw = sum(
        int(paper_by_id[paper_id].get("accepted_finding_count", 0))
        for paper_id in successful_included
    )
    quarantined_raw = sum(
        int(paper_by_id[paper_id].get("quarantined_finding_count", 0))
        for paper_id in successful_included
    )
    quarantine_denominator = accepted_raw + quarantined_raw

    requested_ids = [_finding_id(finding) for finding in exact_all]
    verification = reconcile_verification(requested_ids, verification_decisions)
    decisions_by_id = {item["finding_id"]: item for item in verification["decisions"]}
    verification_eligible = [
        finding
        for finding in exact_primary
        if not bool(finding.get("section_flagged"))
        and finding.get("effect_direction") in PRIMARY_DIRECTIONS
    ]
    verification_excluded = sum(
        not verification_allows_primary(decisions_by_id[_finding_id(finding)])
        for finding in verification_eligible
    )

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    mixed_or_unclear = sum(
        finding.get("effect_direction") in {"mixed", "unclear"} for finding in accepted_primary
    )
    section_flagged = sum(bool(finding.get("section_flagged")) for finding in accepted_primary)
    return {
        "grounded_fraction": {
            "numerator": len(exact_primary),
            "denominator": accepted_denominator,
            "value": ratio(len(exact_primary), accepted_denominator),
        },
        "quarantine_fraction": {
            "numerator": quarantined_raw,
            "denominator": quarantine_denominator,
            "value": ratio(quarantined_raw, quarantine_denominator),
            "g3_eligible": quarantine_denominator > 0,
        },
        "cross_model_agreement": {
            "numerator": verification["model_agree"],
            "denominator": verification["requested_exact_grounded"],
            "value": verification["cross_model_agreement"],
        },
        "mixed_or_unclear_fraction": {
            "numerator": mixed_or_unclear,
            "denominator": accepted_denominator,
            "value": ratio(mixed_or_unclear, accepted_denominator),
        },
        "section_flagged_fraction": {
            "numerator": section_flagged,
            "denominator": accepted_denominator,
            "value": ratio(section_flagged, accepted_denominator),
        },
        "verification_excluded_fraction": {
            "numerator": verification_excluded,
            "denominator": len(verification_eligible),
            "value": ratio(verification_excluded, len(verification_eligible)),
        },
    }


__all__ = [
    "DEFAULT_AUDIT_FIELDS",
    "AuditContractError",
    "audit_candidate_rows",
    "build_audit_artifact",
    "compute_quality_metrics",
    "select_audit_sample",
    "wilson_interval",
]
