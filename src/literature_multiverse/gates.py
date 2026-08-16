"""Audit finalization and exact G3 trust/story artifact construction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from literature_multiverse.analysis import derive_primary_cohort
from literature_multiverse.audit import (
    audit_candidate_rows,
    compute_quality_metrics,
    wilson_interval,
)
from literature_multiverse.cohort import cohort_sha256
from literature_multiverse.config import QuestionConfig, config_sha256
from literature_multiverse.disagreement import (
    CANONICAL_LABELS,
    bootstrap_disagreement,
    evaluate_g3,
    paper_balanced_finding_summary,
    paper_modal_summary,
)
from literature_multiverse.lineage import canonical_json_bytes, hash_canonical, sha256_bytes
from literature_multiverse.models import (
    AuditDecision,
    AuditRecord,
    VerificationRecord,
)


class GateContractError(ValueError):
    """An audit or G3 gate input failed exact reconciliation."""


def finalize_audit_record(
    *,
    seed: int,
    sampled_finding_ids: Sequence[str],
    raw_decisions: Sequence[Mapping[str, Any]],
    anchor_results: Mapping[str, bool],
    requested_sample_size: int = 20,
    newly_added_eligible_papers: int | None = None,
    newly_added_audit_eligible_papers: int | None = None,
    sampled_new_distinct_papers: int | None = None,
) -> AuditRecord:
    """Create the strict human-audit artifact from a frozen sample and decisions."""

    sampled = list(sampled_finding_ids)
    if len(sampled) != len(set(sampled)):
        raise GateContractError("audit_sample_ids_must_be_unique")
    by_id: dict[str, AuditDecision] = {}
    taxonomy: Counter[str] = Counter()
    for raw in raw_decisions:
        finding_id = raw.get("finding_id")
        if not isinstance(finding_id, str):
            raise GateContractError("audit_decision_missing_finding_id")
        if finding_id in by_id:
            raise GateContractError(f"audit_decision_duplicate:{finding_id}")
        if finding_id not in sampled:
            raise GateContractError(f"audit_decision_unknown:{finding_id}")
        payload = dict(raw)
        if "checks" not in payload and "fields" in payload:
            payload["checks"] = payload.pop("fields")
        try:
            decision = AuditDecision.model_validate(payload)
        except ValueError as exc:
            raise GateContractError(f"audit_decision_invalid:{finding_id}") from exc
        by_id[finding_id] = decision
        for name, passed in decision.checks.model_dump().items():
            if not passed:
                taxonomy[name] += 1
        taxonomy.update(decision.error_codes)
    missing = set(sampled) - set(by_id)
    if missing:
        raise GateContractError("audit_decisions_missing:" + ",".join(sorted(missing)))
    if not anchor_results or any(not isinstance(value, bool) for value in anchor_results.values()):
        raise GateContractError("audit_anchor_results_invalid")
    decisions = [by_id[finding_id] for finding_id in sampled]
    correct = sum(decision.correct for decision in decisions)
    interval = wilson_interval(correct, len(decisions))
    if interval[0] is None or interval[1] is None:
        raise GateContractError("audit_wilson_interval_undefined")
    return AuditRecord(
        audit_version="1",
        seed=seed,
        requested_sample_size=requested_sample_size,
        sampled_finding_ids=sampled,
        decisions=decisions,
        anchor_results=dict(anchor_results),
        correct_count=correct,
        total_count=len(decisions),
        wilson_interval=(interval[0], interval[1]),
        error_taxonomy=dict(sorted(taxonomy.items())),
        newly_added_eligible_papers=newly_added_eligible_papers,
        newly_added_audit_eligible_papers=newly_added_audit_eligible_papers,
        sampled_new_distinct_papers=sampled_new_distinct_papers,
    )


def build_g3_artifact(
    *,
    config: QuestionConfig,
    papers: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    verification: VerificationRecord | Mapping[str, Any],
    audit: AuditRecord | Mapping[str, Any],
    g1b_passed: bool,
) -> dict[str, Any]:
    """Recompute every trust/story value and bind it to exact input hashes."""

    verification_record = (
        verification
        if isinstance(verification, VerificationRecord)
        else VerificationRecord.model_validate(verification)
    )
    audit_record = audit if isinstance(audit, AuditRecord) else AuditRecord.model_validate(audit)
    expected_anchors = {
        anchor.paper_id for anchor in (config.anchor_papers or [])
    }
    if set(audit_record.anchor_results) != expected_anchors:
        raise GateContractError("audit_anchor_result_set_mismatch")
    assert config.outcomes.primary_family is not None
    primary_rows = derive_primary_cohort(
        papers,
        findings,
        verification_record,
        primary_family=config.outcomes.primary_family,
    )
    decisions = [decision.model_dump(mode="json") for decision in verification_record.decisions]
    quality = compute_quality_metrics(
        papers,
        findings,
        decisions,
        primary_family=config.outcomes.primary_family,
    )

    if primary_rows:
        finding_summary = paper_balanced_finding_summary(primary_rows)
        paper_summary = paper_modal_summary(primary_rows)
        entropy_bootstrap = bootstrap_disagreement(
            primary_rows,
            n_bootstraps=1000,
            seed=config.analysis.seed,
        )
        interval = entropy_bootstrap["paper_entropy"]["interval_90"]
        classifiable = paper_summary["n_papers_classifiable"]
        distinct_by_direction = finding_summary["distinct_papers_by_class"]
    else:
        finding_summary = None
        paper_summary = None
        entropy_bootstrap = {
            "seed": config.analysis.seed,
            "n_bootstraps": 1000,
            "finding_entropy": {"interval_90": [None, None], "invalid_draws": 1000},
            "paper_entropy": {"interval_90": [None, None], "invalid_draws": 1000},
        }
        interval = [None, None]
        classifiable = 0
        distinct_by_direction = {label: 0 for label in CANONICAL_LABELS}

    audit_candidates = audit_candidate_rows(
        papers, findings, primary_family=config.outcomes.primary_family
    )
    failed_audit_ids = {
        decision.finding_id for decision in audit_record.decisions if not decision.correct
    }
    cohort_finding_ids = {str(row.get("finding_id")) for row in primary_rows}
    release_request_ids = {
        str(row["finding_id"]) for row in audit_candidates
    } - failed_audit_ids
    release_agree = sum(
        decision.model_status == "agree"
        for decision in verification_record.decisions
        if decision.finding_id in release_request_ids
    )
    release_set_agreement = (
        release_agree / len(release_request_ids) if release_request_ids else None
    )
    gate = evaluate_g3(
        audit_correct=audit_record.correct_count,
        audit_total=audit_record.total_count,
        anchors_passed=sum(audit_record.anchor_results.values()),
        anchors_total=len(audit_record.anchor_results),
        cross_model_agreement=quality["cross_model_agreement"]["value"],
        quarantine_fraction=quality["quarantine_fraction"]["value"],
        g1b_passed=g1b_passed,
        paper_entropy_interval_90=interval,
        classifiable_papers=classifiable,
        distinct_papers_by_direction=distinct_by_direction,
        audit_candidate_count=len(audit_candidates),
        audit_failed_ids_in_cohort=sorted(failed_audit_ids & cohort_finding_ids),
        release_set_cross_model_agreement=release_set_agreement,
    )
    cohort_hash = cohort_sha256(primary_rows)
    return {
        "g3_version": "1",
        **gate,
        "cohort_sha256": cohort_hash,
        "config_sha256": config_sha256(config),
        "audit_sha256": sha256_bytes(canonical_json_bytes(audit_record) + b"\n"),
        "verification_sha256": sha256_bytes(
            canonical_json_bytes(verification_record) + b"\n"
        ),
        "input_hashes": {
            "papers": hash_canonical(list(papers)),
            "findings": hash_canonical(list(findings)),
        },
        "quality": quality,
        "disagreement": {
            "finding": finding_summary,
            "paper": paper_summary,
            "bootstrap": {
                "seed": entropy_bootstrap["seed"],
                "n_bootstraps": entropy_bootstrap["n_bootstraps"],
                "finding_entropy_interval_90": entropy_bootstrap["finding_entropy"][
                    "interval_90"
                ],
                "finding_invalid_draws": entropy_bootstrap["finding_entropy"][
                    "invalid_draws"
                ],
                "paper_entropy_interval_90": interval,
                "paper_invalid_draws": entropy_bootstrap["paper_entropy"]["invalid_draws"],
            },
        },
    }


__all__ = ["GateContractError", "build_g3_artifact", "finalize_audit_record"]
