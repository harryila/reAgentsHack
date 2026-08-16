from __future__ import annotations

from copy import deepcopy

import pytest

from literature_multiverse.gates import (
    GateContractError,
    build_g3_artifact,
    finalize_audit_record,
)
from literature_multiverse.models import VerificationRecord


def _gate_inputs(fixture_config, hash64: str):
    papers = []
    findings = []
    decisions = []
    for index in range(20):
        paper_id = f"doc:p{index:02d}"
        finding_id = f"f{index:02d}"
        direction = "increase" if index < 10 else "decrease"
        papers.append(
            {
                "paper_id": paper_id,
                "screen_status": "included",
                "map_status": "success",
                "eligible": True,
                "accepted_finding_count": 1,
                "quarantined_finding_count": 0,
            }
        )
        findings.append(
            {
                "finding_id": finding_id,
                "paper_id": paper_id,
                "effect_direction": direction,
                "outcome_family": fixture_config.outcomes.primary_family,
                "grounding_status": "exact",
                "section_flagged": False,
                "quarantined": False,
            }
        )
        decisions.append(
            {"finding_id": finding_id, "model_status": "agree", "adjudication": "none"}
        )
    verification = VerificationRecord(
        provider="fixture",
        model="fixture-verifier",
        prompt_version="1",
        prompt_sha256=hash64,
        requested_finding_ids=[row["finding_id"] for row in findings],
        decisions=decisions,
    )
    checks = {
        "eligibility": True,
        "atomicity": True,
        "intervention": True,
        "comparator": True,
        "outcome": True,
        "timepoint": True,
        "direction": True,
        "quote_support": True,
    }
    audit = finalize_audit_record(
        seed=fixture_config.analysis.seed,
        sampled_finding_ids=[row["finding_id"] for row in findings],
        raw_decisions=[
            {"finding_id": row["finding_id"], "checks": checks} for row in findings
        ],
        anchor_results={fixture_config.anchor_papers[0].paper_id: True},
    )
    return papers, findings, verification, audit


def test_g3_artifact_passes_trust_and_story_on_balanced_fixture(
    fixture_config, hash64: str
) -> None:
    papers, findings, verification, audit = _gate_inputs(fixture_config, hash64)
    artifact = build_g3_artifact(
        config=fixture_config,
        papers=papers,
        findings=findings,
        verification=verification,
        audit=audit,
        g1b_passed=True,
    )
    assert artifact["trust_passed"] is True
    assert artifact["story_passed"] is True
    assert artifact["action"] == "run_m4"
    assert artifact["quality"]["cross_model_agreement"]["value"] == 1.0
    assert artifact["disagreement"]["bootstrap"]["n_bootstraps"] == 1000
    assert artifact["disagreement"]["bootstrap"]["paper_entropy_interval_90"][0] >= 0.4


def test_g3_trust_failure_blocks_even_when_story_passes(fixture_config, hash64: str) -> None:
    papers, findings, verification, audit = _gate_inputs(fixture_config, hash64)
    decisions = [decision.model_dump(mode="json") for decision in audit.decisions]
    for decision in decisions[:4]:
        decision["checks"]["direction"] = False
    failed_audit = finalize_audit_record(
        seed=audit.seed,
        sampled_finding_ids=audit.sampled_finding_ids,
        raw_decisions=decisions,
        anchor_results=audit.anchor_results,
    )
    artifact = build_g3_artifact(
        config=fixture_config,
        papers=papers,
        findings=findings,
        verification=verification,
        audit=failed_audit,
        g1b_passed=True,
    )
    assert artifact["trust_passed"] is False
    assert artifact["story_passed"] is True
    assert artifact["action"] == "block_release"


def test_audit_rejects_missing_decision_and_anchor_set_mismatch(
    fixture_config, hash64: str
) -> None:
    papers, findings, verification, audit = _gate_inputs(fixture_config, hash64)
    with pytest.raises(GateContractError, match="audit_decisions_missing"):
        finalize_audit_record(
            seed=audit.seed,
            sampled_finding_ids=audit.sampled_finding_ids,
            raw_decisions=[
                decision.model_dump(mode="json") for decision in audit.decisions[:-1]
            ],
            anchor_results=audit.anchor_results,
        )
    altered = deepcopy(audit.model_dump(mode="json"))
    altered["anchor_results"] = {"doc:not-registered": True}
    with pytest.raises(GateContractError, match="anchor_result_set_mismatch"):
        build_g3_artifact(
            config=fixture_config,
            papers=papers,
            findings=findings,
            verification=verification,
            audit=altered,
            g1b_passed=True,
        )


def test_g3_census_audit_rule_demands_no_failed_row_in_cohort() -> None:
    from literature_multiverse.disagreement import evaluate_g3

    base = dict(
        anchors_passed=2,
        anchors_total=2,
        cross_model_agreement=0.95,
        quarantine_fraction=0.02,
        g1b_passed=True,
        paper_entropy_interval_90=[0.5, 0.9],
        classifiable_papers=25,
        distinct_papers_by_direction={"increase": 6, "no_effect": 8, "decrease": 5},
    )
    # Census (pool of 13, all audited): a failed row still in the cohort blocks trust.
    blocked = evaluate_g3(
        audit_correct=12,
        audit_total=13,
        audit_candidate_count=13,
        audit_failed_ids_in_cohort=["doc:bad"],
        **base,
    )
    assert blocked["trust_passed"] is False
    assert blocked["action"] == "block_release"
    # Same census, failed row excluded from the cohort by verification: trust passes.
    passed = evaluate_g3(
        audit_correct=12,
        audit_total=13,
        audit_candidate_count=13,
        audit_failed_ids_in_cohort=[],
        **base,
    )
    assert passed["trust_passed"] is True
    assert passed["action"] == "run_m4"
    # Sampling mode is untouched: 20-of-many still requires >=17 correct.
    sampled = evaluate_g3(
        audit_correct=16,
        audit_total=20,
        audit_candidate_count=57,
        audit_failed_ids_in_cohort=[],
        **base,
    )
    assert sampled["trust_passed"] is False


def test_g3_census_cross_model_rule_uses_release_set_rate() -> None:
    from literature_multiverse.disagreement import evaluate_g3

    base = dict(
        audit_correct=12,
        audit_total=13,
        audit_candidate_count=13,
        audit_failed_ids_in_cohort=[],
        anchors_passed=2,
        anchors_total=2,
        quarantine_fraction=0.02,
        g1b_passed=True,
        paper_entropy_interval_90=[0.5, 0.9],
        classifiable_papers=25,
        distinct_papers_by_direction={"increase": 6, "no_effect": 8, "decrease": 5},
    )
    census = evaluate_g3(
        cross_model_agreement=26 / 43,
        release_set_cross_model_agreement=11 / 12,
        **base,
    )
    rule = next(r for r in census["trust_rules"] if r["name"] == "cross_model_agreement")
    assert rule["passed"] is True
    assert rule["observed"]["all_exact_grounded"] == 26 / 43
    assert census["trust_passed"] is True

    below = evaluate_g3(
        cross_model_agreement=26 / 43,
        release_set_cross_model_agreement=0.80,
        **base,
    )
    assert below["trust_passed"] is False

    # Without a release-set rate (sampling mode), the original full-rate rule applies.
    sampling = dict(base, audit_correct=17, audit_total=20, audit_candidate_count=60)
    original = evaluate_g3(cross_model_agreement=26 / 43, **sampling)
    assert original["trust_passed"] is False
