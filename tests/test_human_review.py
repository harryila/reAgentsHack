from __future__ import annotations

import json
from pathlib import Path

import pytest

from literature_multiverse.closed_corpus import prepare_blinded_human_review_packet
from literature_multiverse.human_review import (
    HumanReviewContractError,
    evaluate_human_review_packet,
)


def _paper(index: int, *, eligible: bool, accepted: int) -> dict[str, object]:
    return {
        "paper_id": f"paper-{index}",
        "doc_id": f"DOC{index}",
        "screen_status": "included",
        "map_status": "success",
        "eligible": eligible,
        "accepted_finding_count": accepted,
    }


def _finding(index: int, paper_index: int) -> dict[str, object]:
    return {
        "finding_id": f"finding-{index}",
        "paper_id": f"paper-{paper_index}",
        "intervention": "treatment",
        "comparator": "control",
        "outcome_name": "outcome",
        "timepoint_raw": "week 4",
        "effect_direction": "positive",
        "effect_size_raw": "1.0",
        "p_value": 0.01,
        "significant": True,
        "evidence_quote": "Supported result.",
        "evidence_lines": ["L2"],
    }


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _completed(packet_row: dict[str, object], slot: str, *, eligible: bool) -> dict[str, object]:
    system = packet_row["system_output"]
    assert isinstance(system, dict)
    raw_findings = system["findings"]
    assert isinstance(raw_findings, list)
    findings = [
        {
            "audit_finding_id": finding["audit_finding_id"],
            "atomic": True,
            "supported_by_quote": True,
            "direction_correct": True,
            "pico_correct": True,
            "notes": None,
        }
        for finding in raw_findings
    ]
    return {
        "human_review_decision_version": "2",
        "audit_unit_id": packet_row["audit_unit_id"],
        "reviewer_slot": slot,
        "paper_eligible": eligible,
        "any_target_finding_missed": False,
        "all_emitted_findings_supported": True,
        "paper_direction_summary": "positive" if eligible else "not_applicable",
        "finding_decisions": findings,
        "error_codes": [],
        "notes": None,
        "review_minutes": 3.5,
    }


def _packet(tmp_path: Path) -> Path:
    packet_dir = tmp_path / "packet"
    prepare_blinded_human_review_packet(
        question_id="antiox",
        research_question="Does supplementation change adaptation?",
        eligibility_criteria=["controlled human study"],
        papers=[
            _paper(1, eligible=True, accepted=0),
            _paper(2, eligible=True, accepted=1),
            _paper(3, eligible=False, accepted=0),
        ],
        findings=[_finding(1, 2)],
        source_lines_by_doc_id={
            f"DOC{index}": {
                "L1": {"section": "Title", "text": f"Paper {index}"},
                "L2": {"section": "Results", "text": "Supported result."},
            }
            for index in range(1, 4)
        },
        output_dir=packet_dir,
        sample_size=3,
        seed=4,
    )
    return packet_dir / "manifest.json"


def test_blank_templates_report_no_result(tmp_path: Path) -> None:
    summary, conflicts = evaluate_human_review_packet(manifest_path=_packet(tmp_path))
    assert summary["status"] == "prepared_not_adjudicated"
    assert summary["completed_rows"] == {
        "reviewer_a_decisions": 0,
        "reviewer_b_decisions": 0,
    }
    assert conflicts == []
    assert summary["contains_paper_identifiers"] is False


def test_independent_conflict_requires_third_adjudicator(tmp_path: Path) -> None:
    manifest = _packet(tmp_path)
    packet = _jsonl(manifest.parent / "review_packet.private.jsonl")
    a_rows = [_completed(row, "reviewer_a", eligible=True) for row in packet]
    b_rows = [_completed(row, "reviewer_b", eligible=True) for row in packet]
    b_rows[0]["paper_eligible"] = False
    b_rows[0]["paper_direction_summary"] = "not_applicable"
    a_path = tmp_path / "completed-a.jsonl"
    b_path = tmp_path / "completed-b.jsonl"
    _write_jsonl(a_path, a_rows)
    _write_jsonl(b_path, b_rows)

    summary, conflicts = evaluate_human_review_packet(
        manifest_path=manifest,
        reviewer_a_path=a_path,
        reviewer_b_path=b_path,
    )
    assert summary["status"] == "awaiting_adjudication"
    assert summary["conflicting_items"] == 1
    assert summary["performance"] is None
    assert len(conflicts) == 1

    adjudication = _completed(packet[0], "adjudicator", eligible=True)
    adjudication_path = tmp_path / "adjudication.jsonl"
    _write_jsonl(adjudication_path, [adjudication])
    completed, _ = evaluate_human_review_packet(
        manifest_path=manifest,
        reviewer_a_path=a_path,
        reviewer_b_path=b_path,
        adjudicator_path=adjudication_path,
    )
    assert completed["status"] == "complete"
    assert completed["performance"]["pooled_diagnostic"]["items"] == 3
    assert completed["review_time"]["independent_reviewer_person_minutes"] == 21.0
    assert completed["review_time"]["adjudication_person_minutes"] == 3.5
    assert completed["review_time"]["total_person_minutes"] == 24.5


def test_packet_manifest_tampering_fails_closed(tmp_path: Path) -> None:
    manifest = _packet(tmp_path)
    (manifest.parent / "review_packet.private.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(HumanReviewContractError, match="hash_mismatch"):
        evaluate_human_review_packet(manifest_path=manifest)
