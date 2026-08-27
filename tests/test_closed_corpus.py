"""Closed-corpus denominator, oracle-ablation, and human-packet contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from literature_multiverse.closed_corpus import (
    FORBIDDEN_REVIEWER_KEYS,
    ClosedCorpusContractError,
    ClosedCorpusGoldQuestion,
    ClosedCorpusPrediction,
    build_oracle_corpus_prediction,
    build_oracle_extraction_prediction,
    build_paper_audit_candidates,
    evaluate_closed_corpus,
    prepare_blinded_human_review_packet,
    select_paper_audit_candidates,
)


def _gold() -> list[ClosedCorpusGoldQuestion]:
    return [
        ClosedCorpusGoldQuestion(
            question_id="q1",
            split="test",
            gold_paper_ids=["a", "b"],
            gold_conclusion="positive",
        ),
        ClosedCorpusGoldQuestion(
            question_id="q2",
            split="test",
            gold_paper_ids=["c"],
            gold_conclusion="negative",
        ),
    ]


def _system_predictions() -> list[ClosedCorpusPrediction]:
    return [
        ClosedCorpusPrediction(
            question_id="q1",
            arm="system",
            retrieval_source="system",
            extraction_source="system",
            retrieved_paper_ids=["a", "b", "x"],
            extracted_paper_ids=["a"],
            predicted_conclusion="positive",
        ),
        ClosedCorpusPrediction(
            question_id="q2",
            arm="system",
            retrieval_source="not_run",
            extraction_source="not_run",
            predicted_conclusion="negative",
        ),
    ]


def _paper(index: int, *, eligible: bool, accepted: int) -> dict[str, Any]:
    return {
        "paper_id": f"paper-{index}",
        "doc_id": f"DOC{index}",
        "screen_status": "included",
        "map_status": "success",
        "eligible": eligible,
        "accepted_finding_count": accepted,
    }


def _finding(index: int, paper_index: int) -> dict[str, Any]:
    return {
        "finding_id": f"finding-{index}",
        "paper_id": f"paper-{paper_index}",
        "intervention": "vitamin",
        "comparator": "placebo",
        "outcome_name": "strength",
        "timepoint_raw": "12 weeks",
        "effect_direction": "increase",
        "effect_size_raw": None,
        "p_value": 0.04,
        "significant": True,
        "evidence_quote": "Supported result.",
        "evidence_lines": ["L2"],
        "confidence": 0.99,
    }


def _find_forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in FORBIDDEN_REVIEWER_KEYS:
                found.add(key.casefold())
            found.update(_find_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_find_forbidden_keys(item))
    return found


def test_gold_papers_must_be_nonempty_sorted_unique() -> None:
    with pytest.raises(ValidationError, match="gold_paper_ids_must_be_sorted_unique"):
        ClosedCorpusGoldQuestion(
            question_id="q",
            split="test",
            gold_paper_ids=["b", "a"],
        )
    with pytest.raises(ValidationError, match="gold_paper_ids_must_be_nonempty"):
        ClosedCorpusGoldQuestion(question_id="q", split="test", gold_paper_ids=[])


def test_prediction_extractions_must_be_subset_of_retrieval() -> None:
    with pytest.raises(ValidationError, match="extracted_papers_must_be_subset"):
        ClosedCorpusPrediction(
            question_id="q",
            arm="system",
            retrieval_source="system",
            extraction_source="system",
            retrieved_paper_ids=["a"],
            extracted_paper_ids=["b"],
        )


def test_zero_finding_gold_paper_remains_in_end_to_end_denominator() -> None:
    result = evaluate_closed_corpus(gold=_gold(), predictions=_system_predictions())
    system = result["arms"]["system"]

    assert system["retrieval"]["gold_papers"] == 3
    assert system["retrieval"]["hits"] == 2
    assert system["retrieval"]["micro_recall_missing_as_zero"] == pytest.approx(2 / 3)
    assert system["extraction"]["gold_papers_with_one_or_more_findings"] == 1
    assert (
        system["extraction"]["gold_papers_with_zero_extracted_findings_after_retrieval"]
        == 1
    )
    assert system["extraction"]["end_to_end_micro_paper_recall"] == pytest.approx(1 / 3)
    assert system["conclusion"]["strict_accuracy_with_unanswered_as_errors"] == 1.0
    assert result["arms"]["oracle_corpus"]["status"] == "not_run"


def test_oracle_corpus_must_retrieve_exact_gold() -> None:
    invalid = ClosedCorpusPrediction(
        question_id="q1",
        arm="oracle_corpus",
        retrieval_source="evaluator_oracle",
        extraction_source="system",
        retrieved_paper_ids=["a"],
        extracted_paper_ids=["a"],
        predicted_conclusion="positive",
    )
    with pytest.raises(ClosedCorpusContractError, match="oracle_retrieval_not_exact_gold"):
        evaluate_closed_corpus(gold=_gold(), predictions=[invalid])


def test_oracle_corpus_builder_fixes_retrieval_only() -> None:
    prediction = build_oracle_corpus_prediction(
        gold=_gold()[0],
        extracted_paper_ids=["a"],
        predicted_conclusion="positive",
    )
    assert prediction.retrieved_paper_ids == ["a", "b"]
    assert prediction.extracted_paper_ids == ["a"]
    assert prediction.retrieval_source == "evaluator_oracle"
    assert prediction.extraction_source == "system"


def test_oracle_extraction_does_not_copy_gold_conclusion() -> None:
    prediction = build_oracle_extraction_prediction(
        gold=_gold()[0],
        predicted_conclusion="negative",
    )
    result = evaluate_closed_corpus(gold=[_gold()[0]], predictions=[prediction])
    oracle = result["arms"]["oracle_extraction"]
    assert oracle["retrieval"]["micro_recall_missing_as_zero"] == 1.0
    assert oracle["extraction"]["end_to_end_micro_paper_recall"] == 1.0
    assert oracle["conclusion"]["correct"] == 0


def test_duplicate_or_unknown_predictions_are_rejected() -> None:
    prediction = _system_predictions()[0]
    with pytest.raises(ClosedCorpusContractError, match="duplicate_prediction"):
        evaluate_closed_corpus(gold=_gold(), predictions=[prediction, prediction])
    unknown = prediction.model_copy(update={"question_id": "unknown"})
    with pytest.raises(ClosedCorpusContractError, match="outside_gold"):
        evaluate_closed_corpus(gold=_gold(), predictions=[unknown])


def test_paper_candidates_include_pipeline_eligible_zero_findings() -> None:
    papers = [
        _paper(1, eligible=True, accepted=0),
        _paper(2, eligible=True, accepted=1),
        _paper(3, eligible=False, accepted=0),
    ]
    candidates = build_paper_audit_candidates(papers, [_finding(1, 2)])
    by_id = {row["paper_id"]: row for row in candidates}
    assert by_id["paper-1"]["selection_stratum"] == "pipeline_eligible_zero_findings"
    assert by_id["paper-1"]["findings"] == []
    assert by_id["paper-3"]["selection_stratum"] == "pipeline_ineligible_zero_findings"


def test_audit_selection_always_includes_all_eligible_zero_finding_papers() -> None:
    papers = [
        _paper(1, eligible=True, accepted=0),
        _paper(2, eligible=True, accepted=0),
        _paper(3, eligible=True, accepted=1),
        _paper(4, eligible=False, accepted=0),
        _paper(5, eligible=False, accepted=0),
    ]
    candidates = build_paper_audit_candidates(papers, [_finding(1, 3)])
    selected = select_paper_audit_candidates(candidates, sample_size=3, seed=7)
    assert {"paper-1", "paper-2", "paper-3"} == {
        row["paper_id"] for row in selected
    }


def test_audit_selection_refuses_to_invent_items() -> None:
    candidates = build_paper_audit_candidates(
        [_paper(1, eligible=True, accepted=0)], []
    )
    with pytest.raises(ClosedCorpusContractError, match="candidate_pool_too_small"):
        select_paper_audit_candidates(candidates, sample_size=2)


def test_blinded_packet_hides_confidence_ranking_and_private_identity(tmp_path: Path) -> None:
    papers = [
        _paper(1, eligible=True, accepted=0),
        _paper(2, eligible=True, accepted=1),
        _paper(3, eligible=False, accepted=0),
        _paper(4, eligible=False, accepted=0),
    ]
    source_lines = {
        f"DOC{index}": {
            "L1": {"section": "Title", "text": f"Paper {index}"},
            "L2": {"section": "Results", "text": "Supported result."},
        }
        for index in range(1, 5)
    }
    output_dir = tmp_path / "packet"
    manifest = prepare_blinded_human_review_packet(
        question_id="antiox",
        research_question="Does supplementation change training adaptation?",
        eligibility_criteria=["controlled human training study"],
        papers=papers,
        findings=[_finding(1, 2)],
        source_lines_by_doc_id=source_lines,
        output_dir=output_dir,
        sample_size=3,
        seed=8,
    )

    packet = [
        json.loads(line)
        for line in (output_dir / "review_packet.private.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest["all_eligible_zero_finding_papers_included"] is True
    assert manifest["all_pipeline_eligible_papers_included"] is True
    assert manifest["contains_model_confidence"] is False
    assert manifest["manifest_contains_article_text"] is False
    assert manifest["review_packet_contains_article_text"] is True
    assert _find_forbidden_keys(packet) == set()
    assert all("paper_id" not in row and "doc_id" not in row for row in packet)
    assert any(row["system_output"] == {"eligible": True, "findings": []} for row in packet)
    finding_item = next(row for row in packet if row["system_output"]["findings"])
    assert len(finding_item["review_form"]["finding_decisions"]) == 1
    assert finding_item["review_form"]["finding_decisions"][0]["direction_correct"] is None
    key_text = (output_dir / "identity_key.private.jsonl").read_text(encoding="utf-8")
    assert "selection_stratum" in key_text
    assert len(
        (output_dir / "reviewer_a_decisions.private.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 3
    assert len(
        (output_dir / "reviewer_b_decisions.private.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 3


def test_packet_requires_source_text_for_every_selected_paper(tmp_path: Path) -> None:
    with pytest.raises(ClosedCorpusContractError, match="audit_source_lines_missing"):
        prepare_blinded_human_review_packet(
            question_id="q",
            research_question="Question?",
            eligibility_criteria=["criterion"],
            papers=[_paper(1, eligible=True, accepted=0)],
            findings=[],
            source_lines_by_doc_id={},
            output_dir=tmp_path / "packet",
            sample_size=1,
        )
