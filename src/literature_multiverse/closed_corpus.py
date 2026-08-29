"""Closed-corpus evaluation and confidence-blinded human-review packets.

The evaluator keeps retrieval, extraction coverage, and final synthesis separate.  In
particular, a gold paper that is retrieved but produces no accepted finding remains in
the extraction-recall denominator.  Oracle-corpus and oracle-extraction arms are
explicit evaluator interventions, not labels that a system may attach to an ordinary
run.

Human-review packets are paper-level.  Every successfully mapped, screened-in paper is
eligible for sampling, and every paper that the pipeline marked eligible but from which
it emitted zero findings is included before random fill.  The reviewer-facing schema is
closed and deliberately has no confidence, score, priority, rank, disagreement, or
selection-stratum field.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from literature_multiverse.lineage import (
    atomic_write_json,
    atomic_write_jsonl,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.models import ContractModel

AblationArm = Literal["system", "oracle_corpus", "oracle_extraction"]
EvidenceSource = Literal["system", "evaluator_oracle", "not_run"]

ABLATION_ARMS: tuple[AblationArm, ...] = (
    "system",
    "oracle_corpus",
    "oracle_extraction",
)

# These names are forbidden recursively in reviewer-facing objects.  The packet shows
# the extraction itself because that is what the reviewer adjudicates; it does not show
# any signal used to rank or express model certainty about the item.
FORBIDDEN_REVIEWER_KEYS = frozenset(
    {
        "confidence",
        "cost",
        "disagreement",
        "influence",
        "priority",
        "rank",
        "risk",
        "score",
        "selection_stratum",
    }
)


class ClosedCorpusContractError(ValueError):
    """A closed-corpus input, ablation, or audit packet is not auditable."""


def _sorted_unique_nonempty(values: list[str], *, field: str) -> list[str]:
    if not values or any(not value for value in values):
        raise ValueError(f"{field}_must_be_nonempty")
    if values != sorted(set(values)):
        raise ValueError(f"{field}_must_be_sorted_unique")
    return values


class ClosedCorpusGoldQuestion(ContractModel):
    """Private evaluator truth for one question and its frozen included-paper set."""

    closed_corpus_gold_version: Literal["1"] = "1"
    question_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    gold_paper_ids: list[str]
    gold_conclusion: str | None = None

    @field_validator("gold_paper_ids")
    @classmethod
    def validate_gold_papers(cls, value: list[str]) -> list[str]:
        return _sorted_unique_nonempty(value, field="gold_paper_ids")

    @field_validator("gold_conclusion")
    @classmethod
    def validate_gold_conclusion(cls, value: str | None) -> str | None:
        if value is not None and not value:
            raise ValueError("gold_conclusion_cannot_be_empty")
        return value


class ClosedCorpusPrediction(ContractModel):
    """One frozen output for the ordinary system or a named oracle intervention."""

    closed_corpus_prediction_version: Literal["1"] = "1"
    question_id: str = Field(min_length=1)
    arm: AblationArm
    retrieval_source: EvidenceSource
    extraction_source: EvidenceSource
    retrieved_paper_ids: list[str] | None = None
    extracted_paper_ids: list[str] | None = None
    predicted_conclusion: str | None = None
    abstained: bool = False

    @field_validator("retrieved_paper_ids", "extracted_paper_ids")
    @classmethod
    def validate_optional_paper_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if any(not item for item in value) or value != sorted(set(value)):
            raise ValueError("prediction_paper_ids_must_be_sorted_unique")
        return value

    @field_validator("predicted_conclusion")
    @classmethod
    def validate_prediction(cls, value: str | None) -> str | None:
        if value is not None and not value:
            raise ValueError("predicted_conclusion_cannot_be_empty")
        return value

    @model_validator(mode="after")
    def validate_sources_and_output(self) -> ClosedCorpusPrediction:
        if (self.retrieved_paper_ids is None) != (self.retrieval_source == "not_run"):
            raise ValueError("retrieval_source_output_mismatch")
        if (self.extracted_paper_ids is None) != (self.extraction_source == "not_run"):
            raise ValueError("extraction_source_output_mismatch")
        if self.extracted_paper_ids is not None:
            if self.retrieved_paper_ids is None:
                raise ValueError("extraction_requires_retrieval_output")
            if not set(self.extracted_paper_ids) <= set(self.retrieved_paper_ids):
                raise ValueError("extracted_papers_must_be_subset_of_retrieved_papers")
        if self.abstained and self.predicted_conclusion is not None:
            raise ValueError("abstention_cannot_have_predicted_conclusion")

        if self.arm == "system":
            if self.retrieval_source not in {"system", "not_run"}:
                raise ValueError("system_arm_cannot_use_oracle_retrieval")
            if self.extraction_source not in {"system", "not_run"}:
                raise ValueError("system_arm_cannot_use_oracle_extraction")
        elif self.arm == "oracle_corpus":
            if self.retrieval_source != "evaluator_oracle":
                raise ValueError("oracle_corpus_requires_oracle_retrieval")
            if self.extraction_source not in {"system", "not_run"}:
                raise ValueError("oracle_corpus_requires_system_extraction")
        else:
            if self.retrieval_source != "evaluator_oracle":
                raise ValueError("oracle_extraction_requires_oracle_retrieval")
            if self.extraction_source != "evaluator_oracle":
                raise ValueError("oracle_extraction_requires_oracle_extraction")
        return self


def build_oracle_corpus_prediction(
    *,
    gold: ClosedCorpusGoldQuestion,
    extracted_paper_ids: Sequence[str] | None,
    predicted_conclusion: str | None,
    abstained: bool = False,
) -> ClosedCorpusPrediction:
    """Build an oracle-corpus ablation while leaving extraction/synthesis to the system."""

    extracted = None if extracted_paper_ids is None else sorted(set(extracted_paper_ids))
    return ClosedCorpusPrediction(
        question_id=gold.question_id,
        arm="oracle_corpus",
        retrieval_source="evaluator_oracle",
        extraction_source="not_run" if extracted is None else "system",
        retrieved_paper_ids=gold.gold_paper_ids,
        extracted_paper_ids=extracted,
        predicted_conclusion=predicted_conclusion,
        abstained=abstained,
    )


def build_oracle_extraction_prediction(
    *,
    gold: ClosedCorpusGoldQuestion,
    predicted_conclusion: str | None,
    abstained: bool = False,
) -> ClosedCorpusPrediction:
    """Build an oracle-retrieval + oracle-extraction arm for system synthesis.

    This helper never copies ``gold_conclusion`` into the prediction.  The caller must
    run the frozen synthesis procedure on evaluator-provided extractions and supply its
    actual conclusion or abstention.
    """

    return ClosedCorpusPrediction(
        question_id=gold.question_id,
        arm="oracle_extraction",
        retrieval_source="evaluator_oracle",
        extraction_source="evaluator_oracle",
        retrieved_paper_ids=gold.gold_paper_ids,
        extracted_paper_ids=gold.gold_paper_ids,
        predicted_conclusion=predicted_conclusion,
        abstained=abstained,
    )


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _validate_evaluation_inputs(
    gold: Sequence[ClosedCorpusGoldQuestion],
    predictions: Sequence[ClosedCorpusPrediction],
) -> tuple[
    dict[str, ClosedCorpusGoldQuestion],
    dict[tuple[str, AblationArm], ClosedCorpusPrediction],
]:
    gold_by_id: dict[str, ClosedCorpusGoldQuestion] = {}
    for question in gold:
        if question.question_id in gold_by_id:
            raise ClosedCorpusContractError(f"duplicate_gold_question:{question.question_id}")
        gold_by_id[question.question_id] = question
    if not gold_by_id:
        raise ClosedCorpusContractError("gold_questions_empty")

    prediction_by_key: dict[tuple[str, AblationArm], ClosedCorpusPrediction] = {}
    for prediction in predictions:
        if prediction.question_id not in gold_by_id:
            raise ClosedCorpusContractError(
                f"prediction_question_outside_gold:{prediction.question_id}"
            )
        key = (prediction.question_id, prediction.arm)
        if key in prediction_by_key:
            raise ClosedCorpusContractError(
                f"duplicate_prediction:{prediction.question_id}:{prediction.arm}"
            )
        question = gold_by_id[prediction.question_id]
        if (
            prediction.arm in {"oracle_corpus", "oracle_extraction"}
            and prediction.retrieved_paper_ids != question.gold_paper_ids
        ):
            raise ClosedCorpusContractError(
                f"oracle_retrieval_not_exact_gold:{prediction.question_id}:{prediction.arm}"
            )
        if (
            prediction.arm == "oracle_extraction"
            and prediction.extracted_paper_ids != question.gold_paper_ids
        ):
            raise ClosedCorpusContractError(
                f"oracle_extraction_not_exact_gold:{prediction.question_id}"
            )
        prediction_by_key[key] = prediction
    return gold_by_id, prediction_by_key


def _evaluate_arm(
    *,
    arm: AblationArm,
    gold: Sequence[ClosedCorpusGoldQuestion],
    prediction_by_key: Mapping[tuple[str, AblationArm], ClosedCorpusPrediction],
) -> dict[str, Any]:
    predictions_present = sum((question.question_id, arm) in prediction_by_key for question in gold)
    if predictions_present == 0:
        return {
            "status": "not_run",
            "questions": len(gold),
            "predictions_present": 0,
        }

    retrieval_missing = retrieval_empty = retrieval_supplied = exact_retrieval = 0
    extraction_missing = extraction_supplied = 0
    total_gold = total_retrieved = retrieval_hits = extraction_hits = 0
    retrieved_gold_denominator = 0
    macro_retrieval_recall: list[float] = []
    macro_extraction_recall: list[float] = []
    macro_retrieval_precision: list[float] = []
    macro_conditional_extraction: list[float] = []
    gold_papers_with_zero_findings = 0
    answered = correct = abstained = missing_conclusion = labeled = 0

    for question in gold:
        prediction = prediction_by_key.get((question.question_id, arm))
        gold_ids = set(question.gold_paper_ids)
        total_gold += len(gold_ids)

        retrieved = None if prediction is None else prediction.retrieved_paper_ids
        if retrieved is None:
            retrieval_missing += 1
            retrieved_ids: set[str] = set()
        else:
            retrieval_supplied += 1
            retrieved_ids = set(retrieved)
            retrieval_empty += int(not retrieved_ids)
            exact_retrieval += int(retrieved_ids == gold_ids)
        hits = len(gold_ids & retrieved_ids)
        retrieval_hits += hits
        total_retrieved += len(retrieved_ids)
        retrieved_gold_denominator += hits
        macro_retrieval_recall.append(hits / len(gold_ids))
        if retrieved_ids:
            macro_retrieval_precision.append(hits / len(retrieved_ids))

        extracted = None if prediction is None else prediction.extracted_paper_ids
        if extracted is None:
            extraction_missing += 1
            extracted_ids: set[str] = set()
        else:
            extraction_supplied += 1
            extracted_ids = set(extracted)
        extracted_gold = len(gold_ids & extracted_ids)
        extraction_hits += extracted_gold
        macro_extraction_recall.append(extracted_gold / len(gold_ids))
        if hits:
            macro_conditional_extraction.append(extracted_gold / hits)
        gold_papers_with_zero_findings += len((gold_ids & retrieved_ids) - extracted_ids)

        if question.gold_conclusion is None:
            continue
        labeled += 1
        if prediction is None or (
            prediction.predicted_conclusion is None and not prediction.abstained
        ):
            missing_conclusion += 1
        elif prediction.abstained:
            abstained += 1
        else:
            answered += 1
            correct += int(prediction.predicted_conclusion == question.gold_conclusion)

    return {
        "status": "complete" if predictions_present == len(gold) else "partial",
        "questions": len(gold),
        "predictions_present": predictions_present,
        "retrieval": {
            "supplied": retrieval_supplied,
            "missing": retrieval_missing,
            "explicit_empty": retrieval_empty,
            "exact_set_match": exact_retrieval,
            "gold_papers": total_gold,
            "retrieved_papers": total_retrieved,
            "hits": retrieval_hits,
            "macro_recall_missing_as_zero": sum(macro_retrieval_recall)
            / len(macro_retrieval_recall),
            "micro_recall_missing_as_zero": _safe_ratio(retrieval_hits, total_gold),
            "micro_precision_on_retrieved": _safe_ratio(retrieval_hits, total_retrieved),
            "macro_precision_on_nonempty": (
                sum(macro_retrieval_precision) / len(macro_retrieval_precision)
                if macro_retrieval_precision
                else None
            ),
        },
        "extraction": {
            "supplied": extraction_supplied,
            "missing": extraction_missing,
            "gold_papers_with_one_or_more_findings": extraction_hits,
            "gold_papers_with_zero_extracted_findings_after_retrieval": (
                gold_papers_with_zero_findings
            ),
            "end_to_end_macro_paper_recall": sum(macro_extraction_recall)
            / len(macro_extraction_recall),
            "end_to_end_micro_paper_recall": _safe_ratio(extraction_hits, total_gold),
            "conditional_micro_yield_on_retrieved_gold": _safe_ratio(
                extraction_hits, retrieved_gold_denominator
            ),
            "conditional_macro_yield_on_retrieved_gold": (
                sum(macro_conditional_extraction) / len(macro_conditional_extraction)
                if macro_conditional_extraction
                else None
            ),
            "denominator_note": (
                "Every gold included paper remains in end-to-end extraction recall; a "
                "retrieved eligible paper with zero accepted findings is a miss, not an "
                "omitted denominator."
            ),
        },
        "conclusion": {
            "gold_labeled_questions": labeled,
            "answered": answered,
            "abstained": abstained,
            "missing": missing_conclusion,
            "correct": correct,
            "coverage": _safe_ratio(answered, labeled),
            "selective_accuracy": _safe_ratio(correct, answered),
            "strict_accuracy_with_unanswered_as_errors": _safe_ratio(correct, labeled),
        },
    }


def evaluate_closed_corpus(
    *,
    gold: Sequence[ClosedCorpusGoldQuestion],
    predictions: Sequence[ClosedCorpusPrediction],
) -> dict[str, Any]:
    """Evaluate system and oracle arms without exposing per-question evaluator data."""

    gold_by_id, prediction_by_key = _validate_evaluation_inputs(gold, predictions)
    ordered_gold = [gold_by_id[key] for key in sorted(gold_by_id)]
    return {
        "closed_corpus_evaluation_version": "1",
        "questions": len(ordered_gold),
        "gold_sha256": hash_canonical(ordered_gold),
        "predictions_sha256": hash_canonical(list(predictions)),
        "contains_question_text": False,
        "contains_article_text": False,
        "contains_paper_identifiers": False,
        "arms": {
            arm: _evaluate_arm(
                arm=arm,
                gold=ordered_gold,
                prediction_by_key=prediction_by_key,
            )
            for arm in ABLATION_ARMS
        },
        "oracle_interpretation": (
            "Oracle arms are evaluator interventions. oracle_corpus fixes retrieval only; "
            "oracle_extraction fixes retrieval and paper-level extraction coverage while "
            "leaving final synthesis to the frozen system."
        ),
    }


class ReviewerFinding(ContractModel):
    """Allowlisted model output shown to a reviewer; no confidence-like fields exist."""

    audit_finding_id: str
    intervention: str | None
    comparator: str | None
    outcome_name: str
    timepoint_raw: str | None
    effect_direction: str
    effect_size_raw: str | None
    p_value: float | None
    significant: bool | None
    evidence_quote: str | None
    evidence_lines: list[str] | None


class BlindedPaperAuditItem(ContractModel):
    """One confidence-blinded paper-level review item."""

    audit_packet_item_version: Literal["1"] = "1"
    audit_unit_id: str
    display_order: int = Field(ge=1)
    research_question: str
    eligibility_criteria: list[str]
    source_lines: dict[str, dict[str, str]]
    system_output: dict[Literal["eligible", "findings"], bool | list[ReviewerFinding]]
    review_form: dict[str, Any]

    @model_validator(mode="after")
    def validate_system_output(self) -> BlindedPaperAuditItem:
        if set(self.system_output) != {"eligible", "findings"}:
            raise ValueError("reviewer_system_output_fields_changed")
        return self


def _forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            rendered = str(key).casefold()
            if rendered in FORBIDDEN_REVIEWER_KEYS:
                found.add(rendered)
            found.update(_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_forbidden_keys(item))
    return found


def _eligible_for_paper_audit(paper: Mapping[str, Any]) -> bool:
    return paper.get("screen_status") == "included" and paper.get("map_status") == "success"


def _paper_id(paper: Mapping[str, Any]) -> str:
    value = paper.get("paper_id")
    if not isinstance(value, str) or not value:
        raise ClosedCorpusContractError("paper_ledger_invalid_paper_id")
    return value


def build_paper_audit_candidates(
    papers: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build a paper census in which zero-finding outputs remain explicit candidates."""

    paper_by_id: dict[str, Mapping[str, Any]] = {}
    for paper in papers:
        paper_id = _paper_id(paper)
        if paper_id in paper_by_id:
            raise ClosedCorpusContractError(f"duplicate_paper_id:{paper_id}")
        paper_by_id[paper_id] = paper

    findings_by_paper: dict[str, list[dict[str, Any]]] = {}
    finding_ids: set[str] = set()
    for finding in findings:
        finding_id = finding.get("finding_id")
        paper_id = finding.get("paper_id")
        if not isinstance(finding_id, str) or not finding_id:
            raise ClosedCorpusContractError("finding_ledger_invalid_finding_id")
        if finding_id in finding_ids:
            raise ClosedCorpusContractError(f"duplicate_finding_id:{finding_id}")
        finding_ids.add(finding_id)
        if paper_id not in paper_by_id:
            raise ClosedCorpusContractError(f"orphan_finding:{finding_id}")
        findings_by_paper.setdefault(str(paper_id), []).append(dict(finding))

    candidates: list[dict[str, Any]] = []
    for paper_id, paper in sorted(paper_by_id.items()):
        if not _eligible_for_paper_audit(paper):
            continue
        paper_findings = sorted(
            findings_by_paper.get(paper_id, []),
            key=lambda row: str(row["finding_id"]),
        )
        declared_count = paper.get("accepted_finding_count")
        if not isinstance(declared_count, int) or declared_count != len(paper_findings):
            raise ClosedCorpusContractError(
                f"accepted_finding_count_mismatch:{paper_id}:"
                f"declared={declared_count}:ledger={len(paper_findings)}"
            )
        pipeline_eligible = paper.get("eligible")
        if not isinstance(pipeline_eligible, bool):
            raise ClosedCorpusContractError(f"mapped_paper_eligibility_missing:{paper_id}")
        if pipeline_eligible and not paper_findings:
            stratum = "pipeline_eligible_zero_findings"
        elif pipeline_eligible:
            stratum = "pipeline_eligible_with_findings"
        else:
            stratum = "pipeline_ineligible_zero_findings"
        candidates.append(
            {
                "paper_id": paper_id,
                "doc_id": paper.get("doc_id"),
                "pipeline_eligible": pipeline_eligible,
                "findings": paper_findings,
                "selection_stratum": stratum,
            }
        )
    if not candidates:
        raise ClosedCorpusContractError("paper_audit_candidate_pool_empty")
    return candidates


def select_paper_audit_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    sample_size: int = 60,
    seed: int = 20260827,
) -> list[dict[str, Any]]:
    """Census pipeline-eligible papers (including zeros), then random-fill negatives."""

    if sample_size < 1:
        raise ClosedCorpusContractError("audit_sample_size_must_be_positive")
    rows = [dict(row) for row in candidates]
    if len(rows) < sample_size:
        raise ClosedCorpusContractError(
            f"audit_candidate_pool_too_small:required={sample_size}:found={len(rows)}"
        )
    ids = [str(row.get("paper_id", "")) for row in rows]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ClosedCorpusContractError("audit_candidate_paper_ids_invalid")

    mandatory = [row for row in rows if row.get("pipeline_eligible") is True]
    if len(mandatory) > sample_size:
        raise ClosedCorpusContractError(
            f"pipeline_eligible_papers_exceed_sample:required={len(mandatory)}:sample={sample_size}"
        )
    mandatory_ids = {str(row["paper_id"]) for row in mandatory}
    remaining = [row for row in rows if str(row["paper_id"]) not in mandatory_ids]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    selected = [*mandatory, *remaining[: sample_size - len(mandatory)]]
    rng.shuffle(selected)
    return selected


def _reviewer_finding(
    finding: Mapping[str, Any], *, audit_unit_id: str, index: int
) -> ReviewerFinding:
    return ReviewerFinding(
        audit_finding_id=f"{audit_unit_id}-finding-{index:02d}",
        intervention=finding.get("intervention"),
        comparator=finding.get("comparator"),
        outcome_name=str(finding.get("outcome_name") or ""),
        timepoint_raw=finding.get("timepoint_raw"),
        effect_direction=str(finding.get("effect_direction") or ""),
        effect_size_raw=finding.get("effect_size_raw"),
        p_value=finding.get("p_value"),
        significant=finding.get("significant"),
        evidence_quote=finding.get("evidence_quote"),
        evidence_lines=finding.get("evidence_lines"),
    )


def prepare_blinded_human_review_packet(
    *,
    question_id: str,
    research_question: str,
    eligibility_criteria: Sequence[str],
    papers: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    source_lines_by_doc_id: Mapping[str, Mapping[str, Mapping[str, Any]]],
    output_dir: Path,
    sample_size: int = 60,
    seed: int = 20260827,
    force: bool = False,
) -> dict[str, Any]:
    """Freeze a randomized packet, two blank forms, and a private identity key.

    The reviewer packet may contain article text and therefore belongs in ignored local
    storage.  The returned manifest contains only counts and hashes and is safe to
    summarize publicly.
    """

    criteria = [item.strip() for item in eligibility_criteria if item.strip()]
    if not question_id or not research_question or not criteria:
        raise ClosedCorpusContractError("audit_question_and_criteria_required")
    candidates = build_paper_audit_candidates(papers, findings)
    selected = select_paper_audit_candidates(
        candidates,
        sample_size=sample_size,
        seed=seed,
    )

    packet_rows: list[BlindedPaperAuditItem] = []
    key_rows: list[dict[str, Any]] = []
    forms: dict[str, list[dict[str, Any]]] = {"reviewer_a": [], "reviewer_b": []}
    selected_strata: Counter[str] = Counter()
    for order, candidate in enumerate(selected, start=1):
        paper_id = str(candidate["paper_id"])
        doc_id = candidate.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise ClosedCorpusContractError(f"audit_candidate_doc_id_missing:{paper_id}")
        raw_lines = source_lines_by_doc_id.get(doc_id)
        if not isinstance(raw_lines, Mapping) or not raw_lines:
            raise ClosedCorpusContractError(f"audit_source_lines_missing:{paper_id}:{doc_id}")
        source_lines: dict[str, dict[str, str]] = {}
        for line_id, raw in sorted(raw_lines.items()):
            if not isinstance(raw, Mapping):
                raise ClosedCorpusContractError(f"audit_source_line_invalid:{paper_id}:{line_id}")
            section = raw.get("section")
            text = raw.get("text")
            if not isinstance(section, str) or not isinstance(text, str):
                raise ClosedCorpusContractError(f"audit_source_line_invalid:{paper_id}:{line_id}")
            source_lines[str(line_id)] = {"section": section, "text": text}

        audit_unit_id = f"audit-{hash_canonical({'seed': seed, 'paper_id': paper_id})[:20]}"
        reviewer_findings = [
            _reviewer_finding(row, audit_unit_id=audit_unit_id, index=index)
            for index, row in enumerate(candidate["findings"], start=1)
        ]
        review_form = {
            "paper_eligible": None,
            "any_target_finding_missed": None,
            "all_emitted_findings_supported": None,
            "paper_direction_summary": None,
            "finding_decisions": [
                {
                    "audit_finding_id": finding.audit_finding_id,
                    "atomic": None,
                    "supported_by_quote": None,
                    "direction_correct": None,
                    "pico_correct": None,
                    "notes": None,
                }
                for finding in reviewer_findings
            ],
            "error_codes": [],
            "notes": None,
            # Measured after the decision is complete.  This is a realized review
            # duration, not a model-side cost estimate or scheduling signal.
            "review_minutes": None,
        }
        item = BlindedPaperAuditItem(
            audit_unit_id=audit_unit_id,
            display_order=order,
            research_question=research_question,
            eligibility_criteria=criteria,
            source_lines=source_lines,
            system_output={
                "eligible": bool(candidate["pipeline_eligible"]),
                "findings": reviewer_findings,
            },
            review_form=review_form,
        )
        forbidden = _forbidden_keys(item.model_dump(mode="json"))
        if forbidden:
            raise ClosedCorpusContractError(f"forbidden_reviewer_packet_keys:{sorted(forbidden)}")
        packet_rows.append(item)
        stratum = str(candidate["selection_stratum"])
        selected_strata[stratum] += 1
        key_rows.append(
            {
                "audit_unit_id": audit_unit_id,
                "paper_id": paper_id,
                "doc_id": doc_id,
                "finding_ids": [str(row["finding_id"]) for row in candidate["findings"]],
                "selection_stratum": stratum,
            }
        )
        for reviewer in forms:
            forms[reviewer].append(
                {
                    "human_review_decision_version": "2",
                    "audit_unit_id": audit_unit_id,
                    "reviewer_slot": reviewer,
                    **review_form,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = output_dir / "review_packet.private.jsonl"
    key_path = output_dir / "identity_key.private.jsonl"
    reviewer_a_path = output_dir / "reviewer_a_decisions.private.jsonl"
    reviewer_b_path = output_dir / "reviewer_b_decisions.private.jsonl"
    manifest_path = output_dir / "manifest.json"
    for path in (packet_path, key_path, reviewer_a_path, reviewer_b_path, manifest_path):
        if path.exists() and not force:
            raise ClosedCorpusContractError(f"audit_packet_output_exists:{path}")

    atomic_write_jsonl(packet_path, packet_rows, force=force)
    atomic_write_jsonl(key_path, key_rows, force=force)
    atomic_write_jsonl(reviewer_a_path, forms["reviewer_a"], force=force)
    atomic_write_jsonl(reviewer_b_path, forms["reviewer_b"], force=force)
    all_strata = Counter(str(row["selection_stratum"]) for row in candidates)
    manifest = {
        "human_review_packet_manifest_version": "2",
        "question_id": question_id,
        "seed": seed,
        "sample_size": sample_size,
        "candidate_papers": len(candidates),
        "candidate_strata": dict(sorted(all_strata.items())),
        "selected_strata": dict(sorted(selected_strata.items())),
        "eligible_zero_finding_papers_required": all_strata["pipeline_eligible_zero_findings"],
        "eligible_zero_finding_papers_selected": selected_strata["pipeline_eligible_zero_findings"],
        "all_eligible_zero_finding_papers_included": (
            all_strata["pipeline_eligible_zero_findings"]
            == selected_strata["pipeline_eligible_zero_findings"]
        ),
        "pipeline_eligible_papers_required": (
            all_strata["pipeline_eligible_zero_findings"]
            + all_strata["pipeline_eligible_with_findings"]
        ),
        "pipeline_eligible_papers_selected": (
            selected_strata["pipeline_eligible_zero_findings"]
            + selected_strata["pipeline_eligible_with_findings"]
        ),
        "all_pipeline_eligible_papers_included": (
            all_strata["pipeline_eligible_zero_findings"]
            + all_strata["pipeline_eligible_with_findings"]
            == selected_strata["pipeline_eligible_zero_findings"]
            + selected_strata["pipeline_eligible_with_findings"]
        ),
        "reviewers": 2,
        "reviewer_blinding": {
            "model_confidence_hidden": True,
            "risk_and_ranking_signals_hidden": True,
            "selection_stratum_hidden": True,
            "system_output_visible_for_adjudication": True,
            "article_identity_may_be_visible_in_source_text": True,
        },
        "manifest_contains_article_text": False,
        "review_packet_contains_article_text": True,
        "contains_model_confidence": False,
        "local_private_files": {
            "review_packet": {
                "path": packet_path.name,
                "rows": len(packet_rows),
                "sha256": sha256_file(packet_path),
            },
            "identity_key": {
                "path": key_path.name,
                "rows": len(key_rows),
                "sha256": sha256_file(key_path),
            },
            "reviewer_a_decisions": {
                "path": reviewer_a_path.name,
                "rows": len(forms["reviewer_a"]),
                "sha256": sha256_file(reviewer_a_path),
            },
            "reviewer_b_decisions": {
                "path": reviewer_b_path.name,
                "rows": len(forms["reviewer_b"]),
                "sha256": sha256_file(reviewer_b_path),
            },
        },
    }
    atomic_write_json(manifest_path, manifest, force=force)
    return manifest


__all__ = [
    "ABLATION_ARMS",
    "FORBIDDEN_REVIEWER_KEYS",
    "BlindedPaperAuditItem",
    "ClosedCorpusContractError",
    "ClosedCorpusGoldQuestion",
    "ClosedCorpusPrediction",
    "ReviewerFinding",
    "build_oracle_corpus_prediction",
    "build_oracle_extraction_prediction",
    "build_paper_audit_candidates",
    "evaluate_closed_corpus",
    "prepare_blinded_human_review_packet",
    "select_paper_audit_candidates",
]
