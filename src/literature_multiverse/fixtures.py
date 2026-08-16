"""Deterministic, provider-free fixtures for the complete scientific pipeline.

The generator deliberately stops at inputs.  It creates an s3-compatible terminal
paper/finding boundary, authoritative source lines, frozen audit and verifier inputs,
and the comparison baseline.  G3, s5, release-selection, and demo artifacts must be
derived by their production implementations.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from literature_multiverse.analysis import derive_primary_cohort
from literature_multiverse.audit import DEFAULT_AUDIT_FIELDS, select_audit_sample
from literature_multiverse.baseline import create_baseline
from literature_multiverse.cohort import cohort_sha256
from literature_multiverse.config import (
    FIXTURE_QUESTION_IDS,
    QuestionConfig,
    config_sha256,
    load_config_for_question,
)
from literature_multiverse.extract import (
    ExtractionLedgers,
    MapEnvelope,
    assemble_extraction_ledgers,
    extraction_prompt_replacements,
)
from literature_multiverse.lineage import (
    artifact_ref,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    extraction_cfghash,
    hash_canonical,
    sha256_file,
    write_run_record,
)
from literature_multiverse.models import RunRecord, derive_paper_id
from literature_multiverse.normalize import normalized_frames
from literature_multiverse.paths import PATHS, ProjectPaths
from literature_multiverse.prompting import render_prompt_file
from literature_multiverse.schemas import generate_extraction_schema
from literature_multiverse.verification import fixture_verification

FixtureFaultMode = Literal["after-25-bootstrap"]

FIXTURE_ORDER: tuple[str, ...] = (
    "fixture-a",
    "fixture-b-story",
    "fixture-b-m4",
    "fixture-b-incomplete",
)
FIXTURE_VERSION = "1"
FIXTURE_PROMPT_VERSION = "extraction-v1"
FIXTURE_FAULT_MODE: FixtureFaultMode = "after-25-bootstrap"
FIXTURE_CREATED_AT = datetime(2026, 8, 15, 19, 0, tzinfo=UTC)
ELIGIBLE_FINDING_PAPERS = 22
HEAVY_PAPER_INDEX = 21


class FixtureContractError(ValueError):
    """A fixture request or generated corpus violated the frozen contract."""


@dataclass(frozen=True, slots=True)
class GeneratedFixture:
    """Paths and reconciled counts returned after one complete generation."""

    question_id: str
    papers_path: Path
    findings_path: Path
    quarantine_path: Path
    run_record_path: Path
    authoritative_lines_path: Path
    map_envelopes_path: Path
    audit_sample_path: Path
    audit_decisions_path: Path
    verification_path: Path
    baseline_path: Path
    fault_injection_path: Path | None
    counts: Mapping[str, int]


def _paper_doc_id(question_id: str, index: int) -> str:
    return f"{question_id}-anchor" if index == 0 else f"{question_id}-study-{index + 1:02d}"


def _directions(question_id: str, index: int) -> tuple[str, str]:
    if index == 0:
        return {
            "fixture-a": ("increase", "no_effect"),
            "fixture-b-story": ("no_effect", "no_effect"),
            "fixture-b-m4": ("increase", "decrease"),
            "fixture-b-incomplete": ("increase", "decrease"),
        }[question_id]
    if question_id == "fixture-b-story":
        return ("no_effect", "no_effect")
    if question_id == "fixture-a":
        if index <= 7:
            direction = "increase"
        elif index <= 10:
            direction = "no_effect"
        elif index <= 18:
            direction = "decrease"
        else:
            direction = "no_effect"
        return (direction, direction)
    direction = ("increase", "no_effect", "decrease")[(index - 1) % 3]
    return (direction, direction)


def _dose_regime(question_id: str, index: int) -> str | None:
    # A true missing tested-moderator value is planted without losing registered coverage.
    if index == HEAVY_PAPER_INDEX:
        return None
    if question_id == "fixture-a":
        return "high" if index <= 10 else "low"
    return "high" if index % 2 else "low"


def _training_status(index: int) -> str | None:
    # Deliberately independent of the planted A dose signal and the B direction cycle.
    if index == HEAVY_PAPER_INDEX - 1:
        return None
    return "trained" if index % 2 else "untrained"


def _quote(direction: str, *, endpoint: str, timepoint: str) -> str:
    predicate = {
        "increase": "was higher than control",
        "no_effect": "did not differ from control",
        "decrease": "was lower than control",
        "mixed": "increased in one comparison and decreased in another",
        "unclear": "had an unclear direction relative to control",
    }[direction]
    return f"At {timepoint}, {endpoint} {predicate} after the synthetic intervention."


def _raw_finding(
    *,
    question_id: str,
    paper_index: int,
    array_position: int,
    endpoint: str,
    direction: str,
    dose_regime: str | None,
    training_status: str | None,
    line_number: int,
    section: str = "Results",
    grounding_pathology: str | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    timepoint = f"week {array_position + 1}"
    timing = "acute" if array_position % 2 == 0 else "chronic"
    dose_raw = (
        "100 mg + 50 mg cofactor"
        if paper_index == 4 and array_position == 0
        else f"{dose_regime or 'unreported'} synthetic dose"
    )
    quote = _quote(direction, endpoint=endpoint, timepoint=timepoint)
    authoritative_quote = quote
    evidence_quote: str | None = quote
    evidence_lines: list[str] | None = [f"L{line_number}"]
    if grounding_pathology == "mismatch":
        evidence_quote = "This sentence is intentionally absent from the authoritative line."
    elif grounding_pathology == "missing":
        evidence_quote = None
        evidence_lines = None
    row = {
        "study_type": "controlled primary research",
        "species": "human",
        "model": None,
        "population_state": None if paper_index == 19 else (
            "healthy" if paper_index % 2 == 0 else "clinical"
        ),
        "intervention": "synthetic intervention",
        "intervention_class": "synthetic",
        "comparator": "control condition",
        "dose_raw": dose_raw,
        "duration_raw": "4 weeks",
        "timing_context": timing,
        "outcome_name": endpoint,
        "outcome_family": "performance",
        "timepoint_raw": timepoint,
        "effect_direction": direction,
        "effect_size_raw": None,
        "p_value": 0.01 if direction in {"increase", "decrease"} else 0.62,
        "significant": direction in {"increase", "decrease"},
        "sample_size": 24 + paper_index,
        "evidence_quote": evidence_quote,
        "evidence_lines": evidence_lines,
        "confidence": 0.96,
        "moderators": {
            "dose_regime": dose_regime,
            "training_status": training_status,
            "population_state": None if paper_index == 19 else (
                "healthy" if paper_index % 2 == 0 else "clinical"
            ),
            "timing_context": timing,
        },
    }
    line = {
        "line_id": f"L{line_number}",
        "text": authoritative_quote,
        "section": section,
    }
    return row, line


def _paper_findings(
    question_id: str, paper_index: int
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    directions = _directions(question_id, paper_index)
    dose = _dose_regime(question_id, paper_index)
    training = _training_status(paper_index)
    rows: list[dict[str, Any]] = []
    lines: list[dict[str, str]] = []

    count = 15 if paper_index == HEAVY_PAPER_INDEX else 2
    for position in range(count):
        endpoint = "peak_power" if position % 2 == 0 else "fatigue_time"
        direction = directions[position] if position < 2 else directions[0]
        section = "Results"
        pathology: str | None = None
        if paper_index == HEAVY_PAPER_INDEX:
            if position == 10:
                direction = "mixed"
            elif position == 11:
                direction = "unclear"
            elif position == 12:
                pathology = "mismatch"
            elif position == 13:
                pathology = "missing"
            elif position == 14:
                section = "References"
        row, line = _raw_finding(
            question_id=question_id,
            paper_index=paper_index,
            array_position=position,
            endpoint=endpoint,
            direction=direction,
            dose_regime=dose,
            training_status=training,
            line_number=position + 1,
            section=section,
            grounding_pathology=pathology,
        )
        rows.append(row)
        lines.append(line)

    if paper_index == HEAVY_PAPER_INDEX:
        invalid_direction = dict(rows[0])
        invalid_direction["moderators"] = dict(rows[0]["moderators"])
        invalid_direction["effect_direction"] = "sideways"
        invalid_direction["timepoint_raw"] = "quarantine direction"
        extra_field = dict(rows[1])
        extra_field["moderators"] = dict(rows[1]["moderators"])
        extra_field["timepoint_raw"] = "quarantine extra field"
        extra_field["unregistered_claim"] = "must be quarantined"
        rows.extend([invalid_direction, extra_field])
    return rows, lines


def _screened_paper(
    *,
    question_id: str,
    config_hash: str,
    index: int,
    screen_status: str = "included",
) -> dict[str, Any]:
    doc_id = _paper_doc_id(question_id, index)
    is_published_twin = index == 5
    doi = f"10.5555/{question_id}.published" if is_published_twin else None
    paper_id = derive_paper_id(doc_id=doc_id, doi=doi)
    return {
        "paper_id": paper_id,
        "doc_id": doc_id,
        "alternate_doc_ids": [f"{question_id}-study-06-preprint"] if is_published_twin else [],
        "doi": doi,
        "pmid": None,
        "title": f"Synthetic controlled performance study {index + 1:02d}",
        "first_author": f"FixtureAuthor{index + 1:02d}",
        "pub_year": 2026 - (index % 6),
        "source": "fixture",
        "article_type": "review" if screen_status == "excluded" else "research-article",
        "query_families": ["direct", "null-negative"],
        "search_result_ids": [f"fixture-search-{question_id}-{index + 1:02d}"],
        "content_tier": "full_text" if index % 3 else "abstract_only",
        "publication_status": "peer_reviewed",
        "screen_status": screen_status,
        "screen_reason": "article_type_excluded:review" if screen_status == "excluded" else None,
        "dedupe_cluster_id": f"fixture-cluster-{question_id}-{index + 1:02d}",
        "dedupe_preferred": True,
        "map_status": "not_mapped",
        "eligible": None,
        "exclusion_reason": None,
        "map_result_id": None,
        "raw_artifact_path": None,
        "raw_finding_count": 0,
        "accepted_finding_count": 0,
        "quarantined_finding_count": 0,
        "failure_code": None,
        "dataset_or_cohort_id": "fixture-shared-cohort" if index in {3, 4} else None,
        "prompt_version": None,
        "schema_version": "1",
        "config_sha256": config_hash,
        "cfghash": None,
        "created_at": FIXTURE_CREATED_AT,
    }


def _fixture_sources(
    config: QuestionConfig,
) -> tuple[list[dict[str, Any]], list[MapEnvelope], dict[str, list[dict[str, str]]]]:
    qid = config.question_id
    cfg_hash = config_sha256(config)
    screened = [
        _screened_paper(question_id=qid, config_hash=cfg_hash, index=index)
        for index in range(26)
    ]
    screened[-1] = _screened_paper(
        question_id=qid,
        config_hash=cfg_hash,
        index=25,
        screen_status="excluded",
    )

    map_result_id = f"fixture-map-{qid}-v1"
    envelopes: list[MapEnvelope] = []
    source_lines: dict[str, list[dict[str, str]]] = {}
    for index, paper in enumerate(screened[:-1]):
        doc_id = str(paper["doc_id"])
        if index < ELIGIBLE_FINDING_PAPERS:
            findings, lines = _paper_findings(qid, index)
            payload: Mapping[str, Any] | None = {
                "eligible": True,
                "exclusion_reason": None,
                "findings": findings,
            }
            status = "success"
            provider_message = None
            source_lines[doc_id] = lines
        elif index == 22:
            payload = {"eligible": True, "exclusion_reason": None, "findings": []}
            status = "success"
            provider_message = None
            source_lines[doc_id] = [
                {"line_id": "L1", "text": "No locked endpoint was reported.", "section": "Results"}
            ]
        elif index == 23:
            payload = {
                "eligible": False,
                "exclusion_reason": "comparator not reported",
                "findings": [],
            }
            status = "success"
            provider_message = None
            source_lines[doc_id] = [
                {
                    "line_id": "L1",
                    "text": "This report has no control condition.",
                    "section": "Methods",
                }
            ]
        else:
            payload = None
            status = "failed"
            provider_message = "fixture terminal map failure"
            source_lines[doc_id] = [
                {"line_id": "L1", "text": "The source remained archived.", "section": "Unknown"}
            ]
        envelopes.append(
            MapEnvelope(
                map_result_id=map_result_id,
                position=index,
                status=status,
                title=str(paper["title"]),
                doc_id=doc_id,
                payload=payload,
                provider_message=provider_message,
            )
        )
    return screened, envelopes, source_lines


def _envelope_payload(question_id: str, envelopes: Sequence[MapEnvelope]) -> dict[str, Any]:
    return {
        "fixture_map_version": FIXTURE_VERSION,
        "question_id": question_id,
        "map_result_id": f"fixture-map-{question_id}-v1",
        "envelopes": [
            {
                "map_result_id": envelope.map_result_id,
                "position": envelope.position,
                "status": envelope.status,
                "title": envelope.title,
                "doc_id": envelope.doc_id,
                "payload": envelope.payload,
                "provider_message": envelope.provider_message,
            }
            for envelope in envelopes
        ],
    }


def _audit_candidates(ledgers: ExtractionLedgers, *, primary_family: str) -> list[dict[str, Any]]:
    eligible = {
        paper.paper_id
        for paper in ledgers.papers
        if paper.screen_status.value == "included"
        and paper.map_status.value == "success"
        and paper.eligible is True
    }
    return [
        finding.model_dump(mode="json")
        for finding in ledgers.findings
        if finding.paper_id in eligible
        and finding.outcome_family == primary_family
        and finding.effect_direction.value in {"increase", "no_effect", "decrease"}
        and finding.grounding_status.value == "exact"
        and not finding.section_flagged
    ]


def _preflight(paths: Sequence[Path], *, force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        rendered = ",".join(sorted(path.as_posix() for path in existing))
        raise FixtureContractError(f"fixture_outputs_exist:{rendered}")


def _anchor_expectations(config: QuestionConfig, ledgers: ExtractionLedgers) -> dict[str, Any]:
    anchors = [anchor.model_dump(mode="json") for anchor in (config.anchor_papers or [])]
    by_paper: dict[str, list[str]] = {}
    for finding in ledgers.findings:
        by_paper.setdefault(finding.paper_id, []).append(finding.effect_direction.value)
    for anchor in anchors:
        paper = next(
            (candidate for candidate in ledgers.papers if candidate.paper_id == anchor["paper_id"]),
            None,
        )
        if paper is None or paper.eligible is not anchor["expected_eligible"]:
            raise FixtureContractError("fixture_anchor_eligibility_mismatch")
        observed = by_paper.get(anchor["paper_id"], [])
        if len(observed) != anchor["expected_finding_count"]:
            raise FixtureContractError("fixture_anchor_finding_count_mismatch")
        if observed != anchor["expected_directions"]:
            raise FixtureContractError("fixture_anchor_direction_mismatch")
    return {
        "anchor_expectations_version": FIXTURE_VERSION,
        "question_id": config.question_id,
        "anchors": anchors,
    }


def generate_fixture(
    question_id: str,
    *,
    explicit_fixture: bool,
    fault_injection: FixtureFaultMode | None = None,
    force: bool = False,
    paths: ProjectPaths = PATHS,
) -> GeneratedFixture:
    """Generate one frozen fixture without invoking a provider or deriving gates/results."""

    if question_id not in FIXTURE_QUESTION_IDS:
        raise FixtureContractError(f"fixture_question_not_allowed:{question_id}")
    config = load_config_for_question(question_id, root=paths.root, require_locked=True)
    config.authorize_stage("s3", explicit_fixture=explicit_fixture, live_provider=False)
    if fault_injection is not None:
        if fault_injection != FIXTURE_FAULT_MODE:
            raise FixtureContractError(f"fixture_fault_mode_unknown:{fault_injection}")
        if not explicit_fixture:
            raise FixtureContractError("fixture_fault_injection_requires_fixture_flag")
        if question_id != "fixture-b-incomplete":
            raise FixtureContractError(
                "fixture_fault_injection_requires_fixture_b_incomplete"
            )

    schema = generate_extraction_schema(config)
    schema_path = paths.schema_path(question_id)
    if not schema_path.is_file():
        raise FixtureContractError(f"fixture_schema_missing:{schema_path}")
    try:
        archived_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FixtureContractError(f"fixture_schema_invalid:{schema_path}") from exc
    if hash_canonical(archived_schema) != hash_canonical(schema):
        raise FixtureContractError(f"fixture_schema_stale:{schema_path}")

    prompt = render_prompt_file(
        paths.prompts_dir / "extraction.md", extraction_prompt_replacements(config)
    )
    if prompt.prompt_version != FIXTURE_PROMPT_VERSION:
        raise FixtureContractError("fixture_extraction_prompt_version_changed")
    cfg_hash = extraction_cfghash(config, prompt.text, schema)
    raw_dir = paths.raw_map_dir(question_id)
    extracted_dir = paths.extracted_dir(question_id)
    processed_dir = paths.processed_dir(question_id)
    analysis_dir = paths.analysis_dir(question_id)

    lines_path = raw_dir / "authoritative_lines.json"
    envelopes_path = raw_dir / "fixture_map_envelopes.json"
    rendered_prompt_path = raw_dir / "rendered_extraction_prompt.md"
    fixture_contract_path = raw_dir / "fixture_contract.json"
    papers_path = extracted_dir / "papers.jsonl"
    findings_path = extracted_dir / "findings.jsonl"
    quarantine_path = extracted_dir / "quarantine.jsonl"
    run_path = extracted_dir / "run.json"
    audit_sample_path = processed_dir / "audit_sample.json"
    audit_decisions_path = processed_dir / "audit_decisions.json"
    anchor_path = processed_dir / "anchor_expectations.json"
    verification_path = processed_dir / "verification.json"
    baseline_path = analysis_dir / "baseline.json"
    fault_path = (
        processed_dir / "fixture_fault_injection.json" if fault_injection is not None else None
    )
    targets = [
        lines_path,
        envelopes_path,
        rendered_prompt_path,
        fixture_contract_path,
        papers_path,
        findings_path,
        quarantine_path,
        run_path,
        audit_sample_path,
        audit_decisions_path,
        anchor_path,
        verification_path,
        baseline_path,
    ]
    if fault_path is not None:
        targets.append(fault_path)
    _preflight(targets, force=force)

    screened, envelopes, source_lines = _fixture_sources(config)
    atomic_write_json(lines_path, source_lines, force=force)
    atomic_write_json(envelopes_path, _envelope_payload(question_id, envelopes), force=force)
    atomic_write_text(rendered_prompt_path, prompt.text, force=force)
    ledgers = assemble_extraction_ledgers(
        screened,
        envelopes,
        config=config,
        prompt_version=prompt.prompt_version,
        cfghash=cfg_hash,
        raw_artifact_path=paths.repository_relative(envelopes_path),
        source_lines_by_doc=source_lines,
        created_at=FIXTURE_CREATED_AT,
    )
    paper_rows = [paper.model_dump(mode="json") for paper in ledgers.papers]
    finding_rows = [finding.model_dump(mode="json") for finding in ledgers.findings]
    quarantine_rows = [dict(row) for row in ledgers.quarantine]
    atomic_write_jsonl(papers_path, paper_rows, force=force)
    atomic_write_jsonl(findings_path, finding_rows, force=force)
    atomic_write_jsonl(quarantine_path, quarantine_rows, force=force)

    exact_ids = [
        finding.finding_id
        for finding in ledgers.findings
        if finding.grounding_status.value == "exact"
    ]
    pathology_ids = {
        "mixed": [
            finding.finding_id
            for finding in ledgers.findings
            if finding.effect_direction.value == "mixed"
        ],
        "unclear": [
            finding.finding_id
            for finding in ledgers.findings
            if finding.effect_direction.value == "unclear"
        ],
        "ungrounded": [
            finding.finding_id
            for finding in ledgers.findings
            if finding.grounding_status.value != "exact"
        ],
        "section_flagged": [
            finding.finding_id for finding in ledgers.findings if finding.section_flagged
        ],
    }
    fixture_contract = {
        "fixture_contract_version": FIXTURE_VERSION,
        "question_id": question_id,
        "created_at": FIXTURE_CREATED_AT,
        "extraction_tuple": {
            "prompt_version": prompt.prompt_version,
            "schema_version": config.schema_version,
            "cfghash": cfg_hash,
        },
        "counts": dict(ledgers.counts),
        "exact_grounded_finding_ids": exact_ids,
        "pathology_finding_ids": pathology_ids,
        "heavy_paper_id": ledgers.papers[HEAVY_PAPER_INDEX].paper_id,
    }
    atomic_write_json(fixture_contract_path, fixture_contract, force=force)

    primary_family = config.outcomes.primary_family
    assert primary_family is not None
    audit_candidates = _audit_candidates(ledgers, primary_family=primary_family)
    audit_sample = select_audit_sample(
        audit_candidates,
        sample_size=20,
        seed=config.analysis.seed,
    )
    atomic_write_json(
        audit_sample_path,
        {"audit_sample_version": "1", "scope": "v1", **audit_sample},
        force=force,
    )
    checks = {name: True for name in DEFAULT_AUDIT_FIELDS}
    atomic_write_json(
        audit_decisions_path,
        {
            "audit_decisions_version": "1",
            "human_completion_required": False,
            "decisions": [
                {"finding_id": finding_id, "checks": checks, "error_codes": []}
                for finding_id in audit_sample["finding_ids"]
            ],
            "anchor_results": {
                anchor.paper_id: True for anchor in (config.anchor_papers or [])
            },
        },
        force=force,
    )
    atomic_write_json(anchor_path, _anchor_expectations(config, ledgers), force=force)

    definitions = {
        "increase": config.target_relation.increase_definition,
        "no_effect": config.target_relation.no_effect_definition,
        "decrease": config.target_relation.decrease_definition,
        "comparator": config.target_relation.comparator,
        "outcome": config.target_relation.outcome,
    }
    request_rows = [
        {
            "finding_id": finding["finding_id"],
            "proposed_direction": finding["effect_direction"],
            "outcome_name": finding["outcome_name"],
            "timepoint": finding["timepoint_raw"],
            "comparator": finding["comparator"],
            "evidence_quote": finding["evidence_quote"],
            "evidence_lines": finding["evidence_lines"],
        }
        for finding in finding_rows
        if finding["grounding_status"] == "exact"
    ]
    verification_prompt = render_prompt_file(
        paths.prompts_dir / "quote_verification.md",
        {
            "DIRECTION_DEFINITIONS_JSON": json.dumps(
                definitions,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            "FINDINGS_JSON": json.dumps(
                request_rows,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        },
    )
    verification = fixture_verification(
        findings=finding_rows,
        rendered_prompt=verification_prompt,
    )
    atomic_write_json(verification_path, verification, force=force)

    # Compute the same flattened representation s4 writes, but do not write or forge s4 outputs.
    papers_frame, findings_frame = normalized_frames(
        ledgers.papers,
        ledgers.findings,
        moderator_names=[moderator.name for moderator in config.moderators],
        moderator_types={moderator.name: moderator.type for moderator in config.moderators},
    )
    normalized_papers = papers_frame.to_dict(orient="records")
    normalized_findings = findings_frame.to_dict(orient="records")
    primary = derive_primary_cohort(
        normalized_papers,
        normalized_findings,
        verification,
        primary_family=primary_family,
    )
    baseline = create_baseline(
        cohort_hash=cohort_sha256(primary),
        research_question=config.research_question,
        primary_rows=primary,
        prompt_path=paths.prompts_dir / "baseline_consensus.md",
        attempted_at=FIXTURE_CREATED_AT + timedelta(minutes=5),
        fixture_mode=True,
        provider=None,
    )
    atomic_write_json(baseline_path, baseline, force=force)

    if fault_path is not None:
        atomic_write_json(
            fault_path,
            {
                "fixture_fault_injection_version": "1",
                "question_id": question_id,
                "mode": fault_injection,
            },
            force=force,
        )

    config_path = paths.config_path(question_id)
    evidence_outputs = [
        audit_sample_path,
        audit_decisions_path,
        anchor_path,
        verification_path,
        baseline_path,
    ]
    if fault_path is not None:
        evidence_outputs.append(fault_path)
    run = RunRecord(
        run_id=f"s3-fixture-{question_id}-v1",
        question_id=question_id,
        stage="s3",
        stage_version="1",
        status="complete",
        started_at=FIXTURE_CREATED_AT,
        completed_at=FIXTURE_CREATED_AT + timedelta(seconds=1),
        code_version="fixture-generator-v1",
        command_argv=[
            "scripts/generate_fixture.py",
            "--question",
            question_id,
            "--fixture",
            *(
                ["--fault-injection", fault_injection]
                if fault_injection is not None
                else []
            ),
        ],
        config_path=paths.repository_relative(config_path),
        config_sha256=config_sha256(config),
        prompt_path=paths.repository_relative(rendered_prompt_path),
        prompt_sha256=prompt.sha256,
        schema_path=paths.repository_relative(schema_path),
        schema_sha256=sha256_file(schema_path),
        cfghash=cfg_hash,
        upstream=[],
        inputs=[
            artifact_ref(config_path, root=paths.root),
            artifact_ref(schema_path, root=paths.root),
            artifact_ref(lines_path, root=paths.root),
            artifact_ref(envelopes_path, root=paths.root),
        ],
        outputs=[
            artifact_ref(papers_path, root=paths.root, rows=len(paper_rows)),
            artifact_ref(findings_path, root=paths.root, rows=len(finding_rows)),
            artifact_ref(quarantine_path, root=paths.root, rows=len(quarantine_rows)),
            artifact_ref(rendered_prompt_path, root=paths.root),
            artifact_ref(fixture_contract_path, root=paths.root),
            *[artifact_ref(path, root=paths.root) for path in evidence_outputs],
        ],
        external_result_ids={},
        counts={
            **dict(ledgers.counts),
            "exact_grounded_findings": len(exact_ids),
            "audit_decisions": 20,
            "verification_decisions": len(verification.decisions),
        },
        warnings=["fixture_synthetic_inputs"],
    )
    write_run_record(run_path, run, force=force)
    return GeneratedFixture(
        question_id=question_id,
        papers_path=papers_path,
        findings_path=findings_path,
        quarantine_path=quarantine_path,
        run_record_path=run_path,
        authoritative_lines_path=lines_path,
        map_envelopes_path=envelopes_path,
        audit_sample_path=audit_sample_path,
        audit_decisions_path=audit_decisions_path,
        verification_path=verification_path,
        baseline_path=baseline_path,
        fault_injection_path=fault_path,
        counts=run.counts,
    )


def generate_all_fixtures(
    *,
    explicit_fixture: bool,
    force: bool = False,
    paths: ProjectPaths = PATHS,
) -> list[GeneratedFixture]:
    """Generate the four scenarios in their frozen order."""

    return [
        generate_fixture(
            question_id,
            explicit_fixture=explicit_fixture,
            force=force,
            paths=paths,
        )
        for question_id in FIXTURE_ORDER
    ]


__all__ = [
    "FIXTURE_CREATED_AT",
    "FIXTURE_FAULT_MODE",
    "FIXTURE_ORDER",
    "FIXTURE_VERSION",
    "FixtureContractError",
    "GeneratedFixture",
    "generate_all_fixtures",
    "generate_fixture",
]
