#!/usr/bin/env python3
"""Ingest an isolated, hard-capped ten-paper triage extraction probe."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from literature_multiverse.config import config_sha256, load_config_for_question
from literature_multiverse.extract import (
    assemble_extraction_ledgers,
    extraction_prompt_replacements,
    parse_map_file,
)
from literature_multiverse.lineage import (
    artifact_ref,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    code_version,
    extraction_cfghash,
    sha256_file,
    verify_artifact,
    write_run_record,
)
from literature_multiverse.live import live_map_to_results_file
from literature_multiverse.models import RunRecord, UpstreamRef
from literature_multiverse.paths import PATHS
from literature_multiverse.prompting import render_prompt_file
from literature_multiverse.schemas import generate_extraction_schema


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--question", required=True)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--map-output", type=Path)
    mode.add_argument("--live", action="store_true")
    result.add_argument("--from-result")
    result.add_argument("--resume-map-id")
    result.add_argument("--concurrency", type=int, default=None, help="gated: GXL testers only")
    result.add_argument("--screened-papers", type=Path)
    result.add_argument("--source-lines", type=Path)
    result.add_argument("--force", action="store_true")
    return result


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    started = datetime.now(UTC)
    config = load_config_for_question(args.question)
    if config.status != "triage":
        raise ValueError("triage_probe_requires_status_triage_config")
    screen_path = args.screened_papers or (
        PATHS.raw_screen_dir(config.question_id) / "screened_papers.jsonl"
    )
    screened = _read_jsonl(screen_path)
    included_count = sum(paper.get("screen_status") == "included" for paper in screened)
    config.authorize_stage(
        "triage_probe", live_provider=args.live, triage_paper_count=included_count
    )
    if len(screened) != 10 or included_count != 10:
        raise ValueError("triage_probe_requires_exactly_10_included_logged_papers")

    destination = PATHS.triage_dir(config.question_id)
    schema = generate_extraction_schema(config)
    schema_path = destination / "extraction.schema.json"
    atomic_write_json(schema_path, schema, force=args.force)
    prompt = render_prompt_file(
        PATHS.prompts_dir / "extraction.md", extraction_prompt_replacements(config)
    )
    prompt_path = destination / "rendered_extraction_prompt.md"
    atomic_write_text(prompt_path, prompt.text, force=args.force)
    cfg_hash = extraction_cfghash(config, prompt.text, schema)
    provider_artifacts: list[Path] = []
    if args.live:
        if bool(args.from_result) == bool(args.resume_map_id):
            raise ValueError("live_triage_requires_exactly_one_from_result_or_resume_map_id")
        if args.concurrency is not None and args.concurrency < 1:
            raise ValueError("concurrency_must_be_positive")
        live = live_map_to_results_file(
            archive_dir=destination,
            archive_stem=f"map-{cfg_hash[:8]}",
            from_result=None if args.resume_map_id else args.from_result,
            schema_json=None if args.resume_map_id else json.dumps(schema, sort_keys=True),
            prompt=None if args.resume_map_id else prompt.text,
            concurrency=None if args.resume_map_id else args.concurrency,
            resume_map_id=args.resume_map_id,
            retry_failed=bool(args.resume_map_id),
            force=args.force,
        )
        map_path = live.results_path
        provider_artifacts = list(live.artifacts)
    else:
        assert args.map_output is not None
        map_path = args.map_output.resolve()
    map_relative = PATHS.repository_relative(map_path)
    source_lines = json.loads(args.source_lines.read_text()) if args.source_lines else None
    ledgers = assemble_extraction_ledgers(
        screened,
        parse_map_file(map_path),
        config=config,
        prompt_version=prompt.prompt_version,
        cfghash=cfg_hash,
        raw_artifact_path=map_relative,
        source_lines_by_doc=source_lines,
        created_at=started,
        allow_triage=True,
    )
    papers_path = destination / "papers.jsonl"
    findings_path = destination / "findings.jsonl"
    quarantine_path = destination / "quarantine.jsonl"
    report_path = destination / "probe_report.json"
    atomic_write_jsonl(
        papers_path,
        [paper.model_dump(mode="json") for paper in ledgers.papers],
        force=args.force,
    )
    atomic_write_jsonl(
        findings_path,
        [finding.model_dump(mode="json") for finding in ledgers.findings],
        force=args.force,
    )
    atomic_write_jsonl(quarantine_path, ledgers.quarantine, force=args.force)
    report = {
        "report_version": "1",
        "question_id": config.question_id,
        "isolated": True,
        "production_upstream_authorized": False,
        "counts": dict(ledgers.counts),
    }
    atomic_write_json(report_path, report, force=args.force)

    s2_run_path = PATHS.run_record_path(config.question_id, "s2")
    s2_run = RunRecord.model_validate_json(s2_run_path.read_text())
    upstream = UpstreamRef(
        stage="s2",
        run_id=s2_run.run_id,
        run_record_path=PATHS.repository_relative(s2_run_path),
        run_record_sha256=sha256_file(s2_run_path),
    )
    for output in s2_run.outputs:
        verify_artifact(output, root=PATHS.root)
    config_path = PATHS.config_path(config.question_id)
    inputs = [
        artifact_ref(config_path, root=PATHS.root),
        artifact_ref(screen_path.resolve(), root=PATHS.root, rows=10),
        artifact_ref(map_path, root=PATHS.root),
    ]
    if args.source_lines:
        inputs.append(artifact_ref(args.source_lines.resolve(), root=PATHS.root))
    record = RunRecord(
        run_id=f"triage-probe-{uuid.uuid4().hex}",
        question_id=config.question_id,
        stage="triage_probe",
        stage_version="1",
        status="complete",
        started_at=started,
        completed_at=datetime.now(UTC),
        code_version=code_version(PATHS.root),
        command_argv=["scripts/triage_probe.py", *(argv if argv is not None else sys.argv[1:])],
        config_path=PATHS.repository_relative(config_path),
        config_sha256=config_sha256(config),
        prompt_path=PATHS.repository_relative(prompt_path),
        prompt_sha256=prompt.sha256,
        schema_path=PATHS.repository_relative(schema_path),
        schema_sha256=sha256_file(schema_path),
        cfghash=cfg_hash,
        upstream=[upstream],
        inputs=inputs,
        outputs=[
            artifact_ref(papers_path, root=PATHS.root, rows=len(ledgers.papers)),
            artifact_ref(findings_path, root=PATHS.root, rows=len(ledgers.findings)),
            artifact_ref(quarantine_path, root=PATHS.root, rows=len(ledgers.quarantine)),
            artifact_ref(report_path, root=PATHS.root),
            artifact_ref(prompt_path, root=PATHS.root),
            artifact_ref(schema_path, root=PATHS.root),
            *(artifact_ref(path, root=PATHS.root) for path in provider_artifacts),
        ],
        external_result_ids={
            "paperclip": sorted(
                {paper.map_result_id for paper in ledgers.papers if paper.map_result_id}
            )
        },
        counts=dict(ledgers.counts),
        warnings=["isolated_triage_artifacts_forbidden_as_production_upstream"],
    )
    write_run_record(
        PATHS.run_record_path(config.question_id, "triage_probe"), record, force=args.force
    )
    print(f"triage probe complete: {len(ledgers.findings)} accepted findings from 10 papers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
