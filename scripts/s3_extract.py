#!/usr/bin/env python3
"""Map only s2-included papers and build lossless terminal s3 ledgers."""

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
    hash_canonical,
    sha256_file,
    validate_upstream_chain,
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
    mode.add_argument(
        "--map-output",
        type=Path,
        nargs="+",
        help="archived map stdout artifact(s) for offline ingestion, one per corpus batch",
    )
    mode.add_argument("--live", action="store_true")
    result.add_argument(
        "--from-result",
        action="append",
        help="exact provider set(s) jointly matching the s2 include list; repeat per batch",
    )
    result.add_argument("--resume-map-id", action="append")
    result.add_argument("--source-lines", type=Path, help="JSON mapping doc_id to line records")
    result.add_argument("--concurrency", type=int, default=None, help="gated: GXL testers only")
    result.add_argument("--fixture", action="store_true")
    result.add_argument("--force", action="store_true")
    return result


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.live and not (args.from_result or args.resume_map_id):
        raise ValueError("live_s3_requires_from_result_or_resume_map_id")
    if args.from_result and args.resume_map_id:
        raise ValueError("from_result_and_resume_map_id_are_mutually_exclusive")
    if args.concurrency is not None and args.concurrency < 1:
        raise ValueError("concurrency_must_be_positive")

    started = datetime.now(UTC)
    config = load_config_for_question(args.question, require_locked=True)
    config.authorize_stage("s3", explicit_fixture=args.fixture, live_provider=args.live)
    config_hash = config_sha256(config)
    s2_run_path = PATHS.run_record_path(config.question_id, "s2")
    s2_run = RunRecord.model_validate_json(s2_run_path.read_text())
    upstream = UpstreamRef(
        stage="s2",
        run_id=s2_run.run_id,
        run_record_path=PATHS.repository_relative(s2_run_path),
        run_record_sha256=sha256_file(s2_run_path),
    )
    validate_upstream_chain(
        current_stage="s3",
        upstream=[upstream],
        root=PATHS.root,
        expected_config_sha256=config_hash,
    )
    for output in s2_run.outputs:
        verify_artifact(output, root=PATHS.root)

    screen_dir = PATHS.raw_screen_dir(config.question_id)
    screened_path = screen_dir / "screened_papers.jsonl"
    include_path = screen_dir / "include_paper_ids.json"
    exclude_path = screen_dir / "exclude_paper_ids.json"
    screened = _read_jsonl(screened_path)
    include_ids = json.loads(include_path.read_text())
    exclude_ids = json.loads(exclude_path.read_text())
    ledger_include = sorted(
        paper["paper_id"] for paper in screened if paper["screen_status"] == "included"
    )
    ledger_exclude = sorted(
        paper["paper_id"] for paper in screened if paper["screen_status"] == "excluded"
    )
    if sorted(include_ids) != ledger_include or sorted(exclude_ids) != ledger_exclude:
        raise ValueError("s2_include_exclude_files_do_not_match_screened_ledger")
    if args.live and len(include_ids) > 10:
        g1b_path = PATHS.root / "data/raw/smoke/g1b_report.json"
        g1b = json.loads(g1b_path.read_text()) if g1b_path.exists() else {}
        if g1b.get("g1b_passed") is not True:
            raise ValueError("g1b_must_pass_before_live_map_over_10_papers")

    schema = generate_extraction_schema(config)
    schema_path = PATHS.schema_path(config.question_id)
    if schema_path.exists():
        if hash_canonical(json.loads(schema_path.read_text())) != hash_canonical(schema):
            raise ValueError("stale_extraction_schema")
    else:
        atomic_write_json(schema_path, schema)
    prompt = render_prompt_file(
        PATHS.prompts_dir / "extraction.md", extraction_prompt_replacements(config)
    )
    raw_dir = PATHS.raw_map_dir(config.question_id)
    rendered_prompt_path = raw_dir / "rendered_extraction_prompt.md"
    atomic_write_text(rendered_prompt_path, prompt.text, force=args.force)
    cfg_hash = extraction_cfghash(config, prompt.text, schema)

    provider_artifacts: list[Path] = []
    map_paths: list[Path] = []
    if args.live:
        if args.resume_map_id:
            batches = [{"resume": map_id} for map_id in args.resume_map_id]
        else:
            batches = [{"from_result": result_id} for result_id in args.from_result]
        for index, batch in enumerate(batches, start=1):
            resume_id = batch.get("resume")
            live = live_map_to_results_file(
                archive_dir=raw_dir,
                archive_stem=f"map-{cfg_hash[:8]}-b{index:02d}",
                from_result=None if resume_id else batch["from_result"],
                schema_json=None if resume_id else json.dumps(schema, sort_keys=True),
                prompt=None if resume_id else prompt.text,
                concurrency=None if resume_id else args.concurrency,
                resume_map_id=resume_id,
                retry_failed=bool(resume_id),
                force=args.force,
            )
            map_paths.append(live.results_path)
            provider_artifacts.extend(live.artifacts)
    else:
        assert args.map_output is not None
        map_paths = [output.resolve() for output in args.map_output]
    try:
        raw_relatives = [PATHS.repository_relative(path) for path in map_paths]
    except ValueError as exc:
        raise ValueError("map_output_must_be_archived_inside_repository") from exc
    envelopes = []
    raw_relative_by_doc: dict[str, str] = {}
    for path, relative in zip(map_paths, raw_relatives, strict=True):
        parsed = parse_map_file(path)
        envelopes.extend(parsed)
        for envelope in parsed:
            raw_relative_by_doc[envelope.doc_id] = relative
    raw_relative = raw_relatives[0]

    source_lines = None
    if args.source_lines is not None:
        source_lines = json.loads(args.source_lines.read_text())
        if not isinstance(source_lines, dict):
            raise ValueError("source_lines_root_must_be_mapping")
    ledgers = assemble_extraction_ledgers(
        screened,
        envelopes,
        config=config,
        prompt_version=prompt.prompt_version,
        cfghash=cfg_hash,
        raw_artifact_path=raw_relative,
        raw_artifact_path_by_doc=raw_relative_by_doc,
        source_lines_by_doc=source_lines,
        created_at=started,
    )

    extracted_dir = PATHS.extracted_dir(config.question_id)
    papers_path = extracted_dir / "papers.jsonl"
    findings_path = extracted_dir / "findings.jsonl"
    quarantine_path = extracted_dir / "quarantine.jsonl"
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
    config_path = PATHS.config_path(config.question_id)
    inputs = [
        artifact_ref(config_path, root=PATHS.root),
        artifact_ref(screened_path, root=PATHS.root, rows=len(screened)),
        artifact_ref(include_path, root=PATHS.root, rows=len(include_ids)),
        artifact_ref(exclude_path, root=PATHS.root, rows=len(exclude_ids)),
        *(artifact_ref(path, root=PATHS.root) for path in map_paths),
    ]
    if args.source_lines is not None:
        inputs.append(artifact_ref(args.source_lines.resolve(), root=PATHS.root))
    record = RunRecord(
        run_id=f"s3-{uuid.uuid4().hex}",
        question_id=config.question_id,
        stage="s3",
        stage_version="1",
        status="complete",
        started_at=started,
        completed_at=datetime.now(UTC),
        code_version=code_version(PATHS.root),
        command_argv=["scripts/s3_extract.py", *(argv if argv is not None else sys.argv[1:])],
        config_path=PATHS.repository_relative(config_path),
        config_sha256=config_hash,
        prompt_path=PATHS.repository_relative(rendered_prompt_path),
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
            artifact_ref(rendered_prompt_path, root=PATHS.root),
            *(artifact_ref(path, root=PATHS.root) for path in provider_artifacts),
        ],
        external_result_ids={
            "paperclip": sorted({envelope.map_result_id for envelope in envelopes})
        },
        counts=dict(ledgers.counts),
        warnings=[] if source_lines is not None else ["source_lines_unavailable_rows_unverifiable"],
    )
    write_run_record(PATHS.run_record_path(config.question_id, "s3"), record, force=args.force)
    print(
        f"s3 complete: {len(ledgers.papers)} papers, {len(ledgers.findings)} accepted, "
        f"{len(ledgers.quarantine)} quarantined"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
