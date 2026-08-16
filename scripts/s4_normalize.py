#!/usr/bin/env python3
"""Regenerate strict processed parquet ledgers from immutable s3 JSONL."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml

from literature_multiverse.config import config_sha256, load_config_for_question
from literature_multiverse.extract import reconcile_ledger_counts
from literature_multiverse.lineage import (
    artifact_ref,
    atomic_write_json,
    code_version,
    sha256_file,
    validate_upstream_chain,
    verify_artifact,
    write_run_record,
)
from literature_multiverse.models import PaperRecord, RunRecord, UpstreamRef
from literature_multiverse.normalize import apply_patches, write_processed_ledgers
from literature_multiverse.paths import PATHS
from literature_multiverse.schemas import validate_finding_row


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--question", required=True)
    result.add_argument("--patches", type=Path)
    result.add_argument("--allow-mixed", action="store_true")
    result.add_argument("--fixture", action="store_true")
    result.add_argument("--force", action="store_true")
    return result


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_patches(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    value = yaml.safe_load(path.read_text())
    if isinstance(value, dict):
        value = value.get("patches")
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("patch_file_must_contain_a_patches_array")
    if any(item.get("ledger", "findings") != "findings" for item in value):
        raise ValueError("s4_patch_ledger_must_be_findings")
    return [{key: item for key, item in patch.items() if key != "ledger"} for patch in value]


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    started = datetime.now(UTC)
    config = load_config_for_question(args.question, require_locked=True)
    config.authorize_stage("s4", explicit_fixture=args.fixture, live_provider=False)
    config_hash = config_sha256(config)
    s3_run_path = PATHS.run_record_path(config.question_id, "s3")
    s3_run = RunRecord.model_validate_json(s3_run_path.read_text())
    upstream = UpstreamRef(
        stage="s3",
        run_id=s3_run.run_id,
        run_record_path=PATHS.repository_relative(s3_run_path),
        run_record_sha256=sha256_file(s3_run_path),
    )
    validate_upstream_chain(
        current_stage="s4",
        upstream=[upstream],
        root=PATHS.root,
        expected_config_sha256=config_hash,
    )
    for output in s3_run.outputs:
        verify_artifact(output, root=PATHS.root)

    extracted_dir = PATHS.extracted_dir(config.question_id)
    papers_jsonl = extracted_dir / "papers.jsonl"
    findings_jsonl = extracted_dir / "findings.jsonl"
    quarantine_jsonl = extracted_dir / "quarantine.jsonl"
    papers = [PaperRecord.model_validate(row) for row in _read_jsonl(papers_jsonl)]
    raw_findings = _read_jsonl(findings_jsonl)
    quarantine_rows = _read_jsonl(quarantine_jsonl)
    patches_path = args.patches
    if patches_path is None:
        default_patch_path = PATHS.patches_path(config.question_id)
        patches_path = default_patch_path if default_patch_path.exists() else None
    patches = _load_patches(patches_path)
    patched_findings, patch_audit = apply_patches(raw_findings, patches)
    findings = [validate_finding_row(row, config) for row in patched_findings]
    reconcile_ledger_counts(
        [paper.model_dump(mode="json") for paper in papers],
        [finding.model_dump(mode="json") for finding in findings],
        quarantine_rows,
        s2_paper_ids=[paper.paper_id for paper in papers],
    )

    processed_dir = PATHS.processed_dir(config.question_id)
    papers_parquet = processed_dir / "papers.parquet"
    findings_parquet = processed_dir / "findings.parquet"
    report_path = processed_dir / "normalization_report.json"
    paper_count, finding_count = write_processed_ledgers(
        papers,
        findings,
        papers_path=papers_parquet,
        findings_path=findings_parquet,
        allow_mixed=args.allow_mixed,
        moderator_names=[moderator.name for moderator in config.moderators],
        moderator_types={moderator.name: moderator.type for moderator in config.moderators},
        force=args.force,
    )
    tuples = sorted(
        {(finding.prompt_version, finding.schema_version, finding.cfghash) for finding in findings}
    )
    quarantine_count = len(quarantine_rows)
    report = {
        "report_version": "1",
        "question_id": config.question_id,
        "papers": paper_count,
        "findings": finding_count,
        "quarantine_rows": quarantine_count,
        "patches_applied": len(patch_audit),
        "patch_audit": patch_audit,
        "extraction_tuples": [list(item) for item in tuples],
        "allow_mixed": args.allow_mixed,
        "referential_integrity": "passed",
    }
    atomic_write_json(report_path, report, force=args.force)

    config_path = PATHS.config_path(config.question_id)
    inputs = [
        artifact_ref(config_path, root=PATHS.root),
        artifact_ref(papers_jsonl, root=PATHS.root, rows=len(papers)),
        artifact_ref(findings_jsonl, root=PATHS.root, rows=len(raw_findings)),
        artifact_ref(quarantine_jsonl, root=PATHS.root, rows=quarantine_count),
    ]
    if patches_path is not None:
        inputs.append(artifact_ref(patches_path.resolve(), root=PATHS.root, rows=len(patches)))
    record = RunRecord(
        run_id=f"s4-{uuid.uuid4().hex}",
        question_id=config.question_id,
        stage="s4",
        stage_version="1",
        status="complete",
        started_at=started,
        completed_at=datetime.now(UTC),
        code_version=code_version(PATHS.root),
        command_argv=["scripts/s4_normalize.py", *(argv if argv is not None else sys.argv[1:])],
        config_path=PATHS.repository_relative(config_path),
        config_sha256=config_hash,
        prompt_path=None,
        prompt_sha256=None,
        schema_path=None,
        schema_sha256=None,
        cfghash=None,
        upstream=[upstream],
        inputs=inputs,
        outputs=[
            artifact_ref(papers_parquet, root=PATHS.root, rows=paper_count),
            artifact_ref(findings_parquet, root=PATHS.root, rows=finding_count),
            artifact_ref(report_path, root=PATHS.root),
        ],
        external_result_ids={},
        counts={
            "papers": paper_count,
            "findings": finding_count,
            "quarantine_rows": quarantine_count,
            "patches_applied": len(patch_audit),
            "extraction_tuples": len(tuples),
        },
        warnings=["diagnostic_allow_mixed"] if args.allow_mixed else [],
    )
    write_run_record(PATHS.run_record_path(config.question_id, "s4"), record, force=args.force)
    print(f"s4 complete: {paper_count} papers, {finding_count} findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
