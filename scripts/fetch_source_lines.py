#!/usr/bin/env python3
"""Fetch full content.lines for every s2-included paper and derive grounded source lines."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from literature_multiverse.config import config_sha256, load_config_for_question
from literature_multiverse.lineage import (
    artifact_ref,
    atomic_write_json,
    code_version,
    write_run_record,
)
from literature_multiverse.live import paperclip_executable
from literature_multiverse.models import RunRecord
from literature_multiverse.paperclip_cli import run_paperclip
from literature_multiverse.paths import PATHS
from literature_multiverse.source_lines import SourceLinesParseError, parse_content_lines

HEAD_LINES = "100000"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--question", required=True)
    result.add_argument(
        "--from-archive",
        action="store_true",
        help="re-derive sections offline from archived content_lines attempts (no provider)",
    )
    result.add_argument(
        "--prior",
        type=Path,
        default=None,
        help="fallback source_lines.json from a prior fetch run; its verbatim text is "
        "reconstructed for docs whose archives were clobbered by failed retries",
    )
    result.add_argument("--force", action="store_true")
    return result


def _raw_from_prior(entry: dict) -> str:
    lines = sorted(entry.items(), key=lambda item: int(item[0][1:]))
    return "\n".join(f"{line_id}: {record['text']}" for line_id, record in lines)


def _raw_from_archive(raw_dir: Path, doc_id: str) -> str | None:
    candidates = sorted(raw_dir.glob(f"{doc_id.lower()}.attempt-*.stdout"), reverse=True)
    for path in candidates:
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            parse_content_lines(raw)
        except SourceLinesParseError:
            continue
        return raw
    return None


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    started = datetime.now(UTC)
    config = load_config_for_question(args.question, require_locked=True)
    config.authorize_stage("s3", explicit_fixture=False, live_provider=True)

    screen_dir = PATHS.raw_screen_dir(config.question_id)
    screened = [
        json.loads(line)
        for line in (screen_dir / "screened_papers.jsonl").read_text().splitlines()
        if line.strip()
    ]
    include_docs = sorted(
        str(paper["doc_id"]) for paper in screened if paper["screen_status"] == "included"
    )
    if not include_docs:
        raise ValueError("no included papers to fetch")

    raw_dir = PATHS.raw_map_dir(config.question_id) / "content_lines"
    prior: dict[str, dict] = {}
    if args.prior is not None:
        prior = json.loads(args.prior.read_text())
    source_lines: dict[str, dict[str, dict[str, str]]] = {}
    failures: dict[str, str] = {}
    source_counts = {"live": 0, "archive": 0, "prior": 0}
    executable = None if args.from_archive else paperclip_executable()
    for index, doc_id in enumerate(include_docs, start=1):
        raw: str | None = None
        if args.from_archive:
            raw = _raw_from_archive(raw_dir, doc_id)
            if raw is not None:
                source_counts["archive"] += 1
            elif doc_id in prior:
                raw = _raw_from_prior(prior[doc_id])
                source_counts["prior"] += 1
            else:
                failures[doc_id] = "no_parseable_archive"
                continue
        else:
            run = run_paperclip(
                [executable, "head", "-" + HEAD_LINES, f"/papers/{doc_id}/content.lines"],
                archive_dir=raw_dir,
                archive_stem=f"{doc_id.lower()}",
                timeout_seconds=180,
                max_retries=2,
                force=args.force,
            )
            if not run.ok:
                failures[doc_id] = str(run.final.failure_code)
                continue
            raw = run.final.stdout.decode("utf-8", "replace")
            source_counts["live"] += 1
        try:
            source_lines[doc_id] = parse_content_lines(raw)
        except SourceLinesParseError as exc:
            failures[doc_id] = f"parse:{exc}"
        if index % 25 == 0:
            print(f"fetched {index}/{len(include_docs)}", flush=True)

    output_path = PATHS.raw_map_dir(config.question_id) / "source_lines.json"
    atomic_write_json(output_path, source_lines, force=args.force)
    failures_path = PATHS.raw_map_dir(config.question_id) / "source_lines_failures.json"
    atomic_write_json(failures_path, failures, force=args.force)

    config_path = PATHS.config_path(config.question_id)
    record = RunRecord(
        run_id=f"source-lines-{uuid.uuid4().hex}",
        question_id=config.question_id,
        stage="s3",
        stage_version="1-source-lines",
        status="complete",
        started_at=started,
        completed_at=datetime.now(UTC),
        code_version=code_version(PATHS.root),
        command_argv=[
            "scripts/fetch_source_lines.py",
            *(argv if argv is not None else sys.argv[1:]),
        ],
        config_path=PATHS.repository_relative(config_path),
        config_sha256=config_sha256(config),
        prompt_path=None,
        prompt_sha256=None,
        schema_path=None,
        schema_sha256=None,
        cfghash=None,
        upstream=[],
        inputs=[artifact_ref(config_path, root=PATHS.root)],
        outputs=[
            artifact_ref(output_path, root=PATHS.root),
            artifact_ref(failures_path, root=PATHS.root),
        ],
        external_result_ids={},
        counts={
            "included_papers": len(include_docs),
            "fetched_papers": len(source_lines),
            "failed_papers": len(failures),
            **{f"source_{key}": value for key, value in source_counts.items() if value},
        },
        warnings=(
            ([f"content_fetch_failures:{len(failures)}"] if failures else [])
            + (["offline_archived_input"] if args.from_archive else [])
        ),
    )
    write_run_record(
        PATHS.raw_map_dir(config.question_id) / "source_lines_run.json",
        record,
        force=args.force,
    )
    print(
        f"source lines complete: {len(source_lines)}/{len(include_docs)} papers "
        f"({len(failures)} failures) -> {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
