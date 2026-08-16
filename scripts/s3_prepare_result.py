#!/usr/bin/env python3
"""Build and verify exact provider result sets covering the whole s2 include list.

Uses dedicated paper repos (``repo add`` with explicit document IDs) plus repo-scoped
searches, then proves by CSV export that each saved result equals its batch exactly.
The provider caps any single search result at 500 papers, so include lists larger than
``BATCH_SIZE`` are split into deterministic sorted batches — one repo and one result id
per batch — and downstream s3 maps every batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json

from literature_multiverse.config import load_config_for_question
from literature_multiverse.lineage import atomic_write_json
from literature_multiverse.live import paperclip_executable, search_id_from_stdout
from literature_multiverse.paperclip_cli import require_success, run_paperclip
from literature_multiverse.paths import PATHS
from literature_multiverse.search import parse_search_csv

# Provider search results silently cap at 500 rows; stay comfortably below.
BATCH_SIZE = 450


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--question", required=True)
    result.add_argument("--force", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config_for_question(args.question, require_locked=True)
    config.authorize_stage("s3", explicit_fixture=False, live_provider=True)

    screen_dir = PATHS.raw_screen_dir(config.question_id)
    include_docs = sorted(
        str(paper["doc_id"])
        for paper in (
            json.loads(line)
            for line in (screen_dir / "screened_papers.jsonl").read_text().splitlines()
            if line.strip()
        )
        if paper["screen_status"] == "included"
    )
    if not include_docs:
        raise ValueError("no included papers")

    digest = hashlib.sha256("\n".join(include_docs).encode()).hexdigest()[:10]
    executable = paperclip_executable()
    archive = PATHS.raw_map_dir(config.question_id) / "corpus_set"
    chunks = [
        include_docs[start : start + BATCH_SIZE]
        for start in range(0, len(include_docs), BATCH_SIZE)
    ]

    batches = []
    for index, chunk in enumerate(chunks, start=1):
        stem = f"b{index:02d}"
        repo_name = f"{config.question_id}-corpus-{digest}-{stem}"
        init_run = run_paperclip(
            [executable, "repo", "init", repo_name, "verified full-corpus extraction set"],
            archive_dir=archive,
            archive_stem=f"repo-init-{stem}",
            timeout_seconds=120,
            force=args.force,
        )
        init_text = (init_run.final.stdout + init_run.final.stderr).decode("utf-8", "replace")
        if not init_run.ok and "already exists" not in init_text.casefold():
            raise ValueError(f"corpus repo init failed: see {init_run.final.stderr_path}")
        add_run = run_paperclip(
            [executable, "--repo", repo_name, "repo", "add", *chunk],
            archive_dir=archive,
            archive_stem=f"repo-add-{stem}",
            timeout_seconds=900,
            force=args.force,
        )
        require_success(add_run)

        search_run = run_paperclip(
            [
                executable,
                "--repo",
                repo_name,
                "--repo-only",
                "search",
                "-s",
                ",".join(config.search.sources),
                "-n",
                str(min(500, len(chunk) + 50)),
                "--all",
                config.research_question,
            ],
            archive_dir=archive,
            archive_stem=f"corpus-search-{stem}",
            timeout_seconds=600,
            force=args.force,
        )
        search_attempt = require_success(search_run)
        result_id = search_id_from_stdout(search_attempt.stdout)

        corpus_csv = archive / f"corpus-set-{stem}.results.csv"
        if corpus_csv.exists() and not args.force:
            raise FileExistsError(f"refusing to replace {corpus_csv}")
        export_run = run_paperclip(
            [executable, "results", result_id, "--save", str(corpus_csv)],
            archive_dir=archive,
            archive_stem=f"corpus-set-{stem}.export",
            timeout_seconds=600,
            force=args.force,
        )
        require_success(export_run)
        observed = sorted(str(r["doc_id"]) for r in parse_search_csv(corpus_csv.read_bytes()))
        if observed != chunk:
            missing = sorted(set(chunk) - set(observed))
            extra = sorted(set(observed) - set(chunk))
            raise ValueError(
                f"corpus batch {stem} mismatch: missing={missing[:10]} ({len(missing)}) "
                f"extra={extra[:10]} ({len(extra)})"
            )
        batches.append(
            {
                "batch": stem,
                "corpus_repo": repo_name,
                "result_id": result_id,
                "paper_count": len(chunk),
            }
        )

    log = {
        "log_version": "2",
        "question_id": config.question_id,
        "paper_count": len(include_docs),
        "batch_size": BATCH_SIZE,
        "batches": batches,
    }
    atomic_write_json(archive / "corpus_set.json", log, force=args.force)
    summary = ", ".join(f"{b['result_id']} ({b['paper_count']})" for b in batches)
    print(
        f"corpus set verified: {len(include_docs)} papers "
        f"across {len(batches)} batches: {summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
