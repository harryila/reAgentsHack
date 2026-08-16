#!/usr/bin/env python3
"""Create the deterministic, identity-deduplicated s2 screening ledger."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import UTC, datetime

from literature_multiverse.config import config_sha256, load_config_for_question
from literature_multiverse.lineage import (
    artifact_ref,
    atomic_write_json,
    atomic_write_jsonl,
    code_version,
    sha256_file,
    validate_upstream_chain,
    verify_artifact,
    write_run_record,
)
from literature_multiverse.models import RunRecord, UpstreamRef
from literature_multiverse.paths import PATHS
from literature_multiverse.screen import screen_candidates
from literature_multiverse.search import load_occurrences_jsonl


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--question", required=True)
    result.add_argument("--fixture", action="store_true")
    result.add_argument("--force", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    started = datetime.now(UTC)
    config = load_config_for_question(args.question)
    config.authorize_stage("s2", explicit_fixture=args.fixture, live_provider=False)
    config_hash = config_sha256(config)

    s1_run_path = PATHS.run_record_path(config.question_id, "s1")
    s1_run = RunRecord.model_validate_json(s1_run_path.read_text())
    upstream = UpstreamRef(
        stage="s1",
        run_id=s1_run.run_id,
        run_record_path=PATHS.repository_relative(s1_run_path),
        run_record_sha256=sha256_file(s1_run_path),
    )
    validate_upstream_chain(
        current_stage="s2",
        upstream=[upstream],
        root=PATHS.root,
        expected_config_sha256=config_hash,
    )
    for output in s1_run.outputs:
        verify_artifact(output, root=PATHS.root)

    candidate_path = PATHS.raw_search_dir(config.question_id) / "candidate_papers.jsonl"
    occurrences = load_occurrences_jsonl(candidate_path)
    result = screen_candidates(
        occurrences,
        allowed_article_types=config.eligibility.article_types,
        config_sha256=config_hash,
        schema_version=config.schema_version,
        created_at=started,
        audit_excluded_doc_ids={
            exclusion.paper_id.removeprefix("doc:"): exclusion.reason
            for exclusion in config.audit_paper_exclusions
        },
    )
    destination = PATHS.raw_screen_dir(config.question_id)
    screened_path = destination / "screened_papers.jsonl"
    include_path = destination / "include_paper_ids.json"
    exclude_path = destination / "exclude_paper_ids.json"
    dedupe_path = destination / "dedupe_log.jsonl"
    atomic_write_jsonl(screened_path, result.papers, force=args.force)
    atomic_write_json(include_path, list(result.include_paper_ids), force=args.force)
    atomic_write_json(exclude_path, list(result.exclude_paper_ids), force=args.force)
    atomic_write_jsonl(
        dedupe_path,
        [event.model_dump() for event in result.dedupe_log],
        force=args.force,
    )
    config_path = PATHS.config_path(config.question_id)
    record = RunRecord(
        run_id=f"s2-{uuid.uuid4().hex}",
        question_id=config.question_id,
        stage="s2",
        stage_version="1",
        status="complete",
        started_at=started,
        completed_at=datetime.now(UTC),
        code_version=code_version(PATHS.root),
        command_argv=["scripts/s2_screen.py", *(argv if argv is not None else sys.argv[1:])],
        config_path=PATHS.repository_relative(config_path),
        config_sha256=config_hash,
        prompt_path=None,
        prompt_sha256=None,
        schema_path=None,
        schema_sha256=None,
        cfghash=None,
        upstream=[upstream],
        inputs=[
            artifact_ref(config_path, root=PATHS.root),
            artifact_ref(candidate_path, root=PATHS.root, rows=len(occurrences)),
        ],
        outputs=[
            artifact_ref(screened_path, root=PATHS.root, rows=len(result.papers)),
            artifact_ref(include_path, root=PATHS.root, rows=len(result.include_paper_ids)),
            artifact_ref(exclude_path, root=PATHS.root, rows=len(result.exclude_paper_ids)),
            artifact_ref(dedupe_path, root=PATHS.root, rows=len(result.dedupe_log)),
        ],
        external_result_ids={},
        counts={
            "candidate_doc_family_rows": len(occurrences),
            "identity_deduped_papers": len(result.papers),
            "included_papers": len(result.include_paper_ids),
            "excluded_papers": len(result.exclude_paper_ids),
            "dedupe_events": len(result.dedupe_log),
        },
        warnings=[
            "ambiguous_fuzzy_pairs_require_human_disposition"
            for _ in [None]
            if any(event.event == "human_review_required" for event in result.dedupe_log)
        ],
    )
    write_run_record(PATHS.run_record_path(config.question_id, "s2"), record, force=args.force)
    print(
        f"s2 complete: {len(result.papers)} canonical papers "
        f"({len(result.include_paper_ids)} included, {len(result.exclude_paper_ids)} excluded)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
