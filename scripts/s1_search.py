#!/usr/bin/env python3
"""Run or ingest s1 searches while preserving every doc/query-family occurrence."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from literature_multiverse.config import config_sha256, load_config_for_question
from literature_multiverse.lineage import (
    artifact_ref,
    atomic_write_jsonl,
    code_version,
    write_run_record,
)
from literature_multiverse.live import live_search_to_csv
from literature_multiverse.models import RunRecord
from literature_multiverse.paths import PATHS
from literature_multiverse.search import (
    consolidate_occurrences,
    occurrences_for_query,
    search_result_id,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--question", required=True)
    result.add_argument("--all", action="store_true", dest="use_all")
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--raw-dir", type=Path)
    result.add_argument("--fixture", action="store_true")
    result.add_argument("--force", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    started = datetime.now(UTC)
    config = load_config_for_question(args.question)
    config.authorize_stage("s1", explicit_fixture=args.fixture, live_provider=args.live)
    if not args.use_all or not config.search.use_all:
        raise ValueError("s1_requires_recorded_all_corpus_search")

    destination = PATHS.raw_search_dir(config.question_id)
    all_occurrences = []
    external_ids: list[str] = []
    raw_outputs: list[Path] = []
    raw_inputs: list[Path] = []
    query_count = 0
    for family in config.search.query_families:
        for query_index, query in enumerate(family.queries, start=1):
            query_count += 1
            stem = f"{family.id}-{query_index:02d}"
            if args.live:
                live = live_search_to_csv(
                    query,
                    sources=tuple(config.search.sources),
                    archive_dir=destination,
                    archive_stem=stem,
                    limit=config.search.per_query_limit,
                    use_all=True,
                    exclude_article_types=tuple(config.eligibility.exclude_article_types),
                    force=args.force,
                )
                raw = live.csv_bytes
                fmt = "csv"
                result_id = live.result_id
                raw_outputs.extend(live.artifacts)
            else:
                assert args.raw_dir is not None
                csv_candidate = (args.raw_dir / f"{stem}.results.csv").resolve()
                json_candidate = (args.raw_dir / f"{stem}.json").resolve()
                raw_path = csv_candidate if csv_candidate.exists() else json_candidate
                PATHS.repository_relative(raw_path)
                raw = raw_path.read_bytes()
                fmt = "csv" if raw_path == csv_candidate else "json"
                raw_inputs.append(raw_path)
                result_id = (
                    search_result_id(raw)
                    if fmt == "json"
                    else f"s_archived_{stem}"
                )
            external_ids.append(result_id)
            all_occurrences.extend(
                occurrences_for_query(
                    raw,
                    query_family=family.id,
                    query=query,
                    source=",".join(config.search.sources),
                    search_result_id=result_id,
                    fmt=fmt,
                )
            )

    candidates = consolidate_occurrences(all_occurrences)
    candidate_path = destination / "candidate_papers.jsonl"
    atomic_write_jsonl(
        candidate_path,
        [candidate.model_dump() for candidate in candidates],
        force=args.force,
    )
    config_path = PATHS.config_path(config.question_id)
    outputs = [artifact_ref(candidate_path, root=PATHS.root, rows=len(candidates))]
    outputs.extend(artifact_ref(path, root=PATHS.root) for path in raw_outputs)
    inputs = [artifact_ref(config_path, root=PATHS.root)]
    inputs.extend(artifact_ref(path, root=PATHS.root) for path in raw_inputs)
    record = RunRecord(
        run_id=f"s1-{uuid.uuid4().hex}",
        question_id=config.question_id,
        stage="s1",
        stage_version="1",
        status="complete",
        started_at=started,
        completed_at=datetime.now(UTC),
        code_version=code_version(PATHS.root),
        command_argv=["scripts/s1_search.py", *(argv if argv is not None else sys.argv[1:])],
        config_path=PATHS.repository_relative(config_path),
        config_sha256=config_sha256(config),
        prompt_path=None,
        prompt_sha256=None,
        schema_path=None,
        schema_sha256=None,
        cfghash=None,
        upstream=[],
        inputs=inputs,
        outputs=outputs,
        external_result_ids={"paperclip": sorted(set(external_ids))},
        counts={
            "queries": query_count,
            "raw_occurrences": len(all_occurrences),
            "candidate_doc_family_rows": len(candidates),
            "distinct_documents": len({candidate.doc_id for candidate in candidates}),
        },
        warnings=[] if args.live else ["offline_archived_input"],
    )
    write_run_record(PATHS.run_record_path(config.question_id, "s1"), record, force=args.force)
    print(f"s1 complete: {len(candidates)} document-family candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
