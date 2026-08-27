"""CLI implementation for a Paperclip-free s1 literature harvest."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from literature_multiverse.config import config_sha256, load_config_for_question
from literature_multiverse.lineage import (
    OutputExistsError,
    artifact_ref,
    atomic_write_jsonl,
    code_version,
    write_run_record,
)
from literature_multiverse.models import ArtifactRef, RunRecord
from literature_multiverse.paths import PATHS
from literature_multiverse.search import consolidate_occurrences

from .archive import ImmutableArchive
from .contracts import FullTextSource, SearchSource
from .http import PoliteHttpClient
from .pipeline import HarvestQuery, LiteratureHarvester
from .sources import (
    ArxivFullTextSource,
    CompositeFullTextSource,
    DirectOpenAccessSource,
    EuropePmcFullTextSource,
    FrozenCorpusSource,
    FrozenFullTextSource,
    OpenAlexSearchSource,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--all", action="store_true", dest="use_all")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--openalex", action="store_true")
    mode.add_argument("--frozen-corpus", type=Path)
    parser.add_argument("--frozen-sha256")
    parser.add_argument("--mailto")
    parser.add_argument(
        "--fetch-full-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resolve public OA/Europe PMC/arXiv full text (default: enabled)",
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--min-interval-seconds", type=float, default=0.1)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _unique_artifact_refs(paths: list[Path], *, rows_by_path: dict[Path, int]) -> list[ArtifactRef]:
    unique = {path.resolve(): path for path in paths}
    return [
        artifact_ref(path, root=PATHS.root, rows=rows_by_path.get(path.resolve()))
        for path in sorted(unique.values())
    ]


def _sources(
    args: argparse.Namespace,
    *,
    client: PoliteHttpClient | None,
) -> tuple[SearchSource, FullTextSource | None, Path | None]:
    if args.openalex:
        assert client is not None
        search_source: SearchSource = OpenAlexSearchSource(client)
        full_text_source: FullTextSource | None = None
        if args.fetch_full_text:
            full_text_source = CompositeFullTextSource(
                (
                    DirectOpenAccessSource(client),
                    EuropePmcFullTextSource(client),
                    ArxivFullTextSource(client),
                )
            )
        return search_source, full_text_source, None

    corpus_path = args.frozen_corpus.resolve()
    PATHS.repository_relative(corpus_path)
    search_source = FrozenCorpusSource(corpus_path, expected_sha256=args.frozen_sha256)
    full_text_source = FrozenFullTextSource(corpus_path) if args.fetch_full_text else None
    return search_source, full_text_source, corpus_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.use_all:
        raise ValueError("s1_harvester_requires_all_configured_queries")
    if args.frozen_sha256 and args.openalex:
        raise ValueError("frozen_sha256_requires_frozen_corpus")
    if args.mailto and not args.openalex:
        raise ValueError("mailto_is_only_used_by_openalex")

    started = datetime.now(UTC)
    config = load_config_for_question(args.question)
    config.authorize_stage(
        "s1",
        explicit_fixture=args.fixture,
        live_provider=bool(args.openalex),
    )
    if not config.search.use_all:
        raise ValueError("question_config_must_record_use_all")

    destination = PATHS.raw_search_dir(config.question_id)
    candidate_path = destination / "candidate_papers.jsonl"
    run_path = PATHS.run_record_path(config.question_id, "s1")
    if not args.force:
        existing = next((path for path in (candidate_path, run_path) if path.exists()), None)
        if existing is not None:
            raise OutputExistsError(existing.as_posix())

    needs_http = bool(args.openalex)
    client = (
        PoliteHttpClient(
            contact_email=args.mailto,
            timeout_seconds=args.timeout_seconds,
            max_attempts=args.max_attempts,
            min_interval_seconds=args.min_interval_seconds,
        )
        if needs_http
        else None
    )
    try:
        search_source, full_text_source, frozen_input = _sources(args, client=client)
        archive = ImmutableArchive(destination / "harvest", path_base=PATHS.root)
        harvester = LiteratureHarvester(
            search_source,
            archive,
            full_text_source=full_text_source,
            page_size=args.page_size,
        )
        queries = [
            HarvestQuery(family=family.id, query=query)
            for family in config.search.query_families
            for query in family.queries
        ]
        result = harvester.run(queries, per_query_limit=config.search.per_query_limit)
    finally:
        if client is not None:
            client.close()

    candidates = consolidate_occurrences(result.occurrences)
    atomic_write_jsonl(
        candidate_path,
        [candidate.model_dump() for candidate in candidates],
        force=args.force,
    )
    archive_paths = [
        PATHS.root / path
        for entry in result.archive_entries
        for path in (entry.blob_path, entry.receipt_path)
    ]
    outputs = _unique_artifact_refs(
        [candidate_path, *archive_paths],
        rows_by_path={candidate_path.resolve(): len(candidates)},
    )
    config_path = PATHS.config_path(config.question_id)
    input_paths = [config_path]
    if frozen_input is not None:
        input_paths.append(frozen_input)
    inputs = _unique_artifact_refs(input_paths, rows_by_path={})
    source_name = "openalex" if args.openalex else "frozen"
    warnings = [
        f"paperclip_free_harvester:{source_name}",
        *result.warnings,
    ]
    if not args.fetch_full_text:
        warnings.append("full_text_resolution_disabled")
    record = RunRecord(
        run_id=f"s1-harvest-{uuid.uuid4().hex}",
        question_id=config.question_id,
        stage="s1",
        stage_version="2",
        status="complete",
        started_at=started,
        completed_at=datetime.now(UTC),
        code_version=code_version(PATHS.root),
        command_argv=["scripts/s1_harvest.py", *(argv if argv is not None else sys.argv[1:])],
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
        external_result_ids={source_name: list(result.external_result_ids)},
        counts={
            "queries": len(queries),
            "search_pages": result.search_pages,
            "raw_occurrences": len(result.occurrences),
            "candidate_doc_family_rows": len(candidates),
            "distinct_documents": len({candidate.doc_id for candidate in candidates}),
            "documents_with_full_text": result.documents_with_full_text,
            "immutable_archive_objects": len(result.archive_entries),
        },
        warnings=sorted(set(warnings)),
    )
    write_run_record(run_path, record, force=args.force)
    print(
        f"s1 harvest complete: {len(candidates)} document-family candidates; "
        f"{result.documents_with_full_text} documents with archived full text"
    )
    return 0


__all__ = ["build_parser", "main"]
