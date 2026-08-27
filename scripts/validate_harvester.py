#!/usr/bin/env python3
"""Run the one-result OpenAlex live-to-frozen harvester validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from literature_multiverse.harvester import (
    ArxivFullTextSource,
    CompositeFullTextSource,
    DirectOpenAccessSource,
    EuropePmcFullTextSource,
    HarvesterValidationRunFailed,
    OpenAlexSearchSource,
    PoliteHttpClient,
)
from literature_multiverse.harvester.validation import (
    FIXED_LIVE_PAGE_SIZE,
    FIXED_OPENALEX_QUERY,
    FIXED_QUERY_FAMILY,
    FIXED_REPLAY_PAGE_SIZE,
    FIXED_RESULT_LIMIT,
    run_harvester_validation_cycle,
)
from literature_multiverse.paths import PATHS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PATHS.data_dir / "cache" / "harvester-openalex-validation-v1",
        help="Gitignored raw archive and frozen-corpus destination.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PATHS.artifacts_dir / "paper" / "harvester" / "validation_summary.json",
        help="Metadata-only paper artifact.",
    )
    parser.add_argument("--mailto")
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--min-interval-seconds", type=float, default=0.1)
    return parser


def _validate_destinations(cache_dir: Path, summary_path: Path) -> tuple[Path, Path]:
    cache = cache_dir.resolve()
    summary = summary_path.resolve()
    try:
        cache.relative_to((PATHS.data_dir / "cache").resolve())
    except ValueError as exc:
        raise ValueError("harvester_validation_cache_must_be_under_data_cache") from exc
    try:
        summary.relative_to((PATHS.artifacts_dir / "paper" / "harvester").resolve())
    except ValueError as exc:
        raise ValueError("harvester_validation_summary_must_be_under_paper_harvester") from exc
    return cache, summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_dir, summary_path = _validate_destinations(args.cache_dir, args.summary)
    with PoliteHttpClient(
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        min_interval_seconds=args.min_interval_seconds,
        max_response_bytes=25_000_000,
    ) as client:
        search_source = OpenAlexSearchSource(client)
        full_text_source = CompositeFullTextSource(
            (
                DirectOpenAccessSource(client),
                EuropePmcFullTextSource(client),
                ArxivFullTextSource(client),
            )
        )
        try:
            summary = run_harvester_validation_cycle(
                live_search_source=search_source,
                live_full_text_source=full_text_source,
                query=FIXED_OPENALEX_QUERY,
                query_family=FIXED_QUERY_FAMILY,
                result_limit=FIXED_RESULT_LIMIT,
                live_page_size=FIXED_LIVE_PAGE_SIZE,
                replay_page_size=FIXED_REPLAY_PAGE_SIZE,
                source_scope="cross_domain_openalex_index_fixed_cs_probe",
                cache_dir=cache_dir,
                summary_path=summary_path,
                path_base=PATHS.root,
            )
        except HarvesterValidationRunFailed as exc:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "summary": summary_path.as_posix(),
                        "failure": exc.summary.failure.model_dump(mode="json")
                        if exc.summary.failure
                        else None,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 1
    print(
        json.dumps(
            {
                "status": summary.status,
                "validation_passed": summary.validation_passed,
                "summary": summary_path.as_posix(),
                "live_documents": summary.counts.live_documents,
                "documents_with_archived_full_text": (
                    summary.counts.documents_with_archived_full_text
                ),
                "exact_identity_equivalence": (
                    summary.identity.exact_identity_equivalence if summary.identity else False
                ),
                "retrieval_recall_evidence": summary.retrieval_recall_evidence,
            },
            sort_keys=True,
        )
    )
    return 0 if summary.validation_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
