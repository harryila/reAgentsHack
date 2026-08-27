"""Offline fixture tests for the harvester live-to-frozen validation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import validate_harvester
from scripts.validate_harvester import build_parser

from literature_multiverse.harvester import (
    FrozenCorpusSource,
    FrozenFullTextSource,
    HarvesterValidationRunFailed,
    HarvestHttpError,
)
from literature_multiverse.harvester.validation import (
    FIXED_OPENALEX_QUERY,
    FIXED_RESULT_LIMIT,
    load_harvester_validation_summary,
    run_harvester_validation_cycle,
    summary_contains_forbidden_text_fields,
)

FIXTURES = Path(__file__).parent / "fixtures" / "harvester"


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for child in value.values() for key in _all_keys(child)}
    if isinstance(value, list):
        return {key for child in value for key in _all_keys(child)}
    return set()


def test_offline_fixture_freezes_replays_and_emits_metadata_only_summary(
    tmp_path: Path,
) -> None:
    fixture_corpus = FIXTURES / "frozen" / "corpus.json"
    summary_path = tmp_path / "artifacts" / "summary.json"
    summary = run_harvester_validation_cycle(
        live_search_source=FrozenCorpusSource(fixture_corpus),
        live_full_text_source=FrozenFullTextSource(fixture_corpus),
        query="training adaptation",
        query_family="offline-fixture",
        result_limit=2,
        live_page_size=2,
        replay_page_size=1,
        source_scope="offline_fixture",
        cache_dir=tmp_path / "data" / "cache" / "validation",
        summary_path=summary_path,
        path_base=tmp_path,
    )

    assert summary.status == "complete"
    assert summary.validation_passed is True
    assert summary.retrieval_recall_evidence is False
    assert summary.metadata_only_summary is True
    assert summary.identity is not None
    assert summary.identity.live_document_ids == ["LOCAL-1", "LOCAL-2"]
    assert summary.identity.replay_document_ids == ["LOCAL-1", "LOCAL-2"]
    assert summary.identity.exact_identity_equivalence is True
    assert summary.counts.live_documents == 2
    assert summary.counts.replay_documents == 2
    assert summary.counts.documents_with_archived_full_text == 1
    assert summary.counts.archive_objects == summary.counts.archive_receipts_verified
    assert summary.archive.all_receipts_verified is True

    by_id = {document.document_id: document for document in summary.documents}
    assert by_id["LOCAL-1"].full_text.status == "archived"
    assert by_id["LOCAL-1"].full_text.media_type == "application/xml"
    assert by_id["LOCAL-1"].full_text.bytes
    assert by_id["LOCAL-1"].full_text.sha256
    assert by_id["LOCAL-2"].full_text.status == "unavailable"
    assert by_id["LOCAL-2"].full_text.bytes is None

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert {"abstract", "title", "body", "full_text_body"}.isdisjoint(_all_keys(payload))
    rendered = json.dumps(payload)
    assert "Training adaptation under heat" not in rendered
    assert "increased adaptation relative to control" not in rendered
    assert summary_contains_forbidden_text_fields(summary) is False
    assert load_harvester_validation_summary(summary_path) == summary

    frozen_corpus = tmp_path / "data" / "cache" / "validation" / "frozen_metadata_corpus.json"
    assert "Training adaptation under heat" in frozen_corpus.read_text(encoding="utf-8")


class _FailingSearchSource:
    name = "offline_failure_fixture"

    def search(
        self,
        query: str,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> None:
        del query, cursor, limit
        raise HarvestHttpError("fixture_transport_failure")


def test_transport_failure_is_preserved_without_text(tmp_path: Path) -> None:
    cache_dir = tmp_path / "data" / "cache" / "failed"
    summary_path = tmp_path / "artifacts" / "failed-summary.json"

    with pytest.raises(HarvesterValidationRunFailed) as caught:
        run_harvester_validation_cycle(
            live_search_source=_FailingSearchSource(),  # type: ignore[arg-type]
            live_full_text_source=None,
            query="fixture query",
            query_family="offline-fixture",
            result_limit=1,
            live_page_size=1,
            replay_page_size=1,
            source_scope="offline_fixture",
            cache_dir=cache_dir,
            summary_path=summary_path,
            path_base=tmp_path,
        )

    summary = caught.value.summary
    assert summary.status == "failed"
    assert summary.validation_passed is False
    assert summary.failure is not None
    assert summary.failure.error_type == "HarvestHttpError"
    assert summary.failure.error_message == "fixture_transport_failure"
    assert summary.counts.live_documents == 0
    assert (cache_dir / "live_failure.json").is_file()
    assert summary_path.is_file()
    assert summary_contains_forbidden_text_fields(summary) is False


def test_live_cli_query_and_limit_are_fixed() -> None:
    parser = build_parser()
    help_text = parser.format_help()

    assert "--query" not in help_text
    assert "--limit" not in help_text
    assert FIXED_OPENALEX_QUERY == "Attention Is All You Need"
    assert FIXED_RESULT_LIMIT == 1


def test_live_cli_returns_nonzero_when_completed_invariants_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_invariant_summary = SimpleNamespace(
        status="complete",
        validation_passed=False,
        retrieval_recall_evidence=False,
        counts=SimpleNamespace(
            live_documents=0,
            documents_with_archived_full_text=0,
        ),
        identity=SimpleNamespace(exact_identity_equivalence=False),
    )
    monkeypatch.setattr(
        validate_harvester,
        "run_harvester_validation_cycle",
        lambda **_: failed_invariant_summary,
    )

    assert validate_harvester.main([]) == 2
