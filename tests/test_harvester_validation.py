"""Offline fixture tests for the harvester live-to-frozen validation."""

from __future__ import annotations

import json
from copy import deepcopy
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
    HARVESTER_VALIDATION_ENTRYPOINT,
    HARVESTER_VALIDATION_SOURCE_PATHS,
    PINNED_PUBLIC_V1_PAYLOAD_SHA256,
    HarvesterValidationError,
    harvester_validation_source_hashes,
    load_harvester_validation_summary,
    reseal_pinned_public_harvester_validation_summary,
    run_harvester_validation_cycle,
    summary_contains_forbidden_text_fields,
)
from literature_multiverse.lineage import hash_canonical, sha256_file

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
    assert summary.harvester_validation_version == "2"
    assert summary.reproducibility.construction == "live_run"
    assert summary.reproducibility.source_files_sha256 == harvester_validation_source_hashes()
    assert (
        summary.reproducibility.generator_entrypoint_sha256
        == summary.reproducibility.source_files_sha256[HARVESTER_VALIDATION_ENTRYPOINT]
    )
    summary_payload = summary.model_dump(mode="json")
    observed_summary_hash = summary_payload.pop("artifact_payload_sha256")
    assert observed_summary_hash == hash_canonical(summary_payload)
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
    assert "--validate-existing" in help_text
    assert "--reseal-pinned-existing" in help_text
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
        artifact_payload_sha256="a" * 64,
        reproducibility=SimpleNamespace(source_bundle_sha256="b" * 64),
    )
    monkeypatch.setattr(
        validate_harvester,
        "run_harvester_validation_cycle",
        lambda **_: failed_invariant_summary,
    )

    assert validate_harvester.main([]) == 2


def test_source_inventory_is_complete_exact_and_current() -> None:
    expected = (
        "pyproject.toml",
        "scripts/validate_harvester.py",
        "src/literature_multiverse/__init__.py",
        "src/literature_multiverse/harvester/__init__.py",
        "src/literature_multiverse/harvester/archive.py",
        "src/literature_multiverse/harvester/contracts.py",
        "src/literature_multiverse/harvester/http.py",
        "src/literature_multiverse/harvester/pipeline.py",
        "src/literature_multiverse/harvester/sources.py",
        "src/literature_multiverse/harvester/validation.py",
        "src/literature_multiverse/lineage.py",
        "src/literature_multiverse/models.py",
        "src/literature_multiverse/paths.py",
        "src/literature_multiverse/search.py",
        "uv.lock",
    )
    assert expected == HARVESTER_VALIDATION_SOURCE_PATHS
    repository_root = Path(__file__).resolve().parents[1]
    assert harvester_validation_source_hashes(repository_root) == {
        relative: sha256_file(repository_root / relative) for relative in expected
    }


def test_strict_loader_rejects_payload_tampering(tmp_path: Path) -> None:
    fixture_corpus = FIXTURES / "frozen" / "corpus.json"
    summary_path = tmp_path / "artifacts" / "summary.json"
    run_harvester_validation_cycle(
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
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["counts"]["live_documents"] += 1
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        HarvesterValidationError,
        match="harvester_validation_summary_contract_invalid",
    ):
        load_harvester_validation_summary(summary_path)


def test_strict_loader_rejects_rehashed_stale_source_lineage(tmp_path: Path) -> None:
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
    payload = summary.model_dump(mode="json")
    source_hashes = payload["reproducibility"]["source_files_sha256"]
    source_hashes["pyproject.toml"] = "f" * 64
    payload["reproducibility"]["source_bundle_sha256"] = hash_canonical(source_hashes)
    payload_without_hash = deepcopy(payload)
    payload_without_hash.pop("artifact_payload_sha256")
    payload["artifact_payload_sha256"] = hash_canonical(payload_without_hash)
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        HarvesterValidationError,
        match="harvester_validation_source_lineage_stale",
    ):
        load_harvester_validation_summary(summary_path)


def test_pinned_public_reseal_is_byte_deterministic(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    public_path = repository_root / "artifacts" / "paper" / "harvester" / "validation_summary.json"
    legacy = json.loads(public_path.read_text(encoding="utf-8"))
    if legacy.get("harvester_validation_version") == "2":
        legacy.pop("artifact_payload_sha256")
        legacy.pop("reproducibility")
        legacy["harvester_validation_version"] = "1"
    assert hash_canonical(legacy) == PINNED_PUBLIC_V1_PAYLOAD_SHA256
    reseal_path = tmp_path / "validation_summary.json"
    reseal_path.write_text(json.dumps(legacy), encoding="utf-8")

    first = reseal_pinned_public_harvester_validation_summary(reseal_path)
    first_bytes = reseal_path.read_bytes()
    first_physical_hash = sha256_file(reseal_path)
    second = reseal_pinned_public_harvester_validation_summary(reseal_path)

    assert first.reproducibility.construction == "pinned_public_v1_reseal"
    assert first.reproducibility.legacy_payload_sha256 == PINNED_PUBLIC_V1_PAYLOAD_SHA256
    assert first == second
    assert reseal_path.read_bytes() == first_bytes
    assert sha256_file(reseal_path) == first_physical_hash
