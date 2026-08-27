"""Offline tests for the source-agnostic open literature harvester."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

import literature_multiverse.config as config_module
import literature_multiverse.harvester.cli as harvester_cli
from literature_multiverse.harvester import (
    ArchiveIntegrityError,
    ArxivFullTextSource,
    EuropePmcFullTextSource,
    FrozenCorpusSource,
    FrozenFullTextSource,
    FullTextFetch,
    FullTextSource,
    HarvestDocument,
    HarvestQuery,
    ImmutableArchive,
    LiteratureHarvester,
    OpenAlexSearchSource,
    PoliteHttpClient,
    RetrievedPayload,
    SearchSource,
    UnsafeHarvestUrl,
    document_from_openalex,
)
from literature_multiverse.models import RunRecord
from literature_multiverse.paths import ProjectPaths
from literature_multiverse.screen import screen_candidates
from literature_multiverse.search import consolidate_occurrences, load_occurrences_jsonl

FIXTURES = Path(__file__).parent / "fixtures" / "harvester"


def _mock_client(handler: httpx.MockTransport) -> PoliteHttpClient:
    return PoliteHttpClient(
        transport=handler,
        min_interval_seconds=0,
        sleeper=lambda _: None,
    )


def test_http_client_ignores_ambient_proxy_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class DummyClient:
        def close(self) -> None:
            return None

    def construct_client(**kwargs: object) -> DummyClient:
        captured.update(kwargs)
        return DummyClient()

    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:9")
    monkeypatch.setattr(httpx, "Client", construct_client)
    client = PoliteHttpClient()
    client.close()

    assert captured["trust_env"] is False


def test_openalex_cross_domain_page_normalizes_identifiers_and_abstract() -> None:
    raw = (FIXTURES / "openalex_page.json").read_bytes()

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.openalex.org"
        assert request.url.params["search"] == "training adaptation"
        assert request.url.params["cursor"] == "*"
        return httpx.Response(200, content=raw, headers={"Content-Type": "application/json"})

    with _mock_client(httpx.MockTransport(respond)) as client:
        source = OpenAlexSearchSource(client)
        assert isinstance(source, SearchSource)
        batch = source.search("training adaptation", limit=2)

    assert len(batch.documents) == 2
    journal, preprint = batch.documents
    assert journal.document_id == "W1111111111"
    assert journal.doi == "10.1000/example.1"
    assert journal.pmid == "12345678"
    assert journal.pmcid == "PMC1234567"
    assert journal.abstract == "Conditions moderate adaptation"
    assert journal.article_type == "research-article"
    assert journal.publication_status == "peer_reviewed"
    assert journal.full_text_urls == ("https://example.org/articles/1.pdf",)
    assert preprint.arxiv_id == "2401.12345"
    assert preprint.publication_status == "preprint"


def test_openalex_article_without_journal_evidence_has_unknown_review_status() -> None:
    document = document_from_openalex(
        {
            "id": "https://openalex.org/W123",
            "title": "Unlocated article",
            "type": "article",
            "primary_location": None,
        }
    )

    assert document.publication_status == "unknown"


def test_europe_pmc_and_arxiv_resolvers_need_no_credentials() -> None:
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.path.endswith("/rest/search"):
            return httpx.Response(
                200,
                json={"resultList": {"result": [{"pmcid": "PMC7654321"}]}},
                headers={"Content-Type": "application/json"},
            )
        if request.url.path.endswith("/PMC7654321/fullTextXML"):
            return httpx.Response(
                200,
                content=b"<article><body>open text</body></article>",
                headers={"Content-Type": "application/xml"},
            )
        if request.url.path.endswith("/pdf/2401.12345"):
            return httpx.Response(
                200,
                content=b"%PDF-1.7\nfixture",
                headers={"Content-Type": "application/pdf"},
            )
        raise AssertionError(f"unexpected URL {request.url}")

    with _mock_client(httpx.MockTransport(respond)) as client:
        europe = EuropePmcFullTextSource(client)
        arxiv = ArxivFullTextSource(client)
        assert isinstance(europe, FullTextSource)
        epmc_result = europe.fetch(
            HarvestDocument(
                document_id="D1",
                source="fixture",
                title="Europe PMC paper",
                doi="10.1000/example",
            )
        )
        arxiv_result = arxiv.fetch(
            HarvestDocument(
                document_id="D2",
                source="fixture",
                title="arXiv paper",
                arxiv_id="2401.12345",
            )
        )

    assert epmc_result.content is not None
    assert epmc_result.content.body.startswith(b"<article>")
    assert len(epmc_result.trace) == 2
    assert arxiv_result.content is not None
    assert arxiv_result.content.body.startswith(b"%PDF-")
    assert all("authorization" not in url.casefold() for url in requests)


def test_frozen_corpus_paginates_replays_exact_query_and_reads_local_full_text() -> None:
    corpus_path = FIXTURES / "frozen" / "corpus.json"
    expected = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    source = FrozenCorpusSource(corpus_path, expected_sha256=expected)
    first = source.search("training adaptation", limit=1)
    assert [document.document_id for document in first.documents] == ["LOCAL-1"]
    assert first.next_cursor is not None
    second = source.search("training adaptation", cursor=first.next_cursor, limit=1)
    assert [document.document_id for document in second.documents] == ["LOCAL-2"]
    assert second.next_cursor is None
    assert first.supporting_payloads[0].body == corpus_path.read_bytes()

    resolver = FrozenFullTextSource(corpus_path)
    fetched = resolver.fetch(first.documents[0])
    assert fetched.content is not None
    assert b"increased adaptation" in fetched.content.body
    with pytest.raises(ValueError, match="frozen_corpus_hash_mismatch"):
        FrozenCorpusSource(corpus_path, expected_sha256="0" * 64)


def test_harvester_archives_provenance_and_feeds_existing_paper_record_flow(
    tmp_path: Path,
) -> None:
    corpus_path = FIXTURES / "frozen" / "corpus.json"
    source = FrozenCorpusSource(corpus_path)
    archive = ImmutableArchive(tmp_path / "archive", path_base=tmp_path)
    harvester = LiteratureHarvester(
        source,
        archive,
        full_text_source=FrozenFullTextSource(corpus_path),
        page_size=1,
    )
    result = harvester.run(
        [HarvestQuery(family="direct", query="training adaptation")],
        per_query_limit=2,
    )
    candidates = consolidate_occurrences(result.occurrences)

    assert result.search_pages == 2
    assert result.documents_with_full_text == 1
    assert len(candidates) == 2
    by_id = {candidate.doc_id: candidate for candidate in candidates}
    assert by_id["LOCAL-1"].content_tier == "full_text"
    provenance = by_id["LOCAL-1"].raw_metadata["_literature_multiverse_harvester"]
    content = provenance["full_text"]["content"]
    assert len(content["sha256"]) == 64
    assert (tmp_path / content["blob_path"]).is_file()
    assert by_id["LOCAL-2"].content_tier == "abstract_only"

    screened = screen_candidates(
        candidates,
        allowed_article_types=["research-article"],
        config_sha256="a" * 64,
        created_at=datetime(2026, 8, 26, tzinfo=UTC),
    )
    assert len(screened.papers) == 2
    assert all(paper["screen_status"] == "included" for paper in screened.papers)
    assert {paper["paper_id"] for paper in screened.papers} == {
        "doi:10.5555/local.1",
        "doc:LOCAL-2",
    }
    for entry in result.archive_entries:
        archive.verify(entry)


def test_harvester_rejects_full_text_for_a_different_document(tmp_path: Path) -> None:
    class MismatchedResolver:
        name = "mismatched"

        def fetch(self, document: HarvestDocument) -> FullTextFetch:
            return FullTextFetch(self.name, "WRONG-DOCUMENT", None)

    corpus_path = FIXTURES / "frozen" / "corpus.json"
    harvester = LiteratureHarvester(
        FrozenCorpusSource(corpus_path),
        ImmutableArchive(tmp_path / "archive", path_base=tmp_path),
        full_text_source=MismatchedResolver(),
    )

    with pytest.raises(ValueError, match="full_text_document_id_mismatch"):
        harvester.run(
            [HarvestQuery(family="direct", query="training adaptation")],
            per_query_limit=1,
        )


def test_immutable_archive_is_idempotent_and_detects_mutation(tmp_path: Path) -> None:
    archive = ImmutableArchive(tmp_path / "archive", path_base=tmp_path)
    payload = RetrievedPayload(
        url="https://example.org/raw.json",
        retrieved_at=datetime(2026, 8, 26, tzinfo=UTC),
        status_code=200,
        media_type="application/json",
        body=b'{"result":1}',
    )
    first = archive.archive(payload, kind="search_response", source_name="fixture")
    second = archive.archive(payload, kind="search_response", source_name="fixture")
    assert first == second
    archive.verify(first)

    receipt_path = tmp_path / first.receipt_path
    receipt_path.chmod(0o644)
    receipt_path.write_text(
        receipt_path.read_text(encoding="utf-8").replace(
            '"source":"fixture"', '"source":"changed"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ArchiveIntegrityError, match="archive_receipt_hash_mismatch"):
        archive.verify(first)

    blob_path = tmp_path / first.blob_path
    blob_path.chmod(0o644)
    blob_path.write_bytes(b"mutated")
    with pytest.raises(ArchiveIntegrityError, match="archive_blob_hash_mismatch"):
        archive.verify(first)


def test_http_transport_retries_retry_after_and_rejects_private_urls() -> None:
    attempts = 0
    sleeps: list[float] = []

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, text="ok", headers={"Content-Type": "text/plain"})

    with PoliteHttpClient(
        transport=httpx.MockTransport(respond),
        min_interval_seconds=0,
        sleeper=sleeps.append,
    ) as client:
        payload = client.get("https://api.example.org/resource")
        with pytest.raises(UnsafeHarvestUrl):
            client.get("http://127.0.0.1/private")
        with pytest.raises(UnsafeHarvestUrl, match="embedded_credentials"):
            client.get("https://user:secret@example.org/private")

    assert payload.body == b"ok"
    assert attempts == 2
    assert sleeps == [2.0]


def test_frozen_full_text_rejects_path_escape(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus" / "corpus.json"
    corpus.parent.mkdir()
    corpus.write_text(json.dumps({"documents": []}), encoding="utf-8")
    document = HarvestDocument(
        document_id="escape",
        source="fixture",
        title="Unsafe path",
        raw_metadata={"full_text_path": "../outside.xml"},
    )
    result = FrozenFullTextSource(corpus).fetch(document)
    assert result.content is None
    assert result.errors == ("frozen:path_escape",)


def test_cli_materializes_drop_in_s1_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = ProjectPaths(tmp_path)
    config_path = paths.config_path("fixture-a")
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(
        (Path(__file__).parents[1] / "configs" / "questions" / "fixture-a.yaml").read_bytes()
    )
    corpus_path = tmp_path / "frozen-corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "DROP-IN-1",
                        "title": "Synthetic intervention performance trial",
                        "source": "frozen-cli-test",
                        "article_type": "research-article",
                        "publication_status": "peer_reviewed",
                        "abstract": "The synthetic intervention changed performance.",
                    }
                ],
                "search_results": {
                    "synthetic intervention AND performance": ["DROP-IN-1"],
                    "synthetic intervention AND (null OR decrease)": ["DROP-IN-1"],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "PATHS", paths)
    monkeypatch.setattr(harvester_cli, "PATHS", paths)

    exit_code = harvester_cli.main(
        [
            "--question",
            "fixture-a",
            "--all",
            "--frozen-corpus",
            str(corpus_path),
            "--fixture",
            "--no-fetch-full-text",
        ]
    )

    assert exit_code == 0
    candidate_path = paths.raw_search_dir("fixture-a") / "candidate_papers.jsonl"
    candidates = load_occurrences_jsonl(candidate_path)
    assert len(candidates) == 2
    assert {candidate.query_family for candidate in candidates} == {"direct", "null-negative"}
    run = RunRecord.model_validate_json(paths.run_record_path("fixture-a", "s1").read_text())
    assert run.stage == "s1"
    assert run.stage_version == "2"
    assert run.counts["distinct_documents"] == 1
    assert run.external_result_ids["frozen"]
    assert all((tmp_path / output.path).is_file() for output in run.outputs)
