"""OpenAlex, frozen-corpus, and public full-text source adapters."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from literature_multiverse.lineage import canonical_json_bytes

from .contracts import (
    FullTextFetch,
    FullTextSource,
    HarvestDocument,
    RetrievedPayload,
    SearchBatch,
)
from .http import HarvestHttpError, PoliteHttpClient, UnsafeHarvestUrl

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPE_PMC_FULL_TEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
ARXIV_PDF_URL = "https://export.arxiv.org/pdf/{arxiv_id}"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PMCID_RE = re.compile(r"\bPMC\d+\b", re.IGNORECASE)
_ARXIV_URL_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(?P<identifier>(?:[a-z-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?)",
    re.IGNORECASE,
)
_ARXIV_ID_RE = re.compile(r"^(?:[a-z-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?$", re.IGNORECASE)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _normalize_doi(value: object) -> str | None:
    result = _optional_string(value)
    if result is None:
        return None
    lowered = result.casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if lowered.startswith(prefix):
            result = result[len(prefix) :]
            break
    return result.strip().casefold() or None


def _identifier_tail(value: object) -> str | None:
    result = _optional_string(value)
    if result is None:
        return None
    return result.rstrip("/").rsplit("/", 1)[-1] or None


def _abstract_from_inverted_index(value: object) -> str | None:
    if not isinstance(value, Mapping) or not value:
        return None
    positioned: list[tuple[int, str]] = []
    for token, raw_positions in value.items():
        if not isinstance(raw_positions, list):
            continue
        for position in raw_positions:
            if isinstance(position, int) and position >= 0:
                positioned.append((position, str(token)))
    if not positioned:
        return None
    positioned.sort()
    return " ".join(token for _, token in positioned)


def _openalex_article_type(value: object) -> str | None:
    raw = (_optional_string(value) or "").casefold().replace("_", "-")
    return {
        "article": "research-article",
        "review": "review-article",
        "preprint": "preprint",
        "book-chapter": "book-chapter",
        "dataset": "dataset",
    }.get(raw, raw or None)


def _openalex_publication_status(record: Mapping[str, Any]) -> str:
    work_type = str(record.get("type") or "").casefold()
    if work_type == "preprint":
        return "preprint"
    primary = record.get("primary_location")
    if isinstance(primary, Mapping):
        source = primary.get("source")
        if isinstance(source, Mapping):
            source_type = str(source.get("type") or "").casefold()
            if source_type == "journal":
                return "peer_reviewed"
            if source_type == "repository" and work_type == "preprint":
                return "preprint"
    # Work type alone does not establish editorial peer review.  Only retain the
    # stronger label when OpenAlex also identifies a journal source.
    return "unknown"


def document_from_openalex(record: Mapping[str, Any]) -> HarvestDocument:
    """Normalize one OpenAlex work without discarding any source fields."""

    document_id = _identifier_tail(record.get("id"))
    title = _optional_string(record.get("title") or record.get("display_name"))
    if document_id is None or title is None:
        raise ValueError("openalex_work_missing_id_or_title")

    raw_ids = record.get("ids")
    ids = dict(raw_ids) if isinstance(raw_ids, Mapping) else {}
    doi = _normalize_doi(record.get("doi") or ids.get("doi"))
    pmid_tail = _identifier_tail(ids.get("pmid"))
    pmid = pmid_tail if pmid_tail and pmid_tail.isdigit() else None

    authorships = record.get("authorships")
    first_author: str | None = None
    if isinstance(authorships, list) and authorships:
        first = authorships[0]
        if isinstance(first, Mapping):
            author = first.get("author")
            if isinstance(author, Mapping):
                first_author = _optional_string(author.get("display_name"))

    locations: list[Mapping[str, Any]] = []
    for candidate in (
        record.get("best_oa_location"),
        record.get("primary_location"),
        *(record.get("locations") if isinstance(record.get("locations"), list) else []),
    ):
        if isinstance(candidate, Mapping):
            locations.append(candidate)
    full_text_urls: list[str] = []
    landing_page_url: str | None = None
    searchable_urls: list[str] = [str(value) for value in ids.values() if value]
    for location in locations:
        pdf_url = _optional_string(location.get("pdf_url"))
        landing = _optional_string(location.get("landing_page_url"))
        if pdf_url and pdf_url not in full_text_urls:
            full_text_urls.append(pdf_url)
        if landing:
            searchable_urls.append(landing)
            if landing_page_url is None:
                landing_page_url = landing
            if (
                location.get("is_oa") is True
                and (landing.casefold().endswith(".pdf") or "/pdf/" in landing.casefold())
                and landing not in full_text_urls
            ):
                full_text_urls.append(landing)

    joined_urls = " ".join(searchable_urls)
    pmcid_match = _PMCID_RE.search(joined_urls)
    arxiv_match = _ARXIV_URL_RE.search(joined_urls)
    arxiv_id = arxiv_match.group("identifier") if arxiv_match else None
    if arxiv_id is None and doi and doi.startswith("10.48550/arxiv."):
        arxiv_id = doi.removeprefix("10.48550/arxiv.")

    identifiers = {str(key): str(value) for key, value in ids.items() if value is not None}
    if doi:
        identifiers["doi"] = doi
    if pmid:
        identifiers["pmid"] = pmid
    if pmcid_match:
        identifiers["pmcid"] = pmcid_match.group(0).upper()
    if arxiv_id:
        identifiers["arxiv"] = arxiv_id

    year_value = record.get("publication_year")
    pub_year = (
        int(year_value)
        if isinstance(year_value, (int, str)) and str(year_value).isdigit()
        else None
    )
    return HarvestDocument(
        document_id=document_id,
        source="openalex",
        title=title,
        doi=doi,
        pmid=pmid,
        pmcid=pmcid_match.group(0).upper() if pmcid_match else None,
        arxiv_id=arxiv_id,
        first_author=first_author,
        pub_year=pub_year,
        article_type=_openalex_article_type(record.get("type")),
        abstract=_abstract_from_inverted_index(record.get("abstract_inverted_index")),
        publication_status=_openalex_publication_status(record),
        full_text_urls=tuple(full_text_urls),
        landing_page_url=landing_page_url,
        identifiers=identifiers,
        raw_metadata=dict(record),
    )


class OpenAlexSearchSource:
    """Cross-domain OpenAlex Works search using only its public API."""

    name = "openalex"

    def __init__(self, client: PoliteHttpClient) -> None:
        self.client = client

    def search(
        self,
        query: str,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> SearchBatch:
        if not query.strip():
            raise ValueError("openalex_query_missing")
        if not 1 <= limit <= 200:
            raise ValueError("openalex_page_limit_must_be_1_to_200")
        payload = self.client.get(
            OPENALEX_WORKS_URL,
            params={"search": query, "per-page": limit, "cursor": cursor or "*"},
            headers={"Accept": "application/json"},
        )
        try:
            decoded = json.loads(payload.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("openalex_response_invalid_json") from exc
        if not isinstance(decoded, Mapping) or not isinstance(decoded.get("results"), list):
            raise ValueError("openalex_response_missing_results")
        documents = tuple(
            document_from_openalex(record)
            for record in decoded["results"]
            if isinstance(record, Mapping)
        )
        if len(documents) != len(decoded["results"]):
            raise ValueError("openalex_response_result_not_object")
        meta = decoded.get("meta")
        next_cursor = (
            _optional_string(meta.get("next_cursor")) if isinstance(meta, Mapping) else None
        )
        return SearchBatch(
            source_name=self.name,
            query=query,
            cursor=cursor,
            next_cursor=next_cursor,
            documents=documents,
            response=payload,
        )


def _canonical_document(record: Mapping[str, Any], *, default_source: str) -> HarvestDocument:
    if "authorships" in record or str(record.get("id") or "").startswith("https://openalex.org/"):
        return document_from_openalex(record)
    document_id = _optional_string(
        record.get("document_id") or record.get("doc_id") or record.get("id")
    )
    title = _optional_string(record.get("title") or record.get("name"))
    if document_id is None or title is None:
        raise ValueError("frozen_document_missing_id_or_title")
    raw_identifiers = record.get("identifiers")
    identifiers = (
        {str(key): str(value) for key, value in raw_identifiers.items() if value is not None}
        if isinstance(raw_identifiers, Mapping)
        else {}
    )
    raw_urls = record.get("full_text_urls")
    if isinstance(raw_urls, str):
        urls = (raw_urls,)
    elif isinstance(raw_urls, list):
        urls = tuple(dict.fromkeys(str(value) for value in raw_urls if value))
    else:
        urls = ()
    year_value = record.get("pub_year", record.get("publication_year", record.get("year")))
    pub_year = int(year_value) if year_value is not None and str(year_value).isdigit() else None
    publication_status = str(record.get("publication_status") or "unknown")
    raw_metadata = record.get("raw_metadata")
    return HarvestDocument(
        document_id=document_id,
        source=str(record.get("source") or default_source),
        title=title,
        doi=_normalize_doi(record.get("doi") or identifiers.get("doi")),
        pmid=_optional_string(record.get("pmid") or identifiers.get("pmid")),
        pmcid=_optional_string(record.get("pmcid") or identifiers.get("pmcid")),
        arxiv_id=_optional_string(record.get("arxiv_id") or identifiers.get("arxiv")),
        first_author=_optional_string(record.get("first_author")),
        pub_year=pub_year,
        article_type=_optional_string(record.get("article_type") or record.get("type")),
        abstract=_optional_string(record.get("abstract")),
        publication_status=publication_status,
        full_text_urls=urls,
        landing_page_url=_optional_string(record.get("landing_page_url") or record.get("url")),
        identifiers=identifiers,
        raw_metadata=dict(raw_metadata) if isinstance(raw_metadata, Mapping) else dict(record),
    )


class FrozenCorpusSource:
    """Deterministic local JSON/JSONL search with optional exact query replay."""

    name = "frozen"

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str | None = None,
        retrieved_at: datetime | None = None,
    ) -> None:
        self.path = path.resolve()
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and self.sha256 != expected_sha256.casefold():
            raise ValueError(
                f"frozen_corpus_hash_mismatch:expected={expected_sha256}:observed={self.sha256}"
            )
        self._retrieved_at = retrieved_at or datetime.now(UTC)
        if self._retrieved_at.tzinfo is None or self._retrieved_at.utcoffset() is None:
            raise ValueError("frozen_corpus_retrieved_at_requires_timezone")
        self._source_payload = RetrievedPayload(
            url=self.path.as_uri(),
            retrieved_at=self._retrieved_at,
            status_code=200,
            media_type="application/x-ndjson"
            if self.path.suffix == ".jsonl"
            else "application/json",
            body=raw,
        )
        decoded, search_results = self._decode(raw)
        documents = tuple(
            _canonical_document(record, default_source="frozen") for record in decoded
        )
        by_id: dict[str, HarvestDocument] = {}
        for document in documents:
            if document.document_id in by_id:
                raise ValueError(f"frozen_corpus_duplicate_document_id:{document.document_id}")
            by_id[document.document_id] = document
        unknown = {
            document_id
            for ids in search_results.values()
            for document_id in ids
            if document_id not in by_id
        }
        if unknown:
            raise ValueError(f"frozen_search_result_unknown_document:{','.join(sorted(unknown))}")
        self._documents = documents
        self._by_id = by_id
        self._search_results = search_results

    def exact_search_result_ids(self, query: str) -> tuple[str, ...] | None:
        """Return the frozen exhaustive membership for ``query``, when declared."""

        return self._search_results.get(query)

    @staticmethod
    def _decode(raw: bytes) -> tuple[list[Mapping[str, Any]], dict[str, tuple[str, ...]]]:
        text = raw.decode("utf-8")
        try:
            decoded: Any = json.loads(text)
        except json.JSONDecodeError:
            rows: list[Mapping[str, Any]] = []
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"frozen_corpus_invalid_jsonl:line={line_number}") from exc
                if not isinstance(value, Mapping):
                    raise ValueError(f"frozen_corpus_row_not_object:line={line_number}") from None
                rows.append(value)
            return rows, {}
        if isinstance(decoded, list):
            if not all(isinstance(value, Mapping) for value in decoded):
                raise ValueError("frozen_corpus_document_not_object")
            return list(decoded), {}
        if not isinstance(decoded, Mapping):
            raise ValueError("frozen_corpus_root_invalid")
        raw_documents = next(
            (
                decoded[key]
                for key in ("documents", "results", "items")
                if isinstance(decoded.get(key), list)
            ),
            None,
        )
        if raw_documents is None or not all(isinstance(value, Mapping) for value in raw_documents):
            raise ValueError("frozen_corpus_missing_documents")
        raw_searches = decoded.get("search_results", {})
        if not isinstance(raw_searches, Mapping):
            raise ValueError("frozen_search_results_invalid")
        searches: dict[str, tuple[str, ...]] = {}
        for query, ids in raw_searches.items():
            if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
                raise ValueError(f"frozen_search_result_invalid:{query}")
            searches[str(query)] = tuple(ids)
        return list(raw_documents), searches

    def _matching(self, query: str) -> list[HarvestDocument]:
        exact_ids = self._search_results.get(query)
        if exact_ids is not None:
            return [self._by_id[document_id] for document_id in exact_ids]
        terms = set(_TOKEN_RE.findall(query.casefold()))
        if not terms:
            return list(self._documents)
        scored: list[tuple[int, str, HarvestDocument]] = []
        for document in self._documents:
            haystack = set(
                _TOKEN_RE.findall(f"{document.title} {document.abstract or ''}".casefold())
            )
            score = len(terms & haystack)
            if score:
                scored.append((-score, document.document_id, document))
        return [document for _, _, document in sorted(scored)]

    def search(
        self,
        query: str,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> SearchBatch:
        if limit < 1:
            raise ValueError("frozen_page_limit_must_be_positive")
        cursor_prefix = f"offset:{self.sha256[:12]}:"
        if cursor is None:
            offset = 0
        elif cursor.startswith(cursor_prefix) and cursor.removeprefix(cursor_prefix).isdigit():
            offset = int(cursor.removeprefix(cursor_prefix))
        else:
            raise ValueError("frozen_cursor_invalid")
        matches = self._matching(query)
        documents = tuple(matches[offset : offset + limit])
        next_offset = offset + len(documents)
        next_cursor = f"{cursor_prefix}{next_offset}" if next_offset < len(matches) else None
        response_body = canonical_json_bytes(
            {
                "format": "literature-multiverse-frozen-search-v1",
                "corpus_sha256": self.sha256,
                "query": query,
                "offset": offset,
                "limit": limit,
                "result_document_ids": [document.document_id for document in documents],
                "next_cursor": next_cursor,
            }
        )
        response = RetrievedPayload(
            url=f"{self.path.as_uri()}#search-offset-{offset}",
            retrieved_at=self._retrieved_at,
            status_code=200,
            media_type="application/json",
            body=response_body,
        )
        return SearchBatch(
            source_name=self.name,
            query=query,
            cursor=cursor,
            next_cursor=next_cursor,
            documents=documents,
            response=response,
            supporting_payloads=(self._source_payload,),
        )


def _content_looks_usable(payload: RetrievedPayload) -> bool:
    if not payload.body:
        return False
    media_type = (payload.media_type or "").partition(";")[0].casefold()
    if payload.body.startswith(b"%PDF-"):
        return True
    return media_type in {
        "application/pdf",
        "application/xml",
        "application/xhtml+xml",
        "text/html",
        "text/plain",
        "text/xml",
    }


class DirectOpenAccessSource:
    """Fetch provider-declared open PDF/XML/HTML URLs."""

    name = "direct_oa"

    def __init__(self, client: PoliteHttpClient) -> None:
        self.client = client

    def fetch(self, document: HarvestDocument) -> FullTextFetch:
        trace: list[RetrievedPayload] = []
        errors: list[str] = []
        for url in document.full_text_urls:
            try:
                payload = self.client.get(
                    url,
                    headers={
                        "Accept": "application/pdf, application/xml, text/xml, text/html;q=0.8"
                    },
                )
            except (HarvestHttpError, UnsafeHarvestUrl) as exc:
                errors.append(f"direct_oa:{type(exc).__name__}:{url}")
                continue
            trace.append(payload)
            if _content_looks_usable(payload):
                return FullTextFetch(
                    self.name, document.document_id, payload, tuple(trace), tuple(errors)
                )
            errors.append(f"direct_oa:unsupported_content:{url}")
        return FullTextFetch(self.name, document.document_id, None, tuple(trace), tuple(errors))


class EuropePmcFullTextSource:
    """Resolve and fetch Europe PMC open full-text XML without an API key."""

    name = "europe_pmc"

    def __init__(self, client: PoliteHttpClient) -> None:
        self.client = client

    def fetch(self, document: HarvestDocument) -> FullTextFetch:
        trace: list[RetrievedPayload] = []
        errors: list[str] = []
        pmcid = document.pmcid
        if pmcid is None:
            query: str | None = None
            if document.pmid:
                query = f"EXT_ID:{document.pmid} AND SRC:MED"
            elif document.doi:
                query = f'DOI:"{document.doi}"'
            if query is None:
                return FullTextFetch(self.name, document.document_id, None)
            try:
                discovery = self.client.get(
                    EUROPE_PMC_SEARCH_URL,
                    params={"query": query, "format": "json", "pageSize": 1, "resultType": "core"},
                    headers={"Accept": "application/json"},
                )
            except HarvestHttpError as exc:
                return FullTextFetch(
                    self.name,
                    document.document_id,
                    None,
                    errors=(f"europe_pmc:{type(exc).__name__}",),
                )
            trace.append(discovery)
            try:
                decoded = json.loads(discovery.body)
                results = decoded.get("resultList", {}).get("result", [])
                first = results[0] if results else {}
                candidate = first.get("pmcid") if isinstance(first, Mapping) else None
                pmcid = (
                    str(candidate).upper()
                    if candidate and _PMCID_RE.fullmatch(str(candidate))
                    else None
                )
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError):
                errors.append("europe_pmc:discovery_invalid_json")
            if pmcid is None:
                return FullTextFetch(
                    self.name, document.document_id, None, tuple(trace), tuple(errors)
                )
        try:
            content = self.client.get(
                EUROPE_PMC_FULL_TEXT.format(pmcid=pmcid),
                headers={"Accept": "application/xml, text/xml"},
            )
        except HarvestHttpError as exc:
            errors.append(f"europe_pmc:{type(exc).__name__}:{pmcid}")
            return FullTextFetch(self.name, document.document_id, None, tuple(trace), tuple(errors))
        trace.append(content)
        if not _content_looks_usable(content):
            errors.append(f"europe_pmc:unsupported_content:{pmcid}")
            return FullTextFetch(self.name, document.document_id, None, tuple(trace), tuple(errors))
        return FullTextFetch(self.name, document.document_id, content, tuple(trace), tuple(errors))


class ArxivFullTextSource:
    """Fetch arXiv PDFs for works carrying a validated arXiv identifier."""

    name = "arxiv"

    def __init__(self, client: PoliteHttpClient) -> None:
        self.client = client

    def fetch(self, document: HarvestDocument) -> FullTextFetch:
        arxiv_id = document.arxiv_id
        if arxiv_id is None:
            return FullTextFetch(self.name, document.document_id, None)
        if not _ARXIV_ID_RE.fullmatch(arxiv_id):
            return FullTextFetch(
                self.name,
                document.document_id,
                None,
                errors=("arxiv:invalid_identifier",),
            )
        try:
            content = self.client.get(
                ARXIV_PDF_URL.format(arxiv_id=arxiv_id),
                headers={"Accept": "application/pdf"},
            )
        except HarvestHttpError as exc:
            return FullTextFetch(
                self.name,
                document.document_id,
                None,
                errors=(f"arxiv:{type(exc).__name__}:{arxiv_id}",),
            )
        if not _content_looks_usable(content):
            return FullTextFetch(
                self.name,
                document.document_id,
                None,
                trace=(content,),
                errors=("arxiv:unsupported_content",),
            )
        return FullTextFetch(self.name, document.document_id, content, trace=(content,))


class FrozenFullTextSource:
    """Read declared full-text files while preventing corpus-directory escape."""

    name = "frozen_full_text"

    def __init__(
        self,
        corpus_path: Path,
        *,
        retrieved_at: datetime | None = None,
    ) -> None:
        self.root = corpus_path.resolve().parent
        self.retrieved_at = retrieved_at or datetime.now(UTC)
        if self.retrieved_at.tzinfo is None or self.retrieved_at.utcoffset() is None:
            raise ValueError("frozen_full_text_retrieved_at_requires_timezone")

    def fetch(self, document: HarvestDocument) -> FullTextFetch:
        raw_path = document.raw_metadata.get("full_text_path")
        if not isinstance(raw_path, str) or not raw_path:
            return FullTextFetch(self.name, document.document_id, None)
        declared = Path(raw_path)
        if declared.is_absolute():
            return FullTextFetch(
                self.name, document.document_id, None, errors=("frozen:absolute_path_forbidden",)
            )
        path = (self.root / declared).resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            return FullTextFetch(
                self.name, document.document_id, None, errors=("frozen:path_escape",)
            )
        if not path.is_file():
            return FullTextFetch(
                self.name, document.document_id, None, errors=("frozen:file_missing",)
            )
        body = path.read_bytes()
        observed = hashlib.sha256(body).hexdigest()
        expected = document.raw_metadata.get("full_text_sha256")
        if expected is not None and str(expected).casefold() != observed:
            return FullTextFetch(
                self.name, document.document_id, None, errors=("frozen:full_text_hash_mismatch",)
            )
        media_type = _optional_string(document.raw_metadata.get("full_text_media_type"))
        if media_type is None:
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        content = RetrievedPayload(
            url=path.as_uri(),
            retrieved_at=self.retrieved_at,
            status_code=200,
            media_type=media_type,
            body=body,
        )
        if not _content_looks_usable(content):
            return FullTextFetch(
                self.name,
                document.document_id,
                None,
                trace=(content,),
                errors=("frozen:unsupported_content",),
            )
        return FullTextFetch(self.name, document.document_id, content, trace=(content,))


class CompositeFullTextSource:
    """Try resolvers in order while retaining every successful response trace."""

    name = "composite_open_full_text"

    def __init__(self, sources: Sequence[FullTextSource]) -> None:
        if not sources:
            raise ValueError("composite_full_text_requires_source")
        self.sources = tuple(sources)

    def fetch(self, document: HarvestDocument) -> FullTextFetch:
        trace: list[RetrievedPayload] = []
        errors: list[str] = []
        for source in self.sources:
            result = source.fetch(document)
            trace.extend(result.trace)
            errors.extend(result.errors)
            if result.content is not None:
                return FullTextFetch(
                    result.source_name,
                    document.document_id,
                    result.content,
                    tuple(trace),
                    tuple(errors),
                )
        return FullTextFetch(self.name, document.document_id, None, tuple(trace), tuple(errors))


__all__ = [
    "ArxivFullTextSource",
    "CompositeFullTextSource",
    "DirectOpenAccessSource",
    "EuropePmcFullTextSource",
    "FrozenCorpusSource",
    "FrozenFullTextSource",
    "OpenAlexSearchSource",
    "document_from_openalex",
]
