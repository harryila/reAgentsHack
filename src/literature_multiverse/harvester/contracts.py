"""Source-agnostic contracts for open literature retrieval.

The harvester deliberately separates provider records, search pagination, and full-text
resolution.  Provider adapters can therefore be replaced without changing the s1/s2
``SearchOccurrence`` boundary used by the rest of Literature Multiverse.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


def _require_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("retrieval_timestamp_requires_timezone")


@dataclass(frozen=True, slots=True)
class HarvestDocument:
    """Normalized provider document retaining its complete source record."""

    document_id: str
    source: str
    title: str
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    arxiv_id: str | None = None
    first_author: str | None = None
    pub_year: int | None = None
    article_type: str | None = None
    abstract: str | None = None
    publication_status: str = "unknown"
    full_text_urls: tuple[str, ...] = ()
    landing_page_url: str | None = None
    identifiers: Mapping[str, str] = field(default_factory=dict)
    raw_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("harvest_document_id_missing")
        if not self.source.strip():
            raise ValueError("harvest_document_source_missing")
        if not self.title.strip():
            raise ValueError("harvest_document_title_missing")
        if self.pub_year is not None and not 1000 <= self.pub_year <= 3000:
            raise ValueError("harvest_document_year_invalid")
        if self.publication_status not in {"peer_reviewed", "preprint", "unknown"}:
            raise ValueError("harvest_document_publication_status_invalid")
        if len(self.full_text_urls) != len(set(self.full_text_urls)):
            raise ValueError("harvest_document_full_text_urls_duplicate")


@dataclass(frozen=True, slots=True)
class RetrievedPayload:
    """One exact HTTP or local payload, before any provider-specific parsing."""

    url: str
    retrieved_at: datetime
    status_code: int
    media_type: str | None
    body: bytes
    response_headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_aware(self.retrieved_at)
        if not self.url:
            raise ValueError("retrieved_payload_url_missing")
        if not 100 <= self.status_code <= 599:
            raise ValueError("retrieved_payload_status_invalid")


@dataclass(frozen=True, slots=True)
class SearchBatch:
    """A single provider page plus the exact response used to construct it."""

    source_name: str
    query: str
    cursor: str | None
    next_cursor: str | None
    documents: tuple[HarvestDocument, ...]
    response: RetrievedPayload
    supporting_payloads: tuple[RetrievedPayload, ...] = ()


@dataclass(frozen=True, slots=True)
class FullTextFetch:
    """A full-text resolution attempt, including discovery-response provenance."""

    source_name: str
    document_id: str
    content: RetrievedPayload | None
    trace: tuple[RetrievedPayload, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.content is not None and all(payload is not self.content for payload in self.trace):
            raise ValueError("full_text_content_missing_from_trace")


@runtime_checkable
class SearchSource(Protocol):
    """A paginated literature-index adapter."""

    @property
    def name(self) -> str: ...

    def search(
        self,
        query: str,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> SearchBatch: ...


@runtime_checkable
class FullTextSource(Protocol):
    """A resolver that may return open full text for a normalized document."""

    @property
    def name(self) -> str: ...

    def fetch(self, document: HarvestDocument) -> FullTextFetch: ...


__all__ = [
    "FullTextFetch",
    "FullTextSource",
    "HarvestDocument",
    "RetrievedPayload",
    "SearchBatch",
    "SearchSource",
]
