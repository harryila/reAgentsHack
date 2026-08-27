"""Provider-neutral orchestration and the existing s1 occurrence adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from literature_multiverse.search import SearchOccurrence

from .archive import ArchivedPayload, ImmutableArchive
from .contracts import FullTextFetch, FullTextSource, HarvestDocument, SearchSource


@dataclass(frozen=True, slots=True)
class HarvestQuery:
    family: str
    query: str

    def __post_init__(self) -> None:
        if not self.family.strip() or not self.query.strip():
            raise ValueError("harvest_query_family_and_text_required")


@dataclass(frozen=True, slots=True)
class HarvestResult:
    occurrences: tuple[SearchOccurrence, ...]
    archive_entries: tuple[ArchivedPayload, ...]
    external_result_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    search_pages: int
    documents_with_full_text: int


def _archive_metadata(entry: ArchivedPayload) -> dict[str, Any]:
    return entry.model_dump()


def document_to_occurrence(
    document: HarvestDocument,
    *,
    query_family: str,
    query: str,
    result_id: str,
    search_archive: ArchivedPayload,
    full_text_provenance: dict[str, Any] | None = None,
) -> SearchOccurrence:
    """Adapt a normalized harvested work to the unchanged s1/s2 boundary."""

    raw_metadata = dict(document.raw_metadata)
    if document.abstract is not None:
        raw_metadata.setdefault("abstract", document.abstract)
    raw_metadata["_literature_multiverse_harvester"] = {
        "version": "1",
        "identifiers": dict(document.identifiers),
        "landing_page_url": document.landing_page_url,
        "search": _archive_metadata(search_archive),
        "full_text": full_text_provenance,
    }
    has_full_text = bool(full_text_provenance and full_text_provenance.get("content"))
    content_tier = (
        "full_text" if has_full_text else ("abstract_only" if document.abstract else "unknown")
    )
    return SearchOccurrence(
        doc_id=document.document_id,
        query_family=query_family,
        queries=(query,),
        source=document.source,
        search_result_ids=(result_id,),
        title=document.title,
        doi=document.doi,
        pmid=document.pmid,
        first_author=document.first_author,
        pub_year=document.pub_year,
        article_type=document.article_type,
        content_tier=content_tier,
        publication_status=document.publication_status,
        raw_metadata=raw_metadata,
    )


class LiteratureHarvester:
    """Harvest bounded query families while archiving every received payload."""

    def __init__(
        self,
        search_source: SearchSource,
        archive: ImmutableArchive,
        *,
        full_text_source: FullTextSource | None = None,
        page_size: int = 100,
    ) -> None:
        if not 1 <= page_size <= 200:
            raise ValueError("harvester_page_size_must_be_1_to_200")
        self.search_source = search_source
        self.full_text_source = full_text_source
        self.archive = archive
        self.page_size = page_size

    def run(
        self,
        queries: tuple[HarvestQuery, ...] | list[HarvestQuery],
        *,
        per_query_limit: int,
    ) -> HarvestResult:
        if not queries:
            raise ValueError("harvester_requires_query")
        if per_query_limit < 1:
            raise ValueError("harvester_query_limit_must_be_positive")

        occurrences: list[SearchOccurrence] = []
        entries_by_receipt: dict[str, ArchivedPayload] = {}
        external_ids: set[str] = set()
        warnings: list[str] = []
        full_text_cache: dict[str, dict[str, Any] | None] = {}
        pages = 0
        full_text_documents: set[str] = set()

        for query_spec in queries:
            cursor: str | None = None
            observed_cursors: set[str] = set()
            harvested_for_query = 0
            while harvested_for_query < per_query_limit:
                remaining = per_query_limit - harvested_for_query
                batch = self.search_source.search(
                    query_spec.query,
                    cursor=cursor,
                    limit=min(self.page_size, remaining),
                )
                if batch.source_name != self.search_source.name:
                    raise ValueError("search_batch_source_mismatch")
                if batch.query != query_spec.query or batch.cursor != cursor:
                    raise ValueError("search_batch_request_echo_mismatch")
                if len(batch.documents) > remaining:
                    raise ValueError("search_batch_exceeds_requested_limit")
                pages += 1
                search_entry = self.archive.archive(
                    batch.response,
                    kind="search_response",
                    source_name=batch.source_name,
                    context={
                        "query_family": query_spec.family,
                        "query": query_spec.query,
                        "cursor": cursor,
                    },
                )
                entries_by_receipt[search_entry.receipt_path] = search_entry
                for supporting_payload in batch.supporting_payloads:
                    supporting_entry = self.archive.archive(
                        supporting_payload,
                        kind="search_supporting_input",
                        source_name=batch.source_name,
                        context={"query_family": query_spec.family, "query": query_spec.query},
                    )
                    entries_by_receipt[supporting_entry.receipt_path] = supporting_entry
                result_id = f"{batch.source_name}:{search_entry.sha256[:20]}"
                external_ids.add(result_id)

                for document in batch.documents:
                    full_text_provenance = full_text_cache.get(document.document_id)
                    if document.document_id not in full_text_cache:
                        full_text_provenance = self._fetch_and_archive_full_text(
                            document,
                            entries_by_receipt=entries_by_receipt,
                            warnings=warnings,
                        )
                        full_text_cache[document.document_id] = full_text_provenance
                        if full_text_provenance and full_text_provenance.get("content"):
                            full_text_documents.add(document.document_id)
                    occurrences.append(
                        document_to_occurrence(
                            document,
                            query_family=query_spec.family,
                            query=query_spec.query,
                            result_id=result_id,
                            search_archive=search_entry,
                            full_text_provenance=full_text_provenance,
                        )
                    )
                harvested_for_query += len(batch.documents)

                next_cursor = batch.next_cursor
                if next_cursor is None or harvested_for_query >= per_query_limit:
                    break
                if not batch.documents:
                    warnings.append(
                        f"empty_search_page_with_cursor:{query_spec.family}:{next_cursor}"
                    )
                    break
                if next_cursor in observed_cursors or next_cursor == cursor:
                    raise ValueError("search_cursor_cycle")
                observed_cursors.add(next_cursor)
                cursor = next_cursor

        return HarvestResult(
            occurrences=tuple(occurrences),
            archive_entries=tuple(entries_by_receipt[path] for path in sorted(entries_by_receipt)),
            external_result_ids=tuple(sorted(external_ids)),
            warnings=tuple(sorted(set(warnings))),
            search_pages=pages,
            documents_with_full_text=len(full_text_documents),
        )

    def _fetch_and_archive_full_text(
        self,
        document: HarvestDocument,
        *,
        entries_by_receipt: dict[str, ArchivedPayload],
        warnings: list[str],
    ) -> dict[str, Any] | None:
        if self.full_text_source is None:
            return None
        result: FullTextFetch = self.full_text_source.fetch(document)
        if result.document_id != document.document_id:
            raise ValueError(
                "full_text_document_id_mismatch:"
                f"{document.document_id}:{result.document_id}"
            )
        trace_entries: list[ArchivedPayload] = []
        content_entry: ArchivedPayload | None = None
        for index, payload in enumerate(result.trace):
            entry = self.archive.archive(
                payload,
                kind="full_text_content" if payload is result.content else "full_text_discovery",
                source_name=result.source_name,
                context={"document_id": document.document_id, "trace_index": index},
            )
            entries_by_receipt[entry.receipt_path] = entry
            trace_entries.append(entry)
            if payload is result.content:
                content_entry = entry
        for error in result.errors:
            warnings.append(f"full_text_resolution:{document.document_id}:{error}")
        if not trace_entries and not result.errors:
            return None
        return {
            "resolver": result.source_name,
            "content": None if content_entry is None else _archive_metadata(content_entry),
            "trace": [_archive_metadata(entry) for entry in trace_entries],
            "errors": list(result.errors),
        }


__all__ = [
    "HarvestQuery",
    "HarvestResult",
    "LiteratureHarvester",
    "document_to_occurrence",
]
