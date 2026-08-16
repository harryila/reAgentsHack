"""Search-result parsing and lossless query-provenance consolidation for s1."""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class SearchParseError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class SearchOccurrence:
    """One document by query-family s1 candidate before identity dedupe."""

    doc_id: str
    query_family: str
    queries: tuple[str, ...]
    source: str
    search_result_ids: tuple[str, ...]
    title: str
    doi: str | None
    pmid: str | None
    first_author: str | None
    pub_year: int | None
    article_type: str | None
    content_tier: str
    publication_status: str
    raw_metadata: Mapping[str, Any]

    def model_dump(self) -> dict[str, Any]:
        result = asdict(self)
        result["queries"] = list(self.queries)
        result["search_result_ids"] = list(self.search_result_ids)
        return result


def _records_from_json(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        if not all(isinstance(record, Mapping) for record in value):
            raise SearchParseError("SEARCH_RESULTS_INVALID", "result array contains a non-object")
        return list(value)
    if not isinstance(value, Mapping):
        raise SearchParseError("SEARCH_RESULTS_INVALID", "search JSON must be an object or array")
    for key in ("results", "papers", "items", "data", "documents"):
        nested = value.get(key)
        if isinstance(nested, list):
            if not all(isinstance(record, Mapping) for record in nested):
                raise SearchParseError("SEARCH_RESULTS_INVALID", f"{key} contains a non-object")
            return list(nested)
    # A single-document JSON object is also legal.
    if any(key in value for key in ("doc_id", "id", "pmcid")):
        return [value]
    raise SearchParseError("SEARCH_RESULTS_MISSING", "could not locate a result array")


def parse_search_json(raw: str | bytes) -> list[Mapping[str, Any]]:
    """Parse Paperclip search JSON or JSONL without dropping source fields."""

    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SearchParseError("SEARCH_OUTPUT_NOT_UTF8", str(exc)) from exc
    elif isinstance(raw, str):
        text = raw
    else:
        raise TypeError("search output must be str or bytes")
    stripped = text.strip()
    if not stripped:
        return []
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        records: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SearchParseError(
                    "SEARCH_OUTPUT_INVALID_JSON",
                    f"line {line_number}: {exc.msg} at offset {exc.pos}",
                ) from exc
            if not isinstance(item, Mapping):
                raise SearchParseError(
                    "SEARCH_RESULTS_INVALID", f"line {line_number} is not an object"
                ) from None
            records.append(item)
        return records
    return _records_from_json(decoded)


_CSV_REQUIRED_COLUMNS = frozenset({"title", "id"})


def parse_search_csv(raw: str | bytes) -> list[Mapping[str, Any]]:
    """Parse a ``paperclip results <s_id> --save`` CSV export into search records.

    This is the canonical machine-readable search artifact on the installed CLI build,
    which has no ``--json`` search flag.  The CSV columns observed and pinned by the live
    G1b probes are ``title,authors,id,source,date,url,abstract``.
    """

    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SearchParseError("SEARCH_OUTPUT_NOT_UTF8", str(exc)) from exc
    elif isinstance(raw, str):
        text = raw
    else:
        raise TypeError("search CSV must be str or bytes")
    reader = csv.DictReader(io.StringIO(text))
    header = set(reader.fieldnames or ())
    missing = _CSV_REQUIRED_COLUMNS - header
    if missing:
        raise SearchParseError(
            "SEARCH_CSV_HEADER_INVALID", f"missing required columns {sorted(missing)}"
        )
    records: list[Mapping[str, Any]] = []
    for line_number, row in enumerate(reader, start=2):
        doc_id = (row.get("id") or "").strip()
        if not doc_id:
            raise SearchParseError(
                "SEARCH_DOC_ID_MISSING", f"CSV line {line_number} has no document id"
            )
        record: dict[str, Any] = {
            "doc_id": doc_id,
            "title": (row.get("title") or "").strip(),
            "source": (row.get("source") or "").strip() or None,
            "url": (row.get("url") or "").strip() or None,
            "abstract": (row.get("abstract") or "").strip() or None,
            "authors": [
                author.strip()
                for author in (row.get("authors") or "").split(",")
                if author.strip()
            ],
            # The CSV export carries no body-content signal; tiers resolve at s3/s4.
            "content_tier": "unknown",
            "csv_line": line_number,
        }
        date = (row.get("date") or "").strip()
        if len(date) >= 4 and date[:4].isdigit():
            record["pub_year"] = int(date[:4])
            record["pub_date"] = date
        records.append(record)
    return records


def search_result_id(raw: str | bytes) -> str:
    """Extract the saved search identity from a provider JSON response."""

    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(
            r"(?:Found\s+\d+\s+papers?|(?:Search\s+)?results?)\s*\[(s_[0-9A-Za-z]+)\]",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)
        raise SearchParseError("SEARCH_RESULT_ID_MISSING", "search output is not JSON") from exc
    if isinstance(decoded, Mapping):
        for key in ("result_id", "search_result_id", "search_id"):
            value = decoded.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        metadata = decoded.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("result_id", "search_result_id", "search_id"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    raise SearchParseError("SEARCH_RESULT_ID_MISSING", "provider JSON has no saved result ID")


def _nested(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        for key in keys:
            if key in metadata and metadata[key] not in (None, ""):
                return metadata[key]
    return None


def _doc_id(record: Mapping[str, Any]) -> str:
    value = _nested(record, "doc_id", "id", "pmcid", "document_id")
    if value is None or not str(value).strip():
        raise SearchParseError("SEARCH_DOC_ID_MISSING", "search result has no document id")
    return str(value).strip()


def _first_author(record: Mapping[str, Any]) -> str | None:
    direct = _nested(record, "first_author", "author")
    if direct is not None:
        return str(direct).strip() or None
    authors = _nested(record, "authors")
    if isinstance(authors, list) and authors:
        first = authors[0]
        if isinstance(first, Mapping):
            for key in ("name", "full_name", "family", "surname"):
                if first.get(key):
                    return str(first[key]).strip()
        return str(first).strip() or None
    return None


def _year(record: Mapping[str, Any]) -> int | None:
    value = _nested(record, "pub_year", "year", "publication_year")
    if value is None:
        return None
    try:
        year = int(value)
    except (TypeError, ValueError) as exc:
        raise SearchParseError(
            "SEARCH_YEAR_INVALID", f"invalid publication year {value!r}"
        ) from exc
    if year < 1500 or year > 2200:
        raise SearchParseError("SEARCH_YEAR_INVALID", f"publication year out of range: {year}")
    return year


def _content_tier(record: Mapping[str, Any]) -> str:
    explicit = _nested(record, "content_tier")
    if explicit in {"full_text", "abstract_only", "unknown"}:
        return str(explicit)
    if _nested(record, "full_text", "content", "content_lines", "lines"):
        return "full_text"
    if _nested(record, "abstract"):
        return "abstract_only"
    return "unknown"


def _publication_status(record: Mapping[str, Any]) -> str:
    explicit = _nested(record, "publication_status")
    if explicit in {"peer_reviewed", "preprint", "unknown"}:
        return str(explicit)
    source = str(_nested(record, "source", "repository") or "").casefold()
    article_type = str(_nested(record, "article_type", "type") or "").casefold()
    if "preprint" in article_type or any(name in source for name in ("biorxiv", "medrxiv")):
        return "preprint"
    if source == "pmc" or str(_doc_id(record)).upper().startswith("PMC"):
        return "peer_reviewed"
    return "unknown"


def occurrences_for_query(
    raw: str | bytes,
    *,
    query_family: str,
    query: str,
    source: str,
    search_result_id: str,
    fmt: str = "json",
) -> list[SearchOccurrence]:
    """Convert one archived query response into source-preserving occurrences."""

    if fmt == "csv":
        records = parse_search_csv(raw)
    elif fmt == "json":
        records = parse_search_json(raw)
    else:
        raise ValueError(f"unsupported search artifact format: {fmt!r}")
    occurrences: list[SearchOccurrence] = []
    for record in records:
        title = _nested(record, "title", "name")
        if title is None or not str(title).strip():
            raise SearchParseError(
                "SEARCH_TITLE_MISSING", f"search result {_doc_id(record)} has no title"
            )
        record_source = str(_nested(record, "source", "repository") or source)
        occurrences.append(
            SearchOccurrence(
                doc_id=_doc_id(record),
                query_family=query_family,
                queries=(query,),
                source=record_source,
                search_result_ids=(search_result_id,),
                title=str(title).strip(),
                doi=None if _nested(record, "doi") is None else str(_nested(record, "doi")),
                pmid=None if _nested(record, "pmid") is None else str(_nested(record, "pmid")),
                first_author=_first_author(record),
                pub_year=_year(record),
                article_type=(
                    None
                    if _nested(record, "article_type", "type") is None
                    else str(_nested(record, "article_type", "type"))
                ),
                content_tier=_content_tier(record),
                publication_status=_publication_status(record),
                raw_metadata=dict(record),
            )
        )
    return occurrences


def consolidate_occurrences(
    occurrences: Iterable[SearchOccurrence],
) -> list[SearchOccurrence]:
    """Remove only repeated doc/query hits, preserving one row per doc and family."""

    consolidated: dict[tuple[str, str], SearchOccurrence] = {}
    for occurrence in occurrences:
        key = (occurrence.doc_id, occurrence.query_family)
        existing = consolidated.get(key)
        if existing is None:
            consolidated[key] = occurrence
            continue
        # Metadata disagreements remain visible in raw_metadata rather than silently changing
        # identity.  Prefer the first hit and union only the provenance lists.
        raw_metadata = dict(existing.raw_metadata)
        alternates = list(raw_metadata.get("_repeated_query_hits", []))
        alternates.append(dict(occurrence.raw_metadata))
        raw_metadata["_repeated_query_hits"] = alternates
        consolidated[key] = SearchOccurrence(
            doc_id=existing.doc_id,
            query_family=existing.query_family,
            queries=tuple(sorted(set(existing.queries) | set(occurrence.queries))),
            source=existing.source,
            search_result_ids=tuple(
                sorted(set(existing.search_result_ids) | set(occurrence.search_result_ids))
            ),
            title=existing.title,
            doi=existing.doi,
            pmid=existing.pmid,
            first_author=existing.first_author,
            pub_year=existing.pub_year,
            article_type=existing.article_type,
            content_tier=existing.content_tier,
            publication_status=existing.publication_status,
            raw_metadata=raw_metadata,
        )
    return [consolidated[key] for key in sorted(consolidated)]


def load_occurrences_jsonl(path: str | Path) -> list[SearchOccurrence]:
    records: list[SearchOccurrence] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            records.append(
                SearchOccurrence(
                    doc_id=str(value["doc_id"]),
                    query_family=str(value["query_family"]),
                    queries=tuple(str(item) for item in value["queries"]),
                    source=str(value["source"]),
                    search_result_ids=tuple(str(item) for item in value["search_result_ids"]),
                    title=str(value["title"]),
                    doi=value.get("doi"),
                    pmid=value.get("pmid"),
                    first_author=value.get("first_author"),
                    pub_year=value.get("pub_year"),
                    article_type=value.get("article_type"),
                    content_tier=str(value.get("content_tier", "unknown")),
                    publication_status=str(value.get("publication_status", "unknown")),
                    raw_metadata=dict(value.get("raw_metadata", {})),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SearchParseError(
                "SEARCH_CANDIDATE_JSONL_INVALID", f"line {line_number}: {exc}"
            ) from exc
    return records
