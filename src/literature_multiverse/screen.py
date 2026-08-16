"""Deterministic s2 filtering, identity dedupe, and canonical paper-ledger creation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

from rapidfuzz.fuzz import ratio

from .search import SearchOccurrence

FUZZY_AUTO_MERGE_THRESHOLD = 96.0
FUZZY_AMBIGUOUS_THRESHOLD = 90.0

_DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
_TITLE_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_TRIAL_PATTERNS = (
    re.compile(r"\bNCT\d{8}\b", re.IGNORECASE),
    re.compile(r"\bISRCTN\d{8}\b", re.IGNORECASE),
    re.compile(r"\bACTRN\d{14}\b", re.IGNORECASE),
    re.compile(r"\b(?:UMIN|JPRN)-?\w[\w-]{4,}\b", re.IGNORECASE),
)


class ScreeningError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class DocumentCandidate:
    doc_id: str
    title: str
    doi: str | None
    pmid: str | None
    first_author: str | None
    pub_year: int | None
    source: str
    article_type: str | None
    query_families: tuple[str, ...]
    search_result_ids: tuple[str, ...]
    content_tier: str
    publication_status: str
    raw_metadata: tuple[Mapping[str, Any], ...]
    deterministic_included: bool
    screen_reason: str | None


@dataclass(frozen=True, slots=True)
class DedupeEvent:
    event: str
    cluster_id: str | None
    preferred_doc_id: str | None
    member_doc_ids: tuple[str, ...]
    identity_key: str | None
    title_score: float | None
    reason: str

    def model_dump(self) -> dict[str, Any]:
        result = asdict(self)
        result["member_doc_ids"] = list(self.member_doc_ids)
        return result


@dataclass(frozen=True, slots=True)
class ScreenResult:
    papers: tuple[Mapping[str, Any], ...]
    include_paper_ids: tuple[str, ...]
    exclude_paper_ids: tuple[str, ...]
    dedupe_log: tuple[DedupeEvent, ...]


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        preferred_root, other_root = sorted((left_root, right_root))
        self.parent[other_root] = preferred_root


def normalize_doi(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _DOI_PREFIX_RE.sub("", value.strip()).strip().rstrip(".").casefold()
    if not normalized:
        return None
    if "/" not in normalized:
        raise ScreeningError("SCREEN_DOI_INVALID", f"invalid DOI {value!r}")
    return normalized


def normalize_pmid(value: str | None) -> str | None:
    if value is None:
        return None
    digits = "".join(character for character in str(value) if character.isdigit())
    return digits or None


def normalize_title(value: str) -> str:
    normalized = _TITLE_TOKEN_RE.sub(" ", value.casefold())
    return " ".join(normalized.split())


def author_surname(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"[^A-Za-z' -]", " ", value).strip()
    if not cleaned:
        return None
    surname = value.split(",", 1)[0] if "," in value else cleaned.split()[-1]
    return surname.casefold().strip("-' ") or None


def derive_paper_id(*, doi: str | None, pmid: str | None, doc_id: str) -> str:
    normalized_doi = normalize_doi(doi)
    if normalized_doi:
        return f"doi:{normalized_doi}"
    normalized_pmid = normalize_pmid(pmid)
    if normalized_pmid:
        return f"pmid:{normalized_pmid}"
    if not doc_id.strip():
        raise ScreeningError("SCREEN_DOC_ID_MISSING", "cannot derive paper_id without identity")
    return f"doc:{doc_id.strip()}"


def _candidate_dict(value: SearchOccurrence | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, SearchOccurrence):
        return value.model_dump()
    return dict(value)


def _article_allowed(article_type: str | None, allowed: set[str]) -> tuple[bool, str | None]:
    if not allowed:
        return True, None
    if article_type is None:
        # The installed CLI's search/CSV export carries no article-type metadata, so a
        # missing type is *unconfirmed*, not disqualifying: retrieval already excludes
        # confirmed mismatches (`--exclude-article-type`), and extraction remains the
        # authoritative eligibility judgment.  The pass-through stays visible in the
        # ledger via the recorded reason.
        return True, "article_type_unconfirmed"
    normalized = "-".join(article_type.strip().casefold().replace("_", "-").split())
    if normalized not in allowed:
        return False, "article_type_not_allowed"
    return True, None


def consolidate_documents(
    occurrences: Sequence[SearchOccurrence | Mapping[str, Any]],
    *,
    allowed_article_types: Sequence[str],
) -> list[DocumentCandidate]:
    """Union s1 family occurrences to one record per provider document ID."""

    allowed = {
        "-".join(value.strip().casefold().replace("_", "-").split())
        for value in allowed_article_types
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for occurrence in occurrences:
        row = _candidate_dict(occurrence)
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id:
            raise ScreeningError("SCREEN_DOC_ID_MISSING", "s1 occurrence has no doc_id")
        grouped.setdefault(doc_id, []).append(row)

    documents: list[DocumentCandidate] = []
    for doc_id in sorted(grouped):
        rows = grouped[doc_id]
        first = rows[0]
        titles = {str(row.get("title", "")).strip() for row in rows}
        if "" in titles:
            raise ScreeningError("SCREEN_TITLE_MISSING", f"{doc_id} has no title")
        if len({normalize_title(title) for title in titles}) != 1:
            raise ScreeningError(
                "SCREEN_DOC_METADATA_CONFLICT", f"{doc_id} has conflicting titles across queries"
            )
        article_type = first.get("article_type")
        included, reason = _article_allowed(
            None if article_type is None else str(article_type), allowed
        )
        query_families = sorted(
            {str(row.get("query_family")) for row in rows if row.get("query_family") is not None}
        )
        result_ids = sorted(
            {str(result_id) for row in rows for result_id in row.get("search_result_ids", [])}
        )
        documents.append(
            DocumentCandidate(
                doc_id=doc_id,
                title=str(first["title"]).strip(),
                doi=normalize_doi(first.get("doi")),
                pmid=normalize_pmid(first.get("pmid")),
                first_author=first.get("first_author"),
                pub_year=None if first.get("pub_year") is None else int(first["pub_year"]),
                source=str(first.get("source", "paperclip")),
                article_type=None if article_type is None else str(article_type),
                query_families=tuple(query_families),
                search_result_ids=tuple(result_ids),
                content_tier=str(first.get("content_tier", "unknown")),
                publication_status=str(first.get("publication_status", "unknown")),
                raw_metadata=tuple(dict(row.get("raw_metadata", {})) for row in rows),
                deterministic_included=included,
                screen_reason=reason,
            )
        )
    return documents


def _same_fuzzy_identity(left: DocumentCandidate, right: DocumentCandidate) -> bool:
    left_surname = author_surname(left.first_author)
    right_surname = author_surname(right.first_author)
    if left_surname is None or right_surname is None or left_surname != right_surname:
        return False
    return not (
        left.pub_year is None or right.pub_year is None or abs(left.pub_year - right.pub_year) > 1
    )


def _preferred_key(document: DocumentCandidate) -> tuple[int, int, int, int, str]:
    status_rank = {"peer_reviewed": 0, "unknown": 1, "preprint": 2}.get(
        document.publication_status, 1
    )
    return (
        0 if document.deterministic_included else 1,
        status_rank,
        0 if document.content_tier == "full_text" else 1,
        0 if (document.doi or document.pmid) else 1,
        document.doc_id,
    )


def _cluster_id(doc_ids: Sequence[str]) -> str:
    digest = hashlib.sha256("|".join(sorted(doc_ids)).encode()).hexdigest()[:12]
    return f"dedupe:{digest}"


def infer_dataset_or_cohort_id(documents: Sequence[DocumentCandidate]) -> str | None:
    text = json.dumps(
        [
            {
                "title": document.title,
                "metadata": document.raw_metadata,
            }
            for document in documents
        ],
        sort_keys=True,
        default=str,
    )
    matches = sorted(
        {match.group(0).upper() for pattern in _TRIAL_PATTERNS for match in pattern.finditer(text)}
    )
    return matches[0] if len(matches) == 1 else None


def screen_candidates(
    occurrences: Sequence[SearchOccurrence | Mapping[str, Any]],
    *,
    allowed_article_types: Sequence[str],
    config_sha256: str,
    schema_version: str = "1",
    created_at: str | datetime | None = None,
    audit_excluded_doc_ids: Mapping[str, str] | None = None,
) -> ScreenResult:
    """Create one canonical screened PaperRecord-shaped row per identity cluster."""

    documents = consolidate_documents(occurrences, allowed_article_types=allowed_article_types)
    if audit_excluded_doc_ids:
        # Papers the frozen human audit excluded at full-text level (config-recorded,
        # PRISMA-style).  The exclusion is deterministic screen state, not model output.
        documents = [
            document
            if document.doc_id not in audit_excluded_doc_ids
            else replace(
                document,
                deterministic_included=False,
                screen_reason=f"audit_excluded:{audit_excluded_doc_ids[document.doc_id]}",
            )
            for document in documents
        ]
    union = _UnionFind(document.doc_id for document in documents)
    events: list[DedupeEvent] = []

    exact_keys: dict[tuple[str, str], str] = {}
    for document in documents:
        for kind, value in (("doi", document.doi), ("pmid", document.pmid)):
            if value is None:
                continue
            key = (kind, value)
            existing = exact_keys.get(key)
            if existing is None:
                exact_keys[key] = document.doc_id
                continue
            union.union(existing, document.doc_id)
            events.append(
                DedupeEvent(
                    event="auto_merge",
                    cluster_id=None,
                    preferred_doc_id=None,
                    member_doc_ids=tuple(sorted((existing, document.doc_id))),
                    identity_key=f"{kind}:{value}",
                    title_score=None,
                    reason=f"exact_{kind}",
                )
            )

    for left_index, left in enumerate(documents):
        for right in documents[left_index + 1 :]:
            if union.find(left.doc_id) == union.find(right.doc_id):
                continue
            if not _same_fuzzy_identity(left, right):
                continue
            score = float(ratio(normalize_title(left.title), normalize_title(right.title)))
            if score >= FUZZY_AUTO_MERGE_THRESHOLD:
                union.union(left.doc_id, right.doc_id)
                events.append(
                    DedupeEvent(
                        event="auto_merge",
                        cluster_id=None,
                        preferred_doc_id=None,
                        member_doc_ids=tuple(sorted((left.doc_id, right.doc_id))),
                        identity_key=None,
                        title_score=score,
                        reason="fuzzy_title_author_year",
                    )
                )
            elif score >= FUZZY_AMBIGUOUS_THRESHOLD:
                events.append(
                    DedupeEvent(
                        event="human_review_required",
                        cluster_id=None,
                        preferred_doc_id=None,
                        member_doc_ids=tuple(sorted((left.doc_id, right.doc_id))),
                        identity_key=None,
                        title_score=score,
                        reason="ambiguous_fuzzy_identity",
                    )
                )

    clusters: dict[str, list[DocumentCandidate]] = {}
    for document in documents:
        clusters.setdefault(union.find(document.doc_id), []).append(document)

    if created_at is None:
        created_at_value = datetime.now(UTC).isoformat()
    elif isinstance(created_at, datetime):
        if created_at.tzinfo is None:
            raise ScreeningError("SCREEN_TIMESTAMP_NAIVE", "created_at must be timezone aware")
        created_at_value = created_at.isoformat()
    else:
        created_at_value = created_at

    papers: list[dict[str, Any]] = []
    include_ids: list[str] = []
    exclude_ids: list[str] = []
    finalized_events: list[DedupeEvent] = []
    for members in sorted(clusters.values(), key=lambda group: sorted(x.doc_id for x in group)):
        preferred = min(members, key=_preferred_key)
        member_doc_ids = sorted(document.doc_id for document in members)
        cluster_id = _cluster_id(member_doc_ids) if len(members) > 1 else None
        included_members = [document for document in members if document.deterministic_included]
        screen_status = "included" if included_members else "excluded"
        # Ledger invariant: an included paper's screen_reason is null (the reason column
        # explains exclusions only).  The article-type-unconfirmed pass-through remains
        # visible through the ledger's null article_type plus the run-record count.
        screen_reason = (
            None if included_members else preferred.screen_reason or "deterministic_exclusion"
        )

        paper_id = derive_paper_id(doi=preferred.doi, pmid=preferred.pmid, doc_id=preferred.doc_id)
        query_families = sorted({family for member in members for family in member.query_families})
        result_ids = sorted(
            {result_id for member in members for result_id in member.search_result_ids}
        )
        alternates = [doc_id for doc_id in member_doc_ids if doc_id != preferred.doc_id]
        paper = {
            "paper_id": paper_id,
            "doc_id": preferred.doc_id,
            "alternate_doc_ids": alternates,
            "doi": preferred.doi,
            "pmid": preferred.pmid,
            "title": preferred.title,
            "first_author": preferred.first_author,
            "pub_year": preferred.pub_year,
            "source": preferred.source,
            "article_type": preferred.article_type,
            "query_families": query_families,
            "search_result_ids": result_ids,
            "content_tier": preferred.content_tier,
            "publication_status": preferred.publication_status,
            "screen_status": screen_status,
            "screen_reason": screen_reason,
            "dedupe_cluster_id": cluster_id,
            "dedupe_preferred": True,
            "map_status": "not_mapped",
            "eligible": None,
            "exclusion_reason": None,
            "map_result_id": None,
            "raw_artifact_path": None,
            "raw_finding_count": 0,
            "accepted_finding_count": 0,
            "quarantined_finding_count": 0,
            "failure_code": None,
            "dataset_or_cohort_id": infer_dataset_or_cohort_id(members),
            "prompt_version": None,
            "schema_version": schema_version,
            "config_sha256": config_sha256,
            "cfghash": None,
            "created_at": created_at_value,
        }
        papers.append(paper)
        (include_ids if screen_status == "included" else exclude_ids).append(paper_id)
        if cluster_id:
            finalized_events.append(
                DedupeEvent(
                    event="cluster_finalized",
                    cluster_id=cluster_id,
                    preferred_doc_id=preferred.doc_id,
                    member_doc_ids=tuple(member_doc_ids),
                    identity_key=None,
                    title_score=None,
                    reason="published_preferred_then_content_identity",
                )
            )

    cluster_lookup = {
        doc_id: _cluster_id([member.doc_id for member in members])
        for members in clusters.values()
        if len(members) > 1
        for doc_id in (member.doc_id for member in members)
    }
    for event in events:
        cluster_id = next(
            (cluster_lookup[doc_id] for doc_id in event.member_doc_ids if doc_id in cluster_lookup),
            None,
        )
        finalized_events.append(
            DedupeEvent(
                event=event.event,
                cluster_id=cluster_id,
                preferred_doc_id=event.preferred_doc_id,
                member_doc_ids=event.member_doc_ids,
                identity_key=event.identity_key,
                title_score=event.title_score,
                reason=event.reason,
            )
        )

    papers.sort(key=lambda row: row["paper_id"])
    return ScreenResult(
        papers=tuple(papers),
        include_paper_ids=tuple(sorted(include_ids)),
        exclude_paper_ids=tuple(sorted(exclude_ids)),
        dedupe_log=tuple(
            sorted(
                finalized_events,
                key=lambda event: (event.event, event.member_doc_ids, event.reason),
            )
        ),
    )
