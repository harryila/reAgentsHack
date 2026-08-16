from __future__ import annotations

import json
from datetime import UTC, datetime

from literature_multiverse.screen import (
    FUZZY_AMBIGUOUS_THRESHOLD,
    FUZZY_AUTO_MERGE_THRESHOLD,
    normalize_doi,
    screen_candidates,
)
from literature_multiverse.search import (
    consolidate_occurrences,
    occurrences_for_query,
    search_result_id,
)

NOW = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)


def _occurrence(
    doc_id: str,
    title: str,
    *,
    family: str = "broad",
    doi: str | None = None,
    pmid: str | None = None,
    author: str | None = "Example, Alice",
    year: int | None = 2020,
    article_type: str | None = "research-article",
    status: str = "peer_reviewed",
    content_tier: str = "full_text",
    raw_metadata: dict | None = None,
) -> dict:
    return {
        "doc_id": doc_id,
        "query_family": family,
        "queries": [f"query-{family}"],
        "source": "pmc",
        "search_result_ids": [f"s-{doc_id}-{family}"],
        "title": title,
        "doi": doi,
        "pmid": pmid,
        "first_author": author,
        "pub_year": year,
        "article_type": article_type,
        "content_tier": content_tier,
        "publication_status": status,
        "raw_metadata": raw_metadata or {},
    }


def test_exact_doi_dedupe_prefers_published_and_preserves_provenance() -> None:
    candidates = [
        _occurrence(
            "PREPRINT1",
            "Vitamin C during resistance training",
            family="null",
            doi="https://doi.org/10.1234/ABC.",
            status="preprint",
        ),
        _occurrence(
            "PMC1",
            "Vitamin C during resistance training",
            family="broad",
            doi="doi:10.1234/abc",
            status="peer_reviewed",
        ),
        # Repeated query hit for the published document: s1 may preserve both families,
        # but s2 must still create one canonical identity.
        _occurrence(
            "PMC1",
            "Vitamin C during resistance training",
            family="negative",
            doi="10.1234/abc",
            status="peer_reviewed",
        ),
    ]

    result = screen_candidates(
        candidates,
        allowed_article_types=["research-article"],
        config_sha256="a" * 64,
        created_at=NOW,
    )

    assert len(result.papers) == 1
    [paper] = result.papers
    assert paper["paper_id"] == "doi:10.1234/abc"
    assert paper["doc_id"] == "PMC1"
    assert paper["alternate_doc_ids"] == ["PREPRINT1"]
    assert paper["query_families"] == ["broad", "negative", "null"]
    assert len(paper["search_result_ids"]) == 3
    assert paper["publication_status"] == "peer_reviewed"
    assert result.include_paper_ids == ("doi:10.1234/abc",)
    assert not result.exclude_paper_ids
    assert any(event.reason == "exact_doi" for event in result.dedupe_log)
    assert any(event.event == "cluster_finalized" for event in result.dedupe_log)


def test_yfanti_style_year_adjacent_title_match_auto_merges() -> None:
    candidates = [
        _occurrence(
            "PMC-Y-2010",
            "Antioxidant supplementation does not alter endurance training adaptation",
            author="Yfanti, Christina",
            year=2010,
        ),
        _occurrence(
            "PMC-Y-2011",
            "Antioxidant supplementation does not alter endurance training adaptations",
            author="Christina Yfanti",
            year=2011,
        ),
    ]

    result = screen_candidates(
        candidates,
        allowed_article_types=["research-article"],
        config_sha256="b" * 64,
        created_at=NOW,
    )

    assert len(result.papers) == 1
    assert len(result.papers[0]["alternate_doc_ids"]) == 1
    fuzzy_event = next(
        event for event in result.dedupe_log if event.reason == "fuzzy_title_author_year"
    )
    assert fuzzy_event.title_score is not None
    assert fuzzy_event.title_score >= FUZZY_AUTO_MERGE_THRESHOLD


def test_ambiguous_fuzzy_pair_is_logged_but_not_merged() -> None:
    candidates = [
        _occurrence(
            "D1",
            "Vitamin supplementation and exercise training responses in adults",
            author="Yfanti, C",
            year=2010,
        ),
        _occurrence(
            "D2",
            "Vitamin supplementation and exercise responses in adults",
            author="C Yfanti",
            year=2011,
        ),
    ]

    result = screen_candidates(
        candidates,
        allowed_article_types=["research-article"],
        config_sha256="c" * 64,
        created_at=NOW,
    )

    assert len(result.papers) == 2
    event = next(event for event in result.dedupe_log if event.event == "human_review_required")
    assert event.title_score is not None
    assert FUZZY_AMBIGUOUS_THRESHOLD <= event.title_score < FUZZY_AUTO_MERGE_THRESHOLD


def test_deterministic_article_filter_and_identity_funnel_reconcile() -> None:
    result = screen_candidates(
        [
            _occurrence("A", "Included trial"),
            _occurrence("B", "Systematic review", article_type="review-article"),
            _occurrence("C", "Unknown type", article_type=None),
        ],
        allowed_article_types=["research-article"],
        config_sha256="d" * 64,
        created_at=NOW,
    )

    # A confirmed disallowed type is excluded; a missing type is unconfirmed metadata on
    # this CLI build and passes through visibly (extraction owns final eligibility).
    assert len(result.include_paper_ids) == 2
    assert len(result.exclude_paper_ids) == 1
    assert len(result.include_paper_ids) + len(result.exclude_paper_ids) == len(result.papers)
    reasons = {
        paper["screen_reason"] for paper in result.papers if paper["screen_status"] == "excluded"
    }
    assert reasons == {"article_type_not_allowed"}
    included = [paper for paper in result.papers if paper["screen_status"] == "included"]
    assert all(paper["screen_reason"] is None for paper in included)
    unconfirmed = [paper for paper in included if paper["article_type"] is None]
    assert len(unconfirmed) == 1


def test_trial_identifier_is_recorded_as_cohort_id() -> None:
    result = screen_candidates(
        [
            _occurrence(
                "D",
                "Training trial NCT12345678",
                raw_metadata={"abstract": "Registration NCT12345678"},
            )
        ],
        allowed_article_types=["research-article"],
        config_sha256="e" * 64,
        created_at=NOW,
    )

    assert result.papers[0]["dataset_or_cohort_id"] == "NCT12345678"


def test_doi_normalization_is_stable() -> None:
    assert normalize_doi(" HTTPS://doi.org/10.1000/XYZ. ") == "10.1000/xyz"


def test_search_json_preserves_doc_family_provenance_and_saved_result_id() -> None:
    first = json.dumps(
        {
            "result_id": "s_first",
            "results": [
                {
                    "doc_id": "PMC1",
                    "title": "Trial",
                    "source": "pmc",
                    "article_type": "research-article",
                    "year": 2024,
                }
            ],
        }
    )
    second = json.dumps(
        {
            "search_result_id": "s_second",
            "results": [
                {
                    "doc_id": "PMC1",
                    "title": "Trial",
                    "source": "pmc",
                    "article_type": "research-article",
                    "year": 2024,
                }
            ],
        }
    )
    occurrences = [
        *occurrences_for_query(
            first,
            query_family="direct",
            query="first query",
            source="pmc",
            search_result_id=search_result_id(first),
        ),
        *occurrences_for_query(
            second,
            query_family="direct",
            query="second query",
            source="pmc",
            search_result_id=search_result_id(second),
        ),
    ]

    [candidate] = consolidate_occurrences(occurrences)
    assert candidate.doc_id == "PMC1"
    assert candidate.queries == ("first query", "second query")
    assert candidate.search_result_ids == ("s_first", "s_second")
    assert search_result_id("Search results [s_human]\n") == "s_human"


def test_audit_paper_exclusion_is_deterministic_screen_state() -> None:
    from literature_multiverse.screen import screen_candidates

    occurrences = [
        {
            "doc_id": "PMCAAA",
            "title": "Kept paper",
            "query_family": "direct",
            "query": "q",
            "source": "pmc",
            "search_result_ids": ["s_1"],
            "raw_metadata": {},
        },
        {
            "doc_id": "PMCBBB",
            "title": "Audit-excluded paper",
            "query_family": "direct",
            "query": "q",
            "source": "pmc",
            "search_result_ids": ["s_1"],
            "raw_metadata": {},
        },
    ]
    result = screen_candidates(
        occurrences,
        allowed_article_types=["research-article"],
        config_sha256="deadbeef",
        audit_excluded_doc_ids={"PMCBBB": "no structured training program"},
    )
    by_doc = {row["doc_id"]: row for row in result.papers}
    assert by_doc["PMCAAA"]["screen_status"] == "included"
    assert by_doc["PMCBBB"]["screen_status"] == "excluded"
    assert by_doc["PMCBBB"]["screen_reason"] == "audit_excluded:no structured training program"
