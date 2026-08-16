"""Offline contracts for the CSV search-export path pinned by the live G1b probes."""

from __future__ import annotations

import pytest

from literature_multiverse.live import map_id_from_stdout, search_id_from_stdout
from literature_multiverse.paperclip_cli import PaperclipBoundaryError
from literature_multiverse.search import (
    SearchParseError,
    occurrences_for_query,
    parse_search_csv,
    search_result_id,
)

# Byte-for-byte shape observed from `paperclip results s_... --save file.csv` on
# 2026-08-15; the abstract column is a provider-generated summary, not a real abstract.
PINNED_CSV = (
    "title,authors,id,source,date,url,abstract\n"
    'Antioxidants and Exercise Performance,"Madalyn Riley Higgins, Azimeh Izadi",'
    "PMC7697466,PMC,2020-11-15,https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7697466/,"
    '"This paper reviews antioxidant supplementation."\n'
    "Does Vitamin C and E Supplementation Impair Adaptations?,"
    '"Michalis G. Nikolaidis",PMC3425865,PMC,2012-01-01,'
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3425865/,"
    '"This study investigated interference with exercise adaptations."\n'
)

PINNED_SEARCH_STDOUT = "Found 10 papers  [s_1304e91f]\n\n  1. Some Title\n"


def test_parse_search_csv_yields_canonical_records() -> None:
    records = parse_search_csv(PINNED_CSV)
    assert [record["doc_id"] for record in records] == ["PMC7697466", "PMC3425865"]
    assert records[0]["pub_year"] == 2020
    assert records[0]["authors"][0] == "Madalyn Riley Higgins"
    assert records[0]["content_tier"] == "unknown"
    assert records[1]["pub_year"] == 2012


def test_parse_search_csv_rejects_missing_required_columns() -> None:
    with pytest.raises(SearchParseError) as excinfo:
        parse_search_csv("name,value\na,1\n")
    assert excinfo.value.code == "SEARCH_CSV_HEADER_INVALID"


def test_parse_search_csv_rejects_blank_doc_id() -> None:
    broken = "title,authors,id,source,date,url,abstract\nSome Title,A Author,,PMC,2020,,\n"
    with pytest.raises(SearchParseError) as excinfo:
        parse_search_csv(broken)
    assert excinfo.value.code == "SEARCH_DOC_ID_MISSING"


def test_occurrences_for_query_accepts_csv_format() -> None:
    occurrences = occurrences_for_query(
        PINNED_CSV,
        query_family="direct",
        query="antioxidant exercise",
        source="pmc",
        search_result_id="s_1304e91f",
        fmt="csv",
    )
    assert len(occurrences) == 2
    first = occurrences[0]
    assert first.doc_id == "PMC7697466"
    assert first.first_author == "Madalyn Riley Higgins"
    assert first.pub_year == 2020
    assert first.publication_status == "peer_reviewed"
    assert first.search_result_ids == ("s_1304e91f",)


def test_search_result_id_parses_found_papers_header() -> None:
    assert search_result_id(PINNED_SEARCH_STDOUT) == "s_1304e91f"
    assert search_id_from_stdout(PINNED_SEARCH_STDOUT.encode()) == "s_1304e91f"


def test_map_id_parses_from_progress_stream() -> None:
    assert map_id_from_stdout(b"  [##...] 1/10 papers  run m_2bc51e4b  [2s]") == "m_2bc51e4b"
    with pytest.raises(PaperclipBoundaryError):
        map_id_from_stdout(b"no identifier here")
