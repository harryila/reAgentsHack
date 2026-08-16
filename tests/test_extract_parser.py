from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from literature_multiverse.config import config_sha256, load_question_config
from literature_multiverse.extract import (
    MapParseError,
    assemble_extraction_ledgers,
    iter_raw_findings,
    parse_map_file,
    parse_map_text,
    reconcile_envelopes,
)
from literature_multiverse.normalize import normalize_raw_finding
from literature_multiverse.paperclip_cli import run_paperclip

ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_PROBE = ROOT / "data/raw/smoke/probe_map_m_2bc51e4b.json"


def _screened_paper(doc_id: str, *, status: str = "included", reason: str | None = None) -> dict:
    config = load_question_config(ROOT / "configs/questions/fixture-a.yaml")
    return {
        "paper_id": f"doc:{doc_id}",
        "doc_id": doc_id,
        "alternate_doc_ids": [],
        "doi": None,
        "pmid": None,
        "title": f"Paper {doc_id}",
        "first_author": "Example",
        "pub_year": 2024,
        "source": "fixture",
        "article_type": "research-article",
        "query_families": ["direct"],
        "search_result_ids": [f"search-{doc_id}"],
        "content_tier": "full_text",
        "publication_status": "peer_reviewed",
        "screen_status": status,
        "screen_reason": reason,
        "dedupe_cluster_id": None,
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
        "dataset_or_cohort_id": None,
        "prompt_version": None,
        "schema_version": "1",
        "config_sha256": config_sha256(config),
        "cfghash": None,
        "created_at": "2026-08-15T12:00:00+00:00",
    }


def _authoritative_lines() -> dict[str, dict[str, dict[str, str]]]:
    return {
        "PMC12384908": {
            "L18": {
                "section": "Results",
                "text": (
                    "However, the AS group had higher increases in arm lean mass "
                    "(Δ = 0.96 vs 0.59 kg; P = .003, d = 0.74), skeletal muscle mass "
                    "index (Δ = 0.71 vs 0.42 kg/m²; P = .004, d = 0.71), handgrip "
                    "strength (Δ = 3.66 vs 1.16 kg; P = .047, d = 0.51), and knee "
                    "extension strength (Δ = 2.28 vs 1.02 kg; P < .001, d = 0.89) "
                    "than the PLA group. There were no differences in physical "
                    "performance between the RT conditions over time."
                ),
            }
        },
        "PMC12845069": {
            "L318": {
                "section": "References",
                "text": "Vitamin C and E Supplementation Alters Protein Signalling After a",
            },
            "L319": {
                "section": "References",
                "text": (
                    "Strength Training Session, but Not Muscle Growth During 10 weeks of Training"
                ),
            },
        },
        "PMC12793614": {},
        "PMC12785077": {},
    }


def test_archived_probe_contract_is_derived_offline() -> None:
    envelopes = parse_map_file(ARCHIVED_PROBE)

    assert len(envelopes) == 4
    assert [envelope.doc_id for envelope in envelopes] == [
        "PMC12384908",
        "PMC12845069",
        "PMC12793614",
        "PMC12785077",
    ]
    assert {envelope.map_result_id for envelope in envelopes} == {"m_2bc51e4b"}
    assert sum(envelope.raw_finding_count for envelope in envelopes) == 6

    findings = []
    quarantine = []
    terminal_papers = []
    authoritative_lines = _authoritative_lines()
    for envelope in envelopes:
        assert envelope.successful
        assert envelope.payload is not None
        eligible = envelope.payload["eligible"]
        accepted_for_paper = 0
        quarantined_for_paper = 0
        for raw_finding in iter_raw_findings(envelope, paper_id=f"doc:{envelope.doc_id}"):
            normalized, rejected = normalize_raw_finding(
                raw_finding,
                prompt_version="smoke-v1",
                schema_version="1",
                cfghash="0" * 64,
                source_lines=authoritative_lines[envelope.doc_id],
            )
            if normalized is not None:
                findings.append(normalized)
                accepted_for_paper += 1
            else:
                assert rejected is not None
                quarantine.append(rejected)
                quarantined_for_paper += 1
        terminal_papers.append(
            {
                "doc_id": envelope.doc_id,
                "eligible": eligible,
                "raw_finding_count": envelope.raw_finding_count,
                "accepted_finding_count": accepted_for_paper,
                "quarantined_finding_count": quarantined_for_paper,
            }
        )

    assert len(terminal_papers) == 4
    assert len(findings) == 6
    assert len(quarantine) == 0
    assert sum(row["section_flagged"] for row in findings) == 1
    assert sum(not row["section_flagged"] for row in findings) == 5
    assert sum(not paper["eligible"] for paper in terminal_papers) == 2
    assert all(
        paper["raw_finding_count"] == 0 for paper in terminal_papers if not paper["eligible"]
    )
    # Identity always comes from the parsed envelope.
    assert {row["doc_id"] for row in findings} == {"PMC12384908", "PMC12845069"}


def test_model_identity_is_discarded_at_boundary() -> None:
    payload = json.dumps(
        {
            "eligible": True,
            "exclusion_reason": None,
            "findings": [
                {
                    "paper_id": "model-paper",
                    "doc_id": "model-doc",
                    "map_result_id": "model-map",
                    "array_position": 99,
                    "outcome_name": "VO2max",
                    "effect_direction": "no_effect",
                    "evidence_quote": "No difference",
                    "evidence_lines": ["L1"],
                }
            ],
        }
    )
    raw = (
        "Map results  [m_test]\n"
        "--- [1] [success] Paper ---\n"
        "doc_id: authoritative-doc\n"
        f"{payload}\n"
    )
    [envelope] = parse_map_text(raw)
    [finding] = list(iter_raw_findings(envelope, paper_id="doc:authoritative-doc"))

    assert finding.paper_id == "doc:authoritative-doc"
    assert finding.doc_id == "authoritative-doc"
    assert finding.map_result_id == "m_test"
    assert finding.array_position == 0
    assert not ({"paper_id", "doc_id", "map_result_id", "array_position"} & set(finding.payload))


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("no header", "MAP_RESULT_ID_MISSING"),
        ("Map results [m]\n", "MAP_ENVELOPES_MISSING"),
        (
            "Map results [m]\n--- [1] [success] T ---\ndoc_id: d\nnot json\n",
            "MAP_SUCCESS_PAYLOAD_INVALID_JSON",
        ),
        (
            "Map results [m]\n--- [1] [success] T ---\n{}\n",
            "MAP_DOC_ID_MISSING",
        ),
    ],
)
def test_parser_failures_have_stable_codes(raw: str, code: str) -> None:
    with pytest.raises(MapParseError) as exc_info:
        parse_map_text(raw)
    assert exc_info.value.code == code


def test_resume_reconciliation_replaces_failure_without_duplicate_success() -> None:
    failed = parse_map_text(
        "Map results [m]\n--- [1] [failed] T ---\ndoc_id: d1\nprovider failed\n"
    )
    resumed = parse_map_text(
        "Map results [m]\n--- [1] [success] T ---\ndoc_id: d1\n"
        '{"eligible": false, "exclusion_reason": "not eligible", "findings": []}\n'
    )

    reconciled = reconcile_envelopes([failed, resumed], expected_doc_ids=["d1"])

    assert len(reconciled) == 1
    assert reconciled[0].successful


def test_paperclip_boundary_rejects_shell_and_archives_redacted_bytes(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError):
        run_paperclip(
            "paperclip map; echo unsafe",  # type: ignore[arg-type]
            archive_dir=tmp_path,
            archive_stem="unsafe",
        )

    calls = []

    def fake_runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=b'{"result":"ok","api_key":"secret-value"}',
            stderr=b"Authorization: Bearer pretend-token",
        )

    run = run_paperclip(
        ["paperclip", "map", "--json"],
        archive_dir=tmp_path,
        archive_stem="map",
        runner=fake_runner,
    )

    assert run.ok
    assert calls[0][1]["shell"] is False
    assert b"secret-value" not in run.final.stdout_path.read_bytes()
    assert b"pretend-token" not in run.final.stderr_path.read_bytes()
    metadata = json.loads(run.final.metadata_path.read_text())
    assert metadata["argv"] == ["paperclip", "map", "--json"]


def test_paperclip_retries_only_allowlisted_transport_failure(tmp_path: Path) -> None:
    outcomes = [
        subprocess.CompletedProcess(["paperclip"], 1, b"", b"HTTP 429 rate limit"),
        subprocess.CompletedProcess(["paperclip"], 0, b"ok", b""),
    ]

    def fake_runner(argv, **kwargs):
        return outcomes.pop(0)

    run = run_paperclip(
        ["paperclip", "map"],
        archive_dir=tmp_path,
        archive_stem="retry",
        max_retries=1,
        retry_delay_seconds=0,
        runner=fake_runner,
    )
    assert len(run.attempts) == 2
    assert run.ok

    calls = 0

    def permanent_runner(argv, **kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(argv, 2, b"", b"invalid schema")

    permanent = run_paperclip(
        ["paperclip", "map"],
        archive_dir=tmp_path,
        archive_stem="permanent",
        max_retries=3,
        retry_delay_seconds=0,
        runner=permanent_runner,
    )
    assert calls == 1
    assert permanent.final.failure_code == "PAPERCLIP_NONZERO_EXIT"


def test_s3_assembly_preserves_terminal_papers_and_quarantines_bad_rows() -> None:
    config = load_question_config(ROOT / "configs/questions/fixture-a.yaml")
    valid_finding = {
        "study_type": "randomized trial",
        "species": "human",
        "model": None,
        "population_state": "healthy",
        "intervention": "synthetic intervention",
        "intervention_class": "synthetic",
        "comparator": "control",
        "dose_raw": "high",
        "duration_raw": "4 weeks",
        "timing_context": "chronic",
        "outcome_name": "peak_power",
        "outcome_family": "performance",
        "timepoint_raw": "4 weeks",
        "effect_direction": "increase",
        "effect_size_raw": None,
        "p_value": 0.02,
        "significant": True,
        "sample_size": 20,
        "evidence_quote": "Peak power was higher than control.",
        "evidence_lines": ["L10"],
        "confidence": 0.9,
        "moderators": {
            "dose_regime": "high",
            "training_status": "trained",
            "population_state": "healthy",
            "timing_context": "chronic",
        },
    }
    invalid_finding = {**valid_finding, "effect_direction": None, "outcome_name": "fatigue_time"}
    payload = json.dumps(
        {
            "eligible": True,
            "exclusion_reason": None,
            "findings": [valid_finding, invalid_finding],
        }
    )
    envelopes = parse_map_text(
        "Map results [m_fixture]\n"
        "--- [1] [success] Included ---\n"
        "doc_id: included\n"
        f"{payload}\n"
        "--- [2] [failed] Failed ---\n"
        "doc_id: failed\n"
        "provider error\n"
    )

    ledgers = assemble_extraction_ledgers(
        [
            _screened_paper("included"),
            _screened_paper("failed"),
            _screened_paper("excluded", status="excluded", reason="article_type_not_allowed"),
        ],
        envelopes,
        config=config,
        prompt_version="extraction-v1",
        cfghash="f" * 64,
        raw_artifact_path="data/raw/map/fixture-a/map.stdout",
        source_lines_by_doc={
            "included": {
                "L10": {"text": "Peak power was higher than control.", "section": "Results"}
            }
        },
    )

    assert len(ledgers.papers) == 3
    assert len(ledgers.findings) == 1
    assert len(ledgers.quarantine) == 1
    assert ledgers.quarantine[0]["reason_code"] == "FINDING_DIRECTION_JSON_NULL"
    terminal = {paper.doc_id: paper for paper in ledgers.papers}
    assert terminal["included"].raw_finding_count == 2
    assert terminal["included"].accepted_finding_count == 1
    assert terminal["included"].quarantined_finding_count == 1
    assert terminal["failed"].map_status.value == "failed"
    assert terminal["failed"].failure_code == "PAPERCLIP_MAP_FAILED"
    assert terminal["excluded"].map_status.value == "not_mapped"
    assert ledgers.findings[0].grounding_status.value == "exact"
    assert ledgers.counts["map_success"] == 1
    assert ledgers.counts["map_failure"] == 1
    assert ledgers.counts["not_mapped"] == 1


def test_wrapped_entry_titles_parse() -> None:
    """Observed live: results-export titles can wrap across lines (entry 178, 2026-08-15)."""
    from literature_multiverse.extract import parse_map_text

    raw = (
        "Map results  [m_wrap]\n"
        "Query: q\n\n"
        "--- [1] [success] A One Line Title ---\n"
        "doc_id: PMCAAA\n"
        '{"eligible": false, "exclusion_reason": "x", "findings": []}\n\n'
        "--- [2] [success] A Very Long Title That The Export\n"
        "Wraps Onto A Second Line ---\n"
        "doc_id: PMCBBB\n"
        '{"eligible": false, "exclusion_reason": "y", "findings": []}\n'
    )
    envelopes = parse_map_text(raw)
    assert [envelope.doc_id for envelope in envelopes] == ["PMCAAA", "PMCBBB"]
    assert envelopes[1].title == "A Very Long Title That The Export\nWraps Onto A Second Line"


def test_error_entry_with_paper_path_and_empty_body_parses() -> None:
    """Observed live: worker-error envelopes can carry only a paper path in the title slot
    with an empty body (entry 151 of the 2026-08-15 batch-2 map)."""
    from literature_multiverse.extract import MapParseError, parse_map_text

    raw = (
        "Map results  [m_errpath]\n"
        "Query: q\n\n"
        "--- [1] [success] A Fine Paper ---\n"
        "doc_id: PMCAAA\n"
        '{"eligible": false, "exclusion_reason": "x", "findings": []}\n\n'
        "--- [2] [error] /papers/PMC9692807/ ---\n\n"
        "--- [3] [success] Another Fine Paper ---\n"
        "doc_id: PMCBBB\n"
        '{"eligible": false, "exclusion_reason": "y", "findings": []}\n'
    )
    envelopes = parse_map_text(raw)
    assert [envelope.doc_id for envelope in envelopes] == ["PMCAAA", "PMC9692807", "PMCBBB"]
    error_envelope = envelopes[1]
    assert error_envelope.status == "error"
    assert error_envelope.successful is False
    assert error_envelope.payload is None
    assert error_envelope.provider_message is None

    # A success entry without a doc_id line must still hard-fail even if a path-like
    # string appears in the title slot.
    bad = (
        "Map results  [m_badpath]\n"
        "Query: q\n\n"
        "--- [1] [success] /papers/PMCZZZ/ ---\n"
        '{"eligible": false, "exclusion_reason": "x", "findings": []}\n'
    )
    try:
        parse_map_text(bad)
    except MapParseError as exc:
        assert exc.code == "MAP_DOC_ID_MISSING"
    else:
        raise AssertionError("success entry without doc_id must not parse")
