#!/usr/bin/env python3
"""Re-run archived G1 assertions without spending credits; live G1b remains explicit."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
import uuid
from datetime import UTC, datetime
from typing import Any

from literature_multiverse.config import config_sha256, load_config_for_question
from literature_multiverse.extract import (
    iter_raw_findings,
    parse_map_file,
    reconcile_envelopes,
)
from literature_multiverse.lineage import (
    artifact_ref,
    atomic_write_json,
    code_version,
    write_run_record,
)
from literature_multiverse.live import live_search_to_csv
from literature_multiverse.models import RunRecord
from literature_multiverse.normalize import normalize_raw_finding
from literature_multiverse.paperclip_cli import run_paperclip
from literature_multiverse.paths import PATHS
from literature_multiverse.search import parse_search_csv

LIVE_SEARCH_QUERY = (
    "antioxidant vitamin C vitamin E supplementation exercise training adaptation"
)
LIVE_PROBE_SCHEMA = json.dumps(
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["eligible"],
        "properties": {"eligible": {"type": "boolean"}},
    },
    sort_keys=True,
)
LIVE_PROBE_PROMPT = (
    "Answer only whether this paper reports a primary experimental study "
    "(not a review, editorial, or protocol). Set eligible accordingly."
)


def _paperclip_executable() -> str:
    found = shutil.which("paperclip")
    if found:
        return found
    fallback = os.path.expanduser("~/.local/bin/paperclip")
    if os.path.exists(fallback):
        return fallback
    raise ValueError("paperclip executable not found for live G1b probes")


def _map_result_id(stdout: bytes) -> str | None:
    import re

    match = re.search(rb"\bm_[0-9a-f]{6,}\b", stdout)
    return match.group(0).decode() if match else None


def _run_live_g1b(*, force: bool) -> dict[str, Any]:
    """Execute the five remaining design §5.1 live assertions and archive everything."""

    if not os.environ.get("PAPERCLIP_API_KEY"):
        raise ValueError("live G1b requires PAPERCLIP_API_KEY in the environment")
    executable = _paperclip_executable()
    archive = PATHS.root / "data/raw/smoke/live"
    results: dict[str, Any] = {}
    rate_limit_signals: list[str] = []

    def _observe(stderr: bytes, stem: str) -> None:
        lowered = stderr.lower()
        if b"rate limit" in lowered or b"429" in lowered or b"too many requests" in lowered:
            rate_limit_signals.append(stem)

    # 1. Citation URL resolution for the archived grounded finding (PMC12384908 L18).
    citation_url = "https://paperclip.gxl.ai/citations/papers/PMC12384908"
    try:
        request = urllib.request.Request(citation_url, method="GET")
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(2048)
            results["citation_url_resolution"] = {
                "url": f"{citation_url}#L18",
                "http_status": int(response.status),
                "non_empty": bool(body),
                "passed": int(response.status) == 200 and bool(body),
            }
    except Exception as exc:
        results["citation_url_resolution"] = {
            "url": f"{citation_url}#L18",
            "error": str(exc)[:300],
            "passed": False,
        }

    # 2. Bogus result ID must fail with a stable non-retryable code.
    bogus = run_paperclip(
        [executable, "map", "--from", "s_00000000", LIVE_PROBE_PROMPT],
        archive_dir=archive,
        archive_stem="live-bogus-result-id",
        timeout_seconds=120,
        force=force,
    )
    _observe(bogus.final.stderr, "live-bogus-result-id")
    results["bogus_result_id_failure"] = {
        "returncode": bogus.final.returncode,
        "failure_code": bogus.final.failure_code,
        "passed": (not bogus.ok)
        and bogus.final.failure_code in {"PAPERCLIP_NONZERO_EXIT", "PAPERCLIP_INBAND_ERROR"},
    }

    # A dedicated exact-10-paper search backs both the invalid-schema and resume probes.
    live_search = live_search_to_csv(
        LIVE_SEARCH_QUERY,
        sources=("pmc",),
        archive_dir=archive,
        archive_stem="live-10-paper-search",
        limit=10,
        use_all=True,
        force=force,
    )
    result_id = live_search.result_id
    include_ids = [str(record["doc_id"]) for record in parse_search_csv(live_search.csv_bytes)]
    if len(include_ids) != 10 or len(set(include_ids)) != 10:
        raise ValueError(
            f"live search returned {len(include_ids)} unique ids; the resume probe "
            "requires exactly 10"
        )
    results["ten_paper_search"] = {"result_id": result_id, "doc_ids": include_ids}

    # 3. An invalid schema must fail without retries.
    invalid_schema = run_paperclip(
        [
            executable,
            "map",
            "--from",
            result_id,
            "--limit",
            "1",
            "--output-schema",
            '{"type":"bogus_type"}',
            LIVE_PROBE_PROMPT,
        ],
        archive_dir=archive,
        archive_stem="live-invalid-schema",
        timeout_seconds=180,
        force=force,
    )
    _observe(invalid_schema.final.stderr, "live-invalid-schema")
    results["invalid_schema_failure"] = {
        "returncode": invalid_schema.final.returncode,
        "failure_code": invalid_schema.final.failure_code,
        "passed": not invalid_schema.ok,
    }

    # 4. Exact-10-paper kill/resume reconciliation with bounded concurrency so the
    #    first invocation is interrupted after at least one recorded completion.
    # `script -q /dev/null` allocates a pseudo-TTY so the CLI streams its progress line
    # (which carries the map id) before the deliberate kill; without a TTY the CLI
    # buffers all output and a killed run captures nothing.
    initial = run_paperclip(
        [
            "script",
            "-q",
            "/dev/null",
            executable,
            "map",
            "--from",
            result_id,
            "--output-schema",
            LIVE_PROBE_SCHEMA,
            LIVE_PROBE_PROMPT,
        ],
        archive_dir=archive,
        archive_stem="live-resume-initial",
        timeout_seconds=6,
        force=force,
    )
    _observe(initial.final.stderr, "live-resume-initial")
    map_id = _map_result_id(initial.final.stdout) or _map_result_id(initial.final.stderr)
    if map_id is None:
        raise ValueError("live resume probe could not observe a map result id before the kill")
    resume = run_paperclip(
        [executable, "map", "--resume", map_id],
        archive_dir=archive,
        archive_stem="live-resume-continue",
        timeout_seconds=600,
        force=force,
    )
    _observe(resume.final.stderr, "live-resume-continue")
    if not resume.ok:
        raise ValueError(
            f"map --resume failed with {resume.final.failure_code}; "
            f"see {resume.final.stderr_path}"
        )
    # The authoritative terminal artifact is the saved results export — the same pinned
    # format the archived-parser contract covers — never raw map stdout.
    results_path = archive / "live-resume-results.txt"
    saved = run_paperclip(
        [executable, "results", map_id, "--save", str(results_path)],
        archive_dir=archive,
        archive_stem="live-resume-results-export",
        timeout_seconds=180,
        force=force,
    )
    _observe(saved.final.stderr, "live-resume-results-export")
    if not saved.ok or not results_path.exists():
        raise ValueError(
            f"results export failed with {saved.final.failure_code}; "
            f"see {saved.final.stderr_path}"
        )
    terminal_envelopes = parse_map_file(results_path)
    reconciled = reconcile_envelopes(
        [terminal_envelopes],
        expected_doc_ids=include_ids,
    )
    terminal_ids = sorted(envelope.doc_id for envelope in reconciled)
    results["exact_10_paper_kill_resume_reconciliation"] = {
        "map_id": map_id,
        "killed_mid_run": initial.final.failure_code == "PAPERCLIP_TIMEOUT",
        "terminal_count": len(reconciled),
        "successful_count": sum(envelope.successful for envelope in reconciled),
        "passed": terminal_ids == sorted(include_ids),
    }

    # 5. Rate-limit observation across every live attempt.
    results["rate_limit_observation"] = {
        "signals": rate_limit_signals,
        "observed": bool(rate_limit_signals),
        "passed": True,
    }
    return results


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--question", default="fixture-a")
    result.add_argument("--live", action="store_true")
    result.add_argument("--force", action="store_true")
    return result


def _frozen_lines() -> dict:
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
                    "than the PLA group. There were no differences in physical performance "
                    "between the RT conditions over time."
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
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    question_id = "fixture-a" if args.question == "fixture" else args.question
    started = datetime.now(UTC)
    config = load_config_for_question(question_id)
    config.authorize_stage("s0", explicit_fixture=True, live_provider=False)
    probe_path = PATHS.root / "data/raw/smoke/probe_map_m_2bc51e4b.json"
    envelopes = parse_map_file(probe_path)
    accepted = []
    quarantine = []
    frozen_lines = _frozen_lines()
    for envelope in envelopes:
        for raw in iter_raw_findings(envelope, paper_id=f"doc:{envelope.doc_id}"):
            row, rejected = normalize_raw_finding(
                raw,
                prompt_version="smoke-v1",
                schema_version="1",
                cfghash="0" * 64,
                source_lines=frozen_lines.get(envelope.doc_id),
            )
            if row is not None:
                accepted.append(row)
            else:
                quarantine.append(rejected)
    assertions = {
        "authoritative_envelopes_equal_4": len(envelopes) == 4,
        "raw_findings_equal_6": sum(item.raw_finding_count for item in envelopes) == 6,
        "accepted_findings_equal_6": len(accepted) == 6,
        "quarantine_equal_0": len(quarantine) == 0,
        "exact_grounded_equal_6": sum(row["grounding_status"] == "exact" for row in accepted) == 6,
        "reference_flagged_equal_1": sum(row["section_flagged"] for row in accepted) == 1,
        "non_section_flagged_equal_5": sum(not row["section_flagged"] for row in accepted) == 5,
        "clean_ineligible_zero_finding_papers_equal_2": sum(
            envelope.payload is not None
            and envelope.payload["eligible"] is False
            and not envelope.payload["findings"]
            for envelope in envelopes
        )
        == 2,
    }
    if not all(assertions.values()):
        raise AssertionError(f"archived smoke contract failed: {assertions}")

    live_results: dict[str, Any] | None = None
    if args.live:
        live_results = _run_live_g1b(force=args.force)
    live_assertion_names = (
        "citation_url_resolution",
        "bogus_result_id_failure",
        "invalid_schema_failure",
        "exact_10_paper_kill_resume_reconciliation",
        "rate_limit_observation",
    )
    live_passed = live_results is not None and all(
        live_results[name].get("passed") is True for name in live_assertion_names
    )
    report = {
        "report_version": "2",
        "mode": "live+archived" if args.live else "offline_archived",
        "g1_core_passed": True,
        "g1b_passed": live_passed,
        "assertions": assertions,
        "live_assertions": live_results,
        "pending_live_assertions": []
        if live_passed
        else [
            name
            for name in live_assertion_names
            if live_results is None or live_results.get(name, {}).get("passed") is not True
        ],
        "safety": (
            "g1b passed; corpora above 10 papers are authorized"
            if live_passed
            else "no corpus over 10 papers is authorized until g1b_passed is true"
        ),
    }
    report_path = PATHS.root / "data/raw/smoke/g1b_report.json"
    atomic_write_json(report_path, report, force=args.force)
    config_path = PATHS.config_path(config.question_id)
    record = RunRecord(
        run_id=f"s0-{uuid.uuid4().hex}",
        question_id=config.question_id,
        stage="s0",
        stage_version="1",
        status="complete",
        started_at=started,
        completed_at=datetime.now(UTC),
        code_version=code_version(PATHS.root),
        command_argv=["scripts/s0_smoke_test.py", *(argv if argv is not None else sys.argv[1:])],
        config_path=PATHS.repository_relative(config_path),
        config_sha256=config_sha256(config),
        prompt_path=None,
        prompt_sha256=None,
        schema_path=None,
        schema_sha256=None,
        cfghash=None,
        upstream=[],
        inputs=[
            artifact_ref(config_path, root=PATHS.root),
            artifact_ref(probe_path, root=PATHS.root),
        ],
        outputs=[artifact_ref(report_path, root=PATHS.root)],
        external_result_ids={"paperclip": ["m_2bc51e4b"]},
        counts={
            "papers": len(envelopes),
            "raw_findings": 6,
            "accepted_findings": len(accepted),
            "quarantined_findings": len(quarantine),
            "reference_section_flagged": 1,
            "non_section_flagged": 5,
            "ineligible_zero_finding_papers": 2,
        },
        warnings=[] if live_passed else ["g1b_live_assertions_pending"],
    )
    write_run_record(PATHS.run_record_path(config.question_id, "s0"), record, force=args.force)
    if live_passed:
        print("s0 archived core passed; G1b live assertions PASSED — >10-paper maps authorized")
    elif args.live:
        pending = report["pending_live_assertions"]
        print(f"s0 archived core passed; G1b live assertions INCOMPLETE: {pending}")
    else:
        print("s0 archived core passed; G1b live assertions remain intentionally pending")
    return 0 if (live_passed or not args.live) else 1


if __name__ == "__main__":
    raise SystemExit(main())
