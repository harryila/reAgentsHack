#!/usr/bin/env python3
"""Freeze a human-audit sample, then finalize the audit and G3 gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from literature_multiverse.audit import (
    DEFAULT_AUDIT_FIELDS,
    audit_candidate_rows,
    select_audit_sample,
)
from literature_multiverse.config import authorize_stage, load_config_for_question
from literature_multiverse.gates import GateContractError, build_g3_artifact, finalize_audit_record
from literature_multiverse.lineage import atomic_write_json
from literature_multiverse.models import AuditRecord, VerificationRecord
from literature_multiverse.paths import PATHS
from literature_multiverse.records import read_parquet_records

PRIMARY_DIRECTIONS = frozenset({"increase", "no_effect", "decrease"})


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateContractError(f"invalid_json:{path}") from exc
    if not isinstance(value, dict):
        raise GateContractError(f"json_root_must_be_object:{path}")
    return value


def _read_records(path: Path) -> list[dict[str, Any]]:
    try:
        return read_parquet_records(path)
    except Exception as exc:
        raise GateContractError(f"invalid_parquet:{path}") from exc


def _eligible_paper_ids(papers: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(paper["paper_id"])
        for paper in papers
        if paper.get("screen_status") == "included"
        and paper.get("map_status") == "success"
        and paper.get("eligible") is True
    }


def _audit_candidates(
    papers: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    *,
    primary_family: str,
) -> list[dict[str, Any]]:
    return audit_candidate_rows(papers, findings, primary_family=primary_family)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--scope", choices=("v1", "scaled"), default="v1")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--new-paper-target", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--finalize-g3", action="store_true")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _write_sample(
    *,
    args: argparse.Namespace,
    config: Any,
    papers: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    processed: Path,
) -> None:
    primary_family = config.outcomes.primary_family
    assert primary_family is not None
    candidates = _audit_candidates(papers, findings, primary_family=primary_family)
    new_paper_ids: list[str] = []
    if args.scope == "scaled":
        new_ids_path = processed / "new_paper_ids.json"
        new_payload = _read_object(new_ids_path)
        raw_ids = new_payload.get("paper_ids")
        if not isinstance(raw_ids, list) or any(not isinstance(value, str) for value in raw_ids):
            raise GateContractError("new_paper_ids_invalid")
        new_paper_ids = raw_ids
    sample = select_audit_sample(
        candidates,
        sample_size=args.sample_size,
        seed=args.seed,
        new_paper_ids=new_paper_ids,
        minimum_new_distinct_papers=args.new_paper_target,
    )
    sample_path = processed / "audit_sample.json"
    decisions_path = processed / "audit_decisions.json"
    if decisions_path.exists():
        raise GateContractError(f"audit_decisions_already_exist:{decisions_path}")
    atomic_write_json(
        sample_path,
        {
            "audit_sample_version": "1",
            "scope": args.scope,
            **sample,
        },
        force=args.force,
    )

    if args.fixture:
        checks = {name: True for name in DEFAULT_AUDIT_FIELDS}
        decisions = [
            {"finding_id": finding_id, "checks": checks, "error_codes": []}
            for finding_id in sample["finding_ids"]
        ]
        anchors = {anchor.paper_id: True for anchor in config.anchor_papers or []}
    else:
        decisions = [
            {
                "finding_id": finding_id,
                "checks": {name: None for name in DEFAULT_AUDIT_FIELDS},
                "error_codes": [],
            }
            for finding_id in sample["finding_ids"]
        ]
        anchors = {anchor.paper_id: None for anchor in config.anchor_papers or []}
    atomic_write_json(
        decisions_path,
        {
            "audit_decisions_version": "1",
            "human_completion_required": not args.fixture,
            "decisions": decisions,
            "anchor_results": anchors,
        },
    )
    print(
        json.dumps(
            {
                "status": "sample_frozen",
                "question_id": args.question,
                "sample_size": len(sample["finding_ids"]),
                "sample": PATHS.repository_relative(sample_path),
                "decisions": PATHS.repository_relative(decisions_path),
            },
            sort_keys=True,
        )
    )


def _finalize(
    *,
    args: argparse.Namespace,
    config: Any,
    papers: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    processed: Path,
) -> None:
    sample = _read_object(processed / "audit_sample.json")
    decisions_payload = _read_object(processed / "audit_decisions.json")
    raw_decisions = decisions_payload.get("decisions")
    anchors = decisions_payload.get("anchor_results")
    finding_ids = sample.get("finding_ids")
    if not isinstance(raw_decisions, list) or not isinstance(anchors, Mapping):
        raise GateContractError("audit_decisions_shape_invalid")
    if not isinstance(finding_ids, list) or any(
        not isinstance(value, str) for value in finding_ids
    ):
        raise GateContractError("audit_sample_finding_ids_invalid")

    new_ids: set[str] = set()
    if args.scope == "scaled":
        raw_new = _read_object(processed / "new_paper_ids.json").get("paper_ids")
        if not isinstance(raw_new, list):
            raise GateContractError("new_paper_ids_invalid")
        new_ids = {str(value) for value in raw_new}
    primary_family = config.outcomes.primary_family
    assert primary_family is not None
    candidates = _audit_candidates(papers, findings, primary_family=primary_family)
    eligible_ids = _eligible_paper_ids(papers)
    audit_eligible_ids = {str(row["paper_id"]) for row in candidates}
    audit = finalize_audit_record(
        seed=int(sample.get("seed", args.seed)),
        sampled_finding_ids=finding_ids,
        raw_decisions=raw_decisions,
        anchor_results={str(key): value for key, value in anchors.items()},
        requested_sample_size=int(sample.get("sample_size", args.sample_size)),
        newly_added_eligible_papers=(
            len(new_ids & eligible_ids) if args.scope == "scaled" else None
        ),
        newly_added_audit_eligible_papers=(
            len(new_ids & audit_eligible_ids) if args.scope == "scaled" else None
        ),
        sampled_new_distinct_papers=(
            int(sample.get("sampled_new_distinct_papers", 0))
            if args.scope == "scaled"
            else None
        ),
    )
    audit_path = processed / "audit.json"
    verification = VerificationRecord.model_validate(
        _read_object(processed / "verification.json")
    )
    if args.fixture:
        g1b_passed = True
    else:
        g1b = _read_object(PATHS.data_dir / "raw" / "smoke" / "g1b_report.json")
        g1b_passed = g1b.get("g1b_passed") is True
    g3 = build_g3_artifact(
        config=config,
        papers=papers,
        findings=findings,
        verification=verification,
        audit=AuditRecord.model_validate(audit),
        g1b_passed=g1b_passed,
    )
    g3_path = processed / "g3_gate.json"
    # Validate and compute both artifacts before promoting either path.
    if not args.force and (audit_path.exists() or g3_path.exists()):
        raise GateContractError("audit_or_g3_output_already_exists")
    atomic_write_json(audit_path, audit, force=args.force)
    atomic_write_json(g3_path, g3, force=args.force)
    print(
        json.dumps(
            {
                "status": "complete",
                "question_id": args.question,
                "audit_correct": audit.correct_count,
                "audit_total": audit.total_count,
                "trust_passed": g3["trust_passed"],
                "story_passed": g3["story_passed"],
                "action": g3["action"],
                "audit": PATHS.repository_relative(audit_path),
                "g3_gate": PATHS.repository_relative(g3_path),
            },
            sort_keys=True,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config_for_question(args.question, require_locked=True)
    authorize_stage(config, "s4", explicit_fixture=args.fixture, live_provider=False)
    processed = PATHS.processed_dir(args.question)
    papers = _read_records(processed / "papers.parquet")
    findings = _read_records(processed / "findings.parquet")
    if args.finalize_g3:
        _finalize(
            args=args,
            config=config,
            papers=papers,
            findings=findings,
            processed=processed,
        )
    else:
        _write_sample(
            args=args,
            config=config,
            papers=papers,
            findings=findings,
            processed=processed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
