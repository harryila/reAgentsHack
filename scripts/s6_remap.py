#!/usr/bin/env python3
"""Propose, execute, or finalize the optional post-hoc exploratory remap."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from literature_multiverse.analysis import derive_primary_cohort, read_parquet_records
from literature_multiverse.config import config_sha256, load_config_for_question
from literature_multiverse.lineage import (
    artifact_ref,
    atomic_write_json,
    atomic_write_text,
    code_version,
    hash_canonical,
    sha256_file,
    write_run_record,
)
from literature_multiverse.models import RunRecord, UpstreamRef
from literature_multiverse.paths import PATHS
from literature_multiverse.remap import (
    RemapContractError,
    approval_template,
    execute_approved_remap,
    not_run_trace,
    propose_remap,
    validate_approval,
)

_FIELD = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$")


def _json_any(path: Path) -> Any:
    if not path.is_file():
        raise RemapContractError(f"missing_json:{path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RemapContractError(f"invalid_json:{path}") from exc


def _json(path: Path) -> dict[str, Any]:
    value = _json_any(path)
    if not isinstance(value, dict):
        raise RemapContractError(f"json_root_not_object:{path}")
    return value


def _yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RemapContractError(f"missing_yaml:{path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RemapContractError(f"invalid_yaml:{path}") from exc
    if not isinstance(value, dict):
        raise RemapContractError(f"yaml_root_not_object:{path}")
    return value


def _write_parquet(path: Path, rows: list[dict[str, Any]], *, force: bool) -> None:
    if path.exists() and not force:
        raise RemapContractError(f"output_exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pd.DataFrame(rows).to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _field(value: str) -> str:
    if not _FIELD.fullmatch(value):
        raise RemapContractError(f"invalid_remap_field:{value}")
    return value


def _primary(config: Any, processed_dir: Path) -> list[dict[str, Any]]:
    assert config.outcomes.primary_family is not None
    return derive_primary_cohort(
        read_parquet_records(processed_dir / "papers.parquet"),
        read_parquet_records(processed_dir / "findings.parquet"),
        _json(processed_dir / "verification.json"),
        primary_family=config.outcomes.primary_family,
    )


def _propose(args: argparse.Namespace, config: Any, analysis_dir: Path) -> int:
    if args.proposal_response is None:
        raise RemapContractError(
            "proposal_provider_not_injected:pass --proposal-response with archived "
            "structured output"
        )
    headline = _json(analysis_dir / "headline.json")
    contradictions = read_parquet_records(analysis_dir / "contradictions.parquet")
    response = _json(args.proposal_response)
    proposal = propose_remap(
        contradictions,
        headline,
        base_moderator_names=[spec.name for spec in config.moderators],
        proposer=lambda _: response,
    )
    field = _field(str(proposal["moderator"]["name"]))
    remap_dir = analysis_dir / "remap" / field
    proposal_path = remap_dir / "proposal.json"
    approval_path = remap_dir / "approval.yaml"
    if not args.force and (proposal_path.exists() or approval_path.exists()):
        raise RemapContractError(f"proposal_outputs_exist:{remap_dir}")
    atomic_write_json(proposal_path, proposal, force=args.force)
    approval = approval_template(proposal)
    atomic_write_text(
        approval_path,
        yaml.safe_dump(approval, sort_keys=False, allow_unicode=True),
        force=args.force,
    )
    atomic_write_json(
        remap_dir / "propose_operation.json",
        {
            "status": "complete",
            "operation": "propose_only",
            "completed_at": datetime.now(UTC).isoformat(),
            "proposal_sha256": proposal["proposal_sha256"],
        },
        force=args.force,
    )
    print(f"s6 proposal ready: {field}; edit {approval_path}")
    return 0


def _execute(args: argparse.Namespace, config: Any, processed_dir: Path, analysis_dir: Path) -> int:
    field = _field(args.field)
    remap_dir = analysis_dir / "remap" / field
    proposal = _json(remap_dir / "proposal.json")
    approval = _yaml(remap_dir / "approval.yaml")
    if validate_approval(proposal, approval) != "approved":
        raise RemapContractError("execute_requires_human_approval")
    if args.responses is None:
        raise RemapContractError(
            "remap_provider_not_injected:pass --responses with archived echo-back output"
        )
    raw = _json_any(args.responses)
    responses = raw.get("responses") if isinstance(raw, dict) else raw
    if not isinstance(responses, list) or any(not isinstance(item, dict) for item in responses):
        raise RemapContractError("remap_responses_must_be_array_of_objects")
    primary = _primary(config, processed_dir)
    execution = execute_approved_remap(
        proposal=proposal,
        approval=approval,
        primary_rows=primary,
        mapper=lambda _: responses,
    )
    atomic_write_json(remap_dir / "execution.json", execution, force=args.force)
    _write_parquet(
        remap_dir / "side_table.parquet",
        list(execution["side_table"]),
        force=args.force,
    )
    atomic_write_json(
        remap_dir / "execute_operation.json",
        {
            "status": "complete",
            "operation": "execute_approved",
            "completed_at": datetime.now(UTC).isoformat(),
            "proposal_sha256": proposal["proposal_sha256"],
            "request_sha256": execution.get("request_sha256"),
        },
        force=args.force,
    )
    print(
        f"s6 remap executed {args.question}/{field}: "
        f"join={execution['reconciliation']['join_fraction']:.3f}"
    )
    return 0


def _mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


def _finalize_run_record(
    *,
    args: argparse.Namespace,
    config: Any,
    processed_dir: Path,
    analysis_dir: Path,
    remap_dir: Path,
    trace_path: Path,
    started_at: datetime,
    completed_at: datetime,
) -> None:
    s5_run = analysis_dir / "run.json"
    if not s5_run.is_file():
        if args.fixture:
            upstream: list[UpstreamRef] = []
        else:
            raise RemapContractError("missing_s5_run_record")
    else:
        payload = _json(s5_run)
        if payload.get("stage") != "s5" or payload.get("status") != "complete":
            raise RemapContractError("s5_upstream_not_complete")
        upstream = [
            UpstreamRef(
                stage="s5",
                run_id=str(payload["run_id"]),
                run_record_path=PATHS.repository_relative(s5_run),
                run_record_sha256=sha256_file(s5_run),
            )
        ]
    input_paths = [
        path
        for path in (
            remap_dir / "proposal.json",
            remap_dir / "approval.yaml",
            remap_dir / "execution.json",
            remap_dir / "analysis.json",
        )
        if path.is_file()
    ]
    run_identity = {
        "question": args.question,
        "field": args.finalize,
        "trace": sha256_file(trace_path),
    }
    record = RunRecord(
        run_id=f"s6-{hash_canonical(run_identity)[:16]}",
        question_id=args.question,
        stage="s6",
        stage_version="1",
        status="complete",
        started_at=started_at,
        completed_at=completed_at,
        code_version=code_version(PATHS.root),
        command_argv=[
            "scripts/s6_remap.py",
            "--question",
            args.question,
            "--finalize",
            args.finalize,
        ],
        config_path=PATHS.repository_relative(PATHS.config_path(args.question)),
        config_sha256=config_sha256(config),
        prompt_path=None,
        prompt_sha256=None,
        schema_path=None,
        schema_sha256=None,
        cfghash=None,
        upstream=upstream,
        inputs=[artifact_ref(path, root=PATHS.root) for path in input_paths],
        outputs=[artifact_ref(trace_path, root=PATHS.root)],
        external_result_ids={},
        counts={"primary_findings": len(_primary(config, processed_dir))},
        warnings=[],
    )
    # s5 and s6 share the analysis artifact directory in the current foundation paths;
    # keep the s6 operational run nested so the frozen s5 run remains addressable.
    write_run_record(remap_dir / "run.json", record, force=args.force)


def _finalize(
    args: argparse.Namespace, config: Any, processed_dir: Path, analysis_dir: Path
) -> int:
    field = _field(args.finalize)
    remap_dir = analysis_dir / "remap" / field
    proposal_path = remap_dir / "proposal.json"
    approval_path = remap_dir / "approval.yaml"
    proposal = _json(proposal_path)
    try:
        approval = _yaml(approval_path)
        approval_state = validate_approval(proposal, approval)
    except RemapContractError as exc:
        if str(exc) != "human_approval_unavailable":
            raise
        approval_state = "unavailable"
        approval = approval_template(proposal)
    started_at = datetime.now(UTC)
    if approval_state == "unavailable":
        trace = not_run_trace("human_approval_unavailable")
    elif approval_state == "rejected":
        trace = not_run_trace("human_rejected")
    else:
        execution_path = remap_dir / "execution.json"
        analysis_path = remap_dir / "analysis.json"
        execution = _json(execution_path)
        analysis = _json(analysis_path)
        if execution.get("proposal_sha256") != proposal.get("proposal_sha256"):
            raise RemapContractError("trace_execution_proposal_hash_mismatch")
        if analysis.get("proposal_sha256") != proposal.get("proposal_sha256"):
            raise RemapContractError("trace_analysis_proposal_hash_mismatch")
        trace_body = {
            "trace_version": "1",
            "status": "complete",
            "proposal": proposal,
            "approval": approval,
            "execution": {key: value for key, value in execution.items() if key != "side_table"},
            "analysis": analysis,
            "decision": analysis["decision"],
        }
        existing_timestamps: dict[str, Any] | None = None
        existing_trace_path = analysis_dir / "trace.json"
        if existing_trace_path.is_file():
            existing_trace = _json(existing_trace_path)
            timestamps = existing_trace.get("timestamps")
            if {
                key: value for key, value in existing_trace.items() if key != "timestamps"
            } == trace_body and isinstance(timestamps, dict):
                existing_timestamps = dict(timestamps)
        trace = {
            **trace_body,
            "timestamps": existing_timestamps
            or {
                "proposed_at": _mtime(proposal_path),
                "approved_at": _mtime(approval_path),
                "executed_at": _mtime(execution_path),
                "analyzed_at": _mtime(analysis_path),
                "finalized_at": datetime.now(UTC).isoformat(),
            },
        }
    trace_path = analysis_dir / "trace.json"
    atomic_write_json(trace_path, trace, force=args.force)
    completed_at = datetime.now(UTC)
    _finalize_run_record(
        args=args,
        config=config,
        processed_dir=processed_dir,
        analysis_dir=analysis_dir,
        remap_dir=remap_dir,
        trace_path=trace_path,
        started_at=started_at,
        completed_at=completed_at,
    )
    outcome = trace["decision"] if "decision" in trace else trace["reason"]
    print(f"s6 finalized {args.question}/{field}: {outcome}")
    return 0


def run(args: argparse.Namespace) -> int:
    config = load_config_for_question(args.question, require_locked=True)
    config.authorize_stage("s6", explicit_fixture=args.fixture, live_provider=False)
    processed_dir = args.processed_dir or PATHS.processed_dir(args.question)
    analysis_dir = args.analysis_dir or PATHS.analysis_dir(args.question)
    if args.propose_only:
        return _propose(args, config, analysis_dir)
    if args.execute_approved:
        return _execute(args, config, processed_dir, analysis_dir)
    return _finalize(args, config, processed_dir, analysis_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--proposal-response", type=Path)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--field")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--propose-only", action="store_true")
    mode.add_argument("--execute-approved", action="store_true")
    mode.add_argument("--finalize", metavar="FIELD")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.execute_approved and not args.field:
            raise RemapContractError("--execute-approved requires --field")
        if not args.execute_approved and args.field:
            raise RemapContractError("--field is valid only with --execute-approved")
        return run(args)
    except (RemapContractError, ValueError, OSError) as exc:
        print(f"s6 failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
