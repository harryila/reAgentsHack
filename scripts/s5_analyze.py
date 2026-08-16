#!/usr/bin/env python3
"""Run or explicitly freeze the deterministic s5 scientific analysis stage."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from literature_multiverse.analysis import (
    JSON_ARTIFACT_NAMES,
    TABLE_ARTIFACT_NAMES,
    AnalysisContractError,
    CheckpointContext,
    analyze_s5,
    derive_primary_cohort,
    finalize_incomplete_s5,
    read_parquet_records,
    write_analysis_bundle,
)
from literature_multiverse.cohort import cohort_sha256
from literature_multiverse.config import (
    config_sha256,
    load_config_for_question,
)
from literature_multiverse.lineage import (
    artifact_ref,
    atomic_write_json,
    canonical_checkpoint_archive_path,
    code_version,
    frozen_run_identity,
    hash_canonical,
    sha256_file,
    write_run_record,
)
from literature_multiverse.models import (
    M4_CHECKPOINT_ADAPTER,
    M4SourceCheckpoint,
    RunRecord,
    UpstreamRef,
    validate_frozen_s5_completion,
)
from literature_multiverse.paths import PATHS
from literature_multiverse.remap import evaluate_remap_candidate

FIXTURE_INTERRUPTION_QID = "fixture-b-incomplete"
FIXTURE_INTERRUPTION_MODE = "after-25-bootstrap"
FIXTURE_INTERRUPTION_EXIT = 75
FIXTURE_FAULT_MARKER = "fixture_fault_injection.json"


class FixtureInjectedInterruption(RuntimeError):
    """Private, explicit fixture control-flow signal carrying the last clean checkpoint."""

    def __init__(self, checkpoint: M4SourceCheckpoint) -> None:
        super().__init__(FIXTURE_INTERRUPTION_MODE)
        self.checkpoint = checkpoint


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AnalysisContractError(f"missing_json:{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisContractError(f"invalid_json:{path}") from exc
    if not isinstance(value, dict):
        raise AnalysisContractError(f"json_root_not_object:{path}")
    return value


def _all_valid_rows(findings: list[dict[str, Any]], primary_family: str) -> list[dict[str, Any]]:
    return [
        row
        for row in findings
        if row.get("outcome_family") == primary_family
        and row.get("effect_direction") in {"increase", "no_effect", "decrease"}
    ]


def _upstream(processed_dir: Path, *, fixture: bool) -> list[UpstreamRef]:
    path = processed_dir / "run.json"
    if not path.is_file():
        if fixture:
            return []
        raise AnalysisContractError(f"missing_s4_run_record:{path}")
    payload = _json(path)
    if payload.get("stage") != "s4" or payload.get("status") != "complete":
        raise AnalysisContractError("s4_upstream_not_complete")
    return [
        UpstreamRef(
            stage="s4",
            run_id=str(payload["run_id"]),
            run_record_path=PATHS.repository_relative(path),
            run_record_sha256=sha256_file(path),
        )
    ]


def _checkpoint_writer(checkpoint: M4SourceCheckpoint) -> None:
    path = canonical_checkpoint_archive_path(PATHS, checkpoint)
    if path.exists():
        existing = M4SourceCheckpoint.model_validate(_json(path))
        if existing != checkpoint:
            raise AnalysisContractError(f"checkpoint_path_collision:{path}")
        return
    atomic_write_json(path, checkpoint)


def _fixture_fault_mode(args: argparse.Namespace, processed_dir: Path) -> str | None:
    """Validate the exact marker/CLI handshake required for fixture-only interruption."""

    mode = args.fixture_fault_injection
    if mode is None:
        return None
    if not args.fixture:
        raise AnalysisContractError("fixture_fault_injection_requires_explicit_fixture")
    if args.question != FIXTURE_INTERRUPTION_QID:
        raise AnalysisContractError("fixture_fault_injection_wrong_question")
    if args.finalize_incomplete_from is not None or args.with_remap is not None:
        raise AnalysisContractError("fixture_fault_injection_requires_ordinary_s5")
    marker = _json(processed_dir / FIXTURE_FAULT_MARKER)
    expected = {
        "fixture_fault_injection_version": "1",
        "question_id": FIXTURE_INTERRUPTION_QID,
        "mode": FIXTURE_INTERRUPTION_MODE,
    }
    if marker != expected or mode != FIXTURE_INTERRUPTION_MODE:
        raise AnalysisContractError("fixture_fault_injection_marker_mismatch")
    return mode


def _checkpoint_writer_with_fixture_interrupt(
    mode: str | None,
) -> Callable[[M4SourceCheckpoint], None]:
    """Archive every checkpoint, then interrupt only at the registered fixture boundary."""

    def writer(checkpoint: M4SourceCheckpoint) -> None:
        _checkpoint_writer(checkpoint)
        if mode == FIXTURE_INTERRUPTION_MODE and len(checkpoint.completed_bootstrap_indices) == 25:
            raise FixtureInjectedInterruption(checkpoint)

    return writer


def _assert_fixture_interruption_output_is_clean(analysis_dir: Path, *, force: bool) -> None:
    scientific_names = {*JSON_ARTIFACT_NAMES, *TABLE_ARTIFACT_NAMES, "trace.json"}
    stale = sorted(name for name in scientific_names if (analysis_dir / name).exists())
    if stale:
        raise AnalysisContractError(
            "fixture_fault_injection_scientific_outputs_exist:" + ",".join(stale)
        )
    if (analysis_dir / "run.json").exists() and not force:
        raise AnalysisContractError("fixture_fault_injection_run_record_exists")


def _write_partial_fixture_run(
    *,
    args: argparse.Namespace,
    checkpoint: M4SourceCheckpoint,
    processed_dir: Path,
    analysis_dir: Path,
    config_path: Path,
    primary: list[dict[str, Any]],
) -> None:
    checkpoint_path = canonical_checkpoint_archive_path(PATHS, checkpoint)
    inputs = [
        artifact_ref(processed_dir / name, root=PATHS.root)
        for name in (
            "papers.parquet",
            "findings.parquet",
            "g3_gate.json",
            "verification.json",
            FIXTURE_FAULT_MARKER,
        )
    ]
    record = RunRecord(
        run_id=checkpoint.source_run_id,
        question_id=args.question,
        stage="s5",
        stage_version="1",
        status="partial",
        completion_mode="normal",
        checkpoint_sha256=None,
        started_at=checkpoint.source_started_at,
        completed_at=checkpoint.checkpointed_at,
        code_version=checkpoint.code_version,
        command_argv=[
            "scripts/s5_analyze.py",
            "--question",
            args.question,
            "--fixture",
            "--fixture-fault-injection",
            FIXTURE_INTERRUPTION_MODE,
        ],
        config_path=PATHS.repository_relative(config_path),
        config_sha256=checkpoint.config_sha256,
        prompt_path=None,
        prompt_sha256=None,
        schema_path=None,
        schema_sha256=None,
        cfghash=None,
        upstream=_upstream(processed_dir, fixture=True),
        inputs=inputs,
        outputs=[artifact_ref(checkpoint_path, root=PATHS.root)],
        external_result_ids={},
        counts={
            "primary_findings": len(primary),
            "primary_papers": len({row["paper_id"] for row in primary}),
            "completed_bootstrap_draws": len(checkpoint.completed_bootstrap_indices),
            "completed_permutation_attempts": len(checkpoint.completed_permutation_attempt_indices),
            "successful_permutations": len(checkpoint.successful_permutation_indices),
        },
        warnings=["fixture_injected_after_25_bootstrap"],
    )
    write_run_record(analysis_dir / "run.json", record, force=args.force)


def _write_run(
    *,
    args: argparse.Namespace,
    bundle: Any,
    written: dict[str, Path],
    processed_dir: Path,
    analysis_dir: Path,
    config_path: Path,
    cfg_sha: str,
    current_code: str,
    started_at: datetime,
    completed_at: datetime,
    run_id: str,
    command_argv: list[str],
    checkpoint_sha256: str | None,
    completion_mode: str,
) -> None:
    inputs = [
        artifact_ref(processed_dir / name, root=PATHS.root)
        for name in ("papers.parquet", "findings.parquet", "g3_gate.json", "verification.json")
    ]
    outputs = []
    for name, path in sorted(written.items()):
        rows = len(bundle.table_artifacts[name]) if name in bundle.table_artifacts else None
        outputs.append(artifact_ref(path, root=PATHS.root, rows=rows))
    record = RunRecord(
        run_id=run_id,
        question_id=args.question,
        stage="s5",
        stage_version="1",
        status="complete",
        completion_mode=completion_mode,
        checkpoint_sha256=checkpoint_sha256,
        started_at=started_at,
        completed_at=completed_at,
        code_version=current_code,
        command_argv=command_argv,
        config_path=PATHS.repository_relative(config_path),
        config_sha256=cfg_sha,
        prompt_path=None,
        prompt_sha256=None,
        schema_path=None,
        schema_sha256=None,
        cfghash=None,
        upstream=_upstream(processed_dir, fixture=args.fixture),
        inputs=inputs,
        outputs=outputs,
        external_result_ids={},
        counts={
            "primary_findings": len(bundle.primary_rows),
            "primary_papers": len({row["paper_id"] for row in bundle.primary_rows}),
            "moderators": len(bundle.table_artifacts["moderators.parquet"]),
            "contradictions": len(bundle.table_artifacts["contradictions.parquet"]),
            "evidence_gap_cells": len(bundle.table_artifacts["evidence_gaps.parquet"]),
        },
        warnings=[],
    )
    checkpoint_model = bundle.json_artifacts["m4_checkpoint.json"]
    validate_frozen_s5_completion(
        record,
        M4_CHECKPOINT_ADAPTER.validate_python(checkpoint_model),
        m4_gate_status=bundle.json_artifacts["m4_gate.json"]["status"],
    )
    write_run_record(analysis_dir / "run.json", record, force=args.force)


def _run_remap_analysis(args: argparse.Namespace, config: Any) -> int:
    processed_dir = args.processed_dir or PATHS.processed_dir(args.question)
    analysis_dir = args.analysis_dir or PATHS.analysis_dir(args.question)
    field = args.with_remap
    remap_dir = analysis_dir / "remap" / field
    headline = _json(analysis_dir / "headline.json")
    proposal = _json(remap_dir / "proposal.json")
    execution = _json(remap_dir / "execution.json")
    papers = read_parquet_records(processed_dir / "papers.parquet")
    findings = read_parquet_records(processed_dir / "findings.parquet")
    verification = _json(processed_dir / "verification.json")
    assert config.outcomes.primary_family is not None
    primary = derive_primary_cohort(
        papers,
        findings,
        verification,
        primary_family=config.outcomes.primary_family,
    )
    result = evaluate_remap_candidate(
        proposal=proposal,
        execution=execution,
        primary_rows=primary,
        frozen_headline=headline,
        all_valid_rows=_all_valid_rows(findings, config.outcomes.primary_family),
        seed=config.analysis.seed,
        max_folds=config.analysis.cv_max_folds,
    )
    atomic_write_json(remap_dir / "analysis.json", result, force=args.force)
    print(f"s5 remap analysis {args.question}/{field}: {result['decision']}")
    return 0


def run(args: argparse.Namespace) -> int:
    config = load_config_for_question(args.question, require_locked=True)
    config.authorize_stage("s5", explicit_fixture=args.fixture, live_provider=False)
    processed_dir = args.processed_dir or PATHS.processed_dir(args.question)
    analysis_dir = args.analysis_dir or PATHS.analysis_dir(args.question)
    fault_mode = _fixture_fault_mode(args, processed_dir)
    if fault_mode is not None:
        _assert_fixture_interruption_output_is_clean(analysis_dir, force=args.force)
    if args.with_remap is not None:
        return _run_remap_analysis(args, config)

    config_path = PATHS.config_path(args.question)
    papers_path = processed_dir / "papers.parquet"
    findings_path = processed_dir / "findings.parquet"
    g3_path = processed_dir / "g3_gate.json"
    verification_path = processed_dir / "verification.json"
    papers = read_parquet_records(papers_path)
    findings = read_parquet_records(findings_path)
    g3_gate = _json(g3_path)
    verification = _json(verification_path)
    assert config.outcomes.primary_family is not None
    primary = derive_primary_cohort(
        papers,
        findings,
        verification,
        primary_family=config.outcomes.primary_family,
    )
    cfg_sha = config_sha256(config)
    current_code = code_version(PATHS.root)
    cohort_sha = cohort_sha256(primary)
    g3_sha = sha256_file(g3_path)
    input_hashes = {
        "papers": sha256_file(papers_path),
        "findings": sha256_file(findings_path),
        "verification": sha256_file(verification_path),
    }
    started_at = datetime.now(UTC)

    if args.finalize_incomplete_from is not None:
        checkpoint_path = args.finalize_incomplete_from.resolve()
        try:
            checkpoint_path.relative_to(PATHS.root)
        except ValueError as exc:
            raise AnalysisContractError("checkpoint_path_outside_repository") from exc
        checkpoint = M4SourceCheckpoint.model_validate(_json(checkpoint_path))
        canonical_path = canonical_checkpoint_archive_path(PATHS, checkpoint)
        if (
            checkpoint_path.name.endswith(".json")
            and len(checkpoint_path.stem) == 64
            and checkpoint_path != canonical_path
        ):
            raise AnalysisContractError("checkpoint_canonical_archive_path_mismatch")
        if canonical_path.exists():
            archived = M4SourceCheckpoint.model_validate(_json(canonical_path))
            if archived != checkpoint:
                raise AnalysisContractError("checkpoint_archive_collision")
        else:
            atomic_write_json(canonical_path, checkpoint)
        bundle = finalize_incomplete_s5(
            checkpoint=checkpoint,
            config=config,
            papers=papers,
            findings=findings,
            verification=verification,
            g3_gate=g3_gate,
            expected_config_sha256=cfg_sha,
            expected_code_version=current_code,
            expected_cohort_sha256=cohort_sha,
            expected_g3_gate_sha256=g3_sha,
            expected_input_hashes=input_hashes,
        )
        identity = frozen_run_identity(PATHS, checkpoint)
        completed_at = identity["completed_at"]
        started_at = identity["started_at"]
        run_id = identity["run_id"]
        command_argv = identity["command_argv"]
    else:
        run_identity = {
            "qid": args.question,
            "cohort": cohort_sha,
            "code": current_code,
            "g3": g3_sha,
        }
        run_id = f"s5-{hash_canonical(run_identity)[:16]}"
        context = CheckpointContext(
            source_run_id=run_id,
            source_started_at=started_at,
            question_id=args.question,
            config_sha256=cfg_sha,
            code_version=current_code,
            cohort_sha256=cohort_sha,
            g3_gate_sha256=g3_sha,
            input_hashes=input_hashes,
            seed=config.analysis.seed,
            bootstrap_count=config.analysis.bootstrap_count,
            permutation_success_count=config.analysis.permutation_count,
            permutation_max_attempts=125,
            checkpointed_at=lambda: datetime.now(UTC),
            writer=_checkpoint_writer_with_fixture_interrupt(fault_mode),
        )
        try:
            bundle = analyze_s5(
                config=config,
                papers=papers,
                findings=findings,
                verification=verification,
                g3_gate=g3_gate,
                all_valid_rows=_all_valid_rows(findings, config.outcomes.primary_family),
                checkpoint_context=context,
            )
        except FixtureInjectedInterruption as exc:
            _write_partial_fixture_run(
                args=args,
                checkpoint=exc.checkpoint,
                processed_dir=processed_dir,
                analysis_dir=analysis_dir,
                config_path=config_path,
                primary=primary,
            )
            print(
                "s5 fixture interruption after 25 bootstrap draws; "
                f"checkpoint={canonical_checkpoint_archive_path(PATHS, exc.checkpoint)}",
                file=sys.stderr,
            )
            return FIXTURE_INTERRUPTION_EXIT
        completed_at = datetime.now(UTC)
        command_argv = ["scripts/s5_analyze.py", "--question", args.question]
        if args.fixture:
            command_argv.append("--fixture")
    written = write_analysis_bundle(bundle, analysis_dir, force=args.force)
    _write_run(
        args=args,
        bundle=bundle,
        written=written,
        processed_dir=processed_dir,
        analysis_dir=analysis_dir,
        config_path=config_path,
        cfg_sha=cfg_sha,
        current_code=current_code,
        started_at=started_at,
        completed_at=completed_at,
        run_id=run_id,
        command_argv=command_argv,
        checkpoint_sha256=bundle.checkpoint_sha256,
        completion_mode=bundle.completion_mode,
    )
    variant = bundle.json_artifacts["headline.json"]["narrative_variant"]
    print(f"s5 complete {args.question}: Variant {variant} -> {analysis_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument(
        "--fixture-fault-injection",
        choices=(FIXTURE_INTERRUPTION_MODE,),
        help="fixture-only controlled interruption used by the incomplete-M4 pipeline test",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--finalize-incomplete-from", type=Path)
    mode.add_argument("--with-remap", metavar="FIELD")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (AnalysisContractError, ValueError, OSError) as exc:
        print(f"s5 failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
