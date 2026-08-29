#!/usr/bin/env python3
"""Operate one locked, compare-and-swap verifier audit workspace.

This entry point is intentionally standalone.  It does not alter the established
``lm`` command or the immutable verifier-v5 component closure.  Adjudication and
checkpoint receipt paths are passed unopened to the transactional runtime, which
first acquires the outer lock, validates the caller's predecessor expectation, and
replays the complete verifier state before reading any cost- or outcome-bearing
bytes.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from literature_multiverse.adaptive_calibration import AdaptiveCalibrationBundle
from literature_multiverse.item_risk_artifacts import ItemRiskScoringRunReceipt
from literature_multiverse.pipeline_fingerprint import PipelineFingerprint
from literature_multiverse.sequential_verification import (
    SequentialStateExpectation,
    SequentialVerificationState,
)
from literature_multiverse.transactional_audit_workspace_v1 import (
    AuditWorkspaceMutationResultV1,
    advance_transactional_audit_workspace_v1,
    checkpoint_transactional_audit_workspace_v1,
    initialize_transactional_audit_workspace_v1,
)
from literature_multiverse.verifier import load_claim_manifest, load_corpus


def _json_value(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}_json_unreadable:{path}") from exc


def _unwrap(payload: Any, key: str) -> Any:
    if isinstance(payload, dict) and key in payload:
        return payload[key]
    return payload


def _adaptive_bundle(path: Path) -> AdaptiveCalibrationBundle:
    payload = _unwrap(
        _json_value(path, label="adaptive_calibration_bundle"),
        "adaptive_calibration_bundle",
    )
    return AdaptiveCalibrationBundle.model_validate(payload)


def _pipeline_fingerprint(path: Path | None) -> PipelineFingerprint | None:
    if path is None:
        return None
    payload = _unwrap(
        _json_value(path, label="pipeline_fingerprint"),
        "pipeline_fingerprint",
    )
    return PipelineFingerprint.model_validate(payload)


def _item_risk_receipt(path: Path | None) -> ItemRiskScoringRunReceipt | None:
    if path is None:
        return None
    payload = _unwrap(
        _json_value(path, label="item_risk_scoring_receipt"),
        "item_risk_scoring_receipt",
    )
    return ItemRiskScoringRunReceipt.model_validate(payload)


def _state(path: Path) -> SequentialVerificationState:
    payload = _unwrap(
        _json_value(path, label="sequential_audit_state"),
        "sequential_audit_state",
    )
    return SequentialVerificationState.model_validate(payload)


def _expectation(path: Path) -> SequentialStateExpectation:
    payload = _unwrap(
        _json_value(path, label="state_expectation"),
        "state_expectation",
    )
    return SequentialStateExpectation.model_validate(payload)


def _aware_timestamp(raw: str | None) -> datetime:
    if raw is None:
        return datetime.now(UTC)
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"timestamp_invalid:{raw}") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp_requires_timezone")
    return value


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Private canonical transactional audit workspace",
    )
    parser.add_argument("--claim", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--budget-minutes", type=float, required=True)
    parser.add_argument("--adaptive-calibration", type=Path, required=True)
    parser.add_argument("--pipeline-fingerprint", type=Path)
    parser.add_argument("--pipeline-root", type=Path)
    parser.add_argument("--item-risk-scoring-receipt", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transactional-audit-workspace-v1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser(
        "init",
        help="Replay and initialize one canonical active adaptive audit state.",
    )
    _add_common_arguments(initialize)
    initialize.add_argument("--state", type=Path, required=True)
    initialize.add_argument(
        "--initialized-at",
        help="Timezone-aware ISO-8601 timestamp",
    )

    checkpoint = subparsers.add_parser(
        "checkpoint",
        help="CAS-commit a typed cumulative active-cost checkpoint.",
    )
    _add_common_arguments(checkpoint)
    checkpoint.add_argument(
        "--expected",
        type=Path,
        required=True,
        help="Caller-frozen predecessor state expectation",
    )
    checkpoint.add_argument(
        "--expected-pointer-sha256",
        required=True,
        help="Caller-frozen canonical predecessor pointer SHA-256",
    )
    checkpoint.add_argument(
        "--checkpoint-receipt",
        type=Path,
        required=True,
        help=("Typed self-hashed receipt; unopened until locked CAS and verifier replay succeed"),
    )

    advance = subparsers.add_parser(
        "advance",
        help=("CAS-resolve, rerun science, then release or auto-select and publish."),
    )
    _add_common_arguments(advance)
    advance.add_argument(
        "--expected",
        type=Path,
        required=True,
        help="Caller-frozen predecessor state expectation",
    )
    advance.add_argument(
        "--expected-pointer-sha256",
        required=True,
        help="Caller-frozen canonical predecessor pointer SHA-256",
    )
    advance.add_argument(
        "--adjudication-receipt",
        type=Path,
        required=True,
        help=("Typed self-hashed receipt; unopened until locked CAS and verifier replay succeed"),
    )
    advance.add_argument(
        "--corrected-corpus",
        type=Path,
        help="Corrected corpus bound by a corrected-disposition receipt",
    )
    return parser


def _common_inputs(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = args.pipeline_root or Path(__file__).resolve().parents[1]
    manifest = load_claim_manifest(args.claim)
    corpus = load_corpus(
        args.corpus,
        legacy_settings=manifest.legacy_adapter,
        repository_root=repository_root,
    )
    return {
        "manifest": manifest,
        "corpus": corpus,
        "budget_minutes": args.budget_minutes,
        "adaptive_calibration_bundle": _adaptive_bundle(args.adaptive_calibration),
        "expected_pipeline_fingerprint": _pipeline_fingerprint(args.pipeline_fingerprint),
        "pipeline_root": args.pipeline_root,
        "item_risk_scoring_receipt": _item_risk_receipt(args.item_risk_scoring_receipt),
    }


def _print_result(
    result: AuditWorkspaceMutationResultV1,
    *,
    workspace: Path,
) -> None:
    pointer = result.pointer
    authorization = pointer.authorization
    print(
        json.dumps(
            {
                "authorization_sha256": (
                    None if authorization is None else authorization.authorization_sha256
                ),
                "certificate_path": (workspace / pointer.certificate_path).as_posix(),
                "certificate_sha256": pointer.certificate_sha256,
                "certificate_status": pointer.certificate_status,
                "expectation_path": (
                    workspace / pointer.generation_path / "state-expectation.json"
                ).as_posix(),
                "expectation_sha256": (pointer.state_expectation.expectation_sha256),
                "generation": pointer.generation,
                "pointer_path": (workspace / "current-pointer.json").as_posix(),
                "pointer_sha256": pointer.pointer_sha256,
                "result_sha256": result.result_sha256,
                "state_path": (workspace / pointer.state_path).as_posix(),
                "state_sha256": pointer.state_expectation.state_sha256,
                "transition_kind": result.transition_kind,
                "workspace_id": pointer.workspace_id,
            },
            sort_keys=True,
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = _common_inputs(args)
    if args.command == "init":
        result = initialize_transactional_audit_workspace_v1(
            workspace=args.workspace,
            state=_state(args.state),
            initialized_at=_aware_timestamp(args.initialized_at),
            **inputs,
        )
    elif args.command == "checkpoint":
        result = checkpoint_transactional_audit_workspace_v1(
            workspace=args.workspace,
            expected=_expectation(args.expected),
            expected_pointer_sha256=args.expected_pointer_sha256,
            receipt_path=args.checkpoint_receipt,
            **inputs,
        )
    else:
        result = advance_transactional_audit_workspace_v1(
            workspace=args.workspace,
            expected=_expectation(args.expected),
            expected_pointer_sha256=args.expected_pointer_sha256,
            receipt_path=args.adjudication_receipt,
            corrected_corpus_path=args.corrected_corpus,
            **inputs,
        )
    _print_result(result, workspace=args.workspace)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
