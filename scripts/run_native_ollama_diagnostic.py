#!/usr/bin/env python3
"""Prepare, run, and finalize the provider-free native Antiox Ollama diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.lineage import atomic_write_json
from literature_multiverse.local_ollama import LocalOllamaClient
from literature_multiverse.native_ollama_diagnostic import (
    DEFAULT_GENERATION_CONFIG,
    finalize_diagnostic,
    prepare_input_bundle,
    run_generation_schema_compatibility_preflight,
    run_prediction_stage,
    validate_current_diagnostic_context,
    validate_input_bundle,
    validate_prediction_ledger,
    validate_public_summary,
)

DEFAULT_CONFIG = Path("configs/benchmarks/native-antiox-ollama-v1.json")
DEFAULT_WORKSPACE = Path("data/cache/native-antiox-ollama-v2-final-v1")
DEFAULT_PUBLIC_SUMMARY = Path("artifacts/diagnostics/native-antiox-ollama/summary.json")


def _paths(workspace: Path) -> dict[str, Path]:
    return {
        "input_bundle": workspace / "input-bundle.json",
        "receipts": workspace / "generation-receipts",
        "prediction_ledger": workspace / "prediction-ledger.json",
        "final": workspace / "final",
        "private_report": workspace / "final" / "private-report.json",
    }


def _assert_descendant(path: Path, parent: Path, *, error: str) -> None:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError as exc:
        raise ValueError(error) from exc


def _assert_private_workspace(workspace: Path, repository_root: Path) -> None:
    _assert_descendant(
        workspace,
        repository_root / "data" / "cache",
        error="private native Ollama artifacts must stay under ignored data/cache/",
    )
    gitignore = repository_root / ".gitignore"
    if gitignore.is_symlink() or not gitignore.is_file():
        raise ValueError("repository .gitignore is unavailable for private artifact check")
    ignore_rules = {
        line.strip()
        for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if "data/cache/" not in ignore_rules and "/data/cache/" not in ignore_rules:
        raise ValueError("private native Ollama workspace is not covered by .gitignore")


def _assert_public_output(path: Path, repository_root: Path) -> None:
    _assert_descendant(
        path,
        repository_root / "artifacts" / "diagnostics",
        error="public native Ollama summary must stay under artifacts/diagnostics/",
    )


def _assert_prepare_workspace_has_no_predictions(workspace: Path) -> None:
    paths = _paths(workspace)
    contaminated = [
        path
        for path in (
            paths["prediction_ledger"],
            paths["receipts"],
            paths["final"],
        )
        if path.exists()
    ]
    if contaminated:
        rendered = ",".join(sorted(path.as_posix() for path in contaminated))
        raise ValueError(
            "native prepare workspace already contains prediction/final artifacts; "
            f"choose a fresh ignored workspace: {rendered}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _add_workspace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--repository-root", type=Path, default=Path("."))


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)


def _add_finalize(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--public-summary-output", type=Path, default=DEFAULT_PUBLIC_SUMMARY)
    parser.add_argument("--budget-minutes", type=float, default=60.0)
    parser.add_argument("--force", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Freeze the exact label-blind 19-record source-line bundle.",
    )
    _add_workspace(prepare)
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--force", action="store_true")

    predict = subparsers.add_parser(
        "predict",
        help="Run/resume terminal local-Ollama receipts without source/label access.",
    )
    _add_workspace(predict)
    _add_runtime(predict)
    predict.add_argument(
        "--limit",
        type=int,
        help="Run at most this many currently missing rows (smoke/resume aid).",
    )

    finalize = subparsers.add_parser(
        "finalize",
        help="Validate the complete freeze, replay v4 grounding, and abstain.",
    )
    _add_workspace(finalize)
    _add_finalize(finalize)

    run = subparsers.add_parser(
        "run",
        help="Execute prepare, all local generations, and finalization in order.",
    )
    _add_workspace(run)
    _add_runtime(run)
    _add_finalize(run)
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    _assert_private_workspace(args.workspace, root)
    _assert_prepare_workspace_has_no_predictions(args.workspace)
    paths = _paths(args.workspace)
    bundle = prepare_input_bundle(
        config_path=args.config,
        repository_root=root,
    )
    atomic_write_json(paths["input_bundle"], bundle, force=args.force)
    return {
        "stage": "prepare",
        "status": bundle["status"],
        "rows": bundle["row_count"],
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "selection_scope": bundle["selection_scope"],
        "contains_legacy_findings": False,
        "contains_legacy_directions": False,
        "contains_anchor_expectations": False,
        "contains_downstream_claim_payload": False,
    }


def _predict(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    _assert_private_workspace(args.workspace, root)
    paths = _paths(args.workspace)
    bundle = validate_input_bundle(_load_json(paths["input_bundle"]))
    validate_current_diagnostic_context(bundle, repository_root=root)
    client = LocalOllamaClient(args.base_url, timeout_seconds=args.timeout_seconds)
    schema_preflight = run_generation_schema_compatibility_preflight(
        client=client,
        config=DEFAULT_GENERATION_CONFIG,
    )
    ledger = run_prediction_stage(
        input_bundle=bundle,
        receipts_dir=paths["receipts"],
        prediction_ledger_path=paths["prediction_ledger"],
        client=client,
        config=DEFAULT_GENERATION_CONFIG,
        limit=getattr(args, "limit", None),
    )
    validate_prediction_ledger(ledger, bundle=bundle, require_complete=False)
    return {
        "stage": "predict",
        "status": ledger["status"],
        "receipts": ledger["receipt_count"],
        "expected_receipts": ledger["input_row_count"],
        "status_counts": ledger["status_counts"],
        "prediction_ledger_sha256": ledger["prediction_ledger_sha256"],
        "prediction_stage_opened_source_or_label_files": False,
        "prediction_stage_received_downstream_claim_payload": False,
        "schema_compatibility_preflight": schema_preflight,
        "external_provider_calls": 0,
        "external_provider_cost_usd": 0.0,
    }


def _finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    _assert_private_workspace(args.workspace, root)
    _assert_public_output(args.public_summary_output, root)
    paths = _paths(args.workspace)
    bundle = validate_input_bundle(_load_json(paths["input_bundle"]))
    ledger = validate_prediction_ledger(
        _load_json(paths["prediction_ledger"]),
        bundle=bundle,
        require_complete=True,
    )
    private, public = finalize_diagnostic(
        input_bundle=bundle,
        prediction_ledger=ledger,
        receipts_dir=paths["receipts"],
        repository_root=root,
        private_output_dir=paths["final"],
        budget_minutes=args.budget_minutes,
        force=args.force,
    )
    validate_public_summary(public)
    atomic_write_json(args.public_summary_output, public, force=args.force)
    return {
        "stage": "finalize",
        "status": private["status"],
        "certificate_status": private["certificate_status"],
        "certificate_blocker_codes": private["certificate_blocker_codes"],
        "grounding_package_sha256": private["grounding_package_sha256"],
        "private_report_sha256": private["private_report_sha256"],
        "public_summary_sha256": public["public_summary_sha256"],
        "accuracy_evaluated": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = _prepare(args)
    elif args.command == "predict":
        result = _predict(args)
    elif args.command == "finalize":
        result = _finalize(args)
    else:
        _prepare(args)
        prediction = _predict(args)
        if prediction["receipts"] != prediction["expected_receipts"]:
            raise ValueError("native Ollama run did not freeze all 19 terminal receipts")
        result = _finalize(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
