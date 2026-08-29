#!/usr/bin/env python3
"""Run the staged, all-or-nothing bounded native Ollama diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.lineage import OutputExistsError, atomic_write_json
from literature_multiverse.local_ollama import LocalOllamaClient
from literature_multiverse.native_bounded_ollama_diagnostic import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_INVENTORY_CONFIG,
    NativeBoundedOllamaDiagnosticError,
    finalize_bounded_diagnostic,
    load_bounded_json_artifact,
    prepare_bounded_input_bundle,
    run_bounded_prediction_stage,
    run_bounded_schema_compatibility_preflight,
    validate_bounded_finalized_artifacts_with_private_replay,
    validate_bounded_input_bundle,
    validate_bounded_public_summary,
    validate_current_bounded_context,
)

DEFAULT_WORKSPACE = Path("data/cache/native-antiox-bounded-v1-final-v1")
DEFAULT_PUBLIC_SUMMARY = Path(
    "artifacts/diagnostics/native-antiox-bounded-ollama/summary.json"
)


def _paths(workspace: Path) -> dict[str, Path]:
    return {
        "input_bundle": workspace / "input-bundle.json",
        "inventory_receipts": workspace / "inventory-receipts",
        "packet_receipts": workspace / "packet-receipts",
        "attempt_intents": workspace / "pre-call-intents",
        "prediction_ledger": workspace / "prediction-ledger.json",
        "preflight": workspace / "schema-preflight",
        "final_dir": workspace / "final",
        "private_report": workspace / "final" / "private-report.json",
    }


def _assert_descendant(path: Path, parent: Path, *, code: str) -> None:
    try:
        path.absolute().resolve(strict=False).relative_to(
            parent.resolve(strict=True)
        )
    except (OSError, ValueError) as exc:
        raise NativeBoundedOllamaDiagnosticError(code) from exc


def _assert_no_symlink_ancestors(path: Path, *, code: str) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.exists() and candidate.is_symlink():
            raise NativeBoundedOllamaDiagnosticError(code)


def _assert_private_workspace(workspace: Path, root: Path) -> None:
    _assert_no_symlink_ancestors(
        workspace, code="native_bounded_workspace_symlink_ancestor_forbidden"
    )
    _assert_descendant(
        workspace,
        root / "data" / "cache",
        code="native_bounded_workspace_must_be_ignored_data_cache",
    )
    gitignore = root / ".gitignore"
    if gitignore.is_symlink() or not gitignore.is_file():
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_gitignore_unavailable"
        )
    rules = {
        line.strip()
        for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if "data/cache/" not in rules and "/data/cache/" not in rules:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_workspace_not_gitignored"
        )


def _assert_public_output(path: Path, root: Path) -> None:
    _assert_no_symlink_ancestors(
        path, code="native_bounded_public_output_symlink_ancestor_forbidden"
    )
    expected = (root / DEFAULT_PUBLIC_SUMMARY).resolve(strict=False)
    if path.absolute().resolve(strict=False) != expected:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_public_output_not_exact_registered_path"
        )


def _assert_fresh_prepare_workspace(workspace: Path) -> None:
    _assert_no_symlink_ancestors(
        workspace, code="native_bounded_workspace_symlink_ancestor_forbidden"
    )
    if workspace.exists() and (
        not workspace.is_dir() or any(workspace.iterdir())
    ):
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_prepare_requires_fresh_workspace"
        )


def _rooted(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _add_workspace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--repository-root", type=Path, default=Path("."))


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)


def _add_anchor(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-input-bundle-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    _add_workspace(prepare)
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)

    preflight = subparsers.add_parser("preflight")
    _add_workspace(preflight)
    _add_runtime(preflight)
    _add_anchor(preflight)

    predict = subparsers.add_parser("predict")
    _add_workspace(predict)
    _add_runtime(predict)
    _add_anchor(predict)
    predict.add_argument("--inventory-limit", type=int)
    predict.add_argument("--packet-limit", type=int)

    finalize = subparsers.add_parser("finalize")
    _add_workspace(finalize)
    _add_anchor(finalize)
    finalize.add_argument(
        "--public-summary-output", type=Path, default=DEFAULT_PUBLIC_SUMMARY
    )

    validate_public = subparsers.add_parser("validate-public")
    validate_public.add_argument("--repository-root", type=Path, default=Path("."))
    validate_public.add_argument(
        "--public-summary", type=Path, default=DEFAULT_PUBLIC_SUMMARY
    )

    validate_private = subparsers.add_parser("validate-private")
    _add_workspace(validate_private)
    _add_anchor(validate_private)
    validate_private.add_argument(
        "--public-summary", type=Path, default=DEFAULT_PUBLIC_SUMMARY
    )
    return parser


def _load_workspace_bundle(paths: dict[str, Path]) -> dict[str, Any]:
    return validate_bounded_input_bundle(
        load_bounded_json_artifact(paths["input_bundle"])
    )


def _write_or_validate_exact(path: Path, payload: dict[str, Any], *, code: str) -> None:
    if path.exists():
        if load_bounded_json_artifact(path) != payload:
            raise NativeBoundedOllamaDiagnosticError(code)
        return
    try:
        atomic_write_json(path, payload, force=False)
    except OutputExistsError:
        if load_bounded_json_artifact(path) != payload:
            raise NativeBoundedOllamaDiagnosticError(code) from None


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    workspace = _rooted(args.workspace, root)
    _assert_private_workspace(workspace, root)
    _assert_fresh_prepare_workspace(workspace)
    bundle = prepare_bounded_input_bundle(
        config_path=_rooted(args.config, root),
        repository_root=root,
    )
    atomic_write_json(_paths(workspace)["input_bundle"], bundle, force=False)
    return {
        "stage": "prepare",
        "status": bundle["status"],
        "source_rows": bundle["source_rows"],
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "next_stage_requires_exact_expected_input_bundle_sha256": True,
        "prediction_calls": 0,
    }


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    workspace = _rooted(args.workspace, root)
    _assert_private_workspace(workspace, root)
    paths = _paths(workspace)
    bundle = validate_current_bounded_context(
        _load_workspace_bundle(paths),
        repository_root=root,
        reverify_source_adapter=False,
    )
    if bundle["input_bundle_sha256"] != args.expected_input_bundle_sha256:
        raise NativeBoundedOllamaDiagnosticError(
            "native_bounded_preflight_input_bundle_freeze_anchor_mismatch"
        )
    client = LocalOllamaClient(args.base_url, timeout_seconds=args.timeout_seconds)
    identity = client.inspect_identity(DEFAULT_INVENTORY_CONFIG)
    receipt = run_bounded_schema_compatibility_preflight(
        client=client,
        identity=identity,
        preflight_dir=paths["preflight"],
        bundle=bundle,
    )
    return {
        "stage": "preflight",
        "status": receipt["status"],
        "synthetic_calls": receipt["synthetic_calls"],
        "preflight_sha256": receipt["preflight_sha256"],
        "bound_to_frozen_bundle_and_required_before_paper_calls": True,
    }


def _predict(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    workspace = _rooted(args.workspace, root)
    _assert_private_workspace(workspace, root)
    paths = _paths(workspace)
    bundle = _load_workspace_bundle(paths)
    client = LocalOllamaClient(args.base_url, timeout_seconds=args.timeout_seconds)
    ledger = run_bounded_prediction_stage(
        input_bundle=bundle,
        inventory_receipts_dir=paths["inventory_receipts"],
        packet_receipts_dir=paths["packet_receipts"],
        attempt_intents_dir=paths["attempt_intents"],
        preflight_dir=paths["preflight"],
        prediction_ledger_path=paths["prediction_ledger"],
        repository_root=root,
        expected_input_bundle_sha256=args.expected_input_bundle_sha256,
        client=client,
        inventory_limit=args.inventory_limit,
        packet_limit=args.packet_limit,
    )
    return {
        "stage": "predict",
        "status": ledger["status"],
        "inventory_receipts": ledger["inventory_receipts"],
        "packet_receipts": ledger["packet_receipts"],
        "all_expected_terminal_receipts_frozen": ledger[
            "all_expected_terminal_receipts_frozen"
        ],
        "prediction_ledger_sha256": ledger["prediction_ledger_sha256"],
        "ambiguous_execution_is_nonresumable": True,
        "generation_retries": 0,
    }


def _finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    workspace = _rooted(args.workspace, root)
    public_output = _rooted(args.public_summary_output, root)
    _assert_private_workspace(workspace, root)
    _assert_public_output(public_output, root)
    paths = _paths(workspace)
    bundle = _load_workspace_bundle(paths)
    ledger = load_bounded_json_artifact(paths["prediction_ledger"])
    private_report, public_summary = finalize_bounded_diagnostic(
        input_bundle=bundle,
        prediction_ledger=ledger,
        inventory_receipts_dir=paths["inventory_receipts"],
        packet_receipts_dir=paths["packet_receipts"],
        attempt_intents_dir=paths["attempt_intents"],
        preflight_dir=paths["preflight"],
        prediction_ledger_path=paths["prediction_ledger"],
        repository_root=root,
        expected_input_bundle_sha256=args.expected_input_bundle_sha256,
    )
    _write_or_validate_exact(
        paths["private_report"],
        private_report,
        code="native_bounded_existing_private_report_mismatch",
    )
    _write_or_validate_exact(
        public_output,
        public_summary,
        code="native_bounded_existing_public_summary_mismatch",
    )
    return {
        "stage": "finalize",
        "status": public_summary["status"],
        "private_report_sha256": private_report["private_report_sha256"],
        "public_summary_sha256": public_summary["summary_sha256"],
        "official_native_v1_estimable_publications": public_summary[
            "official_native_v1_estimable_publications"
        ],
        "extraction_accuracy_reported": False,
        "claim_release_authority": False,
    }


def _validate_public(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    public_path = _rooted(args.public_summary, root)
    _assert_public_output(public_path, root)
    summary = validate_bounded_public_summary(
        load_bounded_json_artifact(public_path),
        repository_root=root,
    )
    return {
        "stage": "validate-public",
        "status": "valid_current_code_config_and_aggregate_shape",
        "summary_sha256": summary["summary_sha256"],
        "empirical_counts_privately_replayed": False,
    }


def _validate_private(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    workspace = _rooted(args.workspace, root)
    public_path = _rooted(args.public_summary, root)
    _assert_private_workspace(workspace, root)
    _assert_public_output(public_path, root)
    paths = _paths(workspace)
    private_report, public_summary = (
        validate_bounded_finalized_artifacts_with_private_replay(
            input_bundle=_load_workspace_bundle(paths),
            prediction_ledger=load_bounded_json_artifact(
                paths["prediction_ledger"]
            ),
            inventory_receipts_dir=paths["inventory_receipts"],
            packet_receipts_dir=paths["packet_receipts"],
            attempt_intents_dir=paths["attempt_intents"],
            preflight_dir=paths["preflight"],
            prediction_ledger_path=paths["prediction_ledger"],
            private_report=load_bounded_json_artifact(paths["private_report"]),
            public_summary=load_bounded_json_artifact(
                public_path
            ),
            repository_root=root,
            expected_input_bundle_sha256=args.expected_input_bundle_sha256,
        )
    )
    return {
        "stage": "validate-private",
        "status": "valid_full_private_receipt_replay",
        "private_report_sha256": private_report["private_report_sha256"],
        "summary_sha256": public_summary["summary_sha256"],
        "empirical_counts_privately_replayed": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = _prepare(args)
    elif args.command == "preflight":
        result = _preflight(args)
    elif args.command == "predict":
        result = _predict(args)
    elif args.command == "finalize":
        result = _finalize(args)
    elif args.command == "validate-public":
        result = _validate_public(args)
    else:
        result = _validate_private(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
