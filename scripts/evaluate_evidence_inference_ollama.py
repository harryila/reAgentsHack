#!/usr/bin/env python3
"""Prepare, predict, and score the staged offline Evidence Inference Ollama diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.evidence_inference_ollama import (
    DEFAULT_GENERATION_CONFIG,
    build_public_summary,
    canonical_json_file_sha256,
    prepare_input_bundle,
    run_prediction_stage,
    score_frozen_predictions,
    validate_input_bundle,
    validate_prediction_ledger,
    validate_private_report,
    validate_public_summary,
)
from literature_multiverse.evidence_inference_ollama_reporting import (
    augment_public_summary,
    validate_augmented_public_summary,
)
from literature_multiverse.lineage import OutputExistsError, atomic_write_json, sha256_file
from literature_multiverse.local_ollama import LocalOllamaClient

DEFAULT_WORKSPACE = Path("data/cache/evidence-inference-ollama")
DEFAULT_MANIFEST = Path("data/cache/evidence-inference-gepa/manifest.json")
DEFAULT_PROVIDER_FREE_REPORT = Path(
    "data/cache/evidence-inference-diagnostic/provider-free-report.json"
)
DEFAULT_LEXICAL_LEDGER = Path(
    "data/cache/evidence-inference-diagnostic/prediction-ledger.json"
)
DEFAULT_PUBLIC_SUMMARY = Path(
    "artifacts/diagnostics/evidence-inference-ollama/summary.json"
)


def _private_workspace_paths(workspace: Path) -> dict[str, Path]:
    return {
        "input_bundle": workspace / "input-bundle.json",
        "receipts": workspace / "receipts",
        "prediction_ledger": workspace / "prediction-ledger.json",
        "private_report": workspace / "private-report.json",
    }


def _assert_ignored_cache_path(path: Path) -> None:
    cache_root = Path("data/cache").resolve()
    try:
        path.resolve().relative_to(cache_root)
    except ValueError as exc:
        raise ValueError("private Ollama artifacts must stay under ignored data/cache/") from exc


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_new_or_force(path: Path, value: object, *, force: bool) -> None:
    atomic_write_json(path, value, force=force)


def _add_shared_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--provider-free-report",
        type=Path,
        default=DEFAULT_PROVIDER_FREE_REPORT,
    )
    parser.add_argument(
        "--lexical-prediction-ledger",
        type=Path,
        default=DEFAULT_LEXICAL_LEDGER,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Freeze input-only provider-call-unseen Results projections.",
    )
    _add_shared_sources(prepare)
    prepare.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    prepare.add_argument("--force", action="store_true")

    predict = subparsers.add_parser(
        "predict",
        help="Run/resume deterministic local Ollama predictions without label access.",
    )
    predict.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    predict.add_argument("--base-url", default="http://127.0.0.1:11434")
    predict.add_argument("--timeout-seconds", type=float, default=300.0)
    predict.add_argument(
        "--limit",
        type=int,
        help="Run at most this many currently missing rows (smoke/resume aid).",
    )
    predict.add_argument(
        "--retry-failures",
        action="store_true",
        help="Replace only validated execution-failure receipts; successful rows stay frozen.",
    )

    score = subparsers.add_parser(
        "score",
        help="Validate the complete prediction freeze, then open labels and score.",
    )
    _add_shared_sources(score)
    score.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    score.add_argument(
        "--public-summary-output",
        type=Path,
        default=DEFAULT_PUBLIC_SUMMARY,
    )
    score.add_argument("--bootstrap-seed", type=int, default=20260827)
    score.add_argument("--bootstrap-replicates", type=int, default=2000)
    score.add_argument("--force", action="store_true")

    run = subparsers.add_parser(
        "run",
        help="Execute prepare, prediction, and scoring in their enforced order.",
    )
    _add_shared_sources(run)
    run.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    run.add_argument("--base-url", default="http://127.0.0.1:11434")
    run.add_argument("--timeout-seconds", type=float, default=300.0)
    run.add_argument("--retry-failures", action="store_true")
    run.add_argument(
        "--public-summary-output",
        type=Path,
        default=DEFAULT_PUBLIC_SUMMARY,
    )
    run.add_argument("--bootstrap-seed", type=int, default=20260827)
    run.add_argument("--bootstrap-replicates", type=int, default=2000)
    run.add_argument("--force", action="store_true")
    return parser


def _prepare(args: argparse.Namespace) -> dict[str, object]:
    paths = _private_workspace_paths(args.workspace)
    _assert_ignored_cache_path(args.workspace)
    bundle = prepare_input_bundle(
        manifest_path=args.manifest,
        provider_free_report_path=args.provider_free_report,
        lexical_prediction_ledger_path=args.lexical_prediction_ledger,
    )
    validate_input_bundle(bundle)
    _write_new_or_force(paths["input_bundle"], bundle, force=args.force)
    return {
        "stage": "prepare",
        "status": bundle["status"],
        "rows": bundle["row_count"],
        "articles": bundle["article_count"],
        "input_bundle_sha256": bundle["input_bundle_sha256"],
        "contains_gold_labels": False,
    }


def _predict(args: argparse.Namespace) -> dict[str, object]:
    paths = _private_workspace_paths(args.workspace)
    _assert_ignored_cache_path(args.workspace)
    bundle = validate_input_bundle(_load_json(paths["input_bundle"]))
    client = LocalOllamaClient(
        args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    ledger = run_prediction_stage(
        input_bundle=bundle,
        receipts_dir=paths["receipts"],
        prediction_ledger_path=paths["prediction_ledger"],
        client=client,
        config=DEFAULT_GENERATION_CONFIG,
        limit=args.limit,
        retry_failures=args.retry_failures,
    )
    validate_prediction_ledger(ledger, bundle=bundle, require_complete=False)
    return {
        "stage": "predict",
        "status": ledger["status"],
        "receipt_count": ledger["receipt_count"],
        "input_row_count": ledger["input_row_count"],
        "prediction_ledger_sha256": ledger["prediction_ledger_sha256"],
        "prediction_stage_received_label_fields": False,
        "external_provider_calls": 0,
        "external_provider_cost_usd": 0.0,
    }


def _score(args: argparse.Namespace) -> dict[str, object]:
    paths = _private_workspace_paths(args.workspace)
    _assert_ignored_cache_path(args.workspace)
    bundle = validate_input_bundle(_load_json(paths["input_bundle"]))
    ledger = validate_prediction_ledger(
        _load_json(paths["prediction_ledger"]),
        bundle=bundle,
        require_complete=True,
    )
    if sha256_file(paths["input_bundle"]) != canonical_json_file_sha256(bundle):
        raise ValueError("input bundle file bytes are not the bound canonical JSON")
    if sha256_file(paths["prediction_ledger"]) != canonical_json_file_sha256(ledger):
        raise ValueError("prediction ledger file bytes are not the bound canonical JSON")
    report = score_frozen_predictions(
        input_bundle=bundle,
        prediction_ledger=ledger,
        receipts_dir=paths["receipts"],
        manifest_path=args.manifest,
        provider_free_report_path=args.provider_free_report,
        lexical_prediction_ledger_path=args.lexical_prediction_ledger,
        seed=args.bootstrap_seed,
        replicates=args.bootstrap_replicates,
    )
    validate_private_report(report)
    public_summary = build_public_summary(report)
    validate_public_summary(public_summary)
    public_summary = augment_public_summary(
        report=report,
        public_summary=public_summary,
        input_bundle=bundle,
        prediction_ledger=ledger,
        receipts_dir=paths["receipts"],
    )
    validate_augmented_public_summary(public_summary, require_current_sources=True)
    _write_new_or_force(paths["private_report"], report, force=args.force)
    _write_new_or_force(
        args.public_summary_output,
        public_summary,
        force=args.force,
    )
    direction = public_summary["paired_comparison"]["metrics"]["direction_accuracy"]
    return {
        "stage": "score",
        "status": report["status"],
        "rows": report["population"]["rows"],
        "articles": report["population"]["articles"],
        "local_ollama_direction_accuracy": direction["local_ollama"]["estimate"],
        "fixed_lexical_direction_accuracy": direction["fixed_lexical"]["estimate"],
        "paired_difference": direction["local_ollama_minus_fixed_lexical"],
        "report_sha256": report["report_sha256"],
        "public_summary_sha256": public_summary["public_summary_sha256"],
        "external_provider_calls": 0,
        "external_provider_cost_usd": 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = _prepare(args)
    elif args.command == "predict":
        result = _predict(args)
    elif args.command == "score":
        result = _score(args)
    else:
        try:
            _prepare(args)
        except OutputExistsError:
            if args.force:
                raise
        _predict(args)
        result = _score(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
