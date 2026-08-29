#!/usr/bin/env python3
"""Run the label-blind 10-question/32-paper bounded MetaSyn yield diagnostic."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from literature_multiverse.local_ollama import LocalOllamaClient
from literature_multiverse.metasyn_bounded_runtime import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_EXECUTION_WORKSPACE,
    DEFAULT_PILOT_WORKSPACE,
    finalize_metasyn_bounded_yield_runtime,
    load_current_metasyn_bounded_execution_bundle,
    prepare_metasyn_bounded_execution_bundle,
    run_metasyn_bounded_prediction_stage,
    run_metasyn_schema_preflight,
    validate_metasyn_bounded_finalized_runtime,
    write_metasyn_bounded_execution_bundle,
)


def _rooted(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _add_workspace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--workspace", type=Path, default=DEFAULT_EXECUTION_WORKSPACE
    )


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)


def _add_anchor(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-execution-bundle-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help=(
            "freeze provider-neutral prompts, the exact 32-row roster, runtime config, "
            "and current pipeline identity without making model calls"
        ),
    )
    _add_workspace(prepare)
    prepare.add_argument(
        "--pilot-workspace", type=Path, default=DEFAULT_PILOT_WORKSPACE
    )
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)

    validate_bundle = subparsers.add_parser(
        "validate-bundle",
        help="externally replay the prepare corpus, prompts, config, roster, and code closure",
    )
    _add_workspace(validate_bundle)

    preflight = subparsers.add_parser(
        "preflight",
        help=(
            "run eight source-free compact-provider schema compatibility calls before any "
            "source-bearing request; returned JSON must pass the bound full v2 schema"
        ),
        description=(
            "Run all three inventory states and five completed-effect synthetic "
            "compatibility calls "
            "through the compact provider schema and full v2 acceptance stack. After "
            "independent raw, typed, and preservation validation, fixtures must be "
            "canonically equal; raw differences may only omit proven Pydantic-declared "
            "defaults. This proves whole-request compatibility only, not provider keyword "
            "enforcement, production context compilation, extraction yield, or scientific "
            "validity."
        ),
    )
    _add_workspace(preflight)
    _add_runtime(preflight)

    predict = subparsers.add_parser(
        "predict",
        help=(
            "run/resume one-shot source requests; ambiguity poisons only the frozen "
            "attempt and can never be retried"
        ),
    )
    _add_workspace(predict)
    _add_runtime(predict)
    _add_anchor(predict)
    predict.add_argument("--inventory-limit", type=int)
    predict.add_argument("--packet-limit", type=int)

    finalize = subparsers.add_parser(
        "finalize",
        help=(
            "freeze the private full-roster yield report; derive but do not write/register "
            "the aggregate-only public summary"
        ),
        description=(
            "Freeze the private full-roster yield report. Derive but do not write/register "
            "the aggregate-only public summary."
        ),
    )
    _add_workspace(finalize)
    _add_anchor(finalize)

    validate_final = subparsers.add_parser(
        "validate-final",
        help="externally replay every private receipt, incident, ledger row, and yield count",
    )
    _add_workspace(validate_final)
    _add_anchor(validate_final)
    return parser


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    bundle = prepare_metasyn_bounded_execution_bundle(
        repository_root=root,
        pilot_workspace=_rooted(args.pilot_workspace, root),
        config_path=_rooted(args.config, root),
    )
    output = write_metasyn_bounded_execution_bundle(
        execution_bundle=bundle,
        workspace=_rooted(args.workspace, root),
        repository_root=root,
    )
    return {
        "stage": "prepare",
        "status": bundle.status,
        "execution_bundle_path": output.relative_to(root).as_posix(),
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
        "native_schema_v2_contract_sha256": (
            bundle.native_schema_v2_contract_sha256
        ),
        "provider_grammar_scope_sha256": bundle.provider_grammar_scope_sha256,
        "schema_v2_preflight_fingerprint": bundle.schema_v2_preflight_fingerprint,
        "question_count": bundle.question_count,
        "component_count": bundle.component_count,
        "publication_count": bundle.publication_count,
        "reference_fields_unopened": bundle.reference_fields_unopened,
        "model_calls_made": bundle.model_calls_made,
    }


def _validate_bundle(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    _, bundle = load_current_metasyn_bounded_execution_bundle(
        workspace=_rooted(args.workspace, root),
        repository_root=root,
        external_replay=True,
    )
    return {
        "stage": "validate-bundle",
        "status": "valid_current_external_replay",
        "execution_bundle_sha256": bundle.execution_bundle_sha256,
        "runtime_pipeline_sha256": bundle.runtime_pipeline_sha256,
        "native_schema_v2_contract_sha256": (
            bundle.native_schema_v2_contract_sha256
        ),
        "provider_grammar_scope_sha256": bundle.provider_grammar_scope_sha256,
        "schema_v2_preflight_fingerprint": bundle.schema_v2_preflight_fingerprint,
        "publication_count": bundle.publication_count,
        "reference_fields_unopened": True,
        "model_calls_made": False,
    }


def _client(args: argparse.Namespace) -> LocalOllamaClient:
    return LocalOllamaClient(
        args.base_url, timeout_seconds=args.timeout_seconds
    )


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    receipt = run_metasyn_schema_preflight(
        workspace=_rooted(args.workspace, root),
        repository_root=root,
        client=_client(args),
    )
    return {
        "stage": "preflight",
        "status": receipt["status"],
        "preflight_sha256": receipt["preflight_sha256"],
        "structural_skeleton_count": receipt["structural_skeleton_count"],
        "synthetic_generation_call_attempts": receipt[
            "synthetic_generation_call_attempts"
        ],
        "fixture_comparison_version": receipt["fixture_comparison_version"],
        "fixture_comparison_mode": receipt["fixture_comparison_mode"],
        "raw_fixture_equal_call_count": receipt[
            "raw_fixture_equal_call_count"
        ],
        "canonical_fixture_equal_call_count": receipt[
            "canonical_fixture_equal_call_count"
        ],
        "declared_default_omission_call_count": receipt[
            "declared_default_omission_call_count"
        ],
        "omitted_declared_default_path_count": receipt[
            "omitted_declared_default_path_count"
        ],
        "whole_request_compatibility_only": receipt[
            "whole_request_compatibility_only"
        ],
        "provider_keyword_enforcement_validated": receipt[
            "provider_keyword_enforcement_validated"
        ],
        "production_context_schema_compilation_validated": receipt[
            "production_context_schema_compilation_validated"
        ],
        "production_enum_or_cardinality_compilation_validated": receipt[
            "production_enum_or_cardinality_compilation_validated"
        ],
        "source_bearing_generation_call_attempts": 0,
    }


def _predict(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    ledger = run_metasyn_bounded_prediction_stage(
        workspace=_rooted(args.workspace, root),
        repository_root=root,
        client=_client(args),
        expected_execution_bundle_sha256=(
            args.expected_execution_bundle_sha256
        ),
        inventory_limit=args.inventory_limit,
        packet_limit=args.packet_limit,
    )
    return {
        "stage": "predict",
        "status": ledger.status,
        "prediction_ledger_sha256": ledger.ledger_sha256,
        "terminal_row_count": ledger.terminal_row_count,
        "publication_count": ledger.publication_count,
        "all_rows_terminal": ledger.all_rows_terminal,
        "observed_source_generation_calls": (
            ledger.observed_source_generation_calls
        ),
        "possible_ambiguous_source_generation_calls": (
            ledger.possible_ambiguous_source_generation_calls
        ),
        "generation_retries": ledger.generation_retries,
    }


def _finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    report, summary = finalize_metasyn_bounded_yield_runtime(
        workspace=_rooted(args.workspace, root),
        repository_root=root,
        expected_execution_bundle_sha256=(
            args.expected_execution_bundle_sha256
        ),
    )
    return {
        "stage": "finalize",
        "status": report.status,
        "private_report_sha256": report.report_sha256,
        "aggregate_only_public_summary_sha256": summary.summary_sha256,
        "public_summary_materialized_or_registered": False,
        "typed_publication_output_count": report.typed_publication_output_count,
        "release_grade_typed_publication_count": (
            report.release_grade_typed_publication_count
        ),
        "typed_finding_count": report.typed_finding_count,
        "direction_agreement_reported": False,
        "extraction_accuracy_reported": False,
        "claim_release_authority": False,
    }


def _validate_final(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root.resolve(strict=True)
    report, summary = validate_metasyn_bounded_finalized_runtime(
        workspace=_rooted(args.workspace, root),
        repository_root=root,
        expected_execution_bundle_sha256=(
            args.expected_execution_bundle_sha256
        ),
    )
    return {
        "stage": "validate-final",
        "status": "valid_full_private_external_replay",
        "private_report_sha256": report.report_sha256,
        "aggregate_only_public_summary_sha256": summary.summary_sha256,
        "public_summary_materialized_or_registered": False,
        "publication_count": report.publication_count,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = _prepare(args)
    elif args.command == "validate-bundle":
        result = _validate_bundle(args)
    elif args.command == "preflight":
        result = _preflight(args)
    elif args.command == "predict":
        result = _predict(args)
    elif args.command == "finalize":
        result = _finalize(args)
    else:
        result = _validate_final(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())
