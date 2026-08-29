#!/usr/bin/env python3
"""Run the provider-free Evidence Inference diagnostic and archive replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.evidence_inference_diagnostic import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    build_provider_free_diagnostic_bundle,
    build_public_diagnostic_summary,
    validate_diagnostic_report,
    validate_prediction_ledger,
    validate_public_diagnostic_summary,
)
from literature_multiverse.lineage import OutputExistsError, atomic_write_json

DEFAULT_ARCHIVE_ROOTS = (
    Path("data/cache/gepa"),
    Path("data/cache/gepa-failed-runs"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/cache/evidence-inference-gepa/manifest.json"),
        help="Full immutable Evidence Inference conversion manifest.",
    )
    parser.add_argument(
        "--previously-opened-manifest",
        type=Path,
        default=Path("data/cache/evidence-inference-gepa-low-budget/manifest.json"),
        help="Manifest identifying every test paper touched by earlier provider evaluation.",
    )
    parser.add_argument(
        "--seed-prompt",
        type=Path,
        default=Path("prompts/evidence_inference_extraction.md"),
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        action="append",
        help="GEPA archive root; repeat to override the two local defaults.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/cache/evidence-inference-diagnostic/provider-free-report.json"
        ),
    )
    parser.add_argument(
        "--prediction-ledger-output",
        type=Path,
        help=(
            "Separate redacted prediction ledger. Defaults to prediction-ledger.json "
            "beside --output."
        ),
    )
    parser.add_argument(
        "--public-summary-output",
        type=Path,
        help=(
            "Explicitly write a check-in-safe metadata-only summary, for example "
            "artifacts/diagnostics/evidence-inference/summary.json."
        ),
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive_roots = tuple(args.archive_root or DEFAULT_ARCHIVE_ROOTS)
    ledger_output = args.prediction_ledger_output or args.output.with_name(
        "prediction-ledger.json"
    )
    output_paths = [args.output, ledger_output]
    if args.public_summary_output is not None:
        output_paths.append(args.public_summary_output)
    if not args.force:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            raise OutputExistsError(
                ",".join(path.as_posix() for path in existing)
            )
    report, prediction_ledger = build_provider_free_diagnostic_bundle(
        manifest_path=args.manifest,
        previously_opened_manifest_path=args.previously_opened_manifest,
        seed_prompt_path=args.seed_prompt,
        archive_roots=archive_roots,
        seed=args.bootstrap_seed,
        replicates=args.bootstrap_replicates,
    )
    validate_diagnostic_report(report)
    validate_prediction_ledger(prediction_ledger)
    public_summary = None
    if args.public_summary_output is not None:
        public_summary = build_public_diagnostic_summary(report, prediction_ledger)
        validate_public_diagnostic_summary(public_summary)
    atomic_write_json(ledger_output, prediction_ledger, force=args.force)
    atomic_write_json(args.output, report, force=args.force)
    if args.public_summary_output is not None and public_summary is not None:
        atomic_write_json(args.public_summary_output, public_summary, force=args.force)
    provider_unseen = report[
        "provider_call_unseen_paper_input_only_lexical_diagnostic"
    ]["summary"]
    archived = report["archived_gepa_response_replay"]
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "prediction_ledger_output": ledger_output.as_posix(),
                "prediction_ledger_sha256": prediction_ledger["ledger_sha256"],
                "public_summary_output": (
                    args.public_summary_output.as_posix()
                    if args.public_summary_output is not None
                    else None
                ),
                "public_summary_sha256": (
                    public_summary["public_summary_sha256"]
                    if public_summary is not None
                    else None
                ),
                "report_sha256": report["report_sha256"],
                "status": report["status"],
                "provider_calls_made": report["provider_calls_made"],
                "provider_call_unseen_rows": provider_unseen["rows"],
                "provider_call_unseen_articles": provider_unseen["articles"],
                "provider_call_unseen_direction_accuracy": provider_unseen["metrics"][
                    "direction_accuracy"
                ]["estimate"],
                "provider_call_unseen_joint_validity": provider_unseen["metrics"][
                    "schema_direction_provenance_joint_validity"
                ]["estimate"],
                "eligible_diagnostic_mutations": archived[
                    "eligible_diagnostic_mutation_accounting"
                ]["discovered"],
                "excluded_failed_mutations": archived[
                    "excluded_failed_mutation_accounting"
                ]["discovered"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
