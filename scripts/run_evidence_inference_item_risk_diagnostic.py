#!/usr/bin/env python3
"""Build and validate the non-pristine Evidence Inference item-risk diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.evidence_inference_item_risk import (
    build_public_summary,
    freeze_design,
    materialize_units,
    validate_public_summary_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    design = commands.add_parser(
        "freeze-design",
        help="Freeze the standalone pipeline, label-free score, and bins before row access.",
    )
    design.add_argument("--config", type=Path, required=True)
    design.add_argument("--repository-root", type=Path, default=Path("."))
    design.add_argument("--work-dir", type=Path, required=True)
    design.add_argument("--force", action="store_true")

    materialize = commands.add_parser(
        "materialize-units",
        help="Verify the anchored design, then join the historically opened benchmark labels.",
    )
    materialize.add_argument("--config", type=Path, required=True)
    materialize.add_argument("--repository-root", type=Path, default=Path("."))
    materialize.add_argument("--work-dir", type=Path, required=True)
    materialize.add_argument("--expected-design-receipt-sha256", required=True)
    materialize.add_argument("--force", action="store_true")

    summarize = commands.add_parser(
        "summarize",
        help="Project a validated calibration run to an identifier-free public aggregate.",
    )
    summarize.add_argument("--config", type=Path, required=True)
    summarize.add_argument("--repository-root", type=Path, default=Path("."))
    summarize.add_argument("--work-dir", type=Path, required=True)
    summarize.add_argument("--calibration-run", type=Path, required=True)
    summarize.add_argument("--expected-design-receipt-sha256", required=True)
    summarize.add_argument("--expected-materialization-receipt-sha256", required=True)
    summarize.add_argument("--expected-calibration-run-receipt-sha256", required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--force", action="store_true")

    validate = commands.add_parser(
        "validate-public",
        help="Validate aggregate hashes, counts, bounds, caveats, and redaction without caches.",
    )
    validate.add_argument("--summary", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-design":
        result = freeze_design(
            repository_root=args.repository_root.resolve(),
            config_path=args.config,
            work_dir=args.work_dir,
            force=args.force,
        )
        payload = {
            "stage": "design_frozen_before_private_paired_report_access",
            "design_receipt_sha256": result.receipt_sha256,
            "diagnostic_pipeline_sha256": result.diagnostic_pipeline_sha256,
            "score_model_sha256": result.score_model_sha256,
            "fixed_bins_receipt_sha256": result.fixed_bins.receipt_sha256,
            "sampling_protocol_sha256": result.sampling_protocol_sha256,
            "adjudication_protocol_sha256": result.adjudication_protocol_sha256,
            "shift_detector_id": "not-assessed-retrospective-diagnostic-v1",
            "shift_detector_sha256": result.shift_detector_sha256,
            "paired_report_opened": False,
            "private_row_labels_opened_in_this_stage": False,
        }
    elif args.command == "materialize-units":
        result = materialize_units(
            repository_root=args.repository_root.resolve(),
            config_path=args.config,
            work_dir=args.work_dir,
            expected_design_receipt_sha256=args.expected_design_receipt_sha256,
            force=args.force,
        )
        payload = {
            "stage": "paper_disjoint_units_materialized_after_design_freeze",
            "materialization_receipt_sha256": result.receipt_sha256,
            "development_units": result.development_unit_count,
            "calibration_units": result.calibration_unit_count,
            "feature_labels_used": result.feature_labels_used,
            "labels_historically_opened_before_protocol": (
                result.labels_historically_opened_before_protocol
            ),
        }
    elif args.command == "summarize":
        result = build_public_summary(
            repository_root=args.repository_root.resolve(),
            config_path=args.config,
            work_dir=args.work_dir,
            calibration_run_path=args.calibration_run,
            expected_design_receipt_sha256=args.expected_design_receipt_sha256,
            expected_materialization_receipt_sha256=(args.expected_materialization_receipt_sha256),
            expected_calibration_run_receipt_sha256=(args.expected_calibration_run_receipt_sha256),
            output_path=args.output,
            force=args.force,
        )
        payload = {
            "stage": "identifier_free_public_summary_written",
            "public_summary_sha256": result["public_summary_sha256"],
            "confirmatory_claim_allowed": result["confirmatory_claim_allowed"],
            "release_probability_authority": result["calibration"]["release_probability_authority"],
            "shift_assessment_status": result["shift_assessment"]["status"],
        }
    else:
        result = validate_public_summary_file(args.summary)
        payload = {
            "stage": "public_summary_validated_without_private_cache",
            "public_summary_sha256": result["public_summary_sha256"],
            "status": "valid",
        }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
