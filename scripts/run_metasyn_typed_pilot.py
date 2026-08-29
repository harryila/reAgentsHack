#!/usr/bin/env python3
"""Prepare or replay the private, label-blind MetaSyn typed-synthesis pilot."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from literature_multiverse.metasyn_typed_pilot import (
    MetaSynTypedPilotPrepareReceiptV1,
    prepare_metasyn_typed_pilot,
    validate_metasyn_typed_pilot_prepare,
)

DEFAULT_SCREENING_WORK_DIR = Path(
    "data/cache/metasyn/screening-study-v1-final-v2"
)
DEFAULT_REVIEWS_TRAIN = Path("data/cache/metasyn/reviews-train.parquet")
DEFAULT_CORPUS_MANIFEST = Path("configs/benchmarks/metasyn-corpus-c8fa07d.json")
DEFAULT_WORKSPACE = Path("data/cache/metasyn/typed-oracle-pilot-v1")


def _summary(receipt: MetaSynTypedPilotPrepareReceiptV1) -> dict[str, Any]:
    return {
        "directional_agreement_evaluation_eligible": (
            receipt.directional_agreement_evaluation_eligible
        ),
        "official_test_opened": receipt.official_test_opened,
        "permitted_scientific_outputs": receipt.permitted_scientific_outputs,
        "prepare_bundle_sha256": receipt.prepare_bundle_sha256,
        "prepare_receipt_sha256": receipt.prepare_receipt_sha256,
        "reference_fields_unopened": receipt.reference_fields_unopened,
        "release_grade_source_grounding_count": (
            receipt.release_grade_source_grounding_count
        ),
        "selected_component_count": receipt.selected_component_count,
        "selected_paper_count": receipt.selected_paper_count,
        "selected_question_count": receipt.selected_question_count,
        "source_modality_counts": receipt.source_modality_counts,
        "source_strength_counts": receipt.source_strength_counts,
        "status": receipt.status,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="freeze the private 10-question/32-paper calibration-only pilot",
    )
    prepare.add_argument("--screening-work-dir", type=Path, default=DEFAULT_SCREENING_WORK_DIR)
    prepare.add_argument("--reviews-train", type=Path, default=DEFAULT_REVIEWS_TRAIN)
    prepare.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    prepare.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    prepare.add_argument("--force", action="store_true")

    validate = subparsers.add_parser(
        "validate-prepare",
        help="externally rebuild the frozen selection and every source projection",
    )
    validate.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        receipt = prepare_metasyn_typed_pilot(
            repository_root=args.repository_root,
            screening_work_dir=args.screening_work_dir,
            reviews_train_path=args.reviews_train,
            corpus_manifest_path=args.corpus_manifest,
            workspace=args.workspace,
            force=args.force,
        )
    else:
        receipt = validate_metasyn_typed_pilot_prepare(
            repository_root=args.repository_root,
            workspace=args.workspace,
        )
    print(json.dumps(_summary(receipt), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())
