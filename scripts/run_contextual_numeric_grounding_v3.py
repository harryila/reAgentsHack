#!/usr/bin/env python3
"""Build or validate the zero-call contextual-grounding v3 feasibility suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.contextual_numeric_grounding_v3 import (
    ContextualGroundingOfflineFeasibilitySuiteV3,
    freeze_contextual_grounding_offline_feasibility_suite_v3,
    validate_contextual_grounding_offline_feasibility_suite_v3,
)
from literature_multiverse.lineage import atomic_write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Externally replay immutable MetaSyn v2 and build or validate the "
            "non-authorizing contextual-grounding v3 offline feasibility suite."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Replay sources and write a new suite.")
    build.add_argument("--repository-root", type=Path, default=Path("."))
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--force", action="store_true")

    validate = subparsers.add_parser(
        "validate", help="Validate a saved suite with full external replay."
    )
    validate.add_argument("--repository-root", type=Path, default=Path("."))
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument(
        "--contract-only",
        action="store_true",
        help="Skip the default immutable-v2 and source-byte external replay.",
    )

    status = subparsers.add_parser(
        "status", help="Print the non-authorizing status of a saved suite."
    )
    status.add_argument("--input", type=Path, required=True)
    return parser


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("contextual_grounding_v3_cli_input_not_object")
    return value


def _status(suite: ContextualGroundingOfflineFeasibilitySuiteV3) -> dict[str, Any]:
    return {
        "suite_version": suite.suite_version,
        "status": suite.status,
        "suite_sha256": suite.suite_sha256,
        "pipeline_sha256": suite.pipeline_sha256,
        "offline_witness_count": suite.offline_witness_count,
        "contextual_grounding_completed_count": (suite.contextual_grounding_completed_count),
        "typed_graph_mechanics_completed_count": (suite.typed_graph_mechanics_completed_count),
        "provider_calls_made": suite.provider_calls_made,
        "extraction_accuracy_authority": suite.extraction_accuracy_authority,
        "synthesis_input_authority": suite.synthesis_input_authority,
        "scientific_synthesis_authority": suite.scientific_synthesis_authority,
        "scientific_effectiveness_authority": (suite.scientific_effectiveness_authority),
        "calibration_authority": suite.calibration_authority,
        "claim_release_authority": suite.claim_release_authority,
        "witnesses": [
            {
                "witness_id": item.witness_id,
                "receipt_sha256": item.receipt_sha256,
                "grounded_effect_sha256": item.grounded_effect_sha256,
                "native_projection_status": item.native_projection.status,
                "source_content_scope": item.source_content_scope,
                "release_grade_source_grounding_eligible": (
                    item.release_grade_source_grounding_eligible
                ),
            }
            for item in suite.receipts
        ],
    }


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build":
        suite = freeze_contextual_grounding_offline_feasibility_suite_v3(
            repository_root=args.repository_root
        )
        atomic_write_json(args.output, suite, force=args.force)
    elif args.command == "validate":
        suite = validate_contextual_grounding_offline_feasibility_suite_v3(
            suite=_load(args.input),
            repository_root=args.repository_root,
            external_replay=not args.contract_only,
        )
    else:
        suite = ContextualGroundingOfflineFeasibilitySuiteV3.model_validate(_load(args.input))
    print(json.dumps(_status(suite), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
