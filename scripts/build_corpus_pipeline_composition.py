#!/usr/bin/env python3
"""Build or externally replay a corpus-pipeline composition receipt offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.corpus_pipeline_composition_runtime import (
    build_corpus_pipeline_composition_external_replay_receipt_v1,
    load_corpus_pipeline_composition_external_replay_receipt_v1,
    validate_corpus_pipeline_composition_external_replay_receipt_v1,
)
from literature_multiverse.lineage import atomic_write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="Replay package, bridge, extraction, verifier, and join policy; write receipt.",
    )
    build.add_argument("--repository-root", type=Path, default=Path("."))
    build.add_argument("--grounding-package", type=Path, required=True)
    build.add_argument("--hosted-bridge-receipt", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--force", action="store_true")

    validate = subparsers.add_parser(
        "validate",
        help="Rebuild from the source artifacts/current code and require exact receipt equality.",
    )
    validate.add_argument("--repository-root", type=Path, default=Path("."))
    validate.add_argument("--grounding-package", type=Path, required=True)
    validate.add_argument("--hosted-bridge-receipt", type=Path, required=True)
    validate.add_argument("--receipt", type=Path, required=True)
    return parser


def _status(receipt: object) -> dict[str, object]:
    # The runtime contract is deliberately the only source of these public aliases.
    value = receipt
    return {
        "receipt_version": value.receipt_version,
        "external_replay_completed": value.external_replay_completed,
        "receipt_sha256": value.receipt_sha256,
        "composition_join_sha256": value.composition_join.join_sha256,
        "corpus_ingress_projection_sha256": (
            value.composition_join.corpus_ingress_projection_sha256
        ),
        "extraction_pipeline_sha256": (value.composition_join.extraction_pipeline_sha256),
        "verifier_core_pipeline_sha256": (value.composition_join.verifier_core_pipeline_sha256),
        "join_policy_pipeline_sha256": (value.composition_join.join_policy_pipeline_sha256),
        "calibration_pipeline_sha256": value.calibration_pipeline_sha256,
        "release_pipeline_sha256": value.release_pipeline_sha256,
        "scientific_authority": value.scientific_authority,
        "calibration_authority": value.calibration_authority,
        "claim_release_authority": value.claim_release_authority,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        receipt = build_corpus_pipeline_composition_external_replay_receipt_v1(
            repository_root=args.repository_root,
            grounding_package_path=args.grounding_package,
            hosted_bridge_receipt_path=args.hosted_bridge_receipt,
        )
        atomic_write_json(args.output, receipt, force=args.force)
    else:
        loaded = load_corpus_pipeline_composition_external_replay_receipt_v1(args.receipt)
        receipt = validate_corpus_pipeline_composition_external_replay_receipt_v1(
            receipt=loaded,
            repository_root=args.repository_root,
            grounding_package_path=args.grounding_package,
            hosted_bridge_receipt_path=args.hosted_bridge_receipt,
        )
    print(json.dumps(_status(receipt), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
