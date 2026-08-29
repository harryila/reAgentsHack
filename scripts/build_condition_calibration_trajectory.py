#!/usr/bin/env python3
"""Build an exact multi-arm outcome-free condition-calibration trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.condition_trajectory_builder import (
    build_condition_calibration_question_trajectory,
    preflight_condition_trajectory_output,
    read_condition_calibration_collection_source,
    require_directory_without_symlinks,
)
from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help=(
            "One independently collected, complete, outcome-free single-arm "
            "ConditionCalibrationCollectionSourceV1 JSON file; repeat for every arm."
        ),
    )
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        help="Repository root used to externally replay every source fingerprint.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace an existing regular output; symlinks remain forbidden.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_paths: list[Path] = args.source
    output = preflight_condition_trajectory_output(
        args.output,
        source_paths=source_paths,
        force=args.force,
    )
    pipeline_root = (
        None
        if args.pipeline_root is None
        else require_directory_without_symlinks(
            args.pipeline_root,
            purpose="pipeline_root",
        )
    )
    loaded = [read_condition_calibration_collection_source(path) for path in source_paths]
    sources = [source for source, _ in loaded]
    source_file_sha256s = sorted(file_sha256 for _, file_sha256 in loaded)
    trajectory = build_condition_calibration_question_trajectory(
        sources,
        pipeline_root=pipeline_root,
    )
    atomic_write_json(output, trajectory, force=args.force)
    receipt_payload = {
        "receipt_version": "condition-calibration-trajectory-builder-receipt-v1",
        "stage": "outcome_free_multi_arm_trajectory_built_before_assessments",
        "question_id": trajectory.base_visible.question_id,
        "split": trajectory.base_visible.split,
        "policy_arm_ids": [arm.base_arm.policy_arm_id for arm in trajectory.arms],
        "source_file_sha256s": source_file_sha256s,
        "source_collection_sha256s": sorted(
            source.collection_source_sha256 for source in sources
        ),
        "source_count": len(sources),
        "trajectory_sha256": trajectory.trajectory_sha256,
        "trajectory_file_sha256": sha256_file(output),
        "output": output.as_posix(),
        "condition_assessments_opened": False,
        "gate_outcomes_opened": False,
        "reference_labels_opened": False,
        "calibration_bundles_opened": False,
        "second_collection_pass_required": True,
    }
    print(
        json.dumps(
            {
                **receipt_payload,
                "receipt_sha256": hash_canonical(receipt_payload),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
