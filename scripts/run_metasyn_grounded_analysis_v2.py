#!/usr/bin/env python3
"""Build the label-blind MetaSyn grounded quantitative-mechanics artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from literature_multiverse.cohort_reconciliation import (
    ReviewerCohortReconciliationArtifact,
)
from literature_multiverse.lineage import atomic_write_json
from literature_multiverse.metasyn_grounded_analysis_v2 import (
    freeze_metasyn_grounded_analysis_v2,
)
from literature_multiverse.metasyn_grounded_publication_bridge_v2 import (
    MetaSynGroundedPublicationCorpusBridgeV2,
)


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metasyn_grounded_analysis_v2_json_not_object:{path}")
    return value


def _reviewers(path: Path | None) -> dict[str, ReviewerCohortReconciliationArtifact]:
    if path is None:
        return {}
    value = _object(path)
    if set(value) != {"reviewer_map_version", "artifacts_by_question"}:
        raise ValueError("metasyn_grounded_analysis_v2_reviewer_map_keys_invalid")
    if value["reviewer_map_version"] != "metasyn-grounded-analysis-reviewer-map-v1":
        raise ValueError("metasyn_grounded_analysis_v2_reviewer_map_version_invalid")
    artifacts = value["artifacts_by_question"]
    if not isinstance(artifacts, dict):
        raise ValueError("metasyn_grounded_analysis_v2_reviewer_map_not_object")
    return {
        str(question_id): ReviewerCohortReconciliationArtifact.model_validate(artifact)
        for question_id, artifact in artifacts.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--reviewer-artifacts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    bridge = MetaSynGroundedPublicationCorpusBridgeV2.model_validate(_object(args.bridge))
    analysis = freeze_metasyn_grounded_analysis_v2(
        bridge=bridge,
        repository_root=args.repository_root,
        reviewer_artifacts_by_question=_reviewers(args.reviewer_artifacts),
    )
    atomic_write_json(args.output, analysis, force=args.force)
    print(
        json.dumps(
            {
                "analysis_sha256": analysis.analysis_sha256,
                "kernel_abstained_units": analysis.kernel_abstained_unit_count,
                "kernel_completed_units": analysis.kernel_completed_unit_count,
                "kernel_invoked_units": analysis.kernel_invoked_unit_count,
                "output": args.output.as_posix(),
                "publication_count": analysis.publication_count,
                "question_count": analysis.question_count,
                "scientific_synthesis_authority": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
