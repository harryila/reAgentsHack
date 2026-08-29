#!/usr/bin/env python3
"""Freeze a private two-reviewer source-identity packet for exact synthesis units."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from literature_multiverse.cohort_reconciliation import NativeCohortReconciliationReceipt
from literature_multiverse.synthesis_authorization_review import (
    RequestedSynthesisUnit,
    SynthesisAuthorizationReviewRequest,
    freeze_synthesis_authorization_review_request,
    prepare_synthesis_authorization_review,
)
from literature_multiverse.typed_extraction import TypedEvidenceCorpus


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"synthesis_review_json_invalid:{path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"synthesis_review_json_requires_object:{path}")
    return value


def _request(path: Path) -> SynthesisAuthorizationReviewRequest:
    value = _object(path)
    if set(value) != {"request_version", "synthesis_units"}:
        raise ValueError("synthesis_review_request_keys_mismatch")
    if value["request_version"] != "synthesis-authorization-review-request-v1":
        raise ValueError("synthesis_review_request_version_unsupported")
    if not isinstance(value["synthesis_units"], list):
        raise ValueError("synthesis_review_request_units_invalid")
    return freeze_synthesis_authorization_review_request(
        [RequestedSynthesisUnit.model_validate(item) for item in value["synthesis_units"]]
    )


def _timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--created-at must be an ISO-8601 timestamp") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Fresh directory strictly below "
            "data/cache/synthesis-authorization-review; it must not already exist."
        ),
    )
    parser.add_argument(
        "--created-at",
        type=_timestamp,
        required=True,
        help="Explicit UTC packet-freeze time; ambient wall-clock time is never inferred.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = prepare_synthesis_authorization_review(
        corpus=TypedEvidenceCorpus.model_validate(_object(args.corpus)),
        reconciliation=NativeCohortReconciliationReceipt.model_validate(
            _object(args.reconciliation)
        ),
        request=_request(args.request),
        repository_root=args.repository_root,
        output_dir=args.output_dir,
        created_at=args.created_at,
    )
    print(
        json.dumps(
            {
                "status": "prepared_private_review",
                "review_target_count": manifest.review_target_count,
                "source_identity_visible": manifest.source_identity_visible,
                "system_scores_included": manifest.system_scores_included,
                "pipeline_fingerprint_sha256": manifest.pipeline_fingerprint_sha256,
                "manifest": (args.output_dir / "manifest.private.json").as_posix(),
                "manifest_sha256": manifest.manifest_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
