#!/usr/bin/env python3
"""Evaluate private synthesis-authorization reviews and emit an aggregate receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.cohort_reconciliation import NativeCohortReconciliationReceipt
from literature_multiverse.synthesis_authorization_review import (
    evaluate_synthesis_authorization_review,
    write_synthesis_authorization_review_evaluation,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--adjudicator", type=Path)
    parser.add_argument(
        "--private-output",
        type=Path,
        required=True,
        help="New file below data/cache/synthesis-authorization-review.",
    )
    parser.add_argument(
        "--public-output",
        type=Path,
        required=True,
        help="New aggregate-only file below artifacts/diagnostics/synthesis-authorization-review.",
    )
    parser.add_argument(
        "--adjudication-template-output",
        type=Path,
        help=(
            "Required on an awaiting-adjudication pass and forbidden once complete; "
            "must be below the ignored private review root."
        ),
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    corpus = TypedEvidenceCorpus.model_validate(_object(args.corpus))
    reconciliation = NativeCohortReconciliationReceipt.model_validate(_object(args.reconciliation))
    private, public, adjudication_template = evaluate_synthesis_authorization_review(
        manifest_path=args.manifest,
        corpus=corpus,
        reconciliation=reconciliation,
        repository_root=args.repository_root,
        reviewer_a_path=args.reviewer_a,
        reviewer_b_path=args.reviewer_b,
        adjudicator_path=args.adjudicator,
    )
    if args.require_complete and public.status != "complete":
        raise ValueError("synthesis_authorization_review_not_complete")
    if (args.adjudication_template_output is None) != (adjudication_template is None):
        state = "required" if adjudication_template is not None else "forbidden"
        raise ValueError(f"--adjudication-template-output is {state} for this evaluation")
    write_synthesis_authorization_review_evaluation(
        repository_root=args.repository_root,
        private_output=args.private_output,
        public_output=args.public_output,
        private=private,
        public=public,
        adjudication_template_output=args.adjudication_template_output,
        adjudication_template=adjudication_template,
    )
    print(
        json.dumps(
            {
                "status": public.status,
                "review_target_count": public.review_target_count,
                "disagreement_count": public.disagreement_count,
                "adjudicated_count": public.adjudicated_count,
                "authorized_unit_count": public.authorized_unit_count,
                "abstained_unit_count": public.abstained_unit_count,
                "pipeline_fingerprint_sha256": public.pipeline_fingerprint_sha256,
                "summary_sha256": public.summary_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
