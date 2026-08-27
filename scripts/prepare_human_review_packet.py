#!/usr/bin/env python3
"""Freeze a confidence-blinded, two-reviewer paper audit packet from local ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from literature_multiverse.closed_corpus import prepare_blinded_human_review_packet
from literature_multiverse.records import read_parquet_records


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid_json:{path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return value


def _config(path: Path) -> tuple[str, str, list[str]]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid_question_config:{path}") from exc
    if not isinstance(value, dict):
        raise ValueError("question_config_root_must_be_object")
    eligibility = value.get("eligibility")
    if not isinstance(eligibility, dict) or not isinstance(eligibility.get("include"), list):
        raise ValueError("question_config_eligibility_include_missing")
    return (
        str(value.get("question_id") or ""),
        str(value.get("research_question") or ""),
        [str(item) for item in eligibility["include"]],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-config", type=Path, required=True)
    parser.add_argument("--papers", type=Path, required=True)
    parser.add_argument("--findings", type=Path, required=True)
    parser.add_argument("--source-lines", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    question_id, research_question, criteria = _config(args.question_config)
    manifest = prepare_blinded_human_review_packet(
        question_id=question_id,
        research_question=research_question,
        eligibility_criteria=criteria,
        papers=read_parquet_records(args.papers),
        findings=read_parquet_records(args.findings),
        source_lines_by_doc_id=_object(args.source_lines),
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "manifest": (args.output_dir / "manifest.json").as_posix(),
                "sample_size": manifest["sample_size"],
                "selected_strata": manifest["selected_strata"],
                "all_eligible_zero_finding_papers_included": manifest[
                    "all_eligible_zero_finding_papers_included"
                ],
                "contains_model_confidence": manifest["contains_model_confidence"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
