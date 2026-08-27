#!/usr/bin/env python3
"""Evaluate frozen closed-corpus system/oracle JSONL files without network access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from literature_multiverse.closed_corpus import (
    ClosedCorpusGoldQuestion,
    ClosedCorpusPrediction,
    evaluate_closed_corpus,
)
from literature_multiverse.lineage import atomic_write_json


def _jsonl(path: Path) -> list[Any]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid_jsonl:{path}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    gold = TypeAdapter(list[ClosedCorpusGoldQuestion]).validate_python(_jsonl(args.gold))
    predictions = TypeAdapter(list[ClosedCorpusPrediction]).validate_python(
        _jsonl(args.predictions)
    )
    evaluation = evaluate_closed_corpus(gold=gold, predictions=predictions)
    atomic_write_json(args.output, evaluation, force=args.force)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "questions": evaluation["questions"],
                "arm_status": {
                    arm: result["status"] for arm, result in evaluation["arms"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
