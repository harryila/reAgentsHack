#!/usr/bin/env python3
"""Generate deterministic provider-free inputs for the four pipeline scenarios."""

from __future__ import annotations

import argparse
import json
import sys

from literature_multiverse.fixtures import (
    FIXTURE_FAULT_MODE,
    FixtureContractError,
    generate_all_fixtures,
    generate_fixture,
)
from literature_multiverse.paths import PATHS


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    selection = result.add_mutually_exclusive_group(required=True)
    selection.add_argument("--question")
    selection.add_argument("--all", action="store_true")
    result.add_argument("--fixture", action="store_true")
    result.add_argument(
        "--fault-injection",
        choices=(FIXTURE_FAULT_MODE,),
        help="write the fixture-b-incomplete control consumed by the real s5 runner",
    )
    result.add_argument("--force", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.all:
            if args.fault_injection is not None:
                raise FixtureContractError(
                    "fixture_fault_injection_requires_single_question"
                )
            generated = generate_all_fixtures(
                explicit_fixture=args.fixture,
                force=args.force,
                paths=PATHS,
            )
        else:
            assert args.question is not None
            generated = [
                generate_fixture(
                    args.question,
                    explicit_fixture=args.fixture,
                    fault_injection=args.fault_injection,
                    force=args.force,
                    paths=PATHS,
                )
            ]
    except (FixtureContractError, ValueError, OSError) as exc:
        print(f"fixture generation failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "fixture_version": "1",
                "status": "complete",
                "questions": [
                    {
                        "question_id": item.question_id,
                        "counts": dict(item.counts),
                        "run_record": PATHS.repository_relative(item.run_record_path),
                        "fault_injection": (
                            PATHS.repository_relative(item.fault_injection_path)
                            if item.fault_injection_path is not None
                            else None
                        ),
                    }
                    for item in generated
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
