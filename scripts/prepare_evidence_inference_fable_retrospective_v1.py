#!/usr/bin/env python3
"""Freeze or externally replay the label-safe Fable retrospective plans."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from literature_multiverse.evidence_inference_fable_retrospective_v1 import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_FULL_PLAN_PATH,
    DEFAULT_PILOT_PLAN_PATH,
    DEFAULT_RECOVERY_PILOT_PLAN_PATH,
    EvidenceInferenceFableRetrospectiveError,
    EvidenceInferenceFableRetrospectivePlanV1,
    validate_evidence_inference_fable_retrospective_plan_v1,
    write_evidence_inference_fable_retrospective_plan_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("build", "freeze and atomically write one plan"),
        ("validate", "externally replay a plan from label-safe inputs"),
        ("status", "validate only the serialized contract and self-hash"),
    ):
        item = subcommands.add_parser(command, help=help_text)
        item.add_argument("--repository-root", type=Path, default=Path("."))
        item.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
        item.add_argument(
            "--mode",
            choices=(
                "pilot30_paired",
                "pilot30_recovery_v2_paired",
                "full_paired",
            ),
            required=True,
        )
        item.add_argument("--plan", type=Path)
        if command == "build":
            item.add_argument("--force", action="store_true")
    return parser


def _default_path(mode: str) -> Path:
    return {
        "pilot30_paired": DEFAULT_PILOT_PLAN_PATH,
        "pilot30_recovery_v2_paired": DEFAULT_RECOVERY_PILOT_PLAN_PATH,
        "full_paired": DEFAULT_FULL_PLAN_PATH,
    }[mode]


def _safe_existing_plan(*, root: Path, selected: Path) -> Path:
    if selected.is_absolute() or ".." in selected.parts:
        raise EvidenceInferenceFableRetrospectiveError(
            "evidence_inference_fable_cli_plan_path_escape"
        )
    candidate = root / selected
    if candidate.is_symlink() or not candidate.is_file():
        raise EvidenceInferenceFableRetrospectiveError(
            "evidence_inference_fable_cli_plan_file_invalid"
        )
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EvidenceInferenceFableRetrospectiveError(
            "evidence_inference_fable_cli_plan_path_escape"
        ) from exc
    return resolved


def _read_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceInferenceFableRetrospectiveError(
            "evidence_inference_fable_cli_plan_json_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise EvidenceInferenceFableRetrospectiveError(
            "evidence_inference_fable_cli_plan_not_object"
        )
    return value


def _summary(
    plan: EvidenceInferenceFableRetrospectivePlanV1, *, validation: str
) -> dict[str, Any]:
    return {
        "status": plan.status,
        "validation": validation,
        "mode": plan.mode,
        "population": plan.population,
        "plan_sha256": plan.plan_sha256,
        "requests": plan.request_count,
        "examples": plan.unique_examples,
        "articles": plan.unique_articles,
        "provider_calls_made": plan.provider_calls_made,
        "diagnostic_known_surface_cost_usd_micros": (
            plan.total_diagnostic_known_surface_cost_usd_micros
        ),
        "full_context_hard_liability_usd_micros": (
            plan.total_full_context_hard_liability_usd_micros
        ),
        "claim_boundary": plan.comparison_interpretation,
        "confirmatory_claim_authority": plan.confirmatory_claim_authority,
        "claim_release_authority": plan.claim_release_authority,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve(strict=True)
    selected = args.plan or _default_path(args.mode)
    if args.command == "build":
        plan = write_evidence_inference_fable_retrospective_plan_v1(
            repository_root=root,
            mode=args.mode,
            config_path=args.config,
            output_path=selected,
            force=args.force,
        )
        validation = "frozen_from_label_safe_inputs_and_written_atomically"
    else:
        raw = _read_object(_safe_existing_plan(root=root, selected=selected))
        if args.command == "validate":
            plan = validate_evidence_inference_fable_retrospective_plan_v1(
                repository_root=root,
                config_path=args.config,
                plan=raw,
            )
            validation = "external_label_safe_input_replay_exact"
        else:
            try:
                plan = EvidenceInferenceFableRetrospectivePlanV1.model_validate(raw)
            except ValueError as exc:
                raise EvidenceInferenceFableRetrospectiveError(
                    "evidence_inference_fable_cli_plan_contract_invalid"
                ) from exc
            validation = "serialized_contract_and_self_hash_only"
        if plan.mode != args.mode:
            raise EvidenceInferenceFableRetrospectiveError(
                "evidence_inference_fable_cli_mode_mismatch"
            )
    print(json.dumps(_summary(plan, validation=validation), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
