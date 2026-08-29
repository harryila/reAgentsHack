#!/usr/bin/env python3
"""Freeze or inspect the offline Evidence Inference GEPA candidate-search plan."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from literature_multiverse.evidence_inference_gepa_candidate_search_v1 import (
    DEFAULT_CONFIG_PATH,
    EvidenceInferenceGEPACandidateSearchPlanV1,
    GEPACandidateSearchPlanError,
    load_gepa_candidate_search_config_v1,
    validate_evidence_inference_gepa_candidate_search_plan_v1,
    write_evidence_inference_gepa_candidate_search_plan_v1,
)


def _aware_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frozen-at must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("frozen-at must include a UTC offset")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser("build", help="freeze and atomically write the plan")
    build.add_argument("--repository-root", type=Path, default=Path("."))
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    build.add_argument("--output", type=Path)
    build.add_argument("--frozen-at", type=_aware_datetime, required=True)
    build.add_argument("--force", action="store_true")

    for command, help_text in (
        ("validate", "externally replay the plan from its frozen train/dev inputs"),
        ("status", "validate the serialized contract without reading split payloads"),
    ):
        item = subcommands.add_parser(command, help=help_text)
        item.add_argument("--repository-root", type=Path, default=Path("."))
        item.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
        item.add_argument("--plan", type=Path)
    return parser


def _safe_plan_path(
    *, repository_root: Path, config_path: Path, plan_path: Path | None
) -> Path:
    root = repository_root.resolve(strict=True)
    selected = plan_path
    if selected is None:
        selected = load_gepa_candidate_search_config_v1(
            repository_root=root, config_path=config_path
        ).output_plan_path
    if selected.is_absolute() or ".." in selected.parts:
        raise GEPACandidateSearchPlanError("gepa_candidate_search_cli_plan_path_escape")
    candidate = root / selected
    if candidate.is_symlink() or not candidate.is_file():
        raise GEPACandidateSearchPlanError("gepa_candidate_search_cli_plan_file_invalid")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise GEPACandidateSearchPlanError(
            "gepa_candidate_search_cli_plan_path_escape"
        ) from exc
    return resolved


def _read_plan(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GEPACandidateSearchPlanError(
            "gepa_candidate_search_cli_plan_json_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise GEPACandidateSearchPlanError("gepa_candidate_search_cli_plan_not_object")
    return value


def _summary(
    plan: EvidenceInferenceGEPACandidateSearchPlanV1, *, validation: str
) -> dict[str, Any]:
    prior = plan.prior_diagnosis
    return {
        "status": plan.status,
        "validation": validation,
        "plan_sha256": plan.plan_sha256,
        "frozen_at": plan.frozen_at.isoformat(),
        "provider": {
            "model": plan.provider.model,
            "effort": plan.provider.effort,
            "provider_calls_made": plan.provider_calls_made,
        },
        "prior_results": {
            "obsolete_first_pass": {
                "candidate_count": prior.obsolete_first_pass_candidate_count,
                "distinct_mutation_count": (
                    prior.obsolete_first_pass_distinct_mutation_count
                ),
                "seed_retained": prior.obsolete_first_pass_seed_retained,
            },
            "authoritative_scaled": {
                "candidate_count": prior.authoritative_scaled_candidate_count,
                "reflection_proposals": prior.authoritative_scaled_reflection_proposals,
                "seed_retained": prior.authoritative_scaled_seed_retained,
                "status": prior.authoritative_scaled_status,
                "observed_improvement_rule_satisfied": (
                    prior.authoritative_scaled_observed_improvement_rule_satisfied
                ),
            },
        },
        "tiers": [
            {
                "tier": item.tier,
                "initial_candidate_count": item.call_budget.initial_candidate_count,
                "initial_distinct_nonseed_candidate_count": (
                    item.call_budget.initial_distinct_nonseed_candidate_count
                ),
                "task_provider_call_ceiling": (
                    item.call_budget.task_provider_call_ceiling
                ),
                "reflection_call_ceiling": item.call_budget.reflection_call_ceiling,
                "total_provider_call_ceiling": (
                    item.call_budget.total_provider_call_ceiling
                ),
                "total_hard_cost_liability_usd_micros": (
                    item.call_budget.total_hard_cost_liability_usd_micros
                ),
            }
            for item in plan.tiers
        ],
        "test_boundary": {
            "payload_opened": plan.test_payload_opened,
            "payload_hashed": plan.test_payload_hashed,
            "labels_opened": plan.test_labels_opened,
            "labels_scored": plan.test_labels_scored,
        },
        "authorities": {
            "improvement": plan.improvement_authority,
            "generalization": plan.generalization_authority,
            "scientific_effectiveness": plan.scientific_effectiveness_authority,
            "claim_release": plan.claim_release_authority,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve(strict=True)
    if args.command == "build":
        plan = write_evidence_inference_gepa_candidate_search_plan_v1(
            repository_root=root,
            config_path=args.config,
            output_path=args.output,
            frozen_at=args.frozen_at,
            force=args.force,
        )
        validation = "frozen_from_train_dev_inputs_and_written_atomically"
    else:
        path = _safe_plan_path(
            repository_root=root,
            config_path=args.config,
            plan_path=args.plan,
        )
        raw = _read_plan(path)
        if args.command == "validate":
            plan = validate_evidence_inference_gepa_candidate_search_plan_v1(
                repository_root=root,
                config_path=args.config,
                plan=raw,
            )
            validation = "external_train_dev_replay_exact"
        else:
            try:
                plan = EvidenceInferenceGEPACandidateSearchPlanV1.model_validate(raw)
            except ValueError as exc:
                raise GEPACandidateSearchPlanError(
                    "gepa_candidate_search_cli_plan_contract_invalid"
                ) from exc
            validation = "serialized_contract_and_self_hash_only"
    print(json.dumps(_summary(plan, validation=validation), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
