#!/usr/bin/env python3
"""Run the prospective claim-release boundary from one closed JSON request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.budgeted_verification import (
    AuditCandidate,
    ClaimModel,
    ProbabilityBasis,
    ReleaseGuardConfig,
    ScenarioKind,
)
from literature_multiverse.calibration import FrozenCalibrationBundle
from literature_multiverse.claim_release import (
    AuditResolutionReceipt,
    ClaimReleaseConfig,
    ClaimTarget,
    assess_claim_release,
)
from literature_multiverse.evidence_graph import EvidenceGraph
from literature_multiverse.lineage import atomic_write_json

_REQUIRED_KEYS = {
    "graph",
    "question_id",
    "population_id",
    "domain",
    "pipeline_sha256",
    "target",
    "audit_candidates",
    "claim_model",
    "audit_resolution_receipts",
    "audit_budget",
    "frozen_calibration_bundle",
}
_OPTIONAL_KEYS = {"config", "audit_guard_config"}


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("claim_release_request_must_be_object")
    missing = sorted(_REQUIRED_KEYS - set(payload))
    extra = sorted(set(payload) - _REQUIRED_KEYS - _OPTIONAL_KEYS)
    if missing or extra:
        raise ValueError(f"claim_release_request_keys_invalid:missing={missing}:extra={extra}")
    return payload


def _audit_candidate(payload: object) -> AuditCandidate:
    if not isinstance(payload, dict):
        raise ValueError("claim_release_audit_candidate_must_be_object")
    values = dict(payload)
    values["probability_basis"] = ProbabilityBasis(values["probability_basis"])
    values["scenario_kind"] = ScenarioKind(values["scenario_kind"])
    return AuditCandidate(**values)


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"claim_release_{key}_must_be_nonempty_string")
    return value


def _resolution_receipts(payload: object) -> list[AuditResolutionReceipt]:
    if not isinstance(payload, list):
        raise ValueError("claim_release_audit_resolution_receipts_must_be_list")
    return [AuditResolutionReceipt.model_validate(receipt) for receipt in payload]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assess a prespecified increase/decrease claim without accepting oracle labels."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Closed request JSON")
    parser.add_argument("--output", type=Path, required=True, help="Assessment JSON")
    parser.add_argument("--force", action="store_true", help="Replace an existing output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = _load_object(args.input)
    bundle_payload = payload["frozen_calibration_bundle"]
    bundle = (
        None
        if bundle_payload is None
        else FrozenCalibrationBundle.model_validate(bundle_payload)
    )
    config_payload = payload.get("config")
    guard_payload = payload.get("audit_guard_config")
    result = assess_claim_release(
        graph=EvidenceGraph.model_validate(payload["graph"]),
        question_id=_required_string(payload, "question_id"),
        population_id=_required_string(payload, "population_id"),
        domain=_required_string(payload, "domain"),
        pipeline_sha256=_required_string(payload, "pipeline_sha256"),
        target=ClaimTarget.model_validate(payload["target"]),
        audit_candidates=[
            _audit_candidate(candidate) for candidate in payload["audit_candidates"]
        ],
        claim_model=ClaimModel(**payload["claim_model"]),
        audit_resolution_receipts=_resolution_receipts(
            payload["audit_resolution_receipts"]
        ),
        audit_budget=float(payload["audit_budget"]),
        frozen_calibration_bundle=bundle,
        config=(
            None if config_payload is None else ClaimReleaseConfig.model_validate(config_payload)
        ),
        audit_guard_config=(
            None if guard_payload is None else ReleaseGuardConfig(**guard_payload)
        ),
    )
    atomic_write_json(args.output, result, force=args.force)
    print(
        json.dumps(
            {
                "status": result.status.value,
                "question_id": result.question_id,
                "decision_sha256": result.decision_sha256,
                "reasons": result.reasons,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
