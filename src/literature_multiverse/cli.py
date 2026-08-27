"""Supported command-line interface for Literature Multiverse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from literature_multiverse.calibration import FrozenCalibrationBundle
from literature_multiverse.certificate import write_certificate_artifacts
from literature_multiverse.claim_release import AuditResolutionReceipt
from literature_multiverse.verifier import (
    build_offline_fixture,
    load_claim_manifest,
    load_corpus,
    run_verification,
)


def _json_value(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}_json_unreadable:{path}") from exc


def _calibration_bundle(path: Path | None) -> FrozenCalibrationBundle | None:
    if path is None:
        return None
    payload = _json_value(path, label="calibration_bundle")
    if isinstance(payload, dict) and "frozen_calibration_bundle" in payload:
        payload = payload["frozen_calibration_bundle"]
    return FrozenCalibrationBundle.model_validate(payload)


def _audit_receipts(path: Path | None) -> list[AuditResolutionReceipt]:
    if path is None:
        return []
    payload = _json_value(path, label="audit_receipts")
    if isinstance(payload, dict) and "receipts" in payload:
        payload = payload["receipts"]
    if not isinstance(payload, list):
        raise ValueError("audit_receipts_json_must_be_list_or_receipts_object")
    return [AuditResolutionReceipt.model_validate(item) for item in payload]


def _add_verify_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "verify",
        help="Verify one AI-generated claim against a frozen literature corpus.",
        description=(
            "Build the evidence graph, synthesize effects, prioritize graph-derived "
            "verification actions, assess release gates, and write JSON/HTML certificates."
        ),
    )
    parser.add_argument("--claim", type=Path, help="Claim manifest YAML or JSON")
    parser.add_argument(
        "--corpus",
        type=Path,
        help=(
            "Evidence graph JSON, verifier corpus bundle, legacy findings JSONL/parquet, "
            "or a directory containing evidence_graph.json/findings.parquet"
        ),
    )
    parser.add_argument(
        "--budget-minutes",
        type=float,
        required=True,
        help="Maximum prospective human-verification time",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Optional frozen calibration bundle JSON",
    )
    parser.add_argument(
        "--receipts",
        type=Path,
        help="Optional completed hash-bound audit-resolution receipts JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory (default: artifacts/verification/<run-id>)",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Run the embedded provider-free integration fixture instead of input files",
    )
    parser.add_argument("--force", action="store_true", help="Replace certificate files")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lm", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_verify_parser(subparsers)
    return parser


def _verify(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.fixture:
        if args.claim is not None or args.corpus is not None:
            parser.error("--fixture cannot be combined with --claim or --corpus")
        manifest, corpus = build_offline_fixture()
    else:
        missing = [
            option
            for option, value in (("--claim", args.claim), ("--corpus", args.corpus))
            if value is None
        ]
        if missing:
            parser.error(f"verify requires {' and '.join(missing)} unless --fixture is used")
        manifest = load_claim_manifest(args.claim)
        corpus = load_corpus(args.corpus, legacy_settings=manifest.legacy_adapter)

    certificate = run_verification(
        manifest=manifest,
        corpus=corpus,
        budget_minutes=args.budget_minutes,
        frozen_calibration_bundle=_calibration_bundle(args.calibration),
        audit_resolution_receipts=_audit_receipts(args.receipts),
    )
    output_dir = args.output_dir or (
        Path("artifacts") / "verification" / certificate.run_id
    )
    artifacts = write_certificate_artifacts(certificate, output_dir, force=args.force)
    print(
        json.dumps(
            {
                "certificate_sha256": certificate.certificate_sha256,
                "decision_sha256": certificate.release_assessment.decision_sha256,
                "html_path": artifacts.html_path,
                "html_sha256": artifacts.html_sha256,
                "json_path": artifacts.json_path,
                "json_sha256": artifacts.json_sha256,
                "question_id": manifest.question_id,
                "reasons": certificate.reasons,
                "run_id": certificate.run_id,
                "status": certificate.status,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "verify":
        return _verify(args, parser)
    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover - argparse exits


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
