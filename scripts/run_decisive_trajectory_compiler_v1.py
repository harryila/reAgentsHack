#!/usr/bin/env python3
"""Freeze a source roster, compile, or externally replay decisive trajectories."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from literature_multiverse.adjudication_replay import AdjudicationReplayPackageV1
from literature_multiverse.decisive_claim_evaluation_v1 import (
    DecisiveEvaluationConfigV1,
    DecisiveSplitManifestV1,
    FitStageReceiptV1,
)
from literature_multiverse.decisive_trajectory_compiler_v1 import (
    DecisiveTrajectoryCompilationResultV1,
    DecisiveTrajectorySourceRosterV1,
    NormalizedConditionSetArtifactV1,
    _read_model,
    _read_production_certificate,
    _resolve_source_path,
    compile_decisive_trajectory_bundle_v1,
    freeze_adjudication_replay_package_locator_v1,
    freeze_condition_set_source_binding_v1,
    freeze_decisive_trajectory_source_roster_v1,
    freeze_question_trajectory_source_v1,
    freeze_transactional_workspace_locator_v1,
    freeze_verifier_certificate_locator_v1,
    replay_decisive_trajectory_compilation_v1,
    write_decisive_trajectory_compilation_v1,
)
from literature_multiverse.lineage import atomic_write_json
from literature_multiverse.transactional_audit_workspace_v1 import (
    load_transactional_audit_workspace_v1,
)


def _bytes(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"decisive_trajectory_compiler_cli_file_invalid:{label}:{path}")
    return path.read_bytes()


def _model[ModelT: BaseModel](path: Path, model_type: type[ModelT], *, label: str) -> ModelT:
    raw = _bytes(path, label=label)
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("not_object")
        return model_type.model_validate(value)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"decisive_trajectory_compiler_cli_model_invalid:{label}:{path}") from exc


def _time(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("decisive_trajectory_compiler_cli_timestamp_requires_timezone")
    return parsed


def _assignment(value: str, *, label: str) -> tuple[str, str]:
    left, separator, right = value.partition("=")
    if not separator or not left or not right:
        raise ValueError(f"decisive_trajectory_compiler_cli_assignment_invalid:{label}")
    return left, right


def _condition_assignment(value: str) -> tuple[str, str, str]:
    left, relative_path = _assignment(value, label="condition_binding")
    question_id, separator, certificate_sha256 = left.partition(":")
    if not separator or not question_id or not certificate_sha256:
        raise ValueError("decisive_trajectory_compiler_cli_condition_binding_invalid")
    return question_id, certificate_sha256, relative_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    roster = commands.add_parser(
        "freeze-roster",
        help="validate current workspace pointers and freeze local relative source locators",
    )
    roster.add_argument("--split-manifest", type=Path, required=True)
    roster.add_argument("--source-root", type=Path, required=True)
    roster.add_argument(
        "--workspace",
        action="append",
        default=[],
        metavar="QUESTION_ID=RELATIVE_PATH",
        help="Repeat once for every transactional workspace branch",
    )
    roster.add_argument(
        "--adjudication-package",
        action="append",
        default=[],
        metavar="QUESTION_ID=RELATIVE_PATH",
        help=(
            "Required exact replay package binding raw reviewer decisions, timing, "
            "resolution, protocol, and operator trust registry files"
        ),
    )
    roster.add_argument(
        "--condition-binding",
        action="append",
        default=[],
        metavar="QUESTION_ID:CERTIFICATE_SHA256=RELATIVE_PATH",
        help="Bind each final condition-v7 certificate to an exact normalized condition artifact",
    )
    roster.add_argument(
        "--certificate",
        action="append",
        default=[],
        metavar="QUESTION_ID=RELATIVE_PATH",
        help="Repeat for standalone preselection certificates absent from workspaces",
    )
    roster.add_argument("--output", type=Path, required=True)

    for name, help_text in (
        ("compile", "compile the exact policy-visited prefix union"),
        ("validate", "externally replay a saved compilation"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--split-manifest", type=Path, required=True)
        command.add_argument("--development-receipt", type=Path, required=True)
        command.add_argument("--calibration-receipt", type=Path, required=True)
        command.add_argument("--source-roster", type=Path, required=True)
        command.add_argument("--source-root", type=Path, required=True)
        command.add_argument("--repository-root", type=Path, required=True)
        if name == "compile":
            command.add_argument("--compiled-at", required=True)
            command.add_argument("--output-bundle", type=Path, required=True)
            command.add_argument("--output-receipt", type=Path, required=True)
        else:
            command.add_argument("--receipt", type=Path, required=True)
    return parser


def _inputs(args: argparse.Namespace) -> dict[str, Any]:
    roster = _model(
        args.source_roster,
        DecisiveTrajectorySourceRosterV1,
        label="source_roster",
    )
    return {
        "config": _model(args.config, DecisiveEvaluationConfigV1, label="config"),
        "split_manifest": _model(
            args.split_manifest, DecisiveSplitManifestV1, label="split_manifest"
        ),
        "development_receipt": _model(
            args.development_receipt, FitStageReceiptV1, label="development_receipt"
        ),
        "calibration_receipt": _model(
            args.calibration_receipt, FitStageReceiptV1, label="calibration_receipt"
        ),
        "source_roster": roster,
        "source_roster_path": args.source_roster,
        "source_root": args.source_root,
        "repository_root": args.repository_root,
    }


def _freeze_roster(args: argparse.Namespace) -> DecisiveTrajectorySourceRosterV1:
    split_manifest = _model(args.split_manifest, DecisiveSplitManifestV1, label="split_manifest")
    workspaces: dict[str, list[Any]] = {}
    for raw in args.workspace:
        question_id, relative = _assignment(raw, label="workspace")
        workspace = _resolve_source_path(args.source_root, relative)
        config, pointer = load_transactional_audit_workspace_v1(workspace)
        workspaces.setdefault(question_id, []).append(
            freeze_transactional_workspace_locator_v1(
                relative_path=relative,
                expected_workspace_config_sha256=config.config_sha256,
                expected_terminal_pointer_sha256=pointer.pointer_sha256,
            )
        )
    adjudication_packages: dict[str, Any] = {}
    for raw in args.adjudication_package:
        question_id, relative = _assignment(raw, label="adjudication_package")
        if question_id in adjudication_packages:
            raise ValueError("decisive_trajectory_compiler_cli_adjudication_package_duplicate")
        package_path = _resolve_source_path(args.source_root, relative)
        package, artifact_binding = _read_model(
            package_path,
            AdjudicationReplayPackageV1,
            relative_path=relative,
            artifact_kind="adjudication_replay_package",
        )
        if package.question_id != question_id:
            raise ValueError(
                "decisive_trajectory_compiler_cli_adjudication_package_question_mismatch"
            )
        adjudication_packages[question_id] = freeze_adjudication_replay_package_locator_v1(
            relative_path=relative,
            expected_file_sha256=artifact_binding.file_sha256,
            expected_package_sha256=package.package_sha256,
        )
    conditions: dict[str, list[Any]] = {}
    for raw in args.condition_binding:
        question_id, certificate_sha256, relative = _condition_assignment(raw)
        condition_path = _resolve_source_path(args.source_root, relative)
        condition_artifact, artifact_binding = _read_model(
            condition_path,
            NormalizedConditionSetArtifactV1,
            relative_path=relative,
            artifact_kind="normalized_condition_set_artifact",
        )
        conditions.setdefault(question_id, []).append(
            freeze_condition_set_source_binding_v1(
                certificate_sha256=certificate_sha256,
                relative_path=relative,
                expected_file_sha256=artifact_binding.file_sha256,
                condition_set_artifact_sha256=condition_artifact.artifact_sha256,
            )
        )
    certificates: dict[str, list[Any]] = {}
    for raw in args.certificate:
        question_id, relative = _assignment(raw, label="certificate")
        certificate_path = _resolve_source_path(args.source_root, relative)
        certificate, artifact_binding = _read_production_certificate(
            certificate_path,
            relative_path=relative,
            artifact_kind="standalone_verification_certificate",
        )
        certificates.setdefault(question_id, []).append(
            freeze_verifier_certificate_locator_v1(
                relative_path=relative,
                expected_file_sha256=artifact_binding.file_sha256,
                expected_certificate_sha256=certificate.certificate_sha256,
            )
        )
    evaluation_ids = {
        row.question_id for row in split_manifest.identities if row.split.value == "evaluation"
    }
    if (
        set(workspaces) != evaluation_ids
        or set(adjudication_packages) != evaluation_ids
        or set(conditions) - evaluation_ids
        or set(certificates) - evaluation_ids
    ):
        raise ValueError("decisive_trajectory_compiler_cli_evaluation_workspace_roster_incomplete")
    questions = [
        freeze_question_trajectory_source_v1(
            question_id=question_id,
            workspaces=workspaces[question_id],
            adjudication_replay_package=adjudication_packages[question_id],
            verifier_certificates=certificates.get(question_id, []),
            condition_set_bindings=conditions.get(question_id, []),
        )
        for question_id in sorted(evaluation_ids)
    ]
    return freeze_decisive_trajectory_source_roster_v1(
        split_manifest=split_manifest,
        questions=questions,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-roster":
        roster = _freeze_roster(args)
        atomic_write_json(args.output, roster, force=False)
        print(
            json.dumps(
                {
                    "artifact_type": type(roster).__name__,
                    "evaluation_question_count": len(roster.questions),
                    "evaluation_reference_labels_present": False,
                    "output": args.output.as_posix(),
                    "scientific_claim_authority": False,
                    "source_roster_sha256": roster.source_roster_sha256,
                },
                sort_keys=True,
            )
        )
        return 0

    inputs = _inputs(args)
    if args.command == "compile":
        result = compile_decisive_trajectory_bundle_v1(
            **inputs,
            compiled_at=_time(args.compiled_at),
        )
        write_decisive_trajectory_compilation_v1(
            result,
            bundle_path=args.output_bundle,
            receipt_path=args.output_receipt,
        )
        print(
            json.dumps(
                {
                    "artifact_type": type(result).__name__,
                    "claim_release_authority": False,
                    "compilation_sha256": (result.compilation_receipt.compilation_sha256),
                    "compilation_lineage_identity_sha256": (
                        result.compilation_receipt.compilation_lineage_identity.identity_sha256
                    ),
                    "evaluation_question_count": len(result.trajectory_bundle.trajectories),
                    "evaluation_reference_labels_opened": False,
                    "output_bundle": args.output_bundle.as_posix(),
                    "output_receipt": args.output_receipt.as_posix(),
                    "scientific_claim_authority": False,
                    "trajectory_bundle_sha256": (result.trajectory_bundle.bundle_sha256),
                },
                sort_keys=True,
            )
        )
        return 0

    expected = _model(
        args.receipt,
        DecisiveTrajectoryCompilationResultV1,
        label="compilation_receipt",
    )
    replayed = replay_decisive_trajectory_compilation_v1(
        expected=expected,
        **inputs,
    )
    print(
        json.dumps(
            {
                "artifact_type": type(replayed).__name__,
                "compilation_sha256": replayed.compilation_receipt.compilation_sha256,
                "compilation_lineage_identity_sha256": (
                    replayed.compilation_receipt.compilation_lineage_identity.identity_sha256
                ),
                "evaluation_reference_labels_opened": False,
                "external_replay": "passed",
                "scientific_claim_authority": False,
                "trajectory_bundle_sha256": replayed.trajectory_bundle.bundle_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
