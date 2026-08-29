"""Dependency-neutral lineage contracts for decisive trajectory compilation.

The compiler embeds :class:`DecisiveCompilationLineageIdentityV1`.  The decisive
label-opening lifecycle accepts real trajectories only after independently reading
the exact compiler result and source roster, replaying the compiler from its raw
sources, and freezing :class:`DecisiveCompilationReplayProofV1`.

The module intentionally has no top-level dependency on either the compiler or the
decisive evaluator.  Its replay helper imports the compiler only when invoked, which
keeps the shared contracts free of an import cycle.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, model_validator

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel

_MAX_COMPILATION_RESULT_BYTES = 512 * 1024 * 1024


class DecisiveCompilationLineageV1Error(ValueError):
    """The claimed compiler lineage cannot be independently replayed."""


class _FrozenExactModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]


def _self_hash(model: _FrozenExactModel, field_name: str) -> None:
    expected = hash_canonical(model.model_dump(mode="json", exclude={field_name}))
    if getattr(model, field_name) != expected:
        raise ValueError(f"decisive_compilation_lineage_self_hash_mismatch:{field_name}")


def _sorted_unique(values: list[str], field_name: str) -> list[str]:
    if values != sorted(set(values)):
        raise ValueError(f"decisive_compilation_lineage_not_canonical:{field_name}")
    return values


class DecisiveCompilationLineageIdentityV1(_FrozenExactModel):
    identity_version: Literal["decisive-compilation-lineage-identity-v1"] = (
        "decisive-compilation-lineage-identity-v1"
    )
    compiler_component_sha256: Sha256
    config_sha256: Sha256
    split_manifest_sha256: Sha256
    development_receipt_sha256: Sha256
    calibration_receipt_sha256: Sha256
    source_roster_file_sha256: Sha256
    source_roster_sha256: Sha256
    trajectory_bundle_sha256: Sha256
    trajectory_membership_sha256: Sha256
    evaluation_question_ids: Annotated[list[str], Field(min_length=1)]
    question_receipt_sha256s: Annotated[list[Sha256], Field(min_length=1)]
    adjudication_package_binding_sha256s: Annotated[list[Sha256], Field(min_length=1)]
    raw_adjudication_workflow_replay_required: Literal[True] = True
    operator_registry_is_not_external_expertise_proof: Literal[True] = True
    scientific_claim_authority: Literal[False] = False
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_identity(self) -> DecisiveCompilationLineageIdentityV1:
        _sorted_unique(self.evaluation_question_ids, "evaluation_question_ids")
        _sorted_unique(self.question_receipt_sha256s, "question_receipt_sha256s")
        _sorted_unique(
            self.adjudication_package_binding_sha256s,
            "adjudication_package_binding_sha256s",
        )
        if len(self.evaluation_question_ids) != len(self.question_receipt_sha256s) or len(
            self.evaluation_question_ids
        ) != len(self.adjudication_package_binding_sha256s):
            raise ValueError("decisive_compilation_lineage_question_projection_mismatch")
        _self_hash(self, "identity_sha256")
        return self


def freeze_decisive_compilation_lineage_identity_v1(
    *,
    compiler_component_sha256: str,
    config_sha256: str,
    split_manifest_sha256: str,
    development_receipt_sha256: str,
    calibration_receipt_sha256: str,
    source_roster_file_sha256: str,
    source_roster_sha256: str,
    trajectory_bundle_sha256: str,
    trajectory_membership_sha256: str,
    evaluation_question_ids: Sequence[str],
    question_receipt_sha256s: Sequence[str],
    adjudication_package_binding_sha256s: Sequence[str],
) -> DecisiveCompilationLineageIdentityV1:
    payload = {
        "identity_version": "decisive-compilation-lineage-identity-v1",
        "compiler_component_sha256": compiler_component_sha256,
        "config_sha256": config_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "development_receipt_sha256": development_receipt_sha256,
        "calibration_receipt_sha256": calibration_receipt_sha256,
        "source_roster_file_sha256": source_roster_file_sha256,
        "source_roster_sha256": source_roster_sha256,
        "trajectory_bundle_sha256": trajectory_bundle_sha256,
        "trajectory_membership_sha256": trajectory_membership_sha256,
        "evaluation_question_ids": sorted(evaluation_question_ids),
        "question_receipt_sha256s": sorted(question_receipt_sha256s),
        "adjudication_package_binding_sha256s": sorted(adjudication_package_binding_sha256s),
        "raw_adjudication_workflow_replay_required": True,
        "operator_registry_is_not_external_expertise_proof": True,
        "scientific_claim_authority": False,
    }
    return DecisiveCompilationLineageIdentityV1.model_validate(
        {**payload, "identity_sha256": hash_canonical(payload)}
    )


class DecisiveCompilationReplayProofV1(_FrozenExactModel):
    proof_version: Literal["decisive-compilation-external-replay-proof-v1"] = (
        "decisive-compilation-external-replay-proof-v1"
    )
    lineage_identity: DecisiveCompilationLineageIdentityV1
    compiler_result_file_sha256: Sha256
    compiler_result_sha256: Sha256
    compilation_sha256: Sha256
    exact_compiler_result_file_replayed: Literal[True] = True
    exact_source_roster_and_workspaces_replayed: Literal[True] = True
    current_compiler_code_identity_matched: Literal[True] = True
    exact_trajectory_bundle_argument_matched: Literal[True] = True
    bare_trajectory_bundle_is_not_sufficient: Literal[True] = True
    external_reviewer_identity_or_expertise_proven: Literal[False] = False
    evaluation_reference_labels_opened: Literal[False] = False
    scientific_claim_authority: Literal[False] = False
    proof_sha256: Sha256

    @model_validator(mode="after")
    def validate_proof(self) -> DecisiveCompilationReplayProofV1:
        _self_hash(self, "proof_sha256")
        return self


def _read_regular_no_follow(path: Path, *, label: str) -> bytes:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise DecisiveCompilationLineageV1Error(
                f"decisive_compilation_lineage_source_unreadable:{label}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise DecisiveCompilationLineageV1Error(
                f"decisive_compilation_lineage_source_symlink:{label}"
            )
    try:
        descriptor = os.open(absolute, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise DecisiveCompilationLineageV1Error(
            f"decisive_compilation_lineage_source_unreadable:{label}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > _MAX_COMPILATION_RESULT_BYTES
        ):
            raise DecisiveCompilationLineageV1Error(
                f"decisive_compilation_lineage_source_file_invalid:{label}"
            )
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise DecisiveCompilationLineageV1Error(
                    f"decisive_compilation_lineage_source_short_read:{label}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
            raise DecisiveCompilationLineageV1Error(
                f"decisive_compilation_lineage_source_changed_during_read:{label}"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DecisiveCompilationLineageV1Error(
            f"decisive_compilation_lineage_json_invalid:{label}"
        ) from exc
    if not isinstance(value, dict):
        raise DecisiveCompilationLineageV1Error(
            f"decisive_compilation_lineage_json_not_object:{label}"
        )
    return value


def replay_decisive_compilation_lineage_v1(
    *,
    config: Any,
    split_manifest: Any,
    development_receipt: Any,
    calibration_receipt: Any,
    trajectory_bundle: Any,
    compiler_result_path: Path,
    source_roster_path: Path,
    source_root: Path,
    repository_root: Path,
) -> DecisiveCompilationReplayProofV1:
    """Externally replay exact compiler sources and bind them to one v1 bundle.

    Imports are local by design: the compiler imports the neutral identity model from
    this module, while the evaluator invokes this helper only after all modules have
    initialized.
    """

    from literature_multiverse.decisive_trajectory_compiler_v1 import (
        DecisiveTrajectoryCompilationResultV1,
        DecisiveTrajectorySourceRosterV1,
        replay_decisive_trajectory_compilation_v1,
    )

    result_raw = _read_regular_no_follow(
        compiler_result_path,
        label="trajectory_compilation_result",
    )
    roster_raw = _read_regular_no_follow(
        source_roster_path,
        label="trajectory_compilation_source_roster",
    )
    try:
        result = DecisiveTrajectoryCompilationResultV1.model_validate(
            _json_object(result_raw, label="trajectory_compilation_result")
        )
        source_roster = DecisiveTrajectorySourceRosterV1.model_validate(
            _json_object(roster_raw, label="trajectory_compilation_source_roster")
        )
    except ValueError as exc:
        raise DecisiveCompilationLineageV1Error(
            "decisive_compilation_lineage_source_model_invalid"
        ) from exc
    replayed = replay_decisive_trajectory_compilation_v1(
        expected=result,
        config=config,
        split_manifest=split_manifest,
        development_receipt=development_receipt,
        calibration_receipt=calibration_receipt,
        source_roster=source_roster,
        source_roster_path=source_roster_path,
        source_root=source_root,
        repository_root=repository_root,
    )
    if replayed.trajectory_bundle != trajectory_bundle:
        raise DecisiveCompilationLineageV1Error(
            "decisive_compilation_lineage_trajectory_bundle_argument_mismatch"
        )
    receipt = replayed.compilation_receipt
    lineage = receipt.compilation_lineage_identity
    if (
        lineage.compiler_component_sha256 != receipt.compiler_component_sha256
        or lineage.config_sha256 != config.config_sha256
        or lineage.split_manifest_sha256 != split_manifest.manifest_sha256
        or lineage.development_receipt_sha256 != development_receipt.receipt_sha256
        or lineage.calibration_receipt_sha256 != calibration_receipt.receipt_sha256
        or lineage.source_roster_file_sha256 != hashlib.sha256(roster_raw).hexdigest()
        or lineage.source_roster_sha256 != source_roster.source_roster_sha256
        or lineage.trajectory_bundle_sha256 != trajectory_bundle.bundle_sha256
        or lineage.trajectory_membership_sha256 != trajectory_bundle.trajectory_membership_sha256
    ):
        raise DecisiveCompilationLineageV1Error(
            "decisive_compilation_lineage_identity_projection_mismatch"
        )
    payload = {
        "proof_version": "decisive-compilation-external-replay-proof-v1",
        "lineage_identity": lineage,
        "compiler_result_file_sha256": hashlib.sha256(result_raw).hexdigest(),
        "compiler_result_sha256": replayed.result_sha256,
        "compilation_sha256": receipt.compilation_sha256,
        "exact_compiler_result_file_replayed": True,
        "exact_source_roster_and_workspaces_replayed": True,
        "current_compiler_code_identity_matched": True,
        "exact_trajectory_bundle_argument_matched": True,
        "bare_trajectory_bundle_is_not_sufficient": True,
        "external_reviewer_identity_or_expertise_proven": False,
        "evaluation_reference_labels_opened": False,
        "scientific_claim_authority": False,
    }
    return DecisiveCompilationReplayProofV1.model_validate(
        {**payload, "proof_sha256": hash_canonical(payload)}
    )


__all__ = [
    "DecisiveCompilationLineageIdentityV1",
    "DecisiveCompilationLineageV1Error",
    "DecisiveCompilationReplayProofV1",
    "freeze_decisive_compilation_lineage_identity_v1",
    "replay_decisive_compilation_lineage_v1",
]
