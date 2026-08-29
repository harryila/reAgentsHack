"""Build one outcome-free multi-arm condition-calibration trajectory.

Each input is a complete, independently collected, single-arm
``ConditionCalibrationCollectionSourceV1``.  The builder structurally validates and
externally replays every source, checks that all arms describe the same scientific
question and corpus, then emits the one canonical
``PolicyVisibleQuestionTrajectoryV2`` that must be embedded in a second collection
pass for every arm.

This module deliberately has no API for condition assessments, gate results,
reference labels, or calibration bundles.  Those artifacts are opened only after
the second-pass collection sources have been frozen into a source roster.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from literature_multiverse.adaptive_calibration import (
    ConfirmationAwareArmTrajectoryV2,
    PolicyVisibleQuestionTrajectoryV2,
    freeze_policy_visible_question_trajectory,
    freeze_policy_visible_question_trajectory_v2,
)
from literature_multiverse.certificate import ConditionCalibrationCollectionSourceV1
from literature_multiverse.lineage import OutputExistsError, sha256_bytes
from literature_multiverse.verifier import (
    validate_condition_calibration_collection_source_external_replay,
)


class ConditionTrajectoryBuilderError(ValueError):
    """A collection source cannot safely enter a multi-arm visible trajectory."""


_OUTCOME_BEARING_KEYS = frozenset(
    {
        "adaptive_calibration_bundle",
        "adaptive_calibration_bundle_v2",
        "calibration_gate_result",
        "condition_confirmation_assessment",
        "condition_confirmation_outcome",
        "condition_terminal_gate_result",
        "frozen_calibration_bundle",
        "gate_assessment",
        "gold_label",
        "ground_truth",
        "reference",
        "reference_sha256",
        "reference_verdict",
        "release_qualification_proof",
        "scientific_gate_passed",
        "terminal_gate_result",
    }
)


def _reject_outcome_bearing_input(value: Any, *, path: str = "source") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            if normalized in _OUTCOME_BEARING_KEYS:
                raise ConditionTrajectoryBuilderError(
                    f"condition_trajectory_outcome_bearing_input_forbidden:{path}.{key}"
                )
            _reject_outcome_bearing_input(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_outcome_bearing_input(item, path=f"{path}[{index}]")


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _first_symlink_component(path: Path) -> Path | None:
    """Return the first extant symlink without resolving any path component."""

    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            # Descendants cannot exist once an ancestor is absent.
            return None
        except OSError as exc:
            raise ConditionTrajectoryBuilderError(
                f"condition_trajectory_path_uninspectable:{current}"
            ) from exc
        if stat.S_ISLNK(mode):
            return current
    return None


def require_regular_source_file(path: Path) -> Path:
    """Validate one input path without following a symlink at any level."""

    symlink = _first_symlink_component(path)
    if symlink is not None:
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_source_symlink_forbidden:{symlink}"
        )
    absolute = _absolute_lexical(path)
    try:
        mode = absolute.lstat().st_mode
    except FileNotFoundError as exc:
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_source_missing:{path}"
        ) from exc
    except OSError as exc:
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_source_unreadable:{path}"
        ) from exc
    if not stat.S_ISREG(mode):
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_source_not_regular_file:{path}"
        )
    return absolute


def require_directory_without_symlinks(path: Path, *, purpose: str) -> Path:
    """Require one existing directory whose lexical path contains no symlink."""

    symlink = _first_symlink_component(path)
    if symlink is not None:
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_{purpose}_symlink_forbidden:{symlink}"
        )
    absolute = _absolute_lexical(path)
    try:
        mode = absolute.lstat().st_mode
    except FileNotFoundError as exc:
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_{purpose}_missing:{path}"
        ) from exc
    except OSError as exc:
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_{purpose}_uninspectable:{path}"
        ) from exc
    if not stat.S_ISDIR(mode):
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_{purpose}_not_directory:{path}"
        )
    return absolute


def preflight_condition_trajectory_output(
    output: Path,
    *,
    source_paths: Sequence[Path],
    force: bool,
) -> Path:
    """Fail before reading sources if output safety or immutability is violated."""

    absolute_output = _absolute_lexical(output)
    symlink = _first_symlink_component(absolute_output)
    if symlink is not None:
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_output_symlink_forbidden:{symlink}"
        )
    existing_parent = absolute_output.parent
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        if existing_parent.is_symlink():
            raise ConditionTrajectoryBuilderError(
                f"condition_trajectory_output_symlink_forbidden:{existing_parent}"
            )
        existing_parent = existing_parent.parent
    if not existing_parent.is_dir():
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_output_parent_not_directory:{existing_parent}"
        )

    canonical_sources = [require_regular_source_file(path) for path in source_paths]
    source_file_ids = [(path.stat().st_dev, path.stat().st_ino) for path in canonical_sources]
    if len(source_file_ids) != len(set(source_file_ids)):
        raise ConditionTrajectoryBuilderError(
            "condition_trajectory_source_file_overlap"
        )
    if absolute_output.exists():
        try:
            mode = absolute_output.lstat().st_mode
        except OSError as exc:
            raise ConditionTrajectoryBuilderError(
                f"condition_trajectory_output_uninspectable:{output}"
            ) from exc
        if not stat.S_ISREG(mode):
            raise ConditionTrajectoryBuilderError(
                f"condition_trajectory_output_not_regular_file:{output}"
            )
        if any(os.path.samefile(absolute_output, source) for source in canonical_sources):
            raise ConditionTrajectoryBuilderError(
                "condition_trajectory_output_must_not_alias_source"
            )
        if not force:
            raise OutputExistsError(absolute_output.as_posix())
    elif absolute_output in canonical_sources:
        raise ConditionTrajectoryBuilderError(
            "condition_trajectory_output_must_not_alias_source"
        )
    return absolute_output


def read_condition_calibration_collection_source(
    path: Path,
) -> tuple[ConditionCalibrationCollectionSourceV1, str]:
    """Read exactly one closed collection-source object from a regular file."""

    absolute = require_regular_source_file(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_source_unreadable:{path}"
        ) from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ConditionTrajectoryBuilderError(
                f"condition_trajectory_source_not_regular_file:{path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_source_invalid_json:{path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_source_must_be_json_object:{path}"
        )
    if payload.get("certificate_version") != "condition-calibration-collection-source-v1":
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_source_contract_only:{path}"
        )
    _reject_outcome_bearing_input(payload)
    try:
        source = ConditionCalibrationCollectionSourceV1.model_validate(payload)
    except ValidationError as exc:
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_source_contract_invalid:{path}"
        ) from exc
    return source, sha256_bytes(raw)


def _own_single_arm(
    source: ConditionCalibrationCollectionSourceV1,
) -> ConfirmationAwareArmTrajectoryV2:
    visible = source.policy_visible_question_trajectory
    if visible is None:
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_source_incomplete:{source.policy_arm_id}"
        )
    if len(visible.arms) != 1 or len(visible.base_visible.arms) != 1:
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_source_must_be_independent_single_arm:{source.policy_arm_id}"
        )
    arm = visible.arms[0]
    if (
        arm.base_arm.policy_arm_id != source.policy_arm_id
        or arm.base_arm.policy_context_sha256
        != source.adaptive_policy_context.policy_context_sha256
    ):
        raise ConditionTrajectoryBuilderError(
            f"condition_trajectory_source_arm_binding_mismatch:{source.policy_arm_id}"
        )
    return arm


def _require_same[T](
    sources: Sequence[ConditionCalibrationCollectionSourceV1],
    *,
    label: str,
    value: Callable[[ConditionCalibrationCollectionSourceV1], T],
) -> T:
    expected = value(sources[0])
    if any(value(source) != expected for source in sources[1:]):
        raise ConditionTrajectoryBuilderError(f"condition_trajectory_{label}_mismatch")
    return expected


def build_condition_calibration_question_trajectory(
    sources: Sequence[ConditionCalibrationCollectionSourceV1],
    *,
    pipeline_root: Path | None = None,
) -> PolicyVisibleQuestionTrajectoryV2:
    """Externally replay single-arm sources and assemble one canonical family.

    At least two sources are required.  Input order is immaterial; output arms are
    sorted by ``policy_arm_id`` by the underlying closed trajectory contracts.
    """

    if len(sources) < 2:
        raise ConditionTrajectoryBuilderError(
            "condition_trajectory_multiple_collection_sources_required"
        )
    replayed: list[ConditionCalibrationCollectionSourceV1] = []
    for index, source in enumerate(sources):
        try:
            canonical = ConditionCalibrationCollectionSourceV1.model_validate(
                source.model_dump(mode="json")
            )
        except (AttributeError, ValidationError) as exc:
            raise ConditionTrajectoryBuilderError(
                f"condition_trajectory_source_contract_invalid:index={index}"
            ) from exc
        try:
            replayed.append(
                validate_condition_calibration_collection_source_external_replay(
                    canonical,
                    pipeline_root=pipeline_root,
                )
            )
        except ValueError as exc:
            raise ConditionTrajectoryBuilderError(
                "condition_trajectory_source_external_replay_failed:"
                f"{canonical.policy_arm_id}:{exc}"
            ) from exc

    replayed.sort(key=lambda source: source.policy_arm_id)
    arm_ids = [source.policy_arm_id for source in replayed]
    if arm_ids != sorted(set(arm_ids)):
        raise ConditionTrajectoryBuilderError(
            "condition_trajectory_policy_arm_overlap"
        )
    context_hashes = [
        source.adaptive_policy_context.policy_context_sha256 for source in replayed
    ]
    if len(context_hashes) != len(set(context_hashes)):
        raise ConditionTrajectoryBuilderError(
            "condition_trajectory_policy_context_overlap"
        )

    question_id = _require_same(replayed, label="question", value=lambda row: row.question_id)
    split = _require_same(
        replayed,
        label="split",
        value=lambda row: row.collection_split,
    )
    population_id = _require_same(
        replayed,
        label="population",
        value=lambda row: row.claim_manifest.get("population_id"),
    )
    domain = _require_same(
        replayed,
        label="domain",
        value=lambda row: row.claim_manifest.get("domain"),
    )
    corpus = _require_same(
        replayed,
        label="corpus",
        value=lambda row: row.complete_corpus_identity,
    )
    _require_same(
        replayed,
        label="source_graph",
        value=lambda row: row.source_evidence_graph_sha256,
    )
    target_semantics = _require_same(
        replayed,
        label="target_semantics",
        value=lambda row: row.condition_target_semantics,
    )
    independence_identity = _require_same(
        replayed,
        label="independence_semantics",
        value=lambda row: row.condition_independence_identity,
    )
    _require_same(
        replayed,
        label="pipeline",
        value=lambda row: row.pipeline_verification.computed_pipeline_sha256,
    )

    arms = [_own_single_arm(source) for source in replayed]
    base_visible = freeze_policy_visible_question_trajectory(
        question_id=question_id,
        split=split,
        population_id=str(population_id),
        domain=str(domain),
        corpus=corpus,
        arms=[arm.base_arm for arm in arms],
    )
    return freeze_policy_visible_question_trajectory_v2(
        base_visible=base_visible,
        target_semantics=target_semantics,
        independence_identity=independence_identity,
        arms=arms,
    )


__all__ = [
    "ConditionTrajectoryBuilderError",
    "build_condition_calibration_question_trajectory",
    "preflight_condition_trajectory_output",
    "read_condition_calibration_collection_source",
    "require_directory_without_symlinks",
    "require_regular_source_file",
]
