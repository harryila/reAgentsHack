"""Centralized, side-effect-free repository paths.

No pipeline stage should construct a data or artifact path ad hoc.  All helpers in
this module validate the question identifier and merely return ``Path`` objects;
they never create directories or files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

QuestionStage = Literal[
    "s0",
    "triage_probe",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "s7",
]

_QUESTION_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_STAGES: frozenset[str] = frozenset(
    {"s0", "triage_probe", "s1", "s2", "s3", "s4", "s5", "s6", "s7"}
)


class InvalidQuestionIdError(ValueError):
    """Raised before an unsafe or non-canonical question path can be formed."""


def validate_question_id(question_id: str) -> str:
    """Return a canonical slug or raise a stable validation error."""

    if not isinstance(question_id, str) or not _QUESTION_ID_RE.fullmatch(question_id):
        raise InvalidQuestionIdError(
            "invalid_question_id: expected lowercase kebab-case [a-z0-9-] slug"
        )
    return question_id


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """All planned paths for one repository checkout."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.resolve())

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs" / "questions"

    @property
    def schemas_dir(self) -> Path:
        return self.root / "schemas"

    @property
    def prompts_dir(self) -> Path:
        return self.root / "prompts"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    def config_path(self, question_id: str) -> Path:
        return self.configs_dir / f"{validate_question_id(question_id)}.yaml"

    def patches_path(self, question_id: str) -> Path:
        return self.configs_dir / f"{validate_question_id(question_id)}.patches.yaml"

    def schema_path(self, question_id: str) -> Path:
        return self.schemas_dir / f"extraction.{validate_question_id(question_id)}.schema.json"

    def triage_dir(self, question_id: str) -> Path:
        return self.data_dir / "raw" / "triage" / validate_question_id(question_id)

    def raw_search_dir(self, question_id: str) -> Path:
        return self.data_dir / "raw" / "search" / validate_question_id(question_id)

    def raw_screen_dir(self, question_id: str) -> Path:
        return self.data_dir / "raw" / "screen" / validate_question_id(question_id)

    def raw_map_dir(self, question_id: str) -> Path:
        return self.data_dir / "raw" / "map" / validate_question_id(question_id)

    def extracted_dir(self, question_id: str) -> Path:
        return self.data_dir / "extracted" / validate_question_id(question_id)

    def processed_dir(self, question_id: str) -> Path:
        return self.data_dir / "processed" / validate_question_id(question_id)

    def checkpoint_dir(self, question_id: str, stage: str = "s5") -> Path:
        if stage not in _STAGES:
            raise ValueError(f"invalid_stage: {stage}")
        return self.data_dir / "checkpoints" / validate_question_id(question_id) / stage

    def analysis_dir(self, question_id: str) -> Path:
        return self.artifacts_dir / validate_question_id(question_id) / "analysis"

    def demo_dir(self, question_id: str) -> Path:
        return self.artifacts_dir / validate_question_id(question_id) / "demo"

    def stage_dir(self, question_id: str, stage: QuestionStage | str) -> Path:
        """Return the fixed output directory for a stage without creating it."""

        qid = validate_question_id(question_id)
        mapping = {
            "s0": self.data_dir / "raw" / "smoke",
            "triage_probe": self.triage_dir(qid),
            "s1": self.raw_search_dir(qid),
            "s2": self.raw_screen_dir(qid),
            "s3": self.extracted_dir(qid),
            "s4": self.processed_dir(qid),
            "s5": self.analysis_dir(qid),
            "s6": self.analysis_dir(qid),
            "s7": self.demo_dir(qid),
        }
        try:
            return mapping[stage]
        except KeyError as exc:
            raise ValueError(f"invalid_stage: {stage}") from exc

    def run_record_path(self, question_id: str, stage: QuestionStage | str) -> Path:
        return self.stage_dir(question_id, stage) / "run.json"

    def planned_stage_paths(self, question_id: str) -> dict[str, Path]:
        """Return every planned stage directory, with no filesystem side effects."""

        return {stage: self.stage_dir(question_id, stage) for stage in sorted(_STAGES)}

    def repository_relative(self, path: Path) -> str:
        """Return a POSIX repository-relative path and reject path escape."""

        resolved = path.resolve()
        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise ValueError(f"path_outside_repository: {path}") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
PATHS = ProjectPaths(REPO_ROOT)

