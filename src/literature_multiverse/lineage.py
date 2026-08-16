"""Canonical hashing, atomic persistence, and explicit lineage validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from literature_multiverse.models import (
    ArtifactRef,
    M4SourceCheckpoint,
    RunRecord,
    UpstreamRef,
    canonical_model_sha256,
)
from literature_multiverse.paths import ProjectPaths

_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|authorization|bearer|token|secret)(=|:)([^\s]+)"),
    re.compile(r"\b(?:sk-ant-|gxl_)[A-Za-z0-9_-]+"),
)
_STAGE_ORDER = {
    "s0": 0,
    "triage_probe": 0,
    "s1": 1,
    "s2": 2,
    "s3": 3,
    "s4": 4,
    "s5": 5,
    "s6": 6,
    "s7": 7,
}
_PRODUCTION_STAGES = frozenset({"s3", "s4", "s5", "s6", "s7"})


class LineageError(ValueError):
    code = "lineage_error"

    def __init__(self, detail: str = "") -> None:
        suffix = f":{detail}" if detail else ""
        super().__init__(f"{self.code}{suffix}")


class MissingArtifactError(LineageError):
    code = "lineage_missing_artifact"


class MissingHashError(LineageError):
    code = "lineage_missing_hash"


class HashMismatchError(LineageError):
    code = "lineage_hash_mismatch"


class StaleInputError(LineageError):
    code = "lineage_stale_input"


class MixedContractError(LineageError):
    code = "lineage_mixed_extraction_tuple"


class DirtyLineageError(LineageError):
    code = "lineage_dirty_code"


class InvalidUpstreamError(LineageError):
    code = "lineage_invalid_upstream"


class OutputExistsError(LineageError):
    code = "lineage_output_exists"


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("canonical_json_requires_timezone_aware_datetime")
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical_json_forbids_nonfinite_float")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize scientific identity inputs deterministically and without whitespace."""

    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def hash_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    if not path.is_file():
        raise MissingArtifactError(path.as_posix())
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extraction_cfghash(
    config: BaseModel | Mapping[str, Any],
    rendered_prompt: str | bytes,
    output_schema: BaseModel | Mapping[str, Any] | str | bytes,
) -> str:
    """Hash canonical config plus exact prompt and canonical schema with domain tags."""

    prompt_bytes = (
        rendered_prompt.encode("utf-8")
        if isinstance(rendered_prompt, str)
        else rendered_prompt
    )
    if isinstance(output_schema, bytes):
        schema_bytes = output_schema
    elif isinstance(output_schema, str):
        schema_bytes = output_schema.encode("utf-8")
    else:
        schema_bytes = canonical_json_bytes(output_schema)
    digest = hashlib.sha256()
    for label, payload in (
        (b"config\0", canonical_json_bytes(config)),
        (b"prompt\0", prompt_bytes),
        (b"schema\0", schema_bytes),
    ):
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, content: bytes, *, force: bool = False) -> None:
    """Durably replace one file using a same-directory temporary file."""

    if path.exists() and not force:
        raise OutputExistsError(path.as_posix())
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, content: str, *, force: bool = False) -> None:
    atomic_write_bytes(path, content.encode("utf-8"), force=force)


def atomic_write_json(path: Path, value: Any, *, force: bool = False) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n", force=force)


def atomic_write_jsonl(path: Path, rows: Iterable[Any], *, force: bool = False) -> None:
    content = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    atomic_write_bytes(path, content, force=force)


def redact_text(value: str) -> str:
    redacted = value
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 3:
            redacted = pattern.sub(
                lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", redacted
            )
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redact_command_argv(argv: Sequence[str]) -> list[str]:
    return [redact_text(argument) for argument in argv]


def write_run_record(path: Path, record: RunRecord, *, force: bool = False) -> RunRecord:
    """Redact argv defensively, validate again, then atomically write ``run.json``."""

    safe_record = record.model_copy(
        update={"command_argv": redact_command_argv(record.command_argv)}
    )
    safe_record = RunRecord.model_validate(safe_record.model_dump())
    atomic_write_json(path, safe_record, force=force)
    return safe_record


def artifact_ref(path: Path, *, root: Path, rows: int | None = None) -> ArtifactRef:
    if not path.is_file():
        raise MissingArtifactError(path.as_posix())
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact_outside_repository:{path}") from exc
    return ArtifactRef(
        path=relative,
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        rows=rows,
    )


def verify_artifact(reference: ArtifactRef, *, root: Path) -> Path:
    if not reference.sha256:
        raise MissingHashError(reference.path)
    path = root / reference.path
    if not path.is_file():
        raise MissingArtifactError(reference.path)
    observed = sha256_file(path)
    if observed != reference.sha256:
        raise HashMismatchError(f"{reference.path}:expected={reference.sha256}:observed={observed}")
    if path.stat().st_size != reference.bytes:
        raise HashMismatchError(f"{reference.path}:byte_count")
    return path


def assert_uniform_extraction_tuple(rows: Iterable[Any]) -> tuple[str, str, str]:
    """Reject mixed prompt/schema/cfghash values before normalization or analysis."""

    tuples: set[tuple[str | None, str | None, str | None]] = set()
    for row in rows:
        if isinstance(row, BaseModel):
            data = row.model_dump()
        elif isinstance(row, Mapping):
            data = row
        else:
            raise TypeError("extraction_tuple_row_must_be_model_or_mapping")
        tuples.add((data.get("prompt_version"), data.get("schema_version"), data.get("cfghash")))
    if not tuples:
        raise MissingHashError("extraction_tuple_empty")
    if len(tuples) != 1:
        rendered = sorted(repr(item) for item in tuples)
        raise MixedContractError("|".join(rendered))
    prompt_version, schema_version, cfghash = next(iter(tuples))
    if not prompt_version or not schema_version or not cfghash:
        raise MissingHashError("extraction_tuple")
    return prompt_version, schema_version, cfghash


def validate_upstream_chain(
    *,
    current_stage: str,
    upstream: Sequence[UpstreamRef],
    root: Path,
    expected_config_sha256: str,
    allow_dirty_demo: bool = False,
) -> list[RunRecord]:
    """Re-open and validate every declared upstream run record and its hash."""

    if current_stage not in _STAGE_ORDER:
        raise InvalidUpstreamError(f"unknown_stage={current_stage}")
    records: list[RunRecord] = []
    for reference in upstream:
        if reference.stage == "triage_probe" and current_stage in _PRODUCTION_STAGES:
            raise InvalidUpstreamError("triage_probe_cannot_feed_production")
        if (
            reference.stage not in _STAGE_ORDER
            or _STAGE_ORDER[reference.stage] >= _STAGE_ORDER[current_stage]
        ):
            raise InvalidUpstreamError(f"stage_order={reference.stage}->{current_stage}")
        path = root / reference.run_record_path
        if not path.is_file():
            raise MissingArtifactError(reference.run_record_path)
        observed_hash = sha256_file(path)
        if observed_hash != reference.run_record_sha256:
            raise HashMismatchError(reference.run_record_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            run = RunRecord.model_validate(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            raise InvalidUpstreamError(f"invalid_run_record={reference.run_record_path}") from exc
        if run.run_id != reference.run_id or run.stage != reference.stage:
            raise InvalidUpstreamError(f"run_identity={reference.run_record_path}")
        if run.status != "complete":
            raise InvalidUpstreamError(f"run_not_complete={reference.run_record_path}")
        if run.config_sha256 != expected_config_sha256:
            raise StaleInputError(reference.run_record_path)
        if current_stage == "s7" and run.code_version.startswith("dirty:") and not allow_dirty_demo:
            raise DirtyLineageError(reference.run_record_path)
        records.append(run)
    return records


def source_tree_sha256(root: Path) -> str:
    """Hash code-owned files only; environment/secrets and scientific data are excluded."""

    candidates: list[Path] = []
    for relative_root in ("src", "scripts", "app"):
        directory = root / relative_root
        if directory.is_dir():
            candidates.extend(path for path in directory.rglob("*.py") if path.is_file())
    for name in ("pyproject.toml", ".python-version"):
        path = root / name
        if path.is_file():
            candidates.append(path)
    digest = hashlib.sha256()
    for path in sorted(set(candidates), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def code_version(root: Path) -> str:
    """Return commit SHA for a clean tree or ``dirty:<code-owned-source-hash>``."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return f"dirty:{source_tree_sha256(root)}"
    return commit if not dirty else f"dirty:{source_tree_sha256(root)}"


def canonical_checkpoint_archive_path(
    paths: ProjectPaths, checkpoint: M4SourceCheckpoint
) -> Path:
    digest = canonical_model_sha256(checkpoint)
    return paths.checkpoint_dir(checkpoint.question_id, "s5") / f"{digest}.json"


def frozen_run_identity(
    paths: ProjectPaths, checkpoint: M4SourceCheckpoint
) -> dict[str, Any]:
    """Deterministic identity fields for explicit incomplete finalization."""

    digest = canonical_model_sha256(checkpoint)
    archive_path = canonical_checkpoint_archive_path(paths, checkpoint)
    return {
        "run_id": f"s5-frozen-{digest[:16]}",
        "started_at": checkpoint.source_started_at,
        "completed_at": checkpoint.checkpointed_at,
        "checkpoint_sha256": digest,
        "command_argv": [
            "scripts/s5_analyze.py",
            "--question",
            checkpoint.question_id,
            "--finalize-incomplete-from",
            paths.repository_relative(archive_path),
        ],
    }
