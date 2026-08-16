from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from literature_multiverse.lineage import (
    DirtyLineageError,
    HashMismatchError,
    InvalidUpstreamError,
    MissingArtifactError,
    MissingHashError,
    MixedContractError,
    OutputExistsError,
    StaleInputError,
    artifact_ref,
    assert_uniform_extraction_tuple,
    atomic_write_json,
    canonical_json_bytes,
    extraction_cfghash,
    frozen_run_identity,
    hash_canonical,
    sha256_file,
    source_tree_sha256,
    validate_upstream_chain,
    verify_artifact,
    write_run_record,
)
from literature_multiverse.models import (
    ArtifactRef,
    M4CheckpointFrozenIncomplete,
    RunRecord,
    UpstreamRef,
    canonical_model_sha256,
)
from literature_multiverse.paths import InvalidQuestionIdError, ProjectPaths


def run_record_payload(
    hash64: str,
    *,
    stage: str = "s2",
    run_id: str = "run-1",
    code_version: str = "abc123",
) -> dict:
    return {
        "run_id": run_id,
        "question_id": "fixture-a",
        "stage": stage,
        "stage_version": "1",
        "status": "complete",
        "completion_mode": "normal",
        "checkpoint_sha256": None,
        "started_at": datetime(2026, 8, 15, 10, 0, tzinfo=UTC),
        "completed_at": datetime(2026, 8, 15, 10, 1, tzinfo=UTC),
        "code_version": code_version,
        "command_argv": ["script.py", "API_KEY=super-secret"],
        "config_path": "configs/questions/fixture-a.yaml",
        "config_sha256": hash64,
        "prompt_path": None,
        "prompt_sha256": None,
        "schema_path": None,
        "schema_sha256": None,
        "cfghash": None,
        "upstream": [],
        "inputs": [],
        "outputs": [],
        "external_result_ids": {},
        "counts": {"papers": 1},
        "warnings": [],
    }


def test_canonical_json_and_hash_are_mapping_order_independent() -> None:
    left = {"b": [2, 1], "a": {"value": "é"}}
    right = {"a": {"value": "é"}, "b": [2, 1]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert hash_canonical(left) == hash_canonical(right)


def test_extraction_cfghash_binds_config_prompt_and_schema() -> None:
    baseline = extraction_cfghash({"eligibility": ["a"]}, b"prompt", {"type": "object"})
    variants = {
        extraction_cfghash({"eligibility": ["b"]}, b"prompt", {"type": "object"}),
        extraction_cfghash({"eligibility": ["a"]}, b"prompt!", {"type": "object"}),
        extraction_cfghash({"eligibility": ["a"]}, b"prompt", {"type": "array"}),
    }
    assert baseline not in variants
    assert len(variants) == 3


def test_atomic_json_refuses_overwrite_without_force_and_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "value.json"
    atomic_write_json(target, {"value": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}
    with pytest.raises(OutputExistsError):
        atomic_write_json(target, {"value": 2})
    atomic_write_json(target, {"value": 2}, force=True)
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 2}
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_artifact_hash_validation_has_distinct_missing_and_mismatch_errors(
    tmp_path: Path,
) -> None:
    missing = ArtifactRef(path="missing.bin", sha256="a" * 64, bytes=0, rows=None)
    with pytest.raises(MissingArtifactError):
        verify_artifact(missing, root=tmp_path)

    path = tmp_path / "present.bin"
    path.write_bytes(b"real")
    mismatch = ArtifactRef(path="present.bin", sha256="a" * 64, bytes=4, rows=None)
    with pytest.raises(HashMismatchError):
        verify_artifact(mismatch, root=tmp_path)
    reference = artifact_ref(path, root=tmp_path)
    assert verify_artifact(reference, root=tmp_path) == path


def test_mixed_or_missing_extraction_tuple_fails_stably(hash64: str) -> None:
    good = {"prompt_version": "1", "schema_version": "1", "cfghash": hash64}
    assert assert_uniform_extraction_tuple([good, good]) == ("1", "1", hash64)
    with pytest.raises(MixedContractError):
        assert_uniform_extraction_tuple([good, {**good, "prompt_version": "2"}])
    with pytest.raises(MissingHashError):
        assert_uniform_extraction_tuple([{**good, "cfghash": None}])
    with pytest.raises(MissingHashError):
        assert_uniform_extraction_tuple([])


def test_run_json_is_redacted_atomic_and_revalidated(tmp_path: Path, hash64: str) -> None:
    path = tmp_path / "runs" / "run.json"
    record = RunRecord.model_validate(run_record_payload(hash64))
    written = write_run_record(path, record)
    assert written.command_argv[-1] == "API_KEY=[REDACTED]"
    serialized = path.read_text(encoding="utf-8")
    assert "super-secret" not in serialized
    assert json.loads(serialized)["run_record_version"] == "1"
    with pytest.raises(OutputExistsError):
        write_run_record(path, record)


def write_upstream(
    root: Path,
    record: RunRecord,
    *,
    relative: str,
) -> UpstreamRef:
    path = root / relative
    write_run_record(path, record)
    return UpstreamRef(
        stage=record.stage,
        run_id=record.run_id,
        run_record_path=relative,
        run_record_sha256=sha256_file(path),
    )


def test_upstream_validation_rejects_stale_triage_and_dirty_demo(
    tmp_path: Path, hash64: str
) -> None:
    s2 = RunRecord.model_validate(run_record_payload(hash64, stage="s2"))
    reference = write_upstream(tmp_path, s2, relative="runs/s2/run.json")
    assert validate_upstream_chain(
        current_stage="s3",
        upstream=[reference],
        root=tmp_path,
        expected_config_sha256=hash64,
    ) == [s2.model_copy(update={"command_argv": ["script.py", "API_KEY=[REDACTED]"]})]
    with pytest.raises(StaleInputError):
        validate_upstream_chain(
            current_stage="s3",
            upstream=[reference],
            root=tmp_path,
            expected_config_sha256="b" * 64,
        )

    triage = RunRecord.model_validate(
        run_record_payload(hash64, stage="triage_probe", run_id="triage-run")
    )
    triage_reference = write_upstream(
        tmp_path,
        triage,
        relative="runs/triage/run.json",
    )
    with pytest.raises(InvalidUpstreamError, match="triage_probe_cannot_feed"):
        validate_upstream_chain(
            current_stage="s3",
            upstream=[triage_reference],
            root=tmp_path,
            expected_config_sha256=hash64,
        )

    dirty = RunRecord.model_validate(
        run_record_payload(
            hash64,
            stage="s5",
            run_id="dirty-s5",
            code_version=f"dirty:{hash64}",
        )
    )
    dirty_reference = write_upstream(tmp_path, dirty, relative="runs/s5/run.json")
    with pytest.raises(DirtyLineageError):
        validate_upstream_chain(
            current_stage="s7",
            upstream=[dirty_reference],
            root=tmp_path,
            expected_config_sha256=hash64,
        )
    assert validate_upstream_chain(
        current_stage="s7",
        upstream=[dirty_reference],
        root=tmp_path,
        expected_config_sha256=hash64,
        allow_dirty_demo=True,
    )


def test_source_tree_hash_changes_with_code_owned_alias_table(tmp_path: Path) -> None:
    source = tmp_path / "src" / "literature_multiverse"
    source.mkdir(parents=True)
    aliases = source / "models.py"
    aliases.write_text('DIRECTION_ALIASES = {"null": "no_effect"}\n', encoding="utf-8")
    baseline = source_tree_sha256(tmp_path)
    aliases.write_text(
        'DIRECTION_ALIASES = {"null": "no_effect", "bad": "increase"}\n',
        encoding="utf-8",
    )
    assert source_tree_sha256(tmp_path) != baseline


def test_frozen_checkpoint_identity_is_reproducible_and_content_addressed(
    tmp_path: Path, source_checkpoint
) -> None:
    paths = ProjectPaths(tmp_path)
    first = frozen_run_identity(paths, source_checkpoint)
    second = frozen_run_identity(paths, source_checkpoint)
    assert first == second
    assert first["started_at"] == source_checkpoint.source_started_at
    assert first["completed_at"] == source_checkpoint.checkpointed_at
    assert first["checkpoint_sha256"] == canonical_model_sha256(source_checkpoint)
    assert first["command_argv"][-1].endswith(f"{first['checkpoint_sha256']}.json")

    wrapper = M4CheckpointFrozenIncomplete(
        source_checkpoint_sha256=first["checkpoint_sha256"],
        checkpoint=source_checkpoint,
    )
    assert canonical_model_sha256(wrapper) == canonical_model_sha256(wrapper.model_copy())


def test_planned_paths_are_side_effect_free_and_question_ids_cannot_escape(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths(tmp_path)
    planned = paths.planned_stage_paths("fixture-a")
    assert set(planned) == {"s0", "triage_probe", "s1", "s2", "s3", "s4", "s5", "s6", "s7"}
    assert not (tmp_path / "data").exists()
    with pytest.raises(InvalidQuestionIdError):
        paths.config_path("../escape")

