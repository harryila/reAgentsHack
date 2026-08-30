from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from tests.private_cache_support import require_private_cache

from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.postlive_recovery_v4_join_v1 import (
    PostLiveRecoveryV4JoinArtifactV1,
    PostLiveRecoveryV4JoinV1Error,
    build_postlive_recovery_v4_join_from_artifact_v1,
    freeze_postlive_recovery_v4_join_artifact_v1,
)

pytestmark = pytest.mark.private_cache

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RELATIVE = "data/cache/metasyn/contextual-frontier-recovery-v4-posthoc-v1/artifact.json"
SOURCE = ROOT / SOURCE_RELATIVE
NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def _source() -> dict[str, Any]:
    value = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _join(
    *, source: dict[str, Any] | None = None, **kwargs: Any
) -> PostLiveRecoveryV4JoinArtifactV1:
    if source is None:
        return build_postlive_recovery_v4_join_from_artifact_v1(
            repository_root=ROOT,
            posthoc_artifact_path=SOURCE,
            generated_at=NOW,
            target_direction="increase",
            **kwargs,
        )
    return freeze_postlive_recovery_v4_join_artifact_v1(
        posthoc_artifact=source,
        posthoc_artifact_file_sha256=sha256_file(SOURCE),
        generated_at=NOW,
        target_direction="increase",
        **kwargs,
    )


def _authority_values(value: Any) -> list[bool]:
    found: list[bool] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("_authority"):
                assert isinstance(item, bool)
                found.append(item)
            found.extend(_authority_values(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_authority_values(item))
    return found


def test_actual_posthoc_projection_joins_without_runtime_success_or_authority() -> None:
    require_private_cache(SOURCE_RELATIVE)
    artifact = _join()

    assert artifact.status == "composed_offline_mechanics_completed_non_authorizing"
    assert artifact.source_v4_terminal_status == "contextual_validation_failed_closed"
    assert not artifact.source_v4_runtime_workspace_success
    assert artifact.canonicalizer_provider_calls_made == 0
    assert artifact.upstream_v4_provider_attempt_count == 1
    assert artifact.upstream_v4_provider_response_completed
    assert artifact.source_posthoc_external_replay_performed
    assert artifact.source_posthoc_external_replay_sha256 is not None
    assert artifact.post_hoc_syntactic_canonicalization
    assert artifact.post_hoc_source_span_repair
    assert artifact.typed_graph_mechanics_observed
    assert artifact.canonicalization_pipeline_sha256 != artifact.integration_pipeline_sha256
    assert len(artifact.source_repair_changes) > 0
    assert len(artifact.evidence_graph.publications) == 1
    assert len(artifact.evidence_graph.outcome_estimates) == 1
    assert len(artifact.audit_mechanics.audit_candidates) == 1
    assert artifact.condition_mechanics.status == "not_scientifically_defined"
    assert "post_hoc_source_span_repair" in artifact.blockers
    assert "post_hoc_syntactic_canonicalization" in artifact.blockers
    assert "source_v4_terminal_failed_closed" in artifact.blockers
    assert not any(_authority_values(artifact.model_dump(mode="json")))
    assert not artifact.release_authorizing


def test_join_binds_actual_native_projection_and_replays_synthesis() -> None:
    require_private_cache(SOURCE_RELATIVE)
    source = _source()
    artifact = _join(source=source)
    projection = source["evaluation"]["native_projection"]

    assert artifact.native_projection_sha256 == projection["projection_sha256"]
    assert artifact.fragment_sha256 == projection["fragment_sha256"]
    assert artifact.evidence_graph.model_dump(mode="json") == projection["fragment"]["graph"]
    assert artifact.synthesis == projection["quantitative_mechanics_result"]
    assert artifact.audit_mechanics.sequential_state.graph_sha256 == (
        artifact.evidence_graph_sha256
    )
    assert artifact.audit_mechanics.sequential_state.synthesis_sha256 == (artifact.synthesis_sha256)


def test_source_span_repair_ledger_tamper_fails_closed() -> None:
    require_private_cache(SOURCE_RELATIVE)
    source = deepcopy(_source())
    source["canonicalization_changes"][0]["json_pointer"] += "/tampered"
    source["artifact_sha256"] = hash_canonical(
        {key: value for key, value in source.items() if key != "artifact_sha256"}
    )

    with pytest.raises(PostLiveRecoveryV4JoinV1Error, match="source_replay_mismatch"):
        _join(source=source)


def test_failed_v4_terminal_lineage_cannot_be_substituted() -> None:
    require_private_cache(SOURCE_RELATIVE)
    source = deepcopy(_source())
    source["immutable_v4_terminal_sha256"] = "f" * 64
    source["artifact_sha256"] = hash_canonical(
        {key: value for key, value in source.items() if key != "artifact_sha256"}
    )

    with pytest.raises(PostLiveRecoveryV4JoinV1Error, match="immutable_lineage_mismatch"):
        _join(source=source)


def test_missing_posthoc_repair_blocker_fails_closed() -> None:
    require_private_cache(SOURCE_RELATIVE)
    source = deepcopy(_source())
    source["blockers"] = []
    source["artifact_sha256"] = hash_canonical(
        {key: value for key, value in source.items() if key != "artifact_sha256"}
    )

    with pytest.raises(PostLiveRecoveryV4JoinV1Error, match="repair_blocker_missing"):
        _join(source=source)


def test_join_authority_tamper_fails_even_with_recomputed_hash() -> None:
    require_private_cache(SOURCE_RELATIVE)
    raw = _join().model_dump(mode="json")
    raw["scientific_synthesis_authority"] = True
    raw["artifact_sha256"] = hash_canonical(
        {key: value for key, value in raw.items() if key != "artifact_sha256"}
    )

    with pytest.raises(ValidationError):
        PostLiveRecoveryV4JoinArtifactV1.model_validate(raw)


def test_join_schema_has_no_v1_runtime_success_aliases() -> None:
    require_private_cache(SOURCE_RELATIVE)
    raw = _join().model_dump(mode="json")

    assert "runtime_workspace_validation_sha256" not in raw
    assert "terminal_report_sha256" not in raw
    assert "successful_validation_sha256" not in raw
    assert "terminal_status" not in raw
    assert raw["source_v4_terminal_status"] == "contextual_validation_failed_closed"


def test_explicit_moderator_is_exploratory_and_insufficient() -> None:
    require_private_cache(SOURCE_RELATIVE)
    artifact = _join(prespecified_moderators=["dose"])

    assert artifact.condition_mechanics.analysis_executed
    assert artifact.condition_mechanics.status == "executed_insufficient"
    assert not artifact.condition_mechanics.condition_claim_authority


def test_status_cli_preserves_failed_source_and_non_authorizing_boundary(
    tmp_path: Path,
) -> None:
    require_private_cache(SOURCE_RELATIVE)
    artifact_path = tmp_path / "join.json"
    artifact_path.write_text(
        json.dumps(_join().model_dump(mode="json")),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_postlive_recovery_v4_join_v1.py"),
            "status",
            "--input",
            str(artifact_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    status = json.loads(completed.stdout)

    assert status["source_v4_terminal_status"] == "contextual_validation_failed_closed"
    assert not status["source_v4_runtime_workspace_success"]
    assert status["canonicalizer_provider_calls_made"] == 0
    assert status["upstream_v4_provider_attempt_count"] == 1
    assert status["upstream_v4_provider_response_completed"]
    assert status["source_posthoc_external_replay_performed"]
    assert status["post_hoc_source_span_repair"]
    assert not status["extraction_accuracy_authority"]
    assert not status["scientific_synthesis_authority"]
    assert not status["claim_release_authority"]
    assert not status["release_authorizing"]


def test_cli_build_and_validate_externally_replay_posthoc_source(tmp_path: Path) -> None:
    require_private_cache(SOURCE_RELATIVE)
    output = tmp_path / "join.json"
    script = str(ROOT / "scripts/run_postlive_recovery_v4_join_v1.py")
    built = subprocess.run(
        [
            sys.executable,
            script,
            "build",
            "--repository-root",
            str(ROOT),
            "--posthoc-artifact",
            str(SOURCE),
            "--output",
            str(output),
            "--generated-at",
            NOW.isoformat(),
            "--target-direction",
            "increase",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    built_status = json.loads(built.stdout)
    assert built_status["source_posthoc_external_replay_performed"]
    assert output.is_file()

    validated = subprocess.run(
        [
            sys.executable,
            script,
            "validate",
            "--repository-root",
            str(ROOT),
            "--posthoc-artifact",
            str(SOURCE),
            "--input",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(validated.stdout)["artifact_sha256"] == (built_status["artifact_sha256"])
