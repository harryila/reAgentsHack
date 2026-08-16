from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from literature_multiverse.config import QuestionConfig, load_question_config
from literature_multiverse.models import (
    CheckpointArtifactHashes,
    CheckpointBudgets,
    CheckpointResult,
    M4SourceCheckpoint,
    make_finding_id,
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def fixture_configs(repo_root: Path) -> dict[str, QuestionConfig]:
    directory = repo_root / "configs" / "questions"
    return {
        path.stem: load_question_config(path, require_locked=True)
        for path in sorted(directory.glob("fixture-*.yaml"))
    }


@pytest.fixture
def fixture_config(fixture_configs: dict[str, QuestionConfig]) -> QuestionConfig:
    return fixture_configs["fixture-a"]


@pytest.fixture
def hash64() -> str:
    return "a" * 64


@pytest.fixture
def successful_paper_payload(hash64: str) -> dict[str, Any]:
    return {
        "paper_id": "doc:doc-1",
        "doc_id": "doc-1",
        "alternate_doc_ids": [],
        "doi": None,
        "pmid": None,
        "title": "A controlled fixture study",
        "first_author": "Example",
        "pub_year": 2024,
        "source": "fixture",
        "article_type": "research-article",
        "query_families": ["direct"],
        "search_result_ids": ["search-1"],
        "content_tier": "full_text",
        "publication_status": "peer_reviewed",
        "screen_status": "included",
        "screen_reason": None,
        "dedupe_cluster_id": "cluster-1",
        "dedupe_preferred": True,
        "map_status": "success",
        "eligible": True,
        "exclusion_reason": None,
        "map_result_id": "map-1",
        "raw_artifact_path": "data/raw/map/fixture-a/map-1.txt",
        "raw_finding_count": 1,
        "accepted_finding_count": 1,
        "quarantined_finding_count": 0,
        "failure_code": None,
        "dataset_or_cohort_id": None,
        "prompt_version": "1",
        "schema_version": "1",
        "config_sha256": hash64,
        "cfghash": hash64,
        "created_at": datetime(2026, 8, 15, 12, 0, tzinfo=UTC),
    }


@pytest.fixture
def finding_payload(hash64: str) -> dict[str, Any]:
    identity = {
        "paper_id": "doc:doc-1",
        "map_result_id": "map-1",
        "array_position": 0,
        "outcome_name": "peak_power",
        "timepoint_raw": "post",
        "dose_raw": "100 mg",
        "effect_direction": "increase",
    }
    return {
        "finding_id": make_finding_id(**identity),
        **identity,
        "doc_id": "doc-1",
        "prompt_version": "1",
        "schema_version": "1",
        "cfghash": hash64,
        "grounding_status": "exact",
        "evidence_section": "Results",
        "section_flagged": False,
        "normalization_warnings": [],
        "study_type": "randomized controlled trial",
        "species": "human",
        "model": None,
        "population_state": "healthy",
        "intervention": "synthetic intervention",
        "intervention_class": "synthetic",
        "comparator": "control",
        "duration_raw": "4 weeks",
        "timing_context": "chronic",
        "outcome_family": "performance",
        "effect_size_raw": None,
        "p_value": 0.04,
        "significant": True,
        "sample_size": 40,
        "evidence_quote": "Peak power increased relative to control.",
        "evidence_lines": ["L10"],
        "confidence": 0.9,
        "moderators": {
            "dose_regime": "low",
            "training_status": "trained",
            "population_state": "healthy",
            "timing_context": "chronic",
        },
    }


@pytest.fixture
def source_checkpoint(hash64: str) -> M4SourceCheckpoint:
    return M4SourceCheckpoint(
        source_run_id="s5-source-run",
        source_started_at=datetime(2026, 8, 15, 12, 0, tzinfo=UTC),
        checkpointed_at=datetime(2026, 8, 15, 12, 5, tzinfo=UTC),
        question_id="fixture-b-incomplete",
        config_sha256=hash64,
        code_version=f"dirty:{hash64}",
        cohort_sha256=hash64,
        g3_gate_sha256=hash64,
        input_hashes={"papers": hash64, "findings": hash64},
        seed=20260815,
        registered_budgets=CheckpointBudgets(
            bootstrap_count=200,
            permutation_success_count=100,
            permutation_max_attempts=125,
        ),
        completed_bootstrap_indices=[0],
        completed_permutation_attempt_indices=[0],
        successful_permutation_indices=[0],
        bootstrap_results=[CheckpointResult(index=0, status="success", result={}, error_code=None)],
        permutation_results=[
            CheckpointResult(index=0, status="success", result={}, error_code=None)
        ],
        guard_failures=[],
        artifact_hashes=CheckpointArtifactHashes(
            descriptive_inputs=hash64,
            descriptive_outputs=hash64,
            residual_inputs=hash64,
            residual_outputs=hash64,
            evidence_gap_inputs=hash64,
            evidence_gap_outputs=hash64,
        ),
    )


def clone(value: dict[str, Any]) -> dict[str, Any]:
    """Small explicit helper for tests that mutate nested contract payloads."""

    return deepcopy(value)
