from __future__ import annotations

from argparse import Namespace
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from scripts.s5_analyze import (
    FIXTURE_INTERRUPTION_MODE,
    FixtureInjectedInterruption,
    _checkpoint_writer_with_fixture_interrupt,
    _fixture_fault_mode,
)

from literature_multiverse.analysis import (
    AnalysisContractError,
    G3TrustBlockedError,
    InferenceOverrides,
    analyze_s5,
    compute_deterministic_components,
    derive_primary_cohort,
    deterministic_checkpoint_hashes,
    finalize_incomplete_s5,
    read_parquet_records,
    write_analysis_bundle,
)
from literature_multiverse.cohort import cohort_sha256
from literature_multiverse.config import config_sha256, load_question_config
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.models import (
    CheckpointBudgets,
    CheckpointResult,
    M4SourceCheckpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def _config():
    return load_question_config(ROOT / "configs/questions/fixture-a.yaml", require_locked=True)


def _data() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    papers: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []
    decisions: list[dict[str, str]] = []
    directions = ("increase", "no_effect", "decrease")
    for index in range(30):
        paper_id = f"doc:p{index:02d}"
        direction = directions[index % 3]
        dose = "high" if index % 2 == 0 else "low"
        training = "trained" if index % 2 == 0 else "untrained"
        finding_id = f"f{index:02d}"
        papers.append(
            {
                "paper_id": paper_id,
                "doc_id": f"p{index:02d}",
                "title": f"Fixture paper {index}",
                "doi": None,
                "screen_status": "included",
                "map_status": "success",
                "eligible": True,
            }
        )
        findings.append(
            {
                "finding_id": finding_id,
                "paper_id": paper_id,
                "effect_direction": direction,
                "outcome_family": "performance",
                "outcome_name": "peak_power" if index % 2 == 0 else "fatigue_time",
                "grounding_status": "exact",
                "section_flagged": False,
                "evidence_quote": f"supported quote {index}",
                "evidence_lines": ["L10"],
                "evidence_section": "Results",
                "intervention_class": "synthetic",
                "comparator": "control",
                "population_state": "healthy",
                "timing_context": "acute",
                "timepoint_raw": "week 4",
                "mod__dose_regime": dose,
                "mod__training_status": training,
                "mod__population_state": "healthy",
                "mod__timing_context": "acute",
            }
        )
        decisions.append(
            {
                "finding_id": finding_id,
                "model_status": "agree",
                "adjudication": "none",
            }
        )
    verification: dict[str, object] = {
        "verification_version": "1",
        "provider": "fixture",
        "model": "fixture-verifier",
        "prompt_version": "v1",
        "prompt_sha256": "a" * 64,
        "requested_finding_ids": [row["finding_id"] for row in findings],
        "decisions": decisions,
    }
    return papers, findings, verification


def _g3(*, trust: bool = True, story: bool = True) -> dict[str, object]:
    return {
        "trust_passed": trust,
        "story_passed": story,
        "action": "run_m4" if story else "select_variant_b_story",
    }


def _comparison() -> dict[str, object]:
    return {
        "eligible": True,
        "reason": None,
        "n_findings": 30,
        "n_papers": 30,
        "coverage_papers": 1.0,
        "global_mode": "increase",
        "agreement_q": 0.40,
        "agreement_p": 0.80,
        "absolute_gain": 0.40,
        "support_papers": {"low": 15, "high": 15},
        "contrast": {
            "level_a": "low",
            "direction_a": "decrease",
            "n_papers_a": 15,
            "level_b": "high",
            "direction_b": "increase",
            "n_papers_b": 15,
        },
    }


def _inference(*, signal_passes: bool = True) -> InferenceOverrides:
    comparison = _comparison()
    signal = {
        "moderator": "dose_regime",
        "k": 5,
        "delta_ll": 0.08 if signal_passes else 0.005,
        "positive_folds": 4,
        "narrated_level_support": [15, 15],
        "comparison": comparison,
        "all_valid_sensitivity": {
            "positive_gain": True,
            "directions_preserved": True,
        },
    }
    null = {
        "moderator": "training_status",
        "k": 5,
        "delta_ll": 0.0,
        "positive_folds": 1,
        "narrated_level_support": [15, 15],
        "comparison": comparison,
        "all_valid_sensitivity": {
            "positive_gain": False,
            "directions_preserved": False,
        },
    }
    return InferenceOverrides(
        moderator_results=[signal, null],
        permutation={
            "status": "complete",
            "success_count": 100,
            "attempt_count": 100,
            "p_values": {
                "dose_regime": {"raw": 0.01, "westfall_young": 0.03},
                "training_status": {"raw": 0.8, "westfall_young": 0.9},
            },
        },
        bootstrap_stability={
            "dose_regime": {
                "n_bootstraps": 200,
                "pattern_fraction": 0.8,
                "eligible_fraction": 0.9,
                "top3_fraction": 0.9,
            },
            "training_status": {
                "n_bootstraps": 200,
                "pattern_fraction": 0.1,
                "eligible_fraction": 0.9,
                "top3_fraction": 0.2,
            },
        },
    )


def test_g3_trust_failure_refuses_scientific_analysis() -> None:
    papers, findings, verification = _data()
    with pytest.raises(G3TrustBlockedError, match="g3_trust_failed"):
        analyze_s5(
            config=_config(),
            papers=papers,
            findings=findings,
            verification=verification,
            g3_gate=_g3(trust=False),
        )


def test_story_failure_writes_typed_variant_b_without_running_m4() -> None:
    papers, findings, verification = _data()
    bundle = analyze_s5(
        config=_config(),
        papers=papers,
        findings=findings,
        verification=verification,
        g3_gate=_g3(story=False),
    )
    gate = bundle.json_artifacts["m4_gate.json"]
    headline = bundle.json_artifacts["headline.json"]
    assert gate["status"] == "not_run"
    assert gate["cohort_hash"] == cohort_sha256(bundle.primary_rows)
    assert gate["selected_variant"] == "B"
    assert headline["selection_reason"] == "g3_story_not_viable"
    assert headline["global_baseline"]["modal_direction"] is None
    assert bundle.json_artifacts["permutation.json"]["status"] == "not_run"
    assert bundle.json_artifacts["tree.json"]["status"] == "not_run"
    assert bundle.trace == {"status": "not_run", "reason": "g3_story_not_viable"}
    assert all(row["status"] == "not_run" for row in bundle.table_artifacts["moderators.parquet"])


def test_complete_m4_selects_a_only_when_every_rule_passes() -> None:
    papers, findings, verification = _data()
    passing = analyze_s5(
        config=_config(),
        papers=papers,
        findings=findings,
        verification=verification,
        g3_gate=_g3(),
        inference_overrides=_inference(signal_passes=True),
    )
    assert passing.json_artifacts["m4_gate.json"]["selected_variant"] == "A"
    assert passing.json_artifacts["headline.json"]["narrative_variant"] == "A"
    assert passing.json_artifacts["headline.json"]["global_baseline"] == {
        "modal_direction": "increase",
        "agreement_q": 0.40,
    }
    assert passing.trace is None
    root_stability = passing.json_artifacts["tree.json"]["root_split_bootstrap"]
    assert root_stability["status"] == "complete"
    assert root_stability["n_bootstraps"] == 200
    assert len(root_stability["root_splits"]) == 200

    failed = analyze_s5(
        config=_config(),
        papers=papers,
        findings=findings,
        verification=verification,
        g3_gate=_g3(),
        inference_overrides=_inference(signal_passes=False),
    )
    assert failed.json_artifacts["m4_gate.json"]["selected_variant"] == "B"
    assert failed.json_artifacts["headline.json"]["selection_reason"] == "m4_no_moderator"
    assert failed.trace == {"status": "not_run", "reason": "m4_selected_variant_b"}


def _checkpoint_fixture():
    config = _config()
    papers, findings, verification = _data()
    g3_gate = _g3()
    primary = derive_primary_cohort(
        papers,
        findings,
        verification,
        primary_family="performance",
    )
    components = compute_deterministic_components(
        config=config,
        papers=papers,
        findings=findings,
        primary_rows=primary,
        inventory_reason="m4_incomplete",
    )
    artifact_hashes = deterministic_checkpoint_hashes(
        primary_rows=primary,
        papers=papers,
        findings=findings,
        components=components,
        config=config,
    )
    inputs = {"papers": "b" * 64, "findings": "c" * 64}
    checkpoint = M4SourceCheckpoint(
        source_run_id="fixture-interrupted-run",
        source_started_at=datetime.fromisoformat("2026-08-15T12:00:00-07:00"),
        checkpointed_at=datetime.fromisoformat("2026-08-15T12:10:00-07:00"),
        question_id=config.question_id,
        config_sha256=config_sha256(config),
        code_version="fixture-code",
        cohort_sha256=cohort_sha256(primary),
        g3_gate_sha256=hash_canonical(g3_gate),
        input_hashes=inputs,
        seed=config.analysis.seed,
        registered_budgets=CheckpointBudgets(
            bootstrap_count=200,
            permutation_success_count=100,
            permutation_max_attempts=125,
        ),
        completed_bootstrap_indices=list(range(25)),
        completed_permutation_attempt_indices=list(range(25)),
        successful_permutation_indices=list(range(25)),
        bootstrap_results=[
            CheckpointResult(
                index=index,
                status="success",
                result={"draw": index},
                error_code=None,
            )
            for index in range(25)
        ],
        permutation_results=[
            CheckpointResult(
                index=index,
                status="success",
                result={"attempt": index},
                error_code=None,
            )
            for index in range(25)
        ],
        guard_failures=[],
        artifact_hashes=artifact_hashes,
    )
    return config, papers, findings, verification, g3_gate, primary, inputs, checkpoint


def test_frozen_incomplete_finalization_adds_no_draw_and_is_deterministic() -> None:
    config, papers, findings, verification, g3_gate, primary, inputs, checkpoint = (
        _checkpoint_fixture()
    )
    kwargs = {
        "checkpoint": checkpoint,
        "config": config,
        "papers": papers,
        "findings": findings,
        "verification": verification,
        "g3_gate": g3_gate,
        "expected_config_sha256": config_sha256(config),
        "expected_code_version": "fixture-code",
        "expected_cohort_sha256": cohort_sha256(primary),
        "expected_g3_gate_sha256": hash_canonical(g3_gate),
        "expected_input_hashes": inputs,
    }
    first = finalize_incomplete_s5(**kwargs)
    second = finalize_incomplete_s5(**kwargs)
    assert first.json_artifacts == second.json_artifacts
    assert first.table_artifacts == second.table_artifacts
    assert first.completion_mode == "frozen_incomplete"
    assert first.json_artifacts["m4_gate.json"]["status"] == "incomplete"
    assert first.json_artifacts["headline.json"]["selection_reason"] == "m4_incomplete"
    assert first.json_artifacts["bootstrap.json"]["model_stability"]["completed_draws"] == 25
    assert len(checkpoint.completed_bootstrap_indices) == 25


def test_checkpoint_identity_or_deterministic_hash_change_fails_before_output() -> None:
    config, papers, findings, verification, g3_gate, primary, inputs, checkpoint = (
        _checkpoint_fixture()
    )
    with pytest.raises(AnalysisContractError, match="checkpoint_identity_mismatch"):
        finalize_incomplete_s5(
            checkpoint=checkpoint,
            config=config,
            papers=papers,
            findings=findings,
            verification=verification,
            g3_gate=g3_gate,
            expected_config_sha256=config_sha256(config),
            expected_code_version="different-code",
            expected_cohort_sha256=cohort_sha256(primary),
            expected_g3_gate_sha256=hash_canonical(g3_gate),
            expected_input_hashes=inputs,
        )

    forged = checkpoint.model_copy(deep=True)
    forged.artifact_hashes.descriptive_outputs = "d" * 64
    with pytest.raises(AnalysisContractError, match="artifact_hash_mismatch"):
        finalize_incomplete_s5(
            checkpoint=forged,
            config=config,
            papers=papers,
            findings=findings,
            verification=verification,
            g3_gate=g3_gate,
            expected_config_sha256=config_sha256(config),
            expected_code_version="fixture-code",
            expected_cohort_sha256=cohort_sha256(primary),
            expected_g3_gate_sha256=hash_canonical(g3_gate),
            expected_input_hashes=inputs,
        )


def test_bundle_writer_is_deterministic_and_guards_overwrite(tmp_path: Path) -> None:
    papers, findings, verification = _data()
    bundle = analyze_s5(
        config=_config(),
        papers=papers,
        findings=findings,
        verification=verification,
        g3_gate=_g3(story=False),
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = write_analysis_bundle(bundle, first_dir)
    second = write_analysis_bundle(bundle, second_dir)
    assert set(first) == set(second)
    assert {name: sha256_file(path) for name, path in first.items()} == {
        name: sha256_file(path) for name, path in second.items()
    }
    with pytest.raises(AnalysisContractError, match="outputs_exist"):
        write_analysis_bundle(bundle, first_dir)

    bundle.table_artifacts["contradictions.parquet"] = []
    empty_dir = tmp_path / "empty"
    write_analysis_bundle(bundle, empty_dir)
    assert tuple(pd.read_parquet(empty_dir / "contradictions.parquet").columns) == (
        "pair_id",
        "outcome_family",
        "left_direction",
        "right_direction",
        "shared_context_fields",
        "shared_context_count",
        "distance",
        "distance_components",
        "left_citation",
        "right_citation",
    )


def test_parquet_list_cells_normalize_without_scalar_item_failure(tmp_path: Path) -> None:
    path = tmp_path / "list-cells.parquet"
    pd.DataFrame(
        [
            {
                "paper_id": "doc:p01",
                "alternate_doc_ids": ["pmid:1", "doi:10.1/example"],
                "query_families": ["direct", "null-negative"],
            }
        ]
    ).to_parquet(path, index=False)
    records = read_parquet_records(path)
    assert records == [
        {
            "paper_id": "doc:p01",
            "alternate_doc_ids": ["pmid:1", "doi:10.1/example"],
            "query_families": ["direct", "null-negative"],
        }
    ]


def test_fixture_interruption_requires_exact_marker_and_stops_at_25(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "fixture_fault_injection.json"
    marker.write_text(
        '{"fixture_fault_injection_version":"1","question_id":'
        '"fixture-b-incomplete","mode":"after-25-bootstrap"}',
        encoding="utf-8",
    )
    args = Namespace(
        fixture_fault_injection=FIXTURE_INTERRUPTION_MODE,
        fixture=True,
        question="fixture-b-incomplete",
        finalize_incomplete_from=None,
        with_remap=None,
    )
    assert _fixture_fault_mode(args, tmp_path) == FIXTURE_INTERRUPTION_MODE

    _, _, _, _, _, _, _, checkpoint = _checkpoint_fixture()
    checkpoint = checkpoint.model_copy(
        update={
            "completed_bootstrap_indices": list(range(25)),
            "bootstrap_results": checkpoint.bootstrap_results,
        }
    )
    archived: list[M4SourceCheckpoint] = []
    monkeypatch.setattr("scripts.s5_analyze._checkpoint_writer", archived.append)
    writer = _checkpoint_writer_with_fixture_interrupt(FIXTURE_INTERRUPTION_MODE)
    with pytest.raises(FixtureInjectedInterruption) as raised:
        writer(checkpoint)
    assert raised.value.checkpoint == checkpoint
    assert archived == [checkpoint]

    args.fixture = False
    with pytest.raises(AnalysisContractError, match="requires_explicit_fixture"):
        _fixture_fault_mode(args, tmp_path)
