from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from literature_multiverse.gepa_pilot_summary import (
    GEPAPilotSummaryError,
    write_gepa_pilot_metadata_summary,
)
from literature_multiverse.lineage import (
    atomic_write_json,
    atomic_write_text,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.prompt_optimization import (
    OptimizationSplitManifest,
    SplitArtifact,
)


def _split(name: str) -> SplitArtifact:
    digest_character = {"train": "1", "dev": "2", "test": "3"}[name]
    return SplitArtifact(
        path=f"{name}.jsonl",
        sha256=(digest_character * 64),
        rows=1,
        example_ids=[f"example-{name}"],
        paper_ids=[f"paper-{name}"],
        group_ids=[f"group-{name}"],
    )


def _receipt(
    path: Path,
    *,
    status: str,
    cost: float,
    failure: str | None = None,
    input_tokens: int = 10,
    output_tokens: int = 2,
) -> None:
    atomic_write_json(
        path,
        {
            "status": status,
            "failure": failure,
            "cost_basis": "reported_usage",
            "estimated_cost_usd": cost,
            "provider": "fixture-provider",
            "model": "fixture-model",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            "prompt": "PICO SOURCE EVIDENCE SECRET",
            "parsed_json": {"per_example_label": "MODEL OUTPUT SECRET"},
        },
    )


def _pilot_fixture(root: Path) -> dict[str, Path]:
    run_dir = root / "successful"
    run_dir.mkdir()
    manifest_path = root / "manifest.json"
    manifest = OptimizationSplitManifest(
        algorithm="official-paper-groups-v1",
        seed=0,
        train_fraction=0.6,
        dev_fraction=0.2,
        source_examples_sha256="f" * 64,
        train=_split("train"),
        dev=_split("dev"),
        test=_split("test"),
    )
    atomic_write_json(manifest_path, manifest)

    seed_prompt_path = root / "seed.md"
    seed_prompt = "Extract the outcome as strict JSON."
    mutation_prompt = "MODEL OUTPUT SECRET mutation prompt"
    atomic_write_text(seed_prompt_path, seed_prompt)
    atomic_write_text(run_dir / "frozen_extraction.md", seed_prompt)

    candidates = [
        {"extraction_prompt": seed_prompt},
        {"extraction_prompt": mutation_prompt},
    ]
    candidate_hashes = [hash_canonical(candidate) for candidate in candidates]
    manifest_sha = sha256_file(manifest_path)
    prompt_sha = sha256_file(seed_prompt_path)
    trace = {
        "optimization_trace_version": "1",
        "run_id": "gepa-fixture",
        "optimizer": "gepa.optimize",
        "gepa_version": "0.1.4",
        "manifest_path": "manifest.json",
        "manifest_sha256": manifest_sha,
        "manifest_seed": 0,
        "optimizer_seed": 17,
        "optimization_splits": ["train", "dev"],
        "test_split_opened": False,
        "test_evaluated": False,
        "objective_weights": {
            "extraction_correctness": 0.7,
            "grounding_schema_validity": 0.25,
            "cost_efficiency": 0.05,
        },
        "max_metric_calls_per_prompt": 1,
        "train_example_ids": manifest.train.example_ids,
        "dev_example_ids": manifest.dev.example_ids,
        "seed_prompt_sha256s": {"extraction": prompt_sha},
        "winning_prompt_sha256s": {"extraction": prompt_sha},
        "component_traces": {
            "extraction": {
                "component": "extraction_prompt",
                "best_idx": 0,
                "best_score": 0.8,
                "best_candidate_sha256": candidate_hashes[0],
                "candidates": candidates,
                "candidate_sha256s": candidate_hashes,
                "parents": [[None], [0]],
                "val_aggregate_scores": [0.8, 0.5],
                "total_metric_calls": 2,
            }
        },
    }
    atomic_write_json(run_dir / "optimization_trace.json", trace)
    winner = {
        "frozen_prompt_bundle_version": "1",
        "run_id": "gepa-fixture",
        "manifest_sha256": manifest_sha,
        "optimization_trace_sha256": sha256_file(
            run_dir / "optimization_trace.json"
        ),
        "seed_prompt_sha256s": {"extraction": prompt_sha},
        "test_evaluated_at_freeze": False,
        "prompts": {
            "extraction": {
                "path": "frozen_extraction.md",
                "sha256": prompt_sha,
            }
        },
    }
    atomic_write_json(run_dir / "frozen_winner.json", winner)
    atomic_write_json(
        run_dir / "heldout-test.json",
        {
            "heldout_evaluation_version": "1",
            "manifest_sha256": manifest_sha,
            "split": "test",
            "test_example_ids": manifest.test.example_ids,
            "winner_sha256": sha256_file(run_dir / "frozen_winner.json"),
            "results": {
                "extraction": {
                    "rows": 1,
                    "mean_scalar_score": 0.8,
                    "mean_objective_scores": {
                        "extraction_correctness": 0.8,
                        "grounding_schema_validity": 0.8,
                        "cost_efficiency": 0.8,
                    },
                    "outputs": [
                        {
                            "example_id": "example-test",
                            "label": "per-example-label-secret",
                            "output": "MODEL OUTPUT SECRET",
                        }
                    ],
                }
            },
        },
    )
    optimization_receipts = run_dir / "provider_attempts"
    optimization_receipts.mkdir()
    _receipt(optimization_receipts / "one.provider.json", status="complete", cost=0.1)
    _receipt(optimization_receipts / "two.provider.json", status="complete", cost=0.2)
    test_receipts = run_dir / "test_provider_attempts"
    test_receipts.mkdir()
    _receipt(
        test_receipts / "test.provider.json",
        status="failed",
        failure="PROVIDER_INVALID_STRUCTURED_JSON",
        cost=0.3,
    )

    failed_run_dir = root / "failed"
    failed_receipts = failed_run_dir / "provider_attempts"
    failed_receipts.mkdir(parents=True)
    _receipt(
        failed_receipts / "raw.provider.json",
        status="failed",
        failure="BadRequestError",
        cost=0.4,
        input_tokens=0,
        output_tokens=0,
    )
    failed_summary_path = root / "failed-summary.json"
    atomic_write_json(
        failed_summary_path,
        {
            "failed_run_summary_version": "1",
            "status": "invalid_run_no_winner",
            "started_at_date": "2026-08-26",
            "manifest_sha256": "a" * 64,
            "seed_prompt_sha256": prompt_sha,
            "optimizer": {
                "implementation": "gepa.optimize",
                "seed": 17,
                "max_metric_calls": 10,
            },
            "test_split_opened": False,
            "test_evaluated": False,
            "provider_attempts": {
                "count": 1,
                "completed": 0,
                "failed": 1,
                "failure_types": {"BadRequestError": 1},
                "reported_input_tokens": 0,
                "reported_output_tokens": 0,
                "archived_conservative_failure_ceiling_usd": 0.4,
            },
            "local_restricted_trace": "failed",
        },
    )
    return {
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "seed_prompt_path": seed_prompt_path,
        "failed_summary_path": failed_summary_path,
        "failed_run_dir": failed_run_dir,
    }


def _write_summary(root: Path, inputs: dict[str, Path], name: str) -> Path:
    return write_gepa_pilot_metadata_summary(
        **inputs,
        output_path=root / name,
        repository_root=root,
    )


def test_gepa_pilot_summary_is_reproducible_and_contains_only_aggregates(
    tmp_path: Path,
) -> None:
    inputs = _pilot_fixture(tmp_path)
    first = _write_summary(tmp_path, inputs, "summary-one.json")
    second = _write_summary(tmp_path, inputs, "summary-two.json")

    assert first.read_bytes() == second.read_bytes()
    summary: dict[str, Any] = json.loads(first.read_text(encoding="utf-8"))
    successful = summary["successful_pilot"]
    assert successful["optimization"]["observed_metric_calls"] == 2
    assert successful["optimization"]["batch_boundary_overshoot_calls"] == 1
    assert successful["optimization"]["winner_retained_seed"] is True
    assert successful["optimization"]["reflection_lm_usage"] == {
        "status": "unavailable_historical_trace",
        "total_cost_usd": None,
        "total_input_tokens": None,
        "total_output_tokens": None,
    }
    assert successful["heldout_test"]["mean_scalar_score"] == 0.8
    assert successful["heldout_test"]["task_provider_receipts"]["status_counts"] == {
        "failed": 1
    }
    assert summary["failed_raw_schema_run"]["included_in_successful_pilot_metrics"] is False

    serialized = first.read_text(encoding="utf-8")
    for forbidden in (
        "PICO SOURCE EVIDENCE SECRET",
        "MODEL OUTPUT SECRET",
        "per-example-label-secret",
        "example-test",
    ):
        assert forbidden not in serialized
    for flag in (
        "contains_pico_text",
        "contains_source_text",
        "contains_evidence_text",
        "contains_model_outputs",
        "contains_per_example_labels",
        "contains_per_example_scores",
    ):
        assert summary[flag] is False


def test_gepa_pilot_summary_rejects_tampered_frozen_prompt(tmp_path: Path) -> None:
    inputs = _pilot_fixture(tmp_path)
    (inputs["run_dir"] / "frozen_extraction.md").write_text(
        "tampered", encoding="utf-8"
    )

    with pytest.raises(GEPAPilotSummaryError, match="winner prompt file hash mismatch"):
        _write_summary(tmp_path, inputs, "summary.json")
