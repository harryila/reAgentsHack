"""Reproducible, text-free summaries of archived GEPA pilot runs.

The source run bundle is intentionally local because its GEPA trace, provider
receipts, and held-out report contain benchmark text and model outputs.  This module
reads those restricted artifacts only to validate lineage and aggregate metadata; it
never copies prompts, outputs, evidence, PICO fields, example identifiers, or
per-example labels into the paper artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from literature_multiverse.lineage import atomic_write_json, hash_canonical, sha256_file
from literature_multiverse.prompt_optimization import (
    PROMPT_COMPONENTS,
    OptimizationSplitManifest,
)


class GEPAPilotSummaryError(ValueError):
    """Raised when an archived pilot cannot support a trustworthy summary."""


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GEPAPilotSummaryError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise GEPAPilotSummaryError(f"{label} must be a JSON object")
    return value


def _repository_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(
            repository_root.resolve(strict=True)
        ).as_posix()
    except (OSError, ValueError) as exc:
        raise GEPAPilotSummaryError("summary input must be inside the repository") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GEPAPilotSummaryError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise GEPAPilotSummaryError(f"{label} is outside its valid range")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GEPAPilotSummaryError(f"{label} must be an integer >= {minimum}")
    return value


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GEPAPilotSummaryError(f"{label} must be numeric")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise GEPAPilotSummaryError(f"{label} must be numeric") from exc
    if not result.is_finite() or result < 0:
        raise GEPAPilotSummaryError(f"{label} must be finite and nonnegative")
    return result


def _string_counter(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _provider_receipt_summary(directory: Path) -> dict[str, Any]:
    receipt_paths = sorted(directory.glob("*.provider.json"))
    if not receipt_paths:
        raise GEPAPilotSummaryError("provider receipt directory is empty")

    statuses: list[str] = []
    failures: list[str] = []
    cost_bases: list[str] = []
    providers: set[str] = set()
    models: set[str] = set()
    input_tokens = 0
    output_tokens = 0
    estimated_cost = Decimal("0")
    receipt_hashes: list[str] = []

    for path in receipt_paths:
        receipt = _read_json_object(path, "provider receipt")
        status = receipt.get("status")
        if status not in {"complete", "failed"}:
            raise GEPAPilotSummaryError("provider receipt has an invalid status")
        statuses.append(status)
        if status == "failed":
            failure = receipt.get("failure")
            if not isinstance(failure, str) or not failure:
                raise GEPAPilotSummaryError("failed provider receipt has no failure type")
            failures.append(failure)

        cost_basis = receipt.get("cost_basis")
        if not isinstance(cost_basis, str) or not cost_basis:
            raise GEPAPilotSummaryError("provider receipt has no cost basis")
        cost_bases.append(cost_basis)
        estimated_cost += _decimal(
            receipt.get("estimated_cost_usd"), "receipt estimated cost"
        )

        provider = receipt.get("provider")
        model = receipt.get("model")
        if not isinstance(provider, str) or not provider:
            raise GEPAPilotSummaryError("provider receipt has no provider")
        if not isinstance(model, str) or not model:
            raise GEPAPilotSummaryError("provider receipt has no model")
        providers.add(provider)
        models.add(model)

        usage = receipt.get("usage")
        if not isinstance(usage, Mapping):
            raise GEPAPilotSummaryError("provider receipt has no usage object")
        input_tokens += _integer(usage.get("input_tokens"), "input tokens")
        output_tokens += _integer(usage.get("output_tokens"), "output tokens")
        receipt_hashes.append(sha256_file(path))

    return {
        "attempts": len(receipt_paths),
        "status_counts": _string_counter(statuses),
        "failure_type_counts": _string_counter(failures),
        "estimated_cost_usd": float(estimated_cost),
        "reported_input_tokens": input_tokens,
        "reported_output_tokens": output_tokens,
        "cost_basis_counts": _string_counter(cost_bases),
        "providers": sorted(providers),
        "models": sorted(models),
        "receipt_set_sha256": hash_canonical(receipt_hashes),
    }


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GEPAPilotSummaryError(f"{label} mismatch")


def _historical_or_recorded_reflection_usage(trace: Mapping[str, Any]) -> dict[str, Any]:
    usage = trace.get("reflection_lm_usage")
    if usage is None:
        return {
            "status": "unavailable_historical_trace",
            "total_cost_usd": None,
            "total_input_tokens": None,
            "total_output_tokens": None,
        }
    if not isinstance(usage, Mapping):
        raise GEPAPilotSummaryError("reflection LM usage must be an object")
    status = usage.get("status")
    if status != "available":
        return {
            "status": "unavailable",
            "total_cost_usd": None,
            "total_input_tokens": None,
            "total_output_tokens": None,
        }
    return {
        "status": "available",
        "total_cost_usd": _number(
            usage.get("total_cost_usd"), "reflection cost", minimum=0
        ),
        "total_input_tokens": _integer(
            usage.get("total_input_tokens"), "reflection input tokens"
        ),
        "total_output_tokens": _integer(
            usage.get("total_output_tokens"), "reflection output tokens"
        ),
    }


def _successful_run_summary(
    *,
    run_dir: Path,
    manifest_path: Path,
    seed_prompt_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    trace_path = run_dir / "optimization_trace.json"
    winner_path = run_dir / "frozen_winner.json"
    report_path = run_dir / "heldout-test.json"
    trace = _read_json_object(trace_path, "optimization trace")
    winner = _read_json_object(winner_path, "frozen winner")
    report = _read_json_object(report_path, "held-out report")
    try:
        manifest = OptimizationSplitManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise GEPAPilotSummaryError("cannot read split manifest") from exc

    _assert_equal(trace.get("optimization_trace_version"), "1", "trace version")
    _assert_equal(winner.get("frozen_prompt_bundle_version"), "1", "winner version")
    _assert_equal(report.get("heldout_evaluation_version"), "1", "report version")
    _assert_equal(trace.get("optimization_splits"), ["train", "dev"], "optimization splits")
    _assert_equal(trace.get("test_split_opened"), False, "trace test-opened flag")
    _assert_equal(trace.get("test_evaluated"), False, "trace test-evaluated flag")
    _assert_equal(
        winner.get("test_evaluated_at_freeze"), False, "winner test-evaluated flag"
    )
    _assert_equal(report.get("split"), "test", "held-out split")

    manifest_sha = sha256_file(manifest_path)
    trace_sha = sha256_file(trace_path)
    winner_sha = sha256_file(winner_path)
    report_sha = sha256_file(report_path)
    _assert_equal(trace.get("manifest_sha256"), manifest_sha, "trace manifest hash")
    _assert_equal(winner.get("manifest_sha256"), manifest_sha, "winner manifest hash")
    _assert_equal(report.get("manifest_sha256"), manifest_sha, "report manifest hash")
    _assert_equal(
        winner.get("optimization_trace_sha256"), trace_sha, "winner trace hash"
    )
    _assert_equal(report.get("winner_sha256"), winner_sha, "report winner hash")
    _assert_equal(trace.get("run_id"), winner.get("run_id"), "run ID")
    _assert_equal(trace.get("manifest_seed"), manifest.seed, "manifest seed")
    _assert_equal(
        trace.get("train_example_ids"), manifest.train.example_ids, "train IDs"
    )
    _assert_equal(trace.get("dev_example_ids"), manifest.dev.example_ids, "dev IDs")
    _assert_equal(report.get("test_example_ids"), manifest.test.example_ids, "test IDs")

    component_traces = trace.get("component_traces")
    seed_hashes = trace.get("seed_prompt_sha256s")
    winning_hashes = trace.get("winning_prompt_sha256s")
    winner_seed_hashes = winner.get("seed_prompt_sha256s")
    winner_prompts = winner.get("prompts")
    if not all(
        isinstance(value, Mapping)
        for value in (
            component_traces,
            seed_hashes,
            winning_hashes,
            winner_seed_hashes,
            winner_prompts,
        )
    ):
        raise GEPAPilotSummaryError("prompt lineage metadata is malformed")
    if set(component_traces) != {"extraction"} or set(winner_prompts) != {"extraction"}:
        raise GEPAPilotSummaryError("pilot summary requires one extraction component")

    seed_text = seed_prompt_path.read_text(encoding="utf-8")
    seed_prompt_sha = _sha256_text(seed_text)
    _assert_equal(seed_hashes.get("extraction"), seed_prompt_sha, "trace seed prompt hash")
    _assert_equal(
        winner_seed_hashes.get("extraction"), seed_prompt_sha, "winner seed prompt hash"
    )

    prompt_metadata = winner_prompts["extraction"]
    if not isinstance(prompt_metadata, Mapping):
        raise GEPAPilotSummaryError("winner prompt metadata is malformed")
    prompt_relative = prompt_metadata.get("path")
    if not isinstance(prompt_relative, str):
        raise GEPAPilotSummaryError("winner prompt path is missing")
    prompt_candidate_path = Path(prompt_relative)
    if (
        prompt_candidate_path.is_absolute()
        or ".." in prompt_candidate_path.parts
        or prompt_candidate_path.as_posix() != prompt_relative
    ):
        raise GEPAPilotSummaryError("winner prompt path is unsafe")
    frozen_prompt_path = run_dir / prompt_candidate_path
    frozen_prompt_sha = sha256_file(frozen_prompt_path)
    _assert_equal(
        prompt_metadata.get("sha256"), frozen_prompt_sha, "winner prompt file hash"
    )
    _assert_equal(
        winning_hashes.get("extraction"), frozen_prompt_sha, "trace winning prompt hash"
    )

    component_trace = component_traces["extraction"]
    if not isinstance(component_trace, Mapping):
        raise GEPAPilotSummaryError("extraction component trace is malformed")
    component = PROMPT_COMPONENTS["extraction"]
    _assert_equal(component_trace.get("component"), component, "component name")
    candidates = component_trace.get("candidates")
    candidate_hashes = component_trace.get("candidate_sha256s")
    scores = component_trace.get("val_aggregate_scores")
    if not (
        isinstance(candidates, list)
        and isinstance(candidate_hashes, list)
        and isinstance(scores, list)
        and len(candidates) == len(candidate_hashes) == len(scores)
        and candidates
    ):
        raise GEPAPilotSummaryError("candidate trace is malformed")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get(component), str):
            raise GEPAPilotSummaryError("candidate prompt is malformed")
        _assert_equal(
            candidate_hashes[index], hash_canonical(dict(candidate)), "candidate hash"
        )
    _assert_equal(
        _sha256_text(candidates[0][component]), seed_prompt_sha, "seed candidate prompt"
    )
    best_index = _integer(component_trace.get("best_idx"), "best candidate index")
    if best_index >= len(candidates):
        raise GEPAPilotSummaryError("best candidate index is out of range")
    candidate_scores = [
        _number(value, f"candidate {index} score", minimum=0)
        for index, value in enumerate(scores)
    ]
    _assert_equal(
        component_trace.get("best_candidate_sha256"),
        candidate_hashes[best_index],
        "best candidate hash",
    )
    if not math.isclose(
        _number(component_trace.get("best_score"), "best score", minimum=0),
        candidate_scores[best_index],
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise GEPAPilotSummaryError("best score mismatch")
    _assert_equal(
        _sha256_text(candidates[best_index][component]),
        frozen_prompt_sha,
        "best candidate prompt",
    )

    total_metric_calls = _integer(
        component_trace.get("total_metric_calls"), "total metric calls"
    )
    metric_call_cap = _integer(
        trace.get("max_metric_calls_per_prompt"), "metric call cap", minimum=1
    )
    optimization_receipts = _provider_receipt_summary(run_dir / "provider_attempts")
    _assert_equal(
        optimization_receipts["attempts"], total_metric_calls, "optimization receipt count"
    )

    results = report.get("results")
    if (
        not isinstance(results, Mapping)
        or set(results) != {"extraction"}
        or not isinstance(results["extraction"], Mapping)
    ):
        raise GEPAPilotSummaryError("held-out report must contain one extraction result")
    result = results["extraction"]
    heldout_rows = _integer(result.get("rows"), "held-out row count", minimum=1)
    outputs = result.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != heldout_rows:
        raise GEPAPilotSummaryError("held-out output count mismatch")
    _assert_equal(heldout_rows, manifest.test.rows, "held-out manifest row count")
    objective_scores = result.get("mean_objective_scores")
    objective_weights = trace.get("objective_weights")
    if not isinstance(objective_scores, Mapping) or not isinstance(
        objective_weights, Mapping
    ):
        raise GEPAPilotSummaryError("objective metadata is malformed")
    _assert_equal(set(objective_scores), set(objective_weights), "objective names")
    normalized_objectives = {
        str(name): _number(value, f"{name} held-out score", minimum=0)
        for name, value in sorted(objective_scores.items())
    }
    normalized_weights = {
        str(name): _number(value, f"{name} objective weight", minimum=0)
        for name, value in sorted(objective_weights.items())
    }
    scalar_score = _number(result.get("mean_scalar_score"), "held-out scalar score")
    recomputed_scalar = sum(
        normalized_objectives[name] * normalized_weights[name]
        for name in normalized_objectives
    )
    if not math.isclose(scalar_score, recomputed_scalar, rel_tol=0, abs_tol=1e-12):
        raise GEPAPilotSummaryError("held-out scalar score does not match its weights")
    test_receipts = _provider_receipt_summary(run_dir / "test_provider_attempts")
    _assert_equal(test_receipts["attempts"], heldout_rows, "test receipt count")

    retained_seed = seed_prompt_sha == frozen_prompt_sha
    return {
        "run_id": trace["run_id"],
        "status": "valid_frozen_heldout_evaluation",
        "optimizer": {
            "implementation": trace.get("optimizer"),
            "gepa_version": trace.get("gepa_version"),
            "seed": trace.get("optimizer_seed"),
        },
        "split_rows": {
            "train": manifest.train.rows,
            "dev": manifest.dev.rows,
            "test": manifest.test.rows,
        },
        "artifacts": {
            "optimization_trace": {
                "path": _repository_path(trace_path, repository_root),
                "sha256": trace_sha,
            },
            "frozen_winner": {
                "path": _repository_path(winner_path, repository_root),
                "sha256": winner_sha,
            },
            "heldout_test_report": {
                "path": _repository_path(report_path, repository_root),
                "sha256": report_sha,
            },
            "split_manifest": {
                "path": _repository_path(manifest_path, repository_root),
                "sha256": manifest_sha,
            },
            "seed_prompt": {
                "path": _repository_path(seed_prompt_path, repository_root),
                "sha256": seed_prompt_sha,
            },
            "frozen_prompt": {
                "path": _repository_path(frozen_prompt_path, repository_root),
                "sha256": frozen_prompt_sha,
            },
        },
        "optimization": {
            "objective_weights": normalized_weights,
            "metric_call_cap": metric_call_cap,
            "observed_metric_calls": total_metric_calls,
            "batch_boundary_overshoot_calls": max(0, total_metric_calls - metric_call_cap),
            "full_dev_candidates_evaluated": len(candidates),
            "full_dev_mutations_evaluated": len(candidates) - 1,
            "candidate_dev_scalar_scores": [
                {
                    "candidate_role": "seed" if index == 0 else f"mutation_{index}",
                    "score": score,
                }
                for index, score in enumerate(candidate_scores)
            ],
            "selected_candidate_index": best_index,
            "winner_retained_seed": retained_seed,
            "winner_byte_identical_to_seed": retained_seed,
            "task_provider_receipts": optimization_receipts,
            "reflection_lm_usage": _historical_or_recorded_reflection_usage(trace),
        },
        "heldout_test": {
            "n": heldout_rows,
            "mean_scalar_score": scalar_score,
            "mean_objective_scores": normalized_objectives,
            "task_provider_receipts": test_receipts,
        },
    }


def _failed_raw_schema_summary(
    *,
    failed_summary_path: Path,
    failed_run_dir: Path,
    repository_root: Path,
) -> dict[str, Any]:
    failed = _read_json_object(failed_summary_path, "failed-run summary")
    _assert_equal(failed.get("failed_run_summary_version"), "1", "failed-run version")
    _assert_equal(failed.get("status"), "invalid_run_no_winner", "failed-run status")
    _assert_equal(failed.get("test_split_opened"), False, "failed-run test-opened flag")
    _assert_equal(failed.get("test_evaluated"), False, "failed-run test flag")
    if (failed_run_dir / "frozen_winner.json").exists():
        raise GEPAPilotSummaryError("failed raw-schema run unexpectedly has a winner")
    _assert_equal(
        failed.get("local_restricted_trace"),
        _repository_path(failed_run_dir, repository_root),
        "failed-run local path",
    )

    observed = _provider_receipt_summary(failed_run_dir / "provider_attempts")
    declared = failed.get("provider_attempts")
    if not isinstance(declared, Mapping):
        raise GEPAPilotSummaryError("failed-run provider summary is malformed")
    _assert_equal(observed["attempts"], declared.get("count"), "failed attempt count")
    _assert_equal(
        observed["status_counts"].get("complete", 0),
        declared.get("completed"),
        "failed-run completed count",
    )
    _assert_equal(
        observed["status_counts"].get("failed", 0),
        declared.get("failed"),
        "failed-run failure count",
    )
    _assert_equal(
        observed["failure_type_counts"],
        declared.get("failure_types"),
        "failed-run failure types",
    )
    _assert_equal(
        observed["reported_input_tokens"],
        declared.get("reported_input_tokens"),
        "failed-run input tokens",
    )
    _assert_equal(
        observed["reported_output_tokens"],
        declared.get("reported_output_tokens"),
        "failed-run output tokens",
    )
    declared_ceiling = _number(
        declared.get("archived_conservative_failure_ceiling_usd"),
        "failed-run conservative ceiling",
        minimum=0,
    )
    if not math.isclose(
        observed["estimated_cost_usd"], declared_ceiling, rel_tol=0, abs_tol=1e-12
    ):
        raise GEPAPilotSummaryError("failed-run conservative cost ceiling mismatch")

    optimizer = failed.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise GEPAPilotSummaryError("failed-run optimizer metadata is malformed")
    failed_provider_attempts = dict(observed)
    archived_ceiling = failed_provider_attempts.pop("estimated_cost_usd")
    failed_provider_attempts.update(
        {
            "archived_conservative_failure_ceiling_usd": archived_ceiling,
            "actual_api_charge_asserted": False,
        }
    )
    return {
        "status": "excluded_invalid_provider_contract_run",
        "included_in_successful_pilot_metrics": False,
        "scientific_result_available": False,
        "exclusion_reason": "raw_schema_rejected_before_generation",
        "started_at_date": failed.get("started_at_date"),
        "manifest_sha256": failed.get("manifest_sha256"),
        "seed_prompt_sha256": failed.get("seed_prompt_sha256"),
        "optimizer": {
            "implementation": optimizer.get("implementation"),
            "seed": optimizer.get("seed"),
            "max_metric_calls": optimizer.get("max_metric_calls"),
        },
        "test_split_opened": False,
        "test_evaluated": False,
        "provider_attempts": failed_provider_attempts,
        "artifacts": {
            "metadata_summary": {
                "path": _repository_path(failed_summary_path, repository_root),
                "sha256": sha256_file(failed_summary_path),
            },
            "restricted_run_directory": _repository_path(
                failed_run_dir, repository_root
            ),
        },
    }


def write_gepa_pilot_metadata_summary(
    *,
    run_dir: str | Path,
    manifest_path: str | Path,
    seed_prompt_path: str | Path,
    failed_summary_path: str | Path,
    failed_run_dir: str | Path,
    output_path: str | Path,
    repository_root: str | Path,
    force: bool = False,
) -> Path:
    """Validate archived GEPA artifacts and write a deterministic paper summary."""

    root = Path(repository_root)
    successful = _successful_run_summary(
        run_dir=Path(run_dir),
        manifest_path=Path(manifest_path),
        seed_prompt_path=Path(seed_prompt_path),
        repository_root=root,
    )
    failed = _failed_raw_schema_summary(
        failed_summary_path=Path(failed_summary_path),
        failed_run_dir=Path(failed_run_dir),
        repository_root=root,
    )
    summary = {
        "gepa_pilot_summary_version": "1",
        "status": "superseded_non_citable_pending_receipt_bounded_reconstruction",
        "scientific_claims_allowed": False,
        "superseded_by": (
            "artifacts/diagnostics/evidence-inference/ollama-gepa-study-v1.json"
        ),
        "benchmark": "Evidence Inference 2.0",
        "contains_pico_text": False,
        "contains_source_text": False,
        "contains_evidence_text": False,
        "contains_model_outputs": False,
        "contains_per_example_labels": False,
        "contains_per_example_scores": False,
        "successful_pilot": successful,
        "failed_raw_schema_run": failed,
    }
    target = Path(output_path)
    atomic_write_json(target, summary, force=force)
    return target


__all__ = [
    "GEPAPilotSummaryError",
    "write_gepa_pilot_metadata_summary",
]
