"""Deterministic s5 scientific orchestration.

The module composes the pure science helpers into the exact analysis inventory.  It
keeps filesystem concerns at the edge so ordinary, fixture, and frozen-incomplete runs
exercise the same branch logic.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from literature_multiverse.cohort import cohort_sha256
from literature_multiverse.config import ModeratorSpec, QuestionConfig, config_sha256
from literature_multiverse.contradictions import find_contradictions, residual_summary
from literature_multiverse.disagreement import (
    CANONICAL_LABELS,
    bootstrap_disagreement,
    paper_balanced_finding_summary,
    paper_modal_summary,
)
from literature_multiverse.evidence_gaps import build_evidence_gap_grid
from literature_multiverse.grounding import verification_allows_primary
from literature_multiverse.lineage import (
    atomic_write_json,
    canonical_json_bytes,
    hash_canonical,
)
from literature_multiverse.models import (
    CheckpointArtifactHashes,
    CheckpointBudgets,
    CheckpointResult,
    M4CheckpointFrozenIncomplete,
    M4CheckpointNotApplicable,
    M4SourceCheckpoint,
    VerificationRecord,
    canonical_model_sha256,
)
from literature_multiverse.moderators import evaluate_all_moderators, evaluate_moderator
from literature_multiverse.resampling import (
    build_variant_a_headline,
    build_variant_b_headline,
    comparison_statistics,
    evaluate_m4_gate,
    paper_bootstrap_sample,
    paper_permutation,
    summarize_westfall_young,
)
from literature_multiverse.tree import build_tree_artifact, root_split_bootstrap_frequency

JSON_ARTIFACT_NAMES = (
    "m4_checkpoint.json",
    "tree.json",
    "bootstrap.json",
    "permutation.json",
    "m4_gate.json",
    "headline.json",
)
TABLE_ARTIFACT_NAMES = (
    "moderators.parquet",
    "contradictions.parquet",
    "evidence_gaps.parquet",
)
TABLE_SCHEMAS = {
    "contradictions.parquet": (
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
    ),
    "evidence_gaps.parquet": (
        "cell_id",
        "primary_endpoint",
        "axis_values",
        "n_papers_total",
        "n_papers_grounded",
        "n_findings",
        "grounded_fraction",
        "classifiable_fraction",
        "paper_entropy",
        "status",
    ),
}


class AnalysisContractError(ValueError):
    """An s5 scientific input or output violated the locked contract."""


class G3TrustBlockedError(AnalysisContractError):
    """G3 trust failed, so no scientific release may be produced."""


@dataclass(frozen=True, slots=True)
class InferenceOverrides:
    """Offline/test injection point for an already-computed registered battery."""

    moderator_results: Sequence[Mapping[str, Any]] | None = None
    permutation: Mapping[str, Any] | None = None
    bootstrap_stability: Mapping[str, Mapping[str, Any]] | None = None


@dataclass(slots=True)
class AnalysisBundle:
    """Complete in-memory s5 inventory before atomic persistence."""

    json_artifacts: dict[str, dict[str, Any]]
    table_artifacts: dict[str, list[dict[str, Any]]]
    primary_rows: list[dict[str, Any]]
    trace: dict[str, Any] | None = None
    completion_mode: Literal["normal", "frozen_incomplete"] = "normal"
    checkpoint_sha256: str | None = None
    source_started_at: datetime | None = None
    source_completed_at: datetime | None = None


@dataclass(slots=True)
class CheckpointContext:
    """Identity required to write resumable ordinary-run checkpoints."""

    source_run_id: str
    source_started_at: datetime
    question_id: str
    config_sha256: str
    code_version: str
    cohort_sha256: str
    g3_gate_sha256: str
    input_hashes: dict[str, str]
    seed: int
    bootstrap_count: int
    permutation_success_count: int
    permutation_max_attempts: int
    checkpointed_at: Callable[[], datetime]
    writer: Callable[[M4SourceCheckpoint], None]


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _normalize_value(value: Any) -> Any:
    """Convert pandas/Arrow containers and scalars without calling ``item`` on arrays."""

    value = _enum_value(value)
    if value is None or value is pd.NA:
        return None
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, set):
        return [_normalize_value(item) for item in sorted(value, key=repr)]
    if hasattr(value, "tolist") and callable(value.tolist):
        converted = value.tolist()
        if converted is not value:
            return _normalize_value(converted)
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item") and callable(value.item):
        return _normalize_value(value.item())
    return value


def _normalize_record(row: Mapping[str, Any]) -> dict[str, Any]:
    output = {str(key): _normalize_value(value) for key, value in row.items()}
    moderators = output.get("moderators")
    if isinstance(moderators, str):
        try:
            decoded = json.loads(moderators)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            output["moderators"] = decoded
    return output


def read_parquet_records(path: Path) -> list[dict[str, Any]]:
    """Read parquet into JSON-compatible records with nulls restored."""

    if not path.is_file():
        raise AnalysisContractError(f"missing_parquet:{path}")
    return [_normalize_record(row) for row in pd.read_parquet(path).to_dict(orient="records")]


def _verification_decisions(
    verification: VerificationRecord | Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    if isinstance(verification, VerificationRecord):
        record = verification
    else:
        record = VerificationRecord.model_validate(verification)
    return {
        decision.finding_id: {
            "finding_id": decision.finding_id,
            "model_status": decision.model_status,
            "adjudication": decision.adjudication,
        }
        for decision in record.decisions
    }


def derive_primary_cohort(
    papers: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    verification: VerificationRecord | Mapping[str, Any],
    *,
    primary_family: str,
) -> list[dict[str, Any]]:
    """Apply every primary-cohort paper, finding, grounding, and verifier filter."""

    paper_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in papers:
        paper = _normalize_record(raw)
        paper_id = paper.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            raise AnalysisContractError("paper_ledger_invalid_id")
        if paper_id in paper_by_id:
            raise AnalysisContractError(f"paper_ledger_duplicate_id:{paper_id}")
        paper_by_id[paper_id] = paper
    decisions = _verification_decisions(verification)
    primary: list[dict[str, Any]] = []
    finding_ids: set[str] = set()
    for raw in findings:
        row = _normalize_record(raw)
        finding_id = row.get("finding_id")
        paper_id = row.get("paper_id")
        if not isinstance(finding_id, str) or not finding_id:
            raise AnalysisContractError("finding_ledger_invalid_id")
        if finding_id in finding_ids:
            raise AnalysisContractError(f"finding_ledger_duplicate_id:{finding_id}")
        finding_ids.add(finding_id)
        if paper_id not in paper_by_id:
            raise AnalysisContractError(f"finding_ledger_orphan:{finding_id}")
        paper = paper_by_id[paper_id]
        paper_eligible = (
            _enum_value(paper.get("screen_status")) == "included"
            and _enum_value(paper.get("map_status")) == "success"
            and paper.get("eligible") is True
        )
        if not paper_eligible:
            continue
        if row.get("outcome_family") != primary_family:
            continue
        if _enum_value(row.get("effect_direction")) not in CANONICAL_LABELS:
            continue
        if _enum_value(row.get("grounding_status")) != "exact" or bool(row.get("section_flagged")):
            continue
        decision = decisions.get(finding_id)
        if decision is None:
            raise AnalysisContractError(f"primary_verification_decision_missing:{finding_id}")
        if not verification_allows_primary(decision):
            continue
        row["effect_direction"] = _enum_value(row["effect_direction"])
        primary.append(row)
    return sorted(primary, key=lambda row: str(row["finding_id"]))


def _moderator_key(spec: ModeratorSpec) -> str:
    if spec.kind == "within_paper":
        return f"mod__{spec.name}__paper_summary"
    return f"mod__{spec.name}"


def _row_value(row: Mapping[str, Any], name: str) -> Any:
    if name in row:
        return row[name]
    raw_name = name.removeprefix("mod__")
    moderators = row.get("moderators")
    if isinstance(moderators, Mapping):
        return moderators.get(raw_name)
    return None


def moderator_inventory(
    rows: Sequence[Mapping[str, Any]], config: QuestionConfig, *, reason: str
) -> list[dict[str, Any]]:
    """Emit every configured moderator without running inferential CV."""

    all_papers = {row["paper_id"] for row in rows}
    inventory: list[dict[str, Any]] = []
    for spec in config.moderators:
        key = _moderator_key(spec)
        nonmissing = [row for row in rows if _row_value(row, key) is not None]
        support: dict[str, set[str]] = defaultdict(set)
        for row in nonmissing:
            support[str(_row_value(row, key))].add(row["paper_id"])
        inventory.append(
            {
                "moderator": spec.name,
                "display_name": spec.display_name,
                "role": spec.role,
                "kind": spec.kind,
                "permutation": spec.permutation,
                "status": "not_run",
                "reason": reason,
                "n_findings": len(nonmissing),
                "n_papers": len({row["paper_id"] for row in nonmissing}),
                "coverage_findings": len(nonmissing) / len(rows) if rows else 0.0,
                "coverage_papers": (
                    len({row["paper_id"] for row in nonmissing}) / len(all_papers)
                    if all_papers
                    else 0.0
                ),
                "support_papers": {level: len(papers) for level, papers in sorted(support.items())},
                "k": None,
                "delta_ll": None,
                "positive_folds": None,
                "westfall_young_p": None,
            }
        )
    return inventory


def _disagreement_artifact(
    primary_rows: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    if not primary_rows:
        raise AnalysisContractError("primary_cohort_empty")
    finding = paper_balanced_finding_summary(primary_rows)
    paper = paper_modal_summary(primary_rows)
    bootstraps = bootstrap_disagreement(primary_rows, n_bootstraps=1000, seed=seed)
    return {"finding_level": finding, "paper_level": paper, "bootstrap": bootstraps}


def _axis_levels(config: QuestionConfig) -> dict[str, list[Any]]:
    assert config.variant_b is not None
    by_name = {spec.name: spec for spec in config.moderators}
    return {name: by_name[name].declared_levels for name in config.variant_b.axes}


def _deterministic_components(
    *,
    config: QuestionConfig,
    papers: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    primary_rows: Sequence[Mapping[str, Any]],
    inventory_reason: str,
) -> dict[str, Any]:
    assert config.outcomes.primary_family is not None
    assert config.variant_b is not None
    disagreement = _disagreement_artifact(primary_rows, seed=config.analysis.seed)
    inventory = moderator_inventory(primary_rows, config, reason=inventory_reason)
    contradictions = find_contradictions(
        primary_rows,
        papers,
        primary_family=config.outcomes.primary_family,
    )
    gaps = build_evidence_gap_grid(
        findings,
        axis_levels=_axis_levels(config),
        primary_endpoints=config.variant_b.primary_endpoints,
    )
    return {
        "disagreement": disagreement,
        "moderators": inventory,
        "contradictions": contradictions,
        "evidence_gaps": gaps,
        "residuals": residual_summary(contradictions),
    }


def compute_deterministic_components(
    *,
    config: QuestionConfig,
    papers: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    primary_rows: Sequence[Mapping[str, Any]],
    inventory_reason: str,
) -> dict[str, Any]:
    """Public deterministic component builder used by checkpoint producers/validators."""

    return _deterministic_components(
        config=config,
        papers=[_normalize_record(row) for row in papers],
        findings=[_normalize_record(row) for row in findings],
        primary_rows=[_normalize_record(row) for row in primary_rows],
        inventory_reason=inventory_reason,
    )


def deterministic_checkpoint_hashes(
    *,
    primary_rows: Sequence[Mapping[str, Any]],
    papers: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    components: Mapping[str, Any],
    config: QuestionConfig,
) -> CheckpointArtifactHashes:
    """Bind the deterministic descriptive/residual/gap inputs and outputs."""

    return CheckpointArtifactHashes(
        descriptive_inputs=hash_canonical(
            {
                "primary_rows": primary_rows,
                "moderators": [spec.model_dump(mode="json") for spec in config.moderators],
                "seed": config.analysis.seed,
            }
        ),
        descriptive_outputs=hash_canonical(
            {
                "disagreement": components["disagreement"],
                "moderators": components["moderators"],
            }
        ),
        residual_inputs=hash_canonical(
            {
                "primary_rows": primary_rows,
                "papers": papers,
                "primary_family": config.outcomes.primary_family,
            }
        ),
        residual_outputs=hash_canonical(components["contradictions"]),
        evidence_gap_inputs=hash_canonical(
            {
                "findings": findings,
                "axis_levels": _axis_levels(config),
                "endpoints": config.variant_b.primary_endpoints
                if config.variant_b is not None
                else [],
            }
        ),
        evidence_gap_outputs=hash_canonical(components["evidence_gaps"]),
    )


def _comparison_matches(candidate: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    left = candidate.get("contrast") or {}
    right = expected.get("contrast") or {}
    observed = (
        left.get("level_a"),
        left.get("direction_a"),
        left.get("level_b"),
        left.get("direction_b"),
    )
    wanted = (
        right.get("level_a"),
        right.get("direction_a"),
        right.get("level_b"),
        right.get("direction_b"),
    )
    return observed == wanted or (observed[2], observed[3], observed[0], observed[1]) == wanted


def _registered_moderator_results(
    primary_rows: Sequence[Mapping[str, Any]],
    all_valid_rows: Sequence[Mapping[str, Any]],
    config: QuestionConfig,
) -> list[dict[str, Any]]:
    tested = [spec for spec in config.moderators if spec.role == "tested"]
    evaluated = evaluate_all_moderators(
        primary_rows,
        tested,
        seed=config.analysis.seed,
        max_folds=config.analysis.cv_max_folds,
    )
    spec_by_name = {spec.name: spec for spec in tested}
    output: list[dict[str, Any]] = []
    for raw in evaluated:
        result = dict(raw)
        spec = spec_by_name[result["moderator"]]
        key = _moderator_key(spec)
        comparison = comparison_statistics(
            primary_rows,
            key,
            config_level_order=spec.declared_levels,
            minimum_level_papers=config.analysis.min_papers_per_level,
        )
        sensitivity = comparison_statistics(
            all_valid_rows,
            key,
            config_level_order=spec.declared_levels,
            minimum_level_papers=config.analysis.min_papers_per_level,
        )
        support = comparison.get("support_papers", {})
        contrast = comparison.get("contrast") or {}
        narrated = [
            int(support.get(str(contrast.get("level_a")), 0)),
            int(support.get(str(contrast.get("level_b")), 0)),
        ]
        result.update(
            {
                "moderator_key": key,
                "comparison": comparison,
                "narrated_level_support": narrated,
                "all_valid_sensitivity": {
                    "positive_gain": sensitivity.get("absolute_gain") is not None
                    and sensitivity["absolute_gain"] > 0,
                    "directions_preserved": _comparison_matches(sensitivity, comparison),
                    "comparison": sensitivity,
                },
            }
        )
        output.append(result)
    return output


def _run_permutations(
    primary_rows: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    config: QuestionConfig,
    *,
    progress: Callable[[int, Sequence[Mapping[str, Any] | None]], None] | None = None,
) -> dict[str, Any]:
    eligible = [
        result
        for result in results
        if result.get("status") == "eligible" and isinstance(result.get("delta_ll"), (float, int))
    ]
    if not eligible:
        return {
            "status": "complete",
            "reason": "no_permutation_eligible_moderators",
            "required_successes": config.analysis.permutation_count,
            "max_attempts": 125,
            "attempt_count": 0,
            "success_count": 0,
            "guard_failures": [],
            "family": [],
            "p_values": {},
        }
    observed = {str(result["moderator"]): float(result["delta_ll"]) for result in eligible}
    specs = {spec.name: spec for spec in config.moderators}
    scores: list[Mapping[str, float] | None] = []
    success_count = 0
    for attempt in range(125):
        try:
            family: dict[str, float] = {}
            for result in eligible:
                key = str(result["moderator_key"])
                permuted = paper_permutation(
                    primary_rows,
                    key,
                    seed=config.analysis.seed,
                    attempt_index=attempt,
                )
                evaluated = evaluate_moderator(
                    permuted,
                    str(result["moderator"]),
                    seed=config.analysis.seed,
                    max_folds=config.analysis.cv_max_folds,
                    kind=specs[str(result["moderator"])].kind,
                    inferential_key=(
                        f"{result['moderator']}__paper_summary"
                        if specs[str(result["moderator"])].kind == "within_paper"
                        else None
                    ),
                )
                if evaluated.get("delta_ll") is None:
                    raise AnalysisContractError("permutation_moderator_became_ineligible")
                family[str(result["moderator"])] = float(evaluated["delta_ll"])
            scores.append(family)
            success_count += 1
        except Exception:
            scores.append(None)
        if progress and (attempt + 1) % 25 == 0:
            progress(attempt + 1, scores)
        if success_count >= config.analysis.permutation_count:
            break
    return summarize_westfall_young(
        observed,
        scores,
        required_successes=config.analysis.permutation_count,
        max_attempts=125,
    )


def _run_model_bootstraps(
    primary_rows: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    config: QuestionConfig,
    *,
    progress: Callable[[int, Sequence[Mapping[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    draws: list[dict[str, Any]] = []
    specs = {spec.name: spec for spec in config.moderators}
    for draw_index in range(config.analysis.bootstrap_count):
        try:
            sample = paper_bootstrap_sample(
                primary_rows, seed=config.analysis.seed, draw_index=draw_index
            )
            per_moderator: dict[str, dict[str, Any]] = {}
            deltas: list[tuple[float, str]] = []
            for original in results:
                name = str(original["moderator"])
                key = str(original["moderator_key"])
                evaluated = evaluate_moderator(
                    sample,
                    name,
                    seed=config.analysis.seed,
                    max_folds=config.analysis.cv_max_folds,
                    kind=specs[name].kind,
                    inferential_key=(
                        f"{name}__paper_summary" if specs[name].kind == "within_paper" else None
                    ),
                )
                comparison = comparison_statistics(
                    sample,
                    key,
                    config_level_order=next(
                        spec.declared_levels for spec in config.moderators if spec.name == name
                    ),
                    minimum_level_papers=config.analysis.min_papers_per_level,
                )
                material = (
                    comparison.get("absolute_gain") is not None
                    and comparison["absolute_gain"] >= 0.10
                )
                pattern = bool(
                    comparison.get("eligible")
                    and material
                    and _comparison_matches(comparison, original["comparison"])
                )
                delta = evaluated.get("delta_ll")
                if isinstance(delta, (float, int)):
                    deltas.append((float(delta), name))
                per_moderator[name] = {
                    "eligible": bool(comparison.get("eligible")),
                    "pattern_recurred": pattern,
                    "delta_ll": delta,
                    "absolute_gain": comparison.get("absolute_gain"),
                    "reason": comparison.get("reason") or evaluated.get("reason"),
                }
            top_three = {
                name for _, name in sorted(deltas, key=lambda item: (-item[0], item[1]))[:3]
            }
            for name, item in per_moderator.items():
                item["top3"] = name in top_three
            draws.append(
                {"draw_index": draw_index, "status": "success", "moderators": per_moderator}
            )
        except Exception as exc:
            draws.append(
                {
                    "draw_index": draw_index,
                    "status": "error",
                    "reason": type(exc).__name__,
                    "message": str(exc),
                    "moderators": {},
                }
            )
        if progress and (draw_index + 1) % 25 == 0:
            progress(draw_index + 1, draws)
    summaries: dict[str, dict[str, Any]] = {}
    for result in results:
        name = str(result["moderator"])
        items = [draw["moderators"].get(name) for draw in draws if draw["status"] == "success"]
        items = [item for item in items if item is not None]
        summaries[name] = {
            "n_bootstraps": config.analysis.bootstrap_count,
            "pattern_fraction": sum(item["pattern_recurred"] for item in items)
            / config.analysis.bootstrap_count,
            "eligible_fraction": sum(item["eligible"] for item in items)
            / config.analysis.bootstrap_count,
            "top3_fraction": sum(item["top3"] for item in items) / config.analysis.bootstrap_count,
        }
    return {
        "status": "complete",
        "n_bootstraps": config.analysis.bootstrap_count,
        "draws": draws,
        "error_count": sum(draw["status"] == "error" for draw in draws),
        "moderators": summaries,
    }


def _checkpoint_result(index: int, value: Mapping[str, Any] | None) -> CheckpointResult:
    if value is None:
        return CheckpointResult(
            index=index,
            status="guard_failure",
            result=None,
            error_code="guard_failure",
        )
    return CheckpointResult(index=index, status="success", result=dict(value), error_code=None)


def build_source_checkpoint(
    context: CheckpointContext,
    artifact_hashes: CheckpointArtifactHashes,
    *,
    permutation_scores: Sequence[Mapping[str, Any] | None],
    bootstrap_draws: Sequence[Mapping[str, Any]],
) -> M4SourceCheckpoint:
    """Build the foundation-owned exact checkpoint model from registered progress."""

    permutation_results = [
        _checkpoint_result(index, value) for index, value in enumerate(permutation_scores)
    ]
    bootstrap_results = [
        _checkpoint_result(index, value if value.get("status") == "success" else None)
        for index, value in enumerate(bootstrap_draws)
    ]
    return M4SourceCheckpoint(
        source_run_id=context.source_run_id,
        source_started_at=context.source_started_at,
        checkpointed_at=context.checkpointed_at(),
        question_id=context.question_id,
        config_sha256=context.config_sha256,
        code_version=context.code_version,
        cohort_sha256=context.cohort_sha256,
        g3_gate_sha256=context.g3_gate_sha256,
        input_hashes=context.input_hashes,
        seed=context.seed,
        registered_budgets=CheckpointBudgets(
            bootstrap_count=context.bootstrap_count,
            permutation_success_count=context.permutation_success_count,
            permutation_max_attempts=context.permutation_max_attempts,
        ),
        completed_bootstrap_indices=list(range(len(bootstrap_results))),
        completed_permutation_attempt_indices=list(range(len(permutation_results))),
        successful_permutation_indices=[
            result.index for result in permutation_results if result.status == "success"
        ],
        bootstrap_results=bootstrap_results,
        permutation_results=permutation_results,
        guard_failures=[
            f"permutation:{result.index}"
            for result in permutation_results
            if result.status == "guard_failure"
        ]
        + [
            f"bootstrap:{result.index}"
            for result in bootstrap_results
            if result.status == "guard_failure"
        ],
        artifact_hashes=artifact_hashes,
    )


def _attach_inference(
    results: list[dict[str, Any]],
    permutation: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> list[dict[str, Any]]:
    p_values = permutation.get("p_values", {})
    stability = bootstrap.get("moderators", bootstrap)
    for result in results:
        name = str(result["moderator"])
        result["westfall_young_p"] = (p_values.get(name) or {}).get("westfall_young")
        result["permutation_raw_p"] = (p_values.get(name) or {}).get("raw")
        result["stability"] = dict(stability.get(name, {}))
    return results


def _b_headline(
    *,
    reason: str,
    components: Mapping[str, Any],
    m4_gate: Mapping[str, Any],
) -> dict[str, Any]:
    paper = components["disagreement"]["paper_level"]
    interval = components["disagreement"]["bootstrap"]["paper_entropy"]["interval_90"]
    entropy = paper["primary"]["normalized_entropy"]
    if entropy is None or interval[0] is None or interval[1] is None:
        raise AnalysisContractError("variant_b_disagreement_is_undefined")
    gaps = components["evidence_gaps"]
    failures = [
        {
            "moderator": item["moderator"],
            "failed_rules": item.get("failed_rules", []),
        }
        for item in m4_gate.get("moderators", [])
        if item.get("passed") is False
    ]
    majority = components["disagreement"]["finding_level"]["majority"]
    headline = build_variant_b_headline(
        selection_reason=reason,
        disagreement={
            "n_papers": paper["n_papers_classifiable"],
            "n_findings": components["disagreement"]["finding_level"]["n_findings"],
            "paper_entropy": entropy,
            "interval_90": interval,
        },
        residuals=components["residuals"],
        sparse_or_empty_cells=sum(row["status"] in {"empty", "sparse"} for row in gaps),
        total_cells=len(gaps),
        m4_failures=failures,
        global_baseline={
            "modal_direction": majority["modal_direction"],
            "agreement_q": majority["agreement"],
        },
    )
    return headline


def _bind_m4_gate(
    gate: dict[str, Any],
    *,
    config: QuestionConfig,
    primary_rows: Sequence[Mapping[str, Any]],
    g3_gate: Mapping[str, Any],
    permutation: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    moderators: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind every M4 branch to its exact cohort and scientific inputs/outputs."""

    cohort_hash = cohort_sha256(primary_rows)
    gate.update(
        {
            "cohort_hash": cohort_hash,
            "config_sha256": config_sha256(config),
            "input_hashes": {
                "primary_cohort": cohort_hash,
                "g3_gate": hash_canonical(g3_gate),
            },
            "output_hashes": {
                "permutation": hash_canonical(permutation),
                "bootstrap": hash_canonical(bootstrap),
                "moderators": hash_canonical(moderators),
            },
        }
    )
    return gate


def analyze_s5(
    *,
    config: QuestionConfig,
    papers: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    verification: VerificationRecord | Mapping[str, Any],
    g3_gate: Mapping[str, Any],
    all_valid_rows: Sequence[Mapping[str, Any]] | None = None,
    inference_overrides: InferenceOverrides | None = None,
    checkpoint_context: CheckpointContext | None = None,
) -> AnalysisBundle:
    """Run ordinary s5 and make one atomic, deterministic narrative decision."""

    if config.status != "locked":
        raise AnalysisContractError("s5_requires_locked_config")
    if g3_gate.get("trust_passed") is not True:
        raise G3TrustBlockedError("g3_trust_failed")
    assert config.outcomes.primary_family is not None
    primary = derive_primary_cohort(
        papers,
        findings,
        verification,
        primary_family=config.outcomes.primary_family,
    )
    normalized_papers = [_normalize_record(row) for row in papers]
    normalized_findings = [_normalize_record(row) for row in findings]
    components = _deterministic_components(
        config=config,
        papers=normalized_papers,
        findings=normalized_findings,
        primary_rows=primary,
        inventory_reason=(
            "g3_story_not_viable"
            if g3_gate.get("story_passed") is not True
            else "m4_registered_battery"
        ),
    )
    checkpoint_components = dict(components)
    checkpoint_components["moderators"] = moderator_inventory(
        primary, config, reason="m4_incomplete"
    )
    checkpoint_hashes = deterministic_checkpoint_hashes(
        primary_rows=primary,
        papers=normalized_papers,
        findings=normalized_findings,
        components=checkpoint_components,
        config=config,
    )
    entropy_bootstrap = {"status": "complete", **components["disagreement"]["bootstrap"]}

    if g3_gate.get("story_passed") is not True:
        tested_inventory = [
            item for item in components["moderators"] if item.get("role") == "tested"
        ]
        permutation = {
            "status": "not_run",
            "reason": "g3_story_not_viable",
            "success_count": 0,
            "attempt_count": 0,
        }
        model_bootstrap = {
            "status": "not_run",
            "reason": "g3_story_not_viable",
            "n_bootstraps": 0,
        }
        m4_gate = _bind_m4_gate(
            evaluate_m4_gate(
                tested_inventory,
                config_order=[item["moderator"] for item in tested_inventory],
                g3_story_passed=False,
            ),
            config=config,
            primary_rows=primary,
            g3_gate=g3_gate,
            permutation=permutation,
            bootstrap=model_bootstrap,
            moderators=components["moderators"],
        )
        headline = _b_headline(reason="g3_story_not_viable", components=components, m4_gate=m4_gate)
        return AnalysisBundle(
            json_artifacts={
                "m4_checkpoint.json": M4CheckpointNotApplicable(reason="m4_not_run").model_dump(
                    mode="json"
                ),
                "tree.json": {
                    "status": "not_run",
                    "reason": "g3_story_not_viable",
                    "nodes": [],
                },
                "bootstrap.json": {
                    "status": "complete",
                    "entropy": entropy_bootstrap,
                    "model_stability": model_bootstrap,
                },
                "permutation.json": permutation,
                "m4_gate.json": m4_gate,
                "headline.json": headline,
            },
            table_artifacts={
                "moderators.parquet": components["moderators"],
                "contradictions.parquet": components["contradictions"],
                "evidence_gaps.parquet": components["evidence_gaps"],
            },
            primary_rows=primary,
            trace={"status": "not_run", "reason": "g3_story_not_viable"},
        )

    all_valid = [dict(row) for row in (all_valid_rows or primary)]
    overrides = inference_overrides or InferenceOverrides()
    if overrides.moderator_results is None:
        results = _registered_moderator_results(primary, all_valid, config)
    else:
        results = [dict(result) for result in overrides.moderator_results]

    permutation_scores_for_checkpoint: list[Mapping[str, Any] | None] = []
    bootstrap_draws_for_checkpoint: list[Mapping[str, Any]] = []

    def permutation_progress(_attempts: int, scores: Sequence[Mapping[str, Any] | None]) -> None:
        permutation_scores_for_checkpoint[:] = scores
        if checkpoint_context is not None:
            checkpoint_context.writer(
                build_source_checkpoint(
                    checkpoint_context,
                    checkpoint_hashes,
                    permutation_scores=permutation_scores_for_checkpoint,
                    bootstrap_draws=bootstrap_draws_for_checkpoint,
                )
            )

    def bootstrap_progress(_draws: int, values: Sequence[Mapping[str, Any]]) -> None:
        bootstrap_draws_for_checkpoint[:] = values
        if checkpoint_context is not None:
            checkpoint_context.writer(
                build_source_checkpoint(
                    checkpoint_context,
                    checkpoint_hashes,
                    permutation_scores=permutation_scores_for_checkpoint,
                    bootstrap_draws=bootstrap_draws_for_checkpoint,
                )
            )

    permutation = (
        dict(overrides.permutation)
        if overrides.permutation is not None
        else _run_permutations(primary, results, config, progress=permutation_progress)
    )
    model_bootstrap = (
        {"status": "complete", "moderators": dict(overrides.bootstrap_stability)}
        if overrides.bootstrap_stability is not None
        else _run_model_bootstraps(primary, results, config, progress=bootstrap_progress)
    )
    results = _attach_inference(results, permutation, model_bootstrap)
    inferred_by_name = {str(result["moderator"]): result for result in results}
    inventory_by_name = {str(result["moderator"]): result for result in components["moderators"]}
    moderator_table = [
        inferred_by_name.get(spec.name, inventory_by_name[spec.name]) for spec in config.moderators
    ]
    config_order = [spec.name for spec in config.moderators if spec.role == "tested"]
    m4_gate = _bind_m4_gate(
        evaluate_m4_gate(results, config_order=config_order, seed=config.analysis.seed),
        config=config,
        primary_rows=primary,
        g3_gate=g3_gate,
        permutation=permutation,
        bootstrap=model_bootstrap,
        moderators=moderator_table,
    )
    selected = m4_gate["selected_moderator"]
    if m4_gate["selected_variant"] == "A":
        winner = next(result for result in results if result["moderator"] == selected)
        display = next(spec.display_name for spec in config.moderators if spec.name == selected)
        headline = build_variant_a_headline(winner, moderator_display_name=display)
        trace = None
        tree = build_tree_artifact(
            primary,
            [spec.name for spec in config.moderators],
            seed=config.analysis.seed,
            supporting=True,
        )
    else:
        headline = _b_headline(reason="m4_no_moderator", components=components, m4_gate=m4_gate)
        trace = {"status": "not_run", "reason": "m4_selected_variant_b"}
        tree = build_tree_artifact(
            primary,
            [spec.name for spec in config.moderators],
            seed=config.analysis.seed,
            supporting=False,
        )
    root_split = tree.get("root_split")
    if isinstance(root_split, Mapping) and isinstance(root_split.get("moderator"), str):
        tree["root_split_bootstrap"] = {
            "status": "complete",
            **root_split_bootstrap_frequency(
                primary,
                [spec.name for spec in config.moderators],
                expected_moderator=str(root_split["moderator"]),
                n_bootstraps=config.analysis.bootstrap_count,
                seed=config.analysis.seed,
            ),
        }
    else:
        tree["root_split_bootstrap"] = {
            "status": "not_run",
            "reason": "no_observed_root_split",
            "n_bootstraps": 0,
            "frequency": None,
        }
    return AnalysisBundle(
        json_artifacts={
            "m4_checkpoint.json": M4CheckpointNotApplicable(reason="m4_completed").model_dump(
                mode="json"
            ),
            "tree.json": tree,
            "bootstrap.json": {
                "status": "complete",
                "entropy": entropy_bootstrap,
                "model_stability": model_bootstrap,
            },
            "permutation.json": permutation,
            "m4_gate.json": m4_gate,
            "headline.json": headline,
        },
        table_artifacts={
            "moderators.parquet": moderator_table,
            "contradictions.parquet": components["contradictions"],
            "evidence_gaps.parquet": components["evidence_gaps"],
        },
        primary_rows=primary,
        trace=trace,
    )


def finalize_incomplete_s5(
    *,
    checkpoint: M4SourceCheckpoint | Mapping[str, Any],
    config: QuestionConfig,
    papers: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    verification: VerificationRecord | Mapping[str, Any],
    g3_gate: Mapping[str, Any],
    expected_config_sha256: str,
    expected_code_version: str,
    expected_cohort_sha256: str,
    expected_g3_gate_sha256: str,
    expected_input_hashes: Mapping[str, str],
) -> AnalysisBundle:
    """Freeze a genuine interruption without performing another resampling draw."""

    source = (
        checkpoint
        if isinstance(checkpoint, M4SourceCheckpoint)
        else M4SourceCheckpoint.model_validate(checkpoint)
    )
    expected = {
        "question_id": config.question_id,
        "config_sha256": expected_config_sha256,
        "code_version": expected_code_version,
        "cohort_sha256": expected_cohort_sha256,
        "g3_gate_sha256": expected_g3_gate_sha256,
        "input_hashes": dict(expected_input_hashes),
        "seed": config.analysis.seed,
    }
    observed = {
        "question_id": source.question_id,
        "config_sha256": source.config_sha256,
        "code_version": source.code_version,
        "cohort_sha256": source.cohort_sha256,
        "g3_gate_sha256": source.g3_gate_sha256,
        "input_hashes": source.input_hashes,
        "seed": source.seed,
    }
    mismatches = [name for name in expected if observed[name] != expected[name]]
    if mismatches:
        raise AnalysisContractError("checkpoint_identity_mismatch:" + ",".join(sorted(mismatches)))
    if not source.genuinely_incomplete:
        raise AnalysisContractError("checkpoint_not_genuinely_incomplete")
    if g3_gate.get("trust_passed") is not True or g3_gate.get("story_passed") is not True:
        raise AnalysisContractError("incomplete_finalization_requires_g3_trust_and_story")
    assert config.outcomes.primary_family is not None
    primary = derive_primary_cohort(
        papers,
        findings,
        verification,
        primary_family=config.outcomes.primary_family,
    )
    normalized_papers = [_normalize_record(row) for row in papers]
    normalized_findings = [_normalize_record(row) for row in findings]
    components = _deterministic_components(
        config=config,
        papers=normalized_papers,
        findings=normalized_findings,
        primary_rows=primary,
        inventory_reason="m4_incomplete",
    )
    hashes = deterministic_checkpoint_hashes(
        primary_rows=primary,
        papers=normalized_papers,
        findings=normalized_findings,
        components=components,
        config=config,
    )
    if hashes != source.artifact_hashes:
        raise AnalysisContractError("checkpoint_deterministic_artifact_hash_mismatch")
    source_hash = canonical_model_sha256(source)
    wrapper = M4CheckpointFrozenIncomplete(
        source_checkpoint_sha256=source_hash,
        checkpoint=source,
    )
    permutation = {
        "status": "incomplete",
        "reason": "m4_incomplete",
        "success_count": len(source.successful_permutation_indices),
        "attempt_count": len(source.completed_permutation_attempt_indices),
        "required_successes": source.registered_budgets.permutation_success_count,
        "max_attempts": source.registered_budgets.permutation_max_attempts,
        "attempts": [item.model_dump(mode="json") for item in source.permutation_results],
    }
    model_bootstrap = {
        "status": "incomplete",
        "completed_draws": len(source.completed_bootstrap_indices),
        "registered_draws": source.registered_budgets.bootstrap_count,
        "draws": [item.model_dump(mode="json") for item in source.bootstrap_results],
    }
    tested_inventory = [item for item in components["moderators"] if item.get("role") == "tested"]
    m4_gate = _bind_m4_gate(
        evaluate_m4_gate(
            tested_inventory,
            config_order=[spec.name for spec in config.moderators if spec.role == "tested"],
            seed=config.analysis.seed,
            completion_status="incomplete",
        ),
        config=config,
        primary_rows=primary,
        g3_gate=g3_gate,
        permutation=permutation,
        bootstrap=model_bootstrap,
        moderators=components["moderators"],
    )
    headline = _b_headline(reason="m4_incomplete", components=components, m4_gate=m4_gate)
    return AnalysisBundle(
        json_artifacts={
            "m4_checkpoint.json": wrapper.model_dump(mode="json"),
            "tree.json": {
                "status": "incomplete",
                "reason": "m4_incomplete",
                "nodes": [],
            },
            "bootstrap.json": {
                "status": "incomplete",
                "entropy": {"status": "complete", **components["disagreement"]["bootstrap"]},
                "model_stability": model_bootstrap,
            },
            "permutation.json": permutation,
            "m4_gate.json": m4_gate,
            "headline.json": headline,
        },
        table_artifacts={
            "moderators.parquet": components["moderators"],
            "contradictions.parquet": components["contradictions"],
            "evidence_gaps.parquet": components["evidence_gaps"],
        },
        primary_rows=primary,
        trace={"status": "not_run", "reason": "m4_incomplete"},
        completion_mode="frozen_incomplete",
        checkpoint_sha256=source_hash,
        source_started_at=source.source_started_at,
        source_completed_at=source.checkpointed_at,
    )


def _parquet_scalar(value: Any) -> Any:
    value = _enum_value(value)
    if isinstance(value, (Mapping, list, tuple, set)):
        return canonical_json_bytes(value).decode("utf-8")
    return value


def _write_parquet_atomic(path: Path, rows: Sequence[Mapping[str, Any]], *, force: bool) -> None:
    if path.exists() and not force:
        raise AnalysisContractError(f"output_exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [{str(key): _parquet_scalar(value) for key, value in row.items()} for row in rows]
    frame = pd.DataFrame(normalized)
    if frame.empty and path.name in TABLE_SCHEMAS:
        frame = pd.DataFrame(columns=TABLE_SCHEMAS[path.name])
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_analysis_bundle(
    bundle: AnalysisBundle, output_dir: Path, *, force: bool = False
) -> dict[str, Path]:
    """Persist a fully-computed bundle with per-file atomic replacement guards."""

    planned = [output_dir / name for name in bundle.json_artifacts]
    planned += [output_dir / name for name in bundle.table_artifacts]
    if bundle.trace is not None:
        planned.append(output_dir / "trace.json")
    if not force:
        existing = [path for path in planned if path.exists()]
        if existing:
            raise AnalysisContractError("outputs_exist:" + ",".join(path.name for path in existing))
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, value in bundle.json_artifacts.items():
        path = output_dir / name
        atomic_write_json(path, value, force=force)
        written[name] = path
    for name, rows in bundle.table_artifacts.items():
        path = output_dir / name
        _write_parquet_atomic(path, rows, force=force)
        written[name] = path
    if bundle.trace is not None:
        path = output_dir / "trace.json"
        atomic_write_json(path, bundle.trace, force=force)
        written["trace.json"] = path
    return written


__all__ = [
    "JSON_ARTIFACT_NAMES",
    "TABLE_ARTIFACT_NAMES",
    "TABLE_SCHEMAS",
    "AnalysisBundle",
    "AnalysisContractError",
    "CheckpointContext",
    "G3TrustBlockedError",
    "InferenceOverrides",
    "analyze_s5",
    "build_source_checkpoint",
    "compute_deterministic_components",
    "derive_primary_cohort",
    "deterministic_checkpoint_hashes",
    "finalize_incomplete_s5",
    "moderator_inventory",
    "read_parquet_records",
    "write_analysis_bundle",
]
