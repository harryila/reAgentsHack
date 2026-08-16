"""Guarded exploratory moderator proposal, echo reconciliation, and incremental testing."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from literature_multiverse.disagreement import CANONICAL_LABELS, paper_balanced_weights
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.moderators import (
    align_and_floor_probabilities,
    feasible_grouped_splits,
    weighted_log_loss,
)
from literature_multiverse.resampling import paper_bootstrap_sample

_SLUG = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$")
_PROPOSAL_TYPES = frozenset({"categorical", "float", "int", "bool"})
_KINDS = frozenset({"paper_constant", "within_paper"})
_PERMUTATIONS = frozenset({"paper", "paper_summary"})


class RemapContractError(ValueError):
    """The exploratory remap protocol was violated."""


@dataclass(frozen=True, slots=True)
class RemapInferenceOverrides:
    """Offline injection point for expensive registered candidate-only inference."""

    incremental_cv: Mapping[str, Any] | None = None
    permutation: Mapping[str, Any] | None = None
    bootstrap: Mapping[str, Any] | None = None
    all_valid_sensitivity: Mapping[str, Any] | None = None


def select_proposal_pairs(
    contradictions: Sequence[Mapping[str, Any]], *, minimum: int = 5, maximum: int = 10
) -> list[dict[str, Any]]:
    """Select the fixed contradiction ranking prefix without result-aware reranking."""

    if not 1 <= minimum <= maximum:
        raise RemapContractError("invalid_proposal_pair_budget")
    ordered = sorted(
        (dict(pair) for pair in contradictions),
        key=lambda pair: (float(pair.get("distance", math.inf)), str(pair.get("pair_id", ""))),
    )
    if not ordered:
        raise RemapContractError("no_residual_pairs_for_proposal")
    if len(ordered) < minimum:
        raise RemapContractError(
            f"insufficient_residual_pairs_for_proposal:{len(ordered)}<{minimum}"
        )
    return ordered[:maximum]


def _validate_moderator_spec(
    raw: Mapping[str, Any], *, base_moderator_names: Sequence[str]
) -> dict[str, Any]:
    required = {
        "name",
        "type",
        "kind",
        "categories",
        "bins",
        "paper_summary",
        "permutation",
        "extraction_prompt",
        "rationale",
    }
    if set(raw) != required:
        raise RemapContractError(
            "proposal_spec_keys_mismatch:"
            f"missing={sorted(required - set(raw))}:extra={sorted(set(raw) - required)}"
        )
    name = raw["name"]
    if not isinstance(name, str) or not _SLUG.fullmatch(name):
        raise RemapContractError("proposal_invalid_name")
    if name in set(base_moderator_names):
        raise RemapContractError("proposal_duplicates_prespecified_moderator")
    proposal_type = raw["type"]
    if proposal_type not in _PROPOSAL_TYPES:
        raise RemapContractError("proposal_invalid_type")
    kind = raw["kind"]
    permutation = raw["permutation"]
    paper_summary = raw["paper_summary"]
    if kind not in _KINDS or permutation not in _PERMUTATIONS:
        raise RemapContractError("proposal_invalid_kind_or_permutation")
    if kind == "paper_constant" and (permutation != "paper" or paper_summary is not None):
        raise RemapContractError("proposal_paper_constant_requires_paper_permutation")
    if kind == "within_paper" and (
        permutation != "paper_summary"
        or not isinstance(paper_summary, str)
        or not paper_summary.strip()
    ):
        raise RemapContractError("proposal_within_paper_requires_approved_summary")
    categories = raw["categories"]
    bins = raw["bins"]
    if proposal_type in {"categorical", "bool"}:
        if not isinstance(categories, list) or len(categories) < 2 or bins is not None:
            raise RemapContractError("proposal_categorical_requires_categories_only")
        if len({hash_canonical(value) for value in categories}) != len(categories):
            raise RemapContractError("proposal_duplicate_category")
    elif categories is not None or not isinstance(bins, list) or len(bins) < 2:
        raise RemapContractError("proposal_numeric_requires_bins_only")
    if not isinstance(raw["extraction_prompt"], str) or not raw["extraction_prompt"].strip():
        raise RemapContractError("proposal_missing_extraction_prompt")
    if not isinstance(raw["rationale"], str) or not raw["rationale"].strip():
        raise RemapContractError("proposal_missing_rationale")
    return dict(raw)


def propose_remap(
    contradictions: Sequence[Mapping[str, Any]],
    frozen_headline: Mapping[str, Any],
    *,
    base_moderator_names: Sequence[str],
    proposer: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Call an injected proposal model once with only logged inputs."""

    if frozen_headline.get("narrative_variant") != "A":
        raise RemapContractError("remap_proposal_requires_variant_a")
    pairs = select_proposal_pairs(contradictions)
    logged_input = {
        "frozen_headline_sha256": hash_canonical(frozen_headline),
        "frozen_headline": dict(frozen_headline),
        "residual_pairs": pairs,
        "instruction": (
            "Propose one exploratory context field not in the pre-specified family; do not "
            "reinterpret outcomes or replace the frozen headline."
        ),
    }
    raw = proposer(logged_input)
    if not isinstance(raw, Mapping):
        raise RemapContractError("proposal_model_output_must_be_mapping")
    moderator = _validate_moderator_spec(raw, base_moderator_names=base_moderator_names)
    artifact = {
        "proposal_version": "1",
        "status": "proposed",
        "frozen_headline_sha256": logged_input["frozen_headline_sha256"],
        "proposal_input_sha256": hash_canonical(logged_input),
        "residual_pair_ids": [pair["pair_id"] for pair in pairs],
        "moderator": moderator,
    }
    artifact["proposal_sha256"] = hash_canonical(
        {key: value for key, value in artifact.items() if key != "proposal_sha256"}
    )
    return artifact


def approval_template(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Generate the exact human-editable approval record; null is not approval."""

    return {
        "approval_version": "1",
        "proposal_sha256": proposal.get("proposal_sha256"),
        "approved": None,
        "approved_moderator": proposal.get("moderator"),
        "reviewer": None,
        "reason": None,
    }


def validate_approval(
    proposal: Mapping[str, Any], approval: Mapping[str, Any]
) -> Literal["approved", "rejected"]:
    required = {
        "approval_version",
        "proposal_sha256",
        "approved",
        "approved_moderator",
        "reviewer",
        "reason",
    }
    if set(approval) != required or approval.get("approval_version") != "1":
        raise RemapContractError("approval_contract_invalid")
    if approval.get("proposal_sha256") != proposal.get("proposal_sha256"):
        raise RemapContractError("approval_proposal_hash_mismatch")
    if approval.get("approved") not in {True, False}:
        raise RemapContractError("human_approval_unavailable")
    if not isinstance(approval.get("reviewer"), str) or not approval["reviewer"].strip():
        raise RemapContractError("approval_requires_reviewer")
    if not isinstance(approval.get("reason"), str) or not approval["reason"].strip():
        raise RemapContractError("approval_requires_reason")
    if approval["approved"]:
        if approval.get("approved_moderator") != proposal.get("moderator"):
            raise RemapContractError("approved_moderator_differs_from_proposal")
        return "approved"
    if approval.get("approved_moderator") is not None:
        raise RemapContractError("rejected_approval_moderator_must_be_null")
    return "rejected"


def reconcile_echo_responses(
    expected_finding_ids: Sequence[str],
    responses: Sequence[Mapping[str, Any]],
    *,
    minimum_join_fraction: float = 0.95,
    value_validator: Callable[[Any], bool] | None = None,
) -> dict[str, Any]:
    """Reconcile echo IDs while allowing at most the registered 5% missing responses."""

    expected = list(expected_finding_ids)
    if not expected or len(expected) != len(set(expected)):
        raise RemapContractError("expected_remap_ids_must_be_nonempty_unique")
    expected_set = set(expected)
    observed: dict[str, Any] = {}
    duplicates: set[str] = set()
    unknown: set[str] = set()
    invalid: list[int] = []
    for index, response in enumerate(responses):
        if set(response) != {"finding_id", "value"}:
            invalid.append(index)
            continue
        finding_id = response.get("finding_id")
        if not isinstance(finding_id, str) or not finding_id:
            invalid.append(index)
            continue
        if finding_id in observed:
            duplicates.add(finding_id)
            continue
        if finding_id not in expected_set:
            unknown.add(finding_id)
            continue
        value = response.get("value")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            invalid.append(index)
            continue
        if value_validator is not None and not value_validator(value):
            invalid.append(index)
            continue
        observed[finding_id] = value
    missing = sorted(expected_set - observed.keys())
    join_fraction = len(observed) / len(expected)
    technical_valid = (
        not duplicates and not unknown and not invalid and join_fraction >= minimum_join_fraction
    )
    table = [
        {
            "finding_id": finding_id,
            "value": observed.get(finding_id),
            "response_status": "valid" if finding_id in observed else "missing",
        }
        for finding_id in expected
    ]
    return {
        "technical_valid": technical_valid,
        "minimum_join_fraction": minimum_join_fraction,
        "expected_count": len(expected),
        "valid_unique_count": len(observed),
        "join_fraction": join_fraction,
        "null_count": sum(value is None for value in observed.values()),
        "duplicates": sorted(duplicates),
        "unknown": sorted(unknown),
        "missing": missing,
        "invalid_response_indices": invalid,
        "table": table,
    }


def execute_approved_remap(
    *,
    proposal: Mapping[str, Any],
    approval: Mapping[str, Any],
    primary_rows: Sequence[Mapping[str, Any]],
    mapper: Callable[[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Run an injected mapper on the entire eligible v1 primary cohort."""

    state = validate_approval(proposal, approval)
    if state != "approved":
        return {
            "status": "not_run",
            "reason": "human_rejected",
            "proposal_sha256": proposal["proposal_sha256"],
            "side_table": [],
        }
    finding_ids = [str(row["finding_id"]) for row in primary_rows]
    if len(finding_ids) != len(set(finding_ids)):
        raise RemapContractError("primary_remap_finding_ids_not_unique")
    moderator = proposal["moderator"]
    request = {
        "proposal_sha256": proposal["proposal_sha256"],
        "moderator": moderator,
        "findings": [
            {
                "finding_id": row["finding_id"],
                "evidence_quote": row.get("evidence_quote"),
                "outcome_name": row.get("outcome_name"),
                "dose_raw": row.get("dose_raw"),
            }
            for row in primary_rows
        ],
    }
    raw_responses = mapper(request)

    def valid_value(value: Any) -> bool:
        if value is None:
            return True
        proposal_type = moderator["type"]
        if proposal_type == "categorical":
            return isinstance(value, str) and value in moderator["categories"]
        if proposal_type == "bool":
            return type(value) is bool and value in moderator["categories"]
        if proposal_type == "int":
            return isinstance(value, int) and not isinstance(value, bool)
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    reconciliation = reconcile_echo_responses(
        finding_ids, raw_responses, value_validator=valid_value
    )
    quarantine = [
        {"finding_id": finding_id, "reason": "remap_missing_response"}
        for finding_id in reconciliation["missing"]
    ]
    for index in reconciliation["invalid_response_indices"]:
        quarantine.append({"response_index": index, "reason": "remap_invalid_response"})
    return {
        "status": "complete",
        "proposal_sha256": proposal["proposal_sha256"],
        "request_sha256": hash_canonical(request),
        "moderator": moderator,
        "reconciliation": {key: value for key, value in reconciliation.items() if key != "table"},
        "side_table": reconciliation["table"],
        "quarantine": quarantine,
    }


def _joined_rows(
    primary_rows: Sequence[Mapping[str, Any]], side_table: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    values = {str(row["finding_id"]): row.get("value") for row in side_table}
    output: list[dict[str, Any]] = []
    for row in primary_rows:
        clone = dict(row)
        clone["__candidate__"] = values.get(str(row["finding_id"]))
        output.append(clone)
    return output


def _base_value(row: Mapping[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    moderators = row.get("moderators")
    if isinstance(moderators, Mapping):
        return moderators.get(key.removeprefix("mod__"))
    return None


def incremental_cv(
    rows: Sequence[Mapping[str, Any]],
    *,
    base_moderator_key: str,
    seed: int,
    max_folds: int = 5,
) -> dict[str, Any]:
    """Compare frozen base vs base+candidate on identical rows, folds, and weights."""

    subset = [
        dict(row)
        for row in rows
        if row.get("effect_direction") in CANONICAL_LABELS
        and _base_value(row, base_moderator_key) is not None
        and row.get("__candidate__") is not None
    ]
    if not subset:
        return {
            "status": "insufficient_for_cv",
            "k": None,
            "delta_ll": None,
            "positive_folds": None,
            "folds": [],
            "n_rows": 0,
        }
    labels = [str(row["effect_direction"]) for row in subset]
    papers = [str(row["paper_id"]) for row in subset]
    split = feasible_grouped_splits(labels, papers, seed=seed, max_folds=max_folds)
    if not split["splits"]:
        return {
            "status": split["status"],
            "k": split["k"],
            "delta_ll": None,
            "positive_folds": None,
            "folds": [],
            "n_rows": len(subset),
        }
    base = np.asarray([[str(_base_value(row, base_moderator_key))] for row in subset], dtype=object)
    candidate = np.asarray([[str(row["__candidate__"])] for row in subset], dtype=object)
    y = np.asarray(labels, dtype=object)
    folds: list[dict[str, Any]] = []
    for fold_index, (train, test) in enumerate(split["splits"]):
        base_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        candidate_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        base_train = base_encoder.fit_transform(base[train])
        base_test = base_encoder.transform(base[test])
        candidate_train = candidate_encoder.fit_transform(candidate[train])
        candidate_test = candidate_encoder.transform(candidate[test])
        after_train = np.concatenate([base_train, candidate_train], axis=1)
        after_test = np.concatenate([base_test, candidate_test], axis=1)
        train_rows = [subset[index] for index in train]
        test_rows = [subset[index] for index in test]
        train_weights = np.asarray(paper_balanced_weights(train_rows), dtype=float)
        test_weights = np.asarray(paper_balanced_weights(test_rows), dtype=float)
        before_model = LogisticRegression(C=1, solver="lbfgs", max_iter=2000)
        after_model = LogisticRegression(C=1, solver="lbfgs", max_iter=2000)
        before_model.fit(base_train, y[train], sample_weight=train_weights)
        after_model.fit(after_train, y[train], sample_weight=train_weights)
        before_probability = align_and_floor_probabilities(
            before_model.predict_proba(base_test), before_model.classes_
        )
        after_probability = align_and_floor_probabilities(
            after_model.predict_proba(after_test), after_model.classes_
        )
        before_loss = weighted_log_loss(y[test].tolist(), before_probability, test_weights)
        after_loss = weighted_log_loss(y[test].tolist(), after_probability, test_weights)
        folds.append(
            {
                "fold": fold_index,
                "train_indices": train.tolist(),
                "test_indices": test.tolist(),
                "train_papers": sorted({papers[index] for index in train}),
                "test_papers": sorted({papers[index] for index in test}),
                "before_log_loss": before_loss,
                "after_log_loss": after_loss,
                "incremental_delta_ll": before_loss - after_loss,
                "test_weight_sum": float(test_weights.sum()),
            }
        )
    deltas = [fold["incremental_delta_ll"] for fold in folds]
    return {
        "status": split["status"],
        "k": split["k"],
        "delta_ll": float(np.mean(deltas)),
        "positive_folds": sum(delta > 0 for delta in deltas),
        "folds": folds,
        "n_rows": len(subset),
        "n_papers": len(set(papers)),
        "finding_ids": [row["finding_id"] for row in subset],
    }


def _permute_candidate(
    rows: Sequence[Mapping[str, Any]], *, seed: int, attempt: int
) -> list[dict[str, Any]]:
    by_paper: dict[str, set[Any]] = defaultdict(set)
    for row in rows:
        if row.get("__candidate__") is not None:
            by_paper[str(row["paper_id"])].add(row["__candidate__"])
    if any(len(values) > 1 for values in by_paper.values()):
        raise RemapContractError("candidate_summary_not_paper_constant")
    papers = sorted(by_paper)
    values = [next(iter(by_paper[paper])) for paper in papers]
    rng = np.random.default_rng(np.random.SeedSequence([seed, attempt]))
    shuffled = list(np.asarray(values, dtype=object)[rng.permutation(len(values))])
    by_new_paper = dict(zip(papers, shuffled, strict=True))
    output = []
    for row in rows:
        clone = dict(row)
        if clone.get("__candidate__") is not None:
            clone["__candidate__"] = by_new_paper[str(clone["paper_id"])]
        output.append(clone)
    return output


def candidate_permutation(
    rows: Sequence[Mapping[str, Any]],
    *,
    base_moderator_key: str,
    observed_delta: float,
    seed: int,
    max_folds: int,
    required_successes: int = 100,
    max_attempts: int = 125,
) -> dict[str, Any]:
    scores: list[float] = []
    failures: list[dict[str, Any]] = []
    for attempt in range(max_attempts):
        try:
            permuted = _permute_candidate(rows, seed=seed, attempt=attempt)
            result = incremental_cv(
                permuted,
                base_moderator_key=base_moderator_key,
                seed=seed,
                max_folds=max_folds,
            )
            if result["delta_ll"] is None:
                raise RemapContractError("candidate_permutation_undefined")
            scores.append(float(result["delta_ll"]))
        except Exception as exc:
            failures.append(
                {
                    "attempt_index": attempt,
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
        if len(scores) == required_successes:
            break
    complete = len(scores) == required_successes
    return {
        "status": "complete" if complete else "indeterminate",
        "required_successes": required_successes,
        "max_attempts": max_attempts,
        "attempt_count": len(scores) + len(failures),
        "success_count": len(scores),
        "p_value": (
            (1 + sum(score >= observed_delta for score in scores)) / (required_successes + 1)
            if complete
            else None
        ),
        "scores": scores,
        "guard_failures": failures,
    }


def candidate_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    base_moderator_key: str,
    seed: int,
    max_folds: int,
    n_bootstraps: int = 200,
) -> dict[str, Any]:
    draws: list[dict[str, Any]] = []
    positive = 0
    for draw in range(n_bootstraps):
        try:
            sample = paper_bootstrap_sample(rows, seed=seed, draw_index=draw)
            result = incremental_cv(
                sample,
                base_moderator_key=base_moderator_key,
                seed=seed,
                max_folds=max_folds,
            )
            gain = result.get("delta_ll")
            is_positive = gain is not None and gain > 0
            positive += is_positive
            draws.append(
                {
                    "draw_index": draw,
                    "status": "success",
                    "incremental_delta_ll": gain,
                    "positive": is_positive,
                }
            )
        except Exception as exc:
            draws.append(
                {
                    "draw_index": draw,
                    "status": "error",
                    "reason": type(exc).__name__,
                    "message": str(exc),
                    "positive": False,
                }
            )
    return {
        "status": "complete",
        "n_bootstraps": n_bootstraps,
        "positive_count": positive,
        "positive_fraction": positive / n_bootstraps,
        "draws": draws,
    }


def evaluate_remap_candidate(
    *,
    proposal: Mapping[str, Any],
    execution: Mapping[str, Any],
    primary_rows: Sequence[Mapping[str, Any]],
    frozen_headline: Mapping[str, Any],
    all_valid_rows: Sequence[Mapping[str, Any]] | None = None,
    seed: int = 20260815,
    max_folds: int = 5,
    overrides: RemapInferenceOverrides | None = None,
) -> dict[str, Any]:
    """Apply every §6.4 rule and return kept/discarded/indeterminate verbatim."""

    if frozen_headline.get("narrative_variant") != "A":
        raise RemapContractError("remap_analysis_requires_frozen_variant_a")
    if execution.get("proposal_sha256") != proposal.get("proposal_sha256"):
        raise RemapContractError("remap_execution_proposal_hash_mismatch")
    reconciliation = execution.get("reconciliation") or {}
    moderator = proposal["moderator"]
    base_name = frozen_headline.get("moderator", {}).get("name")
    if not isinstance(base_name, str):
        raise RemapContractError("frozen_headline_missing_moderator")
    base_key = f"mod__{base_name}"
    joined = _joined_rows(primary_rows, execution.get("side_table", []))
    all_papers = {row["paper_id"] for row in joined}
    nonnull_papers = {row["paper_id"] for row in joined if row.get("__candidate__") is not None}
    coverage = len(nonnull_papers) / len(all_papers) if all_papers else 0.0
    level_papers: dict[str, set[str]] = defaultdict(set)
    paper_values: dict[str, set[str]] = defaultdict(set)
    for row in joined:
        if row.get("__candidate__") is not None:
            level_papers[str(row["__candidate__"])].add(row["paper_id"])
            paper_values[str(row["paper_id"])].add(str(row["__candidate__"]))
    supported_levels = {
        level: len(papers) for level, papers in level_papers.items() if len(papers) >= 5
    }
    technical_valid = reconciliation.get("technical_valid") is True
    summary_is_paper_constant = all(len(values) == 1 for values in paper_values.values())
    exchangeability_valid = summary_is_paper_constant and (
        (moderator["kind"] == "paper_constant" and moderator["permutation"] == "paper")
        or (
            moderator["kind"] == "within_paper"
            and moderator["permutation"] == "paper_summary"
            and bool(moderator.get("paper_summary"))
        )
    )
    injected = overrides or RemapInferenceOverrides()
    if not technical_valid or not exchangeability_valid:
        cv = None
        permutation = None
        bootstrap = None
        sensitivity = None
    else:
        cv = (
            dict(injected.incremental_cv)
            if injected.incremental_cv is not None
            else incremental_cv(
                joined,
                base_moderator_key=base_key,
                seed=seed,
                max_folds=max_folds,
            )
        )
        if cv.get("delta_ll") is None:
            permutation = None
            bootstrap = None
        else:
            permutation = (
                dict(injected.permutation)
                if injected.permutation is not None
                else candidate_permutation(
                    joined,
                    base_moderator_key=base_key,
                    observed_delta=float(cv["delta_ll"]),
                    seed=seed,
                    max_folds=max_folds,
                )
            )
            bootstrap = (
                dict(injected.bootstrap)
                if injected.bootstrap is not None
                else candidate_bootstrap(
                    joined,
                    base_moderator_key=base_key,
                    seed=seed,
                    max_folds=max_folds,
                )
            )
        if injected.all_valid_sensitivity is not None:
            sensitivity = dict(injected.all_valid_sensitivity)
        else:
            valid_joined = _joined_rows(all_valid_rows or primary_rows, execution["side_table"])
            valid_cv = incremental_cv(
                valid_joined,
                base_moderator_key=base_key,
                seed=seed,
                max_folds=max_folds,
            )
            sensitivity = {
                "incremental_delta_ll": valid_cv.get("delta_ll"),
                "positive_gain": valid_cv.get("delta_ll") is not None and valid_cv["delta_ll"] > 0,
            }

    k = cv.get("k") if cv else None
    required_positive_folds = math.ceil(0.6 * k) if isinstance(k, int) else None
    rules = [
        {
            "name": "technical_join",
            "observed": reconciliation.get("join_fraction"),
            "threshold": ">=0.95; no duplicate, unknown, or invalid responses",
            "passed": technical_valid,
            "classification": "technical",
        },
        {
            "name": "non_null_paper_coverage",
            "observed": coverage,
            "threshold": ">=0.60",
            "passed": coverage >= 0.60,
            "classification": "numeric",
        },
        {
            "name": "level_support",
            "observed": supported_levels,
            "threshold": ">=2 levels with >=5 papers",
            "passed": len(supported_levels) >= 2,
            "classification": "numeric",
        },
        {
            "name": "feasible_k",
            "observed": k,
            "threshold": ">=3",
            "passed": isinstance(k, int) and k >= 3,
            "classification": "numeric",
        },
        {
            "name": "incremental_delta_ll",
            "observed": cv.get("delta_ll") if cv else None,
            "threshold": ">=0.02",
            "passed": bool(cv and cv.get("delta_ll") is not None and cv["delta_ll"] >= 0.02),
            "classification": "numeric",
        },
        {
            "name": "positive_folds",
            "observed": cv.get("positive_folds") if cv else None,
            "threshold": f">={required_positive_folds}",
            "passed": bool(
                cv
                and required_positive_folds is not None
                and cv.get("positive_folds") is not None
                and cv["positive_folds"] >= required_positive_folds
            ),
            "classification": "numeric",
        },
        {
            "name": "candidate_permutation",
            "observed": permutation.get("p_value") if permutation else None,
            "threshold": "100 successes within 125; add-one p<0.10",
            "passed": bool(
                permutation
                and permutation.get("status") == "complete"
                and permutation.get("success_count") == 100
                and permutation.get("p_value") is not None
                and permutation["p_value"] < 0.10
            ),
            "classification": "technical"
            if not permutation or permutation.get("status") != "complete"
            else "numeric",
        },
        {
            "name": "bootstrap_positive",
            "observed": bootstrap.get("positive_fraction") if bootstrap else None,
            "threshold": ">=0.60 of all 200",
            "passed": bool(
                bootstrap
                and bootstrap.get("n_bootstraps") == 200
                and bootstrap.get("positive_fraction") is not None
                and bootstrap["positive_fraction"] >= 0.60
            ),
            "classification": "technical"
            if not bootstrap or bootstrap.get("status") != "complete"
            else "numeric",
        },
        {
            "name": "all_valid_sensitivity",
            "observed": sensitivity.get("incremental_delta_ll") if sensitivity else None,
            "threshold": "positive gain",
            "passed": bool(sensitivity and sensitivity.get("positive_gain") is True),
            "classification": "technical" if sensitivity is None else "numeric",
        },
    ]
    if not exchangeability_valid:
        decision = "indeterminate"
        reason = "undefined_exchangeability"
    elif not technical_valid:
        decision = "indeterminate"
        reason = "invalid_echo_join"
    elif any(rule["classification"] == "technical" and not rule["passed"] for rule in rules):
        decision = "indeterminate"
        reason = "inference_incomplete_or_undefined"
    elif any(not rule["passed"] for rule in rules):
        decision = "discarded"
        reason = "numeric_rule_failed"
    else:
        decision = "kept_exploratory"
        reason = "all_exploratory_keep_rules_passed"
    return {
        "status": "complete",
        "decision": decision,
        "reason": reason,
        "proposal_sha256": proposal["proposal_sha256"],
        "frozen_headline_sha256": hash_canonical(frozen_headline),
        "moderator": moderator,
        "before_model": {"moderator": base_name, "frozen": True},
        "after_model": {"moderators": [base_name, moderator["name"]]},
        "same_subset": {
            "finding_ids": cv.get("finding_ids", []) if cv else [],
            "n_rows": cv.get("n_rows", 0) if cv else 0,
            "folds": cv.get("folds", []) if cv else [],
        },
        "reconciliation": reconciliation,
        "coverage_papers": coverage,
        "support_papers": {level: len(papers) for level, papers in level_papers.items()},
        "incremental_cv": cv,
        "permutation": permutation,
        "bootstrap": bootstrap,
        "all_valid_sensitivity": sensitivity,
        "rules": rules,
        "language_guard": (
            "cross-validated incremental gain, moderator proposed post hoc"
            if decision == "kept_exploratory"
            else "moderator proposed post hoc"
        ),
    }


def not_run_trace(reason: str) -> dict[str, Any]:
    allowed = {
        "g3_story_not_viable",
        "m4_selected_variant_b",
        "m4_incomplete",
        "human_approval_unavailable",
        "human_rejected",
    }
    if reason not in allowed:
        raise RemapContractError(f"invalid_not_run_reason:{reason}")
    return {"status": "not_run", "reason": reason}


__all__ = [
    "RemapContractError",
    "RemapInferenceOverrides",
    "approval_template",
    "candidate_bootstrap",
    "candidate_permutation",
    "evaluate_remap_candidate",
    "execute_approved_remap",
    "incremental_cv",
    "not_run_trace",
    "propose_remap",
    "reconcile_echo_responses",
    "select_proposal_pairs",
    "validate_approval",
]
