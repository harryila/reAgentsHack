"""Misspecified simulation stress test for budgeted adaptive verification.

The generator in this module is intentionally not the objective optimized by the
proposed policy.  Allocation sees noisy item-risk, disagreement, cost, and
leave-one-out influence signals.  Hidden outcomes separately include correlated
shared-cohort failures, mixed-sign extraction errors, nonlinear synthesis,
imperfect reviewers, inaccessible full text, and a combined shifted population.

This is a mechanism/adversarial stress test only.  It does not estimate real-world
claim error, human verification cost, or scientific validity.
"""

from __future__ import annotations

import hashlib
import math
import platform
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from literature_multiverse.budgeted_verification import (
    AllocationPolicy,
    AuditCandidate,
    ClaimModel,
    ProbabilityBasis,
    ScenarioKind,
    rank_candidates,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE

STRESS_STUDY_VERSION = "adaptive-audit-misspecified-stress-v1"
QUESTION_RECEIPT_VERSION = "adaptive-stress-question-receipt-v1"
SUMMARY_VERSION = "adaptive-stress-summary-v1"
ARTIFACT_VERSION = "adaptive-stress-artifact-v1"

FIXED_BUDGET_POLICIES = (
    "adaptive_static",
    "adaptive_sequential",
    "random",
    "risk_only",
    "risk_per_cost",
    "disagreement_only",
    "disagreement_per_cost",
    "influence_only",
    "influence_per_cost",
    "fixed_count",
)
REFERENCE_POLICIES = ("no_audit", "audit_all")
ALL_POLICIES = FIXED_BUDGET_POLICIES + REFERENCE_POLICIES

DEFAULT_SCENARIOS = (
    "iid_control",
    "correlated_shared_cohort",
    "miscalibrated_scores",
    "nonlinear_interactions",
    "reviewer_mistakes",
    "missing_full_text",
    "combined_domain_shift",
)

# These profiles are copied into and sealed by every frozen run configuration.
# Values deliberately vary which signal is useful; no profile plants outcomes as
# an algebraic function of risk * influence / cost.
SCENARIO_PROFILES: dict[str, dict[str, float | str]] = {
    "iid_control": {
        "population_role": "source_like_control",
        "base_error_rate": 0.16,
        "cohort_error_correlation": 0.08,
        "error_scale": 0.12,
        "shared_shift_scale": 0.04,
        "missing_full_text_rate": 0.02,
        "missing_error_logit_boost": 0.35,
        "risk_logit_slope": 1.00,
        "risk_logit_bias": 0.00,
        "risk_noise_sd": 0.35,
        "disagreement_logit_slope": 0.55,
        "disagreement_noise_sd": 0.65,
        "interaction_strength": 0.00,
        "reviewer_accuracy": 0.98,
        "reviewer_error_scale": 0.05,
    },
    "correlated_shared_cohort": {
        "population_role": "source_like_stress",
        "base_error_rate": 0.20,
        "cohort_error_correlation": 0.78,
        "error_scale": 0.11,
        "shared_shift_scale": 0.13,
        "missing_full_text_rate": 0.05,
        "missing_error_logit_boost": 0.60,
        "risk_logit_slope": 0.80,
        "risk_logit_bias": -0.10,
        "risk_noise_sd": 0.55,
        "disagreement_logit_slope": 0.70,
        "disagreement_noise_sd": 0.60,
        "interaction_strength": 1.25,
        "reviewer_accuracy": 0.95,
        "reviewer_error_scale": 0.07,
    },
    "miscalibrated_scores": {
        "population_role": "sensor_shift_stress",
        "base_error_rate": 0.22,
        "cohort_error_correlation": 0.38,
        "error_scale": 0.14,
        "shared_shift_scale": 0.09,
        "missing_full_text_rate": 0.08,
        "missing_error_logit_boost": 0.75,
        "risk_logit_slope": -0.55,
        "risk_logit_bias": -0.15,
        "risk_noise_sd": 0.85,
        "disagreement_logit_slope": 0.65,
        "disagreement_noise_sd": 0.70,
        "interaction_strength": 1.00,
        "reviewer_accuracy": 0.94,
        "reviewer_error_scale": 0.08,
    },
    "nonlinear_interactions": {
        "population_role": "synthesis_shift_stress",
        "base_error_rate": 0.20,
        "cohort_error_correlation": 0.35,
        "error_scale": 0.13,
        "shared_shift_scale": 0.08,
        "missing_full_text_rate": 0.07,
        "missing_error_logit_boost": 0.60,
        "risk_logit_slope": 0.75,
        "risk_logit_bias": -0.05,
        "risk_noise_sd": 0.60,
        "disagreement_logit_slope": 0.60,
        "disagreement_noise_sd": 0.70,
        "interaction_strength": 6.00,
        "reviewer_accuracy": 0.94,
        "reviewer_error_scale": 0.08,
    },
    "reviewer_mistakes": {
        "population_role": "review_process_stress",
        "base_error_rate": 0.21,
        "cohort_error_correlation": 0.42,
        "error_scale": 0.13,
        "shared_shift_scale": 0.09,
        "missing_full_text_rate": 0.08,
        "missing_error_logit_boost": 0.65,
        "risk_logit_slope": 0.75,
        "risk_logit_bias": -0.05,
        "risk_noise_sd": 0.60,
        "disagreement_logit_slope": 0.65,
        "disagreement_noise_sd": 0.65,
        "interaction_strength": 1.50,
        "reviewer_accuracy": 0.76,
        "reviewer_error_scale": 0.16,
    },
    "missing_full_text": {
        "population_role": "access_shift_stress",
        "base_error_rate": 0.20,
        "cohort_error_correlation": 0.40,
        "error_scale": 0.13,
        "shared_shift_scale": 0.08,
        "missing_full_text_rate": 0.36,
        "missing_error_logit_boost": 1.20,
        "risk_logit_slope": 0.75,
        "risk_logit_bias": -0.10,
        "risk_noise_sd": 0.65,
        "disagreement_logit_slope": 0.60,
        "disagreement_noise_sd": 0.70,
        "interaction_strength": 1.50,
        "reviewer_accuracy": 0.93,
        "reviewer_error_scale": 0.09,
    },
    "combined_domain_shift": {
        "population_role": "heldout_combined_target_shift",
        "base_error_rate": 0.29,
        "cohort_error_correlation": 0.82,
        "error_scale": 0.18,
        "shared_shift_scale": 0.17,
        "missing_full_text_rate": 0.27,
        "missing_error_logit_boost": 1.30,
        "risk_logit_slope": -0.70,
        "risk_logit_bias": -0.25,
        "risk_noise_sd": 1.00,
        "disagreement_logit_slope": -0.15,
        "disagreement_noise_sd": 0.95,
        "interaction_strength": 6.50,
        "reviewer_accuracy": 0.80,
        "reviewer_error_scale": 0.18,
    },
}


class AdaptiveStressStudyError(ValueError):
    """A frozen study or receipt violated its deterministic contract."""


@dataclass(frozen=True, slots=True)
class StressVisibleItem:
    """Information available to an allocation policy before adjudication."""

    item_id: str
    cohort_id: str
    source_rank: int
    observed_contribution: float
    risk_score: float
    disagreement_score: float
    verification_cost_minutes: float
    full_text_available: bool


@dataclass(frozen=True, slots=True)
class StressOracleItem:
    """Hidden simulation outcome, passed only to the audit/evaluation boundary."""

    item_id: str
    true_contribution: float
    is_extraction_error: bool
    reviewed_contribution: float
    reviewer_is_correct: bool
    shared_error_component: bool


@dataclass(frozen=True, slots=True)
class StressQuestion:
    question_id: str
    scenario: str
    generator_seed: int
    visible_items: tuple[StressVisibleItem, ...]
    oracle_items: tuple[StressOracleItem, ...]
    intercept: float
    interaction_strength: float
    true_decision: bool


def _derived_seed(*parts: object) -> int:
    identity = "adaptive-stress-v1\0" + "\0".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _logit(value: float) -> float:
    clipped = min(max(float(value), 1e-8), 1.0 - 1e-8)
    return math.log(clipped / (1.0 - clipped))


def _budget_arm_id(budget: float) -> str:
    return f"minutes-{format(float(budget), '.12g')}"


def _validate_probability_grid(values: Sequence[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if (
        not result
        or result != sorted(set(result))
        or any(not math.isfinite(value) or not 0 <= value <= 1 for value in result)
    ):
        raise AdaptiveStressStudyError(f"{name}_must_be_sorted_unique_probabilities")
    return result


def freeze_stress_study_config(
    *,
    seed: int = 20260827,
    questions_per_scenario: int = 160,
    items_per_question: int = 24,
    budgets_minutes: Sequence[float] = (15.0, 30.0, 60.0),
    release_risk_thresholds: Sequence[float] = (
        0.01,
        0.025,
        0.05,
        0.10,
        0.20,
        0.40,
        1.00,
    ),
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    fixed_count: int = 5,
    release_monte_carlo_draws: int = 192,
    bootstrap_draws: int = 2_000,
    primary_budget_minutes: float = 30.0,
    primary_release_risk_threshold: float = 0.10,
    source_files_sha256: Mapping[str, str] | None = None,
    runtime_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Freeze every generator, policy, release, and uncertainty choice before results."""

    if not isinstance(seed, int) or seed < 0:
        raise AdaptiveStressStudyError("seed_must_be_nonnegative_integer")
    if questions_per_scenario < 1:
        raise AdaptiveStressStudyError("questions_per_scenario_must_be_positive")
    if items_per_question < 12:
        raise AdaptiveStressStudyError("items_per_question_must_be_at_least_twelve")
    budgets = [float(value) for value in budgets_minutes]
    if (
        not budgets
        or budgets != sorted(set(budgets))
        or any(not math.isfinite(value) or value <= 0 for value in budgets)
    ):
        raise AdaptiveStressStudyError("budgets_must_be_sorted_unique_positive")
    thresholds = _validate_probability_grid(
        release_risk_thresholds, "release_risk_thresholds"
    )
    scenario_names = list(scenarios)
    if (
        not scenario_names
        or scenario_names != list(dict.fromkeys(scenario_names))
        or any(name not in SCENARIO_PROFILES for name in scenario_names)
    ):
        raise AdaptiveStressStudyError("scenarios_invalid_or_duplicate")
    if fixed_count < 1 or fixed_count > items_per_question:
        raise AdaptiveStressStudyError("fixed_count_invalid")
    if release_monte_carlo_draws < 32:
        raise AdaptiveStressStudyError("release_monte_carlo_draws_too_small")
    if bootstrap_draws < 100:
        raise AdaptiveStressStudyError("bootstrap_draws_too_small")
    primary_budget = float(primary_budget_minutes)
    primary_threshold = float(primary_release_risk_threshold)
    if primary_budget not in budgets:
        raise AdaptiveStressStudyError("primary_budget_not_in_frozen_budget_family")
    if primary_threshold not in thresholds:
        raise AdaptiveStressStudyError("primary_threshold_not_in_frozen_threshold_family")
    source_hashes = dict(sorted((source_files_sha256 or {}).items()))
    if any(not path or not SHA256_RE.fullmatch(value) for path, value in source_hashes.items()):
        raise AdaptiveStressStudyError("source_file_hash_invalid")
    runtime = dict(
        sorted(
            (
                runtime_identity
                or {
                    "python_implementation": platform.python_implementation(),
                    "python_version": platform.python_version(),
                    "numpy_version": np.__version__,
                    "numpy_bit_generator": "PCG64",
                }
            ).items()
        )
    )
    if not runtime or any(not key or not value for key, value in runtime.items()):
        raise AdaptiveStressStudyError("runtime_identity_invalid")

    payload: dict[str, Any] = {
        "study_version": STRESS_STUDY_VERSION,
        "seed": seed,
        "questions_per_scenario": questions_per_scenario,
        "items_per_question": items_per_question,
        "budgets_minutes": budgets,
        "release_risk_thresholds": thresholds,
        "scenarios": scenario_names,
        "scenario_profiles": {
            name: dict(SCENARIO_PROFILES[name]) for name in scenario_names
        },
        "fixed_budget_policies": list(FIXED_BUDGET_POLICIES),
        "reference_policies": list(REFERENCE_POLICIES),
        "fixed_count": fixed_count,
        "release_monte_carlo_draws": release_monte_carlo_draws,
        "bootstrap_draws": bootstrap_draws,
        "primary_budget_minutes": primary_budget,
        "primary_release_risk_threshold": primary_threshold,
        "cost_unit": "simulated_person_minutes",
        "allocation_contract": {
            "adaptive_static": (
                "production rank_candidates: frozen risk_score * exact leave-one-out "
                "probability influence / cost"
            ),
            "adaptive_sequential": (
                "production rank_candidates after rerunning nonlinear synthesis and "
                "updating same-cohort risk from each observed reviewer correction"
            ),
            "oracle_labels_policy_visible": False,
            "budgets_use_realized_simulated_person_minutes": True,
            "fixed_count_order": "frozen_source_rank_first_five_subject_to_budget",
            "audit_all_role": "unmatched_cost_exhaustive_review_reference",
        },
        "release_contract": {
            "score": "threshold-blind joint perturbation flip frequency",
            "perturbations": "mixed-sign correlated unresolved-item sensitivity draws",
            "ranking_score_reused_as_release_score": False,
            "threshold_family_frozen_before_generation": True,
        },
        "source_files_sha256": source_hashes,
        "runtime_identity": runtime,
    }
    return {**payload, "config_sha256": hash_canonical(payload)}


def validate_stress_study_config(config: Mapping[str, Any]) -> None:
    full_config = dict(config)
    config_dict = dict(full_config)
    observed_hash = config_dict.pop("config_sha256", None)
    if not isinstance(observed_hash, str) or not SHA256_RE.fullmatch(observed_hash):
        raise AdaptiveStressStudyError("config_hash_missing_or_invalid")
    if hash_canonical(config_dict) != observed_hash:
        raise AdaptiveStressStudyError("config_hash_mismatch")
    if config_dict.get("study_version") != STRESS_STUDY_VERSION:
        raise AdaptiveStressStudyError("study_version_mismatch")
    if tuple(config_dict.get("fixed_budget_policies", ())) != FIXED_BUDGET_POLICIES:
        raise AdaptiveStressStudyError("fixed_budget_policy_family_mismatch")
    if tuple(config_dict.get("reference_policies", ())) != REFERENCE_POLICIES:
        raise AdaptiveStressStudyError("reference_policy_family_mismatch")
    scenarios = config_dict.get("scenarios")
    profiles = config_dict.get("scenario_profiles")
    if not isinstance(scenarios, list) or not isinstance(profiles, Mapping):
        raise AdaptiveStressStudyError("scenario_contract_missing")
    if set(scenarios) != set(profiles):
        raise AdaptiveStressStudyError("scenario_profile_membership_mismatch")
    _validate_probability_grid(
        config_dict.get("release_risk_thresholds", ()), "release_risk_thresholds"
    )
    try:
        expected = freeze_stress_study_config(
            seed=config_dict["seed"],
            questions_per_scenario=config_dict["questions_per_scenario"],
            items_per_question=config_dict["items_per_question"],
            budgets_minutes=config_dict["budgets_minutes"],
            release_risk_thresholds=config_dict["release_risk_thresholds"],
            scenarios=config_dict["scenarios"],
            fixed_count=config_dict["fixed_count"],
            release_monte_carlo_draws=config_dict["release_monte_carlo_draws"],
            bootstrap_draws=config_dict["bootstrap_draws"],
            primary_budget_minutes=config_dict["primary_budget_minutes"],
            primary_release_risk_threshold=config_dict[
                "primary_release_risk_threshold"
            ],
            source_files_sha256=config_dict["source_files_sha256"],
            runtime_identity=config_dict["runtime_identity"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AdaptiveStressStudyError("config_semantics_invalid") from exc
    if expected != full_config:
        raise AdaptiveStressStudyError("config_not_exactly_code_owned_frozen_contract")


def _synthesis_scores(
    question: StressQuestion,
    contribution_matrix: np.ndarray,
) -> np.ndarray:
    """Rerun the actual nonlinear synthetic claim model for one or many states."""

    matrix = np.asarray(contribution_matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    score = question.intercept + np.sum(matrix, axis=1)
    cohort_members: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(question.visible_items):
        cohort_members[item.cohort_id].append(index)
    if question.interaction_strength:
        interaction = np.zeros(matrix.shape[0], dtype=float)
        for indices in cohort_members.values():
            for left_offset, left in enumerate(indices):
                for right in indices[left_offset + 1 :]:
                    interaction += matrix[:, left] * matrix[:, right]
        score += question.interaction_strength * interaction
    return score


def _generate_question(
    config: Mapping[str, Any], *, scenario: str, question_index: int
) -> StressQuestion:
    validate_stress_study_config(config)
    profile = dict(config["scenario_profiles"][scenario])
    generator_seed = _derived_seed(config["seed"], scenario, question_index)
    rng = np.random.Generator(np.random.PCG64(generator_seed))
    item_count = int(config["items_per_question"])
    cohort_count = max(4, item_count // 3)
    cohort_indices = np.arange(item_count, dtype=int) % cohort_count
    rng.shuffle(cohort_indices)

    true_contributions = rng.normal(0.0, 0.075, item_count)
    cohort_difficulty = rng.normal(0.0, 0.80, cohort_count)
    available = rng.random(item_count) >= float(profile["missing_full_text_rate"])
    base_logit = _logit(float(profile["base_error_rate"]))
    true_error_probability = np.empty(item_count, dtype=float)
    for index in range(item_count):
        logit_probability = (
            base_logit
            + 0.70 * cohort_difficulty[cohort_indices[index]]
            + float(profile["missing_error_logit_boost"]) * (not available[index])
            + 1.15 * abs(true_contributions[index])
        )
        true_error_probability[index] = _sigmoid(logit_probability)

    correlation = float(profile["cohort_error_correlation"])
    shared_uniform = rng.random(cohort_count)
    independent_uniform = rng.random(item_count)
    shared_events = np.zeros(item_count, dtype=bool)
    extraction_errors = np.zeros(item_count, dtype=bool)
    for index, probability in enumerate(true_error_probability):
        shared_probability = correlation * probability
        shared = shared_uniform[cohort_indices[index]] < shared_probability
        independent_probability = (
            (probability - shared_probability) / (1.0 - shared_probability)
            if shared_probability < 1.0
            else 0.0
        )
        independent = independent_uniform[index] < independent_probability
        shared_events[index] = shared
        extraction_errors[index] = shared or independent

    cohort_sign = rng.choice(np.asarray([-1.0, 1.0]), size=cohort_count)
    cohort_shift = (
        cohort_sign
        * rng.lognormal(-2.35, 0.55, cohort_count)
        * float(profile["shared_shift_scale"])
        / 0.095
    )
    item_sign = rng.choice(np.asarray([-1.0, 1.0]), size=item_count)
    # Only 62% of item signs follow the cohort sign.  Errors therefore remain
    # mixed-direction even under strong shared-cohort correlation.
    follow_cohort = rng.random(item_count) < 0.62
    item_sign = np.where(follow_cohort, cohort_sign[cohort_indices], item_sign)
    item_magnitude = rng.lognormal(-2.25, 0.65, item_count)
    item_magnitude *= float(profile["error_scale"]) / 0.13
    distortion = np.zeros(item_count, dtype=float)
    for index in range(item_count):
        if extraction_errors[index]:
            distortion[index] = item_sign[index] * item_magnitude[index]
            if shared_events[index]:
                distortion[index] += cohort_shift[cohort_indices[index]]
    observed = true_contributions + distortion

    risk_noise = rng.normal(0.0, float(profile["risk_noise_sd"]), item_count)
    risk_scores = np.asarray(
        [
            _sigmoid(
                float(profile["risk_logit_bias"])
                + float(profile["risk_logit_slope"]) * _logit(probability)
                + noise
            )
            for probability, noise in zip(
                true_error_probability, risk_noise, strict=True
            )
        ]
    )
    disagreement_noise = rng.normal(
        0.0, float(profile["disagreement_noise_sd"]), item_count
    )
    disagreement_scores = np.asarray(
        [
            _sigmoid(
                float(profile["disagreement_logit_slope"]) * _logit(probability)
                + noise
            )
            for probability, noise in zip(
                true_error_probability, disagreement_noise, strict=True
            )
        ]
    )
    costs = np.clip(rng.lognormal(math.log(4.8), 0.38, item_count), 1.5, 12.0)

    reviewer_correct = rng.random(item_count) < float(profile["reviewer_accuracy"])
    reviewed = true_contributions.copy()
    reviewer_noise = rng.normal(
        0.0, float(profile["reviewer_error_scale"]), item_count
    )
    for index in range(item_count):
        if not reviewer_correct[index]:
            # A mistaken adjudication can retain or amplify the observed distortion;
            # it is not silently replaced with oracle truth.
            retained = 0.65 * observed[index] + 0.35 * true_contributions[index]
            reviewed[index] = retained + reviewer_noise[index]

    interaction_strength = float(profile["interaction_strength"])
    # Random interaction sign prevents a single monotone correction direction.
    if interaction_strength and rng.random() < 0.35:
        interaction_strength *= -1.0
    question_id = f"{scenario}-q-{question_index:05d}"
    visible_items = tuple(
        StressVisibleItem(
            item_id=f"{question_id}-item-{index:03d}",
            cohort_id=f"{question_id}-cohort-{cohort_indices[index]:03d}",
            source_rank=index,
            observed_contribution=float(observed[index]),
            risk_score=float(risk_scores[index]),
            disagreement_score=float(disagreement_scores[index]),
            verification_cost_minutes=float(costs[index]),
            full_text_available=bool(available[index]),
        )
        for index in range(item_count)
    )
    oracle_items = tuple(
        StressOracleItem(
            item_id=visible_items[index].item_id,
            true_contribution=float(true_contributions[index]),
            is_extraction_error=bool(extraction_errors[index]),
            reviewed_contribution=float(reviewed[index]),
            reviewer_is_correct=bool(reviewer_correct[index]),
            shared_error_component=bool(shared_events[index]),
        )
        for index in range(item_count)
    )

    temporary = StressQuestion(
        question_id=question_id,
        scenario=scenario,
        generator_seed=generator_seed,
        visible_items=visible_items,
        oracle_items=oracle_items,
        intercept=0.0,
        interaction_strength=interaction_strength,
        true_decision=False,
    )
    target_true_score = float(rng.normal(0.0, 0.28))
    raw_true_score = float(_synthesis_scores(temporary, true_contributions)[0])
    intercept = target_true_score - raw_true_score
    return StressQuestion(
        question_id=question_id,
        scenario=scenario,
        generator_seed=generator_seed,
        visible_items=visible_items,
        oracle_items=oracle_items,
        intercept=intercept,
        interaction_strength=interaction_strength,
        true_decision=target_true_score >= 0.0,
    )


def _production_ranking(
    *,
    question: StressQuestion,
    policy: str,
    contributions: np.ndarray,
    current_risks: np.ndarray,
    candidate_indices: Sequence[int],
    seed: int,
) -> list[tuple[float, str, int]]:
    """Use the production ranker with direct nonlinear synthesis rerun scores."""

    if not candidate_indices:
        return []
    policy_mapping = {
        "adaptive_static": AllocationPolicy.EXPECTED_CLAIM_LOSS_PER_COST,
        "adaptive_sequential": AllocationPolicy.EXPECTED_CLAIM_LOSS_PER_COST,
        "random": AllocationPolicy.RANDOM,
        "risk_only": AllocationPolicy.RISK_ONLY,
        "risk_per_cost": AllocationPolicy.RISK_PER_COST,
        "disagreement_only": AllocationPolicy.DISAGREEMENT,
        "disagreement_per_cost": AllocationPolicy.DISAGREEMENT,
        "influence_only": AllocationPolicy.INFLUENCE_ONLY,
        "influence_per_cost": AllocationPolicy.INFLUENCE_PER_COST,
    }
    try:
        production_policy = policy_mapping[policy]
    except KeyError as exc:
        raise AdaptiveStressStudyError(f"unknown_production_policy:{policy}") from exc
    baseline_score = float(_synthesis_scores(question, contributions)[0])
    baseline_probability = _sigmoid(baseline_score)
    baseline_decision = baseline_score >= 0.0
    candidates: list[AuditCandidate] = []
    index_by_id: dict[str, int] = {}
    for index in candidate_indices:
        item = question.visible_items[index]
        counterfactual = contributions.copy()
        counterfactual[index] = 0.0
        counterfactual_score = float(_synthesis_scores(question, counterfactual)[0])
        candidate = AuditCandidate(
            item_id=item.item_id,
            baseline_contribution=float(contributions[index]),
            counterfactual_contribution=0.0,
            error_probability=float(current_risks[index]),
            probability_basis=ProbabilityBasis.PLANTED_SIMULATION,
            probability_source="synthetic_noisy_policy_sensor",
            verification_cost=item.verification_cost_minutes,
            cost_unit="simulated_person_minutes",
            disagreement_score=item.disagreement_score,
            scenario_kind=ScenarioKind.LEAVE_ONE_OUT,
            scenario_source="exact_nonlinear_leave_one_out_synthesis_rerun",
            baseline_decision_score=baseline_probability,
            counterfactual_decision_score=_sigmoid(counterfactual_score),
            decision_score_source="exact_synthetic_nonlinear_synthesis_rerun",
            baseline_decision=baseline_decision,
            counterfactual_decision=counterfactual_score >= 0.0,
        )
        candidates.append(candidate)
        index_by_id[item.item_id] = index
    ranking = rank_candidates(
        candidates,
        ClaimModel(claim_id=question.question_id, intercept=0.0),
        production_policy,
        seed=seed,
    )
    if policy == "disagreement_per_cost":
        return sorted(
            (
                (
                    record.disagreement_score / record.verification_cost,
                    record.item_id,
                    index_by_id[record.item_id],
                )
                for record in ranking
            ),
            key=lambda row: (-row[0], row[1]),
        )
    return [
        (record.priority, record.item_id, index_by_id[record.item_id])
        for record in ranking
    ]


def _audit_item(
    *,
    index: int,
    question: StressQuestion,
    contributions: np.ndarray,
    current_risks: np.ndarray,
    resolved: set[int],
    sequential_update: bool,
) -> None:
    oracle = question.oracle_items[index]
    old_value = float(contributions[index])
    contributions[index] = oracle.reviewed_contribution
    resolved.add(index)
    current_risks[index] = 0.0
    if not sequential_update:
        return
    observed_change = abs(float(contributions[index]) - old_value)
    scale = min(observed_change / 0.18, 1.5)
    cohort_id = question.visible_items[index].cohort_id
    for neighbor, item in enumerate(question.visible_items):
        if neighbor in resolved or item.cohort_id != cohort_id:
            continue
        if scale >= 0.20:
            current_risks[neighbor] = min(
                1.0, current_risks[neighbor] * (1.0 + 1.25 * scale)
            )
        else:
            current_risks[neighbor] *= 0.72


def _estimate_policy_visible_flip_risk(
    *,
    question: StressQuestion,
    contributions: np.ndarray,
    current_risks: np.ndarray,
    resolved: set[int],
    draws: int,
) -> float:
    unresolved = [index for index in range(len(contributions)) if index not in resolved]
    if not unresolved:
        return 0.0
    baseline_decision = float(_synthesis_scores(question, contributions)[0]) >= 0.0
    rng = np.random.Generator(
        np.random.PCG64(_derived_seed(question.generator_seed, "release-sensitivity"))
    )
    item_count = len(contributions)
    item_uniform = rng.random((draws, item_count))
    magnitude = rng.uniform(0.35, 1.55, (draws, item_count))
    direction_flip = rng.random((draws, item_count)) < 0.22
    additive_noise = rng.normal(0.0, 0.025, (draws, item_count))
    cohort_ids = sorted({item.cohort_id for item in question.visible_items})
    cohort_uniform = rng.random((draws, len(cohort_ids)))
    cohort_lookup = {cohort_id: index for index, cohort_id in enumerate(cohort_ids)}
    perturbed = np.repeat(contributions[None, :], draws, axis=0)
    for index in unresolved:
        item = question.visible_items[index]
        risk = float(current_risks[index])
        if not item.full_text_available:
            risk = min(1.0, risk + 0.20)
        cohort_members = [
            neighbor
            for neighbor, candidate in enumerate(question.visible_items)
            if candidate.cohort_id == item.cohort_id and neighbor not in resolved
        ]
        cohort_risk = float(np.mean(current_risks[cohort_members]))
        shared_trigger = (
            cohort_uniform[:, cohort_lookup[item.cohort_id]] < 0.28 * cohort_risk
        )
        independent_trigger = item_uniform[:, index] < 0.76 * risk
        trigger = shared_trigger | independent_trigger
        shrink_delta = -0.70 * contributions[index] * magnitude[:, index]
        shrink_delta = np.where(direction_flip[:, index], -shrink_delta, shrink_delta)
        delta = shrink_delta + additive_noise[:, index]
        perturbed[:, index] = np.where(
            trigger, contributions[index] + delta, contributions[index]
        )
    decisions = _synthesis_scores(question, perturbed) >= 0.0
    return float(np.mean(decisions != baseline_decision))


def _evaluate_policy_state(
    *,
    question: StressQuestion,
    policy: str,
    budget_minutes: float | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    item_count = len(question.visible_items)
    contributions = np.asarray(
        [item.observed_contribution for item in question.visible_items], dtype=float
    )
    initial_contributions = contributions.copy()
    current_risks = np.asarray(
        [item.risk_score for item in question.visible_items], dtype=float
    )
    resolved: set[int] = set()
    selected: list[int] = []
    spent = 0.0

    if policy == "no_audit":
        pass
    elif policy == "audit_all":
        for index, item in sorted(
            enumerate(question.visible_items), key=lambda pair: pair[1].source_rank
        ):
            if not item.full_text_available:
                continue
            selected.append(index)
            spent += item.verification_cost_minutes
            _audit_item(
                index=index,
                question=question,
                contributions=contributions,
                current_risks=current_risks,
                resolved=resolved,
                sequential_update=False,
            )
    elif policy == "adaptive_sequential":
        if budget_minutes is None:
            raise AdaptiveStressStudyError("sequential_policy_requires_budget")
        while True:
            feasible = [
                index
                for index, item in enumerate(question.visible_items)
                if index not in resolved
                and item.full_text_available
                and spent + item.verification_cost_minutes <= budget_minutes + 1e-9
            ]
            if not feasible:
                break
            priorities = _production_ranking(
                question=question,
                policy=policy,
                contributions=contributions,
                current_risks=current_risks,
                candidate_indices=feasible,
                seed=_derived_seed(config["seed"], question.question_id, len(selected)),
            )
            _, _, selected_index = priorities[0]
            selected.append(selected_index)
            spent += question.visible_items[selected_index].verification_cost_minutes
            _audit_item(
                index=selected_index,
                question=question,
                contributions=contributions,
                current_risks=current_risks,
                resolved=resolved,
                sequential_update=True,
            )
    else:
        if budget_minutes is None:
            raise AdaptiveStressStudyError("fixed_policy_requires_budget")
        available_indices = [
            index
            for index, item in enumerate(question.visible_items)
            if item.full_text_available
        ]
        if policy == "fixed_count":
            ranked = [
                (-float(question.visible_items[index].source_rank), item.item_id, index)
                for index, item in enumerate(question.visible_items)
                if item.full_text_available
            ]
            ranked.sort(key=lambda row: (-row[0], row[1]))
        else:
            ranked = _production_ranking(
                question=question,
                policy=policy,
                contributions=contributions,
                current_risks=current_risks,
                candidate_indices=available_indices,
                seed=_derived_seed(config["seed"], question.question_id, policy),
            )
        selection_limit = int(config["fixed_count"]) if policy == "fixed_count" else None
        for _, _, index in ranked:
            if selection_limit is not None and len(selected) >= selection_limit:
                break
            cost = question.visible_items[index].verification_cost_minutes
            if spent + cost > budget_minutes + 1e-9:
                continue
            selected.append(index)
            spent += cost
            _audit_item(
                index=index,
                question=question,
                contributions=contributions,
                current_risks=current_risks,
                resolved=resolved,
                sequential_update=False,
            )

    baseline_score = float(_synthesis_scores(question, initial_contributions)[0])
    current_score = float(_synthesis_scores(question, contributions)[0])
    current_decision = current_score >= 0.0
    predicted_flip_risk = _estimate_policy_visible_flip_risk(
        question=question,
        contributions=contributions,
        current_risks=current_risks,
        resolved=resolved,
        draws=int(config["release_monte_carlo_draws"]),
    )
    arm_id = (
        policy
        if policy in REFERENCE_POLICIES
        else f"{policy}__{_budget_arm_id(float(budget_minutes))}"
    )
    return {
        "policy": policy,
        "arm_id": arm_id,
        "nominal_budget_minutes": (
            None if budget_minutes is None else float(budget_minutes)
        ),
        "budget_role": (
            "zero_cost_reference"
            if policy == "no_audit"
            else "unmatched_exhaustive_review_reference"
            if policy == "audit_all"
            else "matched_fixed_person_minutes"
        ),
        "spent_person_minutes": float(spent),
        "selected_item_ids": [question.visible_items[index].item_id for index in selected],
        "selected_count": len(selected),
        "unresolved_count": item_count - len(resolved),
        "unresolved_missing_full_text_count": sum(
            not item.full_text_available and index not in resolved
            for index, item in enumerate(question.visible_items)
        ),
        "baseline_decision": baseline_score >= 0.0,
        "current_decision": current_decision,
        "true_decision": question.true_decision,
        "claim_decision_error": current_decision != question.true_decision,
        "baseline_synthesis_score": baseline_score,
        "current_synthesis_score": current_score,
        "policy_visible_predicted_flip_risk": predicted_flip_risk,
    }


def _question_receipt(config: Mapping[str, Any], question: StressQuestion) -> dict[str, Any]:
    fixed_evaluations = [
        _evaluate_policy_state(
            question=question,
            policy=policy,
            budget_minutes=float(budget),
            config=config,
        )
        for policy in FIXED_BUDGET_POLICIES
        for budget in config["budgets_minutes"]
    ]
    reference_evaluations = [
        _evaluate_policy_state(
            question=question,
            policy="no_audit",
            budget_minutes=0.0,
            config=config,
        ),
        _evaluate_policy_state(
            question=question,
            policy="audit_all",
            budget_minutes=None,
            config=config,
        ),
    ]
    oracle_by_id = {item.item_id: item for item in question.oracle_items}
    payload: dict[str, Any] = {
        "receipt_version": QUESTION_RECEIPT_VERSION,
        "question_id": question.question_id,
        "scenario": question.scenario,
        "generator_seed": question.generator_seed,
        "config_sha256": config["config_sha256"],
        "generator_diagnostics": {
            "items": len(question.visible_items),
            "extraction_errors": sum(
                item.is_extraction_error for item in question.oracle_items
            ),
            "shared_component_errors": sum(
                item.shared_error_component for item in question.oracle_items
            ),
            "missing_full_text": sum(
                not item.full_text_available for item in question.visible_items
            ),
            "reviewer_mistakes_among_accessible": sum(
                item.full_text_available
                and not oracle_by_id[item.item_id].reviewer_is_correct
                for item in question.visible_items
            ),
            "interaction_strength": question.interaction_strength,
        },
        "evaluations": fixed_evaluations + reference_evaluations,
    }
    return {**payload, "receipt_sha256": hash_canonical(payload)}


def validate_stress_question_receipt(
    receipt: Mapping[str, Any], *, expected_config_sha256: str | None = None
) -> None:
    payload = dict(receipt)
    observed_hash = payload.pop("receipt_sha256", None)
    if not isinstance(observed_hash, str) or not SHA256_RE.fullmatch(observed_hash):
        raise AdaptiveStressStudyError("question_receipt_hash_missing_or_invalid")
    if hash_canonical(payload) != observed_hash:
        raise AdaptiveStressStudyError("question_receipt_hash_mismatch")
    if payload.get("receipt_version") != QUESTION_RECEIPT_VERSION:
        raise AdaptiveStressStudyError("question_receipt_version_mismatch")
    if (
        expected_config_sha256 is not None
        and payload.get("config_sha256") != expected_config_sha256
    ):
        raise AdaptiveStressStudyError("question_receipt_config_mismatch")
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list):
        raise AdaptiveStressStudyError("question_receipt_evaluations_missing")
    keys = [(row.get("policy"), row.get("arm_id")) for row in evaluations]
    if len(keys) != len(set(keys)):
        raise AdaptiveStressStudyError("question_receipt_duplicate_policy_arm")


def generate_stress_question_receipts(
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Generate complete independent-question evaluation receipts."""

    validate_stress_study_config(config)
    receipts: list[dict[str, Any]] = []
    for scenario in config["scenarios"]:
        for question_index in range(int(config["questions_per_scenario"])):
            question = _generate_question(
                config, scenario=scenario, question_index=question_index
            )
            receipt = _question_receipt(config, question)
            validate_stress_question_receipt(
                receipt, expected_config_sha256=str(config["config_sha256"])
            )
            receipts.append(receipt)
    return receipts


_NORMAL_975 = 1.959963984540054


def _wilson_interval(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    proportion = successes / total
    denominator = 1.0 + _NORMAL_975**2 / total
    center = (proportion + _NORMAL_975**2 / (2.0 * total)) / denominator
    half_width = (
        _NORMAL_975
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + _NORMAL_975**2 / (4.0 * total**2)
        )
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _percentile_interval(values: np.ndarray) -> list[float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [None, None]
    quantiles = np.quantile(finite, [0.025, 0.975], method="linear")
    return [float(quantiles[0]), float(quantiles[1])]


def _curve_point(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    released = np.asarray(
        [float(row["policy_visible_predicted_flip_risk"]) <= threshold for row in rows],
        dtype=bool,
    )
    errors = np.asarray([bool(row["claim_decision_error"]) for row in rows], dtype=bool)
    spent = np.asarray([float(row["spent_person_minutes"]) for row in rows])
    released_count = int(np.sum(released))
    released_errors = int(np.sum(released & errors))
    correct_releases = int(np.sum(released & ~errors))
    total_minutes = float(np.sum(spent))
    return {
        "max_policy_visible_flip_risk": float(threshold),
        "questions": len(rows),
        "released_claims": released_count,
        "coverage": released_count / len(rows),
        "coverage_question_level_wilson_ci_95": _wilson_interval(
            released_count, len(rows)
        ),
        "released_claim_errors": released_errors,
        "released_claim_error_rate": (
            released_errors / released_count if released_count else None
        ),
        "released_claim_error_rate_question_level_wilson_ci_95": (
            _wilson_interval(released_errors, released_count)
            if released_count
            else [None, None]
        ),
        "correct_claims_released": correct_releases,
        "correct_claims_released_per_100_questions": 100.0
        * correct_releases
        / len(rows),
        "total_person_minutes": total_minutes,
        "mean_person_minutes_per_question": total_minutes / len(rows),
        "correct_claims_released_per_person_hour": (
            60.0 * correct_releases / total_minutes if total_minutes > 0 else None
        ),
    }


def _bootstrap_operating_point(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    released = np.asarray(
        [float(row["policy_visible_predicted_flip_risk"]) <= threshold for row in rows],
        dtype=bool,
    )
    errors = np.asarray([bool(row["claim_decision_error"]) for row in rows], dtype=bool)
    spent = np.asarray([float(row["spent_person_minutes"]) for row in rows])
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(rows), size=(draws, len(rows)))
    boot_released = released[indices]
    coverage = np.mean(boot_released, axis=1)
    release_counts = np.sum(boot_released, axis=1)
    error_counts = np.sum(boot_released & errors[indices], axis=1)
    error_rate = np.divide(
        error_counts,
        release_counts,
        out=np.full(draws, np.nan),
        where=release_counts > 0,
    )
    correct_counts = np.sum(boot_released & ~errors[indices], axis=1)
    spent_minutes = np.sum(spent[indices], axis=1)
    efficiency = np.divide(
        60.0 * correct_counts,
        spent_minutes,
        out=np.full(draws, np.nan),
        where=spent_minutes > 0,
    )
    return {
        "method": "question_cluster_nonparametric_percentile_bootstrap",
        "confidence_level": 0.95,
        "draws": draws,
        "seed": seed,
        "bit_generator": "PCG64",
        "quantile_method": "linear",
        "coverage_ci_95": _percentile_interval(coverage),
        "released_claim_error_rate_ci_95": _percentile_interval(error_rate),
        "correct_claims_released_per_person_hour_ci_95": _percentile_interval(
            efficiency
        ),
        "finite_draws": {
            "coverage": int(np.sum(np.isfinite(coverage))),
            "released_claim_error_rate": int(np.sum(np.isfinite(error_rate))),
            "correct_claims_released_per_person_hour": int(
                np.sum(np.isfinite(efficiency))
            ),
        },
    }


def _paired_bootstrap_contrast(
    proposed_rows: Sequence[Mapping[str, Any]],
    comparator_rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if len(proposed_rows) != len(comparator_rows) or not proposed_rows:
        raise AdaptiveStressStudyError("paired_contrast_rows_invalid")
    proposed = sorted(proposed_rows, key=lambda row: str(row["question_id"]))
    comparator = sorted(comparator_rows, key=lambda row: str(row["question_id"]))
    if [row["question_id"] for row in proposed] != [
        row["question_id"] for row in comparator
    ]:
        raise AdaptiveStressStudyError("paired_contrast_question_ids_mismatch")

    def arrays(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, ...]:
        released = np.asarray(
            [
                float(row["policy_visible_predicted_flip_risk"]) <= threshold
                for row in rows
            ],
            dtype=bool,
        )
        errors = np.asarray(
            [bool(row["claim_decision_error"]) for row in rows], dtype=bool
        )
        spent = np.asarray([float(row["spent_person_minutes"]) for row in rows])
        return released, errors, spent

    proposed_arrays = arrays(proposed)
    comparator_arrays = arrays(comparator)
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(proposed), size=(draws, len(proposed)))

    def bootstrap_metrics(values: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        released, errors, spent = values
        sampled_released = released[indices]
        release_count = np.sum(sampled_released, axis=1)
        coverage = np.mean(sampled_released, axis=1)
        error_count = np.sum(sampled_released & errors[indices], axis=1)
        error_rate = np.divide(
            error_count,
            release_count,
            out=np.full(draws, np.nan),
            where=release_count > 0,
        )
        correct_count = np.sum(sampled_released & ~errors[indices], axis=1)
        total_minutes = np.sum(spent[indices], axis=1)
        efficiency = np.divide(
            60.0 * correct_count,
            total_minutes,
            out=np.full(draws, np.nan),
            where=total_minutes > 0,
        )
        return coverage, error_rate, efficiency

    proposed_boot = bootstrap_metrics(proposed_arrays)
    comparator_boot = bootstrap_metrics(comparator_arrays)
    proposed_point = _curve_point(proposed, threshold)
    comparator_point = _curve_point(comparator, threshold)

    def difference(name: str) -> float | None:
        left = proposed_point[name]
        right = comparator_point[name]
        if left is None or right is None:
            return None
        return float(left) - float(right)

    return {
        "paired_questions": len(proposed),
        "coverage_proposed_minus_comparator": difference("coverage"),
        "coverage_difference_ci_95": _percentile_interval(
            proposed_boot[0] - comparator_boot[0]
        ),
        "released_claim_error_rate_proposed_minus_comparator": difference(
            "released_claim_error_rate"
        ),
        "released_claim_error_rate_difference_ci_95": _percentile_interval(
            proposed_boot[1] - comparator_boot[1]
        ),
        "correct_claims_per_person_hour_proposed_minus_comparator": difference(
            "correct_claims_released_per_person_hour"
        ),
        "correct_claims_per_person_hour_difference_ci_95": _percentile_interval(
            proposed_boot[2] - comparator_boot[2]
        ),
        "finite_difference_draws": {
            "coverage": int(
                np.sum(np.isfinite(proposed_boot[0] - comparator_boot[0]))
            ),
            "released_claim_error_rate": int(
                np.sum(np.isfinite(proposed_boot[1] - comparator_boot[1]))
            ),
            "correct_claims_per_person_hour": int(
                np.sum(np.isfinite(proposed_boot[2] - comparator_boot[2]))
            ),
        },
        "bootstrap": {
            "method": "paired_question_cluster_nonparametric_percentile_bootstrap",
            "draws": draws,
            "seed": seed,
            "confidence_level": 0.95,
            "bit_generator": "PCG64",
            "quantile_method": "linear",
        },
    }


def summarize_stress_question_receipts(
    receipts: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Aggregate release-error/coverage curves using complete questions as units."""

    validate_stress_study_config(config)
    expected_questions = int(config["questions_per_scenario"]) * len(config["scenarios"])
    if len(receipts) != expected_questions:
        raise AdaptiveStressStudyError("question_receipt_count_mismatch")
    question_ids: set[str] = set()
    evaluation_rows: list[dict[str, Any]] = []
    scenario_counts: dict[str, int] = defaultdict(int)
    for receipt in receipts:
        validate_stress_question_receipt(
            receipt, expected_config_sha256=str(config["config_sha256"])
        )
        question_id = str(receipt["question_id"])
        if question_id in question_ids:
            raise AdaptiveStressStudyError("duplicate_question_id")
        question_ids.add(question_id)
        scenario = str(receipt["scenario"])
        scenario_counts[scenario] += 1
        for evaluation in receipt["evaluations"]:
            evaluation_rows.append(
                {**dict(evaluation), "question_id": question_id, "scenario": scenario}
            )
    if set(scenario_counts) != set(config["scenarios"]) or any(
        count != int(config["questions_per_scenario"])
        for count in scenario_counts.values()
    ):
        raise AdaptiveStressStudyError("scenario_question_grid_incomplete")

    rows_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        rows_by_arm[str(row["arm_id"])].append(row)
    expected_arms = {
        f"{policy}__{_budget_arm_id(float(budget))}"
        for policy in FIXED_BUDGET_POLICIES
        for budget in config["budgets_minutes"]
    } | set(REFERENCE_POLICIES)
    if set(rows_by_arm) != expected_arms or any(
        len(rows) != expected_questions for rows in rows_by_arm.values()
    ):
        raise AdaptiveStressStudyError("policy_budget_question_grid_incomplete")

    thresholds = [float(value) for value in config["release_risk_thresholds"]]
    arm_summaries: dict[str, Any] = {}
    primary_threshold = float(config["primary_release_risk_threshold"])
    bootstrap_draws = int(config["bootstrap_draws"])
    for arm_id in sorted(rows_by_arm):
        rows = rows_by_arm[arm_id]
        representative = rows[0]
        scopes: dict[str, Any] = {}
        for scope in ("overall", *config["scenarios"]):
            scoped_rows = (
                rows if scope == "overall" else [row for row in rows if row["scenario"] == scope]
            )
            curve = [_curve_point(scoped_rows, threshold) for threshold in thresholds]
            operating_bootstrap_seed = _derived_seed(
                config["seed"], "operating-bootstrap", arm_id, scope
            )
            scopes[scope] = {
                "curve": curve,
                "operating_point_question_cluster_uncertainty": _bootstrap_operating_point(
                    scoped_rows,
                    threshold=primary_threshold,
                    draws=bootstrap_draws,
                    seed=operating_bootstrap_seed,
                ),
            }
        arm_summaries[arm_id] = {
            "policy": representative["policy"],
            "budget_role": representative["budget_role"],
            "nominal_budget_minutes": representative["nominal_budget_minutes"],
            "scopes": scopes,
        }

    primary_budget = float(config["primary_budget_minutes"])
    proposed_arm = f"adaptive_sequential__{_budget_arm_id(primary_budget)}"
    proposed_rows = rows_by_arm[proposed_arm]
    comparator_arms = sorted(
        {
            f"{policy}__{_budget_arm_id(primary_budget)}"
            for policy in FIXED_BUDGET_POLICIES
            if policy != "adaptive_sequential"
        }
        | set(REFERENCE_POLICIES)
    )
    paired_contrasts: dict[str, Any] = {}
    for comparator_arm in comparator_arms:
        comparator_rows = rows_by_arm[comparator_arm]
        seed = _derived_seed(
            config["seed"], "paired-bootstrap", proposed_arm, comparator_arm
        )
        paired_contrasts[comparator_arm] = _paired_bootstrap_contrast(
            proposed_rows,
            comparator_rows,
            threshold=primary_threshold,
            draws=bootstrap_draws,
            seed=seed,
        )

    diagnostic_totals = {
        key: int(
            sum(int(receipt["generator_diagnostics"][key]) for receipt in receipts)
        )
        for key in (
            "items",
            "extraction_errors",
            "shared_component_errors",
            "missing_full_text",
            "reviewer_mistakes_among_accessible",
        )
    }
    return {
        "summary_version": SUMMARY_VERSION,
        "independent_questions": expected_questions,
        "questions_per_scenario": dict(sorted(scenario_counts.items())),
        "generator_diagnostic_totals": diagnostic_totals,
        "primary_operating_point": {
            "budget_minutes": primary_budget,
            "max_policy_visible_flip_risk": primary_threshold,
            "proposed_policy": "adaptive_sequential",
        },
        "arms": arm_summaries,
        "primary_paired_contrasts": paired_contrasts,
        "uncertainty": {
            "sampling_unit": "independent_complete_simulated_question",
            "curve_rate_intervals": "two_sided_question_level_wilson_score_95",
            "operating_point_intervals": (
                "question_cluster_nonparametric_percentile_bootstrap_95"
            ),
            "paired_contrasts": (
                "paired_question_cluster_nonparametric_percentile_bootstrap_95"
            ),
            "bootstrap_draws": bootstrap_draws,
            "multiple_comparison_adjustment": None,
            "simultaneous_curve_guarantee": False,
        },
        "interpretation": (
            "Misspecified simulation stress evidence only. Curves estimate behavior "
            "under the frozen synthetic generator, not real scientific-claim error, "
            "real reviewer performance, or a calibrated release guarantee."
        ),
    }


def build_adaptive_stress_study_artifact(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the frozen study and return a compact, self-hashed aggregate artifact."""

    validate_stress_study_config(config)
    receipts = generate_stress_question_receipts(config)
    summary = summarize_stress_question_receipts(receipts, config)
    receipt_manifest = [
        {
            "question_id": receipt["question_id"],
            "scenario": receipt["scenario"],
            "receipt_sha256": receipt["receipt_sha256"],
        }
        for receipt in receipts
    ]
    manifest_payload = {
        "manifest_version": "adaptive-stress-receipt-manifest-v1",
        "config_sha256": config["config_sha256"],
        "receipts": receipt_manifest,
    }
    manifest = {
        **manifest_payload,
        "manifest_sha256": hash_canonical(manifest_payload),
    }
    payload: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "evidence_scope": {
            "artifact_kind": "misspecified_adversarial_simulation",
            "simulation_only": True,
            "real_world_evidence": False,
            "human_adjudication_performed": False,
            "scientific_truth_evaluated": False,
            "release_calibration_guarantee": False,
            "audit_all_is_matched_cost": False,
        },
        "frozen_config": dict(config),
        "question_receipt_manifest": manifest,
        "summary": summary,
        "limitations": [
            (
                "All evidence values, errors, access failures, costs, and reviewer "
                "outcomes are synthetic."
            ),
            "Question-level bootstrap intervals quantify Monte Carlo population variation only.",
            (
                "Release thresholds are sensitivity operating points, not calibrated "
                "real-world risk bounds."
            ),
            (
                "Audit-all is an explicitly unmatched-cost exhaustive-review reference; "
                "all fixed-budget policies share person-minute budgets."
            ),
            (
                "The combined domain-shift profile is prespecified synthetic shift, not "
                "evidence of transfer to a real domain."
            ),
        ],
    }
    artifact = {**payload, "artifact_sha256": hash_canonical(payload)}
    validate_adaptive_stress_study_artifact(artifact)
    return artifact


def validate_adaptive_stress_study_artifact(artifact: Mapping[str, Any]) -> None:
    payload = dict(artifact)
    observed_hash = payload.pop("artifact_sha256", None)
    if not isinstance(observed_hash, str) or not SHA256_RE.fullmatch(observed_hash):
        raise AdaptiveStressStudyError("artifact_hash_missing_or_invalid")
    if hash_canonical(payload) != observed_hash:
        raise AdaptiveStressStudyError("artifact_hash_mismatch")
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        raise AdaptiveStressStudyError("artifact_version_mismatch")
    config = payload.get("frozen_config")
    if not isinstance(config, Mapping):
        raise AdaptiveStressStudyError("artifact_frozen_config_missing")
    validate_stress_study_config(config)
    manifest = payload.get("question_receipt_manifest")
    if not isinstance(manifest, Mapping):
        raise AdaptiveStressStudyError("artifact_receipt_manifest_missing")
    manifest_payload = dict(manifest)
    manifest_hash = manifest_payload.pop("manifest_sha256", None)
    if hash_canonical(manifest_payload) != manifest_hash:
        raise AdaptiveStressStudyError("artifact_receipt_manifest_hash_mismatch")
    if manifest_payload.get("config_sha256") != config["config_sha256"]:
        raise AdaptiveStressStudyError("artifact_manifest_config_mismatch")
    receipts = manifest_payload.get("receipts")
    expected = int(config["questions_per_scenario"]) * len(config["scenarios"])
    if not isinstance(receipts, list) or len(receipts) != expected:
        raise AdaptiveStressStudyError("artifact_manifest_question_count_mismatch")
    if any(
        not isinstance(row, Mapping)
        or set(row) != {"question_id", "scenario", "receipt_sha256"}
        or not isinstance(row.get("question_id"), str)
        or row.get("scenario") not in config["scenarios"]
        or not isinstance(row.get("receipt_sha256"), str)
        or not SHA256_RE.fullmatch(str(row.get("receipt_sha256")))
        for row in receipts
    ):
        raise AdaptiveStressStudyError("artifact_manifest_receipt_row_invalid")
    question_ids = [row.get("question_id") for row in receipts]
    if len(question_ids) != len(set(question_ids)):
        raise AdaptiveStressStudyError("artifact_manifest_duplicate_question")
    scenario_counts: dict[str, int] = defaultdict(int)
    for row in receipts:
        scenario_counts[str(row["scenario"])] += 1
    if scenario_counts != {
        scenario: int(config["questions_per_scenario"])
        for scenario in config["scenarios"]
    }:
        raise AdaptiveStressStudyError("artifact_manifest_scenario_grid_incomplete")
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise AdaptiveStressStudyError("artifact_summary_missing")
    if (
        summary.get("summary_version") != SUMMARY_VERSION
        or summary.get("independent_questions") != expected
        or summary.get("questions_per_scenario") != dict(sorted(scenario_counts.items()))
    ):
        raise AdaptiveStressStudyError("artifact_summary_identity_mismatch")


__all__ = [
    "ALL_POLICIES",
    "ARTIFACT_VERSION",
    "DEFAULT_SCENARIOS",
    "FIXED_BUDGET_POLICIES",
    "QUESTION_RECEIPT_VERSION",
    "REFERENCE_POLICIES",
    "SCENARIO_PROFILES",
    "STRESS_STUDY_VERSION",
    "SUMMARY_VERSION",
    "AdaptiveStressStudyError",
    "StressOracleItem",
    "StressQuestion",
    "StressVisibleItem",
    "build_adaptive_stress_study_artifact",
    "freeze_stress_study_config",
    "generate_stress_question_receipts",
    "summarize_stress_question_receipts",
    "validate_adaptive_stress_study_artifact",
    "validate_stress_question_receipt",
    "validate_stress_study_config",
]
