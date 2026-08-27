"""Allocate a fixed human-verification budget by expected claim impact.

This module deliberately separates information available to an allocation policy
from oracle audit labels used only for retrospective evaluation.  A candidate's
counterfactual contribution must come from a documented leave-one-out or candidate
correction rerun.  It is not an oracle correction unless an audit has established
that fact.

The default allocation score is a scenario-based quantity::

    error_probability * abs(baseline_claim_probability
                            - counterfactual_claim_probability)
    / verification_cost

It is an expected reduction in absolute claim-probability loss only under the
single-item counterfactual and error-probability assumptions.  In particular, a
heuristic error score does not become calibrated merely because it is used here.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum


class ProbabilityBasis(StrEnum):
    """How an item's pre-audit error probability was obtained."""

    CALIBRATED = "calibrated"
    HEURISTIC = "heuristic"
    PLANTED_SIMULATION = "planted_simulation"


class ScenarioKind(StrEnum):
    """Counterfactual used to measure corpus-level conclusion influence."""

    CANDIDATE_CORRECTION = "candidate_correction"
    LEAVE_ONE_OUT = "leave_one_out"


class AllocationPolicy(StrEnum):
    """Deterministic rank-then-pack allocation policies."""

    RANDOM = "random"
    COST_ONLY = "cost_only"
    RISK_ONLY = "risk_only"
    DISAGREEMENT = "disagreement"
    INFLUENCE_ONLY = "influence_only"
    RISK_X_INFLUENCE = "risk_x_influence"
    RISK_PER_COST = "risk_per_cost"
    INFLUENCE_PER_COST = "influence_per_cost"
    EXPECTED_CLAIM_LOSS_PER_COST = "risk_x_influence_per_cost"


class ReleaseGuardStatus(StrEnum):
    """Whether this audit-specific guard permits evaluation by later gates."""

    BLOCKED = "blocked"
    ELIGIBLE_FOR_DOWNSTREAM_GATES = "eligible_for_downstream_gates"


@dataclass(frozen=True, slots=True)
class ClaimModel:
    """Auditable additive snapshot of one corpus-level binary claim.

    Contributions are on a logit-like score scale.  ``temperature`` makes the
    mapping explicit rather than silently assuming that upstream scores are logits.
    For a non-additive synthesizer, callers should calculate local contribution
    replacements that reproduce its baseline and one-item counterfactual reruns.
    """

    intercept: float
    temperature: float = 1.0
    decision_threshold: float = 0.5
    claim_id: str = "claim"

    def __post_init__(self) -> None:
        if not self.claim_id.strip():
            raise ValueError("claim_id_empty")
        if not math.isfinite(self.intercept):
            raise ValueError("claim_intercept_nonfinite")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("claim_temperature_must_be_positive")
        if not math.isfinite(self.decision_threshold) or not (
            0 < self.decision_threshold < 1
        ):
            raise ValueError("claim_decision_threshold_invalid")

    def probability(self, contributions: Sequence[float]) -> float:
        if any(not math.isfinite(value) for value in contributions):
            raise ValueError("claim_contribution_nonfinite")
        score = self.intercept + math.fsum(contributions)
        scaled = score / self.temperature
        if scaled >= 0:
            return 1.0 / (1.0 + math.exp(-scaled))
        exp_scaled = math.exp(scaled)
        return exp_scaled / (1.0 + exp_scaled)

    def conclusion(self, probability: float) -> bool:
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("claim_probability_invalid")
        return probability >= self.decision_threshold


@dataclass(frozen=True, slots=True)
class AuditCandidate:
    """Policy-visible contract for one item that a human could verify.

    ``counterfactual_contribution`` is a scenario, not a hidden gold label.  For a
    leave-one-out scenario it must be zero.  All candidates in one allocation must
    use the same ``cost_unit``.
    """

    item_id: str
    baseline_contribution: float
    counterfactual_contribution: float
    error_probability: float
    probability_basis: ProbabilityBasis
    probability_source: str
    verification_cost: float
    cost_unit: str
    disagreement_score: float
    scenario_kind: ScenarioKind
    scenario_source: str

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("audit_item_id_empty")
        for label, value in (
            ("baseline_contribution", self.baseline_contribution),
            ("counterfactual_contribution", self.counterfactual_contribution),
        ):
            if not math.isfinite(value):
                raise ValueError(f"audit_{label}_nonfinite")
        if not math.isfinite(self.error_probability) or not (
            0 <= self.error_probability <= 1
        ):
            raise ValueError("audit_error_probability_invalid")
        if not math.isfinite(self.verification_cost) or self.verification_cost <= 0:
            raise ValueError("audit_verification_cost_must_be_positive")
        if not math.isfinite(self.disagreement_score) or not (
            0 <= self.disagreement_score <= 1
        ):
            raise ValueError("audit_disagreement_score_invalid")
        if not isinstance(self.probability_basis, ProbabilityBasis):
            raise ValueError("audit_probability_basis_invalid")
        if not isinstance(self.scenario_kind, ScenarioKind):
            raise ValueError("audit_scenario_kind_invalid")
        for label, value in (
            ("probability_source", self.probability_source),
            ("cost_unit", self.cost_unit),
            ("scenario_source", self.scenario_source),
        ):
            if not value.strip():
                raise ValueError(f"audit_{label}_empty")
        if (
            self.scenario_kind is ScenarioKind.LEAVE_ONE_OUT
            and self.counterfactual_contribution != 0.0
        ):
            raise ValueError("leave_one_out_contribution_must_be_zero")


@dataclass(frozen=True, slots=True)
class AuditOracle:
    """Audit-only outcome; never accepted by ranking functions."""

    item_id: str
    is_error: bool
    corrected_contribution: float
    label_source: str

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("audit_oracle_item_id_empty")
        if not isinstance(self.is_error, bool):
            raise ValueError("audit_oracle_is_error_must_be_boolean")
        if not math.isfinite(self.corrected_contribution):
            raise ValueError("audit_oracle_corrected_contribution_nonfinite")
        if not self.label_source.strip():
            raise ValueError("audit_oracle_label_source_empty")


@dataclass(frozen=True, slots=True)
class PriorityRecord:
    item_id: str
    policy: AllocationPolicy
    rank: int
    priority: float
    error_probability: float
    probability_basis: ProbabilityBasis
    verification_cost: float
    disagreement_score: float
    baseline_claim_probability: float
    counterfactual_claim_probability: float
    probability_influence: float
    conclusion_flip: bool
    expected_claim_loss_reduction: float
    expected_claim_loss_reduction_per_cost: float


@dataclass(frozen=True, slots=True)
class BudgetSelection:
    policy: AllocationPolicy
    budget: float
    spent: float
    cost_unit: str
    selected_item_ids: tuple[str, ...]
    ranking: tuple[PriorityRecord, ...]


@dataclass(frozen=True, slots=True)
class ReleaseGuardConfig:
    """Conservative thresholds for unresolved audit candidates.

    The default requires calibrated item-error probabilities.  Setting that flag to
    false permits a heuristic prioritization workflow, but cannot create a calibrated
    release-risk guarantee.
    """

    max_unresolved_item_influence: float = 0.05
    max_unresolved_expected_claim_loss: float = 0.05
    block_counterfactual_conclusion_flips: bool = True
    require_calibrated_error_probabilities: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_unresolved_item_influence) or not (
            0 <= self.max_unresolved_item_influence <= 1
        ):
            raise ValueError("release_guard_item_influence_limit_invalid")
        if not math.isfinite(self.max_unresolved_expected_claim_loss) or (
            self.max_unresolved_expected_claim_loss < 0
        ):
            raise ValueError("release_guard_expected_loss_limit_invalid")


@dataclass(frozen=True, slots=True)
class ReleaseGuardDecision:
    """Audit-gate result; eligibility here never constitutes final claim release."""

    status: ReleaseGuardStatus
    reasons: tuple[str, ...]
    resolved_item_ids: tuple[str, ...]
    unresolved_item_ids: tuple[str, ...]
    unresolved_conclusion_flip_item_ids: tuple[str, ...]
    unresolved_high_influence_item_ids: tuple[str, ...]
    unresolved_noncalibrated_item_ids: tuple[str, ...]
    unresolved_expected_claim_loss: float
    config: ReleaseGuardConfig


def _validate_candidates(candidates: Sequence[AuditCandidate]) -> tuple[str, dict[str, int]]:
    if not candidates:
        raise ValueError("audit_candidates_empty")
    indices: dict[str, int] = {}
    cost_units = set()
    for index, candidate in enumerate(candidates):
        if candidate.item_id in indices:
            raise ValueError(f"audit_candidate_id_duplicate:{candidate.item_id}")
        indices[candidate.item_id] = index
        cost_units.add(candidate.cost_unit)
    if len(cost_units) != 1:
        raise ValueError("audit_candidate_cost_units_mixed")
    return next(iter(cost_units)), indices


def _random_priority(*, seed: int, item_id: str) -> float:
    digest = hashlib.sha256(f"budgeted-verification-v1\0{seed}\0{item_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def rank_candidates(
    candidates: Sequence[AuditCandidate],
    claim_model: ClaimModel,
    policy: AllocationPolicy,
    *,
    seed: int = 0,
) -> tuple[PriorityRecord, ...]:
    """Rank candidates without accepting or consulting audit-oracle labels."""

    _, indices = _validate_candidates(candidates)
    baseline_contributions = [candidate.baseline_contribution for candidate in candidates]
    baseline_probability = claim_model.probability(baseline_contributions)
    baseline_conclusion = claim_model.conclusion(baseline_probability)
    pending: list[tuple[float, str, dict[str, object]]] = []
    for candidate in candidates:
        counterfactual = list(baseline_contributions)
        counterfactual[indices[candidate.item_id]] = candidate.counterfactual_contribution
        counterfactual_probability = claim_model.probability(counterfactual)
        influence = abs(baseline_probability - counterfactual_probability)
        expected_reduction = candidate.error_probability * influence
        expected_per_cost = expected_reduction / candidate.verification_cost
        if policy is AllocationPolicy.RANDOM:
            priority = _random_priority(seed=seed, item_id=candidate.item_id)
        elif policy is AllocationPolicy.COST_ONLY:
            priority = 1.0 / candidate.verification_cost
        elif policy is AllocationPolicy.RISK_ONLY:
            priority = candidate.error_probability
        elif policy is AllocationPolicy.DISAGREEMENT:
            priority = candidate.disagreement_score
        elif policy is AllocationPolicy.INFLUENCE_ONLY:
            priority = influence
        elif policy is AllocationPolicy.RISK_X_INFLUENCE:
            priority = expected_reduction
        elif policy is AllocationPolicy.RISK_PER_COST:
            priority = candidate.error_probability / candidate.verification_cost
        elif policy is AllocationPolicy.INFLUENCE_PER_COST:
            priority = influence / candidate.verification_cost
        elif policy is AllocationPolicy.EXPECTED_CLAIM_LOSS_PER_COST:
            priority = expected_per_cost
        else:  # pragma: no cover - defensive for callers bypassing the enum
            raise ValueError(f"audit_policy_unknown:{policy}")
        pending.append(
            (
                priority,
                candidate.item_id,
                {
                    "error_probability": candidate.error_probability,
                    "probability_basis": candidate.probability_basis,
                    "verification_cost": candidate.verification_cost,
                    "disagreement_score": candidate.disagreement_score,
                    "counterfactual_claim_probability": counterfactual_probability,
                    "probability_influence": influence,
                    "conclusion_flip": (
                        claim_model.conclusion(counterfactual_probability)
                        != baseline_conclusion
                    ),
                    "expected_claim_loss_reduction": expected_reduction,
                    "expected_claim_loss_reduction_per_cost": expected_per_cost,
                },
            )
        )
    pending.sort(key=lambda row: (-row[0], row[1]))
    return tuple(
        PriorityRecord(
            item_id=item_id,
            policy=policy,
            rank=rank,
            priority=priority,
            error_probability=float(values["error_probability"]),
            probability_basis=values["probability_basis"],  # type: ignore[arg-type]
            verification_cost=float(values["verification_cost"]),
            disagreement_score=float(values["disagreement_score"]),
            baseline_claim_probability=baseline_probability,
            counterfactual_claim_probability=float(
                values["counterfactual_claim_probability"]
            ),
            probability_influence=float(values["probability_influence"]),
            conclusion_flip=bool(values["conclusion_flip"]),
            expected_claim_loss_reduction=float(
                values["expected_claim_loss_reduction"]
            ),
            expected_claim_loss_reduction_per_cost=float(
                values["expected_claim_loss_reduction_per_cost"]
            ),
        )
        for rank, (priority, item_id, values) in enumerate(pending, start=1)
    )


def select_under_budget(
    candidates: Sequence[AuditCandidate],
    claim_model: ClaimModel,
    policy: AllocationPolicy,
    *,
    budget: float,
    seed: int = 0,
) -> BudgetSelection:
    """Rank, then greedily take every fitting item under an explicit cost cap."""

    if not math.isfinite(budget) or budget < 0:
        raise ValueError("audit_budget_invalid")
    cost_unit, _ = _validate_candidates(candidates)
    ranking = rank_candidates(candidates, claim_model, policy, seed=seed)
    selected: list[str] = []
    spent = 0.0
    for record in ranking:
        next_spent = spent + record.verification_cost
        if next_spent <= budget + 1e-12:
            selected.append(record.item_id)
            spent = next_spent
    return BudgetSelection(
        policy=policy,
        budget=budget,
        spent=spent,
        cost_unit=cost_unit,
        selected_item_ids=tuple(selected),
        ranking=ranking,
    )


def assess_prospective_release_guard(
    candidates: Sequence[AuditCandidate],
    claim_model: ClaimModel,
    *,
    resolved_item_ids: Sequence[str],
    config: ReleaseGuardConfig | None = None,
) -> ReleaseGuardDecision:
    """Block on material unresolved counterfactuals before downstream release gates.

    ``resolved_item_ids`` means human adjudication is complete.  If adjudication found
    an error, the caller must first incorporate the correction into a new baseline
    snapshot and rerun this function.  Merely allocating or assigning an item does not
    resolve it.

    The summed scenario loss is a triage burden, not a joint probabilistic bound:
    counterfactuals can interact and heuristic error scores are not probabilities.
    """

    _validate_candidates(candidates)
    if config is None:
        config = ReleaseGuardConfig()
    if len(set(resolved_item_ids)) != len(resolved_item_ids):
        raise ValueError("release_guard_resolved_item_ids_duplicate")
    candidate_ids = {candidate.item_id for candidate in candidates}
    resolved = set(resolved_item_ids)
    unknown = sorted(resolved - candidate_ids)
    if unknown:
        raise ValueError(f"release_guard_resolved_item_ids_unknown:{unknown}")

    ranking = rank_candidates(
        candidates,
        claim_model,
        AllocationPolicy.EXPECTED_CLAIM_LOSS_PER_COST,
    )
    candidate_by_id = {candidate.item_id: candidate for candidate in candidates}
    unresolved = [record for record in ranking if record.item_id not in resolved]
    flips = tuple(record.item_id for record in unresolved if record.conclusion_flip)
    high_influence = tuple(
        record.item_id
        for record in unresolved
        if record.probability_influence > config.max_unresolved_item_influence
    )
    noncalibrated = tuple(
        record.item_id
        for record in unresolved
        if candidate_by_id[record.item_id].probability_basis
        is not ProbabilityBasis.CALIBRATED
    )
    expected_loss = math.fsum(
        record.expected_claim_loss_reduction for record in unresolved
    )
    reasons: list[str] = []
    if config.block_counterfactual_conclusion_flips and flips:
        reasons.append("unresolved_counterfactual_can_flip_conclusion")
    if high_influence:
        reasons.append("unresolved_item_influence_exceeds_limit")
    if expected_loss > config.max_unresolved_expected_claim_loss:
        reasons.append("unresolved_expected_claim_loss_exceeds_limit")
    if config.require_calibrated_error_probabilities and noncalibrated:
        reasons.append("unresolved_error_probabilities_not_calibrated")
    return ReleaseGuardDecision(
        status=(
            ReleaseGuardStatus.BLOCKED
            if reasons
            else ReleaseGuardStatus.ELIGIBLE_FOR_DOWNSTREAM_GATES
        ),
        reasons=tuple(reasons),
        resolved_item_ids=tuple(sorted(resolved)),
        unresolved_item_ids=tuple(record.item_id for record in unresolved),
        unresolved_conclusion_flip_item_ids=flips,
        unresolved_high_influence_item_ids=high_influence,
        unresolved_noncalibrated_item_ids=noncalibrated,
        unresolved_expected_claim_loss=expected_loss,
        config=config,
    )


def _validate_oracles(
    candidates: Sequence[AuditCandidate], oracles: Sequence[AuditOracle]
) -> Mapping[str, AuditOracle]:
    candidate_ids = {candidate.item_id for candidate in candidates}
    oracle_by_id: dict[str, AuditOracle] = {}
    for oracle in oracles:
        if oracle.item_id in oracle_by_id:
            raise ValueError(f"audit_oracle_id_duplicate:{oracle.item_id}")
        oracle_by_id[oracle.item_id] = oracle
    oracle_ids = set(oracle_by_id)
    if oracle_ids != candidate_ids:
        missing = sorted(candidate_ids - oracle_ids)
        extra = sorted(oracle_ids - candidate_ids)
        raise ValueError(f"audit_oracle_identity_mismatch:missing={missing}:extra={extra}")
    return oracle_by_id


def evaluate_fixed_budgets(
    candidates: Sequence[AuditCandidate],
    oracles: Sequence[AuditOracle],
    claim_model: ClaimModel,
    *,
    budgets: Sequence[float],
    policies: Sequence[AllocationPolicy] = tuple(AllocationPolicy),
    seed: int = 0,
) -> list[dict[str, object]]:
    """Evaluate allocation policies using oracle labels hidden during ranking.

    The evaluation loss is absolute distance from the fully oracle-corrected claim
    probability.  A selected item changes the evaluated claim only when the audit
    oracle marks it erroneous.
    """

    _, indices = _validate_candidates(candidates)
    oracle_by_id = _validate_oracles(candidates, oracles)
    if not budgets:
        raise ValueError("audit_budgets_empty")
    if not policies:
        raise ValueError("audit_policies_empty")
    if len(set(budgets)) != len(budgets):
        raise ValueError("audit_budgets_duplicate")
    if len(set(policies)) != len(policies):
        raise ValueError("audit_policies_duplicate")
    for budget in budgets:
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("audit_budget_invalid")

    baseline_contributions = [candidate.baseline_contribution for candidate in candidates]
    oracle_contributions = list(baseline_contributions)
    for item_id, oracle in oracle_by_id.items():
        if oracle.is_error:
            oracle_contributions[indices[item_id]] = oracle.corrected_contribution
    baseline_probability = claim_model.probability(baseline_contributions)
    oracle_probability = claim_model.probability(oracle_contributions)
    baseline_loss = abs(baseline_probability - oracle_probability)
    baseline_conclusion = claim_model.conclusion(baseline_probability)
    oracle_conclusion = claim_model.conclusion(oracle_probability)
    total_errors = sum(oracle.is_error for oracle in oracles)

    rows: list[dict[str, object]] = []
    for policy in policies:
        for budget in budgets:
            selection = select_under_budget(
                candidates,
                claim_model,
                policy,
                budget=budget,
                seed=seed,
            )
            audited_contributions = list(baseline_contributions)
            errors_found = 0
            for item_id in selection.selected_item_ids:
                oracle = oracle_by_id[item_id]
                if oracle.is_error:
                    errors_found += 1
                    audited_contributions[indices[item_id]] = oracle.corrected_contribution
            audited_probability = claim_model.probability(audited_contributions)
            audited_loss = abs(audited_probability - oracle_probability)
            audited_conclusion = claim_model.conclusion(audited_probability)
            rows.append(
                {
                    "policy": policy.value,
                    "budget": budget,
                    "spent": selection.spent,
                    "cost_unit": selection.cost_unit,
                    "selected": len(selection.selected_item_ids),
                    "selected_item_ids": list(selection.selected_item_ids),
                    "errors_found": errors_found,
                    "total_errors": total_errors,
                    "error_recall": errors_found / total_errors if total_errors else None,
                    "baseline_claim_probability": baseline_probability,
                    "audited_claim_probability": audited_probability,
                    "oracle_claim_probability": oracle_probability,
                    "baseline_claim_loss": baseline_loss,
                    "audited_claim_loss": audited_loss,
                    "claim_loss_reduction": baseline_loss - audited_loss,
                    "claim_loss_recovery_fraction": (
                        (baseline_loss - audited_loss) / baseline_loss
                        if baseline_loss > 0
                        else None
                    ),
                    "baseline_conclusion": baseline_conclusion,
                    "audited_conclusion": audited_conclusion,
                    "oracle_conclusion": oracle_conclusion,
                    "audited_conclusion_correct": audited_conclusion == oracle_conclusion,
                    "claim_repaired": (
                        baseline_conclusion != oracle_conclusion
                        and audited_conclusion == oracle_conclusion
                    ),
                }
            )
    return rows


__all__ = [
    "AllocationPolicy",
    "AuditCandidate",
    "AuditOracle",
    "BudgetSelection",
    "ClaimModel",
    "PriorityRecord",
    "ProbabilityBasis",
    "ReleaseGuardConfig",
    "ReleaseGuardDecision",
    "ReleaseGuardStatus",
    "ScenarioKind",
    "assess_prospective_release_guard",
    "evaluate_fixed_budgets",
    "rank_candidates",
    "select_under_budget",
]
