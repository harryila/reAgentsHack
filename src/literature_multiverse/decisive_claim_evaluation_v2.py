"""Additive realized-cost and fixed-error frontiers for decisive evaluation v1.

Version 1 deliberately compares policies at equal *nominal* per-question deadlines and
marks same-realized-cost authority false.  This module leaves every v1 object and hash
unchanged.  It compiles already-scored v1 policy populations into two label-safe views:

* step frontiers at prespecified common realized-person-minute ceilings; and
* coverage/correct-release frontiers at a prespecified released-error ceiling.

The extension never reruns a policy, opens a label file, or invents an audit cost.  It
requires real expert-adjudicated, certificate-bound, complete-question v1 inputs and
rejects simulation, diagnostics, mixed populations, and unequal provenance.  Optional
``AdaptiveCalibrationBundleV2`` objects are externally revalidated and must match the
evaluation pipeline, population, domains, adjudication protocol, arm, and budget.  A
frontier point receives error-control authority only when that exact bundle is embedded
in every terminal production replay used by the point.  Descriptive observed-error
selection never acquires authority merely because its test-set error is small.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import ConfigDict, Field, field_validator, model_validator

from literature_multiverse.adaptive_calibration import (
    AdaptiveCalibrationBundleV2,
    validate_adaptive_calibration_bundle_v2_integrity,
)
from literature_multiverse.decisive_claim_evaluation_v1 import (
    PRIMARY_POLICY_ARM_ID,
    DecisiveClaimEvaluationResultV1,
    FrozenPolicyPopulationV1,
    ScoredQuestionOutcomeV1,
    _score_question_v1,
)
from literature_multiverse.lineage import hash_canonical, sha256_file
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.question_evaluation import (
    BenchmarkEvidenceKind,
    ConditionProductionReplayBindingV7,
    ReferenceVerdictSource,
    ReplaySource,
)

CONFIG_VERSION = "decisive-frontier-config-v2"
SOURCE_ANCHOR_VERSION = "decisive-frontier-source-anchor-v2"
CALIBRATION_ANCHOR_VERSION = "decisive-frontier-calibration-anchor-v2"
POINT_VERSION = "decisive-compiled-policy-point-v2"
COST_ROW_VERSION = "decisive-realized-cost-frontier-row-v2"
FIXED_ERROR_ROW_VERSION = "decisive-fixed-error-frontier-row-v2"
SELECTION_VERSION = "decisive-fixed-error-policy-selection-v2"
COMPARISON_VERSION = "decisive-frontier-paired-comparison-v2"
RESULT_VERSION = "decisive-claim-evaluation-frontiers-v2"
MODULE_PATH = "src/literature_multiverse/decisive_claim_evaluation_v2.py"
CLI_PATH = "scripts/run_decisive_claim_evaluation_v2.py"

_TOLERANCE = 1e-9
_INTERVAL_METRICS = (
    "release_coverage",
    "released_claim_error",
    "correct_releases_per_question",
    "correct_releases_per_human_hour",
    "worst_domain_release_coverage",
    "worst_domain_released_claim_error",
    "worst_domain_correct_releases_per_question",
    "worst_domain_correct_releases_per_human_hour",
)
_DELTA_METRICS = tuple(f"{name}_delta" for name in _INTERVAL_METRICS)


class DecisiveClaimEvaluationV2Error(ValueError):
    """The additive frontier contract failed closed."""


class _FrozenExactModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[str, Field(pattern=SHA256_RE.pattern)]


def _self_hash(model: _FrozenExactModel, field_name: str) -> None:
    expected = hash_canonical(model.model_dump(mode="json", exclude={field_name}))
    if getattr(model, field_name) != expected:
        raise ValueError(f"decisive_evaluation_v2_self_hash_mismatch:{field_name}")


def _finite(value: float, label: str, *, nonnegative: bool = True) -> float:
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"decisive_evaluation_v2_number_invalid:{label}")
    return value


def _sorted_unique(values: Sequence[str], label: str) -> list[str]:
    rows = list(values)
    if not rows or rows != sorted(set(rows)) or any(not row.strip() for row in rows):
        raise ValueError(f"decisive_evaluation_v2_not_sorted_unique:{label}")
    return rows


def _seed(base: int, *parts: object) -> int:
    payload = "\0".join(("decisive-frontier-v2", str(base), *(str(row) for row in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def compute_decisive_evaluation_v2_component_sha256(repository_root: Path) -> str:
    """Hash the additive evaluator and its local semantic dependencies."""

    if repository_root.is_symlink():
        raise DecisiveClaimEvaluationV2Error("decisive_evaluation_v2_repository_root_invalid")
    root = repository_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise DecisiveClaimEvaluationV2Error("decisive_evaluation_v2_repository_root_invalid")
    paths = (
        MODULE_PATH,
        CLI_PATH,
        "src/literature_multiverse/decisive_claim_evaluation_v1.py",
        "src/literature_multiverse/adaptive_calibration.py",
        "src/literature_multiverse/question_evaluation.py",
        "src/literature_multiverse/lineage.py",
        "src/literature_multiverse/models.py",
        "pyproject.toml",
        "uv.lock",
    )
    rows: list[dict[str, Any]] = []
    for relative in paths:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise DecisiveClaimEvaluationV2Error(
                f"decisive_evaluation_v2_component_file_missing:{relative}"
            )
        rows.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return hash_canonical(rows)


class DecisiveFrontierConfigV2(_FrozenExactModel):
    config_version: Literal["decisive-frontier-config-v2"] = CONFIG_VERSION
    common_realized_person_minutes_per_question_cutoffs: Annotated[list[float], Field(min_length=1)]
    released_error_ceiling: Annotated[float, Field(gt=0, lt=1)]
    bootstrap_draws: Annotated[int, Field(ge=100)] = 2000
    bootstrap_seed: Annotated[int, Field(ge=0)] = 20260831
    primary_policy_arm_id: Literal["risk_x_influence_per_cost_adaptive"] = PRIMARY_POLICY_ARM_ID
    minimum_complete_evaluation_questions: Annotated[int, Field(ge=20)] = 20
    minimum_questions_per_domain: Annotated[int, Field(ge=1)] = 2
    point_selection_rule: Literal[
        "largest_observed_realized_spend_not_exceeding_common_ceiling_then_largest_nominal_budget"
    ] = "largest_observed_realized_spend_not_exceeding_common_ceiling_then_largest_nominal_budget"
    fixed_error_selection_rule: Literal[
        "typed_calibration_max_nominal_budget_else_descriptive_max_coverage_then_correct_"
        "then_min_cost"
    ] = (
        "typed_calibration_max_nominal_budget_else_descriptive_max_coverage_then_correct_"
        "then_min_cost"
    )
    require_real_expert_adjudicated_complete_questions: Literal[True] = True
    require_identical_question_population_and_pipeline: Literal[True] = True
    require_typed_calibration_for_claim_authority: Literal[True] = True
    config_sha256: Sha256

    @field_validator("common_realized_person_minutes_per_question_cutoffs")
    @classmethod
    def validate_cutoffs(cls, values: list[float]) -> list[float]:
        if values != sorted(set(values)):
            raise ValueError("decisive_evaluation_v2_cutoffs_not_sorted_unique")
        for value in values:
            _finite(value, "realized_cost_cutoff")
        return values

    @model_validator(mode="after")
    def validate_config(self) -> DecisiveFrontierConfigV2:
        _self_hash(self, "config_sha256")
        return self


def freeze_decisive_frontier_config_v2(
    *,
    common_realized_person_minutes_per_question_cutoffs: Sequence[float] = (
        15.0,
        30.0,
        60.0,
    ),
    released_error_ceiling: float = 0.05,
    bootstrap_draws: int = 2000,
    bootstrap_seed: int = 20260831,
    minimum_complete_evaluation_questions: int = 20,
    minimum_questions_per_domain: int = 2,
) -> DecisiveFrontierConfigV2:
    payload: dict[str, Any] = {
        "config_version": CONFIG_VERSION,
        "common_realized_person_minutes_per_question_cutoffs": sorted(
            float(value) for value in common_realized_person_minutes_per_question_cutoffs
        ),
        "released_error_ceiling": float(released_error_ceiling),
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": bootstrap_seed,
        "primary_policy_arm_id": PRIMARY_POLICY_ARM_ID,
        "minimum_complete_evaluation_questions": minimum_complete_evaluation_questions,
        "minimum_questions_per_domain": minimum_questions_per_domain,
        "point_selection_rule": (
            "largest_observed_realized_spend_not_exceeding_common_ceiling_then_largest_nominal_budget"
        ),
        "fixed_error_selection_rule": (
            "typed_calibration_max_nominal_budget_else_descriptive_max_coverage_then_correct_then_min_cost"
        ),
        "require_real_expert_adjudicated_complete_questions": True,
        "require_identical_question_population_and_pipeline": True,
        "require_typed_calibration_for_claim_authority": True,
    }
    return DecisiveFrontierConfigV2.model_validate(
        {**payload, "config_sha256": hash_canonical(payload)}
    )


class DecisiveFrontierSourceAnchorV2(_FrozenExactModel):
    anchor_version: Literal["decisive-frontier-source-anchor-v2"] = SOURCE_ANCHOR_VERSION
    source_result_version: Literal["decisive-claim-evaluation-result-v1"]
    source_result_sha256: Sha256
    source_policy_freeze_sha256: Sha256
    source_component_sha256: Sha256
    split_manifest_sha256: Sha256
    trajectory_bundle_sha256: Sha256
    opened_label_membership_sha256: Sha256
    scored_population_membership_sha256: Sha256
    pipeline_sha256: Sha256
    question_population_ids: list[str]
    question_population_membership_sha256: Sha256
    adjudication_protocol_sha256: Sha256
    evidence_kind: Literal["real_expert_adjudicated"]
    evaluation_question_ids: list[str]
    domains: list[str]
    complete_question_inputs_structurally_replayed_from_v1: Literal[True] = True
    identical_question_population_across_policies: Literal[True] = True
    identical_pipeline_across_policies: Literal[True] = True
    simulation_or_fixture_input: Literal[False] = False
    anchor_sha256: Sha256

    @field_validator("evaluation_question_ids", "domains", "question_population_ids")
    @classmethod
    def validate_lists(cls, values: list[str], info: Any) -> list[str]:
        return _sorted_unique(values, info.field_name)

    @model_validator(mode="after")
    def validate_anchor(self) -> DecisiveFrontierSourceAnchorV2:
        _self_hash(self, "anchor_sha256")
        return self


class DecisiveFrontierCalibrationAnchorV2(_FrozenExactModel):
    anchor_version: Literal["decisive-frontier-calibration-anchor-v2"] = CALIBRATION_ANCHOR_VERSION
    bundle_sha256: Sha256
    selected_candidate_sha256: Sha256
    policy_context_sha256: Sha256
    policy_arm_id: Annotated[str, Field(min_length=1)]
    budget_minutes_per_question: Annotated[float, Field(ge=0)]
    pipeline_sha256: Sha256
    population_id: Annotated[str, Field(min_length=1)]
    adjudication_protocol_sha256: Sha256
    alpha: Annotated[float, Field(gt=0, lt=1)]
    delta: Annotated[float, Field(gt=0, lt=1)]
    development_question_count: Annotated[int, Field(ge=1)]
    calibration_question_count: Annotated[int, Field(ge=1)]
    calibration_domains: list[str]
    label_source: Literal["expert_adjudication"]
    independence_verified: Literal[True] = True
    real_release_eligible: Literal[True] = True
    exact_bundle_embedded_in_every_terminal_replay: bool
    released_error_ceiling_passed: bool
    point_claim_authority: bool
    authority_blockers: list[str]
    anchor_sha256: Sha256

    @field_validator("calibration_domains")
    @classmethod
    def validate_domains(cls, values: list[str]) -> list[str]:
        return _sorted_unique(values, "calibration_domains")

    @field_validator("authority_blockers")
    @classmethod
    def validate_blockers(cls, values: list[str]) -> list[str]:
        if values != sorted(set(values)) or any(not value for value in values):
            raise ValueError("decisive_evaluation_v2_calibration_blockers_invalid")
        return values

    @model_validator(mode="after")
    def validate_anchor(self) -> DecisiveFrontierCalibrationAnchorV2:
        expected = (
            self.exact_bundle_embedded_in_every_terminal_replay
            and self.released_error_ceiling_passed
            and not self.authority_blockers
        )
        if self.point_claim_authority != expected:
            raise ValueError("decisive_evaluation_v2_calibration_authority_mismatch")
        _self_hash(self, "anchor_sha256")
        return self


class FrontierMetricsV2(_FrozenExactModel):
    n_complete_independent_questions: Annotated[int, Field(ge=1)]
    total_realized_person_minutes: Annotated[float, Field(ge=0)]
    mean_realized_person_minutes_per_question: Annotated[float, Field(ge=0)]
    released_claims: Annotated[int, Field(ge=0)]
    released_claim_errors: Annotated[int, Field(ge=0)]
    correct_releases: Annotated[int, Field(ge=0)]
    release_coverage: Annotated[float, Field(ge=0, le=1)]
    released_claim_error: Annotated[float, Field(ge=0, le=1)] | None
    correct_releases_per_question: Annotated[float, Field(ge=0, le=1)]
    correct_releases_per_human_hour: Annotated[float, Field(ge=0)] | None

    @model_validator(mode="after")
    def validate_metrics(self) -> FrontierMetricsV2:
        n = self.n_complete_independent_questions
        if (
            self.released_claim_errors > self.released_claims
            or self.correct_releases > self.released_claims
            or self.released_claim_errors + self.correct_releases != self.released_claims
        ):
            raise ValueError("decisive_evaluation_v2_release_counts_invalid")
        expected_error = (
            self.released_claim_errors / self.released_claims if self.released_claims else None
        )
        expected_efficiency = (
            self.correct_releases * 60.0 / self.total_realized_person_minutes
            if self.total_realized_person_minutes > 0
            else None
        )
        checks = (
            (
                self.mean_realized_person_minutes_per_question,
                self.total_realized_person_minutes / n,
            ),
            (self.release_coverage, self.released_claims / n),
            (self.correct_releases_per_question, self.correct_releases / n),
        )
        if any(
            not math.isclose(left, right, rel_tol=1e-12, abs_tol=_TOLERANCE)
            for left, right in checks
        ):
            raise ValueError("decisive_evaluation_v2_metric_projection_mismatch")
        if (self.released_claim_error is None) != (expected_error is None) or (
            expected_error is not None
            and not math.isclose(
                float(self.released_claim_error),
                expected_error,
                rel_tol=1e-12,
                abs_tol=_TOLERANCE,
            )
        ):
            raise ValueError("decisive_evaluation_v2_error_rate_mismatch")
        if (self.correct_releases_per_human_hour is None) != (expected_efficiency is None) or (
            expected_efficiency is not None
            and not math.isclose(
                float(self.correct_releases_per_human_hour),
                expected_efficiency,
                rel_tol=1e-12,
                abs_tol=_TOLERANCE,
            )
        ):
            raise ValueError("decisive_evaluation_v2_efficiency_mismatch")
        return self


class DomainFrontierMetricsV2(_FrozenExactModel):
    domain: Annotated[str, Field(min_length=1)]
    metrics: FrontierMetricsV2


class WorstDomainMetricsV2(_FrozenExactModel):
    release_coverage_domain: Annotated[str, Field(min_length=1)]
    release_coverage: Annotated[float, Field(ge=0, le=1)]
    released_claim_error_domain: str | None
    released_claim_error: Annotated[float, Field(ge=0, le=1)] | None
    correct_releases_per_question_domain: Annotated[str, Field(min_length=1)]
    correct_releases_per_question: Annotated[float, Field(ge=0, le=1)]
    correct_releases_per_human_hour_domain: str | None
    correct_releases_per_human_hour: Annotated[float, Field(ge=0)] | None
    domains_without_releases: list[str]

    @field_validator("domains_without_releases")
    @classmethod
    def validate_no_release_domains(cls, values: list[str]) -> list[str]:
        if values != sorted(set(values)):
            raise ValueError("decisive_evaluation_v2_domains_without_releases_invalid")
        return values


class FrontierBootstrapIntervalV2(_FrozenExactModel):
    metric: Annotated[str, Field(min_length=1)]
    confidence_level: Literal[0.95] = 0.95
    interval: Annotated[list[float], Field(min_length=2, max_length=2)] | None
    valid_draws: Annotated[int, Field(ge=0)]
    requested_draws: Annotated[int, Field(ge=100)]
    undefined_draws: Annotated[int, Field(ge=0)]
    bootstrap_fraction_delta_gt_zero: Annotated[float, Field(ge=0, le=1)] | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> FrontierBootstrapIntervalV2:
        if self.valid_draws + self.undefined_draws != self.requested_draws:
            raise ValueError("decisive_evaluation_v2_bootstrap_draw_accounting_invalid")
        if (self.interval is None) != (self.valid_draws == 0):
            raise ValueError("decisive_evaluation_v2_bootstrap_interval_presence_invalid")
        if self.interval is not None and self.interval[0] > self.interval[1]:
            raise ValueError("decisive_evaluation_v2_bootstrap_interval_order_invalid")
        if self.bootstrap_fraction_delta_gt_zero is not None and not self.metric.endswith("_delta"):
            raise ValueError("decisive_evaluation_v2_positive_fraction_non_delta")
        return self


class FrontierBootstrapV2(_FrozenExactModel):
    method: Literal["paired_or_marginal_domain_stratified_complete_question_bootstrap"] = (
        "paired_or_marginal_domain_stratified_complete_question_bootstrap"
    )
    cluster_unit: Literal["complete_independent_review_question"] = (
        "complete_independent_review_question"
    )
    strata: Literal["declared_scientific_domain"] = "declared_scientific_domain"
    bit_generator: Literal["PCG64"] = "PCG64"
    quantile_method: Literal["linear"] = "linear"
    seed: Annotated[int, Field(ge=0)]
    draws: Annotated[int, Field(ge=100)]
    intervals: list[FrontierBootstrapIntervalV2]
    small_sample_or_finite_sample_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_bootstrap(self) -> FrontierBootstrapV2:
        metrics = [row.metric for row in self.intervals]
        if not metrics or metrics != sorted(set(metrics)):
            raise ValueError("decisive_evaluation_v2_bootstrap_metrics_invalid")
        if any(row.requested_draws != self.draws for row in self.intervals):
            raise ValueError("decisive_evaluation_v2_bootstrap_draws_mismatch")
        return self


class CompiledPolicyPointV2(_FrozenExactModel):
    point_version: Literal["decisive-compiled-policy-point-v2"] = POINT_VERSION
    policy_arm_id: Annotated[str, Field(min_length=1)]
    nominal_budget_minutes_per_question: Annotated[float, Field(ge=0)]
    source_scored_population_sha256: Sha256
    source_frozen_population_sha256: Sha256
    question_ids: list[str]
    metrics: FrontierMetricsV2
    domain_metrics: list[DomainFrontierMetricsV2]
    worst_domain_metrics: WorstDomainMetricsV2
    question_clustered_uncertainty: FrontierBootstrapV2
    calibration_anchor_sha256: Sha256 | None
    typed_calibration_anchor_present: bool
    released_error_claim_authority: bool
    point_sha256: Sha256

    @field_validator("question_ids")
    @classmethod
    def validate_question_ids(cls, values: list[str]) -> list[str]:
        return _sorted_unique(values, "compiled_point_question_ids")

    @model_validator(mode="after")
    def validate_point(self) -> CompiledPolicyPointV2:
        if self.domain_metrics != sorted(self.domain_metrics, key=lambda row: row.domain):
            raise ValueError("decisive_evaluation_v2_domain_metrics_not_canonical")
        if len({row.domain for row in self.domain_metrics}) != len(self.domain_metrics):
            raise ValueError("decisive_evaluation_v2_domain_metrics_duplicate")
        if self.metrics.n_complete_independent_questions != len(self.question_ids):
            raise ValueError("decisive_evaluation_v2_point_question_count_mismatch")
        if self.typed_calibration_anchor_present != (self.calibration_anchor_sha256 is not None):
            raise ValueError("decisive_evaluation_v2_point_calibration_presence_mismatch")
        if self.released_error_claim_authority and not self.typed_calibration_anchor_present:
            raise ValueError("decisive_evaluation_v2_point_authority_without_calibration")
        _self_hash(self, "point_sha256")
        return self


class RealizedCostFrontierRowV2(_FrozenExactModel):
    row_version: Literal["decisive-realized-cost-frontier-row-v2"] = COST_ROW_VERSION
    policy_arm_id: Annotated[str, Field(min_length=1)]
    common_realized_person_minutes_per_question_cutoff: Annotated[float, Field(ge=0)]
    common_total_realized_person_minutes_cutoff: Annotated[float, Field(ge=0)]
    point_available: bool
    selected_point_sha256: Sha256 | None
    selected_nominal_budget_minutes_per_question: float | None
    realized_person_minutes_per_question: float | None
    total_realized_person_minutes: float | None
    unused_person_minutes_per_question: float | None
    exact_realized_spend_equals_cutoff: bool
    point_released_error_claim_authority: bool
    common_ceiling_comparison_not_exact_spend_equality: Literal[True] = True
    row_sha256: Sha256

    @model_validator(mode="after")
    def validate_row(self) -> RealizedCostFrontierRowV2:
        optionals = (
            self.selected_point_sha256,
            self.selected_nominal_budget_minutes_per_question,
            self.realized_person_minutes_per_question,
            self.total_realized_person_minutes,
            self.unused_person_minutes_per_question,
        )
        if self.point_available != all(value is not None for value in optionals):
            raise ValueError("decisive_evaluation_v2_cost_row_presence_mismatch")
        if not self.point_available and (
            self.exact_realized_spend_equals_cutoff or self.point_released_error_claim_authority
        ):
            raise ValueError("decisive_evaluation_v2_unavailable_cost_row_invalid")
        if self.point_available:
            assert self.realized_person_minutes_per_question is not None
            assert self.unused_person_minutes_per_question is not None
            if self.realized_person_minutes_per_question > (
                self.common_realized_person_minutes_per_question_cutoff + _TOLERANCE
            ):
                raise ValueError("decisive_evaluation_v2_cost_ceiling_exceeded")
            expected_unused = (
                self.common_realized_person_minutes_per_question_cutoff
                - self.realized_person_minutes_per_question
            )
            if not math.isclose(
                self.unused_person_minutes_per_question,
                expected_unused,
                rel_tol=1e-12,
                abs_tol=_TOLERANCE,
            ):
                raise ValueError("decisive_evaluation_v2_cost_slack_mismatch")
            expected_exact = math.isclose(expected_unused, 0.0, abs_tol=_TOLERANCE)
            if self.exact_realized_spend_equals_cutoff != expected_exact:
                raise ValueError("decisive_evaluation_v2_exact_spend_flag_mismatch")
        _self_hash(self, "row_sha256")
        return self


class FixedErrorFrontierRowV2(_FrozenExactModel):
    row_version: Literal["decisive-fixed-error-frontier-row-v2"] = FIXED_ERROR_ROW_VERSION
    point_sha256: Sha256
    policy_arm_id: Annotated[str, Field(min_length=1)]
    nominal_budget_minutes_per_question: Annotated[float, Field(ge=0)]
    released_error_ceiling: Annotated[float, Field(gt=0, lt=1)]
    observed_released_claim_error: Annotated[float, Field(ge=0, le=1)] | None
    observed_ceiling_status: Literal["meets", "exceeds", "vacuous_no_releases"]
    observed_ceiling_eligible: bool
    typed_calibration_ceiling_eligible: bool
    nondominated_within_observed_ceiling: bool
    descriptive_test_outcome_frontier_only: Literal[True] = True
    row_sha256: Sha256

    @model_validator(mode="after")
    def validate_row(self) -> FixedErrorFrontierRowV2:
        expected_status = (
            "vacuous_no_releases"
            if self.observed_released_claim_error is None
            else (
                "meets"
                if self.observed_released_claim_error <= self.released_error_ceiling + _TOLERANCE
                else "exceeds"
            )
        )
        if self.observed_ceiling_status != expected_status:
            raise ValueError("decisive_evaluation_v2_fixed_error_status_mismatch")
        if self.observed_ceiling_eligible != (expected_status != "exceeds"):
            raise ValueError("decisive_evaluation_v2_fixed_error_eligibility_mismatch")
        _self_hash(self, "row_sha256")
        return self


class FixedErrorPolicySelectionV2(_FrozenExactModel):
    selection_version: Literal["decisive-fixed-error-policy-selection-v2"] = SELECTION_VERSION
    policy_arm_id: Annotated[str, Field(min_length=1)]
    released_error_ceiling: Annotated[float, Field(gt=0, lt=1)]
    point_available: bool
    selected_point_sha256: Sha256 | None
    selection_basis: Literal[
        "typed_calibration_selected_point",
        "observed_evaluation_error_descriptive",
        "no_point_meets_ceiling",
    ]
    release_coverage: float | None
    correct_releases_per_question: float | None
    mean_realized_person_minutes_per_question: float | None
    released_error_claim_authority: bool
    selection_sha256: Sha256

    @model_validator(mode="after")
    def validate_selection(self) -> FixedErrorPolicySelectionV2:
        optionals = (
            self.selected_point_sha256,
            self.release_coverage,
            self.correct_releases_per_question,
            self.mean_realized_person_minutes_per_question,
        )
        if self.point_available != all(value is not None for value in optionals):
            raise ValueError("decisive_evaluation_v2_selection_presence_mismatch")
        if self.point_available != (self.selection_basis != "no_point_meets_ceiling"):
            raise ValueError("decisive_evaluation_v2_selection_basis_mismatch")
        expected_authority = self.selection_basis == "typed_calibration_selected_point"
        if self.released_error_claim_authority != expected_authority:
            raise ValueError("decisive_evaluation_v2_selection_authority_mismatch")
        _self_hash(self, "selection_sha256")
        return self


class FrontierPairedComparisonV2(_FrozenExactModel):
    comparison_version: Literal["decisive-frontier-paired-comparison-v2"] = COMPARISON_VERSION
    comparison_family: Literal["common_realized_cost_ceiling", "fixed_released_error_ceiling"]
    comparison_id: Annotated[str, Field(min_length=1)]
    primary_policy_arm_id: Annotated[str, Field(min_length=1)]
    baseline_policy_arm_id: Annotated[str, Field(min_length=1)]
    common_cutoff_or_error_ceiling: Annotated[float, Field(ge=0)]
    primary_point_sha256: Sha256
    baseline_point_sha256: Sha256
    question_ids: list[str]
    identical_complete_question_population: Literal[True] = True
    identical_pipeline_and_provenance: Literal[True] = True
    identical_common_resource_or_error_constraint: Literal[True] = True
    exact_realized_spend_match: bool
    primary_minus_baseline_point_deltas: dict[str, float | None]
    paired_question_clustered_uncertainty: FrontierBootstrapV2
    released_error_claim_authority: bool
    comparison_sha256: Sha256

    @field_validator("question_ids")
    @classmethod
    def validate_question_ids(cls, values: list[str]) -> list[str]:
        return _sorted_unique(values, "comparison_question_ids")

    @model_validator(mode="after")
    def validate_comparison(self) -> FrontierPairedComparisonV2:
        if sorted(self.primary_minus_baseline_point_deltas) != sorted(_DELTA_METRICS):
            raise ValueError("decisive_evaluation_v2_comparison_delta_roster_mismatch")
        if sorted(row.metric for row in self.paired_question_clustered_uncertainty.intervals) != (
            sorted(_DELTA_METRICS)
        ):
            raise ValueError("decisive_evaluation_v2_comparison_interval_roster_mismatch")
        _self_hash(self, "comparison_sha256")
        return self


class DecisiveClaimEvaluationFrontiersV2(_FrozenExactModel):
    result_version: Literal["decisive-claim-evaluation-frontiers-v2"] = RESULT_VERSION
    evaluator_component_sha256: Sha256
    config: DecisiveFrontierConfigV2
    source_anchor: DecisiveFrontierSourceAnchorV2
    calibration_anchors: list[DecisiveFrontierCalibrationAnchorV2]
    compiled_policy_points: list[CompiledPolicyPointV2]
    realized_cost_frontier: list[RealizedCostFrontierRowV2]
    realized_cost_paired_comparisons: list[FrontierPairedComparisonV2]
    fixed_error_frontier: list[FixedErrorFrontierRowV2]
    fixed_error_policy_selections: list[FixedErrorPolicySelectionV2]
    fixed_error_paired_comparisons: list[FrontierPairedComparisonV2]
    metric_definitions: dict[str, str]
    input_labels_or_private_files_opened_by_v2: Literal[False] = False
    policy_trajectories_rerun_by_v2: Literal[False] = False
    simulation_or_fixture_inputs_accepted: Literal[False] = False
    equal_nominal_deadline_misreported_as_equal_realized_cost: Literal[False] = False
    realized_cost_frontier_claim_authority: bool
    fixed_error_frontier_claim_authority: bool
    scientific_claim_eligible: bool
    claim_release_authority: Literal[False] = False
    causal_or_prospective_authority: Literal[False] = False
    small_sample_or_finite_sample_authority: Literal[False] = False
    authority_blockers: list[str]
    result_sha256: Sha256

    @field_validator("authority_blockers")
    @classmethod
    def validate_blockers(cls, values: list[str]) -> list[str]:
        if values != sorted(set(values)) or any(not value for value in values):
            raise ValueError("decisive_evaluation_v2_authority_blockers_invalid")
        return values

    @model_validator(mode="after")
    def validate_result(self) -> DecisiveClaimEvaluationFrontiersV2:
        if self.calibration_anchors != sorted(
            self.calibration_anchors,
            key=lambda row: (row.policy_arm_id, row.budget_minutes_per_question),
        ):
            raise ValueError("decisive_evaluation_v2_calibration_anchors_not_canonical")
        point_keys = [
            (row.policy_arm_id, row.nominal_budget_minutes_per_question)
            for row in self.compiled_policy_points
        ]
        if not point_keys or point_keys != sorted(set(point_keys)):
            raise ValueError("decisive_evaluation_v2_points_not_canonical")
        arms = sorted({row.policy_arm_id for row in self.compiled_policy_points})
        if self.config.primary_policy_arm_id not in arms:
            raise ValueError("decisive_evaluation_v2_primary_policy_point_missing")
        if any(
            row.question_ids != self.source_anchor.evaluation_question_ids
            for row in self.compiled_policy_points
        ):
            raise ValueError("decisive_evaluation_v2_point_population_anchor_mismatch")

        anchor_by_sha = {row.anchor_sha256: row for row in self.calibration_anchors}
        if len(anchor_by_sha) != len(self.calibration_anchors) or len(
            {
                (row.policy_arm_id, row.budget_minutes_per_question)
                for row in self.calibration_anchors
            }
        ) != len(self.calibration_anchors):
            raise ValueError("decisive_evaluation_v2_calibration_anchor_duplicate")
        referenced_anchor_sha256s: set[str] = set()
        for point in self.compiled_policy_points:
            anchor = (
                None
                if point.calibration_anchor_sha256 is None
                else anchor_by_sha.get(point.calibration_anchor_sha256)
            )
            if point.typed_calibration_anchor_present:
                if (
                    anchor is None
                    or anchor.policy_arm_id != point.policy_arm_id
                    or not math.isclose(
                        anchor.budget_minutes_per_question,
                        point.nominal_budget_minutes_per_question,
                        rel_tol=1e-12,
                        abs_tol=_TOLERANCE,
                    )
                    or point.released_error_claim_authority != anchor.point_claim_authority
                ):
                    raise ValueError("decisive_evaluation_v2_point_calibration_anchor_mismatch")
                referenced_anchor_sha256s.add(anchor.anchor_sha256)
            elif anchor is not None:
                raise ValueError("decisive_evaluation_v2_untyped_point_references_calibration")
        if referenced_anchor_sha256s != set(anchor_by_sha):
            raise ValueError("decisive_evaluation_v2_calibration_anchor_unreferenced")

        expected_cost_rows = _cost_rows(
            points=self.compiled_policy_points,
            config=self.config,
        )
        if self.realized_cost_frontier != expected_cost_rows:
            raise ValueError("decisive_evaluation_v2_cost_frontier_replay_mismatch")
        expected_fixed_rows = _fixed_error_rows(
            points=self.compiled_policy_points,
            config=self.config,
        )
        if self.fixed_error_frontier != expected_fixed_rows:
            raise ValueError("decisive_evaluation_v2_fixed_frontier_replay_mismatch")
        expected_fixed_selections = _fixed_error_selections(
            points=self.compiled_policy_points,
            rows=expected_fixed_rows,
            config=self.config,
        )
        if self.fixed_error_policy_selections != expected_fixed_selections:
            raise ValueError("decisive_evaluation_v2_fixed_selection_replay_mismatch")

        points_by_sha = {row.point_sha256: row for row in self.compiled_policy_points}
        cost_rows_by_key = {
            (row.policy_arm_id, row.common_realized_person_minutes_per_question_cutoff): row
            for row in self.realized_cost_frontier
        }
        fixed_selection_by_arm = {
            row.policy_arm_id: row for row in self.fixed_error_policy_selections
        }

        def comparison_key(
            comparison: FrontierPairedComparisonV2,
        ) -> tuple[float, str]:
            return (
                comparison.common_cutoff_or_error_ceiling,
                comparison.baseline_policy_arm_id,
            )

        expected_cost_keys: list[tuple[float, str]] = []
        for cutoff in self.config.common_realized_person_minutes_per_question_cutoffs:
            primary_row = cost_rows_by_key[(self.config.primary_policy_arm_id, cutoff)]
            if not primary_row.point_available:
                continue
            expected_cost_keys.extend(
                (cutoff, arm)
                for arm in arms
                if arm != self.config.primary_policy_arm_id
                and cost_rows_by_key[(arm, cutoff)].point_available
            )
        if [comparison_key(row) for row in self.realized_cost_paired_comparisons] != (
            expected_cost_keys
        ):
            raise ValueError("decisive_evaluation_v2_cost_comparison_roster_mismatch")

        primary_selection = fixed_selection_by_arm[self.config.primary_policy_arm_id]
        expected_fixed_keys = (
            []
            if not primary_selection.point_available
            else [
                (self.config.released_error_ceiling, arm)
                for arm in arms
                if arm != self.config.primary_policy_arm_id
                and fixed_selection_by_arm[arm].point_available
            ]
        )
        if [comparison_key(row) for row in self.fixed_error_paired_comparisons] != (
            expected_fixed_keys
        ):
            raise ValueError("decisive_evaluation_v2_fixed_comparison_roster_mismatch")

        for family, comparisons in (
            ("common_realized_cost_ceiling", self.realized_cost_paired_comparisons),
            ("fixed_released_error_ceiling", self.fixed_error_paired_comparisons),
        ):
            for comparison in comparisons:
                primary = points_by_sha.get(comparison.primary_point_sha256)
                baseline = points_by_sha.get(comparison.baseline_point_sha256)
                if (
                    comparison.comparison_family != family
                    or primary is None
                    or baseline is None
                    or comparison.primary_policy_arm_id != self.config.primary_policy_arm_id
                    or primary.policy_arm_id != comparison.primary_policy_arm_id
                    or baseline.policy_arm_id != comparison.baseline_policy_arm_id
                    or comparison.question_ids != primary.question_ids
                    or comparison.question_ids != baseline.question_ids
                ):
                    raise ValueError("decisive_evaluation_v2_comparison_point_binding_mismatch")
                expected_exact = math.isclose(
                    primary.metrics.total_realized_person_minutes,
                    baseline.metrics.total_realized_person_minutes,
                    rel_tol=1e-12,
                    abs_tol=_TOLERANCE,
                )
                if comparison.exact_realized_spend_match != expected_exact:
                    raise ValueError("decisive_evaluation_v2_comparison_exact_spend_mismatch")
                if family == "common_realized_cost_ceiling":
                    cutoff = comparison.common_cutoff_or_error_ceiling
                    primary_row = cost_rows_by_key[(primary.policy_arm_id, cutoff)]
                    baseline_row = cost_rows_by_key[(baseline.policy_arm_id, cutoff)]
                    expected_authority = (
                        primary.released_error_claim_authority
                        and baseline.released_error_claim_authority
                    )
                    selected_hashes = (
                        primary_row.selected_point_sha256,
                        baseline_row.selected_point_sha256,
                    )
                else:
                    expected_authority = (
                        fixed_selection_by_arm[primary.policy_arm_id].released_error_claim_authority
                        and fixed_selection_by_arm[
                            baseline.policy_arm_id
                        ].released_error_claim_authority
                    )
                    selected_hashes = (
                        fixed_selection_by_arm[primary.policy_arm_id].selected_point_sha256,
                        fixed_selection_by_arm[baseline.policy_arm_id].selected_point_sha256,
                    )
                if (
                    selected_hashes != (primary.point_sha256, baseline.point_sha256)
                    or comparison.released_error_claim_authority != expected_authority
                ):
                    raise ValueError("decisive_evaluation_v2_comparison_authority_binding_mismatch")

        expected_cost_authority = (
            bool(self.realized_cost_paired_comparisons)
            and all(
                row.released_error_claim_authority for row in self.realized_cost_paired_comparisons
            )
            and len(self.realized_cost_paired_comparisons)
            == len(self.config.common_realized_person_minutes_per_question_cutoffs)
            * (len(arms) - 1)
        )
        expected_fixed_authority = (
            bool(self.fixed_error_paired_comparisons)
            and all(
                row.released_error_claim_authority for row in self.fixed_error_paired_comparisons
            )
            and len(self.fixed_error_paired_comparisons) == len(arms) - 1
        )
        if (
            self.realized_cost_frontier_claim_authority != expected_cost_authority
            or self.fixed_error_frontier_claim_authority != expected_fixed_authority
        ):
            raise ValueError("decisive_evaluation_v2_frontier_authority_mismatch")
        expected_blockers: list[str] = []
        if not self.calibration_anchors:
            expected_blockers.append("real_typed_complete_question_calibration_bundles_missing")
        if any(not row.point_claim_authority for row in self.calibration_anchors):
            expected_blockers.append(
                "typed_calibration_not_applied_to_exact_terminal_policy_points"
            )
        if not expected_cost_authority:
            expected_blockers.append(
                "common_realized_cost_frontier_contains_uncalibrated_comparisons"
            )
        expected_cost_count = len(
            self.config.common_realized_person_minutes_per_question_cutoffs
        ) * (len(arms) - 1)
        if len(self.realized_cost_paired_comparisons) != expected_cost_count:
            expected_blockers.append("common_realized_cost_frontier_comparison_roster_incomplete")
        if not expected_fixed_authority:
            expected_blockers.append(
                "fixed_error_frontier_contains_descriptive_or_uncalibrated_comparisons"
            )
        if len(self.fixed_error_paired_comparisons) != len(arms) - 1:
            expected_blockers.append("fixed_error_frontier_comparison_roster_incomplete")
        if self.authority_blockers != sorted(expected_blockers):
            raise ValueError("decisive_evaluation_v2_authority_blocker_mismatch")
        expected_scientific = (
            self.realized_cost_frontier_claim_authority
            and self.fixed_error_frontier_claim_authority
            and not self.authority_blockers
        )
        if self.scientific_claim_eligible != expected_scientific:
            raise ValueError("decisive_evaluation_v2_scientific_authority_mismatch")
        _self_hash(self, "result_sha256")
        return self


def _metrics(outcomes: Sequence[ScoredQuestionOutcomeV1]) -> FrontierMetricsV2:
    if not outcomes:
        raise DecisiveClaimEvaluationV2Error("decisive_evaluation_v2_empty_outcomes")
    n = len(outcomes)
    released = sum(row.released for row in outcomes)
    errors = sum(row.released_claim_error for row in outcomes)
    correct = sum(row.correct_release for row in outcomes)
    spent = math.fsum(row.realized_minutes for row in outcomes)
    return FrontierMetricsV2(
        n_complete_independent_questions=n,
        total_realized_person_minutes=spent,
        mean_realized_person_minutes_per_question=spent / n,
        released_claims=released,
        released_claim_errors=errors,
        correct_releases=correct,
        release_coverage=released / n,
        released_claim_error=errors / released if released else None,
        correct_releases_per_question=correct / n,
        correct_releases_per_human_hour=correct * 60.0 / spent if spent > 0 else None,
    )


def _domain_metrics(
    outcomes: Sequence[ScoredQuestionOutcomeV1],
) -> list[DomainFrontierMetricsV2]:
    grouped: dict[str, list[ScoredQuestionOutcomeV1]] = defaultdict(list)
    for row in outcomes:
        grouped[row.domain].append(row)
    return [
        DomainFrontierMetricsV2(domain=domain, metrics=_metrics(rows))
        for domain, rows in sorted(grouped.items())
    ]


def _worst_domain(rows: Sequence[DomainFrontierMetricsV2]) -> WorstDomainMetricsV2:
    if not rows:
        raise DecisiveClaimEvaluationV2Error("decisive_evaluation_v2_domains_empty")

    def minimum(name: str) -> tuple[str, float]:
        values = [(row.domain, float(getattr(row.metrics, name))) for row in rows]
        return min(values, key=lambda value: (value[1], value[0]))

    coverage_domain, coverage = minimum("release_coverage")
    correct_domain, correct = minimum("correct_releases_per_question")
    errors = [
        (row.domain, row.metrics.released_claim_error)
        for row in rows
        if row.metrics.released_claim_error is not None
    ]
    error_domain, error = (
        max(errors, key=lambda value: (float(value[1]), value[0])) if errors else (None, None)
    )
    efficiencies = [
        (row.domain, row.metrics.correct_releases_per_human_hour)
        for row in rows
        if row.metrics.correct_releases_per_human_hour is not None
    ]
    efficiency_domain, efficiency = (
        min(efficiencies, key=lambda value: (float(value[1]), value[0]))
        if efficiencies
        else (None, None)
    )
    return WorstDomainMetricsV2(
        release_coverage_domain=coverage_domain,
        release_coverage=coverage,
        released_claim_error_domain=error_domain,
        released_claim_error=error,
        correct_releases_per_question_domain=correct_domain,
        correct_releases_per_question=correct,
        correct_releases_per_human_hour_domain=efficiency_domain,
        correct_releases_per_human_hour=efficiency,
        domains_without_releases=sorted(
            row.domain for row in rows if row.metrics.released_claim_error is None
        ),
    )


def _metric_vector(outcomes: Sequence[ScoredQuestionOutcomeV1]) -> dict[str, float | None]:
    aggregate = _metrics(outcomes)
    worst = _worst_domain(_domain_metrics(outcomes))
    return {
        "release_coverage": aggregate.release_coverage,
        "released_claim_error": aggregate.released_claim_error,
        "correct_releases_per_question": aggregate.correct_releases_per_question,
        "correct_releases_per_human_hour": aggregate.correct_releases_per_human_hour,
        "worst_domain_release_coverage": worst.release_coverage,
        "worst_domain_released_claim_error": worst.released_claim_error,
        "worst_domain_correct_releases_per_question": worst.correct_releases_per_question,
        "worst_domain_correct_releases_per_human_hour": (worst.correct_releases_per_human_hour),
    }


def _interval_rows(
    values: Mapping[str, Sequence[float]], *, draws: int, deltas: bool
) -> list[FrontierBootstrapIntervalV2]:
    rows: list[FrontierBootstrapIntervalV2] = []
    for metric in sorted(values):
        valid = list(values[metric])
        interval = None
        fraction = None
        if valid:
            lower, upper = np.quantile(valid, [0.025, 0.975], method="linear")
            interval = [float(lower), float(upper)]
            if deltas:
                fraction = sum(value > 0 for value in valid) / len(valid)
        rows.append(
            FrontierBootstrapIntervalV2(
                metric=metric,
                interval=interval,
                valid_draws=len(valid),
                requested_draws=draws,
                undefined_draws=draws - len(valid),
                bootstrap_fraction_delta_gt_zero=fraction,
            )
        )
    return rows


def _resample_indices(
    outcomes: Sequence[ScoredQuestionOutcomeV1], rng: np.random.Generator
) -> list[int]:
    by_domain: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(outcomes):
        by_domain[row.domain].append(index)
    indices: list[int] = []
    for domain in sorted(by_domain):
        members = by_domain[domain]
        sampled = rng.integers(0, len(members), size=len(members))
        indices.extend(members[int(index)] for index in sampled)
    return indices


def _marginal_bootstrap(
    outcomes: Sequence[ScoredQuestionOutcomeV1], *, draws: int, seed: int
) -> FrontierBootstrapV2:
    values: dict[str, list[float]] = {name: [] for name in _INTERVAL_METRICS}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        indices = _resample_indices(outcomes, rng)
        vector = _metric_vector([outcomes[index] for index in indices])
        for name, value in vector.items():
            if value is not None and math.isfinite(value):
                values[name].append(float(value))
    return FrontierBootstrapV2(
        seed=seed,
        draws=draws,
        intervals=_interval_rows(values, draws=draws, deltas=False),
    )


def _paired_deltas(
    primary: Sequence[ScoredQuestionOutcomeV1],
    baseline: Sequence[ScoredQuestionOutcomeV1],
) -> dict[str, float | None]:
    if [row.question_id for row in primary] != [row.question_id for row in baseline]:
        raise DecisiveClaimEvaluationV2Error(
            "decisive_evaluation_v2_paired_question_population_mismatch"
        )
    if [row.domain for row in primary] != [row.domain for row in baseline]:
        raise DecisiveClaimEvaluationV2Error("decisive_evaluation_v2_paired_domain_mismatch")
    left = _metric_vector(primary)
    right = _metric_vector(baseline)
    return {
        f"{name}_delta": (
            None
            if left[name] is None or right[name] is None
            else float(left[name]) - float(right[name])
        )
        for name in _INTERVAL_METRICS
    }


def _paired_bootstrap(
    primary: Sequence[ScoredQuestionOutcomeV1],
    baseline: Sequence[ScoredQuestionOutcomeV1],
    *,
    draws: int,
    seed: int,
) -> FrontierBootstrapV2:
    if [row.question_id for row in primary] != [row.question_id for row in baseline]:
        raise DecisiveClaimEvaluationV2Error(
            "decisive_evaluation_v2_paired_question_population_mismatch"
        )
    values: dict[str, list[float]] = {name: [] for name in _DELTA_METRICS}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        indices = _resample_indices(primary, rng)
        deltas = _paired_deltas(
            [primary[index] for index in indices],
            [baseline[index] for index in indices],
        )
        for name, value in deltas.items():
            if value is not None and math.isfinite(value):
                values[name].append(value)
    return FrontierBootstrapV2(
        seed=seed,
        draws=draws,
        intervals=_interval_rows(values, draws=draws, deltas=True),
    )


def _validate_source(
    source: DecisiveClaimEvaluationResultV1,
    *,
    config: DecisiveFrontierConfigV2,
) -> tuple[DecisiveClaimEvaluationResultV1, DecisiveFrontierSourceAnchorV2]:
    try:
        canonical = DecisiveClaimEvaluationResultV1.model_validate(source.model_dump(mode="json"))
    except (AttributeError, ValueError) as exc:
        raise DecisiveClaimEvaluationV2Error(
            "decisive_evaluation_v2_source_result_invalid"
        ) from exc
    freeze = canonical.policy_freeze
    if canonical.evidence_kind is not BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED:
        raise DecisiveClaimEvaluationV2Error(
            "decisive_evaluation_v2_simulation_or_fixture_input_rejected"
        )
    if freeze.trajectory_bundle.evidence_kind is not BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED:
        raise DecisiveClaimEvaluationV2Error("decisive_evaluation_v2_evidence_kind_mismatch")
    question_ids = freeze.evaluation_question_ids
    if len(question_ids) < config.minimum_complete_evaluation_questions:
        raise DecisiveClaimEvaluationV2Error(
            "decisive_evaluation_v2_complete_question_population_too_small"
        )
    all_identities = freeze.split_manifest.identities
    identities = [row for row in all_identities if row.question_id in set(question_ids)]
    populations = sorted({row.population_id for row in identities})
    domains = sorted({row.domain for row in identities})
    counts = {domain: sum(row.domain == domain for row in identities) for domain in domains}
    if any(value < config.minimum_questions_per_domain for value in counts.values()):
        raise DecisiveClaimEvaluationV2Error(
            "decisive_evaluation_v2_domain_complete_question_population_too_small"
        )
    if any(
        row.evidence_kind is not BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED
        or any(
            state.replay_source is not ReplaySource.FROZEN_PIPELINE_RERUN
            for state in row.replay_states
        )
        for row in freeze.trajectory_bundle.trajectories
    ):
        raise DecisiveClaimEvaluationV2Error(
            "decisive_evaluation_v2_nonproduction_trajectory_rejected"
        )
    if any(
        label.reference_verdict.source is not ReferenceVerdictSource.EXPERT_ADJUDICATION
        for label in canonical.opened_labels
    ):
        raise DecisiveClaimEvaluationV2Error("decisive_evaluation_v2_nonexpert_reference_rejected")
    if {
        freeze.development_receipt.label_source,
        freeze.calibration_receipt.label_source,
    } != {"expert_adjudication"}:
        raise DecisiveClaimEvaluationV2Error(
            "decisive_evaluation_v2_nonexpert_fit_stage_provenance_rejected"
        )
    protocols = {row.reference_verdict.protocol_sha256 for row in canonical.opened_labels}
    if len(protocols) != 1:
        raise DecisiveClaimEvaluationV2Error("decisive_evaluation_v2_adjudication_provenance_mixed")
    frozen_by_key = {
        (row.policy_arm.arm_id, row.budget_minutes): row for row in freeze.policy_populations
    }
    labels = {row.question_id: row for row in canonical.opened_labels}
    if set(labels) != set(question_ids):
        raise DecisiveClaimEvaluationV2Error("decisive_evaluation_v2_reference_population_mismatch")
    for scored in canonical.scored_policy_populations:
        frozen_population = frozen_by_key.get((scored.policy_arm.arm_id, scored.budget_minutes))
        if frozen_population is None or scored.question_ids != question_ids:
            raise DecisiveClaimEvaluationV2Error(
                "decisive_evaluation_v2_scored_population_provenance_unequal"
            )
        expected = [
            _score_question_v1(row, labels[row.question_id]) for row in frozen_population.questions
        ]
        if scored.outcomes != expected:
            raise DecisiveClaimEvaluationV2Error(
                "decisive_evaluation_v2_scored_outcome_external_replay_mismatch"
            )
    payload: dict[str, Any] = {
        "anchor_version": SOURCE_ANCHOR_VERSION,
        "source_result_version": canonical.result_version,
        "source_result_sha256": canonical.result_sha256,
        "source_policy_freeze_sha256": freeze.freeze_sha256,
        "source_component_sha256": canonical.component_sha256,
        "split_manifest_sha256": freeze.split_manifest.manifest_sha256,
        "trajectory_bundle_sha256": freeze.trajectory_bundle.bundle_sha256,
        "opened_label_membership_sha256": canonical.opened_label_membership_sha256,
        "scored_population_membership_sha256": hash_canonical(
            [row.scored_population_sha256 for row in canonical.scored_policy_populations]
        ),
        "pipeline_sha256": freeze.split_manifest.pipeline_sha256,
        "question_population_ids": populations,
        "question_population_membership_sha256": hash_canonical(
            [
                {"question_id": row.question_id, "population_id": row.population_id}
                for row in sorted(identities, key=lambda item: item.question_id)
            ]
        ),
        "adjudication_protocol_sha256": next(iter(protocols)),
        "evidence_kind": "real_expert_adjudicated",
        "evaluation_question_ids": question_ids,
        "domains": domains,
        "complete_question_inputs_structurally_replayed_from_v1": True,
        "identical_question_population_across_policies": True,
        "identical_pipeline_across_policies": True,
        "simulation_or_fixture_input": False,
    }
    anchor = DecisiveFrontierSourceAnchorV2.model_validate(
        {**payload, "anchor_sha256": hash_canonical(payload)}
    )
    return canonical, anchor


def _terminal_bundle_sha256(
    *, source: DecisiveClaimEvaluationResultV1, population: FrozenPolicyPopulationV1
) -> set[str | None]:
    trajectories = {
        row.question_identity.question_id: row
        for row in source.policy_freeze.trajectory_bundle.trajectories
    }
    result: set[str | None] = set()
    for question in population.questions:
        trajectory = trajectories[question.question_id]
        state = next(
            row
            for row in trajectory.replay_states
            if row.replay_sha256 == question.final_replay_sha256
        )
        binding = state.production_binding
        if isinstance(binding, ConditionProductionReplayBindingV7):
            result.add(
                binding.certificate.source_certificate_v6.adaptive_calibration_bundle_v2.bundle_sha256
            )
        else:
            result.add(None)
    return result


def _calibration_anchors(
    *,
    source: DecisiveClaimEvaluationResultV1,
    source_anchor: DecisiveFrontierSourceAnchorV2,
    config: DecisiveFrontierConfigV2,
    bundles: Sequence[AdaptiveCalibrationBundleV2],
) -> list[DecisiveFrontierCalibrationAnchorV2]:
    populations = {
        (row.policy_arm.arm_id, row.budget_minutes): row
        for row in source.policy_freeze.policy_populations
        if row.budget_minutes is not None
    }
    evaluation_ids = set(source_anchor.evaluation_question_ids)
    anchors: list[DecisiveFrontierCalibrationAnchorV2] = []
    seen_keys: set[tuple[str, float]] = set()
    for supplied in bundles:
        try:
            bundle = validate_adaptive_calibration_bundle_v2_integrity(supplied)
        except (AttributeError, ValueError) as exc:
            raise DecisiveClaimEvaluationV2Error(
                "decisive_evaluation_v2_calibration_bundle_invalid"
            ) from exc
        if (
            bundle.label_source != "expert_adjudication"
            or not bundle.real_release_eligible
            or not bundle.independence_verified
            or bundle.selected is None
        ):
            raise DecisiveClaimEvaluationV2Error(
                "decisive_evaluation_v2_nonreal_or_uncalibrated_bundle_rejected"
            )
        candidate = bundle.selected.candidate
        contexts = {
            row.policy_arm_id: row for row in bundle.development_freeze.base_freeze.policy_contexts
        }
        context = contexts.get(candidate.policy_arm_id)
        if context is None or candidate.policy_context_sha256 != context.policy_context_sha256:
            raise DecisiveClaimEvaluationV2Error(
                "decisive_evaluation_v2_calibration_context_missing"
            )
        key = (candidate.policy_arm_id, float(context.budget_minutes))
        if key in seen_keys:
            raise DecisiveClaimEvaluationV2Error(
                "decisive_evaluation_v2_calibration_point_duplicate"
            )
        seen_keys.add(key)
        population = populations.get(key)
        if population is None:
            raise DecisiveClaimEvaluationV2Error(
                "decisive_evaluation_v2_calibration_policy_or_budget_provenance_unequal"
            )
        pipeline_values = {row.pipeline_sha256 for row in contexts.values()}
        if pipeline_values != {source_anchor.pipeline_sha256}:
            raise DecisiveClaimEvaluationV2Error(
                "decisive_evaluation_v2_calibration_pipeline_provenance_unequal"
            )
        if bundle.population_id != context.population_id:
            raise DecisiveClaimEvaluationV2Error(
                "decisive_evaluation_v2_calibration_population_provenance_unequal"
            )
        protocol_context = context.corpus_protocol_context
        required_population_bridge = {
            "decisive_split_manifest_sha256": (source.policy_freeze.split_manifest.manifest_sha256),
            "decisive_identity_membership_sha256": (
                source.policy_freeze.split_manifest.identity_membership_sha256
            ),
            "decisive_evaluation_population_membership_sha256": (
                source_anchor.question_population_membership_sha256
            ),
        }
        if any(
            protocol_context.get(name) != value
            for name, value in required_population_bridge.items()
        ):
            raise DecisiveClaimEvaluationV2Error(
                "decisive_evaluation_v2_calibration_population_bridge_missing_or_unequal"
            )
        if bundle.adjudication_protocol_sha256 != source_anchor.adjudication_protocol_sha256:
            raise DecisiveClaimEvaluationV2Error(
                "decisive_evaluation_v2_calibration_adjudication_provenance_unequal"
            )
        if bundle.calibration.domains != source_anchor.domains:
            raise DecisiveClaimEvaluationV2Error(
                "decisive_evaluation_v2_calibration_domain_provenance_unequal"
            )
        development_ids = set(bundle.development_freeze.base_freeze.development.question_ids)
        calibration_ids = set(bundle.calibration.question_ids)
        if evaluation_ids & (development_ids | calibration_ids):
            raise DecisiveClaimEvaluationV2Error(
                "decisive_evaluation_v2_calibration_evaluation_question_overlap"
            )
        if (
            sorted(development_ids) != source.policy_freeze.split_manifest.development_question_ids
            or sorted(calibration_ids)
            != source.policy_freeze.split_manifest.calibration_question_ids
        ):
            raise DecisiveClaimEvaluationV2Error(
                "decisive_evaluation_v2_calibration_split_roster_provenance_unequal"
            )
        embedded = _terminal_bundle_sha256(source=source, population=population) == {
            bundle.bundle_sha256
        }
        blockers: list[str] = []
        if not embedded:
            blockers.append("typed_calibration_not_embedded_in_every_terminal_replay")
        if bundle.alpha > config.released_error_ceiling + _TOLERANCE:
            blockers.append("calibrated_upper_error_bound_exceeds_predeclared_ceiling")
        payload: dict[str, Any] = {
            "anchor_version": CALIBRATION_ANCHOR_VERSION,
            "bundle_sha256": bundle.bundle_sha256,
            "selected_candidate_sha256": candidate.candidate_sha256,
            "policy_context_sha256": context.policy_context_sha256,
            "policy_arm_id": candidate.policy_arm_id,
            "budget_minutes_per_question": context.budget_minutes,
            "pipeline_sha256": context.pipeline_sha256,
            "population_id": bundle.population_id,
            "adjudication_protocol_sha256": bundle.adjudication_protocol_sha256,
            "alpha": bundle.alpha,
            "delta": bundle.delta,
            "development_question_count": len(
                bundle.development_freeze.base_freeze.development.question_ids
            ),
            "calibration_question_count": len(bundle.calibration.question_ids),
            "calibration_domains": bundle.calibration.domains,
            "label_source": "expert_adjudication",
            "independence_verified": True,
            "real_release_eligible": True,
            "exact_bundle_embedded_in_every_terminal_replay": embedded,
            "released_error_ceiling_passed": (
                bundle.alpha <= config.released_error_ceiling + _TOLERANCE
            ),
            "point_claim_authority": not blockers,
            "authority_blockers": sorted(blockers),
        }
        anchors.append(
            DecisiveFrontierCalibrationAnchorV2.model_validate(
                {**payload, "anchor_sha256": hash_canonical(payload)}
            )
        )
    return sorted(anchors, key=lambda row: (row.policy_arm_id, row.budget_minutes_per_question))


def _compile_points(
    *,
    source: DecisiveClaimEvaluationResultV1,
    anchors: Sequence[DecisiveFrontierCalibrationAnchorV2],
    config: DecisiveFrontierConfigV2,
) -> tuple[list[CompiledPolicyPointV2], dict[str, Sequence[ScoredQuestionOutcomeV1]]]:
    anchor_by_key = {(row.policy_arm_id, row.budget_minutes_per_question): row for row in anchors}
    points: list[CompiledPolicyPointV2] = []
    outcomes_by_sha: dict[str, Sequence[ScoredQuestionOutcomeV1]] = {}
    for population in source.scored_policy_populations:
        if population.budget_minutes is None:
            continue
        anchor = anchor_by_key.get((population.policy_arm.arm_id, population.budget_minutes))
        metrics = _metrics(population.outcomes)
        domains = _domain_metrics(population.outcomes)
        payload: dict[str, Any] = {
            "point_version": POINT_VERSION,
            "policy_arm_id": population.policy_arm.arm_id,
            "nominal_budget_minutes_per_question": population.budget_minutes,
            "source_scored_population_sha256": population.scored_population_sha256,
            "source_frozen_population_sha256": population.frozen_population_sha256,
            "question_ids": population.question_ids,
            "metrics": metrics,
            "domain_metrics": domains,
            "worst_domain_metrics": _worst_domain(domains),
            "question_clustered_uncertainty": _marginal_bootstrap(
                population.outcomes,
                draws=config.bootstrap_draws,
                seed=_seed(
                    config.bootstrap_seed,
                    "point",
                    population.policy_arm.arm_id,
                    population.budget_minutes,
                ),
            ),
            "calibration_anchor_sha256": None if anchor is None else anchor.anchor_sha256,
            "typed_calibration_anchor_present": anchor is not None,
            "released_error_claim_authority": (
                False if anchor is None else anchor.point_claim_authority
            ),
        }
        point = CompiledPolicyPointV2.model_validate(
            {**payload, "point_sha256": hash_canonical(payload)}
        )
        points.append(point)
        outcomes_by_sha[point.point_sha256] = population.outcomes
    points.sort(key=lambda row: (row.policy_arm_id, row.nominal_budget_minutes_per_question))
    return points, outcomes_by_sha


def _cost_rows(
    *, points: Sequence[CompiledPolicyPointV2], config: DecisiveFrontierConfigV2
) -> list[RealizedCostFrontierRowV2]:
    by_arm: dict[str, list[CompiledPolicyPointV2]] = defaultdict(list)
    for point in points:
        by_arm[point.policy_arm_id].append(point)
    rows: list[RealizedCostFrontierRowV2] = []
    for cutoff in config.common_realized_person_minutes_per_question_cutoffs:
        for arm in sorted(by_arm):
            candidates = [
                row
                for row in by_arm[arm]
                if row.metrics.mean_realized_person_minutes_per_question <= cutoff + _TOLERANCE
            ]
            selected = (
                max(
                    candidates,
                    key=lambda row: (
                        row.metrics.mean_realized_person_minutes_per_question,
                        row.nominal_budget_minutes_per_question,
                    ),
                )
                if candidates
                else None
            )
            n = by_arm[arm][0].metrics.n_complete_independent_questions
            realized = (
                None
                if selected is None
                else selected.metrics.mean_realized_person_minutes_per_question
            )
            payload: dict[str, Any] = {
                "row_version": COST_ROW_VERSION,
                "policy_arm_id": arm,
                "common_realized_person_minutes_per_question_cutoff": cutoff,
                "common_total_realized_person_minutes_cutoff": cutoff * n,
                "point_available": selected is not None,
                "selected_point_sha256": None if selected is None else selected.point_sha256,
                "selected_nominal_budget_minutes_per_question": (
                    None if selected is None else selected.nominal_budget_minutes_per_question
                ),
                "realized_person_minutes_per_question": realized,
                "total_realized_person_minutes": (
                    None if selected is None else selected.metrics.total_realized_person_minutes
                ),
                "unused_person_minutes_per_question": (
                    None if realized is None else cutoff - realized
                ),
                "exact_realized_spend_equals_cutoff": (
                    False
                    if realized is None
                    else math.isclose(cutoff, realized, abs_tol=_TOLERANCE)
                ),
                "point_released_error_claim_authority": (
                    False if selected is None else selected.released_error_claim_authority
                ),
                "common_ceiling_comparison_not_exact_spend_equality": True,
            }
            rows.append(
                RealizedCostFrontierRowV2.model_validate(
                    {**payload, "row_sha256": hash_canonical(payload)}
                )
            )
    return rows


def _comparison(
    *,
    family: Literal["common_realized_cost_ceiling", "fixed_released_error_ceiling"],
    constraint: float,
    primary: CompiledPolicyPointV2,
    baseline: CompiledPolicyPointV2,
    outcomes_by_sha: Mapping[str, Sequence[ScoredQuestionOutcomeV1]],
    config: DecisiveFrontierConfigV2,
    authority: bool,
) -> FrontierPairedComparisonV2:
    primary_outcomes = outcomes_by_sha[primary.point_sha256]
    baseline_outcomes = outcomes_by_sha[baseline.point_sha256]
    comparison_id = (
        f"{family}__{primary.policy_arm_id}__minus__{baseline.policy_arm_id}"
        f"__constraint_{constraint:g}"
    )
    payload: dict[str, Any] = {
        "comparison_version": COMPARISON_VERSION,
        "comparison_family": family,
        "comparison_id": comparison_id,
        "primary_policy_arm_id": primary.policy_arm_id,
        "baseline_policy_arm_id": baseline.policy_arm_id,
        "common_cutoff_or_error_ceiling": constraint,
        "primary_point_sha256": primary.point_sha256,
        "baseline_point_sha256": baseline.point_sha256,
        "question_ids": primary.question_ids,
        "identical_complete_question_population": True,
        "identical_pipeline_and_provenance": True,
        "identical_common_resource_or_error_constraint": True,
        "exact_realized_spend_match": math.isclose(
            primary.metrics.total_realized_person_minutes,
            baseline.metrics.total_realized_person_minutes,
            rel_tol=1e-12,
            abs_tol=_TOLERANCE,
        ),
        "primary_minus_baseline_point_deltas": _paired_deltas(primary_outcomes, baseline_outcomes),
        "paired_question_clustered_uncertainty": _paired_bootstrap(
            primary_outcomes,
            baseline_outcomes,
            draws=config.bootstrap_draws,
            seed=_seed(
                config.bootstrap_seed,
                family,
                primary.policy_arm_id,
                baseline.policy_arm_id,
                constraint,
                primary.point_sha256,
                baseline.point_sha256,
            ),
        ),
        "released_error_claim_authority": authority,
    }
    return FrontierPairedComparisonV2.model_validate(
        {**payload, "comparison_sha256": hash_canonical(payload)}
    )


def _cost_comparisons(
    *,
    rows: Sequence[RealizedCostFrontierRowV2],
    points: Sequence[CompiledPolicyPointV2],
    outcomes_by_sha: Mapping[str, Sequence[ScoredQuestionOutcomeV1]],
    config: DecisiveFrontierConfigV2,
) -> list[FrontierPairedComparisonV2]:
    points_by_sha = {row.point_sha256: row for row in points}
    by_key = {
        (row.policy_arm_id, row.common_realized_person_minutes_per_question_cutoff): row
        for row in rows
    }
    arms = sorted({row.policy_arm_id for row in points})
    comparisons: list[FrontierPairedComparisonV2] = []
    for cutoff in config.common_realized_person_minutes_per_question_cutoffs:
        primary_row = by_key[(config.primary_policy_arm_id, cutoff)]
        if not primary_row.point_available:
            continue
        assert primary_row.selected_point_sha256 is not None
        primary = points_by_sha[primary_row.selected_point_sha256]
        for baseline_arm in arms:
            if baseline_arm == config.primary_policy_arm_id:
                continue
            baseline_row = by_key[(baseline_arm, cutoff)]
            if not baseline_row.point_available:
                continue
            assert baseline_row.selected_point_sha256 is not None
            baseline = points_by_sha[baseline_row.selected_point_sha256]
            comparisons.append(
                _comparison(
                    family="common_realized_cost_ceiling",
                    constraint=cutoff,
                    primary=primary,
                    baseline=baseline,
                    outcomes_by_sha=outcomes_by_sha,
                    config=config,
                    authority=(
                        primary.released_error_claim_authority
                        and baseline.released_error_claim_authority
                    ),
                )
            )
    return comparisons


def _dominates(left: CompiledPolicyPointV2, right: CompiledPolicyPointV2) -> bool:
    left_values = (
        left.metrics.release_coverage,
        left.metrics.correct_releases_per_question,
        -left.metrics.mean_realized_person_minutes_per_question,
    )
    right_values = (
        right.metrics.release_coverage,
        right.metrics.correct_releases_per_question,
        -right.metrics.mean_realized_person_minutes_per_question,
    )
    return all(a >= b - _TOLERANCE for a, b in zip(left_values, right_values, strict=True)) and any(
        a > b + _TOLERANCE for a, b in zip(left_values, right_values, strict=True)
    )


def _fixed_error_rows(
    *, points: Sequence[CompiledPolicyPointV2], config: DecisiveFrontierConfigV2
) -> list[FixedErrorFrontierRowV2]:
    eligible = [
        point
        for point in points
        if point.metrics.released_claim_error is None
        or point.metrics.released_claim_error <= config.released_error_ceiling + _TOLERANCE
    ]
    rows: list[FixedErrorFrontierRowV2] = []
    for point in points:
        error = point.metrics.released_claim_error
        observed_eligible = point in eligible
        nondominated = observed_eligible and not any(
            other.point_sha256 != point.point_sha256 and _dominates(other, point)
            for other in eligible
        )
        status = (
            "vacuous_no_releases"
            if error is None
            else ("meets" if observed_eligible else "exceeds")
        )
        payload: dict[str, Any] = {
            "row_version": FIXED_ERROR_ROW_VERSION,
            "point_sha256": point.point_sha256,
            "policy_arm_id": point.policy_arm_id,
            "nominal_budget_minutes_per_question": point.nominal_budget_minutes_per_question,
            "released_error_ceiling": config.released_error_ceiling,
            "observed_released_claim_error": error,
            "observed_ceiling_status": status,
            "observed_ceiling_eligible": observed_eligible,
            "typed_calibration_ceiling_eligible": (
                point.released_error_claim_authority and observed_eligible
            ),
            "nondominated_within_observed_ceiling": nondominated,
            "descriptive_test_outcome_frontier_only": True,
        }
        rows.append(
            FixedErrorFrontierRowV2.model_validate(
                {**payload, "row_sha256": hash_canonical(payload)}
            )
        )
    return rows


def _fixed_error_selections(
    *,
    points: Sequence[CompiledPolicyPointV2],
    rows: Sequence[FixedErrorFrontierRowV2],
    config: DecisiveFrontierConfigV2,
) -> list[FixedErrorPolicySelectionV2]:
    row_by_sha = {row.point_sha256: row for row in rows}
    by_arm: dict[str, list[CompiledPolicyPointV2]] = defaultdict(list)
    for point in points:
        by_arm[point.policy_arm_id].append(point)
    selections: list[FixedErrorPolicySelectionV2] = []
    for arm in sorted(by_arm):
        calibrated = [row for row in by_arm[arm] if row.released_error_claim_authority]
        predeclared_calibrated = (
            max(calibrated, key=lambda row: row.nominal_budget_minutes_per_question)
            if calibrated
            else None
        )
        if (
            predeclared_calibrated is not None
            and row_by_sha[predeclared_calibrated.point_sha256].typed_calibration_ceiling_eligible
        ):
            selected = predeclared_calibrated
            basis = "typed_calibration_selected_point"
        else:
            candidates = [
                row for row in by_arm[arm] if row_by_sha[row.point_sha256].observed_ceiling_eligible
            ]
            selected = (
                max(
                    candidates,
                    key=lambda row: (
                        row.metrics.release_coverage,
                        row.metrics.correct_releases_per_question,
                        -row.metrics.mean_realized_person_minutes_per_question,
                        -row.nominal_budget_minutes_per_question,
                    ),
                )
                if candidates
                else None
            )
            basis = (
                "observed_evaluation_error_descriptive"
                if selected is not None
                else "no_point_meets_ceiling"
            )
        payload: dict[str, Any] = {
            "selection_version": SELECTION_VERSION,
            "policy_arm_id": arm,
            "released_error_ceiling": config.released_error_ceiling,
            "point_available": selected is not None,
            "selected_point_sha256": None if selected is None else selected.point_sha256,
            "selection_basis": basis,
            "release_coverage": None if selected is None else selected.metrics.release_coverage,
            "correct_releases_per_question": (
                None if selected is None else selected.metrics.correct_releases_per_question
            ),
            "mean_realized_person_minutes_per_question": (
                None
                if selected is None
                else selected.metrics.mean_realized_person_minutes_per_question
            ),
            "released_error_claim_authority": basis == "typed_calibration_selected_point",
        }
        selections.append(
            FixedErrorPolicySelectionV2.model_validate(
                {**payload, "selection_sha256": hash_canonical(payload)}
            )
        )
    return selections


def _fixed_error_comparisons(
    *,
    selections: Sequence[FixedErrorPolicySelectionV2],
    points: Sequence[CompiledPolicyPointV2],
    outcomes_by_sha: Mapping[str, Sequence[ScoredQuestionOutcomeV1]],
    config: DecisiveFrontierConfigV2,
) -> list[FrontierPairedComparisonV2]:
    point_by_sha = {row.point_sha256: row for row in points}
    selection_by_arm = {row.policy_arm_id: row for row in selections}
    primary_selection = selection_by_arm[config.primary_policy_arm_id]
    if not primary_selection.point_available:
        return []
    assert primary_selection.selected_point_sha256 is not None
    primary = point_by_sha[primary_selection.selected_point_sha256]
    comparisons: list[FrontierPairedComparisonV2] = []
    for baseline_arm in sorted(selection_by_arm):
        if baseline_arm == config.primary_policy_arm_id:
            continue
        baseline_selection = selection_by_arm[baseline_arm]
        if not baseline_selection.point_available:
            continue
        assert baseline_selection.selected_point_sha256 is not None
        baseline = point_by_sha[baseline_selection.selected_point_sha256]
        comparisons.append(
            _comparison(
                family="fixed_released_error_ceiling",
                constraint=config.released_error_ceiling,
                primary=primary,
                baseline=baseline,
                outcomes_by_sha=outcomes_by_sha,
                config=config,
                authority=(
                    primary_selection.released_error_claim_authority
                    and baseline_selection.released_error_claim_authority
                ),
            )
        )
    return comparisons


def build_decisive_claim_evaluation_frontiers_v2(
    *,
    source_result: DecisiveClaimEvaluationResultV1,
    config: DecisiveFrontierConfigV2,
    repository_root: Path,
    calibration_bundles: Sequence[AdaptiveCalibrationBundleV2] = (),
) -> DecisiveClaimEvaluationFrontiersV2:
    """Compile v1 outcomes into v2 frontiers without opening files or rerunning policies."""

    try:
        canonical_config = DecisiveFrontierConfigV2.model_validate(config.model_dump(mode="json"))
    except (AttributeError, ValueError) as exc:
        raise DecisiveClaimEvaluationV2Error("decisive_evaluation_v2_config_invalid") from exc
    source, source_anchor = _validate_source(source_result, config=canonical_config)
    component_sha256 = compute_decisive_evaluation_v2_component_sha256(repository_root)
    anchors = _calibration_anchors(
        source=source,
        source_anchor=source_anchor,
        config=canonical_config,
        bundles=calibration_bundles,
    )
    points, outcomes_by_sha = _compile_points(
        source=source,
        anchors=anchors,
        config=canonical_config,
    )
    cost_rows = _cost_rows(points=points, config=canonical_config)
    cost_comparisons = _cost_comparisons(
        rows=cost_rows,
        points=points,
        outcomes_by_sha=outcomes_by_sha,
        config=canonical_config,
    )
    fixed_rows = _fixed_error_rows(points=points, config=canonical_config)
    fixed_selections = _fixed_error_selections(
        points=points,
        rows=fixed_rows,
        config=canonical_config,
    )
    fixed_comparisons = _fixed_error_comparisons(
        selections=fixed_selections,
        points=points,
        outcomes_by_sha=outcomes_by_sha,
        config=canonical_config,
    )
    policy_arm_count = len({row.policy_arm_id for row in points})
    expected_cost_comparisons = len(
        canonical_config.common_realized_person_minutes_per_question_cutoffs
    ) * (policy_arm_count - 1)
    expected_fixed_comparisons = policy_arm_count - 1
    cost_authority = (
        len(cost_comparisons) == expected_cost_comparisons
        and bool(cost_comparisons)
        and all(row.released_error_claim_authority for row in cost_comparisons)
    )
    fixed_authority = (
        len(fixed_comparisons) == expected_fixed_comparisons
        and bool(fixed_comparisons)
        and all(row.released_error_claim_authority for row in fixed_comparisons)
    )
    blockers: list[str] = []
    if not anchors:
        blockers.append("real_typed_complete_question_calibration_bundles_missing")
    if any(not row.point_claim_authority for row in anchors):
        blockers.append("typed_calibration_not_applied_to_exact_terminal_policy_points")
    if not cost_authority:
        blockers.append("common_realized_cost_frontier_contains_uncalibrated_comparisons")
    if len(cost_comparisons) != expected_cost_comparisons:
        blockers.append("common_realized_cost_frontier_comparison_roster_incomplete")
    if not fixed_authority:
        blockers.append("fixed_error_frontier_contains_descriptive_or_uncalibrated_comparisons")
    if len(fixed_comparisons) != expected_fixed_comparisons:
        blockers.append("fixed_error_frontier_comparison_roster_incomplete")
    payload: dict[str, Any] = {
        "result_version": RESULT_VERSION,
        "evaluator_component_sha256": component_sha256,
        "config": canonical_config,
        "source_anchor": source_anchor,
        "calibration_anchors": anchors,
        "compiled_policy_points": points,
        "realized_cost_frontier": cost_rows,
        "realized_cost_paired_comparisons": cost_comparisons,
        "fixed_error_frontier": fixed_rows,
        "fixed_error_policy_selections": fixed_selections,
        "fixed_error_paired_comparisons": fixed_comparisons,
        "metric_definitions": {
            "common_realized_cost_ceiling": (
                "for each policy and prespecified mean person-minute ceiling, select the compiled "
                "point with the largest observed realized spend not exceeding that ceiling, "
                "breaking ties by larger frozen nominal budget; this is a shared resource ceiling, "
                "not a claim that policies spent exactly equal minutes"
            ),
            "correct_releases": (
                "released exact five-way decisions, including exact normalized condition-set "
                "identity for condition-dependent decisions, matching the expert reference"
            ),
            "fixed_released_error_ceiling": (
                "observed evaluation error rows are descriptive; an authoritative point requires "
                "an externally revalidated complete-question calibration bundle whose simultaneous "
                "upper risk bound is at or below the prespecified ceiling and is embedded in every "
                "terminal production replay"
            ),
            "worst_domain": (
                "minimum coverage/correct-release rate/efficiency and maximum defined released "
                "error over prespecified domains; domains without releases remain explicit"
            ),
            "bootstrap": (
                "paired or marginal domain-stratified resampling of whole independent questions; "
                "frontier point selection is frozen outside each bootstrap draw"
            ),
        },
        "input_labels_or_private_files_opened_by_v2": False,
        "policy_trajectories_rerun_by_v2": False,
        "simulation_or_fixture_inputs_accepted": False,
        "equal_nominal_deadline_misreported_as_equal_realized_cost": False,
        "realized_cost_frontier_claim_authority": cost_authority,
        "fixed_error_frontier_claim_authority": fixed_authority,
        "scientific_claim_eligible": cost_authority and fixed_authority and not blockers,
        "claim_release_authority": False,
        "causal_or_prospective_authority": False,
        "small_sample_or_finite_sample_authority": False,
        "authority_blockers": sorted(blockers),
    }
    return DecisiveClaimEvaluationFrontiersV2.model_validate(
        {**payload, "result_sha256": hash_canonical(payload)}
    )


def validate_decisive_claim_evaluation_frontiers_v2(
    *,
    result: DecisiveClaimEvaluationFrontiersV2,
    source_result: DecisiveClaimEvaluationResultV1,
    repository_root: Path,
    calibration_bundles: Sequence[AdaptiveCalibrationBundleV2] = (),
) -> DecisiveClaimEvaluationFrontiersV2:
    """Externally replay a saved v2 artifact from its v1 result and calibration inputs."""

    replayed = build_decisive_claim_evaluation_frontiers_v2(
        source_result=source_result,
        config=result.config,
        repository_root=repository_root,
        calibration_bundles=calibration_bundles,
    )
    if replayed != result:
        raise DecisiveClaimEvaluationV2Error("decisive_evaluation_v2_external_replay_mismatch")
    return result


__all__ = [
    "CompiledPolicyPointV2",
    "DecisiveClaimEvaluationFrontiersV2",
    "DecisiveClaimEvaluationV2Error",
    "DecisiveFrontierCalibrationAnchorV2",
    "DecisiveFrontierConfigV2",
    "DecisiveFrontierSourceAnchorV2",
    "FixedErrorFrontierRowV2",
    "FixedErrorPolicySelectionV2",
    "FrontierBootstrapIntervalV2",
    "FrontierBootstrapV2",
    "FrontierMetricsV2",
    "FrontierPairedComparisonV2",
    "RealizedCostFrontierRowV2",
    "build_decisive_claim_evaluation_frontiers_v2",
    "compute_decisive_evaluation_v2_component_sha256",
    "freeze_decisive_frontier_config_v2",
    "validate_decisive_claim_evaluation_frontiers_v2",
]
