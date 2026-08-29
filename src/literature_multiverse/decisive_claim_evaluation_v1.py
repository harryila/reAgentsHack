"""Leakage-firewalled claim-level evaluation for adaptive verification policies.

This additive module implements the evaluation lifecycle needed by Literature
Multiverse's central hypothesis.  Its unit is one complete review question and its
resource is *realized* total person-minutes.  Development and calibration labels may
only be used by their declared stages.  Evaluation reference verdicts remain in
separate, unreadable files while every policy trajectory and scientific-state replay is
frozen.  Scoring opens those files only after externally replaying the freeze.

The module intentionally does not create expert labels or claim authority from a
mechanics fixture.  Real-world claim eligibility additionally requires expert verdicts,
realized human audit costs, and certificate-bound scientific states after every
completed audit.  A deterministic planted fixture is provided only to test mechanics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from itertools import permutations
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import ConfigDict, Field, field_validator, model_validator

from literature_multiverse.decisive_compilation_lineage_v1 import (
    DecisiveCompilationReplayProofV1,
    replay_decisive_compilation_lineage_v1,
)
from literature_multiverse.lineage import (
    atomic_write_bytes,
    atomic_write_json,
    hash_canonical,
    sha256_file,
)
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.question_evaluation import (
    AuditCostBasis,
    BenchmarkEvidenceKind,
    QuestionAuditEvent,
    QuestionReplayState,
    ReferenceClaimVerdict,
    ReferenceClaimVerdictValue,
    ReferenceVerdictSource,
    ReplayPolicyInput,
    ReplayReleaseStatus,
    ReplaySource,
    freeze_question_audit_event,
    freeze_question_replay_state,
    freeze_reference_claim_verdict,
)

MODULE_PATH = "src/literature_multiverse/decisive_claim_evaluation_v1.py"
CLI_PATH = "scripts/run_decisive_claim_evaluation_v1.py"
CONFIG_PATH = "configs/benchmarks/decisive-claim-evaluation-v1.json"

CONFIG_VERSION = "decisive-claim-evaluation-config-v1"
IDENTITY_VERSION = "decisive-question-identity-v1"
SPLIT_MANIFEST_VERSION = "decisive-question-split-manifest-v1"
FIT_RECEIPT_VERSION = "decisive-fit-stage-receipt-v1"
TRAJECTORY_VERSION = "decisive-question-trajectory-v1"
TRAJECTORY_BUNDLE_VERSION = "decisive-trajectory-bundle-v1"
LABEL_ENTRY_VERSION = "decisive-evaluation-label-entry-v1"
LABEL_MANIFEST_VERSION = "decisive-evaluation-label-manifest-v1"
LABEL_ENVELOPE_VERSION = "decisive-evaluation-reference-envelope-v1"
LABEL_ENVELOPE_BYTES = 4096
SEALED_STAT_VERSION = "decisive-sealed-label-stat-v1"
READINESS_VERSION = "decisive-evaluation-readiness-v1"
STEP_VERSION = "decisive-policy-step-v1"
QUESTION_FREEZE_VERSION = "decisive-policy-question-freeze-v1"
POLICY_FREEZE_VERSION = "decisive-policy-freeze-v1"
RESULT_VERSION = "decisive-claim-evaluation-result-v1"

MIN_DEVELOPMENT_QUESTIONS = 5
MIN_CALIBRATION_QUESTIONS = 5
MIN_EVALUATION_QUESTIONS = 20
_COST_TOLERANCE = 1e-9
_MAX_JSON_BYTES = 64 * 1024 * 1024

REPLAY_ASSUMPTIONS_V1 = [
    (
        "retrospective_item_order_invariance: each action's adjudicated disposition and "
        "realized total-person-minutes are treated as unchanged by the policy and prior "
        "audit order"
    ),
    (
        "sequential_observability: a policy uses only its current frozen state's inputs; "
        "the selected action receipt and realized duration are opened only after selection"
    ),
    (
        "exact_scientific_rerun: every completed action must map to a hash-bound graph, "
        "synthesis, release assessment, and certificate replay for that exact ordered prefix"
    ),
    (
        "hard_deadline: an action whose realized completion crosses the fixed per-question "
        "deadline is charged only through the deadline, is not applied, and forces abstention"
    ),
    (
        "retrospective_scope: paired results are off-policy replay evidence and do not imply "
        "a randomized, causal, prospective, or scientific-truth comparison"
    ),
]


class DecisiveClaimEvaluationV1Error(ValueError):
    """The decisive evaluation contract failed closed."""


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
        raise ValueError(f"decisive_evaluation_v1_self_hash_mismatch:{field_name}")


def _strict_sorted_unique(values: Sequence[str], label: str) -> list[str]:
    rows = list(values)
    if any(not row.strip() for row in rows):
        raise ValueError(f"decisive_evaluation_v1_identity_empty:{label}")
    if rows != sorted(set(rows)):
        raise ValueError(f"decisive_evaluation_v1_not_sorted_unique:{label}")
    return rows


def _finite(value: float, label: str, *, nonnegative: bool = True) -> float:
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"decisive_evaluation_v1_number_invalid:{label}")
    return value


def _aware(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"decisive_evaluation_v1_timezone_required:{label}")
    return value


def _canonical_datetime(value: datetime) -> str:
    rendered = value.isoformat()
    return f"{rendered[:-6]}Z" if rendered.endswith("+00:00") else rendered


class StudySplit(StrEnum):
    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    EVALUATION = "evaluation"


class FitStage(StrEnum):
    DEVELOPMENT = "development_optimizer_fit"
    CALIBRATION = "calibration_policy_and_threshold_freeze"


class AdaptationMode(StrEnum):
    STATIC = "static_baseline_scores"
    ADAPTIVE = "adaptive_state_scores"
    NOT_APPLICABLE = "not_applicable"


class ScoreFamily(StrEnum):
    RANDOM = "random"
    RISK = "item_risk"
    DISAGREEMENT = "disagreement"
    INFLUENCE = "influence"
    RISK_X_INFLUENCE = "risk_x_influence"
    FIXED_COUNT = "fixed_count"
    NO_AUDIT = "no_audit"
    AUDIT_EVERYTHING = "audit_everything_upper_bound"


class PolicyArmV1(_FrozenExactModel):
    arm_id: Annotated[str, Field(min_length=1)]
    score_family: ScoreFamily
    cost_normalized: bool
    adaptation: AdaptationMode
    matched_budget: bool

    @model_validator(mode="after")
    def validate_arm(self) -> PolicyArmV1:
        if (
            self.score_family
            in {
                ScoreFamily.RANDOM,
                ScoreFamily.FIXED_COUNT,
                ScoreFamily.NO_AUDIT,
                ScoreFamily.AUDIT_EVERYTHING,
            }
            and self.cost_normalized
        ):
            raise ValueError("decisive_evaluation_v1_cost_normalization_undefined")
        if self.score_family in {ScoreFamily.NO_AUDIT, ScoreFamily.AUDIT_EVERYTHING}:
            if self.adaptation is not AdaptationMode.NOT_APPLICABLE:
                raise ValueError("decisive_evaluation_v1_adaptation_not_applicable")
        elif self.score_family in {ScoreFamily.RANDOM, ScoreFamily.FIXED_COUNT}:
            if self.adaptation is not AdaptationMode.STATIC:
                raise ValueError("decisive_evaluation_v1_fixed_policy_must_be_static")
        elif self.adaptation is AdaptationMode.NOT_APPLICABLE:
            raise ValueError("decisive_evaluation_v1_scored_policy_requires_adaptation")
        if self.score_family is ScoreFamily.AUDIT_EVERYTHING:
            if self.matched_budget:
                raise ValueError("decisive_evaluation_v1_upper_bound_is_not_budget_matched")
        elif not self.matched_budget:
            raise ValueError("decisive_evaluation_v1_non_upper_bound_must_be_budget_matched")
        expected = _policy_arm_id(
            family=self.score_family,
            cost_normalized=self.cost_normalized,
            adaptation=self.adaptation,
        )
        if self.arm_id != expected:
            raise ValueError("decisive_evaluation_v1_policy_arm_id_mismatch")
        return self


def _policy_arm_id(
    *, family: ScoreFamily, cost_normalized: bool, adaptation: AdaptationMode
) -> str:
    stem = family.value
    if cost_normalized:
        stem = f"{stem}_per_cost"
    if adaptation is AdaptationMode.STATIC:
        return f"{stem}_static"
    if adaptation is AdaptationMode.ADAPTIVE:
        return f"{stem}_adaptive"
    return stem


def required_policy_roster_v1() -> list[PolicyArmV1]:
    """Return the prespecified central-hypothesis roster in canonical order."""

    arms: list[PolicyArmV1] = []
    arms.append(
        PolicyArmV1(
            arm_id="random_static",
            score_family=ScoreFamily.RANDOM,
            cost_normalized=False,
            adaptation=AdaptationMode.STATIC,
            matched_budget=True,
        )
    )
    for family in (
        ScoreFamily.RISK,
        ScoreFamily.DISAGREEMENT,
        ScoreFamily.INFLUENCE,
        ScoreFamily.RISK_X_INFLUENCE,
    ):
        for cost_normalized in (False, True):
            for adaptation in (AdaptationMode.STATIC, AdaptationMode.ADAPTIVE):
                arms.append(
                    PolicyArmV1(
                        arm_id=_policy_arm_id(
                            family=family,
                            cost_normalized=cost_normalized,
                            adaptation=adaptation,
                        ),
                        score_family=family,
                        cost_normalized=cost_normalized,
                        adaptation=adaptation,
                        matched_budget=True,
                    )
                )
    arms.extend(
        [
            PolicyArmV1(
                arm_id="fixed_count_static",
                score_family=ScoreFamily.FIXED_COUNT,
                cost_normalized=False,
                adaptation=AdaptationMode.STATIC,
                matched_budget=True,
            ),
            PolicyArmV1(
                arm_id="no_audit",
                score_family=ScoreFamily.NO_AUDIT,
                cost_normalized=False,
                adaptation=AdaptationMode.NOT_APPLICABLE,
                matched_budget=True,
            ),
            PolicyArmV1(
                arm_id="audit_everything_upper_bound",
                score_family=ScoreFamily.AUDIT_EVERYTHING,
                cost_normalized=False,
                adaptation=AdaptationMode.NOT_APPLICABLE,
                matched_budget=False,
            ),
        ]
    )
    return arms


_REQUIRED_POLICY_IDS = [arm.arm_id for arm in required_policy_roster_v1()]
PRIMARY_POLICY_ARM_ID = "risk_x_influence_per_cost_adaptive"


class DecisiveEvaluationConfigV1(_FrozenExactModel):
    config_version: Literal["decisive-claim-evaluation-config-v1"] = CONFIG_VERSION
    budgets_minutes_per_question: Annotated[list[float], Field(min_length=1)]
    fixed_count: Annotated[int, Field(ge=1)] = 5
    random_seed: int = 20260829
    bootstrap_seed: int = 20260830
    bootstrap_draws: Annotated[int, Field(ge=100)] = 2000
    minimum_development_questions: Annotated[int, Field(ge=5)] = MIN_DEVELOPMENT_QUESTIONS
    minimum_calibration_questions: Annotated[int, Field(ge=5)] = MIN_CALIBRATION_QUESTIONS
    minimum_evaluation_questions: Annotated[int, Field(ge=20)] = MIN_EVALUATION_QUESTIONS
    primary_policy_arm_id: Literal["risk_x_influence_per_cost_adaptive"] = PRIMARY_POLICY_ARM_ID
    required_policy_arm_ids: list[str]
    prohibit_cross_split_question_overlap: Literal[True] = True
    prohibit_cross_split_claim_overlap: Literal[True] = True
    prohibit_cross_split_paper_overlap: Literal[True] = True
    prohibit_cross_split_cohort_overlap: Literal[True] = True
    require_realized_total_person_minutes: Literal[True] = True
    require_post_audit_scientific_rerun: Literal[True] = True
    require_evaluation_labels_sealed_through_freeze: Literal[True] = True
    small_sample_interval_semantics: Literal[
        "question_clustered_resampling_uncertainty_not_asymptotic_or_finite_sample_authority"
    ] = "question_clustered_resampling_uncertainty_not_asymptotic_or_finite_sample_authority"
    config_sha256: Sha256

    @field_validator("budgets_minutes_per_question")
    @classmethod
    def validate_budgets(cls, values: list[float]) -> list[float]:
        if values != sorted(set(values)):
            raise ValueError("decisive_evaluation_v1_budgets_not_sorted_unique")
        for value in values:
            _finite(value, "budget")
        return values

    @field_validator("required_policy_arm_ids")
    @classmethod
    def validate_policy_ids(cls, values: list[str]) -> list[str]:
        if values != _REQUIRED_POLICY_IDS:
            raise ValueError("decisive_evaluation_v1_policy_roster_not_exact")
        return values

    @model_validator(mode="after")
    def validate_config(self) -> DecisiveEvaluationConfigV1:
        _self_hash(self, "config_sha256")
        return self


def freeze_decisive_evaluation_config_v1(
    *,
    budgets_minutes_per_question: Sequence[float] = (15.0, 30.0, 60.0),
    fixed_count: int = 5,
    random_seed: int = 20260829,
    bootstrap_seed: int = 20260830,
    bootstrap_draws: int = 2000,
    minimum_development_questions: int = MIN_DEVELOPMENT_QUESTIONS,
    minimum_calibration_questions: int = MIN_CALIBRATION_QUESTIONS,
    minimum_evaluation_questions: int = MIN_EVALUATION_QUESTIONS,
) -> DecisiveEvaluationConfigV1:
    payload: dict[str, Any] = {
        "config_version": CONFIG_VERSION,
        "budgets_minutes_per_question": sorted(float(row) for row in budgets_minutes_per_question),
        "fixed_count": fixed_count,
        "random_seed": random_seed,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_draws": bootstrap_draws,
        "minimum_development_questions": minimum_development_questions,
        "minimum_calibration_questions": minimum_calibration_questions,
        "minimum_evaluation_questions": minimum_evaluation_questions,
        "primary_policy_arm_id": PRIMARY_POLICY_ARM_ID,
        "required_policy_arm_ids": _REQUIRED_POLICY_IDS,
        "prohibit_cross_split_question_overlap": True,
        "prohibit_cross_split_claim_overlap": True,
        "prohibit_cross_split_paper_overlap": True,
        "prohibit_cross_split_cohort_overlap": True,
        "require_realized_total_person_minutes": True,
        "require_post_audit_scientific_rerun": True,
        "require_evaluation_labels_sealed_through_freeze": True,
        "small_sample_interval_semantics": (
            "question_clustered_resampling_uncertainty_not_asymptotic_or_finite_sample_authority"
        ),
    }
    return DecisiveEvaluationConfigV1.model_validate(
        {**payload, "config_sha256": hash_canonical(payload)}
    )


class QuestionIdentityV1(_FrozenExactModel):
    identity_version: Literal["decisive-question-identity-v1"] = IDENTITY_VERSION
    split: StudySplit
    question_id: Annotated[str, Field(min_length=1)]
    claim_id: Annotated[str, Field(min_length=1)]
    domain: Annotated[str, Field(min_length=1)]
    population_id: Annotated[str, Field(min_length=1)]
    pipeline_sha256: Sha256
    corpus_sha256: Sha256
    paper_ids: Annotated[list[str], Field(min_length=1)]
    cohort_ids: Annotated[list[str], Field(min_length=1)]
    identity_sha256: Sha256

    @field_validator("paper_ids", "cohort_ids")
    @classmethod
    def validate_ids(cls, values: list[str], info: Any) -> list[str]:
        return _strict_sorted_unique(values, info.field_name)

    @model_validator(mode="after")
    def validate_identity(self) -> QuestionIdentityV1:
        _self_hash(self, "identity_sha256")
        return self


def freeze_question_identity_v1(
    *,
    split: StudySplit,
    question_id: str,
    claim_id: str,
    domain: str,
    population_id: str,
    pipeline_sha256: str,
    corpus_sha256: str,
    paper_ids: Sequence[str],
    cohort_ids: Sequence[str],
) -> QuestionIdentityV1:
    payload: dict[str, Any] = {
        "identity_version": IDENTITY_VERSION,
        "split": split,
        "question_id": question_id,
        "claim_id": claim_id,
        "domain": domain,
        "population_id": population_id,
        "pipeline_sha256": pipeline_sha256,
        "corpus_sha256": corpus_sha256,
        "paper_ids": sorted(set(paper_ids)),
        "cohort_ids": sorted(set(cohort_ids)),
    }
    return QuestionIdentityV1.model_validate(
        {**payload, "identity_sha256": hash_canonical(payload)}
    )


class DecisiveSplitManifestV1(_FrozenExactModel):
    manifest_version: Literal["decisive-question-split-manifest-v1"] = SPLIT_MANIFEST_VERSION
    split_salt_sha256: Sha256
    identities: Annotated[list[QuestionIdentityV1], Field(min_length=1)]
    development_question_ids: list[str]
    calibration_question_ids: list[str]
    evaluation_question_ids: list[str]
    pipeline_sha256: Sha256
    identity_membership_sha256: Sha256
    manifest_sha256: Sha256

    @field_validator(
        "development_question_ids", "calibration_question_ids", "evaluation_question_ids"
    )
    @classmethod
    def validate_question_ids(cls, values: list[str], info: Any) -> list[str]:
        return _strict_sorted_unique(values, info.field_name)

    @model_validator(mode="after")
    def validate_manifest(self) -> DecisiveSplitManifestV1:
        if self.identities != sorted(
            self.identities, key=lambda row: (row.split.value, row.question_id)
        ):
            raise ValueError("decisive_evaluation_v1_identities_not_canonical")
        expected_by_split = {
            split: sorted(row.question_id for row in self.identities if row.split is split)
            for split in StudySplit
        }
        if (
            expected_by_split[StudySplit.DEVELOPMENT] != self.development_question_ids
            or expected_by_split[StudySplit.CALIBRATION] != self.calibration_question_ids
            or expected_by_split[StudySplit.EVALUATION] != self.evaluation_question_ids
        ):
            raise ValueError("decisive_evaluation_v1_split_projection_mismatch")
        if {row.pipeline_sha256 for row in self.identities} != {self.pipeline_sha256}:
            raise ValueError("decisive_evaluation_v1_pipeline_mixed")
        for label, values in (
            ("question", [row.question_id for row in self.identities]),
            ("claim", [row.claim_id for row in self.identities]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"decisive_evaluation_v1_cross_split_overlap:{label}")
        for label, attribute in (("paper", "paper_ids"), ("cohort", "cohort_ids")):
            by_split = {
                split: {
                    value
                    for row in self.identities
                    if row.split is split
                    for value in getattr(row, attribute)
                }
                for split in StudySplit
            }
            for left_index, left in enumerate(StudySplit):
                for right in list(StudySplit)[left_index + 1 :]:
                    if by_split[left] & by_split[right]:
                        raise ValueError(f"decisive_evaluation_v1_cross_split_overlap:{label}")
        if self.identity_membership_sha256 != hash_canonical(
            [row.identity_sha256 for row in self.identities]
        ):
            raise ValueError("decisive_evaluation_v1_identity_membership_mismatch")
        _self_hash(self, "manifest_sha256")
        return self


def freeze_decisive_split_manifest_v1(
    *, identities: Sequence[QuestionIdentityV1], split_salt_sha256: str
) -> DecisiveSplitManifestV1:
    rows = sorted(identities, key=lambda row: (row.split.value, row.question_id))
    pipelines = {row.pipeline_sha256 for row in rows}
    if len(pipelines) != 1:
        raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_pipeline_mixed")
    payload: dict[str, Any] = {
        "manifest_version": SPLIT_MANIFEST_VERSION,
        "split_salt_sha256": split_salt_sha256,
        "identities": rows,
        "development_question_ids": sorted(
            row.question_id for row in rows if row.split is StudySplit.DEVELOPMENT
        ),
        "calibration_question_ids": sorted(
            row.question_id for row in rows if row.split is StudySplit.CALIBRATION
        ),
        "evaluation_question_ids": sorted(
            row.question_id for row in rows if row.split is StudySplit.EVALUATION
        ),
        "pipeline_sha256": next(iter(pipelines)),
        "identity_membership_sha256": hash_canonical([row.identity_sha256 for row in rows]),
    }
    return DecisiveSplitManifestV1.model_validate(
        {**payload, "manifest_sha256": hash_canonical(payload)}
    )


class FitStageReceiptV1(_FrozenExactModel):
    receipt_version: Literal["decisive-fit-stage-receipt-v1"] = FIT_RECEIPT_VERSION
    stage: FitStage
    question_ids: list[str]
    claim_ids: list[str]
    paper_ids: list[str]
    cohort_ids: list[str]
    pipeline_sha256: Sha256
    input_manifest_sha256: Sha256
    label_source: Literal["expert_adjudication", "planted_simulation", "diagnostic_proxy"]
    labels_opened_by_this_stage: Literal[True] = True
    evaluation_labels_opened: Literal[False] = False
    frozen_optimizer_or_policy_sha256: Sha256
    frozen_threshold_or_bounds_sha256: Sha256 | None
    completed_at: datetime
    receipt_sha256: Sha256

    @field_validator("question_ids", "claim_ids", "paper_ids", "cohort_ids")
    @classmethod
    def validate_ids(cls, values: list[str], info: Any) -> list[str]:
        return _strict_sorted_unique(values, info.field_name)

    @field_validator("completed_at")
    @classmethod
    def validate_completed(cls, value: datetime) -> datetime:
        return _aware(value, "fit_stage_completed_at")

    @model_validator(mode="after")
    def validate_receipt(self) -> FitStageReceiptV1:
        if self.stage is FitStage.CALIBRATION:
            if self.frozen_threshold_or_bounds_sha256 is None:
                raise ValueError("decisive_evaluation_v1_calibration_bounds_missing")
        elif self.frozen_threshold_or_bounds_sha256 is not None:
            raise ValueError("decisive_evaluation_v1_development_threshold_forbidden")
        _self_hash(self, "receipt_sha256")
        return self


def freeze_fit_stage_receipt_v1(
    *,
    stage: FitStage,
    identities: Sequence[QuestionIdentityV1],
    pipeline_sha256: str,
    input_manifest_sha256: str,
    label_source: str,
    frozen_optimizer_or_policy_sha256: str,
    frozen_threshold_or_bounds_sha256: str | None,
    completed_at: datetime,
) -> FitStageReceiptV1:
    expected_split = (
        StudySplit.DEVELOPMENT
        if FitStage(stage) is FitStage.DEVELOPMENT
        else StudySplit.CALIBRATION
    )
    rows = sorted(identities, key=lambda row: row.question_id)
    if not rows or any(row.split is not expected_split for row in rows):
        raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_fit_receipt_split_mismatch")
    payload: dict[str, Any] = {
        "receipt_version": FIT_RECEIPT_VERSION,
        "stage": stage,
        "question_ids": [row.question_id for row in rows],
        "claim_ids": sorted(row.claim_id for row in rows),
        "paper_ids": sorted({value for row in rows for value in row.paper_ids}),
        "cohort_ids": sorted({value for row in rows for value in row.cohort_ids}),
        "pipeline_sha256": pipeline_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "label_source": label_source,
        "labels_opened_by_this_stage": True,
        "evaluation_labels_opened": False,
        "frozen_optimizer_or_policy_sha256": frozen_optimizer_or_policy_sha256,
        "frozen_threshold_or_bounds_sha256": frozen_threshold_or_bounds_sha256,
        "completed_at": _canonical_datetime(completed_at),
    }
    return FitStageReceiptV1.model_validate({**payload, "receipt_sha256": hash_canonical(payload)})


class DecisivePolicyInputProvenanceV1(_FrozenExactModel):
    provenance_version: Literal["decisive-policy-input-provenance-v1"] = (
        "decisive-policy-input-provenance-v1"
    )
    development_receipt_sha256: Sha256
    calibration_receipt_sha256: Sha256
    frozen_policy_artifact_sha256: Sha256
    frozen_threshold_or_bounds_sha256: Sha256
    fit_question_ids: list[str]
    fit_claim_ids: list[str]
    fit_paper_ids: list[str]
    permitted_development_and_calibration_labels_were_opened: Literal[True] = True
    policy_and_thresholds_frozen_after_permitted_fit_stages: Literal[True] = True
    policy_and_thresholds_frozen_before_evaluation_reference_labels: Literal[True] = True
    observes_evaluation_reference_verdict: Literal[False] = False
    observes_future_audit_outcomes: Literal[False] = False
    provenance_sha256: Sha256

    @field_validator("fit_question_ids", "fit_claim_ids", "fit_paper_ids")
    @classmethod
    def validate_fit_ids(cls, values: list[str], info: Any) -> list[str]:
        return _strict_sorted_unique(values, info.field_name) if values else []

    @model_validator(mode="after")
    def validate_provenance(self) -> DecisivePolicyInputProvenanceV1:
        _self_hash(self, "provenance_sha256")
        return self


def freeze_decisive_policy_input_provenance_v1(
    *,
    development_receipt: FitStageReceiptV1,
    calibration_receipt: FitStageReceiptV1,
) -> DecisivePolicyInputProvenanceV1:
    if (
        development_receipt.stage is not FitStage.DEVELOPMENT
        or calibration_receipt.stage is not FitStage.CALIBRATION
        or calibration_receipt.frozen_threshold_or_bounds_sha256 is None
    ):
        raise DecisiveClaimEvaluationV1Error(
            "decisive_evaluation_v1_policy_provenance_fit_receipt_invalid"
        )
    payload: dict[str, Any] = {
        "provenance_version": "decisive-policy-input-provenance-v1",
        "development_receipt_sha256": development_receipt.receipt_sha256,
        "calibration_receipt_sha256": calibration_receipt.receipt_sha256,
        "frozen_policy_artifact_sha256": (calibration_receipt.frozen_optimizer_or_policy_sha256),
        "frozen_threshold_or_bounds_sha256": (
            calibration_receipt.frozen_threshold_or_bounds_sha256
        ),
        "fit_question_ids": sorted(
            set(development_receipt.question_ids) | set(calibration_receipt.question_ids)
        ),
        "fit_claim_ids": sorted(
            set(development_receipt.claim_ids) | set(calibration_receipt.claim_ids)
        ),
        "fit_paper_ids": sorted(
            set(development_receipt.paper_ids) | set(calibration_receipt.paper_ids)
        ),
        "permitted_development_and_calibration_labels_were_opened": True,
        "policy_and_thresholds_frozen_after_permitted_fit_stages": True,
        "policy_and_thresholds_frozen_before_evaluation_reference_labels": True,
        "observes_evaluation_reference_verdict": False,
        "observes_future_audit_outcomes": False,
    }
    return DecisivePolicyInputProvenanceV1.model_validate(
        {**payload, "provenance_sha256": hash_canonical(payload)}
    )


class ReplayConditionSetBindingV1(_FrozenExactModel):
    """Opaque condition identity for one condition-dependent replay state.

    The condition contents stay in the separately governed adjudication artifact.  This
    binding is sufficient to prevent a category-only ``condition_dependent`` match from
    being scored as correct when the proposed conditions differ.
    """

    binding_version: Literal["decisive-replay-condition-set-binding-v1"] = (
        "decisive-replay-condition-set-binding-v1"
    )
    replay_sha256: Sha256
    condition_set_artifact_sha256: Sha256
    binding_sha256: Sha256

    @model_validator(mode="after")
    def validate_binding(self) -> ReplayConditionSetBindingV1:
        _self_hash(self, "binding_sha256")
        return self


class QuestionTrajectoryV1(_FrozenExactModel):
    trajectory_version: Literal["decisive-question-trajectory-v1"] = TRAJECTORY_VERSION
    question_identity: QuestionIdentityV1
    evidence_kind: BenchmarkEvidenceKind
    policy_input_provenance: DecisivePolicyInputProvenanceV1
    audit_events: Annotated[list[QuestionAuditEvent], Field(min_length=1)]
    replay_states: Annotated[list[QuestionReplayState], Field(min_length=2)]
    condition_set_bindings: list[ReplayConditionSetBindingV1]
    question_reference_verdict_present: Literal[False] = False
    trajectory_sha256: Sha256

    @model_validator(mode="after")
    def validate_trajectory(self) -> QuestionTrajectoryV1:
        identity = self.question_identity
        if identity.split is not StudySplit.EVALUATION:
            raise ValueError("decisive_evaluation_v1_trajectory_not_evaluation_split")
        if self.audit_events != sorted(self.audit_events, key=lambda row: row.item_id):
            raise ValueError("decisive_evaluation_v1_audit_events_not_canonical")
        event_ids = [row.item_id for row in self.audit_events]
        if event_ids != sorted(set(event_ids)):
            raise ValueError("decisive_evaluation_v1_audit_event_ids_invalid")
        state_keys = [tuple(row.audit_sequence) for row in self.replay_states]
        if len(state_keys) != len(set(state_keys)):
            raise ValueError("decisive_evaluation_v1_replay_sequence_duplicate")
        if self.replay_states != sorted(
            self.replay_states, key=lambda row: (len(row.audit_sequence), row.audit_sequence)
        ):
            raise ValueError("decisive_evaluation_v1_replay_states_not_canonical")
        state_by_sequence = {tuple(row.audit_sequence): row for row in self.replay_states}
        if self.condition_set_bindings != sorted(
            self.condition_set_bindings, key=lambda row: row.replay_sha256
        ):
            raise ValueError("decisive_evaluation_v1_condition_bindings_not_canonical")
        if len({row.replay_sha256 for row in self.condition_set_bindings}) != len(
            self.condition_set_bindings
        ):
            raise ValueError("decisive_evaluation_v1_condition_binding_replay_duplicate")
        condition_state_hashes = {
            row.replay_sha256
            for row in self.replay_states
            if row.claim_classification == "condition_dependent"
        }
        if {row.replay_sha256 for row in self.condition_set_bindings} != condition_state_hashes:
            raise ValueError("decisive_evaluation_v1_condition_binding_state_mismatch")
        baseline = state_by_sequence.get(())
        if baseline is None:
            raise ValueError("decisive_evaluation_v1_baseline_replay_missing")
        baseline_ids = [row.item_id for row in baseline.policy_inputs]
        if baseline_ids != event_ids:
            raise ValueError("decisive_evaluation_v1_baseline_event_identity_mismatch")
        order = {row.item_id: row.canonical_order for row in baseline.policy_inputs}
        if sorted(order.values()) != list(range(1, len(order) + 1)):
            raise ValueError("decisive_evaluation_v1_canonical_order_not_contiguous")
        canonical = tuple(sorted(event_ids, key=order.__getitem__))
        full = state_by_sequence.get(canonical)
        if full is None or full.policy_inputs:
            raise ValueError("decisive_evaluation_v1_full_canonical_replay_missing")
        for state in self.replay_states:
            if (
                state.question_id != identity.question_id
                or state.pipeline_sha256 != identity.pipeline_sha256
            ):
                raise ValueError("decisive_evaluation_v1_replay_identity_mismatch")
            if len(state.audit_sequence) != len(set(state.audit_sequence)):
                raise ValueError("decisive_evaluation_v1_replay_sequence_duplicate_item")
            remaining = sorted(set(event_ids) - set(state.audit_sequence))
            if [row.item_id for row in state.policy_inputs] != remaining:
                raise ValueError("decisive_evaluation_v1_replay_pending_identity_mismatch")
            for policy_input in state.policy_inputs:
                if policy_input.canonical_order != order[policy_input.item_id]:
                    raise ValueError("decisive_evaluation_v1_canonical_order_changed")
        expected_cost = {
            BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED: (AuditCostBasis.REALIZED_HUMAN_MINUTES),
            BenchmarkEvidenceKind.SIMULATION: AuditCostBasis.SIMULATED_MINUTES,
            BenchmarkEvidenceKind.DIAGNOSTIC: AuditCostBasis.DIAGNOSTIC_MINUTES,
        }[self.evidence_kind]
        if any(row.cost_basis is not expected_cost for row in self.audit_events):
            raise ValueError("decisive_evaluation_v1_cost_basis_mismatch")
        expected_source = {
            BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED: ReplaySource.FROZEN_PIPELINE_RERUN,
            BenchmarkEvidenceKind.SIMULATION: ReplaySource.PLANTED_SIMULATION,
            BenchmarkEvidenceKind.DIAGNOSTIC: ReplaySource.DIAGNOSTIC_APPROXIMATION,
        }[self.evidence_kind]
        if any(row.replay_source is not expected_source for row in self.replay_states):
            raise ValueError("decisive_evaluation_v1_replay_source_mismatch")
        allowed_risk_bases = {
            BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED: {"calibrated_cell_rate_ucl"},
            BenchmarkEvidenceKind.SIMULATION: {"simulation"},
            BenchmarkEvidenceKind.DIAGNOSTIC: {"heuristic"},
        }[self.evidence_kind]
        if any(
            policy_input.risk_basis not in allowed_risk_bases
            for state in self.replay_states
            for policy_input in state.policy_inputs
        ):
            raise ValueError("decisive_evaluation_v1_risk_basis_not_authorized")
        if self.evidence_kind is BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED and any(
            state.production_binding is None for state in self.replay_states
        ):
            raise ValueError("decisive_evaluation_v1_real_state_certificate_missing")
        if identity.question_id in self.policy_input_provenance.fit_question_ids:
            raise ValueError("decisive_evaluation_v1_policy_fit_question_leak")
        if identity.claim_id in self.policy_input_provenance.fit_claim_ids:
            raise ValueError("decisive_evaluation_v1_policy_fit_claim_leak")
        if set(identity.paper_ids) & set(self.policy_input_provenance.fit_paper_ids):
            raise ValueError("decisive_evaluation_v1_policy_fit_paper_leak")
        _self_hash(self, "trajectory_sha256")
        return self


def freeze_question_trajectory_v1(
    *,
    question_identity: QuestionIdentityV1,
    evidence_kind: BenchmarkEvidenceKind,
    policy_input_provenance: DecisivePolicyInputProvenanceV1,
    audit_events: Sequence[QuestionAuditEvent],
    replay_states: Sequence[QuestionReplayState],
    condition_set_artifact_sha256_by_replay_sha256: Mapping[str, str] | None = None,
) -> QuestionTrajectoryV1:
    states = sorted(
        replay_states,
        key=lambda row: (len(row.audit_sequence), tuple(row.audit_sequence)),
    )
    supplied_conditions = dict(condition_set_artifact_sha256_by_replay_sha256 or {})
    required_condition_states = {
        row.replay_sha256 for row in states if row.claim_classification == "condition_dependent"
    }
    if set(supplied_conditions) != required_condition_states:
        raise DecisiveClaimEvaluationV1Error(
            "decisive_evaluation_v1_condition_binding_state_mismatch"
        )
    condition_bindings: list[ReplayConditionSetBindingV1] = []
    for replay_sha256 in sorted(supplied_conditions):
        condition_payload = {
            "binding_version": "decisive-replay-condition-set-binding-v1",
            "replay_sha256": replay_sha256,
            "condition_set_artifact_sha256": supplied_conditions[replay_sha256],
        }
        condition_bindings.append(
            ReplayConditionSetBindingV1.model_validate(
                {
                    **condition_payload,
                    "binding_sha256": hash_canonical(condition_payload),
                }
            )
        )
    payload: dict[str, Any] = {
        "trajectory_version": TRAJECTORY_VERSION,
        "question_identity": question_identity,
        "evidence_kind": evidence_kind,
        "policy_input_provenance": policy_input_provenance,
        "audit_events": sorted(audit_events, key=lambda row: row.item_id),
        "replay_states": states,
        "condition_set_bindings": condition_bindings,
        "question_reference_verdict_present": False,
    }
    return QuestionTrajectoryV1.model_validate(
        {**payload, "trajectory_sha256": hash_canonical(payload)}
    )


class TrajectoryBundleV1(_FrozenExactModel):
    bundle_version: Literal["decisive-trajectory-bundle-v1"] = TRAJECTORY_BUNDLE_VERSION
    split_manifest_sha256: Sha256
    pipeline_sha256: Sha256
    evidence_kind: BenchmarkEvidenceKind
    trajectories: Annotated[list[QuestionTrajectoryV1], Field(min_length=1)]
    trajectory_membership_sha256: Sha256
    bundle_sha256: Sha256

    @model_validator(mode="after")
    def validate_bundle(self) -> TrajectoryBundleV1:
        if self.trajectories != sorted(
            self.trajectories, key=lambda row: row.question_identity.question_id
        ):
            raise ValueError("decisive_evaluation_v1_trajectories_not_canonical")
        if len({row.question_identity.question_id for row in self.trajectories}) != len(
            self.trajectories
        ):
            raise ValueError("decisive_evaluation_v1_trajectory_question_duplicate")
        if {row.evidence_kind for row in self.trajectories} != {self.evidence_kind}:
            raise ValueError("decisive_evaluation_v1_evidence_kind_mixed")
        if {row.question_identity.pipeline_sha256 for row in self.trajectories} != {
            self.pipeline_sha256
        }:
            raise ValueError("decisive_evaluation_v1_trajectory_pipeline_mixed")
        if self.trajectory_membership_sha256 != hash_canonical(
            [row.trajectory_sha256 for row in self.trajectories]
        ):
            raise ValueError("decisive_evaluation_v1_trajectory_membership_mismatch")
        _self_hash(self, "bundle_sha256")
        return self


def freeze_trajectory_bundle_v1(
    *,
    split_manifest: DecisiveSplitManifestV1,
    evidence_kind: BenchmarkEvidenceKind,
    trajectories: Sequence[QuestionTrajectoryV1],
) -> TrajectoryBundleV1:
    rows = sorted(trajectories, key=lambda row: row.question_identity.question_id)
    payload: dict[str, Any] = {
        "bundle_version": TRAJECTORY_BUNDLE_VERSION,
        "split_manifest_sha256": split_manifest.manifest_sha256,
        "pipeline_sha256": split_manifest.pipeline_sha256,
        "evidence_kind": evidence_kind,
        "trajectories": rows,
        "trajectory_membership_sha256": hash_canonical([row.trajectory_sha256 for row in rows]),
    }
    return TrajectoryBundleV1.model_validate({**payload, "bundle_sha256": hash_canonical(payload)})


class EnvelopeNonceOrigin(StrEnum):
    EXTERNAL_CUSTODIAN = "external_custodian"
    PLANTED_SIMULATION_FIXTURE = "planted_simulation_fixture"
    DIAGNOSTIC_FIXTURE = "diagnostic_fixture"


class EvaluationReferenceEnvelopeV1(_FrozenExactModel):
    envelope_version: Literal["decisive-evaluation-reference-envelope-v1"] = LABEL_ENVELOPE_VERSION
    custodian_nonce_hex: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    nonce_origin: EnvelopeNonceOrigin
    reference_verdict: ReferenceClaimVerdict
    reference_condition_set_artifact_sha256: Sha256 | None = None
    padding: Annotated[str, Field(min_length=1)]

    @field_validator("padding")
    @classmethod
    def validate_padding(cls, value: str) -> str:
        if set(value) != {"_"}:
            raise ValueError("decisive_evaluation_v1_envelope_padding_invalid")
        return value

    @model_validator(mode="after")
    def validate_envelope(self) -> EvaluationReferenceEnvelopeV1:
        expected_source = {
            EnvelopeNonceOrigin.EXTERNAL_CUSTODIAN: (ReferenceVerdictSource.EXPERT_ADJUDICATION),
            EnvelopeNonceOrigin.PLANTED_SIMULATION_FIXTURE: (
                ReferenceVerdictSource.PLANTED_SIMULATION
            ),
            EnvelopeNonceOrigin.DIAGNOSTIC_FIXTURE: (ReferenceVerdictSource.DIAGNOSTIC_PROXY),
        }[self.nonce_origin]
        if self.reference_verdict.source is not expected_source:
            raise ValueError("decisive_evaluation_v1_envelope_nonce_origin_mismatch")
        condition_dependent = (
            self.reference_verdict.verdict is ReferenceClaimVerdictValue.CONDITION_DEPENDENT
        )
        if condition_dependent != (self.reference_condition_set_artifact_sha256 is not None):
            raise ValueError("decisive_evaluation_v1_reference_condition_binding_mismatch")
        return self


def _canonical_envelope_bytes(value: EvaluationReferenceEnvelopeV1) -> bytes:
    return json.dumps(
        value.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def freeze_evaluation_reference_envelope_v1(
    *,
    reference_verdict: ReferenceClaimVerdict,
    custodian_nonce_hex: str,
    nonce_origin: EnvelopeNonceOrigin,
    reference_condition_set_artifact_sha256: str | None = None,
) -> bytes:
    """Create one fixed-size envelope; real nonces originate outside evaluation."""

    base = EvaluationReferenceEnvelopeV1(
        custodian_nonce_hex=custodian_nonce_hex,
        nonce_origin=nonce_origin,
        reference_verdict=reference_verdict,
        reference_condition_set_artifact_sha256=(reference_condition_set_artifact_sha256),
        padding="_",
    )
    base_bytes = _canonical_envelope_bytes(base)
    padding_length = LABEL_ENVELOPE_BYTES - len(base_bytes) + 1
    if padding_length < 1:
        raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_envelope_payload_too_large")
    envelope = base.model_copy(update={"padding": "_" * padding_length})
    envelope = EvaluationReferenceEnvelopeV1.model_validate(envelope.model_dump(mode="json"))
    content = _canonical_envelope_bytes(envelope)
    if len(content) != LABEL_ENVELOPE_BYTES:
        raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_envelope_fixed_size_failed")
    return content


def parse_evaluation_reference_envelope_v1(
    content: bytes,
) -> EvaluationReferenceEnvelopeV1:
    if len(content) != LABEL_ENVELOPE_BYTES:
        raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_envelope_size_invalid")
    try:
        raw = json.loads(content)
        envelope = EvaluationReferenceEnvelopeV1.model_validate(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_envelope_invalid") from exc
    if _canonical_envelope_bytes(envelope) != content:
        raise DecisiveClaimEvaluationV1Error(
            "decisive_evaluation_v1_envelope_not_exact_canonical_bytes"
        )
    return envelope


class EvaluationLabelEntryV1(_FrozenExactModel):
    entry_version: Literal["decisive-evaluation-label-entry-v1"] = LABEL_ENTRY_VERSION
    question_id: Annotated[str, Field(min_length=1)]
    claim_id: Annotated[str, Field(min_length=1)]
    relative_path: Annotated[str, Field(min_length=1)]
    expected_envelope_sha256: Sha256
    fixed_envelope_bytes: Literal[4096] = LABEL_ENVELOPE_BYTES
    nonce_present_in_manifest: Literal[False] = False
    entry_sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("decisive_evaluation_v1_label_path_invalid")
        if path.suffix != ".json":
            raise ValueError("decisive_evaluation_v1_label_path_not_json")
        return value

    @model_validator(mode="after")
    def validate_entry(self) -> EvaluationLabelEntryV1:
        expected_path = f"{hashlib.sha256(self.question_id.encode('utf-8')).hexdigest()}.json"
        if self.relative_path != expected_path:
            raise ValueError("decisive_evaluation_v1_label_path_not_public_question_derivation")
        _self_hash(self, "entry_sha256")
        return self


class EvaluationLabelManifestV1(_FrozenExactModel):
    manifest_version: Literal["decisive-evaluation-label-manifest-v1"] = LABEL_MANIFEST_VERSION
    split_manifest_sha256: Sha256
    evidence_kind: BenchmarkEvidenceKind
    entries: Annotated[list[EvaluationLabelEntryV1], Field(min_length=1)]
    question_ids: list[str]
    entry_membership_sha256: Sha256
    label_values_present: Literal[False] = False
    verdict_fields_present: Literal[False] = False
    nonce_values_present: Literal[False] = False
    all_envelopes_identical_fixed_bytes: Literal[4096] = LABEL_ENVELOPE_BYTES
    envelope_security_semantics: Literal[
        "full_sha_is_enumeration_resistant_only_with_external_unpredictable_128_bit_nonce"
    ] = "full_sha_is_enumeration_resistant_only_with_external_unpredictable_128_bit_nonce"
    manifest_sha256: Sha256

    @field_validator("question_ids")
    @classmethod
    def validate_question_ids(cls, values: list[str]) -> list[str]:
        return _strict_sorted_unique(values, "label_manifest_question_ids")

    @model_validator(mode="after")
    def validate_manifest(self) -> EvaluationLabelManifestV1:
        if self.entries != sorted(self.entries, key=lambda row: row.question_id):
            raise ValueError("decisive_evaluation_v1_label_entries_not_canonical")
        if [row.question_id for row in self.entries] != self.question_ids:
            raise ValueError("decisive_evaluation_v1_label_question_projection_mismatch")
        if len({row.claim_id for row in self.entries}) != len(self.entries):
            raise ValueError("decisive_evaluation_v1_label_claim_duplicate")
        if len({row.relative_path for row in self.entries}) != len(self.entries):
            raise ValueError("decisive_evaluation_v1_label_path_duplicate")
        if {row.fixed_envelope_bytes for row in self.entries} != {LABEL_ENVELOPE_BYTES}:
            raise ValueError("decisive_evaluation_v1_label_envelope_sizes_not_fixed")
        if self.entry_membership_sha256 != hash_canonical(
            [row.entry_sha256 for row in self.entries]
        ):
            raise ValueError("decisive_evaluation_v1_label_membership_mismatch")
        _self_hash(self, "manifest_sha256")
        return self


def freeze_evaluation_label_manifest_v1(
    *,
    split_manifest_sha256: str,
    evidence_kind: BenchmarkEvidenceKind,
    entries: Sequence[EvaluationLabelEntryV1],
) -> EvaluationLabelManifestV1:
    rows = sorted(entries, key=lambda row: row.question_id)
    payload: dict[str, Any] = {
        "manifest_version": LABEL_MANIFEST_VERSION,
        "split_manifest_sha256": split_manifest_sha256,
        "evidence_kind": evidence_kind,
        "entries": rows,
        "question_ids": [row.question_id for row in rows],
        "entry_membership_sha256": hash_canonical([row.entry_sha256 for row in rows]),
        "label_values_present": False,
        "verdict_fields_present": False,
        "nonce_values_present": False,
        "all_envelopes_identical_fixed_bytes": LABEL_ENVELOPE_BYTES,
        "envelope_security_semantics": (
            "full_sha_is_enumeration_resistant_only_with_external_unpredictable_128_bit_nonce"
        ),
    }
    return EvaluationLabelManifestV1.model_validate(
        {**payload, "manifest_sha256": hash_canonical(payload)}
    )


def freeze_evaluation_label_entry_v1(
    *,
    question_id: str,
    claim_id: str,
    relative_path: str,
    expected_envelope_sha256: str,
) -> EvaluationLabelEntryV1:
    payload: dict[str, Any] = {
        "entry_version": LABEL_ENTRY_VERSION,
        "question_id": question_id,
        "claim_id": claim_id,
        "relative_path": relative_path,
        "expected_envelope_sha256": expected_envelope_sha256,
        "fixed_envelope_bytes": LABEL_ENVELOPE_BYTES,
        "nonce_present_in_manifest": False,
    }
    return EvaluationLabelEntryV1.model_validate(
        {**payload, "entry_sha256": hash_canonical(payload)}
    )


class SealedEvaluationLabelStatV1(_FrozenExactModel):
    stat_version: Literal["decisive-sealed-label-stat-v1"] = SEALED_STAT_VERSION
    question_id: Annotated[str, Field(min_length=1)]
    relative_path: Annotated[str, Field(min_length=1)]
    device: Annotated[int, Field(ge=0)]
    inode: Annotated[int, Field(gt=0)]
    file_bytes: Annotated[int, Field(gt=0, le=_MAX_JSON_BYTES)]
    mtime_ns: Annotated[int, Field(ge=0)]
    link_count: Literal[1] = 1
    mode_permissions: Literal[0] = 0
    content_opened_or_hashed: Literal[False] = False
    stat_sha256: Sha256

    @model_validator(mode="after")
    def validate_stat(self) -> SealedEvaluationLabelStatV1:
        _self_hash(self, "stat_sha256")
        return self


def _canonical_directory(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        root_stat = absolute.lstat()
    except OSError as exc:
        raise DecisiveClaimEvaluationV1Error(
            f"decisive_evaluation_v1_directory_missing:{label}"
        ) from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise DecisiveClaimEvaluationV1Error(f"decisive_evaluation_v1_directory_invalid:{label}")
    return absolute


def _safe_label_path(label_root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_label_path_invalid")
    current = label_root
    for part in relative.parts[:-1]:
        current = current / part
        try:
            value = current.lstat()
        except OSError as exc:
            raise DecisiveClaimEvaluationV1Error(
                "decisive_evaluation_v1_label_parent_missing"
            ) from exc
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
            raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_label_parent_invalid")
    return label_root.joinpath(*relative.parts)


def inspect_sealed_evaluation_labels_v1(
    *, label_root: Path, label_manifest: EvaluationLabelManifestV1
) -> tuple[int, int, list[SealedEvaluationLabelStatV1]]:
    """Use lstat only; evaluation label bytes are deliberately never opened here."""

    root = _canonical_directory(label_root, "evaluation_label_root")
    root_stat = root.lstat()
    rows: list[SealedEvaluationLabelStatV1] = []
    for entry in label_manifest.entries:
        target = _safe_label_path(root, entry.relative_path)
        try:
            file_stat = target.lstat()
        except OSError as exc:
            raise DecisiveClaimEvaluationV1Error(
                f"decisive_evaluation_v1_label_file_missing:{entry.question_id}"
            ) from exc
        if not stat.S_ISREG(file_stat.st_mode) or stat.S_ISLNK(file_stat.st_mode):
            raise DecisiveClaimEvaluationV1Error(
                f"decisive_evaluation_v1_label_file_not_regular:{entry.question_id}"
            )
        if file_stat.st_nlink != 1:
            raise DecisiveClaimEvaluationV1Error(
                f"decisive_evaluation_v1_label_file_hardlinked:{entry.question_id}"
            )
        if file_stat.st_size != entry.fixed_envelope_bytes:
            raise DecisiveClaimEvaluationV1Error(
                f"decisive_evaluation_v1_label_file_size_mismatch:{entry.question_id}"
            )
        permissions = stat.S_IMODE(file_stat.st_mode)
        if permissions != 0:
            raise DecisiveClaimEvaluationV1Error(
                f"decisive_evaluation_v1_evaluation_label_not_sealed:{entry.question_id}"
            )
        payload: dict[str, Any] = {
            "stat_version": SEALED_STAT_VERSION,
            "question_id": entry.question_id,
            "relative_path": entry.relative_path,
            "device": file_stat.st_dev,
            "inode": file_stat.st_ino,
            "file_bytes": file_stat.st_size,
            "mtime_ns": file_stat.st_mtime_ns,
            "link_count": 1,
            "mode_permissions": 0,
            "content_opened_or_hashed": False,
        }
        rows.append(
            SealedEvaluationLabelStatV1.model_validate(
                {**payload, "stat_sha256": hash_canonical(payload)}
            )
        )
    return root_stat.st_dev, root_stat.st_ino, rows


def compute_decisive_evaluation_component_sha256_v1(repository_root: Path) -> str:
    root = _canonical_directory(repository_root, "repository_root")
    paths = (
        MODULE_PATH,
        CLI_PATH,
        CONFIG_PATH,
        "src/literature_multiverse/decisive_compilation_lineage_v1.py",
        "src/literature_multiverse/question_evaluation.py",
        "src/literature_multiverse/certificate.py",
        "src/literature_multiverse/lineage.py",
        "src/literature_multiverse/models.py",
        "pyproject.toml",
        "uv.lock",
    )
    rows: list[dict[str, Any]] = []
    for relative in paths:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise DecisiveClaimEvaluationV1Error(
                f"decisive_evaluation_v1_component_file_missing:{relative}"
            )
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return hash_canonical(
        {
            "component_id": "decisive-claim-evaluation-v1",
            "files": rows,
        }
    )


class DecisiveEvaluationReadinessV1(_FrozenExactModel):
    readiness_version: Literal["decisive-evaluation-readiness-v1"] = READINESS_VERSION
    assessed_at: datetime
    config_sha256: Sha256
    component_sha256: Sha256
    portable_semantic_inputs_sha256: Sha256 | None
    split_manifest_sha256: Sha256 | None
    development_receipt_sha256: Sha256 | None
    calibration_receipt_sha256: Sha256 | None
    trajectory_bundle_sha256: Sha256 | None
    compilation_replay_proof: DecisiveCompilationReplayProofV1 | None = None
    label_manifest_sha256: Sha256 | None
    evidence_kind: BenchmarkEvidenceKind | None
    pipeline_sha256: Sha256 | None
    development_question_count: Annotated[int, Field(ge=0)]
    calibration_question_count: Annotated[int, Field(ge=0)]
    evaluation_question_count: Annotated[int, Field(ge=0)]
    label_root_device: Annotated[int, Field(ge=0)] | None
    label_root_inode: Annotated[int, Field(gt=0)] | None
    sealed_label_stats: list[SealedEvaluationLabelStatV1]
    evaluation_label_files_lstat_only: Literal[True] = True
    evaluation_label_contents_opened: Literal[False] = False
    development_and_calibration_labels_may_have_been_opened: Literal[True] = True
    status: Literal["ready", "blocked"]
    blockers: list[str]
    real_scored_run_candidate: bool
    scientific_claim_authority: Literal[False] = False
    custody_semantics: Literal[
        "machine_local_nonportable_label_seal_and_preopen_custody_receipt"
    ] = "machine_local_nonportable_label_seal_and_preopen_custody_receipt"
    readiness_sha256: Sha256

    @field_validator("assessed_at")
    @classmethod
    def validate_assessed_at(cls, value: datetime) -> datetime:
        return _aware(value, "readiness_assessed_at")

    @field_validator("blockers")
    @classmethod
    def validate_blockers(cls, values: list[str]) -> list[str]:
        return _strict_sorted_unique(values, "readiness_blockers") if values else []

    @model_validator(mode="after")
    def validate_readiness(self) -> DecisiveEvaluationReadinessV1:
        if (self.status == "ready") == bool(self.blockers):
            raise ValueError("decisive_evaluation_v1_readiness_status_mismatch")
        if self.status == "ready":
            required = (
                self.portable_semantic_inputs_sha256,
                self.split_manifest_sha256,
                self.development_receipt_sha256,
                self.calibration_receipt_sha256,
                self.trajectory_bundle_sha256,
                self.label_manifest_sha256,
                self.evidence_kind,
                self.pipeline_sha256,
                self.label_root_device,
                self.label_root_inode,
            )
            if any(row is None for row in required) or not self.sealed_label_stats:
                raise ValueError("decisive_evaluation_v1_ready_inputs_incomplete")
        if self.compilation_replay_proof is not None:
            lineage = self.compilation_replay_proof.lineage_identity
            if (
                self.evidence_kind is not BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED
                or lineage.config_sha256 != self.config_sha256
                or lineage.split_manifest_sha256 != self.split_manifest_sha256
                or lineage.development_receipt_sha256 != self.development_receipt_sha256
                or lineage.calibration_receipt_sha256 != self.calibration_receipt_sha256
                or lineage.trajectory_bundle_sha256 != self.trajectory_bundle_sha256
                or (
                    self.sealed_label_stats
                    and lineage.evaluation_question_ids
                    != sorted(row.question_id for row in self.sealed_label_stats)
                )
            ):
                raise ValueError("decisive_evaluation_v1_compilation_proof_projection_mismatch")
        elif (
            self.status == "ready"
            and self.evidence_kind is BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED
        ):
            raise ValueError("decisive_evaluation_v1_real_compilation_proof_missing")
        if self.real_scored_run_candidate != (
            self.status == "ready"
            and self.evidence_kind is BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED
            and self.compilation_replay_proof is not None
        ):
            raise ValueError("decisive_evaluation_v1_real_candidate_mismatch")
        hash_payload = self.model_dump(mode="json", exclude={"readiness_sha256"})
        if self.compilation_replay_proof is None:
            hash_payload.pop("compilation_replay_proof", None)
        if hash_canonical(hash_payload) != self.readiness_sha256:
            raise ValueError("decisive_evaluation_v1_self_hash_mismatch:readiness_sha256")
        return self


def _identities_for_split(
    manifest: DecisiveSplitManifestV1, split: StudySplit
) -> list[QuestionIdentityV1]:
    return [row for row in manifest.identities if row.split is split]


def _portable_semantic_inputs_sha256_v1(
    *,
    config: DecisiveEvaluationConfigV1,
    split_manifest: DecisiveSplitManifestV1,
    development_receipt: FitStageReceiptV1,
    calibration_receipt: FitStageReceiptV1,
    trajectory_bundle: TrajectoryBundleV1,
    label_manifest: EvaluationLabelManifestV1,
    compilation_replay_proof: DecisiveCompilationReplayProofV1 | None,
) -> str:
    payload = {
        "semantic_inputs_version": "decisive-semantic-inputs-v1",
        "config_sha256": config.config_sha256,
        "split_manifest_sha256": split_manifest.manifest_sha256,
        "development_receipt_sha256": development_receipt.receipt_sha256,
        "calibration_receipt_sha256": calibration_receipt.receipt_sha256,
        "trajectory_bundle_sha256": trajectory_bundle.bundle_sha256,
        "label_manifest_sha256": label_manifest.manifest_sha256,
    }
    if compilation_replay_proof is not None:
        payload["compilation_replay_proof_sha256"] = compilation_replay_proof.proof_sha256
    return hash_canonical(payload)


def assess_decisive_evaluation_readiness_v1(
    *,
    config: DecisiveEvaluationConfigV1,
    repository_root: Path,
    assessed_at: datetime,
    split_manifest: DecisiveSplitManifestV1 | None = None,
    development_receipt: FitStageReceiptV1 | None = None,
    calibration_receipt: FitStageReceiptV1 | None = None,
    trajectory_bundle: TrajectoryBundleV1 | None = None,
    trajectory_compilation_result_path: Path | None = None,
    trajectory_compilation_source_roster_path: Path | None = None,
    trajectory_compilation_source_root: Path | None = None,
    label_manifest: EvaluationLabelManifestV1 | None = None,
    label_root: Path | None = None,
) -> DecisiveEvaluationReadinessV1:
    """Build a self-hashed readiness receipt without opening evaluation labels."""

    blockers: set[str] = set()
    component_sha256 = compute_decisive_evaluation_component_sha256_v1(repository_root)
    for name, value in (
        ("split_manifest", split_manifest),
        ("development_receipt", development_receipt),
        ("calibration_receipt", calibration_receipt),
        ("trajectory_bundle", trajectory_bundle),
        ("label_manifest", label_manifest),
        ("label_root", label_root),
    ):
        if value is None:
            blockers.add(f"missing_{name}")

    development_count = (
        len(split_manifest.development_question_ids) if split_manifest is not None else 0
    )
    calibration_count = (
        len(split_manifest.calibration_question_ids) if split_manifest is not None else 0
    )
    evaluation_count = (
        len(split_manifest.evaluation_question_ids) if split_manifest is not None else 0
    )
    if development_count < config.minimum_development_questions:
        blockers.add("insufficient_development_complete_questions")
    if calibration_count < config.minimum_calibration_questions:
        blockers.add("insufficient_calibration_complete_questions")
    if evaluation_count < config.minimum_evaluation_questions:
        blockers.add("insufficient_evaluation_complete_questions")

    if split_manifest is not None:
        development_rows = _identities_for_split(split_manifest, StudySplit.DEVELOPMENT)
        calibration_rows = _identities_for_split(split_manifest, StudySplit.CALIBRATION)
        evaluation_rows = _identities_for_split(split_manifest, StudySplit.EVALUATION)
        expected_stage_source = None
        if trajectory_bundle is not None:
            expected_stage_source = {
                BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED: "expert_adjudication",
                BenchmarkEvidenceKind.SIMULATION: "planted_simulation",
                BenchmarkEvidenceKind.DIAGNOSTIC: "diagnostic_proxy",
            }[trajectory_bundle.evidence_kind]
        for receipt, stage, rows in (
            (development_receipt, FitStage.DEVELOPMENT, development_rows),
            (calibration_receipt, FitStage.CALIBRATION, calibration_rows),
        ):
            if receipt is None:
                continue
            if receipt.stage is not stage:
                raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_fit_stage_mismatch")
            if (
                receipt.question_ids != sorted(row.question_id for row in rows)
                or receipt.claim_ids != sorted(row.claim_id for row in rows)
                or receipt.paper_ids != sorted({value for row in rows for value in row.paper_ids})
                or receipt.cohort_ids != sorted({value for row in rows for value in row.cohort_ids})
                or receipt.pipeline_sha256 != split_manifest.pipeline_sha256
                or receipt.input_manifest_sha256 != split_manifest.manifest_sha256
            ):
                raise DecisiveClaimEvaluationV1Error(
                    "decisive_evaluation_v1_fit_receipt_projection_mismatch"
                )
            if expected_stage_source is not None and receipt.label_source != expected_stage_source:
                raise DecisiveClaimEvaluationV1Error(
                    "decisive_evaluation_v1_fit_label_source_mismatch"
                )
        if (
            development_receipt is not None
            and calibration_receipt is not None
            and development_receipt.completed_at > calibration_receipt.completed_at
        ):
            raise DecisiveClaimEvaluationV1Error(
                "decisive_evaluation_v1_fit_stage_time_order_invalid"
            )
        if trajectory_bundle is not None:
            if (
                trajectory_bundle.split_manifest_sha256 != split_manifest.manifest_sha256
                or trajectory_bundle.pipeline_sha256 != split_manifest.pipeline_sha256
                or [row.question_identity for row in trajectory_bundle.trajectories]
                != evaluation_rows
            ):
                raise DecisiveClaimEvaluationV1Error(
                    "decisive_evaluation_v1_trajectory_split_projection_mismatch"
                )
            fit_question_ids = set(split_manifest.development_question_ids) | set(
                split_manifest.calibration_question_ids
            )
            fit_claim_ids = {row.claim_id for row in development_rows + calibration_rows}
            fit_paper_ids = {
                value for row in development_rows + calibration_rows for value in row.paper_ids
            }
            eval_question_ids = set(split_manifest.evaluation_question_ids)
            for trajectory in trajectory_bundle.trajectories:
                provenance = trajectory.policy_input_provenance
                if (
                    development_receipt is None
                    or calibration_receipt is None
                    or provenance.development_receipt_sha256 != development_receipt.receipt_sha256
                    or provenance.calibration_receipt_sha256 != calibration_receipt.receipt_sha256
                    or provenance.frozen_policy_artifact_sha256
                    != calibration_receipt.frozen_optimizer_or_policy_sha256
                    or provenance.frozen_threshold_or_bounds_sha256
                    != calibration_receipt.frozen_threshold_or_bounds_sha256
                    or set(provenance.fit_question_ids) != fit_question_ids
                    or set(provenance.fit_claim_ids) != fit_claim_ids
                    or set(provenance.fit_paper_ids) != fit_paper_ids
                    or set(provenance.fit_question_ids) & eval_question_ids
                ):
                    raise DecisiveClaimEvaluationV1Error(
                        "decisive_evaluation_v1_policy_fit_outside_prespecified_splits"
                    )
        if label_manifest is not None:
            if (
                label_manifest.split_manifest_sha256 != split_manifest.manifest_sha256
                or label_manifest.question_ids != split_manifest.evaluation_question_ids
                or [row.claim_id for row in label_manifest.entries]
                != [row.claim_id for row in evaluation_rows]
            ):
                raise DecisiveClaimEvaluationV1Error(
                    "decisive_evaluation_v1_label_split_projection_mismatch"
                )
            if (
                trajectory_bundle is not None
                and label_manifest.evidence_kind is not trajectory_bundle.evidence_kind
            ):
                raise DecisiveClaimEvaluationV1Error(
                    "decisive_evaluation_v1_label_evidence_kind_mismatch"
                )

    label_root_device: int | None = None
    label_root_inode: int | None = None
    sealed_stats: list[SealedEvaluationLabelStatV1] = []
    if label_manifest is not None and label_root is not None:
        label_root_device, label_root_inode, sealed_stats = inspect_sealed_evaluation_labels_v1(
            label_root=label_root,
            label_manifest=label_manifest,
        )

    evidence_kind = trajectory_bundle.evidence_kind if trajectory_bundle is not None else None
    pipeline_sha256 = split_manifest.pipeline_sha256 if split_manifest is not None else None
    compilation_replay_proof: DecisiveCompilationReplayProofV1 | None = None
    compilation_inputs = {
        "trajectory_compilation_result": trajectory_compilation_result_path,
        "trajectory_compilation_source_roster": (trajectory_compilation_source_roster_path),
        "trajectory_compilation_source_root": trajectory_compilation_source_root,
    }
    if evidence_kind is BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED:
        for name, value in compilation_inputs.items():
            if value is None:
                blockers.add(f"missing_{name}")
        if (
            all(value is not None for value in compilation_inputs.values())
            and split_manifest is not None
            and development_receipt is not None
            and calibration_receipt is not None
            and trajectory_bundle is not None
        ):
            assert trajectory_compilation_result_path is not None
            assert trajectory_compilation_source_roster_path is not None
            assert trajectory_compilation_source_root is not None
            compilation_replay_proof = replay_decisive_compilation_lineage_v1(
                config=config,
                split_manifest=split_manifest,
                development_receipt=development_receipt,
                calibration_receipt=calibration_receipt,
                trajectory_bundle=trajectory_bundle,
                compiler_result_path=trajectory_compilation_result_path,
                source_roster_path=trajectory_compilation_source_roster_path,
                source_root=trajectory_compilation_source_root,
                repository_root=repository_root,
            )
    elif any(value is not None for value in compilation_inputs.values()):
        raise DecisiveClaimEvaluationV1Error(
            "decisive_evaluation_v1_nonreal_compilation_lineage_forbidden"
        )
    semantic_inputs_sha256 = None
    if all(
        row is not None
        for row in (
            split_manifest,
            development_receipt,
            calibration_receipt,
            trajectory_bundle,
            label_manifest,
        )
    ):
        assert split_manifest is not None
        assert development_receipt is not None
        assert calibration_receipt is not None
        assert trajectory_bundle is not None
        assert label_manifest is not None
        semantic_inputs_sha256 = _portable_semantic_inputs_sha256_v1(
            config=config,
            split_manifest=split_manifest,
            development_receipt=development_receipt,
            calibration_receipt=calibration_receipt,
            trajectory_bundle=trajectory_bundle,
            label_manifest=label_manifest,
            compilation_replay_proof=compilation_replay_proof,
        )
    ordered_blockers = sorted(blockers)
    payload: dict[str, Any] = {
        "readiness_version": READINESS_VERSION,
        "assessed_at": _canonical_datetime(assessed_at),
        "config_sha256": config.config_sha256,
        "component_sha256": component_sha256,
        "portable_semantic_inputs_sha256": semantic_inputs_sha256,
        "split_manifest_sha256": (
            split_manifest.manifest_sha256 if split_manifest is not None else None
        ),
        "development_receipt_sha256": (
            development_receipt.receipt_sha256 if development_receipt is not None else None
        ),
        "calibration_receipt_sha256": (
            calibration_receipt.receipt_sha256 if calibration_receipt is not None else None
        ),
        "trajectory_bundle_sha256": (
            trajectory_bundle.bundle_sha256 if trajectory_bundle is not None else None
        ),
        **(
            {}
            if compilation_replay_proof is None
            else {"compilation_replay_proof": compilation_replay_proof}
        ),
        "label_manifest_sha256": (
            label_manifest.manifest_sha256 if label_manifest is not None else None
        ),
        "evidence_kind": evidence_kind,
        "pipeline_sha256": pipeline_sha256,
        "development_question_count": development_count,
        "calibration_question_count": calibration_count,
        "evaluation_question_count": evaluation_count,
        "label_root_device": label_root_device,
        "label_root_inode": label_root_inode,
        "sealed_label_stats": sealed_stats,
        "evaluation_label_files_lstat_only": True,
        "evaluation_label_contents_opened": False,
        "development_and_calibration_labels_may_have_been_opened": True,
        "status": "blocked" if ordered_blockers else "ready",
        "blockers": ordered_blockers,
        "real_scored_run_candidate": (
            not ordered_blockers
            and evidence_kind is BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED
            and compilation_replay_proof is not None
        ),
        "scientific_claim_authority": False,
        "custody_semantics": ("machine_local_nonportable_label_seal_and_preopen_custody_receipt"),
    }
    return DecisiveEvaluationReadinessV1.model_validate(
        {**payload, "readiness_sha256": hash_canonical(payload)}
    )


class FrozenPolicyStepV1(_FrozenExactModel):
    step_version: Literal["decisive-policy-step-v1"] = STEP_VERSION
    step_index: Annotated[int, Field(ge=1)]
    pre_audit_sequence: list[str]
    pre_replay_sha256: Sha256
    selected_item_id: Annotated[str, Field(min_length=1)]
    selected_policy_input_sha256: Sha256
    selected_score_state_sha256: Sha256
    selection_priority: float
    estimated_minutes_visible_at_selection: Annotated[float, Field(gt=0)]
    action_receipt_opened_after_selection: Literal[True] = True
    action_event_sha256: Sha256
    realized_minutes_opened_after_selection: Annotated[float, Field(gt=0)]
    action_outcome: Literal["completed_and_rerun", "active_at_budget_deadline"]
    charged_realized_minutes: Annotated[float, Field(gt=0)]
    post_audit_sequence: list[str]
    post_replay_sha256: Sha256 | None
    step_sha256: Sha256

    @field_validator("selection_priority")
    @classmethod
    def validate_priority(cls, value: float) -> float:
        return _finite(value, "selection_priority", nonnegative=False)

    @model_validator(mode="after")
    def validate_step(self) -> FrozenPolicyStepV1:
        if self.step_index != len(self.pre_audit_sequence) + 1:
            raise ValueError("decisive_evaluation_v1_step_index_mismatch")
        completed = self.action_outcome == "completed_and_rerun"
        if completed:
            if (
                self.post_audit_sequence != [*self.pre_audit_sequence, self.selected_item_id]
                or self.post_replay_sha256 is None
                or not math.isclose(
                    self.charged_realized_minutes,
                    self.realized_minutes_opened_after_selection,
                    rel_tol=1e-12,
                    abs_tol=_COST_TOLERANCE,
                )
            ):
                raise ValueError("decisive_evaluation_v1_completed_step_invalid")
        elif (
            self.post_audit_sequence != self.pre_audit_sequence
            or self.post_replay_sha256 is not None
            or self.charged_realized_minutes
            > self.realized_minutes_opened_after_selection + _COST_TOLERANCE
        ):
            raise ValueError("decisive_evaluation_v1_active_step_invalid")
        _self_hash(self, "step_sha256")
        return self


class FrozenPolicyQuestionV1(_FrozenExactModel):
    question_freeze_version: Literal["decisive-policy-question-freeze-v1"] = QUESTION_FREEZE_VERSION
    question_id: Annotated[str, Field(min_length=1)]
    claim_id: Annotated[str, Field(min_length=1)]
    domain: Annotated[str, Field(min_length=1)]
    policy_arm_id: Annotated[str, Field(min_length=1)]
    budget_minutes: float | None
    selected_item_ids: list[str]
    resolved_item_ids: list[str]
    active_item_id: str | None
    steps: list[FrozenPolicyStepV1]
    historical_realized_minutes: Annotated[float, Field(ge=0)]
    active_truncated_realized_minutes: Annotated[float, Field(ge=0)]
    total_realized_minutes: Annotated[float, Field(ge=0)]
    stop_reason: Literal[
        "no_audit_policy",
        "first_frozen_release_eligible_state",
        "all_items_resolved",
        "fixed_count_reached",
        "budget_exhausted_without_active_action",
        "budget_exhausted_with_active_action",
        "no_eligible_candidate_fits_estimated_budget",
        "exhaustive_upper_bound_complete",
    ]
    final_replay_sha256: Sha256
    release_status: ReplayReleaseStatus
    claim_classification: Literal[
        "supported",
        "contradicted",
        "condition_dependent",
        "inconclusive",
        "not_evaluable",
    ]
    condition_set_artifact_sha256: Sha256 | None = None
    release_reasons: list[str]
    reference_verdict_opened: Literal[False] = False
    question_freeze_sha256: Sha256

    @field_validator(
        "historical_realized_minutes",
        "active_truncated_realized_minutes",
        "total_realized_minutes",
    )
    @classmethod
    def validate_costs(cls, value: float, info: Any) -> float:
        return _finite(value, info.field_name)

    @model_validator(mode="after")
    def validate_question_freeze(self) -> FrozenPolicyQuestionV1:
        if self.selected_item_ids != [row.selected_item_id for row in self.steps]:
            raise ValueError("decisive_evaluation_v1_selected_step_projection_mismatch")
        expected_resolved = [
            row.selected_item_id
            for row in self.steps
            if row.action_outcome == "completed_and_rerun"
        ]
        if self.resolved_item_ids != expected_resolved:
            raise ValueError("decisive_evaluation_v1_resolved_step_projection_mismatch")
        active_steps = [
            row for row in self.steps if row.action_outcome == "active_at_budget_deadline"
        ]
        if len(active_steps) > 1 or (
            (active_steps[0].selected_item_id if active_steps else None) != self.active_item_id
        ):
            raise ValueError("decisive_evaluation_v1_active_step_projection_mismatch")
        if self.active_item_id is not None and self.steps[-1] is not active_steps[0]:
            raise ValueError("decisive_evaluation_v1_active_step_not_terminal")
        if not math.isclose(
            self.total_realized_minutes,
            self.historical_realized_minutes + self.active_truncated_realized_minutes,
            rel_tol=1e-12,
            abs_tol=_COST_TOLERANCE,
        ):
            raise ValueError("decisive_evaluation_v1_total_cost_mismatch")
        if self.budget_minutes is not None:
            _finite(self.budget_minutes, "question_budget")
            if self.total_realized_minutes > self.budget_minutes + _COST_TOLERANCE:
                raise ValueError("decisive_evaluation_v1_budget_violation")
        if self.active_item_id is not None and (
            self.release_status is not ReplayReleaseStatus.ABSTAINED
            or "budget_exhausted_active_audit_action_unresolved" not in self.release_reasons
        ):
            raise ValueError("decisive_evaluation_v1_active_action_must_abstain")
        if self.release_status is ReplayReleaseStatus.RELEASED and self.release_reasons:
            raise ValueError("decisive_evaluation_v1_released_state_has_reasons")
        if self.release_status is ReplayReleaseStatus.ABSTAINED and not self.release_reasons:
            raise ValueError("decisive_evaluation_v1_abstained_state_missing_reason")
        condition_dependent = self.claim_classification == "condition_dependent"
        if condition_dependent != (self.condition_set_artifact_sha256 is not None):
            raise ValueError("decisive_evaluation_v1_frozen_condition_binding_mismatch")
        _self_hash(self, "question_freeze_sha256")
        return self


class FrozenPolicyPopulationV1(_FrozenExactModel):
    policy_arm: PolicyArmV1
    budget_minutes: float | None
    question_ids: list[str]
    questions: Annotated[list[FrozenPolicyQuestionV1], Field(min_length=1)]
    population_sha256: Sha256

    @field_validator("question_ids")
    @classmethod
    def validate_question_ids(cls, values: list[str]) -> list[str]:
        return _strict_sorted_unique(values, "policy_population_question_ids")

    @model_validator(mode="after")
    def validate_population(self) -> FrozenPolicyPopulationV1:
        if self.questions != sorted(self.questions, key=lambda row: row.question_id):
            raise ValueError("decisive_evaluation_v1_policy_questions_not_canonical")
        if [row.question_id for row in self.questions] != self.question_ids:
            raise ValueError("decisive_evaluation_v1_policy_population_projection_mismatch")
        if any(
            row.policy_arm_id != self.policy_arm.arm_id or row.budget_minutes != self.budget_minutes
            for row in self.questions
        ):
            raise ValueError("decisive_evaluation_v1_policy_population_identity_mismatch")
        if self.policy_arm.matched_budget != (self.budget_minutes is not None):
            raise ValueError("decisive_evaluation_v1_policy_budget_match_mismatch")
        _self_hash(self, "population_sha256")
        return self


class DecisivePolicyFreezeV1(_FrozenExactModel):
    freeze_version: Literal["decisive-policy-freeze-v1"] = POLICY_FREEZE_VERSION
    frozen_at: datetime
    component_sha256: Sha256
    portable_semantic_inputs_sha256: Sha256
    config: DecisiveEvaluationConfigV1
    split_manifest: DecisiveSplitManifestV1
    development_receipt: FitStageReceiptV1
    calibration_receipt: FitStageReceiptV1
    trajectory_bundle: TrajectoryBundleV1
    compilation_replay_proof: DecisiveCompilationReplayProofV1 | None = None
    label_manifest: EvaluationLabelManifestV1
    policy_populations: Annotated[list[FrozenPolicyPopulationV1], Field(min_length=1)]
    evaluation_question_ids: list[str]
    replay_assumptions: list[str]
    evaluation_reference_labels_opened: Literal[False] = False
    policy_predictions_and_certificates_frozen: Literal[True] = True
    development_and_calibration_label_access_scope: Literal[
        "only_their_prespecified_stages_before_evaluation_freeze"
    ] = "only_their_prespecified_stages_before_evaluation_freeze"
    freeze_portability: Literal[
        "portable_semantic_artifact_excludes_local_custody_device_inode_mode_and_mtime"
    ] = "portable_semantic_artifact_excludes_local_custody_device_inode_mode_and_mtime"
    real_scored_run_candidate: bool
    scientific_claim_authority: Literal[False] = False
    freeze_sha256: Sha256

    @field_validator("frozen_at")
    @classmethod
    def validate_frozen_at(cls, value: datetime) -> datetime:
        return _aware(value, "policy_frozen_at")

    @field_validator("evaluation_question_ids")
    @classmethod
    def validate_question_ids(cls, values: list[str]) -> list[str]:
        return _strict_sorted_unique(values, "freeze_evaluation_question_ids")

    @model_validator(mode="after")
    def validate_freeze(self) -> DecisivePolicyFreezeV1:
        if self.frozen_at < self.calibration_receipt.completed_at:
            raise ValueError("decisive_evaluation_v1_freeze_predates_calibration")
        expected_semantic = _portable_semantic_inputs_sha256_v1(
            config=self.config,
            split_manifest=self.split_manifest,
            development_receipt=self.development_receipt,
            calibration_receipt=self.calibration_receipt,
            trajectory_bundle=self.trajectory_bundle,
            label_manifest=self.label_manifest,
            compilation_replay_proof=self.compilation_replay_proof,
        )
        if self.portable_semantic_inputs_sha256 != expected_semantic:
            raise ValueError("decisive_evaluation_v1_semantic_input_hash_mismatch")
        if (
            self.development_receipt.input_manifest_sha256 != self.split_manifest.manifest_sha256
            or self.calibration_receipt.input_manifest_sha256 != self.split_manifest.manifest_sha256
            or self.trajectory_bundle.split_manifest_sha256 != self.split_manifest.manifest_sha256
            or self.label_manifest.split_manifest_sha256 != self.split_manifest.manifest_sha256
        ):
            raise ValueError("decisive_evaluation_v1_semantic_split_binding_mismatch")
        if (
            self.development_receipt.question_ids != self.split_manifest.development_question_ids
            or self.calibration_receipt.question_ids != self.split_manifest.calibration_question_ids
            or [row.question_identity.question_id for row in self.trajectory_bundle.trajectories]
            != self.split_manifest.evaluation_question_ids
            or self.label_manifest.question_ids != self.split_manifest.evaluation_question_ids
        ):
            raise ValueError("decisive_evaluation_v1_semantic_population_mismatch")
        if (
            len(self.split_manifest.development_question_ids)
            < self.config.minimum_development_questions
            or len(self.split_manifest.calibration_question_ids)
            < self.config.minimum_calibration_questions
            or len(self.split_manifest.evaluation_question_ids)
            < self.config.minimum_evaluation_questions
        ):
            raise ValueError("decisive_evaluation_v1_semantic_population_too_small")
        if self.evaluation_question_ids != self.split_manifest.evaluation_question_ids:
            raise ValueError("decisive_evaluation_v1_freeze_question_projection_mismatch")
        if self.replay_assumptions != REPLAY_ASSUMPTIONS_V1:
            raise ValueError("decisive_evaluation_v1_replay_assumptions_mismatch")
        real_evidence = (
            self.trajectory_bundle.evidence_kind is BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED
        )
        if real_evidence != (self.compilation_replay_proof is not None):
            raise ValueError("decisive_evaluation_v1_compilation_proof_kind_mismatch")
        if self.compilation_replay_proof is not None:
            lineage = self.compilation_replay_proof.lineage_identity
            if (
                lineage.config_sha256 != self.config.config_sha256
                or lineage.split_manifest_sha256 != self.split_manifest.manifest_sha256
                or lineage.development_receipt_sha256 != self.development_receipt.receipt_sha256
                or lineage.calibration_receipt_sha256 != self.calibration_receipt.receipt_sha256
                or lineage.trajectory_bundle_sha256 != self.trajectory_bundle.bundle_sha256
                or lineage.trajectory_membership_sha256
                != self.trajectory_bundle.trajectory_membership_sha256
                or lineage.evaluation_question_ids != self.evaluation_question_ids
            ):
                raise ValueError("decisive_evaluation_v1_compilation_proof_projection_mismatch")
        expected_real = real_evidence and self.compilation_replay_proof is not None
        if self.real_scored_run_candidate != expected_real:
            raise ValueError("decisive_evaluation_v1_real_candidate_mismatch")
        expected_keys = [
            (arm_id, budget)
            for arm_id in self.config.required_policy_arm_ids
            if arm_id != "audit_everything_upper_bound"
            for budget in self.config.budgets_minutes_per_question
        ] + [("audit_everything_upper_bound", None)]
        observed_keys = [
            (row.policy_arm.arm_id, row.budget_minutes) for row in self.policy_populations
        ]
        if observed_keys != expected_keys:
            raise ValueError("decisive_evaluation_v1_policy_population_roster_mismatch")
        if any(row.question_ids != self.evaluation_question_ids for row in self.policy_populations):
            raise ValueError("decisive_evaluation_v1_policy_population_unequal")
        hash_payload = self.model_dump(mode="json", exclude={"freeze_sha256"})
        if self.compilation_replay_proof is None:
            hash_payload.pop("compilation_replay_proof", None)
        if hash_canonical(hash_payload) != self.freeze_sha256:
            raise ValueError("decisive_evaluation_v1_self_hash_mismatch:freeze_sha256")
        return self


def _stable_random_priority(*, seed: int, question_id: str, item_id: str) -> float:
    digest = hashlib.sha256(
        f"decisive-evaluation-random-v1\0{seed}\0{question_id}\0{item_id}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _priority_v1(
    row: ReplayPolicyInput,
    *,
    arm: PolicyArmV1,
    seed: int,
    question_id: str,
) -> float:
    if arm.score_family is ScoreFamily.RANDOM:
        value = _stable_random_priority(seed=seed, question_id=question_id, item_id=row.item_id)
    elif arm.score_family is ScoreFamily.RISK:
        value = row.risk_score
    elif arm.score_family is ScoreFamily.DISAGREEMENT:
        value = row.disagreement_score
    elif arm.score_family is ScoreFamily.INFLUENCE:
        value = row.influence_score
    elif arm.score_family is ScoreFamily.RISK_X_INFLUENCE:
        value = row.risk_score * row.influence_score
    elif arm.score_family is ScoreFamily.FIXED_COUNT:
        value = -float(row.canonical_order)
    else:
        raise DecisiveClaimEvaluationV1Error(
            f"decisive_evaluation_v1_policy_not_rankable:{arm.arm_id}"
        )
    if arm.cost_normalized:
        value /= row.estimated_minutes
    return value


def _state_release_eligible(
    state: QuestionReplayState, *, evidence_kind: BenchmarkEvidenceKind
) -> bool:
    if evidence_kind is BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED:
        binding = state.production_binding
        if binding is None:
            raise DecisiveClaimEvaluationV1Error(
                "decisive_evaluation_v1_real_release_state_missing_certificate"
            )
        if binding.evaluated_active_action_item_id is not None:
            raise DecisiveClaimEvaluationV1Error(
                "decisive_evaluation_v1_frozen_prefix_has_active_action"
            )
        return binding.full_release_eligible
    return state.release_status is ReplayReleaseStatus.RELEASED


def _freeze_step_v1(
    *,
    step_index: int,
    pre_sequence: Sequence[str],
    pre_state: QuestionReplayState,
    selected: ReplayPolicyInput,
    priority: float,
    event: QuestionAuditEvent,
    outcome: Literal["completed_and_rerun", "active_at_budget_deadline"],
    charged: float,
    post_sequence: Sequence[str],
    post_state: QuestionReplayState | None,
) -> FrozenPolicyStepV1:
    payload: dict[str, Any] = {
        "step_version": STEP_VERSION,
        "step_index": step_index,
        "pre_audit_sequence": list(pre_sequence),
        "pre_replay_sha256": pre_state.replay_sha256,
        "selected_item_id": selected.item_id,
        "selected_policy_input_sha256": hash_canonical(selected),
        "selected_score_state_sha256": selected.score_state_sha256,
        "selection_priority": priority,
        "estimated_minutes_visible_at_selection": selected.estimated_minutes,
        "action_receipt_opened_after_selection": True,
        "action_event_sha256": event.event_sha256,
        "realized_minutes_opened_after_selection": event.realized_minutes,
        "action_outcome": outcome,
        "charged_realized_minutes": charged,
        "post_audit_sequence": list(post_sequence),
        "post_replay_sha256": post_state.replay_sha256 if post_state is not None else None,
    }
    return FrozenPolicyStepV1.model_validate({**payload, "step_sha256": hash_canonical(payload)})


def _freeze_policy_question_v1(
    *,
    trajectory: QuestionTrajectoryV1,
    arm: PolicyArmV1,
    budget_minutes: float | None,
    fixed_count: int,
    random_seed: int,
) -> FrozenPolicyQuestionV1:
    identity = trajectory.question_identity
    state_by_sequence = {tuple(state.audit_sequence): state for state in trajectory.replay_states}
    condition_by_replay = {
        row.replay_sha256: row.condition_set_artifact_sha256
        for row in trajectory.condition_set_bindings
    }
    baseline = state_by_sequence[()]
    baseline_by_id = {row.item_id: row for row in baseline.policy_inputs}
    event_by_id = {row.item_id: row for row in trajectory.audit_events}
    canonical_ids = [
        row.item_id for row in sorted(baseline.policy_inputs, key=lambda row: row.canonical_order)
    ]

    resolved: list[str] = []
    selected_ids: list[str] = []
    steps: list[FrozenPolicyStepV1] = []
    historical_spent = 0.0
    active_item_id: str | None = None
    active_truncated = 0.0

    if arm.score_family is ScoreFamily.NO_AUDIT:
        state = baseline
        stop_reason = "no_audit_policy"
    elif arm.score_family is ScoreFamily.AUDIT_EVERYTHING:
        for item_id in canonical_ids:
            state = state_by_sequence.get(tuple(resolved))
            if state is None:
                raise DecisiveClaimEvaluationV1Error(
                    f"decisive_evaluation_v1_replay_prefix_missing:{identity.question_id}:{resolved}"
                )
            current_by_id = {row.item_id: row for row in state.policy_inputs}
            policy_input = current_by_id.get(item_id)
            if policy_input is None:
                raise DecisiveClaimEvaluationV1Error(
                    "decisive_evaluation_v1_exhaustive_pending_item_missing"
                )
            event = event_by_id[item_id]
            post_sequence = [*resolved, item_id]
            post_state = state_by_sequence.get(tuple(post_sequence))
            if post_state is None:
                raise DecisiveClaimEvaluationV1Error(
                    f"decisive_evaluation_v1_post_audit_rerun_missing:{identity.question_id}:{post_sequence}"
                )
            steps.append(
                _freeze_step_v1(
                    step_index=len(resolved) + 1,
                    pre_sequence=resolved,
                    pre_state=state,
                    selected=policy_input,
                    priority=-float(policy_input.canonical_order),
                    event=event,
                    outcome="completed_and_rerun",
                    charged=event.realized_minutes,
                    post_sequence=post_sequence,
                    post_state=post_state,
                )
            )
            selected_ids.append(item_id)
            resolved = post_sequence
            historical_spent = math.fsum((historical_spent, event.realized_minutes))
        state = state_by_sequence[tuple(resolved)]
        stop_reason = "exhaustive_upper_bound_complete"
    else:
        if budget_minutes is None:
            raise DecisiveClaimEvaluationV1Error(
                "decisive_evaluation_v1_matched_policy_budget_missing"
            )
        while True:
            state = state_by_sequence.get(tuple(resolved))
            if state is None:
                raise DecisiveClaimEvaluationV1Error(
                    f"decisive_evaluation_v1_replay_prefix_missing:{identity.question_id}:{resolved}"
                )
            if _state_release_eligible(state, evidence_kind=trajectory.evidence_kind):
                stop_reason = "first_frozen_release_eligible_state"
                break
            if not state.policy_inputs:
                stop_reason = "all_items_resolved"
                break
            if arm.score_family is ScoreFamily.FIXED_COUNT and len(selected_ids) >= fixed_count:
                stop_reason = "fixed_count_reached"
                break
            remaining_budget = max(0.0, budget_minutes - historical_spent)
            if remaining_budget <= _COST_TOLERANCE:
                stop_reason = "budget_exhausted_without_active_action"
                break
            current_by_id = {row.item_id: row for row in state.policy_inputs}
            if arm.adaptation is AdaptationMode.ADAPTIVE:
                ranking_rows = list(state.policy_inputs)
            else:
                ranking_rows = [baseline_by_id[item_id] for item_id in current_by_id]
            ranked = sorted(
                ranking_rows,
                key=lambda row: (
                    -_priority_v1(
                        row,
                        arm=arm,
                        seed=random_seed,
                        question_id=identity.question_id,
                    ),
                    row.item_id,
                ),
            )
            fitting = [
                row
                for row in ranked
                if row.eligible and row.estimated_minutes <= remaining_budget + _COST_TOLERANCE
            ]
            if not fitting:
                stop_reason = "no_eligible_candidate_fits_estimated_budget"
                break
            selected = fitting[0]
            if selected.item_id not in current_by_id:
                raise DecisiveClaimEvaluationV1Error(
                    "decisive_evaluation_v1_selected_item_not_current"
                )
            priority = _priority_v1(
                selected,
                arm=arm,
                seed=random_seed,
                question_id=identity.question_id,
            )
            event = event_by_id[selected.item_id]
            selected_ids.append(selected.item_id)
            if event.realized_minutes > remaining_budget + _COST_TOLERANCE:
                active_item_id = selected.item_id
                active_truncated = remaining_budget
                steps.append(
                    _freeze_step_v1(
                        step_index=len(resolved) + 1,
                        pre_sequence=resolved,
                        pre_state=state,
                        selected=selected,
                        priority=priority,
                        event=event,
                        outcome="active_at_budget_deadline",
                        charged=remaining_budget,
                        post_sequence=resolved,
                        post_state=None,
                    )
                )
                stop_reason = "budget_exhausted_with_active_action"
                break
            post_sequence = [*resolved, selected.item_id]
            post_state = state_by_sequence.get(tuple(post_sequence))
            if post_state is None:
                raise DecisiveClaimEvaluationV1Error(
                    f"decisive_evaluation_v1_post_audit_rerun_missing:{identity.question_id}:{post_sequence}"
                )
            steps.append(
                _freeze_step_v1(
                    step_index=len(resolved) + 1,
                    pre_sequence=resolved,
                    pre_state=state,
                    selected=selected,
                    priority=priority,
                    event=event,
                    outcome="completed_and_rerun",
                    charged=event.realized_minutes,
                    post_sequence=post_sequence,
                    post_state=post_state,
                )
            )
            resolved = post_sequence
            historical_spent = math.fsum((historical_spent, event.realized_minutes))
        state = state_by_sequence[tuple(resolved)]

    total_spent = math.fsum((historical_spent, active_truncated))
    release_eligible = _state_release_eligible(state, evidence_kind=trajectory.evidence_kind)
    released = release_eligible and active_item_id is None
    release_reasons = list(state.release_reasons)
    if active_item_id is not None:
        release_reasons.append("budget_exhausted_active_audit_action_unresolved")
    if not released and not release_reasons:
        release_reasons.append("frozen_scientific_state_not_release_eligible")
    payload: dict[str, Any] = {
        "question_freeze_version": QUESTION_FREEZE_VERSION,
        "question_id": identity.question_id,
        "claim_id": identity.claim_id,
        "domain": identity.domain,
        "policy_arm_id": arm.arm_id,
        "budget_minutes": budget_minutes,
        "selected_item_ids": selected_ids,
        "resolved_item_ids": resolved,
        "active_item_id": active_item_id,
        "steps": steps,
        "historical_realized_minutes": historical_spent,
        "active_truncated_realized_minutes": active_truncated,
        "total_realized_minutes": total_spent,
        "stop_reason": stop_reason,
        "final_replay_sha256": state.replay_sha256,
        "release_status": (
            ReplayReleaseStatus.RELEASED if released else ReplayReleaseStatus.ABSTAINED
        ),
        "claim_classification": state.claim_classification,
        "condition_set_artifact_sha256": condition_by_replay.get(state.replay_sha256),
        "release_reasons": list(dict.fromkeys(release_reasons)),
        "reference_verdict_opened": False,
    }
    return FrozenPolicyQuestionV1.model_validate(
        {**payload, "question_freeze_sha256": hash_canonical(payload)}
    )


def _freeze_policy_population_v1(
    *,
    trajectories: Sequence[QuestionTrajectoryV1],
    arm: PolicyArmV1,
    budget_minutes: float | None,
    config: DecisiveEvaluationConfigV1,
) -> FrozenPolicyPopulationV1:
    rows = [
        _freeze_policy_question_v1(
            trajectory=trajectory,
            arm=arm,
            budget_minutes=budget_minutes,
            fixed_count=config.fixed_count,
            random_seed=config.random_seed,
        )
        for trajectory in trajectories
    ]
    rows.sort(key=lambda row: row.question_id)
    payload: dict[str, Any] = {
        "policy_arm": arm,
        "budget_minutes": budget_minutes,
        "question_ids": [row.question_id for row in rows],
        "questions": rows,
    }
    return FrozenPolicyPopulationV1.model_validate(
        {**payload, "population_sha256": hash_canonical(payload)}
    )


def freeze_decisive_policy_trajectories_v1(
    *,
    config: DecisiveEvaluationConfigV1,
    readiness: DecisiveEvaluationReadinessV1,
    split_manifest: DecisiveSplitManifestV1,
    development_receipt: FitStageReceiptV1,
    calibration_receipt: FitStageReceiptV1,
    trajectory_bundle: TrajectoryBundleV1,
    trajectory_compilation_result_path: Path | None = None,
    trajectory_compilation_source_roster_path: Path | None = None,
    trajectory_compilation_source_root: Path | None = None,
    label_manifest: EvaluationLabelManifestV1,
    label_root: Path,
    repository_root: Path,
    frozen_at: datetime,
) -> DecisivePolicyFreezeV1:
    """Freeze all matched policy trajectories while evaluation verdicts are sealed."""

    replayed_readiness = assess_decisive_evaluation_readiness_v1(
        config=config,
        repository_root=repository_root,
        assessed_at=readiness.assessed_at,
        split_manifest=split_manifest,
        development_receipt=development_receipt,
        calibration_receipt=calibration_receipt,
        trajectory_bundle=trajectory_bundle,
        trajectory_compilation_result_path=trajectory_compilation_result_path,
        trajectory_compilation_source_roster_path=(trajectory_compilation_source_roster_path),
        trajectory_compilation_source_root=trajectory_compilation_source_root,
        label_manifest=label_manifest,
        label_root=label_root,
    )
    if replayed_readiness != readiness or readiness.status != "ready":
        raise DecisiveClaimEvaluationV1Error(
            "decisive_evaluation_v1_readiness_external_replay_mismatch"
        )
    if frozen_at < calibration_receipt.completed_at:
        raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_freeze_predates_calibration")
    arms = {arm.arm_id: arm for arm in required_policy_roster_v1()}
    populations: list[FrozenPolicyPopulationV1] = []
    for arm_id in config.required_policy_arm_ids:
        arm = arms[arm_id]
        if arm.score_family is ScoreFamily.AUDIT_EVERYTHING:
            continue
        for budget in config.budgets_minutes_per_question:
            populations.append(
                _freeze_policy_population_v1(
                    trajectories=trajectory_bundle.trajectories,
                    arm=arm,
                    budget_minutes=budget,
                    config=config,
                )
            )
    upper = arms["audit_everything_upper_bound"]
    populations.append(
        _freeze_policy_population_v1(
            trajectories=trajectory_bundle.trajectories,
            arm=upper,
            budget_minutes=None,
            config=config,
        )
    )
    component_sha256 = compute_decisive_evaluation_component_sha256_v1(repository_root)
    semantic_inputs_sha256 = _portable_semantic_inputs_sha256_v1(
        config=config,
        split_manifest=split_manifest,
        development_receipt=development_receipt,
        calibration_receipt=calibration_receipt,
        trajectory_bundle=trajectory_bundle,
        label_manifest=label_manifest,
        compilation_replay_proof=readiness.compilation_replay_proof,
    )
    if readiness.portable_semantic_inputs_sha256 != semantic_inputs_sha256:
        raise DecisiveClaimEvaluationV1Error(
            "decisive_evaluation_v1_custody_semantic_binding_mismatch"
        )
    payload: dict[str, Any] = {
        "freeze_version": POLICY_FREEZE_VERSION,
        "frozen_at": _canonical_datetime(frozen_at),
        "component_sha256": component_sha256,
        "portable_semantic_inputs_sha256": semantic_inputs_sha256,
        "config": config,
        "split_manifest": split_manifest,
        "development_receipt": development_receipt,
        "calibration_receipt": calibration_receipt,
        "trajectory_bundle": trajectory_bundle,
        **(
            {}
            if readiness.compilation_replay_proof is None
            else {"compilation_replay_proof": readiness.compilation_replay_proof}
        ),
        "label_manifest": label_manifest,
        "policy_populations": populations,
        "evaluation_question_ids": split_manifest.evaluation_question_ids,
        "replay_assumptions": REPLAY_ASSUMPTIONS_V1,
        "evaluation_reference_labels_opened": False,
        "policy_predictions_and_certificates_frozen": True,
        "development_and_calibration_label_access_scope": (
            "only_their_prespecified_stages_before_evaluation_freeze"
        ),
        "freeze_portability": (
            "portable_semantic_artifact_excludes_local_custody_device_inode_mode_and_mtime"
        ),
        "real_scored_run_candidate": (
            trajectory_bundle.evidence_kind is BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED
            and readiness.compilation_replay_proof is not None
        ),
        "scientific_claim_authority": False,
    }
    return DecisivePolicyFreezeV1.model_validate(
        {**payload, "freeze_sha256": hash_canonical(payload)}
    )


def replay_decisive_policy_freeze_v1(
    *,
    frozen: DecisivePolicyFreezeV1,
    custody: DecisiveEvaluationReadinessV1,
    repository_root: Path,
    label_root: Path,
    trajectory_compilation_result_path: Path | None = None,
    trajectory_compilation_source_roster_path: Path | None = None,
    trajectory_compilation_source_root: Path | None = None,
) -> DecisivePolicyFreezeV1:
    """Externally recompute schedules after unsealing, without opening label bytes."""

    current_component = compute_decisive_evaluation_component_sha256_v1(repository_root)
    if current_component != frozen.component_sha256:
        raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_component_identity_changed")
    if (
        custody.status != "ready"
        or custody.component_sha256 != frozen.component_sha256
        or custody.portable_semantic_inputs_sha256 != frozen.portable_semantic_inputs_sha256
    ):
        raise DecisiveClaimEvaluationV1Error(
            "decisive_evaluation_v1_local_custody_binding_mismatch"
        )
    compilation_inputs = (
        trajectory_compilation_result_path,
        trajectory_compilation_source_roster_path,
        trajectory_compilation_source_root,
    )
    real_evidence = (
        frozen.trajectory_bundle.evidence_kind is BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED
    )
    if real_evidence:
        if any(value is None for value in compilation_inputs):
            raise DecisiveClaimEvaluationV1Error(
                "decisive_evaluation_v1_real_compilation_replay_inputs_required"
            )
        assert trajectory_compilation_result_path is not None
        assert trajectory_compilation_source_roster_path is not None
        assert trajectory_compilation_source_root is not None
        replayed_compilation = replay_decisive_compilation_lineage_v1(
            config=frozen.config,
            split_manifest=frozen.split_manifest,
            development_receipt=frozen.development_receipt,
            calibration_receipt=frozen.calibration_receipt,
            trajectory_bundle=frozen.trajectory_bundle,
            compiler_result_path=trajectory_compilation_result_path,
            source_roster_path=trajectory_compilation_source_roster_path,
            source_root=trajectory_compilation_source_root,
            repository_root=repository_root,
        )
        if (
            replayed_compilation != frozen.compilation_replay_proof
            or replayed_compilation != custody.compilation_replay_proof
        ):
            raise DecisiveClaimEvaluationV1Error(
                "decisive_evaluation_v1_compilation_external_replay_mismatch"
            )
    elif any(value is not None for value in compilation_inputs):
        raise DecisiveClaimEvaluationV1Error(
            "decisive_evaluation_v1_nonreal_compilation_lineage_forbidden"
        )
    root = _canonical_directory(label_root, "evaluation_label_root")
    root_stat = root.lstat()
    if (
        root_stat.st_dev != custody.label_root_device
        or root_stat.st_ino != custody.label_root_inode
    ):
        raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_label_root_identity_changed")
    stat_by_question = {row.question_id: row for row in custody.sealed_label_stats}
    for entry in frozen.label_manifest.entries:
        target = _safe_label_path(root, entry.relative_path)
        try:
            current = target.lstat()
        except OSError as exc:
            raise DecisiveClaimEvaluationV1Error(
                f"decisive_evaluation_v1_label_file_missing_after_freeze:{entry.question_id}"
            ) from exc
        expected = stat_by_question[entry.question_id]
        if (
            not stat.S_ISREG(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or current.st_nlink != 1
            or current.st_dev != expected.device
            or current.st_ino != expected.inode
            or current.st_size != expected.file_bytes
            or current.st_mtime_ns != expected.mtime_ns
        ):
            raise DecisiveClaimEvaluationV1Error(
                f"decisive_evaluation_v1_label_file_identity_changed:{entry.question_id}"
            )
    arms = {arm.arm_id: arm for arm in required_policy_roster_v1()}
    replayed: list[FrozenPolicyPopulationV1] = []
    for arm_id in frozen.config.required_policy_arm_ids:
        arm = arms[arm_id]
        if arm.score_family is ScoreFamily.AUDIT_EVERYTHING:
            continue
        for budget in frozen.config.budgets_minutes_per_question:
            replayed.append(
                _freeze_policy_population_v1(
                    trajectories=frozen.trajectory_bundle.trajectories,
                    arm=arm,
                    budget_minutes=budget,
                    config=frozen.config,
                )
            )
    replayed.append(
        _freeze_policy_population_v1(
            trajectories=frozen.trajectory_bundle.trajectories,
            arm=arms["audit_everything_upper_bound"],
            budget_minutes=None,
            config=frozen.config,
        )
    )
    if replayed != frozen.policy_populations:
        raise DecisiveClaimEvaluationV1Error(
            "decisive_evaluation_v1_policy_freeze_external_replay_mismatch"
        )
    return frozen


class OpenedEvaluationLabelV1(_FrozenExactModel):
    question_id: Annotated[str, Field(min_length=1)]
    claim_id: Annotated[str, Field(min_length=1)]
    relative_path: Annotated[str, Field(min_length=1)]
    envelope_sha256: Sha256
    envelope_bytes: Literal[4096] = LABEL_ENVELOPE_BYTES
    reference_verdict: ReferenceClaimVerdict
    reference_condition_set_artifact_sha256: Sha256 | None = None
    opened_only_after_freeze_sha256: Sha256
    opened_label_sha256: Sha256

    @model_validator(mode="after")
    def validate_opened(self) -> OpenedEvaluationLabelV1:
        if (
            self.reference_verdict.question_id != self.question_id
            or self.reference_verdict.claim_id != self.claim_id
        ):
            raise ValueError("decisive_evaluation_v1_opened_label_identity_mismatch")
        condition_dependent = (
            self.reference_verdict.verdict is ReferenceClaimVerdictValue.CONDITION_DEPENDENT
        )
        if condition_dependent != (self.reference_condition_set_artifact_sha256 is not None):
            raise ValueError("decisive_evaluation_v1_opened_condition_binding_mismatch")
        _self_hash(self, "opened_label_sha256")
        return self


def _read_exact_regular_file_no_follow(path: Path, expected_stat: os.stat_result) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DecisiveClaimEvaluationV1Error(
            "decisive_evaluation_v1_label_file_open_failed"
        ) from exc
    try:
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_dev != expected_stat.st_dev
            or observed.st_ino != expected_stat.st_ino
            or observed.st_size != expected_stat.st_size
            or observed.st_mtime_ns != expected_stat.st_mtime_ns
        ):
            raise DecisiveClaimEvaluationV1Error(
                "decisive_evaluation_v1_label_file_changed_during_open"
            )
        chunks: list[bytes] = []
        remaining = observed.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_label_file_short_read")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise DecisiveClaimEvaluationV1Error(
                "decisive_evaluation_v1_label_file_grew_during_read"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def open_evaluation_reference_labels_v1(
    *,
    frozen: DecisivePolicyFreezeV1,
    custody: DecisiveEvaluationReadinessV1,
    label_root: Path,
    repository_root: Path,
    trajectory_compilation_result_path: Path | None = None,
    trajectory_compilation_source_roster_path: Path | None = None,
    trajectory_compilation_source_root: Path | None = None,
) -> list[OpenedEvaluationLabelV1]:
    """Replay every pre-open gate, then open exact evaluation verdict files."""

    replay_decisive_policy_freeze_v1(
        frozen=frozen,
        custody=custody,
        repository_root=repository_root,
        label_root=label_root,
        trajectory_compilation_result_path=trajectory_compilation_result_path,
        trajectory_compilation_source_roster_path=(trajectory_compilation_source_roster_path),
        trajectory_compilation_source_root=trajectory_compilation_source_root,
    )

    root = _canonical_directory(label_root, "evaluation_label_root")
    if custody.portable_semantic_inputs_sha256 != frozen.portable_semantic_inputs_sha256:
        raise DecisiveClaimEvaluationV1Error(
            "decisive_evaluation_v1_local_custody_binding_mismatch"
        )
    expected_by_question = {row.question_id: row for row in custody.sealed_label_stats}
    expected_origin = {
        BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED: (EnvelopeNonceOrigin.EXTERNAL_CUSTODIAN),
        BenchmarkEvidenceKind.SIMULATION: (EnvelopeNonceOrigin.PLANTED_SIMULATION_FIXTURE),
        BenchmarkEvidenceKind.DIAGNOSTIC: EnvelopeNonceOrigin.DIAGNOSTIC_FIXTURE,
    }[frozen.trajectory_bundle.evidence_kind]
    output: list[OpenedEvaluationLabelV1] = []
    for entry in frozen.label_manifest.entries:
        target = _safe_label_path(root, entry.relative_path)
        current = target.lstat()
        expected = expected_by_question[entry.question_id]
        if (
            not stat.S_ISREG(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or current.st_nlink != 1
            or current.st_dev != expected.device
            or current.st_ino != expected.inode
            or current.st_size != expected.file_bytes
            or current.st_mtime_ns != expected.mtime_ns
        ):
            raise DecisiveClaimEvaluationV1Error(
                f"decisive_evaluation_v1_label_file_identity_changed:{entry.question_id}"
            )
        if stat.S_IMODE(current.st_mode) & 0o444 == 0:
            raise DecisiveClaimEvaluationV1Error(
                f"decisive_evaluation_v1_label_file_still_sealed:{entry.question_id}"
            )
        content = _read_exact_regular_file_no_follow(target, current)
        digest = hashlib.sha256(content).hexdigest()
        if len(content) != entry.fixed_envelope_bytes or digest != entry.expected_envelope_sha256:
            raise DecisiveClaimEvaluationV1Error(
                f"decisive_evaluation_v1_label_file_hash_mismatch:{entry.question_id}"
            )
        envelope = parse_evaluation_reference_envelope_v1(content)
        verdict = envelope.reference_verdict
        if (
            verdict.question_id != entry.question_id
            or verdict.claim_id != entry.claim_id
            or envelope.nonce_origin is not expected_origin
        ):
            raise DecisiveClaimEvaluationV1Error(
                f"decisive_evaluation_v1_label_content_mismatch:{entry.question_id}"
            )
        payload: dict[str, Any] = {
            "question_id": entry.question_id,
            "claim_id": entry.claim_id,
            "relative_path": entry.relative_path,
            "envelope_sha256": digest,
            "envelope_bytes": LABEL_ENVELOPE_BYTES,
            "reference_verdict": verdict,
            "reference_condition_set_artifact_sha256": (
                envelope.reference_condition_set_artifact_sha256
            ),
            "opened_only_after_freeze_sha256": frozen.freeze_sha256,
        }
        output.append(
            OpenedEvaluationLabelV1.model_validate(
                {**payload, "opened_label_sha256": hash_canonical(payload)}
            )
        )
    return output


class ScoredQuestionOutcomeV1(_FrozenExactModel):
    question_id: Annotated[str, Field(min_length=1)]
    claim_id: Annotated[str, Field(min_length=1)]
    domain: Annotated[str, Field(min_length=1)]
    policy_arm_id: Annotated[str, Field(min_length=1)]
    budget_minutes: float | None
    policy_question_freeze_sha256: Sha256
    reference_verdict_sha256: Sha256
    reference_verdict: Literal[
        "supported",
        "contradicted",
        "condition_dependent",
        "inconclusive",
        "not_evaluable",
    ]
    predicted_classification: Literal[
        "supported",
        "contradicted",
        "condition_dependent",
        "inconclusive",
        "not_evaluable",
    ]
    predicted_condition_set_artifact_sha256: Sha256 | None = None
    reference_condition_set_artifact_sha256: Sha256 | None = None
    released: bool
    abstained: bool
    classification_exact_match: bool
    condition_set_exact_match: bool | None
    decision_exact_match: bool
    released_claim_error: bool
    correct_release: bool
    appropriate_abstention: bool
    missed_correct_decision_abstention: bool
    realized_minutes: Annotated[float, Field(ge=0)]
    selected_actions: Annotated[int, Field(ge=0)]
    resolved_actions: Annotated[int, Field(ge=0)]
    active_action_at_deadline: bool
    outcome_sha256: Sha256

    @model_validator(mode="after")
    def validate_outcome(self) -> ScoredQuestionOutcomeV1:
        if self.released == self.abstained:
            raise ValueError("decisive_evaluation_v1_release_abstention_mismatch")
        expected_classification_match = self.predicted_classification == self.reference_verdict
        if self.classification_exact_match != expected_classification_match:
            raise ValueError("decisive_evaluation_v1_classification_match_mismatch")
        predicted_condition = self.predicted_classification == "condition_dependent"
        reference_condition = self.reference_verdict == "condition_dependent"
        if predicted_condition != (self.predicted_condition_set_artifact_sha256 is not None):
            raise ValueError("decisive_evaluation_v1_predicted_condition_binding_mismatch")
        if reference_condition != (self.reference_condition_set_artifact_sha256 is not None):
            raise ValueError("decisive_evaluation_v1_scored_reference_condition_binding_mismatch")
        expected_condition_match = (
            self.predicted_condition_set_artifact_sha256
            == self.reference_condition_set_artifact_sha256
            if predicted_condition and reference_condition
            else None
        )
        if self.condition_set_exact_match != expected_condition_match:
            raise ValueError("decisive_evaluation_v1_condition_set_match_mismatch")
        expected_decision_match = expected_classification_match and (
            bool(expected_condition_match) if reference_condition else True
        )
        if self.decision_exact_match != expected_decision_match:
            raise ValueError("decisive_evaluation_v1_decision_match_mismatch")
        if self.released_claim_error != (self.released and not self.decision_exact_match):
            raise ValueError("decisive_evaluation_v1_release_error_mismatch")
        if self.correct_release != (self.released and self.decision_exact_match):
            raise ValueError("decisive_evaluation_v1_correct_release_mismatch")
        if self.appropriate_abstention != (self.abstained and not self.decision_exact_match):
            raise ValueError("decisive_evaluation_v1_appropriate_abstention_mismatch")
        if self.missed_correct_decision_abstention != (
            self.abstained and self.decision_exact_match
        ):
            raise ValueError("decisive_evaluation_v1_missed_abstention_mismatch")
        _self_hash(self, "outcome_sha256")
        return self


def _score_question_v1(
    frozen: FrozenPolicyQuestionV1, label: OpenedEvaluationLabelV1
) -> ScoredQuestionOutcomeV1:
    if frozen.question_id != label.question_id or frozen.claim_id != label.claim_id:
        raise DecisiveClaimEvaluationV1Error(
            "decisive_evaluation_v1_score_question_identity_mismatch"
        )
    released = frozen.release_status is ReplayReleaseStatus.RELEASED
    reference_value = label.reference_verdict.verdict.value
    classification_exact = frozen.claim_classification == reference_value
    condition_exact = (
        frozen.condition_set_artifact_sha256 == label.reference_condition_set_artifact_sha256
        if frozen.claim_classification == "condition_dependent"
        and reference_value == "condition_dependent"
        else None
    )
    exact = classification_exact and (
        bool(condition_exact) if reference_value == "condition_dependent" else True
    )
    payload: dict[str, Any] = {
        "question_id": frozen.question_id,
        "claim_id": frozen.claim_id,
        "domain": frozen.domain,
        "policy_arm_id": frozen.policy_arm_id,
        "budget_minutes": frozen.budget_minutes,
        "policy_question_freeze_sha256": frozen.question_freeze_sha256,
        "reference_verdict_sha256": label.reference_verdict.verdict_sha256,
        "reference_verdict": reference_value,
        "predicted_classification": frozen.claim_classification,
        "predicted_condition_set_artifact_sha256": (frozen.condition_set_artifact_sha256),
        "reference_condition_set_artifact_sha256": (label.reference_condition_set_artifact_sha256),
        "released": released,
        "abstained": not released,
        "classification_exact_match": classification_exact,
        "condition_set_exact_match": condition_exact,
        "decision_exact_match": exact,
        "released_claim_error": released and not exact,
        "correct_release": released and exact,
        "appropriate_abstention": not released and not exact,
        "missed_correct_decision_abstention": not released and exact,
        "realized_minutes": frozen.total_realized_minutes,
        "selected_actions": len(frozen.selected_item_ids),
        "resolved_actions": len(frozen.resolved_item_ids),
        "active_action_at_deadline": frozen.active_item_id is not None,
    }
    return ScoredQuestionOutcomeV1.model_validate(
        {**payload, "outcome_sha256": hash_canonical(payload)}
    )


def _aggregate_scored_outcomes_v1(
    outcomes: Sequence[ScoredQuestionOutcomeV1], *, human_minutes: bool
) -> dict[str, Any]:
    n = len(outcomes)
    if n == 0:
        raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_empty_scored_population")
    released = sum(row.released for row in outcomes)
    errors = sum(row.released_claim_error for row in outcomes)
    correct = sum(row.correct_release for row in outcomes)
    appropriate = sum(row.appropriate_abstention for row in outcomes)
    missed = sum(row.missed_correct_decision_abstention for row in outcomes)
    spent = math.fsum(row.realized_minutes for row in outcomes)
    selected = sum(row.selected_actions for row in outcomes)
    resolved = sum(row.resolved_actions for row in outcomes)
    active = sum(row.active_action_at_deadline for row in outcomes)
    budget = outcomes[0].budget_minutes
    if any(row.budget_minutes != budget for row in outcomes):
        raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_metric_budget_mixed")
    capacity = None if budget is None else budget * n
    efficiency = correct * 60.0 / spent if spent > 0 else None
    return {
        "n_complete_independent_questions": n,
        "budget_minutes_per_question": budget,
        "total_available_budget_minutes": capacity,
        "total_realized_minutes": spent,
        "mean_realized_minutes_per_question": spent / n,
        "budget_utilization": (None if capacity is None or capacity == 0 else spent / capacity),
        "selected_audit_actions": selected,
        "resolved_audit_actions": resolved,
        "active_actions_at_deadline": active,
        "released_claims": released,
        "abstained_claims": n - released,
        "release_coverage": released / n,
        "abstention_rate": (n - released) / n,
        "released_claim_errors": errors,
        "released_claim_error": errors / released if released else None,
        "correct_releases": correct,
        "correct_releases_per_human_hour": efficiency if human_minutes else None,
        "correct_releases_per_declared_cost_hour": efficiency,
        "appropriate_abstentions": appropriate,
        "missed_correct_decision_abstentions": missed,
        "abstention_utility": (appropriate - missed) / n,
    }


_INTERVAL_METRICS = (
    "release_coverage",
    "abstention_rate",
    "released_claim_error",
    "correct_releases_per_human_hour",
    "correct_releases_per_declared_cost_hour",
    "abstention_utility",
    "mean_realized_minutes_per_question",
    "budget_utilization",
)


def _bootstrap_seed_v1(base: int, *parts: object) -> int:
    seed_payload = "\0".join(
        ["decisive-question-bootstrap-v1", str(base), *(str(row) for row in parts)]
    )
    digest = hashlib.sha256(seed_payload.encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _question_clustered_intervals_v1(
    outcomes: Sequence[ScoredQuestionOutcomeV1],
    *,
    human_minutes: bool,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    by_domain: dict[str, list[ScoredQuestionOutcomeV1]] = defaultdict(list)
    for row in outcomes:
        by_domain[row.domain].append(row)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {name: [] for name in _INTERVAL_METRICS}
    for _ in range(draws):
        sample: list[ScoredQuestionOutcomeV1] = []
        for domain in sorted(by_domain):
            rows = by_domain[domain]
            indices = rng.integers(0, len(rows), size=len(rows))
            sample.extend(rows[int(index)] for index in indices)
        metrics = _aggregate_scored_outcomes_v1(sample, human_minutes=human_minutes)
        for name in _INTERVAL_METRICS:
            value = metrics[name]
            if value is not None and math.isfinite(value):
                values[name].append(float(value))
    intervals: dict[str, Any] = {}
    for name in _INTERVAL_METRICS:
        valid = values[name]
        interval = None
        if valid:
            lower, upper = np.quantile(valid, [0.025, 0.975], method="linear")
            interval = [float(lower), float(upper)]
        intervals[name] = {
            "confidence_level": 0.95,
            "interval": interval,
            "valid_draws": len(valid),
            "requested_draws": draws,
            "undefined_draws": draws - len(valid),
            "semantics": (
                "question_clustered_stratified_resampling_uncertainty; not an asymptotic, "
                "conformal, causal, or finite-sample coverage guarantee"
            ),
        }
    return {
        "method": "question_clustered_stratified_percentile_bootstrap",
        "cluster_unit": "complete_independent_review_question",
        "strata": "declared_scientific_domain",
        "seed": seed,
        "draws": draws,
        "small_sample_authority": False,
        "intervals": intervals,
    }


class ScoredPolicyPopulationV1(_FrozenExactModel):
    policy_arm: PolicyArmV1
    budget_minutes: float | None
    frozen_population_sha256: Sha256
    question_ids: list[str]
    outcomes: Annotated[list[ScoredQuestionOutcomeV1], Field(min_length=1)]
    metrics: dict[str, Any]
    question_clustered_uncertainty: dict[str, Any]
    scored_population_sha256: Sha256

    @field_validator("question_ids")
    @classmethod
    def validate_question_ids(cls, values: list[str]) -> list[str]:
        return _strict_sorted_unique(values, "scored_policy_question_ids")

    @model_validator(mode="after")
    def validate_scored_population(self) -> ScoredPolicyPopulationV1:
        if self.outcomes != sorted(self.outcomes, key=lambda row: row.question_id):
            raise ValueError("decisive_evaluation_v1_scored_outcomes_not_canonical")
        if [row.question_id for row in self.outcomes] != self.question_ids:
            raise ValueError("decisive_evaluation_v1_scored_population_projection_mismatch")
        if any(
            row.policy_arm_id != self.policy_arm.arm_id or row.budget_minutes != self.budget_minutes
            for row in self.outcomes
        ):
            raise ValueError("decisive_evaluation_v1_scored_population_identity_mismatch")
        _self_hash(self, "scored_population_sha256")
        return self


def _score_policy_population_v1(
    *,
    population: FrozenPolicyPopulationV1,
    labels: Mapping[str, OpenedEvaluationLabelV1],
    human_minutes: bool,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> ScoredPolicyPopulationV1:
    outcomes = [_score_question_v1(row, labels[row.question_id]) for row in population.questions]
    outcomes.sort(key=lambda row: row.question_id)
    metrics = _aggregate_scored_outcomes_v1(outcomes, human_minutes=human_minutes)
    uncertainty = _question_clustered_intervals_v1(
        outcomes,
        human_minutes=human_minutes,
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )
    payload: dict[str, Any] = {
        "policy_arm": population.policy_arm,
        "budget_minutes": population.budget_minutes,
        "frozen_population_sha256": population.population_sha256,
        "question_ids": population.question_ids,
        "outcomes": outcomes,
        "metrics": metrics,
        "question_clustered_uncertainty": uncertainty,
    }
    return ScoredPolicyPopulationV1.model_validate(
        {**payload, "scored_population_sha256": hash_canonical(payload)}
    )


_PAIRED_METRICS = (
    "release_coverage",
    "abstention_rate",
    "released_claim_error",
    "correct_releases_per_human_hour",
    "correct_releases_per_declared_cost_hour",
    "abstention_utility",
    "mean_realized_minutes_per_question",
    "budget_utilization",
)


def _paired_deltas_v1(
    primary: Sequence[ScoredQuestionOutcomeV1],
    baseline: Sequence[ScoredQuestionOutcomeV1],
    *,
    human_minutes: bool,
) -> dict[str, float | None]:
    if [row.question_id for row in primary] != [row.question_id for row in baseline]:
        raise DecisiveClaimEvaluationV1Error(
            "decisive_evaluation_v1_paired_question_population_mismatch"
        )
    if [row.budget_minutes for row in primary] != [row.budget_minutes for row in baseline]:
        raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_paired_budget_mismatch")
    left = _aggregate_scored_outcomes_v1(primary, human_minutes=human_minutes)
    right = _aggregate_scored_outcomes_v1(baseline, human_minutes=human_minutes)
    output: dict[str, float | None] = {}
    for name in _PAIRED_METRICS:
        left_value = left[name]
        right_value = right[name]
        output[name] = (
            None if left_value is None or right_value is None else float(left_value - right_value)
        )
    output["correct_releases_per_question"] = (
        left["correct_releases"] - right["correct_releases"]
    ) / len(primary)
    output["released_claim_errors_per_question"] = (
        left["released_claim_errors"] - right["released_claim_errors"]
    ) / len(primary)
    return output


def _paired_intervals_v1(
    primary: Sequence[ScoredQuestionOutcomeV1],
    baseline: Sequence[ScoredQuestionOutcomeV1],
    *,
    human_minutes: bool,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if [row.question_id for row in primary] != [row.question_id for row in baseline]:
        raise DecisiveClaimEvaluationV1Error(
            "decisive_evaluation_v1_paired_question_population_mismatch"
        )
    by_domain: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(primary):
        if row.domain != baseline[index].domain:
            raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_paired_domain_mismatch")
        by_domain[row.domain].append(index)
    metric_names = (
        *_PAIRED_METRICS,
        "correct_releases_per_question",
        "released_claim_errors_per_question",
    )
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        indices: list[int] = []
        for domain in sorted(by_domain):
            domain_indices = by_domain[domain]
            sampled = rng.integers(0, len(domain_indices), size=len(domain_indices))
            indices.extend(domain_indices[int(index)] for index in sampled)
        deltas = _paired_deltas_v1(
            [primary[index] for index in indices],
            [baseline[index] for index in indices],
            human_minutes=human_minutes,
        )
        for name, value in deltas.items():
            if value is not None and math.isfinite(value):
                values[name].append(value)
    output: dict[str, Any] = {}
    for name in metric_names:
        valid = values[name]
        interval = None
        probability_positive = None
        if valid:
            lower, upper = np.quantile(valid, [0.025, 0.975], method="linear")
            interval = [float(lower), float(upper)]
            probability_positive = sum(value > 0 for value in valid) / len(valid)
        output[name] = {
            "confidence_level": 0.95,
            "interval": interval,
            "valid_draws": len(valid),
            "requested_draws": draws,
            "undefined_draws": draws - len(valid),
            "bootstrap_fraction_delta_gt_zero": probability_positive,
            "semantics": (
                "paired_complete-question resampling uncertainty only; not causal, "
                "asymptotic, conformal, or finite-sample authority"
            ),
        }
    return output


class PairedPolicyComparisonV1(_FrozenExactModel):
    comparison_id: Annotated[str, Field(min_length=1)]
    primary_policy_arm_id: Annotated[str, Field(min_length=1)]
    baseline_policy_arm_id: Annotated[str, Field(min_length=1)]
    budget_minutes_per_question: Annotated[float, Field(ge=0)]
    question_ids: list[str]
    identical_question_population: Literal[True] = True
    identical_budget_cap_and_deadline: Literal[True] = True
    realized_cost_matched: Literal[False] = False
    same_realized_cost_claim_authority: Literal[False] = False
    delta_definition: Literal["primary_minus_baseline"] = "primary_minus_baseline"
    point_deltas: dict[str, float | None]
    paired_question_clustered_uncertainty: dict[str, Any]
    comparison_sha256: Sha256

    @field_validator("question_ids")
    @classmethod
    def validate_question_ids(cls, values: list[str]) -> list[str]:
        return _strict_sorted_unique(values, "paired_question_ids")

    @model_validator(mode="after")
    def validate_comparison(self) -> PairedPolicyComparisonV1:
        _self_hash(self, "comparison_sha256")
        return self


def _freeze_paired_comparison_v1(
    *,
    primary: ScoredPolicyPopulationV1,
    baseline: ScoredPolicyPopulationV1,
    human_minutes: bool,
    draws: int,
    seed: int,
) -> PairedPolicyComparisonV1:
    if (
        primary.budget_minutes is None
        or baseline.budget_minutes is None
        or primary.budget_minutes != baseline.budget_minutes
        or primary.question_ids != baseline.question_ids
    ):
        raise DecisiveClaimEvaluationV1Error(
            "decisive_evaluation_v1_paired_population_or_budget_mismatch"
        )
    comparison_id = (
        f"{primary.policy_arm.arm_id}__minus__{baseline.policy_arm.arm_id}"
        f"__budget_{primary.budget_minutes:g}"
    )
    payload: dict[str, Any] = {
        "comparison_id": comparison_id,
        "primary_policy_arm_id": primary.policy_arm.arm_id,
        "baseline_policy_arm_id": baseline.policy_arm.arm_id,
        "budget_minutes_per_question": primary.budget_minutes,
        "question_ids": primary.question_ids,
        "identical_question_population": True,
        "identical_budget_cap_and_deadline": True,
        "realized_cost_matched": False,
        "same_realized_cost_claim_authority": False,
        "delta_definition": "primary_minus_baseline",
        "point_deltas": _paired_deltas_v1(
            primary.outcomes, baseline.outcomes, human_minutes=human_minutes
        ),
        "paired_question_clustered_uncertainty": {
            "method": "paired_question_clustered_stratified_percentile_bootstrap",
            "seed": seed,
            "draws": draws,
            "small_sample_authority": False,
            "intervals": _paired_intervals_v1(
                primary.outcomes,
                baseline.outcomes,
                human_minutes=human_minutes,
                draws=draws,
                seed=seed,
            ),
        },
    }
    return PairedPolicyComparisonV1.model_validate(
        {**payload, "comparison_sha256": hash_canonical(payload)}
    )


class DecisiveClaimEvaluationResultV1(_FrozenExactModel):
    result_version: Literal["decisive-claim-evaluation-result-v1"] = RESULT_VERSION
    scored_at: datetime
    component_sha256: Sha256
    policy_freeze: DecisivePolicyFreezeV1
    opened_labels: Annotated[list[OpenedEvaluationLabelV1], Field(min_length=1)]
    opened_label_membership_sha256: Sha256
    scored_policy_populations: Annotated[list[ScoredPolicyPopulationV1], Field(min_length=1)]
    paired_policy_comparisons: list[PairedPolicyComparisonV1]
    metric_definitions: dict[str, str]
    evaluation_reference_labels_opened_after_policy_freeze: Literal[True] = True
    evaluation_unit: Literal["complete_independent_review_question"] = (
        "complete_independent_review_question"
    )
    resource_unit: Literal["realized_total_person_minutes"] = "realized_total_person_minutes"
    evidence_kind: BenchmarkEvidenceKind
    scientific_claim_eligible: bool
    released_claim_error_claim_authority: bool
    human_efficiency_claim_authority: bool
    same_realized_cost_claim_authority: bool
    typed_replayed_calibration_artifact_present: Literal[False] = False
    scientific_authority_blocker: Literal[
        "typed_replayed_complete_question_calibration_artifact_not_wired_in_v1"
    ] = "typed_replayed_complete_question_calibration_artifact_not_wired_in_v1"
    claim_release_authority: Literal[False] = False
    expert_labels_fabricated: Literal[False] = False
    empirical_scope: Literal[
        "retrospective_expert_adjudicated_certificate_bound_evaluation",
        "planted_simulation_mechanics_only_non_empirical",
        "diagnostic_proxy_mechanics_only_non_empirical",
    ]
    causal_or_prospective_authority: Literal[False] = False
    small_sample_interval_authority: Literal[False] = False
    result_sha256: Sha256

    @field_validator("scored_at")
    @classmethod
    def validate_scored_at(cls, value: datetime) -> datetime:
        return _aware(value, "evaluation_scored_at")

    @model_validator(mode="after")
    def validate_result(self) -> DecisiveClaimEvaluationResultV1:
        if self.scored_at < self.policy_freeze.frozen_at:
            raise ValueError("decisive_evaluation_v1_score_predates_freeze")
        if self.component_sha256 != self.policy_freeze.component_sha256:
            raise ValueError("decisive_evaluation_v1_result_component_mismatch")
        if self.opened_labels != sorted(self.opened_labels, key=lambda row: row.question_id):
            raise ValueError("decisive_evaluation_v1_opened_labels_not_canonical")
        if [
            row.question_id for row in self.opened_labels
        ] != self.policy_freeze.evaluation_question_ids:
            raise ValueError("decisive_evaluation_v1_opened_label_population_mismatch")
        if self.opened_label_membership_sha256 != hash_canonical(
            [row.opened_label_sha256 for row in self.opened_labels]
        ):
            raise ValueError("decisive_evaluation_v1_opened_label_membership_mismatch")
        expected_scored_keys = [
            (row.policy_arm.arm_id, row.budget_minutes)
            for row in self.policy_freeze.policy_populations
        ]
        observed_scored_keys = [
            (row.policy_arm.arm_id, row.budget_minutes) for row in self.scored_policy_populations
        ]
        if observed_scored_keys != expected_scored_keys:
            raise ValueError("decisive_evaluation_v1_scored_policy_roster_mismatch")
        if any(
            row.question_ids != self.policy_freeze.evaluation_question_ids
            for row in self.scored_policy_populations
        ):
            raise ValueError("decisive_evaluation_v1_scored_policy_population_unequal")
        if any(
            (
                self.scientific_claim_eligible,
                self.released_claim_error_claim_authority,
                self.human_efficiency_claim_authority,
                self.same_realized_cost_claim_authority,
            )
        ):
            raise ValueError("decisive_evaluation_v1_authority_escalation")
        expected_scope = {
            BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED: (
                "retrospective_expert_adjudicated_certificate_bound_evaluation"
            ),
            BenchmarkEvidenceKind.SIMULATION: ("planted_simulation_mechanics_only_non_empirical"),
            BenchmarkEvidenceKind.DIAGNOSTIC: ("diagnostic_proxy_mechanics_only_non_empirical"),
        }[self.evidence_kind]
        if self.empirical_scope != expected_scope:
            raise ValueError("decisive_evaluation_v1_empirical_scope_mismatch")
        _self_hash(self, "result_sha256")
        return self


def _comparison_specs_v1(
    scored: Mapping[tuple[str, float | None], ScoredPolicyPopulationV1],
    budgets: Sequence[float],
) -> list[tuple[str, str, float]]:
    specs: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str, float]] = set()
    matched_ids = [
        row
        for row in _REQUIRED_POLICY_IDS
        if row not in {PRIMARY_POLICY_ARM_ID, "audit_everything_upper_bound"}
    ]
    for budget in budgets:
        for baseline in matched_ids:
            key = (PRIMARY_POLICY_ARM_ID, baseline, budget)
            if key not in seen:
                seen.add(key)
                specs.append(key)
    for family in (
        ScoreFamily.RISK,
        ScoreFamily.DISAGREEMENT,
        ScoreFamily.INFLUENCE,
        ScoreFamily.RISK_X_INFLUENCE,
    ):
        for cost_normalized in (False, True):
            adaptive = _policy_arm_id(
                family=family,
                cost_normalized=cost_normalized,
                adaptation=AdaptationMode.ADAPTIVE,
            )
            static = _policy_arm_id(
                family=family,
                cost_normalized=cost_normalized,
                adaptation=AdaptationMode.STATIC,
            )
            for budget in budgets:
                key = (adaptive, static, budget)
                if key not in seen:
                    seen.add(key)
                    specs.append(key)
    for primary, baseline, budget in specs:
        if (primary, budget) not in scored or (baseline, budget) not in scored:
            raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_comparison_policy_missing")
    return specs


def score_decisive_claim_evaluation_v1(
    *,
    frozen: DecisivePolicyFreezeV1,
    custody: DecisiveEvaluationReadinessV1,
    repository_root: Path,
    label_root: Path,
    scored_at: datetime,
    trajectory_compilation_result_path: Path | None = None,
    trajectory_compilation_source_roster_path: Path | None = None,
    trajectory_compilation_source_root: Path | None = None,
) -> DecisiveClaimEvaluationResultV1:
    """Replay the label-blind freeze, then open references and compute metrics."""

    opened = open_evaluation_reference_labels_v1(
        frozen=frozen,
        custody=custody,
        label_root=label_root,
        repository_root=repository_root,
        trajectory_compilation_result_path=trajectory_compilation_result_path,
        trajectory_compilation_source_roster_path=(trajectory_compilation_source_roster_path),
        trajectory_compilation_source_root=trajectory_compilation_source_root,
    )
    opened.sort(key=lambda row: row.question_id)
    labels = {row.question_id: row for row in opened}
    human_minutes = (
        frozen.trajectory_bundle.evidence_kind is BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED
    )
    scored_populations: list[ScoredPolicyPopulationV1] = []
    for population in frozen.policy_populations:
        seed = _bootstrap_seed_v1(
            frozen.config.bootstrap_seed,
            population.policy_arm.arm_id,
            population.budget_minutes,
        )
        scored_populations.append(
            _score_policy_population_v1(
                population=population,
                labels=labels,
                human_minutes=human_minutes,
                bootstrap_draws=frozen.config.bootstrap_draws,
                bootstrap_seed=seed,
            )
        )
    scored_by_key = {(row.policy_arm.arm_id, row.budget_minutes): row for row in scored_populations}
    comparisons: list[PairedPolicyComparisonV1] = []
    for primary_id, baseline_id, budget in _comparison_specs_v1(
        scored_by_key, frozen.config.budgets_minutes_per_question
    ):
        seed = _bootstrap_seed_v1(
            frozen.config.bootstrap_seed,
            primary_id,
            baseline_id,
            budget,
        )
        comparisons.append(
            _freeze_paired_comparison_v1(
                primary=scored_by_key[(primary_id, budget)],
                baseline=scored_by_key[(baseline_id, budget)],
                human_minutes=human_minutes,
                draws=frozen.config.bootstrap_draws,
                seed=seed,
            )
        )
    evidence_kind = frozen.trajectory_bundle.evidence_kind
    scope = {
        BenchmarkEvidenceKind.REAL_EXPERT_ADJUDICATED: (
            "retrospective_expert_adjudicated_certificate_bound_evaluation"
        ),
        BenchmarkEvidenceKind.SIMULATION: ("planted_simulation_mechanics_only_non_empirical"),
        BenchmarkEvidenceKind.DIAGNOSTIC: ("diagnostic_proxy_mechanics_only_non_empirical"),
    }[evidence_kind]
    payload: dict[str, Any] = {
        "result_version": RESULT_VERSION,
        "scored_at": _canonical_datetime(scored_at),
        "component_sha256": frozen.component_sha256,
        "policy_freeze": frozen,
        "opened_labels": opened,
        "opened_label_membership_sha256": hash_canonical(
            [row.opened_label_sha256 for row in opened]
        ),
        "scored_policy_populations": scored_populations,
        "paired_policy_comparisons": comparisons,
        "metric_definitions": {
            "abstention": "questions not released / complete independent questions",
            "abstention_utility": (
                "(appropriate abstentions minus missed exact-decision abstentions) / "
                "complete independent questions"
            ),
            "budget_use": (
                "realized completed-action minutes plus time consumed by an action still "
                "active at the fixed deadline; estimates are never counted as spending"
            ),
            "correct_releases_per_human_hour": (
                "exact five-way reference-matching releases * 60 / realized total "
                "person-minutes; defined only for expert-adjudicated human-cost evidence"
            ),
            "coverage": "released questions / complete independent questions",
            "cost_comparison_scope": (
                "policies share the same per-question budget cap and deadline, but may use "
                "different realized total person-minutes; v1 has no equal-realized-cost "
                "frontier authority"
            ),
            "condition_dependent_exact_match": (
                "condition_dependent decisions match only when both the five-way class and "
                "the nonce-sealed normalized condition-set artifact SHA match"
            ),
            "released_claim_error": (
                "released five-way decisions unequal to the expert/planted reference / "
                "released decisions; undefined when no decision is released"
            ),
            "static_vs_adaptive": (
                "static arms retain baseline policy-visible scores and estimates; adaptive "
                "arms use the exact rerun state's scores after every completed audit"
            ),
        },
        "evaluation_reference_labels_opened_after_policy_freeze": True,
        "evaluation_unit": "complete_independent_review_question",
        "resource_unit": "realized_total_person_minutes",
        "evidence_kind": evidence_kind,
        "scientific_claim_eligible": False,
        "released_claim_error_claim_authority": False,
        "human_efficiency_claim_authority": False,
        "same_realized_cost_claim_authority": False,
        "typed_replayed_calibration_artifact_present": False,
        "scientific_authority_blocker": (
            "typed_replayed_complete_question_calibration_artifact_not_wired_in_v1"
        ),
        "claim_release_authority": False,
        "expert_labels_fabricated": False,
        "empirical_scope": scope,
        "causal_or_prospective_authority": False,
        "small_sample_interval_authority": False,
    }
    return DecisiveClaimEvaluationResultV1.model_validate(
        {**payload, "result_sha256": hash_canonical(payload)}
    )


def validate_decisive_claim_evaluation_result_v1(
    *,
    result: DecisiveClaimEvaluationResultV1,
    custody: DecisiveEvaluationReadinessV1,
    repository_root: Path,
    label_root: Path,
    trajectory_compilation_result_path: Path | None = None,
    trajectory_compilation_source_roster_path: Path | None = None,
    trajectory_compilation_source_root: Path | None = None,
) -> DecisiveClaimEvaluationResultV1:
    replayed = score_decisive_claim_evaluation_v1(
        frozen=result.policy_freeze,
        custody=custody,
        repository_root=repository_root,
        label_root=label_root,
        scored_at=result.scored_at,
        trajectory_compilation_result_path=trajectory_compilation_result_path,
        trajectory_compilation_source_roster_path=(trajectory_compilation_source_roster_path),
        trajectory_compilation_source_root=trajectory_compilation_source_root,
    )
    if replayed != result:
        raise DecisiveClaimEvaluationV1Error(
            "decisive_evaluation_v1_result_external_replay_mismatch"
        )
    return result


class DecisiveMechanicsFixtureReceiptV1(_FrozenExactModel):
    fixture_version: Literal["decisive-mechanics-fixture-v1"] = "decisive-mechanics-fixture-v1"
    fixture_seed: Literal[20260829] = 20260829
    config_sha256: Sha256
    split_manifest_sha256: Sha256
    development_receipt_sha256: Sha256
    calibration_receipt_sha256: Sha256
    trajectory_bundle_sha256: Sha256
    label_manifest_sha256: Sha256
    readiness_sha256: Sha256
    policy_freeze_sha256: Sha256
    evaluation_result_sha256: Sha256
    evidence_kind: Literal["simulation"] = "simulation"
    logical_fixture_is_deterministic: Literal[True] = True
    seal_identity_is_filesystem_specific: Literal[True] = True
    real_empirical_evidence: Literal[False] = False
    released_claim_error_claim_authority: Literal[False] = False
    human_efficiency_claim_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    expert_labels_fabricated: Literal[False] = False
    fixture_receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_fixture(self) -> DecisiveMechanicsFixtureReceiptV1:
        _self_hash(self, "fixture_receipt_sha256")
        return self


def _fixture_hash(value: str) -> str:
    return hash_canonical({"decisive_mechanics_fixture_v1": value})


def _fixture_state(
    *,
    question_id: str,
    audit_sequence: Sequence[str],
    item_ids: Sequence[str],
    truth: str,
) -> QuestionReplayState:
    resolved_set = set(audit_sequence)
    synthesis_sha256 = _fixture_hash(f"synthesis:{question_id}:{','.join(sorted(resolved_set))}")
    policy_inputs: list[ReplayPolicyInput] = []
    for canonical_order, item_id in enumerate(item_ids, start=1):
        if item_id in resolved_set:
            continue
        suffix = item_id.rsplit("-", maxsplit=1)[-1]
        if not resolved_set:
            risk, disagreement, influence, estimated = {
                "a": (0.90, 0.40, 0.90, 3.0),
                "b": (0.80, 0.70, 0.70, 2.0),
                "c": (0.50, 0.30, 0.80, 1.5),
            }[suffix]
        elif any(value.endswith("-a") for value in resolved_set):
            risk, disagreement, influence, estimated = {
                "b": (0.10, 0.15, 0.20, 2.0),
                "c": (0.95, 0.90, 0.95, 1.5),
            }.get(suffix, (0.30, 0.30, 0.30, 3.0))
        elif any(value.endswith("-b") for value in resolved_set):
            risk, disagreement, influence, estimated = {
                "a": (0.20, 0.20, 0.25, 3.0),
                "c": (0.92, 0.85, 0.90, 1.5),
            }.get(suffix, (0.30, 0.30, 0.30, 2.0))
        else:
            risk, disagreement, influence, estimated = {
                "a": (0.88, 0.80, 0.90, 3.0),
                "b": (0.20, 0.20, 0.20, 2.0),
            }.get(suffix, (0.30, 0.30, 0.30, 1.5))
        policy_inputs.append(
            ReplayPolicyInput(
                item_id=item_id,
                canonical_order=canonical_order,
                risk_score=risk,
                risk_basis="simulation",
                disagreement_score=disagreement,
                influence_score=influence,
                estimated_minutes=estimated,
                eligible=True,
                ineligibility_reasons=[],
                score_state_sha256=synthesis_sha256,
            )
        )
    suffixes = {row.rsplit("-", maxsplit=1)[-1] for row in resolved_set}
    opposite = "contradicted" if truth == "supported" else "supported"
    if len(resolved_set) == len(item_ids) or suffixes == {"a", "c"}:
        classification, released = truth, True
    elif suffixes == {"a", "b"}:
        classification, released = opposite, True
    else:
        classification, released = "inconclusive", False
    return freeze_question_replay_state(
        question_id=question_id,
        pipeline_sha256=_fixture_hash("pipeline"),
        audit_sequence=audit_sequence,
        policy_inputs=policy_inputs,
        release_status=(
            ReplayReleaseStatus.RELEASED if released else ReplayReleaseStatus.ABSTAINED
        ),
        claim_classification=classification,
        release_reasons=[] if released else ["simulation_fixture_not_release_eligible"],
        graph_sha256=_fixture_hash(f"graph:{question_id}:{','.join(sorted(resolved_set))}"),
        synthesis_sha256=synthesis_sha256,
        release_assessment_sha256=_fixture_hash(
            f"release:{question_id}:{','.join(sorted(resolved_set))}"
        ),
        replay_source=ReplaySource.PLANTED_SIMULATION,
    )


def build_decisive_mechanics_fixture_v1(
    *,
    output_root: Path,
    repository_root: Path,
    config: DecisiveEvaluationConfigV1,
) -> DecisiveMechanicsFixtureReceiptV1:
    """Materialize and score planted mechanics data; never create expert evidence."""

    root = Path(os.path.abspath(output_root))
    if root.exists():
        raise DecisiveClaimEvaluationV1Error("decisive_evaluation_v1_fixture_output_must_not_exist")
    root.mkdir(parents=True)
    label_root = root / "evaluation-labels"
    label_root.mkdir()
    pipeline_sha256 = _fixture_hash("pipeline")
    identities: list[QuestionIdentityV1] = []
    counts = {
        StudySplit.DEVELOPMENT: config.minimum_development_questions,
        StudySplit.CALIBRATION: config.minimum_calibration_questions,
        StudySplit.EVALUATION: config.minimum_evaluation_questions,
    }
    for split in StudySplit:
        for index in range(1, counts[split] + 1):
            question_id = f"fixture-{split.value}-{index:03d}"
            identities.append(
                freeze_question_identity_v1(
                    split=split,
                    question_id=question_id,
                    claim_id=f"claim-{split.value}-{index:03d}",
                    domain=f"fixture-domain-{index % 3}",
                    population_id=f"fixture-population-{split.value}-{index:03d}",
                    pipeline_sha256=pipeline_sha256,
                    corpus_sha256=_fixture_hash(f"corpus:{split.value}:{index:03d}"),
                    paper_ids=[f"paper-{split.value}-{index:03d}"],
                    cohort_ids=[f"cohort-{split.value}-{index:03d}"],
                )
            )
    split_manifest = freeze_decisive_split_manifest_v1(
        identities=identities,
        split_salt_sha256=_fixture_hash("split-salt"),
    )
    development_rows = _identities_for_split(split_manifest, StudySplit.DEVELOPMENT)
    calibration_rows = _identities_for_split(split_manifest, StudySplit.CALIBRATION)
    evaluation_rows = _identities_for_split(split_manifest, StudySplit.EVALUATION)
    start = datetime(2026, 8, 29, 0, 0, tzinfo=UTC)
    development_receipt = freeze_fit_stage_receipt_v1(
        stage=FitStage.DEVELOPMENT,
        identities=development_rows,
        pipeline_sha256=pipeline_sha256,
        input_manifest_sha256=split_manifest.manifest_sha256,
        label_source="planted_simulation",
        frozen_optimizer_or_policy_sha256=_fixture_hash("development-optimizer"),
        frozen_threshold_or_bounds_sha256=None,
        completed_at=start,
    )
    calibration_receipt = freeze_fit_stage_receipt_v1(
        stage=FitStage.CALIBRATION,
        identities=calibration_rows,
        pipeline_sha256=pipeline_sha256,
        input_manifest_sha256=split_manifest.manifest_sha256,
        label_source="planted_simulation",
        frozen_optimizer_or_policy_sha256=_fixture_hash("calibrated-policy"),
        frozen_threshold_or_bounds_sha256=_fixture_hash("calibrated-bounds"),
        completed_at=start + timedelta(minutes=1),
    )
    provenance = freeze_decisive_policy_input_provenance_v1(
        development_receipt=development_receipt,
        calibration_receipt=calibration_receipt,
    )
    trajectories: list[QuestionTrajectoryV1] = []
    label_entries: list[EvaluationLabelEntryV1] = []
    for index, identity in enumerate(evaluation_rows, start=1):
        item_ids = [f"{identity.question_id}-{suffix}" for suffix in ("a", "b", "c")]
        events = [
            freeze_question_audit_event(
                item_id=item_id,
                disposition=("corrected" if suffix == "a" else "confirmed"),
                completed_at=start + timedelta(minutes=2 + index),
                realized_minutes={"a": 3.0, "b": 2.0, "c": 1.0}[suffix],
                cost_basis=AuditCostBasis.SIMULATED_MINUTES,
                adjudicator_count=1,
                protocol_sha256=_fixture_hash("audit-protocol"),
                artifact_sha256=_fixture_hash(f"audit:{item_id}"),
                correction_sha256=(
                    _fixture_hash(f"correction:{item_id}") if suffix == "a" else None
                ),
            )
            for item_id, suffix in zip(item_ids, ("a", "b", "c"), strict=True)
        ]
        truth = (
            "condition_dependent"
            if index % 5 == 0
            else ("supported" if index % 2 == 0 else "contradicted")
        )
        reference_condition_sha256 = (
            _fixture_hash(f"condition-set:{identity.question_id}:reference")
            if truth == "condition_dependent"
            else None
        )
        states: list[QuestionReplayState] = []
        for length in range(0, len(item_ids) + 1):
            for sequence in permutations(item_ids, length):
                states.append(
                    _fixture_state(
                        question_id=identity.question_id,
                        audit_sequence=sequence,
                        item_ids=item_ids,
                        truth=truth,
                    )
                )
        condition_by_replay = {
            state.replay_sha256: reference_condition_sha256
            for state in states
            if state.claim_classification == "condition_dependent"
            and reference_condition_sha256 is not None
        }
        trajectories.append(
            freeze_question_trajectory_v1(
                question_identity=identity,
                evidence_kind=BenchmarkEvidenceKind.SIMULATION,
                policy_input_provenance=provenance,
                audit_events=events,
                replay_states=states,
                condition_set_artifact_sha256_by_replay_sha256=condition_by_replay,
            )
        )
        verdict = freeze_reference_claim_verdict(
            question_id=identity.question_id,
            claim_id=identity.claim_id,
            verdict=truth,
            source=ReferenceVerdictSource.PLANTED_SIMULATION,
            adjudicator_count=1,
            protocol_sha256=_fixture_hash("reference-protocol"),
            artifact_sha256=_fixture_hash(f"reference:{identity.question_id}"),
        )
        relative_path = f"{hashlib.sha256(identity.question_id.encode('utf-8')).hexdigest()}.json"
        label_path = label_root / relative_path
        envelope_bytes = freeze_evaluation_reference_envelope_v1(
            reference_verdict=verdict,
            custodian_nonce_hex=_fixture_hash(f"planted-envelope-nonce:{identity.question_id}")[
                :32
            ],
            nonce_origin=EnvelopeNonceOrigin.PLANTED_SIMULATION_FIXTURE,
            reference_condition_set_artifact_sha256=reference_condition_sha256,
        )
        atomic_write_bytes(label_path, envelope_bytes, force=False)
        label_entries.append(
            freeze_evaluation_label_entry_v1(
                question_id=identity.question_id,
                claim_id=identity.claim_id,
                relative_path=relative_path,
                expected_envelope_sha256=sha256_file(label_path),
            )
        )
    trajectory_bundle = freeze_trajectory_bundle_v1(
        split_manifest=split_manifest,
        evidence_kind=BenchmarkEvidenceKind.SIMULATION,
        trajectories=trajectories,
    )
    label_manifest = freeze_evaluation_label_manifest_v1(
        split_manifest_sha256=split_manifest.manifest_sha256,
        evidence_kind=BenchmarkEvidenceKind.SIMULATION,
        entries=label_entries,
    )
    for entry in label_manifest.entries:
        os.chmod(label_root / entry.relative_path, 0)
    readiness = assess_decisive_evaluation_readiness_v1(
        config=config,
        repository_root=repository_root,
        assessed_at=start + timedelta(minutes=2),
        split_manifest=split_manifest,
        development_receipt=development_receipt,
        calibration_receipt=calibration_receipt,
        trajectory_bundle=trajectory_bundle,
        label_manifest=label_manifest,
        label_root=label_root,
    )
    frozen = freeze_decisive_policy_trajectories_v1(
        config=config,
        readiness=readiness,
        split_manifest=split_manifest,
        development_receipt=development_receipt,
        calibration_receipt=calibration_receipt,
        trajectory_bundle=trajectory_bundle,
        label_manifest=label_manifest,
        label_root=label_root,
        repository_root=repository_root,
        frozen_at=start + timedelta(minutes=3),
    )
    for entry in label_manifest.entries:
        os.chmod(label_root / entry.relative_path, 0o400)
    result = score_decisive_claim_evaluation_v1(
        frozen=frozen,
        custody=readiness,
        repository_root=repository_root,
        label_root=label_root,
        scored_at=start + timedelta(minutes=4),
    )
    artifacts: list[tuple[str, object]] = [
        ("config.json", config),
        ("split-manifest.json", split_manifest),
        ("development-receipt.json", development_receipt),
        ("calibration-receipt.json", calibration_receipt),
        ("trajectory-bundle.json", trajectory_bundle),
        ("label-manifest.json", label_manifest),
        ("readiness.json", readiness),
        ("policy-freeze.json", frozen),
        ("evaluation-result.json", result),
    ]
    for name, artifact in artifacts:
        atomic_write_json(root / name, artifact, force=False)
    receipt_payload: dict[str, Any] = {
        "fixture_version": "decisive-mechanics-fixture-v1",
        "fixture_seed": 20260829,
        "config_sha256": config.config_sha256,
        "split_manifest_sha256": split_manifest.manifest_sha256,
        "development_receipt_sha256": development_receipt.receipt_sha256,
        "calibration_receipt_sha256": calibration_receipt.receipt_sha256,
        "trajectory_bundle_sha256": trajectory_bundle.bundle_sha256,
        "label_manifest_sha256": label_manifest.manifest_sha256,
        "readiness_sha256": readiness.readiness_sha256,
        "policy_freeze_sha256": frozen.freeze_sha256,
        "evaluation_result_sha256": result.result_sha256,
        "evidence_kind": "simulation",
        "logical_fixture_is_deterministic": True,
        "seal_identity_is_filesystem_specific": True,
        "real_empirical_evidence": False,
        "released_claim_error_claim_authority": False,
        "human_efficiency_claim_authority": False,
        "claim_release_authority": False,
        "expert_labels_fabricated": False,
    }
    receipt = DecisiveMechanicsFixtureReceiptV1.model_validate(
        {
            **receipt_payload,
            "fixture_receipt_sha256": hash_canonical(receipt_payload),
        }
    )
    atomic_write_json(root / "fixture-receipt.json", receipt, force=False)
    return receipt


__all__ = [
    "AdaptationMode",
    "DecisiveClaimEvaluationResultV1",
    "DecisiveClaimEvaluationV1Error",
    "DecisiveEvaluationConfigV1",
    "DecisiveEvaluationReadinessV1",
    "DecisiveMechanicsFixtureReceiptV1",
    "DecisivePolicyFreezeV1",
    "DecisivePolicyInputProvenanceV1",
    "DecisiveSplitManifestV1",
    "EnvelopeNonceOrigin",
    "EvaluationLabelEntryV1",
    "EvaluationLabelManifestV1",
    "EvaluationReferenceEnvelopeV1",
    "FitStage",
    "FitStageReceiptV1",
    "PolicyArmV1",
    "QuestionIdentityV1",
    "QuestionTrajectoryV1",
    "ReplayConditionSetBindingV1",
    "ScoreFamily",
    "StudySplit",
    "TrajectoryBundleV1",
    "assess_decisive_evaluation_readiness_v1",
    "build_decisive_mechanics_fixture_v1",
    "compute_decisive_evaluation_component_sha256_v1",
    "freeze_decisive_evaluation_config_v1",
    "freeze_decisive_policy_input_provenance_v1",
    "freeze_decisive_policy_trajectories_v1",
    "freeze_decisive_split_manifest_v1",
    "freeze_evaluation_label_entry_v1",
    "freeze_evaluation_label_manifest_v1",
    "freeze_evaluation_reference_envelope_v1",
    "freeze_fit_stage_receipt_v1",
    "freeze_question_identity_v1",
    "freeze_question_trajectory_v1",
    "freeze_trajectory_bundle_v1",
    "parse_evaluation_reference_envelope_v1",
    "replay_decisive_policy_freeze_v1",
    "required_policy_roster_v1",
    "score_decisive_claim_evaluation_v1",
    "validate_decisive_claim_evaluation_result_v1",
]
