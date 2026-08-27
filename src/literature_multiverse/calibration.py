"""Question-level selective-risk calibration for scientific claim release.

The calibration unit is one complete question--corpus run.  Findings, papers,
and bootstrap draws are deliberately rejected as independent calibration units.
The module separates three roles:

* development examples fit a scalar risk score;
* calibration examples choose a release threshold with a simultaneous
  one-sided Clopper--Pearson bound; and
* held-out test examples estimate the resulting risk--coverage trade-off only
  after the fitted model and calibrated policy have been frozen.

The guarantee is about the supplied binary loss label (for example, disagreement
with an adjudicated expert verdict), not scientific truth.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel

SplitName = Literal["development", "calibration", "test"]
LabelSource = Literal["benchmark_annotation", "expert_adjudication", "simulation"]


class CalibrationContractError(ValueError):
    """A calibration input violated independence or lineage requirements."""


class RiskExample(ContractModel):
    """One independent question--corpus outcome used by selective calibration."""

    question_id: str
    split: SplitName
    population_id: str
    domain: str
    pipeline_sha256: str
    paper_ids: list[str]
    features: dict[str, float]
    unsupported_claim: bool
    label_source: LabelSource

    @field_validator("pipeline_sha256")
    @classmethod
    def validate_pipeline_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("invalid_pipeline_sha256")
        return value

    @field_validator("paper_ids")
    @classmethod
    def validate_paper_ids(cls, value: list[str]) -> list[str]:
        if not value or any(not item for item in value):
            raise ValueError("paper_ids_must_be_nonempty")
        if value != sorted(set(value)):
            raise ValueError("paper_ids_must_be_sorted_unique")
        return value

    @field_validator("features")
    @classmethod
    def validate_features(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("risk_features_must_be_nonempty")
        for name, number in value.items():
            if not name or not math.isfinite(number):
                raise ValueError("risk_features_must_be_named_and_finite")
        return dict(sorted(value.items()))


class LogisticRiskModel(ContractModel):
    """JSON-serializable standardized logistic model for unsupported-claim risk."""

    model_version: Literal["logistic-risk-v1"] = "logistic-risk-v1"
    feature_names: list[str]
    means: list[float]
    scales: list[float]
    coefficients: list[float]
    intercept: float
    development_question_ids: list[str]
    pipeline_sha256: str
    population_id: str
    seed: int

    @field_validator("pipeline_sha256")
    @classmethod
    def validate_pipeline_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("invalid_pipeline_sha256")
        return value

    @model_validator(mode="after")
    def validate_dimensions(self) -> LogisticRiskModel:
        width = len(self.feature_names)
        if width == 0 or len(set(self.feature_names)) != width:
            raise ValueError("feature_names_must_be_nonempty_unique")
        if not all(len(values) == width for values in (self.means, self.scales, self.coefficients)):
            raise ValueError("logistic_model_dimension_mismatch")
        if any(scale <= 0 or not math.isfinite(scale) for scale in self.scales):
            raise ValueError("logistic_model_scales_must_be_positive")
        numeric = (*self.means, *self.coefficients, self.intercept)
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("logistic_model_values_must_be_finite")
        if self.development_question_ids != sorted(set(self.development_question_ids)):
            raise ValueError("development_question_ids_must_be_sorted_unique")
        return self

    def score_features(self, features: Mapping[str, float]) -> float:
        """Return a bounded unsupported-claim risk score for one feature row."""

        if set(features) != set(self.feature_names):
            missing = sorted(set(self.feature_names) - set(features))
            extra = sorted(set(features) - set(self.feature_names))
            raise CalibrationContractError(
                f"risk_feature_set_mismatch:missing={missing}:extra={extra}"
            )
        standardized = [
            (float(features[name]) - mean) / scale
            for name, mean, scale in zip(self.feature_names, self.means, self.scales, strict=True)
        ]
        logit = self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, standardized, strict=True)
        )
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)


class ThresholdCandidate(ContractModel):
    threshold: float
    accepted: int = Field(ge=0)
    errors: int = Field(ge=0)
    empirical_risk: float | None
    simultaneous_upper_risk: float | None
    passed: bool

    @model_validator(mode="after")
    def validate_counts(self) -> ThresholdCandidate:
        if self.errors > self.accepted:
            raise ValueError("threshold_errors_exceed_accepted")
        if self.accepted == 0 and (
            self.empirical_risk is not None or self.simultaneous_upper_risk is not None
        ):
            raise ValueError("empty_threshold_risk_must_be_null")
        return self


class CalibratedReleasePolicy(ContractModel):
    """Frozen release threshold and its calibration evidence."""

    policy_version: Literal["question-risk-ltt-v1"] = "question-risk-ltt-v1"
    alpha: float = Field(gt=0, lt=1)
    delta: float = Field(gt=0, lt=1)
    threshold: float | None
    selected: ThresholdCandidate | None
    candidates: list[ThresholdCandidate]
    calibration_question_ids: list[str]
    population_id: str
    pipeline_sha256: str
    score_model_sha256: str
    correction: Literal["bonferroni-clopper-pearson"] = "bonferroni-clopper-pearson"
    status: Literal["calibrated", "abstain_all"]

    @field_validator("pipeline_sha256", "score_model_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("invalid_policy_sha256")
        return value

    @model_validator(mode="after")
    def validate_selection(self) -> CalibratedReleasePolicy:
        if not self.candidates:
            raise ValueError("calibration_candidate_family_empty")
        thresholds = [candidate.threshold for candidate in self.candidates]
        if thresholds != sorted(set(thresholds)):
            raise ValueError("calibration_candidates_must_be_sorted_unique")
        simultaneous_delta = self.delta / len(self.candidates)
        for candidate in self.candidates:
            if candidate.accepted > len(self.calibration_question_ids):
                raise ValueError("calibration_candidate_acceptance_exceeds_split")
            expected_empirical = (
                candidate.errors / candidate.accepted if candidate.accepted else None
            )
            if candidate.empirical_risk != expected_empirical:
                raise ValueError("calibration_candidate_empirical_risk_mismatch")
            expected_upper = (
                clopper_pearson_upper(
                    candidate.errors,
                    candidate.accepted,
                    delta=simultaneous_delta,
                )
                if candidate.accepted
                else None
            )
            if (
                candidate.simultaneous_upper_risk is None
                if expected_upper is not None
                else candidate.simultaneous_upper_risk is not None
            ):
                raise ValueError("calibration_candidate_upper_risk_mismatch")
            if expected_upper is not None and not math.isclose(
                candidate.simultaneous_upper_risk or 0.0,
                expected_upper,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError("calibration_candidate_upper_risk_mismatch")
            if candidate.passed != (expected_upper is not None and expected_upper <= self.alpha):
                raise ValueError("calibration_candidate_pass_status_mismatch")

        calibrated = self.status == "calibrated"
        if calibrated != (self.threshold is not None and self.selected is not None):
            raise ValueError("calibration_status_selection_mismatch")
        if self.calibration_question_ids != sorted(set(self.calibration_question_ids)):
            raise ValueError("calibration_question_ids_must_be_sorted_unique")
        passing = [candidate for candidate in self.candidates if candidate.passed]
        expected_selected = (
            max(
                passing,
                key=lambda candidate: (
                    candidate.accepted,
                    -float(candidate.simultaneous_upper_risk or 1.0),
                    candidate.threshold,
                ),
            )
            if passing
            else None
        )
        if self.selected != expected_selected:
            raise ValueError("calibration_selected_candidate_mismatch")
        if self.threshold != (
            None if expected_selected is None else expected_selected.threshold
        ):
            raise ValueError("calibration_selected_threshold_mismatch")
        return self


class PolicyEvaluation(ContractModel):
    split: Literal["test"] = "test"
    total: int = Field(ge=1)
    accepted: int = Field(ge=0)
    errors: int = Field(ge=0)
    coverage: float = Field(ge=0, le=1)
    empirical_risk: float | None
    risk_interval_95: tuple[float, float] | None
    by_domain: dict[str, dict[str, float | int | None]]
    test_question_ids: list[str]


class FrozenSplitIdentity(ContractModel):
    """The minimum identity ledger needed to reject later test overlap."""

    split: Literal["development", "calibration"]
    question_ids: list[str]
    paper_ids: list[str]

    @model_validator(mode="after")
    def validate_identities(self) -> FrozenSplitIdentity:
        if not self.question_ids or self.question_ids != sorted(set(self.question_ids)):
            raise ValueError("frozen_question_ids_must_be_nonempty_sorted_unique")
        if not self.paper_ids or self.paper_ids != sorted(set(self.paper_ids)):
            raise ValueError("frozen_paper_ids_must_be_nonempty_sorted_unique")
        return self


class FrozenCalibrationBundle(ContractModel):
    """Self-hashed development/calibration state frozen before test access."""

    bundle_version: Literal["question-risk-freeze-v1"] = "question-risk-freeze-v1"
    freeze_state: Literal["test_labels_unopened"] = "test_labels_unopened"
    population_id: str
    pipeline_sha256: str
    label_source: LabelSource
    feature_names: list[str]
    development: FrozenSplitIdentity
    calibration: FrozenSplitIdentity
    development_calibration_input_sha256: str
    score_model: LogisticRiskModel
    score_model_sha256: str
    policy: CalibratedReleasePolicy
    policy_sha256: str
    bundle_sha256: str

    @field_validator(
        "pipeline_sha256",
        "development_calibration_input_sha256",
        "score_model_sha256",
        "policy_sha256",
        "bundle_sha256",
    )
    @classmethod
    def validate_bundle_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("invalid_frozen_bundle_sha256")
        return value

    @model_validator(mode="after")
    def validate_bundle_lineage(self) -> FrozenCalibrationBundle:
        if self.feature_names != sorted(set(self.feature_names)) or not self.feature_names:
            raise ValueError("frozen_feature_names_must_be_nonempty_sorted_unique")
        if self.development.split != "development":
            raise ValueError("frozen_development_identity_split_mismatch")
        if self.calibration.split != "calibration":
            raise ValueError("frozen_calibration_identity_split_mismatch")
        if set(self.development.question_ids) & set(self.calibration.question_ids):
            raise ValueError("frozen_question_identity_overlap")
        if set(self.development.paper_ids) & set(self.calibration.paper_ids):
            raise ValueError("frozen_paper_identity_overlap")
        if self.score_model.feature_names != self.feature_names:
            raise ValueError("frozen_score_model_feature_schema_mismatch")
        if self.score_model.development_question_ids != self.development.question_ids:
            raise ValueError("frozen_score_model_development_identity_mismatch")
        if self.policy.calibration_question_ids != self.calibration.question_ids:
            raise ValueError("frozen_policy_calibration_identity_mismatch")
        if self.score_model.pipeline_sha256 != self.pipeline_sha256:
            raise ValueError("frozen_score_model_pipeline_mismatch")
        if self.policy.pipeline_sha256 != self.pipeline_sha256:
            raise ValueError("frozen_policy_pipeline_mismatch")
        if self.score_model.population_id != self.population_id:
            raise ValueError("frozen_score_model_population_mismatch")
        if self.policy.population_id != self.population_id:
            raise ValueError("frozen_policy_population_mismatch")
        if hash_canonical(self.score_model) != self.score_model_sha256:
            raise ValueError("frozen_score_model_hash_mismatch")
        if hash_canonical(self.policy) != self.policy_sha256:
            raise ValueError("frozen_policy_hash_mismatch")
        if self.policy.score_model_sha256 != self.score_model_sha256:
            raise ValueError("frozen_policy_score_model_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"bundle_sha256"})
        if hash_canonical(payload) != self.bundle_sha256:
            raise ValueError("frozen_bundle_hash_mismatch")
        return self


@dataclass(frozen=True, slots=True)
class ScoredExample:
    example: RiskExample
    score: float


def _validate_probability(value: float, name: str) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name}_must_be_between_zero_and_one")


def validate_split_integrity(
    examples: Sequence[RiskExample], *, require_disjoint_question_papers: bool = False
) -> None:
    """Reject split leakage and, when requested, dependent question corpora."""

    if not examples:
        raise CalibrationContractError("risk_examples_empty")
    by_question: dict[str, list[RiskExample]] = defaultdict(list)
    for example in examples:
        by_question[example.question_id].append(example)
    duplicated = sorted(question for question, rows in by_question.items() if len(rows) != 1)
    if duplicated:
        raise CalibrationContractError(f"question_not_independent:{duplicated}")

    pipelines = {example.pipeline_sha256 for example in examples}
    if len(pipelines) != 1:
        raise CalibrationContractError("pipeline_changed_across_risk_splits")
    populations = {example.population_id for example in examples}
    if len(populations) != 1:
        raise CalibrationContractError("population_changed_across_risk_splits")
    label_sources = {example.label_source for example in examples}
    if len(label_sources) != 1:
        raise CalibrationContractError("label_source_changed_across_risk_splits")

    feature_sets = {tuple(example.features) for example in examples}
    if len(feature_sets) != 1:
        raise CalibrationContractError("risk_feature_schema_changed_across_splits")

    paper_owner: dict[str, tuple[SplitName, str]] = {}
    for example in examples:
        for paper_id in example.paper_ids:
            prior = paper_owner.get(paper_id)
            if prior is not None:
                if prior[0] != example.split:
                    raise CalibrationContractError(
                        "paper_crosses_risk_split:"
                        f"{paper_id}:{prior[0]}:{example.split}:{prior[1]}:{example.question_id}"
                    )
                if require_disjoint_question_papers and prior[1] != example.question_id:
                    raise CalibrationContractError(
                        "paper_shared_between_question_units:"
                        f"{paper_id}:{example.split}:{prior[1]}:{example.question_id}"
                    )
            paper_owner[paper_id] = (example.split, example.question_id)


def _split(examples: Sequence[RiskExample], split: SplitName) -> list[RiskExample]:
    return sorted(
        (example for example in examples if example.split == split),
        key=lambda example: example.question_id,
    )


def fit_logistic_risk_model(
    examples: Sequence[RiskExample], *, seed: int = 20260826
) -> LogisticRiskModel:
    """Fit a scalar risk score only on the development question set."""

    validate_split_integrity(examples, require_disjoint_question_papers=True)
    development = _split(examples, "development")
    if len(development) < 4:
        raise CalibrationContractError("development_requires_at_least_four_questions")
    labels = np.asarray([int(example.unsupported_claim) for example in development])
    if len(set(labels.tolist())) != 2:
        raise CalibrationContractError("development_requires_both_loss_classes")
    feature_names = list(development[0].features)
    matrix = np.asarray(
        [[example.features[name] for name in feature_names] for example in development],
        dtype=float,
    )
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales == 0] = 1.0
    standardized = (matrix - means) / scales
    estimator = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        solver="liblinear",
    )
    estimator.fit(standardized, labels)
    first = examples[0]
    return LogisticRiskModel(
        feature_names=feature_names,
        means=means.tolist(),
        scales=scales.tolist(),
        coefficients=estimator.coef_[0].tolist(),
        intercept=float(estimator.intercept_[0]),
        development_question_ids=[example.question_id for example in development],
        pipeline_sha256=first.pipeline_sha256,
        population_id=first.population_id,
        seed=seed,
    )


def score_examples(
    examples: Iterable[RiskExample], model: LogisticRiskModel
) -> list[ScoredExample]:
    return [
        ScoredExample(example=example, score=model.score_features(example.features))
        for example in examples
    ]


def clopper_pearson_interval(
    errors: int, total: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Exact two-sided binomial interval for descriptive test reporting."""

    _validate_probability(confidence, "confidence")
    if total <= 0 or errors < 0 or errors > total:
        raise ValueError("invalid_binomial_counts")
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if errors == 0 else float(beta.ppf(tail, errors, total - errors + 1))
    upper = 1.0 if errors == total else float(beta.ppf(1.0 - tail, errors + 1, total - errors))
    return lower, upper


def clopper_pearson_upper(errors: int, total: int, *, delta: float) -> float:
    """One-sided exact upper confidence bound for a Bernoulli loss rate."""

    _validate_probability(delta, "delta")
    if total <= 0 or errors < 0 or errors > total:
        raise ValueError("invalid_binomial_counts")
    if errors == total:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, total - errors))


def calibrate_release_policy(
    examples: Sequence[RiskExample],
    model: LogisticRiskModel,
    *,
    alpha: float,
    delta: float,
    candidate_thresholds: Sequence[float] | None = None,
) -> CalibratedReleasePolicy:
    """Choose maximum calibrated coverage with family-wise risk control.

    Candidate thresholds are tested simultaneously using Bonferroni-corrected
    one-sided Clopper--Pearson bounds. If no family is supplied, its thresholds
    are learned from development scores, independently of calibration outcomes.
    If none certifies risk <= ``alpha``, the returned policy abstains on every
    question.
    """

    _validate_probability(alpha, "alpha")
    _validate_probability(delta, "delta")
    validate_split_integrity(examples, require_disjoint_question_papers=True)
    calibration = _split(examples, "calibration")
    if not calibration:
        raise CalibrationContractError("calibration_split_empty")
    if model.pipeline_sha256 != examples[0].pipeline_sha256:
        raise CalibrationContractError("score_model_pipeline_mismatch")
    if model.population_id != examples[0].population_id:
        raise CalibrationContractError("score_model_population_mismatch")
    development = _split(examples, "development")
    if [example.question_id for example in development] != model.development_question_ids:
        raise CalibrationContractError("score_model_development_identity_mismatch")
    scored = score_examples(calibration, model)
    if candidate_thresholds is None:
        if not development:
            raise CalibrationContractError(
                "default_candidate_thresholds_require_development_split"
            )
        thresholds = sorted({row.score for row in score_examples(development, model)})
    else:
        thresholds = sorted(set(float(value) for value in candidate_thresholds))
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise CalibrationContractError("candidate_thresholds_invalid")
    simultaneous_delta = delta / len(thresholds)
    candidates: list[ThresholdCandidate] = []
    for threshold in thresholds:
        accepted_rows = [row for row in scored if row.score <= threshold]
        accepted = len(accepted_rows)
        errors = sum(row.example.unsupported_claim for row in accepted_rows)
        if accepted:
            empirical = errors / accepted
            upper = clopper_pearson_upper(errors, accepted, delta=simultaneous_delta)
            passed = upper <= alpha
        else:
            empirical = None
            upper = None
            passed = False
        candidates.append(
            ThresholdCandidate(
                threshold=threshold,
                accepted=accepted,
                errors=errors,
                empirical_risk=empirical,
                simultaneous_upper_risk=upper,
                passed=passed,
            )
        )
    passing = [candidate for candidate in candidates if candidate.passed]
    selected = (
        max(
            passing,
            key=lambda candidate: (
                candidate.accepted,
                -float(candidate.simultaneous_upper_risk or 1.0),
                candidate.threshold,
            ),
        )
        if passing
        else None
    )
    model_hash = hash_canonical(model)
    return CalibratedReleasePolicy(
        alpha=alpha,
        delta=delta,
        threshold=None if selected is None else selected.threshold,
        selected=selected,
        candidates=candidates,
        calibration_question_ids=[example.question_id for example in calibration],
        population_id=examples[0].population_id,
        pipeline_sha256=examples[0].pipeline_sha256,
        score_model_sha256=model_hash,
        status="abstain_all" if selected is None else "calibrated",
    )


def freeze_calibration_bundle(
    examples: Sequence[RiskExample],
    *,
    alpha: float,
    delta: float,
    seed: int = 20260826,
    candidate_thresholds: Sequence[float] | None = None,
) -> FrozenCalibrationBundle:
    """Fit and calibrate from a file that physically excludes test rows.

    The returned bundle records the development/calibration question and paper
    identities needed by a later process to reject held-out overlap. Its
    ``bundle_sha256`` covers every field except the hash itself.
    """

    rows = sorted(examples, key=lambda row: (row.split, row.question_id))
    observed_splits = {row.split for row in rows}
    if "test" in observed_splits:
        raise CalibrationContractError("freeze_input_must_exclude_test_rows")
    if observed_splits != {"development", "calibration"}:
        raise CalibrationContractError("freeze_input_requires_development_and_calibration_only")
    validate_split_integrity(rows, require_disjoint_question_papers=True)
    model = fit_logistic_risk_model(rows, seed=seed)
    policy = calibrate_release_policy(
        rows,
        model,
        alpha=alpha,
        delta=delta,
        candidate_thresholds=candidate_thresholds,
    )
    development = _split(rows, "development")
    calibration = _split(rows, "calibration")
    development_identity = FrozenSplitIdentity(
        split="development",
        question_ids=[row.question_id for row in development],
        paper_ids=sorted({paper_id for row in development for paper_id in row.paper_ids}),
    )
    calibration_identity = FrozenSplitIdentity(
        split="calibration",
        question_ids=[row.question_id for row in calibration],
        paper_ids=sorted({paper_id for row in calibration for paper_id in row.paper_ids}),
    )
    payload: dict[str, Any] = {
        "bundle_version": "question-risk-freeze-v1",
        "freeze_state": "test_labels_unopened",
        "population_id": rows[0].population_id,
        "pipeline_sha256": rows[0].pipeline_sha256,
        "label_source": rows[0].label_source,
        "feature_names": model.feature_names,
        "development": development_identity,
        "calibration": calibration_identity,
        "development_calibration_input_sha256": hash_canonical(rows),
        "score_model": model,
        "score_model_sha256": hash_canonical(model),
        "policy": policy,
        "policy_sha256": hash_canonical(policy),
    }
    return FrozenCalibrationBundle.model_validate(
        {**payload, "bundle_sha256": hash_canonical(payload)}
    )


def validate_frozen_test_examples(
    examples: Sequence[RiskExample], bundle: FrozenCalibrationBundle
) -> list[RiskExample]:
    """Validate held-out rows against the identities and schema frozen earlier."""

    rows = sorted(examples, key=lambda row: row.question_id)
    if not rows:
        raise CalibrationContractError("test_split_empty")
    if {row.split for row in rows} != {"test"}:
        raise CalibrationContractError("test_input_must_contain_test_rows_only")
    validate_split_integrity(rows, require_disjoint_question_papers=True)
    if {row.pipeline_sha256 for row in rows} != {bundle.pipeline_sha256}:
        raise CalibrationContractError("frozen_test_pipeline_mismatch")
    if {row.population_id for row in rows} != {bundle.population_id}:
        raise CalibrationContractError("frozen_test_population_mismatch")
    if {row.label_source for row in rows} != {bundle.label_source}:
        raise CalibrationContractError("frozen_test_label_source_mismatch")
    if any(list(row.features) != bundle.feature_names for row in rows):
        raise CalibrationContractError("frozen_test_feature_schema_mismatch")

    frozen_questions = set(bundle.development.question_ids) | set(bundle.calibration.question_ids)
    test_questions = {row.question_id for row in rows}
    question_overlap = sorted(frozen_questions & test_questions)
    if question_overlap:
        raise CalibrationContractError(f"frozen_test_question_overlap:{question_overlap}")
    frozen_papers = set(bundle.development.paper_ids) | set(bundle.calibration.paper_ids)
    test_papers = {paper_id for row in rows for paper_id in row.paper_ids}
    paper_overlap = sorted(frozen_papers & test_papers)
    if paper_overlap:
        raise CalibrationContractError(f"frozen_test_paper_overlap:{paper_overlap}")
    return rows


def _evaluate_rows(
    rows: Sequence[ScoredExample], threshold: float | None
) -> dict[str, float | int | tuple[float, float] | None]:
    accepted_rows = [] if threshold is None else [row for row in rows if row.score <= threshold]
    accepted = len(accepted_rows)
    errors = sum(row.example.unsupported_claim for row in accepted_rows)
    return {
        "total": len(rows),
        "accepted": accepted,
        "errors": errors,
        "coverage": accepted / len(rows) if rows else 0.0,
        "empirical_risk": errors / accepted if accepted else None,
        "risk_interval_95": (clopper_pearson_interval(errors, accepted) if accepted else None),
    }


def evaluate_release_policy(
    examples: Sequence[RiskExample],
    model: LogisticRiskModel,
    policy: CalibratedReleasePolicy,
) -> PolicyEvaluation:
    """Evaluate a frozen policy once on held-out test question--corpora."""

    validate_split_integrity(examples, require_disjoint_question_papers=True)
    test = _split(examples, "test")
    if not test:
        raise CalibrationContractError("test_split_empty")
    if hash_canonical(model) != policy.score_model_sha256:
        raise CalibrationContractError("policy_score_model_hash_mismatch")
    if policy.pipeline_sha256 != examples[0].pipeline_sha256:
        raise CalibrationContractError("policy_pipeline_mismatch")
    if policy.population_id != examples[0].population_id:
        raise CalibrationContractError("policy_population_mismatch")
    scored = score_examples(test, model)
    overall = _evaluate_rows(scored, policy.threshold)
    by_domain: dict[str, dict[str, float | int | None]] = {}
    for domain in sorted({row.example.domain for row in scored}):
        summary = _evaluate_rows(
            [row for row in scored if row.example.domain == domain], policy.threshold
        )
        interval = summary.pop("risk_interval_95")
        if isinstance(interval, tuple):
            summary["risk_interval_lower_95"] = interval[0]
            summary["risk_interval_upper_95"] = interval[1]
        else:
            summary["risk_interval_lower_95"] = None
            summary["risk_interval_upper_95"] = None
        by_domain[domain] = summary  # type: ignore[assignment]
    return PolicyEvaluation(
        total=int(overall["total"]),
        accepted=int(overall["accepted"]),
        errors=int(overall["errors"]),
        coverage=float(overall["coverage"]),
        empirical_risk=(
            None if overall["empirical_risk"] is None else float(overall["empirical_risk"])
        ),
        risk_interval_95=overall["risk_interval_95"],  # type: ignore[arg-type]
        by_domain=by_domain,
        test_question_ids=[example.question_id for example in test],
    )


def evaluate_frozen_calibration_bundle(
    examples: Sequence[RiskExample], bundle: FrozenCalibrationBundle
) -> dict[str, Any]:
    """Evaluate test-only input and bind the result to the prior freeze hash."""

    rows = validate_frozen_test_examples(examples, bundle)
    evaluation = evaluate_release_policy(rows, bundle.score_model, bundle.policy)
    return {
        "calibration_test_artifact_version": "1",
        "evaluation_stage": "held_out_test_after_freeze",
        "guarantee_scope": (
            "descriptive held-out estimate for the frozen label-risk policy; not a new "
            "calibration guarantee, scientific-truth guarantee, or robustness guarantee "
            "under unknown distribution shift"
        ),
        "frozen_bundle_sha256": bundle.bundle_sha256,
        "score_model_sha256": bundle.score_model_sha256,
        "policy_sha256": bundle.policy_sha256,
        "population_id": bundle.population_id,
        "pipeline_sha256": bundle.pipeline_sha256,
        "label_source": bundle.label_source,
        "test_input_sha256": hash_canonical(rows),
        "test_evaluation": evaluation.model_dump(mode="json"),
        "test_risk_coverage_curve": risk_coverage_curve(rows, bundle.score_model, split="test"),
    }


def risk_coverage_curve(
    examples: Sequence[RiskExample],
    model: LogisticRiskModel,
    *,
    split: Literal["calibration", "test"] = "test",
) -> list[dict[str, float | int | None]]:
    """Return every empirical operating point for plotting, without calibration claims."""

    validate_split_integrity(examples, require_disjoint_question_papers=True)
    rows = score_examples(_split(examples, split), model)
    points: list[dict[str, float | int | None]] = []
    for threshold in sorted({row.score for row in rows}):
        summary = _evaluate_rows(rows, threshold)
        points.append(
            {
                "threshold": threshold,
                "accepted": int(summary["accepted"]),
                "errors": int(summary["errors"]),
                "coverage": float(summary["coverage"]),
                "empirical_risk": (
                    None if summary["empirical_risk"] is None else float(summary["empirical_risk"])
                ),
            }
        )
    return points


def calibration_artifact(
    *,
    examples: Sequence[RiskExample],
    model: LogisticRiskModel,
    policy: CalibratedReleasePolicy,
    evaluation: PolicyEvaluation,
) -> dict[str, Any]:
    """Build a one-shot diagnostic artifact.

    This compatibility helper receives all three splits together. Empirical
    evaluation must use :func:`freeze_calibration_bundle` followed by
    :func:`evaluate_frozen_calibration_bundle` so test labels are not opened
    before the decision rule is frozen.
    """

    validate_split_integrity(examples, require_disjoint_question_papers=True)
    if {row.label_source for row in examples} != {"simulation"}:
        raise CalibrationContractError("one_shot_artifact_requires_simulation_labels")
    return {
        "calibration_artifact_version": "1",
        "loss_definition": "binary unsupported-claim label on one question-corpus",
        "guarantee_scope": (
            "simultaneous calibration bound under independent exchangeable question-corpora; "
            "not a guarantee of scientific truth or robustness to unknown distribution shift"
        ),
        "label_source": examples[0].label_source,
        "population_id": examples[0].population_id,
        "pipeline_sha256": examples[0].pipeline_sha256,
        "input_sha256": hash_canonical(list(examples)),
        "score_model": model.model_dump(mode="json"),
        "score_model_sha256": hash_canonical(model),
        "policy": policy.model_dump(mode="json"),
        "policy_sha256": hash_canonical(policy),
        "test_evaluation": evaluation.model_dump(mode="json"),
        "test_risk_coverage_curve": risk_coverage_curve(examples, model, split="test"),
    }


__all__ = [
    "CalibratedReleasePolicy",
    "CalibrationContractError",
    "FrozenCalibrationBundle",
    "FrozenSplitIdentity",
    "LogisticRiskModel",
    "PolicyEvaluation",
    "RiskExample",
    "calibrate_release_policy",
    "calibration_artifact",
    "clopper_pearson_interval",
    "clopper_pearson_upper",
    "evaluate_frozen_calibration_bundle",
    "evaluate_release_policy",
    "fit_logistic_risk_model",
    "freeze_calibration_bundle",
    "risk_coverage_curve",
    "score_examples",
    "validate_frozen_test_examples",
    "validate_split_integrity",
]
