"""Proof-carrying calibration of group-average evidence-item error-rate UCLs.

The raw item score only chooses a bin.  It is never interpreted as an error
probability.  A simultaneous cell-rate upper confidence limit can only be emitted
from a validated, self-hashed bundle built from question- and paper-disjoint
adjudicated units with fixed bins and one-sided Clopper--Pearson bounds.  That UCL
estimates a group-average rate; it is scheduling/blocking evidence, never an
individual-item or claim-decision risk bound.

This deliberately conservative contract uses at most one calibration item per
question and per paper.  Correlated rows must not be presented as independent
calibration evidence merely because they are separate estimates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator
from scipy.stats import beta

from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.pipeline_fingerprint import (
    PipelineFingerprintVerification,
    validate_pipeline_verification_integrity,
)

CalibrationSplit = Literal["development", "calibration"]
AdjudicatedLabelSource = Literal["benchmark_annotation", "expert_adjudication"]
RiskBoundStatus = Literal[
    "cell_rate_ucl_available",
    "empty_calibration_bin",
    "pipeline_mismatch",
    "score_model_mismatch",
    "calibration_question_overlap",
    "calibration_paper_overlap",
    "population_mismatch",
    "domain_out_of_scope",
    "shift_not_assessed",
    "shift_detected",
    "shift_assessment_mismatch",
]

_BOUND_ASSUMPTIONS = [
    "one_calibration_unit_per_unique_question_and_unique_paper",
    "calibration_units_exchangeable_with_deployment_within_population_domain_and_score_bin",
    "score_model_pipeline_and_bin_family_frozen_before_calibration_labels",
    "observed_error_matches_the_prespecified_adjudication_error_event",
    "bonferroni_simultaneous_one_sided_clopper_pearson_bounds",
    "bound_invalid_when_population_domain_or_shift_contract_fails",
    "ucl_estimand_is_group_average_error_rate_within_domain_and_score_bin",
    "ucl_is_not_an_individual_item_marginal_or_conditional_error_probability",
    "adaptive_selection_can_change_the_unresolved_subset_error_distribution",
    "ucl_is_scheduling_and_blocking_evidence_not_claim_release_authority",
]

_CELL_RATE_ESTIMAND = "group_average_item_error_rate_within_domain_score_bin"
_SCORE_SEMANTICS = "externally_supplied_scheduling_score_not_recomputed"


class ItemRiskCalibrationError(ValueError):
    """Calibration evidence or a prospective scoring request failed its contract."""


def _validate_sha256(value: str, field_name: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"invalid_sha256:{field_name}")
    return value


def _validate_optional_sha256(value: str | None, field_name: str) -> str | None:
    if value is not None:
        _validate_sha256(value, field_name)
    return value


def _validate_sorted_unique_nonempty(values: list[str], field_name: str) -> list[str]:
    if not values or any(not value for value in values):
        raise ValueError(f"{field_name}_must_be_nonempty")
    if values != sorted(set(values)):
        raise ValueError(f"{field_name}_must_be_sorted_unique")
    return values


def _one_sided_clopper_pearson_upper(errors: int, total: int, *, delta: float) -> float:
    if total <= 0 or errors < 0 or errors > total:
        raise ValueError("invalid_binomial_counts")
    if not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("invalid_binomial_delta")
    if errors == total:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, total - errors))


class RiskBinSpec(ContractModel):
    """One interval in a predeclared partition of the raw score range [0, 1]."""

    bin_id: Annotated[str, Field(pattern=r"^risk-bin-[0-9]{3}$")]
    lower: Annotated[float, Field(ge=0, le=1)]
    upper: Annotated[float, Field(gt=0, le=1)]
    upper_inclusive: bool = False

    @model_validator(mode="after")
    def validate_interval(self) -> RiskBinSpec:
        if not self.lower < self.upper:
            raise ValueError("risk_bin_interval_empty")
        return self

    def contains(self, score: float) -> bool:
        if self.upper_inclusive:
            return self.lower <= score <= self.upper
        return self.lower <= score < self.upper


class FixedRiskBinFamily(ContractModel):
    """Self-hashed score partition frozen independently of calibration outcomes."""

    family_version: Literal["fixed-item-risk-bins-v1"] = "fixed-item-risk-bins-v1"
    score_name: Annotated[str, Field(min_length=1)]
    score_model_sha256: str
    definition_source: Literal["prespecified", "development_only"]
    definition_artifact_sha256: str
    bins: list[RiskBinSpec]
    family_sha256: str

    @field_validator(
        "score_model_sha256", "definition_artifact_sha256", "family_sha256"
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "risk_bin_family")

    @model_validator(mode="after")
    def validate_family_integrity(self) -> FixedRiskBinFamily:
        if not self.bins:
            raise ValueError("risk_bin_family_empty")
        expected_ids = [f"risk-bin-{index:03d}" for index in range(len(self.bins))]
        if [item.bin_id for item in self.bins] != expected_ids:
            raise ValueError("risk_bin_ids_or_order_invalid")
        if self.bins[0].lower != 0.0 or self.bins[-1].upper != 1.0:
            raise ValueError("risk_bins_must_cover_unit_interval")
        for left, right in pairwise(self.bins):
            if left.upper != right.lower:
                raise ValueError("risk_bins_must_be_contiguous")
        if any(item.upper_inclusive for item in self.bins[:-1]):
            raise ValueError("only_final_risk_bin_may_include_upper_edge")
        if not self.bins[-1].upper_inclusive:
            raise ValueError("final_risk_bin_must_include_one")
        payload = self.model_dump(mode="json", exclude={"family_sha256"})
        if hash_canonical(payload) != self.family_sha256:
            raise ValueError("risk_bin_family_hash_mismatch")
        return self

    def bin_for_score(self, score: float) -> RiskBinSpec:
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ItemRiskCalibrationError("item_risk_score_invalid")
        matches = [item for item in self.bins if item.contains(score)]
        if len(matches) != 1:
            raise ItemRiskCalibrationError("item_risk_score_bin_not_unique")
        return matches[0]


def make_fixed_risk_bin_family(
    *,
    edges: Sequence[float],
    score_name: str,
    score_model_sha256: str,
    definition_source: Literal["prespecified", "development_only"],
    definition_artifact_sha256: str,
) -> FixedRiskBinFamily:
    """Seal explicit bin edges; no calibration labels are accepted by this API."""

    numeric_edges = [float(edge) for edge in edges]
    if (
        len(numeric_edges) < 2
        or any(not math.isfinite(edge) for edge in numeric_edges)
        or numeric_edges != sorted(set(numeric_edges))
        or numeric_edges[0] != 0.0
        or numeric_edges[-1] != 1.0
    ):
        raise ItemRiskCalibrationError("fixed_risk_bin_edges_invalid")
    bins = [
        RiskBinSpec(
            bin_id=f"risk-bin-{index:03d}",
            lower=lower,
            upper=upper,
            upper_inclusive=index == len(numeric_edges) - 2,
        )
        for index, (lower, upper) in enumerate(pairwise(numeric_edges))
    ]
    payload = {
        "family_version": "fixed-item-risk-bins-v1",
        "score_name": score_name,
        "score_model_sha256": score_model_sha256,
        "definition_source": definition_source,
        "definition_artifact_sha256": definition_artifact_sha256,
        "bins": bins,
    }
    return FixedRiskBinFamily.model_validate(
        {**payload, "family_sha256": hash_canonical(payload)}
    )


class ItemRiskCalibrationUnit(ContractModel):
    """One adjudicated item sampled from a unique question and unique paper."""

    unit_version: Literal["item-risk-unit-v1"] = "item-risk-unit-v1"
    split: CalibrationSplit
    item_id: Annotated[str, Field(min_length=1)]
    question_id: Annotated[str, Field(min_length=1)]
    paper_id: Annotated[str, Field(min_length=1)]
    population_id: Annotated[str, Field(min_length=1)]
    domain: Annotated[str, Field(min_length=1)]
    pipeline_sha256: str
    score_model_sha256: str
    score_input_sha256: str
    risk_score: Annotated[float, Field(ge=0, le=1)]
    observed_error: bool
    label_source: AdjudicatedLabelSource
    adjudication_protocol_sha256: str
    adjudication_artifact_sha256: str
    unit_sha256: str

    @field_validator(
        "pipeline_sha256",
        "score_model_sha256",
        "score_input_sha256",
        "adjudication_protocol_sha256",
        "adjudication_artifact_sha256",
        "unit_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "item_risk_unit")

    @field_validator("observed_error", mode="before")
    @classmethod
    def validate_strict_boolean(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise ValueError("item_risk_observed_error_must_be_boolean")
        return value

    @field_validator("risk_score")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("item_risk_score_nonfinite")
        return value

    @model_validator(mode="after")
    def validate_unit_integrity(self) -> ItemRiskCalibrationUnit:
        payload = self.model_dump(mode="json", exclude={"unit_sha256"})
        if hash_canonical(payload) != self.unit_sha256:
            raise ValueError("item_risk_unit_hash_mismatch")
        return self


def seal_item_risk_calibration_unit(
    *,
    split: CalibrationSplit,
    item_id: str,
    question_id: str,
    paper_id: str,
    population_id: str,
    domain: str,
    pipeline_sha256: str,
    score_model_sha256: str,
    score_input_sha256: str,
    risk_score: float,
    observed_error: bool,
    label_source: AdjudicatedLabelSource,
    adjudication_protocol_sha256: str,
    adjudication_artifact_sha256: str,
) -> ItemRiskCalibrationUnit:
    """Create a hash-bound adjudicated calibration row."""

    payload = {
        "unit_version": "item-risk-unit-v1",
        "split": split,
        "item_id": item_id,
        "question_id": question_id,
        "paper_id": paper_id,
        "population_id": population_id,
        "domain": domain,
        "pipeline_sha256": pipeline_sha256,
        "score_model_sha256": score_model_sha256,
        "score_input_sha256": score_input_sha256,
        "risk_score": risk_score,
        "observed_error": observed_error,
        "label_source": label_source,
        "adjudication_protocol_sha256": adjudication_protocol_sha256,
        "adjudication_artifact_sha256": adjudication_artifact_sha256,
    }
    return ItemRiskCalibrationUnit.model_validate(
        {**payload, "unit_sha256": hash_canonical(payload)}
    )


class ItemRiskSplitIdentity(ContractModel):
    split: CalibrationSplit
    item_ids: list[str]
    question_ids: list[str]
    paper_ids: list[str]

    @field_validator("item_ids", "question_ids", "paper_ids")
    @classmethod
    def validate_ids(cls, value: list[str], info: Any) -> list[str]:
        return _validate_sorted_unique_nonempty(value, info.field_name)


class DomainRiskBinCalibration(ContractModel):
    """One simultaneous domain/bin group-average error-rate UCL."""

    domain: Annotated[str, Field(min_length=1)]
    bin_id: Annotated[str, Field(pattern=r"^risk-bin-[0-9]{3}$")]
    estimand: Literal[
        "group_average_item_error_rate_within_domain_score_bin"
    ] = _CELL_RATE_ESTIMAND
    cell_calibration_units: Annotated[int, Field(ge=0)]
    cell_observed_errors: Annotated[int, Field(ge=0)]
    empirical_cell_error_rate: Annotated[float, Field(ge=0, le=1)] | None
    familywise_delta: Annotated[float, Field(gt=0, lt=1)]
    family_cell_count: Annotated[int, Field(gt=0)]
    cellwise_delta: Annotated[float, Field(gt=0, lt=1)]
    upper_cell_error_rate: Annotated[float, Field(ge=0, le=1)] | None
    status: Literal["calibrated", "empty"]

    @model_validator(mode="after")
    def validate_bound(self) -> DomainRiskBinCalibration:
        expected_cellwise_delta = self.familywise_delta / self.family_cell_count
        if not math.isclose(
            self.cellwise_delta,
            expected_cellwise_delta,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("risk_bin_cellwise_delta_mismatch")
        if self.cell_observed_errors > self.cell_calibration_units:
            raise ValueError("risk_bin_errors_exceed_units")
        if self.cell_calibration_units == 0:
            if (
                self.status != "empty"
                or self.empirical_cell_error_rate is not None
                or self.upper_cell_error_rate is not None
            ):
                raise ValueError("empty_risk_bin_bound_mismatch")
            return self
        if self.status != "calibrated":
            raise ValueError("nonempty_risk_bin_must_be_calibrated")
        expected_empirical = self.cell_observed_errors / self.cell_calibration_units
        if self.empirical_cell_error_rate != expected_empirical:
            raise ValueError("risk_bin_empirical_rate_mismatch")
        expected_upper = _one_sided_clopper_pearson_upper(
            self.cell_observed_errors,
            self.cell_calibration_units,
            delta=self.cellwise_delta,
        )
        if self.upper_cell_error_rate is None or not math.isclose(
            self.upper_cell_error_rate,
            expected_upper,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("risk_bin_upper_bound_mismatch")
        return self

    @property
    def calibration_units(self) -> int:
        """Deprecated alias for diagnostic readers."""

        return self.cell_calibration_units

    @property
    def observed_errors(self) -> int:
        """Deprecated alias for diagnostic readers."""

        return self.cell_observed_errors

    @property
    def empirical_error_rate(self) -> float | None:
        """Deprecated alias for the group-average empirical cell rate."""

        return self.empirical_cell_error_rate

    @property
    def simultaneous_delta(self) -> float:
        """Deprecated alias; this is the per-cell Bonferroni delta."""

        return self.cellwise_delta

    @property
    def upper_error_probability(self) -> float | None:
        """Deprecated alias; this is not an individual error probability."""

        return self.upper_cell_error_rate


class ItemRiskCalibrationBundle(ContractModel):
    """Self-contained proof object for simultaneous group-average cell-rate UCLs."""

    bundle_version: Literal["item-risk-calibration-v2"] = "item-risk-calibration-v2"
    freeze_state: Literal["fixed_before_deployment_scoring"] = (
        "fixed_before_deployment_scoring"
    )
    population_id: Annotated[str, Field(min_length=1)]
    calibration_domains: list[str]
    supported_deployment_domains: list[str]
    pipeline_sha256: str
    pipeline_verification: PipelineFingerprintVerification
    pipeline_verification_sha256: str
    score_model_sha256: str
    score_semantics: Literal[
        "externally_supplied_scheduling_score_not_recomputed"
    ] = _SCORE_SEMANTICS
    cell_rate_estimand: Literal[
        "group_average_item_error_rate_within_domain_score_bin"
    ] = _CELL_RATE_ESTIMAND
    release_probability_authority: Literal[False] = False
    bin_family: FixedRiskBinFamily
    bin_family_sha256: str
    sampling_protocol_sha256: str
    adjudication_protocol_sha256: str
    error_event_definition: Annotated[str, Field(min_length=1)]
    shift_detector_id: Annotated[str, Field(min_length=1)]
    shift_detector_sha256: str
    familywise_delta: Annotated[float, Field(gt=0, lt=1)]
    correction: Literal["bonferroni-clopper-pearson"] = (
        "bonferroni-clopper-pearson"
    )
    label_sources: list[AdjudicatedLabelSource]
    development: ItemRiskSplitIdentity
    calibration: ItemRiskSplitIdentity
    units: list[ItemRiskCalibrationUnit]
    calibration_input_sha256: str
    bounds: list[DomainRiskBinCalibration]
    assumptions: list[str]
    bundle_sha256: str

    @field_validator(
        "pipeline_sha256",
        "pipeline_verification_sha256",
        "score_model_sha256",
        "bin_family_sha256",
        "sampling_protocol_sha256",
        "adjudication_protocol_sha256",
        "shift_detector_sha256",
        "calibration_input_sha256",
        "bundle_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "item_risk_bundle")

    @field_validator("calibration_domains", "supported_deployment_domains")
    @classmethod
    def validate_domains(cls, value: list[str], info: Any) -> list[str]:
        return _validate_sorted_unique_nonempty(value, info.field_name)

    @field_validator("label_sources")
    @classmethod
    def validate_label_sources(
        cls, value: list[AdjudicatedLabelSource]
    ) -> list[AdjudicatedLabelSource]:
        if not value or value != sorted(set(value)):
            raise ValueError("item_risk_label_sources_must_be_nonempty_sorted_unique")
        return value

    @model_validator(mode="after")
    def validate_bundle_integrity(self) -> ItemRiskCalibrationBundle:
        if self.development.split != "development" or self.calibration.split != "calibration":
            raise ValueError("item_risk_split_identity_mismatch")
        if not set(self.supported_deployment_domains) <= set(self.calibration_domains):
            raise ValueError("supported_domain_absent_from_calibration")
        if self.assumptions != _BOUND_ASSUMPTIONS:
            raise ValueError("item_risk_bound_assumptions_mismatch")
        if (
            self.pipeline_verification.status != "matched"
            or self.pipeline_verification.computed is None
            or self.pipeline_verification.computed_pipeline_sha256 != self.pipeline_sha256
            or self.pipeline_verification.expected_pipeline_sha256 != self.pipeline_sha256
            or self.pipeline_verification.verification_sha256
            != self.pipeline_verification_sha256
        ):
            raise ValueError("item_risk_pipeline_verification_mismatch")
        if self.bin_family.family_sha256 != self.bin_family_sha256:
            raise ValueError("item_risk_bin_family_hash_mismatch")
        if self.bin_family.score_model_sha256 != self.score_model_sha256:
            raise ValueError("item_risk_bin_score_model_mismatch")
        if not self.units:
            raise ValueError("item_risk_calibration_units_empty")
        expected_order = sorted(
            self.units, key=lambda row: (row.split, row.question_id, row.item_id)
        )
        if self.units != expected_order:
            raise ValueError("item_risk_units_must_be_canonically_sorted")
        _validate_unit_collection(self.units)
        first = self.units[0]
        for unit in self.units:
            if (
                unit.population_id != self.population_id
                or unit.pipeline_sha256 != self.pipeline_sha256
                or unit.score_model_sha256 != self.score_model_sha256
                or unit.adjudication_protocol_sha256 != self.adjudication_protocol_sha256
            ):
                raise ValueError("item_risk_unit_bundle_lineage_mismatch")
        if first.population_id != self.population_id:
            raise ValueError("item_risk_bundle_population_mismatch")
        development = [row for row in self.units if row.split == "development"]
        calibration = [row for row in self.units if row.split == "calibration"]
        if not development or not calibration:
            raise ValueError("item_risk_requires_development_and_calibration_units")
        if self.development != _split_identity("development", development):
            raise ValueError("item_risk_development_identity_mismatch")
        if self.calibration != _split_identity("calibration", calibration):
            raise ValueError("item_risk_calibration_identity_mismatch")
        observed_domains = sorted({row.domain for row in calibration})
        if self.calibration_domains != observed_domains:
            raise ValueError("item_risk_calibration_domains_mismatch")
        if self.label_sources != sorted({row.label_source for row in self.units}):
            raise ValueError("item_risk_label_sources_mismatch")
        if hash_canonical(self.units) != self.calibration_input_sha256:
            raise ValueError("item_risk_calibration_input_hash_mismatch")
        expected_bounds = _calculate_bounds(
            calibration,
            domains=self.calibration_domains,
            bin_family=self.bin_family,
            familywise_delta=self.familywise_delta,
        )
        if self.bounds != expected_bounds:
            raise ValueError("item_risk_calibration_bounds_mismatch")
        payload = self.model_dump(mode="json", exclude={"bundle_sha256"})
        if hash_canonical(payload) != self.bundle_sha256:
            raise ValueError("item_risk_bundle_hash_mismatch")
        return self


def _validate_unit_collection(units: Sequence[ItemRiskCalibrationUnit]) -> None:
    unit_ids = [unit.item_id for unit in units]
    question_ids = [unit.question_id for unit in units]
    paper_ids = [unit.paper_id for unit in units]
    if len(unit_ids) != len(set(unit_ids)):
        raise ItemRiskCalibrationError("item_risk_item_ids_must_be_unique")
    if len(question_ids) != len(set(question_ids)):
        raise ItemRiskCalibrationError(
            "item_risk_units_must_be_question_disjoint_one_item_per_question"
        )
    if len(paper_ids) != len(set(paper_ids)):
        raise ItemRiskCalibrationError(
            "item_risk_units_must_be_paper_disjoint_one_item_per_paper"
        )


def _split_identity(
    split: CalibrationSplit, units: Sequence[ItemRiskCalibrationUnit]
) -> ItemRiskSplitIdentity:
    return ItemRiskSplitIdentity(
        split=split,
        item_ids=sorted(row.item_id for row in units),
        question_ids=sorted(row.question_id for row in units),
        paper_ids=sorted(row.paper_id for row in units),
    )


def _calculate_bounds(
    calibration: Sequence[ItemRiskCalibrationUnit],
    *,
    domains: Sequence[str],
    bin_family: FixedRiskBinFamily,
    familywise_delta: float,
) -> list[DomainRiskBinCalibration]:
    cell_count = len(domains) * len(bin_family.bins)
    cellwise_delta = familywise_delta / cell_count
    bounds: list[DomainRiskBinCalibration] = []
    for domain in domains:
        for risk_bin in bin_family.bins:
            rows = [
                row
                for row in calibration
                if row.domain == domain and risk_bin.contains(row.risk_score)
            ]
            total = len(rows)
            errors = sum(row.observed_error for row in rows)
            bounds.append(
                DomainRiskBinCalibration(
                    domain=domain,
                    bin_id=risk_bin.bin_id,
                    estimand=_CELL_RATE_ESTIMAND,
                    cell_calibration_units=total,
                    cell_observed_errors=errors,
                    empirical_cell_error_rate=None if not total else errors / total,
                    familywise_delta=familywise_delta,
                    family_cell_count=cell_count,
                    cellwise_delta=cellwise_delta,
                    upper_cell_error_rate=(
                        None
                        if not total
                        else _one_sided_clopper_pearson_upper(
                            errors, total, delta=cellwise_delta
                        )
                    ),
                    status="empty" if not total else "calibrated",
                )
            )
    return bounds


def calibrate_item_risk_bounds(
    units: Sequence[ItemRiskCalibrationUnit],
    *,
    pipeline_verification: PipelineFingerprintVerification,
    bin_family: FixedRiskBinFamily,
    familywise_delta: float,
    sampling_protocol_sha256: str,
    error_event_definition: str,
    shift_detector_id: str,
    shift_detector_sha256: str,
    supported_deployment_domains: Sequence[str] | None = None,
) -> ItemRiskCalibrationBundle:
    """Freeze conservative domain/bin bounds from real adjudicated units.

    ``units`` cannot contain simulation labels, direct manifest probabilities, or
    repeated question/paper identities.  The bin family is already self-hashed and
    fixed; calibration outcomes cannot alter its edges.
    """

    if not math.isfinite(familywise_delta) or not 0 < familywise_delta < 1:
        raise ItemRiskCalibrationError("item_risk_familywise_delta_invalid")
    if not units:
        raise ItemRiskCalibrationError("item_risk_calibration_units_empty")
    try:
        verification = validate_pipeline_verification_integrity(pipeline_verification)
        family = FixedRiskBinFamily.model_validate(bin_family.model_dump(mode="json"))
        rows = sorted(
            [
                ItemRiskCalibrationUnit.model_validate(unit.model_dump(mode="json"))
                for unit in units
            ],
            key=lambda row: (row.split, row.question_id, row.item_id),
        )
    except (AttributeError, ValueError) as exc:
        raise ItemRiskCalibrationError("item_risk_calibration_input_integrity_changed") from exc
    if verification.status != "matched" or verification.computed is None:
        raise ItemRiskCalibrationError("item_risk_requires_matched_pipeline_verification")
    _validate_unit_collection(rows)
    development = [row for row in rows if row.split == "development"]
    calibration = [row for row in rows if row.split == "calibration"]
    if not development or not calibration:
        raise ItemRiskCalibrationError(
            "item_risk_requires_development_and_calibration_units"
        )
    scalar_fields = {
        "population_id": {row.population_id for row in rows},
        "pipeline_sha256": {row.pipeline_sha256 for row in rows},
        "score_model_sha256": {row.score_model_sha256 for row in rows},
        "adjudication_protocol_sha256": {
            row.adjudication_protocol_sha256 for row in rows
        },
    }
    for name, values in scalar_fields.items():
        if len(values) != 1:
            raise ItemRiskCalibrationError(f"item_risk_mixed_{name}")
    score_model_sha256 = next(iter(scalar_fields["score_model_sha256"]))
    pipeline_sha256 = next(iter(scalar_fields["pipeline_sha256"]))
    if (
        verification.expected_pipeline_sha256 != pipeline_sha256
        or verification.computed_pipeline_sha256 != pipeline_sha256
    ):
        raise ItemRiskCalibrationError("item_risk_pipeline_verification_mismatch")
    if family.score_model_sha256 != score_model_sha256:
        raise ItemRiskCalibrationError("item_risk_bin_score_model_mismatch")
    calibration_domains = sorted({row.domain for row in calibration})
    supported = (
        calibration_domains
        if supported_deployment_domains is None
        else sorted(set(supported_deployment_domains))
    )
    if not supported or not set(supported) <= set(calibration_domains):
        raise ItemRiskCalibrationError("item_risk_supported_domains_invalid")
    bounds = _calculate_bounds(
        calibration,
        domains=calibration_domains,
        bin_family=family,
        familywise_delta=familywise_delta,
    )
    payload: dict[str, Any] = {
        "bundle_version": "item-risk-calibration-v2",
        "freeze_state": "fixed_before_deployment_scoring",
        "population_id": next(iter(scalar_fields["population_id"])),
        "calibration_domains": calibration_domains,
        "supported_deployment_domains": supported,
        "pipeline_sha256": pipeline_sha256,
        "pipeline_verification": verification,
        "pipeline_verification_sha256": verification.verification_sha256,
        "score_model_sha256": score_model_sha256,
        "score_semantics": _SCORE_SEMANTICS,
        "cell_rate_estimand": _CELL_RATE_ESTIMAND,
        "release_probability_authority": False,
        "bin_family": family,
        "bin_family_sha256": family.family_sha256,
        "sampling_protocol_sha256": sampling_protocol_sha256,
        "adjudication_protocol_sha256": next(
            iter(scalar_fields["adjudication_protocol_sha256"])
        ),
        "error_event_definition": error_event_definition,
        "shift_detector_id": shift_detector_id,
        "shift_detector_sha256": shift_detector_sha256,
        "familywise_delta": familywise_delta,
        "correction": "bonferroni-clopper-pearson",
        "label_sources": sorted({row.label_source for row in rows}),
        "development": _split_identity("development", development),
        "calibration": _split_identity("calibration", calibration),
        "units": rows,
        "calibration_input_sha256": hash_canonical(rows),
        "bounds": bounds,
        "assumptions": _BOUND_ASSUMPTIONS,
    }
    try:
        return ItemRiskCalibrationBundle.model_validate(
            {**payload, "bundle_sha256": hash_canonical(payload)}
        )
    except ValueError as exc:
        raise ItemRiskCalibrationError("item_risk_bundle_construction_failed") from exc


def validate_item_risk_calibration_bundle_integrity(
    bundle: ItemRiskCalibrationBundle,
) -> ItemRiskCalibrationBundle:
    """Revalidate nested units, bounds, and every self-hash from a JSON snapshot."""

    if not isinstance(bundle, ItemRiskCalibrationBundle):
        raise ItemRiskCalibrationError("item_risk_bundle_contract_invalid")
    try:
        return ItemRiskCalibrationBundle.model_validate(bundle.model_dump(mode="json"))
    except ValueError as exc:
        raise ItemRiskCalibrationError("item_risk_bundle_integrity_changed") from exc


class ShiftAssessment(ContractModel):
    """A bundle-bound prospective shift decision produced outside the claim manifest."""

    assessment_version: Literal["item-risk-shift-v1"] = "item-risk-shift-v1"
    calibration_bundle_sha256: str
    detector_id: Annotated[str, Field(min_length=1)]
    detector_sha256: str
    candidate_population_id: Annotated[str, Field(min_length=1)]
    candidate_domain: Annotated[str, Field(min_length=1)]
    status: Literal["no_shift_detected", "shift_detected", "not_assessed"]
    assessment_input_sha256: str
    assessment_artifact_sha256: str | None
    assessment_sha256: str

    @field_validator(
        "calibration_bundle_sha256",
        "detector_sha256",
        "assessment_input_sha256",
        "assessment_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "item_risk_shift_assessment")

    @field_validator("assessment_artifact_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return _validate_optional_sha256(value, "item_risk_shift_artifact")

    @model_validator(mode="after")
    def validate_assessment_integrity(self) -> ShiftAssessment:
        if (self.status == "not_assessed") != (self.assessment_artifact_sha256 is None):
            raise ValueError("item_risk_shift_status_artifact_mismatch")
        payload = self.model_dump(mode="json", exclude={"assessment_sha256"})
        if hash_canonical(payload) != self.assessment_sha256:
            raise ValueError("item_risk_shift_assessment_hash_mismatch")
        return self


def seal_shift_assessment(
    *,
    bundle: ItemRiskCalibrationBundle,
    candidate_population_id: str,
    candidate_domain: str,
    status: Literal["no_shift_detected", "shift_detected", "not_assessed"],
    assessment_input_sha256: str,
    assessment_artifact_sha256: str | None,
) -> ShiftAssessment:
    """Bind an external shift detector result to one frozen calibration bundle."""

    bundle = validate_item_risk_calibration_bundle_integrity(bundle)
    payload = {
        "assessment_version": "item-risk-shift-v1",
        "calibration_bundle_sha256": bundle.bundle_sha256,
        "detector_id": bundle.shift_detector_id,
        "detector_sha256": bundle.shift_detector_sha256,
        "candidate_population_id": candidate_population_id,
        "candidate_domain": candidate_domain,
        "status": status,
        "assessment_input_sha256": assessment_input_sha256,
        "assessment_artifact_sha256": assessment_artifact_sha256,
    }
    return ShiftAssessment.model_validate(
        {**payload, "assessment_sha256": hash_canonical(payload)}
    )


class ItemRiskCandidate(ContractModel):
    """Prospective score input; deliberately has no probability or outcome field."""

    candidate_version: Literal["item-risk-candidate-v1"] = "item-risk-candidate-v1"
    item_id: Annotated[str, Field(min_length=1)]
    question_id: Annotated[str, Field(min_length=1)]
    paper_id: Annotated[str, Field(min_length=1)]
    population_id: Annotated[str, Field(min_length=1)]
    domain: Annotated[str, Field(min_length=1)]
    pipeline_sha256: str
    score_model_sha256: str
    score_input_sha256: str
    risk_score: Annotated[float, Field(ge=0, le=1)]
    shift_assessment: ShiftAssessment | None = None
    candidate_sha256: str

    @field_validator(
        "pipeline_sha256",
        "score_model_sha256",
        "score_input_sha256",
        "candidate_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "item_risk_candidate")

    @field_validator("risk_score")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("item_risk_candidate_score_nonfinite")
        return value

    @model_validator(mode="after")
    def validate_candidate_integrity(self) -> ItemRiskCandidate:
        if self.shift_assessment is not None and (
            self.shift_assessment.candidate_population_id != self.population_id
            or self.shift_assessment.candidate_domain != self.domain
        ):
            raise ValueError("item_risk_candidate_shift_scope_mismatch")
        payload = self.model_dump(mode="json", exclude={"candidate_sha256"})
        if hash_canonical(payload) != self.candidate_sha256:
            raise ValueError("item_risk_candidate_hash_mismatch")
        return self


def seal_item_risk_candidate(
    *,
    item_id: str,
    question_id: str,
    paper_id: str,
    population_id: str,
    domain: str,
    pipeline_sha256: str,
    score_model_sha256: str,
    score_input_sha256: str,
    risk_score: float,
    shift_assessment: ShiftAssessment | None,
) -> ItemRiskCandidate:
    """Seal prospective score inputs without accepting a manifest probability."""

    payload = {
        "candidate_version": "item-risk-candidate-v1",
        "item_id": item_id,
        "question_id": question_id,
        "paper_id": paper_id,
        "population_id": population_id,
        "domain": domain,
        "pipeline_sha256": pipeline_sha256,
        "score_model_sha256": score_model_sha256,
        "score_input_sha256": score_input_sha256,
        "risk_score": risk_score,
        "shift_assessment": shift_assessment,
    }
    return ItemRiskCandidate.model_validate(
        {**payload, "candidate_sha256": hash_canonical(payload)}
    )


class RiskBound(ContractModel):
    """Self-hashed scheduling-only group-average cell-rate UCL.

    Even a successful result is not an individual item's probability and sets
    ``usable_for_release`` to false.  The current bundle binds an opaque score-model
    hash but cannot execute that model against current evidence, so the supplied raw
    score is fit only for scheduling and conservative audit blocking.
    """

    bound_version: Literal["item-risk-bound-v2"] = "item-risk-bound-v2"
    item_id: Annotated[str, Field(min_length=1)]
    candidate_sha256: str
    calibration_bundle_sha256: str
    pipeline_sha256: str
    pipeline_verification_sha256: str
    population_id: Annotated[str, Field(min_length=1)]
    domain: Annotated[str, Field(min_length=1)]
    raw_risk_score: Annotated[float, Field(ge=0, le=1)]
    bin_id: Annotated[str, Field(pattern=r"^risk-bin-[0-9]{3}$")]
    score_semantics: Literal[
        "externally_supplied_scheduling_score_not_recomputed"
    ] = _SCORE_SEMANTICS
    estimand: Literal[
        "group_average_item_error_rate_within_domain_score_bin"
    ] = _CELL_RATE_ESTIMAND
    status: RiskBoundStatus
    usable_for_scheduling: bool
    usable_for_release: Literal[False] = False
    rate_basis: Literal["calibrated_cell_rate_ucl"] | None
    rate_source: str | None
    upper_cell_error_rate: Annotated[float, Field(ge=0, le=1)] | None
    cell_calibration_units: Annotated[int, Field(ge=0)] | None
    cell_observed_errors: Annotated[int, Field(ge=0)] | None
    familywise_delta: Annotated[float, Field(gt=0, lt=1)] | None
    family_cell_count: Annotated[int, Field(gt=0)] | None
    cellwise_delta: Annotated[float, Field(gt=0, lt=1)] | None
    assumptions: list[str]
    risk_bound_sha256: str

    @field_validator(
        "candidate_sha256",
        "calibration_bundle_sha256",
        "pipeline_sha256",
        "pipeline_verification_sha256",
        "risk_bound_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _validate_sha256(value, "item_risk_bound")

    @model_validator(mode="after")
    def validate_bound_integrity(self) -> RiskBound:
        if self.assumptions != _BOUND_ASSUMPTIONS:
            raise ValueError("item_risk_bound_assumptions_mismatch")
        numeric_proof = (
            self.upper_cell_error_rate,
            self.cell_calibration_units,
            self.cell_observed_errors,
            self.familywise_delta,
            self.family_cell_count,
            self.cellwise_delta,
        )
        text_proof = (self.rate_basis, self.rate_source)
        if self.status == "cell_rate_ucl_available":
            if not self.usable_for_scheduling or any(value is None for value in numeric_proof):
                raise ValueError("item_cell_rate_ucl_incomplete")
            if any(value is None for value in text_proof):
                raise ValueError("item_cell_rate_ucl_source_incomplete")
            total = int(self.cell_calibration_units or 0)
            errors = int(self.cell_observed_errors or 0)
            familywise_delta = float(self.familywise_delta or 0.0)
            family_cell_count = int(self.family_cell_count or 0)
            delta = float(self.cellwise_delta or 0.0)
            if family_cell_count <= 0 or not math.isclose(
                delta,
                familywise_delta / family_cell_count,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError("item_cell_rate_ucl_delta_mismatch")
            expected = _one_sided_clopper_pearson_upper(errors, total, delta=delta)
            if self.upper_cell_error_rate is None or not math.isclose(
                self.upper_cell_error_rate, expected, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError("item_cell_rate_ucl_value_mismatch")
        elif (
            self.usable_for_scheduling
            or any(value is not None for value in numeric_proof)
            or any(value is not None for value in text_proof)
        ):
            raise ValueError("unavailable_item_cell_rate_ucl_must_not_carry_rate")
        payload = self.model_dump(mode="json", exclude={"risk_bound_sha256"})
        if hash_canonical(payload) != self.risk_bound_sha256:
            raise ValueError("item_risk_bound_hash_mismatch")
        return self

    @property
    def probability_basis(self) -> str | None:
        """Deprecated diagnostic alias; the value is not a probability basis."""

        return self.rate_basis

    @property
    def probability_source(self) -> str | None:
        """Deprecated diagnostic alias for the cell-rate source."""

        return self.rate_source

    @property
    def upper_error_probability(self) -> float | None:
        """Deprecated diagnostic alias; this is a group-average cell-rate UCL."""

        return self.upper_cell_error_rate

    @property
    def calibration_units(self) -> int | None:
        return self.cell_calibration_units

    @property
    def observed_errors(self) -> int | None:
        return self.cell_observed_errors

    @property
    def simultaneous_delta(self) -> float | None:
        return self.cellwise_delta


def _risk_bound(
    *,
    candidate: ItemRiskCandidate,
    bundle: ItemRiskCalibrationBundle,
    risk_bin: RiskBinSpec,
    status: RiskBoundStatus,
    cell: DomainRiskBinCalibration | None = None,
) -> RiskBound:
    available = status == "cell_rate_ucl_available"
    family_cell_count = len(bundle.calibration_domains) * len(bundle.bin_family.bins)
    payload = {
        "bound_version": "item-risk-bound-v2",
        "item_id": candidate.item_id,
        "candidate_sha256": candidate.candidate_sha256,
        "calibration_bundle_sha256": bundle.bundle_sha256,
        "pipeline_sha256": candidate.pipeline_sha256,
        "pipeline_verification_sha256": bundle.pipeline_verification_sha256,
        "population_id": candidate.population_id,
        "domain": candidate.domain,
        "raw_risk_score": candidate.risk_score,
        "bin_id": risk_bin.bin_id,
        "score_semantics": _SCORE_SEMANTICS,
        "estimand": _CELL_RATE_ESTIMAND,
        "status": status,
        "usable_for_scheduling": available,
        "usable_for_release": False,
        "rate_basis": "calibrated_cell_rate_ucl" if available else None,
        "rate_source": (
            f"item-risk-calibration:{bundle.bundle_sha256}:{candidate.domain}:{risk_bin.bin_id}"
            if available
            else None
        ),
        "upper_cell_error_rate": (
            cell.upper_cell_error_rate if available and cell is not None else None
        ),
        "cell_calibration_units": (
            cell.cell_calibration_units if available and cell is not None else None
        ),
        "cell_observed_errors": (
            cell.cell_observed_errors if available and cell is not None else None
        ),
        "familywise_delta": bundle.familywise_delta if available else None,
        "family_cell_count": family_cell_count if available else None,
        "cellwise_delta": (
            cell.cellwise_delta if available and cell is not None else None
        ),
        "assumptions": _BOUND_ASSUMPTIONS,
    }
    return RiskBound.model_validate(
        {**payload, "risk_bound_sha256": hash_canonical(payload)}
    )


def score_item_risk_bound(
    *,
    candidate: ItemRiskCandidate,
    bundle: ItemRiskCalibrationBundle,
    pipeline_verification: PipelineFingerprintVerification,
) -> RiskBound:
    """Return a group-average cell-rate UCL or a proof-carrying failure status.

    The raw score is only used to select a fixed bin.  This function has no input
    for a manifest-supplied error probability and never promotes the cell UCL to an
    individual or claim-release probability.
    """

    bundle = validate_item_risk_calibration_bundle_integrity(bundle)
    try:
        verification = validate_pipeline_verification_integrity(pipeline_verification)
        candidate = ItemRiskCandidate.model_validate(candidate.model_dump(mode="json"))
    except (AttributeError, ValueError) as exc:
        raise ItemRiskCalibrationError("item_risk_candidate_integrity_changed") from exc
    risk_bin = bundle.bin_family.bin_for_score(candidate.risk_score)
    if (
        verification.status != "matched"
        or verification.computed is None
        or verification.expected_pipeline_sha256 != bundle.pipeline_sha256
        or verification.computed_pipeline_sha256 != bundle.pipeline_sha256
        or verification.verification_sha256 != bundle.pipeline_verification_sha256
    ):
        return _risk_bound(
            candidate=candidate, bundle=bundle, risk_bin=risk_bin, status="pipeline_mismatch"
        )
    if candidate.pipeline_sha256 != bundle.pipeline_sha256:
        return _risk_bound(
            candidate=candidate, bundle=bundle, risk_bin=risk_bin, status="pipeline_mismatch"
        )
    if candidate.score_model_sha256 != bundle.score_model_sha256:
        return _risk_bound(
            candidate=candidate,
            bundle=bundle,
            risk_bin=risk_bin,
            status="score_model_mismatch",
        )
    frozen_questions = set(bundle.development.question_ids) | set(
        bundle.calibration.question_ids
    )
    if candidate.question_id in frozen_questions:
        return _risk_bound(
            candidate=candidate,
            bundle=bundle,
            risk_bin=risk_bin,
            status="calibration_question_overlap",
        )
    frozen_papers = set(bundle.development.paper_ids) | set(bundle.calibration.paper_ids)
    if candidate.paper_id in frozen_papers:
        return _risk_bound(
            candidate=candidate,
            bundle=bundle,
            risk_bin=risk_bin,
            status="calibration_paper_overlap",
        )
    if candidate.population_id != bundle.population_id:
        return _risk_bound(
            candidate=candidate,
            bundle=bundle,
            risk_bin=risk_bin,
            status="population_mismatch",
        )
    if candidate.domain not in bundle.supported_deployment_domains:
        return _risk_bound(
            candidate=candidate,
            bundle=bundle,
            risk_bin=risk_bin,
            status="domain_out_of_scope",
        )
    assessment = candidate.shift_assessment
    if assessment is None or assessment.status == "not_assessed":
        return _risk_bound(
            candidate=candidate,
            bundle=bundle,
            risk_bin=risk_bin,
            status="shift_not_assessed",
        )
    if (
        assessment.calibration_bundle_sha256 != bundle.bundle_sha256
        or assessment.detector_id != bundle.shift_detector_id
        or assessment.detector_sha256 != bundle.shift_detector_sha256
    ):
        return _risk_bound(
            candidate=candidate,
            bundle=bundle,
            risk_bin=risk_bin,
            status="shift_assessment_mismatch",
        )
    if assessment.status == "shift_detected":
        return _risk_bound(
            candidate=candidate, bundle=bundle, risk_bin=risk_bin, status="shift_detected"
        )
    cells = {
        (bound.domain, bound.bin_id): bound
        for bound in bundle.bounds
    }
    cell = cells[(candidate.domain, risk_bin.bin_id)]
    if cell.status == "empty":
        return _risk_bound(
            candidate=candidate,
            bundle=bundle,
            risk_bin=risk_bin,
            status="empty_calibration_bin",
        )
    return _risk_bound(
        candidate=candidate,
        bundle=bundle,
        risk_bin=risk_bin,
        status="cell_rate_ucl_available",
        cell=cell,
    )


def verified_audit_cell_rate_ucl_fields(
    *,
    bound: RiskBound,
    bundle: ItemRiskCalibrationBundle,
    pipeline_verification: PipelineFingerprintVerification,
) -> Mapping[str, float | str]:
    """Expose a validated scheduling-only group-average cell-rate UCL."""

    bundle = validate_item_risk_calibration_bundle_integrity(bundle)
    try:
        verification = validate_pipeline_verification_integrity(pipeline_verification)
        bound = RiskBound.model_validate(bound.model_dump(mode="json"))
    except (AttributeError, ValueError) as exc:
        raise ItemRiskCalibrationError("item_risk_bound_integrity_changed") from exc
    if (
        bound.status != "cell_rate_ucl_available"
        or not bound.usable_for_scheduling
        or bound.usable_for_release
        or bound.calibration_bundle_sha256 != bundle.bundle_sha256
        or bound.pipeline_sha256 != bundle.pipeline_sha256
        or bound.pipeline_verification_sha256 != bundle.pipeline_verification_sha256
        or verification.status != "matched"
        or verification.expected_pipeline_sha256 != bundle.pipeline_sha256
        or verification.computed_pipeline_sha256 != bundle.pipeline_sha256
        or verification.verification_sha256 != bundle.pipeline_verification_sha256
        or bound.upper_cell_error_rate is None
        or bound.rate_basis is None
        or bound.rate_source is None
    ):
        raise ItemRiskCalibrationError("item_risk_cell_rate_ucl_not_scheduling_eligible")
    return {
        "item_cell_rate_ucl": bound.upper_cell_error_rate,
        "rate_basis": bound.rate_basis,
        "rate_source": bound.rate_source,
        "estimand": bound.estimand,
        "risk_bound_sha256": bound.risk_bound_sha256,
        "calibration_bundle_sha256": bundle.bundle_sha256,
    }


def verified_audit_probability_fields(
    *,
    bound: RiskBound,
    bundle: ItemRiskCalibrationBundle,
    pipeline_verification: PipelineFingerprintVerification,
) -> Mapping[str, float | str]:
    """Fail closed: a cell-rate UCL is not a release probability.

    The legacy function name remains importable so old integrations fail with an
    explicit scientific-contract error instead of silently upgrading a scheduling
    statistic into release authority.
    """

    verified_audit_cell_rate_ucl_fields(
        bound=bound,
        bundle=bundle,
        pipeline_verification=pipeline_verification,
    )
    raise ItemRiskCalibrationError("item_risk_cell_rate_ucl_not_release_probability")


__all__ = [
    "DomainRiskBinCalibration",
    "FixedRiskBinFamily",
    "ItemRiskCalibrationBundle",
    "ItemRiskCalibrationError",
    "ItemRiskCalibrationUnit",
    "ItemRiskCandidate",
    "ItemRiskSplitIdentity",
    "RiskBinSpec",
    "RiskBound",
    "ShiftAssessment",
    "calibrate_item_risk_bounds",
    "make_fixed_risk_bin_family",
    "score_item_risk_bound",
    "seal_item_risk_calibration_unit",
    "seal_item_risk_candidate",
    "seal_shift_assessment",
    "validate_item_risk_calibration_bundle_integrity",
    "verified_audit_cell_rate_ucl_fields",
    "verified_audit_probability_fields",
]
