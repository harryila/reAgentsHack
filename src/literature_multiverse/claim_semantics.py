"""Typed, hash-bound semantics for condition-qualified scientific claims.

Version-one claim release intentionally supports only unqualified directional targets.
This module adds a separate version-two contract without changing that legacy surface.
Conditions are exact predicates over extracted moderator values: no fuzzy coercion is
performed, and missing or type-incompatible values remain explicit so callers can fail
closed instead of silently changing the target population.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from literature_multiverse.effects import HarmonizedMeasure
from literature_multiverse.evidence_graph import ArmRole
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import SHA256_RE, ContractModel

type ConditionScalar = StrictBool | StrictInt | StrictFloat | StrictStr
type NumericConditionScalar = StrictInt | StrictFloat


class ClaimSemanticsContractError(ValueError):
    """A qualified claim cannot be interpreted without changing its meaning."""


class ClaimDirection(StrEnum):
    """Oriented effects supported by the qualified-claim contract."""

    INCREASE = "increase"
    DECREASE = "decrease"


class ConditionOperator(StrEnum):
    """Closed predicate language supported by the evidence-graph filter."""

    EQUALS = "equals"
    IN = "in"
    BETWEEN = "between"


class ConditionMatchStatus(StrEnum):
    MATCHED = "matched"
    NOT_MATCHED = "not_matched"
    MISSING = "missing"
    TYPE_MISMATCH = "type_mismatch"


class ConditionSetStatus(StrEnum):
    MATCHED = "matched"
    NOT_MATCHED = "not_matched"
    MISSING = "missing"
    TYPE_MISMATCH = "type_mismatch"


class ClaimSpecificationStatus(StrEnum):
    """Whether the exact qualifier was frozen before or discovered from these data."""

    PRESPECIFIED = "prespecified"
    DISCOVERED_HYPOTHESIS = "discovered_hypothesis"


class QualifiedClaimVerdictState(StrEnum):
    """Evidence-only verdicts that cannot conflate discovery with confirmation."""

    PRESPECIFIED_SUPPORTED = "prespecified_supported"
    PRESPECIFIED_CONTRADICTED = "prespecified_contradicted"
    PRESPECIFIED_INCONCLUSIVE = "prespecified_inconclusive"
    PRESPECIFIED_NOT_EVALUABLE = "prespecified_not_evaluable"
    DISCOVERED_HYPOTHESIS_ONLY = "discovered_hypothesis_only"


def _normalize_scalar(value: ConditionScalar) -> ConditionScalar:
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError("condition_scalar_string_empty")
        return normalized
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("condition_scalar_nonfinite")
    return value


def _scalar_key(value: ConditionScalar) -> tuple[str, str]:
    if isinstance(value, bool):
        kind = "bool"
    elif isinstance(value, int):
        kind = "int"
    elif isinstance(value, float):
        kind = "float"
    else:
        kind = "str"
    return kind, repr(value)


class ConditionPredicate(ContractModel):
    """One exact categorical or numeric moderator constraint.

    The operator-specific fields are deliberately disjoint. ``equals`` uses ``value``;
    ``in`` uses a sorted unique ``values`` list; and ``between`` uses finite numeric
    ``lower`` and ``upper`` bounds. Extracted values are never converted between strings,
    booleans, integers, and floats during matching.
    """

    moderator: Annotated[str, Field(min_length=1)]
    operator: ConditionOperator
    value: ConditionScalar | None = None
    values: list[ConditionScalar] = Field(default_factory=list)
    lower: NumericConditionScalar | None = None
    upper: NumericConditionScalar | None = None
    include_lower: bool = True
    include_upper: bool = True
    missing_policy: Literal["fail_closed"] = "fail_closed"

    @field_validator("moderator")
    @classmethod
    def normalize_moderator(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("condition_moderator_empty")
        return normalized

    @field_validator("value")
    @classmethod
    def normalize_value(cls, value: ConditionScalar | None) -> ConditionScalar | None:
        return _normalize_scalar(value) if value is not None else None

    @field_validator("values")
    @classmethod
    def normalize_values(cls, values: list[ConditionScalar]) -> list[ConditionScalar]:
        normalized = [_normalize_scalar(value) for value in values]
        keyed = {_scalar_key(value): value for value in normalized}
        if len(keyed) != len(normalized):
            raise ValueError("condition_values_duplicate")
        return [keyed[key] for key in sorted(keyed)]

    @field_validator("lower", "upper")
    @classmethod
    def finite_bounds(cls, value: NumericConditionScalar | None) -> NumericConditionScalar | None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("condition_bound_nonfinite")
        return value

    @model_validator(mode="after")
    def validate_operator_fields(self) -> ConditionPredicate:
        if self.operator is ConditionOperator.EQUALS:
            if self.value is None:
                raise ValueError("equals_condition_requires_value")
            if self.values or self.lower is not None or self.upper is not None:
                raise ValueError("equals_condition_has_inapplicable_fields")
            if not self.include_lower or not self.include_upper:
                raise ValueError("nonrange_condition_cannot_change_bound_inclusion")
        elif self.operator is ConditionOperator.IN:
            if not self.values:
                raise ValueError("in_condition_requires_values")
            if self.value is not None or self.lower is not None or self.upper is not None:
                raise ValueError("in_condition_has_inapplicable_fields")
            if not self.include_lower or not self.include_upper:
                raise ValueError("nonrange_condition_cannot_change_bound_inclusion")
        elif self.operator is ConditionOperator.BETWEEN:
            if self.lower is None or self.upper is None:
                raise ValueError("between_condition_requires_bounds")
            if self.value is not None or self.values:
                raise ValueError("between_condition_has_inapplicable_fields")
            if self.lower >= self.upper:
                raise ValueError("between_condition_bounds_not_ordered")
        return self

    def match(self, moderators: Mapping[str, object]) -> ConditionMatchStatus:
        """Evaluate without coercion; missing and incompatible types stay distinct."""

        if self.moderator not in moderators or moderators[self.moderator] is None:
            return ConditionMatchStatus.MISSING
        observed = moderators[self.moderator]
        if not isinstance(observed, (str, int, float, bool)):
            return ConditionMatchStatus.TYPE_MISMATCH
        try:
            normalized = _normalize_scalar(observed)
        except (TypeError, ValueError):
            return ConditionMatchStatus.TYPE_MISMATCH
        if self.operator is ConditionOperator.EQUALS:
            assert self.value is not None
            matched = _scalar_key(normalized) == _scalar_key(self.value)
        elif self.operator is ConditionOperator.IN:
            allowed = {_scalar_key(value) for value in self.values}
            matched = _scalar_key(normalized) in allowed
        else:
            if isinstance(normalized, bool) or not isinstance(normalized, (int, float)):
                return ConditionMatchStatus.TYPE_MISMATCH
            assert self.lower is not None and self.upper is not None
            above_lower = (
                normalized >= self.lower if self.include_lower else normalized > self.lower
            )
            below_upper = (
                normalized <= self.upper if self.include_upper else normalized < self.upper
            )
            matched = above_lower and below_upper
        return ConditionMatchStatus.MATCHED if matched else ConditionMatchStatus.NOT_MATCHED


class ConditionPredicateEvaluation(ContractModel):
    moderator: str
    predicate_sha256: str
    status: ConditionMatchStatus

    @field_validator("predicate_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("condition_predicate_sha256_invalid")
        return value


class ConditionSetEvaluation(ContractModel):
    """Complete conjunction result for one evidence item."""

    status: ConditionSetStatus
    predicates: list[ConditionPredicateEvaluation]
    missing_moderators: list[str]
    type_mismatch_moderators: list[str]
    nonmatching_moderators: list[str]

    @model_validator(mode="after")
    def validate_evaluation(self) -> ConditionSetEvaluation:
        for name in (
            "missing_moderators",
            "type_mismatch_moderators",
            "nonmatching_moderators",
        ):
            values = getattr(self, name)
            if values != sorted(set(values)):
                raise ValueError(f"condition_evaluation_values_not_sorted_unique:{name}")
        expected = ConditionSetStatus.MATCHED
        if self.type_mismatch_moderators:
            expected = ConditionSetStatus.TYPE_MISMATCH
        elif self.missing_moderators:
            expected = ConditionSetStatus.MISSING
        elif self.nonmatching_moderators:
            expected = ConditionSetStatus.NOT_MATCHED
        if self.status is not expected:
            raise ValueError("condition_set_status_mismatch")
        return self


def _condition_sort_key(predicate: ConditionPredicate) -> tuple[str, str, str]:
    return predicate.moderator, predicate.operator.value, hash_canonical(predicate)


def evaluate_condition_predicates(
    moderators: Mapping[str, object],
    predicates: Sequence[ConditionPredicate],
) -> ConditionSetEvaluation:
    """Evaluate an AND-conjunction, preserving every fail-closed reason."""

    if not predicates:
        raise ClaimSemanticsContractError("qualified_claim_requires_conditions")
    ordered = sorted(predicates, key=_condition_sort_key)
    if len({predicate.moderator for predicate in ordered}) != len(ordered):
        raise ClaimSemanticsContractError("condition_moderators_duplicate")
    rows = [
        ConditionPredicateEvaluation(
            moderator=predicate.moderator,
            predicate_sha256=hash_canonical(predicate),
            status=predicate.match(moderators),
        )
        for predicate in ordered
    ]
    missing = sorted(row.moderator for row in rows if row.status is ConditionMatchStatus.MISSING)
    mismatched = sorted(
        row.moderator for row in rows if row.status is ConditionMatchStatus.TYPE_MISMATCH
    )
    nonmatching = sorted(
        row.moderator for row in rows if row.status is ConditionMatchStatus.NOT_MATCHED
    )
    if mismatched:
        status = ConditionSetStatus.TYPE_MISMATCH
    elif missing:
        status = ConditionSetStatus.MISSING
    elif nonmatching:
        status = ConditionSetStatus.NOT_MATCHED
    else:
        status = ConditionSetStatus.MATCHED
    return ConditionSetEvaluation(
        status=status,
        predicates=rows,
        missing_moderators=missing,
        type_mismatch_moderators=mismatched,
        nonmatching_moderators=nonmatching,
    )


class MeaningfulEffectThreshold(ContractModel):
    """Minimum scientifically meaningful magnitude on one harmonized effect scale."""

    threshold_version: Literal["meaningful-effect-v1"] = "meaningful-effect-v1"
    minimum_magnitude: Annotated[float, Field(gt=0)]
    measure: HarmonizedMeasure
    unit: str | None = None

    @field_validator("minimum_magnitude")
    @classmethod
    def validate_magnitude(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("meaningful_effect_threshold_nonfinite")
        return value

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("meaningful_effect_threshold_unit_empty")
        return normalized

    @model_validator(mode="after")
    def validate_scale(self) -> MeaningfulEffectThreshold:
        if self.measure is HarmonizedMeasure.MEAN_DIFFERENCE and self.unit is None:
            raise ValueError("mean_difference_threshold_requires_unit")
        if self.measure is not HarmonizedMeasure.MEAN_DIFFERENCE and self.unit is not None:
            raise ValueError("unitless_harmonized_threshold_forbids_unit")
        return self


class ClaimTargetV2(ContractModel):
    """A frozen, condition-qualified, magnitude-aware directional target."""

    claim_schema_version: Literal["condition-qualified-claim-v2"] = "condition-qualified-claim-v2"
    claim_id: Annotated[str, Field(min_length=1)]
    direction: ClaimDirection
    outcome_name: Annotated[str, Field(min_length=1)]
    contrast_id: Annotated[str, Field(min_length=1)] | None = None
    conditions: Annotated[list[ConditionPredicate], Field(min_length=1)]
    meaningful_effect_threshold: MeaningfulEffectThreshold
    specification_status: ClaimSpecificationStatus
    parent_claim_sha256: str | None = None
    discovery_source_sha256: str | None = None
    claim_sha256: str

    @field_validator("claim_id", "outcome_name")
    @classmethod
    def normalize_required_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("qualified_claim_name_empty")
        return normalized

    @field_validator("contrast_id")
    @classmethod
    def normalize_optional_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("qualified_claim_contrast_empty")
        return normalized

    @field_validator("parent_claim_sha256", "discovery_source_sha256", "claim_sha256")
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("qualified_claim_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_target(self) -> ClaimTargetV2:
        ordered = sorted(self.conditions, key=_condition_sort_key)
        if self.conditions != ordered:
            raise ValueError("claim_conditions_must_be_canonically_sorted")
        moderators = [condition.moderator for condition in self.conditions]
        if len(moderators) != len(set(moderators)):
            raise ValueError("claim_condition_moderators_duplicate")
        discovered = self.specification_status is ClaimSpecificationStatus.DISCOVERED_HYPOTHESIS
        if discovered != (self.parent_claim_sha256 is not None):
            raise ValueError("discovered_claim_parent_hash_mismatch")
        if discovered != (self.discovery_source_sha256 is not None):
            raise ValueError("discovered_claim_source_hash_mismatch")
        payload = self.model_dump(mode="json", exclude={"claim_sha256"})
        if hash_canonical(payload) != self.claim_sha256:
            raise ValueError("qualified_claim_hash_mismatch")
        return self


def freeze_claim_target_v2(
    *,
    claim_id: str,
    direction: ClaimDirection,
    outcome_name: str,
    conditions: Sequence[ConditionPredicate | Mapping[str, Any]],
    meaningful_effect_threshold: MeaningfulEffectThreshold,
    contrast_id: str | None = None,
    specification_status: ClaimSpecificationStatus = ClaimSpecificationStatus.PRESPECIFIED,
    parent_claim_sha256: str | None = None,
    discovery_source_sha256: str | None = None,
) -> ClaimTargetV2:
    """Canonicalize and self-hash a qualified claim target."""

    parsed = [
        condition
        if isinstance(condition, ConditionPredicate)
        else ConditionPredicate.model_validate(condition)
        for condition in conditions
    ]
    parsed.sort(key=_condition_sort_key)
    payload: dict[str, Any] = {
        "claim_schema_version": "condition-qualified-claim-v2",
        "claim_id": claim_id.strip(),
        "direction": direction,
        "outcome_name": outcome_name.strip(),
        "contrast_id": contrast_id.strip() if contrast_id is not None else None,
        "conditions": parsed,
        "meaningful_effect_threshold": meaningful_effect_threshold,
        "specification_status": specification_status,
        "parent_claim_sha256": parent_claim_sha256,
        "discovery_source_sha256": discovery_source_sha256,
    }
    return ClaimTargetV2.model_validate({**payload, "claim_sha256": hash_canonical(payload)})


class GlobalConditionDependenceTargetV1(ContractModel):
    """Prespecified target for a global qualitative effect-modification verdict.

    This is deliberately distinct from :class:`ClaimTargetV2`, which asks whether a
    directional claim is supported *inside* one fixed condition stratum. Here the
    target is reproducibly opposite effect polarity across a frozen moderator family.
    ``reference_direction`` only orients audit sensitivity scores; it is not the
    condition-dependence verdict.
    """

    target_version: Literal["global-condition-dependence-target-v1"] = (
        "global-condition-dependence-target-v1"
    )
    claim_id: Annotated[str, Field(min_length=1)]
    reference_direction: ClaimDirection
    outcome_name: Annotated[str, Field(min_length=1)]
    contrast_id: str | None = None
    contrast_label: Annotated[str, Field(min_length=1)]
    estimand: Annotated[str, Field(min_length=1)]
    positive_direction_means: Annotated[str, Field(min_length=1)]
    treatment_role: ArmRole
    comparator_role: ArmRole
    measure: HarmonizedMeasure
    unit: str | None = None
    moderator_names: Annotated[list[str], Field(min_length=1)]
    specification_status: Literal["prespecified"] = "prespecified"
    target_semantics: Literal[
        "qualitative predictive effect modification across prespecified moderator "
        "levels; not a causal interaction"
    ] = (
        "qualitative predictive effect modification across prespecified moderator "
        "levels; not a causal interaction"
    )
    target_sha256: str

    @field_validator(
        "claim_id",
        "outcome_name",
        "contrast_label",
        "estimand",
        "positive_direction_means",
    )
    @classmethod
    def normalize_required_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("global_condition_target_name_empty")
        return normalized

    @field_validator("contrast_id")
    @classmethod
    def normalize_optional_contrast_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("global_condition_target_contrast_id_empty")
        return normalized

    @field_validator("moderator_names")
    @classmethod
    def validate_moderator_names(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)) or any(not name.strip() for name in value):
            raise ValueError("global_condition_target_moderators_not_sorted_unique")
        return value

    @field_validator("target_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("global_condition_target_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_target(self) -> GlobalConditionDependenceTargetV1:
        if self.measure is HarmonizedMeasure.MEAN_DIFFERENCE:
            if self.unit is None or not self.unit.strip():
                raise ValueError("global_condition_mean_difference_requires_unit")
        elif self.unit is not None:
            raise ValueError("global_condition_unitless_measure_forbids_unit")
        payload = self.model_dump(mode="json", exclude={"target_sha256"})
        if hash_canonical(payload) != self.target_sha256:
            raise ValueError("global_condition_target_hash_mismatch")
        return self


def freeze_global_condition_dependence_target(
    *,
    claim_id: str,
    reference_direction: ClaimDirection,
    outcome_name: str,
    contrast_label: str,
    estimand: str,
    positive_direction_means: str,
    treatment_role: ArmRole,
    comparator_role: ArmRole,
    measure: HarmonizedMeasure,
    moderator_names: Sequence[str],
    contrast_id: str | None = None,
    unit: str | None = None,
) -> GlobalConditionDependenceTargetV1:
    """Freeze one explicit global condition-dependence target."""

    payload: dict[str, Any] = {
        "target_version": "global-condition-dependence-target-v1",
        "claim_id": claim_id.strip(),
        "reference_direction": reference_direction,
        "outcome_name": outcome_name.strip(),
        "contrast_id": contrast_id.strip() if contrast_id is not None else None,
        "contrast_label": contrast_label.strip(),
        "estimand": estimand.strip(),
        "positive_direction_means": positive_direction_means.strip(),
        "treatment_role": treatment_role,
        "comparator_role": comparator_role,
        "measure": measure,
        "unit": unit.strip() if unit is not None else None,
        "moderator_names": sorted(moderator_names),
        "specification_status": "prespecified",
        "target_semantics": (
            "qualitative predictive effect modification across prespecified moderator "
            "levels; not a causal interaction"
        ),
    }
    return GlobalConditionDependenceTargetV1.model_validate(
        {**payload, "target_sha256": hash_canonical(payload)}
    )


class QualifiedClaimAmendment(ContractModel):
    """Deterministic hypothesis amendment that is ineligible on its discovery corpus."""

    amendment_version: Literal["qualified-claim-amendment-v1"] = "qualified-claim-amendment-v1"
    parent_target: ClaimTargetV2
    parent_claim_sha256: str
    source_synthesis_sha256: str
    proposed_target: ClaimTargetV2
    status: Literal["hypothesis_for_independent_confirmation"] = (
        "hypothesis_for_independent_confirmation"
    )
    eligible_for_source_corpus_release: Literal[False] = False
    amendment_sha256: str

    @field_validator("parent_claim_sha256", "source_synthesis_sha256", "amendment_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("qualified_claim_amendment_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_amendment(self) -> QualifiedClaimAmendment:
        if self.parent_target.specification_status is not ClaimSpecificationStatus.PRESPECIFIED:
            raise ValueError("amendment_parent_must_be_prespecified")
        if self.parent_claim_sha256 != self.parent_target.claim_sha256:
            raise ValueError("amendment_parent_hash_mismatch")
        proposed = self.proposed_target
        if proposed.specification_status is not ClaimSpecificationStatus.DISCOVERED_HYPOTHESIS:
            raise ValueError("amendment_target_must_be_discovered_hypothesis")
        if proposed.parent_claim_sha256 != self.parent_claim_sha256:
            raise ValueError("amendment_target_parent_hash_mismatch")
        if proposed.discovery_source_sha256 != self.source_synthesis_sha256:
            raise ValueError("amendment_target_source_hash_mismatch")
        for name in ("direction", "outcome_name", "contrast_id", "meaningful_effect_threshold"):
            if getattr(proposed, name) != getattr(self.parent_target, name):
                raise ValueError(f"amendment_changes_frozen_claim_field:{name}")
        parent_conditions = {hash_canonical(row) for row in self.parent_target.conditions}
        proposed_conditions = {hash_canonical(row) for row in proposed.conditions}
        if not parent_conditions < proposed_conditions:
            raise ValueError("amendment_requires_at_least_one_additional_condition")
        payload = self.model_dump(mode="json", exclude={"amendment_sha256"})
        if hash_canonical(payload) != self.amendment_sha256:
            raise ValueError("qualified_claim_amendment_hash_mismatch")
        return self


def freeze_qualified_claim_amendment(
    *,
    parent_target: ClaimTargetV2,
    source_synthesis_sha256: str,
    amended_claim_id: str,
    discovered_conditions: Sequence[ConditionPredicate | Mapping[str, Any]],
) -> QualifiedClaimAmendment:
    """Freeze a deterministic, non-release-eligible hypothesis for later confirmation."""

    discovered = [
        condition
        if isinstance(condition, ConditionPredicate)
        else ConditionPredicate.model_validate(condition)
        for condition in discovered_conditions
    ]
    proposed = freeze_claim_target_v2(
        claim_id=amended_claim_id,
        direction=parent_target.direction,
        outcome_name=parent_target.outcome_name,
        contrast_id=parent_target.contrast_id,
        conditions=[*parent_target.conditions, *discovered],
        meaningful_effect_threshold=parent_target.meaningful_effect_threshold,
        specification_status=ClaimSpecificationStatus.DISCOVERED_HYPOTHESIS,
        parent_claim_sha256=parent_target.claim_sha256,
        discovery_source_sha256=source_synthesis_sha256,
    )
    payload: dict[str, Any] = {
        "amendment_version": "qualified-claim-amendment-v1",
        "parent_target": parent_target,
        "parent_claim_sha256": parent_target.claim_sha256,
        "source_synthesis_sha256": source_synthesis_sha256,
        "proposed_target": proposed,
        "status": "hypothesis_for_independent_confirmation",
        "eligible_for_source_corpus_release": False,
    }
    return QualifiedClaimAmendment.model_validate(
        {**payload, "amendment_sha256": hash_canonical(payload)}
    )


class QualifiedClaimVerdict(ContractModel):
    """Self-hashed synthesis verdict; audit and calibration remain separate gates."""

    verdict_version: Literal["qualified-claim-verdict-v1"] = "qualified-claim-verdict-v1"
    target_sha256: str
    specification_status: ClaimSpecificationStatus
    synthesis_sha256: str
    state: QualifiedClaimVerdictState
    reason: Annotated[str, Field(min_length=1)]
    mode: str
    matched_estimate_ids: list[str]
    condition_excluded_estimate_ids: list[str]
    missing_condition_estimate_ids: list[str]
    type_mismatch_estimate_ids: list[str]
    estimate: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    prediction_interval_lower: float | None = None
    prediction_interval_upper: float | None = None
    decision_margin: float | None = None
    synthesis_gate_passed: bool
    gate_semantics: Literal[
        "synthesis evidence only; audit and calibration gates remain required"
    ] = "synthesis evidence only; audit and calibration gates remain required"
    verdict_sha256: str

    @field_validator("target_sha256", "synthesis_sha256", "verdict_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("qualified_claim_verdict_sha256_invalid")
        return value

    @model_validator(mode="after")
    def validate_verdict(self) -> QualifiedClaimVerdict:
        identity_sets: list[set[str]] = []
        for name in (
            "matched_estimate_ids",
            "condition_excluded_estimate_ids",
            "missing_condition_estimate_ids",
            "type_mismatch_estimate_ids",
        ):
            values = getattr(self, name)
            if values != sorted(set(values)):
                raise ValueError(f"qualified_verdict_ids_not_sorted_unique:{name}")
            identity_sets.append(set(values))
        if sum(len(values) for values in identity_sets) != len(set().union(*identity_sets)):
            raise ValueError("qualified_verdict_identity_partitions_overlap")
        for lower_name, upper_name in (
            ("ci_lower", "ci_upper"),
            ("prediction_interval_lower", "prediction_interval_upper"),
        ):
            lower = getattr(self, lower_name)
            upper = getattr(self, upper_name)
            if (lower is None) != (upper is None):
                raise ValueError(f"qualified_verdict_interval_incomplete:{lower_name}")
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"qualified_verdict_interval_not_ordered:{lower_name}")
        numeric = (
            self.estimate,
            self.ci_lower,
            self.ci_upper,
            self.prediction_interval_lower,
            self.prediction_interval_upper,
            self.decision_margin,
        )
        if any(value is not None and not math.isfinite(value) for value in numeric):
            raise ValueError("qualified_verdict_value_nonfinite")
        discovered = self.specification_status is ClaimSpecificationStatus.DISCOVERED_HYPOTHESIS
        if discovered != (self.state is QualifiedClaimVerdictState.DISCOVERED_HYPOTHESIS_ONLY):
            raise ValueError("qualified_verdict_discovery_state_mismatch")
        expected_gate = self.state is QualifiedClaimVerdictState.PRESPECIFIED_SUPPORTED
        if self.synthesis_gate_passed != expected_gate:
            raise ValueError("qualified_verdict_gate_status_mismatch")
        payload = self.model_dump(mode="json", exclude={"verdict_sha256"})
        if hash_canonical(payload) != self.verdict_sha256:
            raise ValueError("qualified_claim_verdict_hash_mismatch")
        return self


def freeze_qualified_claim_verdict(
    *,
    target: ClaimTargetV2,
    synthesis_sha256: str,
    state: QualifiedClaimVerdictState,
    reason: str,
    mode: str,
    matched_estimate_ids: Sequence[str] = (),
    condition_excluded_estimate_ids: Sequence[str] = (),
    missing_condition_estimate_ids: Sequence[str] = (),
    type_mismatch_estimate_ids: Sequence[str] = (),
    estimate: float | None = None,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    prediction_interval_lower: float | None = None,
    prediction_interval_upper: float | None = None,
    decision_margin: float | None = None,
) -> QualifiedClaimVerdict:
    """Build a canonical verdict tied to the exact target and synthesis hashes."""

    payload: dict[str, Any] = {
        "verdict_version": "qualified-claim-verdict-v1",
        "target_sha256": target.claim_sha256,
        "specification_status": target.specification_status,
        "synthesis_sha256": synthesis_sha256,
        "state": state,
        "reason": reason,
        "mode": mode,
        "matched_estimate_ids": sorted(set(matched_estimate_ids)),
        "condition_excluded_estimate_ids": sorted(set(condition_excluded_estimate_ids)),
        "missing_condition_estimate_ids": sorted(set(missing_condition_estimate_ids)),
        "type_mismatch_estimate_ids": sorted(set(type_mismatch_estimate_ids)),
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "prediction_interval_lower": prediction_interval_lower,
        "prediction_interval_upper": prediction_interval_upper,
        "decision_margin": decision_margin,
        "synthesis_gate_passed": (state is QualifiedClaimVerdictState.PRESPECIFIED_SUPPORTED),
        "gate_semantics": ("synthesis evidence only; audit and calibration gates remain required"),
    }
    return QualifiedClaimVerdict.model_validate(
        {**payload, "verdict_sha256": hash_canonical(payload)}
    )


__all__ = [
    "ClaimDirection",
    "ClaimSemanticsContractError",
    "ClaimSpecificationStatus",
    "ClaimTargetV2",
    "ConditionMatchStatus",
    "ConditionOperator",
    "ConditionPredicate",
    "ConditionPredicateEvaluation",
    "ConditionSetEvaluation",
    "ConditionSetStatus",
    "GlobalConditionDependenceTargetV1",
    "MeaningfulEffectThreshold",
    "QualifiedClaimAmendment",
    "QualifiedClaimVerdict",
    "QualifiedClaimVerdictState",
    "evaluate_condition_predicates",
    "freeze_claim_target_v2",
    "freeze_global_condition_dependence_target",
    "freeze_qualified_claim_amendment",
    "freeze_qualified_claim_verdict",
]
