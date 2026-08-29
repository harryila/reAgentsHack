"""Bounded, two-stage model-facing contract for native evidence extraction.

The official :class:`NativePublicationExtraction` schema remains the sole acceptance
authority.  This module deliberately gives a local model a smaller generation surface:

1. a bounded, value-free inventory of candidate numerical findings; and
2. one fixed-shape packet for each frozen candidate.

Packets are joined all-or-nothing.  No list is sliced and no successful subset is
salvaged when another candidate is missing, invalid, truncated, or inconsistent.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from functools import cache
from typing import Annotated, Any, Literal

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from pydantic import Field, TypeAdapter, field_validator, model_validator

from literature_multiverse.effects import (
    EffectFormat,
    EquivalenceConclusion,
    ReportedSignificance,
)
from literature_multiverse.evidence_graph import (
    ArmRole,
    OutcomeTimepoint,
    TimepointKind,
    TimeUnit,
)
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.models import ContractModel
from literature_multiverse.native_extraction import (
    NativeArm,
    NativeCohort,
    NativeContrast,
    NativeEffectPayload,
    NativeEvidenceSpan,
    NativeFinding,
    NativeModeratorValue,
    NativePublicationExtraction,
    NativeStudy,
)
from literature_multiverse.schemas import assert_closed_object_schema

INVENTORY_VERSION = "native-candidate-inventory-v1"
PACKET_VERSION = "native-candidate-packet-v1"
GENERATION_CONTRACT_VERSION = "native-bounded-two-stage-generation-v1"

# Nine is a sentinel capacity, not an accepted scientific payload size.  An inventory
# that reaches it is rejected wholesale even when the model says there are no more.
INVENTORY_SENTINEL_CAP = 9
MAX_ACCEPTED_CANDIDATES = INVENTORY_SENTINEL_CAP - 1
MAX_CANDIDATE_LINE_IDS = 4
MAX_IDENTITY_VALUES = 8
MAX_MODERATORS = 8
MAX_EVIDENCE_QUOTE_CHARACTERS = 1800
MAX_NUMERIC_SUPPORT_ITEMS = 24
MAX_DECIMAL_LEXEME_CHARACTERS = 32
MAX_ABSOLUTE_DECIMAL = Decimal("1e12")
MAX_COUNT = 1_000_000_000
MAX_TIMEPOINT = Decimal("1e6")

_DECIMAL_LEXEME_RE = re.compile(
    r"^-?(?:(?:0|[1-9][0-9]{0,12})(?:\.[0-9]{1,12})?|\.[0-9]{1,12})"
    r"(?:[eE][+-]?(?:0|[1-9][0-9]{0,2}))?$"
)
_UNSIGNED_INTEGER_LEXEME_RE = re.compile(r"^(?:0|[1-9][0-9]{0,9})$")
_NUMERIC_SIGN_CHARACTERS = frozenset(
    "+-\u2212\u2010\u2011\u2012\u2013\u2014\ufe63\uff0d\u207a\u207b\u208a\u208b"
)
_INEQUALITY_CHARACTERS = frozenset("<>\u2264\u2265\u2266\u2267")
_NUMERIC_GROUPING_WHITESPACE = " \t\u00a0\u2007\u2009\u202f"

EffectKind = Literal[
    "direct_standard_error",
    "direct_variance",
    "direct_confidence_interval",
    "continuous_group_statistics",
    "binary_group_statistics",
]


class NativeBoundedGenerationError(ValueError):
    """A bounded inventory, packet, or deterministic assembly is unsafe."""


BoundedDecimalLexeme = Annotated[
    str, Field(min_length=1, max_length=MAX_DECIMAL_LEXEME_CHARACTERS)
]
BoundedUnsignedIntegerLexeme = Annotated[
    str, Field(min_length=1, max_length=10)
]


def _parse_decimal(
    value: str,
    *,
    code: str,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
    minimum_exclusive: bool = False,
    maximum_exclusive: bool = False,
) -> Decimal:
    """Parse the closed JSON-string number grammar without float overflow/coercion."""

    if not _DECIMAL_LEXEME_RE.fullmatch(value):
        raise ValueError(f"{code}_lexeme_invalid")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:  # pragma: no cover - regex already excludes it
        raise ValueError(f"{code}_lexeme_invalid") from exc
    if not parsed.is_finite() or abs(parsed) > MAX_ABSOLUTE_DECIMAL:
        raise ValueError(f"{code}_magnitude_invalid")
    if minimum is not None and (
        parsed <= minimum if minimum_exclusive else parsed < minimum
    ):
        raise ValueError(f"{code}_below_minimum")
    if maximum is not None and (
        parsed >= maximum if maximum_exclusive else parsed > maximum
    ):
        raise ValueError(f"{code}_above_maximum")
    return parsed


def _parse_unsigned_integer(
    value: str,
    *,
    code: str,
    minimum: int = 0,
    maximum: int = MAX_COUNT,
) -> int:
    if not _UNSIGNED_INTEGER_LEXEME_RE.fullmatch(value):
        raise ValueError(f"{code}_lexeme_invalid")
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{code}_magnitude_invalid")
    return parsed


class NativeCandidateDescriptor(ContractModel):
    """Value-free source anchor frozen before any numerical packet is requested."""

    candidate_index: Annotated[int, Field(ge=1, le=INVENTORY_SENTINEL_CAP)]
    outcome_name: Annotated[str, Field(min_length=1, max_length=64)]
    effect_kind: EffectKind
    line_ids: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=32)]],
        Field(min_length=1, max_length=MAX_CANDIDATE_LINE_IDS),
    ]

    @field_validator("line_ids")
    @classmethod
    def validate_line_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("native_candidate_line_ids_not_sorted_unique")
        return value

    @property
    def descriptor_sha256(self) -> str:
        return hash_canonical(self)


class NativeCandidateInventory(ContractModel):
    inventory_version: Literal["native-candidate-inventory-v1"] = INVENTORY_VERSION
    inventory_status: Literal[
        "candidates_found",
        "no_candidate_found",
        "overflow_or_uncertain",
    ]
    candidates: Annotated[
        list[NativeCandidateDescriptor], Field(max_length=INVENTORY_SENTINEL_CAP)
    ]
    has_more_or_uncertain: bool

    @model_validator(mode="after")
    def validate_inventory(self) -> NativeCandidateInventory:
        indices = [candidate.candidate_index for candidate in self.candidates]
        if indices != list(range(1, len(indices) + 1)):
            raise ValueError("native_candidate_indices_not_contiguous")
        signatures = [
            (
                candidate.outcome_name,
                candidate.effect_kind,
                tuple(candidate.line_ids),
            )
            for candidate in self.candidates
        ]
        if len(signatures) != len(set(signatures)):
            raise ValueError("native_candidate_descriptor_signature_duplicate")
        if self.inventory_status == "candidates_found":
            if not self.candidates or self.has_more_or_uncertain:
                raise ValueError("native_candidate_inventory_found_state_invalid")
        elif self.inventory_status == "no_candidate_found":
            if self.candidates or self.has_more_or_uncertain:
                raise ValueError("native_candidate_inventory_empty_state_invalid")
        elif not self.has_more_or_uncertain:
            raise ValueError("native_candidate_inventory_overflow_requires_uncertainty")
        return self

    def authorizes_packet_generation(self) -> bool:
        return (
            self.inventory_status == "candidates_found"
            and not self.has_more_or_uncertain
            and 0 < len(self.candidates) < INVENTORY_SENTINEL_CAP
        )

    def blocking_status(self) -> str | None:
        if self.inventory_status == "no_candidate_found":
            return "inventory_no_candidate_non_authorizing"
        if not self.authorizes_packet_generation():
            return "inventory_capacity_or_uncertainty_non_authorizing"
        return None


BoundedKey = Annotated[str, Field(min_length=1, max_length=64)]
BoundedLabel = Annotated[str, Field(min_length=1, max_length=256)]
OptionalBoundedLabel = Annotated[str, Field(min_length=1, max_length=256)] | None
IdentityValue = Annotated[str, Field(min_length=1, max_length=128)]


class BoundedStudyHeader(ContractModel):
    key: BoundedKey
    source_label: BoundedLabel
    design: OptionalBoundedLabel = None
    registration_ids: Annotated[
        list[IdentityValue], Field(max_length=MAX_IDENTITY_VALUES)
    ] = Field(default_factory=list)

    @field_validator("registration_ids")
    @classmethod
    def validate_registration_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("bounded_study_registration_ids_not_sorted_unique")
        return value


class BoundedCohortHeader(ContractModel):
    key: BoundedKey
    source_labels: Annotated[
        list[BoundedLabel], Field(min_length=1, max_length=MAX_IDENTITY_VALUES)
    ]
    registry_ids: Annotated[
        list[IdentityValue], Field(max_length=MAX_IDENTITY_VALUES)
    ] = Field(default_factory=list)
    dataset_ids: Annotated[
        list[IdentityValue], Field(max_length=MAX_IDENTITY_VALUES)
    ] = Field(default_factory=list)
    population_description: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    recruitment_period: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    total_sample_size: BoundedUnsignedIntegerLexeme | None = None

    @field_validator("source_labels", "registry_ids", "dataset_ids")
    @classmethod
    def validate_identity_lists(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("bounded_cohort_identity_values_not_sorted_unique")
        return value

    @field_validator("total_sample_size")
    @classmethod
    def validate_total_sample_size(cls, value: str | None) -> str | None:
        if value is not None:
            _parse_unsigned_integer(
                value, code="bounded_total_sample_size", minimum=1
            )
        return value


class BoundedArm(ContractModel):
    key: BoundedKey
    label: BoundedLabel
    role: ArmRole
    description: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    sample_size: BoundedUnsignedIntegerLexeme | None = None

    @field_validator("sample_size")
    @classmethod
    def validate_sample_size(cls, value: str | None) -> str | None:
        if value is not None:
            _parse_unsigned_integer(value, code="bounded_arm_sample_size", minimum=1)
        return value

    def to_native(self) -> NativeArm:
        payload = self.model_dump(mode="json")
        if self.sample_size is not None:
            payload["sample_size"] = _parse_unsigned_integer(
                self.sample_size, code="bounded_arm_sample_size", minimum=1
            )
        return NativeArm.model_validate(payload)


class BoundedContrast(ContractModel):
    key: BoundedKey
    label: BoundedLabel
    estimand: OptionalBoundedLabel = None
    positive_direction_means: BoundedLabel


class BoundedTimepoint(ContractModel):
    kind: TimepointKind
    value: BoundedDecimalLexeme | None = None
    lower: BoundedDecimalLexeme | None = None
    upper: BoundedDecimalLexeme | None = None
    unit: TimeUnit | None = None
    anchor: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    raw_label: Annotated[str, Field(min_length=1, max_length=128)] | None = None

    @field_validator("value", "lower", "upper")
    @classmethod
    def validate_finite(cls, value: str | None) -> str | None:
        if value is not None:
            _parse_decimal(
                value,
                code="bounded_timepoint_value",
                minimum=Decimal("0"),
                maximum=MAX_TIMEPOINT,
            )
        return value

    @model_validator(mode="after")
    def validate_official_shape(self) -> BoundedTimepoint:
        self.to_native()
        return self

    def to_native(self) -> OutcomeTimepoint:
        payload = self.model_dump(mode="json")
        for field_name in ("value", "lower", "upper"):
            raw = payload[field_name]
            if raw is not None:
                payload[field_name] = float(
                    _parse_decimal(
                        raw,
                        code="bounded_timepoint_value",
                        minimum=Decimal("0"),
                        maximum=MAX_TIMEPOINT,
                    )
                )
        return OutcomeTimepoint.model_validate(payload)


class BoundedFindingHeader(ContractModel):
    key: BoundedKey
    outcome_name: Annotated[str, Field(min_length=1, max_length=64)]
    timepoint: BoundedTimepoint
    analysis_population: OptionalBoundedLabel = None


class BoundedEvidence(ContractModel):
    source_locator: Annotated[str, Field(min_length=1, max_length=512)]
    quote: Annotated[
        str, Field(min_length=1, max_length=MAX_EVIDENCE_QUOTE_CHARACTERS)
    ]
    section: Annotated[str, Field(min_length=1, max_length=64)]
    line_ids: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=32)]],
        Field(min_length=1, max_length=MAX_CANDIDATE_LINE_IDS),
    ]

    @field_validator("line_ids")
    @classmethod
    def validate_line_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("bounded_evidence_line_ids_not_sorted_unique")
        return value

    def to_native(self) -> NativeEvidenceSpan:
        return NativeEvidenceSpan(
            source_locator=self.source_locator,
            quote=self.quote,
            section=self.section,
            page=None,
            char_start=None,
            char_end=None,
            line_ids=self.line_ids,
        )


ModeratorScalar = Annotated[str, Field(min_length=1, max_length=128)] | bool | None


class BoundedModerator(ContractModel):
    """A bounded source label; v1 never silently treats it as a numeric covariate."""

    name: Annotated[str, Field(min_length=1, max_length=64)]
    value: ModeratorScalar


_DIRECT_EFFECT_FORMATS = set(EffectFormat) - {EffectFormat.UNSPECIFIED}
_CONTINUOUS_EFFECT_FORMATS = {
    EffectFormat.MEAN_DIFFERENCE,
    EffectFormat.COHENS_D,
    EffectFormat.HEDGES_G,
}
_BINARY_EFFECT_FORMATS = {
    EffectFormat.ODDS_RATIO,
    EffectFormat.LOG_ODDS_RATIO,
    EffectFormat.RISK_RATIO,
    EffectFormat.LOG_RISK_RATIO,
}
_POSITIVE_RATIO_FORMATS = {EffectFormat.ODDS_RATIO, EffectFormat.RISK_RATIO}


def _validate_direct_effect_format(value: EffectFormat) -> None:
    if value not in _DIRECT_EFFECT_FORMATS:
        raise ValueError("bounded_direct_effect_format_incompatible")


def _validate_direct_effect_domain(
    effect_format: EffectFormat,
    *values: Decimal,
) -> None:
    if effect_format in _POSITIVE_RATIO_FORMATS and any(
        value <= Decimal("0") for value in values
    ):
        raise ValueError("bounded_direct_ratio_value_not_positive")


def _validate_continuous_effect_format(value: EffectFormat) -> None:
    if value not in _CONTINUOUS_EFFECT_FORMATS:
        raise ValueError("bounded_continuous_effect_format_incompatible")


def _validate_binary_effect_format(value: EffectFormat) -> None:
    if value not in _BINARY_EFFECT_FORMATS:
        raise ValueError("bounded_binary_effect_format_incompatible")


class BoundedEffectCommon(ContractModel):
    reported_p_value: BoundedDecimalLexeme | None = None
    reported_significance: ReportedSignificance = ReportedSignificance.NOT_REPORTED
    equivalence_conclusion: EquivalenceConclusion = EquivalenceConclusion.NOT_TESTED
    equivalence_margin: BoundedDecimalLexeme | None = None
    moderators: Annotated[
        list[BoundedModerator], Field(max_length=MAX_MODERATORS)
    ] = Field(default_factory=list)
    # No model-authored derivations in v1. Deterministic harmonization occurs after
    # every reported input is source-token grounded.
    extraction_method: Literal["reported"] = "reported"

    @field_validator("reported_p_value")
    @classmethod
    def validate_reported_p_value(cls, value: str | None) -> str | None:
        if value is not None:
            _parse_decimal(
                value,
                code="bounded_reported_p_value",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            )
        return value

    @field_validator("equivalence_margin")
    @classmethod
    def validate_equivalence_margin(cls, value: str | None) -> str | None:
        if value is not None:
            _parse_decimal(
                value,
                code="bounded_equivalence_margin",
                minimum=Decimal("0"),
                minimum_exclusive=True,
            )
        return value

    @field_validator("moderators")
    @classmethod
    def validate_moderators(cls, value: list[BoundedModerator]) -> list[BoundedModerator]:
        names = [item.name for item in value]
        if names != sorted(set(names)):
            raise ValueError("bounded_moderator_names_not_sorted_unique")
        return value

    def _common_native(self) -> dict[str, Any]:
        return {
            "reported_p_value": (
                float(_parse_decimal(self.reported_p_value, code="bounded_reported_p_value"))
                if self.reported_p_value is not None
                else None
            ),
            "reported_significance": self.reported_significance,
            "equivalence_conclusion": self.equivalence_conclusion,
            "equivalence_margin": (
                float(
                    _parse_decimal(
                        self.equivalence_margin,
                        code="bounded_equivalence_margin",
                        minimum=Decimal("0"),
                        minimum_exclusive=True,
                    )
                )
                if self.equivalence_margin is not None
                else None
            ),
            "moderators": [
                NativeModeratorValue(name=item.name, value=item.value)
                for item in self.moderators
            ],
            "extraction_method": self.extraction_method,
        }


class DirectStandardErrorEffect(BoundedEffectCommon):
    effect_kind: Literal["direct_standard_error"] = "direct_standard_error"
    effect_format: EffectFormat
    estimate: BoundedDecimalLexeme
    standard_error: BoundedDecimalLexeme
    unit: Annotated[str, Field(min_length=1, max_length=64)] | None = None

    @model_validator(mode="after")
    def validate_values_and_format(self) -> DirectStandardErrorEffect:
        _validate_direct_effect_format(self.effect_format)
        estimate = _parse_decimal(self.estimate, code="bounded_effect_estimate")
        _validate_direct_effect_domain(self.effect_format, estimate)
        _parse_decimal(
            self.standard_error,
            code="bounded_effect_standard_error",
            minimum=Decimal("0"),
            minimum_exclusive=True,
        )
        return self

    def to_native(self) -> NativeEffectPayload:
        return NativeEffectPayload(
            effect_format=self.effect_format,
            estimate=float(_parse_decimal(self.estimate, code="bounded_effect_estimate")),
            standard_error=float(
                _parse_decimal(
                    self.standard_error,
                    code="bounded_effect_standard_error",
                    minimum=Decimal("0"),
                    minimum_exclusive=True,
                )
            ),
            unit=self.unit,
            **self._common_native(),
        )


class DirectVarianceEffect(BoundedEffectCommon):
    effect_kind: Literal["direct_variance"] = "direct_variance"
    effect_format: EffectFormat
    estimate: BoundedDecimalLexeme
    variance: BoundedDecimalLexeme
    unit: Annotated[str, Field(min_length=1, max_length=64)] | None = None

    @model_validator(mode="after")
    def validate_values_and_format(self) -> DirectVarianceEffect:
        _validate_direct_effect_format(self.effect_format)
        estimate = _parse_decimal(self.estimate, code="bounded_effect_estimate")
        _validate_direct_effect_domain(self.effect_format, estimate)
        _parse_decimal(
            self.variance,
            code="bounded_effect_variance",
            minimum=Decimal("0"),
            minimum_exclusive=True,
        )
        return self

    def to_native(self) -> NativeEffectPayload:
        return NativeEffectPayload(
            effect_format=self.effect_format,
            estimate=float(_parse_decimal(self.estimate, code="bounded_effect_estimate")),
            variance=float(
                _parse_decimal(
                    self.variance,
                    code="bounded_effect_variance",
                    minimum=Decimal("0"),
                    minimum_exclusive=True,
                )
            ),
            unit=self.unit,
            **self._common_native(),
        )


class DirectConfidenceIntervalEffect(BoundedEffectCommon):
    effect_kind: Literal["direct_confidence_interval"] = "direct_confidence_interval"
    effect_format: EffectFormat
    estimate: BoundedDecimalLexeme
    ci_lower: BoundedDecimalLexeme
    ci_upper: BoundedDecimalLexeme
    ci_level: BoundedDecimalLexeme
    unit: Annotated[str, Field(min_length=1, max_length=64)] | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> DirectConfidenceIntervalEffect:
        _validate_direct_effect_format(self.effect_format)
        estimate = _parse_decimal(self.estimate, code="bounded_effect_estimate")
        lower = _parse_decimal(self.ci_lower, code="bounded_effect_ci_lower")
        upper = _parse_decimal(self.ci_upper, code="bounded_effect_ci_upper")
        _validate_direct_effect_domain(self.effect_format, estimate, lower, upper)
        _parse_decimal(
            self.ci_level,
            code="bounded_effect_ci_level",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
            minimum_exclusive=True,
            maximum_exclusive=True,
        )
        if not lower < upper:
            raise ValueError("bounded_effect_confidence_interval_not_ordered")
        if not lower <= estimate <= upper:
            raise ValueError("bounded_effect_estimate_outside_interval")
        return self

    def to_native(self) -> NativeEffectPayload:
        return NativeEffectPayload(
            effect_format=self.effect_format,
            estimate=float(_parse_decimal(self.estimate, code="bounded_effect_estimate")),
            ci_lower=float(_parse_decimal(self.ci_lower, code="bounded_effect_ci_lower")),
            ci_upper=float(_parse_decimal(self.ci_upper, code="bounded_effect_ci_upper")),
            ci_level=float(_parse_decimal(self.ci_level, code="bounded_effect_ci_level")),
            unit=self.unit,
            **self._common_native(),
        )


class ContinuousGroupEffect(BoundedEffectCommon):
    effect_kind: Literal["continuous_group_statistics"] = "continuous_group_statistics"
    effect_format: EffectFormat
    treatment_mean: BoundedDecimalLexeme
    treatment_sd: BoundedDecimalLexeme
    treatment_n: BoundedUnsignedIntegerLexeme
    control_mean: BoundedDecimalLexeme
    control_sd: BoundedDecimalLexeme
    control_n: BoundedUnsignedIntegerLexeme
    unit: Annotated[str, Field(min_length=1, max_length=64)] | None = None

    @model_validator(mode="after")
    def validate_statistics_and_format(self) -> ContinuousGroupEffect:
        _validate_continuous_effect_format(self.effect_format)
        _parse_decimal(self.treatment_mean, code="bounded_treatment_mean")
        _parse_decimal(
            self.treatment_sd,
            code="bounded_treatment_sd",
            minimum=Decimal("0"),
            minimum_exclusive=True,
        )
        _parse_unsigned_integer(self.treatment_n, code="bounded_treatment_n", minimum=2)
        _parse_decimal(self.control_mean, code="bounded_control_mean")
        _parse_decimal(
            self.control_sd,
            code="bounded_control_sd",
            minimum=Decimal("0"),
            minimum_exclusive=True,
        )
        _parse_unsigned_integer(self.control_n, code="bounded_control_n", minimum=2)
        return self

    def to_native(self) -> NativeEffectPayload:
        return NativeEffectPayload(
            effect_format=self.effect_format,
            treatment_mean=float(
                _parse_decimal(self.treatment_mean, code="bounded_treatment_mean")
            ),
            treatment_sd=float(
                _parse_decimal(self.treatment_sd, code="bounded_treatment_sd")
            ),
            treatment_n=_parse_unsigned_integer(
                self.treatment_n, code="bounded_treatment_n", minimum=2
            ),
            control_mean=float(
                _parse_decimal(self.control_mean, code="bounded_control_mean")
            ),
            control_sd=float(
                _parse_decimal(self.control_sd, code="bounded_control_sd")
            ),
            control_n=_parse_unsigned_integer(
                self.control_n, code="bounded_control_n", minimum=2
            ),
            unit=self.unit,
            **self._common_native(),
        )


class BinaryGroupEffect(BoundedEffectCommon):
    effect_kind: Literal["binary_group_statistics"] = "binary_group_statistics"
    effect_format: EffectFormat
    treatment_events: BoundedUnsignedIntegerLexeme
    treatment_total: BoundedUnsignedIntegerLexeme
    control_events: BoundedUnsignedIntegerLexeme
    control_total: BoundedUnsignedIntegerLexeme

    @model_validator(mode="after")
    def validate_counts(self) -> BinaryGroupEffect:
        _validate_binary_effect_format(self.effect_format)
        treatment_events = _parse_unsigned_integer(
            self.treatment_events, code="bounded_treatment_events"
        )
        treatment_total = _parse_unsigned_integer(
            self.treatment_total, code="bounded_treatment_total", minimum=1
        )
        control_events = _parse_unsigned_integer(
            self.control_events, code="bounded_control_events"
        )
        control_total = _parse_unsigned_integer(
            self.control_total, code="bounded_control_total", minimum=1
        )
        if treatment_events > treatment_total:
            raise ValueError("bounded_treatment_events_exceed_total")
        if control_events > control_total:
            raise ValueError("bounded_control_events_exceed_total")
        return self

    def to_native(self) -> NativeEffectPayload:
        return NativeEffectPayload(
            effect_format=self.effect_format,
            treatment_events=_parse_unsigned_integer(
                self.treatment_events, code="bounded_treatment_events"
            ),
            treatment_total=_parse_unsigned_integer(
                self.treatment_total, code="bounded_treatment_total", minimum=1
            ),
            control_events=_parse_unsigned_integer(
                self.control_events, code="bounded_control_events"
            ),
            control_total=_parse_unsigned_integer(
                self.control_total, code="bounded_control_total", minimum=1
            ),
            **self._common_native(),
        )


NumericFieldPath = Literal[
    "cohort.total_sample_size",
    "treatment_arm.sample_size",
    "comparator_arm.sample_size",
    "finding.timepoint.value",
    "finding.timepoint.lower",
    "finding.timepoint.upper",
    "effect.estimate",
    "effect.standard_error",
    "effect.variance",
    "effect.ci_lower",
    "effect.ci_upper",
    "effect.ci_level",
    "effect.treatment_mean",
    "effect.treatment_sd",
    "effect.treatment_n",
    "effect.control_mean",
    "effect.control_sd",
    "effect.control_n",
    "effect.treatment_events",
    "effect.treatment_total",
    "effect.control_events",
    "effect.control_total",
    "effect.reported_p_value",
    "effect.equivalence_margin",
]


class BoundedNumericSupport(ContractModel):
    """One exact source-token receipt for one emitted scientific numeric leaf."""

    field_path: NumericFieldPath
    verbatim_token: Annotated[
        str, Field(min_length=1, max_length=MAX_DECIMAL_LEXEME_CHARACTERS + 1)
    ]
    normalization: Literal["identity", "percent_to_proportion"] = "identity"
    quote_start: Annotated[str, Field(min_length=1, max_length=4)]
    quote_end: Annotated[str, Field(min_length=1, max_length=4)]

    @model_validator(mode="after")
    def validate_offsets_and_token(self) -> BoundedNumericSupport:
        if (
            self.normalization == "percent_to_proportion"
            and not self.verbatim_token.endswith("%")
        ):
            raise ValueError("bounded_numeric_support_percent_marker_missing")
        numeric_lexeme = (
            self.verbatim_token[:-1]
            if self.normalization == "percent_to_proportion"
            and self.verbatim_token.endswith("%")
            else self.verbatim_token
        )
        _parse_decimal(numeric_lexeme, code="bounded_numeric_support_token")
        start = _parse_unsigned_integer(
            self.quote_start,
            code="bounded_numeric_support_quote_start",
            maximum=MAX_EVIDENCE_QUOTE_CHARACTERS,
        )
        end = _parse_unsigned_integer(
            self.quote_end,
            code="bounded_numeric_support_quote_end",
            maximum=MAX_EVIDENCE_QUOTE_CHARACTERS,
        )
        if end <= start:
            raise ValueError("bounded_numeric_support_offsets_not_ordered")
        return self


def _packet_numeric_lexemes(packet: NativeCandidatePacket[Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    optional_values = {
        "cohort.total_sample_size": packet.cohort.total_sample_size,
        "treatment_arm.sample_size": packet.treatment_arm.sample_size,
        "comparator_arm.sample_size": packet.comparator_arm.sample_size,
        "finding.timepoint.value": packet.finding.timepoint.value,
        "finding.timepoint.lower": packet.finding.timepoint.lower,
        "finding.timepoint.upper": packet.finding.timepoint.upper,
        "effect.reported_p_value": packet.effect.reported_p_value,
        "effect.equivalence_margin": packet.effect.equivalence_margin,
    }
    effect_fields = (
        "estimate",
        "standard_error",
        "variance",
        "ci_lower",
        "ci_upper",
        "ci_level",
        "treatment_mean",
        "treatment_sd",
        "treatment_n",
        "control_mean",
        "control_sd",
        "control_n",
        "treatment_events",
        "treatment_total",
        "control_events",
        "control_total",
    )
    optional_values.update(
        {
            f"effect.{field_name}": getattr(packet.effect, field_name, None)
            for field_name in effect_fields
        }
    )
    for path, value in optional_values.items():
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"bounded_packet_numeric_value_not_lexeme:{path}")
            values[path] = value
    return values


def _assert_numeric_source_token_boundary(
    *,
    quote: str,
    start: int,
    end: int,
    normalization: Literal["identity", "percent_to_proportion"],
    field_path: str,
) -> None:
    """Reject a supported slice that is only a fragment of a numeric expression."""

    prefix = quote[:start]
    suffix = quote[end:]
    stripped_prefix = prefix.rstrip(_NUMERIC_GROUPING_WHITESPACE)
    stripped_suffix = suffix.lstrip(_NUMERIC_GROUPING_WHITESPACE)
    immediate_prefix_invalid = bool(prefix) and (
        prefix[-1].isdigit()
        or (
            prefix[-1] in ".eE"
            and len(prefix) > 1
            and prefix[-2].isdigit()
        )
        or (
            prefix[-1] == ","
            and len(prefix) > 1
            and prefix[-2].isdigit()
        )
    )
    expression_prefix_invalid = bool(stripped_prefix) and (
        stripped_prefix[-1] in _NUMERIC_SIGN_CHARACTERS
        or stripped_prefix[-1] in _INEQUALITY_CHARACTERS
        or (
            len(stripped_prefix) < len(prefix)
            and stripped_prefix[-1].isdigit()
        )
    )
    immediate_suffix_invalid = bool(suffix) and (
        suffix[0].isdigit()
        or (
            suffix[0] == "."
            and len(suffix) > 1
            and suffix[1].isdigit()
        )
        or (
            suffix[0] == ","
            and len(suffix) > 1
            and suffix[1].isdigit()
        )
        or (
            suffix[0] in "eE"
            and len(suffix) > 1
            and (
                suffix[1].isdigit()
                or (
                    suffix[1] in _NUMERIC_SIGN_CHARACTERS
                    and len(suffix) > 2
                    and suffix[2].isdigit()
                )
            )
        )
    )
    grouped_suffix_invalid = (
        len(stripped_suffix) < len(suffix)
        and len(stripped_suffix) >= 3
        and stripped_suffix[:3].isdigit()
    )
    if (
        immediate_prefix_invalid
        or expression_prefix_invalid
        or immediate_suffix_invalid
        or grouped_suffix_invalid
    ):
        raise ValueError(f"bounded_numeric_support_token_boundary:{field_path}")
    if normalization == "identity" and stripped_suffix.startswith("%"):
        raise ValueError(f"bounded_numeric_support_unlicensed_percent_token:{field_path}")


def _validate_numeric_support(packet: NativeCandidatePacket[Any]) -> None:
    expected = _packet_numeric_lexemes(packet)
    paths = [item.field_path for item in packet.numeric_support]
    if paths != sorted(set(paths)):
        raise ValueError("bounded_numeric_support_paths_not_sorted_unique")
    if set(paths) != set(expected):
        raise ValueError("bounded_numeric_support_field_set_mismatch")
    spans = [(item.quote_start, item.quote_end) for item in packet.numeric_support]
    if len(spans) != len(set(spans)):
        raise ValueError("bounded_numeric_support_source_span_reused")
    for item in packet.numeric_support:
        start = _parse_unsigned_integer(
            item.quote_start,
            code="bounded_numeric_support_quote_start",
            maximum=MAX_EVIDENCE_QUOTE_CHARACTERS,
        )
        end = _parse_unsigned_integer(
            item.quote_end,
            code="bounded_numeric_support_quote_end",
            maximum=MAX_EVIDENCE_QUOTE_CHARACTERS,
        )
        if end > len(packet.evidence.quote):
            raise ValueError(f"bounded_numeric_support_offset_outside_quote:{item.field_path}")
        if packet.evidence.quote[start:end] != item.verbatim_token:
            raise ValueError(f"bounded_numeric_support_verbatim_mismatch:{item.field_path}")
        _assert_numeric_source_token_boundary(
            quote=packet.evidence.quote,
            start=start,
            end=end,
            normalization=item.normalization,
            field_path=item.field_path,
        )
        emitted = _parse_decimal(
            expected[item.field_path], code="bounded_numeric_support_emitted_value"
        )
        if item.normalization == "percent_to_proportion":
            if item.field_path not in {"effect.ci_level", "effect.reported_p_value"}:
                raise ValueError(
                    f"bounded_numeric_support_percent_normalization_forbidden:{item.field_path}"
                )
            if not item.verbatim_token.endswith("%"):
                raise ValueError(
                    f"bounded_numeric_support_percent_marker_missing:{item.field_path}"
                )
            supported = _parse_decimal(
                item.verbatim_token[:-1], code="bounded_numeric_support_token"
            )
            supported /= Decimal("100")
        else:
            if item.verbatim_token.endswith("%"):
                raise ValueError(
                    f"bounded_numeric_support_unlicensed_percent_token:{item.field_path}"
                )
            supported = _parse_decimal(
                item.verbatim_token, code="bounded_numeric_support_token"
            )
        if emitted != supported:
            raise ValueError(f"bounded_numeric_support_value_mismatch:{item.field_path}")


class NativeCandidatePacket[EffectT: BoundedEffectCommon](ContractModel):
    packet_version: Literal["native-candidate-packet-v1"] = PACKET_VERSION
    packet_status: Literal["completed"] = "completed"
    candidate_index: Annotated[int, Field(ge=1, le=MAX_ACCEPTED_CANDIDATES)]
    study: BoundedStudyHeader
    cohort: BoundedCohortHeader
    treatment_arm: BoundedArm
    comparator_arm: BoundedArm
    contrast: BoundedContrast
    finding: BoundedFindingHeader
    effect: EffectT
    evidence: BoundedEvidence
    numeric_support: Annotated[
        list[BoundedNumericSupport],
        Field(min_length=1, max_length=MAX_NUMERIC_SUPPORT_ITEMS),
    ]

    @model_validator(mode="after")
    def validate_packet_references(self) -> NativeCandidatePacket[EffectT]:
        if self.treatment_arm.key == self.comparator_arm.key:
            raise ValueError("bounded_packet_arms_not_distinct")
        if self.treatment_arm.role not in {ArmRole.INTERVENTION, ArmRole.EXPOSURE}:
            raise ValueError("bounded_packet_treatment_arm_role_invalid")
        if self.comparator_arm.role not in {ArmRole.COMPARATOR, ArmRole.CONTROL}:
            raise ValueError("bounded_packet_comparator_arm_role_invalid")
        _validate_numeric_support(self)
        return self


class NativeCandidateUnableToComplete(ContractModel):
    """Value-free, terminal escape when the frozen candidate cannot be completed."""

    packet_version: Literal["native-candidate-packet-v1"] = PACKET_VERSION
    packet_status: Literal["unable_to_complete"] = "unable_to_complete"
    candidate_index: Annotated[int, Field(ge=1, le=MAX_ACCEPTED_CANDIDATES)]
    reason: Literal[
        "source_ambiguous",
        "candidate_misrouted",
        "insufficient_numeric_support",
        "unsupported_effect_representation",
        "capacity_or_other_uncertainty",
    ]


type NativeCandidatePacketOutcome = (
    NativeCandidatePacket[Any] | NativeCandidateUnableToComplete
)


PACKET_MODELS: dict[str, type[NativeCandidatePacket[Any]]] = {
    "direct_standard_error": NativeCandidatePacket[DirectStandardErrorEffect],
    "direct_variance": NativeCandidatePacket[DirectVarianceEffect],
    "direct_confidence_interval": NativeCandidatePacket[
        DirectConfidenceIntervalEffect
    ],
    "continuous_group_statistics": NativeCandidatePacket[ContinuousGroupEffect],
    "binary_group_statistics": NativeCandidatePacket[BinaryGroupEffect],
}


@cache
def _packet_outcome_adapter(effect_kind: str) -> TypeAdapter[Any]:
    model = PACKET_MODELS[effect_kind]
    outcome = Annotated[
        model | NativeCandidateUnableToComplete,
        Field(discriminator="packet_status"),
    ]
    return TypeAdapter(outcome)


def _regex_free_schema(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _regex_free_schema(item)
            for key, item in value.items()
            if key not in {"pattern", "$schema", "$id"}
        }
    if isinstance(value, list):
        return [_regex_free_schema(item) for item in value]
    return value


def _constrain_named_properties(
    value: Any,
    *,
    line_ids: Sequence[str],
    outcomes: Sequence[str],
    source_locator: str | None = None,
    candidate_index: int | None = None,
    inventory_indices: Sequence[int] | None = None,
    sections: Sequence[str] | None = None,
    positive_direction_means: str | None = None,
) -> None:
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            outcome = properties.get("outcome_name")
            if isinstance(outcome, dict):
                outcome["enum"] = list(outcomes)
            locator = properties.get("source_locator")
            if isinstance(locator, dict) and source_locator is not None:
                locator["enum"] = [source_locator]
            section = properties.get("section")
            if isinstance(section, dict) and sections is not None:
                section["enum"] = list(sections)
            direction = properties.get("positive_direction_means")
            if isinstance(direction, dict) and positive_direction_means is not None:
                direction["enum"] = [positive_direction_means]
            index = properties.get("candidate_index")
            if isinstance(index, dict):
                if candidate_index is not None:
                    index["enum"] = [candidate_index]
                elif inventory_indices is not None:
                    index["enum"] = list(inventory_indices)
            line_schema = properties.get("line_ids")
            if isinstance(line_schema, dict) and isinstance(line_schema.get("items"), dict):
                line_schema["items"]["enum"] = list(line_ids)
        for item in value.values():
            _constrain_named_properties(
                item,
                line_ids=line_ids,
                outcomes=outcomes,
                source_locator=source_locator,
                candidate_index=candidate_index,
                inventory_indices=inventory_indices,
                sections=sections,
                positive_direction_means=positive_direction_means,
            )
    elif isinstance(value, list):
        for item in value:
            _constrain_named_properties(
                item,
                line_ids=line_ids,
                outcomes=outcomes,
                source_locator=source_locator,
                candidate_index=candidate_index,
                inventory_indices=inventory_indices,
                sections=sections,
                positive_direction_means=positive_direction_means,
            )


def _schema_types(node: Mapping[str, Any]) -> set[str]:
    raw = node.get("type")
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return set(raw)
    return set()


def assert_bounded_generation_schema(schema: Mapping[str, Any]) -> None:
    """Mechanically reject any reachable open object or unbounded container/string."""

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            types = _schema_types(value)
            if "array" in types and not isinstance(value.get("maxItems"), int):
                raise NativeBoundedGenerationError(
                    f"native_generation_array_unbounded:{path}"
                )
            if (
                "string" in types
                and "enum" not in value
                and "const" not in value
                and not isinstance(value.get("maxLength"), int)
            ):
                raise NativeBoundedGenerationError(
                    f"native_generation_string_unbounded:{path}"
                )
            if "object" in types and value.get("additionalProperties") is not False:
                raise NativeBoundedGenerationError(
                    f"native_generation_object_open:{path}"
                )
            if types.intersection({"integer", "number"}) and not (
                isinstance(value.get("enum"), list) and value["enum"]
            ):
                raise NativeBoundedGenerationError(
                    f"native_generation_numeric_lexeme_unbounded:{path}"
                )
            for key, item in value.items():
                visit(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(schema, "$")


def inventory_generation_schema(
    *,
    exposed_line_ids: Sequence[str],
    allowed_outcomes: Sequence[str],
) -> dict[str, Any]:
    line_ids = sorted(set(exposed_line_ids)) or ["NO_EXPOSED_SOURCE_LINE"]
    outcomes = sorted(set(allowed_outcomes))
    if not outcomes:
        raise NativeBoundedGenerationError("native_inventory_outcomes_empty")
    schema = _regex_free_schema(
        NativeCandidateInventory.model_json_schema(mode="validation")
    )
    _constrain_named_properties(
        schema,
        line_ids=line_ids,
        outcomes=outcomes,
        inventory_indices=range(1, INVENTORY_SENTINEL_CAP + 1),
    )
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "urn:literature-multiverse:native-candidate-inventory:v1"
    assert_closed_object_schema(schema)
    assert_bounded_generation_schema(schema)
    return schema


def packet_generation_schema(
    *,
    candidate: NativeCandidateDescriptor,
    exposed_line_ids: Sequence[str],
    source_locator: str,
    allowed_outcomes: Sequence[str],
    allowed_moderators: Sequence[str] = (),
    allowed_sections: Sequence[str] = ("FigureTable", "Methods", "Results"),
    outcome_positive_directions: Mapping[str, str],
) -> dict[str, Any]:
    if candidate.outcome_name not in set(allowed_outcomes):
        raise NativeBoundedGenerationError("native_packet_candidate_outcome_not_allowed")
    if not set(candidate.line_ids).issubset(set(exposed_line_ids)):
        raise NativeBoundedGenerationError("native_packet_candidate_line_not_exposed")
    if not allowed_sections:
        raise NativeBoundedGenerationError("native_packet_allowed_sections_empty")
    positive_direction = outcome_positive_directions.get(candidate.outcome_name)
    if not isinstance(positive_direction, str) or not positive_direction:
        raise NativeBoundedGenerationError(
            "native_packet_candidate_positive_direction_missing"
        )
    schema = _regex_free_schema(
        _packet_outcome_adapter(candidate.effect_kind).json_schema(mode="validation")
    )
    _constrain_named_properties(
        schema,
        line_ids=candidate.line_ids,
        outcomes=[candidate.outcome_name],
        source_locator=source_locator,
        candidate_index=candidate.candidate_index,
        sections=sorted(set(allowed_sections)),
        positive_direction_means=positive_direction,
    )
    _constrain_moderator_schema(schema, allowed_moderators=allowed_moderators)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema_context_sha256 = hash_canonical(
        {
            "candidate": candidate.model_dump(mode="json"),
            "exposed_line_ids": sorted(set(exposed_line_ids)),
            "source_locator": source_locator,
            "allowed_outcomes": sorted(set(allowed_outcomes)),
            "allowed_moderators": sorted(set(allowed_moderators)),
            "allowed_sections": sorted(set(allowed_sections)),
            "outcome_positive_directions": dict(
                sorted(outcome_positive_directions.items())
            ),
        }
    )
    schema["$id"] = (
        "urn:literature-multiverse:native-candidate-packet:v1:"
        f"{candidate.effect_kind}:{schema_context_sha256[:24]}"
    )
    assert_closed_object_schema(schema)
    assert_bounded_generation_schema(schema)
    return schema


def _constrain_moderator_schema(
    value: Any,
    *,
    allowed_moderators: Sequence[str],
) -> None:
    names = sorted(set(allowed_moderators))
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            moderators = properties.get("moderators")
            if isinstance(moderators, dict) and not names:
                moderators["maxItems"] = 0
            name = properties.get("name")
            if isinstance(name, dict) and names:
                name["enum"] = names
        for item in value.values():
            _constrain_moderator_schema(item, allowed_moderators=names)
    elif isinstance(value, list):
        for item in value:
            _constrain_moderator_schema(item, allowed_moderators=names)


def _strict_json_schema_postvalidate(
    value: Any,
    *,
    schema: Mapping[str, Any],
    error_prefix: str,
) -> dict[str, Any]:
    payload = (
        value.model_dump(mode="json")
        if isinstance(value, ContractModel)
        else deepcopy(value)
    )
    if not isinstance(payload, dict):
        raise NativeBoundedGenerationError(f"{error_prefix}_payload_not_object")
    validator_class = validator_for(schema)
    try:
        validator_class.check_schema(schema)
    except SchemaError as exc:  # pragma: no cover - generated schema is unit tested
        raise NativeBoundedGenerationError(f"{error_prefix}_schema_invalid") from exc
    errors = list(validator_class(schema).iter_errors(payload))
    if errors:
        validators = sorted(
            {
                str(error.validator) if error.validator is not None else "unknown"
                for error in errors
            }
        )
        raise NativeBoundedGenerationError(
            f"{error_prefix}_json_schema_validation_error:" + ",".join(validators)
        )
    return payload


def validate_inventory_for_row(
    value: Any,
    *,
    exposed_line_ids: Sequence[str],
    allowed_outcomes: Sequence[str],
) -> NativeCandidateInventory:
    payload = _strict_json_schema_postvalidate(
        value,
        schema=inventory_generation_schema(
            exposed_line_ids=exposed_line_ids,
            allowed_outcomes=allowed_outcomes,
        ),
        error_prefix="native_inventory",
    )
    inventory = NativeCandidateInventory.model_validate(payload)
    line_ids = set(exposed_line_ids)
    outcomes = set(allowed_outcomes)
    if not line_ids and inventory.inventory_status == "candidates_found":
        raise NativeBoundedGenerationError("native_inventory_zero_projection_candidates")
    if any(not set(candidate.line_ids).issubset(line_ids) for candidate in inventory.candidates):
        raise NativeBoundedGenerationError("native_inventory_line_id_not_exposed")
    if any(candidate.outcome_name not in outcomes for candidate in inventory.candidates):
        raise NativeBoundedGenerationError("native_inventory_outcome_not_prespecified")
    return inventory


def validate_packet_for_candidate(
    value: Any,
    *,
    candidate: NativeCandidateDescriptor,
    exposed_line_ids: Sequence[str],
    source_locator: str,
    allowed_outcomes: Sequence[str],
    allowed_moderators: Sequence[str],
    allowed_sections: Sequence[str] = ("FigureTable", "Methods", "Results"),
    outcome_positive_directions: Mapping[str, str],
) -> NativeCandidatePacketOutcome:
    payload = _strict_json_schema_postvalidate(
        value,
        schema=packet_generation_schema(
            candidate=candidate,
            exposed_line_ids=exposed_line_ids,
            source_locator=source_locator,
            allowed_outcomes=allowed_outcomes,
            allowed_moderators=allowed_moderators,
            allowed_sections=allowed_sections,
            outcome_positive_directions=outcome_positive_directions,
        ),
        error_prefix="native_packet",
    )
    packet = _packet_outcome_adapter(candidate.effect_kind).validate_python(payload)
    if packet.candidate_index != candidate.candidate_index:
        raise NativeBoundedGenerationError("native_packet_candidate_index_mismatch")
    if isinstance(packet, NativeCandidateUnableToComplete):
        return packet
    if packet.finding.outcome_name != candidate.outcome_name:
        raise NativeBoundedGenerationError("native_packet_candidate_outcome_mismatch")
    if packet.effect.effect_kind != candidate.effect_kind:
        raise NativeBoundedGenerationError("native_packet_candidate_effect_kind_mismatch")
    if packet.evidence.line_ids != candidate.line_ids:
        raise NativeBoundedGenerationError("native_packet_candidate_lineage_mismatch")
    if packet.evidence.source_locator != source_locator:
        raise NativeBoundedGenerationError("native_packet_source_locator_mismatch")
    if packet.evidence.section not in set(allowed_sections):
        raise NativeBoundedGenerationError("native_packet_section_not_exposed")
    if packet.contrast.positive_direction_means != outcome_positive_directions.get(
        candidate.outcome_name
    ):
        raise NativeBoundedGenerationError(
            "native_packet_positive_direction_mismatch"
        )
    if packet.finding.outcome_name not in set(allowed_outcomes):
        raise NativeBoundedGenerationError("native_packet_outcome_not_prespecified")
    if not set(packet.evidence.line_ids).issubset(set(exposed_line_ids)):
        raise NativeBoundedGenerationError("native_packet_line_id_not_exposed")
    moderator_names = {item.name for item in packet.effect.moderators}
    if not moderator_names.issubset(set(allowed_moderators)):
        raise NativeBoundedGenerationError("native_packet_moderator_not_prespecified")
    return packet


def _require_same(
    existing: Any | None,
    incoming: Any,
    *,
    code: str,
) -> Any:
    if existing is not None and hash_canonical(existing) != hash_canonical(incoming):
        raise NativeBoundedGenerationError(code)
    return incoming if existing is None else existing


def assemble_candidate_packets(
    *,
    inventory: NativeCandidateInventory,
    packets: Sequence[NativeCandidatePacketOutcome],
    exposed_line_ids: Sequence[str],
    source_locator: str,
    allowed_outcomes: Sequence[str],
    allowed_moderators: Sequence[str],
    allowed_sections: Sequence[str] = ("FigureTable", "Methods", "Results"),
    outcome_positive_directions: Mapping[str, str],
) -> NativePublicationExtraction:
    """Join the exact candidate set into official v1 or reject the whole publication."""

    inventory = NativeCandidateInventory.model_validate(inventory.model_dump(mode="json"))
    if not inventory.authorizes_packet_generation():
        raise NativeBoundedGenerationError(
            inventory.blocking_status() or "native_inventory_non_authorizing"
        )
    if len(packets) != len(inventory.candidates):
        raise NativeBoundedGenerationError("native_packet_candidate_membership_mismatch")
    expected = [candidate.candidate_index for candidate in inventory.candidates]
    observed = [packet.candidate_index for packet in packets]
    if observed != expected:
        raise NativeBoundedGenerationError("native_packet_candidate_membership_mismatch")
    validated_packets: list[NativeCandidatePacket[Any]] = []
    for candidate, packet_value in zip(inventory.candidates, packets, strict=True):
        packet = validate_packet_for_candidate(
            packet_value,
            candidate=candidate,
            exposed_line_ids=exposed_line_ids,
            source_locator=source_locator,
            allowed_outcomes=allowed_outcomes,
            allowed_moderators=allowed_moderators,
            allowed_sections=allowed_sections,
            outcome_positive_directions=outcome_positive_directions,
        )
        if isinstance(packet, NativeCandidateUnableToComplete):
            raise NativeBoundedGenerationError(
                f"native_packet_unable_to_complete:{packet.reason}"
            )
        validated_packets.append(packet)

    studies: dict[str, BoundedStudyHeader] = {}
    cohort_headers: dict[tuple[str, str], BoundedCohortHeader] = {}
    arms: dict[tuple[str, str], dict[str, BoundedArm]] = {}
    contrasts: dict[tuple[str, str], dict[str, NativeContrast]] = {}
    findings: dict[tuple[str, str], dict[str, NativeFinding]] = {}
    descriptor_by_index = {
        candidate.candidate_index: candidate for candidate in inventory.candidates
    }
    for packet in validated_packets:
        descriptor = descriptor_by_index[packet.candidate_index]
        if (
            packet.finding.outcome_name != descriptor.outcome_name
            or packet.effect.effect_kind != descriptor.effect_kind
            or packet.evidence.line_ids != descriptor.line_ids
        ):
            raise NativeBoundedGenerationError("native_packet_descriptor_mismatch")
        studies[packet.study.key] = _require_same(
            studies.get(packet.study.key),
            packet.study,
            code="native_packet_study_metadata_conflict",
        )
        cohort_key = (packet.study.key, packet.cohort.key)
        cohort_headers[cohort_key] = _require_same(
            cohort_headers.get(cohort_key),
            packet.cohort,
            code="native_packet_cohort_metadata_conflict",
        )
        arm_map = arms.setdefault(cohort_key, {})
        for arm in (packet.treatment_arm, packet.comparator_arm):
            arm_map[arm.key] = _require_same(
                arm_map.get(arm.key),
                arm,
                code="native_packet_arm_metadata_conflict",
            )
        contrast = NativeContrast(
            key=packet.contrast.key,
            treatment_arm_key=packet.treatment_arm.key,
            comparator_arm_key=packet.comparator_arm.key,
            label=packet.contrast.label,
            estimand=packet.contrast.estimand,
            positive_direction_means=packet.contrast.positive_direction_means,
        )
        contrast_map = contrasts.setdefault(cohort_key, {})
        contrast_map[contrast.key] = _require_same(
            contrast_map.get(contrast.key),
            contrast,
            code="native_packet_contrast_metadata_conflict",
        )
        finding = NativeFinding(
            key=packet.finding.key,
            contrast_key=contrast.key,
            outcome_name=packet.finding.outcome_name,
            timepoint=packet.finding.timepoint.to_native(),
            analysis_population=packet.finding.analysis_population,
            effect=packet.effect.to_native(),
            evidence=packet.evidence.to_native(),
        )
        finding_map = findings.setdefault(cohort_key, {})
        if finding.key in finding_map:
            raise NativeBoundedGenerationError("native_packet_finding_key_duplicate")
        finding_map[finding.key] = finding

    native_studies: list[NativeStudy] = []
    for study_key in sorted(studies):
        header = studies[study_key]
        native_cohorts: list[NativeCohort] = []
        for cohort_key in sorted(key for key in cohort_headers if key[0] == study_key):
            cohort = cohort_headers[cohort_key]
            native_cohorts.append(
                NativeCohort(
                    key=cohort.key,
                    source_labels=cohort.source_labels,
                    registry_ids=cohort.registry_ids,
                    dataset_ids=cohort.dataset_ids,
                    population_description=cohort.population_description,
                    recruitment_period=cohort.recruitment_period,
                    total_sample_size=(
                        _parse_unsigned_integer(
                            cohort.total_sample_size,
                            code="bounded_total_sample_size",
                            minimum=1,
                        )
                        if cohort.total_sample_size is not None
                        else None
                    ),
                    arms=[
                        arms[cohort_key][key].to_native()
                        for key in sorted(arms[cohort_key])
                    ],
                    contrasts=[
                        contrasts[cohort_key][key]
                        for key in sorted(contrasts[cohort_key])
                    ],
                    findings=[
                        findings[cohort_key][key]
                        for key in sorted(findings[cohort_key])
                    ],
                )
            )
        native_studies.append(
            NativeStudy(
                key=header.key,
                source_label=header.source_label,
                design=header.design,
                registration_ids=header.registration_ids,
                cohorts=native_cohorts,
            )
        )
    official = NativePublicationExtraction(
        extraction_schema_version="native-publication-extraction-v1",
        status="estimable",
        studies=native_studies,
        non_estimability_reason=None,
        non_estimability_detail=None,
        warnings=[],
    )
    return NativePublicationExtraction.model_validate(official.model_dump(mode="json"))


def packet_payload_sha256(packet: NativeCandidatePacketOutcome) -> str:
    return hash_canonical(packet.model_dump(mode="json"))


def canonical_packet_snapshot(packet: NativeCandidatePacketOutcome) -> dict[str, Any]:
    return deepcopy(packet.model_dump(mode="json"))


__all__ = [
    "GENERATION_CONTRACT_VERSION",
    "INVENTORY_SENTINEL_CAP",
    "INVENTORY_VERSION",
    "MAX_ACCEPTED_CANDIDATES",
    "PACKET_MODELS",
    "PACKET_VERSION",
    "NativeBoundedGenerationError",
    "NativeCandidateDescriptor",
    "NativeCandidateInventory",
    "NativeCandidatePacket",
    "NativeCandidatePacketOutcome",
    "NativeCandidateUnableToComplete",
    "assemble_candidate_packets",
    "assert_bounded_generation_schema",
    "canonical_packet_snapshot",
    "inventory_generation_schema",
    "packet_generation_schema",
    "packet_payload_sha256",
    "validate_inventory_for_row",
    "validate_packet_for_candidate",
]
