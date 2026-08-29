"""Fail-closed assembly of grounded passage packets into typed effects.

The grounding-v2 receipt proves exact source support, but it is intentionally
smaller than the legacy ``NativeCandidatePacket``.  This module joins one passage
candidate, its frozen projection, the frozen protocol orientation, and one replayed
grounding receipt.  It authorizes a new typed effect only when the complete numeric
core and required identities are present.

The legacy packet defaults ``reported_significance=not_reported``,
``equivalence_conclusion=not_tested``, and ``moderators=[]``.  Those values can be
mistaken for paper-level absence.  Assembly v2 therefore records explicit coverage
states and withholds a legacy packet until those optional scientific families have
their own affirmative grounding contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, StrictStr, TypeAdapter, field_validator, model_validator

from literature_multiverse.effects import EffectFormat
from literature_multiverse.evidence_graph import ArmRole
from literature_multiverse.lineage import hash_canonical
from literature_multiverse.metasyn_candidate_inventory_v2 import (
    MetaSynPassageCandidateV2,
)
from literature_multiverse.metasyn_extraction_inputs_v2 import (
    MetaSynExtractionQuestionSurfaceV2,
)
from literature_multiverse.metasyn_projection_v2 import FrozenMetaSynProjectionV2
from literature_multiverse.models import SHA256_RE, ContractModel
from literature_multiverse.native_bounded_generation import (
    BinaryGroupEffect,
    BoundedTimepoint,
    ContinuousGroupEffect,
    DirectConfidenceIntervalEffect,
    DirectStandardErrorEffect,
    DirectVarianceEffect,
    EffectKind,
)
from literature_multiverse.native_packet_grounding_v2 import (
    NativePacketGroundingV2Error,
    PacketGroundingAbstentionReceiptV2,
    PacketGroundingCompletedReceiptV2,
    PacketGroundingReceiptV2,
    PacketPassageCandidateBindingV2,
    validate_passage_packet_grounding_receipt_v2,
)
from literature_multiverse.native_question_projection import (
    NativeQuestionProjectionError,
    QuestionProjectionSpecV1,
    freeze_question_projection_spec,
)

PACKET_ASSEMBLY_V2_VERSION = "native-packet-assembly-v2"
GROUNDED_TYPED_EFFECT_V2_VERSION = "grounded-typed-effect-v2"
ASSEMBLY_COMPLETED_V2_VERSION = "native-packet-assembly-completed-v2"
ASSEMBLY_ABSTENTION_V2_VERSION = "native-packet-assembly-abstention-v2"
STABLE_KEY_POLICY_V2_VERSION = "native-packet-assembly-stable-key-policy-v2"
ANALYSIS_POLICY_V2_VERSION = "native-packet-assembly-analysis-policy-v2"
PROTOCOL_ORIENTATION_V2_VERSION = "native-packet-assembly-protocol-orientation-v2"

MAX_ASSEMBLY_BLOCKERS = 32
MAX_MISSING_FIELD_PATHS = 32

_EFFECT_FIELDS = frozenset(
    {
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
    }
)
_REQUIRED_EFFECT_FIELDS: dict[EffectKind, frozenset[str]] = {
    "direct_standard_error": frozenset(
        {"effect.estimate", "effect.standard_error"}
    ),
    "direct_variance": frozenset({"effect.estimate", "effect.variance"}),
    "direct_confidence_interval": frozenset(
        {
            "effect.estimate",
            "effect.ci_lower",
            "effect.ci_upper",
            "effect.ci_level",
        }
    ),
    "continuous_group_statistics": frozenset(
        {
            "effect.treatment_mean",
            "effect.treatment_sd",
            "effect.treatment_n",
            "effect.control_mean",
            "effect.control_sd",
            "effect.control_n",
        }
    ),
    "binary_group_statistics": frozenset(
        {
            "effect.treatment_events",
            "effect.treatment_total",
            "effect.control_events",
            "effect.control_total",
        }
    ),
}


class NativePacketAssemblyV2Error(ValueError):
    """An input or saved assembly artifact cannot be replayed safely."""


class _FrozenExactModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,
        str_strip_whitespace=False,
        use_enum_values=False,
    )


Sha256 = Annotated[StrictStr, Field(pattern=SHA256_RE.pattern)]
DecimalLexeme = Annotated[StrictStr, Field(min_length=1, max_length=32)]
UnsignedIntegerLexeme = Annotated[StrictStr, Field(min_length=1, max_length=10)]
BoundedIdentity = Annotated[StrictStr, Field(min_length=1, max_length=512)]


def _validate_self_hash(model: _FrozenExactModel, field_name: str) -> None:
    payload = model.model_dump(mode="json", exclude={field_name})
    if getattr(model, field_name) != hash_canonical(payload):
        raise ValueError(f"packet_assembly_v2_self_hash_mismatch:{field_name}")


def _validate_effect_unit(*, effect_format: EffectFormat, unit: str | None) -> None:
    if (effect_format == EffectFormat.MEAN_DIFFERENCE) != (unit is not None):
        raise ValueError("packet_assembly_v2_effect_format_unit_shape_mismatch")


class PacketAssemblyAnalysisPolicyV2(_FrozenExactModel):
    """Caller-frozen scale and correction choices for reported group statistics."""

    analysis_policy_version: Literal[
        "native-packet-assembly-analysis-policy-v2"
    ] = ANALYSIS_POLICY_V2_VERSION
    continuous_group_effect_format: Literal[
        EffectFormat.MEAN_DIFFERENCE,
        EffectFormat.HEDGES_G,
    ]
    continuous_computation_policy: Literal[
        "difference_and_independent_group_sampling_variance",
        "pooled_sd_cohens_d_then_exact_gamma_hedges_correction",
    ]
    binary_group_effect_format: Literal[
        EffectFormat.ODDS_RATIO,
        EffectFormat.RISK_RATIO,
    ]
    binary_computation_policy: Literal[
        "log_ratio_from_2x2_counts_haldane_anscombe_0.5_on_required_zero_cells"
    ] = "log_ratio_from_2x2_counts_haldane_anscombe_0.5_on_required_zero_cells"
    analysis_policy_sha256: Sha256

    @model_validator(mode="after")
    def validate_policy(self) -> PacketAssemblyAnalysisPolicyV2:
        expected_continuous = (
            "difference_and_independent_group_sampling_variance"
            if self.continuous_group_effect_format == EffectFormat.MEAN_DIFFERENCE
            else "pooled_sd_cohens_d_then_exact_gamma_hedges_correction"
        )
        if self.continuous_computation_policy != expected_continuous:
            raise ValueError("packet_assembly_v2_continuous_policy_mismatch")
        _validate_self_hash(self, "analysis_policy_sha256")
        return self


def freeze_packet_assembly_analysis_policy_v2(
    *,
    continuous_group_effect_format: Literal[
        EffectFormat.MEAN_DIFFERENCE,
        EffectFormat.HEDGES_G,
    ],
    binary_group_effect_format: Literal[
        EffectFormat.ODDS_RATIO,
        EffectFormat.RISK_RATIO,
    ],
) -> PacketAssemblyAnalysisPolicyV2:
    payload = {
        "analysis_policy_version": ANALYSIS_POLICY_V2_VERSION,
        "continuous_group_effect_format": continuous_group_effect_format,
        "continuous_computation_policy": (
            "difference_and_independent_group_sampling_variance"
            if continuous_group_effect_format == EffectFormat.MEAN_DIFFERENCE
            else "pooled_sd_cohens_d_then_exact_gamma_hedges_correction"
        ),
        "binary_group_effect_format": binary_group_effect_format,
        "binary_computation_policy": (
            "log_ratio_from_2x2_counts_haldane_anscombe_0.5_on_required_zero_cells"
        ),
    }
    return PacketAssemblyAnalysisPolicyV2.model_validate(
        {**payload, "analysis_policy_sha256": hash_canonical(payload)}
    )


class PacketAssemblyProtocolOrientationV2(_FrozenExactModel):
    """Concrete arm roles derived only from a frozen MetaSyn question surface."""

    orientation_version: Literal[
        "native-packet-assembly-protocol-orientation-v2"
    ] = PROTOCOL_ORIENTATION_V2_VERSION
    question_surface_sha256: Sha256
    question_surface_question_spec_sha256: Sha256
    protocol_question_spec_sha256: Sha256
    protocol_projection_spec_sha256: Sha256
    question_id: BoundedIdentity
    frozen_treatment_role: Literal["intervention_or_exposure"]
    frozen_comparator_role: Literal["comparator"]
    relation_kind: Literal["intervention", "exposure"]
    treatment_arm_role: Literal[ArmRole.INTERVENTION, ArmRole.EXPOSURE]
    comparator_arm_role: Literal[ArmRole.COMPARATOR] = ArmRole.COMPARATOR
    orientation_sha256: Sha256

    @model_validator(mode="after")
    def validate_orientation(self) -> PacketAssemblyProtocolOrientationV2:
        expected_treatment = (
            ArmRole.INTERVENTION
            if self.relation_kind == "intervention"
            else ArmRole.EXPOSURE
        )
        if self.treatment_arm_role is not expected_treatment:
            raise ValueError("packet_assembly_v2_relation_kind_role_mismatch")
        _validate_self_hash(self, "orientation_sha256")
        return self


def replay_metasyn_question_projection_spec_v2(
    *,
    question_surface: MetaSynExtractionQuestionSurfaceV2 | Mapping[str, Any],
) -> QuestionProjectionSpecV1:
    """Rebuild the exact generic-role protocol exposed by extraction input v2."""

    try:
        surface = MetaSynExtractionQuestionSurfaceV2.model_validate(
            question_surface.model_dump(mode="json")
            if isinstance(question_surface, MetaSynExtractionQuestionSurfaceV2)
            else question_surface
        )
        outcome_texts = [
            surface.allowed_outcome_text_by_id[outcome_id]
            for outcome_id in surface.allowed_outcome_ids
        ]
        positive_by_text = {
            surface.allowed_outcome_text_by_id[outcome_id]: (
                surface.raw_positive_direction_meaning_by_outcome_id[outcome_id]
            )
            for outcome_id in surface.allowed_outcome_ids
        }
        protocol = freeze_question_projection_spec(
            question_id=surface.question_id,
            population=surface.population,
            intervention_or_exposure=surface.intervention_or_exposure,
            comparison=surface.comparison,
            outcome_texts=outcome_texts,
            treatment_role=surface.treatment_role,
            comparator_role=surface.comparator_role,
            contrast_estimand=surface.contrast_estimand,
            positive_direction_means_by_outcome=positive_by_text,
        )
    except (KeyError, NativeQuestionProjectionError, ValueError) as exc:
        raise NativePacketAssemblyV2Error(
            "packet_assembly_v2_question_surface_protocol_replay_invalid"
        ) from exc
    replayed_outcomes = {
        item.outcome_id: item.outcome_text
        for item in protocol.question_fields.outcomes
    }
    replayed_directions = {
        item.outcome_id: item.positive_direction_means
        for item in protocol.question_fields.outcomes
    }
    if (
        replayed_outcomes != surface.allowed_outcome_text_by_id
        or replayed_directions
        != surface.raw_positive_direction_meaning_by_outcome_id
    ):
        raise NativePacketAssemblyV2Error(
            "packet_assembly_v2_question_surface_protocol_replay_mismatch"
        )
    return protocol


def freeze_packet_assembly_protocol_orientation_v2(
    *,
    question_surface: MetaSynExtractionQuestionSurfaceV2 | Mapping[str, Any],
) -> PacketAssemblyProtocolOrientationV2:
    """Bind generic frozen protocol roles to the surface's exact relation kind."""

    try:
        surface = MetaSynExtractionQuestionSurfaceV2.model_validate(
            question_surface.model_dump(mode="json")
            if isinstance(question_surface, MetaSynExtractionQuestionSurfaceV2)
            else question_surface
        )
    except ValueError as exc:
        raise NativePacketAssemblyV2Error(
            "packet_assembly_v2_question_surface_invalid"
        ) from exc
    protocol = replay_metasyn_question_projection_spec_v2(
        question_surface=surface
    )
    payload = {
        "orientation_version": PROTOCOL_ORIENTATION_V2_VERSION,
        "question_surface_sha256": surface.question_surface_sha256,
        "question_surface_question_spec_sha256": surface.question_spec_sha256,
        "protocol_question_spec_sha256": protocol.question_spec_sha256,
        "protocol_projection_spec_sha256": protocol.projection_spec_sha256,
        "question_id": surface.question_id,
        "frozen_treatment_role": surface.treatment_role,
        "frozen_comparator_role": surface.comparator_role,
        "relation_kind": surface.relation_kind,
        "treatment_arm_role": (
            ArmRole.INTERVENTION
            if surface.relation_kind == "intervention"
            else ArmRole.EXPOSURE
        ),
        "comparator_arm_role": ArmRole.COMPARATOR,
    }
    return PacketAssemblyProtocolOrientationV2.model_validate(
        {**payload, "orientation_sha256": hash_canonical(payload)}
    )


class GroundedDirectStandardErrorEffectV2(_FrozenExactModel):
    effect_kind: Literal["direct_standard_error"] = "direct_standard_error"
    effect_format: EffectFormat
    estimate: DecimalLexeme
    standard_error: DecimalLexeme
    unit: Annotated[StrictStr, Field(min_length=1, max_length=64)] | None

    @model_validator(mode="after")
    def validate_effect(self) -> GroundedDirectStandardErrorEffectV2:
        _validate_effect_unit(effect_format=self.effect_format, unit=self.unit)
        DirectStandardErrorEffect(
            effect_format=self.effect_format,
            estimate=self.estimate,
            standard_error=self.standard_error,
            unit=self.unit,
        )
        return self


class GroundedDirectVarianceEffectV2(_FrozenExactModel):
    effect_kind: Literal["direct_variance"] = "direct_variance"
    effect_format: EffectFormat
    estimate: DecimalLexeme
    variance: DecimalLexeme
    unit: Annotated[StrictStr, Field(min_length=1, max_length=64)] | None

    @model_validator(mode="after")
    def validate_effect(self) -> GroundedDirectVarianceEffectV2:
        _validate_effect_unit(effect_format=self.effect_format, unit=self.unit)
        DirectVarianceEffect(
            effect_format=self.effect_format,
            estimate=self.estimate,
            variance=self.variance,
            unit=self.unit,
        )
        return self


class GroundedDirectConfidenceIntervalEffectV2(_FrozenExactModel):
    effect_kind: Literal["direct_confidence_interval"] = (
        "direct_confidence_interval"
    )
    effect_format: EffectFormat
    estimate: DecimalLexeme
    ci_lower: DecimalLexeme
    ci_upper: DecimalLexeme
    ci_level: DecimalLexeme
    unit: Annotated[StrictStr, Field(min_length=1, max_length=64)] | None

    @model_validator(mode="after")
    def validate_effect(self) -> GroundedDirectConfidenceIntervalEffectV2:
        _validate_effect_unit(effect_format=self.effect_format, unit=self.unit)
        DirectConfidenceIntervalEffect(
            effect_format=self.effect_format,
            estimate=self.estimate,
            ci_lower=self.ci_lower,
            ci_upper=self.ci_upper,
            ci_level=self.ci_level,
            unit=self.unit,
        )
        return self


class GroundedContinuousGroupEffectV2(_FrozenExactModel):
    effect_kind: Literal["continuous_group_statistics"] = (
        "continuous_group_statistics"
    )
    effect_format: EffectFormat
    treatment_mean: DecimalLexeme
    treatment_sd: DecimalLexeme
    treatment_n: UnsignedIntegerLexeme
    control_mean: DecimalLexeme
    control_sd: DecimalLexeme
    control_n: UnsignedIntegerLexeme
    unit: Annotated[StrictStr, Field(min_length=1, max_length=64)] | None
    source_measurement_unit: Annotated[
        StrictStr, Field(min_length=1, max_length=64)
    ] | None

    @model_validator(mode="after")
    def validate_effect(self) -> GroundedContinuousGroupEffectV2:
        _validate_effect_unit(effect_format=self.effect_format, unit=self.unit)
        if (
            self.effect_format == EffectFormat.MEAN_DIFFERENCE
            and self.source_measurement_unit != self.unit
        ):
            raise ValueError("packet_assembly_v2_mean_difference_source_unit_mismatch")
        ContinuousGroupEffect(
            effect_format=self.effect_format,
            treatment_mean=self.treatment_mean,
            treatment_sd=self.treatment_sd,
            treatment_n=self.treatment_n,
            control_mean=self.control_mean,
            control_sd=self.control_sd,
            control_n=self.control_n,
            unit=self.unit,
        )
        return self


class GroundedBinaryGroupEffectV2(_FrozenExactModel):
    effect_kind: Literal["binary_group_statistics"] = "binary_group_statistics"
    effect_format: EffectFormat
    treatment_events: UnsignedIntegerLexeme
    treatment_total: UnsignedIntegerLexeme
    control_events: UnsignedIntegerLexeme
    control_total: UnsignedIntegerLexeme

    @model_validator(mode="after")
    def validate_effect(self) -> GroundedBinaryGroupEffectV2:
        BinaryGroupEffect(
            effect_format=self.effect_format,
            treatment_events=self.treatment_events,
            treatment_total=self.treatment_total,
            control_events=self.control_events,
            control_total=self.control_total,
        )
        return self


type GroundedEffectCoreV2 = Annotated[
    GroundedDirectStandardErrorEffectV2
    | GroundedDirectVarianceEffectV2
    | GroundedDirectConfidenceIntervalEffectV2
    | GroundedContinuousGroupEffectV2
    | GroundedBinaryGroupEffectV2,
    Field(discriminator="effect_kind"),
]


class GroundedTypedEffectV2(_FrozenExactModel):
    typed_effect_version: Literal["grounded-typed-effect-v2"] = (
        GROUNDED_TYPED_EFFECT_V2_VERSION
    )
    candidate_descriptor_sha256: Sha256
    candidate_binding_sha256: Sha256
    projection_sha256: Sha256
    question_spec_sha256: Sha256
    publication_source_identity_sha256: Sha256
    analysis_policy_sha256: Sha256
    protocol_orientation_sha256: Sha256
    study_key: Annotated[StrictStr, Field(pattern=r"^study-[0-9a-f]{40}$")]
    cohort_key: Annotated[StrictStr, Field(pattern=r"^cohort-[0-9a-f]{40}$")]
    treatment_arm_key: Annotated[
        StrictStr, Field(pattern=r"^arm-treatment-[0-9a-f]{40}$")
    ]
    comparator_arm_key: Annotated[
        StrictStr, Field(pattern=r"^arm-comparator-[0-9a-f]{40}$")
    ]
    contrast_key: Annotated[StrictStr, Field(pattern=r"^contrast-[0-9a-f]{40}$")]
    finding_key: Annotated[StrictStr, Field(pattern=r"^finding-[0-9a-f]{40}$")]
    study_source_label: BoundedIdentity
    cohort_source_labels: Annotated[list[BoundedIdentity], Field(min_length=1)]
    treatment_arm_label: BoundedIdentity
    treatment_arm_role: Literal[ArmRole.INTERVENTION, ArmRole.EXPOSURE]
    comparator_arm_label: BoundedIdentity
    comparator_arm_role: Literal[ArmRole.COMPARATOR, ArmRole.CONTROL]
    contrast_label: BoundedIdentity
    contrast_estimand: BoundedIdentity
    positive_direction_means: BoundedIdentity
    positive_direction_coverage: Literal[
        "prespecified_in_frozen_protocol",
        "not_prespecified_in_frozen_protocol",
    ]
    canonical_outcome_id: Annotated[StrictStr, Field(min_length=1, max_length=64)]
    outcome_concept_quote: Annotated[StrictStr, Field(min_length=1, max_length=256)]
    timepoint: BoundedTimepoint
    effect: GroundedEffectCoreV2
    extraction_method: Literal[
        "reported",
        "computed_from_reported_statistics",
    ]
    reported_p_value: DecimalLexeme | None
    reported_significance_coverage: Literal[
        "not_extracted_from_selected_support",
        "p_value_only_extracted_conclusion_not_extracted",
    ]
    equivalence_margin: DecimalLexeme | None
    equivalence_coverage: Literal[
        "not_extracted_from_selected_support",
        "margin_only_extracted_conclusion_not_extracted",
    ]
    moderator_coverage: Literal["not_extracted_from_selected_support"] = (
        "not_extracted_from_selected_support"
    )
    analysis_population: BoundedIdentity | None
    analysis_population_coverage: Literal[
        "grounded_exact_text",
        "not_extracted_from_selected_support",
    ]
    evidence_receipt_sha256: Sha256
    effect_format_receipt_sha256: Sha256 | None
    numeric_receipt_sha256s: Annotated[list[Sha256], Field(min_length=1)]
    identity_receipt_sha256s: list[Sha256]
    authorizes_typed_effect: Literal[True] = True
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    typed_effect_sha256: Sha256

    @field_validator(
        "cohort_source_labels",
        "numeric_receipt_sha256s",
        "identity_receipt_sha256s",
    )
    @classmethod
    def validate_sorted_unique(cls, value: list[Any], info: Any) -> list[Any]:
        if value != sorted(set(value)):
            raise ValueError(
                f"packet_assembly_v2_values_not_sorted_unique:{info.field_name}"
            )
        return value

    @model_validator(mode="after")
    def validate_typed_effect(self) -> GroundedTypedEffectV2:
        if self.treatment_arm_label == self.comparator_arm_label:
            raise ValueError("packet_assembly_v2_arm_labels_not_distinct")
        expected_significance = (
            "p_value_only_extracted_conclusion_not_extracted"
            if self.reported_p_value is not None
            else "not_extracted_from_selected_support"
        )
        if self.reported_significance_coverage != expected_significance:
            raise ValueError("packet_assembly_v2_significance_coverage_mismatch")
        expected_equivalence = (
            "margin_only_extracted_conclusion_not_extracted"
            if self.equivalence_margin is not None
            else "not_extracted_from_selected_support"
        )
        if self.equivalence_coverage != expected_equivalence:
            raise ValueError("packet_assembly_v2_equivalence_coverage_mismatch")
        expected_population = (
            "grounded_exact_text"
            if self.analysis_population is not None
            else "not_extracted_from_selected_support"
        )
        if self.analysis_population_coverage != expected_population:
            raise ValueError("packet_assembly_v2_population_coverage_mismatch")
        is_direct = self.effect.effect_kind in {
            "direct_standard_error",
            "direct_variance",
            "direct_confidence_interval",
        }
        if is_direct != (self.extraction_method == "reported") or is_direct != (
            self.effect_format_receipt_sha256 is not None
        ):
            raise ValueError("packet_assembly_v2_extraction_method_shape_mismatch")
        _validate_self_hash(self, "typed_effect_sha256")
        return self


AssemblyBlocker = Literal[
    "candidate_outcome_concept_not_bound_to_protocol",
    "candidate_outcome_not_in_protocol",
    "effect_contract_invalid",
    "effect_numeric_field_set_incompatible",
    "grounding_abstained",
    "protocol_orientation_binding_mismatch",
    "protocol_orientation_unsupported",
    "question_spec_hash_mismatch",
    "required_identity_ambiguous",
    "required_identity_missing",
]


class NativePacketAssemblyCompletedV2(_FrozenExactModel):
    assembly_version: Literal["native-packet-assembly-v2"] = PACKET_ASSEMBLY_V2_VERSION
    receipt_version: Literal["native-packet-assembly-completed-v2"] = (
        ASSEMBLY_COMPLETED_V2_VERSION
    )
    status: Literal["typed_effect_completed"] = "typed_effect_completed"
    candidate_descriptor_sha256: Sha256
    projection_sha256: Sha256
    protocol_projection_spec_sha256: Sha256
    protocol_orientation: PacketAssemblyProtocolOrientationV2
    protocol_orientation_sha256: Sha256
    analysis_policy: PacketAssemblyAnalysisPolicyV2
    analysis_policy_sha256: Sha256
    grounding_receipt_sha256: Sha256
    typed_effect: GroundedTypedEffectV2
    typed_effect_sha256: Sha256
    native_candidate_packet_status: Literal[
        "withheld_optional_scientific_coverage_not_extracted"
    ] = "withheld_optional_scientific_coverage_not_extracted"
    native_candidate_packet: None = None
    native_candidate_packet_withholding_codes: Annotated[
        list[
            Literal[
                "equivalence_conclusion_not_extracted",
                "moderators_not_extracted",
                "reported_significance_not_extracted",
            ]
        ],
        Field(min_length=3, max_length=3),
    ]
    authorizes_typed_effect: Literal[True] = True
    authorizes_native_candidate_packet: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    assembly_receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> NativePacketAssemblyCompletedV2:
        if (
            self.protocol_orientation_sha256
            != self.protocol_orientation.orientation_sha256
        ):
            raise ValueError(
                "packet_assembly_v2_protocol_orientation_hash_alias_mismatch"
            )
        if self.analysis_policy_sha256 != self.analysis_policy.analysis_policy_sha256:
            raise ValueError("packet_assembly_v2_analysis_policy_hash_alias_mismatch")
        if self.typed_effect_sha256 != self.typed_effect.typed_effect_sha256:
            raise ValueError("packet_assembly_v2_typed_effect_hash_alias_mismatch")
        if self.candidate_descriptor_sha256 != (
            self.typed_effect.candidate_descriptor_sha256
        ):
            raise ValueError(
                "packet_assembly_v2_candidate_descriptor_hash_alias_mismatch"
            )
        if self.projection_sha256 != self.typed_effect.projection_sha256:
            raise ValueError("packet_assembly_v2_projection_hash_alias_mismatch")
        if self.analysis_policy_sha256 != self.typed_effect.analysis_policy_sha256:
            raise ValueError(
                "packet_assembly_v2_typed_effect_analysis_policy_hash_mismatch"
            )
        if (
            self.protocol_orientation_sha256
            != self.typed_effect.protocol_orientation_sha256
        ):
            raise ValueError(
                "packet_assembly_v2_typed_effect_protocol_orientation_hash_mismatch"
            )
        expected_codes = [
            "equivalence_conclusion_not_extracted",
            "moderators_not_extracted",
            "reported_significance_not_extracted",
        ]
        if self.native_candidate_packet_withholding_codes != expected_codes:
            raise ValueError("packet_assembly_v2_packet_withholding_codes_mismatch")
        _validate_self_hash(self, "assembly_receipt_sha256")
        return self


class NativePacketAssemblyAbstentionV2(_FrozenExactModel):
    assembly_version: Literal["native-packet-assembly-v2"] = PACKET_ASSEMBLY_V2_VERSION
    receipt_version: Literal["native-packet-assembly-abstention-v2"] = (
        ASSEMBLY_ABSTENTION_V2_VERSION
    )
    status: Literal["unable_to_assemble"] = "unable_to_assemble"
    candidate_descriptor_sha256: Sha256
    projection_sha256: Sha256
    protocol_projection_spec_sha256: Sha256
    protocol_orientation: PacketAssemblyProtocolOrientationV2
    protocol_orientation_sha256: Sha256
    analysis_policy_sha256: Sha256
    grounding_receipt_sha256: Sha256
    primary_blocker: AssemblyBlocker
    blocker_codes: Annotated[
        list[AssemblyBlocker], Field(min_length=1, max_length=MAX_ASSEMBLY_BLOCKERS)
    ]
    missing_field_paths: Annotated[
        list[Annotated[StrictStr, Field(min_length=1, max_length=128)]],
        Field(max_length=MAX_MISSING_FIELD_PATHS),
    ]
    authorizes_typed_effect: Literal[False] = False
    authorizes_native_candidate_packet: Literal[False] = False
    synthesis_input_authority: Literal[False] = False
    claim_release_authority: Literal[False] = False
    assembly_receipt_sha256: Sha256

    @field_validator("blocker_codes", "missing_field_paths")
    @classmethod
    def validate_sorted_unique(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError(
                f"packet_assembly_v2_values_not_sorted_unique:{info.field_name}"
            )
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> NativePacketAssemblyAbstentionV2:
        if (
            self.protocol_orientation_sha256
            != self.protocol_orientation.orientation_sha256
        ):
            raise ValueError(
                "packet_assembly_v2_protocol_orientation_hash_alias_mismatch"
            )
        if self.primary_blocker not in self.blocker_codes:
            raise ValueError("packet_assembly_v2_primary_blocker_not_listed")
        _validate_self_hash(self, "assembly_receipt_sha256")
        return self


type NativePacketAssemblyOutcomeV2 = Annotated[
    NativePacketAssemblyCompletedV2 | NativePacketAssemblyAbstentionV2,
    Field(discriminator="status"),
]
_ASSEMBLY_ADAPTER = TypeAdapter(NativePacketAssemblyOutcomeV2)
_GROUNDING_ADAPTER = TypeAdapter(PacketGroundingReceiptV2)


def _stable_key(*, prefix: str, identity_payload: Mapping[str, Any]) -> str:
    digest = hash_canonical(
        {
            "key_policy_version": STABLE_KEY_POLICY_V2_VERSION,
            "entity_kind": prefix,
            "identity": dict(identity_payload),
        }
    )
    return f"{prefix}-{digest[:40]}"


def _abstention(
    *,
    candidate: MetaSynPassageCandidateV2,
    projection: FrozenMetaSynProjectionV2,
    protocol: QuestionProjectionSpecV1,
    protocol_orientation: PacketAssemblyProtocolOrientationV2,
    analysis_policy: PacketAssemblyAnalysisPolicyV2,
    grounding_receipt_sha256: str,
    blocker_codes: set[AssemblyBlocker],
    missing_field_paths: set[str] | None = None,
) -> NativePacketAssemblyAbstentionV2:
    ordered_blockers = sorted(blocker_codes)
    payload: dict[str, Any] = {
        "assembly_version": PACKET_ASSEMBLY_V2_VERSION,
        "receipt_version": ASSEMBLY_ABSTENTION_V2_VERSION,
        "status": "unable_to_assemble",
        "candidate_descriptor_sha256": candidate.descriptor_sha256,
        "projection_sha256": projection.projection_sha256,
        "protocol_projection_spec_sha256": protocol.projection_spec_sha256,
        "protocol_orientation": protocol_orientation,
        "protocol_orientation_sha256": protocol_orientation.orientation_sha256,
        "analysis_policy_sha256": analysis_policy.analysis_policy_sha256,
        "grounding_receipt_sha256": grounding_receipt_sha256,
        "primary_blocker": ordered_blockers[0],
        "blocker_codes": ordered_blockers,
        "missing_field_paths": sorted(missing_field_paths or set()),
        "authorizes_typed_effect": False,
        "authorizes_native_candidate_packet": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return NativePacketAssemblyAbstentionV2.model_validate(
        {**payload, "assembly_receipt_sha256": hash_canonical(payload)}
    )


def _identity_map(receipt: PacketGroundingCompletedReceiptV2) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for item in receipt.identity_receipts:
        output.setdefault(item.field_path, []).append(item.verbatim_identity_text)
    return {key: sorted(set(values)) for key, values in sorted(output.items())}


def _numeric_map(receipt: PacketGroundingCompletedReceiptV2) -> dict[str, str]:
    return {
        item.normalization_receipt.field_path: (
            item.normalization_receipt.normalized_numeric_lexeme
        )
        for item in receipt.numeric_receipts
    }


def _build_timepoint(
    *, receipt: PacketGroundingCompletedReceiptV2, numeric: Mapping[str, str]
) -> BoundedTimepoint:
    reported = receipt.model_outcome.timepoint
    payload: dict[str, Any] = {
        "kind": reported.kind,
        "value": numeric.get("finding.timepoint.value"),
        "lower": numeric.get("finding.timepoint.lower"),
        "upper": numeric.get("finding.timepoint.upper"),
        "unit": getattr(reported, "unit", None),
        "anchor": getattr(reported, "anchor", None),
        "raw_label": getattr(reported, "raw_label", None),
    }
    return BoundedTimepoint.model_validate(payload)


def _build_effect_core(
    *,
    effect_kind: EffectKind,
    reported_effect_format: EffectFormat | None,
    source_unit: str | None,
    analysis_policy: PacketAssemblyAnalysisPolicyV2,
    numeric: Mapping[str, str],
) -> GroundedEffectCoreV2:
    if effect_kind == "direct_standard_error":
        if reported_effect_format is None:
            raise ValueError("packet_assembly_v2_direct_effect_format_missing")
        return GroundedDirectStandardErrorEffectV2(
            effect_format=reported_effect_format,
            estimate=numeric["effect.estimate"],
            standard_error=numeric["effect.standard_error"],
            unit=source_unit,
        )
    if effect_kind == "direct_variance":
        if reported_effect_format is None:
            raise ValueError("packet_assembly_v2_direct_effect_format_missing")
        return GroundedDirectVarianceEffectV2(
            effect_format=reported_effect_format,
            estimate=numeric["effect.estimate"],
            variance=numeric["effect.variance"],
            unit=source_unit,
        )
    if effect_kind == "direct_confidence_interval":
        if reported_effect_format is None:
            raise ValueError("packet_assembly_v2_direct_effect_format_missing")
        return GroundedDirectConfidenceIntervalEffectV2(
            effect_format=reported_effect_format,
            estimate=numeric["effect.estimate"],
            ci_lower=numeric["effect.ci_lower"],
            ci_upper=numeric["effect.ci_upper"],
            ci_level=numeric["effect.ci_level"],
            unit=source_unit,
        )
    if effect_kind == "continuous_group_statistics":
        effect_format = analysis_policy.continuous_group_effect_format
        result_unit = (
            source_unit
            if effect_format == EffectFormat.MEAN_DIFFERENCE
            else None
        )
        return GroundedContinuousGroupEffectV2(
            effect_format=effect_format,
            treatment_mean=numeric["effect.treatment_mean"],
            treatment_sd=numeric["effect.treatment_sd"],
            treatment_n=numeric["effect.treatment_n"],
            control_mean=numeric["effect.control_mean"],
            control_sd=numeric["effect.control_sd"],
            control_n=numeric["effect.control_n"],
            unit=result_unit,
            source_measurement_unit=source_unit,
        )
    return GroundedBinaryGroupEffectV2(
        effect_format=analysis_policy.binary_group_effect_format,
        treatment_events=numeric["effect.treatment_events"],
        treatment_total=numeric["effect.treatment_total"],
        control_events=numeric["effect.control_events"],
        control_total=numeric["effect.control_total"],
    )


def _canonical_grounding_receipt(
    *,
    value: PacketGroundingReceiptV2 | Mapping[str, Any],
    candidate: MetaSynPassageCandidateV2,
    projection: FrozenMetaSynProjectionV2,
) -> PacketGroundingReceiptV2:
    try:
        canonical = _GROUNDING_ADAPTER.validate_python(value)
        return validate_passage_packet_grounding_receipt_v2(
            receipt=canonical,
            model_outcome=canonical.model_outcome.model_dump(mode="json"),
            candidate=candidate,
            projection=projection,
        )
    except (ValueError, NativePacketGroundingV2Error) as exc:
        raise NativePacketAssemblyV2Error(
            "packet_assembly_v2_grounding_receipt_replay_invalid"
        ) from exc


def assemble_native_packet_v2(
    *,
    candidate: MetaSynPassageCandidateV2,
    projection: FrozenMetaSynProjectionV2,
    protocol: QuestionProjectionSpecV1,
    protocol_orientation: PacketAssemblyProtocolOrientationV2,
    analysis_policy: PacketAssemblyAnalysisPolicyV2,
    grounding_receipt: PacketGroundingReceiptV2 | Mapping[str, Any],
) -> NativePacketAssemblyOutcomeV2:
    """Assemble one replayed p2 grounding receipt without filling missing science."""

    candidate = MetaSynPassageCandidateV2.model_validate(
        candidate.model_dump(mode="json")
    )
    projection = FrozenMetaSynProjectionV2.model_validate(
        projection.model_dump(mode="json")
    )
    protocol = QuestionProjectionSpecV1.model_validate(protocol.model_dump(mode="json"))
    protocol_orientation = PacketAssemblyProtocolOrientationV2.model_validate(
        protocol_orientation.model_dump(mode="json")
    )
    analysis_policy = PacketAssemblyAnalysisPolicyV2.model_validate(
        analysis_policy.model_dump(mode="json")
    )
    receipt = _canonical_grounding_receipt(
        value=grounding_receipt,
        candidate=candidate,
        projection=projection,
    )
    receipt_sha256 = receipt.receipt_sha256

    if projection.lineage_binding.question_spec_sha256 != protocol.question_spec_sha256:
        return _abstention(
            candidate=candidate,
            projection=projection,
            protocol=protocol,
            protocol_orientation=protocol_orientation,
            analysis_policy=analysis_policy,
            grounding_receipt_sha256=receipt_sha256,
            blocker_codes={"question_spec_hash_mismatch"},
        )
    if (
        protocol_orientation.protocol_question_spec_sha256
        != protocol.question_spec_sha256
        or protocol_orientation.protocol_projection_spec_sha256
        != protocol.projection_spec_sha256
        or protocol_orientation.question_id != protocol.question_id
        or protocol_orientation.frozen_treatment_role
        != protocol.question_fields.treatment_role
        or protocol_orientation.frozen_comparator_role
        != protocol.question_fields.comparator_role
    ):
        return _abstention(
            candidate=candidate,
            projection=projection,
            protocol=protocol,
            protocol_orientation=protocol_orientation,
            analysis_policy=analysis_policy,
            grounding_receipt_sha256=receipt_sha256,
            blocker_codes={"protocol_orientation_binding_mismatch"},
        )
    outcomes = {
        item.outcome_id: item for item in protocol.question_fields.outcomes
    }
    outcome = outcomes.get(candidate.canonical_outcome_id)
    if outcome is None:
        return _abstention(
            candidate=candidate,
            projection=projection,
            protocol=protocol,
            protocol_orientation=protocol_orientation,
            analysis_policy=analysis_policy,
            grounding_receipt_sha256=receipt_sha256,
            blocker_codes={"candidate_outcome_not_in_protocol"},
        )
    if candidate.outcome_concept_quote not in outcome.outcome_text:
        return _abstention(
            candidate=candidate,
            projection=projection,
            protocol=protocol,
            protocol_orientation=protocol_orientation,
            analysis_policy=analysis_policy,
            grounding_receipt_sha256=receipt_sha256,
            blocker_codes={"candidate_outcome_concept_not_bound_to_protocol"},
        )
    if isinstance(receipt, PacketGroundingAbstentionReceiptV2):
        return _abstention(
            candidate=candidate,
            projection=projection,
            protocol=protocol,
            protocol_orientation=protocol_orientation,
            analysis_policy=analysis_policy,
            grounding_receipt_sha256=receipt_sha256,
            blocker_codes={"grounding_abstained"},
        )
    if not isinstance(receipt.candidate_binding, PacketPassageCandidateBindingV2):
        raise NativePacketAssemblyV2Error(
            "packet_assembly_v2_passage_binding_required"
        )

    treatment_role = protocol_orientation.treatment_arm_role
    comparator_role = protocol_orientation.comparator_arm_role

    identities = _identity_map(receipt)
    required_singletons = (
        "study.source_label",
        "treatment_arm.label",
        "comparator_arm.label",
        "contrast.label",
    )
    missing = {path for path in required_singletons if not identities.get(path)}
    if not identities.get("cohort.source_label"):
        missing.add("cohort.source_label")
    ambiguous = {
        path for path in required_singletons if len(identities.get(path, [])) > 1
    }
    blockers: set[AssemblyBlocker] = set()
    if missing:
        blockers.add("required_identity_missing")
    if ambiguous:
        blockers.add("required_identity_ambiguous")
    if blockers:
        return _abstention(
            candidate=candidate,
            projection=projection,
            protocol=protocol,
            protocol_orientation=protocol_orientation,
            analysis_policy=analysis_policy,
            grounding_receipt_sha256=receipt_sha256,
            blocker_codes=blockers,
            missing_field_paths=missing | ambiguous,
        )

    numeric = _numeric_map(receipt)
    required_numeric = _REQUIRED_EFFECT_FIELDS[candidate.effect_kind]
    observed_effect = set(numeric).intersection(_EFFECT_FIELDS)
    numeric_missing = required_numeric - observed_effect
    numeric_extra = observed_effect - required_numeric
    if numeric_missing or numeric_extra:
        return _abstention(
            candidate=candidate,
            projection=projection,
            protocol=protocol,
            protocol_orientation=protocol_orientation,
            analysis_policy=analysis_policy,
            grounding_receipt_sha256=receipt_sha256,
            blocker_codes={"effect_numeric_field_set_incompatible"},
            missing_field_paths=numeric_missing | numeric_extra,
        )
    try:
        timepoint = _build_timepoint(receipt=receipt, numeric=numeric)
        effect = _build_effect_core(
            effect_kind=candidate.effect_kind,
            reported_effect_format=(
                receipt.effect_format_receipt.effect_format
                if receipt.effect_format_receipt is not None
                else None
            ),
            source_unit=receipt.model_outcome.effect_unit,
            analysis_policy=analysis_policy,
            numeric=numeric,
        )
    except (KeyError, ValueError):
        return _abstention(
            candidate=candidate,
            projection=projection,
            protocol=protocol,
            protocol_orientation=protocol_orientation,
            analysis_policy=analysis_policy,
            grounding_receipt_sha256=receipt_sha256,
            blocker_codes={"effect_contract_invalid"},
        )

    binding_sha = receipt.candidate_binding.binding_sha256
    positive_direction_is_prespecified = (
        outcome.positive_direction_means
        != "not_prespecified_from_question_metadata"
    )
    publication_source_identity = (
        projection.lineage_binding.row_source_identity_sha256
    )
    study_key = _stable_key(
        prefix="study",
        identity_payload={
            "publication_source_identity_sha256": publication_source_identity,
            "source_label": identities["study.source_label"][0],
            "registration_ids": identities.get("study.registration_id", []),
        },
    )
    cohort_key = _stable_key(
        prefix="cohort",
        identity_payload={
            "study_key": study_key,
            "source_labels": identities["cohort.source_label"],
            "registry_ids": identities.get("cohort.registry_id", []),
            "dataset_ids": identities.get("cohort.dataset_id", []),
        },
    )
    treatment_arm_key = _stable_key(
        prefix="arm-treatment",
        identity_payload={
            "cohort_key": cohort_key,
            "role": treatment_role,
            "label": identities["treatment_arm.label"][0],
        },
    )
    comparator_arm_key = _stable_key(
        prefix="arm-comparator",
        identity_payload={
            "cohort_key": cohort_key,
            "role": comparator_role,
            "label": identities["comparator_arm.label"][0],
        },
    )
    contrast_key = _stable_key(
        prefix="contrast",
        identity_payload={
            "treatment_arm_key": treatment_arm_key,
            "comparator_arm_key": comparator_arm_key,
            "label": identities["contrast.label"][0],
            "estimand": protocol.question_fields.contrast_estimand,
        },
    )
    typed_payload: dict[str, Any] = {
        "typed_effect_version": GROUNDED_TYPED_EFFECT_V2_VERSION,
        "candidate_descriptor_sha256": candidate.descriptor_sha256,
        "candidate_binding_sha256": binding_sha,
        "projection_sha256": projection.projection_sha256,
        "question_spec_sha256": protocol.question_spec_sha256,
        "publication_source_identity_sha256": publication_source_identity,
        "analysis_policy_sha256": analysis_policy.analysis_policy_sha256,
        "protocol_orientation_sha256": protocol_orientation.orientation_sha256,
        "study_key": study_key,
        "cohort_key": cohort_key,
        "treatment_arm_key": treatment_arm_key,
        "comparator_arm_key": comparator_arm_key,
        "contrast_key": contrast_key,
        "finding_key": _stable_key(
            prefix="finding",
            identity_payload={
                "candidate_binding_sha256": binding_sha,
                "contrast_key": contrast_key,
            },
        ),
        "study_source_label": identities["study.source_label"][0],
        "cohort_source_labels": identities["cohort.source_label"],
        "treatment_arm_label": identities["treatment_arm.label"][0],
        "treatment_arm_role": treatment_role,
        "comparator_arm_label": identities["comparator_arm.label"][0],
        "comparator_arm_role": comparator_role,
        "contrast_label": identities["contrast.label"][0],
        "contrast_estimand": protocol.question_fields.contrast_estimand,
        "positive_direction_means": outcome.positive_direction_means,
        "positive_direction_coverage": (
            "prespecified_in_frozen_protocol"
            if positive_direction_is_prespecified
            else "not_prespecified_in_frozen_protocol"
        ),
        "canonical_outcome_id": candidate.canonical_outcome_id,
        "outcome_concept_quote": candidate.outcome_concept_quote,
        "timepoint": timepoint,
        "effect": effect,
        "extraction_method": (
            "reported"
            if candidate.effect_kind
            in {
                "direct_standard_error",
                "direct_variance",
                "direct_confidence_interval",
            }
            else "computed_from_reported_statistics"
        ),
        "reported_p_value": numeric.get("effect.reported_p_value"),
        "reported_significance_coverage": (
            "p_value_only_extracted_conclusion_not_extracted"
            if "effect.reported_p_value" in numeric
            else "not_extracted_from_selected_support"
        ),
        "equivalence_margin": numeric.get("effect.equivalence_margin"),
        "equivalence_coverage": (
            "margin_only_extracted_conclusion_not_extracted"
            if "effect.equivalence_margin" in numeric
            else "not_extracted_from_selected_support"
        ),
        "moderator_coverage": "not_extracted_from_selected_support",
        "analysis_population": (
            identities.get("finding.analysis_population", [None])[0]
        ),
        "analysis_population_coverage": (
            "grounded_exact_text"
            if identities.get("finding.analysis_population")
            else "not_extracted_from_selected_support"
        ),
        "evidence_receipt_sha256": receipt.evidence_receipt.evidence_receipt_sha256,
        "effect_format_receipt_sha256": (
            receipt.effect_format_receipt.effect_format_receipt_sha256
            if receipt.effect_format_receipt is not None
            else None
        ),
        "numeric_receipt_sha256s": sorted(
            item.numeric_receipt_sha256 for item in receipt.numeric_receipts
        ),
        "identity_receipt_sha256s": sorted(
            item.identity_receipt_sha256 for item in receipt.identity_receipts
        ),
        "authorizes_typed_effect": True,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    try:
        typed_effect = GroundedTypedEffectV2.model_validate(
            {
                **typed_payload,
                "typed_effect_sha256": hash_canonical(typed_payload),
            }
        )
    except ValueError:
        return _abstention(
            candidate=candidate,
            projection=projection,
            protocol=protocol,
            protocol_orientation=protocol_orientation,
            analysis_policy=analysis_policy,
            grounding_receipt_sha256=receipt_sha256,
            blocker_codes={"effect_contract_invalid"},
        )
    payload = {
        "assembly_version": PACKET_ASSEMBLY_V2_VERSION,
        "receipt_version": ASSEMBLY_COMPLETED_V2_VERSION,
        "status": "typed_effect_completed",
        "candidate_descriptor_sha256": candidate.descriptor_sha256,
        "projection_sha256": projection.projection_sha256,
        "protocol_projection_spec_sha256": protocol.projection_spec_sha256,
        "protocol_orientation": protocol_orientation,
        "protocol_orientation_sha256": protocol_orientation.orientation_sha256,
        "analysis_policy": analysis_policy,
        "analysis_policy_sha256": analysis_policy.analysis_policy_sha256,
        "grounding_receipt_sha256": receipt_sha256,
        "typed_effect": typed_effect,
        "typed_effect_sha256": typed_effect.typed_effect_sha256,
        "native_candidate_packet_status": (
            "withheld_optional_scientific_coverage_not_extracted"
        ),
        "native_candidate_packet": None,
        "native_candidate_packet_withholding_codes": [
            "equivalence_conclusion_not_extracted",
            "moderators_not_extracted",
            "reported_significance_not_extracted",
        ],
        "authorizes_typed_effect": True,
        "authorizes_native_candidate_packet": False,
        "synthesis_input_authority": False,
        "claim_release_authority": False,
    }
    return NativePacketAssemblyCompletedV2.model_validate(
        {**payload, "assembly_receipt_sha256": hash_canonical(payload)}
    )


def validate_native_packet_assembly_v2(
    *,
    assembly: NativePacketAssemblyOutcomeV2 | Mapping[str, Any],
    candidate: MetaSynPassageCandidateV2,
    projection: FrozenMetaSynProjectionV2,
    protocol: QuestionProjectionSpecV1,
    protocol_orientation: PacketAssemblyProtocolOrientationV2,
    analysis_policy: PacketAssemblyAnalysisPolicyV2,
    grounding_receipt: PacketGroundingReceiptV2 | Mapping[str, Any],
) -> NativePacketAssemblyOutcomeV2:
    """Replay an assembly artifact; coherent rehashes do not gain authority."""

    try:
        canonical = _ASSEMBLY_ADAPTER.validate_python(assembly)
    except ValueError as exc:
        raise NativePacketAssemblyV2Error(
            "packet_assembly_v2_saved_assembly_invalid"
        ) from exc
    replayed = assemble_native_packet_v2(
        candidate=candidate,
        projection=projection,
        protocol=protocol,
        protocol_orientation=protocol_orientation,
        analysis_policy=analysis_policy,
        grounding_receipt=grounding_receipt,
    )
    if canonical != replayed:
        raise NativePacketAssemblyV2Error(
            "packet_assembly_v2_external_replay_mismatch"
        )
    return canonical


__all__ = [
    "ANALYSIS_POLICY_V2_VERSION",
    "ASSEMBLY_ABSTENTION_V2_VERSION",
    "ASSEMBLY_COMPLETED_V2_VERSION",
    "GROUNDED_TYPED_EFFECT_V2_VERSION",
    "PACKET_ASSEMBLY_V2_VERSION",
    "PROTOCOL_ORIENTATION_V2_VERSION",
    "GroundedBinaryGroupEffectV2",
    "GroundedContinuousGroupEffectV2",
    "GroundedDirectConfidenceIntervalEffectV2",
    "GroundedDirectStandardErrorEffectV2",
    "GroundedDirectVarianceEffectV2",
    "GroundedEffectCoreV2",
    "GroundedTypedEffectV2",
    "NativePacketAssemblyAbstentionV2",
    "NativePacketAssemblyCompletedV2",
    "NativePacketAssemblyOutcomeV2",
    "NativePacketAssemblyV2Error",
    "PacketAssemblyAnalysisPolicyV2",
    "PacketAssemblyProtocolOrientationV2",
    "assemble_native_packet_v2",
    "freeze_packet_assembly_analysis_policy_v2",
    "freeze_packet_assembly_protocol_orientation_v2",
    "replay_metasyn_question_projection_spec_v2",
    "validate_native_packet_assembly_v2",
]
